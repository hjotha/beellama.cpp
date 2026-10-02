# Plano — placement de KV por ocupação (split por posição) CUDA + Radeon

Data: 2026-10-02. Contexto: `docs/xxl-prefill-bottleneck-20261002.md` e
`docs/xxxl-tier-plan-20261002.md`.

## 1. Problema

O cache misto atual divide o KV **por camada** e decide pela **capacidade** do
perfil: no XXXL de 131072, 5 das 16 camadas full-attention ficam inteiras na
Radeon (Q4) desde o token 0. O tier XXXL resolve só a granularidade
(≤ 102400 → XXL 100% CUDA; acima → XXXL misto), mas:

- logo acima de 102400 já se paga a atenção da Radeon sobre **todo** o
  histórico de 5 camadas (5 × 105k = 525k camada-tokens a 105k);
- a troca XXL → XXXL exige handoff de KV (`cache_source=mixed_handoff`,
  ~95.872 tokens reaproveitados, mas com conversão e realocação), e a volta
  XXXL → XXL ainda precisa ser garantida.

## 2. Objetivo

Um único perfil de 131072 em que, para **todas** as 16 camadas:

- posições `[0, P)` ficam na CUDA em KVarN4 (P = 102400, o que já cabia);
- posições `[P, C)` (C = 131072) ficam na Radeon em Q4 (overflow);
- a atenção de cada camada é `merge(FA_cuda([0,P)), FA_vulkan([P,n_kv)))`
  por log-sum-exp;
- enquanto `n_kv ≤ P`, o grafo não tem nenhum nó Vulkan: desempenho igual ao
  XXL só-CUDA.

Comparação de trabalho na Radeon (camada-tokens de KV lidos por ubatch):

| Ocupação | Split por camada (N=5) | Split por posição (16 × overflow) |
|---:|---:|---:|
| 100k | 500k (XXXL) / 0 (XXL) | 0 |
| 105k | 525k | 41k |
| 115k | 575k | 202k |
| 131k | 655k | 459k |

[D] Com ~9 ms/1k por camada em `auto` (medido), a 105k o termo Radeon cai de
~4,7 s para ~0,4 s por ubatch de 256. Em contrapartida, as fronteiras passam de
5 para 16 camadas (~8 ms por camada por ubatch depois do fix do readback →
~0,13 s/ubatch) e o decode passa a ter 16 sincronizações cruzadas por token.

## 3. Arquitetura

### 3.1 Cache de KV particionado por posição

Novo modo do cache misto (`llama_kv_cache_kvarn` + parte padrão Q4), em
`src/llama-kv-cache*.cpp` / `src/llama-kv-mixed-*`:

- Para cada camada full-attention, dois armazenamentos:
  - **local**: KVarN4 na CUDA, capacidade P (alinhada a 256 e ao grupo KVarN
    de 128 tokens);
  - **overflow**: Q4 na Vulkan0, capacidade C − P (alinhada a 256).
- Uma sequência (`parallel = 1`, já exigido pelo adaptativo), sem SWA, sem
  context shift. Células atribuídas por posição: `pos < P` → local,
  `pos ≥ P` → overflow (mapeamento fixo; nada de células livres intercaladas).
- `seq_rm`/rollback/truncate funcionam em cada faixa: remover `[p0, end)` com
  `p0 < P` limpa todo o overflow e o fim da faixa local.
- O tail/stage do KVarN continua só na faixa local; o overflow Q4 não tem
  tail.
- Alocação do overflow: Vulkan/GTT 16 × (C − P) × 1152 B ≈ 16 × 31,5 MiB ≈
  504 MiB (contra 720 MiB das 5 camadas inteiras hoje).
- Orçamento: a parte local é o mesmo do XXL de 102400 com MTP0 (já validado
  como cabendo). O auto-placement deixa de escolher N e passa a escolher P:
  maior P (múltiplo de 256) tal que `16 × local_bytes(P) + reservas` caiba na
  CUDA; overflow = C − P, limitado pelo orçamento Vulkan/RAM UMA.

### 3.2 Escrita de K/V (store)

- `cpy_k`/`cpy_v` (`src/llama-graph.cpp:4021-4047`) recebem índices de
  destino por faixa. Para simplificar, **forçar fronteira de ubatch em P** no
  `llama_decode` (um ubatch nunca atravessa P): cada ubatch escreve só numa
  faixa (store KVarN na CUDA ou `SET_ROWS` Q4 na Vulkan).

### 3.3 Atenção com merge

Para cada camada full-attention, em `build_attn` (`src/llama-graph.cpp:3938`)
e `build_layer_attn` (`src/models/qwen35.cpp:335`):

1. `n_kv ≤ P`: caminho atual só CUDA (KVarN windowed), sem alteração.
2. `n_kv > P`:
   - `o1, lse1 = FA_cuda(Q, K_local[0:P], V_local, mask_local)`; a faixa
     local é toda anterior às queries do ubatch (que estão em `pos ≥ P`),
     então a máscara local só cobre padding;
   - copiar Q (e o K/V novo, para o store) para a Vulkan;
   - `o2, lse2 = FA_vulkan(Q, K_ovf[0:n_kv−P], V_ovf, mask_causal_ovf)`;
   - copiar `o2, lse2` para a CUDA;
   - `o = merge(o1, lse1, o2, lse2)` na CUDA:
     `M = max(lse1, lse2)`,
     `o = (o1·e^{lse1−M} + o2·e^{lse2−M}) / (e^{lse1−M} + e^{lse2−M})`.
   - O gate (`sigmoid · mul`) e `wo` continuam na CUDA, como no fix de
     2026-10-02.

### 3.4 API GGML necessária

- `ggml_flash_attn_ext_set_output_lse(fa)` (nome a definir): marca o FA para
  produzir, além de `dst`, um tensor `lse` F32 `[n_head, n_tokens]`. Avaliar
  reaproveitar a infraestrutura de parciais existente:
  - CUDA: rota windowed do KVarN (`fattn-mma-kvarn-case.cuh`), que já calcula
    parciais por chunk e meta (`partial_ptr`, `acc_meta`, `dst_meta`) e faz
    merge;
  - Vulkan: `fa_split_k_reduce` (`ggml-vulkan.cpp:~13099`), que já combina
    parciais com m/l;
  - `ggml_kv_tail_attention_merge*` (`ggml.h:2610-2631`), que já faz merge
    body + tail dentro de um backend.
- `ggml_attn_merge_lse(o1, lse1, o2, lse2)`: op nova com implementação
  CPU/CUDA/Vulkan e caso em `test-backend-ops`. Alternativa sem op nova:
  compor com `ggml_exp`/`mul`/`add`/`div` (mais nós e mais tráfego).
- Domínio: as duas parciais precisam estar no mesmo domínio (original) antes do
  merge. O LSE é invariante à rotação de Hadamard aplicada igualmente a Q e K,
  mas a saída da Vulkan precisa da rotação inversa de V (já existe no caminho
  Q4 com `attn_rot_v`), e a saída CUDA KVarN usa
  `ROTATED_K_ORIGINAL_V` — conferir.

### 3.5 Scheduler e sobreposição

- Fase inicial: sequencial (CUDA FA local → cópia → Vulkan FA overflow →
  cópia → merge). Mesmo sequencial, o trabalho na Radeon cai pela tabela do §2.
- Fase de otimização: emitir a FA Vulkan **antes** da FA CUDA local da mesma
  camada, para que rodem concorrentemente (as parciais são independentes).
  Exige revisar `ggml_backend_sched_compute_splits` (`ggml-backend.cpp:~1770`:
  sync do backend anterior quando `n_inputs == 0`) para não serializar, e usar
  eventos em vez de `ggml_backend_synchronize`. Ganho máximo:
  `min(t_cuda_local, t_vulkan_ovf)` por camada.

### 3.6 Estado, snapshots, prompt cache e transições

- Novo `layout` de cache (`common_prompt_cache_layout`) para "split por
  posição", serializado como duas faixas por camada.
- Snapshot de XXL/só-CUDA com `n ≤ P` é **diretamente compatível**: só a faixa
  local é preenchida. Isso elimina o handoff XXL → XXXL; o perfil XXL pode ser
  aposentado ou mantido como alias.
- Snapshot com `n > P`: faixa local (KVarN) + overflow (Q4). Restaurar em
  perfil só-CUDA menor exige conversão Q4 → KVarN do overflow
  (`auto_convertible_q4_layout`/conversor existente) ou rejeitar.
- `TMPDIR` em disco continua obrigatório para o spool misto.
- Checkpoints recorrentes (DeltaNet) não mudam.

### 3.7 MTP

Fora do escopo inicial: o perfil roda com `spec-draft-n-max = 0` (como XXL e
XXXL hoje). Reavaliar depois.

## 4. Fases e critérios de aceite

| Fase | Entrega | Aceite |
|---|---|---|
| 0 | Protótipo do merge LSE em `test-backend-ops` (CPU/CUDA/Vulkan) | `FA([0,n)) ≈ merge(FA([0,P)), FA([P,n)))` com erro ≤ tolerância de FA, incluindo P alinhado/desalinhado, máscara causal, GQA 24/4, D=256, Q4 e F16 |
| 1 | Opção LSE no FA (CUDA KVarN windowed e Vulkan Q4) | paridade com a referência CPU em `test-backend-ops` |
| 2 | Cache particionado por posição + store com fronteira de ubatch em P | testes unitários de células, `seq_rm`, rollback e alocação; `test-kv-mixed-*` existentes verdes |
| 3 | Grafo com dois FA + merge; sem nós Vulkan enquanto `n_kv ≤ P` | Qwen3.5-4B com P pequeno (ex.: 512): logits iguais (KL ≈ ruído) ao caminho só-CUDA com o mesmo formato por faixa; `graph splits` = 2 abaixo de P |
| 4 | Estado/snapshots/prompt cache/disco | save/restore abaixo e acima de P, em processo novo; troca de perfil menor → split sem reprocessar |
| 5 | Integração com o servidor adaptativo (perfil único 131072 ou XXXL em modo split) | 27B: frio de ~96k igual ao XXL (~600 tok/s); acima de P, tempo por ubatch crescendo só com `n_kv − P` |
| 6 | Sobreposição CUDA/Vulkan | tempo por camada ≈ `max(t_cuda, t_vulkan)` medido com `GGML_BACKEND_COPY_PROFILE`/timestamps |
| 7 | Promoção | janela de manutenção, A/B contra XXL/XXXL, KL vs referência, decode medido, rollback documentado |

## 5. Riscos

- Decode: 16 fronteiras por token em vez de 5 (latência de sync).
  Mitigação: só ativar o overflow quando `n_kv > P`; medir antes de promover.
- Corretude do merge e da máscara na fronteira P (testes da Fase 0).
- Complexidade do KVarN (grupos de 128, stage/tail, records): P alinhado a 256;
  overflow sem tail.
- Scheduler: uma ordem errada de cópias pode reintroduzir sincronizações
  totais.
- Fence/potência da APU: o governor continua em `auto`; o trabalho na Radeon cai
  perto do limiar, o que reduz o tempo de carga sustentada.

## 6. Esforço estimado

Alto: GGML (op/flag LSE nos backends), cache novo, grafo, scheduler e estado.
O tier XXXL (já implementado) continua como solução de produção enquanto as
fases 0–5 não forem aprovadas.
