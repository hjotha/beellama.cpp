# Plano — placement de KV por ocupação (split por posição) CUDA + Radeon

Data: 2026-10-02. Contexto: `docs/xxl-prefill-bottleneck-20261002.md` e
`docs/xxxl-tier-plan-20261002.md`.
Revisão: 2026-10-03, contra o código `982eaadaa`.

Status: proposta de implementação. As metas de aceite abaixo são requisitos
iniciais, não resultados medidos. Os números históricos do diagnóstico devem
ser reproduzidos no mesmo modelo, configuração e hardware antes do A/B.

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

- posições `[0, P)` ficam na CUDA em KVarN4 (P alvo = 102400, sujeito ao
  orçamento completo);
- posições `[P, C)` (C = 131072) ficam na Radeon em Q4 (overflow);
- a atenção de cada camada é `merge(FA_cuda([0,P)), FA_vulkan([P,n_kv)))`
  por log-sum-exp;
- enquanto `n_kv ≤ P`, o grafo executado não tem nenhum nó Vulkan; desempenho
  comparável ao XXL só-CUDA dentro do limite de regressão do §4.1.

Comparação de trabalho na Radeon (camada-tokens de KV lidos por ubatch):

| Ocupação | Split por camada (N=5) | Split por posição (16 × overflow) |
|---:|---:|---:|
| 100k | 500k (XXXL) / 0 (XXL) | 0 |
| 105k | 525k | 41k |
| 115k | 575k | 202k |
| 131k | 655k | 459k |

[D] Usando os ~9 ms/1k por camada em `auto` do diagnóstico histórico, estima-se
que a 105k o termo Radeon caia de ~4,7 s para ~0,4 s por ubatch de 256. Em contrapartida, as fronteiras passam de
5 para 16 camadas (~8 ms por camada por ubatch depois do fix do readback →
~0,13 s/ubatch) e o decode passa a ter 16 sincronizações cruzadas por token.
Esses valores estimam o termo Radeon, não a latência total: CUDA local,
transferências, stores, merge e sincronizações entram na medição ponta a ponta.
A tabela usa P = 102400; recalcular se o orçamento exigir outro P.

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
- Uma tabela lógica única de posições/células governa as duas faixas e os
  checkpoints recorrentes. `seq_rm`/rollback/truncate validam a operação inteira
  antes de alterar qualquer faixa: remover `[p0, end)` com `p0 < P` remove
  overflow e sufixo local somente se o rollback KVarN for representável.
  Remoções desalinhadas fora da reserva exata exigem restauração de checkpoint
  e replay pelo caminho existente, ou rejeição sem mutação; não presumir que
  truncar records comprimidos seja equivalente a truncar Q4.
- O tail/stage do KVarN continua só na faixa local; o overflow Q4 não tem
  tail.
- Alocação do overflow: Vulkan/GTT 16 × (C − P) × 1152 B ≈ 16 × 31,5 MiB ≈
  504 MiB (contra 720 MiB das 5 camadas inteiras hoje).
- Orçamento: 102400 é uma meta; o XXL caber isoladamente não prova que o grafo
  misto caiba. Escolher P na inicialização e mantê-lo fixo por contexto, múltiplo
  de 256, após calcular o pico completo descrito no §3.8. Persistir P e C no
  snapshot; não mover a fronteira silenciosamente durante a execução.

### 3.2 Escrita de K/V (store) e fronteira de ubatch

- `cpy_k`/`cpy_v` em `src/llama-graph.cpp` recebem índices de destino por faixa.
  O allocator deve produzir ubatches que não atravessem P, antes de preparar
  memória recorrente ou atenção; cada ubatch usa store KVarN CUDA ou Q4 Vulkan.
- Integrar a fronteira em `llama_memory_hybrid::init_batch` e no allocator.
  O código atual mantém os últimos `1 + n_rs_seq` tokens juntos para preservar
  rollback recorrente. Um corte posterior em `llama_decode` violaria essa regra.
- Se a janela protegida atravessar P, a versão inicial rejeita a preparação
  antes de qualquer mutação, com motivo explícito; não divide a janela nem
  desativa rollback silenciosamente. O servidor deve usar o perfil existente
  compatível ou informar a incompatibilidade. Suportar esse caso no split exige
  depois store em duas faixas no mesmo ubatch e novos testes de checkpoint.
- Mesmo com MTP0, provar os valores e invariantes recorrentes efetivos: testar
  `embd_all`, prefill, decode e retomada. Falha de prepare em uma faixa não pode
  publicar estado parcial na outra nem avançar o estado DeltaNet.

### 3.3 Atenção com merge

Para cada camada full-attention, em `build_attn` (`src/llama-graph.cpp:3938`)
e `build_layer_attn` (`src/models/qwen35.cpp:335`):

1. `n_kv ≤ P`: caminho atual só CUDA, com dispatch próprio de prefill e decode.
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

### 3.4 Contrato de atenção e LSE

- Reaproveitar os metadados/parciais existentes antes de criar API ou op nova.
  Uma opção de saída LSE deve expor F32 por query/head (e dimensão de sequência
  explícita; inicialmente `parallel = 1`), com strides, lifetime, alocação e
  transferência conhecidos pelo scheduler. A saída normal do FA permanece O.
- Contrato: `lse = log(sum(exp(score)))` em logaritmo natural, após scale,
  máscara, bias e softcap. Mesmas posições absolutas/RoPE, GQA e convenções
  nas duas faixas. Máscara causal do overflow usa offset P, nunca reinicia RoPE.
  Sinks ficam fora do primeiro modo; rejeitar configuração incompatível.
- LSE deve normalizar a saída completa da faixa, incluindo body + tail KVarN.
  Não exportar apenas o normalizador do body ou de um chunk. Se os kernels
  guardarem `(m, l)` ou log2, converter para o contrato comum explicitamente.
- Matriz obrigatória de rotas, com teste que confirme qual rota executou:

| Backend/rota | Exigência |
|---|---|
| CUDA KVarN prefill windowed | Reusar `partial_ptr`, `acc_meta`, `dst_meta`; exportar LSE após todos os chunks e tail |
| CUDA KVarN decode de 1 query | Cobrir fast-decode em `fattn-kvarn-dispatch.cu` e fallback MMA selecionado pelo dispatch; windowed recusa `Q->ne[1] <= 1` |
| CUDA KVarN prefill sem windowed/fallback habilitado | Mesmo contrato O/LSE, ou rejeição explícita do modo antes de servir |
| Vulkan Q4 com `split_k > 1` | Exportar após `fa_split_k_reduce`, incluindo todas as parciais |
| Vulkan Q4 com `split_k = 1` | Exportar no kernel final; o redutor de split-K não executa nesse caso |
| Body + tail / tail sem body | Reusar `ggml_kv_tail_attention_merge*`, com O e LSE finais consistentes |

- Faixa vazia ou query inteiramente mascarada: O = 0, LSE = -infinito.
  Se só uma faixa é válida, retornar sua saída; se ambas são vazias, retornar
  zero. Tratar esses casos antes de subtrair infinitos para evitar NaN.
- Merge pela fórmula do §3.3, primeiro como protótipo CPU e composição GGML
  com operações existentes; criar op dedicada só se medição demonstrar custo
  relevante. No produto, o merge ocorre na CUDA; não exigir kernel Vulkan de
  merge sem necessidade demonstrada.
- As saídas devem chegar ao merge no domínio V original. Conferir a inversa
  `attn_rot_v` da Vulkan e `ROTATED_K_ORIGINAL_V` da CUDA. Rotação ortogonal de
  Q/K preserva o produto em aritmética exata; quantização pode alterar scores.
  Testes de merge usam a mesma representação de KV por faixa na referência;
  a comparação de qualidade entre formatos é um gate separado (§4.1).

### 3.5 Scheduler e sobreposição

- Fase inicial: sequencial (CUDA FA local → cópia → Vulkan FA overflow →
  cópia → merge). Mesmo sequencial, o trabalho na Radeon cai pela tabela do §2.
- Fase opcional, somente após aprovação sequencial e ganho medido: emitir a FA
  Vulkan **antes** da FA CUDA local da mesma camada, para que rodem concorrentemente (as parciais são independentes).
  Exige revisar `ggml_backend_sched_compute_splits` (`ggml-backend.cpp:~1770`:
  sync do backend anterior quando `n_inputs == 0`) para não serializar, e usar
  eventos em vez de `ggml_backend_synchronize`. Ganho máximo:
  `min(t_cuda_local, t_vulkan_ovf)` por camada.

### 3.6 Estado, snapshots, prompt cache e transições

Desenhar o formato antes do cache. O envelope misto atual tem versão 1 e
`mixed_state_layers` rejeita `layer_id` repetido com
`mixed state has duplicate layer ownership`. Duas faixas por camada exigem
nova versão, não apenas preencher dois descritores.

- Propor envelope misto v2 (confirmar número livre na implementação), com layout
  `position_split`, P, C, ocupação lógica, identidade do modelo/RoPE e formatos.
  Cada descritor identifica `(layer_id, range_id)`, início inclusivo/fim exclusivo,
  backend/formato, dimensões, strides, comprimento e integridade do payload.
  A tabela de posições/células é única e compartilhada entre as faixas.
- Validar exatamente as faixas local `[0,P)` e overflow `[P,C)` por camada,
  sem lacunas/sobreposições, duplicatas de faixa, índices fora do limite ou
  payload truncado. Uma faixa vazia pode omitir bytes, mas não sua descrição.
  Revisar o envelope e a versão externa KVarN se o framing também mudar.
- Atualizar writer, reader, validadores, spool, handoff, identidade de layout,
  predicados de reutilização/conversão e índice de disco. Leitor antigo deve
  rejeitar v2; leitor novo preserva v1 por camada pelo caminho legado, sem
  reinterpretá-lo como split. Versão desconhecida é rejeitada antes de mutar.
- XXL KVarN puro com `n <= P` exige **importação explícita**, validando modelo,
  tipos, domínio, posições, tail e estado recorrente. Usar preparação/restauração
  em etapas e commit existentes; preencher local e deixar overflow vazio.
  A seleção no índice de disco deve reconhecer a conversão autorizada: os
  predicados atuais de layout não dão compatibilidade direta automaticamente.
- Teste obrigatório em processo novo, com **somente o snapshot XXL no disco**,
  sem cache residente: provar hit/importação, tokens efetivamente reutilizados
  e ausência de prefill do prefixo importado. Incluir falha de importação com
  estado destino intacto e fallback explícito de cache miss/reprocessamento.
- Split com o mesmo P/C e layout: restore exato, abaixo e acima de P.
  P diferente: versão inicial rejeita reutilização direta e faz cache miss;
  reparticionar/requantizar é conversão futura explícita, com orçamento próprio.
  Snapshot com ocupação maior que a capacidade destino também é rejeitado.
- Split → só-CUDA: importação da faixa local só quando `n <= min(P, capacidade)`
  e o restante do contrato coincide. Com overflow ocupado, a versão inicial
  rejeita; conversor Q4 → KVarN existente não prova compatibilidade do envelope.
- Restaurar KV, tabela lógica e checkpoint DeltaNet como uma transação. Testar
  sufixo divergente e rollback antes/em/depois de P com o estado recorrente;
  manter o algoritmo DeltaNet não dispensa validar a integração.
- `TMPDIR` em disco continua obrigatório para o spool misto. Preservar o
  snapshot original em qualquer falha e publicar arquivos novos atomicamente.

### 3.7 MTP

Fora do escopo inicial: o perfil roda com `spec-draft-n-max = 0` (como XXL e
XXXL hoje). Reavaliar depois.

### 3.8 Orçamento de memória e escolha de P

Antes de fixar P, calcular e medir simultaneamente: pesos, KV local comprimido,
metadados, stage/tail e reserva de rollback, estado recorrente, grafos reservados
para pior prefill **e** geração, O/LSE parciais, Q/K/V transferidos, buffers de
cópia/staging, scratch/allocator e eventual coexistência durante restore.
Não usar apenas a memória depois da inicialização como evidência de pico.

- Escolher o maior P múltiplo de 256 até 102400 que respeite todos os limites.
  Margem CUDA mínima: `max(512 MiB, 5% da VRAM total)` livre no pico observado.
  Registrar limite calculado, pico medido e folga, sem dupla contagem de buffers
  reutilizados. Se nenhum P suportado couber, rejeitar a configuração.
- Calcular KV overflow, scratch Vulkan, buffers host e spool separadamente.
  Exigir pelo menos 2 GiB de `MemAvailable`, sem OOM nem atividade de swap
  durante a carga; registrar também GTT/heap Vulkan disponível e seu pico.
- Fixar P por contexto e registrá-lo em logs/snapshot. Mudança de disponibilidade
  não autoriza migração implícita: novo contexto recalcula P e aplica §3.6.
- Incluir alocação/reserva e execução reais de prefill, decode, restore e passagem
  por P. A reserva pode conter buffers Vulkan mesmo abaixo de P; a execução
  abaixo de P deve continuar sem kernels nem cópias Vulkan de atenção.

## 4. Fases e critérios de aceite

Todas as fases começam pela versão sequencial. Os gates do §4.1 são cumulativos;
falha bloqueia promoção, sem afrouxar tolerância depois de observar resultados.

| Fase | Entrega | Aceite |
|---|---|---|
| 0 | Contratos O/LSE, envelope v2, rollback/ubatch e orçamento; protótipo do merge | Revisão dos §§3.2–3.8; referência CPU e casos vazios/mascarados aprovados pelo §4.1 |
| 1 | LSE final nas rotas CUDA/Vulkan | Todas as rotas do §3.4 exercitadas, incluindo decode, sem split-K e tail; paridade numérica |
| 2 | Cache por posição, store e allocator | Tabela única; rollback e falhas atômicas; fronteira integrada antes de prepare; testes mistos existentes verdes |
| 3 | Dois FA + merge sequencial | 4B com P=512 e 1024; prefill/decode cruzando P; sem execução Vulkan abaixo de P; gates estruturais |
| 4 | Estado, snapshots e descoberta no disco | Matriz do §3.6 em processo novo, incluindo XXL isolado no disco, P diferente e snapshot inválido |
| 5 | Integração adaptativa e orçamento 27B | P real registrado; gates de latência/qualidade/memória e retomada em todos os comprimentos |
| 6 (opcional) | Sobreposição CUDA/Vulkan | Só após fases 0–5 verdes; benefício ponta a ponta e ausência de regressão comprovados pelo §4.1 |
| 7 | Promoção | Gates sequenciais verdes, janela de manutenção, canário e rollback para XXL/XXXL documentado |

### 4.1 Protocolo e limites objetivos

**Reprodutibilidade.** Registrar commit, modelos e hashes, comando/configuração,
P/C efetivos, tipos por faixa, seed, corpus/prompt e hash, sampling, `-b`/`-ub`,
rotas executadas, driver/build, clocks/potência/governor, cache frio/quente e
concorrência. Baseline e candidato usam o mesmo cenário; repetir A/B alternado
5 vezes após 1 aquecimento descartado. Usar medianas por execução e p95 das
latências de tokens, sem misturar frio/quente ou prefill/decode.

**Corretude estrutural.** Referência CPU de atenção sobre o KV dequantizado da
mesma representação por faixa, com acumulação estável; isolar erro de merge do
erro causado pela troca KVarN/Q4.
Definir `NRMSE = RMS(O - ref) / max(RMS(ref), 1e-6)`. Exigir NRMSE <= 1e-3, erro absoluto máximo de O <= 1e-2 e erro absoluto
de LSE finito <= 1e-2. Linhas vazias exigem O=0/LSE=-infinito exatos; nenhuma
saída válida pode conter NaN/Inf. Logits split versus referência com o mesmo
KV por faixa: KL média <= 1e-4 nat/token e p99 <= 1e-3 em logits teacher-forced,
com KL calculada em FP64. Referência só-CUDA com formato diferente não substitui
essa referência estrutural.

Cobrir Q=1 e prefill Q=2/256, GQA 24/4, D=256, F16 e Q4, body/tail, rota rápida
ou fallback, máscaras causal/total, uma/ambas faixas vazias. No teste matemático,
P alinhado e desalinhado; no cache real, P múltiplo de 256. Exercitar ocupações
e remoções P-129, P-128, P-1, P, P+1, P+128; gerar de P-1 até P+128. Testar
rollback dentro/fora da reserva exata, janela recorrente cruzando P, falha de
prepare, snapshot corrompido e retomada com prefixo divergente. Comparar estado
DeltaNet e próximos logits com execução de referência, não apenas texto gerado.

**Qualidade entre formatos.** Em corpus fixo e prefixos idênticos, comparar
baseline XXXL por camada e candidato contra a mesma referência F16. Aceitar
KL média do candidato <= KL média da baseline + 0,001 nat/token e perplexidade
<= 1,01 × baseline; abaixo de P, usar XXL como baseline. Registrar resultados
por comprimento. Esses limites são metas iniciais, ainda não medidas.

**Desempenho.** Abaixo de P (incluindo ~96k quando P permitir): regressão máxima
5% em tempo de prefill/TTFT e p95 de latência por token contra XXL. Acima de P,
comparar com XXXL por camada em 105000, 115000 e 130560 tokens: tempo total de
prefill/TTFT pelo menos 10% menor em cada ponto e regressão máxima 5% no p95 de
decode. Se P mudar, incluir também P+256 e registrar que a comparação mudou.
Separar tempo de FA local, overflow, cópias, store e merge; menor trabalho Radeon
isolado não satisfaz o gate. O número de graph splits é diagnóstico, não um
valor fixo de aprovação; provar ausência de execução Vulkan de atenção abaixo
de P. Sobreposição só entra se reduzir a mediana ponta a ponta pelo menos 5%
contra sequencial em dois comprimentos acima de P, sem piorar demais gates.

**Memória e estabilidade.** Cumprir §3.8 durante cada cenário. Testar 131072
posições ocupadas, rejeição limpa ao exceder C e geração 130560 + 512 até C.
Executar 10 ciclos consecutivos de restore, sufixo divergente e geração junto
ao limite, sem OOM, erro de driver ou corrupção; memória retida após o ciclo
10 não pode superar a do primeiro ciclo aquecido em mais de 64 MiB por backend.

**Testes existentes a estender.** `test-backend-ops`, `test-kvarn`,
`test-kv-mixed-state`, `test-kv-mixed-state-stream`, `test-kv-mixed-io`,
`test-kv-mixed-placement`, `test-kv-mixed-handoff` e
`test-state-convert-q4-kvarn`; acrescentar os casos de allocator híbrido e
integração de disco no harness correspondente. Registrar comando e saída reais;
esta revisão documental não executa nem declara esses testes aprovados.

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
Os tiers XXL/XXXL existentes permanecem como solução de produção até os gates
sequenciais e a promoção da fase 7 serem aprovados. A fase 6 é opcional e não
bloqueia uma versão sequencial que cumpra todos os limites. Esta revisão altera
somente o plano; não autoriza ativação nem declara capacidade/ganho comprovados.
