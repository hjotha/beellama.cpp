> Historical MTP implementation report. The shared expert cache was removed from the integration (cross-stream race); the rest is in the branch.
>
> Historical agent report (2026-11-21). Current status and numbers: docs/PLAN-strata-parity.md.

# IMPL_REPORT — Fase 2 (Strata parity): MTP nativo do Qwen4Exp na A100-40G

Checkout isolado: `/tmp/opencode/impl-mtp` (cópia de `beellama.cpp` @ `49a560ca3`).
GPU não usada; toda a validação foi CPU-only. Nenhum commit/push.

## 1. Resumo

O bloqueio principal não está no grafo MTP: **o GGUF Q2_0 de produção
(`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, shard 1 + shard 2) foi convertido
sem o bloco NextN/MTP**. O checkpoint original declara `mtp_num_hidden_layers: 1`,
mas o GGUF não tem nem a chave `qwen4exp.nextn_predict_layers` nem qualquer tensor
`blk.48.nextn.*`/`mtp.*`. Com esse arquivo, `--spec-type draft-mtp` não tem o que
executar; antes das mudanças o `llama_init_from_model` recusava o contexto MTP e o
`llama-server` abortava o load com `failed to create MTP context`.

Entregue nesta fase:

1. **Diagnóstico do arquivo** com evidência de metadata/tensores (seções 2.1–2.2).
2. **Load não quebra mais** quando `draft-mtp` é pedido para um arquivo sem MTP:
   o servidor sobe sem especulação, com um aviso único e acionável; a checagem de
   `--fit` também foi ensinada a não sondar um contexto MTP inexistente.
3. **Reserva de VRAM do contexto draft** no auto-size do `--moe-cache-mib -1`
   (KV exato + limite superior dos buffers de compute), com log do plano.
4. **Cache de experts compartilhado** entre o contexto alvo e o draft MTP (mesmo
   modelo/device): o draft não aloca um segundo cache nem reenvia experts por
   passo; o cache do alvo já cobre a camada MTP quando ela está em host.
5. **`graph_mtp`/`kv_only`**: a condição de catch-up foi corrigida (um lote com
   `n_outputs == 0` sem `embeddings_nextn_masked` deixava `inp_out_ids` nulo e
   derrubava o `graph_mtp` do qwen35; seção 2.5, baseline em 4.2), e o qwen4exp ganhou log de debug.
6. **Teste CPU novo** `test-mtp-qwen4exp` cobrindo decode MTP real, `kv_only`,
   rollback de 0–4 drafts e KV q8_0 em `n_ubatch` 32/512/1024.

O critério de performance da A100 **não pode ser atingido com o GGUF publicado**;
o comando e o critério para um arquivo com MTP estão na seção 6.

## 2. Diagnóstico do estado atual

### 2.1 O GGUF de produção não contém o bloco NextN (evidência)

Leitura dos primeiros 16 MiB do shard 1 via HTTP range (só a metadata + índice de
tensores; nenhum download dos 37,6 GB):

```text
$ python3 (parser GGUF do header)
version 3 tensors 1223 keys 73
nextn tensors: 0 mtp tensors: 0
max layer: 47
nextn_predict_layers key present: False
keys with nextn: []
```

O shard 2 tem `split.tensors.count = 1224` e **um** tensor:
`per_layer_token_embd.weight` (160 × 320001536, IQ4_NL, a tabela PLE lazy). Ou
seja, os 1224 tensores do modelo são tronco + PLE; não existe `blk.48.*`.

As tabelas `tensor-allocation/*.rco-allocation.txt` publicadas para Q2_0, IQ3_S e
IQ2_XS também terminam em `blk.47.*` e têm **0** ocorrências de `nextn`.

O checkpoint original, por outro lado, declara MTP:

```text
Qwen/Qwen3.8-Flash-Next config.json:
mtp_num_hidden_layers = 1, num_hidden_layers = 48,
hidden_size = 2560, num_experts = 512, num_experts_per_tok = 10,
moe_intermediate_size = 640
```

O conversor do fork suporta exportar o MTP (`conversion/qwen4exp.py`,
`supports_mtp_export = True`, mapeia `mtp.fc_embedding`/`fc_hidden`/mixer para
`nextn_*`), então o caminho correto é **reconverter o checkpoint com o MTP**, não
mexer no grafo para “inventar” um drafter.

### 2.2 Comportamento antes das mudanças

- `common_model_params_to_llama` liga `load_mtp` quando `draft-mtp` está nos tipos,
  mas `nextn_flags()` só marca os tensores como opcionais; sem a chave
  `nextn_predict_layers` o modelo fica com `n_layer_nextn = 0`.
- `llama_init_from_model(ctx_type = MTP)` já tinha um guard
  (`n_layer_nextn == 0`) que retornava `nullptr` com um `LOG_WARN` genérico.
- O servidor tratava `ctx_dft == nullptr` como erro fatal
  (`failed to create MTP context`), então a configuração de produção **não subia**.
- Com `--fit` (default auto), o fitter ainda tentava medir um “extra model” MTP e
  imprimia um `ERROR` de contexto MTP antes do aviso do servidor.

Não havia assert/segfault nesse caminho, mas também não havia como habilitar o MTP
sem um arquivo com o bloco.

### 2.3 O que funciona quando o modelo TEM o bloco NextN

Verificado por análise de código e pelo teste CPU novo (seção 4):

- `load_arch_tensors` carrega `blk.<n_layer>.nextn.*` (e projeta
  `e_proj`/`h_proj` ou `eh_proj`) quando `load_mtp` está ligado;
  `test-nextn-flags` cobre a distinção “ausente vs. explicitamente pulado”.
- `graph_mtp` do qwen4exp constrói `[enorm(e); hnorm(h)] -> eh_proj -> HC ->
  QSA -> MoE -> HC head -> LM head`, com o KV/QSA do draft próprios.
- O catch-up (`n_outputs == 0`) escreve KV + pools QSA e retorna antes do MoE/head.
- `llama_memory_hybrid_idx` mantém o cache do indexer célula a célula com o KV de
  atenção (assert existente) e o `seq_rm` parcial reconstrói o layout de pools.

### 2.4 Interação com o auto-size do cache de experts

Ordem real: modelo carregado (com ou sem MTP) → contexto alvo (KV + compute +
auto-size do cache) → contexto draft (KV + compute próprios).

Antes:

- `common_base_params_to_speculative` força `moe_cache_size = 0` no draft (comentário
  “the MoE cache is only used by the target context”), então o draft **não** criava
  um segundo cache;
- mas o auto-size do alvo não descontava nada do draft: o draft dependia só da
  margem `--vram-reserve-mib` (default 1536 MiB) para o KV + buffers de compute;
- quando a camada MTP fica em host (`-cmoe` / `-ncmoe` cobrindo `blk.48`), o draft
  não tinha cache: cada passo reenviava os experts selecionados (ou pior, o tensor
  inteiro no primeiro uso), e um cache separado duplicaria VRAM.

Depois: as seções 3.1/3.2 implementam a reserva explícita e o compartilhamento.

### 2.5 `kv_only`: bug real encontrado (qwen35, mesmo padrão do qwen4exp)

`build_inp_out_ids()` passou a devolver `nullptr` quando `n_outputs == 0`
(`bf4f7b081`). O `graph_mtp` do qwen35 chamava `ggml_get_rows(ctx0, cur, inp_out_ids)`
sem checar; um decode multi-ubatch cujo primeiro ubatch não tem saídas
(`--test-mtp-ubatch-sync`, `n_ubatch=2` e 4 tokens com logits só no último) causava
`SIGSEGV` em `ggml_get_rows_as` (`b == nullptr`). O baseline `49a560ca3` reproduz o
crash; o qwen4exp já usava `inp_out_ids ? ... : flat`, por isso não caía.

Além disso, a condição de `kv_only` exigia `cparams.embeddings_nextn_masked`. Isso
não é necessário: se `n_outputs == 0` e embeddings não são pedidos, nenhuma linha é
consumida. A condição foi alinhada nos dois modelos para
`n_outputs == 0 && !embeddings && (!embeddings_nextn || embeddings_nextn_masked)`.

## 3. Mudanças

### 3.1 `src/llama-context.cpp`, `src/llama-context.h` — guard e reserva

- `llama_init_from_model`: a mensagem do guard MTP agora é `ERROR` e diz o que
  fazer (`use --spec-type none` ou um arquivo com o NextN), incluindo arquitetura.
- `estimate_mtp_reserve_bytes(..., per_device)`: devolve também o KV do draft por
  device (usa `layer_bytes` do estimador existente e `model.dev_layer(il)`).
- Auto-size (`MoE cache auto`): quando `params.mtp_reserve_enabled` (spec MTP com
  `n_max > 0`, derivado em `common_context_params_to_llama`), o orçamento de cada
  device passa a ser `free - reserve - draft_kv(device) - compute_buffer(device)`.
  O limite superior de compute do draft é o buffer de compute do próprio alvo
  naquele device (o grafo MTP é uma camada de decoder, então não é maior).
  Loga `MoE cache auto: reserving N MiB of draft-MTP KV (...)`.
- Cache compartilhado: em contexto `ctx_type == MTP` com `params.ctx_other` do
  **mesmo objeto de modelo**, não `no_alloc`, e com `moe_cache` no alvo, o draft
  adota `params.ctx_other->moe_cache` (`moe_cache_shared`), zera
  `cparams.moe_cache_size` e loga `MTP draft context shares the target MoE cache`.
  `moe_cache_get()` é usado no `graph_params` e nos dois callbacks de cópia.
- `memory_breakdown` continua contando o cache só no dono (sem dupla contagem).

### 3.2 `src/llama-moe-cache.cpp` — backend por device

O cache compartilhado é executado pelo scheduler do draft, que tem outro objeto de
backend no mesmo device. As duas comparações `backend == devices[g].backend`
passaram a comparar `ggml_backend_get_device(...)`; as cópias continuam sendo
enfileiradas no backend do chamador (mesma stream que lê o resultado).

### 3.3 `src/models/qwen4exp.cpp` / `qwen35.cpp` — `kv_only`

- Condição nova (seção 2.5) nos dois `graph_mtp`.
- `qwen4exp` loga em `DEBUG` quando o caminho KV-only é usado
  (`writing KV and QSA pools only (n_outputs == 0)`), o que torna o teste e o
  profiling observáveis.
- `--spec-draft-n-max 2/4` não precisou de mudança no driver: os rollbacks de
  1..5 posições (0–4 aceitos + âncora) já eram cobertos por `seq_rm` + replay, e o
  teste novo exercita exatamente esses tamanhos com KV q8_0.

### 3.4 `tools/server/server-context.cpp` — load não quebra

- Se `draft-mtp` foi pedido, não há draft model e
  `llama_model_n_layer_nextn(model_tgt) == 0`, remove `DRAFT_MTP` dos tipos,
  loga um `WRN` único e segue sem especulação. O estágio de progresso “spec_model”
  é concluído para a UI não travar.
- `rebuild_active_speculation` (perfis adaptativos) só tenta MTP se o modelo tem
  NextN, evitando marcar o contexto adaptativo como indisponível.

### 3.5 `common/common.cpp`, `common/common.h` — `--fit` sem sonda fantasma

- `common_gguf_info` ganhou `n_layer_nextn`, lido da metadata
  (`<arch>.nextn_predict_layers`).
- `common_fit_apply` só passa o “extra model” MTP ao fitter quando o arquivo tem
  NextN (arquivos remotos/streamados sem metadata legível mantêm o comportamento
  antigo).
- `common_probe_context_overhead` só sonda o contexto MTP se o modelo carregado
  tem NextN.

### 3.6 `tests/test-llama-archs.cpp`, `tests/CMakeLists.txt` — teste novo

`--test-mtp-qwen4exp` / ctest `test-mtp-qwen4exp`:

- fixture GGUF sintético `qwen4exp` com 2 camadas de tronco (linear + QSA) + 1
  camada MTP QSA (`block_count=3`, `compress_ratios=[4,4,4]`, `load_mtp=true`);
- contexto alvo com KV **q8_0** e `llama_set_embeddings_nextn`;
- contexto draft `ctx_type=MTP`, `ctx_other` apontando para o alvo;
- decode do prompt → `h_nextn` do alvo (todas as linhas finitas);
- catch-up KV-only de 40 posições → `seq_pos_max == 39` e 16 ocorrências do
  caminho KV-only no log (3 tamanhos de ubatch);
- rollback + replay para sufixos de 1, 2, 3 e 5 posições (0–4 drafts aceitos +
  âncora), conferindo `seq_pos_max` após cada corte e após o replay;
- passo de draft completo (atenção + MoE + head): logits e `h_nextn` finitos;
- um modelo sem NextN deve recusar `ctx_type=MTP` antes de qualquer decode;
- `n_ubatch` 32, 512 e 1024.

## 4. Testes executados (CPU-only)

Build (regra do pedido):

```bash
cmake -B build-cpu -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGGML_CUDA=OFF \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build build-cpu -j4 --target test-llama-archs test-recurrent-state-rollback \
  test-speculative-rejection-sampling test-arg-parser test-nextn-flags \
  test-recurrent-zero test-backend-ops llama-server
```

O `cmake --build build-cpu -j4` **sem target** falha em
`tools/rkva-native-key-inspect` (erro pré-existente do HEAD, arquivo não tocado):
`llama_seq_id has not been declared` e `too few arguments to positions_bytes`.
Os targets acima compilam com `rc=0`.

### 4.1 Teste novo e MTP existentes

```text
$ ./build-cpu/bin/test-llama-archs --test-mtp-qwen4exp -v 4
... graph_mtp: writing KV and QSA pools only (n_outputs == 0)   [16x]
... test_mtp_qwen4exp: qwen4exp MTP decode, KV-only catch-up and rollback: OK
rc=0

$ ctest --test-dir build-cpu -R 'test-mtp-qwen4exp|test-mtp-ubatch-sync|test-mtp-request-reset|test-nextn-flags|test-recurrent-zero|test-speculative-rejection-sampling'
100% tests passed out of 7
```

### 4.2 Baseline `49a560ca3` (mesmo build, worktree separada)

Evidência para separar regressões introduzidas de falhas pré-existentes:

| teste | HEAD 49a560ca3 | esta branch |
|---|---|---|
| `--test-mtp-ubatch-sync` | SIGSEGV (`ggml_get_rows_as`, `b == nullptr`) | **rc=0** |
| `--test-mtp-qwen4exp` | não existe | **rc=0** |
| `--test-mtp-kvarn-routing` | SIGSEGV | SIGSEGV (inalterado) |
| `test-arg-parser` | aborta em `mparams_cmoe.keep_host_pinned` | aborta igual |
| `test-generate-models` | falha em `sensenova_loop.attention.recurrent_layers` | falha igual |

### 4.3 `ctest -R 'spec|recurrent|llama-archs|arg-parser'`

```text
103 test-generate-models ................ Failed (sensenova_loop fixture, pré-existente)
104 test-generate-speculative-target .... Passed
105 test-speculative-rejection-sampling . Passed
107-111 test-recurrent-state-rollback ... Not Run (dependem do fixture 103)
135 test-arg-parser ..................... Subprocess aborted (sem GPU)
226 test-recurrent-zero ................. Passed
275 test-rkva-native-object-inspect .... Passed
```

Motivos, ambos confirmados no baseline:

- **test-arg-parser**: a asserção `mparams_cmoe.keep_host_pinned` exige um device
  com host buffer paginado (`common.cpp` só liga `keep_host_pinned` quando
  `ggml_backend_dev_host_buffer_type(host_buft)` existe). O build CPU-only lista
  `Available devices: (none)` (`llama-server --list-devices`), então a asserção
  não pode valer. Não é regressão desta branch.
- **test-generate-models**: o fixture sintético não escreve
  `sensenova_loop.attention.recurrent_layers`, chave que o loader exige; o
  `test-llama-archs -o` morre nesse arquétipo antes de gerar os modelos dos testes
  de rollback. Também ocorre no baseline.

### 4.4 Smoke do servidor com o GGUF sem MTP

Modelo sintético `qwen4exp-moe.gguf` (19 MB), CPU:

```bash
./build-cpu/bin/llama-server -m qwen4exp-moe.gguf -c 128 -b 32 -ub 32 -np 1 \
  --no-webui --port 8199 --spec-type draft-mtp --spec-draft-n-max 2
```

```text
W srv load_model: --spec-type draft-mtp requested, but
  '/tmp/opencode/q4e-model/qwen4exp-moe.gguf' declares no NextN/MTP block;
  serving without speculative decoding (the GGUF was converted without the MTP tensors)
I srv server_on_li: listening on http://127.0.0.1:8199
$ curl -s -o /dev/null -w '%{http_code}' .../health
200
```

Mesmo resultado com `-fit on` e `--spec-draft-n-max 4`. Antes, o load terminava em
`failed to create MTP context`.

## 5. Riscos e limites

1. **O GGUF publicado é o bloqueio de performance**: sem `blk.48.nextn.*` não existe MTP
   para acelerar. As metas de 65/55/54 t/s não são alcançáveis com esse arquivo;
   o servidor agora sobe no modo sem especulação e mantém o baseline medido
   (54,6/46,4/45,1 t/s).
2. **Cache compartilhado só foi validado por inspeção**: `llama_moe_cache` exige
   backend GPU (`MoE cache requires a GPU backend`), então o CI CPU não cobre a
   adoção. O ponto sensível é a comparação de backend relaxada para “mesmo device”;
   as cópias usam o backend do chamador. Precisa de um smoke CUDA (target + draft,
   `-cmoe` ou `-ncmoe 49`) antes de produção.
3. **Orçamento do auto-size é conservador**: o compute do draft é limitado pelo
   buffer de compute do alvo, que pode ser maior que o necessário e encolher o
   cache. O log `MoE cache auto: reserving ...` mostra o valor; se a A100 perder
   cache demais, o próximo passo é medir o buffer real do contexto draft e usar um
   limite menor.
4. **`--moe-cache-mib N` explícito não é limitado**: a reserva do draft só entra
   no modo `-1` (auto). Com tamanho fixo, o operador é responsável por deixar
   espaço para o contexto draft (o log de criação do draft mostra o KV e o compute
   reais).
5. **Custo intrínseco dos pesos MTP**: com `-ncmoe 2` só `blk.0..1` ficam em CPU;
   `blk.48` (MTP) fica na GPU (~750 MiB a Q2_0: 3 × 640×2560×512 × 0,3125 B).
   Isso reduz o orçamento do cache antes mesmo do contexto draft. Para preservar o
   cache, mover só os experts do bloco MTP para host
   (`-ot 'blk\.48\.ffn_(up|down|gate)_exps=CPU'`); aí o cache compartilhado passa a
   cobrir a camada MTP.
6. **`kv_only` ampliado**: agora qualquer lote `n_outputs == 0` sem embeddings
   pedidos pula o FFN/head. Nenhuma linha é consumida nesse caso; o único efeito é
   a forma do grafo. Coberto pelo teste novo e pelo `--test-mtp-ubatch-sync`.
7. **Falhas pré-existentes não corrigidas** (fora do escopo): crash
   `--test-mtp-kvarn-routing` em `llama_memory_hybrid_idx::init_batch`
   (`get_mem_attn()` nulo no sidecar Qwen4Exp com KVarN), fixture
   `sensenova_loop` e asserção GPU-only do `test-arg-parser`.

## 6. Comando e critério de aceite na A100-40G

### 6.1 Pré-requisito obrigatório

Um GGUF que contenha o bloco NextN. Checar antes:

```bash
python3 - <<'EOF'
import struct
f=open('Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf','rb')
f.read(4); struct.unpack('<I',f.read(4)); nt=struct.unpack('<Q',f.read(8))[0]; nkv=struct.unpack('<Q',f.read(8))[0]
def rs(f):
    n=struct.unpack('<Q',f.read(8))[0]; return f.read(n).decode('utf-8','replace')
SZ={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
def rv(f,t):
    if t==8: return rs(f)
    if t==9:
        at=struct.unpack('<I',f.read(4))[0]; n=struct.unpack('<Q',f.read(8))[0]
        return [rv(f,at) for _ in range(n)]
    fmt=SZ[t]; return struct.unpack('<'+fmt,f.read(struct.calcsize(fmt)))[0]
keys=[]
for _ in range(nkv):
    k=rs(f); t=struct.unpack('<I',f.read(4))[0]; rv(f,t); keys.append(k)
names=[]
for _ in range(nt):
    nm=rs(f); nd=struct.unpack('<I',f.read(4))[0]
    struct.unpack('<'+'Q'*nd,f.read(8*nd))
    struct.unpack('<IQ',f.read(12))
    names.append(nm)
print('nextn key:', [k for k in keys if k.endswith('nextn_predict_layers')])
print('nextn tensors:', sum('nextn' in n for n in names))
EOF
# esperado num arquivo com MTP:
# nextn key: ['qwen4exp.nextn_predict_layers']   nextn tensors: >= 6
```

Se o arquivo for o Q2_0 publicado, o resultado é
`nextn key: []` / `nextn tensors: 0` e o MTP não pode ser ligado. Nesse caso o
aceite desta fase é o da seção 4.4 (sobe, avisa, health 200, decode no baseline).

### 6.2 Comando de medição (config de produção + MTP)

```bash
./build-a100/bin/llama-server \
  -m /models/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  -c 32768 -b 2048 -ub 512 -fa on \
  --load-mode mmap --lazy-mode on \
  -ctk q8_0 -ctv q8_0 \
  -ncmoe 2 --moe-cache-mib -1 --vram-reserve-mib 1536 \
  --spec-type draft-mtp --spec-draft-n-max 4 \
  --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 \
  -np 1 --host 127.0.0.1 --port 8190 --metrics
```

Repetir com `--spec-draft-n-max 2` e com `--spec-type none` (controle).
Logs esperados no start:

```text
MoE cache auto: reserving <N> MiB of draft-MTP KV (ctx 32768, q8_0/q8_0)
MoE cache auto: <M> MiB for <H> MiB of host experts (...)
MTP draft context shares the target MoE cache (<S> MiB), no extra cache VRAM
```

### 6.3 Critério de aceite

- **Throughput**: +20% de decode vs. o controle sem especulação em **≥2 dos 3
  prompts** (curto, pós-4,6k, pós-28,6k), com piso absoluto de **≥65 / ≥55 / ≥54
  t/s** respectivamente (baseline 54,6 / 46,4 / 45,1; Strata 85,9 / 67,4 / 75,0).
- **VRAM**: pico ≤ **39,4 GiB** (`nvidia-smi --query-gpu=memory.used
  --format=csv,noheader,nounits`, amostrado durante decode e prefill).
- **Corretude**: 0–4 drafts aceitos sem erro; conferir em `/metrics`
  `llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0".."3"}` e
  `llamacpp:spec_decode_num_drafts_total` > 0, com `/v1/chat/completions` greedy
  determinístico e respostas idênticas ao controle (mesmo seed/temperatura).
- **Estabilidade**: 3 execuções alternadas por prompt, sem `GGML_ASSERT`, sem
  OOM, `/health` 200 após cada rodada.

Se o cache auto encolher além do aceitável por causa do compute reservado, o
próximo ajuste é medir o buffer real do draft (`sched_reserve` do contexto MTP) e
substituir o limite superior pelo valor medido.

## 7. Próximos passos (priorizados)

1. **Reconverter o Qwen3.8-Flash-Next com MTP** (ou obter um GGUF que preserve
   `blk.48.nextn.*`) e rodar 6.2/6.3. Sem isso não há ganho de decode possível.
2. Smoke CUDA do cache compartilhado (target + draft, `-cmoe` e
   `-ot blk\.48...=CPU`), conferindo hit rate e ausência de dupla alocação.
3. Corrigir o crash pré-existente do sidecar Qwen4Exp + KVarN
   (`llama_memory_hybrid_idx::init_batch` com `mem_attn` nulo) — impacta
   `--test-mtp-kvarn-routing`, não o caminho de produção desta fase.
4. Consertar o fixture `sensenova_loop` e a asserção GPU-only do `test-arg-parser`
   para o `ctest -R 'spec|recurrent|llama-archs|arg-parser'` ficar verde em
   CPU-only.

## 8. `git status --short`

```text
 M common/common.cpp
 M common/common.h
 M src/llama-context.cpp
 M src/llama-context.h
 M src/llama-moe-cache.cpp
 M src/models/qwen35.cpp
 M src/models/qwen4exp.cpp
 M tests/CMakeLists.txt
 M tests/test-llama-archs.cpp
 M tools/server/server-context.cpp
?? IMPL_REPORT.md
```
