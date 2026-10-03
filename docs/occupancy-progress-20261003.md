# Occupancy position-split — log de implementação (branch `feat/occupancy-position-split`)

Data: 2026-10-03. Máquina: gokaya (192.168.1.57 / .52). Plano:
`docs/occupancy-placement-plan-20261002.md` (revisão 2026-10-03 preservada de
`docs/occupancy-plan-review-20261003`, sha256 `cc5cb1b1...`).

## Baseline de produção — ATENÇÃO: binário original NÃO é recuperável
> O binário e as bibliotecas originais foram **sobrescritos** pelos builds em
> `build-optimized`. "Recuperável" aqui significa apenas os *artefatos de
> observação* (linha de processo, INI, ps), não o executável. Reconstruir um
> commit conhecido **não** equivale ao binário original `3e104c3fd-dirty`, e o
> serviço (`llama-server-root`) continua apontando para `build-optimized`:
> reiniciá-lo agora sobe o **candidato**, não o original. Qualquer retomada
> deve (a) preservar o incremental atual do candidato, (b) preparar um baseline
> **conhecido** em diretório separado com hashes/configuração próprios, e (c)
> declarar esse baseline como novo, nunca como o original.

## Baseline de produção (artefatos de observação, antes da 1ª alteração)
- Branch base: `main` @ `982eaadaa` (`perf(server): allow prompt cache reuse from 1% common prefix`).
- Branch tarefa: `feat/occupancy-position-split` baseada no `main`, com a revisão
  do plano carregada como working tree (commitada neste log, não descartada).
- Binário produção: `build-optimized/bin/llama-server` (stub 16k + `.so`),
  `version: 0.4.7-dev (build 12344, commit 3e104c3fd-dirty)`.
- Serviço systemd: `llama-server-root.service` (`WorkingDirectory=.../build-optimized`,
  `ExecStart=.../bin/llama-server --models-preset /home/hjotha/router-production.ini
  --models-max 1 --host 0.0.0.0 --port 8090 --metrics`,
  env `GGML_CUDA_GRAPH_RECOVERY_HEADROOM_MB=128` + drop-ins
  `20-mixed-kv.conf` (=18/`GGML_KVARN_WINDOW_CHUNK=4096`) e `30-tmpdir.conf`
  (`TMPDIR=/var/tmp/beellama-spool`)). PIDs 137629 (router :8090) + 137663
  (inferência 127.0.0.1:46583, preset expandido XXL 102400 puro / XXXL 131072
  misto 11+5, `remote-attn vulkan:0 q4_0`, MTP0 em XXL/XXXL).
- Modelo: `/home/hjotha/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf`
  (11G, sha256 `e23eb251...`), `router-production.ini` vigente, slot-cache
  `/home/hjotha/llama-slot-cache/` (9,1G), `ps` completo em
  `/tmp/prod-llama-ps-baseline.txt`.
- GPU: RTX 4070 12GiB, driver 610.57.04. Vulkan `vulkaninfo` falha sem DISPLAY
  (`vkCreateDevice ... ERROR_INITIALIZATION_FAILED`) — atenção Radeon via
  backend Vulkan do llama ainda a validar em runtime isolado.
- Restauração: **INVALIDADA** — `systemctl restart llama-server-root` +
  checagem `:8090/metrics` sobe o candidato de `build-optimized`, não o
  baseline (que foi sobrescrito). Rollback real exige o baseline conhecido
  separado preparado cedo nesta campanha.
  e inferência real; binários/configs acima são o estado a restaurar.

## Mapeamento (subagentes read-only, nada editado por eles)
- Cache misto/atenção: `src/llama-kv-cache*.cpp`, `llama-kv-mixed-*`,
  `llama-memory-hybrid.cpp:92-148`, `llama-batch.cpp:513/684`,
  `llama-graph.cpp:3938-4097` (`build_attn`), `models/qwen35.cpp:335-483`,
  `llama-context.cpp:1223-1397` (auto-placement N remoto), snapshots
  `llama-kv-cache-kvarn.cpp:3684-4123` + `llama-kv-mixed-state.*` (codec v1
  `KMS1`, anti-duplicata `serialize:372`/`parse:611`), handoff
  `server-mixed-kv-handoff.cpp` + `server-context.cpp:6129-6186`.
- FA/LSE: CUDA `fattn-kvarn-dispatch.cu:1165-1340`, `vec/decode/MMA/windowed`,
  `(m,l)=float2` interno nunca exposto como LSE; Vulkan `kvarn_flash_attn.comp`
  + Q4 `split_k`/`fa_split_k_reduce` sem saída LSE; scheduler
  `ggml-backend.cpp:1770-1778` serializa troca de backend (`n_inputs==0`);
  merge LSE inexistente; domínios V `ROTATED_K_ORIGINAL_V` vs `attn_rot_v`.
- Servidor adaptativo: tiers S..XXXL em `router-production.ini:61-116`,
  seleção `common.cpp:3021-3056` + `server-context.cpp:5420-5450/7432`,
  RAM vs disco (`auto_index_*`, `auto_restore_*`), staging puro→misto,
  orçamentos `common.cpp:1916-2136` + `llama-context.cpp:1377-1413`, grafos
  pré-reservados `llama-context.cpp:1927-2012`.

## Fase 0 — entregue neste commit
- `src/llama-position-split.h` (novo): contratos CPU/testáveis — `C=131072`,
  `P` alvo 102400, alinhamento 256/grupo 128, `is_valid_p/align_p_down`,
  `range_desc` + `validate_ranges` (local `[0,P)` CUDA/KVarN4, overflow `[P,C)`
  Vulkan0/Q4, sem lacuna/sobreposição/duplicata), `ubatch_crosses_p` +
  `validate_recurrent_window` (rejeita prepare sem mutação se janela cruza P),
  `lse_from_ml` (`m+log(l)`, vazio→`-inf`), `merge_o_lse/merge_lse` (fórmula
  §3.3 + regras §3.4), `nrmse`, `choose_p` (§3.8: maior P≤102400 com margem
  `max(512MiB,5% VRAM)`).
- `tests/test-position-split-merge.cpp` (novo, 51 checks): merge básico vs
  referência direta, faixas vazias (`O=0/LSE=-inf`, passthrough), `lse_from_ml`
  (incl. NaN/Inf→`-inf`), faixas (ok/lacuna/sobreposição/duplicata/ausente/
  backend errado, P desalinhado), fronteira ubatch + casos `P±129/128/1/0` +
  janela recorrente, `choose_p` (cheio/parcial/rejeição).
  Validação: `g++ -std=c++17 -O2 -Wall -Wextra -I src ...` → 51 checks,
  0 falhas (`/tmp/test-position-split-merge`).
- `tests/CMakeLists.txt`: alvo `test-position-split-merge` (label `main`).
- Revisão do plano preservada integralmente (diff de
  `docs/occupancy-plan-review-20261003` commitado aqui).

Comandos:
```
g++ -std=c++17 -O2 -Wall -Wextra -I src tests/test-position-split-merge.cpp \
  -o /tmp/test-position-split-merge && /tmp/test-position-split-merge
git checkout -b feat/occupancy-position-split main
```

## Fases 2-7 — pendentes (não declaradas concluídas)
- F1 (LSE nas rotas): expor F32/query/head após chunks+tail no CUDA
  (windowed/decode Q=1/MMA/portable), após `fa_split_k_reduce` (`split_k>1`)
  ou kernel final (`=1`) no Vulkan, estender `ggml_kv_tail_attention_merge*`
  para O+LSE; matriz de rotas §3.4 + prova de rota executada.
- F2 (cache por posição): tabela única + 2 stores/camada em
  `llama-kv-cache-kvarn.h:482-518/570/573`, `cpy_*` por faixa
  (`llama-kv-cache.h:324-337`, `llama-graph.cpp:3820/3885/3938/4016-4052`),
  corte P em `llama-memory-hybrid.cpp:92-114` + `llama-batch.cpp:513` antes de
  `mem_recr->prepare`, rejeição explícita se janela `n_rs_seq+1` cruza P,
  atomicidade entre faixas + DeltaNet.
- F3 (2 FA + merge sequencial): `build_attn` + `qwen35.cpp:335` com
  `n_kv≤P` só-CUDA (sem nó Vulkan) e `n_kv>P` com máscara local só-padding +
  overflow offset P, cópias Q/O/LSE, merge CUDA, gate/`wo` na CUDA; validar
  4B P=512/1024 prefill/decode cruzando P.
- F4 (estado/snapshots): envelope v2 `position_split` + P/C/ocupação +
  `(layer,range)`, writer/reader/validadores/spool/handoff/índice disco,
  importação explícita XXL (`n≤P`) em processo novo só com XXL em disco,
  P diferente→miss, split→CUDA só se `n≤min(P,cap)`, corrompido/versão
  desconhecida rejeitados antes de mutar.
- F5 (adaptativa + 27B): P real fixo por contexto via pico completo §3.8
  (pesos+KV+stage/tail+RS+grafos pp/tg+O/LSE+cópias+restore cruzando P),
  A/B alternado 5× + aquecimento, 105k/115k/130560, 131072 cheio,
  130560+512, 10 ciclos restore/divergência, picos por backend.
- F6 (sobreposição): só após F0-F5 verdes + ganho mediano ≥5% ponta a ponta.
- F7 (prontidão): canário temporário + rollback XXL/XXXL documentado; sem
  merge no main nem ativação permanente.

Nenhum ganho presumido; números históricos (§2/D) são estimativas do termo
Radeon, não latência total. Próximo passo: F1 com 4B P pequeno, depois
campanha 27B, tudo em porta isolada sem contaminar produção.

## F1 — contrato executável + CPU verde (2026-10-03/04, produção parada)
- Contrato GGML: `GGML_FLASH_ATTN_EXT_OP_PARAM_LSE_OUT=7` + 
  `ggml_flash_attn_ext_add_lse_out(a, lse)` / `has_lse_out` (`ggml.h`/`ggml.c`).
  LSE reusa `src[4]` (slot de sinks); sinks e LSE mutuamente exclusivos
  (attach asserts). Layout: F32 contíguo `ne={n_head_q, n_q, n_batch}`,
  `lse[(b*n_q+q)*n_head_q+h]`; vazio/mascarado: O=0, LSE=-inf.
  Backends sem LSE na rota devem falhar o op (fail-closed), não escrever lixo.
- CPU (`ggml-cpu/ops.cpp`): helper `ggml_fattn_ext_write_lse` + 5 sites
  (one-chunk, tiled, split-reduce, kvarn-native, tail_ref), com gate de sinks.
- CUDA (NÃO compilado ainda — validação pendente no rebuild total):
  helpers em `fattn-mma-kvarn-case-decl.cuh`, LSE em decode-combine
  (cobre split e vec), windowed single/multi finalize, portable-combine;
  gates de sinks; rejeição explícita no portable single-split com LSE.
  Faltam: FA genérica (`fattn.cu`), MMA fallback genérico e `fattn-tail.cuh`.
- `tests/test-position-split-lse.cpp` (novo, 306 checks, 0 falhas no build
  CPU-only `/tmp/build-split-cpu`): Q=1 decode + prefill Q=2/32, GQA 4/2,
  D=64 e D=256, F16 e Q4_0, causal/cheia/fileira-mascarada, merge E2E
  FA[0,5)+FA[5,12) vs FA cheia. Gates estritos cumpridos SEM afrouxar:
  NRMSE≤1e-3, |O|≤1e-2, |LSE|≤1e-2, -inf exato, sem NaN/Inf.
  Achado registrado (não é afrouxamento): a referência usa KV dequantizado +
  Q efetivo do kernel (CPU Q4_0 usa dot Q8_0 — kernel LSE bate com referência
  Q8-FP64 nos 5 decimais; o "erro" vs Q pristino é viés de representação
  pré-existente, que o softmax cancela em O). KVarN-native e body+tail no
  F3/integração.
- Produção parada (`systemctl stop llama-server-root`, VRAM 11881→1 MiB);
  `test-kvarn` passou (36s) sem contenção. Baseline recuperável em
  `/tmp/prod-ps-full-20261003.txt` + `/tmp/router-production-baseline.ini`.

## F1 validação CUDA (compilação)
- Primeira tentativa de rebuild total expôs 3 erros reais (todos corrigidos):
  `domain()` usado antes de definido em `case-decl.cuh` (reordenado),
  `window_enabled`/`window_chunk` duplicados entre `case.cuh` e `case-decl`
  (unificados no decl), falta de `<cstdlib>` e de include do decl no
  `fattn-tail.cuh`. Um erro fantasma no scatter ("expected an expression"
  em código pré-existente intacto) era cascata do decl quebrado — bisseção
  por hunks confirmou que cada hunk do tail compila isoladamente; após os
  fixes o TU compila com o conteúdo integral.
- Rebuild total em andamento (`build-optimized`, commit `56c46ce05`).

## F1b — revisão externa A/B/C: contrato LSE como nó real (2026-10-04)
- Revisão C (fatal, aceita): o LSE em `src[4]` **não tem aresta FA→LSE**. O
  scheduler podia stagear uma cópia de leaf do tensor LSE no backend do FA, o
  kernel escrevia nessa cópia e um consumidor posterior lia o buffer original.
- Contrato novo (`GGML_OP_FLASH_ATTN_EXT_LSE`, op 108, inserido no fim do
  enum para não deslocar valores existentes):
  - `ggml_flash_attn_ext_lse_out(ctx, fa)` cria um tensor F32
    `ne={n_head_q,n_q,n_batch}` com `src[0]=fa`: existe no grafo como **nó**,
    então a aresta FA→LSE dá ordenação, lifetime de buffer para todos os
    consumidores e cópia cross-backend correta a partir do buffer do FA.
  - O FA guarda apenas um back-pointer para o LSE em `op_params[8]` (slot
    `GGML_FLASH_ATTN_EXT_OP_PARAM_LSE_PTR`, via memcpy de `uintptr_t`); o
    hash de tensor é por identidade de ponteiro, então o grafo não muda de
    hash; nada é serializado (RPC não suporta FA-LSE).
  - `src[4]` volta a ser exclusivamente sinks. `ggml_flash_attn_ext_add_lse_out`
    foi removida (API interna da branch).
  - Backends: caso no-op de compute + `supports_op` em CPU
    (`ggml-cpu.c`/`ggml-cpu.cpp`), CUDA (`ggml-cuda.cu`) e Vulkan
    (`ggml-vulkan.cpp`); kernels resolvem o destino por
    `ggml_flash_attn_ext_get_lse_out(dst)`.
  - Scheduler: novo **pass 4b** em `ggml_backend_sched_split_graph` fixa o
    LSE no backend do nó FA depois de todos os heurísticos (nenhuma passagem
    posterior move o nó), com assert de suporte.
  - `GGML_OP_COUNT` 107→108 e `RPC_PROTO_PATCH_VERSION` 2→3
    (`ggml-rpc.h` static_assert).
- F1a Vulkan: causa raiz do desvio `+ln2` encontrada e corrigida. Em
  `flash_attn_base.glsl` o range de blocos KV por split é
  `start_j = split_k_index*split_kv/Bc`, `end_j = CEIL_DIV(min(KV,(split_k_index+1)*split_kv), Bc)`;
  com `split_kv` não múltiplo de `Bc` (bisection forçado) dois splits liam o
  mesmo bloco, duplicando o denominador (+ln2 em LSE e no O). Agora o split
  forçado arredonda para múltiplo de `Bc`
  (`split_kv = ROUNDUP(CEIL_DIV(KV,2), Bc)`, `split_k = CEIL_DIV(KV, split_kv)`,
  assert `split_k>=2 && split_kv%Bc==0`). `supported_op` declara
  `KV >= 2*Bc` (Bc de `get_fa_tuning_params`) e **recusa fail-closed** abaixo
  disso — nada de skip.
- Testes novos/atualizados (todos fail-closed; recusa é check, não skip):
  - `tests/test-position-split-lse-vk.cpp` (registrado no CMake): 335 checks,
    0 falhas. Sonda o limiar do device (neste host `kv_min=128`, Bc=64),
    exige "recusa abaixo, aceita acima"; O da rota LSE é **bit a bit igual**
    ao da rota FA sem LSE (NRMSE 0.000e+00 em todos os casos), LSE vs FP64
    ≤ 9.2e-3 (gate do plano 1e-2); cobertura F16 e Q4_0, decode/prefill,
    GQA 2/4, máscara causal/cheio/fileira vazia.
  - `tests/test-position-split-lse-sched.cpp` (novo, revisão C): FA em
    Vulkan, consumidor `sum` pinado em CPU → 20 checks, 0 falhas. Prova nó de
    grafo real, ordem FA→LSE (callback de eval), co-localização
    `lse==fa==Vulkan0`, `splits=2`/`copies=1` e que o consumidor CPU lê
    exatamente os valores escritos pelo kernel (sum_gpu==sum_cpu), que era o
    hazard do `src[4]`.
  - `tests/test-position-split-lse-cu.cpp` (novo, registrado): 7 checks,
    0 falhas. FA genérico CUDA continua **recusando** LSE (fail-closed) e o
    nó `FLASH_ATTN_EXT_LSE` é suportado/co-localizado no CUDA para as rotas
    KVarN.
  - `tests/test-position-split-lse.cpp` (CPU): 306 checks, 0 falhas com o nó.
- Regressões: `test-backend-ops -o FLASH_ATTN_EXT` em CPU `4/4 backends
  passed` (exit 0) e em **Vulkan0 5392 casos OK, `4/4 backends passed`**
  (exit 0) — a rota split-K corrigida não afeta o FA sem LSE.
- `ctest -R "position-split|kvarn|test-alloc"`: 22/23. A única falha,
  `test-kvarn-mtp-sharing-static`, é **pré-existente**: o arquivo
  `src/llama-kv-cache-kvarn.cpp` está intocado nesta branch e já não contém a
  string exigida em `main` (`git show main:...| grep -c` = 0).

## Segunda revisão (2026-10-04): itens 0-4 tratados
- **Item 0 (lifetime/cópias) — corrigido.** `ggml-alloc.c` agora reserva o nó
  `FLASH_ATTN_EXT_LSE` na posição do produtor (antes de liberar as entradas do
  FA), tanto em `alloc_graph` quanto em `reserve`;
  `ggml-backend.cpp:graph_copy_dup_tensor` duplica e religa o tensor LSE no
  clone; `ggml-rpc.cpp` recusa FA-LSE e o nó LSE em `supports_op` e afirma em
  `serialize_graph` antes de enviar ponteiro. Regressão nova
  `tests/test-position-split-lse-alloc.cpp`: **230 checks, 0 falhas** (duas
  faixas + merge, assert de intervalo LSE vs entrada/saída do produtor,
  reserve, reset do scheduler, reexecução, com e sem callback de eval, e
  graph-copy com sentinel). Com a reserva desabilitada o mesmo teste acusa
  **12 falhas de overlap** (verificado).
- **Item 1 (gate FP64) — restaurado.** A reprodução estrita acusava O vs FP64
  de 1.17e-3..4.33e-3. Causa medida: o acumulador `Of` do FA Vulkan é
  `FLOAT_TYPE` (fp16 no modo default); com `GGML_PREC_F32` os mesmos casos dão
  3.9e-4..8.5e-4, e a rota LSE é **bit a bit igual** à rota sem LSE. Gates
  atuais: O vs FP64 com acumulação fp32 (≤1e-3/≤1e-2), split-vs-unsplit no
  modo de produção (≤1e-3/≤1e-2, exatamente 0 quando ambas usam o reduce),
  LSE vs FP64 ≤1e-2 com fp32 (o modo default expõe o erro absoluto do score:
  3e-2..6e-2 em D=256, reportado com teto de sanidade), CPU apenas como
  cross-check. Vazamento `*(new std::vector<float>())` corrigido.
- **Item 4 (Vulkan split_k=1) — implementado.** Um pedido de LSE sempre passa
  pelos partials, mesmo com `split_k == 1`, e o reduce publica O e LSE de um
  único parcial. Os shaders escrevem partials com `k_num == 1` via novo
  `LSE_PARTIALS_BIT` no `mask_n_head_log2` empacotado (mesmo padrão do
  `SINK_ENABLE_BIT`; um campo dedicado não cabia em 128 bytes de push
  constant), com a condição `k_num > 1 || LSE_PARTIALS_BIT` para não alterar a
  rota split_k>1 existente. A recusa `KV >= 2*Bc` foi removida.
  Cobertura: KV 1/3/17 (split_k=1), 128/256/384/512, D=256 com 24:4 e prefill
  de 256 tokens, F16 e Q4_0, decode/prefill, GQA 1/2/4/6,
  causal/cheio/fileira vazia → **1105 checks, 0 falhas**. Regressão
  `test-backend-ops -o FLASH_ATTN_EXT` no Vulkan0: 5392 casos OK, 4/4 backends,
  exit 0.
- **Item 2/3 (CUDA) — parcial, com um bug real encontrado.** O atalho
  single-window `Q >= 512` agora é pulado quando há LSE (ele retornava antes
  dos finalizadores). Os sub-passos do tail (`body_pass`/`tail_pass`) recebem a
  limpieza do attachment LSE e o tail-only resolve o destino por
  `ggml_flash_attn_ext_get_lse_out` (src[4] voltou a ser só sinks).
  Prova numérica no harness de `tests/test-kvarn.cpp` (novo parâmetro
  `lse_output`, sentinel antes do compute): rota KVarN **decode** publicando
  LSE com `max|dLSE| = 1.621e-05` contra a referência CPU.
  **Bug encontrado e fail-closed**: com `Q > 1` a rota windowed multi-chunk
  escrevia O errado junto com LSE (rmse 2.4e-1 contra a mesma rota sem LSE em
  Q=2, enquanto Q=1 bate em 2.9e-5). Até a correção do metadata multi-chunk,
  `ggml_cuda_flash_attn_ext_kvarn_supported` **recusa** LSE com `Q > 1`.
- **Item 5 (cobertura) — parcial.** CUDA agora tem prova numérica (não só
  `supports_op`) via harness do test-kvarn; Vulkan tem a matriz do plano
  (D=256, GQA 24:4, Q=256). A falha ampla do CUDA em `hsk=320`
  (`GGML_ASSERT(vec_case != nullptr)`, `fattn.cu:595`) segue intocada e não foi
  investigada: é anterior à branch e fora das formas obrigatórias.
- **Item 6 (documentação) — corrigido** no topo (baseline não recuperável) e
  no F2a (divisão em P, não rejeição).

## Terceira revisão (2026-10-04): correções e fail-closed
- P1 do clone: `graph_copy_dup_tensor` publica o clone no mapa **antes** de
  percorrer arestas que podem retornar a ele. Regressão do teste de clone:
  232 checks, 0 falhas (eram 2).
- Vulkan LSE: a rota LSE agora força acumulação fp32 (`!device->fp16 ||
  prec==F32 || ktype==BF16 || LSE presente`). A matriz real
  (D=256, nqh=24, nkh=4, nq=256, nkv=512) passou a casar no modo de produção
  do teste: O nrmse=7.558e-4 (gate 1e-3), LSE max=5.008e-3 (gate 1e-2); o
  mesmo pedido com `GGML_PREC_F32` explícito casa; `split(f32)` exato 0. O
  modo default da rota plain continua exposto como piso fp16 her드ado e é
  reportado, não aceito como"rota LSE de produção".
- CUDA KVarN LSE: revisado o harness (`route_reset` antes do compute, O da
  rota LSE contra a rota plain). Descoberta gravíssima: a rota decode KVarN
  publica O errado com LSE (rmse 2.3e-1 em Q=1 contra a mesma rota sem LSE) e
  a windowed Q=2 tem o mesmo defeito (rmse 2.4e-1). Mesmo com LSE corretao
  (Q=1: max|dLSE|=1.621e-05 vs CPU), o merge produziria O errado. Blindagem:
  `ggml_cuda_flash_attn_ext_kvarn_supported` e o dispatch do executor
  recusam/abortam qualquer LSE sobre vistas KVarN (fail-closed), com o motivo
  medido registrado. A reabilitação depende de `dst->src[8]` (metadata) ser
  sempre publicado também quando um LSE exclusivo é anexado. O tail com LSE
  permanece recusado fail-closed porque o caminho interno (FA genérico) não
  exporta LSE; o caminho KVarN-native permanece bloqueado pela mesma
  blindagem.
- Testes: clone-edge reparado (232/0), VK real Q256 (485/0), suíte LSE
  completa 6/6 verde, `test-kvarn` verde. Pendência implementável removida do
  cronograma atual: não anunciar LSE em KVarN como entregue.

## Pendências reais após esta rodada
- CUDA KVarN multi-query com LSE (Q>1): corrigir o metadata multi-chunk da rota
  windowed e reabilitar; enquanto isso, recusa fail-closed.
- Vulkan D=256: LSE no modo default tem erro absoluto 3e-2..6e-2 (precisão
  fp16 herdada do device); o gate obrigatório roda com acumulação fp32.
- `fattn.cu:595` (hsk=320) e `test-kvarn-mtp-sharing-static` (pré-existente em
  `main`): fora do escopo desta rodada.
- F2 (allocator/cache por posição), F3 (grafo/merge), F4 (snapshots), F5
  (27B), F7 (canário/rollback) continuam não implementados.
- Produção segue parada; binário original perdido (ver aviso no topo).

## Desenho F2/F3/F4 (para implementar após o rebuild verde)
- F2a (ubatch/P, **vigente**): o allocator **DIVIDE** os ubatches comuns na
  fronteira P antes de preparar atenção/recorrência; somente a janela
  protegida `1+n_rs_seq` (DeltaNet) que cruza P é **rejeitada antes de
  qualquer mutação** (`FAILED_PREPARE` com motivo explícito, prepare ainda sem
  rodar). A versão anterior deste documento propunha rejeitar qualquer ubatch
  que cruzasse P; isso está corrigido aqui e é o comportamento a implementar.
  Atomicidade a testar: se a segunda preparação (após o split) falhar, nada
  pode ter sido mutado — inclusive o caminho DeltaNet. P chega via cparams
  (contexto) — origem: perfil adaptativo §3.8.
- F2b (cache por posição, núcleo): em `llama_kv_cache_kvarn`, quando P>0,
  cada camada full ganha faixa overflow Q4 em `standard_cache` (cap C-P por
  camada, Vulkan0), tabela lógica única `pos<->cell` no `metadata`,
  `get_k/v_for_attention` por faixa, `cpy_k/v` com índices por faixa,
  `seq_rm`/rollback atômicos (valida tudo antes de mutar; fora da reserva
  exata → checkpoint+replay ou rejeição), tail/stage só no local.
  Tudo atrás de `position_split_enabled()` (P>0); P==0 = comportamento atual.
- F3 (grafo): em `build_attn`/`build_layer_attn`, se P>0 e `n_kv>P`: FA
  local (KVarN/CUDA, máscara só-padding) + FA overflow (Q4/Vulkan, máscara
  com offset P) + cópias Q/O/LSE + merge por composição GGML
  (`exp/sub/mul/add/div` existentes, primeiro; op dedicada só se medição
  mandar) + gate/`wo` na CUDA. Se `n_kv<=P`: caminho só-CUDA atual, e o
  teste prova ausência de nós Vulkan no grafo. Q copiado para Vulkan uma
  vez por camada (reuso entre... não há reuso entre camadas; 16 cópias Q).
- F4 (estado v2): envelope `KMS2`/`position_split` + P/C/ocupação +
  descritores `(layer,range)`; writer/reader/validadores/spool/handoff/
  índice; leitor velho rejeita v2, novo preserva v1; importação XXL
  (`n<=P`) em processo novo só com XXL em disco; P diferente→miss.
