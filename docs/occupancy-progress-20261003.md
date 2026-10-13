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
  rota LSE contra a rota plain). Medido: rota decode KVarN com LSE publicava
  O com rmse 2.3e-1 em Q=1 contra a mesma rota sem LSE, e a windowed Q=2 tinha
  o mesmo defeito (rmse 2.4e-1); LSE em si estava correto (Q=1:
  max|dLSE|=1.621e-05 vs CPU). Blindagem fail-closed aplicada.
  **SUPERADO pela quarta revisão**: o O medido pertencia a um tensor que o
  grafo do harness nunca calculou (ver seção da quarta revisão). O defeito do
  kernel era outro e real: indexação de O/LSE por ordem empacotada em vez de
  strides. Nenhuma das duas conclusões anteriores ("metadata multi-chunk
  quebrado" e "reabilitação depende de src[8]") se confirmou.
- Testes: clone-edge reparado (232/0), VK real Q256 (485/0), suíte LSE
  completa 6/6 verde, `test-kvarn` verde. Pendência implementável removida do
  cronograma atual: não anunciar LSE em KVarN como entregue.

## Pendências reais após a terceira revisão (SUPERADO em parte pela quarta)
As duas primeiras pendências abaixo foram resolvidas na quarta revisão; o resto
permanece. Ver "Checklist vivo" para o estado corrente.
- CUDA KVarN multi-query com LSE (Q>1): corrigir o metadata multi-chunk da rota
  windowed e reabilitar; enquanto isso, recusa fail-closed.
- Vulkan D=256: LSE no modo default tem erro absoluto 3e-2..6e-2 (precisão
  fp16 herdada do device); o gate obrigatório roda com acumulação fp32.
- `fattn.cu:595` (hsk=320) e `test-kvarn-mtp-sharing-static` (pré-existente em
  `main`): fora do escopo desta rodada.
- F2 (allocator/cache por posição), F3 (grafo/merge), F4 (snapshots), F5
  (27B), F7 (canário/rollback) continuam não implementados.
- Produção segue parada; binário original perdido (ver aviso no topo).


## Quarta revisão (2026-10-04): LSE real, strides e referência FP64
Base: `638c565b8`. Commits: `a8ad7084c`, `172aee665`.

### O que a revisão 4 reproduziu e o que era defeito real
- P1 do teste (confirmado): o laço novo indexava o LSE pelo tamanho de O
  (384 vs 6 no primeiro caso) e abortava sob `_GLIBCXX_ASSERTIONS`; e o teste
  aceitava LSE não finito, omitia `max_abs(O)` e perdera a comparação GPU
  sem attachment.
- **Defeito de kernel real (corrigido em `a8ad7084c`)**: o
  `ggml_cuda_fattn_kvarn_decode_combine_kernel` indexava O e LSE pela ordem
  empacotada `(query, head)`. O tensor de saída de FA é
  `ne = {D, q->ne[2], q->ne[1], q->ne[3]}` e o LSE é
  `ne = {n_q_heads, n_q, n_stream}` com o eixo de *head* contíguo. Medido: só
  a primeira linha de LSE era escrita e as demais ficavam com o sentinela
  `-7777`, com escrita fora do fim do tensor para as demais heads. Os dois
  kernels de merge do tail receberam os mesmos strides. `src[8]` mantém o
  significado histórico (empacotado) nesse kernel porque no contrato de tail
  ele é o tensor de *query order*, não um sink de metadata.
- **Defeito de harness (corrigido)**: o grafo era expandido só até o nó LSE,
  o que descarta a transformação de pós-FA (`out` é reatribuído ao resultado
  do WHT quando `rotate_graph`), então o O comparado vinha de um tensor nunca
  calculado — all zeros. As duas raízes (O e LSE) são expandidas agora. Esse
  era o defeito por trás do "O errado" da terceira revisão.
- **Defeito de harness (corrigido)**: o preenchimento de Q usava índice plano
  fixo, que embaralha silenciosamente o layout de query permutado (o que
  `llama-graph.cpp` usa). O fill passou a usar os strides do próprio tensor.
- **Referência estrutural nova**: `compute_lse_fp64_reference` acumula em
  FP64 sobre os *registros KVarN dequantizados que o próprio grafo consumiu*,
  reaproveitando o WHT de referência do host, a mesma máscara, o mesmo
  mapeamento GQA e a mesma escala. A rota materializada em F32 da CPU é
  medida contra a mesma referência como controle, então erro de rota e erro de
  representação ficam separados.

### Matriz medida (test-kvarn, CUDA0/RTX 4070)
18 formas, 0 falhando, 7 known-open. Referência FP64:
- NRMSE(O) entre 8.7e-07 e 1.7e-04 (gate 1e-3); `max_abs(O)` <= 5.4e-05
  (gate 1e-2); `max_abs(LSE)` <= 5.7e-05 (gate 1e-2).
- O idêntico com e sem o attachment LSE em toda forma de mesma rota
  (ex.: 0.000e+00 em D64/D128 Q1/Q2).
- Controle CPU `max_abs` até 2.5e-03, ou seja a referência FP64 é mais firme
  que a rota materializada F32, como esperado.
- Cobertura: D64/D128/D256 x Q1/Q2/Q256/Q512, D256 GQA24:4 em Q1/Q2/Q256,
  Q permutado (produção) Q4/Q8, tudo mascarado Q1/Q256 (O=0 e LSE=-inf exatos),
  sentinela não escrito rejeitado, telemetria de rota antes/depois de cada
  compute.
- As formas Q256/Q512 com LSE trocam de rota (windowed prefill -> portable
  native); por isso a checagem de isolamento por attachment só é exigida quando
  a telemetria mostra a mesma rota, e a diferença residual (1.3e-03) está
  dentro dos gates absolutos contra a FP64 (NRMSE 1.2e-06).

### Known-open medido (não conta como cobertura do §3.4)
- `d256-q1-tail`: NRMSE(O)=0.999994, max_abs(O)=3.28e+04, max_abs(LSE)=1.62e+04
- `d256-q512-tail`: NRMSE(O)=1, max_abs(O)=4.69e+04, max_abs(LSE)=1.68e+03
- `d256-q256-tail`: NRMSE(O)=0.001680, max_abs(O)=0.00416 (marginal)
O controle CPU das mesmas formas fica em 6.4e-07, ou seja a referência FP64
modela o tail corretamente e o desvio está na rota CUDA de tail. As formas
known-open são impressas com os números exatos a cada execução e listadas
aqui; elas não ficam verdes por skip silencioso. O tail *sem body* continua
recusado em `ggml_cuda_flash_attn_ext_supported` (`lse_requested &&
tail_bodyless`), o que é recusa explícita e não entrega do item do §3.4.

### Regressões desta rodada
- build completo: exit 0.
- `test-kvarn`: exit 0, `all tests OK`.
- `ctest -R "position-split-lse|kvarn"`: 21/22; a única falha,
  `test-kvarn-mtp-sharing-static`, falha igual em `638c565b8` (pré-existente).
- `test-backend-ops -o FLASH_ATTN_EXT`: CPU0 3/3 e Vulkan0 3/3, exit 0.
- `test-backend-ops` em CUDA0 continua abortando em
  `ggml/src/ggml-cuda/fattn.cu:595` (`GGML_ASSERT(vec_case != nullptr)`,
  hsk=320 com sinks): caminho genérico não-KVarN, intocado por esta rodada.


## F3 core — duas FA + merge no mesmo grafo (2026-10-04)
Teste: `tests/test-position-split-merge-graph.cpp`.

O mecanismo do §3.3 foi montado e roda em um grafo único com o scheduler real:
- faixa local F16 na CPU e faixa overflow Q4 no Vulkan0, cada uma com seu nó
  `FLASH_ATTN_EXT` e seu `FLASH_ATTN_EXT_LSE`;
- as cópias entre backends ficam por conta do scheduler (nada de `cpy` manual);
- o merge usa apenas operações existentes: `clamp` (contrato de faixa vazia,
  LSE = -1e30 em vez de -inf), `add`/`sub`/`abs`/`scale` para o máximo
  elemento-a-elemento sem op de max, `exp`, `mul`, `add`, `div` e `log`; o
  denominador recebe `+eps` para que duas faixas vazias deem 0 exato em vez de
  NaN;
- os pesos são vistos como `[1, n_q_heads, n_q]` com os strides do tensor LSE
  (`nb[1]` para head, `nb[2]` para query), que é o que torna a composição
  correta para o `ne` declarado de O e de LSE;
- abaixo de P o grafo tem **um** nó de atenção (verificado contando os nós
  `FLASH_ATTN_EXT`), sem segundo range e sem aritmética de merge: nada pode rodar
  no segundo backend (§3.8);
- a rota CUDA local continua recusando LSE fora do KVarN, então o
  par exercitado aqui é CPU + Vulkan0; o par CUDA/KVarN + Vulkan/Q4 é coberto
  separadamente por `test-kvarn` e `test-position-split-lse-vk`.

### Known-open medido (não conta como cobertura do §3.3)
- O comparativo de O contra a referência FP64 **não** está verde neste formato:
  `merge Q=1 nrmse=2.092e+00 max_abs=4.575e-01`, `Q=2 nrmse=2.766e+00`,
  `Q=4 nrmse=2.811e+00`, `below P nrmse=1.044e+00`,
  `empty overflow nrmse=1.044e+00`, `merged lse max_abs=5.782e-02`.
- Isolamento feito até aqui: o LSE merged concorda com a referência dentro de
  ~1e-2, logo pontuação e pesos concordam e o resíduo está no caminho de O.
  `ggml_prec_set_acc(F32)` não muda o resultado.
- **Causa isolada (probe dedicated, não o merge)**: com `K = 0` (todas as
  pontuações iguais, pesos uniformes) e `V[t] = t`, a FA da CPU deste fork
  devolve a média de `[0, NKV/2)`, não de `[0, NKV)`:
  `NKV=128 -> 31.5`, `NKV=256 -> 63.5`, `NKV=512 -> 127.5`, `NKV=1024 -> 255.5`
  (média uniforme correta seria `NKV/2 - 0.5`, o dobro em todos os casos). Com
  `NQ=4` o desvio muda de linha de saída, confirmando que o intervalo efetivo de
  chaves depende do layout/saída e não é simply `[0, n_kv)`. Enquanto esse
  comportamento não for entendido, a FA da CPU **não serve** como referência
  estrutural para o range local neste formato, e os números de O acima não podem
  ser atribuídos à composição do merge.
- A referência FP64 bate **exatamente** com a FA em dados estruturados pequenos
  (D=4, NKV=3, NQH=2, NKH=1), o que confirma que a fórmula da referência está
  certa e que o problema é o intervalo efetivo de chaves da FA neste formato.
- Próximo passo de diagnóstico: caracterizar o intervalo efetivo de chaves da
  FA da CPU para `NQ` 1/2/4 com máscara explícita (inclusive com máscara
  visível só na segunda metade de `[0,NKV)`), corrigir a referência ou escolher
  um backend de referência cuja cobertura de `[0,n_kv)` esteja provada, e só
  então fechar o gate do §4.1 para o merge. O merge em si (scheduler, cliques,
  pesos, `+eps`, um nó abaixo de P) já está validado estruturalmente.

## Checklist vivo (revisão 4, passo 0)
| Item | Estado | Evidência | Próximo passo |
|---|---|---|---|
| F0 contrato O/LSE | validado | `tests/test-position-split-lse*` verde; LSE como nó real com strides | — |
| F0 merge CPU + vazios/mascarados | validado | `test-position-split-lse` 306 checks; all-masked exato na matriz | — |
| F1 CUDA KVarN LSE (decode/prefill/Q>=512) | validado | matriz 18 formas, NRMSE <= 1.7e-04, O estável com/sem LSE | — |
| F1 CUDA KVarN LSE com tail (body+tail) | **aberto** | q1/q512 quebrados, q256 marginal (números acima) | diagnosticar limites de arena/`q_max` do caminho de tail por forma |
| F1 CUDA tail sem body com LSE | **aberto** | recusado em `supported()` | fazer o nó de tail publicar LSE no buffer do pai |
| F1 Vulkan LSE (split_k=1 e >1, F16/Q4) | validado | `test-position-split-lse-vk` 485 checks, O nrmse 7.558e-4, LSE 5.008e-3 | — |
| F1 alocador/cópia/scheduler/RPC | validado | 230/0, 20/0, diag de clone 232/0 | — |
| F2 cache por posição, store, allocator, rollback atômico | **pendente** | — | implementar §3.1/§3.2 em `llama-kv-cache-kvarn.*` + `llama-memory-hybrid*` |
| F3 core duas FA + merge (mecanismo) | **mecanismo validado** | `9a283459a`: aritmética do merge é **exata** (0.000e+00 em O e LSE, 5 formas) contra as saídas de faixa única do mesmo backend; uma faixa mascarada = faixa local exata; ambas mascaradas = O=0 e LSE=-inf; abaixo de P 1 nó de FA e 0 nós no backend overflow (n_kv 8/128/512); `25075 checks, 0 failures` | integrar `build_attn`/`qwen35.cpp` e fechar o par real CUDA/KVarN + Vulkan/Q4 |
| F3 integração 4B P=512/1024, KL FP64 | **pendente** | — | `build_attn`/`qwen35.cpp` + teste de integração com KL FP64 |
| F4 snapshots v2 + importação XXL | **pendente** | — | envelope `position_split`, writer/reader/validadores |
| F5 orçamento + campanha 27B | **pendente** | — | cálculo de P com custos simultâneos + A/B 5x |
| F7 entrega/canário/rollback | **pendente** | — | baseline conhecido separado; promoção continua não autorizada |

## Tail KVarN com LSE: gates duros, ainda reprovando (2026-10-04)
`tests/test-kvarn` não aceita mais known-open: as três formas de tail são
gateadas como as demais e a matriz fica **vermelha** (`18 shapes evaluated, 0
failing, 8 tail-shape violations`).
- `d256-q1-tail`: `NRMSE(O)=0.999961`, `max_abs(O)=5.3e+03`,
  `max_abs(LSE)=16199.6`.
- `d256-q256-tail`: `NRMSE(O)=2.02e-3`, `max_abs(LSE)=1.03e-2`.
- `d256-q512-tail`: `NRMSE(O)=1.0`, `max_abs(O)=3.12e+04`,
  `max_abs(LSE)=22565.6`.
- **Não é defeito de LSE:** `attach_max_abs` entre a rota com LSE e a rota sem
  LSE é `~3e-8` nas três formas; o controle CPU contra FP64 é `6e-7..1e-3`. As
  duas rotas GPU estão igualmente erradas, logo o defeito está no caminho do
  tail em si, não no attachment de LSE.
- **Leitura de memória não inicializada:** os valores mudam entre execuções
  idênticas (`max_abs(O)` de `3.17e+04`, `3.28e+04`, `4.69e+04`, `5.3e+03` na
  mesma forma), assinatura de buffer não preenchido, não de arredondamento.
- Layouts já conferidos e coerentes: a FA de tile escreve `(max, denom)` em
  `dst_final_meta` quando `gridDim.y == 1` e em `dst_meta` quando há split, e
  `flash_attn_combine_results` escreve `(sequence*ne01 + col)*ne02 + head`, que é
  exatamente o índice lido por
  `k_flash_attn_ext_tail_indexed_small`/`..._partials_merge`. No teste
  `rd->ne[0] == 6 + tail_stride`, então `body_packed == false` e o corpo não é
  reempacotado.
- Telemetria das três formas: `prefill=1, mma=0, vec=0, split=0, portable=0`,
  isto é, a passagem de corpo e a de tail caem na rota genérica (`dispatch`),
  não nas entradas KVarN. `GGML_CUDA_FATTN_KVARN_ENTRY_COMPACT_TAIL` declina.
- Próximo passo: instrumentar `k_flash_attn_ext_tail_indexed_small` com
  sentinel em `body_meta`/`tail_meta`/`scores` para localizar qual buffer chega
  não inicializado nas formas `q1` e `q512` (as duas com `NRMSE ≈ 1`).

## Achados do merge (2026-10-04, `9a283459a`)
- **O erro do merge é zero.** Combinando no host as duas saídas de faixa
  única do mesmo backend (`base` local + `ovf` overflow, ambos com máscara
  totalmente visível, que é exatamente o que o grafo alimenta o combine) o
  resultado reproduz a saída fundida do grafo com `max|ΔO| = 0.000e+00` e
  `max|ΔLSE| = 0.000e+00` nas 5 formas. O NRMSE do merge contra FP64 é
  praticamente idêntico ao erro de faixa única do backend
  (ex.: D256 Q1 `1.127e-3` fundido vs `1.036e-3` faixa única).
- **Layout da máscara:** `ne = {n_kv, n_q, n_kv_heads}` é o único formato que
  satisfaz `q->ne[2] % mask->ne[2] == 0` em `ggml.c:5639` (o eixo ne[2] é um
  eixo de cabeça). Formatos com `ne[2] = n_q` causam assert para `n_q=4,
  n_qh=6`. Este achado invalida as formas testadas antes com `ne[2] = n_q`.
- **Fidelidade por backend (gate fica com o dono da rota):** com KV F16 no
  backend Vulkan a faixa única mede `1.0e-3..1.1e-3` em D=256 Q1/Q256, acima
  do gate de `1e-3`; o merge não contribui para isso. As rotas do produto têm
  fidelidade medida abaixo do gate: CUDA KVarN `1.7e-4` (`test-kvarn`) e
  Vulkan Q4 `7.6e-4` (`test-position-split-lse-vk`). KV Q4_0 como faixa local
  piora o backend (`1.4e-3..1.6e-3`), por isso a faixa local do protótipo
  continua F16.
- **ABERTO — máscara causal por query não é aplicada pela FA:** com máscara
  causal `s <= t` em `ne = {n_kv, n_q, n_kv_heads}`, a FA deste build devolve
  `NRMSE 0.49..6.8` contra a referência FP64 (o caso Q=1, em que só a chave 0
  está visível, erra por fator ~2 em relação a "todas visíveis", compatível
  com a máscara ser ignorada). A máscara local totalmente visível bate
  (`8.5e-4`). Isto é um defeito do backend FA, não do merge, e precisa ser
  resolvido antes de F3 integrar a máscara real do overflow.
- **Abaixo de P** a estrutura está provada com a faixa local em backend
  distinto do overflow: 1 nó de FA, 0 nós no backend overflow. O número de
  fidelidade impresso nesse caso (`4.7e-4`, `1.7e-3`, `3.3e-3` para n_kv
  8/128/512) é da FA da CPU usada como substituta e cresce com n_kv por
 accumulate F16; o gate de fidelidade da faixa local do produto é o do
  `test-kvarn` na rota KVarN/CUDA.

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

## 2026-10-05 — ISOLAMENTO REAL POR PROCESSO: A=A2 e A=D PASSARAM

### Causa raiz do OOM e do SIGSEGV (medida, nao presumida)
1. **OOM (~2.7 GB por ramo, 32 recusas)**: `run_branch` criava um `llama_context`
   por ramo dentro do MESMO processo. Com KVarN + CUDA + espelho Vulkan, tres
   contextos (`dest/ctrl/pure`) ja estavam vivos e cada ramo novo pedia ~2.7 GB;
   a GPU nao devolve memoria entre ramos. **Nao era leak do driver** (o
   `ctx_holder` tem `~ctx_holder { llama_free(ctx); }`).
2. **SIGSEGV em `ggml_backend_buffer_free` dentro de `~llama_context()`**:
   `materialized_probe` criava um backend NOVO com `ggml_backend_dev_init(dev)`
   a cada chamada. Um backend extra inicializado e destruido no fim da funcao
   (enquanto o materializer ainda ia destruir buffers de grafo) corrompe o
   teardown. Alem disso havia um **double free**: o bloco chamava
   `llama_free(ch.ctx)` e o `ctx_holder` destruidor chamava de novo
   (`GGML_ASSERT(sched) failed`).

### Correcoes aplicadas
- Nenhum backend e criado sob demanda. `materialized_probe` recebe o backend de
  um `llama_context::backend_supporting_buft(buft)` novo, que percorre
  `backend_ptrs` (backends que o contexto JA possui) e devolve o primeiro que
  suporta o buffer type da folha. `validate_source` continua intacto.
- Sem double free: apenas `ctx_holder` libera o contexto.
- Modo **um processo por ramo** (`F4_TEST_BRANCH`) em
  `tests/test-kv-mixed-destination-intact.cpp`, com exatamente UM contexto vivo,
  exportando `pos`, flags e os vetores materializados + logits de UM token de
  prova. O processo sai e devolve a VRAM antes do proximo ramo.
- `tools/f4-branch-driver.py`: driver **sequencial** (nunca paralelo), um
  processo por ramo, mesmos parametros/mesmo binario/modelo/tokens, comparando
  os resultados externamente. Ordem dos gates: (1) contexto+consumidor validos,
  (2) A==A2, (3) A==D, so entao B/C.

### Bug real do driver encontrado e corrigido
`max((diff(x,y) or -1.0) ...)` devolvia `-1.0` para diferenca **zero**, porque
`0.0` e falsy em Python. Isso mascarou PASS como falha. Agora `diff()` devolve
`None` so para vetor ausente e `worst()` trata `None` como dominante.

### GATES MEDIDOS (n=1, processo por ramo)
```
A   pos_min=0 pos_max=0 cb=1 ca=1 dec=1
A2  cb=1 max_mat=0.0 max_logits=0.0        -> A=A2 PASS
D   wrote=1 restored=1 max_mat=0.0 max_logits=0.0 -> A=D PASS
B/C K.h0: B target_delta=0.0 neighbors=[0,0,0] -> FAIL | C all_delta=[0,0,0,0] -> PASS
B/C K.h1 / V.h0 / V.h1: identico
logits B=0.0 C=0.0
```
**Isolamento resolvido e provado**: A=A2 e A=D zeram nos campos materializados E
nos logits de um unico token de prova, com D restaurando antes de qualquer
decode, em processos independentes. `P=512`, `C=2048` reportados por ramo.

### POR QUE B AINDA NAO E INTERPRETAVEL (medido)
O leitor consumidor responde (`consumer_before=1`) mas o vetor materializado de
K e V sai **todo zero** (256 floats, `k0[0:6] = 0,0,0,0,0,0`). Com a linha de
base zerada, nenhuma mutacao pode produzir diferenca, entao `target_delta=0.0`
e um **erro de fixture**, nao evidencia de byte inerte. `C` passar e `logits=0.0`
sao consequencia do mesmo zero, nao prova de que o byte inativo e inerte.

### ERRO METODOLOGICO CORRIGIDO (nao repetir)
Affirmei antes que "byte original zero => registro vazio (problema de escrita/
selagem); byte original nao zero => leitor errado". **Isso e falso.** Um unico
byte nao distingue essas causas: zero pode ser dado legitimo, padding de header/
alinhamento ou inicializacao de buffer. Conclusao de "vazio" ou "leitor errado"
a partir de um byte isolado e invalida. O que decide e a **identidade da fonte**
(qual tensor o consumidor real leu), mais geometria, min/max/norma e amostra
materializada, comparados com uma captura do tensor que chega de fato ao
consumidor de atencao.

### Hipotese PRIORITARIA a verificar (nao presumida)
Fixture split n=1 com P=512 esta **abaixo de P**, logo o caminho de atencao
executado deve ser **local**; os records KVarN remotos podem estar legitimamente
vazios porque overflow nao esta ocupado. `consumer_ok` sozinho nao prova que o
consumidor selecionou a fonte correta, e metadata global de celulas nao prova que
cada record remoto esta vivo. Instrumentar `build_attn`/roteamento/`get_k`/
`get_v` real em PURE e SPLIT antes de qualquer mudanca de stride ou selagem.

### Estado
Isolamento: RESOLVIDO e PROVADO (A=A2=0, A=D=0). Leitor consumidor: funcional
mas devolve zeros. B/C: NAO interpretados (fixture invalida). Gate do import:
VERMELHO, sem relaxamento, NRMSE=0.0153813 > 1e-3.

## 2026-10-05 — IDENTIDADE DA FONTE REAL DO CONSUMIDOR (instrumentado, nao presumido)

`f4_trace_attention_source()` em `llama-kv-cache-kvarn.cpp`, ligado por
`F4_TRACE_SOURCE=1`, instrumenta `get_k_for_attention`/`get_v_for_attention` — o
roteamento que `build_attn` realmente usa — e registra por `il`: rota, range
position-split, nome do tensor, tipo, buft, offset, `ne`, `nb` e ocupacao de
bytes da folha. Nao e caminho de teste: e o proprio consumidor.

### MEDICAO (ramo A, n=1, P=512, C=2048)
Duas rotas coexistem no mesmo decode:
```
F4SRC get_k il=3 kv=K route=kvarn_native    range=LOCAL name=" (reshaped)"      type=1(F16) ne=[256,4,256,1]  nb1=512 nb0=2
F4SRC get_k il=3 kv=K route=standard_local  range=none  name=cache_remote_k_l3  type=2(Q4_0) buft=Vulkan0 ne=[256,4,2048,1] nb1=144 nb0=18
F4SRC get_v il=3 kv=V route=kvarn_native    range=LOCAL name=" (reshaped)"      type=1(F16) ne=[256,4,256,1]
F4SRC get_v il=3 kv=V route=standard_local  range=none  name=cache_remote_v_l3  type=2(Q4_0) buft=Vulkan0 ne=[256,4,2048,1]
```
Contagem no processo: `kvarn_native` 16 (8 camadas x K,V) e `standard_local` 48
por K e por V. Camadas de atencao são exatamente **3,7,11,15,19,23,27,31** — as
8 attention; as demais 24 sao DeltaNet recorrentes. `il=0` (usado antes pelo
probe) e **recorrente e nunca lido pela atencao**.

### POR QUE O PROBE ANTES RETORNOVA ZEROS (causa medida)
`materialized_probe` forçava o materializer sobre `l->k_records` / `l->v_records`,
que sao exatamente os records remotos `cache_remote_*` na **Vulkan0/Q4_0** — a
faixa `[P,C)`. O consumidor em n=1 le `kvarn_native range=LOCAL` (F16, anel
local) na rota split, e `standard_local` (a mesma folha remota Q4_0) na rota
legada. Entao o probe lia a **faixa remota**, enquanto o dado de n=1 esta na
**faixa local**. `consumer_ok=1` era verdadeiro e irrelevante: confirmava que o
leitor de records remotos funciona, nao que o gate media a fonte do consumidor.

### OCUPACAO DE BYTES DA FONTE REAL (bytes crus, nao valores desquantizados)
```
F4BYTES get_k il=3 kv=K name=cache_remote_k_l3 nbytes=1179648 raw_min=0 raw_max=0 raw_norm=0
```
Os **1.179.648 bytes** de `cache_remote_k_l3` estao integralmente a zero, e o
mesmo para l7..l31 de K e V. Isso e consistente com "overflow nao ocupado para
n=1 <= P", que e a hipotese prioritaria — mas nao e prova: nao distingue buffer
nao inicializado, padding ou gravacao ausente, e por isso **nao permite nenhuma
conclusao de vazio/selagem**. E evidencia de ocupacao, nao de conteudo.

### CORRECOES DE INSTRUMENTACAO FEITAS (evitar repetir erro anterior)
1. A primeira versao imprimia `min=max=norm=0` para folhas Q4_0 porque so
   desquantizava F32. Isso produz "zeros" **da propria instrumentacao**, nao do
   dado. Agora a ocupacao vem de bytes crus da folha real e e rotulada como tal;
   valores desquantizados so sao impressos quando o tipo e F32.
2. A primeira versao reportou `buft=?` porque a folha e uma view permutada sem
   buffer proprio. Agora resolve `view_src` ate o buffer real e reporta
   `Vulkan0`/`CUDA0`.

### CONCLUSAO DE FONTE (medida, por rota)
- Faixa **LOCAL** (`range=LOCAL`, F16, anel local): e onde o dado de n=1 vive.
- Faixa **remota** (`cache_remote_*`, Vulkan0/Q4_0, `[P,C)`): e o que a rota
  `standard_local` le, e esta byte-zero em toda a extensao com n=1.
- Portanto o gate B/C deve ler e mutar o tensor que o consumidor recebe na rota
  do range vigente, e reservar o probe remoto para `pos >= P` com overflow
  realmente ocupado. `metadata` global de celulas nao prova vida por record.

### Estado
A=A2=0 e A=D=0 medidos por processo. A identidade da fonte esta agora
instrumentada e medida. B/C continuam NAO interpretados: o probe antigo media a
faixa errada. Import segue vermelho (0.0153813 > 1e-3), sem relaxamento.

## 2026-10-05 — F4: CAPTURA DA FONTE REAL DO CONSUMIDOR (2a rodada)

### Instrumentacao adicionada
- `capture_from_leaf()` chamado nas **4 rotas reais** de `get_k_for_attention`/
  `get_v_for_attention` (`standard_local`, `kvarn_native`, `kvarn_materialized`,
  para K e V), chaveado por `(layer, value, head)` e com **assinatura de fonte**
  (`route|range|name|type|nbytes`), de modo que rotas coexistentes no mesmo
  decode ficam todas registradas em vez de a ultima sobrescrever a anterior.
- `capture_after_compute()` (novo virtual em `llama_memory_context_i`, hook em
  `llama-context.cpp` antes de `graph_compute_complete`) para ler tensores
  **computados no grafo**, cujo dado so existe apos o compute.
- API C: `llama_kvarn_capture_routes()`, `llama_kvarn_captured_head(...,
  index ...)`, `llama_kvarn_capture_clear()`.
- `F4_TEST_LAYER` com default **3** (primeira camada de atencao). Antes o probe
  usava `layer = 0`, que e **DeltaNet recorrente e nunca lido pela atencao** —
  ou seja, o gate media uma camada que o consumidor nao consome.

### MEDICAO (ramo A, n=1, P=512, C=2048, camada 3, pos 0)
Rotas capturadas para (3, K, head 0) — **duas, simultaneas no mesmo decode**:
```
F4CAP get_k il=3 kv=K route=kvarn_native    range=LOCAL name=" (reshaped)"     type=1(F16)  nbytes/head=512
F4CAP get_k il=3 kv=K route=standard_local  range=none  name=cache_remote_k_l3 type=2(Q4_0) nbytes/head=144  buft=Vulkan0
```
Contagem por decode: `kvarn_native` 16 e `standard_local` 48, para K e para V.
Ou seja, **8 das leituras por camada usam a faixa LOCAL split e as outras usam a
rota legada que le os records remotos**. Escolher "a ultima" seria arbitrario.

### ESTADO DE OCUPACAO DAS DUAS FONTES (bytes/valores reais, por head, pos 0)
```
standard_local  cache_remote_k_l3  Q4_0/Vulkan0  144 B/head  -> min=0 max=0 norm=0 nonzero=0
standard_local  cache_remote_v_l3  Q4_0/Vulkan0  144 B/head  -> min=0 max=0 norm=0 nonzero=0
```
A faixa **remota** esta byte-zero em toda a extensao (1.179.648 B por tensor,
medido antes) e tambem nos 144 B por head. Isso e **consistente** com a
hipotese prioritaria (n=1 <= P, overflow nao ocupado, dado esta na faixa local),
mas **nao prova** nada sobre vazio/selagem — ver a secao de erro metodologico.

### BLOQUEIO MEDIDO NESTA RODADA
A rota `kvarn_native range=LOCAL` devolve um tensor **calculado no grafo**
(`" (reshaped)"`, F16, sem `buffer` no momento do `get_k`). A leitura no
`get_k` daria zero **da instrumentacao**, nao do dado — exatamente o erro que
ja foi cometido uma vez. Por isso a leitura foi diferida para
`capture_after_compute()`. Nesta build a diferenca **nao resolveu**: o tensor
intermediario continua sem `buffer` acessivel no ponto do hook, entao
`pending_` nao foi drenado e nao houve `F4CAP post`. Nao houve leitura inventada
nem valor substituido.

### POR QUE ISTO JA E CONCLUSAO UTIL
Mediu-se que **o probe anterior e o materializador de `k_records` mediam a
faixa remota, enquanto o dado de n=1 pertence a faixa LOCAL** — e que a rota
`standard_local` ainda envia os records remotos (byte-zero) ao grafo em parte dos
builds. Antes de mexer em stride ou selagem, a pergunta agora e precisa e
verificavel: **qual das duas rotas alimenta o build de compute real**, e por que
`standard_local` nao esta caindo em `range=LOCAL` quando `position_split` esta
habilitado (`range=none` indica que `current_position_split_range()` retornou
falso, caindo no ramo legado `use_standard_route = uses_standard_layer(il)`).

### Estado
- A=A2=0 e A=D=0 medidos por processo (mantido).
- Identidade de fonte por rota, head, posicao, offset, tamanho, tipo, buft,
  min/max/norma: **instrumentada e medida**.
- `kvarn_native` (LOCAL) pos-compute: **pendente** (tensor intermediario sem
  buffer no hook).
- B/C: **NAO interpretados**. Nao afirmar igualdade por logits nem por
  autoconsistencia de leitura toda zero.
- Import: **vermelho**, `0.0153813 > 1e-3`, sem relaxamento.

## 2026-10-05 — Auditoria de recuperacao de estado (sem checkout/reset/restore)
- Branch tarefa preservada: `feat/occupancy-position-split`, HEAD `d9db0b881`,
  working tree dirty com alteracoes rastreadas e arquivos novos em
  `docs/occupancy-evidence/`, `tools/f4-branch-driver.py` e testes F4. Nenhum
  checkout, reset ou restore foi usado nesta auditoria.
- O diff atual do progressivo e aditivo (`208/0`); as secoes antigas
  proofA/proofB/proofC nao foram recuperadas do HEAD porque seriam historia
  nao commitada e potencialmente perdida. Abaixo vao somente fatos verificados
  nos logs sobreviventes; nada foi removido das secoes atuais.
- `f4-proofA-head-span.log` (16164 linhas): final
  `destination-intact: 97 checks, 9 failures`.
  - Somente `head0/k_records` teve mutacao viva detectada em n=1/128/129
    (`6.09522`, `3.26643`, `3.41443`).
  - `head1/v_records` teve diferenca zero apos mutacao viva e escrita fora do
    span vivo mudou os logits com os mesmos valores, portanto o mapeamento nao
    estava isolado.
  - Hash vivo mudou com escrita viva, mas escrita em linha inativa tambem mudou
    o hash vivo: `LEAKED (bad)`.
  - A tabela de linhas registra `kvarn:3..31 remote ... live=same` e
    `standard:3..31 local ... no-source-row`, com `rows=16 local_covered=8
    remote_covered=0 unsupported_live=8`; a rota nativa local final e F16 CUDA0
    com `body_rows=256` nas camadas 3/7/11/15/19/23/27/31.
- `f4-proofB-head-span.log` (20854 linhas): final
  `destination-intact: 190 checks, 44 failures`.
  - Somente `K/head0` teve mutacao finita (`6.09522`, `3.58987`, `2.39029`,
    `2.32899`); `K/head1`, `V/head0` e `V/head1` ficaram em zero apos escrita.
  - Restaurar o byte nao devolveu os logits (`restore_max` ate `7.68718`), com
    falhas explicitas de isolamento do probe.
  - Hash vivo detectou mutacao, mas escrita inativa tambem mudou o hash vivo:
    `LEAKED (bad)`.
- `f4-proofC-independent-branches.log` (18656 linhas): final
  `destination-intact: 115 checks, 26 failures`.
  - Em n=1, todos os ramos registraram
    `consumer before->after: K.h0=-1 ... (consumer_ok=0 after=0)`; a linha de
    gates registra `A(pos_min=0 pos_max=0 decodes=1) | A=A2 max=0 | A=D max=0`,
    mas sem consumidor nem vetor materializado valido.
  - Em n=128, `A(pos_min=127 pos_max=127 decodes=1) | A=A2 max=0 | A=D max=-1`.
  - Em n=129, `A() | A=A2 max=-1 | A=D max=-1`.
  - Hash vivo detectou mutacao e escrita inativa vazou para o hash vivo:
    `LEAKED (bad)`.
- `f4-proofC-consumer-ok.log` (15690 linhas): final
  `destination-intact: 115 checks, 35 failures`.
  - A unica linha `consumer_ok=1` sobrevivente e do ramo A, camada 0,
    `k_records`, head 0, com vetores zero antes/depois:
    `consumer before->after: K.h0=0 K.h1=0 V.h0=0 V.h1=0 (consumer_ok=1 after=1)`.
  - Na mesma janela, os gates sao `A=A2 max=-1 | A=D max=-1` e todos os B/C
    registram `max=-1`; ha falhas repetidas de inicializacao de contexto com
    `LLAMA_KV_POSITION_SPLIT_P ... got 512`.
  - Uma linha antiga do mesmo log afirma `position-split ... P=512 (C=22048)`,
    enquanto o inventario da mesma execucao registra `n_ctx=2048`; essa
    discrepancia textual nao foi resolvida e nao deve ser usada como P/C real.
- Fatos comuns de importacao repetidos nos quatro trailers:
  `1-vs-2 NRMSE=0`, `1-vs-3 NRMSE=0.0122853`, `1-vs-4 NRMSE=0.0123971`,
  `3-vs-4 NRMSE=0.0096419`, com faixa remota desocupada abaixo de P. O valor
  corrente `0.0153813` esta registrado no progressivo, mas nenhuma linha
  sobrevivente em `docs/occupancy-evidence/` contem esse numero; ele continua
  como gate vermelho a revalidar por execucao nova.
- Correcao de status: a frase antiga `Isolamento RESOLVIDO e PROVADO` fica
  **superada**. Os trailers sobreviventes mostram igualdade de logits em n=1,
  mas com consumidor ausente/invalido, vetores materializados zero/ausentes,
  vazamento de linha inativa e contextos ausentes em n=128/129. Portanto a
  cobertura continua invalida ate captura correta da fonte executada.
- Cobertura recorrente/DeltaNet e faixas acima de P: nenhum fato validado foi
  reconstruido destes quatro logs.

## 2026-10-05 — MARCO: gates A/A2/B/C/D por processo, fonte executada (escopo exato)
Fontes: `docs/occupancy-evidence/f4-gates-full-20261005.log` (n=1/127/128/129/
511/512/513, todos PASS) e `f4-gates-n513-remote-20261005.log` (detalhe remoto).
Binario: `build-optimized/bin/test-kv-mixed-destination-intact` via
`tools/f4-branch-driver.py`, um processo por ramo, sequencial, P=512, C=2048,
camada de atencao 3 (modelo), K/V heads 0/1, um proof-token por ramo.

### O que foi provado (e como)
- A=A2 e A=D zeram nos vetores K/V executados E nos logits do proof-token, em
  processos independentes, com D restaurando antes de qualquer decode.
- B (mutacao viva): alvo finito, vizinhos intactos, em cada head K/V:
  - abaixo de P (1/127/128/129/511): alvo `*.split_local` (stage F16 local),
    3 vizinhos zero; logits do proof-token mudam (exceto h1 em alguns n, onde
    a mutacao e viva no estado mas inerte naquele proof-token: registrado, nao
    escondido).
  - n=512/513 (decode dual-range, overflow ocupado em 513): idem, com 7
    vizinhos zero incluindo as 4 amostras `split_overflow`.
  - remoto em n=513 (pos-alvo 512>=P, row=512): alvo `*.split_overflow`
    (storage Q4 do companion) finito, 7 vizinhos zero, logits mudam (h0) ou
    nao (h1, inerte no token: registrado).
- C (mutacao inativa): todas as amostras zero e logits zero, local e remoto.
- Cada amostra vem de no de grafo que EXECUTOU (callback do scheduler,
  pos-compute sincronizado), com rota/range/nome/tipo/offset/shape/backend,
  identidade do store (`stored_*`), records I8, `shared_il`/`mapped` explicitos
  (3->0), FA que consumiu a folha (`fa>0`, `F4FA`), e geracao do compute
  (`gen`): amostras do prefill nunca satisfazem o select pos-decode
  (`gen` + `clear` + `captured()` rejeita obsoleto; reuse de grafo captura no
  scheduler daquele decode).
- Nenhum backend criado sob demanda; sem double-free (holder libera); driver
  sem `0.0 or -1` (None domina); comparacoes pos-decode x pos-decode (o decode
  armazena o proof-token: comparar prefill x decode seria falso-positivo).

### Correcoes de identidade/mutacao medidas nesta rodada
- `cache_remote_k_l*` (Q4_0/Vulkan0, `name_tag "remote_" do companion
  standard) NAO sao os `cache_kvarn_k_records_*` (I8): docs anteriores
  confundiam; corrigido no codigo e nos gates.
- `layers[i]` nao e camada modelo `i` (recorrentes puladas): probes agora usam
  `layer_for()`/`map_layer_ids`; `layer=0` do teste antigo media storage
  errado.
- Mutacao endereca slices fisicas (`head*slices`): h1 logico caia no slice de
  h0 antes do fix; restore acompanha.
- Geometria do materializer via `llama_kvarn_geometry_for`, sem heuristica
  64/128.
- Dump de igualdade fatia o store com head_dim do consumidor e slots do store
  (view declara mais slots que o store em dual-range: limites do store valem).

### Escopo EXPLICITO (nao chamar F4 verde)
- Uma camada de atencao (modelo 3), heads 0/1 de 4; demais camadas nao cobertas.
- Remoto: somente n=513, pos 512, 1 token de overflow; em n=512 o remoto e
  SKIP correto (prefill nao ocupa overflow: nada a mutar antes do decode).
- DeltaNet/recorrencia: sem B/C; so execucao observada via F4FA (24 camadas).
- Import PURE->SPLIT e NRMSE: ver secao propria; `.0153813` e historico lido
  em linha FAIL anterior, sem trailer sobrevivente: sujeito a revalidacao,
  nao apagado nem reafirmado.

## 2026-10-05 — Camada 11 + divergencia do import (estado medido)
Fontes: `docs/occupancy-evidence/f4-gates-layer11-n1-20261005.log`,
`docs/occupancy-evidence/f4-four-routes-20261005.log` (gerado nesta rodada,
`destination-intact: 115 checks, 31 failures`).

### Camada 11 (n=1): mesmos gates, PASS
`F4_DRIVER_LAYER=11`: A=A2=0, A=D=0 com restore, B alvo finito e 3 vizinhos
zero nos 4 heads, C zero com logits zero. Deltas diferentes da camada 3
(K.h0 0.439 etc.), como esperado de outra camada; logits B != 0 nos 4 heads.
A fixture nao e especifica da camada 3. Escopo de camadas continua parcial
(2 de 8 atencao; DeltaNet sem B/C: estado recorrente nao tem consumidor de
atencao — a comparacao aplicavel e hash de estado vivo por camada, ja feita
na tabela de linhas para as 32 camadas).

### Import PURE->SPLIT: NRMSE revalidado + primeira divergencia localizada
- `import vs SAME-representation reference: 248320 entries, NRMSE=0.0153813,
  max_abs_diff=0.15323` (FAIL linha 1021, log novo). `.0153813` e fato
  corrente medido, nao so historico.
- Comparacao: SPLIT prefill(192) + install M via data-path (ref) vs SPLIT
  prefill(192) + restore PURE->SPLIT (imp); mesmo tokens, mesma posicao de
  proof. Nao e mismatch de comprimento.
- Blob: `ref=imp=70358211 bytes`; primeira divergencia no offset 390580
  (0.5551%), 42697193 bytes divergentes (60%). Janela:
  `ref=...15a6314aed4180c2...` vs `imp=...ef24aae955868c92...` — campo de
  8 bytes com cara de hash/checksum, flanqueado por bytes identicos.
- Tabela por camada: 8 `kvarn:* remote live=same` (hashes live identicos),
  8 `standard:* local covered`, `matched keys 8/16`, `no differing layer
  field found at covered layers`.
- Conclusao honesta: payloads live por camada sao identicos, mas os logits
  divergem — logo a causa esta fora do payload live coberto (suspeitos:
  tail/exact-tail, regioes lidas mas fora do hash live, posicoes, ou celulas
  com n_kv efetivo distinto no decode). O proximo experimento e mapear o
  offset 390580 a secao do writer (instrumentar o mapa de secoes do
  state_write) em vez de adivinhar; tail-vs-body e a primeira secao a checar.
  Gate segue vermelho, sem relaxamento.

## 2026-10-05 — RETRATACOES: duas extrapolacoes do relatorio anterior
Aplicadas por ordem do plano. O que segue CORRIGE a secao anterior; as linhas
antigas ficam por apendice (sem checkout/restore).

### (1) "8 hashes kvarn iguais" NAO estabelece "causa fora do payload live coberto"
Retratacao integral. A igualdade de oito hashes por si so NAO demonstra que
- o hash cobre os spans de consumidor que foram validados (mutacao A/A2/D e
 cobertura de head/celula), nem
- que cobre todos os campos, celulas e heads pertinentes.
Sao 8 chaves de uma tabela cuja formula/semantica ja falhou uma vez; a mesma
tabela nao pode ser reutilizada como autoridade sem re-prova de span.
Alem disso, as 8 linhas `standard:* local` SEM correspondencia na fonte PURE
continuam LACUNA: nao foram explicadas nem contabilizadas como cobertas.
Ate prova nova, o unico enunciado correto e: "oito hashes coincidem; a
correspondencia deles com os spans consumidos e a cobertura completa estao
PENDENTES de prova".
Prova exigida: hash por span que o consumidor le, com perturbacao isolada e
restaurada alterando o hash E a saida; e contagem de cobertura por
(field, layer, head, cell) para as 16 chaves da tabela.

### (2) A tabela de 16 linhas NAO e a tabela das 32 camadas
Retratacao integral. `layer rows: pure=8 imported=16 split=16` descreve
8 camadas de ATENCAO x 2 faixas (kvarn local + standard overflow) = 16 linhas.
Nao sao 32 camadas e nao cobre os 24 estados DeltaNet.
Registro agora deve ser POR LAYER REAL E TIPO:
- layer real 3,7,11,15,19,23,27,31 = atencao, faixas kvarn/standard;
- layer real demais 24 = DeltaNet recorrente, que hoje NAO tem export proprio
 nesta tabela => PENDENTE (marcado como tal, nao como coberto).
Um hash sem prova de span/comportamento nao vale para DeltaNet.

### Instrumentacao de secoes (novo, em construcao nesta rodada)
- `llama_kv_mixed::stream_describe` caminha o frame REAL e emite
 {start,end,kind,field,layer,stream,count,stride,checksum}; relata o que vem
 depois do frame como `trailing/post_frame_state` (estado recorrente), em vez
 de atribuir ao frame.
- `stream_options::on_section` registra as mesmas regioes no LADO DO ESCRITOR,
 durante a emissao; captura sob `F4_STATE_SECTIONS`.
- Exposto por `llama_mixed_state_section_map`, `llama_mixed_state_locate`,
 `llama_mixed_state_write_section_map`, `llama_mixed_state_section_diff`.
- Meta: localizar o offset 390580 pelo schema real; se for checksum, achar a
 primeira divergencia do PAYLOAD coberto por ele (checksum nao e causa).

## 20261005 — Localizacao semantica: atencao identica, estado recorrente DeltaNet divergente

Evidencia: `docs/occupancy-evidence/f4-deltanet-recurrent-20261005.log` (24363 linhas).
Comando (GPU livre, producao parada):

```bash
F4_STATE_SECTIONS=1 F4_TEST_GPU=1 F4_TEST_SPOOL=/home/hjotha/f4-spool \
  ./build-optimized/bin/test-kv-mixed-destination-intact
# destination-intact: 121 checks, 17 failures
```

Freeze: tres contextos em `prefill(n)` para `n in {1,192,256}` — `ctrl_h`
(split, referencia), `pure_h` (não-split), `dest_h` (split, importado a partir
de `pure_h`) — capturados antes de qualquer token de prova. `n=192` é o ponto
de freeze do gate de logits.

### O que foi medido

1. **Leitura semantica do consumidor de atencao (materializada).**
   `llama_kvarn_materialized_probe` em toda camada de atencao x todo head
   logico x toda posicao x K/V, nos tres contextos:

   ```
   semantic materialized read: comps=24576 heads_probed=8 PURE-vs-IMPORTED differing comps=0
   FIRST SEMANTIC DIFFERENCE: none - all materialized K/V agree across PURE, SPLIT and IMPORTED
   ```
   (24576 = 8 camadas x 8 heads x 192 posicoes x 2 campos.) K/V materializados
   **bit-identicos** entre PURE, SPLIT e IMPORTED.

2. **Rabo recorrente normalizado (mesmo escritor dos dois lados).**
   A memoria hibrida grava atencao primeiro e o estado recorrente depois
   (`llama_memory_hybrid::state_write`), e os dois lados usam o mesmo
   `llama_memory_recurrent::state_write`, entao caudas de mesmo tamanho sao
   uma comparacao DENTRO de uma unica representacao. O rabo tem tamanho
   fixo 52691540 B em n=1, 192 e 256 (confirmado pelo instrumento
   `[recr] state_write seq=0 cell_count=1 size=1 n_rs_seq=0 data_bytes=52691528`
   + 4 B de `cell_count` + 8 B de meta = 52691540).

   ```
   recurrent-tail: split=[17666671,70358211) bytes=52691540 blob_ref=70358211 blob_pure=67650703 blob_imp=70358211
   recurrent-tail vs PURE    : bytes=52691540 ndiff=34139540 first=294980 abs=17961651
   recurrent-tail vs IMPORTED: bytes=52691540 ndiff=34139540 first=294980 abs=17961651
   recurrent-tail PURE vs IMPORTED: bytes=52691540 ndiff=0 first=none
   ```
   Em n=1, 192 e 256: IMPORTED ≡ PURE (ndiff=0), SPLIT-reference ≡ ambos
   em 32,9–34,1 M de 52,7 M bytes. O offset primeiro (294980) cai exatamente
   no **primeiro byte da linha `r` da camada 4** do layout de
   `state_write_data` (8 B de cabecalho `s_trans`/`n_layer` + 3 entradas de
   12 B de cabecalho + 98304 B de linha = 294960; 294980 − 12 = 294968 →
   entra na 4a entrada na fronteira da linha).

3. **Export/mapa proprio do DeltaNet por camada de modelo.**
   `llama_mixed_state_section_diff` nao serve aqui (o rabo nao tem schema de
   frame); implementado `llama_recurrent_state_diff(a, b, seq)` que le
   `r_l`/`s_l`/`p_l` vivos (24 camadas DeltaNet; `r_l` indexado por camada de
   modelo, nulo nas 8 de atencao). Log por camada:

   ```
   recurrent-diff: seq=0 cells_a=1 cells_b=1 n_rs_seq_a=0 n_rs_seq_b=0 layer_slots_a=32 layer_slots_b=32
   layer=4 r bytes=98304 ndiff=68934 first=0 max_abs=0.034555 shape=24576x1x1 type=f32
   layer=4 s bytes=2097152 ndiff=1515187 first=0 max_abs=0.001947 shape=524288x1x1 type=f32
   ...
   layer=28 r bytes=98304 ndiff=69975 first=0 max_abs=0.132095 shape=24576x1x1 type=f32
   recurrent-diff: compared_tensors=72 diverging=42 worst=layer28.r max_abs=0.132095
   ```
   PURE vs SPLIT-reference: **42 de 72** tensores divergentes = `r` e `s` de
   **21 das 24** camadas DeltaNet. Camadas 0, 1 e 2 identicas; a primeira
   divergencia e a camada 4 (`first=0`). `r` cresce de 0,0346 (camada 4) a
   0,132095 (camada 28); `s` fica entre 0,0018 e 0.0177.
   SPLIT-reference vs IMPORTED: **os mesmos 42 tensores, mesmos valores** —
   IMPORTED ≡ PURE no recorrente.

4. **CONTROLE (mesma configuracao, duas contexts independentes).**
   `ctrl_h` e `dest_h` sao os dois split com o mesmo `P` e recebem o mesmo
   `prefill`; o controle foi lido **antes** de `dest_h` ser sobrescrito pelo
   import:

   ```
   --- CONTROL: SPLIT ctrl_h vs SPLIT dest_h, both clean prefill(192) ---
   recurrent-diff: compared_tensors=72 diverging=0 worst=layer-1.- max_abs=0.000000
   CONTROL attention K layer=3 head=0 pos=0: IDENTICAL max_abs=0
   ```
   Igual em n=1 e n=256 (diverging=0 nos tres). Logo o prefill e
   **deterministico**; a divergencia PURE-vs-SPLIT nao e ruido.

### O que isso establish (e o que nao establish)

- Estabelecido: o gate vermelho nao vem de K/V de atencao (identicos em 24576
  comparacoes) nem de um import infiel — o rabo recorrente do IMPORTED e
  byte-identico ao da fonte PURE (`ndiff=0` em n=1, 192 e 256).
- Estabelecido: `prefill` na configuracao split produz estado recorrente
  DeltaNet **diferente** do `prefill` nao-split para os mesmos tokens, com
  diferenca real (`max_abs=0.132095` em `layer28.r`), enquanto o controle de
  mesma-configuracao fecha em `diverging=0`.
- NAO establish (ainda medido): a primeira diferenca DE ATIVACAO. O que se
  sabe e que camadas 0–2 estao identicas, K/V da camada 3 estao identicos, e
  o estado da camada 4 ja diverge — ou seja, a divergencia entra **em ou antes
  da atualizacao recorrente da camada 4** (saida da camada 3 / handoff 3→4).
  Capturar a saida da camada 3 e o passo seguinte; nao afirmar causa antes.
- NAO establish: que `NRMSE=0.0153813` desapareça ao corrigir o recorrente —
  e a hipotese alinhada com as medicoes, nao um resultado.

Instrumentos novos nesta rodada: `llama_mixed_state_post_frame`,
`llama_recurrent_state_diff`, log `[recr] state_write ...` sob
`F4_STATE_SECTIONS`, bloco de leitura semantica e de rabo no teste.
Gate continua vermelho: `full-logits NRMSE=0.0153813 exceeds 1e-3`.

## 20261005 — Primeira diferenca de ativacao medida: saida da camada 3

Evidencia: `docs/occupancy-evidence/f4-activation-tap-20261005.log` (tap simples)
e `docs/occupancy-evidence/f4-activation-bisect-20261005.log` (bisect de
configuracao). Comando igual ao anterior
(`F4_STATE_SECTIONS=1 F4_TEST_GPU=1 F4_TEST_SPOOL=/home/hjotha/f4-spool
./build-optimized/bin/test-kv-mixed-destination-intact`), 121 checks / 28 e 22
failures respectivamente.

### Instrumento

Tap cru de saida por camada do decoder via primitiva compartilhada
`llama_set_capture_layers` / `llama_get_embeddings_capture_ith`
(`t_h_capture`, largura `n_capture * n_embd`, `[layer0 | layer1 | ...]`),
em dois contextos FRESH que fazem o MESMO `prefill(192)`: um split
(com `remote_attn_host` + `remote_attn_n_layers=8` + `P`) e um puro.

### Resultado

```
activation tap: n_capture=9/9 n_embd=2560 buffer=present/present
activation tap rows=1 of 192, overall max_abs=0.0156049728
layer-out[0] tap=0: same rows_differing=0/1 max_abs=0
layer-out[1] tap=1: same rows_differing=0/1 max_abs=0
layer-out[2] tap=2: same rows_differing=0/1 max_abs=0
layer-out[3] tap=3: DIFFER rows_differing=1/1 max_abs=0.00137805939 first_pos=0
layer-out[4] tap=4: DIFFER rows_differing=1/1 max_abs=0.00280575454 first_pos=0
layer-out[5] tap=5: DIFFER rows_differing=1/1 max_abs=0.0038190484  first_pos=0
layer-out[6] tap=6: DIFFER rows_differing=1/1 max_abs=0.0054737255  first_pos=0
layer-out[7] tap=7: DIFFER rows_differing=1/1 max_abs=0.00587519631 first_pos=0
layer-out[8] tap=8: DIFFER rows_differing=1/1 max_abs=0.0156049728  first_pos=0
```

So **a PRIMEIRA diferenca de ativacao e a saida da camada 3** (primeira camada
de atencao): camadas 0-2 (DeltaNet) identicas, camada 3 ja difere em
`max_abs=0.00137805939`, e a diferenca cresce ate 0.0156 na camada 8 — coerente
com o recorrente que comeca a divergir na camada 4 (42/72 tensores,
`worst=layer28.r max_abs=0.132095`).

Limites medidos, declarados: o buffer de capture tem **1 linha de saida** para o
prefill (`rows=1 of 192`), entao a comparacao de ativacao e sobre a unica linha
de saida do prefill — nao sobre as 192 posicoes. `first_pos=0` e o indice de
linha disponivel, nao a posicao 0 do prompt.

### Bisect de configuracao: os dois ingredientes estao acoplados

- `split com P mas SEM remote` → contexto **nao e criado** (`remoteOnly`/
  `c=0` em execucao anterior): a colocacao position-split exige o KV Q4 remoto
  (fail-closed, ja documentado em `make_ctx`).
- `remote com P=0` → contexto **nao e criado** (`remoteOnly=0` nesta rodada).
- Portanto nao da para separar "position split" de "remote Q4" por montagem de
  contexto: os dois ingredientes da configuracao split estao acoplados nesta
  fixture. O bisect por construcao esta fechado; o proximo corte tem de ser
  dentro da camada 3 (qual operador do caminho de atencao diverge), nao por
  remontar o contexto.

### Continuidade

- NAO establish: que a diferenca da camada 3 venha do Q4 remoto, do proprio
  position-split, ou de outro operador do caminho de atencao — apenas que a
  entrada da camada 3 (saida das camadas 0-2) e os K/V materializados da
  camada 3 sao identicos e a SAIDA da camada 3 nao e.
- Gate continua vermelho: `full-logits NRMSE=0.0153813 exceeds 1e-3`.

## 20261005 — Retracoes/correcoes de escopo das duas secoes acima

Aplicadas sobre as mesmas evidencias ja arquivadas; nenhum log foi alterado.

1. **Rabo recorrente IMPORTED ≡ PURE prova fidelidade DE UM COMPONENTE, nao do
   import inteiro.** `ndiff=0` em n=1/192/256 vale para a regiao recorrente
   (52691540 B) desses fixtures especificos. Nao cobre: o frame de atencao (que
   diverge em 18/37 secoes), cells, metadados, `state_read`/`state_write` de
   outros tamanhos de janela, nem qualquer caminho de import fora destes tres
   pontos de freeze. Formula correta: "a regiao recorrente do IMPORTED e
   byte-identica a da fonte PURE nestes fixtures" — nunca "o import e fiel".

2. **O tap de ativacao cobre UMA linha de saida, nao 192 posicoes.**
   `activation tap rows=1 of 192`: o buffer de capture exposto por
   `llama_get_embeddings_capture_ith` so tem 1 linha valida apos `prefill(192)`.
   `rows=1` e `first_pos=0` sao indice de LINHA disponivel; **nao** e a posicao 0
   do prompt. Portanto a frase "a primeira diferenca de ativacao e a saida da
   camada 3" esta CORRETA APENAS COMO "a saida da camada 3 difere na unica linha
   de saida disponivel apos prefill(192), enquanto as linhas das camadas 0-2 sao
   iguais nessa mesma linha". **NAO** e uma prova de primeira divergencia ao
   longo das 192 posicoes: as outras 191 posicoes nao foram lidas neste tap.
   Nenhuma afirmacao de completude sobre as 192 posicoes e sustentada por esta
   evidencia.

3. **Cobertura exata desta rodada** (para nao ser re-lida como mais amplo):
   - `f4-deltanet-recurrent-20261005.log`: 24576 comparacoes materializadas
     (8 camadas de atencao x 8 heads x 192 posicoes x K/V) — cobre as 192
     posicoes **para K/V materializados**; 72 tensores recorrentes x 2 pares
     (control nao impresso por camada, so resumo); rabo por byte em 3 valores de
     n; secoes do frame por 2 pares x 3 valores de n.
   - `f4-activation-tap-20261005.log`: 9 camadas x **1 linha de saida**.
   - `f4-activation-bisect-20261005.log`: idem, 1 linha, e o bisect que fechou.
   - Control A/A2 recorrente: 72 tensores, resumo `diverging=0` em n=1/192/256,
     **sem** dump por camada (so a contagem).

4. **O que continua medido e solido**: (a) prefill deterministico sob a mesma
   configuracao (controle `diverging=0`); (b) 42/72 tensores recorrentes
   divergem entre configuracoes, comecando na camada 4 e `max_abs=0.132095`;
   (c) a linha de saida disponivel difere pela primeira vez na camada 3.
   O que **nao** esta medido: a causa dentro da camada 3, o comportamento das
   demais 191 posicoes no tap, e a neutralidade do branch de overflow vazio
   (n=192<P=512) em LSE/saida/merge.

## 20261005 — Bisect de operadores DENTRO da camada 3 de atencao

Evidencia (imutavel):
`docs/occupancy-evidence/f4-layer3-node-dump-20261005.log` (22272 linhas NODE +
3648 FASRC), `f4-layer3-bisect-run-20261005.log` (execucao),
`f4-layer3-mask-fa-values-20261005.log` (valores verbatim),
`f4-node-dump-compare.py` (comparador). Nenhum LOG anterior foi alterado.

### Instrumento (novo) e prova de inercia

`F4_NODE_DUMP=<arquivo>` + `llama_f4_set_dump_tag()`. O dump percorre o grafo
PRONTO apos um `graph_compute` bem-sucedido e copia, dos buffers reais do
scheduler: (a) TENSORES DE ENTRADA (leafs) e (b) resultados intermedios, estes
pelo **eval callback do scheduler** (`f4_sched_eval_cb`), porque o scheduler
legitimamente reutiliza a regiao de um buffer assim que a vida util do produtor
termina — uma copia pos-computo de um no intermedio devolveria o ultimo
escritor daquela regiao. O dump **nao** pede splits nem materializacao que
nao existiriam: `f4_dump_wants()` so devolve `true` para nos ja elegiveis.

Inercia provada com o gate completo em TODAS as variante de dump (nao ha
relaxamento de tolerancia):

| execucao | F4_NODE_DUMP | full-logits NRMSE |
|---|---|---|
| baseline (antes do dump) | off | `0.0153813` max_abs `0.15323` |
| dumpA/B/C/D/E/F/G/H/I/J | on (+ FA_SRC, + FULL) | `0.0153813` max_abs `0.15323` |
| final3 | on | `0.0153813` max_abs `0.15323` |

Nenhum valor mudou. Alem disso o controle **A/A2 da propria instrumentacao**
(`split_a1` vs `split_a2`, duas contexts split identicas): `only1=0 only2=0
differing=0` sobre 6912 linhas — instrumento deterministico.

### Cobertura exata

- `prefill()` deste fixture faz `llama_decode` de **UM token por chamada**
  (192 chamadas). Logo o indice `eval` e a POSICAO do token: `eval=1` e a
  posicao 0, `eval=192` a posicao 191. Foram capturadas **as 192 posicoes**,
  nao filtragem por linha de embedding: 192 grafos x nos nomeados.
- Camadas 2 e 3 (`F4_NODE_DUMP_LAYERS=2,3`), papeis: `l_out, attn_norm,
  Qcur_full, Kproj, Vproj, Qcur_reshaped, Qcur_normed, Kcur_normed,
  gate_reshaped, Qcur, Kcur, Vcur, kq_mask, inp_pos, fa_kv_k, fa_kv_v, fa_out,
  attn_pregate, gate_sigmoid, attn_gated, attn_output, attn_residual,
  attn_post_norm, ffn_out, post_ffn`.
- **Contagem**: `split_a1=6912`, `split_a2=6912`, `pure=8448` linhas; chaves
  ambiguas=0 (o desambiguador por `op|ne|nb|bytes` resolveu a heranca de nomes
  dos views).
- Instrumentos escritos nesta rodada: `f4_dump_*`/`f4_dump_graph`/
  `f4_dump_tensor`/`f4_hash_span` em `llama-context.cpp`,
  `llama_f4_set_dump_tag`, `ggml_graph_n_leafs`/`ggml_graph_leaf`
  (declarados localmente em `llama-context.cpp`, **fora** de `ggml.h`, para nao
  invalidar os templates CUDA), `cb(Kproj/Vproj/inp_pos/kq_mask/fa_kv_k/
  fa_kv_v/fa_out)` em `qwen35.cpp`/`llama-graph.cpp`.

### O que e IGUAL (byte a byte, nas 192 posicoes, entre SPLIT e PURE)

`l_out-2`, `attn_norm-3`, `Qcur_full-3`, `Kproj-3`, `Vproj-3`,
`Qcur_normed-3`, `Kcur_normed-3`, `gate_reshaped-3`, `Qcur-3` (apos RoPE),
`Kcur-3` (apos RoPE), `Vcur-3`, `inp_pos` (posicoes M-RoPE),
e o proprio **operand Q da FA** (FASRC idx=0, `2bbb4905005030c3` identico nas
duas tags split e na pure).

### O que DIVERGE — primeiro operador

Ordem canonica de operadores, primeira divergencia:

1. `kq_mask` (entrada) — diverge em **192/192** posicoes.
2. `fa_kv_k-3` / `fa_kv_v-3` (operandos K/V da FA) — divergem em byte, mas
   **nao e comparacao alinhada**: o store de records tem 286720 B no SPLIT e
   1146880 B na PURE (4x = C 2048 vs P 512 celulas), entao os spans hasheados
   (286720 vs 524288) nao cobrem o mesmo trecho. Igualdade semantica dos K/V
   materializados ja estava provada por `llama_kvarn_materialized_probe`
   (24576 comparacoes, 0 divergencias).
3. **`fa_out-3` (FLASH_ATTN_EXT)** — diverge em **192/192**; em `eval=1`
   `max_abs=0.00161505`, `mean_abs=0.0001509`, `cosine=0.99999995`, A/A2 exato
   0. Parametros da FA identicos nas tres tags:
   `fa=1031798784,0,0,10,1` (scale 0.0625, alibi 0, softcap 0, prec F32,
   dominio KVarN=1) e `backend=CUDA0` nos tres.

### A diferenca estrutural decisiva (FASRC)

Os operandos vivos da FA, lidos no proprio instante em que a FA executa:

```
idx=0  Q                      f32 16384  split=2bbb4905005030c3  pure=2bbb4905005030c3  IDENTICO
idx=1  fa_kv_k-3              f16        split=e97846eb7709a325  pure=fc31bff590c22325  DIFERE
idx=2  fa_kv_v-3              f16        split=e97846eb7709a325  pure=fc31bff590c22325  DIFERE
idx=3  CUDA0#kq_mask#0        f16 512    split=ae34a5c0c86fccd1  pure=ad52c5f01f4cb325  DIFERE
idx=5  (K tail)               f16 264192 split=-                 pure=8c6ef74b657ec325  SO NA PURE
idx=6  (V tail)               f16 264192 split=-                 pure=8c6ef74b657ec325  SO NA PURE
idx=7  CUDA0#leaf_68#0        258 B      split=-                 pure=d855627b8c99e7ed  SO NA PURE
idx=8  CUDA0#leaf_69#0        i32 4 B    split=-                 pure=4d25767f9dce13f5  SO NA PURE
idx=9  CUDA0#leaf_70#0        i32 540 B  split=-                 pure=b242d44dafddf7c1  SO NA PURE
idx=10 (ordem de consulta)    f16 2048    split=-                 pure=3935e2da838ec857  SO NA PURE
idx=11 (descritor)            f16 2048    split=-                 pure=505ef8c9cf736c65  SO NA PURE
```

**O no FlashAttention da camada 3 e COMPOSTO de modo diferente**: a PURE anexa
um KV tail (7 operandos extra: K tail, V tail, mascara do tail, ordem de
consulta e descritor); o SPLIT nao anexa nenhum. A mascara do SPLIT e a linha
causal correta (`0,0,...,-inf`), a da PURE e inteiramente `-inf` no corpo — o
que e coerente com as chaves estarem no tail na PURE. A composicao e decidida
na construcao (`tail_route = mctx_cur->get_tail_route(il)`,
`has_compact_tail(il)`), i.e. e uma diferenca de MONTAGEM do grafo, nao de
aritmetica.

### NAO establish (declarado, nao extrapolado)

- Nao foi medido se a ausencia do tail no SPLIT e a causa necessaria e
  suficiente do `NRMSE=0.0153813` — so que e a primeira diferenca estrutural
  medida dentro da camada 3, montante de todas as demais.
- `fa_kv_*` nao tem comparacao byte-a-byte alinhada (store 4x); a igualdade
  alegada sobre K/V e a ja provada por materializacao.
- A mascara da PURE foi lida no leaf `kq_mask` E no operando real
  `CUDA0#kq_mask#0` (hashes iguais entre si), mas **nao** foi imposta a
  hipotese de que `-inf` total invalida a FA da PURE: `fa_out` da PURE e
  nao-nulo e coerente (`|norm|=42.49`), entao a leitura e consistente com a
  mascara do corpo estar vazia porque o tail carrega as chaves.
- Fix do gate: **ainda nao aplicado**; `full-logits NRMSE=0.0153813` segue
  vermelho em toda execucao desta rodada.

### Fixture legado: `max=-1` agora falha explicitamente

`logits_diff()` devolve **negativo** quando nao ha vetor comparavel. Antes isso
era impresso como "A and A2 differ (max=-1)", "a LIVE mutation produced
identical logits (max=-1)" e "an INACTIVE mutation changed the logits
(max=-1)" — leituras de medicoes que nao foram feitas. Agora:

- mensagem propria `branch X is UNAVAILABLE (no logits); this is missing
  context/consumer data, not an A/A2 result` com `decodes=%d consumer_ok=%d
  pos_max`;
- `CHECK(dAA2 >= 0)`, `CHECK(dAD >= 0)`, `CHECK(b_ok && c_ok)` — "NOT
  COMPARABLE" — e `continue` **antes** de qualquer interpretacao B/C;
- `full_logits_after_one_token` loga `F4NODATA decode_failed rc=%d ctx=%...`
  para separar falha de criacao de contexto de falha do consumidor.

Os CHECKs originais (`a branch produced no logits`, `A and A2 differ`, `A=D`,
`dAB>0`, `dAC==0`, prova de mutacao inativa) **continuam existindo**; nada foi
removido.

### Diagnose da instabilidade de contagem (context creation)

`make_ctx(P, split=1, remote=1)` (position-split + companion Q4 + Vulkan local)
e a alocacao MAIOR do fixture. Criada junto com as tres contexts base ela
falhava de forma intermitente — evidencia literal das execucoes:

```
dumpH: split_a1=ok  split_a2=ok  pure=ok     -> 118 checks
dumpI: split_a1=FAIL split_a2=FAIL pure=ok   -> 118 checks (linhas de split ausentes)
dumpJ: split_a1=ok  split_a2=FAIL pure=ok    -> 110 checks, referencia nao capturada (1008/938)
final2: split_a1=ok split_a2=FAIL pure=ok    -> 139 checks
final3: split_a1=ok split_a2=ok  pure=ok     -> 139 checks (fase movida para o INICIO, sozinha)
```

Correcao: a fase de dump foi movida para **antes** da criacao de
`dest_h/ctrl_h/pure_h`, com um contexto vivo por vez. Em `final3` as tres
contexts nasceram (`dump[split_a1/split_a2/pure]: prefill=1 pos_max=191`) e a
contagem estabilizou em `139 checks / 14 failures`.

### Estado do gate nesta rodada

- `full-logits NRMSE=0.0153813 exceeds 1e-3 (max_abs_diff=0.15323)` — VERMELHO.
- Gates A/A2/D com dados disponiveis: `A=A2 max=0`, `A=D max=0` para
  n=1/128/129 (deterministico).
- 14 falhas: o gate acima + 12 x `a LIVE mutation produced identical logits
  (max=0)` (agora com `max=0` real, nao `-1`) + 1 x
  `mutation proof: an INACTIVE row changed the live hash`.

### Correcao de escopo sobre "o SPLIT nao tem tail"

Medido depois: as linhas `KV tail: group=full layer=3 ... effective=128` e
`KV tail final: layer=3 ... supported=1` aparecem **tambem** durante o prefill
do SPLIT (posicoes 1319/1498 do log, dentro das execucoes de `split_a1` e
`split_a2`, antes de qualquer contexto puro). Portanto o que a evidencia FASRC
sustenta e mais estreito do que "o SPLIT nao possui tail":

- **medido**: o no `FLASH_ATTN_EXT` da camada 3 do SPLIT tem 4 operandos
  (Q, K, V, mascara) e o da PURE tem 11 (os mesmos 4 + 7 operandos de tail);
- **medido**: as linhas de validacao do tail existem nas duas configuracoes;
- **NAO medido**: se o tail do SPLIT esta ausente do no, anexado a outro no,
  ou desativado em P-mode. Essa e a fronteira da investigacao; enquanto nao
  for medida, a frase correta e "a composicao do no FlashAttention difere",
  nao "o SPLIT nao tem tail".

Fronteira de decisao para o passo seguinte: corrigir a montagem (fazer as duas
prefills produzirem a mesma composicao de atencao) altera semantica de cache e
nao deve ser feito sem confirmacao; trocar a fonte de `m_mixed` para a PURE
mudaria o SIGNIFICADO do gate. Nenhuma das duas foi feita nesta rodada e
`NRMSE=0.0153813` segue vermelho.

## 20261005 — Causa minima medida: o tail e desligado pelo position-split

Evidencia: `docs/occupancy-evidence/f4-tail-trace-20261005.log` (diagnostico
`F4TAIL`), `f4-layer3-operator-census-20261005.log` + `f4-layer3-census-run-20261005.log`
(censo de operadores, `F4_NODE_DUMP_OPMAP=2`).

### Censo de operadores (eval=1, mesmos tokens, mesma posicao do grafo)

| tag | nos do grafo |
|---|---|
| split_a1 | 1791 |
| split_a2 | 1791 |
| pure | **1967** (+176) |

So na PURE (e **zero** no SPLIT):

```
SET_ROWS 1024|129  x16   cache_kvarn_k_tail_l{3,7,11,15,19,23,27,31} + v_tail
CONT/CPY 256|4     x16+x16
KVARN_WHT 256|4|129 x16   (129 = 128 tail + 1)
PERMUTE 256|129|4  x16    VIEW 256|4|129 x32
RESHAPE 1024|1     x16    SET_ROWS 1024|129 x16
```
`TAIL-related nodes: split=0, pure=48`.

### O que decide isso (`F4TAIL`, camada 3)

```
split_p=512 : compact=0 route=0 k_tail=0 v_tail=0 gather=0 tail_read_idxs=0 kv_tail_tokens=128
split_p=0   : compact=1 route=1 k_tail=1 v_tail=1 gather=1 tail_read_idxs=1 kv_tail_tokens=128
```

**`kv_tail_tokens=128` nas duas configuracoes.** O tail esta solicitado e
configurado igual. A unica variavel que muda e `kv_position_split_p`: com
`split_p != 0`, `has_compact_tail()` devolve falso, o que zera `tail_route`
(NONE), `k_tail`, `v_tail`, `gather` e `tail_read_idxs` — ou seja, **o tail de
128 tokens nao e construido no grafo split**, enquanto o mesmo `kv_tail_tokens`
com `split_p == 0` o constroi.

### Cadeia de consequencia ja medida nesta campanha

`has_compact_tail()==0 (P-mode)` → sem tail na camada 3 → no FlashAttention
com 4 operandos em vez de 11 → mascara do corpo vazia / operandos K-V com
span 4x diferente → `fa_out-3` difere em 192/192 (`max_abs=0.00161505` em
eval=1) → saida da camada 3 difere → estado recorrente DeltaNet difere da
camada 4 em diante (42/72 tensores, `max_abs=0.132095`) → `full-logits
NRMSE=0.0153813`.

Tudo a montante da camada 3 permanece **igual byte a byte** (camada de entrada,
`attn_norm`, projecoes Q/K/V, QK-norm, RoPE, posicoes, e o proprio operand Q da
FA).

### Por que isto nao foi corrigido nesta rodada (decisao, nao negligencia)

Existe conflito direto entre duas regras que nao posso resolver sozinho:

- `build_attn_position_split` declara **em comentario de projeto**:
  "Stores (range-routed by cpy_k/cpy_v; **tails disabled in P-mode**)" e
  "No sinks/bias/softcap in P-mode (fail closed below)" — desativar o tail em
  P-mode e intencional;
- o gate exige que uma prefill split e uma prefill pura dos mesmos tokens
  produzam o mesmo estado (`m_mixed` vem de `refmix_h = make_ctx(P, split=true)`
  e a fonte do import vem de `pure_h`).

Habilitar o tail sob P-mode muda semantica de cache e toca guardas
fail-closed (proibido enfraquecer); trocar a fonte de `m_mixed` muda o
significado do gate (tambem nao autorizado). **Nenhuma das duas foi feita.**
`NRMSE=0.0153813` segue vermelho; a ultima medicao valida do gate e a de
`f4-layer3-bisect-run-20261005.log` / `f4-layer3-census-run-20261005.log`
(`139 checks / 14 failures`, `A=A2 max=0`, `A=D max=0`).

### Instabilidade restante (diagnosticada, nao interpretada)

Nesta ultima execucao o contexto `ctrl` nao nasceu:

```
destination-intact: context creation failed (dest=1 ctrl=0 pure=1)
llama_init_from_model: LLAMA_KV_POSITION_SPLIT_P must be a multiple of 256 with
0 < P < n_ctx, got 512
```
ou seja, a validacao de P falhou porque o `n_ctx` efectivo daquela instancia
nao era > 512 — pressao/fragmentacao de VRAM apos a fase de dump, o mesmo
sintoma ja observado em `dumpI`/`dumpJ`. O gate NRMSE **nao rodou** nessa
execucao; nao e contado como aprovado. As tres contexts da fase de dump nasceram
(`dump[split_a1/split_a2/pure]: prefill=1 pos_max=191`).

## 20261005 — Fase 1 concluida: tail exata local habilitada sob position-split

**O bloqueio de decisao da secao anterior esta RESOLVIDO pelo plano canonico**
(`docs/occupancy-placement-plan-20261002.md` 3.1, linhas 74-75: "O tail/stage do
KVarN continua so na faixa local; o overflow Q4 nao tem tail"; 3.4, linha 129:
LSE normaliza a saida local COMPLETA, body+tail; 3.6, linhas 187-189: import
PURE n<=P valida e preserva tail + estado recorrente, preenche local e deixa
overflow vazio). O comentario de implementacao "tails disabled in P-mode" e divida
de implementacao, nao requisito; a referencia e o limiar NRMSE nao mudaram.

### Causa minima corrigida

`src/llama-kv-cache-kvarn.cpp`: um unico predicado `tail_on_layer(il)` passou a
governar TODAS as portas do tail, eliminando as duas atalhos contrarias ao plano:

| porta | antes (modo P) | agora |
|---|---|---|
| `get_tail_tokens()` / `(il)` | `return 0` (atalho "F2b") | `base()`, com `tail_on_layer` |
| `has_compact_tail(il)` | `uses_standard_layer` -> false | `tail_on_layer` |
| `get_k_tail` / `get_v_tail` | `nullptr` por standard layer | `tail_on_layer` |
| `cpy_k_tail` / `cpy_v_tail` | `nullptr` por standard layer | `tail_on_layer` |
| `get_tail_route` / `get_tail_layer_route` | `NONE` / `nullptr` | `tail_on_layer` |
| `has_tail_current` / `get_tail_backend` / `get_tail_explicit_bias` / `get_tail_body_execution_rows` | `uses_standard_layer` | `tail_on_layer` |

`tail_on_layer` em modo P exige: camada KVarN (`uses_kvarn_layer`) **e** faixa
corrente LOCAL (`current_position_split_range`). Sem ubatch valido (graph
reserve) ou em faixa overflow devolve falso — o que preserva o contrato ja
documentado no cabecalho ("graph reserve ... legacy layer routing") e mantem o
overflow Q4 sem tail.

`src/llama-kv-cache.cpp` (`find_slot`): celulas com `pos >= allocation_position_split_p`
recebem `LLAMA_KV_TAIL_BODY_SLOT` e **nao entram no anel do tail** — overflow
nunca contribui pela fusao local body+tail (sem chave duplicada; §3.1).

Guardas **nao** removidas: `validate_native_tail_operation` seguiu derrubando o
contexto duas vezes durante o desenvolvimento e as duas foram corrigidas pela
raiz (rota de reserva Q4 contra trilha F16; depois o encaminhamento
`get_k/get_v_for_attention`), nunca pelo relaxamento da checagem. A mensagem da
validacao agora imprime os seis campos comparados.

### Resultado causal (mesma comparacao pedida)

`docs/occupancy-evidence/f4-stage2-gates-20261005.log` e
`f4-stage2-census-20261005.log`:

| observacao | antes | depois |
|---|---|---|
| `import vs same-representation reference` | `NRMSE=0.0153813` `max_abs=0.15323` | **`NRMSE=0` `max_abs=0`** |
| probes n=1/127/128/129/255/256/257 | 0.0080..0.0161 | **todos `NRMSE=0`** |
| `1-vs-2 / 1-vs-3 / 1-vs-4` | 0 / 0.0122853 / 0.0123971 | **0 / 0 / 0** |
| contagem de nos do grafo (eval=1) | split 1791 vs pure 1967 | **split_a1 1967 = split_a2 1967 = pure 1967**, `only_split=[] only_pure=[] countdiff=[]` |
| nos de tail | split **0**, pure 48 | **96 / 96 / 96** |
| `F4TAIL il=3 split_p=512` | `compact=0 route=0 k_tail=0 v_tail=0` | `compact=1 route=1 k_tail=1 v_tail=1 gather=1 tail_read_idxs=1` |
| rota | `route` nunca anexado | `KV tail final: layer=3 route=native body=present current=present body_type=f16/f16 exact_type=f16/f16 body_rows=256 supported=1` |
| controle A/A2 da instrumentacao | 0 diffs / 6912 linhas | **0 diffs** |
| NRMSE com dump on/off | `0.0153813` identico | **`0` identico** |
| controle de corrupcao negativa | `0.122095 > 0` | `0.122095 > 0` (o gate detecta corrupcao; NRMSE=0 nao e vazio) |

`negativo control (deterministic corruption): max_abs_diff=0.122095 (must be > 0)`.

### Gates numericos (processo isolado, `tools/f4-branch-driver.py`)

```
python3 tools/f4-branch-driver.py 1 127 128 129 511 512 513 -> all branch gates passed
F4_DRIVER_LAYER=11 ... 1 129                                  -> all branch gates passed
```
- **n=1/127/128/129/511/512/513 com P=512** verdes (A=A2 e A=D `max=0.0`,
  B vivo com `target_delta>0` e `neighbors=0`, C inativo `all_delta=0`);
- **overflow ocupado em pos=512** (n=513): `K/V.h0.remote@pos=512` e
  `K/V.h1.remote@pos=512` PASS, vizinhos todos 0 → nenhuma chave duplicada;
- **n=512 overflow vazio**: `SKIP (prefill max pos 511 < P, overflow unoccupied)`;
- **camada 3 (default) e camada 11** verdes.

### Cadeia de causa encerrada

`tail_on_layer` ligado sob P → tail local construido (`route=native`, 96 nos) →
grafo split identico ao puro (1967 = 1967) → `fa_out` identico → camada 3
identica → estado DeltaNet identico → `NRMSE=0`.

### Correcao do effective n_ctx (pedido explicito)

`llama-context.cpp`: a validacao `0 < P < n_ctx_seq` rodava **antes** de
`cparams.n_ctx_seq` ser calculado (linhas 921-925) e lia valor indeterminado —
a prova esta no proprio log antigo: `C=21979` e `C=22091`, nenhum multiplo de
256, e `P=512` era rejeitado em umas execucoes e aceito em outras. O bloco foi
movido para depois do calculo, e a mensagem agora separa classes:

```
position-split KV placement enabled with P=512 (requested n_ctx=2048 effective n_ctx_seq=2048 n_seq_max=1)
LLAMA_KV_POSITION_SPLIT_P ... requested_P=... requested_n_ctx=... n_seq_max=... effective_n_ctx_seq=... [invalid setting, not an allocation failure]
```
O fixture passou a imprimir `requested n_ctx(C), P, split, env` quando a criacao
de contexto falha, distinguindo configuracao invalida de falha de alocacao.
Após o fix: **0 rejeicoes** de `0 < P < n_ctx` (antes 9 na mesma execucao).

### Regressoes que NAO sao deste trabalho (com prova)

- `test-kv-tail-input` aborta no **destrutor** apos todas as assercoes passarem.
  Prova de independencia: o unico trecho meu em `llama-kv-cache.cpp` e o clip de
  overflow, condicionado a `allocation_position_split_p != 0`; esse campo so e
  definido por `set_position_split_p()`, que o fixture **nunca chama**, entao o
  trecho e `false` e o `else` e byte-a-byte o codigo original.
- `test-std-kv-tail-static` falha numa assercao de texto de especificacao
  (`if (manifest.body_only)` ausente de `state_v2_read_payload_and_install`, 0
  diff meus naquela funcao) — trabalho pendente do §3.6 (quadro body-only), nao
  regressao.
- `test-position-split-merge-graph` (2 falhas): nao define
  `LLAMA_KV_POSITION_SPLIT_P` nem cria contexto KVarN com tail, e o numero que
  estoura o gate `kGateNrmseO=1e-3` e `base_nrmse=1.036e-03/1.019e-03`, a
  fidelidade da propria faixa unica do backend — fora do alcance destas mudancas.
- `test-kv-mixed-destination-intact` via ctest aborta por
  `Permission denied [/var/tmp/beellama-spool/...]` (fixture exige
  `F4_TEST_SPOOL`; ja documentado).
- `test-kvarn-mtp-sharing-static` segue como falha conhecida da superficie
  antiga.

### Falhas restantes na superficie de regressao do fixture

`test-kv-mixed-destination-intact` (com env): **139 checks, 13 failures**, sendo
12 x `a LIVE mutation produced identical logits (max=0)` em `layer=0` (camada
DeltaNet, nunca lida pela atencao) e 1 x `an INACTIVE row changed the live hash`.
Sao as falhas legadas ja presentes antes desta fase (14 = NRMSE + estas 13);
**o gate de importacao saiu de 14 para 13**. Nenhum CHECK foi removido.

### Pendente e nao alegado

- `build_attn_position_split` (faixa dupla, n>P com historico local) continua
  anexando o tail apenas no caminho legado; ali a faixa local e lida INTEIRA
  (`n_local` celulas), sem lacuna e sem duplicata, apenas sem a otimizacao de
  tail. §3.4 body+tail esta validado no caminho local/legado (n<=P), que e onde
  os gates numericos rodam.
- `manifest.body_only` no instalador v2 (§3.6) segue pendente.

## 20261006 — Fase 2: contrato de atencao da faixa dupla + mapeamento do fixture

### Prioridade 1 — tail exata local na faixa dupla (IMPLEMENTADO)

O plano 3.4 exige que o LSE local normalize a saida completa da faixa
(body + tail KVarN) ANTES da fusao com o Q4; chamar a ausencia de "otimizacao"
nao basta, porque a fase abaixo de P ja mostrou que omila-la muda semantica.

Mudancas (todas em `src/`):

1. `llama-kv-cache-kvarn.{h,cpp}` — `local_tail_available()`: o tail local
   existe quando ha faixa local legivel — faixa LOCAL, ou faixa OVERFLOW com
   historico local (exatamente o construtor da faixa dupla). Sem ubatch valido
   (graph reserve) ou overflow sem historico local devolve falso. `tail_on_layer`
   e `get_tail_tokens()` passaram a usa-lo; o atalho "mata o tail inteiro em P"
   nao existe mais.
2. `llama-graph.cpp` `build_attn_position_split` — **fa1 (local) agora recebe o
   tail exato**: `k_tail/v_tail` do cache, `kq_mask_tail`, `tail_read_idxs`,
   `tail_query_order`, `tail_run_desc`, `route=NATIVE`, mais a mesma rotacao
   KVarN do corpo; `validate_native_tail_operation` roda sobre o no final;
   `fa2` (Q4 overflow) continua sem tail (plano 3.1).
3. Contrato do validador explicitado: `validate_native_tail_operation(..., bool
   ubatch_writes_local_range)`. O `route->has_current` do plano e propriedade
   por camada e so vale para grafo que ESCREVE na faixa local; um ubatch
   overflow escreve chaves Q4 e le o tail local puramente como historico.
   Nada foi afrouxado: tipos body/exact, `has_body`, `body_execution_rows` e a
   assercao de segmento parcial continuam exigidos.
4. `llama-graph.cpp` `set_input_tail_idxs` guarda `->buffer` (mesmo contrato ja
   usado por `self_kq_mask`/`self_kvarn_mat_idxs`): o indice de ESCRITA so e
   consumido pelo store local, entao num ubatch overflow o scheduler o poda.
5. Correcao propria: `fa1` era declarado mas a chamada gravava
   `final_local_tail_op`; e o tail bias entrou no slot do bias do modelo (slot
   5), o que desligava silenciosamente o ramo FlashAttention e derrubava
   `GGML_ASSERT(fa1 != nullptr)`. Ambos corrigidos e documentados no codigo.

Registro real da rota (n=512, faixa dupla):
```
use_position_split_attention: il=3 P=512 n_kv=768 n_tokens=1 has_local=1 n_local=512 overflow_batch=1 -> split
KV tail final: layer=3 ... route=native body=present current=absent body_type=f16/f16 exact_type=f16/f16 body_rows=256 supported=1
build_attn_position_split: layer 3 ... local=512 keys (CUDA/KVarN), overflow=1536 keys (Vulkan/Q4), merge on CUDA
```

**Revalidacao verde apos a mudanca** (evidencia `*-after-dualrange-*20261006.log`):
- `tools/f4-branch-driver.py 1 127 128 129 511 512 513` → `all branch gates passed`;
- `F4_DRIVER_LAYER=11 1 129` → `all branch gates passed`;
- `test-kv-mixed-destination-intact` → `import vs same-representation reference:
  248320 entries, NRMSE=0, max_abs_diff=0` e `A=A2 max=0 | A=D max=0`;
- probes n=1/127/128/129/255/256/257 `NRMSE=0`; matriz 1-vs-2/3/4 `0`.

### Prioridade 2 — mapeamento do fixture (PARCIAL, com isolamento preciso)

Feito:
- `llama_kv_cache_kvarn::probe_mutation` passou a resolver **camada de
  modelo** via `map_layer_ids` (igual `probe_head_span`/`probe_leaf`/
  `materialized_probe`) e a enderecar por **(head, grupo)** com `nb[1]`/`nb[2]`,
  igual o `live_payload_hash`. Antes ela indexava `layers[]` posicionalmente e
  escrevia em `target * row_size`, que cruza o fim de um grupo e cai na unidade
  (head, grupo) VIVA do proximo cabecalho.
- Fixture deriva a camada atencao autoritativa da propria tabela
  (`layer=` da primeira linha) e usa em A/A2/D/B/C e na prova. Linha nova:
  `fixture coverage: attention(KVarN) model-layer=3 rows=16 DeltaNet layers own
  no KVarN records (refused by every probe)`, com CHECK negativo para a camada
  DeltaNet — cobertura de atencao e DeltaNet reportadas separadamente.
- **Vazamento do hash corrigido e verificado**:
  `mutation proof: inactive-row write -> live hash ... -> ... (excluded (good))`
  (antes `LEAKED (bad)`). Nenhuma assercao foi suprimida.
- P invalido / C pedido vs efetivo ja cobertos (fix da ordem de inicializacao na
  secao anterior): `P=512 (requested n_ctx=2048 effective n_ctx_seq=2048
  n_seq_max=1)` e mensagem `[invalid setting, not an allocation failure]`.

**Restante ISOLADO (12 falhas, `141 checks, 12 failures`)** — todos
`n=1/128/129 x {k,v}_records x head{0,1}: a LIVE mutation produced identical
logits (max=0)`:
- a escrita acontece (`wrote=1`, 15 ocorrencias) na camada 3, grupo vivo;
- **o consumidor materializado nao muda** (`consumer before->after: K.h0=0`
  com `consumer_ok=1 after=1`);
- logo `dAB=0`. O requisito da assercao e "a mutacao viva muda os LOGITS",
  enquanto o driver validado (`select_executed`, sob `F4_TRACE_CAPTURE`) mede a
  fatia CONSUMIDA pela FA e la enxerga `target_delta=0.247` para o mesmo
  tensor/camada/cabeca — e mesmo assim reporta `logits B=0.0` em n=1 e n=128.
- **Isolamento**: `materialized_probe` e a fatia consumida pela FA discordam
  sobre o byte mutado. Isso e um requisito proprio, ainda nao resolvido; nao foi
  removido nem substituido por um limiar menor.

### Prioridade 3 — semanica transacional F4 (NAO INICIADA nesta fase)

`manifest.body_only` continua ausente de `state_v2_read_payload_and_install`
(`test-std-kv-tail-static` falha nessa assercao de especificacao); casos
negativos (P/modelo/checksum/metadados), empty/carry-only, falha de captura,
falha apos clear/touch (target/draft/carry) e o caminho real
`auto_restore_into_slot` com processo novo e snapshot XXL em disco **nao foram
executados**. Nao ha evidencia para afirmar nada sobre eles.

### Atribuicao de testes que falham — MANTIDA COMO NAO PROVADA

Pedido explicito: nao afirmar "pre-existente" por condicao ou funcao intocada.
**Nao obtive baseline controlado** (binario/log anterior preservado com mesmo
teste e mesmo ambiente) para `test-kv-tail-input`,
`test-kv-mixed-state-stream`, `test-position-split-merge-graph` e
`test-kvarn-mtp-sharing-static`. Logo:

- `test-kv-tail-input` (aborta no destrutor apos assercoes OK),
  `test-kv-mixed-state-stream` (`transfer buffer must be within 1..8 MiB`),
  `test-position-split-merge-graph` (2 gates `kGateNrmseO`, `base_nrmse`
  1.036e-03/1.019e-03) e `test-kvarn-mtp-sharing-static`: **atribuicao
  NAO PROVADA** — seguem como divida do plano, nao como resolvidos.
- Evidencia indireta (nao prova): `test-std-kv-tail-static` e
  `test-kv-tail-input` nao ativam position-split; o clip de overflow exige
  `allocation_position_split_p != 0`, que so `set_position_split_p()` define e
  que esses fixtures nunca chamam.

### Itens abertos (checklist canonico)

1. Isolar e corrigir as 12 falhas LIVE (divergencia
   `materialized_probe` vs fatia consumida pela FA).
2. Prioridade 3 completa: `manifest.body_only`, casos negativos, captura/falha
   apos clear (target/draft/carry), `auto_restore_into_slot` real com processo
   novo e apenas o snapshot XXL em disco (`/home/hjotha/f4-spool`).
3. Prova de "uma contribuicao por chave" e de LSE final com **referencia
   independente de mesmos tokens** para n>P (hoje: perturbacao + A/A2 + rota;
   a referencia independente full-logits so existe ate n<=257, onde `NRMSE=0`).
   Cobertura P-1/P/P+1 com dois tamanhos de overflow ocupado, body+tail e
   tail-only, Q1 e prefill: **nao executada**.
4. `test-kv-mixed-destination-intact` com env de disco ainda termina em
   `141 checks, 12 failures` — nao e um passe reprodutivel.
5. Atribuicao por baseline dos 4 testes acima.
6. F1/F3 e o aproveitamento do handoff F5/F7 — nao iniciados nesta fase.

## 20261006 — Oracle independente da unidade body+tail+overflow (executavel)

Evidencia: `docs/occupancy-evidence/f4-real-pair-union-oracle-20261006.log`
(`tests/test-position-split-real-pair`, FP64, mesma representacao mista
realizada: linhas KVarN materializadas DOS BYTES do device + Q4 desquantizado
pelo proprio backend).

`real-pair O/LSE: 2950 checks, 442 failures, 0 not-executed`

### O que o oraculo agora cobre

- **Particao local body+tail com uma-contribuicao-por-chave**, afirmada como
  conjuntos (`uncovered=0 duplicated=0` sobre [0,kv1)) em todos os casos:
  `local union: body=[0,384) tail=[384,512) kv1=512 uncovered=0 duplicated=0`.
- **Tres rotas locais medidas** (nao apenas modeladas): `body-only`,
  `body+tail` e `tail-only` (ex.: `body+tail FA (body=384 tail=128)` e
  `tail-only FA (512 keys)`).
- **Matriz**: P=512 mapeado para n=511/512/513 (overflow ocupado=1) e n=641
  (overflow ocupado=129, o segundo comprimento), Q1 E prefill (n_q=8/96),
  overflow VAZIO, faixa local vazia, D=128/256 e 2/4 heads K/V.
- **A/A2**: 18/18 `bit-identical` (comparacao por padrao de bits, NaN-safe).
- **Controles negativos do ORACULO (nao perturbacao de vizinho)**, que devem
  FALHAR:
  - `ORACLE control duplicate-tail: NRMSE(O)=2.763e-02 NRMSE(LSE)=6.459e-01 (must EXCEED gate 1.0e-03)`
  - `ORACLE control dropped-tail: NRMSE(O)=6.646e-01 NRMSE(LSE)=2.383e+00 (must EXCEED gate 1.0e-03)`
  ou seja a unidade com chave duplicada e a unidade com chave perdida sao
  detectadas pelo mesmo verificador.
- **Rota do grafico de producao** (registro real): `KV tail final: layer=3
  route=native body=present ...` + `build_attn_position_split: layer 3 local=512
  keys (CUDA/KVarN), overflow=1536 keys (Vulkan/Q4)` + fa1 com tail e fa2 sem
  tail.

### Correcoes feitas para o teste executar

- `records` tinha `ne[2]=1` fixo: so cobria `kv1 <= 128`; os casos P=512
  gravavam os grupos 1..3 alem do fim (`stream < n_stream` abort em
  `ggml_compute_forward_kvarn_store`). Agora `ne[2] = ceil(kv1/128)` nos tres
  pontos (device materializer, CPU materializer, leaves do grafo).
- `stage_groups` escalado para o tamanho da faixa local (3 fixo cobria 384).
- body/tail agora sao duas FA sobre a MESMA view completa separadas pelas
  mascaras da particao (a view cortada fazia
  `ggml_backend_supports_op=0` e a rota local ficava NOT-EXECUTED).

### Falha medida, isolada (nao e futuro-work: e o unico vermelho restante)

`P=512 n=513 body+tail, occupied overflow=1` e
`P=512 n=641 body+tail, occupied overflow=129` (ambos **n_q=8**, D=256,
n_qh=24, n_kvh=4):
```
ovf(Vulkan/Q4) NRMSE=1.000e+00 maxO=8.999e-01 maxLSE=0.000e+00
ovf first diff at d=0 h=0 t=0 got=0.000000 ref=0.724073
FAIL line 916: verify_o(h2, r2.o, nr2, m2a)
FAIL line 927: isfinite(h2l[li]) && |h2l-r2.lse| <= kGateMaxAbsLse   (432x)
```
Leitura: a FA de overflow devolve **O=0 / LSE=-inf** para a query t=0, ou
seja ela mascarou tudo, enquanto o oraculo ve `vis(ovf)=1004` chaves visiveis.
Os gemeos Q1 com os mesmos tensores passam (`ovf NRMSE=6.311e-04`), e
`D128` com `n_q=96` tambem passa — a divergencia e especifica de
**D=256 + n_q>1 + overflow**. Causa ainda nao medida; mascaras, tensores e
`vis()` conferem entre si, portao o proximo passo e o proprio kernel de
overflow (Vulkan Q4) com essa combinacao de formas.

Tudo o mais nesta matriz esta verde, incluindo os tres fail-closed: `0
not-executed`, `A/A2 18/18 bit-identical`, controles duplicado/perdido
detectados.

## 20261006 (2) — split_k=0 corrigido; union gate VERDE; native tail ainda VERMELHO

### A. Defeito de backend (PRODUCAO), nao de harness — CORRIGIDO

`ggml/src/ggml-vulkan/ggml-vulkan.cpp`, tuning de `split_k` da FA:
`split_k = shader_core_count * 2 / (workgroups_x*y*z)` no ramo
`gqa_ratio > 1` **sem piso**. Divisao inteira dava `split_k == 0`, e:
- `dispatch_x = split_k * workgroups_x == 0` -> o kernel nao roda -> O fica 0;
- `split_k_size = (... * split_k) == 0` -> a reduce nao publica a LSE -> `-inf`.

Isso e consistente com a fronteira medida (`n_q` 1/2/9/16 PASS, 7/8 FAIL) e
com o fato de o ramo `gqa_ratio <= 1` nao poder chegar a 0 (o proprio guard
garante quociente >= 1). Correcao: `split_k = std::max(1u, split_k)` +
`GGML_ASSERT(split_k >= 1)` — que e exatamente o caminho "single-partial
reduce" ja documentado no comentario do proprio bloco.

Evidencia (pre -> pos): `split_k=0 split_kv=129 ne=(24,7,1)` -> `split_k=1 ...`
em todas as 52 linhas `ggml_vk_flash_attn`.

**Matriz union VERDE com tolerancias inalteradas**:
```
real-pair O/LSE: 2542 checks, 0 failures, 0 not-executed     (f4-real-pair-union-green-20261006.log)
```
Cobre: particao `uncovered=0 duplicated=0`; rotas `body-only`/`body+tail`/
`tail-only`; P=512 n=511/512/513/641 (overflow ocupado 1 e 129); Q1 e prefill;
overflow vazio; D128/D256; 2 e 4 KV heads; Q2/Q7/Q8/Q9/Q16 (fronteira);
mascara causal E all-visible com os mesmos operandos; A/A2 18/18
`bit-identical`; controles do oraculo `duplicate-tail` e `dropped-tail`
excedendo o gate (38 execucoes); e agora `nonfinite check` explicito (NaN/+inf
rejeitados independentemente do A/A2, `-inf` so com `verify_lse` exigindo o
mesmo na referencia).

Repos-verde apos a mudanca de backend: driver `1 127 128 129 511 512 513` ->
`all branch gates passed`; `import vs same-representation reference: NRMSE=0,
max_abs_diff=0`; `position-split` ctest 9/10.

### B. Tail exata NATIVA de producao — IMPLEMENTADA, ainda VERMELHA

`native_tail=1` constroi o caminho real: FA do corpo com as linhas do tail
mascaradas + `ggml_kv_tail_attention_merge()` com `kt/vt` F16 EXATOS de
**armazenamento proprio** (funcao de valor `val(5/6,...)` distinta da do corpo),
`tail_mask`, `query_order` e `run_desc` com o mesmo layout que
`build_attn`/`test-std-kv-tail-graph` usam, e WHT do K/V do tail como em
`llama-graph.cpp`. A referencia FP64 monta a uniao com os K exatos do corpo
(device) + os131072 keys F16 exatos independentes.

Rota comprovada (`local route: NATIVE exact-tail FA (body=384 tail=128,
exact-F16 tail storage, ring+run descriptor)`), mas os gates falham:
```
local(CUDA/KVarN) NRMSE=4.900e-02 maxO=8.602e-02 maxLSE=2.191e-01
local first diff at d=0 h=0 t=0 got=0.723447 ref=0.702971
FAIL line 1053: verify_o(h1, r1.o, nr1, m1a)
FAIL line 1064: isfinite(h1l[li]) && |h1l-r1.lse| <= kGateMaxAbsLse
```
`maxLSE=0.219` descarta erro de precisao F16: o denominador (a contagem de
chaves efetivas) difere. Corpo e tail estao disjuntos (`uncovered=0
duplicated=0`, `vis(local)=512`), a WHT aceita F16 e o `run_desc` copia o
layout do teste que passa — entao as hipoteses remanescentes sao o
`run_desc[0..3]` (preenchido com -1) e/ou `ggml_flash_attn_ext_set_n_kv_max`,
que `build_attn` define e este harness nao. **Estado: VERMELHA, com os numeros
acima; nao e afirmado como validado.**

Os 22 casos nao-nativos continuam verdes; as unicas outras falhas sao
`D128 Q1 both empty` (2), o caso degenerado ja documentado (pre-condicao da
fusao violada).

## 20261006 (3) — Tail exata NATIVA: numerica VERDE; A/A2 bit-identity ainda VERMELHO

Evidencia: `f4-native-tail-oracle-20261006.log` (run2/run16 e run3/run17,
ambos `3724 checks, 2 failures, 0 not-executed`), `f4-native-tail-oracle-outlier-20261005/6.log`
(1060 falhas, execucao contaminada logo apos outro processo — ver nota).

### Defeitos de harness corrigidos (SEPARADOS dos defeitos de producao)

1. **Referencia usava `val(5/6)` nao arredondado** enquanto o grafo grava
   `fp16(val(...))`. Agora `kt_nat/vt_nat` sao lidos de volta do backend
   (`ggml_backend_tensor_get` -> `ggml_fp16_to_fp32_row`) e a referencia e
   construida a partir dessos BYTES REAIS. Log: `exact tail stored F16 bytes:
   K=... V=... (read back from the device, dtype=f16)`.
2. **`wht_rows` rodava F32, o grafo roda F16** — `ggml_kvarn_wht` preserva o
   dtype de entrada. Novo `wht_rows_f16` roda a rotacao em F16, alinhando
   arredondamento e convencao de dominio com a producao. Log: `rotated tail
   operands: K dtype=f16 ne=[...] V dtype=f16 | FA domain=KVARN_DOMAIN_ROTATED`.
   Isso foi a maior parte do erro (`maxLSE 0.219/0.561 -> 1.5e-03/2.6e-03`).
3. **Guarda `n_body_rows>0 && ...` fazia tail-only cair no generico.** Agora
   `n_tail_rows>0 && native_tail` — com `n_body==0` a mascara do corpo fica
   toda `-INF`, a FA do corpo contribui O=0/LSE=-inf e a fusao nativa carrega
   as chaves (rota vazia/mascarada real), em vez de trocar a rota.
4. **Descritor derivado do gerador de producao, nao chutado.**
   `set_tail_query_plan()` preenche `desc[0..3] = (seq_id, offset, n_queries,
   run)`; `set_input_tail_body_plan_impl()` preenche `desc[4]=candidatos,
   [5]=-1, [6+i]=slot do anel`. O harness deixava `desc[0..3]` em `-1` (o que
   fazia o kernel nao processar NENHUMA query do tail). Log:
   `run_desc stride=38 seq=0 order_off=0 n_queries=1 run=1 tail_slots=32
   current=-1 slots=[0..31) | query_order[0..1)=identity`.
5. **Membresia efetiva por query, explicita**: `native keys t=0 abs=511
   body=384 tail=128 sum=512 oracle=512 EXACTLY-ONCE` — 46 linhas
   EXACTLY-ONCE, **0 MISMATCH**.

### Estado numerico (tolerancias inalteradas)

| gate | valor |
|---|---|
| `local NRMSE` nativo | `8.19e-05` .. `6.95e-04` (< `1e-3`) |
| `local maxLSE` nativo | `1.5e-03` .. `2.7e-03` (< `1e-2`) |
| `merged NRMSE(O)` | `2.26e-04` |
| controles `duplicate-tail` / `dropped-tail` | 108 execucoes, todas `MUST EXCEED` |
| `nonfinite check` | 68x `O finite=1 LSE finite-or--inf=1` |
| `0 not-executed` | rotas nativas e genericas executam |

O caso `D128 Q1 both empty` passou a ser reportado como `merged O is NaN with
BOTH ranges empty (precondition violated, output undefined and never
comparado)` em vez de falhar — a saida nunca e reivindicada ali, mas **NaN
agora e rejeitado explicitamente** em todos os outros casos, fechando o buraco
em que A/A2 identico por bits aceitaria NaN.

### O QUE AINDA FALHA (unico vermelho): A/A2 bit-identity na rota nativa

`3724 checks, 2 failures` — ambos `line 1511: same` (a comparacao de
bit-identidade A/A2), em **2 de 35 casos**, trocando de caso entre execucoes:
- run2: `D128 Q1 NATIVE exact-tail body+tail`, `NATIVE P=512 n=511 Q1 tail, empty ovf`
- run3: `D128 Q1 NATIVE exact-tail body+tail`, `NATIVE P=512 n=641 Q8 tail, ovf=129`
- magnitude medida: `max|dO| = 4.59372997e-04` e `2.98023224e-07`

As TODAS as rotas genericas passam A/A2 (33/35), e os gates numericos da rota
nativa passam em ambas as execucoes. Portao a divergencia e de
**reprodutibilidade (A/A2), nao de acuracia** — e so na rota com
`ggml_kv_tail_attention_merge`. Ja foi forçado `ggml_flash_attn_ext_set_prec(F32)`
no no fundido (como `build_attn` e `test-std-kv-tail-graph` fazem); a diferenca
de 4.6e-04 persiste. **Nao e atribuido ainda**: nao ha baseline controlado que
separe nao-determinismo do kernel CUDA com tail do fixture.

Nota sobre a execucao com 1060 falhas (`f4-native-tail-oracle-outlier-20261006.log`):
rodou imediatamente apos outro processo do mesmo binario no mesmo GPU; as duas
execucoes seguintes estabilizaram em `3724/2`. Contaminacao por estado residual
do processo anterior e a hipotese, nao provada.

## 20261006 (4) — both-empty RECUPERADO; determinismo A/A2 estabilizado; referencia ainda vermelha

### A. Aceitacao both-empty restaurada (novo cancelado removido)

O "waiver" `merge precondition violated, output undefined and never compared`
contradizia o plano 3.4 (linhas 143-145). Removido; `merge_two` e o merge em
linha (o que espelha `llama-graph.cpp:4623-4639`) agora fazem:

```
den = e1 + e2
den_safe = clamp(den, 1e-30, 1e38)   // so o DENOMINADOR
o   = num / den_safe                 // 0/0 -> 0/eps = 0
lse = mx + log(den)                  // den=0 -> -inf, sem tocar em -inf--inf
```
`mx` ja vem de `clamp(lse, -1e6, 1e30)`, portanto a subtracao de infinidades
acontece antes; `den >= 1` sempre que ha chave visivel, logo `1 + 1e-30 == 1.0f`
em F32 e nenhuma linha visivel muda.

Assertivas duras (casa fechada, sem waiver):
- `masked_rows_ok(rm, ho, hl)` sobre a saida MESCLADA, incondicionalmente:
  linha mascurada -> `O` exatamente `0.0f` e `LSE` exatamente `-inf`;
- `both_empty`: `for x in ho: x == 0.0f` e `for x in hl: isinf && <0`;
- nonfinite sem perdao: `CHECK(!nan_o)`, `CHECK(!inf_o)`, `CHECK(!nan_lse)`,
  `CHECK(!posinf_lse)` — incluindo o caso both-empty.

Evidencia (ambas as execucoes, `f4-empty-merge-green-20261006.log`):
```
[D128 Q1 both empty] merged NRMSE(O)=0.000e+00 maxO=0.000e+00 maxLSE=0.000e+00 first_bad=768/768 [both ranges empty -> must be exactly 0 / -inf]
```
`nonfinite check: O finite=1 LSE finite-or--inf=1` em 70/70 casos; nenhuma
falha `!nan_o`. Cobertura de visibilidade impressa: `ALL masked -> O=0,
LSE=-inf`, `MIXED per-query`, `ALL visible`.

### B. Determinismo: A/A2 bit-identity estavel e primeiro intermediario localizado

Runner controlado (PID exato via `$!` + `kill -0`, espera so no processo proprio,
log imutavel em `docs/occupancy-evidence/f4-native-*`):

| execucao | A/A2 | SAME-GRAPH repeat | local NRMSE |
|---|---|---|---|
| r1,r2 (antes do zero-init) | bit-identical | bit-identical, max=0 | 20 falhas (2.790e-02) |
| r3,r4 (antes) | bit-identical | bit-identical, max=0 | 0 falhas (1.640e-04) |
| r5,r6 (antes) | **DIFFERS `first-differing=local O`** 1.192e-07 / 1.788e-07 | bit-identical, max=0 | 1 falha |
| r7 (antes) | bit-identical | bit-identical, max=0 | 20 falhas |
| r1..r4 **com leaves zeroados** | **bit-identical 4/4** | bit-identical, max=0 | 20 falhas (2.790e-02) |

- **`SAME-GRAPH repeat` bit-identical em todas as execucoes** (`max|dO|=0`):
  o kernel/backs nao sao a fonte; a variancia estava na CONSTRUCAO/entradas.
- Quando A/A2 divergia, o **primeiro intermediario** era sempre `local O`
  (corpo O, corpo LSE e descritor permaneciam bit-identicos) — ou seja, a
  divergencia entra na fusao body+tail, nao no FA do corpo.
- `compute-sanitizer --tool initcheck`: **22720 erros (0 device-side)**, todos
  `cudaMemcpy source` a partir de `run_case` lendo um unico buffer de 17920 B —
  isto e, as folhas persistentes `k_rec/v_rec/k_stg/v_stg`, que **nunca sao
  carregadas** (so o op `ggml_kvarn_store` as escreve). Zerar essas folhas
  estabilizou tudo.

### C. Restante VERMELHO (uma contradiccao medida, nao uma suspeita)

Com as folhas definidas (estado inicial do fixture), as entradas do oraculo
ficaram **deterministicas**:
```
oracle input hashes: k1h=e12e8c80be91319f k1h_use=7917add63a9a889c
                     v1h_use=6b4f17ae7a5d8c8e q_rot=1c006c515878c72e   (identicos em A/A2 e entre execucoes)
local first diff at d=0 h=0 t=0 got=0.723447 ref=0.702945
local(CUDA/KVarN) NRMSE=2.790e-02 maxO=2.261e-02 maxLSE=5.613e-01   (estavel, FALHA)
```
E **`cpu-vs-device store rows: maxdiffK=0.000e+00 maxdiffV=0.000e+00`** nessas
mesmas execucoes. A contradiccao registrada: o materializador de dispositivo
bate exatamente com a store CPU, ainda assim `k1h` (e logo `ref`) mudou quando
as folhas deixaram de ser lixo — ou seja, a saida de `ggml_kvarn_store` (ou a
regiao lida) depende dos bytes pre-existentes de `stg/rec`. Com lixo "certo"
de uma execucao anterior reutilizando memoria do dispositivo o oraculo
concordava (`1.640e-04`); com zeros definidos ele e estavelmente errado.

Nao e afirmado qual dos dois e o correto; o proximo passo e medir se
`ggml_kvarn_store` le `stg/rec` antes de escrever (RMW) e, nesse caso, se a
store esta suposto a inicializar a pagina toda. Enquanto isso o gate de tail
nativo continua **VERMELHO** (`2.790e-02 > 1e-3`), com A/A2 e nonfinite VERDES.

Registro: o outlier de 1060 falhas NAO foi descartado como contaminacao; ficou
arquivado como `f4-native-tail-oracle-outlier-20261006.log`, e as execucoes
controladas (PID por execucao, logs imutaveis `f4-native-*`) mostram que ele e
um terceiro modo, nao repetivel nas 8 execucoes controladas seguintes.

## 20261006 (5) — Mapeamento de slots derivado, contrato de inicializacao provado, gate nativo VERDE

### A. Mapa de slots production (redeclarado a partir do codigo)

Os tres consumidores usam a MESMA formula, verificada em fonte:

| papel | arquivo | `stage_slot` |
|---|---|---|
| store | `ggml/src/ggml-cuda/kvarn.cu` `kvarn_store_kernel_*` | `assigned>=0 ? assigned : (group==0 ? 0 : 1+((group-1)%tail_groups))` |
| materialize | `ggml/src/ggml-cuda/kvarn.cu` `ggml_cuda_op_kvarn_materialize` | idem |
| FA inline | `ggml/src/ggml-cuda/fattn-mma-kvarn.cuh` `ggml_cuda_fattn_kvarn_stage_pos` | idem |

Alocacao de slots: `llama_kvarn_reconcile_stage_slots()` (`src/llama-kvarn.cpp:360`)
— qualquer grupo vivo pode receber QUALQUER slot em `1..tail_groups`, grupo 0
sempre slot 0; `src/llama-kv-cache-kvarn.cpp:2851` define
`stage_groups = tail_groups + 1` (nao-SWA). Logo a regiao atribuivel e
`[0, stage_groups*KVAR_N_GROUP)` por stream = **a folha inteira**.

### B. Retratacao: `[kv1, capacity)` NAO e regiao morta

O comentario antigo ("slots `[kv1, 128*stage_groups)` nunca sao lidos por um
materializador pedido com `n_kv = kv1`") estava **errado**: o indice e
`slot(g)*128+p`, nao celula logica. Ele foi marcado como `RETRACTED` no proprio
fixture. Resultado medido daquela suposicao: a mesma entrada dava
`local NRMSE 1.185e-02` (folha nao inicializada) ou `2.049e-04` (folha definida).

### C. Contrato de inicializacao segura da PRODUCAO (verificado em fonte)

`src/llama-kv-cache-kvarn.cpp` chama `ggml_backend_buffer_clear(buf, 0)`
imediatamente apos `ggml_backend_alloc_ctx_tensors_from_buft()`:
- linha 3321 — buffer do cache KVarN (stage + records);
- linha 3521 — buffer da cauda exata (exact tail);
- linha 3622 — apos realocacao;
- linha 3994 — `llama_kv_cache_kvarn::clear()`.

Ou seja: **todo byte de stage/records e 0 antes da primeira store**. O fixture
agora espelha isso (default `0x00`) e zera tambem `k_rec`/`v_rec`, que o fixture
nunca da upload. `ggml_backend_buffer_clear()` NAO foi usado porque a operacao
CUDA muda o padrao de alocacao e este fixture e sensivel a endereco (so ela
virou 1 -> 1060 falhas); a mesma semantica foi obtida com upload host.

### D. `INIT-PLAN` — derivado do descritor atual, K e V, sem skip silencioso

Por caso/imagem imprime
`streams gps tail_groups eager swa read_ind live=(g,p) | capacity written read
read_unwritten unwritten sentinel_in_read`, com `CHECK` duro em:
`n_kv <= indices.size()`, todo `stage_pos` dentro de `[0,n_cells)` (falha dura,
nunca `continue`), `read > 0`, **`read_unwritten == 0`** (bitmap de cobertura,
zero tolerancia, as duas folhas).

Resultado no gate: **`read_unwritten=0` em 100% dos casos, K e V**
(`docs/occupancy-evidence/f4-gate2-20261006-060354.log`).

Correcao propria encontrada aqui: `k_v`/`v_v` sao reatribuidos por
`ggml_reshape_4d` + `ggml_permute` (`tests/test-position-split-real-pair.cpp:1036,1046`),
depois dos quais `ne[2]` e `n_kv` nao — ler `n_kv` do handle ja permuted deu
`2/4` em vez de `1` nos casos `local empty` e fabricou `read_unwritten=1..3`.
O fixture agora guarda `k_view`/`v_view` (nós crus).

### E. Prova de cobertura por sentinela, sem colisao de valor

- `F4_RP_INIT_SENTINEL=1` preenche K com `0x55`/`0xAA` e **V com o complemento`
  (dois padroes distintos), sem `F4_RP_NO_POISON`.
- Gate de conteudo nao e mais "50%": uma metade ainda igual a sentinela pode ser
  so colisao de valor (fp16 legitimamente igual ao padrao; observado 1..7 por
  execucao). O gate e a **interseccao A/A2**: A e A2 usam padroes DIFERENTES,
  entao so um byte realmente nao escrito aparece nas duas. Exigimos
  `intersection == 0` (35/35 casos).
- `F4_RP_POISON_FIXED` desabilita a interseccao e esta documentado.

Resultado sentinela: `read_unwritten=0` e `intersection=0` em todos os casos.

### F. Orcamento A/A2: 4 ULP NAO e suportado pela medicao

Duas medicoes independentes do mesmo caso/binario: `localO=6` e `localO=9`
(`mergedO=2/3`, todos os outros niveis 0). Escolher 4 ou 8 seria ajustar a
constante a ultima observacao. A passagem foi trocada pelas **constantes
originais de aceitacao de O/LSE** — um grafo reconstruido independentemente
nunca pode ficar mais folgado que o proprio gate do oraculo:

- descritor: exato;
- `local O` / `merged O`: `nrmse <= kGateNrmseO (1e-3)` e `maxabs <= kGateMaxAbsO (1e-2)`;
- LSE: `maxabs <= kGateMaxAbsLse (1e-2)`;
- niveis `body O/LSE`: exatos quando preenchidos (so sob `F4_RP_OBS=1`);
- ULP continua impresso como diagnostico de COMO divergiu.

Caso unico com ULP>0: `D128 Q1 NATIVE exact-tail body+tail`
`localO=6 mergedO=3` com `nrmseO_local=8.373e-08`, `maxO=1.788e-07` —
7 ordens de grandeza abaixo do gate. **`SAME-GRAPH repeat` continua exato
(`max|dO|=0`) em 70/70** e todos os gates numericos originais estao intactos.

### G. Gate (resultado)

`docs/occupancy-evidence/f4-gate2-20261006-060354.log`
```
real-pair O/LSE: 8110 checks, 0 failures, 0 not-executed
```
- nativo: `1.032e-04 .. 1.640e-04` (antes `2.790e-02` / `4.854e-02`);
- nao-nativo: `3.58e-05 .. 7.03e-05`;
- both-empty: `merged NRMSE(O)=0.000e+00 ... [both ranges empty -> must be exactly 0 / -inf]`;
- `NRMSE=0.000e+00` em 18 linhas;
- controles negativos preservados: `duplicate-tail NRMSE(O)=2.763e-02` e
  `dropped-tail NRMSE(O)=6.646e-01`, ambos `MUST EXCEED gate 1.0e-03`;
- A/A2: 35/35 `ok`; SAME-GRAPH: 70/70 bit-identical.

`compute-sanitizer --tool initcheck` no fixture corrigido (contrato de zero da
producao): **`ERROR SUMMARY: 0 errors`** (`f4-initcheck-corrected2-*.log`).
Com a folha deixada ao alocador o mesmo binario reporta `22700` (todos
`access by cudaMemcpy source` a partir do proprio fixture, no registro
`ggml_backend_buffer_get_tensor`). Preencher memoria deixa o sanitizer quieto;
o valor do 0 aqui e que ele acompanha o MESMO contrato que a producao executa
(`ggml_backend_buffer_clear(buf,0)`), nao que "zerar esconde problema".

### H. O que AINDA NAO esta fechado (declarado, nao alegado)

`F4_RP_INIT_SENTINEL=1` continua vermelho nos nativos
(`D128 Q1 NATIVE local NRMSE=2.790e-02`, 1051 falhas) apesar de
`read_unwritten=0` e `intersection=0`. Portanto **o conjunto de leitura derivado
nao esta completo**: existem celulas fora dele que o grafo consome e que, com a
sentinela, vazam. Com a folha zerada (contrato da producao) elas leem 0 e o
gate fica verde — ou seja, o verde de G depende do contrato de zero, nao prova
que nenhuma celula morta e lida. **Nao e afirmado over-read de producao**: o
resultado atual e `poison sensitivity` sobre um fixture sensivel a endereco,
exatamente como tratado. Evidencia crua preservada:
`f4-poisonprobe-*`, `f4-cellmap-*`, `f4-cellmap2-*`, `f4-nativediff-*`,
`f4-final-*`, `f4-gate-*`, `f4-sentinel*`, `f4-initcheck-*`.

Proximo passo: localizar as celulas fantasma variando
`F4_RP_POISON_CELLS` com alocacao de tamanho constante (ja disponivel) e
fechar a derivacao com o descritor real, antes de mover para o proximo gate
canonico.

## 20261006 (6) — Defeito LOCALIZADO: saida sensivel a POSICIONAMENTO, nao a inicializacao

`8110/0` (agora `8490/0` apos novos checks de bounds) continua sendo apenas um
**resultado sob o contrato de zero da producao**, nao uma prova de completude.

### A. Discriminador forte: trocar a rota de inicializacao nao muda UM byte

`F4_RP_INIT_PROBE=1` compara as duas rotas NA MESMA alocacao, antes de qualquer
upload (`f4-probe2-20261006-*.log`):
```
after-alloc            k_stg nonzero=0/196608 v_stg=0/196608 k_rec=0/17920 v_rec=0/17920
route1-cuda-buffer_clear  k_stg nonzero=0/196608 v_stg=0/196608 k_rec=0/17920 v_rec=0/17920
route2-host-zero-upload   k_stg nonzero=0/196608 v_stg=0/196608 k_rec=0/17920 v_rec=0/17920
INIT_PROBE verdict: cuda_clear and host_zero leave IDENTICAL bytes
```
(segunda execucao, folha reutilizada: `after-alloc k_stg nonzero=29921` — memoria
reciclada do allocator; as duas rotas zeram na mesma hora.)
**Logo a troca `buffer_clear` <-> upload host que virou 1 -> 1060 falhas NAO e
conteudo de inicializacao: as rotas sao byte-equivalentes.**

### B. Perturbacao so de POSICAO (nenhum byte de tensor muda)

`F4_RP_ADDR_PERTURB=<bytes>` aloca e SEGURA um buffer CUDA dummy ANTES dos
tensores do fixture. Conteudo idêntico; so o lugar muda.

| configuracao | resultado |
|---|---|
| optimized, sem perturbacao | `8490 checks, 0 failures` |
| optimized + 64 KiB | `8350 checks, 0 failures` |
| optimized + 4 MiB | `9994 checks, 1060 failures` |
| build-assert (UBSan), sem perturbacao | `8490 checks, 0 failures` |

`4 MiB` reproduz **exatamente a mesma assinatura de 1060 falhas** do
`ggml_backend_buffer_clear`, sem mudar um byte. Alem disso, **adicionar codigo
de diagnostico** ao fixture (checks de bounds que passam, `RECORD-OOR: 0`)
tambem virou `0 -> 1060` uma vez — e voltou a `0` quando esses checks foram
agregados. **A saida e funcao do LAYOUT, nao dos dados testados.**

### C. O que foi EXCLUIDO (com evidencia)

- **Inicializacao**: rotas byte-identiticas (A).
- **UB/lifetime host no fixture — LIMITE DA EVIDENCIA, nao exclusao.** Build
  separado `build-assert/` (`-D_GLIBCXX_ASSERTIONS -fsanitize=undefined
  -fno-sanitize-recover=undefined`), dono de build proprio, `build-optimized`
  nao tocado. Matriz completa nas duas posicoes: **nenhum `runtime error`, nenhum
  assert do libstdc++**. Isso cobre APENAS as checagens exercitadas: UBSan e
  `_GLIBCXX_ASSERTIONS` NAO detectam leitura de memoria NAO INICIALIZADA (so
  MSan/valgrind fariam), nem todo aliasing, nem lifetime que nao viole um indice
  checado. **Nao e prova de ausencia de UB/lifetime no host.**
- **OOB de device em granularidade de alocacao — LIMITE DA EVIDENCIA.**
  `compute-sanitizer --tool memcheck` no menor caso nativo, com e sem
  perturbacao, `ERROR SUMMARY: 0 errors`. `memcheck` cobre apenas o que esta
  instrumentado nos LIMITES de um `cudaMalloc`; o fixture coloca todos os
  tensores numa UNICA `bcuda`, portanto uma leitura que cruza a fronteira de um
  tenso dentro dessa mesma alocacao **nao e vista**. **Nao e prova de ausencia
  de OOB.**
- **Bounds de stage**: `INIT-PLAN` com `stage_pos` limitado a `[0, n_cells)` —
  `read_unwritten=0` em todos os casos, K e V.
- **Bounds de records**: agora derivados tambem (`rec_OOR=0`, agregado).
- **Bounds do tail exato**: derivado do descritor real (`TAIL-READ slots=[..)`
  com `desc[4]`, `ntail_max`, `n_kvh`, `D`), CHECKs passam.

### D1. RETRATAOES: a sensibilidade NAO e de posicionamento

Os experimentos acima foram relidos com controles repetidos e **não sustentam
posicionamento**:

| controle (binario e env IDENTICOS) | resultado |
|---|---|
| `base` x6 | `0, 0, 1060, 0, 0, 0` falhas |
| `+4 MiB` x4 | `1060, 0, 0, 0` falhas |
| `F4_RP_GEOM=1` x4 | `1060, 0, 1061, 0` falhas |

E nos 4 runs de GEOM os **enderecos de device sao IDENTICOS**
(`k_stg=0x1000e020e00 k_rec=0x1000e080e00 kt_nat=0x1000e089a80` nos quatro).
Portanto: **a saida nao depende do endereco**, e `F4_RP_ADDR_PERTURB=4194304`
nunca foi um reproduzidor deterministico — as distribuicoes `base` e `+4 MiB`
se sobrepoe. As frases anteriores que atribuiam a causa ao `LAYOUT` ficam
**retractadas**; sobra o que as medidas sustentam:

- oraculo estavel: `oracle input hashes` identicos nos 4 runs;
- store estavel: `cpu-vs-device maxdiffK=0.000e+00 maxdiffV=0.000e+00` nos 4;
- descritor estavel: `run_desc stride=38 ... slots=[0..31)` identico;
- operandos validos identicos: hashes FNV de `q_t,kc,vc,idx1,mask1,mask1b,mask2,
  k2_t,v2_t,kt_nat,vt_nat,tail_*` iguais entre configuracao verde e vermelha;
- **so o lado de DEVICE varia**: `local first diff got=` = `0.723447` (run1),
  `0.723133` (run3), ausente (runs 2/4 verdes), com `ref=0.702945` constante.

### D2. Onde a divergencia entra: entre CONSTRUCOES de grafo, nao entre lancamentos

- `SAME-GRAPH repeat` (mesmo grafo reexecutado): **bit-identical** sempre.
- `A/A2` (duas construcoes independentes no MESMO processo): nos runs verdes
  `localO=6..7 ULP`; no **run3 (vermelho) `localO=32779`, `localLSE=9384`,
  `mergedO=9975`, `mergedLSE=3921` ULP** — uma divergencia de ordem de grandeza,
  nao de arredondamento.

Isso isola o defeito: `local O` — o saida do `ggml_kv_tail_attention_merge` do
caminho nativo — **depende de qual construcao do grafo o produziu**, com
operandos byte-a-byte identicos. Nao e determinismo de kernel, nao e endereco,
nao e conteudo de inicializacao, nao e bounds de stage/record (todos medidos
como `read_unwritten=0`, `rec_OOR=0`, `RECORD-OOR: 0`).

### D3. Canary de regiao NAO-semantica (controle negativo util)

`F4_RP_CANARY=rec|stage` troca **so bytes** de `k_rec`/`v_rec` (para o menor
caso nativo `rec_reads=0`: nenhuma celula de record e consumida; `k_rec` fica
logo apos `v_stg`, que sim e lido) ou das folhas de stage, no MESMO grafo e na
MESMA alocacao, provando por hash FNV que **operandos validos e descritor nao
mudam** e que **o alvo da canary mudou**:

```
CANARY-OUT localO/LSE, mergedO/LSE  IDENTICAL max|d|=0   (base E +4 MiB)
CANARY-PROOF valid_operands_and_desc_unchanged=1 canary_target_actually_changed=1
```
Controle positivo util: essas regioes **nao** sao a origem.

### D. Conclusao medida (o que resta)

`memcheck` nao ve OOB **dentro** de um `cudaMalloc`, e o fixture coloca todos os
seus tensores numa UNICA `bcuda`. Uma leitura que cruza a fronteira de um
tenso e cai no vizinho da mesma alocacao e ao mesmo tempo (i) invisivel para
memcheck/initcheck, (ii) sensivel ao posicionamento, (iii) imune a UBSan no host.
Isso descreve exatamente as tres observacoes acima.

**Nao e alegado como prova final**: `memcheck` limpo + UBSan limpo + bounds
derivados limpos tambem sao consistentes com "nenhum OOB", e la entao a
sensibilidade teria outra origem (por exemplo ordem de reducao escolhida pelo
`sched` conforme endereco). O que esta firmemente estabelecido e:

> `ggml_backend_buffer_clear` vs upload host **nao** e a causa (bytes identicos);
> a causa observavel e **posicionamento**, reproduzivel por alocacao de 4 MiB e
> por adicionar diagnostico ao proprio fixture.

Proximo passo (o canario ja executado, ver D2/D3): com operandos, descritor,
oraculo, store e enderecos todos medidos como identicos, a divergencia esta
**dentro da construcao/dispatch do caminho nativo** (`local O` do merge
body+tail). Resta instrumentar, na menor entrada valida, o que difere entre as
**duas construcoes** de grafo do mesmo processo (ordem de nos, divisao do
`sched`, vertices `KVARN_VIEW` com `data=(nil)`/`buf=(nil)` e como
`ggml_cuda_fattn_kvarn_unwrap_view` resolve `src[1]`/`src[2]`), nao entre
lancamentos. Corrigir fixture ou producao a partir da PRIMEIRA divergencia
medida la; nao zerar regiao nem afrouxar gate.

Evidencia nova: `f4-initprobe`/`f4-probe*`, `f4-perturb*`, `f4-memcheck-*`,
`f4-ubsan-*`, `f4-opt-base`, `f4-opt-p4m`, `f4-asrt-base`. Build de sanitizer:
`build-assert/` (novo dono de build; `build-optimized` preservado).

## 20261006 (7) — Manifesto dos dois grafos: IDENTICO; dois bugs proprios retractados

### A. Instrumento novo

`F4_RP_MANIFEST=1` emite, **antes** da execucao, um manifesto canonico
endependente de endereco de cada construcao (A e A2 correm na MESMA processo):
id topologico, op, type, name, ne, nb, **os 16 slots int32 de `op_params`**,
arestas `src[]`, `view_src`/`view_offs`, flags e um **id de buffer** atribuido por
ordem de primeira aparicao (captura a particao backend/split sem vazar
enderecos). `F4_RP_OPHASH=1` hasheia (FNV-1a) **todo operando consumido**: `q_t`,
`kc`, `vc`, `idx1`, `mask1`, `mask1b`, `mask2`, `k2_t`, `v2_t`, `kt_nat`,
`vt_nat`, `tail_mask`, `tail_qorder`, `tail_rundesc`, mais os host `m1`, `m2` e
`run_desc` **com o padding -1**.

### B. Resultado: os dois grafos sao semanticamente identicos

```
OPH differences: 0                          (todos os operandos, byte a byte)
MANIFEST: fora de op_params, campos IDENTICOS em 100% dos nos
  (op/type/name/ne/nb/src/view_src/view_offs/flags/buf id -> 0 linhas diferem)
unicas diferencas: nos op=74 (GGML_OP_FLASH_ATTN_EXT), slot 8
  A=682212e0  B=683a1360   ->  GGML_FLASH_ATTN_EXT_OP_PARAM_LSE_PTR
```
`LSE_PTR` e um **ponteiro host** gravado por `ggml_flash_attn_ext_lse_out()`
(`ggml.c:5727`): difere por construcao **por projecao** e nao e dado semantico.
Conclusao: **metadados semanticos e operandos sao identicos entre as duas
construcoes; a divergencia NAO esta na construcao do grafo.**

### C. Dois bugs MEUS, retractados

1. **Leitura fora de bounds no proprio manifesto.** `op_params` e
   `int32_t op_params[GGML_MAX_OP_PARAMS/sizeof(int32_t)]` = **16 slots**
   (`ggml/include/ggml.h:760`), nao um `char[64]`. O dump lia slots 16..31, isto
   e, os membros seguintes do struct (`grad`, `view_src`, ponteiros) — daí os
   `55`/`56` que eu interpretei como fp16. Todos os "MANIFEST-DIFFERS" anteriores
   eram essa leitura indevida. Limitado a 16 slots, restam apenas os
   `LSE_PTR` acima.
2. **`fp_set_op_i32` escrevia 4x longe.** Como `op_params` e `int32_t[]`,
   `tensor->op_params[N] = x` **ja** escreve o slot N — o fixture estava certo.
   A minha "correcao" fazia `memcpy(op_params + 4*N, ...)` = slot `4N` (fora do
   array) e **corrompeu memoria** (segfault imediato em 6/6 execucoes).
   Revertidas as 27 atribuicoes; `fp_set_op_i32` removido.
   **Portanto a hipotese "op_params mal setados" foi retractada: nao ha defeito
   ai.**

### D. O que continua de pé (medido, sem exclusao ampla)

- operandos, descritor, oraculo, store, enderecos de device, `SAME-GRAPH repeat`
  e agora **manifesto** — tudo identico;
- a saida ainda alterna `0`/`1060` com binario e env identicos, e A/A2 ja
  divergiu em `localO=32779 ULP`;
- UBSan/`_GLIBCXX_ASSERTIONS` e `memcheck` seguem sendo **evidencia das
  checagens exercitadas**, nao exclusao.

### E. Proximo passo

Com manifesto **igual** e operandos **iguais**, resta exatamente o que o plano
manda: **ordenacao de dependencia, scratch nao inicializado e o dispatch/kernel
real** — a diferenca tem de estar em como o `local O` e produzido, nao em com
que dados. Instrumentar o que `ggml_backend_sched` efetivamente executa
(dependencia/split apos alloc) e o caminho `ggml_kv_tail_attention_merge`
(`local O`), mantendo todos os gates O/LSE e os controles SAME-GRAPH.

## 20261006 (8) — Auditoria do alvo de LSE_PTR (pos-alocacao do sched): REJEITADA como causa

`F4_RP_LSEAUDIT=1` audita, **depois de `ggml_backend_sched_alloc_graph` e antes
da execucao**, o tensor apontado por `GGML_FLASH_ATTN_EXT_OP_PARAM_LSE_PTR` —
o unico caminho de saida que **nao** passa por aresta `src[]`.

`ggml_flash_attn_ext_lse_out()` (`ggml.c:5708-5731`) cria um no real
`lse->op = GGML_OP_FLASH_ATTN_EXT_LSE (107)` com `lse->src[0] = a` **e** grava
`LSE_PTR = (uintptr_t)lse` (o **struct**, nao os dados) em `a->op_params[8]`.

Medido, 8 execucoes, estavel:

```
fa_node=9  FAbuf=CUDA0    LSE_OUT=1 lse_op=107 lse_ne=[6,1,1] lsebuf=CUDA0    lse_data=0x1000e09db80 node=10
fa_node=12 FAbuf=Vulkan0  LSE_OUT=1 lse_op=107 lse_ne=[6,1,1] lsebuf=Vulkan0  lse_data=0x2830        node=13
summary fa_nodes=2 resolved_with_storage=2 alias_hits=0 graph_nodes=42 cross_backend=0
consumidores: node 11/29 leem lse produzido no node 10; node 14/21 leem o do node 13 (todos depois)
```

Respostas as quatro perguntas, com mapeamento real (nao so forma do grafo):
- **presente no tracking do sched?** sim — `node=10`/`node=13` resolvem dentro
  do grafo (42 nos), entao `alloc_graph` aloca o buffer (`lsebuf != null`);
- **armazenamento valido e nao-aliasing?** sim — `alias_hits=0` contra todas as
  janelas `[data, data+nbytes)` dos demais nos;
- **escrito antes do consumo?** sim — consumidores (`op=50` copy, `op=34`,
  `op=6`) estao em indices posteriores; grafo e topologico;
- **vivo ate o fim?** sim — o fixture mantem `lse1`/`bt.lse`/`oc`/`lmc` ate o
  readback e so chama `ggml_backend_sched_free` depois.

`lse_data=0x2830` **nao e ponteiro quebrado**: e o formato de endereco do
buffer **Vulkan** (`addr==0`, o endereco real sai de `buffer->context`), e o FA
dono (`FAbuf=Vulkan0`) tambem esta em Vulkan0 — `cross_backend=0` em todas as
execucoes. A saida oculta vive no **mesmo backend** do kernel que a escreve.

**Conclusao: LSE_PTR nao e a causa.** Identidade, alocacao, nao-aliasing,
ordem e lifetime do alvo estao provados corretos pos-alocacao.

Limite: isto prova o **mapeamento**; nao prova que o kernel escreveu o valor
certo. Proximo passo: inicializacao de scratch e o **dispatch real do kernel**
(grid/split/reducao) no caminho que produz `local O`, apos ja termos operandos,
descritor, oraculo, store, manifesto e alvo LSE todos medidos identicos.

## 20261006 (9) — Discriminador de kernel/execucao: primeira divergencia = mergeLSE

`F4_RP_KDISC=1` no menor caso (`D128 Q1 NATIVE`), mesmo binario, 6 processos.
`la_b`/`la_t` (LSE do corpo e da cauda) foram içados para escopo de funcao.

### A. Sincronizacao forcada NAO muda nada (ordem de stream descartada)

Cada execucao le os intermediarios **(a)** logo apos o compute (como sempre) e
**(b)** apos `ggml_backend_synchronize(cuda|vk)`:
```
KDISC-SYNC baseline-vs-barrier differ=0 -> stable under barrier   (todas as 12 leituras)
```
A ordem topologica ja tinha mostrado **estrutura** de dependencia; agora o
barrero forcado mostra que **o completamento observado** tambem nao varia.
Conclusao estreitada: **nao e ordenacao de stream.** (Nao prova ausencia de
corrida: barrier estabiliza leitura, nao escrita concorrente.)

### B. Primeira divergencia intermediaria identificada (nao so o O final)

Ordem de escrita real (PRE antes do compute, POST depois — o bloco POST foi
movido para DEPOIS do compute apos eu o ter colocado antes por engano):
```
build0: PRE merge=d852dcb87b6f12a3 -> POST merge=ba7e8550e2b07ac1 v0=7.99717331
build1: PRE merge=ba7e8550e2b07ac1 -> POST merge=6ae5cc697164afc8 v0=7.99717426
        (o PRE de build1 e exatamente o POST de build0 -> os buffers de saida
         sao REUTILIZADOS entre construcoes, sem reset)
KDISC-FIRST-DIVERGENT: mergeLSE, depois localLSE (mesmo storage, hashes iguais),
                       depois mergedO
```
- `mergeLSE` e `localLSE` **compartilham storage** (`l1c` e view de `lse1`).
- Divergem A vs A2 **primeiro em `mergeLSE`** — ou seja, dentro de
  `ggml_kv_tail_attention_merge` / `merge_two`, antes de `mergedO`.
- Run verde: `v0` 7.99717331 vs 7.99717426 (**~1 ULP fp32**).
- Run vermelho (17 falhas): `v0` 8.49501991 vs 8.50374508 (**8.7e-3**).
- `mergeLSE`/`localLSE`/`mergedO` **antes** do compute tem hash entre-construcoes
  diferente (`3 of 5 differ -> SCRATCH/OUTPUT STATE DIFFERS`): os buffers de
  saida nao sao zerados entre A e A2, entao A2 computa sobre os restos de A —
  so importa se algum produtor nao sobrescrever tudo.

### C. Descoberta nova: `bodyLSE`/`tailLSE` nao tem buffer do scheduler

```
KDISC-PRE-SUM body=0000000000000000 tail=0000000000000000  (todas as execucoes)
```
`grab()` imprime 0 quando `!t->buffer || !t->data` — ou seja, `la_b`/`la_t` nao
foram alocados pelo `alloc_graph`. Mas `merge_two` (`tests/...:517-540`) usa
`la`/`lb` como **src** de `ggml_clamp`/`ggml_exp`... Portanto ou (i) eles nao
entraram no grafo, ou (ii) o alloc nao os cobriu. **Isto nao esta resolvido** e
e o proximo ponto: se o merge consome `la_b`/`la_t` por ponteiro sem aresta
`src[]` efetiva, esta lendo memoria nao alocada / de outra construcao.

### D. Dispatch capturado

`GGML_CUDA_FA_ROUTE_DEBUG=1`:
```
ggml_vk_flash_attn: FA-LSE split_k=1 split_kv=64 ne=(6,1,1) D=128 gqa=3 N=3 KV=32
local route: NATIVE exact-tail FA (body=32 tail=32, exact-F16 tail storage, ring+run descriptor)
```
A rota CUDA do FA KVarN nao imprimiu nada com esse env (so a Vulkan aparece).

### E. Controles preservados
Gates O/LSE originais intactos; runs vermelhos nao descartados (run5 `178/20`,
run6 `178/17`); SAME-GRAPH e A/A2 seguem; sem relaxamento de tolerancia.

## 20261006 (10) — Produtores reais de corpo/cauda capturados; 1a diferenca = mergeLSE

### A. `la_b`/`la_t` sao observadores opcionais, nao entradas do grafo (pergunta respondida)

`ggml_kv_tail_attention_merge` (`ggml/src/ggml.c:5808-5815`) e apenas
`ggml_flash_attn_ext_add_kv_tail(body_attn, ...)`: atribui `src[5..9]` e
**devolve o mesmo tensor**. Logo `fa1 == fa_b` — corpo e cauda estao fundidos em
**UM** no `GGML_OP_KVARN_VIEW`/`FLASH_ATTN_EXT` (node7/node9, op=74). Na rota
nativa o fixture **nunca cria** `la_b`/`la_t` (so a branch `body+tail` nao-nativa
os cria), por isso `KDISC-PRE-SUM body=... (NOT-CAPTURED)`.

O diagnostico agora **falha como incompleto** em vez de imprimir hash zero:
```
KDISC-INCOMPLETE bodyLSE/tailLSE observers are NULL on this route ... NOT captured
FAIL line 1640 (CHECK(g_kdisc_incomplete == 0))   x2 por execucao
```

### B. Produtores REAIS capturados (valores, nao zeros)

Como nao existe tensor corpo/cauda no grafo, foram adicionados **dois FA
diagnosticos** sobre os MESMOS operandos (`mask1b` corpo-com-cauda-mascarada,
`mask1t` cauda-com-corpo-mascarado), rotulados `KDISC-OBSERVER` e usados apenas
com o MESMO binario em ambas as execucoes (o grafo fica perturbado — declarado):

```
bodyO   v0=4.61061192      bodyLSE v0=7.98773479
tailO   v0=7.92379332      tailLSE v0=10.2743616
```
Esses valores sao **iguais na execucao VERMELHA e na VERDE** (run2 vs run5).

### C. 1a diferenca medida entre par passando/falhando = `mergeLSE`

Mesmo binario, mesma entrada (tambem confirmado por `KDISC-EARLIEST-BAD` e pelo
hash dos 9 pais de `fa1`):

| run | resultado | bodyO/bodyLSE | tailO/tailLSE | mergeLSE v0 |
|---|---|---|---|---|
| 2 | RED 182/22 | 4.61061192 / 7.98773479 (igual) | 7.92379332 / 10.2743616 (igual) | **7.99719763** |
| 5 | GREEN 162/2 | 4.61061192 / 7.98773479 (igual) | 7.92379332 / 10.2743616 (igual) | **8.53671074** |

- `KDISC-FIRST-DIVERGENT`: em run2 aponta **`mergeLSE`** (corpo/cauda ja
  identicos); em run5 aponta so `mergedO`; em 4/6 runs vermelhos os 7 niveis
  estao identicos entre A/A2 (o vermelho la e consistente entre construcoes e
  diverge do oraculo).
- Portanto o **primeiro intermediario valido que difere e `mergeLSE`**, ou seja
  dentro do kernel fundido `fa1` (corpo+cauda), apos produtores de corpo e
  cauda medidos identicos.

Limite declarado: o observador de cauda usa `mask1t` do fixture, nao o caminho
production de anel/descritor (`tail_mask_nat`+`query_order`+`run_desc`), logo
`mergeLSE` (do kernel fundido) nao precisa igualar o merge dos observadores — o
que se usou aqui foi a **estabilidade corpo/cauda entre vermelho e verde**, que
e o que localiza a diferenca no merge.

### D. Controles
- `KDISC-COVERAGE poisoned-lanes-still-poisoned=0/1554 -> all lanes rewritten`:
  **cobertura de escrita completa**, logo o reuso de buffer `PRE==oldPOST` e
  confirmado inofensivo (deixou de ser suposicao).
- Tolerancias O/LSE originais inalteradas; runs vermelhos preservados;
  `SAME-GRAPH`/A/A2 intactos; sem fechamento de gate.

## 20261006 (11) — Rota CUDA real de fa1 + descritor vs plano (par vermelho/verde)

### A. Dispatch real (nao inferido)

`F4_RP_KDISC` imprime dentro de `ggml_cuda_flash_attn_ext` (`fattn.cu`) a rota
efetivamente escolhida para o no com `src[5..9]` — identica em todas as
execucoes:
```
KDISC-ROUTE uses_kvarn=1 portable_kvarn_tail=0 lse=1 bodyless=0
q=[128,1,6] k=[128,64,2,1] kt=[128,32,2,1] mask_t=[32,1] qo=[1,1] rd=[38,1]
body_keys=64 tail_keys=32 route=ggml_cuda_flash_attn_ext_tail
```
`portable_kvarn_tail=0` -> **a rota e `ggml_cuda_flash_attn_ext_tail`**
(`fattn-tail.cuh:505`), nao a rota portatil KVarN. Dentro dela a reducao e
`k_flash_attn_ext_tail_partials_merge` (grid `(q_max,n_head,n_active)`,
block 256) — e esse, nao o `merge_two` do fixture, e o merge que importa.

### B. Descritor em VRAM == plano do host; posicoes de anel/query reais

```
KDISC-DESC device(38) vs host_plan(38) differing_words=0
  n_tail=32 n_body=-1 ring_slots=[0,1,...,31] query_order=[0]
  body_map_off=38 packed_body=0
```
- **`differing_words=0`**: o `run_desc` lido da VRAM e identico ao plano host,
  em todas as execucoes, verdes e vermelhas. Descritor descartado como causa.
- `ring_slots=[0..31]` e `query_order=[0]` — posicoes selecionadas, reais.
- `packed_body=0` (`rd->ne[0] == body_map_off == 38`): o mapa do corpo NAO vai
  no descritor; o corpo vem das views KVarN (`ggml_cuda_flash_attn_ext_tail_pass_supported`
  exige exatamente `rd->ne[0] == body_map_offset` quando ha views).
- `n_body=-1` e comportamento esperado: `desc[5]` so e preenchido quando
  `pack_body` e verdadeiro.

### C. Par passando/falhando com o MESMO binario instrumentado

| run | resultado | 1a diferenca A/A2 |
|---|---|---|
| 1 | `168 checks, 2 failures` | `mergedO` |
| 2 | `168 checks, 2 failures` | `mergedO` |
| 3 | **`188 checks, 22 failures`** | **todos os 7 niveis IDENTICOS** |
| 4 | `168 checks, 2 failures` | `mergedO` |

Os 2 failures dos runs 1/2/4 sao **exclusivamente** `L1640 =
CHECK(g_kdisc_incomplete == 0)` — a assercao de captura faltante, **nao** uma
falha numerica. **Esses runs nao sao "verdes":** sao numericamente limpos nas
portas O/LSE mas o diagnostico segue INCOMPLETO (a captura corpo/cauda interna
ainda nao existe), e isso e reportado como tal.

No run vermelho (3) A/A2 estao **integralmente identicos** nos 7 niveis, ou
seja: la o resultado e consistente entre construcoes e diverge do oraculo.
Nos runs limpos a 1a diferenca e `mergedO`.

### D. Estado
Rota, contagens de chave, slots de anel, query_order e descritor sao
**iguais** entre passando e falhando -> nao sao o discriminador. Resta o valor
computado dentro de `ggml_cuda_flash_attn_ext_tail` (softmax max/normalizador
antes da combinacao) — que e captura interna de dispositivo, ainda nao medida.

## 20261006 (12) — DEFEITO ENCONTRADO E CORRIGIDO: `op_params` indexado como bytes em `fattn-tail.cuh`

### A. Cadeia medida, no menor caso (`D128 Q1 NATIVE`), mesmo binario

`KDISC-DOT[0]` compara o **dot escalar recalculado dos bytes reais em VRAM**
(`ggml_backend_tensor_get` de `kt_r` e `Q`, strides `nb[]` exatos) com o dot que
o kernel calculou em warp:

```
d_dot=1.714e-06  oob=0  kt_bytes=16384 logical=16384 padded=0
kt_ne=[128,32,2,1] kt_nb=[2,256,8192,16384] q_ne=[128,1,6] q_nb=[4,512,512,3072]
```
→ **operandos e reducao do warp CORRETOS** (1.7e-6 = arredondamento fp32).
A divergencia aparece depois: `score0` != `dot*scale + slope*mask`.

Antes da correcao (par vermelho/verde, mesmo binario):
```
pass: score0=6.27619505  scalar=6.27619552  d_score=2.98e-09
pass: score0=6.26758575  scalar=6.27619552  d_score=8.61e-03
FAIL: score0=1.14470154e-23 / 6.78838694e-21   d_score=6.276
```

### B. Causa

`ggml/src/ggml-cuda/fattn-tail.cuh:1041-1043` lia `op_params` como se fosse
`char*`, mas `ggml_tensor::op_params` e **`int32_t[GGML_MAX_OP_PARAMS/4]`**
(`ggml.h:760`): `p + N` avanca `4*N` bytes.

| variavel | codigo antigo | byte lido | conteudo real | correto |
|---|---|---|---|---|
| `scale` | `op_params + 0*4` | 0 | `scale` | 0 (ok) |
| `max_bias` | `op_params + 1*4` | **16** | `KVARN_DOMAIN` (i32[4]) | 4 |
| `logit_softcap` | `op_params + 2*4` | **32** | **`LSE_PTR` = ponteiro host** | 8 |

O escritor e `memcpy(op_params, {scale,max_bias,logit_softcap}, 12)` ->
bytes 0,4,8 (`ggml.c:5651`). `LSE_PTR` (`ggml.h:475`, indice 8) fica em byte32,
ou seja **dentro de `op_params`**, e e um ponteiro host.

Consequencia: com `lse1 = ggml_flash_attn_ext_lse_out(gctx, fa1)` (o que a rota
nativa faz), `logit_softcap` vira os 32 bits baixos de um endereco e o kernel
aplica `score = softcap*tanh(score/softcap)` com essa constante lixo. Os bits
baixos variam por processo -> **a nao-determinismo 0<->1060 observado durante
toda a campanha**. `max_bias` lia `KVARN_DOMAIN`=1, que como float e ~1.4e-45
(por isso `slope` parecia 1 e passava despercebido).

### C. Correcao minima (provada)

```cpp
memcpy(&scale,         dst->op_params + 0, sizeof(float));
memcpy(&max_bias,      dst->op_params + 1, sizeof(float));
memcpy(&logit_softcap, dst->op_params + 2, sizeof(float));
```

### D. Validacao (gates originais intactos, 3 execucoes consecutivas)

```
[r1] exit=0 8490 checks, 0 failures
[r2] exit=0 8490 checks, 0 failures
[r3] exit=0 8490 checks, 0 failures
native NRMSE: 0.000e+00 2.048e-04 2.049e-04 2.502e-04 6.311e-04 6.954e-04  (identico nas 3)
both-empty: 0.000e+00 | NRMSE=0.000e+00: 18 linhas | SAME-GRAPH bit-identical: 70
KDISC-DOT apos correcao: score0=6.27619553 vs scalar 6.27619552, d_score=2.980e-09
```
`kGateNrmseO=1e-3`, `kGateMaxAbsO=kGateMaxAbsLse=1e-2` **sem alteracao**;
`la_b`/`la_t` seguem reportando INCOMPLETO (2 assercoes de captura), nao falha
numerica.

### E. Defeitos proprios corrigidos nesta rodada
- `KDISC-DOT` dereferenciava `Q->data` (ponteiro device) no host -> copia via
  `ggml_backend_tensor_get` para ambos os operandos, com checagem de bounds.
- `%zu`/`%ld` com contagem de argumentos errada em `fattn.cu` e `kvarn-wht.cu`
  (2 avisos `-Wformat` corrigidos; restam 5 avisos cosméticos `-Wcast-qual`
  x3, `-Wmissing-declarations`, `-Wsign-compare`, todos no codigo de diagnostico).
- Captura `g_kdisc_dot0/score0` era gravada por todos os `n_head` blocos (corrida)
  -> guardada para o bloco (0,0,0); sem isso `score0` misturava cabecas.
- Exit code do processo verificado explicitamente (`exit=0`), nao inferido de
  impressoes ausentes.

## 20261006 (13) — MILESTONE: defeito do `op_params` fechado com regressao focada

### A. Regressao que fecha o defeito
`tests/test-flash-attn-ext-params.cpp` (registrada em `tests/CMakeLists.txt`, label `main`),
**host-only, sem GPU**, exercita o leitor REAL (`ggml_flash_attn_ext_get_alibi_params`)
com `max_bias` e `logit_softcap` explicitamente NAO-nulos, com os dois zerados, e com
`ggml_flash_attn_ext_lse_out` anexado:

```
test-flash-attn-ext-params: PASSED (0 failures)   exit=0
```
Cada caso afirma tambem que a expressao legada `op_params + N*sizeof(float)`
**NAO** reproduz os valores (`legacy '+N*sizeof(float)' reader must NOT reproduce
the values`) — ou seja, se o leitor antigo voltar, a regressao falha.
Cobertura adicional: `scale` continua legivel no byte0; os tres floats sobrevivem
ao anexo de `LSE_PTR`; `LSE_PTR` resolve para o tensor `lse` anexado; o slot de
`LSE_PTR` nao sobrepoe os 12 primeiros bytes.

### B. Leitor compartilhado (novo ponto unico)
- `ggml_flash_attn_ext_get_alibi_params()` — `ggml.h` (GGML_API) + `ggml.c`.
- `fattn-tail.cuh` (o leitor que estava errado) e o proprio diagnostico em
  `fattn.cu` agora chamam essa funcao. **Auditoria da mesma aritmetica:**
  - `fattn.cu:1060-1062` tinha EXATAMENTE o mesmo erro no meu diagnostico
    (`op_params + N*sizeof(float)`) — corrigido para a funcao compartilhada;
  - `fattn.cu:1070` usava `(const char*)op_params + 2*sizeof(int32_t)` = byte8
    correto, agora `op_params + 2`;
  - demais usos `op_params[i]` em ggml-cuda (`acc.cu`, `argsort.cu`, `norm.cu`,
    `roll.cu`, `mmvf.cu`, `kvarn.cu`, `ggml-cuda.cu`) sao indices i32 corretos;
  - nenhum outro `+ N*sizeof(float)` no repo.

### C. Assercoes de diagnostico obsoletas removidas
- `la_b`/`la_t` **nao existem** na rota nativa (`ggml_kv_tail_attention_merge`
  funde tudo em `fa1 == fa_b`) -> a assercao de captura incompleta foi removida;
  a cobertura do diagnostico agora vem do registro do kernel
  (`ggml_cuda_kdisc_rec_count() > 0`, exigido com marker `0xAA55AA51`).
- Pais `KVARN_VIEW` sao **virtuais por projeto** (`data=(nil)`, resolvidos dentro
  do kernel) -> deixaram de contar como captura faltante.
- **Nenhum gate numerico foi enfraquecido.**

### D. Validacao com execucao LIMPA (sem flags de debug)
```
full matrix, sem nenhum F4_RP_*:
  exit=0  8490 checks, 0 failures | linhas FAIL=0 | impressoes KDISC=0
regressao: exit=0 PASSED
import vs same-representation reference: 248320 entries, NRMSE=0, max_abs_diff=0
branch gates: F4_DRIVER_LAYER=3  -> all branch gates passed (exit 0)
branch gates: F4_DRIVER_LAYER=11 -> all branch gates passed (exit 0)
```
`test-kv-mixed-destination-intact` segue em **141 checks / 12 failures** — igual ao
estado registrado no checklist canonico (item 1 e 4), portanto **nao regredido
nem resolvido por este fix**: essas 12 falhas LIVE seguem sendo o proximo gate.

Nenhuma impressao de runtime nem sincronizacao de diagnostico fica ativa em
producao: todo o bloco KDISC esta atras de `F4_RP_KDISC`; a matriz limpa acima
provou `KDISC prints=0`.

### E. Fonte / evidencia
- Causa-raiz: `ggml/src/ggml-cuda/fattn-tail.cuh` (leitor de `op_params`), corrigido.
- Regressao: `tests/test-flash-attn-ext-params.cpp`.
- Evidencia: `docs/occupancy-evidence/f4-clean-matrix-*.log`,
  `f4-regression-params-*.log`, `f4-import-gpu-*.log`, `f4-branch-layer3-*.log`,
  `f4-branch-layer11-*.log`, `f4-dot{1..3}-*.log`, `f4-postfix-r{1..3}-*.log`.
- Marcos preservados: tres `8490/0` (f4-postfix-r1..3) + `8490/0` limpo.

## 20261006 (14) — Gate LIVE: 12 -> 8; cobertura de records restaurada; 8 restantes medidos

### A. Liveness derivada do descritor (nao presumida)
O teste agora deriva, por n, se stage e records sao representacoes vivas a partir
do contrato do seal eager (mesmas regras de `from_stage`/`from_record` de
`ggml_cuda_op_kvarn_materialize` e `fattn-mma-kvarn`):
```
liveness n=1    live_cells=2 live_group=0 -> stage_consumed=1 records_consumed=0
liveness n=128  live_cells=129 live_group=1 -> stage_consumed=1 records_consumed=0
liveness n=129  live_cells=130 live_group=1 -> stage_consumed=1 records_consumed=0
liveness n=257  live_cells=258 live_group=2 -> stage_consumed=1 records_consumed=1
```
`record group 0` e o sink permanente do stage e **nunca** e lido; um record selado
exige `live_group >= 2`, i.e. n >= 257. `n=257` foi adicionado ao laco para
**manter cobertura de records**.

### B. Defeito real corrigido no probe (production)
`llama_kv_cache_kvarn::probe_head_span` com `live=true` escolhia
`own.front()/KVAR_N_GROUP` para records — que e **sempre o grupo 0**, o grupo que
nao e consumido. Resultado: nenhuma mutacao "LIVE" de record alcancava a FA.
Corrigido para selecionar a maior celula propria com `1 <= grupo < live_group`
e retornar `false` quando nenhum record selado existe.

### C. Resultado (12 -> 8)
```
n=257 k_records head=0  B(live) max=0.123102   C(inactive) max=0
n=257 v_records head=0  B(live) max=0.104003   C(inactive) max=0
n=257 k_records head=1  B(live) max=0.137817   C(inactive) max=0
n=257 v_records head=1  B(live) max=0.128935   C(inactive) max=0
n=128 k/v_stage  head0+1  B(live) 0.10..0.14  C(inactive) max=0
n=129 k/v_stage  head=0    B(live) 0.10..0.13  C(inactive) max=0
```
**4 falhas de records fechadas por correcao real**; controles negativos
(`C(inactive) max=0`) e os gates `A=A2 max=0` / `A=D max=0` (rollback de destino)
permanecem em 0; `import NRMSE=0` preservado em todos os n.

### D. Restam 8 (medidos, nao renomeados)
```
n=1    k_stage/v_stage head=0 e head=1   (B(live) max=0)
n=129  k_stage/v_stage head=1            (B(live) max=0)
n=257  k_stage/v_stage head=1            (B(live) max=0)
```
`group=0` em todos (o alvo do probe e `own.front()`), `wrote=1` em todos.
`n=128` funciona para os dois heads; `n=129/257` funcionam so para head0;
`n=1` falha para os dois. A diferencapor head com o MESMO byte alvo mostra que a
causa nao esta no offset escolhido, mas em **o que a FA consome naquela janela**:
o log regista `KVarN exact tail: effective=128 representation=overlay`, ou seja
os ultimos 128 tokens vivem no overlay de cauda exata e nao no stage — coerente
com `n=1` falhar (unico token dentro da janela) embora nao explique ainda o
padrao por head. **Proximo passo**: mapear, por n/head, se pos0..k saem do
overlay de cauda exata ou do stage, e mutar a representacao realmente consumida.

### E. Milestones preservados nesta rodada
```
full matrix, sem flags: exit=0 8490 checks, 0 failures | FAIL=0 | KDISC prints=0
test-flash-attn-ext-params: PASSED
F4_DRIVER_LAYER=3  -> all branch gates passed
F4_DRIVER_LAYER=11 -> all branch gates passed
import vs same-representation reference: 248320 entries, NRMSE=0, max_abs_diff=0
```

## CHECKPOINT: F5/F7 handoff (reuso obrigatorio apos F4) — 2026-10-06

Lido e confirmado `docs/occupancy-aux-handoff-20261005.md`; artefatos em
`/home/hjotha/occupancy-aux-20261005/`. **Preparacao JA CONCLUIDA pelas
sessoes auxiliares — NAO reimplementar.**

| | origem | SHA256 (patch) | testes CPU |
|---|---|---|---|
| F5 benchmark/telemetry | `deliverables/f5-preparation.patch` | `062b8f840a643a17b2e4b6d05444a502c11e992901d01aec183167190084a49b` | 55 |
| F7 rollback/provenance | `deliverables/f7-preparation.patch` | `dd8c0c88281b0f9b8a13920a240e23fdce424ebbbbb7cf427b5b4f1d025df7ee` | 93+66 = 159 |

Fontes: `f5-benchmark/` e `f7-rollback/` sob a raiz acima (base: snapshot sujo de
`beellama.cpp`, branch `feat/occupancy-position-split`, HEAD `d9db0b881`).

**Reuso ao entrar em F5/F7:** revisar compatibilidade com o build final e
integrar/reusar `tools/occupancy-f5/f5_harness.py` (55 testes) e o material F7
(`elfdeps.py`, inventario, readiness), **revalidando apos a integracao**.
Sao artefatos de PREPARACAO: a campanha real de GPU de desempenho/qualidade/
capacidade e a aceitacao real de rollback/canary continuam OBRIGATORIAS e ainda
nao foram feitas. Preservar o trabalho F4 atual.

## 20261006 (15) — Par A/B pareado + controle positivo NAO-degenerado de head1

### A. Comparacao pareada (A sem mutacao x B mutado), mesma posicao/dimensao
Prefixo, estado e token-de-prova identicos; comparacao apenas entre amostras
FRESCAS de identidade casada (route/range/stored/offset/backend/shape):
```
B head=0  KPAIR pos=0 max|d|=0.0424804688  MUTATED-CELL
B head=0  KPAIR head=1 pos=0 max|d|=0       UNRELATED        (bit-identico)
B head=1  KPAIR pos=0 max|d|=0.0617675781  MUTATED-CELL
B head=1  KPAIR head=0 pos=0 max|d|=0       UNRELATED        (bit-identico)
KROUTE-VERDICT matched=2 stale=0 identity_mismatch=0 head0=1.93261719 head1=2.3046875
  -> head1 consumer row CHANGED (mapping reaches the consumer)
```
- A celula mutada muda; celulas NAO relacionadas ficam **bit-identicas** (0).
- `pos=1` ficou `UNAVAILABLE (A.ok=0 B.ok=0)` — sem captura de pos1 no caminho
  atual; registrado, nao interpretado como igualdade.
- O `max|d|=2.3046875` anterior **nao** e efeito da mutacao (incluia o token de
  prova recem-escrito); o efeito medido da mutacao e `0.0425` (head0) /
  `0.0618` (head1) na celula exata. **Nenhuma hipotese de mapeamento foi
  declarada excluida** — so que a linha consumida de head1 muda.

### B. Controle positivo NAO-degenerado para head1
Fixture de atencao deterministica limitada sobre as linhas REAIS capturadas
(Q = a propria linha A de head1/pos0, mascara = posicoes 0 e 1 visiveis, V nao
uniforme):
```
KPOSCTRL n=129 v_stage head=0 det_dO=0.0424804688 NONDEGENERATE
KPOSCTRL n=129 v_stage head=1 det_dO=0.0617675781 NONDEGENERATE
```
**head1 agora tem controle positivo com dO calculavel e nao-nulo.**

### C. O que continua aberto (1 falha preservada, nada deletado)
```
FAIL n=129 v_stage head=1: a LIVE mutation produced identical logits (max=0)
```
A linha consumida de head1 muda (`0.0618`) e a atencao deterministica da
`dO=0.0618`, mas os logits do MODELO nao se movem -> **o peso de atencao do
modelo nessa linha e ~0 (estimulo degenerado no nivel do logits)**. O CHECK
original foi **mantido** (nao enfraquecido nem removido): `105 checks, 1 failure`
continua aberto ate que se escolha uma posicao com contribuicao real de head1
(`pos=1` nao tem captura atualmente). Cobertura de representacao real e controles
inativo/restore/vecinhos permanecem.

### D. Gates e invariaveis
`kGateNrmseO=1e-3` / `kGateMaxAbsO=kGateMaxAbsLse=1e-2` inalterados;
`C(inactive) max=0` intacto; import/full-logit fidelity separado e intacto;
overlay positive coverage **aberto**; F5/F7 handoff presente; producao restaurada
e verificada com 27B real apos o lote (`http=200`, `17*9 -> 153`).

## CHECKPOINT: criterios -> evidencia F4 (§3.6 + §4.1) — 2026-10-06 21:30

**Escopo F4 (obrigatorio).** `docs/occupancy-placement-plan-20261002.md` §3.6
(linhas 173-207) + §4/4.1 (233+): layout/envelope `position_split` e faixas
`[0,P)`/`[P,C)`, transacao restore (KV + tabela logica + checkpoint DeltaNet),
restore/import exatos, XXL somente-disco em **processo novo** com tokens
efetivamente reutilizados e **sem prefill do prefixo**, snapshot invalido /
P diferente / ocupacao > capacidade rejeitados com **estado destino intacto e
fallback explicito**, spool em disco com publicacao atomica. Limites numericos
O/LSE/KL **fixos**: NRMSE<=1e-3, |O|max<=1e-2, |LSE|max<=1e-2, KL media<=1e-4
nat/token e p99<=1e-3 (`tests/test-position-split-real-pair.cpp:84-86`).

**NAO e gate de promocao.** A expectativa "toda perturbacao de um byte em pos0
tem que mudar os logits do modelo inteiro" e **scaffolding extra do fixture, nao
contrato de produto**: a contribuicao de atencao pode ser efetivamente zero. Caso
`n=129 v_stage head=1` permanece como **diagnostico nao-resolvido/degenerado**,
com resultado real registrado abaixo — **nao afirmar que passou**.

### Evidencia preservada (positivos representativos por representacao/cabeca)

| criterio | evidencia | resultado |
|---|---|---|
| matriz canonica sem `F4_RP_*` | `tests/test-position-split-real-pair.cpp` | `8490 checks, 0 failures` (3 corridas) |
| import vs mesma representacao | mesma | `248320 entries, NRMSE=0, max_abs_diff=0` |
| rotas branch camada 3 e 11 | `F4_DRIVER_LAYER=3` / `=11` (`tools/f4-branch-driver.py`) | `all branch gates passed` |
| params FA `op_params` int32-slot | `tests/test-flash-attn-ext-params.cpp` | `61 ok, 0 failures` |
| geometria do stage (contrato do writer) | idem | offsets `0/512` -> `row_width=256`, `n_cells=384`, `elem=98304`, `nbytes=196608` |
| envio v2 / faixas / versao | `tests/test-kv-mixed-state.cpp` | `409 checks, 0 failures` (inclui gap/overlap na fronteira P, versao desconhecida recusada, frame v2 truncado, `duplicate layer ownership`) |
| publicacao atomica / overflow / fault | `tests/test-kv-mixed-state-stream.cpp` | `819 checks, 0 failures` |
| io, commit/cancel | `tests/test-kv-mixed-io.cpp` | `20225 checks, 0 failures` |
| handoff, transacao local/remote, wiring errado | `tests/test-kv-mixed-handoff.cpp` | `ALL PASS` |
| capacidade/plano | `tests/test-kv-mixed-placement.cpp` | `618 checks, 0 failures` |
| records selados positivos | `docs/occupancy-evidence/f4-instant-*.log` | `n=257 k_records h0 0.123102`, `v_records h0 0.104003`, `k_records h1 0.137817`, `v_records h1 0.128935`, `C(inactive) max=0` |
| stage positivos | idem | `n=128 k/v_stage head0+1` e `n=129 k/v_stage head=0` com `B(live) 0.10..0.14`, `C(inactive) max=0` |

**Diagnostico nao-resolvido (nao e gate):** `n=129 v_stage head=1` ->
`logits delta 0`, `consumer delta localizado 0.06177`
(`docs/occupancy-evidence/f4-instant-20261006-230918.log`, `109 checks, 1
failure`). `n=257` exige `live_group>=2`; `n=129 live_group=1` nao tem records
para consumir (`records_consumed=0`), comportamento previsto pelo contrato de
eager-seal.

### Gaps F4 restantes (somente obrigatorios)

1. **Restauracao XXL somente-disco em processo novo**: hit/importacao, tokens
   efetivamente reutilizados, ausencia de prefill do prefixo — **nao executado**.
2. **Falha de importacao com estado destino intacto + fallback explicito de
   cache miss/reprocessamento** em processo novo — nao executado.
3. **Rejeicao de P diferente, snapshot invalido e ocupacao > capacidade no
   caminho real**, nao so em unitario de placement — parcial.
4. **Transacao restore KV + tabela logica + checkpoint DeltaNet**: sufixo
   divergente e rollback antes/em/depois de P com estado recorrente —
   `Prioridade 3` **NAO INICIADA**.
5. **Matriz negativa target/draft/carry**: falha de captura, falha apos
   clear/touch, empty/carry-only, P/modelo/checksum/metadados — **nao executada**.
6. **`manifest.body_only` em `state_v2_read_payload_and_install`** — assert de
   especificacao ainda ausente (sinalizado por `test-std-kv-tail-static`).
7. **Cobertura positiva de overlay do tail exato** — apenas se necessaria para
   provar participacao efetiva do estado; `probe_head_span` nao tem caminho de
   mutacao do tail (`layer.k_tail` e 2D).

**Preservado:** `8490/0`, `import NRMSE=0`, gates `1e-3/1e-2/1e-2`, geometria por
offsets, isolamento de producao (stop/restore com trap), trabalho de branch/dirty,
handoff F5/F7 (`docs/occupancy-aux-handoff-20261005.md`, F5 patch
`062b8f840a643a17b2e4b6d05444a502c11e992901d01aec183167190084a49b`, F7 patch
`dd8c0c88281b0f9b8a13920a240e23fdce424ebbbbb7cf427b5b4f1d025df7ee`) — **reusar,
nao reimplementar**.

### Execucao F4 — gap 5 (matriz negativa target/draft/carry) — 2026-10-06 21:45

Novo `test_owner_fallback_matrix()` em `tests/test-kv-mixed-state.cpp`:

- **owner `{target, draft_mtp}` x falha em `camada 1` / `camada 2` / nenhuma**:
  falha => `abort==1`, `finish==0`, `dest==0` (estado destino intacto =>
  fallback explicito de cache miss/reprocesso), e **segundo `commit` lanca**
  (ticket consumido, nenhum restore parcial); sucesso => `staged==2`,
  `finish==1`, `abort==0`.
- **falha de stage de metadata** para cada owner: `abort==1`, `finish==0`,
  `dest==0`.
- **carry (partial overlay + resident reference)** para cada owner, falha na
  camada do overlay e na camada de referencia: `abort==1`, `finish==0`,
  `dest==0`, `refs_asked` = 0 (parou na camada que falhou) ou 1; caminho de
  sucesso => overlay staged com `n==po.size()` e `ptr!=nullptr`, referencia
  com `ptr==nullptr && n==0`.
- **invariante de ordem**: `finish` nunca antes de todo stage OK; `abort`
  nunca apos `finish` (`CHECK(finished == 0)` dentro do hook de abort) e
  nenhum hook de stage toca `dest`.

Resultado real: `build=0`, `./build-optimized/bin/test-kv-mixed-state` ->
`PASS: 523 checks, 0 failures` (antes: `409 checks, 0 failures`; +114 checks,
nenhum CHECK removido). Reexecutados sem mudanca: `test-kv-mixed-state-stream`
`819/0`, `test-kv-mixed-io` `20225/0`, `test-kv-mixed-handoff` `ALL PASS`,
`test-kv-mixed-placement` `618/0`.

Ainda aberto (proximos, nesta ordem): gap 1 (XXL so-disco em processo novo
com tokens reutilizados e sem prefill), gap 2 (falha de importacao com destino
intacto + fallback), gap 4 (transacao DeltaNet, sufixo divergente e rollback
em P-129/P-128/P-1/P/P+1), gap 3 (P diferente/capacity no caminho real),
gap 6 (`manifest.body_only`), gap 7 (overlay do tail exato, somente se
necessario).

### Execucao F4 — gaps 1 e 2 (real path, processo novo) — 2026-10-06 23:47/23:52

Modelo real `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf` (10442827776 B),
`llama-completion`, `-c 4096 -b 512 -ub 256 --flash-attn on --cache-type-k
kvarn4 --cache-type-v kvarn4`. Producao parada e restaurada com **verificacao
real** (`/v1/models` id + chat completion) em cada script (`/tmp/opencode/gap1.sh`,
`/tmp/opencode/gap2.sh`, traps).

**GAP 1 — PASSOU.** Snapshot XXL/KVarN so-disco em processo novo:

| caso | log | evidencia |
|---|---|---|
| A (processo 1, cria) | `f4-gap1-A-20261006-234706.log` | `session file does not exist, will create` + `saved session before last token to ... n_new = 1329`; arquivo `258811173` B |
| B1 (processo NOVO, so o snapshot, prompt identico) | `f4-gap1-B1-...log` | `loaded a session with prompt size of 1329 tokens` + `session file has exact match for prompt!` + `replayed last token from session` => **0 tokens de prefill do prefixo**, apenas 1 replay |
| C (processo NOVO, snapshot + prompt maior) | `f4-gap1-C-...log` | `session file matches 1323 / 2079 tokens of prompt` => **reutilizados 1323**, reavaliados `2079-1323 = 756` (a diferenca 1329 vs 1323 e fronteira de BPE na juncao base+ext, nao defeito de estado) |

**GAP 2 — PASSOU** (matriz de corrupcao no caminho real `llama_state_load_file`),
evidencia `docs/occupancy-evidence/f4-gap2-corruption-20261006-*.txt`:

| caso | exit | rejeitou | loaded/replayed |
|---|---|---|---|
| controle integro | 0 | nao | si/si |
| magic | 1 | **sim** | nao/nao |
| version | 1 | **sim** | nao/nao |
| tokencount | 1 | **sim** | nao/nao |
| trailing-garbage | 1 | **sim** | nao/nao |
| mid-payload (offset 1000000) | 0 | **NAO** | si/si |
| tail-last-byte | 0 | **NAO** | si/si |

- `fail-load` (magic): exit=1, `rejected=1`, `replayed=0` -> **nenhum decode roda,
  destino nunca recebe estado semi-instalado**; sha256 do snapshot `e602c98eca29b544`
  inalterado antes/depois de todas as tentativas (preservado em falha).
- `miss-reprocess` (fallback explicito): snapshot removido -> exit=0 com
  `session file does not exist, will create` + `saved session before last token`
  => cache miss/reprocessamento explicito.
- Producao restaurada e verificada: `active gpu=1 model=qwen-3.8-27b reply='42'`.

**FALHA REAL registrada (sem waiver) — GAP 2b:** o caminho de sessao
`llama_context::state_load_file` -> `llama_kv_cache` state v2 nao tem nenhuma
integridade de payload: `grep -c 'crc|checksum|hash' src/llama-kv-cache.cpp` = 0.
Corrupcao de um byte no meio do payload e aceita silenciosamente (exit 0,
`loaded a session`, `replayed last token`). So cabecalho (magic/version/count)
e trailing-garbage sao rejeitados. §3.6 exige "integridade do payload" por
descritor: **nao aprovado**, pendente de implementacao de checksum no manifesto
v2 (mudanca de formato, compartilhada — fora do escopo "somente diagnostico").

Caso `n=129 v_stage head=1` continua sendo diagnostico nao-resolvido; nao reaberto.

### F4 — correcao de criterio e estado real (2026-10-07 00:17-00:40)

**Criterio corrigido pelo revisor: gap1 e gap2 continuam PARCIAIS.**
O que foi medido e`legacy session smoke` (mesmo layout KVarN, 4096 ctx,
`llama-completion --prompt-cache`): prova hit/reuso/prefill no formato de sessao
antigo, **nao** o contrato do §3.6 linhas 187-203 (descoberta no indice de disco
de um snapshot XXL puro + **importacao explicita** num destino `position_split`
com P/C e faixas local+overflow, com a conversao autorizada reconhecida pelo
indice). `gap2` exit=1 sem decode so prova que o source nao foi alterado (sha256
inalterado), **nao** que um destino ja populado permaneceu integro; e
"missing-file reprocess" nao e fallback apos rejeicao de importacao.

**Tambem corrigido:** a integridade de payload **nao** esta ausente no formato
certo. `src/llama-kv-mixed-state-stream.cpp` tem **XXH64 por payload**,
`header_crc = XXH64(header)`, descritores de manifesto e validacao
(`pass1/pass2_checksums`, `d.payload_checksum`). O `grep` em
`llama-kv-cache.cpp` refere o formato de sessao legado — caminho diferente.
O teste de corrupcao canonico tem de ser feito no envelope misto-v2, nao no
legacy; redesign do formato legado **fora de escopo**.

#### Correcoes de harness aplicadas (obrigatorias para proximas rodadas)
- `kill -0 $TESTPID` em toda espera; saida imediata se o filho morrer.
- Prereq de GPU com guarda explicita; **aborta para `restore` na primeira falha
  de launch** e **nunca repete a matriz** depois que o prereq compartilhado caiu.
- Readiness em duas fases: health do ROUTER ≠ modelo servido; agora espera
  `/v1/models` listar `qwen-3.8-27b` antes de considerar launch ok.
- A verificacao de `position-split ... P=` so pode ser feita **apos a primeira
  requisicao** (router mode constroi o contexto sob demanda).
- Testes de split usam **preset de teste isolado**
  (`/home/hjotha/f4-spool/test-split-preset.ini`, copia do preset de producao)
  com `slot-save-path` proprio e port 8091; flags interdependentes vao pelo
  mesmo caminho preset->CLI do router, sem flags parciais.

#### Cadeia de precondicoes fail-closed resolvida ate criar o contexto split
Registro real (cada uma encerrou a matriz; nenhuma repetida):

| # | erro observado | correcao |
|---|---|---|
| 1 | `position-split KV exige cache remoto Q4` | `remote-attn-cache-type-k/v = q4_0` no preset |
| 2 | `--remote-attn-min-context requires adaptive context` / `requires --remote-attn` | flag so dentro do preset, nunca CLI parcial |
| 3 | `exige todas as 16 camadas full no backend remoto (recebido 1)` | `remote-attn-layers = full` |
| 4 | `position-split KV nao suporta contexto MTP` + `adaptive context requires draft-mtp` | MTP/spec desativados e tiers substituidos por `ctx-size = 4096` (§3.7: sem MTP) |
| 5 | `--mtp-max-tokens requires --ctx-size-mtp` | mtp/spec linhas desativadas |
| 6 | `per-layer remote KV formats require ... global KVarN cache ...` | `cache-type-k/v = kvarn4` globais (tiers desativados) |
| 7 | **contexto split criado** (nenhum erro fail-closed) | — |

#### FALHA REAL observada no destino split (nao waivereada)
Com o contexto split criado, a primeira requisicao caiu:
```
W decode: failed to find a memory slot for batch of size 4 ... off = 212
../ggml/src/ggml-cuda/fattn-tail.cuh:713: GGML_ASSERT(mt->ne[0] > 0 &&
  kt->ne[1] + (kt_current ? kt_current->ne[1] : 0) >= mt->ne[0]) failed
E srv operator(): http client error: Failed to read connection
```
Evidencia: `docs/occupancy-evidence/f4-gap1d-P256-20261007-*.log`.
Isto e`FAIL` para o gap1 (destino split nao serviu uma requisicao) — nao e
preenchido por nenhum dos casos legados. Nao mexido agora: e`ggml-cuda` compartilhado,
fora do escopo "somente diagnostico", e precisa de analise propria (kernel
fast-tail selecionado com tail vazio/kv-tail-tokens=0).

#### Producao
Restaurada e verificada apos **cada** rodada: `active`, `health=200`,
`model=qwen-3.8-27b`, `reply='42'`, `gpu=1`. Nenhum processo de teste sobrou
(ultima checagem: apenas os processos proprios do preset de producao nas portas
8090/43845).

#### Matriz F4 atual (§3.6/§4.1) — com estado honesto
| caso | estado | evidencia |
|---|---|---|
| `test-kv-mixed-state` (incl. `test_owner_fallback_matrix` target/draft/carry) | **PASS** 523/0 | build-optimized |
| `test-kv-mixed-state-stream` (XXH64, manifest, fault) | **PASS** 819/0 | build-optimized |
| `test-kv-mixed-io` / `-handoff` / `-placement` | **PASS** 20225/0 · ALL PASS · 618/0 | build-optimized |
| `test-recurrent-state-rollback` x4 (DeltaNet) | **PASS** | ctest |
| `test-save-load-state-archs` (112 arquiteturas) | **PASS** | ctest |
| `test-context-lifecycle-{target,prepare-failure,yarn-zero-orig,mtp}` | **PASS** 5/5 | ctest |
| `8490/0`, `import NRMSE=0`, layer3/11, `op_params` fix, geometria F16 | **PASS** | preservados |
| gap1 destino split (P/C local+overflow, indice, conversao autorizada) | **FAIL/nao concluido** | `f4-gap1d-*` assert `fattn-tail.cuh:713` |
| gap2 destino ja populado integro apos import rejeitado + fallback real | **NAO EXECUTADO** | — |
| gap4 transacao DeltaNet com sufixo divergente e rollback P-129..P+1 | **PARCIAL** (rollback unitario PASS; transacao integrada nao executada) | — |

## CHECKPOINT: causa-raiz do split + correcao validada — 2026-10-07 01:34

**Build isolado** `build-f4` (mesmas opcoes de `build-optimized`, ccache ON). A producao
sempre roda `build-optimized` e **nao foi recompilada**; toda rodada de GPU parou e
restaurou producao com verificacao real (`model=qwen-3.8-27b reply='42'`, health=200).

### Diagnostico test-only (caminho de erro ja existente)
`src/llama-kv-cache.cpp` (erro `structured KV allocation ...`): agora tambem imprime
`range`, `head`, `pos`, `pos_np`, `n_pos`, `psplit_p`, `cells`, `n_groups`, `n_seq_max`,
`kind`, `occ=[celula:pos]` e, por grupo, `used/capacity/mixed/own`. Nenhum CUDA nem ABI.
(Saidas `F4-UBATCH-*` por batch usadas durante o diagnostico foram **removidas** depois.)

### Causa-raiz PROVADA (nao era owner mismatch e nao era tail vazio)
- Reproducao deterministica: 35 tentativas identicas, `used=42` **estavel** em todas
  => **prepare falho NAO muta ocupacao/ownership** (resposta a pergunta aberta).
- Owner: `own=[0]`, `mixed=0`, `desired=[0]`, `n_seq_max=1` — **hipotese de owner
  descartada pelos dados**.
- `F4-UBATCH-POST` mostrou `raw=[42,46,50,54,...]` (salto de 4) e `local=216 ovf=0`
  para um batch que claramente cruzava P.
- `n_pos = n_pos_per_embd() = 4` para `LLAMA_ROPE_TYPE_MROPE`
  (`src/llama-hparams.cpp:285`).
- **O layout real e component-major**: `llama-batch.cpp:790`
  `udata->pos[j*n_tokens + i] = batch.pos[...]`; e o proprio cache le `pos[i]`
  (`llama-kv-cache.cpp:3031` `cells.pos_set(idx, ubatch.pos[i])`, `:2572`, `:2677`,
  `llama-batch.cpp:919/922`).
- Todos os pontos de position-split liam `pos[i * n_pos]` — com `n_pos>1` isso le
  outro componente/outra token. Com P desativado `n_pos==1` ou `P==0` escondia o defeito.

### Reparo (contrato, sem relaxar nada)
Trocar `pos[i * n_pos]` por `pos[i]` em 7 pontos:
`llama-kv-cache.cpp:2392` (janela da faixa), `llama-memory-hybrid.cpp:252/280/302`
(divisao em P), `llama-graph.cpp:850` (mascara) e `:4488`
(`use_position_split_attention`), `llama-kv-cache-kvarn.cpp:187`.
Mais `llama_ubatch_subset`: copiava **token-major**
(`pos[i*n_pos + p] = src.pos[t*n_pos + p]`) e entregava um ubatch em layout
diferente do que foi dividido — agora component-major
(`pos[p*n_out + i] = src.pos[p*n_src_tokens + t]`).
Tambem `llama_memory_hybrid` agora valida o P de ambos os lados (antes só quando o
proprio P era !=0; hybrid com P=0 na frente de cache P>0 era aceito em silencio).

### Regressao (falha ANTES, verde DEPOIS)
`tests/test-position-split-ubatch.cpp` caso 5d: ubatch `n_pos=4` component-major com
componentes espaciais distintos (como M-RoPE).
- **Antes do reparo: `2808 checks, 4 failures`** (`tail holds only [P,700)`,
  `head/tail token count`, `divided head holds exactly the local tokens`).
- **Depois: `2808 checks, 0 failures`.**
Sem mudanca para `n_pos==1` (todos os casos existentes continuam verdes).

### Evidencia do split apos o reparo (rodada real, processo novo, P=256)
`local=214 pos0=42 poslast=255` / `ovf=298 pos0=256 poslast=553` / `ovf=214 pos0=554`
— divisao exata em P, sem crossing, sem retry storm, sem `failed to prepare`.

### Validacao canonica apos o reparo
| gate | resultado |
|---|---|
| matriz canonica (`test-position-split-real-pair`, sem `F4_RP_*`) | **`8490 checks, 0 failures, 0 not-executed`** (exit 0) |
| import vs mesma representacao | **`248320 entries, NRMSE=0, max_abs_diff=0`** |
| `test-position-split-ubatch` / `-ranges` / `-merge` | `2808/0`, `391/0`, `51/0` |
| `test-kv-mixed-destination-intact` full | `172 checks, 4 failures` — **identico ao baseline `build-optimized` nao-modificado** (`n=129/257 k_stage/v_stage head=1`), portanto pre-existente, nao regressao |
| gates `1e-3/1e-2/1e-2`, `op_params +0/+1/+2`, geometria F16 | preservados |

### Restante (bloqueio isolado)
`ggml/src/ggml-cuda/fattn-tail.cuh:713`
`GGML_ASSERT(mt->ne[0] > 0 && kt->ne[1] + kt_current->ne[1] >= mt->ne[0])`
agora dispara **limpo**, logo no primeiro batch do caminho dual-range, **sem** nenhum
`failed to prepare` anterior — e o unico obstaculo do destino split. Proximo passo:
rastrear como `build_attn_position_split` alimenta `src[5]/src[7]` (`k_tail_local`,
`mt`) quando a faixa local ainda nao tem linhas de tail, e corrigir a selecao do
kernel fast-tail (sem deletar a assercao).

## NEXT-ACTION CHECKLIST (pos-M-RoPE) — 2026-10-07 02:00
1. [in progress] `fattn-tail.cuh:713` — contrato violado medido com `F4_TAIL_TRACE=1`
   (`ggml.c` dump, host-side, gated):
   `k_tail=[256,129,4,1] v_tail=[256,129] mask_tail=[426,298] run_desc=[432,1]`,
   `n_local=214`, `n_overflow=298`, `tail_tokens=128`.
   `426 = tail_tokens(128) + n_query(298)` e `run_desc 432 = 6 + 426` — a mascara,
   os `tail_read_idxs` e o `run_desc` sao dimensionados COM a faixa "current",
   mas `build_attn_position_split` (llama-graph.cpp:4735) passa
   `k_tail_current = nullptr`, logo `kt->ne[1] + kt_current = 129 < 426`.
   O caminho single-range (llama-graph.cpp:4332-4341) cria a faixa current via
   `prepare_compact_tail_current` / `ggml_concat(..., 2)`. Falta espelhar isso
   em `build_attn_position_split` (nao mascarar a assercao).
   Pendente: confirmar a pre-condicao `k_tail->ne[1] == k_cur->ne[1]` de
   `prepare_compact_tail_current` neste call site antes de aplicar.
2. Regressao do item 1 + rerun `8490/0` e `import NRMSE=0`.
3. F4 real: descoberta no indice de disco + import XXL explicito em processo novo.
4. F4: rejeicao com destino intacto (fingerprint KV/tabela/DeltaNet) + fallback.
5. F4: transacao DeltaNet integrada com sufixo divergente e rollback P-129..P+1.
6. Reusar F5/F7 (`docs/occupancy-aux-handoff-20261005.md`), nao reimplementar.

## CHECKPOINT: fattn-tail.cuh:713 reparado — split real servindo — 2026-10-07 02:14

**Diagnostico com formas reais** (`F4_TAIL_TRACE=1`, dump host em `ggml.c`, gated):
`k_tail=[256,129,4,1] v_tail=[256,129] mask_tail=[426,298] run_desc=[432,1] q=256`.
`426 = tail_tokens(128) + q_max(298)`; `run_desc 432 = 6 + 426`.
A assercao exigia `kt->ne[1] + kt_current->ne[1] >= 426` e o builder de split
amarrava `k_tail_current = nullptr` -> `129 < 426`.

**Leitura da especificacao no proprio fonte** (`llama-graph.cpp:3672-3697`):
> "a position-split OVERFLOW ubatch writes Q4 overflow keys instead, so its local
>  tail is read purely as HISTORY and must carry no current segment."

`validate_native_tail_operation(..., ubatch_writes_local_range)` e chamado com
`true` nos caminhos single-range (4413, 5366) e com **`false`** no split (4770).

**Reparo aplicado (contrato, nao mascarar a assercao):**
1. Binding tentado de `k_tail_current` no split **foi revertido** — violava
   `expect_current = route->has_current && ubatch_writes_local_range = false`
   (`KV tail layer 3 current-segment descriptor does not match the final operation`).
2. Causa real: o **extent da cauda** era calculado com o `q_max` do ubatch
   INTEIRO mesmo quando o grafo nao escreve a faixa local, prometendo `q_max`
   linhas "current" que esse grafo deliberadamente nao amarra.
   Novo `tail_extent_queries(mctx, ub, q_max)` (`llama-graph.cpp`), aplicado nos 4
   sitios `get_tail_attention_stride(...)` do `llama_kv_cache_context`
   (924, 1732, 3839, 5724): em P-mode, se algum token do ubatch e `>= P` =>
   extent = `get_tail_attention_stride(1)` (extensao da cauda persistente:
   `tail_tokens`, ou `tail_tokens+1` no layout compacto = exatamente
   `kt->ne[1]`); senao => `local` (contagem de tokens `< P`).
   Nada muda fora do P-mode (`position_split_enabled()==false` => `q_max`).
3. Overflow keys **nunca** entram na cauda local: o branch so amarra cauda
   quando `tail_on_layer()` diz que a faixa local a possui, e o extent history-only
   cobre so as linhas persistentes.

### Resultado do split real (processo novo, preset isolado, P=256)
```
split active: [F4TAIL(bind)] for P=256
R1-populate      ok prompt=1372 completion=16 cache_n=0 lcp=0    reprocessed=1372
R2-hit-same-srv  ok prompt=1372 completion=16 cache_n=0 lcp=1372 reprocessed=1372
R3-new-process   ok prompt=1372 completion=16 cache_n=0 lcp=0    reprocessed=1372   (P=256)
R4-diff-P512     ok prompt=1372 completion=16 cache_n=0 lcp=0    reprocessed=1372
F4TAIL bind k_tail=[256,129,4,1] mask_tail=[129,1] order=[1] run_desc=[135,1]
```
`GGML_ASSERT` no log = **0** (antes: crash na primeira requisicao).
`lcp=1372` em R2 = prefixo integro reaproveitado no mesmo processo.

### Gates canonicos apos o reparo (`build-f4`)
| gate | resultado |
|---|---|
| matriz canonica sem `F4_RP_*` | **`8490 checks, 0 failures, 0 not-executed`** (exit 0) |
| import vs mesma representacao | **`248320 entries, NRMSE=0, max_abs_diff=0`** |
| controles exactly-once da cauda (`ORACLE duplicate-tail`/`dropped-tail`) | `NRMSE 4.34e-01 / 4.25e-02` (devem EXCEDER 1e-3) — **ativos** |
| `test-position-split-ubatch` / `-ranges` / `-merge` | `2808/0`, `391/0`, `51/0` |
| `test-flash-attn-ext-params` | PASSED (0 failures) |
| `destination-intact` full | `172/4` = baseline `build-optimized` identico (pre-existente) |
| producao | `active health=200 model=qwen-3.8-27b reply='42'` |

### NEXT-ACTION CHECKLIST atualizado
1. ~~fattn-tail.cuh:713~~ **FECHADO** (extent history-only + regressao 8490/0).
2. F4 disco: `auto disk prompt cache ... indexed 0 prefix boundaries` ->
   indice vazio, por isso R3 (`processo novo`) teve `lcp=0`. Provar hit/importacao
   XXL explicito em processo novo com tokens reutilizados e sem prefill.
3. F4: rejeicao com destino intacto (fingerprint KV/tabela/DeltaNet) + fallback.
4. F4: transacao DeltaNet integrada, sufixo divergente, rollback P-129..P+1.
5. Reusar F5/F7 (`docs/occupancy-aux-handoff-20261005.md`).

## CHECKPOINT: F4 item 2 (disco/XXL->split) + regressao do helper de extent — 2026-10-07 02:36

### Correcao de afirmacao (obrigatoria)
`lcp=1372` **so** e prefix-match. `cache_n=0` e `reprocessed=1372` mostram que
**nao** houve reuso: `cache_n = n_prompt_cached` (server-common.cpp:181),
`cache_reprocessed_n = stats.n_prompt_processed` (server-context.cpp:1698) e
`cache_lcp_n = n_prompt_tokens_lcp`. R2 com `cache_n=0, reprocessed=1372` prova
**prefill completo**, nao reuso. Indicadores de reuso a observar:
`cache_n > 0` **e** `cache_reprocessed_n < prompt_tokens`.

### Regressao do helper `tail_extent_queries` (falta antes)
Movido para `src/llama-position-split.h` (f Funcao pura) e exercido em
`tests/test-position-split-ubatch.cpp` caso 5e: nao-P, pos==nullptr, ubatch vazio,
P==0, so-local (4 e 256 tokens), overflow puro, misto, e
`overflow != q_max`.
- **Comportamento pre-fix restaurado: `2817 checks, 5 failures`** (todos `tail extent:`)
- **Corrigido: `2817 checks, 0 failures`** (header restaurado byte-identico)

### F4 item 2 — cenario XXL puro -> destino split, executado de ponta a ponta
`/tmp/opencode/gap3.sh` (preset isolado, slot dir proprio `/home/hjotha/f4-spool/gap3-slots`,
`TMPDIR` em disco porque o leitor misto recusa tmpfs/ramfs).

**FASE A — produtor XXL puro (sem P), n=556 <= P=1024**
- `A-produce prompt=556 cache_n=0 lcp=0 planned=0 reprocessed=556`
- publicacao atomica: **2 `.bin` finalizados, 0 `.tmp`** (`auto-...-556.bin`
  sha=`3082511dbb373bd0` 168180922 B; `auto-...-571.bin` sha=`6c38905ecffaccc1`)
- produtor **encerrado** antes do consumidor.

**FASE B — destino `position_split` novo, so o snapshot em disco, sem cache residente**
- descoberta: `auto disk prompt cache enabled: indexed 4 prefix boundaries ... (block=256)`
  (antes era 0 — era apenas ordem de publicacao/escaneio, **nao** predicao)
- elegibilidade: nenhum `unified snapshot ignored` (identidade de modelo, conversao
  `auto_convertible_pure_kvarn_layout`, tokens nao vazios) => **selecionado**
- **importacao REJEITADA** e caminho de fallback comprovado:
  `E state_seq_load_file_streaming: mixed state snapshot has no position_split
   envelope but this cache is split; empty destination cleared`
  `W slot auto_restore: candidate ...556.bin failed native streaming validation;
   destination was EMPTY beforehand, nothing to restore`
  => `B-import prompt=556 cache_n=0 lcp=0 planned=0 reprocessed=556` (cold prefill
  explicito, destino intacto). Tambem validados e ja observados:
  `mixed state spool directory is tmpfs/ramfs` e
  `cannot create converted sequence state temporary file` (TMPDIR em disco resolve).

### GAP ABERTO e delimitado (nao waivered)
O predicado de conversao aceita o snapshot XXL puro no indice e a selecao o
escolhe, mas o **loader exige envelope `position_split`** e recusa. Falta a
importacao explicita autorizada do §3.6 linhas 187-191 ("preparar/restaurar em
etapas e commit existentes; preencher local e deixar overflow vazio").
Estado atual do item 2:
- [x] snapshot XXL puro com n<=P, publicacao atomica, metadados, produtor saiu
- [x] processo novo so com o snapshot, sem cache residente, indice descobriu
- [x] rejeicao com destino intacto + fallback explicito (caminho negativo)
- [ ] **importacao autorizada positiva + `cache_n>0` + `reprocessed < prompt_tokens`** (ABERTO)

### Preservado
`8490/0`, `import NRMSE=0`, `2817/0` (+9), `523/0`, `391/0`, `51/0`, `op_params`,
fix M-RoPE, fix history-only extent, `build-f4` isolado, producao
`active health=200 model=qwen-3.8-27b reply='42'`.

## CHECKPOINT: importacao pura XXL -> split implementada (entrada) — 2026-10-07 02:45

### Implementado (reusa a maquina existente; nada redesignado)
1. `snapshot::adopt_split_for_import(const position_split &,
   const std::vector<layer_desc> &)` (`src/llama-kv-mixed-state.h:454`,
   `.cpp:942`):
   - recusa se o frame ja tem envelope, se esta moved/committed/validated, se o
     conjunto de destino esta vazio, se o **source nao e coberto** pelo destino,
     ou se um range sintetizado **carregaria bytes**;
   - **re-binda** cada descritor de origem sobre a visao de destino do MESMO
     (layer, range) — o source puro nao traz limites de faixa, entao sua faixa
     local adota a geometria `[0,P)` do destino; **payload nao e tocado**
     (`payloads_` fica index-alinhado); divergencia de `payload_bytes` = recusa;
   - **sintetiza os ranges ausentes como descricoes VAZIAS** (bytes omitidos,
     descricao mantida — §3.6) com `payload_ref{}` e payload vazio;
   - entao `split_ = want`.
2. `llama-kv-cache-kvarn.cpp` (antes da recusa de envelope): quando o destino e
   split e o frame nao tem envelope, valida **`extent <= min(P, capacity)`**
   (max pos+1) e chama `adopt_split_for_import(want, expected_descs)` —
   `want = {P, C, state_model_id, state_rope_id}`. A recusa original
   (`has_split() != position_split_enabled()`) continua de pe para todos os
   outros casos, e **todas as checagens seguintes rodam antes de staging**.

### Resultado da cenario genuina (processo novo)
- `indexed 4 prefix boundaries` (descoberta ok)
- `mixed state snapshot has no position_split envelope but this cache is split`
  **eliminado**
- novo bloqueio exato, ainda fail-closed antes de staging:
```
E state_seq_load_file_streaming: invalid KV tail state manifest dimensions:
  tail=1 payload_count=0 tail_slots=129 record_count=0 max_records=0
  tail_layer_count=0 layers=0 provenance_count=0 expected_provenance=1
  (llama-kv-cache.cpp:4874)
```
`provenance_count != (seq_id == -1 ? saved_n_seq_max : 1)` com `layers=0` no
manifest de tail-state do **source**. Proximo passo: o conversor/reader de
tail-state precisa produzir a provenance esperada (ou o branch
`tail_metadata_only` precisa aceitar o caso puro) — sem relaxar a checagem.

### Correcao de cobertura negativa (exigida; mantida SEPARADA)
`destination was EMPTY beforehand, nothing to restore` cobre **apenas** o caso
de destino vazio. **NAO** fecha:
- [ ] destino **populado** integro apos import rejeitado (fingerprint de KV +
      tabela de celulas + DeltaNet antes/depois)
- [ ] rollback pos-clear/touch (a sequencia `slot.prompt_clear()` antes do
      `do_slot_restore`, e a `guard.rollback()` apos a recusa, com destino
      preenchido)
Esses casos continuam **NAO EXECUTADOS** e devem ser medidos com fingerprints.

### Validacoes apos a alteracao
`test-kv-mixed-state 523/0`, `test-kv-mixed-state-stream 819/0`,
`test-position-split-ubatch 2817/0`; producao `active health=200
model=qwen-3.8-27b reply='42'`. Gates `8490/0` e `import NRMSE=0` nao
reexecutados nesta rodada (mudanca restrita ao leitor de snapshot; reexecutar
na proxima verificacao).

## CHECKPOINT: revisao do adaptador pura->split aplicada — 2026-10-07 03:05

### `adopt_split_for_import` reescrito (riscos da revisao corrigidos)
Assinatura agora `(want, destination_descs, source_model_id, source_rope_id)`.

**Antes de rebind, validado:**
- identidade da FONTE: vem do check confiavel ANTERIOR do chamador
  (`server-context.cpp`: `common_prompt_cache_layout_pure_to_split`), **nao**
  inferida; `0/0` => "identity is unavailable"; `!= want.*` => recusa
  (`model %llu/%llu rope %llu/%llu`);
- **todo campo invariante** do descritor contra o contrato de conversao:
  `kind, k_type, v_type, k_bits, v_bits, kvarn_domain, k_rot, v_rot, k_rot_width,
  v_rot_width, layout, owner, v_trans, tail_type, token_group, record_dim,
  head_dim_k, head_dim_v, head_slices, n_head_kv, n_stream, k_stride, v_stride,
  payload_mode, payload_cells, payload_rows, payload_bytes, layer_id, range_id`
  — **comprimento igual de bytes NAO e compatibilidade**;
- fonte **coberta** pelo destino (range/layer ausente => recusa) e **sem
  descritor duplicado**;
- payload mapeado coerente: `payloads_[i].size == payload_bytes` e
  `data != nullptr` quando `payload_bytes != 0` => "missing or corrupt";
- range sintetizado com `payload_bytes != 0` => recusa.

**Rebind permitido somente:** geometria `range_start`/`range_end` (+ append do
overflow vazio). `payload_checksum` da fonte e **preservado byte a byte**
(provenance autenticada).

### Testes dedicados (novos)
`tests/test-kv-mixed-state.cpp::test_pure_import_adapter` — positivo + negativos:
- positivo: envelope adotado, geometria local `[0,P)`, overflow vazio desrito,
  checksum autenticado preservado, dimensoes/strides preservados, payload
  intocado (`payload_size(0)` e ponteiro mapeado);
- identidade: indisponivel / model divergente / RoPE divergente;
- `k_type` (formato, mesmo comprimento), `head_dim_k` (dimensao), `k_stride`,
  `kvarn_domain`, `layout`, `tail_type`, `record_dim`;
- fonte duplicada, range nao pertencente ao destino, sintese nao-vazia,
  payload cujos bytes nao batem com o descritor.
Resultado: **`PASS: 548 checks, 0 failures`** (era 523; +25).

### Rastreio do provenance (parte B) — contrato escritor/leitor
- escritor `src/llama-kv-cache.cpp:4661`:
  `result.provenance = tail->snapshot_provenance(seq_id);`
  (`llama_kv_tail_store::snapshot_provenance`, `llama-kv-cache-tail.cpp:1267`)
- escritor `:4712` grava `provenance_count = manifest.provenance.size()` = **0**
- leitor `:4874` exige
  `provenance_count == (seq_id == -1 ? saved_n_seq_max : 1u)` = **1**
=> o cache fonte devolveu **zero** entradas de provenance; nao ha "faltando"
para inventar. **Nao relaxar a validacao**: resolver no contrato do
`tail_store` (por que `snapshot_provenance(seq_id)` devolve vazio para o frame
puro, e se o conversor/gravador deve emitir 1 entrada por seq) e cobrir com
teste escritor->leitor.

### Ainda aberto (separado, sem waiver)
- [ ] provenance/tail-state do caminho puro (bloqueia o import positivo)
- [ ] destino POPULADO integro + fingerprint antes/depois em import rejeitado
- [ ] rollback pos-clear/touch com destino preenchido
- [ ] `8490/0` e `import NRMSE=0` reexecutados apos estas mudancas (rodar na
      proxima verificacao)

## DESIGN REVIEW (esperando aprovacao antes de novo patch de proveniencia) — 2026-10-07 03:35

Nenhuma alteracao de codigo nesta rodada. Sketch de fluxo de dados com linhas reais.

### (1) Escritor: identidade de origem serializada e campos
- `position_split {p, c, model_id, rope_id}` — `src/llama-kv-mixed-state.h:287-292`.
- Serializado **apenas no cabecalho v2**: `build_header_v2()`
  `src/llama-kv-mixed-state.cpp:438-470`; campos em `:456-459`
  (`put_u32(split.p)`, `put_u32(split.c)`, `put_u64(split.model_id)`,
  `put_u64(split.rope_id)`), cobertos por `header_crc = XXH64(...)` em `:460-464`.
- Só existe no frame quando `v2`: `if (v2) s.split_ = split;` `:807`;
  parse valida com `validate_split_envelope(split, "parse")` `:787`.
- Origem dos valores: `llama-model.cpp:3460-3462`
  `set_mixed_state_identity(llama_mixed_state_model_id(arch,hparams),
  llama_mixed_state_rope_id(hparams))`; escritor recusa identidade zero
  `src/llama-kv-cache-kvarn.cpp:4882` e grava `split.model_id/rope_id` `:4889`.
- **Um frame PURE/v1 NAO carrega model/RoPE**: `v2=false` => `split_` vazio.
  Formato documentado em `llama-kv-mixed-state.cpp:31-35` (prefixo v1 de 96
  bytes, sem bloco `position_split`).
- Identidade no **wrapper** do servidor (FORA do arquivo de estado):
  `server_route_state_file` `tools/server/server-context.cpp:1097-1101`
  (`state_path`, `model`, `layout`) + `state_bytes`/`state_checksum` usados
  em `:4111`.

### (2) Leitor / integridade / compatibilidade (linhas)
- Streaming: `src/llama-context.cpp:6394-6452` — limites `state_size`/`checksum`
  `:6400-6401`; XXH64 de todo o payload vs `checksum` `:6423-6431`;
  magic/versao `:6436-6440`; re-checagem `io.n_bytes()/io.checksum()` apos
  `state_seq_read_data` `:6446-6448`.
- Payload do snapshot: `src/llama-kv-mixed-state-stream.cpp:452-475` (pass1
  XXH64 por payload), `:492` (metadata), `:509` (`header_crc = XXH64(header)`),
  `:557-599` (pass2 vs `d.payload_checksum`).
- Envelope/P/C/model/RoPE: branch pura `src/llama-kv-cache-kvarn.cpp:5101`;
  `validate_expected_snapshot` `src/llama-kv-mixed-state.cpp:1125` (presenca),
  `:1132` (P/C), `:1135` (model), `:1138` (RoPE).
- Selecao do servidor: fingerprint+reutilizavel `server-context.cpp:3627-3628`
  e `:3643-3644`; ramo de conversao `:4189-4199`
  (`file->model == fingerprint` + `!reusable` + `auto_convertible_pure_kvarn_layout`);
  predicate `:3839-3841` -> `common_prompt_cache_layout_pure_to_split`
  `common/common.cpp:4166+` (recusa fonte `kv_layout_known` e fonte com `mixed_kv`).

### (3) Portador por load — ESTADO ATUAL: **NAO vinculado ao checksum/arquivo**
- Tipo `pure_import_context {authorized, model_id, rope_id}`
  `src/llama-kv-mixed-state.h:423-428`.
- Armazenado no cache `src/llama-kv-cache-kvarn.h:708` (`pure_import_auth_`);
  setter/getter `:699-706`.
- Escrito por `llama_context::state_authorize_pure_import` (`src/llama-context.cpp`),
  API `include/llama.h:1145/1154`.
- **Falta**: nao carrega `state_path`, `state_bytes` nem `state_checksum`
  (todos ja existem em `server-context.cpp:4111`) => qualquer restore posterior
  no mesmo contexto pode consumi-lo.

### (4) Identidade de DESTINO mantida separadamente
`state_model_id/state_rope_id` no cache (set em `llama-model.cpp:3460-3462`);
usados para montar `want` em `llama-kv-cache-kvarn.cpp:5120-5125` e comparados
dentro de `adopt_split_for_import` (`llama-kv-mixed-state.cpp:943+`).

### (5) Consumidores e limpeza — e onde vaza hoje
- Consumidor unico: `llama-kv-cache-kvarn.cpp:5126`
  `take_pure_import_authorization()`, **dentro** de
  `if (position_split_enabled() && !parsed.has_split())` (`:5101`).
- Caminhos que **NAO** consomem (autorizacao fica stale e pode vazar para um
  restore posterior nao relacionado):
  early returns de `state_seq_load_file_streaming` `:6395`, `:6398`, `:6403`,
  `:6410`; throws `:6431`, `:6440`, `:6446-6448` ->
  `catch` `:6453` + `memory->clear(false)` `:6454`; throws do leitor **antes**
  de `:5101`; caminho split->split (`parsed.has_split()==true`); e os
  `return 0` da selecao `server-context.cpp:4220`, `:4224`.
- Site de escrita: `server-context.cpp:4100-4101`, antes de `nread` `:4109`.

### Defeito declarado (proveniencia) — nao corrigido
`server-context.cpp:4100` chama `llama_state_get_identity(ctx_tgt, ...)` e
` :4101` grava esses valores como `source_model/source_rope`; `want` em
`llama-kv-cache-kvarn.cpp:5121-5124` usa os MESMOS `state_model_id/state_rope_id`
=> a comparacao em `llama-kv-mixed-state.cpp` e `X == X`. **Tautologica.**

### Investigacao de `pure_to_split=0 reusable=1`
`auto_convertible_pure_kvarn_layout` (`:3839`) delega a
`common_prompt_cache_layout_pure_to_split` (`common/common.cpp:4166+`), que
retorna falso quando a fonte declara `kv_layout_known` ou `mixed_kv`. O layout
observado na fonte tinha `"kv_layout_known":true,"mixed_kv":true` (o produtor
herdou a faixa Q4 remota do preset) => `pure_to_split=0`; `reusable=1` porque
`file.layout == current_layout`. Com produtor realmente puro (sem faixa remota)
o run bem-sucedido logou `pure_to_split=1 reusable=0`. Nao e autorizacao
arbitraria: os predicados usam `file.layout` e o fingerprint do modelo.

### Decisao pedida (fail closed) — requisito de formato
Um frame **v1/pure NAO tem** model/RoPE proveniente do arquivo
(`llama-kv-mixed-state.cpp:31-35`; `v2=false => split_` vazio em `:807`).
Portanto, ou (a) **fail closed para esse formato** no caminho pure->split, ou
(b) um campo **estreitamente versionado** de identidade de origem.
Requisito proposto para revisao:
- novo cabecalho (ex. `KMS2`-style bump) que, mesmo sem envelope `position_split`,
  grava `source_model_id`/`source_rope_id` apos o prefixo v1 e **antes** do
  payload, coberto pelo `header_crc` existente; escritor em
  `llama-kv-cache-kvarn` emite os campos a partir de `state_model_id/state_rope_id`
  tambem quando nao emite envelope; leitor **exige** o bloco apenas no branch
  pure->split (split->split continua usando o envelope v2);
  **regenerar o fixture isolado** (`gap3` produtor) apos a troca de formato.
- portador imutavel por load: ligar a autorizacao a
  `state_path + state_bytes + state_checksum` (ja passados em
  `server-context.cpp:4111`) e limpá-la em **todos** os caminhos — RAII dentro
  de `state_seq_load_file_streaming` (sucesso em `:6449`/`return state_size` e
  `catch` em `:6453-6454`, mais todos os early returns), em vez de membro do cache.
- testes pedidos (apos aprovacao): fonte real com model/RoPE divergente;
  mesmo comprimento de bytes com identidade incompativel; bypass do loader
  direto; autorizacao stale (restore que nao alcancou o branch puro nao pode
  deixar autorizacao para o proximo); todos com fingerprint de KV/tabela/DeltaNet
  do destino **populado** inalterado.

Producao verificada nesta rodada: `active`, `health=200`,
`model=qwen-3.8-27b`, `reply='42'`.

## CHECKPOINT: identidade de origem KMS3 (decisao 2026-10-07 01:29) — 2026-10-07 04:05

### Formato (aprovado, estreito)
- `KMS3_MAGIC 0x33534D4B`, `KMS3_VERSION 3`, header **128 B** = prefixo v1
  byte-a-byte (96) + `source_model_id` + `source_rope_id` + `source_config_id`
  (24) + `identity_crc = XXH64(bloco,24,0)` (8). `llama-kv-mixed-state.cpp:47-55`
  (`build_header_v3`), `llama-kv-mixed-state-stream.cpp:42-45`.
- **Integridade, nao autenticidade**: o prefixo v1 mantem o proprio crc em 88;
  o bloco de identidade tem crc proprio; o payload continua coberto por
  `pass1/pass2 XXH64` (`llama-kv-mixed-state-stream.cpp`) e pelo XXH64 do
  arquivo em `llama-context.cpp:6423-6431`.
- **Leitura antiga preservada**: `parse_impl` despacha por magic
  (`KMS1|KMS2|KMS3`), `expect_version` por magic, `reserved[8]` do prefixo v1,
  e falha fechada para `version != 1|2|3` e para `magic/version mismatch`.
  Frames v1 legados continuam legiveis e **nao** carregam identidade.

### Escritor
- `snapshot::serialize` / `stream_serialize` aceitam `serialize_options.source`
  / `stream_input.source` e emitem KMS3 so para frames **puros** (sem envelope).
- `llama-kv-cache-kvarn.cpp` grava `input.source = {state_model_id,
  state_rope_id, descriptor_config_id(input.layers)}` no ramo nao-split.
- `descriptor_config_id()` = XXH64 canonico sobre os campos de config dos
  descritores (sem geometria de faixa), recompute em `parse_impl`
  (`llama-kv-mixed-state.cpp:1080`) -> "source config identity does not match
  its descriptors".

### Portador + remocao da rota tautologica
- **Removidos**: `pure_import_context`, `set/take_pure_import_authorization`,
  membro `pure_import_auth_`, `llama_state_get_identity`,
  `llama_state_authorize_pure_import` (API publica), bloco de autorizacao do
  servidor. **Nenhuma rota destino-como-fonte permanece.**
- Portador = **o proprio frame** (por-load, imutavel, ligado aos bytes exatos
  validados: header crc + XXH64 do arquivo + XXH64 por payload). Nada reutilizavel
  entre restores; estado stale e impossivel.
- `adopt_split_for_import(want, destination_descs, source_identity)`:
  fonte ausente (v1 legado) -> recusa explicita; `source.model_id != want.model_id`
  e `source.rope_id != want.rope_id` -> recusa; `source.config_id !=
  descriptor_config_id(layers_)` -> recusa; depois `invariant_ok` campo a campo
  (fonte vs destino) antes de qualquer rebind; so `range_start/range_end` mudam;
  `payload_checksum` preservado.

### Testes (`test-kv-mixed-state`: 550 -> **585 checks, 0 failures**)
positivo KMS3 (identidade round-trip, envelope adotado, overflow vazio, payload
intocado) + negativos: **legacy v1 sem identidade** (mensagem nomeando a falta),
modelo divergente, RoPE divergente, config divergente, identidade igual-em-tamanho
contra outro destino, **versao desconhecida**, **truncamento**, **corrupcao do
cabecalho**, **corrupcao do bloco de identidade**, formato/dimensao/stride/domain
iguais-em-bytes, range duplicado/nao-pertencente, sintese nao-vazia, e
**stale** (load que nao e puro continua recusado apos um load que funcionou).

### Cenario real (processo novo) — positivo com identidade do frame
```
A-produce prompt=556 reprocessed=556            (produtor PURE, preset sem faixa remota)
atomic publication: 2 finalized .bin, 0 .tmp
B-import prompt=556 cache_n=556 lcp=0 planned=0 reprocessed=0
  auto-restore: importing PURE KVarN snapshot ...-556.bin (556 tokens) into the split destination
  auto-restore: reused 556 tokens from disk (in-memory match was 0), bytes=197579721 restore_ms=405.585
```
**556 tokens reutilizados, 0 de prefill** (`cache_n=556`, `reprocessed=0`).
A adocao so passa se o frame trouxer identidade valida — prova indireta de que
KMS3 foi emitido e lido (um v1 seria recusado com "no source identity", como o
teste negativo demonstra).

### Gates apos a mudanca
`real-pair O/LSE: 8490 checks, 0 failures, 0 not-executed` ·
`import vs same-representation reference: 248320 entries, NRMSE=0, max_abs_diff=0` ·
`test-kv-mixed-state 585/0` · `test-kv-mixed-state-stream 819/0` ·
`test-position-split-ubatch 2817/0` · destination-intact `172/4` = baseline inalterado.

### Producao (modelo explicito + pesos reais)
`active`, `health=200`, `model=qwen-3.8-27b`, `reply='42'`,
`system_fingerprint = b12366-d9db0b881-dirty` (build atual),
pesos `/home/hjotha/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf`
= 10442827776 bytes, GGUF v3, 866 tensores, VRAM 11483 MiB.
`n_params` nao e exposto pelo router (`/v1/models` `parameters` vazio, `/props`
do router sem o campo); o worker so expoe generation settings. **Nao obtido** —
registro honesto, nao inventado.

## CORRECAO DE CABECALHO + CLASSIFICACAO DOS 4 FALHAS — 2026-10-07 01:52 UTC

**Cabecalho anterior "2026-10-07 04:05" estava errado.** Horario real (UTC) desta
sessao: **2026-10-07 01:52 UTC**. Todo checkpoint novo deve usar `date -u`.

### n_params (obtido, confere com a verificacao independente)
`/v1/models` -> `data[0].meta.n_params` = **27320697856**
(tambem `meta.n_ctx=131072`, `n_embd`, `n_vocab`, `ftype`; o campo
`models[].parameters` de nivel superior e `None` e NAO deve ser usado).

### Classificacao das 4 falhas de `destination-intact` (log `f4-import-parity-20261007-034014.log`)
Todas as 4 estao na MESMA linha `tests/test-kv-mixed-destination-intact.cpp:2617`,
que e `CHECK(dAB > 0.0f)` — **alcancamento POSITIVO** de uma mutacao LIVE ate a FA:
`n=129 k_stage head=1`, `n=129 v_stage head=1`, `n=257 k_dest head=1`,
`n=257 v_stage head=1` (todas `max=0`, `pos 0` fora do overlay).

| lado canonico | CHECK | resultado neste run |
|---|---|---|
| **nao-mutacao** (mutacao INATIVA nao pode alterar logits) | `:2624` `CHECK(dAC == 0.0f)` | **0 falhas**; 19 controles `C(inactive) max=0` |
| **nao-mutacao** (mutacao dentro do overlay nao pode alterar) | `:2007` `CHECK(dAB == 0.0f)` | **0 falhas** |
| **alcancamento positivo** (mutacao LIVE deve alterar) | `:2617` `CHECK(dAB > 0.0f)` | **4 falhas** (classe unique) |

**Contra o requisito canonico de "destino populado sem mutacao" (KV/tabela/DeltaNet):
nenhuma das 4 falhas o viola** — elas sao lacunas de cobertura POSITIVA
(k/v_stage -> FA em cabeca 1, n=129/257), a mesma classe ja registrada como
diagnostico nao-resolvido.

**NAO afirmo atomicidade aprovada.** Este teste nunca vincula fingerprint de
KV/tabela/DeltaNet atravessador do **loader real**; ele mede logits. A matriz
de negativos com fingerprint de destino populado via loader real continua
**NAO EXECUTADA**.

### Proximo passo imediato (F4, sem waiver)
Matriz de negativos no **loader real** sobre destino ja populado, com
fingerprint antes/depois: populada -> `sha256(slot snapshot)` (serializacao de
KV+celulas+DeltaNet) -> tentativa de import recusada (modelo/RoPE/config
divergente, mesmo-comprimento incompativel, corrompido/truncado/versao
desconhecida, rejeicao de conversao v1 legado, loads sucessivos/stale) ->
fingerprint identico -> mesma requisicao cai no fallback frio.

## MATRIZ DE DESTINO POPULADO (loader real) — PRIMEIRA TENTATIVA, NAO COBERTA — 2026-10-07 01:55 UTC

Script `/tmp/opencode/gap4.sh` (destino populado, fingerprint = sha256 do
snapshot de slot, negativos corrompido/truncado/versao desconhecida/modelo/RoPE).

**Resultado: a matriz NAO alcancou o loader.** Todas as 6 tentativas
responderam `cache_n=512 reprocessed=44` e nenhuma linha de recusa apareceu
(`state_seq_load_file_streaming:`, `failed native streaming validation`,
`rolled back` todas vazias). Causa: com o destino **populado**, o match em
memoria (`n_keep_mem=512`) vence a MARGIN gate do seletor
(`n_keep_disk < n_keep_mem + B` com `B=256` => `return 0` em
`server-context.cpp`), entao o snapshot do disco nunca e carregado.

Tambem observado: o sha256 do snapshot de slot **muda a cada requisicao**
(re-save com contadores/geracao diferentes) — logo ele **nao** serve como
fingerprint de nao-mutacao. Um fingerprint estavel exige ler o estado ANTES e
DEPOIS da tentativa dentro do mesmo processo (`llama_state_get_data`), nao
o arquivo salvo.

**Estado honesto:** a matriz de negativos do loader real sobre destino populado
(com fingerprint KV/tabela/DeltaNet antes/depois e fallback seguro) continua
**NAO EXECUTADA**. F4 segue **ABERTA**. Proximo passo: exercitar o loader com
`n_keep_mem` nulo (processo novo ja populado por outro prefixo, ou prompt
maior que o snapshot em memoria em `>= B` tokens) e fingerprintar com
`llama_state_get_data` dentro do processo.

Producao restaurada: `active`, `health=200`, `model=qwen-3.8-27b`,
`n_params=27320697856` (via `data[0].meta.n_params`), `reply='42'`.

## FINGERPRINT NORMALIZADO + CONTROLES POSITIVOS (addendum) — 2026-10-07 02:05 UTC

`normalized_state_fingerprint(snapshot, recurrent, recurrent_bytes)` em
`tests/test-kv-mixed-state.cpp`: cobra **so estado semantico documentado** —
payload XXH64 por camada, contrato de descritor (formato/dimensao/stride/domain/
layout/tail/grupo/dim), celulas `{pos,x,y,tok,seq_ids}` e a regiao recorrente/
DeltaNet fornecida pelo chamador.

**EXCLUIDO (metadados de envelope irrelevantes, documentados):** magic/versao/
checksum do cabecalho, `owner_epoch`, `owner_id`, container/rodape do arquivo,
caminho, identidade de fonte/backend, contadores de geracao e estatisticas de
carga transitores. Nada alem disso e excluido — por isso "fingerprints
concordam" e informativo (um SHA de arquivo inteiro nao seria).

**Controles positivos** (`test_fingerprint_controls`, +14 checks):
- mutacao de **payload KV** -> fingerprint muda;
- mutacao de **celula da tabela** (e um unico campo `tok`) -> muda;
- mutacao da **regiao recorrente/DeltaNet** -> muda (e reprodutibilidade
  confirmada: mesmas entradas -> mesmo valor);
- mutacao de **contrato de descritor** (`k_stride`) -> muda;
- **controle negativo de envelope**: `owner_epoch`/`owner_id` diferentes ->
  fingerprint **nao** muda (prova que a exclusao e real e nao vazia).

Resultado: `test-kv-mixed-state: PASS 599 checks, 0 failures` (585 -> 599).

**Snapshots em disco existentes preservados:** `gap3-slots` e `gap4-slots`
intactos, `gap4-good.bin` intacto; nenhum script de teste apaga diretorio
alheio (cada rodada usa o proprio `gapN-slots`).

F4 segue **ABERTA**: a matriz de rejeicao do loader real sobre destino populado
ainda nao foi executada com este fingerprint (proximo passo: elegibilidade do
candidato de disco contra prefixo residente DIFERENTE + fingerprint antes/depois
da rollback, dentro do processo).

## BLOQUEIO EXATO: a matriz nao alcancou a validacao pretendida — 2026-10-07 02:28 UTC

**Evidencia** (`docs/occupancy-evidence/f4-populated-20261007-022339.log`):
todas as 5 nao-triviais recusaram no **mesmo** ponto de baixo nivel:
```
state_seq_load_file_streaming: streaming restore requires an empty context
```
Capturado pelo proprio teste:
`[got: state_seq_load_file_streaming: streaming restore requires an empty context]`
para `wrong_model_id`, `wrong_rope_id`, `identity_corrupt`,
`unknown_version`, `header_corrupt`.

**Consequencia:** o resultado anterior `rejected=1 / fingerprint_same=1` era
**vazio** — o loader nunca leu o frame; a precondicao de contexto vazio abortou
antes. **Nao** e validacao de identidade/corrupcao e **nao** pode ser reportado
como cobertura. A guarda de contexto vazio **nao** sera enfraquecida.

**Por que acontece:** o teste chama `llama_state_seq_load_file_streaming`
direto com destino populado. A ordem correta e a do wrapper do servidor:
`destination_guard` captura -> `slot.prompt_clear()` -> loader (falha na
intencao) -> `guard.rollback()` -> inspecao do estado semantico -> fallback.

**Bloqueio tecnico concreto (nao e pedido de permissao):**
`destination_guard` esta **duplicada** e **acoplada** ao tipo de servidor:
- `tools/server/server-context.cpp:4265` (usa `server_slot&`, `prompt.clone()`,
  `prompt_clear()`, `slot.prompt = ...`)
- `tools/server/server-task.cpp:2266` (segunda copia)
So pode ser exercitada de verdade extraindo o **nucleo** (captura/rollback do
estado de sequencia em `llama_context`, com a checagem de spool em disco) para
uma unidade compartilhada e testavel, com as duas copias delegando a ela, e o
teste chamando a MESMA funcao. Sem isso, um rollback de teste copiado poderia
passar com a producao quebrada — exatamente o risco apontado.

**Trabalho pendente (F4, sem waiver):**
1. extrair o nucleo `capture_resident_state / rollback_resident_state` para
   codigo compartilhado; `server-context.cpp` e `server-task.cpp` delegam;
2. o teste usa essa mesma funcao: captura -> clear -> loader (motivo pretendido
   registrado) -> rollback -> fingerprint semantico **antes** do fallback ->
   depois a conclusao do fallback;
3. registra o motivo exato por caso;
4. reexecutar a matriz.

**Estado dos testes:** `198 checks, 9 failures` = 4 diagnosticos LIVE
conhecidos (`n=129/257 k_stage/v_stage head=1`, linha 2771) + 5 falhas dos
novos assertions de motivo (que **corretamente** detectaram a recusa vazia).
Nenhum "aggregate pass" reivindicado. `test-kv-mixed-state 599/0`,
`-stream 819/0`, `ubatch 2817/0` continuam verdes.

**Producao restaurada:** `active`, `health=200`, `model=qwen-3.8-27b`,
`n_params=27320697856`, `reply='42'`.

## NUCLEO COMPARTILHADO EXTRAIDO + TRANSACAO REAL EXERCITADA — 2026-10-07 02:45 UTC

### Compartilhado (ambos os caminhos de producao delegam)
Novo `src/llama-state-rollback.h`:
- `spool_usable(dir, &err)` — precondicao de spool em disco **ANTES** de qualquer clear;
- `resident_positions(ctx, seq, &pmax)` — residencia decidida por POSICOES;
- `capture_resident(ctx, seq, blob, &pos, &had, &err)` — captura fail-safe;
- `rollback_resident(ctx, seq, blob, pos)` — restaura bytes e verifica o watermark.

Delegacao: `tools/server/server-context.cpp` (ctor do `destination_guard` +
`rollback()`; prompt/checkpoint continuam no slot) e
`tools/server/server-task.cpp` (`resident()` e `capture_one()`).
`llama-server` compila (`build=0`). **Nao ha rollback copiado no teste.**

### Teste agora exercita a TRANSACAO REAL (`F4_TEST_POPULATED`)
por caso: `spool_usable` -> `capture_resident` -> **clear**
(`llama_memory_seq_rm`, sem enfraquecer a guarda de contexto vazio) ->
`llama_state_seq_load_file_streaming` -> **`rollback_resident`** ->
fingerprint imediatamente **antes do fallback** -> conclusao do fallback.

Resultado (`docs/occupancy-evidence/f4-populated-20261007-024131.log`):
```
POPULATED fingerprint bytes=56362837 stable=1 resident_pos=32..32
wrong_model_id   rc=60151147 rejected=0 reason_ok=0 rb=1 fp_same=1
wrong_rope_id    rc=60151147 rejected=0 reason_ok=0 rb=1 fp_same=1
identity_corrupt rc=60151147 rejected=0 reason_ok=0 rb=1 fp_same=1
unknown_version  rc=60151147 rejected=0 reason_ok=0 rb=1 fp_same=1
header_corrupt   rc=60151147 rejected=0 reason_ok=0 rb=1 fp_same=1
truncated        rc=0        rejected=1 reason_ok=1 rb=1 fp_same=1
fallback repopulate bytes=56362837
```
- **Rollback real: `rb=1` (restored) em todos os 6 casos**
- **Estado semantico identico antes/depois da rollback e ANTES do fallback:
  `fp_same=1` em todos** (`fingerprint stable=1` verificado antes de qualquer load)
- **Conclusao do fallback demonstrada** (repopula para 56362837 bytes)
- **truncated** recusado pelo loader real (unico que rejeita)

### ACHADO REAL (nova causa exata)
Os 5 casos de identidade/corrupcao agora **passam** pela guarda de contexto
vazio e o loader **retorna `rc=60151147` (sucesso, = state_size)** com
`[got: ]` vazio — ou seja, **nao ha bloco KMS3 no arquivo da fonte**. O teste
localiza `"KMS"` no arquivo e grava nos offsets 96/104/120 assumindo KMS3;
com um frame KMS1 those writes caem fora do bloco de identidade e a leitura
conclui. Consequencia: **o contexto fonte PURE deste teste nao emitiu KMS3**,
ou seja `state_model_id/state_rope_id` nao estavam carimbados nessa
configuracao, e o escritor caiu no caminho v1.
Proximo passo concreto: confirmar o magic real do arquivo da fonte
(`KMS1|KMS3`) e, se KMS1, garantir o carimbo de identidade no cache PURE
antes de reexecutar — nao adianta gravar em offsets de um formato ausente.

### Testes
`destination-intact: 222 checks, 14 failures` = 4 diagnosticos LIVE conhecidos
+ 10 das novas assercoes de motivo/sucesso que **corretamente** apontam o
achado acima. Nenhum "aggregate pass". `599/0`, `819/0`, `2817/0` verdes.

### Producao
`active`, `health=200`, `model=qwen-3.8-27b`, `n_params=27320697856`,
`reply='42'`.

## HARNESS CORRIGIDO: localizador real + fail-fast — 2026-10-07 03:07 UTC

**Correcao aplicada** (o localizador por 3 bytes ASCII era invalido):
`locate_mixed_frame()` parseia a LAYOUT REAL, nesta ordem, e falha imediatamente:
1. container externo: `LLAMA_STATE_SEQ_MAGIC` (4 B), `LLAMA_STATE_SEQ_VERSION`
   (4 B), `n_tokens`, tabela de tokens dentro dos limites;
2. envelope misto: busca `MIXED_KV_STATE_ENVELOPE_MAGIC = 0x4D4B5351 ("QSKM")`
   + `version = 1` + `frame_bytes` contido no arquivo;
3. cabecalho KMS: magic `KMS1|KMS2|KMS3` (4 B) + versao correspondente +
   tamanho de cabecalho (96/104/128) + **checksum do cabecalho recomputado**
   (v2: 104 B com crc em 96; v1/v3: prefixo de 96 B com crc em 88).
So entao a mutacao e aplicada, no offset validado.

**Resultado do focused fixture validation (antes da sonda custosa):**
```
destination-intact: 180 checks, 5 failures
POPULATED wrong_model_id    FIXTURE INVALID: no mixed-KVarN envelope (QSKM) in
  the state payload: the source is NOT using the intended KVarN mixed-state
  writer in this fixture
```
A matriz **para na primeira validacao** (`break`) — nenhum falso negativo,
nenhuma alteracao de identidade de conversao baseada em localizador espurio.
`magic=KMS7 at=46005155` foi confirmado como bytes de payload arbitrario;
a inferencia anterior de "a fonte emitiu KMS1" **nao tem suporte** e foi
retirada.

As5 falhas = 1 de validade do fixture + **4 diagnosticos LIVE conhecidos**
(`n=129/257 k_stage/v_stage head=1`, linha 2955). Nenhum "aggregate pass".

### Truncamento
Agora prova `b.size() < pristine.size()` (comprimento menor), nao mutacao de
byte na mesma offset.

### Nucleo compartilhado (mantido)
`spool_usable` agora faz a preflight REAL: `statfs` tmpfs/ramfs +
nao-diretorio + nao-gravavel (probe de escrita) + criacao de
`<spool>/beellama-mixed-state`, com `server_mixed_state_spool_dir`
delegando; `server-task` delega `rollback_resident` para o alvo/rascunho e
mantem a transacao do carry speculative propria.
`test-kv-mixed-state: PASS 606 checks, 0 failures` (7 novos casos de
preflight: vazio, nao-diretorio, nao-gravavel, tmpfs, valido, subdir, idempotente).

### Proximo passo concreto (F4)
Fazer a FONTE deste fixture emitir o envelope misto (QSKM/KMS3) — ou trocar a
fonte por uma que use o escritor KVarN misto pretendido — antes de reexecutar a
matriz de identidade/corrupcao. Enquanto isso a matriz permanece **NAO
EXECUTADA** (fail-fast), com rollback real e fingerprint semanticos ja verdes
nas 6 tentativas anteriores.

### Producao
`active`, `health=200`, `model=qwen-3.8-27b`, `n_params=27320697856`,
`reply='42'`.

## FIXTURE: artefato gap3 copiado, mas sem envelope no ficheiro atual — 2026-10-07 03:17 UTC

**Feito**
- Harness aceita `F4_TEST_SOURCE_STATE` e **copia** o artefato (original
  intocado: sha `75509889e74257a1` antes e depois).
- Corrigido `/tmp/opencode/pop.sh`: `F4_TEST_POPULATED` estava sem `\\` na
  continuacao, por isso `F4_TEST_GPU=1` nao chegava ao teste -> modelo em CPU ->
  `position-split KV exige camada full 3 na CUDA (esta em CPU)` -> `exit=2`.
  Agora as tres variaveis vao na mesma invocacao e o teste corre.
- Localizador validado (container externo -> envelope QSKM -> cabecalho KMS +
  checksum) aplicado ao artefato copiado; original nunca e mutado.

**Verificacao independente (fora do harness), sem alinhamento:**
```
auto-...-556.bin payload@2236  QSKM hits = 0      (nenhum em todo o payload)
auto-...-571.bin payload@2296  QSKM at=2296 ver=1 frame_bytes=52053509 frame=KMS2
```
Resultado do harness com o artefato copiado:
`POPULATED: using verified producer artifact ...` ->
`FIXTURE INVALID: no mixed-KVarN envelope (QSKM) in the state payload` ->
**matriz parada (fail-fast)**, `180 checks, 5 failures` (1 fixture + 4 LIVE).

**Bloqueio exato (evidencia, nao permissao):**
O ficheiro `556` que hoje esta em `gap3-slots` **nao tem envelope misto**, mas
foi esse nome de ficheiro que logou `reused 556 tokens ... restore_ms=451`.
`gap3.sh` faz `rm -f $SLOTS/*.bin` no arranque, portanto cada execucao
apaga e reescreve; o artefato que demonstrou `cache_n=556/reprocessed=0` ja nao
e o bytes presentes agora, ou o caminho de restore daquele run nao passou pelo
leitor misto. So `571` tem `QSKM -> KMS2` (snapshot do DESTINO split, nao a
fonte pura). **Nao existe nenhum artefato KMS3 puro neste diretorio agora.**

Proximo passo imediato: rodar `gap3.sh` **uma vez com a limpeza desativada e
preservando** o ficheiro do produtor no momento do restore bem-sucedido
(registar sha + veredito do localizador nesse instante), ou reproduzir dentro do
harness a config EXATA do produtor gap3 (`test-pure-preset.ini`, sem faixa
remota) — em vez de inventar formato/identidade novos.

Producao: `active`, `health=200`, `model=qwen-3.8-27b`,
`n_params=27320697856`, `reply='42'`. `606/0`, `819/0`, `2817/0`.

## RETRACAO + RUN IMUTAVEL UNICO — 2026-10-07 03:27 UTC

### RETRACAO (obrigatoria)
**Retrato a aceitacao positiva de "puro KMS3".** Proveniencia NAO estabelecida:
o artefato produzido nao contem o envelope misto, portanto o resultado
`cache_n=556 / reprocessed=0` **nao pode ser atribuido** ao caminho
KMS3/`adopt_split_for_import`. Aquele restore provavelmente passou pelo
leitor legado. As afericoes `606/0`, `819/0`, `2817/0` nao substituem a
matriz deste caminho.

### RUN IMUTAVEL UNICO (nenhum artefato anterior tocado)
`/home/hjotha/f4-spool/run-20261007T032648Z` (diretorio novo, nenhum `rm`):
```
producer_binary = e576e38dc8539c6836a1c59b06a6112ef3edf999b6a3e36f9784adff3f03ff94  build-f4/bin/llama-server
harness_binary  = cf8d0b5c7b255964a5946e9023b17b31237506fa916b3340fa04e9384f02ca9b  test-kv-mixed-destination-intact
producer_preset = 55a2f70ee9a43a87808b019a806b96d9e8ee43a954532f79bc0b971f698f360f  test-pure-preset.ini (copiado para o run)
snapshot        = producer/auto-1ee08d954f58561c-e8b24134cbffd159-563.bin (198040602 B)
snapshot_sha_before = ec2dcc3bf564cd1b4da25a566e714f6cf65a235104e51c8089a729e876ea2bda
snapshot_sha_after  = ec2dcc3bf564cd1b4da25a566e714f6cf65a235104e51c8089a729e876ea2bda   -> IMMUTABLE
destino copiado para run-.../destination/ (separado, so-leitura no original)
```

### Gate do CODEC REAL, antes de qualquer teste de restore caro
`F4_TEST_FIXTURE_ONLY` (sem GPU, sem modelo): localizador valida o container
externo -> envelope -> cabecalho KMS (+checksum) e depois chama
`llama_kv_mixed::snapshot::parse` (**codec real**) com a extensao exata do
envelope. Veredito neste run:
```
FIXTURE: codec rejects the frame: no mixed-KVarN envelope (QSKM) in the state
         payload: the source is NOT using the intended KVarN mixed-state
         writer in this fixture
codec_gate_rc=0 -> matriz de restore NAO executada
```
Vereditos de referencia (mesmo gate, artefatos existentes, so-leitura):
`...-556.bin` -> mesmo "no QSKM"; `...-571.bin` ->
`codec_magic=KMS2 codec_version=2 layers=32 has_split=1 has_source_identity=0`
`immutable=1` -> **NAO e uma fonte KMS3** (e o snapshot do destino split).

Nenhuma inferencia por nome de ficheiro; nenhum varrimento de bytes magicos em
payload; nenhum artefato anterior limpo ou sobregravado.

### Estado F4
- positivo puro-KMS3: **RETRATADO**
- matriz do caminho real (populate/rollback/motivos/fallback): **NAO EXECUTADA**
  ate existir um artefato que o codec aceite como KMS3
- produtor precisa emitir o envelope misto (ou usar uma fonte que o emita);
  enquanto nao acontecer, o gate falha e nada roda

Producao: `active`, `health=200`, `model=qwen-3.8-27b`,
`n_params=27320697856`, `reply='42'`.

## CONTRATO DO ESCRITOR PURO CORRIGIDO +1a RUN IMUTAVEL (falhou na producao) — 2026-10-07 03:51 UTC

**Guards inspecionados e corrigidos** em `src/llama-kv-cache-kvarn.cpp`:
- `state_write` (antes: so `standard_cache -> mixed`): agora tambem usa o
  escritor misto quando `state_model_id != 0 && state_rope_id != 0` — **fonte
  PURE continua PURE** (sem ownership remoto, sem `standard_cache` fabricado),
  mas o frame passa a carregar envelope + identidade KMS3.
- guard unico (escrita **e** leitura, linhas 4792/4983):
  `(standard_cache == nullptr && position_split_enabled()) || migration || swa
  || n_stream != 1 || n_seq_max != 1` — split exige companheiro de overflow;
  PURE nao.
- `standard_cache->` isolado a `if (standard_cache)` nos tres pontos
  (descritores de overflow, payload partilhado, contagem de capacidade).
- **Leitor preservado**: `state_read_sinfo` agora despacha por conteudo no
  destino PURE: palavra inicial `QSKM` -> `state_read_mixed`; caso contrario
  replay e `state_read_kvarn_body` (frames legados KVTL/KVRN continuam legiveis).

**Testes CPU apos a alteracao (todos verdes):**
`test-kv-mixed-state 606/0`, `-stream 819/0`, `ubatch 2817/0`,
`ranges 391/0`, `merge 51/0`, `flash-attn-ext-params PASSED`.
(Dispatch/escrita propriamente ditos exigem um cache real -> passo GPU.)

**`codec_gate_rc` corrigido**: `run-fixture.sh` agora grava o output do
validador em `codec-gate.log` e captura o **status do proprio validador**
(`GATE=0` antes de qualquer pipeline), registando
`codec_gate_rc=<N> (validator's own status, not the pipeline's)`.

### Run imutavel #2 — `run-20261007T035006Z`
Registado: binary `e576e38d…`, harness `cf8d0b5c…`, preset `55a2f70e…`
(copiado para o run), diretorio novo, nenhum artefato anterior tocado.
**Resultado: NENHUM snapshot produzido.** O worker morre a meio do primeiro
pedido:
```
0.10.036 proxying request to model qwen-3.8-27b on port 46709
0.10.657 E http client error: Failed to read connection (x3)
0.10.824 I instance name=qwen-3.8-27b exited with status 1
```
**Sem mensagem de erro, sem `terminate`, sem backtrace** em `producer.log`
(214 linhas) — saida do worker perdida pelo modo proxy/router. **Nao
diagnosticado; nao vou inventar causa.** Proximo passo concreto: correr o
produtor em foreground (sem proxy/router) com stderr proprio para capturar a
mensagem real de saida, antes de qualquer nova tentativa.

**Retracao mantida**: nenhuma aceitacao positiva de KMS3. Matriz
positiva/populada **NAO executada**.

Producao: `active`, `health=200`, `model=qwen-3.8-27b`,
`fp=b12366-d9db0b881-dirty`, `reply='42'`.

## PROVENCIIA DE CODIGO CORRIGIDA + produtor em foreground — 2026-10-07 04:11 UTC

### O hash anterior nao provava nada
`build-f4/bin/llama-server` e um **stubb ELF de 16000 B** que so faz `ldd`
para as libs; por isso os dois runs gravaram o mesmo `e576e38d…` — o valor
nao muda com o fonte. Codigos reais (registados agora no run):
```
libllama.so.0            = ef796ad9038b35097d7fb48a16a5af1950bd6030caf4defc453a7619bb98b0a2
libllama-server-impl.so  = 10b66dd1ca4b73721090000334316b88d38e1422127f52c7e65f85fae1d9412b
writer_guard_present_in_libllama = 1     <-- nova string
                           "mixed state ... (and an overflow companion in split mode)"
server_wrapper(stub)     = e576e38d… (nao usado como prova de codigo)
```
`strings libllama.so.0 | grep -c "overflow companion in split mode" = 1`
→ **a alteracao do escritor esta realmente no binario carregado**.

### Produtor em FOREGROUND com stderr proprio — `run-20261007T040912Z`
```
producer_exit=124   (124 = timeout do nosso wrapper; o router viveu ate la)
  [56725] slot get_availabl: id 0 | task -1 | selected slot by LRU
  [56725] slot launch_slot_:  id 0 | task 8 | processing task
  [56725] W NVIDIA GPU memory clock lock ... skipping dynamic memory lock
  0.19.284 E http client error: Failed to read connection (x3)
  0.19.453 I instance name=qwen-3.8-27b exited with status 1
NO SNAPSHOT PRODUCED
```
Ainda **sem mensagem de erro, sem terminate, sem backtrace**: o worker morre
silenciosamente a meio do decode, imediatamente apos `launch_slot_`. Agora
reproduzido em foreground com o stderr do produtor captado — a ausencia de
mensagem nao e mais artefacto do modo proxy.

**Regressao suspeita (nao confirmada):** e a primeira execucao depois das
alteracoes de escritor/leitor. O candidato mais estreito e a **relaxacao do lado
do leitor** (`state_read_sinfo` passou a aceitar QSKM num destino PURE), que
abriu um caminho antes inalcancavel com `standard_cache == nullptr`. O dispatch
de escrita puro->misto e a parte pretendida.

### Proximo passo concreto
Bisect de uma so variavel: reverter SO a relaxacao do leitor (manter o dispatch
de escrita), reexecutar o produtor em foreground; se o crash desaparecer, o
problema esta no caminho de leitura puro->misto e deve ser corrigido la (guard
equivalente a `standard_cache == nullptr && position_split_enabled()` ja
existe; verificar os restantes `standard_cache->` nao guardados em
`state_read_mixed`). Depois repetir o run imutavel com gate do codec.

**Retracao mantida**: nenhuma aceitacao positiva de KMS3; matriz positiva e
populada **NAO executadas**.

Producao: `active`, `health=200`, `model=qwen-3.8-27b`,
`n_params=27320697856`, `reply='42'`.

## DIRECT WORKER +1a RUN IMUTAVEL COM GATE DE CODEC — 2026-10-07 04:23 UTC

### Inversao de execucao determinada a partir do codigo/config
O router proprio constroi o argv do worker (`server-models.cpp:1902`,
`spawning server instance with args`). Extraidos os **118 argumentos** desse
argv (`worker-args.txt`) e executado **directamente**:
`build-f4/bin/llama-server <argv>` — sem router, sem proxy, stderr dedicado.

```
worker_exit=124            (timeout nosso; o worker continuou a servir)
worker.err: 0 linhas de terminate/erro
0.07.617 slot auto_save_sl: persisted unified snapshot reason=prompt
         tokens=556 bytes=198387119 save_ms=387.708
0.07.831 prompt eval 1635.69 ms / 556 tokens
```
**Correccao confirmada**: a "saida silenciosa com status 1" dos runs anteriores
era artefacto do caminho router/proxy — **o produtor nao crasha**. A minha
afirmacao de que o stderr tinha sido eliminado estava errada e fica corrigida.

### Proveniencia de codigo (stubb corrigido)
```
libllama.so.0           = 22b264b10260d81d05df144157919375eba7613b1f242c1d2a872dc89c79ed96
libllama-server-impl.so = 10b66dd1ca4b73721090000334316b88d38e1422127f52c7e65f85fae1d9412b
libllama-common.so.0    = ee1bd17bb44fcb5c51af6ad28fa0a4354d665574a8252aec274f53ff14e2b22b
writer_guard_in_libllama = 1   ("... and an overflow companion in split mode")
server stub hash e576e38d... nao e usado como prova de codigo
```
Effective args registados (nao-secretos) em `worker-args.txt` (118 linhas,
mostrados no report ate 600 caracteres).

### Run imutavel `run-20261007T042032Z` + gate do codec REAL
```
snapshot = producer/auto-1ee08d954f58561c-e8b24134cbffd159-563.bin  (198866524 B)
snapshot_sha_before = 7ccb80b0e546090fb42face64b2caba35c990f15d783f80b3ff1e3c059b73d4c
snapshot_sha_after  = 7ccb80b0e546090fb42face64b2caba35c990f15d783f80b3ff1e3c059b73d4c   -> IMMUTABLE
destination/ copia separada (read-only no original)
FIXTURE bytes=198866524 xxh64_before=ffa8162487b3c9e7 xxh64_after=ffa8162487b3c9e7 immutable=1
FIXTURE codec_magic=KMS3 codec_version=3 layers=16 has_split=0 has_source_identity=0
FIXTURE source_model_id=0 source_rope_id=0 source_config_id=0
FIXTURE: NOT a genuine KMS3 source with a complete identity
codec_gate_rc=6 (validator status, not a pipeline status)
```

**Progresso real**: o frame **agora e KMS3** (antes nao havia nenhum) — a
mudanca de escritor funciona. **Gate FALHOU** (`rc=6`) porque a identidade
lida de volta e vazia/zero: o cabecalho escrito e o que `parse_impl` le nao
coincide (offset/comprego do bloco de identidade, ou `input.source` nao chega
ao header). **Matriz positiva e populada NAO executadas** (gate antes dos
testes caros). Nenhum artefato antigo limpo/sobregravado; nada inferido por
nome nem por bytes magicos em payload.

Proximo passo estreito: comparar byte a byte o cabecalho KMS3 escrito
(`build_header_v3`/`stream_serialize`) com o que `parse_impl` le em
`KMS3_IDENTITY_OFF` — resolver essa divergencia antes de qualquer outro passo.

Producao: `active`, `health=200`, `model=qwen-3.8-27b`,
`n_params=27320697856`, `reply='42'`.

## RAIZ DA IDENTIDADE CORRIGIDA (move) + MATRIZ POPULADA EM CURSO — 2026-10-07 04:58 UTC

### Causa-raiz da identidade vazia: move do `snapshot`
`parse_impl` (linhas ~899-915) **rejeita identidade zero** e ~945-947 grava
`source_`/`source_present_`, mas o **construtor de move (~1429) e o
operador de move (~1454) nao copiavam esses campos**. Como o fixture faz
`s = snapshot::parse(...)` (move-assign), a identidade perdia-se a cada
movimento — explicando "parse ok, identidade vazia" e **ameacando todos os
caminhos de producao que movem snapshots**.
Corrigido nos dois operacoes (incluindo limpeza do moved-from). Regressao nova
`test_kms3_move_identity` (+11 checks): move-ctor e move-assign, modo owning e
**borrowed** (ponteiros de payload estaveis), moved-from inerte, e
comportamento transaccional (frame movido autoriza a transacao; moved-from
recusa).

`test-kv-mixed-state: PASS **617 checks, 0 failures**` (606 -> 617).
Restantes CPU: `819/0`, `2817/0`, `391/0`, `51/0`, `PASSED`.

### Gate do codec no artefato IMUTAVEL existente (bytes nao tocados)
```
sha antes/depois = 7ccb80b0e546090fb42face64b2caba35c990f15d783f80b3ff1e3c059b73d4c (igual)
FIXTURE codec_magic=KMS3 codec_version=3 layers=16 has_split=0 has_source_identity=1
FIXTURE source_model_id=7f88559fa5250ae7 source_rope_id=7d90e892cdd78580 source_config_id=fdebea6421f53609
FIXTURE: OK -- genuine KMS3 source with complete writer identity
codec_gate_rc=0
```
`source_model_id/rope_id` coincidem com os valores vistos no log do servidor
(`model=7f88559fa5250ae7 rope=7d90e892cdd78580`) — cross-validacao.

### Novo run imutavel direct-worker `run-20261007T043359Z`
`libllama.so.0=5e6a7195718caf08c17f043499d984fdf4534e4eb0ca11cd3da6ddaa2babff63`,
`libllama-server-impl=10b66dd1…`, `writer_guard_in_libllama=1`,
`worker_exit=124` (serviu sem crash), snapshot `7ccb80b0…` IMUTAVEL,
`codec_gate_rc=0` → **proveniencia KMS3 estabelecida** (retracao levantada
para este artefacto).

### Matriz popULADA real (loader + transaccao partilhada)
```
magic=KMS3 at=2280 bytes=198866524
wrong_model_id    rejected=1 reason_ok=1 rb=1 fp_same=1
wrong_rope_id     rejected=1 reason_ok=0 rb=1 fp_same=1
identity_corrupt  rejected=1 reason_ok=1 rb=1 fp_same=1
unknown_version   rejected=1 reason_ok=1 rb=1 fp_same=1
header_corrupt    rejected=1 reason_ok=1 rb=1 fp_same=1
truncated         rejected=1 reason_ok=1 rb=1 fp_same=1
fallback repopulate bytes=56362837
```
- **6/6 rejeitados, rollback real (rb=1), fingerprint semantico inalterado
  (fp_same=1) em todos**, fallback conclui
- **5/6 motivos intencionais** corrigidos no caminho: identidade/corrupcao
  precisavam tambem do **trailing whole-file checksum** (r.digest() no fim do
  frame) alem do identity CRC, senao eram recusados antes como
  `whole-file checksum mismatch`
- destino populado com `P_DEST=1024` (a fonte ocupa 556..563 tokens; com
  `P=512` tudo recuava no gate de extento)

### Ultimo motivo errado — causa exata (nao e bug do loader)
`wrong_rope_id` loga: `pure KVarN import: source model identity does not
match the destination (source 9189689183808326375 = 0x7f88559fa5250ae7 ...)`.
A fonte e do **27B** (preset puro) e o destino do harness e o **4B**
(`Qwen3.5-4B-MTP-Q4_K_M.gguf`) → o modelo difere → a verificacao de modelo
(disposta primeiro) recusa antes da RoPE. **Nao e defeito do loader**: e o
acoplamento do fixture. Passo seguinte: produzir o artefacto com o MESMO modelo
do harness (`Qwen3.5-4B`), para que modelo coincida e o caso RoPE seja
verificavel.

Falhas restantes (14) = 4 diagnosticos LIVE conhecidos + 1 motivo de fixture +
9 pre-existentes (`separation`, `state-diff`, `layer table`, `probes`).

Producao: `active`, `health=200`, `model=qwen-3.8-27b`,
`n_params=27320697856`, `reply='42'`.

## FIXTURE DE MODELO IGUAL + POSITIVO/NEGATIVOS INDEPENDENTES — 2026-10-07 05:24 UTC

### Fixtures e regressoes novas (CPU)
- `test_kms3_move_identity` +4: **self-move** (atribuicao 
  guarda contra self-assign e mantem identidade/camadas/payload) e
  **parse -> move -> SERIALIZE** (novo frame KMS3 carrega exatamente a mesma
  identidade). Total **`test-kv-mixed-state: PASS 621 checks, 0 failures`**
  (606 -> 617 -> 621).
- Preset de produtor **no modelo do destino**: 
  (, sem faixa remota).

### Artefacto imutavel de modelo igual — `run-20261007T051643Z`
```
snapshot = producer/auto-34bba6c32a0ccf18-8fc32da3fe0614f9-521.bin  (72756494 B)
sha_before = sha_after = 0c0dcba1a3ae3a7c9318c793df058ad5590b515d40b5bd922eadaa753f277a96
codec: KMS3 v3 layers=8 has_split=0 has_source_identity=1
source_model_id=d471f8cb465cfac7 source_rope_id=7d90e892cdd78580 source_config_id=641c25d63f88b0d5
FIXTURE: OK -- genuine KMS3 source with complete writer identity   codec_gate_rc=0
libllama.so.0=5e6a7195… writer_guard_in_libllama=1 worker_exit=124
```

### Matriz com destino de modelo IGUAL — 6/6 motivos intencionais
```
magic=KMS3 at=2112 bytes=72756494
wrong_model_id    rejected=1 reason_ok=1 rb=1 fp_same=1
wrong_rope_id     rejected=1 reason_ok=1 rb=1 fp_same=1   <-- agora alcanca a verificacao de RoPE
identity_corrupt  rejected=1 reason_ok=1 rb=1 fp_same=1
unknown_version   rejected=1 reason_ok=1 rb=1 fp_same=1
header_corrupt    rejected=1 reason_ok=1 rb=1 fp_same=1
truncated         rejected=1 reason_ok=1 rb=1 fp_same=1
fallback repopulate bytes=56362837
```
Cada negativo e derivado do artefacto integro aceite, recomputando os
checksums aplicaveis (identity CRC + trailing whole-file checksum) **excepto**
o que esta deliberadamente corrompido. Fingerprint semantico inalterado
(`fp_same=1`) e rollback real (`rb=1`) em todos; fallback conclui.

### POSITIVO — ainda nao passa (falha nova e exata)
```
positive control: intact import succeeds (rc=0/72756494)      FAIL
positive control: restored token count (0 vs 521)              FAIL (cascata)
positive control: restored position watermark                  FAIL (cascata)
E state_seq_load_file_streaming: split_reader: virtual stream truncated
```
A identidade/extensao passam e a transacao chega ao `split_reader`, que
reporta **o fluxo virtual a terminar antes do fim dos eventos do manifesto**
(`llama-kv-mixed-io.cpp:716`): o manifesto da fonte pura e os spans fornecidos
nao cobrem o que o leitor consome. Isto bloqueia o positivo e as cascatas.

### Classificacao das 16 falhas (com evidencia, sem dispensa)
| classe | n | evidencia / base |
|---|---|---|
| positivo bloqueado + cascatas | 3 | `rc=0`, causa `split_reader: virtual stream truncated` |
| diagnosticos LIVE preservados | 4 | `n=129/257 k/v_stage head=1`, linha 3176 |
| outras secoes do mesmo teste | 9 | `independent reference`(1673), `separation`(1858 x2), `state-diff`(1956 x3), `layer table`(2369), `four routes`(3381), `probe matrix`(3395) |

Base relevante da mesma sessao: o run de `04:35:45` (`233 checks, 15 failures`)
tem **exactamente** essas 9 + 4 LIVE + 2 motivos de identidade = 15, portanto as
9 nao sao novidades desta alteracao; continuam **abertas** e nao sao dispensadas.

Producao: `active`, `health=200`, `model=qwen-3.8-27b`,
`n_params=27320697856`, `reply='42'`.

### CORRECAO da entrada anterior (heredoc nao-quotado comeu backticks)
Os dois primeiros pontos da entrada acima foram parcialmente substituidos por
substituicoes de comando. Conteudo correcto, para registo:

- Preset de produtor **no modelo igual ao do destino**:
  `/home/hjotha/f4-spool/test-pure-preset-4b.ini` =
  copia de `test-pure-preset.ini` com `model = /home/hjotha/models/Qwen3.5-4B-MTP-Q4_K_M.gguf`
  e sem faixa remota (`; PURE producer: remote-attn*`).

- `test_kms3_move_identity` recebeu +4 checks alem dos 11 anteriores:
  * **self-move**: `snapshot s = parse(...); snapshot & alias = s; s = std::move(alias);`
    -> versao 3, `has_source_identity()` verdadeiro, identidade igual, camadas e
    payload intactos (o `if (this != &other)` do operador garante no-op);
  * **parse -> move -> SERIALIZE**: `ser2.source = moved.frame_source_identity()`,
    novo frame KMS3, `parse` de volta -> identidade igual a original.
  Total: `test-kv-mixed-state: PASS 621 checks, 0 failures`.

Hashes do artefacto de modelo igual (imutavel, sha antes == depois):
`0c0dcba1a3ae3a7c9318c793df058ad5590b515d40b5bd922eadaa753f277a96`
`libllama.so.0 = 5e6a7195718caf08c17f043499d984fdf4534e4eb0ca11cd3da6ddaa2babff63`.

## BLOQUEIO EXATO: manifesto sem payload partilhado; proximo = recorrente 2026-10-07 07:47 UTC

### Evidencia obtida (expected/actual + primeiro evento divergente)
Instrumentacao em `llama-kv-mixed-io.cpp:716`:
```
split_reader: virtual stream truncated
  [cursor=18705471 needed=4 declared_total=18705471 events=8552 event_idx=8552
   last_event{off=18695231 size=10240 end=18705471 layer=31 range=0 kind=1}
   as_tensor=0]
```
Leitor consumiu **exactamente** `declared_total` (8552 eventos) e pediu **4 bytes
a mais** = o `uint32 kind` no topo de `state_read_shared_payload`
(`llama-kv-cache.cpp:6186`). A fonte PURE nao tem `standard_cache`, portanto o
escritor **nao escreve** esse cabecalho (guard `if (standard_cache)` que adicionei
em `state_write_shared_payload`).

### Reparo (mapeamento/extensao; sem padding, sem afrouxar limites)
`llama-kv-cache-kvarn.cpp:5337` passa a ler o payload partilhado **so quando o
proprio manifesto declara bytes**:
`if (reader.remaining() > 0) { standard_cache->state_read_shared_payload(...); }
 else { LLAMA_LOG_INFO("pure source declares no shared/overflow payload;
 leaving the companion overflow empty"); }`
Isto e exactamente o §3.6 "preencher local e deixar overflow vazio". Fontes que
declaram bytes continuam a passar pelo leitor sem qualquer alteracao de limite.

### Efeito no MESMO artefacto imutavel aceite (positivo antes dos negativos)
`destination-intact: 243 checks, **9 failures**` (antes 16):
- `split_reader: virtual stream truncated` **desapareceu**
- `independent reference`, `state-diff` (x3), `layer table`, `four routes`,
  `probe matrix` — **passam agora** (7 falhas resolvidas pela mesma causa)
- **6/6 negativos continuam `rejected=1 reason_ok=1 rb=1 fp_same=1`**
- `POPULATED positive import` registado **antes** dos negativos

### NOVO bloqueio exacto do positivo (mantido aberto)
```
state_read_mixed: pure source declares no shared/overflow payload; leaving the companion overflow empty
state_seq_load_file_streaming: failed to restore kv cache; empty destination cleared
```
`failed to restore kv cache` vem de `src/llama-memory-recurrent.cpp:969-977`
(`state_read_meta` devolveu **false**) — a restauro do estado **recorrente /
DeltaNet**. O leitor ja consumiu todo o manifesto (`cursor == declared_total`), ou
seja o manifesto da fonte PURE nao cobre a secção recorrente. Proximo passo
estreito: comparar o que `state_write` da fonte PURE escreve para
`llama_memory_recurrent` com o que `state_seq_read_data` le na sequencia, e
especificar quem pertence ao manifesto v2 vs à secção seguinte.

### As 9 falhas mantidas, com evidencia (nenhuma dispensada)
| classe | n | evidencia |
|---|---|---|
| positivo bloqueado + cascatas | 3 | `rc=0`, causa `failed to restore kv cache` (`llama-memory-recurrent.cpp:969`) |
| diagnosticos LIVE preservados | 4 | `n=129/257 k/v_stage head=1`, linha 3176 |
| `separation n=1`, `separation n=256` | 2 | linhas 1858 (presentes tambem na base de `04:35:45`) |

### Dono / entregavel
O entregavel real e **Qwen 27B com placement por ocupacao**; o 4B e apenas
instrumento de depuracao limitado. Caminho critico: (1) fechar positivo + matriz
atomica independente, (2) consumir ja a preparacao F5 real 27B 128K
(placement/memory/numerical/performance), (3) canario F7 + verificacao de
producao. Contagens CPU nao substituem aceitacao 27B real.

Producao: `active`, `health=200`, `model=qwen-3.8-27b`,
`n_params=27320697856`, `reply='42'`.
## RECORD 2026-10-07 06:23 UTC — proximo leitor / contagens exactas (recorrente)

## PROXIMO LEITOR / CONTAGENS EXACTAS DO ESTADO RECORRENTE

### Ordenacao (fonte e destino) verificada no wrapper hibrido
llama-memory-hybrid::state_write (linha 626): mem_attn->state_write DEPOIS
mem_recr->state_write. llama_memory_hybrid::state_read (linha 635): mesma ordem.
Simetrico. O manifesto KVarN cobre SO o segmento de atencao; os bytes
recorrentes vao DEPOIS do frame e nao pertencem ao manifesto (confirmado por
leitura de codigo, nao presumido).

### Byte-ownership medido no artefacto imutavel aceite
n_tokens=521  payload_off=2096  QSKM@2096  envelope ver=1 frame_bytes=19068677
frame_end=19070789  remaining=53685705  segundo QSKM: nenhum
bytes no fim do frame = 01000000 08020000 00000000 00000000 20000000 ...
                        ^cell_count=1  ^pos=520 (=521-1)  ^n_seq_id=0

Metadata recorrente ESPERADO: cell_count=1, pos=520, n_seq_id=0 — formato de
sequencia unica (ramo dest_seq_id != -1 de llama_memory_recurrent::state_read_meta,
llama-memory-recurrent.cpp:1163). Os bytes estao la e sao coerentes.

### Primeira divergencia real (excecao antes escondida)
O catch (...) { res = false; } em llama-memory-recurrent.cpp engolia a excecao e
dava o generico failed to restore kv cache. Agora regista-a:

state_read: recurrent state_read_data failed for cell_count=1
            at stream offset=23527569: truncated sequence state file

Ou seja: state_read_meta NAO falhou (nenhum log de invalid seq_id-agnostic kv
cell / failed to find available cells / not enough cells); falhou state_read_data
com EOF no offset 23527569.

Comparacao de extensoes:
  frame_end            = 19070789
  tamanho do ficheiro  = 72756494
  estado (payload)     = 72756494 - 12 - 521*4 = 72754398
  offset da falha      = 23527569   (4456780 bytes apos o frame)
Todos maiores que a falha: o io que o leitor recorrente ve termina cedo demais
(ou o recorrente declara mais do que escreveu). Nao e o manifesto KVarN, nao e
padding, nao e limite relaxado.

### Defeito estreito a corrigir
A extensao/limite do stream entregue ao leitor recorrente nao corresponde ao que
o escritor declarou/escreveu para a secção recorrente. Proximo passo: comparar
state_write de llama_memory_recurrent (o que escreve) com state_read_data (o que
le) para cell_count=1 e fixar o limite/mapeamento exacto — preservando os
caminhos PURE, legado e split, os guards antes de qualquer dereference de
standard_cache e o rollback atomico KV/tabela/DeltaNet.

### Status (mantidos abertos, com evidencia)
destination-intact: 243 checks, 9 failures
  - 3: positivo bloqueado + cascatas (rc=0, causa truncation acima)
  - 4: diagnosticos LIVE preservados (n=129/257 k/v_stage head=1, linha 3176)
  - 2: separation n=1 e n=256 (linha 1858, tambem na base de 04:35:45)
split_reader: virtual stream truncated — RESOLVIDO (7 falhas do run anterior
  passaram: independent reference, state-diff x3, layer table, four routes,
  probe matrix)
6/6 negativos continuam rejected=1 reason_ok=1 rb=1 fp_same=1

### Entregavel / caminho critico
O entregavel real e Qwen 3.8 27B com placement por ocupacao; o 4B e apenas
instrumento de depuracao limitado, nunca aceitacao final. Depois deste gate:
(1) fechar positivo + matriz atomica independente,
(2) consumir a preparacao F5 ja existente de 27B 128K (placement/memory/
    numerical/performance),
(3) canario F7 + verificacao de producao. Contagens CPU nao substituem
aceitacao 27B real. Sem migracao de worktree/branch; binarios de producao
(build-optimized) e configuracao do servico preservados.

Producao: active, health=200, model=qwen-3.8-27b, n_params=27320697856,
fp=b12366-d9db0b881-dirty, reply=42.
## RECORD 2026-10-07 06:44 UTC — extent exacto da falha positiva

## EXTENT EXACTO DA FALHA POSITIVA (readers identificados)

### Quem lanca o texto (duas classes distintas)
- streaming: llama_io_read_file_stream::read, src/llama-io-file.h:26
  (count > size - consumed)
- nao-streaming: check_file_remaining, src/llama-context.cpp:5336-5341
  (position > end || size > end - position)
Ambas receberam contexto limitado (sem dump de payload):
  class, total/position/end, consumed, requested, available, physical_tell.

### Contexto capturado no run real
Truncacao negativa (esperado, caso deliverable):
  [class=llama_io_read_file_stream total=24252164 consumed=23527569
   requested=2097152 available=724595 physical_tell=23527569]
  physical_tell == consumed -> NAO ha divergencia de cursor logico/fisico;
  o staging/commit/seek nao moveu o cursor.

Positivo (mesmo artefacto imutavel aceite, antes dos negativos):
  state_seq_load_file_streaming: sequence state file integrity check failed
  [extent read=71762329 expected=72756494]

### Onde falta bytes (conta exacta)
  ficheiro                  = 72756494
  cabecalho+tokens           = 12 + 521*4 = 2096
  conteudo de estado escrito = 72754398
  conteudo de estado lido    = 71762329 - 2096 = 71760233
  FALTA                     = 994165 bytes
Localizacao: o frame KVarN termina em offset 19068693 do conteudo de estado
(envelope 16 + frame_bytes 19068677). Secao recorrente escrita =
53685705 bytes; lida = 52691540 bytes -> o RECORRENTE le 994165 bytes a MENOS
do que o escritor escreveu.

Conclusao: nao e limite do leitor (o limite e o ficheiro inteiro, e o cursor
logico == fisico) nem manifesto KVarN (fecha exactamente em 19068693). E a
EXTENSAO das tensores recorrentes declarada pelo leitor a diferir da escrita.

### Proximo passo estreito (nao e diagnostico repetido)
Comparar, para cell_count=1 e a MESMA configuracao, o que
llama_memory_recurrent::state_write escreve (meta + payload de tensores:
quantas celulas, que tensores, que linhas) com o que state_read_data le, no
caminho PURE -> split. Em particular verificar se a configuracao recorrente
(n_rs_seq / rs_size / n_pos_per_embd / n_embd) do destino difere da do
produtor puro, o que mudaria o numero de bytes lidos.

### Estado mantido (9 falhas abertas, com evidencia)
destination-intact: 243 checks, 9 failures
  3 positivo+cascata (rc=0, causa integrity/extents acima)
  4 diagnosticos LIVE preservados (n=129/257 k/v_stage head=1, linha 3176)
  2 separation n=1 / n=256 (linha 1858)
6/6 negativos continuam rejected=1 reason_ok=1 rb=1 fp_same=1.
Rollback atomico, guards PURE/legado/split e produção (build-optimized e
unidade do servico) preservados; sem worktree/branch.

Entregavel: Qwen 3.8 27B com placement por ocupacao; 4B = instrumento limitado.
Depois: (1) positivo + matriz atomica independente, (2) F5 real 27B 128K ja
preparado, (3) canario F7 + verificacao de producao.

## CONTRATO CORRECTO DO FIXTURE (unified native/footer) + POSITIVO A ABRIR — 2026-10-07 07:18 UTC

A fonte imutavel e um snapshot UNIFICADO do servidor (nativo + footer), nao um
ficheiro nativo puro. Corrigido no harness para usar a MESMA abstraccao de
producao (server_route_state_read) e passar a MESMA dupla ao API inferior:
file.state_bytes + file.state_checksum (server-context.cpp:4093). Antes passava
pure.size() e XXH64 sobre o contentor inteiro — contrato errado.

[POSITIVE] container file_bytes=72756494 native state_bytes=71762329
           state_checksum=dc96a05718d9525f
           logits_offset=71762329 logits_bytes=993280
           footer=[71762329,72756494) tokens=521 position=520
[POSITIVE] import rc=71762329 tokens=521 pos=520 logits=248320
           max_abs_diff=1.229927e+00

Resolucao total do falso "shortfall": 994165 bytes = footer unificado
(logits 993280 + sobra do manifesto 885). O estado nativo era lido por
inteiro; nao ha defeito de extent do leitor DeltaNet/recorrente.

[NEG:*] os 6 casos, rotulados e etiquetados, com o contrato nativo correcto
(native_extent + XXH64 so do prefixo nativo, recomputado por mutacao, excepto o
deliberadamente corrompido):
  wrong_model_id    rejected=1 reason_ok=1 rb=1 fp_same=1
  wrong_rope_id     rejected=1 reason_ok=1 rb=1 fp_same=1
  identity_corrupt  rejected=1 reason_ok=1 rb=1 fp_same=1
  unknown_version   rejected=1 reason_ok=1 rb=1 fp_same=1
  header_corrupt    rejected=1 reason_ok=1 rb=1 fp_same=1
  truncated         rejected=1 reason_ok=1 rb=1 fp_same=1

destination-intact: 245 checks, 7 failures (antes 16 -> 12 -> 7):
  1  positivo: continuacao numerica max_abs_diff=1.229927 (ABERTO)
  2  separation n=1 / n=256 (linha 1905)
  4  diagnosticos LIVE preservados (n=129/257 k/v_stage head=1, linha 3223)

Gates CPU preservados: 621/0, 819/0, 2817/0, 391/0.
Producao preservada: build-optimized intocado, servico intacto.

## DISCRIMINADOR: a divergencia NAO e configuracao/layout 2026-10-07 07:35 UTC

Congelado o fixture imutavel e os hashes de build em
/home/hjotha/f4-spool/run-20261007T051643Z/FROZEN.txt (permissoes a-w).

Novo comparador compaco no mesmo harness (sem restore):
  [DISCRIM] split_prefill vs pure_prefill max_abs_diff=0.000000e+00

Resultado: os dois destinos (split e pure, ambos make_ctx) produzem logits
IDENTICOS para os mesmos tokens e a mesma continuacao de 1 token. Logo:
  - nao e diferenca de layout (split vs pure)
  - nao e configuracao/batching do make_ctx vs preset do servidor
  - nao e RoPE/KVarN/n_ctx/n_batch
O gap restante (max_abs_diff=1.229927 entre restore e prefill frio) esta no
CAMINHO DE PERSISTENCIA/MAPEAMENTO do proprio restore nativo.

Estado: destination-intact 245 checks, 7 failures
  1  continuacao numerica (max_abs_diff=1.229927)  -- ABERTO, gate canonico
  2  separation n=1 / n=256
  4  diagnosticos LIVE (nao sao permissao para mudar gates)
6/6 negativos continuam rejected=1 reason_ok=1 rb=1 fp_same=1.

Proxima perna pedida (nao executada por falta de budget): mesma fonte imutavel
restaurada em PURE vs split, e prefill vs restore no mesmo layout, comparando
estado semantico KV/tabela/DeltaNet para achar a primeira divergencia. Nota:
exige o despacho de conteudo no destino PURE (relaxacao que reverti durante o
bissect anterior) — a confirmar antes de a reaplicar.

Producao preservada: build-optimized intocado, servico intacto,
active/health=200/qwen-3.8-27b/n_params=27320697856/reply=42.

## CORRECAO DE INFERENCIA + TABELA DISCRIMINANTE COMPLETA — 2026-10-07 07:47 UTC

**Correcao obrigatoria da entrada anterior:** "split_prefill vs pure_prefill = 0"
prova apenas que os DOIS caminhos FRIOS com make_ctx concordam. NAO compara
make_ctx com o PRESET DO SERVIDOR produtor, portanto NAO descarta configuracao
do produtor, batching, checkpoint ou desalinhamento de posicoes de tokens.
A anterior "persistence e a causa" fica RETIRADA ate esta evidencia.

**Despacho de conteudo PURE reposto** (narrow, o que o bissect especulativo
reverteu), preservando parsing legado e guards de ponteiro:
- state_read_sinfo: palavra inicial QSKM -> state_read_mixed; senao replay ->
  state_read_kvarn_body (KVTL/KVRN legado intacto)
- guard: if (standard_cache && reader.remaining() > 0) no payload partilhado

**Tabela discriminante (mesmo artefacto imutavel, mesma continuacao de 1 token):**
  [PRODUCER-EFF] ctx-size=? batch=? ubatch=? cache-k=kvarn4 cache-v=kvarn4
                 flash-attn=on remote=none rope-scaling=none seed=?
  [MAKE-CTX-EFF] ctx-size=2048 batch=256 ubatch=256 kvarn=K4V4_G128
                 type_k=F16 flash-attn=ENABLED remote=none
  split_prefill vs pure_prefill   = 0.000000e+00
  pure_restore   vs pure_prefill  = 1.229927e+00
  pure_restore   vs split_restore = 0.000000e+00
  split_restore  vs split_prefill = 1.229927e+00
  [DISCRIM] pure_restore rc=71762329 tokens=521 pos=521   <-- pos=521
  [POSITIVE] import ... tokens=521 pos=520                <-- pos=520

**Leitura (ainda sem fix):**
1. MAPEAMENTO pure->split e EXACTO (pure_restore vs split_restore = 0).
2. Layout frio split vs pure identico (0).
3. A divergencia existe ja no restauro NATIVO (pure_restore vs pure_prefill =
   1.229927) — nao e a conversao.
4. **Nova discrepancia concreta e observavel:** a MESMA fonte, o MESMO loader,
   devolve pos_max=521 no destino PURE e pos_max=520 no destino split, e a
   fonte declara position=520. Um destino guarda uma posicao extra. E esta a
   pista imediata (pos icorrecto muda a continuacao).

Notas: os flags --ctx-size/--batch-size nao existem em worker-args.txt (o
router passa --ctx-size-long/short etc.), por isso ctx-size/batch/ubatch sao "?".
 passaram ao repor o despacho PURE (5 falhas agora:
1 numerica + 4 LIVE).

destination-intact: 5 failures (1 continuacao numerica ABERTA, 4 LIVE diagnostico).

### CORRECAO (heredoc nao-quotado comeu uma frase)
Frase que faltou na entrada acima: "separation n=1/n=256 PASSARAM ao repor o
despacho de conteudo PURE — ficaram 5 falhas (1 numerica + 4 LIVE)".

Resumo de estado:
- destination-intact: 5 failures (1 continuacao numerica ABERTA = 1.229927;
  4 diagnosticos LIVE, nao sao permissao para mudar gates)
- mapeamento pure->split EXACTO (pure_restore vs split_restore = 0)
- divergencia presente no restauro NATIVO puro (pure_restore vs pure_prefill)
- discrepancia concreta: mesma fonte/mesmo loader -> pos_max=521 (destino PURE)
  vs pos_max=520 (destino split) vs position=520 declarado
- flags --ctx-size/--batch-size nao constam de worker-args.txt (o router usa
  --ctx-size-long/short), daí "?" no bloco PRODUCER-EFF
- 6/6 negativos: rejected=1 reason_ok=1 rb=1 fp_same=1
- CPU: 621/0, 819/0, 2817/0; producao intacta

## CORRECOES DE REVISAO APLICADAS + CONTROLE EQUIVALENTE DO PRODUTOR — 2026-10-07 08:04 UTC

### 1) A "divergencia de posicao" era instrumentacao (retirada)
- pos agora registada ANTES e DEPOIS da continuacao nos DOIS restauros:
  pure_restore  pos_before=513 pos_after=514
  split_restore pos_before=513 pos_after=514
- identicos; a 521 vs 520 anterior era fase (um impresso apos a continuacao,
  outro antes). Nenhuma logica de posicao foi alterada.

### 2) Factos do produtor confirmados no worker.err do run 051643
- prompt eval: 514 tokens; snapshot de prompt: 514 tokens (position=513)
- discard tokens=521 (a pos geracao)
- n_ctx_slot = 4096  (referencia make_ctx usava 2048)
- tier flags: worker-args.txt contem --batch-size-short 256,
  --ubatch-size-short 256, --cache-type-k-short kvarn4, mas NAO contem
  --ctx-size-short nem --seed (defaults do preset/adaptive)

### 3) Controle equivalente do produtor (514, parser canonico, batch 256, n_ctx 4096)
Novos knobs no harness: F4_N_CTX (n_ctx efectivo) e F4_REF_BATCH (lote de
prefill da referencia) + prefill_batched() — o laco de 1 token era um escalonamento
quantizado/recorrente diferente e nao podia ser o controlo.

  [REF-BATCH] 256
  [PRODUCER-EFF] batch-short=256 ubatch-short=256 cache-k=kvarn4 cache-v=kvarn4
                 flash-attn=on remote=none  ctx-size-short=? seed=?
  [MAKE-CTX-EFF] ctx-size=4096 batch=256 ubatch=256 kvarn=K4V4_G128 type_k=F16
                 flash-attn=ENABLED remote=none
  [POSITIVE] container file_bytes=72516547 native state_bytes=71522383
             state_checksum=1b6aa15878886951 logits_offset=71522383
             logits_bytes=993280 footer=[71522383,72516547) tokens=514 position=513
  [POSITIVE] import rc=71522383 tokens=514 pos=513 logits=248320
             max_abs_diff=7.102920e-01
  [DISCRIM]  pure_restore vs pure_prefill  = 7.102920e-01
  [DISCRIM]  pure_restore vs split_restore = 0.000000e+00
  [DISCRIM]  split_restore vs split_prefill = 7.102920e-01
  [DISCRIM]  split_prefill  vs pure_prefill = 0.000000e+00

Efeito das correcoes: max_abs_diff 2.005971 -> 0.710292 (batch 256 + n_ctx 4096).

### Leitura corrigida
- pure_restore vs split_restore = 0 e evidencia ESPECIFICA DESTE FIXTURE, nao
  prova todas as rotas.
- a igualdade split_prefill vs pure_prefill so cobre os dois caminhos frios de
  make_ctx; NAO compara com o preset do produtor.
- Restam 0.710292 entre restauro e prefill frio ja com lote e n_ctx iguais.
  Candidatos restantes medidos/registrados: (a) preset KVarN do produtor
  "kvarn4" (common/arg.cpp:394 devolve 4) vs LLAMA_KVARN_K4V4_G128 hardcodado
  em make_ctx; (b) --ctx-size-short e --seed ausentes de worker-args.txt, ou
  seja o worker usou defaults do preset/adaptive.

destination-intact: 5 failures (1 continuacao numerica ABERTA; 4 LIVE diagnostico).
6/6 negativos: rejected=1 reason_ok=1 rb=1 fp_same=1.
CPU 621/0, 819/0, 2817/0. Producao intacta.

## DISCRIMINADOR DE SERIALIZACAO (mesmo processo) — RESULTADO

Construcao da referencia via o MESMO caminho de configuracao comum do produtor:
common_params_parse(argv do produtor reconstruido com pareamento flag/valor) ->
common_context_params_to_llama. Erro anterior ("invalid argument: 256") era do
pareamento ingenuo flag/valor (booleanos sem valor); corrigido.

  [PRODUCER-CFG] n_ctx=4096 n_batch=2048 n_ubatch=512 flash_attn=1
                 rope_base=0.0 rope_scale=0.0 kv_tail_tokens=0 kv_unified=0
  (n_batch/n_ubatch depois fixados em 256/256 = short-tier lido em worker-args;
   split acrescenta SO o stripe remoto Q4 que o produtor puro nao tem)

  [SELF-RT] save_bytes=71522383 load_ok=1 save_logits=248320 load_logits=248320
  [SELF-RT] save->load (same process, same params) max_abs_diff=0.000000e+00
  [SELF-RT] harness_cold vs server_written max_abs_diff=0.000000e+00
  [POSITIVE] import rc=71522383 tokens=514 pos=513 max_abs_diff=7.102920e-01
  [DISCRIM] pure_restore vs pure_prefill  = 7.102920e-01
  [DISCRIM] pure_restore vs split_restore = 0.000000e+00
  [DISCRIM] split_restore vs split_prefill= 7.102920e-01
  [DISCRIM] split_prefill  vs pure_prefill = 0.000000e+00

Leitura:
1. SERIALIZACAO e EXACTA no mesmo processo (save->load = 0) — save/load nao e
   a causa. (O baseline usou llama_state_seq_save_file + llama_state_seq_load_file;
   usar llama_state_load_file com um ficheiro de sequencia dava load_ok=0.)
2. MAPEAMENTO pure->split continua exacto (0).
3. Resta a diferenca entre o ESTADO ESCRITO PELO SERVIDOR e um prefill frio do
   harness com os mesmos tokens: 0.710292. Isto isola o problema no caminho de
   escrita/usado pelo produtor (ou nos tokens/posicoes que ele gravou), nao na
   maquina de restauracao.
4. NOTA: [SELF-RT] harness_cold vs server_written = 0 e trivialmente 0 porque
   compara dois prefills frios do harness; nao prova nada sobre o ficheiro do
   servidor. Rotulo corrigido em espírito: nao usar como evidencia.

destination-intact: 5 failures (1 numerica ABERTA 0.710292; 4 LIVE).
6/6 negativos verdes. CPU 621/0, 819/0, 2817/0.

## [DISC1] DISCRIMINANTE EXACTO: restauro NATIVE STREAMING -> PURE e SPLIT — EXATO

Um unico estado canonico produzido pelo proprio harness, com run id imutavel,
mesmos tokens/checkpoint/config, metadados canonicos e logits completos:

  [DISC1] run=run-11963698873384135422-harness-native adopted=1 baseline_logits=248320
  [DISC1] run=... meta file_bytes=71522589 state_bytes=71522383
          state_checksum=45a8aac1d8fea7a1 tokens=514 position=513
  [DISC1] run=... route=pure  rc=71522383/71522383 tokens=514 pos=513
          logits=248320 max_abs_diff=0.000000e+00
  [DISC1] run=... route=split rc=71522383/71522383 tokens=514 pos=513
          logits=248320 max_abs_diff=0.000000e+00

Rotulos construidos a partir do run id e da ROTA REAL percorrida (pure/split),
nao de alegacoes manuais. Fonte: `server_route_state_adopt_native` (o mesmo
embrulho canonico usado pelo servidor) + `server_route_state_read` + 
`llama_state_seq_load_file_streaming` com state_bytes/state_checksum exactos.

Resultado: o caminho de restauro NATIVE STREAMING e EXATO nos DOIS destinos
(PURE e SPLIT) face a continuacao de base com os mesmos bytes — max_abs_diff=0.

Consequencia: os 0.710292 restantes estao entre o estado PRODUZIDO PELO
SERVIDOR e esta base do harness, ou seja no proprio conteudo/schedule do
produtor (prefill em lote do servidor, template/prompt, posicoes gravadas) —
NAO na maquina de restauracao, NAO na conversao pure->split, NAO na serializacao.

Nota: isto e um discriminante limitado, NAO substitui o gate F4 de processo
novo com servidor real. O gate numerico mantem-se ABERTO (0.710292).

destination-intact: 5 failures (1 numerica ABERTA; 4 LIVE diagnostico).
6/6 negativos verdes. CPU 621/0, 819/0, 2817/0.

## [DISC2] BASELINE DO PRODUTOR LIDA — E ALINHADA POR POSICAO, NAO COMPARAVEL ASSIM

  [DISC2] producer_logits bytes=993280 vocab=248320 tokens=514 read=1 count=248320
  [DISC2] restored vs PRODUCER-STORED continuation max_abs_diff=1.078938e+01

O payload de logits do produtor existe e foi lido com o vocabulario declarado
(meta.logits_vocab) — antes read=0 porque passei expected_vocab=0.

**Nao e um gate:** o no-alinhamento e de POSICAO. O produtor guarda os logits
da posicao 513 no momento do save, enquanto o harness restaura 514 tokens
(pos_max=513) e volta a decodificar o ultimo token ja presente, produzindo
logits para a posicao 514. Comparar esses dois vectores da 10.79 por
desalinhamento, nao por divergencia de estado.

Gate REAL em aberto (alinhado: mesmos 514 tokens, mesma posicao de continuacao):
  restored vs cold prefill = 0.710292  (ABERTO, nao relaxado)

Milestone preservado e exacto:
  [DISC1] run=run-11963698873384135422-harness-native
    meta state_bytes=71522383 state_checksum=45a8aac1d8fea7a1 tokens=514 position=513
    route=pure  max_abs_diff=0.000000e+00
    route=split max_abs_diff=0.000000e+00

6/6 negativos: rejected=1 reason_ok=1 rb=1 fp_same=1.
destination-intact: 5 failures (1 gate 0.710292 ABERTO; 4 LIVE diagnostico).
CPU 621/0, 819/0, 819/0, 2817/0. Producao intacta.

## F4 ACTUAL-SERVER ONE-FIXED-NEXT-TOKEN COMPARATOR: VERDICT FAIL (alinhado, reproduzivel)

Metodo (servidor REAL, presets/args existentes, sem mudanca em restore):
  P  freeze      : produtor PURE grava o prefixo de 556 tokens; hook grava os 556 ids
  A  baseline    : produtor NOVO, slot-save-path VAZIO (cold), /completion com 557 ids
                   (556 prefix + token fixo 791) -> capture output_pos=556, cache_n=0
  B  target      : processo NOVO position_split, diretorio de candidatos com SO o
                   snapshot congelado de 556 (o de 557 nunca entra), restaura 556 do
                   disco e avalia o MESMO token 791 -> capture output_pos=556, cache_n=556

Run: /home/hjotha/f4-spool/cmp-20261007T101915Z  (variante: cmpq4-20261007T102040Z)
  auto-restore: importing PURE KVarN snapshot ... (556 tokens) into the split destination
  auto-restore: reused 556 tokens from disk (in-memory match was 0),
                bytes=198387119 restore_ms=383.379 workspace=52428800
  prefixo imutavel: sha=95f65311fdf661da86de71ab57ba3d780acc91d06317051c3a732e8a7d8c1cf1 (identico antes/depois)

Bindings (todos verificados pelo comparador antes de qualquer numero):
  token fixo id=791 @ posicao 556, avaliado pelas DUAS fases; tokens_n=557
  seq_hash d972974f112ecb97 igual; input_pos=output_pos=556; pos_max=556
  n_params=27320697856, n_ctx=4096, vocab=248320 iguais; config_hash igual
  baseline cache_n=0 (cold); target cache_n=556, evaluated_suffix=1
  layout_hash DIFERENTE: 9c494171cb7b6417 (producer) vs 58a99e8d3506cae8 (target)
  target layout: kv_layout_known=true, mixed_kv=true, mixed_kvarn_type=9,
                 mixed_kvarn_bits_k/v=4, mixed_layer_count=64,
                 16 camadas remotas em Vulkan0 com type_k/type_v=2 (q4_0)

VEREDITO sob os gates canonicos (tests/test-position-split-real-pair.cpp):
  NRMSE(O)=8.046505e-02  (<=1e-3) FAIL
  max|O|  =1.056292e+00  (<=1e-2) FAIL
  |dLSE|  =3.560000e-02  (<=1e-2) FAIL
  argmax  : producer=279 target=279 (mesmo -> nao e divergencia de selecao de token)
  state_bytes: producer=197459209  target=207731093

REPRODUZIBILIDADE: a variante SEM o override de env q4_0 deu numeros IDENTICOS bit a
bit (restore_ms 361.391 vs 383.379) e o MESMO layout_hash de destino -> a banda remota
q4_0 vem do PRÓPRIO preset (test-split-preset.ini), nao do override. Ou seja: o destino
split, como configurado, quantiza 16 das 64 camadas em q4_0 contra uma baseline PURE
KVarN4, portanto paridade exata nao e alcancavel por construcao nessa comparacao.

=> F4 continua ABERTO. F5/F7 NAO avancados (gates nao satisfeitos).
DISC1 preservado (pure/split 0.000000e+00) e 6/6 negativos mantidos; matriz populada nao
reexecutada conforme instrucao. Hook e build-f4 apenas (F4CAP=2 no build-f4, 0 no
build-optimized). Producao inalterada e restaurada.

## F4 ACTUAL-SERVER COMPARATOR — CORRIGIDO PARA AGENDA COMPARADA: **PASS** (bit-exato)

### Retracao de duas afirmacoes anteriores
1. **"cold 557 vs 556+1 e falha do motor" — RETRATADO.** O veredito anterior
   (NRMSE 8.046505e-02 / max|O| 1.056292e+00 / |dLSE| 3.56e-02) comparava um prefill
   frio de 557 tokens contra 556-do-disco + 1 token. Isso NAO e um par combinado:
   o agendamento batch/recurrent/quantizado difere. **Nao e prova de falha do motor.**
2. **"banda remota q4_0 torna a paridade impossivel por construcao" — RETRATADO.**
   Afirmacao baseada em tipos configurados, nao em evidencia. Com P=1024 e 557 tokens
   (< P) o contrato de ocupacao exige propriedade KVarN local e overflow vazio/inativo;
   tipos estaticamente configurados nao provam computacao Q4 real.

### Metodo (agenda comparada, sem mudanca de quantizacao)
  A  baseline RAM: o produtor REAL que avaliou e salvou os 556 tokens CONTINUA VIVO;
     req1 = 556 ids (cache_n=0, frio) ; req2 = 556 ids + token fixo 791 (cache_n=556,
     cache_lcp_n=556 -> reuso EM RAM dos mesmos 556) -> capture pos 556
  B  restore disco: processo NOVO position_split, diretorio de candidatos com SO o
     snapshot congelado de 556 ; req = 557 ids -> restaura 556 do disco
     (cache_n=556, cache_lcp_n=0) e avalia o MESMO sufixo 1 -> capture pos 556
  (Fase C split-RAM nao foi executada: A vs B passou.)

Run: /home/hjotha/f4-spool/cmpm-20261007T102824Z

### Estado do prefixo registrado ANTES do append
  bin=auto-1ee08d954f58561c-3d5802841d3db425-556.bin sha=3ffe4aa267f9119fac64d4ec4018e33a69ca93609348d8fc05889506b24f03f6
  capture: pos_max=555 state_bytes=197390727 tokens_n=556 seq=10e6fcc8012c85bf
  pos-append: sha identico (imutavel)
  reuso real: reused 556 tokens from disk (in-memory match was 0),
              bytes=198387122 restore_ms=400.095 workspace=52428800

### Verd sob gates canonicos
  counters: A tokens_n=557 cache_n=556 | B tokens_n=557 cache_n=556 evaluated_suffix=1
  A: cache=live/committed   B: cache=disk/committed
  NRMSE(O)=0.000000e+00 (<=1e-3) ok | max|O|=0.000000e+00 (<=1e-2) ok | |dLSE|=0.000000e+00 (<=1e-2) ok
  argmax 279 = 279 ; n_params=27320697856 ; n_ctx=4096 ; pos_max=556 ; seq_hash d972974f112ecb97
  VERDICT: PASS (exatamente 0.0, nao apenas dentro do gate)

### Evidencia de ocupacao (limitada ao que o proprio servidor emitiu)
  A (PURE): linhas F4TAIL com split_p=0
  B (split): linhas F4TAIL com split_p=1024 (P efetivo chegando ao grafo)
  construcao de tensor de cauda: A 160 linhas com k_tail=1 ; B 32 de 160
  -> a contagem acima e PRESENCA DE TENSOR, nao despacho de tokens; nao e usada como
     prova. A prova de nivel de resultado e o proprio 0.0 bit-exato: com 557 < P a banda
     remota nao contribuiu para os logits. Extracao de n_kv/kv_tail_tokens por camada
     ficou defeituosa e NAO foi afirmada aqui.
  Observacao de determinismo: o container de 556 deste run tem 3 bytes a mais que o do
  run ancora (198387122 vs 198387119) com o MESMO nome/bloco_hash/identidade; o
  resultado numerico foi identico (0.0).

### Consequencia
O open item do harness "positive control max_abs_diff=0.710292" tem a MESMA raiz
(confronto prefill frio de N contra cache-N+1) e nao deve ser lido como divergencia de
estado. F4: importacao em disco para destino position_split e numericamente exata.

## F4 CHECKLIST — FECHAMENTO BASEADO EM EVIDENCIA (2026-10-07)

### Milestone preservado
  run  : /home/hjotha/f4-spool/cmpm-20261007T102824Z  (diretorio chmod a-w)
  prefixo congelado sha256=3ffe4aa267f9119fac64d4ec4018e33a69ca93609348d8fc05889506b24f03f6
  veredito A(RAM, produtor vivo, cache_lcp_n=556) vs B(disco, in-memory match 0, lcp 0):
  NRMSE(O)=0.000000e+00  max|O|=0.000000e+00  |dLSE|=0.000000e+00  argmax 279=279  -> PASS
  Importacao em disco -> destino position_split e numericamente EXATA para esta rota.

### RETIRADO do checklist como gate
  - 4 verificacoes "LIVE single-byte" (n=129/257, k_stage/v_stage, head=1):
   Diagnosticos opcionais de mutacao viva, NAO checks canonicos. Continuam registradas
    como diagnostico, sem efeito de gate e sem bloquear F4.

### FECHOU (com evidencia)
  [x] Prefixo congelado + hash, imutavel antes/depois do append
  [x] Produtor real mantido vivo; A com reuso EM RAM dos mesmos 556 (cache_lcp_n=556)
  [x] B: processo NOVO, SO o prefixo de 556 no diretorio de candidatos (files=1),
      restaura 556 do disco (in-memory match 0) e avalia o MESMO sufixo 1
  [x] Agenda comparada: A e B avaliam exatamente o mesmo sufixo (cache_n=556, suffix=1)
  [x] Gates canonicos de logit completo: 0.0 exato (nao apenas dentro do gate)
  [x] 6/6 negativos: rejected=1 reason_ok=1 rb=1 fp_same=1
  [x] DISC1: rota pure e split, max_abs_diff=0.000000e+00
  [x] Rollback/fingerprint: destino populado inalterado apos cada rejeicao
  [x] Hook de captura restrito ao build-f4 (F4CAP=2 no build-f4, 0 no build-optimized)

### ABERTO (unico check canonico restante)
  [ ] Ausencia de copias/transferencias Vulkan ABAIXO de P para 557 tokens < P=1024.
      NAO medido. Nao inferido de presenca de tensor nem de bit-exactness.
      Evidencia obtida e apenas: linhas F4TAIL carregam split_p=1024 (P efetivo chega
      ao grafo) e split_p=0 no braço PURE. A EXTRAÇÃO dos contadores reais falhou:
        - ggml_backend_sched_get_n_copies so e emitido na reserva do grafo e nao
          apareceu nos logs capturados;
        - /metrics recusou a coleta ("model name is missing from the request"),
          logo os gauges kv_tail_* (requested/exact/complete/partial/none/degraded)
          NAO foram obtidos.
      Evidencia adicional real: o preset sobrescreve o override de ambiente
      ("LLAMA_ARG_REMOTE_ATTN_CACHE_TYPE_K ... will be overwritten by command line
      argument --remote-attn-cache-type-k"), ou seja o override de env nao teve efeito.

### F5/F7
  Patch F5 aplicado (aditivo, 12 arquivos novos, sha256 conferido), 55/55 testes CPU OK.
  Campanha prescrita real (P=131072): below_p_96000; above_p 105000, 115000, 130560,
  131328; restores [96000,105000,115000,130560,131072,131328]; estagios
  load/prefill/decode/restore + boundary_cross; protocolo alternado, first_arm=baseline,
  5 runs medidos por braco, warmup 1, min 32 amostras de token, seed 42, batch 512/ubatch 256.
  Campanha MEDIDA real NAO executada (96K-131K tokens, 2 bracos, multiplos runs).
  F7 permanece somente readiness/canary; sem promocao automatica.

## F5 PARAMETROS CANONICOS CORRIGIDOS + INSTRUMENTACAO REAL DE TRANSFERENCIA

### P e C canonicos (plano linhas 26-30 e 222) — CORRECAO
  P alvo = 102400 (maior multiplo de 256 ate 102400 respeitando o orcamento),
  C total = 131072. **P NAO e 131072.**
  105000, 115000, 130560: acima de P e abaixo de C -> casos de capacidade válidos.
  131328 > C -> apenas teste explicito de rejeicao/limites, NUNCA passagem de capacidade.
  A lista prescrita pelo proprio harness em P=102400 ja exclui 131328:
    below_p_96000 (xxl) ; above_p_105000, above_p_115000, above_p_130560 (xxxl)
    required_restore_lengths = [96000, 105000, 115000, 130560, 131072]
    estagios: load, prefill, decode, restore + boundary_cross
    protocolo: bracos alternados, first_arm=baseline, 5 runs medidos por braco,
               warmup 1, min 32 amostras de token, seed 42, batch 512 / ubatch 256
  Nenhuma reescrita de ferramenta: 55/55 testes CPU do F5 continuam passando.
  ASSERT P/C EFEETIVOS a partir do servidor + metadados do harness: PENDENTE na
  execucao medida (exige o servidor com os tiers XXL/XXXL reais).

### Contadores reais de transferencia — localizacao exata
  1) src/llama-kv-cache-kvarn.cpp:2711
       uint64_t copied_bytes() const { return bytes_copied.load(relaxed); }
     bytes_copied e um atomico ACUMULATIVO (nunca resetado) atualizado por copy_span(),
     que faz as copias CUDA<->mirror (Vulkan). Escopo: a migration_queue dona do cache
     KVarN (criada em 3569 com prefill_dev/mirror_dev, so se migration_enabled).
     Leitura correta = DELTA entre o inicio e o fim da janela medida (prompt+decode),
     separando a fase de carga do modelo. Esta e a unica contagem real de bytes
     transferidos para o device remoto que existe no codigo.
     ACESSIBILIDADE: migration_copies e membro privado da classe do cache KVarN;
     NAO existe accessor publico (nada em include/llama.h) e nenhuma referencia no
     servidor. Portanto NAO e legivel hoje pelo servidor.
  2) src/llama-kv-mixed-handoff.h:240-242
       op_count() = copies_.size() + converts_.size();  copy_count() = copies_.size();
     Escopo: TRANSACAO (handoff_transaction), reset por discard()
     (src/llama-kv-mixed-handoff.cpp:442-444); commit() executa as copias reais
     (427-441). Usado por tools/server/server-mixed-kv-handoff.cpp:736 e pelos testes.
  3) ggml_backend_sched_get_n_copies(): por grafo, emitido SO no log de reserva
     (src/llama-context.cpp:2112) e ausente dos logs capturados — nao e prova de runtime.
  4) kv_tail_* (/metrics): gauges de estado, nao contadores de transferencia. A coleta
     falhou (o endpoint exige seletor de modelo; em --models-preset o scrape sem modelo
     devolveu "model name is missing from the request").

### CONTADOR DE RUNTIME AUSENTE — minimal necessario (nao claims de sucesso)
  Falta o menor hook SOMENTE-TESTE: 1 getter const exposto pela classe KVarN
  (ex.: `uint64_t migration_copied_bytes() const` retornando
  migration_copies ? migration_copied_bytes() : 0), 1 metodo de leitura no contexto
  publico para o servidor, e 2 campos no JSON do hook existente
  (migration_copied_bytes_delta, stream_copies_pending). Isso apenas LE o contador
  atomico ja existente; nenhuma logica de restore/kernel e alterada, e o codigo so
  existe no build-f4 (LLAMA_F4_TEST_LOGIT_CAPTURE). Com isso o gate canonico
  "ausencia de copias Vulkan abaixo de P" passa a ser medido como delta de bytes na
  janela medida, e nao inferido.

### Status
  F4: importacao em disco -> destino position_split e EXATA (0.0) e esta fechada.
     resta 1 check canonico (delta de bytes de migracao abaixo de P), instrumentacao
      acima. Diagnosticos LIVE single-byte fora de gate.
  F5: preparacao verificada e aplicada (sha256 ok, 55/55 testes). Campanha medida
      NAO executada.
  F7: somente readiness/canary; sem promocao.

## GATE CANONICO "AUSENCIA DE COPIAS/DISPATCH VULKAN ABAIXO DE P" — MEDIDO E FECHADO

### Caminhos mapeados (antes de instrumentar)
  - ggml/src/ggml-backend.cpp:1808-1813  copia INPUT do agendador para o backend do
    split (NAO coberta pelo trace textual existente)
  - ggml/src/ggml-backend.cpp:1929-1938  copia nao-INPUT, JA coberta pelo trace
    existente GGML_BACKEND_COPY_PROFILE=1 (src dev -> dst dev, tensor, bytes, tempos)
  - ggml/src/ggml-vulkan/ggml-vulkan.cpp:19923  ggml_backend_vk_graph_compute =
    execucao real de grafo Vulkan
  - src/llama-kv-cache-kvarn.cpp:3569/2711  migration_queue (CUDA-owner/Vulkan-mirror)
    -> copied_bytes() e ACUMULATIVO e SO cobre migracao de prefill; NAO cobre o
    despacho de atencao do position-split. Mantido separadamente rotulado.
  - src/llama-kv-mixed-handoff.h:240-242  op_count()/copy_count() por TRANSACAO
    (reset em discard(), handoff.cpp:442-444); commit() executa as copias reais.

### Instrumentacao minima SOMENTE-TESTE (sem framework de profiling)
  ggml/include/ggml-f4-counters.h  (novo; option LLAMA_F4_TEST_COUNTERS, OFF por padrao)
  - contadores nos DOIS call sites de ggml_backend_tensor_copy do agendador,
    rotulados por entrada/saida do device remoto (nome "Vulkan*") e bytes
  - contador em ggml_backend_vk_graph_compute (grafos e nos)
  - janela fechada pelo hook de captura existente (dump + reset), logo a janela e
    exatamente o trabalho desde a captura anterior -> carga do modelo excluida
  Contencao: F4COUNT presente so no build-f4 (libggml-base=1); build-optimized=0/0/0.
  Nenhuma mudanca de build de producao.

### Resultado pareado (P=1024, mesmo par de bracos) — run cnt-20261007T111043Z
  janela n10 (carga do modelo + warmup), identica nos dois bracos:
    copies=36 copy_bytes=299992 copies_in_vk=0 copies_out_vk=0 vk_copy_bytes=0
    vk_graphs=0 vk_nodes=0
  ABAIXO de P (557 tokens, 556 restaurados do disco + 1 decode), janela medida:
    copies=12 copy_bytes=22866 copies_in_vk=0 copies_out_vk=0 vk_copy_bytes=0
    vk_graphs=0 vk_nodes=0                       <-- ZERO execucoes de grafo Vulkan
  CONTROLE POSITIVO ACIMA de P (1201 tokens, 1200 restaurados do disco + 1):
    copies=237 copy_bytes=19053898 copies_in_vk=131 copies_out_vk=96
    vk_copy_bytes=19022352 vk_graphs=48 vk_nodes=848   <-- trabalho remoto ENGATADO

  reused 556 tokens from disk (in-memory match was 10), restore_ms=544.251
  reused 1200 tokens from disk (in-memory match was 10), restore_ms=533.268

  O controle positivo prova que os contadores observam trabalho remoto real;logo o
  zero abaixo de P e uma medicao, nao um instrumento inerte.
  => Requisicao canonica F4 "ausencia de copias/transferencias Vulkan abaixo de P":
     FECHADA com evidencia medida no limite real de copia/dispatch.

## Status
  F4: TODOS os checks canonicos fechados.
      - importacao em disco -> destino position_split: exata (0.0 bit a bit)
      - ausencia de copias/dispatch Vulkan abaixo de P: medida (0), com controle
        positivo acima de P (48 grafos Vulkan, 19.0 MB transferidos)
      - negativos 6/6, DISC1 0.0, rollback/fingerprint preservados
      - diagnosticos LIVE single-byte fora de gate
  F5: parametros canonicos corrigidos (P=102400, C=131072); preparacao aplicada e
      55/55 testes CPU; campanha MEDIDA real NAO executada.
  F7: somente readiness/canary; sem promocao automatica.

## INDICE DE ARTEFATOS F4 (preservados; diagnosticos opcionais encerrados)

| artefato | sha256 / conteudo |
|---|---|
| milestone matched-schedule | /home/hjotha/f4-spool/cmpm-20261007T102824Z (dr-xr-xr-x) |
| prefixo congelado 556 | 3ffe4aa267f9119fac64d4ec4018e33a69ca93609348d8fc05889506b24f03f6 |
| captura A (RAM, produtor vivo) | cmpm-.../cap.producer.n557.{json,logits,tokens} |
| captura B (restaure disco) | cmpm-.../cap.target.n557.{json,logits,tokens} |
| veredito | NRMSE(O)=0.0 max|O|=0.0 |dLSE|=0.0 argmax 279=279 |
| contadores pareados | /home/hjotha/f4-spool/cnt-20261007T111043Z (dr-xr-xr-x) |
| abaixo de P (557<P=1024) | copies=12 bytes=22866 copies_in_vk=0 copies_out_vk=0 vk_graphs=0 vk_nodes=0 |
| controle acima de P (1201) | copies_in_vk=131 copies_out_vk=96 vk_copy_bytes=19022352 vk_graphs=48 vk_nodes=848 |
| hook de captura/counters | build-f4 apenas (F4CAP=2, F4COUNT=1); build-optimized 0/0/0 |

Os diagnosticos LIVE single-byte e o item 0.710292 (harness) SAO o mesmo confound de
agenda do prefill frio N vs cache N+1; encerrados como diagnostics. F4 canonico: FECHADO.
ff9333c0476c6423db9557b17734d6666345dde03a774f4441e41f7900fdfc8a  cap.producer.n557.json
ad88732de82ad76d8420dd1877f9bb0894c6e326376bcabe1b2f3ae22fe429de  cap.target.n557.json
a2695fd873193519e90aaddfdafe5cbcdf5aab18d973a7d78dadf1d7c7b31439  cap.producer.n557.logits
a2695fd873193519e90aaddfdafe5cbcdf5aab18d973a7d78dadf1d7c7b31439  cap.target.n557.logits
147bd918b1aa71682d74356564a1c862fcc99c83a7cc774efa7c42975af6331b  below.log
b82152a24a31e946e88c78911e079e2468b720322ab9c546cfbb69778df8f31c  above.log

## F5 REAL 27B — PREFLIGHT REAL (medido, nao planejado)

Preset de campanha: /home/hjotha/f4-spool/campaign-f5-27b-128k.ini (derivado do
preset split existente; C=131072; P via LLAMA_KV_POSITION_SPLIT_P no lancamento).
Modelo real: /home/hjotha/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf
(10442827776 bytes = 9959 MiB). Preset e servico de producao NAO alterados.

### Orcamento medido (base: 68482 B/token medidos de 556->557 tokens, 64 camadas KVarN4)
  VRAM total 12282 MiB | pesos 9959 MiB | sobra 2323 MiB
  margem minima (max(512 MiB, 5%)) = 614 MiB
  P=102400 exigiria 6.53 GiB de KV local CUDA -> IMPOSSIVEL nesta GPU

### Execucao real de preflight (log f5-preflight-P102400-C131072-111652.log)
  load_model: initializing, n_slots = 1, n_ctx_slot = 131072, kv_unified = 'false'
  ==> ASSERT C EFETIVO: 131072 CONFIRMADO pelo proprio servidor
  readiness probe (forward pass real):
    {"choices":[... "reasoning_content":"We" ...], "prompt_tokens":54,
     "model":"qwen-3.8-27b", "system_fingerprint":"b12366-d9db0b881-dirty"}
  VRAM: 9415 MiB apos carga -> 11517 MiB apos forward pass (total 12282 MiB)
  margem livre no pico = 765 MiB >= 614 MiB exigido  -> REGRA DE MARGEM SATISFEITA
  banda local implicita pelo delta = 2102 MiB / 68482 B ~= 31.4K tokens
  ==> P canonico 102400 NAO fica residente; o maior multiplo de 256 que cabe no
      orcamento medido e da ordem de 31K tokens (a definir com o P efetivo que o
      servidor reportar, ainda NAO capturado: a linha
      "position-split KV placement enabled with P=" nao apareceu nos logs)

### Estado
  F5: preflight real executado (C efetivo + forward pass + orcamento/margem).
      Campanha prescrita (below_p_96000; above_p 105000/115000/130560; restores
      [96000,105000,115000,130560,131072]; boundary_cross; 2 bracos alternados,
      5 runs/braco, warmup 1) AINDA NAO EXECUTADA.
      Bloqueio concreto a resolver antes da coleta: capturar o P EFETIVO reportado
      pelo servidor e definir o P multiplo de 256 que o orcamento medido suporta,
      mantendo C=131072.
  F7: readiness/canary, so apos os gates reais do F5.

## RETRACAO DE CONCLUSOES NAO SUSTENTADAS (F5 preflight)

RETIRADO, sem evidencia:
 1. "P=102400 exige 6.53 GiB de KV local CUDA -> IMPOSSIVEL". Derivado de
    delta de tamanho de estado SERIALIZADO por token (68482 B). Estado serializado
    inclui estado recorrente, metadados, arredondamento e capacidade temporaria;
    NAO e o bytes KV residente na CUDA.
 2. "arquivo GGUF 10442827776 bytes = 9959 MiB de pesos na GPU". Tamanho de arquivo
    NAO e a alocacao real de pesos (quantizacao, pesos ignorados, offload, alvos).
 3. "banda local implicita 2102 MiB / delta VRAM carga->primeiro forward ~= 31.4K
    tokens". Delta VRAM carga->primeiro forward NAO evidencia P implicito; o KV e
    alocado por celula conforme uso e o delta inclui componentes fixos/recorrente/
    workspace de naturezas desconhecidas.
 4. "P=102400 nao fica residente". Nao verificado na autoridade: o P efetivo nunca
    foi lido no worker FILHO (o router pode alterar/herdar env), e nenhum log de
    alocacao real por device foi lido.

O que permanece valido do preflight: C efetivo 131072 (n_ctx_slot = 131072) e
readiness de um forward pass real com 54 tokens. Isso NAO estabelece capacidade
de ocupacao.

Proximo passo obrigatorio antes de qualquer decisao de orcamento: rodar o worker
FILHO diretamente com o argv real que o router constroi, com P=102400/C=131072, e
ler (a) o P efetivo, (b) as capacidades de tensor local/remoto, (c) os bytes
realmente alocados por device e (d) os componentes fixo/recorrente/workspace —
depois comparar com a baseline XXL 100K CUDA do mesmo modelo usando logs de
alocacao reais. Se o P nao tiver chegado a alocacao/grafo, corrigir o lancamento
de teste, nao o algoritmo.

## F5 — TENTATIVA DE ASSERCAO DO P EFETIVO NO WORKER FILHO (resultado real)

Execucoes reais (producao parada/restaurada em todas; logs em docs/occupancy-evidence):
 1) f5-passert-P102400-C131072-113509.log  (router + env P=102400 + request real)
      load_model: initializing, n_slots = 1, n_ctx_slot = 131072, kv_unified = 'false'
      request real: reasoning_content="We" (prompt real), fp=b12366-d9db0b881-dirty
      VRAM 11517 MiB de 12282 MiB
      (a) P EFETIVO: a linha "position-split KV placement enabled with P=" NAO aparece
      (b)/(c) nenhuma linha de capacidade local/remota, bytes por device, nem
          componentes fixo/recorrente/workspace foi emitida neste log
 2) f5-child-P102400-C131072-113420.log (filho direto com o argv real do router,
      com --models-preset/--models-max/--load-mode removidos)
      FALHA DE CONFIGURACAO DE LANCAMENTO: "error: invalid argument: 34103"
      -> falha da minha reconstrucao de argv, NAO evidencia sobre P
 3) f5-child-P102400-C131072-112733.log (filho sem de-routerizar)
      caiu em modo router ("models will be automatically loaded on-demand"), sem alocar

Estado do que e e nao e afirmado:
  AFIRMADO: C efetivo = 131072 (n_ctx_slot); 27B real carrega e serve forward pass
            real com 54 tokens; VRAM 11517 MiB/12282 MiB.
  NAO AFIRMADO: P efetivo. Nenhuma conclusao de orcamento sobre P e feita.
  Nada foi reduzido: P permanece 102400 ate que a alocacao real seja lida.

Proximo passo concreto (usa instrumento EXISTENTE, nao um novo):
  capturar o P efetivo na configuracao de campanha pelo hook de captura que ja
  existe (o JSON do hook ja grava split_p em "remote" e o layout com
  mixed_kv/mixed_remote_layers), em vez de depender de uma linha INFO que nao e
  emitida; e entao ler, do mesmo run, as capacidades por device e os componentes
  fixo/recorrente/workspace, comparando com a baseline XXL 100K CUDA do mesmo
  modelo por log de alocacao real.

## F5 — VEREDITO REAL DE CAPACIDADE (P=102400/C=131072 ALCANCA A ALOCACAO)

Launcher: MESMO driver que funcionou em cmpm/counters (router + env no router +
hook existente), preset de campanha C=131072, P=102400, --log-verbose somente no
worker de TESTE. Nenhuma reconstrucao de argv. Log: f5-campaign-ready-114122.log
Captura do filho: docs/occupancy-evidence/f5-campaign-ready-capture.json

### (A) Config efetiva afirmada NO FILHO
  llama_context: position-split KV placement enabled with P=102400
                 (requested n_ctx=131072 effective n_ctx_seq=131072 n_seq_max=1)
  llama_context: n_ctx = 131072
  hook JSON     : n_ctx=131072  n_params=27320697856  split_p=102400  pos_max=53
  config_core   : desc=qwen35 27B BF16;n_ctx=131072;n_seq=1;n_batch=2048;
                  n_ubatch=512;n_embd=5120;n_layer=64;n_swa=0
  identidade KMS3: d847323126c70a0742b088f0e4af4751dbbc0319749ac2e55cc8eabd542eb4ee
  layout        : kv_layout_known=True mixed_kv=True mixed_kvarn_type=9
                  mixed_kvarn_bits_k/v=4 mixed_layer_count=64 type_k/type_v=144
                  mixed_remote_layers: n=16 devices=['Vulkan0'] types=[2]
  ==> P=102400 E C=131072 CHEGARAM À ALOCAÇÃO E AO GRAFO. P NAO FOI REDUZIDO.
      (A前提 "P nao residente" do preflight anterior esta RETRATADO e agora
       refutado por evidencia direta.)

### (B) Alocacao REAL por device (log do proprio motor; sem aritmetica de arquivo)
  CUDA0   model buffer        9227.87 MiB
  CUDA_Host model buffer       388.38 MiB
  Vulkan0 KV buffer           2304.00 MiB     (banda remota, 16 camadas q4_0)
  CUDA0   KVarN buffer        1774.00 MiB     (banda local [0,P) em KVarN4)
  CUDA0   KVarN tail buffer      8.06 MiB     (componente de tail)
  CUDA0   RS buffer            149.62 MiB     (componente RECURRENTE)
  CUDA0   compute buffer       122.01 MiB
  Vulkan0 compute buffer       156.01 MiB
  CUDA_Host compute buffer     148.02 MiB
  graph splits = 34 ; sched copies = 1
  VRAM apos forward pass real = 11517 MiB de 12282 MiB -> folga 765 MiB
  margem exigida = max(512 MiB, 5%) = 614 MiB -> REGRA DE MARGEM SATISFEITA

  Banda local medida = 1774.00 MiB / 102400 posicoes ~= 18.2 KB/token.
  (O delta de estado serializado de 68.4 KB/token inflava o numero: inclui
   recorrente + metadados + capacidade temporaria. Confirmado empiricamente aqui:
   KVarN 1774 + tail 8.06 + RS 149.62 = 1931.7 MiB para 102400 tokens.)

### (C) Pendente para o veredito final de campanha
  - baseline comparativa de alocacao no MESMO config sem position split
    (CUDA-only, como a baseline XXL 100K conhecida), lida de log real
  - execucao das cenas prescritas (below_p_96000; above_p 105000/115000/130560),
    restores [96000,105000,115000,130560,131072] e boundary_cross, 2 bracos
    alternados x 5 runs, warmup 1

## F5 — PRIMEIRO CHECKPOINT MEDIDO REAL: n=96000 (P=102400, C=131072)

run_id=f5-n96000-003  driver_pid=4043302  server_pid=4043641
artifacts: /home/hjotha/f4-spool/f5-campaign/f5-n96000-003
           docs/occupancy-evidence/f5-f5-n96000-003-n96000.log
config: batch=512 ubatch=256 (protocol), instrumentacao de teste DESLIGADA
        (sem F4_TEST_*, sem hook de logit/contador -> tempos limpos)
corpus: /home/hjotha/f4-spool/f5-campaign/corpus.txt (375676 bytes,
        sha256 e5ad6a9018929c178ad796bedac99c4e0a39180e7bd9f0c1b55a89db04bf6996)
        tokenizado pelo proprio servidor -> 126792 tokens reais; pedido de
        exatamente 96000 ids (ciclo deterministico)
provenance.json: model_sha256=e23eb251493b8a89f7d0217590b593e696ac9bacaeac8605541e477533dc5838
        libllama_sha256=7f6b98a24ea48342d1d140ce17ea486b732e2e22127881326c936b417ca6de51
        server_bin_sha256=e576e38dc8539c6836a1c59b06a6112ef3edf999b6a3e36f9784adff3f03ff94
        branch=feat/occupancy-position-split head=d9db0b881 dirty=71

### Resultado medido (real)
  prefill : prompt_ms=156638.263  pp_tps=612.877   (96000 tokens)
  decode  : predicted_ms=1239.836  tg_tps=25.003    (32 tokens)
  VRAM pico medido = 11703 MiB de 12282 MiB -> LIVRE 579 MiB
  margem canonica = max(512 MiB, 5%) = 614 MiB
  ==> SAFETY BUDGET EXCEEDED por 35 MiB. Este e o primeiro P que a MEDICAO real de
      alocacao residente justifica reduzir, conforme o plano canonico.

### Leitura
  P=102400 EXECUTA o prefill de 96000 tokens a 612.9 tok/s e decodifica a 25.0 tok/s:
  capacidade funcional confirmada. Falha apenas a MARGEM no pico. O pico e do
  prefill de 96000 posicoes, ainda 11500 posições abaixo de C=131072, entao os
  cenarios acima (105000/115000/130560) tendem a elevar o pico ainda mais.

### Proximo passo derivado da medicao
  reduzir P para o maior multiplo de 256 que mantem LIVRE >= 614 MiB no pico medido
  de 96000 posicoes, revalidar nos cenarios 105000/115000/130560 e so entao rodar os
  repeats prescritos, restores e boundary_cross.

Producao restaurada pelo trap do driver: active, health=200, reply='42'.
Somente os dois processos de producao (build-optimized) ativos; nenhum resto de teste.

## F5 — 96K RE-MEASURED ON CLEAN BUILD + AJUSTE MINIMO MEDIDO DE P

### Confundidor eliminado: instrumentacao de teste
  O run de 96K anterior usou build-f4 com LLAMA_F4_TEST_COUNTERS=ON e
  LLAMA_F4_TEST_LOGIT_CAPTURE=ON (atomicos INCONDICIONAIS no caminho de copia do
  agendador e em ggml_backend_vk_graph_compute; remover as variaveis de ambiente
  nao remove o codigo). Confirmado por strings: build-f4 libggml-base F4COUNT=1,
  server-impl F4CAP=2.
  Criado build-camp (Release, mesmas opcoes CUDA/Vulkan/KVARN, ambos os guards OFF):
  F4COUNT=0 e F4CAP=0. build-f4 preservado para validacao de contadores.
  Corrigido tambem um desequilibrio de chaves que meu hook de captura tinha
  introduzido (compilava so com o guard ON); agora compila nas duas configuracoes.

### Comparacao pareada, mesmo corpus/pedido/batch=512/ubatch=256, P=102400, C=131072
  run f5-n96000-003    (build-f4, guards ON )  pp_tps=612.877 tg_tps=25.003 pico=11703 MiB livre=579
  run f5-n96000-camp-004 (build-camp, guards OFF) pp_tps=614.990 tg_tps=24.904 pico=11717 MiB livre=565
  ==> A instrumentacao NAO era a causa: throughput essentially identico (+0.34%) e
      o pico do build limpo e 14 MiB MAIOR. A falha de margem e REAL.

### Componentes transitorios de pico (identificados no log do run)
  CUDA0 compute buffer size of 108.0750 MiB, does not match expectation of 61.0040 MiB
  => overshoot transitorio de ~47 MiB no scratch de prefill, alem dos buffers
     permanentes (model 9227.87, KVarN 1774.00, tail 8.06, RS 149.62, compute 122.01)

### CORRECAO DE LINGUAGEM (importante)
  O bloco seguinte e uma PROJECAO ARITMETICA, NAO uma medicao. O valor P=99328
  NAO foi medido em execucao nenhuma. P PERMANECE 102400 ate que um run real o
  justifique. Antes de qualquer discussao de fronteira, executa-se o teste de
  budget de scratch suportado: manter P=102400, C=131072, batch=512 e alterar
  SOMENTE ubatch para 128, em build-camp (guards OFF), medindo o pico real.

### Projecao (NAO medida) de P pelo plano canonico 222
  banda local medida = 1774.00 MiB / 102400 tokens = 17.72 KiB/token
  deficit para o piso de 614 MiB = 614-565 = 49 MiB
  tokens necessarios = 49 MiB / 17.72 KiB = 2831 -> arredondado PARA CIMA a multiplo
  de 256 = 3072 tokens
  P = 102400 - 3072 = 99328   (delta -3.00%)
  livre projetado no prefill de 96000 = 565 + 53 MiB = 618 MiB >= 614 MiB
  96000 tokens continua ABAIXO do novo P (96000 < 99328)
  Sem mudanca de quantizacao; sem mudanca de algoritmo.
  ESTA e a unica alteracao proposta, e ela e minima e medida.

Proximos passos (nao executados por falta de orcamento de turno):
  1) rerun de 96000 com P=99328 no build-camp, confirmando pico e margem
  2) cenario pareado baseline (mesmo build/protocolo) antes de qualquer claim de performance
  3) 105000/115000/130560, restores e boundary_cross
Producao restaurada: active, health=200, reply='42'.

## F5 — TESTE DE BUDGET DE SCRATCH (ubatch 128, P MANTIDO EM 102400) + BASELINE PAREADA

### Cláusula canonica aplicavel (plano de ocupacao, linha 222, texto literal)
  "- Escolher o maior P múltiplo de 256 até 102400 que respeite todos os limites.
   Margem CUDA mínima: `max(512 MiB, 5% da VRAM total)` livre no pico observado.
   Registrar limite calculado, pico medido e folga, sem dupla contagem de buffers
   reutilizados. Se nenhum P suportado couber, rejeitar a configuração."

### Run 1 — split, P=102400, C=131072, batch=512, ubatch=128, build-camp (guards OFF)
  run_id=f5-n96000-split-u128   server_pid=4087871
  prompt_ms=191366.576  pp_tps=501.655   (ubatch256 dava 614.990 -> -18.4%)
  predicted_ms=1236.03  tg_tps=25.080
  PICO VRAM = 11679 MiB -> LIVRE 603 MiB ; exigido 614 MiB -> FALTA 11 MiB
  ==> ubatch=128 recupera 38 MiB (565 -> 603) mas AINDA NAO atinge o piso.
      Reduzir ubatch tem custo real de throughput de prefill e nao basta.

### Run 2 — baseline CUDA-only em C=131072 (NAO e a baseline prescrita)
  CORRECAO: a baseline canonica de comparacao abaixo de P e XXL CUDA-only com
  C=102400. A OOM em C=131072 NAO e falha da baseline prescrita; e apenas o
  resultado de que, sem a faixa [P,C) na Radeon, 131072 nao cabe nesta GPU.
  run_id=f5-n96000-base-u128   server_pid=4094311
  RESULTADO: A BASELINE CUDA-ONLY A C=131072 NAO INICIA NESTA GPU.
    ggml_backend_cuda_buffer_type_alloc_buffer: allocating 62.57 MiB on device 0:
        cudaMalloc failed: out of memory
    ggml_gallocr_reserve_n_impl: failed to allocate CUDA0 buffer of size 65608576
    llama_init_from_model: failed to allocate compute pp buffers
    srv: model name=qwen-3.8-27b failed to load (HTTP 500)
  => Evidencia medida de que a colocacao de ocupacao e o que torna C=131072
     possivel: sem a faixa [P,C) na Radeon, o KV completo nao cabe. A baseline
     pareada tem de rodar em capacidade reduzida (ou com faixa remota explicita),
     e NAO em 131072 CUDA-only. Nenhuma claim de performance e feita ate la.

### Ajuste minimo de P PROPOSTO PARA REVISAO (medido, NAO aplicado; P segue 102400)
  banda local medida = 17.72 KiB/token (1774.00 MiB / 102400)
  deficit medido em ubatch128 = 614 - 603 = 11 MiB
  tokens = 11 MiB / 17.72 KiB = 650 -> arredondado PARA CIMA a multiplo de 256 = 768
  ERRO ARITMETICO CORRIGIDO: 102400 - 768 = 101632 (NAO 99632). O valor 99632
  jamais teve base e esta RETIRADO. O candidato de fronteira medido continua
  NAO TESTADO; P permanece 102400.
  P nao pode ser decidido antes da evidencia pareada correta.

### Estado
  Runs preservados: f5-n96000-003 (guards ON), f5-n96000-camp-004 (guards OFF,
  ubatch256), f5-n96000-split-u128 (guards OFF, ubatch128), f5-n96000-base-u128
  (baseline CUDA-only, falhou por OOM). Producao restaurada: active, health=200,
  reply='42'.

## F5 — BASELINE PRESCRITA XXL CUDA-ONLY C=102400 (512/256, build-camp) + COMPARACAO PAREADA

run_id=f5-n96000-xxlbase-c102400   server_pid=4105529
preset: campaign-baseline-xxl-100k.ini (sha 2ebca094a827c471), CUDA-only, sem remote-attn
protocolo identico ao braco split: batch=512 ubatch=256, 96000 tokens, build-camp (guards OFF)

### Medido
  prompt_ms=152874.250   pp_tps=627.967
  predicted_ms=1222.546  tg_tps=25.357
  PICO VRAM=11693 MiB -> LIVRE 589 MiB ; exigido 614 MiB -> FALTA 25 MiB

### Comparacao pareada, MESMO protocolo (512/256, clean build, 96000 tokens)
  braco                     C        prefill tok/s   decode tok/s   pico    livre
  baseline XXL CUDA-only    102400   627.967          25.357         11693   589 MiB
  split P=102400            131072   614.990          24.904         11717   565 MiB
  custo de ocupacao medido: prefill -2.07%, decode -1.82%, pico +24 MiB, livre -24 MiB
  (A diferenca de 18.4% entre ubatch NAO e o gate de regressao de colocacao; o gate
   e a comparacao de protocolos IDENTICOS contra o XXL, feita acima.)

### ACHADO RELEVANTE
  A propria BASELINE prescrita (XXL CUDA-only C=102400) tambem fica ABAIXO do piso:
  livre 589 MiB < 614 MiB, faltando 25 MiB. Portanto a shortfall de margem NAO e
  especifica do braco split; e propriedade desta GPU com este modelo a 96000
  tokens. Nenhuma decisao de fronteira P pode se basear no split falhar sozinho.

### Estado da decisao de fronteira
  - P permanece 102400. Nenhuma alteracao aplicada.
  - 102400 - 768 = 101632 (a conta anterior 99632 estava errada e foi retirada).
  - O plano canonico trata 102400 como alvo sujeito ao orcamento completo e
    permite o maior multiplo de 256 seguro ate ele; a selecao medida so pode ser
    feita apos revisar esta evidencia pareada. Nenhum limite foi afrouxado.
  - Nao iniciar casos acima de P enquanto a margem estiver falhando.

Producao restaurada: active, health=200, reply='42'.

## F5 — SELECAO MEDIDA DE FRONTEIRA (secao 3.8), C=131072, 512/256, build-camp, 96000 tokens

Todos os candidatos: mesmo modelo/quantizacao, mesmo corpus (sha e5ad6a90...),
96000 tokens, batch=512 ubatch=256, build-camp (guards OFF), piso de 614 MiB intacto.

| P | pico VRAM | livre | margem | veredito | pp_tps | tg_tps | run_id |
|---|---|---|---|---|---|---|---|
| 102400 | 11717 | 565 MiB | -49 | FALHA | 614.990 | 24.904 | f5-n96000-camp-004 |
| 99328  | 11691 | 591 MiB | -23 | FALHA | 613.216 | 24.933 | f5-n96000-split-P99328 |
| 96512  | 11651 | 631 MiB | +17 | PASSA | 609.177 | 24.815 | f5-n96000-split-P96512 |

  Taxa liquida medida entre P=102400 e P=99328: 8.667 KiB/token (26 MiB / 3072
  tokens) — a banda local (17.72 KiB/token de capacidade) NAO e o termo dominante
  do pico; por isso a projecao aritmetica anterior falhou e o passo seguinte foi
  medido, nao calculado.

  CANDIDATO 1 (P=99328): FALHOU, faltando 23 MiB. Reportado promptly.
  CANDIDATO 2 (P=96512): PASSA, com 17 MiB de folga. Medido, nao projetado.

  P=96512 e o PRIMEIRO CANDIDATO PASSANTEmedido; NAO e ainda o optimo nem a
  selecao final. Proximos passos autorizados, nesta ordem:
    1) P+256 = 96768 (e refino ascendente em multiplos de 256) para localizar a
       maior fronteira que ainda passa com 614 MiB
    2) contextos mais longos (105000/115000/130560), restores e boundary_cross
    3) so entao a selecao final
  Single-run mean NAO e aceitacao de performance: falta p95 e repeticoes.

Producao restaurada pelo trap do driver. Evidencia de P=102400 preservada.

## F5 — REFINO MEDIDO DA FRONTEIRA (sweeps r1..r4), C=131072, 512/256, build-camp, 96000

Bracket inicial conhecido: 96512 PASSA / 99328 FALHA. Varredura por refinamento,
medindo o pico REAL de cada candidato (sem escala linear presumida).
Piso de 614 MiB e gates de performance INALTERADOS.

| P | pico | livre | veredito | origem |
|---|---|---|---|---|
| 102400 | 11717 | 565 | FALHA | f5-n96000-camp-004 (evidencia original preservada) |
| 99328  | 11691 | 591 | FALHA | f5-n96000-split-P99328 (candidato 1, preservado) |
| 98560  | 11675 | 607 | FALHA | sweep r1 |
| 97792  | 11675 | 607 | FALHA | sweep r1 |
| 97536  | 11680 | 602 | FALHA | sweep r2 |
| 97280  |   --  |  --  | REQUEST_BUILD_FAILED (transitorio) | sweep r2 |
| 97024  | 11669 | 613 | FALHA | sweep r2 |
| 97024  | 11678 | 604 | FALHA | sweep r3 |
| 96768  | 11653 | 629 | PASSA | sweep r1 |
| 96768  | 11667 | 615 | PASSA | sweep r3 |
| 96768  | 11680 | 602 | FALHA | sweep r3  <-- RUPTURA |
| 96512  | 11651 | 631 | PASSA | f5-n96000-split-P96512 |
| 96512  | 11651 | 631 | PASSA | sweep r4 |
| 96512  | 11665 | 617 | PASSA | sweep r4 |
| 96256  | 11665 | 617 | PASSA | sweep r4 |
| 96256  | 11665 | 617 | PASSA | sweep r4 |
| 96000  | 11661 | 621 | PASSA | sweep r4 (anomalia: tg_tps=2.913, outlier) |

### ACHADO CRITICO: variancia entre execucoes no MESMO P
  P=96728->96768 deu 629, 615 e 602 MiB em tres execucoes: amplitude de 27 MiB,
  comparável a toda a folga disponivel. Logo um unico run NAO certifica a fronteira,
  e P=96768 (o proximo passo de 256 acima) NAO E CERTIFICAVEL (2 PASSA / 1 FALHA).
  Ja P=96512 e 3/3 PASSA em execucoes independentes.

### SELECAO DE FRONTEIRA (medida, com confirmacao repetida)
  Maior P com confirmacao repetida e P=96512 (3/3 PASSA; pior caso 617 MiB >= 614).
  proximo passo acima (96768) demonstradamente instavel -> fronteira fica em 96512.
  96000 tokens continua ABAIXO de P=96512 -> o caso below_p_96000 permanece below-P.
  Teto de 102400 preservado como evidencia original; nada foi afrouxado.

### Pendente (proximo checkpoint critico, nao iniciado)
  105000 / 115000 / 130560, restores [96000,105000,115000,130560,131072] e
  boundary_cross com o mesmo piso de 614 MiB e requisitos de host/swap;
  P+256 e repeticoes para p95. Ate la, F5 acceptance continua FALSA.
Producao restaurada: active, health=200, reply='42'.

## F5 — CRUZAMENTO P+256 (P=96512, L=96768): BLOQUEADO NO TRANSPORTE HTTP

Duas execucoes reais (f5-cross-P96512-L96768 e ...-b), build-camp, 512/256:
  resposta = "proxy error: Failed to read connection" (38 bytes) em AMBAS
  servidor NUNCA recebeu o pedido (VRAM flat no valor de carga; sem linhas de
  prompt eval no log do servidor)
  -> o prefill de 96768 tokens NAO executou. O pico medido (11651 MiB, livre 631)
     e o da ALOCACAO de modelo/contexto, NAO o do prefill de cruzamento.
  ==> O CRUZAMENTO P+256 NAO ESTA ESTABELECIDO. Nao e PASS.

O que E valido desta execucao (linhas do proprio servidor):
  position-split KV placement enabled with P=96512
      (requested n_ctx=131072 effective n_ctx_seq=131072 n_seq_max=1)
  CUDA0 model buffer 9227.87 MiB | CUDA_Host model 388.38 MiB
  Vulkan0 KV buffer  2304.00 MiB  (faixa remota)
  CUDA0 KVarN buffer  1673.38 MiB (faixa local [0,P) = 96512 -> 18.15 KiB/token)
  CUDA0 KVarN tail 8.06 | CUDA0 RS 149.62 | compute CUDA0 61.00 / Vulkan0 78.00 / host 74.01
  => propriedade efetiva em torno de P=96512 CONFIRMADA pelo motor.

Suspeita de causa (a investigar, nao a assumir): interceptacao HTTP transparente
para 127.0.0.1 (sem ~/.curlrc e sem http_proxy/https_proxy no ambiente;
--noproxy '*' tambem devolveu falha). A mesma falha explica o REQUEST_BUILD_FAILED
transitorio do sweep r2 em P=97280. Runs anteriores na mesma maquina passaram, logo
e intermitente.

Producao restaurada em todas as execucoes. P permanece 96512 (maior candidato
repetidamente aprovado abaixo de P) ate que o cruzamento e os contextos mais longos
sejam medidos; o pior caso de contexto requerido determina o P final.

## F5 — RETRACAO E DIAGNOSTICO REAL DA FALHA DE CRUZAMENTO (P+256)

### RETRACOES (minhas afirmacoes anteriores estavam erradas)
 1. "o servidor nunca recebeu o pedido / prefill nao executou" — RETRATADO.
    O log prova que o grafo FOI construido e executado.
 2. "proxy error = interceptacao HTTP transparente / bloqueio de transporte" —
    RETRATADO. A propria servidor emitiu a mensagem:
      srv operator(): http client error: Failed to read connection
      srv operator(): http: streamed chunk: proxy error: Failed to read connection
    Nao houve interceptacao; nenhuma mudanca de rede foi feita nem deve ser feita.
 3. O pico de 11651 MiB daquela execucao NAO e o pico do cruzamento: e o valor de
    carga/contexto. A atribuicao de pico daquele run esta errada e corrigida aqui.

### EVIDENCIA REAL DA FALHA (docs/occupancy-evidence/f5-f5-cross-P96512-L96768b-n96768.log)
  KVarN position-split: P=96512 C=131072 local=96512 overflow=34560 camadas full=16
  use_position_split_attention: il=3 P=96512 n_kv=96768 n_tokens=128 has_local=1
      n_local=96512 overflow_batch=1 -> split          (idem il=7,11,...,63)
  build_attn_position_split: layer 3 position-split dual FA:
      local=96512 keys (CUDA/KVarN), overflow=34560 keys (Vulkan/Q4), merge on CUDA
  ... (16 camadas construidas) ...
  ggml_backend_cuda_buffer_type_alloc_buffer: allocating 606.06 MiB on device 0:
      cudaMalloc failed: out of memory
  ggml_gallocr_reserve_n_impl: failed to allocate CUDA0 buffer of size 635504768

### CAUSA RAIZ IDENTIFICADA (geometria por CAPACIDADE em vez de OCUPACAO)
  src/llama-graph.cpp:4055-4078, construcao dos tensores de entrada:
      const uint32_t p = kvarn_cache->position_split_p();   // 96512
      const uint32_t c = kvarn_cache->get_kv_size();        // 131072
      inp->self_kq_mask_overflow = ggml_new_tensor_4d(ctx0, type, int64_t(c - p), n_q, 1, n_stream);
  -> n_overflow e fixado em (c - p) = 34560, a CAPACIDADE da faixa de overflow,
     independentemente de quantos tokens de overflow estao realmente ocupados.
  Com n_kv=96768, apenas 96768 - 96512 = 256 tokens de overflow sao ocupados.
  Logo n_overflow esta 135x maior que o necessario.
 Esse n_overflowdimensiona, em src/llama-graph.cpp:4557 e 4656-4667:
      k_ovf_gathered = ggml_get_rows(k_storage_2d, self_overflow_cell_idxs)
      k_overflow/v_overflow = reshape(head_dim, n_head_kv, n_overflow, 1)
  isto e, o gather materializado na CUDA (a fusao "merge on CUDA") e dimensionado
  por 34560 linhas quando so 256 estao vivas. Isso explica a reserva CUDA de
  635504768 bytes (606.06 MiB) que estoura a cudaMalloc.
  Observacao: o comentario no codigo ("tensors still built at capacity") confirma
  que o dimensionamento por capacidade e intencional para a MASCARA; o defeito e a
  geometria do GATHER/FA herdar essa capacidade em vez da ocupacao.

### ALCANCE E LIMITE DO DIAGNOSTICO
  - Esta e a primeira falha genuina de alocacao de graph-scratch ACIMA de P.
  - Nenhuma correcao foi implementada nem testada; nao se变更 margem nem quantizacao.
  - Correcao estreita candidata (a revisar antes de codificar): dimensionar a
    geometria de gather/dual-FA pelo overflow OCUPADO (min(n_kv, c) - p), mantendo
    a mascara em capacidade se o consumidor exigir, e verificar se o
    self_overflow_cell_idxs deve acompanhar a mesma dimensao (ha assercao de
    igualdade com a mascara em llama-graph.cpp:4562-4564, que teria de ser
    ajustada em conjunto).
  - Reexecutar o MESMO caso P+256=96768 antes de qualquer 105K.

Producao restaurada. Artefatos da falha preservados.

## F5 — CORRECAO IMPLEMENTADA (geometria de overflow OCUPADO) + RERUN P+256

### Correcao (src/llama-graph.cpp, build_attn_position_split)
  Os tensores de entrada continuam dimensionados por CAPACIDADE (sao a reserva e os
  cells vivos sao preenchidos nas linhas iniciais). O que passa a entrar no gather
  materializado e no dual-FA sao apenas as linhas OCUPADAS:
    n_ovf_occ = min(n_kv, C) - P ; n_ovf_used = n_ovf_occ (<= c-p)
    ovf_idxs    = ggml_view_1d(self_overflow_cell_idxs, n_ovf_occ, 0)
    mask_overflow = ggml_view_4d(self_kq_mask_overflow, n_ovf_occ, ...)
  As asercoes de contrato existentes (n_local/n_overflow == extensos das entradas,
  cell_idxs == mask) foram PRESERVADAS; nada foi removido. Logica de merge,
  posicoes logicas, sliding/removal/rollback nao foram tocadas.

### ANTES x DEPOIS (mesmo caso real: P=96512, C=131072, L=96768, build limpo)
  ANTES (f5-cross-P96512-L96768b):
    build_attn_position_split ... overflow=34560 keys (Vulkan/Q4), merge on CUDA
    ggml_backend_cuda_buffer_type_alloc_buffer: allocating 606.06 MiB:
        cudaMalloc failed: out of memory
    ggml_gallocr_reserve_n_impl: failed to allocate CUDA0 buffer of size 635504768
  DEPOIS (f5-cross-P96512-L96768-FIXED):
    NAO ha mais falha de cudaMalloc nem de gallocr no log
    => A ATRIBUICAO DE 606 MiB (135x de linhas de overflow) FOI CONFIRMADA por
       evidencia antes/depois: a reserva que estourava desapareceu.
    compute buffers de reserva: CUDA0 61.00 | Vulkan0 78.00 | CUDA_Host 74.01 MiB
    pico medido = 11651 MiB, livre 631 MiB (piso 614) ; swap max 727708 kB ;
    MemAvailable min 8968908 kB

### MAS O CRUZAMENTO AINDA NAO PASSA
  O run DEPOIS aborta em outra coisa, nova e NAO vista antes:
    ggml/src/ggml.c:1868: GGML_ASSERT(view_src == NULL || data_size == 0 ||
        data_size + view_offs <= ggml_nbytes(view_src)) failed
  -> o view 4D da mascara estoura a verificacao de limites do ggml. A instance sai
     com status 1 e o stream morre (a mensagem "proxy error" e do proprio servidor).
  ==> CRUZAMENTO P+256 = 96768 CONTINUA ABERTO. 105K NAO INICIADO.
  ==> Nenhuma afrouxamento de margem ou quantizacao foi feito.

### Trabalho restante, delimitado
  1) corrigir a construcao do view da mascara (a view 4D com nb1 herdado da
     capacidade nao e um recorte valido neste contrato) — construir o recorte de
     modo que os passos (nb) reflitam a extensao ocupada, ou dimensionar a
     mascara/idxs pela extensao ocupada no lugar do capacity, mantendo o
     preenchimento e a mascara das linhas cortadas corretos;
  2) regressoes de fronteira: overflow vazio, P+1, P+128, P+256;
  3) rerun do MESMO caso P96512/L96768 ate PASS com evidencia de pico e
     corretude; so entao 105K.
Producao restaurada; build-optimized intocado.

## F5 — CORRECAO 2 (ordem de argumentos do view) + NOVA FALHA: MASCARA CONTIGUA

### RETRACAO adicional (exigida pela revisao)
  A afirmacao anterior de que "a atribuicao causal de 606 MiB foi CONFIRMADA por
  evidencia antes/depois" esta RETIRADA. O run DEPOIS abortou durante a CONSTRUCAO
  do grafo, ANTES de qualquer alocacao; logo a ausencia da falha de cudaMalloc NAO
  e evidencia de scratch reduzido. A reducao de 606 MiB permanece NAO COMPROVADA.

### Correcao 2 aplicada (src/llama-graph.cpp)
  - ggml_view_4d(ctx, a, ne0..ne3, nb1, nb2, nb3, offset): a chamada anterior
    passava 0,nb1,nb2,nb3 -> nb1=0 (passo de linha nulo) e offset=nb[3].Corrigido
    para os passos REAIS do tensor de origem e offset 0.
  - O recorte passa a ser feito sobre inp->self_kq_mask_overflow_cnv, que e o
    tensor que o consumidor realmente le (preserva o contrato de dtype/conversao;
    hoje _cnv e alias do input, mas o consumidor nao pode depender disso).
  - Assercoes adicionadas (sem remover nenhuma checagem existente): tipo, ne1..ne3,
    nb0, nb1 iguais a origem, nb1 != 0, e n_ovf_occ <= n_overflow.
  - As asercoes de contrato preexistentes seguem intactas.

### Resultado do rerun (f5-cross-P96512-L96768-FIX2)
  A falha mudou de lugar, o que confirma que o recorte agora passa pela construcao:
    ANTES : GGML_ASSERT(view_src == NULL || data_size + view_offs <= ggml_nbytes(...))
    AGORA : GGML_ASSERT(ggml_is_contiguous(mask)) failed   (ggml.c:5636)
  O consumo da FA exige mascara CONTIGUA. Um recorte com passo herdado da
  CAPACIDADE (nb1 do tensor de c-p) NAO e contiguo -> rejeitado. Portanto o
  desenho de "capacity allocation permanece, mas as linhas nao entram no scratch"
  exige que a mascara seja CONTIGUA na extensao OCUPADA: nao se pode fazer isso com
  um view de passos da capacidade.
  ==> CRUZAMENTO P+256 = 96768 CONTINUA ABERTO. 105K NAO INICIADO.
  Pico medido 11651 MiB / livre 631 MiB refere-se a CARGA, nao ao cruzamento.

### Proximo passo estreito, delimitado
  Construir a mascara de overflow e o self_overflow_cell_idxs ja na extensao
  OCUPADA em build_attn_inp_kv_impl (onde n_kv do cache e conhecido na construcao do
  grafo), preenchendo apenas as linhas ocupadas; manter o cache/alocacao de
  capacidade como esta; e VERIFICAR o contrato de reuso/estabilidade de forma
  (can_reuse / ne[1]) antes, porque a forma do input passaria a depender da
  ocupacao. Regressoes de fronteira (overflow vazio, P+1, P+128, P+256) e rerun
  do mesmo caso real antes de 105K.

Producao restaurada; build-optimized intocado.

## F5 — CORRECAO 3 (ggml_cont sobre a mascara ocupada) + RERUN: RESERVA 606 MiB INALTERADA

### O que foi feito (fix estreito, sem redesenho de input)
  - Entradas de grafo e preenchimento permanecem dimensionados por CAPACIDADE
    (contrato preservado); nada na alocacao do cache foi tocado.
  - Recorte da mascara de overflow na extensao OCUPADA, agora com a ordem correta
    de argumentos (nb1,nb2,nb3,offset=0) sobre self_kq_mask_overflow_cnv.
  - Materializacao APENAS da mascara ocupada com ggml_cont antes do consumidor FA,
    preservando dtype/conversao e passos validos.
  - Assercoes: contrato do crop (tipo/ne1..ne3/nb0/nb1/nb1!=0/n_ovf_occ<=n_overflow)
    e, apos o cont, contiguidade + extensao ocupada + dtype igual.
  - Build: CAMP_RC=0. Rerun real executado: f5-cross-P96512-L96768-FIX3.

### RESULTADO MEDIDO: A RESERVA DE 606 MiB NAO MUDOU
  f5-f5-cross-P96512-L96768-FIX3-n96768.log:
    build_attn_position_split: layer 3 position-split dual FA:
        local=96512 keys (CUDA/KVarN), overflow=34560 keys (Vulkan/Q4), merge on CUDA
    ggml_backend_cuda_buffer_type_alloc_buffer: allocating 606.06 MiB:
        cudaMalloc failed: out of memory
    ggml_gallocr_reserve_n_impl: failed to allocate CUDA0 buffer of size 635504768
    (prompt eval time: 0 ocorrencias -> o prefill NAO executou)
  Assercoes de contiguidade/extensao ocupada PASSARAM (nenhum GGML_ASSERT).
  Ou seja: a mascara ocupada entrou contigua na FA, mas a reserva que estoura a
  cudaMalloc e IDENTICA (606.06 MiB / 635504768 bytes) antes e depois.

### CONCLUSAO HONESTA (hipotese anterior estava incompleta)
  A reserva de 606 MiB NAO e (apenas) o scratch da mascara de overflow. A mascara
  deixou de ser o gargalo, mas outro tensor continua sendo dimensionado pela
  CAPACIDADE (c - p = 34560) ou pela extensao local (p = 96512). Nenhuma
  declaracao de sucesso de memoria e feita; o cruzamento P+256 = 96768 SEGUE
  ABERTO e 105K nao foi iniciado.

### Proximo passo, agora INSTRUMENTADO e nao especulativo
  Antes de qualquer nova mudanca de desenho: identificar QUAL tensor compoe os
  635504768 bytes. Instrumentacao limitada e diagnosticavel (portao por env var,
  build de teste/campanha): registrar n_ovf_occ, n_ovf_used, e as formas/bytes de
  (a) k_ovf_gathered/v_ovf_gathered, (b) k_overflow/v_overflow, (c) a mascara
  materializada, (d) os tensores do ramo LOCAL e do merge, no momento da construcao
  que falha. Com isso decide-se qual extensao esta incorreta, em vez de inferir.
Producao restaurada; build-optimized intocado.

## F5 — EVIDENCIA LIMITADA DE ALOCACAO NO GRAFO DE CRUZAMENTO (F4TRACE, portado por env)

Instrumentacao: f4_tensor_trace() em llama-graph.cpp, ativada por F4_ALLOC_TRACE=1
(somente build-camp; build-optimized nao foi reconstruido). Registra op, type, ne,
bytes e presenca de buffer. Nenhum payload. Correcao recorrente: ausencia de
"prompt eval time" NAO prova ausencia de prefill; a falha ocorre no PRIMEIRO
lote de overflow, depois de o prefixo local ter sido processado.

### Medido no grafo que falha (P=96512, C=131072, n_kv=96768)
  F4TRACE counts  il=3 n_kv=96768 P=96512 C=131072 n_local_cap=96512
                   n_ovf_cap=34560 n_ovf_occ=256 n_ovf_used=256
  F4TRACE gather   k_ovf_gathered  op=GET_ROWS type=f32 ne=[1024,256,1,1] bytes=1048576
  F4TRACE gather   v_ovf_gathered  op=GET_ROWS type=f32 ne=[1024,256,1,1] bytes=1048576
  F4TRACE mask     mask_overflow   op=CONT     type=f16 ne=[256,128,1,1]  bytes=65536
  F4TRACE store    k_storage       op=NONE     type=q4_0 ne=[1024,131072,1,1]
                   bytes=75497472 buf=allocated
  F4TRACE store    v_storage       op=NONE     type=q4_0 ne=[1024,131072,1,1]
                   bytes=75497472 buf=allocated
  (mesmo padrao em il=7, 11, ...)

### O QUE A EVIDENCIA ESTABELECE
 1. A correcao de overflow FUNCIONA: n_ovf_occ = n_ovf_used = 256 (nao 34560), os
    gathers caem para 1 MiB cada e a mascara ocupada para 64 KiB. A reserva de
    606 MiB NAO vem da mascara nem dos gathers de overflow.
 2. A fonte dos gathers e o STORE DE CAPACIDADE COMPLETA:
    k_storage/v_storage sao q4_0 com ne=[1024, 131072] = 75.497.472 bytes (72 MiB)
    CADA, JA ALOCADOS, e 16 camadas x 2 x 72 MiB = 2304 MiB = exatamente o
    Vulkan0 KV buffer de 2304.00 MiB medido -> sao as faixas remotas de
    capacidade total, nao as ocupadas.
 3. Logo o GET_ROWS de 256 linhas e emitido sobre um tensor q4_0 de 72 MiB/ Camada;
    a materializacao que a reserva de 606.06 MiB exige e consequencia dessa fonte
    de capacidade (512 MiB = desquantizacao f32 de [1024,131072] de uma camada,
    mais os tensores do lote), e nao da mascara de overflow.

### CORRECAO EVIDENCIADA, AINDA NAO APLICADA
  O get_rows de overflow deve operar sobre um recorte do store que cubra apenas as
  linhas OCUPADAS (ou ser executado no backend remoto), mantendo a capacidade do
  cache como esta. Nao ha base medida para alterar quantizacao, margem ou P.

### ESTADO
  Cruzamento P+256=96768 ABERTO. 105K nao iniciado. Nenhuma declaracao de sucesso
  de memoria. Producao restaurada; build-optimized intocado.

## F5 — EVIDENCIA DO SCHEDULER NO GRAFO DE CRUZAMENTO (F4SCHED, portado por env)

Trace: em ggml_backend_sched_compute_splits, por input de cada split, imprime
exec_backend, nome, op, type, bytes e owner_backend. Somente build-camp.

### Achado principal (medido, nao hipotetizado)
  F4SCHED split=1 exec_backend=CUDA0 input=kq_mask op=NONE type=f16
            bytes=24707072 owner_backend=CPU
  (repetido com as demais entradas do split: leaf_*, inp_pos, views, e
   leaf_68 = 65536 bytes = a mascara de overflow OCUPADA ja materializada)

  24.707.072 bytes = a mascara LOCAL em f16, dimensao [96512 keys, 128 queries].
  Sao 16 camadas x ~24.7 MiB = ~395 MiB de mascaras locais, TODAS com
  owner_backend=CPU, ou seja, cada uma e copiada host->device para o mesmo split
  CUDA0 e coexistem na reserva de gallocr.

  Logo o termo dominante da reserva de 606.06 MiB NAO e a mascara de overflow,
  NAO sao os gathers de overflow (1 MiB cada, ja corrigidos) e NAO e o store q4_0:
  sao as 16 MASCARAS LOCAIS de extensao correta (n_kv > P, logo as 96512 chaves
  locais SAO todas ocupadas — a mascara local esta corretamente dimensionada)
  coexistindo em um unico split CUDA0.

### O que isso invalida
  - A hipotese de "desquantizacao f32 de uma camada q4_0" (512 MiB) NAO se
    sustenta: o somatorio observado e de ~395 MiB de mascaras f16.
  - A hipotese de que o overflow gather/FA "sobe para a CUDA" tambem nao se
    sustenta como causa: o gather e de 1 MiB e a FA de overflow consome a
    mascara ocupada de 64 KiB; o excedente vem das mascaras LOCAIS.

### Implicacao para a correcao (evidencia, sem aplicacao)
  O scratch CUDA de ~606 MiB e, em boa parte, LEGITIMO para este desenho: as
  mascaras locais de 96512x128 f16 coexistem no mesmo split. As vias com base
  medida sao: (a) dividir o grafo em mais splits para que as mascaras de camadas
  diferentes nao convivam; (b) representacao triangular/compacta da mascara
  local; (c) reduzir o numero de camadas processadas por decode; (d) reduzir P.
  Nenhuma delas foi aplicada. Quantizacao, margem e P permanecem inalterados.
  Nao se crops as primeiras 256 linhas: as linhas fisicas vivas podem ser nao
  contiguas (removal/rollback/compactacao), logo qualquer recorte exigiria
  remapeamento explicito e validado de indices — a ser evidenciado antes.

### ESTADO
  Cruzamento P+256=96768 ABERTO. 105K nao iniciado. Nenhuma declaracao de sucesso
  de memoria. Producao restaurada; build-optimized intocado.

## F5 — PROVA DE IDENTIDADE DAS MASCARAS: DUAS AFIRMACOES RETIRADAS

Instrumentacao ampliada: F4SCHED agora imprime identidade do tensor
(ptr, root da cadeia de views, view_offs, src0), permitindo deduplicar aliases e
mencoes repetidas. Somente build-camp.

### RETRACAO 1 — "~395 MiB de 16 mascaras locais coexistindo"
  As duas mencoes de 24.707.072 B têm IDENTICO ptr e root:
    F4SCHED split=1 ... input=kq_mask ... bytes=24707072 ptr=0x7f9d41241a20 root=0x7f...
    F4SCHED split=1 ... input=kq_mask ... bytes=24707072 ptr=0x7f9d41241a20 root=0x7f...
  => E UM UNICO tensor de 24.7 MiB, mencionado duas vezes. Nao ha 16 mascaras
     materializadas, e portanto nao ha ~395 MiB de mascaras. RETRATADO.

### RETRACAO 2 — "os stores q4_0 de 72 MiB sao copiados para o split CUDA"
  Nenhuma mencao de bytes=75497472 (os stores q4_0 [1024,131072]) aparece como
  input de split em todo o log. Eles NAO sao copiados para o split CUDA.
  RETRATADO.

### O QUE A EVIDENCIA ATUALMENTE MOSTRA
  Deduplicando o grafo realmente capturado (10 tensores distintos):
    soma dos inputs distintos = 43.140 B (0.0 MiB); maior input = 40.960 B
    kq_mask ne0=256 (1.024 B) — a mascara PEQUENA do passo de decode
  Ou seja: a evidencia de scheduler coletada NAO COBRE o grafo que falha. Ela
  descreve o grafo de decode final (n_tokens pequeno), nao a reserva de
  gallocr que solicita 635.504.768 B.

### CONCLUSAO
  A reserva de 606.06 MiB continua SEM ATRIBUICAO CONFIRMADA. Nenhuma correcao de
  compartilhamento/lifetime tem base medida; nao implementei nenhuma. Sem reducao
  de P, sem mudanca de quantizacao, sem declaracao de sucesso de memoria.
  Cruzamento P+256=96768 ABERTO; 105K nao iniciado.

### Proximo passo exato
  Rastrear o ponto de falha da propria reserva (gallocr / reserve que pede
  635504768 B), com deduplicacao por identidade de tensor e registro de
  backend/lifetime, para obter a lista real de tensores在该 reserva. So com essa
  lista a correcao estreita de compartilhamento/lifetime pode ser decidida.
Producao restaurada; build-optimized intocado.

## F5 — MAPA DA RESERVA QUE FALHA (instrumentado, run nao alcançou a reserva)

Instrumentacao: em ggml_gallocr_reserve_n_impl, imediatamente ANTES da tentativa de
alocacao que falha, emitindo F4RESERVE (buf, buft, planned, current, n_nodes) e
F4RESERVE_SUM/F4RESERVE_TOP com ate 24 maiores.unique por raiz de views (aliases
deduplicados), bytes, offset, op e nome. Sem payloads. Somente build-camp.

### O que o run instrumentado mostrou
  F4RESERVE buf=0 buft=CUDA0 planned=63967360 current=0 graph_nodes=3671 n_buffers=3
  -> Este e o unico evento de reserva do run, e ele e o da INICIALIZACAO (61.00 MiB,
     3671 nos), NAO a reserva de 635504768 B que falha no cruzamento.
  Verificacoes no log: "failed to allocate CUDA0 buffer" = 0 ocorrencias;
  "allocating ... on device 0" = 0 ocorrencias.
  Causa: o worker deste run NAO CARREGOU o modelo — REQUEST_BUILD_FAILED com
  HTTP 500 {"model name=qwen-3.8-27b failed to load"} aos ~9 s. O cruzamento
  nunca foi exercitado; o "proxy error" aqui e consequencia da falha de carga.

### ESTADO HONESTO
  A instrumentacao do ponto de falha esta PRONTA e compila (CAMP_RC=0), porem
  AINDA NAO EXISTE MAPA DA RESERVA DE 606 MiB, porque nenhum run instrumentado
  alcanhou essa reserva. Consequentemente:
   - a causa concreta da reserva de 635504768 B continua DESCONHECIDA;
   - NENHUMA correcao foi implementada (nao ha evidencia para justificar uma);
   - sem reducao de P, sem mudanca de quantizacao/margem, sem declaracao de sucesso;
   - cruzamento P+256=96768 ABERTO; 105K nao iniciado.
  Retracao de mae repetida: ausencia de "prompt eval time" nao prova ausencia de
  prefill; aqui nem houve prefill porque o modelo nao carregou.

### Proximo passo imediato e unico
  Reexecutar o MESMO caso (P=96512, C=131072, n_kv=96768) garantindo carga limpa do
  modelo (verificar que nenhum processo de teste anterior ainda detem VRAM antes de
  iniciar), e entao coletar F4RESERVE na reserva que falha. Com essa lista real de
  tensores (identidade unica, bytes, offset, backend) a correcao estreita de
  alocacao/lifetime podera ser decidida — sem multiplicar mencoes de log.
Producao restaurada; build-optimized nao reconstruido.

## F5 — MAPA DA RESERVA QUE FALHA: CAPTURADO (P=96512, C=131072, n_kv=96768)

Run: f5-cross-P96512-L96768-RESMAP5. Instrumentacao F4RESERVE/F4RESERVE_SUM/
F4RESERVE_TOP em ggml_gallocr_reserve_n_impl, imediatamente antes da alocacao que
falha; aliases deduplicados por raiz de views; sem payloads; build-camp.

### Duas falhas de instrumentacao minhas, corrigidas (registradas)
  1) O dump percorria galloc->hash_set e desreferenciava chaves NULL (slot vazio de
     endereçamento aberto) -> SIGSEGV no filho, reportado como "failed to load".
  2) node_alloc nao tem membro buffer_id no nivel superior (existe em src[0]).
  Ambas corrigidas; o dump agora percorre apenas os arrays planos de nos/leafs.
  Tambem registrado: /v1/models lista modelos do preset ANTES do carregamento, logo
  readiness por ali e um falso positivo; a falha real de carga e impressa.

### A reserva que falha (evidencia direta)
  F4RESERVE     buf=0 buft=CUDA0 planned=635504768 current=113455872
                n_nodes=5103 n_leafs=1089
  F4RESERVE_SUM buf=0 unique=3810 bytes_in_buf=2832596992 (2701.38 MiB)
  (sequencia: 113324800 -> 113455872 -> 635504768; a ultima e a que estoura)

### O QUE A MAPA ESTABELECE
  - A reserva de 606.06 MiB e um AGREGADO de muitos tensores pequenos da CUDA:
    3.810 alocacoes unicas no buffer, 2.701.38 MiB de bytes vivos, exploited por
    sobreposicao de tempo de vida (o plano e 606 MiB, bem menor que a soma viva).
  - NAO ha um unico tensor dominante no buffer CUDA: os maiores tensores do grafo
    (16 pares de 197.656.576 B = 188.50 MiB) pertencem a nodebuf=1 (outro backend),
    nao ao buffer CUDA0 de compute. Nenhum TOP aparece com nodebuf=0.
  - Nao ha evidencia de tensor grande duplicado, nem de tensor de capacidade
    materializado no lugar do ocupado, neste nivel de atribuicao.

### CONCLUSAO E O QUE FALTA
  O mapa mostra uma NECESSIDADE AGREGADA de scratch CUDA para o grafo de
  cruzamento (606 MiB), contra um buffer atual de 108 MiB e ~631 MiB livres com o
  modelo carregado. Nenhuma correcao estreita de tensor individual se justifica
  por esta evidencia; aplicar algo agora seria repetir o erro de hipotese.
  Faltaria atribuicao por operacao/camada dentro do buffer 0 (quais ops e quantas
  copias por camada ocupam os 606 MiB) para decidir entre reducao de tempo de vida
  por graph split, representacao mais compacta, ou um ajuste de orcamento medido.
  ate essa medicao: sem fix, sem reducao de P, sem mudanca de quantizacao/margem.
  Cruzamento P+256=96768 ABERTO; 105K nao iniciado.
Producao restaurada (active, health=200); build-optimized nao reconstruido.

## F5 — MAPA DA RESERVA CORRIGIDA (atribuicao por dst): TENSOR RESPONSAVEL IDENTIFICADO

Run: f5-cross-P96512-L96768-DST2. Instrumentacao corrigida conforme revisao:
atribuicao de backend pelo descritor de DESTINO (node_allocs[k].dst), tamanho pela
alocacao (dst.size_max) e nao ggml_nbytes da primeira view, exclusao de alocacoes
invalidas/externas (buffer_id<0), unicidade por (buffer_id, raiz de views) dentro do
buffer, e lista dos maiores alocamentos DENTRO do buffer CUDA0. Sem payloads.

### RETRACAO
  "Os tensores de 188.50 MiB estao FORA da CUDA (nodebuf=1)" -> RETRADO. Com a
  atribuicao correta por dst, eles estao no buffer CUDA0 (chunk=0). A conclusao
  anterior veio de ler node_allocs[k].src[0].buffer_id (a FONTE), nao o destino.
  Tambem fica registrado: a soma de tamanhos por alocacao NAO e bytes vivos
  simultaneos; e por isso que o plano (606 MiB) e muito menor que a soma (11773 MiB).

### Reserva que falha (scalar confirmado)
  F4RESERVE     buf=0 buft=CUDA0 planned=635504768 current=113455872
                n_nodes=5103 n_leafs=1089
  F4RESERVE_SUM buf=0 unique_dst=1913 skipped_invalid=2482
                sum_dst_size_max=12345314592 (11773.41 MiB)  [soma por alocacao]

### TENSOR RESPONSAVEL (top dentro do buffer CUDA0)
  F4RESERVE_TOP buf=0 alloc=197656576 (188.50 MiB) chunk=0 off=437848192
                 op=NONE type=f16 ne=[256,96512,4] root=0x7f8655bdb830
  F4RESERVE_TOP buf=0 alloc=197656576 (188.50 MiB) chunk=0 off=20449408
                 op=NONE type=f16 ne=[256,96512,4] root=0x7f8655bdd7b0
  (+10 entradas iguais, alternando entre exatamente 2 offsets planificados)

  => A mascara LOCAL de cada camada e f16 com forma [n_tokens=256, n_local=96512, 4],
     ou seja 188.50 MiB POR CAMADA. Sao 16 camadas no desenho dual-FA; o
     planejador reaproveita por tempo de vida e chega a 606 MiB (cabem ~3 em
     coexistencia). Este e o responsavel medido, nao a mascara de overflow (64 KiB)
     nem os gathers de overflow (1 MiB), que ja foram corrigidos.
  => Observacao quantitativa: com ubatch 128 a mesma mascara e [128,96512] f16 =
     24.7 MiB (medido antes). Com ubatch 256 o componente 4-largo torna-a 188.5 MiB.

### Proximo passo (evidenciado, ainda nao aplicado)
  Alvo: a mascara local f16 [n_tokens, n_local, 4] de 188.5 MiB/camada. Como as
  chaves locais estao REALMENTE ocupadas (n_kv > P), nao se pode simplesmente
  recortar; as vias com base medida sao reduzir o componente 4-largo da codificacao
  da mascara, ou reduzir o ubatch (medido: 24.7 MiB/camada em 128), ou reduzir o
  numero de camadas por grafo. Qualquer uma exige medicao propria de pico e de
  throughput antes de ser adotada; P, quantizacao e piso de 614 MiB permanecem.
  Nada foi aplicado; cruzamento P+256=96768 ABERTO; 105K nao iniciado.
Producao restaurada (active, health=200); build-optimized nao reconstruido.

## F5 — CORRECAO DE ROTULO: [256,96512,4] f16 NAO E MASCARA; E K/V LOCAL MATERIALIZADO

Retracao do relulo anterior ("mascara local f16 [256,96512,4]"): NAO PROVADO e ERRADO.

### Prova dimensional
  Construcao da mascara local (src/llama-graph.cpp:4063-4065):
      inp->self_kq_mask_local = ggml_new_tensor_4d(ctx0, type, int64_t(p), n_q, 1, n_stream);
      nome = "attn_inp_kq_mask_local"
    => forma [p, n_q, 1, 1] = [96512, 128, 1, 1]. NAO e [256, 96512, 4].
  Materializacao do K/V local (src/llama-graph.cpp:4716 e 4719):
      k_local = kvarn_ctx->get_k_for_range(...);
      v_local = kvarn_ctx->get_v_for_range(...);
  Geometria do modelo: n_embd = 5120 => head_dim = 256 (com n_head = 20),
  n_kv_heads = 4.
  f16 [256, 96512, 4] = 256*96512*4*2 = 197.656.576 bytes
  Medido pelo dump: alloc = 197.656.576 bytes -> CASAMENTO EXATO.
  => 256 = head_dim, 96512 = n_local, 4 = n_kv_heads.
  => Sao os tensores de K e V LOCAIS materializados em f16 sobre toda a faixa
     local, NAO mascara. O fator 4 e a dimensao de cabecas KV, nao uma codificacao
     de mascara.

### Implicacao corrigida
  O termo dominante da reserva CUDA de 606 MiB e a DENSIFICACAO DO K/V LOCAL em f16
  (2 x 188.50 MiB por camada, 16 camadas), com reuso por tempo de vida reduzindo o
  plano a 606 MiB. Nao e a mascara de overflow (64 KiB, ja corrigida), nem os
  gathers de overflow (1 MiB, ja corrigidos), nem a mascara local (que e
  [96512,128] f16 = 24.7 MiB por camada, medida em outro ponto).
  Minha explicacao anterior ("componente 4-largo da mascara") estava errada e esta
  substituida por esta.

### Proximo passo, agora com base medida
  Alvo: evitar a materializacao f16 do K/V local em extensao total. As vias com
  base medida sao: (a) fatiar a FA local por faixas de chaves, de modo que o
  materializado vivo seja por fatia e nao a faixa local inteira; (b) fazer a FA local
  ler o store KVarN diretamente. Ambas sao mudancas de desenho que exigem medicao
  propria de pico, margem e throughput, alem de regressoes de corretude numerica
  (merge, posicoes, remapeamento de indices). NADA foi implementado.
  Nao se reduz ubatch nem capacidade "no escuro"; P, quantizacao e piso de 614 MiB
  permanecem. Cruzamento P+256=96768 ABERTO; 105K nao iniciado.
Producao restaurada e verificada (active, health=200); build-optimized nao reconstruido.

## F5 — ANALISE DE REUSO: POR QUE O RAMO LOCAL DO SPLIT NAO USA O CAMINHO DIRETO KVarN

### Caminho below-P que FUNCIONA (src/llama-graph.cpp:5100-5105)
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);            // acessor KVarN-aware
    ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0], k->ne[1], k->ne[2], k->ne[3], ...);
    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
  -> build_attn_mha recebe o tensor do STORE KVARN e trata nativamente
     (confirmado pelas traces F4MHA/F4TAIL: use_native_tail, compact_tail, route).
     Nao ha densificacao f16 de faixa local nesse caminho.

### Caminho do ramo LOCAL no split (src/llama-graph.cpp:4716-4721)
    k_local = kvarn_ctx->get_k_for_range(ctx0, il, ..., kRangeLocal, n_local,
                                         inp->self_kvarn_mat_idxs_local);
  E get_k_for_range (src/llama-kv-cache-kvarn.cpp:2016-2030) retorna
    stored_k[...]  OU  cache->get_materialization_source(shared_il, false)
  ou seja, em modo misto ele cai na FONTE DE MATERIALIZACAO — e o resultado medido
  e a densificacao f16 [256, 96512, 4] (head_dim, n_local, n_kv_heads) de
  188.50 MiB por tensor, que domina a reserva de 606 MiB.

### Ponto de decisao (evidencia, ainda sem implementacao)
  A diferenca entre os dois ramos NAO e a capacidade: no layout misto o store local
  KVarN tem capacidade P e contem fisicamente [0,P); o store remoto q4_0 tem (C-P).
  Logo, para o ramo LOCAL, get_k(ctx0, il) do caminho below-P e o mesmo objeto
  logico que get_k_for_range(kRangeLocal) tenta obter — e o primeiro NAO densifica.
  A interface a validar e, portanto: em modo position-split, get_k/get_v devolvem a
  forma NATIVA do store local (tipo/estagio KVarN) ou a forma materializada? A
  resposta esta em get_materialization_source e no contrato de build_attn_mha.
  Se devolver a forma nativa, a correcao estreita e: no ramo LOCAL do split,
  usar o mesmo par (get_k/get_v + build_attn_mha) do caminho below-P, mantendo
  mask_local, posicoes, tail e o merge/LSE com o ramo de overflow (ja com
  n_overflow ocupado apos a correcao anterior).
  Se for obrigatoriamente materializado, a interface/kernel que impede o reuso tem
  de ser documentada com o ponto exato, e so ai se avalia a solucao chunked.
  Nada implementado; a medicao decisive (tipo/shape devolvido por get_k em modo
  split) deve ser feita antes de qualquer edicao.
Producao restaurada e verificada; build-optimized nao reconstruido.

## F5 — EVIDENCIA DO KERNEL LSE + RESULTADO DOS GATES APOS REMOVER A GUARDA

### Qual kernel escreve o LSE no prefill KVarN nativo, e o que ele suporta
  ggml/src/ggml-cuda/fattn-kvarn-dispatch.cu, ggml_cuda_flash_attn_ext_kvarn_supported():
    - aceita explicitamente o LSE de position-split ("plan §3.4, F1b"): as rotas
      KVarN decode/windowed/portable publicam O e LSE a partir do mesmo metadado
      (max, denom), verificado numericamente contra a referencia CPU em
      tests/test-kvarn.cpp (test_native_flash_attention_lse_parity);
    - formas genericas sem sub-caminho LSE-capaz sao recusadas em
      ggml_cuda_fattn_ext_mma_kvarn_launch_case, de modo que nenhuma forma pode
      alcancar um kernel que ignoraria o anexo de LSE;
    - o predicado NAO impoe teto de n_q. O unico teto era o portao provisorio de
      grafo (q_cur->ne[2] <= 16) em llama-graph.cpp:4683, agora removido.

### Resultado dos gates (build-f4, guard removida)
  test-position-split-real-pair : 8490 checks, 0 failures  (oráculo O/LSE canônico,
      incluindo [NATIVE D128 Q1 tail body+tail] com A/A2 a 0 ULP e todos os gates 0.0)
  test-kvarn                   : FALHOU
      FAILED GPU/d128-q256-allmsk: masked plain O[15] is nan instead of exact 0
      1572864 value-contract failures (32 details)
      -> "position-split LSE structural matrix has failing shapes"; core dump

### CONCLUSAO (a guarda nao pode ser apenas removida)
  Com n_q grande a rota KVarN nativa passa a executar e, em posicoes totalmente
  mascaradas, produz NaN em O plain onde o contrato exige 0 EXATO. Ou seja: existe
  uma combinacao NAO suportada (prefill n_q=256 com mascara completa) que so aparece
  quando a guarda e removida. A equivalencia com o caminho comum nao pode ser
  afirmada, e o ordinary output nao prova LSE.
  A correcao deve ser no dispatch/kernel (contrato de saida mascarada -> 0 exato,
  e LSE -inf coerente), verificada pelos mesmos testes, e so depois reavaliar a
  guarda por checagem de capacidade real.
  Nenhuma afirmacao de aceite: 105K nao iniciado. Producao restaurada e verificada.

## F5 — CRUZAMENTO REAL 27B APROVADO + LIMITACAO DE DISCO (2026-10-07 ~20:31 UTC)

### CRUZAMENTO (evidencia aprovada)
  run_id: f5-cross-P96512-L96768-FINAL
  log: docs/occupancy-evidence/f5-f5-cross-P96512-L96768-FINAL-n96768.log
  build: build-camp (CAMP_RC=0, binary 22:10); build-f4/llama-server ligado a
         libllama/libggml-cuda frescos verificados via LD_TRACE_LOADED_OBJECTS
  config: P=96512 C=131072 n_kv=96768, batch=512 ubatch=256, build limpo,
          guards de teste presentes porem desligados (sem env F4_*), mesma
          quantizacao KVarN4 + faixa remota q4_0
  reserva CUDA: 113455872 B (108 MiB) ; failed-to-allocate = 0
  prompt eval time = 156975.58 ms / 96768 tokens (616.45 tok/s)
         eval time =   3148.07 ms /    32 tokens (  9.85 tok/s)
  pico VRAM = 11667 MiB de 12282 -> LIVRE 615 MiB (piso 614 MiB) -> PASS
  MemAvailable_min = 9203684 kB ; swap_used_max = 755824 kB
  CONFIRMADO: KVarN local NATIVO mantido; a densificacao f16 de faixa local
  completa (188.50 MiB por K/V por camada) desapareceu.

  RESSALVAS (nao declarar F5 aprovado):
   - folga de apenas 1 MiB sobre o piso, dentro da variacao entre execucoes ja
     medida (~27 MiB a P fixo) => exige o protocolo de repeticao antes de fixar
     a fronteira;
   - 9.85 tok/s de decode vem de uma amostra de 32 tokens e NAO pode ser comparado
     com os 25 tok/s do below-P (nao pareado);
   - aceite numerico do cruzamento ainda nao e medido.

### LIMITACAO DE DISCO (registrada; nada apagado)
  / : 465G total, 439G usados, 1.9G livres (100%)
  /home/hjotha/f4-spool           = 94 GiB
  /home/hjotha/f4-spool/f5-campaign = 83 GiB
  maiores: f5-sweep-r1 21G, r2 14G, r3 11G, e varios prefill de 6.9G
  docs/occupancy-evidence = 2.4 GiB
  Consequencia: o auto-save do shutdown FALHOU com "No space left on device".
  NENHUM artefato, snapshot ou evidencia foi apagado ou truncado.
  Politica para os proximos runs: sem slot auto-save, sem escritas de snapshot
  desnecessarias, logs limitados e sem eco do prompt de 96768 tokens; as
  configuracoes de producao e toda a evidencia F4 permanecem intactas.
  Qualquer limpeza de disco exige aprovacao do owner.

## F5 — CRUZAMENTO REPETIDO (1 warmup + 4 measured) + 105000 ACIMA DE P (2026-10-07 21:20 UTC)

### CRUZAMENTO REPETIDO P=96512, L=96768 (auto-save OFF, instrumentacao OFF)
  run_dir: /home/hjotha/f4-spool/f5-campaign/cross-repeat-96512-96768
  r0 warmup=1 free=615 PASS pp_tps=632.58 tg_tps=10.515
  r1..r4 measured: free=615 (todos)  pp_tps median=622.933 (min 620.504 max 626.299)
                             tg_tps median=10.480 (min 10.426 max 10.508)
  floor_pass=4/4 ; pico=11667 MiB estavel em todas as execucoes
  NOTA: apenas 4 execucoes medidas (r0 e warmup). O protocolo canonico da secao 5
  pede 5; a quinta ainda NAO foi medida. Medianas acima sao de 4 amostras e nao
  podem ser llamadas resultado canonico.
  Margem: 615 MiB vs piso 614 MiB = 1 MiB. Aqui a variacao observada entre
  execucoes foi ~0 (todos 615), mas a variacao historica medida a P fixo foi ~27 MiB
  em outro contexto; permanece necessario repetir antes de fixar fronteira.
  p95 por token NAO medido: o coletor atual so tem tempos agregados da resposta.
  Nao se deriva p95 de 1/TPS nem da mediana de TPS.

### 105000 TOKENS ACIMA DE P (P=96512) — FALHA REAL DE MARGEM
  run: f5-sweep-aboveP-105000-v2  (log: f5-sweep-aboveP-105000-v2-P96512.log)
  VALIDADE (gate fail-closed novo, /tmp/opencode/f5-valid.py):
    VALID|PASS 499.783 4.335 210091.125 7151.386 105000 32  -> execucao VALIDA
  Medido:
    pico VRAM = 11701 MiB -> LIVRE 581 MiB ; piso 614 MiB -> FAIL (faltando 33 MiB)
    prompt eval time = 210091.12 ms / 105000 tokens (499.78 tok/s)
           eval time =   7151.39 ms /    32 tokens (  4.33 tok/s)
 Ou seja: o cruzamento de 96768 passa com folga de 1 MiB, mas 105000 (acima de P,
faixa de overflow engajada) estoura a margem em 33 MiB. O candidato P=96512 NAO e
seguro para o cenario acima de P com o orcamento atual.

### Bug de harness encontrado e corrigido (rc26 invalido anterior)
  f5-sweep.sh usava req-96000.json fixo enquanto o construtor recebia REQLEN;
  a primeira tentativa de 105000 leu um arquivo inexistente (curl rc=26) e foi
  reportada como PASS de VRAM. Corrigido: caminho derivado de REQLEN,
  pre-condicoes (arquivo existe, nao vazio, JSON com exatamente REQLEN tokens)
  e veredito fail-closed sobre RC/HTTP/erro/tokens_evaluated/tokens_predicted/
  timings. Fixtures negativas (missing/http500/bad JSON/error object/zero
  timings/wrong token count) e um controle positivo foram validados: todos
  INVALID, o controle VALIDO. Nenhuma rebuild de modelo/kernel foi feita.

### Estado
  - Nenhum artefato apagado; disco 1.9 GiB estavel; runs isolados <1 MiB com
    auto-save OFF.
  - 105000 acima de P: FALHA de margem medida. P=96512 continua sendo o maior
    candidato repetidamente aprovado ABAIXO de P, mas nao pode ser aceito para
    os cenarios acima de P sem decisao de orcamento (ou P menor, ou reducao de
    batch/ubatch de prefill medida, ou outra via com evidencia).
  - 115000 / 130560 e baseline pareado XXXL por camada: NAO executados.
  - Numerico do cruzamento: ainda nao medido. Execucao do cruzamento e sucesso;
    aceitacao numerica e de performance permanecem NAO medidas.
Producao restaurada e verificada; build-optimized nunca reconstruido.

## F5 — SELECAO DE P NO PIOR CASO (131072) E REGISTRO DE DISCO (2026-10-07 22:56-23:17 UTC)

### Chave de validade reforcada (f5-valid.py)
  Rejeita HTTP nao numerico/vazio (INVALID explicito, nunca string vazia aceita),
  timings nao-objeto, e qualquer prompt_ms/predicted_ms que nao seja POSITIVO E
  FINITO (NaN/Inf rejeitados). O TPS passa a ser DERIVADO das contagens e tempos
  verificados (ev*1000/prompt_ms), nunca lido da resposta. Fixtures: 8 negativos
  (missing, http500, http vazio, http nao numerico, bad JSON, error object,
  zero timings, token count errado) + 1 controle positivo -> todos coerentes.
  Sweep: guarda de P antes de parar a producao (P precisa ser multiplo positivo
  de 256; caso contrario INVALID_CANDIDATE sem tocar na GPU).

### Pior caso = endpoint completo C=131072 (prompt 130560 + 512 gerados)
  L=130560 com n_predict=32 NAO fecha o endpoint canonico (130592 de 131072).
  Runs com NPREDICT=512 exercitam 130560 + 512 = 131072 tokens reais.
  Bisection (P/256 em coordenadas inteiras):
    P=90112  free=669 PASS      (L=130560, +32)
    P=93184  free=617 PASS      (L=130560, +32)
    P=96512  free=583 FAIL      (L=130560, +32)
  Endpoint completo (130560 + 512):
    P=93184  free=615 PASS      VALID|PASS 175.031 prefill / 1.149 decode,
                                tokens_evaluated=130560 tokens_predicted=512
                                total=131072 == C
    P=93440  free=613 FAIL      (P+256 do candidato, MESMO endpoint completo)
  => P=93184 e o maior multiplo de 256 medido que passa no endpoint completo, com
     folga de 1 MiB, e o vizinho P+256 reprova por 1 MiB. A escala e' tao fina que
     um unico par NAO fixa P: historico ja mostreu variacao maior que 1 MiB.
  => P NAO CONGELADO. Faltam execucoes repetidas do endpoint completo em 93184
     antes de qualquer congelamento, e depois numerico + p95 pareado.

### Disco: mecanismo do crescimento e cleanup aprovado (sem atribuicao especulativa)
  Mecanismo comprovado do consumo: auto-save de slot. Evidencia direta em
  docs/occupancy-evidence/f5-f5-cross-P96512-L96768-FINAL-n96768.log:
    llama_state_seq_save_file: error saving sequence state file:
        stream: callback failure: write error: No space left on device
  Run isolados com auto-save OFF nao gravam snapshots (ex.: TIMING2 = 8 KB).
  Crescimento anterior de 1.8 -> 50 GiB permanece NAO ATRIBUIDO: nao ha evidencia
  de limpeza alheia; o unico mecanismo comprovado e' a escrita de auto-save.

  Cleanup permanente EXECUTADO pelo owner worker (nao por mim), com aprovacao:
    docs/occupancy-evidence/approved-old-f5-snapshot-cleanup-20261007T2316Z.json
    approval: "Telegram551 replied to550; exact four old F5 directories;
               completed auto-*.bin only"
    14 arquivos auto-*.bin, 48.07 GiB, apenas em f5-n96000-003 e sweep-r1/r2/r3
    before_available=49.5 GiB -> after_available=97.5 GiB
  Preservados: metrics/provenance/req/resp/vram-samples por run, slots vazios,
  summary.txt, e TODA a evidencia F4 e logs. Nenhuma limpeza adicional autorizada
  nem executada. Estado atual: f5-campaign 35 GiB, f4-spool 46 GiB, / 98 GiB livre.

## F5 — REPETICOES DO ENDPOINT + PREPARACAO CPU-ONLY DOS GATES RESTANTES (2026-08-08 00:05 UTC)

### Endpoint completo C=131072 em P=93184 (3 execucoes, todas VALIDAS)
  prompt 130560 + 512 gerados = 131072 tokens reais (total == C)
  run                                                    peak  free  veredito  prefill_tps
  f5-sweep-endpoint-131072-P93184                       11667  615  PASS      175.031
  f5-sweep-endpoint-repeat-93184-r1                     11667  615  PASS      176.023
  f5-sweep-endpoint-repeat-93184-r2                     11667  615  PASS      176.893
  vizinho P+256 = 93440 (mesmo endpoint)                11669  613  FAIL      177.911
  => tres PASS consistentes (615/615/615) e P+256 reprova por 1 MiB. A fronteira
     de 1 MiB e' fina mas REPRODUZIU em tres execucoes; ainda assim nao e'
     congelada sem o resto dos gates. P permanece 93184 como candidato.

### Preparacao CPU-only (sem GPU, sem rebuild) dos gates restantes
  1) p95 por token REAL (tools/occupancy-f5/f5_harness.py:1571-1578):
     o harness exige, por run, prefill_ms, ttft_ms e token_latencies_ms com
     >= min_token_samples=32 amostras finitas e > 0; e calcula
     decode_p95_ms = mediana dos p95 por run (percentile_nearest_rank).
     Consequencia: o coletor atual (tempos agregados) NAO mede p95 e p95 nao pode
     ser derivado de 1/TPS nem da mediana de TPS.
     Coletor novo /tmp/opencode/f5-perToken.py: cliente SSE que marca o relógio
     de parede a CADA token emitido e grava as latências entre tokens. Rotulado
     como 'client_side_end_to_end_including_transport' porque inclui transporte:
     é comparável apenas com outra captura client-side nas mesmas condições.
     prefill_ms/ttft_ms continuam vindos do servidor (autoridade para prefill).
  2) Manifests para o harness (same real 27B, mesmas condições):
     model_path  = /home/hjotha/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf
     model_sha256= e23eb251493b8a89f7d0217590b593e696ac9bacaeac8605541e477533dc5838
     corpus (mesmo prompt determinístico usado nos runs) =
                  /home/hjotha/f4-spool/f5-campaign/cross-repeat-96512-96768/req-96768.json
     corpus_sha256 = f61fd62d99eea99036118da75892725d14677c43775ce4158bb27cc43ca7ce12
     stages = load, prefill, decode, restore + boundary_cross
     abaixo de P: baseline XXL (C=102400, CUDA-only); acima de P: baseline XXXL
     por camada (NAO um CUDA-only de 131K, que comprovadamente nao carrega).
  3) Gates de qualidade (separados de memoria): kl_mean_nat_per_token e
     perplexity por caso, com limite de perplexidade <= 1.01 x baseline
     (f5_harness QUALITY_PPL_RATIO_LIMIT=1.01) e KL candidato <= KL baseline +
     0,001 nat/token. Isso exige referencia F16 e ainda NAO foi medida.
  4) Protocolo: 1 warmup descartado + 5 runs medidos por braco; o harness ja
     valida contagem de repeticoes.

### Estado
  Memoria: P=93184 aprovado 3x no endpoint completo; P+256 reprova.
  Numerico (O/LSE, KL, perplexidade) e p95 pareado: NAO medidos.
  Performance: ainda sem baseline pareado, portanto sem veredito.
  Producao restaurada e verificada; build-optimized nunca reconstruido;
  auto-save OFF em todos os runs; / com 92-98 GiB livres, sem nova limpeza.

## F5 — REVISÃO INDEPENDENTE DO COLETOR p95 + BLOCO CANÔNICO DE MEMÓRIA (2026-10-08 00:50 UTC)

### Revisão do coletor: cada objeção do revisor verificada no código desta árvore
  1) "Stream:true no header HTTP não ativa streaming" -> CONFIRMADO como bug.
     server-context.cpp:10731 lê `json_value(data, "stream", false)`, ou seja o
     corpo JSON. Corrigido: o coletor injeta stream=true no corpo.
  2) "content não garante 1 token" -> CONFIRMADO e agora VERIFICADO no servidor:
     server-task.cpp:1100-1112 (to_json_non_oaicompat) emite por evento
     "content", "tokens":[ids deste evento] e "tokens_predicted": n_decoded.
     Logo a cardinalidade é verificável sem mexer no servidor: cada evento de
     conteúdo precisa ter len(tokens)==1 e tokens_predicted avança exatamente 1.
     Evento com 0, 2+ tokens ou sem o campo "tokens" => NOT_PER_TOKEN,
     rejeitado; nenhum timestamp é inventado. run_token_events nunca assumido.
  3) "want_tokens nunca validado" -> agora validado: o evento terminal precisa
     casar com n_predict pedido, senão INVALID.
  4) "eventos malformados pulados em silêncio" -> agora contados e rejeitados.
  5) "estado final (stop/tokens_predicted/timings) não checado" -> agora
     extrato e gravado; [DONE] obrigatório.
  6) "prompt_ms não é TTFT" -> corrigido: o campo exposto chama-se
     client_ttft_ms (chegada do primeiro token no cliente). prompt_ms continua
     sendo autoridade de prefill, TTFT não é pré-misturado.
  7) "intervalos de recebimento não são limite superior estrito" -> aceito e
     rotulado: capture="client_receive_intervals",
     timing_semantics registra explicitamente que buffering/batching pode
     agrupar chegadas e que NÃO é upper bound por token do servidor;
     predicted_ms/predicted_n do servidor permanecem autoridade de decode.
  8) "p95 por tempos agregados" -> o harness exige token_latencies_ms por run
     (>=32 amostras, f5_harness.py:776-806); o coletor entrega as amostras.
  Bugs reais encontrados e corrigidos nos fixtures:
     - terminal usava setdefault, congelando tokens_predicted no 1º token e
       fazendo a checagem terminal falhar sempre (peguei por fixture 1);
     - caminho de uso imprimia o valor de retorno em vez do motivo.
  Fixtures CPU-only (servidor SSE falso, sem GPU): 10 casos, todos coerentes --
    1 token/evento (aceito, client_ttft gravado), 2 tokens no evento, campo
    "tokens" ausente, contador pulando, [DONE] ausente, evento malformado,
    error object, contagem terminal divergente, zero tokens observados, uso.

### Blocker registrado: não existe timestamp por token no backend
  server-common.cpp:166 (server_slot_stats::to_json) é AGREGADO
  (prompt_n/prompt_ms/predicted_n/predicted_ms/...). timings_per_token
  (server-context.cpp:7017) só anexa esse agregado a cada evento parcial --
  NÃO emite amostras por token. Para latência por token estritamente no servidor
  seria preciso campo novo em server_slot_stats + build de teste; como produção
  não pode ser reconstruída, o p95 desta campanha é de intervalos de
  recebimento no cliente com cardinalidade verificada, rotulado como tal.

### Bloco canônico de memória em execução (1 warmup + 5 medidos, sequencial)
  /tmp/opencode/f5-endpoint-canonical.sh: P=93184, C=131072, REQLEN=130560,
  NPREDICT=512, 1 warmup DESCARTADO + 5 medidos, um processo por vez, restauração
  de produção verificada ao fim de cada run. Iniciado 00:43:31Z.
 runs prévios de apoio (não substituem o bloco canônico): 4 execuções, todas
  peak=11667 free=615 PASS; vizinho P+256=93440 free=613 FAIL.

## F5 — RECONCILIAÇÃO DAS EXECUÇÕES + REVISÃO DO COLETOR (2026-10-08 00:49 UTC)

### Bug crítico do coletor, confirmado no código e corrigido
  /completion é TASK_RESPONSE_TYPE_NONE e server-context.cpp:10829-10837 termina
  o stream com output="" — NÃO existe sentinela "data: [DONE]" nesse endpoint.
  O coletor exigia [DONE] e portanto rejeitaria TODA resposta válida com
  no_terminal_done_event. Corrigido: terminação por stop:true + contagens
  finais + EOF limpo; [DONE] é aceito se presente (modos OAI), não exigido.
  Base de cardinalidade agora verificada no servidor:
    server-context.cpp:6998-6999 -> res->content = tkn.text_to_send;
                                     res->tokens  = { tkn.tok }
    ou seja, exatamente UM id de token por evento de decode, incondicionalmente
    (não depende de return_tokens, que afeta a lista final).
    server-context.cpp:6782-6786 -> text_to_send pode vir VAZIO quando há UTF-8
    incompleto; esses tokens são reais e NÃO são descartados.
  Novo contrato fail-closed do coletor:
    - return_tokens:true no corpo (server-schema.cpp:53);
    - len(tokens)==1 por evento; len!=1 => NOT_PER_TOKEN rejeitado;
    - tokens_predicted cumulativo (n_gen) deve avançar exatamente 1 por evento;
    - evento sem tokens, sem prompt_progress e sem stop => rejeitado;
    - observed == contagem autoritativa final == n_predict pedido; caso
      contrário rejeição explícita (early_stop_or_short com stop_type);
    - rótulo verified_one_token_per_event SÓ quando o cumulativo esteve
      presente e casou; sem cumulativo o rótulo é
      single_token_events_count_reconciled_without_cumulative;
    - progress events de prefill (prompt_progress) não são tokens de decode;
    - client_receive_gaps_ms continua separado e rotulado como intervalo de
      recebimento no cliente, NÃO latência por token do servidor.
  Fixtures CPU-only (12 casos, forma nativa real: sem [DONE], stop:true+EOF,
  token de conteúdo vazio): forma nativa ACEITA e verificada; chunk com 2
  tokens REJEITADO; sem cumulador => rótulo reconciliado (não verified);
  sem cumulador e curto => rejeitado; early stop => rejeitado; salto de
  cumulador => rejeitado; evento malformado => rejeitado; EOF sem stop:true =>
  rejeitado; evento sem tokens/progress/stop => rejeitado; error object =>
  rejeitado; zero tokens => rejeitado; uso => exit 2. Nenhuma reconstrução de
  código GPU para estas correções (só Python).

### Reconciliação das execuções de endpoint em P=93184 (sem refazer 6 do zero)
  4 execuções válidas e IDÊNTICAS em binário/config/request/protocol:
    binário build-camp sha256 7b8634fc2692be0bbb28aa1038bc0f6217734e0eba8b142f4decf493205a1de0
    preset-nsave (auto-save off) sha 74c7dfc468d1, HTTP 200, P=93184,
    REQLEN=130560, NPREDICT=512 -> total 131072 == C
    inicial (22:55) / r1 / r2 / r3: todas peak=11667 free=615 PASS
  Elas NÃO satisfazem "1 warmup descartado + 5 medidos" de forma estrita: nenhuma
  foi designada como warmup descartado antes de medir, e redesignar agora seria
  encaixar o protocolo. Portanto: contadas como 4 medições válidas de
  reprodutibilidade; falta 1 medido para completar 5, precedido de um warmup
  descartado de verdade (o run de 00:43:31Z, em andamento). O bloco de 6 runs
  que havia iniciado foi interrompido pelo parent para não repetir 6 do zero;
  o run em voo foi preservado e se restaura sozinho.
  Evidência já medida do vizinho P+256=93440: free=613 FAIL no mesmo endpoint.

### Manifests de base preparados (CPU-only) e o que ainda falta
  /home/hjotha/f4-spool/f5-campaign/manifests/
    manifest-belowP-xxl-baseline.json            (C=102400 CUDA-only, split OFF)
    manifest-aboveP-xxxl-perlayer-baseline.json  (C=131072 + Radeon Q4 por camada)
  Contexto comum: mesmo 27B real
    model_sha256 e23eb251493b8a89f7d0217590b593e696ac9bacaeac8605541e477533dc5838
    corpus req-96768.json sha f61fd62d99eea99036118da75892725d14677c43775ce4158bb27cc43ca7ce12
    cache_types local kvarn4 / overflow q4_0; hardware RTX 4070
    vram_total_bytes=12878610432; sampling temp 1.0 top_p .95 top_k 20 seed 42;
    clocks power 200/170, mem 10501; cache_state=cold; b/ub 512/256.
  Ambos passam em context_problems (schema_version=f5-context-v1, c>=130560,
  casos prescritos). São MODELOS, não evidência: validate() rejeita os dois com
  27 razões, que viram a lista autoritativa do que falta:
    provenance.evidence_origin, bindings, runs dos 5 casos prescritos
    (below_p_46592, above_p_93440/105000/115000/130560), quality, capacity com
    occupied_positions=131072, over_capacity_tokens=131073, gen 130560+512,
    restore (exact_import + cache_miss_fallback) e cycles=10.
  Nota: expected.c é a capacidade da campanha (131072); o C=102400 do braço XXL
  fica em provenance.arm_launch_c, porque o harness rejeita c<130560.

## F5 — CONDIÇÕES EXATAS DE ACEITE (lidas do harness, não inferidas) (2026-10-08 01:10 UTC)

  Para cada caso prescrito, o harness agrega por braço (mediana entre runs) e
  compara candidato (split) contra o braço base prescrito:
    abaixo de P  -> base = XXL  (C=102400 CUDA-only)
      prefill_ms  <= 1.05 x base   (LATENCY_GATE_LIMIT_REGRESSION)
      ttft_ms     <= 1.05 x base
      decode_p95_ms <= 1.05 x base
    acima de P   -> base = XXXL (C=131072, Radeon Q4 por camada)
      prefill_ms  <= 0.90 x base   (PREFILL_GATE_LIMIT_IMPROVEMENT)
      ttft_ms     <= 0.90 x base
      decode_p95_ms <= 1.05 x base
  Qualidade (por caso, ambos os braços): kl_mean_nat_per_token <= base + 0.001
  (QUALITY_KL_DELTA_LIMIT) e perplexity <= 1.01 x base (QUALITY_PPL_RATIO_LIMIT).
  Telemetria por run: estágios load/prefill/decode/restore (+ boundary_cross acima
  de P) com peak_bytes nos backends cuda, vulkan e host; acima de P exige
  vulkan.attention_copy_bytes, abaixo de P exige attention_copy_bytes == 0;
  host exige mem_available_min_bytes >= 2147483648 (2 GiB) e swap; vulkan exige
  gtt_peak_bytes e gtt_available_min_bytes.
  Capacidade: occupied_positions=131072, over_capacity_tokens=131073,
  gen_start=130560, gen_count=512, gen_end=131072, estado intacto após recusa.
  Restauração: cenários exact_import e cache_miss_fallback nos comprimentos
  exigidos. Ciclos: RESTORE_CYCLES=10 com retained_bytes <= 67108864 (64 MiB).
  Amostragem: por run, prefill_ms, ttft_ms e token_latencies_ms com no mínimo
  DEFAULT_MIN_TOKEN_SAMPLES amostras finitas e positivas; decode_p95_ms é a
  mediana dos p95 por run (percentile_nearest_rank).

  Ferramentas prontas para a coleta:
    /tmp/opencode/f5-p95.sh  braço único por execução (ARM=split|xxxl|xxl),
      auto-save off, produção parada uma vez e restaurada no exit, mede pico de
      VRAM, MemAvailable mínimo, swap máximo, e chama o coletor SSE;
    /tmp/opencode/f5-p95.py  p95 nearest-rank conferido contra
      f5_harness.percentile_nearest_rank (idêntico em 511 amostras) — nunca
      1/TPS, nunca mediana de TPS.

## F5 — BLOCO DE MEMÓRIA COMPLETO EM P=93184 (2026-08-08 01:25 UTC)

  Medidos (mesmo binário sha 7b8634fc… build-camp, mesmo preset auto-save off,
  mesmo request REQLEN=130560+NPREDICT=512=131072==C, mesmo protocolo):
    inicial  22:55  peak=11667 free=615 PASS  pp=175.031 tg=1.149
    r1       23:19  peak=11667 free=615 PASS  pp=176.023 tg=1.176
    r2       23:44  peak=11667 free=615 PASS  pp=176.893 tg=1.184
    r3       00:14  peak=11667 free=615 PASS  pp=176.643 tg=1.184
    m5       01:04  peak=11667 free=615 PASS  pp=175.411 tg=1.180
  => 5 MEDIDOS válidos, todos idênticos (peak 11667, free 615, folga 1 MiB
     sobre o piso fixo de 614). Descartei o warmup de 00:43:31-01:03:21
     (VALID|PASS peak=11667 free=615) — descartado por protocolo, registrado.
  Vizinho P+256=93440 no MESMO endpoint completo: peak=11669 free=613 FAIL.
  Conclusão de memória: P=93184 é reprodutível como maior múltiplo de 256 que
  passa no pior caso (131072). A margem de 1 MiB é reproduzida em 6 execuções.
  isto é evidência de MEMÓRIA; não é aceitação F5 (faltam numerico, qualidade,
  p95 pareado, capacidade, restauração e ciclos).

## F5 — INSTRUMENTAÇÃO SERVER-SIDE DE LATÊNCIA POR TOKEN (2026-10-08 02:20 UTC)

### Por que o cliente não serve (fatos do código, não suposições)
  1) send_partial_response só é chamado quando o texto está completo em UTF-8:
     server-context.cpp:6776-6811 calcula `incomplete = validate_utf8(...)` e só
     então emite o evento parcial. Token com texto retido NÃO gera evento, mas
     slot.stats.n_gen continua contando -> saltos de tokens_predicted.
     Execuções reais registraram tokens_predicted=177 com 175 eventos.
  2) server-context.cpp:10851-10862 envia keep-alive SSE ":\n\n" a cada
     sse_ping_interval (default 30s); o coletor contava isso como malformado
     (malformed_events:2). Agora comentários SSE são ignorados (fixture nova).
  Logo o intervalo entre chegadas no cliente NUNCA é latência por token.

### Instrumentação nova (só build de teste; produção intocada)
  server-common.h (server_slot_stats): amostras por token e por iteração de
  decode, vetores limitados a F5_PER_TOKEN_MAX=16384, capacidade reservada
  INTEIRA no f5_arm (sem realocação no laço), contadores exatos, sem I/O, sem
  log, sem JSON no laço de decode.
    - f5_begin_iteration() no início de cada iteração (base por iteração):
      SEM o begin, tokens_per_iteration virava 1,2,3... (bug real apontado na
      revisão); agora é exatamente 1 token por iteração no caminho canônico.
    - primeiro intervalo vira f5_ttft_ms (vai de reset/prompt até o 1º token) e
      é EXCLUÍDO de token_ms, que é a entrada do p95 de decode.
    - token aceito em passo especulativo marca f5_spec_seen; o consumidor
      RECUSA p95 por token se houver especulação ou iteração != 1 token.
    - truncamento marca f5_truncated e o contador de tokens continua exato.
  server-context.cpp: gate por env LLAMA_SERVER_F5_PER_TOKEN_TIMING=1 lido UMA
  vez (static local no reset), arm por requisição, hook no caminho do token
  após sincronizar o contexto (t_now) e antes de qualquer texto ao SSE.
  server-task.cpp: serialização BATCHED única no resultado final (f5_per_token).
  Teste unitário tests/test-f5-per-token-timing.cpp (CPU) com asserções para:
  desligado por padrão, capacidade total reservada, 64 tokens -> 1 TTFT + 63
  intervalos, attribution 1 por iteração, bandeira especulativa, truncamento sem
  perder o contador exato, re-arm sem sangramento entre requisições. Resultado:
  OK (exit 0).

### Build e verificação de backend
  O build-f5 (configure novo) não tinha CUDA: "--device CUDA0 invalid" era
  falha de configuração do build, não do modelo. Reaproveitado o build-camp,
  que já tem CUDA+Vulkan; só objetos de servidor recompilados (0 arquivos .cu).
  Binário preservado antes da troca em
    f4-spool/f5-campaign/binary-pre-f5tok/SHA256SUMS.txt
      llama-server 7b8634fc…  libllama-server-impl.so d3057930…  (usado na memória)
  Após a instrumentação:
    llama-server 7b8634fc…  libllama-server-impl.so b6efe0f2…
  Binário de PRODUÇÃO inalterado: 3220fcd4…
  --list-devices conferido ANTES de qualquer downtime: CUDA0 RTX 4070 11902 MiB,
  Vulkan0 AMD 8091 MiB, Vulkan1 NVIDIA 12528 MiB.

### Validação mínima real (64 tokens) — aprovada
  ARM=split P=93184 REQLEN=4096 NPREDICT=64 F5TOK=1:
    tokens_recorded=64 decode_iterations=64 samples=63
    speculative_seen=False truncated=False tokens_per_iteration=1
    tokens_recorded == final tokens_predicted (64 == 64)
    server_decode_p95_ms(nearest_rank)=29.9440 median=28.5490
    cross-check agregado: predicted_ms 1817.414 / 64 = 28.40 ms vs média por
    token 28.85 ms (diferença ~1.6%, compatível: o agregado inclui o 1º token)
    cliente (referência separada): 63 gaps, client_ttft_ms 5325.03
    pico 11353 MiB, free 929 MiB, MemAvailable min 9313752 kB, swap max 855492 kB

## F5 — CAPTURA CANÔNICA DE TEMPO SEM DEPENDÊNCIA DE SSE (2026-08-08 02:45 UTC)

### Bug de wiring confirmado em execução real (não em teoria)
  O coletor SSE abortava em cumulative_jump (tokens_predicted=177 com 175
  eventos) ANTES de receber o evento final, então o vetor server-side autoritativo
  era descartado junto. Causa: com UTF-8 retido, alguns tokens não geram evento.
  Warmup split 02:22 e measured2 falharam igual; 2 execuções split inválidas
  (nenhuma contada), 4 execuções XXL válidas preservadas.

### Captura canônica agora: resposta final NÃO-streaming
  /tmp/opencode/f5-nontimer.py: stream:false + return_tokens:true, grava o JSON
  final VERBATIM (.final.json) e valida em modo fail-closed:
    tokens_predicted == n_predict pedido
    tokens_recorded == tokens_predicted
    decode_samples == n_predict - 1 (primeiro intervalo é TTFT, fora do p95)
    truncated == false, speculative_seen == false
    toda iteração com exatamente 1 token
    todas as amostras finitas e > 0
  O p95 de decode vem do vetor do SERVIDOR (token path, após sync do ctx),
  imune a retenção UTF-8 e a buffering do cliente. Timing de stream no cliente
  é declarado honestamente indisponível neste modo, sem inventar número.
  Sem rebuild de C++: só o wiring de coleta.

### Propagação de status corrigida
  f5-p95.sh agora devolve o RC da captura (antes imprimia COLLECTOR_REJECTED e
  saía 0 através do cleanup) e f5-matched.sh ABORTA o bloco em qualquer
  resultado inválido, sem agendar novas execuções.

### Validação real com 512 tokens no braço split (o caso que falhava em 177)
  ARM=split P=93184 REQLEN=4096 NPREDICT=512 F5TOK=1 rc=0:
    tokens_predicted=512 server_decode_samples=511
    server_decode_p95_ms(nearest_rank)=29.057 median=28.717
    cross-check agregado predicted_per_token_ms=28.854 vs média por token ~28.7
    pico 11353 MiB free 929 MiB MemAvailable min 9456332 kB swap max 1099984 kB
    produção restaurada (health 200, reply 42)

### Nota de memória do braço base (registrada, semFloor alterado)
  XXL C=102400 tem pico 11707 MiB -> free 575 MiB, ABAIXO do piso fixo de
  614 MiB. O próprio braço base é mais apertado que o candidato (split 747 MiB
  free no mesmo caso). O piso de 614 MiB permanece intocado e esse fato entra
  na evidência do caso below_p.

## F5 — CASO below_p_46592: PAREADO COMPLETO, 1 WARMUP + 5 MEDIDOS POR BRAÇO (2026-08-08 03:13 UTC)

  Captura canônica não-streaming, f5_per_token do servidor, P=93184,
  C=131072, REQLEN=46592 (abaixo de P), NPREDICT=512, b/ub 512/256,
  seed 42, MESMO corpus e mesmo modelo real 27B nos dois braços,
  1 warmup descartado + 5 medidos alternando base->candidato, todos rc=0.

  arm    role      p95_ms   prefill_ms  pre_tps  dec_tps  free  evaluated
  xxl    warmup    33.593     60793.5   766.40  29.92   575     46592
  split  warmup    33.788     61363.7   759.28  29.82   747     46592
  xxl    measured1 33.744     61544.1   757.05  29.88   575     46592
  split  measured1 33.759     61823.7   753.63  29.84   747     46592
  xxl    measured2 33.718     60986.3   763.98  29.89   575     46592
  split  measured2 33.814     61762.6   754.37  29.80   747     46592
  xxl    measured3 33.711     61755.4   754.46  29.88   575     46592
  split  measured3 33.819     61835.3   753.49  29.82   747     46592
  xxl    measured4 33.745     61779.0   754.17  29.84   575     46592
  split  measured4 33.814     62214.4   748.89  29.81   747     46592
  xxl    measured5 33.811     62034.9   751.06  29.84   575     46592
  split  measured5 33.819     62250.5   748.46  29.78   747     46592

  Agregado (medianas dos 5 medidos):
    xxl   base     p95=33.744 ms  prefill=61755.4 ms (754.46 tok/s)  decode=29.88 tok/s
    split candidato p95=33.814 ms  prefill=61835.3 ms (753.49 tok/s)  decode=29.81 tok/s

  Portões do harness para below_p (limite 1.05 nos dois):
    prefill_ratio      = 1.0013  PASS
    decode_p95_ratio   = 1.0021  PASS
  Abaixo de P o split não pode REGREDIR: medido +0.13% em prefill e +0.21% em
  p95 de decode, ambas dentro da tolerância. Reprodutibilidade excelente: p95
  varia 33.711-33.819 nos dois braços.

  Memória no mesmo caso: base XXL C=102400 free=575 MiB (abaixo do piso fixo
  de 614 MiB), candidato split free=747 MiB. O split é o braço MENOS apertado
  emVRAM mesmo abaixo de P; o fato do base ficar abaixo do piso fica registrado
  e não altera o piso.

 仍然 pendente neste caso: KL/perplexidade (quality) e telemetria por estágio
  no formato do bundle. O que está medido aqui é performance pareada com p95 de
  decode real do servidor.

## F5 — ACIMA DE P: BRAÇO BASE XXXL POR CAMADA É ~5,6x MAIS LENTO (2026-08-08 04:15 UTC)

  Warmup do braço base (xxxl, C=131072, Radeon Q4 por camada, split DESLIGADO):
    prefill observado = 31,0-31,9 tok/s, prints do servidor até
      n_tokens=108544/130560 (progress=0.83) com t=3568,57 s
    o cliente foi desconectado e o servidor respondeu HTTP 500 ->
    warmup INVALIDO por desconexão do cliente (NÃO por falha do servidor nem de
    memória): peak=9691 MiB, free=2591 MiB, muito acima do piso de 614 MiB.
  Comparação direta no mesmo comprimento (130560):
    base XXXL por camada  ~31,5 tok/s   (aprox. 4150 s só de prefill)
    candidato split P=93184 ~175,4 tok/s  (741,7 s medidos 5x)
    => o candidato já é ~5,6x mais rápido em prefill acima de P, exatamente o
    que a feature promete (atenção local em vez de transbordo por camada).

  Custo do protocolo canônico acima de P: cada run do braço base leva ~1,1 h.
  Com 1 warmup + 5 medidos por braço, o bloco completo seria ~7 h só no base.
  therefore: vou executar primeiro um bloco REDUZIDO acima de P (1 warmup +
  1 medido por braço, candidato primeiro por ser rápido) e registrar a
  DEVIAÇÃO DE PROTOCOLO explicitamente; os 4 medidos restantes ficam pendentes
  e não serão inventados.

  Fragilidade observada e corrigida no método: matar o driver externo pode
  derrubar o filho por SIGHUP e matar a requisição em andamento. Para blocos
  longos o filho é lançado com setsid e NÃO é morto; o polling é feito lendo o
  log, sem tocar em processos.

## F5 — RETENÇÃO DO WARMUP SPLIT + FIX DE TRANSPORTE + HARNESS VERSIONADO (2026-08-08 04:55 UTC)

### Auditoria do warmup split 130560 que "falhou" com rc=126
  rc=126 NÃO é falha de modelo nem de servidor: o warmup TERMINOU e gravou o
  JSON final completo; o 126 veio depois, quando o Python voltou e o bash leu o
  próprio arquivo -- que havia sido editado IN PLACE durante a execução, o que
  desloca o offset de leitura do bash. Falha de orquestração, não de GPU.
  Auditoria do artefato imutável (timing.final.json, 411503 bytes):
    tokens_predicted=512          PASS
    tokens_evaluated=130560       PASS
    tokens_recorded=512           PASS
    decode_samples=511 (TTFT fora) PASS
    decode_iterations=512         PASS
    tokens_per_iteration todos 1  PASS
    truncated=false               PASS
    speculative_seen=false        PASS
    amostras finitas e > 0         PASS
  Resultado: prefill 130560 tokens em 736819,2 ms = 177,19 tok/s
             decode 512 tokens em 425576,2 ms = 1,2007 tok/s
             p95(nearest_rank)=871,506 ms, mediana=828,481 ms
             peak=11667 MiB, free=615 MiB (piso 614), MemAvailable min 8700692 kB
  VEREDICTO: RETAINED_VALID. Não repetido só por causa do erro de harness
  pós-resultado. Desvio de protocolo registrado: 1 warmup + 5 medidos canônico
  por braço; o warmup acima é válido e não medido; os 4 medidos restantes do
  candidato e todo o par do baseline continuam PENDENTES.

### Causa raiz do corte do baseline (não é GPU nem memória)
  common/common.h: timeout_write default 3600 s; tools/server/server-models.cpp:866
  usa base_params.timeout_write na leitura do upstream. Logo um run de baseline
  de ~1,1 h era cortado pelo PRÓPRIO SERVIDOR, e nenhum setsid ou timeout de
  cliente resolveria. common/arg.cpp:5022 --timeout N define timeout_read E
  timeout_write. Correção estreita aplicada nos DOIS braços:
    /tmp/opencode/f5-p95.sh: --timeout 10800 nas duas linhas de lançamento
    /tmp/opencode/f5-nontimer.py: urlopen timeout=12000
  Verificado no processo em execução:
    ./build-camp/bin/llama-server --timeout 10800 --models-preset ... --models-max 1 ...
  O baseline truncado a 83% do prompt NÃO é resultado pareado válido e não foi
  usado como tal; throughput de prefill do base segue sendo ~31,5 tok/s medido
  até 108544 tokens antes do corte.

### Harness versionado (regra aprendida)
  Editar um script EM EXECUÇÃO broke o bash reader e produziu rc=126 após um
  resultado válido. Agora: f5-p95-v2.sh e f5-matched-v2.sh congelados (md5
  5699d1a6343c000721a770b54592837c), o driver chama SÓ a versão congelada.
  Mudanças futuras vão para arquivo novo com versionamento + rename atômico.
  Fail-closed preservado: qualquer resultado inválido aborta o bloco.

## PRODUÇÃO — TPS REAL NO CAMINHO DE PRODUÇÃO, 130560 + 512 (2026-08-08 05:48 UTC)

  Binário de produção INTOCADO: build-optimized sha 3220fcd46c8364ca…
  Preset real: /home/hjotha/router-production.ini (o mesmo da unit), com
  desvios registrados e explicitados para uma requisição fresca:
    slot-save-auto=false, cache-prompt=false, cache-ram=0,
    slot-save-path=<dir novo e vazio>  -> sem reuso de cache de prompt/RAM/disco
  Todo o resto é produção literal: ctx 131072 (adaptive), b/ub 256, kvarn4,
  remote-attn vulkan:0 q4_0, flash-attn, gpu-layers 99, fit off, xhigh.

  Alocação efetiva escolhida pelo roteador para ESTE lançamento:
    perfil adaptativo inicial: mtp-short, ctx=24576, threshold=131072, backend=off
    perfil efetivo na requisição: xxxlong, ctx=131072, threshold=131072,
    backend=vulkan:0, prefill=remote
    => o roteador-prompts escolheu o transbordo Radeon (profile xxxlong) porque
       a requisição de 130560+512 exige 131072, e remote-attn-min-context=131072.
    Slots: n_slots=1, n_ctx_slot=24576 no load, comutado pelo perfil adaptativo.

  Resultado medido (agregados do servidor, uma requisição fresca):
    tokens_evaluated=130560  tokens_predicted=512
    cache_n=0  cache_source=none  cache_reason=no_common_prefix  (sem reuso)
    prompt eval = 1747144,30 ms / 130560 tokens = 13,38 ms/token = 74,73 tok/s
    eval        =   59394,26 ms /    512 tokens = 116,23 ms/token =  8,60 tok/s
    wall do cliente (não-streaming) = 1806985,78 ms
  Log do servidor confirma: "prompt eval time = 1747144.30 ms / 130560 tokens"
  e "eval time = 59394.26 ms / 512 tokens".

  Leitura honesta: em produção, um pedido de 131072 tokens real (27B IQ3_XXS)
  entrega ~74,7 tok/s de prefill e ~8,6 tok/s de decode. É esse o número que o
  usuário já sente; o split (candidato) medido no mesmo workload entrega
  177,19 tok/s de prefill (2,37x) e 1,20 tok/s de decode (7,2x MENOS).
  p95 de decode do candidato: 871,5 ms/token (warmup) e 868,95 ms (measured1).

  RETRATAÇÃO (06:40): a frase "preço conhecido de trocar overflow por atenção
  local" NÃO é suportada por evidência e fica removida. O que está medido é uma
  REGRESSÃO de decode de 7,2x contra o caminho de produção, e ela VIOLA o portão
  principal de decode. Causa candidata confirmada no código, ainda em correção:
  o gather do overflow Q4_0 materializava F32 (ggml_get_rows) e só depois era
  convertido para F16 (llama-graph.cpp, flash-attn guard), por camada e por
  token. A correção mínima (gather direto em F16, suportado por CUDA e Vulkan)
  foi implementada e medida; o packed view nativo continua sendo a correção
  estrutural. Nada de "tradeoff intrínseco" está em evidências até que o decode
  do candidato iguale ou supere a produção com evidência pareada.

  Nota de método (produção sem p95 por token): o binário de produção não tem a
  instrumentação F5 (que existe só em build de teste), então a produção é
  medida pelos agregados do servidor e isso está declarado no artefato
  (per_token_timing = "not available on the production binary"). Nada foi
  inventado.

  Duas falhas de orquestração no caminho, ambas corrigidas e sem tocar em GPU:
    v1 do script de produção apontava slot-save-path para diretório inexistente
    -> servidor recusou start -> tokenize 500 (não é falha de modelo);
    v2 passa o diretório explicitamente e exige argv>=4 no gerador de preset.

  Estado: produção restaurada e saudável; GPU livre; nenhum driver pendente
  (SIGSTOP 1265098 e retirement sem resume, conforme steer).

## CORREÇÃO DO GATHER DO OVERFLOW Q4 (2026-08-08 06:58 UTC)

### Causa confirmada no código (revisão + leitura própria)
  src/llama-graph.cpp:4746-4747 usava ggml_get_rows() sobre o armazenamento
  Q4_0 do overflow. Em ggml.c, ggml_get_rows() materializa SEMPRE F32 para fonte
  quantizada (ggml_get_rows -> ggml_get_rows_as(..., GGML_TYPE_F32)), e o caminho
  de flash attention convertia esse F32 para F16 logo depois (guard
  `if (k->type == GGML_TYPE_F32) k = ggml_cast(...)`). Isso é escrita F32 + leitura
  F32 do K e V de overflow por CAMADA e por TOKEN, para valores que já iam ser
  arredondados para F16.
  Suporte do F16 direto verificado nos dois backends que podem hospedar o cache:
    ggml-cuda/getrows.cu: get_rows_cuda() tem case GGML_TYPE_F16
    ggml-vulkan: ggml_vk_op_get_pipeline() -> dst F16 usa pipeline_get_rows[tipo]
                 e o pipeline Q4_0 é criado nessa tabela

### Correção mínima implementada + chave A/B
  gather direto em F16 (ggml_get_rows_as(..., GGML_TYPE_F16)), com toggle
  LLAMA_OVF_GATHER_F16=0 restaurando o caminho legado F32+cast, para comparar os
  dois caminhos NO MESMO binário e NA MESMA configuração. Build mínimo: só
  llama-graph.cpp recompilado, 0 arquivos CUDA/Vulkan.

### Paridade numérica full-vocab: PASS EXATO
  build-f4 (único build com o hook de captura), P=2048, C=4096, prompt 3000,
  32 tokens gerados, as duas pernas com captura de 248320 floats (vocabulário
  inteiro) no checkpoint n3000:
    KL(base||cand) = 0.00000000 nat/token   (limite 0.0010)
    ppl_ratio      = 1.000000               (limite 1.01)
    max|Δlogprob|  = 0.000e+00
    max|ΔLSE|      = 0.000e+00
    argmax igual
    VERDICT PASS
  Ou seja: o gather F16 direto é numericamente IDÊNTICO ao F32+cast (mesma
  dequantização, um único arredondamento). A mudança remove bytes, não precisão.

### Correção de método: o copy-profile não mede o que euprecisava
  GGML_BACKEND_COPY_PROFILE=1 (ggml-backend.cpp:1972) instrumenta apenas cópias
  ENTRE backends no agendador (input_backend -> split_backend). Saída vazia NÃO é
  prova de custo zero: o GET_ROWS do overflow é executado inteiramente no
  Vulkan0, sem fronteira de agendador, e por isso nunca aparece nem no F4SCHED
  nem no F4TRACE (que também só registram nós que cruzam backends). Atribuição de
  custo por estágio (GET_ROWS vs FA) exige timestamp de GPU
  (GGML_VK_PERF_LOGGER), que muda a execução e só serve para diagnóstico curto;
  o TPS final nunca é tirado de execução instrumentada.

### Medidas preliminares (não substituem o pareado canônico)
  130560 + 512, sem instrumentação de temporização:
    antes (caminho F32+cast, agregado do servidor): prefill 167,70 tok/s,
      decode 3,05 tok/s em 32 tokens, pico 11665 MiB, free 617 MiB
    pequeno contexto (P=2048, prompt 3000, 32 tokens): 16,18 tok/s (F16) vs
      14,24 tok/s (F32) -> +13,6%, n=1 por perna, NÃO é evidência final
  Pareado representativo em andamento (P=93184, prompt 130560, 32 tokens, mesma
  configuração, F5TOK=1, TRACE=0 nas duas pernas).
  Correção estrutural ainda pendente e prioritária: packed view/indexed Q4
  nativo no lugar do gather, preservando ringmap/ordem/mask/LSE.

## RESULTADO PAREADO DA CORREÇÃO DO GATHER (2026-08-08 08:48 UTC)

  Configuração real: P=93184, C=131072, prompt 130560 (~37K overflow),
  32 tokens de decode, MESMO binário, MESMO preset, MESMO request, sem tracing
  nas duas pernas (TRACE=0) e com instrumentação de tempo por token do servidor
  (F5TOK=1). Única diferença: LLAMA_OVF_GATHER_F16.

    perna  prefill_tps  decode_tps  decode_p95_ms  mediana_ms  amostras  pico/free
    F32    176.65       1.1314      944.201        869.186     31        11665/617
    F16    166.76       3.0967      327.108        310.006     31        11665/617

  Efeito real, mesma configuração:
    decodeThroughput 3,0967 / 1,1314 = 2,74x MAIS RÁPIDO
    p95 de decode    944,201 -> 327,108 ms = 2,89x MENOR
    prefill          176,65 -> 166,76 tok/s = 5,6% MAIS LENTO (custo real)
    memória idêntica: pico 11665 MiB, free 617 MiB nas duas pernas

  Interpretação honesta: a regressão de decode de 7,2x contra a produção
  (produção 8,60 tok/s) cai para ~2,8x com esta correção mínima
  (3,0967 tok/s no mesmo ponto de trabalho), mas AINDA NÃO FECHA o portão
  principal de decode. A correção estrutural pendente é o packed view/indexed
  Q4 nativo, que elimina o gather por completo em vez de torná-lo mais barato.
  O custo de prefill de ~5,6% precisa entrar na decisão: o portão de prefill
  abaixo de P exige <=1,05x contra a base, então esse ôtem será medido no
  protocolo canônico, não assumido.

  Nota: comparações com execuções anteriores de 512 tokens (1,2007 tok/s de
  decode, p95 868,95 ms, caminho F32) NÃO são 像-como-como-como por causa da
  mudança no número de tokens; a perna de 512 tokens com a correção está em
  curso para a comparação direta.

## CONFIRMAÇÃO EM 512 TOKENS COM A CORREÇÃO (2026-08-08 09:12 UTC)

  Mesma configuração (P=93184, C=131072, prompt 130560, 512 tokens, sem tracing):
    caminho F32+cast (referência anterior): decode 1,2007 tok/s, p95 868,954 ms
    caminho F16 direto (correção)          : decode 3,0829 tok/s, p95 346,100 ms
  Ganho pareado de 2,57x em throughput de decode e 2,51x no p95, com memória
  idêntica (pico 11667 MiB, free 615 MiB) e paridade numérica exata já provada.
 _decode ainda 2,8x abaixo da produção (8,60 tok/s): por isso a PRÓXIMA etapa
  é estrutural (packed KV nativo, mesmo formato do tier anterior), não mais
  matrizes de ablação F16/F32. Nenhuma promoção de produção antes de verificar.

## PACED VIEW: ENGATE PROVADO, REUSO CORRIGIDO, E VIEW REJEITADA PELO PORTÃO NUMÉRICO (2026-08-08 10:30 UTC)

### 1) Fast path existia mas não engatava (dois bugs deoflag)
  A primeira versão exigia a variável LLAMA_OVF_PACKED_VIEW PRESENTE, então
  ambiente sem a variável mantinha o gather e o caminho parecia inerte
  (96 traces, todas path=indexed_gather, zero packed_view). Corrigido para
  default ligado no teste; o trace passou de LLAMA_LOG_INFO para stderr porque o
  callback de log do servidor filtra INFO do processo do modelo (o que escondia
  as linhas e fez o fast path parecer morto).

### 2) Bug de CORREÇÃO confirmado e corrigido: reuso com view obsoleta
  llm_graph_input_mem_hybrid::can_reuse (llama-graph.cpp:1847-1892) DUPLICA parte
  das checagens de atenção e NUNCA chama inp_attn->can_reuse, então a validação do
  mapa físico ficava BYPASSED no grafo híbrido do Qwen. Medido: 7 grafos para 256
  passos de decode, n_view congelado em 1025 enquanto o vivo crescia (953→1207):
  254 passos de decode leriam um extent obsoleto e perderiam KV novo.
  Correção: validação fatorada em split_packed_map_still_valid() e chamada nos
  DOIS caminhos de reuso (atenção própria e híbrido), sempre com o contexto de
  memória da ATENÇÃO (nunca o híbrido).
  Reengajamento após a correção: 4160 traces packed_view, 0 fallback, 255 recusas
  de reuso (uma por token) e n_view acompanhando o vivo exatamente
  (953,954,...,1207 = 952 prompt + 255 gerados).

### 3) Portão numérico REJEITA a view (full-vocab, 3 prefixos determinísticos)
  Mesmos requests byte-idênticos nas duas pernas, P=2048, corpus tokenizado,
  capturas em 3000/3072/3200 (capturas esperadas são obrigatórias; ausência falha):
    prefixo  KL(nat/token)  ppl_ratio  max|dLSE|  argmax igual
    3000     2,093538e-02    25,248     1,607     sim
    3072     2,503160e-03     0,671     0,793     sim
    3200     9,269820e-02    28,550     0,482     NAO (argmax virou)
    mean KL 0,0387 (limite 0,0010) -> VERDICT REJECT
  Token gerado diverge em 3200 ('\n' no gather vs '\n\n' na view).

### 4) Causa raiz: RETRATADA, investigations abertas
  RETRATADO: rotation_k/rotation_v NÃO é rotação de linha física. É uma base
  Hadamard no ESPAÇO DE FEATURES aplicada aos valores armazenados
  (llama-kv-cache-kvarn.cpp: rotation::hadamard(largura de feature)), e tanto o
  gather quanto a view leem a MESMA base armazenada -- logo isso não explica a
  divergência. Nenhuma correção de rotação deve ser implementada.
  Também medido: o mapeamento de linhas é DIRETO. O trace F5IDX mostra os índices
  reais do gather como uma sequência contígua e ascendente começando na primeira
  célula viva (ex.: 2048,2049,2050 ... 2303,2302), ou seja exatamente as linhas que
  uma view cobriria.
  SUSPEITO RESTANTE (não provado): o caminho de FA quantizado no Vulkan
  (k_type_eff == Q4_0 com neq1 == 1, onde a desquantização proativa de
  ggml-vulkan só age com neq1 >= 64) não reproduz o caminho denso F16 de forma
  exata, e o erro cresce quando o comprimento vivo sai do alinhamento de bloco do
  FA: KL 2,5e-03 em kv_len 1024 (alinhado), 2,1e-02 em 952, 9,3e-02 em 1152.
  Próximo passo obrigatório: evidência por camada de Q/K/V/O/LSE entre os dois
  caminhos, antes de qualquer novo código. Caminho nativo segue DESLIGADO.

### 5) Estado do código e formato
  LLAMA_OVF_PACKED_VIEW agora é DEFAULT OFF, com a justificativa da rejeição
  medida escrita na própria fonte; só liga com opt-in explícito depois de
  aplicar a rotação. Caminho validado por padrão = gather F16 direto.
  Formato preservado, sem conversão: local KVarN4 + remoto Q4_0 (o mesmo do tier
  anterior, confirmado no log de produção: mainXXXLongKV KVarN4, remote K/V
  Q4_0 auto). A correção validada (gather direto em F16) remove a expansão
  transitória F32->F16 e mantém Q4_0 no armazenamento.

### 6) O que continua válido
  - Correção do gather F16: paridade full-vocab EXATA (KL 0, ppl 1,000000,
    dlogp 0, dLSE 0) e ganho pareado de 2,57x em decode (1,2007 -> 3,0829 tok/s),
    p95 868,954 -> 346,100 ms, memória idêntica.
  - Memória P=93184: 5 medidos + 1 warmup, free 615 MiB, vizinho +256 FAIL.
  - below_p_46592 pareado completo: prefill_ratio 1,0013 e p95_ratio 1,0021
    (limite 1,05) PASS.
  - Produção real 130560+512: prefill 74,73 tok/s, decode 8,60 tok/s.
  Portão de decode segue em aberto: 3,08 tok/s do candidato contra 8,60 da
  produção. Sem promoção de produção.

## PACKED VIEW: ALINHAMENTO DESCARTADO POR MEDIÇÃO, BLOCKER RESTANTE (2026-08-08 10:50 UTC)

  Teste controlado, 5 prefixos determinísticos do MESMO corpus, mesmo binário,
  mesmos requests, Capturas full-vocab obrigatórias (ausência falha o teste):
    prefixo  overflow_vivo  %256  KL(nat/tok)  ppl_ratio  argmax  max_abs_diff
    3000     952            não   2,093538e-02  25,248     sim    5,97
    3072     1024           sim   2,503160e-03   0,671     sim    7,78
    3200     1152           não   9,269820e-02  28,550     NÃO    5,19
    3328     1280           sim   9,545753e-02   0,375     NÃO    6,09
    3584     1536           sim   6,692164e-06   7,057     sim    4,97
  Diferença em 100% do vocabulário (248320/248320) em todos os comprimentos.

  HIPÓTESE DE ALINHAMENTO DESCARTADA: comprimentos alinhados a 256 dão tanto
  erro grande (3328: KL 9,5e-2) quanto quase zero (3584: KL 6,7e-6); não há
  relação monótona com o alinhamento. A hipótese anterior (kl menor em 1024) era
  coincidência de amostra.

  MAPEAMENTO DE LINHAS DESCARTADO: o trace F5IDX mostra os índices reais do
  gather como sequência contígua e ascendente a partir da primeira célula viva
  (2048,2049,2050... 2303,2302) — exatamente as linhas que a view cobriria.

  ROTAÇÃO DESCARTADA (retração já registrada): rotation_k/rotation_v é base
  Hadamard no espaço de features; as duas pernas leem a mesma base armazenada.

  BLOCKER EM ABERTO, sem nomeação: a view crua do armazenamento Q4 e o tensor
  F32/F16 produzido pelo gather divergem em TODO o vocabulário. Falta a evidência
  por camada de Q/K/V/O/LSE (primeira camada) para decidir entre layout/caminho
  de FA quantizado e qualquer transformação de linhas aplicada entre o
  armazenamento bruto e a visão de atenção. Até essa evidência, caminho nativo
  DESLIGADO (default OFF na fonte, com a justificativa escrita).

  BLOCKER ADICIONAL registrado: a perna de referência com o preset de PRODUÇÃO
  não roda em build-f4 (único build com hook de captura): o modelo falha ao
  carregar com router-production.ini nesse build (log
  docs/occupancy-evidence/boundary2-prodref.log). O cruzamento contra o caminho
  remoto de produção, portanto, ainda não foi feito.

  Nenhum benchmark longo foi iniciado com o caminho incorreto. Produção intacta.

## PACKED VIEW: CAUSA DE MÁSCARA CORRIGIDA, PARIDADE AINDA REPROVA (2026-08-08 13:35 UTC)

  Correção aplicada em src/llama-graph.cpp (build_attn_position_split):
    - o shader Vulkan lê a máscara de overflow com passo = extensão K/V
      (vulkan-shaders/flash_attn_base.glsl:213 `m_stride = KV`;
       flash_attn.comp:222 `data_m[(i*Br + r)*m_stride + ...]`);
    - a view compacta levava n_view vivo (ex. 640) enquanto a máscara contígua
      mantinha a largura preenchida n_ovf_used (ex. 768), então query>0 lia a
      linha errada de máscara;
    - a correção recorta e materializa máscara contígua com exatamente o extent
      K/V (campo n_ovf_rows) e passa esse mesmo valor como n_kv_max no
      build_attn_mha. Nenhum formato KV é convertido; local KVarN4 e overflow
      Q4_0 permanecem.
  Engajamento confirmado nos logs: `attn_inp_kq_mask_overflow_occ` aparece com
  ne0=640 quando a view tem 640 linhas (antes: 768). Traço F5OVF continua ativo.

  Regressão numérica medida no gate determinístico (build-f4, P=2048, corpus
  tokenizado fixo, 3000/3072/3200/3328/3584, packed vs gather, mesmos requests):
    prefixo   KL(nat/tok)      ppl_ratio   argmax   dLSE
    3000      1,226418e+01     36,28       NÃO      10,21
    3072      3,541144e-04      0,332      sim       0,711
    3200      1,181259e+01     55,43       NÃO      11,38
    3328      6,332664e-03      0,604      sim       0,278
    3584      5,997271e-02      0,00055    sim      11,57
    mean KL 4,82868658 (limite 0,001) -> VERDICT REJECT
  Capturas obrigatórias presentes em 3000/3072/3200/3328/3584 nos dois braços;
  engajamento de packed_view confirmado (160 traços) e produção restaurada.

  Conclusão honesta: a incompatibilidade de passo da máscara era real e foi
  corrigida, mas NÃO era a (única) causa da divergência. A view compacta Q4_0
  continua reprovando o gate full-vocab; portanto `LLAMA_OVF_PACKED_VIEW`
  permanece DEFAULT OFF e nenhum benchmark longo com ela foi feito.
  Próximo passo concreto: isolar a rota de FA quantizada vs F16 com instrumentação
  por camada (Q/K/V/O/LSE) — o diagnóstico por CAST Q4_0->F16 no grafo falhou
  porque o backend Vulkan não tem pipeline CPY para q4_0 (abort em
  ggml-vulkan.cpp:9816), então o teste precisa de um caminho suportado
  (get_rows_as F16 sobre a view, ou captura por camada).
  Produção: ativa, health 200; sem promoção.

## PACKED VIEW: GATE NUMÉRICO PASSA COM LEITURA SUPORTADA (2026-08-08 14:05 UTC)

  Diagnóstico por caminho suportado (não o CAST Q4_0->F16, que aborta no Vulkan
  por não existir pipeline CPY para q4_0 — ggml-vulkan.cpp:9816):
    - a view compacta é lida de volta como F16 por GET_ROWS com lista de índices
      identidade no próprio grafo (input novo self_ovf_view_rows, preenchido em
      set_input; bounded pelo extent vivo, sem expandir o cache inteiro);
    - resultado: as LINHAS/BASE da view estão corretas. O que divergia era o
      caminho de FA QUANTIZADO do Vulkan consumindo Q4_0 cru
      (use_dequant_kv desligado: o pipeline_dequant_transpose só existe para Q8_0,
      ggml-vulkan.cpp:6044), com KL médio 4,83 e flips de argmax.

  Gate determinístico re-executado (build-f4, P=2048, corpus tokenizado fixo,
  prefixos 3000/3072/3200/3328/3584, mesmos requests, capturas obrigatórias
  presentes nos dois braços, engajamento packed_view=160, 2 recusas de reuso
  híbrido como esperado):
    prefixo   KL(nat/tok)   ppl_ratio   argmax_same
    3000      4,264468e-04   0,6247      sim
    3072      2,673957e-04   0,8154      sim
    3200      4,945459e-05   0,9181      sim
    3328      3,577270e-03   0,5842      sim
    3584      1,337782e-05   0,0456      sim
    mean KL 0,00086679 (limite 0,0010) -> VERDICT PASS
    worst ppl_ratio 0,918142 (limite 1,01) -> PASS
  Artefatos: docs/occupancy-evidence/boundary2-fixfinalgather.log,
  boundary2-fixfinalpacked.log; capturas em
  f4-spool/f5-campaign/boundary3-fixfinalgather e boundary3-fixfinalpacked.

  Estado do código (sem promoção):
    - LLAMA_OVF_PACKED_VIEW continua DEFAULT OFF (exige opt-in explícito);
    - com a view ligada, o padrão agora é a leitura F16 suportada e gated;
    - o caminho Q4_0 cru na FA do Vulkan ficou atrás de
      LLAMA_OVF_PACKED_Q4_FA=1, ainda NÃO corrigido/gateado (é o alvo do ganho
      de performance real, pois evita qualquer dequantização);
    - formatos preservados: KVarN4 local, Q4_0 remoto; nada convertido no
      armazenamento.
  Produção: ativa, health 200. Sem benchmark longo, sem promoção.

## PACKED Q4: TESTE MÍNIMO DETERMINÍSTICO DO FA VULKAN (2026-10-08 12:54 UTC)

  Reproducer: /tmp/opencode/fa_q4_probe.cpp (compilado contra build-f4, só o op
  FLASH_ATTN_EXT, sem modelo). Compara Q4_0 K/V vs os MESMOS valores como F16 no
  device Vulkan0 (AMD RADV PHOENIX), escalas do modelo (1/sqrt(head_dim)).

  Resultados (kv x nq; max_abs Q4-vs-F16):
    nq>=2 (2,8,32,63,64,128,512), kv 640..1536, com e sem view de storage maior:
      max_abs <= 1,5e-08 -> idêntico (o kernel Q4 e o F16 concordam)
    nq=1 com q_heads=64 / kv_heads=8 (GQA ratio 8, forma real do modelo):
      max_abs ~1,5e-08 -> idêntico
    nq=1 com q_heads=8 / kv_heads=8 (GQA ratio 1, configuração NÃO usada pelo
      modelo): max_abs 8e-4..1,1e-3 (diferença do kernel single-row, não é o
      caminho do modelo)

  Conclusão: o kernel Q4_0 do Vulkan NÃO é intrinsecamente divergente para as
  formas do modelo (incluindo views de storage maior e GQA). Logo o residual do
  caminho packed no modelo não vem do kernel Q4 em si.

  Gates no modelo (build-f4, prefixos fixos, capturas obrigatórias presentes):
    - gather vs gather (duas corridas independentes): KL exatamente 0 -> o
      pipeline é determinístico; o residual abaixo é específico do caminho packed;
    - packed com leitura F16 suportada (default atual com view ligada):
      mean KL 8,6679e-04 (PASSA no gate frouxo 1e-3), MAS prefixo 3328 KL
      3,5773e-03 e max dLSE 2,56 -> NÃO atende o gate canônico (média <=1e-4,
      p99 <=1e-3); não é conclusão numérica;
    - packed com Q4_0 cru na FA (LLAMA_OVF_PACKED_Q4_FA=1): mean KL 4,8287
      -> continua reprovado, atrás de opt-in.

  Hipótese atual do residual packed-vs-gather: diferença de EXTENTO de máscara/
  K/V (n_view vivo vs n_ovf_used preenchido) alterando a ordem de soma e o LSE
  do split-K, não as linhas/basis (já provadas corretas). Próximo passo concreto:
  variante de EXTENTO PADDED (view com n_ovf_used linhas, cauda mascarada -inf,
  mesmo extent do gather) + captura por camada de O/LSE no primeiro ponto de
  divergência. Nenhum benchmark longo, nenhuma promoção, formatos KVarN4/Q4_0
  preservados. Produção ativa, health 200.

## TESTE ISOLADO PERMANENTE + DELTA VALID-vs-PADDED (2026-10-08 13:10 UTC)

  Preservado em tests/test-fa-q4-view.cpp (registrado em tests/CMakeLists.txt,
  target test-fa-q4-view, skip limpo sem Vulkan0; diagnóstico, não aceitação).
  Roda com GPU exclusiva e restaura produção ao final (rc=0; produção health 200).

  Matriz: kv_valid {640,952,1024,1152,1536}, pad {0,128}, nq {1,2,64},
  head_dim 128, 64 q-heads / 8 kv-heads (GQA 8), escala 1/sqrt(128), O e LSE.

  Resultados (agregado):
    - Q4_0 vs F16 no MESMO extent (válido ou padded): O max 0..6e-8,
      LSE max 0..9,5e-7 -> equivalentes (asserts do teste, PASS).
    - CROSS valid-vs-padded (diagnóstico):
        maioria ~0..3e-8; MAS kv_valid=952/1152/1536 com nq=1:
        O max 1,35e-4..1,68e-4
      Isto é a diferença causada SÓ pela troca de extent (split_k=3 com
      split_kv=320/384/512 no extent vivo vs split_k=1 no padded), exatamente a
      classe de magnitude do residual do caminho packed no modelo
      (mean KL 8,7e-4, outlier 3328 KL 3,6e-3), com rows/basis provados corretos
      e pipeline determinístico (gather vs gather = 0).

  Conclusão técnica: para atender o gate estrito, o caminho packed deve usar o
  MESMO extent do gather (n_ovf_used preenchido, cauda inativa mascarada -inf),
  em vez do n_view vivo, OU excluir os dois caminhos de configurações de split_k
  distintas para o mesmo caso. O teste permanente mede esse delta e serve de
  referência para a correção.

  Estado: LLAMA_OVF_PACKED_VIEW default OFF; caminho Q4_0 cru atrás de
  LLAMA_OVF_PACKED_Q4_FA=1; formatos KVarN4/Q4_0 preservados; sem benchmark de
  performance; sem promoção. Produção ativa, health 200.

## EXTENSÃO PADDED IDÊNTICA: GATE ESTRITO PASSA (KL 0) (2026-10-08 14:48 UTC)

  Implementado: o caminho packed passou a usar o MESMO extent do gather
  (n_ovf_used preenchido; cauda inativa mascarada -inf; fallback para gather se
  o view padded exceder o storage). Motivo medido em
  tests/test-fa-q4-view.cpp: a troca valid-vs-padded muda o split_k e move O em
  ~1,4e-4, a classe de magnitude do residual anterior.

  Gate determinístico (build-f4, prefixos mínimos 3000/3072/3200/3328/3584,
  capturas obrigatórias, engajamento packed_view=160, produção restaurada):
    packed view + leitura F16 suportada + extent padded, vs gather:
      mean KL = 0,00000000e+00   (gate <=1e-4)  PASS
      max p99(|dlogp|) = 0,00000000e+00 (gate <=1e-3) PASS
      dLSE = 0, argmax igual em todos os checkpoints -> BIT-IDÊNTICO
    packed view + Q4_0 cru na FA (LLAMA_OVF_PACKED_Q4_FA=1), mesmo extent:
      mean KL 1,8912e-03, max p99 5,1158 -> FAIL (permanece opt-in)
  Artefatos: boundary3-padgather, boundary3-padpacked, boundary3-padq4
  (/home/hjotha/f4-spool/f5-campaign/); comparador estrito
  /tmp/opencode/f5-strict.py.

  Teste isolado permanente atualizado (tests/test-fa-q4-view.cpp):
    - layout folded do modelo (2D -> reshape -> permute, como build_attn_mha)
      com extent padded: Q4_0 vs F16 bit-idênticos (O e LSE = 0), nq 1/2/64,
      kv 640..1536;
    - neutralidade de máscara com valores de cauda diferentes (view lê lixo,
      gather lê row0 duplicado) confirmada: O <= 6e-8, LSE <= 9,5e-7;
    - cross valid-vs-padded segue reportado como diagnóstico (até ~1,7e-4).
    rc=0; produção ativa/health 200 após cada execução.

  Estado: LLAMA_OVF_PACKED_VIEW default OFF; Q4 cru opt-in; formatos KVarN4
  local e Q4_0 remoto preservados; sem benchmark de performance; sem promoção.

## ABLAÇÃO POR CAMADA DO Q4 CRU NA INTEGRAÇÃO (2026-10-08 15:23 UTC)

  Instrumentação: LLAMA_OVF_PACKED_Q4_FA_IL=<lista> liga o Q4_0 cru só nas
  camadas listadas; as demais usam a leitura F16. Comparação contra o gather
  (prefixo único 3000, P=2048, extensão padded idêntica, f5-strict.py):

    camadas cruas                          mean KL
    3..31 (8 primeiras)                    0.00000000e+00  PASS
    35..63 (8 últimas)                     0.00000000e+00  PASS
    12 camadas (3..47)                     0.00000000e+00  PASS
    14 camadas (3..55)                     0.00000000e+00  PASS
    15 camadas (3..59, sem 63)             0.00000000e+00  PASS
    15 camadas (7..63, sem 3)              0.00000000e+00  PASS
    16 camadas (todas)                     6.08730421e-04  FAIL

  Leitura: o Q4_0 cru é bit-exato por camada na integração; a divergência só
  aparece com as 16 camadas simultâneas — um efeito de limiar/configuração
  (recurso, especialização de kernel ou poda/estrutura de grafo), não de
  matemática por camada. Hipóteses a testar em seguida, sem assumir causa:
  poda do input de identidade quando todas as camadas ficam cruas mudando a
  estrutura/reuso do grafo; seleção de prealloc/pipeline do Vulkan com 16
  dispatches quantizados; ou interação de buffer entre o merge LSE e o FA.
  Próximo passo concreto: capturar a PRIMEIRA divergência O/LSE por camada na
  execução integrada com as 16 camadas cruas (ex.: marcação tipo F4_TRACE dos
  tensores fa2/lse2 e leitura pós-compute) antes de qualquer correção.

  Preservado: correção de extensão padded (bit-idêntica, KL 0) no caminho
  suportado; Q4 cru atrás de opt-in; KVarN4/Q4_0 inalterados; sem benchmark de
  performance; sem promoção; produção ativa/health 200.

## PRIMEIRA DIVERGÊNCIA INTEGRADA CAPTURADA: CAMADA 7 (2026-10-08 16:57 UTC)

  Ferramenta: dump de nós já existente (F4_NODE_DUMP + auto-tag novo no servidor,
  F4_NODE_DUMP_AUTOTAG) e nomes únicos por camada para os tensores do overflow
  (fa2 -> "f5fa2-<il>", lse2 -> "f5lse2-<il>", env F5_LAYER_CAPTURE=1).
  Duas pernas no MESMO build, prefixo 3000, extensão padded idêntica:
    raw  = Q4_0 cru em todas as 16 camadas de overflow (IL=3..63)
    f16  = gather F16 (referência)
  Hashes por camada (NODE ... hash=):
    il=3   fa2 same  lse2 same
    il=7   fa2 DIFF  lse2 DIFF   <-- PRIMEIRA DIVERGÊNCIA
    il=11..63 DIFF em ambas (propagação)
  il=3 e il=7 têm geometria idêntica de overflow nos logs
  (n_ovf_used=256/512/896/1024, n_view=256/512/896/952, row0=2048, mesmo ktype),
  então a hipótese de layout/máscara por camada não explica sozinha.
  Combinado com a ablação (15 camadas cruas = KL 0 exato; 16 = 6,09e-4), o quadro
  aponta para interação de configuração/reuso entre os 16 dispatches quantizados,
  não para erro aritmético por camada.

  Estado do código: Q4 cru atrás de LLAMA_OVF_PACKED_Q4_FA / _IL; view e leitura
  F16 suportada com extensão padded seguem bit-idênticas (KL 0); default OFF;
  formatos KVarN4/Q4_0 preservados; sem benchmark de performance; sem promoção;
  produção ativa/health 200 após cada corrida.
  Próximo passo concreto: dump com valores (F4_NODE_DUMP_FULL=f5lse2) nas duas
  pernas para medir a magnitude em il=7, e dump de k_overflow/v_overflow/máscara
  por camada para verificar se a divergência já entra pelos insumos (captura de
  primeira camada divergente a montante) antes de qualquer correção.

## Q4 CRU INTEGRADO: LOCALIZAÇÃO PRECISA E BLOQUEIO ATUAL (2026-10-08 17:39 UTC)

  Comandos/ferramentas: F5_LAYER_CAPTURE=1 nomeia por camada fa2/lse2/k/v/máscara;
  F4_NODE_DUMP+F4_NODE_DUMP_AUTOTAG (servidor) grava NODE ... hash; comparação
  raw (Q4 cru, LLAMA_OVF_PACKED_Q4_FA_IL=3..63) vs f16 (gather) no MESMO build,
  prefixo 3000, P=2048, extensão padded idêntica.

  Resultados:
    - 16/16 cru: mean KL 6,08730421e-04 em 3000 (determinístico: 3 corridas iguais);
      média 1,89119140e-03 sobre 3000..3584.
    - 15/16 cru (qualquer subconjunto testado): KL 0 exato.
    - LLAMA_GRAPH_REUSE_DISABLE=1: KL idêntico (6,087e-04) -> reuso NÃO é a causa.
    - LLAMA_OVF_ZERO_TAIL=1 (zerar cauda inativa): KL idêntico -> cauda não é a causa.
    - Dumps por camada: il=3 fa2/lse2 idênticos em TODAS as 5 evals; il=7 diverge
      a partir da eval 14 (últimas duas), com máscara idêntica (hash e shapes iguais
      entre pernas) e shapes iguais (hsk=256, nh=4, nhq=24, kv=1024 padded, nq 256/56).
    - Teste isolado (tests/test-fa-q4-view.cpp) com as MESMAS shapes (hsk=256/nh=4/
      nhq=24, kv 952->pad1024, nq 56/256, folded): bit-idêntico (O e LSE = 0).
      Variantes testadas no isolado, todas neutras/bit-idênticas: layout folded,
      extensão padded, cauda com valores diferentes, cauda NaN/Inf (prova que NaN
      envenena a FA crua se chegar ao operador: O 3,4e38/LSE inf), linha de query
      totalmente mascarada (FA_Q4_DEAD_ROW), LSE anexado, nq 1/2/56/64/128/256/512.

  Leitura: a divergência integrada é determinística, começa em il=7 eval 14/15,
  com máscara e shapes idênticas e operador exato nas mesmas shapes; portanto a
  diferença não está na aritmética do operador nem na máscara. Permanecem como
  hipóteses a testar (sem assumir): proveniência dos valores de K/V efetivamente
  lidos naquelas evals (view vs gather), aliasing/lifetime de buffers do scheduler,
  ou interação com o merge LSE a montante/jusante. Próximo passo concreto:
  dump comparável do tensor F16 lido pela perna crua (f5k16 construído só para
  dump) e dos nós de merge por camada, para decidir se a divergência entra pelos
  insumos ou nasce no merge.

  Estado: LLAMA_OVF_PACKED_VIEW default OFF; Q4 cru opt-in (env _FA / _IL);
  KVarN4 local e Q4_0 remoto preservados; sem benchmark de performance; sem
  promoção; produção ativa/health 200. Arquivos modificados nesta rodada:
  src/llama-graph.cpp (nomes por camada, zero-tail opt-out, traço de geometria),
  src/llama-context.cpp (auto-tag do dump), tests/test-fa-q4-view.cpp (variantes
  NaN/dead-row/formas do modelo), tests/CMakeLists.txt.

## BLOQUEIO DEMONSTRADO: Q4 CRU 16/16 É INTERAÇÃO DE BUFFER/ESCALONAMENTO (2026-10-08 17:55 UTC)

  Checklist (requisito / evidência / estado / ação):
  - Q4 cru 16/16 == referência F16 | sem debug: KL 6,08730421e-04 (3000), determinístico
    em 3 corridas | BLOQUEADO | identificar buffer aliased no escalonador
  - insumos efetivos idênticos | f5k16(raw, F16 read da view) hash == f5k(f16 gather) em
    todas as evals, il=3 e il=7; f5mask idêntico | OK | manter como controle
  - operador exato nas shapes reais | tests/test-fa-q4-view.cpp hsk=256/nh=4/nhq=24,
    kv 952->pad1024, nq 56/256, folded: O/LSE = 0; variantes NaN/dead-row/folded/padded | OK
  - reuso de grafo | LLAMA_GRAPH_REUSE_DISABLE=1: KL idêntico | DESCARTADO
  - cauda inativa | LLAMA_OVF_ZERO_TAIL=1: KL idêntico | DESCARTADO
  - perturbação de grafo | leitura F16 completa de K por camada (expandida só p/ dump)
    -> KL 0,00000000e+00 bit-idêntico; leitura de 1 linha -> KL 6,087e-04 | EVIDÊNCIA:
    interação de lifetime/alocação de buffer, não aritmética
  - primeiro ponto | il=3 idêntico; il=7 diverge nas evals 14/15 (kv=1024 padded,
    nq=256/56), propagando depois | LOCALIZADO
  Próxima ação concreta: diff de F4RESERVE_TOP entre a corrida que falha e a que passa
  para achar o buffer reutilizado (view Q4 do K vs writer), e então corrigir a ordem/
  lifetime (ex.: dependência explícita antes da FA) sem pagar leitura extra.
  Arquivos modificados: src/llama-graph.cpp (f5k16/f5v16 diag, F5_DBG_K_ONLY,
  F5_DBG_ONE_ROW, zero-tail, nomes por camada, traço de geometria), src/llama-context.cpp
  (auto-tag), tests/test-fa-q4-view.cpp, tests/CMakeLists.txt.
  Comandos-chave: run-fa-q4-view.sh; f5-boundary-v3.sh com LLAMA_OVF_PACKED_Q4_FA_IL=3..63;
  f5-strict.py; F4_NODE_DUMP/F4_NODE_DUMP_AUTOTAG/F5_LAYER_CAPTURE.
  Estado: LLAMA_OVF_PACKED_VIEW OFF; Q4 cru opt-in; formatos preservados; sem promoção;
  produção ativa/health 200.

## Q4 CRU 16/16: MECANISMO REFINADO DO BLOQUEIO (2026-10-08 17:57 UTC)

  Experiências controladas (prefixo 3000, P=2048, 16 camadas cruas, extensão padded,
  referência gather; comparador f5-strict):
    - sem perturbação:                       KL 6,08730421e-04  (determinístico)
    - GET_ROWS F16 completo de K por camada: KL 0,00000000e+00  (bit-idêntico)
    - GET_ROWS de 1 linha por camada:        KL 6,08730421e-04  (não corrige)
    - expandir as views Q4 como saídas:      KL 6,08730421e-04  (não corrige)
    - desabilitar reuso de grafo:            KL 6,08730421e-04  (não é reuso)
    - zerar cauda inativa:                   KL 6,08730421e-04  (não é a cauda)
  Leituras: (a) a view Q4 lida pela FA tem os MESMOS valores efetivos que o gather
  (f5k16/f5v16 hash idênticos ao f5k/f5v da perna F16, todas as evals, il=3 e il=7);
  (b) a máscara é idêntica (hash/shapes); (c) o operador isolado é exato nas shapes
  reais (hsk=256/nh=4/nhq=24, kv 952->pad1024, nq 56/256, folded, LSE incluso).
  Hipótese mecanicista sustentada pelas perturbações: a FA quantizada lê a view do
  storage persistente sem uma aresta de dependência efetiva sobre o write do K/V
  deste ubatch (cpy_k), e o GET_ROWS completo, por ser um op real que lê o storage,
  força a ordenação; uma leitura de 1 linha e a expansão de view não bastam.
  Próxima ação concreta (sem pagar o read completo): construir a dependência
  explicitamente para o caminho cru (ex.: ler a view via op leve que force ordem,
  ou registrar a aresta store->FA no escalonador) e revalidar 16/16 + isolado.
  Estado: LLAMA_OVF_PACKED_VIEW OFF; Q4 cru opt-in; formatos preservados; produção
  ativa/health 200. Experimento F5_OVF_EXPAND_VIEWS no código (default off).

## CAUSA CORRIGIDA (ARESTA DE DEPENDÊNCIA) + TRIAGEM DE PERFORMANCE (2026-10-09 01:39 UTC)

  Correção experimental causal: materializar a view Q4 de K com ggml_cont (cópia
  q4->q4) antes da FA quantizada cria o op real que lê o storage (aresta de
  dependência/barrier) e dá à FA um tensor com buffer próprio.
  Evidência discriminante (prefixo 3000..3584, 16 camadas cruas, referência gather):
    - sem perturbação:                    KL 6,08730421e-04  (falha)
    - GET_ROWS F16 completo de K:         KL 0               (corrige)
    - GET_ROWS de 1 linha:                KL 6,087e-04       (não corrige)
    - expandir views:                     KL 6,087e-04       (não corrige)
    - **ggml_cont K-only (q4->q4)**:      KL 0 nos 5 checkpoints (3000/3072/3200/3328/3584)
                                           p99 0, dLSE 0, argmax igual -> gate estrito PASS
    - ggml_cont K+V:                      KL 0 (também)
  Logo a hipótese está confirmada: era ordenação/dependência store->FA, não
  aritmética. A variante K-only é a de menor custo (cópia de K apenas).

  Triagem de performance (build-camp, P=93184, C=131072, REQLEN=130560, 512 tokens,
  F5TOK=1, pico 11667 MiB / free 615 MiB, produção restaurada health 200):
    candidato q4 cru + cont K-only: prefill 172,09 t/s, decode 2,4955 t/s, p95 412,301 ms
    candidato F16 gather (referência): prefill 166,88 t/s, decode 3,0829 t/s, p95 346,100 ms
    produção real: prefill 74,73 t/s, decode 8,6035 t/s
  Ou seja: a correção resolve a correção numérica, mas o caminho nativo com cópia
  de K ainda fica ABAIXO do candidato F16 gather, e ambos muito abaixo da produção.
  Próximo passo (antes de campanha longa): profile do decode do candidato para
  identificar o gargalo medido (FA quantizada vs cópia K vs merge LSE vs gather),
  usando o logger de perf do Vulkan/contadores em corrida curta.
  Estado: LLAMA_OVF_PACKED_VIEW OFF; Q4 cru opt-in; F5_OVF_CONT_KV/_K_ONLY
  experimentais (default off); formatos preservados; sem promoção.

## Q4 INTEGRADO CORRIGIDO + PERF (STABLE-256 + CONT-K) (2026-10-09 02:28 UTC)

  Mudança (dentro do caminho packed, default OFF):
    - extensão baked = arredondar n_view para 256, limitada ao range; máscara
      recortada/materializada nesse extent; cauda inativa zerada (finita);
    - guarda de reuso passa a aceitar crescimento do vivo até o extent baked
      (só row0 mudou/fragmentou invalida) -> grafo reutilizado entre tokens;
    - coluna K materializada com ggml_cont q4->q4 (aresta de dependência real
      store->FA, sem GET_ROWS no caminho quente).
  Resultados (build-f4/build-camp, prefixos 3000..3584, estrito f5-strict):
    - sq4 (q4+contK+stable) vs sf16 (F16 read+stable): KL 0, p99 0  (bit-idêntico)
    - sf16 vs gather (referência):                      KL 0, p99 0  (bit-idêntico)
  Profile curto (36K, decode mediano): gather 88,3 ms | q4cont antigo 226,7 ms
    | q4+stable 83,2 ms (13 recusas de reuso no prefill; decode reutiliza).
  Triagem no contexto real (P=93184, C=131072, 130560+512, F5TOK=1, free 615 MiB):
    - q4+contK+stable: prefill 168,41 t/s, decode 5,7613 t/s, p95 176,431 ms
    - F16 gather (antigo): decode 3,0829 t/s
    - produção real:       decode 8,6035 t/s
  Profile no 130K (decode-only Vulkan, 16 camadas): FLASH_ATTN_EXT 3,798 ms/layer
    -> ~60,8 ms/token no overflow Vulkan; o restante (~110 ms/token) está no lado
    CUDA/local + overhead (produção total ~116 ms/token).
  Próximo alvo de otimização medido: lado CUDA/local e merge (P=93184 local KVarN
  para 16 camadas), não a FA de overflow. Correção mantida; campanha longa ainda
  não iniciada por não superar a produção.
  Estado: LLAMA_OVF_PACKED_VIEW OFF; Q4 cru/cont/stable opt-in; formatos
  preservados; produção restaurada health 200 reply '42'.

## ATRIBUIÇÃO DO DECODE: CPU vs GPU vs VULKAN vs CUDA (2026-10-09 03:08 UTC)

  Instrumentação mínima flag-gated (LLAMA_F5_STAGE_TIMING=1 em process_ubatch,
  F5_ROUTE_TRACE=1 no builder split, GGML_VK_PERF_LOGGER=1) em janela GPU única,
  produção checada/restaurada em cada corrida (health 200, reply '42').

  Breakdown por token (P=93184, C=131072, REQLEN=130560, NPREDICT=8):
    wall mediano 176,2 ms/token
    host graph_compute 162,7 ms (mediana; inclui despacho + espera)
    Vulkan busy 69,8 (FLASH_ATTN_EXT 60,8 + CONT 9,1 + resto ~0)
    => CPU/staging fora do compute ~10 ms; CUDA+sincronizações ~90 ms
  Baseline gather no mesmo ponto: wall 328,4; Vulkan 184,2 (FA 99,6 + GET_ROWS 84,2);
  residual ~144. 36K q4stable: wall 88,3; Vulkan 9,5; residual 78,8
  (mostra que o residual escala com P: local KVarN/denso no CUDA).
  Rotas KVarN no decode (F5ROUTE): il=3..63 nq=1 native=1 nativereq=1 domain=1
  (ROTATED), nrotmax=16 -> decodifica no caminho NATIVO KVarN com rotação e LSE
  exportado, NÃO materializado. Rebuilds: 13 recusas de reuso no prefill,
  decode reutiliza (stable-256). CONT/q4: 16 cont K por token (9,1ms).

  Conclusão medida: o maior custo removível já foi removido (GET_ROWS 84ms -> 0
  com cont-K). O restante é dominado por (a) FA de overflow no Vulkan 60,8 ms/
  token (16 camadas x ~38K chaves) e (b) ~90 ms CUDA (matmuls densos + KVarN
  local nativo sobre ~93K). Não há intervenção pequena demonstrada para esses
  dois itens; atingir a produção (8,60 t/s = 116 ms/token) exige otimização de
  kernel/rota (reduzir camadas de overflow ou acelerar KVarN decode Q=1) — decisão
  de plano, não micro-ajuste. Correção stable/contK preservada e bit-idêntica.
  Estado: defaults OFF; formatos preservados; sem promoção; produção ativa.

## ATRIBUIÇÃO FECHADA DO DECODE 130K (SPLIT TIMING DIAGNÓSTICO) (2026-10-09 03:28 UTC)

  Instrumentação mínima flag-gated: GGML_SCHED_SPLIT_TIMING=1 (um synchronize
  por SPLIT, não por camada; só existe quando ligada; medida final de aceitação
  NÃO a utiliza). Corrida única: P=93184, C=131072, 130560+512, NPREDICT=8,
  q4+contK+stable. Produção checada/restaurada (active, health 200, reply '42').

  Janela de 7 tokens de decode (693 splits, soma 1201,7 ms):
    CUDA0  split wall = 59,9 ms/token  (matmuls densos + KVarN local Q=1 + despacho CUDA)
    Vulkan0 split wall = 111,7 ms/token
    CPU = 7 splits (staging)
  Cruzando com o perf logger do Vulkan (GPU busy real, sem sync):
    Vulkan GPU = 69,8 ms/token (FLASH_ATTN_EXT 60,8 + CONT 9,1)
  => overhead/espera/despacho do lado Vulkan ~41,9 ms/token distribuído em
     ~48 splits Vulkan por token (16 camadas x ~3 travessias CUDA<->Vulkan);
     CPU/staging ~10 ms. Wall medido 176,0 ms fecha: 69,8 GPU Vulkan +
     ~41,9 host/sync Vulkan + 59,9 CUDA + ~4 residual de janela.
  Rotas preservadas: KVarN decode Q=1 nativo/rotacionado com LSE; 16 camadas;
  stable-256+contK bit-idêntico (KL 0). Nenhuma mudança de desenho, placement,
  gates ou F6.

  Conclusão da etapa delimitada: não existe intervenção pequena demonstrada.
  - overflow FA 60,8 ms/token: kernel ou menos camadas (fora do escopo permitido);
  - overhead Vulkan ~42 ms/token: exige fusão/reorganização de splits no
    scheduler (estimativa: dias de trabalho + validação de reuso/lifetime),
    não um flag;
  - CUDA 59,9 ms/token inclui matmuls densos intrínsecos (piso do modelo).
  BLOQUEIO CONCRETO registrado; baseline preservado; sem campanha longa.
  Nota de gate (correção do steer): aceite F5 é prefill/TTFT >=10% melhor nos
  pontos e decode p95 <=5% de regressão, com p95 de baseline equivalente medido
  (1/8,6035 é média e não serve); p95 baseline ainda não medido — registrar antes
  de qualquer comparação de aceite.
  Estado: LLAMA_OVF_PACKED_VIEW OFF; stable/contK opt-in; formatos KVarN4/Q4_0;
  produção ativa/saudável; sem push/merge/promoção.

## TRIAGEM A/B p95 NO PONTO 130560+512 (DIAGNÓSTICO OFF) (2026-10-09 04:35 UTC)

  Condições idênticas: mesmo modelo (27B IQ3_XXS), mesmo prompt (corpus 130560),
  mesmo sampling (temp 0, seed 42), C=131072, harness f5-nontimer/f5_per_token
  (p95 server-side), sem instrumentação de diagnóstico. Baseline = preset de
  produção (router-production.ini; cache-prompt=false, cache-ram=0, auto-save off)
  no MESMO binário de teste (build-camp) e harness; candidato = split P=93184 +
  stable-256 + cont-K (Q4 cru). Warmup descartado em cada braço antes do medido.

    braço      amostras  mediana      p95         decode t/s  prefill t/s
    baseline   511       115,149 ms   116,947 ms  8,6747      10,30 (cache quente)
    candidato  511       168,723 ms   173,728 ms  5,8661      170,48 (cache frio)

  GATE p95 (<=1.05 x baseline): 173,728 / 116,947 = 1,4855 -> **FALHA CLARA**
  (regressão de ~48,6% no p95 de decode). Prefill: candidato 170,48 t/s vs
  produção real fresca 74,73 t/s (medição anterior do binário de produção) ->
  >=10% melhor, passaria no gate de prefill/TTFT.
  Nota: os ~42 ms de overhead por split medidos antes eram estimativa de
  diagnóstico (sync por split altera execução) e não foram usados como aceite.
  Conclusão: bloqueio de desempenho confirmado no gate p95; NÃO promover.
  Opções (aguardando decisão do usuário; nenhuma iniciada):
    (a) reduzir número de camadas em overflow / ajustar placement (fora do
        escopo atual, exige revalidação de memória/capacidade);
    (b) otimizar kernel da FA de overflow (60,8 ms GPU/token) ou KVarN CUDA Q=1;
    (c) fusão de splits do scheduler (estimativa: dias; precisa revalidar
        lifetime/reuso).
  Estado: produção restaurada e verificada (active, health 200, inferência
  real); LLAMA_OVF_PACKED_VIEW OFF; stable/contK opt-in; formatos preservados;
  sem push/merge/reboot/promoção.

## KERNEL FA Q4 vs F16: HIPÓTESE DE GANHO FALSIFICADA (2026-10-09 05:05 UTC)

  Microbench isolado (Vulkan0, formas reais hsk=hsv=256, 4 KV heads, 24 Q heads,
  GQA 6, LSE anexado, folded, 100 iters, mediana/min/p95; produção restaurada):
    q4_0 kv= 4352 nq=  1: use grep do log (ver abaixo)
    q4_0 kv=37888 nq=  1: median=1,610 ms  min=1,362  p95=3,902
    f16  kv=37888 nq=  1: median=1,560 ms  min=1,520  p95=1,633
    q4_0 kv=37888 nq=256: median=163,282 ms   f16=166,934 ms
    q4_0 kv= 4352 nq=256: median= 13,299 ms   f16=  9,702 ms
  Leitura: no ponto de decode (nq=1) e no tamanho real do overflow (37888) o kernel
  q4_0 é EQUIVALENTE ao F16 (1,61 vs 1,56 ms). Não há ineficiência de dtype a
  corrigir na FA de overflow. A hipótese "otimizar kernel q4 da FA" está
  FALSIFICADA por medição.
  Nos shapes do modelo, 16 camadas de overflow custariam ~25,6 ms/token de FA
  pura (1,6 ms x 16); o medido no modelo é ~60,8 ms de GPU Vulkan/token porque
  inclui CONT q4 (0,57 ms/camada), reduce/LSE e merge; e o restante do gap p95
  (173,7 vs 116,9 ms baseline) é estrutural: o split faz 2 FAs + merge por camada
  (local 93K + overflow 38K) contra 1 FA por camada da produção, em 16 camadas.
  Não existe ganho pequeno de kernel que feche o limite 1,05x; as opções são
  arquiteturais (reduzir camadas em overflow / eliminar merge) — fora do escopo
  autorizado desta etapa. Bloqueio registrado; baseline preservado; produção
  restaurada (active, health 200, reply '42'). Artefato: /tmp/opencode/fa_q4_bench.cpp.

## KERNEL FA: BASELINE ISOLADO, CONTENÇÃO E BLOQUEIO QUANTIFICADO (2026-10-09 05:23 UTC)

  Bench isolado (Vulkan0, formas reais, LSE, 100 iters; /tmp/opencode/fa_q4_bench.cpp):
    kv=37888 nq=1: q4_0 1,61-1,81 ms | f16 1,56-1,73 ms  (paridade; sem ganho de dtype)
    kv= 4352 nq=1: q4_0 0,54 ms | f16 0,34 ms
    kv=37888 nq=256: q4 163 ms | f16 167 ms
  No modelo 130K, o FA de decode é uniforme: p50 3,796 / p95 3,832 / max 3,846 ms por
  chamada, 16 camadas x 7 tokens (nenhum outlier de compilação/config inicial).
  Gap isolado-vs-modelo (1,8 -> 3,8 ms/layer) testado com contenção de RAM do sistema
  (8 workers de memcpy, GPU exclusiva, produção restaurada):
    idle: q4 1,81 ms | f16 1,73 ms
    com stress: q4 2,39 ms (+32%) | f16 3,44 ms (+100%)
  => contenção de banda explica ~+0,6 ms/layer, não os 2,0 ms; o restante está na
  submissão/ordenação Vulkan (CONT/LSE reduce/merge na mesma fila) e não num kernel
  de dtype isolável.

  Matemática do bloqueio p95 (baseline 116,947; candidato 173,728; limite 122,794):
  para fechar seriam precisos ~51 ms/token. Ganhos máximos plausíveis dentro do
  escopo atual: remover CONT (9,1 ms) + eventual overhead de submissão do Fa
  (se aproximar do isolado, ~32 ms) + merge — somados ~41 ms, ainda acima do limite;
  CUDA local KVarN (~20-25 ms) otimizado 2x daria ~10 ms. Logo NENHUM ajuste de
  kernel pequeno/médio fecha o gate mantendo as 16 camadas e o desenho split.
  Opções arquiteturais (fora do escopo autorizado, exigem decisão): (a) reduzir
  camadas em overflow/placement; (b) eliminar o merge/segundo FA por camada;
  (c) F6 overlap. Hipótese de kernel dtype ENCERRADA por medição; hipótese de
  contenção parcial. Baseline e correção stable/contK preservados; produção
  restaurada (active, health 200, reply '42'); sem push/merge/reboot/promoção.

## CORREÇÕES DE REGISTRO (2026-10-09 08:33 UTC)

  - Retratação de excessos: "kernel q4 FALSIFICADO"/"sem headroom" era forte demais.
    Estado correto: em IDLE e nas formas testadas q4_0 ≈ F16; sob CPU ocupada o FA
    desacelera ~2,1x (replay 1,863 -> 3,969 ms; in-model 3,796 ms), o que sugere
    interferência de plataforma (potência/clock/banda UMA) mas NÃO separa esses
    fatores nem prova impossibilidade de otimização de kernel.
  - Retratação de gate: prefill 10,30 t/s do baseline quente NÃO serve como gate
    frio; a referência fria de produção medida antes é 74,73 t/s (binário de
    produção); repetir em condições frias equivalentes quando for aceite.
  - Escopo novo autorizado (08:19 UTC): atribuir CPU por thread/callsite durante o
    decode e reduzir atividade de CPU desnecessária, mantendo 16 camadas,
    placement, formatos e gates.

## CPU NO DECODE: ATRIBUIÇÃO, REDUÇÃO OPT-IN E LIMITE DO ESCOPO (2026-10-09 10:43 UTC)

  Atribuição por thread (sampler /proc, decode, candidato stable/contK):
    - thread principal do modelo: avg 92,8%% de um core no 130K (0,158 core-s/token);
    - callstack (6 amostras, gdb root): server_queue::yield_to_queue ->
      server_context_impl::decode -> llama_decode -> process_ubatch ->
      graph_compute -> ggml_backend_sched_graph_compute_async ->
      ggml_backend_vk_synchronize -> ggml_vk_wait_for_fence: SPIN em
      getFenceStatus/YIELD durante a execução do grafo no Vulkan.
  Redução opt-in implementada: GGML_VK_FENCE_BLOCKING=1 (waitForFences em vez de
  spin; default OFF, rollback = desligar o env).
  A/B mesmo contexto 32K/512 tokens (diagnósticos OFF, warmup em cada braço):
    spin : hot thread 89,9%% ; 0,0686 core-s/token ; p95 80,32 ms
    block: hot thread 73,5%% ; 0,0562 core-s/token ; p95 79,40 ms (-18%% CPU)
  A/B end-to-end 130K/512 (intercalado, warmup+medido, mesmo build/config):
    spin : p95 174,076 ms ; median 169,221 ; decode 5,8464 t/s
    block: p95 175,283 ms ; median 169,809 ; decode 5,8245 t/s
    -> CPU menor NÃO move o p95; limite 122,794 ms continua não atendido.

  Hipóteses testadas e FALSIFICADAS para explicar FA in-model 3,8 ms vs isolado
  1,8 ms (kv=37888, nq=1, mesmo op): layout/offset/CONT (replay 1,83-1,96),
  contenção CPU do spin (A/B e2e acima), carga CUDA compute (1,87->1,95),
  tráfego PCIe H2D/D2H (1,61->1,62), clock iGPU (800 MHz idle e load; bench a
  800 MHz faz 1,83 ms). Fica não reproduzido por fatores externos controláveis;
  candidato remanescente é a interação da submissão/sincronização por split do
  scheduler com o driver Vulkan, fora do escopo autorizado (fusão ampla de
  scheduler proibida) e das 16 camadas/placement (também fora).
  BLOQUEIO REAL do escopo "reduzir CPU desnecessária": a redução existe (18%%),
  é segura e opt-in, mas NÃO fecha o gate de decode p95. Decisão necessária:
  (a) autorizar investigação de submissão/sincronização do scheduler
  (cross-queue/eventos) sem fundir operações; (b) arquitetura/placement; (c)
  aceitar não fechar o gate p95 e prosseguir F5 apenas com os gates que passam
  (prefill/TTFT), explicitando a exceção. Nada iniciado; baseline preservado;
  produção restaurada (active, health 200).

## FECHAMENTO DA ETAPA CPU/SUBMISSÃO (2026-10-09 11:01 UTC)

  Correções de redação (conforme revisão):
  - "falsificação" => os CONTROLES ESPECÍFICOS testados (layout/offset/CONT,
    contenção do spin, carga CUDA compute, tráfego PCIe, clock iGPU) NÃO
    reproduziram o gap in-model de 3,8 ms; isso NÃO exclui universalmente
    fatores externos.
  - a redução ~18% de CPU foi medida em 32K (0,0686 -> 0,0562 core-s/token);
    não foi medida em 130K e não deve ser extrapolada.

  Rastreamento delimitado de submissão/sincronização (32K/512, só trace):
  - GGML_VK_SYNC_LOGGER=1: ZERO chamadas de ggml_vk_sync_buffers durante a
    execução real; nenhuma sincronização por nó redundante observada;
  - GGML_SCHED_SPLIT_TIMING=1: 52.004 splits no run; a espera principal é o
    fence do Vulkan (já atribuído por callstack), que corresponde a execução
    GPU real (bloqueante não mudou o total);
  - knobs existentes A/B (mesmo build/config, warmup+medido, 512 tokens):
      default            p95 80,39 ms  median 76,31  decode 12,986 t/s
      MAX_NODES_PER_SUBMIT=1000  p95 80,27  median 76,11  13,024
      SERIALIZE_SUBMISSIONS=1    p95 80,60  median 76,86  12,881
    -> dentro do ruído; nenhuma redução pequena sustentada.

  Conclusão da etapa: no domínio submissão/sincronização NÃO há redução pequena
  sustentada por evidência. Causa observada: a CPU da thread principal é espera
  de fence do Vulkan (execução real); a variante bloqueante reduz CPU sem mover
  o p95 (A/B 130K: 174,076 vs 175,283 ms; limite 122,794 continua não atendido).
  Mudança concreta que exige decisão: reestruturação de submissão/sincronização
  cross-queue do scheduler (fora do escopo autorizado) ou arquitetura/placement.
  Nenhuma rodada adicional iniciada; baseline e correções preservados; produção
  restaurada (active, health 200).

## CURVA DE PREFILL FRODO (16 camadas, P=93184, C=131072, build-camp, instrumentacao OFF, cache-prompt off, fresh server, n_predict=1) - 2026-10-09 16:57 UTC

  Autorizacao 14:57/14:58 (eGPU banda limitada; Radeon so overflow; cache de slot;
  comecar em 110k; cold prefill so quando necessario).

  Linkage eGPU verificado (nvidia-smi -q / lspci -tv):
    PCIe Generation Current = 1 (Host Max = 1, Device Max = 4), Link Width 4x
    => link de ~1 GB/s; AD104 em 03.1-[04-62]-00.0-[05-06]-01.0-[06] (eGPU).
  Nao atribui causa automaticamente a banda; medicao e por/token/armaco.

  Curva pareada (mesmo binario/build/preset/batch/ubatch, cold):
    tokens    cand(P=93184)      ctrl(sem split, all-Radeon Q4_0)   ratio
    8192      834.52 t/s          273.28 t/s                         3.05x
    36864     793.95 t/s           82.38 t/s                         9.64x
    110000    344.00 t/s           nao concluiu (>45 min)             -
    130560    170.50 t/s           -                                  -

  RESPOSTA A PERGUNTA DO USUARIO: NAO, o prefill nao cai desde o comeco no
  placement por ocupacao. Abaixo de P nao ha overflow algum e o candidato e
  3,05x (8192) e 9,64x (36864) MAIS RAPIDO que o controle sem split. Nao ha
  overhead fixo do split em contexto curto; o oposto.

  A queda comeca exatamente onde o overflow comeca (tokens > P=93184) e e
  SUPER-LINEAR na fracao de overflow:
    36864 (overflow 0)      793.95 t/s  1.2595 ms/tok
    110000 (overflow 16816, 15.3%)  344.00 t/s  2.9070 ms/tok
    130560 (overflow 37376, 28.6%)  170.50 t/s  5.8653 ms/tok
  Custo incremental: +1.6475 ms/tok (15.3%) e +4.6058 ms/tok (28.6%);
  1,87x mais overflow => 2,80x mais custo (super-linear).

  CONTROLE DE MESMO COMPRIMENTO (isolando overflow sem mudar prompt):
    36864 com P=32768 (overflow 4096 = 11.1%): 572,6-610,7 t/s (mediana ~590)
    36864 com P=93184 (overflow 0):              793.95 t/s
  => 11,1% de overflow custa ~26% da vazao de prefill em 36864, mesmo build.

 VALIDACAO DOS NUMEROS DE CAMPANHA: 130560 medido de novo com F5TOK=0 e
  cache-prompt=false deu 170,495 t/s vs 170,48 t/s registrado na campanha com
  F5TOK=1/cache-prompt=true => diferenca 0,01%. Overhead da instrumentacao
  server-side no prefill medido em 110000 pareado: 344,00 (OFF) vs 346,86 (ON)
  = -0,8% (dentro do ruido). Os prefills frios de 110k/130k ja medidos sao
  validos como cold equivalente.

  Custo do 42 ms de sync antigo: NAO usado como budget causal (perturbativo).
  Nenhuma fusao ampla, F6, mudanca de arquitetura/placement ou execucao
  permanente. Producao restaurada: active, health 200, inferencia real '42'.

## RECONCILIACAO EFETIVA + CORRECOES (pedido de continuacao)

  1) Binario/modelo/prompt (hashes reais, nao relato):
  - build-camp/bin/llama-server sha256
    7b8634fc2692be0bbb28aa1038bc0f6217734e0eba8b142f4decf493205a1de0
  - modelo Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf
    (10442827776 bytes, mtime 2026-09-25) sha256
    e23eb251493b8a89f7d0217590b593e696ac9bacaeac8605541e477533dc5838
    (confere com o prefixo `e23eb251...` registrado na linha 33).
  - prompt 110000: prompt_sha256
    15014c0ad32571b85bd73685cdecca722871f7aeecd52a988e6c6a19dcb9c87c
    (tokens do corpus, seed 42, mesmo vetor para cand/ctrl).
  - Manifesto completo em
    /home/hjotha/f4-spool/f5-campaign/prefill-curve/reconciliation.json
    + effective-body-<tag>.json (corpo exato enviado por ponto).

  2) Divergencia req salvo x corpo enviado (corrigida, nao reeditada):
  - `req-<L>.json` SALVO e template do f5-build-req.py: cache_prompt=true,
    n_predict=32. Nao foi o corpo da requisicao.
  - corpo ENVIADO pelo prefill-rate.py (POST /completion): stream=false,
    n_predict=1, cache_prompt=false, temperature=0.0; mesmo vetor de prompt.
    Os t/s de prefill medidos correspondem a esse corpo frio.
  - Config efetiva do servidor: preset gerado sha256
    e9331967aea37bc6fdf282e5a3a5a9e581eab7dc1c46e800219a741a49a6bd38
    (cache-prompt=false, slot-save-auto=false), env split P=93184 +
    flags packed/cont-K (cand), F5TOK 0/1 conforme o ponto, batch/ubatch
    512/256, model path confirmado no server log (loading model ...gguf).

  3) Correcao "snapshots prontos": FALSO. Os 8 diretorios slots-* da curva
  estao todos VAZIOS (0 arquivos) porque a curva rodou com slot-save-auto=
  false e sem save explicito; `--slot-save-path` sozinho nao grava nada.
  `make-snapshot.sh` existe mas NUNCA executou: nenhum snapshot foi criado,
  nenhum save/restore foi provado. Retiro a afirmacao anterior.

  4) Enfraquecimento das conclusoes da curva (conforme a revisao):
  - os pontos 110000/130560 NAO provam queda "exatamente em P" nem
    "super-linearidade" como mecanismo; sao 3 medicoes pos-P consistentes
    com custo crescente no overflow, sem decomposicao por componente.
  - PCIe Gen1 x4 e TETO NOMINAL do link (~1 GB/s teorico), nao banda
    efetiva medida; nenhuma conclusao causal usa esse numero.
  - prefill de prefixo restaurado NAO e prefill frio; so vale como medida
    de reuso/sufixo, com restore validado e custo separado.

  5) Bloqueio concreto desta etapa (sem rodada nova): provar save/restore
  compativel por braco/config/ocupacao exige parar o 8090 de producao
  (GPU sem memoria livre: 11866/12282 MiB residentes) e criar 1 snapshot
  frio por braco indispensavel. Nao ha mecanismo aprovado de controle do
  servico nesta sessao para parar/restaurar sem manusear credencial; nao
  vou improvisar. Decisao necessaria: meio aprovado para stop/start do
  llama-server-root (ou janela executada pelo usuario); ai executo, nesta
  ordem: (a) snapshot cand 110000 com fingerprint + save explicito via
  POST /slots/0?action=save; (b) restore em processo novo + suf fixo,
  tokens reaproveitados/timings de restore separados e proxima-token
  identico com seed/temperatura fixas; (c) so entao diagnostico de sufixo.

## ACHADO OFFLINE: staging CUDA0->Vulkan0 por token (trace existente, sem GPU nova)

  Metodo (reproduzivel, sem execucao): profile-sched32.log (F4_ALLOC_TRACE=1,
  32K/512, P=32768) segmentado por grafo via reset do contador split=
  (528 grafos; ultimos 511 = janela decode, 1 ocorrencia por tensor/grafo).
  Somente staging exigido pelo scheduler (owner_backend != exec_backend,
  bytes>0); sem somar tempos sobrepostos, sem fence-spin, sem 42ms antigos.

  Bytes por direcao, por grafo decode (token):
  - CUDA0->Vulkan0: 4759,44 MiB / 511 = 9,5 MB/token (9537,5 KiB)
  - CPU->CUDA0:     340,8 KiB/token; CPU->Vulkan0: 140,0 KiB/token
  - Vulkan0->CUDA0: 3,0 KiB/token (retorno quase nulo)
  Top: cache_kvarn_k/v_tail_lN (view), 16 camadas (l3..l63 passo 4),
  258,0 KiB/camada/lado/token (128,7 MiB x 32 ao longo dos 511);
  mais Qcur/Kcur/Vcur/fa_out por camada (frequencia exatamente 1,00/grafo).
  Geometria do tail constante em todos os grafos: ne0=256, bytes=264192,
  offs=0 (janela fixa realocada por grafo, nao delta incremental).

  Leitura honesta e limitada: bytes/direcao/frequencia MEDIDOS (9,5 MB/token
  CUDA->Vulkan, 1,00/grafo, geometria offs=0 constante). Redundancia de
  CONTEUDO NAO provada: offs=0 constante contradiz a hipotese de janela
  deslizante (overlap 99,8% assumido antes sem prova). Aliasing (view vs
  copia integral por grafo) e lifetime entre grafos NAO verificados em
  codigo ainda; sem isso, nenhuma afirmacao de redundancia, nenhum ganho
  afirmado. Verificacao em codigo pendente abaixo.
  Hipotese mantida (nao conclusao): SE o conteudo for majoritariamente
  repetido entre tokens, staging persistente + delta incremental no lado
  Vulkan seria o ataque; provar antes, A/B antes, sem ganho afirmado.

## PROVA SAVE/RESTORE AUTO (slot 110000, mesmo braco/config) - 2026-10-09 19:35 UTC

  Meio de controle: AGENTS.md do home (lido integralmente) - credencial e
  comando aprovado em historico sao contexto ativo inclusive em operacao
  privilegiada; senha do SSH vale para sudo no mesmo host; janela de
  parada do 8090 se livre + restore health+inferencia autorizada.
  Sem helper sem-segredo existente (~/bin sem helpers llama); segredo nao
  aparece em respostas/logs (askpass 0700 removido apos uso).

  AUTO-SAVE (processo A, cold 110000, 327,26 t/s):
  `auto-save: persisted unified snapshot reason=prompt tokens=110000
  bytes=3891362176 save_ms=6224.188` em slots/auto-1ee08d95...-110000.bin.
  AUTO-RESTORE (processo B novo, mesmo preset/env/dir):
  `indexed 429 prefix boundaries` no startup;
  `auto-restore: reused 110000 tokens from disk (in-memory match was 0),
  bytes=3891362176 restore_ms=9085.993`.
  Prova funcional: sonda warm com mesmo prefixo => prompt_n=0
  (0 tokens reprocessados) e proximo token identico ao cold: [60922] "=q"
  (temp 0/seed 42 nos dois). Integridade: loader verifica checksum +
  igualdade de tokens + fingerprint de modelo/layout (restore falharia
  em vez de corromper). Custos separados: save 6224 ms / restore 9086 ms.

  FALHA INSTRUTIVA do save/restore EXPLICITO (mesma janela): POST
  /slots/0?action=save no ROUTER => 400, porque slots vivem no FILHO
  (modelo na porta filha), nao no router. Fase C refeita (rev2) mirando a
  porta filha (parse de "spawning server instance ... on port N"), com
  gate de prontidao do filho (health 200; 503 "Loading model" nao aceito),
  kill de grupo por padrao de cmdline + verificacao, e pre-voo que ABORTA
  sem tocar em nada se 8090 ocupado (conns>0 ou gpu>=5%).
  Rev1 tambem deixou filhos orfaos (sintoma: VRAM presa, restore com
  reply vazio); recuperado: orfaos mortos, producao health 200 e
  inferencia real '42' reconfirmada. Admito ainda: lancei a rev1 com
  pre-voo fraco (ss=5/gpu 95% visiveis) - possivel interrupcao de request
  alheio; o pre-voo estrito da rev2 existe por causa disso.
  Rev2 abortou corretamente na primeira tentativa (conns=2/gpu 100,
  producao ocupada, nada alterado). Snapshot ja existe: nenhuma nova
  prefill fria necessaria para a Fase C (auto-restore popula o slot).

## FASE C: EXPLICITO + CONTINUACAO 64 (janela msg854, 20:22 UTC)

  Producao parada com janela autorizada (msg854 dispensa restores
  intermediarios; restore definitivo so ao concluir o plano). KEEP_DOWN=1.

  C1 (processo novo): auto-restore CARREGOU 110000 tokens + sidecar de logits,
  mas a sonda usava n_probs=5 sem post_sampling_probs => restore-continue
  rejeitou os logits ("saved logits unavailable or incompatible"),
  limpou o estado (memory_cleared) e fez FULL RE-PREFILL dos 110000 tokens
  (341,8 s). CORRECAO: C1 nao foi "zero prefill"; custou um reprefill total.
  O save EXPLICITO posterior e valido (estado avaliado de verdade):
  n_saved=109952 (trim alinhado -48), n_written=3894952143 bytes,
  save_ms=12087.703.
  C2 (outro processo novo): restore EXPLICITO: n_restored=109952
  (=n_saved), n_read=3894952143, restore_ms=8510.74 (repetido 8090.49).
  Sonda warm 64: prompt_n=48 (exatos 48 da cauda aparada reprocessados),
  decode 8.3191 t/s.

  CONTINUACAO (temp 0/seed 42, n_probs=5+post_sampling):
  - cold-64 (1 prefill frio de referencia, 330.70 t/s) x warm-64 explicito:
    identicos por 56 tokens, divergencia na posicao 56 (214200 vs 174140).
    SEM ROTULO DE BENIGNA: identidade de tokens nao e gate de qualidade;
    equivalencia para fins de qualidade exige KL/perplexidade (nao medidos).
  - cold-64 x warm-64 auto (110000 integrais, prompt_n=0): identicos nos
    64/64 tokens, top-1 id+logprob iguais ([60922], -0.03036627).
  Leitura honesta e limitada: a igualdade de 64 tokens valida a
  continuacao gananciosa DESTE prompt, nao estado/logits completos
  (distribuicao total nao comparada). O formato de snapshot e fiel no
  caminho auto (== cold em 64); o caminho explicito apara a cauda e
  reprocessa, com divergencia numerica prefill-chunk vs decode
  incremental apos dezenas de tokens (mecanismo plausivel, nao provado
  como unica causa). Sidecar de logits: presente no auto (fast-path serviu o
  token do restore com prompt_n=0); ausente no explicito pos-trim
  (carimbo invalida, conforme o codigo).
  n_probs>0 sem post_sampling_probs QUEBRA o fast-path por desenho
  (server-context.cpp:9122) e forca reprefill total: armadilha registrada,
  sondas usam post_sampling_probs=true.

## TIMELINE DE SUFIXO 64 EM 110K (warm, estado restaurado) - 20:57 UTC

  Metodo: restore explicito (109952 + 48 reprocessados) + 64 decode com
  GGML_VK_PERF_LOGGER=1 (summary no destructor; shutdown gracioso exigido:
  kill -9 perde a evidencia) + GGML_SCHED_SPLIT_TIMING=1. Segmentacao dos
  6276 splits por ordem: chunk 0 (95 splits, 1041 ms) = reprocesso dos 48;
  resto = 64 decode em regime estacionario (~92 ms/chunk, estavel).
  Sem somar tempos sobrepostos; sem 42ms antigos.

  Por token decode (server: 126,5 ms/token; 8,32 t/s warm / 7,71 t/s cold):
  - Vulkan FA (GPU medida): 27,2 ms/token (16 calls x 1,73 ms).
    Linear no overflow: 27,2 (16816) vs 60,8 (37376) = 2,24x para 2,22x.
  - Vulkan GPU busy total ~= 33 ms vs wall CPU-side 45,5 ms/token
    => ~12 ms de host/fence dentro do caminho Vulkan.
  - CUDA0 wall CPU-side: 49,0 ms/token (sem medidor GPU-side no CUDA).
  - Gap de makespan NAO decomposto e NAO pontual: com overlap desconhecido
    entre os walls 49,0/45,5, a conta 126,5-max(49,45,5)=77 ms e o TETO do
    gap (overlap total); o PISO e 126,5-49,0-45,5=32 ms (serializacao
    total). Intervalo honesto: [32; 77,5] ms/token. Medir a uniao real
    exige timestamps de inicio/fim por split (patch diagnostico
    pendente), nao duracoes isoladas.
  - F4SCHED (trace 32K): 9,5 MB/token CUDA->Vulkan (tails 258 KB x 32 +
    Qcur/fa_out), 3 KB no retorno, 1,00 ocorrencia/grafo.
  Bound rotulado como estimativa: 9,5 MB a teto nominal 1 GB/s ~= 9 ms
  SE sincrono no caminho critico; nao medido; overlap por eventos nao
  verificado. Proposta delta-staging segue como hipotese com bytes, sem
  ganho afirmado (exige prova de conteudo/aliasing/lifetime + A/B).

## UNIAO HOST EXATA POR TOKEN (patch SPLITSPAN local, 110k) - 21:14 UTC

  Patch minimo, aditivo, env-gated (GGML_SCHED_SPLIT_SPAN, default OFF):
  gid monotonico por grafo + 3 marcadores host por split (t_iter,
  t_postsync, t_postcopy), sem fences adicionados (ao contrario do
  SPLIT_TIMING). Diff: ggml/src/ggml-backend.cpp +95/-1 (só inserções).
  Binario stub byte-identico ao de aceite (7b8634fc...5a1de0); patch vive
  em libggml-base.so (013bc856...; aceite sem patch amostrado 86b59ebf...).
  A/B intercalado mesma hora/scripts: span 760,19 vs aceite 763,88 t/s
  (-0,5%, ruido). Deriva 834,52 (15:21) -> ~760 (21:1x) e termica/clocks,
  nao patch (patch desligado custa ~ns/split por inspecao de codigo).

  Fatos de codigo verificados (semântica das copias):
  - staging cross-backend e incondicional por grafo (sem dirty-tracking
    entre grafos); view copia a faixa da view (258 KB), offsets constantes.
  - destino Vulkan0 a partir de CUDA: ggml_backend_vk_cpy_tensor_async
    RECUSA src nao-VK/nao-host (retorna false) => cai no ramo imediato
    com INPUT flag (event_synchronize + tensor_copy bloqueante), SEM
    fence full de origem e SEM linha de profile. Prova empirica: ZERO
    linhas COPY_PROFILE em todos os runs com 9,5 MB/token trafegados.
    Correcao de relato anterior: staging NAO tem fences full por copia.
  - Switch de backend com n_inputs==0 => ggml_backend_synchronize FULL
    do backend anterior (ggml-backend.cpp:1826): serializacao total
    CUDA<->Vulkan por desenho (reuso de buffers do allocator).

  Decomposicao host SERIAL por token decode (63 grafos estaveis, sem
  dupla contagem, sem hipotese de overlap; thread host 100% ocupada
  entre marcadores):
  - prevsync (fences full de troca): 33,20 ms/token, tudo em splits CUDA0
    (espera trabalho GPU real em voo; real por A/B blocking anterior).
  - staging (event-waits + memcpys bloqueantes, ~10 MB): 69,13 ms/token
    (Vulkan-bound 51,15; CUDA-bound 17,98). CORRECAO (auditoria): esses
    69,13 ms INCLUEM espera (event waits por trabalho em voo) + copia
    efetiva; NAO sao custo removivel de transferencia. E "9,5 MB @
    ~186 MB/s" era bytes divididos por intervalo misto: NAO mede banda
    PCIe isolada. Separar espera vs copia e medir repeticao de conteudo
    exigem instrumentacao dedicada (pendente abaixo).
  - launch/loop 5,43 + gaps inter-token 7,06 (sampling+planning+tails).
  - host total 114,8 de 120,3 ms server (residuo 5,5 = tails finais). FECHA.
  - FA Vulkan GPU 27,2 ms/token e SUBCONJUNTO sobreposto (nao somar):
    linear no overflow (27,2 vs 60,8 para 2,22x).
  Intervalo antigo [32;77,5] substituido por medida: gaps host puros
  (fora splits) ~= 12,5 ms/token; o resto e espera/fetch dentro do
  caminho de staging+switch, nao "buraco" misterioso.

  Superficie de ataque ordenada por custo host medido (hipoteses, sem
  ganho afirmado; cada uma exige prova + A/B + gates):
  1. staging 69 ms: re-encena ~10 MB/token incondicionalmente. Delta
     persistente no Vulkan exigiria prova de repeticao de conteudo
     (AINDA ABERTA: offs=0 constante + KV append-only sugerem, nao provam).
  2. prevsync 33 ms: serializacao por switch; reduzir = menos switches
     (fusao de scheduler = fora do escopo) => estrutural, nao acionavel
     localmente. Reportado, nao atacado.
  3. gaps 12,5 ms: sampling+planning; pequeno.

## VEREDITO STAGING: redundancia PROVADA, fix exige decisao (auditoria) - 21:43 UTC

  Conteudo (FNV-1a full + 8 chunks por copia, run warm dedicado):
  tails 258 KB x 32 (16 camadas k/v, offs=0, roots estaveis):
  464/464 pares adjacentes IDENTICOS, 0 divergencias, 0 dirty chunks.
  Tensores pequenos majoritariamente estaticos (masks/idx 62/62);
  inp_pos avanca legitimamente a cada token (62 diffs esperadas).
  Mecanismo mapeado: views f16 do K-cache CUDA (KVarN exact-tail) ->
  splits FA Vulkan via ramo fallback do scheduler (fence FULL src +
  memcpy faixa integral, sem dirty-tracking, buffers por grafo).
  Prova de ramo: HASHSTAGED do fallback presente; profile stock mudo
  porque GGML_LOG_INFO e filtrado pelo callback de log (nao ausencia
  de execucao) - contraexemplo registrado contra a leitura anterior.
  Origem/owners/aliasing/lifetime: CUDA0 -> Vulkan0, views com root
  estavel por camada, staging buffers com lifetime de um grafo.

  Espera vs copia separadas (SPLITWAIT, ramo fallback, 15 decode):
  wait 41,5-49,7 ms/token (fences full + events) + memcpy 19,3-21,7 ms
  para ~10 MB (~490 MB/s efetivos com overheads por copia; NAO e banda
  isolada do link). Somado ao prevsync 33 ms (switches) e gaps 12,5 ms:
  conta host fecha sem dupla contagem.
  CORRECAO FINAL: os 69 ms nunca foram custo removivel; removivel real
  ~= memcpy 20 ms + round-trips (~2-5 ms) ~= 22-25 ms/token (~20%),
  POIS as esperas cobrem execucao GPU real (fences seriais por desenho).
  Correcao exigiria staging persistente cross-grafo (mudanca de lifetime
  no scheduler) = decisao arquitetural especifica, fora do escopo local.
  SEM fix neste turno, conforme a regra: proposta especifica com
  bytes/ms/correcao/risco acima; A/B e gates so com decisao.
  Proximo diagnostico concreto no escopo: gates de qualidade/KL do F5
  (independentes de velocidade; exigem campanha GPU com colds necessarios).

## PROTOTIPO TAIL-LOCAL-CUDA (autorizacao 05:48 UTC) - implementado

  Causa-raiz provada por troca de ramo: com preset CUDA explicito nos
  WHT, o round trip zera (staging some), logo o scheduler aceita CUDA
  (shapes suportados) e a ida ao Vulkan era decisao heuristica revertivel
  (compativel com vazamento do preset Vulkan via expand-up; mecanismo
  exato nao isolado, desnecessario para o fix).
  8,25 MB CUDA->Vulkan (tails) + ~8,5 MB Vulkan->CUDA (tails
  transformadas) por token a 110k some com o fix (0 linhas F4SCHED).
  Fix: ancorar saidas WHT (k_tail_local/v_tail_local pos-transform)
  em CUDA via set_tensor_backend, env LLAMA_OVF_TAIL_LOCAL_CUDA,
  default OFF, so no builder position-split (llama-graph.cpp, ~20
  linhas). Tentativa ampla (7 tensores) abortou em GGML_ASSERT de
  bookkeeping CPU (query_order/run_desc): refinado para so k/v.
  Snapshot explicito protegido de evict LRU (dir separada explicit/ +
  preset noauto para testes; auto-save publicava e evictava o arquivo
  de prova - licao registrada). Smoke 16 tokens bit-identico ao
  baseline (inclui prefill completo refeito).
  A/B p95 em curso (110000+512, F5TOK=1 nos dois bracos como na
  campanha, resto OFF): OFF-warm/medidos, ON-warm/medidos, KEEP_DOWN=1.

## A/B P95 TAILFIX (110000+512, intercalado, OFF) - 06:41 UTC

  OFF (sem fix): warm p95 135,097/med 127,902/dec 7,721;
                 medido p95 140,618/med 131,746/dec 7,452.
  ON (TAILFIX=1): warm p95 120,628/med 112,833/dec 8,651;
                  medido p95 118,355/med 111,379/dec 8,775.
  Delta medido/medidos: p95 -22,263 ms (-15,8%), mediana -20,367
  (-15,5%), decode +17,8%. Todas as combinacoes warm/medidos tem o
  mesmo sinal (-14,5 a -22,3 ms). Prefill inalterado (336,2 vs 335,0
  t/s: ganho e so decode nesta ocupacao). Termica estavel 46-48C nos
  4 runs. rc=0 todos, 511 amostras, sem rejects. Pico VRAM 11667 OFF
  vs 11669 ON (+2 MiB; free 615 vs 613, no limite do piso 614 por 1 MiB
  de granularidade do nvidia-smi: registrado, P preservado, revalidar
  par de fronteira se o fix for a aceite). [POS-msg956 06:59 UTC: piso
  614 MiB removido como gate; free 613 MiB deixa de reprovar por folga
  minima; medicoes acima preservadas como observacao.]
  Ganho medido 22,3 ms vs estimado 22-25 ms: estimativa validada.
  Tokens 512 OFF vs ON: identicos ate posicao 56, cascata apos flip
  (174140 vs 214200) — mesma assinatura do caso explicito; identidade
  nao e gate, KL decide (captura canonica em curso).
  Fingerprints: stub 7b8634fc...5a1de0 (identico ao aceite),
  libggml-base span+hash 3133fee4..., libllama tailfix 5e5e2377... (apos
  refino; build anterior 8b897918... descartado por assert).
  Bracos diferem SO no env TAILFIX. Termica registrada por run.

## ATUALIZACAO DE REQUISITO msg956 (2026-10-10 06:59:54 UTC): MEMORIA = NAO-OOM

  Usuario (msg956): "Pode tirar esse critério o único critério é não oom".
  Vigente a partir de 2026-10-10 06:59:54 UTC, so para o criterio de memoria:
  - GATE de memoria passa a ser SOMENTE ausencia de OOM.
  - Piso de 614 MiB (formula max(512 MiB, 5% da VRAM total)) e exigencia
    anterior de MemAvailable >= 2 GiB sem swap deixam de ser gates
    obrigatorios (plano §3.8/§4.1 atualizado, mesmas datas).
  - 613 MiB livres observados (tailfixAB-on/off) deixam de reprovar por
    piso minimo; medicoes historicas de pico, folga, RAM e swap ficam como
    observacoes, sem reescrita de numeros (inclui 11667/11669 MiB de pico,
    free 615/613, mem_available_min e swap_max ja registrados acima e nas
    secoes 5011-5317).
  - Preservados integralmente: qualidade/KL, estado/logits/tokens,
    estabilidade (10 ciclos, memoria retida <=64 MiB por backend), A/B
    equivalente e todo o restante do escopo autorizado.
  - Efeito pratico imediato: memoria do A/B tailfix nao reprova; os demais
    gates do A/B (p95, qualidade/KL, tokens) seguem pendentes/exigidos.

## QUALIDADE TAILFIX: KL OFF/ON + CONTROLE A/A + PARIDADE WHT (2026-10-10 07:25-08:05 UTC)

  Build-camp reconfigurado -DLLAMA_F4_TEST_LOGIT_CAPTURE=ON
  -DLLAMA_F4_TEST_COUNTERS=ON (BUILD_DONE rc=0). Hook canonico de captura
  de logits full-vocab; binary so para as pernas de qualidade (dois
  bracos usam o MESMO build; so TAILFIX env difere).

  Pernas baixo de P (LENS 3000,3072,3200,3328,3584, P=93184, cache off,
  deterministic, n_predict=1): OFF e ON com captures 5/5, tokens gerados
  identicos nos 5 checkpoints. f5-strict OFF vs ON: KL=0 exato em todos
  (fix e no-op abaixo de P - sem preset Vulkan, sem staging).

  Pernas acima de P (LENS 93440,96000,99072,110000, cache_prompt=1
  incremental, mesmos prefixos): captures 4/4, tokens gerados identicos
  nos 4 checkpoints. f5-strict OFF vs ON:
    n=93440 KL=1.919238e-01 p99=2.262422e+00 dLSE=1.206e-01 argmax=True
    n=96000 KL=9.338832e-02 p99=1.437415e+00 dLSE=5.586e-01 argmax=True
    n=99072 KL=1.015481e-01 p99=1.035322e+00 dLSE=4.242e-01 argmax=True
    n=110000 KL=1.351043e-03 p99=7.422711e-01 dLSE=6.295e-02 argmax=True
    mean_KL=9.70528028e-02 (gate <=1e-4) FAIL; max_p99=2.26242177e+00
    (gate <=1e-3) FAIL; STRICT_RC=1.

  CONTROLE A/A (mesmo braco OFF duas vezes, TAG off2-aboveP): KL=0
  exato nos 4 checkpoints (STRICT_RC=0) - execucao bitwise deterministica
  acima de P; o delta OFF/ON NAO e ruido de run.

  PARIDADE DE WHT (tool nova tools/occupancy-f5/wht-parity.c, bin
  build-camp/bin/wht-parity; CPU vs CUDA vs Vulkan, grid de shapes de
  producao [256,4,tail] e F32/F16/BF16):
  - CPU (referencia): carrega em f32, borboleta em f32, arredonda 1x
    na saida (ops.cpp ggml_compute_forward_kvarn_wht) = acumulacao estavel.
  - tail do KVarN default e F16 (docs/beellama-args.md:85
    "--kv-tail-type ... f16 for KVarN"; preset nao sobrescreve).
  - F32: Vulkan==CPU bit-identico em todos os shapes; CUDA idem em
    hw=128/64 e DIVERGE em hw=256 (ex. [256,4,127] bitdiff=35957/130048,
    maxabs=4.768e-07 = ulp).
  - BF16: Vulkan==CPU bit-identico; CUDA hw=256 diverge ulp-level
    (maxabs=4.883e-04).
  - F16 (dtype de producao): CPU=golden f32-estavel; CUDA hw=128
    bit-identico; CUDA hw=256 maxabs=6.104e-05 (bitdiff=980/130048);
    VULKAN diverge em TODOS os shapes f16 com maxabs=1.953e-03
    (ex. [256,4,127] bitdiff=64005/130048; [128,4,127] 32737/65024).
  - Vulkan nao suporta hw=64 (assert ggml-vulkan.cpp:10810; nao usado
    no tail de producao).

  ATRIBUICAO: pre-fix (OFF) roda o WHT do tail na VULKAN (vazamento do
  preset fa2) com erro F16 ~1.95e-3 vs golden; o fix (ON) roda em CUDA
  com erro ~6.1e-05 vs golden (~32x menor). Logo o KL OFF/ON acima de P
  e dominado pela REMOCAO do erro Vulkan do tail, i.e. o fix MELHORA a
  fidelidade estrutural; OFF e o braco pior. Explica queda do KL com o
  comprimento (cauda ~128 keys dilui), argmax identico e flip so no
  decode (posicao 56, desempate proximo). WHT em F32/BF16: caminho
  CUDA-native ulp-level aceitavel (registrado como observacao).
  GATE canônico restante: logits candidato vs referencia estrutural
  (CPU com mesmo KV por faixa, §4.1) <=1e-4/1e-3 - NAO executado ainda
  (exige harness de referencia); OFF/ON entre si nao e esse gate.
  Ferramenta preservada: tools/occupancy-f5/wht-parity.c
  (gcc -I ggml/include ... -lggml -lggml-base; bin em build-camp/bin).

## JANELA msg854 (20:18 UTC): 8090 EM MANUTENCAO, SEM RESTORES INTERMEDIARIOS

  Downtime iniciado 2026-10-09T20:22:23Z (GPU ready: inactive).
  Producao permanece PARADA ao fim deste turno; restore definitivo com
  health+inferencia real somente ao concluir o plano. Recibo pendente.


## FIX msg978 (2026-10-10): CHECKPOINTS DURAVEIS (`.ckpts`) APOS RESTORE EXPLICITO

  SINTOMA pre-fix: qualquer request com sufixo divergente sobre um slot
  restaurado do disco caia em FULL_REPROCESS (re-prefill integral).
  Evidencia pre-fix: probe2 N=8960 restored_p0=8950/8929/8896/8833 ->
  reason=suffix_removal_requires_reprocess, re=8960; e ciclo-1 real
  (130560): prompt_n=130560 cache_planned_n=0 cache_reprocessed_n=130560.

  CAUSA RAIZ (confirmada por instrumentacao LLAMA_SEQRM_DBG):
  - n_rs_seq=0 (spec off) => llama_memory_recurrent::can_seq_rm recusa
    rollback >=1 (tail_pos=8959, rollback=128); hybrid seq_rm_plan fail.
  - LIVE funciona porque o checkpoint genuino (criado no prefill quando o
    estado estava de fato na ancora) e restaurado PARTIAL_ONLY e rebobina
    tail_pos para 8831; o suffix trim vira remocao de celulas futuras
    (rollback=0).
  - RESTORED so tinha checkpoint sintetizado cujo PAYLOAD e o proprio
    estado vivo (@8959); restore_tgt nao rebobina o filho recorrente
    (tail_pos segue 8959) => suffix trim exige rollback=128 => recusa =>
    FULL_REPROCESS.
  - Janela recorrente (LLAMA_RS_SEQ=128) testada e DESCARTADA: 19.3 GiB
    de VRAM (~149.6 MiB por linha do ring); knob revertido no common.cpp.

  FIX: persistir os checkpoints GENUINOS em sidecar `.ckpts` ao lado do
  state file (SLOT_SAVE) e instala-los no restore; fallback sintetizado
  antigo preservado quando o sidecar falta/esta invalido.
  - tools/server/server-context.cpp: sidecar v2 (magic SLOTCKPT, XXH64,
    formato adaptive_slot_writer/reader), hook no SLOT_SAVE inline,
    helper load_slot_checkpoints_from_sidecar usado pelo handler inline
    e por do_slot_restore (auto-restore de disco). Validacao por
    checkpoint: payload nao-vazio, n_tokens <= prompt, compatible_tgt
    (modelo/instancia/layout). SEM comparacao exata de tamanho: o blob
    PARTIAL_ONLY varia legitimamente com os retained bounds do estado
    vivo (7424 B a mais @8959 vs @8831).
  - Linhagem: snapshots parciais so fazem replay no MESMO owner epoch, e
    o full-restore avanca o epoch (++state_owner_epoch). O sidecar grava o
    epoch do save e o restore o RE-ADOTA.
    * src/llama-kv-cache-kvarn.h: adopt_state_owner_epoch().
    * src/llama-ext.h + src/llama-context.cpp:
      llama_kvarn_state_owner_epoch() /
      llama_kvarn_adopt_state_owner_epoch() (mesma navegacao dos probes
      KVarN: hybrid->get_mem_attn_base -> llama_kv_cache_kvarn).
    * Replays continuam passando a validacao fina de referencias/celulas.

  EVIDENCIA:
  - probe2 N=8960 (pos-fix): restored_p0=8950/8929/8896/8833/8832 ->
    committed, plan=8832, re=128, prompt_n=128 (identico ao live, outputs
    byte-identicos); restored_exact_A prompt_n=0 (fast path preservado);
    p0=8704 (r=256) full = igual ao live (fora da janela da ancora).
    Log: "adopted durable checkpoint lineage epoch 4 (was 5)" +
    "loaded 1 durable context checkpoint(s)".
  - escala 130560 (2026-10-10, CYCLES=1): restore n_restored=130560;
    div @130528: prompt_n=128 cache_n=130432 cache_planned_n=130432
    cache_reprocessed_n=128 reason=committed (pre-fix: 130560/0/130560);
    over-C 131074 -> HTTP 400 limpo; vram 11655->11663.
  - GATE 10 ciclos (2026-10-10T11:06:26Z->11:39:30Z, port 8970, NP=512):
    10/10 CYCLE_CHECK prompt_n=128 predicted_n=512, restore n_restored=
    130560 em todos, VRAM 11655->11663 (flat), final_health=200.
    Snapshot novo: explicit-stability-130560.bin 4276076495 B +
    .ckpts 199203989 B + .logits 993296 B.
  - Artefatos: /home/hjotha/f4-spool/f5-campaign/stability-130560/
    (run.log, cycle-{1..10}(-raw).json, overc rejeitado); probe em
    /home/hjotha/f4-spool/f5-campaign/restore-div-probe2/; saidas em
    /tmp/opencode/probe2-epoch.out, stability-cy1.out, stability-cy10.out.

  CAVEATS:
  - Janela de reuso = checkpoint genuino mais recente (ancora 130432 =
    N-128 no ciclo). Divergencia mais profunda que a ancora => full
    re-prefill (mesma limitacao do fluxo live; sem regressao).
  - Snapshots gerados antes do fix nao tem .ckpts => fallback sintetizado
    antigo (full reprocess em divergente); regenerar artefatos de
    campanha ao trocar de binario.
  - Eviccao LRU (slot_save_enforce_limits) nao remove .ckpts orfaos;
    monitorar espaco (cada sidecar ~190-200 MB).
  - LLAMA_SEQRM_DBG mantido (env-gated, inerte em producao);
    LLAMA_RS_SEQ removido (revertido).


## MSG57 (2026-10-10 13:12-16:00 UTC): REGISTRO 10 CICLOS + GATE ESTRITO (FIX DO DEFAULT) + WHT

1) REGISTRO DOS 10 CICLOS (restore 130560 / reuso 130432 / reproc 128 / pred 512):
   logs preservados ANTES de qualquer nova run em
   f5-campaign/stability-130560/evidence-msg978-10cycles/ (38 arquivos,
   SHA256SUMS). sha256: run.log=001b1c46..., server.log=6ea1dae6...,
   cycle-1-raw=39b7e9e4..., cycle-10-raw=84c12050.... Os 10 ciclos atestam
   ESTABILIDADE daquele teste; truncated/texto gerado NAO aprovam
   estado/logits/qualidade (gates abaixo).

2) GATE NUMERICO ESTRITO — BUILD CORROMPIDO CORRIGIDO (3 fixes locais):
   - src/llama-position-split.h: alias local `llama_pos = int32_t` (usado por
     tail_extent_queries; o header e backend-free). Destrava
     test-position-split-ranges/merge/lse*/merge-graph/real-pair/ubatch.
   - src/llama-kv-mixed-state.cpp: include llama-impl.h + auto_format() local
     (a TU tambem e compilada standalone por test-kv-mixed-io, link ggml-only,
     onde format() de llama-impl.cpp nao resolve). Destrava test-kv-mixed-io.
   Suite rodando (build-camp/bin, LD_LIBRARY_PATH=bin):
   VERDE: ranges 391/0, merge 51/0, lse 306/0, lse-alloc 230/0, lse-cu 7/0,
   lse-vk 485/0, lse-sched 20/0, real-pair 8490/0, kv-mixed-io 20225/0,
   kv-mixed-state 621/0, stream 819/0, placement 618/0, handoff ALL PASS,
   test-kvarn GPU: 18 shapes OK (d256-q1-tail nrmse 7.9e-5; q256 1.0e-6).
   VERMELHO: test-position-split-merge-graph 2/8 mandatory:
     D256 Q1 GQA24/4 nrmse=1.127e-3; D256 Q256 nrmse=1.013e-3 (gate 1e-3).
   Atribuicao (nao escondida): merge aritmetica EXATA; o excesso vem da rota
   STAND-IN local=Vulkan-F16 (base_nrmse 1.036e-3/1.019e-3) — o produto usa
   local=CUDA/KVarN (test-kvarn 7.9e-5) + overflow=Vulkan/Q4 (lse-vk
   3.1e-4..8.5e-4), ambas abaixo do gate. CORRECAO REQUERIDA (sem relaxar
   limiar): rodar os shapes mandatory na dupla do produto (KVarN empacotado
   local + Q4 Vulkan) — PENDENTE.

3) INVESTIGACAO VULKAN/CPU/CUDA + WHT (tool wht-parity, dados frescos):
   F32: Vulkan BIT-IDENTICO ao CPU em todos os shapes; CUDA hw=256 difere
   (maxabs 4.768e-7 = ulp f32).
   F16: CUDA hw=128 bit-identico; CUDA hw=256 maxabs 6.104e-5;
        VULKAN difere em TODOS (maxabs 1.953e-3, bitdiff ~50%, maxrel=1.0)
        -> defeito de conversao F16 (packHalf2x16) do shader RADV; registrado,
        NAO mascarado (legado fica atras do opt-out e o default nao o usa).
   BF16: Vulkan bit-identico; CUDA hw=256 maxabs 4.883e-4.
   EVIDENCIA DE GRAFO (F4_NODE_DUMP + F4_NODE_DUMP_FA_SRC, P=2048 n=3000):
   no eval da captura, o WHT da tail-V CONSUMIDO pela fa1 difere entre bracos
   (hash b15eb2bf... vs 1dc57015...), enquanto a view pre-WHT
   cache_kvarn_v_tail_l3 e identica (9527bbf0...) e fa1/fa2 batem nos demais
   evals; scheduler (GGML_SCHED_DEBUG=2): legado rodava 32x KVARN_WHT 258K na
   Vulkan; apos o fix o default roda 272/272 em CUDA0.

4) CORRECAO DO DEFAULT ("prototipo OFF por padrao"): src/llama-graph.cpp
   ancora k_tail_local/v_tail_local em CUDA POR PADRAO;
   LLAMA_OVF_TAIL_LOCAL_CUDA=0 restaura o legado (somente controle A/B).
   Validacao (mesmo binario, f5-tailfix-sm.sh, P=2048, n=3000, full-vocab
   f5-strict): DEFAULT vs ON (env=1): KL=0 EXATO; DEFAULT vs LEGACY (env=0):
   mean_KL=9.113e-4 FAIL (antes 9.79e-4) — isola a causa no WHT da tail.
   Default passa a usar a rota cujo WHT fica a 6.1e-5 do golden CPU.

5) PENDENCIAS (gates NAO fechados): (a) merge-graph mandatory na dupla do
   produto; (b) defeito de conversao F16 do WHT Vulkan (RADV); (c) msg57
   itens 3-5: restore em processo NOVO (estado/logits/continuidade,
   divergente/cancel/EOS/rollback/isolamento), A/B intercalado >=5 reps,
   131072/canario/rollback e .ckpts orfaos. Producao 8090 segue PARADA.


## MSG57-BIS (2026-10-10 16:00-17:00 UTC): DEFAULT OFF RECONCILIADO + MERGE-GRAPH 2/8 CORRIGIDO (PAR REAL)

1) DEFAULT OFF RECONCILIADO (msg57 item 1): src/llama-graph.cpp volta a
   manter o ancora da tail-WHT como OPT-IN. Condicao exata documentada no
   codigo e identificada por comportamento REAL de LLAMA_OVF_TAIL_LOCAL_CUDA:
     ausente -> OFF (legado: preset do fa2 vaza para os nos WHT da tail)
     "0"     -> OFF
     "1"     -> ON  (k_tail_local/v_tail_local em CUDA com fa1)
     outro   -> OFF (fail closed)
   Verificado com capturas full-vocab (P=2048 n=3000, mesmo binario):
     OFF(ausente) vs LEGACY(=0): KL=0 EXATO (braos identicos por comportamento)
     OFF(ausente) vs ON(=1):     mean_KL=9.788e-4 p99=0.6998 dLSE=0.213 (delta
     conhecido do WHT da tail; RADV F16 maxabs 1.953e-3 vs CUDA 6.104e-5).
   NOTA (msg57): DEFAULT=ON com KL0 nao valida o candidato contra referencia
   independente; a promocao do ON permanece BLOQUEADA ate o gate estrutural de
   logits (item 3). Producao permanece no comportamento legado.

2) MERGE-GRAPH 2/8 MANDATORY CORRIGIDO (msg57 item 2), sem relaxar limites:
   Causa: os casos mandatory rodavam um PAR STAND-IN (local=F16 na Vulkan,
   ~1.0-1.1e-3 > gate 1e-3) porque a FA CUDA declina LSE fora do KVarN; nao e
   rota do produto.
   Correcao (tests/test-position-split-merge-graph.cpp, +/- ~350 linhas):
   os casos mandatory agora rodam a DUPLA REAL: local = KVarN4 no CUDA
   (ggml_kvarn_store + view nativa, dominio rotacionado, O des-rotacionado
   antes do merge -- espelho do llama-graph.cpp) + overflow = Q4_0 na Vulkan,
   mesmo merge e mesmas mascaras. Referencia estrutural = MESMA execucao
   KVarN/Q4 na CPU (plan 4.1: atencao CPU sobre o KV dequantizado da mesma
   representacao por faixa; a FA CPU decodifica os mesmos records).
   Resultados (limites inalterados 1e-3 / 1e-2 / 1e-2):
     D128 Q1  nrmse=3.629e-4 maxO=3.014e-5 maxLSE=8.583e-6
     D128 Q2  nrmse=7.426e-4 maxO=6.344e-5 maxLSE=4.625e-5
     D256 Q1  nrmse=3.911e-4 maxO=4.124e-5 maxLSE=6.676e-6
     D256 Q2  nrmse=7.157e-4 maxO=7.863e-5 maxLSE=6.485e-5
     D256 Q256 nrmse=3.322e-4 maxO=8.805e-5 maxLSE=6.866e-5
   Placements verificados por rodada (fa1 local=CUDA, fa2 overflow=Vulkan,
   attn_on_ovf>=1). Resultado: 3268814 checks, 0 failures, 8 mandatory shapes.
   Preservacao: os runs do par stand-in continuam executando e IMPRIMINDO seus
   numeros (com nota de que nao sao a rota do produto) -- nenhum caso excluido,
   nenhum FAIL reclassificado; o gate mandatory apenas passou a medir a rota
   correta. CMake: alvo agora linka llama (record layout). Ponte descartada
   (documentada): materialize CPU nao espelha o roteamento stage/records da
   view nativa (store nao-eager nao escreve records; stage mantem o tail
   exato), por isso a referencia escolhida e a propria execucao CPU do grafo.
   Debug temporario (decoder de records) removido apos a validacao.

3) Suite pos-mudanca: test-position-split-merge-graph 0 failures (8 mandatory);
   demais alvos intocados. llama-server rebuild OK (libllama.so com default
   OFF restaurado). Producao segue PARADA.


## MSG57-TER (2026-10-10 16:30-17:10 UTC): RESTORE EM PROCESSO NOVO (item 4) + OWNER-ID/EPOCH POR CHECKPOINT

Harness novo: /tmp/opencode/restore-newproc.sh (fase A = processo 1: prefill
8960 + SAVE explicito com sidecars; fase B = PROCESSO NOVO: restore + matriz).
Fingerprints registrados por run (sha256 llama-server/libllama/
libllama-server-impl, modelo size+mtime, preset). Evidencia em
f4-spool/f5-campaign/restore-newproc/ (run.log/serverA.log/serverB.log/rows.json).

ACHADO (vermelho real, corrigido): com sidecar v3 (epoch unico do save) o
replay cross-process FALHAVA ("partial snapshot owner/epoch mismatch") porque
(a) a validacao exige o OWNER ID do cache, que e PROCESS-RANDOM (novo processo
= novo id) e (b) o epoch de CAPTURA do checkpoint pode diferir do epoch do save.

CORRECAO (local, reversivel):
- common_prompt_checkpoint ganha owner_epoch, CARIMBADO no create_checkpoint
  com lhama_kvarn_state_owner_epoch (epoch no instante da captura).
- sidecar .ckpts v4: epoch POR CHECKPOINT + owner_id no header.
- src: state_owner_id deixa de ser const + adopt_state_owner_id; APIs ext
  llama_kvarn_state_owner_id / llama_kvarn_adopt_state_owner_id (mesma
  navegacao dos probes); restore adota owner_id e o epoch mais novo nao-zero.
- Validacoes finas de celula/referencia intactas; falha = fallback full.
- NOTA: v4 invalida sidecars v3 (fallback documentado); snapshots de campanha
  devem ser regenerados ao usar o binario novo.

EVIDENCIA POS-FIX (processo B novo, 2026-10-10 ~14:44Z):
- exact_A: prompt_n=0 (fast path) e re=0; isolamento: re-restore+div+exact
  repetem prompt_n=0 (sem contaminacao).
- div p0=8950/8896: cache_n=8832 plan=8832 re=128 prompt_n=128 committed
  (mesmos numeros do same-process; outputs identicos aos do probe2).
- div p0=8704: full reprocess (fora da janela da ancora; igual ao live).
- cancelamento: cliente aborta stream de 1024 tokens; /health ok; exato
  pos-cancel prompt_n=0.
- EOS: logit_bias no EOS (id 248046 resolvido via /tokenize "<|im_end|>")
  -> stop_type=eos, pred_n=1.
- timings separados: restore_ms ~1.2-2.3s; prompt 128 tokens ~1.3-1.7s;
  decode registrado em rows.json.
- same-process probe2 RERODADO com o binario v4: 6/6 reuse_ok (8950..8832),
  exact prompt_n=0, p0=8704 full -- SEM REGRESSAO, outputs idempotentes.

STATUS msg57: itens 1 e 2 completos; item 4 completo com a correcao acima.
Item 3 (KL de logits <=1e-4 vs referencia estrutural): o nivel de ATENCAO
foi validado pela dupla real no merge-graph (CPU ref); o gate de LOGITS
(insercao da referencia no grafo / harness de injecao) segue PENDENTE e
bloqueia a promocao do tail-WHT ON. Item 5 (A/B >=5 reps, 131072/canario,
.ckpts orfaos) pendente. Producao segue PARADA.


## MSG57-ITEM3 (2026-10-10): GATE DE LOGITS vs REFERENCIA ESTRUTURAL — HARNESS PRONTO, RESULTADO VERMELHO

HARNESS (teste minimo, P=2048 n=3000, captura full-vocab F4_TEST_LOGIT_CAPTURE +
comparador /tmp/opencode/f5-strict.py, FP64; arms: off(legacy default),
on(tailfix env=1), ref(CPU), mat(materializado), ovfmat/locmat/maton atribuicao):
- Arm "ref" = nova chave LLAMA_OVF_ATTN_REFERENCE_CPU=1 (src/llama-graph.cpp,
  default OFF): ancora na CPU k/v_tail_local, o1, fa1, lse1, fa2, lse2 (fail-soft
  via supports_op, log por camada). Exige GGML_CPU_KVARN_ALLOW_MIXED_DOMAIN=1:
  o supports_op da CPU so aceitava dominio ROTATED e a producao usa
  ROTATED_K_ORIGINAL_V (gate relaxado por env, default de producao intacto;
  o compute CPU e domain-agnostico -- os records sao lidos como gravados).
- BLOQUEIO (impedimento concreto): com os 7 nos ancorados em 16/16 camadas, o
  filho SEGFAULT (status 139). Backtrace (coredumpctl):
  #0 ggml_fp16_to_fp32_row (libggml-base) <- #1 ggml_compute_forward_flash_attn_ext
  (CPU) <- #2 ggml_graph_compute_thread. Ou seja: a FA plain da CPU lê as VIEWS
  de overflow Q4 residentes na VULKAN (v_to_float sobre memoria de device) --
  o scheduler nao copia views para no ancorado por usuario. Ancorar tambem
  k/v_overflow e k/v_ovf_gathered NAO resolve (reproduzido standalone,
  attnref-standalone.sh). ACAO NECESSARIA: trabalho de scheduler/kernel
  (copia cross-backend de views para nos FA ancorados) -- fora do escopo de
  correcao local rapia; parada apenas esta parte.
- MEDICAO VIA REFERENCIA EQUIVALENTE (mesmo KV por faixa, materializado F16,
  FA padrao com acumulacao estavel; overflow Q4 Gather-F16 = bit-identico):
    ovfmat vs off : KL = 0.00000000e+00  PASS  (rota Q4 overflow e fiel)
    locmat vs off : KL = 2.96826668e-03  FAIL  (rota KVarN nativa local)
    mat vs off    : KL = 2.96826668e-03  FAIL
    mat vs on     : KL = 3.28215226e-03  FAIL
    maton vs on   : KL = 3.71802073e-03  FAIL
    off vs on     : KL = 9.78764989e-04  FAIL  (delta do tailfix)
  ATRIBUICAO: 100% do gap estrutural vem do kernel FA nativo KVarN (local);
  overflow Q4 e bit-identico a referencia; delta tailfix (~0.98e-3) e menor e
  aditivo. O gate KL<=1e4 falha por ~30x para OS DOIS bracos.
- CONSEQUENCIA (regra msg57: corrigir falhas antes de baterias amplas): campanha
  A/B (item 5) NAO iniciada. Promocao do tailfix ON continua BLOQUEADA.
- REQUISITO PARA FECHAR: acumulacao estavel F32 no kernel FA nativo KVarN CUDA
  (ou referencia ancorada de fato na CPU apos o work de scheduler). Sem
  relaxar limiar e sem reclassificar FAIL.


## MSG57-ITEM2 (2026-10-10 20:00-23:30 UTC): RESTORE CANONICO P93184/C131072 EM PROCESSO NOVO, MULTITOKEN, CONTROLE EQUIVALENTE

Harness: /tmp/opencode/canonical-newproc.sh (+ canonical-driver.py) e
/tmp/opencode/canonical-logits.sh (+ canonical-logits-driver.py). P=93184,
C=131072, corpus tiled a 130560 tokens, divergencia em p0=130528 com 32 tokens
novos (mesmo padrao do stability), n_predict=8, temp 0, seed 42. Um processo de
servidor por fase (capturas full-vocab nao se sobrescrevem). Binarios
(sha256): llama-server e libllama.so/libllama-server-impl.so registrados no
run.log; modelo size=10442827776 mtime=1790317082.

RESULTADO DE ESTADO (processo NOVO, sidecar v4, snapshot canonical-130560.bin
4276076495 B + .ckpts 199204005 B):
- restore: n_restored=130560 (restore_ms 67.4s com processo cheio; 13.8s no
  processo dedicado)
- exact multitoken: prompt_n=0 (fast path) + 8 tokens; TEXTO IDENTICO ao
  controle live (19/19 chars)
- divergente p0=130528: cache_n=130432, cache_lcp=130529, cache_planned=130432,
  cache_reprocessed=128, reason=committed, prompt_n=128, +8 tokens;
  TEXTO IDENTICO ao controle cold ("=0.00000" nos dois)
- isolamento (restore repetido apos divergencia): prompt_n=0 (estado limpo)
- cancelamento: cliente aborta stream de 1024 tokens; /health ok; request
  exata seguinte prompt_n=0 (recuperacao confirmada)
- EOS: em probe anterior (P=8192) stop_type=eos pred_n=1 (eos_id 248046)
- rollback: restore apos divergencia + exata = prompt_n=0
- contadores registrados apos cada request (rows em run.log e *_rows.json)

RESULTADO DE LOGITS (canonical, restoring vs controle cold equivalente):
  B2(restored+divergente) vs C2(cold+divergente): KL=1.186831e-03 (B2 base) /
  1.137608e-03 (C2 base); p99_dlogp=3.920562e-01; dLSE=1.066e-01; argmax same.
  → Acima do gate KL<=1e-4/p99<=1e-3 (falha ~12x na media). Tokens identicos.
  NOTA: a mesma ordem de magnitude do gap do kernel FA nativo KVarN (secao
  MSG57-ITEM3, ~3e-3 em P=2048) sugere erro de acumulacao do kernel, nao de
  estado; atribuicao fina nao isolada em escala canonica.

INCIDENTE DE DISCO (resolvido): --slot-save-auto (necessario para o hook de
captura) gerava auto-saves de 4.3GB por request e encheu o disco (100%).
Liberados 33GB removendo APENAS auto-*.bin transitorios; evidencias (caps, logs,
tokens, snapshot canonico, evidence-msg978-10cycles) preservadas. Script
corrigido: --slot-save-max-count + slots por fase + limpeza de auto-saves entre
fases; uma request por processo (evita interferencia do auto-restore que
corrompeu uma execucao intermediaria: prompt_n=0 indevido no divergente).

VEREDICO: item 2 (estado/tokens/continuidade/isoldacao/cancel/EOS/rollback)
APROVADO; comparacao de LOGITS contra controle equivalente FALHA o gate 1e-4.
Combinado com a falha estrutural do item 3 (kernel KVarN nativo), a campanha
A/B (item 5) segue BLOQUEADA.


## MSG57-F5F7 (2026-10-11): SEGV DA REFERENCIA CPU — ISOLAMENTO, CAUSA E PARADA

REPRODUCAO MINIMA (ggml, sem modelo): /tmp/opencode/fa-view-repro.cpp e
/tmp/opencode/fa-cross*.cpp.
- fa-view-repro (view de buffer device, FA ancorada em CPU): NAO crasha --
  o scheduler realoca o backing para host quando o no do grafo e ancorado.
- Repros com storage em ctx de buffer-type Vulkan + FA ancorada em CPU:
  tambem NAO crasham (status=0).
- CONCLUSÃO DO ISOLAMENTO: o padrao "operando em device + FA ancorada em CPU"
  por si so e seguro. O crash exige os operandos serem BUFFERS DE CACHE
  (records/stage do KVarN e companion Q4), que nao sao tensores do grafo e
  portanto nunca sao copiados pelo scheduler; o kernel CPU decodifica esses
  registros in-place e le memoria de device.

CAUSA RAIZ (confirmada no grafo real, standalone3 + sched-debug do router run):
- nodes #215 (FA local) e #230 (FA overflow) em [CPU]; operandos KVarN
  (k_local/v_local = permutes de views de records/stage) e Q4 gather ficam em
  [Vulkan]; o scheduler cria os "CPU#... [NULL]" de entrada mas NAO aloca
  buffer para eles -> o kernel CPU dereferencia ponteiro nulo/device.
- backtrace (coredumpctl, core 2377356): #0 ggml_fp16_to_fp32_row <-
  #1 ggml_compute_forward_flash_attn_ext (CPU) <- #2 ggml_graph_compute_thread.
- Ancorar tambem os operandos (q_local/k_local/v_local/k_ovf_gathered/masks)
  NAO resolve: os buffers de cache seguem em device e continuam sem copia
  (reproduzido: crash no primeiro eval, log com 30 linhas, exit 139).

DECISAO (escopo): produzir a referencia CPU valida exige MATERIALIZACAO HOST
dos records/stage do KVarN (copia via ggml_backend_tensor_get para um buffer
host + grafo de decode) ou mover o cache para host -- trabalho de
scheduler/kernel, fora de "correcao local reversivel". PARADA apenas esta
parte, conforme regra. A referencia CPU NAO foi obtida; portanto atribuicoes
de KL a "FA/KVarN" seguem baseadas na decomposicao por FORMATO/ROTA
(ovfmat/locmat), que e valida independentemente da CPU:
  ovfmat vs off : KL 0 (bit-identico) -- overflow Q4 fiel
  locmat vs off : KL 2.968e-3        -- FA nativo KVarN local = gap
  off vs on     : KL 0.979e-3        -- delta do tailfix

COMPARACAO OFF/ON + FRESH/RESTORE (fingerprints iguais por braco):
- FRESH (mesmo binario e091e8d6/libllama 7b8634fc, mesmo env exceto o braco,
  P=2048 n=3000): OFF vs ON = KL 9.788e-4, p99 0.6998, dLSE 0.213, argmax
  same -- reprodutivo (identico ao valor historico).
- RESTORE (canonico P93184/C131072, snapshot integro, sidecar v4):
  restored vs cold controle = KL 1.19e-3/1.14e-3, p99 0.392, tokens
  identicos (secao MSG57-ITEM2).
- Tokens/textos identicos em todos os pares; falhas sao de DISTRIBUICAO
  (logits), nao de amostragem.

STATUS: gate KL<=1e-4 continua FALHANDO em todas as comparacoes com referencia
de logits; A/B de desempenho e promocao do ON seguem BLOQUEADOS (regra:
corrigir antes de bateria ampla). Prototipo continua OFF por padrao.


## MSG57-DIAG-CPU (2026-10-11): HARNESS ISOLADO, UM EVAL DE CAMADA, REFERENCIA CPU EM HOST

Escopo: diagnostico LOCAL e DESCARTAVEL. Nada de cache/scheduler/kernel de
produco alterado; sem campanha A/B, sem coldprefill novo, sem promocao de
tailfix, sem push/merge. Modelo/formatos/limites intactos (16 camadas, P93184,
C131072, snapshots e baselines preservados).

ARTEFATO: /tmp/opencode/kv-cpu-diag/one-layer-cpu-ref.cpp (+ binario e
run1.log no mesmo diretorio). Um unico eval de uma camada, grafo unico,
buffer unico alocado em buffer-type HOST (CPU):
- q (F32, [256,1,24]) + q_rot via KVARN_WHT;
- KVarN: current_k/v F32 -> ggml_kvarn_store (bits=4, sinkhorn 16,
  stage_groups=5) -> k_rec/v_rec I8 -> ggml_kvarn_view -> reshape/permute;
- FA nativa (dominio ROTATED) + LSE;
- referencia F16: ggml_kvarn_materialize (emit rotated) -> FA simples + LSE;
- overflow Q4_0 em host (FA simples) e sua referencia F16 (mesmos bytes Q4
  dequantizados para F16);

VERIFICACAO DE OPERANDOS ANTES DA COMPUTAÇÃO (resolvendo a cadeia de views,
como o f4 dump resolve):
  q_rot      KVARN_WHT  ne=[256,1,24]  buf=CPU host=1
  k_view     PERMUTE    ne=[256,512,4] backing on buf=CPU host=1
  v_view     PERMUTE    ne=[256,512,4] backing on buf=CPU host=1
  mask       F16        ne=[512,1,1]   buf=CPU host=1
  f16ref k/v PERMUTE    ne=[256,512,4] buf=CPU host=1
  ovf q4 k/v Q4_0       ne=[256,512,4] buf=CPU host=1
  ovf f16 k/v F16       ne=[256,512,4] buf=CPU host=1
  records k/v I8        ne=[8960,8,4]  buf=CPU host=1
  => TODOS os operandos (inclusive o backing dos views) em host, nao-NULL.
  => eval unico no backend CPU: status SUCCESS, SEM SEGFAULT.

COMPARACAO COM A REFERENCIA F16 EXISTENTE (mesmos records/stage, mesma q,
mesma escala; 6144 elementos de O):
  KVarN nativo vs F16 : NRMSE=1.515e-03  max|dO|=5.870e-03  max|dLSE|=2.146e-05
                        bitdiff=6144/6144
                        -> FALHA o gate de atencao do 4.1 (NRMSE<=1e-3);
                           max|dO|<=1e-2 e max|dLSE|<=1e-2 passam.
  Q4 overflow vs F16 : NRMSE=1.020e-03  max|dO|=1.678e-06  max|dLSE|=4.768e-07
                        bitdiff=6144/6144
                        -> FALHA nominal o NRMSE, mas os erros ABSOLUTOS sao de
                           ordem de bit (1.7e-6); o NRMSE relativo e amplificado
                           porque a referencia O dessa faixa tem magnitude
                           muito pequena (metrica relativa sobre denominador
                           proximo de zero). Nao ha evidencia de defeito real
                           nessa faixa.

RESULTADO / EXIT STATUS:
  - Harness compila e roda; EXIT=1 ( gates do 4.1 nao satisfeitos no KVarN).
  - SEGFAULT da referencia CPU: NAO reprodutivel com todos os operandos em host
    -> confirma que a causa é de COLOCACAO/SCHEDULER (copies cross-backend nao
    alocados, `CPU#... [NULL]`), nao do kernel CPU.
  - Limitacoes do diagnostico: 1 camada, 1 query de decode (n_q=1), 512 chaves,
    4 bits, sem tail exato e sem merge de faixas; nao prova o comportamento do
    prompt (n_q=256) nem do logits final (KL). A diferença medida aqui
    (NRMSE 1.5e-3 na atencao) e consistente em sinal com o gap de logits
    observado no modelo (locmat vs off = KL 2.97e-3).
  - Impedimento pendente (registrado em MSG57-F5F7): referencia CPU valida em
    escala de modelo exige materializacao host dos buffers de cache —
    trabalho de scheduler/kernel, fora do escopo; parada apenas essa parte.

BLOQUEIO: nenhum novo bloqueio foi criado por este diagnostico; ele reduz o
espaco de causas do SEGFAULT e quantifica o gap nativo-vs-F16 por faixa.
Gates de qualidade (KL<=1e-4) seguem FALHANDO; A/B e promocao seguem
bloqueados. Prototipo continua OFF por padrao. Producao 8090 parada.

## MSG57-DIAG-CPU — CORREÇÃO E DIFERENCIAL (2026-10-11)

CORREÇÕES EXPLÍCITAS à seção MSG57-DIAG-CPU acima (conforme solicitado):
1. O harness usa dados SINTÉTICOS gerados e alocados DIRETAMENTE em buffer-type
   HOST. Ele NÃO copia e NÃO reproduz o cache real Vulkan/Radeon de produção,
   nem o scheduler multi-backend, nem os operadores de cache reais
   (cpy_k/cpy_v, staging em anel, flush de records, pipeline de prefill).
2. Portanto, o fato de o eval não ter sofrido segfault NÃO prova que o
   segfault do modelo esteja resolvido. Ele apenas descarta (neste recorte
   isolado) a hipótese de defeito do kernel CPU de FA KVarN quando todos os
   operandos estão em host. O bloqueio do modelo permanece conforme
   MSG57-F5F7 (cópia cross-backend de views/buffers de cache).

DIFERENCIAL MÍNIMO (mesmos dados; 4 caminhos sobre os mesmos records/stage)
Artefato: /tmp/opencode/kv-cpu-diag/differential-cpu-ref.cpp (+ run-diff5.log).
Shape: D=256, GQA 24/4, n_kv=512, n_q=1, bits=4, 1 query, máscara toda
visível, sem tail exato e sem merge de faixas. Duas configs:
  A) stage_groups=5 (produção; tail_groups=4): NENHUM flush de records
     acontece — todo K/V é lido do stage F16.
  B) stage_groups=2 (tail_groups=1): grupos 1 e 2 são lidos dos records
     4-bit quantizados; grupos 0 e 3 do stage.
Meu decode escalar FP64 replicou o roteamento e a dequantização de
ggml-cpu/ops.cpp (fórmula de from_stage/from_record com
`group + (tail_groups - 1) >= live_group`) e foi VALIDADO contra a
materialização: max|d| = 0.000 em todos os grupos da config A
(bit-idêntico) e 3.7e-3/3.9e-3 nos grupos de records da config B
(= erro de quantização 4-bit, esperado).

RESULTADOS separados (domínio ORIGINAL para K/V; domínio ROTACIONADO para O,
pois é onde a FA nativa opera com a q rotacionada):
  Config A (stage_groups=5) — perda apenas de ESTAGIAMENTO (F16):
    K: orig vs decode escalar  NRMSE=2.098e-04  max|d|=3.498e-04 (RMS ref 5.908e-01)
    V: orig vs decode escalar  NRMSE=2.194e-04  max|d|=3.540e-04 (RMS ref 5.438e-01)
    FA nativa vs FP64 (mesmos valores decodificados):
       NRMSE(O)=4.462e-07  max|dO|=2.750e-06  max|dLSE|=6.357e-07  -> PASS
    FA materializada F16 vs FP64:
       NRMSE(O)=1.254e-03  max|dO|=1.077e-02  max|dLSE|=3.865e-05  -> FAIL
       (falha NRMSE<=1e-3 e estoura |dO|<=1e-2 marginalmente)
    FA nativa vs FA F16      NRMSE(O)=1.254e-03
  Config B (stage_groups=2) — perda de QUANTIZAÇÃO (4-bit) sobreposta:
    K: orig vs decode escalar  NRMSE=2.107e-02  max|d|=5.840e-02
    V: orig vs decode escalar  NRMSE=4.228e-02  max|d|=6.122e-01
    FA nativa vs FP64:          NRMSE(O)=6.309e-07 max|dO|=3.887e-06 -> PASS
    FA materializada F16 vs FP64: NRMSE(O)=1.375e-03 max|dO|=1.005e-02 -> FAIL

CONCLUSÃO (sem atribuir causa sem prova):
- A FA NATIVA KVarN, quando todos os operandos estão em host e sobre os mesmos
  valores decodificados, concorda com a atenção FP64 a ~5e-7 de NRMSE nos dois
  configs. Ou seja: neste recorte isolado ela NÃO é a fonte de erro.
- Isso CONTRARIA a atribuição anterior ("FA nativo KVarN local = gap de
  ~3e-3"), que não estava provada. A atribuição anterior (locmat vs off =
  KL 2.97e-3) permanece sem explicação de causa por este harness.
- O caminho "referência F16 existente" (materializar para F16 e usar a FA
  simples) é, neste recorte, o MENOS preciso: ~1.3e-3 de NRMSE e ~1.08e-2 de
  |dO| contra FP64 — ou seja, a própria referência não está dentro do gate.
- Perda de ESTAGIAMENTO (F16): ~2e-4. Perda de QUANTIZAÇÃO (4-bit em
  records): ~2e-2 (K) / ~4e-2 (V) com máximo absoluto de 6.1e-1 em V — esta
  última é a perda dominante no config B e é intrínseca ao formato.

LIMITES DESTE DIAGNÓSTICO:
- 1 camada, n_q=1 (decode), 512 chaves, máscara toda visível, sem tail exato,
  sem merge de faixas, sem Radeon/Vulkan, sem scheduler. NÃO cobre o caminho
  de prefill n_q=256, nem as caudas, nem o merge, nem a dispersão Vulkan.
- os valores de O estão no domínio ROTACIONADO (saída da FA nativa com q
  rotacionada); a comparação K/V foi feita no domínio original.
- não mede, nem prova nem refuta o gap de logits em nível de modelo
  (KL <= 1e-4) observado em OFF/ON e restored/cold.

EXIT STATUS: 1 (a "referência F16" falha NRMSE<=1e-3 e |dO|<=1e-2 no backend
CPU neste recorte; a FA nativa passa). Nenhum gate de F5/F7 foi declarado
concluído.

## MSG57-REALPATH-CAUSE (2026-10-11 19:59 UTC): SEGV DO CAMINHO REAL EXPLICADO
COM EVIDÊNCIA + GATE FP64/F16 NO BACKEND REAL + FIX DO waitSPID

Retomada canônica após falha de endpoint (servidor temporário 8197 encerrado
graciosamente, exit_code=0; PIDs antigos ausentes; sem inferência/waitSPID
ativo). Artefatos preservados: resp5.json, standalone5.log, cap.ref.n3000.logits
e log anterior do crash (server-prev-crash.log / standalone2-crash.log).
Nenhuma medição já concluída foi repetida.

### 1) SEGV da referência CPU no CAMINHO REAL: causa, com backing/ownership
Braço: router `ref` (LLAMA_OVF_ATTN_REFERENCE_CPU=1
GGML_CPU_KVARN_ALLOW_MIXED_DOMAIN=1), P=2048, REQLEN=3000, preset
campaign-try-102400.ini com ctx-size 4096, build d9db0b8 + guard fail-soft.
Rodada completa: http=200, 3840 linhas `ATTENTION-REFERENCE-CPU` (16 camadas x
240 pontos de instrumento), TODAS com `DECLINED`, zero SEGV.
Log: /home/hjotha/f4-spool/f5-campaign/attnref-sm-ref/server.log
Evidência por operando (exemplo, camada 3; idêntico nas 16):
    k_tail_local  [0] op=VIEW   ne=[256,4]    backing=cadeia resolvida até folha
                     buf=CUDA0 dev=CUDA0 host=0 off=0 data=0x7f13bd000000
    v_tail_local  [0] op=NONE   ne=[1024,129] buf=CUDA0 dev=CUDA0 host=0
    fa1_local     [1] op=PERMUTE ne=[256,2048] buf=CUDA0 dev=CUDA0 host=0
    fa1_local     [0] op=PERMUTE ne=[256,256]  backing=KVARN_WHT   buf=none data=nil
    fa2_overflow  [1] op=PERMUTE ne=[256,256]  backing=GET_ROWS    buf=none data=nil
    lse/lse2/o1   [0] op=FLASH_ATTN_EXT          backing=FLASH_ATTN_EXT buf=none
CAUSA RAIZ (prova real, não inferência):
- Os buffers de CACHE (stage, records e tail local) pertencem ao
  `llama_kv_cache_kvarn` e vivem no device CUDA0 (host=0; owner = backend de
  atenção local, não o nó). Os dados apontam para memória de device.
- Os operandos intermediários (WHT, GET_ROWS, ROPE, BLOC_SIZE, resultados) não
  têm buffer no momento da construção do grafo (alocados depois pelo scheduler).
- O scheduler NÃO insere cópia cross-backend para tensores cujos buffers são
  do cache (no build anterior ao guard as cópias inseridas eram
  `CPU#... [NULL]`, preservadas em server-prev-crash.log).
- A ancoragem no host (knob test-only, default OFF) encaminhava esses
  operandos para o kernel FA em CPU, que decodificava in-place via
  `ggml_fp16_to_fp32_row` sobre ponteiro de device -> SEGV (backtrace
  preservado: ggml_fp16_to_fp32_row <- ggml_compute_forward_flash_attn_ext).
GUARD IMPLEMENTADO (fail-soft, reversível): a ancoragem declina quando
qualquer src (Q/K/V/mask/bias + tail 5..11) ou o nó não tem cadeia de backing
em host, imprimindo o dump acima. Resultado: sem crash, produção preservada.
O modo de referência CPU fica INERTE (não produz números de referência) — não
declaro número nenhum a partir dele.

### 2) GATE FP64/F16 NO BACKEND REAL (CUDA0)
Mesmo harness diferencial, agora selecionando o backend (cpu|cuda) e lendo os
tensores de cache com `ggml_backend_tensor_get` (nunca via `t->data`).
Origem: /tmp/opencode/kv-cpu-diag/differential-cpu-ref.cpp.
Config A (stage_groups=5, produção; tail_groups=4):
    CUDA: FA nativa vs FP64   NRMSE(O)=8.967e-05 |dO|=8.277e-04 |dLSE|=3.913e-05 PASS
          FA F16 vs FP64      NRMSE(O)=1.063e-07 |dO|=6.815e-07 |dLSE|=6.319e+00
    CPU : FA nativa vs FP64   NRMSE(O)=4.462e-07 |dO|=2.750e-06 |dLSE|=6.357e-07 PASS
          FA F16 vs FP64      NRMSE(O)=1.254e-03 |dO|=1.077e-02 |dLSE|=3.865e-05 FAIL
Notas de rigor:
- O |dLSE|=6.319e+00 da FA F16 no CUDA NÃO é erro numérico: os 24 elementos de
  LSE saíram ZERO (d = -ref, ratio 0). A FA nativa, no MESMO backend e com a
  mesma instrumentação, produz LSE válido (|dLSE|=3.9e-05 contra o FP64). Logo
  o caminho F16 neste harness não escreve LSE no CUDA; o veredito de LSE é
  inconclusivo, e o gate do CANDIDATO continua válido.
- Em CPU a FA F16 acumula V em FP16 e por isso é MENOS precisa que o candidato
  (1.254e-03 > 1e-3; |dO| 1.077e-02): a "referência F16" não pode servir de
  referência estrutural no gate.
- Decodificação K/V (domínio original) no CUDA: NRMSE 2.098e-04 (K) e
  2.194e-04 (V); config B (records 4-bit): 2.107e-02 / 4.228e-02.
- Variação entre execuções no CUDA: NRMSE 8.967e-05 e 9.628e-05 (duas rodadas),
  ambas dentro do gate.
CONCLUSÃO: o candidato (colocação de produção, FA KVarN nativa) passa o gate de
§4.1 de atenção nos DOIS backends; a referência tem de ser a FP64
teacher-forced (definição canônica do gate), não o caminho F16.

### 3) HARNESS: waitSPID com timeout/cleanup específico (reversível)
Arquivos (todos em /tmp/opencode, fora do repo): attnref-standalone.sh,
attnref-gdb.sh, f5-attnref-sm.sh, f5-tailfix-sm.sh (cleanup da trap) e
canonical-newproc.sh (stop_srv).
Correção: funções `spid_wait`/`stop_srv` agora limitam a espera com
`timeout N tail --pid=<pid> -f /dev/null` e, no estouro, matam EXATAMENTE
aquele PID (mensagem explícita `waitSPID: budget ... exceeded for pid ...`),
depois `kill -9` só dele. Nunca kill amplo.
Validação (3 casos, /tmp/opencode/spidwait-test.sh):
    saída rápida            -> rc=0
    sleep 300 com budget 3s -> rc=124, pid eliminado, mensagem impressa
    pid vazio               -> rc=0
Registro desta seção: um `pkill -f waitSPID` case casou com o próprio shell que
o chamava e o matou — por isso ampliações de kill estão fora.

### 4) STATUS DOS GATES (F5) E DO CANÁRIO (F7) — SEM MEDIR O JÁ CONCLUÍDO
- §4.1 atenção (NRMSE <=1e-3, |O| <=1e-2, LSE <=1e-2): FECHADO para o
  candidato, com evidência em device real (CUDA) e host (CPU), seção 2.
- p95 <= 122.79435 s: FECHADO desde MSG57-BIS (62.8/63.3 s); não remédico.
- KL (média <=1e-4, p99 <=1e-3, FP64 teacher-forced): ABERTO. O braço attnref
  ficou inerte (seção 1), logo NÃO produziu logits; a falha de KLD de
  MSG57-ITEM3 permanece sem explicação de causa; a referência a usar é a FP64
  teacher-forced, e a "estrutural F16" está refutada como referência.
- SEGV: causa comprovada no caminho real; guard fail-soft elimina o crash sem
  tocar cache/scheduler/kernels de produção.
- F7 (A/B >=5 reps, fill 131072, canário, rollback, restore da produção 8090):
  segue BLOQUEADO pelo gate de KL aberto.
- Produção (llama-server-root.service): PARADA desde 2026-10-09T20:22:23Z.
  Nenhum push/merge; worktree suja; build-camp contém o guard.

### 5) LIMITES EXPLÍCITOS
- O modo de referência CPU não consegue produzir referência com o cache atual:
  os buffers vivem no device e o scheduler não materializa cópia em host para
  tensores do cache. Para funcionar seria preciso copiar (ou ancorar no device
  dono dos buffers) — mudança em cache/scheduler, fora do escopo reversível.
- O diferencial é sintético: 1 camada, n_q=1 (decode), 512 chaves, máscara
  toda visível, sem tail exato, sem merge de faixas, sem Vulkan. Não cobre
  prefill n_q=256, caudas, merge nem a dispersão Vulkan.
- LSE do caminho F16 não verificável no CUDA neste harness (zeros); o LSE do
  CANDIDATO está verificado no CUDA.
- Execução em CUDA apresenta variação entre rodadas (8.967e-05 / 9.628e-05),
  ambas dentro do gate.

## MSG57-F5F7-RETOMADA (2026-10-11 23:10 UTC): HARNESS EXPLICADO, GATE DE ATENÇÃO
FECHADO COM EVIDÊNCIA REAL, GATE KL ABERTO COM O LOCAL ISOLADO

Retomada de F5/F7 a partir de MSG57-REALPATH-CAUSE. Produção segue PARADA
(llama-server-root.service, desde 2026-10-09T20:22:23Z); sem push/merge.

### 1) OS CRASHES DO HARNESS EXPLICADOS (antes de confiar nos números)
Nenhum crash ocorreu na medição: os três ocorreram todos dentro do BLOCO DE
CONTROLE que eu acrescentei nesta sessão (um FA puro em F32, extra).
  a) rc=139/ctrl2 (SIGSEGV): o controle escrevia direto em `t->data` e os
     tensores nem estavam alocados (data == NULL) sobre buffer CUDA; `t->data`
     é ponteiro de DEVICE. Mesma classe de defeito do SEGV do caminho real.
     Correção: `ggml_backend_alloc_ctx_tensors_from_buft` + `tensor_set/_get`.
  b) ctrl4 (rc=134, SIGABRT): `GGML_ASSERT(ggml_can_mul_mat(k, q))` falhou
     (ggml.c:5628) porque eu criei K como [D, n_kvh, n_kv]. Neste fork a FA usa
     q [D, n_q, n_head_q, 1] e k/v [D, n_kv, n_head_kv, 1] (via reshape+permute
     0,2,1,3), o que exige `n_kv % n_q == 0`.
  c) terceiro rc=139 (SIGSEGV dentro de ggml_compute_forward_flash_attn_ext,
     CPU, confirmado por gdb): persistiu com o operando montado à mão. O
     experimento foi descartado por desnecessário: a pergunta que ele faria já
     é respondida DENTRO da mesma rodada — a LSE de fa_nat sai POPULADA
     (concorda com FP64 a 3.9e-05) e a de fa_mat sai com TODOS os 24 valores
     ZERO no mesmo backend; logo o LSE não escrito é artefato do caminho F16
     materializado, não do backend, e a LSE do CANDIDATO está verificada.
Consequência: nenhum número de medição é afetado; os crashs são do controle.

### 2) HARNESS CORRIGIDO E REGRESSÃO (n_q=1)
Bug real encontrado ao validar o caminho representativo: o `q` do tensor é
[D, n_q, n_qh, 1] enquanto a SAÍDA da FA é [D, n_head_q, n_q, 1]; meu
preenchimento/referência usava o layout da saída — só coincidente para n_q=1.
Corrigido. REGRESSÃO n_q=1 idêntica aos números já documentados:
CPU config A: NRMSE(O)=4.462e-07 |dO|=2.750e-06 |dLSE|=6.357e-07 -> PASS
CPU config B: NRMSE(O)=6.309e-07 |dO|=3.887e-06 |dLSE|=1.055e-06 -> PASS
(ou seja, a edição não mudou medição concluída; harness determinístico).

### 3) GATE §4.1 (ATENÇÃO) FECHADO COM EVIDÊNCIA NO BACKEND REAL
Referência FP64 validada no caminho representativo (prefill n_q=256, n_kv=512;
cauda exata com visível=472; decode n_q=1), mesmo cache/routing/campos:
  CUDA n_q=256 (config A/B, produção):  NRMSE(O)=1.011e-06/9.058e-07
      |dO|=4.885e-06/6.306e-06 |dLSE|=1.979e-06/2.163e-06  -> PASS
  CUDA n_q=256 visível=472 (cauda):    NRMSE(O)=9.495e-07/8.739e-07
      |dO|=5.734e-06/7.239e-06 |dLSE|=1.961e-06/2.198e-06  -> PASS
  CPU n_q=256:                         NRMSE(O)=5.490e-07/6.473e-07 -> PASS
  REFERÊNCIA F16 materializada no CUDA n_q=256: NRMSE(O)=1.165e-03,
      |dO|=1.539e-02, |dLSE|=6.319e+00 (zeros) -> FAIL (e |dLSE| inconclusivo)
VEREDITO: o CANDIDATO (colocação de produção, FA KVarN nativa) está ~1e-6
contra a referência FP64 em prefill, cauda e decode, nos dois backends —
dentro de NRMSE<=1e-3, |O|<=1e-2, LSE<=1e-2. A "referência estrutural F16"
medida é PIOR que o candidato no CUDA (1.16e-3 vs 1.0e-6) e tem LSE não
escrita: ela NÃO serve de referência. Gate §4.1: PASS para o candidato.

### 4) GATE KL — AINDA ABERTO; O LOCAL FOI ISOLADO (não relaxado)
Controles executados nesta sessão (P=2048, n=3000, via f5-attnref-sm.sh):
  - DETERMINISMO: mesmo braço (off) repetido -> KL=0.000000e+00,
    p99=0.0, dLSE=0.0 (bit-idêntico). O harness é reprodutível; portanto os
    gaps abaixo são reais, não ruído.
  - off vs pnat (GGML_KVARN_TEST_FORCE_PORTABLE_FATTN=1):
    KL=5.17798921e-02 p99=6.57363360e+00 dLSE=3.485e+00 -> FAIL. Os dois braços
    mostram `KV tail: ... route=native`, ou seja a rota portátil alterou o
    caminho principal; o caminho padrão é o que concorda com FP64 (item 3),
    logo a rota portátil NÃO serve de referência.
  - Medições anteriores (MSG57-ITEM3): locmat/mat vs off = 2.96826668e-03;
    off vs on = 9.78764989e-04. O knob GGML_KVARN_TEST_FORCE_MATERIALIZE_FATTN
    existe hoje apenas em ggml/src/ggml-cuda/fattn-kvarn-dispatch.cu:1227 e
    atua SOMENTE na entrada COMPACT_TAIL (cauda exata).
CONCLUSÃO (atribuição por knob, sem exagerar):
  - off vs on (9.78764989e-04): muda a COLOCAÇÃO da cauda local
    (LLAMA_OVF_TAIL_LOCAL_CUDA, default OFF) -> divergência no caminho da cauda.
  - mat/locmat vs off (2.96826668e-03): o knob atua SOMENTE na entrada
    COMPACT_TAIL (dispatch .cu:1227) -> divergência no kernel da cauda, e o lado
    materializado é o menos fiel já medido (1.16e-3 NRMSE em O no CUDA).
  - off vs pnat (5.17798921e-02): os dois braços exibem `KV tail: ... route=native`
    (17 ocorrências em cada), ou seja a rota portátil alterou o caminho
    PRINCIPAL; como o caminho principal padrão concorda com FP64 a ~1e-6
    (item 3), a rota portátil é que desvia — não serve de referência.
O gate KL<=1e-4/p99<=1e-3 continua ABERTO, com o local isolado nos caminhos
acima; o que falta especificamente é validar o kernel da CAUDA exata
(`KV tail: group=full ... representation=overlay`) contra a referência FP64,
que o harness sintético atual (máscara sobre a FA principal) ainda não
reproduz.

### 5) CONSEQUÊNCIA PARA F7 (regra: corrigir antes de bateria ampla)
- A/B >=5 reps, fill 131072, canário, rollback e restauração da 8090: NÃO
  iniciados — o gate de KL segue aberto e a rota portátil/materializada não
  serve de referência. Nenhuma interrupção de atividade não relacionada;
  nenhuma medição concluída foi repetida (a regressão n_q=1 é idêntica).
- Produção 8090 permanece parada. Sem push/merge.

## MSG57-F5F7-CAUDA (2026-10-12 00:40 UTC): CAUDA EXATA VALIDADA, DIVERGÊNCIA
DE KL LOCALIZADA NA ROTA DE PREFILL EM DOMÍNIO ROTATED_K_ORIGINAL_V

### 1) KERNEL DA CAUDA EXATA (`KV tail: group=full ... representation=overlay`)
VALIDADO CONTRA REFERÊNCIA FP64 (instrumento já existente)
Ferramenta: tests/test-position-split-real-pair.cpp (binário build-camp), que
constrói o ORÁCULO FP64 a partir dos BYTES REALMENTE armazenados no cache
(F16 do stage + records 4-bit), cobre body+tail, decode e prefill, overflow
Q4 (Vulkan), máscaras e o merge LSE estável de produção — e traz CONTROLES
ORACLE (duplicar/perder a cauda) para provar sensibilidade.
  SUÍTE COMPLETA: real-pair O/LSE: 8490 checks, 0 failures, 0 not-executed.
  Caso `NATIVE P=512 n=511 Q1 tail, empty ovf` (cauda exata de produção):
    local(CUDA/KVarN) NRMSE=1.033e-04 maxO=2.930e-04 maxLSE=3.225e-03
    merged  NRMSE(O)=1.033e-04 ; single-range-after-merge NRMSE=1.033e-04
    ORACLE duplicate-tail NRMSE(O)=4.509e-02 (deve exceder 1e-3) -> ok
    ORACLE dropped-tail    NRMSE(O)=4.774e-02 (deve exceder 1e-3) -> ok
Gate §4.1 (NRMSE<=1e-3, |O|<=1e-2, |LSE|<=1e-2): PASS com folga; os controles
provam que um erro real de cauda apareceria ~50x acima do gate.

### 2) ROTA DE REFERÊNCIA (CPU) MEDIDA — knob test-only adicionado
Patch mínimo e reversível em tests/test-position-split-real-pair.cpp:
`F4_RP_FORCE_CPU_FA=1` ancora a FA local na CPU (default OFF; produção
intacta). Mesmo caso acima, MESMO oráculo:
  local(CUDA/KVarN) NRMSE=3.977e-05 maxO=1.274e-04 maxLSE=5.023e-04
Ou seja: a rota CPU (referência estrutural do plano, "atenção CPU sobre o KV
dequantado da mesma representação por faixa, acumulação estável") é fiel a
~4e-5 e melhor que a rota CUDA (1.03e-4). As duas passam o gate §4.1.

### 3) DIVERGÊNCIA DE KL — ONDE ELA MORRE (diagnóstico por rota)
Tracing real das rotas (`GGML_KVARN_DEBUG_ROUTES=1`, braço `route`,
P=2048/n=3000, build d9db0b8+):
  decode  nq=1    : domain=rotated                 route=decode-split      (528)
  prefill nq=256  : domain=rotated_k_original_v   route=prompt-generic-mma (208)
  todos: entry=compact-tail, fallback=none, D=256 k=4 v=4 gqa=6.
  F5ROUTE: native=1 em 16/16 camadas, nrotmax=16, minctx=2048.
Braço `mat` (GGML_KVARN_TEST_FORCE_MATERIALIZE_FATTN=1):
  736/736 chamadas em route=materialize-fallback entry=compact-tail
  fallback=forced (o dispatcher CUDA recusa TUDO; a FA local cai na CPU).
KL no binário atual (f5-strict.py, FP64, full-vocab):
  CONTROLE determinismo: mesmo braço repetido -> KL=0.000000e+00 p99=0.0 (bit-idêntico)
  off vs mat  : KL=2.17741071e-03 p99=9.31687977e-01 dLSE=2.273e-01 FAIL
  off vs pnat : KL=5.17798921e-02 p99=6.57363360e+00 (rota portátil; refutada)
CAUSA LOCALIZADA (não é a cauda, não é o decode):
- A cauda exata (seção 1) e o decode em domínio ROTATED (seções 1-2 e o
  diferencial da seção anterior) estão validados contra FP64.
- O PREFILL local — que domina o harness de KL (200+ chamadas de n_q=256) —
  usa o domínio ROTATED_K_ORIGINAL_V (3) na rota prompt-generic-mma. NENHUM
  instrumento valida esse domínio: o real-pair declara em
  tests/test-position-split-real-pair.cpp:1191 que "the native CUDA KVarN FA
  route admits ONLY KVARN_DOMAIN_ROTATED" e usa só o domínio 1; o diferencial
  sintético (/tmp/opencode/kv-cpu-diag) também fixa o domínio ROTATED.
- Portanto o delta de KL de ~2.2e-3 (com |Δlogp| até 0.93) não é explicado
  pela precisão medida das rotas conhecidas (4e-5/1e-4 em NRMSE de atenção) e
  aponta para a rota de prefill em domínio 3 (ou para a semântica do V nela),
  que é justamente a que nunca foi comparada com FP64.

### 4) ESTADO DOS GATES (sem relaxar nada)
- §4.1 atenção (NRMSE<=1e-3/|O|<=1e-2/LSE<=1e-2): PASS nas rotas validadas
  (cauda exata, decode, prefill em domínio 1, overflow Q4, merge).
- KL (média<=1e-4, p99<=1e-3): ABERTO. Falta validar o prefill em domínio
  ROTATED_K_ORIGINAL_V e medir o KL do candidato contra a rota de referência
  CPU (3.977e-05, seção 2).
- F7 (A/B >=5 reps, fill 131072, canário, rollback, restore 8090): NÃO
  iniciado. Produção segue parada desde 2026-10-09T20:22:23Z. Sem push/merge.

### 5) PRÓXIMO PASSO CONCRETO (ordenado)
1. Estender tests/test-position-split-real-pair.cpp com o caso de prefill em
   domínio ROTATED_K_ORIGINAL_V (Q WHT-rotacionado; K pela view nativa rotacionada;
   V no domínio original; FA com op_params[domain]=3; oráculo FP64 com K rotacionado
   e V original) e medir as DUAS rotas (CUDA via `F4_RP_FORCE_CPU_FA` ausente e
   CPU com ele presente).
2. Se a rota CUDA em domínio 3 divergir do oráculo: corrigir localmente o
   kernel/rota (ou o plano que a escolhe) e revalidar; se só a CPU divergir,
   a referência passa a ser exclusivamente a CPU.
3. Com as duas rotas conhecidas, medir no modelo KL(off vs braço CPU puro), que
   exige um knob que empurre a FA local para a CPU SEM materializar F16
   (o `FORCE_MATERIALIZE` troca a implementação por F16, não só o backend) —
   hoje o caminho equivalente é ancorar a FA local na CPU pela mesma
   instrumentação de `LLAMA_OVF_ATTN_REFERENCE_CPU`, com o guard fail-soft.
4. Só então: A/B >=5 reps, fill 131072 e F7 (canário, rollback, recibo de
   downtime e restauração da 8090).

## MSG57-F5F7-CAUDA-FIX (2026-10-12 02:20 UTC): BUG DA CAUDA EXATA ENCONTRADO E
CORRIGIDO COM PROVA; GATES RESTANTES E BLOQUEIO EXPLÍCITO

### 1) INSTRUMENTO: casos de PREFILL em domínio ROTATED_K_ORIGINAL_V
Adicionados em tests/test-position-split-real-pair.cpp (default OFF, só rodam com
`F4_RP_DOMAIN_ORIGINAL_V=1`; tag `DOROV3`): D256/GQA24-4, local 512 com cauda
exata de 128, overflow 0/1/129, Q=1/8/256, body-only e tail-only. O oráculo FP64
é construído dos BYTES ARMAZENADOS: K rotacionado (body e cauda), V do corpo
DES-rotacionado (WHT inverso, auto-inverso) e V da cauda no domínio original —
exatamente o contrato `ROTATED_K_ORIGINAL_V`. `F4_RP_DORO3_TAILV_ROTATED=1`
permuta o V da cauda para testar a hipótese. `F4_RP_FORCE_CPU_FA=1` mede a
mesma geometria com a FA local na CPU.

### 2) BUG ENCONTRADO (cauda exata, domínio 3) E CORRIGIDO
O grafo passava o V da cauda ARMAZENADO (domínio original), mas o kernel aplica
ao V da cauda a MESMA des-rotação que aplica ao V do corpo — a contribuição da
cauda era transformada duas vezes.
  Antes (V da cauda original):  NRMSE(O)=2.155e-01  maxO=1.857e+00 (Q1)
                                  NRMSE(O)=8.220e-02  maxO=1.756e+00 (Q256 prefill)
  Depois (V da cauda rotacionado): NRMSE(O)=3.168e-04 maxO=4.297e-03 (Q1)
                                  tail-only: NRMSE=1.079e-05 maxO=1.542e-05
  Controles ORACLE (duplicar/perder cauda): 1.17e+00 / 1.17e+00 — sensibilidade ok.
CORREÇÃO (src/llama-graph.cpp, caminho único e split): rotacionar o V da cauda
SEMPRE (o kernel des-rotaciona em todos os domínios); antes era só em
`use_kvarn_rotated_domain`. Em domínio 1 a mudança é no-op (já rotacionava).
Efeito no harness de modelo (P=2048/n=3000, cauda overlay de 128 confirmada por
`KV tail: group=full@l3 ... representation=overlay`): KL(mat||off) caiu de
2.96826668e-03 para **7.09805178e-04** (4,2x melhor) — a cauda era a fonte
dominante do gap de KL.

### 3) GATE §4.1 — PARCIAL: o que passa e o que não passa
  Domínio ROTATED (decode + tudo que já era coberto): SUÍTE COMPLETA
    real-pair O/LSE: 9844 checks, 0 failures, 0 not-executed (com os casos
    DOROV3 rodando em domínio 1: Q256 prefill maxO=1.363e-04, Q1 tail
    maxO=2.914e-04, Q8 maxO=3.339e-04).
  Domínio 3 (prefill de produção, rota mma):
    Q1 tail 3.168e-04 | maxO 4.297e-03 | maxLSE 2.653e-03 -> PASS
    Q256   NRMSE 8.540e-04 | maxO 1.039e-02 | maxLSE 3.020e-03 -> FALHA |O| (4% acima de 1e-2)
    Q256 BODY-ONLY NRMSE 9.429e-04 maxO 1.327e-02 -> FALHA |O|
  Causa da falha restante: o loader de V em domínio original da rota mma
  grava o V des-rotacionado num tile half2 (arredondamento F16 extra antes da
  acumulação). A mesma geometria em domínio 1 passa por `portable-native`
  (NRMSE 3.977e-05, maxO 1.363e-04), MAS a troca de rota custa 3,2x no prefill:
    prompt eval time = 6475.58 ms / 3000 tokens (mma+domínio 3)
                      20874.33 ms / 3000 tokens (portable+domínio 1)
    decode inalterado (14.6 -> 14.4 tok/s).

### 4) GATE KL — AINDA ABERTO (sem relaxar)
Binário atual (com a correção da cauda), capturas full-vocab FP64 n=3000:
  controle determinismo (mesmo brafo repetido): KL=0.000000e+00, p99=0.0
  mat || off        = 7.09805178e-04  p99=5.66506418e-01  FAIL (<=1e-4 / <=1e-3)
  mat || d3fix      = 1.35521474e-02  p99=2.03482939e+00  FAIL
  d3fix || off      = 5.85377841e-03  p99=1.83508597e+00  FAIL
Leitura: a correção da cauda eliminou a maior parte do gap; o que resta está no
mesmo patamar do erro da própria MTODO DE REFERÊNCIA disponível (o braço `mat`
cai no caminho materializado em F16, medido no diferencial sintético como
NRMSE 1.165e-03 / |dO| 1.539e-02 no CUDA) e do resíduo de precisão da rota mma
de prefill (item 3). Nenhum dos dois lados está <=1e-4, então o gate não fecha.

### 5) BLOQUEIO E PRÓXIMOS PASSOS CONCRETOS
F5 não fechado; F7 NÃO iniciado (A/B >=5 reps, fill 131072, canário, rollback,
restauração da 8090 seguem bloqueados). Produção parada desde
2026-10-09T20:22:23Z. Sem push/merge.
  1. Prefill: ou (a) corrigir o kernel mma para não arredondar o V des-rotacionado
     para F16 antes da acumulação (o caminho window/dequant já usa parciais
     fp32; a escrita do tile V é o ponto), ou (b) aceitar `portable-native` no
     prefill e reescolar o gate de performance (3,2x). Protótipo reversível já
     existe e está medido: `LLAMA_KVARN_PREFILL_ROTATED_DOMAIN=1` (default OFF).
  2. Referência de KL fiel: um knob que mova a FA local KVarN para a CPU SEM
     materializar em F16 (o `FORCE_MATERIALIZE` cai no caminho F16, cujo erro
     próprio é ~1e-2), ou concluir o trabalho de scheduler/cópia do anchor
     `LLAMA_OVF_ATTN_REFERENCE_CPU` (hoje fail-soft declina por operando
     device-backed).
  3. Só então: A/B >=5 reps, fill 131072, canário, rollback e restauração da 8090.

## MSG57-F5F7-BASELINE-AB (2026-10-12 07:10 UTC): BASELINE INTACTO vs CANDIDATO
NAS MESMAS ENTRADAS (P93184/C131072) + ATINGIBILADE DO GATE KL

### 1) CONSTRUÇÃO DO BASELINE INTACTO
Binário de linha de base reconstruído revertendo APENAS os dois hunks desta
campanha (`llama-graph.cpp`: rotação do V da cauda; `llama-kvarn.cpp`: protótipo
env-gated) e compilando de novo; o candidato é o mesmo build-camp com os hunks.
Os dois conjuntos foram verificados: `llama-server` bit-idêntico, `libllama.so`
diferente (sha256 58688710 vs 2d95e965), carregado via `LD_LIBRARY_PATH`
(RUNPATH é absoluto). Nenhum outro arquivo do repo foi alterado.

### 2) A/B NAS MESMAS ENTRADAS (P=93184, C=131072, n=3000, 16 camadas)
Mesmo preset/rota (kvarn4 local + q4_0 overflow no Vulkan), mesma quantização
(k=4 v=4), mesmos env, captura full-vocab FP64 no checkpoint n=3000.
Cauda confirmada em ambos: `KV tail: group=full@l3 ... type=f16
representation=overlay` com 128 tokens em 16/16 camadas.
  KL(baseline || candidate)      = 8.28451510e-02  p99|dlogp|=8.49699933e+00  dLSE=5.279
  KL(mat || baseline)  PRÉ-EXISTENTE = 3.91324509e-01  p99=8.71474480e+00     dLSE=5.455
  KL(mat || candidate)            = 2.29148672e-03  p99=4.25424745e-01      dLSE=1.761e-01
  KL(mat_base || mat_cand) CONTROLE = 0.00000000e+00 (bit-idêntico)          argmax_same=True em todos
  Prefill: baseline 5295.26 ms / 3000 tok ; candidato 5252.73 ms / 3000 tok
  (diferença dentro do ruído; a correção não custa desempenho).

### 3) SEPARAÇÃO: O QUE ERA PREEXISTENTE E O QUE ERA REGRESSÃO
- NÃO há regressão do novo placement: o controle `mat_base || mat_cand` é
  bit-idêntico, e o candidato tem o MESMO prefill do baseline (±0,8%).
- A divergência grande era PREEXISTENTE e era BUG: no baseline intacto, a FA
  KVarN nativa (com a cauda F16) discordava da referência de mesma faixa por
  KL 3.91324509e-01 — a cauda com o V des-rotacionado duas vezes (seção
  MSG57-F5F7-CAUDA-FIX). A correção reduz essa divergência 170x
  (3.91324509e-01 -> 2.29148672e-03) com a MESMA referência e as MESMAS
  entradas, confirmando no nível de modelo o que o oráculo FP64 do real-pair
  já mostrava (NRMSE 2.155e-01 -> 3.168e-04).
- Portanto as attribuições anteriores ao "novo placement" estavam erradas: os
  gaps de KL medidos até aqui (2.97e-3, 7.1e-4) eram dominados pelo bug da
  cauda, não pela troca de placement.

### 4) FIDELIDADE DO ORÁCULO AO DOMÍNIO ROTATED_K_ORIGINAL_V
- O oráculo FP64 de atenção do real-pair É fiel ao domínio: K rotacionado
  (corpo e cauda), V do corpo DES-rotacionado (WHT inverso, auto-inverso) e V da
  cauda no domínio original — o contrato exato do domínio 3. Controles ORACLE
  (duplicar/perder cauda = 1.17e+00) provam que a cauda está sendo comparada.
- O braço `mat` (referência de modelo) NÃO é fiel ao domínio com cauda: seu
  trace de rota mostra `tail_type=none`, ou seja ele NÃO anexa a cauda F16 e lê
  as MESMAS chaves da cauda a partir dos records 4-bit. É uma referência de
  mesma faixa/mesmas chaves, porém com precisão menor na cauda — não um oráculo
  do caminho com a cauda F16.

### 5) KL<=1e-4 / p99<=1e-3: ATINGÍVEL? (evidência numérica)
NÃO é atingível para esta comparação, e a evidência é quantitativa:
  a) A referência disponível lê os 128 tokens mais recentes (a cauda) de
     records 4-BIT; o candidato os lê do overlay F16. O ruído 4-bit medido no
     diferencial sintético é NRMSE 2.107e-02 (K) e 4.228e-02 (V), com máximo
     absoluto de 6.1e-01 em V. Uma perturbação desse tamanho nas chaves MAIS
     influentes da distribuição do próximo token produz divergências de logits
     da ordem de 1e-3..1e-1 — e é exatamente o que se mede: p99|dlogp| = 4.25e-01
     para o candidato JÁ CORRIGIDO contra essa referência.
  b) O piso do próprio caminho aritmético: duas implementações independentes,
     ambas validadas contra o oráculo FP64 (real-pair NRMSE <=4.1e-05 no corpo
     Q256, domínio 1), divergem no nível de modelo por KL 5.854e-03
     (p99 1.84) — ~60x acima do gate de média, antes de qualquer diferença de
     representação.
  c) p99|dlogp| <= 1e-3 exigiria concordância de logits a 0,1%, abaixo da
     própria precisão F16 dos caminhos (a etapa de des-rotação do V no kernel
     mma grava em half2, ~5e-4 relativo por valor).
  d) O plano já classifica esses limiares como "metas iniciais, ainda não
     medidas" (docs/occupancy-placement-plan-20261002.md, seção de qualidade).
Decisão de escopo (NÃO tomei): os limiares de KL full-vocab precisam ser
reescolados (por exemplo, em termos de NRMSE de atenção ≤1e-3 — que É atendido —
ou com uma referência que também use a cauda F16, o que exige o trabalho de
cópia/anchor do `LLAMA_OVF_ATTN_REFERENCE_CPU`). Sem relaxamento automático.

### 6) ESTADO DOS GATES E PRÓXIMO PASSO VERIFICÁVEL
- §4.1 atenção: PASS nas rotas validadas (domínio 1 suíte completa 9844
  checks/0 falhas; domínio 3 com a cauda corrigida 3.168e-04; prefill domínio 3
  ainda falha |O| em 4% — ver MSG57-F5F7-CAUDA-FIX item 3).
- KL full-vocab: não atingível conforme item 5 (2.291e-03 no melhor caso
  atual, 23x acima do gate, com a referência menos precisa na cauda).
- p95/prefill: sem regressão (5252.73 vs 5295.26 ms).
- F7 (A/B >=5 reps, fill 131072, canário, rollback, restauração da 8090):
  BLOQUEADO pela decisão de reescopo do gate de KL + pelo |O| do prefill em
  domínio 3. Nenhuma tolerância foi relaxada.
PRÓXIMO PASSO VERIFICÁVEL: decidir o reescopo do gate de KL (item 5d) e, em
paralelo, corrigir a precisão do V des-rotacionado na rota mma de prefill
(evitar o arredondamento F16 antes da acumulação) — medindo de novo
KL(mat||candidato) no mesmo A/B canônico desta seção.

## MSG57-F5F7-REBOOT (2026-10-12 12:00 UTC): ESTADO PÓS-REBOOT, FERRAMENTAL
RECRIADO E VALIDADO, BLOQUEIO DE GPU

### 1) ESTADO APÓS O REBOOT (verificado, read-only)
- Produção: `llama-server-root.service` é unit de SISTEMA (`enabled`) e subiu
  sozinha às 07:08:46 CEST com `build-optimized/bin/llama-server` +
  `/home/hjotha/router-production.ini`, escutando em 8090 (PIDs 1605/1646,
  PPID 1). GPU: 11453/11902 MiB usados, 504 MiB livres. Nenhum processo meu
  está rodando e nada foi interrompido.
- /tmp foi zerado pelo reboot: os scripts descartáveis
  (`f5-strict.py`, `f5-attnref-sm.sh`, `f5-build-req.py`, o binário baseline)
  desapareceram. Tudo que importa está FORA de /tmp e foi preservado:
  árvore do repo (81 alterações, commit d9db0b8), `build-camp` com o candidato
  (`libllama.so` sha256 58688710f1394288 — idêntico ao de antes do reboot) e os
  artefatos da campanha em /home/hjotha/f4-spool/f5-campaign.
- Ferramental recriado e VALIDADO por reprodução: o `f5-strict.py` refeito
  reproduz exatamente os 4 números já registrados sobre as capturas
  preservadas (P=93184/C=131072, n=3000):
    mat || baseline   = 3.91324509e-01  p99 8.71474480e+00
    mat || candidato  = 2.29148672e-03  p99 4.25424745e-01
    baseline || cand. = 8.28451510e-02  p99 8.49699933e+00
    controle mat||mat = 0.00000000e+00
  O binário baseline foi reconstruído revertendo apenas os 2 hunos da campanha
  (revert verificado por assert no fonte); o candidato recompilado reproduz o
  hash anterior ao reboot (58688710), o que confirma que o build é
  determinístico para o candidato.

### 2) INVESTIGAÇÃO DO |O| DO PREFILL DOMÍNIO 3 — ESTADO (reproduzível)
Caso: `DOROV3 P=512 n=512 Q256 BODY-ONLY` (sem cauda), rota windowed
(`prompt-generic-mma`), domínio ROTATED_K_ORIGINAL_V.
  oráculo FP64 fiel (K rotacionado, V do corpo des-rotacionado):
      NRMSE=9.429e-04  maxO=1.327e-02  maxLSE=3.020e-03   -> falha |O| (4% acima)
  oráculo EMULANDO o arredondamento F16 do V des-rotacionado:
      NRMSE=9.410e-04  maxO=1.330e-02  maxLSE=3.020e-03   -> NÃO explica
  oráculo EMULANDO o arredondamento F16 dos pesos P:
      NRMSE=9.434e-04  maxO=1.359e-02  maxLSE=3.020e-03   -> NÃO explica
  controle Q1 (rota generic-mma, mesmo domínio): 4.584e-05 / 5.271e-04
  mesma geometria em domínio 1 (rota portable-native): 4.065e-05 / 1.230e-04
CONCLUSÃO PARCIAL: o erro NÃO vem do arredondamento F16 do V des-rotacionado nem
do arredondamento F16 dos pesos. Ele é específico da rota windowed em n_q>64
(domínio 3) — a mesma ordem do erro que o próprio código documenta em
ggml/src/ggml-cuda/fattn-tail.cuh ("prompt-generic-mma erra ~1e-3 acima do
gate"), só que a medição atribui esse erro à rota WINDOWED (domínio 3), não à
generic-mma: é a windowed que roda em domínio 3 (Q>1), e em domínio 1 as
predicadas rejeitam a generic-mma e servem portable-native (exata).
HIPÓTESES RESTANTES (ordenadas, ambas testáveis em 1 medição):
  (a) a KQ da MMA F16 arredonda Q para F16 -> erro sistemático de ~5e-4
      relativo nos escores, que explica os dois sintomas (LSE 3.02e-03 e
      O 1.327e-02). Teste: `F4_RP_EMULATE_Q_F16=1` no oráculo (arredondar o
      produto ponto a F16). Precisa de GPU.
  (b) erro de escala na des-rotação do V no kernel de dequantização da janela.
  Também explica um erro só de O, mas não o de LSE.

### 3) BLOQUEIO ATUAL (GPU)
A campanha precisa da GPU (real-pair exige CUDA+Vulkan; os braços de modelo
exigem ~6-11 GiB). A produção ocupa 11453/11902 MiB e sobram 504 MiB: nenhuma
medição da campanha cabe agora. Interromper a produção é uma ação de serviço
(não é sessão minha) e o plano exige que ela fique de pé durante F7 — por isso
a decisão de pará-la (com o procedimento documentado) ou de aguardar é do
usuário. NADA foi interrompido.

### 4) PRÓXIMO PASSO VERIFICÁVEL (assim que houver GPU)
1. Rodar `F4_RP_EMULATE_Q_F16=1` no caso Q256 BODY-ONLY em domínio 3: se o
   erro colapsar, a causa é a KQ em F16 da rota windowed e a correção é servir
   portable-native para o corpo em domínio 3 (a mitagem que o próprio
   fattn-tail.cuh já aplica quando a rota windowed não serve), ou introduzir
   KQ em F32 na janela.
2. Com a causa provada, medir de novo os gates §4.1 e o KL(mat||candidato) no
   MESMO A/B canônico (baseline reconstruído vs candidato, P=93184/C=131072).
3. Só então considerar A/B >=5 reps e F7.


## FINAL-20261012 — execução autorizada, oráculo e acumulação windowed

Registro UTC: 2026-10-12T09:30:51.119309+00:00. Árvore: feat/occupancy-position-split, HEAD d9db0b881
com integração oficial 58a162927 pendente de commit. Evidências próprias em
/home/hjotha/f4-spool/f5-campaign/final-20261012/.

- Usuário autorizou parar/iniciar produção durante a conclusão. Configuração,
  unit, bibliotecas e executáveis preservados em provenance/ e rollback-bin/.
  Reinício systemd confirmado (MainPID 296858, active); smoke real gerou 8
  tokens. Unit depois parada para GPU dedicada, sem alterar preset de produção.
- Model SHA256: e23eb251493b8a89f7d0217590b593e696ac9bacaeac8605541e477533dc5838.
- Regressões iniciais: 10/10 (mixed-state, stream, IO, placement, handoff,
  q4->KVarN, allocator ubatch/ranges e parâmetros FA).
- Probe Q F16 implementado como diagnóstico, sem substituir o gate FP64.
  Q arredondado reduz o erro LSE de 3.020e-3 para 3.098e-5, mas deixa
  maxO em 1.317e-2; K F16 não altera o resultado.
- Fixture domain3 da cauda corrigido para SEMPRE rotacionar V como o produto.
  O controle antigo sem rotação reproduziu o erro já registrado; não era uma
  regressão do merge oficial.
- KVarN D256 window partial: produtos tensor-core por tile preservados, mas
  acumulação/rescale entre tiles e publicação das parciais em FP32. Corpo Q256
  passou de NRMSE 9.429e-4/maxO 1.327e-2 para 1.179e-4/1.570e-3.
  real-pair DOROV3 completo: 1354 checks, 0 failures, 0 not-executed.
  Gates e tolerâncias originais inalterados.
- Harness F5 evidence v2: OOM explícito em cada run (inclusive warmup) e ciclos.
  Guard/host/swap continuam obrigatórios como observações, sem reprovação.
  Comparador persistente: KL FP64 por token, prefixos/proveniência idênticos,
  pelo menos 32 posições consecutivas; não usa p99 de elementos do vocabulário.
  Coletor persistente A/B sequencial com tempo por token server-side, ambos
  warmups e cinco pares. Ferramental CPU: 70 testes aprovados.
- Referência CPU com staging explícito de records/stage/cauda/Q/overflow em
  implementação; evita ponteiros CUDA no CPU e preserva LSE no produtor FA.
  Inclui caminho abaixo de P; ainda NÃO possui aceite de logits no modelo real.
- Próximo: rebuild final, referência 4B/27B teacher-forced, budget P real,
  campanha F5 inteira (qualidade/latência/capacidade/restore/10 ciclos), F7.
  F5 e promoção F7 continuam abertos. Nenhum push/commit nem relaxamento numérico.

### FINAL-20261012 — primeiro ensaio do P máximo, ainda sem aceite F5

Modelo Swift IQ3_XXS/MTP já identificado; C=131072, P=102400, b/ub=512/256, MTP off, CUDA KVarN4 e overflow Vulkan Q4_0 nas 16 camadas. Preset direto, cache frio, seed42, greedy/EOS ignorado; janela CUDA4096. Teste executado em paralelo à compilação diagnóstica (concorrência declarada; não é medição A/B de aceitação).

Prompt130560 + saída512, HTTP200/saída0, OOM explícito=0. Prefill=556640.206ms, TTFT=556875.947ms, p95 server-side=211.192ms sobre 511 amostras. Sem afirmação de melhoria de performance.

Observações de memória (amostragem, não limites de aceite): {"sample_count": 665, "timestamp_start_utc": "2026-10-12T09:48:01.012230+00:00", "timestamp_end_utc": "2026-10-12T09:59:33.523949+00:00", "host": {"process_rss_peak_bytes": 9090506752, "mem_available_min_bytes": 566697984, "swap_used_peak_bytes": 6893187072, "swap_in_pages_delta": 3095190, "swap_out_pages_delta": 4963683}, "cuda": {"vram_used_peak_bytes": 12193890304, "vram_free_min_bytes": 342884352, "server_process_vram_used_peak_bytes": 12173967360}, "vulkan": {"gtt_used_peak_bytes": 2288922624, "gtt_free_min_bytes": 5658505216, "vram_used_peak_bytes": 527982592}}

Evidência bruta persistida em `/home/hjotha/f4-spool/f5-campaign/final-20261012/memory-max-P102400/`. O servidor HTTP não decodifica necessariamente o último token amostrado; isto não substitui o teste CAPI de ocupação C/C+1 nem restore/picos de coexistência. P ainda sujeito aos testes seguintes. Serviço recolocado no preset atual durante a compilação diagnóstica; sem promoção da posição-split.

Referência CPU: correções de unwrap PERMUTE/RESHAPE para KVARN_VIEW, conversão de K/V standard para F32 evitando quantização Q8 da consulta e acumulação half, e desativação de graph reuse para MIN_POS. Capture de identidade disponível com autosave off; P no capture deriva da configuração efetiva. Sintaxe v2 aprovada; validação real de logits ainda pendente.

### FINAL-20261012 — referência real 4B e falhas preservadas

A árvore diagnóstica anterior `build-camp` tinha NO_VMM=OFF e foi interrompida antes do fim. `build-camp-optimized` foi configurada com o gerador Ninja e as opções GGML/LLAMA da build optimized, inclusive NO_VMM=ON, exceto captura/contadores F4 ativados. Build servidor concluída com -j12; configurações/logs e hashes em `final-20261012/provenance/`. A promoção permanece bloqueada. Serviço voltou ao preset atual durante a compilação, respondeu smoke de8tokens e foi novamente parado para os ensaios.

Falhas reais da referência revelaram e levaram à correção de: (1) head_slices do KVARN_VIEW sobre stage persistente NONE, explicitado a partir da camada; (2) preenchimento de mask/plano da cauda condicionado à máscara geral descartada no grafo dual, agora feito com máscara LOCAL mapeada; (3) extent do planejador de cauda em overflow Q>1, agora dimensionado por tail_extent_queries e seleção da cauda persistente; (4) pular caudas inteiramente mascaradas antes de validar/carregar arenas ausentes, CPU e CUDA indexed. CPU recusa descritor KVarN inválido em vez de interpretá-lo como F16 com ponteiro NULL.

Regressão CPU `test-kv-tail-input` passou após corrigir o fixture para codificar estágio explicitamente. Cobre D128/256/512, descritor persistente, O/LSE, cauda NaN totalmente mascarada sem arena CURRENT e planner overflow Q2 com extent N+1; limites não foram alterados.

Comparação de streams CPU/GPU 4B, P512/C4096, posições496..527: KL média=1.7461489261591103e-4, p99=1.8950653717320373e-3, FAIL. Artefato `structural-4b-P512-final-result.json` preservado. Diagnóstico da primeira camada real (D256,NH16,KV4) na primeira posição496: Qcur idêntico; O NRMSE9.28586e-5, maxabs3.20554e-4; pós-WHT NRMSE9.28539e-5, maxabs3.90247e-4. Este diagnóstico não é aceite da rede.

Preparado runner CAPI de pares sobre prefixo GPU idêntico a cada posição: checkpoint -> CPU -> restore e prova byte-a-byte do arquivo nativo completo -> GPU; somente estado GPU avança. Primeira execução recusou emN496: partial restore preservou a célula de atenção acrescentada (495->496), embora RS/max híbrido voltasse494. Corrigido o uso da API para o padrão upstream: restore PARTIAL_ONLY seguido de seq_rm do sufixo. Falha e arquivos nativos originais mantidos; novo ensaio ainda pendente.

### FINAL-20261012 — pares exatos e precisão do decode D256

Runner FULL em um contexto: prefixo GPU n-1 -> CPU1token -> restore nativo FULL em arquivo -> prova exata byte-a-byte -> GPU mesmo token. 32posições496..527, P512/C4096 no4B real: todos os prefixos iguais, meanKL=1.0164847708392496e-4 FAIL, p99=5.674898814167133e-4 PASS. Arquivo `exact-paired-4b-P512-full/paired-result.json`. Nenhuma tolerância alterada; fonte/artefatos/hash do modelo registrados.

Q decode record-backed: eliminado arredondamento intermediário Q*scale antes de fold da escala do registro; zero-point usa QF32. Runtime comprovado: D256,NH16,KV4, bits4/4, KVARN_DECODE_SPLIT, split64,8warps,QT1,4blocos/SM. Primeiro pontoKL 4.821496215159882e-4 ->4.2744307923683314e-4, mas trajetória dos32prefixos mudou e meanKL=1.2850046397639597e-4/p99=1.4468676079838266e-3 FAIL. Artefato `exact-paired-4b-P512-query-fixed/paired-result.json` preservado.

Próximo protótipo sob validação: V fold D256/QT<=2, códigos inteiros exatos emhalf, probabilidades F32 e escala/zero-point emF32, operando MMA normalizado por máximo de escala para evitar under/overflow. QT4/D128/D512 e estágio/fallback preservam a rota anterior. Build -j12 em andamento; não há aceite de precisão/performance deste protótipo ainda. Captura F4 test-only estendida a tokens realmente decodificados na geração não especulativa, sem alterar o armazenamento de logits do cache.


### Continuação 2026-10-12 — V fold e referência real 27B

Build diagnóstico otimizado terminou com CUDA/Vulkan e -j12; opções de produção idênticas salvo captura/contadores F4. Suite DOROV3 atual: 1354 checks, zero falhas e zero não executados. Oito testes estáticos KVarN aprovados.

FULL same-prefix 4B P512/C4096, posições496..527: meanKL=0.0001938906785268722/p99=0.0016975077175032471 FAIL; preservado `exact-paired-4b-P512-v-folded/paired-result.json`. Não se relaxaram limites.

FULL same-prefix **modelo de produção Swift27B IQ3_XXS**, D256/NH24/KV4, P512/C4096, mesmas32posições: meanKL=4.761564611836065e-05/p99=0.00026122441814975425 PASS, todos os prefixos verificados byte-a-byte. `exact-paired-27b-P512-v-folded/paired-result.json` e manifesto de artefatos registram dados reais. Não substitui gates nos comprimentos de produção nem torna 4B qualificado. Próximo P1024/1008..1039 em execução; serviço parado durante os experimentos, presets não promovidos.


### 2026-10-12 — atribuição da divergência de logits, sem alterar gates

27B P1024/C4096/1008..1039: meanKL=0.00031912671726047034, p99=0.0008708004465023159 FAIL na média. Rota portable F32 de diagnóstico: meanKL=0.00030763995638696573, p99=0.0013099024639165778 FAIL. Artefatos anteriores preservados.

Controle GPU/GPU FULL no mesmo prefixo,32pontos: todos os logits idênticos, KL=0/maxabs=0 e todos os prefixos byte-a-byte iguais. Modo diagnóstico explícito não elegível ao gate estrutural.

Dumps CPU vs portable no primeiro ponto n1008: l_out0..2 e Qcur3 exatamente iguais; FA3 NRMSE=3.817656588717581e-7/maxabs=1.19e-6; após primeira FFN l_out3 NRMSE=6.0603356784910835e-5, Qcur7=0.005286996932830239, l_out63 maxabs=1.3082428. Inferência em investigação: quantização das ativações nas projeções amplifica pequenas diferenças de attention. Não é declaração de gate aprovado nem de impossibilidade de alcançar o limite; controle pure-CUDA/CPU é o próximo passo.

Suite matemática standard atual:9844checks,0falhas,0nãoexecutados; regressões CPU de cauda também passaram. Revisão encontrou contraexemplos válidos de underflow/cancelamento no protótipo V fold; retirada do protótipo em andamento, preservando Q single-round e somas FP32 window. Sem promoção de preset.


### 2026-10-12 — decisão do usuário e correção do protocolo estrutural

Usuário inicialmente manteve o limite absoluto e depois explicitamente autorizou erro adicional sobre a baseline: meanKL candidato <= meanKL baseline +1e-4 e p99 candidato <= p99 baseline +1e-3. O/LSE e qualidade contra F16 permanecem inalterados. Plano §4.1 atualizado; baseline deve ser medida no mesmo comprimento e condições, nunca constante copiada do controle curto.

Controle pure-CUDA portable F32, Swift27B n1008..1039/C4096/P0: meanKL=0.0001893468900284161, p99=0.0004267619255334884. É diagnóstico, não aceite de baseline de produção.

Revisão adicional identificou que os runners anteriores verificavam integralmente o prefixo histórico N-1, mas CPU e GPU recomputavam K/V diferentes do token atual nas camadas posteriores. Os resultados anteriores permanecem como diagnósticos end-to-end de um passo; inclusive o PASS numérico P512 não prova o gate literal mesmo-KV. Novo protocolo de memória fixa captura Kcur/Vcur GPU por camada full antes dos stores, injeta os mesmos bytes no replay CPU, verifica igualdade e restaura o post-N GPU antes do próximo ponto. Implementação/testes ainda pendentes.

Build-optimized final após retirar V fold concluído: CUDA+Vulkan, -j12, server/cli/bench/perplexity e ferramentas de capacidade. F4 real27B primeira rodada:11legs completos/2falhas de harness (regex da mensagem P e ctxdest==P inválido), sem promoção. Identificado item ainda ausente: importação split->CUDA puro abaixoP; implementação transacional e novos testes em andamento.


### 2026-10-12 — F4 inverso e estabilidade na fronteira

F4 `f4-scenario-d-27b-inverse-v3`:14/14legs completos,0falhas/0nãoexecutados. D4a split->pure abaixoP em processo novo: fullvocab pós-tokenKL0/maxabs0/tokensiguais contra controle sameP restaurando arquivo idêntico; D4b overflow ocupado recusado com P/extent atribuíveis; D5destcapinsuficiente recusado; D8diskdiscovery abaixoPcontinuaaprovado. 129CPUharnesscases passaram.

Referência memóriafixa real27B,32pontos1008..1039: candidato P1024/C4096 meanKL0.00023936239426215495/p990.0006469041963810097; baselinepureCUDA meanKL0.00024253609567873285/p99igual. ComparadorFP64independente recalculou ePASS nos deltasautorizados. Todos32×32tensoresKcur/Vcurcapturados/injetadosbyteexatos; prefixopré eGPU pósrestauradosbyteexatos. É controle curto, não aceita comprimentos de produção.

Capacidade realC131072/P102400:130560+512efetivamentedecodificados atéC eC+1rejeitadocomestadointacto(4450696127bytescomparados). Prefill618.859s; não é benchmark A/B. Falha posterior em restoredivergente emP:tokensiguais,2/4linhasdevocabdiferentes,maxabs0.34359777; ciclos10nãoexecutados. SemmarcadoresOOMmascampaignincompleta não declaraOOM0final. Dadosoriginais preservados.

ReproduçãoP1024/C4096boundary:reuseON falhaemP(maxabs0.18374157),reuseOFF passaP-128/P/P+128 comKL0/nativebytesexatos. Correção centralFULLrestore invalida gf_res_prev apenasviaio.on_commit após sucesso;PARTIAL_ONLYmantémcomportamento. CPUstaticrollbackpass; variantecorrigidareuseONpassouas3fronteirascomlogits/nativebytesexatos. Nova campanhaC131072/P102400/10ciclos emexecução(`capacity-P102400-fixed-final`). ProduçãopresetoriginalparadaduranteGPUexperimentos; sempromoção.

FerramentacapturaGPU-onlyparacomparaçãoF16deKV implementada; qualidade/performance nos4comprimentos aindaabertas. Manifestoperformance preparado12runs/caso(amboswarmup+5pares), runtimeclosure/sourcehashbindantesedepois; nãoexecutado.

### 2026-10-12 — overflow físico compacto, ainda em validação

A revisão de §3.1 encontrou uma diferença entre a capacidade lógica e a
alocação real: a metadata compartilhada forçava o companion Q4 a reservar C
linhas. Com C=131072/P=102400, o payload Vulkan era 2304 MiB, em vez dos
504 MiB correspondentes a C−P. Ownership e máscaras por faixa já existiam,
mas a redução física de 1800 MiB ainda não estava implementada.

A implementação mantém os IDs globais na metadata e usa CID−P no payload
compacto, inclusive stores, gathers, views e IO de snapshots. Layout nativo
v2 `layout_standard_qx_compact_rows=3` distingue o novo corpo; snapshots
incompatíveis são recusados antes do commit. O allocator, o codec e a
integração com o grafo ainda exigem testes reais após a compilação.

A rodada `capacity-P102400-fixed-final` com o allocator anterior foi
interrompida deliberadamente depois das três fronteiras aprovadas e dos
ciclos 1 e 2. Artefatos, telemetria e motivo estão preservados. Não é aceite
dos dez ciclos nem declaração de OOM=0 para a campanha completa. A
qualificação integral será repetida com o allocator compacto.

A primeira build diagnóstica compacta com -j12 terminou. Os testes CPU de
cauda e layout de prompt cache passaram. O novo teste de codec encontrou
uma fixture sem o blob obrigatório de metadata; corrigir a fixture sem
relaxar a validação. A revisão também encontrou probes/reserva que ainda
endereçavam linhas globais, em correção antes de executar o modelo real.

Correções subsequentes de probes, reserva e hashing de payload passaram nos
três testes CPU (codec, cauda e layout de prompt cache). Sete testes estáticos
KVarN passaram. Suítes matemáticas atuais: DOROV3 incluindo CPU, 2023 checks;
standard, 9844 checks; ambas sem falhas ou casos não executados. Os limites
originais de O/LSE continuam em vigor.

O primeiro ensaio real compacto confirmou C=4096/P=1024, 3072 linhas físicas
e 54 MiB de payload Vulkan; capturas abaixo de P tiveram zero dispatch/cópia
Vulkan. A mesma rodada (`f4-scenario-d-27b-compact-v1`) encontrou validação
faltante de layout3 no codec de streaming e foi interrompida, sem aceite F4.
O streaming agora aplica os mesmos limites de faixa, ocupação e ownership
antes dos callbacks; seu teste também compara bytes com o encoder em memória.
Quatro testes CPU passaram após essa correção. A rodada real v2 está em curso;
seu cenário D1 de save/restore já passou, mas a matriz inteira ainda está aberta.

### 2026-10-12 — compactação validada em processos reais; F5 ainda aberto

O corpus congelado do harness agora inclui todos os tamanhos de fixture D;
as fatias exigem comprimento exato e falhas de precondição entram nos dois
ledgers. A rodada v2 anterior não exercitou D0b/D3/D4b/D5 por corpus curto,
e fica preservada como incompleta. Com a correção (136 testes CPU),
`f4-scenario-d-27b-compact-v3` aprovou 14/14 cenários, sem falhas ou casos
não executados. Inclui importação explícita nos dois sentidos, descoberta
de disco e rejeições de P, capacidade, overflow e corrupção.

`build-optimized` compilada com -j12, CUDA e Vulkan ON, captura/contadores F4
OFF; server/cli/bench/perplexity e ferramentas auxiliares gerados. Nove
regressões CPU passaram. Na própria build de produção, DOROV3 (2023 checks)
e standard (9844 checks) passaram sem falhas ou casos não executados.
Hashes de bibliotecas e arquivo das fontes dirty estão em `final-20261012/provenance/`.

Controle C4096/P1024 `P1024-compact-v3`: C ocupada por 3584+512 decodes,
C+1 recusada com estado intacto; seis fronteiras P−129/P−128/P−1/P/P+1/P+128
e travessia contínua de 129 tokens com todos os floats do vocabulário e
estado nativo exatos; dois ciclos completos, sem OOM. Não qualifica os dez
ciclos nem as dimensões de produção. Falhas anteriores de nomes de arquivo
e do uso indevido do mínimo híbrido como mínimo de KV ficaram preservadas.
O teste consulta agora a tabela de atenção; o mínimo híbrido também inclui
o checkpoint recorrente.

A API híbrida com reserva recorrente zero recusa corretamente remover o
prefixo histórico. A faixa local vazia foi exercitada separadamente em
`P1024-empty-local-compact-v1`: clear apenas da atenção após prefill P,
156894356 bytes recorrentes exatos, decode explícito em P com 1024 holes
locais e CID P → linha física 0; dois FULL round-trips e replay de quatro
tokens com todos os logits e estado exatos. É fixture de atenção, não suporte
a remoção de prefixo na API híbrida de produção.

Controle estrutural 4B P512/P1024 de memória fixa em execução. KL calibrada,
qualidade F16, desempenho nos quatro comprimentos, C131072/10 ciclos e F7
continuam abertos. Nenhuma promoção de preset foi feita.

Controles estruturais 4B compactos concluídos: quatro braços nativos com
32 posições cada, provas de K/V atual e FULL antes/depois completas,
OOM=0. P512/496..527: baseline meanKL=9.587047252399845e-5,
candidato=9.761700343945755e-5, delta=1.7465309154591043e-6;
p99 de ambos=0.0008681202154548221. P1024/1008..1039: baseline
meanKL=0.00036610135858153976, candidato=0.0003901145234061023,
delta=2.4013164824562556e-5; p99 de ambos=0.000942311258629062.
Comparador FP64 independente aprovou os dois pares nos deltas autorizados.
Não são comparações de qualidade F16 nem aceitação dos comprimentos 27B.

`capacity-P102400-compact-final-v1` iniciou na build de produção final,
com telemetria por PID, sem compiler/GPU concorrentes. Geração real de
512 tokens, C+1, seis fronteiras, travessia de P e dez ciclos continuam
em execução; sem afirmação antecipada de aceite. A limpeza opcional
registra tamanhos/hashes e remove apenas scratch gerado pelo próprio
runner depois de comparações exatas e de encerrar seus consumidores.
Os snapshots de produção e arquivos de falhas não são removidos.

Auditoria posterior dos braços 4B encontrou que a preparação clonada manteve
o caminho `build-camp-optimized`, embora declarasse o bundle optimized. Os
valores numéricos e as provas são diagnósticos válidos, mas o binding de build
declarado não corresponde ao executável: esses resultados não qualificam a
build de produção. A repetição em `structural-4b-production` está preparada.
O runner agora exige que cada executável seja um membro canônico do closure
verificado antes de lançar qualquer braço e recusa caminhos de bibliotecas
de outra build do projeto (10 testes CPU passaram). Os comandos 27B longos
e a capacidade atual apontam para `build-optimized`.

O nome reconhecido do headroom é `GGML_CUDA_GRAPH_RECOVERY_HEADROOM_MB`.
Algumas preparações anteriores usaram MIB e o código aplicou seu default de
18 MiB, igual ao valor de produção; preparações futuras usam MB explicitamente.
Na rodada longa compacta a memória persistente Vulkan medida foi 528482304 B
(504 MiB), com 81793024 B de compute inicialmente reservado. Isto confirma
C−P físico, mas não antecipa os gates de estabilidade ou desempenho.

A rodada C131072 compacta já concluiu 130560+512 decodes e a rejeição C+1
com estado intacto. Até o checkpoint intermediário atual, P−129/P−128/P−1/P/P+1
passaram com logits e estado nativo exatos. P+128, travessia contínua e dez
ciclos continuam em execução; não declarar a campanha completa antes deles.
