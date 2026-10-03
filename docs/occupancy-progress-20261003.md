# Occupancy position-split — log de implementação (branch `feat/occupancy-position-split`)

Data: 2026-10-03. Máquina: gokaya (192.168.1.57 / .52). Plano:
`docs/occupancy-placement-plan-20261002.md` (revisão 2026-10-03 preservada de
`docs/occupancy-plan-review-20261003`, sha256 `cc5cb1b1...`).

## Baseline de produção (recuperável, antes da 1ª alteração)
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
- Restauração: `systemctl restart llama-server-root` + checagem `:8090/metrics`
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

## Fases 1-7 — pendentes (não declaradas concluídas)
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

## Desenho F2/F3/F4 (para implementar após o rebuild verde)
- F2a (ubatch/P, pequeno e seguro): campo `position_split_p` (0=desligado)
  em `llama_memory_hybrid` (+ setter), checado em `init_batch` APÓS montar
  `ubatches` e ANTES de `mem_recr->prepare`: para cada ubatch, min/max de
  `ubatch.pos[i*n_pos]`; se algum cruza P → `FAILED_PREPARE` com motivo
  explícito, sem mutação (prepare ainda não rodou). A janela recorrente
  (`1+n_rs_seq` juntos) é subcaso: se ela cruza P, o ubatch cruza P.
  P chega via cparams (contexto) — origem: perfil adaptativo §3.8.
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
