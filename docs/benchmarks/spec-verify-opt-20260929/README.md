# Otimizações no caminho de verificação especulativa — 2026-09-29

Tarefa: implementar e medir otimizações no caminho de verificação especulativa do
fork BeeLlama.cpp, preservando a correção, e reportar o ganho real em tok/s.

Repositório: `/home/hjotha/beellama.cpp` · branch `feat/dflash2-shadow-observation`
(exceção do AGENTS.md: sem worktree) · HEAD `3d4128354` + alterações locais preservadas.
**Nenhum commit/merge/push foi feito** (conforme a tarefa). As alterações pré-existentes
não commitadas (shadow-observation) foram preservadas integralmente.

Configuração de benchmark (idêntica nos dois lados): alvo
`Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf`, RTX 4070 eGPU/CUDA0, MTP n4, p_min 0.70,
DFlash auxiliar desabilitado, ctx 16384, batch/ubatch 64, 1 slot, KV q4_0,
flash-attn on, temperature 0, seed 42, 192 tokens, ignore_eos, cache_prompt off,
return_tokens on. APU 20 W, SCLK 2700 MHz, CPU com política controlada e readback
APERF/MPERF. Trava `/tmp/beellama-gpu-benchmark.lock`, porta 59595. Serviços
`llama-server-root.service` (8090) e `qwen35-4b-mtp-8092.service` (8092) parados
durante cada bateria e restaurados em `finally` (conferido).

## Baseline reproduzível

Antes de qualquer modificação, os binários foram preservados em
`/home/hjotha/beellama-baseline-spec-verify-20260929/` (`dl-bin/` = cópia de
`build-dflash-xbox-dl/bin`, `build-bin/` = plugins CUDA/Vulkan/ggml de `build/bin`),
com hashes em `BASELINE-HASHES.txt`. O rebuild do baseline a partir do source
inalterado produziu binários **byte-idênticos** (`REBUILT-BASELINE-HASHES.txt`),
confirmando build determinístico. O runner registra, por grupo, as bibliotecas
realmente mapeadas pelo processo (`/proc/<pid>/maps` + sha256) em
`loaded-libraries.json`, e os modos `snapshot` apontam `LD_PRELOAD`/`LD_LIBRARY_PATH`
exclusivamente para o snapshot — baseline e candidato nunca compartilham biblioteca
modificada (validado nos hashes de `loaded-libraries.json`).

## Evidência de profiling (distribuição de fases no alvo real)

Run `smoke/prof-v1` (GGML_MTP_PROF=1). Ciclo de decode médio (1 forward do alvo
verifica até 5 tokens):

| Fase | Custo típico | Fração do ciclo |
| --- | ---: | ---: |
| Forward do alvo (decode, MTPTGT) | ~38–46 ms | ~75% |
| Draft MTP (phase=draft) | ~8,5 ms | ~17% |
| Catch-up MTP (phase=sync / MTPPROC) | ~3,2 ms | ~7% |
| Verify (sampling/accept, phase=verify) | ~1,3 ms | ~2,5% |

**Conclusão-chave:** o ciclo é dominado pelo forward do alvo (27B, matmul-bound na
eGPU). As otimizações do caminho de verificação atacam frações pequenas: item 3
(cópia de logits) ≤ ~3% (teto do phase=verify), item 4 (catch-up) ≤ ~1% (fração
rejeitada de 7%). O único item que ataca o forward dominante é o item 6 (GDN, ~75%
das camadas são recorrentes — `full_attn_interval=4`).

## Resultados

### Itens 1, 2 e 5 — implementados e medidos

Bateria `paired/paired-item125-v1`: 5 modos × 5 repetições pareadas por cenário
(curto/código/prosa/longo), ordem alternada, mesmo binário com toggles por env.
**0 divergências de tokens em todas as comparações** (greedy preservado).

| Mudança | Δ tok/s pareado (mediana) | Faixa (20 pares) | Veredito |
| --- | ---: | --- | --- |
| base-snap vs base-live | +0,05% | −0,44…+0,57% | equivalência flag-off == baseline ✓ |
| **Item 1** — lazy snapshot + fix ownership do clone | **+0,42%** | +0,07…+0,73% (todos positivos) | **mantido** |
| **Item 2** — single sync ( Outputs já sincronizados) | −0,11% | −0,95…+0,37% | **neutro** (mantido, flag-gated) |
| Item 1+2 combinados | +0,44% | −0,51…+1,26% | dominado pelo item 1 |
| **Item 5** — métricas/separação de fases + fix extrapolação | n/a (instrumentação) | — | **mantido** (correção/relato) |

Medianas por modo (decode tok/s): ver `analyze.py paired/paired-item125-v1 base-live opt1 opt2 opt12 base-snap`.

- **Item 1** (`SPEC_OPT_LAZY_SNAPSHOT`, `tools/server/server-context.cpp` +
  `common/sampling.cpp`): o snapshot do sampler/loop-guard passa a ser condicionado
  à possibilidade real de restore de checkpoint, decidida ANTES de amostrar com o
  rollback máximo (`n_draft`) via `server_speculative_rollback_requires_checkpoint`
  (monotônico → upper-bound válido). No benchmark (RS, n_rs_seq=4, n_draft≤4) o
  restore é inalcançável, eliminando ~745 clones de sampler por bateria. Ganho
  consistente de +0,42%. **Correção de ownership associada:** `common_sampler_clone`
  agora reponta `cur_p.data` para o buffer `cur` do clone (antes apontava para o
  `cur` do sampler de origem — leitura de candidatos do clone antes de um novo
  `sample()` podia observar memória alheia/invalidada), espelhando `common_sampler_copy`.
- **Item 2** (`SPEC_OPT_SINGLE_SYNC`, `src/llama-context.cpp` + `src/llama-context.h`):
  novo membro `outputs_synced`, limpo apenas no único ponto de submissão async
  (`graph_compute` → `ggml_backend_sched_graph_compute_async`) e marcado após a
  barreira. `synchronize()` pula a barreira redundante quando nada foi submetido
  desde a última — os múltiplos getters de output dentro de um mesmo passo de
  sampling não re-disparam a barreira. **Não** usa `n_queued_tokens==0` como prova
  (rastreia a barreira do scheduler diretamente). A barreira do servidor após
  `llama_decode` permanece (outputs_synced=false após graph_compute).Resultado:
  **neutro** (−0,11%, dentro do ruído) — confirma a cautela da revisão de que a
  maioria das barreiras repetidas já encontrava o dispositivo pronto. Mantido
  flag-gated (redução de barreiras inofensiva, sem regressão).
- **Item 5** (`tools/server/server-context.cpp` + `tools/server/server-adaptive-dm.h`):
  (a) `verify_ms` documentado como abrangendo APENAS sampling/accept/trim — o
  forward do alvo (MTPTGT) e o catch-up (phase=sync) ocorrem antes de `t_verify_start`;
  adicionados sub-timers `sample_accept_us` e `rollback_us` (gated por prof, sem sync
  adicional) e fase `verify_rollback` separada. (b) **Fix de correção na extrapolação
  do controlador DFlash:** `est_cycle = n*draft_per_token + (ref.cycle_ms − ref.draft_ms)`
  em ambos os sítios — antes usava `+ ref.verify_ms + ref.accept_ms`, omitindo o custo
  dominante do forward do alvo (cycle_ms já o inclui). Instrumentação neutra em
  desempenho (prof off no benchmark); fix de extrapolação é melhoria de correção para
  usuários do DFlash adaptativo (inativo no modo MTP do benchmark).

### Item 6 — GDN CUDA (leitura indexada dos estados recorrentes, src[6])

**Implementado.** O backend CUDA rejeitava explicitamente `GGML_OP_GATED_DELTA_NET`
com `src[6]` (variante rows-indexed), disponível só em CPU/Metal. Implementado:

- **Kernel** (`ggml/src/ggml-cuda/gated_delta_net.cu`): `gated_delta_net_cuda` recebe
  `const int * rows` + `int64_t state_row_size`. O offset de leitura do estado passa a
  `rows ? rows[sequence]*state_row_size + h_idx*S_v*S_v : sequence*H*S_v*S_v + h_idx*S_v*S_v`,
  espelhando exatamente a referência CPU (`ggml-cpu/ops.cpp:11341-11347`) e o kernel
  Metal. A escrita do estado (tail do dst, indexada por `sequence`) e o layout por-head
  `[S_v,S_v]` são idênticos à forma gather — só a leitura de entrada difere. Propagado
  pelo launcher (4 casos S_v) e pela macro `GDN_LAUNCH`.
- **`supports_op`** (`ggml-cuda.cu:6284`): removida a rejeição de `src[6]` (mantida a
  rejeição MUSA). O fusion matcher `ggml_cuda_try_gdn_cache_fusion` não inspeciona
  `src[6]` (casa o tail de saída + CPY), então a fusão de cache continua válida.
- **Gate** (`src/models/qwen35.cpp:160`): `gdn_state_rows_dev_ok` permanece true para
  CUDA **opt-in** via `GGML_GDN_CUDA_ROWS=1` (default = gather, seguro). `GGML_GDN_STATE_GATHER`
  continua kill-switch. Rows exige `cparams.n_rs_seq>0` (MTP n4 → n_rs_seq=4).

**Build com reuso de objetos** (resposta à preocupação de recompilar tudo): um
`cmake .` regenerou `build.ninja` e invalidou o `.ninja_deps` ("stored deps info out
of date"), forçando rebuild completo. `touch`/cópia de `.o` **não** engana o ninja
(ele valida o deps-info gravado). Como as mudanças reais (`gated_delta_net.cu`,
`ggml-cuda.cu`) já estavam compiladas e as 18 instâncias fattn-mma-kvarn/mmq faltantes
são código ggml estável (fontes inalteradas desde 18/09), elas foram **copiadas de
`build-dflash-xbox/`** (mesmo toolchain nvcc 13.3/gcc 15.3, Release) e a lib foi
**linkada manualmente** (`g++-15 -shared` com a lista de 238 objetos do `ninja -t
commands`), pulando ~10 min/instância. Arch `89` (dos copiados, com PTX) vs `89-real`
(do build, só SASS): o SASS sm_89 é idêntico e a RTX 4070 é sm_89, então o PTX extra é
inofensivo. Lib final 424 MB, hash `0c50c5ee…`, ELF/soname/símbolos OK.

**Correção validada** (`smoke/smoke-item6-v1`, EXIT=0, `TOKEN_DIVERGENCES=0`): hashes
de tokens **idênticos nos 3 modos × 4 casos** — base-snap (lib baseline 25/09),
gdn-gather (lib nova, default) e gdn-rows (`GGML_GDN_CUDA_ROWS=1`):

| caso | base-snap | gdn-gather | gdn-rows |
| --- | --- | --- | --- |
| curto192 | `2a125889` | `2a125889` | `2a125889` |
| codigo192 | `c784f8ad` | `c784f8ad` | `c784f8ad` |
| prosa192 | `7c689d5f` | `7c689d5f` | `7c689d5f` |
| long192 | `351021b3` | `351021b3` | `351021b3` |

`gdn-gather == base-snap` prova que os 18 objetos copiados + o link manual + a mudança
em `gated_delta_net`/`supports_op` são sadios (gather idêntico ao baseline).
`gdn-rows == gdn-gather` prova a correção do kernel rows. O server carregou em rows sem
crash (assert de contiguidade de `src_state` passou) e manteve ~57–85 tok/s — se o op
rows caísse em CPU (3/4 das camadas são GDN), o tok/s desabaria; logo **rows roda em CUDA**.

**Desempenho** (`paired/paired-item6-v1`, 5 repetições pareadas por cenário, ordem
alternada, mesma lib nova com toggle por env, `TOKEN_DIVERGENCES=0`):

| Cenário | gdn-gather tok/s | gdn-rows tok/s | Δ pareado (mediana) | Faixa (5 pares) |
| --- | ---: | ---: | ---: | --- |
| curto192 | 87,074 | 85,129 | −2,44% | −2,91…−1,65% |
| codigo192 | 60,241 | 59,359 | −1,87% | −2,39…−0,56% |
| prosa192 | 57,818 | 56,820 | −1,79% | −2,32…−0,42% |
| long192 | 58,778 | 57,894 | −1,76% | −2,17…−0,98% |
| **ALL** | — | — | **−1,79%** | 20/20 pares negativos |

**Conclusão do item 6:** o kernel rows está correto e roda em CUDA, mas é ~1,8%
**mais lento** que o gather neste benchmark. Causa: com `--parallel 1` (n_seqs=1) o
gather por camada já é uma cópia mínima (um único estado de sequência), enquanto o
modo rows acrescenta indireção (`rows[sequence]` lido da global memory por bloco,
`state_row_size` dinâmico) sem eliminar trabalho relevante — o overhead de indexação
supera a economia do gather. O gather só se torna caro com `n_seqs` grande (decode
batch multi-sequência), onde a cópia densa escala e a leitura indexada pagaria; esse
cenário está fora do escopo deste benchmark (single-slot). **Decisão: manter gather
como default; `GGML_GDN_CUDA_ROWS` permanece opt-in** para investigação futura em
batch multi-sequência. Implementação, validação de correção e benchmark próprios
entregues e isolados, como a tarefa pede.

### Item 3 — verificador greedy retornando somente IDs

**Decisão: não enviado nesta sessão; análise de escopo baseada em evidência.**

A infraestrutura parcial já existe e foi mapeada:
- `build_sampling` (`src/llama-graph.cpp:5461-5468`) já tem um short-circuit ID-only
  para DSpark greedy: quando a cadeia é um único sampler greedy, publica apenas
  `res->t_sampled[row]` (IDs) e faz `continue`, pulando a publicação de
  logits/probs/candidates.
- `llama_sampler_greedy_backend_apply` (`src/llama-sampler.cpp:1076-1091`) já publica
  só `data->sampled = ggml_argmax(...)` e **não** seta `data->logits`.

Os três gaps reais (como a própria tarefa descreve) e por que a conclusão segura é
all-or-nothing no caminho compartilhado:
1. **`build_sampling` publica o tensor de logits:** `data` é inicializado com
   `data.logits = logits_seq` (view do vocabulário) antes de `backend_apply`; como o
   greedy não limpa `data.logits`, o bloco `if (data.logits != nullptr)` publica
   `t_sampled_logits` (vocab inteiro). Limpar isso exige garantir que nenhum outro
   consumidor da cadeia precise de logits.
2. **`needs_raw_logits`** (`src/llama-context.cpp:2525-2540`) só retorna false quando
   **toda** seq de output tem backend sampler. No benchmark `backend_sampling=false`,
   então a cópia `n_outputs×n_vocab` (4,97 MiB/rodada) acontece em
   `2820-2832`. Para evitá-la é preciso **habilitar backend sampling no alvo** —
   mudança de comportamento, não apenas de grafo.
3. **`set_logits`** (`common/sampling.cpp:137-171`) não tem ramo "ID-only": se o
   backend publicar só IDs (sem probs/logits), `sampled_probs`/`sampled_logits` são
   null e o código cai no ramo de logits crus — que não foram copiados
   (`needs_raw_logits=false`) → `GGML_ASSERT(logits != nullptr)` falha. O
   short-circuit de `common_sampler_sample` (`712-729`) roda **depois** de
   `set_logits` e ainda varre `cur_p` para achar `selected`. Completar exige um ramo
   ID-only em `set_logits` + consultar o token do backend antes de materializar.

**Teto medido:** a cópia de logits e a materialização vivem dentro do forward do alvo
(38 ms) + phase=verify (1,3 ms ≈ 2,5% do ciclo). Mesmo eliminando-as por completo, o
ganho é ≤ ~1–2%. **Risco:** (a) habilitar backend sampling no alvo muda o caminho de
sampling compartilhado por **toda** inferência (não só especulativa); (b) o argmax do
backend deve coincidir token-a-token com o argmax CPU greedy inclusive em empates
(tie-break), sob pena de divergência de saída; (c) a matriz de fallback exigida
(gramática, reasoning budget, penalties dependentes de histórico, logprobs/n_probs,
p_min do draft MTP, variante residual estocástica de `sample_and_accept_n` que
precisa de probs completas) é extensa. Dado o teto de ~1–2% contra o forward
dominante e o risco ao caminho compartilhado, a conclusão segura não se justifica
nesta sessão. Caminho concreto para completá-la: generalizar o short-circuit DSpark
para greedy elegível + ramo ID-only em `set_logits` + consultar o token antes de
materializar + `needs_raw_logits=false` quando coberto, tudo gated por elegibilidade
estrita da cadeia efetiva e validado por identidade de tokens contra o baseline.

### Item 4 — catch-up MTP do prefixo confirmado

**Decisão: não enviado; experimento adiado com evidência (a tarefa permite descartar).**

**Impedimento estrutural concreto:** `common_speculative_process` (catch-up que
decodifica `ctx_dft` com os hidden states reais do alvo) roda em
`server-context.cpp:9353`, **antes** do sampling/accept (9557+). Para decodificar
apenas âncora + tokens aceitos, o decode de `ctx_dft` precisaria ser adiado para
depois de `accept()` (que conhece `n_accepted`). Mas `process()` não tem um sinal
limpo de "este batch de verificação será seguido de accept": ele também roda no
**prefill**, onde `accept()` não é chamado — pular o decode ali deixaria o KV de
`ctx_dft` sem o prompt e o `draft()` seguinte geraria propostas erradas. Uma
implementação segura exigiria: (a) um sinal do fluxo do servidor distinguindo
prefill de verificação; (b) capturar `verify_h` em `process()` (antes do próximo
decode do alvo sobrescrever os nextn embeddings) e transferir o decode de `ctx_dft`
para `accept()`/pós-accept; (c) reconciliar com `draft_owns_state`, o `seq_rm` do
draft no bloco de checkpoint (8168-8210), o reset de KV do `draft()`, replay
(`spec_is_replay`), bootstrap e os caminhos chain_heads/mem_shared/multi-head.

**Teto medido:** `mtp_proc` ≈ 3,2 ms ≈ 7% do ciclo; apenas a fração rejeitada é
recuperável (16% das linhas no cenário código, 1–5% nos outros) → ≤ ~0,3–0,5 ms/ciclo
(< 1%). A revisão alerta explicitamente que adiar o decode pode **perder a
sobreposição com o sampling CPU**, podendo tornar o resultado líquido nulo ou
negativo. Dado o teto < 1%, o risco de corromper o estado do KV MTP (saída errada é
pior que não otimizar) e a impossibilidade de validar iterativamente sem o
sinal do servidor, o experimento não é enviado sem evidência medida de melhora —
exatamente o desfecho que a tarefa prevê ("se essa mudança piorar o resultado,
descarte somente essa alteração e registre a evidência"). O catch-up do prefixo
aceito continua usando os hidden states reais do alvo em todos os caminhos.

## Validação de correção

**Itens 1, 2 e 5 — bateria `paired/paired-item125-v1`:**
- 100 requisições medidas (5 modos × 4 cenários × 5 repetições), `state=complete`,
  `restoration_errors=[]`. Serviços 8090/8092 restaurados e ativos; CPU restaurada
  com readback (`cpu-restored.json` confere com `cpu-original.json`).
- **0 divergências de tokens** em todas as comparações (`token-comparisons.jsonl`,
  127 KB): within-mode-repeat e across-mode-same-repeat. Greedy (temp 0, seed 42)
  preservado token-a-token entre base-snap, base-live, opt1, opt2, opt12.
- Isolamento de bibliotecas conferido por `loaded-libraries.json` (sha256 de
  `/proc/<pid>/maps`): modos `snapshot` carregam exclusivamente as libs do snapshot;
  modos `live` carregam as do build. base-snap e base-live carregam libs de hash
  idêntico (build determinístico).
- Testes compilados e aprovados no build de benchmark: `test-server-prompt-checkpoint`
  (tabela-verdade de `server_speculative_rollback_requires_checkpoint`), `test-sampling`,
  e o teste estático `test-server-loop-guard-checkpoint-static.py` (atualizado para o
  novo invariante: decisão de snapshot com rollback máximo ANTES do sampling + assert
  `GGML_ASSERT(may_need_ckpt_restore && smpl_save != nullptr)` no ramo de restore).

**Item 6 — GDN CUDA:** validado por `smoke/smoke-item6-v1` (EXIT=0, `TOKEN_DIVERGENCES=0`,
hashes de tokens idênticos nos 3 modos × 4 casos) e `paired/paired-item6-v1`
(`TOKEN_DIVERGENCES=0` em 40 requisições). `gdn-gather == base-snap` prova que os 18
objetos reaproveitados de `build-dflash-xbox/` + o link manual + as mudanças em
`gated_delta_net`/`supports_op` são sadios; `gdn-rows == gdn-gather` prova a correção
do kernel rows. O server carregou em modo rows sem crash (assert de contiguidade de
`src_state` passou) e o tok/s não desabou → o op rows executa em CUDA (não em
fallback CPU). Desempenho: −1,79% (ver seção Item 6). Bibliotecas realmente
carregadas conferidas por `loaded-libraries.json` (sha256 de `/proc/<pid>/maps`):
modos `cuda=snapshot` carregam a lib baseline de `/home/hjotha/beellama-baseline-spec-verify-20260929/build-bin`;
modos `cuda=live` carregam `build/bin/libggml-cuda.so.0.23.0` (hash `0c50c5ee…`).

## Recomendação

**Configuração recomendada para mais tok/s nesta máquina (RTX 4070 eGPU, Qwen3.8-27B
MTP n4, single-slot, greedy): `SPEC_OPT_LAZY_SNAPSHOT=1` (item 1), GDN em gather
(default; `GGML_GDN_CUDA_ROWS` desligado).**

Resumo objetivo por item (ganho decidido por benchmark, não por contagem):

| Item | Estado | Δ tok/s (medido) | Recomendação |
| --- | --- | --- | --- |
| 1 — lazy snapshot + clone ownership | implementado, validado | **+0,42%** (20/20 pares positivos) | **ligar** (`SPEC_OPT_LAZY_SNAPSHOT=1`) |
| 2 — single sync | implementado, validado | −0,11% (neutro, no ruído) | opcional; inofensivo, sem ganho |
| 3 — greedy ID-only | não enviado | teto ~1–2% (análise) | ver decisão de escopo |
| 4 — catch-up MTP | não enviado | teto <1% (análise) | ver decisão de escopo |
| 5 — métricas + fix extrapolação | implementado | neutro (instrumentação) | **manter** (correção/relato) |
| 6 — GDN CUDA rows | implementado, validado | **−1,79%** (single-slot) | default gather; rows opt-in |

**Por que o ganho agregado é pequeno:** o profiling mostra que o ciclo de decode é
~75% forward do alvo (27B matmul-bound na eGPU, ~38–46 ms), ~17% draft MTP, ~7%
catch-up, ~2,5% verify. As otimizações do caminho de verificação (itens 1–4) atacam
no máximo as frações verify/catch-up/logits — cada uma com teto de ~1–2%. O item 1
(remove ~745 clones de sampler/bateria, cópia CPU de um vetor de n_vocab) é o único
ganho consistente mensurável (+0,42%). Nenhuma dessas mudanças acelera o forward
dominante; o item 6 ataca o forward, mas o modo rows elimina apenas o gather por
camada (não o cômputo GDN nem os matmuls), e para single-slot (n_seqs=1) o gather já
é barato — daí rows medir −1,79% (overhead de indexação > economia). O ganho líquido
recomendado nesta configuração é, portanto, o item 1 (+0,42%); item 2 é neutro e o
item 6 fica desligado por padrão.

**Itens mantidos ligados por padrão no código:** nenhum toggle muda o comportamento
default; `SPEC_OPT_LAZY_SNAPSHOT`, `SPEC_OPT_SINGLE_SYNC` e `GGML_GDN_CUDA_ROWS` são
opt-in. O fix de ownership do `common_sampler_clone` (item 1b) e o fix de extrapolação
do controlador DFlash (item 5b) são **incondicionais** (correções de correção, sem
toggle), pois não alteram a trajetória greedy e previnem leitura de memória dangling
e viés de estimativa, respectivamente.

**Estado final:** branch `feat/dflash2-shadow-observation`, HEAD `3d4128354`, sem
commit/merge/push (conforme a tarefa). Alterações locais preservadas. Binários:
`build-dflash-xbox-dl/bin` (itens 1/2/5 + gate qwen35) e `build/bin/libggml-cuda.so.0.23.0`
(item 6, hash `0c50c5ee…`). Baseline byte-idêntico preservado em
`/home/hjotha/beellama-baseline-spec-verify-20260929/`. Serviços 8090/8092 restaurados.
