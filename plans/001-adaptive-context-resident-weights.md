# Plano 001: alternar contexto e MTP preservando os pesos principais

## Status e objetivo

- Status: EM IMPLEMENTAÇÃO; ver o registro de execução abaixo. O plano continua autoritativo e ainda não está aprovado para DONE.
- Prioridade: P1, desempenho; esforço L; risco alto em ownership de memória e concorrência.
- Base: `55ac3792ac126d19f6878bee22b30d1af638ccc3`, verificada em 12/09/2026.
- Repositório: `/home/hjotha/llama`, branch principal `master`, acompanhamento `fork/master`.
- Worktree deste plano: `/home/hjotha/worktrees/llama-adaptive-context-plan`.
- Branch deste plano: `plan/adaptive-context-resident-weights`. Sem commit, merge ou push.
- Dependências externas novas: nenhuma. Reutilizar C++, RAII, GGML, CMake e pytest do projeto.

Objetivo: um único processo de inferência na GOKAYA (`192.168.1.57`), sem router, recebe `qwen-3.8-27b`; decide entre contexto curto com MTP e contexto longo sem MTP usando prompt completo mais reserva de saída; volta ao curto no próximo pedido pequeno. Os pesos principais permanecem carregados durante as trocas. Liberar KV e outras alocações da GPU, salvando antes o estado reutilizável em RAM e restaurando-o no destino quando compatível. Save/restore entre perfis é requisito desta revisão, não uma otimização futura. A redução de aproximadamente 10 segundos por troca é uma hipótese a medir, informada pelo usuário, não uma medição desta análise.

Este arquivo é autossuficiente para um executor. Ler inteiro antes de implementar. O primeiro trabalho de implementação é um experimento de ciclo de vida e memória, antes de ligar a política automática ao tráfego HTTP. Não anunciar o teto ou a latência como validados antes das provas descritas abaixo.

## Registro de execução — 13/09/2026

O executor trabalha na worktree `/home/hjotha/worktrees/llama-adaptive-context-impl`, branch `feat/adaptive-context-resident-weights`, baseada em `55ac3792ac126d19f6878bee22b30d1af638ccc3`. O checkout principal não foi alterado, e ainda não houve commit, merge, push ou deployment.

Implementado até esta rodada:

- Etapa 1: criação/liberação repetível dos contextos mantendo o mesmo `llama_model`, com ordem de destruição, rollback e testes de ciclo de vida.
- Etapa 2: grupo independente de pesos MTP, backing CPU e primitivas de residência; o perfil longo libera somente a alocação GPU exclusiva MTP.
- Etapa 2b: snapshots RAM do cache, checkpoints/carryover MTP, restore transacional e bootstrap target-only; fixtures Qwen MTP e iSWA foram adicionados aos testes.
- Etapa 3: `--ctx-size-mtp`, `--mtp-max-tokens`, seleção por prompt mais reserva de saída, transição na fila existente, cancelamento/defer e estado de indisponibilidade.
- Status e contrato: estado/perfil/tamanho efetivo em `/props`, `/models` e `/slots`, publicação de `transitioning` antes do teardown, erro 503 após rollback impossível e preservação da forma legada quando o modo está desligado.
- Fault hook de teste `LLAMA_TEST_ADAPTIVE_TRANSITION_FAIL` para falhas do candidato e do rollback, sem efeito quando a variável não está definida.
- Etapa 4: identidade `server_model_identity` preparada antes do load adaptativo, com SHA-256 dos shards, metadados de caminho/symlink e overrides tipados; fontes verificadas por metadata após o load. CMake, teste dedicado e integração no `load_model` estão em `/home/hjotha/worktrees/llama-adaptive-context-impl`; save/restore explícito adaptativo já usa essa identidade.

Reviews do Gauntlet: etapas 1, 2, 2b e 3a passaram; a integração da etapa 3 teve um `REJECT` corrigido e depois `ACCEPT_WITH_NOTES`. A identidade da etapa 4 recebeu `ACCEPT_WITH_NOTES` do Codex Sol persistente em `gauntlet/stage-4-identity-sol-review-1.md`, sem bloqueadores. As notas são TOCTOU residual aceitável apenas sob a premissa de GGUF local imutável, recusa explícita no Windows e cobertura posterior via `server_context::load_model`/default-off. A revisão foi feita no modelo `gpt-5.6-sol`, esforço alto, sessão `01a09a42-0e81-7a12-b208-277191694f53`, após ler o transcript legado em `/home/hjotha/worktrees/llama-adaptive-context-impl/gauntlet/claude-transcript.md`. Não serão feitas novas análises Claude; o transcript é apenas contexto arquivado para o Sol.

Validações concluídas: build Release e ASan dos alvos de ciclo de vida/cache; CTest selecionado de parser, lifecycle, carry, bootstrap e rollback; testes de cache Qwen/iSWA com hit, cold, bytes e logits; evidência CPU do Qwen4B em ida/volta curta-longa-curta; identidade com carga real, SHA-256, aliases, overrides e shards; e build remoto CUDA/Vulkan dos alvos `llama-server`, `test-arg-parser` e `test-save-load-state` (saída 0). Os logs e JSON estão em `/home/hjotha/worktrees/llama-adaptive-context-impl/gauntlet/`.

Rodada GOKAYA inicial: a produção `llama-server-root.service` foi registrada em `remote-production-before.txt`, parada durante o teste e restaurada em `remote-production-restored.txt` com `ActiveState=active`, `MainPID=1149501` e `/health` 200 em 8090 e 8092. O binário CUDA de teste foi iniciado em portas 19200+; tanto ele quanto o binário de produção retornaram `ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected`. `nvidia-smi -q` reportou `Product Brand: GPU requires reset` e `GPU Recovery Action: Reset`, com erros NVRM status `0x00000062`; `nvidia-smi --gpu-reset -i 0` retornou `Not Supported`. Portanto não há resultado CUDA, VRAM, 97.536 ou latência de troca aprovado nessa rodada inicial; a produção foi restaurada naquele momento e não houve deployment.

Pendências que bloqueiam DONE: cobertura HTTP adicional com o ambiente pytest completo e shape/transição transitória, teste adaptativo via `server_context::load_model` e default-off sem hashing, profiling de residência/reupload e leaks em processo persistente, matriz CUDA/GOKAYA com 20 ciclos, tetos 56.320/97.536 e comparação contra o router. O limite 97.536 permanece alvo, não resultado medido.

### Rondas Sol r7–r9 e estado atual — 13/09/2026

- O Sol persistente (`gpt-5.6-sol`, high, sessão `01a09a42-0e81-7a12-b208-277191694f53`) rejeitou r7 porque o decoder recebia o orçamento global de cinco cópias como se estivesse livre. A correção passou `max_file_bytes` ao decoder, mantendo a reserva global de cinco cópias e a validação de contagem antes de `reserve()`; o lifetime da reserva continua abrangendo `fail_restore()`/rollback.
- r8 recebeu `ACCEPT_WITH_NOTES` em `gauntlet/stage-2b-slot-sol-review-8.md`: o adversarial foi executado na CPU e na `CUDA0` da 4070, com `--cache-ram 1024`, `--ctx-checkpoints 750000` e envelope checksum-válido de 141.750.145 bytes. Ambos retornaram saída 0, restores válidos (`full_cache_n=4`, `target_cache_n=0`, `checkpoint_probe_cache_n=350`) e HTTP 400 para contagem acima do limite, para o guard de orçamento e para arquivo esparso de 1 TiB. Os JSON estão em `gauntlet/remote-slot-adversarial-cpu-r8e/result-adversarial.json` e `gauntlet/remote-slot-adversarial-cuda-r8f/result-adversarial.json`; os logs registram `invalid adaptive slot snapshot checkpoint count` e `adaptive slot snapshot exceeds the configured state budget`.
- A nota r8 foi corrigida no harness para calcular o cap por arquivo como `ADAPTIVE_CACHE_RAM/5`, com `python3 -m py_compile`, `git diff --check` e verificação dos dois JSON (`141750145 <= 214748364`, contagem 750.000, status 400). r9 recebeu `ACCEPT` em `gauntlet/stage-2b-slot-sol-review-9.md`.
- O build Release (`gauntlet/stage-2b-slot-reject-r8-build.log`), o build ASAN/UBSAN e o teste Release/ASAN de prompt cache (`gauntlet/stage-2b-slot-reject-r8-prompt-cache.log`, `gauntlet/stage-2b-slot-reject-r8-asan-prompt-cache.log`) terminaram com saída 0. O teste registra hit de cache, `model_load_count=1` e backing MTP; não houve diagnóstico ASAN/UBSAN.
- Após esses testes, a produção foi deixada desligada por decisão explícita do usuário para evitar restaurações intermediárias: estado atual remoto `llama-server-root.service` `ActiveState=inactive`, `SubState=dead`, `MainPID=0`, `ExecMainStatus=9`, sem listener em 8090. A RTX 4070 está livre (`16 MiB` usado, `11887 MiB` livre, `0%`). Nenhum binário foi promovido ou instalado.
- Os gates de 97.536 tokens, 20 ciclos, comparação com router, profiling completo e cobertura integral Chat/Responses/stream/replay continuam abertos; esta rodada fecha somente o blocker de orçamento do decoder.

### Rondas Sol r1–r2 de status e draft externo — 13/09/2026

- A Luna implementou a recusa explícita de `--model-draft` quando o modo adaptativo está habilitado, evitando que o caminho de speculative decoding carregue um segundo modelo. O teste de parser cobre a combinação e `build/bin/test-arg-parser` terminou com saída 0; a validação também preserva o caminho legado quando o modo está desligado. A revisão Sol específica dessa fatia ainda será registrada após o build isolado da Luna.
- O Sol rejeitou r1 porque `/slots` não incluía `enabled` e o harness aceitava essa ausência. A correção adicionou o campo aos seis campos do snapshot adaptativo, retirou a exceção do harness e passou a consultar `/models` nos casos de indisponibilidade. O build Release e `py_compile` terminaram com saída 0. r2 recebeu `ACCEPT_WITH_NOTES` em `gauntlet/stage-4-status-sol-review-2.md`; a nota restante é executar o harness HTTP corrigido em uma instância pós-delta.
- Foi adicionado `tools/server/tests/unit/test_adaptive_context.py`, marcado `adaptive` e `slow`, com campos de `ctx-size-mtp`, `mtp-max-tokens`, `fit` e CPU explícito no `ServerProcess`. A coleta pytest encontrou um caso sem carregar o GGUF (`1 test collected`). A execução real usa `ADAPTIVE_HTTP_MODEL` e cobre `/props`, `/models`, `/slots`, Completion curto/longo/curto, SSE e replay, Chat Completions, Responses, schema de tool call e save/restore do slot. O pytest completo na .57 não pôde iniciar porque o ambiente não tinha `requests`/`openai`; os fluxos equivalentes de Completion, SSE/replay, Chat/Responses/tools e save/restore foram exercitados pelos harnesses HTTP nativos anteriores, e a rodada CUDA de slot abaixo cobre o caminho explícito após a correção.
- `tools/server/README.md` agora tem uma seção manual fora da tabela `HELP_START`/`HELP_END` gerada, documentando os dois switches, o limiar, o ciclo de residência MTP, o contrato `adaptive_context`, o fail-stop e as combinações suportadas/recusadas.
- A regressão `unit/test_speculative.py` foi executada com os recursos do host saturados por outros processos: `7 passed, 1 failed` em 33:18. O único caso falhou por timeout de 600 s em `test_with_ctx_shift`, enquanto o servidor gerou lentamente 192 tokens a aproximadamente 0,3 token/s; o log não mostra crash nem erro adaptativo. Isso não é tratado como aprovação: repetir em host controlado ou na `.57` é obrigatório antes de fechar o gate default-off.

### Correção do REJECT de /models e canários HTTP — 13/09/2026

- O Sol rejeitou a primeira revisão HTTP por regressão default-off: `context_window` havia passado a usar `meta.slot_n_ctx` em `/models`. A correção mantém `meta.n_ctx` por slot e restaura `params.n_ctx` somente para `context_window`/`max_context_window`; o teste permanente em `test_basic.py` cobre os valores com `--ctx-size 512 --parallel 2` e ausência de `adaptive_context`.
- O canário direto remoto CPU na `.57`, porta 8093, retornou `context_window=512`, `max_context_window=512`, `meta_n_ctx=256`, `adaptive_present=false`. Build Release, CTest do parser e build ASAN terminaram com saída 0. A segunda revisão Sol recebeu `ACCEPT_WITH_NOTES` em `gauntlet/stage-4-http-sol-review-2.md`; não há blocker. As notas mantêm abertas concorrência/cancelamento, tool call real, default-off integral, CUDA de teto alto e 20 ciclos.
- O teste adaptativo CPU r9 foi repetido com `timings.cache_n > 0` após restore e terminou `1 passed` em 90,50 s; o log mostra `451 -> long`, `long n_ctx=512`, retorno `mtp n_ctx=256` e uma única linha de carga de modelo. O harness de rollback remoto CPU terminou os quatro cenários esperados (`500`, `503` e recuperações `200`) em `gauntlet/adaptive-transition-rollback-http.json`.
- O build CUDA isolado da `.57` foi concluído após sincronizar `src/models` e reconfigurar CMake com `/opt/cuda/bin/nvcc` 13.3; `build-adaptive-cuda/bin/test-arg-parser` passou. O teste HTTP adaptativo GPU CUDA0 na porta 8094 terminou `1 passed` em 10,31 s. O log mostra uma carga do modelo, `MTP_GPU=0` no perfil longo e `MTP_GPU=81195008` no retorno MTP, sem OOM/Xid; a RTX 4070 voltou a `16 MiB` usados e `11887 MiB` livres.
- A produção segue deliberadamente desligada (`llama-server-root.service` inativo, sem listeners 8090/8092/8093/8094); nenhum binário foi promovido. A matriz de 56.320/97.536, 20 ciclos e comparação contra o router ainda não foi executada.

### Rodadas HTTP na GOKAYA e revisão do slice de integração — 13/09/2026

- A primeira execução remota (`stage-4-http-adaptive-57-r1.log`) foi descartada porque o fixture global iniciou um preset padrão antes do servidor configurado; o conftest passou a ignorar `ServerPreset.load_all()` somente quando `ADAPTIVE_HTTP_MODEL` está definido. A segunda rodada alcançou o servidor adaptativo, mas o save explícito excedeu o orçamento de 64 MiB; a terceira excedeu o contexto longo e a quarta ainda não comportava restore simultâneo com 512 MiB. Esses casos deixaram logs separados e motivaram os ajustes do harness, sem reduzir limites do código.
- A rodada aprovada `gauntlet/stage-4-http-adaptive-57-r7.log` terminou `1 passed` em 90,65 s na CPU de oito núcleos da `.57`, porta 8092, com o GGUF local `/home/hjotha/models/Qwen3.5-4B-MTP-Q4_K_M.gguf`. O teste usa contexto curto 256, longo 512, `--cache-ram 2048`, MTP residente e `--model-draft` ausente; cobre `/props`, `/models`, `/slots`, Completion curto/longo/curto, streaming SSE, replay, Chat Completions, Responses, tool schema e save/restore explícito.
- A repetição com log do servidor (`gauntlet/stage-4-http-adaptive-57-r8.log` e `gauntlet/stage-4-http-adaptive-57-r8-server.log`) também terminou `1 passed` em 89,95 s. O log mostra uma única linha de carga do modelo (`load_model: loading model`), criação do contexto MTP contra o mesmo modelo, seleção `451 -> long`, transição `long, n_ctx=512, MTP_GPU=0`, retorno `mtp, n_ctx=256, MTP_GPU=0` e nova geração após restore. O build CPU não tem CUDA; `MTP_GPU=0` é esperado nesta prova de contrato, não é evidência de bytes GPU.
- A revisão Sol persistente do slice foi enviada em `gauntlet/stage-4-http-sol-review-1.prompt.md` com o plano integral, diff, logs e resultados. A decisão ainda está pendente; não considerar o gate de integração fechado até `ACCEPT` ou `ACCEPT_WITH_NOTES` sem blocker. O teste remoto foi executado após a correção de `/slots` e com Jinja habilitado para o caso de tools.
- A produção continua deliberadamente desligada conforme decisão explícita do usuário: `llama-server-root.service` inativo, sem listeners em 8090/8092, e RTX 4070 livre (`16 MiB` usado, `11887 MiB` livre, `0%`). Nenhum binário foi instalado ou promovido.

## Veredito

**Viável arquiteturalmente, com três partes obrigatórias para preservar limites e reaproveitamento de prefill:**

1. Destruir/recriar os contextos target e draft sem destruir o `llama_model` principal.
2. Tornar os pesos exclusivos MTP liberáveis separadamente na GPU. Hoje eles pertencem aos buffers de pesos do modelo e não desaparecem com `llama_free(ctx)`.
3. Salvar/restaurar target, checkpoints úteis e, quando existente, estado draft/especulativo compatível. A volta ao MTP a partir de snapshot somente target exige a prova específica da seção D.

Somente a primeira parte permite uma prova mais simples, mas mantém memória extra e não comprova a entrega pedida. Não promover silenciosamente um contexto longo menor como se fosse equivalente ao router atual.

**97.536 tokens é o alvo inicial, sustentado pelo perfil longo atual. Se liberarmos as mesmas alocações, o esperado é continuar cabendo esse contexto.** Manter os pesos principais na GPU não acrescenta consumo em relação ao perfil longo atual: eles já ficam carregados nele.

Hoje, encerrar o processo libera os recursos daquele processo pelo driver. No modo proposto, a implementação deve reproduzir a liberação pelo código: cabeça MTP, KV, estados recorrentes, checkpoints, contextos, buffers e grafos antigos, sem referências pendentes e sem sobrepor as alocações de inferência dos dois perfis. Os pesos principais permanecem residentes.

**Revalidar 97.536 significa comprovar que essa liberação foi implementada corretamente e reproduz o consumo do perfil atual, inclusive após várias alternâncias; não presumir que o limite vai diminuir.** Eventual memória residual de bibliotecas/driver ou fragmentação é uma hipótese a investigar se houver diferença medida, não evidência de redução inevitável. Não usar `cudaDeviceReset`: invalidaria os pesos que se pretende preservar. A equivalência no novo modo ainda exige os testes abaixo; esta análise não os executou.

## Evidência atual

Inspeção somente de leitura, realizada do optiplex. Na `.57`:

- `llama-server-root.service` ativo; `MainPID=1675`; início `2026-09-12 14:58:08 CEST`.
- `ExecStart` aponta para `/home/hjotha/llama-releases/fork-upstream-20260911-cgraph/build/bin/llama-server`, com `--models-preset /home/hjotha/prod-two-tier.ini --models-max 1`, porta 8090.
- A árvore remota `/home/hjotha/src/llama.cpp` reportou a revisão acima. Não foi feita uma nova comparação de hashes de todas as bibliotecas carregadas nesta análise.
- GPU consultada: RTX 4070, 12.282 MiB totais; amostra de uso 11.900 MiB. A amostra não mede folga em pico nem garante capacidade.
- RAM às `2026-09-12T18:08:10+02:00`: `MemTotal=15522588 kB`, `MemAvailable=8816696 kB`; `SwapTotal=15521788 kB`, `SwapFree=15520412 kB`. Há margem nessa amostra para a cópia de 332,3 MiB, mas o pico com cache/checkpoints precisa ser medido.
- Mesmo GGUF nos dois perfis: `/home/hjotha/models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf`.
- Comuns: `load-mode=none`, `batch-size=256`, `ubatch-size=256`, `parallel=1`, `device=CUDA0`, flash attention ligada, KV K/V `q4_0`, `fit=off`, cache RAM 2048 MiB, um checkpoint.
- MTP: `ctx-size=56320`, `route-max-tokens=56320`, `spec-type=draft-mtp`, `spec-draft-n-max=2`, `spec-draft-p-min=0.80`, draft K/V `q4_0`.
- Sem MTP: `ctx-size=97536`, fallback do grupo `qwen-3.8-27b`.
- Governador atual: prefill 200 W, decode 170 W, memória no decode 11001. Não alterar isso para atribuir um ganho à troca de contexto.

Metadados do GGUF, sem executar inferência: `general.architecture=qwen35`, `qwen35.block_count=65`, `qwen35.nextn_predict_layers=1`. Os tensores de `blk.64` somam **348469248 bytes, 332,326 MiB**. Total lógico dos tensores: 10431832064 bytes. Esses valores não incluem alinhamento de buffers, workspace ou memória do driver.

**Mesmo arquivo não significa o mesmo conjunto de pesos residente:**

```cpp
// common/common.cpp:2257
mparams.load_mtp = std::find(params.speculative.types.begin(), params.speculative.types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();

// src/models/qwen35.cpp:38
int mtp_flags = !ml.load_mtp ? TENSOR_SKIP : 0;
```

O perfil longo atual ignora os tensores MTP. O grupo `blk.64` medido é a cabeça exclusiva deste arquivo; o código novo deve derivar camadas MTP de `hparams`, nunca hardcodar 64 ou esses bytes.

## Mapa do código e decisões já existentes

Referências são relativas à raiz e à revisão planejada; localizar pelos símbolos se linhas mudarem.

| Arquivo / ponto | Evidência e consequência |
|---|---|
| `common/common.cpp:1277`, `common_init_result::impl` | Já possui `model` e `context` como donos RAII distintos, além de samplers e threadpools. Estender essa separação; não criar um segundo carregador completo no servidor. |
| `common/common.cpp:1785-1914`, construtor | Carrega modelo, configura LoRA/sampling e cria contexto. Separar trabalho único do trabalho repetível. `model_only` retorna antes de várias inicializações; não usá-lo como atalho sem reconstruí-las. |
| `common/common.cpp:1942-2094`, `common_init_from_params` | Aplica adapters/control vectors, warmup, limpeza de memória e reset dos samplers. Reutilizar esse comportamento na construção repetível; não copiar metade e esquecer a outra metade. |
| `common/common.cpp:2337-2401`, `common_threadpools` | `init()` exige pools vazios. Reutilizar/reattach com `llama_attach_threadpool` quando configuração não mudar; não chamar `init()` duas vezes no mesmo objeto. |
| `common/common.cpp:2262`, `server-context.cpp:42` | `n_rs_seq`, saídas máximas e buffers dependem de speculation. Recalcular por perfil, a partir de parâmetros novos. |
| `common/speculative.cpp:2583-2640` | MTP embutido usa `llama_init_from_model(model_tgt, cparams)`. Draft tem contexto do tamanho do target e não precisa recarregar o modelo principal. |
| `common/speculative.cpp:1469-1506` | MTP habilita extração NextN no target/draft; destrutor acessa contexto draft para desligar samplers. `spec` deve morrer antes de `spec_init`. |
| `src/llama-context.h:283-377`, `src/llama-context.cpp:553` | Contexto referencia o modelo e possui memória, scheduler, backends, buffers de saída e snapshots de sequências. Destruição sincroniza. |
| `ggml/src/ggml-cuda/common.cuh:1265`, `ggml-cuda.cu:435,555,702,2452` | Destrutores liberam graphs, pools legado/VMM, streams, cuBLAS/workspaces e backend. Isso não é prova de liberação de todas as alocações internas do driver. |
| `src/models/qwen35.cpp:33-123` | Delimita tensores trunk versus MTP; é o lugar semântico para marcar pesos exclusivos. |
| `src/llama-model-loader.cpp:1117`, `src/llama-model-loader.h:ctx_key` | Agrupamento atual por tipo de buffer e lazy; pesos trunk/MTP podem compartilhar a mesma alocação. |
| `src/llama-model.cpp:1166,1713-1857` | `ctxs_bufs` possui metadados e buffers; alocação é por grupo GGML. Não é possível liberar um tensor interior via `cudaFree(t->data)`. |
| `tools/server/server-context.cpp:964` | `destroy()` chama `llama_init.reset()`, liberando também o modelo. Não serve diretamente para troca leve. Sleep usa esse caminho e também não resolve. |
| `tools/server/server-context.cpp:1135-1425` | Inicializa contextos, slots, batch e cache. Extrair somente a parte dependente de contexto, preservando setup HTTP/template/métricas de processo. |
| `tools/server/server-context.cpp:2525` | `process_single_task` recusa mutações durante `is_yielding`. Ponto da seleção: depois da tokenização CLI, antes de `get_available_slot`, que já pode consultar/restaurar cache. |
| `tools/server/server-context.cpp:4457` | Template/tokenização HTTP produzem `task.tokens`; usar a contagem exata antes de reutilizar cache, não tamanho do JSON ou tokens restantes após cache hit. |
| `tools/server/server-models.cpp:1402-1475` | Router usa prompt + reserva; reserva padrão 4096 se não há saída positiva/configurada; escolhe novamente por pedido. Referência de comportamento, não um componente a manter no novo caminho. |
| `tools/server/server-schema.cpp:44` | Normaliza `n_predict`, `max_completion_tokens`, `max_tokens`; Responses passa pela conversão própria. Reusar o valor efetivo normalizado, sem um segundo parser de aliases divergente. |
| `tools/server/server-queue.cpp:defer,pop_deferred_task,process_new_tasks` | Fila e cancelamento existentes. Não criar outra fila global nem thread de polling. |
| `tools/server/server-context.h:125`, `server-context.cpp:4381,4756,4787` | `update_meta` não é thread-safe e os metadados incluem contexto do slot. Não substituir `unique_ptr<meta>` enquanto HTTP lê. |

Os planos históricos `docs/mtp-adaptive-context-plan.md` e `docs/mtp-router-split-plan.md` analisam gating de MTP com alocações persistentes e contêm limites antigos. A frase de que somente uma fronteira de processo recupera os custos não é uma proibição arquitetural: há custos de contexto liberáveis e custos de pesos que exigem ownership separado. A nova proposta não é simplesmente reativar o plano antigo.

A API upstream também separa criação/liberação de contexto de liberação de modelo: https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h . Consultada pela skill web-search; para este trabalho, prevalece o código local fixado na revisão acima.

## Contrato dos switches propostos

**Estes switches novos ainda não existem.** Nomes a implementar:

```text
--ctx-size 97536
--ctx-size-mtp 56320
--mtp-max-tokens 56320
--spec-type draft-mtp
--spec-draft-n-max 2
--spec-draft-p-min 0.80
--spec-draft-type-k q4_0
--spec-draft-type-v q4_0
--parallel 1
```

- `--ctx-size` continua existente. Em modo adaptativo é o teto longo declarado, não uma ordem para alocar esse contexto junto com MTP na inicialização.
- `--ctx-size-mtp N`: novo, padrão 0 desabilita todo o mecanismo; positivo habilita e define o contexto curto. Não precisa de um terceiro switch booleano.
- `--mtp-max-tokens N`: novo, padrão 0 deriva de `ctx-size-mtp`; define o limiar de prompt + reserva para MTP. Serve para calibrar margem sem desperdiçar flexibilidade útil de hardware.
- Validar `0 < limiar <= contexto_curto <= contexto_longo`, incluindo tamanhos efetivos depois de arredondamento/caps internos. Não hardcodar 56320/97536.
- Usar um único limiar na primeira versão: `budget <= limiar` liga MTP; acima desliga; o próximo pedido pequeno volta a ligar. Dois limiares de histerese mudariam o comportamento solicitado.
- Inicializar no perfil curto, como o preset atual com `load-on-startup=true`: carregar corpo principal e cabeça MTP. Ao ir ao longo, salvar estado útil, destruir contextos e liberar a cabeça da GPU. Ao voltar ao curto, liberar contextos longos e reenviar somente a cabeça a partir do backing CPU, antes de recriar/restaurar target e draft. Os pesos principais são carregados uma vez. O perfil inicial deve usar parâmetros efetivos curtos antes de qualquer alocação de contexto.
- Modo inicial suportado: modelo `qwen35` denso com MTP embutido, uma GPU CUDA e `parallel=1`, KV tradicional, `fit=off`, sem multimodal, LoRA/control vectors, outros speculative backends, slot cap adicional ou sleep automático. CPU é admitida somente para checks pequenos de ownership/política; produção validada é CUDA.
- Rejeitar explicitamente combinações não suportadas quando o modo estiver habilitado. Sem switches novos, os caminhos atuais devem permanecer funcionais, inclusive modelos/recursos excluídos do modo adaptativo.
- Preserve porta e nome público `qwen-3.8-27b`, autenticação, template, tools, reasoning, streaming e APIs textuais Chat/Completions/Responses.

### Contagem e saída

`budget = quantidade total de tokens do prompt já formatado + reserva de saída` em `int64_t`, com validação antes de qualquer teardown. O prompt inclui system prompt, histórico, ferramentas e template. Não contar apenas o trecho novo após um cache hit.

Reusar o `n_predict` efetivo normalizado pelo schema. Valor positivo solicitado/configurado é a reserva; zero explícito é zero; sem limite positivo, manter a regra de planejamento atual de 4096. A reserva padrão não deve mudar silenciosamente a geração ilimitada atual para uma geração limitada em 4096. Nenhuma troca no meio de uma resposta: se a geração sem limite explícito exceder a reserva estimada, continua sujeita ao limite físico e à finalização por contexto do perfil escolhido, como no método atual. Testar e documentar isso.

Pedidos com orçamento explícito acima do teto longo devem receber erro de contexto antes de liberar qualquer recurso. Esse erro antecipado é deliberado no modo novo: não anunciar que um pedido coube se houve truncamento de entrada. Verificar margem de BOS, último token e draft no limite, usando tokens reais; não inferir capacidade por HTTP 200.

Arrays de prompts devem usar as tasks existentes, decidindo por task. `n>1`/tasks pai que exigem múltiplos slots devem receber erro explícito no modo de um slot, nunca ficar em defer para sempre.

Nomes diretos antigos dos perfis contornavam o router. Inventariar consumidores no início da implementação. Se precisam continuar aceitos, propagar a preferência de perfil como enum na task e validar o orçamento contra esse perfil; simples aliases que perdem a semântica de forçar perfil não são compatibilidade. O contrato principal garantido é o nome público automático. Consultas de saúde/métricas/tokenização nunca devem provocar uma troca.

## Design mínimo

### A. Modelo persistente e contextos descartáveis

Estender `common_init_result` com operações internas de liberar contexto e inicializá-lo novamente com o modelo existente. Extrair os blocos já existentes em `common.cpp` em helpers locais, preservando inicialização única de vocabulário/biases/LoRA e a vida dos samplers/threadpools. Não recarregar modelo por `common_init_from_params` em cada transição. Nunca criar simultaneamente os dois perfis completos: na 4070 não há orçamento para isso.

Separar `params_base` imutáveis, que declaram a capacidade do serviço, de parâmetros efetivos de cada construção. Perfil longo precisa de speculation vazia, `n_rs_seq` recalculado sem rollback especulativo, `n_outputs_max` recalculado, sem extração NextN herdada e sem contexto draft. Perfil curto reconstrói target, draft e speculation com os valores atuais.

Não reutilizar o target que participou do MTP como se fosse um target sem MTP. Não fazer resize de KV em uso. Destruir o conjunto completo libera também estado recorrente do Qwen híbrido, buffers de saída, grafos, scratch e snapshots.

### B. Grupo de pesos MTP independente

Separar, **na carga inicial e somente sob opt-in**, os tensores exclusivos MTP em um grupo GGML/buffer próprio por dispositivo. Reusar o agrupamento `ctx_key`/`ctxs_bufs` com um discriminador simples de propriedade MTP; não criar um gerenciador genérico de offload. Marcar semanticamente em `qwen35::load_arch_tensors`, incluindo QKV e eventuais tensores auxiliares/escala. Preservar ponteiros dos tensores principais e embeddings/output compartilhados.

O grupo opcional mantém os metadados `ggml_tensor` vivos e uma cópia CPU dos bytes quantizados necessários para reupload. Orçamento esperado da cópia deste arquivo: aproximadamente 332,3 MiB mais alinhamento, a medir; não reservar cópia do GGUF inteiro nem memória pinned permanente. Pode capturar os bytes do carregamento inicial ou copiar somente o grupo uma vez, reutilizando as APIs GGML de transferência. Não manter segunda cópia CPU por transição.

Ao entrar no perfil longo: com todos os contextos destruídos, liberar o buffer GPU do grupo, limpar `data`/`buffer` dos tensores opcionais sem dangling pointers e marcá-lo não residente. As operações do target normal não podem acessar esse grupo.

Ao voltar ao curto: após liberar o contexto longo, realocar somente o grupo opcional, reconstruir seus bindings pelo alocador GGML e reenviar somente seus bytes. Não duplicar `tok_embd`, `output` ou pesos principais; não reler/copiá-los. Só depois criar os contextos MTP.

A operação deve ser idempotente, com ownership RAII e cleanup de alocação/upload parcial. Expor a primitiva em API interna/staging (`src/llama-ext.h`) e métodos de `llama_model`; se for necessário adicionar uma opção em `llama_model_params` para o opt-in inicial, usar padrão desligado e reconstruir todos os componentes consumidores da ABI. Não assumir que mudar `load_mtp` depois da carga aloca/libera tensores.

A cópia CPU é uma escolha deste plano para minimizar latência. Se ela não couber junto de cache RAM/checkpoints e do compactor 8092 na GOKAYA, registrar a falha e rever o backing para leitura seletiva do grupo no GGUF. Isso ainda evita recarregar o corpo principal, mas exige medir outra latência; não trocar o desenho silenciosamente.

### C. Transição entre pedidos

Sequência executada pela thread proprietária do contexto:

1. Validar task e calcular perfil com tokens reais. Se já é o perfil ativo, seguir o fluxo existente, preservando cache.
2. Se existe geração ativa, `defer` na fila existente. Não esperar bloqueando a thread que precisa concluir o decode. Não fazer teardown durante `is_yielding`.
3. Reavaliar a task ao sair do defer, inclusive cancelamento. Com um slot, usar a ordem existente de fila e não priorizar indefinidamente pedidos do perfil atual.
4. Ao ficar ocioso, marcar estado interno de transição; drenar/sincronizar operações de backend. Guardar a identidade da task pendente e o perfil anterior.
5. Antes do teardown, salvar slot ocioso e checkpoints úteis em buffers CPU independentes, verificar os bytes serializados e manter o cache RAM compatível dentro do orçamento existente (seção D). Depois desanexar samplers, destruir slots e vetores de especulação e liberar todos os snapshots GPU. Preservar filas, respostas já produzidas, histórico de métricas e buffers de replay HTTP independentes da inferência.
6. Destruir `spec`, depois `spec_init`/draft, depois target via operação que preserva `model`. Limpar todos os aliases de contexto nos params e campos do servidor. Não chamar callbacks de release com task ativa inexistente.
7. Ajustar residência do grupo MTP, depois construir novo target e, se curto, draft e `spec`. Recalcular limites e flags; warmup como no caminho estático.
8. Recriar/religar slots, `common_memory`, batch, cache RAM preservado, callback de release e samplers. Após warmup, restaurar o snapshot compatível com a task pendente; limpar qualquer restore parcial em caso de falha. O perfil só se torna ativo após construção completa. Cache miss é permitido, estado inválido não.
9. Registrar tempo de transição e despachar exatamente uma vez a task original. Nenhum token de resposta deve ser emitido antes de concluir a preparação.

Não descartar indiscriminadamente o cache RAM: preservar snapshots compatíveis, inclusive uma variante curta com MTP para retorno futuro. Save/restore explícito entre perfis deve funcionar com a mesma validação de compatibilidade do automático. Não remover arquivos externos do usuário. No mesmo perfil, preservar a reutilização atual.

HTTP precisa continuar aceitando/tokenizando pedidos usando vocabulário e template persistentes. Os metadados de capacidade pública devem informar o teto longo, não o contexto curto momentâneo; assim o cliente não compacta prematuramente. Não chamar `update_meta()` concorrente. Publicar perfil ativo, contexto efetivo e estado de transição por snapshot thread-safe ou tasks de status existentes. O `get_meta()` atual acessa `ctx_tgt`; não chamá-lo enquanto o ponteiro estiver ausente.

Falha recuperável de criação/upload: limpar o candidato e tentar reconstruir o perfil anterior uma única vez e restaurar seu snapshot válido; usar cache vazio se indisponível. Devolver erro estruturado à task e manter o serviço íntegro se a recuperação passar. Se recuperação falhar, entrar em estado indisponível e concluir/rejeitar as tasks pendentes, sem servir com ponteiro nulo. Não transformar erro fatal CUDA/Xid em retry infinito: alguns caminhos atuais abortam antes de um retorno normal, e isso exige detecção e rollback operacional.

### D. Save/restore e cache entre perfis (obrigatório)

**Revisão de 12/09/2026:** substitui a decisão inicial de descartar cache e excluir transferência de KV. Desalocar VRAM não significa perder o estado lógico. Reutilizar `server_prompt_cache`, `server_prompt`, checkpoints e `llama_state_seq_*`, sem router, self-HTTP ou novo serviço de cache.

#### Evidência do fork nesta revisão

- `server-context.cpp:317`, `server_slot::prompt_save`: salva target e draft separadamente em RAM com `LLAMA_STATE_SEQ_FLAGS_NONE`, copiando tokens/checkpoints. Adicionar verificação dos retornos de serialização; hoje `get_data_ext` não tem retorno conferido ali.
- `server-task.cpp:1711,1793`, `server_prompt_cache::alloc/load`: deduplica por prefixo e consome a entrada no restore; não valida perfil. Restaurar uma entrada com draft no longo hoje atinge `GGML_ASSERT(ctx_dft)`. Manter o objeto vivo sem adaptar esses caminhos não basta.
- `server-context.cpp:2704,2754`: o save/restore explícito de slot salva somente target e tokens; não salva sozinho draft, checkpoints do servidor nem carryover MTP.
- `src/llama-memory-hybrid.cpp:190,197`: o estado target inclui atenção e recorrência. `llama-kv-cache.cpp:2508` e `llama-memory-recurrent.cpp:1088` verificam tipos, dimensões e capacidade. Diferença de `n_ctx` não é sozinha prova de incompatibilidade nem garantia de compatibilidade; testar também os valores efetivos diferentes de `n_rs_seq`.
- `src/llama-context.cpp:3292`, `state_seq_read_data`: arquivo de sequência possui magic/versão/tokens e memória, mas não identidade completa de modelo/configuração. Validar essa identidade no servidor; slot ID ou nome público do modelo não bastam.
- `common/speculative.cpp:1379,1515`, `common_speculative_impl_draft_mtp`: MTP tem `pending_h`, cache draft e posições. `begin()` só avisa se falta histórico draft; não o reconstrói. O MTP desta revisão não sobrescreve `get_state/set_state`, portanto os hooks atuais não garantem exportação do carryover MTP.
- `server-context.cpp:3468-3578`: Qwen híbrido pode exigir checkpoint recorrente anterior para aproveitar um prefixo. Sem ele, força prefill integral; prefixo textual igual não comprova cache efetivamente reutilizável.

#### Snapshot, compatibilidade e limites

Salvar tokens/posições efetivamente avaliados, estado completo target (KV de atenção e recorrência), checkpoints úteis e metadados de identidade/configuração; quando disponível, salvar draft e carryover MTP coerentes no mesmo ponto. Não incluir tokens apenas amostrados ainda não avaliados. Sampler/gramática seguem o ciclo normal da nova task: este recurso reaproveita prefill, não retoma uma geração interrompida.

O snapshot deve possuir bytes host independentes. Checkpoints `ON_DEVICE` não podem sobreviver ao contexto: converter os úteis para host antes do teardown ou descartá-los com miss explícito. `PARTIAL_ONLY` não substitui o snapshot completo; serve a checkpoints segundo sua semântica existente. Nenhuma cópia GPU de KV/draft deve ficar viva no perfil longo para facilitar o retorno.

Reusar o limite global de cache RAM (2048 MiB no cenário inspecionado), contabilizando target, draft, carryover, checkpoints e temporários; o backing CPU da cabeça é adicional e entra no pico total de RAM. Evitar duplicar todo o cache por transição. `/dev/shm` também consome RAM. O automático pode usar serialização em memória; o explícito mantém arquivos e recebe metadados/complemento versionado quando necessário, publicados somente depois de completos. Arquivos antigos somente target devem ser reconhecidos; sem evidência suficiente de identidade no modo adaptativo, retornar incompatibilidade explícita, não restaurar às cegas.

Respeitar cache desabilitado e `cache_prompt=false`. Falta de RAM, eviction, ausência/corrupção/incompatibilidade resultam em miss com motivo, limpeza de qualquer estado parcial e prefill normal; nunca crash ou falso hit. Save explícito com falha devolve erro ao chamador. Falha do save automático não obriga abortar a troca, mas deve aparecer na medição.

1. Selecionar perfil usando prompt completo mais saída antes da consulta ao cache; hit não reduz o orçamento de roteamento.
2. Validar prefixo idêntico de tokens, identidade dos pesos, versão do estado, layouts/tipos KV e recorrentes, posições, espaço e parâmetros efetivos que alterem RoPE/atenção. Não rejeitar só porque `n_ctx`/perfil mudou; rejeitar mudanças semânticas. Conversation ID pode ajudar, mas não substitui essas verificações nem deve ser obrigatório para o cache local normal.
3. Curto MTP para longo: restaurar target e checkpoints target compatíveis, ignorando draft no destino e preservando a variante curta completa em RAM dentro do limite. Separar restore target/draft para remover a suposição que leva ao assert atual.
4. Longo para curto: preferir snapshot curto completo com prefixo correspondente e restaurar target/draft/carryover no mesmo ponto. Snapshot longo somente target exige o tratamento abaixo. Se tokens/posições não couberem no curto, buscar snapshot/checkpoint anterior válido; não truncar blob nem aplicar estado recorrente final a um prefixo anterior.
5. Adaptar deduplicação, eviction e consumo: uma entrada target longa não torna automaticamente obsoleta a única variante MTP utilizável. Ambas respeitam um limite global. Manter o snapshot válido até sucesso de todo o restore; falha draft não publica target parcial nem destrói a única cópia boa.
6. Após restore, processar apenas o sufixo necessário a partir do checkpoint válido. Prompt idêntico ainda pode exigir último token para logits; no híbrido, exige estado anterior apropriado. Não prometer prefill zero nem contar tokens restaurados mas depois reprocessados como reutilizados.

#### Volta ao MTP sem draft salvo: prova obrigatória

Recarregar os pesos da cabeça não recria seu KV nem os hidden states de prefill. O snapshot target comum não guarda todas as linhas NextN. **Save slot target sozinho não comprova volta ao MTP sem prefill integral.**

Primeiro ampliar os hooks existentes de estado especulativo para exportar/importar o mínimo carryover MTP e provar retorno com snapshot curto completo, incluindo checkpoints coerentes. Identificar campos persistentes versus scratch pelo fluxo; não serializar indiscriminadamente todos os vetores.

Depois provar o caso somente target vindo do longo: preservar target restaurado, obter hidden state atual via sufixo/checkpoint válido e experimentar bootstrap do draft com histórico parcial e verificação normal pelo target. É hipótese de implementação: verificar posições, ausência de leitura de KV ausente, saída correta e taxa de aceitação útil. O warning de `begin()` não prova suporte. Não manter todas as hidden rows nem recursos MTP GPU no longo como atalho.

Se bootstrap seguro não se comprovar, usar snapshot curto anterior ou prefill normal, com motivo explícito de incompatibilidade MTP completa. Não desativar MTP silenciosamente abaixo do limiar, nem anunciar esse fallback como hit. Se toda volta com snapshot somente longo exigir prefill integral, apresentar essa limitação comprovada antes de considerar atendido o objetivo de desempenho nas duas direções; rever o desenho com essa evidência.

## Implementação em etapas

### 0. Baseline e verificação de divergência

Antes de editar, ler políticas e repetir:

```bash
git status --short --branch
git worktree list
git diff --stat 55ac3792ac126d19f6878bee22b30d1af638ccc3..HEAD -- common src include tools/server ggml/src/ggml-cuda
```

Esperado: base identificada e nenhuma alteração de outra tarefa incorporada. O checkout principal tinha duas exclusões alheias (`CLAUDE.md` e um log em `benches/dgx-spark/`); não restaurar ou incluir essas mudanças. Criar/reusar worktree de implementação própria; não implementar no checkout principal. Este é um fork privado, não uma solicitação de contribuição upstream.

Registrar baseline da configuração atual e dos testes selecionados antes da mudança. A análise presente não executou build/testes nem benchmark de inferência.

### 1. Provar propriedade de contexto sem router

Extrair no `common_init_result` a criação/liberação repetível de contexto com modelo residente. Primeiro experimento usa contextos conservadores, não o máximo da GPU; alterna target+MTP / target sem MTP / target+MTP e gera texto real em cada perfil. Isso valida aliases, threadpools e recuperação antes de conectar HTTP. Usar o runner de teste existente, sem um novo framework.

Verificação: build e teste de ciclo de vida com mesmo PID, mesmo `llama_model*`, nenhuma segunda carga do modelo, geração correta após a volta ao MTP; samplers destruídos na ordem correta sob ASan/UBSan em CPU quando houver fixture compatível. Modelo Stories sem cabeça MTP não substitui a prova real de MTP.

### 2. Separar e alternar somente os pesos MTP

Implementar grupo opcional da seção B com API interna, testar liberar/carregar repetidamente e injetar falha de alocação/upload. Comparar bytes e identidades dos pesos principais antes/depois. Realocar o grupo somente sem contextos vivos. Nenhuma alteração de kernel CUDA deve ser necessária nesta etapa.

Verificação: pesos principais/buffers estáveis; hash dos bytes MTP recarregados igual ao original; no perfil longo, bytes alocados pelo grupo GPU MTP iguais a zero; sem duplicação dos embeddings/output; mesma saída determinística de controles estáticos equivalentes. A diferença de NVML não precisa ser exatamente 332,326 MiB por causa de alinhamento/driver.

### 2b. Provar save/restore antes da política HTTP

Implementar a seção D com contextos conservadores e depois reais: snapshot host antes do teardown; curto para longo; retorno ao snapshot curto completo; longo que caiba no curto; somente target sem draft. Adaptar compatibilidade, deduplicação/seleção e restore transacional no cache existente, preservando API explícita e identificando snapshots antigos somente target.

Verificação: saída comparada ao controle frio do mesmo perfil, tokens efetivamente avaliados/reutilizados, checkpoints híbridos e draft proposto/aceito. Injetar falhas, corrupção e eviction; limpar restore parcial. Resposta correta após prefill integral não passa o caso de hit compatível. Registrar o resultado específico da prova de bootstrap MTP, sem escondê-lo no fallback.

### 3. Implementar flags, seleção e transição no servidor

Adicionar opções em `common/common.h` e `common/arg.cpp`; parse/validação no padrão existente. Extrair o setup dependente de contexto de `server_context_impl::load_model`, inserir seleção em `process_single_task` antes da consulta/restauração de cache e preservar fila/cancelamento. Recalcular os parâmetros a cada perfil, sem modificar o objeto de configuração que threads HTTP consultam.

Verificação: testes de parser e integração com limiar pequeno provam `T-1`, `T`, `T+1`, ida/volta e ausência de troca quando não muda perfil. Task grande pendente não pode ser atropelada indefinidamente, duplicada ou perdida. Consultas de status/tokenize não fazem upload/troca.

### 4. Metadados, erros, streaming e regressão

Atualizar a capacidade estável e status efetivo sem data race. Testar Chat, Responses, Completions textuais, tool calls, streaming com fila e cancelamento, e reconexão pelo mecanismo de replay já existente. Bloquear combinações explicitamente fora de escopo. Documentar os switches em `tools/server/README.md` e qualificar a conclusão histórica do plano adaptativo.

Verificação: testes selecionados passam; erros de validação não causam troca; falhas recuperáveis não deixam serviço com contexto nulo; replay de uma resposta concluída não referencia o KV destruído; pedidos grandes não são rejeitados por metadados antigos do perfil curto.

### 5. Validar equivalência de memória na GOKAYA e comparar com router

Executar em janela controlada com GPU exclusiva para a comparação, após drenar produção. Preservar a configuração e a presença do compactor/consumidores de RAM do cenário de produção. Reusar o padrão de captura/restauração em `scripts/bench-cuda-graph-headroom.py` e `scripts/bench-route-graph-memory.py`. Esses scripts existentes param/reiniciam produção: não executá-los casualmente durante o planejamento. Acrescentar modo explícito e opt-in de comparação adaptativa, sem mudar o comportamento padrão dos scripts.

Primeiro verificar os alvos atuais 56320/97536 com as três partes implementadas, incluindo o pico de RAM dos snapshots, inclusive após várias alternâncias. Comparar alocações vivas e picos com os perfis estáticos equivalentes para comprovar a liberação correta, sem sobreposição dos recursos antigos e novos. Se 97536 falhar, investigar alocações retidas, referências pendentes e memória residual medida antes de atribuir a falha a um limite menor. Corrigir a liberação e repetir a comparação; não reduzir o teto e marcar a exigência original como cumprida. Explorar margens menores em experimento separado é válido, mas não prova equivalência; só informar um teto seguro se efetivamente medido e validado.

Verificação final: critérios abaixo satisfeitos, evidência salva em JSON/logs e rollback restaurando o serviço anterior validado. Depois disso preparar o comando standalone e a alteração mínima de unit/preset para implantação futura. Este plano não autoriza executar implantação agora.

## Arquivos em escopo na implementação

- `common/common.h`, `common/common.cpp`, `common/arg.cpp`: parâmetros, vida do contexto, threadpools.
- `common/speculative.cpp` e `.h`: construção/cleanup, exportação/importação do carryover MTP e prova de bootstrap com target restaurado; reutilizar hooks e preservar verificação de drafts pelo target.
- `src/llama-model-loader.h`, `.cpp`, `src/llama-model.h`, `.cpp`, `src/models/qwen35.cpp`: separar grupo de pesos e residência opcional.
- `src/llama-ext.h`: primitiva interna; `include/llama.h` somente se necessário ao opt-in de carga, com ABI reconstruída.
- `tools/server/server-context.cpp`, `.h`, `server-task.h`, `server-task.cpp`: admissão, teardown, parâmetros/status, API save/restore e cache RAM com compatibilidade, variantes target/MTP e restore transacional.
- `tools/server/server-schema.cpp`: somente caso seja necessária a validação/normalização de contrato descrita, sem alterar semântica do modo normal.
- `tools/server/server-queue.cpp`, `.h`: somente se um teste reproduzir falha de defer/cancelamento que não possa ser resolvida na admissão existente.
- `tests/test-arg-parser.cpp`, testes existentes em `tools/server/tests/unit/test_speculative.py`, `test_completion.py`, `test_chat_completion.py`, `test_compat_oai_responses.py`, `test_slot_save.py`, `test_stream.py` e utilitário existente de servidor: ampliar cobertura sem nova infraestrutura.
- `scripts/bench-route-graph-memory.py`, `scripts/bench-cuda-graph-headroom.py`: controles reproduzíveis e modo adaptativo explicitamente solicitado.
- `tools/server/README.md`, `docs/mtp-adaptive-context-plan.md`, `plans/README.md`: documentação e status.

Fora de escopo: nova engine genérica de paging/offload, mudança de quantização, GPU clocks, context resize no meio da geração, migração direta GPU de KV sem serialização, múltiplos slots/GPU, multimodal, outros modelos, reescrita do router, novo dashboard e novos kernels. Se for necessário tocar `src/llama-context.*` ou GGML CUDA para resolver memória retida não explicada, interromper a implementação da etapa e apresentar evidência para rever o plano; não esconder isso em uma refatoração ampla.

## Comandos de verificação

Comandos do fluxo CMake/pytest existente inspecionados no repositório; **não executados nesta análise**. Rodar a partir da worktree de implementação. Um ambiente de build/testes ausente deve ser preparado nela, sem modificar o checkout principal. Os testes de servidor podem baixar fixtures; o teste real de MTP deve apontar explicitamente para o GGUF local.

```bash
cmake -S . -B build -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=ON
cmake --build build --target llama-server test-arg-parser -j 4
ctest --test-dir build --output-on-failure -R '^test-arg-parser$'
env LLAMA_SERVER_BIN_PATH="$PWD/build/bin/llama-server" PYTEST_WORKERS=1 bash tools/server/tests/tests.sh unit/test_speculative.py unit/test_completion.py unit/test_chat_completion.py -v -x
env LLAMA_SERVER_BIN_PATH="$PWD/build/bin/llama-server" PYTEST_WORKERS=1 bash tools/server/tests/tests.sh unit/test_compat_oai_responses.py unit/test_slot_save.py unit/test_stream.py -v -x
git diff --check
```

Esperado: códigos de saída 0, testes selecionados efetivamente executados, sem skips dos novos casos adaptativos. A suíte `test_speculative.py` atual usa draft-simple em parte da cobertura: serve como regressão, não prova MTP deste modelo. Adicionar os checks do experimento e os casos adaptativos nela ou nos arquivos existentes pertinentes.

Na GOKAYA, para build CUDA isolado:

```bash
cmake -S . -B build-adaptive -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-adaptive --target llama-server test-arg-parser -j 4
ctest --test-dir build-adaptive --output-on-failure -R '^test-arg-parser$'
```

Esperado: build/teste 0 com conjunto coerente de binário e bibliotecas. Não misturar nova ABI e bibliotecas de release anterior. A opção exata do runner de benchmark adaptativo será definida e documentada na implementação; ela ainda não existe, portanto não há comando fictício executável neste plano.

## Matriz de testes e aprovação

### Funcional e ownership

- Desabilitado: comportamento e testes existentes preservados.
- Flags: ausentes, inválidas, negativos, limiar maior que curto, curto maior que longo, arquitetura sem MTP, `parallel>1`, configs excluídas.
- Seleção: curto/curto não troca; curto/longo/curto troca corretamente; orçamento igual ao limiar vai ao curto, mais um vai ao longo.
- Contagem: strings, tokens explícitos, templates/tools/Responses; alias de saída positivo, zero, ausente e ilimitado; prompt já em cache ainda conta inteiro.
- Excesso do teto: erro antes de teardown; nenhuma truncagem de entrada disfarçada de sucesso.
- Fila: pedido longo durante stream curto; pequeno posterior; cancelamento antes da troca, durante preparação e após admissão; nenhuma task presa/duplicada. Batch de prompts e `n>1` não podem provocar deadlock.
- Contexto: target/draft/spec válidos após cada volta; estado recorrente/checkpoints/gramática/sampler não vazam entre pedidos; sem stale pointers sob sanitizers no teste CPU aplicável.
- Falhas: criação target, draft, upload MTP e reconstrução anterior. Testar caminho de indisponibilidade; não assumir que OOM fatal é capturável.
- Status: `/v1/models`/`props` não rebaixam o teto anunciado ao alternar; métricas acumuladas preservadas; `/slots` revela tamanho efetivo; consultas não acionam a política.
- Integração: Chat/Responses/tools, SSE e replay; teste de memória fica separado de conformidade funcional.
- Cache entre perfis: curto para longo, retorno ao snapshot curto completo, longo que cabe/não cabe no curto, prompt idêntico mudando reserva, prefixo divergente/compactado e checkpoint híbrido presente/ausente. Comparar cache reportado com tokens realmente avaliados.
- MTP após restore: target/draft/carryover alinhados, draft proposto/aceito e saída contra controle frio; caso somente target separado. Cabeça residente não prova MTP funcional.
- Robustez: cache desabilitado, `cache_prompt=false`, entrada maior que limite RAM, eviction, erro de save, corrupção/incompatibilidade e target restaurado seguido de falha draft. Sem assert, estado parcial ou falso hit.
- API slot: save/restore no mesmo perfil e nas duas direções quando compatível; snapshots antigos somente target, identidade divergente, complemento incompleto e capacidade insuficiente com tratamento explícito. Tasks serializadas sem corrida com transição.

### GPU real e latência

Comparar os três controles com mesmos prompts, saída, batch/ubatch, KV, flash attention, clocks, GPU e revisão:

1. Router atual com processos separados.
2. Novo modo no mesmo perfil, sem troca, contra processo estático equivalente.
3. Novo modo alternando MTP/no-MTP/MTP.

Usar pelo menos 20 ciclos de ida/volta após aquecimento, com amostras antes/depois de cada estágio; intercalar formatos/tamanhos de prefill e widths diferentes. Incluir máximos de prompt + saída (56320 e 97536), sem context shift, e nova geração após o máximo. Repetir medições na mesma ordem nos controles e registrar percentis/dispersão; não usar uma conclusão isolada como prova.

Registrar, por pedido: tempo em fila, save host, teardown, upload do grupo MTP, criação target/draft, warmup, restore host, bootstrap draft, primeiro prefill, TTFT e tempo total; tokens de entrada realmente processados, tokens reutilizados, saída, draft proposto/aceito. Registrar PID, contagem de carga de pesos principais, bytes transferidos principais/MTP, identidade dos buffers principais, VRAM livre CUDA e NVML, RAM/RSS/PSS e swap, pico durante transição e estado após aquecimento.

O modelo principal deve ter **uma carga por vida do processo**, salvo reinício operacional. Upload dos seus pesos durante a troca deve ser zero; no retorno MTP somente o grupo opcional pode ser enviado. O perfil longo não pode ter buffers exclusivos MTP residentes. Mesmo PID, sozinho, não prova esses requisitos.

Comparar TTFT com cache frio, hit curto para longo, retorno ao snapshot curto MTP completo e retorno somente target longo; incluir conversa com cache transferido no router. Medir bytes/tempo de save/restore, motivos de miss e tokens realmente reutilizados/reprocessados. Comparar economia de recarga com custo das cópias host e prefill residual. Eliminar recarga dos pesos não elimina fila, prefill ou warmup.

Gates propostos: zero crash/Xid/OOM fatal, zero falha de task/corrupção, nenhuma queda progressiva de memória livre entre pontos equivalentes após warmup, ambos os tetos cumpridos e ganho de troca repetível. Objetivo inicial de desempenho: reduzir a mediana da etapa de troca pelo menos à metade do controle router, sem regressão sustentada acima de 5% no decode do mesmo perfil; esses são critérios propostos, não resultados. Relatar também p95/TTFT total, sem esconder piora por miss, cópias de estado ou bootstrap MTP. CUDA Graph fallback recuperável deve ser contado separado e não pode ser necessário para passar os máximos se o controle equivalente os passa sem fallback.

## Conclusão de implementação e rollback

Marcar DONE somente quando ownership, contrato, save/restore com cache compatível nas duas direções, regressão e GPU real passarem. Registrar qualquer limitação comprovada de retorno MTP com snapshot somente target; prefill integral não conta como hit aprovado. O experimento de contextos com cabeça sempre residente é um marco intermediário, não a entrega final. Atualizar o índice com caminhos dos resultados e limites realmente aprovados.

Para implantação posterior: preservar unit/preset/binário/bibliotecas do router como rollback; drenar geração; substituir por standalone com os switches e mesmos demais parâmetros; verificar PID/executável/bibliotecas, configuração efetiva, Chat/Responses reais, ida/volta e journal. Voltar ao router se limites, latência ou integridade falharem. A porta 8092 e o notificador do dock não fazem parte da mudança.

Interromper e reportar se houver divergência de base, buffers MTP ainda compartilhados com o trunk, leitura do grupo ausente por grafo normal, necessidade de manter dois contextos completos simultâneos, insuficiência de RAM para backing, OOM abortando recuperação, tarefa perdida, data race ou teto de 97536 não alcançado. Não mascarar essas condições com retries, desligamento global de CUDA Graphs, contexto menor ou reload integral oculto.

## O que foi e não foi validado neste planejamento

Conferidos: código de ownership/admissão/carregamento/CUDA, configuração e status da `.57`, metadados reais do GGUF e convenções de teste. Revisão auxiliar somente de leitura sobre memória e MTP, consolidada com leitura dos pontos citados.

Ainda não concluídos: profiling completo de transferência, benchmark de troca com 20 ciclos, medição do teto 97.536, cobertura integral Chat/Responses/stream/replay e validação CUDA funcional. O relatório não é auditoria geral de segurança nem certificação de estabilidade do fork. Os dois planos históricos não foram modificados; este plano registra onde sua conclusão precisa ser qualificada.

### Validação desta revisão de cache

Conferidos nesta revisão documental de 12/09/2026: save/restore de slot, cache RAM, serialização híbrida, checkpoints e carryover MTP do código local na base indicada. O registro de execução de 13/09/2026 acrescenta a implementação parcial, os artefatos de build/teste e as tentativas controladas na GOKAYA; a produção foi restaurada na rodada inicial e depois deixada deliberadamente desligada (`llama-server-root.service` inativo, sem listener 8090) até existir um binário final, sem deployment. Objetivo, sequência, escopo, etapas e testes continuam sendo a especificação para concluir as pendências acima.

### Correção do REJECT de rollback CUDA e rodada r2 — 13/09/2026

- A revisão Sol persistente r1 (`gauntlet/stage-5-gpu-sol-review-1.json`) rejeitou o slice porque `test_fault` fazia short-circuit antes de `llama_model_mtp_weights_set_resident()` e `recreate_context()`. Assim, os casos chamados de rollback só provavam falha anterior à mutação, não unload/upload parcial nem rollback real.
- A correção em `tools/server/server-context.cpp` separa as fases: o candidato longo pode falhar depois de liberar a residência MTP; o candidato MTP usa `llama_mtp_weights_fault::allocation`/`upload` (o modo `mtp` do harness injeta upload); e a falha `+rollback` ocorre depois de restaurar a residência e antes da recriação do contexto. O caminho posterior à recriação também possui fault hook próprio e limpa o contexto antes de declarar indisponibilidade.
- `gauntlet/test-adaptive-transition-rollback.py` passou a verificar `mtp_weights_resident` em `/props`, `/models` e `/slots`, inclusive no estado `unavailable`. `python3 -m py_compile`, `git diff --check`, build CPU (`build-luna-model-draft`) e build CUDA (`build-adaptive-cuda`) terminaram com saída 0; `ctest`/`test-arg-parser` passou nos dois builds.
- Na GOKAYA, com produção mantida desligada por decisão explícita (`llama-server-root.service` `inactive`, sem listener 8090/8092/8094), o harness CUDA r2 terminou `RC=0`. `stage-5-http-rollback-57-gpu-r2.log` registra os quatro cenários: `long` e `mtp` recuperáveis; `long+rollback` e `mtp+rollback` em `unavailable` com 503 posterior. Os logs individuais mostram `candidate long after MTP residency`, `injected MTP upload failure` e `rollback after MTP residency`, sem OOM/Xid/assert/abort.
- A nova rodada não fecha ainda os gates de 97.536 tokens, 20 ciclos, profiling de RAM/VRAM/transferências, comparação com o router, cobertura completa de concorrência/cancelamento e modelo Qwen3.8-27B. Nenhum binário foi promovido ou instalado; a próxima revisão Sol deve confirmar a correção antes de avançar.

### Correção do REJECT de rollback-context e rodada r3 — 13/09/2026

- O Sol r2 (`gauntlet/stage-5-gpu-sol-review-2.json`) rejeitou porque o predicado de `rollback-after-residency` também capturava `rollback-context`; por isso o hook posterior à recriação nunca era alcançado.
- A correção em `tools/server/server-context.cpp` faz o hook de pós-residência excluir explicitamente `rollback-context`, deixando o fault posterior executar somente após `recreate_context()`. O caminho então chama `discard_context()` antes de publicar `unavailable`.
- `gauntlet/test-adaptive-transition-rollback.py` ganhou o caso `long+rollback-context`, verifica o marker `rollback after context recreation` no log e confirma residência MTP, estado `unavailable` e 503 na tarefa seguinte. `python3 -m py_compile` e `git diff --check` passaram.
- Build CPU e build CUDA dos alvos `llama-server`/`test-arg-parser` passaram; o parser CUDA terminou `test-arg-parser: all tests OK`.
- Na GOKAYA, com `llama-server-root.service` mantido deliberadamente `inactive`, portas 8090/8092/8094 livres e RTX 4070 em 16 MiB usados/11887 MiB livres, a rodada CUDA r3 terminou `HARNESS_RC=0` nos cinco casos (`long`, `mtp`, `long+rollback`, `long+rollback-context`, `mtp+rollback`). O log do novo caso contém o marker pós-contexto e o JSON confirma 500 no gatilho e 503 após indisponibilidade; os casos recuperáveis retornam 200.
- A tentativa CPU do mesmo harness venceu o readiness fixo de 60 s durante o carregamento do fixture GGUF e não conta como aprovação. Não há `oom`, `Xid`, `cuda error`, `segmentation`, `assert` ou `abort` nos logs CUDA r3.
- A revisão Sol r3 (`gauntlet/stage-5-gpu-sol-review-3.json`, `stage-5-gpu-sol-review-3.md`) respondeu `ACCEPT_WITH_NOTES`: confirmou a ordem unload → rollback de residência → `recreate_context` → `discard_context` → `unavailable`, sem blocker residual neste slice. As notas não bloqueantes pedem medir leaks em processo persistente, reupload por contadores/profiling e cobrir MTP totalmente reconstruído.
- Continuam abertos os gates de 97.536 tokens, 20 ciclos, profiling RAM/VRAM/transferências, comparação com o router, concorrência/cancelamento amplo, Chat/Responses/stream/replay completo e Qwen3.8-27B. Produção permanece parada; nenhum binário foi promovido.

### Save/restore explícito CUDA e rollback transacional — 13/09/2026

- O harness nativo `gauntlet/test-adaptive-slot-rollback.py` foi executado na RTX 4070 da .57 com `ADAPTIVE_CACHE_RAM=2048`, `CUDA0`, portas efêmeras e o GGUF Qwen3.5-4B MTP. Terminou `HARNESS_RC=0`.
- O caso salvou o perfil curto (`n_saved=9`, `n_written=105754503`) e o perfil longo (`n_saved=350`, `n_written=169555142`), tentou restore incompatível e recebeu HTTP 400, manteve o perfil longo pronto (`mtp_weights_resident=false`) e teve `cache_n_after=346` na geração seguinte.
- O processo foi reiniciado e restaurou o snapshot longo com HTTP 200, confirmando a identidade/arquivo persistente fora da vida do primeiro processo. Os logs registram transições longas, evictions dentro do orçamento e uma única carga por processo; não há OOM/Xid/assert/abort.
- Pós-teste: `llama-server-root.service` continuou `inactive`, sem listeners 8090/8092/8094/19210; a RTX 4070 voltou a 16 MiB usados, 11887 MiB livres e 0% de utilização. Nenhum binário foi instalado ou promovido.
- O pytest opt-in não foi executável nessa imagem remota por ausência dos módulos `requests` e `openai`; não foi instalado nada. A cobertura nativa equivalente permanece válida, mas a execução formal de `unit/test_adaptive_context.py` segue gate aberto até existir um ambiente com dependências completas.
- O Sol r1 (`gauntlet/stage-6-slot-sol-review.json`) rejeitou a serialização por UB em `adaptive_slot_writer::raw()` para vetores vazios (`nullptr + 0`). A correção retornou imediatamente para tamanho zero e rejeita ponteiro nulo positivo.
- O Sol r2 (`gauntlet/stage-6-slot-sol-review-2.json`) respondeu `ACCEPT_WITH_NOTES`: confirmou que o snapshot target-only longo agora serializa sem UB, sem blocker residual. Permanecem como notas a prova ASan/UBSan HTTP incompleta, uma geração com `cache_n > 0` após reinício e propostas/aceites MTP após restaurar snapshot curto completo.

### Sondagem de alocação Qwen3.8-27B — 13/09/2026

- Na GOKAYA, com produção mantida `inactive`, o GGUF real `/home/hjotha/models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf` iniciou o modo adaptativo com `context_size=56320`, `context_size_long=97536` e `mtp_weights_resident=true`.
- Com limiar temporário 8 apenas para evitar um prefill de 56 mil tokens, duas avaliações curtas forçaram `mtp → long → mtp`: todos os pedidos retornaram HTTP 200; o perfil longo publicou `n_ctx=97536`, `MTP_GPU=0`, e o retorno publicou `n_ctx=56320`, `MTP_GPU=348469248` (332,326 MiB).
- O log `gauntlet/stage-7-capacity-probe/server.log` mostra uma única carga do modelo, criação dos contextos e cleanup limpo, sem `OOM`, `Xid`, erro CUDA, assert ou abort. Pós-teste: serviço de produção inativo, portas livres, 4070 com 16 MiB usados/11887 MiB livres/0%.
- Esta é prova de alocação e ida/volta no modelo alvo, não prova de prefill máximo, 20 ciclos, latência ou teto útil de 97.536 tokens; esses gates continuam abertos.

### Prefill no limite agregado Qwen3.8-27B — 13/09/2026

- Com a mesma produção deliberadamente desligada e o binário CUDA isolado na .57, o teste `gauntlet/stage-7-max-prefill/run-r2.log` enviou `97.535` tokens de entrada e reservou `1` token de saída, totalizando exatamente `97.536` tokens no perfil longo. A requisição retornou HTTP 200, avaliou `97.535` tokens (`tokens_evaluated=97535`) e produziu `1` token; não houve redução silenciosa do prompt.
- O prefill levou `149813.803 ms` a `651.041 tok/s` de prompt. Durante a medição a 4070 chegou a `11783 MiB` usados (`120 MiB` livres), sem OOM, Xid, erro CUDA, assert ou abort. Depois da transição de volta ao perfil curto, o pedido curto retornou HTTP 200 e a GPU voltou a `16 MiB` usados/`11887 MiB` livres/`0%`.
- Um pedido com `97.536` tokens somente de entrada foi rejeitado antes do decode porque o contrato dinâmico reserva espaço para ao menos um token de saída; por isso o caso aprovado usa `97.535 + 1`, sem diminuir o teto configurado. Com `--no-context-shift`, o servidor registra `truncated=true` ao atingir a capacidade física (`prompt.n_tokens()+1 >= n_ctx`); isso é o marcador de parada no limite, não truncagem de entrada. O corpo ainda reporta `prompt_n=97535` e `tokens_evaluated=97535`.
- Esta rodada fecha a prova de alocação e prefill agregado no teto configurado, mas não fecha o gate de teto útil para conversas que precisem de mais de um token de saída, nem os gates de 20 ciclos, profiling completo, comparação com o router, cobertura API/concor­rência e pytest formal.

### Revisão Sol do prefill máximo — 13/09/2026

- A sessão persistente Sol respondeu `ACCEPT_WITH_NOTES` em `gauntlet/stage-7-max-prefill-sol-review.json`/`.md`. Confirmou que `tokens_evaluated=97535` e `timings.prompt_n=97535` demonstram avaliação integral do prompt, que `97535+1` corresponde à reserva agregada exata de `97536` e que `--no-context-shift` exclui compactação automática.
- O revisor confirmou que `truncated=true` é a parada física prevista por `prompt.n_tokens()+1 >= slot.n_ctx`, não redução silenciosa da entrada; também confirmou que a rejeição de `97536` tokens somente de prompt é compatível com reservar uma saída mínima. O teto configurado permanece `97536`.
- Notas não bloqueantes: o caso não demonstra saída textual útil no último token, pico completo/estabilidade repetida nem bytes zero de reupload do trunk; o limiar 8 é apenas um acelerador de sondagem. Permanecem abertos 20 ciclos, profiling, comparação com router, concorrência/cancelamento, APIs completas e pytest formal. O plano não está DONE.

### 20 ciclos persistentes Qwen3.8 e revisão Sol — 13/09/2026

- O harness `gauntlet/test-adaptive-qwen38-cycles.py` executou 20 pares sequenciais `long → mtp` após aquecimento no mesmo processo CUDA, alternando prompts de aproximadamente 20/64/128/256 tokens e `n_predict=1`. Todas as 40 completions e as 40 consultas `/props` retornaram HTTP 200; cada resposta publicou o perfil, o contexto e a residência esperados.
- O resultado `gauntlet/stage-7-cycles-r1/result.json` registra `server_rc=0`, sem erros: longo `context_size=97536`/`mtp_weights_resident=false`, MTP `context_size=56320`/`mtp_weights_resident=true` em todos os 20 pares. Latência de completion: longo mediana `0.3381 s`, p95 `0.5517 s`, máximo `0.5609 s`; MTP mediana `0.3069 s`, p95 `0.3103 s`, máximo `0.3109 s`.
- As amostras NVML por pedido ficaram em `11781..11785 MiB` usados no longo e `11855..11857 MiB` no MTP, sem tendência crescente observável. O log tem uma única carga `load_model`, 20 marcadores `MTP_GPU=0` e 20 `MTP_GPU=348469248`, sem OOM/Xid/erro CUDA/assert/abort/segmentation fault. Pós-teste: produção inativa, porta efêmera livre e 4070 em `16 MiB` usados/`11887 MiB` livres/`0%`.
- A revisão Sol persistente em `gauntlet/stage-7-cycles-sol-review.json`/`.md` respondeu `ACCEPT_WITH_NOTES`: fechou somente o gate funcional dos 20 ciclos. Notas: amostras não substituem profiling de picos/RSS/PSS/fragmentação/bytes; não há contador de reupload do trunk; a sequência não cobre corrida/cancelamento/stream concorrente; `n_predict=1` não é decode sustentado; o limiar 8 é apenas acelerador de teste. Permanecem abertos esses gates, comparação com router, saída útil no teto, APIs completas, pytest formal e deployment.

### Cobertura HTTP/API Qwen3.8 e revisão Sol — 13/09/2026

- O harness nativo `gauntlet/test-adaptive-qwen38-http.py` r3 cobriu na GOKAYA o modelo Qwen3.8 real em uma sequência serial: `/props`, `/models`, `/slots`, completion curto/longo, SSE, replay, Chat, Responses, tools, restore e reuso. Todas as etapas retornaram HTTP 200; o restore publicou novamente MTP (`context_size=56320`, `mtp_weights_resident=true`) e o mesmo prompt de cinco tokens retornou `cache_n=1`, `prompt_n=4`, comprovando reuso parcial após restaurar o snapshot.
- O log r3 (`gauntlet/stage-8-http-qwen38-r3/server.log`) registra uma carga de modelo, duas transições longas e duas MTP (`MTP_GPU=0`/`348469248`), processo `rc=0` e nenhum marcador fatal (`out of memory`, Xid, erro CUDA, GGML_ASSERT, abort ou segmentation fault). Produção permaneceu `inactive`; a 4070 voltou ao baseline de 16 MiB usados/11887 MiB livres/0%.
- A revisão Sol persistente em `gauntlet/stage-8-http-sol-review.json`/`.md` respondeu `ACCEPT_WITH_NOTES`: fechou somente o contrato HTTP serial básico. Notas: `tool_choice=none` valida schema, não uma execução de tool call; replay 200 não prova igualdade em todos os offsets; não há concorrência/cancelamento, decode sustentado, picos, bytes de transferência ou contador de reupload.
- A tentativa formal CPU local (`gauntlet/stage-8-adaptive-http-cpu.log`) executou as chamadas até save e falhou após 18m12s no restore com HTTP 400 por orçamento efetivo: ~433 MiB disponíveis fizeram o limite conservador por arquivo cair para ~65 MiB, enquanto o snapshot tinha 105.799.659 bytes. Não houve crash; essa tentativa não é aprovação formal e precisa ser repetida em ambiente com RAM/GPU suficiente.
- A rodada r1 do harness remoto foi descartada por salvar um prompt de 10 tokens como longo sob limiar 8; a r2 mostrou apenas a semântica esperada de `cache_n=0` para snapshot de um token. A r3 corrigida é a evidência válida.

### Correção do orçamento de rollback e pytest formal GPU — 13/09/2026

- O pytest formal no Qwen3.8 GPU reproduziu um bloqueador real: com `cache-ram=2048`, a captura transitória de rollback com checkpoints excedia `max=429496729` (`used=332870122`, `extra=156895155`). A tentativa anterior falhou no restore curto sem corromper o slot.
- A correção em `tools/server/server-context.cpp` adiciona `include_checkpoints` ao helper `adaptive_slot_capture`. Save explícito e cache continuam capturando checkpoints (`true`); somente `previous_snapshot` do endpoint restore usa `false`. Tokens, estado target, estado draft e carryover MTP permanecem completos e validados; checkpoints são aceleração opcional, não parte necessária da restauração do estado na posição salva.
- Build CPU (`llama-server`/`test-arg-parser`), CTest parser e build CUDA passaram. O pytest formal `tools/server/tests/unit/test_adaptive_context.py` na .57, com Qwen3.8 real, GPU 4070, `PORT=19406`, `cache-ram=2048`, terminou `1 passed in 37.23s`. Cobriu status, troca curta/longo/curta, SSE, replay, Chat, Responses, tools, save/restore; o restore retornou `n_restored>0`, publicou MTP e o reuso final teve `cache_n>0`.
- O log CUDA `gauntlet/stage-8-adaptive-http-gpu-r2-server.log` mostra uma carga de modelo, seleção `10→mtp`, `450→long`, retorno `4→mtp`, `MTP_GPU=0` no longo e `MTP_GPU=348469248` no MTP, evictions RAM explícitas e nenhum `out of memory`, Xid, erro CUDA, GGML_ASSERT, abort ou segmentation fault. Pós-teste: produção inativa, porta 19406 livre e 4070 em 16 MiB usados/11887 MiB livres/0%.
- A revisão Sol persistente em `gauntlet/stage-8-rollback-budget-sol-review.json`/`.md` respondeu `ACCEPT_WITH_NOTES`: confirmou que a cópia de rollback sem checkpoints mantém estado completo, não gera falso hit e deixa save/cache normais intactos. Nota não bloqueante: ainda falta injetar falha pós-mutação com slot contendo checkpoints para medir a degradação de desempenho; profiling, concorrência/cancelamento, saída útil no teto, comparação router e deployment continuam abertos.

### Telemetria de residência e 20 ciclos persistentes — 13/09/2026

- `tools/server/server-context.cpp` agora registra, somente após uma transição bem-sucedida e a reconstrução completa de target/speculation, `transition_ms`, `MTP_GPU`, `MTP_HOST`, `MTP_ALLOC`, `model_instance`, `model_loads`, `main_gpu_upload_bytes`, `mtp_gpu_upload_bytes` e `mtp_reloads`. A leitura ocorre na thread proprietária depois da confirmação de slots ociosos; não muda ownership, rollback ou lifetime.
- Build CPU (`build-luna-model-draft`, `llama-server` e `test-arg-parser`) e CTest do parser passaram; build CUDA remoto (`build-adaptive-cuda`, `.57`, `llama-server`) passou após sincronização do delta.
- Em Qwen3.8-27B real na RTX 4070, porta isolada 19408, 20 pares long→MTP depois do aquecimento produziram 40 completions e 40 consultas de status HTTP 200, `server_rc=0`. `model_instance=1`, `model_loads=1` e `main_gpu_upload_bytes=9.676.118.016` permaneceram constantes. No perfil longo `MTP_GPU=0`, enquanto no curto `MTP_GPU=MTP_HOST=MTP_ALLOC=348.469.248`; `mtp_reloads=20` e o acumulado `mtp_gpu_upload_bytes=7.317.854.208` equivalem à carga inicial mais 20 reuploads da cabeça MTP. A mediana de transição foi 119,931 ms para longo e 273,736 ms para MTP; latência HTTP mediana foi 0,339546 s e 0,307077 s, respectivamente. Não houve `cuda error`, OOM, Xid, assert, SIGABRT ou abort.
- A revisão Sol persistente (`gauntlet/stage-9-telemetry-sol-review.json`/`.md`) respondeu `ACCEPT_WITH_NOTES`, sem blocker. Confirmou os formatos, a leitura pós-reconstrução e a interpretação cumulativa dos contadores. Ressalvas: os bytes são contadores do caminho instrumentado, não tráfego PCIe físico; `transition_ms` não inclui save/restore volumoso nem rollback; profiling físico, router, concorrência e demais gates continuam abertos.

### RSS/PSS e estabilidade de memória — 13/09/2026

- `gauntlet/test-adaptive-qwen38-cycles.py` passou a coletar, após cada requisição, `/proc/<pid>/status` e `smaps_rollup` (`VmRSS`, `VmHWM`, `VmPeak`, `VmSwap`, `Rss`, `Pss`, privados e swap), além da amostra NVML já existente. `python3 -m py_compile` e `git diff --check` passaram.
- A rodada Qwen3.8/4070 na porta 19409 (`gauntlet/stage-10-memory-cycles-r1/result.json` e `server.log`) repetiu 20 pares, 40 completions e 40 props, todos 200, `server_rc=0`. Nos 40 pontos, `VmRSS` ficou entre 1.683.536 e 1.781.320 KiB, mediana 1.727.648 KiB; `VmHWM` máximo foi 1.781.320 KiB; `PSS` ficou entre 1.668.168 e 1.765.952 KiB; `VmSwap` foi sempre 0. Por perfil, VRAM NVML ficou em 11.781–11.785 MiB no longo e 11.855–11.857 MiB no MTP. Não houve crescimento monotônico nos pontos equivalentes, e os contadores do servidor continuaram em uma instância/carga principal e 20 reuploads MTP.
- A revisão Sol persistente (`gauntlet/stage-10-memory-sol-review.json`/`.md`) respondeu `ACCEPT_WITH_NOTES`, sem blocker no harness. Corrigido no registro: o pico `VmHWM` correto é **1.781.320 KiB** (não 1.780.620). Limitações: `/proc` não é amostra atômica, NVML/RSS são pós-request e não capturam pico físico de teardown/upload; `VmPeak` é espaço virtual; profiling de pico/PCIe, router, concorrência/cancelamento, decode sustentado, saída útil no teto e deployment permanecem abertos.
- Conforme decisão explícita do usuário, a produção continua desligada durante a implementação: `llama-server-root.service` permanece inativo, sem listener em 8090/8092; nenhum binário foi promovido ou instalado. A RTX 4070 foi deixada livre após cada rodada.

### Canário default-off Qwen3.5 e revisão Sol — 13/09/2026

- A tentativa upstream selecionada (`unit/test_basic.py`) com TinyLlama não foi aprovação: o fixture não estava cacheado, o servidor iniciou como router sem modelo (`model_path=none`, `/models.data=[]`) e três asserções falharam por esse estado ambiental. Nenhum erro adaptativo foi observado.
- Para validar o caminho carregado, foi criado `gauntlet/test-default-off-qwen35.py`, sem flags adaptativas, speculative ou router, usando o GGUF local `/home/hjotha/models/Qwen3.5-4B-Q4_K_M.gguf`. O canário r3 na RTX 4070/CUDA0, porta 19412, retornou `/props` 200 com `model_path` GGUF e sem `adaptive_context`; `/models` 200 com um item e `context_window=max_context_window=512`, sem `adaptive_context`; `/slots` 200 com uma slot sem campo adaptativo; e `/completion` 200 com `tokens_predicted=1`. O processo terminou `success=true`, `server_rc=0`, com uma carga do modelo, cleanup normal e nenhum marcador adaptativo, CUDA error, OOM, Xid, assert, SIGABRT, abort ou segmentation fault.
- O harness agora persiste os valores efetivos, exige `tokens_predicted==1`, verifica `max_context_window` e transforma `server_rc != 0` em falha. `python3 -m py_compile` passou; a revisão Sol r3 (`gauntlet/stage-11-default-off-sol-review-r3.json`/`.md`) respondeu `ACCEPT_WITH_NOTES`, sem blocker. Notas: a matriz upstream, presets/router, estado sem modelo, concorrência/cancelamento e ferramentas ainda não foram cobertos integralmente.
- Produção continua deliberadamente desligada por decisão explícita do usuário: `llama-server-root.service` inativo, sem listeners de produção e nenhum binário promovido. Os canários usaram apenas portas efêmeras.

### Concorrência serializada, desconexão e revisão Sol — 14/09/2026

- `gauntlet/test-adaptive-concurrency-cancel.py` verificou que o modo adaptativo rejeita configuração incompatível com `--parallel 2` antes de servir (`adaptive context requires --parallel 1`); a rodada funcional usa explicitamente `--parallel 1`, conforme o contrato atual de transição serializada. A primeira tentativa com `--mtp-max-tokens=8` também foi descartada porque o orçamento real de prompt+saída classificou o caso como longo; o harness foi corrigido para `32` antes da medição válida.
- Na GOKAYA, porta isolada 19415, Qwen3.8-27B/CUDA0, três completions concorrentes retornaram HTTP 200 e uma desconexão do cliente durante `n_predict=128` não deixou slot preso: o `/slots` observou ocupado e depois ocioso, a solicitação seguinte retornou HTTP 200 e o perfil final voltou a MTP. O processo terminou `rc=0`; houve uma única carga do modelo, `model_instance=1`, `model_loads=1`, contadores do trunk constantes e nenhum OOM/Xid/assert/abort.
- A revisão Sol persistente em `gauntlet/stage-12-concurrency-sol-review-r3.json`/`.md` respondeu `ACCEPT_WITH_NOTES`. Ela confirmou fila sem task órfã/duplicada após fechamento da conexão e a rejeição explícita de `parallel>1`, mas classificou a evidência como limitada: o cliente desconectado ainda permitiu que a geração produzisse os 128 tokens, portanto não prova cancelamento efetivo durante defer/transition/decode. Permanecem necessários endpoint/caminho explícito de cancelamento, concorrência mais ampla e sanitizers.

### Teto agregado 97.536 e retorno MTP — 14/09/2026

- O harness `gauntlet/test-adaptive-qwen38-max.py` constrói por `/tokenize` exatamente `97.534` tokens de entrada e envia `n_predict=2`, para exercer `97.534 + 2 = 97.536` no perfil longo sem reduzir a entrada silenciosamente. A rodada corrigida r2 na GOKAYA, porta isolada 19417, retornou HTTP 200 com `tokens_evaluated=97534`, `prompt_n=97534`, `tokens_predicted=2` e conteúdo `"<think>\\n\\n"`; a completion curta inicial e o retorno após a transição também foram HTTP 200 com um token previsto.
- O log r2 (`gauntlet/stage-13-max-useful-r2/server.log`) registra uma única ocorrência `load_model`, `model_instance=1`, `model_loads=1` e `main_gpu_upload_bytes=9676118016` constante. A transição longa publicou `n_ctx=97536`, `MTP_GPU=0`; a volta publicou `n_ctx=56320`, `MTP_GPU=348469248`, `mtp_reloads=1`. O processo terminou `rc=0`, sem OOM, Xid, erro CUDA, assert, abort ou segmentation fault; a 4070 voltou a `16 MiB` usados, `11887 MiB` livres e `0%`, com `llama-server-root.service` inativo.
- A correção após a revisão anterior preserva o HTTP da completion de retorno em `short_return_status` antes de consultar `/props`; `python3 -m py_compile` e `git diff --check` passaram. A revisão Sol persistente em `gauntlet/stage-13-max-useful-sol-review-r2.json`/`.md` respondeu `ACCEPT_WITH_NOTES`, sem blocker neste slice.
- O resultado fecha a prova de avaliação integral no teto agregado configurado e da volta MTP no mesmo processo. Não prova uma resposta semanticamente completa, decode sustentado, picos físicos de memória/PCIe, zero reupload físico do trunk, comparação com o router, cancelamento efetivo ou deployment. O marcador `truncated=true` neste caso é a parada física após os tokens previstos com `--no-context-shift`, não truncagem da entrada.

### Fechamento do cancelamento explícito — revisão Sol r6 — 14/09/2026

- O Sol r4 rejeitou o primeiro harness porque `session_seen` aceitava sessão já concluída; a correção r5 passou a exigir `is_done=false`, slot ocupada antes do DELETE, slot livre depois, resposta streaming 200 sem erro e `ignore_eos=true`. O Sol r5 aceitou essa cadeia com notas; a correção r6 acrescentou asserções finais para `profile=mtp` e `mtp_weights_resident=true`.
- A rodada r6 na GOKAYA, porta 19420, repetiu o caso e terminou `success=true`, `server_rc=0`: sessão ativa e slot ocupada antes do DELETE; `DELETE /v1/stream` HTTP 204; POST streaming HTTP 200 com 122 bytes e sem erro; slot livre; replay HTTP 404; geração posterior HTTP 200; estado final `ready`, perfil MTP e cabeça residente. O log correlaciona a task 189 com `stop: cancel task` e `release`.
- A revisão Sol persistente `gauntlet/stage-12-concurrency-sol-review-r6.json`/`.md` respondeu `ACCEPT`, fechando o gate limitado de cancelamento explícito em uma slot. Corridas simultâneas de cancelamento, múltiplos slots, profiling físico, decode prolongado e comparação com router continuam gates distintos.

### Decode sustentado nos dois perfis e revisão Sol — 14/09/2026

- `gauntlet/test-adaptive-qwen38-cycles.py` foi parametrizado por `ADAPTIVE_CYCLES_N_PREDICT` e `ADAPTIVE_CYCLES_MTP_MAX`, mantendo defaults `1/8`. O modo sustentado grava os parâmetros e `tokens_predicted` por pedido, usa `ignore_eos=true` para obter uma janela fixa e falha se o processo terminar com `server_rc` diferente de zero.
- Na GOKAYA, porta 19423, `N_PREDICT=32` e `MTP_MAX=64`, a rodada terminou `success=true`, `server_rc=0`, com 20 pares long→MTP e 40/40 respostas exatas de 32 tokens. Os orçamentos longos foram 135–328 e o curto 33, classificando os perfis conforme o limiar; todos os estados ficaram `ready`, com residência MTP falsa no longo e verdadeira no curto.
- Tempos de decode registrados: longo mediana `839,569 ms` e p95 `842,170 ms` por 32 tokens; MTP mediana `850,969 ms` e p95 `853,785 ms`. O log mostra `7 accepted / 7 generated` por requisição MTP, uma carga do modelo, 20 transições longas e 20 MTP, `main_gpu_upload_bytes=9676118016` constante e nenhum OOM/Xid/erro CUDA/assert/abort/segmentation. Pós-teste: produção inativa e 4070 em 16 MiB usados/11887 MiB livres/0%.
- A revisão Sol persistente `gauntlet/stage-14-decode-sol-review-r3.json`/`.md` respondeu `ACCEPT_WITH_NOTES`, fechando o gate funcional de decode sustentado. `ignore_eos` é deliberado para medir uma janela fixa; terminação natural, comparação com router/controle estático, profiling físico e deployment continuam pendentes.

### Save/restore GPU nativo e revisão Sol — 14/09/2026

- A execução local do CTest que incluía o caso de save/load foi interrompida porque o caminho CPU estava lento; ela não conta como aprovação. O mesmo alvo foi compilado na `.57` com backend CUDA e executado diretamente contra `/home/hjotha/models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf`, `CUDA0`, `ctx=512/1024`, KV `q4_0`, `--backend-sampling` e `--context-lifecycle mtp-cache`.
- `gauntlet/stage-15-gpu-save-load.log` terminou `HARNESS_RC=0` e `lifecycle: PASS (one model; short/long/short; MTP weights evicted and restored)`. A sequência curta→longa→curta usou o mesmo ponteiro/model instance (`model_instance=1`), uma carga principal (`loads=1`), manteve `main_gpu_upload_bytes=9676118016` constante e liberou `mtp_gpu_bytes` no perfil longo (`0`) antes de restaurar `348469248` no retorno MTP.
- O snapshot salvou target, draft e carry separadamente (`prefix=3`, `pos=2`, `target_bytes=156950132`, `draft_bytes=12408`, `carry_bytes=20496`). As restaurações voltaram à posição correta (`restored_prefix=3`, `evaluated_prompt_tokens=1` nas fases longa e final), o carry foi comparado (`PASS`) e as propostas frias foram idênticas (`identical_drafts=2`); as saídas determinísticas das três fases e os hashes finais de trunk/MTP coincidiram.
- O log inclui subcasos negativos de ausência/alocação/upload inválidos; eles falham limpo antes do `PASS`, sem `OOM`, Xid, erro CUDA, assert, abort ou corrupção. Não houve reload integral dos pesos principais. A revisão Sol persistente `gauntlet/stage-15-gpu-save-load-sol-review.json`/`.md` respondeu `ACCEPT_WITH_NOTES`, sem blocker.
- Notas do revisor: `mtp_reloads=2` é cumulativo e inclui o ensaio de falha mais o retorno final; `mtp_gpu_upload_bytes` é contador instrumentado, não tráfego PCIe físico; o caso é serial e usa contextos 512/1024. Portanto, este gate prova save/restore e residência real na GPU, mas não substitui profiling físico, comparação com o router, concorrência, limites máximos ou cobertura API completa.
- Após a revisão, o build CUDA remoto foi recompilado nos alvos `llama-server`, `test-save-load-state`, `test-server-model-identity`, `test-server-prompt-cache` e `test-arg-parser` (`gauntlet/stage-15-cuda-build-final.log`, exit code 0). O CTest selecionado executou `test-generate-models`, `test-server-model-identity` e `test-arg-parser`: 3/3 passaram em 4,24 s (`gauntlet/stage-15-focused-ctest.log`). O `test-server-prompt-cache` ficou apenas compilado porque `LLAMA_TEST_CACHE_MODEL` não está definido nessa configuração; a execução direta desse alvo é CPU por desenho do teste e não substitui o gate GPU.
- O mesmo comando nativo de save/load foi repetido depois desse build contra o Qwen3.8 real, terminou com `exit_code=0` e o log final preserva os marcadores negativos e `lifecycle: PASS` (`gauntlet/stage-15-gpu-save-load-final.log`). Pós-teste confirmado: `llama-server-root.service` `inactive`, RTX 4070 `16 MiB` usados/`11887 MiB` livres/`0%`.

### Capacidade efetiva após cap do modelo e revisão Sol r2 — 14/09/2026

- A revisão final Sol r1 rejeitou um caso de contrato: `adaptive_long_ctx` mantinha o valor bruto de `--ctx-size`, enquanto slots/status/admissão aplicavam `min(llama_model_n_ctx_train())`. Isso permitia iniciar com curto acima do cap e só falhar depois de teardown/transição.
- A correção mínima em `tools/server/server-context.cpp` calcula, após o carregamento do modelo, `n_ctx_train` e `requested_long_ctx_effective`; `adaptive_long_ctx` passa a ser a capacidade operacional única. Uma declaração longa acima do treino é rejeitada explicitamente e chama `destroy()` antes de continuar. `n_ctx_slot()` usa o mesmo campo, e cache, status, admissão, reconstrução, identidade e snapshots continuam referenciando esse valor efetivo.
- `tests/test-arg-parser.cpp` agora cobre curto 600 inválido e curto 400 válido contra teto efetivo 512. `gauntlet/test-adaptive-effective-cap.py` cobre uma inicialização inválida (longo 512 com fixture `n_ctx_train=256`, `server_rc=1`) e uma configuração válida (longo/efetivo 256): `/props`, `/models` e `/slots` publicam 256; prompt de 256 tokens mais uma saída recebe HTTP 400 `exceed_context_size_error` antes de qualquer transição, permanecendo `mtp/ready/resident=true`.
- A execução remota na `.57` passou `test-arg-parser` CUDA (`rc=0`) e o harness completo (`gauntlet/stage-16-effective-cap-remote/result.json`). O serviço de produção permaneceu `inactive`.
- O save/load CUDA do Qwen3.8 real foi repetido depois da correção (`gauntlet/stage-16-effective-cap-remote/stage-16-gpu-save-load.log`): `HARNESS_RC=0`, `lifecycle: PASS`, `model_instance=1`, `loads=1`, `main_gpu_upload_bytes=9676118016` constante, `MTP_GPU=0` no longo e `348469248` no retorno. A carga inicial ocorreu até aproximadamente 23 s no log; cada ida/volta de contexto levou aproximadamente 1,17 s. Os intervalos de eventos de 0,05–0,15 s registrados no lifecycle são compostos e não isolam save ou restore. Portanto, o save/load GPU não é o gargalo dominante; o custo maior é a carga inicial e a recriação dos contextos.
- O diff completo atualizado está em `gauntlet/final-full-review.diff`, SHA-256 `c231310d47a11724eceb647375b93d2628cd6ae097d9b049a46e0b4dacf698cd` (worktree de implementação). A revisão Sol persistente r2 (`gauntlet/final-sol-review-r2.json`) respondeu `ACCEPT_WITH_NOTES`: o REJECT anterior está fechado.
- Permanecem gates separados: profiling físico de PCIe/picos e reupload, comparação com o router, matriz ampla de APIs/default-off/concorrência/sanitizers, identidade/arquivos persistentes, e deployment/rollback. O plano continua em implementação; nenhum binário foi promovido.

### Revalidação do teto Qwen3.8 após o cap efetivo — 14/09/2026

- O canário `gauntlet/test-adaptive-qwen38-max.py` foi repetido com o binário CUDA que contém a correção do teto efetivo, na porta isolada 19424, contra `/home/hjotha/models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf`.
- A tokenização construiu exatamente `97.534` tokens de entrada; `n_predict=2` completou o orçamento agregado `97.534 + 2 = 97.536`, retornando HTTP 200, `tokens_evaluated=97534`, `tokens_predicted=2`, conteúdo `<think>\n\n`, e `truncated=true` somente na parada física sem context shift. O pedido curto de retorno também foi HTTP 200 com um token.
- O log (`gauntlet/stage-17-max-useful/server.log`) registra uma carga do modelo, `model_loads=1`, `main_gpu_upload_bytes=9676118016` constante, transição para longo em `127.684 ms` (`n_ctx=97536`, `MTP_GPU=0`) e retorno MTP em `293.643 ms` (`n_ctx=56320`, `MTP_GPU=348469248`, `mtp_reloads=1`). O prefill máximo levou cerca de `150.1 s`; esse é o gargalo dominante, não save/load ou a troca de perfil.
- Pós-teste: `llama-server-root.service` continua `inactive`, não há listeners 8090/8092 e a RTX 4070 voltou a `16 MiB` usados, `11887 MiB` livres e `0%`.
- Artefatos: `gauntlet/stage-17-max-useful/result.json` e `server.log` na worktree de implementação. A revisão Sol r2 já aceitou a correção do cap; esta rodada acrescenta a revalidação máxima pós-correção.

### 20 ciclos pós-correção e medição de save/restore — 14/09/2026

- A rodada remota pós-correção `gauntlet/stage-18-cycles/result.json` repetiu 20 pares `long → mtp` no Qwen3.8-27B real, CUDA0, porta efêmera 19425, com `n_predict=32` e `mtp_max_tokens=64`. Foram 40/40 completions e 40/40 consultas `/props` HTTP 200; `server_rc=0`, `success=true` e `errors=[]`.
- Em todos os pares, o longo publicou `context_size=97536` e `mtp_weights_resident=false`; o MTP publicou `context_size=56320` e `mtp_weights_resident=true`. Latência mediana foi `1,251792 s` no longo e `1,172192 s` no MTP; a aceitação especulativa foi `140/140` tokens. NVML oscilou apenas entre `11797..11803 MiB` no longo e `11871..11875 MiB` no MTP; RSS ficou entre `1705276..1816696 KiB`, sem marcador fatal.
- O log `gauntlet/stage-18-cycles/server.log` mantém `model_instance=1`, `model_loads=1` e `main_gpu_upload_bytes=9676118016` constantes em todas as transições. As 20 transições para longo mediram aproximadamente `119,5..126,9 ms`; as 20 voltas para MTP, `281,9..287,6 ms`. O acumulado `mtp_reloads=20` sobe somente o contador da cabeça MTP; não há OOM, Xid, erro CUDA, assert, abort ou segmentation fault.
- A medição nativa `gauntlet/stage-16-effective-cap-remote/stage-16-gpu-save-load.log` separou os custos em nível de lifecycle: carga inicial do modelo aproximadamente `23 s`; o teste direto concluiu cada fase em cerca de `0,8–1,2 s`, mas os marcadores de `0,05–0,15 s` são intervalos compostos e não isolam save/restore. Na rodada HTTP real pós-correção, as transições ficaram em `127,684 ms` e `293,643 ms`. Assim, save/load na GPU não é o gargalo dominante; carga inicial e prefill máximo (cerca de `150,1 s` no gate de 97.536 tokens) são os custos relevantes.
- O gate fecha estabilidade serial, residência real e ausência de reload do trunk após a correção do teto efetivo. Ainda permanecem profiling físico de PCIe/picos, comparação quantitativa com o router, concorrência/múltiplos slots, matriz ampla de APIs e sanitizers, persistência de slots e deployment/rollback. O plano continua em implementação.

### Amostragem física PCIe durante save/load — 14/09/2026

- O teste nativo CUDA foi repetido na .57 com `nvidia-smi dmon -s t -d 1 -c 50 -o DT` em paralelo (`gauntlet/stage-19-pcie/nvidia-dmon.log`) e terminou `HARNESS_RC=0`; o lifecycle continuou `PASS`, com uma instância do modelo e uma única carga principal.
- O dmon observou picos físicos de aproximadamente `2744..2759 MB/s` em RX e `3046 MB/s` em TX, além de TX sustentado na faixa de `595..804 MB/s`. A amostragem de 1 segundo não isola cada evento: os picos também cobrem a carga/inicialização CUDA e a atividade do teste negativo. Ela confirma tráfego físico durante a rodada, mas não permite atribuir cada MB/s exclusivamente ao save, restore ou upload MTP.
- A atribuição confiável continua baseada na transição ponta a ponta instrumentada: aproximadamente `120–127 ms` para longo e `282–288 ms` para MTP nos 20 ciclos, e `~0,29 s` para a volta MTP HTTP pós-correção. Os intervalos aparentes de `~135 ms` e `~47 ms` no teste nativo incluem criação/preenchimento de contexto ou uma fase sem restore efetivo; não são medições isoladas de save/restore. Portanto, mesmo com tráfego PCIe observado, save/load não explica a demora percebida nos workloads medidos; a carga inicial e o prefill continuam dominantes.
- Artefatos: `gauntlet/stage-19-pcie/nvidia-dmon.log` e `save-load.log` na worktree de implementação. Profiling de pico com amostragem mais fina, comparação contra o router e os demais gates continuam abertos; o plano não está DONE.
- A revisão Sol persistente (`gauntlet/stage-19-pcie-sol-review.json`, confirmação r2 em `gauntlet/stage-19-pcie-sol-review-r2.json`) respondeu `ACCEPT_WITH_NOTES`: não há blocker para a conclusão estreita de que save/load GPU não é o gargalo dominante nos workloads medidos. A nota exige manter explícita a ausência de tempos isolados de `save_seq`/`restore_seq`, snapshots próximos de 97 mil tokens, bytes físicos por operação, comparação com o router e profiling completo; a documentação acima foi corrigida conforme essa ressalva.

### Integração upstream e builds Release CUDA+Vulkan — 14/09/2026

- O `master` incorporou o upstream atual `origin/master` em `093a2f86c3e37c54fa3e1f9efb17b304f3433abd`, preservando a implementação adaptativa e resolvendo conflitos somente em `src/CMakeLists.txt`, `tests/CMakeLists.txt`, `tools/server/server-models.cpp` e `tools/server/server-models.h`. O merge da integração é `fe3c817f2`; o `master` resultante é `06a245d5e0acee5f94156754215c78a065a99d17`.
- Esse `master` foi publicado com sucesso no fork `fork/master`. As deleções locais preexistentes de `CLAUDE.md` e `benches/dgx-spark/run-aime-120b-t8-x8-high.log` continuam fora do commit.
- O build local da integração foi configurado como `Release`, `-O3 -DNDEBUG`, `-march=native`, com `GGML_VULKAN=ON`; a compilação de `llama-server` e `test-arg-parser` segue em `build-upstream-vulkan` e registra saída em `gauntlet/upstream-vulkan-build.log`.
- A fonte do mesmo SHA foi arquivada em `/home/hjotha/releases/llama-adaptive-06a245d5e` na GOKAYA. A configuração remota do build final foi concluída como `Release` com `GGML_CUDA=ON`, `GGML_VULKAN=ON`, CUDA Graphs, FlashAttention e NCCL; foram detectados CUDA Toolkit `13.3.73`, arquitetura `89-real` e Vulkan `1.4.357`. A compilação remota de `llama-server` e `test-arg-parser` está em `/home/hjotha/releases/llama-adaptive-06a245d5e/build-cuda-vulkan`.
- Nenhum binário foi promovido durante esta etapa; produção permanece desligada até o build e os testes do binário CUDA+Vulkan terminarem. O plano continua sem status DONE enquanto os gates de revisão, smoke test e rollback não forem comprovados no artefato final.

### Build final e promoção CUDA+Vulkan — 14/09/2026

- O build remoto da fonte integrada terminou com exit `0` nos alvos `llama-server` e `test-arg-parser`, em `/home/hjotha/releases/llama-adaptive-06a245d5e/build-cuda-vulkan`. A configuração efetiva foi `Release`, `-O3 -DNDEBUG`, CUDA Toolkit `13.3.73`, arquitetura `89-real`, `GGML_CUDA=ON`, `GGML_VULKAN=ON`, CUDA Graphs, FlashAttention e NCCL; o binário final foi linkado contra `libggml-cuda.so`, `libggml-vulkan.so`, CUDA 13 e Vulkan 1.4.357. SHA-256 do `llama-server`: `609bda686506a347361a276826a6f0a6c095bded246a2fa7edf85a3ed7120806`.
- `test-arg-parser` terminou com `test-arg-parser: all tests OK` e enumerou a RTX 4070 CUDA e os dois dispositivos Vulkan (Radeon Z1 Extreme e RTX 4070). O canário de 20 pares na porta efêmera 19426 terminou `success=true`, `cycles_recorded=40`, `server_rc=0` e `errors=[]`.
- A unit candidata foi validada por `systemd-analyze verify`, instalada em `/etc/systemd/system/llama-server-root.service` e passou a usar diretamente o binário adaptativo, sem `--models-preset`/router. Os drop-ins antigos que sobrescreviam a ExecStart foram removidos após cópia de rollback em `llama-server-root.service.pre-promote`, `slot-save-path.conf.pre-promote` e `zz-upstream-release.conf.pre-promote`; o requisito de GPU `10-gpu-ready.conf` foi preservado.
- A promoção iniciou o serviço com PID `1312977` e listener `0.0.0.0:8090`. Após a carga, `/health`, `/props` e `/models` retornaram HTTP 200; `/props` publicou `enabled=true`, `profile=mtp`, `state=ready`, `context_size=56320`, `context_size_long=97536` e `mtp_weights_resident=true`. Uma completion real na porta 8090 retornou HTTP 200, `tokens_predicted=1` e `tokens_evaluated=5`.
- A primeira tentativa de completion de smoke falhou por quoting do shell (`curl: (6) Could not resolve host: apenas`), sem requisição válida ao servidor; a repetição com JSON escapado passou. A promoção foi feita por decisão explícita do usuário antes de uma nova rodada adversarial Sol para este merge; as revisões Sol anteriores permanecem `ACCEPT`/`ACCEPT_WITH_NOTES`, e os gates abertos de comparação com router, profiling físico completo, concorrência ampla, sanitizers e rollback continuam registrados. O plano não deve ser marcado DONE apenas por este build e smoke test.

### OOM fatal após promoção e parada preventiva — 14/09/2026

- Depois do smoke inicial, o serviço recebeu workloads com saída padrão de `4096` tokens e processou tarefas adaptativas MTP de aproximadamente `8.806`, `10.598`, `3.676` e `11.942` tokens de prompt. Às `07:00:34`, a tarefa `2165` (`56` prompt + `1` output, perfil MTP) falhou em `ggml_cuda_compute_forward: MUL_MAT failed` com `CUDA error: out of memory`; a pilha aponta `ggml_cuda_graph_evaluate_and_capture`/`ggml_backend_cuda_graph_compute`.
- O processo `1312977` terminou com `SIGABRT`, gerou core dump às `07:00:45` e o systemd fez um restart automático (`NRestarts=1`) às `07:00:51`. A RTX 4070 estava em `11901 MiB` usados e `3 MiB` livres imediatamente antes da falha. O primeiro `v1/chat/completions` vazio e a exceção JSON às `06:58:25` foram uma tentativa de shell malformada; o OOM posterior ocorreu em uma tarefa válida.
- O serviço foi parado preventivamente após a coleta do journal e do coredump; estado final comprovado: `llama-server-root.service` `inactive`, sem listener em 8090 e RTX 4070 em `16 MiB` usados/`11887 MiB` livres. A unit nova, o binário e os backups dos drop-ins permanecem preservados para análise/rollback. A promoção não é considerada aprovada: o OOM fatal bloqueia qualquer retorno à produção até isolar e corrigir a pressão de memória/graph capture e repetir os gates.
