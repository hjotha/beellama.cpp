# Atribuição do tempo de GPU no prefill de Qwen3.8-Flash-Next

**Estado em 22/11/2026:** instrumentação implementada sobre `main` (`1fb324283`). O sandbox bloqueou o worktree no caminho solicitado e o bind para o chroot Noble. O worktree da branch `feat/op-gpu` ficou em `/tmp/beellama-opgpu.SnOJ3K/beellama-opgpu`. O `sudo` retornou `The "no new privileges" flag is set, which prevents sudo from running as root.` O build A100, o pacote, o release e a medição ficaram pendentes. Nenhuma sessão Colab foi criada.

## Implementação

- `ggml/src/ggml-cuda/common.cuh:1387` mantém um pool de pares de eventos temporizados por backend, reutilizado entre grafos. Os executáveis capturados são destruídos antes dos eventos.
- `ggml/src/ggml-cuda/ggml-cuda.cu:6544` mede os nós no stream em que rodam. As fusões são atribuídas ao primeiro nó da sequência. Em execução direta, o início dos nós não fundidos é registrado novamente depois da análise de fusão. Durante a captura, `cudaEventRecordExternal` incorpora os eventos ao grafo para registrar timestamps a cada replay. [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EVENT.html)
- `ggml/src/ggml-cuda/ggml-cuda.cu:6064` espera uma vez pelo stream ao fim do grafo, lê os pares e agrega por `ggml_op_name`. O `total_ms` inclui as ops com menos de 2%. Essas ops são omitidas da linha. Para ver `[OP GPU]`, use `LLAMA_OP_PROFILE_GPU=1` e `-lv 4` no servidor. A linha só sai para grafos com mais de 200 nós.
- `ggml/src/ggml-cuda/ggml-cuda.cu:7028` seleciona o avaliador instrumentado por grafo. Com a flag desligada, o laço de nós não tem chamadas de eventos nem checagens do profiler. A execução direta pode ser perfilada com `GGML_CUDA_DISABLE_GRAPHS=1`.

## Validação

`git diff --check` passou. Um teste local em C++ com eventos simulados verificou reutilização e destruição do pool, escolha entre evento direto e externo, sincronização antes da leitura, soma por operação, inclusão de uma operação com exatamente 2% e omissão de uma com 1%. Um erro simulado de leitura produziu um diagnóstico sem publicar totais parciais.

A validação real ficou pendente para `test-backend-ops -b CUDA0 -o MUL_MAT_ID` e `-o MUL_MAT` (2/2 backends esperado), replay de CUDA graphs, SHA-256 do texto com a flag ligada e desligada na mesma geometria e ausência de linhas `[OP GPU]` com a flag desligada.

## A100-40G

| Caso | Ops por ms e % | Soma de `total_ms` | `prompt_ms` e diferença | Pico de VRAM |
| --- | --- | --- | --- | --- |
| Prefill de 28.542 tokens | Pendente | Pendente | Pendente | Pendente |
| Prefill de 4k | Pendente | Pendente | Pendente | Pendente |
| Decode | Pendente | Pendente | Não se aplica | Pendente |

O critério de sanidade é comparar a soma dos `total_ms` dos grafos de prefill (pelo menos 1.024 nós por ubatch no protocolo) com o `prompt_ms` da mesma requisição. A banda é ±15%. Os intervalos medidos em streams concorrentes podem se sobrepor, e as operações abaixo de 2% em cada grafo contribuem para o total sem aparecer na atribuição por op.

As cinco ops dominantes e suas causas ainda não podem ser identificadas. O próximo patch de kernel de maior retorno depende do ranking medido e da comparação com o `prompt_ms` na A100.
