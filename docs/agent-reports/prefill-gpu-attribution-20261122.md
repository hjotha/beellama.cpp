# Atribuição do tempo de GPU no prefill de Qwen3.8-Flash-Next

## Implementação

O binário foi compilado a partir de `9f01591478d912204584eeb1c513237cfdbb73f0` (`feat/op-gpu`, base `1fb324283`). O build Noble/CUDA 13.4.92 rodou no host .57, com `-j5`, `CMAKE_CUDA_ARCHITECTURES=80`, FlashAttention e CUDA graphs habilitados. `llama-server`, `test-backend-ops` e `test-llama-archs` foram gerados com `rc=0`. O chroot tinha o código montado em `/work-opt` e o diretório Git ficava fora desse bind, por isso o binário informa `build 0 (unknown)`.

- `ggml/src/ggml-cuda/common.cuh:1387` mantém um pool reutilizável de pares de eventos com `cudaEventDefault`. Os executáveis capturados são destruídos antes dos eventos.
- `ggml/src/ggml-cuda/ggml-cuda.cu:6051` registra os eventos no stream do nó. A guarda usa `CUDART_VERSION >= 11010` e exclui HIP e MUSA, enquanto a guarda complementar em `ggml-cuda.cu:7033` conserva o fallback. O header CUDA 13.4 deste build define o símbolo como macro. A nova guarda depende da versão que declara a API, e `nm` confirmou a referência a `cudaEventRecordWithFlags` no objeto. Durante captura, `cudaEventRecordExternal` coloca a gravação no grafo para que o replay atualize os timestamps. A execução direta usa `cudaEventRecord`.
- `ggml-cuda.cu:6064` sincroniza uma vez ao final do grafo e acumula os pares por `ggml_op_name`. Fusões são contabilizadas no primeiro nó. Operações com menos de 2% no grafo entram em `total_ms` e são omitidas da linha. Os intervalos de streams concorrentes podem se sobrepor.
- `ggml-cuda.cu:7028` liga o profiler com `LLAMA_OP_PROFILE_GPU=1` em grafos com mais de 200 nós. O servidor registra `[OP GPU]` com `-lv 4`. O avaliador padrão é compilado sem gravação de eventos no laço de nós. `GGML_CUDA_DISABLE_GRAPHS=1` mantém disponível a execução direta nos toolkits sem eventos externos.

## Validação na .57

| Teste | Resultado |
| --- | --- |
| `test-backend-ops -b CUDA0 -o MUL_MAT_ID` | 1.088/1.088 casos executados no CUDA0, 2/2 no resumo do runner |
| `test-backend-ops -b CUDA0 -o MUL_MAT` | 2.604/2.604 casos executados no CUDA0, 2/2 no resumo do runner |
| Qwen3.5-4B-Q4_K_M, profiler ausente | 895 tokens de prompt, 24 gerados, zero linhas `[OP GPU]` |
| Qwen3.5-4B-Q4_K_M, profiler ligado | Mesmo prompt e geometria, 28 linhas `[OP GPU]`, zero falhas na leitura dos eventos |
| Captura e replay | Em cada modo, 9 execuções diretas, 1 captura com lançamento e 18 replays |
| SHA-256 do texto gerado nos dois modos | `a8773e90c301aacdc0788818a6c116acfa387e4b27672de72535af5d920b8394` |

O runner filtra pelo CUDA0 e marca o CPU como pulado no resumo. O servidor de teste rodou na RTX 4070 com `-ngl 99 -c 4096 -b 512 -ub 256 -fa on --no-warmup --fit off --parallel 1 -lv 4`. Usei temperatura zero, `cache_prompt=false` e o trace de CUDA graphs nos dois modos para confirmar replay. Corrigi a serialização do script auxiliar e repeti os dois pedidos. Encerrei o servidor de teste na porta 8197 e confirmei `{"status":"ok"}` na porta 8090 após religar `llama-server-root.service`.

## Pacote

O ZIP tem 191.315.854 bytes e SHA-256 `99ec968667c5bec97e3e018e850e8a3d6bc1bafa4edfcee8f57f6fd06e97fe96`. A verificação de integridade do ZIP passou, e os três executáveis estão em `beellama/bin`. [Release A100 sm_80 parity build 8](https://github.com/hjotha/beellama.cpp/releases/tag/collabosm-a100-parity8-20261122)

O GitHub confirmou o mesmo tamanho e SHA-256 do asset. A tag do release foi
criada sobre o `master` remoto pelo comando `gh release create`. O commit que
gerou o ZIP está na branch local `feat/op-gpu`.

## A100-40G

As duas invocações de `colab new -s fn40gpu --gpu A100` no host .157
terminaram com `ColabRequestError: Service Unavailable` no POST de alocação
`/tun/m/assign` com `accelerator=A100`. Em ambas, `colab stop -s fn40gpu`
informou `Session 'fn40gpu' not found` e `colab sessions` confirmou que a
`fn40gpu` estava ausente. A sessão preexistente `fn40iq3` permaneceu ativa.

| Caso | Estado | Tempos por op, sanidade e VRAM |
| --- | --- | --- |
| Prefill 28k, 620 blocos do prompt | Bloqueado na alocação | Sem coleta |
| Prefill 4k, 100 blocos do prompt | Bloqueado na alocação | Sem coleta |

O streaming dos shards e as requisições não chegaram a começar. A verificação
de ±15% entre a soma de `total_ms` e `prompt_ms` ficou pendente para os dois
tamanhos. Na validação local, grafos de decode também ultrapassaram 1.024 nós.
A separação das fases deve usar o limite do fim do prompt, com o número de nós
como filtro adicional dos grafos de prefill. A atribuição por operação, as
cinco operações dominantes e a escolha do próximo patch de kernel dependem
das medições válidas na A100.
