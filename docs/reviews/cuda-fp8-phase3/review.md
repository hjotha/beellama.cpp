# Review da Fase 3: codec FP8 CUDA, staging pinned e eventos

Parecer: **aritmética aprovada nos vetores examinados; homologação da Fase 3 pendente**. O codec direto implementa corretamente o contrato E5M2 finito/RNE. A interface ainda tem um erro de ABI, retornos de erro sem aposentadoria das operações CUDA e lacunas de layout, ordenação e integração. O caminho de serving continua usando o codec CPU; os ganhos de PCIe/USB4 descritos no pedido ainda não decorrem deste diff.

Escopo: checkout BeeLlama.cpp, branch `docs/hybrid-attention-master-review`, HEAD `aaa63758a968577c593654d27b80de748dd15c6a`, incluindo fontes locais não commitados. Os hashes estão em [sources.json](sources.json). Não alterei fontes de implementação, nem fiz commit, merge, push ou deploy. Os testes adicionais estão neste diretório; binários e objetos temporários ficaram em `/tmp/beellama-phase3-review`.

## P1 — O proc da interface devolve dados onde o header promete uma função

Referências: `ggml/include/ggml-remote-attn-cuda.h:57`, `ggml/src/ggml-cuda/ggml-cuda.cu:6822`.

O typedef público é um ponteiro para getter `const interface * (*)(void)`. O registro, porém, chama o getter e devolve o endereço da tabela. Um consumidor que faça o cast declarado e invoque `get_interface()` tentará executar a tabela como código. Os procs individuais de pack/unpack não têm esse erro.

Correção exata:

```diff
-        return (void *)ggml_backend_cuda_remote_attn_fp8_interface();
+        return (void *)ggml_backend_cuda_remote_attn_fp8_interface;
```

O patch [abi-cuda-only.patch](abi-cuda-only.patch) contém essa correção e a restrição de compilação CUDA descrita abaixo. `git apply --check` passou. O patch é uma proposta para revisão, não foi aplicado aos fontes. Ainda é necessário um teste que resolva o proc pelo registry real, faça o cast do header e invoque o getter. O probe desta auditoria chama o getter diretamente e, portanto, não valida esse registro.

## P1 — Retornos de erro podem liberar o scratch antes do término do trabalho

Referências: `remote-attn-cuda.cu:217`, `:222`, `:235`, `:238`, `:243`, `:284`, `:288`, `:293`, `:306`, `:315`, `:320`.

Depois de enfileirar uma cópia ou kernel, várias falhas retornam imediatamente sem sincronizar o stream. Um exemplo é a falha de `cudaEventRecord` após kernel e DMA já terem sido submetidos. A função libera `g_scratch_mutex`, embora ainda possa haver trabalho usando `dev_data`, `dev_err`, `host_err` e os buffers do chamador. A próxima chamada pode usar outro stream e reutilizar o mesmo scratch. O chamador também pode interpretar `false` como permissão para liberar o pinned ou o tensor.

O caminho saudável evita esse reuso concorrente porque mantém o mutex até `cudaStreamSynchronize`. Essa garantia precisa valer para os caminhos de erro também. Para uma primeira versão síncrona, concentrar a saída após submissão em um bloco de retirement, ainda sob exclusão do scratch; verificar erros de lançamento imediatamente; aguardar trabalho submetido antes de devolver ownership. Se um erro fatal impedir comprovar aposentadoria, invalidar o contexto/sessão de aceleração e impedir seu reuso. Não tratar um evento cuja gravação falhou como uma fence válida.

`err_flag` é coerente quando a cópia do status e a sincronização terminam com sucesso. Entretanto, ele não cancela os outros threads do kernel e não torna a operação transacional. Pack pode copiar bytes antigos nas posições inválidas antes de retornar `false`; unpack pode ter escrito parte do destino. Nenhum consumidor ou envio TCP pode usar esses resultados até a conclusão e a validação do status. Para o critério do plano de não publicar resultado parcial em falhas, é preciso preservar essa barreira ou decodificar em scratch e publicar somente após validação.

Não foi necessário executar um teste de use-after-free para estabelecer esse defeito de ownership: as saídas antecipadas estão explícitas no fluxo.

## P1 — Falta o contrato de dependência com o produtor de Q

Referências: `remote-attn-cuda.cu:146`, `:206`, `:277` e a interface pública.

O stream privado é `cudaStreamNonBlocking`. Receber um endereço de VRAM não garante que as projeções de Q terminaram em outro stream GGML. Sincronizar o stream do pack ao final não estabelece a dependência de entrada. É preciso usar o próprio stream produtor, ou gravar um evento nele e fazer o stream de pack esperar esse evento. A NVIDIA documenta que streams nonblocking não sincronizam implicitamente com o stream zero e que `cudaStreamWaitEvent` ordena o trabalho futuro pelo evento. [Documentação NVIDIA](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__STREAM.html).

A primeira execução do probe fez uma cópia H2D sem confirmar sua conclusão antes de usar o stream separado. A rota host passou, mas a rota device retornou uma divergência, registrada em [first-run-unjoined-copy.log](first-run-unjoined-copy.log). Com a entrada explicitamente pronta, todas as comparações passaram. Esse experimento é compatível com a ausência da dependência; não o apresento como um teste determinístico da corrida. O teste final usa `cudaDeviceSynchronize` apenas para isolar a aritmética no harness; essa sincronização global não é a solução proposta para serving.

O contrato deve dizer também quando o consumidor da saída pode executá-la. Em uma futura interface realmente assíncrona, o evento de OUT deve ordenar o stream consumidor, e a confirmação de status precisa anteceder a publicação de sucesso.

## P1 — A aceleração ainda não está conectada ao serving

Referências: `ggml/src/ggml-remote-attn/ggml-remote-attn.cpp:1351`, `:1380`, `:776`.

Os novos símbolos aparecem nas declarações, definições e registro; não há consumidor no backend remote-attn ou nos testes do repositório. A preparação continua fazendo `ggml_backend_tensor_get` de Q completo para `slot.f32_scratch` e depois pack CPU. A resposta continua sendo expandida pelo codec CPU. Os slots da Fase 6 usam vetores comuns e não recebem os buffers do novo `alloc_pinned`.

Chamar o helper depois desse `tensor_get` não recuperaria a economia desejada: a rota host faz H2D de F32/F16 e D2H de FP8. Da mesma forma, unpack para destino host faz H2D compacto e D2H F32. A economia exige capturar Q residente em CUDA antes da cópia completa e entregar OUT ao destino CUDA efetivamente consumido pelo grafo.

O fator 4x para Q F32 e 2x para Q F16 descreve a redução do componente de bytes dessas ativações quando o caminho residente estiver integrado. Não é uma redução medida do tráfego total: LSE, K/V e outras cópias continuam relevantes. Não houve benchmark de serving nesta auditoria.

## P2 — O layout de scratch não alinha os campos float

Referências: `remote-attn-cuda.cu:269–282` e leitura de LSE na linha110.

`dev_lse = dev_fp8 + fp8_bytes` converte um offset de bytes diretamente em `float*`. Quando `nq*out_width` não é múltiplo de quatro, o LSE fica desalinhado. O destino intermediário da rota host herda esse desalinhamento.

Teste real: `nq=1`, `out_width=3`, `n_head=1`, LSE válido e destino CUDA alinhado. O Compute Sanitizer identificou uma leitura global de quatro bytes desalinhada, seguida de erro CUDA 719 e erros derivados. A execução terminou com exit 77, definido no comando do sanitizer. Evidências: [memcheck-odd-width.log](memcheck-odd-width.log), [stdout](memcheck-odd-width.stdout).

Isso não aparece no layout RKVA usual `out_width=n_head*256`, mas a API pública aceita a largura 3 sem rejeição. Alinhar os offsets internos, mantendo o wire compacto inalterado:

```cpp
// Após verificar os produtos/somas contra SIZE_MAX:
const size_t lse_offset = (fp8_bytes + alignof(float) - 1) & ~(alignof(float) - 1);
const size_t out_offset = lse_offset + lse_bytes;
const size_t needed = out_offset + f32_out_bytes;
auto * base = static_cast<uint8_t *>(g_scratch.dev_data);
float * dev_lse = reinterpret_cast<float *>(base + lse_offset);
float * dev_f32_out = is_dev ? static_cast<float *>(dst_out)
                             : reinterpret_cast<float *>(base + out_offset);
```

O snippet pressupõe aritmética previamente verificada; não deve ser copiado sem essas verificações.

## P2 — Stride, LSE ausente e tamanhos não são validados

Referências: `remote-attn-cuda.cu:191`, `:200–202`, `:225`, `:261`, `:269–273`, `:295–298`.

Um stride device não múltiplo de quatro é truncado por divisão inteira. Um stride menor que `(out_width+n_head)*sizeof(float)` faz linhas se sobreporem; em CUDA isso pode criar escritores concorrentes no mesmo endereço. A rota host também admite cópias de linhas sobrepostas. Validar o stride mínimo, seu alinhamento e o alinhamento dos ponteiros usados como F32/F16. O pack só representa arrays contíguos: views GGML com strides arbitrários precisam de verificação ou parâmetros próprios.

Com `n_head>0` e `src_lse=nullptr`, o kernel deixa a parte LSE intocada e a função retorna sucesso. O probe confirmou `return=true, output LSE=-123`, preservando o sentinel anterior: [missing-lse.log](missing-lse.log). Na rota host, a cópia inclui essa parte não escrita do scratch. Rejeitar essa combinação ou definir uma opção explícita e uma semântica completa para saída sem LSE.

As multiplicações de elementos por bytes, somas de regiões, extent da saída e arredondamento/narrowing do grid não têm proteção contra overflow. Esses argumentos são internos ao processo, portanto isso não demonstra por si só um ataque remoto; ainda assim, a API não pode aceitar dimensões que alocam menos scratch do que o kernel acessa. Fazer checked-add/checked-mul antes de alocar ou enfileirar, validar o limite do grid e calcular o extent estridado como `(nq-1)*stride + row_bytes`, também com checagem. Quando o chamador possui o tensor GGML, validar esse extent contra o armazenamento real.

## P2 — A inclusão automática da unidade afeta HIP/MUSA

Referências: `ggml/src/ggml-hip/CMakeLists.txt:63`, `ggml/src/ggml-musa/CMakeLists.txt:32`, `ggml/src/ggml-cuda/vendors/hip.h:99`, `vendors/musa.h:73`.

Os backends HIP e MUSA incluem todos os arquivos `ggml-cuda/*.cu`. O novo arquivo usa `cudaHostAlloc` e `cudaHostAllocPortable` sem os guards que já existem em `allreduce.cu`. Os adapters examinados mapeiam `cudaMallocHost`, mas não esses símbolos. Há uma regressão de compilação identificável estaticamente nessas configurações; não executei builds HIP/MUSA, pois as toolchains correspondentes não foram qualificadas aqui.

Para uma Fase 3 especificamente CUDA, o patch proposto restringe tanto a implementação quanto os registros a `!GGML_USE_HIP && !GGML_USE_MUSA`. Outra alternativa é implementar e testar os adapters com a semântica necessária; não anunciar o proc nessas plataformas antes disso. Como CMake usa `file(GLOB)` sem `CONFIGURE_DEPENDS`, um build já configurado também precisa de reconfiguração para incorporar a nova unidade.

## P2 — Streams/eventos existem, mas a API atual é síncrona e globalmente serializada

Referências: `remote-attn-cuda.cu:129–130`, `:196`, `:246`, `:266`, `:323`.

Toda chamada saudável espera `cudaStreamSynchronize` segurando um mutex global. O evento é gravado antes dessa espera e já estará concluído quando a função retornar sucesso. Isso pode ser uma primeira implementação síncrona correta, mas não implementa a sobreposição de cópias e pack/unpack entre os dois slots. O bool também reúne erro de submissão e resultado numérico, que só é conhecido após a conclusão.

Não remover apenas os dois synchronize: isso tornaria o retorno de status prematuro e liberaria o scratch compartilhado ainda ativo. Uma interface assíncrona precisa de estado por operação/slot, status persistente e operações distintas de submissão e conclusão. Preservar uma interface síncrona explícita até esse ownership estar implementado é uma opção segura.

`cudaHostAllocPortable` fornece memória pinned portável entre contextos; o código não usa mapped memory nem `cudaHostGetDevicePointer`. A expressão “zero-copy DMA” no header deve ser ajustada para staging pinned de DMA: há cópias H2D/D2H explícitas.

## P2 — Device e duração do scratch precisam de contrato explícito

Referências: `remote-attn-cuda.cu:132–143`, `:167–181`, `:193`, `:264`.

O device usado é o corrente da thread chamadora. Um worker TCP pode ter um device corrente diferente daquele do tensor. Um ponteiro device de outra GPU é classificado como host, em vez de ser rejeitado ou tratado explicitamente. Managed memory também cai nessa classificação; sua semântica deve ser definida, sem inferir que qualquer ponteiro não reconhecido é um array host comum.

Capturar a identidade CUDA do tensor/backend, validar stream/eventos contra esse contexto e usar um guard de device que restaure o estado do chamador. O comportamento de ponteiros de outra GPU e managed não foi testado em múltiplas GPUs nesta auditoria.

O singleton destrói e recria o scratch ao alternar dispositivos, ignora erros de algumas operações de limpeza/device switch e não tem teardown final. Isso é um cache de memória limitado pelo maior pedido no device corrente, não um vazamento novo por token no caminho saudável. Entretanto, conflita com isolamento de sessões, descarregamento de backend e warmup estável entre GPUs. Preferir estado com ownership explícito por backend/device/slot e destruição após retirement.

## Contrato numérico

Não encontrei defeito na conversão escalar CUDA apresentada:

- A separação por bits rejeita Inf/NaN antes da saturação e preserva o sinal de zero.
- `0x47600000` representa 57344. Para magnitudes acima do maior E5M2 finito, saturar em `0x7b/0xfb` é correto.
- O menor subnormal E5M2 é `2^-16`; seu midpoint com zero é `2^-17` (`0x37000000`), que empata para zero par. O ramo `<=` está correto, inclusive para valores negativos.
- `0x38800000` é o menor normal E5M2, `2^-14`. No ramo subnormal que efetivamente alcança o round-shift, o shift é 24, 23 ou 22. No normal é 21. Não há shift por zero nem por 32 nesse domínio.
- O arredondamento da mantissa pode carregar para o expoente seguinte, como deve. A entrada já limitada ao domínio finito/saturado impede gerar o expoente 31 por esse carry.
- F16→F32 é exato para valores finitos, portanto essa expansão antes da quantização direta não introduz o problema antigo de duplo arredondamento F32→F16→FP8.
- FP8→F16 por deslocamento de oito bits preserva todos os 248 códigos finitos, incluindo subnormais e zeros assinados. Os oito códigos de expoente 31 são rejeitados.

O empacotamento F32 usa operações inteiras nos bits; não depende do rounding mode de uma operação de ponto flutuante intermediária. A referência independente usada foi `__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E5M2)`, cuja semântica RNE é documentada pela [NVIDIA Math API](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__MISC.html).

## Validação executada

Hardware consultado: `NVIDIA GeForce RTX 4070, 12282 MiB, 8.9`. Toolkit: `Cuda compilation tools, release 13.4, V13.4.92`.

Compilação da unidade CUDA e link do probe passaram com `-O2 -std=c++17 -arch=sm_89`. O primeiro link sem otimização/section GC exigiu dependências de funções não usadas trazidas por `common.cuh`; foi corrigido no harness com otimização e coleta de seções não utilizadas, sem modificar a unidade examinada. Isso é um build isolado do helper, não uma build completa de BeeLlama ou validação do registry.

Comandos finais, a partir da raiz do repositório:

```bash
/opt/cuda/bin/nvcc -O2 -std=c++17 -arch=sm_89 -Xcompiler -ffunction-sections -Xcompiler -fdata-sections -Iggml/include -Iggml/src -Iggml/src/ggml-cuda -c ggml/src/ggml-cuda/remote-attn-cuda.cu -o /tmp/beellama-phase3-review/remote-attn-cuda.o
/opt/cuda/bin/nvcc -O2 -std=c++17 -arch=sm_89 -Xlinker --gc-sections -Iggml/include -Iggml/src/ggml-cuda /tmp/beellama-phase3-review/probe.cu /tmp/beellama-phase3-review/remote-attn-cuda.o -o /tmp/beellama-phase3-review/probe
/tmp/beellama-phase3-review/probe valid
/tmp/beellama-phase3-review/probe missing-lse
/opt/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 77 --log-file /tmp/beellama-phase3-review/memcheck-valid.log /tmp/beellama-phase3-review/probe valid
/opt/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 77 --log-file /tmp/beellama-phase3-review/memcheck-odd-width.log /tmp/beellama-phase3-review/probe odd-width
```

Resultados em [numerical.log](numerical.log):

| Verificação | Resultado |
|---|---|
| F32 com entrada host e device | 1.045.703 valores por rota; zero divergências contra o conversor NVIDIA |
| F16 com entrada host e device | Todos os 63.488 padrões finitos por rota; zero divergências |
| Rejeição de F32 especiais | Cinco padrões individuais rejeitados |
| Rejeição de F16 especiais | Um batch contendo todos os 2.048 padrões não finitos foi rejeitado |
| Unpack finito, host e device | Todos os248 códigos exatos, incluindo sinais de zero |
| Unpack com stride válido | Duas linhas de largura 256, LSE e padding conferidos |
| Unpack de FP8 especiais | Todos os oito códigos não finitos rejeitados individualmente |
| LSE não finito | +Inf, -Inf e NaN rejeitados |
| Memcheck dos casos válidos | Exit 0; `ERROR SUMMARY: 0 errors` |
| LSE ausente | Defeito confirmado: sucesso com sentinel preservado |
| Largura 3 com LSE | Defeito confirmado: leitura desalinhada; exit 77 do sanitizer |

A população F32 inclui midpoints e vizinhos representáveis de todas as transições entre códigos positivos finitos adjacentes, os correspondentes negativos, extremos e amostra pseudoaleatória por bits. Não é uma enumeração de todos os padrões F32. A rejeição do batch de especiais F16 não equivale a testar cada padrão não finito individualmente.

Não foram executados build completo CUDA/HIP/MUSA/Windows, CUDA Graph capture, stress de cancelamento, multi-GPU ou benchmark end-to-end. O probe é evidência adicional da auditoria e deve virar teste integrado com fail/skip explícito quando CUDA não estiver disponível.

## Integração necessária com as Fases 6 e 7

1. Corrigir primeiro o ABI, retirement em erro e os contratos de layout/device. Não ligar um caminho novo ao serving enquanto esses erros forem possíveis.
2. Cada slot deve manter seu próprio staging pinned, scratch/status CUDA, eventos e identidade `{epoch,ticket,device}`. Reservar e crescer fora do trecho ativo; liberar somente após aposentadoria das operações e dos consumidores.
3. Ordenar produtor de Q → pack → D2H → confirmação de status → envio TCP. Não acessar o tensor de origem novamente depois de seu lifetime; a fence deve pertencer à execução que o produziu.
4. Ordenar resposta TCP → H2D → unpack → confirmação de status → consumo no grafo. Um ACK de rede não significa que OUT já está pronto na GPU. DONE_OK, wait e drain devem preservar essa distinção, inclusive no cancelamento.
5. Para o wavefront, conclusão de tile/camada deve depender da fence do consumo CUDA, não de uma contagem de ondas ou do retorno de uma submissão. Reuso de buffers exige também o término do consumidor. Os bloqueadores anteriores das Fases 6/7 permanecem sujeitos a seus reviews próprios.
6. Qualificar o caminho real com o registry, tensores CUDA residentes, dois slots ativos, cancelamento/drain em cada estágio e contadores de bytes antes de medir ganho. Registrar F32/F16 inicial, wire negociado e todos os fallbacks CPU.

