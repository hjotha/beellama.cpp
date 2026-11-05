# Investigação da diferença de decode entre fork e BeeLlama oficial — 2026-11-05

## MTP e limites de contexto promovidos após a investigação

Em 2026-11-05, foi promovido o preset `/home/hjotha/router-production-kvarn4-e4-draft-kvarn4-20261105.ini`: S passou de 32.768 para 36.864 tokens e `p-min` de 0,65 para 0,80; M passou de 45.056 para 56.320 e `p-min` de 0,60 para 0,70; L passou de 58.368 para 75.776 e manteve `p-min` 0,65. `profit` permaneceu desligado. O processo de produção confirmou os argumentos, inferência e S → M → L → S; `/health` e `/metrics` responderam. O backup anterior ao deploy está em `/home/hjotha/.codex/artifacts/mtp-parity-tuning-20261105/promotion/preset-before.ini`, e a validação em `promotion/result.json` sob o mesmo diretório.

O controlador `profit` suporta MTP nesta árvore e ajusta a profundidade máxima de draft; ele não ajusta `p-min`. Em uma rodada com três prompts e 256 tokens, seus resultados não mostraram vantagem consistente. Para os primeiros 128 tokens de três prompts, os IDs do fork e do oficial foram idênticos; os aceites foram 48/48, 87/86 e 85/84, respectivamente. Em 512 tokens, as trajetórias divergiram e as contagens variaram nos dois sentidos. Dados em `/home/hjotha/.codex/artifacts/mtp-parity-tuning-20261105/parity-results.json` e `common-prefix/parity-results.json`.

Em três prompts com 512 tokens, os logs preservados mostram no S 24,96/25,92/25,40 tokens/s com 0,65 e 27,60/26,32/26,12 com 0,80; controles e candidato foram repetidos. No M, 0,70 foi superior ao controle nos três prompts dessa rodada, com resultados repetidos de 26,83/30,47/27,12 tokens/s. Esses ganhos dependem da carga. Os logs sobreviventes estão em `tuning-preserved/` sob o diretório de artefatos. Um erro no script reutilizou o diretório de saída e sobrescreveu parte dos logs e JSONs das rodadas anteriores, inclusive o controle consolidado do M; os valores apresentados ao usuário para esses ensaios vieram das saídas dos comandos, mas o arquivo original já não está disponível. A pedido do usuário, a promoção foi feita sem repetir a matriz inteira.

Com a configuração candidata, requisições isoladas de contexto máximo menos 1.024 tokens e 1.024 de saída passaram com amostragem de VRAM a cada 100 ms. S em 36.864 teve mínimo de 69 MiB em duas rodadas; M em 56.320 teve 59 MiB; L em 75.776 terminou com 61 MiB, com mínimo de 59 MiB durante a transição desde M. As sequências S → M → L → S foram verificadas em duas combinações de limites; os limites finais de M e L passaram juntos, com volta ao S final. Não houve aviso de desativação de CUDA graphs nessas rodadas. Evidências em `vram-expanded-2/` e `vram-expanded-4/` no diretório de artefatos acima. Os ensaios isolados desativaram o auto-restore de snapshots; a verificação posterior de produção usou requisições curtas.

O ensaio S em 40.960 chegou a 9 MiB livres e desabilitou CUDA graphs; foi interrompido e descartado. L em 81.920, após M cheio, chegou a 3 MiB, desabilitou CUDA graphs e falhou com OOM; foi descartado. Evidências em `vram-expanded-final/` e `vram-expanded-1-preserved/`.

## Objetivo e estado das conclusões

Investigar por que o decode com MTP do fork é mais lento que o BeeLlama oficial na RTX 4070, sem sacrificar qualidade, rollback, transições de tier ou o funcionamento do Xbox. O `main` usado como base é `4bad0205c`, após o merge do upstream `4260165a9`. Os experimentos foram feitos em builds isoladas; o estado das mudanças promovidas está registrado abaixo.

O A/B repetido mostra uma diferença real para **este prompt**: oficial aproximadamente 27,4–27,5 tokens/s e fork VMM aproximadamente 26,3–26,5 tokens/s. O fork aceita menos propostas de MTP e usa CUDA graphs de forma diferente. O trace também mostra maior tempo em alguns kernels do fork. **Nenhuma dessas observações, isoladamente, prova a causa completa da diferença.** A saída dos dois modelos diverge após 39 tokens, e a aceitação muda o número de verificações; o custo por passo também precisa ser medido com a mesma trajetória.

O candidato que combina a correção de checkpoint, a compactação da cauda e o cálculo paralelo de Q apresentou ganho repetido no A/B e passou nos testes funcionais e de transição descritos abaixo. A carga de referência chegou a desempenho semelhante ao oficial; isso não estabelece paridade geral para todos os prompts e tiers.

Em 2026-11-05, as mudanças validadas foram integradas no `main` no commit `878e29e2c` e promovidas ao serviço `llama-server-root.service`. O release ativo é `/home/hjotha/beellama-releases/decode-cadence-878e29e2c-20261105/bin`, com `libllama-server-impl.so` SHA-256 `7235ba8489dff3ee57aa67e2d1ba962eca85cdd715da9e9e5f0207f44e942b31` e `libggml-cuda.so.0.27.0` SHA-256 `c67bd548a38b7d4346b5eb17f06044647292ab2f98b0b8dab15023d44f6bb6a4`. O bundle foi montado a partir do release anterior com os objetos alterados recompilados e religados; não houve rebuild CUDA integral. Após o deploy, inferência, `/metrics?model=qwen-3.8-27b` e S→M→S passaram, e o processo carregou as bibliotecas do release correto. O preset e VMM foram preservados.

O drop-in de promoção é `/etc/systemd/system/llama-server-root.service.d/zzzzzzzzzzzzzz-decode-checkpoint.conf`. Se for necessário reverter, remover somente esse drop-in, executar `systemctl daemon-reload` e reiniciar `llama-server-root.service` devolve o serviço ao release VMM anterior, desde que os drop-ins subjacentes não tenham sido alterados depois deste registro.

## Ambiente e método

- GPU: NVIDIA GeForce RTX 4070; CPU: AMD Ryzen Z1 Extreme. CUDA toolkit em `/opt/cuda`; o arquivo do release oficial é identificado como CUDA 13.3. A revisão de código para a nova investigação é `4bad0205c`.
- Modelo: `/home/hjotha/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf`, SHA-256 `e23eb251493b8a89f7d0217590b593e696ac9bacaeac8605541e477533dc5838`.
- Oficial: `/home/hjotha/.codex/artifacts/beellama-official-ab-20261105/official/beellama-v0.4.9`; SHA-256 do arquivo oficial `c3c047b2798c7a4b108c964d44d0bec917a67ea21bcdcb2ced1defead20c1dbf`; `libggml-cuda.so` `8d49d424d4d30372cdfc3576b8e4c9dc4b0cbb6b522c0b0364f968cf74c97ff3`.
- Fork de produção com VMM usado como controle antes da promoção: `/home/hjotha/beellama-releases/e4-prefetch-vmm-4bad0205c-20261105/bin`; CUDA `51207f71e7b57736c0f02165349a336f8553bc37e84ece9a74635e45ce1955dc`. O controle CUDA sem VMM tem hash `c784e809dfb467614bf2652e63e58f81ccbf386a9d67d9c9e74159d4660fc00a`. A biblioteca VMM foi criada recompilando e religando o objeto CUDA pertinente; não foi um rebuild CUDA integral.
- Prompt: primeiros 19.456 tokens de `/home/hjotha/.codex/artifacts/prod-draft-kvarn4-tiers-20261105/prompt-57344-tokens.json`, arquivo de tokens SHA-256 `610b9f418d0fdd40e41dbede4a7ebfc58406635144b9f2efe903170dc90ed2e8`; 1.024 tokens de saída, seed 1234, temperatura 0, EOS ignorado e cache de prompt desativado.
- Mesmo perfil estático equivalente ao S: contexto 20.480, batch 2.048, ubatch 512, 8 threads de geração e prefill, 99 camadas GPU, flash attention, `fit=off`, uma sequência; KV target e draft `kvarn4`, MTP `n_max=6`, `p_min=0.65` e ubatch de draft 128. A rota remota do Xbox não é acionada nesses A/B locais.
- As medições recentes do fork usam `GGML_GDN_RAW_GATES_DISABLE=1`, `GGML_CUDA_GDN_PREFIX_MMA=2`, `GGML_CUDA_GRAPH_MIN_STABLE_CALLS=256`, `GGML_CUDA_GRAPH_RECOVERY_HEADROOM_MB=18`, salvo quando uma variante é indicada. O relatório individual e seu `run.py` são a referência para o ambiente exato do oficial e das execuções anteriores; não se deve assumir que variáveis aceitas pelo fork tenham o mesmo efeito no binário oficial.
- **Distinção de produção:** apesar de `systemctl show` listar `GGML_CUDA_GRAPH_MIN_STABLE_CALLS=256` em `Environment`, ela também aparece em `UnsetEnvironment` e está ausente no `/proc` do backend em execução; aplica-se o padrão do código, **4**. Os A/B com valor explícito 256 não representam esse detalhe do ambiente de produção. A comparação com o valor efetivo está registrada adiante.
- Os testes de servidor são processos isolados em GPU exclusiva, sem RAM cache e sem troca adaptativa de tier. As taxas são pontuais: a variação de clocks, aquecimento e sequência de tokens impede tratá-las como médias estatísticas. Scripts encerram o teste e registram logs, bibliotecas e VRAM; alguns supervisores param e restauram produção, portanto devem ser lidos antes de executar.

## A/B do oficial e fork

| Caso | Prefill tokens/s | Decode tokens/s | Propostas / aceitas | Observação |
|---|---:|---:|---:|---|
| Oficial, primeira comparação recente | 901,37 | 27,485 | 741 / 349 | `decode-root-cause-20261105` |
| Fork VMM, controle A / B | 912,96 / 885,98 | 26,465 / 26,418 | 737 / 334 | mesma saída entre as duas execuções |
| Oficial, confirmação A / B | 891,28 / 886,03 | 27,406 / 27,419 | 741 / 349 | `decode-official-recheck-20261105` |
| Fork VMM, entre as confirmações | 881,74 | 26,302 | 737 / 334 | ordem oficial → fork → oficial |
| Fork VMM, graphs desativados | 891,85 | 26,714 | 737 / 334 | oficial no mesmo estudo: 27,663 |

O SHA-256 dos IDs de saída do oficial é `9b3be7ab07a58d426775a56637ab7baf5166f3e9a85f4b12627e95d2d15262eb`; o do fork é `d5c7c5bcd58cd07de0bd38b1ad44f29b6348fe2deb86edb87cd64b1ee2219931`. Na captura Nsight os primeiros 39 IDs coincidem. A diferença não pode ser atribuída somente a throughput dos kernels ou somente à aceitação.

Nos ensaios anteriores ao VMM, o oficial mediu 27,60 tokens/s e o fork 25,69 tokens/s em `beellama-official-confirm-20261105` e `beellama-parity-final-20261105`; o segundo usou estabilidade de graph 1, o primeiro 256. Na comparação inicial 20k, oficial/fork foram 27,57/25,53 com `kvarn4`, mas o fork ainda não incorporava os mesmos ajustes. Esses números históricos não substituem os A/B acima.

Os artefatos principais são `/home/hjotha/.codex/artifacts/decode-root-cause-20261105/results.json`, `/home/hjotha/.codex/artifacts/decode-official-recheck-20261105/results.json` e `/home/hjotha/.codex/artifacts/decode-copilot-tests-20261105/results.json`.

## Trace Nsight e investigações externas

O trace em `/home/hjotha/.codex/artifacts/decode-nsys-20261105/run-2` foi capturado com CUDA, NVTX e nós de graphs, usando uma janela que começa na chegada do primeiro token SSE. As faixas foram validadas contra o relógio monotônico. No intervalo, o oficial registrou 1.632.944 kernels e o fork 1.666.015; 629.024 (38,5%) e 18.900 (1,13%), respectivamente, eram nós de graphs. Chamadas `cudaGraphLaunch`: 4.607 contra 378; `cudaGraphInstantiate`: 1.033 contra 1. Chamadas de runtime durante captura e kernels executados não são contagens equivalentes.

A união dos intervalos de kernels ocupados na GPU foi 29,906 s no oficial e 31,060 s no fork; as janelas totais foram 38,544 e 39,786 s. Os totais por família foram aproximadamente: matvec 21,773/22,378 s, KVarN 4,067/4,339 s, flash attention 0,326/0,603 s e GDN 0,531/0,576 s. Esses totais podem se sobrepor e não devem ser somados como causas independentes. A instrumentação altera os tempos e a janela SSE pode excluir trabalho já enfileirado após o primeiro token.

A especialização de decode KVarN comparável foi D=256, GQA=6, split=64, 8 warps, K/V=4 bits, Q_TILE=1; 32×8 threads, 11.616 bytes de shared estática e 62 registradores por thread. Para gridX=312, chamadas diretas com gridY=4 mediram 155,15 µs oficial e 161,79 µs fork; Y=8, 264,63/279,97 µs; Y=12, 379,48/403,57 µs. Comprimento exato, trajetória e o próprio profiler continuam sendo confundidores.

O Copilot levantou captura de graphs, consultas de VRAM, GDN e diferenças numéricas. O AGY examinou trace/SASS e propôs compactar a cauda e paralelizar partes da transformação de Q; suas estimativas de tempo e alegações de equivalência são **hipóteses**, não prova de causa nem de KLD zero. O relatório do Copilot está em `/home/hjotha/.codex/artifacts/decode-root-cause-20261105/copilot-response.txt`; o do AGY e seu patch em `agy-response.txt` e `agy-parallel-q-experiment.patch` no mesmo diretório. Os testes abaixo contradizem explicações simples baseadas apenas em graphs ou em um dos ajustes individuais.

## Experimentos e resultados

Todos os tempos seguintes são tokens/s de decode, salvo indicação. Valores A/B na mesma linha são repetições de uma sequência de execuções, não intervalos de confiança.

| Tentativa | Controle | Variante | Resultado e situação |
|---|---:|---:|---|
| Graphs desligados | oficial 27,663; fork 26,714 | ambos off | Gap permanece; graphs não são a explicação única. `decode-copilot-tests-20261105` |
| Estabilidade 256 → 1 | fork 26,408 | 25,750 | Mesmos tokens; piora no servidor. `decode-graph-decisions-20261105` |
| Estabilidade 1 / 4, chunk 0 | fork 25,711 / 26,403 | — | Mesmo hash; não elimina o gap. `decode-copilot-tests-20261105` |
| Target-only, sem MTP, 19.456 profundidade + 1.024 | oficial 33,352; fork graph 256 31,723 | fork graph 1 33,117 | Uma amostra por caso; não prova efeito no servidor com MTP. `beellama-decode-fixed-20261105`, `beellama-decode-graph1-20261105` |
| Graphs off / single / chunk32, prompt 1.024 + saída 128 | Q4: 31,53 / 30,90 / 30,70 | KVarN: 30,21 / 29,05 / 28,95 | Outra carga; preservou hashes em cada grupo e não antecipa o resultado 20k. `beellama-upstream-graphs-20261105`, `beellama-gdn-prefix-graphs-20261105` |
| GDN prefix MMA 2 → 0 | 26,465 / 26,418 | 25,682 | Prefill 912,96 → 831,95; trajetória e aceitação mudaram. `decode-root-cause-20261105` |
| GDN segmented=1 | prefix 2: 25,832 | prefix 0: 26,662 | Saídas e aceitação diferentes; não é prova de melhoria preservando comportamento. `decode-gdn-segmented-20261105` |
| Evitar carga Q redundante | controle 26,346 / 26,458 | 26,338 | Uma variante; sem ganho mensurável. `decode-kvarn-variants-20261105` |
| Reordenar condição dos diagnósticos da cauda | controle 26,346 / 26,458 | 26,447 | Uma amostra; combinação com Q: 26,452; sem ganho confiável. Mesmo diretório |
| Q F32 em shared | controle 26,405 / 26,356 | 26,354 / 26,341 | Mesmo hash; sem ganho, com maior shared. `decode-q-shared-ab-20261105` |
| Detectar máscara all-zero antecipadamente | controle 26,430 / 26,367 | 26,359 / 26,378 | Mesmo hash; sem ganho. `decode-tail-mask-ab-20261105` |
| Remover diagnóstico do kernel normal por especialização | controle 26,443 / 26,390 | 26,413 / 26,411 | Mesmo hash; sem ganho. `decode-tail-nodiag-ab-20261105` |
| Fast path denso da cauda | 26,421 | sem resultado | Execução interrompida por sinal 15; não é resultado de performance. `decode-tail-dense-ab-20261105` |
| Compactar índices de pesos nãozero | controle 26,350 / 26,340 | 26,485 / 26,484 | Mesmo hash; ganho pequeno, não apresentado como solução completa. `decode-tail-compact-ab-20261105` |
| Compactar slots e pesos em shared | controle 26,416 / 26,346 | 26,593 / 26,558 | Mesmo hash; ganho pequeno e repetido. `decode-tail-slots-ab-20261105` |
| Cauda compactada + Q paralelo | controle 26,522 / 26,370 | 26,703 / 26,638 | Médias: 26,446 → 26,670 (+0,85%); mesmos tokens e 737 / 334 propostas/aceitas. `decode-final-ab-20261105` |
| Cauda + Q + validador metadata-only | controle 26,440 / 26,339 | 26,597 / 26,535 | Médias: 26,390 → 26,566 (+0,67%); mesmo hash de saída. `decode-integrated-ab-20261105` |
| Q com scratch em blocos de 64 | controle 26,724 / 26,651 | 26,759 / 26,764 | Mesmo resultado F32 nos 31 casos focados; cerca de +0,28% médio, descartado em favor da versão mais simples. `decode-q-scratch-ab-20261105` |

Os caminhos `decode-*` da tabela são subdiretórios de `/home/hjotha/.codex/artifacts/`. Cada `results.json` registra os parâmetros, hashes, logs por execução, timings, saída e amostras de VRAM. Os patches correspondentes estão junto aos relatórios; as bibliotecas candidatas privadas e manifestos ficam nos diretórios homônimos sem o sufixo `-ab`.

### Qualificação funcional do candidato combinado

A confirmação A/B usou a biblioteca candidata SHA-256 `165a6d1043000c0a730c9d866e1d58ab163ccd00e8d6209d229eb63f857e199f`, contra o controle VMM `51207f71e7b57736c0f02165349a336f8553bc37e84ece9a74635e45ce1955dc`; os quatro requests tiveram o mesmo hash de tokens e mínimo de 409 MiB livres amostrados. Os dois valores da tabela são amostras individuais e a média não estabelece significância estatística. O candidato não eliminou a diferença para o oficial.

Em `/home/hjotha/.codex/artifacts/decode-correctness-final-20261105/tail-graph.log`, passaram 12 casos de O/LSE com máscaras selecionadas, F16/BF16 e sem/com segmentos atuais, além de formas Qwen D256. Em `/home/hjotha/.codex/artifacts/decode-correctness-differential-20261105/results.json`, controle e candidato passaram os testes de parâmetros FA, LSE e decode com offset ímpar; o `racecheck` do candidato terminou com **0 erros e 0 avisos**. Esses resultados qualificam os casos executados, não todas as topologias possíveis.

O teste amplo `GGML_KVARN_TEST_PORTABLE_NATIVE_ONLY=1` abortou com código -6 tanto no controle quanto no primeiro candidato. A análise identificou o primeiro caso D256, Q=1, 6/1 cabeças e K4/V4: a guarda CUDA interpretava os metadados de corpo em `src[8]` como cauda incompleta. A incompatibilidade já existia em `4bad0205c`. A correção integrada no commit `878e29e2c` restringe a exceção a metadados KVarN F32 contíguos com shape válido e continua rejeitando pares current incompletos. Com ela, o teste amplo, o D256 focado e os 12 casos de cauda passaram em `/home/hjotha/.codex/artifacts/decode-metadata-validation-20261105/`; o teste CPU da guarda também passou.

O teste focado de D256/K4/V4 executou 31 casos no controle e no candidato anterior ao ajuste da guarda, com hashes F32 idênticos; a telemetria confirmou split 64, oito warps e GQA 6. O racecheck desses casos e o dos 12 casos F16/BF16 de cauda terminaram com zero erros e avisos. Esses resultados estão em `/home/hjotha/.codex/artifacts/decode-kvarn-focused-20261105/` e `/home/hjotha/.codex/artifacts/decode-correctness-differential-20261105/`. A versão anterior ao ajuste da guarda também passou S→M→L→XL→XXL→XXXL e retorno, e um request de 109.490 tokens seguido de retorno ao S; os logs estão em `/home/hjotha/.codex/artifacts/decode-tier-qualification-20261105/`. A biblioteca com a guarda corrigida não repetiu esse teste longo completo.

## Precisão, Q e rollback do draft

O merge `29bb535f1`, de 2026-10-13, alterou a leitura de Q na atenção KVarN: ambos os pais liam a Q já arredondada para F16 em shared; a resolução passou a usar a Q original F32 e arredondar depois da aplicação da escala do eixo. `git diff v0.4.9..4bad0205c -- ggml/src/ggml-cuda/fattn-mma-kvarn-decode.cuh` mostra essa diferença e mudanças separadas de LSE. Uma volta ao caminho F16 pode alterar qualidade e não deve ser promovida por throughput isolado.

Em contexto 20.480, batch 128, ubatch 32, KV KVarN4 e sem MTP, as médias de KLD foram **0,001343** do oficial contra BF16, **0,001255** do fork contra BF16 e **0,001178** do fork contra oficial. Dados em `/home/hjotha/.codex/artifacts/beellama-official-fork-kld-20261105/results.json`. Esses testes não isolam a mudança de Q nem medem aceitação MTP. Um teste separado de GDN com rollback recorrente 6 mediu KLD 0,001194 versus BF16 sem prefix MMA e 0,001250 com prefix MMA 2; referência e controles estão em `/home/hjotha/.codex/artifacts/beellama-gdn-prefix-mma-20261105/kld-rs6-bf16.json` e `kld-rs6-results.json`.

Os logs mostram `n_rs_seq=6` no target de ambos, mas 0 no draft oficial e 6 no fork. Neste Qwen35 o draft MTP possui apenas cache de atenção, não camada recorrente; a razão geral da correção de rollback recorrente não prova uma falha específica nesse draft. Um shim isolado testou controle, draft `n_rs_seq=0` com reserva de cauda preservada, e draft zero sem essa preservação: 26,341 / 26,371 / 26,405 tokens/s, respectivamente; 1.036 chamadas monitoradas de remoção do draft e zero falhas em cada execução. O hash de saída foi o mesmo; a última variante gerou 738 propostas em vez de 737, com 334 aceitas. Não explica o gap ou autoriza desligar proteções em produção. Artefatos: `/home/hjotha/.codex/artifacts/decode-mtp-draft-rs-20261105/`.

## VMM e transições

O VMM foi promovido depois dos testes descritos em `/home/hjotha/.codex/artifacts/vmm-tier-transitions-20261105/`. A sequência S→M→L→XL→XXL→XXXL e retorno passou com prompt curto; S→M→S também passou após prefill de 20.481 tokens. Um request de 109.490 tokens concluiu no XXXL e retornou ao S; prefill 524,76 tokens/s e 232,01 s de parede incluindo transição. O log mostra inicialização da rota remota, mas não contadores individuais de RPC. Snapshots automáticos foram desativados no teste isolado.

O mínimo amostrado no request longo foi 49 MiB livres. No teste curto XXXL houve amostras de 33 MiB com VMM e 11 MiB sem VMM; ambas são margens pequenas e não provam ganho estável. Em um A/B estático de 19.456 + 1.024, sem/com VMM: prefill 911,10/905,85 e decode 26,57/26,55 tokens/s, com IDs idênticos; VRAM livre mínima 119/409 MiB. `/home/hjotha/.codex/artifacts/vmm-single-object-20261105/bench-no-vmm/results.json` e `bench-vmm/results.json` têm as medições. O `build-optimized` foi reconfigurado com `GGML_CUDA_NO_VMM=OFF`; isso configura builds futuras e não recompila por si só os binários existentes.

## Ablação Q2_K MMVQ — 2026-11-05

### Nova pista: divisão do prefill e margem após penalidades

O usuário identificou e a leitura dos logs confirmou que os servidores não dividem os últimos 1.024 tokens do prompt da mesma forma. Ambos chegam a 18.432 tokens em batches lógicos de 2.048; o oficial conclui com **384 + 512 + 128**, enquanto o fork usa **512 + 384 + 128**. Os checkpoints são criados nas posições 18.816 e 19.328 no oficial, e 18.944 e 19.328 no fork. As regras de checkpoint diferem no cálculo da primeira fronteira. Como o modelo tem estado recorrente, essa diferença pode alterar o resultado numérico, mas ainda não demonstra causalidade.

O replay target-only anterior não reproduziu essas duas cadências e seus logits no fim do prompt não coincidiram com os registrados pelo servidor; portanto, ele **não** deve ser tratado como reprodução fiel do prefill do servidor. Também não é uma comparação do fork contra o oficial: ele compara duas rotas Q2_K no fork. No token 40, os IDs crus de maior logit (`264` e `449`) não são os dois tokens efetivamente escolhidos após penalidades (`1575` no fork e `3750` no oficial). Assim, as margens cruas desses top-2 não são a margem da decisão divergente.

Para isolar a ordem, foi executado o **mesmo binário e biblioteca CUDA do fork** em quatro processos novos, na sequência fork → oficial → oficial → fork, mudando apenas os três cortes finais do prefill. Cada braço reprocessou os mesmos 19.456 IDs; o replay usou target-only, `n_rs_seq=6`, cauda 128 e nextn, sem draft MTP nem serialização de checkpoints. A ordem do fork reproduziu exatamente os dois maiores logits brutos observados no primeiro token do servidor fork: IDs `539`/`554`, valores `25,2563686`/`20,9481163`. A ordem do oficial produziu os mesmos IDs, mas valores `25,2384262`/`20,8965416`. As duas repetições de cada ordem foram bit a bit idênticas entre si para os **248.320 logits finais e 5.120 valores nextn**; entre ordens, todos esses valores diferiram, com diferença absoluta máxima de logits `0,216114` e KL entre as duas distribuições de aproximadamente `7,55e-6` nats. A comparação com o servidor é somente dos dois maiores logits do fork, não do vetor completo nem dos logits do servidor oficial.

Este A/B confirma que a ordem dos cortes altera os estados e logits finais no cenário target-only testado. Ainda **não prova** que essa alteração cause a decisão diferente no token 40, pois não reproduz draft MTP, serialização de checkpoints ou todo o histórico do servidor. Artefato: `/home/hjotha/.codex/artifacts/decode-prefill-schedule-20261105/run-1/manifest.json` e seus vetores F32.

Em captura diagnóstica separada dos servidores reais, os primeiros 96 IDs de cada release coincidiram com os respectivos testes sem o interpositor; oficial e fork compartilharam os primeiros 39. No 40º token, o interpositor observou os seguintes logits dos dois candidatos, imediatamente antes e depois do sampler de penalidades:

| Release | ID 1575 antes / depois | ID 3750 antes / depois | Margem após penalidades (1575 − 3750) | Escolhido |
|---|---:|---:|---:|---:|
| Oficial | 17,642824 / 17,642824 | 17,656221 / 17,656221 | −0,013397 | 3750 |
| Fork | 17,616619 / 17,616619 | 17,616390 / 17,616390 | +0,000229 | 1575 |

Esses dois candidatos não foram alterados diretamente pela etapa de penalidades; os IDs `264` e `449`, maiores antes dela, foram penalizados, e os dois candidatos passaram a ocupar as primeiras posições. Os IDs emitidos tinham histórico igual até então, mas os estados internos podem diferir. O trace adiciona sincronização e não serve para TPS. O diagnóstico identifica uma decisão sensível a uma margem pequena, mas ainda não atribui causalmente a diferença à ordem de prefill. Dados em `/home/hjotha/.codex/artifacts/decode-penalty-comparison-20261105/comparison.json` e nos logs correspondentes.

Em seguida, um binário privado do servidor permitiu alternar **somente** o limite usado para o primeiro checkpoint; a mesma biblioteca CUDA VMM e os demais parâmetros foram mantidos. A execução de controle reproduziu os checkpoints em 18.944/19.328 e o hash de tokens da produção. A alternativa reproduziu os checkpoints em 18.816/19.328. Na ordem controle → alternativa → controle → alternativa:

| Cadência | Decode tokens/s | Prefill tokens/s | Tempo total | Drafts / aceitos | Prefixo igual ao oficial |
|---|---:|---:|---:|---:|---:|
| Fork, 512 + 384 + 128 | 26,416 / 26,470 | 914,07 / 896,51 | 60,025 / 60,361 s | 737 / 334 | 39 tokens |
| Alternativa, 384 + 512 + 128 | 27,501 / 27,375 | 899,92 / 890,02 | 58,833 / 59,246 s | 662 / 350 | 139 tokens |

As repetições de cada braço geraram IDs idênticos entre si. A média de decode passou de **26,443 para 27,438 tokens/s**, ganho de aproximadamente **3,76%** nessa carga; o token 40 passou de `1575` para `3750`, igual ao oficial, cuja média na repetição intercalada foi aproximadamente `27,412 tokens/s`. Isso confirma que a cadência de checkpoint é **uma causa concreta para esta divergência nessa carga**, embora as saídas ainda divirjam após 139 tokens e não estabeleça equivalência geral entre versões. O binário de ablação não foi usado em produção. Dados: `/home/hjotha/.codex/artifacts/decode-checkpoint-server-ab-20261105/results.json` e `*-tokens.json`; construção e patch privado em `/home/hjotha/.codex/artifacts/decode-checkpoint-cadence-20261105/`.

O ajuste de código usa o batch lógico para esse corte quando o ubatch físico coincide com o solicitado; quando o físico é reduzido, mantém o limite físico que preserva a cadência do wavefront. O teste unitário `test-server-prompt-checkpoint` passou com as fronteiras de 19.456 tokens e os casos de alinhamento 1, batch menor que ubatch e wavefront 512/256. O controle do relink do servidor reproduziu o SHA da biblioteca anterior à alteração.

Também foi medida a configuração **efetiva de graphs da produção**, com a variável de limite ausente e padrão 4, alternando produção, checkpoint e checkpoint mais CUDA duas vezes. As médias de decode foram **26,437**, **27,388** e **27,741 tokens/s**, respectivamente. A melhoria CUDA adicional foi de aproximadamente **1,29%** sobre checkpoint sozinho e preservou os 1.024 IDs e `662/350` propostas/aceitas; a combinação ficou aproximadamente **4,93%** acima da produção anterior nesse teste. A média de prefill foi, respectivamente, `893,11`, `884,25` e `884,23 tokens/s`; o tempo de parede médio foi `60,50`, `59,37` e `58,89 s`. As taxas variam com o prompt e não são médias estatísticas amplas. Dados e hashes das bibliotecas: `/home/hjotha/.codex/artifacts/decode-promotion-prod-graphs-20261105/results.json`.

O bundle combinado exato passou na sequência S→M→L→XL→XXL→XXXL e volta a S e num request de **109.490 tokens** em XXXL seguido de retorno a S; o prefill longo foi `521,82 tokens/s`, com `233,09 s` de parede. A menor VRAM livre amostrada foi `49 MiB` no request longo; os snapshots automáticos foram desativados no teste isolado e não foi registrada uma contagem individual de RPCs. Resultados em `/home/hjotha/.codex/artifacts/decode-promotion-tiers-20261105/`.

Esta investigação adicional foi solicitada para separar a troca de rota Q2_K da diferença entre fork e oficial. **Não foi feita promoção nem integração automática nesta etapa.** O mesmo binário fork de produção e a mesma biblioteca CUDA VMM foram usados em processos separados, com os mesmos argumentos, modelo, prompt e sampling descritos acima. A única variável deliberada foi `GGML_CUDA_Q2_K_MMVQ=1` ou `=0`; a ordem foi 1 → 0 → 1 → 0. Os hashes do executável e da CUDA coincidem nas quatro execuções. A cauda efetiva do target e do draft era de 128 tokens F16 em ambos os modos; `n_rs_seq=6` nos dois contextos. O oficial também reportou cauda efetiva 128, embora seu draft use `n_rs_seq=0`.

| Flag | Decode tokens/s | Média decode | Prefill médio | Tempo total médio | Drafts / aceitos |
|---|---:|---:|---:|---:|---:|
| `1` — MMVQ para Q2_K, batches 5–8 | 26,284 / 26,392 | 26,338 | 883,72 | 60,87 s | 737 / 334 |
| `0` — limite legado MMVQ 4 | 25,765 / 25,693 | 25,729 | 878,35 | 61,93 s | 582 / 279 |

Nesta carga, `=0` foi cerca de 2,3% mais lento no decode médio e gerou menos propostas e aceitações absolutas. A fração aceita entre propostas foi aproximadamente 45,3% com `=1` e 47,9% com `=0`; uma fração maior não significou maior throughput. Cada modo foi determinístico nas duas repetições dos 1.024 IDs gerados. Os dois modos coincidiram nos primeiros 60 IDs e divergiram no 61º; **ambos já divergiam do oficial no 40º ID**. Assim, a opção Q2_K altera a trajetória nesta carga, mas não explica a primeira divergência observada entre o fork e o oficial.

Uma rodada diagnóstica separada de 96 tokens manteve `n_probs=0` e o mesmo sampler. Um interpositor somente de leitura registrou os dois maiores logits crus F32 do target; `GGML_DFLASH_SHADOW_TRACE=1` registrou propostas e confirmações do MTP. Os 96 IDs de cada modo coincidiram com os primeiros 96 da respectiva execução sem instrumentação. A instrumentação adiciona sincronização e leitura de logits, portanto seus tempos **não** servem para comparar TPS. No primeiro ciclo de draft diferente, posição `19468` e anchor `9111`, `=1` propôs `[18131,494,220,16,24]`, enquanto `=0` propôs `[18131]`; ambos confirmaram `[18131,888]`. As margens dos logits crus diferem mesmo sobre o prefixo emitido comum; os dois maiores IDs crus permaneceram iguais até a primeira divergência de saída entre os modos. No 40º token, por exemplo, a margem crua foi 3,0255 em `=1` e 2,9811 em `=0`, e ambos emitiram `1575`; o oficial emitiu `3750`. Os top-2 registrados são **anteriores às penalidades de sampling**, de modo que o argmax bruto não deve ser confundido com o token final escolhido. Também não se trata de um replay com batches especulativos idênticos: as propostas já podem variar dentro do prefixo comum.

Um trace Nsight separado, de 96 tokens, confirmou o roteamento efetivo: com `=0` foram observadas chamadas Q2_K a `mul_mat_q`; com `=1`, chamadas `mul_mat_vec_q` para batches 5, 6 e 7. A janela começa após o primeiro token e o profiler modifica os tempos. Os dados e os IDs completos estão em `/home/hjotha/.codex/artifacts/decode-q2k-ablation-20261105/results.json`; logits e eventos em `/home/hjotha/.codex/artifacts/decode-q2k-logits-20261105/comparison.json`; traces em `/home/hjotha/.codex/artifacts/decode-q2k-nsys-20261105/run-1/`.

**Estado da hipótese:** confirmado que a política Q2_K pode mudar propostas, logits e trajetória; descartado que, neste prompt, ela explique por si só a primeira divergência do fork com o oficial. O flag permanece inalterado em produção.

### Replay target-only com continuação fixa

Foi executado o replay solicitado, sem MTP e sem sampler: cada processo reprocessou os mesmos 19.456 IDs de prompt e depois consumiu os mesmos 1.024 IDs de continuação, em 169 batches de seis tokens e dois de cinco. Os parâmetros efetivos registrados incluem contexto 20.480, KVarN4, cauda 128, `n_rs_seq=6`, reserva de rollback 6, batch 2.048, ubatch 512 e captura de embeddings nextn. Isso mantém prefixo, tokens e tamanho de batch iguais **entre os dois modos do replay**, mas os estados internos evoluem independentemente após cada batch. O teste não reproduz o estado do draft nem as rejeições MTP; não é um benchmark ponta a ponta do servidor.

| Flag | Tempo dos 171 `llama_decode` + sincronização | Linhas fixas | Top-1 bruto distinto |
|---|---:|---:|---:|
| `0` | 10,533 s | 1.024 | — |
| `1` | 10,632 s | 1.024 | 13 posições em relação a `0` |

Cada braço válido foi executado uma vez, na ordem 0 → 1; a diferença de aproximadamente 0,94% no tempo não demonstra, sozinha, uma vantagem estável. A extração de logits foi medida separadamente e não está incluída nesses tempos. A primeira diferença de top-1 ocorreu após o **token de entrada na posição absoluta 19721**: `0` colocou o ID `303` acima de `430`, com margem 0,03963, enquanto `1` colocou `430` acima de `303`, com margem 0,01904. As margens são logits crus, sem penalidades; logits de uma linha predizem a posição seguinte. As 13 posições não são amostras independentes nem significam 13 erros. Como controle parcial, os 1.024 resumos de logits do modo `0` coincidiram exatamente entre o primeiro ensaio e o teste com log corrigido; não houve A/A do modo `1`. O modelo draft e seus pesos não estavam residentes no replay, portanto o tempo não é equivalente ao serving MTP. Combinado ao A/B MTP, o resultado é consistente com uma contribuição da trajetória e do volume de trabalho, ainda não isolada causalmente; não identifica a origem da divergência fork–oficial no token 40 nem prova qual rota é mais precisa.

O primeiro ensaio do replay em `run-w6` foi considerado inválido porque o nível de log ocultou a confirmação da política Q2_K. O teste corrigido com `-lv 4`, hashes das bibliotecas mapeadas, configuração efetiva e resultados está em `/home/hjotha/.codex/artifacts/decode-q2k-ablation-20261105/run-w6-log4/manifest.json`; o controlador é `replay-run.py` no diretório pai. Nenhuma comparação de precisão contra BF16 foi feita nessa cadência de replay.

### Toolchain e arredondamento

O usuário levantou a hipótese de diferenças de arredondamento na compilação. O fork foi configurado com NVCC CUDA 13.4.92 e GCC 16.2.1; o pacote oficial é da build CUDA 13.3, com GCC 13.3.0. Os arquivos CMake das duas árvores definem `-use_fast_math`, e a inspeção SASS do kernel da cauda encontrou instruções FMA e operações com flush-to-zero em ambos. A comparação do kernel Q2_K MMVQ de batch 1 encontrou código de máquina sm_89 bit a bit idêntico; isso não se estende automaticamente a outros kernels ou tamanhos de batch. Essas evidências mostram diferenças de toolchain, mas não provam que o compilador cause o gap.

Um experimento isolado mudou somente a query do kernel KVarN4 D256/GQA6/split64 para o arredondamento prévio em F16 usado no código oficial, mantendo o restante da biblioteca de produção. Com o mesmo prompt de 19.456 + 1.024, o controle F32 mediu 26,356 / 26,349 tokens/s, 737 propostas e 334 aceites; a variante F16 mediu 25,100 / 25,141 tokens/s, 488 propostas e 238 aceites. A variante F16 continuou coincidindo com o oficial somente nos primeiros 39 tokens. Ela foi descartada como solução de throughput nessa carga; **não foi medida sua KLD isolada**, e não houve mudança de precisão em produção. Evidências: `/home/hjotha/.codex/artifacts/decode-qf16-ab-20261105/results.json` e `*-tokens.json`.

Também foram geradas bibliotecas privadas a partir dos mesmos fontes `4bad0205c`, mantendo o restante dos objetos e VMM de produção. O controle religado reproduziu exatamente o SHA-256 da CUDA em produção. Na variante `nofast`, apenas os TUs KVarN4 decode, `fattn.cu` e `mmvq.cu` foram compilados sem `-use_fast_math`; isso muda FTZ e precisão de divisão/raiz e pode mudar intrínsecos, enquanto FMA continua no padrão do compilador. Na variante `fmad0`, somente `mmvq.cu` manteve fast math com `--fmad=false`. Uma execução de cada, intercalada com dois controles, resultou em:

| Variante | Decode tokens/s | Prefill tokens/s | Drafts / aceitos | Parede |
|---|---:|---:|---:|---:|
| Controle fast math | 26,536 / 26,385 | 911,69 / 889,85 | 737 / 334 | 59,90 / 60,65 s |
| Três TUs sem fast math | 25,785 | 901,33 | 523 / 278 | 61,27 s |
| `mmvq.cu` com `--fmad=false` | 25,096 | 890,48 | 623 / 268 | 62,63 s |

As duas variantes modificadas divergiram dos controles **e do oficial já no 15º token**, contra a divergência original no 40º. Os resultados confirmam que essas opções de compilação podem afetar a trajetória e o TPS, mas não mostram que o oficial use essas opções nem que elas expliquem a divergência original. Cada variante foi medida uma vez, não houve KLD isolada e nenhuma foi promovida. Os TUs individuais `q`, `tail` e `mmvq` também foram compilados para eventual isolamento, mas não foram executados nessa rodada. Evidências e hashes: `/home/hjotha/.codex/artifacts/decode-nofastmath-experiment-20261105/manifest.json` e `/home/hjotha/.codex/artifacts/decode-nofastmath-ab-20261105/results.json`.

## Como examinar e reproduzir

1. Para reproduzir a análise dos traces já salvos sem usar GPU:

   ```bash
   python3 /home/hjotha/.codex/artifacts/decode-nsys-20261105/analyze-decode.py /home/hjotha/.codex/artifacts/decode-nsys-20261105/run-2
   ```

2. Os scripts `run.py` e `results.json` dos diretórios de A/B acima registram argv, biblioteca CUDA, hashes, sequência e métricas. Antes de refazer uma rodada, use diretório de saída novo e confira o script: ele exige GPU livre e serviço parado; alguns supervisores controlam o serviço. Não compare taxas de uma captura Nsight com testes não instrumentados.
3. A captura Nsight pode ser repetida, com produção previamente parada e GPU livre, usando um diretório de saída ainda inexistente:

   ```bash
   python3 /home/hjotha/.codex/artifacts/decode-nsys-20261105/profile-decode.py --run-dir /home/hjotha/.codex/artifacts/decode-nsys-20261105/repro-20261105
   ```

   O modo `--smoke` usa 256 tokens de prompt e 8 de saída; confira os arquivos `.validation.json` e a presença de eventos CUDA antes de confiar na captura longa. `run-1` não é válido para CUDA: o ambiente do subprocesso removia as variáveis de injeção do Nsight; o harness foi corrigido e `smoke-1` e `run-2` são as capturas válidas.
4. Para qualificar o patch da cauda, repetir os A/B no candidato exato e validar O/LSE, máscaras sem peso, intercaladas e totalmente mascaradas, `vt_current` e histórico, F16/BF16, limites de 256 e valores V não finitos em posições mascaradas. A saída de 1.024 tokens idêntica é útil, mas não substitui esses testes de kernel.

Os artefatos em `/home/hjotha/.codex/artifacts/` são locais e podem ser temporários; este documento registra os números, mas a auditoria detalhada depende da preservação desses arquivos.
