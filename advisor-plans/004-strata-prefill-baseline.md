# Flash-Next Q2_0 na A100

Escopo: prefill de 4k e 28k com Qwen3.8-Flash-Next Q2_0, A100 de 40 ou 80 GB, uma
sequência e contexto de 32.768 tokens. A última receita integrada fez 1.946,6
t/s em 28k. Um ganho de 3% precisa economizar cerca de 0,43 s nesse prompt.
Começar pelo perfil do próprio Q2_0. A atribuição publicada de IQ3_S mede
4,665 s de grafo CUDA em 43,8 s de parede, com outro placement e outra quantização.

## Ordem de execução

| Ordem | Trabalho | Esforço | Evidência para avançar |
|---|---|---|---|
| 1 | Baseline pareado e perfil por operação no Q2_0 | S | Medianas e faixas no mesmo runtime. Medir eventos CUDA em rodada separada. |
| 2 | QSA passa IDs visíveis diretamente à FA CUDA | M | Máscara e scan custam tempo ou workspace suficiente para justificar o novo caminho. |
| 3 | Ajustar MMQ Q2_0 à largura efetiva por expert | M para perfil, L para kernel | Histograma, ocupação e tiles desperdiçados da rota real indicam gargalo. |
| 4 | Fusão completa gate/up/GLU/down | L | A prova dos blocos Q8_1 intermediários preserva bytes e ordem de redução. |
| 5 | GDN ou temporários de prefill | S para medir, M/L para mudar | Perfil aponta o scan ou a reserva de memória como limite relevante. |

## 1. Parear antes de atribuir

Medir Strata de upstream e BeeLlama na mesma A100 com os dois shards originais
de `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, revisão
`2a55d75962e22f7a4a1d9963ab6eae3678537831`, variante `Q2_0`.
Registrar tag e commit do código, SHA-256 do executável e do prompt, GPU,
driver, RAM, configuração, tokens efetivos e pico de VRAM. Usar os casos
`short`, `pp4k+dec` e `pp28k+dec` de `bench.py` do comparativo de 22/11,
greedy, dois warmups e três repetições. Exigir zero tokens reaproveitados.
Na Strata, `--prompt-cache 0` desabilita o reuso e a API informa
`cache_n`. Usar throughput de decode reportado pelo engine e guardar
tokens gerados e drafts aceitos/oferecidos. O braço de prefill usa draft
desligado em ambos. DFlash2, quando presente, é uma medição separada, com
pesos do drafter identificados. INT8 da Strata e Q8_0 do Bee são as
respectivas receitas de cache. Guardar diferenças de template e EOS,
mesmo com mensagens idênticas.

No Bee, usar a receita de prefill publicada (`-b 2048 -ub 2048`, cache
MoE de 2.048 MiB, `-ncmoe 2`, GLU e SHARED_Q8 ligados). Rodar uma
segunda passagem diagnóstica com `LLAMA_OP_PROFILE_GPU=1` e `-lv 4`
para obter os eventos GPU. Medir o tempo de parede sem profiler separadamente.
Se o caminho da Strata estiver confirmado, testar em A/B independentes o
limite de chunk com `STRATA_NV_CHUNK` e a rota fused com
`STRATA_PF_FUSED`. Conferir o despacho real nos logs antes de atribuir
a diferença.

## 2. QSA direto à FA

`src/models/qwen4exp.cpp` produz a máscara de seleção em `build_qsa_sel`.
`ggml/src/ggml-cuda/top-k-mask.cu` escreve a máscara densa e
`ggml/src/ggml-cuda/fattn.cu` volta a percorrê-la para achar a união de
células da tile. Uma máscara F16 de 32.768 × 2.048 custa 128 MiB por
tensor nessa geometria. Os bytes alocados e o tráfego de memória são
métricas distintas.

Manter a elegibilidade esparsa local Ampere+, Q8_0, D=256 e a mesma ordem de
células físicas. Passar IDs de pool e da cauda com pertinência por query.
Preservar duplicatas filtradas, sentinelas, visibilidade, streams e o
fallback existente. Rodar `test-backend-ops` nos caminhos CUDA
esparso e denso e `test-llama-archs --test-qsa-exact-tail -s 12345`.
Usar a matriz de geometrias do [plano de paridade](../docs/PLAN-strata-parity.md#desenho-e-ab-do-item-4c-de-qsa).
Exigir conjuntos iguais, NMSE ≤ 1e-6, hashes estáveis na mesma geometria
e medir 4k/28k com três pares alternados.

## 3. MMQ roteado

Em `ggml/src/ggml-cuda/mmq.cu`, o seletor usa a largura total do ubatch
em Ampere. Com 512 experts e top-10, 2.048 tokens dão cerca de 40 linhas
por expert em média. Capturar a distribuição real e a ocupação. A média
sozinha esconde experts quentes e ociosos. Confirmar a rota antes de mexer
no Stream-K, pois a fusão GLU já seleciona outro caminho. Testar apenas
a geometria Q2_0 roteada. O J global de 64 já regrediu e o reuso isolado
do mapa no down rendeu 0,70%.

Para a fusão completa, usar `test-mmq-gate-up` com duas sequências,
m=640, 2.048 tokens, 512 experts e top-10. Comparar máximo, escala,
arredondamento e soma dos blocos Q8_1 do down, depois o resultado por
expert. Rodar `test-backend-ops` em `MUL_MAT_ID`/`MUL_MAT` e
`test-llama-archs -a '^deepseek2$' -b CUDA0 -s 12345`. Validar com e
sem CUDA Graphs. Medir throughput após fechar a paridade numérica.

## 4. Escolher GDN ou memória pelos dados

A fusão do L2 do GDN já foi testada com ganho próximo de zero. Se o scan
aparecer no topo do perfil Q2_0, isolar scan temporal e movimentação de
estado e provar equivalência em chunks e sequências. Se a limitação
aparecer na reserva, medir liveness dos temporários do grafo. O caso com
cache e `-ub 4096` esgotou VRAM ao reservar 2.175 MiB. Repetir a varredura
de tamanho após liberar espaço suficiente. A janela de 256k mede
capacidade e escalabilidade. O CPU keyed segue arquivado após o A/B
IQ3_S mostrar perda de 58,06% no decode e hash divergente.

Builds CUDA na .57, chroot Noble, `-j10` e alvo sm_80 para A100.
Manter a medição fora do profiler para a decisão. Integrar somente acima
do piso de 3% fora da dispersão e com a paridade exigida por cada mudança.

## Tentativa de medição em 23/11/2026

O Colab retornou `Service Unavailable` nas seis tentativas A100 Standard e
na tentativa A100 High Memory. Nenhuma VM foi alocada e nenhuma amostra nova
foi coletada. A referência de 4.629,3 t/s da Strata continua histórica. O
harness antigo fez uma chamada por caso e não registrou `cache_n` no
resumo. A API da Strata desconta o prefixo cacheado de `prompt_n`, então o
reuso pode alterar a geometria medida sem inflar diretamente o numerador.

O pacote oficial Linux CUDA 13 da v0.2.17 traz cubins sm_75, sm_86, sm_89
e sm_120, com PTX sm_120. Para a A100, foi compilada a tag v0.2.20
(`a34979fc94f3d126b9bb8210f4ad311345688b52`, engine 0.1.66) em modo
Release portátil para `80-real`. O `cuobjdump` encontrou 56 cubins sm_80.
SHA-256 do executável:
`25b2f005a91a2c086c5cf40d6733bc2c7254f7cdd90c77df91d9162c1d78d15c`.

O pacote e o harness estão em `/tmp/opencode/strata-base23/` no gokaya
e na .157. O primeiro braço mede Strata sem draft, o segundo mede o
BeeLlama parity13 sem draft e a terceira passagem coleta o perfil Q2_0.
O resultado do harness está na seção de baseline pareado em A100-80G.

## Baseline local na RTX 4070 de 12 GiB

Em 23/11/2026, Strata v0.2.20/0.1.66 (`a34979fc9`) e BeeLlama
`parity13` rodaram o mesmo Qwen3.6-35B-A3B UD-IQ2_XXS no NVMe, com
KV F16, contexto 4.096, uma sequência, greedy e sem tokens de draft.
Os bins eram os pacotes SM80 usados na preparação da A100, executados na
4070. Foram dois warmups por processo e três repetições por comprimento,
com os mesmos IDs de entrada, zero tokens reaproveitados e 64 de saída.
Os hashes dos IDs gerados coincidiram nos três casos entre os braços
principais.

| Braço | pp128 / tg | pp1024 / tg | pp3584 / tg | Pico de VRAM |
|---|---:|---:|---:|---:|
| BeeLlama, camadas na GPU, ubatch 512 | 891,23 / 125,45 | 2.410,51 / 124,40 | 2.828,19 / 121,32 | 11.551 MiB |
| Strata, chunk 512, 206/256 experts por camada em VRAM | 375,15 / 130,43 | 610,36 / 125,75 | 648,72 / 111,39 | 11.265 MiB |
| Strata, chunk 512, `STRATA_PF_FUSED=0` | 364,57 / 129,90 | 609,85 / 125,92 | 646,48 / 111,92 | 11.265 MiB |
| Strata, chunk auto (2.048), 190/256 em VRAM | 333,16 / 130,57 | 532,92 / 125,87 | 572,71 / 111,50 | 11.325 MiB |
| BeeLlama, experts de 8 camadas na CPU | 203,18 / 85,09 | 744,90 / 85,45 | 844,52 / 83,62 | 9.715 MiB |

Valores em t/s, medianas. O placement mudou o pp3584 do próprio Bee em
−70,1%. O braço CPU de oito camadas é uma ablação de placement, com
distribuição distinta do cache da Strata. Na Strata, o GGUF selecionou
os kernels agrupados já no braço padrão. Aumentar o chunk tomou espaço
dos experts e reduziu o pp3584 em 11,7%. O resultado explica por que
comparar apenas t/s com VRAM de pico semelhante esconderia o gargalo.

No perfil separado do Bee, sete grafos de 512 tokens somaram 1.329,22 ms
em eventos CUDA. `MUL_MAT_ID` respondeu por 53,0% e `MUL_MAT` por 30,1%.
`GATED_DELTA_NET` respondeu por 4,0%. Para este GGUF o próximo ponto de
investigação é ganho por projeção MoE em GPU, com contadores de largura
real por expert. O Qwen3.6 não exercita o QSA do Flash-Next, e a diferença
da A100 continua pendente.

Resultados, prompts e logs em `/tmp/opencode/strata-local4070/`.

## SM89 na RTX 4070 e largura de tile do MMQ

Repetição local com builds `89-real` no gokaya, `-j10`. Strata
v0.2.20/0.1.66 (`a34979fc9`), BeeLlama `a32ed2279` com a instrumentação
e o seletor experimental no worktree `diag/strata-native4070`. Mesmo
GGUF UD-IQ2_XXS, contexto de 4.096 tokens, KV F16, 512 tokens por ubatch
no Bee e por chunk no braço principal da Strata, uma sequência, greedy.
Dois warmups e três repetições por caso. O harness exigiu leitura integral
do prompt e zero reuso. As amostras de throughput geraram 64 tokens cada.
Strata ofereceu zero drafts.

| Braço | pp1.024 | pp3.584 | Decode pós-3.584 | Pico de VRAM |
|---|---:|---:|---:|---:|
| BeeLlama SM89 | 2.877,25 | 3.435,55 | 122,99 | 11.551 MiB |
| Strata SM89, chunk 512, 206/256 experts por camada na GPU | 613,84 | 652,62 | 113,45 | 11.265 MiB |
| Strata SM89, chunk auto 2.048, 190/256 experts por camada na GPU | 531,98 | 572,67 | 113,45 | 11.325 MiB |
| BeeLlama Ada, largura de seleção 1× a média por expert | 2.478,86 | 2.846,90 | 123,27 | 11.551 MiB |
| BeeLlama Ada, largura de seleção 2× a média por expert | 3.115,51 | 3.715,16 | 122,75 | 11.551 MiB |

Throughput em t/s, medianas. No pp3.584, o fator 2 ganhou **8,14%**.
As três rodadas do padrão ficaram entre 3.388,55 e 3.440,91 t/s, e as
do fator 2 entre 3.645,29 e 3.729,29. O fator 1 perdeu 17,13%.
Os hashes coincidiram entre os braços em todos os comprimentos medidos.

No segundo prompt, extraído de `src/models/qwen35moe.cpp` e cortado a
3.584 tokens, o fator 2 passou de **3.063,63 para 3.490,38 t/s (+13,93%)**.
As faixas foram 3.037,78–3.066,25 e 3.477,30–3.490,61. Pico idêntico de
11.551 MiB, decode 122,91 → 122,72 t/s e hash igual nas seis execuções.
Corrigi a seleção dos warmups após o primeiro ensaio desse prompt falhar
antes de coletar amostras. O `test-mmq-gate-up CUDA0` passou em 9/9
nos três modos, todos bit-exact, incluindo o fixture Q2_0 de 2.048 tokens,
512 experts e top-10.

Um trace separado, com CUDA Graphs desligados, coletou 1.680 chamadas MMQ
por braço, incluindo os dois prompts. Para ubatch 512, a média foi 16
linhas por expert. No prompt repetitivo, p50/p95/p99 foram **4/68/143**.
No código C++, **6/64/162**. Houve até 511 linhas num expert. O seletor
atual escolheu J=128 em todas as chamadas, 1× escolheu J=16 e 2× escolheu
J=32. A fração estimada de posições úteis nos tiles não vazios foi
16,7%, 70,1% e 50,5%, respectivamente. Essa fração mede preenchimento
do tile. O braço de 70,1% foi mais lento a 3.584 tokens, então o custo
de subdividir os experts mais quentes também importa.

`GGML_CUDA_MMQ_ADA_TYPICAL_MOE` permanece default-off e restrito ao
worktree experimental. Os dados sustentam o fator 2 no Qwen3.6 IQ2_XXS
desta 4070. O fixture verificou corretude para Q2_0 e a medição de ganho
no Flash-Next ainda está pendente. Artefatos em
`/tmp/opencode/strata-native4070/` (`results.json`, `routing.json`,
`mmq-routing-experiment.patch` e logs de build e teste).

## Build SM80 do experimento Q2_0

O candidato está no worktree `diag/strata-sm80-mmq`, sobre
`a32ed2279`. `GGML_CUDA_MMQ_SM80_Q2_MOE_WIDTH=1` usa duas vezes a
largura média no seletor dos GEMMs roteados Q2_0 quando o kernel compilado
é SM80. A flag fica desligada por padrão. O pacote foi compilado no gokaya
em `-j10`, com Flash Attention F16/F16 e Q8_0/Q8_0 e KVarN CUDA desligado.
Os dois braços do A/B usam este mesmo pacote.

O fixture CUDA registrou 9/9 bit-exact e NMSE 0 com a flag em 0 e em 1.
No caso de 2.048 tokens, 512 experts, top-10 e 640 canais, o trace mudou
de J=128 para J=80. A validação local rodou o cubin SM80 na 4070. O
desempenho do Flash-Next depende do A/B com o GGUF real na A100.

Pacote: `/tmp/opencode/strata-sm80-mmq/beellama-sm80-mmq.tgz`.
SHA-256: `944cbb2346d59bd9b4fea98a7d3d756c97f8a762afd750c270acff80279c2bbb`.
O Colab só recebeu a nova solicitação de A100 após a validação do pacote
e a restauração da produção.

As três tentativas desta rodada após a validação (duas Standard e uma
High-RAM) retornaram `Service Unavailable` na conta Pro. O Colab ficou sem
sessões nem alocações ativas. O pacote verificado e o harness ficaram
preparados na .157.

## Pacote G4 para o A/B Q2_0

O worktree `diag/strata-g4-mmq` contém o mesmo experimento de largura
média adaptado ao alvo `sm_120a`, com a flag independente e default-off
`GGML_CUDA_MMQ_SM120_Q2_MOE_WIDTH`. O build de GPU, no gokaya com
`-j10`, inclui Flash Attention F16/F16 e Q8_0/Q8_0. O `cuobjdump`
confirmou cubins `sm_120a`. O pacote inclui o servidor e
`test-mmq-gate-up`, que precisa passar na G4 antes da transferência
dos dois shards do GGUF.

Arquivo: `/tmp/opencode/strata-g4-mmq/beellama-g4-mmq.tgz`, verificado
também na .157. SHA-256:
`1c5c1a9b31f802261b4bf490dc4affb46c0e4b63e5e0cc9493c3eafde28524b2`.

As três solicitações de G4 na conta Pro, feitas após validar o pacote
e restaurar a .57, retornaram `Service Unavailable`. O fixture Blackwell
e o A/B do GGUF seguem sem amostras nessa GPU.

## Pesos reais Q2_0 e protótipo de preparação QSA na 4070

Extraí gate, up e down do bloco 0 do shard fixado do Flash-Next Q2_0.
São 675 MiB de pesos reais, com shapes de gate/up 2.560 × 640 × 512 e
de down 640 × 2.560 × 512. Ativações e IDs de roteamento são sintéticos.
O harness usou CUDA SM80 na 4070, top-10, 512/1.024/2.048 tokens,
três rodadas alternadas e sete amostras por rodada. Os tempos medem
o grafo com sincronização. A captura do despacho rodou separadamente.

Com o tile reduzido em todas as projeções, no caso de 2.048 tokens e
roteamento enviesado, gate e up isolados foram cerca de 4% mais rápidos,
mas o down passou de 7,25 para 8,16 ms. O grafo completo empatou em 14,46 ms.
O modo experimental `GGML_CUDA_MMQ_SM80_Q2_MOE_WIDTH=2` limita a seleção
por largura média às projeções com ativação compartilhada entre experts.
No trace de 2.048 tokens, gate, up e a fusão GLU usaram J=80 e o down
ficou em J=128. O trace confirmou a fusão gate/up/GLU nos dois braços.

| Tokens | Roteamento | Padrão, ms | Tile menor só em gate/up, ms | Capacidade por segundo |
|---|---|---:|---:|---:|
| 512 | Uniforme | 10,441 | 6,756 | +54,53% |
| 512 | Enviesado | 10,307 | 7,099 | +45,19% |
| 1.024 | Uniforme | 11,295 | 8,255 | +36,82% |
| 1.024 | Enviesado | 11,911 | 9,127 | +30,51% |
| 2.048 | Uniforme | 13,185 | 11,893 | +10,86% |
| 2.048 | Enviesado | 14,477 | 13,531 | +7,00% |

Hashes das cinco saídas, incluindo GLU e down do grafo completo,
coincidiram nos seis casos. O ganho mede apenas o caminho roteado de uma
camada. O modo experimental está no worktree `diag/strata-sm80-mmq`.
O pacote SM80 previamente publicado na .157 antecede o modo 2.

O protótipo QSA offline consumiu top-k, scores, pool→célula e cauda e
emitiu união ordenada e oito bits de pertinência por célula. A referência
executou as duas rotinas atuais de seleção por máscara e compactação na
mesma stream CUDA, usando a geometria de máscara binária. União, ordem,
contagem e pertinência coincidiram nas quatro geometrias exercitadas,
incluindo duas sequências, sentinelas e tiles incompletas. Para KV 32.768
e 2.048 queries numa sequência, a preparação passou de **0,865 para
0,391 ms (−54,8%)**. A máscara temporária de **128 MiB** foi substituída
por **4,01 MiB** de pertinência em VRAM e 32 KiB de bitsets em memória
compartilhada por tile. As listas e contagens comuns aos dois braços
ficam fora dessa comparação de memória. O protótipo ainda precisa ser
conectado à FA Q8_0 e medido junto da atenção.

Artefatos e relatório JSON em `/tmp/opencode/flashnext-layer4070/`.
A produção foi restaurada e confirmou `/health {"status":"ok"}`.

## QSA direto conectado à Flash Attention Q8_0

O worktree `diag/qsa-direct`, sobre `a32ed2279`, liga o QSA direto à
Flash Attention pelo opt-in `LLAMA_QSA_DIRECT_FA=1`. No grafo Qwen4Exp,
só troca a seleção por máscara após conferir a geometria esparsa e o
backend CUDA que detém K e V. O controle de seleção continua disponível
para KV tail, KVarN, tamanhos e placements fora da rota. O caminho
esparso carrega a pertinência por query nos próprios índices, preserva
a ordem dos K/V e deixa de alocar a máscara F16 intermediária.

Fixture CUDA na 4070 com binário `80-real`, builds no gokaya com `-j10`.
Compara os tempos da preparação a partir dos top-k e scores até a saída
da FA, com K/V Q8_0 sintéticos, D=256, dois heads KV e 24 heads Q. São
medianas de sete amostras alternadas após três warmups por braço.

| KV / queries / streams | Máscara + FA, ms | IDs diretos + FA, ms | Redução de tempo, Graphs on |
|---|---:|---:|---:|
| 4.096 / 1 / 2 | 0,02594 | 0,02516 | 3,01% |
| 16.384 / 4 / 1 | 0,10469 | 0,09925 | 5,20% |
| 32.768 / 13 / 2 | 0,79729 | 0,73441 | 7,89% |
| 32.768 / 512 / 1 | 6,94813 | 6,12340 | 11,87% |
| 32.768 / 2.048 / 1 | 25,87625 | 24,25715 | 6,26% |

As cinco geometrias passaram bit-exact com NMSE 0 nos dois modos de CUDA
Graphs. Mudar a cauda mudou a saída esperada e preservou a igualdade
entre braços. O trace confirmou 10 capturas e 90 lançamentos nos dois
braços, incluindo as reexecuções após a atualização. O fixture verificou
fallback por tamanho do KV e por GQA. A regressão CUDA legada Q8_0 passou
20/20, e o fixture CPU passou com NMSE 0 para exact-tail e rollback GDN
de duas sequências.

O primeiro teste encontrou a validação CUDA que interpretava todo
operando adicional como cauda. Após a correção, o verificador reconhece
o formato direto e continua aceitando apenas o contrato validado.
A suíte legada ampla reproduziu `max_blocks_per_sm > 0` com a biblioteca
anterior e com a nova. O F16 isolado passou nos dois. Esse caso de ordem
da suíte ficou fora da mudança.

Em 32k × 2.048, o caminho direto retira 128 MiB de máscara intermediária
do grafo. A medição de VRAM do servidor com Flash-Next completo e o A/B
fim a fim ainda estão pendentes. Patch, fixture e resultados em
`/tmp/opencode/qsa-direct4070/`. O pacote gate/up-only SM80
`beellama-sm80-gate-up.tgz` foi verificado separadamente na .157.

## A/B no Flash-Next completo, A100-40G

A sessão Standard de 23/11/2026 entregou uma NVIDIA A100-SXM4-40GB,
40.960 MiB. Após perda da conexão do notebook, o SSH do Colab retomou
a mesma VM. Os dois shards Q2_0 tiveram o SHA-256 da revisão fixada
confirmado. Cada par usou um único binário SM80 nos dois braços.
Configuração `-b 2048 -ub 2048 -ncmoe 2 --moe-cache-mib 2048`,
contexto 32.768, KV Q8_0/Q8_0, `--parallel 1 --fit off` e GLU e SHARED_Q8
ligados. Dois warmups e três amostras por caso, sem draft e `cache_n=0`.

| Experimento | pp4k off → on | pp28k off → on | Decode pós-28k off → on | Pico de VRAM off → on |
|---|---:|---:|---:|---:|
| Tile menor só em gate/up | 1.893,44 → 1.928,98 (+1,88%) | 1.960,20 → 1.963,87 (+0,19%) | 45,55 → 45,23 | 38.904 → 38.906 MiB |
| QSA direto | 1.889,65 → 1.894,85 (+0,27%) | 1.953,67 → 1.968,13 (+0,74%) | 44,96 → 45,25 | 38.906 → 38.906 MiB |

Valores em t/s, medianas. Os dois A/Bs mantiveram o hash de saída nos
três casos, com `short`, `pp4k+dec` e `pp28k+dec`. Nenhum alcançou
o piso de 3%. As faixas de pp28k foram 1.945,28–1.960,29 e
1.948,27–1.964,20 no gate/up e 1.937,25–1.953,73 e
1.950,26–1.968,31 no QSA. Ambos seguem experimentais.

Nos traces separados, o gate/up fundido passou de J=128 para J=80 no
ubatch de 2.048, com down em J=128. O QSA direto eliminou os 204 eventos
de compactação por máscara observados no controle, com 196 eventos de
geometria do indexador nos dois braços. A alocação intermediária
eliminada no fixture local não se traduziu em redução do pico de VRAM
do servidor nesta receita. A sessão foi encerrada e o Colab confirmou
zero alocações ativas. A produção local seguiu saudável.

Artefatos em `/tmp/opencode/a100-flashnext-latest/`: `baseline.json`,
`summary.json`, `bench-logs.tgz` e os manifests dos dois pacotes.

## Baseline pareado na A100-80G

Em 23/11/2026, BeeLlama `a32ed2279` e Strata v0.2.20
(`a34979fc9`, engine 0.1.66) rodaram na mesma A100-SXM4-80GB
com 81.920 MiB e driver 580.82.07. Os SHA-256 dos dois shards originais
Q2_0 foram conferidos. Ambos receberam os mesmos IDs de entrada,
com contexto de 32.768 tokens, greedy, dois warmups e três amostras
por caso. O harness verificou zero reuso de prompt e zero drafts.

| Tokens de entrada | Prefill BeeLlama | Prefill Strata | Decode BeeLlama | Decode Strata |
|---:|---:|---:|---:|---:|
| 24 | 100,51 | 239,28 | 59,33 | 95,78 |
| 4.627 | 1.919,47 | 3.806,66 | 45,75 | 93,41 |
| 28.547 | 1.955,92 | 5.046,94 | 45,18 | 91,92 |

Medianas em t/s. Em 28k, a Strata fez 2,58× no prefill e 2,03× no
decode. A faixa de prefill em 28k foi 1.950,78–1.956,75 no BeeLlama
e 5.046,49–5.055,70 na Strata. Os hashes foram estáveis por engine
e coincidiram entre engines no prompt curto. Os casos longos
também tiveram números diferentes de tokens gerados.

O BeeLlama usou KV Q8_0/Q8_0, GLU e SHARED_Q8, `-b 2048 -ub 2048
-ncmoe 2 --moe-cache-mib 2048 --parallel 1 --fit off`, com pico de
38.904 MiB. A Strata usou KV INT8, `--ple-io ram --prefill auto
--expert-cache auto`, com pico de 37.334 MiB. A PLE de 26,8 GiB
ficou na RAM e o cache GPU de 31,64 GiB carregou os 24.576 experts.
No prefill, a Strata empresta parte desse cache ao workspace e traz de
volta à GPU experts dos slots emprestados. O cache cheio é restaurado
para o decode.
Os logs confirmaram o despacho de prefill para kernels INT8 fundidos
nas 48 camadas. O caso de 28.547 tokens registrou chunk de 28.546.
`--spec 4` reservou o workspace exigido pela CLI. O MTP ficou
desligado e `--suffix-draft 0 --lookup-max-t 0 --lookup-chain 0`
desligaram a especulação por lookup.

Para isolar a diferença em 28k, medir placement dos experts e miss
do cache no BeeLlama, testar chunks menores na Strata sem mudar
placement e separar o tempo de PLE, atenção/QSA, GDN e MMQ com
profiling CUDA. A fusão completa gate/up/GLU/down continua condicionada
à prova numérica. Este comparativo não mediu DFlash2.

Comandos, amostras e hashes em `/tmp/opencode/a100-strata-paired/paired.json`,
`summary.json`, `prompt-tokens.json` e `paired-logs.tgz`. A sessão
`strata-paired-23` ficou ativa para a próxima medição.

## A/B de chunk e placement na mesma A100-80G

O seguimento reutilizou a sessão, os binários, os dois shards e os IDs de
entrada do baseline pareado. Foram dois warmups curtos e três amostras por
caso, com seed 42, temperatura zero, sem drafts e sem reuso de prompt.
Na Strata, variou-se `--prefill` entre `auto`, `auto:8192` e `2048`.
Os testes com `--no-prefill-borrow` separaram o workspace do cache de
experts. No BeeLlama, variaram-se `-ncmoe` e o par `-b/-ub` em A/Bs
independentes. As passagens de profiling usaram somente pp28k seguido
de um token e ficaram fora das medianas.

| Braço | Prefill 4k | Prefill 28k | Decode após 28k | Pico (MiB) |
|---|---:|---:|---:|---:|
| Strata auto, workspace emprestado | 3.807,92 | 5.053,19 | 91,84 | 37.334 |
| Strata teto 8.192, workspace emprestado | 3.809,80 | 5.026,77 | 92,44 | 37.334 |
| Strata teto 2.048, workspace emprestado | 3.285,52 | 3.907,44 | 91,97 | 37.334 |
| Strata 2.048, workspace próprio | 4.194,54 | 4.455,04 | 91,90 | 38.850 |
| Strata 28.546, workspace próprio | 4.751,00 | 5.869,52 | 92,05 | 46.024 |
| BeeLlama `-ncmoe 2 -b 2048 -ub 2048` | 1.912,64 | 1.957,02 | 45,70 | 38.904 |
| BeeLlama `-ncmoe 0` | 1.930,43 | 1.973,07 | 45,77 | 38.898 |
| BeeLlama `-b 3840 -ub 3840` | 1.988,02 | 2.044,02 | 45,48 | 40.770 |
| BeeLlama `-b 4096 -ub 4096` | 2.009,73 | 2.093,79 | 45,55 | 41.036 |
| BeeLlama `-b 8192 -ub 8192` | 2.248,73 | 2.146,92 | 45,81 | 45.268 |

Medianas em tokens/s. No prompt de 28k, a Strata reportou o primeiro
chunk de 7.137 ou 2.039 quando o teto foi 8.192 ou 2.048 e o
workspace foi emprestado.
Nos braços com workspace próprio, a Strata registrou zero experts
transmitidos em 28k. Aumentar o chunk de 2.048 para 28.546 com essa
residência fixa elevou o prefill longo em 31,8%. Separar o workspace
a 2.048 rendeu 14,0% contra o braço de tamanho próximo com empréstimo.
O teto de 8.192 com empréstimo ficou 0,52% abaixo do auto em 28k.

No BeeLlama, o braço `-ncmoe 2` já cacheou 100% dos 1.024 experts das
duas camadas em host. Colocá-las diretamente na GPU rendeu 0,82% em
28k e manteve os hashes dos três casos. Os lotes 4.096 e 8.192
ganharam 7,0% e 9,7% em prefill de 28k, ambos acima do teto de
40.960 MiB. O lote 3.840 ganhou 3,9% em 4k e 4,45% em 28k dentro
do teto medido na A100-80G. Em 28k, as três amostras foram
2.028,65 / 2.044,02 / 2.044,09 t/s, contra
1.950,30 / 1.957,02 / 1.957,59 do controle. O braço 4.096 alterou
o hash em 4k. Os braços 3.840 e 8.192 alteraram os hashes em 4k e
28k. A receita de referência continua em 2.048. A execução de 3.840
na A100 física de 40 GB segue pendente.

Na Strata, o teto de 8.192 mudou o hash em 28k. Os braços com
workspace próprio mantiveram os hashes do auto.

### Perfil de prefill em 28k

No BeeLlama, `LLAMA_OP_PROFILE_GPU=1` registrou 13,548 s em 15 grafos.
`MUL_MAT_ID` foi a maior categoria individual (5,513 s, 40,7%),
seguida de `MUL_MAT` (3,710 s, 27,4%), `TOP_K` (1,257 s),
`FLASH_ATTN_EXT` (1,011 s) e `GATED_DELTA_NET` (0,568 s).
São categorias por operação e incluem fusões; `MUL_MAT` e `TOP_K`
agregam usos distintos. O scheduler registrou 1,30 GiB de cópias
lógicas H2D e 0,142 s de tempo de host nas chamadas de transferência,
sem medir a duração do DMA. O prefill instrumentado levou 15,038 s.

Na Strata, `STRATA_PREFILL_TIMING=1` registrou uma timeline de 4,872 s,
dos quais 1,980 s no trecho rotulado `gemm gate/up`, que inclui
`experts_native` com o down nesta rota. As fases QSA somaram 1,068 s
e as de GDN, 0,900 s. O timer da Strata inclui esperas entre marcas.
Cada engine instrumenta fronteiras diferentes. A categoria
`MUL_MAT_ID` do BeeLlama é o primeiro alvo para decompor gate/up e down
por tamanho do grupo e validar a fusão completa em fixture.

Resultados por repetição em
`/tmp/opencode/a100-strata-paired/placement-ab.json`, com agregados em
`placement-summary.json` e logs em `placement-logs.tgz`.

## Auditoria das flags e do MMQ down na A100-80G

Escopo da auditoria: BeeLlama `a32ed2279` no pacote SM80 da sessão
`strata-paired-23`, Flash-Next Q2_0, Q8_0/Q8_0, 32.768 de contexto,
`-b 2048 -ub 2048 -ncmoe 2 --moe-cache-mib 2048`, um slot e sem draft.
O arquivo `env-audit.json` registra as variáveis recebidas pelo
processo. Os logs, eventos do runtime e lançamentos CUDA confirmam
o despacho onde há instrumentação.

| Flag ou mecanismo | Estado na receita e evidência |
|---|---|
| `LLAMA_MOE_CACHE_PREFILL=1`, `LLAMA_MOE_MAP_DEVICE=1` | Ativos. O log confirma o grupo integral com 1.024 slots, o mapa imutável e o uso durante o prefill. |
| `LLAMA_MMQ_GATE_UP_GLU=1` | Ativa. 644 dispatches em 14 ubatches, 46 das 48 camadas MoE. As duas camadas com pesos no cache percorrem gate e up separados. |
| `LLAMA_MMQ_GATE_UP_SHARED_Q8=1` | Configurada, com zero dispatch na captura de prefill. O pareamento é uma alternativa quando a fusão GLU não casa e a geometria permite o par. |
| `LLAMA_MOE_CACHE_PARTIAL_PREFILL`, `LLAMA_MOE_UPLOAD_OVERLAP`, `LLAMA_MOE_PINNED_RING` | Desligadas. A receita já tem cache integral nas duas camadas. Overlap e ring interessam para preenchimentos do cache. Os pesos em host desta receita já usam `CUDA_Host`. |
| `LLAMA_QSA_INDEXER_BOUNDS`, `LLAMA_GDN_CONV_L2`, `LLAMA_PLE_PREFETCH` | Desligadas. Os A/Bs anteriores no Flash-Next ficaram no ruído, incluindo o A/B frio de PLE. |
| GDN MMA e `GGML_CUDA_GDN_PREFIX_MMA` | O Nsight registrou 504 kernels `gdn_single`. O prefixo experimental exige rollback com `K>1`; o log desta receita mostrou `K=1`. `GGML_CUDA_GDN_SEGMENTED` é específico de Ada. |
| CUDA Graphs e `GGML_CUDA_GRAPH_OPT` | Graphs disponíveis conforme elegibilidade. Os 15 grafos do prefill instrumentado executaram diretamente, todos com razão `WARMUP`. A sobreposição explícita com `GRAPH_OPT=1` em Ampere ficou dentro da dispersão em 28k. |
| `GGML_CUDA_MASK_PACK` | Habilitada por padrão para máscaras binárias host elegíveis. Nenhum kernel de descompactação na captura. A QSA usou a máscara calculada no device, com 84 compactações por máscara observadas no trace. |
| `GGML_CUDA_MMQ_PREC`, `GGML_CUDA_MMQ_J_MAX`, `GGML_CUDA_CUBLAS_COMPUTE_TYPE` | Sem overrides. O MMQ roteado usou Q8_1 nas ativações. O teto global de tile 64 já regrediu em medição anterior. Forçar a precisão do cuBLAS é uma mudança numérica separada. |
| `GGML_CUDA_DQ_MMV`, `GGML_CUDA_DQ_Q6K` | Desligadas por padrão em Ampere. Podem alcançar projeções densas Q4_K/Q5_K/Q6_K de um token. As três quantizações aparecem entre os pesos deste GGUF. |
| `LLAMA_QSA_DIRECT_FA`, `GGML_CUDA_MMQ_SM80_Q2_MOE_WIDTH`, `LLAMA_MMQ_MOE_REUSE_MAP`, `LLAMA_MOE_CPU_FALLBACK` | Variantes experimentais fora da receita. O pacote de perfil contém a variante de largura, fixada em `0`. As outras são de branches separados. |

PDL depende de Hopper ou posterior. Ajustes GB10, EXL3, NVFP4, Q1_0,
Q2_K, PTQ1_0, multi-GPU, HIP, KVarN, RKVA e MTP dependem de hardware,
tipos de peso/cache ou caminhos fora desta execução. DFlash2 e o
controlador profit têm uma receita separada. A auditoria cobre flags
de despacho e tuning desta configuração de benchmark.

Três amostras após dois warmups, mesmas entradas, sem reuso.

| A/B de ambiente | Prefill 28k | Decode curto | Decode pós-28k | Pico (MiB) | Hashes iguais ao controle |
|---|---:|---:|---:|---:|---|
| Controle | 1.957,10 | 60,40 | 45,88 | 38.904 | Sim |
| `GGML_CUDA_GRAPH_OPT=1` | 1.955,75 | 60,04 | 45,73 | 38.904 | Sim |
| `GGML_CUDA_GRAPH_MIN_STABLE_CALLS=1` | 1.958,62 | 61,81 | 46,01 | 38.958 | Sim |
| `GGML_CUDA_DQ_MMV=1 GGML_CUDA_DQ_Q6K=1` | 1.954,61 | 58,90 | 44,88 | 38.904 | Divergiu em 4k |

Medianas em t/s. O limiar de captura 1 ganhou 2,33% no decode
curto com 54 MiB a mais. O caminho DQ perdeu 2,18% no decode
pós-28k. Nenhum dos três overrides entrou na receita.

### Compactar a grade do MMQ down

Em um pp28k instrumentado, o Nsight 2025.3.1 separou 2,94 s no
`mul_mat_q` do down, 2,38 s nos GEMMs de gate/up e 0,16 s de
roteamento mais quantização do down. A análise identificou o down
pela quantização Q8_1 de uma linha por token e expert imediatamente
depois do helper de IDs. Gate/up usa quantização broadcast por token,
com o booleano de fusão no template do kernel.

O down da camada residente lança 163.840 CTAs com 2.048 tokens,
top-10, 512 experts, 2.560 linhas de saída e tile 128×128. O lançador
atual dimensiona as colunas como se cada expert pudesse receber o
ubatch inteiro. A geometria usa Stream-K com um bloco por tile e sem
fixup no trace. Na camada com mapa de cache de 1.024 slots, a grade
chega a 327.680 CTAs.

O candidato é reaproveitar a indexação compacta do gate/up,
`floor(bound/J) + expert`, para o down Q2_0 no SM80, mantendo tile
128 e o mesmo particionamento de K por tile. Usar o buffer
`expert_bounds` existente, sem leitura de volta para o host. Uma
verificação offline reproduziu o conjunto exato de tiles úteis em 816
histogramas reais de roteamento. A grade calculada caiu de
163.840 para 13.440 CTAs em 512 experts e de 327.680 para 23.680 em
1.024 slots. São contagens de lançamento previstas. O tempo de CUDA
da variante ainda precisa ser medido.

O primeiro ensaio de GPU deve usar Q2_0 com K=640, saída 2.560,
512/1.024 slots, top-10, 2.048 tokens e caudas, com experts vazios e
a distribuição real. Exigir os mesmos bits da saída antes da redução
ponderada, em execução direta e com CUDA Graphs. Depois medir pp4k e
pp28k com a receita acima e exigir o piso de 3% fora da dispersão,
hashes iguais e pico dentro de 40.960 MiB. A fusão completa
gate/up/GLU/down fica depois desse A/B. A medição da preparação do
down soma cerca de 1% do pp28k, de modo que eliminar apenas o helper
e a quantização tem um teto baixo nesta configuração.

Artefatos em `/tmp/opencode/a100-strata-paired/`, incluindo
`env-audit.json`, `env-audit-summary.json`,
`moe-kernel-timing-confirmed.json`, o SQLite do Nsight e
`down-grid-coverage.json`.

### Variante SM80 com grade compacta

`GGML_CUDA_MMQ_SM80_Q2_DOWN_COMPACT=1` seleciona uma especialização
para o down roteado Q2_0 na A100. A geometria exige K=640, saída 2.560,
tile 128, 512 ou 1.024 slots e uma janela de execução em que o
Stream-K de referência usa todo o eixo K por tile. A grade usa o
mesmo `expert_bounds` que gate/up já usa. A flag fica desligada por
padrão. A primeira versão usava uma ramificação no próprio kernel.
A variante final tem uma especialização própria, com o uso de
registradores e pilha do kernel de referência preservado.

O build experimental recompilou `mmq.cu` e a unidade de instanciação
Q2_0 e religou `libggml-cuda` na .57, no chroot Noble, com os comandos
do build SM80 anterior. Os outros objetos CUDA e o servidor vieram
daquele build. O patch está em
[`ggml/src/ggml-cuda/mmq.cuh`](../ggml/src/ggml-cuda/mmq.cuh).
Os scripts e artefatos da rodada ficam em `/tmp/opencode/sm80-down-compact/`.

O fixture CUDA passou em seis geometrias, com 531/1.923/2.048 tokens e
512/1.024 slots. Usou histogramas reais, seis repetições com ativações
e IDs renovados, e checou captura e replay. O SHA-256 dos bytes de
saída coincidiu na biblioteca original, na variante desligada, em
execução direta com a variante e com CUDA Graphs.

No Flash-Next Q2_0, a sequência desligado/ligado/ligado/desligado usou
o mesmo binário, dois warmups e três repetições por processo, com os
mesmos IDs, sem reuso e sem draft. Cada valor abaixo é a mediana de
seis execuções por braço, em t/s.

| Métrica | Desligada | Ligada | Delta |
|---|---:|---:|---:|
| Prefill 4.627 tokens | 1.918,28 | 2.069,41 | +7,88% |
| Prefill 28.547 tokens | 1.955,64 | 2.136,06 | +9,23% |
| Decode pós-4k | 46,38 | 46,25 | −0,29% |
| Decode pós-28k | 45,71 | 45,65 | −0,14% |
| Pico de VRAM (MiB) | 38.904 | 38.904 | 0 |

Em 28k, as faixas de prefill foram 1.950,76–1.960,32 e
2.132,07–2.137,66 t/s. Os hashes do prompt curto, 4k e 28k
coincidiram com a biblioteca original. No Nsight, o down passou de
2,941 para 1,591 s de tempo somado de kernel (−45,9%) em 672
lançamentos. A captura confirmou 672×20 blocos no ubatch de 2.048
com 512 slots e 1.184×20 com 1.024 slots.

O resultado ultrapassou o piso de 3% usando a biblioteca reconstruída
seletivamente. A flag experimental continua desligada por padrão.
A validação de release ainda pede build CUDA completo e confirmação
em A100 física de 40 GB. A sessão `strata-paired-23` ficou ativa.
Medianas, amostras, hashes e contagens estão em
`/tmp/opencode/sm80-down-compact/summary.json`.

### Diferença para a Strata após a compactação

A comparação usa os mesmos pesos e IDs de entrada na mesma sessão
A100-80G. As medianas da Strata v0.2.20 vêm do baseline pareado
anterior. O BeeLlama usa a grade compacta ligada. Os dois braços
usaram seus formatos de KV nativos, sem reuso de prompt e sem draft.

| Métrica | BeeLlama | Strata | Strata / BeeLlama |
|---|---:|---:|---:|
| Prefill 4.627 tokens | 2.069,41 | 3.806,66 | 1,84× |
| Prefill 28.547 tokens | 2.136,06 | 5.046,94 | 2,36× |
| Decode pós-4k | 46,25 | 93,41 | 2,02× |
| Decode pós-28k | 45,65 | 91,92 | 2,01× |

Valores em t/s. O pico foi 38.904 MiB no BeeLlama e 37.334 MiB na
Strata. Os 87,3–88,0 t/s históricos do BeeLlama em decode pós-28k
usaram DFlash2 na A100-40G. A medição pareada de DFlash2 está abaixo.

## Comparação com DFlash2 na A100 Standard

O alvo foi o Flash-Next Q2_0 dos mesmos dois shards, com SHA-256
conferidos e contexto de 32.768. O drafter veio de
`incoai/Qwen3.8-Flash-Next-DFlash2`, revisão
`a22a1744e85c17364dcc54a6a420af7135d4c7f6`. Os 81 tensores do
drafter têm bytes idênticos no DF2P da Strata e no GGUF do
BeeLlama, com matrizes Q8_0, codebooks BF16 e vetores F32.

O limite pareado foi de sete propostas. A Strata usa `--spec 8
--draft-cap 8` porque a janela inclui o token âncora. O BeeLlama
usa `--spec-draft-n-max 7`. O braço adicional com `n-max 8`
mantém a configuração histórica do BeeLlama como referência.
Os dois engines usam greedy, uma sequência, sem lookup de sufixo,
MTP ou reuso de prompt. O limite de saída é 16 tokens no prompt
curto e 64 nos demais. Mantivemos os formatos de KV nativos e as
geometrias de prefill das respectivas receitas. As amostras brutas
e os hashes ficam em `/tmp/opencode/dflash-paired-24/`.

BeeLlama `6ee5d6518` com a biblioteca CUDA recompilada seletivamente
e Strata v0.2.20 rodaram na mesma A100-SXM4-40GB. O Bee usou
`-b 2048 -ub 1024 -ncmoe 2 --moe-cache-mib 2048`, KV Q8_0/Q8_0,
GLU e SHARED_Q8. A Strata usou `--prefill auto --expert-cache auto
--ple-io ram --kv int8`. Houve dois warmups curtos e três execuções
por caso. Cada amostra longa emitiu 64 tokens.

| Configuração | Prefill 4k | Decode pós-4k | Prefill 28k | Decode pós-28k | Aceitação pós-28k | Pico (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| BeeLlama, sem draft | 1.674,70 | 45,48 | 1.641,54 | 44,55 | — | 37.838 |
| BeeLlama, 7 propostas | 1.435,50 | 89,40 | 1.430,28 | 76,10 | 47,52% | 39.738 |
| BeeLlama, 7 propostas e grade compacta | 1.511,29 | 89,59 | 1.523,06 | 75,81 | 47,52% | 39.738 |
| BeeLlama, `n-max 8` e grade compacta | 1.504,47 | 88,03 | 1.517,90 | 75,04 | 47,52% | 39.852 |
| BeeLlama, 7 propostas, grade compacta e `-b/-ub 1536` | 1.678,46 | 87,56 | 1.643,43 | 67,12 | 41,44% | 40.298 |
| Strata, sem draft | 3.720,35 | 90,21 | 4.959,87 | 88,96 | — | 37.334 |
| Strata, 7 propostas | 3.651,93 | 124,41 | 4.943,72 | 109,70 | 35,71% | 38.550 |

Throughput em t/s, mediana. A grade compacta ganhou 6,49% em prefill
de 28k no BeeLlama com DFlash2, manteve o pico e o hash nos três
casos e variou −0,38% no decode pós-28k. As faixas de prefill foram
1.414,02–1.434,24 e 1.502,01–1.523,33 t/s. A Strata fez 3,25× o
prefill e 1,45× o decode pós-28k frente à grade compacta. As saídas
entre engines coincidiram no caso curto e em 4k, e divergiram em 28k.
Em 28k, a Strata aceitou 45 de 126 propostas por amostra e o BeeLlama
48 de 101. A primeira amostra longa de cada processo apresentou
decode mais lento que as duas seguintes.

O log do BeeLlama confirma que `n-max 8` foi reduzido para sete
propostas pelo bloco treinado de oito posições. O processo ainda
reservou 114 MiB adicionais. A Strata confirmou DFlash2 ativo,
janelas de até oito posições e MTP ausente.

O BeeLlama com `-b 1536 -ub 1536` ganhou 7,90% de prefill 28k
sobre a grade compacta com `-ub 1024`, caiu 11,46% no decode e
mudou o hash em 28k. O orçamento de 40.960 MiB comportou a execução.
O perfil de referência com DFlash2 mantém `-ub 1024`.

Em 28k, ambos os controles sem draft e a Strata com DFlash2
produziram o mesmo hash. O BeeLlama com DFlash2 produziu outra
sequência greedy nessa geometria, com o mesmo hash ao ligar ou
desligar a grade compacta. Priorizar a localização do primeiro token
divergente no Bee com 64 tokens, primeiro com a mesma geometria e
`n-max 7`, preservando os IDs e os top-2 logits do verificador.
Em uma passagem instrumentada, cronometrar drafter, forward alvo,
aceitação, rollback e sincronizações por janela. A taxa de aceitação
observada por si só não explica a vantagem no decode da Strata.

### Correção do prefill GDN com snapshots

Os cinco primeiros IDs de 28k coincidiram. No sexto, o decode
sem draft usou `421` e o DFlash2 usou `1510`. O contexto DFlash2
com `p_min=1` reproduziu o segundo hash com zero propostas. Um
probe CUDA com a mesma entrada e os mesmos cinco tokens iniciais
isolou a inversão ao passar de zero para sete posições de rollback.
Ativar os taps do DFlash2 no mesmo contexto preservou os logits do
probe. No sexto token, o caso serial sem snapshots deu 20,477304
para `421` e 20,412796 para `1510`. Com snapshots, deu
20,641193 e 20,648544, respectivamente.

O prefill GDN sem snapshots processava o ubatch inteiro por MMA.
Com snapshots, um ubatch de 1.024 tokens processava 1.008 por MMA
e a cauda de 16 pela recorrência escalar. A correção marca o prefill
elegível do Qwen4Exp, usa MMA em todo o ubatch e grava o estado dos
últimos tokens dentro dos tiles MMA correspondentes. O caminho
qualificado exige uma sequência e preserva os guards de spans do
backend. A gravação cobre a posição em que a cauda cruza dois tiles.

No fixture direto da A100-40G, os tamanhos 128, 131, 137, 899 e 1.024,
com oito slots, tiveram atenção e estado final byte a byte iguais ao
MMA sem snapshots. Cada snapshot coincidiu com o estado final da
execução MMA que termina naquela posição (NMSE zero). O A/B de
diagnóstico restaurou o hash sem draft nos três prompts, com a grade
compacta e DFlash2 ativos.

A versão final seleciona o caminho por metadado do grafo Qwen4Exp,
sem a flag de diagnóstico. As três repetições em cada prompt
curto, 4k e 28k passaram com seis e sete propostas, com os hashes
dos controles sem draft. Em 28k, o hash também é o da Strata com
DFlash2. O `test-backend-ops test -b CUDA0 -o GATED_DELTA_NET_CACHE_FUSION`
passou 16/16, incluindo as geometrias novas de 131 e 899 tokens.

| BeeLlama corrigido | Prefill 28k | Decode pós-4k | Decode pós-28k | Pico (MiB) |
|---|---:|---:|---:|---:|
| Sem draft | 1.644,86 | 45,12 | 44,48 | 37.838 |
| DFlash2, sete propostas | 1.517,25 | 83,82 | 70,91 | 39.738 |
| DFlash2, seis propostas | 1.530,72 | 78,39 | 75,31 | 39.624 |

Seis propostas ganharam 6,2% no decode pós-28k e economizaram
114 MiB frente a sete. Em 4k, sete continuaram mais rápidas. Com
saída igual em 28k, a Strata ficou 1,46× à frente dos 75,31 t/s.
A medição usou reconstrução seletiva das bibliotecas CUDA e llama
no chroot da .57. Dados em
`/tmp/opencode/dflash-paired-24/gdn-default-summary.json` e
`gdn-default-results.tgz`.

### Ciclo DFlash2 após a integração

Base `de15cf4fe`, A100-SXM4-40GB, mesmos shards, drafter e IDs da
rodada pareada. O controle usa `-b 2048 -ub 1024`, a grade
compacta, seis propostas, `p_min=0` e sem reuso. Cada ajuste foi
medido separadamente, com três amostras por caso e hashes comparados
ao controle greedy sem draft.

No segundo pedido de 64 tokens após 28k, instrumentado fora do A/B,
a execução do alvo somou **690,0 ms** dos aproximadamente 837 ms
de geração (82,4%). A decomposição do alvo foi 205,4 ms de submissão e
483,1 ms até a sincronização. As outras fases mais caras foram
drafter (69,1 ms), amostragem e aceitação (46,0 ms) e atualização
do drafter com os taps (17,5 ms). A espera do alvo inclui a execução
pendente na GPU. Com `GGML_CUDA_GRAPH_MIN_STABLE_CALLS=1`, os eventos
de launch passaram de 33 para 42 e os grafos diretos de 14 para
5, dentro da janela após o primeiro token. O alvo caiu para
674,1 ms. Os eventos de grafo incluem alvo e draft.

| Braço | Decode pós-4k | Decode pós-28k | Resultado |
|---|---:|---:|---|
| Controle | 78,49 | 75,66 | Hash de referência |
| Captura com limiar 1 | 79,96 | 77,12 | +1,93% em 28k e hashes iguais |
| `--spec-draft-p-min 0.35` | 75,18 | 77,95 | +3,03% em 28k, com hash divergente |
| `--backend-sampling` | — | — | OOM ao reservar 2.058,12 MiB para o grafo de 4k |
| Cópia direta dos taps | 78,53 | 75,90 | +0,32% em decode de 28k e hashes iguais |

Valores em t/s, mediana. O limiar de confiança reduziu as propostas
de 90 para 72 por amostra pós-28k. A saída mudou nos três ensaios,
e o braço foi rejeitado.

A cópia direta preencheu `batch_inject.embd` durante o gather de
taps, eliminando o vetor intermediário. O fixture de CPU do Qwen4Exp
passou com captura, rollback, duas sequências e handoff DFlash2.
No modelo completo, o prefill de 28k subiu de 1.527,33 para
1.560,74 t/s (+2,19%), com pico de 39.624 MiB e hashes iguais.
O ganho ficou abaixo do piso de 3%. O patch da cópia está arquivado
em `/tmp/opencode/dflash-paired-24/tap-copy.patch`.

Em uma captura separada com `LLAMA_OP_PROFILE_GPU=1`, 17 grafos
do alvo na janela de decode registraram 820,8 ms por eventos
CUDA. `MUL_MAT` somou 329,8 ms (40,2%), `TOP_K` 104,0 ms,
`MUL_MAT_ID` 95,6 ms e `FLASH_ATTN_EXT` 44,4 ms. São totais
instrumentados por operação. O profiler altera a sincronização,
e suas durações ficam fora das medianas e da contabilidade de
parede acima. O próximo recorte deve atribuir `MUL_MAT` do alvo
a tensor, dtype e geometria, usando os kernels e as projeções
efetivamente despachadas nos lotes de verificação. O evento
`TOP_K` agrega usos distintos. A medição ainda precisa separar
indexação QSA e roteamento MoE antes de escolher um patch de top-k.

Dados brutos e agregados em `/tmp/opencode/dflash-paired-24/`,
incluindo `gdn-cycle.json` e `gdn-cycle-summary.json`.

### Top-k do verificador e profundidade do draft

Na A100-40G, o trace Nsight do DFlash2 com seis propostas mostrou
aproximadamente 98 ms de `topk_tile` na janela final de decode
pós-28k. A especialização de 4.096 elementos dominou esse tempo.
O QSA do Flash-Next seleciona 512 pools nessa geometria e o
caminho CUDA anterior usava 512 reduções de máximo por tile.

Usar cinco propostas preservou o hash nos três prompts, com
**90,98 t/s em decode pós-4k** e **80,21 t/s pós-28k**, contra
78,49 e 75,66 t/s com seis. A VRAM caiu de 39.624 para 39.512 MiB.
Quatro propostas deram 82,16 e 76,80 t/s, também com hashes iguais.
A receita de cinco refere-se ao drafter pareado da rodada acima,
com `p_min=0`.

O candidato troca só a seleção dentro de cada tile por
`cub::BlockRadixSort` quando o CUDA executa em SM80, a largura
excede 4.096 elementos, o tile é de 4.096 e `k=512`. Usa as
mesmas chaves de 64 bits, com score ordenável e índice como
desempate, e conserva os estágios de ordenação e remapeamento
entre tiles. O fixture comparou os índices byte a byte em 12
geometrias, incluindo empates, infinito, uma cauda de uma coluna
e execução com CUDA Graphs. Os dois novos casos TOP_K passaram
no `test-backend-ops` da A100.

O A/B usou o mesmo drafter com cinco propostas, `-b 2048 -ub 1024`
e grade compacta em ambos os braços. Dois warmups curtos e três
amostras por caso, sem reuso e com o mesmo limite de saída.

| Métrica | Base | Radix | Delta |
|---|---:|---:|---:|
| Prefill 28k | 1.539,83 t/s | 1.581,23 t/s | +2,69% |
| Decode pós-4k | 91,36 t/s | 90,18 t/s | −1,29% |
| Decode pós-28k | 80,35 t/s | 89,89 t/s | +11,87% |
| Pico de VRAM | 39.512 MiB | 39.512 MiB | 0 |

As faixas de decode pós-28k foram 79,10–80,69 e 89,46–90,31 t/s.
Os hashes dos três prompts coincidiram com o controle sem draft.
No mesmo comparativo, a Strata registrou 109,70 t/s pós-28k
com hash igual. A diferença de decode caiu para 1,22×. O build
da rodada recompilou seletivamente as unidades CUDA na .57.
Medianas e amostras em
`/tmp/opencode/dflash-paired-24/verify-topk-summary.json`.

### Top-k radix de um tile

`GGML_CUDA_TOP_K_RADIX_ONE_TILE=1` estende o radix CUDA para
512 índices em linhas de 1.025 a 4.096 scores, com CUB e SM80
ou SM89. A flag fica desligada por padrão. A seleção radix
de dois tiles no SM80 continua automática.

Na RTX 4070, as rodadas desligado/ligado/ligado/desligado usaram
o mesmo binário SM89. Os índices foram idênticos byte a byte
nas 16 geometrias, incluindo empates, infinitos e padding. Os
tempos abaixo medem a operação GGML com seis queries e 1.157
scores por linha, incluindo submissão e sincronização.

| Execução | Desligada | Ligada | Razão |
|---|---:|---:|---:|
| Direta | 285,507 µs | 37,861 µs | 7,54× |
| CUDA Graphs | 289,200 µs | 36,922 µs | 7,83× |

O build SM80 foi compilado na .57. O A/B fim a fim do Flash-Next
na A100 está pendente. Para medir, usar o mesmo binário com flag
`0` e `1`, a grade compacta, DFlash2 com cinco propostas,
`p_min=0` e `-b 2048 -ub 1024`. Registrar 4k e 28k, hashes
dos tokens, faixas de throughput e pico de VRAM. As amostras da
4070 estão em `/tmp/opencode/topk-one-4070/shipping-summary.json`.

### Qwen3.6 na RTX 4070 com o main atual

BeeLlama `47b5c51bf` com recompilação seletiva no chroot da .57
e Strata v0.2.20/0.1.66 (`a34979fc9`). Mesmo Qwen3.6-35B-A3B
UD-IQ2_XXS, SHA-256
`627e2b05f83088448e27861ec392d56efc2b178783cae5adfa4ad2d988441203`.
Os dois engines usaram SM89, KV F16/F16, contexto 4.096, uma sequência,
os mesmos IDs de prompt, greedy, 64 tokens de saída, sem drafts e
sem reaproveitamento de prompt. O controle Bee reuniu duas rodadas
de três amostras por caso. Os demais braços tiveram três amostras.

| Braço | Prefill 3.584 | Prefill 3.584 (código) | Decode pós-3.584 | Pico (MiB) |
|---|---:|---:|---:|---:|
| BeeLlama, ubatch 512 | 3.368,93 | 3.036,89 | 121,40 | 11.551 |
| BeeLlama, ubatch 512, largura Ada IQ | 3.595,99 | 3.446,18 | 119,68 | 11.551 |
| BeeLlama, batch/ubatch 768, largura Ada IQ | 3.551,73 | 3.401,78 | 120,12 | 11.809 |
| Strata, chunk 512 e cache automático | 537,94 | 505,56 | 112,86 | 11.265 |

Valores em t/s. A Strata manteve 206 de 256 experts por camada na
GPU. O braço de 224 slots parou na verificação de espaço para os
buffers na inicialização, antes de gerar amostras. Em 3.584 tokens,
o prompt repetitivo teve hash igual nos dois engines. A saída do
prompt de código divergiu entre engines.

A flag `GGML_CUDA_MMQ_ADA_IQ_TYPICAL_MOE=1` seleciona a largura
usando duas vezes a média de linhas por expert em SM89, para
IQ2_XXS, IQ2_S, IQ3_XXS e IQ4_XS e ubatches de pelo menos
128 tokens. Em ubatch 512, ganhou **6,74%** no prompt repetitivo
e **13,48%** no de código, com hashes iguais ao controle nos
quatro casos. As faixas de pp3.584 foram 3.239,43–3.436,00
e 3.547,88–3.635,67 t/s. A flag é opt-in. O lote 768 mudou o
hash do prompt de código e ficou abaixo do 512 no prefill longo.

O Qwen3.6 exercita GDN e MoE. O Flash-Next usa QSA e outra
geometria de experts. As amostras e comandos desta rodada
estão em `/tmp/opencode/strata-ada-current/`.

### Projeção de saída em microlotes de prefill sem logits

O grafo Qwen4Exp agora encerra a construção depois de publicar a
ativação NextN quando o microlote pede zero logits e o contexto está
em modo de geração. As escritas de KV e GDN e os taps do DFlash2
continuam no grafo. A projeção F32 descartada tem 248.320 colunas de
vocabulário e ocupa 970 MiB em ub1024.

Na fixture CPU e na RTX 4070 com alvo e drafter em CUDA, o controle
projetou nove linhas e o candidato projetou só a linha pedida. Os
logits do último token do prompt e do token seguinte foram bit a bit
idênticos em cada backend. O teste cobre os 16 estados dos taps e do
NextN, além da passagem para o drafter. A matriz Qwen4Exp do CTest passou 5/5.

Na A100-SXM4-40GB, o A/B usou Q2_0 e o drafter DFlash2 pareado,
ambos verificados por SHA-256. Os dois braços usaram o mesmo
executável e os mesmos backends SM80, recompilados seletivamente
na .57, com a biblioteca de modelo como única variável
(`05ca46964` e `0092e660f`). A receita foi `-b 2048 -ub 1024
-ncmoe 2 --moe-cache-mib 2048 --parallel 1 --fit off -fa on
-ctk q8_0 -ctv q8_0`, DFlash2 n-max 6, `p_min=0`, controlador
desligado, GLU, SHARED_Q8, mapa e cache de prefill, e grade compacta.
Os prompts foram de 4.627 e 28.547 tokens, com temperatura 0, seed
42, até 64 tokens de saída, sem cache de prompt, dois warmups e
três rodadas por braço em ordem alternada.

| Métrica | Controle | Candidato |
|---|---:|---:|
| Prefill 4k | 1.539,30 t/s | 1.578,72 t/s |
| Prefill 28k | 1.578,07 t/s | 1.656,23 t/s |
| Decode pós-28k | 86,19 t/s | 87,12 t/s |
| Pico de VRAM | 39.622 MiB | 38.994 MiB |

O prefill 28k ganhou 4,95%. As faixas foram 1.574,87–1.578,10
e 1.647,58–1.656,23 t/s. O pico caiu 628 MiB. Todos os hashes de
saída coincidiram por caso, `aa09c78f0e31` em 4k e `1df1a3cd1b19`
em 28k. Em pós-4k, a primeira rodada de controle teve decode
atípico (39,31 t/s), com 65,78–65,85 t/s nas duas seguintes.
Amostras, linha de comando e trace no diretório
`/tmp/opencode/qwen4exp-no-output-sm80/`.

Com a mesma receita e `-ub 2048`, o controle falhou no primeiro
microlote com `cudaMalloc failed` ao pedir 2.175,09 MiB para o
grafo. O candidato completou três rodadas em 39.490 MiB de pico,
com medianas de **1.967,15 t/s** em prefill 28k
(1.965,25–1.967,83 t/s) e **79,09 t/s** no decode pós-28k.
Isso representa +24,66% de prefill comparado ao controle em 1024.
Ao mudar o tamanho do microlote, o hash de 28k mudou para
`de4e4381cd19` e foi estável nas três rodadas. O hash de 4k
coincidiu com o medido em 1024. O teste de paridade do patch
usa a geometria fixa de 1024.

Frente à Strata pareada da rodada anterior (4.943,72 t/s em
prefill 28k e 109,70 t/s no decode), a diferença de prefill
fica em 2,51× com ub2048 no BeeLlama. Para priorizar o decode
com esse drafter, ub1024 ficou à frente de ub2048.

### Prova CUDA do operando Q8_1 para a projeção down

Na RTX 4070, o empacotamento separado e o caminho que combina
SWiGLU e quantização produziram os mesmos 23.592.960 bytes por
layout (D4/Q2_0 e DS4/Q4_0), incluindo o padding. O fixture usou
2.048 tokens, top-10, 512 experts e 640 canais com padding até
1.024. Ele valida o roteamento com o helper de CUDA e compara
as saídas FP32 do `MUL_MAT_ID` down normal com as do kernel
MMQ alimentado pelo D4 pré-empacotado. Com largura J=32, todos os
52.428.800 valores foram idênticos (NMSE 0). O fixture é uma prova
de corretude, e a fusão no grafo e a medição com o caminho compacto
no SM80 continuam pendentes. Fonte e resultado em
`/tmp/opencode/mmq-glu-down-mma/` e
`/tmp/opencode/qwen4exp-no-output-4070/window-mma.log`.

### Receita DFlash2 n-max 5 depois da redução de workspace

Na mesma A100-40G, com o patch de saída e a receita acima, alteramos
a profundidade para cinco. Cada braço teve dois warmups de um token
e três amostras por caso. O radix de um tile e o protótipo HC 16
linhas foram medidos separadamente em ub1024, com o mesmo binário.

| Braço | Prefill 28k | Decode pós-4k | Decode pós-28k | Pico de VRAM |
|---|---:|---:|---:|---:|
| ub1024, flags de kernel desligadas | 1.653,36 t/s | 92,74 t/s | 91,84 t/s | 38.882 MiB |
| ub1024, radix de um tile | 1.693,21 t/s | 106,48 t/s | 92,77 t/s | 38.882 MiB |
| ub1024, HC 16 linhas | 1.646,91 t/s | 95,28 t/s | 93,86 t/s | 38.882 MiB |
| ub2048, flags de kernel desligadas | 1.970,72 t/s | 98,38 t/s | 86,61 t/s | 39.378 MiB |

No teste exclusivo de decode pós-4k, dois warmups completos de 64
tokens antecederam as três amostras de cada braço. O radix deu
**114,83 t/s** (114,12–115,29) contra **99,48 t/s** (98,61–99,54),
ganho de **15,43%**, com pico idêntico de 38.912 MiB e hash
`aa09c78f0e31`. Para decode, a receita medida usa `-ub 1024`,
DFlash2 n-max 5 e `GGML_CUDA_TOP_K_RADIX_ONE_TILE=1`. A flag
continua opt-in. O HC 16 linhas ganhou 2,74% no pós-4k e 2,20%
no pós-28k. Ficou abaixo do piso de 3% e foi arquivado fora do main.

Para prefill com DFlash2, ub2048 e n-max 5 entregaram 1.947,02–1.971,44
t/s em 28k. O hash em 28k foi estável nesse lote (`de4e4381cd19`).
Em ub1024, todos os braços preservaram o hash da Strata em 4k e
28k. A Strata pareada anterior mediu 4.943,72 / 109,70 t/s em 28k
(prefill/decode). Isso corresponde a 2,51× o prefill da receita
ub2048 do BeeLlama e 1,18× o decode da receita radix ub1024. No
prefill com hash igual ao da Strata, a diferença foi 2,92×.
As medições de kernel estão em
`/tmp/opencode/qwen4exp-no-output-sm80/kernel-results.tgz`.

### Down compacto com menor pressão de registradores

O trace do main no Flash-Next Q2_0 identificou 1.596 ms no down
compacto, com 256 threads, 255 registradores por thread e 56,5 KiB
de memória compartilhada por bloco. Isso limita o kernel a um bloco
residente por SM pelo orçamento de registradores.

`GGML_CUDA_MMQ_SM80_Q2_DOWN_K_OUTER=1`, junto de
`GGML_CUDA_MMQ_SM80_Q2_DOWN_COMPACT=1`, limita a vida útil dos
fragmentos a um subbloco K de cada vez e visa dois blocos residentes
por SM. A ordem de acumulação de cada elemento permanece a mesma.
A variante mantém o tile J128 e o K completo para o down Q2_0
640 → 2560 com 512 ou 1024 slots. O binário SM80 passou de 255
para 128 registradores e de 48 para 8 bytes de stack por thread.
O compilado com a nova ordem e o limite anterior de um bloco usou
167 registradores e ficou abaixo do piso de throughput.

Na A100-40G, com `-b 2048 -ub 2048`, DFlash2 n-max 5 e draft ub128,
o prefill de 28.547 tokens subiu de **1.947,85 para 2.010,14 t/s
(+3,20%)**. As faixas de três amostras foram 1.947,27–1.954,85 e
2.003,87–2.020,49 t/s. Os dois braços usaram o mesmo binário Linux
SM80 recompilado seletivamente na .57, dois warmups, temperatura
zero, seed 42, cache de prompt desligado e até 64 tokens de saída.
Os hashes de saída coincidiram (`de4e4381cd19`) e o pico foi
39.378 MiB nos dois braços. Os dados foram registrados em
`/tmp/opencode/strata-prefill-parity/strata-prefill-follow-up.json`.

O fixture com 512/1024 slots e rotas reais passou nos seis casos,
com os mesmos bytes em lançamento direto e em replay de CUDA
Graphs. O Nsight confirmou 672 lançamentos da especialização de
128 registradores, com os mesmos 56,5 KiB de memória compartilhada.
O tempo somado do kernel caiu de **1.596,02 para 1.138,49 ms**
(−28,67%). Ambos os traces usaram uma requisição de prefill de
28.547 tokens e um token de saída, fora dos números de throughput.

### Captura DFlash2 no mesmo dispositivo CUDA

O perfil de pp28k a ub2048 registrou 11,636 s nos grafos alvo.
Os experts e as projeções densas somaram 67,5% do tempo CUDA
instrumentado. Uma captura Nsight separada registrou 3,214 s
de lacunas entre atividades da GPU. Cerca de 1,695 s dessas
lacunas precediam cópias H2D de 6.553.600 bytes, tamanho de um
microlote de 128 linhas com cinco taps de largura 2.560.

`LLAMA_DFLASH2_PREFILL_D2D=1` mantém os taps no mesmo dispositivo
CUDA no prefill local Qwen4Exp DFlash2 com um slot. O drafter
continua usando 128 linhas por microlote. O buffer de captura
mantém os dados válidos até as cópias e a preparação do próximo
grafo. O modo opt-in preserva o caminho host quando o lote ou a
configuração não qualifica, ou quando a alocação de staging
comprometeria 512 MiB de espaço livre para o scratch CUDA.

Na A100-40G, a comparação usou o mesmo executável e os mesmos
pesos Q2_0 e DFlash2, n-max 5, `-b` igual a `-ub`,
`-ncmoe 2 --moe-cache-mib 2048`, KV Q8_0 e um slot. Os 28.547
IDs de prompt foram fixados, com temperatura zero, seed 42, até
64 tokens de saída, sem cache de prompt, dois warmups e três
amostras por braço.

| Microlote alvo | Prefill host | Prefill D2D | Pico de VRAM host → D2D |
|---|---:|---:|---:|
| 2.048 | 1.946,33 t/s | 2.237,59 t/s (+14,96%) | 39.378 → 39.478 MiB |
| 3.072 | 2.032,80 t/s | 2.343,65 t/s (+15,29%) | 39.872 → 40.022 MiB |
| 4.096 | 2.112,66 t/s | 2.111,00 t/s (fallback host) | 40.368 → 40.368 MiB |

Os hashes de saída coincidiram em cada geometria. A 3.072, as
faixas de host e D2D foram 2.032,40–2.039,41 e 2.342,30–2.349,68 t/s.
A primeira versão sem checagem de headroom sofreu OOM no pool
VMM com staging a ub4096. A versão com checagem passou nas três
rodadas dessa geometria com o pico do controle.

Em uma passagem de Nsight a ub2048, 222 cópias do tamanho de
um microlote do drafter passaram de H2D para D2D. O fixture com
dois taps distintos comparou o estado KV do drafter byte a byte
e passou na A100 e na 4070. Ele cobre fatias de 48+48+32 linhas,
leitura host sob demanda e rejeição de offsets inválidos.

A referência nova da Strata na mesma VM foi **4.932,87 t/s**
(4.932,02–4.933,21 t/s), com os mesmos pesos e IDs de entrada
e o drafter pareado. O prefill do BeeLlama a ub3072 com D2D
atingiu 47,5% desse throughput. O ganho do D2D usa hashes iguais
entre seus próprios braços. Dados brutos e capturas em
`/tmp/opencode/strata-prefill-parity/`.

### Receita combinada de prefill em 3.072

Com o D2D ligado em ambos os braços, adicionar
`GGML_CUDA_MMQ_SM80_Q2_DOWN_K_OUTER=1` e
`GGML_CUDA_TOP_K_RADIX_ONE_TILE=1` levou o pp28k de **2.341,91
para 2.487,18 t/s (+6,20%)**. As faixas de três amostras foram
2.341,63–2.348,17 e 2.485,82–2.492,36 t/s. Os seis hashes coincidiram
com `aa09c78f0e31`, e os dois braços tiveram pico de **40.022 MiB**.
A mediana de decode pós-28k da receita combinada foi 89,45 t/s.

A receita usa a A100-SXM4-40GB, pesos Flash-Next Q2_0 e o drafter
pareado, `-b 3072 -ub 3072 -c 32768 -ncmoe 2 --moe-cache-mib 2048
--parallel 1 --fit off -fa on -ctk q8_0 -ctv q8_0 --load-mode mmap
--lazy-mode on --no-warmup --spec-type draft-dflash --spec-draft-n-max 5
--spec-draft-ubatch-size 128 --spec-draft-p-min 0 --spec-dm-controller off`.
Nos dois braços, mapa CUDA, cache de prefill, GLU, SHARED_Q8,
compactação do down e D2D estavam ligados. A amostra usa 28.547
tokens de entrada, 64 de saída, seed 42, temperatura zero e cache
de prompt desligado, com dois warmups curtos antes das três amostras.

Com 2.487,18 t/s, o BeeLlama chegou a 50,4% dos 4.932,87 t/s
medidos na Strata na mesma A100. Essa comparação usa o mesmo
prompt de entrada. A igualdade de hashes acima refere-se aos
dois braços do BeeLlama.
`/tmp/opencode/strata-prefill-parity/approved-combo-results.tgz`
contém o JSON e os logs.

### D2D com taps do grafo alvo em ub4096

O DFlash2 agora usa os taps do grafo alvo durante a injeção
do draft. Os eventos CUDA mantêm a ordem entre produção,
cópia e reutilização do grafo. A fila do servidor conserva
os yields. Depois de encerrar a transferência, uma leitura
host ainda pode materializar os taps do último grafo válido.
O caminho misto sincroniza a cópia antes de um consumidor
em outro backend.

Na A100-40G, usamos a receita anterior, com `-b 4096 -ub 4096`
e draft ub128. Em pp28k, cada braço teve três amostras após dois
warmups. Os nove hashes coincidiram (`fec827d63cde`) e o pico
foi **40.368 MiB** nos três braços.

| Braço | Prefill 28k | Faixa de prefill |
|---|---:|---:|
| Host | 2.107,68 t/s | 2.105,07–2.116,06 |
| D2D com taps do grafo | 2.444,94 t/s | 2.443,86–2.445,33 |
| D2D com K-outer e radix | **2.597,40 t/s** | 2.596,68–2.599,76 |

O ganho isolado do D2D foi **16,00%**. O fixture comparou o
estado KV do drafter com a rota host em CUDA e no modo CPU/CUDA.
Ele verifica também a leitura dos dois taps depois do término
da transferência, com os buffers host antigos preenchidos
com zeros para detectar dados obsoletos. O trace do modelo
completo registrou 223 injeções do drafter pelo caminho device.
O comparativo da Strata na mesma A100 permanece em 4.932,87 t/s.
A receita mais rápida do BeeLlama chega a 52,7% dessa referência.
Os dados estão em
`/tmp/opencode/strata-prefill-parity/d2d-reviewed-results.tgz`.

### Perfil residual da receita de 2.597 t/s

Para os 28.547 tokens, as medianas sem profiler equivalem a 10,99 s
no BeeLlama e 5,79 s na Strata. Falta economizar 5,20 s. No Nsight
da receita combinada do BeeLlama, com um token de saída, o intervalo
entre o primeiro e o último kernel foi 11,35 s. A união dos intervalos
de kernels, cópias e memset ocupou 9,92 s e deixou 1,44 s ociosos.

O gate/up Q2_0 somou 1.703 ms, as projeções densas Q3_K 1.231 ms e
o down compacto Q2_0 869 ms. A atenção esparsa Q8_0 somou 648 ms,
o kernel GDN 556 ms e as projeções densas IQ4_XS 551 ms. Esses
valores são tempos instrumentados de famílias de kernels.

No A/B do tile denso, o prefill a ub2048 passou de 1.950,78 a
1.973,77 t/s (+1,18%). Com a receita combinada a ub4096, usar
J64/I64 no gate/up e manter o K-outer no down mediu 2.596,95 a
2.625,66 t/s (+1,11%). Ambos ficaram abaixo do piso de 3%.
A variante de ub2048 que troca o tile de ambos os kernels atingiu
2.036,38 t/s frente a 2.006,19 t/s com K-outer, +1,50% sobre a
opção já integrada. O arquivo de amostras e traces da sessão
é `/tmp/opencode/strata-prefill-parity/parity-a100-drained.tgz`.

### Precisão da projeção densa na primeira camada

O A/B de cuBLAS já medido na A100 ganhou 6,02% em pp28k e mudou
a saída greedy. Para localizar a diferença aritmética, usamos a matriz
IQ4_XS real de QKV da camada 0 (10.240 linhas de largura 2.560) e
os primeiros 96 IDs do prompt pareado. Os pesos vieram do GGUF fixado.
As entradas foram reconstruídas pelo grafo CPU de embeddings e
hiperconexões. A primeira camada com PLE nesse modelo é a camada 1.

Na RTX 4070, repetimos as entradas até completar um lote de 512 e
comparamos os 983.040 valores das 96 posições originais com o
produto FP32 dos pesos desquantizados e dessas mesmas entradas.
Todos os valores foram finitos.

| Cálculo na GPU | Erro RMS relativo | Erro absoluto máximo |
|---|---:|---:|
| MMQ de produção, IQ4_XS com ativações Q8_1 | 0,182267% | 0,0369883 |
| cuBLAS com operandos FP16, saída e acumulação FP32 | 0,011761% | 0,00569344 |

A projeção cuBLAS teve erro RMS relativo 15,5× menor nesse recorte.
Para avaliar a qualidade do modelo completo, o protocolo preparado
mede NLL, KL e concordância do token mais provável sobre os mesmos
tokens de continuação. Essa medição continua pendente. Dados e
método em `/tmp/opencode/strata-prefill-parity/dense-full-layer0-method.md`
e `real-layer0-cuda/4070.log` no mesmo diretório.

### Roteamento real da camada 0 em ub4096

Um recorte de 1.508.111.008 bytes preserva os 25 tensores da camada
0, os cinco tensores globais, o vocabulário completo, a recorrência,
as hiperconexões e os 512 experts com seleção top-10. As camadas 1–47,
a tabela PLE e as projeções QSA ficam fora desse modelo. A transferência
HTTP dos tensores somou 791 MB com os três pesos MoE reaproveitados
pelos hashes. Manifesto, hashes e smoke CPU em
`/media/hjotha/GAMES/flash-next-q2-layer0-20261124/`.

Na RTX 4070, um diagnóstico habilitou o kernel compilado para SM80
no dispositivo SM89. Com J128, a variante interrompe as iterações MMA
de gate/up de grupos de 16 colunas sem rotas válidas. No isolador com
pesos reais, ativações sintéticas e 2.048 tokens, o tempo do bloco MoE
caiu de 14,45 para 12,02 ms com roteamento enviesado e de 13,19 para
11,24 ms com roteamento uniforme. As saídas comparadas foram idênticas
em todas as seis combinações de tamanho e roteamento. O Nsight confirmou
o despacho do kernel experimental no isolador.

No servidor com o recorte da camada 0, pp28k, ub4096, temperatura zero,
sem reuso de prompt, dois warmups por processo e ordem off/on/on/off,
as seis amostras por braço deram medianas de 39.941,80 e 39.694,95 t/s
(−0,62%). Os intervalos se sobrepuseram, e o token de saída coincidiu.
O recorte fornece um teste do roteador real; o throughput agregado do
Flash-Next depende das outras 47 camadas. Os seis microlotes completos
produziram histogramas de top-10 que permitiriam eliminar 19,0–27,8%
das colunas MMA arredondadas na camada 0. O lote final expôs apenas
uma linha do roteador porque essa é a última camada do recorte.
A validação fim a fim no SM80 continua pendente. Amostras e traces
em `/tmp/opencode/strata-prefill-parity/ragged-4070/`.

### Flash-Next com quatro camadas na RTX 4070

O recorte preserva as camadas 0–3 do GGUF Q2_0, incluindo a primeira
PLE, a primeira QSA e os 2.048 experts. Ambos os engines carregaram os
mesmos dois shards e a tabela PLE de 26,82 GiB por mmap. A Strata foi
compilada para essa geometria a partir de `a34979fc9`, sem converter
os pesos. O manifesto, as formas e os hashes dos tensores estão em
`/media/hjotha/GAMES/flash-next-q2-layers0-3-20261124/manifest.json`.

Foram 28.547 IDs de entrada, contexto 32.768, uma sequência, um token
greedy de saída, sem draft ou reuso de prefixo. Os dois motores usaram
a RTX 4070, KV de 8 bits, chunks de 4.096 e dois warmups de prompt
inteiro por processo. A Strata manteve todos os experts na GPU e
reportou zero transferências de experts durante as amostras. Todos
os braços produziram o token 56553.

| Engine e variante | Prefill pp28k, mediana (t/s) |
|---|---:|
| BeeLlama, controle | 11.884,45 |
| BeeLlama, upload das linhas IQ4_NL e dequantização CUDA | 12.717,67 |
| BeeLlama, uma transposição da convolução PLE | 12.176,43 |
| BeeLlama, as duas variantes PLE | 13.026,38 |
| Strata, expert cache residente | 29.278,97 |

A amostra do controle reúne três repetições antes e três depois dos
candidatos. Cada outro braço tem três repetições. O ganho combinado
foi 9,61% no recorte. Além do token greedy, os 64 vetores de logits
completos no final do mesmo prompt foram idênticos entre controle e
PLE combinado (SHA-256 do arquivo `c2e6ba6657291dc03742d010bade68b5da51904498683726852f18c2d77c49d3`).
Uma das quatro camadas usa PLE; no modelo completo é uma entre 48.
A relação local de 2,25× entre Strata e o melhor braço BeeLlama
serve para diagnóstico. A qualificação de throughput requer o modelo
completo na A100. Scripts, comandos e amostras em
`/tmp/opencode/strata-prefill-parity/slice4/`.

No microbenchmark MoE independente, os engines receberam os mesmos
pesos da camada 0, os IDs top-10 capturados no pp28k e ativações e
pesos de roteamento sintéticos fixos. O escopo inclui agrupar as
rotas, gate/up, SwiGLU, down e combinar o resultado em FP32 por
token. Nas três rotas de 4.096 tokens, as medianas de nove chamadas
foram **17,24 / 16,19 / 16,23 ms** no BeeLlama com GLU e SHARED_Q8 e
**9,05 / 8,66 / 8,96 ms** nas entradas nativas da Strata. O
NMSE entre saídas variou de 7,87e-6 a 8,24e-6. O teste mantém os
pesos residentes e exclui carregamento e geração de tokens.
`/tmp/opencode/strata-native-vs-bee4070/fused-combine-0.log` contém
a geometria e os tempos alternados dos dois engines.

### Fusão dos quatro operadores MoE com os pesos Q2_0 do GGUF

Na RTX 4070, o protótipo leu diretamente os três tensores da camada 0
e aplicou o arredondamento das ativações e a ordem de acumulação do
BeeLlama. Com 4.096 tokens, dez experts por token, lotes residentes de
128 experts e as mesmas ativações sintéticas e rotas reais do ensaio
acima, foram sete chamadas alternadas por rota. Os tempos incluem a
redução ponderada até um vetor FP32 de 2.560 elementos por token.

| Rota capturada | BeeLlama, mediana (ms) | Kernel com os três tensores originais (ms) |
|---|---:|---:|
| 1 | 17,20 | 9,13 |
| 2 | 16,14 | 8,76 |
| 6 | 16,17 | 9,07 |

Cada rota produziu zero diferenças de bits entre os engines nos
104.857.600 valores intermediários por par e nos 10.485.760 valores
após a redução. O benchmark usa os pesos reais, incluindo 707.788.800
bytes de experts. A variante com três ponteiros elimina a necessidade
de duplicar esses pesos em um buffer intercalado. Os logs são
`/tmp/opencode/strata-native-beemath4070/gpu-result.log` e
`gpu-combine-result.log` no mesmo diretório.

O teste no servidor usa as quatro camadas, o prompt e a configuração
PLE combinada da seção anterior. Ambos os braços usam
`GGML_CUDA_DISABLE_GRAPHS=1`; a única troca é
`GGML_CUDA_STRATA_RAW_Q2=0/1`. O kernel escreve na posição original
de cada par token/expert para que o grafo execute a redução existente.
O log confirma ativação nas duas instâncias do candidato. Todos os
pedidos retornaram 56553. O comparador teacher-forced obteve o
mesmo SHA-256 dos 64 vetores finais com o kernel ligado e desligado,
`c2e6ba6657291dc03742d010bade68b5da51904498683726852f18c2d77c49d3`.
A primeira rodada teve controles entre 1.201 e 12.954 t/s com mmap
em disco USB. Essa dispersão impede usar sua mediana para atribuir ganho.
Os tempos individuais e os logs estão em
`/tmp/opencode/strata-prefill-parity/slice4-native/1795529480/`.

O segundo A/B fez uma leitura antecipada das 1.204 páginas da PLE e
das 65 páginas dos embeddings acessadas pelo prompt. O auxiliar usou
um mapeamento privado gravável para `mlock`. Uma inspeção posterior
de `/proc/<pid>/smaps` confirmou que as páginas bloqueadas eram
cópias anônimas. A retenção das páginas compartilhadas pelo servidor
durante essa rodada não foi demonstrada. A ordem foi controle, candidato,
candidato, controle, com dois warmups e três pedidos medidos por
processo. Os seis resultados do candidato ficaram entre 13.818 e
13.844 t/s; os controles variaram de 1.916 a 12.584 t/s. Mesmo o menor
resultado do candidato superou o maior controle nas duas rodadas,
mas a dispersão do controle persiste. Comparando o candidato à
referência local Strata de 29.278,97 t/s, o BeeLlama atinge cerca
de 47% do prefill nesse recorte. A A100 ainda precisa qualificar
a integração no modelo inteiro, incluindo o custo de desativar os
CUDA Graphs. A segunda rodada está em
`/tmp/opencode/strata-prefill-parity/slice4-native/1795529754/`.

#### Prefill com a política normal de CUDA Graphs

O protótipo local `d519f6a26` contém a receita para compilar o shim
SM80/SM89 a partir de `a34979fc9` da Strata. Repeti o teste dos
64 vetores com esse build na 4070. O log confirmou a fusão no
braço ligado e os dois arquivos de logits tiveram o mesmo SHA-256
registrado acima. Artefatos em
`/tmp/opencode/strata-prefill-parity/slice4-native/1795530032/`.

A revisão experimental `e4c15331b` também permite a fusão quando o
backend escolhe execução direta por conta própria. Com a variável
`GGML_CUDA_DISABLE_GRAPHS` ausente nos dois braços e o restante da
configuração mantido, os tempos do recorte foram:

| Braço | Mediana de seis prefill pp28k (t/s) | Mínimo–máximo (t/s) |
|---|---:|---:|
| Controle | 13.162,95 | 13.117,18–13.207,49 |
| Fusão durante execução direta | 13.780,54 | 13.656,46–13.817,10 |

O ganho foi de 4,69%. Os dois braços geraram o token 56553. Os eventos
por requisição confirmam a execução do kernel durante os lotes de
prefill. O caso tem uma saída e não mede a velocidade de decode.
Os dados e eventos de CUDA Graphs ficam em
`/tmp/opencode/strata-prefill-parity/slice4-native/1795530451/`.

#### Transferência dos embeddings de entrada

Com `GGML_SCHED_DEBUG=1 -lv 5`, o split CPU→CUDA identificou três
entradas de 40 MiB por ubatch cheio: `inp_mixed_embd`, o resultado
de GET_ROWS do ramo misto e `model.input_embed (reshaped)`. O prompt
contém apenas tokens de texto. O trace Nsight do Bee registrou 18
cópias de 41.943.040 bytes nos seis lotes de 4.096 e três de
40.663.040 bytes no lote final de 3.971. Juntas ocuparam 337,32 ms.
O tamanho do mask F16 de QSA cresce com o pool de contexto; os sete
uploads correspondentes somaram mais 90,69 ms. Os valores são da
4070 local e dependem da ligação PCIe com o host.

O protótipo `LLAMA_QWEN4EXP_TOKEN_ROWS=1` reúne as linhas Q3_K
originais na CPU e faz GET_ROWS na GPU. Os logs de atribuição mostram
um split CUDA e uma entrada de 4.505.600 bytes por ubatch cheio no
lugar dos três tensores, além de um índice de quatro bytes.
Mantivemos o kernel MoE, a PLE combinada e radix one-tile ligados
nos dois braços. O hash dos logits teacher-forced permaneceu idêntico.
O A/B com seis amostras por braço mediu 14.006,17 t/s no controle
(faixa 9.517,94–14.031,54) e 14.998,40 no candidato
(9.574,17–17.227,29). As faixas se sobrepõem. O teste confirma a
redução de tráfego e a corretude no recorte; a dispersão impede
promover com base nessa diferença de medianas.
Resultados em `/tmp/opencode/strata-prefill-parity/slice4-native/flat-1795530677/`.

#### Trace pareado após reduzir os uploads

A captura Nsight incluiu o segundo warmup. Um marcador interno
delimitou exclusivamente o terceiro pedido. Os dois traces cobrem sete
chunks QSA e 21 execuções GDN; o primeiro trace da Strata chegou a
apenas seis chunks e foi descartado para essa comparação. Os motores
devolveram 56553. Bee usou as linhas Q3_K compactas, radix one-tile,
PLE combinada e a fusão MoE do build `d519f6a26`.
Os números representam um pedido com profiler em cada engine.

| Recorte de quatro camadas, RTX 4070 | BeeLlama | Strata |
|---|---:|---:|
| Prefill informado pelo engine, ms | 1.666,45 | 972,40 |
| Intervalo total do pedido, ms | 1.675,81 | 975,55 |
| GPU com trabalho registrado (união dos intervalos), ms | 1.574,87 | 972,17 |
| Intervalos sem trabalho registrado na GPU, ms | 100,93 | 3,38 |
| Cópias CUDA e memset, soma em ms | 126,81 | 29,50 |
| Kernels identificados como FA/QSA, soma em ms | 261,01 | 78,40 |
| Kernels identificados como GDN, soma em ms | 104,11 | 77,49 |

O prefill equivale a 17.130 contra 29.357 t/s nesse pedido. A diferença
de tempo dos kernels FA/QSA foi 182,61 ms, seguida de cerca de 98 ms
de diferença nos intervalos sem trabalho na GPU e 97 ms nas cópias.
Os nomes genéricos dos kernels não permitem atribuir todo o tempo a
camadas. O experimento não estabelece o ganho no modelo de 48 camadas.
Os nomes, contagens, configurações, hashes e traces SQLite estão em
`/tmp/opencode/strata-prefill-parity/slice4-nsys/live-1795530891/`.

O A/B seguinte manteve as linhas Q3_K compactas nos dois braços e
variou apenas `LLAMA_QSA_INDEXER_BOUNDS`. O controle transporta a
máscara densa FP16; o candidato transporta dois limites I32 por token.
Na 4070, seis amostras por braço deram medianas de **17.280,85** e
**18.333,19 t/s**. O controle variou de 17.184,11 a 17.296,13 t/s.
Cinco amostras do candidato ficaram entre 18.286,80 e 18.376,00 t/s,
e uma caiu para 1.295,97 t/s. Todos os pedidos geraram 56553; o arquivo
de logits teacher-forced foi idêntico nos dois braços. O ganho local
ainda precisa de confirmação sem a amostra lenta e de medição com a
receita atual no modelo completo. Dados em
`/tmp/opencode/strata-prefill-parity/slice4-native/bounds-1795531092/`.

#### Cobertura da fusão roteada e seleção em dois tiles na RTX 4070

Na revisão experimental `5ca5d9e8e`, o kernel roteado também aceita a
ordem de grafo que coloca gate, up e SwiGLU do expert compartilhado
antes do down roteado. O down escreve na posição original do grafo.
O diagnóstico confirmou 13 fusões adicionais por pedido nas camadas
1 e 3. A camada final só calcula FFN para um token no último lote. Em
paralelo, `5efa09ba3` habilita radix CUDA para top-k de 512 com dois
tiles de 4.096 na Ada. Os 6.144 índices do fixture de empates,
zeros com sinal e infinitos foram idênticos entre os braços.

Mesmo recorte de quatro camadas e 28.547 IDs, 8-bit KV, sem draft ou
reuso, com o pack PLE, as linhas Q3_K e o mask por limites ativos.
A repetição corrigiu o aquecimento do cache do disco usando
`mlock` em arquivo mapeado somente para leitura. A sequência foi
controle, radix, combinado, combinado, controle. Cada processo teve
dois aquecimentos completos e três pedidos medidos.

| Braço | Amostras | Mediana (t/s) | Mínimo–máximo (t/s) |
|---|---:|---:|---:|
| Controle, fusão nas camadas 0 e 2 | 6 | 18.262,55 | 18.201,85–18.292,52 |
| + radix de dois tiles | 3 | 19.393,20 | 19.336,16–19.402,04 |
| + fusão nas camadas 1 e 3 | 6 | 20.666,67 | 20.611,09–20.742,42 |

O combinado ganhou **13,16%** sobre o controle. Todos os pedidos
retornaram 56553 e os 64 vetores de logits teacher-forced foram
idênticos entre o controle e o combinado. Um teste CUDA separado
validou o transporte Q3_K em 12 transições entre texto, embeddings,
entradas mistas e LoRA. O hash dos arquivos com os 12 vetores foi
`2765fd0a7affe4c99cc72f09ff69e239f8652f27ec49916414e048a3f00f6667`
com a opção ligada e desligada.

A captura Nsight pareada do combinado e da Strata cobriu os sete
chunks QSA e as 21 chamadas GDN de cada engine. O prefill foi de
**1.381,30 ms contra 977,60 ms**. Os kernels identificados como
MoE somaram 258,03 contra 251,14 ms, as cópias e memset, 35,84
contra 27,55 ms, e FA/QSA, 170,56 contra 78,97 ms. O intervalo
total sem atividade CUDA registrada foi de 90,66 contra 4,54 ms;
a métrica inclui trabalho do host. Esses tempos são de um pedido
instrumentado por engine, isolados das medianas acima. O núcleo
denso e os kernels genéricos ainda requerem atribuição por operação.

O A/B, os logits e os logs de cobertura estão em
`/tmp/opencode/strata-prefill-parity/slice4-native/combo-1795532659/`;
os traces e a tabela de atividades estão em
`/tmp/opencode/strata-prefill-parity/slice4-nsys/live-1795532843/`.
O teste de transições usou o código experimental `eacaac410` e
está em `slice4-native/bounds-1795531092/transition-{0,1}.{log,bin}`
sob a mesma raiz. A comparação das 48 camadas com DFlash2 na A100
continua necessária antes de promover a fusão roteada.
