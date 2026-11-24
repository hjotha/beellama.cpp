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
