# KV padrão na Radeon: medições de 2026-10-01

## Escopo e reprodução

Experimento na branch `experiment/radeon-q4-20261001`, base `fe9c62ed4`.
Modelo: `/home/hjotha/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf`.
SHA-256: `e23eb251493b8a89f7d0217590b593e696ac9bacaeac8605541e477533dc5838`.
Tamanho: 10.442.827.776 bytes.

Evidência bruta e backup de produção:
`/home/hjotha/beellama-radeon-q4-20261001-122139/`.
O backup `router-production.ini` tem SHA-256
`68606b35929819d9179cbf6344a5a972eaf54be4f4f2a784a0facf9634c764a9`.

O executor [bench-radeon-q4.py](../tools/bench-radeon-q4.py) foi desenvolvido
por Luna/max, revisado pelo principal e corrigido após testes reais. Cada
diretório de execução contém `run.json` (modelo, binário e bibliotecas realmente
mapeadas), `cases.jsonl` (argv exato e placement), `requests.jsonl` (entradas,
respostas e telemetria), `results.jsonl` e logs completos. O código de inferência
e os binários de produção não foram alterados neste experimento.

Hardware: RTX 4070 12 GB externa, Radeon integrada do Ryzen Z1 Extreme
(`Vulkan0`, RADV PHOENIX), 15.158 MiB de RAM física. Clocks e limites de potência
não foram alterados pelo teste; a Radeon estava no governador `auto`.
Os snapshots antes/depois não medem clocks médios nem picos durante o kernel.
As medições de cada configuração são exploratórias, sem intervalo de confiança.

## Modelo completo: todas as 16 atenções na Radeon

Pesos, projeções e estado recorrente na CUDA; atenção/KV em Vulkan0.
K e V usam o mesmo formato global. Um slot, MTP desligado, FlashAttention ligado,
batch/ubatch 256, `kv-tail-tokens=0`, temperatura 0, seed 42, 64 tokens de saída,
`ignore_eos=true`. O sufixo intrínseco de KVarN continua sendo parte do formato.
Não foi feita conversão KVarN→Qx nem implementado cache misto por dispositivo.

### 8.192 tokens realmente processados; capacidade configurada 204.800

Fonte: `formats-8k-ctx200k-v2/`. Quatro casos completos, zero falhas.

| KV na Radeon | Prefill, tok/s | Decode, tok/s | Buffer KV para capacidade 204.800, MiB |
|---|---:|---:|---:|
| Q4_0 | 168,89 | 20,35 | 3.600 |
| Q5_0 | 153,74 | 19,48 | 4.400 |
| Q6_0 | 186,15 | 14,88 | 5.200 |
| Q8_0 | 183,63 | 20,32 | 6.800 |

O workspace Vulkan reservado foi 126,27 MiB, além do KV. Todos os casos
confirmaram `Vulkan-layers=16/16` e `libggml-vulkan` carregada. Isso demonstra
alocação de capacidade 204.800 e inferência com 8K ocupados; não demonstra
throughput do modelo completo com 200K ocupados.

Após o request frio Q8, o snapshot registrou 7.146.627.072 bytes GTT usados no
dispositivo, 6.036.404 kB de memória disponível no sistema e 9.689 MiB usados na
4070. A RAM da Radeon é compartilhada; GTT, buffers Vulkan e memória do sistema
não são três reservas independentes para somar.

Nesta versão do executor, a continuação foi o restante do mesmo corpus.
Q4/Q6/Q8 reutilizaram também tokens coincidentes da geração anterior
(`cache_n=8255`, 65 tokens reavaliados); Q5 reutilizou 8.188 e reavaliou 132.
Essas respostas são válidas, mas suas latências incrementais não formam uma
comparação com trabalho igual. O executor foi corrigido para usar um sufixo
independente no ensaio seguinte. O prefill frio da tabela não é afetado.

### 16.384 tokens realmente processados; capacidade configurada 204.800

Fonte: `formats-16k-ctx200k/`. Dois casos completos, zero falhas.

| KV | Prefill frio, tok/s | Decode após prefill frio, tok/s | Prefill incremental de 132 tokens, ms | Decode incremental, tok/s |
|---|---:|---:|---:|---:|
| Q4_0 | 130,07 | 15,36 | 2.037,00 | 14,80 |
| Q8_0 | 143,85 | 15,88 | 1.699,75 | 15,21 |

O sufixo independente acrescentou 128 tokens. Ambos reutilizaram 16.380 tokens
e reavaliaram 132 devido ao ponto de restauração do estado recorrente.
A queda de decode em relação a 8K impede extrapolar aproximadamente 20 tok/s
para um histórico de 200K. Q8 direto não foi mais lento que Q4 neste ensaio,
mas exige consideravelmente mais memória e tráfego quando o histórico cresce.

## Comparativo inicial KVarN4 e diagnóstico de cauda

Os smokes de 2.048 tokens usaram outro prompt, capacidade 16.384, todas as
16 atenções Vulkan, 32 tokens gerados e os mesmos parâmetros de batch/sampling.

| Formato | Cauda adicional | Prefill, tok/s | Decode, tok/s |
|---|---:|---:|---:|
| KVarN4 | 0 | 83,92 | 13,06 |
| Q4_0 | 0 | 227,20 | 25,07 |
| Q4_0 | 1.024 F16 | 56,82 | 5,87 |

As três respostas textuais coincidiram; isso não é avaliação de qualidade.
A rota Vulkan de cauda explícita desvia da FlashAttention padrão e percorre
todo o KV em um shader próprio. Por instrução do usuário, os ensaios seguintes
não usam cauda adicional e este experimento não tenta otimizar aquele shader.

O ensaio KVarN4 de 8.192 tokens de documentação, ainda com capacidade 16.384,
foi interrompido pelo timeout de 360 segundos: último progresso de 7.424 tokens
em 359,07 s, média parcial de 20,68 tok/s. Não há decode concluído desse caso.
O Q4 com o mesmo prompt concluiu a 162,40/20,67 tok/s. A execução interrompida
está explicitamente marcada em `kvarn-8k-failure.json`.

## Controles CUDA e apenas uma camada remota

Fontes: `controls-8k-cuda-and-one-remote/` e
`controls-8k-q8-cuda-and-one-remote/`. Seis casos completos, zero falhas.
8.192 tokens reais, capacidade 16.384, 64 tokens gerados, MTP desligado,
sem cauda adicional, mesmo corpus natural da matriz principal. Cada linha
muda o tipo global; a comparação 0 versus 1 camada dentro de cada formato
isola o acréscimo da rota Vulkan.

| KV global | Prefill CUDA, tok/s | Prefill com 1 camada Radeon, tok/s | Decode CUDA, tok/s | Decode com 1 camada Radeon, tok/s |
|---|---:|---:|---:|---:|
| KVarN4 | 890,75 | 353,69 | 34,17 | 25,43 |
| Q4_0 | 995,54 | 780,06 | 34,60 | 32,07 |
| Q8_0 | 989,41 | 807,22 | 34,88 | 32,48 |

Com uma camada remota, Q4/Q8 reduziram substancialmente a penalidade observada
no KVarN Vulkan. O custo de decode do Q8 remoto foi aproximadamente 6,9%
neste cenário. Isso não valida ainda KVarN na CUDA com Q8 na Radeon, porque o
formato é global nestes testes, nem garante o mesmo custo a 100K/200K ocupados.

O cache também foi reutilizado nos casos que tinham atenção nas duas GPUs.
Com uma camada Q8 remota, foram reaproveitados 8.188 tokens e reavaliados 132;
o prefill incremental levou 440,934 ms e o decode mediu 32,81 tok/s. No KVarN,
a granularidade efetiva de restauração foi diferente: 8.064 tokens reaproveitados
e 256 reavaliados. Não comparar essas latências como quantidades iguais de
trabalho incremental.

## Atenção isolada com 100K/200K posições e controles AMD aplicados

Fonte: `micro-20w-2700/results.json`; executor
[bench-radeon-attention.cpp](../tools/bench-radeon-attention.cpp), desenvolvido
por Luna/max e revisado pelo principal. O binário foi compilado separadamente
e ligado às bibliotecas existentes; não houve rebuild do servidor de produção.

O usuário solicitou os controles AMD após a matriz do modelo e dispensou
repeti-la. Estes resultados usam outro perfil de energia: `--apu-tdp 20`,
`--amd-sclk-decode 2700`, via o mesmo `server_gpu_power` usado no servidor.
Cada execução confirmou pela tabela RyzenAdj os limites STAPM/fast/slow em
20/20/20 W e leu OD_SCLK mínimo/máximo em 2700/2700 MHz. Após cada caso, os
limites voltaram a 8/20/15 W e o governador voltou a `auto`, com OD 800–2700 MHz.
Os valores instantâneos e as tabelas completas estão no JSON; isso não mede
frequência média durante todas as instruções do kernel.

Cada caso representa uma camada: D=256, 24 query heads, 4 KV heads, uma query,
F32 accumulation, K/V sintéticos determinísticos não constantes e fisicamente
preenchidos, sem máscara nem tail. São 10 warmups e 30 computações medidas com
sincronização. A inicialização/quantização/upload dos dados não entra no tempo.
Os tensores são contíguos no layout de entrada da FlashAttention; o cache do
servidor usa views com strides derivados de linhas por token e heads. Portanto,
este microbenchmark não reproduz todos os detalhes de acesso da inferência real.
O resultado de atenção é conferido como finito e não nulo; isso não substitui
um teste de paridade numérica ou uma avaliação de qualidade de modelo.

| Formato | Atenção com 102.400 posições, mediana ms | Atenção com 204.800 posições, mediana ms | Payload KV/tempo a 204.800, GiB/s |
|---|---:|---:|---:|
| Q4_0 | 5,01094 | 9,73741 | 22,5652 |
| Q5_0 | 6,10492 | 11,9484 | 22,4763 |
| Q6_0 | 9,83701 | 19,3298 | 16,4193 |
| Q8_0 | 9,02685 | 17,8335 | 23,2730 |

As oito execuções concluíram e confirmaram aplicação/restauração dos controles.
Dobrar o KV aproximadamente dobra a latência. Q4/Q5/Q8 têm taxa aparente de
payload semelhante, enquanto Q6 fica abaixo. Isso é compatível com um custo
fortemente ligado ao volume de dados e à eficiência de acesso/decodificação;
**não é uma medição independente da largura de banda física da DRAM**. O tempo
inclui cálculo da atenção e despacho, e não conta explicitamente releituras.

Q8, que empatava com Q4 no decode do modelo a 8K, custa aproximadamente 1,83×
mais por atenção neste microbenchmark a 200K. Essa distinção favorece avaliar
Q4/Q5 como formatos remotos econômicos e usar Q8 como candidato de maior
precisão, sem declarar um vencedor de qualidade antes da conversão/KLD.

Estes números não são TPS do modelo a 100K/200K: não incluem FFN/projeções,
estado recorrente, transferências CUDA↔Vulkan, scheduler completo ou MTP.
Também não devem ser somados diretamente às latências do modelo medidas no
perfil de energia anterior para fabricar uma previsão de throughput.

## Validações e correções do executor

- Regressão numérica existente: `test-backend-ops test -b Vulkan0 -o FLASH_ATTN_EXT -p 'hsk=256,hsv=256,.*type_K=q4_0,type_V=q4_0'`: saída 0, 10/10 casos aprovados, incluindo GQA24/4 e cauda F16.
- Sintaxe Python, geração da matriz de cinco formatos e ordem alternada entre repetições aprovadas.
- Porta com listener ativo é recusada; timeout não positivo é recusado.
- Primeira matriz de capacidade teve HTTP 501 no erase por faltar `--slot-save-path`. Corrigido com diretório isolado por caso, nunca o cache de produção. Não usar aquela matriz como throughput.
- Checagem de porta corrigida para distinguir `TIME_WAIT` de listener ativo, usando `SO_REUSEADDR` sem `SO_REUSEPORT`.
- Contagem incorreta de tokens, truncamento, falha de cache hit ou placement não confirmado fazem o caso falhar.

## Próxima implementação

Ver [plano de implementação e avaliação](radeon-q4-experiment-plan.md).
O conversor existente faz Q4→KVarN. O caminho inverso ainda precisa ser criado
e validado: converter o prefixo antigo uma vez no handoff e gravar novos K/V
diretamente em Qx. A Radeon calcula a atenção do seu próprio KV e retorna
somente o resultado à CUDA. Q8 pode reduzir erro adicional de requantização,
mas não recuperar informação já perdida no KVarN4.


## Implementação mista: evidência intermediária de correção

Branch `experiment/radeon-q4-20261001`; artefatos de implementação em
`/home/hjotha/beellama-mixed-kv-20261001-130910/`. Esta seção distingue os
componentes já executados da integração de estado/handoff ainda pendente.

- O binário novo restaurou o snapshot KVarN puro produzido pelo binário
  anterior: `n_restored=896`, `n_read=77623459`, `cache_n=896`, mesmos tokens
  gerados. Evidência: `stage1-legacy-roundtrip/`.
- Com Qwen3.5-4B, os caminhos Q8 padrão e misto com todas as oito camadas de
  atenção remotas produziram tokens, texto e probabilidades top-16 idênticos
  em prompts de 127, 128, 129 e 385 tokens. O caminho com uma camada Q8 remota
  também executou os quatro prompts. Checkpoints estavam desativados;
  isso não valida restauração de estado misto. Evidência: `stage1-cold-probes/`.
- O controle adaptativo com o binário anterior e o modelo
  `Qwen3.5-4B-MTP-Q4_K_M.gguf` passou de capacidade 512 para 1024:
  prompt 300 frio; prompt 700 reutilizou 256 tokens e processou 444;
  repetição de 700 reutilizou 640 e processou 60. Foram aplicados 20 W e
  2700 MHz, com restauração dos controles registrada no log.
  Evidência: `baseline-adaptive-kvarn/`.
- O materializador em blocos executou na CUDA0, incluindo mapas físicos
  esparsos/reordenados e overlay exato sintético; as 17 cadeias de conversão
  comparadas tiveram `payload_match=yes`. Isso valida a peça de conversão,
  não a captura do tail de um cache vivo do servidor nem qualidade do modelo.
  Evidência: `opencode/conversion/principal-materializer-cuda.json`.
- A compilação integrada `stage2-planner-build-r3.json` terminou com código 0.
  Passaram parser de argumentos, conversor, planner (618 verificações), codec
  (275), encoder streaming (819), regressões KVarN CPU/CUDA e atenção nativa
  CPU/Vulkan. Evidência: `stage2-unit-results.json`.
- Com Qwen3.5-4B sem MTP, capacidade 65536 e ocupação de apenas 385 tokens,
  o auto-placement escolheu zero camadas remotas no orçamento normal:
  estimou 576,0 MiB de KVarN; a alocação registrou 576,03 MiB. Com reserva
  CUDA intencional de `8533M`, escolheu uma camada Q8 remota e sete KVarN
  locais: 504,0 MiB locais estimados e 136,0 MiB remotos estimados/alocados.
  Ambos processaram 385 tokens e geraram oito, com 20 W/2700 MHz.
  A tentativa inicial passou `8533` sem sufixo (bytes); foi preservada e
  excluída da validação de orçamento reduzido. Evidência: `stage2-auto-placement/`.

### Snapshots e MTP estático — stage 3

O build `stage3-state-build-r3.json` passou. No teste real Qwen3.5-4B,
Q4 e Q8 com uma camada remota salvaram/restauraram 896 tokens e continuaram
com `cache_n=896`, `prompt_n=160`, inclusive em **outro processo**. O runner
exigiu mesmos tokens/texto e cadência nas repetições do mesmo formato.
Os arquivos tiveram 75.965.822 bytes (Q4) e 76.883.326 bytes (Q8).
Evidência: `stage3-state-ac/run-r3/`, casos A. O probe independente
`stage3-direct-q8/` também passou.

MTP estático com target misto Q4/Q8 e draft KVarN4 na CUDA passou nos quatro
prefixos 127/128/129/385, com draft realmente utilizado, reuso nos pontos
exigidos e determinismo após reconstrução do mesmo histórico. Evidência:
`stage3-mtp-static/run/`, cenário M1, exit 0. Esses testes usam o modelo 4B e
não demonstram throughput do modelo 27B em contexto longo.

O caso C de arquivo truncado revelou um abort em
`llama_context::state_seq_load_file` (`GGML_ASSERT(nread <= state_size)`).
A correção da fronteira de leitura foi compilada e o caso C passou em
`stage4-state-ac/run/` (exit 0), junto com a repetição dos casos A Q4/Q8.
O servidor rejeitou o arquivo inválido e depois restaurou o válido e reutilizou
o prefixo. Os testes CPU de cabeçalho/tokens/corpo truncados também passaram;
a suíte mais ampla encontrou uma diferença de logits entre cadências de
prefill diferentes. O diagnóstico com o binário anterior e o atual mostrou
a mesma diferença frio-versus-incremental e igualdade exata entre execução
ao vivo e restauração com cadência 183+1. O teste foi corrigido para comparar
a mesma cadência, mantendo a tolerância 0,002, e passou integralmente em
`stage5-state-file-stream.json` (exit 0, 36,17 s).

Foi confirmado localmente que `/tmp` é tmpfs. Os testes de estado misto usam
`TMPDIR=/home/hjotha/beellama-mixed-kv-20261001-130910/spool`, em ext4 no NVMe,
para não criar outra cópia grande em RAM. O reader recusa tmpfs/ramfs.

### Migração efetiva no servidor — stage 5

A primeira proposta de hooks foi rejeitada e preservada em
`opencode/rejected-handoff-v1/`. A integração posterior compilou e passou nos
seguintes testes reais com Qwen3.5-4B-MTP, 20 W/2700 MHz:

| Caso | Prefixo pedido | Reutilizado na troca | Formato remoto | Resultado |
|---|---:|---:|---|---|
| `stage5-B-q4_0-r4` | 300 | 256 | Q4 | exit 0 |
| `stage5-B-q8_0` | 300 | 256 | Q8 | exit 0 |
| `stage5-M2-q4-r2` | 300 | 256 | Q4, MTP ativo após troca | exit 0 |
| `stage5-M2-q8` | 300 | 256 | Q8, MTP ativo após troca | exit 0 |
| `stage5-B-large-q4` | 4096 | 3968 | Q4 | exit 0 |

No caso maior o request de 5000 tokens reutilizou 3968 e processou 1032.
O conversor reportou 144,511 ms, scratch estimado de 1.347.584 bytes, cópias/fontes
comprimidas de 38.449.152 bytes e transferência recorrente de 52.691.540 bytes.
Scratch não é o pico de RAM: há também o snapshot já residente, a memória
CPU de origem e a serialização recorrente.

O destino preserva o prefixo do target; o draft MTP usa o bootstrap existente.
Não se afirma preservação integral do cache/carry do draft. O teste M2 inicial
falhou por consumo duplo do delta de logs pelo runner; as respostas já
mostravam reuso e draft real. A correção conservou o delta e a repetição passou.

A revisão/execução corrigiu identificação do proprietário KVarN no layout,
seleção de candidatos quando o cache genérico retorna `unchanged`, tratamento
do sink em stage slot zero, filtragem do histórico de rollback fora do tail
exato efetivo, validação de células densas e limites de leitura/transação.
O reader mistura cópias locais comprimidas com conversão remota em janelas;
não cria uma cópia F32 ou Qx do prefixo inteiro. Layouts físicos não densos
continuam explicitamente rejeitados nesta migração do servidor.

Nenhum teste acima demonstra TPS em 100k/200k tokens ocupados. A configuração
de produção permanece com o hash do backup original. A validação do modelo
27B foi atribuída ao OpenCode/DeepSeek V4 Flash max, com revisão do principal.


### Qwen 27B real — primeira matriz mista

Execução delegada ao OpenCode/DeepSeek V4 Flash max, revista pelo principal.
Artefatos: `phase3-27b/H-q4`, `H-q8`, `S123`, `S5`. Modelo
`Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf`, batch/ubatch 256,
um slot, sem tail adicional, 20 W/2700 MHz. S123 usa MTP com draft máximo 4;
H usa 2. O alvo e o draft local usam KVarN4.

H-Q4 e H-Q8 reutilizaram 256 tokens de uma entrada comum de 300, processando
444 no request de 700. MTP produziu drafts reais após a troca. A primeira
invocação H-Q4 concluiu o caso, mas o driver teve uma exceção ao agregar seu
resumo; o erro do driver foi corrigido antes de H-Q8. Não houve falha da
validação do caso H-Q4. S123 teve 10/10 casos completos, exit 0.

| Caso | Prefill frio (t/s) | Decode frio (t/s) | Decode repetição (t/s) | Cache repetição |
|---|---:|---:|---:|---:|
| S1-baseline-8192 | 817.44 | 70.65 | 54.15 | 8192 |
| S1-baseline-16384 | 800.74 | 62.22 | 32.59 | 16384 |
| S2-q4_0-8192 | 635.94 | 65.52 | 48.32 | 8192 |
| S2-q5_0-8192 | 637.50 | 63.51 | 47.65 | 8192 |
| S2-q6_0-8192 | 586.93 | 61.33 | 45.03 | 8192 |
| S2-q8_0-8192 | 653.24 | 45.05 | 39.36 | 8192 |
| S3-q4_0-16384 | 373.67 | 37.24 | 30.42 | 16384 |
| S3-q5_0-16384 | 366.70 | 56.06 | 31.14 | 16384 |
| S3-q6_0-16384 | 341.54 | 50.17 | 27.67 | 16384 |
| S3-q8_0-16384 | 381.04 | 54.56 | 25.77 | 16384 |

O decode tem somente 16 tokens por request e a aceitação MTP varia entre
casos: é uma medição curta deste corpus, não um ranking geral dos formatos.
S2/S3 têm uma camada remota. As capacidades diferem: controles de 8704/16896,
mistos de 16384/32768; ocupação de entrada 8192/16384. Repetições processam
32 tokens de sufixo e reutilizam 8192/16384. O comparador de probabilidades
considerou apenas o primeiro token, top-10 com IDs e cadência iguais; isso não
mede KL de vocabulário completo nem qualidade de conversão.

No probe de capacidade 204800, Q4 auto com MTP4 escolheu **14 camadas remotas
e 2 locais de atenção**. Reserva CUDA: 1618,9 MiB = 650 fixos + RS 748,1 +
MTP 220,8. KV remoto alocado: 3150 MiB; KV local: aproximadamente 441,5 MiB;
compute Vulkan: 126,27 MiB. O request ocupou somente 8240 posições. Prefill
frio 188,10 t/s e decode 40,89 t/s nesse request curto. Não extrapolar esse
TPS para 200k ocupados. Q8 foi **pulado pelo guard do runner**, com projeção
conservadora de todas as 16 camadas; não houve tentativa de alocação Q8 e
isso não prova que a opção seja impossível.

Os 14 casos somam 13 completos e um pulado. GTT e buffer Vulkan representam
parte da mesma RAM compartilhada: não somá-los como memórias independentes.
A validação de ocupação longa e a comparação de vocabulário completo após
handoff seguem separadas desta matriz.


### Fitter e RAM UMA — stage 6

O build `stage6-fit-uma-build.json` passou, assim como os testes CPU de sizing,
placement e reserva MTP (`stage6-unit-results.json`). O fitter agora está
ligado a `ctx-size=0` para o cache misto; usa custos por formato/camada e a
representação real do draft MTP, reserva RS e margens de workspace. O runtime
auto identifica o CUDA proprietário das camadas e aplica um teto de RAM
compartilhada `MemAvailable−512 MiB`. Múltiplos CUDA proprietários são
rejeitados. O teto runtime novo cobre a seleção automática; N manual continua
sob responsabilidade dos limites reais de alocação e dos controles do runner.

Testes reais `stage6-fit-gpu/` passaram com prefill de 9 tokens e decode de 8:

- 4B Q8, reserva CUDA intencional de `8000M`: capacidade 262144, cinco camadas
  remotas e três locais, concordando entre fitter e runtime.
- 27B Q4, MTP desligado, reserva CUDA `650M`: capacidade 262144, dez camadas
  remotas e seis locais, também concordantes. RS previsto 149,6 MiB; MTP zero.

Isso valida sizing, alocação e execução curta, não throughput com 262k
preenchidos. As reservas continuam conservadoras e não são uma medição de
todos os grafos futuros. `--fit off --ctx-size 0` mantém o fitting de tamanho
de contexto depois que o modelo foi carregado; o auto-fit geral de pesos é
um mecanismo distinto.

### Distribuição completa depois da migração — coleta válida r3

`phase3-quality/run-r3` teve cinco casos (controle e Q4/Q5/Q6/Q8), três
probes por caso, vocabulário completo de 248320 IDs. O warm-up usa 4096 tokens
e gera 16 tokens com viés de amostragem apenas nessa etapa para forçar a
continuação natural a divergir do estado live. Assim, controle e misto
restauram exatamente o mesmo checkpoint. Probes usam entrada natural
congelada e `n_predict=1`, sem esse viés, probabilidades antes da amostragem.
As cadências foram iguais em todo o histórico: `(3968,1032,1)`,
`(5000,500,1)` e `(5500,500,1)`, nos prompts de 5000, 5500 e 6000 tokens.

A primeira tentativa é inválida para comparação numérica: controle reutilizou
4096 contra 3968 do misto. Todos os números posteriores desse histórico também
ficam confundidos. A proposta de aceitar diferença de 128 foi rejeitada pelo
principal. A segunda tentativa já igualou a cadência, mas falhou no filtro do
marcador do runner; o marcador existia no log bruto. A terceira corrigiu apenas
a leitura desse marcador e terminou com exit 0. Nenhuma tolerância numérica ou
critério de cadência foi relaxado.

KL abaixo usa vetores completos normalizados separadamente em float64 para
remover o resíduo de massa do softmax/serialização. Evidência derivada:
`opencode/quality-normalized-r3.json`; os vetores e respostas brutas estão no
run-r3. Massas antes da normalização: aproximadamente 1,0000003 a 1,0002663.

| Formato | KL em 5000 | KL em 5500 | KL em 6000 | TV em 6000 |
|---|---:|---:|---:|---:|
| q4_0 | 0.00001167 | 0.019691 | 0.090050 | 0.133558 |
| q5_0 | 0.00000626 | 0.017857 | 0.518509 | 0.197839 |
| q6_0 | 0.00000749 | 0.018497 | 0.466965 | 0.219229 |
| q8_0 | 0.00002848 | 0.019639 | 0.358695 | 0.185570 |

Os IDs greedy coincidiram em todos os pontos, embora a distribuição tenha
mudado materialmente no terceiro. Q8 não mostrou melhora consistente de
proximidade ao controle KVarN4 nesses três pontos. A métrica compara a
continuação mista completa após o handoff: inclui backend diferente e tokens
novos diretamente quantizados em Qx, não somente o erro adicional de conversão
de um tensor congelado. Exit 0 significa medição válida; não há limiar que
certifique qualidade geral, perplexidade de corpus ou qualidade em 100k/200k.

### Diagnóstico de desempenho: capacidade e scratch CUDA

A comparação S1/S3 em 16k usou capacidades diferentes. Nos componentes
registrados de KV alvo, KV draft e compute, S3 mantinha aproximadamente
277,77 MiB adicionais na CUDA; o RS era 748,12 MiB nos dois. Existem mensagens
de redução de chunk KVarN por baixa VRAM. No Q4 S3 elas aparecem apenas nos
dois blocos finais de 128 tokens; não explicam sozinhas o tempo adicional já
acumulado. O código também pode abandonar silenciosamente a rota windowed
quando não cabe um chunk de 256. Portanto, a pressão CUDA é um confundidor
plausível e o excesso inteiro não pode ser atribuído à Radeon sem controle
com capacidades iguais. Isso é distinto da constatação de que o prefill remoto
faz trabalho adicional crescente com o histórico.

Durante o teste longo, sysfs registrou a 4070 em PCIe 8,0 GT/s ×4; o máximo
do dispositivo é 16,0 GT/s ×16. Evidência: `pcie-link-during-long.json`.
Isso identifica o enlace negociado, não mede banda útil nem latência de cópia.


## Fechamento da rodada — implementação e limites medidos

### Controle de capacidade e ajuste de scratch CUDA

O controle final usou capacidade **32768**, prompt **16384**, batch/ubatch
256, MTP máximo 4 e os mesmos inputs em ambos os caminhos. O processo de teste
aplicou 20 W/2700 MHz e restaurou os controles ao encerrar. Artefatos:
`phase3-same-capacity/` e `phase3-window4096/`.

| Configuração | Prefill frio t/s | Decode frio t/s | Prefill repetição t/s | Decode repetição t/s |
|---|---:|---:|---:|---:|
| CUDA KVarN4, chunk padrão | 801,28 | 62,98 | 55,1 | 30,31 |
| Uma camada Q4 remota, chunk padrão | 377,69 | 37,86 | 34,5 | 29,47 |
| CUDA KVarN4, chunk 4096 | 810,10 | 68,37 | 55,41 | 36,76 |
| Uma camada Q4 remota, chunk 4096 | 623,71 | 58,48 | 35,82 | 25,97 |

O ajuste é a variável existente `GGML_KVARN_WINDOW_CHUNK=4096`, aplicada
somente aos processos de teste. No caso misto, o pico CUDA amostrado caiu de
11875 para 11503 MiB. O parser correto usa **Q[1]** como número de queries;
Q[0]=256 é a dimensão do head. Com o chunk padrão, houve 497 dispatches
windowed Q256 no misto, limitados a KV8192; com 4096, foram 1008 até KV16128.
O controle CUDA teve 1071 dispatches Q256. Isso sustenta a hipótese de pressão
do scratch/pool CUDA contribuindo para o degrau de desempenho. Não isola
banda PCIe, latência, kernels Vulkan e demais custos de scheduling.

O decode mede somente 16 tokens por request. A aceitação MTP varia; no cold
com chunk4096 foram 12 drafts aceitos de 12 nos dois casos, mas os repeats
não tiveram a mesma aceitação. Portanto, o ganho de prefill não implica ganho
uniforme de decode. A comparação top-10 não comprova qualidade inalterada;
o teste numérico completo anterior usou o chunk padrão.

### Ocupação longa e memória

MTP desligado, auto-placement e tail adicional zero, em `phase3-long/`:

- 32768 tokens, N remoto zero: controle CUDA 809,93 t/s de prefill;
  configuração mista sem camadas remotas 796,05 t/s, comportamento esperado.
- **98304 tokens realmente preenchidos**, capacidade 102400, uma camada Q4
  remota e 15 locais: **403,24 t/s de prefill e 17,69 t/s de decode frio**.
  A repetição reutilizou 98304 tokens e processou 32 em cerca de 5,92 s;
  decode da repetição 18,19 t/s.
- Capacidade 204800 com nove camadas Q8 remotas coube e executou 8192 tokens
  de entrada. GTT amostrado máximo 4027113472 bytes (3,75 GiB); mínimo de
  MemAvailable 8781992 KiB (8,38 GiB). Isso é alocação de capacidade, não
  validação de 200k tokens preenchidos.

Dois testes tentaram entrada de **200000 tokens**, capacidade 204800, nove
camadas Q4 remotas e sete locais. Ambos atingiram o limite de 600 s sem terminar
prefill; o servidor encerrou normalmente, com exit 0 e controles restaurados:

| Scratch CUDA | Último progresso registrado | Tempo registrado | Taxa acumulada registrada |
|---|---:|---:|---:|
| Padrão | 61696 tokens | 596,97 s | 103,35 t/s |
| 4096 | 62208 tokens | 598,00 s | 104,03 t/s |

O segundo está em `phase3-long-tuned/`. Não há medição de decode com 200k
preenchidos. O ajuste recuperou o caso de 16k, mas não resolveu o prefill frio
com muitas camadas remotas. Não extrapolar tempo total linearmente dessa taxa
acumulada, pois a janela de atenção cresce ao longo do prefill.

O runner inicialmente exigia evidência MTP mesmo com `spec-type=none`.
Essa checagem foi corrigida e os resultados brutos foram revalidados em
`opencode/long-principal-revalidation.json`, sem sobrescrever logs: os casos
32k, 98k e Q8 de capacidade têm dados completos; o de 200k permanece timeout.
Alguns wrappers do OpenCode filtraram stdout em pipeline e reportaram exit 0
do filtro; isso não significa aprovação do benchmark. Os outcomes por caso,
os contadores das respostas e o log bruto são a evidência usada aqui.

### Estado final e validação

- Build final: `stage7-final-build.json`, exit 0.
- Fitter com reserva impossível de 20 GiB: falha controlada com exit 1,
  sem criar cache nem cair silenciosamente no tamanho de treino; teste passou
  (`stage7-fit-negative.json`).
- Snapshots Q4/Q8, restore em processo novo, rejeição de arquivo inválido e
  recuperação: `stage7-state-ac/parent-result.json`, exit 0.
- Estado em arquivo CPU: `stage5-state-file-stream.json`, exit 0.
- Helpers de placement/fitter/MTP: `stage6-unit-results.json`, todos exit 0.
- Runner de contexto longo: 33 self-checks, zero falhas; harness MTP: 52 testes;
  harness de qualidade: checks CPU e execução simulada completa aprovados.

A configuração original de produção foi preservada e o serviço ficou parado
durante toda a rodada. O recurso continua opt-in. O draft MTP usa bootstrap
após o handoff; não se promete preservação integral de seu cache/carry. Cache
misto paginado/unificado, múltiplos slots, SWA, repatriação e layouts físicos
não densos não fazem parte do contrato desta implementação. A seleção manual
de N não usa a nova checagem global de RAM do auto-placement.

Próximos trabalhos de desempenho: profiling por operação da atenção remota e
do scratch CUDA; otimização do prefill com muitas camadas remotas; avaliação
numérica em corpus maior. O objetivo de 200k frios com baixa penalidade de TPS
**não foi atingido** nesta máquina com o caminho testado.
