# MTP + DFlash2: uso ativo dos sufixos — 2026-09-29

## Implementação do piloto

O despacho especulativo pode consumir um resultado auxiliar já concluído, antes
de gerar o próximo MTP. Confere epoch, posição, âncora, prefixo incluindo o bônus
real, prazo e tamanho mínimo. O sufixo passa pelo sampler/verificador existente.
Cada bloco é consumido uma vez; a rodada promovida não inicia outro auxiliar.
Resultado ausente, atrasado, incompatível ou insuficiente segue por MTP sem
esperar pelo worker.

O MTP continua executando `process()` após o target e `accept()` após a aceitação
definitiva, inclusive zero aceitos. A origem externa sobrevive ao replay do
checkpoint. Cancelamento/reset descartam a seleção. O piloto confere o carry/KV
MTP antes de gerar e o carry depois da aceitação externa; inconsistência gera
erro. A contagem de aceitação DFlash compara os tokens originais com o resultado
final, excluindo o bônus de correção mesmo quando ele foi redecodificado.

Configuração experimental por ambiente:

| Variável | Função |
| --- | --- |
| `GGML_DFLASH_SHADOW_ACTIVE=1` | Habilita o uso real; padrão 0 mantém observação. |
| `GGML_DFLASH_SHADOW_MTP_MAX=2` | Limita a geração MTP efetiva; 0 preserva o limite configurado. Funciona também sem auxiliar para controle. |
| `GGML_DFLASH_SHADOW_MIN_SUFFIX=1` | Mínimo após aplicar a capacidade do verificador. |
| `GGML_DFLASH_SHADOW_EVERY=1` | Cadência por tentativa MTP; rodadas promovidas não incrementam esse contador. Zero é somente sincronização. |
| `GGML_DFLASH_SHADOW_EARLY=1` | Inicia o worker antes de gerar MTP. |

`--spec-draft-n-max 4` continua dimensionando o verificador, estados e buffers
antes da criação dos contextos. O override reduz o loop MTP de fato, preservando
essa capacidade de quatro. Não se geram quatro tokens MTP para depois cortar dois.

Escopo: um slot textual, MTP Qwen3.5-family com um head e KV próprio (inclui o
Qwen3.8 usado), temperatura ≤0 para consumir sufixos, sem aceitação sintética.
Com temperatura positiva, o auxiliar não lança/consome propostas. Flags não
promovem este experimento a modo de produção.

## Coleta

Alvo `Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf` na RTX 4070 eGPU; auxiliar
`Qwen3.8-27B-DFlash2-Q4_K_M.gguf` na Radeon/Vulkan0, sete propostas. Contexto
16384, batch/ubatch 64, KV q4_0, 192 tokens por caso, greedy, seed 42, cache frio.
APU 20 W, SCLK solicitado 2700 MHz; CPU powersave/EPP power, máximo 3301 MHz.

```bash
python3 docs/benchmarks/mtp-dflash2-hybrid-active-20260929/runner.py smoke --run-id smoke-v1
python3 docs/benchmarks/mtp-dflash2-hybrid-active-20260929/analyze_active.py docs/benchmarks/mtp-dflash2-hybrid-active-20260929/smoke/smoke-v1
```

O smoke compara MTP n4, MTP n2 com capacidade4, observador n2/cap4, híbrido
n2/cap4 EVERY1 e EVERY2 e híbrido n4/cap4 EVERY4 com mínimo3. A fase `paired`
aceita `--modes`, exige cinco repetições por caso e alterna grupos após aquecimento.
Preserva comandos, fontes/diff, hashes de bibliotecas/modelos, IDs de tokens,
primeiras divergências, logs completos, políticas CPU, APERF/MPERF e telemetria.
Usa porta 59595 e a trava comum de GPU; serviços e CPU são restaurados em `finally`.

`analyze_active.py` confere a trajetória emitida contra as decisões definitivas,
audita cada seleção contra o job de origem, conta aceitação total/parcial/zero e
retomada do MTP, e compara tok/s pareados. Divergência entre trajetórias com
tamanhos de verificação diferentes é registrada, sem presumir sua causa ou
declarar igualdade gulosa a partir desses números.

## Triagem ativa: `smoke/smoke-v1`

18 pedidos medidos, seis aquecimentos, uma execução por caso/modo. O consumo
ativo ocorreu de fato, com aceitação zero/parcial/total e retomada MTP auditadas.

| Modo | Repetição tok/s | Código tok/s | Longo tok/s |
| --- | ---: | ---: | ---: |
| MTP n4 / capacidade4 | 86,75 | 60,34 | 58,51 |
| MTP n2 / capacidade4 | 67,58 | 57,25 | 51,24 |
| Observação n2 / capacidade4 EVERY4 | 62,22 | 52,83 | 47,31 |
| Ativo n2 / capacidade4 EVERY1 | 75,49 | 52,53 | 48,49 |
| Ativo n2 / capacidade4 EVERY2 | 68,74 | 55,55 | 46,70 |
| Ativo n4 / capacidade4 EVERY4, mínimo3 | 83,24 | 55,37 | 56,28 |

Nos três pedidos medidos de cada variante ativa:

| Variante | Rodadas DFlash | Propostos | Aceitos reais | Zero / parcial / total | Retomadas MTP observadas |
| --- | ---: | ---: | ---: | --- | ---: |
| n2 EVERY1 | 62 | 248 | 196 | 6 / 13 / 43 | 60 |
| n2 EVERY2 | 42 | 166 | 128 | 5 / 9 / 28 | 40 |
| n4 EVERY4 mínimo3 | 12 | 48 | 25 | 2 / 6 / 4 | 12 |

A diferença entre rodadas promovidas e retomadas conta rodadas no fim de uma
resposta, quando não há novo draft. Nenhuma seleção ficou pendente, nenhum job
foi consumido duas vezes e nenhum erro de worker/carry/KV foi detectado.

Exemplo concreto: em repetição, o n2 EVERY1 aceitou todos os 92 tokens auxiliares
das suas 23 rodadas DFlash. O dispatch total de propostas caiu de 427,172 ms
(MTP n4) para 185,961 ms, mas as verificações especulativas passaram de 41 para
49; a geração total subiu de 2201,649 ms para 2530,181 ms. A economia de draft
foi real e insuficiente para compensar mais verificações e o custo auxiliar.

O piloto n2 confirma ganho contra MTP n2 em repetição (+11,7%), sem superar
a referência MTP n4 (−13,0%). A variante n4/mínimo3 foi a melhor no conjunto
da triagem e seguiu para confirmação, junto do n2 EVERY1:

```bash
python3 docs/benchmarks/mtp-dflash2-hybrid-active-20260929/runner.py paired --run-id paired-v1 --modes mtp4-v4 active-mtp2-v4-e1 active-mtp4-v4-e4-min3
```

Há diferenças de tokens entre alguns modos, registradas em
`smoke/smoke-v1/token-comparisons.jsonl`. O n2 isolado já difere do n4 nos
prompts curtos; o ativo também pode mudar a trajetória quando altera os batches
verificados. A auditoria confirma que os tokens saíram das decisões do target;
isso não estabelece equivalência numérica entre layouts nem explica a causa
da primeira divergência. O prompt longo desta triagem coincidiu nos seis modos.

## Confirmação: `paired/paired-v1`

45 pedidos medidos: cinco repetições por cenário e modo, com nove aquecimentos
e ordem dos grupos alternada. **Todas as 30 comparações híbrido/MTP n4 ficaram
abaixo do controle. Não houve ganho de throughput nesta configuração.**

| Modo | Repetição mediana [mín–máx] tok/s | Código mediana [mín–máx] tok/s | Longo mediana [mín–máx] tok/s |
| --- | --- | --- | --- |
| MTP n4 | 86,221 [85,794–86,430] | 60,170 [59,995–60,206] | 58,383 [58,172–58,772] |
| Ativo n2/cap4 EVERY1 | 74,978 [71,495–76,766] | 50,861 [49,513–52,586] | 47,479 [45,674–48,614] |
| Ativo n4/cap4 EVERY4 mínimo3 | 82,374 [81,558–83,375] | 55,862 [55,562–58,088] | 55,789 [55,201–56,338] |

A mediana das diferenças **pareadas por repetição** foi −13,04%/−15,52%/−19,22%
para n2 EVERY1 e −4,69%/−7,18%/−4,17% para n4 EVERY4 mínimo3, respectivamente
repetição/código/longo. Não confundir essas estatísticas com a razão de duas
medianas, que pode ter valor ligeiramente diferente.

| Variante | Rodadas DFlash | Propostos | Aceitos reais | Zero / parcial / total | Retomadas MTP |
| --- | ---: | ---: | ---: | --- | ---: |
| n2 EVERY1 | 311 | 1238 | 942 (76,1%) | 28 / 88 / 195 | 305 |
| n4 EVERY4 mínimo3 | 48 | 191 | 92 (48,2%) | 7 / 29 / 12 | 48 |

O n2 EVERY1 acertou 453/459 propostas auxiliares no cenário repetitivo (98,7%),
mesmo assim ficou abaixo do MTP n4. O mecanismo funciona como reaproveitamento;
essa taxa não compensa automaticamente o custo de manter o auxiliar e a menor
quantidade de tokens confirmados por rodada quando se encurta o MTP.

A soma do tempo de parede dos 15 pedidos foi 147,503 s no MTP n4, 163,887 s no
híbrido n2 e 157,974 s no híbrido n4. Inclui prefill e geração, exclui carregamento
dos modelos/aquecimentos. Tampouco houve vantagem no tempo completo de requisição.
SCLK médio ativo observado: 2700 MHz nas duas variantes; APU solicitada em 20 W.

### Limites de correção e interpretação

- Auditoria de todos os 54 pedidos desta bateria (45 medidos + 9 aquecimentos)
  passou: tokens retornados conferidos contra decisões definitivas do target,
  origem/prefixo/prazo/capacidade de cada sufixo conferidos e sem dupla utilização.
  As checagens de carry/KV MTP e de worker não detectaram erro.
- Não foi estabelecida equivalência gulosa integral. O híbrido n4 produziu IDs
  idênticos ao controle em 13/15 pares; o n2 em 7/15. As primeiras divergências
  estão em `token-comparisons.jsonl`; sua causa não foi determinada nesta etapa.
- **Mesmo restringindo aos pares de saída exatamente igual**, todas as diferenças
  são negativas: n4 entre −7,66% e −3,24% (13 pares), n2 entre −19,86% e −10,98%
  (7 pares). A conclusão de falta de ganho não depende somente dos casos divergentes.
- Este piloto não valida produção/múltiplos slots/amostragem estocástica nem toda
  a matriz de restauração/cancelamento. O prazo do worker depende do escalonamento,
  portanto as decisões híbridas podem variar entre repetições.

### Decisão

**Manter MTP n4 isolado para obter mais tok/s neste hardware e configuração.**
Etapa 2 executada com resultado negativo de desempenho. O código fica como
experimento opt-in, desligado por padrão. Esses testes não demonstram
impossibilidade para qualquer combinação futura de hardware/modelos/políticas;
demonstram que as políticas implementadas e medidas aqui não vencem o MTP n4.

Produção restaurada e saudável em 8090/8092. As 16 políticas CPU coincidem com
o estado anterior; Radeon voltou a `auto`, faixa SCLK 800–2700 MHz. Conferência
em `paired/paired-v1/restoration-verified.json`; logs registram a restauração dos
limites APU 8/20/15 W após cada processo de teste.

Validação de código: build incremental, quatro checks CTest (observador,
candidato de pipeline, upstream DFlash e posicionamento do seletor), dois testes
Python do auditor ativo e revisão independente do fluxo de origem/replay/accept.
Os snapshots/hashes de cada bateria identificam os binários realmente medidos.
A revisão final de ajuda/comentários da CLI foi feita depois da bateria e não
altera o caminho de inferência.
