# Continuação densa e valor do sufixo — 2026-09-29

## Método

O servidor registra `SHADOWTRACE` somente quando
`GGML_DFLASH_SHADOW_TRACE` está definido. Os eventos incluem início/fim do
pedido, posição/âncora e tempo de cada draft, amostragem sem proposta e aceitação
definitiva (depois de eventual replay). O registro não depende do throttle do
auxiliar. O endpoint também retorna os IDs exatos com `return_tokens: true`.

`analyze.py` mantém um mapa separado por **arquivo, pedido e slot**. Confere
todos os tokens retornados contra as decisões definitivas do servidor; conflitos,
eventos ausentes e posições repetidas de draft/commit causam erro. Este experimento
é textual, greedy, um slot e sem deslocamento de contexto.

Para um bloco pronto e compatível, com `c` tokens confirmados incluindo o bônus:

1. A próxima âncora é `pos0+c`; o sufixo começa em `proposed[c]`.
2. Compara-se o sufixo com os tokens realmente emitidos após essa âncora.
3. Uma lacuna ou fim de resposta censura a medida; não conta como rejeição.
4. O avanço estimado `L+1` é comparado com o avanço da **próxima** rodada MTP,
   incluindo seu bônus, não com `c` da rodada de origem.
5. Aquecimento fica fora das agregações principais. Cada bloco é avaliado uma vez.

Essa comparação é offline e não fornece informação futura ao decodificador.
É compatibilidade com a trajetória de referência, não aceitação real nem ganho
de velocidade de um híbrido. A soma de avanços é um diagnóstico por oportunidade,
não simula uma sequência de decisões híbridas: rodadas menores mudariam as âncoras
seguintes. O tempo `next_draft_us` inclui todo o dispatch do draft e pode incluir
contabilidade/injeção auxiliar; não é uma medida isolada do kernel MTP.

## Configuração e reprodução

- Branch `feat/dflash2-shadow-observation`, base `3d4128354` mais diff preservado.
- Alvo `Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf`, CUDA0/RTX 4070 eGPU.
- Auxiliar `Qwen3.8-27B-DFlash2-Q4_K_M.gguf`, Vulkan0/Radeon, sete propostas.
- Contexto 16384, batch/ubatch 64, KV `q4_0`, um slot, temperatura 0, seed 42.
- APU 20 W; SCLK solicitado 2700 MHz; CPU powersave/EPP power, máximo 3301 MHz.
- MTP limites 2/3/4, `p_min=0.70`; observador EVERY=4/EARLY=1.
- MTP isolado contemporâneo para cada limite; MTP n4 repetido ao final.
- Dois pedidos de repetição e código, um longo de 13850 tokens, 192 tokens de
  saída por pedido; um aquecimento de 64 tokens por modo.
- Porta 59593, trava `/tmp/beellama-gpu-benchmark.lock`, verificação de CPU e
  serviços antes/depois de cada pedido, restauração em `finally`.

```bash
python3 docs/benchmarks/mtp-dflash2-suffix-dense-20260929/runner.py dense-v1
python3 docs/benchmarks/mtp-dflash2-suffix-dense-20260929/analyze.py docs/benchmarks/mtp-dflash2-suffix-dense-20260929/dense-v1
python3 -m unittest discover -s docs/benchmarks/mtp-dflash2-suffix-dense-20260929 -p test_analyze.py -v
```

Cada diretório de execução preserva comandos, ambiente, diff, hashes dos binários
e auxiliar, respostas com IDs, logs completos em `.server-output.txt`, telemetria
APERF/MPERF/Radeon e estado original/restaurado da CPU.
O hash do GGUF alvo foi registrado após a bateria em `target-model-sha256.json`.

O relatório anterior em `../mtp-dflash2-suffix-value-20260929/` foi invalidado:
misturava pedidos no mapa de posições, transformava lacunas em comprimento
zero/parcial e comparava com a rodada errada.

## Resultado: `dense-v1`

35 pedidos medidos e sete aquecimentos. Todos os 18 pedidos com observador
(incluindo seus três aquecimentos) produziram exatamente os IDs do MTP isolado
com o mesmo limite. Os 136 sufixos utilizáveis medidos tiveram continuação
completa, sem lacunas na comparação configurada.

| Limite MTP/verificador | Oportunidades | Primeiro token coincide | L médio | Avanço MTP seguinte | Avanço offline L+1 | Diferença média |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 40 | 36/40 = 90,0% | 1,90 | 158 | 116 | −1,05 |
| 3 | 47 | 44/47 = 93,6% | 2,68 | 165 | 173 | +0,17 |
| 2 | 49 | 43/49 = 87,8% | 1,63 | 123 | 129 | +0,12 |

**A previsão do futuro é plausível; o comprimento e o custo são os obstáculos
observados.** No n4, 26/40 candidatos só oferecem dois tokens e a rodada MTP
seguinte avança, em média, 4,69 tokens nesses casos. Os 14 sufixos de pelo menos
três tokens somam +6 tokens sobre suas próximas rodadas MTP e 130 ms de dispatch
potencialmente evitável, ainda pouco diante do custo total do auxiliar.

### Throughput medido (observação, não híbrido ativo)

| Modo | Repetição tok/s | Código tok/s | Longo tok/s |
| --- | ---: | ---: | ---: |
| MTP n4 antes | 86,420 | 59,929 | 58,262 |
| Observador + MTP n4 | 81,246 | 55,759 | 54,178 |
| MTP n3 | 76,027 | 58,113 | 54,944 |
| Observador + MTP n3 | 71,807 | 53,923 | 50,743 |
| MTP n2 | 67,290 | 57,358 | 51,282 |
| Observador + MTP n2 | 62,356 | 52,799 | 47,192 |
| MTP n4 depois | 85,795 | 60,021 | 58,384 |

Medianas dos dois pedidos curtos; uma execução longa por modo. A amostra é de
triagem, não de promoção: o plano pede cinco repetições pareadas para isso.
O controle n4 antes/depois variou menos de 0,8% nas três medianas.

Somando o tempo de geração dos cinco pedidos por modo, o observador reduz o
throughput em 6,7% (n4), 6,8% (n3) e 7,7% (n2) contra seu controle contemporâneo.
Uma conta local que remove o dispatch MTP evitável, altera o avanço pelo `L`
observado e mantém os outros tempos/âncoras fixos ainda fica abaixo do controle:
n4 −8,1% usando todos os sufixos, n4 −5,3% usando somente comprimento ≥3,
n3 −3,4%, n2 −5,3%. **São estimativas locais, não benchmarks ativos nem limites
matemáticos de velocidade.** O custo real de verificar outro comprimento e a
mudança das próximas âncoras precisam ser medidos num piloto separado.

### Capacidade independente: hipótese para o próximo piloto

O observador limita o sufixo ao máximo atual do verificador. Reduzir MTP para
dois também corta o sufixo em dois, mesmo quando há mais tokens disponíveis.
Reavaliando **offline** os mesmos jobs com capacidade hipotética de quatro:

- n3: 47 comparações completas; avanço 179 contra 165 do MTP (+14).
- n2: 48 comparações completas e uma censurada no fim da resposta; avanço 184
  contra 120 do MTP (+64), ou **+1,33 token por oportunidade**. Nessas 48
  comparações, L médio é 2,83, contra avanço MTP médio de 2,50.

Isso não altera o binário nem constitui aceitação real. O piloto que merece
investigação é **MTP curto + capacidade do verificador independente**, com o
MTP n4 mantido como referência de desempenho. Apenas trocar n4 por n2 mantendo
o limite comum não valida essa hipótese. A frequência EVERY=4 também não pode
ser extrapolada para EVERY=1 sem medir contenção, prontidão e trabalho perdido.

### Correção, hardware e limitações

- Todos os IDs emitidos foram auditados contra os eventos definitivos do
  servidor, incluindo rodadas sem proposta e aceitação parcial. Não houve
  erro do worker nem observação pendente ao fechar os modos.
- SCLK ativo médio observado: n4 2700 MHz; n3 2689 MHz; n2 2700 MHz.
  APU configurada em 20 W. CPU APERF/MPERF agregada: 2,94–2,98 GHz nos controles
  e 2,50–2,61 GHz com auxiliar. Não é medição isolada da thread do target.
- Os modos MTP n2/n3 já diferem do n4 **sem observador**: repetição no índice
  de saída 31 (base zero; IDs 13 versus 680); código n3 no índice 117
  (15673 versus 39492) e n2 no 136 (15598 versus 30057). O prompt longo coincide.
  A causa dessas diferenças não foi estabelecida; não atribuí-las a ruído
  numérico sem verificar logits/estado. Equivalência entre limites diferentes
  não foi validada, embora a equivalência observador/controle do mesmo limite
  tenha passado em todos os pedidos.
- Contexto 16k não significa todos os prompts com 16k tokens: o longo tem
  13850; os curtos têm 25/28 tokens. Saída de 192 tokens.
- Produção restaurada pelo runner, com health checks em 8090 e 8092;
  as 16 políticas CPU foram restauradas e conferidas. Os logs registram a
  restauração de SCLK e dos limites APU ao encerrar cada servidor de teste.

Decisão desta etapa: manter MTP n4 como referência; não habilitar substituição
ativa indiscriminada. A compatibilidade do sufixo passou pela medição corrigida;
o ganho líquido ainda não foi demonstrado. Antes de uma implementação maior,
resolver o contrato de capacidade independente e a referência gulosa, e medir
um piloto opt-in preservando `MTP::process/accept` e rollback upstream.

## Validação da implementação e da análise

Build incremental de `llama-server`, `test-dflash-shadow-observation` e
`test-dflash-pipeline-candidate` concluído, reutilizando os plugins CUDA/Vulkan.
Quatro testes CTest passaram: observador, candidato do pipeline, limite estático
upstream DFlash e posicionamento estático do seletor DFlash2. Três testes Python
da análise passaram: lacuna censurada, comparação com a próxima rodada incluindo
bônus e isolamento de pedidos/conflitos. `git diff --check` passou.
Saídas dos checks preservadas em `dense-v1/validation-*.txt`; estado restaurado
conferido em `dense-v1/restoration-verified.json`.
