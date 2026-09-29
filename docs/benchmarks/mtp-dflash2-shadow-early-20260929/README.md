# Lançamento antecipado do auxiliar (hipótese da janela) — 2026-09-29

> **Revisão do commit `3d4128354`:** a ideia do lançamento antecipado foi
> mantida, mas o commit introduziu uma regressão em `EVERY=0` (virava 1), e a
> cadência era contada em tentativas no modo early e em propostas não vazias no
> modo late. As amostras não eram estritamente pareadas. Os snapshots de CPU
> também diferem: off usou EPP balance_power/máximo 5134889 kHz; on usou
> balance_power/3301000 kHz. O runner antigo não verificava a política por
> requisição. A nova revisão corrige esses pontos e repete o ensaio em diretório
> separado. Resultados na raiz são históricos e não são sobrescritos.

Objetivo: testar se iniciar o bloco auxiliar **antes** do draft MTP primário
(ao invés de depois, comportamento padrão) reduz os atrasos observados no
`EVERY=4` (16 prefixos corretos tardios, mediana 4,693 ms) sem custo no
primário.

Implementação: variável de ambiente experimental `GGML_DFLASH_SHADOW_EARLY=1`
(padrão 0, comportamento anterior preservado). Com lançamento antecipado, se o
draft primário não produz proposta, o job é cancelado (sem verificação para
commitá-lo); a contabilidade fecha em observados+cancelados.

## Configuração

Um binário registrado para os dois casos deste A/B, diferente do binário da
bateria corrigida anterior (o hash de libllama-common mudou): alvo Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp
em CUDA0; auxiliar DFlash2 Q4_K_M em Vulkan0; MTP n=4/p_min 0,70; DFlash n=7;
ctx 16384; b/ub 64; parallel 1; KV q4_0; temperature 0; seed 42; 192 tokens;
APU 20 W e SCLK 2700 MHz; CPU powersave com políticas diferentes conforme os snapshots e o aviso acima;
não houve controle por requisição nessa execução histórica. `GGML_DFLASH_SHADOW_EVERY=4` nos
dois modos; 3 curtas + 3 de código + 2 longas por modo, com aquecimento.

## Resultados históricos

Os contadores de observação abaixo incluem o aquecimento de 64 tokens; os tok/s
usam somente as oito requisições medidas por modo. Excluindo aquecimento,
usáveis foram 54/97 (55,7%) e 68/96 (70,8%), com 20 e 5 atrasos respectivamente.

| Métrica | early-off (padrão) | early-on |
| --- | ---: | ---: |
| Repetição / Código / Longo (tok/s) | 82,72 / 56,37 / 54,36 | 82,80 / 57,18 / 54,29 |
| Lançados | 104 | 117 |
| Observados | 102 | 101 |
| Cancelados | 2 | 16 |
| Prontos no prazo | 79 (77,5%) | 96 (95,0%) |
| Tardios | 23 | 5 |
| Prefixo coincide | 69 (67,6%) | 76 (75,2%) |
| Pronto + coincide + sufixo | 55 (53,9%) | 71 (70,3%) |
| Tokens candidatos | 132 | 186 |
| pending / errors | 0 / 0 | 0 / 0 |

- Hashes de saída idênticos ao MTP isolado nos três cenários. Os valores de
  throughput são os medidos, mas diferenças de CPU e amostragem impedem
  concluir desse ensaio sozinho que o lançamento não tem custo.
- Contabilidade fecha nos dois modos: observados + cancelados = lançados.
- O aumento de cancelados é esperado: jobs antecipados de ciclos em que o
  primário não produziu proposta são cancelados (não haveria verificação).
- Restauração histórica: `cpu-restored.json` difere de `cpu-original.json`.
  O registro não demonstra restauração exata da política encontrada no início.
  O novo runner preserva o original, aplica uma política de teste explícita e
  exige readback antes/depois de cada requisição e na restauração.

## Evidências e reprodução após a revisão

Os logs originais `.server.log` eram ignorados pelo Git. Os eventos necessários
para conferir os contadores agora estão preservados em `*.observer-output.txt`.
Os valores de throughput e hashes são verificáveis pelas respostas e resumos
originais. `git.txt` antigo registrava HEAD e status, sem o diff; por isso a
correspondência exata com a fonte compilada tem essa limitação histórica.

O runner revisado grava em um subdiretório novo (`python3 runner.py review-v2`),
com diff completo, header do observador, todos os binários explicitamente
precarregados (incluindo `libllama-server-impl.so`), ambiente por modo e logs
completos em `*.server-output.txt`. Ele testa sync-only com EARLY=1, early-off/on
com a mesma cadência e cancelamento quando o primário não propõe tokens. A senha
é solicitada por getpass e não é persistida. Serviços e configurações são
restaurados em `finally`.

## Resultado da revisão

A [nova bateria controlada](review-v2/README.md) validou as correções: 101
âncoras pareadas, atrasos 14→4 e oportunidades 67→73, com hashes idênticos.
EVERY=0 não lança blocos; o caso de primário sem proposta encerra corretamente
221 cancelamentos. Build e quatro testes passaram. O modo ainda não usa
sufixos na resposta.
