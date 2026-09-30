# Valor do sufixo em observação (compatibilidade futura) — 2026-09-29

> **Resultado invalidado na revisão seguinte.** O script abaixo usa um único
> mapa `posição → token` para pedidos distintos, classifica lacunas como
> comprimento zero/parcial e compara `(L+1)` com `c` da rodada de origem,
> em vez da rodada MTP seguinte. As taxas e a conclusão de perda abaixo não
> sustentam uma decisão sobre o híbrido. Mantidas somente como histórico.
> A medição corrigida está em
> [`../mtp-dflash2-suffix-dense-20260929/`](../mtp-dflash2-suffix-dense-20260929/).

Análise **offline** (§8 do plano, "Compatibilidade futura em observação"):
para cada job utilizável do run `shadow-every4-early-on`
(`../mtp-dflash2-shadow-early-20260929/`), compara o sufixo `D[c:c+usable]`
com a continuação realmente confirmada pelo alvo nas posições seguintes
(reconstruída dos `confirmed` de todos os jobs). Nenhuma inferência foi
executada; sem mudanças de código.

## Resultado (101 jobs; 71 utilizáveis)

| Métrica | Valor |
| --- | ---: |
| Primeiro token do sufixo coincide | 20/71 (28,2%) |
| Distribuição de L (sufixo aceitável) | 0:51, 1:1, 2:16, 4:3 |
| Média / mediana de L | 0,63 / 0 |
| Ganho por rodada (L+1) − c | média **−2,52**; >0 em 7/71 (9,9%) |
| Tokens da rodada: MTP real | 295 |
| Tokens da rodada: híbrido (L+1) | 116 |
| Comparações truncadas por lacuna (EVERY=4) | 20/71 |

## Leitura

Mesmo quando o prefixo completo coincide (incluindo o bônus), o sufixo raramente
sobrevive ao primeiro token: a trajetória gulosa diverge depois de `c`. Nas
condições medidas, substituir o draft MTP pelo sufixo **perderia** tokens por
rodada na maioria dos casos (média −2,5). Isso é compatível com a ressalva do
plano (§7): compatibilidade na trajetória de referência não prova ganho, e a
aceitação real do verificador híbrido ainda não foi medida.

## Limitações

- Cobertura esparsa (`EVERY=4`): 20 comparações pararam na primeira posição
  sem token confirmado; uma medição `EVERY=1` densa está pendente (GPU ocupada
  por benchmark externo em 15:2x).
- Trajetória de referência é a produzida sob drafting MTP; um modo híbrido
  poderia divergir (ganho/perda poderiam mudar) — e é por isso que a aceitação
  real do modo ativo continua sendo a métrica final.
- Capacidade capada em 4 (limite do verificador/MTP).

Script: `analyze_suffix.py` (entrada: log do early-on; saída acima).
