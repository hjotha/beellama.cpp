# Continuação occupancy position-split

Revisão improve + ponytail, 2026-10-04, base dfd98368f. Os plans/001 e 002 existentes tratam de outro escopo e foram preservados.

| Plano | Prioridade | Estado | Dependências |
|---|---|---|---|
| [001 — Revisão 5 e continuação](001-occupancy-review5-continue.md) | P1 | TODO | Plano original e revisão 4 reconciliados no arquivo |

Ordem: reparar splitter e testes/merge, fechar F1, F2b, F3 4B, F4 disco, F5 27B, F7 autorizado. F6 opcional. Critérios e achados rejeitados constam no plano; execução sem encerramentos parciais. Atualizar o estado aqui e manter evidência detalhada no checklist docs/occupancy-progress-20261003.md.

## Performance RKVA — revisão de 20/10/2026

| Plano | Prioridade | Estado | Dependências |
|---|---|---|---|
| [002 — Regressão CUDA FP8 e cobertura da review](002-rkva276-regression-performance-steer.md) | P1 | TODO — encaminhar AGY para Sol | Build 276 e baseline CPU; instrumentar antes de otimizar |

Escopo independente do plano 001. Ordem interna: profile e A/B controlado, regressão, alocações/cópias, FP8 Xbox, demais gates e wavefront condicionado a evidência. Hipóteses/descartes e critérios estão no plano.
