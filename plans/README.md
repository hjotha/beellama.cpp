# Planos de implementação

Gerado em 12/09/2026 para `/home/hjotha/llama`, base `55ac3792a`.

| Plano | Objetivo | Prioridade | Esforço | Status |
|---|---|---|---|---|
| [001](001-adaptive-context-resident-weights.md) | Alternar contexto/MTP mantendo pesos principais residentes | P1 | L | TODO |

Executar as etapas do plano em ordem: prova de ciclo de vida, grupo MTP liberável, integração no servidor, regressão e calibração real. Nenhuma implementação, build ou implantação foi feita na criação deste plano.

Decisões consideradas:

- Gating de MTP ou limpar KV sem destruir contextos: insuficiente para recuperar memória fixa.
- Manter pesos MTP sempre residentes: útil como experimento, não comprova o teto longo atual.
- Redimensionar continuamente a cada tamanho de pedido: custo e complexidade desnecessários; manter dois perfis.
- Conservar os dois contextos na GPU: incompatível com o orçamento pretendido.
- Transferir KV entre perfis: adiado; o usuário aceita descartá-lo.
- Conclusão histórica de impossibilidade absoluta em um processo: qualificada por separar ownership de contextos e pesos MTP.
