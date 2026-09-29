# Probe histórico de etapas do shadow — 2026-09-29

Este diretório preserva `command.json`, `response.json`, `result.json`, os hashes
em `binary-sha256.json` e os trechos de log em `shadow-log-lines.txt`.

O resultado observado foi 59,6245 tok/s no prompt de repetição, com hash igual
ao MTP n=4. Foram 40 blocos, 2.213.664 us de worker, 627.050 us de `decode_us`,
196.475 us de `selector_us`, 89.780 us de injeção e 43.477 us de cópia.

**Limitação descoberta na revisão:** os timers de decode e encode terminavam
antes de `llama_get_embeddings` e `llama_get_embeddings_nextn`. Esses getters
sincronizam as GPUs. Portanto, os valores de 15,7/4,9 ms não medem as latências
completas do transformer/seletor, e o residual de aproximadamente 35 ms não pode
ser atribuído a preparo/amostragem. A medição corrigida inclui as sincronizações
que já eram necessárias, sem acrescentar uma barreira artificial.

As métricas de prontidão e prefixo deste probe também usam o observador antigo;
ver [a revisão e a nova bateria](../mtp-dflash2-shadow-corrected-20260929/README.md).
