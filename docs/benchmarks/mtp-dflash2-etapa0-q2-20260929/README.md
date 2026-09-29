# Etapa 0 (complemento) — DFlash2 Q2_K contra Q4_K_M — 2026-09-29

Objetivo: medir a diferença do DFlash2 **Q2_K** contra o **Q4_K_M** usado na
Etapa 0, no mesmo binário e ambiente (`main` local com o modo sombra ainda
desligado; ver `git.txt`). O auxiliar não entrou na resposta: é o DFlash
isolado, com split local, prefetch desligado, APU 20 W, SCLK 2700 MHz, CPU na
política original, contexto 16384, batch/ubatch 64, KV q4_0, temperature 0,
seed 42, 192 tokens; 3 execuções curtas e 2 longas por modo (mediana).

Draft: `Qwen3.8-27B-DFlash2-Q2_K.gguf`, sha256
`e3eb7705404817cdbcdabe56049a1952b3b37bcc8df6e4d4efaec5d41563fb7e`
(`draft-model-sha256.json`).

## Resultados

| Auxiliar DFlash2 | Repetição, tok/s | Código, tok/s | Longo, tok/s | Prefill longo, tok/s |
| --- | ---: | ---: | ---: | ---: |
| Q4_K_M n=6 (Etapa 0) | 70,05 | 47,40 | 49,69 | 649 |
| Q2_K n=6 | 67,78 | 45,97 | 40,81 | 653 |
| Q4_K_M n=7 (Etapa 0) | 73,85 | 44,53 | 47,98 | 646 |
| Q2_K n=7 | 71,33 | 44,52 | 38,23 | 650 |

- Q2_K perde 3–4% nos curtos e 18–20% no prompt longo. Q4_K_M permanece a
  referência inicial; este teste isolado não mede reaproveitamento concorrente.
- Saídas finais idênticas ao Q4_K_M nos três cenários (prefixos `031bf5c6`,
  `19f387f9`, `8508b88d`); hashes completos em `summary.json`/`results.json`.
- Clocks: 2700 MHz sustentados; PPT médio no decode ~14,8 W; pico 71 °C.
- Correção após revisão dos logs: os contadores de aceitação estão preservados.
  No longo, Q4 n=6/n=7 teve 67,40%/61,11%; Q2 n=6/n=7 teve 51,80%/44,72%.
  Isso apoia manter Q4 como referência; o custo por bloco e a utilidade do Q2
  no observador concorrente exigem uma medição própria.

## Arquivos e reprodução

`summary.json`, `results.json`, respostas e telemetria por requisição,
`*.command.json`, `*.server.log`, hashes de binário/modelo e políticas de CPU.
`SUDO_PASS=<senha> python3 runner.py` (para e restaura os serviços; grava os
artefatos neste diretório).
