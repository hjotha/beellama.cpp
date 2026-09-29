# Custo do auxiliar sincronizado (Etapa 1, caso 2 da matriz) — 2026-09-29

Objetivo: medir o custo de manter o contexto auxiliar DFlash2 carregado e
sincronizado com o prefixo confirmado, **sem gerar blocos** (caso 2 da matriz de
validação da seção 9 do plano). O auxiliar recebe `begin/process/accept` como o
primário, mas o `draft()` nunca é chamado; nenhum token auxiliar entra na
resposta.

## Configuração

Comando em `mtp-n4-shadow-sync.command.json`. MTP n=4 p_min 0,70 como primário
na CUDA0; auxiliar `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` na `Vulkan0`, split local,
`n_max 7`. Contexto 16384; batch/ubatch 64; KV q4_0; um slot; APU 20 W; SCLK
2700 MHz; CPU na política original. 3 execuções curtas e 2 longas (mediana) com
os mesmos prompts da Etapa 0; `warmup` de 64 tokens.

## Resultados

| Configuração | Repetição, tok/s | Código, tok/s | Longo, tok/s | Prefill longo, tok/s |
| --- | ---: | ---: | ---: | ---: |
| MTP n=4 isolado (Etapa 0) | 84,50 | 59,08 | 57,41 | 666 |
| MTP n=4 + auxiliar sincronizado | 81,65 | 56,61 | 54,76 | 631 |
| Diferença | −3,4% | −4,2% | −4,6% | −5,2% |

- Hashes idênticos ao MTP isolado nos três cenários (prefixos `031bf5c6`,
  `19f387f9`, `1e9b0179`): o auxiliar não alterou a resposta.
- Log confirma `shadow auxiliary drafter registered` e nenhuma falha de
  `process()`; clocks 2700 MHz sustentados; PPT médio ~10,1 W; pico 58 °C.
- O custo é a injeção das features do alvo na KV da Radeon a cada ciclo mais a
  memória do contexto auxiliar. Este é o piso de custo do modo de observação;
  o benefício precisa superá-lo (seção 8 do plano).

## Arquivos e reprodução

Comando, log do servidor, respostas, telemetria por requisição, políticas de
CPU e hashes de binário/modelo neste diretório. `SUDO_PASS=<senha> python3
runner.py` (para e restaura os serviços; grava os artefatos aqui).

Nota: o `accept()` do auxiliar em `is_other == true` é um no-op no split local;
por isso a KV auxiliar contém apenas features reais confirmadas. O teste de
fumaça da inicialização está em `../mtp-dflash2-shadow-init-smoke-20260929/`.
