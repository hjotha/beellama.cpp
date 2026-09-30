# Etapa 0 — referência renovada MTP (n=2/3/4) e DFlash2 (n=6/7) — 2026-09-29

Objetivo: renovar a referência da Etapa 0 do plano
`docs/mtp-dflash2-hybrid-plan.md` no mesmo binário e ambiente do futuro
protótipo. Nenhum modo híbrido foi implementado ou medido aqui.

## Ambiente

- Host: gokaya; alvo `Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf` na RTX 4070 (CUDA0);
  draft `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` na Radeon integrada (Vulkan0).
- Contexto 16384; batch/ubatch 64; parallel 1; KV alvo e draft `q4_0`;
  temperature 0; seed 42; ignore_eos; cache_prompt false; 192 tokens gerados.
- Prompts preservados em `prompts.json` (repetição, código e longo com 13850
  tokens de entrada).
- DFlash: split local ligado, prefetch desligado, APU 20 W, SCLK solicitado em
  2700 MHz (média observada 2700 MHz). CPU na política original (powersave);
  clock efetivo APERF/MPERF ~1620 MHz no DFlash e ~2800 MHz no MTP.
- Fonte: `main` no commit `6281c6b5685d21ab1260308ebe4b0e4bc6f0ab1a`, árvore
  limpa (`git.txt`). Hashes completos dos binários em `binary-sha256.json`;
  idênticos aos registrados nos baselines anteriores. Prefixos, na ordem do
  arquivo: llama-server `bda17a8c`, libllama `7695238c`, libllama-common
  `d31acf12`, ggml-vulkan `f5e5c2fc`, ggml-cuda `625f2b0d`.
- Modelos: alvo com identity-sha256 `63f29a21` (prefixo; valor completo no
  arquivo `.identity-sha256` do modelo); draft Q4_K_M sha256
  `1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd`.

## Resultados (mediana, tok/s)

| Modo | Repetição | Código | Longo | Prefill longo |
| --- | ---: | ---: | ---: | ---: |
| MTP n=2, p_min 0.70 | 67,06 | 57,45 | 51,10 | 676,0 |
| MTP n=3, p_min 0.70 | 74,95 | 57,65 | 54,17 | 670,5 |
| MTP n=4, p_min 0.70 | 84,50 | 59,08 | 57,41 | 665,8 |
| DFlash n=6 | 70,05 | 47,40 | 49,69 | 649,0 |
| DFlash n=7 | 73,85 | 44,53 | 47,98 | 645,9 |

- 3 execuções por prompt curto e 2 por longo; dispersão e hashes em
  `summary.json`/`results.json`; telemetria completa por requisição em
  `*.telemetry.json`; comandos exatos em `*.command.json`; logs em `*.server.log`.
- Hashes determinísticos dentro de cada modo (todas as repetições iguais).
- MTP n=4 é a melhor referência isolada nos três cenários.
- DFlash n=7 supera n=6 apenas em repetição (+5,4%); perde em código (−6,1%) e
  longo (−3,4%). Não há ganho uniforme em aumentar o bloco de difusão.
- Conferência com referências do mesmo binário (2026-09-29): DFlash n=6 deu
  70,05/47,40/49,69 contra 71,43/48,26/49,72 (powersave-20W-SCLK2700) —
  dentro da dispersão. MTP n=4 deu 84,50/59,08/57,41 contra 87,85/60,94/57,98
  do registro de 2026-09-28 (execução única); os curtos ficaram 3–4% abaixo,
  o longo praticamente igual.

## Achado relevante para a Etapa 1 (hashes entre modos)

- Repetição: MTP n=2 = n=3 (prefixo `55b20c7c`); MTP n=4 = DFlash n=6 = n=7
  (prefixo `031bf5c6`).
- Código: MTP n=2 (prefixo `f582cdd9`); n=3 (prefixo `bd1a517e`); n=4 =
  DFlash n=6 = n=7 (prefixo `19f387f9`).
- Longo: todos os MTP (prefixo `1e9b0179`); DFlash n=6 = n=7 (prefixo
  `8508b88d`).

Hashes completos por resposta estão em `results.json`/`summary.json`.

Ou seja: a trajetória gulosa do próprio MTP muda com `n_max` nos prompts curtos
e o DFlash só coincide com o MTP n=4 nesses curtos; no prompt longo as
trajetórias divergem. Isso reforça a validação da seção 10 do plano: a
coincidência de prefixo MTP×DFlash precisa ser medida, não presumida, e o
critério de hash exato entre trajetórias "de referência" é sensível à
configuração.

## Reprodução

- `runner.py` executa a bateria (exige `SUDO_PASS` no ambiente e a senha de
  sudo válida); `cpu-controller.py` é o coletor privilegiado de APERF/MPERF e
  da política de CPU, chamado pelo runner a partir deste diretório.
- O runner grava os artefatos por execução em `/tmp` e esta pasta é a cópia
  preservada no repositório; `prompts.json`, comandos e telemetria são
  autossuficientes para reexecução.

## Ressalvas

- Janela única de aquecimento; DFlash não teve repetição pareada entre modos
  (uma bateria sequencial por modo, alternada apenas na ordem fixa).
- VRAM de pico por modo está nos `*.server.log`; o estado da NVIDIA antes de
  cada modo está em `*.nvidia-before.csv` (limite 170 W, sem mudança entre
  casos).
- Serviços (`llama-server-root.service`, `qwen35-4b-mtp-8092.service`) e
  política de CPU foram restaurados e verificados; `/health` OK em 8090/8092.
