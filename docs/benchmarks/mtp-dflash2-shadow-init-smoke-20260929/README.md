# Smoke — inicialização do contexto auxiliar de observação (Etapa 1, item 1) — 2026-09-29

Objetivo: validar que o modo `--spec-draft-shadow-*` carrega o DFlash2 auxiliar
na Radeon com ownership independente do drafter primário (MTP) e **sem alterar a
resposta**. Nenhum token auxiliar é usado; a pipeline de observação ainda não
está ligada (itens 2–5 da Etapa 1 pendentes).

## Comando

`command.json`. Alvo Qwen3.8 IQ3 XXS MTP na CUDA0, MTP n=4 p_min 0,70 como
primário; auxiliar `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` na `Vulkan0`, `ngl all`,
`n_max 7`. Contexto 16384; batch/ubatch 64; KV q4_0; um slot.

## Resultados

- Log: `shadow auxiliary context ready (model '.../Qwen3.8-27B-DFlash2-Q4_K_M.gguf', n_max 7)`
  (`shadow-log-lines.txt`; log completo em `server.log`).
- Resposta (repetição, 192 tokens, temperature 0, seed 42): hash
  `031bf5c616d5b0a473d97b785404aede9b8a3a6714ea6b55e9908f560c0e80b8` —
  idêntico ao baseline MTP n=4 da Etapa 0; 84,4 tok/s (`result.json`,
  `response.json`).
- Serviços de produção (`llama-server-root.service`, `qwen35-4b-mtp-8092.service`)
  parados durante o smoke e restaurados com `/health` OK.
- Fonte: `main` com as alterações locais desta etapa (`git.txt`); hashes dos
  binários em `binary-sha256.json`.

## Reprodução

`SUDO_PASS=<senha> python3 smoke.py` (requer a senha de sudo; para e restaura os
serviços, grava os artefatos neste diretório). O `LD_PRELOAD` usa caminhos
absolutos das bibliotecas de `build-dflash-xbox-dl/bin`.
