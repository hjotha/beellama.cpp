# Lançamento antecipado do auxiliar (hipótese da janela) — 2026-09-29

Objetivo: testar se iniciar o bloco auxiliar **antes** do draft MTP primário
(ao invés de depois, comportamento padrão) reduz os atrasos observados no
`EVERY=4` (16 prefixos corretos tardios, mediana 4,693 ms) sem custo no
primário.

Implementação: variável de ambiente experimental `GGML_DFLASH_SHADOW_EARLY=1`
(padrão 0, comportamento anterior preservado). Com lançamento antecipado, se o
draft primário não produz proposta, o job é cancelado (sem verificação para
commitá-lo); a contabilidade fecha em observados+cancelados.

## Configuração

Mesmo binário/ambiente da bateria corrigida: alvo Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp
em CUDA0; auxiliar DFlash2 Q4_K_M em Vulkan0; MTP n=4/p_min 0,70; DFlash n=7;
ctx 16384; b/ub 64; parallel 1; KV q4_0; temperature 0; seed 42; 192 tokens;
APU 20 W e SCLK 2700 MHz; CPU powersave/EPP `power`/419175–3301000 kHz (readback
conferido antes e depois de cada requisição). `GGML_DFLASH_SHADOW_EVERY=4` nos
dois modos; 3 curtas + 3 de código + 2 longas por modo, com aquecimento.

## Resultados

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

- Sem custo no primário (tps idêntico dentro do ruído) e hashes de saída
  idênticos ao MTP isolado nos três cenários.
- Contabilidade fecha nos dois modos: observados + cancelados = lançados.
- O aumento de cancelados é esperado: jobs antecipados de ciclos em que o
  primário não produziu proposta são cancelados (não haveria verificação).
- Restauração: serviços e política de CPU conferidos; um ator externo voltou a
  alterar o EPP durante a bateria, restaurado ao baseline `power` (diferença
  NONE).

Arquivos: `runner.py` (A/B por modo com env próprio), `cpu-controller.py`,
`*.server.log`, `*.command.json`, `*.env.json`, `summary.json`, respostas e
telemetria. Reprodução: `SUDO_PASS=<senha> python3 runner.py`.
