# Paridade Strata × BeeLlama — status e pendências (2026-11-22)

Base do fechamento: `main` em `458aadf56`. Pacotes medidos: `parity12` (integração),
`parity13` (GLU Q2_0) e `parity14` (reuso experimental). As medições BeeLlama abaixo
usam a **A100-40G** (40.960 MiB de VRAM), com `--parallel 1`, `--fit off`, KV q8_0/q8_0, `temperature 0`,
`cache_prompt false`, dois warmups e mediana por braço. Os casos de throughput
com geração usam os prompts `pp4k+dec` e `pp28k+dec` de `bench.py`
(4.627 e 28.547 tokens com template, até 128 tokens de saída).

## 1. Onde estamos

| braço | prefill 4k | prefill 28k | decode curto | decode pós-4k | decode pós-28k |
|---|---:|---:|---:|---:|---:|
| **Bee Q2_0**, `-ub 2048`, `-ncmoe 2`, cache 2048 | 1.771 | **1.829** | 49,9 | 46,3 | 45,96 |
| **Bee Q2_0** + GLU e Q8_1 compartilhado | 1.883,5 | **1.946,6** | — | 46,2 | 45,61 |
| Bee Q2_0 + `LLAMA_MMQ_GATE_UP_SHARED_Q8=1` | 1.783 | **1.850** | — | 46,2 | 46,0 |
| Bee Q2_0 + DFlash2 n-max 7 | 1.344 | 1.388 | 58,4 | 83,3 | 81,6 (instável) |
| Bee Q2_0 + DFlash2 **n-max 8** | 1.342 | 1.384 | 74,0 | **95,4** | **87,3** |
| Bee Q2_0 + DFlash2, braço com `--spec-dm-controller profit` solicitado | 1.348 | 1.387 | 72,9 | **96,6** | **88,0** |
| **Strata 0.2.17** (mesmos pesos, INT8, 24.576 experts, DFlash2) | 2.394,7 | **4.629,3** | **85,9** | **67,4** | **75,0** |

Com **IQ3_S** (84 GB, 12 GB a mais que o Q2_0 em pesos):

| braço | prefill 4k | prefill 28k | decode 28k |
|---|---:|---:|---:|
| `-ncmoe 44`, cache 16.000 MiB, sem admissão parcial (`parity7`) | 479,5 | 647,2 | 25,2 |
| `-ncmoe 44`, cache 16.000 MiB, admissão parcial | 545,6 | 818,9 (+26,5%) | 24,5 |
| `-ncmoe 44`, cache 20.000 MiB, admissão parcial | 578,9 | **914,8 (+41,4%)** | 25,5 |

A última linha combina admissão parcial e aumento do cache. Frente ao melhor placement
anterior (740,9 t/s), o ganho foi de 23,5%.

DFlash2 supera a Strata no decode pós-4k e pós-28k. No prompt curto, faz 74,0 t/s contra
85,9 t/s da Strata. Em pp28k, a Strata ainda é 2,38× mais rápida que o melhor Q2_0 integrado.
Seu resultado de 4.629,3 t/s é 5,06× o do BeeLlama com IQ3_S, uma quantização diferente.

## 2. O que já foi entregue e medido (no `main`)

| entrega | flags | ganho medido | validação |
|---|---|---|---|
| QSA esparso Q8_0 | default | prefill pp28k +0,2%, decode 28k +9,9% | 28/28 backend-ops, A100 |
| MTP sidecar | `LLAMA_MTP_RESERVE_WEIGHTS=1` (store-only experimental) | decode 4k/28k +37%/+31% | A100, aceitação ~80% |
| Residência de experts | `LLAMA_MOE_CACHE_PREFILL=1` (recipe) | prefill +3,5% (com cache 2048 hoje: **+7,6%**) | A100, fixture |
| Mapa imutável no device | `LLAMA_MOE_MAP_DEVICE=1` | +5,2% decode curto | A100 |
| Overlap de upload / ring pinned | `LLAMA_MOE_UPLOAD_OVERLAP`, `LLAMA_MOE_PINNED_RING` | ~0 | A100 |
| QSA indexer bounds / GDN conv+L2 | `LLAMA_QSA_INDEXER_BOUNDS`, `LLAMA_GDN_CONV_L2` | ~0 | A100 |
| Admissão parcial do cache | `LLAMA_MOE_CACHE_PARTIAL_PREFILL=1` | **+26,5% pp28k IQ3_S** a 16.000 MiB (idêntico bit a bit) | A100 + `deepseek2` 3/3 |
| MMQ gate/up Q8_1 compartilhado | `LLAMA_MMQ_GATE_UP_SHARED_Q8=1` | **+1,0-1,6% prefill** | bit-exact CUDA |
| Fusão gate/up/GLU Q2_0 | `LLAMA_MMQ_GATE_UP_GLU=1` com Q8_1 compartilhado | **+5,3% pp28k** | bit-exact, CUDA e A100 |
| Profiler por op de GPU | `LLAMA_OP_PROFILE_GPU=1` | diagnóstico | 4070 + A100 |
| PLE prefetch | `LLAMA_PLE_PREFETCH=1` | ~0 (A/B frio real) | A100 |
| DFlash2 (alvo Qwen4Exp) | `--spec-type draft-dflash --spec-draft-n-max 8` | decode pós-4k/pós-28k 95,4/87,3 t/s | A100 |
| Fechamento do ubatch | `-ub 2048` é o teto útil | ver a recipe abaixo | A100 |
| Correções de build | — | `tools/rkva-native-key-inspect` volta a compilar | build CPU |

## 3. Recipes A100-40G

Use o Qwen3.8-Flash-Next Q2_0 do [GGUF fixado para o benchmark](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/2a55d75962e22f7a4a1d9963ab6eae3678537831/Q2_0).
Ajuste `TARGET_GGUF` para o primeiro shard. Mantenha o segundo shard ao lado dele.

```bash
# Prefill (parity13, 1.946,6 t/s em 28k)
LLAMA_MOE_CACHE_PREFILL=1 LLAMA_MOE_MAP_DEVICE=1 \
LLAMA_MMQ_GATE_UP_SHARED_Q8=1 LLAMA_MMQ_GATE_UP_GLU=1 \
llama-server -m "$TARGET_GGUF" -ngl 99 -c 32768 -b 2048 -ub 2048 \
  -ncmoe 2 --moe-cache-mib 2048 --parallel 1 --fit off \
  -fa on -ctk q8_0 -ctv q8_0 --load-mode mmap --lazy-mode on --jinja --no-warmup
```

```bash
# Decode (DFlash2 Q8_0)
LLAMA_MOE_CACHE_PREFILL=1 LLAMA_MOE_MAP_DEVICE=1 \
llama-server -m "$TARGET_GGUF" -ngl 99 -c 32768 -b 2048 -ub 1024 \
  -ncmoe 2 --moe-cache-mib 2048 --parallel 1 --fit off \
  -fa on -ctk q8_0 -ctv q8_0 --load-mode mmap --lazy-mode on --jinja --no-warmup \
  --spec-type draft-dflash --spec-draft-model "$DRAFT_GGUF" --spec-draft-n-max 8
```

Use o drafter Q8_0 e mantenha `-ub 1024` com DFlash2. A varredura anterior esgotou
a VRAM na alocação do grafo com DFlash2 e `-ub 2048`.
[GGUF do drafter](https://huggingface.co/nazeshinjite/Qwen3.8-Flash-Next-DFlash2-GGUF) · [Varredura de ubatch](strata-prefill-ub-20261122.md)

O braço histórico com `--spec-dm-controller profit` registrou 96,6 t/s pós-4k e 88,0
pós-28k. Profit já é o default do parâmetro, e o DFlash2 com seletor top-k desativa
o controlador adaptativo. Use `n-max 8` como ajuste reproduzível.

Com cache, `-ub 3072` fez 1.843,6 t/s (+0,8%) com pico de 39.974 MiB na rodada.
`-ub 4096` coube sem cache (1.869,7 t/s, +2,1%). Mantenha `-ub 2048` com cache.
Alterar `GGML_CUDA_MMQ_J_MAX` também não passou no A/B (`64` caiu de 1.830 para 1.649 t/s
e alterou a saída, `256` empatou com o default).

## 4. O que falta do plano original (Kimi, revisão de 2026-11-21)

### Prefill — tabela de oportunidades

| # | item | status | falta |
|---|---|---|---|
| 1 | Residência de experts no pp28k | **feito** (`LLAMA_MOE_CACHE_PREFILL=1`), +3,5% então, +7,6% hoje | nada |
| 2 | Overlap de upload + staging pinned limitado | **feito** (`LLAMA_MOE_UPLOAD_OVERLAP`, `LLAMA_MOE_PINNED_RING`), ~0 com cache 1024 | qualificação com misses reais + trace de DMA (não bloqueia) |
| 3 | Mapa imutável para grupo integral | **feito** (`LLAMA_MOE_MAP_DEVICE=1`), +5,2% decode curto | nada |
| 4a | Reusar agrupamento/entrada MMQ entre projeções | **feito em parte**: Q8_1 compartilhado (+1,0–1,6%) e fusão GLU Q2_0 (+5,3% em pp28k) integrados | reuso no down medido, abaixo do limiar |
| 4b | Limites de visibilidade no indexer | **feito** (`LLAMA_QSA_INDEXER_BOUNDS`), ~0 | nada |
| 4c | IDs de QSA direto à FA | **não feito** — o gather esparso Q8 existe, o caminho direto não | desenho e A/B no [plano](../PLAN-strata-parity.md#desenho-e-ab-do-item-4c-de-qsa) |
| 4d | Fundir L2 de Q/K no GDN | **feito** (`LLAMA_GDN_CONV_L2`), ~0 | nada |

### Decode — tabela de oportunidades

| # | item | status | falta |
|---|---|---|---|
| Decode-3 | MTP sidecar n-max 2 pareado (reserva + store-only) | **feito**, +31% em 28k | superado pelo DFlash2 (87-96 t/s), que virou a trilha principal |
| Decode-5 | Estabilizar replay CUDA nas larguras reais de verificação | **não feito** | medir decisões de graph por largura de verificação |
| Decode-5b | Hidden do MTP alvo→sidecar via D2D | **não feito** | perdeu prioridade: o DFlash2 já está à frente da Strata no pós-prefill |

### O que o Kimi colocou "fora da fila imediata"

| item | status |
|---|---|
| Port Q8 | não feito |
| Reformulação GDN | não feito |
| MTP embutido neste GGUF | não feito (não existe `blk.48.*`; usa-se sidecar) |
| n-max 4 / patch ratio 4 | medido: n-max 4 é pior que 7/8; ratio 4 não testado |
| Cache compartilhado do draft | removido por corrida entre streams (decisão anterior, não reabrir) |
| CPU no miss | protótipo keyed em `feat/cpufb-verify`, rejeitado no A/B IQ3_S |
| Prefetch SLRU | **feito como PLE** (`LLAMA_PLE_PREFETCH=1`): 0% no A/B frio |
| DFlash2 em trilha separada | **feito** e virou a melhor trilha de decode |

## 5. Lacunas que o plano original não cobria

1. **Tier de CPU para o IQ3_S.** `[OP GPU]` mostra `MUL_MAT_ID` 42,9% + `MUL_MAT` 29,8% do tempo de
   GPU. Os grafos CUDA somaram **4,665 s** de **43,8 s** de parede. Cerca de **89%** ficou
   fora desses grafos, principalmente no split de CPU do IQ3_S. É o item de maior retorno e o mais caro (protocolo de miss
   skip/compactação + escala `amax` + política de reposição; estimativa 6-9 semanas).
2. **Eficiência do `MUL_MAT_ID` no Q2_0.** Com `-ub 2048` e top-10/512 cada expert recebe ~40
   tokens; a fusão GLU reduziu o custo. O reuso do mapa no down isoladamente ficou abaixo do limiar de integração.
3. **`-c 262144`.** O modelo suporta 256k e o KV q8 custaria ~4,1 GiB, mas nunca medimos Bee × Strata
   nessa janela; a Strata usa streaming de KV acima de 64K, o Bee mantém tudo no VRAM.

## 6. Reuso do mapa MMQ no down

Na A100 com `parity14` (`48a6c549a`), Q2_0, GLU e Q8_1 compartilhado ligados nos dois
braços, três execuções por braço. `LLAMA_MMQ_MOE_REUSE_MAP=1` é experimental e default-off
no código arquivado na tag local `archive/strata-expert-major-20261122`.

| flag | pp4k | pp28k (faixa) | decode pós-28k | VRAM pronto/pico |
|---|---:|---:|---:|---:|
| off | 1.882,35 | 1.948,68 (1.947,92–1.948,87) | 45,17 | 37.528 / 38.906 MiB |
| on | 1.896,82 | 1.962,36 (1.962,19–1.962,81) | 45,47 | 37.528 / 38.906 MiB |

Os hashes ficaram idênticos (`09f5c1647f89` / `39d5087ff47a`). O pp28k ganhou
**0,70%**, abaixo do piso de 3% para merge. No pp4k (+0,77% pela mediana), a primeira
execução fria do braço off fez 1.618,38 t/s. O fixture da RTX 4070 registrou NMSE 0
e saída bit a bit para o down, com e sem CUDA Graphs.

## 7. Fallback CPU keyed no IQ3_S

Na A100-40G, com `38e525698`, `-ncmoe 36 --moe-cache-mib 20000 -b 4096 -ub 2048`,
KV Q8_0/Q8_0 e admissão parcial nos dois braços, três execuções por braço:

| `LLAMA_MOE_CPU_FALLBACK` | pp28k (t/s) | Decode pós-28k (t/s) | Hash da saída | Pico de VRAM |
|---|---:|---:|---|---:|
| 0 | 1.070,27 | 26,80 | `9fb4c487bd56` | 40.158 MiB |
| 1 | 1.057,99 | 11,24 | `198b55638dd8` | 40.174 MiB |

O fixture CPU passou 63/63 com NMSE 0. No GGUF real, o decode perdeu 58,06%
e o hash mudou. A flag permanece experimental na branch `feat/cpufb-verify`.
[Validação e faixas completas](strata-parity-closeout-20261122.md#fallback-cpu-keyed)
