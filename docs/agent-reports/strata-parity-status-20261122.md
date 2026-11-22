# Paridade Strata × BeeLlama — status e pendências (2026-11-22)

Branch `main` = `6cefde1a5`. Release da integração: `collabosm-a100-parity12-20261122`
(sha256 `b89fb177…`). Todos os números abaixo são medidos na **mesma A100-40G** do comparativo
(40.960 MiB, 87 GB de RAM), com `--parallel 1`, `--fit off`, KV q8_0/q8_0, `temperature 0`,
`cache_prompt false`, 2 warmups e mediana de N execuções.

## 1. Onde estamos

| braço | prefill 4k | prefill 28k | decode curto | decode pós-4k | decode pós-28k |
|---|---:|---:|---:|---:|---:|
| **Bee Q2_0**, `-ub 2048`, `-ncmoe 2`, cache 2048 | 1.771 | **1.829** | 49,9 | 46,3 | 45,96 |
| **Bee Q2_0** + fusão GLU Q2_0 (`LLAMA_MMQ_GATE_UP_GLU=1`) | 1.884 | **1.947** | — | 46,2 | 45,61 |
| Bee Q2_0 + `LLAMA_MMQ_GATE_UP_SHARED_Q8=1` | 1.783 | **1.850** | — | 46,2 | 46,0 |
| Bee Q2_0 + DFlash2 n-max 7 | 1.344 | 1.388 | 58,4 | 83,3 | 81,6 (instável) |
| Bee Q2_0 + DFlash2 **n-max 8** | 1.342 | 1.384 | 74,0 | **95,4** | **87,3** |
| Bee Q2_0 + DFlash2 + **`--spec-dm-controller profit`** | 1.348 | 1.387 | 72,9 | **96,6** | **88,0** |
| **Strata 0.2.17** (mesmos pesos, INT8, 24.576 experts, DFlash2) | 2.394,7 | **4.629,3** | **85,9** | **67,4** | **75,0** |

Com **IQ3_S** (84 GB, 12 GB a mais que o Q2_0 em pesos):

| braço | prefill 4k | prefill 28k | decode 28k |
|---|---:|---:|---:|
| `-ncmoe 44`, cache 20 GB, sem patch | 555,7 | 740,9 | 26,0 |
| `-ncmoe 44`, cache 20 GB, `LLAMA_MOE_CACHE_PARTIAL_PREFILL=1` | 578,9 | **914,8 (+41,4%)** | 25,5 |

**Leitura:** o decode já está à frente da Strata nas três janelas (87-96 t/s contra 85,9/67,4/75,0).
A lacuna é **prefill**: 2,5× no Q2_0 e 5,1× no IQ3_S.

## 2. O que já foi entregue e medido (no `main`)

| entrega | flags | ganho medido | validação |
|---|---|---|---|
| QSA esparso Q8_0 | default | prefill +3,5%, decode 28k +9,9% | 28/28 backend-ops, A100 |
| MTP sidecar | `LLAMA_MTP_RESERVE_WEIGHTS=1`, `LLAMA_MTP_KV_STORE_ONLY=1` (recipe) | decode 4k/28k +37%/+31% | A100, aceitação ~80% |
| Residência de experts | `LLAMA_MOE_CACHE_PREFILL=1` (recipe) | prefill +3,5% (com cache 2048 hoje: **+7,6%**) | A100, fixture |
| Mapa imutável no device | `LLAMA_MOE_MAP_DEVICE=1` | +5,2% decode curto | A100 |
| Overlap de upload / ring pinned | `LLAMA_MOE_UPLOAD_OVERLAP`, `LLAMA_MOE_PINNED_RING` | ~0 | A100 |
| QSA indexer bounds / GDN conv+L2 | `LLAMA_QSA_INDEXER_BOUNDS`, `LLAMA_GDN_CONV_L2` | ~0 | A100 |
| Admissão parcial do cache | `LLAMA_MOE_CACHE_PARTIAL_PREFILL=1` | **+41,4% prefill IQ3_S** (idêntico bit a bit) | A100 + `deepseek2` 3/3 |
| MMQ gate/up Q8_1 compartilhado | `LLAMA_MMQ_GATE_UP_SHARED_Q8=1` | **+1,0-1,6% prefill** | bit-exact CUDA |
| Profiler por op de GPU | `LLAMA_OP_PROFILE_GPU=1` | diagnóstico | 4070 + A100 |
| PLE prefetch | `LLAMA_PLE_PREFETCH=1` | ~0 (A/B frio real) | A100 |
| DFlash2 (alvo Qwen4Exp) | `--spec-type draft-dflash --spec-draft-n-max 8` | decode 87-96 t/s | A100, greedy idêntico ao histórico |
| Fechamento do ubatch | `-ub 2048` é o teto útil | ver §3 | A100 |
| Correções de build | — | `tools/rkva-native-key-inspect` volta a compilar | build CPU |

## 3. Pendente agora (curto prazo)

1. **P1 do Codex — fusão gate/up/GLU Q2_0 no MMQ** (`feat/mmq-glu`, `9aef0d095`,
   `LLAMA_MMQ_GATE_UP_GLU=1`, default off): fixture **CPU 9/9 bit-exact** (inclui m=640,
   2048 tokens, 512 experts, top-10). **Falta a validação CUDA** — build sm_80 em andamento
   (116/385 objetos quando este documento foi escrito). Depois: `test-mmq-gate-up CUDA0` com a flag
   off/on, backend-ops 2/2, `deepseek2` 3/3 (com e sem graphs), e A/B na A100 contra o `parity12`
   (3 execuções, pp4k/pp28k). Só entra no `main` se sair do ruído.
2. **Frente expert-major do `MUL_MAT_ID`**: só design. Precisa provar que preserva o arredondamento
   da quantização e a ordem da redução antes de virar código.
3. **`-ub 4096` por código**: ficou bloqueado por VRAM. Medições: `-ub 3072` + cache = 1.843,6
   (+0,8%) com pico **39.974 MiB** (94 MiB de folga); `-ub 4096` só cabe **sem cache** (1.869,7,
   +2,1%), e o cache vale +7,6%. **Decisão: manter `-ub 2048`.** Se quiser o 4096, o caminho é
   reduzir o compute buffer do grafo de prefill (código), não placement.
4. **`GGML_CUDA_MMQ_J_MAX`**: sem ganho (default 1.830; `64` → 1.649 e muda numérica; `256` igual).
   Fechado como "não mexer".
5. **DFlash2**: adotar `n-max 8` ou `--spec-dm-controller profit` na recipe (87-96 t/s contra 81,6
   do n-max 7, que é instável). Falta só atualizar recipe/docs.

## 4. O que falta do plano original (Kimi, revisão de 2026-11-21)

### Prefill — tabela de oportunidades

| # | item | status | falta |
|---|---|---|---|
| 1 | Residência de experts no pp28k | **feito** (`LLAMA_MOE_CACHE_PREFILL=1`), +3,5% então, +7,6% hoje | nada |
| 2 | Overlap de upload + staging pinned limitado | **feito** (`LLAMA_MOE_UPLOAD_OVERLAP`, `LLAMA_MOE_PINNED_RING`), ~0 com cache 1024 | qualificação com misses reais + trace de DMA (não bloqueia) |
| 3 | Mapa imutável para grupo integral | **feito** (`LLAMA_MOE_MAP_DEVICE=1`), +5,2% decode curto | nada |
| 4a | Reusar agrupamento/entrada MMQ entre projeções | **feito em parte**: Q8_1 compartilhado (+1,0-1,6%); fusão GLU Q2_0 **em validação** | CUDA + A/B do `feat/mmq-glu` |
| 4b | Limites de visibilidade no indexer | **feito** (`LLAMA_QSA_INDEXER_BOUNDS`), ~0 | nada |
| 4c | IDs de QSA direto à FA | **não feito** — o gather esparso Q8 existe, o caminho direto não | implementar + NMSE/2 seq. |
| 4d | Fundir L2 de Q/K no GDN | **feito** (`LLAMA_GDN_CONV_L2`), ~0 | nada |

### Decode — tabela de oportunidades

| # | item | status | falta |
|---|---|---|---|
| Decode-3 | MTP sidecar n-max 2 pareado (reserva + store-only) | **feito**, +31% em 28k | superado pelo DFlash2 (87-96 t/s), que virou a trilha principal |
| Decode-5 | Estabilizar replay CUDA nas larguras reais de verificação | **não feito** | medir decisões de graph por largura de verificação |
| Decode-5b | Hidden do MTP alvo→sidecar via D2D | **não feito** | perdeu prioridade: o DFlash2 já está à frente da Strata no decode |

### O que o Kimi colocou "fora da fila imediata"

| item | status |
|---|---|
| Port Q8 | não feito |
| Reformulação GDN | não feito |
| MTP embutido neste GGUF | não feito (não existe `blk.48.*`; usa-se sidecar) |
| n-max 4 / patch ratio 4 | medido: n-max 4 é pior que 7/8; ratio 4 não testado |
| Cache compartilhado do draft | removido por corrida entre streams (decisão anterior, não reabrir) |
| CPU no miss | **protótipo BLOCK** (`feat/moe-cpufb`): perde write-back com >1 miss/token e escala `amax` |
| Prefetch SLRU | **feito como PLE** (`LLAMA_PLE_PREFETCH=1`): 0% no A/B frio |
| DFlash2 em trilha separada | **feito** e virou a melhor trilha de decode |

## 5. Lacunas que o plano original não cobria

1. **Tier de CPU para o IQ3_S.** `[OP GPU]` mostra `MUL_MAT_ID` 42,9% + `MUL_MAT` 29,8% do tempo de
   GPU, mas a soma dos grafos CUDA é **4,665 s** contra **43,8 s** de parede: **~89% do prefill do
   IQ3_S é o split de CPU**. É o item de maior retorno e o mais caro (protocolo de miss
   skip/compactação + escala `amax` + política de reposição; estimativa 6-9 semanas).
2. **Eficiência do `MUL_MAT_ID` no Q2_0.** Com `-ub 2048` e top-10/512 cada expert recebe ~40
   tokens; a fusão GLU é o primeiro passo, o expert-major é o seguinte.
3. **`-c 262144`.** O modelo suporta 256k e o KV q8 custaria ~4,1 GiB, mas nunca medimos Bee × Strata
   nessa janela; a Strata usa streaming de KV acima de 64K, o Bee mantém tudo no VRAM.

## 6. Próximos passos priorizados

| ordem | ação | ganho esperado | esforço | aceite |
|---|---|---|---|---|
| 1 | ~~Validar e medir a fusão GLU Q2_0~~ **feito: +5,3%, bit-exact, no `main` (`3fd591f0b`)** | +5,3% prefill | — | cumprido |
| 2 | Recipe: DFlash2 `n-max 8` ou profit | +7% decode pós-28k | S | docs/recipe + hash idêntico |
| 3 | Protocolo de miss do fallback CPU | até ~2× no IQ3_S | L (6-9 semanas) | NMSE ≤1e-6, ids sem colisão, aceitação ±1 p.p. |
| 4 | Expert-major no `MUL_MAT_ID` | desconhecido (2-5%?) | M-L | decomposição numérica provada |
| 5 | QSA IDs direto à FA (4c) | desconhecido | M | conjunto de células idêntico por query |
| 6 | Replay CUDA por largura de verificação (Decode-5) | estabilidade, não throughput | S-M | sem quedas por recompilação de graph |
