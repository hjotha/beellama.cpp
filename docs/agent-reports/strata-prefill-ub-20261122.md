# Prefill: varredura de ubatch e poda da busca de fusão

Data: 2026-11-22 · branch `feat/strata-prefill` (de `1d0ecd6de`).
A100-40G, `parity6` (sm_80), Qwen3.8-Flash-Next Q2_0, `-ncmoe 2 --moe-cache-mib 2048
--parallel 1 --fit off`, Q8/Q8, `LLAMA_MOE_CACHE_PREFILL=1 LLAMA_MOE_MAP_DEVICE=1`.
Harness `bench.py`, 2 runs por config, `temperature 0`, `cache_prompt false`.

## 1. Varredura de ubatch

| config | pp4k | pp28k | pp28k+dec tg | VRAM pronto | pico |
|---|---:|---:|---:|---:|---:|
| `-b 2048 -ub 1024` | 1.449,7\* | 1.575,7 | 40,75 | 37.098 | 38.265 |
| `-b 2048 -ub 2048` | **1.762,9** | **1.831,7** | 44,24 | 37.528 | 38.904 |
| `-b 4096 -ub 4096` | OOM de grafo | OOM de grafo | — | 38.382 |

\* a primeira execução do `ub1024` pegou a carga fria (126 s); a segunda deu
1.551,9. O `pp28k` é estável nas duas (1.575,6 / 1.575,9).

`ub2048` = **+16,2%** no pp28k e **+21,6%** no pp4k (1.551,9 → 1.762,9), sem perda
de decode (44,24 vs 40,75 no mesmo caso; o decode não usa ubatch >1). Fica
**+0,4 GiB** de VRAM pronta.

`ub4096` carrega (38.382 MiB) mas todo request devolve `500 Compute error`; o log mostra
`cudaMalloc failed: out of memory` ao reservar **2.175 MiB** de grafo com `n_batch=4096`
(`failed to reserve graph buffers`). É OOM de grafo, não bug de kernel: 38,4 GB prontos deixam só
~2,5 GB e o grafo do ubatch 4096 precisa de mais. Não vale perseguir: renderia +9% no teto.

**`ub 2048` + DFlash2 não cabe.** Mesmo padrão, com o drafter: 39.226 MiB prontos e o grafo pede
+2.175 MiB → `500 Compute error` em todo request, inclusive com `-ncmoe 3` (que não reduziu o VRAM).
A recipe de decode continua `-ub 1024`; a de prefill puro pode usar `-ub 2048`.

### Modelo de dois pontos

`T(tokens) = fixo_por_ubatch + custo_por_token`

- 28 ubatches (ub 1024): 28.542 / 1.575,7 = 18,11 s
- 14 ubatches (ub 2048): 28.542 / 1.831,7 = 15,58 s
- diferença: 14 × fixo = 2,53 s → **fixo ≈ 181 ms/ubatch**
- custo por token: (15,58 − 14 × 0,181) / 28.542 = **457 µs/token ≈ 2.190 t/s**

Ou seja: metade da parede do pp28k em `-ub 1024` é custo fixo por ubatch
(28 × 181 ms = 5,1 s de 18,1 s). O restante é custo por token — compute do
dispositivo e submissão host proporcional ao número de nós, que cresce com a
largura do ubatch. Por isso `ub2048` rende +16% e não +100%: o fixo cai pela
metade, mas o custo por token fica igual. O teto com ubatch maior é ~2,2k t/s
(com `ub4096` seriam ~1,99k; +9% sobre o `ub2048`).

Consequência para a comparação: o prefill da Strata (4.629 t/s em 28k) está
**~2,1× acima do teto de ubatch** do Bee. A diferença restante é eficiência de
compute por token (GEMMs MoE, GDN, QSA), não agendamento.

## 2. Correção do diagnóstico anterior

O relatório `strata-a100-dflash2-20261122.md` leu `CLASS/mixed device_host_us`
(15,159 s, 96,7% da parede) como "submissão host". O `device_host_us` mede o
intervalo de `ggml_backend_graph_compute_async()`, que inclui qualquer
sincronização interna disparada por um nó (MoE fallback, top-k do QSA, QSA pool)
e o próprio enfileiramento quando o dispositivo já está ocupado. O modelo de
dois pontos acima mostra que **~28% da parede é custo fixo por ubatch** e o resto
acompanha os tokens — então a leitura correta é "misto: fixo por ubatch + compute
por token", não "96,7% host". A atribuição fina exige o perfil por op.

Para isso foi adicionado `LLAMA_OP_PROFILE=1` (default off): imprime
`[OP PROF] nodes=… submit_ms=… fuse_ms=… <op>=<ms> …` com o tempo de host por op
nos primeiros 15 ms de submissão de um grafo de prefill.

## 3. Poda da busca de fusão (patch)

`ggml_backend_cuda_graph_optimize` percorria, para **cada** `MUL_MAT_ID`, todos
os nós até o próximo par `MUL_MAT_ID, MUL_MAT_ID`, chamando
`ggml_cuda_match_shared_expert` em cada um — mas esse matcher exige que o
prefixo em `i` já seja `MUL_MAT_ID, MUL_MAT_ID, GLU`
(`ggml-cuda.cu:2261`, via `ggml_can_fuse_subgraph_ext`). Sem o prefixo, a busca
nunca podia casar. O patch testa o prefixo antes do laço
(`ggml-cuda.cu:6977`), preservando rotação e dependências de alocação.

Também foi condicionada à arquitetura GB10 a construção do mapa
`gb10_shared_q8_consumer_counts` (`ggml-cuda.cu:6096`), que só é lido sob
`cc == GGML_CUDA_CC_DGX_SPARK`: em sm_80 era uma passada + `std::map` por
execução de grafo, sem consumidor.

## 4. A/B `parity6` × `parity7` (A100, `-ub 2048`, 3 runs)

| config | pp4k | pp28k | VRAM pronto | pico | saída |
|---|---:|---:|---:|---:|---|
| `parity6` (sem patch) | 1.766,2 | 1.830,2 | 37.528 | 38.904 | `4cd63c3d` / `12941294` |
| `parity7` (patch) | 1.768,4 | 1.830,5 | 37.528 | 38.904 | **idênticas** |

**+0,02% no pp28k e +0,12% no pp4k — empate técnico.** O patch não muda o
gargalo, como o próprio Codex condicionou ("só contabilizo ganho se aparecer no
A/B"). Fica como remoção de trabalho provadamente impossível, com saída
bit-idêntica e `test-backend-ops -o MUL_MAT_ID` 2/2 no 4070.

## 5. `LLAMA_OP_PROFILE`: o que ele mostrou
Primeiras amostras em `-ub 2048` (grafo de 6.636 nós):

```
[OP PROF] nodes=6636 submit_ms=83.9 fuse_ms=1.9 REPEAT=7 MUL_MAT=77
[OP PROF] nodes=6636 submit_ms=67.8 fuse_ms=0.4 TOP_K=66
[OP PROF] nodes=6636 submit_ms=15.8 fuse_ms=2.0 MUL_MAT=13 MUL_MAT_ID=1
[OP PROF] nodes=6636 submit_ms=76.7 fuse_ms=0.5 FLASH_ATTN_EXT=75
```

Os `submit_ms` acompanham o número de nós e chegam a ~1 ms por nó em `MUL_MAT`,
`TOP_K` e `FLASH_ATTN_EXT`. Como `top-k.cu` não tem sincronização host (só um
`cudaMemcpy2DAsync` D2D) e os kernels são assíncronos, a leitura correta é
**back-pressure da fila CUDA**: o host enfileira até o limite e passa a bloquear
no launch enquanto a GPU está atrás. Ou seja, `device_host_us` ≈ tempo de GPU, e
o prefill é **limitado por compute do dispositivo** (~2.190 t/s assintótico),
não por submissão.

Isso fecha a atribuição: o que resta contra a Strata (4.629 t/s) é eficiência de
kernel por token — GEMMs MoE com Q2_0/MMQ, GDN e FA do QSA — não agendamento,
cópia ou PLE.

## 6. `--n-cpu-moe 0`: sem efeito

Se o tráfego de experts das camadas host fosse relevante, tirá-las do host
ajudaria. Não é o caso:

| config | pp4k | pp28k | VRAM pronto |
|---|---:|---:|---:|
| `-ncmoe 0 -ub 1024` | 1.440,5\* | 1.577,8 | 37.095 |
| `-ncmoe 2 -ub 1024` (referência) | 1.449,7\* | 1.575,7 | 37.098 |
| `-ncmoe 0 -ub 2048` | 1.762,1 | **1.835,5** | 37.522 |
| `-ncmoe 2 -ub 2048` | 1.751,7 | 1.826,2 | 37.528 |

\* primeira execução fria em cada config.

**+0,1% (ub 1024) e +0,5% (ub 2048), dentro do ruído**, e o VRAM pronto é
praticamente idêntico (37.095 vs 37.098) — o `ncmoe` não muda o placement real
neste GGUF. O custo de H2D de experts no prefill não é fator.

## 7. Sugestão #3 do Codex (reuso de gate/up no MMQ): pré-condição falha

O modelo **tem** gate/up separados (48 camadas com `ffn_gate_exps` e
`ffn_up_exps`, sem `ffn_gate_up_exps` empacotado), então o alvo existe. Mas
`ggml_cuda_mmq_fusion_supported` (`mmq.cuh:537`) só aceita
Q4_K/Q5_K/Q6_K/Q8_0/IQ4_XS — **Q2_0 não está na lista**. Habilitar o reuso exige
um caminho fundido novo para tiles Q2_0 (linhas intercaladas up/gate), não um
ajuste de orquestração. O ganho máximo seria a quantização Q8 da ativação e um
agrupamento por par, que são pequenos diante do GEMM: não compensa o risco agora.

## 8. Comandos

```bash
# varredura de ubatch (A100)
for ub in 1024 2048; do
  llama-server … -b 2048 -ub $ub … ; done

# perfil por op
LLAMA_OP_PROFILE=1 llama-server … -lv 4 2>&1 | grep '\[OP PROF\]'
```
