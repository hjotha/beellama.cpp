# A100: shim Strata × MoE nativo e qualidade QSA 4×16 (25/11/2026)

## Escopo

- Host de controle: `gokaya` (`192.168.1.57`).
- GPU: `NVIDIA A100-SXM4-40GB` (Colab, `swa-pp28k-24`).
- Modelo: `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0` (2 shards, 48 camadas).
- Prefill: `pp28k+dec`, 28.547 tokens reais, 64 gerados, seed 42, temp 0.
- A/B de throughput: `-b 4096 -ub 4096`; qualidade: `-b 2048 -ub 2048`.
- Receita base: `-ngl 99 -c 32768 -ncmoe 2 --moe-cache-mib 2048 --parallel 1 --fit off -fa on -ctk q8_0 -ctv q8_0 --load-mode mmap --lazy-mode on`, draft DFlash2 com `--spec-draft-n-max 5`.

Este relatório fecha a fila 2–4 do handover de 25/11. O Passo 1
(`inputrows`/PLE) já foi promovido em `1fc47b633`.

## 1. Correção do registro: o braço de 2.699 t/s não usou o shim

`measure-combo.py` exige a linha `Strata raw Q2_0 routed down active` quando
`enabled=direct`. Nos voos de 25/11 (09:55 e 10:55), o braço direto produziu
`2699.67 / 2698.29 / 2699.98 t/s`, mas terminou com:

```text
RuntimeError: combo-d2d-reviewed-direct-capacity-4096-1: unexpected fusion activation False
```

Causa: `ggml_cuda_strata_raw_q2_enabled()` exige `GGML_CUDA_DISABLE_GRAPHS`
além de `GGML_CUDA_STRATA_RAW_Q2=1`; o braço `direct` roda com CUDA Graphs
ligados. O JSON guardou as amostras antes da asserção, mas o shim nunca foi
enabled nesse braço.

Portanto, **2.699 t/s é um resultado válido de `inputrows + bounds` com o MoE
nativo (K_OUTER), sem o shim**. O ganho isolado de 9,41 ms vs 12,22 ms por
iteração MoE não estava presente nesse número.

## 2. A/B do shim no modelo completo

Quatro braços, três amostras de `pp28k+dec` cada, A100-40GB, hash de saída
`fec827d6` em todos:

| Braço | Flags `inputrows`/PLE | CUDA Graphs | Shim | Média (t/s) | Amostras |
|---|---|---:|---:|---:|---|
| `stock` | off | on | off | **2597,50** | 2601,56 / 2594,17 / 2596,76 |
| `direct` | on | on | off | **2696,85** | 2696,31 / 2697,32 / 2696,93 |
| `stock-nograph` | off | off | off | **2588,59** | 2591,36 / 2586,00 / 2588,40 |
| `direct-nograph` | on | off | **on** | **2631,64** | 2633,03 / 2631,47 / 2630,42 |

- `direct` vs `stock`: **+3,82%**.
- `stock-nograph` vs `stock`: **-0,34%** (custo de desligar grafos).
- `direct-nograph` vs `direct`: **-2,42%** (-65,21 t/s).
- `direct-nograph` vs `stock-nograph`: **+1,66%** (ganho do shim).
- Ativação confirmada no `direct-nograph`:
  `ggml CUDA: Strata raw Q2_0 routed down active on SM80`.
- Soma dos `MTPPHASE phase=sync` grandes (`n_tokens > 8`):
  **631.970 µs** no `direct` vs **799.415 µs** no `direct-nograph`
  (+167.445 µs). O shim adiciona trabalho de host/launch que anula o ganho
  de kernel no modelo completo.

**Decisão:** o shim não é promovido. O caminho nativo com
`GGML_CUDA_MMQ_SM80_Q2_DOWN_K_OUTER=1` já é mais rápido no prefill completo.
O resultado de microbenchmark continua válido como caracterização do kernel,
mas não se traduz em throughput de ponta a ponta.

## 3. QSA 4×16: qualidade em `ub 2048`

Comparação `off.bin` (8×8) × `on.bin` (4×16), mesmo binário e mesmas entradas,
64 posições pontuadas:

| Métrica | Valor |
|---|---:|
| `argmax_agreement` | 1,0000 |
| `max_abs_delta_nll` | 6,820423 |
| `mean_delta_nll` | 0,126101 |
| `max_kl_reference_to_candidate` | 1,4765e-3 |
| `mean_kl_reference_to_candidate` | 2,8136e-5 |
| Controle `off` × `repeat` | 0,0 (bit-exact) |

O ganho medido do 4×16 é **+56,93 t/s (+1,99%)**, abaixo do piso de promoção
de **3%** do handover de 24/11. Mesmo que uma variante bit-exact fosse
implementada, ela não seria promovível por throughput. A divergência de logits
é real e reproduzível; a variante permanece experimental fora de `main`.

A implementação bit-exact descrita no handover (compartilhar K/V entre dois
subgrupos de 4 queries mantendo a união 8×8 e os tiles de 32 linhas) exigiria
uma especialização nova do kernel MMA; não foi iniciada porque o teto de ganho
conhecido (+1,99%) está abaixo do critério de promoção.

## 4. Qualidade em A100-40GB: `UB=2048`

O OOM de `teacher-logits` em `-ub 4096` era aritmético:

| Item | MiB |
|---|---:|
| Modelo CUDA0 | 34.259,80 |
| KV cache principal | 408,00 |
| KV cache do indexer | 102,00 |
| Recurrent state (RS) | 112,57 |
| MoE cache | 1.351,32 |
| **Total antes do compute** | **36.233,69** |
| Livre em 40.013 MiB | **3.779,31** |
| Graph `ub 4096` (falhou) | 4.330,99 |
| Graph `ub 2048` (OK) | 2.166,24 (native) / 2.100,06 (qsa4x16) |

`run-native-quality.sh` e `run-qsa4x16-quality.sh` agora aceitam
`UB` (default `4096`) e `CTX` (default `32768`), usados em `-c/-b/-ub`.

Validação em A100-40GB:

- `UB=2048 bash run-native-quality.sh native`: completa sem OOM; shim ativo;
  `off`, `repeat` e `on` **bit-exact** (`max_abs_delta_nll=0`,
  `max_kl=0`, `argmax=1,0`).
- `UB=2048 bash run-qsa4x16-quality.sh`: completa sem OOM; `off`=`repeat`;
  `on` divergente conforme a tabela da seção 3.
- Em A100-80GB, `UB=4096` continua válido.

## 5. Artefatos

No `gokaya`:

- `/tmp/opencode/shim-results/shim-pp28k.json` e os quatro logs do A/B.
- `/tmp/opencode/quality-results/quality-native/native.json`,
  `quality-qsa4x16/qsa4x16.json`, `quality-qsa4x16/repeat.json` e os logs.
- Scripts: `/tmp/opencode/strata-prefill-parity/shim-flight.sh`,
  `quality-flight.sh`, `measure-shim.py`, `run-native-quality.sh`,
  `run-qsa4x16-quality.sh`.

Na `.157` (execução):

- `shim-results-a100-40gb-20261125T144735Z.tgz`
- `quality-results-a100-40gb-20261125T155418Z.tgz`

## 6. Reprodução

```bash
# A/B do shim (4 braços, JSON novo)
FLIGHT_SCRIPT=shim-flight.sh MAX_ROUNDS=6 bash run-paired-a100-collabosm.sh

# Qualidade em 40 GB
UB=2048 bash run-native-quality.sh native
UB=2048 bash run-qsa4x16-quality.sh
```

O `measure-shim.py` corrige a asserção do harness: o shim só é exigido em
`direct-nograph`; em `direct` (grafos ligados) ele deve estar ausente.
