# Relatório A2 — kernel `pass1` de prefill RKVA no Xbox

Branch: `perf/rkva-pass1-prefill-20261026` (clone `/home/hjotha/beellama-agents/pass1-kernel`,
base `40f22aa6b`). Trabalho **não** foi validado em hardware: nenhum acesso ao Xbox nem à
RTX 4070 foi feito. Tudo abaixo é medição de artefatos existentes, compilação DXC isolada na
máquina 193, emulação numérica em Python e um TU de teste C++ compilado com `g++` neste host.

## 1. Caminho quente real (Passo 1) — identificado, não presumido

O `pass1` de produção em prefill **não** é um kernel "bulk nq>9". O host envia op22 com no
máximo 8 queries por segmento (`chunk_limit = nq > 9 ? 8 : nq`, `ggml-remote-attn.cpp:2299`
para descriptor revision 3), e o app escolhe o tile de 2 queries:

* Shader: `tools/xbox/shaders/rkva/rkva_range_attn_kvarn4_mq.hlsl`, entry `CSRangeAttnKvarn4Mq`,
  `RKVA_MQ_QT=2` → PSO `attn_kvarn4_mq2`, nome de bucket `kvarn4_mq2_fine`.
* Root signature `range_attn_fmt_opt_root_`, 12 constantes (`cbuffer Params`), SRV t0=query,
  t1=K, t2=V, t3=positions, UAV u0=partials, u1=output.
* `numthreads(128,1,1)` (2 waves de 64) · LDS estático **12.384 B**
  (`gs_max`+`gs_den` 96 B + `gs_vec` 3072 floats) · 48 barreiras estáticas na entry.
* Dispatch: `Dispatch(n_head_kv=4, ceil(nq/2)=4, split_k=64)` = **1024 grupos × 2 waves** por
  segmento de 8 queries (`remote-attn-engine.cpp:3114`).
* Política de split "fine" (`rkva-range-profile.h:314`): `target_rows=256`, `max_splits=64`,
  `rows` arredondado a múltiplo de 8 → com 27136 linhas: `split_rows=424`, `split_k=64`.
* KV: descriptor revision 3 — registros KVarN4 de 8960 B (nibbles K `[dim][pos/2]`,
  escala/zero/other F16), estágio rotado F16 de 384 linhas, tail ring ORIGINAL-F16 de 256 linhas.
  Dequant por elemento `precise float affine = float(quant)*scale + zero; return affine*other;`.

Evidência de que é esse o PSO em produção (artefato de perfil do Codex,
`wf256-fp8-mapped-mq295-r1/profile-lines.json`, linha `remote_attn gpu_profile`):

```
"q4_kernel_mode":2,"q4_kernel_mode_name":"q4_gqa_fine"
"mode_map":"1:kvarn4_mq4_balanced 2:kvarn4_mq2_fine 3:per-query kvarn4_gqa_coarse"
"mq2":{"available":true,"status":"ready","calls":56320},"mq4":{"available":true,"status":"ready","calls":0}
bulk_calls=5856 bulk_segments=60400 bulk_queries=454640
```

Dimensões reais (`mq295-gpu-bucket-comparison.json`, 4112 amostras de fim de faixa):

```
bucket=attention/kvarn4-prefix/optimized-kvarn4_mq2_fine/nq2-16/r16k-32k/warm
samples=4112 nq_mean=8.000 rows_mean=27135.767 split_k_mean=63.837
pass1_us mean=2725.694 p50=2726.336 max=2878.035 sum=11208052.812
reduce_us mean=59.572  readback_us mean=1.222  total_us mean=2786.489
pass1 share of segment total = 97.82%
gpu_total_mean_change_pct(294->295): 0.03439200640740214   (MQ2 295 ≈ 294: ruído)
```

Ou seja: por segmento de 8 queries, 97,8% do tempo de GPU é `pass1`, 2,1% `reduce`. Isso
confirma e refina o briefing (96,7% num envelope de 256 queries). O briefing cita 65,30 ms de
`pass1` por envelope 256q×1 camada; com 8 queries/segmento isso equivale a ~1,92 ms/segmento,
enquanto o bucket medido acima dá 2,726 ms/segmento. As duas fontes não são reconciliáveis sem
o trace cru; **uso o bucket (4112 amostras, medido e versionado) como referência de custo por
chamada** e trato os 65,30 ms/237 t/s do briefing como "derivado de outra janela". Nada no
diagnóstico abaixo depende do valor absoluto.

## 2. Análise de gargalo (Passo 2)

Recompilei o alvo de produção com o **mesmo DXC** e obtive o DXIL **byte-idêntico** ao commitado
(prova de que a cadeia de compilação é reproduzível e de que o baseline é o real):

```
committed bytes 118452 sha256 9d81c3beba4e36756c89c6bc70c5807cedb990be67fba683fc714ff7a1c5b431
recompiled bytes 118452 sha256 9d81c3beba4e36756c89c6bc70c5807cedb990be67fba683fc714ff7a1c5b431
IDENTICAL
```

Contagem de instruções do laço quente (tile compartilhado de 8 linhas de KV, 2 queries × 6
cabeças GQA), por linha de KV e por lane, via `tools/xbox/tests/rkva_dxil_loop_cost.py`
(DXIL em classes RDNA2 wave64: `dot4` = 4 FMA, `waveActiveOp` = 6 moves + 6 adds):

```
== mq2_baseline.txt
   loop header %993  body blocks 59  dots/iter 96  rows/iter 8
   per KV row:  TOTAL 512.0   fma_class 237.0   reduce 144.0   sfu 15.0   int 89.1
   useful FMA per row (QK+PV)  96.0  => 40.5% of fma_class, 18.8% of TOTAL
```

Orçamento por linha de KV (lane), derivado dessa contagem:

| item | instr/linha | fração |
| --- | ---: | ---: |
| `WaveActiveSum` (6 adds + 6 shuffles × 12 QG) | 144,0 | 28,1% |
| QK+PV úteis (12 QG × 4 dims × 2) | 96,0 | 18,8% |
| resto do FMA (`s*v`, `body*=old`, `s*scale`, dequant) | 141,0 | 27,5% |
| int/misc (endereços, extracts, phi) | 89,1 | 17,4% |
| SFU (`exp`, 15×) | 15,0 | 2,9% |
| cvt F16→F32 / int→F32 | 12,5 | 2,4% |
| branch/cbuf/loads | 14,4 | 2,8% |

Composição do laço por bloco (DXIL, por iteração de 8 linhas = 2.872 instruções): dequant das
duas metades 575 (20,0%), 24 blocos de cabeça (dot+redução+softmax+PV) 24×81 = 1.944 (67,7%),
header com os 87 `phi` 87 (3,0%), demais 266 (9,3%).

Conclusões:

1. **Não é banda**: o laço lê/dequantiza uma linha de KV uma vez por (kv_head, par de queries)
   e reusa em 12 (query,head); os 31 MB de KV por camada não explicam 2,7 ms.
2. **É issue/ALU**: 512 instruções wave64 por linha de KV. 1.024 grupos × 424 linhas = 434.176
   visitas de linha por segmento → **222 M instruções wave64 por segmento**. Com 52 CUs a
   1,825 GHz e 1 instrução wave64/cycle/CU (64 lanes FP32 por CU = pico de 12,15 TFLOPS),
   o piso de issue é **≈2,34 ms** contra 2,726 ms medidos → **86% de eficiência de issue**.
   O kernel é essencialmente limitado por contagem de instruções, não por latência/ocupação.
3. **A redução é a maior linha isolada**: 144 de 512 (28,1%). Ela é estrutural, não descuido:
   o corpo do VGPR exige 4 dims/lane (12 QG × 4 = 48 VGPR de `body`, +48 de Q, +24 de
   max/denom ≈ 120 arquiteturais), o que força redução de 64 lanes (6 passos) por dot.
   Mapeamentos alternativos foram quantificados e perdem:
   * 8 dims/lane com 2 queries empacotadas nas duas metades da wave (redução de 5 passos
     compartilhada): QK/PV/dequant dobram por lane; 312 vs 354 instr/linha → ganho ≤1,13× e
     reordena a soma FP32 (não bit-exato). Rejeitado.
   * mq4 (4 queries/tile, modo "balanced"): não reduz a redução e estoura VGPR
     (24 QG × 4 dims = 96 só de `body`) — coerente com o **−34,09%** medido em produção
     (`performance-summary.json`, `balanced_rejected.prefill_tps=98.58` vs 149,58).
   * `dot2`/FP16 empacotado: só ataca 96 das 512 instruções e quebra exatidão.
4. **WHT/barreiras e `split_k` não são o gargalo**: os blocos de WHT+merge somam ~3.260
   instruções por grupo contra ~127.000 do laço de linhas (≈2,5%), o que explica o ganho nulo
   ao reduzir as barreiras do WHT de 17→9. Reduzir `split_k` só mexe nos 2,1% de `reduce`.
5. **Pressão de registradores é visível no DXIL, a alocação não**: o header do laço
   (`; <label>:993`) tem **87 nós `phi`** — 87 valores vivos que atravessam a iteração
   (`body` 12×4 = 48, `maximum` 12, `denom` 12, mais endereços/contadores). Isso é o piso
   arquitetural de VGPRs e é o que fixa 4 dims/lane (8 dims/lane exigiria 96 só de `body`).
   A alocação final de VGPR (e a ocupação) acontece no compilador do driver na criação do PSO e
   **não** é exposta por DXC/DXIL: fica como pendência de hardware (RGA ou equivalente na 193).

Reuso de KV: cada linha é dequantizada 1× por (kv_head, par de queries) → 4× por segmento de 8
queries → **128× por envelope de 256 queries** (mesma camada). É o que motiva o tile de queries;
mas como o dequant é só ~8% das instruções, aumentar o tile não compensa (mq4).

## 3. Variante implementada (Passo 3)

`RKVA_MQ_RESCALE_SKIP=1` → PSO separado **`attn_kvarn4_mq2_skip`**, default OFF, atrás do arquivo
de flag LocalState `rkva-mq-rescale-skip.flag`.

Ideia: no tile rápido, quando o máximo da meia-linha não cresce
(`half_max <= maximum[i]`), o original calcula `old_weight = exp(0) = 1.0` e faz
`body *= 1.0` e `denom*1.0 + sum`. Isso é identidade exata; o braço "flat" remove essas
multiplicações, o `exp(0)` e o recálculo de `max`, mantendo as mesmas exponenciais, ordem de
acumulação, splits, decodificação K/V e layout de parciais. **É bit-exato, não aproximado.**

Evidência DXIL do braço rápido (build de medição forçando o caminho flat):

```
== mq2_fastonly.txt
   per KV row:  TOTAL 471.5   fma_class 216.0   reduce 144.0   sfu 12.0   int 72.6
   useful FMA per row (QK+PV)  96.0  => 44.4% of fma_class, 20.4% of TOTAL
```

→ **−7,9% instruções por linha** (−40,5 instr), −20% de `exp`, −18,5% de int/misc, redução
inalterada, 48 barreiras e 12.384 B de LDS inalterados. A fração de meias-linhas que cresce o
máximo é ~`H(106)/106 ≈ 5%` por QG, então a média esperada fica em ≈474 instr/linha (−7,4%).

Arquivos:

* `tools/xbox/shaders/rkva/rkva_range_attn_kvarn4_mq.hlsl` — guarda `RKVA_MQ_RESCALE_SKIP`
  (default 0) e o braço novo.
* `tools/xbox/shaders/rkva/rkva_range_attn_kvarn4_mq_skip.hlsl` — wrapper que define o macro e
  inclui o shader acima (sem duplicar código).
* `tools/xbox/shaders/generated/rkva_range_attn_kvarn4_mq2_skip_dxil.h` — DXIL gerado na 193
  (126.656 B, sha256 do payload `6af837b580cacfa0…`).
* `tools/xbox/scripts/compile-rkva-range-shaders.sh` — alvo `attn_kvarn4_mq2_skip`.
* `tools/xbox/uwp/rkva/remote-attn-engine.{h,cpp}`, `remote-attn.cpp`, `rkva-range-profile.h` —
  PSO, flag, seleção, contador `kvarn4_mq_skip_calls`, bucket próprio e campos de perfil.
* `tools/xbox/tests/rkva_mq_rescale_skip_exact.py`, `tools/xbox/tests/rkva_dxil_loop_cost.py`,
  `tools/xbox/tests/rkva_dxil_block_mix.py`, `tools/xbox/scripts/run-rkva-prefill-kernel-ab.py`.
* `docs/rkva-prefill-rescale-skip.md`.

### Byte-identidade do caminho de produção (obrigatória)

Payload DXIL (não o comentário do header) dos alvos existentes, HEAD vs. regenerado:

```
 rkva_range_attn_kvarn4_mq2_dxil.h        HEAD 9d81c3beba4e3675  NEW 9d81c3beba4e3675  identical=True
 rkva_range_attn_kvarn4_mq4_dxil.h        HEAD ef2878458bd37b6f  NEW ef2878458bd37b6f  identical=True
 rkva_range_attn_kvarn4_mq2_skip_dxil.h   NEW 6af837b580cacfa0  bytes=126656
```

O único diff nos dois headers existentes é a linha `// Source SHA256:` (hash do fonte mudou por
causa do `#if`); os bytes de DXIL são bit a bit os mesmos.

## 4. Validação offline (o que foi realmente executado)

1. **Prova bit a bit da variante** — emulação binary32 das duas formulações (uma operação por
   statement, sem FMA implícito), incluindo `denom==0`, empate exato no máximo, crescimento de
   1 ulp, escores monótonos/decrescentes, subnormais, `±0` e splits completos de 106 meias-linhas:

```
$ python3 tools/xbox/tests/rkva_mq_rescale_skip_exact.py --trials 400
rkva_mq_rescale_skip_exact: 760 replay cases, bit-identical body/denom/maximum; mutation sensitivity confirmed
```

   (o "mutation sensitivity" garante que o harness detecta uma mudança de 1 ulp; sem isso a
   prova não valeria nada).
2. **Teste CPU do perfil/JSON** (TU único, `g++`, `nice -n 19 ionice -c3`, gate `pgrep` sem
   processos do Codex no momento):

```
$ g++ -std=c++17 -Wall -Wextra -Werror -O1 tools/xbox/tests/rkva_range_profile_cpu.cpp -o .oc-artifacts/bin/rkva_range_profile_cpu
$ .oc-artifacts/bin/rkva_range_profile_cpu > .oc-artifacts/bin/profile.json
rkva_range_profile_cpu: ok
$ python3 -m json.tool .oc-artifacts/bin/profile.json > /dev/null && echo JSON_OK
JSON_OK
```

   JSON contém `"mq2_rescale_skip":{"available":true,"status":"ready","calls":4}` e o bucket
   `…/optimized-kvarn4_mq2_skip/…` é coberto por asserção nova.
3. **DXIL na 193** (`dxc.exe` 1.8.2502.11, Windows SDK 10.0.26100.0, `cs_6_0`): baseline
   reproduzido byte a byte, alvo novo compilado, contagens acima.
4. **Script de hardware** em modo preview (sem `--execute`) — ver §5.

O que **não** foi possível offline: executar o kernel, medir VGPR/ocupação real, validar
qualidade MTP/logits, compilar o app UWP (sem MSVC e por regra) e compilar qualquer alvo CUDA
(Codex ativo/regra de 20 W).

## 5. Script para a janela de hardware (Passo 4)

`tools/xbox/scripts/run-rkva-prefill-kernel-ab.py`:

* **Inerte por padrão** (preview JSON, exit 4). Só roda com `--execute`.
* **Recusa** (exit 2) se `pgrep -af 'llama-server|bench-rkva|test-rkva|ninja|cmake --build'`
  achar processo alheio, ou se faltar argumento/flag helper.
* Não instala nada: pressupõe o pacote já instalado pelo Claude; usa dois comandos fornecidos
  pelo operador (`--enable-cmd`/`--disable-cmd`) para criar/remover o flag LocalState, e sempre
  restaura OFF no `finally`.
* Sequência: OFF → `bench-rkva-range.py --execute` (fine + FP8 mapped + tcp-prefetch + profile),
  ON → mesma coisa, restaura.
* Coleta: `profile-lines.json` de cada braço (buckets `…mq2_fine…` vs `…mq2_skip…`),
  `kvarn4_multi_query.mq2_rescale_skip.calls`, `pass1_us` mean/p50/p95, `split_k`, `active_rows`,
  e os IDs gerados dos dois braços para paridade **byte a byte**.
* Status `pass` só se: o braço ON usou o kernel novo (`calls > 0`), os dois buckets existem e os
  IDs gerados são idênticos. Tudo em JSON (`--json`).

Preview real executado:

```
$ python3 tools/xbox/scripts/run-rkva-prefill-kernel-ab.py --json .oc-artifacts/ab-preview.json
 "mode": "preview",
 "kernel_on": "attn_kvarn4_mq2_skip / bucket attention/kvarn4-prefix/optimized-kvarn4_mq2_skip/nq2-16/r16k-32k/warm",
 "foreign_processes": []
```

## 6. Estimativa de ganho (ESTIMATIVA, não medida)

Base: `pass1` = 2.725,7 µs/segmento (medido), 97,8% do segmento, kernel limitado por issue a 86%
de eficiência.

| cenário | instruções/linha | `pass1`/segmento | envelope 256q×1 camada | prefill XXXL |
| --- | ---: | ---: | ---: | ---: |
| atual (medido) | 512,0 | 2.725,7 µs | 65,3 ms (briefing) | ~150 t/s |
| variante (derivado) | ≈474 (−7,4%) | ≈2.532 µs | ≈60,6 ms | ≈158–160 t/s |

Ganho estimado: **−5% a −7,5% no `pass1`**, **+5% a +7% no prefill XXXL** (150 → 158–160 t/s),
com `reduce`/`unpack`/`pack` inalterados. É estimativa derivada de contagem de instruções e da
fração de meias-linhas que cresce o máximo; **não** há medição de hardware nesta entrega.

## 7. Pendências de hardware / riscos

Pendente medir:

1. `pass1` OFF vs ON nas dimensões do fim da faixa (script §5) — confirma ou refuta a estimativa.
2. Paridade byte a byte de IDs gerados e de logits no mesmo prefixo congelado (o script já checa
   IDs; logits exigem o harness do Claude).
3. VGPR/ocupação reais do PSO (RGA ou equivalente na 193) para saber se o kernel é issue-bound
   como o modelo indica; se houver folga de VGPR, o próximo passo é reduzir LDS de 12.384 B para
   6.192 B (merge por cabeça) e testar 8→10 ondas/CU.
4. Custo real da divergência do `if (half_max > maximum[i])`: o compilador não sabe que a
   condição é uniforme na wave. Se o ramo custar caro, a alternativa é um único `if` por
   meia-linha cobrindo os 12 QG (descarta menos instruções, custa 1 branch).

Riscos:

* O flag é lido no HELLO a partir de `g_snapshot_local_folder`; se a pasta ainda estiver vazia
  nesse ponto, o flag não é visto (mesmo comportamento do `rkva-fp8-gpu-off.flag` existente).
  Não alterei essa ordem — validar no primeiro run de hardware.
* O app UWP não foi compilado (sem MSVC nesta máquina e build proibido). As mudanças de C++ são
  pequenas e espelham padrões existentes (PSO opcional + status + contador + JSON), mas
  **compilação do app é pendência**.
* Se o ganho real vier menor que o previsto, a causa provável é custo de branch/reconvergência
  ou o kernel não ser tão issue-bound quanto o modelo (clock sustentado menor em UWP). Nesse caso
  a variante não deve ser promovida a default.
* Nenhuma mudança de protocolo RKVA, de wire, de layout de parciais ou de matemática de produção.

## 8. O que não fiz (e por quê)

* Não toquei em `ggml/include/ggml.h`, `ggml/src/ggml-backend.cpp`,
  `ggml/src/ggml-cuda/ggml-cuda.cu`, `src/llama-graph.cpp`, `ggml/src/ggml-mask-pack.h`
  (região do Codex). O diff de kernel é só no app/shader/profiling.
* Não rodei build completo, CUDA, `llama-server`, nem acessei Xbox/4070.
* Não implementei a variante "pair-pack" (2 queries empacotadas em metades de wave): a contagem
  mostrou 1,13× no melhor caso e ela reordena a soma FP32; registrei a análise em §2.3 em vez de
  gastar risco.
* Não promovi nada a default: a variante é opt-in e o caminho de produção está byte-idêntico.

## 9. Comandos de reprodução dos artefatos

```bash
# DXIL + assembly de qualquer alvo (usa DXC da 193 num diretório próprio)
python3 .oc-artifacts/dxc/dxc193.py -T cs_6_0 -E CSRangeAttnKvarn4Mq -D RKVA_MQ_QT=2 -Fc \
  tools/xbox/shaders/rkva/rkva_range_attn_kvarn4_mq.hlsl -Fo .oc-artifacts/dxc/mq2_baseline.dxil

# custo por linha de KV do laço quente
python3 tools/xbox/tests/rkva_dxil_loop_cost.py .oc-artifacts/dxc/mq2_baseline.txt

# prova bit a bit da variante
python3 tools/xbox/tests/rkva_mq_rescale_skip_exact.py --trials 400

# teste CPU de perfil/JSON
g++ -std=c++17 -Wall -Wextra -Werror -O1 tools/xbox/tests/rkva_range_profile_cpu.cpp -o /tmp/rkva_profile
/tmp/rkva_profile > /tmp/profile.json && python3 -m json.tool /tmp/profile.json > /dev/null

# A/B de hardware (inerte sem --execute)
python3 tools/xbox/scripts/run-rkva-prefill-kernel-ab.py --help
```
