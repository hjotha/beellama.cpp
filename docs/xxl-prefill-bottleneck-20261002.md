# Gargalo do prefill XXL — Qwen3.8-27B, RTX 4070 + Radeon 780M (2026-10-02)

Modelo: `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf`. Código analisado: `main`
em `afd858d2e`, com a alteração não commitada do guard de prompt cache
(25% → 1%) em `tools/server/server-task.cpp`. Evidência bruta:
`/home/hjotha/beellama-mixed-kv-20261001-130910/`.

Legenda: **[O]** observado (comando/log real) · **[D]** derivado por cálculo ·
**[H]** hipótese · **[R]** recomendação.

## 1. Diagnóstico

O gargalo é o kernel de atenção na Radeon (`FLASH_ATTN_EXT` Q4, coopmat1), cujo
custo cresce linearmente com o histórico. As cópias CUDA↔Vulkan têm custo fixo
por ubatch e pesam pouco em contexto longo.

- [O] Tempo por ubatch de 256 tokens no XXL: 20 W/2700 MHz
  **0,664 s + 68,6 ms por 1k de contexto**; sem controles AMD
  **1,348 s + 119 ms/1k**. Fontes: `tier-cache-break-even-20261002/XXL/server-XXL.log`
  e `tier-cache-break-even-20261002-xxl-no-amd-controls/XXL/server-XXL.log`.
- [O] Calibração com 20 W e MTP desligado (`phase3-long`): N=0 remotas
  **0,259 s + 3,70 ms/1k**; N=1 **0,339 s + 6,01 ms/1k**; N=9
  **0,784 s + 54,7 ms/1k**.
- [D] Cada camada remota custa ~**5,9 ms/1k por ubatch**; cada camada CUDA,
  ~**0,23 ms/1k**: ~25× mais cara na Radeon. A 65k de histórico, a atenção
  Radeon é ~90–95% do ubatch; as cópias, ~3–4%.
- [O] O "2.000 Gb/s, x1 2.5 GT/s" está na porta raiz virtual do túnel
  Thunderbolt (`0000:00:03.1`). A 4070 (`0000:06:00.0`) está em 8 GT/s x4; o
  `boltctl` mostra TB3 40 Gb/s. Medição anterior: 2,17 GiB CUDA→Vulkan em
  1,22 s (~1,78 GiB/s).

Correções ao contexto anterior:

1. Houve **dois** resets: boot -2 (20 W) com
   `11:58:55.173578 ... Fence fallback timer expired on ring comp_1.2.0` e
   boot -1 (15 W) com `12:45:02.659717 ... comp_1.3.0`. A primeira rodada de
   20 W chegou a 72.704/73.000 em 899,21 s e foi encerrada pelo
   `--request-timeout 900` do runner, sem crash.
2. O XXL "sem controles AMD" não é controle limpo: o serviço
   `qwen35-4b-mtp-8092` (`--device Vulkan0`) atendeu 106 requisições (~588 s de
   prefill) na mesma Radeon entre 13:00 e 13:27. No boot -2 esse serviço
   estava parado (`Stopped ... 09:38:53`).
3. `build-optimized/bin` foi recompilado entre 13:31 e 13:34; a produção
   reiniciou às 13:43. As quatro rodadas XXL usaram o build anterior.

Causa-raiz: falta de VRAM na 4070 a 131072 manda ~10–12 das 16 atenções para
a Radeon, onde a atenção é ~25× mais cara por camada, executa em série com a
4070 (que fica ociosa) e é limitada pela potência da APU.

## 2. Caminho de uma camada remota (por ubatch)

```
CUDA (4070)                                  host / GTT (UMA)            Vulkan0 (780M)
attn_norm → build_qkv → q/k norm → RoPE  [src/models/qwen35.cpp:335-395]
Qcur F32 6,0 MiB; Kcur/Vcur F32 1,0 MiB cada
  ├─ synchronize(CUDA) ─┐  (cpy_tensor_async recusa CUDA→Vulkan)
  │                     ├─ synchronize(Vulkan)
  ├─ D2H pageable ─────►│ malloc temp   [ggml-backend.cpp:490-511, 1893-1921]
  │                     ├─ memcpy UMA ─────────────────────────► set_rows q4_0 (KV em GTT)
  │                                                              flash_attn_mask_opt
  │                                                              FLASH_ATTN_EXT coopmat1
  │                                                              Br=16 Bc=64, 384 WGs, split_k=1
  │                     ┌─ fence + spin ◄──────────────────────── saída F32 6,0 MiB
  ├◄─ H2D pageable ─────┘
sigmoid(gate)·attn → wo → residual → FFN → 3 camadas DeltaNet (CUDA)
```

- Escolha da camada: `local_split` em `qwen35.cpp:~431`; os nós de
  `build_attn` vão para a Vulkan por `ggml_backend_sched_set_tensor_backend`.
- KV remoto = as **N primeiras** full-attention (`src/llama-model.cpp:2813-2826`,
  `filter_remote_standard` em 3270-3300).
- `build_attn` (`src/llama-graph.cpp:3938`): `cpy_k`/`cpy_v` 4021-4047,
  `get_k`/`get_v` 4063, `ggml_flash_attn_ext` 3372.
- [O] `ggml_vk_flash_attn_kvarn`/`_tail` retornam `false` para a view Q4:
  não há KVarN nem materialização na Vulkan nesta configuração.
- [O] `graph splits = 2 + 2N` (2, 4, 20 e 22 para N = 0, 1, 9 e 10): duas
  fronteiras por camada remota por ubatch; com N≈11, ~22 fronteiras e ~44
  sincronizações completas por ubatch, sem sobreposição CUDA/Vulkan.
- [O] A máscara F16 `[n_kv × 256]` é copiada uma vez por backend por ubatch
  (`ggml-backend.cpp:1513`).

## 3. Placement e custos

| Item | Valor | Tipo |
|---|---|---|
| Camadas | 64 + 1 MTP; `full_attention_interval=4` → 16 full-attention (il 3, 7, …, 63) | [O] |
| Geometria | 24 heads Q, 4 heads KV, dimensão 256 | [O] |
| Regra `auto` | menor N cujo KV local cabe na CUDA (`llama-kv-mixed-placement.cpp:417`); aplicado em `llama-context.cpp:1545` | [O] |
| 262144, MTP off, reserva 650 MiB | 10 Vulkan / 6 CUDA; `Vulkan0 KV buffer size = 2880.00 MiB` | [O] |
| 102400, MTP off | 1 / 15 | [O] |
| 204800, MTP4 | 14 / 2 | [O] |
| 131072, produção | **10 Vulkan / 6 CUDA**, reserva 1240,9 MiB (RS 448,9, MTP 142,0) — medido na §10 | [O] |
| Router esconde o log? | sim: INFO do libllama vira TRACE (4) em `common/log.cpp:530-547`; o filho roda com `verbosity = 3` | [O] |
| `MTP K/V placement: local_attention=off` | só o contexto draft MTP (`common/speculative.cpp:5393-5425`) | [O] |
| `remote-attn-cuda-reserve` padrão | 650 MiB (`common/common.h:810`); a ajuda diz 350M (`common/arg.cpp:3721`) | [O] |

Custos por camada remota, por ubatch de 256 tokens:

| Grandeza | 16k | 65k | 131k | Tipo |
|---|---:|---:|---:|---|
| Q + K + V enviados (F32) | 8,0 MiB | 8,0 MiB | 8,0 MiB | [D] |
| Saída recebida (F32) | 6,0 MiB | 6,0 MiB | 6,0 MiB | [D] |
| Máscara F16 por backend | 8 MiB | 32 MiB | 64 MiB | [D] |
| KV Q4 único lido (1152 B/token) | 18 MiB | 72 MiB | 144 MiB | [D] |
| Tráfego KV com releitura (limite superior) | 1,7 GiB | 6,75 GiB | 13,5 GiB | [D] |
| FLOPs | 103 G | 412 G | 825 G | [D] |
| Atenção Radeon, 20 W (5,9 ms/1k) | ~97 ms | ~387 ms | ~773 ms | [D] |
| Atenção CUDA, camada local (0,23 ms/1k) | ~4 ms | ~15 ms | ~30 ms | [D] |
| Cópias (D2H + memcpy + leitura UMA + H2D) | ~10–16 ms | ~10–16 ms | ~10–16 ms | [D]/[H] |

[D] Modelo: T(73k, N=11, 20 W) ≈ 285 × 0,67 + 66,1 ms × 10.408 ≈ 879 s; o
observado foi 899 s até 72.704 tokens. Eficiência implícita da Radeon:
~1,07 TFLOPS efetivos a 20 W (`RADV PHOENIX ... matrix cores: KHR_coopmat`).

## 4. Evidências observadas

| # | Evidência | Fonte |
|---|---|---|
| E1 | R1 20 W: `72704 ... 899.21 s / 80.85`; depois `force-killing ... after timeout` | `tier-cache-break-even-20261002/XXL` |
| E2 | R2 20 W: `68864 ... 830.23 s / 82.95`; fence `comp_1.2.0`; boot -2 sem shutdown | `-xxl-retry` + `journalctl -b -2 -k` |
| E3 | R3 15 W: `66816 ... 840.80 s / 79.47`; fence `comp_1.3.0`; boot -1 sem shutdown | `-current15w` + `journalctl -b -1 -k` |
| E4 | R4 sem AMD: `1626210.66 ms / 73000 tokens (44.89 tok/s)`, exit 0, sem escrita AMD | `-xxl-no-amd-controls`; `server-gpu-power.cpp:1319,1450,1469` |
| E5 | 106 requisições 4B na Vulkan0 durante R4 | `journalctl --user -u qwen35-4b-mtp-8092 -b 0` |
| E6 | Degrau 1,7 → 4,3 s/ubatch em ~14k só em R4, ~30 s após o início da carga 4B | logs + journal |
| E7 | Warm XXL: 256 tokens sobre 73k em `9560.98 ms` | JSON `-xxl-no-amd-controls` |
| E8 | Sem `NVRM`/`Xid`/AER/MCE/thermal/panic/OOM-kill/ring timeout em -2/-1 | `journalctl` |
| E9 | HHD: `tdp mode 'quiet'`, perfil `power`, CPU boost desligado, CPU ≤ 3,301 GHz, ventoinha manual; RyzenAdj leu `8000/20000/15000 mW` | `/etc/hhd`, logs |
| E10 | Produção: `GGML_KVARN_WINDOW_CHUNK=4096`, `GGML_CUDA_GRAPH_RECOVERY_HEADROOM_MB=18` | `20-mixed-kv.conf` |
| E11 | Tier XL (só CUDA, ub128): ubatch 360 → 3.400 ms, 136,6 tok/s em 57k; L: 300 → 420 ms | logs `current15w` |
| E12 | Telemetria do crash: Radeon 77–93%, ~5,46 GiB GTT, 65–75 °C, 4070 com uso baixo | sessão Codex (não remedido) |

## 5. Gargalos em ordem de probabilidade

1. **Atenção Vulkan** — domina a inclinação; prefill sem agrupamento GQA (6×
   releitura) e 16 tiles de linhas por head.
2. **Falta de VRAM na 4070** — causa-raiz do placement; cada camada que volta
   à CUDA economiza ~59 s a 73k (20 W).
3. **Potência da APU** (HHD `quiet`, STAPM 8 W) — multiplicador ~1,5–1,8×,
   confundido por E5.
4. **Serialização por camada** — a 4070 fica ociosa; ganho máximo com overlap
   ~7% a 65k.
5. **Contenção com outro processo na Vulkan0** — confunde R4.
6. Banda GTT/UMA — parte do item 1.
7. Cópias CUDA↔host — ~3–4% a 65k, ~15–25% a 4k.
8. Scheduler/`prealloc` — sincronizações no meio do grafo a cada ~1024 KV; baixo.
9. PCIe/Thunderbolt — baixo; o "2 Gb/s" é artefato do túnel.
10. KVarN local na CUDA — baixo se a rota windowed se mantiver
    (`fattn-mma-kvarn-case.cuh:376-377` pode cair para a rota por descritor).
11. MTP — baixo no prefill; pesa ~920 MiB de VRAM indiretamente.
12. Split-K — zero no prefill (`split_k=1`).
13. HHD/ventoinha direto — só via item 3.

## 6. Veredito sobre o fence

Combinação: estado de potência/clock sustentado + carga longa na Radeon.

- Bug de sincronização nosso: **improvável**. O governor escreve só na
  transição de fase, uma vez no início do prefill
  (`server-gpu-power.cpp:1306-1501`); R4 completou com o mesmo binário.
- Governor manual com RADV/amdgpu: **co-fator forte**. Os dois resets tiveram
  `manual` + `s 0 2700 / s 1 2700 / c` (`server-gpu-power.cpp:655-672`) e
  STAPM 8 → 15/20 W. Contra: R1 sobreviveu 900 s.
- Saturação legítima: **habilitador**; sozinha deveria travar o processo, não
  reiniciar a máquina.
- Driver/kernel: **sintoma**. "Fence fallback" é IRQ atrasada/perdida, não
  hang: o prefill seguiu ~4m45s depois do fence, nos dois casos, sem ring
  timeout.
- Reset duro sem panic/MCE/thermal ~830–840 s após o início das duas vezes →
  evento de potência/térmico da plataforma (SMU/EC/VRM) [H]. O journal perdeu
  os últimos 1–2 min de cada boot.

## 7. Melhorias priorizadas

| # | Opção | Efeito esperado | Riscos | Dificuldade |
|---|---|---|---|---|
| 1 | MTP desligado só no XXL | libera ~770 MiB → N≈6 → ~530 s (−40%) [H] | OOM médio; decode sem MTP | config |
| 2 | KVarN3/KVarN2 local no XXL | N≈4 com MTP2; possivelmente 0 sem MTP → até ~113 s [H] | precisão (exige KLD); conversão entre tiers | config + validação |
| 3 | Prefill com atenção CUDA lendo KV Q4 por streaming; Radeon só no decode | ~260–300 s (−70%) [H] | baixo fence; mesma precisão | alta |
| 4 | `remote-attn-cuda-reserve` 650 → 400–450M | −1 a 2 camadas → −6 a −13% | OOM médio-alto | config |
| 5 | Agrupamento GQA no FA coopmat1 do prefill (`ggml-vulkan.cpp:12849`) | 1,3–3× na atenção [H] | shader novo | média-alta |
| 6 | Governor AMD em `auto` | baseline seguro | — | — |
| 7 | Buffers pinned + cópias assíncronas | −3 a 5% a 65k; ~20% a 4k | baixo | média |
| 8 | ubatch XXL 512 | −5 a 7% | pode empurrar uma camada para a Radeon | config |
| 9 | Q5/Q6/Q8 remoto | sem ganho; Q8 +89% de memória | GTT/RAM | config |
| 10 | `GGML_KVARN_WINDOW_CHUNK` | ~0 no XXL | VRAM | config |
| 11 | Reduzir split-K | 0 no prefill | — | baixa |
| 12 | Corrigir as stats | pré-requisito | — | baixa-média |

## 8. Instrumentação

[O] `--remote-attn-stats` não mede o caminho nativo:
`ggml_local_split_stats` nunca é incrementada (`ggml-local-split.cpp:125,
340-364`); o único incremento é da rota TCP (`ggml-remote-attn.cpp:229`). O log
só roda com `backend_remote != nullptr` (`llama-context.cpp:1830-1836`), que no
caminho nativo nunca é criado (1626). `ggml_local_split_exec()` (380-384) só
registra erro e retorna `false`.

Onde medir:

1. `cuda_to_host_us`, `host_to_vulkan_us`, `vulkan_to_host_us`,
   `host_to_cuda_us`: fallback de `ggml_backend_sched_compute_splits`
   (`ggml-backend.cpp:~1893-1921`) e `ggml_backend_tensor_copy` (490-511),
   separando `tensor_get` e `tensor_set`. `GGML_BACKEND_COPY_PROFILE=1` hoje só
   separa `src_sync`/`dst_sync`/`copy`.
2. `cross_device_sync_us`: tempo dentro de `ggml_backend_synchronize(src/dst)`,
   com timestamp Vulkan e `cudaEvent` no fim de cada split para separar espera
   real de overhead.
3. `vulkan_attn_us`: `vkCmdWriteTimestamp` ao redor do FA (+ `mask_opt`,
   `split_k_reduce`) em `ggml_vk_flash_attn` (`ggml-vulkan.cpp:12756`), sem a
   barreira por nó do `GGML_VK_PERF_LOGGER` (20258-20263).
4. Rota CUDA local: `GGML_CUDA_FA_ROUTE_DEBUG=1`.
5. Aceite: soma dos componentes a ±5% do tempo de `llama_decode`.
6. Placement: filho com `-lv 4`.

## 9. O que não alterar antes de repetir a medição

1. `/home/hjotha/router-production.ini` (SHA-256
   `f49b5b3f8ef86dac0e72f046f7a0ac1092ababfe560e75f577ecbed60fa77f23`); testar só em cópias.
2. O guard de 1% em `tools/server/server-task.cpp`.
3. O drop-in `20-mixed-kv.conf`.
4. `remote-attn*`, `ctx-size-xxl=131072`, `batch/ubatch-xxl=256`, tipos KV,
   `spec-draft-n-max-xxl=2`.
5. `apu-tdp`/`amd-sclk-*` comentados (governor `auto`).
6. Perfil HHD (registrar, não mudar, durante o A/B).
7. Registrar o SHA-256 das `.so` de `build-optimized/bin`.
8. Kernel `7.1.8-1-cachyos-deckify`, Mesa/`vulkan-radeon` `26.2.1-1`,
   `linux-firmware-amdgpu 20260810-2`.
9. Não usar como baseline os 800–900 tok/s do `ggml_local_split_exec`, que
   zerava a atenção.
10. Não comparar com R4 sem anotar a contenção do serviço 4B.

## 10. Medição P1/P2 — 2026-10-02 15:10–15:22

Janela autorizada: `qwen35-4b-mtp-8092` e `llama-server-root` parados, router
temporário na porta 18130 com cópia do INI de produção (`slot-save-auto=false`,
slot-save isolado, `log-verbosity = 4`, controles AMD comentados), mesmas
variáveis do drop-in (`GGML_KVARN_WINDOW_CHUNK=4096`,
`GGML_CUDA_GRAPH_RECOVERY_HEADROOM_MB=18`), telemetria somente leitura a 1 Hz e
aborto automático se o kernel registrasse fence/timeout/Xid. Artefatos:
`/home/hjotha/beellama-mixed-kv-20261001-130910/xxl-decomposition-20261002/`
(`runner.py`, `window.sh`, `meta.json` com SHA-256 das bibliotecas,
`A-timing/`, `B-profile/`). Resultado: `runner exit 0`, `fence_hit= []`,
serviços restaurados (`root=active 4b=active`, modelo `loaded`), INI de
produção inalterado (`f49b5b3f...`).

### Placement real a 131072 [O]

```
mixed KV auto-placement: CUDA owner=CUDA0 CUDA=852.1/895.2 MiB reserve=1240.9 MiB (RS=448.9, MTP=142.0), Vulkan=1440.0/7493.0 MiB reserve=512.0 MiB, host handoff=9.1 MiB chunk=1024 tokens -> Vulkan Qx layers=10, CUDA KVarN layers=6
local attention placement: context=target Vulkan-layers=10/16 mode=static-split
Vulkan0 KV buffer size =  1440.00 MiB
sched_reserve: graph splits = 22
```

Igual à estimativa por memória da §3. Remotas: as 10 primeiras full-attention
(il 3…39); locais: il 43…63.

### Tempo por ubatch, governor `auto`, HHD `quiet`, sem contenção [O]

Prefill frio de 32.768 tokens (`A2`) e repetição fria de 33.024 (`A3`, ver
TMPDIR abaixo):

| Histórico | A2 ubatch | A3 ubatch |
|---:|---:|---:|
| 0 | 985 ms | 910 ms |
| 8k | 1530 ms | 1440 ms |
| 16k | 1920 ms | 2410 ms |
| 24k | 3250 ms | 3260 ms |
| 28k | 3640 ms | 3655 ms |

Ajuste A2: **0,550 s + 91,7 ms/1k**; A3: 0,493 s + 103,3 ms/1k. A2: 32.640
tokens em 262,67 s. [D] Com N=10, cada camada remota custa ~9,0 ms/1k em
`auto`, contra ~6,7 ms/1k com 20 W/2700 MHz (R1). A 28k, R1 levou 2780 ms por
ubatch e R4 5610 ms: **a contenção do serviço 4B explicava mais da metade da
lentidão de R4**; o controle AMD dá ~1,3× nesse ponto.

### O degrau é o limite de potência da APU [O]

Telemetria (`A-timing/telemetry.jsonl`, 648 amostras): `level=auto` em todas;
SCLK em `0: 800Mhz *` em 569. Nos primeiros ~150 s a Radeon tem picos de até
2064 MHz e a potência média sobe de 5,2 para 10,5 W. Em ~153–168 s (≈24k
tokens) fica travada em **8,0 W** e **800 MHz**, `gpu_busy_percent` sobe para
76–80% e o ubatch salta de ~2,1 para ~3,3 s. É o STAPM de 8 W do modo HHD
`quiet` (`8000/20000/15000 mW`). A 4070 fica em 0–7% de uso durante quase todo
o prefill; GTT 1,42 GiB; temperatura 51–65 °C.

### Atribuição com perfis (`B-profile`, 8.192 tokens) [O]

`GGML_BACKEND_COPY_PROFILE=1`, `GGML_VK_PERF_LOGGER=1` (força barreiras;
serve para atribuição, não para throughput) e `GGML_CUDA_FA_ROUTE_DEBUG=1`.

| Fluxo | Cópias | Volume | Tempo de cópia | Taxa |
|---|---:|---:|---:|---:|
| CUDA0 → Vulkan0 `Qcur_full` | 770 | 7812,8 MiB | 3411 ms | 2,24 GiB/s |
| CUDA0 → Vulkan0 `Qcur` | 770 | 3906,8 MiB | 1788 ms (+ `src_sync` 8777 ms) | 2,13 GiB/s |
| CUDA0 → Vulkan0 `Kcur` / `Vcur` | 770 + 770 | 651,1 + 651,1 MiB | 337 + 340 ms | ~1,88 GiB/s |
| **Vulkan0 → CUDA0 `attn_gated`** | 770 | 3906,8 MiB | **23.066 ms** | **0,17 GiB/s** |

- Total: 16,53 GiB, cópia 28,9 s, `src_sync` 8,8 s (espera pelo trabalho
  CUDA já enfileirado, não overhead).
- `FLASH_ATTN_EXT` com 256 queries: 3,94 ms (n_kv≈256) → 38,4 ms (4k) →
  64,1 ms (8k) por camada; ~0,8 TFLOPS. Ops não-FA no split Vulkan (`CONT`,
  `MUL`, `MUL_MAT` da rotação, `SET_ROWS`, `SIGMOID`): ~1,5–2,2 ms.
- Soma dos splits Vulkan com 256 queries: 22,79 s, contra 28,9 s de cópias no
  mesmo run.
- CUDA local: `KVARN_WINDOWED Q1=256` em 441 dispatches (rota rápida).

[D] Por camada remota e por ubatch de 256: readback ~30 ms, `Qcur_full`
~4,4 ms, `Qcur` ~2,3 ms, K+V ~0,9 ms → **~38 ms de fronteira**. Explica o
intercepto de 37–80 ms/camada da §3. A 8k de histórico, as fronteiras pesam
tanto quanto a atenção; a 65k, ~9%.

Duas descobertas de código/configuração:

1. **Cópia redundante de Q+gate.** `gate_sigmoid` e `attn_gated`
   (`qwen35.cpp:460-464`) consomem a saída Vulkan e são atribuídos à Vulkan pelo
   scheduler. Por isso `Qcur_full` (48.640 B/token) atravessa além de `Qcur`
   (24.576 B/token): ~81 KB/token enviados em vez de ~33 KB.
2. **`TMPDIR` em tmpfs quebra checkpoints do XXL misto.** A3 e B2 deveriam
   reaproveitar o prefixo, mas registraram
   `E state_seq_set_data: error loading state: mixed state spool directory is tmpfs/ramfs; set TMPDIR to a disk-backed directory`
   e `checkpoint restore failed; clearing target, draft and carry`, refazendo o
   prompt inteiro (`src/llama-kv-cache-kvarn.cpp:93-114`). A unit de produção
   não define `TMPDIR` (`systemctl show ... -p Environment`) e `/tmp` é
   `tmpfs`: **em produção, todo turno XXL que precise de checkpoint refaz o
   prompt inteiro**. O journal de produção ainda não registrou o erro
   (0 ocorrências desde 2026-10-01).

### Classificação revisada

1. Atenção Vulkan — domina em contexto longo (~9 ms/1k por camada em `auto`).
2. Falta de VRAM na 4070 — 10/16 camadas remotas.
3. **Readback Vulkan→CUDA a 0,17 GiB/s** — ~0,3 s por ubatch com 10 camadas;
   ~40% do tempo a 8k, ~9% a 65k.
4. Potência da APU — STAPM 8 W fixa 800 MHz depois de ~150 s.
5. Serialização — 4070 ociosa (0–7%).
6. Contenção com o serviço 4B — maior fator isolado da lentidão de R4.
7. Cópia redundante de `Qcur_full` — ~4,4 ms por camada por ubatch.
8. PCIe/TB — `Qcur`/`Qcur_full` a 2,1–2,2 GiB/s; o "2 Gb/s" é artefato.

### Melhorias acrescentadas

| # | Opção | Efeito esperado | Risco | Dificuldade |
|---|---|---|---|---|
| A | `Environment=TMPDIR=<diretório em disco>` na unit de produção | evita refazer 73k (~18–27 min) em turnos XXL com checkpoint | baixo (escrita em disco do spool) | config + reload/restart |
| B | Readback via buffer host **cached** (ou cópia GPU para staging pinned) | 0,17 → >2 GiB/s: −~0,27 s por ubatch; −~26% a 8k, −5 a 9% a 73k [H] | baixo | média |
| C | Fixar `gate_sigmoid`/`attn_gated` na CUDA | elimina `Qcur_full`: −~4,4 ms por camada por ubatch e −60% dos bytes enviados [H] | baixo | baixa |
