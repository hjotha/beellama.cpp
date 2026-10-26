# RKVA nq=1 decode critical path

Measured attribution of one XXXL decode token (Qwen3.8-27B IQ3_XXS on the 4070,
overflow tier on the Xbox), what was changed to shorten it, and what is
deliberately left alone. Every number below is either **measured** (with the
command that produced it) or explicitly marked **derived**.

## 1. Where a token goes

Tools: `tools/xbox/scripts/analyze-rkva-decode-trace.py` (read-only) applied to
the recorded host trace of the 2026-10-26 mapped run.

```bash
python3 tools/xbox/scripts/analyze-rkva-decode-trace.py \
  /home/hjotha/.codex/artifacts/beellama-xbox-rkva-20261026/wf4-cached-p102400-long-r1-trace.3091043.jsonl
```

Selected per-token sums (512 decode tokens, positions 130560..130815, 16
full-attention layers):

| phase | ms/token | events/token | meaning |
|---|---:|---:|---|
| `cuda_graph_host` | 27.507 | 93 | local CUDA subgraph host spans (inclusive) |
| `cuda_graph_memory_guard` | 3.032 | 93 | VRAM headroom `cudaMemGetInfo()` per graph submission |
| `cuda_graph_bookkeeping` | 3.480 | 93 | parent span of the guard + key cache + decision |
| `cuda_producer_sync` | 13.284 | 31 | local FA completion waits (overlapped, not additive) |
| `rpc_total` | 20.913 | 16 | worker transaction, send -> body received |
| `rpc_header_wait` | 18.952 | 16 | send -> response header: console + network |
| `rpc_body_recv` | 1.539 | 16 | response payload |
| `range_wait` | 6.922 | 16 | **exposed** remote wait after local FA |
| `job_prepare` | 2.155 | 16 | scheduler-thread K/V readback + producer + query pack |
| `kv_producer` | 0.820 | 16 | KVarN4 quantization of the new row (CPU) |
| `q_pack` | 0.500 | 16 | FP8 query pack (CUDA extension) |
| `k_readback` + `v_readback` | 0.433 | 32 | 4 KiB D2H each |
| `out_decode` | 0.560 | 16 | FP8 -> F32 output conversion (worker thread) |

Per layer that is a 1.307 ms round trip against ~0.78 ms of local FA, so
~0.43 ms/layer (6.9 ms/token) is exposed waiting. Local FA acceleration cannot
help until the round trip drops below the local FA time.

The `rpc_header_wait` splits into the console engine wall (11.72 ms/token,
~0.73 ms/layer at this position) plus ~0.42 ms/layer of network, framing and
queueing.

## 2. Change 1: cache the graph headroom sample (host)

`ggml/src/ggml-cuda/graph-guard.h` (new, pure, unit tested),
`ggml/src/ggml-cuda/common.cuh`, `ggml/src/ggml-cuda/ggml-cuda.cu`.

`ggml_backend_cuda_context::check_memory_pressure_before_graph()` queries
`cudaMemGetInfo()` before **every** graph submission, 93 times per decode token.
Measured cost: 29-32 us per call, 3.03 ms/token.

A sampled free-memory value stays a valid lower bound while no device memory is
reserved. Every reservation path now bumps
`ggml_cuda_graph_guard::note_device_reservation()`:

* `ggml_cuda_device_malloc` (all `cudaMalloc`/`cudaMallocManaged` callers,
  including both pools and compute buffers),
* `cuMemCreate` in the VMM pool growth,
* `cudaGraphInstantiate`, `cudaGraphExecUpdate`, `cudaStreamEndCapture`,
* `ggml_backend_cuda_buffer_type_alloc_buffer`.

The guard reuses a sample only when it is younger than
`GGML_CUDA_GRAPH_GUARD_SAMPLE_MS` (default 10) **and** the generation counter is
unchanged. Any rejection falls through to the exact query, so the fast path can
never approve a state that per-call sampling would have rejected:

* pressure decisions (`free < headroom`) are unchanged: a sample below the
  headroom never approves,
* graph release/recovery state machines are untouched,
* `GGML_CUDA_GRAPH_GUARD_SAMPLE_MS=0` restores the exact per-call behaviour.

Untracked consumers (another process growing VRAM) are bounded by the sample
age; on this host the only CUDA client is the server itself.

Activation evidence: at context destruction the backend logs
`headroom guard submissions fastpath=<n> queried=<m> sample_age_ms=10`. The A/B
runner requires `fastpath > 0` before it reports a delta.

Expected saving: up to 2.8-2.9 ms/token of the measured 3.03 (**derived**).

## 3. Console fence wait

The console uses `SetEventOnCompletion` and `WaitForSingleObject`, retaining
terminal-failure and device-removal checks. The experimental bounded spin was
removed after the user-reported +0.29% result fell within measurement noise.
See [the final policy](rkva-final-optimizations.md).

## 4. Not changed, with reasons

* **Network/transport.** The host already sets `TCP_QUICKACK` per large read and
  writes header+payload with a single `sendmsg`; the listener answers with two
  `send()` calls. Nagle is not the limiting factor: a 40-byte header followed by
  a full-MSS payload leaves immediately. The remaining ~0.42 ms/layer is
  per-packet latency on the USB 1 GbE path plus console-side socket handling,
  not serialization (per layer ~10.3 KiB TX + 6.3 KiB RX). Recommendations, none
  measured here: replace the USB NIC with a PCIe 2.5 GbE NIC (both ends idle
  latency and jitter), disable interrupt moderation on the host NIC, evaluate
  `SO_BUSY_POLL`/`TCP_NODELAY` on the host socket, and keep the Xbox on wired
  Ethernet without power saving.
* **Query WHT on the console CPU.** Measured on the host with the production
  `butterfly_wht256`: **9.98 us per layer** for 24 heads (`g++ -O2 -mavx2`,
  `.oc-artifacts/wht_bench.cpp`). Too small to justify touching numerics-adjacent
  code; a vectorized rewrite was rejected on risk/benefit.
* **K/V cells validation readback.** `rkva_run_range()` reads back all `n_live`
  cell ids (~112 KiB at 28k rows) before every native async submission, an
  estimated ~50 us/layer (~0.8 ms/token, **derived**, untraced). Validating only
  the appended suffix would need proof that every path that rewrites existing
  cells bumps `range_epoch`; that audit was not done, so the exact scan stays.
* **Decode kernel / split policy.** `kvarn4_decode_split()` already uses
  `split_k = 64` at 28k rows, so the nq=1 grid is 256 workgroups; the consoles'
  0.5+ ms cannot be attributed to missing parallelism without an engine profile
  at the production position.

## 5. Open question for the next window

At position 102400 the console engine costs 313 us/call (`engine_us` p50 of the
recorded decode profile, GPU ~180 us). At 130560 the engine wall is 733 us/call.
The overflow tier grows from 1 to ~28k rows, so the difference has to come from
the nq=1 attention work on the console; no decode engine profile at the
production position exists yet. The A/B runner collects it
(`counters`, `transport_decode_phases`, GPU timestamps) so the
next phase can decide between kernel work and transport work with data.

## 6. How to measure the next phase

```bash
python3 tools/xbox/scripts/run-rkva-decode-host-ab.py \
  --server-binary build/bin/llama-server --model model.gguf \
  --prompt-tokens ids.json --xbox-host 192.168.1.200:50054 \
  --xbox-build-id <installed package> --threads 6 \
  --prefix-snapshot <frozen prefix> --output ~/rkva-decode-ab-r1 \
  --execute
```

Dry-run by default, refuses to start while a Codex benchmark process (or the
Codex CLI, unless `--allow-codex-idle`) is alive, runs `A,B,B,A` through
`bench-rkva-range.py`, requires token-ID parity across all four passes, and
writes `rkva-decode-host-ab.json` plus a per-pass decode attribution.
