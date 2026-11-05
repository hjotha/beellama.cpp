# Four-bit KV cache evaluation (2026-11-04)

All KLD references use **BF16 KV cache**, with the same
`Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf` weights for every case; they
do not compare against BF16 model weights. The model SHA-256 was
`e23eb251493b8a89f7d0217590b593e696ac9bacaeac8605541e477533dc5838`, and the
WikiText-2 test corpus SHA-256 was
`173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`.

The local RTX 4070 evaluation used context 20,480, batch 128, ubatch 32,
FlashAttention, one sequence and no MTP. Each KLD value below scores 10,239
positions from one corpus chunk. The token-IID error estimates reported by
`llama-perplexity` are not confidence intervals across documents. KVarN's
zero-overlay control still has its exact sink and incomplete-group F16 staging.

| Local KV format | F16 tail | Mean KLD |
| --- | ---: | ---: |
| KVarN4 | 0 | 0.001677 |
| KVarN4 | 128 | 0.001238 |
| KVarN4 | 256 | 0.001150 |
| Q4_0 | 0 | 0.002710 |
| Q4_0 | 128 | 0.001425 |
| Q4_0 | 256 | 0.001302 |
| Q4_1 | 0 | 0.002209 |
| Q4_1 | 128 | 0.001272 |
| Q4_1 | 256 | 0.001112 |
| IQ4_NL | 0 | 0.002229 |
| IQ4_NL | 128 | 0.001254 |
| IQ4_NL | 256 | 0.001121 |
| SAW-INT4, BDR16 K+V | 0 | 0.004316 |
| SAW-INT4, BDR16 K+V | 128 | 0.002356 |
| SAW-INT4, BDR16 K+V | 256 | 0.002045 |
| TurboQuant 4-bit NC | 0 | 0.004934 |
| TurboQuant 4-bit NC | 128 | 0.002288 |
| TurboQuant 4-bit NC | 256 | 0.002099 |

The SAW/NC tail cases used the actual 4-bit bodies and native CUDA F16
overlays, verified from the cache plan and layer routes. Their tested CUDA
library SHA-256 was
`c9f4dd004c99f56a4ebfb57197ffc909568e27ff714848e526af050451f2311d`.
Isolated F32 BDR64 K+V and BDR128 K+V prototypes without a tail had KLD
0.004039/0.003914/0.004917 and 0.003754/0.003886/0.003729 respectively on
three adjacent 20,480-token chunks. They are not the integrated BDR16 codec;
their reused experimental type ID is not interoperable. These chunks are not
independent samples from a representative corpus.

The pinned upstream SAW tests and an extended 256-dimensional set passed on
the RTX 4070, but those tests exercise the upstream code, not Bee's codec.
The integrated Bee path rotates both K and V with BDR16; the inspected
upstream configuration uses BDR128 for K and leaves V unrotated. A separate
CPU emulation of the upstream BF16 rounding differed from Bee's scalar codec
on 137 packed bytes out of 8,192 in one fixture. This is evidence of a
rounding difference in that emulation, not a direct Triton/Bee GPU parity test.

Cold local `llama-server` requests used the same model, context capacity
22,016, 20,480 prompt tokens, 64 generated tokens, batch 128, ubatch 32, one
sequence, no MTP and no prompt cache hit. These are medians of two requests;
wall time is measured at the client and excludes server startup.

| Format | F16 tail | Prefill tokens/s | Decode tokens/s | Wall seconds |
| --- | ---: | ---: | ---: | ---: |
| Q4_0 | 0 | 452.31 | 32.23 | 47.24 |
| Q4_0 | 128 | 430.96 | 31.17 | 49.55 |
| Q4_0 | 256 | 447.38 | 31.54 | 47.79 |
| Q4_1 | 0 | 450.51 | 27.22 | 47.79 |
| Q4_1 | 128 | 430.17 | 26.90 | 49.96 |
| Q4_1 | 256 | 445.26 | 27.27 | 48.32 |
| IQ4_NL | 0 | 453.35 | 29.21 | 47.34 |
| IQ4_NL | 128 | 429.36 | 28.56 | 49.91 |
| IQ4_NL | 256 | 446.06 | 28.97 | 48.10 |
| KVarN4 | 0 | 390.97 | 32.25 | 54.35 |
| KVarN4 | 128 | 365.32 | 32.11 | 58.03 |
| KVarN4 | 256 | 377.12 | 32.33 | 56.27 |
| SAW-INT4, BDR16 K+V | 0 | 397.21 | 17.82 | 55.11 |
| SAW-INT4, BDR16 K+V | 128 | 354.08 | 15.27 | 61.98 |
| SAW-INT4, BDR16 K+V | 256 | 365.82 | 15.40 | 60.08 |
| TurboQuant 4-bit NC | 0 | 425.48 | 24.08 | 50.77 |
| TurboQuant 4-bit NC | 128 | 410.42 | 24.68 | 52.47 |
| TurboQuant 4-bit NC | 256 | 424.95 | 24.99 | 50.73 |

Two repetitions do not establish statistical significance. Q4_1 with tail 256
had the lowest local KLD on this chunk, closely followed by IQ4_NL with tail
256. KVarN4 with its native 128-token tail is the leading tested option also
available as the same storage format on the Xbox; explicit positive global
tail sizes are not supported on the current split path. SAW/NC F16 tails are
local-only. Low KLD alone does not establish freedom from generation loops.

With the same KVarN4 format on both devices and an artificial split at token
256, F32, FP8 E5M2 and FP8 E4M3FN transfer gave KLD 0.001242, 0.002352 and
0.001540 respectively on the 20,480-token evaluation. These runs change the
placement and producer/attention implementation as well as transfer precision;
they do not isolate the device. The split deliberately sends nearly the entire
context to Xbox, unlike the production split at 108,032. E4M3FN was more
accurate in that experiment, while E5M2 was faster. E4M3FN and the native
snapshot capability can be negotiated together by a qualified peer.

On Xbox app 1.6.0.928, an E4M3FN session restored an existing 109,505-token
native snapshot, published a new 109,513-token pair and restored that pair in
a fresh process. The test-created pair was then retired; the five previously
observed Xbox native objects remained. The existing 128-token tail is included
in this native state; this does not qualify a global 256-token split tail.
An updated host subsequently restored an older native file, saved without its
Q4 companion payload, and restored the new pair in a fresh process. At 109,513
tokens the host file fell from 2,250,587,800 to 2,223,286,770 bytes for these
fixtures; old readers can reject the new layout.

An E4 CPU-versus-CUDA comparison on the same model and 2,048-token prompt
used the artificial split P=256 and three cold requests per arm. CPU medians
were 72.04 prefill tokens/s, 21.42 decode tokens/s and 31.36 seconds wall;
CUDA medians were 57.23, 21.47 and 38.76 respectively. All generated outputs
matched, and CUDA counters recorded 5,712 pack and 5,712 unpack calls. Both
arms measured KLD 0.000984 against the same BF16 KV reference in a separate
2,048-token run. The generic CUDA E4 path is therefore opt-in through
`GGML_RKVA_CUDA_FP8=1`; these measurements do not establish its performance
at the production split.

An isolated request using the actual production split, E5M2, KVarN4 on both
sides and 109,490 prompt tokens completed with a cache miss: prefill 499.34
tokens/s, decode 18.11 tokens/s for 16 generated tokens, and 242.91 seconds
client wall time. Its 3,168 remote attention calls covered positions through
109,504 in 16 layers. This single request used batch 1024, ubatch 512 and Xbox
app 1.6.0.926; it is not a comparison to another format or a qualification of
the full 162,816-token capacity. Free GPU memory was sampled as low as 21 MiB
during the run. Snapshot auto-save/restore was disabled for this isolated
measurement.

On 2026-11-05 the production router was configured for KVarN4 target KV on
all six tiers, Q4_0 MTP draft KV, and native KVarN4 on Xbox with E5M2 transfer.
Each tier was selected and generated a first token in the live service; this
is an allocation and routing check, not a full-capacity test of each tier. A
109,490-token cache-miss request on the already initialized production XXXL
tier completed at 501.21 prefill tokens/s and 18.27 decode tokens/s for 16
generated tokens, with 219.33 seconds client wall time. Xbox app 1.6.0.927
was running and the host/Xbox native snapshot was published at 109,505 tokens.
The published host file was 2,249,892,456 bytes and the logged save took
21.33 seconds. After a service restart, a continuation restored 109,505 tokens
from disk, reprocessed one prompt token and generated eight tokens in 38.92
seconds client wall time including the tier transition and restore. The log
identified a native snapshot restore. A second cold restore generated the same
eight token IDs and its logged counters showed 128 remote KVarN4 attention
calls in 16 layers, using E5M2 for positions 109,505 through 109,512.
Repeating that same prompt immediately in the live context instead fell back
to full prefill because no usable recurrent checkpoint was available; this
repeat was cancelled rather than run to completion.
GPU memory was sampled as low as 19 MiB free during that prefill; the samples
are not a continuous peak measurement. Production's default log verbosity did
not expose the remote call counters for that request; the call count above is
from the isolated 926 run.

On Xbox 1.6.0.929 and the RTX 4070, flexible global KVarN4 tails were tested
with a 256-token split. The Xbox shader matched an independent FP64 oracle for
tail lengths 0, 1, 17, 127, 128, 129, 257, 513 and 1,023, including rewind.
Cold 1,024-token model requests plus 16 generated tokens completed with tails
0, 1, 17, 128, 129, 257 and 513 using E4M3FN transfer. A tail-257 native
snapshot saved 1,031 evaluated tokens and restored them after a server restart,
reprocessing one token. These are correctness checks, not a matched TPS or KLD
comparison. Snapshot storage still enforces its separate quotas.

For a local KLD reproduction, first save BF16 logits with the same model,
corpus and batching, then compare a selected cache type and tail, for example:

```sh
llama-perplexity -m model.gguf -f wiki.test.raw -c 20480 -b 128 -ub 32 \
  --chunks 1 -t 8 -ngl 99 -fit off -fa on -ctk saw_int4 -ctv saw_int4 \
  --kv-tail-tokens 256 --kv-tail-type f16 --kl-divergence \
  --kl-divergence-base bf16.kld
```
