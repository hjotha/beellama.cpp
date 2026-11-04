# Four-bit KV cache evaluation (2026-11-04)

These measurements are a preliminary comparison, not a production
qualification. All KLD references use **BF16 KV cache**, with the same
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
An isolated BDR128 K+V prototype without a tail had KLD 0.003754, 0.003886
and 0.003729 on three adjacent 20,480-token chunks. It is not the integrated
BDR16 codec and its reused experimental type ID is not interoperable.

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

Two repetitions do not establish statistical significance. Speed measurements
for the other formats and further corpus chunks were still in progress at this
checkpoint. SAW/NC F16 tails are local-only; the current split path rejects
them. Standard precision-tail streaming restore, speculative rollback and
mixed Xbox configurations have separate restrictions and require their own
qualification. Low KLD alone does not establish freedom from generation loops.

For a local KLD reproduction, first save BF16 logits with the same model,
corpus and batching, then compare a selected cache type and tail, for example:

```sh
llama-perplexity -m model.gguf -f wiki.test.raw -c 20480 -b 128 -ub 32 \
  --chunks 1 -t 8 -ngl 99 -fit off -fa on -ctk saw_int4 -ctv saw_int4 \
  --kv-tail-tokens 256 --kv-tail-type f16 --kl-divergence \
  --kl-divergence-base bf16.kld
```
