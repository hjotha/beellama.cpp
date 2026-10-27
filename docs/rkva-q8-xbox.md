# Q8_0 remote attention on Xbox

Q8_0 native overflow uses the Xbox GPU FP8 query/output codec by default when
its pipelines are available. The conventional CPU codec remains the fallback
when GPU pipelines cannot be created. To force the CPU codec, create
`rkva-q8-fp8-gpu-off.flag` in the app's LocalState before connecting; remove it
to restore the default. A GPU failure after execution begins invalidates the
session rather than replaying a partly applied operation on the CPU.

The Q8 multi-query INT8 dot shader is experimental and disabled by default.
Create `rkva-q8-mq-i8.flag` in LocalState before connecting to opt in. It
represents eligible query blocks exactly using two signed-byte vectors and
falls back to F32 per block when exact representation is unavailable. The
change of floating-point arithmetic order is not guaranteed bit-exact on all
inputs or devices. Profile `format_multi_query.q8_0.mq2_i8.calls` counts PSO
selection, not the fraction of blocks that avoid F32 fallback.

The host Q8 producer defaults to CPU. `GGML_RKVA_Q8_PRODUCER=cuda` opts into
experimental CUDA preparation. It currently requires `GGML_CUDA_KVARN`, the
current CUDA device, and synchronous inputs; multi-device, HIP, and MUSA are
not qualified. The current producer still reads full F32 input to the host and
should not be assumed to improve both prefill and decode.
