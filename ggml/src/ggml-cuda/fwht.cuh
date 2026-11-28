#include "common.cuh"

bool ggml_cuda_op_mul_mat_use_fwht(const struct ggml_tensor * op);

// Returns whether the Fast Walsh-Hadamard transform could be used.
bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst);
bool ggml_cuda_op_fwht_signed(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                              const ggml_tensor * signs, ggml_tensor * dst);

// Threads per block for the one-block-per-transform FWHT kernels (fwht_cuda_block, fwht_quantize_q8_1) at width n.
// 256 by default; with few transforms in flight (decode and speculative verify: 64 or fewer) min(n/2, 1024), so each
// thread keeps at least two elements and the few blocks there are use more of the SM. The butterfly runs its stages
// in the same order for any block width, so the result is bit-identical. GGML_CUDA_FWHT_NT overrides the wide value
// (256 disables it).
static inline int ggml_cuda_fwht_block_threads(const int n, const int64_t n_transforms) {
    static const int wide_override = getenv("GGML_CUDA_FWHT_NT") ? atoi(getenv("GGML_CUDA_FWHT_NT")) : 0;
    if (n_transforms > 64 || n < 512) {
        return 256;
    }
    const int nt = wide_override > 0 ? wide_override : (n/2 < 1024 ? n/2 : 1024);
    return (nt == 512 || nt == 1024) && nt <= n ? nt : 256;
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
void ggml_cuda_fwht_ptq1_ada_q8(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                               const ggml_tensor * signs, ggml_tensor * dst, int64_t k);
#endif
