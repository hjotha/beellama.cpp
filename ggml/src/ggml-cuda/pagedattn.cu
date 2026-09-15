#include "pagedattn.cuh"

#include "cpy-utils.cuh"
#include "dequantize.cuh"
#include "fattn.cuh"

static constexpr int PAGED_ATTN_FA_STRIDE = 256;

// flash-decoding: context tokens handled per shared memory tile
static constexpr int PAGED_ATTN_FD_TILE = 256;
// flash-decoding: upper bound for the context split count
static constexpr int PAGED_ATTN_FD_SPLITS_MAX = 256;
// flash-decoding: how many CTAs per SM the split heuristic aims for
static constexpr int PAGED_ATTN_FD_WAVES = 12;
// flash-decoding: largest number of query heads a single CTA handles
static constexpr int PAGED_ATTN_FD_GQA_MAX = 8;

__device__ __forceinline__ float paged_warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
#if defined(GGML_USE_HIP)
        val += __shfl_down_sync(0xffffffffffffffffULL, val, offset, 32);
#else
        val += __shfl_down_sync(0xffffffffu, val, offset, 32);
#endif
    }
    return val;
}

__device__ __forceinline__ float paged_warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
#if defined(GGML_USE_HIP)
        val = fmaxf(val, __shfl_down_sync(0xffffffffffffffffULL, val, offset, 32));
#else
        val = fmaxf(val, __shfl_down_sync(0xffffffffu, val, offset, 32));
#endif
    }
    return val;
}

__device__ __forceinline__ float paged_block_reduce_sum(float val, float * smem, int tid, int n_threads) {
    const int lane    = tid & 31;
    const int warp_id = tid >> 5;
    const int n_warps = (n_threads + 31) >> 5;
    val = paged_warp_reduce_sum(val);
    if (lane == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();
    float reduced = tid < n_warps ? smem[tid] : 0.0f;
    if (warp_id == 0) {
        reduced = paged_warp_reduce_sum(reduced);
        if (lane == 0) {
            smem[0] = reduced;
        }
    }
    __syncthreads();
    const float result = smem[0];
    __syncthreads();
    return result;
}

__device__ __forceinline__ float paged_block_reduce_max(float val, float * smem, int tid, int n_threads) {
    const int lane    = tid & 31;
    const int warp_id = tid >> 5;
    const int n_warps = (n_threads + 31) >> 5;
    val = paged_warp_reduce_max(val);
    if (lane == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();
    float reduced = tid < n_warps ? smem[tid] : -FLT_MAX;
    if (warp_id == 0) {
        reduced = paged_warp_reduce_max(reduced);
        if (lane == 0) {
            smem[0] = reduced;
        }
    }
    __syncthreads();
    const float result = smem[0];
    __syncthreads();
    return result;
}

__global__ void paged_attention_write_kernel(const float * __restrict__ k_new,  // [batch_size, n_heads_kv, head_dim]
                                             const float * __restrict__ v_new,  // [batch_size, n_heads_kv, head_dim]
                                             half * __restrict__ kv_cache,      // The paged cache
                                             const int * __restrict__ write_slots,  // Global slot index for each token
                                             const int * __restrict__ batch_offsets,
                                             const int * __restrict__ batch_lens,
                                             const size_t stride_token,  // Elements between tokens in a block (nb1)
                                             const size_t stride_head,   // Elements between heads (nb2)
                                             const size_t stride_block,  // Elements between physical blocks (nb3)
                                             const int    n_heads_kv,
                                             const int    block_size) {
    const int head_idx = blockIdx.x;   // 0 to n_heads_kv - 1
    const int seq_idx  = blockIdx.y;
    const int tid      = threadIdx.x;  // 0 to head_dim - 1
    const int head_dim = blockDim.x;

    const int seq_start  = batch_offsets[seq_idx];
    const int num_tokens = batch_lens[seq_idx];
    const int token_idx  = blockIdx.z;
    if (token_idx >= num_tokens) {
        return;
    }

    const int token_batch_idx = seq_start + token_idx;
    const int target_slot     = write_slots[token_batch_idx];
    const int block_id        = target_slot / block_size;
    const int token_in_block  = target_slot % block_size;
    const size_t k_cache_idx  = (size_t) block_id * stride_block + (size_t) head_idx * stride_head +
                                (size_t) token_in_block * stride_token + tid;
    const size_t v_cache_idx  = (size_t) block_id * stride_block + (size_t) (n_heads_kv + head_idx) * stride_head +
                                (size_t) token_in_block * stride_token + tid;
    const size_t input_off    = (size_t) token_batch_idx * n_heads_kv * head_dim + (size_t) head_idx * head_dim + tid;

    kv_cache[k_cache_idx] = __float2half(k_new[input_off]);
    kv_cache[v_cache_idx] = __float2half(v_new[input_off]);
}

__global__ void paged_attention_decode_kernel(const float * __restrict__ q,
                                              const half * __restrict__ kv_cache,
                                              const int * __restrict__ block_table,
                                              const int * __restrict__ context_lens,
                                              const int * __restrict__ batch_offsets,
                                              const int * __restrict__ batch_lens,
                                              const size_t stride_token,
                                              const size_t stride_head,
                                              const size_t stride_block,
                                              const int    n_heads_kv,
                                              const int    block_size,
                                              const int    max_blocks,
                                              const float  scale,
                                              float * __restrict__ out) {
    extern __shared__ float smem[];

    const int head_idx = blockIdx.x;
    const int seq_idx  = blockIdx.y;
    const int tid      = threadIdx.x;
    const int n_heads  = gridDim.x;
    const int head_dim = blockDim.x;
    const int kv_head_idx = head_idx / (n_heads / n_heads_kv);
    const int seq_start      = batch_offsets[seq_idx];
    const int num_new_tokens = batch_lens[seq_idx];
    const int token_idx      = blockIdx.z;
    if (token_idx >= num_new_tokens) {
        return;
    }
    const int token_batch_idx = seq_start + token_idx;
    const int ctx_len         = context_lens[seq_idx];
    const int q_pos           = (ctx_len - num_new_tokens) + token_idx;
    const size_t q_offset     = (size_t) token_batch_idx * n_heads * head_dim + (size_t) head_idx * head_dim;
    const float q_val        = q[q_offset + tid] * scale;
    float qk_max  = -FLT_MAX;
    float exp_sum = 0.0f;
    float acc     = 0.0f;

    for (int token = 0; token <= q_pos; ++token) {
        const int bid            = token / block_size;
        const int token_in_block = token % block_size;
        const int physical_block = block_table[seq_idx * max_blocks + bid];
        if (physical_block < 0) {
            continue;  // evicted page: no contribution
        }
        const size_t k_idx       = (size_t) tid + (size_t) token_in_block * stride_token +
                                   (size_t) kv_head_idx * stride_head + (size_t) physical_block * stride_block;
        const size_t v_idx       = (size_t) tid + (size_t) token_in_block * stride_token +
                                   (size_t) (n_heads_kv + kv_head_idx) * stride_head +
                                   (size_t) physical_block * stride_block;
        const float qk           = paged_block_reduce_sum(q_val * __half2float(kv_cache[k_idx]), smem, tid, head_dim);
        const float qk_max_new   = fmaxf(qk_max, qk);
        const float exp_old      = __expf(qk_max - qk_max_new);
        const float exp_new      = __expf(qk - qk_max_new);
        exp_sum = exp_sum * exp_old + exp_new;
        acc     = acc * exp_old + exp_new * __half2float(kv_cache[v_idx]);
        qk_max  = qk_max_new;
    }

    const int out_idx = (size_t) token_batch_idx * n_heads * head_dim + (size_t) head_idx * head_dim + tid;
    out[out_idx] = acc / (exp_sum + 1e-6f);
}

static __device__ __forceinline__ void paged_quantize_block(const float * x, block_q4_0 * dst) {
    quantize_f32_q4_0_block(x, dst);
}

static __device__ __forceinline__ void paged_quantize_block(const float * x, block_q8_0 * dst) {
    quantize_f32_q8_0_block(x, dst);
}

static __device__ __forceinline__ float paged_dequantize_value(const block_q4_0 * row, int index) {
    const int ib  = index / QK4_0;
    const int iqs = index % QK4_0;
    float2 values;
    dequantize_q4_0(row, ib, iqs % (QK4_0 / 2), values);
    return iqs < QK4_0 / 2 ? values.x : values.y;
}

static __device__ __forceinline__ float paged_dequantize_value(const block_q8_0 * row, int index) {
    const int ib  = index / QK8_0;
    const int iqs = index % QK8_0;
    float2 values;
    dequantize_q8_0(row, ib, iqs & ~1, values);
    return (iqs & 1) ? values.y : values.x;
}

template<typename block_t, bool quantized>
static __device__ __forceinline__ float paged_cache_value(
        const void * cache, size_t row_offset, int index) {
    if constexpr (quantized) {
        return paged_dequantize_value(reinterpret_cast<const block_t *>(
            static_cast<const char *>(cache) + row_offset), index);
    } else {
        return __half2float(static_cast<const half *>(cache)[row_offset + index]);
    }
}

template<typename block_t, bool quantized>
static __global__ void paged_attention_gather_f16_kernel(
        const void * __restrict__ kv_cache,
        const int * __restrict__ block_table,
        const size_t stride_token,
        const size_t stride_head,
        const size_t stride_block,
        const int n_heads_kv,
        const int contiguous_base,
        const int block_size,
        const int context_len,
        const int padded_context,
        half * __restrict__ k_f16,
        half * __restrict__ v_f16) {
    const int token    = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int dim      = threadIdx.x;
    const int head_dim = blockDim.x;
    float k = 0.0f;
    float v = 0.0f;

    if (token < context_len) {
        const int bid            = token / block_size;
        const int token_in_block = token % block_size;
        const int physical_block = contiguous_base >= 0
            ? contiguous_base + bid : block_table[bid];
        if (physical_block >= 0) {
            const size_t k_offset = (size_t) physical_block * stride_block +
                                    (size_t) head_idx * stride_head +
                                    (size_t) token_in_block * stride_token;
            const size_t v_offset = (size_t) physical_block * stride_block +
                                    (size_t) (n_heads_kv + head_idx) * stride_head +
                                    (size_t) token_in_block * stride_token;
            k = paged_cache_value<block_t, quantized>(kv_cache, k_offset, dim);
            v = paged_cache_value<block_t, quantized>(kv_cache, v_offset, dim);
        }
    }

    const size_t out = ((size_t) head_idx * padded_context + token) * head_dim + dim;
    k_f16[out] = __float2half(k);
    v_f16[out] = __float2half(v);
}

static __global__ void paged_attention_mask_f16_kernel(
        const int context_len,
        const int n_tokens,
        const int padded_context,
        const int block_size,
        const int contiguous_base,
        const int * __restrict__ block_table,
        half * __restrict__ mask) {
    const int token = blockIdx.x * blockDim.x + threadIdx.x;
    const int query = blockIdx.y;
    if (token >= padded_context) {
        return;
    }

    const int query_pos = context_len - n_tokens + query;
    half value = __float2half(0.0f);
    if (token > query_pos) {
        value = __float2half(-INFINITY);
    } else if (contiguous_base < 0 && block_table != nullptr) {
        const int bid = token / block_size;
        if (block_table[bid] < 0) {
            value = __float2half(-INFINITY);  // evicted page: masked out
        }
    }
    mask[(size_t) query * padded_context + token] = value;
}

// SnapKV accumulator pointer carried in op_params (GGML_MAX_SRC limits srcs to 10)
static float * paged_attn_snapkv_accum(const float * op_params_f) {
    const int32_t * ip = (const int32_t *) (op_params_f + 1);
    uintptr_t addr = (uintptr_t)(uint32_t) ip[4] | ((uintptr_t)(uint32_t) ip[5] << 32);
    return addr != 0 ? (float *) addr : nullptr;
}

static const int * paged_attn_snapkv_capture_from(const float * op_params_f) {
    const int32_t * ip = (const int32_t *) (op_params_f + 1);
    uintptr_t addr = (uintptr_t)(uint32_t) ip[6] | ((uintptr_t)(uint32_t) ip[7] << 32);
    return addr != 0 ? (const int *) addr : nullptr;
}

static const int * paged_attn_snapkv_score_slots(const float * op_params_f) {
    const int32_t * ip = (const int32_t *) (op_params_f + 1);
    uintptr_t addr = (uintptr_t)(uint32_t) ip[8] | ((uintptr_t)(uint32_t) ip[9] << 32);
    return addr != 0 ? (const int *) addr : nullptr;
}

static float * paged_attn_snapkv_token_scores(const float * op_params_f) {
    const int32_t * ip = (const int32_t *) (op_params_f + 1);
    uintptr_t addr = (uintptr_t)(uint32_t) ip[10] | ((uintptr_t)(uint32_t) ip[11] << 32);
    return addr != 0 ? (float *) addr : nullptr;
}

static bool paged_attn_snapkv_streaming(const float * op_params_f) {
    const int32_t * ip = (const int32_t *) (op_params_f + 1);
    return ip[3] != 0;
}

static int paged_attn_snapkv_score_blocks(const float * op_params_f, int fallback) {
    const int32_t * ip = (const int32_t *) (op_params_f + 1);
    return ip[12] > 0 ? ip[12] : fallback;
}

static bool paged_attn_snapkv_sync_debug() {
    static const bool enabled = std::getenv("LLAMA_SNAPKV_SYNC_DEBUG") != nullptr;
    return enabled;
}

template<typename block_t, bool quantized>
static bool ggml_cuda_paged_attn_prefill(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        const ggml_tensor * q,
        const ggml_tensor * k_new,
        const ggml_tensor * kv_cache,
        const ggml_tensor * block_table,
        const ggml_tensor * batch_lens,
        const size_t stride_token,
        const size_t stride_head,
        const size_t stride_block,
        const int block_size,
        const int max_blocks,
        const float scale) {
    const int head_dim       = q->ne[0];
    const int n_heads        = q->ne[1];
    const int n_tokens       = q->ne[2];
    const int n_heads_kv     = k_new->ne[1];
    const int n_seq          = batch_lens->ne[0];
    const int active_context = ((const int32_t *) ((const float *) dst->op_params + 1))[2];
    const int contiguous_base = ((const int32_t *) ((const float *) dst->op_params + 1))[13] - 1;
    if (n_seq != 1 || n_tokens <= PAGED_ATTN_PARALLEL_MAX_TOKENS || active_context <= 0) {
        return false;
    }

    const int padded_context = (active_context + PAGED_ATTN_FA_STRIDE - 1) &
                               ~(PAGED_ATTN_FA_STRIDE - 1);
    GGML_ASSERT(padded_context <= max_blocks * block_size);

    ggml_cuda_pool_alloc<half> k_f16(ctx.pool(),
        (size_t) padded_context * n_heads_kv * head_dim);
    ggml_cuda_pool_alloc<half> v_f16(ctx.pool(),
        (size_t) padded_context * n_heads_kv * head_dim);
    ggml_cuda_pool_alloc<half> mask(ctx.pool(), (size_t) padded_context * n_tokens);

    paged_attention_gather_f16_kernel<block_t, quantized><<<
        dim3(padded_context, n_heads_kv), dim3(head_dim), 0, ctx.stream()>>>(
            kv_cache->data, (const int *) block_table->data,
            stride_token, stride_head, stride_block, n_heads_kv,
            contiguous_base, block_size, active_context, padded_context, k_f16.ptr, v_f16.ptr);
    paged_attention_mask_f16_kernel<<<
        dim3((padded_context + 255) / 256, n_tokens), dim3(256), 0, ctx.stream()>>>(
            active_context, n_tokens, padded_context, block_size,
            contiguous_base, (const int *) block_table->data, mask.ptr);

    ggml_tensor q_fa = *q;
    q_fa.ne[1] = n_tokens;
    q_fa.ne[2] = n_heads;
    q_fa.ne[3] = 1;
    q_fa.nb[0] = sizeof(float);
    q_fa.nb[1] = (size_t) head_dim * n_heads * sizeof(float);
    q_fa.nb[2] = (size_t) head_dim * sizeof(float);
    q_fa.nb[3] = (size_t) head_dim * n_heads * n_tokens * sizeof(float);

    ggml_tensor k_fa = *kv_cache;
    k_fa.type = GGML_TYPE_F16;
    k_fa.data = k_f16.ptr;
    k_fa.view_src = nullptr;
    k_fa.view_offs = 0;
    k_fa.ne[0] = head_dim;
    k_fa.ne[1] = padded_context;
    k_fa.ne[2] = n_heads_kv;
    k_fa.ne[3] = 1;
    k_fa.nb[0] = sizeof(half);
    k_fa.nb[1] = (size_t) head_dim * sizeof(half);
    k_fa.nb[2] = (size_t) head_dim * padded_context * sizeof(half);
    k_fa.nb[3] = (size_t) head_dim * padded_context * n_heads_kv * sizeof(half);

    ggml_tensor v_fa = k_fa;
    v_fa.data = v_f16.ptr;
    v_fa.view_src = nullptr;

    ggml_tensor mask_fa = k_fa;
    mask_fa.data = mask.ptr;
    mask_fa.view_src = nullptr;
    mask_fa.ne[0] = padded_context;
    mask_fa.ne[1] = n_tokens;
    mask_fa.ne[2] = 1;
    mask_fa.ne[3] = 1;
    mask_fa.nb[0] = sizeof(half);
    mask_fa.nb[1] = (size_t) padded_context * sizeof(half);
    mask_fa.nb[2] = mask_fa.nb[1] * n_tokens;
    mask_fa.nb[3] = mask_fa.nb[2];

    ggml_tensor fa = *dst;
    fa.op = GGML_OP_FLASH_ATTN_EXT;
    memset(fa.op_params, 0, sizeof(fa.op_params));
    ((float *) fa.op_params)[0] = scale;
    ((int32_t *) fa.op_params)[3] = GGML_PREC_F32;
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        fa.src[i] = nullptr;
    }
    fa.src[0] = &q_fa;
    fa.src[1] = &k_fa;
    fa.src[2] = &v_fa;
    fa.src[3] = &mask_fa;

    if (!ggml_cuda_flash_attn_ext_supported(ctx.device, &fa)) {
        return false;
    }
    ggml_cuda_flash_attn_ext(ctx, &fa);
    return true;
}

template<typename block_t, bool quantized, int GQA>
static __global__ void paged_attention_flash_decode_kernel(
        const float * __restrict__ q,
        const void * __restrict__ kv_cache,
        const int * __restrict__ block_table,
        const int * __restrict__ context_lens,
        const int * __restrict__ batch_offsets,
        const int * __restrict__ batch_lens,
        const size_t stride_token,
        const size_t stride_head,
        const size_t stride_block,
        const int n_heads_kv,
        const int gqa_ratio,
        const int block_size,
        const int max_blocks,
        const int contiguous_base,
        const int n_splits,
        const float scale,
        float * __restrict__ partial_acc,
        float2 * __restrict__ partial_meta) {
    extern __shared__ float smem[];

    const int tid      = threadIdx.x;
    const int head_dim = blockDim.x;
    const int lane     = tid & 31;
    const int warp_id  = tid >> 5;
    const int n_warps  = head_dim >> 5;

    float * q_shared = smem;
    float * s_logits = q_shared + GQA * head_dim;
    float * s_max    = s_logits + GQA * PAGED_ATTN_FD_TILE;
    float * s_sum    = s_max    + GQA * n_warps;

    const int group     = blockIdx.x;
    const int seq_idx   = blockIdx.y;
    const int split     = blockIdx.z % n_splits;
    const int token_idx = blockIdx.z / n_splits;

    const int num_tokens = batch_lens[seq_idx];
    if (token_idx >= num_tokens) {
        return;
    }

    const int token_batch_idx = batch_offsets[seq_idx] + token_idx;
    const int q_pos           = context_lens[seq_idx] - num_tokens + token_idx;
    const int n_ctx           = q_pos + 1;
    const int head_base       = group * GQA;
    const int kv_head_idx     = head_base / gqa_ratio;
    const int n_heads         = gridDim.x * GQA;

    for (int g = 0; g < GQA; ++g) {
        const size_t q_offset = (size_t) token_batch_idx * n_heads * head_dim +
                                (size_t) (head_base + g) * head_dim;
        q_shared[g * head_dim + tid] = q[q_offset + tid] * scale;
    }
    __syncthreads();

    const int chunk = (n_ctx + n_splits - 1) / n_splits;
    const int t_beg = split * chunk;
    const int t_end = min(n_ctx, t_beg + chunk);

    float m[GQA], l[GQA], acc[GQA];
    for (int g = 0; g < GQA; ++g) {
        m[g]   = -FLT_MAX;
        l[g]   = 0.0f;
        acc[g] = 0.0f;
    }

    for (int tile = t_beg; tile < t_end; tile += PAGED_ATTN_FD_TILE) {
        const int tile_n = min(PAGED_ATTN_FD_TILE, t_end - tile);

        int cached_bid = -1;
        int cached_pb  = 0;
        for (int t = warp_id; t < tile_n; t += n_warps) {
            const int token = tile + t;
            const int bid   = token / block_size;
            if (bid != cached_bid) {
                cached_bid = bid;
                cached_pb  = contiguous_base >= 0
                    ? contiguous_base + bid : block_table[seq_idx * max_blocks + bid];
            }
            float qk[GQA] = {};
            if (cached_pb >= 0) {
                const size_t k_offset = (size_t) cached_pb * stride_block +
                                        (size_t) kv_head_idx * stride_head +
                                        (size_t) (token - bid * block_size) * stride_token;
                for (int dim = lane; dim < head_dim; dim += 32) {
                    const float k = paged_cache_value<block_t, quantized>(kv_cache, k_offset, dim);
                    for (int g = 0; g < GQA; ++g) {
                        qk[g] += q_shared[g * head_dim + dim] * k;
                    }
                }
            } else {
                for (int g = 0; g < GQA; ++g) {
                    qk[g] = -FLT_MAX;  // evicted page: never attends
                }
            }
            for (int g = 0; g < GQA; ++g) {
                qk[g] = paged_warp_reduce_sum(qk[g]);
                if (lane == 0) {
                    s_logits[g * PAGED_ATTN_FD_TILE + t] = qk[g];
                }
            }
        }
        __syncthreads();

        float part[GQA];
        for (int g = 0; g < GQA; ++g) {
            part[g] = -FLT_MAX;
        }
        for (int t = tid; t < tile_n; t += head_dim) {
            for (int g = 0; g < GQA; ++g) {
                part[g] = fmaxf(part[g], s_logits[g * PAGED_ATTN_FD_TILE + t]);
            }
        }
        for (int g = 0; g < GQA; ++g) {
            part[g] = paged_warp_reduce_max(part[g]);
            if (lane == 0) {
                s_max[g * n_warps + warp_id] = part[g];
            }
        }
        __syncthreads();

        float m_new[GQA], resc[GQA];
        for (int g = 0; g < GQA; ++g) {
            float tile_max = -FLT_MAX;
            for (int w = 0; w < n_warps; ++w) {
                tile_max = fmaxf(tile_max, s_max[g * n_warps + w]);
            }
            m_new[g] = fmaxf(m[g], tile_max);
            resc[g]  = __expf(m[g] - m_new[g]);
            part[g]  = 0.0f;
        }
        for (int t = tid; t < tile_n; t += head_dim) {
            for (int g = 0; g < GQA; ++g) {
                const float w = __expf(s_logits[g * PAGED_ATTN_FD_TILE + t] - m_new[g]);
                s_logits[g * PAGED_ATTN_FD_TILE + t] = w;
                part[g] += w;
            }
        }
        for (int g = 0; g < GQA; ++g) {
            part[g] = paged_warp_reduce_sum(part[g]);
            if (lane == 0) {
                s_sum[g * n_warps + warp_id] = part[g];
            }
        }
        __syncthreads();

        for (int g = 0; g < GQA; ++g) {
            float tile_sum = 0.0f;
            for (int w = 0; w < n_warps; ++w) {
                tile_sum += s_sum[g * n_warps + w];
            }
            l[g]   = l[g] * resc[g] + tile_sum;
            acc[g] = acc[g] * resc[g];
            m[g]   = m_new[g];
        }

        cached_bid = -1;
        for (int t = 0; t < tile_n; ++t) {
            const int token = tile + t;
            const int bid   = token / block_size;
            if (bid != cached_bid) {
                cached_bid = bid;
                cached_pb  = contiguous_base >= 0
                    ? contiguous_base + bid : block_table[seq_idx * max_blocks + bid];
            }
            if (cached_pb < 0) {
                continue;  // evicted page: weight is ~0
            }
            const size_t v_offset = (size_t) cached_pb * stride_block +
                                    (size_t) (n_heads_kv + kv_head_idx) * stride_head +
                                    (size_t) (token - bid * block_size) * stride_token;
            const float v = paged_cache_value<block_t, quantized>(kv_cache, v_offset, tid);
            for (int g = 0; g < GQA; ++g) {
                acc[g] += s_logits[g * PAGED_ATTN_FD_TILE + t] * v;
            }
        }
        __syncthreads();
    }

    for (int g = 0; g < GQA; ++g) {
        const size_t base = ((size_t) token_batch_idx * n_heads + head_base + g) * n_splits + split;
        partial_acc[base * head_dim + tid] = acc[g];
        if (tid == 0) {
            partial_meta[base] = make_float2(m[g], l[g]);
        }
    }
}

// rescale and sum the per-split partial results of the flash-decoding kernel
static __global__ void paged_attention_flash_combine_kernel(
        const int * __restrict__ batch_offsets,
        const int * __restrict__ batch_lens,
        const float * __restrict__ partial_acc,
        const float2 * __restrict__ partial_meta,
        const int n_splits,
        float * __restrict__ out) {
    extern __shared__ float smem[];

    const int head_idx  = blockIdx.x;
    const int seq_idx   = blockIdx.y;
    const int token_idx = blockIdx.z;
    const int tid       = threadIdx.x;
    const int n_heads   = gridDim.x;
    const int head_dim  = blockDim.x;

    const int num_tokens = batch_lens[seq_idx];
    if (token_idx >= num_tokens) {
        return;
    }

    float * s_scale = smem;
    float * s_red   = smem + n_splits;

    const int token_batch_idx = batch_offsets[seq_idx] + token_idx;
    const size_t base = ((size_t) token_batch_idx * n_heads + head_idx) * n_splits;

    float m = -FLT_MAX;
    for (int s = tid; s < n_splits; s += head_dim) {
        m = fmaxf(m, partial_meta[base + s].x);
    }
    m = paged_block_reduce_max(m, s_red, tid, head_dim);

    float l = 0.0f;
    for (int s = tid; s < n_splits; s += head_dim) {
        const float2 ml = partial_meta[base + s];
        const float scale = __expf(ml.x - m);
        s_scale[s] = scale;
        l += ml.y * scale;
    }
    l = paged_block_reduce_sum(l, s_red, tid, head_dim);

    float acc = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
        acc += s_scale[s] * partial_acc[(base + s) * head_dim + tid];
    }

    out[((size_t) token_batch_idx * n_heads + head_idx) * head_dim + tid] = acc / (l + 1e-6f);
}

// SnapKV observation kernels: score the context keys with the queries of the
// observation window, softmax-normalize per query/head, and accumulate the
// resulting attention mass per logical page (skipping evicted pages).

static constexpr size_t PAGED_ATTN_OBS_SCRATCH_BUDGET = 128ull * 1024 * 1024;
static constexpr int PAGED_ATTN_OBS_CHUNK_MIN = 8;

static int paged_attn_snapkv_observation_chunk(size_t max_context, int n_heads, int n_seq) {
    const int candidates[] = { 32, 16, PAGED_ATTN_OBS_CHUNK_MIN };
    for (const int chunk : candidates) {
        const size_t scratch = max_context * (size_t) n_heads * (size_t) n_seq * (size_t) chunk * sizeof(float);
        if (scratch <= PAGED_ATTN_OBS_SCRATCH_BUDGET) {
            return chunk;
        }
    }
    return PAGED_ATTN_OBS_CHUNK_MIN;
}

template<typename block_t, bool quantized>
static __global__ void paged_attention_obs_scores_kernel(
        const float * __restrict__ q,
        const void * __restrict__ kv_cache,
        const int * __restrict__ block_table,
        const int * __restrict__ context_lens,
        const int * __restrict__ batch_offsets,
        const int * __restrict__ batch_lens,
        const size_t stride_token,
        const size_t stride_head,
        const size_t stride_block,
        const int n_heads_kv,
        const int gqa_ratio,
        const int block_size,
        const int max_blocks,
        const float scale,
        const size_t max_context,
        const int obs_offset,
        const int * __restrict__ capture_from,
        float * __restrict__ scores) {
    extern __shared__ float q_shared[];

    const int head_idx   = blockIdx.x;
    const int seq_idx    = blockIdx.y;
    const int obs_idx    = blockIdx.z;
    const int tid        = threadIdx.x;
    const int lane       = tid & 31;
    const int warp_id    = tid >> 5;
    const int head_dim   = blockDim.x;
    const int n_warps    = head_dim >> 5;
    const int n_heads    = gridDim.x;
    const int seq_start  = batch_offsets[seq_idx];
    const int num_tokens = batch_lens[seq_idx];
    const int token_idx  = obs_offset + obs_idx;
    if (obs_idx >= num_tokens || token_idx >= num_tokens || capture_from[seq_idx] < 0) {
        return;
    }

    const int token_batch_idx = seq_start + token_idx;
    const int q_pos           = context_lens[seq_idx] - num_tokens + token_idx;
    if (q_pos < capture_from[seq_idx]) {
        return;
    }
    const int kv_head_idx     = head_idx / gqa_ratio;
    const size_t q_offset     = (size_t) token_batch_idx * n_heads * head_dim + (size_t) head_idx * head_dim;
    q_shared[tid] = q[q_offset + tid] * scale;
    __syncthreads();

    float * score_row = scores + (((size_t) seq_idx * gridDim.z + obs_idx) * n_heads + head_idx) * max_context;
    for (int token = warp_id; token <= q_pos; token += n_warps) {
        const int bid            = token / block_size;
        const int token_in_block = token % block_size;
        const int physical_block = block_table[seq_idx * max_blocks + bid];
        float qk = -FLT_MAX;
        if (physical_block >= 0) {
            const size_t k_offset = (size_t) physical_block * stride_block +
                                    (size_t) kv_head_idx * stride_head +
                                    (size_t) token_in_block * stride_token;
            qk = 0.0f;
            for (int dim = lane; dim < head_dim; dim += 32) {
                qk += q_shared[dim] * paged_cache_value<block_t, quantized>(kv_cache, k_offset, dim);
            }
            qk = paged_warp_reduce_sum(qk);
        }
        if (lane == 0) {
            score_row[token] = qk;
        }
    }
}


template<typename block_t, bool quantized, int GQA>
static __global__ void paged_attention_obs_scores_gqa_kernel(
        const float * __restrict__ q,
        const void * __restrict__ kv_cache,
        const int * __restrict__ block_table,
        const int * __restrict__ context_lens,
        const int * __restrict__ batch_offsets,
        const int * __restrict__ batch_lens,
        const size_t stride_token,
        const size_t stride_head,
        const size_t stride_block,
        const int block_size,
        const int max_blocks,
        const float scale,
        const size_t max_context,
        const int obs_offset,
        const int * __restrict__ capture_from,
        float * __restrict__ scores) {
    extern __shared__ float q_shared[];

    const int kv_head_idx = blockIdx.x;
    const int seq_idx     = blockIdx.y;
    const int obs_idx     = blockIdx.z;
    const int tid         = threadIdx.x;
    const int lane        = tid & 31;
    const int warp_id     = tid >> 5;
    const int head_dim    = blockDim.x;
    const int n_warps     = head_dim >> 5;
    const int n_heads     = gridDim.x * GQA;
    const int seq_start   = batch_offsets[seq_idx];
    const int num_tokens  = batch_lens[seq_idx];
    const int token_idx   = obs_offset + obs_idx;
    if (obs_idx >= num_tokens || token_idx >= num_tokens || capture_from[seq_idx] < 0) {
        return;
    }

    const int token_batch_idx = seq_start + token_idx;
    const int q_pos = context_lens[seq_idx] - num_tokens + token_idx;
    if (q_pos < capture_from[seq_idx]) {
        return;
    }
    const int head_base = kv_head_idx * GQA;
    for (int g = 0; g < GQA; ++g) {
        const size_t q_offset = (size_t) token_batch_idx * n_heads * head_dim +
                                (size_t) (head_base + g) * head_dim;
        q_shared[g * head_dim + tid] = q[q_offset + tid] * scale;
    }
    __syncthreads();

    for (int token = warp_id; token <= q_pos; token += n_warps) {
        const int bid            = token / block_size;
        const int token_in_block = token % block_size;
        const int physical_block = block_table[seq_idx * max_blocks + bid];
        float qk[GQA];
        for (int g = 0; g < GQA; ++g) {
            qk[g] = -FLT_MAX;
        }
        if (physical_block >= 0) {
            const size_t k_offset = (size_t) physical_block * stride_block +
                                    (size_t) kv_head_idx * stride_head +
                                    (size_t) token_in_block * stride_token;
            for (int g = 0; g < GQA; ++g) {
                qk[g] = 0.0f;
            }
            for (int dim = lane; dim < head_dim; dim += 32) {
                const float k = paged_cache_value<block_t, quantized>(kv_cache, k_offset, dim);
                for (int g = 0; g < GQA; ++g) {
                    qk[g] += q_shared[g * head_dim + dim] * k;
                }
            }
            for (int g = 0; g < GQA; ++g) {
                qk[g] = paged_warp_reduce_sum(qk[g]);
            }
        }
        if (lane == 0) {
            for (int g = 0; g < GQA; ++g) {
                float * score_row = scores + (((size_t) seq_idx * gridDim.z + obs_idx) * n_heads + (head_base + g)) * max_context;
                score_row[token] = qk[g];
            }
        }
    }
}

static __global__ void paged_attention_obs_softmax_accum_kernel(
        float * __restrict__ scores,
        const int * __restrict__ context_lens,
        const int * __restrict__ batch_lens,
        const size_t max_context,
        const int obs_offset,
        const int * __restrict__ capture_from,
        const int * __restrict__ score_slots) {
    extern __shared__ float smem[];

    const int head_idx   = blockIdx.x;
    const int seq_idx    = blockIdx.y;
    const int obs_idx    = blockIdx.z;
    const int tid        = threadIdx.x;
    const int head_dim   = blockDim.x;
    const int n_heads    = gridDim.x;
    const int num_tokens = batch_lens[seq_idx];
    const int token_idx  = obs_offset + obs_idx;
    if (obs_idx >= num_tokens || token_idx >= num_tokens || capture_from[seq_idx] < 0 || score_slots[seq_idx] < 0) {
        return;
    }

    const int q_pos = context_lens[seq_idx] - num_tokens + token_idx;
    if (q_pos < capture_from[seq_idx]) {
        return;
    }
    float * score_row = scores + (((size_t) seq_idx * gridDim.z + obs_idx) * n_heads + head_idx) * max_context;

    float local_max = -FLT_MAX;
    for (int token = tid; token <= q_pos; token += head_dim) {
        local_max = fmaxf(local_max, score_row[token]);
    }
    const float qk_max = paged_block_reduce_max(local_max, smem, tid, head_dim);

    float local_sum = 0.0f;
    for (int token = tid; token <= q_pos; token += head_dim) {
        local_sum += __expf(score_row[token] - qk_max);
    }
    const float exp_sum = paged_block_reduce_sum(local_sum, smem, tid, head_dim);
    const float inv_sum = 1.0f / (exp_sum + 1e-6f);

    for (int token = tid; token <= q_pos; token += head_dim) {
        const float w = __expf(score_row[token] - qk_max) * inv_sum;
        score_row[token] = w;
    }
}

// Reduce one observation chunk to one atomic add per token instead of one add
// for every query/token pair.  The following page reduction is non-atomic and
// reproduces the host-side max-pool-then-sum operation exactly.
static __global__ void paged_attention_obs_token_reduce_kernel(
        const float * __restrict__ scores,
        const int * __restrict__ block_table,
        const int * __restrict__ context_lens,
        const int * __restrict__ batch_lens,
        const int * __restrict__ capture_from,
        const int * __restrict__ score_slots,
        const size_t max_context,
        const size_t token_stride,
        const int block_size,
        const int max_blocks,
        const int n_heads,
        const int obs_offset,
        const int chunk_n,
        float * __restrict__ token_accum) {
    const int head_idx = blockIdx.y;
    const int seq_idx  = blockIdx.z;
    const size_t token = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (head_idx >= n_heads || token >= max_context) {
        return;
    }

    const int slot = score_slots[seq_idx];
    if (slot < 0) {
        return;
    }

    const int num_tokens = batch_lens[seq_idx];
    float sum = 0.0f;
    for (int obs_idx = 0; obs_idx < chunk_n; ++obs_idx) {
        const int token_idx = obs_offset + obs_idx;
        if (obs_idx >= num_tokens || token_idx >= num_tokens) {
            continue;
        }
        const int q_pos = context_lens[seq_idx] - num_tokens + token_idx;
        if (q_pos < capture_from[seq_idx] || (int) token > q_pos) {
            continue;
        }
        const int bid = (int) token / block_size;
        if (bid >= max_blocks || block_table[seq_idx * max_blocks + bid] < 0) {
            continue;
        }
        const size_t score_idx = (((size_t) seq_idx * chunk_n + obs_idx) * n_heads + head_idx) * max_context + token;
        sum += scores[score_idx];
    }

    if (sum != 0.0f) {
        const size_t token_idx = ((size_t) slot * n_heads + head_idx) * token_stride + token;
        atomicAdd(&token_accum[token_idx], sum);
    }
}

static __global__ void paged_attention_snapkv_page_reduce_kernel(
        const float * __restrict__ token_accum,
        float * __restrict__ page_accum,
        const int * __restrict__ score_slots,
        const size_t max_context,
        const size_t token_stride,
        const int block_size,
        const int max_blocks,
        const int n_heads) {
    const int bid      = blockIdx.x * blockDim.x + threadIdx.x;
    const int head_idx = blockIdx.y;
    const int seq_idx  = blockIdx.z;
    if (bid >= max_blocks || head_idx >= n_heads) {
        return;
    }

    const int slot = score_slots[seq_idx];
    if (slot < 0) {
        return;
    }

    constexpr int pool_radius = 2; // five-token max pool, matching the old CPU path
    const size_t token_base = ((size_t) slot * n_heads + head_idx) * token_stride;
    const size_t token_begin = (size_t) bid * block_size;
    const size_t token_end = min(max_context, token_begin + (size_t) block_size);
    float page_score = 0.0f;
    for (size_t token = token_begin; token < token_end; ++token) {
        const size_t pool_begin = token > (size_t) pool_radius ? token - pool_radius : 0;
        const size_t pool_end = min(max_context, token + pool_radius + 1);
        float pooled = 0.0f;
        for (size_t neighbor = pool_begin; neighbor < pool_end; ++neighbor) {
            pooled = fmaxf(pooled, token_accum[token_base + neighbor]);
        }
        page_score += pooled;
    }

    page_accum[((size_t) slot * n_heads + head_idx) * max_blocks + bid] = page_score;
}

static int paged_attn_flash_decode_gqa(int gqa_ratio);

template<typename block_t, bool quantized>
static void paged_attn_snapkv_capture(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        const ggml_tensor * q,
        const ggml_tensor * kv_cache,
        const ggml_tensor * block_table,
        const ggml_tensor * context_lens,
        const ggml_tensor * batch_offsets,
        const ggml_tensor * batch_lens,
        const size_t stride_token,
        const size_t stride_head,
        const size_t stride_block,
        const int n_heads,
        const int n_heads_kv,
        const int n_seq,
        const int n_tokens,
        const int head_dim,
        const int block_size,
        const int max_blocks,
        const int score_blocks,
        const float scale,
        const int * capture_from_by_seq,
        const int * score_slots_by_seq,
        float * page_accum,
        float * token_accum) {
    if (capture_from_by_seq == nullptr || score_slots_by_seq == nullptr || page_accum == nullptr || token_accum == nullptr) {
        return;
    }
    const int gqa_ratio = n_heads / n_heads_kv;
    const int n_obs = n_tokens;
    const int obs_offset = 0;
    const int active_context = ((const int32_t *) ((const float *) dst->op_params + 1))[2];
    const size_t max_context = (size_t) ((active_context + block_size - 1) / block_size) * block_size;
    const size_t token_stride = (size_t) score_blocks * block_size;

    const int obs_chunk = paged_attn_snapkv_observation_chunk(max_context, n_heads, n_seq);
    auto launch_obs = [&](auto launch_scores) {
        for (int chunk_beg = 0; chunk_beg < n_obs; chunk_beg += obs_chunk) {
            const int chunk_n = std::min(obs_chunk, n_obs - chunk_beg);
            ggml_cuda_pool_alloc<float> scores(ctx.pool(),
                (size_t) n_seq * chunk_n * n_heads * max_context);
            launch_scores(chunk_n, obs_offset + chunk_beg, scores.ptr);
            const size_t reduce_smem = (size_t) ((head_dim + 31) / 32) * sizeof(float);
            paged_attention_obs_softmax_accum_kernel<<<
                dim3(n_heads, n_seq, chunk_n), dim3(head_dim), reduce_smem, ctx.stream()>>>(
                    scores.ptr, (const int *) context_lens->data, (const int *) batch_lens->data,
                    max_context, obs_offset + chunk_beg, capture_from_by_seq, score_slots_by_seq);
            constexpr int reduce_threads = 256;
            paged_attention_obs_token_reduce_kernel<<<
                dim3((max_context + reduce_threads - 1) / reduce_threads, n_heads, n_seq),
                dim3(reduce_threads), 0, ctx.stream()>>>(
                    scores.ptr, (const int *) block_table->data, (const int *) context_lens->data,
                    (const int *) batch_lens->data, capture_from_by_seq, score_slots_by_seq,
                    max_context, token_stride, block_size, max_blocks, n_heads,
                    obs_offset + chunk_beg, chunk_n, token_accum);
            if (paged_attn_snapkv_sync_debug() && active_context >= 45000 && chunk_beg + chunk_n == n_obs) {
                fprintf(stderr, "SNAPKVDBG dims active_context=%d max_context=%zu max_blocks=%d score_blocks=%d n_heads=%d n_seq=%d obs_chunk=%d token_stride=%zu\n",
                        active_context, max_context, max_blocks, score_blocks, n_heads, n_seq, obs_chunk, token_stride);
                CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
            }
        }
    };

    const int gqa = paged_attn_flash_decode_gqa(gqa_ratio);
    if (gqa == 1) {
        const size_t q_smem = (size_t) head_dim * sizeof(float);
        launch_obs([&](int chunk_n, int obs_beg, float * dst) {
            paged_attention_obs_scores_kernel<block_t, quantized><<<
                dim3(n_heads, n_seq, chunk_n), dim3(head_dim), q_smem, ctx.stream()>>>(
                    (const float *) q->data, kv_cache->data, (const int *) block_table->data,
                    (const int *) context_lens->data, (const int *) batch_offsets->data,
                    (const int *) batch_lens->data, stride_token, stride_head, stride_block,
                    n_heads_kv, gqa_ratio, block_size, max_blocks, scale,
                    max_context, obs_beg, capture_from_by_seq, dst);
        });
    } else {
#define SNAPKV_OBS_GQA_CASE(G)                                                                     \
        if (gqa == G) {                                                                             \
            const size_t q_smem = (size_t) G * head_dim * sizeof(float);                            \
            launch_obs([&](int chunk_n, int obs_beg, float * dst) {                                 \
                paged_attention_obs_scores_gqa_kernel<block_t, quantized, G><<<                     \
                    dim3(n_heads / G, n_seq, chunk_n), dim3(head_dim), q_smem, ctx.stream()>>> (    \
                        (const float *) q->data, kv_cache->data, (const int *) block_table->data,   \
                        (const int *) context_lens->data, (const int *) batch_offsets->data,        \
                        (const int *) batch_lens->data, stride_token, stride_head, stride_block,    \
                        block_size, max_blocks, scale, max_context, obs_beg, capture_from_by_seq, dst); \
            });                                                                                     \
        } else
        SNAPKV_OBS_GQA_CASE(8) SNAPKV_OBS_GQA_CASE(7) SNAPKV_OBS_GQA_CASE(6) SNAPKV_OBS_GQA_CASE(5)
        SNAPKV_OBS_GQA_CASE(4) SNAPKV_OBS_GQA_CASE(3) SNAPKV_OBS_GQA_CASE(2) {
            const size_t q_smem = (size_t) 1 * head_dim * sizeof(float);
            launch_obs([&](int chunk_n, int obs_beg, float * dst) {
                paged_attention_obs_scores_kernel<block_t, quantized><<<
                    dim3(n_heads, n_seq, chunk_n), dim3(head_dim), q_smem, ctx.stream()>>>(
                        (const float *) q->data, kv_cache->data, (const int *) block_table->data,
                        (const int *) context_lens->data, (const int *) batch_offsets->data,
                        (const int *) batch_lens->data, stride_token, stride_head, stride_block,
                        n_heads_kv, gqa_ratio, block_size, max_blocks, scale,
                        max_context, obs_beg, capture_from_by_seq, dst);
            });
        }
#undef SNAPKV_OBS_GQA_CASE
    }

    constexpr int reduce_threads = 256;
    paged_attention_snapkv_page_reduce_kernel<<<
        dim3((score_blocks + reduce_threads - 1) / reduce_threads, n_heads, n_seq),
        dim3(reduce_threads), 0, ctx.stream()>>>(
            token_accum, page_accum, score_slots_by_seq, max_context, token_stride,
            block_size, score_blocks, n_heads);
    if (paged_attn_snapkv_sync_debug() && active_context >= 45000) {
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    }
}

static int paged_attn_flash_decode_gqa(int gqa_ratio) {
    for (int g = PAGED_ATTN_FD_GQA_MAX; g > 1; --g) {
        if (gqa_ratio % g == 0) {
            return g;
        }
    }
    return 1;
}

template<typename block_t, bool quantized, int GQA>
static void paged_attn_flash_decode_launch(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        const ggml_tensor * q,
        const ggml_tensor * kv_cache,
        const ggml_tensor * block_table,
        const ggml_tensor * context_lens,
        const ggml_tensor * batch_offsets,
        const ggml_tensor * batch_lens,
        const size_t stride_token,
        const size_t stride_head,
        const size_t stride_block,
        const int n_heads,
        const int n_heads_kv,
        const int n_seq,
        const int n_tokens,
        const int head_dim,
        const int block_size,
        const int max_blocks,
        const int contiguous_base,
        const float scale) {
    const int gqa_ratio = n_heads / n_heads_kv;
    const int n_groups  = n_heads / GQA;
    const int n_warps   = head_dim / 32;
    const int nsm       = ggml_cuda_info().devices[ctx.device].nsm;

    int n_splits = (nsm * PAGED_ATTN_FD_WAVES) / (n_groups * n_seq * n_tokens);
    n_splits = std::min(std::max(n_splits, 1), PAGED_ATTN_FD_SPLITS_MAX);
    n_splits = std::min(n_splits, (max_blocks * block_size + PAGED_ATTN_FD_TILE - 1) / PAGED_ATTN_FD_TILE);
    n_splits = std::max(n_splits, 1);

    ggml_cuda_pool_alloc<float> partial_acc(
        ctx.pool(), (size_t) n_tokens * n_heads * n_splits * head_dim);
    ggml_cuda_pool_alloc<float2> partial_meta(
        ctx.pool(), (size_t) n_tokens * n_heads * n_splits);

    const size_t decode_smem = (size_t) (GQA * head_dim + GQA * PAGED_ATTN_FD_TILE + 2 * GQA * n_warps) * sizeof(float);
    paged_attention_flash_decode_kernel<block_t, quantized, GQA><<<
        dim3(n_groups, n_seq, n_tokens * n_splits), dim3(head_dim), decode_smem, ctx.stream()>>>(
            (const float *) q->data, kv_cache->data, (const int *) block_table->data,
            (const int *) context_lens->data, (const int *) batch_offsets->data,
            (const int *) batch_lens->data, stride_token, stride_head, stride_block,
            n_heads_kv, gqa_ratio, block_size, max_blocks, contiguous_base, n_splits, scale,
            partial_acc.ptr, partial_meta.ptr);

    const size_t combine_smem = (size_t) (n_splits + n_warps) * sizeof(float);
    paged_attention_flash_combine_kernel<<<
        dim3(n_heads, n_seq, n_tokens), dim3(head_dim), combine_smem, ctx.stream()>>>(
            (const int *) batch_offsets->data, (const int *) batch_lens->data,
            partial_acc.ptr, partial_meta.ptr, n_splits, (float *) dst->data);
}

template<typename block_t, bool quantized>
static void paged_attn_flash_decode(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        const ggml_tensor * q,
        const ggml_tensor * kv_cache,
        const ggml_tensor * block_table,
        const ggml_tensor * context_lens,
        const ggml_tensor * batch_offsets,
        const ggml_tensor * batch_lens,
        const size_t stride_token,
        const size_t stride_head,
        const size_t stride_block,
        const int n_heads,
        const int n_heads_kv,
        const int n_seq,
        const int n_tokens,
        const int head_dim,
        const int block_size,
        const int max_blocks,
        const int contiguous_base,
        const float scale) {
#define PAGED_ATTN_FD_CASE(GQA)                                                                  \
    case GQA:                                                                                    \
        paged_attn_flash_decode_launch<block_t, quantized, GQA>(                                 \
            ctx, dst, q, kv_cache, block_table, context_lens, batch_offsets, batch_lens,         \
            stride_token, stride_head, stride_block, n_heads, n_heads_kv, n_seq, n_tokens,       \
            head_dim, block_size, max_blocks, contiguous_base, scale);                           \
        break

    switch (paged_attn_flash_decode_gqa(n_heads / n_heads_kv)) {
        PAGED_ATTN_FD_CASE(8);
        PAGED_ATTN_FD_CASE(7);
        PAGED_ATTN_FD_CASE(6);
        PAGED_ATTN_FD_CASE(5);
        PAGED_ATTN_FD_CASE(4);
        PAGED_ATTN_FD_CASE(3);
        PAGED_ATTN_FD_CASE(2);
        default:
            PAGED_ATTN_FD_CASE(1);
    }
#undef PAGED_ATTN_FD_CASE
}

template<typename block_t, int QK>
static __global__ void paged_attention_quant_write_kernel(
        const float * __restrict__ k_new,
        const float * __restrict__ v_new,
        void * __restrict__ kv_cache,
        const int * __restrict__ write_slots,
        const int * __restrict__ batch_offsets,
        const int * __restrict__ batch_lens,
        const size_t stride_token,
        const size_t stride_head,
        const size_t stride_block,
        const int n_heads_kv,
        const int block_size) {
    extern __shared__ float values[];

    const int head_idx = blockIdx.x;
    const int seq_idx  = blockIdx.y;
    const int tid      = threadIdx.x;
    const int head_dim = blockDim.x;

    const int seq_start  = batch_offsets[seq_idx];
    const int num_tokens = batch_lens[seq_idx];
    char * cache         = static_cast<char *>(kv_cache);
    const int token_idx  = blockIdx.z;
    if (token_idx >= num_tokens) {
        return;
    }

    const int token_batch_idx = seq_start + token_idx;
    const int target_slot     = write_slots[token_batch_idx];
    const int block_id        = target_slot / block_size;
    const int token_in_block  = target_slot % block_size;
    const size_t input_off    = (size_t) token_batch_idx * n_heads_kv * head_dim +
                                (size_t) head_idx * head_dim;

    float * k_values = values;
    float * v_values = values + head_dim;
    k_values[tid] = k_new[input_off + tid];
    v_values[tid] = v_new[input_off + tid];
    __syncthreads();

    // one thread per quantization block, K and V in the same pass
    const int n_qblocks = head_dim / QK;
    if (tid < 2 * n_qblocks) {
        const int is_v = tid >= n_qblocks;
        const int ib   = tid - is_v * n_qblocks;
        const size_t row_offset = (size_t) block_id * stride_block +
                                  (size_t) (is_v ? n_heads_kv + head_idx : head_idx) * stride_head +
                                  (size_t) token_in_block * stride_token;
        paged_quantize_block((is_v ? v_values : k_values) + ib * QK,
                             reinterpret_cast<block_t *>(cache + row_offset) + ib);
    }
}

template<typename block_t, int QK>
static __global__ void paged_attention_quant_decode_kernel(
        const float * __restrict__ q,
        const void * __restrict__ kv_cache,
        const int * __restrict__ block_table,
        const int * __restrict__ context_lens,
        const int * __restrict__ batch_offsets,
        const int * __restrict__ batch_lens,
        const size_t stride_token,
        const size_t stride_head,
        const size_t stride_block,
        const int n_heads_kv,
        const int block_size,
        const int max_blocks,
        const float scale,
        float * __restrict__ out) {
    extern __shared__ float smem[];

    const int head_idx = blockIdx.x;
    const int seq_idx  = blockIdx.y;
    const int tid      = threadIdx.x;
    const int n_heads  = gridDim.x;
    const int head_dim = blockDim.x;
    const int kv_head_idx = head_idx / (n_heads / n_heads_kv);
    const char * cache = static_cast<const char *>(kv_cache);

    const int seq_start      = batch_offsets[seq_idx];
    const int num_new_tokens = batch_lens[seq_idx];
    const int token_idx      = blockIdx.z;
    if (token_idx >= num_new_tokens) {
        return;
    }
    const int token_batch_idx = seq_start + token_idx;
    const int ctx_len         = context_lens[seq_idx];
    const int q_pos           = (ctx_len - num_new_tokens) + token_idx;
    const size_t q_offset     = (size_t) token_batch_idx * n_heads * head_dim + (size_t) head_idx * head_dim;
    const float q_val        = q[q_offset + tid] * scale;
    float qk_max  = -FLT_MAX;
    float exp_sum = 0.0f;
    float acc     = 0.0f;

    for (int token = 0; token <= q_pos; ++token) {
        const int bid            = token / block_size;
        const int token_in_block = token % block_size;
        const int physical_block = block_table[seq_idx * max_blocks + bid];
        if (physical_block < 0) {
            continue;  // evicted page: no contribution
        }
        const size_t k_offset    = (size_t) physical_block * stride_block + (size_t) kv_head_idx * stride_head +
                                   (size_t) token_in_block * stride_token;
        const size_t v_offset    = (size_t) physical_block * stride_block +
                                   (size_t) (n_heads_kv + kv_head_idx) * stride_head +
                                   (size_t) token_in_block * stride_token;
        const block_t * k_row    = reinterpret_cast<const block_t *>(cache + k_offset);
        const block_t * v_row    = reinterpret_cast<const block_t *>(cache + v_offset);
        const float qk           = paged_block_reduce_sum(q_val * paged_dequantize_value(k_row, tid), smem, tid, head_dim);
        const float qk_max_new   = fmaxf(qk_max, qk);
        const float exp_old      = __expf(qk_max - qk_max_new);
        const float exp_new      = __expf(qk - qk_max_new);
        exp_sum = exp_sum * exp_old + exp_new;
        acc     = acc * exp_old + exp_new * paged_dequantize_value(v_row, tid);
        qk_max  = qk_max_new;
    }

    const size_t out_idx = (size_t) token_batch_idx * n_heads * head_dim +
                           (size_t) head_idx * head_dim + tid;
    out[out_idx] = acc / (exp_sum + 1e-6f);
}

template<typename block_t, int QK>
static void ggml_cuda_op_paged_attn_quantized(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q             = dst->src[0];
    const ggml_tensor * k_new         = dst->src[1];
    const ggml_tensor * v_new         = dst->src[2];
    const ggml_tensor * kv_cache      = dst->src[3];
    const ggml_tensor * block_table   = dst->src[5];
    const ggml_tensor * write_slots   = dst->src[6];
    const ggml_tensor * context_lens  = dst->src[7];
    const ggml_tensor * batch_offsets = dst->src[8];
    const ggml_tensor * batch_lens    = dst->src[9];

    const float * op_params_f = (const float *) (dst->op_params);
    const float scale         = op_params_f[0];
    const int block_size      = ((const int32_t *) (op_params_f + 1))[0];
    const int max_blocks      = ((const int32_t *) (op_params_f + 2))[0];
    const int head_dim        = q->ne[0];
    const int n_heads         = q->ne[1];
    const int n_seq           = batch_lens->ne[0];
    const int n_heads_kv      = k_new->ne[1];
    const int n_tokens        = q->ne[2];
    const bool snapkv_streaming = paged_attn_snapkv_streaming(op_params_f);
    const int snapkv_score_blocks = paged_attn_snapkv_score_blocks(op_params_f, max_blocks);

    GGML_ASSERT(head_dim % QK == 0 && "paged quantized KV head_dim must align to its quantization block size");
    GGML_ASSERT(head_dim % 32 == 0 && head_dim <= 1024);
    GGML_ASSERT(n_heads != 0 && n_heads_kv != 0 && n_heads % n_heads_kv == 0);
    GGML_ASSERT(n_tokens <= 65535 && "paged CUDA token batch exceeds grid.z limit");

    const size_t stride_token = kv_cache->nb[1];
    const size_t stride_head  = kv_cache->nb[2];
    const size_t stride_block = kv_cache->nb[3];
    const size_t write_smem   = (size_t) 2 * head_dim * sizeof(float);

    paged_attention_quant_write_kernel<block_t, QK><<<dim3(n_heads_kv, n_seq, n_tokens), dim3(head_dim), write_smem, ctx.stream()>>>(
        (const float *) k_new->data, (const float *) v_new->data, kv_cache->data,
        (const int *) write_slots->data, (const int *) batch_offsets->data, (const int *) batch_lens->data,
        stride_token, stride_head, stride_block, n_heads_kv, block_size);

    float * snapkv_accum = paged_attn_snapkv_accum(op_params_f);
    float * snapkv_token_scores = paged_attn_snapkv_token_scores(op_params_f);
    const int * snapkv_capture_from = paged_attn_snapkv_capture_from(op_params_f);
    const int * snapkv_score_slots = paged_attn_snapkv_score_slots(op_params_f);
    if (snapkv_accum != nullptr && snapkv_token_scores != nullptr) {
        paged_attn_snapkv_capture<block_t, true>(
            ctx, dst, q, kv_cache, block_table, context_lens, batch_offsets, batch_lens,
            stride_token, stride_head, stride_block, n_heads, n_heads_kv, n_seq, n_tokens,
            head_dim, block_size, max_blocks, snapkv_score_blocks, scale, snapkv_capture_from, snapkv_score_slots,
            snapkv_accum, snapkv_token_scores);
    }

    if (!snapkv_streaming && ggml_cuda_paged_attn_prefill<block_t, true>(
            ctx, dst, q, k_new, kv_cache, block_table, batch_lens,
            stride_token, stride_head, stride_block, block_size, max_blocks, scale)) {
        return;
    }

    const int contiguous_base = n_seq == 1
        ? ((const int32_t *) (op_params_f + 1))[13] - 1 : -1;
    if (n_tokens <= PAGED_ATTN_PARALLEL_MAX_TOKENS || snapkv_streaming) {
        paged_attn_flash_decode<block_t, true>(
            ctx, dst, q, kv_cache, block_table, context_lens, batch_offsets, batch_lens,
            stride_token, stride_head, stride_block, n_heads, n_heads_kv, n_seq, n_tokens,
            head_dim, block_size, max_blocks, contiguous_base, scale);
    } else {
        const size_t reduce_smem = (size_t) ((head_dim + 31) / 32) * sizeof(float);
        paged_attention_quant_decode_kernel<block_t, QK><<<
            dim3(n_heads, n_seq, n_tokens), dim3(head_dim), reduce_smem, ctx.stream()>>>(
                (const float *) q->data, kv_cache->data, (const int *) block_table->data,
                (const int *) context_lens->data, (const int *) batch_offsets->data, (const int *) batch_lens->data,
                stride_token, stride_head, stride_block, n_heads_kv, block_size, max_blocks, scale,
                (float *) dst->data);
    }
}

void ggml_cuda_op_paged_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q             = dst->src[0];
    const ggml_tensor * k_new         = dst->src[1];
    const ggml_tensor * v_new         = dst->src[2];
    const ggml_tensor * kv_cache      = dst->src[3];  // KV interleaved layout
    const ggml_tensor * block_table   = dst->src[5];
    const ggml_tensor * write_slots   = dst->src[6];
    const ggml_tensor * context_lens  = dst->src[7];
    const ggml_tensor * batch_offsets = dst->src[8];
    const ggml_tensor * batch_lens    = dst->src[9];

    const float * op_params_f = (const float *) (dst->op_params);
    const float   scale       = op_params_f[0];
    const int     block_size  = ((const int32_t *) (op_params_f + 1))[0];
    const int     max_blocks  = ((const int32_t *) (op_params_f + 2))[0];

    const int head_dim   = q->ne[0];
    const int n_heads    = q->ne[1];
    const int n_seq      = batch_lens->ne[0];
    const int n_heads_kv = k_new->ne[1];
    const int n_tokens   = q->ne[2];
    const bool snapkv_streaming = paged_attn_snapkv_streaming(op_params_f);
    const int snapkv_score_blocks = paged_attn_snapkv_score_blocks(op_params_f, max_blocks);

    GGML_ASSERT(n_heads != 0 && "n_head cannot be 0.");
    GGML_ASSERT(n_heads_kv != 0 && "n_heads_kv cannot be 0.");
    GGML_ASSERT(head_dim <= 1024 && "head_dim exceeds maximum supported (1024)");
    GGML_ASSERT(head_dim % 32 == 0 && "paged CUDA head_dim must be a multiple of 32");
    GGML_ASSERT(n_heads % n_heads_kv == 0 && "n_heads must be divisible by n_heads_kv");
    GGML_ASSERT(n_tokens <= 65535 && "paged CUDA token batch exceeds grid.z limit");

    float * snapkv_accum = paged_attn_snapkv_accum(op_params_f);
    float * snapkv_token_scores = paged_attn_snapkv_token_scores(op_params_f);
    const int * snapkv_capture_from = paged_attn_snapkv_capture_from(op_params_f);
    const int * snapkv_score_slots = paged_attn_snapkv_score_slots(op_params_f);

    if (kv_cache->type == GGML_TYPE_Q4_0) {
        ggml_cuda_op_paged_attn_quantized<block_q4_0, QK4_0>(ctx, dst);
        return;
    }
    if (kv_cache->type == GGML_TYPE_Q8_0) {
        ggml_cuda_op_paged_attn_quantized<block_q8_0, QK8_0>(ctx, dst);
        return;
    }
    GGML_ASSERT(kv_cache->type == GGML_TYPE_F16 && "unsupported paged KV type for CUDA");

    // Extracting strides
    const size_t stride_token = kv_cache->nb[1] / sizeof(half);
    const size_t stride_head  = kv_cache->nb[2] / sizeof(half);
    const size_t stride_block = kv_cache->nb[3] / sizeof(half);

    dim3 block_dims(head_dim);       // one thread per dimension of head
    dim3 grid_dims(n_heads, n_seq, n_tokens);  // one block per head, sequence, and token

    // Write kernel - Grid (n_heads_kv, n_seq), Block (head_dim)
    paged_attention_write_kernel<<<dim3(n_heads_kv, n_seq, n_tokens), dim3(head_dim), 0, ctx.stream()>>>(
        (const float *) k_new->data, (const float *) v_new->data, (half *) kv_cache->data,
        (const int *) write_slots->data, (const int *) batch_offsets->data, (const int *) batch_lens->data,
        stride_token, stride_head, stride_block, n_heads_kv, block_size);

    if (snapkv_accum != nullptr && snapkv_token_scores != nullptr) {
        paged_attn_snapkv_capture<half, false>(
            ctx, dst, q, kv_cache, block_table, context_lens, batch_offsets, batch_lens,
            stride_token, stride_head, stride_block, n_heads, n_heads_kv, n_seq, n_tokens,
            head_dim, block_size, max_blocks, snapkv_score_blocks, scale, snapkv_capture_from, snapkv_score_slots,
            snapkv_accum, snapkv_token_scores);
    }

    if (!snapkv_streaming && ggml_cuda_paged_attn_prefill<half, false>(
            ctx, dst, q, k_new, kv_cache, block_table, batch_lens,
            stride_token, stride_head, stride_block, block_size, max_blocks, scale)) {
        return;
    }

    const int contiguous_base = n_seq == 1
        ? ((const int32_t *) (op_params_f + 1))[13] - 1 : -1;
    if (n_tokens <= PAGED_ATTN_PARALLEL_MAX_TOKENS || snapkv_streaming) {
        paged_attn_flash_decode<half, false>(
            ctx, dst, q, kv_cache, block_table, context_lens, batch_offsets, batch_lens,
            stride_token, stride_head, stride_block, n_heads, n_heads_kv, n_seq, n_tokens,
            head_dim, block_size, max_blocks, contiguous_base, scale);
    } else {
        const size_t reduce_smem = (size_t) ((head_dim + 31) / 32) * sizeof(float);
        paged_attention_decode_kernel<<<grid_dims, block_dims, reduce_smem, ctx.stream()>>>(
            (const float *) q->data, (const half *) kv_cache->data, (const int *) block_table->data,
            (const int *) context_lens->data, (const int *) batch_offsets->data, (const int *) batch_lens->data,
            stride_token, stride_head, stride_block, n_heads_kv, block_size, max_blocks, scale,
            (float *) dst->data);
    }
}
