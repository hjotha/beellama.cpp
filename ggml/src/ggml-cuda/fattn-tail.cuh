#pragma once

#include "fattn-common.cuh"
#include "fattn-kvarn-dispatch.cuh"
#include "fattn-mma-kvarn-case-decl.cuh"
#if defined(GGML_CUDA_KVARN)
#include "kvarn.cuh"
#endif

// ---------------------------------------------------------------------------
// KDISC diagnostic records (behind F4_RP_KDISC only; production never sets the
// flag, so g_kdisc_on stays 0 and no record is written).
// The merge kernel writes one record per (iq_packed, ih, ia) for head0/query0
// with a marker; the host copies them out after a stream synchronize.
// ---------------------------------------------------------------------------
struct ggml_kdisc_rec {
    int      marker;        // 0xAA55AA51 when actually written
    int      iq_packed, ih, ia, iq, is, iq_global;
    int      body_valid, tail_valid;
    float    body_max, body_norm;   // body partial meta
    float    tail_max, tail_norm;   // tail partial meta
    float    m, wb, wt, denom;      // values the merge actually used
    float    body_o0, tail_o0;      // first lane of each partial row
    float    out_o0, out_lse;       // what the merge produced
};

struct ggml_kdisc_pre {
    int    marker;                 // 0x51A1CE01 when written
    int    n_tail;
    int    slot0, row0;            // resolved ring slot/row for token 0
    float  q[4];                   // qrow[0..3]
    float  k0;                     // K value consumed for token 0, lane 0
    float  scores[32];             // scores BEFORE expf conversion
    float  tail_max_raw;           // reduction result before weight conversion
    float  dot0, mask0, score0;    // token0: pre-scale dot, mask, final score
    int    ih_k, d_k;              // head mapping + reduction length
    float  scale;
};

__device__ int                g_kdisc_on = 0;
__device__ int                g_kdisc_n  = 0;
__device__ ggml_kdisc_rec     g_kdisc_rec[8];
__device__ int                g_kdisc_np = 0;
__device__ float              g_kdisc_dot0 = 0.0f, g_kdisc_mask0 = 0.0f, g_kdisc_score0 = 0.0f;
__device__ ggml_kdisc_pre     g_kdisc_pre[4];

template<typename T>
static __global__ void k_flash_attn_ext_tail_pack_arenas(
        const char * src, const char * current, T * dst, const int32_t * run_desc,
        int d, int tail_stride, int n_head, int n_active,
        int desc_stride, int history_slots,
        size_t nb0, size_t nb1, size_t nb2,
        size_t current_nb0, size_t current_nb1, size_t current_nb2) {
    const size_t n = size_t(d)*tail_stride*n_head*n_active;
    for (size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
            i < n; i += size_t(blockDim.x)*gridDim.x) {
        size_t rem = i;
        const int id = int(rem % d); rem /= d;
        const int it = int(rem % tail_stride); rem /= tail_stride;
        const int ih = int(rem % n_head); rem /= n_head;
        const int ia = int(rem);
        const int32_t * desc = run_desc + desc_stride*ia;
        T value = {};
        if (it < desc[4]) {
            const int slot = desc[6 + it];
            const bool from_current = current && slot >= history_slots;
            const int row = from_current ? slot - history_slots : slot;
            const char * base = from_current ? current : src;
            value = *reinterpret_cast<const T *>(base +
                size_t(id)*(from_current ? current_nb0 : nb0) +
                size_t(row)*(from_current ? current_nb1 : nb1) +
                size_t(ih)*(from_current ? current_nb2 : nb2));
        }
        dst[i] = value;
    }
}

template<typename V>
static __global__ void k_flash_attn_ext_tail_pack_arenas_vec(
        const char * src, const char * current, V * dst, const int32_t * run_desc,
        int n_vec, int tail_stride, int n_head, int n_active,
        int desc_stride, int history_slots,
        size_t nb1, size_t nb2, size_t current_nb1, size_t current_nb2) {
    const size_t n = size_t(n_vec)*tail_stride*n_head*n_active;
    for (size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
            i < n; i += size_t(blockDim.x)*gridDim.x) {
        size_t rem = i;
        const int iv = int(rem % n_vec); rem /= n_vec;
        const int it = int(rem % tail_stride); rem /= tail_stride;
        const int ih = int(rem % n_head); rem /= n_head;
        const int ia = int(rem);
        const int32_t * desc = run_desc + desc_stride*ia;
        V value = {};
        if (it < desc[4]) {
            const int slot = desc[6 + it];
            const bool from_current = current && slot >= history_slots;
            const int row = from_current ? slot - history_slots : slot;
            const char * base = from_current ? current : src;
            value = *reinterpret_cast<const V *>(base + size_t(iv)*sizeof(V) +
                    size_t(row)*(from_current ? current_nb1 : nb1) +
                    size_t(ih)*(from_current ? current_nb2 : nb2));
        }
        dst[i] = value;
    }
}

static __global__ void k_flash_attn_ext_tail_pack_body_rows(
        const char * src, uint8_t * dst, const int32_t * run_desc,
        int row_bytes, int n_kv, int body_stride, int n_head, int n_active,
        int desc_stride, int body_map_offset, size_t nb1, size_t nb2, size_t nb3) {
    const size_t n = size_t(row_bytes)*body_stride*n_head*n_active;
    for (size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
            i < n; i += size_t(blockDim.x)*gridDim.x) {
        size_t rem = i;
        const int ib = int(rem % row_bytes); rem /= row_bytes;
        const int it = int(rem % body_stride); rem /= body_stride;
        const int ih = int(rem % n_head); rem /= n_head;
        const int ia = int(rem);
        const int32_t * desc = run_desc + desc_stride*ia;
        uint8_t value = 0;
        if (it < desc[5]) {
            const int flat = desc[body_map_offset + it];
            const int is = flat/n_kv;
            const int ik = flat - is*n_kv;
            value = *(reinterpret_cast<const uint8_t *>(src) +
                    size_t(ik)*nb1 + size_t(ih)*nb2 + size_t(is)*nb3 + ib);
        }
        dst[i] = value;
    }
}

static __global__ void k_flash_attn_ext_tail_pack_body_mask(
        const half * mask, const int32_t * query_order, const int32_t * run_desc,
        half * mask_packed, int n_kv, int n_query, int n_stream,
        int body_stride, int q_max, int n_active, int desc_stride, int body_map_offset,
        size_t m_nb1, size_t m_nb3) {
    const size_t n = size_t(body_stride)*q_max*n_active;
    for (size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
            i < n; i += size_t(blockDim.x)*gridDim.x) {
        size_t rem = i;
        const int it = int(rem % body_stride); rem /= body_stride;
        const int iq_packed = int(rem % q_max); rem /= q_max;
        const int ia = int(rem);
        const int iq_global = query_order[ia*q_max + iq_packed];
        const int32_t * desc = run_desc + desc_stride*ia;
        half value = __float2half(-INFINITY);
        if (iq_global >= 0 && iq_global < n_query*n_stream && it < desc[5]) {
            const int iq = iq_global % n_query;
            const int flat = desc[body_map_offset + it];
            const int is = flat/n_kv;
            const int ik = flat - is*n_kv;
            value = *reinterpret_cast<const half *>(reinterpret_cast<const char *>(mask) +
                    size_t(ik)*sizeof(half) + size_t(iq)*m_nb1 + size_t(is)*m_nb3);
        }
        mask_packed[i] = value;
    }
}

static __global__ void k_flash_attn_ext_tail_pack_q_mask(
        const float * q, const half * mask, const int32_t * query_order,
        float * q_packed, half * mask_packed,
        int d, int n_query, int n_head, int n_stream,
        int mask_stride, int tail_stride, int q_max, int n_active,
        size_t q_nb1, size_t q_nb2, size_t q_nb3,
        size_t m_nb1, size_t m_nb3) {
    const size_t n_q = size_t(d)*q_max*n_head*n_active;
    for (size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
            i < n_q; i += size_t(blockDim.x)*gridDim.x) {
        size_t rem = i;
        const int id = int(rem % d); rem /= d;
        const int iq_packed = int(rem % q_max); rem /= q_max;
        const int ih = int(rem % n_head); rem /= n_head;
        const int ia = int(rem);
        const int iq_global = query_order[ia*q_max + iq_packed];
        float value = 0.0f;
        if (iq_global >= 0 && iq_global < n_query*n_stream) {
            const int is = iq_global/n_query;
            const int iq = iq_global - is*n_query;
            value = *reinterpret_cast<const float *>(reinterpret_cast<const char *>(q) +
                size_t(id)*sizeof(float) + size_t(iq)*q_nb1 + size_t(ih)*q_nb2 + size_t(is)*q_nb3);
        }
        q_packed[i] = value;
    }

    const size_t n_m = size_t(tail_stride)*q_max*n_active;
    for (size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
            i < n_m; i += size_t(blockDim.x)*gridDim.x) {
        size_t rem = i;
        const int it = int(rem % tail_stride); rem /= tail_stride;
        const int iq_packed = int(rem % q_max); rem /= q_max;
        const int ia = int(rem);
        const int iq_global = query_order[ia*q_max + iq_packed];
        half value = __float2half(-INFINITY);
        if (iq_global >= 0 && iq_global < n_query*n_stream && it < mask_stride) {
            const int is = iq_global/n_query;
            const int iq = iq_global - is*n_query;
            value = *reinterpret_cast<const half *>(reinterpret_cast<const char *>(mask) +
                size_t(it)*sizeof(half) + size_t(iq)*m_nb1 + size_t(is)*m_nb3);
        }
        mask_packed[i] = value;
    }
}


template<typename T>
static __device__ __forceinline__ float tail_value_to_float(T value);

template<>
__device__ __forceinline__ float tail_value_to_float<half>(half value) {
    return __half2float(value);
}

template<>
__device__ __forceinline__ float tail_value_to_float<nv_bfloat16>(nv_bfloat16 value) {
    return __bfloat162float(value);
}

// Small indexed tails are faster when consumed in place.  This kernel avoids
// materializing compact K/V/Q/mask tensors and combines the tail partial with
// the already-computed body partial in one launch.  One warp computes each KQ
// dot while all threads cooperatively normalize and accumulate V.
template<typename TK, typename TV, int max_tail>
static __global__ void k_flash_attn_ext_tail_indexed_small(
        const float * q, const char * kt, const char * vt,
        const char * kt_current, const char * vt_current, const half * mt,
        const int32_t * query_order, const int32_t * run_desc,
        const float * body, const float2 * body_meta, float * dst,
        float * lse_out,
        int d_k, int d_v, int n_query, int n_head, int n_stream,
        int q_max, int n_active, int n_head_k, int n_head_v, int desc_stride,
        float scale, float max_bias, float logit_softcap,
        size_t q_nb1, size_t q_nb2, size_t q_nb3,
        int history_slots,
        size_t kt_nb0, size_t kt_nb1, size_t kt_nb2,
        size_t vt_nb0, size_t vt_nb1, size_t vt_nb2,
        size_t ktc_nb0, size_t ktc_nb1, size_t ktc_nb2,
        size_t vtc_nb0, size_t vtc_nb1, size_t vtc_nb2,
        size_t mt_nb1, size_t mt_nb3,
        bool tail_bodyless, bool body_packed,
        size_t body_nb1, size_t body_nb2, size_t body_nb3,
        size_t dst_nb1, size_t dst_nb2, size_t dst_nb3,
        size_t lse_nb0, size_t lse_nb1, size_t lse_nb2) {
    const int iq_packed = blockIdx.x;
    const int ih = blockIdx.y;
    const int ia = blockIdx.z;
    if (iq_packed >= q_max || ih >= n_head || ia >= n_active) {
        return;
    }
    const int iq_global = query_order[ia*q_max + iq_packed];
    if (iq_global < 0 || iq_global >= n_query*n_stream) {
        return;
    }
    const int is = iq_global/n_query;
    const int iq = iq_global - is*n_query;
    const int32_t * desc = run_desc + desc_stride*ia;
    const int n_tail = desc[4];
    if (n_tail < 0 || n_tail > max_tail) {
        return;
    }

    __shared__ float scores[max_tail];
    __shared__ float reduction[256];
    __shared__ float tail_max_shared;
    __shared__ float tail_sum_shared;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int ih_k = ih/(n_head/n_head_k);
    const int ih_v = ih/(n_head/n_head_v);
    const float * qrow = reinterpret_cast<const float *>(
            reinterpret_cast<const char *>(q) + size_t(iq)*q_nb1 + size_t(ih)*q_nb2 + size_t(is)*q_nb3);

    const uint32_t nh_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));
    const float m0 = exp2f(-max_bias/float(nh_log2));
    const float m1 = exp2f(-(max_bias/2.0f)/float(nh_log2));
    const float slope = max_bias > 0.0f ?
            (ih < int(nh_log2) ? powf(m0, ih + 1) : powf(m1, 2*(ih - int(nh_log2)) + 1)) : 1.0f;

    for (int token = warp; token < n_tail; token += 8) {
        const float mask = __half2float(*reinterpret_cast<const half *>(
                reinterpret_cast<const char *>(mt) + size_t(token)*sizeof(half) +
                size_t(iq)*mt_nb1 + size_t(is)*mt_nb3));
        if (mask == -INFINITY) {
            if (lane == 0) { scores[token] = -INFINITY; }
            continue;
        }
        const int slot = desc[6 + token];
        const bool from_current = kt_current && slot >= history_slots;
        const int row = from_current ? slot - history_slots : slot;
        const char * k_source = from_current ? kt_current : kt;
        float dot = 0.0f;
        for (int d = lane; d < d_k; d += 32) {
            const TK kval = *reinterpret_cast<const TK *>(
                    k_source +
                    size_t(d)*(from_current ? ktc_nb0 : kt_nb0) +
                    size_t(row)*(from_current ? ktc_nb1 : kt_nb1) +
                    size_t(ih_k)*(from_current ? ktc_nb2 : kt_nb2));
            dot += qrow[d]*tail_value_to_float(kval);
        }
        dot = warp_reduce_sum(dot);
        if (lane == 0) {
            float score = dot*scale;
            if (logit_softcap != 0.0f) {
                score = logit_softcap*tanhf(score/logit_softcap);
            }
            scores[token] = score + slope*mask;
            // Must be guarded for the SAME block the record is taken from:
            // grid.y = n_head, so without ih==0 the six head blocks race on
            // these device globals and the record would mix heads.
            if (g_kdisc_on && token == 0 && iq_packed == 0 && ih == 0 && ia == 0) {
                g_kdisc_dot0 = dot; g_kdisc_mask0 = mask; g_kdisc_score0 = scores[0];
            }
        }
    }
    __syncthreads();

    // KDISC: snapshot scores BEFORE they are overwritten by expf weights, plus
    // the exact q/K values and the resolved slot/row the kernel consumed.
    if (g_kdisc_on && tid == 0 && iq_packed == 0 && ih == 0 && ia == 0 &&
            n_tail > 0 && isfinite(scores[0])) {
        const int slot0 = desc[6 + 0];
        const bool cur0 = kt_current && slot0 >= history_slots;
        const int  row0 = cur0 ? slot0 - history_slots : slot0;
        const char * src0 = cur0 ? kt_current : kt;
        const TK kv0 = *reinterpret_cast<const TK *>(
                src0 + size_t(row0)*(cur0 ? kt_nb1 : kt_nb1) + size_t(ih_k)*(cur0 ? kt_nb2 : kt_nb2));
        const int ps = atomicAdd(&g_kdisc_np, 1);
        if (ps >= 0 && ps < 4) {
            ggml_kdisc_pre r;
            r.marker = 0x51A1CE01;
            r.n_tail = n_tail;
            r.slot0 = slot0; r.row0 = row0;
            r.q[0] = qrow[0]; r.q[1] = qrow[1]; r.q[2] = qrow[2]; r.q[3] = qrow[3];
            r.k0 = tail_value_to_float(kv0);
            for (int i = 0; i < 32; ++i) { r.scores[i] = (i < n_tail) ? scores[i] : 0.0f; }
            r.tail_max_raw = reduction[0];
            r.dot0 = g_kdisc_dot0; r.mask0 = g_kdisc_mask0; r.score0 = g_kdisc_score0;
            r.ih_k = ih_k; r.d_k = d_k; r.scale = scale;
            g_kdisc_pre[ps] = r;
        }
    }

    float local_max = -INFINITY;
    for (int token = tid; token < n_tail; token += blockDim.x) {
        local_max = fmaxf(local_max, scores[token]);
    }
    reduction[tid] = local_max;
    __syncthreads();
    for (int width = 128; width > 0; width >>= 1) {
        if (tid < width) {
            reduction[tid] = fmaxf(reduction[tid], reduction[tid + width]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        tail_max_shared = reduction[0];
    }
    __syncthreads();

    float weight = 0.0f;
    for (int token = tid; token < n_tail; token += blockDim.x) {
        const float token_weight = isfinite(scores[token]) && isfinite(tail_max_shared) ?
                expf(scores[token] - tail_max_shared) : 0.0f;
        scores[token] = token_weight;
        weight += token_weight;
    }
    reduction[tid] = weight;
    __syncthreads();
    for (int width = 128; width > 0; width >>= 1) {
        if (tid < width) {
            reduction[tid] += reduction[tid + width];
        }
        __syncthreads();
    }
    if (tid == 0) {
        tail_sum_shared = reduction[0];
    }
    __syncthreads();

    const size_t body_row_index = body_packed ?
            (size_t(ia)*q_max + iq_packed)*n_head + ih :
            (size_t(is)*n_query + iq)*n_head + ih;
    const float2 bm = tail_bodyless ? make_float2(0.0f, 0.0f) : body_meta[body_row_index];
    const bool bv = bm.y > 0.0f && isfinite(bm.x) && isfinite(bm.y);
    const bool tv = tail_sum_shared > 0.0f && isfinite(tail_max_shared);
    const float global_max = bv && tv ? fmaxf(bm.x, tail_max_shared) :
            (bv ? bm.x : (tv ? tail_max_shared : -INFINITY));
    const float wb = bv ? bm.y*expf(bm.x - global_max) : 0.0f;
    const float wt = tv ? expf(tail_max_shared - global_max) : 0.0f;
    const float denom = wb + tail_sum_shared*wt;
    const char * brow = tail_bodyless ? nullptr : reinterpret_cast<const char *>(body) + size_t(ih)*body_nb1 +
            size_t(body_packed ? iq_packed : iq)*body_nb2 + size_t(body_packed ? ia : is)*body_nb3;
    char * drow = reinterpret_cast<char *>(dst) + size_t(ih)*dst_nb1 +
            size_t(iq)*dst_nb2 + size_t(is)*dst_nb3;
    for (int d = tid; d < d_v; d += blockDim.x) {
        float tail_acc = 0.0f;
        for (int token = 0; token < n_tail; ++token) {
            const float weight = scores[token];
            if (weight == 0.0f) { continue; }
            const int slot = desc[6 + token];
            const bool from_current = vt_current && slot >= history_slots;
            const int row = from_current ? slot - history_slots : slot;
            const char * v_source = from_current ? vt_current : vt;
            const TV vval = *reinterpret_cast<const TV *>(
                    v_source +
                    size_t(d)*(from_current ? vtc_nb0 : vt_nb0) +
                    size_t(row)*(from_current ? vtc_nb1 : vt_nb1) +
                    size_t(ih_v)*(from_current ? vtc_nb2 : vt_nb2));
            tail_acc += tail_value_to_float(vval)*weight;
        }
        const float body_value = bv ? *reinterpret_cast<const float *>(brow + size_t(d)*sizeof(float)) : 0.0f;
        *reinterpret_cast<float *>(drow + size_t(d)*sizeof(float)) =
                denom > 0.0f ? (body_value*wb + tail_acc*wt)/denom : 0.0f;
    }
    if (tid == 0 && lse_out != nullptr) {
        // Position-split LSE is declared from the query layout as
        // ne = {q->ne[1], q->ne[2], q->ne[3]} (plan section 3.4), so its ne[1]
        // axis is the query and ne[2] the head: index it by strides, exactly
        // like O, instead of assuming a row order.
        *reinterpret_cast<float *>(reinterpret_cast<char *>(lse_out) +
            size_t(ih)*lse_nb0 + size_t(iq)*lse_nb1 + size_t(is)*lse_nb2) =
                denom > 0.0f ? (global_max + logf(denom)) : -INFINITY;
    }
    // KDISC (F4_RP_KDISC only): this is the ACTIVE body+tail combination for
    // small indexed tails -- one launch consumes the body partial and produces
    // the merged O/LSE. Record head0/query0 so the first wrong value can be
    // attributed to body partial, tail partial, normalization or the combine.
    if (tid == 0 && g_kdisc_on && iq_packed == 0 && ih == 0 && ia == 0) {
        const int slot = atomicAdd(&g_kdisc_n, 1);
        if (slot >= 0 && slot < 8) {
            ggml_kdisc_rec r;
            r.marker     = 0xAA55AA51;
            r.iq_packed  = iq_packed; r.ih = ih; r.ia = ia;
            r.iq         = iq;        r.is = is; r.iq_global = iq_global;
            r.body_valid = bv ? 1 : 0;
            r.tail_valid = tv ? 1 : 0;
            r.body_max   = bm.x;
            r.body_norm  = bm.y;
            r.tail_max   = tail_max_shared;
            r.tail_norm  = tail_sum_shared;
            r.m          = global_max;
            r.wb         = wb;
            r.wt         = wt;
            r.denom      = denom;
            r.body_o0    = (bv && brow) ? *reinterpret_cast<const float *>(brow)
                                        : 0.0f;
            r.tail_o0    = 0.0f;   // tail accumulates in registers, not a row
            r.out_o0     = *reinterpret_cast<const float *>(drow);
            r.out_lse    = denom > 0.0f ? (global_max + logf(denom)) : -INFINITY;
            g_kdisc_rec[slot] = r;
        }
    }
}

static __global__ void k_flash_attn_ext_tail_partials_merge(
        const float * body, const float * tail, float * dst,
        float * lse_out,
        const float2 * body_meta, const float2 * tail_meta, const int32_t * query_order,
        int d, int n_query, int n_head, int n_stream, int q_max, int n_active,
        bool body_packed,
        size_t body_nb1, size_t body_nb2, size_t body_nb3,
        size_t tail_nb1, size_t tail_nb2, size_t tail_nb3,
        size_t dst_nb1, size_t dst_nb2, size_t dst_nb3,
        size_t lse_nb0, size_t lse_nb1, size_t lse_nb2) {
    const int iq_packed = blockIdx.x;
    const int ih = blockIdx.y;
    const int ia = blockIdx.z;
    if (iq_packed >= q_max || ih >= n_head || ia >= n_active) {
        return;
    }
    const int iq_global = query_order[ia*q_max + iq_packed];
    if (iq_global < 0 || iq_global >= n_query*n_stream) {
        return;
    }
    const int is = iq_global/n_query;
    const int iq = iq_global - is*n_query;
    const size_t body_row_index = body_packed ?
        (size_t(ia)*q_max + iq_packed)*n_head + ih :
        (size_t(is)*n_query + iq)*n_head + ih;
    const size_t tail_row_index = (size_t(ia)*q_max + iq_packed)*n_head + ih;
    const float2 bm = body_meta[body_row_index];
    const float2 tm = tail_meta[tail_row_index];
    const bool bv = bm.y > 0.0f && isfinite(bm.x) && isfinite(bm.y);
    const bool tv = tm.y > 0.0f && isfinite(tm.x) && isfinite(tm.y);
    const float m = bv && tv ? fmaxf(bm.x, tm.x) : (bv ? bm.x : (tv ? tm.x : -INFINITY));
    const float wb = bv ? bm.y*expf(bm.x - m) : 0.0f;
    const float wt = tv ? tm.y*expf(tm.x - m) : 0.0f;
    const float denom = wb + wt;
    const char * brow = reinterpret_cast<const char *>(body) +
        size_t(ih)*body_nb1 +
        size_t(body_packed ? iq_packed : iq)*body_nb2 +
        size_t(body_packed ? ia : is)*body_nb3;
    const char * trow = reinterpret_cast<const char *>(tail) +
        size_t(ih)*tail_nb1 + size_t(iq_packed)*tail_nb2 + size_t(ia)*tail_nb3;
    char * drow = reinterpret_cast<char *>(dst) +
        size_t(ih)*dst_nb1 + size_t(iq)*dst_nb2 + size_t(is)*dst_nb3;
    for (int id = threadIdx.x; id < d; id += blockDim.x) {
        float numerator = 0.0f;
        if (bv) {
            numerator += *reinterpret_cast<const float *>(brow + size_t(id)*sizeof(float))*wb;
        }
        if (tv) {
            numerator += *reinterpret_cast<const float *>(trow + size_t(id)*sizeof(float))*wt;
        }
        *reinterpret_cast<float *>(drow + size_t(id)*sizeof(float)) = denom > 0.0f ? numerator/denom : 0.0f;
    }
    if (threadIdx.x == 0) {
        if (lse_out != nullptr) {
            *reinterpret_cast<float *>(reinterpret_cast<char *>(lse_out) +
                size_t(ih)*lse_nb0 + size_t(iq)*lse_nb1 + size_t(is)*lse_nb2) =
                    denom > 0.0f ? (m + logf(denom)) : -INFINITY;
        }
        if (g_kdisc_on && iq_packed == 0 && ih == 0 && ia == 0) {
            const int slot = atomicAdd(&g_kdisc_n, 1);
            if (slot >= 0 && slot < 8) {
                ggml_kdisc_rec r;
                r.marker     = 0xAA55AA51;
                r.iq_packed  = iq_packed; r.ih = ih; r.ia = ia;
                r.iq         = iq;        r.is = is; r.iq_global = iq_global;
                r.body_valid = bv ? 1 : 0; r.tail_valid = tv ? 1 : 0;
                r.body_max   = bm.x;      r.body_norm = bm.y;
                r.tail_max   = tm.x;      r.tail_norm = tm.y;
                r.m = m; r.wb = wb; r.wt = wt; r.denom = denom;
                r.body_o0    = bv ? *reinterpret_cast<const float *>(brow) : 0.0f;
                r.tail_o0    = tv ? *reinterpret_cast<const float *>(trow) : 0.0f;
                r.out_o0     = denom > 0.0f ? (bv ? (*reinterpret_cast<const float *>(brow))*wb : 0.0f)
                                                        + (tv ? (*reinterpret_cast<const float *>(trow))*wt : 0.0f)
                                          : 0.0f;
                r.out_lse    = denom > 0.0f ? (m + logf(denom)) : -INFINITY;
                g_kdisc_rec[slot] = r;
            }
        }
    }
}

static void ggml_cuda_tail_make_contiguous(ggml_tensor & t,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t element_size) {
    t.ne[0] = ne0; t.ne[1] = ne1; t.ne[2] = ne2; t.ne[3] = ne3;
    t.nb[0] = element_size;
    t.nb[1] = size_t(ne0)*t.nb[0];
    t.nb[2] = size_t(ne1)*t.nb[1];
    t.nb[3] = size_t(ne2)*t.nb[2];
    t.view_src = nullptr;
    t.view_offs = 0;
}

static void ggml_cuda_tail_make_contiguous_type(ggml_tensor & t,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    t.ne[0] = ne0; t.ne[1] = ne1; t.ne[2] = ne2; t.ne[3] = ne3;
    t.nb[0] = ggml_type_size(t.type);
    t.nb[1] = ggml_row_size(t.type, ne0);
    t.nb[2] = size_t(ne1)*t.nb[1];
    t.nb[3] = size_t(ne2)*t.nb[2];
    t.view_src = nullptr;
    t.view_offs = 0;
}

static __global__ void k_flash_attn_ext_tail_scatter(
        const float * packed, float * dst, const int32_t * query_order,
        int d, int n_query, int n_head, int n_stream, int q_max, int n_active,
        size_t packed_nb1, size_t packed_nb2, size_t packed_nb3,
        size_t dst_nb1, size_t dst_nb2, size_t dst_nb3) {
    const int iq_packed = blockIdx.x;
    const int ih = blockIdx.y;
    const int ia = blockIdx.z;
    if (iq_packed >= q_max || ih >= n_head || ia >= n_active) {
        return;
    }
    const int iq_global = query_order[ia*q_max + iq_packed];
    if (iq_global < 0 || iq_global >= n_query*n_stream) {
        return;
    }
    const int is = iq_global/n_query;
    const int iq = iq_global - is*n_query;
    const char * src_row = reinterpret_cast<const char *>(packed) +
        size_t(ih)*packed_nb1 + size_t(iq_packed)*packed_nb2 + size_t(ia)*packed_nb3;
    char * dst_row = reinterpret_cast<char *>(dst) +
        size_t(ih)*dst_nb1 + size_t(iq)*dst_nb2 + size_t(is)*dst_nb3;
    for (int id = threadIdx.x; id < d; id += blockDim.x) {
        *reinterpret_cast<float *>(dst_row + size_t(id)*sizeof(float)) =
            *reinterpret_cast<const float *>(src_row + size_t(id)*sizeof(float));
    }
}

static int64_t ggml_cuda_tail_compute_stride(int64_t logical_stride) {
    // The direct indexed kernel accepts compact tails as-is. Larger tails use
    // the upstream FA kernels, whose D=512 path requires a 256-row-aligned KV
    // extent. Keep that padding graph-local: it must never leak into the
    // persistent N+R history allocation or the serialized state.
    return logical_stride <= 256 ? logical_stride : GGML_PAD(logical_stride, FATTN_KQ_STRIDE);
}

static size_t ggml_cuda_tail_pass_alloc_size(ggml_backend_cuda_context & ctx, ggml_tensor & pass) {
    pass.data = reinterpret_cast<void *>(uintptr_t(0x10000000));
    return ggml_cuda_flash_attn_ext_get_alloc_size(ctx.device, &pass) + 256;
}

#if defined(GGML_CUDA_KVARN)
static void ggml_cuda_tail_clear_tensor_sources(ggml_tensor & tensor) {
    tensor.op = GGML_OP_NONE;
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        tensor.src[i] = nullptr;
    }
    tensor.view_src = nullptr;
    tensor.view_offs = 0;
}

static ggml_tensor ggml_cuda_tail_materialize_kvarn_view(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor *         tensor,
        void *                      data,
        bool                        emit_rotated) {
    const ggml_tensor * base = ggml_cuda_fattn_kvarn_view_base(tensor);
    GGML_ASSERT(base != nullptr);

    ggml_tensor materialize = *base;
    materialize.op = GGML_OP_KVARN_MATERIALIZE;
    materialize.data = data;
    materialize.view_src = nullptr;
    materialize.view_offs = 0;
    ggml_set_op_params_i32(&materialize, 4, emit_rotated ? 1 : 0);
    ggml_cuda_op_kvarn_materialize(ctx, &materialize);

    ggml_tensor result = *tensor;
    result.data = data;
    ggml_cuda_tail_clear_tensor_sources(result);
    return result;
}
#endif

static void ggml_cuda_flash_attn_ext_tail(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    // KDISC helpers, function scope so both the entry and pre-dispatch probes
    // can see them (they live in separate gated blocks).
    auto kdisc_hash = [](const ggml_tensor * t) -> unsigned long long {
        if (!t || !t->buffer || !t->data) { return 0; }
        std::vector<uint8_t> b(size_t(ggml_nbytes(t)));
        ggml_backend_tensor_get(t, b.data(), 0, b.size());
        unsigned long long h = 1469598103934665603ull;
        for (size_t i = 0; i < b.size(); ++i) { h ^= b[i]; h *= 1099511628211ull; }
        return h;
    };
    unsigned long long q_h0 = 0, kt_h0 = 0;
    // KDISC: record the real launch geometry of the exact-tail route, including
    // the partials->merge reduction, so "split/reduction count" is measured and
    // not inferred from graph shape.
    if (getenv("F4_RP_KDISC") != nullptr) {
        int one = 1, zero = 0;
        cudaError_t e_on = cudaMemcpyToSymbol(g_kdisc_on, &one, sizeof(int));
        cudaError_t e_z  = cudaMemcpyToSymbol(g_kdisc_n, &zero, sizeof(int));
        int back = -1;
        cudaError_t e_rd = cudaMemcpyFromSymbol(&back, g_kdisc_on, sizeof(int), 0, cudaMemcpyDeviceToHost);
        fprintf(stderr, "KDISC-FLAG set_on=%d set_n=%d read_back=%d read_err=%d\n",
                (int) e_on, (int) e_z, back, (int) e_rd);
        // Does the internal body_pass scratch / compute touch the live Q or the
        // rotated tail-K that indexed_small will read? (entry hash below,
        // compared with the pre-dispatch hash further down)
        q_h0 = kdisc_hash(dst->src[0]);
        kt_h0 = kdisc_hash(dst->src[5]);
        fprintf(stderr, "KDISC-IO entry q=%p+%zu qhash=%016llx kt=%p+%zu kthash=%016llx\n",
                dst->src[0]->data, (size_t) ggml_nbytes(dst->src[0]), q_h0,
                dst->src[5]->data, (size_t) ggml_nbytes(dst->src[5]), kt_h0);
        const ggml_tensor * qo = dst->src[8];
        const ggml_tensor * rd = dst->src[9];
        const ggml_tensor * kt = dst->src[5];
        const ggml_tensor * mt = dst->src[7];
        const int64_t n_head = dst->ne[1];
        const int64_t q_max = qo ? qo->ne[0] : -1;
        const int64_t n_active = qo ? qo->ne[1] : -1;
        const int64_t tail_stride = mt ? mt->ne[0] : -1;
        int32_t n_tail = -1;
        if (rd && rd->buffer && rd->ne[0] >= 5) {
            std::vector<int32_t> tmp(size_t(rd->ne[0]) * size_t(rd->ne[1]));
            ggml_backend_tensor_get(rd, tmp.data(), 0, ggml_nbytes(rd));
            n_tail = tmp[4];
        }
        fprintf(stderr,
            "KDISC-TAIL-LAUNCH fn=ggml_cuda_flash_attn_ext_tail grid=(q_max=%lld,n_head=%lld,"
            "n_active=%lld) block=256 kt_rows=%lld tail_stride=%lld n_tail_words=%d "
            "partial_merge=k_flash_attn_ext_tail_partials_merge\n",
            (long long) q_max, (long long) n_head, (long long) n_active,
            (long long) (kt ? kt->ne[1] : -1), (long long) tail_stride, n_tail);
    }
    const ggml_tensor * q  = dst->src[0];
    const ggml_tensor * kb = dst->src[1];
    const ggml_tensor * vb = dst->src[2];
    const ggml_tensor * mb = dst->src[3];
    const ggml_tensor * kt = dst->src[5];
    const ggml_tensor * vt = dst->src[6];
    const ggml_tensor * mt = dst->src[7];
    const ggml_tensor * qo = dst->src[8];
    const ggml_tensor * rd = dst->src[9];
    const ggml_tensor * kt_current = dst->src[10];
    const ggml_tensor * vt_current = dst->src[11];
    GGML_ASSERT(q && kb && vb && mb && kt && vt && mt && qo && rd);
    GGML_ASSERT((kt_current == nullptr) == (vt_current == nullptr));
    if (kt_current) {
        GGML_ASSERT(kt_current->type == kt->type && vt_current->type == vt->type);
        GGML_ASSERT(kt_current->ne[0] == kt->ne[0] && kt_current->ne[2] == kt->ne[2] && kt_current->ne[3] == 1);
        GGML_ASSERT(vt_current->ne[0] == vt->ne[0] && vt_current->ne[2] == vt->ne[2] && vt_current->ne[3] == 1);
        GGML_ASSERT(kt_current->ne[1] == vt_current->ne[1]);
    }
    GGML_ASSERT(qo->type == GGML_TYPE_I32 && rd->type == GGML_TYPE_I32 && rd->ne[0] >= 4);
    GGML_ASSERT(qo->ne[1] == rd->ne[1]);
    GGML_ASSERT(kt->ne[0] == q->ne[0] && vt->ne[0] == dst->ne[0]);
    if (kt->ne[2] != kb->ne[2] || vt->ne[2] != vb->ne[2]) {
        GGML_LOG_ERROR(
                "precision-tail head shard mismatch: K body=%lldx%lld/tail=%lld, V body=%lldx%lld/tail=%lld; "
                "K ops=%s/%s/%s, V ops=%s/%s/%s\n",
                (long long) kb->ne[1], (long long) kb->ne[2], (long long) kt->ne[2],
                (long long) vb->ne[1], (long long) vb->ne[2], (long long) vt->ne[2],
                ggml_op_name(kb->op),
                kb->src[0] ? ggml_op_name(kb->src[0]->op) : "<null>",
                kb->src[0] && kb->src[0]->src[0] ? ggml_op_name(kb->src[0]->src[0]->op) : "<null>",
                ggml_op_name(vb->op),
                vb->src[0] ? ggml_op_name(vb->src[0]->op) : "<null>",
                vb->src[0] && vb->src[0]->src[0] ? ggml_op_name(vb->src[0]->src[0]->op) : "<null>");
    }
    GGML_ASSERT(kt->ne[2] == dst->src[1]->ne[2] && vt->ne[2] == dst->src[2]->ne[2]);
    GGML_ASSERT(mt->ne[0] > 0 &&
            kt->ne[1] + (kt_current ? kt_current->ne[1] : 0) >= mt->ne[0]);

    const int d_k = int(q->ne[0]);
    const int d_v = int(dst->ne[0]);
    const int n_query = int(q->ne[1]);
    const int n_head = int(q->ne[2]);
    const int n_stream = int(q->ne[3]);
    const int tail_stride = int(mt->ne[0]);
    const int q_max = int(qo->ne[0]);
    const int n_active = int(qo->ne[1]);
    const int n_head_k = int(kt->ne[2]);
    const int n_head_v = int(vt->ne[2]);
    const int body_map_offset = 6 + tail_stride;
    const bool body_packed = rd->ne[0] > body_map_offset;
    const int desc_stride = int(rd->ne[0]);
    const int body_stride = body_packed ? desc_stride - body_map_offset : 0;
    const int configured_history_slots = ggml_get_op_params_i32(
            dst, GGML_FLASH_ATTN_EXT_OP_PARAM_TAIL_HISTORY_SLOTS);
    const int history_slots = configured_history_slots > 0 ?
            configured_history_slots : int(kt->ne[1]);
    GGML_ASSERT(history_slots > 0 && history_slots <= kt->ne[1] && vt->ne[1] == kt->ne[1]);
    const bool tail_bodyless = ggml_get_op_params_i32(
            dst, GGML_FLASH_ATTN_EXT_OP_PARAM_TAIL_BODYLESS) != 0;
    const bool indexed_small = tail_stride <= 256;
    const int compute_stride = tail_bodyless ?
            int(GGML_PAD(tail_stride, FATTN_KQ_STRIDE)) :
            int(ggml_cuda_tail_compute_stride(tail_stride));
    const size_t n_tail_rows = size_t(q_max)*n_head*n_active;
    const size_t n_body_rows = body_packed ? n_tail_rows : size_t(n_query)*n_head*n_stream;

    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<float2> body_meta_alloc(pool);
    ggml_cuda_pool_alloc<float2> tail_meta_alloc(pool);
    if (!tail_bodyless) {
        body_meta_alloc.alloc(n_body_rows);
        tail_meta_alloc.alloc(indexed_small ? 1 : n_tail_rows);
        CUDA_CHECK(cudaMemsetAsync(body_meta_alloc.get(), 0, n_body_rows*sizeof(float2), ctx.stream()));
    }

    const size_t kt_elements = size_t(d_k)*compute_stride*n_head_k*n_active;
    const size_t vt_elements = size_t(d_v)*compute_stride*n_head_v*n_active;
    const size_t q_elements = size_t(d_k)*q_max*n_head*n_active;
    const size_t mask_elements = size_t(compute_stride)*q_max*n_active;
    const size_t body_mask_elements = body_packed && !tail_bodyless ?
            size_t(body_stride)*q_max*n_active : 1;
    const size_t kb_row_bytes = ggml_row_size(kb->type, d_k);
    const size_t vb_row_bytes = ggml_row_size(vb->type, d_v);
    const size_t kb_packed_bytes = body_packed && !tail_bodyless ?
            kb_row_bytes*body_stride*kb->ne[2]*n_active : 1;
    const size_t vb_packed_bytes = body_packed && !tail_bodyless ?
            vb_row_bytes*body_stride*vb->ne[2]*n_active : 1;
    ggml_cuda_pool_alloc<uint8_t> kt_alloc(pool,
            indexed_small ? 1 : kt_elements*ggml_type_size(kt->type));
    ggml_cuda_pool_alloc<uint8_t> vt_alloc(pool,
            indexed_small ? 1 : vt_elements*ggml_type_size(vt->type));
    ggml_cuda_pool_alloc<float> q_alloc(pool,
            indexed_small && !body_packed ? 1 : q_elements);
    ggml_cuda_pool_alloc<half> mask_alloc(pool,
            indexed_small && !body_packed ? 1 : mask_elements);
    ggml_cuda_pool_alloc<uint8_t> kb_alloc(pool);
    ggml_cuda_pool_alloc<uint8_t> vb_alloc(pool);
    ggml_cuda_pool_alloc<half> body_mask_alloc(pool);
    if (body_packed && !tail_bodyless) {
        kb_alloc.alloc(kb_packed_bytes);
        vb_alloc.alloc(vb_packed_bytes);
        body_mask_alloc.alloc(body_mask_elements);
    }

    const int threads = 256;
    auto blocks_for = [threads](size_t n) { return int(std::min<size_t>((n + threads - 1)/threads, 65535)); };
    const bool kt_vec = !indexed_small && kt->nb[0] == ggml_type_size(kt->type) &&
            size_t(d_k)*kt->nb[0] % sizeof(uint4) == 0 &&
            kt->nb[1] % alignof(uint4) == 0 && kt->nb[2] % alignof(uint4) == 0 &&
            uintptr_t(kt->data) % alignof(uint4) == 0 && uintptr_t(kt_alloc.get()) % alignof(uint4) == 0 &&
            (!kt_current || (kt_current->nb[0] == kt->nb[0] &&
                kt_current->nb[1] % alignof(uint4) == 0 && kt_current->nb[2] % alignof(uint4) == 0 &&
                uintptr_t(kt_current->data) % alignof(uint4) == 0));
    const bool vt_vec = !indexed_small && vt->nb[0] == ggml_type_size(vt->type) &&
            size_t(d_v)*vt->nb[0] % sizeof(uint4) == 0 &&
            vt->nb[1] % alignof(uint4) == 0 && vt->nb[2] % alignof(uint4) == 0 &&
            uintptr_t(vt->data) % alignof(uint4) == 0 && uintptr_t(vt_alloc.get()) % alignof(uint4) == 0 &&
            (!vt_current || (vt_current->nb[0] == vt->nb[0] &&
                vt_current->nb[1] % alignof(uint4) == 0 && vt_current->nb[2] % alignof(uint4) == 0 &&
                uintptr_t(vt_current->data) % alignof(uint4) == 0));
    if (kt_vec) {
        const size_t n = kt_elements*ggml_type_size(kt->type)/sizeof(uint4);
        k_flash_attn_ext_tail_pack_arenas_vec<uint4><<<blocks_for(n), threads, 0, ctx.stream()>>>(
            (const char *) kt->data, kt_current ? (const char *) kt_current->data : nullptr,
            (uint4 *) kt_alloc.get(), (const int32_t *) rd->data,
            int(size_t(d_k)*kt->nb[0]/sizeof(uint4)), compute_stride, n_head_k, n_active,
            desc_stride, history_slots, kt->nb[1], kt->nb[2],
            kt_current ? kt_current->nb[1] : 0, kt_current ? kt_current->nb[2] : 0);
    } else if (!indexed_small && ggml_type_size(kt->type) == sizeof(uint16_t)) {
        k_flash_attn_ext_tail_pack_arenas<uint16_t><<<blocks_for(kt_elements), threads, 0, ctx.stream()>>>(
            (const char *) kt->data, kt_current ? (const char *) kt_current->data : nullptr,
            (uint16_t *) kt_alloc.get(), (const int32_t *) rd->data,
            d_k, compute_stride, n_head_k, n_active, desc_stride, history_slots,
            kt->nb[0], kt->nb[1], kt->nb[2],
            kt_current ? kt_current->nb[0] : 0,
            kt_current ? kt_current->nb[1] : 0,
            kt_current ? kt_current->nb[2] : 0);
    } else if (!indexed_small) {
        k_flash_attn_ext_tail_pack_arenas<float><<<blocks_for(kt_elements), threads, 0, ctx.stream()>>>(
            (const char *) kt->data, kt_current ? (const char *) kt_current->data : nullptr,
            (float *) kt_alloc.get(), (const int32_t *) rd->data,
            d_k, compute_stride, n_head_k, n_active, desc_stride, history_slots,
            kt->nb[0], kt->nb[1], kt->nb[2],
            kt_current ? kt_current->nb[0] : 0,
            kt_current ? kt_current->nb[1] : 0,
            kt_current ? kt_current->nb[2] : 0);
    }
    if (vt_vec) {
        const size_t n = vt_elements*ggml_type_size(vt->type)/sizeof(uint4);
        k_flash_attn_ext_tail_pack_arenas_vec<uint4><<<blocks_for(n), threads, 0, ctx.stream()>>>(
            (const char *) vt->data, vt_current ? (const char *) vt_current->data : nullptr,
            (uint4 *) vt_alloc.get(), (const int32_t *) rd->data,
            int(size_t(d_v)*vt->nb[0]/sizeof(uint4)), compute_stride, n_head_v, n_active,
            desc_stride, history_slots, vt->nb[1], vt->nb[2],
            vt_current ? vt_current->nb[1] : 0, vt_current ? vt_current->nb[2] : 0);
    } else if (!indexed_small && ggml_type_size(vt->type) == sizeof(uint16_t)) {
        k_flash_attn_ext_tail_pack_arenas<uint16_t><<<blocks_for(vt_elements), threads, 0, ctx.stream()>>>(
            (const char *) vt->data, vt_current ? (const char *) vt_current->data : nullptr,
            (uint16_t *) vt_alloc.get(), (const int32_t *) rd->data,
            d_v, compute_stride, n_head_v, n_active, desc_stride, history_slots,
            vt->nb[0], vt->nb[1], vt->nb[2],
            vt_current ? vt_current->nb[0] : 0,
            vt_current ? vt_current->nb[1] : 0,
            vt_current ? vt_current->nb[2] : 0);
    } else if (!indexed_small) {
        k_flash_attn_ext_tail_pack_arenas<float><<<blocks_for(vt_elements), threads, 0, ctx.stream()>>>(
            (const char *) vt->data, vt_current ? (const char *) vt_current->data : nullptr,
            (float *) vt_alloc.get(), (const int32_t *) rd->data,
            d_v, compute_stride, n_head_v, n_active, desc_stride, history_slots,
            vt->nb[0], vt->nb[1], vt->nb[2],
            vt_current ? vt_current->nb[0] : 0,
            vt_current ? vt_current->nb[1] : 0,
            vt_current ? vt_current->nb[2] : 0);
    }
    if (!indexed_small || body_packed) {
        k_flash_attn_ext_tail_pack_q_mask<<<blocks_for(std::max(q_elements, mask_elements)), threads, 0, ctx.stream()>>>(
            (const float *) q->data, (const half *) mt->data, (const int32_t *) qo->data,
            q_alloc.get(), mask_alloc.get(), d_k, n_query, n_head, n_stream,
            tail_stride, compute_stride, q_max, n_active,
            q->nb[1], q->nb[2], q->nb[3], mt->nb[1], mt->nb[3]);
    }
    if (body_packed && !tail_bodyless) {
        k_flash_attn_ext_tail_pack_body_rows<<<blocks_for(kb_packed_bytes), threads, 0, ctx.stream()>>>(
            (const char *) kb->data, kb_alloc.get(), (const int32_t *) rd->data,
            int(kb_row_bytes), int(kb->ne[1]), body_stride, int(kb->ne[2]), n_active,
            desc_stride, body_map_offset, kb->nb[1], kb->nb[2], kb->nb[3]);
        k_flash_attn_ext_tail_pack_body_rows<<<blocks_for(vb_packed_bytes), threads, 0, ctx.stream()>>>(
            (const char *) vb->data, vb_alloc.get(), (const int32_t *) rd->data,
            int(vb_row_bytes), int(vb->ne[1]), body_stride, int(vb->ne[2]), n_active,
            desc_stride, body_map_offset, vb->nb[1], vb->nb[2], vb->nb[3]);
        k_flash_attn_ext_tail_pack_body_mask<<<blocks_for(body_mask_elements), threads, 0, ctx.stream()>>>(
            (const half *) mb->data, (const int32_t *) qo->data, (const int32_t *) rd->data,
            body_mask_alloc.get(), int(kb->ne[1]), n_query, n_stream,
            body_stride, q_max, n_active, desc_stride, body_map_offset, mb->nb[1], mb->nb[3]);
    }
    CUDA_CHECK(cudaGetLastError());

    ggml_tensor q_packed = *q;
    q_packed.data = q_alloc.get();
    ggml_cuda_tail_make_contiguous(q_packed, d_k, q_max, n_head, n_active, sizeof(float));
    ggml_tensor k_packed = *kt;
    k_packed.data = kt_alloc.get();
    ggml_cuda_tail_make_contiguous(k_packed, d_k, compute_stride, n_head_k, n_active, ggml_type_size(kt->type));
    ggml_tensor v_packed = *vt;
    v_packed.data = vt_alloc.get();
    ggml_cuda_tail_make_contiguous(v_packed, d_v, compute_stride, n_head_v, n_active, ggml_type_size(vt->type));
    ggml_tensor mask_packed = *mt;
    mask_packed.data = mask_alloc.get();
    ggml_cuda_tail_make_contiguous(mask_packed, compute_stride, q_max, 1, n_active, sizeof(half));

    ggml_tensor body_meta = *qo;
    body_meta.type = GGML_TYPE_F32;
    body_meta.data = body_meta_alloc.get();
    ggml_cuda_tail_make_contiguous(body_meta, 2, n_head,
            body_packed ? q_max : n_query, body_packed ? n_active : n_stream, sizeof(float));
    ggml_tensor body_pass = *dst;
    ggml_tensor kb_packed = *kb;
    ggml_tensor vb_packed = *vb;
    ggml_tensor body_mask_packed = *mb;
    if (body_packed && !tail_bodyless) {
        kb_packed.data = kb_alloc.get();
        vb_packed.data = vb_alloc.get();
        body_mask_packed.data = body_mask_alloc.get();
        ggml_cuda_tail_make_contiguous_type(
                kb_packed, d_k, body_stride, kb->ne[2], n_active);
        ggml_cuda_tail_make_contiguous_type(
                vb_packed, d_v, body_stride, vb->ne[2], n_active);
        ggml_cuda_tail_make_contiguous(
                body_mask_packed, body_stride, q_max, 1, n_active, sizeof(half));
        body_pass.src[0] = &q_packed;
        body_pass.src[1] = &kb_packed;
        body_pass.src[2] = &vb_packed;
        body_pass.src[3] = &body_mask_packed;
        ggml_cuda_tail_make_contiguous(
                body_pass, d_v, n_head, q_max, n_active, sizeof(float));
    }
    for (int i = 5; i < GGML_MAX_SRC; ++i) {
        body_pass.src[i] = nullptr;
    }
    // The sub-pass copies the parent tensor struct, which carries op_params
    // verbatim: the parent's LSE attachment must not be inherited, or the body
    // pass would write the parent's LSE buffer. The body gets a transient,
    // discarded LSE of its own instead of none (review 6, R5).
    //
    // Why this is not just a dispatch trick: the tail merge consumes the
    // body's (max, denom) metadata from src[8] AND its O output, so the body
    // must take an exact meta-publishing route. Without an LSE request the
    // dispatcher serves prompt-generic-mma (see the LSE-requested early
    // decline in ggml_cuda_flash_attn_ext_mma_kvarn_launch_case), whose O is
    // ~1e-3 off on some data (the non-tail d256-q256 shape shows the same
    // 1.25e-3 as attach divergence), and the merge would inherit it. With LSE
    // requested the same predicates reject generic-mma and serve
    // portable-native (exact), or decode-split where eligible (q1, unaffected
    // by the flag in route selection). Both plain and LSE parent runs attach
    // it, so attachment isolation compares identical routes.
    //
    // R5 evidence: support/execute agreement is proven per run by route
    // telemetry (same_route + printed route family); the best-kernel support
    // probe (ggml_cuda_get_best_fattn_kernel) reads only src[0..3]/ne/nb/type,
    // so it is unaffected by src[8]/LSE by inspection. Lifetime matches every
    // other tail transient (stream-ordered pool, scoped to this function).
    // Windowed original-V prompt bodies already publish precise metadata.
    // D256/Q1 original-V bodies need the precise route too: long-context
    // coverage disproved the former assumption that every small-Q MMA body
    // meets the O gate. Sinks keep their existing generic route contract.
    ggml_cuda_pool_alloc<float> body_lse_alloc(pool);
    ggml_tensor body_lse = *qo;
    const bool parent_body_sinks = body_pass.src[4] != nullptr;
    const auto body_domain = ggml_cuda_fattn_kvarn_domain(&body_pass);
    const bool body_windowed_applies =
        body_domain == GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_K_ORIGINAL_V &&
        n_query > 1 && !parent_body_sinks;
    // Long D256/Q1 original-V bodies exceed the O error gate on generic MMA.
    // Request precise body metadata/output; portable split-K bounds decode cost.
    const bool body_precise_d256_decode =
        body_pass.src[0]->ne[0] == 256 && body_pass.src[0]->ne[1] == 1 &&
        body_domain == GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_K_ORIGINAL_V;
    if (!tail_bodyless && !parent_body_sinks && !body_windowed_applies &&
            (n_query > 16 || body_precise_d256_decode)) {
        body_lse_alloc.alloc(n_body_rows);
        body_lse.data = body_lse_alloc.get();
        ggml_cuda_tail_make_contiguous(body_lse,
                n_head, body_packed ? q_max : n_query, body_packed ? n_active : n_stream, 1,
                sizeof(float));
        ggml_flash_attn_ext_set_lse_out(&body_pass, &body_lse);
    } else {
        ggml_flash_attn_ext_set_lse_out(&body_pass, nullptr);
    }
    // Reset: when the parent carries sinks the body must stay on the real
    // generic KVarN route (coverage), so no transient LSE attaches at all.
    (void) parent_body_sinks;
    body_pass.src[8] = &body_meta;
    body_pass.view_src = nullptr;
    body_pass.view_offs = 0;
    ggml_cuda_pool_alloc<uint8_t> body_alloc(pool);
    if (!tail_bodyless) {
        const size_t body_alloc_size = ggml_cuda_tail_pass_alloc_size(ctx, body_pass);
        body_alloc.alloc(body_alloc_size);
        body_pass.data = body_alloc.get();
    }
    uint64_t tail_pack_bytes = kt_alloc.actual_size + vt_alloc.actual_size +
            q_alloc.actual_size + mask_alloc.actual_size + kb_alloc.actual_size +
            vb_alloc.actual_size + body_mask_alloc.actual_size;
    const uint64_t tail_plan_input_bytes = ggml_nbytes(mt) + ggml_nbytes(qo) + ggml_nbytes(rd);
    // mt/qo/rd are scheduler-owned graph tensors and are already included in
    // the reported CUDA compute buffer. Keep their footprint visible as a
    // diagnostic, but do not double-count it as CUDA-pool high water.
    uint64_t body_output_bytes = body_alloc.actual_size;
    uint64_t tail_base_bytes = body_meta_alloc.actual_size + tail_meta_alloc.actual_size +
            tail_pack_bytes + body_output_bytes;
    ggml_cuda_pool_alloc<half> k_materialized_alloc(pool);
    ggml_cuda_pool_alloc<half> v_materialized_alloc(pool);
    ggml_cuda_pool_alloc<uint8_t> fallback_body_alloc(pool);
    if (!tail_bodyless) {
        if (ggml_cuda_flash_attn_ext_kvarn_uses_views(&body_pass)) {
            if (!ggml_cuda_flash_attn_ext_kvarn(
                    ctx, &body_pass, GGML_CUDA_FATTN_KVARN_ENTRY_COMPACT_TAIL)) {
#if defined(GGML_CUDA_KVARN)
                const bool mixed_domain = ggml_get_op_params_i32(
                        &body_pass, GGML_FLASH_ATTN_EXT_OP_PARAM_KVARN_DOMAIN) ==
                    GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_K_ORIGINAL_V;
                k_materialized_alloc.alloc(ggml_nelements(kb));
                v_materialized_alloc.alloc(ggml_nelements(vb));
                ggml_tensor k_materialized = ggml_cuda_tail_materialize_kvarn_view(
                        ctx, kb, k_materialized_alloc.get(), true);
                ggml_tensor v_materialized = ggml_cuda_tail_materialize_kvarn_view(
                        ctx, vb, v_materialized_alloc.get(), !mixed_domain);

                if (body_packed) {
                    k_flash_attn_ext_tail_pack_body_rows<<<blocks_for(kb_packed_bytes), threads, 0, ctx.stream()>>>(
                        (const char *) k_materialized.data, kb_alloc.get(), (const int32_t *) rd->data,
                        int(kb_row_bytes), int(kb->ne[1]), body_stride, int(kb->ne[2]), n_active,
                        desc_stride, body_map_offset, k_materialized.nb[1], k_materialized.nb[2], k_materialized.nb[3]);
                    k_flash_attn_ext_tail_pack_body_rows<<<blocks_for(vb_packed_bytes), threads, 0, ctx.stream()>>>(
                        (const char *) v_materialized.data, vb_alloc.get(), (const int32_t *) rd->data,
                        int(vb_row_bytes), int(vb->ne[1]), body_stride, int(vb->ne[2]), n_active,
                        desc_stride, body_map_offset, v_materialized.nb[1], v_materialized.nb[2], v_materialized.nb[3]);
                    CUDA_CHECK(cudaGetLastError());
                    ggml_cuda_tail_clear_tensor_sources(kb_packed);
                    ggml_cuda_tail_clear_tensor_sources(vb_packed);
                } else {
                    body_pass.src[1] = &k_materialized;
                    body_pass.src[2] = &v_materialized;
                }

                const size_t fallback_body_alloc_size = ggml_cuda_tail_pass_alloc_size(ctx, body_pass);
                fallback_body_alloc.alloc(fallback_body_alloc_size);
                body_pass.data = fallback_body_alloc.get();
                ggml_cuda_flash_attn_ext_dispatch(ctx, &body_pass);
                tail_pack_bytes += k_materialized_alloc.actual_size + v_materialized_alloc.actual_size;
                body_output_bytes += fallback_body_alloc.actual_size;
                tail_base_bytes = body_meta_alloc.actual_size + tail_meta_alloc.actual_size +
                    tail_pack_bytes + body_output_bytes;
#else
                GGML_ABORT("KVarN exact-tail body reached a CUDA build without KVarN kernels");
#endif
            }
        } else {
            ggml_cuda_flash_attn_ext_dispatch(ctx, &body_pass);
        }
    }

    if (indexed_small) {
        float scale = 1.0f;
        float max_bias = 0.0f;
        float logit_softcap = 0.0f;
        // op_params is int32_t[GGML_MAX_OP_PARAMS/4]: index N is byte 4*N, NOT
        // byte N*sizeof(float). The old "+ N*sizeof(float)" advanced 4x too far:
        //   max_bias      read byte 16 (= KVARN_DOMAIN, i32[4])
        //   logit_softcap read byte 32 (= LSE_PTR, i32[8], a HOST POINTER)
        // so any FA node with an attached LSE (ggml_flash_attn_ext_lse_out) fed a
        // pointer-derived softcap into score = softcap*tanh(score/softcap),
        // corrupting the tail scores run-to-run. Writer side is
        // memcpy(op_params, {scale,max_bias,logit_softcap}, 12) -> bytes 0,4,8.
        ggml_flash_attn_ext_get_alibi_params(dst, &scale, &max_bias, &logit_softcap);
        float * const lse_out = ggml_cuda_fattn_lse_ptr(dst);
        const ggml_tensor * lse_t = ggml_flash_attn_ext_get_lse_out(dst);
        const size_t lse_nb0 = lse_t ? lse_t->nb[0] : 0;
        const size_t lse_nb1 = lse_t ? lse_t->nb[1] : 0;
        const size_t lse_nb2 = lse_t ? lse_t->nb[2] : 0;
        const dim3 grid(q_max, n_head, n_active);
    // KDISC: AFTER the body scratch allocation and the body pass, immediately
    // before indexed_small consumes Q and the rotated tail K.
    if (getenv("F4_RP_KDISC") != nullptr) {
        const unsigned long long q_h1 = kdisc_hash(dst->src[0]);
        const unsigned long long kt_h1 = kdisc_hash(dst->src[5]);
        const char * bp = (const char *) body_pass.data;
        const char * qp = (const char *) dst->src[0]->data;
        const char * kp = (const char *) dst->src[5]->data;
        const size_t qs = (size_t) ggml_nbytes(dst->src[0]);
        const size_t ks = (size_t) ggml_nbytes(dst->src[5]);
        const size_t bs = (size_t) body_alloc.actual_size;
        fprintf(stderr, "KDISC-IO post-body qhash=%016llx kthash=%016llx %s | "
                "body_pass=%p+%zu q=%p+%zu kt=%p+%zu overlap[bodyQ=%d bodyK=%d]\n",
                q_h1, kt_h1,
                (q_h0 == q_h1 && kt_h0 == kt_h1) ? "UNCHANGED" : "*** MODIFIED BY BODY WORK ***",
                (void *) bp, bs, (void *) qp, qs, (void *) kp, ks,
                (int) (bp && qp && bp < qp + qs && qp < bp + bs),
                (int) (bp && kp && bp < kp + ks && kp < bp + bs));
    }

#define GGML_CUDA_LAUNCH_INDEXED(TK, TV, MAX_TAIL) \
        k_flash_attn_ext_tail_indexed_small<TK, TV, MAX_TAIL><<<grid, 256, 0, ctx.stream()>>>( \
            (const float *) q->data, (const char *) kt->data, (const char *) vt->data, \
            kt_current ? (const char *) kt_current->data : nullptr, \
            vt_current ? (const char *) vt_current->data : nullptr, (const half *) mt->data, \
            (const int32_t *) qo->data, (const int32_t *) rd->data, \
            (const float *) body_pass.data, body_meta_alloc.get(), (float *) dst->data, lse_out, \
            d_k, d_v, n_query, n_head, n_stream, q_max, n_active, n_head_k, n_head_v, desc_stride, \
            scale, max_bias, logit_softcap, q->nb[1], q->nb[2], q->nb[3], \
            history_slots, kt->nb[0], kt->nb[1], kt->nb[2], vt->nb[0], vt->nb[1], vt->nb[2], \
            kt_current ? kt_current->nb[0] : 0, kt_current ? kt_current->nb[1] : 0, kt_current ? kt_current->nb[2] : 0, \
            vt_current ? vt_current->nb[0] : 0, vt_current ? vt_current->nb[1] : 0, vt_current ? vt_current->nb[2] : 0, \
            mt->nb[1], mt->nb[3], \
            tail_bodyless, body_packed, body_pass.nb[1], body_pass.nb[2], body_pass.nb[3], \
            dst->nb[1], dst->nb[2], dst->nb[3], \
            lse_out ? lse_nb0 : 0, lse_out ? lse_nb1 : 0, lse_out ? lse_nb2 : 0)
        if (kt->type == GGML_TYPE_F16 && vt->type == GGML_TYPE_F16) {
            GGML_CUDA_LAUNCH_INDEXED(half, half, 256);
        } else if (kt->type == GGML_TYPE_F16 && vt->type == GGML_TYPE_BF16) {
            GGML_CUDA_LAUNCH_INDEXED(half, nv_bfloat16, 256);
        } else if (kt->type == GGML_TYPE_BF16 && vt->type == GGML_TYPE_F16) {
            GGML_CUDA_LAUNCH_INDEXED(nv_bfloat16, half, 256);
        } else {
            GGML_ASSERT(kt->type == GGML_TYPE_BF16 && vt->type == GGML_TYPE_BF16);
            GGML_CUDA_LAUNCH_INDEXED(nv_bfloat16, nv_bfloat16, 256);
        }
#undef GGML_CUDA_LAUNCH_INDEXED
        CUDA_CHECK(cudaGetLastError());
        ggml_cuda_kv_memory_transient_stats_record_tail(
                body_meta_alloc.actual_size,
                tail_meta_alloc.actual_size,
                tail_pack_bytes,
                body_output_bytes,
                0,
                tail_plan_input_bytes,
                tail_base_bytes);
        return;
    }

    ggml_tensor tail_meta = body_meta;
    if (!tail_bodyless) {
        tail_meta.data = tail_meta_alloc.get();
        ggml_cuda_tail_make_contiguous(tail_meta, 2, n_head, q_max, n_active, sizeof(float));
    }
    ggml_tensor tail_pass = *dst;
    tail_pass.src[0] = &q_packed;
    tail_pass.src[1] = &k_packed;
    tail_pass.src[2] = &v_packed;
    tail_pass.src[3] = &mask_packed;
    // Sinks are part of the body partial for a split source. For a bodyless
    // source the exact pass is the sole softmax and must consume them itself.
    // Position-split LSE (plan §3.4): src[4] is sinks, the LSE destination is
    // the attachment on the parent node. A bodyless tail is the only softmax for
    // its keys, so it writes a packed LSE that is scattered into the parent's
    // LSE tensor afterwards; a body+tail merge combines both partials in
    // k_flash_attn_ext_tail_partials_merge.
    ggml_tensor * const lse_out = ggml_flash_attn_ext_get_lse_out(dst);
    const bool tail_lse = lse_out != nullptr;
    ggml_cuda_pool_alloc<float> tail_lse_packed(pool);
    ggml_tensor lse_packed = {};
    if (tail_lse && tail_bodyless) {
        tail_lse_packed.alloc(size_t(n_head) * q_max * n_active);
        lse_packed = *lse_out;
        lse_packed.data = tail_lse_packed.get();
        ggml_cuda_tail_make_contiguous(lse_packed, n_head, q_max, n_active, 1, sizeof(float));
    }
    // src[4] stays sinks-only; the sub-pass learns its own LSE destination from
    // the attachment below.
    tail_pass.src[4] = tail_bodyless ? dst->src[4] : nullptr;
    ggml_flash_attn_ext_set_lse_out(&tail_pass,
        (tail_lse && tail_bodyless) ? &lse_packed : nullptr);
    for (int i = 5; i < GGML_MAX_SRC; ++i) {
        tail_pass.src[i] = nullptr;
    }
    tail_pass.src[8] = tail_bodyless ? nullptr : &tail_meta;
    ggml_cuda_tail_make_contiguous(tail_pass, d_v, n_head, q_max, n_active, sizeof(float));
    if (q_max == 1 && n_active == 1 &&
            kt->type == GGML_TYPE_BF16 && vt->type == GGML_TYPE_BF16) {
        ggml_set_op_params_i32(&tail_pass, GGML_CUDA_FATTN_OP_PARAM_FORCE_VEC, 1);
    }
    const size_t tail_alloc_size = ggml_cuda_tail_pass_alloc_size(ctx, tail_pass);
    ggml_cuda_pool_alloc<uint8_t> tail_alloc(pool, tail_alloc_size);
    tail_pass.data = tail_alloc.get();
    ggml_cuda_kv_memory_transient_stats_record_tail(
            body_meta_alloc.actual_size,
            tail_meta_alloc.actual_size,
            tail_pack_bytes,
            body_output_bytes,
            tail_alloc.actual_size,
            tail_plan_input_bytes,
            tail_base_bytes + tail_alloc.actual_size);
    ggml_cuda_flash_attn_ext_dispatch(ctx, &tail_pass);

    const ggml_tensor * lse_t = ggml_flash_attn_ext_get_lse_out(dst);
    const size_t lse_nb0 = lse_t ? lse_t->nb[0] : 0;
    const size_t lse_nb1 = lse_t ? lse_t->nb[1] : 0;
    const size_t lse_nb2 = lse_t ? lse_t->nb[2] : 0;
    const dim3 grid(q_max, n_head, n_active);
    if (getenv("F4_RP_KDISC") != nullptr) {
        fprintf(stderr, "KDISC-TAIL-PATH tail_bodyless=%d -> %s\n", (int) tail_bodyless,
                tail_bodyless ? "scatter (merge SKIPPED)" : "partials_merge");
    }
    if (tail_bodyless) {
        k_flash_attn_ext_tail_scatter<<<grid, 256, 0, ctx.stream()>>>(
            (const float *) tail_pass.data, (float *) dst->data, (const int32_t *) qo->data,
            d_v, n_query, n_head, n_stream, q_max, n_active,
            tail_pass.nb[1], tail_pass.nb[2], tail_pass.nb[3],
            dst->nb[1], dst->nb[2], dst->nb[3]);
        CUDA_CHECK(cudaGetLastError());
        if (tail_lse) {
            k_flash_attn_ext_tail_scatter<<<grid, 256, 0, ctx.stream()>>>(
                (const float *) tail_lse_packed.get(), (float *) lse_out->data,
                (const int32_t *) qo->data,
                1, n_query, n_head, n_stream, q_max, n_active,
                size_t(n_head) * sizeof(float), size_t(n_head) * q_max * sizeof(float),
                size_t(n_head) * q_max * sizeof(float),
                sizeof(float), size_t(lse_out->ne[0]) * sizeof(float),
                size_t(lse_out->ne[0]) * lse_out->ne[1] * sizeof(float));
            CUDA_CHECK(cudaGetLastError());
        }
        return;
    }
    if (getenv("F4_RP_KDISC") != nullptr) {
        fprintf(stderr, "KDISC-TAIL-PATH launching partials_merge grid=(%d,%d,%d)\n",
                (int) grid.x, (int) grid.y, (int) grid.z);
    }
    k_flash_attn_ext_tail_partials_merge<<<grid, 256, 0, ctx.stream()>>>(
        (const float *) body_pass.data, (const float *) tail_pass.data, (float *) dst->data,
        ggml_cuda_fattn_lse_ptr(dst),
        body_meta_alloc.get(), tail_meta_alloc.get(), (const int32_t *) qo->data,
        d_v, n_query, n_head, n_stream, q_max, n_active, body_packed,
        body_pass.nb[1], body_pass.nb[2], body_pass.nb[3],
        tail_pass.nb[1], tail_pass.nb[2], tail_pass.nb[3],
        dst->nb[1], dst->nb[2], dst->nb[3],
        lse_nb0, lse_nb1, lse_nb2);
    CUDA_CHECK(cudaGetLastError());
}
