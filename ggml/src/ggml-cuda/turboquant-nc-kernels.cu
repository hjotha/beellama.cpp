// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "turboquant-nc-kernels.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "turboquant-nc.cuh"
#include "turboquant-nc-launch.cuh"
#include "turboquant-nc-view.h"
#include "saw-int4.cuh"

static thread_local bool tqnc_failure = false;

void ggml_cuda_tqnc_clear_failure() { tqnc_failure = false; }
bool ggml_cuda_tqnc_has_failure() { return tqnc_failure; }

static bool tqnc_status_begin(cudaStream_t stream, int ** status) {
    *status = nullptr;
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    cudaError_t err = cudaStreamIsCapturing(stream, &capture);
    if (err != cudaSuccess || capture != cudaStreamCaptureStatusNone) {
        (void) cudaGetLastError();
        return false;
    }
    err = cudaMalloc((void **) status, sizeof(int));
    if (err == cudaSuccess) err = cudaMemsetAsync(*status, 0, sizeof(int), stream);
    if (err != cudaSuccess) {
        if (*status) (void) cudaFree(*status);
        *status = nullptr;
        (void) cudaGetLastError();
        return false;
    }
    return true;
}

static bool tqnc_status_finish(cudaStream_t stream, int * status, int * validation_flags = nullptr) {
    if (validation_flags) *validation_flags = -1;
    int failed = 0;
    cudaError_t err = cudaMemcpyAsync(&failed, status, sizeof(failed), cudaMemcpyDeviceToHost, stream);
    const cudaError_t sync_err = cudaStreamSynchronize(stream);
    const cudaError_t free_err = cudaFree(status);
    if (err != cudaSuccess || sync_err != cudaSuccess || free_err != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    if (validation_flags) *validation_flags = failed;
    return failed == 0;
}

// Match convert.cu's inverse BDR arithmetic and reject only values whose
// actual F16 representation is non-finite. This also handles zeroed padding.
static __device__ __forceinline__ bool saw_valid_f16_group(const block_saw_int4_ggml * block, int group) {
    if ((__float_as_uint(block->scale) & 0x7f800000u) == 0x7f800000u || block->scale < 0 ||
        (__float_as_uint(block->zero) & 0x7f800000u) == 0x7f800000u) return false;
    float values[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int element = group*16 + j;
        const uint8_t packed = block->qs[element % 128];
        const int quant = element < 128 ? packed & 15 : packed >> 4;
        values[j] = __fmul_rn(__fsub_rn(float(quant), block->zero), block->scale);
    }
    saw_fwht(values);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const float reconstructed = __fmul_rn(values[j], 0.25f);
        if ((__half_as_ushort(__float2half_rn(reconstructed)) & 0x7c00u) == 0x7c00u) return false;
    }
    return true;
}

static __global__ void tqnc_tail_validate_kernel(
        const char * q, const char * k, const char * v, const char * body_mask, const char * tail_mask,
        const char * kt, const char * vt, const char * kc, const char * vc,
        const int32_t * order, const int32_t * desc, int * status,
        int nq, int n_body, int tail_stride, int body_stride, int history, int current, bool saw,
        int64_t q1, int64_t q2, int64_t bm1, int64_t tm1,
        int64_t k1, int64_t k2, int64_t v1, int64_t v2,
        int64_t kt1, int64_t kt2, int64_t vt1, int64_t vt2,
        int64_t kc1, int64_t kc2, int64_t vc1, int64_t vc2, int64_t work) {
    const int64_t tid = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const int64_t step = int64_t(blockDim.x)*gridDim.x;
    for (int64_t i = tid; i < work; i += step) {
        if (i < nq) {
            const int x = order[i];
            if (x < 0 || x >= nq) atomicOr(status, 1);
            else for (int64_t j = 0; j < i; ++j) if (order[j] == x) atomicOr(status, 1);
        }
        if (i == 0 && (desc[4] < 0 || desc[4] > tail_stride ||
                (body_stride && (desc[5] < 0 || desc[5] > body_stride)))) atomicOr(status, 2);
        if (i < tail_stride && i < desc[4]) {
            const int slot = desc[6 + i];
            if (slot < 0 || (slot >= history ? (!kc || slot - history >= current) : slot >= history)) {
                atomicOr(status, 2);
            } else {
                for (int64_t j = 0; j < i; ++j) if (desc[6 + j] == slot) atomicOr(status, 2);
            }
        }
        if (i < body_stride && i < desc[5]) {
            const int row = desc[6 + tail_stride + i];
            if (row < 0 || row >= n_body) atomicOr(status, 4);
            else for (int64_t j = 0; j < i; ++j)
                if (desc[6 + tail_stride + j] == row) atomicOr(status, 4);
        }
        if (i < int64_t(nq)*24*256) {
            const int d = i % 256, qi = (i/256) % nq, h = i/(256*nq);
            const float x = *reinterpret_cast<const float *>(q + qi*q1 + h*q2 + d*sizeof(float));
            if ((__float_as_uint(x) & 0x7f800000u) == 0x7f800000u) atomicOr(status, 8);
        }
        if (i < int64_t(n_body)*nq) {
            const int row = i % n_body, qi = i/n_body;
            const auto bits = *reinterpret_cast<const uint16_t *>(body_mask + qi*bm1 + row*sizeof(half));
            if ((bits & 0x7c00u) == 0x7c00u && bits != 0xfc00u) atomicOr(status, 16);
        }
        if (saw && i < int64_t(n_body)*4*16) {
            const int group = i % 16, row = (i/16) % n_body, h = i/(int64_t(16)*n_body);
            const auto * kb = reinterpret_cast<const block_saw_int4_ggml *>(k + row*k1 + h*k2);
            const auto * vb = reinterpret_cast<const block_saw_int4_ggml *>(v + row*v1 + h*v2);
            if (!saw_valid_f16_group(kb, group) || !saw_valid_f16_group(vb, group)) atomicOr(status, 64);
        }
        if (i < int64_t(tail_stride)*nq) {
            const int row = i % tail_stride, qi = i/tail_stride;
            const auto bits = *reinterpret_cast<const uint16_t *>(tail_mask + qi*tm1 + row*sizeof(half));
            if ((bits & 0x7c00u) == 0x7c00u && bits != 0xfc00u) atomicOr(status, 16);
        }
        if (i < int64_t(tail_stride)*4*256) {
            const int d = i % 256, t = (i/256) % tail_stride, h = i/(256*tail_stride);
            if (t < desc[4]) {
                const int slot = desc[6 + t];
                if (slot >= 0 && (slot < history || (kc && slot - history < current))) {
                    const bool from_current = slot >= history;
                    const int row = from_current ? slot - history : slot;
                    const char * kp = from_current ? kc : kt;
                    const char * vp = from_current ? vc : vt;
                    const int64_t ks1 = from_current ? kc1 : kt1, ks2 = from_current ? kc2 : kt2;
                    const int64_t vs1 = from_current ? vc1 : vt1, vs2 = from_current ? vc2 : vt2;
                    const auto kb = *reinterpret_cast<const uint16_t *>(kp + row*ks1 + h*ks2 + d*sizeof(half));
                    const auto vb = *reinterpret_cast<const uint16_t *>(vp + row*vs1 + h*vs2 + d*sizeof(half));
                    if ((kb & 0x7c00u) == 0x7c00u || (vb & 0x7c00u) == 0x7c00u) atomicOr(status, 32);
                }
            }
        }
    }
}

bool ggml_cuda_tqnc_validate_tail(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    const auto * q = dst->src[0];
    const auto * k = dst->src[1];
    const auto * v = dst->src[2];
    const auto * bm = dst->src[3];
    const auto * kt = dst->src[5];
    const auto * vt = dst->src[6];
    const auto * tm = dst->src[7];
    const auto * qo = dst->src[8];
    const auto * rd = dst->src[9];
    const auto * kc = dst->src[10];
    const auto * vc = dst->src[11];
    const auto * lse = ggml_flash_attn_ext_get_lse_out(dst);
    if (!dst->data || !q->data || !k->data || !v->data || !bm->data || !kt->data || !vt->data ||
            !tm->data || !qo->data || !rd->data || (kc && (!kc->data || !vc || !vc->data)) ||
            (lse && !lse->data)) return false;
    const int nq = int(q->ne[1]), n_body = int(k->ne[1]), tail_stride = int(tm->ne[0]);
    const int body_stride = int(rd->ne[0] - 6 - tail_stride);
    const int history_param = ggml_get_op_params_i32(dst, GGML_FLASH_ATTN_EXT_OP_PARAM_TAIL_HISTORY_SLOTS);
    const int history = history_param > 0 ? history_param : int(kt->ne[1]);
    if (nq <= 0 || n_body <= 0 || tail_stride <= 0 || body_stride < 0 || body_stride > 512 ||
            history <= 0 || history > kt->ne[1] || history > vt->ne[1]) return false;
    int * status = nullptr;
    if (!tqnc_status_begin(ctx.stream(), &status)) return false;
    int64_t work = int64_t(nq)*24*256;
    work = std::max(work, int64_t(n_body)*nq);
    if (k->type == GGML_TYPE_SAW_INT4) work = std::max(work, int64_t(n_body)*4*16);
    work = std::max(work, int64_t(tail_stride)*4*256);
    const uint32_t grid = uint32_t(std::min<int64_t>((work + 255)/256, 65535));
    const cudaError_t launch_err = ggml_cuda_tqnc_launch(tqnc_tail_validate_kernel, dim3(grid), dim3(256), ctx.stream(),
        (const char *) q->data, (const char *) k->data, (const char *) v->data,
        (const char *) bm->data, (const char *) tm->data,
        (const char *) kt->data, (const char *) vt->data, kc ? (const char *) kc->data : nullptr,
        vc ? (const char *) vc->data : nullptr, (const int32_t *) qo->data, (const int32_t *) rd->data, status,
        nq, n_body, tail_stride, body_stride, history, kc ? int(kc->ne[1]) : 0, k->type == GGML_TYPE_SAW_INT4,
        q->nb[1], q->nb[2], bm->nb[1], tm->nb[1], k->nb[1], k->nb[2], v->nb[1], v->nb[2],
        kt->nb[1], kt->nb[2], vt->nb[1], vt->nb[2],
        kc ? kc->nb[1] : 0, kc ? kc->nb[2] : 0, vc ? vc->nb[1] : 0, vc ? vc->nb[2] : 0, work);
    int flags = -1;
    const bool valid = tqnc_status_finish(ctx.stream(), status, &flags);
    if (launch_err != cudaSuccess || !valid) {
        GGML_LOG_ERROR("%s: custom tail validation failed (launch=%s flags=%d)\n",
                       __func__, cudaGetErrorString(launch_err), flags);
        tqnc_failure = true;
        return false;
    }
    return true;
}

enum tqnc_row_validation_flags {
    TQNC_ROW_INVALID_INDEX = 1,
    TQNC_ROW_INVALID_OFFSET = 2,
    TQNC_ROW_DUPLICATE = 4,
    TQNC_ROW_INVALID_VALUE = 8,
};

// Same arithmetic and packed layout as the legacy SAW encoder, but writes a
// temporary block and rejects input that cannot safely return through F16.
// This is used only for the explicitly marked exact-tail store.
static __device__ __forceinline__ bool saw_checked_encode(const float * input, block_saw_int4_ggml * output) {
    const int lane = threadIdx.x % 32;
    float values[8];
    bool valid = true;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        values[j] = input[8*lane + j];
        valid = valid && (__float_as_uint(values[j]) & 0x7f800000u) != 0x7f800000u;
    }
    if (!__all_sync(0xffffffffu, valid)) return false;
    saw_fwht(values);
    float lo = INFINITY, hi = -INFINITY;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float partner = __shfl_xor_sync(0xffffffffu, values[j], 1, 32);
        values[j] = __fmul_rn((lane & 1) ? __fsub_rn(partner, values[j]) : __fadd_rn(values[j], partner), 0.25f);
        valid = valid && (__float_as_uint(values[j]) & 0x7f800000u) != 0x7f800000u;
        lo = fminf(lo, values[j]);
        hi = fmaxf(hi, values[j]);
    }
    if (!__all_sync(0xffffffffu, valid)) return false;
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        lo = fminf(lo, __shfl_xor_sync(0xffffffffu, lo, offset, 32));
        hi = fmaxf(hi, __shfl_xor_sync(0xffffffffu, hi, offset, 32));
    }
    const float scale = __fdiv_rn(fmaxf(hi - lo, 1e-8f), 15.0f);
    const float zero = __fdiv_rn(-lo, scale);
    if ((__float_as_uint(scale) & 0x7f800000u) == 0x7f800000u || scale <= 0 ||
        (__float_as_uint(zero) & 0x7f800000u) == 0x7f800000u) return false;
    if (lane == 0) { output->scale = scale; output->zero = zero; }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float norm = __fadd_rn(__fadd_rn(__fdiv_rn(values[j], scale), zero), 0.5f);
        const int quant = min(15, max(0, int(floorf(norm))));
        const int other = __shfl_xor_sync(0xffffffffu, quant, 16, 32);
        if (lane < 16) output->qs[8*lane + j] = uint8_t(quant | (other << 4));
    }
    __syncwarp(0xffffffffu);
    return __all_sync(0xffffffffu, saw_valid_f16_group(output, lane % 16));
}

// 0 = NC key, 1 = NC value, 2 = checked SAW.
template<typename idx_t, int kind>
static __global__ void tqnc_set_rows_kernel(
        const float * src, const idx_t * indices, char * staged, int64_t * destinations,
        unsigned int * owners, int * status, int64_t blocks, int64_t dst_rows, int64_t owner_count,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ni1, int64_t ni2,
        int64_t s1, int64_t s2, int64_t s3, int64_t si0, int64_t si1, int64_t si2,
        int64_t d1, int64_t d2, int64_t d3) {
    const int64_t block = (int64_t(blockIdx.x)*blockDim.x + threadIdx.x)/32;
    if (block >= blocks) return;
    const int64_t element = block*GGML_TQNC_DIM;
    const int64_t i0 = element % ne0;
    const int64_t i1 = (element/ne0) % ne1;
    const int64_t i2 = (element/ne0/ne1) % ne2;
    const int64_t i3 = element/ne0/ne1/ne2;
    const int64_t row = indices[i1*si0 + (i2 % ni1)*si1 + (i3 % ni2)*si2];
    const int lane = threadIdx.x % 32;
    if (row < 0) {
        if (lane == 0) destinations[block] = -1;
        return;
    }
    if (row >= dst_rows) {
        if (lane == 0) atomicOr(status, TQNC_ROW_INVALID_INDEX);
        return;
    }
    const float * input = src + i1*s1 + i2*s2 + i3*s3 + i0;
    constexpr int bytes = kind == 0 ? sizeof(ggml_tqnc_key) :
                          kind == 1 ? sizeof(ggml_tqnc_value) : sizeof(block_saw_int4_ggml);
    const int64_t offset = row*d1 + i2*d2 + i3*d3 + (i0/GGML_TQNC_DIM)*bytes;
    if (offset < 0 || offset % bytes != 0 || offset/bytes >= owner_count) {
        if (lane == 0) atomicOr(status, TQNC_ROW_INVALID_OFFSET);
        return;
    }
    if (lane == 0) {
        destinations[block] = offset;
        if (atomicCAS(owners + offset/bytes, 0u, 1u) != 0u) atomicOr(status, TQNC_ROW_DUPLICATE);
    }
    char * output = staged + block*bytes;
    bool ok;
    if constexpr (kind == 0) {
        ok = tqnc_cuda_encode_key(input, reinterpret_cast<ggml_tqnc_key *>(output));
    } else if constexpr (kind == 1) {
        ok = tqnc_cuda_encode_value(input, reinterpret_cast<ggml_tqnc_value *>(output));
    } else {
        ok = saw_checked_encode(input, reinterpret_cast<block_saw_int4_ggml *>(output));
    }
    if (!ok && lane == 0) atomicOr(status, TQNC_ROW_INVALID_VALUE);
}

template<int kind>
static __global__ void tqnc_commit_rows_kernel(
        const char * staged, const int64_t * destinations, char * dst, int64_t blocks) {
    const int lane = threadIdx.x % 32;
    const int64_t block = (int64_t(blockIdx.x)*blockDim.x + threadIdx.x)/32;
    if (block >= blocks || destinations[block] < 0) return;
    constexpr int bytes = kind == 0 ? sizeof(ggml_tqnc_key) :
                          kind == 1 ? sizeof(ggml_tqnc_value) : sizeof(block_saw_int4_ggml);
    const char * src = staged + block*bytes;
    char * out = dst + destinations[block];
    for (int i = lane; i < bytes; i += 32) out[i] = src[i];
}

template<typename idx_t, int kind>
static bool tqnc_set_rows_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                                 const ggml_tensor * indices, ggml_tensor * dst) {
    GGML_ASSERT(src->type == GGML_TYPE_F32 && src->ne[0] % GGML_TQNC_DIM == 0);
    if (!ggml_cuda_tqnc_tensor_valid(src) || !ggml_cuda_tqnc_tensor_valid(indices) ||
            !ggml_cuda_tqnc_tensor_valid(dst)) return false;
    const int64_t blocks = ggml_nelements(src)/GGML_TQNC_DIM;
    if (blocks == 0) return true;
    const uint64_t grid = uint64_t(blocks)/8 + (blocks % 8 != 0);
    if (grid > UINT32_MAX) return false;
    constexpr size_t bytes = kind == 0 ? sizeof(ggml_tqnc_key) :
                             kind == 1 ? sizeof(ggml_tqnc_value) : sizeof(block_saw_int4_ggml);
    const size_t span = ggml_nbytes(dst);
    if (span == 0 || span % bytes != 0 || span/bytes > SIZE_MAX/sizeof(unsigned int) ||
            uint64_t(span/bytes) > uint64_t(INT64_MAX) ||
            uint64_t(blocks) > SIZE_MAX/bytes || uint64_t(blocks) > SIZE_MAX/sizeof(int64_t)) return false;
    const size_t owner_count = span/bytes;
    int * status = nullptr;
    if (!tqnc_status_begin(ctx.stream(), &status)) return false;
    char * staged = nullptr;
    int64_t * destinations = nullptr;
    unsigned int * owners = nullptr;
    cudaError_t alloc_err = cudaMalloc((void **) &staged, blocks*bytes);
    if (alloc_err == cudaSuccess) alloc_err = cudaMalloc((void **) &destinations, blocks*sizeof(int64_t));
    if (alloc_err == cudaSuccess) alloc_err = cudaMalloc((void **) &owners, owner_count*sizeof(unsigned int));
    if (alloc_err == cudaSuccess) alloc_err = cudaMemsetAsync(owners, 0, owner_count*sizeof(unsigned int), ctx.stream());
    if (alloc_err != cudaSuccess) {
        if (staged) (void) cudaFree(staged);
        if (destinations) (void) cudaFree(destinations);
        if (owners) (void) cudaFree(owners);
        (void) cudaFree(status);
        (void) cudaGetLastError();
        return false;
    }
    constexpr int threads = 256;
    const cudaError_t launch_err = ggml_cuda_tqnc_launch(tqnc_set_rows_kernel<idx_t, kind>,
        dim3(grid), dim3(threads), ctx.stream(),
        (const float *) src->data, (const idx_t *) indices->data, staged, destinations, owners,
        status, blocks, dst->ne[1], owner_count,
        src->ne[0], src->ne[1], src->ne[2], indices->ne[1], indices->ne[2],
        src->nb[1]/sizeof(float), src->nb[2]/sizeof(float), src->nb[3]/sizeof(float),
        indices->nb[0]/sizeof(idx_t), indices->nb[1]/sizeof(idx_t), indices->nb[2]/sizeof(idx_t),
        dst->nb[1], dst->nb[2], dst->nb[3]);
    int validation_flags = -1;
    const bool valid = tqnc_status_finish(ctx.stream(), status, &validation_flags);
    if (launch_err != cudaSuccess || !valid) {
        (void) cudaFree(owners);
        (void) cudaFree(destinations);
        (void) cudaFree(staged);
        GGML_LOG_ERROR("%s: TurboQuant NC stage failed: launch=%s, validation_flags=%d "
                       "(index=1, offset=2, duplicate=4, nonfinite-or-unrepresentable=8, CUDA-status-error=-1)\n",
                       __func__, cudaGetErrorString(launch_err), validation_flags);
        (void) cudaGetLastError();
        return false;
    }
    const cudaError_t commit_launch_err = ggml_cuda_tqnc_launch(tqnc_commit_rows_kernel<kind>,
        dim3(grid), dim3(threads), ctx.stream(),
        staged, destinations, (char *) dst->data, blocks);
    const cudaError_t commit_err = cudaStreamSynchronize(ctx.stream());
    const cudaError_t owners_err = cudaFree(owners);
    const cudaError_t destinations_err = cudaFree(destinations);
    const cudaError_t staged_err = cudaFree(staged);
    const bool committed = commit_launch_err == cudaSuccess && commit_err == cudaSuccess && owners_err == cudaSuccess &&
           destinations_err == cudaSuccess && staged_err == cudaSuccess;
    if (!committed) {
        GGML_LOG_ERROR("%s: TurboQuant NC commit failed (launch %s, synchronize %s); cache may be invalid\n",
                       __func__, cudaGetErrorString(commit_launch_err), cudaGetErrorString(commit_err));
        (void) cudaGetLastError();
    }
    return committed;
}

bool ggml_cuda_tqnc_set_rows(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                             const ggml_tensor * indices, ggml_tensor * dst, bool key) {
    if (!src || !indices || !dst || !src->data || !indices->data || !dst->data ||
            src->type != GGML_TYPE_F32 ||
            (indices->type != GGML_TYPE_I32 && indices->type != GGML_TYPE_I64) ||
            dst->type != (key ? GGML_TYPE_TQNC_K : GGML_TYPE_TQNC_V)) return false;
    if (indices->type == GGML_TYPE_I32) {
        return key ? tqnc_set_rows_launch<int32_t, 0>(ctx, src, indices, dst) :
                     tqnc_set_rows_launch<int32_t, 1>(ctx, src, indices, dst);
    } else {
        return key ? tqnc_set_rows_launch<int64_t, 0>(ctx, src, indices, dst) :
                     tqnc_set_rows_launch<int64_t, 1>(ctx, src, indices, dst);
    }
}

bool ggml_cuda_saw_set_rows_checked(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                                    const ggml_tensor * indices, ggml_tensor * dst) {
    if (!src || !indices || !dst || !src->data || !indices->data || !dst->data ||
            dst->type != GGML_TYPE_SAW_INT4 || src->type != GGML_TYPE_F32) return false;
    if (indices->type == GGML_TYPE_I32) return tqnc_set_rows_launch<int32_t, 2>(ctx, src, indices, dst);
    if (indices->type == GGML_TYPE_I64) return tqnc_set_rows_launch<int64_t, 2>(ctx, src, indices, dst);
    return false;
}

template<typename T, bool key>
static __global__ void tqnc_to_float_kernel(
        const void * src, T * dst, int * status, int64_t blocks, int64_t ne0, int64_t ne1, int64_t ne2,
        int64_t s1, int64_t s2, int64_t s3) {
    const int64_t block = (int64_t(blockIdx.x)*blockDim.x + threadIdx.x)/32;
    if (block >= blocks) return;
    const int64_t block_rows = ne0/GGML_TQNC_DIM;
    const int64_t i0 = block % block_rows;
    const int64_t i1 = (block/block_rows) % ne1;
    const int64_t i2 = (block/block_rows/ne1) % ne2;
    const int64_t i3 = block/block_rows/ne1/ne2;
    const int64_t input = i0 + i1*s1 + i2*s2 + i3*s3;
    T * output = dst + block*GGML_TQNC_DIM;
    bool ok;
    if constexpr (key) {
        ok = tqnc_cuda_decode_key(static_cast<const ggml_tqnc_key *>(src) + input, output);
    } else {
        ok = tqnc_cuda_decode_value(static_cast<const ggml_tqnc_value *>(src) + input, output);
    }
    if (!ok && threadIdx.x % 32 == 0) atomicExch(status, 1);
}

template<typename T, bool key>
static void tqnc_to_float_nc(const void * src, T * dst, int64_t ne0, int64_t ne1,
                             int64_t ne2, int64_t ne3, int64_t s1, int64_t s2, int64_t s3,
                             cudaStream_t stream) {
    GGML_ASSERT(ne0 % GGML_TQNC_DIM == 0);
    const int64_t blocks = (ne0/GGML_TQNC_DIM)*ne1*ne2*ne3;
    if (blocks == 0) return;
    const uint64_t grid = uint64_t(blocks)/8 + (blocks % 8 != 0);
    if (grid > UINT32_MAX) {
        tqnc_failure = true;
        return;
    }
    int * status = nullptr;
    if (!tqnc_status_begin(stream, &status)) {
        tqnc_failure = true;
        return;
    }
    constexpr int threads = 256;
    const cudaError_t launch_err = ggml_cuda_tqnc_launch(tqnc_to_float_kernel<T, key>, dim3(grid), dim3(threads),
                                                         stream, src, dst, status, blocks, ne0, ne1, ne2, s1, s2, s3);
    const bool valid = tqnc_status_finish(stream, status);
    if (launch_err != cudaSuccess || !valid) {
        GGML_LOG_ERROR("%s: invalid TurboQuant NC metadata or CUDA error\n", __func__);
        tqnc_failure = true;
    }
}

template<typename T, bool key>
static void tqnc_to_float_cont(const void * src, T * dst, int64_t n, cudaStream_t stream) {
    GGML_ASSERT(n % GGML_TQNC_DIM == 0);
    const int64_t blocks = n/GGML_TQNC_DIM;
    tqnc_to_float_nc<T, key>(src, dst, GGML_TQNC_DIM, blocks, 1, 1, 1, blocks, blocks, stream);
}

#define GGML_TQNC_DEFINE_CONVERSIONS(name, T, key) \
    void ggml_cuda_tqnc_##name##_cont(const void * src, T * dst, int64_t n, cudaStream_t stream) { \
        tqnc_to_float_cont<T, key>(src, dst, n, stream); \
    } \
    void ggml_cuda_tqnc_##name##_nc(const void * src, T * dst, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, \
                                   int64_t s1, int64_t s2, int64_t s3, cudaStream_t stream) { \
        tqnc_to_float_nc<T, key>(src, dst, ne0, ne1, ne2, ne3, s1, s2, s3, stream); \
    }
GGML_TQNC_DEFINE_CONVERSIONS(key_f16, half, true)
GGML_TQNC_DEFINE_CONVERSIONS(value_f16, half, false)
GGML_TQNC_DEFINE_CONVERSIONS(key_f32, float, true)
GGML_TQNC_DEFINE_CONVERSIONS(value_f32, float, false)
#undef GGML_TQNC_DEFINE_CONVERSIONS
#endif
