#include "scale.cuh"

#define MAX_GRIDDIM_X 0x7FFFFFFF

static __global__ void scale_f32(const float * x, float * dst, const float scale, const float bias, const int64_t nelements) {
    ggml_cuda_pdl_lc();
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;

    ggml_cuda_pdl_sync();
    for (int64_t i = tid; i < nelements; i += stride) {
        dst[i] = scale * x[i] + bias;
    }
}

static __global__ void scale_bf16(const nv_bfloat16 * x, nv_bfloat16 * dst, const float scale, const float bias, const int64_t nelements) {
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;

    for (int64_t i = tid; i < nelements; i += stride) {
        dst[i] = __float2bfloat16(scale * __bfloat162float(x[i]) + bias);
    }
}

static __global__ void scale_f32_vec4(
        const float4 * x, float4 * dst, const float scale, const float bias, const int64_t nelements4) {
    ggml_cuda_pdl_lc();
    const int64_t tid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t) blockDim.x * gridDim.x;

    ggml_cuda_pdl_sync();
    for (int64_t i = tid; i < nelements4; i += stride) {
        const float4 v = x[i];
        dst[i] = make_float4(
            scale * v.x + bias,
            scale * v.y + bias,
            scale * v.z + bias,
            scale * v.w + bias);
    }
}

static void scale_f32_cuda(const float * x, float * dst, const float scale, const float bias, const int64_t nelements, cudaStream_t stream) {
    const int device = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[device].cc;
    if (cc == GGML_CUDA_CC_DGX_SPARK && nelements >= 1024 && nelements % 4 == 0 &&
            (uintptr_t(x) & 0x0F) == 0 && (uintptr_t(dst) & 0x0F) == 0) {
        const int64_t nelements4 = nelements / 4;
        const int64_t num_blocks = (nelements4 + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(
            MIN(MAX_GRIDDIM_X, num_blocks), CUDA_SCALE_BLOCK_SIZE, 0, stream);
        ggml_cuda_kernel_launch(scale_f32_vec4, launch_params,
            (const float4 *) x, (float4 *) dst, scale, bias, nelements4);
        return;
    }
    const int64_t num_blocks = (nelements + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(MIN(MAX_GRIDDIM_X, num_blocks), CUDA_SCALE_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(scale_f32, launch_params, x, dst, scale, bias, nelements);
}

static void scale_bf16_cuda(const nv_bfloat16 * x, nv_bfloat16 * dst, const float scale, const float bias, const int64_t nelements, cudaStream_t stream) {
    const int64_t num_blocks = (nelements + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
    scale_bf16<<<MIN(MAX_GRIDDIM_X, num_blocks), CUDA_SCALE_BLOCK_SIZE, 0, stream>>>(x, dst, scale, bias, nelements);
}

void ggml_cuda_op_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16);

    float scale;
    float bias;
    memcpy(&scale, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&bias,  (float *) dst->op_params + 1, sizeof(float));

    if (src0->type == GGML_TYPE_BF16) {
        scale_bf16_cuda(
            (const nv_bfloat16 *)src0->data, (nv_bfloat16 *)dst->data,
            scale, bias, ggml_nelements(src0), stream);
    } else {
        scale_f32_cuda(
            (const float *)src0->data, (float *)dst->data,
            scale, bias, ggml_nelements(src0), stream);
    }
}
