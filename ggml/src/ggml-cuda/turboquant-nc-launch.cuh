#pragma once

#include <cuda_runtime.h>
#include <utility>

// NC validates and synchronizes each launch; PDL/capture cannot preserve this contract.
template<typename Kernel, typename... Args>
static cudaError_t ggml_cuda_tqnc_launch(Kernel kernel, dim3 grid, dim3 block, cudaStream_t stream, Args && ... args) {
    kernel<<<grid, block, 0, stream>>>(std::forward<Args>(args)...);
    return cudaGetLastError();
}
