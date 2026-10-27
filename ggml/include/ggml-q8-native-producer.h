#pragma once
#include "ggml.h"
// Opt-in CUDA producer for original-domain Q8_0 K/V rows. Inputs may be host
// or device pointers on the current CUDA device; cross-device pointers and
// asynchronous stream handoff are not supported. The current implementation
// is available only when GGML_CUDA_KVARN is enabled. Outputs are caller-owned
// host buffers of count*heads*272 bytes per plane. A false result publishes no
// prepared rows. HIP/MUSA and multiple CUDA devices are not yet qualified.
typedef bool (*ggml_backend_cuda_q8_native_prepare_v1_t)(
    const float * keys, const float * values, uint32_t count, uint32_t heads,
    void * packed_keys, void * packed_values);
