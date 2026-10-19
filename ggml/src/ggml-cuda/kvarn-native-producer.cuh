#pragma once

#include "ggml-kvarn-native-producer.h"

// Original-F32 native RPC owner acceleration; see public helper ABI header.
bool ggml_cuda_kvarn_native_append(
    const float * keys, const float * values, uint32_t global_begin,
    uint32_t count, uint32_t heads, uint32_t iterations,
    const ggml_fp16_t * stage_k, const ggml_fp16_t * stage_v,
    ggml_fp16_t * append_k, ggml_fp16_t * append_v,
    void * records_k, void * records_v);
