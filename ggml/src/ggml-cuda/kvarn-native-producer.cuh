#pragma once

#include "ggml-kvarn-native-producer.h"

// Original-F32 native RPC owner acceleration; see public helper ABI header.
bool ggml_cuda_kvarn_native_append(
    const float * keys, const float * values, uint32_t global_begin,
    uint32_t count, uint32_t heads, uint32_t iterations,
    const ggml_fp16_t * stage_k, const ggml_fp16_t * stage_v,
    ggml_fp16_t * append_k, ggml_fp16_t * append_v,
    void * records_k, void * records_v);

// Q8_0 raw-domain counterpart, independent of KVarN record format.
bool ggml_cuda_q8_native_prepare(const float * keys, const float * values,
        uint32_t count, uint32_t heads, void * packed_keys, void * packed_values);

// Q4_0 raw-domain counterpart, independent of KVarN record format.
bool ggml_cuda_q4_native_prepare(const float * keys, const float * values,
        uint32_t count, uint32_t heads, void * packed_keys, void * packed_values);

// Experimental native Q4_1/IQ4_NL/SAW/NC rows, format-specific wire ABI.
bool ggml_cuda_low4_native_prepare(int format, uint32_t flags, const float * keys, const float * values,
        uint32_t count, uint32_t heads, void * packed_keys, void * packed_values);
