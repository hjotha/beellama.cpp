#pragma once
#include "ggml.h"
// Experimental CUDA-only native remote producer. Requires GGML_CUDA_KVARN.
// format: GGML_TYPE_Q4_1/IQ4_NL/SAW_INT4/TQNC_K (the latter selects both K/V).
// flags: bit 0 selects the existing AVX2 SAW arithmetic (deferred WHT scale
// and reciprocal multiplication); only valid for SAW. All other bits reject.
// 256 values/head, count 1..512, heads 1..32. Host inputs or current-device
// inputs, already synchronized by the caller. Caller owns host output planes:
// count*heads*(160/144/136/132) bytes. NC K pads the 130-byte block to 132.
// False means no prepared rows may be published; never retry with another codec.
// CPU default and existing format/attention gates are independent of this ABI.
typedef bool (*ggml_backend_cuda_low4_native_prepare_v1_t)(
    int format, uint32_t flags, const float * keys, const float * values, uint32_t count, uint32_t heads,
    void * packed_keys, void * packed_values);
