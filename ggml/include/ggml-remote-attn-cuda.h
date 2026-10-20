#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA proc "ggml_backend_cuda_remote_attn_fp8_interface_v1"
// Host-side CUDA FP8 acceleration for Phase 3 hybrid attention.
struct ggml_remote_attn_cuda_fp8_interface {
    // Pack Q float/half to compact FP8 E5M2 RNE directly on GPU.
    // If src_q is in VRAM, packs directly on GPU before D2H to pinned dst_pinned_fp8.
    // If src_q is host pointer, uploads to GPU scratch, packs, and downloads.
    // Clamping to max finite ±57344.0f (0x7b / 0xfb) and Round-to-Nearest-Even.
    // Fails closed (returns false) if any non-finite value (Inf/NaN) is encountered.
    bool (*pack_q)(const void * src_q, bool is_f16, size_t n_elements,
                   uint8_t * dst_pinned_fp8, void * stream, void * event_done);

    // Unpack OUT FP8 E5M2 to F32 directly on GPU.
    // Transferred H2D from pinned src_pinned_fp8 to GPU VRAM and unpacked on GPU.
    // If dst_out is in VRAM, writes directly to dst_out in VRAM.
    // If dst_out is host pointer, unpacks in GPU scratch and downloads.
    // Fails closed (returns false) if any exponent 31 (Inf/NaN) is encountered.
    bool (*unpack_out)(const uint8_t * src_pinned_fp8, const float * src_lse,
                       size_t nq, size_t out_width, size_t n_head,
                       void * dst_out, size_t dst_stride_bytes,
                       void * stream, void * event_done);

    // Allocate pinned host memory for zero-copy DMA.
    void * (*alloc_pinned)(size_t bytes);

    // Free pinned host memory.
    void (*free_pinned)(void * ptr);

    // Synchronize stream.
    bool (*stream_sync)(void * stream);

    // Synchronize event.
    bool (*event_sync)(void * event);

    // Create non-blocking stream.
    void * (*create_stream)(void);

    // Destroy stream.
    void (*destroy_stream)(void * stream);

    // Create event.
    void * (*create_event)(void);

    // Destroy event.
    void (*destroy_event)(void * event);
};

typedef const struct ggml_remote_attn_cuda_fp8_interface * (*ggml_backend_cuda_remote_attn_fp8_interface_v1_t)(void);

#ifdef __cplusplus
}
#endif
