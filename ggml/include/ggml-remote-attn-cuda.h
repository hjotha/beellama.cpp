#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA proc "ggml_backend_cuda_remote_attn_fp8_interface_v1"
// Host-side CUDA FP8 acceleration for Phase 3 hybrid attention.
// All operations are synchronous, including error retirement. A supplied stream
// must contain the Q producer dependency; nullptr conservatively synchronizes
// the Q device. On false, output is invalid and must not be consumed/transmitted.
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


// CUDA proc "ggml_backend_cuda_remote_attn_fp8_interface_v2".
// One context per async slot. Calls complete all CUDA work before returning,
// including on error; callers retain their input/output buffers until return.
// On false, output is invalid and must not be consumed or transmitted.
struct ggml_remote_attn_cuda_fp8_interface_v2 {
    // CUDA device owning ptr; -1 for CPU memory, -2 for unsupported pointers.
    // Managed allocations are deliberately unsupported by this wire fast path.
    int (*device_for_pointer)(const void * ptr);
    void * (*create_context)(int device);
    void (*destroy_context)(void * context);
    // producer_event is an already-recorded CUDA event. With nullptr, synchronize
    // the source device before reading Q. This conservative fallback preserves
    // producer ordering even when the caller does not own its GGML CUDA stream.
    bool (*pack_q)(void * context, const void * src_q, bool is_f16,
                   size_t n_elements, uint8_t * dst_pinned_fp8, void * producer_event);
    bool (*unpack_out)(void * context, const uint8_t * src_pinned_fp8,
                       const float * src_lse, size_t nq, size_t out_width, size_t n_head,
                       void * dst_out, size_t dst_stride_bytes);
    void * (*alloc_pinned)(size_t bytes);
    void (*free_pinned)(void * ptr);
};
typedef const struct ggml_remote_attn_cuda_fp8_interface_v2 *
    (*ggml_backend_cuda_remote_attn_fp8_interface_v2_t)(void);

// CUDA proc "ggml_backend_cuda_remote_attn_fp8_interface_v3".
// Optional mapped-host extension for explicitly small batches (1 <= nq <= 9).
// v1/v2 layouts remain unchanged. Contexts and ordinary pinned buffers are owned
// through base. Small calls require Q/OUT in the context's device memory; their
// CPU wire buffers need not themselves be mapped. Each context owns separate
// grow-only mapped TX/RX sidecars and retires its stream before returning.
// supports_small checks device capabilities; allocation/mapping can still fail.
// A false result is a hard error, never permission to consume a partial output.
struct ggml_remote_attn_cuda_fp8_interface_v3 {
    const struct ggml_remote_attn_cuda_fp8_interface_v2 * base;
    bool (*supports_small)(int device);
    bool (*pack_small)(void * context, const void * src_q, bool is_f16,
                       size_t nq, size_t out_width, uint8_t * dst_fp8,
                       void * producer_event);
    bool (*unpack_small)(void * context, const uint8_t * src_fp8,
                         const float * src_lse, size_t nq, size_t out_width,
                         size_t n_head, void * dst_out, size_t dst_stride_bytes);
};
typedef const struct ggml_remote_attn_cuda_fp8_interface_v3 *
    (*ggml_backend_cuda_remote_attn_fp8_interface_v3_t)(void);

// CUDA proc "ggml_backend_cuda_remote_attn_retirement_interface_v1".
// Separate extension: all FP8 v1/v2/v3 table layouts remain unchanged. Applies
// only to contexts created through base and their private serving stream.
// This is a deallocation barrier, not an execution-success or reuse decision.
// True permits releasing external staging buffers after previous helper calls;
// false means they must remain alive (quarantine). Poison is never cleared.
struct ggml_remote_attn_cuda_retirement_interface_v1 {
    const struct ggml_remote_attn_cuda_fp8_interface_v2 * base;
    bool (*retire_context)(void * context);
};
typedef const struct ggml_remote_attn_cuda_retirement_interface_v1 *
    (*ggml_backend_cuda_remote_attn_retirement_interface_v1_t)(void);

#ifdef __cplusplus
}
#endif
