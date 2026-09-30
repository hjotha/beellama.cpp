#pragma once

// ggml-local-split: In-process heterogeneous accelerator for remote KV cache + attention.
//
// Allows RTX 4070 (CUDA) to keep weights, DeltaNet, prefill, and local attention layers,
// while offloading remote full-attention layers and their KVarN4 KV cache to the Radeon 780M
// (Vulkan:0) via host-pinned persistent ring buffers (zero socket overhead).

#include "ggml-backend.h"
#include "ggml-remote-attn.h"

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Ring buffer configuration parameters
#define GGML_LOCAL_SPLIT_RING_SLOTS 3
#define GGML_LOCAL_SPLIT_MAX_LAYERS 32

// High-resolution profiling metrics per remote layer boundary
struct ggml_local_split_stats {
    uint64_t calls;
    uint64_t bytes_tx;
    uint64_t bytes_rx;
    uint64_t cuda_to_host_us_sum;
    uint64_t host_to_vulkan_us_sum;
    uint64_t vulkan_dispatch_us_sum;
    uint64_t vulkan_attn_us_sum;
    uint64_t vulkan_to_host_us_sum;
    uint64_t host_to_cuda_us_sum;
    uint64_t total_boundary_us_sum;
    uint64_t max_boundary_us;
};

// Initializes the local split in-process accelerator backend.
// vulkan_device_idx: 0 for Radeon 780M (RADV Phoenix).
// ring_slots: 2 (double buffering) or 3 (triple buffering).
GGML_API ggml_backend_t ggml_backend_local_split_init(
    uint32_t vulkan_device_idx,
    uint32_t ring_slots
);

GGML_API bool ggml_backend_is_local_split(ggml_backend_t backend);

// Sets the geometric attention parameters (inherited from ggml_remote_attn_geometry).
GGML_API bool ggml_backend_local_split_set_geometry(
    ggml_backend_t backend,
    const struct ggml_remote_attn_geometry * geo
);

// Connects/Initializes Vulkan compute pipelines and ring buffers.
GGML_API bool ggml_backend_local_split_setup(ggml_backend_t backend);
GGML_API bool ggml_backend_local_split_is_ready(ggml_backend_t backend);
GGML_API void ggml_backend_local_split_set_active(ggml_backend_t backend);

// Session lifecycle (mirrors ggml_remote_attn interface for 100% contract parity)
GGML_API bool ggml_backend_local_split_create_session(
    ggml_backend_t backend,
    uint32_t session_id,
    uint32_t seq_id,
    uint32_t capacity
);

GGML_API bool ggml_backend_local_split_destroy_session(
    ggml_backend_t backend,
    uint32_t session_id
);

GGML_API bool ggml_backend_local_split_reset(
    ggml_backend_t backend,
    uint32_t session_id
);

GGML_API bool ggml_backend_local_split_trim(
    ggml_backend_t backend,
    uint32_t session_id,
    int32_t pos0
);

GGML_API void ggml_backend_local_split_set_active_session(
    ggml_backend_t backend,
    uint32_t session_id
);

GGML_API bool ggml_backend_local_split_failed(ggml_backend_t backend);

// Metrics and JSON profiling
GGML_API const char * ggml_backend_local_split_stats_json(ggml_backend_t backend);
GGML_API void ggml_backend_local_split_reset_stats(ggml_backend_t backend);

// In-process forward execution called during graph compute
GGML_API bool ggml_local_split_exec(struct ggml_tensor * node);

#ifdef __cplusplus
}
#endif
