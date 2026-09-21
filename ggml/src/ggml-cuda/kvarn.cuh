#pragma once

#include "common.cuh"

constexpr uint32_t GGML_CUDA_KVARN_STORE_ROUTE_STATS_ABI_VERSION = 2;

struct ggml_cuda_kvarn_store_route_stats {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t headwide_workspace;
    uint64_t headwide_monolithic;
    uint64_t single_slice_workspace;
    uint64_t direct_store;
    uint64_t high_shared_fallback;
    uint64_t low_shared_store;
    uint64_t sealer_128;
    uint64_t sealer_256;
    uint64_t sealer_candidates;
};

void ggml_cuda_op_kvarn_store(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_kvarn_materialize(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Bulk q4_0 -> KVarN conversion of complete 128-token groups (used by the
// host-side state converter when a CUDA device is available). `rows` is a
// host pointer to n_tokens q4_0 rows of `row_bytes`; `records` receives
// (n_tokens/128) * (n_head_kv*head_dim/128) records of record_bytes each.
// Returns false when the geometry is unsupported or a CUDA call fails so the
// caller can fall back to the CPU converter.
bool ggml_cuda_kvarn_convert_q4(
        const void * rows,
        size_t row_bytes,
        int n_tokens,
        int n_embd,
        int n_head_kv,
        int head_dim,
        int bits,
        int iterations,
        bool value,
        int source_rotation,
        void * records);

size_t ggml_cuda_kvarn_required_shared_bytes();
size_t ggml_cuda_kvarn_low_shared_bytes();

void ggml_cuda_kvarn_profile_dump();
void ggml_cuda_kvarn_store_route_stats_reset();
void ggml_cuda_kvarn_store_route_stats_get(ggml_cuda_kvarn_store_route_stats * stats);
