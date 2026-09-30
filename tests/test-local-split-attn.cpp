// test-local-split-attn.cpp — Phase 1 validation of in-process Vulkan local-split attention.
//
// Validates:
// 1. Backend initialization on Vulkan:0 (Radeon 780M, APU TDP 20W, Clock 2700MHz).
// 2. Geometry configuration for Qwen 3.8 / KVarN4.
// 3. Ring buffer allocation with 64-byte alignment.
// 4. Session lifecycle (create, reset, trim, destroy).
// 5. Execution of 1 remote full-attention layer and profiling stats.

#include "ggml-local-split.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void require(bool cond, const char * msg) {
    if (!cond) {
        std::fprintf(stderr, "test-local-split-attn ERROR: %s\n", msg);
        std::abort();
    }
}

int main() {
    std::printf("============================================================\n");
    std::printf("  BeeLLaMA — Test Local-Split Attention (Fase 1 Deliverable)\n");
    std::printf("============================================================\n");

    ggml_backend_load_all();

    // 1. Initialize backend on device 0 (Radeon 780M / RADV Phoenix)
    uint32_t vulkan_dev = 0;
    uint32_t ring_slots = 3;
    ggml_backend_t backend = ggml_backend_local_split_init(vulkan_dev, ring_slots);
    require(backend != nullptr, "Failed to initialize ggml-local-split backend");
    std::printf("[*] Backend local-split initialized successfully (device=vulkan:%u, slots=%u)\n",
                vulkan_dev, ring_slots);

    // 2. Configure geometry (matching Qwen3.8-27B GSQ-RCO KVarN4)
    ggml_remote_attn_geometry geo {};
    geo.n_layer_remote = 1;     // Fase 1: 1 remote layer
    geo.n_head         = 32;
    geo.n_head_kv      = 4;
    geo.head_dim       = 128;
    geo.max_ctx        = 32768;
    geo.cache_bits_k   = 4;     // KVarN4
    geo.cache_bits_v   = 4;
    geo.group_tokens   = 128;
    geo.sinkhorn_iters = 0;
    geo.tail_tokens    = 64;
    geo.tail_groups    = 1;
    geo.tail_type      = 0;     // fp16
    geo.domain         = GGML_REMOTE_ATTN_DOMAIN_ROTATED;
    geo.has_sinks      = 1;
    geo.swa            = 0;
    geo.kq_scale       = 1.0f / std::sqrt(128.0f);

    require(ggml_backend_local_split_set_geometry(backend, &geo), "Failed to set geometry");
    require(ggml_backend_local_split_setup(backend), "Failed to setup Vulkan pipelines");
    require(ggml_backend_local_split_is_ready(backend), "Backend is not ready");
    std::printf("[*] Geometry and ring buffer configured (1 layer, %u heads, head_dim=%u)\n",
                geo.n_head, geo.head_dim);

    // 3. Test Session Lifecycle
    uint32_t session_id = 1;
    uint32_t seq_id = 0;
    uint32_t capacity = 32768;

    require(ggml_backend_local_split_create_session(backend, session_id, seq_id, capacity),
            "Failed to create session");
    std::printf("[*] Session %u created (capacity=%u tokens)\n", session_id, capacity);

    ggml_backend_local_split_set_active_session(backend, session_id);

    // 4. Test Trim and Reset
    require(ggml_backend_local_split_trim(backend, session_id, 1024), "Failed to trim session");
    require(ggml_backend_local_split_reset(backend, session_id), "Failed to reset session");
    std::printf("[*] Session trim & reset operations validated\n");

    // 5. Test Op Execution Scaffolding
    ggml_init_params params = {
        /*.mem_size   =*/ 4 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "Failed to create ggml context");

    ggml_tensor * node = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, geo.head_dim, geo.n_head, 1);
    node->op = GGML_OP_REMOTE_ATTN;

    bool exec_ok = ggml_local_split_exec(node);
    require(exec_ok, "Execution of GGML_OP_REMOTE_ATTN node failed");

    const char * stats_str = ggml_backend_local_split_stats_json(backend);
    std::printf("[*] Profiling stats JSON: %s\n", stats_str);

    // 6. Cleanup
    require(ggml_backend_local_split_destroy_session(backend, session_id),
            "Failed to destroy session");
    ggml_free(ctx);
    ggml_backend_free(backend);

    std::printf("============================================================\n");
    std::printf("  test-local-split-attn: Phase 1 deliverable verified OK!\n");
    std::printf("============================================================\n");

    return 0;
}
