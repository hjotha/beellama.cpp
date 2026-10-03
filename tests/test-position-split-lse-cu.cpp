// F1b — CUDA FA-LSE route state (plan §3.4).
//
// CUDA LSE export is implemented only for the KVarN-native routes
// (decode-split/vec, windowed, portable, tail). The generic FA path must
// decline an LSE request instead of silently producing a wrong buffer, so this
// test pins that contract:
//
//   1. generic Q4_0 and F16 K/V without LSE stay supported,
//   2. the same shapes with an LSE request are declined as a whole
//      (supports_op(FA) == false), which is the fail-closed behavior,
//   3. the GGML_OP_FLASH_ATTN_EXT_LSE side-output node is supported by CUDA
//      and the scheduler co-locates it with an FA node pinned to CUDA, so the
//      KVarN routes can publish their LSE through the same contract.
//
// Values for the KVarN routes are proven by the F3 integration run (4B,
// P-small, per-route logging), which drives real KVarN records.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <initializer_list>
#include <vector>

static int g_checks = 0;
static int g_failures = 0;
static const char * g_current = "";

#define CHECK(cond) do { \
    ++g_checks; \
    if (!(cond)) { \
        ++g_failures; \
        std::printf("  FAIL [%s] line %d: %s\n", g_current, __LINE__, #cond); \
    } \
} while (0)

static const int D = 64, NQ = 1, NKV = 512, NQH = 4, NKH = 2;

// supports_op for the FA op and for the LSE side-output node, with and without
// the LSE request.
static void probe_support(ggml_backend_t backend, ggml_type kv_type, bool with_lse,
        bool * fa_supported, bool * lse_node_supported) {
    struct ggml_init_params params = { 8 * 1024 * 1024, NULL, true };
    struct ggml_context * ctx = ggml_init(params);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, NQ, NQH, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, kv_type, D, NKV, NKH, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, kv_type, D, NKV, NKH, 1);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, NKV, NQ);
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f, 0.0f, 0.0f);
    ggml_tensor * lse = with_lse ? ggml_flash_attn_ext_lse_out(ctx, fa) : nullptr;

    *fa_supported = ggml_backend_supports_op(backend, fa);
    *lse_node_supported = lse != nullptr && ggml_backend_supports_op(backend, lse);
    ggml_free(ctx);
}

// The scheduler must place the LSE node on the FA node's backend even when the
// FA node is pinned by the user (the position-split placement).
static void probe_colocation(ggml_backend_t backend, bool * colocated) {
    struct ggml_init_params params = { 8 * 1024 * 1024, NULL, true };
    struct ggml_context * ctx = ggml_init(params);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, NQ, NQH, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, NKV, NKH, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, NKV, NKH, 1);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, NKV, NQ);
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f, 0.0f, 0.0f);
    ggml_tensor * lse = ggml_flash_attn_ext_lse_out(ctx, fa);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t backends[2] = { backend, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, NULL, 2, 256, false, true);
    ggml_backend_sched_set_tensor_backend(sched, fa, backend);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, lse);

    *colocated = false;
    if (ggml_backend_sched_alloc_graph(sched, gf)) {
        *colocated = ggml_backend_sched_get_tensor_backend(sched, fa) == backend &&
            ggml_backend_sched_get_tensor_backend(sched, lse) == backend;
    }
    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);
    ggml_free(ctx);
}

int main() {
    ggml_backend_t cuda = ggml_backend_cuda_init(0);
    if (!cuda) {
        std::printf("position-split-lse-cu: no CUDA backend\n");
        return 1;
    }

    for (ggml_type kv_type : { GGML_TYPE_Q4_0, GGML_TYPE_F16 }) {
        g_current = ggml_type_name(kv_type);
        bool fa_sup = false, lse_node_sup = false;

        // 1. plain FA keeps working
        probe_support(cuda, kv_type, false, &fa_sup, &lse_node_sup);
        CHECK(fa_sup);

        // 2. an LSE request on the generic route is declined as a whole
        probe_support(cuda, kv_type, true, &fa_sup, &lse_node_sup);
        CHECK(!fa_sup);

        // 3. the side-output node itself is supported by CUDA
        CHECK(lse_node_sup);
    }

    g_current = "lse_colocation";
    {
        bool colocated = false;
        probe_colocation(cuda, &colocated);
        CHECK(colocated);
    }

    ggml_backend_free(cuda);
    std::printf("position-split-lse-cu: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}