// Phase-1/2 scaffold test for ggml_kvarn_split_fused / GGML_OP_KVARN_SPLIT_FUSED
// ("peça (c)" of the position-split plan: the fused overflow + merge span).
//
// Scope: pure node-creation metadata on CPU. The op is created through the
// public API and the test verifies
//   * the output shape mirrors the current o_merged ([D_v*H_q, Q*B] F32, the
//     shape produced by build_attn_position_split()'s merge chain in
//     src/llama-graph.cpp), and
//   * the op-param roundtrip (ggml_get_op_params_i32/f32) and source wiring.
//
// No backend is involved and ggml_backend_supports_op is deliberately NOT
// called: the backend gate (LLAMA_OVF_FUSED_SPAN=1 + shape contract plus the
// registered executor) lives in ggml_backend_cuda_device_supports_op and is
// covered by the GPU test (test-kvarn-split-fused). The CPU path never
// consults supports_op for this op (ggml-cpu.cpp is intentionally untouched),
// and nothing executes the op - CPU aborts loudly if it ever receives it.
//
// No GPU required.

#include "ggml.h"
// ggml_get_op_params_i32/f32 are internal API (like tests/test-kvarn.cpp).
#include "../ggml/src/ggml-impl.h"

#include <cstdio>
#include <cstring>

static int g_checks = 0;
static int g_failures = 0;

#define CHECK(cond) do { \
    ++g_checks; \
    if (!(cond)) { \
        ++g_failures; \
        std::printf("  FAIL line %d: %s\n", __LINE__, #cond); \
    } \
} while (0)

int main() {
    ggml_init_params params = { 16 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::printf("FAIL: ggml_init failed\n");
        return 1;
    }

    const int64_t D    = 64;
    const int64_t Hq   = 4;
    const int64_t Hkv  = 2;
    const int64_t Q    = 1; // phase-2 contract: decode-only, one query row
    const int64_t B    = 1;
    const int     P    = 128;
    const int     n_ovf_rows = 64;
    const int     n_kv_live  = 160; // P < n_kv_live <= P + n_ovf_rows
    const int     bits_k = 4;
    const int     bits_v = 4;
    const float   kq_scale = 0.125f;
    const int     il = 7;

    ggml_tensor * q_cur   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, Hq,  Q, B);
    ggml_tensor * k_cur   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, Hkv, Q, B);
    ggml_tensor * v_cur   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, Hkv, Q, B);
    // Local results fed to the merge, exactly the shapes the legacy chain
    // passes: o_merged-like [D*H_q, Q*B] and LSE [H_q, Q, B].
    ggml_tensor * o_local = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D*Hq, Q*B);
    ggml_tensor * lse     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Hq, Q, B);
    // Overflow Q4_0 companion views: [D*H_kv, n_ovf_rows] 2D rows over a
    // storage tensor, exactly the packed-view shape the builder passes.
    ggml_tensor * k_storage = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, D*Hkv, 4*n_ovf_rows);
    ggml_tensor * v_storage = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, D*Hkv, 4*n_ovf_rows);
    ggml_tensor * k_ovf   = ggml_view_2d(ctx, k_storage, D*Hkv, n_ovf_rows,
            k_storage->nb[1], k_storage->nb[1]);
    ggml_tensor * v_ovf   = ggml_view_2d(ctx, v_storage, D*Hkv, n_ovf_rows,
            v_storage->nb[1], v_storage->nb[1]);
    ggml_tensor * mask    = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_ovf_rows, Q, 1, B);
    ggml_tensor * k_idxs  = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, Q*B);
    ggml_tensor * v_idxs  = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, Q*B);

    ggml_tensor * op = ggml_kvarn_split_fused(
            ctx,
            q_cur, k_cur, v_cur,
            o_local, lse,
            k_ovf, v_ovf,
            mask,
            k_idxs, v_idxs,
            P, n_kv_live, bits_k, bits_v, n_ovf_rows, kq_scale, il, 4242);

    CHECK(op != nullptr);

    // Shape: mirrors the current o_merged = reshape_2d([D, H_q, Q, B]).
    ggml_tensor * o_merged_ref4d = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, Hq, Q, B);
    ggml_tensor * o_merged_ref = ggml_reshape_2d(ctx, o_merged_ref4d,
            o_merged_ref4d->ne[0]*o_merged_ref4d->ne[1],
            o_merged_ref4d->ne[2]*o_merged_ref4d->ne[3]);

    CHECK(op->op == GGML_OP_KVARN_SPLIT_FUSED);
    CHECK(op->type == GGML_TYPE_F32);
    // Decode-only phase 2: Q*B == 1, so trailing dims collapse (n_dims 1).
    CHECK(ggml_n_dims(op) == (Q*B > 1 ? 2 : 1));
    CHECK(op->ne[0] == D*Hq);
    CHECK(op->ne[1] == Q*B);
    CHECK(op->ne[0] == o_merged_ref->ne[0] && op->ne[1] == o_merged_ref->ne[1]);

    // Op name string.
    CHECK(ggml_op_name(op->op) != nullptr && std::strcmp(ggml_op_name(op->op), "KVARN_SPLIT_FUSED") == 0);
    CHECK(ggml_op_symbol(op->op) != nullptr);
    CHECK(std::strcmp(ggml_op_desc(op), "KVARN_SPLIT_FUSED") == 0);

    // Op-param roundtrip (see ggml_kvarn_split_fused_op_param).
    CHECK(ggml_get_op_params_i32(op, GGML_KVARN_SPLIT_FUSED_PARAM_P)          == P);
    CHECK(ggml_get_op_params_i32(op, GGML_KVARN_SPLIT_FUSED_PARAM_N_KV_LIVE)  == n_kv_live);
    CHECK(ggml_get_op_params_i32(op, GGML_KVARN_SPLIT_FUSED_PARAM_BITS_K)     == bits_k);
    CHECK(ggml_get_op_params_i32(op, GGML_KVARN_SPLIT_FUSED_PARAM_BITS_V)     == bits_v);
    CHECK(ggml_get_op_params_i32(op, GGML_KVARN_SPLIT_FUSED_PARAM_N_OVF_ROWS) == n_ovf_rows);
    CHECK(ggml_get_op_params_f32(op, GGML_KVARN_SPLIT_FUSED_PARAM_KQ_SCALE)   == kq_scale);
    CHECK(ggml_get_op_params_i32(op, GGML_KVARN_SPLIT_FUSED_PARAM_LAYER)      == il);
    CHECK(ggml_get_op_params_i32(op, GGML_KVARN_SPLIT_FUSED_PARAM_BUNDLE_ID)  == 4242);

    // Source wiring (phase-2 slot contract).
    CHECK(op->src[0] == q_cur);
    CHECK(op->src[1] == k_cur);
    CHECK(op->src[2] == v_cur);
    CHECK(op->src[3] == o_local);
    CHECK(op->src[4] == lse);
    CHECK(op->src[5] == k_ovf);
    CHECK(op->src[6] == v_ovf);
    CHECK(op->src[7] == mask);
    CHECK(op->src[8] == k_idxs);
    CHECK(op->src[9] == v_idxs);

    // The raw views keep their companion storage reachable through view_src:
    // the executor binds that storage to perform the overflow store.
    CHECK(k_ovf->view_src == k_storage && v_ovf->view_src == v_storage);

    // A second node re-evaluated after later creations proves the slots are
    // per-node, not shared state.
    ggml_tensor * op2 = ggml_kvarn_split_fused(
            ctx,
            q_cur, k_cur, v_cur,
            o_local, lse,
            k_ovf, v_ovf,
            mask,
            k_idxs, v_idxs,
            P + 32, n_kv_live + 32, bits_k, bits_v, n_ovf_rows, kq_scale + 0.5f, il + 1, 4243);
    CHECK(op2 != nullptr);
    CHECK(ggml_get_op_params_i32(op2, GGML_KVARN_SPLIT_FUSED_PARAM_P) == P + 32);
    CHECK(ggml_get_op_params_i32(op, GGML_KVARN_SPLIT_FUSED_PARAM_P)  == P);
    CHECK(ggml_get_op_params_i32(op2, GGML_KVARN_SPLIT_FUSED_PARAM_LAYER) == il + 1);
    CHECK(ggml_get_op_params_i32(op, GGML_KVARN_SPLIT_FUSED_PARAM_LAYER)  == il);
    CHECK(op2->ne[0] == op->ne[0] && op2->ne[1] == op->ne[1]);

    ggml_free(ctx);

    std::printf("test-kvarn-split-fused-scaffold: %d checks, %d failures\n", g_checks, g_failures);
    if (g_failures != 0) {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
