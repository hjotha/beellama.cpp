// Phase-1 scaffold test for ggml_kvarn_split_fused / GGML_OP_KVARN_SPLIT_FUSED
// ("peça (c)" of the position-split plan: the fused local+overflow attention +
// LSE merge span).
//
// Scope: pure node-creation metadata on CPU. The op is created through the
// public API and the test verifies
//   * the output shape mirrors the current o_merged ([D_v*H_q, Q*B] F32, the
//     shape produced by build_attn_inp_kv_overflow()'s
//     build_attn_mha()/merge chain in src/llama-graph.cpp), and
//   * the op-param roundtrip (ggml_get_op_params_i32) and source wiring.
//
// No backend is involved and ggml_backend_supports_op is deliberately NOT
// called: the scaffold's only backend gate (LLAMA_OVF_FUSED_SPAN=1 + shape
// contract) lives in ggml_backend_cuda_device_supports_op and is covered by
// phase-2 GPU tests. The CPU path never consults supports_op for this op
// (ggml-cpu.cpp is intentionally untouched), and nothing executes the op -
// every backend aborts loudly if it ever receives it ("not implemented in
// phase 1").
//
// No GPU required.

#include "ggml.h"
// ggml_get_op_params_i32 is internal API (like tests/test-kvarn.cpp).
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
    const int64_t Q    = 2;
    const int64_t B    = 1;
    const int     P    = 128;
    const int     n_ovf_rows = 64;
    const int     n_kv_live  = 160; // P < n_kv_live <= P + n_ovf_rows
    const int     bits_k = 4;
    const int     bits_v = 4;

    ggml_tensor * q_cur   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, Hq,  Q, B);
    ggml_tensor * k_cur   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, Hkv, Q, B);
    ggml_tensor * v_cur   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, Hkv, Q, B);
    ggml_tensor * k_local = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, Hkv, P, 1);
    ggml_tensor * v_local = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, Hkv, P, 1);
    // Overflow Q4_0 companion views: [D*H_kv, n_ovf_rows] (2D rows), same
    // extent as the cropped contiguous range mask.
    ggml_tensor * k_ovf   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, D*Hkv, n_ovf_rows);
    ggml_tensor * v_ovf   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, D*Hkv, n_ovf_rows);
    ggml_tensor * mask    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_ovf_rows, Q, 1, B);

    ggml_tensor * op = ggml_kvarn_split_fused(
            ctx,
            q_cur, k_cur, v_cur,
            k_local, v_local,
            k_ovf, v_ovf,
            mask,
            nullptr, nullptr, // no exact tail
            P, n_kv_live, bits_k, bits_v, n_ovf_rows);

    CHECK(op != nullptr);

    // Shape: mirrors the current o_merged = reshape_2d([D, H_q, Q, B]).
    ggml_tensor * o_merged_ref4d = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, Hq, Q, B);
    ggml_tensor * o_merged_ref = ggml_reshape_2d(ctx, o_merged_ref4d,
            o_merged_ref4d->ne[0]*o_merged_ref4d->ne[1],
            o_merged_ref4d->ne[2]*o_merged_ref4d->ne[3]);

    CHECK(op->op == GGML_OP_KVARN_SPLIT_FUSED);
    CHECK(op->type == GGML_TYPE_F32);
    CHECK(ggml_n_dims(op) == 2);
    CHECK(op->ne[0] == D*Hq);
    CHECK(op->ne[1] == Q*B);
    CHECK(op->ne[0] == o_merged_ref->ne[0] && op->ne[1] == o_merged_ref->ne[1]);

    // Op name string.
    CHECK(ggml_op_name(op->op) != nullptr && std::strcmp(ggml_op_name(op->op), "KVARN_SPLIT_FUSED") == 0);
    CHECK(ggml_op_symbol(op->op) != nullptr);
    CHECK(std::strcmp(ggml_op_desc(op), "KVARN_SPLIT_FUSED") == 0);

    // Op-param roundtrip (int32 slots, see ggml_kvarn_split_fused_op_param).
    CHECK(ggml_get_op_params_i32(op, GGML_KVARN_SPLIT_FUSED_PARAM_P)          == P);
    CHECK(ggml_get_op_params_i32(op, GGML_KVARN_SPLIT_FUSED_PARAM_N_KV_LIVE)  == n_kv_live);
    CHECK(ggml_get_op_params_i32(op, GGML_KVARN_SPLIT_FUSED_PARAM_BITS_K)     == bits_k);
    CHECK(ggml_get_op_params_i32(op, GGML_KVARN_SPLIT_FUSED_PARAM_BITS_V)     == bits_v);
    CHECK(ggml_get_op_params_i32(op, GGML_KVARN_SPLIT_FUSED_PARAM_N_OVF_ROWS) == n_ovf_rows);

    // Source wiring (phase-1 slot contract; NULL tail for the no-tail case).
    CHECK(op->src[0] == q_cur);
    CHECK(op->src[1] == k_cur);
    CHECK(op->src[2] == v_cur);
    CHECK(op->src[3] == k_local);
    CHECK(op->src[4] == v_local);
    CHECK(op->src[5] == k_ovf);
    CHECK(op->src[6] == v_ovf);
    CHECK(op->src[7] == mask);
    CHECK(op->src[8] == nullptr);
    CHECK(op->src[9] == nullptr);

    // Tail variant: k_tail/v_tail are wired together (and only together; the
    // creator asserts the mismatch case).
    ggml_tensor * k_tail = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, Hkv, 4, 1);
    ggml_tensor * v_tail = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, Hkv, 4, 1);
    ggml_tensor * op_tail = ggml_kvarn_split_fused(
            ctx,
            q_cur, k_cur, v_cur,
            k_local, v_local,
            k_ovf, v_ovf,
            mask,
            k_tail, v_tail,
            P, n_kv_live, bits_k, bits_v, n_ovf_rows);
    CHECK(op_tail != nullptr);
    CHECK(op_tail->src[8] == k_tail);
    CHECK(op_tail->src[9] == v_tail);
    CHECK(op_tail->ne[0] == op->ne[0] && op_tail->ne[1] == op->ne[1]);
    CHECK(ggml_get_op_params_i32(op_tail, GGML_KVARN_SPLIT_FUSED_PARAM_N_OVF_ROWS) == n_ovf_rows);

    // Re-evaluating the params of the first node after the second creation
    // proves the slots are per-node, not shared state.
    CHECK(ggml_get_op_params_i32(op, GGML_KVARN_SPLIT_FUSED_PARAM_P) == P);

    ggml_free(ctx);

    std::printf("test-kvarn-split-fused-scaffold: %d checks, %d failures\n", g_checks, g_failures);
    if (g_failures != 0) {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
