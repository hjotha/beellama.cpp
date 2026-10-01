// Standalone CPU tests for the planned draft-MTP KV reservation estimator.
// Build (recorded in the artifact report):
//   g++ -std=c++17 -O2 -Wall -Wextra -I src -I include -I ggml/include
//     tests/test-kv-mixed-mtp-budget.cpp src/llama-kv-mixed-mtp-budget.cpp
//     src/llama-kv-mixed-placement.cpp
//     -L /home/hjotha/beellama-mixed-kv-20261001-130910/baseline-bin
//     -lggml -lggml-base -lllama
//     -Wl,-rpath,/home/hjotha/beellama-mixed-kv-20261001-130910/baseline-bin
//     -o /home/hjotha/beellama-mixed-kv-20261001-130910/opencode/snapshots/test-kv-mixed-mtp-budget

#include "llama-kv-mixed-mtp-budget.h"

#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>


static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond) do { \
    ++g_checks; \
    if (!(cond)) { \
        ++g_failures; \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    } \
} while (0)

// Qwen3.6-like MTP layer geometry (KVarN4 target model)
static llama_kv_mixed_mtp_layer qwen_mtp_layer(uint32_t il) {
    llama_kv_mixed_mtp_layer l;
    l.layer = il;
    l.head_dim_k = 256;
    l.head_dim_v = 256;
    l.n_head_kv = 4;
    l.n_embd_k_gqa = 1024;
    l.n_embd_v_gqa = 1024;
    return l;
}

static llama_kv_mixed_mtp_input qwen_input(uint32_t n_ctx) {
    llama_kv_mixed_mtp_input in;
    in.n_ctx = n_ctx;
    in.n_seq_max = 1;
    in.kv_unified = false;
    in.n_batch = 256;
    in.n_ubatch = 256;
    in.kvarn = LLAMA_KVARN_K4V4_G128;
    in.tail_tokens = 0;
    in.tail_type = GGML_TYPE_F16;
    in.tail_rollback_tokens = 1;
    in.layers.push_back(qwen_mtp_layer(32));
    return in;
}

static void test_kvarn4_active() {
    const uint32_t n_ctx = 204800;
    llama_kv_mixed_mtp_input in = qwen_input(n_ctx);
    llama_kv_mixed_mtp_budget out = {};
    const auto status = llama_kv_mixed_mtp_budget_estimate(in, out);
    CHECK(status == LLAMA_KV_MIXED_OK);
    CHECK(out.error[0] == '\0');
    CHECK(out.layer_bytes.size() == 1);
    CHECK(out.total_bytes == out.layer_bytes[0]);

    // direct reference through the reviewed cost model with the same resolved
    // inputs (KVarN4, capacity, canonical stage/tail/rollback)
    llama_kv_mixed_sizing sizing = {};
    sizing.capacity_tokens = n_ctx;
    sizing.n_seq_max = 1;
    sizing.stage_tail_groups = 2; // canonical non-SWA
    sizing.stage_reserve_groups = 1;
    sizing.tail_exact_tokens = 0; // intrinsic min(128, capacity)
    sizing.tail_rollback_tokens = 1;
    sizing.tail_type = GGML_TYPE_F16;
    llama_kv_mixed_layer_params layer = {};
    layer.layer = 32;
    layer.head_dim_k = 256;
    layer.head_dim_v = 256;
    layer.n_head_kv = 4;
    layer.n_embd_k_gqa = 1024;
    layer.n_embd_v_gqa = 1024;
    layer.kvarn_bits_k = 4;
    layer.kvarn_bits_v = 4;
    llama_kv_mixed_layer_cost cost = {};
    CHECK(llama_kv_mixed_estimate_layer_cost(layer, sizing, cost, nullptr, 0) == LLAMA_KV_MIXED_OK);
    CHECK(out.total_bytes == cost.local_bytes);
    // local_bytes is the KVarN representation (records + stage + intrinsic tail)
    CHECK(cost.local_bytes > 0);
    // intrinsic tail: tail 0 resolves to the same bytes as an explicit 128
    // token tail at this capacity
    llama_kv_mixed_mtp_input in128 = in;
    in128.tail_tokens = 128;
    llama_kv_mixed_mtp_budget out128 = {};
    CHECK(llama_kv_mixed_mtp_budget_estimate(in128, out128) == LLAMA_KV_MIXED_OK);
    CHECK(out128.total_bytes == out.total_bytes);
    // rollback depth is honored: deeper rollback adds exact-tail rows
    llama_kv_mixed_mtp_input in_rb = in;
    in_rb.tail_rollback_tokens = 8;
    llama_kv_mixed_mtp_budget out_rb = {};
    CHECK(llama_kv_mixed_mtp_budget_estimate(in_rb, out_rb) == LLAMA_KV_MIXED_OK);
    CHECK(out_rb.total_bytes > out.total_bytes);
}

static void test_standard_types_actual_bytes() {
    const uint32_t n_ctx = 204800;
    const struct { ggml_type k; ggml_type v; } cases[] = {
        { GGML_TYPE_Q4_0, GGML_TYPE_Q4_0 },
        { GGML_TYPE_Q8_0, GGML_TYPE_Q8_0 },
        { GGML_TYPE_F16,  GGML_TYPE_F16 },
        { GGML_TYPE_Q5_0, GGML_TYPE_Q8_0 }, // asymmetric draft K/V
    };
    for (const auto & c : cases) {
        llama_kv_mixed_mtp_input in = qwen_input(n_ctx);
        in.kvarn = LLAMA_KVARN_TYPE_DISABLED;
        in.kvarn_bits = 0;
        in.type_k = c.k;
        in.type_v = c.v;
        llama_kv_mixed_mtp_budget out = {};
        CHECK(llama_kv_mixed_mtp_budget_estimate(in, out) == LLAMA_KV_MIXED_OK);
        // actual ggml row bytes x capacity, no tail for the MTP standard cache
        const uint64_t expected = uint64_t(n_ctx) *
                (uint64_t(ggml_row_size(c.k, 1024)) + uint64_t(ggml_row_size(c.v, 1024)));
        CHECK(out.total_bytes == expected);
        CHECK(out.layer_bytes.size() == 1);
        CHECK(out.layer_bytes[0] == expected);
    }
}

static void test_draft_differs_from_target() {
    // the same layer sized with the DRAFT bits must differ from the TARGET
    // bits (the reason the planner cannot reuse the target type)
    llama_kv_mixed_mtp_input in = qwen_input(204800);
    llama_kv_mixed_mtp_budget kvarn4 = {};
    CHECK(llama_kv_mixed_mtp_budget_estimate(in, kvarn4) == LLAMA_KV_MIXED_OK);
    llama_kv_mixed_mtp_input in6 = in;
    in6.kvarn = LLAMA_KVARN_K6V6_G128;
    in6.kvarn_bits = 0;
    llama_kv_mixed_mtp_budget kvarn6 = {};
    CHECK(llama_kv_mixed_mtp_budget_estimate(in6, kvarn6) == LLAMA_KV_MIXED_OK);
    CHECK(kvarn6.total_bytes > kvarn4.total_bytes);

    // asymmetric K/V bits come from the type (K3V8)
    llama_kv_mixed_mtp_input in38 = in;
    in38.kvarn = LLAMA_KVARN_K3V8_G128;
    in38.kvarn_bits = 0;
    llama_kv_mixed_mtp_budget kvarn38 = {};
    CHECK(llama_kv_mixed_mtp_budget_estimate(in38, kvarn38) == LLAMA_KV_MIXED_OK);
    CHECK(kvarn38.total_bytes > kvarn4.total_bytes);
    CHECK(kvarn38.total_bytes < kvarn6.total_bytes); // 3+8 bits < 6+6 bits

    // K8V8 type directly (no override)
    llama_kv_mixed_mtp_input in8 = in;
    in8.kvarn = LLAMA_KVARN_K8V8_G128;
    in8.kvarn_bits = 0;
    llama_kv_mixed_mtp_budget kvarn8 = {};
    CHECK(llama_kv_mixed_mtp_budget_estimate(in8, kvarn8) == LLAMA_KV_MIXED_OK);
    CHECK(kvarn8.total_bytes > kvarn6.total_bytes);
    // matching packed override is accepted (redundant confirmation)
    in8.kvarn_bits = (uint32_t(8) << 16) | 8;
    llama_kv_mixed_mtp_budget kvarn8b = {};
    CHECK(llama_kv_mixed_mtp_budget_estimate(in8, kvarn8b) == LLAMA_KV_MIXED_OK);
    CHECK(kvarn8b.total_bytes == kvarn8.total_bytes);
    // a packed override that disagrees with the type is not a real width:
    // fail closed instead of hiding an override the engine would not apply
    llama_kv_mixed_mtp_input in_bad = in;
    in_bad.kvarn = LLAMA_KVARN_K2V2_G128;
    in_bad.kvarn_bits = (uint32_t(8) << 16) | 8;
    llama_kv_mixed_mtp_budget out_bad = {};
    CHECK(llama_kv_mixed_mtp_budget_estimate(in_bad, out_bad) == LLAMA_KV_MIXED_ERR_BAD_ARG);
    CHECK(out_bad.error[0] != '\0');
}

static void test_capacity_and_layer_count() {
    // per-token scaling: 32k vs 204800
    llama_kv_mixed_mtp_budget small = {};
    CHECK(llama_kv_mixed_mtp_budget_estimate(qwen_input(32768), small) == LLAMA_KV_MIXED_OK);
    llama_kv_mixed_mtp_budget large = {};
    CHECK(llama_kv_mixed_mtp_budget_estimate(qwen_input(204800), large) == LLAMA_KV_MIXED_OK);
    CHECK(large.total_bytes > small.total_bytes);

    // multiple MTP layers sum up
    llama_kv_mixed_mtp_input multi = qwen_input(32768);
    multi.layers.push_back(qwen_mtp_layer(33));
    multi.layers.push_back(qwen_mtp_layer(34));
    llama_kv_mixed_mtp_budget out = {};
    CHECK(llama_kv_mixed_mtp_budget_estimate(multi, out) == LLAMA_KV_MIXED_OK);
    CHECK(out.layer_bytes.size() == 3);
    CHECK(out.layer_bytes[0] == out.layer_bytes[1]);
    CHECK(out.layer_bytes[1] == out.layer_bytes[2]);
    CHECK(out.total_bytes == out.layer_bytes[0] + out.layer_bytes[1] + out.layer_bytes[2]);
}

static void test_overflow_and_unsupported() {
    llama_kv_mixed_mtp_budget out = {};
    llama_kv_mixed_mtp_input in = qwen_input(204800);

    // zero capacity / no layers
    llama_kv_mixed_mtp_input bad = in;
    bad.n_ctx = 0;
    CHECK(llama_kv_mixed_mtp_budget_estimate(bad, out) == LLAMA_KV_MIXED_ERR_BAD_ARG);
    CHECK(out.error[0] != '\0');
    bad = in;
    bad.layers.clear();
    CHECK(llama_kv_mixed_mtp_budget_estimate(bad, out) == LLAMA_KV_MIXED_ERR_BAD_ARG);

    // invalid kvarn type and invalid bits
    bad = in;
    bad.kvarn = LLAMA_KVARN_TYPE_INVALID;
    CHECK(llama_kv_mixed_mtp_budget_estimate(bad, out) == LLAMA_KV_MIXED_ERR_BAD_ARG);
    bad = in;
    bad.kvarn = LLAMA_KVARN_K4V4_G128;
    bad.kvarn_bits = 1; // invalid packed bits
    CHECK(llama_kv_mixed_mtp_budget_estimate(bad, out) == LLAMA_KV_MIXED_ERR_BAD_ARG);

    // unsupported standard types fail closed (no guessed target type)
    bad = in;
    bad.kvarn = LLAMA_KVARN_TYPE_DISABLED;
    bad.type_k = GGML_TYPE_Q4_K; // not a cache-facing type
    bad.type_v = GGML_TYPE_F16;
    CHECK(llama_kv_mixed_mtp_budget_estimate(bad, out) == LLAMA_KV_MIXED_ERR_BAD_ARG);
    bad = in;
    bad.kvarn = LLAMA_KVARN_TYPE_DISABLED;
    bad.type_k = GGML_TYPE_F32; // unsupported standard K type
    bad.type_v = GGML_TYPE_F16;
    CHECK(llama_kv_mixed_mtp_budget_estimate(bad, out) == LLAMA_KV_MIXED_ERR_BAD_ARG);

    // unsupported KVarN geometry fails closed (96 is outside the canonical
    // 64/128/256/512 geometry; rows are kept consistent so the geometry check
    // fires rather than the row-width check)
    bad = in;
    bad.layers[0].head_dim_k = 96;
    bad.layers[0].head_dim_v = 96;
    bad.layers[0].n_embd_k_gqa = 384;
    bad.layers[0].n_embd_v_gqa = 384;
    CHECK(llama_kv_mixed_mtp_budget_estimate(bad, out) == LLAMA_KV_MIXED_ERR_GEOMETRY);

    // geometry/row mismatch
    bad = in;
    bad.layers[0].n_embd_k_gqa = 512; // head_dim_k * n_head_kv = 1024
    CHECK(llama_kv_mixed_mtp_budget_estimate(bad, out) == LLAMA_KV_MIXED_ERR_BAD_ARG);

    // unsupported tail type
    bad = in;
    bad.tail_type = GGML_TYPE_F32;
    CHECK(llama_kv_mixed_mtp_budget_estimate(bad, out) == LLAMA_KV_MIXED_ERR_BAD_ARG);
}

int main() {
    test_kvarn4_active();
    test_standard_types_actual_bytes();
    test_draft_differs_from_target();
    test_capacity_and_layer_count();
    test_overflow_and_unsupported();

    std::printf("%s: %d checks, %d failures\n", g_failures == 0 ? "PASS" : "FAIL", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}