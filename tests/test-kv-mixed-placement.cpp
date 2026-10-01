// Standalone CPU tests for src/llama-kv-mixed-placement.{h,cpp}.
//
// Build (from the repository root, no CMake; links the immutable baseline
// libraries):
//
//   g++ -std=c++17 -O2 -Wall -Wextra
//       -I ggml/include -I include -I src
//       tests/test-kv-mixed-placement.cpp src/llama-kv-mixed-placement.cpp
//       -o <out>/test-kv-mixed-placement
//       -L <baseline-bin> -Wl,-rpath,<baseline-bin> -lllama -lggml-base
//
// Reference profiles (Qwen D256 hkv4, 1024-wide K/V rows, tail0, F16 tail,
// canonical rollback 1, non-unified single stream):
//   - ctx 204800, KVarN4 local per layer = 231,477,248 B = records
//     229,376,000 + stage 1,572,864 + exact tail 528,384 (128 + 1 rollback
//     slots x 4096 B/slot); remote Qx totals for 16 layers are
//     3600/4400/5200/6800 MiB for Q4_0/Q5_0/Q6_0/Q8_0.
//   - ctx 16384, same profile: records+stage 304 MiB + overlay 8.0625 MiB =
//     312.0625 MiB total for 16 layers (baseline KVarN measurement).
//
// The contract under test: zero-initialized budgets are steady state (no
// staging); the full-prefix staging mode is an explicit opt-in flag; derived
// staging models a CPU converter over F32 host chunks, one remote layer at a
// time (never summed over N); the RAM cap includes reserve + KV + staging
// counted once; the KVarN exact tail (intrinsic + rollback) never leaks into
// the remote Qx side.

#include "llama-kv-mixed-placement.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
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

#define CHECK_EQ(a, b) do { \
    ++g_checks; \
    const auto va = (a); \
    const auto vb = (b); \
    if (!(va == vb)) { \
        ++g_failures; \
        std::printf("  FAIL [%s] line %d: %s == %s  (%" PRIu64 " vs %" PRIu64 ")\n", \
                g_current, __LINE__, #a, #b, uint64_t(va), uint64_t(vb)); \
    } \
} while (0)

static constexpr uint64_t MIB = 1024ull * 1024ull;

// Qwen D256 hkv4 canonical constants (tail0, F16 tail, rollback 1).
static constexpr uint64_t QW_RECORDS = 229376000ull;
static constexpr uint64_t QW_STAGE   = 1572864ull;
static constexpr uint64_t QW_TAIL    = 528384ull;   // (2048+2048) * (128+1)
static constexpr uint64_t QW_LOCAL   = QW_RECORDS + QW_STAGE + QW_TAIL; // 231,477,248
static constexpr uint64_t QW_REMOTE  = 235929600ull; // Q4_0, 225 MiB

static llama_kv_mixed_layer_params qwen_layer(uint32_t il, ggml_type qx_k = GGML_TYPE_Q4_0, ggml_type qx_v = GGML_TYPE_Q4_0) {
    llama_kv_mixed_layer_params p = {};
    p.layer = il;
    p.head_dim_k = 256;
    p.head_dim_v = 256;
    p.n_head_kv = 4;
    p.n_embd_k_gqa = 1024;
    p.n_embd_v_gqa = 1024;
    p.kvarn_bits_k = 4;
    p.kvarn_bits_v = 4;
    p.qx_type_k = qx_k;
    p.qx_type_v = qx_v;
    return p;
}

static llama_kv_mixed_sizing qwen_sizing(uint64_t capacity = 204800) {
    llama_kv_mixed_sizing s = {};
    s.capacity_tokens = capacity;
    s.n_seq_max = 1;
    s.kv_unified = false;
    s.stage_tail_groups = 0;    // -> canonical 2
    s.stage_reserve_groups = 0; // -> 1
    s.tail_exact_tokens = 0;    // -> intrinsic 128
    s.tail_rollback_tokens = 0; // -> 1
    s.tail_type = GGML_TYPE_F16;
    s.remote_tail_tokens = 0;
    return s;
}

static std::vector<llama_kv_mixed_layer_cost> qwen_costs(size_t n, ggml_type qx_k = GGML_TYPE_Q4_0, ggml_type qx_v = GGML_TYPE_Q4_0) {
    char err[256] = {};
    std::vector<llama_kv_mixed_layer_cost> costs;
    for (size_t i = 0; i < n; ++i) {
        llama_kv_mixed_layer_cost c = {};
        const llama_kv_mixed_status st = llama_kv_mixed_estimate_layer_cost(
                qwen_layer(uint32_t(i), qx_k, qx_v), qwen_sizing(), c, err, sizeof(err));
        CHECK(st == LLAMA_KV_MIXED_OK);
        costs.push_back(c);
    }
    return costs;
}

static void test_reference_qwen() {
    g_current = "reference_qwen";
    for (const auto & pair : {
            std::make_pair(GGML_TYPE_Q4_0, 3600ull),
            std::make_pair(GGML_TYPE_Q5_0, 4400ull),
            std::make_pair(GGML_TYPE_Q6_0, 5200ull),
            std::make_pair(GGML_TYPE_Q8_0, 6800ull) }) {
        const auto costs = qwen_costs(16, pair.first, pair.first);
        uint64_t total_remote = 0;
        uint64_t total_local = 0;
        for (const auto & c : costs) {
            total_remote += c.remote_bytes;
            total_local += c.local_bytes;
            CHECK_EQ(c.local_bytes, QW_LOCAL);
        }
        const uint64_t expected = pair.second * MIB;
        CHECK_EQ(total_remote, expected);
        std::printf("  Qwen16 Q%-s total remote = %" PRIu64 " B = %.0f MiB (expected %" PRIu64 " MiB), local/layer = %" PRIu64 " B\n",
                ggml_type_name(pair.first), total_remote,
                double(total_remote) / double(MIB), expected / MIB, QW_LOCAL);
        CHECK_EQ(total_local, QW_LOCAL * 16);
    }

    // ctx 16384 baseline: records+stage 304 MiB, overlay (exact tail) 8.0625 MiB.
    const auto costs16 = qwen_costs(16);
    llama_kv_mixed_sizing s16 = qwen_sizing(16384);
    char err[256] = {};
    uint64_t total_local = 0;
    for (uint32_t i = 0; i < 16; ++i) {
        llama_kv_mixed_layer_cost c = {};
        CHECK(llama_kv_mixed_estimate_layer_cost(qwen_layer(i), s16, c, err, sizeof(err)) == LLAMA_KV_MIXED_OK);
        total_local += c.local_bytes;
    }
    CHECK_EQ(total_local, 327221248ull); // 312.0625 MiB
    CHECK_EQ(total_local, 318767104ull + 8454144ull); // records+stage 304 MiB + overlay 8.0625 MiB
    (void) costs16;
}

static void test_estimator_details() {
    g_current = "estimator_details";
    char err[256] = {};
    llama_kv_mixed_layer_cost c = {};
    CHECK(llama_kv_mixed_estimate_layer_cost(qwen_layer(0), qwen_sizing(), c, err, sizeof(err)) == LLAMA_KV_MIXED_OK);
    CHECK_EQ(c.local_bytes, QW_RECORDS + QW_STAGE + QW_TAIL); // rollback slot captured
    CHECK_EQ(c.remote_bytes, QW_REMOTE);                      // Qx never inherits the KVarN tail
    CHECK_EQ(c.local_per_token, 1120ull);
    CHECK_EQ(c.remote_per_token, 1152ull);
    CHECK_EQ(c.convert_scratch_per_token, 9344ull); // (1024+1024)*4 + 1152

    // K and V independent: only V in Q8_0 changes the remote rows.
    llama_kv_mixed_layer_cost c2 = {};
    CHECK(llama_kv_mixed_estimate_layer_cost(
            qwen_layer(0, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0), qwen_sizing(), c2, err, sizeof(err)) == LLAMA_KV_MIXED_OK);
    CHECK_EQ(c2.remote_bytes, (576ull + 1088ull) * 204800ull);
    CHECK_EQ(c2.remote_per_token, 576ull + 1088ull);

    // Resolved explicit tail: exact 1024 keeps the canonical rollback (1)
    // because the rollback sentinel 0 resolves to 1 whenever the exact tail
    // is non-zero. Local grows by 4096*(1024+1); the remote side is untouched.
    llama_kv_mixed_sizing s = qwen_sizing();
    s.tail_exact_tokens = 1024;
    llama_kv_mixed_layer_cost c3 = {};
    CHECK(llama_kv_mixed_estimate_layer_cost(qwen_layer(0), s, c3, err, sizeof(err)) == LLAMA_KV_MIXED_OK);
    CHECK_EQ(c3.local_bytes, QW_RECORDS + QW_STAGE + 4096ull * 1025ull);
    CHECK_EQ(c3.remote_bytes, QW_REMOTE);

    // Explicit rollback 2 adds another slot: 4096 * (1024 + 2).
    s.tail_rollback_tokens = 2;
    llama_kv_mixed_layer_cost c4 = {};
    CHECK(llama_kv_mixed_estimate_layer_cost(qwen_layer(0), s, c4, err, sizeof(err)) == LLAMA_KV_MIXED_OK);
    CHECK_EQ(c4.local_bytes, QW_RECORDS + QW_STAGE + 4096ull * 1026ull);

    // Explicit remote tail is a separate input.
    s.tail_exact_tokens = 128;
    s.tail_rollback_tokens = 1;
    s.remote_tail_tokens = 1024;
    llama_kv_mixed_layer_cost c5 = {};
    CHECK(llama_kv_mixed_estimate_layer_cost(qwen_layer(0), s, c5, err, sizeof(err)) == LLAMA_KV_MIXED_OK);
    CHECK_EQ(c5.remote_bytes, QW_REMOTE + 4096ull * 1024ull);
    CHECK_EQ(c5.local_bytes, QW_LOCAL);
}

static void test_n0_and_steady_state_default() {
    g_current = "n0_and_steady_state_default";
    const auto costs = qwen_costs(16);

    // Zero-initialized budget with room for everything: N = 0, no staging.
    llama_kv_mixed_budget b = {};
    b.cuda_free_bytes = 16ull * QW_LOCAL + 64ull * MIB;
    b.vulkan_free_bytes = 0;
    llama_kv_mixed_placement p = {};
    CHECK(llama_kv_mixed_choose_placement(costs, qwen_sizing(), b, p) == LLAMA_KV_MIXED_OK);
    CHECK_EQ(p.n_remote, 0u);
    CHECK_EQ(p.n_local, 16u);
    CHECK_EQ(p.remote_kv_bytes, 0ull);
    CHECK_EQ(p.vulkan_used_bytes, 0ull);
    CHECK_EQ(p.local_kv_bytes, 16ull * QW_LOCAL);
    CHECK_EQ(p.cuda_used_bytes, 16ull * QW_LOCAL);
    CHECK_EQ(p.handoff_temp_bytes, 0ull);
    CHECK_EQ(p.ram_used_bytes, 0ull);

    // Zero-initialized budget that needs one remote layer: still no staging.
    // handoff_chunk_tokens == 0 means steady state, NOT full-prefix staging.
    llama_kv_mixed_budget b2 = {};
    b2.cuda_free_bytes = 15ull * QW_LOCAL + 64ull * MIB;
    b2.vulkan_free_bytes = 2ull * QW_REMOTE;
    llama_kv_mixed_placement p2 = {};
    CHECK(llama_kv_mixed_choose_placement(costs, qwen_sizing(), b2, p2) == LLAMA_KV_MIXED_OK);
    CHECK_EQ(p2.n_remote, 1u);
    CHECK_EQ(p2.cuda_used_bytes, 15ull * QW_LOCAL);
    CHECK_EQ(p2.handoff_temp_bytes, 0ull);
    CHECK_EQ(p2.cuda_temp_bytes, 0ull);
    CHECK_EQ(p2.vulkan_temp_bytes, 0ull);
    CHECK_EQ(p2.ram_temp_bytes, 0ull);
    CHECK_EQ(p2.effective_chunk_tokens, 0ull);
}

static void test_n_all_and_no_layer_sum() {
    g_current = "n_all_and_no_layer_sum";
    const auto costs = qwen_costs(16);
    llama_kv_mixed_budget b = {};
    b.cuda_free_bytes = 0;
    b.vulkan_free_bytes = 16ull * QW_REMOTE;
    llama_kv_mixed_placement p = {};
    CHECK(llama_kv_mixed_choose_placement(costs, qwen_sizing(), b, p) == LLAMA_KV_MIXED_OK);
    CHECK_EQ(p.n_remote, 16u);
    CHECK_EQ(p.n_local, 0u);
    CHECK_EQ(p.local_kv_bytes, 0ull);
    CHECK_EQ(p.remote_kv_bytes, 16ull * QW_REMOTE);
    CHECK_EQ(p.cuda_used_bytes, 0ull);
    CHECK_EQ(p.handoff_temp_bytes, 0ull);

    // Chunked staging is per LAYER: 16 remote layers stage the same amount as
    // one remote layer (128 tokens * 9344 B).
    llama_kv_mixed_budget b2 = {};
    b2.cuda_free_bytes = 0;
    b2.vulkan_free_bytes = 16ull * QW_REMOTE;
    b2.handoff_chunk_tokens = 128;
    llama_kv_mixed_placement p2 = {};
    CHECK(llama_kv_mixed_choose_placement(costs, qwen_sizing(), b2, p2) == LLAMA_KV_MIXED_OK);
    CHECK_EQ(p2.n_remote, 16u);
    CHECK_EQ(p2.ram_temp_bytes, 128ull * 9344ull); // not x16
    CHECK_EQ(p2.effective_chunk_tokens, 128ull);
    CHECK_EQ(p2.handoff_temp_bytes, 128ull * 9344ull);
    CHECK(!p2.temp_capped);

    llama_kv_mixed_budget b3 = {};
    b3.cuda_free_bytes = 15ull * QW_LOCAL + 64ull * MIB;
    b3.vulkan_free_bytes = 2ull * QW_REMOTE;
    b3.handoff_chunk_tokens = 128;
    llama_kv_mixed_placement p3 = {};
    CHECK(llama_kv_mixed_choose_placement(costs, qwen_sizing(), b3, p3) == LLAMA_KV_MIXED_OK);
    CHECK_EQ(p3.n_remote, 1u);
    CHECK_EQ(p3.ram_temp_bytes, 128ull * 9344ull); // identical to the N=16 case
}

static void test_cuda_insufficient() {
    g_current = "cuda_insufficient";
    const auto costs = qwen_costs(16);

    // Explicit CUDA staging larger than any feasible suffix + staging.
    llama_kv_mixed_budget b = {};
    b.cuda_free_bytes = 15ull * QW_LOCAL + 64ull * MIB;
    b.vulkan_free_bytes = 16ull * QW_REMOTE;
    b.cuda_temp_bytes = 10ull * 1024ull * MIB;
    llama_kv_mixed_placement p = {};
    CHECK(llama_kv_mixed_choose_placement(costs, qwen_sizing(), b, p) == LLAMA_KV_MIXED_ERR_NO_PLACEMENT);
    CHECK(std::strlen(p.error) > 0);

    // Tiny CUDA AND tiny Vulkan: no N works.
    llama_kv_mixed_budget b2 = {};
    b2.cuda_free_bytes = 100ull * MIB;
    b2.vulkan_free_bytes = 200000000ull; // below one Q4 layer
    llama_kv_mixed_placement p2 = {};
    CHECK(llama_kv_mixed_choose_placement(costs, qwen_sizing(), b2, p2) == LLAMA_KV_MIXED_ERR_NO_PLACEMENT);
    CHECK(std::strlen(p2.error) > 0);
}

static void test_vulkan_insufficient() {
    g_current = "vulkan_insufficient";
    const auto costs = qwen_costs(16);
    llama_kv_mixed_budget b = {};
    b.cuda_free_bytes = 15ull * QW_LOCAL + 64ull * MIB;
    b.vulkan_free_bytes = 200000000ull;
    llama_kv_mixed_placement p = {};
    CHECK(llama_kv_mixed_choose_placement(costs, qwen_sizing(), b, p) == LLAMA_KV_MIXED_ERR_NO_PLACEMENT);
    CHECK(std::strlen(p.error) > 0);

    b.vulkan_free_bytes = 2ull * QW_REMOTE;
    llama_kv_mixed_placement p2 = {};
    CHECK(llama_kv_mixed_choose_placement(costs, qwen_sizing(), b, p2) == LLAMA_KV_MIXED_OK);
    CHECK_EQ(p2.n_remote, 1u);
    CHECK_EQ(p2.remote_kv_bytes, QW_REMOTE);
    CHECK(p2.vulkan_used_bytes <= p2.vulkan_available_bytes);
    CHECK_EQ(p2.vulkan_used_bytes, p2.remote_kv_bytes);

    // Explicit remote staging pushes the single-layer case over the budget.
    llama_kv_mixed_budget b3 = {};
    b3.cuda_free_bytes = 15ull * QW_LOCAL + 64ull * MIB;
    b3.vulkan_free_bytes = 283115520ull; // 1.2 layers
    b3.vulkan_temp_bytes = 104857600ull; // 100 MiB
    llama_kv_mixed_placement p3 = {};
    CHECK(llama_kv_mixed_choose_placement(costs, qwen_sizing(), b3, p3) == LLAMA_KV_MIXED_ERR_NO_PLACEMENT);
    CHECK(std::strlen(p3.error) > 0);
}

static void test_reserve_exceeds_free() {
    g_current = "reserve_exceeds_free";
    const auto costs = qwen_costs(16);

    llama_kv_mixed_budget b = {};
    b.cuda_free_bytes = 16ull * QW_LOCAL + 64ull * MIB;
    b.cuda_reserve_bytes = b.cuda_free_bytes + 1;
    b.vulkan_free_bytes = 16ull * QW_REMOTE;
    llama_kv_mixed_placement p = {};
    CHECK(llama_kv_mixed_choose_placement(costs, qwen_sizing(), b, p) == LLAMA_KV_MIXED_ERR_RESERVE);
    CHECK(std::strlen(p.error) > 0);

    // Vulkan reserve exceeds free, but N = 0 works: remote workspace ignored.
    llama_kv_mixed_budget b2 = {};
    b2.cuda_free_bytes = 16ull * QW_LOCAL + 64ull * MIB;
    b2.vulkan_free_bytes = 1024ull * MIB;
    b2.vulkan_reserve_bytes = 2048ull * MIB;
    llama_kv_mixed_placement p2 = {};
    CHECK(llama_kv_mixed_choose_placement(costs, qwen_sizing(), b2, p2) == LLAMA_KV_MIXED_OK);
    CHECK_EQ(p2.n_remote, 0u);

    // Vulkan reserve exceeds free while CUDA needs at least one remote layer.
    llama_kv_mixed_budget b3 = {};
    b3.cuda_free_bytes = 15ull * QW_LOCAL + 64ull * MIB;
    b3.vulkan_free_bytes = 1024ull * MIB;
    b3.vulkan_reserve_bytes = 2048ull * MIB;
    llama_kv_mixed_placement p3 = {};
    CHECK(llama_kv_mixed_choose_placement(costs, qwen_sizing(), b3, p3) == LLAMA_KV_MIXED_ERR_RESERVE);
    CHECK(std::strlen(p3.error) > 0);
}

static void test_trailing_diff_geometry() {
    g_current = "trailing_diff_geometry";
    // 15 layers D256 hkv4 (1024 rows) + trailing layer D512 hkv4 (2048 rows).
    std::vector<llama_kv_mixed_layer_cost> costs;
    char err[256] = {};
    for (uint32_t i = 0; i < 15; ++i) {
        llama_kv_mixed_layer_cost c = {};
        CHECK(llama_kv_mixed_estimate_layer_cost(qwen_layer(i), qwen_sizing(), c, err, sizeof(err)) == LLAMA_KV_MIXED_OK);
        costs.push_back(c);
    }
    llama_kv_mixed_layer_params heavy = qwen_layer(15);
    heavy.head_dim_k = 512;
    heavy.head_dim_v = 512;
    heavy.n_embd_k_gqa = 2048;
    heavy.n_embd_v_gqa = 2048;
    llama_kv_mixed_layer_cost ch = {};
    CHECK(llama_kv_mixed_estimate_layer_cost(heavy, qwen_sizing(), ch, err, sizeof(err)) == LLAMA_KV_MIXED_OK);
    // records 458,752,000 + stage 3,145,728 + tail (4096+4096)*129 = 1,056,768
    CHECK_EQ(ch.local_bytes, 458752000ull + 3145728ull + 1056768ull);
    CHECK_EQ(ch.remote_bytes, 2304ull * 204800ull);
    costs.push_back(ch);

    // CUDA 3.5 GiB fits 13 light + heavy locally (N=2); the light-only control
    // moves only one layer: the heavier trailing layer forces one more remote.
    llama_kv_mixed_budget b = {};
    b.cuda_free_bytes = 3500000000ull;
    b.vulkan_free_bytes = 943718400ull; // exactly four Q4 layers
    llama_kv_mixed_placement p = {};
    CHECK(llama_kv_mixed_choose_placement(costs, qwen_sizing(), b, p) == LLAMA_KV_MIXED_OK);
    CHECK_EQ(p.n_remote, 2u);
    CHECK_EQ(p.n_local, 14u);
    CHECK_EQ(p.remote_kv_bytes, 2ull * QW_REMOTE);
    CHECK_EQ(p.local_kv_bytes, 13ull * QW_LOCAL + 462954496ull);
    CHECK(p.vulkan_used_bytes <= p.vulkan_available_bytes);

    const auto light = qwen_costs(16);
    llama_kv_mixed_placement pc = {};
    CHECK(llama_kv_mixed_choose_placement(light, qwen_sizing(), b, pc) == LLAMA_KV_MIXED_OK);
    CHECK_EQ(pc.n_remote, 1u);
}

static void test_exact_limits() {
    g_current = "exact_limits";
    const auto costs = qwen_costs(16);
    llama_kv_mixed_budget b = {};
    b.cuda_free_bytes = 16ull * QW_LOCAL;
    b.vulkan_free_bytes = 16ull * QW_REMOTE;
    llama_kv_mixed_placement p = {};
    CHECK(llama_kv_mixed_choose_placement(costs, qwen_sizing(), b, p) == LLAMA_KV_MIXED_OK);
    CHECK_EQ(p.n_remote, 0u);
    CHECK_EQ(p.cuda_used_bytes, p.cuda_available_bytes);
}

static void test_overflow_and_validation() {
    g_current = "overflow_and_validation";
    char err[256] = {};
    llama_kv_mixed_sizing s = qwen_sizing();
    s.capacity_tokens = UINT64_MAX;
    llama_kv_mixed_layer_cost c = {};
    CHECK(llama_kv_mixed_estimate_layer_cost(qwen_layer(0), s, c, err, sizeof(err)) == LLAMA_KV_MIXED_ERR_OVERFLOW);
    CHECK(std::strlen(err) > 0);

    // A large but realistic capacity still works and scales linearly.
    s.capacity_tokens = 1000000000ull;
    CHECK(llama_kv_mixed_estimate_layer_cost(qwen_layer(0), s, c, err, sizeof(err)) == LLAMA_KV_MIXED_OK);
    CHECK_EQ(c.remote_bytes, 1152ull * 1000000000ull);
    CHECK_EQ(c.remote_per_token, 1152ull);

    // Row width disagreement with the geometry.
    llama_kv_mixed_layer_params p = qwen_layer(0);
    p.n_embd_k_gqa = 512; // head_dim 256 * n_head_kv 4 = 1024
    CHECK(llama_kv_mixed_estimate_layer_cost(p, qwen_sizing(), c, err, sizeof(err)) == LLAMA_KV_MIXED_ERR_BAD_ARG);
    p = qwen_layer(0);
    p.n_embd_v_gqa = 2048;
    CHECK(llama_kv_mixed_estimate_layer_cost(p, qwen_sizing(), c, err, sizeof(err)) == LLAMA_KV_MIXED_ERR_BAD_ARG);

    // Qx whitelist: IQ/K variants and COUNT are rejected before ggml asserts.
    p = qwen_layer(0);
    p.qx_type_k = GGML_TYPE_Q4_K;
    CHECK(llama_kv_mixed_estimate_layer_cost(p, qwen_sizing(), c, err, sizeof(err)) == LLAMA_KV_MIXED_ERR_BAD_ARG);
    p = qwen_layer(0);
    p.qx_type_v = GGML_TYPE_COUNT;
    CHECK(llama_kv_mixed_estimate_layer_cost(p, qwen_sizing(), c, err, sizeof(err)) == LLAMA_KV_MIXED_ERR_BAD_ARG);
    CHECK(!llama_kv_mixed_qx_type_allowed(GGML_TYPE_Q4_K));
    CHECK(llama_kv_mixed_qx_type_allowed(GGML_TYPE_Q4_0));
    CHECK(llama_kv_mixed_qx_type_allowed(GGML_TYPE_Q6_0));

    // Tail type whitelist: only F16/BF16.
    s = qwen_sizing();
    s.tail_type = GGML_TYPE_F32;
    CHECK(llama_kv_mixed_estimate_layer_cost(qwen_layer(0), s, c, err, sizeof(err)) == LLAMA_KV_MIXED_ERR_BAD_ARG);
    s = qwen_sizing();
    s.tail_type = GGML_TYPE_BF16; // whitelisted and sized
    CHECK(llama_kv_mixed_estimate_layer_cost(qwen_layer(0), s, c, err, sizeof(err)) == LLAMA_KV_MIXED_OK);
    CHECK_EQ(c.local_bytes, QW_RECORDS + QW_STAGE + 528384ull);

    // Other bad arguments.
    s = qwen_sizing();
    s.capacity_tokens = 0;
    CHECK(llama_kv_mixed_estimate_layer_cost(qwen_layer(0), s, c, err, sizeof(err)) == LLAMA_KV_MIXED_ERR_BAD_ARG);
    s = qwen_sizing();
    s.n_seq_max = 0;
    CHECK(llama_kv_mixed_estimate_layer_cost(qwen_layer(0), s, c, err, sizeof(err)) == LLAMA_KV_MIXED_ERR_BAD_ARG);
    p = qwen_layer(0);
    p.n_head_kv = 0;
    CHECK(llama_kv_mixed_estimate_layer_cost(p, qwen_sizing(), c, err, sizeof(err)) == LLAMA_KV_MIXED_ERR_BAD_ARG);
    p = qwen_layer(0);
    p.kvarn_bits_k = 7;
    CHECK(llama_kv_mixed_estimate_layer_cost(p, qwen_sizing(), c, err, sizeof(err)) == LLAMA_KV_MIXED_ERR_BAD_ARG);
    p = qwen_layer(0);
    p.head_dim_k = 96; // consistent row width, but no canonical KVarN geometry
    p.head_dim_v = 96;
    p.n_embd_k_gqa = 384;
    p.n_embd_v_gqa = 384;
    CHECK(llama_kv_mixed_estimate_layer_cost(p, qwen_sizing(), c, err, sizeof(err)) == LLAMA_KV_MIXED_ERR_GEOMETRY);
}

static void test_unified_streams() {
    g_current = "unified_streams";
    char err[256] = {};
    // Unified cache, two sequences: records/stage scale with one stream while
    // the exact tail slots scale with n_seq_max.
    llama_kv_mixed_sizing su = qwen_sizing();
    su.n_seq_max = 2;
    su.kv_unified = true;
    llama_kv_mixed_layer_cost cu = {};
    CHECK(llama_kv_mixed_estimate_layer_cost(qwen_layer(0), su, cu, err, sizeof(err)) == LLAMA_KV_MIXED_OK);
    CHECK_EQ(cu.local_bytes, QW_RECORDS + 2048ull * 2ull * 128ull * 5ull + 4096ull * 129ull * 2ull);
    CHECK_EQ(cu.local_per_token, 1120ull); // one stream
    CHECK_EQ(cu.remote_per_token, 1152ull);

    // Non-unified cache, two sequences: everything scales with the streams.
    llama_kv_mixed_sizing sn = qwen_sizing();
    sn.n_seq_max = 2;
    sn.kv_unified = false;
    llama_kv_mixed_layer_cost cn = {};
    CHECK(llama_kv_mixed_estimate_layer_cost(qwen_layer(0), sn, cn, err, sizeof(err)) == LLAMA_KV_MIXED_OK);
    CHECK_EQ(cn.local_bytes, QW_RECORDS * 2 + 2048ull * 2ull * 128ull * 3ull * 2ull + 4096ull * 129ull * 2ull);
    CHECK_EQ(cn.local_per_token, 2240ull);
    CHECK_EQ(cn.remote_per_token, 2304ull);
}

static void test_monotonicity() {
    g_current = "monotonicity";
    const auto costs = qwen_costs(16);
    const uint64_t total_local = 16ull * QW_LOCAL;

    // Growing CUDA budget must never increase the chosen N.
    int prev_n = -1;
    bool ever_ok = false;
    for (uint64_t avail = 0; avail <= total_local + 512ull * MIB; avail += 123456789ull) {
        llama_kv_mixed_budget b = {};
        b.cuda_free_bytes = avail;
        b.vulkan_free_bytes = 16ull * QW_REMOTE + 1024ull * MIB;
        llama_kv_mixed_placement p = {};
        const llama_kv_mixed_status st = llama_kv_mixed_choose_placement(costs, qwen_sizing(), b, p);
        if (st == LLAMA_KV_MIXED_OK) {
            if (prev_n >= 0) {
                CHECK(int(p.n_remote) <= prev_n);
            }
            prev_n = int(p.n_remote);
            ever_ok = true;
        }
    }
    CHECK(ever_ok);

    // Growing Vulkan budget must never increase the chosen N either.
    prev_n = -1;
    ever_ok = false;
    for (uint64_t avail = 0; avail <= 16ull * QW_REMOTE + 512ull * MIB; avail += 123456789ull) {
        llama_kv_mixed_budget b = {};
        b.cuda_free_bytes = 15ull * QW_LOCAL + 64ull * MIB; // needs at least one remote layer
        b.vulkan_free_bytes = avail;
        llama_kv_mixed_placement p = {};
        const llama_kv_mixed_status st = llama_kv_mixed_choose_placement(costs, qwen_sizing(), b, p);
        if (st == LLAMA_KV_MIXED_OK) {
            if (prev_n >= 0) {
                CHECK(int(p.n_remote) <= prev_n);
            }
            prev_n = int(p.n_remote);
            ever_ok = true;
        }
    }
    CHECK(ever_ok);
}

static void test_chunk_conversion() {
    g_current = "chunk_conversion";
    const auto costs = qwen_costs(16);
    llama_kv_mixed_budget b = {};
    b.cuda_free_bytes = 3500000000ull; // fits suffix(1) = 3,472,158,720
    b.vulkan_free_bytes = 16ull * QW_REMOTE;

    // Steady state default: no staging at all, N = 1.
    llama_kv_mixed_placement p = {};
    CHECK(llama_kv_mixed_choose_placement(costs, qwen_sizing(), b, p) == LLAMA_KV_MIXED_OK);
    CHECK_EQ(p.n_remote, 1u);
    CHECK_EQ(p.handoff_temp_bytes, 0ull);
    CHECK_EQ(p.effective_chunk_tokens, 0ull);

    // 1024-token chunk: host staging 1024 * 9344, still per single layer.
    b.handoff_chunk_tokens = 1024;
    llama_kv_mixed_placement p2 = {};
    CHECK(llama_kv_mixed_choose_placement(costs, qwen_sizing(), b, p2) == LLAMA_KV_MIXED_OK);
    CHECK_EQ(p2.n_remote, 1u);
    CHECK_EQ(p2.ram_temp_bytes, 1024ull * 9344ull);
    CHECK_EQ(p2.handoff_temp_bytes, 1024ull * 9344ull);
    CHECK_EQ(p2.effective_chunk_tokens, 1024ull);
    CHECK(!p2.temp_capped);

    // Full-prefix staging is an explicit opt-in flag, not a zero default.
    llama_kv_mixed_budget b3 = b;
    b3.handoff_chunk_tokens = 0;
    b3.handoff_full_prefix_staging = true;
    llama_kv_mixed_placement p3 = {};
    CHECK(llama_kv_mixed_choose_placement(costs, qwen_sizing(), b3, p3) == LLAMA_KV_MIXED_OK);
    CHECK_EQ(p3.ram_temp_bytes, 204800ull * 9344ull);
    CHECK_EQ(p3.effective_chunk_tokens, 0ull); // full-prefix marker

    // Cap reduces the chunk to whole 128-token groups.
    llama_kv_mixed_budget b4 = b;
    b4.handoff_chunk_tokens = 204800;
    b4.handoff_temp_max_bytes = 20000000ull;
    llama_kv_mixed_placement p4 = {};
    CHECK(llama_kv_mixed_choose_placement(costs, qwen_sizing(), b4, p4) == LLAMA_KV_MIXED_OK);
    CHECK_EQ(p4.n_remote, 1u);
    CHECK(p4.temp_capped);
    CHECK(p4.ram_temp_bytes <= 20000000ull);
    CHECK_EQ(p4.effective_chunk_tokens % 128ull, 0ull);
    CHECK_EQ(p4.effective_chunk_tokens, 2048ull); // 16 groups
    CHECK_EQ(p4.ram_temp_bytes, 2048ull * 9344ull);

    // A cap below one whole group is a hard failure.
    b4.handoff_temp_max_bytes = 128ull * 9344ull - 1;
    llama_kv_mixed_placement p5 = {};
    CHECK(llama_kv_mixed_choose_placement(costs, qwen_sizing(), b4, p5) == LLAMA_KV_MIXED_ERR_TEMP);
    CHECK(std::strlen(p5.error) > 0);

    // Declared concurrency scales the host staging.
    llama_kv_mixed_budget b6 = b;
    b6.handoff_chunk_tokens = 128;
    b6.handoff_concurrency = 4;
    llama_kv_mixed_placement p6 = {};
    CHECK(llama_kv_mixed_choose_placement(costs, qwen_sizing(), b6, p6) == LLAMA_KV_MIXED_OK);
    CHECK_EQ(p6.ram_temp_bytes, 4ull * 128ull * 9344ull);

    // The RAM cap sees the staging: reserve + KV + staging, counted once.
    llama_kv_mixed_budget b7 = {};
    b7.cuda_free_bytes = 3500000000ull;
    b7.vulkan_free_bytes = 16ull * QW_REMOTE;
    b7.handoff_chunk_tokens = 204800;
    b7.handoff_temp_max_bytes = 20000000ull;
    b7.ram_cap_bytes = 25000000ull;
    llama_kv_mixed_placement p7 = {};
    CHECK(llama_kv_mixed_choose_placement(costs, qwen_sizing(), b7, p7) == LLAMA_KV_MIXED_ERR_NO_PLACEMENT);
    CHECK(std::strstr(p7.error, "RAM cap") != nullptr);
    b7.ram_cap_bytes = 260000000ull;
    llama_kv_mixed_placement p8 = {};
    CHECK(llama_kv_mixed_choose_placement(costs, qwen_sizing(), b7, p8) == LLAMA_KV_MIXED_OK);
    CHECK_EQ(p8.n_remote, 1u);
    CHECK_EQ(p8.ram_used_bytes, QW_REMOTE + 19136512ull);
}

static void test_explicit_temps() {
    g_current = "explicit_temps";
    const auto costs = qwen_costs(16);
    llama_kv_mixed_budget b = {};
    b.cuda_free_bytes = 15ull * QW_LOCAL + 64ull * MIB;
    b.vulkan_free_bytes = 2ull * QW_REMOTE;
    b.cuda_temp_bytes = 12345678ull;
    llama_kv_mixed_placement p = {};
    CHECK(llama_kv_mixed_choose_placement(costs, qwen_sizing(), b, p) == LLAMA_KV_MIXED_OK);
    CHECK_EQ(p.n_remote, 1u);
    CHECK_EQ(p.cuda_temp_bytes, 12345678ull);
    CHECK_EQ(p.cuda_used_bytes, 15ull * QW_LOCAL + 12345678ull);
    CHECK_EQ(p.handoff_temp_bytes, 12345678ull);

    // Explicit staging above the cap is a hard error (never reduced).
    b.handoff_temp_max_bytes = 10000000ull;
    llama_kv_mixed_placement p2 = {};
    CHECK(llama_kv_mixed_choose_placement(costs, qwen_sizing(), b, p2) == LLAMA_KV_MIXED_ERR_TEMP);
    CHECK(std::strlen(p2.error) > 0);

    // Explicit host staging counts against the RAM cap.
    llama_kv_mixed_budget b3 = {};
    b3.cuda_free_bytes = 15ull * QW_LOCAL + 64ull * MIB;
    b3.vulkan_free_bytes = 2ull * QW_REMOTE;
    b3.ram_temp_bytes = 100ull * MIB;
    b3.ram_cap_bytes = 300000000ull; // 235,929,600 + 104,857,600 > 300,000,000
    llama_kv_mixed_placement p3 = {};
    CHECK(llama_kv_mixed_choose_placement(costs, qwen_sizing(), b3, p3) == LLAMA_KV_MIXED_ERR_NO_PLACEMENT);
    CHECK(std::strstr(p3.error, "RAM cap") != nullptr);
    b3.ram_cap_bytes = 400000000ull;
    llama_kv_mixed_placement p4 = {};
    CHECK(llama_kv_mixed_choose_placement(costs, qwen_sizing(), b3, p4) == LLAMA_KV_MIXED_OK);
    CHECK_EQ(p4.n_remote, 1u);
    CHECK_EQ(p4.ram_used_bytes, QW_REMOTE + 104857600ull);
}

static void test_ram_cap_includes_reserve() {
    g_current = "ram_cap_includes_reserve";
    const auto costs = qwen_costs(16);
    llama_kv_mixed_budget b = {};
    b.cuda_free_bytes = 2800000000ull; // needs N = 4 (suffix(4) = 2,771,386,368)
    b.vulkan_free_bytes = 5ull * QW_REMOTE + 64ull * MIB; // reserve subtracted leaves room for 4
    b.vulkan_reserve_bytes = 100ull * MIB;
    llama_kv_mixed_placement p = {};
    CHECK(llama_kv_mixed_choose_placement(costs, qwen_sizing(), b, p) == LLAMA_KV_MIXED_OK);
    CHECK_EQ(p.n_remote, 4u);
    CHECK_EQ(p.ram_used_bytes, 0ull); // cap disabled

    // The cap includes the fixed remote reserve, counted once.
    b.ram_cap_bytes = 104857600ull + 4ull * QW_REMOTE - 1;
    llama_kv_mixed_placement p2 = {};
    CHECK(llama_kv_mixed_choose_placement(costs, qwen_sizing(), b, p2) == LLAMA_KV_MIXED_ERR_NO_PLACEMENT);
    CHECK(std::strstr(p2.error, "RAM cap") != nullptr);

    b.ram_cap_bytes = 104857600ull + 4ull * QW_REMOTE;
    llama_kv_mixed_placement p3 = {};
    CHECK(llama_kv_mixed_choose_placement(costs, qwen_sizing(), b, p3) == LLAMA_KV_MIXED_OK);
    CHECK_EQ(p3.n_remote, 4u);
    CHECK_EQ(p3.ram_used_bytes, 104857600ull + 4ull * QW_REMOTE);
    CHECK_EQ(p3.ram_used_bytes, 104857600ull + p3.remote_kv_bytes); // once, not doubled
}

static void test_empty_costs() {
    g_current = "empty_costs";
    const std::vector<llama_kv_mixed_layer_cost> costs;
    llama_kv_mixed_budget b = {};
    b.cuda_free_bytes = 1024ull * MIB;
    llama_kv_mixed_placement p = {};
    CHECK(llama_kv_mixed_choose_placement(costs, qwen_sizing(), b, p) == LLAMA_KV_MIXED_OK);
    CHECK_EQ(p.n_remote, 0u);
    CHECK_EQ(p.n_local, 0u);
    CHECK_EQ(p.remote_kv_bytes, 0ull);
    CHECK_EQ(p.local_kv_bytes, 0ull);
    CHECK_EQ(p.handoff_temp_bytes, 0ull);
}

static void test_heterogeneous() {
    g_current = "heterogeneous";
    // Odd layers D256 hkv4 (1024 rows), even layers D128 hkv4 (512 rows).
    char err[256] = {};
    std::vector<llama_kv_mixed_layer_cost> costs;
    for (uint32_t i = 0; i < 16; ++i) {
        llama_kv_mixed_layer_params p = qwen_layer(i);
        if (i % 2 == 1) {
            p.head_dim_k = 128;
            p.head_dim_v = 128;
            p.n_embd_k_gqa = 512;
            p.n_embd_v_gqa = 512;
        }
        llama_kv_mixed_layer_cost c = {};
        CHECK(llama_kv_mixed_estimate_layer_cost(p, qwen_sizing(), c, err, sizeof(err)) == LLAMA_KV_MIXED_OK);
        costs.push_back(c);
    }
    // D128 layers cost half of the D256 ones (records, stage and tail halve).
    CHECK_EQ(costs[1].local_bytes, costs[0].local_bytes / 2);
    CHECK_EQ(costs[1].remote_bytes, costs[0].remote_bytes / 2);
    CHECK_EQ(costs[1].convert_scratch_per_token, 4672ull);

    // CUDA holds 15 of the 16 mixed layers: one D256 moves remote.
    const uint64_t total_local = 8ull * QW_LOCAL + 8ull * 115738624ull;
    llama_kv_mixed_budget b = {};
    b.cuda_free_bytes = total_local - QW_LOCAL + 64ull * MIB;
    b.vulkan_free_bytes = 2ull * QW_REMOTE + 64ull * MIB;
    llama_kv_mixed_placement p = {};
    CHECK(llama_kv_mixed_choose_placement(costs, qwen_sizing(), b, p) == LLAMA_KV_MIXED_OK);
    CHECK_EQ(p.n_remote, 1u);
    CHECK_EQ(p.local_kv_bytes, total_local - costs[0].local_bytes);
}

int main() {
    std::printf("llama-kv-mixed-placement CPU tests (revised contract)\n");
    std::printf("linked ggml_row_size(F16,1024) = %zu\n", ggml_row_size(GGML_TYPE_F16, 1024));

    test_reference_qwen();
    test_estimator_details();
    test_n0_and_steady_state_default();
    test_n_all_and_no_layer_sum();
    test_cuda_insufficient();
    test_vulkan_insufficient();
    test_reserve_exceeds_free();
    test_trailing_diff_geometry();
    test_exact_limits();
    test_overflow_and_validation();
    test_unified_streams();
    test_monotonicity();
    test_chunk_conversion();
    test_explicit_temps();
    test_ram_cap_includes_reserve();
    test_empty_costs();
    test_heterogeneous();

    std::printf("\n%u checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}