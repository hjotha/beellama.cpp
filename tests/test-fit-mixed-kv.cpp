// Pure-CPU unit tests for the mixed-KV capacity fit math used by
// common_fit_mixed_kv_context (common/common.cpp): per-layer mixed costs via
// llama_kv_mixed_estimate_layer_cost, placement feasibility under budgets and
// reserves, the active-MTP draft reserve via llama_kv_mixed_mtp_budget_estimate,
// and the recurrent reserve formula mirrored from the runtime planner. The
// thin 256-aligned feasibility search in the fitter is a binary search over
// exactly this math.
//
// No GPU/model/shared build: links only the pure placement/estimator sources
// and stable GGML.
//
// Build:
//   g++ -O2 -std=c++17 -I ggml/include -I src -I include
//       tests/test-fit-mixed-kv.cpp src/llama-kv-mixed-placement.cpp
//       src/llama-kv-mixed-mtp-budget.cpp src/llama-kvarn.cpp
//       -L <baseline-bin> -lggml-base -lggml-cpu -lgomp -lpthread
//       -o <out>/test-fit-mixed-kv
//   LD_LIBRARY_PATH=<baseline-bin> <out>/test-fit-mixed-kv

#include "llama.h"
#include "llama-kv-mixed-placement.h"
#include "llama-kv-mixed-mtp-budget.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); \
        ++g_failures; \
    } \
} while (0)

#define CHECK_EQ(a, b, msg) do { \
    if (!((a) == (b))) { \
        std::fprintf(stderr, "FAIL %s:%d: %s (got %lld, want %lld)\n", __FILE__, __LINE__, msg, \
                (long long) (a), (long long) (b)); \
        ++g_failures; \
    } \
} while (0)

constexpr uint32_t N_ATTN = 16;
constexpr uint32_t N_RECR = 8;
constexpr uint32_t N_MTP = 2;

// Synthetic Qwen35-like geometry: 16 full-attention layers (K and V head
// dim 256, 4 KV heads), 8 recurrent layers, 2 MTP nextn layers.
struct synthetic_model {
    std::vector<llama_kv_mixed_layer_params> attn;
    std::vector<llama_kv_mixed_mtp_layer> mtp;
    uint64_t rs_bytes = 0; // runtime formula reserve (documented reference)
};

synthetic_model make_model(uint32_t n_rs_seq) {
    synthetic_model m;
    for (uint32_t i = 0; i < N_ATTN; ++i) {
        llama_kv_mixed_layer_params l = {};
        l.layer = i;
        l.head_dim_k = 256;
        l.head_dim_v = 256;
        l.n_head_kv = 4;
        l.n_embd_k_gqa = 1024;
        l.n_embd_v_gqa = 1024;
        l.kvarn_bits_k = 4;
        l.kvarn_bits_v = 4;
        l.qx_type_k = GGML_TYPE_Q4_0;
        l.qx_type_v = GGML_TYPE_Q4_0;
        m.attn.push_back(l);
    }
    for (uint32_t i = 0; i < N_MTP; ++i) {
        llama_kv_mixed_mtp_layer l = {};
        l.layer = N_ATTN + N_RECR + i;
        l.head_dim_k = 256;
        l.head_dim_v = 256;
        l.n_head_kv = 4;
        l.n_embd_k_gqa = 1024;
        l.n_embd_v_gqa = 1024;
        m.mtp.push_back(l);
    }
    // Recurrent reserve (mirror of the runtime planner formula): F32 rows of
    // (n_embd_r + n_embd_s) per (1 + n_rs_seq) snapshot over the recurrent
    // layers. Synthetic: n_embd_r = 256, n_embd_s = 64, 8 recurrent layers.
    const uint32_t n_embd_r = 256;
    const uint32_t n_embd_s = 64;
    const uint64_t rs_rows = 1ull * (1u + n_rs_seq);
    m.rs_bytes = uint64_t(n_embd_r + n_embd_s) * sizeof(float) * rs_rows * N_RECR;
    return m;
}

llama_kv_mixed_sizing make_sizing(uint64_t capacity) {
    llama_kv_mixed_sizing sizing = {};
    sizing.capacity_tokens = capacity;
    sizing.n_seq_max = 1;
    sizing.kv_unified = false;
    sizing.stage_tail_groups = 2;
    sizing.stage_reserve_groups = 1;
    sizing.tail_exact_tokens = std::min<uint64_t>(128, capacity);
    sizing.tail_rollback_tokens = 1;
    sizing.tail_type = GGML_TYPE_F16;
    return sizing;
}

// Feasibility of capacity C under budgets + reserves (the fitter's lambda):
// AUTO uses llama_kv_mixed_choose_placement (minimum remote N subject to both
// budgets and the global RAM cap; the Vulkan reserve only applies when N > 0);
// explicit/full are verified by exact sums and never silently changed.
bool feasible(const synthetic_model & m, uint64_t C, uint64_t cuda_free,
        uint64_t vulkan_free, uint64_t cuda_reserve_extra, uint64_t vulkan_reserve,
        uint64_t ram_cap, int remote_policy, uint32_t & n_remote_out) {
    const llama_kv_mixed_sizing sizing = make_sizing(C);
    std::vector<llama_kv_mixed_layer_cost> costs;
    for (const auto & l : m.attn) {
        llama_kv_mixed_layer_cost cost = {};
        char err[256] = {};
        if (llama_kv_mixed_estimate_layer_cost(l, sizing, cost, err, sizeof(err))
                != LLAMA_KV_MIXED_OK) {
            return false;
        }
        costs.push_back(cost);
    }
    const uint64_t cuda_reserve = m.rs_bytes + cuda_reserve_extra;
    if (cuda_reserve > cuda_free) {
        return false;
    }
    const uint64_t cuda_available = cuda_free - cuda_reserve;
    if (remote_policy < 0) {
        llama_kv_mixed_budget budget = {};
        budget.cuda_free_bytes = cuda_free; // raw free; the chooser applies the reserve
        budget.cuda_reserve_bytes = cuda_reserve;
        budget.vulkan_free_bytes = vulkan_free; // reserve applied by the chooser only when N>0
        budget.vulkan_reserve_bytes = vulkan_reserve;
        budget.ram_cap_bytes = ram_cap;
        budget.handoff_chunk_tokens = 1024;
        budget.handoff_concurrency = 1;
        llama_kv_mixed_placement placement = {};
        if (llama_kv_mixed_choose_placement(costs, sizing, budget, placement)
                != LLAMA_KV_MIXED_OK) {
            return false;
        }
        n_remote_out = placement.n_remote;
        return true;
    }
    const uint32_t n_remote = remote_policy == 0 ? N_ATTN : uint32_t(remote_policy);
    uint64_t local_suffix = 0;
    uint64_t remote_prefix = 0;
    for (uint32_t i = 0; i < N_ATTN; ++i) {
        if (i < n_remote) {
            remote_prefix += costs[i].remote_bytes;
        } else {
            local_suffix += costs[i].local_bytes;
        }
    }
    if (local_suffix > cuda_available) {
        return false;
    }
    if (vulkan_reserve > vulkan_free) {
        return false;
    }
    if (remote_prefix > vulkan_free - vulkan_reserve) {
        return false;
    }
    if (ram_cap > 0 && remote_prefix + vulkan_reserve > ram_cap) {
        return false;
    }
    n_remote_out = n_remote;
    return true;
}

// The fitter's search: largest 256-aligned capacity <= train_ctx that is
// feasible (binary search over the same math).
uint64_t fit_search(const synthetic_model & m, uint64_t train_ctx,
        uint64_t cuda_free, uint64_t vulkan_free, uint64_t cuda_reserve_extra,
        uint64_t vulkan_reserve, uint64_t ram_cap, int remote_policy,
        uint32_t & n_remote_out) {
    const uint64_t align = 256;
    uint64_t lo = align;
    uint64_t hi = (train_ctx / align) * align;
    uint64_t best = 0;
    while (lo <= hi) {
        const uint64_t mid = lo + (hi - lo) / 2;
        const uint64_t mid_aligned = (mid / align) * align;
        uint32_t n_remote = 0;
        if (mid_aligned >= align && feasible(m, mid_aligned, cuda_free, vulkan_free,
                cuda_reserve_extra, vulkan_reserve, ram_cap, remote_policy, n_remote)) {
            best = mid_aligned;
            n_remote_out = n_remote;
            lo = mid_aligned + align;
        } else {
            hi = mid_aligned >= align ? mid_aligned - align : 0;
        }
    }
    return best;
}

void test_costs_and_alignment() {
    const synthetic_model m = make_model(1);
    // Per-layer mixed cost at 204800 capacity: KVarN4 K+V records
    // (143360 B/group, 1120 B/token) + stage + intrinsic tail + rollback;
    // Q4_0 rows 1152 B/token.
    const llama_kv_mixed_sizing sizing = make_sizing(204800);
    llama_kv_mixed_layer_cost cost = {};
    char err[256] = {};
    CHECK(llama_kv_mixed_estimate_layer_cost(m.attn[0], sizing, cost, err, sizeof(err))
            == LLAMA_KV_MIXED_OK, "layer cost estimate");
    // KVarN per-token record cost (canonical 1120 B/token for D256 x 4 kv).
    CHECK_EQ(cost.local_per_token, 1120ull, "KVarN local per-token");
    CHECK_EQ(cost.remote_per_token, 1152ull, "Q4_0 remote per-token (K+V)");
    // Q4_0 per-layer capacity @ 204800: 235,929,600 B (matches the measured
    // all-16 Q4 3600 MiB / 16 layers).
    CHECK_EQ(cost.remote_bytes, 235929600ull, "Q4_0 remote capacity per layer");
    // KVarN local capacity @ 204800 (records 229,376,000 + stage 1,572,864 +
    // tail 528,384 = 231,477,248 B, the documented planner value).
    CHECK_EQ(cost.local_bytes, 231477248ull, "KVarN local capacity per layer");
    // 256 alignment of the search result. With ample CUDA the runtime-mirror
    // auto policy keeps everything local (minimum remote N = 0).
    uint32_t n_remote = 0;
    const uint64_t ctx = fit_search(m, 4096 * 4, 12ull << 30, 8ull << 30,
            650ull << 20, 512ull << 20, 0, -1, n_remote);
    CHECK(ctx > 0 && ctx % 256 == 0, "search result must be 256-aligned");
    CHECK(n_remote <= N_ATTN, "auto policy remote count bounded");
}

void test_placement_policies() {
    const synthetic_model m = make_model(1);
    const uint64_t C = 16384;
    uint32_t n_remote = 0;
    // Auto: minimum remote N given the CUDA budget (runtime mirror). Ample
    // CUDA keeps everything local; a constrained CUDA budget pushes layers
    // remote (12 local suffix layers fit ~245 MiB, the 13th does not).
    CHECK(feasible(m, C, 12ull << 30, 8ull << 30, 650ull << 20, 512ull << 20,
            0, -1, n_remote), "auto policy feasible");
    CHECK_EQ(n_remote, 0u, "auto minimum N is 0 with ample CUDA");
    CHECK(feasible(m, C, 900ull << 20, 8ull << 30, 650ull << 20, 512ull << 20,
            0, -1, n_remote), "constrained CUDA auto feasible");
    CHECK_EQ(n_remote, 4u, "constrained CUDA pushes exactly the non-fitting prefix remote");
    // Full policy: ALL attention layers remote; feasible when the remote
    // budget covers it.
    bool full_ok = feasible(m, C, 12ull << 30, 8ull << 30, 650ull << 20,
            512ull << 20, 0, 0, n_remote);
    CHECK(full_ok, "full policy feasible at 16k with 8 GiB remote");
    CHECK_EQ(n_remote, N_ATTN, "full policy remote count");
    // Explicit N must be respected exactly (not silently changed).
    const uint32_t explicit_n = 4;
    CHECK(feasible(m, C, 12ull << 30, 8ull << 30, 650ull << 20, 512ull << 20,
            0, int(explicit_n), n_remote), "explicit N feasible");
    CHECK_EQ(n_remote, explicit_n, "explicit N respected");
    // An explicit N beyond the feasible remote count is rejected.
    CHECK(!feasible(m, C, 12ull << 30, 256ull << 20, 650ull << 20,
            512ull << 20, 0, N_ATTN, n_remote),
            "explicit N beyond the remote budget must be rejected");
    // Constrained CUDA with an explicit N that keeps a smaller suffix: 4
    // remote + 12 local (245 MiB) does not fit 900 MiB - 650 MiB reserve;
    // the explicit N must be rejected rather than silently changed.
    CHECK(!feasible(m, C, 880ull << 20, 8ull << 30, 650ull << 20,
            512ull << 20, 0, 4, n_remote),
            "explicit N with an oversized local suffix must be rejected");
    // Global RAM cap: full policy is rejected when the remote footprint
    // exceeds the cap.
    CHECK(!feasible(m, C, 12ull << 30, 8ull << 30, 650ull << 20,
            512ull << 20, 64ull << 20, 0, n_remote),
            "remote footprint above the global RAM cap must be rejected");
}

void test_feasibility_monotonic_and_rs() {
    const synthetic_model m = make_model(2);
    // RS reserve formula (runtime mirror): (256 + 64) * 4 * 3 snapshots * 8
    // recurrent layers = 30,720 B.
    CHECK_EQ(m.rs_bytes, (uint64_t)(256 + 64) * 4 * 3 * N_RECR,
            "recurrent reserve formula");
    // Monotonicity: feasible at a moderate capacity, infeasible when the
    // capacity demands more than the budgets allow (costs grow with C).
    uint32_t n_remote = 0;
    CHECK(feasible(m, 4096, 12ull << 30, 8ull << 30, 650ull << 20, 512ull << 20,
            0, -1, n_remote), "feasible at 4096");
    CHECK(!feasible(m, 1ull << 30, 12ull << 30, 8ull << 30, 650ull << 20,
            512ull << 20, 0, -1, n_remote), "infeasible at 1G tokens");
    // Search: the found capacity must be feasible and one step larger must
    // not be (largest aligned feasible).
    const uint64_t found = fit_search(m, 1ull << 20, 12ull << 30, 8ull << 30,
            650ull << 20, 512ull << 20, 0, -1, n_remote);
    CHECK(found > 0, "search found a capacity");
    CHECK(feasible(m, found, 12ull << 30, 8ull << 30, 650ull << 20, 512ull << 20,
            0, -1, n_remote), "found capacity feasible");
    if (found + 256 <= (1ull << 20)) {
        CHECK(!feasible(m, found + 256, 12ull << 30, 8ull << 30, 650ull << 20,
                512ull << 20, 0, -1, n_remote), "one step above is infeasible");
    }
}

void test_mtp_budget() {
    // Active MTP draft reserve: kvarn draft at the target capacity.
    llama_kv_mixed_mtp_input mtp = {};
    mtp.n_ctx = 16384;
    mtp.n_seq_max = 1;
    mtp.kv_unified = false;
    mtp.n_batch = 256;
    mtp.n_ubatch = 256;
    mtp.kvarn = LLAMA_KVARN_K4V4_G128;
    mtp.kvarn_bits = 4 | (4 << 16);
    mtp.tail_type = GGML_TYPE_F16;
    mtp.tail_rollback_tokens = 1;
    for (uint32_t i = 0; i < N_MTP; ++i) {
        llama_kv_mixed_mtp_layer l = {};
        l.layer = i;
        l.head_dim_k = 256;
        l.head_dim_v = 256;
        l.n_head_kv = 4;
        l.n_embd_k_gqa = 1024;
        l.n_embd_v_gqa = 1024;
        mtp.layers.push_back(l);
    }
    llama_kv_mixed_mtp_budget budget = {};
    CHECK(llama_kv_mixed_mtp_budget_estimate(mtp, budget) == LLAMA_KV_MIXED_OK,
            "MTP kvarn budget estimate");
    CHECK(budget.total_bytes > 0, "MTP budget non-zero");
    CHECK_EQ(budget.layer_bytes.size(), N_MTP, "MTP per-layer bytes");
    // KVarN per-token records dominate: 1120 B/token x 16384 = 18,350,080
    // plus stage/tail; the total must exceed the record floor.
    CHECK(budget.total_bytes >= 1120ull * 16384, "MTP budget covers records");
    // Standard draft representation is a different (larger) estimate.
    llama_kv_mixed_mtp_input std_mtp = mtp;
    std_mtp.kvarn = LLAMA_KVARN_TYPE_DISABLED;
    std_mtp.kvarn_bits = 0;
    std_mtp.type_k = GGML_TYPE_Q8_0;
    std_mtp.type_v = GGML_TYPE_Q8_0;
    llama_kv_mixed_mtp_budget std_budget = {};
    CHECK(llama_kv_mixed_mtp_budget_estimate(std_mtp, std_budget) == LLAMA_KV_MIXED_OK,
            "MTP standard budget estimate");
    CHECK(std_budget.total_bytes > budget.total_bytes,
            "Q8_0 standard draft cache exceeds the kvarn draft cache");
}

} // namespace

int main() {
    std::printf("test-fit-mixed-kv: pure CPU mixed-KV capacity fit math\n");
    std::printf("NOTE: this suite covers the shared fit math (costs, placement, reserves) only;\n");
    std::printf("the actual common_fit_mixed_kv_context wiring (devices, MemAvailable, search)\n");
    std::printf("requires a loaded model + GPU and is validated by the parent's GPU build.\n");
    test_costs_and_alignment();
    test_placement_policies();
    test_feasibility_monotonic_and_rs();
    test_mtp_budget();
    std::printf("== summary: %s\n", g_failures == 0 ? "ALL PASS" : "FAILURES PRESENT");
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}