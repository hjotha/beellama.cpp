// Integration tests for the planned draft-MTP KV reservation CONTRACT in the
// real common parameter paths (no model, no GPU). The parent links this test
// against the FRESH libcommon/libllama and runs it; this file is only
// syntax-checked in this round (linking new common headers against the old
// baseline library ABI is forbidden here).
// CMake target for Luna (tests/CMakeLists.txt):
//   add_executable(test-kv-mixed-mtp-params test-kv-mixed-mtp-params.cpp)
//   target_link_libraries(test-kv-mixed-mtp-params PRIVATE common llama ggml)
//   add_test(NAME test-kv-mixed-mtp-params COMMAND test-kv-mixed-mtp-params)

#include "common.h"
#include "speculative.h"

#include <cstdio>
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

static llama_context_params mtp_cparams(common_params & params) {
    return common_context_params_to_llama(params);
}

// default params with a draft-MTP profile
static common_params make_mtp_params(int32_t n_max, enum llama_kvarn_type kvarn,
                                     enum ggml_type k, enum ggml_type v) {
    common_params params; // aggregate defaults from the struct initializers
    params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    params.speculative.draft.n_max = n_max;
    params.speculative.draft.kvarn = llama_kvarn_default_params();
    params.speculative.draft.kvarn.type = kvarn;
    if (kvarn != LLAMA_KVARN_TYPE_DISABLED) {
        const auto kp = llama_kvarn_params_for_type(kvarn);
        params.speculative.draft.kvarn.key_bits = kp.key_bits;
        params.speculative.draft.kvarn.value_bits = kp.value_bits;
    }
    params.speculative.draft.cache_type_k = k;
    params.speculative.draft.cache_type_v = v;
    return params;
}

static void test_inactive_and_model_head_irrelevant() {
    // no MTP type in the profile: reserve stays inactive even with a draft
    common_params params;
    params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE };
    params.speculative.draft.n_max = 4;
    llama_context_params cparams = mtp_cparams(params);
    CHECK(!cparams.mtp_reserve_enabled);

    // MTP type but n_max == 0 (long profile off): inactive
    common_params off = make_mtp_params(0, LLAMA_KVARN_TYPE_DISABLED,
                                        GGML_TYPE_F16, GGML_TYPE_F16);
    llama_context_params c_off = mtp_cparams(off);
    CHECK(!c_off.mtp_reserve_enabled);

    // active MTP profile: enabled. The common path cannot see the model head;
    // head relevance is model-side (the constructor), so the default contract
    // must not depend on it.
    common_params on = make_mtp_params(2, LLAMA_KVARN_TYPE_DISABLED,
                                       GGML_TYPE_F16, GGML_TYPE_F16);
    llama_context_params c_on = mtp_cparams(on);
    CHECK(c_on.mtp_reserve_enabled);
}

static void test_active_representations() {
    // standard F16 draft
    {
        common_params p = make_mtp_params(4, LLAMA_KVARN_TYPE_DISABLED,
                                          GGML_TYPE_F16, GGML_TYPE_F16);
        auto cp = mtp_cparams(p);
        CHECK(cp.mtp_reserve_enabled);
        CHECK(cp.mtp_reserve_kvarn == LLAMA_KVARN_TYPE_DISABLED);
        CHECK(cp.mtp_reserve_type_k == GGML_TYPE_F16);
        CHECK(cp.mtp_reserve_type_v == GGML_TYPE_F16);
    }
    // standard Q8_0 draft
    {
        common_params p = make_mtp_params(4, LLAMA_KVARN_TYPE_DISABLED,
                                          GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);
        auto cp = mtp_cparams(p);
        CHECK(cp.mtp_reserve_type_k == GGML_TYPE_Q8_0);
        CHECK(cp.mtp_reserve_type_v == GGML_TYPE_Q8_0);
    }
    // KVarN4 draft
    {
        common_params p = make_mtp_params(4, LLAMA_KVARN_K4V4_G128,
                                          GGML_TYPE_F16, GGML_TYPE_F16);
        auto cp = mtp_cparams(p);
        CHECK(cp.mtp_reserve_kvarn == LLAMA_KVARN_K4V4_G128);
        CHECK(cp.mtp_reserve_kvarn_bits == (uint32_t(4) << 16) | 4);
    }
    // asymmetric KVarN (K3V8)
    {
        common_params p = make_mtp_params(4, LLAMA_KVARN_K3V8_G128,
                                          GGML_TYPE_F16, GGML_TYPE_F16);
        auto cp = mtp_cparams(p);
        CHECK(cp.mtp_reserve_kvarn == LLAMA_KVARN_K3V8_G128);
        CHECK(cp.mtp_reserve_kvarn_bits == (uint32_t(8) << 16) | 3);
    }
}

static void test_actual_rollback() {
    // static draft-mtp n_max 4 with no explicit target tail reserve:
    // need_n_rs_seq returns draft.n_max = 4, so the planned rollback is 4
    // (the old n_rs_seq_target copy would have been 0 and undercounted)
    {
        common_params p = make_mtp_params(4, LLAMA_KVARN_K4V4_G128,
                                          GGML_TYPE_F16, GGML_TYPE_F16);
        p.speculative.n_rs_seq_target = 0;
        auto cp = mtp_cparams(p);
        CHECK(cp.mtp_reserve_enabled);
        CHECK(cp.mtp_reserve_rollback_tokens == 4);
    }
    // adaptive profile: explicit target reserve 6 with n_max 2 -> max(2, 6) = 6
    {
        common_params p = make_mtp_params(2, LLAMA_KVARN_K4V4_G128,
                                          GGML_TYPE_F16, GGML_TYPE_F16);
        p.speculative.n_rs_seq_target = 6;
        auto cp = mtp_cparams(p);
        CHECK(cp.mtp_reserve_rollback_tokens == 6);
    }
    // clamp path: the recurrent rollback is clamped to 0 by the micro-batch
    // check but the explicit attention-KV tail reserve survives
    {
        common_params p = make_mtp_params(2, LLAMA_KVARN_K4V4_G128,
                                          GGML_TYPE_F16, GGML_TYPE_F16);
        p.n_batch = 4;
        p.n_ubatch = 2; // n_ubatch <= n_rs_seq + 1 -> n_rs_seq clamped to 0
        p.speculative.n_rs_seq_target = 5;
        auto cp = mtp_cparams(p);
        CHECK(cp.n_rs_seq == 0);
        CHECK(cp.mtp_reserve_rollback_tokens == 5);
    }
    // both zero -> KVarN canonical minimum 1
    {
        common_params p = make_mtp_params(1, LLAMA_KVARN_K4V4_G128,
                                          GGML_TYPE_F16, GGML_TYPE_F16);
        p.n_batch = 4;
        p.n_ubatch = 2; // clamp n_rs_seq to 0
        p.speculative.n_rs_seq_target = 0;
        auto cp = mtp_cparams(p);
        CHECK(cp.mtp_reserve_rollback_tokens == 1);
    }
}

static void test_mixed_target_draft_remote_policy() {
    // a REAL local-Vulkan backend alias (common_remote_attn_is_local_vulkan
    // accepts "local", "vulkan", "vulkan:N", "local:N"); the precondition
    // asserts the helper recognizes the input BEFORE the policy is applied,
    // so the assertions below test the policy, not alias recognition
    const std::string vulkan_alias = "vulkan:0";
    CHECK(common_remote_attn_is_local_vulkan(vulkan_alias));

    // MIXED target: per-layer remote Qx formats -> the draft params must have
    // the remote host cleared (layers "0") and stay local CUDA (no re-enable),
    // and the per-layer target formats must be cleared for the draft context
    {
        common_params p = make_mtp_params(4, LLAMA_KVARN_K4V4_G128,
                                          GGML_TYPE_F16, GGML_TYPE_F16);
        p.remote_attn_host = vulkan_alias;
        p.remote_attn_cache_type_k = GGML_TYPE_Q8_0;
        p.remote_attn_cache_type_v = GGML_TYPE_Q8_0;
        p.remote_attn_prefill = "remote";
        common_params dft = common_base_params_to_speculative(p);
        CHECK(dft.remote_attn_host.empty()); // mixed target: never re-enabled
        CHECK(dft.remote_attn_layers == "0");
        CHECK(dft.remote_attn_cache_type_k == GGML_TYPE_COUNT);
        CHECK(dft.remote_attn_cache_type_v == GGML_TYPE_COUNT);
    }
    // LEGACY non-mixed: local Vulkan MTP host preserved as before (layers "auto")
    {
        common_params p = make_mtp_params(4, LLAMA_KVARN_K4V4_G128,
                                          GGML_TYPE_F16, GGML_TYPE_F16);
        p.remote_attn_host = vulkan_alias;
        p.remote_attn_cache_type_k = GGML_TYPE_COUNT;
        p.remote_attn_cache_type_v = GGML_TYPE_COUNT;
        p.remote_attn_prefill = "remote";
        common_params dft = common_base_params_to_speculative(p);
        CHECK(!dft.remote_attn_host.empty()); // preserved legacy behavior
        CHECK(dft.remote_attn_host == vulkan_alias);
        CHECK(dft.remote_attn_layers == "auto");
    }
    // INVALID alias negative, separately: "local-vulkan" is NOT a recognized
    // backend alias, so no valid-backend behavior is claimed for it; the
    // policy still clears the host through the ordinary override-clear path
    // (this must not be read as evidence for the mixed-target guard)
    {
        const std::string bogus = "local-vulkan";
        CHECK(!common_remote_attn_is_local_vulkan(bogus));
        common_params p = make_mtp_params(4, LLAMA_KVARN_K4V4_G128,
                                          GGML_TYPE_F16, GGML_TYPE_F16);
        p.remote_attn_host = bogus;
        p.remote_attn_cache_type_k = GGML_TYPE_Q8_0;
        p.remote_attn_cache_type_v = GGML_TYPE_Q8_0;
        p.remote_attn_prefill = "remote";
        common_params dft = common_base_params_to_speculative(p);
        CHECK(dft.remote_attn_host.empty()); // ordinary override clear only
    }
}

int main() {
    test_inactive_and_model_head_irrelevant();
    test_active_representations();
    test_actual_rollback();
    test_mixed_target_draft_remote_policy();

    std::printf("%s: %d checks, %d failures\n", g_failures == 0 ? "PASS" : "FAIL", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}