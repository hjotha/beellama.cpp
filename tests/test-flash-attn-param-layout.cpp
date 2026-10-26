// Hardware-free regression: FA side-output pointers must survive every hint.
// Legacy layout reproducer: compile with -DGGML_TEST_LEGACY_FA_PARAM_LAYOUT.
// That build must fail an actual LSE pointer/flag invariant, not just slot IDs.
#include "ggml.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace {
#ifdef GGML_TEST_LEGACY_FA_PARAM_LAYOUT
constexpr int domain_slot = 9;
constexpr int force_vec_slot = 7;
#else
constexpr int domain_slot = GGML_FLASH_ATTN_EXT_OP_PARAM_KVARN_DOMAIN;
constexpr int force_vec_slot = GGML_FLASH_ATTN_EXT_OP_PARAM_CUDA_FORCE_VEC;
#endif
int checks = 0;
void check(bool ok, const char * message) {
    ++checks;
    if (!ok) throw std::runtime_error(message);
}
uintptr_t stored_pointer(const ggml_tensor * fa) {
    uintptr_t ptr = 0;
    std::memcpy(&ptr, &fa->op_params[GGML_FLASH_ATTN_EXT_OP_PARAM_LSE_PTR], sizeof(ptr));
    return ptr;
}
void check_binding(ggml_tensor * fa, ggml_tensor * expected) {
    check(ggml_flash_attn_ext_has_lse_out(fa) == (expected != nullptr), "hint clobbered LSE enable flag");
    check(ggml_flash_attn_ext_get_lse_out(fa) == expected, "hint clobbered LSE destination pointer");
    check(stored_pointer(fa) == reinterpret_cast<uintptr_t>(expected), "raw LSE pointer bytes changed");
    if (expected && sizeof(uintptr_t) == 8) {
        const uint64_t address = reinterpret_cast<uintptr_t>(expected);
        check((address >> 32) != 0, "test must exercise a real pointer with a nonzero high half");
        uint32_t high = 0;
        std::memcpy(&high, reinterpret_cast<const uint8_t *>(fa->op_params) +
                GGML_FLASH_ATTN_EXT_OP_PARAM_LSE_PTR * sizeof(int32_t) + sizeof(uint32_t), sizeof(high));
        check(high == uint32_t(address >> 32), "LSE pointer high half was overwritten");
    }
}
void check_hints(ggml_tensor * fa, int domain, int force_vec) {
    check(fa->op_params[domain_slot] == domain, "LSE binding clobbered KVarN domain");
    check(fa->op_params[force_vec_slot] == force_vec, "LSE binding clobbered CUDA force-vector hint");
    check(fa->op_params[GGML_FLASH_ATTN_EXT_OP_PARAM_TAIL_BODYLESS] == 1, "bodyless flag changed");
    check(fa->op_params[GGML_FLASH_ATTN_EXT_OP_PARAM_TAIL_HISTORY_SLOTS] == 3, "tail history changed");
    check(fa->op_params[GGML_FLASH_ATTN_EXT_OP_PARAM_KVARN_WINDOW_CHUNK] == 128, "window chunk changed");
    check(fa->op_params[GGML_FLASH_ATTN_EXT_OP_PARAM_KVARN_NON_CAUSAL_MASK] == 1, "non-causal hint changed");
    check(fa->op_params[4] == 8, "sparse n_kv_max hint changed");
    float scale = 0, bias = 0, cap = 0;
    ggml_flash_attn_ext_get_alibi_params(fa, &scale, &bias, &cap);
    check(scale == 0.25f && bias == 2.0f && cap == 3.0f, "leading float parameters changed");
    check(fa->op_params[GGML_FLASH_ATTN_EXT_OP_PARAM_PREC] == GGML_PREC_F32, "precision hint changed");
}
void set_hints(ggml_tensor * fa, int domain, int force_vec) {
    fa->op_params[domain_slot] = domain;
    fa->op_params[force_vec_slot] = force_vec;
    fa->op_params[GGML_FLASH_ATTN_EXT_OP_PARAM_TAIL_BODYLESS] = 1;
    fa->op_params[GGML_FLASH_ATTN_EXT_OP_PARAM_TAIL_HISTORY_SLOTS] = 3;
    fa->op_params[GGML_FLASH_ATTN_EXT_OP_PARAM_KVARN_WINDOW_CHUNK] = 128;
    fa->op_params[GGML_FLASH_ATTN_EXT_OP_PARAM_KVARN_NON_CAUSAL_MASK] = 1;
    ggml_flash_attn_ext_set_n_kv_max(fa, 8);
    ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
}
void run(int domain, bool hints_first) {
    // Tensor metadata lives in this real, aligned stack arena. On 64-bit
    // platforms the LSE tensor therefore exercises both pointer words.
    alignas(GGML_MEM_ALIGN) std::array<uint8_t, 128 * 1024> arena {};
    ggml_init_params params { arena.size(), arena.data(), true };
    ggml_context * ctx = ggml_init(params);
    check(ctx != nullptr, "cannot initialize ggml metadata arena");
    struct guard { ggml_context * ctx; ~guard() { ggml_free(ctx); } } cleanup { ctx };
    auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 32, 2, 4, 1);
    auto * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 32, 8, 2, 1);
    auto * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 32, 8, 2, 1);
    auto * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 8, 2, 1, 1);
    auto * fa = ggml_flash_attn_ext(ctx, q, k, v, mask, 0.25f, 2.0f, 3.0f);
    check_binding(fa, nullptr);
    ggml_tensor * lse = nullptr;
    if (hints_first) {
        set_hints(fa, domain, 1);
        check_binding(fa, nullptr); // FORCE_VEC must never masquerade as LSE_OUT.
        lse = ggml_flash_attn_ext_lse_out(ctx, fa);
    } else {
        lse = ggml_flash_attn_ext_lse_out(ctx, fa);
        check_binding(fa, lse);
        set_hints(fa, domain, 1); // The original DOMAIN=9 clobbers pointer high bits here.
    }
    check_binding(fa, lse);
    check_hints(fa, domain, 1);
    check(lse->op == GGML_OP_FLASH_ATTN_EXT_LSE && lse->src[0] == fa, "LSE graph dependency changed");
    check(lse->ne[0] == 4 && lse->ne[1] == 2 && lse->ne[2] == 1, "LSE shape changed");

    // Toggle every domain after attachment, including AUTO, which used to
    // zero the high pointer word on a perfectly valid 64-bit destination.
    for (int next : { GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_AUTO, GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED,
            GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ORIGINAL, GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_K_ORIGINAL_V }) {
        set_hints(fa, next, 0);
        check_binding(fa, lse);
        check_hints(fa, next, 0);
    }
    set_hints(fa, domain, 1);
    auto * replacement = ggml_dup_tensor(ctx, lse);
    replacement->op = GGML_OP_FLASH_ATTN_EXT_LSE;
    replacement->src[0] = fa;
    ggml_flash_attn_ext_set_lse_out(fa, replacement);
    check_binding(fa, replacement);
    check_hints(fa, domain, 1);
    ggml_tensor clone = *fa;
    ggml_flash_attn_ext_set_lse_out(&clone, lse);
    check_binding(&clone, lse);
    check_binding(fa, replacement);
    check_hints(&clone, domain, 1);
    ggml_flash_attn_ext_set_lse_out(fa, nullptr);
    check_binding(fa, nullptr);
    check_hints(fa, domain, 1);
    auto * reattached = ggml_flash_attn_ext_lse_out(ctx, fa);
    check_binding(fa, reattached);
    check_hints(fa, domain, 1);
}
}
int main() {
    try {
        for (int domain : { GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_AUTO, GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED,
                GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ORIGINAL, GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_K_ORIGINAL_V }) {
            run(domain, false);
            run(domain, true);
        }
        std::printf("FlashAttention parameter layout: %d checks passed (no backend/hardware initialized)\n", checks);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FlashAttention parameter layout FAIL after %d checks: %s\n", checks, error.what());
        return 1;
    }
}
