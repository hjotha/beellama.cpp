#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

static bool decode(llama_context * ctx, const std::vector<llama_token> & tokens, int pos) {
    for (size_t start = 0; start < tokens.size(); start += 32) {
        const int count = int(std::min<size_t>(32, tokens.size() - start));
        auto batch = llama_batch_init(count, 0, 1);
        batch.n_tokens = count;
        for (int i = 0; i < count; ++i) {
            batch.token[i] = tokens[start + i];
            batch.pos[i] = pos + int(start) + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = i + 1 == count;
        }
        const int status = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (status != 0) return false;
    }
    return true;
}

static bool equal_logits(llama_context * a, llama_context * b, int vocab) {
    const float * x = llama_get_logits_ith(a, -1);
    const float * y = llama_get_logits_ith(b, -1);
    if (!x || !y) return false;
    double maximum = 0;
    int top_x = 0, top_y = 0;
    for (int i = 0; i < vocab; ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i])) return false;
        maximum = std::max(maximum, std::abs(double(x[i]) - y[i]));
        if (x[i] > x[top_x]) top_x = i;
        if (y[i] > y[top_y]) top_y = i;
    }
    std::printf("logits max_diff=%g top=%d/%d\n", maximum, top_x, top_y);
    return maximum < 1e-3 && top_x == top_y;
}

static bool snapshot(llama_context * ctx, std::vector<uint8_t> & state) {
    const size_t size = llama_state_seq_get_size(ctx, 0);
    if (!size) return false;
    state.resize(size);
    return llama_state_seq_get_data(ctx, state.data(), state.size(), 0) == size;
}

static bool coverage(llama_context * ctx, int tail) {
    if (tail == 0) return true;
    llama_kv_tail_coverage_info info{};
    if (!llama_kv_tail_get_coverage(ctx, 0, 0, &info)) return false;
    std::printf("tail coverage requested=%u exact=%u flags=%u\n", info.requested, info.exact, info.degradation_flags);
    return info.requested == unsigned(tail) && info.exact == unsigned(tail) && info.degradation_flags == 0;
}

int main(int argc, char ** argv) {
    if (argc != 4 || (std::strcmp(argv[2], "nc") && std::strcmp(argv[2], "saw"))) {
        std::fprintf(stderr, "usage: %s model.gguf nc|saw 0|128|256\n", argv[0]);
        return 2;
    }
    const int tail = std::atoi(argv[3]);
    if (tail != 0 && tail != 128 && tail != 256) return 2;
    const bool nc = std::strcmp(argv[2], "nc") == 0;
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = -1;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { std::fprintf(stderr, "model load failed\n"); return 1; }
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int embedding = llama_model_n_embd_inp(model);
    if (vocab <= 4096 || embedding <= 0) return 1;
    auto cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 32; cp.n_ubatch = 32; cp.n_seq_max = 1;
    cp.n_rs_seq = 0; cp.kv_unified = false; cp.kv_tail_rollback_tokens = 0;
    cp.type_k = nc ? GGML_TYPE_TQNC_K : GGML_TYPE_SAW_INT4;
    cp.type_v = nc ? GGML_TYPE_TQNC_V : GGML_TYPE_SAW_INT4;
    cp.kv_tail_tokens = tail;
    cp.kv_tail_type = GGML_TYPE_F16;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.offload_kqv = true;
    if (tail != 0) {
        const auto expect_rejected = [&](llama_context_params bad, const char * reason) {
            llama_context * unexpected = llama_init_from_model(model, bad);
            if (unexpected) {
                std::fprintf(stderr, "unsupported tail configuration was accepted: %s\n", reason);
                llama_free(unexpected);
                return false;
            }
            return true;
        };
        auto bad = cp; bad.kv_tail_tokens = 64;
        if (!expect_rejected(bad, "tail 64")) return 1;
        bad = cp; bad.kv_tail_type = GGML_TYPE_BF16;
        if (!expect_rejected(bad, "BF16 tail")) return 1;
        bad = cp; bad.n_seq_max = 2;
        if (!expect_rejected(bad, "multiple sequences")) return 1;
        bad = cp; bad.kv_unified = true;
        if (!expect_rejected(bad, "unified cache")) return 1;
        bad = cp; bad.n_rs_seq = 1;
        if (!expect_rejected(bad, "speculative sequence")) return 1;
        bad = cp; bad.kv_tail_rollback_tokens = 1;
        if (!expect_rejected(bad, "tail speculative rollback")) return 1;
        bad = cp; bad.n_ctx = 256; bad.kv_tail_tokens = 256;
        if (!expect_rejected(bad, "tail covers entire context")) return 1;
    }
    llama_context * reference = llama_init_from_model(model, cp);
    llama_context * restored = llama_init_from_model(model, cp);
    if (!reference || !restored) { std::fprintf(stderr, "context init failed\n"); return 1; }
    std::vector<llama_token> prefix(320);
    for (size_t i = 0; i < prefix.size(); ++i) prefix[i] = 1000 + int((37*i + 123) % (vocab - 1000));
    if (!decode(reference, prefix, 0) || !coverage(reference, tail)) return 1;

    std::vector<uint8_t> original, before, after;
    if (!snapshot(reference, original) ||
            llama_state_seq_set_data(restored, original.data(), original.size(), 0) != original.size() ||
            !coverage(restored, tail) || !snapshot(restored, before)) return 1;
    if (llama_state_seq_set_data(restored, original.data(), original.size() - 1, 0) != 0 ||
            !snapshot(restored, after) || before != after ||
            llama_memory_seq_pos_max(llama_get_memory(restored), 0) != 319) {
        std::fprintf(stderr, "truncated snapshot was accepted or modified the existing sequence\n");
        return 1;
    }
    const std::vector<llama_token> next = {321};
    if (!decode(reference, next, 320) || !decode(restored, next, 320) ||
            !equal_logits(reference, restored, vocab)) return 1;

    // The legacy SAW path without a tail is deliberately unchanged and does
    // not promise rejection of non-finite input.
    if (nc || tail != 0) {
        auto invalid = llama_batch_init(1, embedding, 1);
        invalid.n_tokens = 1;
        std::fill(invalid.embd, invalid.embd + embedding, std::numeric_limits<float>::quiet_NaN());
        invalid.pos[0] = 321; invalid.n_seq_id[0] = 1; invalid.seq_id[0][0] = 0; invalid.logits[0] = 1;
        const int status = llama_decode(restored, invalid);
        llama_batch_free(invalid);
        if (status != -3) { std::fprintf(stderr, "invalid embedding returned %d, expected -3\n", status); return 1; }
        const auto pos = llama_memory_seq_pos_max(llama_get_memory(restored), 0);
        if (pos == -1) {
            if (!decode(restored, prefix, 0) || !decode(restored, next, 320)) return 1;
        } else if (pos != 320) {
            std::fprintf(stderr, "unexpected rollback position %d\n", int(pos));
            return 1;
        }
    }
    const std::vector<llama_token> final = {654};
    if (!decode(reference, final, 321) || !decode(restored, final, 321) ||
            !equal_logits(reference, restored, vocab) || !coverage(restored, tail)) return 1;
    std::printf("PASS %s tail=%d prefix=320 snapshot/truncation/%sreuse\n",
            argv[2], tail, nc || tail != 0 ? "rollback/" : "");
    llama_free(restored);
    llama_free(reference);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
