#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens, int pos0) {
    llama_batch batch = llama_batch_init(int(tokens.size()), 0, 1);
    batch.n_tokens = int(tokens.size());
    for (int i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = pos0 + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = i + 1 == batch.n_tokens;
    }
    const int result = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return result == 0;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s path/to/qwen35-d256-model.gguf\n", argv[0]);
        return 2;
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = -1;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { std::fprintf(stderr, "model load failed\n"); return 1; }
    const int vocab_size = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int embedding_size = llama_model_n_embd_inp(model);
    if (vocab_size < 1000 || embedding_size <= 0) {
        std::fprintf(stderr, "unsupported model shape\n");
        llama_model_free(model);
        return 1;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 64; cp.n_ubatch = 32; cp.n_seq_max = 1; cp.n_rs_seq = 0;
    cp.kv_unified = false; cp.kv_tail_rollback_tokens = 0;
    cp.type_k = GGML_TYPE_TQNC_K; cp.type_v = GGML_TYPE_TQNC_V;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.offload_kqv = true;
    llama_context * faulty = llama_init_from_model(model, cp);
    llama_context * reference = llama_init_from_model(model, cp);
    if (!faulty || !reference) {
        std::fprintf(stderr, "context init failed\n");
        if (faulty) llama_free(faulty);
        if (reference) llama_free(reference);
        llama_model_free(model);
        return 1;
    }

    const std::vector<llama_token> prefix = { 123, 456, 789, 456 };
    if (!decode_tokens(faulty, prefix, 0) || !decode_tokens(reference, prefix, 0)) {
        std::fprintf(stderr, "initial prefix failed\n");
        return 1;
    }
    const float inf = std::numeric_limits<float>::infinity();
    const float invalid_values[] = { std::numeric_limits<float>::quiet_NaN(), inf, -inf };
    llama_pos after = -1;
    for (const float value : invalid_values) {
        llama_batch invalid = llama_batch_init(1, embedding_size, 1);
        invalid.n_tokens = 1;
        std::fill(invalid.embd, invalid.embd + embedding_size, value);
        invalid.pos[0] = int(prefix.size());
        invalid.n_seq_id[0] = 1;
        invalid.seq_id[0][0] = 0;
        invalid.logits[0] = 1;
        const int bad_status = llama_decode(faulty, invalid);
        llama_batch_free(invalid);
        if (bad_status != -3) {
            std::fprintf(stderr, "expected NC compute failure -3 for %g, got %d\n", value, bad_status);
            return 1;
        }
        after = llama_memory_seq_pos_max(llama_get_memory(faulty), 0);
        if (after == -1) {
            std::printf("failure for %g invalidated the whole sequence; replaying prefix\n", value);
            if (!decode_tokens(faulty, prefix, 0)) return 1;
        } else if (after != int(prefix.size()) - 1) {
            std::fprintf(stderr, "unexpected surviving position after failure: %d\n", int(after));
            return 1;
        }
    }

    const std::vector<llama_token> continuation = { 321 };
    if (!decode_tokens(faulty, continuation, int(prefix.size())) ||
        !decode_tokens(reference, continuation, int(prefix.size()))) {
        std::fprintf(stderr, "valid continuation failed after rollback\n");
        return 1;
    }
    const float * a = llama_get_logits_ith(faulty, -1);
    const float * b = llama_get_logits_ith(reference, -1);
    if (!a || !b) { std::fprintf(stderr, "missing logits\n"); return 1; }
    double max_diff = 0;
    int top_a = 0, top_b = 0;
    for (int i = 0; i < vocab_size; ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            std::fprintf(stderr, "non-finite logits after recovery\n");
            return 1;
        }
        max_diff = std::max(max_diff, std::abs(double(a[i]) - b[i]));
        if (a[i] > a[top_a]) top_a = i;
        if (b[i] > b[top_b]) top_b = i;
    }
    std::printf("NC model rollback: 3 invalid inputs rejected, surviving_pos=%d max_logit_diff=%g top=%d/%d\n",
                int(after), max_diff, top_a, top_b);
    const bool ok = max_diff < 1e-3 && top_a == top_b;
    llama_free(faulty);
    llama_free(reference);
    llama_model_free(model);
    llama_backend_free();
    return ok ? 0 : 1;
}
