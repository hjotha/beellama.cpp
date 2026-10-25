// CPU production decode opt-in test: normal serial decode vs LLAMA_N256_LOOKAHEAD=1.
// Real Qwen3.5-4B, 512 tokens (two N256 ubatches), one output at position 511.
// Compares final logits bitwise. No GPU/RPC/server/benchmark.

#include "llama.h"
#include "llama-model.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <vector>

namespace {
const char * kDefaultModel = "/home/hjotha/models/Qwen3.5-4B-Q4_K_M.gguf";
int failures = 0;
void check(bool ok, const char * name) {
    std::printf("%s: %s\n", name, ok ? "OK" : "FAIL");
    if (!ok) ++failures;
}
bool file_ok(const char * p) {
    struct stat st;
    return stat(p, &st) == 0 && S_ISREG(st.st_mode) && st.st_size > 0;
}
struct Out {
    std::vector<float> logits;
    llama_token sampled = -1;
    llama_token sampled_dist = -1;
};
bool run_decode(llama_model * model, bool lookahead, Out & out) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = 512;
    cp.n_ubatch = 256;
    cp.n_threads = 1;
    cp.n_threads_batch = 1;
    cp.n_seq_max = 1;
    llama_context * ctx = llama_new_context_with_model(model, cp);
    if (!ctx) return false;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token bos = llama_vocab_bos(vocab);
    if (bos < 0) bos = 0;
    llama_batch b = llama_batch_init(512, 0, 1);
    b.n_tokens = 512;
    for (int i = 0; i < 512; ++i) {
        b.token[i] = bos;
        b.pos[i] = i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = 0;
    }
    b.logits[0] = 1;
    b.logits[255] = 1;
    b.logits[256] = 1;
    b.logits[511] = 1;
    setenv("LLAMA_N256_LOOKAHEAD", lookahead ? "1" : "0", 1);
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    check(rc == 0, lookahead ? "decode lookahead rc=0" : "decode serial rc=0");
    if (rc != 0) {
        llama_free(ctx);
        return false;
    }
    const size_t n = static_cast<size_t>(llama_vocab_n_tokens(vocab));
    const float * l0   = llama_get_logits_ith(ctx, 0);
    const float * l255 = llama_get_logits_ith(ctx, 255);
    const float * l256 = llama_get_logits_ith(ctx, 256);
    const float * l511 = llama_get_logits_ith(ctx, 511);
    if (!l0 || !l255 || !l256 || !l511) {
        std::printf("FAIL: no logits at 0/255/256/511\n");
        llama_free(ctx);
        return false;
    }
    out.logits.resize(4 * n);
    std::copy(l0,   l0 + n,   out.logits.begin());
    std::copy(l255, l255 + n, out.logits.begin() + n);
    std::copy(l256, l256 + n, out.logits.begin() + 2 * n);
    std::copy(l511, l511 + n, out.logits.begin() + 3 * n);
    llama_sampler * smpl = llama_sampler_init_greedy();
    out.sampled = llama_sampler_sample(smpl, ctx, 511);
    llama_sampler_free(smpl);
    llama_sampler * dist = llama_sampler_init_dist(12345);
    out.sampled_dist = llama_sampler_sample(dist, ctx, 511);
    llama_sampler_free(dist);
    bool finite = true;
    for (float v : out.logits) if (!std::isfinite(v)) { finite = false; break; }
    check(finite, lookahead ? "lookahead logits finite" : "serial logits finite");
    llama_free(ctx);
    return finite;
}
} // namespace

int main() {
    const char * path = std::getenv("N256_SMALL_MODEL");
    if (!path || !*path) path = kDefaultModel;
    if (!file_ok(path)) { std::printf("FAIL: model missing\n"); return 1; }
    llama_backend_init();
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_dev_t devices[] = {cpu, nullptr};
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.devices = devices;
    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) { llama_backend_free(); std::printf("FAIL: model load\n"); return 1; }
    if (model->arch != LLM_ARCH_QWEN35) { std::printf("FAIL: arch\n"); llama_model_free(model); llama_backend_free(); return 1; }
    Out a, b;
    const bool ok_a = run_decode(model, false, a);
    const bool ok_b = run_decode(model, true, b);
    check(ok_a, "serial decode completed");
    check(ok_b, "lookahead decode completed");
    if (ok_a && ok_b) {
        check(a.logits.size() == b.logits.size(), "logits size equal");
        int mismatch = 0;
        if (a.logits.size() == b.logits.size()) {
            for (size_t i = 0; i < a.logits.size(); ++i) {
                if (std::memcmp(&a.logits[i], &b.logits[i], sizeof(float)) != 0) ++mismatch;
            }
        }
        std::printf("logits compared=%zu mismatches=%d\n", a.logits.size(), mismatch);
        check(mismatch == 0, "serial vs lookahead logits bitwise");

        // Deterministic sampling fixture: greedy argmax over the last output row
        // must select the same token from equivalent logits.
        auto greedy = [](const std::vector<float> & logits, size_t n_vocab) {
            size_t best = 0;
            for (size_t i = 1; i < n_vocab; ++i) {
                if (logits[i] > logits[best]) best = i;
            }
            return best;
        };
        const size_t n_vocab = a.logits.size() / 4;
        const size_t ta = greedy(std::vector<float>(a.logits.begin() + 3 * n_vocab,
                                                   a.logits.begin() + 4 * n_vocab), n_vocab);
        const size_t tb = greedy(std::vector<float>(b.logits.begin() + 3 * n_vocab,
                                                   b.logits.begin() + 4 * n_vocab), n_vocab);
        std::printf("greedy fixture serial=%zu lookahead=%zu; llama sampler serial=%d lookahead=%d\n",
                    ta, tb, (int) a.sampled, (int) b.sampled);
        check(ta == tb, "deterministic sampling token id equal");
        check(a.sampled == b.sampled && a.sampled >= 0, "llama_sampler greedy token equal");
        std::printf("seeded dist serial=%d lookahead=%d\n", (int) a.sampled_dist, (int) b.sampled_dist);
        check(a.sampled_dist == b.sampled_dist && a.sampled_dist >= 0, "seeded dist sampler token equal");
    }
    llama_model_free(model);
    llama_backend_free();
    std::printf("%s: %d failure(s)\n", failures == 0 ? "OK" : "FAILED", failures);
    return failures == 0 ? 0 : 1;
}
