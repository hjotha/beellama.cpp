// CPU failure/rollback test for the experimental 2xN256 decode path.
// Real Qwen3.5-4B. Forces a deterministic TEST-ONLY failure after U0 commit,
// during U1, or before U1 publish; then verifies a following normal 2-token
// request succeeds and produces the same logits as a fresh context.

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
llama_context * make_ctx(llama_model * model) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = 512;
    cp.n_ubatch = 256;
    cp.n_threads = 1;
    cp.n_threads_batch = 1;
    cp.n_seq_max = 1;
    return llama_new_context_with_model(model, cp);
}
std::vector<float> decode_2(llama_context * ctx, llama_model * model) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token bos = llama_vocab_bos(vocab);
    if (bos < 0) bos = 0;
    llama_batch b = llama_batch_init(2, 0, 1);
    b.n_tokens = 2;
    b.token[0] = bos; b.token[1] = bos;
    b.pos[0] = 0; b.pos[1] = 1;
    b.n_seq_id[0] = 1; b.n_seq_id[1] = 1;
    b.seq_id[0][0] = 0; b.seq_id[1][0] = 0;
    b.logits[0] = 0; b.logits[1] = 1;
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    check(rc == 0, "recovery normal 2-token decode");
    if (rc != 0) return {};
    const float * lp = llama_get_logits_ith(ctx, 1);
    if (!lp) return {};
    const size_t n = static_cast<size_t>(llama_vocab_n_tokens(vocab));
    return std::vector<float>(lp, lp + n);
}
bool test_failure(llama_model * model, const char * mode, const std::vector<float> & fresh) {
    llama_context * ctx = make_ctx(model);
    if (!ctx) return false;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token bos = llama_vocab_bos(vocab);
    if (bos < 0) bos = 0;
    llama_batch b = llama_batch_init(512, 0, 1);
    b.n_tokens = 512;
    for (int i = 0; i < 512; ++i) {
        b.token[i] = bos; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0;
        b.logits[i] = 0;
    }
    b.logits[511] = 1;
    setenv("LLAMA_N256_LOOKAHEAD", "1", 1);
    setenv("LLAMA_N256_FAIL", mode, 1);
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    check(rc != 0, (std::string("failure mode ") + mode + " returned error").c_str());
    unsetenv("LLAMA_N256_FAIL");
    setenv("LLAMA_N256_LOOKAHEAD", "0", 1);
    const std::vector<float> got = decode_2(ctx, model);
    check(got.size() == fresh.size(), (std::string(mode) + ": recovery logits size").c_str());
    int mismatch = 0;
    if (got.size() == fresh.size()) {
        for (size_t i = 0; i < got.size(); ++i) {
            if (std::memcmp(&got[i], &fresh[i], sizeof(float)) != 0) ++mismatch;
        }
    } else {
        mismatch = 1;
    }
    std::printf("%s recovery logits mismatches=%d\n", mode, mismatch);
    check(mismatch == 0, (std::string(mode) + ": recovery logits equal fresh").c_str());
    llama_free(ctx);
    return mismatch == 0;
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
    mp.n_gpu_layers = 0; mp.devices = devices;
    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) { llama_backend_free(); std::printf("FAIL: model load\n"); return 1; }
    if (model->arch != LLM_ARCH_QWEN35) { llama_model_free(model); llama_backend_free(); std::printf("FAIL: arch\n"); return 1; }

    setenv("LLAMA_N256_LOOKAHEAD", "0", 1);
    llama_context * fresh_ctx = make_ctx(model);
    if (!fresh_ctx) { llama_model_free(model); llama_backend_free(); return 1; }
    const std::vector<float> fresh = decode_2(fresh_ctx, model);
    llama_free(fresh_ctx);
    check(!fresh.empty(), "fresh 2-token baseline logits");

    for (const char * mode : {"after_u0_commit", "during_u1", "before_publish"}) {
        test_failure(model, mode, fresh);
    }

    llama_model_free(model);
    llama_backend_free();
    std::printf("%s: %d failure(s)\n", failures == 0 ? "OK" : "FAILED", failures);
    return failures == 0 ? 0 : 1;
}
