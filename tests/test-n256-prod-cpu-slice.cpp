// Failing integration gate for the production 2xN256 CPU lookahead slice.
//
// This test is intentionally NOT registered in CTest: it must fail (exit 1)
// until the real per-layer two-ubatch path is connected. It loads a real small
// qwen35 hybrid model on CPU, validates that the normal serial decode baseline
// works, then checks the explicit integration probe. No skip, no WILL_FAIL, no
// GPU/RPC/Xbox/server/benchmark.
//
// The U0/U1 handle + per-layer commit contract is covered by the passing
// test-n256-ubatch-cursor-cpu; this gate is only for the missing graph builder
// and scheduler. When the probe becomes true, extend this file with the
// 512-token serial/lookahead comparison and register it as a CTest.
//
// Map: beellama-gate/20261024/wavefront-gate-v3/N256-PROD-CPU-MAP.md

#include "llama.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-arch.h"
#include "llama-batch.h"
#include "llama-memory.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>

namespace {

const char * kDefaultModel = "/home/hjotha/models/Qwen3.5-4B-Q4_K_M.gguf";

bool file_ok(const char * path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode) && st.st_size > 0;
}

int fail(const char * why) {
    std::printf("FAIL: %s\n", why);
    return 1;
}

} // namespace

int main() {
    const char * model_path = std::getenv("N256_SMALL_MODEL");
    if (!model_path || !*model_path) model_path = kDefaultModel;
    if (!file_ok(model_path)) {
        std::printf("FAIL: small model missing/unreadable: %s\n", model_path);
        return 1;
    }

    llama_backend_init();

    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu) {
        llama_backend_free();
        return fail("CPU backend device unavailable");
    }
    ggml_backend_dev_t devices[] = {cpu, nullptr};

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.devices = devices;

    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        llama_backend_free();
        return fail("small model load failed");
    }

    if (model->arch != LLM_ARCH_QWEN35) {
        std::printf("FAIL: expected qwen35 architecture, got %d\n", (int) model->arch);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    int n_recr = 0;
    int n_full = 0;
    for (int il = 0; il < model->hparams.n_layer(); ++il) {
        if (model->hparams.is_recr(il)) ++n_recr; else ++n_full;
    }
    std::printf("model: layers=%d recr=%d full=%d\n", model->hparams.n_layer(), n_recr, n_full);
    if (n_recr == 0 || n_full == 0) {
        std::printf("FAIL: small model is not hybrid (need full + recurrent layers)\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = 512;
    cp.n_ubatch = 256;
    cp.n_threads = 1;
    cp.n_threads_batch = 1;
    cp.n_seq_max = 1;

    llama_context * ctx = llama_new_context_with_model(model, cp);
    if (!ctx) {
        llama_model_free(model);
        llama_backend_free();
        return fail("context creation failed");
    }

    // Baseline: normal serial decode must work before we demand the new path.
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token bos = llama_vocab_bos(vocab);
    if (bos < 0) bos = 0;

    llama_batch batch = llama_batch_init(2, 0, 1);
    batch.n_tokens = 2;
    batch.token[0] = bos;
    batch.token[1] = bos;
    batch.pos[0] = 0;
    batch.pos[1] = 1;
    batch.n_seq_id[0] = 1;
    batch.n_seq_id[1] = 1;
    batch.seq_id[0][0] = 0;
    batch.seq_id[1][0] = 0;
    batch.logits[0] = 0;
    batch.logits[1] = 1;

    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        std::printf("FAIL: baseline serial decode rc=%d\n", rc);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    std::printf("baseline: serial 2-token CPU decode OK\n");

    if (!ctx->n256_lookahead_available()) {
        std::printf("FAIL: production 2xN256 CPU lookahead is NOT connected\n");
        std::printf("      normal decode above is serial; it is not being disguised as overlap\n");
        std::printf("      missing: per-layer qwen35 graph split, two-ubatch scheduler, real\n");
        std::printf("      GDN/KV/mask/tail/cell-ID commit cursors, append/commit/publish epochs,\n");
        std::printf("      lifetime/rollback and serial-vs-lookahead tensor comparison\n");
        std::printf("      map: beellama-gate/20261024/wavefront-gate-v3/N256-PROD-CPU-MAP.md\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // Reached only after the real path is connected. Keep this branch failing
    // until the 512-token two-ubatch comparison is implemented, so a stub can
    // never turn this test green.
    std::printf("FAIL: probe is true but the 512-token integration body is not implemented\n");
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 1;
}
