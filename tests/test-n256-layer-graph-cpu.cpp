// CPU integration test for the qwen35 layer-only graph builder.
//
// Real Qwen3.5-4B hybrid model on CPU. Builds one full-attention layer and one
// recurrent/GDN layer through the new layer-only graph, applies the real hybrid
// memory context for a 256-token U0, executes on CPU, and checks finite output
// with N256 shape. It does NOT enable the production scheduler and the probe
// must remain false. No model generation, GPU, RPC, server or benchmark.

#include "llama.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-batch.h"
#include "llama-memory.h"
#include "llama-graph.h"
#include "models.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
#include <vector>

namespace {

const char * kDefaultModel = "/home/hjotha/models/Qwen3.5-4B-Q4_K_M.gguf";
int failures = 0;

void check(bool ok, const char * name) {
    std::printf("%s: %s\n", name, ok ? "OK" : "FAIL");
    if (!ok) ++failures;
}

bool file_ok(const char * path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode) && st.st_size > 0;
}

struct Session {
    llama_context * ctx = nullptr;
    llama_memory_context_ptr mctx;
    llama_batch batch{};
    bool batch_live = false;
};

bool open_session(llama_model * model, const llama_context_params & cp, Session & s) {
    s.ctx = llama_new_context_with_model(model, cp);
    if (!s.ctx) return false;
    llama_memory_t mem = llama_get_memory(s.ctx);
    if (!mem) return false;

    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token bos = llama_vocab_bos(vocab);
    if (bos < 0) bos = 0;

    s.batch = llama_batch_init(512, 0, 1);
    s.batch_live = true;
    s.batch.n_tokens = 512;
    for (int i = 0; i < 512; ++i) {
        s.batch.token[i] = bos;
        s.batch.pos[i] = i;
        s.batch.n_seq_id[i] = 1;
        s.batch.seq_id[i][0] = 0;
    }
    llama_batch_allocr balloc(model->hparams.n_pos_per_embd());
    if (!balloc.init(s.batch, *vocab, mem, model->hparams.n_embd, 1, false)) return false;
    s.mctx = mem->init_batch(balloc, cp.n_ubatch, false);
    if (!s.mctx || llama_memory_status_is_fail(s.mctx->get_status())) return false;
    return true;
}

void close_session(Session & s) {
    s.mctx.reset();
    if (s.batch_live) {
        llama_batch_free(s.batch);
        s.batch_live = false;
    }
    if (s.ctx) {
        llama_free(s.ctx);
        s.ctx = nullptr;
    }
}

bool run_two_layers(Session & s, const llama_model_qwen35 & qmodel, int il_full, int il_recr,
                    uint32_t index, const char * tag) {
    check(s.mctx->set_ubatch_index(index), (std::string(tag) + ": bind active index").c_str());
    const llama_ubatch & u = s.mctx->get_ubatch();
    check(u.n_tokens == 256, (std::string(tag) + ": active ubatch is N256").c_str());
    if (u.n_tokens != 256) return false;

    if (!s.mctx->apply()) {
        std::printf("FAIL: mctx apply\n");
        return false;
    }
    // Same boundary as process_ubatch: clear store slots before graph build.
    s.mctx->reset_graph_store_slots();

    llm_graph_result res(s.ctx->graph_max_nodes(256));
    llm_graph_params params = s.ctx->graph_params(&res, u, s.mctx.get(), LLM_GRAPH_TYPE_DECODER);
    params.n_outputs = 1;

    llama_model_qwen35::graph g(qmodel, params, llama_model_qwen35::graph::layer_only_tag{});
    ggml_tensor * inp = g.build_input_embd_layer();
    ggml_tensor * pos = g.build_input_pos_layer();
    int sections[4];
    std::copy(std::begin(qmodel.hparams.rope_sections),
              std::begin(qmodel.hparams.rope_sections) + 4, sections);

    // Build both layer types in one graph so the hybrid attention and recurrent
    // inputs are both reachable/allocated. This is not the full serial chain.
    ggml_tensor * out_full = g.build_layer_only(il_full, inp, pos, sections);
    ggml_tensor * out_recr = g.build_layer_only(il_recr, inp, pos, sections);
    check(out_full != nullptr, "layer: full graph output");
    check(out_recr != nullptr, "layer: recurrent graph output");
    if (!out_full || !out_recr) return false;
    check(out_full->ne[1] == 256 && out_recr->ne[1] == 256, "layer: outputs N256");
    check(out_full->ne[0] == qmodel.hparams.n_embd && out_recr->ne[0] == qmodel.hparams.n_embd,
          "layer: output embd width");

    ggml_backend_sched_reset(s.ctx->get_sched());
    if (!ggml_backend_sched_alloc_graph(s.ctx->get_sched(), res.get_gf())) {
        std::printf("FAIL: sched alloc\n");
        return false;
    }
    res.set_inputs(&u);
    s.mctx->graph_compute_start();
    const ggml_status st = s.ctx->graph_compute(res.get_gf(), true);
    s.ctx->synchronize();
    s.mctx->graph_compute_complete(s.ctx->get_sched(), st);
    check(st == GGML_STATUS_SUCCESS, "layer: compute status");
    if (st != GGML_STATUS_SUCCESS) return false;

    auto finite = [](ggml_tensor * t, const char * name) {
        std::vector<float> data(static_cast<size_t>(ggml_nelements(t)));
        ggml_backend_tensor_get(t, data.data(), 0, data.size() * sizeof(float));
        for (const float v : data) if (!std::isfinite(v)) {
            std::printf("FAIL: non-finite output %s\n", name);
            return false;
        }
        std::printf("%s: ne=[%lld,%lld] finite\n", name, (long long) t->ne[0], (long long) t->ne[1]);
        return true;
    };
    check(finite(out_full, (std::string(tag) + " full").c_str()), "full layer finite output");
    check(finite(out_recr, (std::string(tag) + " recurrent").c_str()), "recurrent layer finite output");
    return true;
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
        std::printf("FAIL: CPU backend unavailable\n");
        return 1;
    }
    ggml_backend_dev_t devices[] = {cpu, nullptr};

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.devices = devices;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        llama_backend_free();
        std::printf("FAIL: model load failed\n");
        return 1;
    }
    if (model->arch != LLM_ARCH_QWEN35) {
        std::printf("FAIL: expected qwen35\n");
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

    int l_full = -1;
    int l_recr = -1;
    for (int il = 0; il < (int) model->hparams.n_layer(); ++il) {
        if (model->hparams.is_recr(il)) {
            if (l_recr < 0) l_recr = il;
        } else if (l_full < 0) {
            l_full = il;
        }
    }

    {
        Session s;
        if (!open_session(model, cp, s)) {
            std::printf("FAIL: session init\n");
            close_session(s);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        const auto & qmodel = static_cast<const llama_model_qwen35 &>(*model);
        check(l_full >= 0, "model has full layer");
        check(l_recr >= 0, "model has recurrent layer");
        if (l_full >= 0 && l_recr >= 0) {
            check(run_two_layers(s, qmodel, l_full, l_recr, 0, "U0"), "U0 full+recurrent layer graphs");
            check(run_two_layers(s, qmodel, l_full, l_recr, 1, "U1"), "U1 full+recurrent layer graphs");
        }
        check(s.ctx->n256_lookahead_available() == false, "probe remains false");
        close_session(s);
    }

    llama_model_free(model);
    llama_backend_free();
    std::printf("%s: %d failure(s)\n", failures == 0 ? "OK" : "FAILED", failures);
    return failures == 0 ? 0 : 1;
}
