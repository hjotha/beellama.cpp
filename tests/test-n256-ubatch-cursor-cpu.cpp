// CPU contract test for stable U0/U1 ubatch handles and per-layer commit cursors.
//
// Real qwen35 hybrid memory from Qwen3.5-4B: one 512-token batch produces two
// N256 ubatches. The test checks that both ubatches remain address-stable while
// next() advances the normal cursor, and exercises the new per-layer
// append/commit/publish/epoch API. It does NOT run a model graph, does NOT
// enable the production lookahead, and the probe must remain false.
//
// The lease pool is a fixture for ownership/lifetime/rollback; the ubatches and
// memory context are the real hybrid implementation.

#include "llama.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-batch.h"
#include "llama-memory.h"
#include "ggml-backend.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>

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

struct LeasePool {
    struct Slot {
        bool live = false;
        uint64_t gen = 0;
        int owner = -1;
    };
    std::array<Slot, 8> slots{};

    int alloc(int owner) {
        for (int i = 0; i < (int) slots.size(); ++i) {
            if (!slots[i].live) {
                slots[i].live = true;
                ++slots[i].gen;
                slots[i].owner = owner;
                return i;
            }
        }
        return -1;
    }

    bool release(int idx, uint64_t gen) {
        if (idx < 0 || idx >= (int) slots.size() || !slots[idx].live || slots[idx].gen != gen) {
            return false;
        }
        slots[idx].live = false;
        slots[idx].owner = -1;
        return true;
    }

    bool stale_consume(int idx, uint64_t gen) {
        return idx >= 0 && idx < (int) slots.size() &&
               (!slots[idx].live || slots[idx].gen != gen);
    }

    int live_count() const {
        int n = 0;
        for (const auto & s : slots) n += s.live ? 1 : 0;
        return n;
    }
};

struct Session {
    llama_context * ctx = nullptr;
    llama_memory_context_ptr mctx;
    llama_batch batch{};
    bool batch_live = false;
    uint32_t n_layer = 0;
};

bool open_session(llama_model * model, const llama_context_params & cp, Session & s) {
    s.ctx = llama_new_context_with_model(model, cp);
    if (!s.ctx) return false;
    s.n_layer = model->hparams.n_layer();

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
    if (!balloc.init(s.batch, *vocab, mem, model->hparams.n_embd, 1, false)) {
        return false;
    }
    s.mctx = mem->init_batch(balloc, cp.n_ubatch, false);
    if (!s.mctx || llama_memory_status_is_fail(s.mctx->get_status())) {
        return false;
    }
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

int first_recr_layer(const llama_model * model) {
    for (int il = 0; il < (int) model->hparams.n_layer(); ++il) {
        if (model->hparams.is_recr(il)) return il;
    }
    return -1;
}

int first_full_layer(const llama_model * model) {
    for (int il = 0; il < (int) model->hparams.n_layer(); ++il) {
        if (!model->hparams.is_recr(il)) return il;
    }
    return -1;
}

bool test_layer_cursor(Session & s, int layer, const char * tag) {
    const llama_ubatch_handle h0 = s.mctx->get_ubatch_handle(0);
    const llama_ubatch_handle h1 = s.mctx->get_ubatch_handle(1);
    if (h0.index != 0 || h1.index != 1 || layer < 0 || layer >= (int) s.n_layer) return false;

    check(s.mctx->layer_commit_status(h0, layer) == LLAMA_LAYER_COMMIT_NONE,
          (std::string(tag) + ": U0 initial NONE").c_str());
    check(!s.mctx->layer_commit_can_read(h1, layer),
          (std::string(tag) + ": U1 cannot read before U0 commit").c_str());
    check(!s.mctx->layer_commit_append(h1, layer),
          (std::string(tag) + ": U1 append refused before U0 commit").c_str());

    check(s.mctx->layer_commit_append(h0, layer), (std::string(tag) + ": U0 append").c_str());
    check(s.mctx->layer_commit_status(h0, layer) == LLAMA_LAYER_COMMIT_APPENDED,
          (std::string(tag) + ": U0 APPENDED").c_str());
    check(s.mctx->layer_commit_commit(h0, layer), (std::string(tag) + ": U0 commit").c_str());
    check(s.mctx->layer_commit_status(h0, layer) == LLAMA_LAYER_COMMIT_COMMITTED,
          (std::string(tag) + ": U0 COMMITTED").c_str());
    check(s.mctx->layer_commit_publish(h0, layer), (std::string(tag) + ": U0 publish").c_str());
    check(s.mctx->layer_commit_status(h0, layer) == LLAMA_LAYER_COMMIT_PUBLISHED,
          (std::string(tag) + ": U0 PUBLISHED").c_str());

    check(s.mctx->layer_commit_can_read(h1, layer),
          (std::string(tag) + ": U1 can read after U0 commit").c_str());
    check(s.mctx->layer_commit_append(h1, layer), (std::string(tag) + ": U1 append").c_str());
    check(!s.mctx->layer_commit_publish(h1, layer),
          (std::string(tag) + ": U1 publish refused before commit").c_str());
    check(s.mctx->layer_commit_commit(h1, layer), (std::string(tag) + ": U1 commit").c_str());
    check(s.mctx->layer_commit_publish(h1, layer), (std::string(tag) + ": U1 publish").c_str());
    check(s.mctx->layer_commit_status(h1, layer) == LLAMA_LAYER_COMMIT_PUBLISHED,
          (std::string(tag) + ": U1 PUBLISHED").c_str());

    check(!s.mctx->layer_commit_append(h0, layer), (std::string(tag) + ": duplicate U0 append refused").c_str());
    check(!s.mctx->layer_commit_commit(h0, layer), (std::string(tag) + ": duplicate U0 commit refused").c_str());
    check(!s.mctx->layer_commit_publish(h0, layer), (std::string(tag) + ": duplicate U0 publish refused").c_str());
    check(!s.mctx->layer_commit_append(h1, layer), (std::string(tag) + ": duplicate U1 append refused").c_str());
    check(!s.mctx->layer_commit_commit(h1, layer), (std::string(tag) + ": duplicate U1 commit refused").c_str());
    check(!s.mctx->layer_commit_publish(h1, layer), (std::string(tag) + ": duplicate U1 publish refused").c_str());

    const llama_ubatch_handle stale{h1.index, h1.epoch + 1};
    check(s.mctx->layer_commit_status(stale, layer) == LLAMA_LAYER_COMMIT_INVALID,
          (std::string(tag) + ": stale epoch INVALID").c_str());
    check(!s.mctx->layer_commit_append(stale, layer),
          (std::string(tag) + ": stale append refused").c_str());
    check(!s.mctx->layer_commit_commit(stale, layer),
          (std::string(tag) + ": stale commit refused").c_str());
    check(!s.mctx->layer_commit_publish(stale, layer),
          (std::string(tag) + ": stale publish refused").c_str());
    return true;
}

bool test_failure_rollback(Session & s, int layer) {
    LeasePool leases;
    const int l0 = leases.alloc(100);
    const int l1 = leases.alloc(101);
    check(l0 >= 0 && l1 >= 0 && leases.live_count() == 2,
          "rollback: U0/U1 leases allocated");
    if (l0 < 0 || l1 < 0) return false;
    const uint64_t g0 = leases.slots[l0].gen;
    const uint64_t g1 = leases.slots[l1].gen;

    const llama_ubatch_handle h0 = s.mctx->get_ubatch_handle(0);
    const llama_ubatch_handle h1 = s.mctx->get_ubatch_handle(1);
    check(s.mctx->layer_commit_append(h0, layer), "rollback: U0 append");
    check(s.mctx->layer_commit_commit(h0, layer), "rollback: U0 commit");
    check(s.mctx->layer_commit_publish(h0, layer), "rollback: U0 publish");
    check(s.mctx->layer_commit_append(h1, layer), "rollback: U1 append");
    check(s.mctx->layer_commit_status(h1, layer) == LLAMA_LAYER_COMMIT_APPENDED,
          "rollback: U1 APPENDED");

    check(s.mctx->layer_commit_fail(h0, layer), "rollback: U0 failure recorded");
    const llama_ubatch_handle n0 = s.mctx->get_ubatch_handle(0);
    const llama_ubatch_handle n1 = s.mctx->get_ubatch_handle(1);
    check(n0.epoch != h0.epoch && n1.epoch != h1.epoch, "rollback: epoch advanced");
    check(s.mctx->layer_commit_status(h1, layer) == LLAMA_LAYER_COMMIT_INVALID,
          "rollback: old U1 handle invalid");
    check(s.mctx->layer_commit_status(n1, layer) == LLAMA_LAYER_COMMIT_FAILED,
          "rollback: new U1 handle FAILED");
    check(!s.mctx->layer_commit_can_read(n1, layer), "rollback: U1 cannot read after U0 failure");
    check(!s.mctx->layer_commit_publish(n1, layer), "rollback: U1 cannot publish after U0 failure");
    check(s.mctx->layer_commit_status(n0, layer) == LLAMA_LAYER_COMMIT_FAILED,
          "rollback: U0 failed");

    check(leases.release(l0, g0), "rollback: U0 lease released");
    check(leases.release(l1, g1), "rollback: U1 lease released");
    check(leases.live_count() == 0, "rollback: no live leases");
    check(leases.stale_consume(l0, g0) && leases.stale_consume(l1, g1),
          "rollback: old leases are stale");
    check(!s.mctx->layer_commit_publish(h0, layer), "rollback: old U0 handle still refused");
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

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = 512;
    cp.n_ubatch = 256;
    cp.n_threads = 1;
    cp.n_threads_batch = 1;
    cp.n_seq_max = 1;

    const int l_recr = first_recr_layer(model);
    const int l_full = first_full_layer(model);
    check(l_recr >= 0, "model has a recurrent layer");
    check(l_full >= 0, "model has a full layer");
    check(l_recr != l_full, "recurrent and full layers differ");

    {
        Session s;
        if (!open_session(model, cp, s)) {
            std::printf("FAIL: normal session init\n");
            close_session(s);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        check(s.mctx->n_ubatches() == 2, "normal: two ubatches");
        const llama_ubatch_handle h0 = s.mctx->get_ubatch_handle(0);
        const llama_ubatch_handle h1 = s.mctx->get_ubatch_handle(1);
        const llama_ubatch * p0 = s.mctx->get_ubatch_at(0);
        const llama_ubatch * p1 = s.mctx->get_ubatch_at(1);
        check(p0 && p1 && p0 != p1 && p0->n_tokens == 256 && p1->n_tokens == 256,
              "normal: U0/U1 N256 coexist");
        check(h0.epoch == h1.epoch && h0.epoch != 0, "normal: shared stable epoch");
        const llama_ubatch * p0_before = p0;
        const llama_ubatch * p1_before = p1;
        check(s.mctx->next(), "normal: next advances");
        check(s.mctx->get_ubatch_at(0) == p0_before && s.mctx->get_ubatch_at(1) == p1_before,
              "normal: refs stable after next");
        check(&s.mctx->get_ubatch() == p1_before, "normal: current U1");
        check(test_layer_cursor(s, l_full, "full"), "normal: full cursor");
        check(test_layer_cursor(s, l_recr, "recurrent"), "normal: recurrent cursor");
        check(s.ctx->n256_lookahead_available() == false, "normal: production probe remains false");
        close_session(s);
    }

    {
        Session s;
        if (!open_session(model, cp, s)) {
            std::printf("FAIL: rollback session init\n");
            close_session(s);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        check(test_failure_rollback(s, l_recr), "rollback: recurrent layer");
        close_session(s);
    }

    llama_model_free(model);
    llama_backend_free();
    std::printf("%s: %d failure(s)\n", failures == 0 ? "OK" : "FAILED", failures);
    return failures == 0 ? 0 : 1;
}
