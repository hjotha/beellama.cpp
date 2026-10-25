// CPU experiment: layer-wise 2xN256 scheduler on real Qwen3.5-4B hybrid.
//
// Two independent contexts/memory states are run in two schedules:
//   serial:    U0 L0..L31, then U1 L0..L31
//   lookahead: U0 L0; for L=1..31 { U1 L-1 + U0 L } issued together; U1 L31
// Each layer is a real qwen35 layer-only graph over the real hybrid memory.
// The lookahead pair uses two separate CPU backend schedulers; the layer input
// is handed over through a real F32 tensor/host buffer. Commit/publish cursors
// enforce U0(L) before U1(L) read/publish. Outputs of every layer are compared
// bitwise between schedules.
//
// This is CPU-only; no GPU/RPC/server/Xbox/benchmark. It does NOT expose the
// production probe (must stay false) and does not alter normal decode.

#include "llama.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-batch.h"
#include "llama-memory.h"
#include "llama-graph.h"
#include "llama-memory-hybrid.h"
#include "llama-kv-cache.h"
#include "llama-memory-recurrent.h"
#include "models.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <atomic>
#include <thread>
#include <sys/stat.h>
#include <vector>

namespace {

const char * kDefaultModel = "/home/hjotha/models/Qwen3.5-4B-Q4_K_M.gguf";
constexpr int kN = 256;
constexpr int kLayers = 32; // Qwen3.5-4B; validated below
int failures = 0;

void check(bool ok, const char * name) {
    std::printf("%s: %s\n", name, ok ? "OK" : "FAIL");
    if (!ok) ++failures;
}

bool file_ok(const char * path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode) && st.st_size > 0;
}

struct Sched {
    ggml_backend_t backend = nullptr;
    ggml_backend_sched_t sched = nullptr;
};

bool sched_init(Sched & s, size_t graph_size) {
    s.backend = ggml_backend_cpu_init();
    if (!s.backend) return false;
    ggml_backend_t backends[] = { s.backend };
    s.sched = ggml_backend_sched_new(backends, nullptr, 1, graph_size, false, true);
    return s.sched != nullptr;
}

void sched_free(Sched & s) {
    if (s.sched) ggml_backend_sched_free(s.sched);
    if (s.backend) ggml_backend_free(s.backend);
    s = {};
}

struct Session {
    llama_context * ctx = nullptr;
    llama_memory_context_ptr mctx;
    llama_batch batch{};
    bool batch_live = false;
    bool applied[2] = {false, false};
    llama_ubatch_handle handle[2];
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
    if (!balloc.init(s.batch, *vocab, mem, model->hparams.n_embd, 1, false)) return false;
    s.mctx = mem->init_batch(balloc, cp.n_ubatch, false);
    if (!s.mctx || llama_memory_status_is_fail(s.mctx->get_status())) return false;
    if (s.mctx->n_ubatches() != 2) return false;
    s.handle[0] = s.mctx->get_ubatch_handle(0);
    s.handle[1] = s.mctx->get_ubatch_handle(1);
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

uint64_t fnv1a(uint64_t h, const void * data, size_t n) {
    const uint8_t * p = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

uint64_t hash_tensor(uint64_t h, ggml_tensor * t) {
    if (!t || !t->buffer) return h;
    std::vector<uint8_t> bytes(static_cast<size_t>(ggml_nbytes(t)));
    ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
    return fnv1a(h, bytes.data(), bytes.size());
}

struct Job;
uint64_t hash_mask(Session & s, Job & job);

struct Job {
    std::unique_ptr<llm_graph_result> res;
    std::unique_ptr<llama_model_qwen35::graph> graph;
    llm_graph_params params;
    ggml_tensor * inp_hidden = nullptr;
    ggml_tensor * out = nullptr;
    llama_ubatch u;
    uint32_t index = 0;
    int il = -1;
    bool recr = false;
    Sched * sched = nullptr;
    uint64_t mask_hash = 0;
    uint64_t state_hash = 0;
};
uint64_t hash_mask(Session & s, Job & job) {
    if (!job.graph || job.recr) return 0;
    llm_graph_input_attn_kv * inp = job.graph->get_attn_input_layer();
    if (!inp) return 0;
    if (!s.mctx->set_ubatch_index(job.index)) return 0;
    const auto * hy = dynamic_cast<const llama_memory_hybrid_context *>(s.mctx.get());
    const auto * attn = hy ? hy->get_attn() : nullptr;
    if (!attn) return 0;
    const auto & si = attn->current_sinfo();
    if (si.idxs.empty()) return 0;

    auto hash_selected = [&](uint64_t h, ggml_tensor * t) {
        if (!t || !t->buffer) return h;
        const size_t elem = ggml_type_size(t->type);
        const size_t row = size_t(t->ne[0]) * elem;
        std::vector<uint8_t> bytes(static_cast<size_t>(ggml_nbytes(t)));
        ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
        for (const auto & cells : si.idxs) {
            for (const uint32_t cell : cells) {
                if (cell >= (uint32_t) t->ne[0]) continue;
                for (int64_t q = 0; q < t->ne[1]; ++q) {
                    const size_t off = size_t(q) * row + size_t(cell) * elem;
                    if (off + elem <= bytes.size()) h = fnv1a(h, bytes.data() + off, elem);
                }
            }
        }
        return h;
    };

    uint64_t h = 1469598103934665603ULL;
    h = hash_tensor(h, inp->get_k_idxs());
    h = hash_tensor(h, inp->get_v_idxs());
    h = hash_selected(h, inp->get_kq_mask());
    h = hash_selected(h, inp->get_standard_kq_mask());
    return h;
}

uint64_t hash_state(Session & s, uint32_t index, int il, bool recr) {
    if (!s.mctx->set_ubatch_index(index)) return 0;
    auto * hy = dynamic_cast<llama_memory_hybrid_context *>(s.mctx.get());
    if (!hy) return 0;
    uint64_t h = 1469598103934665603ULL;
    if (const auto * attn = hy->get_attn()) {
        const auto & si = attn->current_sinfo();
        h = fnv1a(h, &si.s0, sizeof(si.s0));
        h = fnv1a(h, &si.s1, sizeof(si.s1));
        h = fnv1a(h, si.strm.data(), si.strm.size() * sizeof(si.strm[0]));
        for (const auto & v : si.idxs) h = fnv1a(h, v.data(), v.size() * sizeof(v[0]));
        for (const auto & v : si.stage_slots) h = fnv1a(h, v.data(), v.size() * sizeof(v[0]));
        h = fnv1a(h, si.group_stage_slots.data(), si.group_stage_slots.size() * sizeof(si.group_stage_slots[0]));
        if (!recr) {
            ggml_init_params ip = { 1024 * 1024, nullptr, true };
            ggml_context * ctx = ggml_init(ip);
            if (ctx) {
                h = hash_tensor(h, attn->get_k(ctx, il));
                h = hash_tensor(h, attn->get_v(ctx, il));
                h = hash_tensor(h, attn->get_k_tail(ctx, il));
                h = hash_tensor(h, attn->get_v_tail(ctx, il));
                ggml_free(ctx);
            }
        }
    }
    if (recr) {
        if (const auto * rec = hy->get_recr()) {
            const uint32_t head = rec->get_head();
            const uint32_t nrs = rec->get_n_rs();
            h = fnv1a(h, &head, sizeof(head));
            h = fnv1a(h, &nrs, sizeof(nrs));
            h = hash_tensor(h, rec->get_r_l(il));
            h = hash_tensor(h, rec->get_s_l(il));
            h = hash_tensor(h, rec->get_p_l(il));
        }
    }
    return h;
}



bool prepare_job(Session & s, const llama_model_qwen35 & qmodel, Sched & sched,
                 uint32_t index, int il, const std::vector<float> * hidden_in,
                 Job & job) {
    if (il < 0 || il >= (int) s.n_layer || index > 1) return false;
    if (!s.mctx->set_ubatch_index(index)) return false;
    if (!s.applied[index]) {
        if (!s.mctx->apply()) return false;
        s.applied[index] = true;
    }
    if (index == 1 && !s.mctx->layer_commit_can_read(s.handle[1], il)) {
        std::printf("FAIL: U1 L%d tries to read before U0 commit\n", il);
        return false;
    }

    s.mctx->reset_graph_store_slots();
    const llama_ubatch & u = s.mctx->get_ubatch();
    if (u.n_tokens != kN) return false;

    job = {};
    job.u = u;
    job.index = index;
    job.il = il;
    job.recr = qmodel.hparams.is_recr(il);
    job.sched = &sched;
    job.res = std::make_unique<llm_graph_result>(s.ctx->graph_max_nodes(kN));
    job.params = s.ctx->graph_params(job.res.get(), u, s.mctx.get(), LLM_GRAPH_TYPE_DECODER);
    job.params.n_outputs = 1;
    job.params.sched = sched.sched;
    job.params.backend_cpu = sched.backend;

    job.graph = std::make_unique<llama_model_qwen35::graph>(
            qmodel, job.params, llama_model_qwen35::graph::layer_only_tag{!job.recr});
    job.graph->set_hybrid_input_use(!job.recr, job.recr);

    ggml_tensor * inp = nullptr;
    if (il == 0) {
        inp = job.graph->build_input_embd_layer();
    } else {
        if (!hidden_in || hidden_in->size() != (size_t) qmodel.hparams.n_embd * kN) {
            std::printf("FAIL: missing hidden input for L%d\n", il);
            return false;
        }
        job.inp_hidden = ggml_new_tensor_2d(job.res->get_ctx(), GGML_TYPE_F32,
                qmodel.hparams.n_embd, kN);
        ggml_set_input(job.inp_hidden);
        inp = job.inp_hidden;
    }
    ggml_tensor * pos = job.graph->build_input_pos_layer();
    int sections[4];
    std::copy(std::begin(qmodel.hparams.rope_sections),
              std::begin(qmodel.hparams.rope_sections) + 4, sections);
    job.out = job.graph->build_layer_only(il, inp, pos, sections);
    if (!job.out) return false;

    ggml_backend_sched_reset(sched.sched);
    if (!ggml_backend_sched_alloc_graph(sched.sched, job.res->get_gf())) {
        std::printf("FAIL: sched alloc L%d U%d\n", il, index);
        return false;
    }
    job.res->set_inputs(&job.u);
    if (job.inp_hidden) {
        ggml_backend_tensor_set(job.inp_hidden, hidden_in->data(), 0,
                hidden_in->size() * sizeof(float));
    }
    job.mask_hash = hash_mask(s, job);
    return true;
}

bool finish_job(Session & s, Job & job, ggml_status st, std::vector<float> & out) {
    if (!s.mctx->set_ubatch_index(job.index)) return false;
    s.mctx->graph_compute_finish(st);
    s.mctx->graph_compute_complete(job.sched->sched, st);
    if (st != GGML_STATUS_SUCCESS) {
        std::printf("FAIL: compute L%d U%d status %d\n", job.il, job.index, (int) st);
        return false;
    }
    out.resize(static_cast<size_t>(ggml_nelements(job.out)));
    ggml_backend_tensor_get(job.out, out.data(), 0, out.size() * sizeof(float));
    for (const float v : out) if (!std::isfinite(v)) {
        std::printf("FAIL: non-finite L%d U%d\n", job.il, job.index);
        return false;
    }
    const llama_ubatch_handle h = job.index == 0 ? s.handle[0] : s.handle[1];
    if (!s.mctx->layer_commit_append(h, job.il) ||
        !s.mctx->layer_commit_commit(h, job.il) ||
        !s.mctx->layer_commit_publish(h, job.il)) {
        std::printf("FAIL: cursor L%d U%d\n", job.il, job.index);
        return false;
    }
    job.state_hash = hash_state(s, job.index, job.il, job.recr);
    return true;
}

bool complete_job(Session & s, Job & job, std::vector<float> & out) {
    if (!s.mctx->set_ubatch_index(job.index)) return false;
    s.mctx->graph_compute_start();
    const ggml_status st = ggml_backend_sched_graph_compute(job.sched->sched, job.res->get_gf());
    ggml_backend_sched_synchronize(job.sched->sched);
    return finish_job(s, job, st, out);
}

bool complete_pair_parallel(Session & s, Job & a, Job & b,
                            std::vector<float> & oa, std::vector<float> & ob) {
    if (!s.mctx->set_ubatch_index(a.index)) return false;
    s.mctx->graph_compute_start();
    if (!s.mctx->set_ubatch_index(b.index)) return false;
    s.mctx->graph_compute_start();

    std::atomic<int> started{0};
    std::atomic<int> done{0};
    std::atomic<bool> overlap{false};
    ggml_status sa = GGML_STATUS_FAILED;
    ggml_status sb = GGML_STATUS_FAILED;

    auto run = [&](Job & job, ggml_status & st) {
        started.fetch_add(1, std::memory_order_acq_rel);
        while (started.load(std::memory_order_acquire) < 2) {
            std::this_thread::yield();
        }
        if (done.load(std::memory_order_acquire) == 0) {
            overlap.store(true, std::memory_order_release);
        }
        st = ggml_backend_sched_graph_compute(job.sched->sched, job.res->get_gf());
        ggml_backend_sched_synchronize(job.sched->sched);
        done.fetch_add(1, std::memory_order_acq_rel);
    };

    std::thread ta(run, std::ref(a), std::ref(sa));
    std::thread tb(run, std::ref(b), std::ref(sb));
    ta.join();
    tb.join();
    check(overlap.load(), "lookahead: both computes in flight (physical concurrency)");

    if (!finish_job(s, a, sa, oa)) return false;
    if (!finish_job(s, b, sb, ob)) return false;
    return true;
}

struct Result {
    std::vector<std::vector<float>> out; // [ub*n_layer + layer]
    std::vector<uint64_t> state;         // [ub*n_layer + layer]
    std::vector<uint64_t> mask;          // [ub*n_layer + layer]
};

bool run_serial(Session & s, const llama_model_qwen35 & qmodel, Result & r) {
    Sched sched;
    if (!sched_init(sched, s.ctx->graph_max_nodes(kN))) {
        std::printf("FAIL: serial sched init\n");
        return false;
    }
    r.out.resize(size_t(2) * s.n_layer);
    r.state.resize(size_t(2) * s.n_layer);
    r.mask.resize(size_t(2) * s.n_layer);
    bool ok = true;
    for (uint32_t u = 0; u < 2 && ok; ++u) {
        for (int il = 0; il < (int) s.n_layer && ok; ++il) {
            const size_t idx = size_t(u) * s.n_layer + il;
            const std::vector<float> * hin = il == 0 ? nullptr : &r.out[idx - 1];
            Job job;
            ok = prepare_job(s, qmodel, sched, u, il, hin, job) &&
                 complete_job(s, job, r.out[idx]);
            if (ok) { r.state[idx] = job.state_hash; r.mask[idx] = job.mask_hash; }
            if (!ok) std::printf("FAIL: serial U%u L%d\n", u, il);
        }
    }
    sched_free(sched);
    return ok;
}

bool run_lookahead(Session & s, const llama_model_qwen35 & qmodel, Result & r) {
    Sched sched0;
    Sched sched1;
    if (!sched_init(sched0, s.ctx->graph_max_nodes(kN)) ||
        !sched_init(sched1, s.ctx->graph_max_nodes(kN))) {
        sched_free(sched0); sched_free(sched1);
        std::printf("FAIL: lookahead sched init\n");
        return false;
    }
    r.out.resize(size_t(2) * s.n_layer);
    r.state.resize(size_t(2) * s.n_layer);
    r.mask.resize(size_t(2) * s.n_layer);
    bool ok = true;

    {
        Job job;
        ok = prepare_job(s, qmodel, sched0, 0, 0, nullptr, job) &&
             complete_job(s, job, r.out[0]);
        if (ok) { r.state[0] = job.state_hash; r.mask[0] = job.mask_hash; }
        if (!ok) std::printf("FAIL: lookahead U0 L0\n");
    }

    for (int l = 1; l < (int) s.n_layer && ok; ++l) {
        const std::vector<float> * hin0 = &r.out[size_t(0) * s.n_layer + l - 1];
        const std::vector<float> * hin1 = l - 1 == 0 ? nullptr : &r.out[size_t(1) * s.n_layer + l - 2];

        Job j0;
        Job j1;
        // Build U0(L) and U1(L-1) before computing either: legal layer overlap
        // with two separate CPU schedulers/resources.
        if (!prepare_job(s, qmodel, sched0, 0, l, hin0, j0)) { ok = false; break; }
        if (!prepare_job(s, qmodel, sched1, 1, l - 1, hin1, j1)) { ok = false; break; }
        r.mask[size_t(0) * s.n_layer + l] = j0.mask_hash;
        r.mask[size_t(1) * s.n_layer + l - 1] = j1.mask_hash;
        if (!complete_pair_parallel(s, j0, j1,
                r.out[size_t(0) * s.n_layer + l],
                r.out[size_t(1) * s.n_layer + l - 1])) { ok = false; break; }
        r.state[size_t(0) * s.n_layer + l] = j0.state_hash;
        r.state[size_t(1) * s.n_layer + l - 1] = j1.state_hash;
    }

    if (ok) {
        Job last;
        const std::vector<float> * hin = &r.out[size_t(1) * s.n_layer + s.n_layer - 2];
        ok = prepare_job(s, qmodel, sched1, 1, (int) s.n_layer - 1, hin, last) &&
             complete_job(s, last, r.out[size_t(1) * s.n_layer + s.n_layer - 1]);
        if (ok) {
            const size_t idx = size_t(1) * s.n_layer + s.n_layer - 1;
            r.state[idx] = last.state_hash;
            r.mask[idx] = last.mask_hash;
        }
        if (!ok) std::printf("FAIL: lookahead U1 last\n");
    }

    sched_free(sched0);
    sched_free(sched1);
    return ok;
}

bool compare(const Result & a, const Result & b, int n_layer) {
    check(a.out.size() == b.out.size(), "serial/lookahead output counts");
    if (a.out.size() != b.out.size()) return false;
    int mismatch = 0;
    for (int u = 0; u < 2; ++u) {
        for (int l = 0; l < n_layer; ++l) {
            const size_t i = size_t(u) * n_layer + l;
            if (a.out[i].size() != b.out[i].size() ||
                std::memcmp(a.out[i].data(), b.out[i].data(), a.out[i].size() * sizeof(float)) != 0) {
                if (mismatch < 5) std::printf("MISMATCH U%d L%d\n", u, l);
                ++mismatch;
            }
        }
    }
    check(mismatch == 0, "serial vs lookahead bitwise per layer");
    int state_mismatch = 0;
    int mask_mismatch = 0;
    if (a.state.size() == b.state.size() && a.mask.size() == b.mask.size()) {
        for (size_t i = 0; i < a.state.size(); ++i) {
            if (a.state[i] != b.state[i]) {
                if (state_mismatch < 5) std::printf("STATE MISMATCH idx=%zu\n", i);
                ++state_mismatch;
            }
            if (a.mask[i] != b.mask[i]) {
                if (mask_mismatch < 5) std::printf("MASK MISMATCH idx=%zu\n", i);
                ++mask_mismatch;
            }
        }
    } else {
        state_mismatch = mask_mismatch = 1;
    }
    check(state_mismatch == 0, "serial vs lookahead KV/GDN/tail/cell state hashes");
    check(mask_mismatch == 0, "serial vs lookahead attention mask hashes");
    std::printf("layers compared: %d (U0+U1), out_mismatches=%d state_mismatches=%d mask_mismatches=%d\n",
                2 * n_layer, mismatch, state_mismatch, mask_mismatch);
    return mismatch == 0 && state_mismatch == 0 && mask_mismatch == 0;
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
    if (model->hparams.n_layer() != kLayers) {
        std::printf("FAIL: expected %d layers, got %u\n", kLayers, model->hparams.n_layer());
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

    Session serial;
    Session look;
    if (!open_session(model, cp, serial) || !open_session(model, cp, look)) {
        std::printf("FAIL: session init\n");
        close_session(serial);
        close_session(look);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    const auto & qmodel = static_cast<const llama_model_qwen35 &>(*model);
    Result rs;
    Result rl;
    const bool ok_s = run_serial(serial, qmodel, rs);
    const bool ok_l = run_lookahead(look, qmodel, rl);
    check(ok_s, "serial schedule completed");
    check(ok_l, "lookahead schedule completed");
    if (ok_s && ok_l) compare(rs, rl, (int) serial.n_layer);
    check(serial.ctx->n256_lookahead_available() == false, "probe remains false");

    close_session(serial);
    close_session(look);
    llama_model_free(model);
    llama_backend_free();
    std::printf("%s: %d failure(s)\n", failures == 0 ? "OK" : "FAILED", failures);
    return failures == 0 ? 0 : 1;
}
