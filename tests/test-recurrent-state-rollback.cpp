// TODO: merge with test-save-load-state.cpp
// TODO: merge with test-state-restore-fragmented.cpp

#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "log.h"
#include "llama-cpp.h"
#include "llama.h"

#include "../src/llama-io.h"
#include "../src/llama-memory.h"
#include "../src/llama-memory-hybrid.h"
#include "../src/llama-memory-recurrent.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <set>
#include <string>
#include <vector>

enum class test_status {
    PASS,
    FAIL,
    SKIP,
};

static const char * test_status_str(test_status status) {
    switch (status) {
        case test_status::PASS: return "\033[1;32mPASS\033[m";
        case test_status::FAIL: return "\033[1;31mFAIL\033[m";
        case test_status::SKIP: return "\033[1;33mSKIP\033[m";
    }
    return "";
}

static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens) {
    common_batch batch(ctx);
    for (uint32_t pos = 0; pos < tokens.size(); ++pos) {
        batch.add(tokens[pos], pos, 0, pos + 1 == tokens.size());
    }
    return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
}

static bool decode_one(llama_context * ctx, llama_token tok, llama_pos pos, llama_seq_id seq = 0) {
    common_batch batch(ctx);
    batch.add(tok, pos, seq, true);
    return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
}

struct cache_buffer_collector : llama_io_write_i {
    std::set<ggml_backend_buffer_t> buffers;
    size_t size = 0;

    void write(const void *, size_t n) override {
        size += n;
    }

    void write_tensor(ggml_tensor * tensor, size_t, size_t n) override {
        buffers.insert(tensor->buffer);
        size += n;
    }

    size_t n_bytes() override {
        return size;
    }
};

static llama_context_ptr init_ctx(llama_model * model, llama_context_params cparams, uint8_t fill) {
    llama_context_ptr ctx{llama_init_from_model(model, cparams)};
    if (!ctx || fill == 0) {
        return ctx;
    }

    // Use a full ubatch so buffer discovery preserves prefill allocation sizes.
    const uint32_t n_tokens = llama_n_ubatch(ctx.get());
    if (!decode_tokens(ctx.get(), std::vector<llama_token>(n_tokens, 0))) {
        return nullptr;
    }
    llama_synchronize(ctx.get());
    cache_buffer_collector collector;
    llama_get_memory(ctx.get())->state_write(collector);
    llama_memory_clear(llama_get_memory(ctx.get()), true);
    if (collector.buffers.empty()) {
        LOG_ERR("%s: no cache buffers found\n", __func__);
        return nullptr;
    }
    for (auto * buffer : collector.buffers) {
        ggml_backend_buffer_clear(buffer, fill);
    }
    return ctx;
}

static llama_context_ptr make_ctx(const common_params & params, llama_model * model, uint8_t fill) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = 8;
    cparams.n_batch   = std::max(cparams.n_batch,  (uint32_t) (cparams.n_rs_seq + 1));
    cparams.n_ubatch  = std::max(cparams.n_ubatch, (uint32_t) (cparams.n_rs_seq + 1));
    return init_ctx(model, cparams, fill);
}

static float logit_diff(float a, float b) {
    return std::isfinite(a) && std::isfinite(b) ? std::fabs(a - b) : std::numeric_limits<float>::infinity();
}

static double nmse(const float * a, const float * b, int n) {
    double mse_ab = 0.0;
    double mse_a0 = 0.0;
    for (int i = 0; i < n; i++) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            return std::numeric_limits<double>::infinity();
        }
        const double diff = (double) a[i] - b[i];
        mse_ab += diff*diff;
        mse_a0 += (double) a[i]*a[i];
    }
    return mse_a0 == 0.0 ? (mse_ab == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()) : mse_ab/mse_a0;
}

// Long Qwen prefill must retain every advertised rollback state, including
// after short decodes carry the prefill history into older snapshot planes.
struct recurrent_state_capture : llama_io_write_i {
    struct row { std::string name; std::vector<float> values; };
    std::vector<row> rows;
    size_t bytes = 0;
    bool valid = true;

    void write(const void *, size_t n) override { bytes += n; }
    void write_tensor(ggml_tensor * tensor, size_t offset, size_t n) override {
        bytes += n;
        if (tensor->type != GGML_TYPE_F32 || n % sizeof(float) != 0) {
            valid = false;
            return;
        }
        row item { tensor->name, std::vector<float>(n / sizeof(float)) };
        ggml_backend_tensor_get(tensor, item.values.data(), offset, n);
        rows.push_back(std::move(item));
    }
    size_t n_bytes() override { return bytes; }
};

static bool gdn_prefill_capture(llama_context * ctx, recurrent_state_capture & capture) {
    auto * hybrid = dynamic_cast<llama_memory_hybrid *>(llama_get_memory(ctx));
    auto * recurrent = hybrid ? hybrid->get_mem_recr() : nullptr;
    if (!recurrent) {
        LOG_ERR("%s: expected a hybrid recurrent memory\n", __func__);
        return false;
    }
    llama_synchronize(ctx);
    recurrent->state_write(capture, 0, LLAMA_STATE_SEQ_FLAGS_NONE);
    return capture.valid && !capture.rows.empty();
}

// Read a rollback plane directly without changing rs_idx or the live sequence.
// state_write selects `cell.src` (or the physical tail cell) in the same way.
static bool gdn_capture_physical_age(llama_context * ctx, uint32_t age, recurrent_state_capture & out) {
    auto * hybrid = dynamic_cast<llama_memory_hybrid *>(llama_get_memory(ctx));
    auto * mem = hybrid ? hybrid->get_mem_recr() : nullptr;
    if (!mem || age > mem->n_rs_seq || mem->cells.empty()) return false;
    const int32_t tail = mem->cells[0].tail;
    if (tail < 0 || size_t(tail) >= mem->cells.size()) return false;
    const auto & cell = mem->cells[tail];
    const int32_t source = cell.src >= 0 ? cell.src : tail;
    if (source < 0 || uint32_t(source) >= mem->size) return false;
    const size_t physical_row = size_t(age)*mem->size + size_t(source);
    llama_synchronize(ctx);
    const auto read_row = [&](ggml_tensor * t) {
        if (!t) return true;
        if (t->type != GGML_TYPE_F32 || physical_row >= size_t(t->ne[1])) return false;
        const size_t n = ggml_row_size(t->type, t->ne[0]);
        if (n % sizeof(float)) return false;
        recurrent_state_capture::row row { t->name, std::vector<float>(n / sizeof(float)) };
        ggml_backend_tensor_get(t, row.values.data(), physical_row*t->nb[1], n);
        out.rows.push_back(std::move(row));
        return true;
    };
    // Match llama_memory_recurrent::state_write_data: R and PLE per layer,
    // followed by all S layers. Metadata bytes are deliberately excluded.
    for (size_t i = 0; i < mem->r_l.size(); ++i) {
        if (!mem->r_l[i]) continue;
        if (!read_row(mem->r_l[i]) || !read_row(mem->p_l[i])) return false;
    }
    for (auto * t : mem->s_l) {
        if (!read_row(t)) return false;
    }
    return !out.rows.empty();
}

static bool gdn_states_exact(const recurrent_state_capture & a, const recurrent_state_capture & b,
                             const char * stage, int short_count, int rollback, bool quiet = false) {
    if (!a.valid || !b.valid || a.rows.size() != b.rows.size()) return false;
    for (size_t i = 0; i < a.rows.size(); ++i) {
        const auto & x = a.rows[i];
        const auto & y = b.rows[i];
        if (x.name != y.name || x.values.size() != y.values.size() ||
                memcmp(x.values.data(), y.values.data(), x.values.size()*sizeof(float)) != 0) {
            if (!quiet) LOG_ERR("%s: %s row=%s short=%d rollback=%d differs bytewise\n", __func__, stage,
                                x.name.c_str(), short_count, rollback);
            return false;
        }
    }
    return true;
}

static test_status test_gdn_long_prefill_rollback(const common_params & params, llama_model * model, uint8_t fill) {
    char arch[64] = {};
    llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
    if (strcmp(arch, "qwen35") != 0) {
        LOG_INF("%s: only qwen35 GDN is covered\n", __func__);
        return test_status::SKIP;
    }
    const char * prefill_env = std::getenv("LLAMA_TEST_GDN_PREFILL_T");
    const int prefill = prefill_env ? std::atoi(prefill_env) : 128;
    if (prefill < 70 || prefill > 200) {
        LOG_ERR("%s: prefill length must be between 70 and 200\n", __func__);
        return test_status::FAIL;
    }
    constexpr int rs = 6;
    auto cp = common_context_params_to_llama(params);
    cp.n_seq_max = 1;
    cp.n_rs_seq = rs;
    cp.n_ctx = 256;
    cp.n_batch = std::max(128, prefill);
    cp.n_ubatch = cp.n_batch;
    cp.kv_unified = false;
    llama_context_ptr ctx = init_ctx(model, cp, fill);
    if (!ctx || llama_n_rs_seq(ctx.get()) != rs) {
        LOG_ERR("%s: could not create context with n_rs_seq=6\n", __func__);
        return test_status::FAIL;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const auto token = [n_vocab](llama_pos pos) {
        return (llama_token) ((7*(uint32_t) pos + 1) % (uint32_t) n_vocab);
    };
    const auto decode = [&](llama_pos begin, llama_pos end, bool outputs) {
        common_batch batch(ctx.get());
        for (llama_pos pos = begin; pos < end; ++pos) {
            batch.add(token(pos), pos, 0, outputs || pos + 1 == end);
        }
        return llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
    };
    bool negative_checked = false;
    for (const int n_short : {0, 2}) {
        for (int rollback = 1; rollback <= rs; ++rollback) {
            const llama_pos end = prefill + n_short;
            const llama_pos cut = end - rollback;
            recurrent_state_capture first_selected, first_replayed;
            std::vector<std::vector<float>> first_logits;
            // Same context, initial clear, tokens, prefill, short decodes,
            // rollback, and replay. No T versus T-r numerical comparison.
            for (int repeat = 0; repeat < 2; ++repeat) {
                llama_memory_clear(llama_get_memory(ctx.get()), true);
                if (!decode(0, prefill, false)) return test_status::FAIL;
                for (llama_pos pos = prefill; pos < end; ++pos) {
                    if (!decode(pos, pos + 1, false)) return test_status::FAIL;
                }
                if (llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0) != end - 1) {
                    LOG_ERR("%s: wrong position before rollback\n", __func__);
                    return test_status::FAIL;
                }
                recurrent_state_capture physical;
                if (!gdn_capture_physical_age(ctx.get(), rollback, physical)) return test_status::FAIL;
                if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, cut, -1) ||
                        llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0) != cut - 1) {
                    LOG_ERR("%s: rollback refused or wrong position (short=%d rollback=%d)\n",
                            __func__, n_short, rollback);
                    return test_status::FAIL;
                }
                recurrent_state_capture selected;
                if (!gdn_prefill_capture(ctx.get(), selected) ||
                        !gdn_states_exact(selected, physical, "physical-vs-logical", n_short, rollback)) {
                    return test_status::FAIL;
                }
                if (!negative_checked) {
                    if (physical.rows[0].values.empty()) return test_status::FAIL;
                    auto * byte = reinterpret_cast<unsigned char *>(physical.rows[0].values.data());
                    *byte ^= 1u; // mutate only the host-side oracle copy
                    const bool caught = !gdn_states_exact(selected, physical, "negative", n_short, rollback, true);
                    *byte ^= 1u;
                    if (!caught) {
                        LOG_ERR("%s: negative control failed to detect corruption\n", __func__);
                        return test_status::FAIL;
                    }
                    negative_checked = true;
                }
                if (repeat == 0) first_selected = std::move(selected);
                else if (!gdn_states_exact(selected, first_selected, "repeated-selection", n_short, rollback)) {
                    return test_status::FAIL;
                }
                // Release copied state before replay, limiting host memory use.
                physical.rows.clear();
                if (!decode(cut, end, true)) return test_status::FAIL;
                for (int i = 0; i < rollback; ++i) {
                    const float * p = llama_get_logits_ith(ctx.get(), i);
                    if (!p) return test_status::FAIL;
                    if (repeat == 0) first_logits.emplace_back(p, p + n_vocab);
                    else if (memcmp(p, first_logits[i].data(), size_t(n_vocab)*sizeof(float)) != 0) {
                        const double diff = nmse(p, first_logits[i].data(), n_vocab);
                        LOG_ERR("%s: repeated logits differ (short=%d rollback=%d token=%d nmse=%g); "
                                "check backend reproducibility and graph mode\n", __func__, n_short, rollback, i, diff);
                        return test_status::FAIL;
                    }
                }
                recurrent_state_capture replayed;
                if (!gdn_prefill_capture(ctx.get(), replayed)) return test_status::FAIL;
                if (repeat == 0) first_replayed = std::move(replayed);
                else if (!gdn_states_exact(replayed, first_replayed, "repeated-replay", n_short, rollback)) {
                    return test_status::FAIL;
                }
            }
        }
    }
    if (!negative_checked) return test_status::FAIL;
    LOG_INF("%s: T=%d, n_rs_seq=6, exact physical/logical and repeated replay passed; negative control caught\n",
            __func__, prefill);
    return test_status::PASS;
}

// A short first decode must work when the rollback reserve exceeds the batch.
// Server warmup exercises this shape before the first speculative request.
static bool test_short_initial_decode(const common_params & params, llama_model * model) {
    llama_context_ptr ctx = make_ctx(params, model, 0);
    if (!ctx) {
        return false;
    }
    const std::vector<llama_token> tokens = { 1, 2 };
    const bool ok = decode_tokens(ctx.get(), tokens) && decode_one(ctx.get(), 3, 2);
    return ok;
}

// Roll back multiple sequences, then replay them in a single batch whose
// per-seq token count exceeds n_ubatch: each seq's replay spans several
// ubatches while its rollback restore is still pending. Compared against a
// reference context that never advanced past the rollback point and decodes
// the identical replay batch.
static test_status test_multi_seq_split_replay(const common_params & params, llama_model * model, uint8_t fill) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    constexpr uint32_t  n_seqs     = 2;
    constexpr uint32_t  n_ubatch   = 16;
    constexpr uint32_t  n_prompt   = 20;
    constexpr uint32_t  n_rollback = 3;
    constexpr uint32_t  n_replay   = 40; // > n_ubatch so each seq spans multiple ubatches
    constexpr llama_pos p0         = n_prompt - n_rollback;

    const auto make_ctx_multi = [&]() {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max  = n_seqs;
        cparams.n_rs_seq   = 8;
        cparams.n_ctx      = 256;
        cparams.n_batch    = 256;
        cparams.n_ubatch   = n_ubatch;
        cparams.kv_unified = false;
        return init_ctx(model, cparams, fill);
    };

    llama_context_ptr ctx_roll = make_ctx_multi();
    llama_context_ptr ctx_ref  = make_ctx_multi();
    if (!ctx_roll || !ctx_ref) {
        LOG_ERR("%s: failed to init multi-seq contexts\n", __func__);
        return test_status::FAIL;
    }

    if (llama_n_rs_seq(ctx_roll.get()) < n_rollback) {
        LOG_INF("%s: skipping because n_rs_seq is too small\n", __func__);
        return test_status::SKIP;
    }

    const auto tok = [&](uint32_t seq, llama_pos pos) {
        return (llama_token) ((7*(uint32_t) pos + 31*seq + 1) % (uint32_t) n_vocab);
    };

    bool ok = true;

    // decode tokens [p_begin, p_end) of seq s, a batch belongs to one context so it is built per call
    const auto decode_range = [&](llama_context * ctx, uint32_t s, llama_pos p_begin, llama_pos p_end) {
        common_batch batch(ctx);
        for (llama_pos pos = p_begin; pos < p_end; ++pos) {
            batch.add(tok(s, pos), pos, (llama_seq_id) s, false);
        }
        return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
    };

    // Keep the anchor at p0 - 1 when rolling back the tail.
    for (uint32_t s = 0; s < n_seqs && ok; ++s) {
        ok = ok && decode_range(ctx_roll.get(), s, 0, p0 - 1);
        ok = ok && decode_range(ctx_ref.get(),  s, 0, p0 - 1);

        ok = ok && decode_range(ctx_roll.get(), s, p0 - 1, (llama_pos) n_prompt);
        ok = ok && decode_range(ctx_ref.get(),  s, p0 - 1, p0);

        llama_memory_t memory = llama_get_memory(ctx_roll.get());
        ok = ok && llama_memory_seq_rm(memory, (llama_seq_id) s, p0, -1);
        llama_pos planned_p0 = -1;
        llama_pos planned_p1 = -1;
        ok = ok && !llama_memory_can_seq_rm(memory, (llama_seq_id) s, p0 - 1, -1);
        ok = ok && !llama_memory_seq_rm_plan(memory, (llama_seq_id) s, p0 - 1, -1, &planned_p0, &planned_p1);
        ok = ok && !llama_memory_seq_rm(memory, (llama_seq_id) s, p0 - 1, -1);
    }
    if (!ok) {
        LOG_ERR("%s: multi-seq prefill/rollback failed\n", __func__);
        return test_status::FAIL;
    }

    // all seqs replay in a single batch
    const auto decode_replay = [&](llama_context * ctx) {
        common_batch batch(ctx);
        for (uint32_t s = 0; s < n_seqs; ++s) {
            for (uint32_t i = 0; i < n_replay; ++i) {
                const llama_pos pos = p0 + (llama_pos) i;
                batch.add(tok(s, pos), pos, (llama_seq_id) s, true);
            }
        }
        return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
    };
    ok = decode_replay(ctx_roll.get());
    ok = ok && decode_replay(ctx_ref.get());
    if (!ok) {
        LOG_ERR("%s: multi-seq replay decode failed\n", __func__);
        return test_status::FAIL;
    }

    // both contexts decode identical batches, so the logits should match;
    // random dummy models can still drift up to ~1.7e-5, so the bound is 1e-4
    constexpr float nmse_eps = 1e-4f;

    float    diff_max  = 0.0f;
    uint32_t seq_first = 0;
    int32_t  pos_first = -1;
    double   nmse_ab   = 0.0;
    double   nmse_a0   = 0.0;
    for (uint32_t i = 0; i < n_seqs*n_replay; ++i) {
        const float * l_roll = llama_get_logits_ith(ctx_roll.get(), i);
        const float * l_ref  = llama_get_logits_ith(ctx_ref.get(),  i);
        if (l_roll == nullptr || l_ref == nullptr) {
            LOG_ERR("%s: missing multi-seq logits at index %u\n", __func__, i);
            return test_status::FAIL;
        }
        for (int t = 0; t < n_vocab; ++t) {
            const float r = l_roll[t];
            const float f = l_ref[t];
            const float diff = logit_diff(r, f);
            if (diff > 0.0f && pos_first < 0) {
                seq_first = i/n_replay;
                pos_first = p0 + (int32_t) (i%n_replay);
            }
            diff_max = std::max(diff_max, diff);
            if (std::isfinite(r) && std::isfinite(f)) {
                const double d = (double) r - f;
                nmse_ab += d*d;
                nmse_a0 += (double) r*r;
            } else {
                nmse_ab = std::numeric_limits<double>::infinity();
                nmse_a0 = 1.0;
            }
        }
    }
    const double nmse_val = nmse_a0 == 0.0 ? (nmse_ab == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()) : nmse_ab/nmse_a0;

    if (nmse_val > nmse_eps) {
        LOG_ERR("%s: multi-seq split replay logits mismatch (max diff %g, nmse %g, first at seq %u pos %d)\n",
                __func__, (double) diff_max, nmse_val, seq_first, pos_first);
        return test_status::FAIL;
    }

    LOG_INF("%s: multi-seq split replay matched (max diff %g, nmse %g)\n", __func__, (double) diff_max, nmse_val);

    // seq-1-only decodes must be independent of seq 0's content: diverge seq 0
    // in ctx_ref only, then compare identical seq-1-only continuations bitwise
    constexpr uint32_t n_tail = 4;

    {
        common_batch batch_tail(ctx_ref.get());
        for (uint32_t i = 0; i < n_tail; ++i) {
            const llama_pos pos = p0 + (llama_pos) (n_replay + i);
            batch_tail.add(tok(0, pos + 7), pos, 0, false);
        }
        ok = llama_process(ctx_ref.get(), LLAMA_PROCESS_TYPE_DECODE, batch_tail.get()) == 0;
    }

    float diff_tail = 0.0f;
    double nmse_tail_ab = 0.0;
    double nmse_tail_a0 = 0.0;
    for (uint32_t i = 0; i < n_tail && ok; ++i) {
        const llama_pos pos = p0 + (llama_pos) (n_replay + i);
        ok = decode_one(ctx_roll.get(), tok(1, pos), pos, 1);
        ok = ok && decode_one(ctx_ref.get(), tok(1, pos), pos, 1);
        if (!ok) {
            break;
        }

        const float * l_roll = llama_get_logits_ith(ctx_roll.get(), 0);
        const float * l_ref  = llama_get_logits_ith(ctx_ref.get(),  0);
        ok = l_roll != nullptr && l_ref != nullptr;
        for (int t = 0; ok && t < n_vocab; ++t) {
            const float r = l_roll[t];
            const float f = l_ref[t];
            diff_tail = std::max(diff_tail, logit_diff(r, f));
            if (std::isfinite(r) && std::isfinite(f)) {
                const double d = (double) r - f;
                nmse_tail_ab += d*d;
                nmse_tail_a0 += (double) r*r;
            } else {
                nmse_tail_ab = std::numeric_limits<double>::infinity();
                nmse_tail_a0 = 1.0;
            }
        }
    }
    const double nmse_tail = nmse_tail_a0 == 0.0 ? (nmse_tail_ab == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()) : nmse_tail_ab/nmse_tail_a0;

    if (!ok || nmse_tail > nmse_eps) {
        LOG_ERR("%s: seq-1-only decode leaked seq 0 state (ok=%d, max diff %g, nmse %g)\n",
                __func__, ok ? 1 : 0, (double) diff_tail, nmse_tail);
        return test_status::FAIL;
    }

    LOG_INF("%s: seq-1-only decode independent of seq 0 (max diff %g, nmse %g)\n", __func__, (double) diff_tail, nmse_tail);
    return test_status::PASS;
}

// Save a rolled-back single-seq state, restore it into fresh and dirty
// contexts, and verify exact logit matches on replay.
static test_status test_rollback(const common_params & params, llama_model * model, uint8_t fill) {
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_ptr ctx_src = make_ctx(params, model, fill);
    llama_context_ptr ctx_dst = make_ctx(params, model, fill);
    if (!ctx_src || !ctx_dst) {
        LOG_ERR("%s: failed to init contexts\n", __func__);
        return test_status::FAIL;
    }

    if (llama_n_rs_seq(ctx_src.get()) == 0) {
        LOG_INF("%s: skipping because n_rs_seq is disabled\n", __func__);
        return test_status::SKIP;
    }

    std::vector<llama_token> tokens;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    } else {
        tokens = common_tokenize(ctx_src.get(), "The quick brown fox jumps over the lazy dog", true);
    }
    const uint32_t n_rs_seq = llama_n_rs_seq(ctx_src.get());
    constexpr uint32_t n_rollback = 3;
    if (n_rs_seq < n_rollback) {
        LOG_INF("%s: skipping because n_rs_seq is too small\n", __func__);
        return test_status::SKIP;
    }
    if (tokens.empty()) {
        LOG_ERR("%s: not enough prompt tokens\n", __func__);
        return test_status::FAIL;
    }
    tokens.resize(n_rs_seq + 1, tokens.back());

    const uint32_t  n_tokens     = tokens.size();
    const llama_pos rollback_pos = (llama_pos) n_tokens - n_rollback;

    // Decode the full prompt on the source, then roll back three positions.
    // Replaying them crosses DSV4's ratio-4 compressor boundary.
    // Rollback leaves the recurrent memory in a snapshot state (rs_idx != 0).
    if (!decode_tokens(ctx_src.get(), tokens)) {
        LOG_ERR("%s: failed to decode prompt\n", __func__);
        return test_status::FAIL;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_src.get()), 0, rollback_pos, -1)) {
        LOG_ERR("%s: rollback failed\n", __func__);
        return test_status::FAIL;
    }

    // Save the rolled-back state and restore it into a fresh context.
    common_prompt_checkpoint ckpt;
    ckpt.update_tgt(ctx_src.get(), 0, 0);
    ckpt.load_tgt(ctx_dst.get(), 0, 0);

    constexpr float nmse_eps = 0.0;
    std::vector<std::vector<float>> logits_src_replay(n_rollback);
    const auto replay_and_compare = [&](const char * mode) {
        for (uint32_t i = 0; i < n_rollback; ++i) {
            const llama_pos pos = rollback_pos + i;
            if (!decode_one(ctx_src.get(), tokens[pos], pos) ||
                !decode_one(ctx_dst.get(), tokens[pos], pos)) {
                LOG_ERR("%s: %s replay failed at position %d\n", __func__, mode, pos);
                return false;
            }

            const float * logits_src = llama_get_logits_ith(ctx_src.get(), 0);
            const float * logits_dst = llama_get_logits_ith(ctx_dst.get(), 0);
            if (logits_src == nullptr || logits_dst == nullptr) {
                LOG_ERR("%s: missing %s logits at position %d\n", __func__, mode, pos);
                return false;
            }

            logits_src_replay[i].assign(logits_src, logits_src + n_vocab);
            const double nmse_val = nmse(logits_src, logits_dst, n_vocab);
            int token_first = -1;
            for (int token = 0; token < n_vocab; ++token) {
                if (logit_diff(logits_src[token], logits_dst[token]) > 0.0f && token_first < 0) {
                    token_first = token;
                }
            }
            if (nmse_val > nmse_eps) {
                LOG_ERR("%s: %s logits mismatch at position %d, first token %d, nmse %g\n",
                        __func__, mode, pos, token_first, nmse_val);
                return false;
            }
        }
        return true;
    };
    if (!replay_and_compare("full")) {
        return test_status::FAIL;
    }

    // TODO: this test is invalid because RS rollback is only correct once after a ubatch with more than n_rs_seq tokens
    //       this is not the case here. add asserts and guardrails to prevent such attempts
    //if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, rollback_pos, -1) ||
    //    !llama_memory_seq_rm(llama_get_memory(ctx_dst), 0, rollback_pos, -1)) {
    //    fprintf(stderr, "%s : partial rollback failed\n", __func__);
    //    return 1;
    //}

    //constexpr llama_state_seq_flags partial_flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    //common_prompt_checkpoint ckpt_partial;
    //ckpt_partial.update_tgt(ctx_src, 0, partial_flags);
    //ckpt_partial.load_tgt(ctx_dst, 0, partial_flags);

    //if (!replay_and_compare("partial")) {
    //    return 1;
    //}

    // Repeat the load into a context that already has its own rollback state:
    // groups 1..n_rs_seq hold a different prompt's history, and rs_idx[0] is
    // non-zero at load time. The restore must wipe that state and still match.
    llama_context_ptr ctx_dirty = make_ctx(params, model, fill);
    if (!ctx_dirty) {
        LOG_ERR("%s: failed to init dirty ctx\n", __func__);
        return test_status::FAIL;
    }

    std::vector<llama_token> noise = tokens;
    for (auto & t : noise) {
        t = (t + 1) % n_vocab;
        if (t < 0) {
            t = 0;
        }
    }
    if (!decode_tokens(ctx_dirty.get(), noise)) {
        LOG_ERR("%s: dirty prompt decode failed\n", __func__);
        return test_status::FAIL;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_dirty.get()), 0, rollback_pos, -1)) {
        LOG_ERR("%s: dirty rollback failed\n", __func__);
        return test_status::FAIL;
    }

    ckpt.load_tgt(ctx_dirty.get(), 0, 0);

    for (uint32_t i = 0; i < n_rollback; ++i) {
        const llama_pos pos = rollback_pos + i;
        if (!decode_one(ctx_dirty.get(), tokens[pos], pos)) {
            LOG_ERR("%s: dirty replay failed at position %d\n", __func__, pos);
            return test_status::FAIL;
        }

        const float * logits_dirty = llama_get_logits_ith(ctx_dirty.get(), 0);
        if (logits_dirty == nullptr) {
            LOG_ERR("%s: missing dirty logits at position %d\n", __func__, pos);
            return test_status::FAIL;
        }

        const double nmse_dirty = nmse(logits_src_replay[i].data(), logits_dirty, n_vocab);
        int token_first = -1;
        for (int token = 0; token < n_vocab; ++token) {
            if (logit_diff(logits_src_replay[i][token], logits_dirty[token]) > 0.0f && token_first < 0) {
                token_first = token;
            }
        }
        if (nmse_dirty > nmse_eps) {
            LOG_ERR("%s: dirty-ctx logits mismatch at position %d, first token %d, nmse %g\n",
                    __func__, pos, token_first, nmse_dirty);
            return test_status::FAIL;
        }
    }

    LOG_INF("%s: recurrent rollback checkpoint restored successfully\n", __func__);
    return test_status::PASS;
}

// Decode a prompt into seq 0, share its cells with a second sequence, then keep
// decoding both.
static test_status test_shared_seq_reserve(const common_params & params, llama_model * model, uint8_t fill) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    // these archs reserve the final pp graph with n_seqs = 1, so every multi-seq
    // graph has a different layout and re-reserves by design
    // see [TAG_RESERVE_DIAG_DECAY] in llama-context.cpp
    char arch_str[64] = {};
    llama_model_meta_val_str(model, "general.architecture", arch_str, sizeof(arch_str));
    if (strcmp(arch_str, "kimi-linear") == 0 || strcmp(arch_str, "minimax-01") == 0) {
        LOG_INF("%s: skipping %s, its reserve uses n_seqs = 1\n", __func__, arch_str);
        return test_status::SKIP;
    }

    constexpr uint32_t n_seqs     = 2;
    constexpr uint32_t n_prompt   = 128;
    constexpr uint32_t n_continue = 32;

    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max  = n_seqs;
    cparams.n_ctx      = 512;
    cparams.n_batch    = 256;
    cparams.n_ubatch   = 64;
    cparams.kv_unified = true; // only a unified cache shares cells on seq_cp

    llama_context_ptr ctx = init_ctx(model, cparams, fill);
    if (!ctx) {
        LOG_ERR("%s: failed to init context\n", __func__);
        return test_status::FAIL;
    }

    const auto tok = [&](uint32_t seq, llama_pos pos) {
        return (llama_token) ((7*(uint32_t) pos + 31*seq + 1) % (uint32_t) n_vocab);
    };

    {
        common_batch batch(ctx.get());
        for (llama_pos pos = 0; pos < (llama_pos) n_prompt; ++pos) {
            batch.add(tok(0, pos), pos, 0, false);
        }
        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("%s: prompt decode failed\n", __func__);
            return test_status::FAIL;
        }
    }

    // this is what llama-batched-bench does for -pps
    llama_memory_seq_cp(llama_get_memory(ctx.get()), 0, 1, -1, -1);

    for (uint32_t i = 0; i < n_continue; ++i) {
        const llama_pos pos = (llama_pos) (n_prompt + i);

        common_batch batch(ctx.get());
        for (uint32_t s = 0; s < n_seqs; ++s) {
            batch.add(tok(s, pos), pos, (llama_seq_id) s, true);
        }
        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("%s: shared-seq decode failed at step %u\n", __func__, i);
            return test_status::FAIL;
        }

        for (uint32_t s = 0; s < n_seqs; ++s) {
            const float * logits = llama_get_logits_ith(ctx.get(), (int) s);
            if (logits == nullptr) {
                LOG_ERR("%s: missing shared-seq logits at index %u\n", __func__, s);
                return test_status::FAIL;
            }
            for (int t = 0; t < n_vocab; ++t) {
                if (!std::isfinite(logits[t])) {
                    LOG_ERR("%s: non-finite shared-seq logit at step %u, seq %u, index %d\n", __func__, i, s, t);
                    return test_status::FAIL;
                }
            }
        }
    }

    LOG_INF("%s: shared-seq decode succeeded (%u tokens after seq_cp)\n", __func__, n_continue*n_seqs);
    return test_status::PASS;
}

static test_status merge_status(test_status a, test_status b) {
    if (a == test_status::FAIL || b == test_status::FAIL) {
        return test_status::FAIL;
    }
    if (a == test_status::PASS || b == test_status::PASS) {
        return test_status::PASS;
    }
    return test_status::SKIP;
}

struct test_results {
    test_status rollback = test_status::SKIP;
    test_status replay   = test_status::SKIP;
    test_status shared   = test_status::SKIP;
};

// Run every test for an initialized model over both cache fills.
static test_results run_tests(const common_params & params, llama_model * model) {
    test_results res;
    // Opt-in for the GDN prefill path; avoids multiplying runtime in the broad
    // model matrix while allowing a focused CUDA/CPU regression gate.
    if (std::getenv("LLAMA_TEST_GDN_PREFILL_ROLLBACK_ONLY")) {
        res.rollback = test_gdn_long_prefill_rollback(params, model, 0);
        if (res.rollback == test_status::PASS) {
            res.rollback = test_gdn_long_prefill_rollback(params, model, 0x3e);
        }
        return res;
    }
    for (uint8_t fill : { 0, 0x3e }) {
        LOG_INF("%s: testing with cache fill 0x%02x\n", __func__, fill);
        const test_status rb = test_rollback(params, model, fill);
        const test_status rp = test_multi_seq_split_replay(params, model, fill);
        const test_status ss = test_shared_seq_reserve(params, model, fill);
        res.rollback = merge_status(res.rollback, rb);
        res.replay   = merge_status(res.replay,   rp);
        res.shared   = merge_status(res.shared,   ss);
        if (rb == test_status::FAIL || rp == test_status::FAIL || ss == test_status::FAIL) {
            break;
        }
    }
    return res;
}

// Run the tests for a single model file.
// Returns the per-test statuses.
static test_results run_tests_for_model(const std::string & model_path, const struct common_params & base_params) {
    struct common_params params = base_params;
    params.model.path = model_path;

    auto llama_init = common_init_from_params(params, true);
    auto * model = llama_init->model();

    if (model == nullptr) {
        LOG_ERR("%s: failed to init model '%s'\n", __func__, model_path.c_str());
        // a model that cannot be loaded is a failure, not a skip
        return { test_status::FAIL, test_status::FAIL, test_status::FAIL };
    }

    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        LOG_INF("%s: skipping for non-recurrent model\n", __func__);
        return {};
    }

    if (!test_short_initial_decode(params, model)) {
        LOG_ERR("%s: short initial recurrent decode failed\n", __func__);
        return { test_status::FAIL, test_status::FAIL, test_status::FAIL };
    }
    return run_tests(params, model);
}

static void print_usage(int /* argc */, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n  %s -m your_model.gguf\n", argv[0]);
    LOG("\n  %s --models tests/test-models\n", argv[0]);
    LOG("\n");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;

    common_init();

    // extract our own --models DIR option before handing the rest to the common arg parser
    std::string models_dir;
    std::vector<char *> filtered_argv;
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--models") == 0) {
            if (i + 1 >= argc) {
                LOG_ERR("%s: --models requires a directory argument\n", __func__);
                return 1;
            }
            models_dir = argv[i + 1];
            i++;
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }
    filtered_argv.push_back(nullptr);
    const int fargc = (int)filtered_argv.size() - 1;

    // in --models mode there is no single model; set a placeholder so the common parser's
    // "--model is required" check passes (each model is set individually inside the loop)
    if (!models_dir.empty()) {
        params.model.path = models_dir;
    }

    if (!common_params_parse(fargc, filtered_argv.data(), params, LLAMA_EXAMPLE_COMMON, print_usage)) {
        return 1;
    }

    llama_backend_init();

    if (!models_dir.empty()) {
        // run every test over each dummy model in the directory
        if (!std::filesystem::exists(models_dir) || !std::filesystem::is_directory(models_dir)) {
            LOG_ERR("%s: models directory '%s' does not exist\n", __func__, models_dir.c_str());
            return 1;
        }

        std::vector<std::string> models;
        for (const auto & entry : std::filesystem::directory_iterator(models_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
                models.push_back(entry.path().string());
            }
        }
        std::sort(models.begin(), models.end());

        if (models.empty()) {
            LOG_ERR("%s: no .gguf models found in '%s'\n", __func__, models_dir.c_str());
            return 1;
        }

        size_t name_width = 5; // "Model"
        for (const auto & model_path : models) {
            name_width = std::max(name_width, std::filesystem::path(model_path).filename().string().size());
        }

        // silence everything but the table itself (LOG has verbosity LOG_LEVEL_OUTPUT = 0)
        common_log_set_verbosity_thold(0);

        LOG("%-*s  %-8s  %-11s  %s\n", (int) name_width, "Model", "rollback", "split replay", "shared seq");
        common_log_flush(common_log_main());

        size_t n_pass[3] = { 0, 0, 0 };
        size_t n_skip[3] = { 0, 0, 0 };
        size_t n_fail[3] = { 0, 0, 0 };
        for (const auto & model_path : models) {
            const auto name = std::filesystem::path(model_path).filename().string();

            LOG("%-*s", (int) name_width, name.c_str());

            const test_results res = run_tests_for_model(model_path, params);

            // all status strings have the same raw length, so the columns line up;
            // pad the first status to the width of the "rollback" header + separator
            LOG("  %s      %s         %s", test_status_str(res.rollback), test_status_str(res.replay), test_status_str(res.shared));
            LOG("\n");
            common_log_flush(common_log_main());

            const test_status all[3] = { res.rollback, res.replay, res.shared };
            for (int t = 0; t < 3; ++t) {
                switch (all[t]) {
                    case test_status::PASS: n_pass[t]++; break;
                    case test_status::FAIL: n_fail[t]++; break;
                    case test_status::SKIP: n_skip[t]++; break;
                }
            }
        }

        common_log_set_verbosity_thold(LOG_DEFAULT_LLAMA);
        common_log_flush(common_log_main());

        LOG_INF("%s: rollback:     %zu passed, %zu skipped, %zu failed (of %zu)\n",
                __func__, n_pass[0], n_skip[0], n_fail[0], models.size());
        LOG_INF("%s: split replay: %zu passed, %zu skipped, %zu failed (of %zu)\n",
                __func__, n_pass[1], n_skip[1], n_fail[1], models.size());
        LOG_INF("%s: shared seq:   %zu passed, %zu skipped, %zu failed (of %zu)\n",
                __func__, n_pass[2], n_skip[2], n_fail[2], models.size());

        return (n_fail[0] + n_fail[1] + n_fail[2]) == 0 ? 0 : 1;
    }

    // single-model mode
    const test_results res = run_tests_for_model(params.model.path, params);

    return (res.rollback == test_status::FAIL || res.replay == test_status::FAIL || res.shared == test_status::FAIL) ? 1 : 0;
}
