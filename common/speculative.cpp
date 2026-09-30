#include "speculative.h"

#include "../src/llama-ext.h"
#include "common.h"
#include "dflash-remote.h"
#include "ggml-cpp.h"
#include "ggml.h"
#include "llama.h"

#include <cmath>
#include <limits>
#ifdef LLAMA_DSPARK_MARKOV_CUDA
#    include "dspark-markov.h"
#endif
#ifdef LLAMA_DSPARK_MARKOV_METAL
#    include "dspark-markov-metal.h"
#endif
#ifdef LLAMA_DSPARK_MARKOV_BLAS
#    include <cblas.h>
#endif
#include "../src/llama-ext.h"  // staging API: llama_set_embeddings_nextn / llama_get_embeddings_nextn_ith (used by MTP)
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"
#include "dflash-pipeline-candidate.h"
#include "dflash-shadow-observation.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>

int64_t common_spec_prof_sync_us = 0;
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <fstream>
#include <future>
#include <iomanip>
#include <map>
#include <limits>
#include <random>

#define SPC_DBG(fmt, ...) LOG_DBG("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_TRC(fmt, ...) LOG_TRC("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_INF(fmt, ...) LOG_INF("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_WRN(fmt, ...) LOG_WRN("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_ERR(fmt, ...) LOG_ERR("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_CNT(fmt, ...) LOG_CNT(""              fmt,               __VA_ARGS__)

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

const std::map<std::string, common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft-simple",  COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE},
    {"draft-eagle3",  COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3},
    {"draft-mtp",     COMMON_SPECULATIVE_TYPE_DRAFT_MTP},
    {"draft-dflash",  COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH},
    {"draft-dspark",  COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK},
    {"ngram-simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram-map-k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram-map-k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram-mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram-cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE}
};

static std::string common_speculative_get_devices_str(const std::vector<ggml_backend_dev_t> & devices) {
    std::string result;
    for (size_t i = 0; i < devices.size(); i++) {
        if (devices[i] == nullptr) {
            continue;
        }
        if (!result.empty()) result += ", ";
        result += ggml_backend_dev_name(devices[i]);
    }
    return result.empty() ? "default" : result;
}

struct common_speculative_config {
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(common_speculative_type t,
            const common_params_speculative & p = common_params_speculative{}) : type(t), params(p) {}
};

static bool common_speculative_are_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const auto vocab_type_tgt = llama_vocab_type(vocab_tgt);
    SPC_DBG("vocab_type tgt: %d\n", vocab_type_tgt);

    const auto vocab_type_dft = llama_vocab_type(vocab_dft);
    SPC_DBG("vocab_type dft: %d\n", vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        SPC_WRN("draft model vocab type must match target model to use speculation but "
                "vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        (llama_vocab_get_add_bos(vocab_tgt) && llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft))) {
        SPC_WRN("draft model bos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                llama_vocab_get_add_bos(vocab_tgt), llama_vocab_get_add_bos(vocab_dft),
                llama_vocab_bos(vocab_tgt), llama_vocab_bos(vocab_dft));
        return false;
    }

    if (llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        (llama_vocab_get_add_eos(vocab_tgt) && llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft))) {
        SPC_WRN("draft model eos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                llama_vocab_get_add_eos(vocab_tgt), llama_vocab_get_add_eos(vocab_dft),
                llama_vocab_eos(vocab_tgt), llama_vocab_eos(vocab_dft));
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            SPC_DBG("draft model vocab must closely match target model to use speculation but "
                    "target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                SPC_DBG("draft model vocab must match target model to use speculation but "
                        "token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

using common_speculative_draft_params_vec = std::vector<common_speculative_draft_params>;

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_impl
struct common_speculative_impl {
    const common_speculative_type type;

    uint32_t n_seq;
    int32_t n_max; // maximum draft length after implementation-specific limits

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.

    std::vector<size_t> n_acc_tokens_per_pos; // number of tokens accepted per draft position.
    std::vector<size_t> n_gen_tokens_per_pos; // number of tokens proposed per draft position (profiling).

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    common_speculative_impl(common_speculative_type type, uint32_t n_seq, int32_t n_max) : type(type), n_seq(n_seq), n_max(n_max) {}

    virtual ~common_speculative_impl() = default;

    virtual bool need_embd() const { return false; }

    virtual bool need_embd_nextn() const { return false; }

    virtual bool need_embd_capture() const { return false; }

    virtual bool stage_test_ctx_feat(llama_seq_id, const float *, int64_t, int64_t, const int32_t *) { return false; }

    virtual void begin(llama_seq_id seq_id, const llama_tokens & prompt) = 0;

    virtual bool process(const llama_batch & batch) = 0;

    virtual bool bootstrap(const llama_batch & /*batch*/, uint64_t /*decode_id*/) { return false; }
    virtual bool is_ready(llama_seq_id /*seq_id*/, llama_pos /*next_pos*/) const { return true; }

    virtual void draft(common_speculative_draft_params_vec & dparams) = 0;

    virtual void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) = 0;

    virtual bool adaptive_dm_supported() const { return false; }
    virtual bool draft_memory_is_shared() const { return false; }

    // (optional) serialize/restore per-seq internal state (e.g. eagle3's deferred boundary).
    virtual bool get_state(llama_seq_id /*seq_id*/, std::vector<uint8_t> & /*data*/) const { return false; }
    virtual bool validate_state(llama_seq_id /*seq_id*/, const std::vector<uint8_t> & data) const {
        return data.empty();
    }
    virtual bool set_state(llama_seq_id /*seq_id*/, const std::vector<uint8_t> & data) {
        return data.empty();
    }
    virtual bool set_state(llama_seq_id /*seq_id*/, const std::vector<uint8_t> & /*data*/, llama_pos /*expected_pos*/) { return false; }
};

struct common_speculative_impl_draft_dspark : public common_speculative_impl {
    common_params_speculative_draft params;  // reuses the draft-model params slot (ctx_tgt/ctx_dft)

    int64_t n_embd        = 0;
    int64_t n_vocab       = 0;  // from token_embd's own shape; dspark has no tokenizer/vocab of its own
    int64_t n_capture     = 0;  // target_layer_ids count
    int64_t n_embd_cap    = 0;  // n_capture * n_embd (raw pre-fc tap width)
    int32_t block_size    = 0;
    int32_t mask_token_id = 0;
    int64_t draft_window  = 0;

    // vanilla Markov head weights, host-resident (loaded once at construction
    // via llama_model_dspark_get_markov): [n_vocab * n_rank] row-major, rank
    // fastest-varying. See the resample loop in draft() for how these are used.
    std::vector<float>                                                         markov_w1;
    std::vector<float>                                                         markov_w2;
    std::vector<float>                                                         markov_bias;
    int64_t                                                                    markov_rank = 0;
    bool                                                                       has_markov  = false;
    std::vector<std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>> graph_samplers;
    int64_t                                                                    markov_time_us    = 0;
    int64_t                                                                    markov_calls      = 0;
    bool                                                                       dcut_enabled      = false;
    int32_t                                                                    draft_rows        = 0;
    int32_t                                                                    correction_rows   = 0;
    bool                                                                       correction_prefix = false;
    float                                                                      dcut_costs[4]     = {};
#ifdef LLAMA_DSPARK_MARKOV_CUDA
    std::unique_ptr<dspark_markov_cuda, decltype(&dspark_markov_cuda_free)> markov_cuda{ nullptr,
                                                                                         dspark_markov_cuda_free };
#endif
#ifdef LLAMA_DSPARK_MARKOV_METAL
    std::unique_ptr<dspark_markov_metal, decltype(&dspark_markov_metal_free)> markov_metal{ nullptr,
                                                                                            dspark_markov_metal_free };
#endif

    llama_batch batch;  // ctx_dft batch; no embd channel -- context features are
                        // staged out-of-band via llama_set_dspark_ctx, not batch.embd

    // --- per-seq persistent state --------------------------------------
    // Absolute end of committed draft context, not the number of resident rows.
    std::vector<int64_t> n_cache;

    // growing buffer of not-yet-consumed target-tap context rows, accumulated
    // across process() calls since the last draft() call drained them. Rows
    // are contiguous and strictly increasing in position (asserted in draft()).
    std::vector<std::vector<float>>   ctx_feat;  // [n_seq][rows * n_embd_cap]
    std::vector<std::vector<int32_t>> ctx_pos;   // [n_seq][rows]

    // how many of the currently-buffered rows were appended since the last
    // accept() call. accept() trims exactly this many down to n_accepted+1,
    // discarding the rejected tail, leaving any earlier
    // (already-accepted-but-not-yet-drained) rows untouched. This is what
    // lets dspark's context stay correct even on rounds where a DIFFERENT
    // implementation's draft is the one that gets verified: process() runs
    // (and accumulates) unconditionally for every registered impl, and
    // accept() runs on every impl too (is_other=true for the ones that didn't
    // draft), so dspark's own bookkeeping tracks the real generation stream
    // regardless of who proposed a given round's tokens.
    std::vector<int64_t> rows_since_accept;

    // process()'s per-seq contiguous-range bookkeeping (mirrors draft-mtp).
    std::vector<int32_t> i_batch_beg;
    std::vector<int32_t> i_batch_end;

    common_speculative_impl_draft_dspark(const common_params_speculative & params, uint32_t n_seq) :
        common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK, n_seq, params.draft.n_max),
        params(params.draft) {
        auto * ctx_dft = this->params.ctx_dft;
        auto * ctx_tgt = this->params.ctx_tgt;
        GGML_ASSERT(ctx_dft && ctx_tgt && "dspark requires ctx_tgt and ctx_dft to be set");

        const llama_model * model_dft = llama_get_model(ctx_dft);

        llama_dspark_meta meta;
        if (!llama_model_dspark_get_meta(model_dft, &meta)) {
            throw std::runtime_error(
                "dspark: ctx_dft's model does not look like a dspark drafter (missing dspark.*.block_size KV)");
        }

        n_embd        = meta.n_embd;
        n_vocab       = meta.n_vocab;
        n_capture     = meta.n_capture;
        n_embd_cap    = meta.n_embd_cap;
        block_size    = meta.block_size;
        mask_token_id = meta.mask_token_id;
        markov_rank   = meta.markov_rank;

        if (const char * value = std::getenv("DSPARK_DRAFT_WINDOW")) {
            char * end   = nullptr;
            draft_window = std::strtol(value, &end, 10);
            if (end == value || *end || draft_window < 0 || draft_window > std::numeric_limits<llama_pos>::max()) {
                throw std::runtime_error("DSPARK_DRAFT_WINDOW must be a nonnegative position count");
            }
        }
        LOG_INF("%s: draft_window=%lld (0=full prefix; target cache unchanged)\n", __func__, (long long) draft_window);

        has_markov =
            !meta.graph_corrected && markov_rank > 0 && llama_model_dspark_get_markov(model_dft, markov_w1, markov_w2);
        draft_rows      = block_size;
        correction_rows = block_size;
        if (const char * value = std::getenv("LLAMA_DSPARK_CORRECTION_PREFIX")) {
            char *     end  = nullptr;
            const long keep = std::strtol(value, &end, 10);
            if (end == value || *end || keep < 1 || keep > block_size || !meta.graph_corrected || n_seq != 1 ||
                params.draft.n_max != keep) {
                throw std::runtime_error(
                    "DSpark correction prefix requires graph correction, one sequence, and matching n_max");
            }
            correction_rows   = (int32_t) keep;
            correction_prefix = true;
        }
        for (auto item : { std::make_pair("DSPARK_FORWARD_ROWS", &draft_rows),
                           std::make_pair("DSPARK_CORRECTION_ROWS", &correction_rows) }) {
            if (const char * value = std::getenv(item.first)) {
                char *     end  = nullptr;
                const long rows = std::strtol(value, &end, 10);
                if (end == value || *end || rows < 1 || rows > block_size || !has_markov) {
                    throw std::runtime_error("DSpark experimental row count requires legacy Markov and 1..block_size");
                }
                *item.second = (int32_t) rows;
            }
        }
        if (correction_rows > draft_rows) {
            throw std::runtime_error("DSpark correction rows exceed forward rows");
        }
        LOG_INF("dspark: full_block=%d forward_rows=%d correction_rows=%d\n", block_size, draft_rows,
                correction_rows);
        if (n_vocab > std::numeric_limits<int>::max()) {
            throw std::runtime_error("dspark: vocab size exceeds cblas integer range");
        }
        markov_bias.resize((size_t) n_vocab);

        const char * cuda_mode = std::getenv("LLAMA_DSPARK_MARKOV_CUDA");
        if (const char * value = std::getenv("DSPARK_DCUT_COSTS")) {
            const char * next = value;
            for (int i = 0; i < 4; ++i) {
                char * end    = nullptr;
                dcut_costs[i] = std::strtof(next, &end);
                if (end == next || !std::isfinite(dcut_costs[i]) || dcut_costs[i] <= 0 ||
                    (i < 3 ? *end != ',' : *end != '\0')) {
                    throw std::runtime_error(
                        "DSPARK_DCUT_COSTS requires four positive finite comma-separated round costs");
                }
                next = end + (i < 3);
            }
            if (n_seq != 1 || block_size != 4 || !has_markov || !cuda_mode || std::strcmp(cuda_mode, "1")) {
                throw std::runtime_error("Experimental D-cut requires c1 block-4 legacy CUDA Markov");
            }
            dcut_enabled = true;
            LOG_INF(
                "dspark: dcut_policy=device_probability_cost_v1 costs=%.9g,%.9g,%.9g,%.9g "
                "score=uncalibrated_draft_probability\n",
                dcut_costs[0], dcut_costs[1], dcut_costs[2], dcut_costs[3]);
        }
        const char * metal_mode = std::getenv("LLAMA_DSPARK_MARKOV_METAL");
        if (metal_mode && std::strcmp(metal_mode, "0") && std::strcmp(metal_mode, "1")) {
            throw std::runtime_error("LLAMA_DSPARK_MARKOV_METAL must be 0 or 1");
        }
        const bool want_metal = metal_mode && std::strcmp(metal_mode, "1") == 0;
        if (want_metal && cuda_mode && std::strcmp(cuda_mode, "1") == 0) {
            throw std::runtime_error("Choose either Metal or CUDA Markov, not both");
        }
        if (cuda_mode && std::strcmp(cuda_mode, "0") && std::strcmp(cuda_mode, "1")) {
            throw std::runtime_error("LLAMA_DSPARK_MARKOV_CUDA must be 0 or 1");
        }
        if (cuda_mode && std::strcmp(cuda_mode, "1") == 0) {
            if (!has_markov) {
                throw std::runtime_error(
                    "CUDA Markov requested without a legacy Markov head; graph-corrected drafters use their own path");
            }
#ifdef LLAMA_DSPARK_MARKOV_CUDA
            markov_cuda.reset(
                dspark_markov_cuda_init(markov_w1.data(), markov_w2.data(), n_vocab, markov_rank, mask_token_id));
            if (!markov_cuda) {
                throw std::runtime_error("CUDA Markov initialization failed; refusing CPU fallback");
            }
            LOG_INF("dspark: correction_backend=CUDA_MARKOV\n");
#else
            throw std::runtime_error("CUDA Markov requested but not compiled");
#endif
        } else if (!want_metal) {
            LOG_INF("dspark: correction_backend=%s\n", meta.graph_corrected ? "DRAFT_GRAPH" : "HOST");
        }
        if (want_metal) {
            if (!has_markov) {
                throw std::runtime_error(
                    "Metal Markov requires a legacy Markov head; graph-corrected drafters use their own path");
            }
#ifdef LLAMA_DSPARK_MARKOV_METAL
            markov_metal.reset(
                dspark_markov_metal_init(markov_w1.data(), markov_w2.data(), n_vocab, markov_rank, mask_token_id));
            if (!markov_metal) {
                throw std::runtime_error("Metal Markov initialization failed; refusing CPU fallback");
            }
            LOG_INF("dspark: correction_backend=METAL_MARKOV\n");
#else
            throw std::runtime_error("Metal Markov requested but not compiled");
#endif
        }

        LOG_INF("%s: adding speculative implementation 'draft-dspark'\n", __func__);
        LOG_INF(
            "%s: - block_size=%d, mask_token_id=%d, n_capture=%lld, n_embd=%lld, n_vocab=%lld, markov_rank=%lld, "
            "has_markov=%d\n",
            __func__, block_size, mask_token_id, (long long) n_capture, (long long) n_embd, (long long) n_vocab,
            (long long) markov_rank, (int) has_markov);
        if (markov_rank > 0 && !has_markov && !meta.graph_corrected) {
            LOG_WRN(
                "%s: dspark model reports markov_rank=%lld but its markov head weights could not be read "
                "(gated/rnn markov head type? only 'vanilla' is supported) -- "
                "block logits will NOT be markov-corrected\n",
                __func__, (long long) markov_rank);
        }

        // dspark attention is fully non-causal within a call: the draft block
        // attends over the WHOLE persistent cache plus itself, with no
        // position-based masking (attention_mask=None, is_causal=False in the
        // reference) -- see src/models/dspark.cpp's header comment.
        llama_set_causal_attn(ctx_dft, false);

        const char * greedy_env = std::getenv("LLAMA_DSPARK_GREEDY_IDS");
        if (greedy_env && std::strcmp(greedy_env, "0") && std::strcmp(greedy_env, "1")) {
            throw std::runtime_error("LLAMA_DSPARK_GREEDY_IDS must be 0 or 1");
        }
        if (greedy_env && std::strcmp(greedy_env, "1") == 0) {
            if (!meta.graph_corrected) {
                throw std::runtime_error("DSpark greedy IDs require graph correction");
            }
            graph_samplers.reserve(n_seq);
            for (llama_seq_id seq = 0; seq < (llama_seq_id) n_seq; ++seq) {
                std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> chain(
                    llama_sampler_chain_init(llama_sampler_chain_default_params()), llama_sampler_free);
                llama_sampler_chain_add(chain.get(), llama_sampler_init_greedy());
                graph_samplers.push_back(std::move(chain));
            }
        }

        n_cache.assign(n_seq, 0);
        ctx_feat.assign(n_seq, {});
        ctx_pos.assign(n_seq, {});
        rows_since_accept.assign(n_seq, 0);
        i_batch_beg.assign(n_seq, -1);
        i_batch_end.assign(n_seq, -1);

        const int32_t n_b      = (int32_t) llama_n_batch(ctx_dft);
        batch                  = llama_batch_init(/* n_tokens = */ n_b, /* embd = */ 0, /* n_seq_max = */ 1);
        llama_seq_id attaching = 0;
        try {
            for (; attaching < (llama_seq_id) graph_samplers.size(); ++attaching) {
                if (!llama_set_sampler(ctx_dft, attaching, graph_samplers[attaching].get())) {
                    throw std::runtime_error("DSpark greedy ID offload failed; refusing host fallback");
                }
            }
        } catch (...) {
            // Detach before member destruction frees the samplers, including the failed attachment.
            for (llama_seq_id seq = 0; seq <= attaching && seq < (llama_seq_id) graph_samplers.size(); ++seq) {
                llama_set_sampler(ctx_dft, seq, nullptr);
            }
            llama_batch_free(batch);
            throw;
        }
        if (!graph_samplers.empty()) {
            LOG_INF("dspark: output_backend=GRAPH_GREEDY_IDS\n");
        }
    }

    ~common_speculative_impl_draft_dspark() override {
        for (llama_seq_id seq = 0; seq < (llama_seq_id) graph_samplers.size(); ++seq) {
            llama_set_sampler(params.ctx_dft, seq, nullptr);
        }
        LOG_INF("dspark: correction_calls=%lld correction_wall_us=%lld\n", (long long) markov_calls,
                (long long) markov_time_us);
        llama_batch_free(batch);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & /*prompt*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        // fresh generation: drop any leftover state from a prior generation
        // that reused this seq slot, and make sure ctx_dft's own cache for
        // this seq starts empty.
        n_cache[seq_id] = 0;
        ctx_feat[seq_id].clear();
        ctx_pos[seq_id].clear();
        rows_since_accept[seq_id] = 0;

        llama_memory_seq_rm(llama_get_memory(params.ctx_dft), seq_id, 0, -1);
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        // TODO: how to make it work with vision tokens? (mirrors draft-mtp)
        if (batch_in.token == nullptr || batch_in.embd != nullptr) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        std::fill(i_batch_beg.begin(), i_batch_beg.end(), -1);
        std::fill(i_batch_end.begin(), i_batch_end.end(), -1);

        for (int k = 0; k < n_tokens; ++k) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                GGML_ASSERT(batch_in.n_seq_id[k] == 1);

                if (batch_in.seq_id[k][0] == seq_id) {
                    i_batch_end[seq_id] = k;
                    if (i_batch_beg[seq_id] < 0) {
                        i_batch_beg[seq_id] = k;
                    }
                }
            }
        }

        auto * ctx_tgt = params.ctx_tgt;

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_beg[seq_id] < 0) {
                continue;
            }

            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;

            auto & feat = ctx_feat[seq_id];
            auto & pos  = ctx_pos[seq_id];

            const size_t row0 = pos.size();
            feat.resize((row0 + (size_t) n_rows) * (size_t) n_embd_cap);
            pos.resize(row0 + (size_t) n_rows);

            for (int32_t i = 0; i < n_rows; ++i) {
                const int32_t k = i_batch_beg[seq_id] + i;

                // NOTE: capture rows always use the masked (output-row) layout
                // (see src/llama-context.cpp's get_embeddings_capture_ith) --
                // this requires the caller to have requested logits/output on
                // EVERY row it wants a capture row for (unlike the pre-norm
                // path MTP uses, which can force unmasked extraction). For a
                // long prompt this means every prefill row, not just the
                // last -- a caller-side (main/server driver loop) requirement
                // when a registered impl reports need_embd_capture(), exactly
                // analogous to draft-mtp's own begin()-time warning about
                // need_embd_nextn.
                const float * cap = llama_get_embeddings_capture_ith(ctx_tgt, k);
                if (cap == nullptr) {
                    LOG_ERR(
                        "%s: llama_get_embeddings_capture_ith(%d) returned null -- was "
                        "llama_set_capture_layers() engaged and logits requested for every "
                        "row this impl needs?\n",
                        __func__, k);
                    return false;
                }

                std::memcpy(feat.data() + (row0 + (size_t) i) * (size_t) n_embd_cap, cap,
                            (size_t) n_embd_cap * sizeof(float));
                pos[row0 + i] = batch_in.pos[k];
            }

            rows_since_accept[seq_id] += n_rows;
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        const char * prefix_value = std::getenv("LLAMA_DSPARK_CORRECTION_PREFIX");
        if (correction_prefix) {
            char *     end  = nullptr;
            const long keep = prefix_value ? std::strtol(prefix_value, &end, 10) : 0;
            if (!prefix_value || end == prefix_value || *end || keep != correction_rows) {
                throw std::runtime_error("DSpark correction prefix changed after initialization");
            }
        } else if (prefix_value) {
            throw std::runtime_error("DSpark correction prefix enabled after initialization");
        }
        auto *        ctx_dft     = params.ctx_dft;
        const int64_t n_batch_max = (int64_t) llama_n_batch(ctx_dft);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            auto & feat = ctx_feat[seq_id];
            auto & pos  = ctx_pos[seq_id];

            int64_t       L       = n_cache[seq_id];
            const int64_t start   = dp.pos0;
            int64_t       ctx_len = start - L;

            if (ctx_len <= 0) {
                LOG_WRN(
                    "%s: seq %d has no new context rows staged (n_past=%lld, cache=%lld) -- "
                    "skipping this round\n",
                    __func__, (int) seq_id, (long long) start, (long long) L);
                continue;
            }
            if ((int64_t) pos.size() != ctx_len) {
                LOG_ERR(
                    "%s: seq %d staged context rows (%zu) != expected ctx_len (%lld) -- "
                    "n_past bookkeeping is out of sync with process()/accept(); "
                    "aborting draft for this seq this round\n",
                    __func__, (int) seq_id, pos.size(), (long long) ctx_len);
                continue;
            }
            GGML_ASSERT(pos.front() == (int32_t) L &&
                        "dspark: staged rows do not start at the drafter's cache position");
            GGML_ASSERT(pos.back() == (int32_t) start - 1 &&
                        "dspark: staged rows do not end just before the anchor position");

            // Evict only draft context. Preserve absolute RoPE positions and the commit cursor.
            const int64_t window_begin = draft_window > 0 ? std::max(int64_t(0), start - draft_window) : 0;
            const int64_t skip         = std::max(int64_t(0), window_begin - L);
            L += skip;
            ctx_len -= skip;

            const int64_t chunk_capacity = std::min(n_batch_max, (int64_t) llama_n_ubatch(ctx_dft)) - draft_rows;
            if (chunk_capacity < 1) {
                throw std::runtime_error("dspark: batch cannot hold context and draft block");
            }

            if (draft_window > 0 &&
                !llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, 0, (llama_pos) window_begin)) {
                throw std::runtime_error("dspark: draft window eviction failed");
            }
            for (int64_t offset = 0; offset < ctx_len;) {
                const int64_t count     = std::min(chunk_capacity, ctx_len - offset);
                const int64_t chunk_end = L + offset + count;
                llama_set_dspark_ctx(ctx_dft, feat.data() + (skip + offset) * n_embd_cap, count, n_embd_cap,
                                     pos.data() + skip + offset);
                common_batch_clear(batch);
                for (int64_t i = 0; i < count; ++i) {
                    common_batch_add(batch, 0, (llama_pos) (L + offset + i), { seq_id }, false);
                }
                // Context K/V depend only on target features. Intermediate block outputs are discarded.
                common_batch_add(batch, dp.id_last, (llama_pos) chunk_end, { seq_id }, true);
                for (int32_t k = 1; k < draft_rows; ++k) {
                    common_batch_add(batch, mask_token_id, (llama_pos) (chunk_end + k), { seq_id },
                                     !correction_prefix || k < correction_rows);
                }
                const int32_t rc = llama_decode(ctx_dft, batch);
                llama_set_dspark_ctx(ctx_dft, nullptr, 0, 0, nullptr);
                if (rc != 0) {
                    throw std::runtime_error("dspark: chunk decode failed");
                }
                if (!llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, (llama_pos) chunk_end, -1)) {
                    throw std::runtime_error("dspark: draft tail removal failed");
                }
                offset += count;
            }
            if (draft_window > 0) {
                const auto mem = llama_get_memory(ctx_dft);
                if (llama_memory_seq_pos_min(mem, seq_id) != window_begin ||
                    llama_memory_seq_pos_max(mem, seq_id) != start - 1) {
                    throw std::runtime_error("dspark: draft window position invariant failed");
                }
            }
            n_cache[seq_id] = start;

            feat.clear();
            pos.clear();
            rows_since_accept[seq_id] = 0;  // this round's rows were just consumed

            // --- sequential Markov resample -------------------------------
            // step_logits[k] = base_logits[k] + markov_w2(markov_w1(prev_token)),
            // where prev_token is the block's own anchor token for k==0 and the
            // ACTUALLY SAMPLED token from step k-1 for k>0. This must never be
            // batched over mask_token_id for all block positions at once --
            // that exact bug class already hit the on-device (MLX/Swift) port.
            // The assert below makes the sequential dependency structural
            // rather than just a comment: it is unsatisfiable if this loop is
            // ever refactored to precompute prev_token_ids up front from
            // draft_input_ids instead of chaining the sampled result forward.
            llama_tokens result;
            result.reserve(block_size);
            if (!graph_samplers.empty()) {
                for (int32_t k = 0; k < correction_rows; ++k) {
                    const int32_t     output_rows = correction_prefix ? correction_rows : draft_rows;
                    const llama_token token       = llama_get_sampled_token_ith(ctx_dft, -output_rows + k);
                    if (token < 0 || token >= n_vocab || token == mask_token_id) {
                        throw std::runtime_error("DSpark graph returned an invalid draft token");
                    }
                    result.push_back(token);
                }
                if (result.size() >= (size_t) params.n_min) {
                    *dp.result = std::move(result);
                }
                continue;
            }

            // dense output buffer: only the block_size draft rows requested
            // logits this call, so llama_get_logits() is already exactly
            // block_size*n_vocab floats in row order -- no per-row index
            // resolution needed (mirrors tests/test-dspark-forward.cpp's
            // llama_get_logits(ctx) usage). llama_get_logits_ith(ctx, i)
            // would need i to be the RAW ubatch row (ctx_len + k here), since
            // it resolves through output_resolve_row() same as the
            // pre-norm/capture accessors -- the bulk buffer sidesteps that.
            const float * logits_base = llama_get_logits(ctx_dft);
            if (logits_base == nullptr) {
                LOG_ERR("%s: llama_get_logits(ctx_dft) returned null for seq %d\n", __func__, (int) seq_id);
                continue;
            }

            llama_token   prev_token          = dp.id_last;
            const int64_t correction_start_us = ggml_time_us();

#ifdef LLAMA_DSPARK_MARKOV_METAL
            if (markov_metal) {
                result.resize(correction_rows);
                if (!dspark_markov_metal_resample(markov_metal.get(), logits_base, dp.id_last, correction_rows,
                                                  result.data())) {
                    throw std::runtime_error("Metal Markov resample failed; refusing CPU fallback");
                }
                for (llama_token token : result) {
                    if (token < 0 || token >= n_vocab || token == mask_token_id) {
                        throw std::runtime_error("Metal Markov returned an invalid draft token");
                    }
                }
                markov_time_us += ggml_time_us() - correction_start_us;
                ++markov_calls;
                if (result.size() >= (size_t) params.n_min) {
                    *dp.result = std::move(result);
                }
                continue;
            }
#endif

#ifdef LLAMA_DSPARK_MARKOV_CUDA
            if (markov_cuda) {
                if (dcut_enabled && dp.n_max > 0 && dp.n_max < correction_rows) {
                    throw std::runtime_error("D-cut depth must not be overridden by a fixed verification cap");
                }
                result.resize(correction_rows);
                int32_t    retained = correction_rows;
                const bool ok = dcut_enabled ?
                                    dspark_markov_cuda_dcut(markov_cuda.get(), logits_base, dp.id_last, correction_rows,
                                                            result.data(), dcut_costs, &retained) :
                                    dspark_markov_cuda_resample(markov_cuda.get(), logits_base, dp.id_last,
                                                                correction_rows, result.data());
                if (!ok) {
                    throw std::runtime_error("CUDA Markov resample failed; refusing CPU fallback");
                }
                for (llama_token token : result) {
                    if (token < 0 || token >= n_vocab || token == mask_token_id) {
                        throw std::runtime_error("CUDA Markov returned an invalid draft token");
                    }
                }
                markov_time_us += ggml_time_us() - correction_start_us;
                ++markov_calls;
                result.resize(retained);
                if (result.size() >= (size_t) params.n_min) {
                    *dp.result = std::move(result);
                }
                continue;
            }
#endif

            for (int32_t k = 0; k < correction_rows; ++k) {
                if (k > 0) {
                    GGML_ASSERT(prev_token != mask_token_id &&
                                "dspark: markov resample must chain the previous step's SAMPLED "
                                "token, never mask_token_id -- do not batch this over the block");
                }

                const float * base_logits = logits_base + (size_t) k * n_vocab;

                llama_token best_id = mask_token_id == 0 ? 1 : 0;
                float       best_v  = -std::numeric_limits<float>::infinity();

                if (has_markov) {
                    const float * emb = markov_w1.data() + (size_t) prev_token * (size_t) markov_rank;
#ifdef LLAMA_DSPARK_MARKOV_BLAS
                    cblas_sgemv(CblasRowMajor, CblasNoTrans, (int) n_vocab, (int) markov_rank, 1.0f, markov_w2.data(),
                                (int) markov_rank, emb, 1, 0.0f, markov_bias.data(), 1);

                    for (int64_t v = 0; v < n_vocab; ++v) {
                        if (v == mask_token_id) {
                            continue;
                        }
                        const float logit = base_logits[v] + markov_bias[(size_t) v];
                        if (logit > best_v) {
                            best_v  = logit;
                            best_id = (llama_token) v;
                        }
                    }
#else
                    for (int64_t v = 0; v < n_vocab; ++v) {
                        if (v == mask_token_id) {
                            continue;
                        }
                        const float * w2row = markov_w2.data() + (size_t) v * (size_t) markov_rank;
                        float         bias  = 0.0f;
                        for (int64_t r = 0; r < markov_rank; ++r) {
                            bias += emb[r] * w2row[r];
                        }
                        const float logit = base_logits[v] + bias;
                        if (logit > best_v) {
                            best_v  = logit;
                            best_id = (llama_token) v;
                        }
                    }
#endif
                } else {
                    for (int64_t v = 0; v < n_vocab; ++v) {
                        if (v == mask_token_id) {
                            continue;
                        }
                        if (base_logits[v] > best_v) {
                            best_v  = base_logits[v];
                            best_id = (llama_token) v;
                        }
                    }
                }

                result.push_back(best_id);
                prev_token = best_id;  // chain the SAMPLED token, never mask_token_id
            }

            markov_time_us += ggml_time_us() - correction_start_us;
            ++markov_calls;
            if (result.size() < (size_t) params.n_min) {
                continue;  // dp.result stays empty: treated as a failed draft this round
            }

            *dp.result = std::move(result);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int64_t n_round_rows = rows_since_accept[seq_id];
        rows_since_accept[seq_id]  = 0;
        if (n_round_rows <= 0) {
            return;
        }

        // process() (or the test-only injection hook) unconditionally
        // captured tap features for the WHOLE verify batch, including any
        // positions past the accepted prefix; trim this round's
        // freshly-appended tail down to n_accepted+1 rows (the actually
        // committed context), discarding the rejected continuation. This
        // mirrors the DeepSpec reference's evaluator._update():
        //   context.target_hidden_states = verified_target_hidden[:, :accepted_draft_tokens+1, :]
        // Runs the same way regardless of is_other: dspark's own context must
        // stay correct even on rounds where a different implementation's
        // draft is the one that gets verified.
        const int64_t keep = std::min<int64_t>(n_round_rows, (int64_t) n_accepted + 1);
        const int64_t drop = n_round_rows - keep;

        if (drop > 0) {
            auto & feat = ctx_feat[seq_id];
            auto & pos  = ctx_pos[seq_id];

            const size_t total_rows = pos.size();
            GGML_ASSERT((int64_t) total_rows >= drop);

            feat.resize((total_rows - (size_t) drop) * (size_t) n_embd_cap);
            pos.resize(total_rows - (size_t) drop);
        }
    }

    bool need_embd() const override { return false; }

    bool need_embd_nextn() const override { return false; }

    bool need_embd_capture() const override { return true; }

    bool stage_test_ctx_feat(llama_seq_id    seq_id,
                             const float *   feat_in,
                             int64_t         n_rows,
                             int64_t         n_embd_cap_in,
                             const int32_t * pos_in) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return false;
        }
        if (n_embd_cap_in != n_embd_cap) {
            LOG_ERR("%s: n_embd_cap mismatch: got %lld, expected %lld\n", __func__, (long long) n_embd_cap_in,
                    (long long) n_embd_cap);
            return false;
        }
        if (n_rows <= 0) {
            return true;
        }

        auto & feat = ctx_feat[seq_id];
        auto & pos  = ctx_pos[seq_id];

        const size_t row0 = pos.size();
        feat.resize((row0 + (size_t) n_rows) * (size_t) n_embd_cap);
        pos.resize(row0 + (size_t) n_rows);

        std::memcpy(feat.data() + row0 * (size_t) n_embd_cap, feat_in,
                    (size_t) n_rows * (size_t) n_embd_cap * sizeof(float));
        std::memcpy(pos.data() + row0, pos_in, (size_t) n_rows * sizeof(int32_t));

        rows_since_accept[seq_id] += n_rows;
        return true;
    }
};

struct common_speculative_impl_draft_simple : public common_speculative_impl {
    common_params_speculative_draft params;

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    common_speculative_impl_draft_simple(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, n_seq, params.draft.n_max)
        , params(params.draft)
    {
        auto * ctx_dft = this->params.ctx_dft;
        auto * ctx_tgt = this->params.ctx_tgt;

        if (!ctx_dft) {
            throw std::runtime_error("draft-simple requires a draft context");
        }

        SPC_TRC("%s", "adding speculative implementation 'draft-simple'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%f\n", this->params.n_max, this->params.n_min, this->params.p_min);
        SPC_TRC("- gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n",
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);

        // TODO: optimize or pass from outside?
        // {
        //     common_params_sampling params;
        //     params.no_perf = false;
        //
        //     params.top_k = 40;
        //     params.top_p = 0.9;
        //
        //     params.samplers = {
        //         COMMON_SAMPLER_TYPE_TOP_K,
        //         COMMON_SAMPLER_TYPE_TOP_P,
        //         COMMON_SAMPLER_TYPE_INFILL,
        //     };
        //
        //     result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        // }

        smpls.resize(n_seq);
        for (auto & smpl : smpls) {
            common_params_sampling params;
            params.no_perf = false;
            params.top_k = 10;
            params.samplers = {
                COMMON_SAMPLER_TYPE_TOP_K,
            };

            smpl.reset(common_sampler_init(llama_get_model(ctx_dft), params));
        }

        const bool vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        SPC_DBG("vocab_cmpt = %d\n", vocab_cmpt);

        if (!vocab_cmpt) {
            SPC_ERR("%s", "the target and draft vocabs are not compatible\n");

            throw std::runtime_error("draft model vocab type must match target model to use speculation");
        }

        if (n_seq != llama_n_seq_max(ctx_dft)) {
            SPC_ERR("n_seq mismatch: %d != %d\n", n_seq, llama_n_seq_max(ctx_dft));

            throw std::runtime_error("the draft model number of sequences is incompatible with the speculative n_seq");
        }
    }

    ~common_speculative_impl_draft_simple() override {
        llama_batch_free(batch);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & batch) override {
        auto * ctx_dft = params.ctx_dft;

        llama_batch batch_dft = batch;
        batch_dft.logits = nullptr;

        const int ret = llama_decode(ctx_dft, batch_dft);

        if (ret != 0) {
            SPC_ERR("failed to decode draft batch, ret = %d\n", ret);

            return false;
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            common_batch_add(batch, dp.id_last, dp.pos0, { seq_id }, true);
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            SPC_ERR("llama_decode returned %d\n", ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if ((params.n_max <= (int) result.size()) ||
                    (dp.n_max > 0 && dp.n_max <= (int) result.size())) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, dp.pos0 + i + 1, { seq_id }, true);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\n", i, ret);
                break;
            }

            ++i;
        }

        for (auto & dp : dparams) {
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};


// EAGLE3 speculative decoding state
//
// Input of draft decoder: (This is different compared to MTP)
//   At "pos P", the decoder takes input pair (t_{P+1}, g_P), with RoPE at P.
//     - t_{P+1} = token at sequence pos P+1 (the *next* token after P)
//     - g_P     = encoder output = projection of target's extracted hidden states at P
//
// Deferred boundary (MTP doesn't have this issue):
//   Within a single process() call with n_tokens, we can only write decoder KV for
//   training pos 0..n_tokens-2. The last training pos (n_tokens-1) needs t_{n_tokens}
//   which lies *outside* this batch — it is the token target will sample next or the first token from next ubatch.
//   So the last training pos of each process() call is *deferred* to whichever next call has
//   the missing token in hand:
//     - multi-ubatch prefill: the next process()'s first token completes the pair
//                              (handled by the per-seq "cross-ubatch bridge")
//     - single-ubatch prefill / after verify: draft()'s seed step uses "dp.id_last"
//                              (target's freshest sample) to complete the pair
//
// Per-seq carry-over state:
//   pending_g_last    [n_embd_dec]  ┐  the deferred boundary's (g, pos). Set by
//   pending_pos_last  llama_pos     ┘  process() at end of ubatch (= last row);
//                                       rebased by accept() to first-non-accepted pos.
//   verify_g          [N × n_embd_dec] snapshot of process()'s encoder output;
//   verify_pos_first  llama_pos         consumed by accept() to recover the right
//   verify_g_rows     int32_t           pending_g_last row for any n_accepted value.
//
// Performance is overall good but there is waste in verify cycle:
//   process() runs encoder + decoder on the *full* verify batch including rows for
//   rejected drafts. The KV at those positions is then dropped.
//
// TODO: Not sure if we need optimization for this waste?
// If so we may need hybrid stash:
//      in verify mode, have process() only stash features and let draft() seed run
//      encoder+decoder on n_accepted+1 rows).
struct common_speculative_impl_draft_eagle3 : public common_speculative_impl {
    common_params_speculative_draft params;
    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd_dec = 0;       // draft context row width (n_embd_out: per-layer for DFly)
    int32_t n_embd_enc = 0;       // target_layer_ids_n * target_hidden_size
    int32_t n_embd_tgt = 0;       // target model hidden size
    int32_t n_layer_tgt = 0;      // target model layer count

    const int32_t * target_layer_ids   = nullptr; // model_dft's extract layer indices
    uint32_t        target_layer_ids_n = 0;

    // [per-seq] deferred boundary state
    std::vector<std::vector<float>> pending_g_last;
    std::vector<llama_pos>          pending_pos_last;

    // [per-seq] snapshot of the most recent process()'s encoder output
    std::vector<std::vector<float>> verify_g;         // [n_seq][n_rows * n_embd_dec]
    std::vector<llama_pos>          verify_pos_first; // [n_seq] — pos of verify_g[seq][0]
    std::vector<int32_t>            verify_g_rows;    // [n_seq] — number of rows

    // scratch buffer for concatenated target features [n_tokens, n_embd_enc]
    std::vector<float> features_buf;
    std::vector<float> g_embd_buf;

    common_speculative_impl_draft_eagle3(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, n_seq, params.draft.n_max)
        , params(params.draft)
    {
        SPC_TRC("%s", "adding speculative implementation 'draft-eagle3'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%f, backend_sampling=%d\n", params.draft.n_max, params.draft.n_min, params.draft.p_min, (int) params.draft.backend_sampling);

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "EAGLE3 requires ctx_tgt and ctx_dft to be set");

        const llama_model * model_dft = llama_get_model(ctx_dft);
        const llama_model * model_tgt = llama_get_model(ctx_tgt);

        target_layer_ids   = llama_model_target_layer_ids  (model_dft);
        target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
        if (target_layer_ids_n != 3) {
            throw std::runtime_error("draft model is not eagle3 (expected 3 extract layers, got " +
                                     std::to_string(target_layer_ids_n) + ")");
        }

        n_embd_tgt = llama_model_n_embd(model_tgt);
        n_embd_dec = llama_model_n_embd_out(model_dft);
        n_embd_enc = (int32_t) target_layer_ids_n * n_embd_tgt;
        n_layer_tgt = llama_model_n_layer(model_tgt);

        const int32_t n_b = (int32_t) llama_n_batch(ctx_dft);
        batch = llama_batch_init(/*n_tokens=*/ n_b, /*embd=*/ n_embd_dec, /*n_seq_max=*/ 1);
        // llama_batch_init allocates only one of token/embd; eagle3 decoder needs both.
        // TODO: fix, how to call without malloc
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_b);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        // turn on extraction of the target layers' hidden states
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            if (target_layer_ids[k] < n_layer_tgt) {
                llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);
            } else if (target_layer_ids[k] == n_layer_tgt) {
                llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
            } else {
                GGML_ABORT("EAGLE3: target layer id %d exceeds target n_layer %d", target_layer_ids[k], n_layer_tgt);
            }
        }

        // turn on extraction of the draft model's pre-norm hidden state
        // (used both for the encoder output g_embd and the decoder pre-norm output).
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);

        pending_g_last.assign(n_seq, std::vector<float>(n_embd_dec, 0.0f));
        pending_pos_last.assign(n_seq, -1);

        verify_g.assign(n_seq, std::vector<float>());
        verify_pos_first.assign(n_seq, -1);
        verify_g_rows.assign(n_seq, 0);
    }

    ~common_speculative_impl_draft_eagle3() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        if (batch.token != nullptr) {
            free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }
        // expected state after prefill: ctx_dft has pos 0..N-2 (last position is deferred to
        // draft()'s seed step). Warn only if more than one position is missing.
        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
        if (pos_max < N - 2) {
            SPC_WRN("ctx_dft pos_max=%d < N-2=%d — process() did not run on every prefill ubatch. "
                    "Drafts may degrade.\n",
                    (int) pos_max, N - 2);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        if (batch_in.token == nullptr || batch_in.embd != nullptr) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // i_batch_beg[seq] / i_batch_end[seq]: inclusive batch indices of this seq's
        // first/last token in batch_in. Assumes per-seq tokens are contiguous within
        // the ubatch (server's default ordering).
        std::vector<int32_t> i_batch_beg(n_seq, -1);
        std::vector<int32_t> i_batch_end(n_seq, -1);
        for (int k = 0; k < n_tokens; ++k) {
            GGML_ASSERT(batch_in.n_seq_id[k] == 1);
            const llama_seq_id seq_id = batch_in.seq_id[k][0];
            if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
                continue;
            }
            i_batch_end[seq_id] = k;
            if (i_batch_beg[seq_id] < 0) {
                i_batch_beg[seq_id] = k;
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        // Interleave each extract_layer's hidden state into a contiguous buffer of
        // shape [n_tokens, target_layer_ids_n * n_embd_tgt]. Then run EAGLE3 encoder
        // to get one g_embd row per token.
        features_buf.resize((size_t) n_tokens * n_embd_enc, 0.0f);

        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            const float * layer = target_layer_ids[k] < n_layer_tgt
                ? llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k])
                : llama_get_embeddings_nextn(ctx_tgt);
            if (!layer) {
                GGML_ABORT("EAGLE3: target layer %d input not extracted.", target_layer_ids[k]);
            }
            for (int32_t i = 0; i < n_tokens; ++i) {
                float * dst = features_buf.data() + (size_t) i * n_embd_enc + k * (size_t) n_embd_tgt;
                const float * src = layer + (size_t) i * n_embd_tgt;
                std::memcpy(dst, src, (size_t) n_embd_tgt * sizeof(float));
            }
        }

        g_embd_buf.resize((size_t) n_tokens * n_embd_dec);

        // llama_encode() requires the full encoder batch to fit in n_ubatch.
        // Allow batch > ubatch: eagle3's per-token encoder can be chunked safely.
        const int32_t n_ubatch_dft = (int32_t) llama_n_ubatch(ctx_dft);
        for (int32_t i = 0; i < n_tokens; i += n_ubatch_dft) {
            const int32_t n_chunk = std::min(n_ubatch_dft, n_tokens - i);

            llama_batch enc_batch = {
                /*.n_tokens =*/ n_chunk,
                /*.token    =*/ nullptr,
                /*.embd     =*/ features_buf.data() + (size_t) i * n_embd_enc,
                /*.pos      =*/ nullptr,
                /*.n_seq_id =*/ nullptr,
                /*.seq_id   =*/ nullptr,
                /*.logits   =*/ nullptr,
            };
            const int32_t rc = llama_encode(ctx_dft, enc_batch);
            if (rc != 0) {
                SPC_ERR("llama_encode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                        rc, (int) n_chunk, (int) i);
                return false;
            }

            // g_embd has shape [n_chunk, n_embd_dec] in ctx_dft's pre-norm embeddings buffer.
            const float * g_embd_chunk = llama_get_embeddings_nextn(ctx_dft);
            GGML_ASSERT(g_embd_chunk && "EAGLE3 encoder produced no output.");
            std::memcpy(g_embd_buf.data() + (size_t) i * n_embd_dec,
                        g_embd_chunk,
                        (size_t) n_chunk * n_embd_dec * sizeof(float));
        }

        const float * g_embd = g_embd_buf.data();

        const size_t row_bytes = (size_t) n_embd_dec * sizeof(float);

        // EAGLE3 decoder input convention: at memory pos P the input pair is
        // (token[P+1], g_embd[P]). This shifts the token index "left by one" relative to g_embd.
        //
        // Per seq, in order:
        //   (a) cross-ubatch bridge — when applicable, write the previously-deferred
        //       pos using this ubatch's first token + pending_g_last.
        //   (b) main write loop — for k in [beg, end-1], write (token[k+1], g_embd[k])
        //       at pos[k]. The last training pos (k=end) is left unwritten = new
        //       deferred boundary, completed by the next process() or draft() call.
        //   (c) refresh deferred state — stash this ubatch's full g_embd into verify_g,
        //       update pending_g_last / pending_pos_last to the last row.
        common_batch_clear(batch);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            const int32_t beg = i_batch_beg[seq_id];
            const int32_t end = i_batch_end[seq_id];
            if (beg < 0 || end < 0) {
                continue;
            }

            // cross-ubatch bridge — complete the prior ubatch's deferred boundary.
            // Fires iff all three preconditions hold:
            //   1) pending_pos_last >= 0
            //   2) pending_pos_last + 1 == pos[beg]
            //   3) pending_pos_last > dft_pos_max // TODO: is this check needed?
            const llama_pos pending_pos = pending_pos_last[seq_id];
            if (pending_pos >= 0 && pending_pos + 1 == batch_in.pos[beg]) {
                const llama_pos dft_pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
                if (pending_pos > dft_pos_max) {
                    common_batch_add(batch, batch_in.token[beg], pending_pos, { seq_id }, /*logits=*/ false);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                                pending_g_last[seq_id].data(), row_bytes);
                }
            }

            for (int32_t k = beg; k < end; ++k) {
                common_batch_add(batch, batch_in.token[k + 1], batch_in.pos[k], { seq_id }, /*logits=*/ false);
                std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                            g_embd + (size_t) k * n_embd_dec, row_bytes);
            }

            // refresh deferred state
            const int32_t n_rows = end - beg + 1;
            verify_pos_first[seq_id] = batch_in.pos[beg];
            pending_pos_last[seq_id] = batch_in.pos[end];
            verify_g_rows[seq_id]    = n_rows;
            verify_g[seq_id].resize((size_t) n_rows * n_embd_dec, 0.0f);
            std::memcpy(verify_g[seq_id].data(),       g_embd + (size_t) beg * n_embd_dec, row_bytes * n_rows);
            std::memcpy(pending_g_last[seq_id].data(), g_embd + (size_t) end * n_embd_dec, row_bytes);
        }

        if (batch.n_tokens > 0) {
            const int32_t rc = llama_decode(ctx_dft, batch);
            if (rc != 0) {
                SPC_ERR("llama_decode(ctx_dft) failed rc=%d (n_tokens=%d, ubatch_pos[0]=%d)\n",
                        rc, (int) batch.n_tokens, (int) batch_in.pos[0]);
                return false;
            }
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        const size_t row_bytes = (size_t) n_embd_dec * sizeof(float);

        // Complete the deferred boundary pair (dp.id_last, pending_g_last) at memory
        // pos pending_pos_last. dp.id_last is target's freshest sample (= corrected
        // token after verify, or first generated token after prefill), matching the
        // EAGLE3 input convention (token[P+1], g_embd[P]) at pos P.
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }
            if (pending_pos_last[seq_id] < 0) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, pending_pos_last[seq_id], -1);

            common_batch_add(batch, dp.id_last, pending_pos_last[seq_id], { seq_id }, true);
            std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                        pending_g_last[seq_id].data(),
                        row_bytes);
        }

        if (batch.n_tokens == 0) {
            return;
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            SPC_ERR("llama_decode returned %d\n", ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                // pre-norm hidden state of this position becomes g_embd for the next step
                const float * prenorm = llama_get_embeddings_nextn_ith(ctx_dft, i_batch);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                // (configurable via --spec-draft-p-min, set to 0.0 to disable early-stop)
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if (params.n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, pending_pos_last[seq_id] + (i + 1), { seq_id }, true);
                std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec, prenorm, row_bytes);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\n", i, ret);
                break;
            }

            ++i;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_g_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_g = std::min<int32_t>(n_accepted, n_rows - 1);
        pending_pos_last[seq_id] = verify_pos_first[seq_id] + i_g;
        std::memcpy(pending_g_last[seq_id].data(),
                    verify_g[seq_id].data() + (size_t) i_g * n_embd_dec,
                    (size_t) n_embd_dec * sizeof(float));
    }

    // we only need to stash the deferred boundary's g_embd row for recurrent/hybrid targets:
    // their single-position checkpoints drop it on restore
    bool need_boundary_stash() const {
        const llama_model * model_tgt = llama_get_model(params.ctx_tgt);
        return llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt);
    }

    bool get_state(llama_seq_id seq_id, std::vector<uint8_t> & data) const override {
        if (!need_boundary_stash()) {
            return false;
        }
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq || pending_pos_last[seq_id] < 0) {
            return false;
        }

        const llama_pos          pos = pending_pos_last[seq_id];
        const std::vector<float> & g = pending_g_last[seq_id];

        data.resize(sizeof(llama_pos) + g.size() * sizeof(float));
        std::memcpy(data.data(),                     &pos,     sizeof(llama_pos));
        std::memcpy(data.data() + sizeof(llama_pos), g.data(), g.size() * sizeof(float));
        return true;
    }

    bool validate_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) const override {
        if (!need_boundary_stash()) {
            return data.empty();
        }
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return false;
        }
        if (data.empty()) {
            return true;
        }
        if (data.size() != sizeof(llama_pos) + (size_t) n_embd_dec * sizeof(float)) {
            return false;
        }

        llama_pos pos = -1;
        std::memcpy(&pos, data.data(), sizeof(llama_pos));
        return pos >= 0;
    }

    bool set_state(llama_seq_id seq_id, const std::vector<uint8_t> & data, llama_pos expected_pos) override {
        if (!need_boundary_stash()) {
            return false;
        }
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return false;
        }
        if (data.size() != sizeof(llama_pos) + (size_t) n_embd_dec * sizeof(float)) {
            return false;
        }

        llama_pos pos = -1;
        std::memcpy(&pos, data.data(), sizeof(llama_pos));
        if (pos < 0 || (expected_pos >= 0 && pos != expected_pos)) {
            return false;
        }

        pending_g_last[seq_id].resize(n_embd_dec);
        std::memcpy(pending_g_last[seq_id].data(), data.data() + sizeof(llama_pos), (size_t) n_embd_dec * sizeof(float));
        pending_pos_last[seq_id] = pos;
        return true;
    }

    bool set_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) override {
        if (!validate_state(seq_id, data)) {
            return false;
        }
        if (!need_boundary_stash()) {
            return true;
        }
        if (data.empty()) {
            pending_pos_last[seq_id] = -1;
            std::fill(pending_g_last[seq_id].begin(), pending_g_last[seq_id].end(), 0.0f);
            return true;
        }

        llama_pos pos = -1;
        std::memcpy(&pos, data.data(), sizeof(llama_pos));
        pending_pos_last[seq_id] = pos;
        GGML_ASSERT(pending_g_last[seq_id].size() == (size_t) n_embd_dec);
        std::memcpy(pending_g_last[seq_id].data(), data.data() + sizeof(llama_pos), (size_t) n_embd_dec * sizeof(float));
        return true;
    }

    bool need_embd() const override {
        return false;
    }
};

// DFlash: block-diffusion drafting with a draft-side KV cache injection
struct common_speculative_impl_draft_dflash : public common_speculative_impl {
    common_params_speculative_draft params;

    bool local_split = false;
    bool local_prefetch = false;

    llama_batch batch;        // noise tokens
    llama_batch batch_inject; // target features for KV cache injection

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd_dec = 0;  // draft context row width (n_embd_out: per-layer for DFly)
    int32_t n_embd_enc = 0;  // target_layer_ids_n * target_hidden_size
    int32_t n_embd_tgt = 0;  // target model hidden size
    int32_t n_embd_dft = 0;  // explicit split input/hidden row width

    int32_t     block_size    = 0;
    llama_token mask_token_id = 0;

    bool    is_dflash2     = false;
    bool    is_mrope       = false;
    int32_t selector_top_k = 0;
    std::vector<std::mt19937> selector_rng;
    std::vector<bool> selector_reset;

    // Optional DFlash2 Xbox execution. The target and selector remain on this host.
    std::unique_ptr<dflash_remote::client> remote;
    std::unique_ptr<llama_context, decltype(&llama_free)> selector_owned{nullptr, llama_free};
    llama_context * selector_ctx = nullptr;
    llama_batch selector_batch{};
    std::vector<float> mask_embedding;
    std::vector<uint8_t> rpc_bytes;
    std::ofstream rpc_metrics;
    std::ofstream rpc_capture;
    std::ifstream rpc_replay;
    enum class remote_profile_mode { real, mock_xbox, mock_local } profile_mode = remote_profile_mode::real;
    uint64_t rpc_request_id = 0;
    uint64_t rpc_sync_us = 0;
    uint64_t rpc_capture_us = 0;
    uint64_t rpc_sync_serialize_us = 0;
    uint64_t rpc_sync_send_us = 0;
    uint64_t rpc_sync_wait_us = 0;
    uint64_t rpc_sync_xbox_prepare_us = 0;
    uint64_t rpc_sync_xbox_compute_us = 0;
    uint64_t rpc_rewind_us = 0;
    int32_t remote_last_sync_pos = -1;
    bool remote_failed = false;
    bool remote_draft_pending = false;
    bool remote_trimmed = false;
    bool remote_fused_sync = false;
    bool remote_fused_draft = false;
    std::vector<uint8_t> remote_sync_payload;
    uint32_t remote_sync_tokens = 0;
    bool remote_sync_has_trim = false;
    int32_t remote_sync_last_pos = -1;
    uint64_t remote_trim_us = 0;
    int32_t remote_draft_pos0 = -1;
    int32_t remote_n_proposed = 0;
    dflash_remote::timing remote_draft_timing{};
    int64_t remote_selector_us = 0;
    int64_t remote_selector_copy_us = 0;
    int64_t remote_prepare_us = 0;
    int64_t remote_serialize_us = 0;
    int64_t remote_proposal_us = 0;
    std::chrono::steady_clock::time_point remote_proposal_done;
    std::chrono::steady_clock::time_point remote_draft_start;
    std::chrono::steady_clock::time_point remote_draft_done;

    // DFlash2 Xbox speculative pipeline.  The request runs on the same
    // persistent RPC connection, but on a worker so the target can verify the
    // current block while the Xbox prepares the next one.  Target-side SYNC
    // rows are held until accept() decides whether the bonus-token hypothesis won.
    struct remote_pipeline_sync {
        dflash_remote::op operation = dflash_remote::sync;
        uint32_t n_tokens = 0;
        std::vector<uint8_t> payload;
        int32_t last_pos = -1;
    };
    struct remote_pipeline_result {
        dflash_remote::timing timing{};
        std::vector<uint8_t> output;
        std::chrono::steady_clock::time_point done;
    };

    std::future<remote_pipeline_result> remote_pipeline_future;
    std::vector<remote_pipeline_sync> remote_pipeline_syncs;
    std::vector<float> remote_pipeline_seed_features;
    std::vector<uint8_t> remote_pipeline_output;
    dflash_remote::timing remote_pipeline_timing{};
    std::chrono::steady_clock::time_point remote_pipeline_start;
    std::chrono::steady_clock::time_point remote_pipeline_done;
    int32_t remote_pipeline_pos0 = -1;
    int32_t remote_pipeline_id_last = -1;
    int32_t remote_pipeline_n_max = 0;
    int32_t remote_draft_n_max = 0;
    dflash_pipeline_candidate remote_next_candidate;
    int32_t remote_pipeline_accept_k = -1;
    llama_token remote_current_anchor_id = -1;
    bool remote_pipeline_enabled = false;
    bool remote_pipeline_inflight = false;
    bool remote_pipeline_ready = false;
    bool remote_pipeline_state_synced = false;
    bool remote_pipeline_used = false;
    bool remote_pipeline_discarded = false;
    uint32_t remote_pipeline_attempts = 0;
    uint32_t remote_pipeline_hits = 0;
    uint32_t remote_pipeline_match_mask = 0;
    int32_t remote_pipeline_actual_pos = -1;
    llama_token remote_pipeline_actual_id = -1;
    uint64_t remote_pipeline_wait_us = 0;

    // Explicit local DFlash2 split.  The draft transformer owns the context on
    // the selected draft device; the target-side selector consumes only the
    // normalized hidden rows returned by that context.  This is the in-process
    // equivalent of the compact RPC payload, without a socket or a shared target
    // embedding/output tensor.
    struct local_pipeline_sync {
        std::vector<float> features;
        std::vector<llama_pos> positions;
    };
    struct local_pipeline_result {
        std::vector<float> hidden;
    };

    std::future<local_pipeline_result> local_pipeline_future;
    std::vector<local_pipeline_sync> local_pipeline_syncs;
    std::vector<float> local_pipeline_output;
    std::chrono::steady_clock::time_point local_pipeline_start;
    int32_t local_pipeline_pos0 = -1;
    int32_t local_pipeline_id_last = -1;
    int32_t local_pipeline_n_max = 0;
    bool local_pipeline_inflight = false;
    bool local_pipeline_pending = false; // worker KV still needs commit or rollback
    bool local_pipeline_ready = false;
    bool local_pipeline_state_synced = false;
    bool local_pipeline_used = false;
    bool local_pipeline_discarded = false;
    bool local_draft_pending = false;
    bool local_failed = false;
    int32_t local_draft_pos0 = -1;
    int32_t local_n_proposed = 0;
    dflash_pipeline_candidate local_next_candidate;
    int32_t local_pipeline_accept_k = -1;
    uint32_t local_request_attempts = 0;
    uint32_t local_request_hits = 0;
    uint32_t local_pipeline_attempts = 0;
    uint32_t local_pipeline_hits = 0;
    uint32_t local_pipeline_discards = 0;

    bool remote_pipeline_active() const {
        return remote_pipeline_inflight || remote_pipeline_ready;
    }

    bool local_pipeline_active() const {
        return local_pipeline_pending || local_pipeline_inflight || local_pipeline_ready;
    }

    bool trim_local_context(llama_pos requested_p0) {
        auto * mem = llama_get_memory(params.ctx_dft);
        const bool exact = llama_memory_seq_rm(mem, 0, requested_p0, -1);
        if (exact) {
            const llama_pos pos_max = llama_memory_seq_pos_max(mem, 0);
            if (pos_max < requested_p0) {
                return true;
            }
        }

        llama_pos planned_p0 = requested_p0;
        llama_pos planned_p1 = -1;
        const bool planned = llama_memory_seq_rm_plan(mem, 0, requested_p0, -1, &planned_p0, &planned_p1);
        if (planned &&
                planned_p0 >= 0 && planned_p1 < 0 && planned_p0 < requested_p0) {
            if (llama_memory_seq_rm(mem, 0, planned_p0, planned_p1) &&
                    llama_memory_seq_pos_max(mem, 0) < requested_p0) {
                return true;
            }
        }

        LOG_WRN("DFlash2 local trim could not remove suffix: requested=%d exact=%s planned=%s planned_p0=%d planned_p1=%d pos_max=%d\n",
                (int) requested_p0, exact ? "true" : "false", planned ? "true" : "false",
                (int) planned_p0, (int) planned_p1, (int) llama_memory_seq_pos_max(mem, 0));

        return false;
    }

    // ---- observation-only shadow helpers ----
    //
    // The shadow observer stages target feature rows here (main thread, no
    // ctx_dft access) and injects them later, when no shadow worker is running.
    // Keeping the copies outside process()/draft() lets the auxiliary block run
    // on a worker while the primary drafter and the target continue.
    struct shadow_row {
        std::vector<float> features;      // [n_rows * n_embd_enc]
        std::vector<llama_pos> positions; // [n_rows]
    };

    std::vector<shadow_row> shadow_rows;
    size_t shadow_staged_tokens = 0;
    llama_pos shadow_trim_pos = -1;

    // Main-thread staging only. Drop provisional rows on rejection/replay;
    // the worker owns ctx_dft until its future completes.
    void shadow_clip(llama_pos end) {
        for (auto & row : shadow_rows) {
            const auto it = std::lower_bound(row.positions.begin(), row.positions.end(), end);
            row.positions.resize((size_t) (it - row.positions.begin()));
            row.features.resize(row.positions.size() * (size_t) n_embd_enc);
        }
        shadow_rows.erase(std::remove_if(shadow_rows.begin(), shadow_rows.end(),
                    [](const shadow_row & row) { return row.positions.empty(); }), shadow_rows.end());
        shadow_staged_tokens = 0;
        for (const auto & row : shadow_rows) shadow_staged_tokens += row.positions.size();
    }

    // per-stage counters for the observation worker (shadow mode only)
    bool     shadow_observer      = false;
    bool     shadow_draft_failed  = false; // worker-owned, published through its future
    uint64_t shadow_stage_decode_us   = 0; // DFlash transformer on the draft device
    uint64_t shadow_stage_selector_us = 0; // selector encode on the target device

    // Copy target-layer features for rows [offset, offset + n_rows) of seq 0.
    // Only reads ctx_tgt extraction buffers; never touches ctx_dft.
    bool shadow_stage(const llama_batch & batch_in, int32_t offset, int32_t n_rows) {
        if (!local_split || batch_in.token == nullptr || batch_in.embd != nullptr) {
            return false;
        }
        if (n_rows <= 0) {
            return true;
        }

        shadow_clip(batch_in.pos[offset]); // replay replaces previously staged rows

        auto * ctx_tgt = this->params.ctx_tgt;

        shadow_row row;
        row.features.resize((size_t) n_rows * n_embd_enc);
        row.positions.resize((size_t) n_rows);

        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            const float * layer = llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k]);
            if (!layer) {
                return false;
            }
            for (int32_t i = 0; i < n_rows; ++i) {
                float * dst = row.features.data() + (size_t) i * n_embd_enc + k * (size_t) n_embd_tgt;
                const float * src = layer + (size_t) (offset + i) * n_embd_tgt;
                std::memcpy(dst, src, (size_t) n_embd_tgt * sizeof(float));
            }
        }
        for (int32_t i = 0; i < n_rows; ++i) {
            row.positions[(size_t) i] = batch_in.pos[offset + i];
        }

        shadow_staged_tokens += (size_t) n_rows;
        shadow_rows.push_back(std::move(row));
        return true;
    }

    // Inject staged rows into ctx_dft. The caller guarantees that no shadow
    // worker is running and that local_split is active.
    bool shadow_inject() {
        auto * ctx_dft = params.ctx_dft;

        if (shadow_trim_pos >= 0) {
            if (!trim_local_context(shadow_trim_pos)) return false;
            shadow_trim_pos = -1;
        }

        // remove the speculative noise tail of the previous shadow block first
        if (local_draft_pending) {
            const llama_pos tail0 = local_draft_pos0;
            local_draft_pending = false;
            local_draft_pos0 = -1;
            local_n_proposed = 0;
            local_next_candidate = {};
            if (tail0 >= 0 && llama_memory_seq_pos_max(llama_get_memory(ctx_dft), 0) >= tail0 &&
                    !trim_local_context(tail0)) {
                return false;
            }
        }

        for (auto & row : shadow_rows) {
            if (row.positions.empty()) {
                continue;
            }
            if (row.positions.size() > llama_n_ubatch(ctx_dft)) {
                return false;
            }

            const llama_pos first_pos = row.positions.front();
            if (llama_memory_seq_pos_max(llama_get_memory(ctx_dft), 0) >= first_pos &&
                    !trim_local_context(first_pos)) {
                return false;
            }

            llama_batch sync = batch_inject;
            sync.n_tokens = (int32_t) row.positions.size();
            std::memcpy(sync.embd, row.features.data(), row.features.size() * sizeof(float));
            for (int32_t i = 0; i < sync.n_tokens; ++i) {
                sync.pos[i] = row.positions[(size_t) i];
                if (is_mrope) {
                    sync.pos[sync.n_tokens + i] = sync.pos[i];
                    sync.pos[2 * sync.n_tokens + i] = sync.pos[i];
                    sync.pos[3 * sync.n_tokens + i] = 0;
                }
                sync.n_seq_id[i] = 1;
                sync.seq_id[i][0] = 0;
                sync.logits[i] = true;
            }

            if (llama_decode(ctx_dft, sync) != 0) {
                return false;
            }
        }

        shadow_rows.clear();
        shadow_staged_tokens = 0;
        return true;
    }

    bool wait_local_pipeline() {
        if (!local_pipeline_future.valid()) {
            local_pipeline_inflight = false;
            return true;
        }

        try {
            local_pipeline_result result = local_pipeline_future.get();
            local_pipeline_output = std::move(result.hidden);
        } catch (const std::exception & error) {
            LOG_ERR("DFlash2 local pre-draft failed: %s; disabling local drafting\n", error.what());
            local_failed = true;
            local_pipeline_inflight = false;
            local_pipeline_ready = false;
            local_pipeline_output.clear();
            return false;
        }

        local_pipeline_inflight = false;
        return true;
    }

    bool apply_local_pipeline_syncs() {
        auto * ctx_dft = params.ctx_dft;
        try {
            for (const auto & request : local_pipeline_syncs) {
                if (request.positions.empty()) {
                    continue;
                }

                if (request.positions.size() > llama_n_batch(ctx_dft) ||
                        request.features.size() != request.positions.size() * (size_t) n_embd_enc) {
                    throw std::runtime_error("DFlash2 local SYNC has invalid feature dimensions");
                }
                // The worker has been joined. Reuse the normal injection
                // buffers and support overlapping checkpoint replay batches.
                if (llama_memory_seq_pos_max(llama_get_memory(ctx_dft), 0) >= request.positions.front() &&
                        !trim_local_context(request.positions.front())) {
                    throw std::runtime_error("DFlash2 local SYNC rewind failed");
                }
                llama_batch sync = batch_inject;
                sync.n_tokens = (int32_t) request.positions.size();
                std::memcpy(sync.embd, request.features.data(), request.features.size() * sizeof(float));
                for (int32_t i = 0; i < sync.n_tokens; ++i) {
                    sync.pos[i] = request.positions[(size_t) i];
                    if (is_mrope) {
                        sync.pos[sync.n_tokens + i] = sync.pos[i];
                        sync.pos[2 * sync.n_tokens + i] = sync.pos[i];
                        sync.pos[3 * sync.n_tokens + i] = 0;
                    }
                    sync.n_seq_id[i] = 1;
                    sync.seq_id[i][0] = 0;
                    sync.logits[i] = true;
                }

                const int rc = llama_decode(ctx_dft, sync);
                if (rc != 0) {
                    throw std::runtime_error("DFlash2 local SYNC decode failed");
                }
            }
            local_pipeline_syncs.clear();
            return true;
        } catch (const std::exception & error) {
            LOG_ERR("DFlash2 local deferred SYNC failed: %s\n", error.what());
            local_pipeline_syncs.clear();
            return false;
        }
    }

    bool discard_local_pipeline() {
        if (!local_pipeline_active()) {
            local_pipeline_syncs.clear();
            return true;
        }
        if (!wait_local_pipeline()) {
            local_pipeline_syncs.clear();
            return false;
        }

        // A promoted result retains only proposal bytes. Its real target KV
        // was already restored, so an anchor mismatch must not remove it.
        if (local_pipeline_state_synced) {
            // A new request can process prompt rows before begin() discards
            // this proposal. Apply those rows, including any prompt rewind.
            const bool ok = apply_local_pipeline_syncs();
            local_pipeline_output.clear();
            local_pipeline_pending = false;
            local_pipeline_ready = false;
            local_pipeline_state_synced = false;
            ++local_pipeline_discards;
            local_failed |= !ok;
            return ok;
        }

        // Deferred verification rows may start before the speculative seed.
        // Replay from the earliest row to avoid decoding duplicate positions.
        llama_pos replay_pos = local_pipeline_pos0 - 1;
        for (const auto & request : local_pipeline_syncs) {
            if (!request.positions.empty()) {
                replay_pos = std::min(replay_pos, request.positions.front());
            }
        }
        if (!trim_local_context(replay_pos)) {
            local_pipeline_syncs.clear();
            return false;
        }
        const bool ok = apply_local_pipeline_syncs();
        local_pipeline_output.clear();
        local_pipeline_pending = false;
        local_pipeline_ready = false;
        local_pipeline_discarded = true;
        ++local_pipeline_discards;
        local_failed |= !ok;
        return ok;
    }

    bool promote_local_pipeline() {
        if (!local_pipeline_active() || !wait_local_pipeline()) {
            return false;
        }
        if (local_pipeline_output.empty()) {
            return false;
        }

        // The worker injected the entire real prefix through the predicted
        // rejection point. Keep it, but remove all speculative noise KV even
        // when the proposal is reused in the next draft call.
        if (!trim_local_context(local_pipeline_pos0)) {
            return false;
        }
        local_pipeline_syncs.clear();
        local_pipeline_pending = false;
        local_pipeline_state_synced = true;
        local_pipeline_ready = true;
        local_pipeline_used = false;
        return true;
    }

    bool launch_local_pipeline(const llama_batch & verified) {
        const llama_token hypothesis = local_next_candidate.token;
        const int32_t accepted = local_next_candidate.accepted;
        const llama_pos pos0 = local_draft_pos0 + accepted + 1;
        if (!local_split || !local_prefetch || local_pipeline_active() ||
                local_draft_pos0 < 0 || hypothesis < 0 || accepted < 0 ||
                accepted >= local_n_proposed || local_pipeline_used ||
                (local_request_attempts >= 2 && local_request_hits * 4 < local_request_attempts)) {
            return false;
        }

        const int32_t n_rows = local_pipeline_n_max + 1;
        if (n_rows < 2 || n_rows > block_size || n_embd_dft <= 0) {
            return false;
        }

        // Require the whole real prefix in this verification chunk. Injecting
        // only the last row leaves the old anchor/noise KV in the draft cache.
        const int32_t first = dflash_pipeline_prefix_offset(verified.pos, verified.n_tokens,
                local_draft_pos0, accepted);
        const int32_t n_prefix = accepted + 1;
        if (first < 0) {
            return false;
        }
        const float * prefix_features = verified.embd + (size_t) first * n_embd_enc;
        std::vector<float> seed(prefix_features, prefix_features + (size_t) n_prefix * n_embd_enc);
        std::vector<float> mask = mask_embedding;
        auto * ctx_dft = params.ctx_dft;
        const auto * model_tgt = llama_get_model(params.ctx_tgt);
        const llama_token mask_token = mask_token_id;
        const int32_t embd_width = n_embd_dft;
        const int32_t enc_width = n_embd_enc;

        local_pipeline_pos0 = pos0;
        local_pipeline_id_last = hypothesis;
        local_pipeline_accept_k = accepted;
        local_pipeline_state_synced = false;
        local_pipeline_start = std::chrono::steady_clock::now();
        try {
            local_pipeline_future = std::async(std::launch::async,
                    [this, ctx_dft, model_tgt, pos0, hypothesis, mask_token, n_rows, n_prefix,
                     embd_width, enc_width, seed = std::move(seed), mask = std::move(mask)]() mutable {
                        local_pipeline_result result;
                        llama_batch inject{};
                        llama_batch draft{};
                        try {
                            const llama_pos seed_pos = pos0 - n_prefix;
                            if (seed_pos < 0 || !this->trim_local_context(seed_pos)) {
                                throw std::runtime_error("DFlash2 local pre-draft trim failed");
                            }

                            inject = llama_batch_init(n_prefix, enc_width, 1);
                            if (!inject.embd) {
                                throw std::runtime_error("DFlash2 local pre-draft feature buffer unavailable");
                            }
                            inject.n_tokens = n_prefix;
                            if (is_mrope) {
                                free(inject.pos);
                                inject.pos = (llama_pos *) malloc((size_t) 4 * n_prefix * sizeof(llama_pos));
                                if (!inject.pos) {
                                    throw std::bad_alloc();
                                }
                            }
                            std::memcpy(inject.embd, seed.data(), seed.size() * sizeof(float));
                            for (int32_t i = 0; i < n_prefix; ++i) {
                                inject.pos[i] = seed_pos + i;
                                if (is_mrope) {
                                    inject.pos[n_prefix + i] = seed_pos + i;
                                    inject.pos[2 * n_prefix + i] = seed_pos + i;
                                    inject.pos[3 * n_prefix + i] = 0;
                                }
                                inject.n_seq_id[i] = 1;
                                inject.seq_id[i][0] = 0;
                                inject.logits[i] = true;
                            }
                            if (llama_decode(ctx_dft, inject) != 0) {
                                throw std::runtime_error("DFlash2 local pre-draft feature decode failed");
                            }

                            draft = llama_batch_init(n_rows, embd_width, 1);
                            if (!draft.embd) {
                                throw std::runtime_error("DFlash2 local pre-draft embedding buffer unavailable");
                            }
                            draft.token = (llama_token *) std::malloc((size_t) n_rows * sizeof(llama_token));
                            if (!draft.token) {
                                throw std::bad_alloc();
                            }
                            draft.n_tokens = n_rows;
                            std::vector<float> anchor((size_t) embd_width);
                            if (!llama_model_get_token_embedding_row(model_tgt, hypothesis,
                                    anchor.data(), (size_t) embd_width)) {
                                throw std::runtime_error("DFlash2 local pre-draft anchor embedding lookup failed");
                            }
                            for (int32_t i = 0; i < n_rows; ++i) {
                                draft.token[i] = i == 0 ? hypothesis : mask_token;
                                draft.pos[i] = pos0 + i;
                                draft.n_seq_id[i] = 1;
                                draft.seq_id[i][0] = 0;
                                draft.logits[i] = true;
                                const float * row = i == 0 ? anchor.data() : mask.data();
                                std::memcpy(draft.embd + (size_t) i * embd_width,
                                        row, (size_t) embd_width * sizeof(float));
                            }
                            if (llama_decode(ctx_dft, draft) != 0) {
                                throw std::runtime_error("DFlash2 local pre-draft block decode failed");
                            }

                            float * hidden = llama_get_embeddings(ctx_dft);
                            if (!hidden) {
                                throw std::runtime_error("DFlash2 local pre-draft hidden output unavailable");
                            }
                            result.hidden.assign(hidden, hidden + (size_t) n_rows * embd_width);
                            llama_batch_free(draft);
                            llama_batch_free(inject);
                            return result;
                        } catch (...) {
                            llama_batch_free(draft);
                            llama_batch_free(inject);
                            throw;
                        }
                    });
            local_pipeline_inflight = true;
            local_pipeline_pending = true;
            ++local_pipeline_attempts;
            ++local_request_attempts;
            return true;
        } catch (const std::exception & error) {
            LOG_WRN("DFlash2 local pre-draft launch failed: %s; continuing synchronously\n", error.what());
            local_pipeline_inflight = false;
            return false;
        }
    }

    bool wait_remote_pipeline() {
        if (!remote_pipeline_future.valid()) {
            remote_pipeline_inflight = false;
            return !remote_failed;
        }

        const auto wait_start = std::chrono::steady_clock::now();
        try {
            remote_pipeline_result result = remote_pipeline_future.get();
            remote_pipeline_timing = result.timing;
            remote_pipeline_output = std::move(result.output);
            remote_pipeline_done = result.done;
        } catch (const std::exception & err) {
            LOG_ERR("DFlash2 Xbox pre-draft failed: %s; disabling remote pipeline\n", err.what());
            remote_pipeline_inflight = false;
            remote_pipeline_ready = false;
            remote_pipeline_output.clear();
            remote_failed = true;
            remote_pipeline_wait_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - wait_start).count();
            return false;
        }

        remote_pipeline_inflight = false;
        remote_pipeline_wait_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - wait_start).count();
        return true;
    }

    void queue_remote_pipeline_sync(dflash_remote::op operation, uint32_t n_tokens,
                                    std::vector<uint8_t> payload, int32_t last_pos) {
        remote_pipeline_syncs.push_back({operation, n_tokens, std::move(payload), last_pos});
    }

    bool apply_remote_pipeline_syncs() {
        try {
            for (auto & request : remote_pipeline_syncs) {
                if (profile_mode != remote_profile_mode::mock_local) {
                    const auto timing = remote->call(request.operation, ++rpc_request_id,
                            request.n_tokens, request.payload.data(),
                            (uint32_t) request.payload.size(), rpc_bytes, 0);
                    rpc_sync_us += timing.roundtrip_us;
                    rpc_sync_send_us += timing.send_us;
                    rpc_sync_wait_us += timing.header_wait_us + timing.payload_read_us;
                    rpc_sync_xbox_prepare_us += timing.server.xbox_prepare_us;
                    rpc_sync_xbox_compute_us += timing.server.xbox_compute_us;
                }
                remote_last_sync_pos = request.last_pos;
            }
            remote_pipeline_syncs.clear();
            return true;
        } catch (const std::exception & err) {
            LOG_ERR("DFlash2 Xbox deferred SYNC failed: %s; remote drafting disabled for this run\n", err.what());
            remote_failed = true;
            remote_pipeline_syncs.clear();
            return false;
        }
    }

    bool discard_remote_pipeline() {
        if (!remote_pipeline_active()) {
            remote_pipeline_syncs.clear();
            return true;
        }
        if (!wait_remote_pipeline()) {
            return false;
        }

        try {
            if (remote_pipeline_state_synced) {
                // A new prompt may have queued SYNC rows after the prior
                // promotion. Never drop those rows with the saved output.
                if (!remote_pipeline_syncs.empty() && !apply_remote_pipeline_syncs()) {
                    return false;
                }
            } else if (remote_pipeline_syncs.empty()) {
                if (profile_mode != remote_profile_mode::mock_local) {
                    const dflash_remote::trim_data request{remote_draft_pos0};
                    remote->call(dflash_remote::trim, ++rpc_request_id, 0,
                            &request, sizeof(request), rpc_bytes, 0);
                }
                remote_last_sync_pos = remote_draft_pos0 - 1;
            } else if (!apply_remote_pipeline_syncs()) {
                return false;
            }
        } catch (const std::exception & err) {
            LOG_ERR("DFlash2 Xbox pre-draft rollback failed: %s; remote drafting disabled for this run\n", err.what());
            remote_failed = true;
            return false;
        }

        remote_pipeline_output.clear();
        remote_pipeline_ready = false;
        remote_pipeline_state_synced = false;
        remote_pipeline_discarded = true;
        if (remote_pipeline_attempts >= 2 && remote_pipeline_hits * 4 < remote_pipeline_attempts) {
            remote_pipeline_enabled = false;
            LOG_INF("DFlash2 Xbox pre-draft disabled after %u attempts and %u reused drafts\n",
                    remote_pipeline_attempts, remote_pipeline_hits);
        }
        return !remote_failed;
    }

    bool remote_pipeline_has_real_prefix() const {
        if (remote_pipeline_pos0 <= remote_draft_pos0 || remote_pipeline_syncs.empty()) {
            return false;
        }
        const size_t rows_needed = size_t(remote_pipeline_pos0 - remote_draft_pos0);
        std::vector<bool> covered(rows_needed, false);
        const size_t row_bytes = sizeof(int32_t) + size_t(n_embd_enc) * sizeof(float);
        bool trimmed = false;
        for (const auto & request : remote_pipeline_syncs) {
            const size_t prefix = request.operation == dflash_remote::sync_trim ?
                sizeof(dflash_remote::sync_trim_data) : 0;
            if (request.payload.size() != prefix + size_t(request.n_tokens) * row_bytes) {
                return false;
            }
            if (prefix) {
                dflash_remote::sync_trim_data cut{};
                std::memcpy(&cut, request.payload.data(), sizeof(cut));
                trimmed |= cut.reset || (cut.pos0 >= 0 && cut.pos0 <= remote_draft_pos0);
                if (cut.reset || (cut.pos0 >= 0 && cut.pos0 <= remote_draft_pos0)) {
                    std::fill(covered.begin(), covered.end(), false);
                } else if (cut.pos0 > remote_draft_pos0 && cut.pos0 < remote_pipeline_pos0) {
                    std::fill(covered.begin() + (cut.pos0 - remote_draft_pos0),
                              covered.end(), false);
                }
            }
            for (uint32_t i = 0; i < request.n_tokens; ++i) {
                int32_t pos = -1;
                std::memcpy(&pos, request.payload.data() + prefix + size_t(i) * row_bytes,
                            sizeof(pos));
                if (pos >= remote_draft_pos0 && pos < remote_pipeline_pos0) {
                    covered[size_t(pos - remote_draft_pos0)] = true;
                }
            }
        }
        return trimmed && std::all_of(covered.begin(), covered.end(), [](bool value) { return value; });
    }

    bool promote_remote_pipeline() {
        if (!remote_pipeline_active() || !wait_remote_pipeline()) {
            return false;
        }
        if (remote_pipeline_output.empty()) {
            remote_failed = true;
            return false;
        }

        // The speculative output can be verified as a proposal, but its KV was
        // built with approximate features.  Commit the real target features
        // before the next cycle; only the proposal bytes may be reused.
        if (!remote_pipeline_has_real_prefix()) {
            LOG_ERR("DFlash2 Xbox pre-draft has incomplete real target SYNC; disabling remote drafting\n");
            discard_remote_pipeline();
            remote_failed = true;
            return false;
        }
        if (!apply_remote_pipeline_syncs()) {
            return false;
        }
        remote_draft_pending = false;
        remote_trimmed = true;
        remote_pipeline_state_synced = true;
        remote_pipeline_ready = true;
        return true;
    }

    bool launch_remote_pipeline(llama_token hypothesis, int32_t accepted) {
        if (!remote || !remote_pipeline_enabled || remote_failed ||
                remote_pipeline_active() || remote_pipeline_used ||
                accepted < 0 || accepted >= remote_n_proposed ||
                remote_draft_n_max < 1 || remote_draft_pos0 < 0) {
            return false;
        }

        // The candidate replaces the first rejected draft token after
        // 'accepted' accepted tokens; it is never the rejected top choice.
        const int32_t pos0 = remote_draft_pos0 + accepted + 1;
        if (pos0 <= remote_draft_pos0 || hypothesis < 0) {
            return false;
        }

        const auto * model_tgt = llama_get_model(params.ctx_tgt);
        std::vector<float> embeddings((size_t) (remote_draft_n_max + 1) * n_embd_dec);
        if (!llama_model_get_token_embedding_row(model_tgt, hypothesis,
                embeddings.data(), (size_t) n_embd_dec)) {
            LOG_WRN("DFlash2 Xbox pre-draft disabled: next-token hypothesis embedding lookup failed\n");
            return false;
        }
        for (int32_t i = 1; i <= remote_draft_n_max; ++i) {
            std::memcpy(embeddings.data() + (size_t) i * n_embd_dec,
                    mask_embedding.data(), (size_t) n_embd_dec * sizeof(float));
        }

        if (remote_pipeline_seed_features.size() != (size_t) n_embd_enc) {
            return false;
        }

        const size_t sync_row_bytes = sizeof(int32_t) + (size_t) n_embd_enc * sizeof(float);
        const dflash_remote::sync_trim_data trim{pos0, 0};
        const dflash_remote::draft_data draft{pos0, hypothesis, remote_draft_n_max};
        std::vector<uint8_t> payload(sizeof(trim) + sync_row_bytes + sizeof(draft) +
                embeddings.size() * sizeof(float));
        std::memcpy(payload.data(), &trim, sizeof(trim));
        std::memcpy(payload.data() + sizeof(trim), &pos0, sizeof(pos0));
        std::memcpy(payload.data() + sizeof(trim) + sizeof(pos0), remote_pipeline_seed_features.data(),
                (size_t) n_embd_enc * sizeof(float));
        std::memcpy(payload.data() + sizeof(trim) + sync_row_bytes, &draft, sizeof(draft));
        std::memcpy(payload.data() + sizeof(trim) + sync_row_bytes + sizeof(draft), embeddings.data(),
                embeddings.size() * sizeof(float));

        const uint32_t expected_bytes = (uint32_t) (embeddings.size() * sizeof(float));
        const uint64_t cycle_id = ++rpc_request_id;
        dflash_remote::client * client = remote.get();
        remote_pipeline_pos0 = pos0;
        remote_pipeline_id_last = hypothesis;
        remote_pipeline_accept_k = accepted;
        remote_pipeline_n_max = remote_draft_n_max;
        remote_pipeline_state_synced = false;
        remote_pipeline_start = std::chrono::steady_clock::now();
        try {
            remote_pipeline_future = std::async(std::launch::async,
                    [client, cycle_id, payload = std::move(payload), expected_bytes]() mutable {
                        remote_pipeline_result result;
                        result.timing = client->call(dflash_remote::sync_and_draft, cycle_id, 1,
                                payload.data(), (uint32_t) payload.size(), result.output,
                                expected_bytes);
                        result.done = std::chrono::steady_clock::now();
                        return result;
                    });
            remote_pipeline_inflight = true;
            ++remote_pipeline_attempts;
            return true;
        } catch (const std::exception & err) {
            LOG_WRN("DFlash2 Xbox pre-draft launch failed: %s; continuing synchronously\n", err.what());
            remote_pipeline_inflight = false;
            return false;
        }
    }

    // draft-dspark: the draft carries a Markov head and uses an anchor-first block layout
    bool is_dspark;

    // dspark speculators
    bool sample_from_anchor = true;

    // block-internal attention
    bool causal_attn = false;

    const int32_t * target_layer_ids   = nullptr; // model_dft's extract layer indices
    uint32_t        target_layer_ids_n = 0;

    common_speculative_impl_draft_dflash(const common_params_speculative & params, uint32_t n_seq,
            common_speculative_type type = COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH)
        : common_speculative_impl(type, n_seq, params.draft.n_max)
        , params(params.draft)
    {
        local_split    = this->params.local_split;
        local_prefetch = this->params.local_prefetch;

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "DFlash requires ctx_tgt and ctx_dft to be set");

        const llama_model * model_dft = llama_get_model(ctx_dft);
        const llama_model * model_tgt = llama_get_model(ctx_tgt);

        target_layer_ids   = llama_model_target_layer_ids  (model_dft);
        target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
        GGML_ASSERT(target_layer_ids_n > 0 && "DFlash model has no target_layer_ids");

        // Both lineages declare general.architecture = dflash, so the requested type cannot
        // pick the draft path. The Markov head is the on-disk marker and is already loaded.
        is_dspark = llama_model_has_dspark_markov_head(model_dft);
        const bool type_says_dspark = (type == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK);
        if (type_says_dspark != is_dspark) {
            LOG_WRN("%s: draft model carries %s, but --spec-type requested %s. Using %s, which is "
                    "what the model needs. The wrong path drops confidence truncation, and for "
                    "sample_from_anchor models it also reads the drafts one row late.\n", __func__,
                    is_dspark ? "a DSpark Markov head" : "no DSpark Markov head",
                    common_speculative_type_to_str(type).c_str(),
                    is_dspark ? "DSpark" : "DFlash");
        }

        n_embd_tgt    = llama_model_n_embd(model_tgt);
        n_embd_dec    = llama_model_n_embd_out(model_dft);
        n_embd_dft    = llama_model_n_embd(model_dft);
        n_embd_enc    = (int32_t) target_layer_ids_n * n_embd_tgt;

        // read the trained block size from the dflash.block_size metadata key
        block_size = 16;
        {
            char buf[32] = {};
            if (llama_model_meta_val_str(model_dft, "dflash.block_size", buf, sizeof(buf)) >= 0) {
                block_size = std::atoi(buf);
            }
            if (llama_model_meta_val_str(model_dft, "dflash.sample_from_anchor", buf, sizeof(buf)) >= 0) {
                sample_from_anchor = std::strcmp(buf, "true") == 0;
            }
        }
        causal_attn = common_speculative_dflash_causal_attn(model_dft);

        selector_top_k = llama_model_dflash_selector_top_k(model_dft);
        is_dflash2     = selector_top_k > 0;
        mask_token_id = llama_vocab_mask(llama_model_get_vocab(model_dft));

        if (is_dspark && this->params.p_min > 0.0f) {
            char buf[16] = {};
            const bool has_conf =
                llama_model_meta_val_str(model_dft, "dflash.has_confidence_head", buf, sizeof(buf)) < 0 ||
                std::strcmp(buf, "true") == 0;
            if (!has_conf) {
                throw std::runtime_error("DSpark draft has no confidence head: please set --spec-draft-p-min 0");
            }
        }

        LOG_INF("%s: adding speculative implementation '%s'\n", __func__, common_speculative_type_to_str(type).c_str());
        if (!common_speculative_dflash_adaptive_dm_supported(selector_top_k)) {
            LOG_INF("%s: DFlash2 uses its fixed block limit and selector confidence; Bee adaptive draft-max is disabled\n", __func__);
        }
        LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%.2f\n", __func__, this->params.n_max, this->params.n_min, this->params.p_min);
        LOG_INF("%s: - block_size=%d, mask_token_id=%d, n_extract=%u, sample_from_anchor=%s, lineage=%s\n", __func__,
                block_size, mask_token_id, target_layer_ids_n, sample_from_anchor ? "true" : "false",
                is_dspark ? "dspark" : "dflash");

        // DFlash input is [id_last, <mask> * (block_size-1)]: in-place denoising yields at most
        // block_size-1 draft tokens, anchor-first DSpark yields a full block_size draft tokens
        const int32_t n_draft_max = is_dspark && sample_from_anchor ? block_size : block_size - 1;
        if (this->params.n_max > n_draft_max || this->params.n_min > n_draft_max) {
            LOG_WRN("%s: requested draft size (n_max=%d, n_min=%d) exceeds the trained block size %d -- clamping to %d\n",
                    __func__, this->params.n_max, this->params.n_min, block_size, n_draft_max);
            this->params.n_max = std::min(this->params.n_max, n_draft_max);
            this->params.n_min = std::min(this->params.n_min, n_draft_max);
        }
        this->n_max = this->params.n_max;

        if (local_split && (!is_dflash2 || is_dspark || n_seq != 1 || n_embd_dft != n_embd_tgt ||
                n_embd_dec != n_embd_tgt || block_size < 2 || block_size > 64)) {
            throw std::invalid_argument(
                    "DFlash2 local split requires one plain DFlash2 sequence with matching target/draft widths");
        }
        if (local_prefetch && !local_split) {
            throw std::invalid_argument("DFlash2 local prefetch requires local split mode");
        }

        batch        = llama_batch_init(llama_n_batch(ctx_dft), local_split ? n_embd_dft : 0, n_seq);
        if (local_split) {
            batch.token = (llama_token *) std::malloc((size_t) llama_n_batch(ctx_dft) * sizeof(llama_token));
            if (!batch.token) {
                llama_batch_free(batch);
                batch = {};
                throw std::bad_alloc();
            }
        }
        batch_inject = llama_batch_init(llama_n_ubatch(ctx_dft), n_embd_enc, n_seq);

        // embd batches on an M-RoPE draft need 4 position rows per token
        is_mrope = llama_model_rope_type(model_dft) == LLAMA_ROPE_TYPE_MROPE;
        if (is_mrope) {
            free(batch_inject.pos);
            batch_inject.pos = (llama_pos *) malloc(sizeof(llama_pos) * 4 * llama_n_batch(ctx_dft));
        }

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = is_dflash2 ? selector_top_k : 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(model_dft, sparams));
        }

        selector_rng.resize(n_seq);
        selector_reset.assign(n_seq, true);

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        int32_t target_device_count = 0;
        for (int32_t i = 0; i < llama_model_n_devices(model_tgt); ++i) {
            ggml_backend_dev_t dev = llama_model_get_device(model_tgt, i);
            target_device_count += ggml_backend_dev_is_meta(dev)
                ? (int32_t) ggml_backend_meta_device_count(dev)
                : 1;
        }
        const bool use_backend_sampling = common_speculative_dflash_backend_sampling_allowed(
                this->params.backend_sampling, target_device_count, is_dflash2);
        if (this->params.backend_sampling && !use_backend_sampling && !is_dflash2) {
            SPC_WRN("%s\n", "target output is split across devices; using CPU draft sampler");
        }
        if (use_backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        // turn on extraction of the target layers' input embeddings
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);
        }

        // DFlash2 reads its selector lattice from h_nextn and never consumes raw logits.
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ !is_dflash2);
        llama_set_causal_attn(ctx_dft, causal_attn); // DFlash needs non-causal attention unless the model says otherwise

        const char * remote_endpoint = std::getenv("DFLASH_XBOX_RPC");
        if (local_split && remote_endpoint && *remote_endpoint) {
            throw std::invalid_argument("DFlash2 local split cannot be combined with DFLASH_XBOX_RPC");
        }

        if (local_split) {
            auto selector_params = llama_context_default_params();
            selector_params.n_ctx = block_size;
            selector_params.n_batch = block_size;
            selector_params.n_ubatch = block_size;
            selector_params.n_seq_max = 1;
            selector_params.pooling_type = LLAMA_POOLING_TYPE_NONE;
            selector_params.ctx_other = ctx_tgt;
            selector_params.dflash_selector_only = true;
            selector_owned.reset(llama_init_from_model(const_cast<llama_model *>(model_dft), selector_params));
            if (!selector_owned) {
                throw std::runtime_error("failed to initialize local DFlash2 selector-only context");
            }
            selector_ctx = selector_owned.get();

            mask_embedding.resize(n_embd_dft);
            if (!llama_model_get_token_embedding_row(model_tgt, mask_token_id,
                    mask_embedding.data(), mask_embedding.size())) {
                throw std::runtime_error("target MASK embedding is unavailable for local DFlash2 split");
            }
            selector_batch = llama_batch_init(block_size, n_embd_dft, 1);
            selector_batch.token = (llama_token *) std::malloc((size_t) block_size * sizeof(llama_token));
            if (!selector_batch.token) {
                llama_batch_free(selector_batch);
                selector_batch = {};
                throw std::bad_alloc();
            }
            local_pipeline_n_max = this->params.n_max;
            LOG_INF("%s: DFlash2 explicit local split enabled%s\n", __func__,
                    local_prefetch ? " with local prefetch" : "");
        }

        if (remote_endpoint && *remote_endpoint) {
            const char * mode = std::getenv("DFLASH_XBOX_MOCK_MODE");
            if (mode && std::strcmp(mode, "xbox") == 0) profile_mode = remote_profile_mode::mock_xbox;
            else if (mode && std::strcmp(mode, "local") == 0) profile_mode = remote_profile_mode::mock_local;
            else if (mode && *mode && std::strcmp(mode, "real") != 0) {
                throw std::runtime_error("DFLASH_XBOX_MOCK_MODE must be real, xbox, or local");
            }
            if (!is_dflash2 || is_dspark || n_seq != 1 || n_embd_dec != n_embd_tgt ||
                    block_size < 2 || block_size > 64) {
                throw std::runtime_error("DFLASH_XBOX_RPC requires one plain DFlash2 sequence and matching target/draft widths");
            }
            std::string host(remote_endpoint);
            uint16_t port = 50053;
            const size_t colon = host.rfind(':');
            if (colon != std::string::npos) {
                const std::string port_text = host.substr(colon + 1);
                char * end = nullptr;
                const long parsed = std::strtol(port_text.c_str(), &end, 10);
                if (!end || *end || parsed < 1 || parsed > 65535) {
                    throw std::runtime_error("DFLASH_XBOX_RPC must be IPv4[:port]");
                }
                port = (uint16_t) parsed;
                host.resize(colon);
            }
            remote = std::make_unique<dflash_remote::client>(host, port);
            std::vector<uint8_t> hello_bytes;
            remote->call(dflash_remote::hello, ++rpc_request_id, 0, nullptr, 0,
                    hello_bytes, sizeof(dflash_remote::hello_data));
            dflash_remote::hello_data hello{};
            std::memcpy(&hello, hello_bytes.data(), sizeof(hello));
            if (hello.n_embd_enc != (uint32_t) n_embd_enc || hello.n_embd_dec != (uint32_t) n_embd_dec ||
                    hello.selector_top_k != (uint32_t) selector_top_k || hello.mask_token_id != mask_token_id ||
                    hello.n_ctx < (uint32_t) block_size) {
                throw std::runtime_error("DFlash RPC model geometry differs from local selector model");
            }
            remote->call(dflash_remote::reset, ++rpc_request_id, 0, nullptr, 0, rpc_bytes, 0);
            remote->call(dflash_remote::mock, ++rpc_request_id,
                    profile_mode == remote_profile_mode::mock_xbox ? 1 : 0,
                    nullptr, 0, rpc_bytes, 0);
            // Old v2 apps reject this optional opcode without changing framing.
            // Keep an explicit legacy path for reproducible A/B measurements.
            const char * legacy_trim = std::getenv("DFLASH_XBOX_LEGACY_TRIM");
            if (!legacy_trim || std::strcmp(legacy_trim, "1") != 0) {
                try {
                    remote->call(dflash_remote::sync_trim, ++rpc_request_id, 0,
                            nullptr, 0, rpc_bytes, 0);
                    remote_fused_sync = true;
                } catch (const dflash_remote::server_error & err) {
                    if (err.status != -1) throw;
                }
            }
            LOG_INF("DFlash2 Xbox fused TRIM/SYNC: %s\n", remote_fused_sync ? "enabled" : "legacy");
            const char * separate = std::getenv("DFLASH_XBOX_SEPARATE_DRAFT");
            if (remote_fused_sync && (!separate || std::strcmp(separate, "1") != 0)) {
                try {
                    remote->call(dflash_remote::sync_and_draft, ++rpc_request_id, 0,
                            nullptr, 0, rpc_bytes, 0);
                    remote_fused_draft = true;
                } catch (const dflash_remote::server_error & err) {
                    if (err.status != -1) throw;
                }
            }
            LOG_INF("DFlash2 Xbox fused SYNC/DRAFT: %s\n", remote_fused_draft ? "enabled" : "legacy");
            const char * pipeline = std::getenv("DFLASH_XBOX_PIPELINE");
            remote_pipeline_enabled = remote_fused_draft && profile_mode != remote_profile_mode::mock_local &&
                    pipeline && std::strcmp(pipeline, "1") == 0;
            LOG_INF("DFlash2 Xbox async pre-draft pipeline: %s\n",
                    remote_pipeline_enabled ? "enabled (greedy only)" : "disabled");

            auto selector_params = llama_context_default_params();
            selector_params.n_ctx = block_size;
            selector_params.n_batch = block_size;
            selector_params.n_ubatch = block_size;
            selector_params.n_seq_max = 1;
            selector_params.pooling_type = LLAMA_POOLING_TYPE_NONE;
            selector_params.ctx_other = ctx_tgt;
            selector_params.dflash_selector_only = true;
            selector_owned.reset(llama_init_from_model(const_cast<llama_model *>(model_dft), selector_params));
            if (!selector_owned) {
                throw std::runtime_error("failed to initialize DFlash2 selector-only context");
            }
            selector_ctx = selector_owned.get();
            mask_embedding.resize(n_embd_dec);
            if (!llama_model_get_token_embedding_row(model_tgt, mask_token_id,
                    mask_embedding.data(), mask_embedding.size())) {
                throw std::runtime_error("target MASK embedding is unavailable for DFlash RPC");
            }
            const char * metrics_path = std::getenv("DFLASH_XBOX_METRICS");
            if (metrics_path && *metrics_path) {
                rpc_metrics.open(metrics_path, std::ios::app);
                if (!rpc_metrics) {
                    throw std::runtime_error("cannot open DFLASH_XBOX_METRICS JSONL path");
                }
            }
            const char * capture_path = std::getenv("DFLASH_XBOX_CAPTURE");
            if (profile_mode == remote_profile_mode::mock_local) {
                if (!capture_path || !*capture_path) {
                    throw std::runtime_error("local DFlash mock requires DFLASH_XBOX_CAPTURE");
                }
                rpc_replay.open(capture_path, std::ios::binary);
                if (!rpc_replay) throw std::runtime_error("cannot open DFlash capture for replay");
            } else if (profile_mode == remote_profile_mode::real && capture_path && *capture_path) {
                rpc_capture.open(capture_path, std::ios::binary | std::ios::trunc);
                if (!rpc_capture) throw std::runtime_error("cannot open DFlash capture for recording");
            }
            selector_batch = llama_batch_init(block_size, n_embd_dec, 1);
            selector_batch.token = (llama_token *) std::malloc((size_t) block_size * sizeof(llama_token));
            if (!selector_batch.token) {
                llama_batch_free(selector_batch);
                selector_batch = {};
                throw std::bad_alloc();
            }
            LOG_INF("DFlash2 Xbox RPC connected to %s:%u (protocol=2, model_bytes=%llu, ctx=%u)\n",
                    host.c_str(), port, (unsigned long long) hello.model_bytes, hello.n_ctx);
        }
    }

    ~common_speculative_impl_draft_dflash() override {
        if (local_pipeline_inflight) {
            wait_local_pipeline();
        }
        if (local_split && local_prefetch) {
            LOG_INF("DFlash2 local prefetch: attempts=%u hits=%u discards=%u\n",
                    local_pipeline_attempts, local_pipeline_hits, local_pipeline_discards);
        }
        // The worker holds the persistent RPC client.  Join it before any
        // context or client-owned buffers are destroyed.
        if (remote_pipeline_inflight) {
            wait_remote_pipeline();
        }
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        llama_batch_free(batch);
        llama_batch_free(batch_inject);
        if (selector_batch.embd) {
            llama_batch_free(selector_batch);
        }
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        // begin() can run after process() has already queued prompt features.
        // Drain the old worker and apply those rows before starting a new draft.
        if (remote && remote_pipeline_active() && !discard_remote_pipeline()) {
            remote_failed = true;
        }

        if (local_split) {
            if (local_pipeline_active()) {
                local_failed |= !discard_local_pipeline();
            }
            local_pipeline_syncs.clear();
            local_draft_pending = false;
            local_draft_pos0 = -1;
            local_n_proposed = 0;
            local_next_candidate = {};
            local_request_attempts = 0;
            local_request_hits = 0;
            local_pipeline_used = false;
        }

        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }

        selector_reset[seq_id] = true;

        const llama_pos pos_max = remote ? N - 1 : llama_memory_seq_pos_max(llama_get_memory(params.ctx_dft), seq_id);
        if (pos_max < N - 1) {
            LOG_WRN("%s: ctx_dft pos_max=%d < N-1=%d - process() did not run on every prefill ubatch. "
                    "Drafts may degrade.\n",
                    __func__, (int) pos_max, N - 1);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        if (local_split && local_failed) {
            return true;
        }

        // The target processes accepted tokens before common_speculative_accept()
        // runs. Remove the previous speculative tail before their features arrive.
        if (remote && remote_draft_pending && !remote_trimmed && !remote_fused_sync) {
            try {
                if (profile_mode != remote_profile_mode::mock_local) {
                    const dflash_remote::trim_data request{remote_draft_pos0};
                    const auto timing = remote->call(dflash_remote::trim, ++rpc_request_id, 0,
                            &request, sizeof(request), rpc_bytes, 0);
                    remote_trim_us = timing.roundtrip_us;
                }
                remote_trimmed = true;
            } catch (const std::exception & err) {
                LOG_ERR("DFlash2 Xbox TRIM failed: %s; remote drafting disabled for this run\n", err.what());
                remote_failed = true;
                return true;
            }
        }

        // Target prefill may contain token IDs or multimodal embeddings. Both
        // produce the target-layer features used to seed the draft KV cache, so
        // embeddings are injected too, except the pinned ones skipped below.
        // TODO: revisit after https://github.com/ggml-org/llama.cpp/pull/24669 is merged
        const bool has_tokens     = batch_in.token != nullptr;
        const bool has_embeddings = batch_in.embd  != nullptr;
        if (has_tokens == has_embeddings) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // per-seq inclusive batch range (assumes each seq's tokens are contiguous in the batch)
        std::vector<int32_t> i_batch_beg(n_seq, -1);
        std::vector<int32_t> i_batch_end(n_seq, -1);
        for (int32_t k = 0; k < n_tokens; ++k) {
            GGML_ASSERT(batch_in.n_seq_id[k] == 1);
            const llama_seq_id seq_id = batch_in.seq_id[k][0];
            if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
                continue;
            }
            i_batch_end[seq_id] = k;
            if (i_batch_beg[seq_id] < 0) {
                i_batch_beg[seq_id] = k;
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        if (remote && remote_failed) {
            return true;
        }
        const int32_t n_ubatch = std::min<int32_t>((int32_t) llama_n_ubatch(ctx_dft), remote ? 64 : INT32_MAX);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_beg[seq_id] < 0) {
                continue;
            }
            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;

            // an M-RoPE image pins all its rows to one position, so a windowed draft
            // cache cannot free cells for it - skip it, the draft can jump over the gap
            const bool pos_pinned = batch_in.pos[i_batch_beg[seq_id]] == batch_in.pos[i_batch_end[seq_id]];
            if (has_embeddings && n_rows > 1 && pos_pinned) {
                continue;
            }

            for (int32_t offset = 0; offset < n_rows; offset += n_ubatch) {
                const int32_t n_chunk = std::min(n_ubatch, n_rows - offset);
                const auto capture_start = std::chrono::steady_clock::now();

                // gather target features per extract layer; the fused decode encodes and
                // injects them into the K/V cache at the target positions
                batch_inject.n_tokens = n_chunk;
                for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
                    const float * layer = llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k]);
                    if (!layer) {
                        GGML_ABORT("DFlash: target layer %d input not extracted.", target_layer_ids[k]);
                    }
                    for (int32_t i = 0; i < n_chunk; ++i) {
                        float       * dst = batch_inject.embd + (size_t) i * n_embd_enc + k * (size_t) n_embd_tgt;
                        const float * src = layer + (size_t) (i_batch_beg[seq_id] + offset + i) * n_embd_tgt;
                        std::memcpy(dst, src, (size_t) n_embd_tgt * sizeof(float));
                    }
                }

                for (int32_t i = 0; i < n_chunk; ++i) {
                    const llama_pos p = batch_in.pos[i_batch_beg[seq_id] + offset + i];
                    batch_inject.pos[i] = p;
                    if (is_mrope) {
                        batch_inject.pos[1 * n_chunk + i] = p;
                        batch_inject.pos[2 * n_chunk + i] = p;
                        batch_inject.pos[3 * n_chunk + i] = 0;
                    }
                    batch_inject.n_seq_id[i]  = 1;
                    batch_inject.seq_id[i][0] = seq_id;
                    batch_inject.logits[i]    = local_split;
                }
                if (local_split) {
                    bool pipeline_active = local_pipeline_active();
                    if (local_prefetch && local_draft_pending && local_next_candidate.score >= 0.05f &&
                            !pipeline_active) {
                        pipeline_active = launch_local_pipeline(batch_inject);
                    }

                    if (pipeline_active) {
                        local_pipeline_sync request;
                        request.features.resize((size_t) n_chunk * n_embd_enc);
                        request.positions.resize((size_t) n_chunk);
                        std::memcpy(request.features.data(), batch_inject.embd,
                                request.features.size() * sizeof(float));
                        for (int32_t i = 0; i < n_chunk; ++i) {
                            request.positions[(size_t) i] = batch_inject.pos[i];
                        }
                        local_pipeline_syncs.push_back(std::move(request));
                        continue;
                    }

                    const llama_pos first_pos = batch_inject.pos[0];
                    if (llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id) >= first_pos &&
                            !trim_local_context(first_pos)) {
                        LOG_ERR("DFlash2 local split cannot replace draft KV from position %d\n", (int) first_pos);
                        return false;
                    }
                    const int32_t rc = llama_decode(ctx_dft, batch_inject);
                    if (rc != 0) {
                        LOG_ERR("%s: llama_decode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                                __func__, rc, (int) n_chunk, (int) offset);
                        return false;
                    }
                    continue;
                }
                if (remote && remote_pipeline_enabled) {
                    remote_pipeline_seed_features.resize((size_t) n_embd_enc);
                    std::memcpy(remote_pipeline_seed_features.data(),
                            batch_inject.embd + (size_t) (n_chunk - 1) * n_embd_enc,
                            (size_t) n_embd_enc * sizeof(float));
                }
                if (remote) {
                    try {
                        const bool pipeline_active = remote_pipeline_active();
                        const auto capture_done = std::chrono::steady_clock::now();
                        rpc_capture_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                                capture_done - capture_start).count();
                        // Keep only the last SYNC chunk for the next DRAFT. Flush
                        // older chunks before applying any later rewind/trim.
                        if (remote_sync_tokens) {
                            if (pipeline_active) {
                                queue_remote_pipeline_sync(
                                        remote_sync_has_trim ? dflash_remote::sync_trim : dflash_remote::sync,
                                        remote_sync_tokens, std::move(remote_sync_payload), remote_sync_last_pos);
                            } else if (profile_mode != remote_profile_mode::mock_local) {
                                const auto timing = remote->call(remote_sync_has_trim ? dflash_remote::sync_trim : dflash_remote::sync,
                                        ++rpc_request_id, remote_sync_tokens, remote_sync_payload.data(),
                                        (uint32_t) remote_sync_payload.size(), rpc_bytes, 0);
                                rpc_sync_us += timing.roundtrip_us;
                                rpc_sync_send_us += timing.send_us;
                                rpc_sync_wait_us += timing.header_wait_us + timing.payload_read_us;
                                rpc_sync_xbox_prepare_us += timing.server.xbox_prepare_us;
                                rpc_sync_xbox_compute_us += timing.server.xbox_compute_us;
                            }
                            remote_sync_tokens = 0;
                            remote_sync_payload.clear();
                            remote_sync_last_pos = -1;
                        }
                        const int32_t first_pos = batch_inject.pos[0];
                        const bool rewind = first_pos <= remote_last_sync_pos;
                        const bool trim_pending = remote_draft_pending && !remote_trimmed;
                        const bool fused = remote_fused_sync && (trim_pending || rewind);
                        dflash_remote::sync_trim_data cut{first_pos, rewind && first_pos == 0 ? 1u : 0u};
                        if (trim_pending) cut.pos0 = std::min(cut.pos0, remote_draft_pos0);
                        if (rewind && !fused) {
                            if (profile_mode != remote_profile_mode::mock_local) {
                                const auto rewind_start = std::chrono::steady_clock::now();
                                if (first_pos == 0) {
                                    remote->call(dflash_remote::reset, ++rpc_request_id, 0, nullptr, 0, rpc_bytes, 0);
                                } else {
                                    const dflash_remote::trim_data rewind{first_pos};
                                    remote->call(dflash_remote::trim, ++rpc_request_id, 0,
                                            &rewind, sizeof(rewind), rpc_bytes, 0);
                                }
                                rpc_rewind_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                                        std::chrono::steady_clock::now() - rewind_start).count();
                            }
                            remote_last_sync_pos = first_pos - 1;
                        }
                        const auto sync_serialize_start = std::chrono::steady_clock::now();
                        const size_t row_bytes = sizeof(int32_t) + (size_t) n_embd_enc * sizeof(float);
                        const size_t prefix_bytes = fused ? sizeof(cut) : 0;
                        std::vector<uint8_t> payload(prefix_bytes + (size_t) n_chunk * row_bytes);
                        if (fused) std::memcpy(payload.data(), &cut, sizeof(cut));
                        for (int32_t i = 0; i < n_chunk; ++i) {
                            uint8_t * row = payload.data() + prefix_bytes + (size_t) i * row_bytes;
                            std::memcpy(row, &batch_inject.pos[i], sizeof(int32_t));
                            std::memcpy(row + sizeof(int32_t), batch_inject.embd + (size_t) i * n_embd_enc,
                                    (size_t) n_embd_enc * sizeof(float));
                        }
                        rpc_sync_serialize_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now() - sync_serialize_start).count();
                        if (pipeline_active) {
                            queue_remote_pipeline_sync(fused ? dflash_remote::sync_trim : dflash_remote::sync,
                                    (uint32_t) n_chunk, std::move(payload), batch_inject.pos[n_chunk - 1]);
                        } else if (remote_fused_draft) {
                            remote_sync_payload = std::move(payload);
                            remote_sync_tokens = (uint32_t) n_chunk;
                            remote_sync_has_trim = fused;
                            remote_sync_last_pos = batch_inject.pos[n_chunk - 1];
                        } else if (profile_mode != remote_profile_mode::mock_local) {
                            const auto timing = remote->call(fused ? dflash_remote::sync_trim : dflash_remote::sync, ++rpc_request_id,
                                    (uint32_t) n_chunk, payload.data(), (uint32_t) payload.size(), rpc_bytes, 0);
                            rpc_sync_us += timing.roundtrip_us;
                            rpc_sync_send_us += timing.send_us;
                            rpc_sync_wait_us += timing.header_wait_us + timing.payload_read_us;
                            rpc_sync_xbox_prepare_us += timing.server.xbox_prepare_us;
                            rpc_sync_xbox_compute_us += timing.server.xbox_compute_us;
                        }
                        if (fused && trim_pending) remote_trimmed = true;
                        remote_last_sync_pos = batch_inject.pos[n_chunk - 1];
                    } catch (const std::exception & err) {
                        LOG_ERR("DFlash2 Xbox SYNC failed: %s; remote drafting disabled for this run\n", err.what());
                        remote_failed = true;
                        return true;
                    }
                } else {
                    const int32_t rc = llama_decode(ctx_dft, batch_inject);
                    if (rc != 0) {
                        LOG_ERR("%s: llama_decode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                                __func__, rc, (int) n_chunk, (int) offset);
                        return false;
                    }
                }
            }
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        shadow_draft_failed = false;
        if ((remote && remote_failed) || (local_split && local_failed)) {
            shadow_draft_failed = true;
            return;
        }
        common_batch_clear(batch);

        // build one batch holding every drafting sequence's noise block into a single decode)
        // record where each block starts and its size
        std::vector<int32_t> i_block_beg(n_seq, -1);
        std::vector<int32_t> n_block    (n_seq,  0);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            common_sampler_reset(smpls[seq_id].get());

            const int32_t n = (int32_t) dp.pos0;

            const int32_t n_draft = params.n_max;

            const int32_t n_block_tokens = n_draft + (is_dspark && sample_from_anchor ? 0 : 1);
            i_block_beg[seq_id] = batch.n_tokens;
            n_block    [seq_id] = n_block_tokens;
            for (int32_t i = 0; i < n_block_tokens; ++i) {
                common_batch_add(batch, i == 0 ? dp.id_last : mask_token_id, n + i, { seq_id }, !is_dflash2);
            }
        }

        if (batch.n_tokens == 0) {
            return;
        }
        if (remote) {
            remote_current_anchor_id = batch.token[0];
        }

        // decode all sequences' noise blocks locally, or run only the DFlash
        // transformer on Xbox and the original output-head/selector on this host.
        if (remote) {
            try {
                const int32_t n_rows = batch.n_tokens;
                const uint32_t expected_bytes = (uint32_t) ((size_t) n_rows * n_embd_dec * sizeof(float));
                bool used_pipeline_output = false;
                if (remote_pipeline_ready) {
                    remote_pipeline_actual_pos = batch.pos[0];
                    remote_pipeline_actual_id = batch.token[0];
                    remote_pipeline_match_mask =
                            (batch.pos[0] == remote_pipeline_pos0 ? 1u : 0u) |
                            (batch.token[0] == remote_pipeline_id_last ? 2u : 0u) |
                            (n_rows == remote_pipeline_n_max + 1 ? 4u : 0u) |
                            (remote_pipeline_output.size() == expected_bytes ? 8u : 0u);
                    if (remote_pipeline_match_mask == 15u) {
                        remote_draft_start = remote_pipeline_start;
                        remote_draft_done = remote_pipeline_done;
                        remote_draft_timing = remote_pipeline_timing;
                        rpc_bytes = std::move(remote_pipeline_output);
                        remote_pipeline_ready = false;
                        remote_pipeline_state_synced = false;
                        remote_pipeline_used = true;
                        ++remote_pipeline_hits;
                        remote_pipeline_syncs.clear();
                        used_pipeline_output = true;
                    } else if (!discard_remote_pipeline()) {
                        throw std::runtime_error("DFlash2 Xbox pre-draft state could not be discarded");
                    }
                }

                dflash_remote::draft_data draft_request{};
                if (!used_pipeline_output) {
                    remote_draft_start = std::chrono::steady_clock::now();
                    const auto * model_tgt = llama_get_model(params.ctx_tgt);
                    std::vector<float> embeddings((size_t) n_rows * n_embd_dec);
                    if (!llama_model_get_token_embedding_row(model_tgt, batch.token[0],
                            embeddings.data(), (size_t) n_embd_dec)) {
                        throw std::runtime_error("target anchor embedding lookup failed");
                    }
                    for (int32_t i = 1; i < n_rows; ++i) {
                        std::memcpy(embeddings.data() + (size_t) i * n_embd_dec,
                                mask_embedding.data(), (size_t) n_embd_dec * sizeof(float));
                    }
                    remote_prepare_us = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now() - remote_draft_start).count();
                    draft_request = {batch.pos[0], batch.token[0], n_rows - 1};
                    const auto serialize_start = std::chrono::steady_clock::now();
                    std::vector<uint8_t> payload;
                    if (remote_fused_draft) {
                        if (!remote_sync_tokens || !remote_sync_has_trim) {
                            const dflash_remote::sync_trim_data cut{-1, 0};
                            payload.resize(sizeof(cut));
                            std::memcpy(payload.data(), &cut, sizeof(cut));
                        }
                        payload.insert(payload.end(), remote_sync_payload.begin(), remote_sync_payload.end());
                    }
                    const size_t draft_offset = payload.size();
                    payload.resize(draft_offset + sizeof(draft_request) + embeddings.size() * sizeof(float));
                    std::memcpy(payload.data() + draft_offset, &draft_request, sizeof(draft_request));
                    std::memcpy(payload.data() + draft_offset + sizeof(draft_request), embeddings.data(),
                            embeddings.size() * sizeof(float));
                    remote_serialize_us = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now() - serialize_start).count();
                    struct replay_header { int32_t pos0; int32_t id_last; int32_t n_max; uint32_t bytes; };
                    if (profile_mode == remote_profile_mode::mock_local) {
                        replay_header saved{};
                        rpc_replay.read(reinterpret_cast<char *>(&saved), sizeof(saved));
                        if (!rpc_replay || saved.pos0 != draft_request.pos0 ||
                                saved.id_last != draft_request.id_last || saved.n_max != draft_request.n_max ||
                                saved.bytes != expected_bytes) {
                            throw std::runtime_error("local DFlash replay trajectory mismatch");
                        }
                        rpc_bytes.resize(saved.bytes);
                        rpc_replay.read(reinterpret_cast<char *>(rpc_bytes.data()), saved.bytes);
                        if (!rpc_replay) throw std::runtime_error("local DFlash replay truncated");
                        remote_draft_timing = {};
                        remote_draft_timing.server.cycle_id = ++rpc_request_id;
                    } else {
                        remote_draft_timing = remote->call(remote_fused_draft ? dflash_remote::sync_and_draft : dflash_remote::draft, ++rpc_request_id,
                                remote_fused_draft ? remote_sync_tokens : (uint32_t) n_rows,
                                payload.data(), (uint32_t) payload.size(), rpc_bytes,
                                expected_bytes);
                        if (rpc_capture) {
                            const replay_header saved{draft_request.pos0, draft_request.id_last,
                                                      draft_request.n_max, expected_bytes};
                            rpc_capture.write(reinterpret_cast<const char *>(&saved), sizeof(saved));
                            rpc_capture.write(reinterpret_cast<const char *>(rpc_bytes.data()), rpc_bytes.size());
                        }
                    }
                }

                remote_sync_tokens = 0;
                remote_sync_payload.clear();
                remote_sync_last_pos = -1;
                const auto selector_copy_start = std::chrono::steady_clock::now();
                selector_batch.n_tokens = n_rows;
                for (int32_t i = 0; i < n_rows; ++i) {
                    selector_batch.token[i] = batch.token[i];
                    selector_batch.pos[i] = i;
                    selector_batch.n_seq_id[i] = 1;
                    selector_batch.seq_id[i][0] = 0;
                    selector_batch.logits[i] = true;
                }
                std::memcpy(selector_batch.embd, rpc_bytes.data(), rpc_bytes.size());
                remote_selector_copy_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - selector_copy_start).count();
                const auto selector_start = std::chrono::steady_clock::now();
                const int rc = llama_encode(selector_ctx, selector_batch);
                if (rc != 0 || !llama_get_embeddings_nextn(selector_ctx)) {
                    throw std::runtime_error("DFlash2 host selector encode failed");
                }
                remote_selector_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - selector_start).count();
                remote_draft_done = std::chrono::steady_clock::now();
                remote_draft_pending = true;
                remote_trimmed = used_pipeline_output;
                remote_trim_us = 0;
                remote_draft_pos0 = used_pipeline_output ? remote_pipeline_pos0 : draft_request.pos0;
                remote_draft_n_max = n_rows - 1;
            } catch (const std::exception & err) {
                LOG_ERR("DFlash2 Xbox DRAFT failed: %s; remote drafting disabled for this run\n", err.what());
                remote_failed = true;
                return;
            }
        } else if (local_split) {
            const bool profile_local = std::getenv("GGML_DFLASH_LOCAL_PROF") != nullptr;
            const int64_t local_start_us = profile_local ? ggml_time_us() : 0;
            const int32_t n_rows = batch.n_tokens;
            const uint32_t expected_hidden = (uint32_t) ((size_t) n_rows * n_embd_dft);
            const float * hidden = nullptr;
            bool used_pipeline_output = false;

            if (local_pipeline_ready) {
                const bool match = batch.pos[0] == local_pipeline_pos0 &&
                        batch.token[0] == local_pipeline_id_last &&
                        n_rows == local_pipeline_n_max + 1 &&
                        local_pipeline_output.size() == expected_hidden;
                if (match) {
                    hidden = local_pipeline_output.data();
                    local_pipeline_ready = false;
                    local_pipeline_state_synced = false;
                    local_pipeline_used = true;
                    ++local_pipeline_hits;
                    ++local_request_hits;
                    used_pipeline_output = true;
                } else if (!discard_local_pipeline()) {
                    LOG_ERR("DFlash2 local pre-draft recovery failed; disabling local drafting\n");
                    local_failed = true;
                    return;
                }
            }

            if (!used_pipeline_output) {
                local_pipeline_used = false;
                const auto * model_tgt = llama_get_model(params.ctx_tgt);
                for (int32_t i = 0; i < n_rows; ++i) {
                    batch.pos[i] = batch.pos[0] + i;
                    batch.n_seq_id[i] = 1;
                    batch.seq_id[i][0] = 0;
                    batch.logits[i] = true;
                }
                if (!llama_model_get_token_embedding_row(model_tgt, batch.token[0],
                        batch.embd, (size_t) n_embd_dft)) {
                    LOG_WRN("DFlash2 local split anchor embedding lookup failed\n");
                    shadow_draft_failed = true;
                    return;
                }
                for (int32_t i = 1; i < n_rows; ++i) {
                    std::memcpy(batch.embd + (size_t) i * n_embd_dft,
                            mask_embedding.data(), (size_t) n_embd_dft * sizeof(float));
                }

                const int64_t stage_decode_start = shadow_observer ? ggml_time_us() : 0;
                const int ret = llama_decode(ctx_dft, batch);
                if (ret != 0) {
                    if (shadow_observer) shadow_stage_decode_us += (uint64_t) (ggml_time_us() - stage_decode_start);
                    shadow_draft_failed = true;
                    LOG_WRN("DFlash2 local split decode returned %d\n", ret);
                    return;
                }
                hidden = llama_get_embeddings(ctx_dft);
                if (shadow_observer) {
                    // The getter already synchronizes; include that wait and
                    // readback without introducing another GPU barrier.
                    shadow_stage_decode_us += (uint64_t) (ggml_time_us() - stage_decode_start);
                }
                if (!hidden) {
                    shadow_draft_failed = true;
                    LOG_WRN("DFlash2 local split hidden output is unavailable\n");
                    return;
                }
            }

            const int64_t local_hidden_us = profile_local ? ggml_time_us() : 0;
            selector_batch.n_tokens = n_rows;
            for (int32_t i = 0; i < n_rows; ++i) {
                selector_batch.token[i] = batch.token[i];
                selector_batch.pos[i] = i;
                selector_batch.n_seq_id[i] = 1;
                selector_batch.seq_id[i][0] = 0;
                selector_batch.logits[i] = true;
            }
            std::memcpy(selector_batch.embd, hidden,
                    (size_t) n_rows * n_embd_dft * sizeof(float));
            const int64_t stage_selector_start = shadow_observer ? ggml_time_us() : 0;
            const int selector_rc = llama_encode(selector_ctx, selector_batch);
            if (selector_rc != 0 || !llama_get_embeddings_nextn(selector_ctx)) {
                if (shadow_observer) shadow_stage_selector_us += (uint64_t) (ggml_time_us() - stage_selector_start);
                shadow_draft_failed = true;
                LOG_WRN("DFlash2 local selector encode failed\n");
                return;
            }
            if (shadow_observer) {
                shadow_stage_selector_us += (uint64_t) (ggml_time_us() - stage_selector_start);
            }
            if (profile_local) {
                LOG_INF("DFLOCAL rows=%d draft_us=%lld selector_us=%lld reused=%d\n", n_rows,
                        (long long) (local_hidden_us - local_start_us),
                        (long long) (ggml_time_us() - local_hidden_us), (int) used_pipeline_output);
            }
        } else {
            const int ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
                return;
            }
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_block_beg[seq_id] < 0) {
                continue;
            }
            auto & dp = dparams[seq_id];

            const int32_t beg            = i_block_beg[seq_id];
            const int32_t n_block_tokens = n_block[seq_id];

            auto * smpl = smpls[seq_id].get();

            auto & result = *dp.result;

            if (dp.dists) {
                dp.dists->clear();
            }

            if (is_dflash2) {
                GGML_ASSERT(dp.temperature <= 0.0f || dp.dists);
                const float * lattice = llama_get_embeddings_nextn(remote || local_split ? selector_ctx : ctx_dft);
                GGML_ASSERT(lattice && "DFlash2 selector produced no lattice");
                if (remote) {
                    remote_next_candidate = {};
                }
                if (local_split) {
                    local_next_candidate = {};
                }

                if (selector_reset[seq_id]) {
                    uint32_t seed = dp.seed;
                    if (seed == LLAMA_DEFAULT_SEED) {
                        seed = (uint32_t) std::chrono::high_resolution_clock::now().time_since_epoch().count();
                    }
                    selector_rng[seq_id].seed(seed ^ 0x85ebca6bU);
                    selector_reset[seq_id] = false;
                }

                int32_t predecessor = 0;
                double pipeline_prefix = 1.0;
                for (int32_t i = 1; i < n_block_tokens; ++i) {
                    const float * row = lattice + (size_t) (beg + i) * n_embd_dec;
                    const float * scores = row + selector_top_k + (size_t) predecessor * selector_top_k;

                    if (dp.temperature > 0.0f) {
                        common_speculative_token_dist dist;
                        dist.ids.resize(selector_top_k);
                        dist.probs.resize(selector_top_k);
                        const float max_score = *std::max_element(scores, scores + selector_top_k);
                        float sum = 0.0f;
                        for (int32_t k = 0; k < selector_top_k; ++k) {
                            dist.ids[k] = (llama_token) row[k];
                            dist.probs[k] = std::exp((scores[k] - max_score) / dp.temperature);
                            sum += dist.probs[k];
                        }
                        for (float & p : dist.probs) {
                            p /= sum;
                        }
                        predecessor = common_speculative_dflash_sample(dist.probs, params.p_min, selector_rng[seq_id]);
                        if (predecessor < 0) {
                            break;
                        }
                        result.push_back(dist.ids[predecessor]);
                        dp.dists->push_back(std::move(dist));
                    } else {
                        predecessor = (int32_t) std::distance(scores,
                                std::max_element(scores, scores + selector_top_k));
                        if (params.p_min > 0.0f) {
                            // softmax(scores) at the argmax, i.e. 1 / sum(exp(s_k - s_max))
                            float sum = 0.0f;
                            for (int32_t k = 0; k < selector_top_k; ++k) {
                                sum += std::exp(scores[k] - scores[predecessor]);
                            }
                            const float confidence = 1.0f / sum;
                            if (confidence < params.p_min) {
                                break;
                            }
                        }
                        result.push_back((llama_token) row[predecessor]);
                        if (remote && (remote_pipeline_enabled || rpc_metrics.is_open())) {
                            remote_next_candidate = dflash_pipeline_rank2_step(
                                    row, scores, selector_top_k, predecessor, i - 1,
                                    pipeline_prefix, remote_next_candidate);
                        }
                        if (local_split && local_prefetch) {
                            local_next_candidate = dflash_pipeline_rank2_step(
                                    row, scores, selector_top_k, predecessor, i - 1,
                                    pipeline_prefix, local_next_candidate);
                        }
                    }
                }

                if (result.size() < (size_t) params.n_min) {
                    result.clear();
                    if (dp.dists) {
                        dp.dists->clear();
                    }
                }
                if (remote) {
                    remote_n_proposed = (int32_t) result.size();
                    if (result.empty()) remote_next_candidate = {};
                    remote_proposal_done = std::chrono::steady_clock::now();
                    remote_proposal_us = std::chrono::duration_cast<std::chrono::microseconds>(
                            remote_proposal_done - remote_draft_done).count();
                }
                if (local_split) {
                    local_draft_pending = true;
                    local_draft_pos0 = batch.pos[0];
                    local_n_proposed = (int32_t) result.size();
                    if (result.empty()) local_next_candidate = {};
                }
                continue;
            }

            if (is_dspark) {
                // DSpark: read from the first draft slot, truncate below the confidence threshold
                const float * conf = params.p_min > 0.0f ? llama_get_embeddings_nextn(ctx_dft) : nullptr;
                // bonus-anchor drafts read the mask positions only, like DFlash
                const int32_t i_draft_beg = sample_from_anchor ? 0 : 1;
                for (int32_t i = i_draft_beg; i < n_block_tokens; ++i) {
                    const int32_t idx = beg + i;

                    if (conf && conf[(size_t) idx * n_embd_dec] < params.p_min) {
                        break;
                    }

                    common_sampler_sample(smpl, ctx_dft, idx, true);

                    const auto * cur_p = common_sampler_get_candidates(smpl, true);

                    for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                        LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                                seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                                common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                    }

                    const llama_token id = cur_p->data[0].id;

                    common_sampler_accept(smpl, id, true);

                    result.push_back(id);
                }
            } else {
                // greedily read the predicted block at this sequence's noise positions 1..n_block_tokens-1
                for (int32_t i = 1; i < n_block_tokens; ++i) {
                    common_sampler_sample(smpl, ctx_dft, beg + i, true);

                    const auto * cur_p = common_sampler_get_candidates(smpl, true);

                    for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                        LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                                seq_id, k, i - 1, cur_p->data[k].id, cur_p->data[k].p,
                                common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                    }

                    const llama_token id = cur_p->data[0].id;

                    if (cur_p->data[0].p < params.p_min) {
                        break;
                    }

                    common_sampler_accept(smpl, id, true);

                    result.push_back(id);
                }
            }

            if (result.size() < (size_t) params.n_min) {
                result.clear();
            }
        }
        if (remote && remote_pipeline_enabled && remote_draft_pending &&
                remote_next_candidate.accepted >= 0 &&
                remote_next_candidate.accepted < remote_n_proposed &&
                remote_next_candidate.token >= 0 &&
                remote_next_candidate.score >= 0.25f) {
            launch_remote_pipeline(remote_next_candidate.token,
                                   remote_next_candidate.accepted);
        }
        if (remote && remote_draft_pending && remote_n_proposed == 0) {
            // No target verification follows an empty draft, so accept() will not run.
            // Discard the speculative Xbox KV before the next target feature SYNC.
            accept(0, 0, false);
        }
        if (local_split && local_draft_pending && local_n_proposed == 0) {
            // No target verification follows an empty draft. Roll back the
            // explicit local KV immediately because the server will not call
            // common_speculative_accept() for this round.
            accept(0, 0, false);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t n_accepted, bool is_other) override {
        if (local_split) {
            if (is_other || local_failed) {
                return;
            }

            bool promoted = false;
            if (local_pipeline_active()) {
                promoted = n_accepted == local_pipeline_accept_k && local_n_proposed > 0 && promote_local_pipeline();
                if (!promoted && !discard_local_pipeline()) {
                    LOG_ERR("DFlash2 local pre-draft rollback failed; disabling local drafting\n");
                    local_failed = true;
                    return;
                }
            }

            if (!promoted && local_draft_pos0 >= 0) {
                // Target features through the last accepted draft are committed.
                // Keep those rows for the next round's DFlash KV cache.
                const llama_pos trim_pos = local_draft_pos0 + (llama_pos) n_accepted + 1;
                if (!trim_local_context(trim_pos)) {
                    LOG_ERR("DFlash2 local split draft rollback failed at position %d; disabling local drafting\n", (int) trim_pos);
                    local_failed = true;
                }
            }

            local_draft_pending = false;
            local_draft_pos0 = -1;
            local_n_proposed = 0;
            local_next_candidate = {};
            return;
        }

        if (!remote) {
            return;
        }

        const auto verify_done = std::chrono::steady_clock::now();
        if (remote_pipeline_active()) {
            // process() may have queued target rows while the pre-draft was in
            // flight. Only the predicted rejection point has the expected
            // bonus position; draft() still checks the actual bonus token ID.
            const bool promote = !is_other &&
                n_accepted == remote_pipeline_accept_k && remote_n_proposed > 0;
            if (promote) {
                if (!promote_remote_pipeline()) {
                    remote_draft_pending = false;
                    remote_trimmed = true;
                }
            } else if (!discard_remote_pipeline()) {
                remote_draft_pending = false;
                remote_trimmed = true;
            }
            remote_draft_pending = false;
            remote_trimmed = true;
        }
        const auto pipeline_accept_done = std::chrono::steady_clock::now();
        const auto pipeline_accept_overhead_us =
            std::chrono::duration_cast<std::chrono::microseconds>(
                pipeline_accept_done - verify_done).count();

        if (!remote_draft_pending && remote_failed) {
            return;
        }
        const auto target_verify_us = remote_n_proposed == 0 ? 0 :
                std::chrono::duration_cast<std::chrono::microseconds>(
                        verify_done - remote_draft_done).count();
        if (!remote_trimmed && !remote_failed) {
            try {
                if (profile_mode != remote_profile_mode::mock_local) {
                    const dflash_remote::trim_data request{remote_draft_pos0};
                    const auto timing = remote->call(dflash_remote::trim, ++rpc_request_id, 0,
                            &request, sizeof(request), rpc_bytes, 0);
                    remote_trim_us = timing.roundtrip_us;
                }
                remote_trimmed = true;
            } catch (const std::exception & err) {
                LOG_ERR("DFlash2 Xbox TRIM failed: %s; remote drafting disabled for this run\n", err.what());
                remote_failed = true;
            }
        }
        const auto cycle_done = std::chrono::steady_clock::now();
        if (rpc_metrics.is_open()) {
            const auto & xbox = remote_draft_timing.server;
            const auto tick_us = [](auto point) -> int64_t {
                return std::chrono::duration_cast<std::chrono::microseconds>(
                        point.time_since_epoch()).count();
            };
            const auto cycle_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    cycle_done - remote_draft_start).count();
            rpc_metrics << "{\"cycle_id\":" << remote_draft_timing.server.cycle_id
                        << ",\"mode\":\"" << (profile_mode == remote_profile_mode::mock_xbox ? "mock_xbox" :
                                                 profile_mode == remote_profile_mode::mock_local ? "mock_local" : "real") << "\""
                        << ",\"draft_start_tick_us\":" << tick_us(remote_draft_start)
                        << ",\"draft_done_tick_us\":" << tick_us(remote_draft_done)
                        << ",\"proposal_done_tick_us\":" << tick_us(remote_proposal_done)
                        << ",\"accept_entry_tick_us\":" << tick_us(verify_done)
                        << ",\"accept_done_tick_us\":" << tick_us(cycle_done)
                        << ",\"target_position\":" << remote_draft_pos0
                        << ",\"anchor_id_real\":" << remote_current_anchor_id
                        << ",\"n_proposed\":" << remote_n_proposed
                        << ",\"n_accepted\":" << n_accepted
                        << ",\"pipeline_candidate_k\":" << remote_next_candidate.accepted
                        << ",\"pipeline_candidate_id\":" << remote_next_candidate.token
                        << ",\"pipeline_candidate_score\":" << remote_next_candidate.score
                        << ",\"is_other\":" << (is_other ? "true" : "false")
                        << ",\"capture_hidden_us\":" << rpc_capture_us
                        << ",\"sync_serialize_us\":" << rpc_sync_serialize_us
                        << ",\"serialize_us\":" << remote_serialize_us
                        << ",\"prepare_request_us\":" << remote_prepare_us
                        << ",\"selector_copy_us\":" << remote_selector_copy_us
                        << ",\"proposal_us\":" << remote_proposal_us
                        << ",\"rewind_rpc_us\":" << rpc_rewind_us
                        << ",\"fused_sync\":" << (remote_fused_sync ? "true" : "false")
                        << ",\"fused_draft\":" << (remote_fused_draft ? "true" : "false")
                        << ",\"send_us\":" << remote_draft_timing.send_us
                        << ",\"header_send_us\":" << remote_draft_timing.header_send_us
                        << ",\"payload_send_us\":" << remote_draft_timing.payload_send_us
                        << ",\"header_wait_us\":" << remote_draft_timing.header_wait_us
                        << ",\"payload_read_us\":" << remote_draft_timing.payload_read_us
                        << ",\"sync_rpc_us\":" << rpc_sync_us
                        << ",\"sync_send_us\":" << rpc_sync_send_us
                        << ",\"sync_wait_us\":" << rpc_sync_wait_us
                        << ",\"sync_xbox_prepare_us\":" << rpc_sync_xbox_prepare_us
                        << ",\"sync_xbox_compute_us\":" << rpc_sync_xbox_compute_us
                        << ",\"draft_rpc_us\":" << remote_draft_timing.roundtrip_us
                        << ",\"xbox_receive_us\":" << xbox.xbox_receive_us
                        << ",\"xbox_prepare_us\":" << xbox.xbox_prepare_us
                        << ",\"xbox_compute_us\":" << xbox.xbox_compute_us
                        << ",\"xbox_response_us_prior\":" << xbox.xbox_response_us
                        << ",\"selector_us\":" << remote_selector_us
                        << ",\"target_verify_us\":" << target_verify_us
                        << ",\"pipeline_accept_overhead_us\":" << pipeline_accept_overhead_us
                        << ",\"trim_rpc_us\":" << remote_trim_us
                        << ",\"total_cycle_us\":" << cycle_us
                        << ",\"pipeline_enabled\":" << (remote_pipeline_enabled ? "true" : "false")
                        << ",\"pipeline_attempts\":" << remote_pipeline_attempts
                        << ",\"pipeline_hits\":" << remote_pipeline_hits
                        << ",\"pipeline_used\":" << (remote_pipeline_used ? "true" : "false")
                        << ",\"pipeline_discarded\":" << (remote_pipeline_discarded ? "true" : "false")
                        << ",\"pipeline_match_mask\":" << remote_pipeline_match_mask
                        << ",\"pipeline_hypothesis_pos\":" << remote_pipeline_pos0
                        << ",\"pipeline_hypothesis_id\":" << remote_pipeline_id_last
                        << ",\"pipeline_actual_pos\":" << remote_pipeline_actual_pos
                        << ",\"pipeline_actual_id\":" << remote_pipeline_actual_id
                        << ",\"pipeline_wait_us\":" << remote_pipeline_wait_us
                        << ",\"pipeline_xbox_compute_us\":" << remote_pipeline_timing.server.xbox_compute_us
                        << ",\"rpc_failed\":" << (remote_failed ? "true" : "false") << "}\n";
            rpc_metrics.flush();
        }
        rpc_capture_us = 0;
        rpc_sync_us = 0;
        rpc_sync_serialize_us = 0;
        rpc_sync_send_us = 0;
        rpc_sync_wait_us = 0;
        rpc_sync_xbox_prepare_us = 0;
        rpc_sync_xbox_compute_us = 0;
        rpc_rewind_us = 0;
        remote_draft_pending = false;
        remote_pipeline_used = false;
        remote_pipeline_discarded = false;
        remote_pipeline_match_mask = 0;
        remote_pipeline_actual_pos = -1;
        remote_pipeline_actual_id = -1;
        remote_pipeline_wait_us = 0;
    }

    bool adaptive_dm_supported() const override {
        return common_speculative_dflash_adaptive_dm_supported(selector_top_k);
    }

    bool draft_memory_is_shared() const override {
        // The explicit split context deliberately has no ctx_other and its KV
        // lifetime is managed by this implementation, like the remote DFlash
        // sidecar.  The server must not trim it through the target memory path.
        return local_split;
    }
};

struct common_speculative_impl_draft_mtp : public common_speculative_impl {
    common_params_speculative_draft params; // reuses the draft-model params slot (ctx_tgt/ctx_dft)

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd = 0;

    // One MTP draft driver, three modes (set once in the ctor):
    //   is_mem_shared (gemma4): shares the target KV, runs all heads in one graph.
    //   chain_heads (step35): n_mtp_layers trained heads, one per draft step.
    //   neither (qwen35 / qwen35moe): a single trained MTP head.
    int32_t n_mtp_layers  = 1;
    bool    is_mem_shared = false;   // gemma4
    bool    chain_heads   = false;   // derived in the ctor: n_mtp_layers > 1 && !is_mem_shared

    // Per-sequence cross-batch carryover: pair (h_p, x_{p+1}) at MTP pos p+1.
    // The last h-row of one process() call needs the first token of the NEXT
    // call to pair with, so it's stashed here until that next call fires.
    std::vector<std::vector<float>> pending_h;   // [n_seq][n_embd]
    std::vector<llama_pos> pending_pos;
    std::vector<llama_pos> history_start; // first draft KV position; 0 for a full prefill

    std::vector<int32_t> i_batch_beg;
    std::vector<int32_t> i_batch_end;

    // Hidden rows from the most recent target verification batch, grouped by seq.
    // Row 0 corresponds to the sampled token, row N to the Nth accepted draft token.
    std::vector<std::vector<float>> verify_h;
    std::vector<int32_t> verify_h_rows;
    std::vector<llama_pos> verify_pos_beg;

    std::vector<int>                i_last;
    std::vector<std::vector<float>> chain_h;

    common_speculative_impl_draft_mtp(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, n_seq, params.draft.n_max)
        , params(params.draft)
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "MTP requires ctx_tgt and ctx_dft to be set");

        n_embd = llama_model_n_embd_out(llama_get_model(ctx_dft));
        GGML_ASSERT(n_embd == llama_model_n_embd_out(llama_get_model(ctx_tgt)) &&
                "MTP input row width must match the target h_nextn width");
        n_mtp_layers = std::max(1, (int) llama_model_n_layer_nextn(llama_get_model(ctx_dft)));

        SPC_TRC("%s", "adding speculative implementation 'draft-mtp'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%.2f, n_embd=%d, backend_sampling=%d\n", this->params.n_max, this->params.n_min, this->params.p_min, n_embd, (int) this->params.backend_sampling);
        SPC_TRC("- gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n",
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        const int32_t n_b = (int32_t) llama_n_batch(ctx_dft);
        batch = llama_batch_init(/*n_tokens=*/ n_b, /*embd=*/ n_embd, /*n_seq_max=*/ 1);
        // llama_batch_init allocates only one of token/embd; MTP needs both.
        // TODO: fix, how to call without malloc
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_b);
        if (batch.token != nullptr) {
            std::memset(batch.token, 0, sizeof(llama_token) * (size_t) n_b);
        }
        if (batch.embd != nullptr) {
            std::memset(batch.embd, 0, sizeof(float) * (size_t) n_b * (size_t) n_embd);
        }

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);

        is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt;
        chain_heads   = n_mtp_layers > 1 && !is_mem_shared;

        if (chain_heads) {
            this->params.n_max = std::min(this->params.n_max, n_mtp_layers);

            chain_h.assign(n_seq, {});
            for (auto & c : chain_h) {
                c.reserve((size_t) (this->params.n_max + 1) * n_embd);
            }
        }
        this->n_max = this->params.n_max;

        pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));
        pending_pos.assign(n_seq, -1);
        history_start.assign(n_seq, 0);

        i_last.assign(n_seq, -1);
        i_batch_beg.assign(n_seq, -1);
        i_batch_end.assign(n_seq, -1);

        verify_h.assign(n_seq, {});
        verify_h_rows.assign(n_seq, 0);
        verify_pos_beg.assign(n_seq, -1);
    }

    ~common_speculative_impl_draft_mtp() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        if (batch.token != nullptr) {
            free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
    }

    // Reset carry before processing a sequence from position zero. Nonzero
    // continuations retain the carry restored with their prompt checkpoint.
    void reset_seq_state(llama_seq_id seq_id) {
        if (seq_id < 0 || (size_t) seq_id >= pending_h.size()) {
            return;
        }
        std::fill(pending_h[seq_id].begin(), pending_h[seq_id].end(), 0.0f);
        if ((size_t) seq_id < verify_h.size()) {
            verify_h[seq_id].clear();
        }
        if ((size_t) seq_id < verify_h_rows.size()) {
            verify_h_rows[seq_id] = 0;
        }
        if ((size_t) seq_id < i_last.size()) {
            i_last[seq_id] = -1;
        }
        if ((size_t) seq_id < chain_h.size()) {
            chain_h[seq_id].clear();
        }
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }

        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);

        if (pos_max < N - 1 && !is_mem_shared && history_start[seq_id] == 0) {
            SPC_WRN("ctx_dft pos_max=%d < N-1=%d - "
                    "process() hook may not have run on every prefill ubatch "
                    "(need_embd / logits=1 on every prompt position?). "
                    "Drafts may degrade.\n",
                    (int) pos_max, N - 1);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        const bool managed = llama_model_mtp_weights_get_info(llama_get_model(params.ctx_tgt)).managed;
        // TODO: how to make it work with vision tokens?
        if (batch_in.token == nullptr || batch_in.embd != nullptr) {
            return !managed;
        }

        const int32_t n_tokens = batch_in.n_tokens;
        if (managed) {
            if (n_seq != 1 || !batch_in.pos || !batch_in.n_seq_id || !batch_in.seq_id ||
                    n_tokens > (int32_t) llama_n_batch(params.ctx_tgt) ||
                    !llama_get_embeddings_nextn(params.ctx_tgt) ||
                    !llama_matches_nextn_decode(params.ctx_tgt, llama_get_nextn_decode_id(params.ctx_tgt), batch_in)) {
                SPC_ERR("%s", "resident MTP requires a bounded, explicit single-sequence token batch\n");
                return false;
            }
            for (int32_t i = 0; i < n_tokens; ++i) {
                if (batch_in.n_seq_id[i] != 1 || !batch_in.seq_id[i] || batch_in.seq_id[i][0] != 0 ||
                        batch_in.pos[i] < 0 || (i > 0 && (int64_t) batch_in.pos[i] != (int64_t) batch_in.pos[i-1] + 1)) {
                    SPC_ERR("%s", "resident MTP received an invalid sequence/position batch\n");
                    return false;
                }
            }
        }

        // remember the first and last batch index for each sequence
        std::fill(i_batch_beg.begin(), i_batch_beg.end(), -1);
        std::fill(i_batch_end.begin(), i_batch_end.end(), -1);

        for (int k = 0; k < n_tokens; ++k) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                GGML_ASSERT(batch_in.n_seq_id[k] == 1);

                if (batch_in.seq_id[k][0] == seq_id) {
                    i_batch_end[seq_id] = k;
                    if (i_batch_beg[seq_id] < 0) {
                        i_batch_beg[seq_id] = k;
                    }
                }
            }
        }

        if (managed) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                const int first = i_batch_beg[seq_id];
                if (first >= 0 && batch_in.pos[first] != 0 &&
                        (int64_t) pending_pos[seq_id] + 1 != batch_in.pos[first]) {
                    SPC_ERR("resident MTP carry position %d does not precede batch position %d\n",
                            pending_pos[seq_id], batch_in.pos[first]);
                    return false;
                }
            }
        }

        // begin() is called after prefill, so resetting there would discard
        // the final prompt hidden state needed by the first draft.
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            const int32_t first = i_batch_beg[seq_id];
            if (first >= 0 && batch_in.pos[first] == 0) {
                reset_seq_state(seq_id);
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        // if kv is shared with target (e.g Gemma4), then we can skip this catch-up decode
        if (!is_mem_shared) {
            common_batch_clear(batch);

            for (int k = 0; k < n_tokens; ++k) {
                common_batch_add(batch, batch_in.token[k], batch_in.pos[k], { batch_in.seq_id[k][0] }, 0);
            }

            // shift the tgt embeddings to the right by one position
            // assumes that the tokens in the batch are sequential for each sequence
            // i.e. we cannot have seq_id like this: [0, 0, 0, 1, 1, 0, 1, 1]
            //                                                       ^--- this is a problem
            // TODO:this is generally true, but would be nice to assert it
            {
                const float * h_tgt = llama_get_embeddings_nextn(ctx_tgt);
                std::memcpy(batch.embd + (size_t) 1 * n_embd, h_tgt, row_bytes * (n_tokens-1));
            }

            // fill the pending embeddings from a previous run
            auto set_h = [&](int idx, const float * h_row) {
                std::memcpy(batch.embd + (size_t) idx * n_embd, h_row, row_bytes);
            };

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (i_batch_beg[seq_id] < 0) {
                    continue;
                }

                if (batch_in.pos[i_batch_beg[seq_id]] == 0) {
                    std::fill(pending_h[seq_id].begin(), pending_h[seq_id].end(), 0.0f);
                    pending_pos[seq_id] = -1;
                    history_start[seq_id] = 0;
                }
                set_h(i_batch_beg[seq_id], pending_h[seq_id].data());
            }

            auto * mem_dft = llama_get_memory(ctx_dft);

            bool ok = true;
            for (int head = 0; head < n_mtp_layers; ++head) {
                if (chain_heads) {
                    // ref: https://github.com/ggml-org/llama.cpp/pull/24340/changes#r3413498544
                    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                        if (i_batch_beg[seq_id] < 0) {
                            continue;
                        }
                        llama_memory_seq_rm(mem_dft, seq_id, batch_in.pos[i_batch_beg[seq_id]], -1);
                    }
                    llama_set_nextn_layer_offset(ctx_dft, head);
                }

                const bool    prof_mtp = getenv("GGML_MTP_PROF") != nullptr;
                const int32_t prof_nb  = batch.n_tokens;
                const int64_t prof_t0  = prof_mtp ? ggml_time_us() : 0;

                const int32_t rc = llama_decode(ctx_dft, batch);

                if (prof_mtp) {
                    // NOTE: adds one sync per process() call; validated by A/B throughput
                    llama_synchronize(ctx_dft);
                    fprintf(stderr, "MTPPROC head=%d n_batch=%d us=%lld\n",
                            head, prof_nb, (long long) (ggml_time_us() - prof_t0));
                }

                if (rc != 0) {
                    SPC_ERR("llama_decode(ctx_dft) head=%d failed rc=%d (pos=%d)\n",
                            head, (int) rc, (int) batch_in.pos[0]);
                    ok = false;
                    break;
                }
            }

            if (chain_heads) {
                llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
            }
            if (!ok) {
                return false;
            }
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_end[seq_id] < 0) {
                continue;
            }

            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;
            verify_h_rows[seq_id] = n_rows;
            verify_pos_beg[seq_id] = batch_in.pos[i_batch_beg[seq_id]];
            verify_h[seq_id].resize((size_t) n_rows * n_embd);

            for (int32_t i = 0; i < n_rows; ++i) {
                const float * h = llama_get_embeddings_nextn_ith(ctx_tgt, i_batch_beg[seq_id] + i);
                std::memcpy(verify_h[seq_id].data() + (size_t) i * n_embd, h, row_bytes);
            }

            std::memcpy(pending_h[seq_id].data(),
                    verify_h[seq_id].data() + (size_t) (n_rows - 1) * n_embd, row_bytes);
            pending_pos[seq_id] = batch_in.pos[i_batch_end[seq_id]];
        }

        return true;
    }

    bool bootstrap(const llama_batch & input, uint64_t decode_id) override {
        if (is_mem_shared || chain_heads || n_seq != 1 || input.n_tokens <= 0 ||
                llama_get_model(params.ctx_dft) != llama_get_model(params.ctx_tgt) ||
                !llama_model_mtp_weights_get_info(llama_get_model(params.ctx_tgt)).managed ||
                input.n_tokens > (int32_t) llama_n_ubatch(params.ctx_tgt) ||
                !input.token || input.embd || !input.pos || !input.n_seq_id || !input.seq_id) {
            return false;
        }
        for (int32_t i = 0; i < input.n_tokens; ++i) {
            if (input.n_seq_id[i] != 1 || !input.seq_id[i] || input.seq_id[i][0] != 0 || input.pos[i] < 0 ||
                    (i > 0 && (int64_t) input.pos[i] != (int64_t) input.pos[i - 1] + 1)) {
                return false;
            }
        }
        const auto pos = input.pos[input.n_tokens - 1];
        if (pos == std::numeric_limits<llama_pos>::max() ||
                llama_memory_seq_pos_max(llama_get_memory(params.ctx_tgt), 0) != pos ||
                !llama_matches_nextn_decode(params.ctx_tgt, decode_id, input)) {
            return false;
        }
        llama_synchronize(params.ctx_tgt);
        llama_synchronize(params.ctx_dft);
        if (!llama_get_embeddings_nextn(params.ctx_tgt)) {
            return false;
        }
        const float * row = llama_get_embeddings_nextn_ith(params.ctx_tgt, input.n_tokens - 1);
        if (!row || !std::all_of(row, row + n_embd, [](float x) { return std::isfinite(x); })) {
            return false;
        }
        // This supported draft is plain single-sequence attention. Removing the whole
        // sequence cannot partially fail; all input/hidden validation precedes removal.
        if (!llama_memory_seq_rm(llama_get_memory(params.ctx_dft), 0, -1, -1)) {
            return false;
        }
        reset_carry(0);
        std::memcpy(pending_h[0].data(), row, (size_t) n_embd * sizeof(float));
        pending_pos[0] = pos;
        history_start[0] = pos + 1;
        return true;
    }

    bool is_ready(llama_seq_id seq_id, llama_pos next_pos) const override {
        if (!llama_model_mtp_weights_get_info(llama_get_model(params.ctx_tgt)).managed) {
            return true;
        }
        if (is_mem_shared || chain_heads || n_seq != 1 || seq_id < 0 || seq_id >= (llama_seq_id) n_seq || next_pos <= 0 ||
                (int64_t) pending_pos[seq_id] + 1 != next_pos ||
                llama_memory_seq_pos_max(llama_get_memory(params.ctx_tgt), seq_id) != pending_pos[seq_id]) {
            return false;
        }
        auto * mem = llama_get_memory(params.ctx_dft);
        const auto lo = llama_memory_seq_pos_min(mem, seq_id);
        const auto hi = llama_memory_seq_pos_max(mem, seq_id);
        return (history_start[seq_id] == next_pos && lo == -1 && hi == -1) ||
                (lo == history_start[seq_id] && hi == pending_pos[seq_id]);
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting || dp.n_max == 0) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            common_batch_add(batch, dp.id_last, dp.pos0, { seq_id }, true);
            std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, pending_h[seq_id].data(), row_bytes);

            i_last[seq_id] = batch.n_tokens - 1;

            if (chain_heads) {
                chain_h[seq_id].assign(pending_h[seq_id].begin(), pending_h[seq_id].end());
            }
        }

        int i = 0;

        while (n_drafting > 0) {
            // each step decodes under a different head, i.e. a different decoder layer, and
            // KV is per layer. process() filled this layer's KV only for positions < pos0
            // (prompt + accepted prefix) — nothing in the draft region yet. so reset the
            // draft region (the seq_rm lower bound is pos0, leaving the prompt KV intact)
            // and select head i so it rebuilds its own layer's KV there; decoding just the
            // latest token would leave its attention reading cells only another head wrote.
            if (chain_heads) {
                auto * mem_dft = llama_get_memory(ctx_dft);
                for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                    if (drafting[seq_id]) {
                        llama_memory_seq_rm(mem_dft, seq_id, dparams[seq_id].pos0, -1);
                    }
                }
                llama_set_nextn_layer_offset(ctx_dft, i);
            }

            const bool    prof_mtp  = getenv("GGML_MTP_PROF") != nullptr;
            const int32_t prof_nb   = batch.n_tokens;
            const int64_t prof_dec0 = prof_mtp ? ggml_time_us() : 0;

            int ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\n", i, ret);
                break;
            }

            // rebuild the batch for the next step: the growing-KV paths re-add only the
            // new token (the KV already holds the prefix), while chained heads re-add the
            // whole prefix at the next head. dropped sequences are simply not re-added.
            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_last[seq_id], true);
                const float * h_row = llama_get_embeddings_nextn_ith(ctx_dft, i_last[seq_id]);

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                const int32_t limit = dp.n_max >= 0 ? std::min(params.n_max, dp.n_max) : params.n_max;
                if (limit <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                if (chain_heads) {
                    // ref: https://github.com/ggml-org/llama.cpp/pull/24340#discussion_r3448031546
                    chain_h[seq_id].insert(chain_h[seq_id].end(), h_row, h_row + n_embd);

                    const int n_rows = (int) result.size() + 1; // id_last + tokens drafted so far
                    for (int t = 0; t < n_rows; ++t) {
                        const llama_token tok = (t == 0) ? dp.id_last : result[t - 1];
                        common_batch_add(batch, tok, dp.pos0 + t, { seq_id }, t == n_rows - 1);
                        std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd,
                                    chain_h[seq_id].data() + (size_t) t * n_embd, row_bytes);
                    }
                } else if (is_mem_shared) {
                    // note: with shared memory (e.g. Gemma4 assistants) we use the same position for all draft tokens
                    // ref: https://github.com/huggingface/transformers/blob/effde20942e3f82a1b97449f60b3a48c5ff96145/docs/source/en/model_doc/gemma4_assistant.md?plain=1#L36-L37
                    common_batch_add(batch, id, dp.pos0, { seq_id }, true);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, h_row, row_bytes);
                } else {
                    common_batch_add(batch, id, dp.pos0 + i + 1, { seq_id }, true);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, h_row, row_bytes);
                }

                i_last[seq_id] = batch.n_tokens - 1;
            }

            if (prof_mtp) {
                const int64_t t_sync = common_spec_prof_sync_us;
                const int64_t t_end  = ggml_time_us();
                if (t_sync >= prof_dec0) {
                    fprintf(stderr, "MTPDRAFT depth=%d n_batch=%d dec_us=%lld smp_us=%lld\n",
                            i, prof_nb, (long long) (t_sync - prof_dec0), (long long) (t_end - t_sync));
                } else {
                    fprintf(stderr, "MTPDRAFT depth=%d n_batch=%d dec_us=NA smp_us=%lld\n",
                            i, prof_nb, (long long) (t_end - prof_dec0));
                }
            }

            if (batch.n_tokens == 0) {
                break;
            }

            ++i;
        }

        if (chain_heads) {
            llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_h_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_h = std::min<int32_t>(n_accepted, n_rows - 1);
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        std::memcpy(pending_h[seq_id].data(), verify_h[seq_id].data() + (size_t) i_h * n_embd, row_bytes);
        pending_pos[seq_id] = verify_pos_beg[seq_id] + i_h;
    }

    bool get_state(llama_seq_id seq_id, std::vector<uint8_t> & data) const override {
    if (is_mem_shared || chain_heads || seq_id < 0 || seq_id >= (llama_seq_id) n_seq ||
                pending_pos[seq_id] < 0) {
            return false;
        }
        // MTP2: native uint32 magic/version, width, accepted target position and history start, then row.
        const std::array<uint32_t, 4> header{0x3250544d, (uint32_t) n_embd,
                (uint32_t) pending_pos[seq_id], (uint32_t) history_start[seq_id]};
        const auto & row = pending_h[seq_id];
        data.resize(sizeof(header) + row.size() * sizeof(float));
        std::memcpy(data.data(), header.data(), sizeof(header));
        std::memcpy(data.data() + sizeof(header), row.data(), row.size() * sizeof(float));
        return true;
    }

    bool validate_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) const override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return false;
        }
        if (data.empty()) {
            return true;
        }

        constexpr uint32_t expected_magic = 0x3250544d; // MTP2
        constexpr size_t header_size = sizeof(uint32_t) * 4;
        if (data.size() != header_size + (size_t) n_embd * sizeof(float)) {
            return false;
        }

        std::array<uint32_t, 4> header;
        std::memcpy(header.data(), data.data(), sizeof(header));
        return header[0] == expected_magic && header[1] == (uint32_t) n_embd &&
                header[2] <= (uint32_t) std::numeric_limits<llama_pos>::max() &&
                header[3] <= (uint32_t) std::numeric_limits<llama_pos>::max() &&
                header[3] <= header[2] + 1;
    }

    bool set_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) override {
        return set_state(seq_id, data, -1);
    }

    bool draft_memory_is_shared() const override {
        return is_mem_shared;
    }

    bool need_embd_nextn() const override {
        return true;
    }

    bool set_state(llama_seq_id seq_id, const std::vector<uint8_t> & data, llama_pos expected_pos) override {
        if (is_mem_shared || chain_heads || seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return false;
        }
        if (data.empty()) {
            if (expected_pos >= 0) {
                return false;
            }
            reset_carry(seq_id);
            return true;
        }
        std::array<uint32_t, 4> header;
        auto & row = pending_h[seq_id];
        if (data.size() != sizeof(header) + row.size() * sizeof(float)) {
            return false;
        }
        std::memcpy(header.data(), data.data(), sizeof(header));
        if (header[0] != 0x3250544d || header[1] != (uint32_t) n_embd ||
                header[2] > (uint32_t) std::numeric_limits<llama_pos>::max() ||
                header[3] > (uint32_t) std::numeric_limits<llama_pos>::max() || header[3] > header[2] + 1 ||
                (expected_pos >= 0 && header[2] != (uint32_t) expected_pos)) {
            return false;
        }
        // Validate before changing the live carry; no allocation or partial mutation on failure.
        for (size_t i = 0; i < row.size(); ++i) {
            float value;
            std::memcpy(&value, data.data() + sizeof(header) + i * sizeof(float), sizeof(value));
            if (!std::isfinite(value)) {
                return false;
            }
        }
        reset_carry(seq_id);
        std::memcpy(row.data(), data.data() + sizeof(header), row.size() * sizeof(float));
        pending_pos[seq_id] = (llama_pos) header[2];
        history_start[seq_id] = (llama_pos) header[3];
        return true;
    }

    void reset_carry(llama_seq_id seq_id) {
        std::fill(pending_h[seq_id].begin(), pending_h[seq_id].end(), 0.0f);
        pending_pos[seq_id] = -1;
        history_start[seq_id] = 0;
        verify_h[seq_id].clear();
        verify_h_rows[seq_id] = 0;
        verify_pos_beg[seq_id] = -1;
        i_batch_beg[seq_id] = i_batch_end[seq_id] = i_last[seq_id] = -1;
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_impl_ngram_simple : public common_speculative_impl {
    common_params_speculative_ngram_map params;

    // shared across all sequences
    common_ngram_simple_config config;

    common_speculative_impl_ngram_simple(
            const common_params_speculative & params, uint32_t n_seq,
            common_ngram_simple_config config)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, n_seq, params.ngram_simple.size_m)
        , params(params.ngram_simple)
        , config(config)
    {
        SPC_TRC("%s", "adding speculative implementation 'ngram-simple'\n");
        SPC_TRC("- size_n=%d, size_m=%d, min_hits=%d\n",
                this->params.size_n, this->params.size_m, this->params.min_hits);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            *dp.result = common_ngram_simple_draft(config, *dp.prompt, dp.id_last);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};

struct common_speculative_impl_ngram_map_k : public common_speculative_impl {
    // n_seq configs
    std::vector<common_ngram_map> config;

    common_speculative_impl_ngram_map_k(
            const common_ngram_map & config,
            uint32_t n_seq)
        : common_speculative_impl(config.key_only ? COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K
            : COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, n_seq, config.size_value)
    {
        for (uint32_t i = 0; i < n_seq; i++) {
            this->config.push_back(config);
        }

        SPC_TRC("adding speculative implementation '%s'\n", common_speculative_type_to_str(this->type).c_str());
        SPC_TRC("- size_key=%d, size_value=%d, key_only=%d, min_hits=%d\n",
                config.size_key, config.size_value, config.key_only, config.min_hits);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        GGML_ASSERT(seq_id < (llama_seq_id) n_seq);

        common_ngram_map_begin(config[seq_id], prompt);
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            common_ngram_map_draft(config[seq_id], *dp.prompt, dp.id_last, *dp.result);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        GGML_ASSERT((seq_id < (llama_seq_id) config.size()));

        if (is_other) {
            return;
        }

        common_ngram_map_accept(config[seq_id], n_accepted);
    }
};

struct common_speculative_impl_ngram_mod : public common_speculative_impl {
    common_params_speculative_ngram_mod params;

    // shared across all sequences
    common_ngram_mod mod;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    struct seq_info {
        // the last position in the prompt that was added to the ngram container
        size_t i_last = 0;

        // length of the last drafted n-gram (number of tokens returned by draft)
        size_t n_draft_last = 0;

        // consecutive accept rounds with low acceptance fraction (< 0.5)
        int n_low = 0;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_mod(
            const common_params_speculative & params,
            uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, n_seq, params.ngram_mod.n_max)
        , params(params.ngram_mod)
        , mod(params.ngram_mod.n_match, 4*1024*1024)
        , verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));

        SPC_TRC("%s", "adding speculative implementation 'ngram-mod'\n");
        SPC_TRC("- n_match=%d, n_max=%d, n_min=%d\n",
                this->params.n_match, this->params.n_max, this->params.n_min);
        SPC_TRC("- mod size=%zu (%.3f MB)\n",
                mod.size(), (float)(mod.size_bytes())/1024/1024);

        if (this->params.n_match < 16) {
            SPC_WRN("ngram_mod n_match=%d is too small - poor quality is possible, "
                    "see: https://github.com/ggml-org/llama.cpp/pull/19164\n", this->params.n_match);
        }

        sinfos.resize(n_seq);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        auto & sinfo = sinfos[seq_id];

        sinfo.i_last = 0;
        sinfo.n_draft_last = 0;

        const size_t n = mod.get_n();
        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        sinfo.i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        SPC_TRC("ngram_mod occupancy = %zu/%zu (%.2f)\n", mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            SPC_WRN("ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", f, f_thold);

            mod.reset();
        }
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        sinfo.n_draft_last = 0;

        const size_t cur_len = prompt.size();
        if (cur_len < mod.get_n()) {
            return;
        }

        const size_t n = mod.get_n();

        // add new ngrams in chunks
        if (sinfo.i_last + 32 < cur_len) {
            for (size_t i = sinfo.i_last; i < cur_len - n; ++i) {
                mod.add(prompt.data() + i);
            }

            sinfo.i_last = cur_len - n;
        }

        result.resize(n + params.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt.at(cur_len - n + 1 + i);
        }
        result[n - 1] = dparams.id_last;

        for (int i = 0; i < params.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < params.n_min) {
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted
        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        // store length of drafted n-gram for later acceptance analysis
        sinfo.n_draft_last = result.size();
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        if (is_other) {
            return;
        }

        auto & sinfo = sinfos[seq_id];

        // compute acceptance fraction if we have a recorded draft length
        if (sinfo.n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)sinfo.n_draft_last;
            if (f_acc < 0.25) {
                sinfo.n_low++;
                if (sinfo.n_low >= 5) {
                    if (verbose) {
                        SPC_TRC("low acceptance streak (%d) - resetting ngram_mod\n", sinfo.n_low);
                    }

                    mod.reset();
                    sinfo.n_low = 0;
                    sinfo.i_last = 0;
                }
            } else {
                sinfo.n_low = 0;
            }
        }
    }
};

struct common_speculative_impl_ngram_cache : public common_speculative_impl {
    common_params_speculative_ngram_cache params;

    uint16_t n_draft;

    bool save_dynamic;
    bool save_static;

    struct seq_info {
        size_t cache_size = 0; // number of tokens in n-gram cache

        common_ngram_cache ngram_cache_context;
        common_ngram_cache ngram_cache_dynamic;
        common_ngram_cache ngram_cache_static;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_cache(
            const common_params_speculative & params,
            uint32_t n_seq,
            uint16_t n_draft,
            const std::string & path_static,
            const std::string & path_dynamic,
            bool save_dynamic,
            bool save_static)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, n_seq, n_draft)
        , params(params.ngram_cache)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
    {
        SPC_TRC("%s", "adding speculative implementation 'ngram-cache'\n");
        SPC_TRC("- n_draft=%d, cache_static=%s, cache_dynamic=%s\n",
                n_draft,
                path_static.empty() ? "none" : path_static.c_str(),
                path_dynamic.empty() ? "none" : path_dynamic.c_str());

        sinfos.resize(n_seq);

        if (!path_static.empty()) {
            try {
                auto ngram_cache_static = common_ngram_cache_load(path_static);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_static = ngram_cache_static;
                }
            } catch (...) {
                SPC_ERR("failed to open static lookup cache: %s", path_static.c_str());
                GGML_ABORT("Couldn't read static lookup cache");
            }
        }

        if (!path_dynamic.empty()) {
            try {
                auto ngram_cache_dynamic = common_ngram_cache_load(path_dynamic);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_dynamic = ngram_cache_dynamic;
                }
            } catch (...) {
                SPC_ERR("failed to open dynamic lookup cache: %s", path_dynamic.c_str());
                GGML_ABORT("Couldn't read dynamic lookup cache");
            }
        }
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        if (sinfo.cache_size < prompt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt.size() + 1 - sinfo.cache_size);
            for (size_t j = sinfo.cache_size; j < prompt.size(); ++j) {
                tokens_new.push_back(prompt[j]);
            }
            tokens_new.push_back(dparams.id_last); // add the last token

            // Update context ngram cache with new dparams.prompt:
            common_ngram_cache_update(
                    sinfo.ngram_cache_context,
                    LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    tokens_new, tokens_new.size(), false);
            sinfo.cache_size = prompt.size() + 1;
        }

        llama_tokens inp;
        inp.reserve(prompt.size() + 1);
        for (size_t j = 0; j < prompt.size(); ++j) {
            inp.push_back(prompt[j]);
        }
        inp.push_back(dparams.id_last);

        result.push_back(dparams.id_last);

        common_ngram_cache_draft(
                inp, result, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                sinfo.ngram_cache_context,
                sinfo.ngram_cache_dynamic,
                sinfo.ngram_cache_static);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};

struct common_speculative {
    common_speculative_draft_params_vec dparams;

    // list of implementations to use and their states
    std::vector<std::unique_ptr<common_speculative_impl>> impls;

    // which implementaion was used for a given seq_id
    std::vector<common_speculative_impl *> impl_last;

    std::vector<double> synth_probs;

    // Independent auxiliary drafter. Observation is the default; the opt-in
    // active pilot may use an already completed, prefix-compatible suffix.
    std::unique_ptr<common_speculative_impl> shadow;
    bool shadow_failed = false;

    // A completed proposal remains owned until both commit and deadline arrive.
    struct shadow_work_result {
        llama_tokens tokens;
        bool success = true;
        int64_t done_us = 0;
        uint64_t worker_us = 0;
        uint64_t decode_us = 0;
        uint64_t selector_us = 0;
    };
    std::future<shadow_work_result> shadow_future;
    dflash_shadow_observation shadow_observation;
    uint64_t shadow_epoch = 0;
    int64_t shadow_request_id = -1;
    bool shadow_verifying = false;
    bool shadow_prefix_valid = false;
    bool shadow_early_launch = false; // experimental: start before the primary draft
    int32_t shadow_primary_pos0 = -1;
    bool shadow_active = false;
    int32_t shadow_mtp_max = 0; // 0 preserves the configured MTP depth
    int32_t shadow_min_suffix = 1;
    llama_tokens shadow_selected = {}; // origin marker survives checkpoint replay
    uint64_t shadow_selected_job = 0;
    uint64_t shadow_used = 0;
    uint64_t shadow_used_tokens = 0;
    uint64_t shadow_verified = 0;
    uint64_t shadow_accepted = 0;
    uint64_t shadow_used_cancelled = 0;

    uint64_t shadow_worker_us = 0;
    uint64_t shadow_decode_us = 0;   // decode through synchronized hidden readback
    uint64_t shadow_selector_us = 0; // encode through synchronized lattice readback
    uint64_t shadow_inject_us = 0;  // host submission; may enqueue asynchronous work
    uint64_t shadow_stage_us = 0;
    uint64_t shadow_late_us = 0;
    uint64_t shadow_late_max_us = 0;
    uint64_t shadow_last_worker_us = 0;
    uint64_t shadow_last_decode_us = 0;
    uint64_t shadow_last_selector_us = 0;
    uint64_t shadow_observed = 0;
    uint64_t shadow_cancelled = 0;
    uint64_t shadow_skipped_invalid = 0;
    // observation counters
    uint64_t shadow_launched = 0;
    uint64_t shadow_skipped_busy = 0;
    uint64_t shadow_skipped_throttle = 0;
    uint64_t shadow_launch_counter = 0;
    int32_t  shadow_every = 1; // launch one block every N primary drafts
    uint64_t shadow_ready = 0;
    uint64_t shadow_late = 0;
    uint64_t shadow_match = 0;
    uint64_t shadow_mismatch = 0;
    uint64_t shadow_usable = 0;
    uint64_t shadow_usable_tokens = 0;
    uint64_t shadow_errors = 0;
};

static common_ngram_map get_common_ngram_map(
        common_speculative_type type,
        const common_params_speculative_ngram_map & config) {
    uint16_t size_key   = config.size_n;
    uint16_t size_value = config.size_m;
    bool     key_only   = type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K;
    uint16_t min_hits   = config.min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

static common_speculative_impl_ngram_cache create_state_ngram_cache(
        const common_speculative_config & config,
        uint32_t n_seq,
        const std::string & path_static,
        const std::string & path_dynamic) {
    uint16_t n_draft = 8; // TODO get from config?

    // TODO bool param in common/common.h to set save_static/save_dynamic?
    bool save_static = false;
    bool save_dynamic = false;

    common_speculative_impl_ngram_cache state(config.params, n_seq, n_draft, path_static, path_dynamic, save_static, save_dynamic);

    return state;
}

std::string common_speculative_type_name_str(const std::vector<common_speculative_type> & types) {
    std::string result;

    for (size_t i = 0; i < types.size(); i++) {
        if (i > 0) {
            result += ",";
        }
        result += common_speculative_type_to_str(types[i]);
    }
    return result;
}

const char * common_speculative_all_types_str() {
    static std::string all_types_str = []() {
        std::vector<common_speculative_type> types;
        types.reserve(COMMON_SPECULATIVE_TYPE_COUNT);
        for (int i = 0; i < COMMON_SPECULATIVE_TYPE_COUNT; i++) {
            types.push_back((common_speculative_type) i);
        }
        return common_speculative_type_name_str(types);
    }();
    return all_types_str.c_str();
}

std::string common_speculative_type_to_str(common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:          return "none";
        case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:  return "draft-simple";
        case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:  return "draft-eagle3";
        case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:     return "draft-mtp";
        case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:  return "draft-dflash";
        case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:  return "draft-dspark";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "ngram-simple";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "ngram-map-k";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "ngram-map-k4v";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "ngram-mod";
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram-cache";
        default:                                    return "unknown";
    }
}

std::vector<common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names) {
    std::vector<common_speculative_type> types;
    types.reserve(names.size());

    for (const auto & name : names) {
        auto type = common_speculative_type_from_name_map.find(name);
        if (type != common_speculative_type_from_name_map.end()) {
            if (type->second == COMMON_SPECULATIVE_TYPE_NONE) {
                return std::vector<common_speculative_type> { COMMON_SPECULATIVE_TYPE_NONE };
            }
            types.push_back(type->second);
            continue;
        }
        throw std::invalid_argument("unknown speculative type: " + name);
    }

    return types;
}

common_speculative_type common_speculative_type_from_name(const std::string & name) {
    const auto it = common_speculative_type_from_name_map.find(name);
    if (it == common_speculative_type_from_name_map.end()) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return it->second;
}

std::vector<common_speculative_type> common_speculative_types_from_gguf(const std::string & path) {
    struct gguf_init_params gguf_params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ nullptr,
    };

    gguf_context_ptr gguf_ctx(gguf_init_from_file(path.c_str(), gguf_params));
    if (!gguf_ctx) {
        return {};
    }

    const int64_t arch_id = gguf_find_key(gguf_ctx.get(), "general.architecture");
    if (arch_id < 0 || gguf_get_kv_type(gguf_ctx.get(), arch_id) != GGUF_TYPE_STRING) {
        return {};
    }

    const std::string arch = gguf_get_val_str(gguf_ctx.get(), arch_id);
    if (arch != "dflash") {
        const uint32_t block_count = gguf_get_val_u32(gguf_ctx.get(), gguf_find_key(gguf_ctx.get(), (arch + ".block_count").c_str()));

        if (gguf_find_tensor(gguf_ctx.get(), ("blk." + std::to_string(block_count - 1) + ".nextn.eh_proj.weight").c_str()) >= 0) {
            return { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        }

        return {};
    }

    // the Markov head distinguishes draft-dspark from draft-dflash
    const auto type = gguf_find_tensor(gguf_ctx.get(), "markov_w1.weight") >= 0
                    ? COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK
                    : COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;

    SPC_INF("auto-detected speculative type '%s' from the draft model metadata\n", common_speculative_type_to_str(type).c_str());

    return { type };
}

static uint32_t common_get_enabled_speculative_configs(const std::vector<common_speculative_type> & configs) {
    uint32_t result = 0;
    for (size_t i = 0; i < configs.size(); i++) {
        result |= (1u << configs[i]);
    }
    return result;
}

int32_t common_speculative_n_max(const common_params_speculative * spec) {
    int32_t n_max = 0;

    for (const auto type : spec->types) {
        switch (type) {
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:
            case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:
            case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:
                n_max = std::max(n_max, std::max(0, spec->draft.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:
                n_max = std::max(n_max, (int32_t) spec->ngram_simple.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k4v.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:
                n_max = std::max(n_max, std::max(0, spec->ngram_mod.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:
                n_max = std::max(n_max, (int32_t) 8);
                break;
            case COMMON_SPECULATIVE_TYPE_NONE:
            case COMMON_SPECULATIVE_TYPE_COUNT:
                break;
        }
    }

    return n_max;
}

int32_t common_speculative_n_max(const common_speculative * spec) {
    int32_t n_max = 0;

    if (spec == nullptr) {
        return n_max;
    }

    for (const auto & impl : spec->impls) {
        n_max = std::max(n_max, std::max(0, impl->n_max));
    }

    return n_max;
}

bool common_speculative_dflash_causal_attn(const llama_model * model) {
    GGML_ASSERT(model != nullptr);
    char buf[16] = {};
    return llama_model_meta_val_str(
               model, "dflash.attention.causal", buf, sizeof(buf)) >= 0 &&
           std::strcmp(buf, "true") == 0;
}

bool common_speculative_dflash_adaptive_dm_supported(int32_t selector_top_k) {
    return selector_top_k <= 0;
}

bool common_speculative_dflash_backend_sampling_allowed(
        bool requested, int32_t target_device_count, bool is_dflash2) {
    // DFlash1 borrows the target output projection. Tensor-parallel targets
    // therefore produce vocabulary-axis split logits, which TOP_K cannot
    // consume independently on each shard. DFlash2 uses its selector lattice.
    return requested && target_device_count <= 1 && !is_dflash2;
}

bool common_speculative_adaptive_dm_supported(const common_speculative * spec) {
    if (spec == nullptr) {
        return false;
    }

    for (const auto & impl : spec->impls) {
        if (impl->adaptive_dm_supported()) {
            return true;
        }
    }

    return false;
}

bool common_speculative_draft_memory_is_shared(const common_speculative * spec) {
    if (spec == nullptr) {
        return false;
    }

    for (const auto & impl : spec->impls) {
        if (impl->draft_memory_is_shared()) {
            return true;
        }
    }

    return false;
}

std::vector<double> common_speculative_synth_rates_resolve(const common_params_speculative * spec, int32_t n_max) {
    const bool has_length = spec->synth_len != -1.0;
    const bool has_rates  = !spec->synth_rates.empty();

    if (!has_length && !has_rates) {
        return {};
    }
    if (has_length && has_rates) {
        throw std::invalid_argument("synthetic acceptance length and rates are mutually exclusive");
    }

    if (n_max <= 0) {
        throw std::invalid_argument("synthetic acceptance requires at least one speculative token");
    }

    if (has_rates) {
        const auto & rates = spec->synth_rates;
        if (rates.size() != (size_t) n_max) {
            throw std::invalid_argument(string_format(
                    "synthetic acceptance rates must contain %d values, got %zu", n_max, rates.size()));
        }

        for (size_t i = 0; i < rates.size(); ++i) {
            if (!std::isfinite(rates[i]) || rates[i] < 0.0 || rates[i] > 1.0) {
                throw std::invalid_argument("synthetic acceptance rates must be finite and within [0, 1]");
            }
            if (i > 0 && rates[i] > rates[i - 1]) {
                throw std::invalid_argument("synthetic acceptance rates must be monotonically non-increasing");
            }
        }

        return rates;
    }

    const double length = spec->synth_len;
    const double length_max = (double) n_max + 1.0;
    if (!std::isfinite(length) || length < 1.0 || length > length_max) {
        throw std::invalid_argument(string_format(
                "synthetic acceptance length must be finite and within [1, %.0f]", length_max));
    }

    double p = 0.0;
    if (length == length_max) {
        p = 1.0;
    } else if (length > 1.0) {
        double p_min = 0.0;
        double p_max = 1.0;
        for (int i = 0; i < 32; ++i) {
            const double p_mid = 0.5 * (p_min + p_max);
            double sum = 0.0;
            double term = p_mid;
            for (int32_t j = 0; j < n_max; ++j) {
                sum += term;
                term *= p_mid;
            }

            if (sum < length - 1.0) {
                p_min = p_mid;
            } else {
                p_max = p_mid;
            }
        }
        p = 0.5 * (p_min + p_max);
    }

    std::vector<double> rates;
    rates.reserve(n_max);
    double rate = p;
    for (int32_t i = 0; i < n_max; ++i) {
        rates.push_back(rate);
        rate *= p;
    }

    return rates;
}

const std::vector<double> & common_speculative_get_synth_probs(const common_speculative * spec) {
    GGML_ASSERT(spec);
    return spec->synth_probs;
}

static bool common_speculative_type_owns_draft_context(common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:
        case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:
        case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:
        case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:
        case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:
            return true;
        default:
            return false;
    }
}

void common_validate_draft_kvarn_mode(const common_params_speculative & params) {
    if (params.draft.kvarn.type == LLAMA_KVARN_TYPE_DISABLED) {
        return;
    }

    bool has_supported_owner = false;
    std::vector<std::string> unsupported;
    for (const common_speculative_type type : params.types) {
        if (common_speculative_type_owns_draft_context(type)) {
            if (has_supported_owner) {
                unsupported.push_back(common_speculative_type_to_str(type));
            }
            has_supported_owner = true;
        }
    }

    if (!has_supported_owner || !unsupported.empty()) {
        std::string modes = common_speculative_type_name_str(params.types);
        if (modes.empty()) {
            modes = "none";
        }
        throw std::invalid_argument(string_format(
                "draft KVarN requires exactly one model-backed speculative mode with an owned KV context; selected speculative mode(s): %s. "
                "Choose an ordinary --spec-draft-type-k/v cache type for this mode",
                modes.c_str()));
    }
}

common_params common_base_params_to_speculative(const common_params & params) {
    const bool has_draft = params.speculative.has_dft();

    common_validate_draft_kvarn_mode(params.speculative);

    const auto & params_spec = params.speculative.draft;
    common_params result = params;

    result.embedding    = false;
    result.pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED;
    result.remote_attn_host.clear();
    result.remote_attn_port = 0;
    result.remote_attn_layers = "0";
    // Prefill migration belongs to the target cache only. The auxiliary MTP
    // context owns its nextn-layer KV and must not inherit the migration mode.
    result.remote_attn_prefill = "remote";

    if (has_draft) {
        // default to global devices value
        if (!params_spec.devices.empty()) {
            result.devices           = params_spec.devices;
        }
        result.model                 = params_spec.mparams;
        result.n_gpu_layers          = params_spec.n_gpu_layers;
        result.tensor_buft_overrides = params_spec.tensor_buft_overrides;

// a draft pinned to a single device doesn't need the meta wrapper an inherited -sm tensor would give it
        // (the device list is null-terminated, so a single device means size 2)
        const size_t n_devs = std::count_if(params_spec.devices.begin(), params_spec.devices.end(),
                [](ggml_backend_dev_t d) { return d != nullptr; });
        if (n_devs == 1) {
            result.split_mode = LLAMA_SPLIT_MODE_LAYER;
        }
        }

        if (params_spec.cpuparams.n_threads > 0) {
            result.cpuparams.n_threads       = params_spec.cpuparams.n_threads;
            result.cpuparams_batch.n_threads = params_spec.cpuparams_batch.n_threads;
        }

    result.cache_type_k = params_spec.cache_type_k;
    result.cache_type_v = params_spec.cache_type_v;
    result.cache_kvarn_bits_k = params_spec.cache_kvarn_bits_k;
    result.cache_kvarn_bits_v = params_spec.cache_kvarn_bits_v;
    result.cache_kvarn_swa_bits_k = 0;
    result.cache_kvarn_swa_bits_v = 0;
    result.kvarn = params_spec.kvarn;
    result.kvarn.swa_key_bits = 0;
    result.kvarn.swa_value_bits = 0;
    result.kv_tail_tokens = "0";
    result.kv_tail_type   = GGML_TYPE_F16;
    result.n_outputs_max = params.n_parallel;
    result.n_outputs_max_per_seq = 1;

    // dflash/dspark decode every sequence's full noise block in one pass
    // TODO: refactor such properties to be announced by the speculative types
    //       something like `struct common_speculative_type_props common_speculative_type_get_props(...);`
    const bool has_block_draft = std::any_of(
        params.speculative.types.begin(), params.speculative.types.end(),
        [](common_speculative_type t) {
            return t == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH || t == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK;
        });
    if (has_block_draft) {
        // per-seq output positions: DFlash decodes anchor + n_max masks (n_max + 1); DSpark n_max -> +1 covers both
        const int32_t per_seq = std::max(1, params_spec.n_max + 1);
        result.n_outputs_max = params.n_parallel * per_seq;
        result.n_batch  = std::max(result.n_batch,  result.n_outputs_max);
        result.n_ubatch = std::max(result.n_ubatch, result.n_outputs_max);
        if (params_spec.backend_sampling) {
            result.n_outputs_max_per_seq = per_seq;
        }
    }

    return result;
}

struct common_speculative_init_result::impl {
    impl() = default;
    ~impl() = default;

    // note: the order in which model, context, etc. are declared matters because their destructors will be called bottom-to-top
    llama_model_ptr   model;
    llama_context_ptr context;

    // shadow auxiliary ownership: an independent model + context, freed before
    // the primary pair above
    llama_model_ptr   model_aux;
    llama_context_ptr context_aux;
};

common_speculative_init_result::common_speculative_init_result(
    common_params & params,
      llama_model * model_tgt,
    llama_context * ctx_tgt) :
    pimpl(new impl{}) {
    const bool has_draft = params.speculative.has_dft();
    const bool spec_mtp = std::find(params.speculative.types.begin(),
                                    params.speculative.types.end(),
                                    COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();
    const bool spec_dflash = std::any_of(
        params.speculative.types.begin(), params.speculative.types.end(),
        [](common_speculative_type type) {
            return type == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH ||
                   type == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK;
        });

    if (params.speculative.draft.local_prefetch && !params.speculative.draft.local_split) {
        throw std::invalid_argument("DFlash2 local prefetch requires local split mode");
    }
    if (params.speculative.draft.local_split && (!has_draft || !spec_dflash || spec_mtp)) {
        throw std::invalid_argument("DFlash2 local split requires a DFlash draft model");
    }

    const bool has_shadow = params.speculative.shadow.enabled();
    if (has_shadow) {
        if (!spec_mtp || spec_dflash) {
            throw std::invalid_argument("shadow auxiliary drafting requires --spec-type draft-mtp as the primary speculative mode");
        }
        if (params.speculative.shadow.n_max < 1) {
            throw std::invalid_argument("--spec-draft-shadow-n-max must be positive");
        }
        if (dflash_shadow_env_integer("GGML_DFLASH_SHADOW_ACTIVE", getenv("GGML_DFLASH_SHADOW_ACTIVE"), 0, 1) &&
                !params.mmproj.path.empty()) {
            throw std::invalid_argument("active shadow pilot currently supports text-only requests");
        }
    }

    common_params params_dft = common_base_params_to_speculative(params);
    auto mparams = common_model_params_to_llama(params_dft);
    auto cparams = common_context_params_to_llama(params_dft);

    // Keep selector math beside the borrowed target head. Otherwise scheduler
    // propagation can send vocabulary-sized logits to the draft device just
    // to gather the top-k scores. Explicit user tensor overrides take priority.
    std::vector<llama_model_tensor_buft_override> local_selector_overrides;
    if (spec_dflash && params.speculative.draft.local_split) {
        auto * device = llama_model_get_output_device(model_tgt);
        if (device && !ggml_backend_dev_is_meta(device)) {
            if (mparams.tensor_buft_overrides) {
                for (auto * entry = mparams.tensor_buft_overrides; entry->pattern; ++entry) {
                    local_selector_overrides.push_back(*entry);
                }
            }
            local_selector_overrides.push_back({
                "^selector_(predecessor|successor|hidden)\\.weight$",
                ggml_backend_dev_buffer_type(device),
            });
            local_selector_overrides.push_back({nullptr, nullptr});
            mparams.tensor_buft_overrides = local_selector_overrides.data();
            LOG_INF("DFlash2 local selector default device: %s\n", ggml_backend_dev_name(device));
        }
    }

    if (spec_mtp) {
        cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    }

    // the draft context holds as many tokens per sequence as the target context
    cparams.n_ctx = llama_n_ctx(ctx_tgt);

    // n_rs_seq stays as common_context_params_to_llama set it: the draft context needs the same rollback window as the target, with n_rs_seq == 0 its seq_rm fails silently on partial acceptance and keeps stale positions
    if (spec_dflash && params.speculative.draft.local_split) {
        cparams.ctx_other = nullptr;
        cparams.dflash_split = true;
        cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
    } else {
        cparams.ctx_other = ctx_tgt;
    }
    cparams.kv_tail_tokens = 0;
    cparams.kv_tail_type   = GGML_TYPE_F16;

    std::string model_path;
    if (has_draft) {
        model_path = params.speculative.draft.mparams.path;
        LOG_INF("%s: loading draft model '%s'\n", __func__, model_path.c_str());

        llama_model * model_dft = llama_model_load_from_file(model_path.c_str(), mparams);
        if (model_dft == NULL) {
            LOG_ERR("%s: failed to load draft model, '%s'\n", __func__, model_path.c_str());
            return;
        }

        pimpl->model.reset(model_dft);

        // DFlash block attention is non-causal unless the model explicitly
        // declares otherwise. Bind this before context validation, graph
        // reservation, and KVarN memory construction.
        if (spec_dflash) {
            cparams.attention_type = common_speculative_dflash_causal_attn(model_dft)
                ? LLAMA_ATTENTION_TYPE_CAUSAL
                : LLAMA_ATTENTION_TYPE_NON_CAUSAL;
        }

        llama_context * ctx_dft = llama_init_from_model(model_dft, cparams);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s: failed to create draft context\n", __func__);
            return;
        }

        pimpl->context.reset(ctx_dft);
    } else if (spec_mtp) {
        model_path = params.model.path;

        LOG_INF("%s: creating MTP draft context against the target model '%s'\n", __func__, model_path.c_str());

        llama_context * ctx_dft = llama_init_from_model(model_tgt, cparams);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s: failed to create MTP context\n", __func__);
            return;
        }

        pimpl->context.reset(ctx_dft);
    }

    if (has_shadow) {
        const common_params_speculative_shadow & shadow = params.speculative.shadow;

        common_params params_aux = common_base_params_to_speculative(params);
        params_aux.model        = shadow.mparams;
        params_aux.n_gpu_layers = shadow.n_gpu_layers;
        if (!shadow.devices.empty()) {
            params_aux.devices = shadow.devices;
        } else {
            LOG_WRN("%s: shadow auxiliary has no explicit devices; inheriting the target device list\n", __func__);
        }

        // the auxiliary block owns one anchor plus n_max noise positions
        const int32_t aux_per_seq = std::max(1, shadow.n_max + 1);
        params_aux.n_outputs_max         = params.n_parallel * aux_per_seq;
        params_aux.n_batch               = std::max(params_aux.n_batch,  params_aux.n_outputs_max);
        params_aux.n_ubatch              = std::max(params_aux.n_ubatch, params_aux.n_outputs_max);
        params_aux.n_outputs_max_per_seq = params.speculative.draft.backend_sampling ? aux_per_seq : 1;

        auto mparams_aux = common_model_params_to_llama(params_aux);
        auto cparams_aux = common_context_params_to_llama(params_aux);

        // keep the auxiliary selector beside the borrowed target head, same
        // rule as the primary local split path; explicit user overrides first
        std::vector<llama_model_tensor_buft_override> aux_selector_overrides;
        {
            auto * device = llama_model_get_output_device(model_tgt);
            if (device && !ggml_backend_dev_is_meta(device)) {
                if (mparams_aux.tensor_buft_overrides) {
                    for (auto * entry = mparams_aux.tensor_buft_overrides; entry->pattern; ++entry) {
                        aux_selector_overrides.push_back(*entry);
                    }
                }
                aux_selector_overrides.push_back({
                    "^selector_(predecessor|successor|hidden)\\.weight$",
                    ggml_backend_dev_buffer_type(device),
                });
                aux_selector_overrides.push_back({nullptr, nullptr});
                mparams_aux.tensor_buft_overrides = aux_selector_overrides.data();
            }
        }

        // the auxiliary context is a standalone DFlash context with explicit
        // host feature exchange; it never shares the target KV or tensors
        cparams_aux.n_ctx          = llama_n_ctx(ctx_tgt);
        cparams_aux.ctx_type       = LLAMA_CONTEXT_TYPE_DEFAULT;
        cparams_aux.ctx_other      = nullptr;
        cparams_aux.dflash_split   = true;
        cparams_aux.pooling_type   = LLAMA_POOLING_TYPE_NONE;
        cparams_aux.kv_tail_tokens = 0;
        cparams_aux.kv_tail_type   = GGML_TYPE_F16;

        LOG_INF("%s: loading shadow auxiliary model '%s'\n", __func__, shadow.mparams.path.c_str());

        llama_model * model_aux = llama_model_load_from_file(shadow.mparams.path.c_str(), mparams_aux);
        if (model_aux == nullptr) {
            throw std::runtime_error(string_format("failed to load shadow auxiliary model, '%s'",
                    shadow.mparams.path.c_str()));
        }

        if (!common_speculative_are_compatible(model_tgt, model_aux)) {
            llama_model_free(model_aux);
            throw std::runtime_error(string_format("shadow auxiliary model '%s' is not vocab-compatible with the target model",
                    shadow.mparams.path.c_str()));
        }

        cparams_aux.attention_type = common_speculative_dflash_causal_attn(model_aux)
            ? LLAMA_ATTENTION_TYPE_CAUSAL
            : LLAMA_ATTENTION_TYPE_NON_CAUSAL;

        llama_context * ctx_aux = llama_init_from_model(model_aux, cparams_aux);
        if (ctx_aux == nullptr) {
            llama_model_free(model_aux);
            throw std::runtime_error("failed to create shadow auxiliary context");
        }

        LOG_INF("%s: shadow auxiliary context ready (model '%s', n_max %d); observation pipeline pending\n",
                __func__, shadow.mparams.path.c_str(), shadow.n_max);

        pimpl->model_aux.reset(model_aux);
        pimpl->context_aux.reset(ctx_aux);

        params.speculative.shadow.ctx_tgt = ctx_tgt;
        params.speculative.shadow.ctx_dft = ctx_aux;
    }
}

common_speculative_init_result::~common_speculative_init_result() = default;

llama_model * common_speculative_init_result::model() {
    return pimpl->model.get();
}

llama_context * common_speculative_init_result::context() {
    return pimpl->context.get();
}

llama_context * common_speculative_init_result::context_aux() {
    return pimpl->context_aux.get();
}

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt) {
    return std::make_unique<common_speculative_init_result>(params, model_tgt, ctx_tgt);
}

common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft) {
    const int64_t per_seq = 1 + (int64_t) std::max(0, n_draft);
    const int64_t total   = (int64_t) n_parallel * per_seq;

    return {
        /* .total   = */ (int32_t) std::min<int64_t>(n_batch, total),
        /* .per_seq = */ (int32_t) std::min<int64_t>(n_batch, per_seq),
    };
}

// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq) {
    // Compute the implementations to use based on the config and their order of preference
    std::vector<common_speculative_config> configs = {}; // list of speculative configs to try
    {
        uint32_t enabled_configs = common_get_enabled_speculative_configs(params.types);

        auto add_config_if_enabled = [&](common_speculative_type type, bool available = true) {
            if (available && (enabled_configs & (1u << type))) {
                configs.emplace_back(type, params);
            }
        };

        // when adding a new type - update here the logic above
        static_assert(COMMON_SPECULATIVE_TYPE_COUNT == 11);

        // this list here defines the priority of the speculators
        // the one with highest priority are listed first
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MOD);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE);

        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, params.draft.ctx_dft != nullptr);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_MTP,    params.draft.ctx_dft != nullptr);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH, params.draft.ctx_dft != nullptr);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK, params.draft.ctx_dft != nullptr);
    }

    std::vector<std::unique_ptr<common_speculative_impl>> impls = {};

    for (const common_speculative_config & config : configs) {
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_simple>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_eagle3>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_mtp>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_dflash>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK: {
                    char arch[32] = {};
                    llama_model_meta_val_str(llama_get_model(config.params.draft.ctx_dft), "general.architecture", arch,
                                             sizeof(arch));
                    if (std::string(arch) == "dspark") {
                        impls.emplace_back(new common_speculative_impl_draft_dspark(config.params, n_seq));
                        break;
                    }
                impls.push_back(std::make_unique<common_speculative_impl_draft_dflash>(
                        config.params, n_seq, COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                common_ngram_map ngram_map = get_common_ngram_map(config.type, config.params.ngram_simple);

                uint16_t ngram_size_key   = ngram_map.size_key;
                uint16_t mgram_size_value = ngram_map.size_value;

                auto config_simple = common_ngram_simple_config {
                    /* .size_ngram = */ ngram_size_key,
                    /* .size_mgram = */ mgram_size_value
                };
                auto state = std::make_unique<common_speculative_impl_ngram_simple>(
                    /* .params = */ config.params,
                    /* .n_seq  = */ n_seq,
                    /* .state  = */ config_simple
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k4v), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_mod>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                auto state = create_state_ngram_cache(
                        config, n_seq,
                        params.ngram_cache.lookup_cache_static,
                        params.ngram_cache.lookup_cache_dynamic);
                impls.push_back(std::make_unique<common_speculative_impl_ngram_cache>(state));
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        SPC_TRC("%s", "no implementations specified for speculative decoding\n");
        return nullptr;
    }

    common_speculative_ptr result(new common_speculative {
        /* .dparams     = */ common_speculative_draft_params_vec(n_seq),
        /* .impls       = */ std::move(impls),
        /* .impl_last   = */ std::vector<common_speculative_impl *>(n_seq, nullptr),
        /* .synth_probs = */ {},
    });

    result->shadow_active = dflash_shadow_env_integer("GGML_DFLASH_SHADOW_ACTIVE",
            getenv("GGML_DFLASH_SHADOW_ACTIVE"), 0, 1) != 0;
    result->shadow_mtp_max = dflash_shadow_env_integer("GGML_DFLASH_SHADOW_MTP_MAX",
            getenv("GGML_DFLASH_SHADOW_MTP_MAX"), 0, std::max(0, params.draft.n_max));
    if (result->shadow_active || result->shadow_mtp_max > 0) {
        if (n_seq != 1 || result->impls.size() != 1 ||
                result->impls.front()->type != COMMON_SPECULATIVE_TYPE_DRAFT_MTP) {
            throw std::invalid_argument("shadow active/depth pilot requires one slot and only draft-mtp");
        }
        auto * mtp = static_cast<common_speculative_impl_draft_mtp *>(result->impls.front().get());
        char arch[64] = {};
        llama_model_meta_val_str(llama_get_model(params.draft.ctx_tgt), "general.architecture", arch, sizeof(arch));
        if (mtp->is_mem_shared || mtp->chain_heads ||
                (std::string(arch) != "qwen35" && std::string(arch) != "qwen35moe")) {
            throw std::invalid_argument("shadow active/depth pilot requires Qwen3.5-family MTP with one owned head");
        }
        LOG_INF("shadow pilot: active=%d MTP depth=%d verifier capacity=%d\n",
                (int) result->shadow_active,
                result->shadow_mtp_max > 0 ? result->shadow_mtp_max : params.draft.n_max, params.draft.n_max);
    }
    if (result->shadow_active && !params.shadow.ctx_dft) {
        throw std::invalid_argument("GGML_DFLASH_SHADOW_ACTIVE requires an auxiliary shadow model");
    }

    if (params.shadow.ctx_dft != nullptr) {
        common_params_speculative aux_params = params;
        common_params_speculative_draft & aux = aux_params.draft;

        aux.mparams        = params.shadow.mparams;
        aux.devices        = params.shadow.devices;
        aux.n_gpu_layers   = params.shadow.n_gpu_layers;
        aux.n_max          = params.shadow.n_max;
        aux.n_min          = 0;
        aux.p_min          = params.shadow.p_min;
        aux.local_split    = params.shadow.local_split;
        aux.local_prefetch = false;
        aux.ctx_tgt        = params.shadow.ctx_tgt != nullptr ? params.shadow.ctx_tgt : params.draft.ctx_tgt;
        aux.ctx_dft        = params.shadow.ctx_dft;

        result->shadow = std::make_unique<common_speculative_impl_draft_dflash>(aux_params, n_seq);
        result->shadow->n_call_begin = 0;
        static_cast<common_speculative_impl_draft_dflash *>(result->shadow.get())->shadow_observer = true;

        result->shadow_every = dflash_shadow_env_integer("GGML_DFLASH_SHADOW_EVERY",
                getenv("GGML_DFLASH_SHADOW_EVERY"), 1, std::numeric_limits<int32_t>::max());
        result->shadow_early_launch = dflash_shadow_env_integer("GGML_DFLASH_SHADOW_EARLY",
                getenv("GGML_DFLASH_SHADOW_EARLY"), 0, 1) != 0;
        result->shadow_min_suffix = dflash_shadow_env_integer("GGML_DFLASH_SHADOW_MIN_SUFFIX",
                getenv("GGML_DFLASH_SHADOW_MIN_SUFFIX"), 1, params.draft.n_max);
        if (result->shadow_min_suffix < 1) {
            throw std::invalid_argument("GGML_DFLASH_SHADOW_MIN_SUFFIX must be positive");
        }

        LOG_INF("%s: shadow auxiliary drafter registered (%s, every=%d, early=%d, min_suffix=%d)\n",
                __func__, result->shadow_active ? "active experimental suffix reuse" : "observation only",
                result->shadow_every, (int) result->shadow_early_launch, result->shadow_min_suffix);
    }

    const int32_t n_max_configured = common_speculative_n_max(&params);
    const int32_t n_max_effective  = common_speculative_n_max(result.get());
    const auto rates = common_speculative_synth_rates_resolve(&params, n_max_effective);
    if (result->shadow_active && !rates.empty()) {
        throw std::invalid_argument("active shadow suffix reuse does not support synthetic acceptance");
    }

    std::vector<std::string> rates_str;
    rates_str.reserve(rates.size());
    result->synth_probs.reserve(rates.size());
    double rate_prev = 1.0;
    double acceptance_length = 1.0;
    for (const double rate : rates) {
        result->synth_probs.push_back(rate_prev > 0.0 ? rate / rate_prev : 0.0);
        rates_str.push_back(string_format("%.6g", rate));
        rate_prev = rate;
        acceptance_length += rate;
    }
    if (!result->synth_probs.empty()) {
        SPC_WRN("%s", "synthetic speculative acceptance is enabled for benchmarking; generated output is not valid\n");
        if (n_max_effective != n_max_configured) {
            SPC_WRN("synthetic acceptance draft limit was reduced from %d to %d by the initialized speculative implementations\n",
                    n_max_configured, n_max_effective);
        }
        SPC_INF("synthetic acceptance: n_max = %zu, mean length = %.6f, rates = [%s]\n",
                rates.size(), acceptance_length, string_join(rates_str, ", ").c_str());
    }

    return result.release();
}

static void common_speculative_shadow_record(common_speculative * spec) {
    auto & job = spec->shadow_observation;
    if (!job.id || !job.resolved()) return;
    const auto result = job.evaluate();
    ++spec->shadow_observed;
    if (result.ready) ++spec->shadow_ready;
    else ++spec->shadow_late;
    if (result.prefix_match) ++spec->shadow_match;
    else ++spec->shadow_mismatch;
    if (result.usable) {
        ++spec->shadow_usable;
        spec->shadow_usable_tokens += result.usable_tokens;
    }
    const uint64_t late_us = job.done_us > job.decision_us ? job.done_us - job.decision_us : 0;
    spec->shadow_late_us += late_us;
    spec->shadow_late_max_us = std::max(spec->shadow_late_max_us, late_us);
    if (getenv("GGML_DFLASH_SHADOW_PROF")) {
        const auto ids = [](const std::vector<int32_t> & tokens) {
            std::string text;
            for (int32_t id : tokens) {
                if (!text.empty()) text += ',';
                text += std::to_string(id);
            }
            return text;
        };
        fprintf(stderr, "SHADOWv2 job=%llu epoch=%llu request=%lld pos0=%d anchor=%d c=%zu d=%zu matched=%d prefix_ok=%d position_ok=%d ready=%d remaining=%d usable=%d usable_tokens=%d done_us=%lld decision_us=%lld work_us=%llu dec_us=%llu sel_us=%llu confirmed=[%s] proposed=[%s]\n",
                (unsigned long long) job.id, (unsigned long long) job.epoch,
                (long long) spec->shadow_request_id, job.pos0, job.anchor,
                job.confirmed.size(), job.proposed.size(), result.matched,
                (int) result.prefix_match, (int) result.position_match, (int) result.ready,
                result.remaining, (int) result.usable, result.usable_tokens,
                (long long) job.done_us, (long long) job.decision_us,
                (unsigned long long) spec->shadow_last_worker_us,
                (unsigned long long) spec->shadow_last_decode_us,
                (unsigned long long) spec->shadow_last_selector_us,
                ids(job.confirmed).c_str(), ids(job.proposed).c_str());
    }
    job = {};
}

// Polling a future never discards a completed-but-unpaired result and never
// grants a later deadline. The worker reports its own completion timestamp.
static void common_speculative_shadow_collect(common_speculative * spec, bool record = true) {
    if (spec->shadow_future.valid() &&
            spec->shadow_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        try {
            auto result = spec->shadow_future.get();
            spec->shadow_last_worker_us = result.worker_us;
            spec->shadow_last_decode_us = result.decode_us;
            spec->shadow_last_selector_us = result.selector_us;
            spec->shadow_worker_us += result.worker_us;
            spec->shadow_decode_us += result.decode_us;
            spec->shadow_selector_us += result.selector_us;
            if (!result.success) {
                LOG_ERR("shadow auxiliary decode/selector failed; auxiliary disabled\n");
                ++spec->shadow_errors;
                spec->shadow_failed = true;
                if (spec->shadow_observation.id) ++spec->shadow_cancelled;
                spec->shadow_observation = {};
            } else if (spec->shadow_observation.id) {
                spec->shadow_observation.finish(std::move(result.tokens), result.done_us);
            }
        } catch (const std::exception & error) {
            LOG_ERR("shadow auxiliary worker failed: %s; auxiliary disabled\n", error.what());
            ++spec->shadow_errors;
            spec->shadow_failed = true;
            if (spec->shadow_observation.id) ++spec->shadow_cancelled;
            spec->shadow_observation = {};
        }
    }
    if (record) common_speculative_shadow_record(spec);
}

static bool common_speculative_shadow_flush(common_speculative * spec) {
    if (spec->shadow_failed || spec->shadow_future.valid() || spec->shadow_verifying) return false;
    auto * impl = static_cast<common_speculative_impl_draft_dflash *>(spec->shadow.get());
    const int64_t start = ggml_time_us();
    const bool ok = impl->shadow_inject();
    spec->shadow_inject_us += (uint64_t) (ggml_time_us() - start);
    if (!ok) {
        LOG_ERR("shadow auxiliary injection/trim failed; auxiliary disabled\n");
        ++spec->shadow_errors;
        spec->shadow_failed = true;
    }
    return ok;
}

static void common_speculative_shadow_cancel_selected(common_speculative * spec) {
    if (spec->shadow_selected.empty()) return;
    ++spec->shadow_used_cancelled;
    if (getenv("GGML_DFLASH_SHADOW_PROF")) {
        fprintf(stderr, "SHADOWACTIVE cancel job=%llu request=%lld pos0=%d\n",
                (unsigned long long) spec->shadow_selected_job,
                (long long) spec->shadow_request_id, spec->shadow_primary_pos0);
    }
    spec->shadow_selected.clear();
    spec->shadow_selected_job = 0;
}

// Only request/restore/free boundaries may join a live worker. Cached target
// state is not evidence that the independent auxiliary KV has that prefix.
static void common_speculative_shadow_reset(common_speculative * spec) {
    if (spec->shadow_future.valid()) spec->shadow_future.wait();
    common_speculative_shadow_collect(spec);
    if (spec->shadow_observation.id) {
        ++spec->shadow_cancelled;
        if (getenv("GGML_DFLASH_SHADOW_PROF")) {
            fprintf(stderr, "SHADOWv2 cancel job=%llu epoch=%llu reason=context_reset\n",
                    (unsigned long long) spec->shadow_observation.id,
                    (unsigned long long) spec->shadow_observation.epoch);
        }
    }
    spec->shadow_observation = {};
    common_speculative_shadow_cancel_selected(spec);
    spec->shadow_verifying = false;
    spec->shadow_primary_pos0 = -1;
    spec->shadow_prefix_valid = false;
    ++spec->shadow_epoch;
    auto * impl = static_cast<common_speculative_impl_draft_dflash *>(spec->shadow.get());
    impl->shadow_rows.clear();
    impl->shadow_staged_tokens = 0;
    impl->shadow_trim_pos = -1;
    impl->local_draft_pending = false;
    impl->local_draft_pos0 = -1;
    llama_synchronize(impl->params.ctx_dft);
    if (!impl->trim_local_context(0)) {
        spec->shadow_failed = true;
        ++spec->shadow_errors;
    }
}

void common_speculative_shadow_new_request(common_speculative * spec, llama_seq_id seq_id, int64_t request_id) {
    if (!spec || !spec->shadow || seq_id != 0) return;
    common_speculative_shadow_reset(spec);
    spec->shadow_request_id = request_id;
}

void common_speculative_shadow_end_request(common_speculative * spec, llama_seq_id seq_id) {
    if (!spec || !spec->shadow || seq_id != 0) return;
    common_speculative_shadow_reset(spec);
    common_speculative_print_stats(spec);
}

void common_speculative_shadow_cancel_primary(common_speculative * spec, llama_seq_id seq_id) {
    if (!spec || !spec->shadow || seq_id != 0) return;
    common_speculative_shadow_cancel_selected(spec);
    if (spec->shadow_observation.id && spec->shadow_observation.pos0 == spec->shadow_primary_pos0) {
        if (getenv("GGML_DFLASH_SHADOW_PROF")) {
            fprintf(stderr, "SHADOWv2 cancel job=%llu epoch=%llu request=%lld pos0=%d reason=primary_aborted\n",
                    (unsigned long long) spec->shadow_observation.id,
                    (unsigned long long) spec->shadow_observation.epoch,
                    (long long) spec->shadow_request_id, spec->shadow_primary_pos0);
        }
        ++spec->shadow_cancelled;
        spec->shadow_observation = {};
    }
    spec->shadow_verifying = false;
    spec->shadow_primary_pos0 = -1;
    // The worker still owns ctx_dft; its eventual result is collected but not
    // classified as an observation. The normal fallback never joins it.
}

// Called only after the server's final acceptance decision, including replay.
// accepted contains the real bonus ID, which is absent from verification input.
void common_speculative_shadow_commit(common_speculative * spec, llama_seq_id seq_id,
                                      const llama_tokens & accepted) {
    if (spec && seq_id == 0 && !spec->shadow_selected.empty() && !accepted.empty()) {
        // Count against the original proposal. A replay may have verified an
        // additional replacement bonus; that is not an accepted DFlash token.
        size_t matched = 0;
        while (matched < spec->shadow_selected.size() && matched < accepted.size() &&
                spec->shadow_selected[matched] == accepted[matched]) ++matched;
        ++spec->shadow_verified;
        spec->shadow_accepted += matched;
        if (getenv("GGML_DFLASH_SHADOW_PROF")) {
            fprintf(stderr, "SHADOWACTIVE accept job=%llu request=%lld pos0=%d proposed=%zu accepted=%zu confirmed=%zu\n",
                    (unsigned long long) spec->shadow_selected_job,
                    (long long) spec->shadow_request_id, spec->shadow_primary_pos0,
                    spec->shadow_selected.size(), matched, accepted.size());
        }
        spec->shadow_selected.clear();
        spec->shadow_selected_job = 0;
    }
    if (!spec || !spec->shadow || spec->shadow_failed || seq_id != 0 ||
            !spec->shadow_verifying || spec->shadow_primary_pos0 < 0 || accepted.empty()) return;
    const llama_pos pos0 = spec->shadow_primary_pos0;
    const llama_pos bonus_pos = pos0 + (llama_pos) accepted.size();
    auto * impl = static_cast<common_speculative_impl_draft_dflash *>(spec->shadow.get());
    // The bonus has not been decoded yet: only features strictly before it
    // belong to the committed prefix. Discard rejected feature rows now.
    impl->shadow_clip(bonus_pos);
    impl->shadow_trim_pos = impl->shadow_trim_pos < 0 ? bonus_pos :
            std::min(impl->shadow_trim_pos, bonus_pos);
    spec->shadow_verifying = false;
    if (spec->shadow_observation.id && spec->shadow_observation.pos0 == pos0 &&
            spec->shadow_observation.epoch == spec->shadow_epoch) {
        spec->shadow_observation.commit(accepted);
    }
    common_speculative_shadow_collect(spec);
    common_speculative_shadow_flush(spec);
}

static void common_speculative_shadow_restore(common_speculative * spec, bool clear_all = false) {
    if (!spec || !spec->shadow) return;
    if (!clear_all && spec->shadow_verifying && spec->shadow_primary_pos0 >= 0) {
        // Replay of the same verification: retain the proposal/identity, remove
        // provisional rows, and let the re-decoded real prefix replace them.
        auto * impl = static_cast<common_speculative_impl_draft_dflash *>(spec->shadow.get());
        const llama_pos pos0 = spec->shadow_primary_pos0;
        impl->shadow_clip(pos0);
        impl->shadow_trim_pos = impl->shadow_trim_pos < 0 ? pos0 : std::min(impl->shadow_trim_pos, pos0);
    } else {
        common_speculative_shadow_reset(spec);
    }
}

static void common_speculative_shadow_process(common_speculative * spec, const llama_batch & batch) {
    common_speculative_shadow_collect(spec);
    if (spec->shadow_failed || batch.n_tokens == 0) return;
    auto * impl = static_cast<common_speculative_impl_draft_dflash *>(spec->shadow.get());
    // After restore, only a complete cold prefix can re-enable the auxiliary.
    if (batch.pos && batch.pos[0] == 0) spec->shadow_prefix_valid = true;
    if (!spec->shadow_prefix_valid) return;
    const int64_t start = ggml_time_us();
    const int32_t chunk = std::min<int32_t>(64, llama_n_ubatch(impl->params.ctx_dft));
    for (int32_t off = 0; off < batch.n_tokens; off += chunk) {
        if (!impl->shadow_stage(batch, off, std::min<int32_t>(chunk, batch.n_tokens - off))) {
            spec->shadow_failed = true;
            ++spec->shadow_errors;
            LOG_ERR("shadow auxiliary feature staging failed; auxiliary disabled\n");
            return;
        }
    }
    spec->shadow_stage_us += (uint64_t) (ggml_time_us() - start);
    common_speculative_shadow_flush(spec);
    if (impl->shadow_staged_tokens > 256) {
        spec->shadow_failed = true;
        ++spec->shadow_errors;
        LOG_ERR("shadow auxiliary staging backlog exceeded; auxiliary disabled\n");
    }
}

// This is the actual replacement deadline: before any primary draft executes,
// on every eligible cycle, independently of how often new jobs are launched.
static llama_tokens common_speculative_shadow_decision(common_speculative * spec) {
    const int64_t deadline = ggml_time_us();
    if (spec->dparams.size() != 1 || !spec->dparams[0].drafting) return {};
    auto & job = spec->shadow_observation;
    const auto & dp = spec->dparams[0];
    const bool first_decision = job.id && job.committed && !job.decided;
    if (first_decision) {
        const int32_t configured = common_speculative_n_max(spec);
        const int32_t capacity = dp.n_max >= 0 ? std::min(dp.n_max, configured) : configured;
        job.decide(dp.pos0, dp.id_last, capacity, deadline);
    }
    // Do not retire the observation until the active consumer has examined it.
    // A job published after an earlier decision is never promoted by a later poll.
    common_speculative_shadow_collect(spec, /*record=*/false);
    llama_tokens candidate;
    if (first_decision && spec->shadow_active && dp.temperature <= 0.0f && spec->shadow_prefix_valid) {
        candidate = job.suffix(spec->shadow_epoch, dp.pos0, dp.id_last, spec->shadow_min_suffix);
        if (!candidate.empty()) spec->shadow_selected_job = job.id;
    }
    common_speculative_shadow_record(spec);
    common_speculative_shadow_flush(spec);
    if (spec->shadow_failed) {
        spec->shadow_selected_job = 0;
        return {};
    }
    return candidate;
}

static void common_speculative_shadow_launch(common_speculative * spec, bool early) {
    if (spec->dparams.size() != 1) return;
    const auto & dp = spec->dparams[0];
    if (early) {
        // the primary proposal is not known yet; a cycle that is drafting will
        // either verify a proposal (commit) or cancel this job below
        if (!dp.drafting) return;
    } else if (!dp.result || dp.result->empty()) {
        return;
    }
    if (dp.pos0 < 0 || dp.id_last < 0) return;
    if (!spec->shadow_prefix_valid || dp.temperature > 0.0f) {
        ++spec->shadow_skipped_invalid;
        return;
    }
    common_speculative_shadow_collect(spec);
    if (spec->shadow_future.valid() || spec->shadow_observation.id) {
        ++spec->shadow_skipped_busy;
        return;
    }
    if (!common_speculative_shadow_flush(spec)) return;
    auto * impl = static_cast<common_speculative_impl_draft_dflash *>(spec->shadow.get());
    // Committed rows were flushed at the decision boundary. No rejected token
    // feature is allowed at or beyond the new anchor.
    if (llama_memory_seq_pos_max(llama_get_memory(impl->params.ctx_dft), 0) != dp.pos0 - 1) {
        ++spec->shadow_skipped_invalid;
        spec->shadow_prefix_valid = false;
        LOG_WRN("shadow auxiliary prefix not aligned at pos0=%d; waiting for cold prefill\n", dp.pos0);
        return;
    }
    auto & job = spec->shadow_observation;
    job = {};
    job.id = spec->shadow_launched + 1;
    job.epoch = spec->shadow_epoch;
    job.pos0 = dp.pos0;
    job.anchor = dp.id_last;
    const int32_t pos0 = dp.pos0;
    const llama_token anchor = dp.id_last;
    try {
        spec->shadow_future = std::async(std::launch::async, [impl, pos0, anchor]() {
            common_speculative::shadow_work_result result;
            common_speculative_draft_params_vec params(1);
            auto & draft = params[0];
            draft.drafting = true;
            draft.pos0 = pos0;
            draft.id_last = anchor;
            draft.result = &result.tokens;
            draft.temperature = 0.0f;
            draft.seed = 0;
            const uint64_t decode_before = impl->shadow_stage_decode_us;
            const uint64_t selector_before = impl->shadow_stage_selector_us;
            const int64_t start = ggml_time_us();
            impl->draft(params);
            result.success = !impl->shadow_draft_failed && !impl->local_failed;
            result.done_us = ggml_time_us();
            result.worker_us = (uint64_t) (result.done_us - start);
            result.decode_us = impl->shadow_stage_decode_us - decode_before;
            result.selector_us = impl->shadow_stage_selector_us - selector_before;
            return result;
        });
        ++spec->shadow_launched;
    } catch (const std::exception & error) {
        spec->shadow_observation = {};
        spec->shadow_failed = true;
        ++spec->shadow_errors;
        LOG_ERR("shadow auxiliary launch failed: %s\n", error.what());
        return;
    }
    if (getenv("GGML_DFLASH_SHADOW_PROF")) {
        fprintf(stderr, "SHADOWv2 launch job=%llu epoch=%llu request=%lld pos0=%d anchor=%d n_max=%d\n",
                (unsigned long long) job.id, (unsigned long long) job.epoch,
                (long long) spec->shadow_request_id, pos0, anchor, impl->n_max);
    }
}

void common_speculative_free(common_speculative * spec) {
    if (!spec) return;
    if (spec->shadow) {
        common_speculative_shadow_reset(spec);
        common_speculative_print_stats(spec); // final accounting includes cancellations
    }
    delete spec;
}

common_speculative_draft_params & common_speculative_get_draft_params(
        common_speculative * spec,
        llama_seq_id seq_id) {
    GGML_ASSERT(spec);
    GGML_ASSERT(seq_id < (llama_seq_id) spec->dparams.size());

    return spec->dparams[seq_id];
}

void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(seq_id, prompt);
        impl->n_call_begin++;
    }

    if (spec->shadow) {
        common_time_meas tm(spec->shadow->t_begin_us, !spec->shadow->gen_perf);
        spec->shadow->begin(seq_id, prompt);
        spec->shadow->n_call_begin++;
    }
}

bool common_speculative_process(common_speculative * spec, const llama_batch & batch) {
    bool result = true;

    if (spec == nullptr) {
        return result;
    }

    for (auto & impl : spec->impls) {
        result = result && impl->process(batch);
    }

    if (spec->shadow && !spec->shadow_failed) {
        common_speculative_shadow_process(spec, batch);
    }

    return result;
}

bool common_speculative_bootstrap(common_speculative * spec, const llama_batch & batch, uint64_t decode_id) {
    return spec && spec->impls.size() == 1 && spec->impls.front()->bootstrap(batch, decode_id);
}

bool common_speculative_is_ready(common_speculative * spec, llama_seq_id seq_id, llama_pos next_pos) {
    return spec && std::all_of(spec->impls.begin(), spec->impls.end(), [&](const auto & impl) {
        return impl->is_ready(seq_id, next_pos);
    });
}

bool common_speculative_draft(common_speculative * spec) {
    if (spec == nullptr) {
        return true;
    }

    auto & dparams = spec->dparams;

    {
        int n_drafting = 0;

        for (auto & dp : dparams) {
            GGML_ASSERT(!dp.drafting || dp.result->empty());

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            return true;
        }
    }

    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
        if (dparams[seq_id].drafting && !common_speculative_is_ready(spec, seq_id, dparams[seq_id].pos0)) {
            SPC_ERR("speculation is not ready for seq %d at position %d\n", seq_id, dparams[seq_id].pos0);
            // Preserve the historical postcondition for all callers: an early
            // readiness failure must not leave dp.drafting armed with an empty result.
            dparams[seq_id].drafting = false;
            return false;
        }
    }

    if (spec->shadow_active && dparams[0].drafting) {
        const auto * mtp = static_cast<common_speculative_impl_draft_mtp *>(spec->impls.front().get());
        const auto hi = llama_memory_seq_pos_max(llama_get_memory(mtp->params.ctx_dft), 0);
        const auto expected = dparams[0].pos0 - 1;
        const bool empty_bootstrap = hi == -1 && mtp->history_start[0] == dparams[0].pos0;
        if (mtp->pending_pos[0] != expected || (hi != expected && !empty_bootstrap)) {
            throw std::runtime_error("active shadow: primary MTP carry/KV does not match the committed prefix");
        }
    }

    bool shadow_due = false;
    if (spec->shadow && !spec->shadow_failed) {
        auto candidate = common_speculative_shadow_decision(spec);
        spec->shadow_primary_pos0 = dparams.size() == 1 ? dparams[0].pos0 : -1;
        if (!candidate.empty()) {
            auto & dp = dparams[0];
            GGML_ASSERT(spec->shadow_selected.empty());
            spec->shadow_selected = std::move(candidate);
            *dp.result = spec->shadow_selected;
            if (dp.dists) dp.dists->clear();
            dp.drafting = false;
            spec->impl_last[0] = nullptr;
            spec->shadow_verifying = true;
            ++spec->shadow_used;
            spec->shadow_used_tokens += dp.result->size();
            if (getenv("GGML_DFLASH_SHADOW_PROF")) {
                fprintf(stderr, "SHADOWACTIVE select job=%llu epoch=%llu request=%lld pos0=%d anchor=%d n=%zu\n",
                        (unsigned long long) spec->shadow_selected_job, (unsigned long long) spec->shadow_epoch,
                        (long long) spec->shadow_request_id, dp.pos0, dp.id_last, dp.result->size());
            }
            // One auxiliary block is consumed once. This first active pilot
            // starts no new worker on a promoted round and never waits for one.
            return true;
        }
        shadow_due = dflash_shadow_schedule_due(spec->shadow_every, spec->shadow_launch_counter);
        if (!shadow_due) ++spec->shadow_skipped_throttle;
        if (shadow_due && spec->shadow_early_launch) {
            common_speculative_shadow_launch(spec, /*early=*/true);
        }
    }

    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            const int32_t saved_max = dparams[0].n_max;
            if (spec->shadow_mtp_max > 0) {
                // Initialization restricts this pilot to one MTP implementation
                // and one sequence. Keep the public verifier capacity intact.
                dparams[0].n_max = saved_max >= 0 ? std::min(saved_max, spec->shadow_mtp_max) : spec->shadow_mtp_max;
            }
            try {
                impl->draft(dparams);
            } catch (...) {
                dparams[0].n_max = saved_max;
                throw;
            }
            dparams[0].n_max = saved_max;
            impl->n_call_draft++;
        }

        int n_drafting = 0;

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            auto & result = *dp.result;

            // a new draft has been sampled
            if (dp.drafting && !result.empty()) {
                dp.drafting = false;

                if (dp.n_max > 0) {
                    if (!result.empty() && (int) result.size() > dp.n_max) {
                        SPC_DBG("truncating draft to %d tokens\n", dp.n_max);
                        result.resize(dp.n_max);
                    }
                }

                if (!result.empty()) {
                    SPC_DBG("called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n",
                            common_speculative_type_to_str(impl.get()->type).c_str(), dp.prompt->size(),
                            impl.get()->n_call_draft, result.size());

                    // remember which implementation was used
                    spec->impl_last[seq_id] = impl.get();

                    impl->n_gen_drafts++;
                    impl->n_gen_tokens += result.size();

                    if (impl->n_gen_tokens_per_pos.size() < result.size()) {
                        impl->n_gen_tokens_per_pos.resize(result.size(), 0);
                    }
                    for (size_t i = 0; i < result.size(); ++i) {
                        impl->n_gen_tokens_per_pos[i]++;
                    }
                }
            }

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            break;
        }
    }

    // these sequences failed to generate a draft
    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
        auto & dp = dparams[seq_id];

        if (dp.drafting) {
            dp.drafting = false;
        }
    }

    // observation-only auxiliary block from the same anchor
    if (spec->shadow && !spec->shadow_failed) {
        if (spec->shadow_early_launch) {
            // the job started before the primary draft; if that draft produced
            // nothing there is no verification to commit it
            spec->shadow_verifying = dparams.size() == 1 && dparams[0].result && !dparams[0].result->empty();
            if (!spec->shadow_verifying) {
                common_speculative_shadow_cancel_primary(spec, 0);
            }
        } else {
            if (shadow_due) common_speculative_shadow_launch(spec, /*early=*/false);
            spec->shadow_verifying = dparams.size() == 1 && dparams[0].result && !dparams[0].result->empty();
        }
    }

    return true;
}

void common_speculative_accept(common_speculative * spec, llama_seq_id seq_id, uint16_t n_accepted) {
    if (seq_id == 0 && !spec->shadow_selected.empty()) {
        // process() refreshed every primary implementation from target hidden
        // rows even though its draft() was skipped. Select the true carry row,
        // including on zero acceptance. The server retains ownership of KV
        // rollback/checkpoints; DFlash's block-relative accept() must not run.
        for (auto & primary : spec->impls) primary->accept(seq_id, n_accepted, true);
        const auto * mtp = static_cast<common_speculative_impl_draft_mtp *>(spec->impls.front().get());
        if (mtp->pending_pos[0] != spec->shadow_primary_pos0 + n_accepted) {
            throw std::runtime_error("active shadow: MTP carry failed to follow accepted external draft");
        }
        return;
    }
    common_speculative_impl * impl = spec->impl_last[seq_id];

    if (impl == nullptr) {
        GGML_ASSERT(n_accepted == 0);
        return;
    }

    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);

        if (impl->n_acc_tokens_per_pos.size() < n_accepted) {
            impl->n_acc_tokens_per_pos.resize(n_accepted, 0);
        }

        for (size_t i = 0; i < n_accepted; ++i) {
            impl->n_acc_tokens_per_pos[i]++;
        }

        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->accept(seq_id, n_accepted, false);
        impl->n_call_accept++;
    }

    // accept with the rest of the implementations, using is_other == true
    for (auto & impl_other : spec->impls) {
        if (impl_other.get() != impl) {
            impl_other->accept(seq_id, n_accepted, true);
        }
    }


}

// TODO: support the case of more than one speculative implementations having a state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data) {
    data.clear();
    if (spec == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->get_state(seq_id, data)) {
            return true;
        }
    }

    return false;
}

namespace {
struct common_speculative_state_view {
    uint32_t type = 0;
    const uint8_t * payload = nullptr;
    size_t payload_size = 0;
};

bool common_speculative_parse_state(
        llama_seq_id seq_id,
        const uint8_t * data,
        size_t data_size,
        common_speculative_state_view & view) {
    constexpr size_t header_size = sizeof(uint32_t)*3 + sizeof(llama_seq_id) + sizeof(uint64_t)*2;
    if (data_size < header_size || data == nullptr) {
        return false;
    }
    const uint8_t * src = data;
    const auto read = [&](auto & value) {
        std::memcpy(&value, src, sizeof(value));
        src += sizeof(value);
    };
    uint32_t magic;
    uint32_t version;
    llama_seq_id saved_seq_id;
    uint64_t payload_size;
    uint64_t checksum;
    read(magic);
    read(version);
    read(view.type);
    read(saved_seq_id);
    read(payload_size);
    read(checksum);
    // The saved sequence ID is provenance, not ownership.  RAM snapshots can
    // be restored into a different server slot, so the envelope must not bind
    // otherwise portable implementation state to its source slot number.
    if (magic != 0x43455053 || version != 1 || saved_seq_id < 0 || seq_id < 0 ||
            payload_size != data_size - header_size) {
        return false;
    }
    uint64_t actual_checksum = 1469598103934665603ULL;
    const auto hash_bytes = [&](const void * ptr, size_t size) {
        const auto * bytes = static_cast<const uint8_t *>(ptr);
        for (size_t i = 0; i < size; ++i) {
            actual_checksum = (actual_checksum ^ bytes[i])*1099511628211ULL;
        }
    };
    hash_bytes(&magic, sizeof(magic));
    hash_bytes(&version, sizeof(version));
    hash_bytes(&view.type, sizeof(view.type));
    hash_bytes(&saved_seq_id, sizeof(saved_seq_id));
    hash_bytes(&payload_size, sizeof(payload_size));
    for (size_t i = 0; i < payload_size; ++i) {
        actual_checksum = (actual_checksum ^ src[i])*1099511628211ULL;
    }
    if (actual_checksum != checksum) {
        return false;
    }
    view.payload = src;
    view.payload_size = size_t(payload_size);
    return true;
}
}

bool common_speculative_validate_state(
        common_speculative * spec,
        llama_seq_id seq_id,
        const std::vector<uint8_t> & data) {
    if (spec == nullptr) {
        return data.empty();
    }
    if (data.empty()) {
        return std::all_of(spec->impls.begin(), spec->impls.end(), [&](const auto & impl) {
            return impl->validate_state(seq_id, data);
        });
    }

    common_speculative_state_view view;
    if (!common_speculative_parse_state(seq_id, data.data(), data.size(), view)) {
        return false;
    }
    const std::vector<uint8_t> payload(view.payload, view.payload + view.payload_size);
    for (const auto & impl : spec->impls) {
        if (uint32_t(impl->type) == view.type) {
            return impl->validate_state(seq_id, payload);
        }
    }
    return false;
}

struct common_speculative_state_restore_plan {
    common_speculative * spec = nullptr;
    common_speculative_impl * impl = nullptr;
    llama_seq_id seq_id = -1;
    std::vector<uint8_t> payload;
    bool clear_all = false;
};

common_speculative_state_restore_plan * common_speculative_prepare_state(
        common_speculative * spec,
        llama_seq_id seq_id,
        const uint8_t * data,
        size_t size) {
    try {
        auto plan = std::make_unique<common_speculative_state_restore_plan>();
        plan->spec = spec;
        plan->seq_id = seq_id;

        if (spec == nullptr) {
            return size == 0 ? plan.release() : nullptr;
        }
        if (size == 0) {
            const std::vector<uint8_t> empty;
            if (!std::all_of(spec->impls.begin(), spec->impls.end(), [&](const auto & impl) {
                        return impl->validate_state(seq_id, empty);
                    })) {
                return nullptr;
            }
            plan->clear_all = true;
            return plan.release();
        }

        common_speculative_state_view view;
        if (common_speculative_parse_state(seq_id, data, size, view)) {
            plan->payload.assign(view.payload, view.payload + view.payload_size);
            for (auto & impl : spec->impls) {
                if (uint32_t(impl->type) == view.type) {
                    if (!impl->validate_state(seq_id, plan->payload)) {
                        return nullptr;
                    }
                    plan->impl = impl.get();
                    return plan.release();
                }
            }
            return nullptr;
        }
        if (data == nullptr) {
            return nullptr;
        }

        // raw implementation payload (as produced by common_speculative_get_state)
        plan->payload.assign(data, data + size);
        for (auto & impl : spec->impls) {
            if (impl->validate_state(seq_id, plan->payload)) {
                plan->impl = impl.get();
                return plan.release();
            }
        }
    } catch (const std::bad_alloc &) {
        return nullptr;
    }
    return nullptr;
}

void common_speculative_state_restore_plan_commit(common_speculative_state_restore_plan * plan) {
    GGML_ASSERT(plan != nullptr);
    if (plan->spec == nullptr) {
        return;
    }
    common_speculative_shadow_restore(plan->spec, plan->clear_all);
    if (plan->clear_all) {
        const std::vector<uint8_t> empty;
        for (auto & impl : plan->spec->impls) {
            GGML_ASSERT(impl->set_state(plan->seq_id, empty));
        }
        return;
    }
    GGML_ASSERT(plan->impl != nullptr);
    GGML_ASSERT(plan->impl->set_state(plan->seq_id, plan->payload));
}

void common_speculative_state_restore_plan_free(common_speculative_state_restore_plan * plan) {
    delete plan;
}

bool common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data) {
    std::unique_ptr<common_speculative_state_restore_plan,
            decltype(&common_speculative_state_restore_plan_free)> plan(
        common_speculative_prepare_state(spec, seq_id, data.data(), data.size()),
        common_speculative_state_restore_plan_free);
    if (!plan) {
        return false;
    }
    common_speculative_state_restore_plan_commit(plan.get());
    return true;
}

bool common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data,
        llama_pos expected_pos) {
    if (spec == nullptr) {
        return false;
    }

    bool restored = false;
    for (auto & impl : spec->impls) {
        restored = impl->set_state(seq_id, data, expected_pos) || restored;
    }

    if (restored && seq_id == 0) {
        common_speculative_shadow_restore(spec, data.empty() && expected_pos < 0);
    }

    return restored;
}

void common_speculative_print_stats(const common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    std::vector<const common_speculative_impl *> stats_impls;
    stats_impls.reserve(spec->impls.size() + (spec->shadow ? 1 : 0));
    for (const auto & impl : spec->impls) {
        stats_impls.push_back(impl.get());
    }
    if (spec->shadow) {
        LOG_INF("shadow active: selected=%llu proposed=%llu verified=%llu accepted=%llu cancelled=%llu pending=%d\n",
                (unsigned long long) spec->shadow_used, (unsigned long long) spec->shadow_used_tokens,
                (unsigned long long) spec->shadow_verified, (unsigned long long) spec->shadow_accepted,
                (unsigned long long) spec->shadow_used_cancelled, (int) !spec->shadow_selected.empty());
        stats_impls.push_back(spec->shadow.get());
    }

    for (const common_speculative_impl * impl : stats_impls) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->t_begin_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_draft_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_accept_us / 1000.0;
            str_perf = ", dur(b,g,a) = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        std::string str_stats;
        if (impl->n_call_accept > 0) {
            const double mean =
                1.0 + (double) impl->n_acc_tokens / (double) impl->n_call_accept;
            std::ostringstream tmp;
            tmp << std::fixed << std::setprecision(3);
            for (size_t i = 0; i < impl->n_acc_tokens_per_pos.size(); ++i) {
                if (i > 0) {
                    tmp << ", ";
                }
                tmp << (double) impl->n_acc_tokens_per_pos[i] / (double) impl->n_call_accept;
            }
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << mean;
            str_stats = ", #mean acc len = " + oss.str() + ", #acc rate/pos = (" + tmp.str() + ")";
        }

        if (getenv("GGML_MTP_PROF")) {
            std::string gen_pos;
            std::string acc_pos;
            for (size_t i = 0; i < impl->n_gen_tokens_per_pos.size(); ++i) {
                if (i > 0) {
                    gen_pos += ",";
                }
                gen_pos += std::to_string(impl->n_gen_tokens_per_pos[i]);
            }
            for (size_t i = 0; i < impl->n_acc_tokens_per_pos.size(); ++i) {
                if (i > 0) {
                    acc_pos += ",";
                }
                acc_pos += std::to_string(impl->n_acc_tokens_per_pos[i]);
            }
            fprintf(stderr, "MTPSTATS type=%s steps=%zu gen_drafts=%zu acc_drafts=%zu gen_tokens=%zu acc_tokens=%zu gen_per_pos=[%s] acc_per_pos=[%s]\n",
                    common_speculative_type_to_str(impl->type).c_str(),
                    impl->n_call_accept, impl->n_gen_drafts, impl->n_acc_drafts,
                    impl->n_gen_tokens, impl->n_acc_tokens,
                    gen_pos.c_str(), acc_pos.c_str());
        }

        SPC_TRC("statistics %16s: #calls(b,g,a) = %4zu %6zu %6zu, #gen drafts = %6zu, #acc drafts = %5zu, #gen tokens = %6zu, #acc tokens = %5zu%s%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_stats.c_str(),
                str_perf.c_str());
    }

    if (spec->shadow) {
        LOG_INF("shadow auxiliary: launched=%llu skipped_busy=%llu skipped_throttle=%llu ready=%llu late=%llu prefix_match=%llu prefix_mismatch=%llu usable_blocks=%llu usable_tokens=%llu errors=%llu\n",
                (unsigned long long) spec->shadow_launched, (unsigned long long) spec->shadow_skipped_busy,
                (unsigned long long) spec->shadow_skipped_throttle,
                (unsigned long long) spec->shadow_ready, (unsigned long long) spec->shadow_late,
                (unsigned long long) spec->shadow_match, (unsigned long long) spec->shadow_mismatch,
                (unsigned long long) spec->shadow_usable, (unsigned long long) spec->shadow_usable_tokens,
                (unsigned long long) spec->shadow_errors);
        LOG_INF("shadow accounting: observed=%llu cancelled=%llu pending=%llu skipped_invalid=%llu epoch=%llu\n",
                (unsigned long long) spec->shadow_observed,
                (unsigned long long) spec->shadow_cancelled,
                (unsigned long long) (spec->shadow_launched - spec->shadow_observed - spec->shadow_cancelled),
                (unsigned long long) spec->shadow_skipped_invalid, (unsigned long long) spec->shadow_epoch);
        LOG_INF("shadow stages: worker_us=%llu decode_us=%llu selector_us=%llu inject_submit_us=%llu stage_us=%llu late_total_ms=%.1f late_max_ms=%.1f\n",
                (unsigned long long) spec->shadow_worker_us, (unsigned long long) spec->shadow_decode_us,
                (unsigned long long) spec->shadow_selector_us, (unsigned long long) spec->shadow_inject_us,
                (unsigned long long) spec->shadow_stage_us,
                (double) spec->shadow_late_us / 1000.0, (double) spec->shadow_late_max_us / 1000.0);
    }
}

bool common_speculative_need_embd_capture(common_speculative * spec) {
    if (spec == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->need_embd_capture()) {
            return true;
        }
    }

    if (spec->shadow && spec->shadow->need_embd_capture()) {
        return true;
    }

    return false;
}

bool common_speculative_dspark_stage_ctx_test(common_speculative * spec,
                                              llama_seq_id         seq_id,
                                              const float *        feat,
                                              int64_t              n_rows,
                                              int64_t              n_embd_cap,
                                              const int32_t *      pos) {
    if (!spec) {
        return false;
    }
    bool staged = false;
    for (auto & impl : spec->impls) {
        staged |= impl->stage_test_ctx_feat(seq_id, feat, n_rows, n_embd_cap, pos);
    }
    return staged;
}
