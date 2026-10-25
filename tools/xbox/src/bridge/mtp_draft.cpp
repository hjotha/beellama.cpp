// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// MTP draft generation, mirroring common_speculative_impl_draft_mtp in the
// fork's common/speculative.cpp. Read that for the algorithm; this file keeps
// only the part a single-session frontend needs.
//
// Two behaviours from the reference are load-bearing and kept verbatim:
//
//   * The draft batch carries embeddings, not tokens, after the first row.
//     llama_batch_init(n, 0, 1) allocates only one of token/embd, so the token
//     array is reallocated separately. Getting this wrong yields a batch the
//     draft model reads as zeros.
//   * Drafting stops at the first candidate below p_min. MTP on a 4B target is
//     cheap but not free, and a low-confidence token costs a target decode to
//     reject, so drafting past it is a net loss.

#include "xllama/mtp_draft.h"

#include "xllama/ggml_d3d12.h"
#include "xllama/platform.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace xllama {

namespace {

// Softmax probability of the drafted token, from the draft context's own logits
// at |idx|. llama_sampler_get_candidates would do this, but it is declared in
// the fork's common layer, which this frontend does not link.
//
// Early exit (plan 003, stage 4.4): the candidate fails when 1/sum < p_min,
// i.e. sum > 1/p_min. Every term is positive and the maximum logit's term is
// exactly 1, so once the running sum exceeds the limit the verdict cannot
// change — stop instead of summing the whole vocabulary. The result is
// identical to the full sum; only the work is skipped. The limit is
// recomputed from the same p_min the caller gates on.
float top_prob(llama_context* ctx, int32_t idx, float p_min) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    const float* logits = llama_get_logits_ith(ctx, idx);
    if (!logits || n_vocab <= 0)
        return 0.0f;
    float max_l = logits[0];
    for (int i = 1; i < n_vocab; ++i)
        if (logits[i] > max_l)
            max_l = logits[i];
    const double limit = 1.0 / static_cast<double>(p_min);
    double sum = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum += std::exp(static_cast<double>(logits[i] - max_l));
        if (sum > limit)
            return 0.0f; // 1/sum < p_min is already decided
    }
    // exp of the largest logit is 1 by construction, so it contributes exactly 1.
    return static_cast<float>(1.0 / sum);
}

// Diagnostic-only (bench_swiglu_mtp_diag): the same softmax sum WITHOUT the
// early exit, plus the real probability, so the arm whose gate returned 0 (the
// early-exit marker) can be compared by its actual value.
float top_prob_full(llama_context* ctx, int32_t idx, float p_min, double* full_sum,
                    double* limit_out) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    const float* logits = llama_get_logits_ith(ctx, idx);
    if (!logits || n_vocab <= 0)
        return 0.0f;
    float max_l = logits[0];
    for (int i = 1; i < n_vocab; ++i)
        if (logits[i] > max_l)
            max_l = logits[i];
    const double limit = 1.0 / static_cast<double>(p_min);
    double sum = 0.0;
    for (int i = 0; i < n_vocab; ++i)
        sum += std::exp(static_cast<double>(logits[i] - max_l));
    if (full_sum)
        *full_sum = sum;
    if (limit_out)
        *limit_out = limit;
    return static_cast<float>(1.0 / sum);
}

} // namespace

MtpDrafter::~MtpDrafter() {
    if (m_smpl)
        llama_sampler_free(m_smpl);
    if (m_batch.token)
        llama_batch_free(m_batch);
    if (m_ctx)
        llama_free(m_ctx);
}

bool MtpDrafter::init(llama_model* model, llama_context* target_ctx,
                      llama_context_params target_cparams, const MtpDraftParams& params,
                      int n_embd) {
    if (m_ctx)
        return true;
    if (!model) {
        log_output("[xllama] mtp: no model, drafting disabled\n");
        return false;
    }

    const int32_t dft_n_embd = llama_model_n_embd_out(model);
    if (dft_n_embd != n_embd) {
        // The draft head's input row width has to match the target's, otherwise
        // the embeddings fed back in are silently reinterpreted.
        log_output("[xllama] mtp: draft n_embd_out=" + std::to_string(dft_n_embd) + " != target " +
                   std::to_string(n_embd) + "; drafting disabled\n");
        return false;
    }

    m_params = params;
    if (m_params.n_max < m_params.n_min)
        m_params.n_max = m_params.n_min;

    // Three deviations from the target params, each mirroring what the fork's
    // common layer does when it stands up a draft MTP context:
    //
    //  * ctx_other = the target, as in the native bootstrap. The fork enables
    //    sharing only for architectures that support it; Qwen3.5 clears this
    //    pointer and retains a private draft KV, so committed-prefix catch-up
    //    remains necessary. Passing ctx_other does not prove shared storage.
    //  * ctx_type = LLAMA_CONTEXT_TYPE_MTP, which is what makes the graph
    //    builder emit the NextN head rather than a duplicate target decoder.

    // mtp_reserve_enabled stays at its default false. It pre-allocates the
    // mixed-cache footprint the KVarN path wants; the fork keeps it off by
    // default so a model that merely has an MTP head does not phantom-reserve.
    llama_context_params cparams = target_cparams;
    cparams.n_threads = std::min(
        cparams.n_threads, mtp_draft_threads(std::getenv("XLLAMA_MTP_THREADS"), cparams.n_threads));
    cparams.n_threads_batch = cparams.n_threads;
    // Read target_ctx directly, not m_target_ctx: the member is only
    // assigned after a successful init, so it is still null here.
    cparams.ctx_other = target_ctx;
    cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    // n_rs_seq is deliberately NOT zeroed. The fork's own setup leaves it at
    // whatever the target resolved to, and says why: with n_rs_seq == 0 the
    // draft's seq_rm fails silently on partial acceptance and keeps stale
    // positions. Zeroing it here is what made the draft's cache disagree with
    // the target's.
    cparams.kv_tail_tokens = 0;
    cparams.kv_tail_type = GGML_TYPE_F16;

    // The draft never sees more than n_max + 1 tokens at a time: the committed
    // token whose hidden row is fed back, plus the deepest proposal. Inheriting
    // the target's n_batch made the compute reservation ask for a full-prompt
    // graph, and with the target already holding ~4000 MB on a 4 GB console that
    // allocation failed outright -- "failed to allocate compute pp buffers" --
    // so the draft context never came up.
    const int32_t n_draft_batch = m_params.n_max + 2;
    if (cparams.n_batch > n_draft_batch)
        cparams.n_batch = n_draft_batch;
    if (cparams.n_ubatch > n_draft_batch)
        cparams.n_ubatch = n_draft_batch;

    // No cparams.samplers on purpose. Offloading sampling to the backend needs
    // TOP_K / ARGMAX shaders, which ggml-d3d12 does not implement, so the
    // sampled-token slots stayed LLAMA_TOKEN_NULL and every draft came back
    // empty. It also flips needs_raw_logits false, suppressing the very logits
    // the CPU sampler below reads. Sampling stays on the CPU.

    m_ctx = llama_init_from_model(model, cparams);
    if (!m_ctx) {
        log_output("[xllama] mtp: draft context creation failed (n_ctx=" +
                   std::to_string(cparams.n_ctx) + " n_batch=" + std::to_string(cparams.n_batch) +
                   " n_ubatch=" + std::to_string(cparams.n_ubatch) +
                   " ctx_type=" + std::to_string(static_cast<int>(cparams.ctx_type)) + " shared=" +
                   std::to_string(cparams.ctx_other != nullptr) + "); drafting disabled\n");
        return false;
    }

    m_target_ctx = target_ctx;
    log_output("[xllama] mtp: draft CPU threads=" + std::to_string(cparams.n_threads) +
               " (within the target's shared pool capacity)\n");
    m_n_embd = n_embd;
    m_pending_h.assign(static_cast<size_t>(n_embd), 0.0f);

    const int32_t n_b = static_cast<int32_t>(llama_n_batch(m_ctx));
    m_batch = llama_batch_init(n_b, /*embd=*/n_embd, /*n_seq_max=*/1);
    // llama_batch_init allocates only one of token/embd; MTP needs both.
    m_batch.token =
        static_cast<llama_token*>(std::malloc(sizeof(llama_token) * static_cast<size_t>(n_b)));
    if (m_batch.token)
        std::memset(m_batch.token, 0, sizeof(llama_token) * static_cast<size_t>(n_b));

    // Greedy only (plan 003, stage 4.3): the previous top_k(10) stage neither
    // selected the winner nor saved work — it truncated and sorted cur_p so the
    // greedy stage could pick data[0], but greedy_apply already scans the whole
    // array for the argmax (llama-sampler.cpp:1053). Removing it leaves the
    // selection identical (the global argmax is the max of any top-k) and
    // skips the partial sort. A chain ending in top_k alone would still abort
    // (it never assigns selected), so greedy must be the last stage either way.
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    sparams.no_perf = true;
    m_smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(m_smpl, llama_sampler_init_greedy());

    // Masked draft outputs follow the rows requesting logits. The target is
    // unmasked so process() can read every committed row during catch-up.
    llama_set_embeddings_nextn(m_ctx, true, /*masked=*/true);

    log_output("[xllama] mtp: draft ready, n_max=" + std::to_string(m_params.n_max) +
               " p_min=" + std::to_string(m_params.p_min) +
               " dft_n_batch=" + std::to_string(cparams.n_batch) + "\n");
    // Confirms the draft context is wired for MTP output; if the target's nextn
    // buffer was sized to zero at creation the row read back would be zeros.
    log_output("[xllama] mtp: target nextn head layers=" +
               std::to_string(llama_model_n_layer_nextn(model)) + "\n");
    {
        // .managed is what the fork's own speculative bootstrap gates on: it
        // means load_mtp actually pulled the head's tensors into the model. A
        // GGUF with nextn tensors but no load_mtp leaves n_layer_nextn set and
        // the head unloaded, and then the target graph never emits an h_nextn
        // row at all -- which reads back as zeros.
        const llama_mtp_weights_info wi = llama_model_mtp_weights_get_info(model);
        log_output("[xllama] mtp: split_mtp_weights=" + std::to_string(wi.managed) +
                   " resident=" + std::to_string(wi.resident) +
                   " tensors=" + std::to_string(wi.tensor_count) + "\n");
    }
    return true;
}

std::vector<llama_token> MtpDrafter::draft(llama_token last_token, llama_pos pos,
                                           const float* h_row, int n_embd, int n_max_eff,
                                           bool profile) {
    std::vector<llama_token> out;
    if (!m_ctx || !m_smpl || !h_row || n_embd != m_n_embd)
        return out;

    // The carry row must describe the token about to be predicted. A prompt
    // rewrite or a KV rollback moves pos backwards without the row changing, so
    // a stale carry would draft from the wrong prefix.
    if (m_pending_valid && m_pending_pos != pos - 1) {
        m_pending_valid = false;
        std::memset(m_pending_h.data(), 0, m_pending_h.size() * sizeof(float));
    }
    std::memcpy(m_pending_h.data(), h_row, static_cast<size_t>(n_embd) * sizeof(float));
    m_pending_pos = pos - 1;

    // Drop anything the previous round left at or above pos. The draft cache is
    // rewritten one row at a time, so without this a rejected tail keeps
    // attending to positions that have left the committed prefix, and the
    // recurrent rows desynchronise from the accepted ones. The return value is
    // load-bearing (plan 003, F2.3): a refused rewind leaves the private cache
    // in an unknown state, and drafting on top of it is worse than not
    // drafting — bail out to single-token decoding.
    if (!llama_memory_seq_rm(llama_get_memory(m_ctx), 0, pos, -1)) {
        log_output("[xllama] mtp: draft seq_rm refused at pos " + std::to_string(pos) +
                   "; drafting disabled for this call\n");
        return out;
    }

    // First draft on a real run, logged once. Everything downstream depends on
    // this row being the target's hidden state, and a wrong width, a stale
    // pointer or a non-finite value here surfaces only as an abort inside
    // llama_decode with nothing pointing back at the draft.
    static bool logged_once = false;
    if (!logged_once) {
        logged_once = true;
        float mn = h_row[0], mx = h_row[0];
        bool finite = true;
        for (int i = 0; i < n_embd; ++i) {
            const float v = h_row[i];
            if (!std::isfinite(v)) {
                finite = false;
                break;
            }
            if (v < mn)
                mn = v;
            if (v > mx)
                mx = v;
        }
        log_output(
            "[xllama] mtp: first draft pos=" + std::to_string(pos) +
            " n_embd=" + std::to_string(n_embd) + " n_max=" + std::to_string(m_params.n_max) +
            " row[min,max]=[" + std::to_string(mn) + "," + std::to_string(mx) + "]" +
            (finite ? "" : " NON-FINITE") + " dft_n_batch=" + std::to_string(llama_n_batch(m_ctx)) +
            " dft_pos_max=" + std::to_string(llama_memory_seq_pos_max(llama_get_memory(m_ctx), 0)) +
            "\n");
    }

    // Shared-memory layouts (the reference calls this is_mem_shared, detected
    // via llama_get_ctx_other) reuse one position for every draft row, which is
    // what the Gemma-family assistants require. With a private context each row
    // takes the next position.
    const bool shared = llama_get_ctx_other(m_ctx) != nullptr;

    // Effective depth: the caller may clamp this call below the configured
    // n_max (context / output budget already consumed — plan 003, stage 4.2).
    const int32_t n_max = n_max_eff >= 0 ? std::min(m_params.n_max, n_max_eff) : m_params.n_max;
    if (n_max <= 0)
        return out;

    m_batch.n_tokens = 0;
    m_batch.token[m_batch.n_tokens] = last_token;
    m_batch.pos[m_batch.n_tokens] = pos;
    m_batch.n_seq_id[m_batch.n_tokens] = 1;
    m_batch.seq_id[m_batch.n_tokens][0] = 0;
    m_batch.logits[m_batch.n_tokens] = 1;
    std::memcpy(m_batch.embd + static_cast<size_t>(m_batch.n_tokens) * n_embd, m_pending_h.data(),
                static_cast<size_t>(n_embd) * sizeof(float));
    // Index of the row the current step decodes, i.e. the row just decoded. It
    // changes every depth because the draft batch is rebuilt with the sampled
    // token appended. The fork tracks the same thing as i_last[seq_id].
    m_batch.n_tokens += 1;
    int i_last = m_batch.n_tokens - 1;

    for (int depth = 0; depth < n_max; ++depth) {
        // Counters always; chrono snapshots only with instrumentation on.
        const auto t_dec0 =
            profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const int rc = llama_decode(m_ctx, m_batch);
        if (profile)
            m_stats.decode_ms +=
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_dec0)
                    .count();
        ++m_stats.n_decodes;
        if (rc != 0) {
            log_output("[xllama] mtp: draft decode failed at depth " + std::to_string(depth) +
                       "\n");
            m_batch.n_tokens = 0;
            return {};
        }
        m_batch.n_tokens = 0;

        {
            // Both the sampler and the nextn accessor take an index into the
            // batch that was just decoded, not a token position. The draft
            // context is configured masked (unlike the target), so the accessor
            // resolves the index through output_resolve_row -- and every draft
            // row carries logits=true, so the output index is the batch index.
            const auto t_smpl0 = profile ? std::chrono::steady_clock::now()
                                         : std::chrono::steady_clock::time_point{};
            const llama_token cand = llama_sampler_sample(m_smpl, m_ctx, i_last);
            if (profile)
                m_stats.sample_ms += std::chrono::duration<double, std::milli>(
                                         std::chrono::steady_clock::now() - t_smpl0)
                                         .count();
            if (cand < 0 || cand == LLAMA_TOKEN_NULL)
                break;

            // Diagnostic-only (bench_swiglu_mtp_diag.txt): the draft proposal
            // distribution at this step, with the top-5 margins, so the
            // equal-history comparison can pair it with the target row.
            if (d3d12_swiglu_mtp_diag_enabled()) {
                const float* dl = llama_get_logits_ith(m_ctx, i_last);
                if (dl) {
                    const int nv = llama_vocab_n_tokens(
                        llama_model_get_vocab(llama_get_model(m_ctx)));
                    llama_token best[5] = {-1, -1, -1, -1, -1};
                    float bestv[5] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY};
                    for (int t = 0; t < nv; ++t) {
                        const float v = dl[t];
                        for (int k = 0; k < 5; ++k) {
                            if (v > bestv[k]) {
                                for (int j = 4; j > k; --j) {
                                    bestv[j] = bestv[j - 1];
                                    best[j] = best[j - 1];
                                }
                                bestv[k] = v;
                                best[k] = t;
                                break;
                            }
                        }
                    }
                    char db[640];
                    // 17 significant digits: the full value a float32 carries,
                    // so the two arms can be compared exactly. in_tok/pos are
                    // the input row this proposal was conditioned on.
                    const llama_token in_tok =
                        (i_last >= 0 && i_last < m_batch.n_tokens) ? m_batch.token[i_last] : -1;
                    const llama_pos in_pos =
                        (i_last >= 0 && i_last < m_batch.n_tokens) ? m_batch.pos[i_last] : -1;
                    snprintf(db, sizeof(db),
                             "[xllama] MTPDIAG draft depth=%d row=%d in_tok=%d in_pos=%d "
                             "drafted=%d top5=%d:%.17g,%d:%.17g,%d:%.17g,%d:%.17g,%d:%.17g "
                             "m2=%.17g\n",
                             depth, i_last, static_cast<int>(in_tok), static_cast<int>(in_pos), cand,
                             best[0], static_cast<double>(bestv[0]), best[1],
                             static_cast<double>(bestv[1]), best[2], static_cast<double>(bestv[2]),
                             best[3], static_cast<double>(bestv[3]), best[4],
                             static_cast<double>(bestv[4]),
                             static_cast<double>(bestv[0] - bestv[1]));
                    log_output(db);
                }
            }

            // Only draft while the candidate stays confident: a token the target
            // is unlikely to accept costs a target decode to reject, which is a
            // net loss. The MTP graph emits real logits (qwen35.cpp sets
            // res->t_logits), so this reads the CPU logits.
            //
            // top_prob sums the vocabulary on the CPU (early exit for rejected
            // candidates), per depth and token, so only pay for it when the
            // threshold is actually in use. With p_min <= 0 every candidate passes
            // and the result is discarded. The sum itself early-exits once the
            // verdict is decided (plan 003, stage 4.4).
            if (m_params.p_min > 0.0f) {
                const auto t_p0 = profile ? std::chrono::steady_clock::now()
                                          : std::chrono::steady_clock::time_point{};
                const float p = top_prob(m_ctx, i_last, m_params.p_min);
                if (profile)
                    m_stats.top_prob_ms += std::chrono::duration<double, std::milli>(
                                               std::chrono::steady_clock::now() - t_p0)
                                               .count();
                if (d3d12_swiglu_mtp_diag_enabled()) {
                    double full_sum = 0.0, limit = 0.0;
                    const float p_full = top_prob_full(m_ctx, i_last, m_params.p_min, &full_sum,
                                                       &limit);
                    const llama_pos pos_now = llama_memory_seq_pos_max(
                        llama_get_memory(m_ctx), 0);
                    char pb[320];
                    snprintf(pb, sizeof(pb),
                             "[xllama] MTPDIAG pmin depth=%d drafted=%d pos=%d p=%.17g "
                             "p_full=%.17g sum=%.17g limit=%.17g pmin=%.17g pass=%d early=%d "
                             "delta_full=%.17g\n",
                             depth, cand, static_cast<int>(pos_now), static_cast<double>(p),
                             static_cast<double>(p_full), full_sum, limit,
                             static_cast<double>(m_params.p_min), p >= m_params.p_min ? 1 : 0,
                             p == 0.0f ? 1 : 0,
                             static_cast<double>(p_full) - static_cast<double>(m_params.p_min));
                    log_output(pb);
                }
                if (p < m_params.p_min) {
                    ++m_stats.n_discarded;
                    break;
                }
            }

            const float* h_next = llama_get_embeddings_nextn_ith(m_ctx, i_last);
            if (!h_next)
                break;

            out.push_back(cand);

            if (static_cast<int>(out.size()) >= n_max)
                break;

            // Feed this row's embedding back with the token just sampled.
            const int i = m_batch.n_tokens;
            m_batch.token[i] = cand;
            m_batch.pos[i] = shared ? pos : pos + depth + 1;
            m_batch.n_seq_id[i] = 1;
            m_batch.seq_id[i][0] = 0;
            m_batch.logits[i] = 1;
            std::memcpy(m_batch.embd + static_cast<size_t>(i) * n_embd, h_next,
                        static_cast<size_t>(n_embd) * sizeof(float));
            m_batch.n_tokens = static_cast<int32_t>(i + 1);
            i_last = i;
        }
    }

    m_pending_valid = true;
    return out;
}

bool MtpDrafter::align_to(llama_pos pos) {
    if (!m_ctx)
        return false;

    llama_memory_t mem = llama_get_memory(m_ctx);
    if (pos <= 0) {
        llama_memory_clear(mem, true);
    } else if (!llama_memory_seq_rm(mem, 0, pos, -1)) {
        return false;
    }

    // A positive |pos| keeps the mirror's [0, pos) prefix and the caller proved
    // those positions hold the same tokens as the target's, so a pending row at
    // |pos - 1| is still the carry the next replay needs and must survive. Any
    // other pending position describes a token this trim just dropped, and a full
    // clear leaves nothing to carry: splicing either would feed the draft head a
    // hidden state from a token the mirror no longer holds.
    if (pos <= 0 || !can_carry_from(pos)) {
        m_pending_valid = false;
        m_pending_pos = -1;
        std::memset(m_pending_h.data(), 0, m_pending_h.size() * sizeof(float));
    }
    // Plan 009: the committed carry only describes a live prefix. A clear
    // drops everything; a trim that removes the carried position drops it too.
    if (pos <= 0 || (m_carry_valid && m_carry_pos >= pos))
        m_carry_valid = false;
    return true;
}

bool MtpDrafter::commit_carry(llama_pos committed_end) {
    // Install the snapshot as the pending carry iff it describes exactly the
    // last committed token and the mirror holds at least through it (speculative
    // rows beyond are trimmed by the next align_to). Anything else leaves
    // everything untouched for the caller to fall back (fail-closed).
    if (!m_ctx || !m_carry_valid || m_carry_pos != committed_end)
        return false;
    if (llama_memory_seq_pos_max(llama_get_memory(m_ctx), 0) < committed_end)
        return false;
    m_pending_h = m_carry_h;
    m_pending_pos = m_carry_pos;
    m_pending_valid = true;
    return true;
}

// Plan 009 slot-restore prime (contract in mtp_draft.h). Fail-closed: a
// false return leaves the already-cleared mirror (full prefill).
bool MtpDrafter::prime_zero_carry(llama_pos pos) {
    if (!m_ctx || pos < 0)
        return false;
    if (m_pending_pos == pos && m_pending_valid)
        return true; // already primed for exactly this position
    std::memset(m_pending_h.data(), 0, m_pending_h.size() * sizeof(float));
    m_pending_pos = pos;
    m_pending_valid = true;
    m_carry_valid = false; // a decoy row is not a committed prefix end
    return true;
}

// Plan 009 complete-state persistence (contract in mtp_draft.h).
bool MtpDrafter::get_state(std::vector<uint8_t>& blob) const {
    if (!m_ctx)
        return false;
    try {
        const size_t need = llama_state_seq_get_size(m_ctx, 0);
        if (need == 0)
            return false;
        blob.resize(need);
        const size_t got = llama_state_seq_get_data(m_ctx, blob.data(), blob.size(), 0);
        if (got == 0) {
            blob.clear();
            return false;
        }
        blob.resize(got);
        return true;
    } catch (const std::exception&) {
        blob.clear();
        return false;
    }
}

bool MtpDrafter::set_state(const uint8_t* data, size_t size) {
    if (!m_ctx || !data || size == 0)
        return false;
    try {
        return llama_state_seq_set_data(m_ctx, data, size, 0) > 0;
    } catch (const std::exception&) {
        return false;
    }
}

bool MtpDrafter::prime_pending(const float* row, llama_pos pos) {
    if (!m_ctx || !row || pos < 0)
        return false;
    std::memcpy(m_pending_h.data(), row, m_pending_h.size() * sizeof(float));
    m_pending_pos = pos;
    m_pending_valid = true;
    m_carry_valid = false; // a restored row is not a committed prefix end
    return true;
}

llama_pos MtpDrafter::mirror_pos_min() const {
    if (!m_ctx)
        return -1;
    return llama_memory_seq_pos_min(llama_get_memory(m_ctx), 0);
}

bool MtpDrafter::process(const llama_token* tokens, const float* h_rows, const float* h_pending,
                         int n, llama_pos pos0, bool profile) {
    if (!m_ctx || !tokens || n <= 0)
        return false;
    if (!h_rows && !m_pending_valid)
        return false; // first row would carry garbage

    // The private context mirrors the target's committed prefix: the draft head
    // attends to the same history, and its recurrent rows desynchronise from
    // the target's when they are rebuilt from a different prefix. The reference
    // (speculative.cpp:4093-4190) feeds the target's batch with the hidden rows
    // shifted by one position: row i carries the target's hidden state of token
    // i-1, and the first row carries the carry from the previous round.
    const size_t row_bytes = static_cast<size_t>(m_n_embd) * sizeof(float);
    const int n_batch_eff = std::max(1, static_cast<int>(llama_n_batch(m_ctx)));

    std::vector<float> pending(m_pending_h.size());
    if (h_pending) {
        std::memcpy(pending.data(), h_pending, row_bytes);
    } else if (pos0 == 0) {
        // The very first prompt row has no predecessor: the reference zero-fills
        // the pending row at position 0 (speculative.cpp reset_seq_state).
        std::memset(pending.data(), 0, row_bytes);
    } else if (m_pending_valid && m_pending_pos == pos0 - 1) {
        std::memcpy(pending.data(), m_pending_h.data(), row_bytes);
    } else {
        // No carry for the first replay row: the draft context cannot mirror
        // the target's prefix, and speculating on an unknown state is worse
        // than not speculating (F2.3 fail-closed). A stale carry (a rewrite or
        // rewind between process() calls) is dropped the same way draft()
        // drops it.
        log_output("[xllama] mtp: catch-up has no carry for pos0=" + std::to_string(pos0) +
                   " (pending_pos=" +
                   std::to_string(m_pending_valid ? static_cast<int>(m_pending_pos) : -1) +
                   ", pending_valid=" + (m_pending_valid ? "1" : "0") + ")\n");
        if (m_pending_valid) {
            m_pending_valid = false;
            std::memset(m_pending_h.data(), 0, m_pending_h.size() * sizeof(float));
        }
        return false;
    }

    const auto t0 =
        profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    // The private cache may still hold this round's speculative rows above the
    // committed prefix (draft() wrote them; the target never decoded them into
    // this context). The replay must start at |pos0| with the cache ending at
    // |pos0 - 1| — llama_decode rejects a batch whose start is not past the
    // cache end. Drop the tail first; a refused rewind leaves the cache in an
    // unknown state (F2.3 fail-closed).
    if (pos0 > 0 && !llama_memory_seq_rm(llama_get_memory(m_ctx), 0, pos0, -1)) {
        log_output("[xllama] mtp: catch-up seq_rm refused at pos0=" + std::to_string(pos0) + "\n");
        return false;
    }
    const char* legacy = std::getenv("XLLAMA_MTP_CATCHUP_LOGITS");
    const bool catchup_logits = legacy && std::strcmp(legacy, "1") == 0;
    for (int off = 0; off < n; off += n_batch_eff) {
        const int chunk = std::min(n_batch_eff, n - off);
        m_batch.n_tokens = 0;
        for (int i = 0; i < chunk; ++i) {
            const int idx = m_batch.n_tokens;
            m_batch.token[idx] = tokens[off + i];
            m_batch.pos[idx] = pos0 + static_cast<llama_pos>(off + i);
            m_batch.n_seq_id[idx] = 1;
            m_batch.seq_id[idx][0] = 0;
            // Native common_speculative catch-up requests no outputs: only KV
            // must advance. These logits were never read, but streamed the
            // 521 MB output matrix for every prompt/committed token.
            m_batch.logits[idx] = catchup_logits;
            // Row i carries the target's hidden state of token off+i-1; the
            // first row of the whole prefix carries the carry (pending).
            const float* src = (off + i == 0)
                                   ? pending.data()
                                   : h_rows + static_cast<size_t>(off + i - 1) * m_n_embd;
            std::memcpy(m_batch.embd + static_cast<size_t>(idx) * m_n_embd, src, row_bytes);
            m_batch.n_tokens = static_cast<int32_t>(idx + 1);
        }
        const int rc = llama_decode(m_ctx, m_batch);
        ++m_stats.n_decodes;
        if (rc != 0) {
            log_output("[xllama] mtp: catch-up decode failed at off=" + std::to_string(off) +
                       " rc=" + std::to_string(rc) + "\n");
            m_batch.n_tokens = 0;
            // Plan 009: the mirror may hold partial rows — the committed
            // snapshot no longer describes a trustworthy prefix.
            m_carry_valid = false;
            return false;
        }
        m_batch.n_tokens = 0;
    }
    if (profile)
        m_stats.catchup_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                .count();

    // Carry for the next draft/round is the target's hidden state of the LAST
    // token this prefix ends with — the accept() contract of the reference
    // (speculative.cpp:4421-4434). The draft context's own nextn row is not the
    // target's and must not leak into the carry.
    std::memcpy(m_pending_h.data(), h_rows + static_cast<size_t>(n - 1) * m_n_embd, row_bytes);
    m_pending_pos = pos0 + static_cast<llama_pos>(n) - 1;
    m_pending_valid = true;
    // Plan 009 committed carry: same row/position, in state draft() rounds
    // never touch — so commit_carry() can restore a correct carry at turn
    // end even after speculative rounds moved pending on. Overwritten by
    // every successful replay (always the freshest committed end).
    m_carry_h = m_pending_h;
    m_carry_pos = m_pending_pos;
    m_carry_valid = true;
    return true;
}

} // namespace xllama
