// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// MTP (multi-token prediction) draft generation for the beellama fork.
//
// The fork's MTP is not a second model file: load_mtp on the target
// llama_model_params makes the loader pull the MTP head out of the same
// GGUF, and the draft context is then built from that same model with
// llama_init_from_model. So no extra weights are downloaded and no second
// mmap exists; what the draft context does cost is its own KV cache and its
// own compute per proposed token.
//
// The draft path differs from prompt-lookup in where the hidden state comes
// from. Prompt-lookup invents the next token from token_history. MTP instead
// reads the target's last hidden row via llama_get_embeddings_nextn_ith and
// feeds it back as llama_batch.embd, so the draft model predicts a real
// continuation. That is also why the target context must have been created
// with llama_set_embeddings_nextn(ctx_tgt, true, false).
//
// Everything here is optional and off unless the caller supplies a draft
// context, so a stock upstream pin links and runs unchanged.

#pragma once

#include "llama.h"

// MTP lives behind the fork's internal API rather than llama.h:
// llama_set_embeddings_nextn, llama_get_embeddings_nextn_ith and
// llama_get_ctx_other are declared in the submodule's src/llama-ext.h, and
// that internal directory is on neither the host nor the UWP include path.
// Redeclared here instead of including it, so no build file has to grow an
// include path into llama.cpp's private source tree. They are LLAMA_API, so
// the definitions are exported from the same library and this links.
struct llama_mtp_weights_info {
    bool managed = false;
    bool resident = false;
    size_t host_bytes = 0;
    size_t allocated_bytes = 0;
    size_t gpu_allocated_bytes = 0;
    size_t tensor_count = 0;
    uint64_t model_instance = 0;
    uint64_t model_load_count = 0;
    uint64_t main_gpu_upload_bytes = 0;
    uint64_t mtp_gpu_upload_bytes = 0;
    uint64_t mtp_reloads = 0;
    uint64_t backing_hash = 0;
};

extern "C" {
void llama_set_embeddings_nextn(struct llama_context* ctx, bool value, bool masked);
float* llama_get_embeddings_nextn_ith(struct llama_context* ctx, int32_t i);
struct llama_context* llama_get_ctx_other(struct llama_context* ctx);
struct llama_mtp_weights_info llama_model_mtp_weights_get_info(const struct llama_model* model);
}

#include <cstdint>
#include <vector>

namespace xllama {

// The Xbox has at most six application CPU workers; reject malformed overrides.
inline int mtp_draft_threads(const char* value, int fallback) {
    return value && value[0] >= '1' && value[0] <= '6' && value[1] == '\0' ? value[0] - '0'
                                                                           : fallback;
}

// Knobs mirrored from common_params_speculative in the fork's common layer.
// The defaults match its draft defaults: up to 4 proposed tokens, minimum 1,
// and only keep drafting while the sampled candidate is at least this likely.
struct MtpDraftParams {
    int n_max = 4;
    int n_min = 1;
    float p_min = 0.75f;
};

// Per-call phase accounting, so the decode loop can reconcile where the MTP
// budget goes (plan 003, stage 2). Times are milliseconds; counters are
// counts. The drafter accumulates into this struct during draft() and
// process(); the loop reads it after each call and resets it.
//
// Every chrono snapshot below is gated on the caller's `profile` flag. With it
// off the counters still increment but the time fields stay 0, so an ON/OFF pair
// measures the instrumentation's own cost instead of inferring it from a residual.
struct MtpDraftStats {
    double decode_ms = 0.0;   // llama_decode inside draft() / process()
    double sample_ms = 0.0;   // draft sampler (including the p_min gate)
    double top_prob_ms = 0.0; // the confidence softmax over the vocabulary
    double catchup_ms = 0.0;  // process() catch-up decodes (replayed prefix)
    int n_decodes = 0;        // draft()+process() llama_decode calls
    int n_discarded = 0;      // candidates rejected by the p_min gate
};

// Owns the draft context and its sampler. One instance per Session.
class MtpDrafter {
  public:
    MtpDrafter() = default;
    ~MtpDrafter();

    MtpDrafter(const MtpDrafter&) = delete;
    MtpDrafter& operator=(const MtpDrafter&) = delete;

    // Builds the draft context from |model|, which must already have been
    // loaded with load_mtp = true. Shares the model's weights; only the KV
    // cache and compute buffers are additional. Returns false and logs on
    // failure, in which case draft() is a no-op and callers fall back to
    // single-token decoding.
    bool init(llama_model* model, llama_context* target_ctx, llama_context_params target_cparams,
              const MtpDraftParams& params, int n_embd);

    bool ready() const {
        return m_ctx != nullptr;
    }
    llama_context* ctx() const {
        return m_ctx;
    }

    // Bring the private mirror in line with a target KV that was rewound or
    // cleared. |pos| is how many positions the target now keeps: 0 clears the
    // mirror, a positive value drops the speculative tail at or above it. The
    // pending row describes the mirror, so it cannot survive either. Fails
    // closed when the mirror refuses the trim.
    bool align_to(llama_pos pos);

    // True when the pending row is exactly the hidden state a replay starting at
    // |pos| needs: the mirror holds the prefix below |pos| and the pending row
    // describes position |pos - 1|. The caller must have proven that prefix
    // holds the same tokens as the target's before believing this.
    bool can_carry_from(llama_pos pos) const {
        return m_pending_valid && m_pending_pos == pos - 1;
    }

    // Proposed tokens for the next decode position. |last_token| is the token
    // just committed at |pos|; |h_row| is the target's hidden state for that
    // token (llama_get_embeddings_nextn_ith on the target context). Returns an
    // empty vector when MTP declines, which the caller must treat as "decode
    // one token classically" rather than as an error.
    //
    // |n_max_eff| clamps this call's depth below params().n_max (context or
    // output budget already consumed); -1 keeps the configured depth.
    std::vector<llama_token> draft(llama_token last_token, llama_pos pos, const float* h_row,
                                   int n_embd, int n_max_eff = -1, bool profile = true);

    // Replay a committed prefix into the private context so the draft head
    // attends to the same history the target has. Mirrors the catch-up decode
    // in common_speculative_impl_draft_mtp::process: the first row carries the
    // pending hidden state (|h_pending|), rows 1..n-1 carry |h_rows| shifted by
    // one (the target's row for the token before the one being fed), and the
    // last row's embedding is stashed as the next pending state. Chunked at the
    // draft context's own n_batch, which is n_max+2 and far below a prompt.
    //
    // |tokens| are the committed token ids (target-side), |pos0| the position
    // of tokens[0] in the target cache. Returns false on decode failure, in
    // which case the drafter must be re-validated before drafting again.
    bool process(const llama_token* tokens, const float* h_rows, const float* h_pending, int n,
                 llama_pos pos0, bool profile = true);

    // Phase accounting for the last draft()/process() calls (see MtpDraftStats).
    const MtpDraftStats& stats() const {
        return m_stats;
    }
    void reset_stats() {
        m_stats = MtpDraftStats{};
    }

    const MtpDraftParams& params() const {
        return m_params;
    }

    // Plan 009 complete-state persistence: the private draft KV is real
    // state (Qwen3.5 keeps a private draft KV), so a slot file must be able
    // to carry it or a restore silently degrades the drafter to a partial
    // mirror (documented hole) with a decoy pending row. Test-only accessors
    // follow for the aligned-comparison probes.
    bool get_state(std::vector<uint8_t>& blob) const;
    bool set_state(const uint8_t* data, size_t size);
    // Install a REAL pending row (the true hidden state carried from the
    // save). Fail-closed: no row, no valid pos => false and nothing changes.
    bool prime_pending(const float* row, llama_pos pos);
    llama_pos mirror_pos_min() const;

    // Plan 009 S3 probe (test-only accessors): the pending row bytes and the
    // mirror's position extent, so a diagnostic can compare drafter-boundary
    // state between arms. Production paths never read these.
    const float* pending_row_data() const {
        return m_pending_valid ? m_pending_h.data() : nullptr;
    }
    llama_pos mirror_pos_max() const {
        return m_ctx ? llama_memory_seq_pos_max(llama_get_memory(m_ctx), 0) : -1;
    }

    // Turn-end observability (plan 009): the pending carry + mirror end that
    // the NEXT turn's F3.5 can_carry check will see. Log-only support; the
    // reuse decision never reads these.
    bool pending_valid() const {
        return m_pending_valid;
    }
    llama_pos pending_pos() const {
        return m_pending_pos;
    }

    // Plan 009 committed carry: install the snapshot (last successfully
    // replayed prefix end) as the pending carry when it describes exactly
    // |committed_end| and the mirror holds at least through it. Returns
    // false (leaving everything untouched) otherwise. This is what lets a
    // turn that ended after speculative draft rounds still carry into the
    // next turn: draft() clobbers pending with future positions, but the
    // snapshot survives (only process() success writes it).
    bool commit_carry(llama_pos committed_end);

    // Plan 009 slot-restore: after the mirror was cleared because a disk
    // restore replaced the target prefix (new process), the pending row is
    // gone, so F3.5 would refuse the carry and the next turn would full-
    // prefill. Priming a ZERO pending row at |pos| (the last restored
    // token) makes can_carry_from(pos+1) true again so the tail-only turn
    // keeps MTP drafting. The row is a deliberate DECOY: the draft context
    // below |pos| stays empty and this row does not describe the target's
    // hidden state of that token. It is safe by the same contract every
    // draft already lives under — the target verifies every candidate, so
    // a wrong draft costs time, never output text, and draft() returning
    // empty falls back to classic decode (F2.3). Worst case measured on
    // the rev234 zerocarry arms: acceptance halves, text identical, no
    // state divergence in the target. Fail-closed: returns false (leaving
    // the cleared mirror, i.e. full prefill) unless the drafter is ready
    // and |pos| >= 0.
    bool prime_zero_carry(llama_pos pos);

  private:
    llama_context* m_ctx = nullptr;
    llama_sampler* m_smpl = nullptr;
    llama_batch m_batch{};
    llama_context* m_target_ctx = nullptr;
    int m_n_embd = 0;
    MtpDraftParams m_params{};
    std::vector<float> m_pending_h;
    llama_pos m_pending_pos = -1;
    bool m_pending_valid = false;
    // Plan 009 committed carry: snapshot of the last successfully replayed
    // prefix end (row + position), written only by process() on success.
    // draft() rounds overwrite m_pending_* with future positions, but never
    // touch these — so commit_carry() can restore a correct carry at turn
    // end. Invalidated whenever the mirror prefix dies (align_to rules
    // below, process() decode failure); never persisted (re-established
    // every turn from live rows).
    std::vector<float> m_carry_h;
    llama_pos m_carry_pos = -1;
    bool m_carry_valid = false;
    MtpDraftStats m_stats{};
};

} // namespace xllama
