#include "llama-memory-hybrid.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-context.h"
#include "llama-state-q4.h"
#include "llama-position-split.h"

//
// llama_memory_hybrid
//

llama_memory_hybrid::llama_memory_hybrid(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload_kv,
                     bool   offload_rs,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
                 uint32_t   n_ubatch,
                 uint32_t   tail_tokens,
                ggml_type   tail_type,
                 uint32_t   tail_tokens_requested,
                 uint32_t   tail_rollback_tokens,
    const layer_device_cb & device_for_layer,
                 uint32_t   position_split_p) :
    hparams(model.hparams),
    mem_attn(new llama_kv_cache(
        model,
        model.hparams,
        type_k,
        type_v,
        v_trans,
        offload_kv,
        unified,
        kv_size,
        n_seq_max,
        n_pad,
        n_swa,
        swa_type,
        nullptr,
        filter_attn == nullptr ?
            [&](int32_t il) { return !hparams.is_recr(il); }
            : filter_attn,
        nullptr,
        nullptr,
        "",
        n_ubatch,
        tail_tokens,
        tail_type,
        tail_tokens_requested,
        false,
        tail_rollback_tokens,
        0, false, device_for_layer
    )),
    mem_recr(new llama_memory_recurrent(
        model,
        type_r,
        type_s,
        offload_rs,
        rs_size,
        n_seq_max,
        n_rs_seq,
        filter_recr == nullptr ?
            [&](int32_t il) { return hparams.is_recr(il); }
            : filter_recr
    )),
    position_split_p(position_split_p) {
    if (position_split_p > 0 && !llama_position_split::is_valid_p(position_split_p)) {
        GGML_ABORT("position split boundary P=%u is not a valid multiple of %u below the capacity",
                   position_split_p, llama_position_split::kAlignTokens);
    }
}

llama_memory_hybrid::llama_memory_hybrid(
        const llama_model & model,
        std::unique_ptr<llama_memory_i> mem_attn,
        std::unique_ptr<llama_memory_recurrent> mem_recr) :
    hparams(model.hparams),
    mem_attn(std::move(mem_attn)),
    mem_recr(std::move(mem_recr)) {
}


//
// Position-split ubatch handling (plan §3.2).
//
// An attention range boundary P must never fall inside a prepared ubatch: the
// store of the ubatch would have to write two ranges (KVarN local and Q4
// overflow) with a single store, which the cache does not implement yet. The
// boundary is therefore enforced *before* any prepare runs:
//
//   - a common ubatch that crosses P is divided at P;
//   - the protected recurrent window (the trailing 1 + n_rs_seq tokens that
//     keep the rollback snapshots valid) may not be divided, so a window that
//     crosses P rejects the preparation with an explicit reason, before any
//     state (KV, slot table or recurrent state) is mutated.
//

// Copies one half of an ubatch into an owning llama_ubatch. `sel` lists the
// token indices (in ubatch order) that belong to the new ubatch.
static llama_ubatch llama_ubatch_subset(const llama_ubatch & src, const std::vector<int32_t> & sel) {
    GGML_ASSERT(!sel.empty() && sel.size() <= src.n_tokens);

    llama_ubatch out;
    out.b_equal_seqs = src.b_equal_seqs;
    out.n_tokens     = uint32_t(sel.size());
    out.n_pos        = src.n_pos;

    // Every subset of a single-sequence-set ubatch keeps that sequence set, so
    // n_seq_tokens divides evenly; mixed ubatches are rejected by the caller.
    out.n_seqs     = src.n_seqs;
    out.n_seq_tokens = out.n_tokens / out.n_seqs;
    GGML_ASSERT(out.n_seq_tokens * out.n_seqs == out.n_tokens);

    auto data = std::make_shared<llama_ubatch::data_t>();
    data->token.resize(out.n_tokens);
    data->pos.resize(size_t(out.n_tokens) * src.n_pos);
    data->n_seq_id.resize(out.n_tokens);
    data->seq_id.resize(out.n_tokens);
    data->output.resize(out.n_tokens);
    for (uint32_t s = 0; s < src.n_seqs_unq; ++s) {
        data->seq_id_unq.push_back(src.seq_id_unq[s]);
    }
    data->seq_idx.assign(LLAMA_MAX_SEQ, -1);
    out.n_seqs_unq = src.n_seqs_unq;

    for (size_t i = 0; i < sel.size(); ++i) {
        const int32_t t = sel[i];
        data->token[i] = src.token[t];
        for (uint32_t p = 0; p < src.n_pos; ++p) {
            data->pos[i * src.n_pos + p] = src.pos[size_t(t) * src.n_pos + p];
        }
        data->n_seq_id[i] = src.n_seq_id[t];
        data->output[i]  = src.output[t];

        const int32_t n_ids = src.n_seq_id[t];
        data->seq_id[i] = data->seq_id_data.data() + data->seq_id_data.size();
        for (int32_t s = 0; s < n_ids; ++s) {
            data->seq_id_data.push_back(src.seq_id[t][s]);
        }
    }
    for (uint32_t s = 0; s < src.n_seqs_unq; ++s) {
        data->seq_idx[data->seq_id_unq[s]] = src.seq_idx[data->seq_id_unq[s]];
    }

    out.data       = data;
    out.token      = data->token.data();
    // llama_ubatch carries no embedding width, so the position split is only
    // defined for token ubatches. An embedding ubatch that crosses P is
    // rejected by the caller with an explicit reason instead of being copied
    // with a guessed row stride.
    out.embd       = nullptr;
    out.pos        = data->pos.data();
    out.n_seq_id   = data->n_seq_id.data();
    out.seq_id     = data->seq_id.data();
    out.seq_id_unq = data->seq_id_unq.data();
    out.seq_idx    = data->seq_idx.data();
    out.output     = data->output.data();

    return out;
}

bool llama_position_split::divide_ubatches_at_p(
        const std::vector<llama_ubatch> & in,
                             uint32_t   p,
                             uint32_t   n_keep,
                  std::vector<llama_ubatch> & out,
                             std::string & error) {
    out.clear();
    error.clear();

    for (const llama_ubatch & ub : in) {
        // Absolute position range of this ubatch.
        llama_pos pos_min = ub.pos[0];
        llama_pos pos_max = ub.pos[0];
        for (uint32_t i = 1; i < ub.n_tokens; ++i) {
            pos_min = std::min(pos_min, ub.pos[i]);
            pos_max = std::max(pos_max, ub.pos[i]);
        }

        const bool crosses = pos_min < llama_pos(p) && pos_max >= llama_pos(p);

        if (!crosses) {
            out.push_back(ub);
            continue;
        }

        if (ub.embd != nullptr) {
            error = "embedding ubatch crosses P: the position split is defined for token "
                    "ubatches only";
            return false;
        }
        if (ub.n_seqs != 1 || ub.n_pos != 1) {
            error = "ubatch crossing P has " + std::to_string(ub.n_seqs) +
                    " sequence sets and " + std::to_string(ub.n_pos) +
                    " position axes: only single-sequence-set token ubatches can be divided";
            return false;
        }

        // The protected recurrent window is the trailing 1 + n_rs_seq tokens of
        // the sequence. It must stay in one ubatch so the rollback snapshots
        // remain valid, so it can never be divided.
        if (n_keep > 0 && ub.n_seq_tokens > n_keep) {
            const uint32_t win_start = ub.n_seq_tokens - n_keep;
            const llama_pos win_begin = ub.pos[win_start];
            const llama_pos win_end   = pos_max + 1;
            const std::string reason = llama_position_split::validate_recurrent_window(
                    uint32_t(std::max<llama_pos>(win_begin, 0)),
                    uint32_t(std::max<llama_pos>(win_end, 0)), p);
            if (!reason.empty()) {
                error = reason;
                return false;
            }
        } else if (n_keep > 0) {
            // The whole ubatch is inside the protected window.
            error = "protected recurrent window (1 + n_rs_seq = " + std::to_string(n_keep) +
                    ") crosses P";
            return false;
        }

        // Divide at P: the trailing part keeps the protected window together.
        std::vector<int32_t> head;
        std::vector<int32_t> tail;
        head.reserve(ub.n_tokens);
        tail.reserve(ub.n_tokens);
        for (uint32_t i = 0; i < ub.n_tokens; ++i) {
            if (ub.pos[i] < llama_pos(p)) {
                head.push_back(int32_t(i));
            } else {
                tail.push_back(int32_t(i));
            }
        }
        if (head.empty() || tail.empty()) {
            error = "position split produced an empty half";
            return false;
        }
        out.push_back(llama_ubatch_subset(ub, head));
        out.push_back(llama_ubatch_subset(ub, tail));
    }

    return true;
}

bool llama_memory_hybrid::position_split_ubatches(
        const std::vector<llama_ubatch> & in,
                             uint32_t   p,
                  std::vector<llama_ubatch> & out,
                             std::string & error) const {
    const uint32_t n_keep = mem_recr != nullptr && mem_recr->n_rs_seq > 0
        ? mem_recr->n_rs_seq + 1 : 0;
    return llama_position_split::divide_ubatches_at_p(in, p, n_keep, out, error);
}

llama_memory_context_ptr llama_memory_hybrid::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (mem_attn->get_kv_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = mem_recr->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // Position split (plan §3.2): enforce the boundary P before anything is
        // prepared, so a rejection cannot leave partial state behind.
        if (position_split_p > 0) {
            std::vector<llama_ubatch> split;
            std::string split_error;
            if (!position_split_ubatches(ubatches, position_split_p, split, split_error)) {
                LLAMA_LOG_ERROR("%s: position split rejected before prepare: %s\n",
                                __func__, split_error.c_str());
                return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
            }
            ubatches = std::move(split);
        }

        // prepare the recurrent batches first
        if (!mem_recr->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto ctx_attn = mem_attn->init_kv_batch(ubatches);
        if (!ctx_attn || llama_memory_status_is_fail(ctx_attn->get_status())) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        return std::make_unique<llama_memory_hybrid_context>(
                this, std::move(ctx_attn), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid::init_full() {
    return std::make_unique<llama_memory_hybrid_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_context>(this, lctx, optimize);
}

bool llama_memory_hybrid::supports_prefill_migration() const {
    return mem_attn->supports_prefill_migration();
}

bool llama_memory_hybrid::handoff_prefill_migration(bool to_remote) {
    return mem_attn->handoff_prefill_migration(to_remote);
}

void llama_memory_hybrid::release_prefill_migration_inactive_buffers() noexcept {
    mem_attn->release_prefill_migration_inactive_buffers();
}

bool llama_memory_hybrid::drain_prefill_migration() {
    return mem_attn->drain_prefill_migration();
}

bool llama_memory_hybrid::get_can_shift() const {
    // Shifting is trivially supported for recurrent
    return mem_attn->get_can_shift();
}

llama_memory_i::seq_rm_capability llama_memory_hybrid::get_seq_rm_capability() const {
    return llama_memory_seq_rm_capability_all({ mem_attn.get(), mem_recr.get() });
}

void llama_memory_hybrid::clear(bool data) {
    mem_attn->clear(data);
    mem_recr->clear(data);
}

bool llama_memory_hybrid::can_seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const {
    return mem_recr->can_seq_rm(seq_id, p0, p1) &&
           mem_attn->can_seq_rm(seq_id, p0, p1);
}

bool llama_memory_hybrid::seq_rm_plan(
        llama_seq_id seq_id, llama_pos p0, llama_pos p1,
        llama_pos & planned_p0, llama_pos & planned_p1) const {
    return llama_memory_seq_rm_plan_all(
            seq_id, p0, p1, { mem_attn.get(), mem_recr.get() }, planned_p0, planned_p1);
}

bool llama_memory_hybrid::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (!can_seq_rm(seq_id, p0, p1)) {
        return false;
    }

    // Try removing from the recurrent cache first since it may fail. If it does
    // fail, the cache will not have been mutated.
    if (!mem_recr->seq_rm(seq_id, p0, p1)) {
        return false;
    }
    return mem_attn->seq_rm(seq_id, p0, p1);
}

bool llama_memory_hybrid::seq_rm_cell(llama_seq_id seq_id, uint32_t cell_idx) {
    return mem_attn->seq_rm_cell(seq_id, cell_idx);
}

int llama_memory_hybrid::cells_at_pos(llama_seq_id seq_id, llama_pos pos, uint32_t * cell_indices, int n_max) {
    return mem_attn->cells_at_pos(seq_id, pos, cell_indices, n_max);
}

void llama_memory_hybrid::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    mem_attn->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    mem_recr->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_hybrid::seq_keep(llama_seq_id seq_id) {
    mem_attn->seq_keep(seq_id);
    mem_recr->seq_keep(seq_id);
}

void llama_memory_hybrid::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    mem_attn->seq_add(seq_id, p0, p1, shift);
    mem_recr->seq_add(seq_id, p0, p1, shift);
}

void llama_memory_hybrid::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    mem_attn->seq_div(seq_id, p0, p1, d);
    mem_recr->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_hybrid::seq_pos_min(llama_seq_id seq_id) const {
    // the min of the total cache is the max of the two caches' min values
    return std::max(mem_attn->seq_pos_min(seq_id), mem_recr->seq_pos_min(seq_id));
}

llama_pos llama_memory_hybrid::seq_pos_max(llama_seq_id seq_id) const {
    // the max of the total cache is the min of the two caches' max values
    return std::min(mem_attn->seq_pos_max(seq_id), mem_recr->seq_pos_max(seq_id));
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = mem_attn->memory_breakdown();
    for (const auto & buft_size : mem_recr->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    return mb;
}

llama_kv_memory_stats llama_memory_hybrid::kv_memory_stats() const {
    return mem_attn->kv_memory_stats();
}

ggml_type llama_memory_hybrid::get_kv_tail_type() const {
    return mem_attn->get_kv_tail_type();
}

uint32_t llama_memory_hybrid::get_kv_tail_group_count() const {
    return mem_attn->get_kv_tail_group_count();
}

bool llama_memory_hybrid::get_kv_tail_coverage(
        uint32_t group_index, llama_seq_id seq_id, llama_kv_tail_coverage_info & out) const {
    return mem_attn->get_kv_tail_coverage(group_index, seq_id, out);
}

void llama_memory_hybrid::reset_kv_tail_planner_timing() {
    mem_attn->reset_kv_tail_planner_timing();
}

uint64_t llama_memory_hybrid::get_kv_tail_planner_timing_ns() const {
    return mem_attn->get_kv_tail_planner_timing_ns();
}

bool llama_memory_hybrid::requires_state_for_partial_restore() const {
    return mem_attn->requires_state_for_partial_restore() ||
           mem_recr->requires_state_for_partial_restore();
}

bool llama_memory_hybrid::state_seq_can_save(llama_seq_id seq_id) const {
    return mem_attn->state_seq_can_save(seq_id) &&
           mem_recr->state_seq_can_save(seq_id);
}

bool llama_memory_hybrid::state_seq_can_restore(llama_seq_id seq_id) const {
    return mem_attn->state_seq_can_restore(seq_id) &&
           mem_recr->state_seq_can_restore(seq_id);
}

bool llama_memory_hybrid::state_seq_can_save(
        llama_seq_id seq_id, llama_state_seq_flags flags) const {
    return mem_attn->state_seq_can_save(seq_id, flags) &&
           mem_recr->state_seq_can_save(seq_id, flags);
}

bool llama_memory_hybrid::state_seq_can_restore(
        llama_seq_id seq_id, llama_state_seq_flags flags) const {
    return mem_attn->state_seq_can_restore(seq_id, flags) &&
           mem_recr->state_seq_can_restore(seq_id, flags);
}

bool llama_memory_hybrid::state_streaming_restore_supported() const {
    return mem_attn->state_streaming_restore_supported() &&
           mem_recr->state_streaming_restore_supported();
}

bool llama_memory_hybrid::state_parse_q4(
        llama_state_q4_source & src,
        const llama_hparams & hparams,
        llama_state_q4_info & info,
        std::string & error) {
    if (!mem_attn->state_parse_q4(src, hparams, info, error)) {
        return false;
    }
    // The recurrent/conv state follows the attention state and is preserved
    // verbatim: it is not part of the quantized attention representation.
    const uint64_t recr_begin = src.tell();
    const uint64_t recr_end = src.size();
    if (recr_end <= recr_begin) {
        error = "source state is missing its recurrent state section";
        return false;
    }
    try {
        // The recurrent state is one logical cell per sequence, positioned at
        // the end of the attention prefix.
        uint32_t cell_count = 0;
        src.read_raw(&cell_count, sizeof(cell_count));
        if (cell_count != 1) {
            error = "source recurrent state does not hold exactly one sequence";
            return false;
        }
        int32_t pos = 0;
        int32_t seq_id = 0;
        src.read_raw(&pos, sizeof(pos));
        src.read_raw(&seq_id, sizeof(seq_id));
        if (seq_id != 0 || pos != int32_t(info.n_tokens) - 1) {
            error = "source recurrent state does not match the attention prefix end";
            return false;
        }
        const uint64_t expected = mem_recr->state_data_bytes(cell_count);
        if (recr_end - src.tell() != expected) {
            error = "source recurrent state data size is inconsistent";
            return false;
        }
    } catch (const std::exception & err) {
        error = err.what();
        return false;
    }
    info.recr_offset = recr_begin;
    info.recr_bytes = recr_end - recr_begin;
    return true;
}

size_t llama_memory_hybrid::state_convert_q4(
        llama_state_q4_source & src,
        const llama_state_q4_info & info,
        const char * dst_path,
        std::vector<uint8_t> * out_mem) {
    return mem_attn->state_convert_q4(src, info, dst_path, out_mem);
}

void llama_memory_hybrid::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    const bool include_attn = (flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0 ||
                              mem_attn->requires_state_for_partial_restore();
    if (include_attn) {
        mem_attn->state_write(io, seq_id, flags);
    }
    mem_recr->state_write(io, seq_id, flags);
}

void llama_memory_hybrid::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    const bool include_attn = (flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0 ||
                              mem_attn->requires_state_for_partial_restore();
    if (include_attn) {
        mem_attn->state_read(io, seq_id, flags);
    }
    mem_recr->state_read(io, seq_id, flags);
}

llama_kv_cache * llama_memory_hybrid::get_mem_attn() const {
    return dynamic_cast<llama_kv_cache *>(mem_attn.get());
}

llama_memory_recurrent * llama_memory_hybrid::get_mem_recr() const {
    return mem_recr.get();
}

llama_memory_hybrid_context::llama_memory_hybrid_context(llama_memory_status status) : status(status) {}

llama_memory_hybrid_context::llama_memory_hybrid_context(llama_memory_hybrid * mem) :
    ctx_attn(mem->get_mem_attn_base()->init_full()),
    ctx_recr(mem->get_mem_recr()->init_full()),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_context::llama_memory_hybrid_context(
        llama_memory_hybrid * mem,
              llama_context * lctx,
                       bool   optimize) :
    ctx_attn(mem->get_mem_attn_base()->init_update(lctx, optimize)),
    ctx_recr(mem->get_mem_recr()->init_update(lctx, optimize)),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_context::llama_memory_hybrid_context(
              llama_memory_hybrid * mem,
        llama_memory_context_ptr   ctx_attn_in,
        std::vector<llama_ubatch>   ubatches) :
    ubatches(std::move(ubatches)),
    ctx_attn(std::move(ctx_attn_in)),
    ctx_recr(new llama_memory_recurrent_context(mem->get_mem_recr(), this->ubatches)),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

bool llama_memory_hybrid_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    ctx_attn->next();
    ctx_recr->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_memory_hybrid_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    bool res = true;

    res = res & ctx_attn->apply();
    res = res & ctx_recr->apply();

    return res;
}

void llama_memory_hybrid_context::graph_compute_start() {
    ctx_attn->graph_compute_start();
    ctx_recr->graph_compute_start();
}

void llama_memory_hybrid_context::graph_compute_finish(ggml_status compute_status) {
    ctx_attn->graph_compute_finish(compute_status);
    ctx_recr->graph_compute_finish(compute_status);
}

void llama_memory_hybrid_context::graph_compute_complete(
        ggml_backend_sched_t sched, ggml_status compute_status) {
    ctx_attn->graph_compute_complete(sched, compute_status);
    ctx_recr->graph_compute_complete(sched, compute_status);
}

llama_memory_status llama_memory_hybrid_context::get_status() const {
    return status;
}

const llama_ubatch & llama_memory_hybrid_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return ubatches[i_next];
}

const llama_kv_cache_context * llama_memory_hybrid_context::get_attn() const {
    auto * result = dynamic_cast<const llama_kv_cache_context *>(ctx_attn.get());
    GGML_ASSERT(result != nullptr);
    return result;
}

const llama_memory_context_i * llama_memory_hybrid_context::get_attn_memory_context() const {
    return ctx_attn.get();
}

const llama_kv_cache_context * llama_memory_hybrid_context::get_attn_kv_context() const {
    auto * result = dynamic_cast<const llama_kv_cache_context *>(ctx_attn.get());
    GGML_ASSERT(result != nullptr);
    return result;
}

const llama_memory_recurrent_context * llama_memory_hybrid_context::get_recr() const {
    return static_cast<const llama_memory_recurrent_context *>(ctx_recr.get());
}
