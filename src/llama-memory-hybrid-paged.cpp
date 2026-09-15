#include "llama-memory-hybrid-paged.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-context.h"

//
// llama_memory_hybrid_paged
//

llama_memory_hybrid_paged::llama_memory_hybrid_paged(
        const llama_model & model,
                            /* attn (paged) */
                ggml_type   type_kv,
                 uint32_t   block_size,
                 uint32_t   n_gpu_blocks,
                 uint32_t   initial_gpu_blocks,
                 uint32_t   growth_gpu_blocks,
                 uint32_t   n_cpu_blocks,
                   float    watermark,
                 uint32_t   n_ubatch,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                            /* backends */
    const std::vector<ggml_backend_t> & layer_backends,
    const std::vector<ggml_backend_t> & kv_backends,
             ggml_backend_t backend_cpu,
                     bool   dynamic_spill,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr) :
    hparams(model.hparams),
    mem_attn(new llama_kv_cache_paged(
        hparams.n_embd_head_v(),
        hparams.n_head(),
        hparams.n_head_kv(),
        block_size,
        hparams.n_layer(),
        n_ubatch,
        n_seq_max,
        filter_attn == nullptr ?
            [&](int32_t il) { return !hparams.is_recr(il); }
            : filter_attn
    )),
    mem_recr(new llama_memory_recurrent(
        model,
        type_r,
        type_s,
        offload,
        rs_size,
        n_seq_max,
        n_rs_seq,
        filter_recr == nullptr ?
            [&](int32_t il) { return hparams.is_recr(il); }
            : filter_recr
    )) {
    mem_attn->init(
        layer_backends, kv_backends, backend_cpu, type_kv,
        n_gpu_blocks, n_cpu_blocks, watermark, initial_gpu_blocks, growth_gpu_blocks, dynamic_spill);
}

llama_memory_context_ptr llama_memory_hybrid_paged::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
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
                // paged attention requires a single ubatch (n_ubatch == n_batch)
                const bool unified = true;

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

        // prepare the recurrent batches first
        if (!mem_recr->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_paged_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        if (!mem_attn->prepare_batch(balloc.get_batch())) {
            LLAMA_LOG_WARN("%s: failed to prepare paged attention batch\n", __func__);
            return std::make_unique<llama_memory_hybrid_paged_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        return std::make_unique<llama_memory_hybrid_paged_context>(
                this, std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_paged_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_paged::init_full() {
    return std::make_unique<llama_memory_hybrid_paged_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_paged::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_paged_context>(this, lctx, optimize);
}

bool llama_memory_hybrid_paged::get_can_shift() const {
    // Shifting is trivially supported for recurrent
    return mem_attn->get_can_shift();
}

bool llama_memory_hybrid_paged::reserve(uint32_t n_tokens) {
    return mem_attn->reserve(n_tokens);
}

void llama_memory_hybrid_paged::clear(bool data) {
    mem_attn->clear(data);
    mem_recr->clear(data);
}

bool llama_memory_hybrid_paged::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // Try removing from the recurrent cache first since it may fail. If it does
    // fail, the cache will not have been mutated.
    if (!mem_recr->seq_rm(seq_id, p0, p1)) {
        return false;
    }
    return mem_attn->seq_rm(seq_id, p0, p1);
}

bool llama_memory_hybrid_paged::seq_rm_cell(llama_seq_id seq_id, uint32_t cell_idx) {
    return mem_attn->seq_rm_cell(seq_id, cell_idx);
}

int llama_memory_hybrid_paged::cells_at_pos(llama_seq_id seq_id, llama_pos pos, uint32_t * cell_indices, int n_max) {
    return mem_attn->cells_at_pos(seq_id, pos, cell_indices, n_max);
}

void llama_memory_hybrid_paged::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    mem_attn->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    mem_recr->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_hybrid_paged::seq_keep(llama_seq_id seq_id) {
    mem_attn->seq_keep(seq_id);
    mem_recr->seq_keep(seq_id);
}

void llama_memory_hybrid_paged::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    mem_attn->seq_add(seq_id, p0, p1, shift);
    mem_recr->seq_add(seq_id, p0, p1, shift);
}

void llama_memory_hybrid_paged::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    mem_attn->seq_div(seq_id, p0, p1, d);
    mem_recr->seq_div(seq_id, p0, p1, d);
}

void llama_memory_hybrid_paged::set_snapkv_prefill_end(llama_seq_id seq_id, llama_pos prefill_end) {
    mem_attn->set_snapkv_prefill_end(seq_id, prefill_end);
}

llama_pos llama_memory_hybrid_paged::seq_pos_min(llama_seq_id seq_id) const {
    // the min of the total cache is the max of the two caches' min values
    return std::max(mem_attn->seq_pos_min(seq_id), mem_recr->seq_pos_min(seq_id));
}

llama_pos llama_memory_hybrid_paged::seq_pos_max(llama_seq_id seq_id) const {
    // the max of the total cache is the min of the two caches' max values
    return std::min(mem_attn->seq_pos_max(seq_id), mem_recr->seq_pos_max(seq_id));
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_paged::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = mem_attn->memory_breakdown();
    for (const auto & buft_size : mem_recr->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    return mb;
}

void llama_memory_hybrid_paged::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        mem_attn->state_write(io, seq_id, flags);
    }
    mem_recr->state_write(io, seq_id, flags);
}

void llama_memory_hybrid_paged::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        mem_attn->state_read(io, seq_id, flags);
    }
    mem_recr->state_read(io, seq_id, flags);
}

llama_kv_cache_paged * llama_memory_hybrid_paged::get_mem_attn() const {
    return mem_attn.get();
}

llama_memory_recurrent * llama_memory_hybrid_paged::get_mem_recr() const {
    return mem_recr.get();
}

llama_memory_hybrid_paged_context::llama_memory_hybrid_paged_context(llama_memory_status status) : status(status) {}

llama_memory_hybrid_paged_context::llama_memory_hybrid_paged_context(llama_memory_hybrid_paged * mem) :
    ctx_attn(mem->get_mem_attn()->init_full()),
    ctx_recr(mem->get_mem_recr()->init_full()),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_paged_context::llama_memory_hybrid_paged_context(
        llama_memory_hybrid_paged * mem,
              llama_context * lctx,
                       bool   optimize) :
    ctx_attn(mem->get_mem_attn()->init_update(lctx, optimize)),
    ctx_recr(mem->get_mem_recr()->init_update(lctx, optimize)),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_paged_context::llama_memory_hybrid_paged_context(
              llama_memory_hybrid_paged * mem,
        std::vector<llama_ubatch>         ubatches) :
    ubatches(std::move(ubatches)),
    ctx_attn(new llama_kv_cache_paged_context(mem->get_mem_attn(), this->ubatches)),
    ctx_recr(new llama_memory_recurrent_context(mem->get_mem_recr(), this->ubatches)),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
    // The server path builds metadata from each ubatch. The scheduler path is
    // still available for examples/paged and supplies its own arrays.
    auto * paged_ctx = static_cast<llama_kv_cache_paged_context *>(ctx_attn.get());
    GGML_ASSERT(!this->ubatches.empty());
    paged_ctx->set_batch_data(this->ubatches.front());
}

bool llama_memory_hybrid_paged_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    ctx_attn->next();
    ctx_recr->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_memory_hybrid_paged_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    bool res = true;

    res = res & ctx_attn->apply();
    res = res & ctx_recr->apply();

    return res;
}

llama_memory_status llama_memory_hybrid_paged_context::get_status() const {
    return status;
}

const llama_ubatch & llama_memory_hybrid_paged_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return ubatches[i_next];
}

const llama_kv_cache_paged_context * llama_memory_hybrid_paged_context::get_attn() const {
    return static_cast<const llama_kv_cache_paged_context *>(ctx_attn.get());
}

const llama_memory_recurrent_context * llama_memory_hybrid_paged_context::get_recr() const {
    return static_cast<const llama_memory_recurrent_context *>(ctx_recr.get());
}
