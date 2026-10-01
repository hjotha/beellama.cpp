// Server glue implementation: one-way PURE KVarN -> MIXED prefix reuse.
// File-scope helpers only. Source restore uses the authoritative legacy
// parsers (hybrid memory state_read for KV+RS; recurrent state_write/read
// for the candidate RS) through minimal byte-view IO adapters mirroring the
// reviewed llama_io_read_host/write_host transaction semantics.

#include "server-mixed-kv-handoff.h"
#include "server-task.h"
#include "server-common.h"

#include "common.h"
#include "speculative.h"
#include "llama.h"

#include "src/llama-context.h"
#include "src/llama-model.h"
#include "src/llama-memory.h"
#include "src/llama-memory-hybrid.h"
#include "src/llama-memory-recurrent.h"
#include "src/llama-kv-cache.h"
#include "src/llama-kv-cache-kvarn.h"
#include "src/llama-kv-cache-tail.h"
#include "src/llama-io.h"
#include "src/llama-kv-mixed-handoff.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <fstream>
#include <limits>
#include <sstream>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace server_mixed_kv_handoff {

namespace {

// RAM sequence-state header: [u32 io_magic][llama_seq_id]. The body follows
// immediately (memory->state_write stream). The magic value is the same
// constant the context uses for llama_state_seq_get_data.
constexpr uint32_t SERVER_MIXED_HANDOFF_RAM_MAGIC = 0xaf143cd8;
constexpr size_t   SERVER_MIXED_HANDOFF_RAM_HEADER = sizeof(uint32_t) + sizeof(llama_seq_id);

// Minimal byte-view reader implementing the llama_io_read_i transaction
// contract (mirrors the reviewed llama_io_read_host semantics; nothing is
// written until commit()).
class byte_reader final : public llama_io_read_i {
public:
    byte_reader(const uint8_t * data, size_t size) : data_(data), size_(size) {}
    ~byte_reader() override { cancel(); }

    void read(void * dst, size_t size) override {
        bounds(size);
        std::memcpy(dst, data_ + pos_, size);
        pos_ += size;
    }
    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        check_tensor(tensor, offset, size); // bounds only; borrowed span, no copy
        bounds(size);
        infos_.push_back({ tensor, data_ + pos_, {}, size, offset, false });
        pos_ += size;
    }
    void stage_tensor_set(ggml_tensor * tensor, const void * src, size_t offset, size_t size) override {
        check_tensor(tensor, offset, size); // bounds
        if (size > MAX_FIXUP_BYTES || fixup_bytes_ > MAX_FIXUP_BYTES - size) {
            throw std::runtime_error("server mixed handoff: owned staging budget exceeded");
        }
        fixup_bytes_ += size; // OWNED copies only; borrowed spans are zero-copy
        info info = { tensor, nullptr, {}, size, offset, false };
        info.owned.resize(size);
        std::memcpy(info.owned.data(), src, size);
        infos_.push_back(std::move(info));
    }
    void stage_tensor_clear(ggml_tensor * tensor, size_t offset, size_t size) override {
        check_tensor(tensor, offset, size);
        infos_.push_back({ tensor, nullptr, {}, size, offset, true });
    }
    void on_commit(std::function<void()> callback) override {
        callbacks_.push_back(std::move(callback));
    }
    void commit() override {
        for (const auto & info : infos_) {
            if (info.clear) {
                ggml_backend_tensor_memset(info.tensor, 0, info.offset, info.size);
            } else {
                const void * src = info.owned.empty() ? info.ptr : info.owned.data();
                ggml_backend_tensor_set(info.tensor, src, info.offset, info.size);
            }
        }
        for (auto & callback : callbacks_) {
            callback();
        }
        cancel();
    }
    void cancel() override {
        infos_.clear();
        callbacks_.clear();
    }
    size_t n_bytes() override { return pos_; }

private:
    struct info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        std::vector<uint8_t> owned;
        size_t size;
        size_t offset;
        bool clear;
    };
    void check_tensor(ggml_tensor * tensor, size_t offset, size_t size) {
        if (tensor == nullptr || tensor->buffer == nullptr || size == 0) {
            throw std::runtime_error("server mixed handoff: invalid tensor write target");
        }
        if (offset > ggml_nbytes(tensor) || size > ggml_nbytes(tensor) - offset) {
            throw std::runtime_error("server mixed handoff: tensor write out of bounds");
        }
    }
    void bounds(size_t size) const {
        if (size > size_ - pos_) {
            throw std::runtime_error("server mixed handoff: byte reader out of bounds");
        }
    }
    static constexpr size_t MAX_FIXUP_BYTES = 64u << 20; // total staged fixups
    const uint8_t * data_;
    size_t size_;
    size_t pos_ = 0;
    size_t fixup_bytes_ = 0;
    std::vector<info> infos_;
    std::vector<std::function<void()>> callbacks_;
};

// Minimal byte writer implementing llama_io_write_i (tensor bytes fetched via
// ggml_backend_tensor_get at write_tensor time). Bounded: the budget is set
// at construction and every write/insert is checked BEFORE mutating.
class byte_writer final : public llama_io_write_i {
public:
    explicit byte_writer(uint64_t budget) : budget_(budget) {}
    void write(const void * src, size_t size) override {
        check_add(size);
        const uint8_t * p = static_cast<const uint8_t *>(src);
        bytes_.insert(bytes_.end(), p, p + size);
    }
    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (tensor == nullptr || tensor->buffer == nullptr || size == 0 ||
                offset > ggml_nbytes(tensor) || size > ggml_nbytes(tensor) - offset) {
            throw std::runtime_error("server mixed handoff: invalid tensor write source");
        }
        check_add(size);
        const size_t old = bytes_.size();
        bytes_.resize(old + size);
        ggml_backend_tensor_get(tensor, bytes_.data() + old, offset, size);
    }
    size_t n_bytes() override { return bytes_.size(); }
    const std::vector<uint8_t> & bytes() const { return bytes_; }
    std::vector<uint8_t> take() { return std::move(bytes_); }

private:
    void check_add(size_t size) const {
        if (uint64_t(size) > budget_ || bytes_.size() > size_t(budget_ - uint64_t(size))) {
            throw std::runtime_error("server mixed handoff: writer budget exceeded");
        }
    }
    const uint64_t budget_;
    std::vector<uint8_t> bytes_;
};

// Positions the reader past the RAM sequence-state header, validating the
// magic. Returns false (and sets error) for a malformed buffer.
bool skip_ram_header(byte_reader & reader, size_t total_size, std::string & error) {
    if (total_size < SERVER_MIXED_HANDOFF_RAM_HEADER) {
        error = "snapshot smaller than the RAM state header";
        return false;
    }
    uint32_t magic = 0;
    reader.read(&magic, sizeof(magic));
    if (magic != SERVER_MIXED_HANDOFF_RAM_MAGIC) {
        error = "snapshot RAM state magic mismatch";
        return false;
    }
    llama_seq_id saved_seq_id = 0;
    reader.read(&saved_seq_id, sizeof(saved_seq_id));
    if (saved_seq_id < 0) {
        error = "snapshot RAM state carries a negative sequence id";
        return false;
    }
    return true;
}

} // namespace

source_capture capture_source(llama_context * ctx) {
    source_capture out;
    if (ctx == nullptr) {
        return out;
    }
    llama_memory_i * memory = llama_get_memory(ctx);
    if (memory == nullptr) {
        return out;
    }
    auto * hybrid = dynamic_cast<llama_memory_hybrid *>(memory);
    auto * kvarn = hybrid
            ? dynamic_cast<llama_kv_cache_kvarn *>(hybrid->get_mem_attn_base()) : nullptr;
    if (kvarn == nullptr || kvarn->has_standard_cache()) {
        return out; // only a PURE KVarN source is handoff-eligible
    }
    const auto allocation_bytes = [](const llama_memory_i * mem) {
        uint64_t sum = 0;
        for (const auto & item : mem->memory_breakdown()) {
            if (item.second > std::numeric_limits<uint64_t>::max() - sum) {
                throw std::overflow_error("source memory allocation size overflow");
            }
            sum += item.second;
        }
        return sum;
    };
    out.memory_bytes = allocation_bytes(memory);
    out.recurrent_bytes = allocation_bytes(hybrid->get_mem_recr());
    out.valid = true;
    out.cparams = ctx->get_cparams();
    // Deterministic memory params: mem_other is nullptr and swa_full false
    // (the handoff source is single-stream non-SWA; swa_full would reserve a
    // full-size SWA cache that is never read here).
    out.mem_params.mem_other = nullptr;
    out.mem_params.swa_full = false;
    out.mem_params.ctx_type = out.cparams.ctx_type;
    out.mem_params.kvarn = out.cparams.kvarn;
    out.mem_params.kv_tail_tokens = out.cparams.kv_tail_tokens;
    out.mem_params.kv_tail_tokens_swa = out.cparams.kv_tail_tokens_swa;
    out.mem_params.kv_tail_tokens_requested = out.cparams.kv_tail_tokens_requested;
    out.mem_params.kv_tail_tokens_swa_requested = out.cparams.kv_tail_tokens_swa_requested;
    out.mem_params.kv_tail_native_exact = out.cparams.kv_tail_native_exact;
    out.mem_params.kv_tail_native_exact_swa = out.cparams.kv_tail_native_exact_swa;
    out.mem_params.kv_tail_rollback_tokens = out.cparams.kv_tail_rollback_tokens;
    out.mem_params.kv_tail_type = out.cparams.kv_tail_type;
    out.mem_params.type_k = GGML_TYPE_F16;
    out.mem_params.type_v = GGML_TYPE_F16;
    out.n_ctx_seq = llama_n_ctx_seq(ctx);
    out.exact_tail_tokens = kvarn->get_exact_tail_tokens();
    out.has_cell_ext = kvarn->get_metadata_cache()->has_cell_ext();
    out.kvarn_layers = kvarn->kvarn_layer_layout();
    out.source_layout = common_prompt_cache_layout(ctx);
    out.source_model_instance = llama_model_mtp_weights_get_info(
            llama_get_model(ctx)).model_instance;
    return out;
}

namespace {

// Exact-tail overlay source: reads ORIGINAL-domain rows from the source
// exact-tail tensors (slot -> token mapping from the authoritative snapshot).
struct tail_source {
    ggml_tensor * k_tail = nullptr;
    ggml_tensor * v_tail = nullptr;
    const std::unordered_map<uint32_t, int32_t> * slot_of_token = nullptr;
    uint32_t head_dim = 0;
    uint32_t n_heads = 0;
    bool value = false;
    bool failed = false; // sticky: a wrong/missing expected row fails the handoff
};

bool tail_read(void * user, uint32_t token, uint32_t head_dim,
        uint32_t n_heads, float * rows) {
    auto * t = static_cast<tail_source *>(user);
    if (t == nullptr) {
        return false;
    }
    if (t->slot_of_token == nullptr ||
            head_dim != t->head_dim || n_heads != t->n_heads) {
        t->failed = true;
        return false;
    }
    const auto it = t->slot_of_token->find(token);
    if (it == t->slot_of_token->end()) {
        // Full expected tail coverage was validated before conversion. Tokens
        // outside that window use the materializer's compressed/stage path.
        return false;
    }
    if (it->second < 0) {
        t->failed = true;
        return false;
    }
    ggml_tensor * tail = t->value ? t->v_tail : t->k_tail;
    if (tail == nullptr) {
        t->failed = true;
        return false;
    }
    if (tail->ne[0] != int64_t(head_dim) * int64_t(n_heads)) {
        t->failed = true; // row dimension must match the overlay geometry
        return false;
    }
    const size_t row_bytes = ggml_row_size(tail->type, tail->ne[0]);
    if (tail->type != GGML_TYPE_F16 && tail->type != GGML_TYPE_F32) {
        t->failed = true; // only F16/F32 exact tail rows are supported
        return false;
    }
    if (size_t(it->second) * row_bytes + row_bytes > ggml_nbytes(tail)) {
        t->failed = true;
        return false;
    }
    std::vector<uint8_t> row(row_bytes);
    ggml_backend_tensor_get(tail, row.data(), size_t(it->second) * row_bytes, row_bytes);
    if (tail->type == GGML_TYPE_F16) {
        const ggml_fp16_t * p = reinterpret_cast<const ggml_fp16_t *>(row.data());
        for (size_t i = 0; i < size_t(tail->ne[0]); ++i) {
            rows[i] = ggml_fp16_to_fp32(p[i]);
        }
    } else {
        std::memcpy(rows, row.data(), size_t(tail->ne[0]) * sizeof(float));
    }
    return true;
}

// Candidate cleanup: armed BEFORE the first candidate mutation; every
// failure/exception path clears the target + draft + spec. Disarmed only
// after the transfer completed and the positions were verified.
struct candidate_guard {
    llama_context * tgt = nullptr;
    llama_context * dft = nullptr;
    common_speculative * spec = nullptr;
    llama_seq_id seq = 0;
    bool disarmed = false;
    ~candidate_guard() {
        if (disarmed) {
            return;
        }
        if (tgt) {
            llama_memory_seq_rm(llama_get_memory(tgt), seq, -1, -1);
        }
        if (dft) {
            llama_memory_seq_rm(llama_get_memory(dft), seq, -1, -1);
        }
        if (spec) {
            common_speculative_set_state(spec, seq, {});
        }
    }
};

} // namespace

result try_handoff_impl(const input & in, result & out,
        const std::function<result(const char *)> & fail,
        candidate_guard & cguard);

result try_handoff(const input & in) {
    using namespace llama_kv_handoff;
    result out;
    const auto fail = [&out](const char * reason) -> result {
        out.committed = false;
        out.reason = reason;
        out.marker = handoff_marker_failed(reason);
        return out;
    };
    // Candidate cleanup armed BEFORE any candidate mutation and active for
    // EVERY path: exceptions from digest/JSON/parsers cannot escape.
    candidate_guard cguard;
    try {
        return try_handoff_impl(in, out, fail, cguard);
    } catch (const std::exception & error) {
        return fail(("handoff exception: " + std::string(error.what())).c_str());
    }
}

result try_handoff_impl(const input & in, result & out,
        const std::function<result(const char *)> & fail,
        candidate_guard & cguard) {
    using namespace llama_kv_handoff;

    uint64_t rs_bytes_ = 0; // recurrent state transfer bytes (diagnostic)
    if (in.ctx_tgt == nullptr || in.model == nullptr || in.source == nullptr ||
            in.state == nullptr || in.tokens_new == nullptr ||
            !in.source->valid) {
        return fail("invalid handoff input");
    }
    if (in.has_mtmd) {
        return fail("MTMD token mapping is unsupported by the handoff");
    }
    auto * hybrid = dynamic_cast<llama_memory_hybrid *>(llama_get_memory(in.ctx_tgt));
    auto * kvarn = hybrid
            ? dynamic_cast<llama_kv_cache_kvarn *>(hybrid->get_mem_attn_base()) : nullptr;
    if (kvarn == nullptr || !kvarn->has_standard_cache()) {
        return fail("destination is not a mixed KVarN+Qx cache");
    }
    if (in.state->prompt.tokens.empty() || in.tokens_new->empty()) {
        return out; // no source (cold path, no marker)
    }
    const llama_mtp_weights_info mtp_info = llama_model_mtp_weights_get_info(in.model);

    // Source identity: model + instance + integrity + quarantine + geometry
    // matching the captured source (not any unknown layout).
    if (in.state->model != in.model ||
            in.state->model_instance != mtp_info.model_instance ||
            in.state->model_instance != in.source->source_model_instance ||
            in.state->quarantined || in.state->digest() != in.state->checksum) {
        return out; // not an eligible source (cold path)
    }
    const common_json layout = common_json::parse(in.state->layout_tgt);
    if (!layout.is_object() || layout.value("kv_layout_known", true) ||
            layout.value("mixed_kv", false)) {
        return out; // only PURE KVarN source snapshots are eligible
    }
    // The snapshot layout must match the CAPTURED source exactly (same
    // context instance and capacity), never an arbitrary unknown layout.
    if (in.source->source_layout != in.state->layout_tgt) {
        return out;
    }
    if (in.state->pos_tgt < 0 ||
            in.state->pos_tgt >= (llama_pos) in.source->n_ctx_seq) {
        return out;
    }

    const size_t lcp = in.state->prompt.tokens.get_common_prefix(*in.tokens_new);
    if (lcp == 0) {
        out.reason = "no common prefix with the source snapshot";
        out.marker = handoff_marker_no_source();
        return out;
    }
    std::vector<handoff_prefix_candidate> candidates;
    for (const auto & c : in.state->prompt.checkpoints) {
        candidates.push_back({ uint32_t(std::max<int64_t>(0, c->n_tokens)), c->pos_max,
                c->host_only() });
    }
    const uint32_t selected = handoff_select_prefix(candidates.data(), candidates.size(),
            uint32_t(std::min<size_t>(lcp, UINT32_MAX)));
    const common_prompt_checkpoint * checkpoint = nullptr;
    if (selected > 0) {
        for (const auto & c : in.state->prompt.checkpoints) {
            if (c->host_only() && c->n_tokens == (int64_t) selected &&
                    c->pos_max == (int64_t) selected - 1 &&
                    (c->flags_tgt & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) == 0 &&
                    c->pos_max <= in.state->pos_tgt) {
                checkpoint = c.get();
                break;
            }
        }
    }
    if (checkpoint == nullptr || checkpoint->data_tgt.empty()) {
        out.reason = "no compatible host checkpoint within the LCP";
        out.marker = handoff_marker_no_source();
        return out;
    }

    // The source snapshot and candidate GPU buffers are already resident.
    // Reserve RAM for the CPU source allocation, RS serialization, parser
    // fixups and conversion workspaces before allocating a second cache.
    // Linux MemAvailable includes reclaimable pages and reflects Radeon UMA
    // allocations; never add GTT again as if it were separate host memory.
#ifdef __linux__
    uint64_t available = 0;
    std::ifstream meminfo("/proc/meminfo");
    std::string memline;
    while (std::getline(meminfo, memline)) {
        if (memline.compare(0, 13, "MemAvailable:") == 0) {
            std::istringstream value(memline.substr(13));
            uint64_t kib = 0;
            if (value >> kib && kib <= std::numeric_limits<uint64_t>::max() / 1024) {
                available = kib * 1024;
            }
            break;
        }
    }
    constexpr uint64_t working_reserve = 256ull << 20;
    const uint64_t rs_reserve = handoff_rs_budget(in.source->recurrent_bytes, 1ull << 30);
    if (available <= working_reserve || rs_reserve > available - working_reserve ||
            in.source->memory_bytes > available - working_reserve - rs_reserve) {
        return fail("insufficient available host RAM for CPU source and recurrent transfer");
    }
#endif

    // RAII CPU backend + source memory (offload=false, no llama_context).
    struct cpu_raii {
        ggml_backend_t backend = nullptr;
        std::unique_ptr<llama_memory_i> memory;
        ~cpu_raii() {
            memory.reset();
            if (backend) {
                ggml_backend_free(backend);
            }
        }
    } cpu;
    cpu.backend = ggml_backend_cpu_init();
    if (cpu.backend == nullptr) {
        return fail("CPU backend init failed");
    }
    llama_cparams source_cparams = in.source->cparams;
    source_cparams.offload_kqv = false;
    source_cparams.offload_rs = false;
    source_cparams.remote_attn_enabled = false;
    source_cparams.remote_attn_layers = 0;
    source_cparams.remote_attn_cache_type_k = GGML_TYPE_COUNT;
    source_cparams.remote_attn_cache_type_v = GGML_TYPE_COUNT;
    source_cparams.local_attn_backend = nullptr;
    source_cparams.local_attn_migration = false;
    source_cparams.local_attn_migration_backend = nullptr;
    source_cparams.local_attn_prefill_backend = nullptr;
    source_cparams.ctx_other = nullptr;
    source_cparams.n_ctx_seq = in.source->n_ctx_seq;
    llama_memory_params source_mem_params = in.source->mem_params;
    source_mem_params.remote_attn_cache_type_k = GGML_TYPE_COUNT;
    source_mem_params.remote_attn_cache_type_v = GGML_TYPE_COUNT;
    try {
        const size_t n_layers = in.model->hparams.n_layer();
        const std::vector<ggml_backend_t> cpu_backends(n_layers, cpu.backend);
        cpu.memory.reset(in.model->create_memory(
                source_mem_params, source_cparams, cpu_backends, cpu_backends, cpu.backend));
    } catch (const std::exception & error) {
        return fail(("CPU source memory creation failed: " + std::string(error.what())).c_str());
    }
    auto * source_hybrid = dynamic_cast<llama_memory_hybrid *>(cpu.memory.get());
    auto * source_kvarn = source_hybrid
            ? dynamic_cast<llama_kv_cache_kvarn *>(source_hybrid->get_mem_attn_base()) : nullptr;
    if (source_kvarn == nullptr || source_hybrid->get_mem_recr() == nullptr) {
        return fail("CPU source memory is not a hybrid KVarN+RS cache");
    }

    // Phase 1: FULL restore (compressed body + exact tail + RS) from the
    // saved snapshot, then COMMIT. The CPU source must hold the complete
    // body before any PARTIAL checkpoint is applied.
    try {
        byte_reader full_io(in.state->data.main.data(), in.state->data.main.size());
        std::string header_error;
        if (!skip_ram_header(full_io, in.state->data.main.size(), header_error)) {
            return fail(("full source parse: " + header_error).c_str());
        }
        cpu.memory->state_read(full_io, in.seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (full_io.n_bytes() != in.state->data.main.size()) {
            return fail("full source restore did not consume the complete frame");
        }
        full_io.commit();
    } catch (const std::exception & error) {
        return fail(("full source restore failed: " + std::string(error.what())).c_str());
    }
    if (llama_memory_seq_pos_max(cpu.memory.get(), in.seq_id) != in.state->pos_tgt) {
        return fail("full source restore position does not match the snapshot");
    }

    // Phase 2: selected PARTIAL checkpoint (stage window + exact tail + RS),
    // then COMMIT. This overrides the live window with the authoritative
    // checkpoint content at the transferred prefix.
    llama_kv_cache::slot_info_vec_t sinfos;
    try {
        byte_reader partial_io(checkpoint->data_tgt.data(), checkpoint->data_tgt.size());
        std::string header_error;
        if (!skip_ram_header(partial_io, checkpoint->data_tgt.size(), header_error)) {
            return fail(("partial checkpoint parse: " + header_error).c_str());
        }
        source_kvarn->state_read_sinfo(partial_io, in.seq_id,
                LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY, &sinfos, nullptr);
        source_hybrid->get_mem_recr()->state_read(partial_io, in.seq_id,
                LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        // Complete frame consumption must be verified BEFORE commit.
        if (partial_io.n_bytes() != checkpoint->data_tgt.size()) {
            return fail("partial checkpoint did not consume the complete frame");
        }
        partial_io.commit();
    } catch (const std::exception & error) {
        return fail(("partial checkpoint restore failed: " + std::string(error.what())).c_str());
    }
    if (sinfos.empty() || sinfos[0].idxs.empty() || sinfos[0].idxs[0].size() != selected) {
        return fail("partial checkpoint did not restore the selected prefix");
    }

    // sinfos logical ordering: the physical cell of token i must be the cell
    // whose position is i (cells must be dense ordered by token position for
    // this milestone layout); unsupported mappings are rejected explicitly.
    {
        const auto & cells = source_kvarn->get_metadata_cache()->get_cells(in.seq_id);
        for (uint32_t i = 0; i < selected; ++i) {
            const uint32_t cell = uint32_t(sinfos[0].idxs[0][i]);
            if (cell != i || cell >= cells.size() || cells.is_empty(cell) ||
                    cells.pos_get(cell) != (llama_pos) i) {
                return fail("source cell map is not dense ordered by token position");
            }
        }
    }

    // Canonical trim: the PARTIAL checkpoint restores the live window while
    // later attention cells from the FULL restore may remain; remove them so
    // the source prefix is exactly 0..selected-1.
    if (!cpu.memory->seq_rm(in.seq_id, (llama_pos) selected, -1)) {
        return fail("source prefix trim failed");
    }
    if (llama_memory_seq_pos_max(cpu.memory.get(), in.seq_id) != (llama_pos) selected - 1 ||
            llama_memory_seq_pos_max(source_hybrid->get_mem_recr(), in.seq_id) !=
                    (llama_pos) selected - 1) {
        return fail("trimmed source positions (KV/RS) do not match the selected prefix");
    }

    // Tail slot -> token mapping (AFTER the canonical trim): must be COMPLETE
    // for the expected tail window; a missing entry is a hard failure (never
    // requantize the body).
    const uint32_t effective_tail_begin = selected > in.source->exact_tail_tokens
            ? selected - in.source->exact_tail_tokens : 0;
    std::unordered_map<uint32_t, int32_t> tail_slot_of_token;
    try {
        for (const auto & entry : source_kvarn->get_metadata_cache()->state_tail_snapshot(in.seq_id)) {
            if (entry.seq_id == in.seq_id && entry.position >= (llama_pos) effective_tail_begin &&
                    entry.position < (llama_pos) selected) {
                // Snapshot history also contains rollback-only rows. They
                // are not the effective exact attention window at this prefix.
                tail_slot_of_token[uint32_t(entry.position)] = entry.slot;
            }
        }
    } catch (const std::exception & error) {
        return fail(("tail snapshot lookup failed: " + std::string(error.what())).c_str());
    }
    if (in.source->exact_tail_tokens > 0) {
        const uint32_t tail_begin = selected > in.source->exact_tail_tokens
                ? selected - in.source->exact_tail_tokens : 0;
        for (uint32_t pos = tail_begin; pos < selected; ++pos) {
            if (tail_slot_of_token.find(pos) == tail_slot_of_token.end()) {
                return fail("expected exact-tail rows are missing from the source snapshot");
            }
        }
    }

    cguard.tgt = in.ctx_tgt;
    cguard.dft = in.ctx_dft;
    cguard.spec = in.spec;
    cguard.seq = in.seq_id;

    // Candidate logical metadata (private candidate; published only on success).
    try {
        std::vector<llama_kv_cell_ext> exts;
        if (in.source->has_cell_ext) {
            const auto & cells = source_kvarn->get_metadata_cache()->get_cells(in.seq_id);
            exts.reserve(selected);
            for (uint32_t pos = 0; pos < selected; ++pos) {
                exts.push_back(cells.ext_get(sinfos[0].idxs[0][pos]));
            }
        }
        kvarn->get_metadata_cache()->import_sequence_prefix(in.seq_id, selected, exts);
        if (in.source->exact_tail_tokens > 0) {
            const uint32_t tail_begin = selected > in.source->exact_tail_tokens
                    ? selected - in.source->exact_tail_tokens : 0;
            kvarn->get_metadata_cache()->import_sequence_tail(in.seq_id, tail_begin, selected);
        }
    } catch (const std::exception & error) {
        return fail(("candidate metadata import failed: " + std::string(error.what())).c_str());
    }

    // Remote layers from the candidate's authoritative standard layout.
    std::vector<handoff_remote_layer> remote_layers;
    for (const auto & layer : kvarn->standard_layer_layout()) {
        handoff_remote_layer remote;
        remote.il = layer.layer_id;
        remote.qx_type_k = (int32_t) layer.type_k;
        remote.qx_type_v = (int32_t) layer.type_v;
        remote.rotation_k = layer.rotation_k;
        remote.rotation_v = layer.rotation_v;
        remote.v_transposed = layer.v_transposed;
        remote.head_dim_k = layer.head_dim_k;
        remote.head_dim_v = layer.head_dim_v;
        remote.n_head_kv = layer.n_head_kv;
        remote.row_bytes_k = size_t(layer.row_stride_k);
        remote.row_bytes_v = size_t(layer.row_stride_v);
        remote_layers.push_back(remote);
    }

    handoff_source_desc source_desc;
    source_desc.n_tokens = selected;
    source_desc.n_ctx_seq = in.source->n_ctx_seq;
    source_desc.n_seq_max = in.source->cparams.n_seq_max;
    source_desc.kv_unified = in.source->cparams.kv_unified;
    source_desc.kvarn_type = (int32_t) in.source->mem_params.kvarn.type;
    source_desc.key_bits = in.source->mem_params.kvarn.key_bits;
    source_desc.value_bits = in.source->mem_params.kvarn.value_bits;
    source_desc.stage_groups = int32_t(source_kvarn->get_stage_groups());
    source_desc.tail_groups = int32_t(source_kvarn->get_tail_groups());
    source_desc.exact_tail_tokens = in.source->exact_tail_tokens;
    source_desc.has_cell_ext = in.source->has_cell_ext;
    source_desc.source_state_bytes = checkpoint->data_tgt.size();

    std::vector<handoff_source_layer> source_layers;
    for (const auto & layer : in.source->kvarn_layers) {
        handoff_source_layer l;
        l.il = layer.layer_id;
        l.record_dim_k = layer.record_dim_k;
        l.record_dim_v = layer.record_dim_v;
        l.k_slices = layer.head_slices_k;
        l.v_slices = layer.head_slices_v;
        l.n_head_kv = layer.n_head_kv;
        l.head_dim_k = layer.head_dim_k;
        l.head_dim_v = layer.head_dim_v;
        source_layers.push_back(l);
    }
    handoff_plan plan = handoff_plan_build(source_desc, source_layers, remote_layers,
            uint32_t(std::max(0, kvarn->params_key_bits())),
            uint32_t(std::max(0, kvarn->params_value_bits())), true);
    if (plan.st != handoff_status::ok) {
        return fail(("plan rejected: " + std::string(handoff_status_name(plan.st))).c_str());
    }

    // Effective READ index map from the POST-TRIM metadata: a cell is a
    // stage read only when the allocator marks it (allocation_cell_uses_stage)
    // with the ACTUAL stage slot of its group (sink group 0 -> slot 0);
    // sealed groups are record reads (plain positions). sinfo.stage_slots are
    // not guaranteed to be the read map; the allocator map is authoritative.
    const auto src_meta = source_kvarn->get_metadata_cache();
    std::vector<int64_t> index_map(selected);
    {
        for (uint32_t i = 0; i < selected; ++i) {
            const uint32_t pos = uint32_t(sinfos[0].idxs[0][i]);
            if (src_meta->allocation_cell_uses_stage(pos)) {
                const int32_t slot = src_meta->allocation_cell_stage_slot(pos);
                if (slot < 0) {
                    return fail("source allocator stage slot missing");
                }
                index_map[i] = llama_kvarn_encode_stage_cell(pos, uint32_t(slot));
            } else {
                index_map[i] = int64_t(pos); // sealed record read
            }
        }
    }

    // Destination dense row map: the candidate cells MUST be dense
    // (physical cell == logical token) for the milestone rawstorage writes;
    // verified, not assumed.
    std::vector<uint32_t> dst_row_of_token(selected);
    {
        const auto & dst_cells = kvarn->get_metadata_cache()->get_cells(in.seq_id);
        for (uint32_t pos = 0; pos < selected; ++pos) {
            if (dst_cells.pos_get(pos) != (llama_pos) pos) {
                return fail("candidate cells are not dense (physical != logical)");
            }
            dst_row_of_token[pos] = pos;
        }
    }

    handoff_transaction txn;
    const auto dst_meta = kvarn->get_metadata_cache();
    // Destination tail slots: authoritative position->slot map from the
    // candidate snapshot (state_tail_payload_slots is an ORDERED slot vector,
    // NOT indexed by token position).
    std::unordered_map<uint32_t, int32_t> dst_tail_slot_of_token;
    try {
        for (const auto & entry : dst_meta->state_tail_snapshot(in.seq_id)) {
            if (entry.seq_id == in.seq_id) {
                dst_tail_slot_of_token[uint32_t(entry.position)] = entry.slot;
            }
        }
    } catch (const std::exception & error) {
        return fail(("candidate tail snapshot lookup failed: " + std::string(error.what())).c_str());
    }

    size_t expected_ops = 0;
    const uint32_t n_groups = (selected + 127) / 128;

    // Local layers: compressed records groups + stage cells + exact tail.
    for (const uint32_t il : plan.local_layers) {
        llama_kvarn_handoff_layer_view src_view;
        llama_kvarn_handoff_layer_view dst_view;
        if (!source_kvarn->handoff_layer_view(int32_t(il), src_view) ||
                !kvarn->handoff_layer_view(int32_t(il), dst_view)) {
            return fail(("local layer tensor view missing: " + std::to_string(il)).c_str());
        }
        // Every required compressed component must EXIST (records K/V,
        // stage K/V, tail K/V); a missing component is a hard failure.
        for (ggml_tensor * pair : { src_view.k_records, src_view.v_records }) {
            if (pair == nullptr) {
                return fail(("local records component missing (layer " + std::to_string(il) + ")").c_str());
            }
            ggml_tensor * dst = pair == src_view.k_records ? dst_view.k_records : dst_view.v_records;
            if (dst == nullptr || pair->nb[2] != dst->nb[2]) {
                return fail("local records tensor mismatch");
            }
            const size_t group_bytes = size_t(pair->nb[2]);
            for (uint32_t g = 0; g < n_groups; ++g) {
                if (size_t(g) * group_bytes + group_bytes > ggml_nbytes(pair) ||
                        size_t(g) * group_bytes + group_bytes > ggml_nbytes(dst)) {
                    return fail("local records group out of bounds");
                }
                const handoff_status st = txn.add_copy(pair, size_t(g) * group_bytes,
                        dst, size_t(g) * group_bytes, group_bytes);
                if (st != handoff_status::ok) {
                    return fail("local records copy rejected");
                }
                ++expected_ops;
            }
        }
        // Stage cells: ONLY cells whose source metadata allocates an F16
        // stage row (canonical ownership; sealed groups stay in records).
        for (uint32_t c = 0; c < selected; ++c) {
            const uint32_t pos = uint32_t(sinfos[0].idxs[0][c]);
            if (!src_meta->allocation_cell_uses_stage(pos)) {
                continue; // sealed record cell; no F16 stage row
            }
            const int32_t src_slot = src_meta->allocation_cell_stage_slot(pos);
            const int32_t dst_slot = dst_meta->allocation_cell_stage_slot(pos);
            if (src_slot < 0 || dst_slot < 0) {
                return fail("local stage slot mapping missing");
            }
            const uint32_t lane = pos % 128;
            const size_t src_base = (size_t(src_slot) * 128 + lane);
            const size_t dst_base = (size_t(dst_slot) * 128 + lane);
            for (ggml_tensor * pair : { src_view.k_stage, src_view.v_stage }) {
                if (pair == nullptr) {
                    return fail(("local stage component missing (layer " + std::to_string(il) + ")").c_str());
                }
                ggml_tensor * dst = pair == src_view.k_stage ? dst_view.k_stage : dst_view.v_stage;
                const size_t nb = size_t(pair->nb[2]);
                if (dst == nullptr || src_base * nb + nb > ggml_nbytes(pair) ||
                        dst_base * nb + nb > ggml_nbytes(dst)) {
                    return fail("local stage slice out of bounds");
                }
                const handoff_status st = txn.add_copy(pair, src_base * nb, dst, dst_base * nb, nb);
                if (st != handoff_status::ok) {
                    return fail("local stage copy rejected");
                }
                ++expected_ops;
            }
        }
        // Exact tail rows (K and V) mapped by POSITION through each side's
        // authoritative snapshot maps.
        for (uint32_t pos = 0; pos < selected; ++pos) {
            const auto it = tail_slot_of_token.find(pos);
            if (it == tail_slot_of_token.end() || it->second < 0) {
                continue;
            }
            const auto dst_it = dst_tail_slot_of_token.find(pos);
            if (dst_it == dst_tail_slot_of_token.end() || dst_it->second < 0) {
                return fail("candidate tail slot mapping missing");
            }
            const int32_t dst_slot = dst_it->second;
            for (ggml_tensor * pair : { src_view.k_tail, src_view.v_tail }) {
                if (pair == nullptr) {
                    return fail(("local tail component missing (layer " + std::to_string(il) + ")").c_str());
                }
                ggml_tensor * dst = pair == src_view.k_tail ? dst_view.k_tail : dst_view.v_tail;
                const size_t row_bytes = ggml_row_size(pair->type, pair->ne[0]);
                if (dst == nullptr || pair->type != dst->type ||
                        size_t(it->second) * row_bytes + row_bytes > ggml_nbytes(pair) ||
                        size_t(dst_slot) * row_bytes + row_bytes > ggml_nbytes(dst)) {
                    return fail("local tail row out of bounds");
                }
                const handoff_status st = txn.add_copy(pair,
                        size_t(it->second) * row_bytes, dst, size_t(dst_slot) * row_bytes, row_bytes);
                if (st != handoff_status::ok) {
                    return fail("local tail copy rejected");
                }
                ++expected_ops;
            }
        }
    }

    // Remote layers: bounded conversion, K and V with their OWN tensors and
    // stable overlay contexts (owned until all conversions finish).
    std::vector<tail_source> tail_contexts;
    tail_contexts.reserve(plan.remote_layers.size() * 2);
    for (const auto & remote : plan.remote_layers) {
        llama_kvarn_handoff_layer_view src_view;
        if (!source_kvarn->handoff_layer_view(int32_t(remote.il), src_view)) {
            return fail(("remote layer source view missing: " + std::to_string(remote.il)).c_str());
        }
        ggml_tensor * dst_k = kvarn->get_standard_cache()
                ? kvarn->get_standard_cache()->get_k_storage(int32_t(remote.il)) : nullptr;
        ggml_tensor * dst_v = kvarn->get_standard_cache()
                ? kvarn->get_standard_cache()->get_v_storage(int32_t(remote.il)) : nullptr;
        if (dst_k == nullptr || dst_v == nullptr) {
            return fail("candidate standard storage missing for a remote layer");
        }
        const uint32_t k_slices = remote.head_dim_k == 64 ? 1 : remote.head_dim_k / 128;
        const uint32_t v_slices = remote.head_dim_v == 64 ? 1 : remote.head_dim_v / 128;

        tail_contexts.push_back({ src_view.k_tail, src_view.v_tail, &tail_slot_of_token,
                remote.head_dim_k, remote.n_head_kv, false });
        tail_contexts.push_back({ src_view.k_tail, src_view.v_tail, &tail_slot_of_token,
                remote.head_dim_v, remote.n_head_kv, true });

        handoff_convert_params convert_k;
        convert_k.records = src_view.k_records;
        convert_k.stage = src_view.k_stage;
        convert_k.source_backend = cpu.backend;
        convert_k.index_map = index_map.data();
        convert_k.n_tokens = selected;
        convert_k.tail = tail_read;
        convert_k.tail_user = &tail_contexts[tail_contexts.size() - 2];
        convert_k.dst = dst_k;
        convert_k.dst_offset = 0;
        convert_k.dst_row_of_token = dst_row_of_token.data();
        convert_k.head_dim = remote.head_dim_k;
        convert_k.n_heads = remote.n_head_kv;
        convert_k.value = false;
        convert_k.record_dim = src_view.record_dim_k;
        convert_k.head_slices = k_slices;
        convert_k.kvarn_bits = source_desc.key_bits;
        convert_k.stage_groups = source_desc.stage_groups;
        convert_k.qx_type = remote.qx_type_k;
        convert_k.rotation_wht = remote.rotation_k;
        convert_k.row_bytes = remote.row_bytes_k;
        convert_k.max_window = 64;
        if (txn.add_convert(convert_k) != handoff_status::ok) {
            return fail(("remote K conversion rejected (layer " + std::to_string(remote.il) + ")").c_str());
        }
        ++expected_ops;

        handoff_convert_params convert_v = convert_k;
        convert_v.records = src_view.v_records;
        convert_v.stage = src_view.v_stage;
        convert_v.record_dim = src_view.record_dim_v;
        convert_v.tail = tail_read;
        convert_v.tail_user = &tail_contexts[tail_contexts.size() - 1];
        convert_v.dst = dst_v;
        convert_v.head_dim = remote.head_dim_v;
        convert_v.head_slices = v_slices;
        convert_v.value = true;
        convert_v.kvarn_bits = source_desc.value_bits;
        convert_v.qx_type = remote.qx_type_v;
        convert_v.rotation_wht = remote.rotation_v;
        convert_v.row_bytes = remote.row_bytes_v;
        if (txn.add_convert(convert_v) != handoff_status::ok) {
            return fail(("remote V conversion rejected (layer " + std::to_string(remote.il) + ")").c_str());
        }
        ++expected_ops;
    }

    // Every required op must be accounted before the first candidate write.
    if (txn.op_count() != expected_ops) {
        return fail("required handoff op accounting mismatch");
    }

    const handoff_status prepared = txn.prepare();
    if (prepared != handoff_status::ok) {
        return fail(("prepare failed: " + std::string(handoff_status_name(prepared))).c_str());
    }
    for (const auto & tail : tail_contexts) {
        if (tail.failed) {
            return fail("exact-tail overlay failed for a remote layer");
        }
    }

    // Recurrent state: authoritative parser round-trip (source state_write ->
    // candidate state_read), handles layout/rollback-depth differences by
    // failing closed instead of raw array copies. The transfer budget derives
    // from the source's ACTUALLY allocated recurrent geometry (never a fixed
    // threshold), with a checked writer budget and complete reader
    // consumption verified before commit.
    try {
        uint64_t rs_allocated = 0;
        for (const auto & allocation : source_hybrid->get_mem_recr()->memory_breakdown()) {
            if (allocation.second > std::numeric_limits<uint64_t>::max() - rs_allocated) {
                return fail("recurrent allocation size overflow");
            }
            rs_allocated += allocation.second;
        }
        const uint64_t rs_budget = handoff_rs_budget(rs_allocated, 1ull << 30);
        byte_writer rs_writer(rs_budget);
        source_hybrid->get_mem_recr()->state_write(rs_writer, in.seq_id,
                LLAMA_STATE_SEQ_FLAGS_NONE);
        rs_bytes_ = rs_writer.n_bytes();
        std::vector<uint8_t> rs_bytes = rs_writer.take();
        byte_reader rs_reader(rs_bytes.data(), rs_bytes.size());
        hybrid->get_mem_recr()->state_read(rs_reader, in.seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (rs_reader.n_bytes() != rs_bytes.size()) {
            return fail("recurrent state reader did not consume the complete frame");
        }
        rs_reader.commit();
    } catch (const std::exception & error) {
        return fail(("recurrent state transfer failed: " + std::string(error.what())).c_str());
    }

    // Candidate KV/RS positions verified once more before publication. The
    // draft is NOT part of the milestone (target-only + existing bootstrap),
    // so only the target and its recurrent state are checked.
    if (llama_memory_seq_pos_max(llama_get_memory(in.ctx_tgt), in.seq_id) !=
            (llama_pos) selected - 1) {
        return fail("candidate KV position does not match the selected prefix");
    }
    if (llama_memory_seq_pos_max(hybrid->get_mem_recr(), in.seq_id) !=
            (llama_pos) selected - 1) {
        return fail("candidate recurrent position does not match the selected prefix");
    }

    const handoff_status committed = txn.commit();
    if (committed != handoff_status::ok) {
        return fail(("commit failed: " + std::string(handoff_status_name(committed))).c_str());
    }

    // MTP milestone: target-only prefix + the existing bootstrap re-anchors
    // the carry at the first real suffix decode (no synthetic carry).
    out.mtp_bootstrap = in.ctx_dft != nullptr;
    out.marker = handoff_marker(selected, uint32_t(plan.remote_layers.size()),
            txn.source_bytes(), txn.scratch_bytes(), txn.convert_ms());
    out.marker += " rs_bytes=" + std::to_string(rs_bytes_);
    out.prefix_tokens = in.state->prompt.tokens.get_tokens();
    if (out.prefix_tokens.size() > selected) {
        out.prefix_tokens.resize(selected);
    }

    // Disarm LAST: every failure/exception above still clears the candidate.
    out.committed = true;
    cguard.disarmed = true;
    return out;
}

} // namespace server_mixed_kv_handoff