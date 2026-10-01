#include "llama-kv-mixed-handoff.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>

namespace llama_kv_handoff {

const char * handoff_status_name(handoff_status st) {
    switch (st) {
        case handoff_status::ok:                  return "ok";
        case handoff_status::invalid_args:        return "invalid_args";
        case handoff_status::invalid_geometry:    return "invalid_geometry";
        case handoff_status::invalid_type:        return "invalid_type";
        case handoff_status::invalid_domain:      return "invalid_domain";
        case handoff_status::invalid_indices:     return "invalid_indices";
        case handoff_status::no_compatible_prefix:return "no_compatible_prefix";
        case handoff_status::mixed_to_pure_rejected: return "mixed_to_pure_rejected";
        case handoff_status::changed_qx_rejected: return "changed_qx_rejected";
        case handoff_status::changed_kvarn_rejected: return "changed_kvarn_rejected";
        case handoff_status::destination_missing: return "destination_missing";
        case handoff_status::payload_overflow:    return "payload_overflow";
        case handoff_status::workspace_overflow:  return "workspace_overflow";
        case handoff_status::callback_failure:    return "callback_failure";
        case handoff_status::sink_failure:        return "sink_failure";
        case handoff_status::compute_failure:     return "compute_failure";
    }
    return "unknown";
}

namespace {

constexpr size_t COPY_CHUNK_BYTES = 1u << 20; // 1 MiB bounded apply chunks
constexpr uint64_t KVAR_N_GROUP = 128;

bool checked_add(size_t a, size_t b, size_t & out) {
    if (a > std::numeric_limits<size_t>::max() - b) {
        return false;
    }
    out = a + b;
    return true;
}

// Adapts the engine's tail callback to the materializer's exact_token_fn.
struct tail_adapter {
    handoff_tail_fn fn = nullptr;
    void * user = nullptr;
    uint32_t head_dim = 0;
    uint32_t n_heads = 0;
};

bool tail_adapter_call(void * user, uint32_t token, float * rows) {
    tail_adapter * a = static_cast<tail_adapter *>(user);
    return a->fn(a->user, token, a->head_dim, a->n_heads, rows);
}

// Sink for convert_to_sink: writes each bounded block into the private
// candidate tensor at its token-row offset (rawstorage per token). With a
// destination row map the rows are scattered by the allocator's physical
// layout; without it the DENSE mapping (physical row == logical token) is
// enforced by the row-stride bounds of add_convert.
struct candidate_sink {
    ggml_tensor * dst = nullptr;
    size_t dst_offset = 0;
    size_t row_bytes = 0;
    const uint32_t * dst_row_of_token = nullptr;
    uint64_t bytes_written = 0;
    bool failed = false;
};

bool candidate_sink_call(void * user, const llama_kv_qx::sink_block & block) {
    candidate_sink * s = static_cast<candidate_sink *>(user);
    const size_t bytes = size_t(block.n_tokens) * block.row_bytes;
    if (s->dst_row_of_token == nullptr) {
        size_t offset = 0;
        if (!checked_add(s->dst_offset, size_t(block.t0) * block.row_bytes, offset) ||
                offset > ggml_nbytes(s->dst) || bytes > ggml_nbytes(s->dst) - offset) {
            s->failed = true;
            return false;
        }
        ggml_backend_tensor_set(s->dst, block.data, offset, bytes);
        s->bytes_written += bytes;
        return true;
    }
    // Scattered rows: one row per token through the allocator's map.
    const uint8_t * data = block.data;
    for (uint32_t t = 0; t < block.n_tokens; ++t) {
        const uint32_t row = s->dst_row_of_token[size_t(block.t0) + t];
        size_t offset = 0;
        if (!checked_add(s->dst_offset, size_t(row) * block.row_bytes, offset) ||
                offset > ggml_nbytes(s->dst) ||
                block.row_bytes > ggml_nbytes(s->dst) - offset) {
            s->failed = true;
            return false;
        }
        ggml_backend_tensor_set(s->dst, data, offset, block.row_bytes);
        data += block.row_bytes;
        s->bytes_written += block.row_bytes;
    }
    return true;
}

bool valid_qx_type(int32_t type) {
    return type == (int32_t) GGML_TYPE_Q4_0 || type == (int32_t) GGML_TYPE_Q5_0 ||
           type == (int32_t) GGML_TYPE_Q6_0 || type == (int32_t) GGML_TYPE_Q8_0;
}

bool valid_wht_width(uint32_t w) {
    return w == 0 || w == 64 || w == 128 || w == 256 || w == 512;
}

} // namespace

uint32_t handoff_select_prefix(
        const handoff_prefix_candidate * candidates,
        size_t count,
        uint32_t lcp) {
    uint32_t best = 0;
    if (candidates == nullptr) {
        return best;
    }
    for (size_t i = 0; i < count; ++i) {
        const auto & c = candidates[i];
        if (!c.host_only || c.n_tokens == 0 || c.pos_max < 0 ||
                int64_t(c.n_tokens) != int64_t(c.pos_max) + 1 || c.n_tokens > lcp) {
            continue;
        }
        best = std::max(best, c.n_tokens);
    }
    return best;
}

handoff_plan handoff_plan_build(
        const handoff_source_desc & source,
        const std::vector<handoff_source_layer> & source_layers,
        const std::vector<handoff_remote_layer> & remote_layers,
        uint32_t dest_kvarn_key_bits,
        uint32_t dest_kvarn_value_bits,
        bool dest_has_standard_cache) {
    handoff_plan plan;
    if (source.n_tokens == 0 || source.n_ctx_seq == 0 || source.key_bits <= 0 ||
            source.value_bits <= 0 || source.stage_groups < 2) {
        plan.st = handoff_status::invalid_args;
        return plan;
    }
    if (!dest_has_standard_cache) {
        // One-way only: a destination without standard remote layers would
        // require requantizing the converted prefix back into pure KVarN.
        plan.st = handoff_status::mixed_to_pure_rejected;
        return plan;
    }
    if (source.key_bits != int32_t(dest_kvarn_key_bits) ||
            source.value_bits != int32_t(dest_kvarn_value_bits)) {
        // KVarN params must be preserved compressed; never requantize.
        plan.st = handoff_status::changed_kvarn_rejected;
        return plan;
    }
    for (const auto & remote : remote_layers) {
        if (!valid_qx_type(remote.qx_type_k) || !valid_qx_type(remote.qx_type_v) ||
                remote.row_bytes_k == 0 || remote.row_bytes_v == 0 ||
                remote.n_head_kv == 0 || remote.head_dim_k == 0 || remote.head_dim_v == 0 ||
                !valid_wht_width(remote.rotation_k) || !valid_wht_width(remote.rotation_v) ||
                (remote.rotation_k > 0 && remote.head_dim_k % remote.rotation_k != 0) ||
                (remote.rotation_v > 0 && remote.head_dim_v % remote.rotation_v != 0)) {
            plan.st = handoff_status::changed_qx_rejected;
            return plan;
        }
        plan.remote_layers.push_back(remote);
        // Checked multiplication for the per-layer Qx footprint before any
        // addition (the product itself can wrap uint64).
        const uint64_t row_sum = uint64_t(remote.row_bytes_k) + uint64_t(remote.row_bytes_v);
        const uint64_t max_u64 = std::numeric_limits<uint64_t>::max();
        if (row_sum == 0 || uint64_t(source.n_tokens) > max_u64 / row_sum ||
                plan.qx_bytes > max_u64 - uint64_t(source.n_tokens) * row_sum) {
            plan.st = handoff_status::payload_overflow;
            return plan;
        }
        plan.qx_bytes += uint64_t(source.n_tokens) * row_sum;
    }
    // Local layers: every source attention layer not in the remote set.
    std::vector<uint32_t> remote_ids;
    remote_ids.reserve(plan.remote_layers.size());
    for (const auto & remote : plan.remote_layers) {
        remote_ids.push_back(remote.il);
    }
    std::sort(remote_ids.begin(), remote_ids.end());
    for (const auto & layer : source_layers) {
        if (!std::binary_search(remote_ids.begin(), remote_ids.end(), layer.il)) {
            plan.local_layers.push_back(layer.il);
        }
    }
    plan.n_tokens = source.n_tokens;
    plan.st = handoff_status::ok;
    return plan;
}

handoff_status handoff_transaction::add_copy(
        ggml_tensor * src, size_t src_offset,
        ggml_tensor * dst, size_t dst_offset, size_t bytes) {
    if (prepared_ || committed_ || failed_) {
        last_ = handoff_status::invalid_args; // post-prepare mutation rejected
        return last_;
    }
    if (src == nullptr || dst == nullptr || bytes == 0 ||
            src->buffer == nullptr || dst->buffer == nullptr ||
            src == dst || // source/destination alias rejected
            src->buffer == dst->buffer ||
            src_offset > ggml_nbytes(src) || bytes > ggml_nbytes(src) - src_offset ||
            dst_offset > ggml_nbytes(dst) || bytes > ggml_nbytes(dst) - dst_offset) {
        last_ = handoff_status::invalid_args;
        return last_;
    }
    copies_.push_back({ src, src_offset, dst, dst_offset, bytes });
    if (source_bytes_ > std::numeric_limits<uint64_t>::max() - bytes) {
        copies_.pop_back();
        last_ = handoff_status::payload_overflow;
        return last_;
    }
    source_bytes_ += bytes;
    last_ = handoff_status::ok;
    return last_;
}

handoff_status handoff_transaction::add_convert(const handoff_convert_params & params) {
    if (prepared_ || committed_ || failed_) {
        last_ = handoff_status::invalid_args; // post-prepare mutation rejected
        return last_;
    }
    if (params.records == nullptr || params.stage == nullptr ||
            params.source_backend == nullptr || params.index_map == nullptr ||
            params.n_tokens == 0 || params.dst == nullptr ||
            params.head_dim == 0 || params.n_heads == 0 ||
            !valid_qx_type(params.qx_type) || !valid_wht_width(params.rotation_wht) ||
            (params.rotation_wht > 0 && params.head_dim % params.rotation_wht != 0) ||
            params.row_bytes == 0 || params.max_window == 0 ||
            params.max_window > llama_kv_qx::LLAMA_KV_MATERIALIZER_MAX_WINDOW) {
        last_ = handoff_status::invalid_args;
        return last_;
    }
    if (params.dst->buffer == nullptr ||
            size_t(params.n_tokens) > ggml_nbytes(params.dst) / params.row_bytes) {
        last_ = handoff_status::invalid_args;
        return last_;
    }
    // Source/destination alias rejection: the converted bytes must never be
    // written into the source tensors themselves.
    if (params.dst == params.records || params.dst == params.stage) {
        last_ = handoff_status::invalid_args;
        return last_;
    }
    // Real destination type and stride: the storage tensor must be exactly
    // the whitelisted Qx type with row_bytes per token row.
    if (params.dst->type != (ggml_type) params.qx_type ||
            params.dst->nb[1] != params.row_bytes) {
        last_ = handoff_status::invalid_args;
        return last_;
    }
    // Checked multiplication for the destination footprint.
    size_t dst_footprint = 0;
    if (!checked_add(params.dst_offset, size_t(params.n_tokens) * params.row_bytes,
            dst_footprint) || dst_footprint > ggml_nbytes(params.dst)) {
        last_ = handoff_status::invalid_args;
        return last_;
    }
    if (params.dst_row_of_token != nullptr) {
        // Scatter map: every row must be in bounds and unique (a duplicated
        // physical row would silently overwrite).
        const uint64_t capacity = ggml_nbytes(params.dst) / params.row_bytes;
        std::vector<uint32_t> rows(params.n_tokens);
        for (uint32_t i = 0; i < params.n_tokens; ++i) {
            const uint64_t row = params.dst_row_of_token[i];
            if (row >= capacity) {
                last_ = handoff_status::invalid_args;
                return last_;
            }
            rows[i] = uint32_t(row);
        }
        std::sort(rows.begin(), rows.end());
        for (size_t i = 1; i < rows.size(); ++i) {
            if (rows[i] == rows[i - 1]) {
                last_ = handoff_status::invalid_args;
                return last_;
            }
        }
    }
    converts_.push_back({ params });
    last_ = handoff_status::ok;
    return last_;
}

handoff_status handoff_transaction::prepare() {
    if (prepared_ || committed_ || failed_) {
        last_ = handoff_status::invalid_args;
        return last_;
    }
    // Validate every staged op before running any conversion.
    for (const auto & op : copies_) {
        if (op.src->buffer == nullptr || op.dst->buffer == nullptr ||
                op.src_offset > ggml_nbytes(op.src) ||
                op.bytes > ggml_nbytes(op.src) - op.src_offset ||
                op.dst_offset > ggml_nbytes(op.dst) ||
                op.bytes > ggml_nbytes(op.dst) - op.dst_offset) {
            last_ = handoff_status::invalid_args;
            return last_;
        }
    }
    // Metrics: compressed source bytes read by the local copies are counted at
    // add_copy; F32 traffic and scratch ESTIMATE are per conversion.
    for (const auto & op : converts_) {
        const auto & p = op.params;
        f32_traffic_bytes_ += uint64_t(p.n_tokens) * p.head_dim * p.n_heads * 4;
        const uint64_t scratch = uint64_t(p.max_window) *
                (uint64_t(p.head_dim) * p.n_heads * 4 + p.row_bytes) + (1u << 20);
        scratch_bytes_ = std::max<uint64_t>(scratch_bytes_, scratch);
    }

    // Preflight EVERY conversion (materializer validity, index map, converter
    // totals) before any conversion runs: a second invalid convert must leave
    // ZERO writes in the candidate.
    std::vector<std::unique_ptr<llama_kv_qx::materializer>> materializers;
    materializers.reserve(converts_.size());
    std::vector<tail_adapter> adapters(converts_.size());
    std::vector<llama_kv_qx::config> ccfgs(converts_.size());
    for (size_t ci = 0; ci < converts_.size(); ++ci) {
        const auto & p = converts_[ci].params;
        tail_adapter & adapter = adapters[ci];
        adapter.fn = p.tail;
        adapter.user = p.tail_user;
        adapter.head_dim = p.head_dim;
        adapter.n_heads = p.n_heads;

        llama_kv_qx::materializer_config mcfg;
        mcfg.head_dim = p.head_dim;
        mcfg.n_heads = p.n_heads;
        mcfg.value = p.value;
        mcfg.record_dim = p.record_dim;
        mcfg.head_slices = p.head_slices;
        mcfg.bits = p.kvarn_bits;
        mcfg.stage_groups = p.stage_groups;
        mcfg.max_window = std::min(p.max_window,
                llama_kv_qx::LLAMA_KV_MATERIALIZER_MAX_WINDOW);

        llama_kv_qx::materializer_source msrc;
        msrc.records = p.records;
        msrc.stage = p.stage;
        msrc.backend = p.source_backend;

        auto materializer = std::make_unique<llama_kv_qx::materializer>(mcfg, msrc);
        if (!materializer->valid()) {
            failed_ = true;
            last_ = llama_kv_qx::mat_status::invalid_geometry == materializer->last_status()
                    ? handoff_status::invalid_geometry : handoff_status::invalid_args;
            return last_;
        }
        const llama_kv_qx::mat_status mp = materializer->set_prefix(
                p.index_map, p.n_tokens,
                p.tail ? tail_adapter_call : nullptr, p.tail ? &adapter : nullptr);
        if (mp != llama_kv_qx::mat_status::ok) {
            failed_ = true;
            last_ = mp == llama_kv_qx::mat_status::invalid_indices
                    ? handoff_status::invalid_indices : handoff_status::invalid_args;
            return last_;
        }

        llama_kv_qx::config ccfg;
        ccfg.dst_type = p.qx_type;
        ccfg.head_dim = p.head_dim;
        ccfg.n_heads = p.n_heads;
        ccfg.value = p.value;
        ccfg.src.record_dim = p.record_dim;
        ccfg.src.rotated = true;
        ccfg.dst.wht_width = p.rotation_wht;
        ccfg.max_tokens_per_block = p.max_window;
        if (llama_kv_qx::payload_bytes(ccfg, p.n_tokens) !=
                size_t(p.n_tokens) * p.row_bytes) {
            failed_ = true;
            last_ = handoff_status::invalid_geometry;
            return last_;
        }
        ccfgs[ci] = ccfg;
        materializers.push_back(std::move(materializer));
    }

    // Execute the conversions (bounded) into private candidate tensors.
    const int64_t t_start = ggml_time_us();
    for (size_t ci = 0; ci < converts_.size(); ++ci) {
        const auto & p = converts_[ci].params;
        candidate_sink sink;
        sink.dst = p.dst;
        sink.dst_offset = p.dst_offset;
        sink.row_bytes = p.row_bytes;
        sink.dst_row_of_token = p.dst_row_of_token;

        const llama_kv_qx::result r = llama_kv_qx::convert_to_sink(
                ccfgs[ci], p.n_tokens, llama_kv_qx::materializer_read_block,
                materializers[ci].get(), candidate_sink_call, &sink);
        if (r.st != llama_kv_qx::status::ok || sink.failed) {
            failed_ = true;
            last_ = r.st == llama_kv_qx::status::non_finite_input
                    ? handoff_status::invalid_domain
                    : (r.st == llama_kv_qx::status::callback_failure
                            ? handoff_status::callback_failure
                            : (r.st == llama_kv_qx::status::sink_failure
                                    ? handoff_status::sink_failure
                                    : handoff_status::compute_failure));
            return last_;
        }
        if (sink.bytes_written != uint64_t(p.n_tokens) * p.row_bytes) {
            failed_ = true;
            last_ = handoff_status::compute_failure;
            return last_;
        }
    }
    convert_ms_ = (ggml_time_us() - t_start) / 1000.0;
    prepared_ = true;
    last_ = handoff_status::ok;
    return last_;
}

handoff_status handoff_transaction::commit() {
    if (!prepared_ || committed_ || failed_) {
        last_ = handoff_status::invalid_args;
        return last_;
    }
    for (const auto & op : copies_) {
        size_t done = 0;
        while (done < op.bytes) {
            const size_t chunk = std::min(op.bytes - done, COPY_CHUNK_BYTES);
            std::vector<uint8_t> scratch(chunk);
            ggml_backend_tensor_get(op.src, scratch.data(), op.src_offset + done, chunk);
            ggml_backend_tensor_set(op.dst, scratch.data(), op.dst_offset + done, chunk);
            done += chunk;
            scratch_bytes_ = std::max<uint64_t>(scratch_bytes_, chunk);
        }
    }
    committed_ = true;
    last_ = handoff_status::ok;
    return last_;
}

void handoff_transaction::discard() {
    copies_.clear();
    converts_.clear();
    prepared_ = false;
    committed_ = false;
    failed_ = false;
    source_bytes_ = 0;
    f32_traffic_bytes_ = 0;
    scratch_bytes_ = 0;
    convert_ms_ = 0.0;
}

std::string handoff_marker(
        uint64_t reused_tokens,
        uint32_t converted_layers,
        uint64_t source_bytes,
        uint64_t scratch_bytes,
        double convert_ms) {
    return "mixed KV handoff: completed "
        "reused_tokens=" + std::to_string(reused_tokens) +
        " converted_layers=" + std::to_string(converted_layers) +
        " source_bytes=" + std::to_string(source_bytes) +
        " scratch_bytes=" + std::to_string(scratch_bytes) +
        " convert_ms=" + std::to_string(convert_ms);
}

std::string handoff_marker_failed(const char * reason) {
    return std::string("mixed KV handoff: failed reason=") + (reason ? reason : "unknown");
}

std::string handoff_marker_no_source() {
    return "mixed KV handoff: no compatible source prefix";
}

// Bounded recurrent-state transfer budget: the cap derives from the SOURCE's
// actually allocated recurrent geometry (never a fixed 4B-model threshold)
// plus a small metadata margin, and is clamped by a physical RAM budget.
uint64_t handoff_rs_budget(uint64_t rs_allocated_bytes, uint64_t physical_ram_budget) {
    constexpr uint64_t RS_MARGIN = 16ull << 20;
    const uint64_t max_u64 = std::numeric_limits<uint64_t>::max();
    const uint64_t cap = rs_allocated_bytes > max_u64 - RS_MARGIN
            ? max_u64 : rs_allocated_bytes + RS_MARGIN;
    return std::min<uint64_t>(cap, physical_ram_budget);
}

} // namespace llama_kv_handoff