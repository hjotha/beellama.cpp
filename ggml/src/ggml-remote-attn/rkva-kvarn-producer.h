#pragma once

#include "ggml-kvarn-cpu-store.h"
#include "../ggml-impl.h"
#include "remote-attn-protocol.h"
#include "rkva-kvarn-rewind.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

// Experimental authoritative native owner with bounded resident rollback. Original projected
// F32 K/V are consumed before the separate standard-cache bookkeeping store.
// FULL native snapshots remain unsupported; resident rewinds retain original F32
// inputs for the most recent 256 rows and support at most 128 removed rows. No already-quantized source can initialize this owner.
class rkva_kvarn_producer {
public:
    rkva_kvarn_producer(uint32_t origin, uint32_t capacity, uint32_t heads,
                        uint32_t iterations, ggml_backend_cpu_kvarn_store_direct_v1_t store, uint32_t precision_tail_tokens = 128)
        : origin_(origin), capacity_(capacity), heads_(heads), iterations_(iterations), precision_tail_tokens_(precision_tail_tokens), store_(store) {
        if (!store || !heads || !iterations || (precision_tail_tokens != 0 && precision_tail_tokens != 128) ||
                origin % 128 || !capacity ||
                uint64_t(origin) + capacity > INT32_MAX) throw std::invalid_argument("native KVarN geometry");
        const size_t groups = (uint64_t(origin) + capacity + 127) / 128;
        // Full global addressing is intentionally retained for this first
        // implementation so the canonical CPU store sees the true group ID.
        // Only overflow record groups are ever sent to the console.
        records_k_.resize(groups * physical_heads() * record_bytes);
        records_v_.resize(records_k_.size());
        stage_k_.resize(size_t(384) * width());
        stage_v_.resize(stage_k_.size());
        tail_k_.resize(size_t(precision_tail_tokens_) * width());
        tail_v_.resize(tail_k_.size());
        original_k_.resize(size_t(256) * width());
        original_v_.resize(original_k_.size());
        original_pos_.assign(256, UINT32_MAX);
    }

    static constexpr uint32_t max_rewind = 128;
    uint32_t rows() const { return rows_; }
    bool can_rewind(uint32_t keep) const {
        if (keep > rows_ || rows_ - keep > max_rewind) return false;
        const uint32_t begin = std::min(keep / 128 * 128, keep > 128 ? keep - 128 : 0);
        for (uint32_t row = begin; row < keep; ++row) {
            if (original_pos_[row % 256] != row) return false;
        }
        return true;
    }

    // Chronological overwritten ORIGINAL-domain precision-tail rows, reconstructed without any
    // Q4/Q8 decoding. Used by the atomic all-layer rewind RPC before publication.
    bool rewind_tail(uint32_t keep, std::vector<uint8_t> & output) const {
        if (!can_rewind(keep)) return false;
        const auto span = rkva_kvarn_rewind_tail(rows_, keep, precision_tail_tokens_);
        for (const auto * source : {&original_k_, &original_v_}) {
            for (uint32_t row = span.begin; row < span.begin + span.rows; ++row) {
                for (size_t d = 0; d < width(); ++d) {
                    const ggml_fp16_t half = ggml_fp32_to_fp16((*source)[size_t(row % 256) * width() + d]);
                    const auto * bytes = reinterpret_cast<const uint8_t *>(&half);
                    output.insert(output.end(), bytes, bytes + sizeof(half));
                }
            }
        }
        return true;
    }

    bool rewind(uint32_t keep) {
        if (!can_rewind(keep)) return false;
        // Rebuild only the affected incomplete group through the canonical
        // original-F32 store. Earlier sealed native records are untouched.
        const uint32_t group_begin = keep / 128 * 128;
        std::vector<float> k(size_t(keep - group_begin) * width()), v(k.size());
        std::vector<int32_t> positions(keep - group_begin);
        for (uint32_t row = group_begin; row < keep; ++row) {
            std::copy_n(original_k_.data() + size_t(row % 256) * width(), width(), k.data() + size_t(row - group_begin) * width());
            std::copy_n(original_v_.data() + size_t(row % 256) * width(), width(), v.data() + size_t(row - group_begin) * width());
            positions[row - group_begin] = int32_t(origin_ + row);
        }
        rows_ = group_begin;
        std::vector<uint8_t> ignored;
        if (keep != group_begin && !append(k.data(), v.data(), positions.data(), keep - group_begin, false, ignored)) return false;
        rows_ = keep;
        for (uint32_t row = keep - std::min(keep, precision_tail_tokens_); row < keep; ++row) {
            for (size_t d = 0; d < width(); ++d) {
                tail_k_[size_t(row % 128) * width() + d] = ggml_fp32_to_fp16(original_k_[size_t(row % 256) * width() + d]);
                tail_v_[size_t(row % 128) * width() + d] = ggml_fp32_to_fp16(original_v_[size_t(row % 256) * width() + d]);
            }
        }
        return true;
    }
    size_t host_bytes() const {
        return records_k_.size() + records_v_.size() +
            sizeof(float) * (original_k_.size() + original_v_.size()) +
            sizeof(ggml_fp16_t) * (stage_k_.size() + stage_v_.size() + tail_k_.size() + tail_v_.size());
    }

    // Emits one op16 payload. Both input arrays are original-domain F32
    // [token][KV head][256]; no Q4/Q8 decoder occurs on this path.
    bool append(const float * keys, const float * values, const int32_t * positions,
                uint32_t count, bool reset_snapshot, std::vector<uint8_t> & payload) {
        if (!keys || !values || !positions || !count || count > capacity_ - rows_) return false;
        for (uint32_t i = 0; i < count; ++i) {
            if (positions[i] != int64_t(origin_) + rows_ + i) return false;
        }
        for (size_t i = 0; i < size_t(count) * width(); ++i) {
            if (!std::isfinite(keys[i]) || !std::isfinite(values[i])) return false;
        }
        const uint32_t before = rows_;
        const uint32_t after = before + count;
        const uint32_t record_begin = reset_snapshot ? 0 : before / 128;
        const uint32_t record_groups = after / 128 - record_begin;
        std::vector<ggml_fp16_t> append_k(size_t(count) * width()), append_v(append_k.size());
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
            ggml_init({16 * ggml_tensor_overhead(), nullptr, true}), ggml_free);
        if (!ctx) return false;
        auto * current = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 128, physical_heads(), 1);
        auto * index = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, 1);
        auto * sk = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, 128, physical_heads(), 384);
        auto * sv = ggml_dup_tensor(ctx.get(), sk);
        auto * rk = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_I8, record_bytes, physical_heads(),
                                      records_k_.size() / (record_bytes * physical_heads()));
        auto * rv = ggml_dup_tensor(ctx.get(), rk);
        sk->data = stage_k_.data(); sv->data = stage_v_.data();
        rk->data = records_k_.data(); rv->data = records_v_.data();
        auto * kstore = ggml_kvarn_store(ctx.get(), current, index, sk, rk, 4, iterations_, false, 3);
        auto * vstore = ggml_kvarn_store(ctx.get(), current, index, sv, rv, 4, iterations_, true, 3);
        for (auto * op : {kstore, vstore}) {
            ggml_set_op_params_i32(op, 5, 2); // two physical slices per model KV head
            ggml_set_op_params_i32(op, 8, 2); // canonical ping-pong tail slots
            ggml_set_op_params_i32(op, 9, 1); // eager records, exactly as the Bee cache
        }
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t row = before + i;
            std::copy_n(keys + size_t(i) * width(), width(), original_k_.data() + size_t(row % 256) * width());
            std::copy_n(values + size_t(i) * width(), width(), original_v_.data() + size_t(row % 256) * width());
            original_pos_[row % 256] = row;
            int64_t global = positions[i];
            index->data = &global;
            current->data = const_cast<float *>(keys + size_t(i) * width());
            if (!store_(kstore)) return false;
            current->data = const_cast<float *>(values + size_t(i) * width());
            if (!store_(vstore)) return false;
            const uint32_t group = uint32_t(global) / 128;
            const uint32_t slot = group == 0 ? 0 : 1 + (group - 1) % 2;
            const size_t stage_offset = (size_t(slot) * 128 + uint32_t(global) % 128) * width();
            std::memcpy(append_k.data() + size_t(i) * width(), stage_k_.data() + stage_offset, width() * 2);
            std::memcpy(append_v.data() + size_t(i) * width(), stage_v_.data() + stage_offset, width() * 2);
            const size_t tail_offset = size_t(before + i) % 128 * width();
            for (size_t d = 0; precision_tail_tokens_ && d < width(); ++d) {
                tail_k_[tail_offset + d] = ggml_fp32_to_fp16(keys[size_t(i) * width() + d]);
                tail_v_[tail_offset + d] = ggml_fp32_to_fp16(values[size_t(i) * width() + d]);
            }
        }
        const rkva_kvarn_range_upload header {
            after, record_begin, record_groups, reset_snapshot ? 1u : 0u,
            reset_snapshot ? 0u : before, reset_snapshot ? 0u : count, {0, 0}};
        payload.clear();
        const auto emit = [&](const void * data, size_t bytes) {
            const auto * begin = static_cast<const uint8_t *>(data);
            payload.insert(payload.end(), begin, begin + bytes);
        };
        emit(&header, sizeof(header));
        const size_t record_offset = size_t(origin_ / 128 + record_begin) * physical_heads() * record_bytes;
        const size_t record_size = size_t(record_groups) * physical_heads() * record_bytes;
        emit(records_k_.data() + record_offset, record_size);
        emit(records_v_.data() + record_offset, record_size);
        if (reset_snapshot) {
            emit(stage_k_.data(), stage_k_.size() * 2);
            emit(stage_v_.data(), stage_v_.size() * 2);
            for (const auto * tail : {&tail_k_, &tail_v_}) {
                for (uint32_t row = after - std::min(after, precision_tail_tokens_); row < after; ++row) {
                    emit(tail->data() + size_t(row % 128) * width(), width() * 2);
                }
            }
        } else {
            emit(append_k.data(), append_k.size() * 2);
            emit(append_v.data(), append_v.size() * 2);
            // Per-append original tail rows must be emitted from the original
            // input, since a >128-token append wraps the persistent tail ring.
            for (const float * source : {keys, values}) {
                for (size_t i = 0; precision_tail_tokens_ && i < size_t(count) * width(); ++i) {
                    const ggml_fp16_t half = ggml_fp32_to_fp16(source[i]);
                    emit(&half, sizeof(half));
                }
            }
        }
        rows_ = after;
        return true;
    }

private:
    static constexpr size_t record_bytes = 8960;
    size_t width() const { return size_t(heads_) * 256; }
    uint32_t physical_heads() const { return heads_ * 2; }
    uint32_t origin_, capacity_, heads_, iterations_, precision_tail_tokens_, rows_ = 0;
    ggml_backend_cpu_kvarn_store_direct_v1_t store_;
    std::vector<uint8_t> records_k_, records_v_;
    std::vector<float> original_k_, original_v_;
    std::vector<uint32_t> original_pos_;
    std::vector<ggml_fp16_t> stage_k_, stage_v_, tail_k_, tail_v_;
};
