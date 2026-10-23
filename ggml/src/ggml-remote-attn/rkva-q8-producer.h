#pragma once

#include "ggml.h"
#include "remote-attn-protocol.h"
#include "rkva-native-owner.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

// Descriptor layout 5 ("native Q8_0 overflow rows"): original-domain projected
// F32 K/V are packed as ordinary ggml block_q8_0 rows on the host, with no
// sinkhorn/WHT, records, stage ring or exact tail. Head-major planes are K then
// V; each 256-dim head is 8 q8_0 blocks (272 bytes), so a token row is
// n_head_kv * 272 bytes per plane (K and V are separate planes). Uploads use
// the regular op16 header: record_begin/record_groups stay 0 and
// append_begin/append_rows describe the chronological delta (<live_count>,
// first upload flagged RESET).
class rkva_q8_producer : public rkva_native_owner {
public:
    rkva_q8_producer(uint32_t origin, uint32_t capacity, uint32_t heads)
        : origin_(origin), capacity_(capacity), heads_(heads) {
        if (!heads_ || origin % 256 || !capacity_ ||
                uint64_t(origin_) + capacity_ > INT32_MAX) {
            throw std::invalid_argument("native Q8_0 range geometry");
        }
    }

    static constexpr uint32_t head_dim = 256;
    static constexpr uint32_t record_bytes = head_dim / 32 * 34; // 272 B per head

    static size_t row_bytes(uint32_t heads) { return size_t(heads) * record_bytes; }

    uint32_t rows() const override { return rows_; }

    // The CPU quantizer runs on the scheduler thread inside append(); there is
    // no optional CUDA producer for layout 5.
    bool prepare(const float * keys, const float * values, const int32_t * positions,
                 uint32_t count, const float * device_keys = nullptr,
                 const float * device_values = nullptr) override {
        (void) keys; (void) values; (void) positions; (void) count;
        (void) device_keys; (void) device_values;
        return true;
    }

    bool append(const float * keys, const float * values, const int32_t * positions,
                uint32_t count, bool reset_snapshot, std::vector<uint8_t> & payload) override {
        if (!keys || !values || !positions || !count || count > capacity_ - rows_) return false;
        for (uint32_t i = 0; i < count; ++i) {
            if (positions[i] != int64_t(origin_) + rows_ + i) return false;
        }
        const size_t elements = size_t(count) * heads_ * head_dim;
        if (!all_finite(keys, elements) || !all_finite(values, elements)) return false;

        const uint32_t before = rows_;
        const uint32_t after = before + count;
        const rkva_kvarn_range_upload header {
            after, 0u, 0u, reset_snapshot ? RKVA_KVARN_UPLOAD_RESET : 0u,
            reset_snapshot ? 0u : before, reset_snapshot ? after : count, {0u, 0u}};

        const size_t plane = size_t(count) * row_bytes(heads_);
        payload.clear();
        payload.resize(sizeof(header) + 2 * plane);
        std::memcpy(payload.data(), &header, sizeof(header));

        uint8_t * dst = payload.data() + sizeof(header);
        for (const float * source : {keys, values}) {
            // One call packs count * heads contiguous 256-value rows and keeps
            // the ggml block ordering (head-major) expected by the Q8_0 store.
            const size_t written = ggml_quantize_chunk(GGML_TYPE_Q8_0, source, dst, 0,
                    int64_t(count) * heads_, head_dim, nullptr);
            if (written != plane) return false;
            dst += plane;
        }
        rows_ = after;
        return true;
    }

    // Layout 5 has no ORIGINAL-F16 precision tail, so the resident KVarN4
    // rewind payload (op21) does not apply. A session-wide range trim (op8) is
    // exact for these plain Q8_0 rows, so suffix rollback is advertised through
    // can_rewind() and rewind() only mirrors the row count for the next
    // append_begin value.
    bool can_rewind(uint32_t keep) const override { return keep <= rows_; }
    bool rewind_tail(uint32_t keep, std::vector<uint8_t> & output) const override {
        (void) keep; (void) output; return false;
    }
    bool rewind(uint32_t keep) override {
        if (keep > rows_) return false;
        rows_ = keep;
        return true;
    }

private:
    static bool all_finite(const float * data, size_t elements) {
        for (size_t i = 0; i < elements; ++i) {
            if (!std::isfinite(data[i])) return false;
        }
        return true;
    }

    uint32_t origin_, capacity_, heads_, rows_ = 0;
};

static_assert(rkva_q8_producer::record_bytes == 272, "native Q8_0 record bytes");
