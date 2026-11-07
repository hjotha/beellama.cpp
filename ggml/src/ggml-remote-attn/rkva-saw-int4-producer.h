#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "remote-attn-protocol.h"
#include "rkva-native-owner.h"
#include "rkva-low4-cuda-prepared.h"
#include "../../../tools/xbox/uwp/rkva/rkva-saw-int4.h"

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

// Descriptor layout 6 ("native SAW-INT4 / BDR-INT4 overflow rows"): original-domain
// projected F32 K/V are transformed with BDR-16 (16x16 block-diagonal Walsh-Hadamard)
// and packed as 136-byte block_saw_int4 rows on CPU or opt-in CUDA (128B split-nibble qs +
// float scale + float zero per 256-dim head). Head-major planes are K then V; each
// token row is n_head_kv * 136 bytes per plane (5.56% smaller than Q4_0's 144 B/head).
// Uploads use the regular op16 header: record_begin/record_groups stay 0 and
// append_begin/append_rows describe the chronological delta (<live_count>,
// first upload flagged RESET).
class rkva_saw_int4_producer : public rkva_native_owner {
public:
    rkva_saw_int4_producer(uint32_t origin, uint32_t capacity, uint32_t heads)
        : origin_(origin), capacity_(capacity), heads_(heads) {
        if (!heads_ || origin % 256 || !capacity_ ||
                uint64_t(origin_) + capacity_ > INT32_MAX) {
            throw std::invalid_argument("native SAW-INT4 range geometry");
        }
        cuda_.configure("GGML_RKVA_SAW_PRODUCER", GGML_TYPE_SAW_INT4, heads_, record_bytes);
    }

    static constexpr uint32_t head_dim = 256;
    static constexpr uint32_t record_bytes = sizeof(block_saw_int4); // 136 B per head

    static size_t row_bytes(uint32_t heads) { return size_t(heads) * record_bytes; }

    uint32_t rows() const override { return rows_; }

    bool prepare(const float * keys, const float * values, const int32_t * positions,
                 uint32_t count, const float * device_keys = nullptr,
                 const float * device_values = nullptr) override {
        return cuda_.prepare(keys, values, positions, count, origin_, rows_, capacity_, device_keys, device_values);
    }

    bool append(const float * keys, const float * values, const int32_t * positions,
                uint32_t count, bool reset_snapshot, std::vector<uint8_t> & payload) override {
        if (!keys || !values || !positions || !count || count > capacity_ - rows_ || (reset_snapshot && rows_ != 0)) return false;
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
        if (plane > (UINT32_MAX - sizeof(header)) / 2) return false;
        std::vector<uint8_t> next(sizeof(header) + 2 * plane);
        std::memcpy(next.data(), &header, sizeof(header));

        uint8_t * dst = next.data() + sizeof(header);
        if (cuda_) {
            if (!cuda_.copy(keys, values, rows_, count, dst)) return false;
        } else {
            for (const float * source : {keys, values}) {
                rkva_saw_int4::quantize_row_saw_int4(
                    source, reinterpret_cast<block_saw_int4 *>(dst), int64_t(elements));
                dst += plane;
            }
        }
        payload.swap(next);
        rows_ = after;
        return true;
    }

    bool can_rewind(uint32_t keep) const override { return keep <= rows_; }
    bool rewind_tail(uint32_t keep, std::vector<uint8_t> & output) const override {
        (void) keep; (void) output; return false;
    }
    bool rewind(uint32_t keep) override {
        if (keep > rows_) return false;
        cuda_.invalidate();
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

    rkva_low4_cuda_prepared cuda_;
    uint32_t origin_, capacity_, heads_, rows_ = 0;
};

static_assert(rkva_saw_int4_producer::record_bytes == 136, "native SAW-INT4 record bytes");
