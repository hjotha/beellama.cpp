#pragma once
#include "ggml.h"
#include "remote-attn-protocol.h"
#include "rkva-native-owner.h"
#include "rkva-low4-cuda-prepared.h"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

// Layouts 9/10: Ordinary GGML Q4_1/IQ4_NL blocks, not Q4_0.
// 256 values/head, token-major/head-major planes K then V. No WHT/tail/stage.
// The shared native owner gives these rows the existing async op22 transport.
class rkva_low4_producer : public rkva_native_owner {
public:
    static constexpr uint32_t head_dim = 256;
    rkva_low4_producer(uint32_t origin, uint32_t capacity, uint32_t heads, ggml_type type)
        : origin_(origin), capacity_(capacity), heads_(heads), type_(type) {
        if (!heads || heads > UINT32_MAX / head_dim || origin % 256 || !capacity ||
                uint64_t(origin) + capacity > INT32_MAX ||
                !record_bytes(type))
            throw std::invalid_argument("native range geometry/type");
        if (type_ == GGML_TYPE_Q4_1 || type_ == GGML_TYPE_IQ4_NL) {
            cuda_.configure("GGML_RKVA_LOW4_PRODUCER", type_, heads_, record_bytes(type_));
        }
    }
    static uint32_t record_bytes(ggml_type type) {
        if (type == GGML_TYPE_F32) return 1024u;
        if (type == GGML_TYPE_BF16 || type == GGML_TYPE_F16) return 512u;
        return uint32_t(ggml_row_size(type, head_dim));
    }
    size_t row_bytes() const { return size_t(heads_) * record_bytes(type_); }
    uint32_t rows() const override { return rows_; }
    bool prepare(const float * keys, const float * values, const int32_t * positions,
                 uint32_t count, const float * device_keys = nullptr,
                 const float * device_values = nullptr) override {
        return cuda_.prepare(keys, values, positions, count, origin_, rows_, capacity_, device_keys, device_values);
    }

    bool append(const float * keys, const float * values, const int32_t * positions,
                uint32_t count, bool reset, std::vector<uint8_t> & payload) override {
        if (!keys || !values || !positions || !count || count > capacity_ - rows_ ||
                (reset && rows_ != 0)) return false;
        for (uint32_t i = 0; i < count; ++i)
            if (positions[i] != int64_t(origin_) + rows_ + i) return false;
        const size_t elements = size_t(count) * heads_ * head_dim;
        for (const float * src : {keys, values})
            for (size_t i = 0; i < elements; ++i)
                if (!std::isfinite(src[i]) || std::fabs(src[i]) > 65504.0f) return false;
        const size_t plane = size_t(count) * row_bytes();
        // rkva_request.payload_bytes is uint32_t. Reject before allocating.
        if (plane > (UINT32_MAX - sizeof(rkva_kvarn_range_upload)) / 2) return false;
        const rkva_kvarn_range_upload header {
            rows_ + count, 0u, 0u, reset ? RKVA_KVARN_UPLOAD_RESET : 0u,
            rows_, count, {0u, 0u}};
        std::vector<uint8_t> next(sizeof(header) + 2 * plane);
        std::memcpy(next.data(), &header, sizeof(header));
        uint8_t * dst = next.data() + sizeof(header);
        if (cuda_) {
            if (!cuda_.copy(keys, values, rows_, count, dst)) return false;
        } else {
            for (const float * src : {keys, values}) {
                if (type_ == GGML_TYPE_F32) {
                    std::memcpy(dst, src, plane);
                } else if (type_ == GGML_TYPE_BF16) {
                    ggml_fp32_to_bf16_row(src, reinterpret_cast<ggml_bf16_t*>(dst), int64_t(count) * heads_ * head_dim);
                } else if (type_ == GGML_TYPE_F16) {
                    ggml_fp32_to_fp16_row(src, reinterpret_cast<ggml_fp16_t*>(dst), int64_t(count) * heads_ * head_dim);
                } else {
                    if (ggml_quantize_chunk(type_, src, dst, 0, int64_t(count) * heads_, head_dim,
                                            nullptr) != plane) return false;
                }
                if (type_ == GGML_TYPE_Q4_1 || type_ == GGML_TYPE_IQ4_NL) {
                    const size_t block_bytes = type_ == GGML_TYPE_Q4_1 ? 20u : 18u;
                    for (size_t off = 0; off < plane; off += block_bytes) {
                        uint16_t d; std::memcpy(&d, dst + off, sizeof(d));
                        if ((d & 0x7c00u) == 0x7c00u) return false;
                        if (type_ == GGML_TYPE_Q4_1) {
                            uint16_t m; std::memcpy(&m, dst + off + 2, sizeof(m));
                            if ((m & 0x7c00u) == 0x7c00u) return false;
                        }
                    }
                }
                dst += plane;
            }
        }
        payload.swap(next);
        rows_ += count;
        return true;
    }
    bool can_rewind(uint32_t keep) const override { return keep <= rows_; }
    bool rewind_tail(uint32_t, std::vector<uint8_t> &) const override { return false; }
    bool rewind(uint32_t keep) override {
        if (!can_rewind(keep)) return false;
        cuda_.invalidate();
        rows_ = keep;
        return true;
    }
private:
    rkva_low4_cuda_prepared cuda_;
    uint32_t origin_, capacity_, heads_, rows_ = 0;
    ggml_type type_;
};
