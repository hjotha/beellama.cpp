#pragma once

#include "../ggml-turboquant-nc.h"
#include "remote-attn-protocol.h"
#include "rkva-native-owner.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

// Layout 8 stores independent token-major K and V planes. The host K block is
// 130 bytes; its wire record has two zero pad bytes for D3D12 4-byte loads.
class rkva_turboquant_nc_producer : public rkva_native_owner {
public:
    static constexpr uint32_t head_dim = 256;
    static constexpr uint32_t record_bytes = 132;

    rkva_turboquant_nc_producer(uint32_t origin, uint32_t capacity, uint32_t heads)
        : origin_(origin), capacity_(capacity), heads_(heads) {
        const uint16_t endian = 1;
        if (!heads || origin % 256 || !capacity || uint64_t(origin) + capacity > INT32_MAX ||
                *reinterpret_cast<const uint8_t *>(&endian) != 1) {
            throw std::invalid_argument("native TurboQuant NC range geometry or byte order");
        }
    }

    static size_t row_bytes(uint32_t heads) { return size_t(heads)*record_bytes; }
    uint32_t rows() const override { return rows_; }

    bool prepare(const float *, const float *, const int32_t *, uint32_t,
                 const float * = nullptr, const float * = nullptr) override { return true; }

    bool append(const float * keys, const float * values, const int32_t * positions,
                uint32_t count, bool reset_snapshot, std::vector<uint8_t> & payload) override {
        if (!keys || !values || !positions || !count || rows_ > capacity_ ||
                count > capacity_ - rows_ || reset_snapshot != (rows_ == 0)) return false;
        for (uint32_t i = 0; i < count; ++i) {
            if (positions[i] != int64_t(origin_) + rows_ + i) return false;
        }
        const uint64_t plane64 = uint64_t(count)*heads_*record_bytes;
        if (plane64 > (uint64_t(256u << 20) - sizeof(rkva_kvarn_range_upload))/2 ||
                plane64 > SIZE_MAX/2) return false;
        const size_t plane = size_t(plane64);
        std::vector<uint8_t> next(sizeof(rkva_kvarn_range_upload) + 2*plane, 0);
        const uint32_t after = rows_ + count;
        const rkva_kvarn_range_upload header {
            after, 0u, 0u, reset_snapshot ? RKVA_KVARN_UPLOAD_RESET : 0u,
            rows_, count, {0u, 0u}};
        std::memcpy(next.data(), &header, sizeof(header));
        uint8_t * kdst = next.data() + sizeof(header);
        uint8_t * vdst = kdst + plane;
        const size_t n_heads = size_t(count)*heads_;
        for (size_t i = 0; i < n_heads; ++i) {
            ggml_tqnc_key key;
            ggml_tqnc_value value;
            if (!ggml_tqnc_quantize_key(keys + i*head_dim, &key) ||
                    !ggml_tqnc_quantize_value(values + i*head_dim, &value)) return false;
            std::memcpy(kdst + i*record_bytes, &key, sizeof(key));
            std::memcpy(vdst + i*record_bytes, &value, sizeof(value));
        }
        payload.swap(next);
        rows_ = after;
        return true;
    }

    bool can_rewind(uint32_t keep) const override { return keep <= rows_; }
    bool rewind_tail(uint32_t, std::vector<uint8_t> &) const override { return false; }
    bool rewind(uint32_t keep) override {
        if (keep > rows_) return false;
        rows_ = keep;
        return true;
    }

private:
    uint32_t origin_, capacity_, heads_, rows_ = 0;
};

static_assert(sizeof(ggml_tqnc_key) == 130, "TurboQuant NC host key ABI");
static_assert(sizeof(ggml_tqnc_value) == 132, "TurboQuant NC host value ABI");
