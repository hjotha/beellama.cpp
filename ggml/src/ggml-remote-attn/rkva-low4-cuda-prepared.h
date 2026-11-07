#pragma once
#include "ggml-backend.h"
#include "ggml-low4-native-producer.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

// Transactional prepack storage shared by the four format-specific owners.
// Host snapshots bind append chunks to the prepared parent. A failed prepare,
// rewind, or CUDA call cannot expose stale packed data. CPU is still the default.
class rkva_low4_cuda_prepared {
  public:
    void configure(const char * env, int format, uint32_t heads, size_t record) {
#if defined(__AVX2__)
        flags_ = format == GGML_TYPE_SAW_INT4 ? 1u : 0u;
#endif
        format_           = format;
        heads_            = heads;
        record_           = record;
        const char * mode = std::getenv(env);
        if (!mode) {
            mode = std::getenv("GGML_RKVA_NATIVE_PRODUCER");
            if (mode && !std::strcmp(mode, "cuda")) {
                if (format == GGML_TYPE_SAW_INT4 || format == GGML_TYPE_TQNC_K) {
                    // Historical SAW/NC owners ignored the generic CUDA switch.
                    // Preserve their CPU behavior; only the specific switch opts in.
                    std::fprintf(stderr, "RKVA: CPU producer retained; CUDA requires %s=cuda explicitly\n", env);
                    return;
                }
                throw std::invalid_argument("native low4 CUDA requires GGML_RKVA_LOW4_PRODUCER=cuda explicitly");
            }
        }
        if (!mode || !std::strcmp(mode, "cpu")) {
            return;
        }
        if (std::strcmp(mode, "cuda")) {
            throw std::invalid_argument("native low4 producer must be cpu or cuda");
        }
        auto reg = ggml_backend_reg_by_name("CUDA");
        cuda_    = reg ? reinterpret_cast<ggml_backend_cuda_low4_native_prepare_v1_t>(
                          ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_low4_native_prepare_v1")) :
                         nullptr;
        if (!cuda_) {
            throw std::runtime_error("native low4 CUDA producer unavailable");
        }
        if (!heads || heads > 32) {
            throw std::invalid_argument("native low4 CUDA producer requires 1..32 heads");
        }
    }

    explicit operator bool() const { return cuda_ != nullptr; }

    void invalidate() { count_ = 0; }

    bool prepare(const float *   k,
                 const float *   v,
                 const int32_t * pos,
                 uint32_t        n,
                 uint32_t        origin,
                 uint32_t        rows,
                 uint32_t        capacity,
                 const float *   dk,
                 const float *   dv) {
        if (!cuda_) {
            return true;
        }
        invalidate();
        if (!k || !v || !pos || !n || n > 512 || rows > capacity || n > capacity - rows) {
            return false;
        }
        for (uint32_t i = 0; i < n; ++i) {
            if (pos[i] != int64_t(origin) + rows + i) {
                return false;
            }
        }
        const size_t elements = size_t(n) * heads_ * 256;
        for (const float * p : { k, v }) {
            for (size_t i = 0; i < elements; ++i) {
                if (!std::isfinite(p[i]) ||
                    ((format_ == GGML_TYPE_Q4_1 || format_ == GGML_TYPE_IQ4_NL) && std::fabs(p[i]) > 65504)) {
                    return false;
                }
            }
        }
        keys_.assign(k, k + elements);
        values_.assign(v, v + elements);
        pk_.resize(size_t(n) * heads_ * record_);
        pv_.resize(pk_.size());
        if (!cuda_(format_, flags_, dk ? dk : k, dv ? dv : v, n, heads_, pk_.data(), pv_.data())) {
            return false;
        }
        begin_ = rows;
        count_ = n;
        return true;
    }

    bool copy(const float * k, const float * v, uint32_t rows, uint32_t n, uint8_t * dst) const {
        if (!cuda_ || !count_ || rows < begin_ || rows - begin_ > count_ || n > count_ - (rows - begin_)) {
            return false;
        }
        const size_t offset = rows - begin_, elements = size_t(n) * heads_ * 256;
        if (std::memcmp(k, keys_.data() + offset * heads_ * 256, elements * sizeof(float)) ||
            std::memcmp(v, values_.data() + offset * heads_ * 256, elements * sizeof(float))) {
            return false;
        }
        const size_t plane = size_t(n) * heads_ * record_;
        std::memcpy(dst, pk_.data() + offset * heads_ * record_, plane);
        std::memcpy(dst + plane, pv_.data() + offset * heads_ * record_, plane);
        return true;
    }
  private:
    int                                        format_ = 0;
    uint32_t                                   flags_  = 0;
    uint32_t                                   heads_ = 0, begin_ = 0, count_ = 0;
    size_t                                     record_ = 0;
    ggml_backend_cuda_low4_native_prepare_v1_t cuda_   = nullptr;
    std::vector<float>                         keys_, values_;
    std::vector<uint8_t>                       pk_, pv_;
};
