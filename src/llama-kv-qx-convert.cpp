#include "llama-kv-qx-convert.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace llama_kv_qx {

const char * status_name(status st) {
    switch (st) {
        case status::ok:                   return "ok";
        case status::invalid_args:         return "invalid_args";
        case status::invalid_type:         return "invalid_type";
        case status::invalid_geometry:     return "invalid_geometry";
        case status::invalid_domain:       return "invalid_domain";
        case status::unsupported_swa:      return "unsupported_swa";
        case status::unsupported_multi_stream: return "unsupported_multi_stream";
        case status::payload_overflow:     return "payload_overflow";
        case status::non_finite_input:     return "non_finite_input";
        case status::callback_failure:     return "callback_failure";
        case status::sink_failure:         return "sink_failure";
    }
    return "unknown";
}

namespace {

constexpr uint32_t KVAR_N_GROUP = 128;
constexpr uint32_t DEFAULT_BLOCK = 64;

// Normalized in-place WHT of width 64 or 128. Exact equations of the KVarN
// rotation (ggml_compute_forward_kvarn_wht / llama_kvarn_hadamard_*).
bool hadamard_n(float * values, uint32_t width) {
    if (values == nullptr || (width != 64 && width != KVAR_N_GROUP)) {
        return false;
    }
    for (uint32_t stride = 1; stride < width; stride *= 2) {
        for (uint32_t base = 0; base < width; base += 2 * stride) {
            for (uint32_t i = 0; i < stride; ++i) {
                const float a = values[base + i];
                const float b = values[base + stride + i];
                values[base + i] = a + b;
                values[base + stride + i] = a - b;
            }
        }
    }
    const float scale = width == 64 ? 0.125f : 0.08838834764831845f;
    for (uint32_t i = 0; i < width; ++i) {
        values[i] *= scale;
    }
    return true;
}

// Canonical KVarN head transform: WHT-128 per slice, then mixing across
// slices (self-inverse). This is the normalized WHT of width head_dim.
bool wht_head(float * values, uint32_t head_dim, uint32_t record_dim) {
    if (values == nullptr) {
        return false;
    }
    if (record_dim == 64) {
        return head_dim == 64 && hadamard_n(values, 64);
    }
    if (record_dim != KVAR_N_GROUP || head_dim % KVAR_N_GROUP != 0) {
        return false;
    }
    const uint32_t slices = head_dim / KVAR_N_GROUP;
    if (slices < 1 || slices > 4) {
        return false;
    }
    for (uint32_t s = 0; s < slices; ++s) {
        if (!hadamard_n(values + s * KVAR_N_GROUP, KVAR_N_GROUP)) {
            return false;
        }
    }
    if (slices == 1) {
        return true;
    }
    const float scale = slices == 2 ? 0.7071067811865475f : 0.5f;
    for (uint32_t d = 0; d < KVAR_N_GROUP; ++d) {
        float x[4] = {};
        for (uint32_t s = 0; s < slices; ++s) {
            x[s] = values[s * KVAR_N_GROUP + d];
        }
        for (uint32_t stride = 1; stride < slices; stride <<= 1) {
            for (uint32_t base = 0; base < slices; base += 2 * stride) {
                for (uint32_t i = 0; i < stride; ++i) {
                    const float a = x[base + i];
                    const float b = x[base + stride + i];
                    x[base + i] = a + b;
                    x[base + stride + i] = a - b;
                }
            }
        }
        for (uint32_t s = 0; s < slices; ++s) {
            values[s * KVAR_N_GROUP + d] = x[s] * scale;
        }
    }
    return true;
}

// Standard normalized WHT of width w (64/128/256/512), the same transform
// family as the KVarN rot matrices / ggml_gen_hadamard.
bool wht_block(float * values, uint32_t width) {
    if (width == 64) {
        return hadamard_n(values, 64);
    }
    return wht_head(values, width, KVAR_N_GROUP);
}

bool valid_head_dim(uint32_t head_dim) {
    return head_dim == 64 || head_dim == 128 || head_dim == 256 || head_dim == 512;
}

bool valid_wht_width(uint32_t width) {
    return width == 0 || valid_head_dim(width);
}

// Explicit whitelist BEFORE touching ggml_type_traits: an arbitrary int
// (negative, GGML_TYPE_COUNT+1, UINT_MAX, or another 32-wide type such as
// Q5_1) must be rejected without enum-UB or indexing trait tables.
bool supported_dst_type(int32_t type) {
    return type == (int32_t) GGML_TYPE_Q4_0 || type == (int32_t) GGML_TYPE_Q5_0 ||
           type == (int32_t) GGML_TYPE_Q6_0 || type == (int32_t) GGML_TYPE_Q8_0;
}

status validate_config(const config & cfg) {
    if (cfg.n_streams != 1) {
        return status::unsupported_multi_stream;
    }
    if (cfg.swa) {
        return status::unsupported_swa;
    }
    if (!valid_head_dim(cfg.head_dim) || cfg.n_heads == 0) {
        return status::invalid_geometry;
    }
    if (cfg.src.record_dim != 64 && cfg.src.record_dim != KVAR_N_GROUP) {
        return status::invalid_geometry;
    }
    if (cfg.src.record_dim == 64 && cfg.head_dim != 64) {
        return status::invalid_geometry;
    }
    if (cfg.src.record_dim == KVAR_N_GROUP &&
            (cfg.head_dim % KVAR_N_GROUP != 0 || cfg.head_dim / KVAR_N_GROUP > 4)) {
        return status::invalid_geometry;
    }
    if (!valid_wht_width(cfg.dst.wht_width)) {
        return status::invalid_domain;
    }
    if (cfg.dst.wht_width > 0 && cfg.head_dim % cfg.dst.wht_width != 0) {
        return status::invalid_domain;
    }
    if (!supported_dst_type(cfg.dst_type)) {
        return status::invalid_type;
    }
    const ggml_type_traits * traits = ggml_get_type_traits((ggml_type) cfg.dst_type);
    if (traits == nullptr || !traits->is_quantized ||
            traits->blck_size != 32 || traits->type_size == 0) {
        return status::invalid_type;
    }
    const uint64_t row_floats = uint64_t(cfg.head_dim) * uint64_t(cfg.n_heads);
    if (row_floats % uint64_t(traits->blck_size) != 0) {
        return status::invalid_geometry;
    }
    return status::ok;
}

bool checked_payload_bytes(const config & cfg, uint32_t n_tokens, size_t & out) {
    const ggml_type_traits * traits = ggml_get_type_traits((ggml_type) cfg.dst_type);
    if (traits == nullptr || traits->blck_size <= 0 || traits->type_size == 0) {
        return false;
    }
    const uint64_t row_floats = uint64_t(cfg.head_dim) * uint64_t(cfg.n_heads);
    if (row_floats % uint64_t(traits->blck_size) != 0) {
        return false;
    }
    const uint64_t row = (row_floats / uint64_t(traits->blck_size)) * uint64_t(traits->type_size);
    if (row == 0) {
        return false;
    }
    // Division-based overflow checks (portable, no wrapping multiply):
    // n_tokens * row must not overflow uint64 and must fit size_t (relevant
    // on 32-bit hosts).
    const uint64_t max_u64 = std::numeric_limits<uint64_t>::max();
    if (uint64_t(n_tokens) > max_u64 / row) {
        return false;
    }
    const uint64_t total = uint64_t(n_tokens) * row;
    if (total > uint64_t(std::numeric_limits<size_t>::max())) {
        return false;
    }
    out = size_t(total);
    return true;
}

} // namespace

size_t row_bytes(const config & cfg) {
    if (validate_config(cfg) != status::ok) {
        return 0;
    }
    size_t bytes = 0;
    return checked_payload_bytes(cfg, 1, bytes) ? bytes : 0;
}

size_t payload_bytes(const config & cfg, uint32_t n_tokens) {
    if (validate_config(cfg) != status::ok || n_tokens == 0) {
        return 0;
    }
    size_t bytes = 0;
    return checked_payload_bytes(cfg, n_tokens, bytes) ? bytes : 0;
}

// Shared per-window validation: undoes the KVarN source rotation, applies
// the destination rotation and checks finiteness of the WHOLE window. The
// window buffer is overwritten in place with the destination-domain rows.
static status transform_window(
        const config & cfg, uint32_t n, uint64_t row_floats, float * window) {
    for (uint32_t t = 0; t < n; ++t) {
        for (uint32_t h = 0; h < cfg.n_heads; ++h) {
            float * head = window + size_t(h) * cfg.head_dim +
                    size_t(t) * size_t(row_floats);
            if (cfg.src.rotated) {
                if (!wht_head(head, cfg.head_dim, cfg.src.record_dim)) {
                    return status::invalid_domain;
                }
            }
            if (cfg.dst.wht_width > 0) {
                const uint32_t blocks = cfg.head_dim / cfg.dst.wht_width;
                for (uint32_t b = 0; b < blocks; ++b) {
                    if (!wht_block(head + size_t(b) * cfg.dst.wht_width,
                            cfg.dst.wht_width)) {
                        return status::invalid_domain;
                    }
                }
            }
        }
        for (size_t i = 0; i < size_t(row_floats); ++i) {
            if (!std::isfinite(window[size_t(t) * size_t(row_floats) + i])) {
                return status::non_finite_input;
            }
        }
    }
    return status::ok;
}

result convert_to_sink(
        const config & cfg,
        uint32_t       n_tokens,
        read_block_fn  read_block,
        void         * read_user,
        sink_fn        sink,
        void         * sink_user) {
    result out = {};
    if (read_block == nullptr || sink == nullptr) {
        out.st = status::invalid_args;
        return out;
    }
    if (n_tokens == 0) {
        out.st = status::invalid_args;
        return out;
    }

    out.st = validate_config(cfg);
    if (out.st != status::ok) {
        return out;
    }
    size_t required = 0;
    if (!checked_payload_bytes(cfg, n_tokens, required)) {
        out.st = status::payload_overflow;
        return out;
    }
    const size_t row = required / n_tokens;

    const uint32_t block_cap = std::min(n_tokens,
            cfg.max_tokens_per_block == 0 ? DEFAULT_BLOCK : cfg.max_tokens_per_block);
    const uint64_t row_floats = uint64_t(cfg.head_dim) * uint64_t(cfg.n_heads);

    // Bounded scratch sizing must be checked before any allocation and
    // before the multiplies themselves: block_cap * row_floats can wrap
    // uint64 (e.g. 2^24 windows x 2^40 floats = 2^64) even when the payload
    // math is fine (Q4_0 bytes per float < 1). Guard by division first,
    // then check the size_t/float budget for the allocation.
    const uint64_t max_u64 = std::numeric_limits<uint64_t>::max();
    const uint64_t max_win_floats =
            uint64_t(std::numeric_limits<size_t>::max()) / sizeof(float);
    if (block_cap > 0 && row_floats > max_u64 / uint64_t(block_cap)) {
        out.st = status::payload_overflow;
        return out;
    }
    const uint64_t win_floats = uint64_t(block_cap) * row_floats;
    if (win_floats > max_win_floats) {
        out.st = status::payload_overflow;
        return out;
    }
    // Quantized block scratch: block_cap * row quantized bytes, guarded.
    if (block_cap > 0 && uint64_t(row) > max_u64 / uint64_t(block_cap)) {
        out.st = status::payload_overflow;
        return out;
    }
    const uint64_t block_bytes = uint64_t(block_cap) * uint64_t(row);
    if (block_bytes > uint64_t(std::numeric_limits<size_t>::max())) {
        out.st = status::payload_overflow;
        return out;
    }
    std::vector<float> window;
    std::vector<uint8_t> qwindow;
    try {
        window.assign(size_t(win_floats), 0.0f);
        qwindow.assign(size_t(block_bytes), 0);
    } catch (const std::exception &) {
        out.st = status::payload_overflow; // workspace allocation failed
        return out;
    }

    uint32_t t0 = 0;
    while (t0 < n_tokens) {
        const uint32_t n = std::min(block_cap, n_tokens - t0);
        block_request req = { t0, n, cfg.head_dim, cfg.n_heads, cfg.value, window.data() };
        bool served = false;
        try {
            served = read_block(read_user, req);
        } catch (const std::exception &) {
            // The callback contract is non-throwing; an escaping exception is
            // a callback failure. Nothing of the failing window is published.
            out.st = status::callback_failure;
            out.tokens_completed = t0;
            return out;
        }
        if (!served) {
            out.st = status::callback_failure;
            out.tokens_completed = t0;
            return out;
        }

        // Validate the whole window before emitting any of its bytes.
        out.st = transform_window(cfg, n, row_floats, window.data());
        if (out.st != status::ok) {
            out.tokens_completed = t0;
            return out;
        }

        // Quantize the window into the bounded block scratch.
        for (uint32_t t = 0; t < n; ++t) {
            ggml_quantize_chunk((ggml_type) cfg.dst_type,
                    window.data() + size_t(t) * size_t(row_floats),
                    qwindow.data() + size_t(t) * row, 0, 1, int64_t(row_floats), nullptr);
        }

        // Emit the block (logical row order; the sink owns staging/publication).
        sink_block block = { t0, n, qwindow.data(), row, cfg.head_dim, cfg.n_heads, cfg.value };
        bool accepted = false;
        try {
            accepted = sink(sink_user, block);
        } catch (const std::exception &) {
            out.st = status::sink_failure;
            out.tokens_completed = t0;
            return out;
        }
        if (!accepted) {
            out.st = status::sink_failure;
            out.tokens_completed = t0;
            return out;
        }
        t0 += n;
    }

    out.st = status::ok;
    out.payload_bytes = required;
    out.tokens_completed = n_tokens;
    return out;
}

result convert(
        const config & cfg,
        uint32_t       n_tokens,
        read_block_fn  read_block,
        void         * user,
        void         * payload,
        size_t         payload_capacity) {
    result out = {};

    if (read_block == nullptr) {
        out.st = status::invalid_args;
        return out;
    }
    if (n_tokens == 0) {
        out.st = status::invalid_args;
        return out;
    }
    if (payload == nullptr || payload_capacity == 0) {
        out.st = status::invalid_args;
        return out;
    }

    out.st = validate_config(cfg);
    if (out.st != status::ok) {
        return out;
    }
    size_t required = 0;
    if (!checked_payload_bytes(cfg, n_tokens, required)) {
        out.st = status::payload_overflow;
        return out;
    }
    if (payload_capacity < required) {
        out.st = status::payload_overflow;
        return out;
    }

    // Full-payload staged buffer: the internal writer sink fills it in
    // logical row order. The capacity check ran before any callback.
    struct writer {
        uint8_t * dst;
    };
    writer w = { static_cast<uint8_t *>(payload) };
    auto write_sink = [](void * u, const sink_block & block) -> bool {
        writer * w = static_cast<writer *>(u);
        std::memcpy(w->dst + size_t(block.t0) * block.row_bytes,
                block.data, size_t(block.n_tokens) * block.row_bytes);
        return true;
    };
    return convert_to_sink(cfg, n_tokens, read_block, user, write_sink, &w);
}

} // namespace llama_kv_qx
