// Standalone CPU test for llama_kv_qx::convert (KVarN -> Q4_0/Q5_0/Q6_0/Q8_0
// bounded conversion).
//
// The KVarN source is built with the REAL canonical CPU operators
// (ggml_kvarn_store + ggml_kvarn_materialize) on the CPU backend, so the
// records/sink/stages/intrinsic-tail resolution is the production one -- not
// random floats. The payload produced by the converter for that source is
// compared bit-exactly against an independent reference that de-rotates the
// SAME materialized KVarN source with the canonical WHT equation, applies the
// declared destination rotation, and quantizes with ggml_quantize_chunk.
// Conversion error is additionally measured against floats reconstructed
// from the same KVarN source (Q8 does not recover precision already lost in
// KVarN).
//
// Build (CPU only, links the immutable baseline libraries):
//   g++ -O2 -std=c++17 -I ggml/include -I src tests/test-kv-qx-convert.cpp
//       src/llama-kv-qx-convert.cpp src/llama-kv-kvarn-materializer.cpp
//       -L <baseline-bin> -lggml-cpu -lggml-base -lgomp -lpthread
//       -o <out>/test-kv-qx-convert
//   LD_LIBRARY_PATH=<baseline-bin> <out>/test-kv-qx-convert
//
// It does not touch llama contexts or ABI structs under edit; it only uses
// stable ggml APIs plus this module's own header.

#include "llama-kv-qx-convert.h"
#include "llama-kv-kvarn-materializer.h"
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using llama_kv_qx::config;
using llama_kv_qx::convert;
using llama_kv_qx::convert_to_sink;
using llama_kv_qx::sink_block;
using llama_kv_qx::status;
using llama_kv_qx::block_request;

int g_failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); \
        ++g_failures; \
    } \
} while (0)

#define CHECK_EQ(a, b, msg) do { \
    if (!((a) == (b))) { \
        std::fprintf(stderr, "FAIL %s:%d: %s (got %s, want %s)\n", __FILE__, __LINE__, msg, \
                std::to_string((long long) (a)).c_str(), std::to_string((long long) (b)).c_str()); \
        ++g_failures; \
    } \
} while (0)

// ---------------------------------------------------------------------------
// Canonical KVarN WHT equations (independent copy of the production ones from
// ggml/src/ggml-cpu/ops.cpp and src/llama-kvarn.cpp). The normalized WHT is
// self-inverse, so the same function de-rotates the rotated materialized
// domain.
// ---------------------------------------------------------------------------

void ref_hadamard_n(float * values, int width) {
    for (int stride = 1; stride < width; stride *= 2) {
        for (int base = 0; base < width; base += 2 * stride) {
            for (int i = 0; i < stride; ++i) {
                const float a = values[base + i];
                const float b = values[base + stride + i];
                values[base + i] = a + b;
                values[base + stride + i] = a - b;
            }
        }
    }
    const float scale = width == 64 ? 0.125f : 0.08838834764831845f;
    for (int i = 0; i < width; ++i) {
        values[i] *= scale;
    }
}

void ref_wht_head(float * values, int head_dim, int record_dim) {
    if (record_dim == 64) {
        ref_hadamard_n(values, 64);
        return;
    }
    const int slices = head_dim / 128;
    for (int s = 0; s < slices; ++s) {
        ref_hadamard_n(values + s * 128, 128);
    }
    if (slices == 1) {
        return;
    }
    const float scale = slices == 2 ? 0.7071067811865475f : 0.5f;
    for (int d = 0; d < 128; ++d) {
        float x[4] = {};
        for (int s = 0; s < slices; ++s) {
            x[s] = values[s * 128 + d];
        }
        for (int stride = 1; stride < slices; stride <<= 1) {
            for (int base = 0; base < slices; base += 2 * stride) {
                for (int i = 0; i < stride; ++i) {
                    const float a = x[base + i];
                    const float b = x[base + stride + i];
                    x[base + i] = a + b;
                    x[base + stride + i] = a - b;
                }
            }
        }
        for (int s = 0; s < slices; ++s) {
            values[s * 128 + d] = x[s] * scale;
        }
    }
}

// ---------------------------------------------------------------------------
// Index encodings: canonical formulas from src/llama-kvarn.h (kept local so
// the test stays header-light; they are validated by the materializer's own
// stage/record resolution).
// ---------------------------------------------------------------------------

int64_t encode_store_cell(uint32_t cell, uint32_t slot) {
    return int64_t((uint64_t(slot) + 1u) << 32u | uint64_t(cell));
}

int64_t encode_stage_cell(uint32_t cell, uint32_t slot) {
    return -encode_store_cell(cell, slot) - 2;
}

// ---------------------------------------------------------------------------
// Real KVarN source built with the canonical ggml CPU operators.
// ---------------------------------------------------------------------------

struct kvarn_source {
    ggml_backend_t backend = nullptr;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * current = nullptr;
    ggml_tensor * indices = nullptr;
    ggml_tensor * stage = nullptr;
    ggml_tensor * records = nullptr;
    ggml_tensor * stored = nullptr;
    ggml_tensor * full_mat = nullptr;

    uint32_t n_tokens = 0;
    uint32_t head_dim = 0;      // full head width (record_dim * head_slices)
    uint32_t n_heads = 0;       // logical heads (physical = n_heads * head_slices)
    uint32_t record_dim = 128;  // KVarN record width
    uint32_t head_slices = 1;
    int bits = 4;
    bool value = false;
    int stage_groups = 3;

    std::vector<ggml_fp16_t> mat_f16;              // [record_dim, phys_heads, n_tokens]
    std::vector<float> rotated;                    // [head_dim, n_heads, n_tokens] F32
    std::vector<float> input_logical;              // [head_dim, n_heads, n_tokens] F32 pre-KVarN
    std::vector<float> orig;                       // de-rotated same-source floats
};

void build_source(kvarn_source & s, uint32_t n_tokens, uint32_t head_dim,
        uint32_t n_heads_logical, int bits, bool value) {
    s.n_tokens = n_tokens;
    s.head_dim = head_dim;
    s.n_heads = n_heads_logical;
    s.bits = bits;
    s.value = value;
    s.head_slices = head_dim == 64 ? 1 : head_dim / 128;
    s.record_dim = s.head_slices == 1 && head_dim == 64 ? 64 : 128;
    const uint32_t phys_heads = n_heads_logical * s.head_slices;
    const uint32_t stage_cap = 128 * uint32_t(s.stage_groups);
    const uint32_t n_groups = (n_tokens + 127) / 128;

    const int64_t rows = value ? 128 : s.record_dim;
    const int64_t cols = value ? s.record_dim : 128;
    const size_t record_bytes = (size_t(rows) * cols * size_t(bits) + 7) / 8 +
        (size_t(2 * rows + cols)) * sizeof(ggml_fp16_t);

    ggml_init_params params = { 16 * 1024 * 1024, nullptr, true };
    s.ctx = ggml_init(params);
    CHECK(s.ctx != nullptr, "ggml_init failed");

    s.current = ggml_new_tensor_3d(s.ctx, GGML_TYPE_F32, s.record_dim, phys_heads, n_tokens);
    s.indices = ggml_new_tensor_1d(s.ctx, GGML_TYPE_I64, n_tokens);
    s.stage = ggml_new_tensor_3d(s.ctx, GGML_TYPE_F16, s.record_dim, phys_heads, stage_cap);
    s.records = ggml_new_tensor_3d(s.ctx, GGML_TYPE_I8, record_bytes, phys_heads, n_groups);
    CHECK(s.current != nullptr && s.indices != nullptr && s.stage != nullptr &&
            s.records != nullptr, "tensor creation failed");

    s.stored = ggml_kvarn_store(s.ctx, s.current, s.indices, s.stage, s.records,
            bits, 16, value, s.stage_groups);
    s.stored->op_params[5] = (int32_t) s.head_slices;
    s.stored->op_params[9] = 1; // eager records, as the production cache does
    s.full_mat = ggml_kvarn_materialize(s.ctx, s.records, s.stored, s.indices,
            n_tokens, 0, 1, bits, value, s.stage_groups);
    s.full_mat->op_params[4] = 1; // rotated domain (production default)
    s.full_mat->op_params[5] = (int32_t) s.head_slices;
    s.full_mat->op_params[9] = 1;

    ggml_cgraph * graph = ggml_new_graph(s.ctx);
    ggml_build_forward_expand(graph, s.full_mat);

    s.backend = ggml_backend_cpu_init();
    CHECK(s.backend != nullptr, "cpu backend init failed");
    s.buffer = ggml_backend_alloc_ctx_tensors(s.ctx, s.backend);
    CHECK(s.buffer != nullptr, "tensor allocation failed");

    const size_t n_elem = size_t(s.record_dim) * phys_heads * n_tokens;
    std::vector<float> input(n_elem);
    for (uint32_t t = 0; t < n_tokens; ++t) {
        for (uint32_t h = 0; h < phys_heads; ++h) {
            for (uint32_t d = 0; d < s.record_dim; ++d) {
                const int idx = int((h * 31 + d * 13 + t * 17) % 31 - 15);
                input[size_t(d) + size_t(h) * s.record_dim + size_t(t) * s.record_dim * phys_heads] =
                    std::sin(float(d) * 0.071f + float(h) * 0.31f + float(t) * 0.013f) +
                    std::cos(float(d) * 0.017f + float(h) * 0.11f + float(t) * 0.037f) +
                    float(idx) * 0.01f;
            }
        }
    }
    std::vector<int64_t> idx(n_tokens);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        idx[i] = i;
    }
    std::vector<uint8_t> zeros(ggml_nbytes(s.stage) + ggml_nbytes(s.records), 0);

    ggml_backend_tensor_set(s.current, input.data(), 0, ggml_nbytes(s.current));
    ggml_backend_tensor_set(s.indices, idx.data(), 0, ggml_nbytes(s.indices));
    ggml_backend_tensor_set(s.stage, zeros.data(), 0, ggml_nbytes(s.stage));
    ggml_backend_tensor_set(s.records, zeros.data(), 0, ggml_nbytes(s.records));

    CHECK(ggml_backend_graph_compute(s.backend, graph) == GGML_STATUS_SUCCESS,
            "store+materialize graph compute failed");

    s.mat_f16.resize(ggml_nelements(s.full_mat));
    ggml_backend_tensor_get(s.full_mat, s.mat_f16.data(), 0, ggml_nbytes(s.full_mat));

    // Reshape the materializer output [record_dim, phys_heads, n_tokens] into
    // the attention-path layout [head_dim, n_heads, n_tokens] (slice-major),
    // exactly as llama_kv_cache_kvarn::materialize does.
    s.rotated.assign(size_t(s.head_dim) * s.n_heads * n_tokens, 0.0f);
    s.input_logical.assign(s.rotated.size(), 0.0f);
    for (uint32_t t = 0; t < n_tokens; ++t) {
        for (uint32_t l = 0; l < s.n_heads; ++l) {
            for (uint32_t d = 0; d < s.head_dim; ++d) {
                const uint32_t slice = s.head_slices == 1 ? 0 : d / s.record_dim;
                const uint32_t lane = s.head_slices == 1 ? d : d % s.record_dim;
                const uint32_t ph = l * s.head_slices + slice;
                const size_t src_off = size_t(lane) + size_t(ph) * s.record_dim +
                        size_t(t) * s.record_dim * phys_heads;
                const size_t dst_off = size_t(d) + size_t(l) * s.head_dim +
                        size_t(t) * s.head_dim * s.n_heads;
                s.rotated[dst_off] = ggml_fp16_to_fp32(s.mat_f16[src_off]);
                s.input_logical[dst_off] = input[src_off];
            }
        }
    }

    // Reference de-rotation: canonical inverse of the KVarN domain applied to
    // the SAME materialized source.
    s.orig = s.rotated;
    for (uint32_t t = 0; t < n_tokens; ++t) {
        for (uint32_t l = 0; l < s.n_heads; ++l) {
            ref_wht_head(s.orig.data() + size_t(l) * s.head_dim +
                    size_t(t) * s.head_dim * s.n_heads, (int) s.head_dim, (int) s.record_dim);
        }
    }
}

void free_source(kvarn_source & s) {
    if (s.buffer) {
        ggml_backend_buffer_free(s.buffer);
    }
    if (s.ctx) {
        ggml_free(s.ctx);
    }
    if (s.backend) {
        ggml_backend_free(s.backend);
    }
}

// ---------------------------------------------------------------------------
// Reference payload: de-rotate + destination rotation + canonical Qx
// quantization of the SAME KVarN source, rows per token, head-major.
// ---------------------------------------------------------------------------

struct reference_payload {
    std::vector<float> rows;    // [n_tokens][head_dim*n_heads] destination-domain floats
    std::vector<uint8_t> data;  // quantized payload
};

void build_reference(const kvarn_source & s, ggml_type qx, uint32_t dst_wht,
        reference_payload & ref) {
    const size_t row_floats = size_t(s.head_dim) * s.n_heads;
    ref.rows.assign(size_t(s.n_tokens) * row_floats, 0.0f);
    for (uint32_t t = 0; t < s.n_tokens; ++t) {
        for (uint32_t l = 0; l < s.n_heads; ++l) {
            const size_t src_off = size_t(l) * s.head_dim + size_t(t) * row_floats;
            const size_t row_off = size_t(t) * row_floats + size_t(l) * s.head_dim;
            std::memcpy(ref.rows.data() + row_off, s.orig.data() + src_off,
                    s.head_dim * sizeof(float));
            if (dst_wht > 0) {
                for (uint32_t b = 0; b < s.head_dim / dst_wht; ++b) {
                    ref_wht_head(ref.rows.data() + row_off + size_t(b) * dst_wht,
                            (int) dst_wht, dst_wht == 64 ? 64 : 128);
                }
            }
        }
    }
    const size_t payload = size_t(s.n_tokens) * row_floats / 32 *
        ggml_type_size(qx);
    ref.data.assign(payload, 0);
    const size_t written = ggml_quantize_chunk(qx, ref.rows.data(), ref.data.data(),
            0, s.n_tokens, int64_t(row_floats), nullptr);
    CHECK_EQ(written, payload, "reference quantize size");
}

// Dequantize a Qx payload into floats (traits canonical to_float, one block).
void dequant(ggml_type qx, const uint8_t * payload, size_t n_tokens,
        size_t row_floats, std::vector<float> & out) {
    const ggml_type_traits * traits = ggml_get_type_traits(qx);
    const size_t blck = size_t(traits->blck_size);
    out.assign(n_tokens * row_floats, 0.0f);
    for (size_t t = 0; t < n_tokens; ++t) {
        const uint8_t * row = payload + t * (row_floats / blck) * traits->type_size;
        for (size_t b = 0; b < row_floats / blck; ++b) {
            traits->to_float(row + b * traits->type_size,
                    out.data() + t * row_floats + b * blck, int64_t(blck));
        }
    }
}

double rmse(const std::vector<float> & a, const std::vector<float> & b) {
    double sum = 0.0;
    double max_abs = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double diff = double(a[i]) - double(b[i]);
        sum += diff * diff;
        max_abs = std::max(max_abs, std::fabs(diff));
    }
    return std::sqrt(sum / double(a.size()));
}

// ---------------------------------------------------------------------------
// Converter callback: serves windows from the (reshaped) full materialized
// KVarN source, optionally with a corrupted value injected.
// ---------------------------------------------------------------------------

struct window_source {
    const std::vector<float> * rotated = nullptr;
    uint32_t head_dim = 0;
    uint32_t n_heads = 0;
    int fail_after_calls = -1;
    int calls = 0;
    bool inject_nan = false;
    uint32_t nan_token = 0;
    uint32_t nan_head = 0;
    uint32_t nan_dim = 0;
};

bool serve_window(void * user, const block_request & req) {
    window_source * w = static_cast<window_source *>(user);
    ++w->calls;
    if (w->fail_after_calls >= 0 && w->calls > w->fail_after_calls) {
        return false;
    }
    const uint32_t n = req.n_tokens;
    for (uint32_t t = 0; t < n; ++t) {
        for (uint32_t h = 0; h < req.n_heads; ++h) {
            const size_t src_off = size_t(h) * req.head_dim +
                    size_t(req.t0 + t) * req.head_dim * req.n_heads;
            const size_t dst_off = size_t(h) * req.head_dim + size_t(t) * req.head_dim * req.n_heads;
            std::memcpy(req.out_f32 + dst_off, w->rotated->data() + src_off,
                    req.head_dim * sizeof(float));
            if (w->inject_nan && req.t0 + t == w->nan_token && h == w->nan_head) {
                req.out_f32[dst_off + w->nan_dim] = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Windowed read_indirect materialization: the bounded route a serving wrapper
// would use to materialize only the needed window through the index map. Its
// output must match the full materialization for the same cells.
// ---------------------------------------------------------------------------

void test_windowed_materialization(const kvarn_source & s) {
    const uint32_t n_tokens = s.n_tokens;
    const uint32_t phys_heads = s.n_heads * s.head_slices;
    const int tail_groups = s.stage_groups - 1;
    const uint32_t live_group = (n_tokens - 1) / 128;
    const uint32_t live_pos = (n_tokens - 1) % 128;
    const bool live_in_stage = !(live_pos == 127);

    struct window { uint32_t t0, n; };
    std::vector<window> windows;
    for (uint32_t t0 = 0; t0 < n_tokens; t0 += 64) {
        windows.push_back({ t0, std::min(64u, n_tokens - t0) });
    }

    for (const window & win : windows) {
        if (win.n == 0) {
            continue;
        }
        ggml_context * ctx = ggml_init({ 4 * 1024 * 1024, nullptr, true });
        CHECK(ctx != nullptr, "window ggml_init failed");
        ggml_tensor * win_indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, win.n);
        std::vector<int64_t> idx(win.n);
        for (uint32_t c = 0; c < win.n; ++c) {
            const uint32_t pos = win.t0 + c;
            const uint32_t g = pos / 128;
            if (g == 0) {
                idx[c] = encode_stage_cell(pos, 0);
            } else if (live_in_stage && g == live_group) {
                idx[c] = encode_stage_cell(pos, 1 + ((g - 1) % uint32_t(tail_groups)));
            } else {
                idx[c] = pos;
            }
        }
        ggml_tensor * win_mat = ggml_kvarn_materialize(ctx, s.records, s.stage, win_indices,
                win.n, 0, 1, s.bits, s.value, s.stage_groups);
        win_mat->op_params[4] = 1;
        win_mat->op_params[5] = (int32_t) s.head_slices;
        win_mat->op_params[9] = 1;
        win_mat->op_params[10] = 1; // read_indirect
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, win_mat);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, s.backend);
        CHECK(buf != nullptr, "window buffer alloc failed");
        ggml_backend_tensor_set(win_indices, idx.data(), 0, ggml_nbytes(win_indices));
        CHECK(ggml_backend_graph_compute(s.backend, graph) == GGML_STATUS_SUCCESS,
                "window materialize compute failed");

        std::vector<ggml_fp16_t> win_f16(ggml_nelements(win_mat));
        ggml_backend_tensor_get(win_mat, win_f16.data(), 0, ggml_nbytes(win_mat));
        int mism = 0;
        for (uint32_t c = 0; c < win.n; ++c) {
            const uint32_t pos = win.t0 + c;
            for (uint32_t h = 0; h < phys_heads; ++h) {
                for (uint32_t d = 0; d < s.record_dim; ++d) {
                    const size_t full_off = size_t(d) + size_t(h) * s.record_dim +
                            size_t(pos) * s.record_dim * phys_heads;
                    const size_t win_off = size_t(d) + size_t(h) * s.record_dim +
                            size_t(c) * s.record_dim * phys_heads;
                    if (win_f16[win_off] != s.mat_f16[full_off]) {
                        if (mism < 4) {
                            std::fprintf(stderr,
                                    "  window[%u,%u) pos=%u h=%u d=%u win=%.6f full=%.6f enc=%lld\n",
                                    win.t0, win.n, pos, h, d,
                                    (double) ggml_fp16_to_fp32(win_f16[win_off]),
                                    (double) ggml_fp16_to_fp32(s.mat_f16[full_off]),
                                    (long long) idx[c]);
                        }
                        ++mism;
                    }
                }
            }
        }
        if (mism > 0) {
            std::fprintf(stderr, "  window [%u,%u) mismatches=%d (source D%u heads=%u tokens=%u bits=%d %s)\n",
                    win.t0, win.n, mism, s.head_dim, s.n_heads, s.n_tokens, s.bits, s.value ? "V" : "K");
            ++g_failures;
        }
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
}

// ---------------------------------------------------------------------------
// Main conversion test for one source x (qx, dest_wht, block limit).
// ---------------------------------------------------------------------------

void test_conversion_case(const kvarn_source & s, ggml_type qx,
        uint32_t dst_wht, uint32_t block_limit) {
    config cfg;
    cfg.dst_type = qx;
    cfg.head_dim = s.head_dim;
    cfg.n_heads = s.n_heads;
    cfg.value = s.value;
    cfg.src.record_dim = s.record_dim;
    cfg.src.rotated = true;
    cfg.dst.wht_width = dst_wht;
    cfg.max_tokens_per_block = block_limit;

    const size_t row_floats = size_t(s.head_dim) * s.n_heads;
    const size_t payload_size = size_t(s.n_tokens) * row_floats / 32 * ggml_type_size(qx);

    reference_payload ref;
    build_reference(s, qx, dst_wht, ref);

    std::vector<uint8_t> payload(payload_size, 0xA5);
    window_source w;
    w.rotated = &s.rotated;
    w.head_dim = s.head_dim;
    w.n_heads = s.n_heads;
    const auto res = convert(cfg, s.n_tokens, serve_window, &w,
            payload.data(), payload.size());
    CHECK(res.st == status::ok, "convert failed");
    CHECK_EQ(res.payload_bytes, payload_size, "payload size");
    CHECK_EQ(res.tokens_completed, s.n_tokens, "tokens completed");
    CHECK(std::memcmp(payload.data(), ref.data.data(), payload_size) == 0,
            "converter payload differs from reference payload");

    std::vector<float> deq;
    dequant(qx, payload.data(), s.n_tokens, row_floats, deq);
    // Undo the destination rotation (self-inverse) so the error is measured
    // against floats reconstructed from the SAME KVarN source, regardless of
    // the declared destination domain.
    if (dst_wht > 0) {
        for (size_t t = 0; t < s.n_tokens; ++t) {
            for (uint32_t l = 0; l < s.n_heads; ++l) {
                float * head = deq.data() + t * row_floats + size_t(l) * s.head_dim;
                for (uint32_t b = 0; b < s.head_dim / dst_wht; ++b) {
                    ref_wht_head(head + size_t(b) * dst_wht,
                            (int) dst_wht, dst_wht == 64 ? 64 : 128);
                }
            }
        }
    }
    std::vector<float> orig_rows(size_t(s.n_tokens) * row_floats);
    for (uint32_t t = 0; t < s.n_tokens; ++t) {
        for (uint32_t l = 0; l < s.n_heads; ++l) {
            std::memcpy(orig_rows.data() + size_t(t) * row_floats + size_t(l) * s.head_dim,
                    s.orig.data() + size_t(l) * s.head_dim + size_t(t) * row_floats,
                    s.head_dim * sizeof(float));
        }
    }
    std::vector<float> input_rows(size_t(s.n_tokens) * row_floats);
    for (uint32_t t = 0; t < s.n_tokens; ++t) {
        for (uint32_t l = 0; l < s.n_heads; ++l) {
            std::memcpy(input_rows.data() + size_t(t) * row_floats + size_t(l) * s.head_dim,
                    s.input_logical.data() + size_t(l) * s.head_dim + size_t(t) * row_floats,
                    s.head_dim * sizeof(float));
        }
    }
    const double kvarn_floor = rmse(orig_rows, input_rows);
    const double qx_added = rmse(deq, orig_rows);
    const double overall = rmse(deq, input_rows);
    std::printf("  convert %-5s D%-3u heads=%u tokens=%u kvarn_bits=%d %s dest_wht=%-3u block=%-4u "
            "payload_match=yes rmse(vs same-source)=%.6g floor=%.6g overall=%.6g\n",
            ggml_type_name(qx), s.head_dim, s.n_heads, s.n_tokens, s.bits,
            s.value ? "V" : "K", dst_wht, block_limit, qx_added, kvarn_floor, overall);
}

// ---------------------------------------------------------------------------

void test_error_paths(const kvarn_source & s) {
    config cfg;
    cfg.dst_type = GGML_TYPE_Q4_0;
    cfg.head_dim = s.head_dim;
    cfg.n_heads = s.n_heads;
    cfg.value = s.value;
    cfg.src.record_dim = s.record_dim;
    cfg.src.rotated = true;
    cfg.dst.wht_width = 0;

    const size_t row_floats = size_t(s.head_dim) * s.n_heads;
    const size_t payload_size = size_t(s.n_tokens) * row_floats / 32 * ggml_type_size(GGML_TYPE_Q4_0);
    std::vector<uint8_t> payload(payload_size, 0);

    window_source w;
    w.rotated = &s.rotated;
    w.head_dim = s.head_dim;
    w.n_heads = s.n_heads;

    config bad = cfg;

    // invalid dst type: only the Q4_0/Q5_0/Q6_0/Q8_0 whitelist is accepted,
    // BEFORE any ggml_type_traits lookup (negative/count/out-of-range enum
    // values and other 32-wide quant types must be rejected). Values are
    // passed as plain ints; config stores the type as int32_t so hostile
    // inputs never load an invalid ggml_type enum object.
    const int32_t invalid_types[] = {
        -1,
        GGML_TYPE_COUNT,
        GGML_TYPE_COUNT + 1,
        (int32_t) 0xFFFFFFFFu,
        (int32_t) 0x7FFFFFFF,
        GGML_TYPE_F16,
        GGML_TYPE_F32,
        GGML_TYPE_BF16,
        GGML_TYPE_Q4_1,
        GGML_TYPE_Q5_1,
        GGML_TYPE_Q8_1,
        GGML_TYPE_Q2_K,
    };
    for (int32_t t : invalid_types) {
        bad = cfg;
        bad.dst_type = t;
        CHECK(convert(bad, s.n_tokens, serve_window, &w, payload.data(), payload.size()).st ==
                status::invalid_type, "non-whitelisted dst type must be rejected");
    }

    // invalid geometry
    bad = cfg;
    bad.head_dim = 100;
    CHECK(convert(bad, s.n_tokens, serve_window, &w, payload.data(), payload.size()).st ==
            status::invalid_geometry, "invalid head_dim must be rejected");
    bad = cfg;
    bad.n_heads = 0;
    CHECK(convert(bad, s.n_tokens, serve_window, &w, payload.data(), payload.size()).st ==
            status::invalid_geometry, "zero heads must be rejected");
    bad = cfg;
    bad.src.record_dim = 64;
    bad.head_dim = 256;
    CHECK(convert(bad, s.n_tokens, serve_window, &w, payload.data(), payload.size()).st ==
            status::invalid_geometry, "record_dim 64 with D256 must be rejected");
    bad = cfg;
    bad.src.record_dim = 33;
    CHECK(convert(bad, s.n_tokens, serve_window, &w, payload.data(), payload.size()).st ==
            status::invalid_geometry, "invalid record_dim must be rejected");

    // invalid domain
    bad = cfg;
    bad.dst.wht_width = 32;
    CHECK(convert(bad, s.n_tokens, serve_window, &w, payload.data(), payload.size()).st ==
            status::invalid_domain, "invalid dest wht width must be rejected");
    if (s.head_dim == 64) {
        bad = cfg;
        bad.dst.wht_width = 128;
        CHECK(convert(bad, s.n_tokens, serve_window, &w, payload.data(), payload.size()).st ==
                status::invalid_domain, "dest wht not dividing head_dim must be rejected");
    }

    // explicit scope rejections
    bad = cfg;
    bad.n_streams = 2;
    CHECK(convert(bad, s.n_tokens, serve_window, &w, payload.data(), payload.size()).st ==
            status::unsupported_multi_stream, "multi-stream must be rejected");
    bad = cfg;
    bad.swa = true;
    CHECK(convert(bad, s.n_tokens, serve_window, &w, payload.data(), payload.size()).st ==
            status::unsupported_swa, "SWA must be rejected");

    // args
    CHECK(convert(cfg, 0, serve_window, &w, payload.data(), payload.size()).st ==
            status::invalid_args, "zero tokens must be rejected");
    CHECK(convert(cfg, s.n_tokens, nullptr, &w, payload.data(), payload.size()).st ==
            status::invalid_args, "null callback must be rejected");

    // payload overflow
    CHECK(convert(cfg, s.n_tokens, serve_window, &w, payload.data(), payload.size() - 1).st ==
            status::payload_overflow, "payload overflow must be rejected");

    // callback failure on the second window: the staged prefix already
    // quantized into the buffer must be preserved and match the reference.
    reference_payload ref;
    build_reference(s, GGML_TYPE_Q4_0, 0, ref);
    const size_t row = payload.size() / s.n_tokens;
    std::memset(payload.data(), 0, payload.size());
    w.fail_after_calls = 1;
    w.calls = 0;
    const auto cb_res = convert(cfg, s.n_tokens, serve_window, &w, payload.data(), payload.size());
    CHECK(cb_res.st == status::callback_failure, "callback failure must surface");
    CHECK_EQ(cb_res.tokens_completed, 64u, "callback failure staged prefix");
    CHECK(std::memcmp(payload.data(), ref.data.data(), 64 * row) == 0,
            "staged prefix before a failed callback must match the reference");
    w.fail_after_calls = -1;
    w.calls = 0;

    // NaN in the source block: must abort before publishing that window
    w.inject_nan = true;
    w.nan_token = 100;
    w.nan_head = 0;
    w.nan_dim = 0;
    const auto nan_res = convert(cfg, s.n_tokens, serve_window, &w, payload.data(), payload.size());
    CHECK(nan_res.st == status::non_finite_input, "NaN must be rejected");
    CHECK_EQ(nan_res.tokens_completed, 64u, "NaN staged prefix stops at the window boundary");
    w.inject_nan = false;
    w.calls = 0;

    // block_cap clamped to n_tokens: 1 token with cap UINT32_MAX must not
    // attempt a giant scratch, and must convert that single token.
    config one = cfg;
    one.max_tokens_per_block = UINT32_MAX;
    std::vector<uint8_t> one_payload(row_bytes(one), 0);
    const auto one_res = convert(one, 1, serve_window, &w, one_payload.data(), one_payload.size());
    CHECK(one_res.st == status::ok, "single token with huge block cap must convert");
    CHECK_EQ(one_res.tokens_completed, 1u, "single token completed");
    CHECK_EQ(one_res.payload_bytes, one_payload.size(), "single token payload size");
    CHECK_EQ(w.calls, 1, "single token served by exactly one window");
    w.calls = 0;

    // Throwing callbacks must surface as callback_failure without publishing
    // the failing window. The staged prefix already written stays valid.
    struct throw_ctx {
        const kvarn_source * src;
        int calls;
        int throw_on_call;
    };
    struct throw_ctx tc = { &s, 0, 1 };
    auto serve_throw = [](void * u, const block_request & req) -> bool {
        throw_ctx * c = static_cast<throw_ctx *>(u);
        ++c->calls;
        if (c->calls == c->throw_on_call) {
            throw std::runtime_error("callback boom");
        }
        // serve the window normally
        const std::vector<float> & rot = c->src->rotated;
        for (uint32_t t = 0; t < req.n_tokens; ++t) {
            for (uint32_t h = 0; h < req.n_heads; ++h) {
                std::memcpy(req.out_f32 + size_t(h) * req.head_dim + size_t(t) * req.head_dim * req.n_heads,
                        rot.data() + size_t(h) * req.head_dim +
                                size_t(req.t0 + t) * req.head_dim * req.n_heads,
                        req.head_dim * sizeof(float));
            }
        }
        return true;
    };
    const auto throw1 = convert(cfg, s.n_tokens, serve_throw, &tc, payload.data(), payload.size());
    CHECK(throw1.st == status::callback_failure, "throwing callback must be caught");
    CHECK_EQ(throw1.tokens_completed, 0u, "no prefix before first-window throw");
    CHECK_EQ(throw1.payload_bytes, 0u, "no payload published on first-window throw");
    tc.calls = 0;
    tc.throw_on_call = 2;
    std::memset(payload.data(), 0, payload.size());
    const auto throw2 = convert(cfg, s.n_tokens, serve_throw, &tc, payload.data(), payload.size());
    CHECK(throw2.st == status::callback_failure, "throwing callback on the second window must be caught");
    CHECK_EQ(throw2.tokens_completed, 64u, "staged prefix stops at the failing window");
    CHECK(std::memcmp(payload.data(), ref.data.data(), 64 * row) == 0,
            "staged prefix before a throwing window must match the reference");
    tc.calls = 0;
    struct bad_alloc_ctx { const kvarn_source * src; int calls; };
    struct bad_alloc_ctx ba = { &s, 0 };
    auto serve_alloc = [](void * u, const block_request &) -> bool {
        ++static_cast<bad_alloc_ctx *>(u)->calls;
        throw std::bad_alloc();
    };
    const auto throw3 = convert(cfg, s.n_tokens, serve_alloc, &ba, payload.data(), payload.size());
    CHECK(throw3.st == status::callback_failure, "bad_alloc from the callback must be caught");
    CHECK_EQ(throw3.tokens_completed, 0u, "no prefix when the first window throws");
}

// ---------------------------------------------------------------------------
// Overflow hardening: rejected before any callback call or allocation.
// ---------------------------------------------------------------------------

void test_hardening() {
    config big;
    big.dst_type = GGML_TYPE_Q8_0;
    big.head_dim = 512;
    big.n_heads = UINT32_MAX;
    big.src.record_dim = 128;
    big.src.rotated = true;
    big.dst.wht_width = 0;
    // A huge configured block cap forces block_cap = n_tokens so the bounded
    // scratch overflow is detected by arithmetic before any allocation.
    big.max_tokens_per_block = UINT32_MAX;

    int calls = 0;
    auto counting = [](void * u, const block_request &) -> bool {
        ++*static_cast<int *>(u);
        return true;
    };
    std::vector<uint8_t> dummy(1, 0);

    // n_tokens * row overflows uint64: rejected before capacity comparison,
    // before any callback and before any allocation.
    const auto r1 = convert(big, UINT32_MAX, counting, &calls, dummy.data(), dummy.size());
    CHECK(r1.st == status::payload_overflow, "uint64 payload overflow must be rejected");
    CHECK_EQ(calls, 0, "no callback on payload overflow");
    CHECK_EQ(r1.tokens_completed, 0u, "no progress on payload overflow");

    // Payload fits size_t, but the bounded window (block_cap = min(n_tokens,
    // cap)) cannot be sized: rejected before allocation, callback untouched.
    // The fake pointer/capacity is never dereferenced because the workspace
    // check returns first.
    const auto r2 = convert(big, 7800000u, counting, &calls, (void *) 0x1, SIZE_MAX);
    CHECK(r2.st == status::payload_overflow, "workspace overflow must be rejected");
    CHECK_EQ(calls, 0, "no callback on workspace overflow");
    CHECK_EQ(r2.tokens_completed, 0u, "no progress on workspace overflow");

    // Reviewer regression: block_cap * row_floats wraps uint64 exactly.
    // D=512, n_heads=1<<31 => row_floats = 2^40; n_tokens = cap = 1<<24 =>
    // win_floats = 2^40 * 2^24 = 2^64 wraps to 0. The Q4_0 payload is
    // 9*2^60 bytes and fits uint64, so only the guarded workspace multiply
    // can reject this -- before the callback and before any allocation.
    config wrap;
    wrap.dst_type = GGML_TYPE_Q4_0;
    wrap.head_dim = 512;
    wrap.n_heads = uint32_t(1) << 31;
    wrap.src.record_dim = 128;
    wrap.src.rotated = true;
    wrap.dst.wht_width = 0;
    wrap.max_tokens_per_block = uint32_t(1) << 24;
    const uint32_t wrap_tokens = uint32_t(1) << 24;
    CHECK(payload_bytes(wrap, wrap_tokens) == (size_t) (9ull << 60),
            "reviewer regression payload must fit (9*2^60 bytes)");
    std::vector<uint8_t> one_byte(1, 0);
    int wrap_calls = 0;
    const auto rw = convert(wrap, wrap_tokens, counting, &wrap_calls,
            one_byte.data(), payload_bytes(wrap, wrap_tokens));
    CHECK(rw.st == status::payload_overflow, "wrapping workspace size must be rejected");
    CHECK_EQ(wrap_calls, 0, "no callback on wrapping workspace size");
    CHECK_EQ(rw.tokens_completed, 0u, "no progress on wrapping workspace size");
}

// ---------------------------------------------------------------------------
// Streaming convert_to_sink tests.
// ---------------------------------------------------------------------------

struct stream_collector {
    std::vector<uint8_t> bytes;
    uint32_t max_chunk = 0;
    size_t max_chunk_bytes = 0;
    uint32_t n_blocks = 0;
    uint32_t last_t0 = 0;
    uint32_t last_n = 0;
    bool ordered = true;
};

bool collect_sink(void * user, const sink_block & block) {
    stream_collector * c = static_cast<stream_collector *>(user);
    ++c->n_blocks;
    c->max_chunk = std::max(c->max_chunk, block.n_tokens);
    c->max_chunk_bytes = std::max(c->max_chunk_bytes,
            size_t(block.n_tokens) * block.row_bytes);
    if (c->n_blocks > 1 && c->last_t0 + c->last_n != block.t0) {
        c->ordered = false;
    }
    c->last_t0 = block.t0;
    c->last_n = block.n_tokens;
    c->bytes.insert(c->bytes.end(), block.data,
            block.data + size_t(block.n_tokens) * block.row_bytes);
    return true;
}

void test_streaming_equivalence(const kvarn_source & s, ggml_type qx,
        uint32_t dst_wht, uint32_t block_limit) {
    config cfg;
    cfg.dst_type = qx;
    cfg.head_dim = s.head_dim;
    cfg.n_heads = s.n_heads;
    cfg.value = s.value;
    cfg.src.record_dim = s.record_dim;
    cfg.src.rotated = true;
    cfg.dst.wht_width = dst_wht;
    cfg.max_tokens_per_block = block_limit;

    const size_t payload_size = payload_bytes(cfg, s.n_tokens);
    std::vector<uint8_t> full(payload_size, 0);
    window_source w;
    w.rotated = &s.rotated;
    w.head_dim = s.head_dim;
    w.n_heads = s.n_heads;
    const auto res = convert(cfg, s.n_tokens, serve_window, &w,
            full.data(), full.size());
    CHECK(res.st == status::ok, "stream equivalence convert failed");
    CHECK_EQ(res.payload_bytes, payload_size, "stream equivalence payload size");

    stream_collector c;
    const auto rs = convert_to_sink(cfg, s.n_tokens, serve_window, &w,
            collect_sink, &c);
    CHECK(rs.st == status::ok, "streaming convert failed");
    CHECK_EQ(rs.tokens_completed, s.n_tokens, "streaming tokens completed");
    CHECK_EQ(rs.payload_bytes, payload_size, "streaming payload size");
    CHECK(c.ordered, "sink blocks must arrive in global token order");
    CHECK(c.bytes.size() == full.size() &&
            std::memcmp(c.bytes.data(), full.data(), full.size()) == 0,
            "streamed output must be byte-identical to convert");
    const uint32_t cap = block_limit == 0 ? 64 : block_limit;
    CHECK(c.max_chunk <= std::min(cap, s.n_tokens),
            "sink chunk must respect the window limit");
    CHECK(c.last_t0 + c.last_n == s.n_tokens,
            "final short window must carry the true remainder");
    CHECK_EQ(c.n_blocks, (s.n_tokens + std::min(cap, s.n_tokens) - 1) /
            std::min(cap, s.n_tokens), "sink block count");
}

void test_streaming_errors(const kvarn_source & s) {
    config cfg;
    cfg.dst_type = GGML_TYPE_Q4_0;
    cfg.head_dim = s.head_dim;
    cfg.n_heads = s.n_heads;
    cfg.value = s.value;
    cfg.src.record_dim = s.record_dim;
    cfg.src.rotated = true;
    cfg.dst.wht_width = 0;

    const size_t payload_size = payload_bytes(cfg, s.n_tokens);
    reference_payload ref;
    build_reference(s, GGML_TYPE_Q4_0, 0, ref);
    const size_t row = payload_size / s.n_tokens;

    window_source w;
    w.rotated = &s.rotated;
    w.head_dim = s.head_dim;
    w.n_heads = s.n_heads;

    // NaN in the second window: no sink call for that window; confirmed
    // earlier windows only.
    w.inject_nan = true;
    w.nan_token = 100;
    w.nan_head = 0;
    w.nan_dim = 0;
    {
        stream_collector c;
        const auto r = convert_to_sink(cfg, s.n_tokens, serve_window, &w,
                collect_sink, &c);
        CHECK(r.st == status::non_finite_input, "NaN must abort streaming");
        CHECK_EQ(r.tokens_completed, 64u, "NaN streaming prefix");
        CHECK_EQ(c.n_blocks, 1u, "no sink call for the non-finite window");
        CHECK(c.bytes.size() == 64 * row &&
                std::memcmp(c.bytes.data(), ref.data.data(), 64 * row) == 0,
                "confirmed prefix must match the reference");
    }
    w.inject_nan = false;
    w.calls = 0;

    // read_block failure on the second window.
    w.fail_after_calls = 1;
    w.calls = 0;
    {
        stream_collector c;
        const auto r = convert_to_sink(cfg, s.n_tokens, serve_window, &w,
                collect_sink, &c);
        CHECK(r.st == status::callback_failure, "read failure must abort streaming");
        CHECK_EQ(r.tokens_completed, 64u, "read failure prefix");
        CHECK_EQ(c.n_blocks, 1u, "no sink call after the failed read");
        CHECK(c.bytes.size() == 64 * row &&
                std::memcmp(c.bytes.data(), ref.data.data(), 64 * row) == 0,
                "read-failure prefix must match the reference");
    }
    w.fail_after_calls = -1;
    w.calls = 0;

    // sink failure on the second block.
    {
        struct fail_ctx { int calls; };
        fail_ctx fc = { 0 };
        auto fail_sink = [](void * u, const sink_block &) -> bool {
            return ++static_cast<fail_ctx *>(u)->calls < 2;
        };
        const auto r = convert_to_sink(cfg, s.n_tokens, serve_window, &w,
                fail_sink, &fc);
        CHECK(r.st == status::sink_failure, "sink failure must surface");
        CHECK_EQ(r.tokens_completed, 64u, "sink failure prefix");
        CHECK_EQ(fc.calls, 2, "failing window reached the sink once");
    }
    // sink exception on the second block.
    {
        struct throw_ctx { int calls; };
        throw_ctx tc = { 0 };
        auto throw_sink = [](void * u, const sink_block &) -> bool {
            if (++static_cast<throw_ctx *>(u)->calls >= 2) {
                throw std::runtime_error("sink boom");
            }
            return true;
        };
        const auto r = convert_to_sink(cfg, s.n_tokens, serve_window, &w,
                throw_sink, &tc);
        CHECK(r.st == status::sink_failure, "throwing sink must be caught");
        CHECK_EQ(r.tokens_completed, 64u, "throwing sink prefix");
        CHECK_EQ(tc.calls, 2, "throwing window reached the sink once");
    }

    // Invalid config and overflow: no read_block and no sink call.
    {
        config bad = cfg;
        bad.dst_type = GGML_TYPE_F16;
        int reads = 0, sinks = 0;
        auto count_read = [](void * u, const block_request &) -> bool {
            ++*static_cast<int *>(u);
            return true;
        };
        auto count_sink = [](void * u, const sink_block &) -> bool {
            ++*static_cast<int *>(u);
            return true;
        };
        CHECK(convert_to_sink(bad, s.n_tokens, count_read, &reads,
                count_sink, &sinks).st == status::invalid_type,
                "invalid type must be rejected before callbacks");
        CHECK_EQ(reads, 0, "no read callback on invalid type");
        CHECK_EQ(sinks, 0, "no sink callback on invalid type");
        config overflow = cfg;
        overflow.head_dim = 512;
        overflow.n_heads = UINT32_MAX;
        overflow.src.record_dim = 128;
        overflow.max_tokens_per_block = UINT32_MAX;
        CHECK(convert_to_sink(overflow, UINT32_MAX, count_read, &reads,
                count_sink, &sinks).st == status::payload_overflow,
                "unrepresentable totals must be rejected before callbacks");
        CHECK_EQ(reads, 0, "no read callback on overflow");
        CHECK_EQ(sinks, 0, "no sink callback on overflow");
    }
}

// 64-bit FNV-1a over the streamed bytes (bounded virtual prefix, no full
// output allocation).
uint64_t fnv1a_update(uint64_t h, const uint8_t * data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        h ^= data[i];
        h *= 1099511628211ULL;
    }
    return h;
}

void test_streaming_bounded() {
    // 200k-token virtual prefix with a small geometry (D64 V, 1 head, Q4_0):
    // the sink only ever receives one bounded block; the full output
    // (~7.2 MiB) is hashed, never allocated by the API.
    kvarn_source s;
    build_source(s, 200000, 64, 1, 4, true);
    config cfg;
    cfg.dst_type = GGML_TYPE_Q4_0;
    cfg.head_dim = 64;
    cfg.n_heads = 1;
    cfg.value = true;
    cfg.src.record_dim = 64;
    cfg.src.rotated = true;
    cfg.dst.wht_width = 0;
    cfg.max_tokens_per_block = 64;
    const size_t row_bytes = (size_t(cfg.head_dim) * cfg.n_heads) / 32 *
        ggml_type_size(GGML_TYPE_Q4_0);
    CHECK(row_bytes == 36, "bounded test row bytes");

    window_source w;
    w.rotated = &s.rotated;
    w.head_dim = cfg.head_dim;
    w.n_heads = cfg.n_heads;
    struct hash_ctx {
        uint64_t hash = 1469598103934665603ULL;
        uint32_t max_chunk = 0;
        size_t max_chunk_bytes = 0;
        uint32_t n_blocks = 0;
        uint64_t total = 0;
        bool ordered = true;
        uint32_t last_t0 = 0;
        uint32_t last_n = 0;
    };
    hash_ctx h;
    auto hash_sink = [](void * u, const sink_block & b) -> bool {
        hash_ctx * h = static_cast<hash_ctx *>(u);
        ++h->n_blocks;
        h->max_chunk = std::max(h->max_chunk, b.n_tokens);
        h->max_chunk_bytes = std::max(h->max_chunk_bytes,
                size_t(b.n_tokens) * b.row_bytes);
        if (h->n_blocks > 1 && h->last_t0 + h->last_n != b.t0) {
            h->ordered = false;
        }
        h->last_t0 = b.t0;
        h->last_n = b.n_tokens;
        h->total += size_t(b.n_tokens) * b.row_bytes;
        h->hash = fnv1a_update(h->hash, b.data, size_t(b.n_tokens) * b.row_bytes);
        return true;
    };
    const auto r = convert_to_sink(cfg, 200000, serve_window, &w, hash_sink, &h);
    CHECK(r.st == status::ok, "bounded streaming failed");
    CHECK_EQ(r.tokens_completed, 200000u, "bounded tokens completed");
    CHECK(r.payload_bytes == size_t(200000) * row_bytes, "bounded payload size");
    CHECK(h.total == r.payload_bytes, "bounded sink total");
    CHECK_EQ(h.n_blocks, 3125u, "bounded block count (200000/64)");
    CHECK(h.max_chunk <= 64, "bounded sink chunk");
    CHECK(h.max_chunk_bytes <= 64 * row_bytes, "bounded sink chunk bytes");
    CHECK(h.ordered, "bounded blocks must be ordered");
    CHECK(h.last_t0 + h.last_n == 200000, "bounded final window");
    std::printf("  bounded 200k D64/V/Q4: blocks=%u max_chunk=%u max_bytes=%zu hash=%016llx\n",
            h.n_blocks, h.max_chunk, h.max_chunk_bytes, (unsigned long long) h.hash);

    // Streamed prefix (385 tokens) must be byte-identical to convert.
    config small = cfg;
    stream_collector c;
    const auto rp = convert_to_sink(small, 385, serve_window, &w, collect_sink, &c);
    CHECK(rp.st == status::ok, "bounded prefix streaming failed");
    std::vector<uint8_t> full(size_t(385) * row_bytes, 0);
    const auto rc = convert(small, 385, serve_window, &w, full.data(), full.size());
    CHECK(rc.st == status::ok, "bounded prefix convert failed");
    CHECK(c.bytes == full, "bounded prefix must match convert byte-for-byte");
    free_source(s);
}

void test_streaming_chain(const kvarn_source & s) {
    // Real KVarN source -> reviewed materializer -> convert_to_sink, compared
    // against the reference quantization of the same effective source.
    if (s.n_tokens != 385 || s.bits != 4) {
        return;
    }
    const uint32_t live_group = (s.n_tokens - 1) / 128;
    const uint32_t live_pos = (s.n_tokens - 1) % 128;
    const bool live_in_stage = !(live_pos == 127);
    std::vector<int64_t> map(s.n_tokens);
    for (uint32_t pos = 0; pos < s.n_tokens; ++pos) {
        const uint32_t g = pos / 128;
        if (g == 0) {
            map[pos] = encode_stage_cell(pos, 0);
        } else if (live_in_stage && g == live_group) {
            map[pos] = encode_stage_cell(pos, 1 + ((g - 1) % 2));
        } else {
            map[pos] = pos;
        }
    }

    llama_kv_qx::materializer_config mcfg;
    mcfg.head_dim = s.head_dim;
    mcfg.n_heads = s.n_heads;
    mcfg.value = s.value;
    mcfg.record_dim = s.record_dim;
    mcfg.head_slices = s.head_slices;
    mcfg.bits = s.bits;
    mcfg.stage_groups = s.stage_groups;
    mcfg.max_window = 256;
    llama_kv_qx::materializer_source msrc;
    msrc.records = s.records;
    msrc.stage = s.stage;
    msrc.backend = s.backend;
    llama_kv_qx::materializer m(mcfg, msrc);
    CHECK(m.valid(), "chain materializer must validate");
    CHECK(m.set_prefix(map.data(), s.n_tokens) == llama_kv_qx::mat_status::ok,
            "chain set_prefix failed");

    config cfg;
    cfg.dst_type = GGML_TYPE_Q4_0;
    cfg.head_dim = s.head_dim;
    cfg.n_heads = s.n_heads;
    cfg.value = s.value;
    cfg.src.record_dim = s.record_dim;
    cfg.src.rotated = true;
    cfg.dst.wht_width = 0;
    cfg.max_tokens_per_block = 64;

    reference_payload ref;
    build_reference(s, GGML_TYPE_Q4_0, 0, ref);
    stream_collector c;
    const auto r = convert_to_sink(cfg, s.n_tokens,
            llama_kv_qx::materializer_read_block, &m, collect_sink, &c);
    CHECK(r.st == status::ok, "chain streaming failed");
    CHECK(c.bytes.size() == ref.data.size() &&
            std::memcmp(c.bytes.data(), ref.data.data(), ref.data.size()) == 0,
            "chain stream must match the reference payload");
    std::printf("  chain-stream: %s D%u tokens=%u payload_match=yes\n",
            s.value ? "V" : "K", s.head_dim, s.n_tokens);
}

} // namespace

int main() {
    std::printf("test-kv-qx-convert: KVarN -> Qx bounded CPU conversion\n");
    std::printf("KVarN source built with ggml_kvarn_store + ggml_kvarn_materialize (CPU backend)\n");

    test_hardening();
    test_streaming_bounded();

    const ggml_type qx_types[] = {
        GGML_TYPE_Q4_0, GGML_TYPE_Q5_0, GGML_TYPE_Q6_0, GGML_TYPE_Q8_0,
    };
    const uint32_t tokens[] = { 127, 128, 129, 385 };

    // Geometry matrix: K D256 (2 slices), K D128 (1 slice), V D64, K D512 (4 slices).
    struct geom { uint32_t head_dim; uint32_t n_heads; bool value; };
    const geom geoms[] = {
        { 256, 4, false },
        { 128, 3, false },
        {  64, 4, true  },
        { 512, 2, false },
    };

    double floor_bits4 = -1.0, floor_bits8 = -1.0;

    for (const geom & g : geoms) {
        for (uint32_t n_tokens : tokens) {
            for (int bits : { 4, 8 }) {
                kvarn_source s;
                build_source(s, n_tokens, g.head_dim, g.n_heads, bits, g.value);
                std::printf("source D%u heads=%u tokens=%u bits=%d %s\n",
                        g.head_dim, g.n_heads, n_tokens, bits, g.value ? "V" : "K");
                test_windowed_materialization(s);

                const uint32_t dest_whts[] = { 0u, g.head_dim == 64 ? 64u : (g.head_dim == 128 ? 0u : 256u) };
                for (uint32_t dst_wht : dest_whts) {
                    if (dst_wht == 0) {
                        dst_wht = 0;
                    }
                    for (ggml_type qx : qx_types) {
                        test_conversion_case(s, qx, dst_wht, 64);
                        test_streaming_equivalence(s, qx, dst_wht, 64);
                    }
                }
                // bounded chunking must be invariant
                if (bits == 4) {
                    for (uint32_t blk : { 16u, 385u, 0u }) {
                        test_conversion_case(s, GGML_TYPE_Q4_0, 0, blk);
                        test_streaming_equivalence(s, GGML_TYPE_Q4_0, 0, blk);
                    }
                }
                // error ordering and KVarN floor reporting (Q8 cannot recover
                // precision already lost in the KVarN records)
                {
                    std::vector<float> deq;
                    const size_t row_floats = size_t(g.head_dim) * g.n_heads;
                    std::vector<uint8_t> payload(size_t(n_tokens) * row_floats / 32 * 34);
                    window_source w;
                    w.rotated = &s.rotated;
                    w.head_dim = g.head_dim;
                    w.n_heads = g.n_heads;
                    config cfg;
                    cfg.dst_type = GGML_TYPE_Q4_0;
                    cfg.head_dim = g.head_dim;
                    cfg.n_heads = g.n_heads;
                    cfg.value = g.value;
                    cfg.src.record_dim = s.record_dim;
                    const auto r4 = convert(cfg, n_tokens, serve_window, &w, payload.data(), payload.size());
                    CHECK(r4.st == status::ok, "Q4 conversion failed");
                    dequant(GGML_TYPE_Q4_0, payload.data(), n_tokens, row_floats, deq);
                    std::vector<float> orig_rows(size_t(n_tokens) * row_floats);
                    for (uint32_t t = 0; t < n_tokens; ++t) {
                        for (uint32_t l = 0; l < g.n_heads; ++l) {
                            std::memcpy(orig_rows.data() + size_t(t) * row_floats + size_t(l) * g.head_dim,
                                    s.orig.data() + size_t(l) * g.head_dim + size_t(t) * row_floats,
                                    g.head_dim * sizeof(float));
                        }
                    }
                    std::vector<float> input_rows(size_t(n_tokens) * row_floats);
                    for (uint32_t t = 0; t < n_tokens; ++t) {
                        for (uint32_t l = 0; l < g.n_heads; ++l) {
                            std::memcpy(input_rows.data() + size_t(t) * row_floats + size_t(l) * g.head_dim,
                                    s.input_logical.data() + size_t(l) * g.head_dim + size_t(t) * row_floats,
                                    g.head_dim * sizeof(float));
                        }
                    }
                    const double floor = rmse(orig_rows, input_rows);
                    std::vector<float> deq8;
                    cfg.dst_type = GGML_TYPE_Q8_0;
                    const auto r8 = convert(cfg, n_tokens, serve_window, &w, payload.data(), payload.size());
                    CHECK(r8.st == status::ok, "Q8 conversion failed");
                    dequant(GGML_TYPE_Q8_0, payload.data(), n_tokens, row_floats, deq8);
                    if (bits == 4) {
                        CHECK(rmse(deq8, orig_rows) < rmse(deq, orig_rows),
                                "Q8 must add less conversion error than Q4 on the same source");
                        floor_bits4 = floor;
                        std::printf("  err D%u tokens=%u: KVarN4 floor=%.6g Q4-added=%.6g Q8-added=%.6g\n",
                                g.head_dim, n_tokens, floor, rmse(deq, orig_rows),
                                rmse(deq8, orig_rows));
                    } else {
                        floor_bits8 = floor;
                        std::printf("  err D%u tokens=%u: KVarN8 floor=%.6g (same-geometry comparison across bits)\n",
                                g.head_dim, n_tokens, floor);
                        // Only meaningful when records were actually quantized
                        // (groups beyond the sink/tail); for <=129 tokens the
                        // whole source stays in F16 stage and both bits share
                        // the same floor.
                        if (n_tokens == 385 && floor_bits4 >= 0.0) {
                            CHECK(floor_bits8 < floor_bits4,
                                    "KVarN8 floor must be lower than KVarN4 floor for the same geometry");
                        }
                    }
                }
                test_error_paths(s);
                test_streaming_errors(s);
                test_streaming_chain(s);
                free_source(s);
            }
        }
    }

    std::printf("== summary: %s\n", g_failures == 0 ? "ALL PASS" : "FAILURES PRESENT");
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}