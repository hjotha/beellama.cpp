// Standalone CPU test for llama_kv_qx::materializer (bounded backend-aware
// read_block adapter feeding llama_kv_qx::convert from REAL KVarN records +
// stage tensors).
//
// The KVarN source is built with the canonical ggml CPU operators
// (ggml_kvarn_store + ggml_kvarn_materialize). Every window served by the
// adapter is compared bit-exactly against an independent complete
// materialization of the SAME effective source (full read_indirect
// materialize with the same index map). The adapter's window graph is then
// chained into llama_kv_qx::convert and the resulting Qx payload is compared
// against a reference quantization of the same effective source.
//
// Build (CPU only by default, immutable baseline libraries):
//   g++ -O2 -std=c++17 -I ggml/include -I ggml/src -I src
//       tests/test-kv-kvarn-materializer.cpp src/llama-kv-kvarn-materializer.cpp
//       src/llama-kv-qx-convert.cpp -L <baseline-bin> -lggml-cpu -lggml-base
//       -lgomp -lpthread -o <out>/test-kv-kvarn-materializer
//   LD_LIBRARY_PATH=<baseline-bin> <out>/test-kv-kvarn-materializer
//
// Optional (principal runs it, NOT executed here): select a GPU backend with
//   <out>/test-kv-kvarn-materializer --backend CUDA0 [--backend-path <dir>]
// The source (records/stage/current) is then allocated ON that backend and
// every graph runs on it; expected outputs are read back with backend APIs.
// --backend-path is machine-independent: it defaults to the BEELAMA_BASELINE_DIR
// environment variable, then to <cwd>/bin; if neither resolves to a directory
// containing libggml.so the run fails with instructions (no hardcoded paths).
// The default path is CPU and never touches GPU or the runtime loader.

#include "llama-kv-kvarn-materializer.h"
#include "llama-kv-qx-convert.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h" // test-only mock backend (unsupported_op)
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#if defined(__linux__)
#include <dlfcn.h>
#endif

namespace {

using llama_kv_qx::mat_status;
using llama_kv_qx::materializer;
using llama_kv_qx::materializer_config;
using llama_kv_qx::materializer_source;
using llama_kv_qx::materializer_read_block;
using llama_kv_qx::block_request;
using llama_kv_qx::config;
using llama_kv_qx::convert;
using llama_kv_qx::status;

int g_failures = 0;
ggml_backend_t g_backend = nullptr; // set by main (CPU default, optional GPU)

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); \
        ++g_failures; \
    } \
} while (0)

#define CHECK_EQ(a, b, msg) do { \
    if (!((a) == (b))) { \
        std::fprintf(stderr, "FAIL %s:%d: %s (got %lld, want %lld)\n", __FILE__, __LINE__, msg, \
                (long long) (a), (long long) (b)); \
        ++g_failures; \
    } \
} while (0)

// ---------------------------------------------------------------------------
// Canonical KVarN WHT (independent copy of the production equations) and
// index encodings (canonical formulas from src/llama-kvarn.h).
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

int64_t encode_store_cell(uint32_t cell, uint32_t slot) {
    return int64_t((uint64_t(slot) + 1u) << 32u | uint64_t(cell));
}

int64_t encode_stage_cell(uint32_t cell, uint32_t slot) {
    return -encode_store_cell(cell, slot) - 2;
}

// ---------------------------------------------------------------------------
// Real KVarN source (canonical store + full materializations).
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

    uint32_t n_tokens = 0;
    uint32_t head_dim = 0;
    uint32_t n_heads = 0;       // logical
    uint32_t record_dim = 128;
    uint32_t head_slices = 1;
    int bits = 4;
    bool value = false;
    int stage_groups = 3;
    int tail_groups = 2;

    std::vector<float> input;      // [record_dim, phys_heads, n_tokens] F32 pre-KVarN
    std::vector<float> input_logical; // [head_dim, n_heads, n_tokens] pre-KVarN
    std::vector<float> rotated_full;  // independent full read_indirect reference
    std::vector<float> rotated_nonindirect; // canonical eager non-indirect full materialize
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

    const size_t rows = value ? 128 : s.record_dim;
    const size_t cols = value ? s.record_dim : 128;
    const size_t record_bytes = (rows * cols * size_t(bits) + 7) / 8 +
        (2 * rows + cols) * sizeof(ggml_fp16_t);

    ggml_init_params params = { 32 * 1024 * 1024, nullptr, true };
    s.ctx = ggml_init(params);
    CHECK(s.ctx != nullptr, "ggml_init failed");

    s.current = ggml_new_tensor_3d(s.ctx, GGML_TYPE_F32, s.record_dim, phys_heads, n_tokens);
    s.indices = ggml_new_tensor_1d(s.ctx, GGML_TYPE_I64, n_tokens);
    s.stage = ggml_new_tensor_3d(s.ctx, GGML_TYPE_F16, s.record_dim, phys_heads, stage_cap);
    s.records = ggml_new_tensor_3d(s.ctx, GGML_TYPE_I8, record_bytes, phys_heads, n_groups);

    s.stored = ggml_kvarn_store(s.ctx, s.current, s.indices, s.stage, s.records,
            bits, 16, value, s.stage_groups);
    s.stored->op_params[5] = (int32_t) s.head_slices;
    s.stored->op_params[9] = 1; // eager records, as the production cache

    // Canonical non-indirect full materialization (reference baseline).
    ggml_tensor * full = ggml_kvarn_materialize(s.ctx, s.records, s.stored, s.indices,
            n_tokens, 0, 1, bits, value, s.stage_groups);
    full->op_params[4] = 1;
    full->op_params[5] = (int32_t) s.head_slices;
    full->op_params[9] = 1;

    ggml_cgraph * graph = ggml_new_graph(s.ctx);
    ggml_build_forward_expand(graph, full);
    s.backend = g_backend;
    CHECK(s.backend != nullptr, "test backend init failed");
    s.buffer = ggml_backend_alloc_ctx_tensors(s.ctx, s.backend);
    CHECK(s.buffer != nullptr, "tensor allocation failed");

    const size_t n_elem = size_t(s.record_dim) * phys_heads * n_tokens;
    s.input.resize(n_elem);
    for (uint32_t t = 0; t < n_tokens; ++t) {
        for (uint32_t h = 0; h < phys_heads; ++h) {
            for (uint32_t d = 0; d < s.record_dim; ++d) {
                const int idx = int((h * 31 + d * 13 + t * 17) % 31 - 15);
                s.input[size_t(d) + size_t(h) * s.record_dim + size_t(t) * s.record_dim * phys_heads] =
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
    ggml_backend_tensor_set(s.current, s.input.data(), 0, ggml_nbytes(s.current));
    ggml_backend_tensor_set(s.indices, idx.data(), 0, ggml_nbytes(s.indices));
    ggml_backend_tensor_set(s.stage, zeros.data(), 0, ggml_nbytes(s.stage));
    ggml_backend_tensor_set(s.records, zeros.data(), 0, ggml_nbytes(s.records));

    CHECK(ggml_backend_graph_compute(s.backend, graph) == GGML_STATUS_SUCCESS,
            "store+materialize graph compute failed");

    s.rotated_nonindirect.assign(size_t(s.head_dim) * s.n_heads * n_tokens, 0.0f);
    std::vector<ggml_fp16_t> full_f16(ggml_nelements(full));
    ggml_backend_tensor_get(full, full_f16.data(), 0, ggml_nbytes(full));
    for (uint32_t t = 0; t < n_tokens; ++t) {
        for (uint32_t l = 0; l < s.n_heads; ++l) {
            for (uint32_t d = 0; d < s.head_dim; ++d) {
                const uint32_t slice = s.head_slices == 1 ? 0 : d / s.record_dim;
                const uint32_t lane = s.head_slices == 1 ? d : d % s.record_dim;
                const uint32_t ph = l * s.head_slices + slice;
                s.rotated_nonindirect[size_t(d) + size_t(l) * s.head_dim +
                        size_t(t) * s.head_dim * s.n_heads] =
                    ggml_fp16_to_fp32(full_f16[size_t(lane) + size_t(ph) * s.record_dim +
                            size_t(t) * s.record_dim * phys_heads]);
            }
        }
    }

    s.input_logical.assign(size_t(s.head_dim) * s.n_heads * n_tokens, 0.0f);
    for (uint32_t t = 0; t < n_tokens; ++t) {
        for (uint32_t l = 0; l < s.n_heads; ++l) {
            for (uint32_t d = 0; d < s.head_dim; ++d) {
                const uint32_t slice = s.head_slices == 1 ? 0 : d / s.record_dim;
                const uint32_t lane = s.head_slices == 1 ? d : d % s.record_dim;
                const uint32_t ph = l * s.head_slices + slice;
                s.input_logical[size_t(d) + size_t(l) * s.head_dim +
                        size_t(t) * s.head_dim * s.n_heads] =
                    s.input[size_t(lane) + size_t(ph) * s.record_dim +
                            size_t(t) * s.record_dim * phys_heads];
            }
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
    s.backend = nullptr; // the test backend is shared (g_backend); freed by main
}

// Owner-side effective index map: group 0 sink in stage slot 0; completed
// groups in records (plain positions); live incomplete group in the stage
// slot the eager store uses. Mirrors the compact read plan semantics.
std::vector<int64_t> make_effective_map(const kvarn_source & s) {
    const uint32_t live_group = (s.n_tokens - 1) / 128;
    const uint32_t live_pos = (s.n_tokens - 1) % 128;
    const bool live_in_stage = !(live_pos == 127);
    std::vector<int64_t> map(s.n_tokens);
    for (uint32_t pos = 0; pos < s.n_tokens; ++pos) {
        const uint32_t g = pos / 128;
        if (g == 0) {
            map[pos] = encode_stage_cell(pos, 0);
        } else if (live_in_stage && g == live_group) {
            map[pos] = encode_stage_cell(pos, 1 + ((g - 1) % uint32_t(s.tail_groups)));
        } else {
            map[pos] = pos;
        }
    }
    return map;
}

// Independent complete materialization of the SAME effective source: full
// read_indirect materialize with the full index map (map.size() entries),
// cast to F32 and reshaped to [head_dim, n_heads, n_kv].
bool full_reference(const kvarn_source & s, const std::vector<int64_t> & map,
        std::vector<float> & out) {
    const int64_t n_kv = int64_t(map.size());
    ggml_context * ctx = ggml_init({ 32 * 1024 * 1024, nullptr, true });
    if (ctx == nullptr) {
        return false;
    }
    ggml_tensor * full_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_kv);
    ggml_tensor * full = ggml_kvarn_materialize(ctx, s.records, s.stage, full_idx,
            n_kv, 0, 1, s.bits, s.value, s.stage_groups);
    full->op_params[4] = 1;
    full->op_params[5] = (int32_t) s.head_slices;
    full->op_params[10] = 1;
    ggml_tensor * cast = ggml_cast(ctx, full, GGML_TYPE_F32);
    ggml_tensor * reshaped = ggml_reshape_4d(ctx, cast, s.head_dim, s.n_heads, n_kv, 1);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, reshaped);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, s.backend);
    if (buf == nullptr) {
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(full_idx, map.data(), 0, map.size() * sizeof(map[0]));
    if (ggml_backend_graph_compute(s.backend, graph) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }
    out.resize(ggml_nelements(reshaped));
    ggml_backend_tensor_get(reshaped, out.data(), 0, ggml_nbytes(reshaped));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

materializer_config make_mat_cfg(const kvarn_source & s, uint32_t max_window = 256) {
    materializer_config cfg;
    cfg.head_dim = s.head_dim;
    cfg.n_heads = s.n_heads;
    cfg.value = s.value;
    cfg.record_dim = s.record_dim;
    cfg.head_slices = s.head_slices;
    cfg.bits = s.bits;
    cfg.stage_groups = s.stage_groups;
    cfg.max_window = max_window;
    return cfg;
}

materializer_source make_mat_src(const kvarn_source & s) {
    materializer_source src;
    src.records = s.records;
    src.stage = s.stage;
    src.backend = s.backend;
    return src;
}

// ---------------------------------------------------------------------------
// Window serving: every served window must equal the reference slice
// bit-exactly.
// ---------------------------------------------------------------------------

void check_window_equals_ref(const kvarn_source & s, const std::vector<float> & ref,
        const std::vector<float> & win, uint32_t t0, uint32_t n, const char * what) {
    const uint32_t row = s.head_dim * s.n_heads;
    for (uint32_t t = 0; t < n; ++t) {
        for (uint32_t i = 0; i < row; ++i) {
            const float want = ref[size_t(t0 + t) * row + i];
            const float got = win[size_t(t) * row + i];
            if (got != want) {
                std::fprintf(stderr, "  window %s t0=%u n=%u token=%u idx=%u got=%.9g want=%.9g\n",
                        what, t0, n, t, i, (double) got, (double) want);
                ++g_failures;
                return;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Test drivers
// ---------------------------------------------------------------------------

void test_window_serving(const kvarn_source & s) {
    const std::vector<int64_t> map = make_effective_map(s);
    std::vector<float> ref;
    CHECK(full_reference(s, map, ref), "full read_indirect reference failed");
    // The canonical non-indirect eager materialize must agree with the
    // owner map for the standard ownership layout.
    {
        size_t mism = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            if (ref[i] != s.rotated_nonindirect[i] && ++mism <= 3) {
                std::fprintf(stderr, "  map-vs-eager mismatch at %zu: %g vs %g\n",
                        i, (double) ref[i], (double) s.rotated_nonindirect[i]);
            }
        }
        CHECK(mism == 0, "effective map must match the canonical eager materialization");
    }

    materializer m(make_mat_cfg(s), make_mat_src(s));
    CHECK(m.valid(), "materializer must validate");
    CHECK(m.set_prefix(map.data(), s.n_tokens) == mat_status::ok, "set_prefix failed");

    // Arbitrary windows: sizes and offsets covering group boundaries, the
    // partial final window, holes and repeated calls (graph reuse).
    for (uint32_t t0 = 0; t0 < s.n_tokens;) {
        const uint32_t n = std::min<uint32_t>(64, s.n_tokens - t0);
        std::vector<float> win(size_t(s.head_dim) * s.n_heads * n, 0.0f);
        block_request req = { t0, n, s.head_dim, s.n_heads, s.value, win.data() };
        CHECK(m.serve(nullptr, req), "serve failed");
        check_window_equals_ref(s, ref, win, t0, n, "run");
        t0 += n;
    }
    // Arbitrary single windows (independent of the run above).
    const uint32_t offs[] = { 0, 1, 63, 64, 127, 128, 255, 256, 384 };
    for (uint32_t off : offs) {
        if (off >= s.n_tokens) {
            continue;
        }
        const uint32_t n = std::min<uint32_t>(std::min<uint32_t>(129, 64), s.n_tokens - off);
        if (n == 0) {
            continue;
        }
        std::vector<float> win(size_t(s.head_dim) * s.n_heads * n, 0.0f);
        block_request req = { off, n, s.head_dim, s.n_heads, s.value, win.data() };
        CHECK(m.serve(nullptr, req), "serve failed (offset window)");
        check_window_equals_ref(s, ref, win, off, n, "offset");
    }
    // Partial final window (1 token when 385 % 64 == 1).
    {
        const uint32_t t0 = s.n_tokens - 1;
        std::vector<float> win(size_t(s.head_dim) * s.n_heads, 0.0f);
        block_request req = { t0, 1, s.head_dim, s.n_heads, s.value, win.data() };
        CHECK(m.serve(nullptr, req), "serve failed (final token)");
        check_window_equals_ref(s, ref, win, t0, 1, "final");
    }
    // Repeated calls of the same window must be identical (graph reuse).
    {
        std::vector<float> a(size_t(s.head_dim) * s.n_heads * 64, 0.0f);
        std::vector<float> b(size_t(s.head_dim) * s.n_heads * 64, 0.0f);
        block_request ra = { 0, 64, s.head_dim, s.n_heads, s.value, a.data() };
        block_request rb = { 0, 64, s.head_dim, s.n_heads, s.value, b.data() };
        CHECK(m.serve(nullptr, ra) && m.serve(nullptr, rb), "repeated serve failed");
        CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0,
                "repeated serve must be identical");
    }
    // A hole (-1) in the map materializes as zeros and is served fine.
    if (s.n_tokens > 130) {
        std::vector<int64_t> hole_map = map;
        hole_map[130] = -1;
        materializer mh(make_mat_cfg(s), make_mat_src(s));
        CHECK(mh.set_prefix(hole_map.data(), s.n_tokens) == mat_status::ok,
                "hole map must be accepted");
        std::vector<float> win(size_t(s.head_dim) * s.n_heads * 64, 0.0f);
        block_request req = { 128, 64, s.head_dim, s.n_heads, s.value, win.data() };
        CHECK(mh.serve(nullptr, req), "serve failed (hole window)");
        for (uint32_t i = 0; i < s.head_dim * s.n_heads; ++i) {
            CHECK(win[size_t(2) * s.head_dim * s.n_heads + i] == 0.0f,
                    "hole cell must materialize as zeros");
        }
        CHECK(std::memcmp(win.data(), ref.data() + size_t(128) * s.head_dim * s.n_heads,
                s.head_dim * s.n_heads * 2 * sizeof(float)) == 0,
                "cells before the hole must match the reference");
    }
}

void test_graph_reuse(const kvarn_source & s) {
    materializer m(make_mat_cfg(s, 1024), make_mat_src(s));
    const std::vector<int64_t> map = make_effective_map(s);
    CHECK(m.set_prefix(map.data(), s.n_tokens) == mat_status::ok, "set_prefix failed");
    std::vector<float> win(size_t(s.head_dim) * s.n_heads * 64, 0.0f);
    block_request req = { 0, 64, s.head_dim, s.n_heads, s.value, win.data() };
    for (int i = 0; i < 5; ++i) {
        CHECK(m.serve(nullptr, req), "reuse serve failed");
    }
    CHECK_EQ(m.graph_count(), 1u, "one window size must reuse one graph");
    // A different size adds a second graph; serving 385 tokens with a 64
    // block uses two sizes (64 and 1).
    uint32_t t0 = 0;
    while (t0 < s.n_tokens) {
        const uint32_t n = std::min<uint32_t>(64, s.n_tokens - t0);
        std::vector<float> w(size_t(s.head_dim) * s.n_heads * n, 0.0f);
        block_request r = { t0, n, s.head_dim, s.n_heads, s.value, w.data() };
        CHECK(m.serve(nullptr, r), "partition serve failed");
        t0 += n;
    }
    CHECK(m.graph_count() <= 2, "partitioned run must reuse at most two graphs");
}

void test_read_only(const kvarn_source & s) {
    materializer m(make_mat_cfg(s), make_mat_src(s));
    const std::vector<int64_t> map = make_effective_map(s);
    CHECK(m.set_prefix(map.data(), s.n_tokens) == mat_status::ok, "set_prefix failed");
    std::vector<uint8_t> stage_before(ggml_nbytes(s.stage), 0);
    std::vector<uint8_t> records_before(ggml_nbytes(s.records), 0);
    ggml_backend_tensor_get(s.stage, stage_before.data(), 0, stage_before.size());
    ggml_backend_tensor_get(s.records, records_before.data(), 0, records_before.size());
    for (uint32_t t0 = 0; t0 < s.n_tokens; t0 += 64) {
        const uint32_t n = std::min<uint32_t>(64, s.n_tokens - t0);
        std::vector<float> win(size_t(s.head_dim) * s.n_heads * n, 0.0f);
        block_request req = { t0, n, s.head_dim, s.n_heads, s.value, win.data() };
        CHECK(m.serve(nullptr, req), "serve failed (read-only)");
    }
    std::vector<uint8_t> stage_after(ggml_nbytes(s.stage), 0);
    std::vector<uint8_t> records_after(ggml_nbytes(s.records), 0);
    ggml_backend_tensor_get(s.stage, stage_after.data(), 0, stage_after.size());
    ggml_backend_tensor_get(s.records, records_after.data(), 0, records_after.size());
    CHECK(stage_before == stage_after, "adapter must not write to the stage");
    CHECK(records_before == records_after, "adapter must not write to the records");
}

void test_validation(const kvarn_source & s) {
    // Every negative case below starts from a FRESH valid config, and the
    // valid baseline is asserted first (no dirty-config false positives).
    const materializer_config base_cfg = make_mat_cfg(s);
    {
        materializer m(base_cfg, make_mat_src(s));
        CHECK(m.valid(), "valid baseline materializer expected");
    }

    // Bad geometry / bad source shapes must fail at construction.
    {
        materializer_config bad = base_cfg;
        bad.head_dim = 100;
        materializer m(bad, make_mat_src(s));
        CHECK(!m.valid(), "head_dim 100 must be rejected");
    }
    {
        materializer_config bad = base_cfg;
        bad.max_window = 0;
        materializer m(bad, make_mat_src(s));
        CHECK(!m.valid(), "max_window 0 must be rejected");
    }
    {
        materializer_config bad = base_cfg;
        bad.max_window = 2048;
        materializer m(bad, make_mat_src(s));
        CHECK(!m.valid(), "max_window > 1024 must be rejected");
    }
    // Wrong tensor types.
    {
        materializer_source bad = make_mat_src(s);
        bad.stage = s.current; // F32, not F16
        materializer m(base_cfg, bad);
        CHECK(!m.valid(), "non-F16 stage must be rejected");
    }
    // Leaf requirement: a store-op view must never be accepted as the stage
    // (it would re-execute producer writes through the graph).
    {
        materializer_source bad = make_mat_src(s);
        bad.stage = s.stored; // GGML_OP_KVARN_STORE view
        materializer m(base_cfg, bad);
        CHECK(!m.valid(), "non-leaf stage must be rejected");
    }
    // A plain GGML_OP_VIEW leaf (e.g. a cache *_stream view) is also not a
    // plain leaf and must be rejected.
    {
        ggml_context * vctx = ggml_init({ 1024 * 1024, nullptr, true });
        CHECK(vctx != nullptr, "view ctx");
        ggml_tensor * view = ggml_view_1d(vctx, s.stage, ggml_nelements(s.stage), 0);
        CHECK(view != nullptr && view->op == GGML_OP_VIEW, "view creation");
        materializer_source bad = make_mat_src(s);
        bad.stage = view;
        materializer m(base_cfg, bad);
        CHECK(!m.valid(), "op VIEW stage must be rejected");
        ggml_free(vctx);
    }
    // Unallocated leaf (no buffer) must be rejected before compute.
    {
        ggml_context * uctx = ggml_init({ 1024 * 1024, nullptr, true });
        CHECK(uctx != nullptr, "unalloc ctx");
        ggml_tensor * unalloc = ggml_new_tensor_3d(uctx, GGML_TYPE_F16,
                s.record_dim, s.n_heads * s.head_slices, 384);
        CHECK(unalloc != nullptr && unalloc->buffer == nullptr, "unallocated tensor");
        materializer_source bad = make_mat_src(s);
        bad.stage = unalloc;
        materializer m(base_cfg, bad);
        CHECK(!m.valid(), "unallocated stage must be rejected");
        ggml_free(uctx);
    }
    // Strided (non-contiguous) leaf must be rejected: the compute kernel
    // asserts contiguity and dereferences device pointers directly.
    {
        ggml_context * sctx = ggml_init({ 4 * 1024 * 1024, nullptr, true });
        CHECK(sctx != nullptr, "strided ctx");
        ggml_tensor * copy = ggml_new_tensor_3d(sctx, GGML_TYPE_F16,
                s.record_dim, s.n_heads * s.head_slices, 384);
        ggml_backend_buffer_t sbuf = ggml_backend_alloc_ctx_tensors(sctx, g_backend);
        CHECK(sbuf != nullptr, "strided buffer");
        std::vector<ggml_fp16_t> stage_data(ggml_nelements(s.stage));
        ggml_backend_tensor_get(s.stage, stage_data.data(), 0, ggml_nbytes(s.stage));
        ggml_backend_tensor_set(copy, stage_data.data(), 0, ggml_nbytes(copy));
        copy->nb[1] *= 2; // break contiguity
        materializer_source bad = make_mat_src(s);
        bad.stage = copy;
        materializer m(base_cfg, bad);
        CHECK(!m.valid(), "strided stage must be rejected");
        ggml_backend_buffer_free(sbuf);
        ggml_free(sctx);
    }
    // Null backend / tensors.
    {
        materializer_source bad = make_mat_src(s);
        bad.backend = nullptr;
        materializer m(base_cfg, bad);
        CHECK(!m.valid(), "null backend must be rejected");
    }
    // Baseline materializer for the map/window validation below.
    materializer m(base_cfg, make_mat_src(s));
    CHECK(m.valid(), "valid materializer expected");

    // Invalid index maps must fail in set_prefix, before any backend compute.
    const std::vector<int64_t> map = make_effective_map(s);
    std::vector<int64_t> bad;

    bad = map;
    bad[0] = uint32_t(s.records->ne[2]) * 128 + 5; // beyond physical capacity
    CHECK(m.set_prefix(bad.data(), s.n_tokens) == mat_status::invalid_indices,
            "out-of-range physical cell must be rejected");
    bad = map;
    bad[0] = encode_stage_cell(0, uint32_t(s.stage_groups)); // slot out of range
    CHECK(m.set_prefix(bad.data(), s.n_tokens) == mat_status::invalid_indices,
            "out-of-range stage slot must be rejected");
    bad = map;
    bad[0] = encode_store_cell(0, 0); // write-side store encoding (high word)
    CHECK(m.set_prefix(bad.data(), s.n_tokens) == mat_status::invalid_indices,
            "store-cell encoding must be rejected for reads");
    bad = map;
    bad[0] = -2; // invalid negative (not a stage encoding)
    CHECK(m.set_prefix(bad.data(), s.n_tokens) == mat_status::invalid_indices,
            "invalid negative index must be rejected");
    CHECK(m.set_prefix(nullptr, s.n_tokens) == mat_status::invalid_args,
            "null map must be rejected");
    CHECK(m.set_prefix(map.data(), 0) == mat_status::invalid_args,
            "zero-length map must be rejected");

    // Window requests beyond max_window or with mismatched geometry fail.
    materializer m2(make_mat_cfg(s, 16), make_mat_src(s));
    CHECK(m2.set_prefix(map.data(), s.n_tokens) == mat_status::ok, "set_prefix failed");
    std::vector<float> win(size_t(s.head_dim) * s.n_heads * 64, 0.0f);
    block_request big = { 0, 64, s.head_dim, s.n_heads, s.value, win.data() };
    CHECK(!m2.serve(nullptr, big), "window > max_window must be rejected");
    CHECK(m2.last_status() == mat_status::invalid_args, "oversized window status");
    block_request geo = { 0, 4, s.head_dim + 1, s.n_heads, s.value, win.data() };
    CHECK(!m2.serve(nullptr, geo), "mismatched geometry window must be rejected");
    block_request oob = { s.n_tokens, 1, s.head_dim, s.n_heads, s.value, win.data() };
    CHECK(!m2.serve(nullptr, oob), "out-of-range window must be rejected");
}

void test_unsupported_backend(const kvarn_source & s) {
    // Test-only mock device/backend whose supports_op always returns false,
    // built from the internal backend structs. The materializer's preflight
    // (ggml_backend_supports_op before allocation/compute) must surface
    // unsupported_backend; the mock is never used for allocation or compute.
    static ggml_backend_device fake_device = {};
    fake_device.iface.supports_op = [](ggml_backend_dev_t, const ggml_tensor *) -> bool {
        return false;
    };
    fake_device.iface.supports_buft = [](ggml_backend_dev_t, ggml_backend_buffer_type_t) -> bool {
        return true; // source tensors validate; only op support is missing
    };
    static ggml_backend fake_backend = {};
    fake_backend.device = &fake_device;

    materializer_source src = make_mat_src(s);
    src.backend = &fake_backend;
    materializer m(make_mat_cfg(s), src);
    CHECK(m.valid(), "mock backend source must validate");
    const std::vector<int64_t> map = make_effective_map(s);
    CHECK(m.set_prefix(map.data(), s.n_tokens) == mat_status::ok, "mock set_prefix ok");
    std::vector<float> win(size_t(s.head_dim) * s.n_heads * 64, 0.0f);
    block_request req = { 0, 64, s.head_dim, s.n_heads, s.value, win.data() };
    CHECK(!m.serve(nullptr, req), "unsupported backend must refuse the window");
    CHECK(m.last_status() == mat_status::unsupported_backend,
            "unsupported backend status");
    CHECK_EQ(m.graph_count(), 0u, "no graph must be cached for an unsupported backend");
}

void test_invalidation(const kvarn_source & s) {
    materializer m(make_mat_cfg(s), make_mat_src(s));
    const std::vector<int64_t> map = make_effective_map(s);
    CHECK(m.set_prefix(map.data(), s.n_tokens) == mat_status::ok, "set_prefix ok");
    std::vector<float> win(size_t(s.head_dim) * s.n_heads * 64, 0.0f);
    block_request req = { 0, 64, s.head_dim, s.n_heads, s.value, win.data() };
    CHECK(m.serve(nullptr, req), "serve ok before invalidation");

    // A failed set_prefix must invalidate the installed map: serving must
    // now fail with invalid_args instead of returning stale data.
    std::vector<int64_t> bad_map(s.n_tokens, 0);
    bad_map[0] = uint32_t(s.records->ne[2]) * 128 + 1; // beyond physical capacity
    CHECK(m.set_prefix(bad_map.data(), s.n_tokens) == mat_status::invalid_indices,
            "bad map rejected");
    CHECK(!m.serve(nullptr, req), "stale prefix must not be served after failed set_prefix");
    CHECK(m.last_status() == mat_status::invalid_args, "stale serve status");

    // Reinstalling a good map restores serving.
    CHECK(m.set_prefix(map.data(), s.n_tokens) == mat_status::ok, "reinstall ok");
    CHECK(m.serve(nullptr, req), "serve ok after reinstall");
}

void test_sparse_physical(const kvarn_source & s) {
    // Real compact_read_plan semantics: the prefix map may reference sparse
    // or high PHYSICAL cells after rollback while the logical token count is
    // small. 3 logical positions mapped to physical cells {0, 130, 384} of a
    // larger allocated source (capacity = records groups x 128 = 512):
    //   - cell 0    -> sink, explicitly encoded stage slot 0;
    //   - cell 130  -> group 1 pos 2, sealed records (plain physical cell);
    //   - cell 384  -> group 3 pos 0, live stage slot 1 (explicit).
    if (s.n_tokens != 385) {
        return;
    }
    const uint32_t capacity = uint32_t(s.records->ne[2]) * 128;
    std::vector<int64_t> map(3);
    map[0] = encode_stage_cell(0, 0);         // sink
    map[1] = 130;                             // records group 1
    map[2] = encode_stage_cell(384, 1);       // live tail in stage slot 1
    std::vector<float> ref;
    CHECK(full_reference(s, map, ref), "sparse physical reference failed");

    materializer m(make_mat_cfg(s), make_mat_src(s));
    CHECK(m.set_prefix(map.data(), 3) == mat_status::ok,
            "sparse physical map must be accepted");
    // Windows over the 3 logical positions: 1-wide and full.
    for (uint32_t t0 = 0; t0 < 3; ++t0) {
        std::vector<float> win(size_t(s.head_dim) * s.n_heads, 0.0f);
        block_request req = { t0, 1, s.head_dim, s.n_heads, s.value, win.data() };
        CHECK(m.serve(nullptr, req), "sparse single serve failed");
        check_window_equals_ref(s, ref, win, t0, 1, "sparse-single");
    }
    {
        std::vector<float> win(size_t(s.head_dim) * s.n_heads * 3, 0.0f);
        block_request req = { 0, 3, s.head_dim, s.n_heads, s.value, win.data() };
        CHECK(m.serve(nullptr, req), "sparse full serve failed");
        check_window_equals_ref(s, ref, win, 0, 3, "sparse-full");
    }
    // Reordered map (physical cells out of logical order) must be accepted
    // and served in map order.
    std::vector<int64_t> reordered(3);
    reordered[0] = encode_stage_cell(384, 1);
    reordered[1] = 130;
    reordered[2] = encode_stage_cell(0, 0);
    std::vector<float> ref2;
    CHECK(full_reference(s, reordered, ref2), "reordered reference failed");
    materializer m2(make_mat_cfg(s), make_mat_src(s));
    CHECK(m2.set_prefix(reordered.data(), 3) == mat_status::ok,
            "reordered map must be accepted");
    {
        std::vector<float> win(size_t(s.head_dim) * s.n_heads * 3, 0.0f);
        block_request req = { 0, 3, s.head_dim, s.n_heads, s.value, win.data() };
        CHECK(m2.serve(nullptr, req), "reordered serve failed");
        check_window_equals_ref(s, ref2, win, 0, 3, "reordered");
    }
    // Cells beyond the physical capacity must be rejected for both plain and
    // explicit stage encodings.
    materializer m3(make_mat_cfg(s), make_mat_src(s));
    std::vector<int64_t> beyond(1);
    beyond[0] = capacity; // plain, group = groups_per_stream
    CHECK(m3.set_prefix(beyond.data(), 1) == mat_status::invalid_indices,
            "plain cell at physical capacity must be rejected");
    beyond[0] = encode_stage_cell(capacity, 0);
    CHECK(m3.set_prefix(beyond.data(), 1) == mat_status::invalid_indices,
            "stage cell at physical capacity must be rejected");
}

void test_group_bounds(const kvarn_source & s) {
    // Records capacity smaller than the cell space the owner wants to use:
    // cells at or beyond the physical capacity (records groups x 128) must
    // be rejected for BOTH plain and explicit stage encodings; cells inside
    // the capacity remain servable from records or explicit stage slots.
    if (s.n_tokens != 385 || s.head_dim != 256) {
        return;
    }
    const uint32_t phys = s.n_heads * s.head_slices;
    const size_t rbytes = (size_t) s.records->ne[0];
    ggml_context * ctx = ggml_init({ 4 * 1024 * 1024, nullptr, true });
    CHECK(ctx != nullptr, "truncated ctx");
    ggml_tensor * small = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, rbytes, phys, 2);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, s.backend);
    CHECK(buf != nullptr, "truncated buffer");
    std::vector<uint8_t> full_bytes(ggml_nbytes(s.records), 0);
    ggml_backend_tensor_get(s.records, full_bytes.data(), 0, full_bytes.size());
    // Copy only group 1 (records group 0 is the sink and is never written).
    const size_t group_bytes = rbytes * phys;
    std::vector<uint8_t> small_bytes(2 * group_bytes, 0);
    std::memcpy(small_bytes.data() + group_bytes, full_bytes.data() + group_bytes, group_bytes);
    ggml_backend_tensor_set(small, small_bytes.data(), 0, small_bytes.size());

    materializer_source src = make_mat_src(s);
    src.records = small;
    materializer m(make_mat_cfg(s), src);
    CHECK(m.valid(), "truncated source must validate");

    // cell 256 (= physical capacity of the truncated source) is rejected for
    // both encodings.
    std::vector<int64_t> bad(1);
    bad[0] = 256; // plain: group 2 >= groups_per_stream 2
    CHECK(m.set_prefix(bad.data(), 1) == mat_status::invalid_indices,
            "plain cell at truncated capacity must be rejected");
    bad[0] = encode_stage_cell(256, 2); // explicit: cell beyond capacity
    CHECK(m.set_prefix(bad.data(), 1) == mat_status::invalid_indices,
            "stage cell at truncated capacity must be rejected");

    // Cells inside the capacity (0..255): sink via explicit slot 0, group 1
    // via plain records; a window over group 1 must match the reference
    // built on the SAME truncated source.
    std::vector<int64_t> good(256);
    for (uint32_t pos = 0; pos < 256; ++pos) {
        good[pos] = pos < 128 ? encode_stage_cell(pos, 0) : pos;
    }
    CHECK(m.set_prefix(good.data(), 256) == mat_status::ok,
            "cells inside truncated capacity must be accepted");
    kvarn_source t = s; // same tensors, truncated records
    t.records = small;
    std::vector<float> ref;
    CHECK(full_reference(t, good, ref), "truncated reference failed");
    std::vector<float> win(size_t(s.head_dim) * s.n_heads * 64, 0.0f);
    block_request req = { 128, 64, s.head_dim, s.n_heads, s.value, win.data() };
    CHECK(m.serve(nullptr, req), "serve failed (truncated)");
    check_window_equals_ref(t, ref, win, 128, 64, "truncated");

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

void test_exact_overlay(const kvarn_source & s) {
    if (s.n_tokens != 385 || s.bits != 4) {
        return;
    }
    // Override tokens 128..133 (sealed records group 1) with the original
    // input rows. The adapter must rotate them into the rotated source
    // domain; the converter de-rotates them back.
    struct overlay_ctx {
        const kvarn_source * src;
        uint32_t t0;
        uint32_t n;
    };
    struct overlay_ctx oc = { &s, 128, 6 };
    auto overlay = [](void * u, uint32_t token, float * rows) -> bool {
        overlay_ctx * c = static_cast<overlay_ctx *>(u);
        if (token < c->t0 || token >= c->t0 + c->n) {
            return false;
        }
        const uint32_t D = c->src->head_dim;
        const uint32_t H = c->src->n_heads;
        const uint32_t rd = c->src->record_dim;
        const uint32_t hs = c->src->head_slices;
        const uint32_t phys = H * hs;
        for (uint32_t h = 0; h < H; ++h) {
            for (uint32_t d = 0; d < D; ++d) {
                const uint32_t slice = hs == 1 ? 0 : d / rd;
                const uint32_t lane = hs == 1 ? d : d % rd;
                const uint32_t ph = h * hs + slice;
                rows[size_t(d) + size_t(h) * D] =
                    c->src->input[size_t(lane) + size_t(ph) * rd + size_t(token) * rd * phys];
            }
        }
        return true;
    };

    const std::vector<int64_t> map = make_effective_map(s);
    materializer m(make_mat_cfg(s), make_mat_src(s));
    CHECK(m.set_prefix(map.data(), s.n_tokens, overlay, &oc) == mat_status::ok,
            "set_prefix with overlay failed");

    // The served window for overridden tokens must equal the rotated input
    // (bit-exact) and differ from the records-based body.
    std::vector<float> win(size_t(s.head_dim) * s.n_heads * 64, 0.0f);
    block_request req = { 128, 64, s.head_dim, s.n_heads, s.value, win.data() };
    CHECK(m.serve(nullptr, req), "serve failed (overlay)");
    const uint32_t row = s.head_dim * s.n_heads;
    for (uint32_t t = 0; t < 6; ++t) {
        // The adapter must return the ORIGINAL input rows rotated into the
        // source domain exactly as the canonical WHT does (single rotation,
        // bit-exact; a second self-inverse rotation is NOT bit-exact in F32).
        std::vector<float> want_row(row);
        const uint32_t D = s.head_dim;
        const uint32_t H = s.n_heads;
        const uint32_t rd = s.record_dim;
        const uint32_t hs = s.head_slices;
        const uint32_t phys = H * hs;
        for (uint32_t h = 0; h < H; ++h) {
            for (uint32_t d = 0; d < D; ++d) {
                const uint32_t slice = hs == 1 ? 0 : d / rd;
                const uint32_t lane = hs == 1 ? d : d % rd;
                const uint32_t ph = h * hs + slice;
                want_row[size_t(d) + size_t(h) * D] =
                    s.input[size_t(lane) + size_t(ph) * rd +
                            size_t(128 + t) * rd * phys];
            }
            ref_wht_head(want_row.data() + size_t(h) * D, (int) D, (int) rd);
        }
        for (uint32_t i = 0; i < row; ++i) {
            const float got = win[size_t(t) * row + i];
            const float want = want_row[i];
            if (got != want) {
                std::fprintf(stderr, "  overlay token=%u i=%u got=%.9g want=%.9g\n",
                        t, i, (double) got, (double) want);
                ++g_failures;
                return;
            }
        }
        // must differ from the body (records quantized) value
        const float body = s.rotated_nonindirect[size_t(128 + t) * row + 0];
        const float over = win[size_t(t) * row + 0];
        if (body == over) {
            std::fprintf(stderr, "  overlay had no effect on token %u\n", t);
            ++g_failures;
        }
    }

    // Chained conversion: with the overlay, the dequantized Q8 payload for
    // the overridden tokens is much closer to the original input.
    config cc;
    cc.dst_type = GGML_TYPE_Q8_0;
    cc.head_dim = s.head_dim;
    cc.n_heads = s.n_heads;
    cc.value = s.value;
    cc.src.record_dim = s.record_dim;
    cc.src.rotated = true;
    cc.dst.wht_width = 0;
    cc.max_tokens_per_block = 64;
    const size_t row_floats = size_t(s.head_dim) * s.n_heads;
    const size_t payload_size = size_t(s.n_tokens) * row_floats / 32 * ggml_type_size(GGML_TYPE_Q8_0);
    std::vector<uint8_t> payload(payload_size, 0);
    const auto res = convert(cc, s.n_tokens, materializer_read_block, &m,
            payload.data(), payload.size());
    CHECK(res.st == status::ok, "chained conversion failed");

    const ggml_type_traits * traits = ggml_get_type_traits(GGML_TYPE_Q8_0);
    std::vector<float> deq(size_t(s.n_tokens) * row_floats);
    for (size_t t = 0; t < s.n_tokens; ++t) {
        const uint8_t * r = payload.data() + t * (row_floats / 32) * traits->type_size;
        for (size_t b = 0; b < row_floats / 32; ++b) {
            traits->to_float(r + b * traits->type_size, deq.data() + t * row_floats + b * 32, 32);
        }
    }
    auto rmse_tokens = [&](uint32_t t0, uint32_t n) {
        double sum = 0.0;
        for (uint32_t t = t0; t < t0 + n; ++t) {
            for (uint32_t i = 0; i < row; ++i) {
                const uint32_t d = i % s.head_dim;
                const uint32_t h = i / s.head_dim;
                const uint32_t slice = s.head_slices == 1 ? 0 : d / s.record_dim;
                const uint32_t lane = s.head_slices == 1 ? d : d % s.record_dim;
                const uint32_t ph = h * s.head_slices + slice;
                const double diff = double(deq[size_t(t) * row_floats + i]) -
                    double(s.input[size_t(lane) + size_t(ph) * s.record_dim +
                            size_t(t) * s.record_dim * s.n_heads * s.head_slices]);
                sum += diff * diff;
            }
        }
        return std::sqrt(sum / double(n) / double(row));
    };
    const double with_overlay = rmse_tokens(128, 6);
    // Without overlay: re-run the chain with a plain materializer.
    materializer plain(make_mat_cfg(s), make_mat_src(s));
    CHECK(plain.set_prefix(map.data(), s.n_tokens) == mat_status::ok, "set_prefix failed");
    const auto res2 = convert(cc, s.n_tokens, materializer_read_block, &plain,
            payload.data(), payload.size());
    CHECK(res2.st == status::ok, "plain chain failed");
    for (size_t t = 0; t < s.n_tokens; ++t) {
        const uint8_t * r = payload.data() + t * (row_floats / 32) * traits->type_size;
        for (size_t b = 0; b < row_floats / 32; ++b) {
            traits->to_float(r + b * traits->type_size, deq.data() + t * row_floats + b * 32, 32);
        }
    }
    const double without_overlay = rmse_tokens(128, 6);
    std::printf("  exact overlay: overridden tokens rmse-with=%.6g rmse-without=%.6g\n",
            with_overlay, without_overlay);
    CHECK(with_overlay < without_overlay * 0.5,
            "exact overlay must reduce the error of overridden tokens");
}

void test_chain_payload(const kvarn_source & s) {
    // Adapter -> converter: the payload must equal a reference quantization
    // of the same effective source (de-rotated full reference).
    materializer m(make_mat_cfg(s), make_mat_src(s));
    const std::vector<int64_t> map = make_effective_map(s);
    CHECK(m.set_prefix(map.data(), s.n_tokens) == mat_status::ok, "set_prefix failed");

    std::vector<float> ref;
    CHECK(full_reference(s, map, ref), "chain reference failed");
    // De-rotate into the original domain (the converter's output domain).
    std::vector<float> orig = ref;
    const uint32_t row = s.head_dim * s.n_heads;
    for (uint32_t t = 0; t < s.n_tokens; ++t) {
        for (uint32_t h = 0; h < s.n_heads; ++h) {
            ref_wht_head(orig.data() + size_t(t) * row + size_t(h) * s.head_dim,
                    (int) s.head_dim, (int) s.record_dim);
        }
    }
    std::vector<float> ref_rows(size_t(s.n_tokens) * row);
    for (uint32_t t = 0; t < s.n_tokens; ++t) {
        for (uint32_t h = 0; h < s.n_heads; ++h) {
            std::memcpy(ref_rows.data() + size_t(t) * row + size_t(h) * s.head_dim,
                    orig.data() + size_t(t) * row + size_t(h) * s.head_dim,
                    s.head_dim * sizeof(float));
        }
    }
    const size_t payload_size = size_t(s.n_tokens) * row / 32 * ggml_type_size(GGML_TYPE_Q4_0);
    std::vector<uint8_t> ref_payload(payload_size);
    const size_t written = ggml_quantize_chunk(GGML_TYPE_Q4_0, ref_rows.data(),
            ref_payload.data(), 0, s.n_tokens, int64_t(row), nullptr);
    CHECK(written == payload_size, "reference quantize size");

    config cc;
    cc.dst_type = GGML_TYPE_Q4_0;
    cc.head_dim = s.head_dim;
    cc.n_heads = s.n_heads;
    cc.value = s.value;
    cc.src.record_dim = s.record_dim;
    cc.src.rotated = true;
    cc.dst.wht_width = 0;
    cc.max_tokens_per_block = 64;
    std::vector<uint8_t> payload(payload_size, 0);
    const auto res = convert(cc, s.n_tokens, materializer_read_block, &m,
            payload.data(), payload.size());
    CHECK(res.st == status::ok, "chained convert failed");
    CHECK(res.payload_bytes == payload_size, "chained payload size");
    CHECK(std::memcmp(payload.data(), ref_payload.data(), payload_size) == 0,
            "chained payload must match the reference quantization");
    std::printf("  chain: %s D%u heads=%u tokens=%u payload_match=yes\n",
            s.value ? "V" : "K", s.head_dim, s.n_heads, s.n_tokens);
}

void test_non_finite(const kvarn_source & s) {
    if (s.n_tokens != 385 || s.head_dim != 256) {
        return;
    }
    // Corrupt one F16 stage row (token 5, sink slot 0) to NaN. The adapter
    // must refuse the window with non_finite_input instead of publishing it.
    ggml_context * ctx = ggml_init({ 4 * 1024 * 1024, nullptr, true });
    CHECK(ctx != nullptr, "nan ctx");
    ggml_tensor * stage_copy = ggml_new_tensor_3d(ctx, GGML_TYPE_F16,
            s.record_dim, s.n_heads * s.head_slices, 384);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, s.backend);
    CHECK(buf != nullptr, "nan buffer");
    std::vector<ggml_fp16_t> stage_data(ggml_nelements(s.stage));
    ggml_backend_tensor_get(s.stage, stage_data.data(), 0, ggml_nbytes(s.stage));
    const uint32_t phys = s.n_heads * s.head_slices;
    stage_data[size_t(3) + size_t(0) * s.record_dim + size_t(5) * s.record_dim * phys] =
        ggml_fp32_to_fp16(std::numeric_limits<float>::quiet_NaN());
    ggml_backend_tensor_set(stage_copy, stage_data.data(), 0, ggml_nbytes(stage_copy));

    materializer_source src = make_mat_src(s);
    src.stage = stage_copy;
    materializer m(make_mat_cfg(s), src);
    CHECK(m.valid(), "nan source must validate");
    const std::vector<int64_t> map = make_effective_map(s);
    CHECK(m.set_prefix(map.data(), s.n_tokens) == mat_status::ok, "set_prefix failed");

    std::vector<float> win(size_t(s.head_dim) * s.n_heads * 64, 0.0f);
    block_request req = { 0, 64, s.head_dim, s.n_heads, s.value, win.data() };
    CHECK(!m.serve(nullptr, req), "NaN window must be refused");
    CHECK(m.last_status() == mat_status::non_finite_input, "NaN status");
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

} // namespace

// Optional GPU backend selection for the principal (never executed here):
// dlopen the runtime loader (libggml.so) only in this path, so the default
// CPU run has no dependency on GPU libraries.
bool init_selected_backend(const std::string & name, const std::string & path) {
#if defined(__linux__)
    if (name == "CPU") {
        g_backend = ggml_backend_cpu_init();
        return g_backend != nullptr;
    }
    void * loader = dlopen((path + "/libggml.so").c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (loader == nullptr) {
        std::fprintf(stderr, "dlopen loader: %s\n", dlerror());
        return false;
    }
    // Exact API types from the header declaration (no linking needed).
    auto load_all = reinterpret_cast<decltype(&ggml_backend_load_all_from_path)>(
            dlsym(loader, "ggml_backend_load_all_from_path"));
    auto init_by_name = reinterpret_cast<decltype(&ggml_backend_init_by_name)>(
            dlsym(loader, "ggml_backend_init_by_name"));
    if (load_all == nullptr || init_by_name == nullptr) {
        std::fprintf(stderr, "loader symbols missing\n");
        return false;
    }
    load_all(path.c_str());
    g_backend = init_by_name(name.c_str(), nullptr);
    return g_backend != nullptr;
#else
    (void) name;
    (void) path;
    return false;
#endif
}

// Machine-independent resolution for the backend library directory:
// explicit --backend-path, then BEELAMA_BASELINE_DIR, then <cwd>/bin.
// No hardcoded host paths.
bool resolve_backend_path(const std::string & explicit_path, std::string & out) {
    if (!explicit_path.empty()) {
        out = explicit_path;
        return true;
    }
    const char * env = std::getenv("BEELAMA_BASELINE_DIR");
    if (env != nullptr && env[0] != '\0') {
        out = env;
        return true;
    }
    FILE * probe = std::fopen("./bin/libggml.so", "rb");
    if (probe != nullptr) {
        std::fclose(probe);
        out = "./bin";
        return true;
    }
    return false;
}

int main(int argc, char ** argv) {
    std::string backend_name = "CPU";
    std::string backend_path;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--backend" && i + 1 < argc) {
            backend_name = argv[++i];
        } else if (arg == "--backend-path" && i + 1 < argc) {
            backend_path = argv[++i];
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return 2;
        }
    }
    std::string resolved_path;
    if (backend_name != "CPU" && !resolve_backend_path(backend_path, resolved_path)) {
        std::fprintf(stderr,
                "no backend library directory for --backend %s: pass --backend-path <dir>, "
                "set BEELAMA_BASELINE_DIR, or provide <cwd>/bin/libggml.so\n",
                backend_name.c_str());
        return 2;
    }
    if (!init_selected_backend(backend_name, backend_name == "CPU" ? "" : resolved_path)) {
        std::fprintf(stderr, "backend %s unavailable\n", backend_name.c_str());
        return 2;
    }
    std::printf("test-kv-kvarn-materializer: bounded backend-aware read_block adapter (backend=%s)\n",
            backend_name.c_str());
    std::printf("KVarN source built with ggml_kvarn_store + ggml_kvarn_materialize\n");

    struct geom { uint32_t head_dim; uint32_t n_heads; bool value; };
    const geom geoms[] = {
        { 256, 4, false },
        { 128, 3, false },
        {  64, 4, true  },
        { 512, 2, false },
    };
    const uint32_t tokens[] = { 127, 128, 129, 385 };

    for (const geom & g : geoms) {
        for (uint32_t n_tokens : tokens) {
            kvarn_source s;
            build_source(s, n_tokens, g.head_dim, g.n_heads, 4, g.value);
            std::printf("source D%u heads=%u tokens=%u bits=4 %s\n",
                    g.head_dim, g.n_heads, n_tokens, g.value ? "V" : "K");
            test_window_serving(s);
            test_graph_reuse(s);
            test_read_only(s);
            test_validation(s);
            test_unsupported_backend(s);
            test_invalidation(s);
            test_chain_payload(s);
            test_exact_overlay(s);
            test_sparse_physical(s);
            test_group_bounds(s);
            test_non_finite(s);
            free_source(s);
        }
    }
    // A KVarN bits-8 source: the adapter must pass through whatever the
    // records hold; the chain comparison still holds.
    {
        kvarn_source s;
        build_source(s, 385, 256, 4, 8, false);
        std::printf("source D256 heads=4 tokens=385 bits=8 K\n");
        test_chain_payload(s);
        test_window_serving(s);
        free_source(s);
    }

    std::printf("== summary: %s\n", g_failures == 0 ? "ALL PASS" : "FAILURES PRESENT");
    ggml_backend_free(g_backend);
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}