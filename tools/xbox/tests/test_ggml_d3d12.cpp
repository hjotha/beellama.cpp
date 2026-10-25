// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/ggml_d3d12.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace xllama;

namespace {

D3d12MatmulDesc decode_desc(ggml_type t, std::int64_t n, std::int64_t k) {
    D3d12MatmulDesc d;
    d.src0_type = t;
    d.ne00 = k;
    d.ne01 = n;
    d.ne10 = k;
    d.ne11 = 1;
    d.src0_in_weight_buffer = true;
    return d;
}

// Max |emulated - reference| / max |reference| for one type and shape, with
// padded activation/output strides. Reference: ggml's own dequantizer, double sums.
double emulate_rel_err(ggml_type t, int n, int k, int ncols, std::size_t x_pad, std::size_t y_pad) {
    std::mt19937 rng(42u + static_cast<unsigned>(n * 7 + k + ncols));
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    std::vector<float> wf(static_cast<std::size_t>(n) * k);
    for (float& v : wf)
        v = uni(rng);
    const std::size_t row_bytes = ggml_row_size(t, k);
    std::vector<std::uint8_t> q(row_bytes * n + 8); // slack: 2-byte-aligned dword reads
    ggml_quantize_chunk(t, wf.data(), q.data(), 0, n, k, nullptr);

    const std::size_t x_stride = static_cast<std::size_t>(k) + x_pad;
    const std::size_t y_stride = static_cast<std::size_t>(n) + y_pad;
    std::vector<float> x(x_stride * ncols, 1e30f); // padding must never be read
    for (int c = 0; c < ncols; ++c)
        for (int i = 0; i < k; ++i)
            x[c * x_stride + i] = uni(rng);
    std::vector<float> y(y_stride * ncols, -7.f);
    d3d12_mmv_emulate(t, q.data(), row_bytes, x.data(), x_stride, y.data(), y_stride, n, k, ncols);

    const auto* traits = ggml_get_type_traits(t);
    std::vector<float> wrow(static_cast<std::size_t>(k));
    double max_ref = 0.0, max_diff = 0.0;
    for (int r = 0; r < n; ++r) {
        traits->to_float(q.data() + r * row_bytes, wrow.data(), k);
        for (int c = 0; c < ncols; ++c) {
            double acc = 0.0;
            for (int i = 0; i < k; ++i)
                acc += static_cast<double>(wrow[i]) * x[c * x_stride + i];
            max_ref = std::max(max_ref, std::fabs(acc));
            max_diff = std::max(max_diff, std::fabs(acc - y[c * y_stride + r]));
        }
    }
    // Output padding untouched.
    for (int c = 0; c < ncols; ++c)
        for (std::size_t p = n; p < y_stride; ++p)
            if (y[c * y_stride + p] != -7.f)
                return 1e9;
    return max_ref > 0.0 ? max_diff / max_ref : max_diff;
}

} // namespace

TEST_CASE("ggml_d3d12: weight types") {
    CHECK(d3d12_weight_type_supported(GGML_TYPE_Q4_0));
    CHECK(d3d12_weight_type_supported(GGML_TYPE_Q4_K));
    CHECK(d3d12_weight_type_supported(GGML_TYPE_Q5_K));
    CHECK(d3d12_weight_type_supported(GGML_TYPE_Q6_K));
    CHECK(d3d12_weight_type_supported(GGML_TYPE_Q8_0));
    CHECK_FALSE(d3d12_weight_type_supported(GGML_TYPE_F16));
    CHECK_FALSE(d3d12_weight_type_supported(GGML_TYPE_F32));
}

TEST_CASE("ggml_d3d12: supports_op rules for MUL_MAT") {
    D3d12MatmulDesc d = decode_desc(GGML_TYPE_Q4_K, 2048, 2048);
    CHECK(d3d12_mm_supported(d));

    // The buft probe: a weight still in D3D12_Host is refused, so llama.cpp
    // moves on to D3D12_Weights.
    D3d12MatmulDesc host = d;
    host.src0_in_weight_buffer = false;
    CHECK_FALSE(d3d12_mm_supported(host));

    D3d12MatmulDesc empty = decode_desc(GGML_TYPE_Q6_K, 248320, 2560);
    empty.ne11 = 0; // native catch-up has no logits; keep its LM head on the GPU
    CHECK(d3d12_mm_supported(empty));
    empty.src0_in_weight_buffer = false;
    CHECK_FALSE(d3d12_mm_supported(empty));
    empty.src0_in_weight_buffer = true;
    empty.src0_type = GGML_TYPE_IQ4_NL;
    CHECK_FALSE(d3d12_mm_supported(empty));
    empty.src0_type = GGML_TYPE_Q6_K;
    empty.src1_type = GGML_TYPE_F16;
    CHECK_FALSE(d3d12_mm_supported(empty));

    D3d12MatmulDesc prefill = d; // the loader probes with 512 columns
    prefill.ne11 = 512;
    CHECK(d3d12_mm_supported(prefill));

    D3d12MatmulDesc t = d;
    t.src0_type = GGML_TYPE_F16;
    CHECK_FALSE(d3d12_mm_supported(t));
    t = d;
    t.src1_type = GGML_TYPE_F16;
    CHECK_FALSE(d3d12_mm_supported(t));
    t = d;
    t.dst_type = GGML_TYPE_F16;
    CHECK_FALSE(d3d12_mm_supported(t));
    t = d;
    t.ne00 = t.ne10 = 2080; // not a multiple of 256
    CHECK_FALSE(d3d12_mm_supported(t));
    t = d;
    t.ne10 = 1024; // K mismatch
    CHECK_FALSE(d3d12_mm_supported(t));
    t = d;
    t.ne12 = 4; // batched attention-style matmul
    CHECK_FALSE(d3d12_mm_supported(t));
    t = d;
    t.ne02 = 2; // 3-D weight
    CHECK_FALSE(d3d12_mm_supported(t));
    t = d;
    t.src1_contiguous = false;
    CHECK_FALSE(d3d12_mm_supported(t));
    t = d;
    t.src0_contiguous = false;
    CHECK_FALSE(d3d12_mm_supported(t));

    // Coder-3B tied lm_head: 151936 rows → 37984 groups, inside the limit.
    CHECK(d3d12_mm_supported(decode_desc(GGML_TYPE_Q6_K, 151936, 2048)));
    CHECK_FALSE(d3d12_mm_supported(decode_desc(GGML_TYPE_Q6_K, 4 * 65536, 2048)));
}

TEST_CASE("ggml_d3d12: dispatch planner") {
    D3d12Dispatch d = d3d12_mm_dispatch(2048, 1);
    CHECK(d.ok);
    CHECK(d.groups_x == 512);
    CHECK(d.groups_y == 1);
    d = d3d12_mm_dispatch(6, 512);
    CHECK(d.groups_x == 2);
    CHECK(d.groups_y == 512);
    CHECK(d3d12_mm_dispatch(4 * 65535, 1).ok);
    CHECK_FALSE(d3d12_mm_dispatch(4 * 65535 + 1, 1).ok);
    CHECK_FALSE(d3d12_mm_dispatch(256, 65536).ok);
    CHECK_FALSE(d3d12_mm_dispatch(0, 1).ok);
    CHECK_FALSE(
        d3d12_mm_dispatch(248320, 0).ok); // no empty dispatch, even when placement is accepted
}

TEST_CASE("ggml_d3d12: q8_0 eh_proj widths and CPU/GPU switch") {
    const bool original = d3d12_q8_enabled();
    CHECK(original);
    CHECK(ggml_blck_size(GGML_TYPE_Q8_0) == 32);
    CHECK(ggml_type_size(GGML_TYPE_Q8_0) == 34);
    CHECK(ggml_row_size(GGML_TYPE_Q8_0, 5120) == 5440);
    for (int width : {1, 2, 3, 4, 5, 512}) {
        D3d12MatmulDesc d = decode_desc(GGML_TYPE_Q8_0, 2560, 5120);
        d.ne11 = width;
        d3d12_set_q8_enabled(true);
        CHECK(d3d12_mm_supported(d));
        CHECK(d3d12_mm_dispatch(2560, width).groups_x == 640);
        CHECK(d3d12_mm_dispatch(2560, width).groups_y == static_cast<unsigned>(width));
        CHECK_FALSE(d3d12_2col_allowlisted(GGML_TYPE_Q8_0, 2560, 5120, width));
        d.src0_in_weight_buffer = false;
        CHECK_FALSE(d3d12_mm_supported(d));
        d.src0_in_weight_buffer = true;
        d3d12_set_q8_enabled(false);
        CHECK_FALSE(d3d12_mm_supported(d));
        CHECK(d3d12_mm_supported(decode_desc(GGML_TYPE_Q4_K, 2560, 5120)));
    }
    d3d12_set_q8_enabled(original);

    // Identical input prefixes must produce identical rows across B3/B4;
    // padded X/Y and odd N also catch column/row addressing mistakes.
    constexpr int n = 13, k = 5120, xs = k + 5, ys = n + 7;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    const std::size_t row_bytes = ggml_row_size(GGML_TYPE_Q8_0, k);
    std::vector<std::uint8_t> q(row_bytes * n);
    REQUIRE(d3d12_gen_weights(GGML_TYPE_Q8_0, n, k, rng, q.data()));
    std::vector<float> x(xs * 5, 1e30f), full(ys * 5, -7.f);
    for (int c = 0; c < 5; ++c)
        for (int i = 0; i < k; ++i)
            x[c * xs + i] = uni(rng);
    d3d12_mmv_emulate(GGML_TYPE_Q8_0, q.data(), row_bytes, x.data(), xs, full.data(), ys, n, k, 5);
    for (int width : {1, 2, 3, 4}) {
        std::vector<float> prefix(ys * width, -7.f);
        d3d12_mmv_emulate(GGML_TYPE_Q8_0, q.data(), row_bytes, x.data(), xs, prefix.data(), ys, n,
                          k, width);
        CHECK(std::equal(prefix.begin(), prefix.end(), full.begin()));
        for (int c = 0; c < width; ++c)
            for (int p = n; p < ys; ++p)
                CHECK(prefix[c * ys + p] == -7.f);
    }
}

TEST_CASE("ggml_d3d12: two-column dispatch planner") {
    // Same row groups as the base planner, half (ceil) the column groups.
    D3d12Dispatch d = d3d12_mm_dispatch_2col(2048, 1);
    CHECK(d.ok);
    CHECK(d.groups_x == 512);
    CHECK(d.groups_y == 1);
    d = d3d12_mm_dispatch_2col(9216, 5);
    CHECK(d.ok);
    CHECK(d.groups_x == 2304);
    CHECK(d.groups_y == 3); // tail group covers column 4 alone
    d = d3d12_mm_dispatch_2col(1023, 2);
    CHECK(d.ok);
    CHECK(d.groups_x == 256); // ceil(1023/4): the row-clamp group exists
    CHECK(d.groups_y == 1);
    CHECK(d3d12_mm_dispatch_2col(6, 512).groups_y == 256);
    CHECK(d3d12_mm_dispatch_2col(4 * 65535, 2).ok);
    CHECK_FALSE(d3d12_mm_dispatch_2col(4 * 65535 + 1, 2).ok);
    CHECK_FALSE(d3d12_mm_dispatch_2col(256, 2 * 65535 + 1).ok);
    CHECK_FALSE(d3d12_mm_dispatch_2col(0, 2).ok);
    CHECK_FALSE(d3d12_mm_dispatch_2col(256, 0).ok);
}

TEST_CASE("ggml_d3d12: GDN narrow contract and native CPU reference") {
    const bool previous = d3d12_gdn_enabled();
    CHECK_FALSE(previous);
    CHECK_FALSE(d3d12_gdn_supported(nullptr));
    ggml_init_params ip = {};
    ip.mem_size = 24 * ggml_tensor_overhead();
    ip.no_alloc = true;
    ggml_context* ctx = ggml_init(ip);
    REQUIRE(ctx != nullptr);
    ggml_tensor* q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 16, 3, 1);
    ggml_tensor* k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 16, 3, 1);
    ggml_tensor* v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 32, 3, 1);
    ggml_tensor* g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 32, 3, 1);
    ggml_tensor* b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 32, 3, 1);
    ggml_tensor* s = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 128, 32, 1);
    ggml_tensor* op = ggml_gated_delta_net(ctx, q, k, v, g, b, s, 5);
    CHECK_FALSE(d3d12_gdn_supported(op));
    d3d12_set_gdn_enabled(true);
    CHECK(d3d12_gdn_supported(op));
    CHECK(op->ne[0] == 4096);
    CHECK(op->ne[1] == 3 + 5 * 128);
    op->src[6] = q; // rows-indexed mode is outside this shader's contract
    CHECK_FALSE(d3d12_gdn_supported(op));
    op->src[6] = nullptr;
    op->op_params[1] = 1; // raw gates
    CHECK_FALSE(d3d12_gdn_supported(op));
    op->op_params[1] = 0;
    op->op_params[0] = 18;
    CHECK_FALSE(d3d12_gdn_supported(op));
    op->op_params[0] = 5;
    g->ne[0] = 128; // KDA vector gate
    CHECK_FALSE(d3d12_gdn_supported(op));
    g->ne[0] = 1;
    q->type = GGML_TYPE_F16;
    CHECK_FALSE(d3d12_gdn_supported(op));
    q->type = GGML_TYPE_F32;
    q->nb[1] += 1; // cannot encode this stride as float indices
    CHECK_FALSE(d3d12_gdn_supported(op));
    q->nb[1] -= 1;
    v->ne[3] = 2;
    CHECK_FALSE(d3d12_gdn_supported(op));
    v->ne[3] = 1;
    CHECK(d3d12_gdn_supported(op));
    for (int tokens : {1, 2}) {
        ggml_tensor* qt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 16, tokens, 1);
        ggml_tensor* kt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 16, tokens, 1);
        ggml_tensor* vt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 32, tokens, 1);
        ggml_tensor* gt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 32, tokens, 1);
        ggml_tensor* bt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 32, tokens, 1);
        ggml_tensor* width_op = ggml_gated_delta_net(ctx, qt, kt, vt, gt, bt, s, 5);
        d3d12_set_gdn_enabled(true);
        CHECK(d3d12_gdn_supported(width_op)); // both widths satisfy the full GPU contract
        d3d12_set_gdn_enabled(true, 2);
        CHECK(d3d12_gdn_supported(width_op) == (tokens == 2)); // hybrid CPU T1 / GPU T2
    }
    d3d12_set_gdn_enabled(previous);
    ggml_free(ctx);

    // Actual ggml CPU GDN op vs eight-lane emulation on Linux / HLSL on Xbox.
    // Includes every written rollback snapshot, untouched old slots/inputs,
    // T1..5/64, K1/5, padded Q/K/V and exact prefix invariance across T/K.
    std::vector<D3d12SelftestRow> rows;
    run_d3d12_gdn_selftest(&rows);
    REQUIRE(rows.size() == 14);
    for (const auto& row : rows) {
        CAPTURE(row.type);
        CAPTURE(row.ncols);
        CAPTURE(row.pads);
        CAPTURE(row.error);
        CHECK(row.ok);
        CHECK(row.rel_err <= 1e-4);
        CHECK(row.cpu_ms >= 0);
#if !defined(_WIN32)
        CHECK_FALSE(row.d3d12_ran);
#endif
    }
    CHECK(d3d12_gdn_enabled() == previous);
}

TEST_CASE("ggml_d3d12: Q6 LM-head tiles preserve columns, strides and tails") {
    const int previous = d3d12_q6_columns();
    CHECK(previous == 1);
    for (int columns : {0, 1, 2, 4}) {
        d3d12_set_q6_columns(columns);
        CHECK(d3d12_q6_columns() == columns);
    }
    for (int columns : {-1, 3, 8}) {
        d3d12_set_q6_columns(columns);
        CHECK(d3d12_q6_columns() == 1);
    }
    d3d12_set_q6_columns(0);
    CHECK(d3d12_q6_columns_for(248320, 2560, 2) == 2);
    CHECK(d3d12_q6_columns_for(248320, 2560, 3) == 4);
    CHECK(d3d12_q6_columns_for(248320, 2560, 5) == 4);
    CHECK(d3d12_q6_columns_for(248320, 2560, 1) == 1);
    CHECK(d3d12_q6_columns_for(248320, 2560, 4) == 1);
    CHECK(d3d12_q6_columns_for(248319, 2560, 3) == 1);
    d3d12_set_q6_columns(previous);
    for (int width : {2, 3, 5}) {
        CHECK(d3d12_q6_allowlisted(248320, 2560, width));
        const auto d2 = d3d12_mm_dispatch_2col(248320, width);
        const auto d4 = d3d12_mm_dispatch_4col(248320, width);
        CHECK(d2.ok);
        CHECK(d4.ok);
        CHECK(d2.groups_x == 62080);
        CHECK(d4.groups_x == 62080);
        CHECK(d2.groups_y == static_cast<unsigned>((width + 1) / 2));
        CHECK(d4.groups_y == static_cast<unsigned>((width + 3) / 4));
    }
    for (int width : {0, 1, 4, 64})
        CHECK_FALSE(d3d12_q6_allowlisted(248320, 2560, width));
    CHECK_FALSE(d3d12_q6_allowlisted(248319, 2560, 3));
    CHECK_FALSE(d3d12_q6_allowlisted(248320, 4096, 3));
    CHECK(d3d12_mm_dispatch_4col(13, 5).groups_x == 4);
    CHECK(d3d12_mm_dispatch_4col(13, 5).groups_y == 2);
    CHECK(d3d12_mm_dispatch_4col(4 * 65535, 4 * 65535).ok);
    CHECK_FALSE(d3d12_mm_dispatch_4col(4 * 65535 + 1, 1).ok);
    CHECK_FALSE(d3d12_mm_dispatch_4col(1, 4 * 65535 + 1).ok);
    CHECK_FALSE(d3d12_mm_dispatch_4col(1, 0).ok);

    // Small isolated matrices exercise 64/128-thread paths, odd row tails,
    // 2-byte-aligned packed rows, padded X/Y/W and missing tile columns.
    constexpr int n = 13, ys = n + 7;
    for (int k : {2304, 4096}) {
        CAPTURE(k);
        const auto row_bytes = ggml_row_size(GGML_TYPE_Q6_K, k);
        const auto pitch = row_bytes + 2;
        const int xs = k + 13;
        std::mt19937 rng(19u + k);
        std::uniform_real_distribution<float> uni(-1.f, 1.f);
        std::vector<std::uint8_t> flat(row_bytes * n), w(pitch * n + 8, 0xAB);
        REQUIRE(d3d12_gen_weights(GGML_TYPE_Q6_K, n, k, rng, flat.data()));
        for (int r = 0; r < n; ++r)
            std::memcpy(w.data() + r * pitch, flat.data() + r * row_bytes, row_bytes);
        std::vector<float> x(xs * 5, 1e30f);
        for (int c = 0; c < 5; ++c)
            for (int i = 0; i < k; ++i)
                x[c * xs + i] = uni(rng);
        for (int width : {2, 3, 5}) {
            CAPTURE(width);
            std::vector<float> old(ys * 5, -7.f);
            d3d12_mmv_emulate(GGML_TYPE_Q6_K, w.data(), pitch, x.data(), xs, old.data(), ys, n, k,
                              width);
            for (int columns : {2, 4}) {
                CAPTURE(columns);
                std::vector<float> tiled(ys * 5, -7.f);
                d3d12_q6_tile_emulate(w.data(), pitch, x.data(), xs, tiled.data(), ys, n, k, width,
                                      columns);
                CHECK(std::memcmp(old.data(), tiled.data(), old.size() * sizeof(float)) == 0);
                std::vector<float> dequant(k);
                double max_ref = 0, max_diff = 0;
                for (int r = 0; r < n; ++r) {
                    ggml_get_type_traits(GGML_TYPE_Q6_K)
                        ->to_float(flat.data() + r * row_bytes, dequant.data(), k);
                    for (int c = 0; c < width; ++c) {
                        double ref = 0;
                        for (int i = 0; i < k; ++i)
                            ref += static_cast<double>(dequant[i]) * x[c * xs + i];
                        max_ref = std::max(max_ref, std::fabs(ref));
                        max_diff = std::max(max_diff, std::fabs(ref - tiled[c * ys + r]));
                    }
                }
                CHECK(max_diff / max_ref <= 1e-4);
            }
        }
    }
#if !defined(_WIN32)
    CHECK_FALSE(d3d12_q6_tiled_available(2));
    CHECK_FALSE(d3d12_q6_tiled_available(4));
    CHECK(d3d12_q6_tiled_matmuls() == 0);
#endif
}

TEST_CASE("ggml_d3d12: padded layouts match a strided double reference") {
    // Mirrors run_paired_case's padded setup: flat generation, strided memcpy
    // over a sentinel fill, sentinel-padded X, then the shader-math emulation
    // against a double reference that pitches rows at nb[1] (NOT the packed
    // size — passing the packed size misaligns every row past the first, the
    // exact rev58 bench bug this test pins).
    const ggml_type t = GGML_TYPE_Q4_K;
    const int n = 35, k = 512, ncols = 3;
    const int x_pad = 13, y_pad = 7;
    const std::size_t w_pad = 64;
    const std::size_t row_bytes = ggml_row_size(t, k);
    const std::size_t w_nb1 = row_bytes + w_pad;
    const std::size_t xs = static_cast<std::size_t>(k) + x_pad;
    const std::size_t ys = static_cast<std::size_t>(n) + y_pad;
    std::mt19937 rng(1234u + static_cast<unsigned>(n + k));
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    std::vector<std::uint8_t> qflat(static_cast<std::size_t>(n) * row_bytes);
    REQUIRE(d3d12_gen_weights(t, n, k, rng, qflat.data()));
    // nbytes-style sizing (like the device buffers): no storage past the last
    // valid row/column, so padding checks skip the final row/column exactly as
    // check_float_pad/check_byte_pad do on device readback.
    std::vector<std::uint8_t> q(row_bytes + static_cast<std::size_t>(n - 1) * w_nb1, 0xAB);
    for (int r = 0; r < n; ++r)
        std::memcpy(q.data() + static_cast<std::size_t>(r) * w_nb1,
                    qflat.data() + static_cast<std::size_t>(r) * row_bytes, row_bytes);
    std::vector<float> xf(static_cast<std::size_t>(k) + static_cast<std::size_t>(ncols - 1) * xs,
                          1e30f);
    for (int c = 0; c < ncols; ++c)
        for (int i = 0; i < k; ++i)
            xf[static_cast<std::size_t>(c) * xs + i] = uni(rng);
    std::vector<float> y(static_cast<std::size_t>(n) + static_cast<std::size_t>(ncols - 1) * ys,
                         -7.f);
    d3d12_mmv_emulate(t, q.data(), w_nb1, xf.data(), xs, y.data(), ys, n, k, ncols);

    const auto* traits = ggml_get_type_traits(t);
    std::vector<float> wrow(k);
    double max_ref = 0.0, max_diff = 0.0;
    for (int r = 0; r < n; ++r) {
        traits->to_float(q.data() + static_cast<std::size_t>(r) * w_nb1, wrow.data(), k);
        for (int c = 0; c < ncols; ++c) {
            double acc = 0.0;
            for (int i = 0; i < k; ++i)
                acc += static_cast<double>(wrow[i]) * xf[static_cast<std::size_t>(c) * xs + i];
            max_ref = std::max(max_ref, std::fabs(acc));
            max_diff = std::max(max_diff, std::fabs(acc - y[static_cast<std::size_t>(c) * ys + r]));
        }
    }
    CHECK(max_ref > 0.0);
    CHECK(max_diff / max_ref <= 1e-4);
    // Padding untouched by construction here; the device run proves the kernel.
    // Final row/column carry no padding storage (nbytes sizing), mirrored skips.
    for (int c = 0; c + 1 < ncols; ++c) {
        for (std::size_t i = k; i < xs; ++i)
            CHECK(xf[static_cast<std::size_t>(c) * xs + i] == 1e30f);
        for (std::size_t i = n; i < ys; ++i)
            CHECK(y[static_cast<std::size_t>(c) * ys + i] == -7.f);
    }
    for (int r = 0; r + 1 < n; ++r)
        for (std::size_t i = row_bytes; i < w_nb1; ++i)
            CHECK(q[static_cast<std::size_t>(r) * w_nb1 + i] == 0xAB);
}

TEST_CASE("ggml_d3d12: two-column allowlist is explicit, not a rule") {
    // Rev57 same-run pair winners: five q4_k shapes at B=2/3/5.
    for (int b : {2, 3, 5}) {
        CHECK(d3d12_2col_allowlisted(GGML_TYPE_Q4_K, 2560, 4096, b));
        CHECK(d3d12_2col_allowlisted(GGML_TYPE_Q4_K, 4096, 2560, b));
        CHECK(d3d12_2col_allowlisted(GGML_TYPE_Q4_K, 8192, 2560, b));
        CHECK(d3d12_2col_allowlisted(GGML_TYPE_Q4_K, 2560, 9216, b));
        CHECK(d3d12_2col_allowlisted(GGML_TYPE_Q4_K, 9216, 2560, b));
    }
    // Everything else stays OLD: B=1 (nothing to tile), B=4 and prefill
    // widths (unmeasured), small shapes (measured regressions), other types.
    for (int b : {1, 4, 7, 512}) {
        CHECK_FALSE(d3d12_2col_allowlisted(GGML_TYPE_Q4_K, 9216, 2560, b));
        CHECK_FALSE(d3d12_2col_allowlisted(GGML_TYPE_Q4_K, 2560, 4096, b));
    }
    CHECK_FALSE(d3d12_2col_allowlisted(GGML_TYPE_Q4_K, 1024, 2560, 2));
    CHECK_FALSE(d3d12_2col_allowlisted(GGML_TYPE_Q4_K, 1023, 2560, 2));
    CHECK_FALSE(d3d12_2col_allowlisted(GGML_TYPE_Q4_K, 1, 256, 2));
    CHECK_FALSE(d3d12_2col_allowlisted(GGML_TYPE_Q5_K, 8192, 2560, 5));
    CHECK_FALSE(d3d12_2col_allowlisted(GGML_TYPE_Q6_K, 2560, 9216, 5));
    CHECK_FALSE(d3d12_2col_allowlisted(GGML_TYPE_Q4_0, 1024, 1024, 2));
    CHECK_FALSE(d3d12_2col_allowlisted(GGML_TYPE_Q4_K, 0, 2560, 2));
}

TEST_CASE("ggml_d3d12: kernel width follows K (D2a runs 1 and 2)") {
    CHECK(d3d12_mm_threads(1024) == 64); // lm_head of LFM2.5-350M
    CHECK(d3d12_mm_threads(2048) == 64); // most projections
    CHECK(d3d12_mm_threads(3840) == 64);
    CHECK(d3d12_mm_threads(4096) == 128);  // first width with 16 chunks
    CHECK(d3d12_mm_threads(8192) == 128);  // LFM2.5-1.2B ffn_down
    CHECK(d3d12_mm_threads(11008) == 128); // Coder-3B ffn_down
}

TEST_CASE("ggml_d3d12: kernel emulation matches ggml dequantizers") {
    struct Case {
        ggml_type t;
        int n, k;
    };
    const Case cases[] = {
        {GGML_TYPE_Q4_0, 13, 1024},
        {GGML_TYPE_Q4_K, 13, 1024},
        {GGML_TYPE_Q5_K, 13, 1024},
        {GGML_TYPE_Q6_K, 13, 1024},
        {GGML_TYPE_Q8_0, 13, 256},
        {GGML_TYPE_Q8_0, 13, 5120},
        // 9 super-blocks: 1890-byte rows, odd rows start 2-byte aligned.
        {GGML_TYPE_Q6_K, 6, 2304},
        // Coder-3B ffn_down width: 43 super-blocks, 9030-byte rows.
        {GGML_TYPE_Q6_K, 3, 11008},
        {GGML_TYPE_Q4_0, 5, 2304},
        // Qwen3.5-4B attention and SSM projections: 2560 and 4096 wide.
        {GGML_TYPE_Q5_K, 5, 2560},
        {GGML_TYPE_Q5_K, 3, 4096},
        {GGML_TYPE_Q5_K, 11, 5120},
        // Two-column tile inputs: odd row counts with several columns and pads.
        {GGML_TYPE_Q4_K, 13, 2560},
        {GGML_TYPE_Q4_K, 1023, 2560},
    };
    for (const auto& c : cases) {
        CAPTURE(ggml_type_name(c.t));
        CAPTURE(c.k);
        CHECK(emulate_rel_err(c.t, c.n, c.k, 1, 0, 0) <= 1e-4);
        CHECK(emulate_rel_err(c.t, c.n, c.k, 7, 12, 3) <= 1e-4);
        CHECK(emulate_rel_err(c.t, c.n, c.k, 3, 5, 7) <= 1e-4);
        CHECK(emulate_rel_err(c.t, c.n, c.k, 5, 1, 9) <= 1e-4);
    }
    CHECK(ggml_row_size(GGML_TYPE_Q6_K, 11008) % 4 == 2); // the case the ld32 trick covers
    // Q5_K blocks are 176 B, so every dword the shader reads stays inside a block
    // and its loads need no alignment trick.
    CHECK(ggml_row_size(GGML_TYPE_Q5_K, 2560) % 4 == 0);
}

TEST_CASE("ggml_d3d12: selftest CSV and non-Windows behaviour") {
    D3d12SelftestRow r;
    r.type = "q6_k";
    r.n = 2048;
    r.k = 11008;
    r.ncols = 1;
    r.error = "a,b";
    const std::string line = format_d3d12_selftest_row(r, "host");
    int commas = 0, hcommas = 0;
    for (char ch : line)
        commas += ch == ',' ? 1 : 0;
    for (const char* p = d3d12_selftest_csv_header(); *p; ++p)
        hcommas += *p == ',' ? 1 : 0;
    CHECK(commas == hcommas);
#if !defined(_WIN32)
    CHECK_FALSE(ggml_d3d12_register());
    std::vector<D3d12SelftestRow> rows;
    run_d3d12_selftest(&rows);
    REQUIRE(rows.size() == 1);
    CHECK_FALSE(rows[0].d3d12_ran);
    CHECK_FALSE(rows[0].error.empty());
#endif
}

TEST_CASE("ggml_d3d12: bounded weight generation is byte-identical to one-shot") {
    const struct {
        ggml_type t;
        int n, k;
    } cases[] = {
        {GGML_TYPE_Q4_0, 64, 1024},   // small: the Q4_0 32-element block path
        {GGML_TYPE_Q4_0, 5000, 1024}, // two 16 MiB chunks
        {GGML_TYPE_Q4_K, 8, 2560},    // one chunk
        {GGML_TYPE_Q4_K, 4096, 2560}, // three chunks
        {GGML_TYPE_Q5_K, 3000, 4096}, // three chunks
        {GGML_TYPE_Q6_K, 4096, 2560}, // three chunks
        {GGML_TYPE_Q8_0, 13, 5120},   // MTP eh_proj rows
    };
    for (const auto& c : cases) {
        const std::size_t bytes = ggml_row_size(c.t, c.k) * static_cast<std::size_t>(c.n);
        std::vector<std::uint8_t> got(bytes), want(bytes);
        // The run_case seed encoding: shape only, never ncols.
        std::mt19937 a(1234u + static_cast<unsigned>(c.n + c.k));
        std::mt19937 b(1234u + static_cast<unsigned>(c.n + c.k));
        REQUIRE(d3d12_gen_weights_reference(c.t, c.n, c.k, a, want.data()));
        REQUIRE(d3d12_gen_weights(c.t, c.n, c.k, b, got.data()));
        CHECK(got == want);
        // The stream continues identically after the weights: run_case draws
        // the activations next, and both paths must hand it the same numbers.
        std::uniform_real_distribution<float> uni(-1.f, 1.f);
        for (int i = 0; i < 64; ++i)
            CHECK(uni(a) == uni(b));
    }
}

TEST_CASE("ggml_d3d12: bounded weight generation rejects bad inputs") {
    std::mt19937 rng(7);
    std::vector<std::uint8_t> buf(64 * 1024, 0xA5);
    CHECK_FALSE(d3d12_gen_weights(GGML_TYPE_Q4_K, 8, 100, rng, buf.data())); // k % 256 != 0
    CHECK_FALSE(d3d12_gen_weights(GGML_TYPE_Q4_0, 8, 48, rng, buf.data()));  // k % 32 != 0
    CHECK_FALSE(d3d12_gen_weights(GGML_TYPE_F32, 8, 1024, rng, buf.data())); // not a weight type
    CHECK_FALSE(d3d12_gen_weights(GGML_TYPE_Q4_K, 0, 2560, rng, buf.data()));
    CHECK_FALSE(d3d12_gen_weights(GGML_TYPE_Q4_K, 8, 2560, rng, nullptr));
    CHECK(std::all_of(buf.begin(), buf.end(), [](std::uint8_t v) { return v == 0xA5; }));
}

TEST_CASE("ggml_d3d12: shape-cost CSV keeps the D2a schema untouched") {
    CHECK(d3d12_block_median({}) == 0.0);
    CHECK(d3d12_block_range({}) == 0.0);
    CHECK(d3d12_block_median({2.0}) == doctest::Approx(2.0));
    CHECK(d3d12_block_median({3.0, 1.0, 2.0}) == doctest::Approx(2.0));
    CHECK(d3d12_block_median({5.0, 1.0, 4.0, 2.0, 3.0}) == doctest::Approx(3.0));
    CHECK(d3d12_block_median({4.0, 1.0, 3.0, 2.0}) == doctest::Approx(2.0)); // lower middle
    CHECK(d3d12_block_range({5.0, 1.0, 4.0, 2.0, 3.0}) == doctest::Approx(4.0));
    CHECK(std::string(d3d12_shapecost_csv_header()).find("gpu_ms_range") != std::string::npos);
    CHECK(std::string(d3d12_shapecost_csv_header()).find("wall_ms_range") != std::string::npos);
    CHECK(std::string(d3d12_shapecost_csv_header()).find(",variant,") != std::string::npos);
    CHECK(std::string(d3d12_shapecost_csv_header()).find(",pair,") != std::string::npos);
    CHECK(std::string(d3d12_shapecost_csv_header()).find(",twocol,") != std::string::npos);
    CHECK(std::string(d3d12_shapecost_csv_header()).find(",pads,") != std::string::npos);
    D3d12SelftestRow r;
    r.type = "q6_k";
    r.n = 248320;
    r.k = 2560;
    r.ncols = 5;
    r.wall_ms = 12.5;
    r.weight_mb = 521.472;
    r.peak_ws_mb = 600.0;
    r.error = "x,y";
    // The new fields ride along in the diagnostic row...
    const std::string sc = format_d3d12_shapecost_row(r, "host");
    CHECK(sc.find("248320,2560,5") != std::string::npos);
    CHECK(sc.find("521.472") != std::string::npos);
    int commas = 0, hcommas = 0;
    for (char ch : sc)
        commas += ch == ',' ? 1 : 0;
    for (const char* p = d3d12_shapecost_csv_header(); *p; ++p)
        hcommas += *p == ',' ? 1 : 0;
    CHECK(commas == hcommas);
    // ...and never leak into the gated D2a formatter.
    const std::string be = format_d3d12_selftest_row(r, "host");
    CHECK(be.find("wall_ms") == std::string::npos);
    CHECK(be.find("521.472") == std::string::npos);
    CHECK(be.find("600.0") == std::string::npos);
}

// The one place llama params get the GPU-layer request (src/bridge/llama_gpu.h).
#include "../src/bridge/llama_gpu.h"

TEST_CASE("ggml_d3d12: GPU-layer request maps to llama params") {
    llama_model_params mp = llama_model_default_params();
    CHECK(apply_gguf_gpu_layers(0, mp) == 0);
    CHECK(mp.n_gpu_layers == 0);
    CHECK_FALSE(mp.no_host); // a CPU load keeps llama's default weight bufts
    REQUIRE(mp.devices != nullptr);
    CHECK(mp.devices[0] == nullptr); // explicit empty device list: CPU only

    llama_context_params cp = llama_context_default_params();
    const bool kqv_default = cp.offload_kqv;
    apply_gguf_gpu_context(0, cp);
    CHECK(cp.offload_kqv == kqv_default);
    apply_gguf_gpu_context(28, cp);
    CHECK_FALSE(cp.offload_kqv); // KV and attention stay on the CPU

#if !defined(_WIN32)
    const char* old_fa = std::getenv("XLLAMA_FLASH_ATTN");
    const std::string saved_fa = old_fa ? old_fa : "";
    for (const char* mode : {"1", "2"}) {
        setenv("XLLAMA_FLASH_ATTN", mode, 1);
        llama_context_params forced = llama_context_default_params();
        apply_gguf_gpu_context(28, forced);
        CHECK(forced.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_ENABLED);
    }
    if (old_fa)
        setenv("XLLAMA_FLASH_ATTN", saved_fa.c_str(), 1);
    else
        unsetenv("XLLAMA_FLASH_ATTN");
    // No D3D12 on Linux: a request falls back to the CPU load unchanged.
    mp = llama_model_default_params();
    CHECK(apply_gguf_gpu_layers(99, mp) == 0);
    CHECK(mp.n_gpu_layers == 0);
    CHECK(mp.devices[0] == nullptr);
#endif
}

TEST_CASE("ggml_d3d12: z-0 weight identity is exact and narrow") {
    // Exact gate weight only.
    CHECK(d3d12_is_z0_weight("blk.0.attn_gate.weight", 2560, 4096, true));
    // Wrong name (other layers, other tensors, empty, null).
    CHECK_FALSE(d3d12_is_z0_weight("blk.1.attn_gate.weight", 2560, 4096, true));
    CHECK_FALSE(d3d12_is_z0_weight("blk.0.attn_gate.weight ", 2560, 4096, true));
    CHECK_FALSE(d3d12_is_z0_weight("", 2560, 4096, true));
    CHECK_FALSE(d3d12_is_z0_weight(nullptr, 2560, 4096, true));
    CHECK_FALSE(d3d12_is_z0_weight("blk.0.wqkv.weight", 2560, 4096, true));
    // Wrong shape or quant: never broaden.
    CHECK_FALSE(d3d12_is_z0_weight("blk.0.attn_gate.weight", 4096, 2560, true));
    CHECK_FALSE(d3d12_is_z0_weight("blk.0.attn_gate.weight", 2560, 4096, false));
}

TEST_CASE("ggml_d3d12: GPU timing state never reuses a stale sample") {
    GpuTimingState st;
    // Two valid samples and one unavailable: only valid samples accumulate.
    st.begin_call();
    st.add_sample(1.5);
    st.begin_call();
    st.add_unavailable();
    st.begin_call();
    st.add_sample(2.5);
    CHECK(st.total_ms == doctest::Approx(4.0));
    CHECK(st.valid == 2);
    CHECK(st.unavailable == 1);
    CHECK(st.last_ms == doctest::Approx(2.5));
    // The rev130 defect shape: a disabled/failed call must not re-add the
    // previous sample. ON-OFF-ON sequence leaves the total unchanged at OFF.
    st.begin_call();
    st.add_unavailable();
    CHECK(st.total_ms == doctest::Approx(4.0));
    CHECK(st.last_ms == doctest::Approx(0.0));
    st.begin_call();
    st.add_sample(0.0); // a genuinely measured zero is a valid sample
    CHECK(st.total_ms == doctest::Approx(4.0));
    CHECK(st.valid == 3);
    CHECK(st.unavailable == 2);
    st.reset();
    CHECK(st.total_ms == doctest::Approx(0.0));
    CHECK(st.valid == 0);
    CHECK(st.unavailable == 0);
}

TEST_CASE("ggml_d3d12: SWIGLU mode applies to every context profile (MTP included)") {
    CHECK(effective_swiglu_mode(0, false) == 0);
    CHECK(effective_swiglu_mode(1, false) == 1);
    CHECK(effective_swiglu_mode(2, false) == 2);
    CHECK(effective_swiglu_mode(0, true) == 0);
    CHECK(effective_swiglu_mode(1, true) == 1);
    CHECK(effective_swiglu_mode(2, true) == 2);
}

TEST_CASE("ggml_d3d12: SWIGLU profile binds once and rejects conflicting contexts") {
    // Knob OFF: every context is acceptable and the capability stays off.
    SwigluModePolicy off;
    CHECK(off.accepts(0, false));
    CHECK(off.accepts(0, true));
    off.finalize(0, false);
    CHECK(off.mode == 0);

    // Seq first with the knob ON: bound seq/on; the mode is the same for
    // every profile, so MTP contexts are accepted too, and finalize on an
    // already-bound policy never changes the mode.
    SwigluModePolicy seq;
    CHECK(seq.accepts(1, false));
    CHECK(seq.accepts(1, true)); // not bound yet: first context may choose
    seq.finalize(1, false);
    CHECK(seq.mode == 1);
    CHECK(seq.bound);
    CHECK(seq.accepts(1, false));
    CHECK(seq.accepts(1, true)); // same effective mode: no conflict
    CHECK(seq.accepts(0, true)); // accepts() never mutates the bound mode
    seq.finalize(1, true);       // bound: no-op, must not flip the mode
    CHECK(seq.mode == 1);
    CHECK(seq.mtp_capable == false);

    // MTP first with the knob ON: bound MTP/on, mode now applies to MTP too.
    SwigluModePolicy mtp;
    CHECK(mtp.accepts(2, true));
    mtp.finalize(2, true);
    CHECK(mtp.mode == 2);
    CHECK(mtp.mtp_capable);
    CHECK(mtp.accepts(2, false));
    CHECK(mtp.accepts(2, true));
}
