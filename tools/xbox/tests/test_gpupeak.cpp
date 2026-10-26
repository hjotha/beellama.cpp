// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Host tests for the GPU ceiling probe's pure logic. The D3D12 driver itself
// only compiles on Windows; these tests pin the option contract, the dispatch
// planning, the analytic pattern XOR that validates the stream kernels, and the
// JSON schema the console run hands back.

#include <doctest/doctest.h>

#include "xllama/gpupeak.h"

#include <algorithm>
#include <string>
#include <vector>

using namespace xllama;

namespace {

// Straightforward O(n) transliteration of the shader's grid-stride mapping,
// used only to prove the period-folding optimisation in gpupeak_pattern_xor.
std::uint32_t naive_pattern_xor(std::uint64_t n_vec, std::uint32_t gx, std::uint32_t gy,
                                std::uint32_t mode) {
    const std::uint64_t stride = static_cast<std::uint64_t>(gx) * gy * 256u;
    std::uint32_t out = 0;
    for (std::uint64_t i = 0; i < n_vec; ++i) {
        const std::uint64_t flat = i % stride;
        const std::uint64_t dtid_x = flat % (static_cast<std::uint64_t>(gx) * 256u);
        const std::uint32_t gid_x = static_cast<std::uint32_t>(dtid_x / 256u);
        out ^= static_cast<std::uint32_t>(dtid_x + 1u);
        out ^= gid_x + 1u;
        out ^= static_cast<std::uint32_t>(n_vec);
        out ^= mode;
    }
    return out;
}

} // namespace

TEST_CASE("gpupeak: default options match the documented plan") {
    const GpupeakOptions o;
    CHECK(o.stream_bytes == kGpupeakDefaultStreamBytes);
    CHECK(o.iterations == kGpupeakDefaultIterations);
    CHECK_FALSE(o.run_big_stream);
    CHECK(o.run_stream);
    CHECK(o.run_alu);
    CHECK(o.run_latency);
    CHECK(o.run_occupancy);
}

TEST_CASE("gpupeak: options parser accepts overrides and ignores junk") {
    const std::string text =
        "# console plan\n"
        "stream_bytes=536870912\n"
        "\n"
        "iterations = 7\n"
        "run_big_stream=on\n"
        "big_stream_bytes=1073741824\n"
        "run_alu=off\n"
        "unknown_key=42\n"
        "iterations=999\n"     // out of range: ignored
        "stream_bytes=12\n";   // too small: ignored
    const GpupeakOptions o = gpupeak_parse_options(text);
    CHECK(o.stream_bytes == 536870912u);
    CHECK(o.iterations == 7);
    CHECK(o.run_big_stream);
    CHECK(o.big_stream_bytes == 1073741824u);
    CHECK_FALSE(o.run_alu);
    CHECK(o.run_latency);
}

TEST_CASE("gpupeak: options summary round-trips the fields the JSON carries") {
    GpupeakOptions o;
    o.run_big_stream = true;
    o.iterations = 3;
    const std::string s = gpupeak_options_summary(o);
    CHECK(s.find("stream_bytes=" + std::to_string(o.stream_bytes)) != std::string::npos);
    CHECK(s.find("big_stream=1") != std::string::npos);
    CHECK(s.find("iterations=3") != std::string::npos);
}

TEST_CASE("gpupeak: grid planning never exceeds the D3D12 per-axis limit") {
    for (std::uint32_t groups : {1u, 2u, 2048u, 65535u, 65536u, 200000u}) {
        const GpupeakGrid grid = gpupeak_plan_grid(groups);
        CHECK(grid.groups_x >= 1u);
        CHECK(grid.groups_y >= 1u);
        CHECK(grid.groups_x <= kGpupeakMaxGroupsPerDim);
        CHECK(grid.groups_y <= kGpupeakMaxGroupsPerDim);
        const std::uint64_t covered =
            static_cast<std::uint64_t>(grid.groups_x) * grid.groups_y;
        CHECK(covered >= groups);
    }
}

TEST_CASE("gpupeak: analytic pattern XOR matches the naive grid-stride walk") {
    // Small shapes exercise the period folding and the tail path; the larger
    // one matches the real stream plan (2048 groups).
    struct Shape {
        std::uint64_t n_vec;
        std::uint32_t gx;
        std::uint32_t gy;
    } shapes[] = {{0, 1, 1},       {1, 1, 1},        {255, 1, 1},     {256, 1, 1},
                  {257, 1, 1},     {1000, 2, 1},     {512, 2, 1},     {4097, 3, 1},
                  {2048, 1, 2},    {16384, 4, 2},    {65536, 2048, 1}};
    for (const Shape& shape : shapes) {
        const std::uint32_t got = gpupeak_pattern_xor(shape.n_vec, shape.gx, shape.gy, 0u);
        const std::uint32_t want = naive_pattern_xor(shape.n_vec, shape.gx, shape.gy, 0u);
        CHECK(got == want);
    }
    // mode is part of the stored pattern word; an odd vector count keeps it
    // from cancelling under the XOR fold.
    CHECK(gpupeak_pattern_xor(4097, 2, 1, 1u) != gpupeak_pattern_xor(4097, 2, 1, 0u));
    CHECK(gpupeak_pattern_xor(4096, 2, 1, 1u) == gpupeak_pattern_xor(4096, 2, 1, 0u));
}

TEST_CASE("gpupeak: FLOP accounting matches the shader sources") {
    const std::uint32_t groups = 52;
    const std::uint64_t iters = 1024;
    const std::uint64_t threads = static_cast<std::uint64_t>(groups) * 64u;
    CHECK(gpupeak_alu_flops("fma_f32", groups, iters) == threads * iters * 8u * 2u);
    CHECK(gpupeak_alu_flops("fma_f16", groups, iters) == threads * iters * 4u * 4u);
    CHECK(gpupeak_alu_flops("fma_f16_min16", groups, iters) == threads * iters * 4u * 4u);
    CHECK(gpupeak_alu_flops("dot2_f16", groups, iters) == threads * iters * 4u * 4u);
    CHECK(gpupeak_alu_flops("dot4_i8", groups, iters) == threads * iters * 4u * 8u);
    CHECK(gpupeak_alu_flops("unknown", groups, iters) == 0u);
}

TEST_CASE("gpupeak: metric helpers refuse to divide by zero") {
    CHECK(gpupeak_gbs(1e9, 0.0) == 0.0);
    CHECK(gpupeak_gbs(1e9, 0.5) == doctest::Approx(2.0));
    CHECK(gpupeak_tflops(1e12, 0.5) == doctest::Approx(2.0));
    CHECK(gpupeak_gops(1e9, 0.5) == doctest::Approx(2.0));
    CHECK(gpupeak_fraction(6.0, 12.0) == doctest::Approx(50.0));
    CHECK(gpupeak_fraction(6.0, 0.0) == 0.0);
}

TEST_CASE("gpupeak: median and percentile use nearest-rank") {
    CHECK(gpupeak_median({}) == 0.0);
    CHECK(gpupeak_median({3.0}) == doctest::Approx(3.0));
    CHECK(gpupeak_median({4.0, 1.0, 3.0, 2.0}) == doctest::Approx(2.5));
    CHECK(gpupeak_median({5.0, 1.0, 3.0}) == doctest::Approx(3.0));
    std::vector<double> v;
    for (int i = 1; i <= 100; ++i)
        v.push_back(static_cast<double>(i));
    CHECK(gpupeak_percentile(v, 0.50) == doctest::Approx(50.0));
    CHECK(gpupeak_percentile(v, 0.95) == doctest::Approx(95.0));
    CHECK(gpupeak_percentile(v, 0.0) == doctest::Approx(1.0));
    CHECK(gpupeak_percentile(v, 1.0) == doctest::Approx(100.0));
}

TEST_CASE("gpupeak: stream summary separates checksum evidence from rates") {
    std::vector<GpupeakRow> rows;
    GpupeakRow validate;
    validate.probe = "read";
    validate.bytes = 1024;
    validate.checksum_checked = true;
    validate.checksum_ok = true;
    rows.push_back(validate);
    for (double sec : {0.010, 0.008, 0.009}) {
        GpupeakRow row;
        row.probe = "read";
        row.bytes = 1024;
        row.gpu_seconds = sec;
        row.gpu_valid = true;
        rows.push_back(row);
    }
    GpupeakRow other;
    other.probe = "write";
    other.bytes = 1024;
    other.gpu_seconds = 0.001;
    other.gpu_valid = true;
    rows.push_back(other);

    const GpupeakStreamSummary summary = gpupeak_summarize_stream(rows, "read", 1024);
    CHECK(summary.runs == 3u);
    CHECK(summary.checksum_ok);
    CHECK(summary.best_gbs == doctest::Approx(gpupeak_gbs(1024, 0.008)));
    CHECK(summary.median_gbs == doctest::Approx(gpupeak_gbs(1024, 0.009)));

    // A failed validation must not be hidden by the rows that only carry rates.
    rows[0].checksum_ok = false;
    CHECK_FALSE(gpupeak_summarize_stream(rows, "read", 1024).checksum_ok);
}

TEST_CASE("gpupeak: JSON carries schema, nominal table and every row") {
    GpupeakOptions options;
    GpupeakResult result;
    result.d3d12_ran = true;
    result.error.clear();
    result.device.adapter = "Xbox Series X";
    result.device.timestamp_frequency = 10000000;
    result.device.timestamp_us_per_tick = 1e6 / 1e7;
    result.device.local_budget = 4348654388ull;
    result.device.wave_lane_min = 64;
    result.device.wave_lane_max = 64;
    result.pso_errors.push_back("dot4_i8: CreateComputePipelineState hr=0x80070057");

    GpupeakRow row;
    row.suite = "stream";
    row.probe = "read";
    row.variant = "268435456";
    row.bytes = 268435456;
    row.gpu_seconds = 0.0009;
    row.gpu_seconds_median = 0.00095;
    row.groups = 2048;
    row.threads = 256;
    row.runs = 5;
    row.gpu_valid = true;
    row.checksum_checked = true;
    row.checksum_ok = true;
    result.rows.push_back(row);

    GpupeakRow latency;
    latency.suite = "latency";
    latency.probe = "empty_dispatch_gpu";
    latency.variant = "p50";
    latency.gpu_seconds = 0.000004;
    latency.cpu_seconds = 0.00009;
    latency.runs = 60;
    latency.gpu_valid = true;
    result.rows.push_back(latency);

    const std::string json = gpupeak_json(result, options, "unittest");
    CHECK(json.find("\"schema\":\"xllama-gpu-ceiling-v1\"") != std::string::npos);
    CHECK(json.find("\"d3d12_ran\":true") != std::string::npos);
    CHECK(json.find("\"fp32_tflops\":12.150") != std::string::npos);
    CHECK(json.find("\"local_budget_bytes\":4348654388") != std::string::npos);
    CHECK(json.find("\"rows\":[") != std::string::npos);
    CHECK(json.find("\"probe\":\"read\"") != std::string::npos);
    CHECK(json.find("\"probe\":\"empty_dispatch_gpu\"") != std::string::npos);
    CHECK(json.find("hr=0x80070057") != std::string::npos);
    CHECK(json.find("\"checksum_ok\":true") != std::string::npos);
    // No clock is inferred from the dependent FMA chain anymore.
    CHECK(json.find("implied_clock") == std::string::npos);
    CHECK(json.find("\"derived\"") == std::string::npos);
    // Two rows + one closing bracket, and the document is balanced.
    std::size_t commas = 0;
    for (char c : json)
        commas += c == ',' ? 1 : 0;
    CHECK(commas > 10u);
    CHECK(json.find("\"recorded_utc\"") != std::string::npos);
}

TEST_CASE("gpupeak: non-D3D12 host reports unavailable without rows") {
    const GpupeakResult r = measure_gpupeak(GpupeakOptions{});
    CHECK_FALSE(r.d3d12_ran);
    CHECK(r.rows.empty());
    CHECK_FALSE(r.error.empty());
}
