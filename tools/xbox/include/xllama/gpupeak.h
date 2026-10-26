// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// GPU ceiling probe: what the Series X actually delivers inside the UWP
// sandbox, measured with the same system-D3D12 path as the RKVA engine.
//
// Four sub-suites, all GPU-timestamped:
//   stream   read / write / copy over large buffers (default 256 MiB, optional
//            1 GiB) with a host-verified pattern checksum.
//   alu      FP32 FMA, FP16 packed FMA, dot2 F16->F32, dot4 I8->I32 over a
//            dispatch-size sweep, so the peak and the occupancy knee are both
//            visible.
//   latency  near-empty dispatch: GPU time per dispatch, CPU submit+fence
//            wall, and readback Map cost. This is the fixed cost of each of
//            the 16 remote calls a decoded token makes.
//   occupancy LDS-limited group sweeps (32 KiB and 16 KiB per group) whose
//            step positions give two independent estimates of the CU count.
//
// Pure helpers (options parsing, dispatch planning, metric math, JSON
// fragment formatting) compile and unit-test on Linux; measure_gpupeak()
// returns d3d12_ran=false with an explicit error elsewhere. No number is ever
// invented: every reported rate comes from a timestamp pair around the work
// that produced it.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace xllama {

// ---------------------------------------------------------------------------
// Published Series X nominal figures. These are NOT measurements and are used
// only as the denominator of an "achieved fraction" in the report. Sources:
// Microsoft's Series X specification (52 CUs at 1.825 GHz, 12.15 TFLOPS FP32,
// 24.30 TFLOPS FP16, 48.6 TOPS INT8) and the GDDR6 memory split (10 GB at
// 560 GB/s on a 320-bit bus, 6 GB at 336 GB/s on a 192-bit bus).
// ---------------------------------------------------------------------------
inline constexpr double kGpupeakNominalFp32Tflops = 12.15;
inline constexpr double kGpupeakNominalFp16Tflops = 24.30;
inline constexpr double kGpupeakNominalInt8Tops = 48.6;
inline constexpr double kGpupeakNominalFastGbs = 560.0;
inline constexpr double kGpupeakNominalSlowGbs = 336.0;
inline constexpr uint32_t kGpupeakNominalDualCu = 52;

// Default plan. The 1 GiB pass is opt-in through gpupeak-opts.txt because two
// 1 GiB committed buffers plus the host pattern vector are close to the
// reported 4.35 GB GPU budget once the app's own allocations exist.
inline constexpr std::uint64_t kGpupeakDefaultStreamBytes = std::uint64_t{256} << 20;
inline constexpr std::uint64_t kGpupeakDefaultBigStreamBytes = std::uint64_t{1024} << 20;
inline constexpr int kGpupeakDefaultIterations = 5;
inline constexpr std::uint32_t kGpupeakMaxGroupsPerDim = 65535u;

// D3D12 caps one workgroup at 32 KiB of LDS, so these are the two occupancy
// probes: 32 KiB admits 2 groups per 64 KiB CU LDS, 16 KiB admits 4.
inline constexpr std::uint32_t kGpupeakCuProbeGroupsCeiling = 8192u;

struct GpupeakOptions {
    std::uint64_t stream_bytes = kGpupeakDefaultStreamBytes;
    std::uint64_t big_stream_bytes = kGpupeakDefaultBigStreamBytes;
    bool run_big_stream = false;
    int iterations = kGpupeakDefaultIterations;
    bool run_stream = true;
    bool run_alu = true;
    bool run_latency = true;
    bool run_occupancy = true;
};

// key=value lines, '#' comments, unknown keys ignored. Keeps the flag-file
// contract free of a JSON dependency on the UWP side.
GpupeakOptions gpupeak_parse_options(const std::string& text);

// Telegram-style option echo for the JSON/CLI ("stream_bytes=268435456 ...").
std::string gpupeak_options_summary(const GpupeakOptions& o);

// Dispatch for one group axis pair so x and y stay inside the D3D12 limit.
struct GpupeakGrid {
    std::uint32_t groups_x = 1;
    std::uint32_t groups_y = 1;
};

GpupeakGrid gpupeak_plan_grid(std::uint32_t total_groups);

// Analytic XOR of the whole pattern the write kernel stores, replicating the
// shader's grid-stride mapping exactly:
//   flat = i % (gx*gy*256); dtid.x = flat % (gx*256); gid.x = dtid.x / 256
//   words = (dtid.x + 1, gid.x + 1, n_vec, mode)
// The read validation pass folds one accumulator per flat thread and must
// equal this value, which proves every 16-byte vector was actually loaded.
std::uint32_t gpupeak_pattern_xor(std::uint64_t n_vec, std::uint32_t groups_x,
                                  std::uint32_t groups_y, std::uint32_t mode);

// Metrics. bytes is the traffic the kernel actually moved (read+write for a
// copy), not the buffer size; flops follows the per-probe accounting below.
inline double gpupeak_gbs(double bytes, double seconds) {
    return seconds > 0.0 ? bytes / 1e9 / seconds : 0.0;
}

inline double gpupeak_tflops(double flops, double seconds) {
    return seconds > 0.0 ? flops / 1e12 / seconds : 0.0;
}

inline double gpupeak_gops(double ops, double seconds) {
    return seconds > 0.0 ? ops / 1e9 / seconds : 0.0;
}

inline double gpupeak_fraction(double achieved, double nominal) {
    return nominal > 0.0 ? 100.0 * achieved / nominal : 0.0;
}

// FLOP accounting per shader instruction, one place so the C++ probe and the
// Python post-processing cannot disagree:
//   F32 FMA     1 instruction = 1 multiply + 1 add = 2 FLOP
//   packed F16  1 half2 mad   = 2 FMA               = 4 FLOP
//   dot2 F16    1 dot2add    = 2 MAC               = 4 FLOP
//   dot4 I8     1 dot4add    = 4 MAC               = 8 int ops
inline constexpr double kGpupeakFlopPerFmaF32 = 2.0;
inline constexpr double kGpupeakFlopPerFmaF16x2 = 4.0;
inline constexpr double kGpupeakFlopPerDot2F16 = 4.0;
inline constexpr double kGpupeakOpsPerDot4I8 = 8.0;

// FLOP counts actually issued by the probe kernels, from the source:
//   fma_f32   8 chains x iters                -> 8*iters FMA per thread
//   fma_f16   4 chains x iters                -> 4*iters packed FMA per thread
//   dot2_f16  4 chains x iters                -> 4*iters dot2add per thread
//   dot4_i8   4 chains x iters                -> 4*iters dot4add per thread
inline std::uint64_t gpupeak_alu_flops(const char* probe, std::uint32_t groups,
                                       std::uint64_t iters) {
    const std::uint64_t threads = std::uint64_t{groups} * 64u;
    const std::string name = probe ? probe : "";
    if (name == "fma_f32")
        return threads * iters * 8u * static_cast<std::uint64_t>(kGpupeakFlopPerFmaF32);
    if (name == "fma_f16" || name == "fma_f16_min16")
        return threads * iters * 4u * static_cast<std::uint64_t>(kGpupeakFlopPerFmaF16x2);
    if (name == "dot2_f16")
        return threads * iters * 4u * static_cast<std::uint64_t>(kGpupeakFlopPerDot2F16);
    if (name == "dot4_i8")
        return threads * iters * 4u * static_cast<std::uint64_t>(kGpupeakOpsPerDot4I8);
    return 0u;
}

// One probe row. Every timed row keeps its own GPU and CPU seconds so the
// report can separate device time from submit/fence overhead.
struct GpupeakRow {
    std::string suite;   // stream | alu | latency | occupancy
    std::string probe;   // read | write | copy | fma_f32 | ... | empty | cu32
    std::string variant; // buffer size, group count, chain length, statistic
    std::uint64_t bytes = 0;   // traffic attributed to this row
    std::uint64_t flops = 0;   // FLOP/OPS attributed to this row
    double gpu_seconds = 0.0;  // best timestamp pair of the timed runs
    double gpu_seconds_median = 0.0;
    double cpu_seconds = 0.0;  // best submit + fence wall
    double readback_seconds = 0.0;
    std::uint32_t groups = 0;
    std::uint32_t threads = 0;
    std::uint32_t runs = 0;
    std::uint64_t iters = 0;
    std::uint32_t checksum = 0;          // observed fold (validation rows only)
    std::uint32_t expected_checksum = 0; // analytic fold (validation rows only)
    bool gpu_valid = false;
    bool checksum_ok = false;
    bool checksum_checked = false;
    std::string error;
};

// Best-of and median helpers used by the latency suite (pure, host-tested).
double gpupeak_median(std::vector<double> values);
double gpupeak_percentile(std::vector<double> values, double fraction);

struct GpupeakStreamSummary {
    std::uint64_t bytes = 0;       // buffer size
    std::uint64_t traffic = 0;     // read+write actually issued
    double best_gbs = 0.0;
    double median_gbs = 0.0;
    std::uint32_t runs = 0;
    bool checksum_ok = false;
};

// Aggregate the per-run rows of one stream direction with the same accounting
// the JSON report uses. traffic_per_run lets copy count read+write while read
// and write count one direction.
GpupeakStreamSummary gpupeak_summarize_stream(const std::vector<GpupeakRow>& rows, const char* probe,
                                              std::uint64_t traffic_per_run);

struct GpupeakDeviceInfo {
    std::string adapter;
    std::uint32_t vendor_id = 0;
    std::uint32_t device_id = 0;
    std::uint64_t dedicated_vram = 0;
    std::uint64_t shared_system = 0;
    std::uint64_t local_budget = 0;  // DXGI_QUERY_VIDEO_MEMORY_INFO budget
    std::uint64_t local_usage = 0;
    std::uint64_t local_available = 0;
    std::uint32_t max_lds_bytes = 0;
    std::uint32_t wave_lane_min = 0;
    std::uint32_t wave_lane_max = 0;
    std::uint64_t timestamp_frequency = 0;
    double timestamp_us_per_tick = 0.0;
    std::string feature_error;
};

struct GpupeakResult {
    GpupeakDeviceInfo device;
    std::vector<GpupeakRow> rows;
    std::vector<std::string> pso_errors; // "probe: hr=0x..." — absent PSOs, no fake rows
    bool d3d12_ran = false;
    std::string error;
};

// UWP/Windows: D3D12 system device, all sub-suites; the options come from
// LocalState\gpupeak-opts.txt. Elsewhere: d3d12_ran=false + error, no throw.
GpupeakResult measure_gpupeak(const GpupeakOptions& options);

// JSON document for LocalState\gpupeak-result.json. Pure: takes the result the
// driver produced and formats it, so the schema is host-testable.
std::string gpupeak_json(const GpupeakResult& result, const GpupeakOptions& options,
                         const std::string& host_label);

} // namespace xllama
