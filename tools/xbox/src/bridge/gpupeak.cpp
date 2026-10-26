// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// GPU ceiling probe driver. See include/xllama/gpupeak.h for the contract.
//
// Timing discipline: every rate comes from a GPU timestamp pair recorded in
// the same command list as the work, resolved only after the timed dispatch
// has completed. CPU spans (submit+fence, readback Map) are recorded
// separately so the report never mixes device time with host overhead.

#include "xllama/gpupeak.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cwchar>

#if defined(_WIN32)
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <Windows.h>
    #include <d3d12.h>
    #include <dxgi1_4.h>
    #include <wrl/client.h>

    #include "gpupeak_cu_probe_16_dxil.h"
    #include "gpupeak_cu_probe_32_dxil.h"
    #include "gpupeak_dot2_f16_dxil.h"
    #include "gpupeak_dot4_i8_dxil.h"
    #include "gpupeak_empty_dxil.h"
    #include "gpupeak_fma_f16_dxil.h"
    #include "gpupeak_fma_f32_dxil.h"
    #include "gpupeak_latency_fma_dxil.h"
    #include "gpupeak_stream_copy_dxil.h"
    #include "gpupeak_stream_read_dxil.h"
    #include "gpupeak_stream_write_dxil.h"
    #include "xllama/d3d12_compute.h"
    #include "xllama/d3d12_dyn.h"

using Microsoft::WRL::ComPtr;
#endif

namespace xllama {

// ---------------------------------------------------------------------------
// Pure helpers (host-testable on Linux)
// ---------------------------------------------------------------------------

namespace {

std::string trim_copy(const std::string& in) {
    std::size_t begin = 0;
    while (begin < in.size() && (in[begin] == ' ' || in[begin] == '\t' || in[begin] == '\r'))
        ++begin;
    std::size_t end = in.size();
    while (end > begin && (in[end - 1] == ' ' || in[end - 1] == '\t' || in[end - 1] == '\r'))
        --end;
    return in.substr(begin, end - begin);
}

bool parse_u64(const std::string& text, std::uint64_t* out) {
    if (!out || text.empty())
        return false;
    std::uint64_t value = 0;
    for (char c : text) {
        if (c < '0' || c > '9')
            return false;
        const std::uint64_t digit = static_cast<std::uint64_t>(c - '0');
        if (value > (UINT64_MAX - digit) / 10)
            return false;
        value = value * 10 + digit;
    }
    *out = value;
    return true;
}

bool parse_bool(const std::string& text, bool* out) {
    if (!out)
        return false;
    if (text == "1" || text == "on" || text == "true" || text == "yes") {
        *out = true;
        return true;
    }
    if (text == "0" || text == "off" || text == "false" || text == "no") {
        *out = false;
        return true;
    }
    return false;
}

} // namespace

GpupeakOptions gpupeak_parse_options(const std::string& text) {
    GpupeakOptions o;
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t end = text.find('\n', start);
        const std::string line =
            trim_copy(text.substr(start, end == std::string::npos ? std::string::npos : end - start));
        if (!line.empty() && line[0] != '#') {
            const std::size_t eq = line.find('=');
            if (eq != std::string::npos) {
                const std::string key = trim_copy(line.substr(0, eq));
                const std::string value = trim_copy(line.substr(eq + 1));
                std::uint64_t number = 0;
                if (key == "stream_bytes" && parse_u64(value, &number) && number >= (1u << 16))
                    o.stream_bytes = number;
                else if (key == "big_stream_bytes" && parse_u64(value, &number) &&
                         number >= (1u << 16))
                    o.big_stream_bytes = number;
                else if (key == "iterations" && parse_u64(value, &number) && number >= 1 &&
                         number <= 64)
                    o.iterations = static_cast<int>(number);
                else if (key == "run_big_stream")
                    parse_bool(value, &o.run_big_stream);
                else if (key == "run_stream")
                    parse_bool(value, &o.run_stream);
                else if (key == "run_alu")
                    parse_bool(value, &o.run_alu);
                else if (key == "run_latency")
                    parse_bool(value, &o.run_latency);
                else if (key == "run_occupancy")
                    parse_bool(value, &o.run_occupancy);
            }
        }
        if (end == std::string::npos)
            break;
        start = end + 1;
    }
    if (!o.run_big_stream || o.big_stream_bytes < o.stream_bytes)
        o.big_stream_bytes = std::max(o.big_stream_bytes, o.stream_bytes);
    return o;
}

std::string gpupeak_options_summary(const GpupeakOptions& o) {
    char buf[320];
    std::snprintf(buf, sizeof(buf),
                  "stream_bytes=%llu big_stream=%d big_stream_bytes=%llu iterations=%d "
                  "stream=%d alu=%d latency=%d occupancy=%d",
                  static_cast<unsigned long long>(o.stream_bytes), o.run_big_stream ? 1 : 0,
                  static_cast<unsigned long long>(o.big_stream_bytes), o.iterations,
                  o.run_stream ? 1 : 0, o.run_alu ? 1 : 0, o.run_latency ? 1 : 0,
                  o.run_occupancy ? 1 : 0);
    return buf;
}

GpupeakGrid gpupeak_plan_grid(std::uint32_t total_groups) {
    GpupeakGrid grid;
    if (total_groups == 0)
        return grid;
    if (total_groups <= kGpupeakMaxGroupsPerDim) {
        grid.groups_x = total_groups;
        return grid;
    }
    grid.groups_x = kGpupeakMaxGroupsPerDim;
    const std::uint32_t y = (total_groups + kGpupeakMaxGroupsPerDim - 1) / kGpupeakMaxGroupsPerDim;
    grid.groups_y = std::min(y, kGpupeakMaxGroupsPerDim);
    return grid;
}

std::uint32_t gpupeak_pattern_xor(std::uint64_t n_vec, std::uint32_t groups_x,
                                  std::uint32_t groups_y, std::uint32_t mode) {
    if (n_vec == 0 || groups_x == 0 || groups_y == 0)
        return 0;
    const std::uint64_t threads = static_cast<std::uint64_t>(groups_x) * groups_y * 256u;
    const std::uint64_t period = std::min<std::uint64_t>(threads, n_vec);
    // XOR over one full grid period, then fold the leftover tail. A value
    // repeated an even number of times cancels under XOR, so whole periods
    // only matter when their count is odd.
    std::uint32_t one_period = 0;
    for (std::uint64_t i = 0; i < period; ++i) {
        const std::uint64_t dtid_x = i % (static_cast<std::uint64_t>(groups_x) * 256u);
        const std::uint32_t gid_x = static_cast<std::uint32_t>(dtid_x / 256u);
        one_period ^= static_cast<std::uint32_t>(dtid_x + 1u);
        one_period ^= gid_x + 1u;
        one_period ^= static_cast<std::uint32_t>(n_vec);
        one_period ^= mode;
    }
    std::uint32_t out = 0;
    if (threads > 0) {
        const std::uint64_t periods = n_vec / threads;
        if ((periods & 1u) != 0)
            out ^= one_period;
    }
    const std::uint64_t rem_begin = (n_vec / threads) * threads;
    for (std::uint64_t i = rem_begin; i < n_vec; ++i) {
        const std::uint64_t flat = i - rem_begin;
        const std::uint64_t dtid_x = flat % (static_cast<std::uint64_t>(groups_x) * 256u);
        const std::uint32_t gid_x = static_cast<std::uint32_t>(dtid_x / 256u);
        out ^= static_cast<std::uint32_t>(dtid_x + 1u);
        out ^= gid_x + 1u;
        out ^= static_cast<std::uint32_t>(n_vec);
        out ^= mode;
    }
    return out;
}

double gpupeak_median(std::vector<double> values) {
    if (values.empty())
        return 0.0;
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    return (n % 2u) ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

double gpupeak_percentile(std::vector<double> values, double fraction) {
    if (values.empty())
        return 0.0;
    fraction = std::min(1.0, std::max(0.0, fraction));
    std::sort(values.begin(), values.end());
    std::size_t rank = static_cast<std::size_t>(std::ceil(fraction * values.size()));
    if (rank == 0)
        rank = 1;
    if (rank > values.size())
        rank = values.size();
    return values[rank - 1];
}

GpupeakStreamSummary gpupeak_summarize_stream(const std::vector<GpupeakRow>& rows, const char* probe,
                                              std::uint64_t traffic_per_run) {
    GpupeakStreamSummary summary;
    summary.traffic = traffic_per_run;
    std::vector<double> rates;
    for (const GpupeakRow& row : rows) {
        if (row.probe != (probe ? probe : ""))
            continue;
        if (row.checksum_checked && !summary.checksum_ok) {
            summary.checksum_ok = row.checksum_ok;
            summary.bytes = row.bytes;
        }
        if (row.gpu_valid && row.gpu_seconds > 0.0 && row.bytes > 0) {
            rates.push_back(gpupeak_gbs(static_cast<double>(traffic_per_run), row.gpu_seconds));
            summary.bytes = row.bytes;
        }
    }
    if (!rates.empty()) {
        summary.best_gbs = *std::max_element(rates.begin(), rates.end());
        summary.median_gbs = gpupeak_median(rates);
        summary.runs = static_cast<std::uint32_t>(rates.size());
    }
    return summary;
}

namespace {

std::string json_escape(const std::string& in) {
    std::string out;
    out.reserve(in.size() + 8);
    for (char c : in) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                out += buf;
            } else {
                out += c;
            }
        }
    }
    return out;
}

void append_utc_now(std::string* out) {
    char date_buf[32] = "unknown";
    const std::time_t now = std::time(nullptr);
    if (const std::tm* tm_utc = std::gmtime(&now))
        std::strftime(date_buf, sizeof(date_buf), "%Y-%m-%dT%H:%M:%SZ", tm_utc);
    *out += date_buf;
}

} // namespace

std::string gpupeak_json(const GpupeakResult& result, const GpupeakOptions& options,
                         const std::string& host_label) {
    std::string o;
    o.reserve(16 * 1024);
    char buf[512];

    o += "{\"schema\":\"xllama-gpu-ceiling-v1\",\"host\":\"";
    o += json_escape(host_label);
    o += "\",\"recorded_utc\":\"";
    append_utc_now(&o);
    o += "\",\"d3d12_ran\":";
    o += result.d3d12_ran ? "true" : "false";
    o += ",\"error\":\"";
    o += json_escape(result.error);
    o += "\",\"options\":{\"summary\":\"";
    o += json_escape(gpupeak_options_summary(options));
    std::snprintf(buf, sizeof(buf),
                  "\",\"stream_bytes\":%llu,\"run_big_stream\":%s,\"big_stream_bytes\":%llu,"
                  "\"iterations\":%d},\"nominal\":{\"fp32_tflops\":%.3f,\"fp16_tflops\":%.3f,"
                  "\"int8_tops\":%.3f,\"fast_gbs\":%.1f,\"slow_gbs\":%.1f,\"dual_cu\":%u,"
                  "\"note\":\"published Series X specification; not measured\"}",
                  static_cast<unsigned long long>(options.stream_bytes),
                  options.run_big_stream ? "true" : "false",
                  static_cast<unsigned long long>(options.big_stream_bytes), options.iterations,
                  kGpupeakNominalFp32Tflops, kGpupeakNominalFp16Tflops, kGpupeakNominalInt8Tops,
                  kGpupeakNominalFastGbs, kGpupeakNominalSlowGbs, kGpupeakNominalDualCu);
    o += buf;

    const GpupeakDeviceInfo& d = result.device;
    o += ",\"device\":{\"adapter\":\"";
    o += json_escape(d.adapter);
    std::snprintf(buf, sizeof(buf),
                  "\",\"vendor_id\":%u,\"device_id\":%u,\"dedicated_vram_bytes\":%llu,"
                  "\"shared_system_bytes\":%llu,\"local_budget_bytes\":%llu,"
                  "\"local_usage_bytes\":%llu,\"local_available_bytes\":%llu,"
                  "\"max_lds_bytes\":%u,\"wave_lane_min\":%u,\"wave_lane_max\":%u,"
                  "\"timestamp_frequency\":%llu,\"timestamp_us_per_tick\":%.9f,"
                  "\"feature_error\":\"%s\"}",
                  d.vendor_id, d.device_id,
                  static_cast<unsigned long long>(d.dedicated_vram),
                  static_cast<unsigned long long>(d.shared_system),
                  static_cast<unsigned long long>(d.local_budget),
                  static_cast<unsigned long long>(d.local_usage),
                  static_cast<unsigned long long>(d.local_available), d.max_lds_bytes,
                  d.wave_lane_min, d.wave_lane_max,
                  static_cast<unsigned long long>(d.timestamp_frequency),
                  d.timestamp_us_per_tick, json_escape(d.feature_error).c_str());
    o += buf;

    o += ",\"derived\":{\"implied_clock_mhz\":";
    std::snprintf(buf, sizeof(buf), "%.1f,\"implied_clock_seconds_per_fma\":%.12f,"
                                    "\"implied_clock_note\":\"derived from a dependent FP32 FMA "
                                    "chain assuming the documented RDNA2 4-clock VALU latency; "
                                    "not a measured clock\","
                                    "\"catalog_frequency_mhz\":%u}",
                  result.implied_clock_mhz, result.implied_clock_seconds_per_fma,
                  result.catalog_frequency_mhz);
    o += buf;

    o += ",\"pso_errors\":[";
    for (std::size_t i = 0; i < result.pso_errors.size(); ++i) {
        if (i)
            o += ',';
        o += '"';
        o += json_escape(result.pso_errors[i]);
        o += '"';
    }
    o += "],\"rows\":[";
    for (std::size_t i = 0; i < result.rows.size(); ++i) {
        const GpupeakRow& r = result.rows[i];
        if (i)
            o += ',';
        const double gbs = gpupeak_gbs(static_cast<double>(r.bytes), r.gpu_seconds);
        const double tflops = gpupeak_tflops(static_cast<double>(r.flops), r.gpu_seconds);
        const double gops = gpupeak_gops(static_cast<double>(r.flops), r.gpu_seconds);
        std::snprintf(buf, sizeof(buf),
                      "{\"suite\":\"%s\",\"probe\":\"%s\",\"variant\":\"%s\",\"bytes\":%llu,"
                      "\"flops\":%llu,\"gpu_seconds\":%.9f,\"gpu_seconds_median\":%.9f,"
                      "\"cpu_seconds\":%.9f,\"readback_seconds\":%.9f,\"gbs\":%.3f,"
                      "\"tflops\":%.4f,\"gops\":%.4f,\"groups\":%u,\"threads\":%u,\"runs\":%u,"
                      "\"iters\":%llu,\"gpu_valid\":%s,\"checksum_checked\":%s,"
                      "\"checksum_ok\":%s,\"error\":\"%s\"}",
                      json_escape(r.suite).c_str(), json_escape(r.probe).c_str(),
                      json_escape(r.variant).c_str(),
                      static_cast<unsigned long long>(r.bytes),
                      static_cast<unsigned long long>(r.flops), r.gpu_seconds,
                      r.gpu_seconds_median, r.cpu_seconds, r.readback_seconds, gbs, tflops, gops,
                      r.groups, r.threads, r.runs, static_cast<unsigned long long>(r.iters),
                      r.gpu_valid ? "true" : "false", r.checksum_checked ? "true" : "false",
                      r.checksum_ok ? "true" : "false", json_escape(r.error).c_str());
        o += buf;
    }
    o += "]}\n";
    return o;
}

// ---------------------------------------------------------------------------
// Non-Windows: honest unavailable, no fabricated rows.
// ---------------------------------------------------------------------------

#if !defined(_WIN32)

GpupeakResult measure_gpupeak(const GpupeakOptions& options) {
    (void) options;
    GpupeakResult r;
    r.d3d12_ran = false;
    r.error = "d3d12 unavailable on this platform";
    return r;
}

#else

namespace {

using d3d12c::ComPtr;

constexpr std::uint32_t kThreadsStream = 256;
constexpr std::uint32_t kThreadsAlu = 64;
constexpr std::uint32_t kStreamGroups = 2048;
constexpr std::uint64_t kAluIters = 1024;
constexpr std::uint64_t kLatencyIters = 65536;
constexpr std::uint64_t kOccupancyIters = 200000;
constexpr std::uint32_t kSinkGroups = 8192; // 8192 * 64 * 4 B = 2 MiB

struct PsoEntry {
    const char* name = nullptr;
    const void* dxil = nullptr;
    std::size_t size = 0;
    ComPtr<ID3D12PipelineState> pso;
};

struct Driver {
    GpupeakResult* result = nullptr;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12DescriptorHeap> heap_a;
    ComPtr<ID3D12DescriptorHeap> heap_b;
    ComPtr<ID3D12Fence> fence;
    HANDLE fence_event = nullptr;
    std::uint64_t fence_value = 0;
    ComPtr<ID3D12QueryHeap> query_heap;
    ComPtr<ID3D12Resource> query_readback;
    ComPtr<ID3D12Resource> sink;
    ComPtr<ID3D12Resource> sink_readback;
    ComPtr<ID3D12Resource> buf_a;
    ComPtr<ID3D12Resource> buf_b;
    ComPtr<ID3D12Resource> cb;
    std::uint64_t timestamp_frequency = 0;
    std::uint64_t max_bytes = 0;
    bool timestamps_ok = false;

    bool submit_and_wait(double* cpu_sec) {
        const auto t0 = std::chrono::steady_clock::now();
        ID3D12CommandList* lists[] = {list.Get()};
        queue->ExecuteCommandLists(1, lists);
        const std::uint64_t v = ++fence_value;
        if (FAILED(queue->Signal(fence.Get(), v)))
            return false;
        if (fence->GetCompletedValue() < v) {
            if (FAILED(fence->SetEventOnCompletion(v, fence_event)))
                return false;
            WaitForSingleObject(fence_event, INFINITE);
        }
        if (cpu_sec)
            *cpu_sec =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return true;
    }

    bool resolve_timestamps(std::uint64_t* t0, std::uint64_t* t1) {
        if (!timestamps_ok)
            return false;
        alloc->Reset();
        list->Reset(alloc.Get(), nullptr);
        list->ResolveQueryData(query_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2,
                               query_readback.Get(), 0);
        list->Close();
        if (!submit_and_wait(nullptr))
            return false;
        void* mapped = nullptr;
        const D3D12_RANGE range = {0, sizeof(std::uint64_t) * 2};
        if (FAILED(query_readback->Map(0, &range, &mapped)) || !mapped)
            return false;
        const auto* stamps = static_cast<const std::uint64_t*>(mapped);
        *t0 = stamps[0];
        *t1 = stamps[1];
        query_readback->Unmap(0, nullptr);
        return *t1 > *t0;
    }

    void write_params(std::uint32_t p0, std::uint32_t gx, std::uint32_t gy, std::uint32_t mode) {
        void* mapped = nullptr;
        if (FAILED(cb->Map(0, nullptr, &mapped)) || !mapped)
            return;
        const std::uint32_t params[4] = {p0, gx, gy, mode};
        std::memcpy(mapped, params, sizeof(params));
        cb->Unmap(0, nullptr);
    }

    void begin_graphics(ID3D12PipelineState* pso, ID3D12DescriptorHeap* heap) {
        alloc->Reset();
        list->Reset(alloc.Get(), pso);
        list->SetComputeRootSignature(root.Get());
        ID3D12DescriptorHeap* heaps[] = {heap};
        list->SetDescriptorHeaps(1, heaps);
        list->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
        list->EndQuery(query_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
    }

    void dispatch_only(std::uint32_t gx, std::uint32_t gy) { list->Dispatch(gx, gy, 1); }

    void end_list() {
        D3D12_RESOURCE_BARRIER uav = {};
        uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        uav.UAV.pResource = sink.Get();
        list->ResourceBarrier(1, &uav);
        list->EndQuery(query_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
        list->Close();
    }

    // Single-dispatch list with no resource transitions (ALU/occupancy/empty).
    void end_dispatch(std::uint32_t gx, std::uint32_t gy) {
        dispatch_only(gx, gy);
        end_list();
    }

    void transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                    D3D12_RESOURCE_STATES after) {
        D3D12_RESOURCE_BARRIER b = {};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = resource;
        b.Transition.StateBefore = before;
        b.Transition.StateAfter = after;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        list->ResourceBarrier(1, &b);
    }

    bool create_pso(PsoEntry* entry) {
        if (!entry->dxil || entry->size == 0)
            return false;
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
        desc.pRootSignature = root.Get();
        desc.CS.pShaderBytecode = entry->dxil;
        desc.CS.BytecodeLength = entry->size;
        const HRESULT hr = device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&entry->pso));
        if (FAILED(hr)) {
            result->pso_errors.push_back(std::string(entry->name) + ": " +
                                         d3d12c::hr_message("CreateComputePipelineState", hr));
            return false;
        }
        return true;
    }

};

bool build_descriptors(Driver& drv) {
    struct Range {
        D3D12_DESCRIPTOR_RANGE_TYPE type;
        UINT count;
        UINT reg;
        UINT offset;
    } ranges[4] = {
        {D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0, 0},
        {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 1},
        {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 2},
        {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 1, 3},
    };
    D3D12_DESCRIPTOR_RANGE desc_ranges[4] = {};
    for (int i = 0; i < 4; ++i) {
        desc_ranges[i].RangeType = ranges[i].type;
        desc_ranges[i].NumDescriptors = ranges[i].count;
        desc_ranges[i].BaseShaderRegister = ranges[i].reg;
        desc_ranges[i].RegisterSpace = 0;
        desc_ranges[i].OffsetInDescriptorsFromTableStart = ranges[i].offset;
    }
    D3D12_ROOT_PARAMETER param = {};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    param.DescriptorTable.NumDescriptorRanges = 4;
    param.DescriptorTable.pDescriptorRanges = desc_ranges;
    param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC desc = {};
    desc.NumParameters = 1;
    desc.pParameters = &param;
    desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> sig, err;
    const auto serialize = d3d12_dyn::SerializeRootSignature();
    if (!serialize) {
        drv.result->error = "D3D12SerializeRootSignature not available";
        return false;
    }
    if (FAILED(serialize(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err))) {
        drv.result->error = "D3D12SerializeRootSignature failed";
        if (err)
            drv.result->error += std::string(": ") + static_cast<const char*>(err->GetBufferPointer());
        return false;
    }
    if (FAILED(drv.device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
                                               IID_PPV_ARGS(&drv.root)))) {
        drv.result->error = "CreateRootSignature failed";
        return false;
    }

    D3D12_DESCRIPTOR_HEAP_DESC heap_desc = {};
    heap_desc.NumDescriptors = 4;
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(drv.device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&drv.heap_a))) ||
        FAILED(drv.device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&drv.heap_b)))) {
        drv.result->error = "CreateDescriptorHeap failed";
        return false;
    }

    const UINT incr =
        drv.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CONSTANT_BUFFER_VIEW_DESC cbv = {};
    cbv.BufferLocation = drv.cb->GetGPUVirtualAddress();
    cbv.SizeInBytes = 256;

    D3D12_UNORDERED_ACCESS_VIEW_DESC sink_uav = {};
    sink_uav.Format = DXGI_FORMAT_UNKNOWN;
    sink_uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    sink_uav.Buffer.NumElements = kSinkGroups * kThreadsAlu;
    sink_uav.Buffer.StructureByteStride = sizeof(std::uint32_t);

    struct Binding {
        ID3D12DescriptorHeap* heap;
        ID3D12Resource* srv;
        ID3D12Resource* uav;
    } bindings[2] = {
        {drv.heap_a.Get(), drv.buf_a.Get(), drv.buf_b.Get()},
        {drv.heap_b.Get(), drv.buf_b.Get(), drv.buf_a.Get()},
    };
    for (const Binding& binding : bindings) {
        D3D12_CPU_DESCRIPTOR_HANDLE cpu = binding.heap->GetCPUDescriptorHandleForHeapStart();
        drv.device->CreateConstantBufferView(&cbv, cpu);

        D3D12_CPU_DESCRIPTOR_HANDLE cpu_srv = cpu;
        cpu_srv.ptr += incr;
        D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
        srv.Format = DXGI_FORMAT_UNKNOWN;
        srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Buffer.NumElements = static_cast<UINT>(drv.max_bytes / 4);
        srv.Buffer.StructureByteStride = 0;
        drv.device->CreateShaderResourceView(binding.srv, &srv, cpu_srv);

        D3D12_CPU_DESCRIPTOR_HANDLE cpu_sink = cpu;
        cpu_sink.ptr += 2 * static_cast<SIZE_T>(incr);
        drv.device->CreateUnorderedAccessView(drv.sink.Get(), nullptr, &sink_uav, cpu_sink);

        D3D12_CPU_DESCRIPTOR_HANDLE cpu_raw = cpu;
        cpu_raw.ptr += 3 * static_cast<SIZE_T>(incr);
        D3D12_UNORDERED_ACCESS_VIEW_DESC raw_uav = {};
        raw_uav.Format = DXGI_FORMAT_UNKNOWN;
        raw_uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        raw_uav.Buffer.NumElements = static_cast<UINT>(drv.max_bytes / 4);
        raw_uav.Buffer.StructureByteStride = 0;
        drv.device->CreateUnorderedAccessView(binding.uav, nullptr, &raw_uav, cpu_raw);
    }
    return true;
}

bool validate_stream(Driver& drv, ID3D12DescriptorHeap* heap, ID3D12Resource* source,
                     ID3D12PipelineState* read_pso, std::uint64_t n_vec, std::uint32_t gx,
                     std::uint32_t gy, bool* ok) {
    *ok = false;
    drv.write_params(static_cast<std::uint32_t>(n_vec), gx, gy, /*mode=*/1);
    drv.begin_graphics(read_pso, heap);
    drv.transition(source, D3D12_RESOURCE_STATE_COMMON,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    drv.dispatch_only(gx, gy);
    drv.transition(source, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                   D3D12_RESOURCE_STATE_COMMON);
    drv.end_list();
    if (!drv.submit_and_wait(nullptr)) // the read dispatch must run before any reset
        return false;
    // Copy the whole sink and fold it on the host.
    drv.alloc->Reset();
    drv.list->Reset(drv.alloc.Get(), nullptr);
    drv.list->CopyResource(drv.sink_readback.Get(), drv.sink.Get());
    drv.list->Close();
    if (!drv.submit_and_wait(nullptr))
        return false;
    void* mapped = nullptr;
    const D3D12_RANGE range = {0, static_cast<SIZE_T>(gx) * gy * sizeof(std::uint32_t)};
    if (FAILED(drv.sink_readback->Map(0, &range, &mapped)) || !mapped)
        return false;
    const auto* words = static_cast<const std::uint32_t*>(mapped);
    std::uint32_t folded = 0;
    for (std::uint32_t i = 0; i < gx * gy; ++i)
        folded ^= words[i];
    drv.sink_readback->Unmap(0, nullptr);
    *ok = folded == gpupeak_pattern_xor(n_vec, gx, gy, 0);
    return true;
}

} // namespace

GpupeakResult measure_gpupeak(const GpupeakOptions& options) {
    GpupeakResult result;
    Driver drv;
    drv.result = &result;

    std::string err;
    drv.device = d3d12c::create_device(&err);
    if (!drv.device) {
        result.error = err.empty() ? "D3D12CreateDevice failed" : err;
        return result;
    }

    // --- Adapter description + local memory budget -------------------------
    {
        ComPtr<IDXGIFactory4> factory;
        if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
            ComPtr<IDXGIAdapter1> adapter;
            if (SUCCEEDED(factory->EnumAdapterByLuid(drv.device->GetAdapterLuid(), IID_PPV_ARGS(&adapter)))) {
                DXGI_ADAPTER_DESC1 desc = {};
                if (SUCCEEDED(adapter->GetDesc1(&desc))) {
                    const std::size_t len = wcslen(desc.Description);
                    std::string name;
                    name.reserve(len);
                    for (std::size_t i = 0; i < len; ++i)
                        name.push_back(static_cast<char>(desc.Description[i]));
                    result.device.adapter = name;
                    result.device.vendor_id = desc.VendorId;
                    result.device.device_id = desc.DeviceId;
                    result.device.dedicated_vram = desc.DedicatedVideoMemory;
                    result.device.shared_system = desc.SharedSystemMemory;
                }
                ComPtr<IDXGIAdapter3> adapter3;
                if (SUCCEEDED(adapter.As(&adapter3))) {
                    DXGI_QUERY_VIDEO_MEMORY_INFO info = {};
                    if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
                                                                &info))) {
                        result.device.local_budget = info.Budget;
                        result.device.local_usage = info.CurrentUsage;
                        result.device.local_available = info.AvailableForReservation;
                    }
                }
            }
        }
    }

    // --- Feature support ---------------------------------------------------
    {
        D3D12_FEATURE_DATA_D3D12_OPTIONS options12 = {};
        if (SUCCEEDED(drv.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options12,
                                                      sizeof(options12)))) {
            result.device.max_lds_bytes = options12.MaxComputeSharedMemorySize;
        } else {
            result.device.feature_error = "D3D12_FEATURE_D3D12_OPTIONS unavailable";
        }
        D3D12_FEATURE_DATA_D3D12_OPTIONS1 options1 = {};
        if (SUCCEEDED(drv.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &options1,
                                                      sizeof(options1)))) {
            result.device.wave_lane_min = options1.WaveLaneCountMin;
            result.device.wave_lane_max = options1.WaveLaneCountMax;
        }
    }

    // --- Queue, list, fence ------------------------------------------------
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(drv.device->CreateCommandQueue(&qd, IID_PPV_ARGS(&drv.queue))) ||
        FAILED(drv.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                  IID_PPV_ARGS(&drv.alloc))) ||
        FAILED(drv.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                             drv.alloc.Get(), nullptr, IID_PPV_ARGS(&drv.list))) ||
        FAILED(drv.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&drv.fence)))) {
        result.error = "D3D12 queue/list/fence creation failed";
        return result;
    }
    drv.fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!drv.fence_event) {
        result.error = "CreateEventW failed";
        return result;
    }
    drv.list->Close();

    // --- Timestamps --------------------------------------------------------
    {
        D3D12_QUERY_HEAP_DESC qhd = {};
        qhd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        qhd.Count = 2;
        if (SUCCEEDED(drv.device->CreateQueryHeap(&qhd, IID_PPV_ARGS(&drv.query_heap))) &&
            SUCCEEDED(drv.queue->GetTimestampFrequency(&drv.timestamp_frequency)) &&
            drv.timestamp_frequency > 0) {
            drv.query_readback = d3d12c::create_buffer(
                drv.device.Get(), 256, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST, "gpupeak_query_readback", &err);
            drv.timestamps_ok = drv.query_readback != nullptr;
        }
        result.device.timestamp_frequency = drv.timestamp_frequency;
        result.device.timestamp_us_per_tick =
            drv.timestamp_frequency
                ? 1e6 / static_cast<double>(drv.timestamp_frequency)
                : 0.0;
        if (!drv.timestamps_ok)
            result.error = "GPU timestamps unavailable; refusing to report CPU-timed rates";
    }
    if (!drv.timestamps_ok) {
        if (drv.fence_event)
            CloseHandle(drv.fence_event);
        return result;
    }

    // --- Buffers -----------------------------------------------------------
    drv.max_bytes = options.run_big_stream ? std::max(options.stream_bytes, options.big_stream_bytes)
                                           : options.stream_bytes;
    drv.buf_a = d3d12c::create_buffer(drv.device.Get(), drv.max_bytes, D3D12_HEAP_TYPE_DEFAULT,
                                      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                      D3D12_RESOURCE_STATE_COMMON, "gpupeak_buf_a", &err);
    drv.buf_b = d3d12c::create_buffer(drv.device.Get(), drv.max_bytes, D3D12_HEAP_TYPE_DEFAULT,
                                      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                      D3D12_RESOURCE_STATE_COMMON, "gpupeak_buf_b", &err);
    const UINT64 sink_bytes = static_cast<UINT64>(kSinkGroups) * kThreadsAlu * sizeof(std::uint32_t);
    drv.sink = d3d12c::create_buffer(drv.device.Get(), sink_bytes, D3D12_HEAP_TYPE_DEFAULT,
                                     D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                     D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "gpupeak_sink", &err);
    drv.sink_readback = d3d12c::create_buffer(
        drv.device.Get(), sink_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COPY_DEST, "gpupeak_sink_readback", &err);
    drv.cb = d3d12c::create_buffer(drv.device.Get(), 256, D3D12_HEAP_TYPE_UPLOAD,
                                   D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ,
                                   "gpupeak_cb", &err);
    if (!drv.buf_a || !drv.buf_b || !drv.sink || !drv.sink_readback || !drv.cb) {
        result.error = err.empty() ? "GPU buffer creation failed" : err;
        CloseHandle(drv.fence_event);
        return result;
    }
    if (!build_descriptors(drv)) {
        CloseHandle(drv.fence_event);
        return result;
    }

    // --- PSOs --------------------------------------------------------------
    PsoEntry stream_read{"stream_read", g_gpupeak_stream_read_dxil,
                         g_gpupeak_stream_read_dxil_size, nullptr};
    PsoEntry stream_write{"stream_write", g_gpupeak_stream_write_dxil,
                          g_gpupeak_stream_write_dxil_size, nullptr};
    PsoEntry stream_copy{"stream_copy", g_gpupeak_stream_copy_dxil,
                         g_gpupeak_stream_copy_dxil_size, nullptr};
    PsoEntry fma_f32{"fma_f32", g_gpupeak_fma_f32_dxil, g_gpupeak_fma_f32_dxil_size, nullptr};
    PsoEntry fma_f16{"fma_f16", g_gpupeak_fma_f16_dxil, g_gpupeak_fma_f16_dxil_size, nullptr};
    PsoEntry dot2_f16{"dot2_f16", g_gpupeak_dot2_f16_dxil, g_gpupeak_dot2_f16_dxil_size, nullptr};
    PsoEntry dot4_i8{"dot4_i8", g_gpupeak_dot4_i8_dxil, g_gpupeak_dot4_i8_dxil_size, nullptr};
    PsoEntry latency_fma{"latency_fma", g_gpupeak_latency_fma_dxil,
                         g_gpupeak_latency_fma_dxil_size, nullptr};
    PsoEntry cu_probe_32{"cu_probe_32", g_gpupeak_cu_probe_32_dxil,
                         g_gpupeak_cu_probe_32_dxil_size, nullptr};
    PsoEntry cu_probe_16{"cu_probe_16", g_gpupeak_cu_probe_16_dxil,
                         g_gpupeak_cu_probe_16_dxil_size, nullptr};
    PsoEntry empty{"empty", g_gpupeak_empty_dxil, g_gpupeak_empty_dxil_size, nullptr};
    PsoEntry* all[] = {&stream_read, &stream_write, &stream_copy, &fma_f32,   &fma_f16,
                       &dot2_f16,    &dot4_i8,      &latency_fma, &cu_probe_32, &cu_probe_16,
                       &empty};
    for (PsoEntry* entry : all)
        drv.create_pso(entry);

    // --- Stream suite ------------------------------------------------------
    if (options.run_stream && stream_write.pso && stream_read.pso) {
        struct Pass {
            std::uint64_t bytes;
        } passes[2] = {{options.stream_bytes},
                       {options.run_big_stream ? options.big_stream_bytes : 0}};
        for (const Pass& pass : passes) {
            if (pass.bytes < (1u << 16))
                continue;
            const std::uint64_t n_vec = pass.bytes / 16u;
            const GpupeakGrid grid = gpupeak_plan_grid(kStreamGroups);
            const std::uint32_t gx = grid.groups_x;
            const std::uint32_t gy = grid.groups_y;
            const std::string size_tag = std::to_string(pass.bytes);

            // Fill A with the deterministic GPU-side pattern (heap_b maps its
            // raw UAV to buf_a), then prove the read kernel covers it before
            // any rate is reported.
            drv.write_params(static_cast<std::uint32_t>(n_vec), gx, gy, 0);
            double cpu_sec = 0.0;
            drv.begin_graphics(stream_write.pso.Get(), drv.heap_b.Get());
            drv.transition(drv.buf_a.Get(), D3D12_RESOURCE_STATE_COMMON,
                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            drv.dispatch_only(gx, gy);
            drv.transition(drv.buf_a.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                           D3D12_RESOURCE_STATE_COMMON);
            drv.end_list();
            if (!drv.submit_and_wait(&cpu_sec))
                continue;
            bool ok = false;
            if (validate_stream(drv, drv.heap_a.Get(), drv.buf_a.Get(), stream_read.pso.Get(), n_vec,
                                gx, gy, &ok)) {
                GpupeakRow row;
                row.suite = "stream";
                row.probe = "read";
                row.variant = size_tag + "_validate";
                row.bytes = pass.bytes;
                row.checksum_checked = true;
                row.checksum_ok = ok;
                row.error = ok ? "" : "pattern checksum mismatch";
                result.rows.push_back(row);
            }

            auto run_stream_row = [&](const char* probe, PsoEntry* pso, ID3D12DescriptorHeap* heap,
                                      ID3D12Resource* read_res, ID3D12Resource* write_res) {
                if (!pso || !pso->pso)
                    return;
                std::vector<double> gpu_times, cpu_times;
                bool valid = true;
                for (int it = 0; it < options.iterations + 1; ++it) {
                    drv.write_params(static_cast<std::uint32_t>(n_vec), gx, gy, 0);
                    drv.begin_graphics(pso->pso.Get(), heap);
                    if (read_res) {
                        drv.transition(read_res, D3D12_RESOURCE_STATE_COMMON,
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                    }
                    if (write_res) {
                        drv.transition(write_res, D3D12_RESOURCE_STATE_COMMON,
                                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                    }
                    drv.dispatch_only(gx, gy);
                    if (read_res) {
                        drv.transition(read_res, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                       D3D12_RESOURCE_STATE_COMMON);
                    }
                    if (write_res) {
                        drv.transition(write_res, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                       D3D12_RESOURCE_STATE_COMMON);
                    }
                    drv.end_list();
                    double c = 0.0;
                    if (!drv.submit_and_wait(&c)) {
                        valid = false;
                        break;
                    }
                    std::uint64_t t0 = 0, t1 = 0;
                    if (!drv.resolve_timestamps(&t0, &t1)) {
                        valid = false;
                        break;
                    }
                    if (it > 0) { // first dispatch is warmup
                        gpu_times.push_back(static_cast<double>(t1 - t0) /
                                            static_cast<double>(drv.timestamp_frequency));
                        cpu_times.push_back(c);
                    }
                }
                if (valid && !gpu_times.empty()) {
                    GpupeakRow row;
                    row.suite = "stream";
                    row.probe = probe;
                    row.variant = size_tag;
                    row.bytes = pass.bytes;
                    row.groups = gx * gy;
                    row.threads = kThreadsStream;
                    row.runs = static_cast<std::uint32_t>(gpu_times.size());
                    row.gpu_seconds = *std::min_element(gpu_times.begin(), gpu_times.end());
                    row.gpu_seconds_median = gpupeak_median(gpu_times);
                    row.cpu_seconds = *std::min_element(cpu_times.begin(), cpu_times.end());
                    row.gpu_valid = true;
                    result.rows.push_back(row);
                }
            };

            // heap_b's raw UAV is buf_a; heap_a's raw UAV is buf_b.
            run_stream_row("write", &stream_write, drv.heap_b.Get(), nullptr, drv.buf_a.Get());
            run_stream_row("read", &stream_read, drv.heap_a.Get(), drv.buf_a.Get(), nullptr);
            run_stream_row("copy", &stream_copy, drv.heap_a.Get(), drv.buf_a.Get(), drv.buf_b.Get());

            // Validate the copy destination too: B must now hold the pattern.
            if (stream_copy.pso) {
                bool copy_ok = false;
                if (validate_stream(drv, drv.heap_b.Get(), drv.buf_b.Get(), stream_read.pso.Get(),
                                    n_vec, gx, gy, &copy_ok)) {
                    GpupeakRow row;
                    row.suite = "stream";
                    row.probe = "copy";
                    row.variant = size_tag + "_validate";
                    row.bytes = pass.bytes;
                    row.checksum_checked = true;
                    row.checksum_ok = copy_ok;
                    row.error = copy_ok ? "" : "copied pattern checksum mismatch";
                    result.rows.push_back(row);
                }
            }
        }
    }

    // --- ALU throughput sweep ---------------------------------------------
    if (options.run_alu) {
        struct AluProbe {
            PsoEntry* entry;
            const char* name;
        } probes[] = {{&fma_f32, "fma_f32"},
                      {&fma_f16, "fma_f16"},
                      {&dot2_f16, "dot2_f16"},
                      {&dot4_i8, "dot4_i8"}};
        const std::uint32_t sweep[] = {52, 104, 208, 416, 832, 1664};
        for (const AluProbe& probe : probes) {
            if (!probe.entry->pso)
                continue;
            for (std::uint32_t groups : sweep) {
                const GpupeakGrid grid = gpupeak_plan_grid(groups);
                std::vector<double> gpu_times, cpu_times;
                bool valid = true;
                for (int it = 0; it < options.iterations + 1; ++it) {
                    drv.write_params(static_cast<std::uint32_t>(kAluIters), grid.groups_x,
                                     grid.groups_y, 0);
                    drv.begin_graphics(probe.entry->pso.Get(), drv.heap_a.Get());
                    drv.end_dispatch(grid.groups_x, grid.groups_y);
                    double c = 0.0;
                    if (!drv.submit_and_wait(&c)) {
                        valid = false;
                        break;
                    }
                    std::uint64_t t0 = 0, t1 = 0;
                    if (!drv.resolve_timestamps(&t0, &t1)) {
                        valid = false;
                        break;
                    }
                    if (it > 0) {
                        gpu_times.push_back(static_cast<double>(t1 - t0) /
                                            static_cast<double>(drv.timestamp_frequency));
                        cpu_times.push_back(c);
                    }
                }
                if (!valid || gpu_times.empty())
                    continue;
                GpupeakRow row;
                row.suite = "alu";
                row.probe = probe.name;
                row.variant = std::to_string(groups) + "_groups";
                row.flops = gpupeak_alu_flops(probe.name, grid.groups_x * grid.groups_y, kAluIters);
                row.groups = grid.groups_x * grid.groups_y;
                row.threads = kThreadsAlu;
                row.iters = kAluIters;
                row.runs = static_cast<std::uint32_t>(gpu_times.size());
                row.gpu_seconds = *std::min_element(gpu_times.begin(), gpu_times.end());
                row.gpu_seconds_median = gpupeak_median(gpu_times);
                row.cpu_seconds = *std::min_element(cpu_times.begin(), cpu_times.end());
                row.gpu_valid = true;
                result.rows.push_back(row);
            }
        }

        // Dependent chain: latency, and the implied clock from it.
        if (latency_fma.pso) {
            std::vector<double> per_fma;
            for (int it = 0; it < 2; ++it) {
                drv.write_params(static_cast<std::uint32_t>(kLatencyIters), 1, 1, 0);
                drv.begin_graphics(latency_fma.pso.Get(), drv.heap_a.Get());
                drv.end_dispatch(1, 1);
                double c = 0.0;
                if (!drv.submit_and_wait(&c))
                    continue;
                std::uint64_t t0 = 0, t1 = 0;
                if (!drv.resolve_timestamps(&t0, &t1))
                    continue;
                const double sec =
                    static_cast<double>(t1 - t0) / static_cast<double>(drv.timestamp_frequency);
                per_fma.push_back(sec / static_cast<double>(kLatencyIters));
            }
            if (!per_fma.empty()) {
                const double best = *std::min_element(per_fma.begin(), per_fma.end());
                result.implied_clock_seconds_per_fma = best;
                result.implied_clock_mhz = gpupeak_implied_clock_mhz(best);
                GpupeakRow row;
                row.suite = "alu";
                row.probe = "latency_fma";
                row.variant = std::to_string(kLatencyIters) + "_chain";
                row.iters = kLatencyIters;
                row.groups = 1;
                row.threads = kThreadsAlu;
                row.runs = static_cast<std::uint32_t>(per_fma.size());
                row.gpu_seconds = best * static_cast<double>(kLatencyIters);
                row.gpu_seconds_median = row.gpu_seconds;
                row.gpu_valid = true;
                result.rows.push_back(row);
            }
        }
    }

    // --- Empty dispatch latency -------------------------------------------
    if (options.run_latency && empty.pso) {
        constexpr int kCalls = 64;
        std::vector<double> gpu_per_call, cpu_per_call, readback;
        drv.write_params(0x5a5a5a5au, 1, 1, 0);
        bool valid = true;
        for (int i = 0; i < kCalls; ++i) {
            drv.begin_graphics(empty.pso.Get(), drv.heap_a.Get());
            drv.end_dispatch(1, 1);
            double c = 0.0;
            if (!drv.submit_and_wait(&c)) {
                valid = false;
                break;
            }
            std::uint64_t t0 = 0, t1 = 0;
            if (!drv.resolve_timestamps(&t0, &t1)) {
                valid = false;
                break;
            }
            if (i >= 4) { // first four calls are warmup
                gpu_per_call.push_back(static_cast<double>(t1 - t0) /
                                       static_cast<double>(drv.timestamp_frequency));
                cpu_per_call.push_back(c);
            }
            // Readback Map cost: the sink never changes, so this isolates the
            // Map + memcpy + Unmap path a decode readback pays.
            drv.alloc->Reset();
            drv.list->Reset(drv.alloc.Get(), nullptr);
            // Four bytes are enough: the cost under test is the Map/wake-up
            // path, not the copy volume.
            drv.list->CopyBufferRegion(drv.sink_readback.Get(), 0, drv.sink.Get(), 0, 4);
            drv.list->Close();
            if (!drv.submit_and_wait(nullptr)) {
                valid = false;
                break;
            }
            const auto t_rb = std::chrono::steady_clock::now();
            void* mapped = nullptr;
            const D3D12_RANGE range = {0, 64};
            if (SUCCEEDED(drv.sink_readback->Map(0, &range, &mapped)) && mapped) {
                volatile std::uint32_t probe = static_cast<const std::uint32_t*>(mapped)[0];
                (void) probe;
                drv.sink_readback->Unmap(0, nullptr);
                readback.push_back(
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t_rb).count());
            }
        }
        auto push_latency = [&](const char* probe, const char* stat, const std::vector<double>& v,
                                bool gpu_time) {
            if (v.empty())
                return;
            GpupeakRow row;
            row.suite = "latency";
            row.probe = probe;
            row.variant = stat;
            row.groups = 1;
            row.threads = kThreadsAlu;
            row.runs = static_cast<std::uint32_t>(v.size());
            const std::string name = stat;
            const double value = name == "min"   ? *std::min_element(v.begin(), v.end())
                                  : name == "p50" ? gpupeak_percentile(v, 0.50)
                                  : name == "p95" ? gpupeak_percentile(v, 0.95)
                                                  : gpupeak_median(v);
            if (gpu_time) {
                row.gpu_seconds = value;
                row.gpu_seconds_median = gpupeak_median(v);
                row.gpu_valid = true;
            } else {
                row.cpu_seconds = value;
            }
            result.rows.push_back(row);
        };
        if (valid && !gpu_per_call.empty()) {
            push_latency("empty_dispatch_gpu", "min", gpu_per_call, true);
            push_latency("empty_dispatch_gpu", "p50", gpu_per_call, true);
            push_latency("empty_dispatch_gpu", "p95", gpu_per_call, true);
            push_latency("empty_submit_fence_cpu", "min", cpu_per_call, false);
            push_latency("empty_submit_fence_cpu", "p50", cpu_per_call, false);
            push_latency("empty_submit_fence_cpu", "p95", cpu_per_call, false);
            push_latency("empty_readback_cpu", "p50", readback, false);
        }

        // Amortized per-dispatch cost inside one list: 64 dispatches, one pair.
        drv.write_params(0x5a5a5a5au, 1, 1, 0);
        drv.begin_graphics(empty.pso.Get(), drv.heap_a.Get());
        for (int i = 0; i < kCalls; ++i) {
            drv.list->Dispatch(1, 1, 1);
            D3D12_RESOURCE_BARRIER uav = {};
            uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            uav.UAV.pResource = drv.sink.Get();
            drv.list->ResourceBarrier(1, &uav);
        }
        drv.list->EndQuery(drv.query_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
        drv.list->Close();
        if (drv.submit_and_wait(nullptr)) {
            std::uint64_t t0 = 0, t1 = 0;
            if (drv.resolve_timestamps(&t0, &t1)) {
                GpupeakRow row;
                row.suite = "latency";
                row.probe = "empty_dispatch_batch";
                row.variant = std::to_string(kCalls) + "_calls";
                row.groups = 1;
                row.threads = kThreadsAlu;
                row.runs = static_cast<std::uint32_t>(kCalls);
                row.gpu_seconds =
                    static_cast<double>(t1 - t0) / static_cast<double>(drv.timestamp_frequency) /
                    static_cast<double>(kCalls);
                row.gpu_seconds_median = row.gpu_seconds;
                row.gpu_valid = true;
                result.rows.push_back(row);
            }
        }
    }

    // --- Occupancy (CU inference) sweep -----------------------------------
    if (options.run_occupancy && (cu_probe_32.pso || cu_probe_16.pso)) {
        struct OccupancyProbe {
            PsoEntry* entry;
            const char* name;
        } probes[] = {{&cu_probe_32, "cu32"}, {&cu_probe_16, "cu16"}};
        const std::uint32_t sweep[] = {13, 26, 52, 78, 104, 130, 156, 208, 260, 312, 364, 416, 468,
                                       520, 624};
        for (const OccupancyProbe& probe : probes) {
            if (!probe.entry->pso)
                continue;
            for (std::uint32_t groups : sweep) {
                const GpupeakGrid grid = gpupeak_plan_grid(groups);
                std::vector<double> gpu_times;
                for (int it = 0; it < 3; ++it) { // one warmup + two samples
                    drv.write_params(static_cast<std::uint32_t>(kOccupancyIters), grid.groups_x,
                                     grid.groups_y, 0);
                    drv.begin_graphics(probe.entry->pso.Get(), drv.heap_a.Get());
                    drv.end_dispatch(grid.groups_x, grid.groups_y);
                    double c = 0.0;
                    if (!drv.submit_and_wait(&c))
                        break;
                    std::uint64_t t0 = 0, t1 = 0;
                    if (!drv.resolve_timestamps(&t0, &t1))
                        break;
                    if (it > 0)
                        gpu_times.push_back(static_cast<double>(t1 - t0) /
                                            static_cast<double>(drv.timestamp_frequency));
                }
                if (gpu_times.empty())
                    continue;
                GpupeakRow row;
                row.suite = "occupancy";
                row.probe = probe.name;
                row.variant = std::to_string(groups) + "_groups";
                row.groups = grid.groups_x * grid.groups_y;
                row.threads = kThreadsAlu;
                row.iters = kOccupancyIters;
                row.runs = static_cast<std::uint32_t>(gpu_times.size());
                row.gpu_seconds = *std::min_element(gpu_times.begin(), gpu_times.end());
                row.gpu_seconds_median = gpupeak_median(gpu_times);
                row.gpu_valid = true;
                result.rows.push_back(row);
            }
        }
    }

    CloseHandle(drv.fence_event);
    drv.fence_event = nullptr;
    result.d3d12_ran = true;
    return result;
}

#endif // _WIN32

} // namespace xllama
