// RemoteAttnEngine — D3D12 executor for the RKVA service (Xbox Series X).
//
// F16 bring-up path: stores K/V as F16 (GGML RNE) in a per-session, per-layer
// DEFAULT-heap buffer and runs causal GQA flash attention on the GPU. This
// validates the full split-execution pipeline (RPC -> session -> store ->
// attention -> readback) independently of KVarN numerics, and remains the
// materialize/correctness fallback once the native KVarN-over-records kernels
// land. One engine serves the single persistent RKVA connection; calls are
// serialized by the listener, so no internal locking is needed.

#include "remote-attn-engine.h"

#include "rkva-kvarn-bulk.h"
#include "rkva-kvarn-range.h"
#include "rkva-kvarn-rewind.h"
#include "rkva-range-profile.h"
#include "rkva_flash_attn_dxil.h"
#include "rkva_flash_attn_kvarn4_dxil.h"
#include "rkva_flash_attn_split_k_reduce_dxil.h"
#include "rkva_range_attn_dxil.h"
#include "rkva_range_attn_f16_mq2_dxil.h"
#include "rkva_range_attn_f16_mq4_dxil.h"
#include "rkva_range_attn_f16_opt_dxil.h"
#include "rkva_range_attn_kvarn4_dxil.h"
#include "rkva_range_attn_kvarn4_mq2_dxil.h"
#if __has_include("rkva_range_attn_kvarn4_mq4_dxil.h")
    #include "rkva_range_attn_kvarn4_mq4_dxil.h"
    #define RKVA_HAS_KVARN4_MQ4_DXIL 1
#else
    #define RKVA_HAS_KVARN4_MQ4_DXIL 0
#endif
#include "rkva_range_attn_kvarn4_opt_dxil.h"
#include "rkva_range_attn_kvarn4_opt_prefix_dxil.h"
#if __has_include("rkva_range_attn_kvarn4_decode_prefix_dxil.h")
    #include "rkva_range_attn_kvarn4_decode_prefix_dxil.h"
    #define RKVA_HAS_KVARN4_DECODE_PREFIX_DXIL 1
#else
    #define RKVA_HAS_KVARN4_DECODE_PREFIX_DXIL 0
#endif
#include "rkva_range_attn_kvarn4_prefix_dxil.h"
#include "rkva_range_attn_q4_dxil.h"
#include "rkva_range_attn_q4_mq2_dxil.h"
#include "rkva_range_attn_q4_mq4_dxil.h"
#include "rkva_range_attn_q4_opt_dxil.h"
#include "rkva_range_attn_q8_dxil.h"
#include "rkva_range_attn_q8_mq2_dxil.h"
#include "rkva_range_attn_q8_mq4_dxil.h"
#include "rkva_range_attn_q8_opt_dxil.h"
#include "rkva_range_reduce_dxil.h"
#include "rkva_range_reduce_kvarn4_dxil.h"
#include "rkva_range_reduce_opt_dxil.h"
#include "rkva_store_f16_dxil.h"
#include "rkva_store_kvarn4_dxil.h"
#include "rkva_wht_dxil.h"
#include "xllama/d3d12_dyn.h"
#include <chrono>
#include <dxgi1_4.h>
#include <roapi.h>
#include <winrt/Windows.System.h>

static const uint8_t* kRkvaStoreKvarn4Dxil = g_rkva_store_kvarn4_dxil;
static const size_t kRkvaStoreKvarn4DxilSize = sizeof(g_rkva_store_kvarn4_dxil);

static const uint8_t* kRkvaFlashAttnKvarn4Dxil = g_rkva_flash_attn_kvarn4_dxil;
static const size_t kRkvaFlashAttnKvarn4DxilSize = sizeof(g_rkva_flash_attn_kvarn4_dxil);

static const uint8_t* kRkvaFlashAttnSplitKReduceDxil = g_rkva_flash_attn_split_k_reduce_dxil;
static const size_t kRkvaFlashAttnSplitKReduceDxilSize =
    sizeof(g_rkva_flash_attn_split_k_reduce_dxil);

#include <dxgi1_4.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <new>

using Microsoft::WRL::ComPtr;

namespace {

uint64_t aligned4(uint64_t bytes) {
    return (bytes + 3ull) & ~3ull;
}

bool fail(std::string* error, const char* stage, HRESULT hr = S_OK) {
    if (error) {
        char text[160];
        if (hr == S_OK) {
            std::snprintf(text, sizeof(text), "%s", stage);
        } else {
            std::snprintf(text, sizeof(text), "%s: 0x%08X", stage, static_cast<unsigned>(hr));
        }
        *error = text;
    }
    return false;
}

namespace rp = rkva_rprof;
using prof_clock = std::chrono::steady_clock;

uint64_t elapsed_ns(prof_clock::time_point from, prof_clock::time_point to) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(to - from).count());
}

// Disjoint consecutive CPU spans; inactive (no clock reads) unless profiling.
struct SpanTimer {
    bool on = false;
    prof_clock::time_point last;
    uint64_t* spans = nullptr;
    void start(bool enabled, uint64_t* target) {
        on = enabled;
        spans = target;
        if (on) {
            last = prof_clock::now();
        }
    }
    void mark(uint32_t span) {
        if (!on) {
            return;
        }
        const prof_clock::time_point now = prof_clock::now();
        spans[span] += elapsed_ns(last, now);
        last = now;
    }
};

// Counts a profiled call that returned before recording its sample.
struct FailedCallGuard {
    uint64_t* failed = nullptr;
    bool done = false;
    ~FailedCallGuard() {
        if (failed && !done) {
            ++*failed;
        }
    }
};

// Timestamp slots layout per set. Non-bulk uses slots 0..5.
// Bulk uses:
//   Slot 0: envelope begin
//   Segments 0..31: 8 slots each starting at slot 1 + i * 8:
//     upload_begin, upload_end, attn_begin, attn_pass1, attn_reduce, attn_end,
//     pack_end, copy_end. upload_end..attn_begin includes bulk unpack/WHT.
//   Slot 1 + segment_count * 8: envelope end
// Total max slots for 32 segments: 2 + 32 * 8 = 258 <= 512.
constexpr uint32_t kTsUploadBegin = 0;
constexpr uint32_t kTsUploadEnd = 1;
constexpr uint32_t kTsAttnBegin = 2;
constexpr uint32_t kTsAttnPass1 = 3;
constexpr uint32_t kTsAttnReduce = 4;
constexpr uint32_t kTsAttnCopy = 5;

constexpr uint32_t kTsBulkEnvelopeBegin = 0;
constexpr uint32_t kTsBulkUploadBegin = 0;
constexpr uint32_t kTsBulkUploadEnd = 1;
constexpr uint32_t kTsBulkAttnBegin = 2;
constexpr uint32_t kTsBulkAttnPass1 = 3;
constexpr uint32_t kTsBulkAttnReduce = 4;
constexpr uint32_t kTsBulkAttnCopy = 5;
constexpr uint32_t kTsBulkPackEnd = 6;
constexpr uint32_t kTsBulkCopyEnd = 7;
constexpr uint32_t kTsBulkSlotsPerSegment = 8;
constexpr uint32_t kTsBulkMaxSegments = 32;

constexpr uint32_t kTsPerSet = 512;
constexpr uint32_t kTsSets = 4;

struct BulkSegmentProfile {
    uint32_t n_queries = 0;
    uint32_t append_rows = 0;
    uint32_t live_count = 0;
    uint32_t active_count = 0;
    uint32_t split_k = 0;
    uint32_t groups = 0;
    uint32_t dispatches = 0;
    uint8_t q_tile = 0;
    uint8_t decode_opt = 0;
    uint8_t direct_output = 0;
    uint8_t optimized = 0;
    uint64_t upload_bytes = 0;
    uint64_t query_bytes = 0;
    uint64_t out_bytes = 0;
    uint64_t map_copy_upload_ns = 0;
    uint64_t map_copy_query_ns = 0;
    uint64_t record_upload_ns = 0;
    uint64_t record_attn_ns = 0;
    uint64_t readback_ns = 0;
};

struct BulkSessionProfiling {
    bool active = false;
    uint32_t segment_index = 0;
    std::vector<BulkSegmentProfile> segments;
};

} // namespace

struct RemoteAttnEngine::RangeProfile {
    rp::Recorder recorder;
    BulkSessionProfiling bulk;
};

// Per-session KV storage. Each remote layer has its own dedicated committed DEFAULT
// resource in `layer_kv`, preventing >2 GiB single-resource allocation failures
// and 32-bit offset overflow in ByteAddressBuffer at 128K-256K+ context.
// F16 path: [kind: 0=K,1=V][capacity][n_head_kv][head_dim].
// KVarN path (cache_bits 2..8): per layer slot the buffer is laid out as
// [records K][stage K][records V][stage V], where a record group covers 128
// positions (KVAR_N_GROUP) and the stage holds stage_groups*128 positions of
// F16 (group 0 sink + tail_groups live groups).
struct RemoteAttnEngine::Session {
    uint32_t id = 0;
    uint32_t capacity = 0;
    int32_t high_water = -1; // highest stored position; -1 = empty
    bool valid = true;
    bool range_configured = false;
    uint32_t range_storage = 0; // 0=F16, 1=Q4_0, 2=Q8_0
    uint32_t range_origin = 0;
    rkva_kvarn_range_config range_descriptor{};
    uint32_t precision_tail_tokens = 0;
    std::vector<uint32_t> range_record_groups;
    std::vector<bool> range_ready;
    // Revision-3 per-layer tail ring / stage coverage, published after the fence.
    std::vector<rkva_kvarn_range::prefix_coverage> prefix_coverage;
    uint64_t native_generation = 0;
    std::vector<uint32_t> range_counts; // per layer; never share a high-water mark

    std::vector<Buffer> layer_kv; // 1 committed buffer per layer slot
    uint64_t kv_bytes = 0;
    uint64_t layer_stride = 0; // bytes per (slot) covering K+V
    uint64_t half_stride = 0;  // bytes per K or V plane (F16 path)
    // KVarN path sizes (bits>8 none; cache_bits 2..8).
    uint64_t record_words = 0;  // per record group, u32 words
    uint64_t records_bytes = 0; // per layer per kind (K or V)
    uint64_t stage_bytes = 0;   // per layer per kind (K or V)
    uint64_t groups_per_stream = 0;
    uint64_t rec_off = 0;   // byte offset of records K within a layer slot
    uint64_t stg_off = 0;   // byte offset of stage K within a layer slot
    uint64_t rec_v_off = 0; // byte offset of records V within a layer slot
    uint64_t stg_v_off = 0; // byte offset of stage V within a layer slot
};

RemoteAttnEngine::RemoteAttnEngine() = default;

bool RemoteAttnEngine::confirm_execution_retired() {
    if (!unconfirmed_submission_)
        return true; // normal path: no extra D3D call
    rkva_execution_lifetime::Snapshot snapshot;
    snapshot.unconfirmed = true;
    snapshot.has_fence = bool(fence_);
    snapshot.has_device = bool(device_);
    snapshot.submitted = fence_value_;
    if (fence_)
        snapshot.completed = fence_->GetCompletedValue();
    if (device_)
        snapshot.device_removed = FAILED(device_->GetDeviceRemovedReason());
    if (!rkva_execution_lifetime::releasable(snapshot))
        return false;
    unconfirmed_submission_ = false;
    return true;
}

RemoteAttnEngine::~RemoteAttnEngine() {
    // The serial listener owner must retain an uncertain engine BEFORE delete.
    // A destructor cannot cancel deletion. If another caller violates ownership,
    // fail fast before any event/COM/resource member is released.
    if (!confirm_execution_retired()) {
        OutputDebugStringA("RKVA invariant: destroying engine with unconfirmed GPU work\n");
        std::terminate();
    }
    if (fence_event_) {
        CloseHandle(fence_event_);
        fence_event_ = nullptr;
    }
}

bool RemoteAttnEngine::fail(std::string* error, const char* stage, HRESULT hr) {
    ready_ = false;
    return ::fail(error, stage, hr);
}

bool RemoteAttnEngine::create_buffer(uint64_t bytes, D3D12_HEAP_TYPE heap,
                                     D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state,
                                     Buffer* out, std::string* error) {
    if (!device_ || !out || !bytes) {
        return fail(error, "invalid buffer size");
    }
    const HRESULT hr = try_create_buffer(bytes, heap, flags, state, out);
    if (FAILED(hr)) {
        return fail(error, "CreateCommittedResource", hr);
    }
    return true;
}

HRESULT RemoteAttnEngine::try_create_buffer(uint64_t bytes, D3D12_HEAP_TYPE heap,
                                            D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state,
                                            Buffer* out) {
    if (!device_ || !out || !bytes) {
        return E_INVALIDARG;
    }
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = heap;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = aligned4(bytes);
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = flags;
    ComPtr<ID3D12Resource> resource;
    const HRESULT hr = device_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, state,
                                                        nullptr, IID_PPV_ARGS(&resource));
    if (FAILED(hr)) {
        return hr;
    }
    out->resource = std::move(resource);
    out->bytes = rd.Width;
    out->state = state;
    if (profiling_ && profile_) {
        ++profile_->recorder.counters.buffer_creates;
        profile_->recorder.counters.buffer_create_bytes += rd.Width;
    }
    return S_OK;
}

void RemoteAttnEngine::transition(Buffer& buffer, D3D12_RESOURCE_STATES state) {
    if (buffer.state == state || !buffer.resource) {
        return;
    }
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = buffer.resource.Get();
    barrier.Transition.StateBefore = buffer.state;
    barrier.Transition.StateAfter = state;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmd_->ResourceBarrier(1, &barrier);
    buffer.state = state;
}

void RemoteAttnEngine::uav_barrier(Buffer& buffer) {
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = buffer.resource.Get();
    cmd_->ResourceBarrier(1, &barrier);
}

bool RemoteAttnEngine::begin(std::string* error) {
    if (bulk_recording_)
        return true;
    if (terminal_failure_ || unconfirmed_submission_) {
        return fail(error, "engine terminal after unconfirmed submission");
    }
    HRESULT hr = allocator_->Reset();
    if (FAILED(hr)) {
        return fail(error, "CommandAllocator Reset", hr);
    }
    hr = cmd_->Reset(allocator_.Get(), nullptr);
    if (FAILED(hr)) {
        return fail(error, "CommandList Reset", hr);
    }
    return true;
}

bool RemoteAttnEngine::submit(std::string* error) {
    if (bulk_recording_)
        return true;
    if (terminal_failure_ || unconfirmed_submission_) {
        return fail(error, "engine terminal after unconfirmed submission");
    }
    const bool prof = profiling_ && profile_;
    prof_clock::time_point exec_start;
    if (prof) {
        exec_start = prof_clock::now();
    }
    HRESULT hr = cmd_->Close();
    if (FAILED(hr)) {
        return fail(error, "CommandList Close", hr);
    }
    if (fence_value_ >= UINT64_MAX - 1) {
        terminal_failure_ = true;
        return fail(error, "GPU fence value exhausted");
    }
    ID3D12CommandList* lists[] = {cmd_.Get()};
    queue_->ExecuteCommandLists(1, lists);
    unconfirmed_submission_ = true;
    const UINT64 value = ++fence_value_;
    hr = queue_->Signal(fence_.Get(), value);
    if (FAILED(hr)) {
        terminal_failure_ = true;
        return fail(error, "CommandQueue Signal", hr);
    }
    const bool blocked = fence_->GetCompletedValue() < value;
    if (prof) {
        last_submit_exec_ns_ = elapsed_ns(exec_start, prof_clock::now());
        last_submit_blocked_ = blocked;
        rp::Counters& c = profile_->recorder.counters;
        ++c.submits;
        ++(blocked ? c.fence_blocking_waits : c.fence_already_complete);
    }
    if (blocked) {
        hr = fence_->SetEventOnCompletion(value, fence_event_);
        if (FAILED(hr)) {
            terminal_failure_ = true;
            return fail(error, "Fence SetEventOnCompletion", hr);
        }
        if (WaitForSingleObject(fence_event_, 60000) != WAIT_OBJECT_0) {
            terminal_failure_ = true;
            return fail(error, "GPU fence wait failed");
        }
    }
    const uint64_t completed = fence_->GetCompletedValue();
    hr = device_->GetDeviceRemovedReason();
    if (completed == UINT64_MAX) {
        terminal_failure_ = true;
        // MAX is device-removal, never an ordinary completed fence value.
        // Release is safe only when the device independently confirms removal.
        if (FAILED(hr))
            unconfirmed_submission_ = false;
        return fail(error, "GPU fence device-removal sentinel", hr);
    }
    if (completed < value) {
        terminal_failure_ = true;
        return fail(error, "GPU execution did not complete", hr);
    }
    unconfirmed_submission_ = false;
    if (FAILED(hr)) {
        terminal_failure_ = true;
        return fail(error, "D3D12 device removed", hr);
    }
    return true;
}

bool RemoteAttnEngine::create_device(std::string* error) {
    auto create_device_fn = xllama::d3d12_dyn::CreateDevice();
    if (!create_device_fn) {
        return fail(error, "D3D12CreateDevice unavailable");
    }
    ComPtr<IDXGIFactory4> factory;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) {
        return fail(error, "CreateDXGIFactory1", hr);
    }
    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> adapter;
        if (factory->EnumAdapters1(index, &adapter) == DXGI_ERROR_NOT_FOUND) {
            break;
        }
        DXGI_ADAPTER_DESC1 desc = {};
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
            continue;
        }
        hr = create_device_fn(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device_));
        if (SUCCEEDED(hr)) {
            break;
        }
    }
    if (!device_) {
        return fail(error, "no hardware D3D12 adapter");
    }
    D3D12_COMMAND_QUEUE_DESC queue_desc = {};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    hr = device_->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue_));
    if (FAILED(hr)) {
        return fail(error, "CreateCommandQueue", hr);
    }
    hr = device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator_));
    if (FAILED(hr)) {
        return fail(error, "CreateCommandAllocator", hr);
    }
    hr = device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator_.Get(), nullptr,
                                    IID_PPV_ARGS(&cmd_));
    if (FAILED(hr)) {
        return fail(error, "CreateCommandList", hr);
    }
    hr = cmd_->Close();
    if (FAILED(hr)) {
        return fail(error, "CommandList initial Close", hr);
    }
    hr = device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_));
    if (FAILED(hr)) {
        return fail(error, "CreateFence", hr);
    }
    fence_event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!fence_event_) {
        return fail(error, "CreateEventW failed");
    }
    return true;
}

bool RemoteAttnEngine::create_pipelines(std::string* error) {
    auto serialize = xllama::d3d12_dyn::SerializeRootSignature();
    if (!serialize) {
        return fail(error, "D3D12SerializeRootSignature unavailable");
    }
    auto make_root = [&](uint32_t n_const, uint32_t n_srv, uint32_t n_uav,
                         ID3D12RootSignature** out) -> bool {
        // Use a fixed-size array (max 8 params) to avoid vector pointer math.
        D3D12_ROOT_PARAMETER p[8] = {};
        uint32_t idx = 0;
        if (n_const) {
            p[idx].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
            p[idx].Constants.ShaderRegister = 0;
            p[idx].Constants.Num32BitValues = n_const;
            p[idx].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            ++idx;
        }
        for (uint32_t s = 0; s < n_srv; ++s) {
            p[idx].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
            p[idx].Descriptor.ShaderRegister = s;
            p[idx].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            ++idx;
        }
        for (uint32_t u = 0; u < n_uav; ++u) {
            p[idx].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
            p[idx].Descriptor.ShaderRegister = u;
            p[idx].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            ++idx;
        }
        D3D12_ROOT_SIGNATURE_DESC desc = {};
        desc.NumParameters = idx;
        desc.pParameters = p;
        ComPtr<ID3DBlob> blob;
        ComPtr<ID3DBlob> messages;
        HRESULT hr = serialize(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &messages);
        if (FAILED(hr)) {
            return fail(error, "SerializeRootSignature", hr);
        }
        hr = device_->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                          IID_PPV_ARGS(out));
        if (FAILED(hr)) {
            return fail(error, "CreateRootSignature", hr);
        }
        return true;
    };

    // Attention: 8 constants + SRV t0..t3 (4) + UAV u0 (1).
    if (!make_root(8, 4, 1, attn_root_.GetAddressOf())) {
        return false;
    }
    // Store: 4 constants + SRV t0..t1 (2) + UAV u0 (1).
    if (!make_root(4, 2, 1, store_root_.GetAddressOf())) {
        return false;
    }
    // KVarN WHT: 3 constants + SRV t0 (1) + UAV u0 (1).
    if (!make_root(3, 1, 1, wht_root_.GetAddressOf())) {
        return false;
    }
    // KVarN store: 11 constants + SRV t0..t1 (2) + UAV u0..u1 (2).
    if (!make_root(11, 2, 2, store_kvarn_root_.GetAddressOf())) {
        return false;
    }
    // KVarN attention: 14 constants (including split_k) + SRV t0..t5 (6) + UAV u0 (1).
    if (!make_root(14, 6, 1, attn_kvarn_root_.GetAddressOf())) {
        return false;
    }
    // Split-K reduction: 4 constants + SRV t0 (1) + UAV u0 (1).
    if (!make_root(4, 1, 1, split_k_reduce_root_.GetAddressOf())) {
        return false;
    }

    if (!make_root(11, 4, 1, range_attn_root_.GetAddressOf()) ||
        !make_root(4, 1, 1, range_reduce_root_.GetAddressOf())) {
        return false;
    }

    D3D12_COMPUTE_PIPELINE_STATE_DESC pso = {};
    pso.pRootSignature = attn_root_.Get();
    pso.CS = {kRkvaFlashAttnDxil, kRkvaFlashAttnDxilSize};
    HRESULT hr = device_->CreateComputePipelineState(&pso, IID_PPV_ARGS(&attn_decode_pso_));
    if (FAILED(hr)) {
        return fail(error, "CreateComputePipelineState attn", hr);
    }
    attn_prefill_pso_ = attn_decode_pso_; // same kernel; grid differs

    pso.pRootSignature = store_root_.Get();
    pso.CS = {kRkvaStoreF16Dxil, kRkvaStoreF16DxilSize};
    hr = device_->CreateComputePipelineState(&pso, IID_PPV_ARGS(&store_k_pso_));
    if (FAILED(hr)) {
        return fail(error, "CreateComputePipelineState store", hr);
    }
    store_v_pso_ = store_k_pso_;

    pso.pRootSignature = wht_root_.Get();
    pso.CS = {kRkvaWhtDxil, kRkvaWhtDxilSize};
    hr = device_->CreateComputePipelineState(&pso, IID_PPV_ARGS(&wht_pso_));
    if (FAILED(hr)) {
        return fail(error, "CreateComputePipelineState wht", hr);
    }

    pso.pRootSignature = store_kvarn_root_.Get();
    pso.CS = {kRkvaStoreKvarn4Dxil, kRkvaStoreKvarn4DxilSize};
    hr = device_->CreateComputePipelineState(&pso, IID_PPV_ARGS(&store_kvarn_pso_));
    if (FAILED(hr)) {
        return fail(error, "CreateComputePipelineState store kvarn", hr);
    }

    pso.pRootSignature = attn_kvarn_root_.Get();
    pso.CS = {kRkvaFlashAttnKvarn4Dxil, kRkvaFlashAttnKvarn4DxilSize};
    hr = device_->CreateComputePipelineState(&pso, IID_PPV_ARGS(&attn_kvarn_pso_));
    if (FAILED(hr)) {
        return fail(error, "CreateComputePipelineState attn kvarn", hr);
    }

    pso.pRootSignature = split_k_reduce_root_.Get();
    pso.CS = {kRkvaFlashAttnSplitKReduceDxil, kRkvaFlashAttnSplitKReduceDxilSize};
    hr = device_->CreateComputePipelineState(&pso, IID_PPV_ARGS(&split_k_reduce_pso_));
    if (FAILED(hr)) {
        return fail(error, "CreateComputePipelineState split_k reduce", hr);
    }
    pso.pRootSignature = range_attn_root_.Get();
    const D3D12_SHADER_BYTECODE range_shaders[] = {
        {g_rkva_range_attn_dxil, sizeof(g_rkva_range_attn_dxil)},
        {g_rkva_range_attn_q4_dxil, sizeof(g_rkva_range_attn_q4_dxil)},
        {g_rkva_range_attn_q8_dxil, sizeof(g_rkva_range_attn_q8_dxil)},
        {g_rkva_range_attn_kvarn4_dxil, sizeof(g_rkva_range_attn_kvarn4_dxil)},
    };
    for (uint32_t storage = 0; storage < 4; ++storage) {
        pso.CS = range_shaders[storage];
        hr = device_->CreateComputePipelineState(&pso, IID_PPV_ARGS(&range_attn_pso_[storage]));
        if (FAILED(hr)) {
            return fail(error, "CreateComputePipelineState range attention", hr);
        }
    }
    // Revision-3 serial-prefix reference kernel: optional, never fatal for the engine.
    pso.CS = {g_rkva_range_attn_kvarn4_prefix_dxil, sizeof(g_rkva_range_attn_kvarn4_prefix_dxil)};
    hr = device_->CreateComputePipelineState(&pso, IID_PPV_ARGS(&range_attn_kvarn4_prefix_pso_));
    if (FAILED(hr)) {
        char text[96];
        std::snprintf(text, sizeof(text), "kvarn4 serial-prefix pipeline state: 0x%08X",
                      unsigned(hr));
        kvarn4_prefix_status_ = text;
        range_attn_kvarn4_prefix_pso_.Reset();
    } else {
        kvarn4_prefix_status_ = "ready";
    }
    pso.pRootSignature = range_reduce_root_.Get();
    pso.CS = {g_rkva_range_reduce_dxil, sizeof(g_rkva_range_reduce_dxil)};
    hr = device_->CreateComputePipelineState(&pso, IID_PPV_ARGS(&range_reduce_pso_));
    if (FAILED(hr)) {
        return fail(error, "CreateComputePipelineState range reduction", hr);
    }
    return true;
}

// Optional optimized Q4_0 decode pipelines. Failure only disables modes 1..3;
// the reference range pipelines and engine readiness are unaffected.
void RemoteAttnEngine::create_q4_opt_pipelines() {
    D3D12_FEATURE_DATA_D3D12_OPTIONS1 options = {};
    if (FAILED(device_->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &options,
                                            sizeof(options))) ||
        !options.WaveOps) {
        q4_opt_status_ = "wave operations unsupported";
        return;
    }
    wave_lane_min_ = options.WaveLaneCountMin;
    wave_lane_max_ = options.WaveLaneCountMax;
    if (wave_lane_min_ < rp::kOptMinWaveLanes || wave_lane_max_ > rp::kOptMaxWaveLanes) {
        q4_opt_status_ = "wave lane count outside 32..128";
        return;
    }
    if (geo_.head_dim != rp::kOptHeadDim || geo_.n_head != rp::kOptGqa * geo_.n_head_kv) {
        q4_opt_status_ = "geometry is not head_dim 256 with 6 Q heads per KV head";
        return;
    }
    auto serialize = xllama::d3d12_dyn::SerializeRootSignature();
    if (!serialize) {
        q4_opt_status_ = "D3D12SerializeRootSignature unavailable";
        return;
    }
    // 10 constants, SRV t0..t3, UAV u0 (partials) and u1 (direct output).
    D3D12_ROOT_PARAMETER params[7] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.ShaderRegister = 0;
    params[0].Constants.Num32BitValues = 10;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    for (uint32_t i = 0; i < 4; ++i) {
        params[1 + i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        params[1 + i].Descriptor.ShaderRegister = i;
        params[1 + i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    for (uint32_t i = 0; i < 2; ++i) {
        params[5 + i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[5 + i].Descriptor.ShaderRegister = i;
        params[5 + i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    D3D12_ROOT_SIGNATURE_DESC desc = {};
    desc.NumParameters = 7;
    desc.pParameters = params;
    ComPtr<ID3DBlob> blob;
    ComPtr<ID3DBlob> messages;
    char text[96];
    HRESULT hr = serialize(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &messages);
    if (SUCCEEDED(hr)) {
        hr = device_->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                          IID_PPV_ARGS(&range_attn_q4_opt_root_));
    }
    if (FAILED(hr)) {
        std::snprintf(text, sizeof(text), "q4 optimized root signature: 0x%08X", unsigned(hr));
        q4_opt_status_ = text;
        range_attn_q4_opt_root_.Reset();
        return;
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC pso = {};
    pso.pRootSignature = range_attn_q4_opt_root_.Get();
    pso.CS = {g_rkva_range_attn_q4_opt_dxil, sizeof(g_rkva_range_attn_q4_opt_dxil)};
    hr = device_->CreateComputePipelineState(&pso, IID_PPV_ARGS(&range_attn_q4_opt_pso_));
    if (SUCCEEDED(hr)) {
        pso.pRootSignature = range_reduce_root_.Get();
        pso.CS = {g_rkva_range_reduce_opt_dxil, sizeof(g_rkva_range_reduce_opt_dxil)};
        hr = device_->CreateComputePipelineState(&pso, IID_PPV_ARGS(&range_reduce_opt_pso_));
    }
    if (SUCCEEDED(hr)) {
        pso.pRootSignature = range_reduce_root_.Get();
        pso.CS = {g_rkva_range_reduce_kvarn4_dxil, sizeof(g_rkva_range_reduce_kvarn4_dxil)};
        hr = device_->CreateComputePipelineState(&pso, IID_PPV_ARGS(&range_reduce_kvarn4_pso_));
    }
    if (FAILED(hr)) {
        std::snprintf(text, sizeof(text), "q4 optimized pipeline state: 0x%08X", unsigned(hr));
        q4_opt_status_ = text;
        range_attn_q4_opt_pso_.Reset();
        range_reduce_opt_pso_.Reset();
        range_reduce_kvarn4_pso_.Reset();
        return;
    }
    q4_opt_status_ = "ready";
    // Multi-query tile kernels share the per-query root signature and reducer.
    const D3D12_SHADER_BYTECODE mq_cs[2] = {
        {g_rkva_range_attn_q4_mq2_dxil, sizeof(g_rkva_range_attn_q4_mq2_dxil)},
        {g_rkva_range_attn_q4_mq4_dxil, sizeof(g_rkva_range_attn_q4_mq4_dxil)}};
    for (uint32_t i = 0; i < 2; ++i) {
        pso.pRootSignature = range_attn_q4_opt_root_.Get();
        pso.CS = mq_cs[i];
        hr = device_->CreateComputePipelineState(&pso, IID_PPV_ARGS(&range_attn_q4_mq_pso_[i]));
        if (FAILED(hr)) {
            std::snprintf(text, sizeof(text), "q4 multi-query pipeline state: 0x%08X",
                          unsigned(hr));
            q4_mq_status_[i] = text;
            range_attn_q4_mq_pso_[i].Reset();
        } else {
            q4_mq_status_[i] = "ready";
        }
    }
}

// Optional optimized F16 / Q8_0 / native KVarN4 decode pipelines. They share the
// Q4 prerequisites (wave ops, lanes, geometry, opt reducer and partials) and one
// 12-constant root signature; each PSO is created and reported on its own, and a
// missing PSO keeps that storage on its reference kernel.
void RemoteAttnEngine::create_format_opt_pipelines() {
    if (!range_attn_q4_opt_pso_ || !range_reduce_opt_pso_) {
        const std::string why = "requires the q4 optimized prerequisites: " + q4_opt_status_;
        fmt_opt_status_[rp::kStorageF16] = why;
        fmt_opt_status_[rp::kStorageQ8] = why;
        fmt_opt_status_[rp::kStorageKvarn4] = why;
        kvarn4_opt_prefix_status_ = why;
        kvarn4_mq_status_ = why;
        kvarn4_mq4_status_ = why;
        kvarn4_decode_prefix_status_ = why;
        for (uint32_t i = 0; i < 2; ++i) {
            fmt_mq_status_[rp::kStorageF16][i] = why;
            fmt_mq_status_[rp::kStorageQ8][i] = why;
        }
        return;
    }
    auto serialize = xllama::d3d12_dyn::SerializeRootSignature();
    // 12 constants (Q4 opt layout + records_bytes, precision_tail_tokens),
    // SRV t0..t3, UAV u0 (partials) and u1 (direct output).
    D3D12_ROOT_PARAMETER params[7] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.ShaderRegister = 0;
    params[0].Constants.Num32BitValues = 12;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    for (uint32_t i = 0; i < 4; ++i) {
        params[1 + i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        params[1 + i].Descriptor.ShaderRegister = i;
        params[1 + i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    for (uint32_t i = 0; i < 2; ++i) {
        params[5 + i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[5 + i].Descriptor.ShaderRegister = i;
        params[5 + i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    D3D12_ROOT_SIGNATURE_DESC desc = {};
    desc.NumParameters = 7;
    desc.pParameters = params;
    ComPtr<ID3DBlob> blob;
    ComPtr<ID3DBlob> messages;
    char text[96];
    HRESULT hr = serialize ? serialize(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &messages)
                           : E_NOINTERFACE;
    if (SUCCEEDED(hr)) {
        hr = device_->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                          IID_PPV_ARGS(&range_attn_fmt_opt_root_));
    }
    if (FAILED(hr)) {
        std::snprintf(text, sizeof(text), "format optimized root signature: 0x%08X", unsigned(hr));
        range_attn_fmt_opt_root_.Reset();
        fmt_opt_status_[rp::kStorageF16] = text;
        fmt_opt_status_[rp::kStorageQ8] = text;
        fmt_opt_status_[rp::kStorageKvarn4] = text;
        kvarn4_opt_prefix_status_ = text;
        kvarn4_mq_status_ = text;
        kvarn4_mq4_status_ = text;
        kvarn4_decode_prefix_status_ = text;
        for (uint32_t i = 0; i < 2; ++i) {
            fmt_mq_status_[rp::kStorageF16][i] = text;
            fmt_mq_status_[rp::kStorageQ8][i] = text;
        }
        return;
    }
    struct Kernel {
        uint32_t storage;
        const void* bytecode;
        size_t size;
    };
    const Kernel kernels[] = {
        {rp::kStorageF16, g_rkva_range_attn_f16_opt_dxil, sizeof(g_rkva_range_attn_f16_opt_dxil)},
        {rp::kStorageQ8, g_rkva_range_attn_q8_opt_dxil, sizeof(g_rkva_range_attn_q8_opt_dxil)},
        {rp::kStorageKvarn4, g_rkva_range_attn_kvarn4_opt_dxil,
         sizeof(g_rkva_range_attn_kvarn4_opt_dxil)},
    };
    for (const Kernel& k : kernels) {
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso = {};
        pso.pRootSignature = range_attn_fmt_opt_root_.Get();
        pso.CS = {k.bytecode, k.size};
        hr = device_->CreateComputePipelineState(&pso,
                                                 IID_PPV_ARGS(&range_attn_fmt_opt_pso_[k.storage]));
        if (FAILED(hr)) {
            std::snprintf(text, sizeof(text), "%s optimized pipeline state: 0x%08X",
                          rp::storage_name(k.storage), unsigned(hr));
            range_attn_fmt_opt_pso_[k.storage].Reset();
            fmt_opt_status_[k.storage] = text;
        } else {
            fmt_opt_status_[k.storage] = "ready";
        }
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC prefix = {};
    prefix.pRootSignature = range_attn_fmt_opt_root_.Get();
    prefix.CS = {g_rkva_range_attn_kvarn4_opt_prefix_dxil,
                 sizeof(g_rkva_range_attn_kvarn4_opt_prefix_dxil)};
    hr = device_->CreateComputePipelineState(&prefix,
                                             IID_PPV_ARGS(&range_attn_kvarn4_opt_prefix_pso_));
    if (FAILED(hr)) {
        std::snprintf(text, sizeof(text), "kvarn4 serial-prefix optimized pipeline state: 0x%08X",
                      unsigned(hr));
        range_attn_kvarn4_opt_prefix_pso_.Reset();
        kvarn4_opt_prefix_status_ = text;
    } else {
        kvarn4_opt_prefix_status_ = "ready";
    }
    // Native KVarN4 revision-3 2-query (mode 2) and 4-query (mode 1) tiles: same root signature;
    // one 4-dim unit per lane, so they are only created when every wave has at least 64 lanes.
    if (wave_lane_min_ < rp::kKvarn4MqWaveLanes) {
        std::snprintf(text, sizeof(text), "requires wave lanes >= %u (reported min %u)",
                      rp::kKvarn4MqWaveLanes, wave_lane_min_);
        kvarn4_mq_status_ = text;
        kvarn4_mq4_status_ = text;
    } else {
        D3D12_COMPUTE_PIPELINE_STATE_DESC mq = {};
        mq.pRootSignature = range_attn_fmt_opt_root_.Get();
        mq.CS = {g_rkva_range_attn_kvarn4_mq2_dxil, sizeof(g_rkva_range_attn_kvarn4_mq2_dxil)};
        hr = device_->CreateComputePipelineState(&mq, IID_PPV_ARGS(&range_attn_kvarn4_mq_pso_));
        if (FAILED(hr)) {
            std::snprintf(text, sizeof(text), "kvarn4 multi-query pipeline state: 0x%08X",
                          unsigned(hr));
            range_attn_kvarn4_mq_pso_.Reset();
            kvarn4_mq_status_ = text;
        } else {
            kvarn4_mq_status_ = "ready";
        }
#if RKVA_HAS_KVARN4_MQ4_DXIL
        mq.pRootSignature = range_attn_fmt_opt_root_.Get();
        mq.CS = {g_rkva_range_attn_kvarn4_mq4_dxil, sizeof(g_rkva_range_attn_kvarn4_mq4_dxil)};
        hr = device_->CreateComputePipelineState(&mq, IID_PPV_ARGS(&range_attn_kvarn4_mq4_pso_));
        if (FAILED(hr)) {
            std::snprintf(text, sizeof(text), "kvarn4 mq4 pipeline state: 0x%08X", unsigned(hr));
            range_attn_kvarn4_mq4_pso_.Reset();
            kvarn4_mq4_status_ = text;
        } else {
            kvarn4_mq4_status_ = "ready";
        }
#else
        kvarn4_mq4_status_ = "dxil header not generated (compile attn_kvarn4_mq4)";
#endif
    }
    // Native KVarN4 revision-3 wave64 decode kernel (nq == 1, 12 KiB LDS):
    // Requires wave lanes >= 64 and the generated decode DXIL.
    if (wave_lane_min_ < rp::kKvarn4DecodeWaveLanes) {
        std::snprintf(text, sizeof(text), "requires wave lanes >= %u (reported min %u)",
                      rp::kKvarn4DecodeWaveLanes, wave_lane_min_);
        kvarn4_decode_prefix_status_ = text;
    } else {
#if RKVA_HAS_KVARN4_DECODE_PREFIX_DXIL
        D3D12_COMPUTE_PIPELINE_STATE_DESC dec = {};
        dec.pRootSignature = range_attn_fmt_opt_root_.Get();
        dec.CS = {g_rkva_range_attn_kvarn4_decode_prefix_dxil,
                  sizeof(g_rkva_range_attn_kvarn4_decode_prefix_dxil)};
        hr = device_->CreateComputePipelineState(
            &dec, IID_PPV_ARGS(&range_attn_kvarn4_decode_prefix_pso_));
        if (FAILED(hr)) {
            std::snprintf(text, sizeof(text), "kvarn4 serial-prefix decode pipeline state: 0x%08X",
                          unsigned(hr));
            range_attn_kvarn4_decode_prefix_pso_.Reset();
            kvarn4_decode_prefix_status_ = text;
        } else {
            kvarn4_decode_prefix_status_ = "ready";
        }
#else
        kvarn4_decode_prefix_status_ =
            "header not built (run compile-rkva-range-shaders.sh attn_kvarn4_decode_prefix)";
#endif
    }
    // F16 / Q8_0 multi-query tiles: same root signature, one PSO per (storage, tile).
    const struct {
        uint32_t storage;
        uint32_t index; // q_tile / 4
        const void* bytecode;
        size_t size;
    } mq_kernels[] = {
        {rp::kStorageF16, 0, g_rkva_range_attn_f16_mq2_dxil,
         sizeof(g_rkva_range_attn_f16_mq2_dxil)},
        {rp::kStorageF16, 1, g_rkva_range_attn_f16_mq4_dxil,
         sizeof(g_rkva_range_attn_f16_mq4_dxil)},
        {rp::kStorageQ8, 0, g_rkva_range_attn_q8_mq2_dxil, sizeof(g_rkva_range_attn_q8_mq2_dxil)},
        {rp::kStorageQ8, 1, g_rkva_range_attn_q8_mq4_dxil, sizeof(g_rkva_range_attn_q8_mq4_dxil)},
    };
    for (const auto& k : mq_kernels) {
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso = {};
        pso.pRootSignature = range_attn_fmt_opt_root_.Get();
        pso.CS = {k.bytecode, k.size};
        ComPtr<ID3D12PipelineState>& slot = range_attn_fmt_mq_pso_[k.storage][k.index];
        hr = device_->CreateComputePipelineState(&pso, IID_PPV_ARGS(&slot));
        if (FAILED(hr)) {
            std::snprintf(text, sizeof(text), "%s multi-query pipeline state: 0x%08X",
                          rp::storage_name(k.storage), unsigned(hr));
            slot.Reset();
            fmt_mq_status_[k.storage][k.index] = text;
        } else {
            fmt_mq_status_[k.storage][k.index] = "ready";
        }
    }
}

// Timestamp heap + readback ring, created once from the control path. A failure
// leaves CPU span profiling available and reports gpu_timestamps=false.
bool RemoteAttnEngine::ensure_profile_resources(std::string* error) {
    if (timestamp_heap_ && timestamp_readback_.resource && timestamp_frequency_) {
        return true;
    }
    char text[96];
    UINT64 frequency = 0;
    HRESULT hr = queue_->GetTimestampFrequency(&frequency);
    if (FAILED(hr) || !frequency) {
        std::snprintf(text, sizeof(text), "GetTimestampFrequency: 0x%08X", unsigned(hr));
        timestamp_status_ = text;
        return ::fail(error, text);
    }
    D3D12_QUERY_HEAP_DESC heap_desc = {};
    heap_desc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    heap_desc.Count = kTsPerSet * kTsSets;
    ComPtr<ID3D12QueryHeap> heap;
    hr = device_->CreateQueryHeap(&heap_desc, IID_PPV_ARGS(&heap));
    if (FAILED(hr)) {
        std::snprintf(text, sizeof(text), "CreateQueryHeap timestamp: 0x%08X", unsigned(hr));
        timestamp_status_ = text;
        return ::fail(error, text);
    }
    Buffer readback;
    hr = try_create_buffer(uint64_t(kTsPerSet) * kTsSets * sizeof(uint64_t),
                           D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
                           D3D12_RESOURCE_STATE_COPY_DEST, &readback);
    if (FAILED(hr)) {
        std::snprintf(text, sizeof(text), "timestamp readback: 0x%08X", unsigned(hr));
        timestamp_status_ = text;
        return ::fail(error, text);
    }
    timestamp_frequency_ = frequency;
    timestamp_heap_ = std::move(heap);
    timestamp_readback_ = std::move(readback);
    timestamp_set_ = 0;
    timestamp_status_ = "ready";
    return true;
}

void RemoteAttnEngine::timestamp(uint32_t slot) {
    cmd_->EndQuery(timestamp_heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
                   timestamp_set_ * kTsPerSet + slot);
}

void RemoteAttnEngine::resolve_timestamps(uint32_t first, uint32_t count) {
    const uint32_t index = timestamp_set_ * kTsPerSet + first;
    cmd_->ResolveQueryData(timestamp_heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, index, count,
                           timestamp_readback_.resource.Get(), uint64_t(index) * sizeof(uint64_t));
    ++profile_->recorder.counters.timestamp_resolves;
}

// Called only after the submission's existing fence wait completed.
bool RemoteAttnEngine::read_timestamps(uint32_t first, uint32_t count, uint64_t* ticks) {
    const uint32_t index = timestamp_set_ * kTsPerSet + first;
    timestamp_set_ = (timestamp_set_ + 1) % kTsSets;
    D3D12_RANGE range{static_cast<SIZE_T>(index) * sizeof(uint64_t),
                      static_cast<SIZE_T>(index + count) * sizeof(uint64_t)};
    void* mapped = nullptr;
    if (FAILED(timestamp_readback_.resource->Map(0, &range, &mapped)) || !mapped) {
        ++profile_->recorder.counters.timestamp_map_failures;
        return false;
    }
    std::memcpy(ticks, static_cast<const uint8_t*>(mapped) + range.Begin, count * sizeof(uint64_t));
    D3D12_RANGE empty{0, 0};
    timestamp_readback_.resource->Unmap(0, &empty);
    for (uint32_t i = 1; i < count; ++i) {
        if (ticks[i] < ticks[i - 1]) {
            return false;
        }
    }
    return count > 0 && ticks[0] != 0;
}

void RemoteAttnEngine::finish_profile_sample(rp::Sample& sample, bool timestamps,
                                             uint32_t first_slot, uint32_t slot_count,
                                             uint32_t copy_regions, uint32_t dispatches) {
    const prof_clock::time_point start = prof_clock::now();
    // The fence span was measured around the whole submit() call; move its
    // Close/Execute/Signal part into the submit span so the spans stay disjoint.
    const uint64_t exec = std::min(last_submit_exec_ns_, sample.cpu_ns[rp::kCpuFenceWait]);
    sample.cpu_ns[rp::kCpuSubmit] += exec;
    sample.cpu_ns[rp::kCpuFenceWait] -= exec;
    sample.fence_blocked = last_submit_blocked_;
    if (timestamps) {
        uint64_t ticks[kTsPerSet] = {};
        if (slot_count >= 2 && slot_count <= rp::kGpuStageCount + 1 &&
            read_timestamps(first_slot, slot_count, ticks)) {
            for (uint32_t i = 1; i < slot_count; ++i) {
                sample.gpu_ticks[i - 1] = ticks[i] - ticks[i - 1];
            }
            sample.gpu_valid = 1;
        }
    }
    rp::Counters& c = profile_->recorder.counters;
    ++c.calls[sample.kind];
    c.copy_regions += copy_regions;
    c.dispatches += dispatches;
    if (sample.kind == rp::kAttention && sample.storage < rp::kStorageCount) {
        if (sample.optimized) {
            ++c.optimized_by_storage[sample.storage];
            if (sample.storage == rp::kStorageQ4) {
                ++c.optimized_calls;
                if (sample.q_tile == 2 || sample.q_tile == 4) {
                    ++c.q4_mq_calls[sample.q_tile / 4];
                }
            } else if ((sample.storage == rp::kStorageF16 || sample.storage == rp::kStorageQ8) &&
                       (sample.q_tile == 2 || sample.q_tile == 4)) {
                ++c.fmt_mq_calls[sample.storage][sample.q_tile / 4];
            } else if (sample.storage == rp::kStorageKvarn4 && sample.serial_prefix) {
                if (sample.decode_opt) {
                    ++c.kvarn4_decode_calls;
                } else if (sample.q_tile == rp::kKvarn4MqTile) {
                    ++c.kvarn4_mq_calls;
                } else if (sample.q_tile == rp::kKvarn4MqTileMax) {
                    ++c.kvarn4_mq4_calls;
                }
            }
        } else {
            ++c.reference_by_storage[sample.storage];
            if (sample.storage == rp::kStorageQ4) {
                ++c.reference_q4_calls;
            }
        }
    }
    sample.overhead_ns = elapsed_ns(start, prof_clock::now());
    profile_->recorder.record(sample);
}

bool RemoteAttnEngine::set_range_profiling(bool enabled, bool reset, uint32_t q4_kernel_mode,
                                           std::string& error) {
    if (!ready_ || !device_ || !queue_) {
        error = "range profiling: engine not ready";
        return false;
    }
    if (unconfirmed_submission_ || terminal_failure_) {
        error = "range profiling: engine has an unconfirmed or failed submission";
        return false;
    }
    if (!rp::valid_mode(q4_kernel_mode)) {
        error = "range profiling: invalid q4 kernel mode";
        return false;
    }
    if (q4_kernel_mode != rp::kModeReference) {
        if (!range_attn_q4_opt_pso_ || !range_reduce_opt_pso_) {
            error = "range profiling: optimized q4 kernel unavailable: " + q4_opt_status_;
            return false;
        }
        if (!range_opt_partials_.resource) {
            const uint64_t bytes = uint64_t(rp::kOptMaxSplits) * rp::kOptMaxQueries * geo_.n_head *
                                   (geo_.head_dim + 2) * sizeof(float);
            const HRESULT hr = try_create_buffer(
                bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &range_opt_partials_);
            if (FAILED(hr)) {
                char text[96];
                std::snprintf(text, sizeof(text), "range profiling: q4 partials alloc 0x%08X",
                              unsigned(hr));
                error = text;
                return false;
            }
        }
    }
    try {
        if (!profile_) {
            profile_ = std::make_unique<RangeProfile>();
        }
        if (enabled) {
            profile_->recorder.configure();
        }
    } catch (const std::bad_alloc&) {
        error = "range profiling: sample ring allocation failed";
        return false;
    }
    if (enabled) {
        std::string ignored; // GPU timestamps are optional; status is in the JSON
        ensure_profile_resources(&ignored);
    }
    if (reset) {
        profile_->recorder.reset();
        bulk_scratch_.reset_counters();
    }
    profiling_ = enabled;
    q4_kernel_mode_ = q4_kernel_mode;
    return true;
}

static rp::MemoryInfo rkva_memory_info(ID3D12Device* device) noexcept {
    rp::MemoryInfo info;
    info.timestamp_unix_ms =
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::system_clock::now().time_since_epoch())
                                  .count());
    const HRESULT ro = RoInitialize(RO_INIT_MULTITHREADED);
    try {
        info.app_usage_bytes = winrt::Windows::System::MemoryManager::AppMemoryUsage();
        info.app_limit_bytes = winrt::Windows::System::MemoryManager::AppMemoryUsageLimit();
        info.app_available = true;
    } catch (...) { /* Optional observation must not prevent resource cleanup. */
    }
    if (SUCCEEDED(ro))
        RoUninitialize();
    if (device) {
        Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
        if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
            const LUID wanted = device->GetAdapterLuid();
            for (UINT index = 0;; ++index) {
                Microsoft::WRL::ComPtr<IDXGIAdapter1> candidate;
                if (FAILED(factory->EnumAdapters1(index, &candidate)))
                    break;
                DXGI_ADAPTER_DESC1 desc = {};
                if (FAILED(candidate->GetDesc1(&desc)) ||
                    desc.AdapterLuid.LowPart != wanted.LowPart ||
                    desc.AdapterLuid.HighPart != wanted.HighPart)
                    continue;
                Microsoft::WRL::ComPtr<IDXGIAdapter3> adapter;
                DXGI_QUERY_VIDEO_MEMORY_INFO memory = {};
                if (SUCCEEDED(candidate.As(&adapter)) &&
                    SUCCEEDED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
                                                            &memory))) {
                    info.gpu_local_available = true;
                    info.gpu_local_usage_bytes = memory.CurrentUsage;
                    info.gpu_local_budget_bytes = memory.Budget;
                }
                break;
            }
        }
    }

    return info;
}

std::string RemoteAttnEngine::get_memory_profile_json() {
    return rp::memory_json(rkva_memory_info(device_.Get()));
}

std::string RemoteAttnEngine::get_range_profile_json() {
    rp::Header header;
    header.memory = rkva_memory_info(device_.Get());
    header.profiling = profiling_;
    header.q4_kernel_mode = q4_kernel_mode_;
    header.gpu_timestamps = timestamp_heap_ && timestamp_frequency_;
    header.gpu_timestamp_frequency = timestamp_frequency_;
    header.gpu_timestamp_status = timestamp_status_;
    header.q4_optimized_available = range_attn_q4_opt_pso_ && range_reduce_opt_pso_;
    header.q4_optimized_status = q4_opt_status_;
    for (uint32_t st = 0; st < rp::kStorageCount; ++st) {
        header.optimized_available[st] =
            st == rp::kStorageQ4
                ? header.q4_optimized_available
                : range_attn_fmt_opt_root_ && range_attn_fmt_opt_pso_[st] && range_reduce_opt_pso_;
        header.optimized_status[st] = st == rp::kStorageQ4 ? q4_opt_status_ : fmt_opt_status_[st];
    }
    header.wave_lane_min = wave_lane_min_;
    header.wave_lane_max = wave_lane_max_;
    header.kvarn4_prefix_available[0] = bool(range_attn_kvarn4_prefix_pso_);
    header.kvarn4_prefix_status[0] = kvarn4_prefix_status_;
    header.kvarn4_prefix_available[1] =
        range_attn_fmt_opt_root_ && range_attn_kvarn4_opt_prefix_pso_ && range_reduce_kvarn4_pso_;
    header.kvarn4_prefix_status[1] = kvarn4_opt_prefix_status_;
    header.kvarn4_mq_available = header.kvarn4_prefix_available[1] && range_attn_kvarn4_mq_pso_;
    header.kvarn4_mq_status =
        header.kvarn4_prefix_available[1]
            ? kvarn4_mq_status_
            : "requires the kvarn4 serial-prefix optimized pipeline: " + kvarn4_opt_prefix_status_;
    header.kvarn4_mq4_available = header.kvarn4_prefix_available[1] && range_attn_kvarn4_mq4_pso_;
    header.kvarn4_mq4_status =
        header.kvarn4_prefix_available[1]
            ? kvarn4_mq4_status_
            : "requires the kvarn4 serial-prefix optimized pipeline: " + kvarn4_opt_prefix_status_;
    header.kvarn4_decode_available =
        header.kvarn4_prefix_available[1] && range_attn_kvarn4_decode_prefix_pso_;
    header.kvarn4_decode_status =
        header.kvarn4_prefix_available[1]
            ? kvarn4_decode_prefix_status_
            : "requires the kvarn4 serial-prefix optimized pipeline: " + kvarn4_opt_prefix_status_;
    for (uint32_t i = 0; i < 2; ++i) {
        header.q4_mq_available[i] = header.q4_optimized_available && range_attn_q4_mq_pso_[i];
        header.q4_mq_status[i] = header.q4_optimized_available
                                     ? q4_mq_status_[i]
                                     : "requires the q4 optimized pipelines: " + q4_opt_status_;
    }
    for (uint32_t st : {uint32_t(rp::kStorageF16), uint32_t(rp::kStorageQ8)}) {
        for (uint32_t i = 0; i < 2; ++i) {
            header.fmt_mq_available[st][i] =
                header.optimized_available[st] && range_attn_fmt_mq_pso_[st][i];
            header.fmt_mq_status[st][i] = header.optimized_available[st]
                                              ? fmt_mq_status_[st][i]
                                              : std::string("requires the ") +
                                                    rp::storage_name(st) +
                                                    " optimized pipeline: " + fmt_opt_status_[st];
        }
    }
    std::string json = profile_ ? profile_->recorder.json(header) : rp::Recorder().json(header);
    json.pop_back();
    return json + ",\"bulk_scratch\":" + bulk_scratch_.json() +
           ",\"bulk_fp8_gpu\":{\"available\":" +
           (bulk_unpack_q_fp8_pso_ && bulk_pack_out_fp8_pso_ ? "true" : "false") +
           ",\"status\":\"" + rp::json_escape(bulk_fp8_status_) + "\"}}";
}

bool RemoteAttnEngine::initialize(const Geometry& geo, std::string* error) {
    if (ready_) {
        return fail(error, "engine already initialized");
    }
    if (geo.n_head == 0 || geo.n_head_kv == 0 || geo.head_dim == 0 || geo.max_ctx == 0 ||
        geo.n_head % geo.n_head_kv != 0) {
        return fail(error, "invalid geometry");
    }
    geo_ = geo;
    if (!create_device(error) || !create_pipelines(error)) {
        return false;
    }
    create_q4_opt_pipelines();
    create_format_opt_pipelines();
    // Initial scratch for a modest prefill chunk; grown on demand.
    scratch_tokens_ = 0;
    ready_ = true;
    return true;
}

RemoteAttnEngine::Session* RemoteAttnEngine::find_session(uint32_t session_id) {
    for (auto& s : sessions_) {
        if (s && s->id == session_id) {
            return s.get();
        }
    }
    return nullptr;
}

const RemoteAttnEngine::Session* RemoteAttnEngine::find_session(uint32_t session_id) const {
    for (const auto& s : sessions_) {
        if (s && s->id == session_id) {
            return s.get();
        }
    }
    return nullptr;
}

bool RemoteAttnEngine::create_session(uint32_t session_id, uint32_t capacity, std::string* error,
                                      bool defer_allocation) {
    if (!ready_) {
        return fail(error, "engine not ready");
    }
    if (find_session(session_id)) {
        return true; // idempotent
    }
    const uint32_t cap = capacity ? capacity : geo_.max_ctx;
    if (!cap || cap > geo_.max_ctx) {
        return ::fail(error, "session capacity exceeds negotiated maximum");
    }
    auto session = std::make_unique<Session>();
    session->id = session_id;
    session->capacity = cap;
    const uint32_t n_layers = geo_.n_layers ? geo_.n_layers : 1;
    const bool kvarn = geo_.cache_bits_k >= 2 && geo_.cache_bits_k <= 8;
    if (kvarn) {
        // KVarN4 layout per layer slot:
        //   records K: groups_per_stream * n_heads * record_words * 4
        //   stage   K: stage_groups*128 * n_heads * 64 * 4  (F16 packed pairs)
        //   records V / stage V: same sizes
        const uint32_t bits = geo_.cache_bits_k;
        const uint32_t n_heads = geo_.n_head_kv * geo_.head_slices;
        session->groups_per_stream = (uint64_t)(cap + 127) / 128;
        session->record_words = (uint64_t)bits * 512 + 192;
        session->records_bytes = session->groups_per_stream * n_heads * session->record_words * 4;
        session->stage_bytes = (uint64_t)geo_.stage_groups * 128 * n_heads * 64 * 4;
        session->rec_off = 0;
        session->stg_off = session->records_bytes;
        session->rec_v_off = session->stg_off + session->stage_bytes;
        session->stg_v_off = session->rec_v_off + session->records_bytes;
        session->layer_stride = session->stg_v_off + session->stage_bytes;
        session->half_stride = 0;
    } else {
        session->half_stride = uint64_t(cap) * geo_.n_head_kv * geo_.head_dim * 2; // F16
        session->layer_stride = session->half_stride * 2;                          // K + V
    }
    session->kv_bytes = defer_allocation ? 0 : session->layer_stride * n_layers;
    // Keep room for command/scratch/runtime resources within the advertised budget.
    constexpr uint64_t kv_budget = 4ull * 1024 * 1024 * 1024;
    if (session->kv_bytes > kv_budget || allocated_bytes_ > kv_budget - session->kv_bytes) {
        return ::fail(error, "session KV exceeds budget");
    }
    session->range_counts.resize(n_layers, 0);
    session->range_record_groups.resize(n_layers, 0);
    session->range_ready.resize(n_layers, false);
    session->prefix_coverage.resize(n_layers);
    session->layer_kv.resize(n_layers);
    for (uint32_t l = 0; !defer_allocation && l < n_layers; ++l) {
        if (!create_buffer(session->layer_stride, D3D12_HEAP_TYPE_DEFAULT,
                           D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &session->layer_kv[l], error)) {
            session->layer_kv.clear();
            return false;
        }
    }
    allocated_bytes_ += session->kv_bytes;
    sessions_.push_back(std::move(session));
    return true;
}

bool RemoteAttnEngine::destroy_session(uint32_t session_id) {
    if (unconfirmed_submission_ || bulk_recording_)
        return false;
    for (auto it = sessions_.begin(); it != sessions_.end(); ++it) {
        if (*it && (*it)->id == session_id) {
            allocated_bytes_ -= (*it)->kv_bytes;
            (*it)->layer_kv.clear();
            sessions_.erase(it);
            return true;
        }
    }
    return false;
}

bool RemoteAttnEngine::has_session(uint32_t session_id) const {
    return find_session(session_id) != nullptr;
}

bool RemoteAttnEngine::session_invalid(uint32_t session_id) const {
    const Session* s = find_session(session_id);
    return s ? !s->valid : true;
}

bool RemoteAttnEngine::reset_session(uint32_t session_id, std::string* error) {
    if (unconfirmed_submission_ || bulk_recording_)
        return ::fail(error, "reset: GPU execution unconfirmed/recording");
    Session* s = find_session(session_id);
    if (!s) {
        return fail(error, "reset: no session");
    }
    if (s->range_storage == 3 && !rkva_kvarn_generation_mutable(s->native_generation))
        return fail(error, "native generation exhausted");
    if (s->range_storage == 3) {
        ++s->native_generation;
    }
    s->high_water = -1;
    std::fill(s->range_counts.begin(), s->range_counts.end(), 0);
    std::fill(s->range_record_groups.begin(), s->range_record_groups.end(), 0);
    std::fill(s->range_ready.begin(), s->range_ready.end(), false);
    std::fill(s->prefix_coverage.begin(), s->prefix_coverage.end(),
              rkva_kvarn_range::prefix_coverage{});
    s->valid = true;
    return true;
}

bool RemoteAttnEngine::trim_session(uint32_t session_id, int32_t pos0, std::string* error) {
    Session* s = find_session(session_id);
    if (!s) {
        return fail(error, "trim: no session");
    }
    if (pos0 < 0) {
        return fail(error, "trim: negative position");
    }
    if (s->range_configured && s->range_storage == 3) {
        if (!rkva_kvarn_generation_mutable(s->native_generation))
            return fail(error, "native generation exhausted");
        ++s->native_generation;
        // Stage slots and the exact-tail overlay cannot be reconstructed after a rewind.
        // Invalidate every layer; require a complete host snapshot before any attention.
        std::fill(s->range_counts.begin(), s->range_counts.end(), 0);
        std::fill(s->range_record_groups.begin(), s->range_record_groups.end(), 0);
        std::fill(s->range_ready.begin(), s->range_ready.end(), false);
        std::fill(s->prefix_coverage.begin(), s->prefix_coverage.end(),
                  rkva_kvarn_range::prefix_coverage{});
    } else if (s->range_configured) {
        const uint32_t keep = uint32_t(pos0) <= s->range_origin
                                  ? 0
                                  : std::min(uint32_t(pos0) - s->range_origin, s->capacity);
        for (auto& count : s->range_counts) {
            count = std::min(count, keep);
        }
    }
    if (s->high_water >= pos0) {
        s->high_water = pos0 - 1;
    }
    return true;
}

// Ensure upload/out/readback scratch fits n_tokens. Grow-only.
bool RemoteAttnEngine::ensure_scratch(uint32_t n_tokens, std::string* error) {
    if (n_tokens <= scratch_tokens_ && qkv_upload_.resource) {
        return true;
    }
    const uint32_t tokens = std::max(n_tokens, std::max(scratch_tokens_, 256u));
    const uint64_t q_bytes = uint64_t(tokens) * geo_.n_head * geo_.head_dim * 4;
    const uint64_t kv_bytes = uint64_t(tokens) * geo_.n_head_kv * geo_.head_dim * 4;
    const uint64_t pos_bytes = uint64_t(tokens) * 4;
    const uint64_t upload_bytes = q_bytes + 2 * kv_bytes + pos_bytes;
    const uint64_t out_bytes = q_bytes;

    qkv_upload_ = Buffer{};
    wht_workspace_ = Buffer{}; // DEFAULT UAV: attention output before readback
    split_accum_ = Buffer{};   // DEFAULT UAV: split-K partial accumulators
    q_rotated_ = Buffer{};     // DEFAULT UAV: WHT(Q) input / inverse-WHT output
    out_readback_ = Buffer{};  // READBACK: CPU-visible copy of the output

    if (!create_buffer(upload_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                       D3D12_RESOURCE_STATE_GENERIC_READ, &qkv_upload_, error)) {
        return false;
    }
    if (!create_buffer(out_bytes, D3D12_HEAP_TYPE_DEFAULT,
                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &wht_workspace_, error)) {
        return false;
    }
    const uint64_t split_accum_bytes =
        32ull * out_bytes + 32ull * (uint64_t(tokens) * geo_.n_head * 2 * 4);
    if (!create_buffer(split_accum_bytes, D3D12_HEAP_TYPE_DEFAULT,
                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &split_accum_, error)) {
        return false;
    }
    if (!create_buffer(out_bytes, D3D12_HEAP_TYPE_DEFAULT,
                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &q_rotated_, error)) {
        return false;
    }
    if (!create_buffer(out_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
                       D3D12_RESOURCE_STATE_COPY_DEST, &out_readback_, error)) {
        return false;
    }
    scratch_tokens_ = tokens;
    return true;
}

bool RemoteAttnEngine::attend(uint32_t session_id, uint32_t layer_abs, uint32_t n_tokens,
                              const int32_t* positions, const float* q, const float* k,
                              const float* v, float* out, std::string* error) {
    if (!ready_) {
        return fail(error, "engine not ready");
    }
    Session* s = find_session(session_id);
    if (!s || !s->valid) {
        return fail(error, "attend: invalid session");
    }
    if (n_tokens == 0 || !positions || !q || !k || !v || !out) {
        return fail(error, "attend: bad arguments");
    }

    // Assign a dense KV slot for this absolute layer index.
    uint32_t slot;
    auto it = layer_slot_.find(layer_abs);
    if (it == layer_slot_.end()) {
        slot = (uint32_t)layer_slot_.size();
        if (geo_.n_layers && slot >= geo_.n_layers) {
            return fail(error, "attend: too many remote layers");
        }
        layer_slot_[layer_abs] = slot;
    } else {
        slot = it->second;
    }

    // Validate positions are within capacity.
    int32_t max_pos = s->high_water;
    for (uint32_t t = 0; t < n_tokens; ++t) {
        if (positions[t] < 0 || (uint32_t)positions[t] >= s->capacity) {
            return fail(error, "attend: position out of range");
        }
        max_pos = std::max(max_pos, positions[t]);
    }

    if (!ensure_scratch(n_tokens, error)) {
        return false;
    }

    const uint64_t q_bytes = uint64_t(n_tokens) * geo_.n_head * geo_.head_dim * 4;
    const uint64_t kv_bytes = uint64_t(n_tokens) * geo_.n_head_kv * geo_.head_dim * 4;
    const uint64_t q_off = 0;
    const uint64_t k_off = q_off + q_bytes;
    const uint64_t v_off = k_off + kv_bytes;
    const uint64_t pos_off = v_off + kv_bytes;

    if (!begin(error)) {
        return false;
    }

    // Upload Q/K/V/positions into the persistent UPLOAD buffer.
    {
        void* mapped = nullptr;
        HRESULT hr = qkv_upload_.resource->Map(0, nullptr, &mapped);
        if (FAILED(hr)) {
            return fail(error, "Map upload", hr);
        }
        auto* base = static_cast<uint8_t*>(mapped);
        std::memcpy(base + q_off, q, q_bytes);
        std::memcpy(base + k_off, k, kv_bytes);
        std::memcpy(base + v_off, v, kv_bytes);
        std::memcpy(base + pos_off, positions, uint64_t(n_tokens) * 4);
        qkv_upload_.resource->Unmap(0, nullptr);
    }

    if (slot >= s->layer_kv.size()) {
        return fail(error, "slot out of range for session layers");
    }
    Buffer& layer_buf = s->layer_kv[slot];
    const D3D12_GPU_VIRTUAL_ADDRESS upload_addr = qkv_upload_.resource->GetGPUVirtualAddress();
    const uint64_t k_plane = layer_buf.resource->GetGPUVirtualAddress();
    const uint64_t v_plane = k_plane + s->half_stride;

    // Ensure layer KV is in UAV state for the store.
    transition(layer_buf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    s->high_water = max_pos;
    const uint32_t active_count = (uint32_t)(s->high_water + 1);

    const bool kvarn = geo_.cache_bits_k >= 2 && geo_.cache_bits_k <= 8;
    Buffer* readback_src = &wht_workspace_;
    if (kvarn) {
        // KVarN path: WHT(Q) -> store K/V (records + stage) -> attention over
        // records -> inverse WHT(out). All buffers are DEFAULT UAV; SRV reads
        // happen after UAV barriers (the data is already ordered).
        const uint64_t rec_k = k_plane + s->rec_off;
        const uint64_t stg_k = k_plane + s->stg_off;
        const uint64_t rec_v = k_plane + s->rec_v_off;
        const uint64_t stg_v = k_plane + s->stg_v_off;
        const uint32_t n_heads_phys = geo_.n_head_kv * geo_.head_slices;
        const uint32_t gqa = geo_.n_head / geo_.n_head_kv;
        const uint32_t gqa_group_size = std::min(gqa, 4u);
        const uint32_t gqa_groups = (gqa + gqa_group_size - 1) / gqa_group_size;

        // 1) Rotate Q: rows = n_tokens * n_head, F32 in -> F32 out (q_rotated_).
        transition(q_rotated_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cmd_->SetComputeRootSignature(wht_root_.Get());
        cmd_->SetPipelineState(wht_pso_.Get());
        cmd_->SetComputeRoot32BitConstant(0, n_tokens * geo_.n_head, 0);
        cmd_->SetComputeRoot32BitConstant(0, geo_.head_dim, 1);
        cmd_->SetComputeRoot32BitConstant(0, 0u, 2);
        cmd_->SetComputeRootShaderResourceView(1, upload_addr + q_off);
        cmd_->SetComputeRootUnorderedAccessView(2, q_rotated_.resource->GetGPUVirtualAddress());
        cmd_->Dispatch(n_tokens * geo_.n_head, 1, 1);
        uav_barrier(q_rotated_);

        // 2) Store K and V (separate dispatches; `value` flips orientation).
        cmd_->SetComputeRootSignature(store_kvarn_root_.Get());
        cmd_->SetPipelineState(store_kvarn_pso_.Get());
        for (uint32_t pass = 0; pass < 2; ++pass) {
            const bool is_v = pass == 1;
            cmd_->SetComputeRoot32BitConstant(0, n_heads_phys, 0);
            cmd_->SetComputeRoot32BitConstant(0, n_tokens, 1);
            cmd_->SetComputeRoot32BitConstant(0, (uint32_t)s->groups_per_stream, 2);
            cmd_->SetComputeRoot32BitConstant(0, (uint32_t)s->record_words, 3);
            cmd_->SetComputeRoot32BitConstant(0, geo_.cache_bits_k, 4);
            cmd_->SetComputeRoot32BitConstant(0, geo_.sinkhorn_iters, 5);
            cmd_->SetComputeRoot32BitConstant(0, is_v ? 1u : 0u, 6);
            cmd_->SetComputeRoot32BitConstant(0, geo_.head_slices, 7);
            cmd_->SetComputeRoot32BitConstant(0, geo_.stage_groups, 8);
            cmd_->SetComputeRoot32BitConstant(0, geo_.tail_groups, 9);
            cmd_->SetComputeRoot32BitConstant(0, geo_.head_dim, 10);
            cmd_->SetComputeRootShaderResourceView(1, upload_addr + pos_off);
            cmd_->SetComputeRootShaderResourceView(2, upload_addr + (is_v ? v_off : k_off));
            cmd_->SetComputeRootUnorderedAccessView(3, is_v ? stg_v : stg_k);
            cmd_->SetComputeRootUnorderedAccessView(4, is_v ? rec_v : rec_k);
            cmd_->Dispatch(geo_.n_head_kv, 1, 1);
            uav_barrier(layer_buf);
        }

        // 3) Attention over records+stage (rotated domain). Records/stage and
        // the rotated Q are read as SRVs: transition out of UAV first.
        transition(layer_buf, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(q_rotated_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        // Split-K parallelization on Xbox Series X 52 CUs:
        uint32_t split_k = 1;
        if (n_tokens == 1 && active_count > 2048) {
            split_k = (active_count >= 32768) ? 32 : 16;
        }

        Buffer* attn_target = (split_k > 1) ? &split_accum_ : &wht_workspace_;
        transition(*attn_target, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        cmd_->SetComputeRootSignature(attn_kvarn_root_.Get());
        cmd_->SetPipelineState(attn_kvarn_pso_.Get());
        cmd_->SetComputeRoot32BitConstant(0, geo_.n_head, 0);
        cmd_->SetComputeRoot32BitConstant(0, geo_.n_head_kv, 1);
        cmd_->SetComputeRoot32BitConstant(0, n_heads_phys, 2);
        cmd_->SetComputeRoot32BitConstant(0, geo_.head_dim, 3);
        cmd_->SetComputeRoot32BitConstant(0, n_tokens, 4);
        cmd_->SetComputeRoot32BitConstant(0, active_count, 5);
        cmd_->SetComputeRoot32BitConstant(0, (uint32_t)s->groups_per_stream, 6);
        cmd_->SetComputeRoot32BitConstant(0, geo_.cache_bits_k, 7);
        cmd_->SetComputeRoot32BitConstant(0, geo_.head_slices, 8);
        cmd_->SetComputeRoot32BitConstant(0, geo_.stage_groups, 9);
        cmd_->SetComputeRoot32BitConstant(0, geo_.tail_groups, 10);
        cmd_->SetComputeRoot32BitConstant(0, (uint32_t)s->record_words, 11);
        cmd_->SetComputeRoot32BitConstant(0, *(const uint32_t*)&geo_.kq_scale, 12);
        cmd_->SetComputeRoot32BitConstant(0, split_k, 13);
        cmd_->SetComputeRootShaderResourceView(1, q_rotated_.resource->GetGPUVirtualAddress());
        cmd_->SetComputeRootShaderResourceView(2, rec_k);
        cmd_->SetComputeRootShaderResourceView(3, stg_k);
        cmd_->SetComputeRootShaderResourceView(4, rec_v);
        cmd_->SetComputeRootShaderResourceView(5, stg_v);
        cmd_->SetComputeRootShaderResourceView(6, upload_addr + pos_off);
        cmd_->SetComputeRootUnorderedAccessView(7, attn_target->resource->GetGPUVirtualAddress());
        cmd_->Dispatch(n_tokens, geo_.n_head_kv * gqa_groups, split_k);
        uav_barrier(*attn_target);

        if (split_k > 1) {
            transition(split_accum_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            transition(wht_workspace_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            cmd_->SetComputeRootSignature(split_k_reduce_root_.Get());
            cmd_->SetPipelineState(split_k_reduce_pso_.Get());
            cmd_->SetComputeRoot32BitConstant(0, n_tokens, 0);
            cmd_->SetComputeRoot32BitConstant(0, geo_.n_head, 1);
            cmd_->SetComputeRoot32BitConstant(0, geo_.head_dim, 2);
            cmd_->SetComputeRoot32BitConstant(0, split_k, 3);
            cmd_->SetComputeRootShaderResourceView(1,
                                                   split_accum_.resource->GetGPUVirtualAddress());
            cmd_->SetComputeRootUnorderedAccessView(
                2, wht_workspace_.resource->GetGPUVirtualAddress());
            cmd_->Dispatch(n_tokens, geo_.n_head, 1);
            uav_barrier(wht_workspace_);
            transition(split_accum_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }

        // 4) Inverse WHT on the output (rotated -> original domain); the WHT is
        // self-inverse, so the same kernel rotates out back into q_rotated_.
        transition(wht_workspace_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(q_rotated_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cmd_->SetComputeRootSignature(wht_root_.Get());
        cmd_->SetPipelineState(wht_pso_.Get());
        cmd_->SetComputeRoot32BitConstant(0, n_tokens * geo_.n_head, 0);
        cmd_->SetComputeRoot32BitConstant(0, geo_.head_dim, 1);
        cmd_->SetComputeRoot32BitConstant(0, 0u, 2);
        cmd_->SetComputeRootShaderResourceView(1, wht_workspace_.resource->GetGPUVirtualAddress());
        cmd_->SetComputeRootUnorderedAccessView(2, q_rotated_.resource->GetGPUVirtualAddress());
        cmd_->Dispatch(n_tokens * geo_.n_head, 1, 1);
        uav_barrier(q_rotated_);
        readback_src = &q_rotated_;
    } else {
        // Store K and V (F32 upload -> packed F16 KV at positions). The store
        // kernel packs two F16 dims per thread, so the element count is halved.
        const uint32_t store_pairs = n_tokens * geo_.n_head_kv * (geo_.head_dim / 2);
        const uint32_t store_groups = (store_pairs + 255) / 256;
        {
            cmd_->SetComputeRootSignature(store_root_.Get());
            cmd_->SetPipelineState(store_k_pso_.Get());
            cmd_->SetComputeRoot32BitConstant(0, n_tokens, 0);
            cmd_->SetComputeRoot32BitConstant(0, geo_.n_head_kv, 1);
            cmd_->SetComputeRoot32BitConstant(0, geo_.head_dim, 2);
            cmd_->SetComputeRoot32BitConstant(0, s->capacity, 3);
            cmd_->SetComputeRootShaderResourceView(1, upload_addr + pos_off);
            cmd_->SetComputeRootShaderResourceView(2, upload_addr + k_off);
            cmd_->SetComputeRootUnorderedAccessView(3, k_plane);
            cmd_->Dispatch(store_groups, 1, 1);
            uav_barrier(layer_buf);

            cmd_->SetComputeRootShaderResourceView(2, upload_addr + v_off);
            cmd_->SetComputeRootUnorderedAccessView(3, v_plane);
            cmd_->Dispatch(store_groups, 1, 1);
            uav_barrier(layer_buf);
        }

        // Attention: KV transitions to SRV; out buffer (wht_workspace_) is UAV.
        transition(layer_buf, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(wht_workspace_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        {
            cmd_->SetComputeRootSignature(attn_root_.Get());
            cmd_->SetPipelineState(n_tokens == 1 ? attn_decode_pso_.Get()
                                                 : attn_prefill_pso_.Get());
            cmd_->SetComputeRoot32BitConstant(0, n_tokens, 0);     // query_count
            cmd_->SetComputeRoot32BitConstant(0, active_count, 1); // active_count
            cmd_->SetComputeRoot32BitConstant(0, geo_.n_head, 2);
            cmd_->SetComputeRoot32BitConstant(0, geo_.n_head_kv, 3);
            cmd_->SetComputeRoot32BitConstant(0, geo_.head_dim, 4);
            cmd_->SetComputeRoot32BitConstant(0, *(const uint32_t*)&geo_.kq_scale, 5);
            cmd_->SetComputeRoot32BitConstant(0, s->capacity, 6);
            cmd_->SetComputeRoot32BitConstant(0, 0u, 7);
            cmd_->SetComputeRootShaderResourceView(1, upload_addr + q_off);   // queries
            cmd_->SetComputeRootShaderResourceView(2, k_plane);               // keys
            cmd_->SetComputeRootShaderResourceView(3, v_plane);               // values
            cmd_->SetComputeRootShaderResourceView(4, upload_addr + pos_off); // query_positions
            cmd_->SetComputeRootUnorderedAccessView(
                5, wht_workspace_.resource->GetGPUVirtualAddress());
            cmd_->Dispatch(geo_.n_head, n_tokens, 1);
            uav_barrier(wht_workspace_);
        }
    }

    // Readback out (DEFAULT UAV -> READBACK).
    transition(*readback_src, D3D12_RESOURCE_STATE_COPY_SOURCE);
    transition(out_readback_, D3D12_RESOURCE_STATE_COPY_DEST);
    cmd_->CopyBufferRegion(out_readback_.resource.Get(), 0, readback_src->resource.Get(), 0,
                           q_bytes);
    // Restore states for the next call.
    transition(*readback_src, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    transition(layer_buf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    if (!submit(error)) {
        s->valid = false;
        return false;
    }

    {
        void* mapped = nullptr;
        D3D12_RANGE range{0, q_bytes};
        HRESULT hr = out_readback_.resource->Map(0, &range, &mapped);
        if (FAILED(hr)) {
            return fail(error, "Map readback", hr);
        }
        std::memcpy(out, mapped, q_bytes);
        D3D12_RANGE empty{0, 0};
        out_readback_.resource->Unmap(0, &empty);
    }
    return true;
}

bool RemoteAttnEngine::configure_range(uint32_t id, uint32_t origin, uint32_t capacity,
                                       std::string* error, uint32_t storage) {
    Session* s = find_session(id);
    if (!ready_ || !s || !s->valid || geo_.cache_bits_k != 16 || geo_.cache_bits_v != 16 ||
        geo_.domain != 0 || capacity != s->capacity || origin > uint32_t(INT32_MAX) - capacity ||
        storage > 2) {
        return ::fail(error, "range config: invalid geometry/session/origin");
    }
    const uint64_t row_bytes = uint64_t(geo_.n_head_kv) *
                               (storage ? uint64_t(geo_.head_dim / 32) * (storage == 1 ? 18 : 34)
                                        : uint64_t(geo_.head_dim) * 2);
    const uint64_t half_stride = uint64_t(capacity) * row_bytes;
    const uint64_t bytes = 2 * half_stride * s->layer_kv.size();
    constexpr uint64_t budget = 4ull * 1024 * 1024 * 1024;
    const uint64_t other_bytes = allocated_bytes_ - s->kv_bytes;
    if (bytes > budget || other_bytes > budget - bytes) {
        return ::fail(error, "range config: KV exceeds budget");
    }
    if (s->kv_bytes != bytes || s->range_storage != storage || !s->layer_kv[0].resource) {
        // Release the old format before allocating its replacement. Range
        // configuration discards history, so no simultaneous duplicate is needed.
        allocated_bytes_ -= s->kv_bytes;
        s->kv_bytes = 0;
        for (auto& buffer : s->layer_kv)
            buffer = Buffer{};
        s->valid = false;
        for (auto& buffer : s->layer_kv) {
            if (!create_buffer(2 * half_stride, D3D12_HEAP_TYPE_DEFAULT,
                               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &buffer, error)) {
                for (auto& failed : s->layer_kv)
                    failed = Buffer{};
                return false;
            }
        }
        s->kv_bytes = bytes;
        allocated_bytes_ += bytes;
        s->half_stride = half_stride;
        s->layer_stride = 2 * half_stride;
        s->valid = true;
    }
    s->range_storage = storage;
    s->range_configured = true;
    s->range_origin = origin;
    s->high_water = -1;
    std::fill(s->range_counts.begin(), s->range_counts.end(), 0);
    std::fill(s->range_record_groups.begin(), s->range_record_groups.end(), 0);
    std::fill(s->range_ready.begin(), s->range_ready.end(), false);
    return true;
}

bool RemoteAttnEngine::configure_kvarn_range(uint32_t id, const rkva_kvarn_range_config& c,
                                             std::string* error) {
    Session* s = find_session(id);
    if (!ready_ || !s || !s->valid || c.capacity != s->capacity || geo_.cache_bits_k != 16 ||
        geo_.cache_bits_v != 16 || geo_.domain != 0 ||
        !rkva_kvarn_range::valid_descriptor(c, geo_.head_dim)) {
        return ::fail(error, "KVarN range: unsupported descriptor/session");
    }
    // Revision 3 fails closed without its own kernel: the revision 1/2 PSOs read
    // the resident image and a 128-row ring, never the serial prefix.
    if (rkva_kvarn_range::serial_prefix(c) && !range_attn_kvarn4_prefix_pso_)
        return ::fail(error, "KVarN range: revision 3 serial-prefix pipeline unavailable");
    const uint64_t generation = rkva_kvarn_new_generation_seed();
    if (!generation)
        return fail(error, "native configuration epochs exhausted");
    const uint64_t heads = uint64_t(geo_.n_head_kv) * 2;
    const uint64_t groups = (uint64_t(c.capacity) + 127) / 128;
    const uint64_t records = groups * heads * 8960;
    const uint64_t stage = 384 * heads * 128 * 2;
    // Revision 3 keeps a 256-row physical ring for the logical 128-row tail.
    const uint64_t tail =
        uint64_t(rkva_kvarn_range::physical_tail_rows(c)) * geo_.n_head_kv * 256 * 2;
    const uint64_t half = records + stage + tail;
    const uint64_t bytes = 2 * half * s->layer_kv.size();
    constexpr uint64_t budget = 4ull * 1024 * 1024 * 1024;
    const uint64_t other = allocated_bytes_ - s->kv_bytes;
    if (bytes > budget || other > budget - bytes || half * 2 >= uint64_t(UINT32_MAX)) {
        return ::fail(error, "KVarN range: KV exceeds budget/address width");
    }
    allocated_bytes_ -= s->kv_bytes;
    s->kv_bytes = 0;
    for (auto& b : s->layer_kv)
        b = Buffer{};
    s->valid = false;
    for (auto& b : s->layer_kv) {
        if (!create_buffer(2 * half, D3D12_HEAP_TYPE_DEFAULT,
                           D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &b, error)) {
            for (auto& failed : s->layer_kv)
                failed = Buffer{};
            return false;
        }
    }
    s->half_stride = half;
    s->layer_stride = 2 * half;
    s->records_bytes = records;
    s->stage_bytes = stage;
    s->range_descriptor = c;
    s->native_generation = generation;
    s->precision_tail_tokens = c.precision_tail_tokens;
    s->kv_bytes = bytes;
    allocated_bytes_ += bytes;
    s->range_storage = 3;
    s->range_configured = true;
    s->range_origin = c.origin;
    s->high_water = -1;
    std::fill(s->range_counts.begin(), s->range_counts.end(), 0);
    std::fill(s->range_record_groups.begin(), s->range_record_groups.end(), 0);
    std::fill(s->range_ready.begin(), s->range_ready.end(), false);
    std::fill(s->prefix_coverage.begin(), s->prefix_coverage.end(),
              rkva_kvarn_range::prefix_coverage{});
    s->valid = true;
    return true;
}

uint64_t RemoteAttnEngine::native_range_generation(uint32_t id) const {
    const Session* s = find_session(id);
    return s && s->range_configured && s->range_storage == 3 ? s->native_generation : 0;
}

// Atomic all-layer bounded rewind. Two stage ping-pong groups retain every
// prefix row needed after <=128 removals; removed suffix rows are masked by
// range_counts and overwritten before a group is sealed again. Only the
// ORIGINAL-domain precision-tail ring needs GPU repair.
bool RemoteAttnEngine::rewind_kvarn_range(uint32_t id, const rkva_kvarn_range_rewind& request,
                                          const uint8_t* data, size_t bytes, std::string* error) {
    Session* s = find_session(id);
    if (!ready_ || !s || !s->valid || !s->range_configured || s->range_storage != 3 ||
        request.reserved || request.pos0 > uint32_t(INT32_MAX) ||
        request.n_layers != s->range_counts.size() ||
        request.tail_tokens != s->precision_tail_tokens || !data)
        return fail(error, "native rewind: invalid session/descriptor");
    rkva_kvarn_rewind_plan plan;
    if (!rkva_kvarn_plan_rewind(request, s->range_origin, s->capacity, geo_.n_head_kv,
                                s->precision_tail_tokens, s->native_generation, s->range_counts,
                                s->range_ready, data, bytes, plan))
        return fail(error, "native rewind: malformed payload or stale coverage");
    const auto& layers = plan.layers;
    const auto& offsets = plan.tail_offsets;
    const uint64_t row_bytes = uint64_t(geo_.n_head_kv) * 256 * 2;
    const bool prefix = rkva_kvarn_range::serial_prefix(s->range_descriptor);
    // Validate every layer before the first mutation. One command list and
    // fence cover every repaired ring; no per-layer RTT or partial ACK.
    if (bytes && range_upload_.bytes < bytes &&
        !create_buffer(bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                       D3D12_RESOURCE_STATE_GENERIC_READ, &range_upload_, error))
        return false;
    void* mapped = nullptr;
    const HRESULT hr = range_upload_.resource->Map(0, nullptr, &mapped);
    if (FAILED(hr))
        return fail(error, "Map native rewind upload", hr);
    std::memcpy(mapped, data, bytes);
    range_upload_.resource->Unmap(0, nullptr);
    if (!begin(error))
        return false;
    for (uint32_t il = 0; il < request.n_layers; ++il) {
        const auto span = rkva_kvarn_rewind_tail(layers[il].expected_live, layers[il].new_live,
                                                 request.tail_tokens);
        const uint32_t tail_rows = span.rows;
        if (!tail_rows)
            continue;
        auto& dst = s->layer_kv[il];
        transition(dst, D3D12_RESOURCE_STATE_COPY_DEST);
        for (uint32_t kind = 0; kind < 2; ++kind) {
            const uint64_t source = offsets[il] + uint64_t(kind) * tail_rows * row_bytes;
            auto copy = [&](uint32_t row, uint32_t target, uint32_t count) {
                cmd_->CopyBufferRegion(dst.resource.Get(),
                                       uint64_t(kind) * s->half_stride + s->records_bytes +
                                           s->stage_bytes + uint64_t(target) * row_bytes,
                                       range_upload_.resource.Get(),
                                       source + uint64_t(row) * row_bytes,
                                       uint64_t(count) * row_bytes);
            };
            if (prefix)
                rkva_kvarn_range::tail_ring_chunks(span.begin, tail_rows,
                                                   rkva_kvarn_range::kPrefixTailRing, copy);
            else
                rkva_kvarn_range::tail_chunks(span.begin, tail_rows, copy);
        }
        transition(dst, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    if (!submit(error)) {
        s->valid = false;
        return false;
    }
    ++s->native_generation;
    for (uint32_t il = 0; il < request.n_layers; ++il) {
        if (prefix) {
            const auto span = rkva_kvarn_rewind_tail(layers[il].expected_live, layers[il].new_live,
                                                     request.tail_tokens);
            s->prefix_coverage[il] = rkva_kvarn_range::coverage_after_rewind(
                s->prefix_coverage[il], layers[il].new_live, span.begin, span.rows);
        }
        s->range_counts[il] = layers[il].new_live;
        s->range_record_groups[il] = layers[il].new_live / 128;
        s->range_ready[il] = layers[il].new_live != 0;
    }
    if (s->high_water >= int32_t(request.pos0))
        s->high_water = int32_t(request.pos0) - 1;
    return true;
}

bool RemoteAttnEngine::upload_kvarn_range(uint32_t id, uint32_t layer,
                                          const rkva_kvarn_range_upload& u, const uint8_t* data,
                                          size_t bytes, std::string* error) {
    const bool prof = profiling_ && profile_ && !bulk_recording_;
    const bool bulk_prof = profiling_ && profile_ && bulk_recording_ && profile_->bulk.active;
    rp::Sample sample;
    SpanTimer timer;
    timer.start(prof, sample.cpu_ns);
    FailedCallGuard guard{prof ? &profile_->recorder.counters.failed_calls : nullptr};
    Session* s = find_session(id);
    if (!ready_ || !s || !rkva_kvarn_generation_mutable(s->native_generation) || !s->valid ||
        !s->range_configured || s->range_storage != 3 || layer >= s->layer_kv.size() || !data ||
        !rkva_kvarn_range::valid_upload(
            u, s->capacity, s->range_counts[layer], s->range_record_groups[layer],
            s->range_ready[layer],
            rkva_kvarn_range::compact_stage_supported(s->range_descriptor))) {
        return ::fail(error, "KVarN range upload: invalid descriptor/coverage/snapshot");
    }
    const uint64_t record_group_bytes = uint64_t(geo_.n_head_kv) * 2 * 8960;
    const uint64_t rec_bytes = uint64_t(u.record_groups) * record_group_bytes;
    const bool reset = (u.flags & 1) != 0;
    const bool prefix = rkva_kvarn_range::serial_prefix(s->range_descriptor);
    const uint64_t row_bytes = uint64_t(geo_.n_head_kv) * 256 * 2;
    const uint32_t stage_rows = rkva_kvarn_range::upload_stage_rows(u);
    const uint32_t tail_rows =
        s->precision_tail_tokens
            ? (reset ? std::min(u.live_count, s->precision_tail_tokens) : u.append_rows)
            : 0;
    const uint64_t stage_payload = uint64_t(stage_rows) * row_bytes;
    const uint64_t tail_payload = uint64_t(tail_rows) * row_bytes;
    const uint64_t expected = 2 * (rec_bytes + stage_payload + tail_payload);
    if (bytes != expected)
        return ::fail(error, "KVarN range upload: wrong payload size");
    timer.mark(rp::kCpuValidate);
    const bool grow = range_upload_.bytes < expected;
    if (grow && !create_buffer(expected, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                               D3D12_RESOURCE_STATE_GENERIC_READ, &range_upload_, error))
        return false;
    timer.mark(rp::kCpuAlloc);
    const prof_clock::time_point t_map_start =
        bulk_prof ? prof_clock::now() : prof_clock::time_point{};
    void* mapped = nullptr;
    HRESULT hr = range_upload_.resource->Map(0, nullptr, &mapped);
    if (FAILED(hr))
        return fail(error, "Map KVarN range upload", hr);
    std::memcpy(mapped, data, bytes);
    range_upload_.resource->Unmap(0, nullptr);
    timer.mark(rp::kCpuMapCopy);
    if (bulk_prof) {
        profile_->bulk.segments[profile_->bulk.segment_index].map_copy_upload_ns +=
            elapsed_ns(t_map_start, prof_clock::now());
    }
    if (!begin(error))
        return false;
    const prof_clock::time_point t_rec_start =
        bulk_prof ? prof_clock::now() : prof_clock::time_point{};
    const bool ts = prof && timestamp_heap_;
    if (ts)
        timestamp(kTsUploadBegin);
    if (bulk_prof && timestamp_heap_) {
        const uint32_t slot =
            1 + profile_->bulk.segment_index * kTsBulkSlotsPerSegment + kTsBulkUploadBegin;
        timestamp(slot);
    }
    Buffer& dst = s->layer_kv[layer];
    transition(dst, D3D12_RESOURCE_STATE_COPY_DEST);
    const uint64_t record_offset = uint64_t(u.record_begin) * record_group_bytes;
    if (rec_bytes) {
        cmd_->CopyBufferRegion(dst.resource.Get(), record_offset, range_upload_.resource.Get(), 0,
                               rec_bytes);
        cmd_->CopyBufferRegion(dst.resource.Get(), s->half_stride + record_offset,
                               range_upload_.resource.Get(), rec_bytes, rec_bytes);
    }
    for (uint32_t kind = 0; kind < 2; ++kind) {
        const uint64_t plane = kind * s->half_stride;
        const uint64_t stage_source = 2 * rec_bytes + kind * stage_payload;
        if (reset && stage_payload) {
            cmd_->CopyBufferRegion(dst.resource.Get(), plane + s->records_bytes,
                                   range_upload_.resource.Get(), stage_source, stage_payload);
        } else {
            rkva_kvarn_range::stage_chunks(
                s->range_origin + u.append_begin, stage_rows,
                [&](uint32_t source, uint32_t target, uint32_t rows) {
                    cmd_->CopyBufferRegion(dst.resource.Get(),
                                           plane + s->records_bytes + target * row_bytes,
                                           range_upload_.resource.Get(),
                                           stage_source + source * row_bytes, rows * row_bytes);
                });
        }
        const uint64_t tail_source = 2 * (rec_bytes + stage_payload) + kind * tail_payload;
        const uint32_t tail_begin = reset ? u.live_count - tail_rows : u.append_begin;
        auto copy_tail = [&](uint32_t source, uint32_t target, uint32_t rows) {
            cmd_->CopyBufferRegion(
                dst.resource.Get(), plane + s->records_bytes + s->stage_bytes + target * row_bytes,
                range_upload_.resource.Get(), tail_source + source * row_bytes, rows * row_bytes);
        };
        // A prefill may upload >128 new rows: only its newest 128 (revision 3:
        // 256, the physical ring) stay in the overlay.
        if (prefix)
            rkva_kvarn_range::tail_ring_chunks(tail_begin, tail_rows,
                                               rkva_kvarn_range::kPrefixTailRing, copy_tail);
        else
            rkva_kvarn_range::tail_chunks(tail_begin, tail_rows, copy_tail);
    }
    transition(dst, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (ts) {
        timestamp(kTsUploadEnd);
        resolve_timestamps(kTsUploadBegin, 2);
    }
    if (bulk_prof) {
        if (timestamp_heap_) {
            const uint32_t slot =
                1 + profile_->bulk.segment_index * kTsBulkSlotsPerSegment + kTsBulkUploadEnd;
            timestamp(slot);
        }
        auto& seg = profile_->bulk.segments[profile_->bulk.segment_index];
        seg.record_upload_ns += elapsed_ns(t_rec_start, prof_clock::now());
        seg.upload_bytes = bytes;
        seg.append_rows = u.append_rows;
        seg.live_count = u.live_count;
    }
    timer.mark(rp::kCpuRecord);
    if (!submit(error)) {
        s->valid = false;
        return false;
    }
    timer.mark(rp::kCpuFenceWait);
    s->range_record_groups[layer] =
        (u.flags & 1) ? u.record_groups
                      : std::max(s->range_record_groups[layer], u.record_begin + u.record_groups);
    ++s->native_generation;
    if (prefix)
        s->prefix_coverage[layer] =
            reset ? rkva_kvarn_range::coverage_after_reset(u.live_count, s->precision_tail_tokens)
                  : rkva_kvarn_range::coverage_after_append(s->prefix_coverage[layer], u.live_count,
                                                            s->precision_tail_tokens);
    s->range_counts[layer] = u.live_count;
    s->range_ready[layer] = true;
    if (prof) {
        sample.kind = rp::kUpload;
        sample.storage = 3;
        sample.serial_prefix = prefix;
        sample.mode = uint8_t(q4_kernel_mode_);
        sample.alloc_grew = grow;
        sample.n_tokens = u.append_rows;
        sample.rows = u.append_rows;
        sample.bytes = bytes;
        finish_profile_sample(sample, ts, kTsUploadBegin, 2, 0, 0);
        guard.done = true;
    }
    return true;
}

bool RemoteAttnEngine::upload_range(uint32_t id, uint32_t layer, uint32_t start_row,
                                    uint32_t n_rows, const uint16_t* k, const uint16_t* v,
                                    std::string* error) {
    // In bulk recording the submit is deferred to the envelope, so the
    // timestamp readback would run before the GPU; bulk profiling collects
    // upload timings from the envelope instead.
    const bool prof = profiling_ && profile_ && !bulk_recording_;
    rp::Sample sample;
    SpanTimer timer;
    timer.start(prof, sample.cpu_ns);
    FailedCallGuard guard{prof ? &profile_->recorder.counters.failed_calls : nullptr};
    Session* s = find_session(id);
    if (!ready_ || !s || !s->valid || !s->range_configured || layer >= s->layer_kv.size() ||
        s->range_storage == 3 || !n_rows || !k || !v || start_row > s->capacity ||
        n_rows > s->capacity - start_row || (start_row && start_row != s->range_counts[layer])) {
        return ::fail(error, "range upload: invalid/noncontiguous rows");
    }
    const uint64_t row_bytes =
        uint64_t(geo_.n_head_kv) *
        (s->range_storage ? uint64_t(geo_.head_dim / 32) * (s->range_storage == 1 ? 18 : 34)
                          : uint64_t(geo_.head_dim) * 2);
    const uint64_t bytes = uint64_t(n_rows) * row_bytes;
    timer.mark(rp::kCpuValidate);
    const bool grow = range_upload_.bytes < 2 * bytes;
    if (grow && !create_buffer(2 * bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                               D3D12_RESOURCE_STATE_GENERIC_READ, &range_upload_, error)) {
        return false;
    }
    timer.mark(rp::kCpuAlloc);
    void* mapped = nullptr;
    HRESULT hr = range_upload_.resource->Map(0, nullptr, &mapped);
    if (FAILED(hr)) {
        return fail(error, "Map range upload", hr);
    }
    std::memcpy(mapped, k, bytes);
    std::memcpy(static_cast<uint8_t*>(mapped) + bytes, v, bytes);
    range_upload_.resource->Unmap(0, nullptr);
    timer.mark(rp::kCpuMapCopy);
    if (!begin(error)) {
        return false;
    }
    const bool ts = prof && timestamp_heap_;
    if (ts) {
        timestamp(kTsUploadBegin);
    }
    Buffer& dst = s->layer_kv[layer];
    transition(dst, D3D12_RESOURCE_STATE_COPY_DEST);
    const uint64_t offset = uint64_t(start_row) * row_bytes;
    cmd_->CopyBufferRegion(dst.resource.Get(), offset, range_upload_.resource.Get(), 0, bytes);
    cmd_->CopyBufferRegion(dst.resource.Get(), s->half_stride + offset,
                           range_upload_.resource.Get(), bytes, bytes);
    transition(dst, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (ts) {
        timestamp(kTsUploadEnd);
        resolve_timestamps(kTsUploadBegin, 2);
    }
    timer.mark(rp::kCpuRecord);
    if (!submit(error)) {
        s->valid = false;
        return false;
    }
    timer.mark(rp::kCpuFenceWait);
    s->range_counts[layer] = start_row + n_rows;
    if (prof) {
        sample.kind = rp::kUpload;
        sample.storage = uint8_t(s->range_storage);
        sample.mode = uint8_t(q4_kernel_mode_);
        sample.alloc_grew = grow;
        sample.n_tokens = n_rows;
        sample.rows = n_rows;
        sample.bytes = 2 * bytes;
        finish_profile_sample(sample, ts, kTsUploadBegin, 2, 2, 0);
        guard.done = true;
    }
    return true;
}

bool RemoteAttnEngine::ensure_range_scratch(uint32_t n_tokens, std::string* error) {
    if (range_scratch_tokens_ >= n_tokens) {
        return true;
    }
    const uint64_t q_heads = uint64_t(n_tokens) * geo_.n_head;
    const uint64_t q_bytes = q_heads * geo_.head_dim * 4;
    const uint64_t out_bytes = q_bytes + q_heads * 4;
    const uint64_t partial_bytes = 32 * q_heads * (geo_.head_dim + 2) * 4;
    if (!create_buffer(2 * q_bytes + uint64_t(n_tokens) * 4, D3D12_HEAP_TYPE_UPLOAD,
                       D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, &range_query_,
                       error) ||
        !create_buffer(partial_bytes, D3D12_HEAP_TYPE_DEFAULT,
                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &range_partials_, error) ||
        !create_buffer(out_bytes, D3D12_HEAP_TYPE_DEFAULT,
                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &range_output_, error) ||
        !create_buffer(out_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
                       D3D12_RESOURCE_STATE_COPY_DEST, &range_readback_, error)) {
        return false;
    }
    range_scratch_tokens_ = n_tokens;
    return true;
}

// Both F32 and F16/FP8 bulk paths retire timestamp readback after their existing
// single submit/fence. This helper never submits or waits for GPU work.
void RemoteAttnEngine::finish_bulk_profile(rp::BulkEnvelopeSample& envelope, bool timestamps) {
    const prof_clock::time_point t_overhead_start = prof_clock::now();
    const uint32_t total_slots = 2 + envelope.segment_count * kTsBulkSlotsPerSegment;
    uint64_t ticks[kTsPerSet] = {};
    bool gpu_valid = false;
    if (timestamps && read_timestamps(0, total_slots, ticks)) {
        gpu_valid = true;
    }

    uint64_t total_map_copy_ns = 0;
    uint64_t total_record_ns = 0;
    uint64_t total_upload_bytes = 0;
    uint64_t total_query_bytes = 0;
    uint64_t total_output_bytes = 0;

    uint64_t env_gpu_upload_ticks = 0;
    uint64_t env_gpu_unpack_ticks = 0;
    uint64_t env_gpu_pack_ticks = 0;
    uint64_t env_gpu_pass1_ticks = 0;
    uint64_t env_gpu_reduce_ticks = 0;
    uint64_t env_gpu_copy_ticks = 0;

    rp::Counters& c = profile_->recorder.counters;
    ++c.bulk_calls;
    c.bulk_segments += envelope.segment_count;
    c.bulk_queries += envelope.total_queries;
    if (envelope.wire == RKVA_WIRE_FP8) {
        if (envelope.fp8_gpu)
            ++c.fp8_gpu_calls;
        else
            ++c.fp8_cpu_fallback_calls;
    }

    for (size_t i = 0; i < envelope.segment_count; ++i) {
        const auto& seg = profile_->bulk.segments[i];
        total_map_copy_ns += (seg.map_copy_upload_ns + seg.map_copy_query_ns);
        total_record_ns += (seg.record_upload_ns + seg.record_attn_ns);
        total_upload_bytes += seg.upload_bytes;
        total_query_bytes += seg.query_bytes;
        total_output_bytes += seg.out_bytes;

        const uint32_t base = 1 + static_cast<uint32_t>(i) * kTsBulkSlotsPerSegment;
        uint64_t u_ticks = 0;
        uint64_t p1_ticks = 0;
        uint64_t red_ticks = 0;
        uint64_t cp_ticks = 0;
        if (gpu_valid) {
            u_ticks = ticks[base + kTsBulkUploadEnd] - ticks[base + kTsBulkUploadBegin];
            p1_ticks = ticks[base + kTsBulkAttnPass1] - ticks[base + kTsBulkAttnBegin];
            red_ticks = ticks[base + kTsBulkAttnReduce] - ticks[base + kTsBulkAttnPass1];
            cp_ticks = ticks[base + kTsBulkAttnCopy] - ticks[base + kTsBulkAttnReduce];

            env_gpu_upload_ticks += u_ticks;
            env_gpu_unpack_ticks += ticks[base + kTsBulkAttnBegin] - ticks[base + kTsBulkUploadEnd];
            env_gpu_pack_ticks += ticks[base + kTsBulkPackEnd] - ticks[base + kTsBulkAttnCopy];
            cp_ticks += ticks[base + kTsBulkCopyEnd] - ticks[base + kTsBulkPackEnd];
            env_gpu_pass1_ticks += p1_ticks;
            env_gpu_reduce_ticks += red_ticks;
            env_gpu_copy_ticks += cp_ticks;
        }

        // Per-segment upload Sample
        rp::Sample up_sample;
        up_sample.kind = rp::kUpload;
        up_sample.storage = 3;
        up_sample.serial_prefix = 1;
        up_sample.mode = uint8_t(q4_kernel_mode_);
        up_sample.alloc_grew = 0;
        up_sample.bulk_segment = 1;
        up_sample.n_tokens = seg.append_rows;
        up_sample.rows = seg.append_rows;
        up_sample.bytes = seg.upload_bytes;
        up_sample.cpu_ns[rp::kCpuMapCopy] = seg.map_copy_upload_ns;
        up_sample.cpu_ns[rp::kCpuRecord] = seg.record_upload_ns;
        if (gpu_valid) {
            up_sample.gpu_ticks[0] = u_ticks;
            up_sample.gpu_valid = 1;
        }
        profile_->recorder.record(up_sample);
        ++c.calls[rp::kUpload];
        c.copy_regions += 1;

        // Per-segment attention Sample
        rp::Sample attn_sample;
        attn_sample.kind = rp::kAttention;
        attn_sample.storage = 3;
        attn_sample.serial_prefix = 1;
        attn_sample.mode = uint8_t(q4_kernel_mode_);
        attn_sample.alloc_grew = 0;
        attn_sample.bulk_segment = 1;
        attn_sample.optimized = seg.optimized;
        attn_sample.direct_output = seg.direct_output;
        attn_sample.split_k = seg.split_k;
        attn_sample.groups = seg.groups;
        attn_sample.q_tile = seg.q_tile;
        attn_sample.decode_opt = seg.decode_opt;
        attn_sample.n_tokens = seg.n_queries;
        attn_sample.rows = seg.active_count;
        attn_sample.bytes = seg.query_bytes + seg.out_bytes;
        attn_sample.cpu_ns[rp::kCpuMapCopy] = seg.map_copy_query_ns;
        attn_sample.cpu_ns[rp::kCpuRecord] = seg.record_attn_ns;
        attn_sample.cpu_ns[rp::kCpuReadback] = seg.readback_ns;
        if (gpu_valid) {
            attn_sample.gpu_ticks[0] = p1_ticks;
            attn_sample.gpu_ticks[1] = red_ticks;
            attn_sample.gpu_ticks[2] = cp_ticks;
            attn_sample.gpu_valid = 1;
        }
        profile_->recorder.record(attn_sample);
        ++c.calls[rp::kAttention];
        c.dispatches += seg.dispatches;
        c.copy_regions += 1;
        if (seg.optimized) {
            ++c.optimized_by_storage[rp::kStorageKvarn4];
            if (seg.decode_opt) {
                ++c.kvarn4_decode_calls;
            } else if (seg.q_tile == rp::kKvarn4MqTile) {
                ++c.kvarn4_mq_calls;
            } else if (seg.q_tile == rp::kKvarn4MqTileMax) {
                ++c.kvarn4_mq4_calls;
            }
        } else {
            ++c.reference_by_storage[rp::kStorageKvarn4];
        }
    }

    envelope.cpu_ns[rp::kCpuMapCopy] = total_map_copy_ns;
    if (!envelope.cpu_ns[rp::kCpuRecord])
        envelope.cpu_ns[rp::kCpuRecord] = total_record_ns;
    envelope.upload_bytes = total_upload_bytes;
    envelope.query_bytes = total_query_bytes;
    envelope.output_bytes = total_output_bytes;

    if (gpu_valid) {
        envelope.gpu_valid = 1;
        envelope.gpu_upload_ticks = env_gpu_upload_ticks;
        envelope.gpu_unpack_ticks = env_gpu_unpack_ticks;
        envelope.gpu_pack_ticks = env_gpu_pack_ticks;
        envelope.gpu_pass1_ticks = env_gpu_pass1_ticks;
        envelope.gpu_reduce_ticks = env_gpu_reduce_ticks;
        envelope.gpu_copy_ticks = env_gpu_copy_ticks;
        const uint32_t env_end_slot =
            1 + static_cast<uint32_t>(envelope.segment_count) * kTsBulkSlotsPerSegment;
        envelope.gpu_total_ticks = ticks[env_end_slot] - ticks[kTsBulkEnvelopeBegin];
    }

    envelope.overhead_ns = elapsed_ns(t_overhead_start, prof_clock::now());
    profile_->recorder.record_bulk(envelope);
    profile_->bulk.active = false;
}

bool RemoteAttnEngine::attend_kvarn_bulk(uint32_t id, uint32_t layer, uint32_t n_tokens,
                                         const uint8_t* data, size_t bytes, float* out, float* lse,
                                         std::string* error, uint64_t cycle_id) {
    const bool prof = profiling_ && profile_ && !bulk_recording_;
    const prof_clock::time_point t_val_start = prof ? prof_clock::now() : prof_clock::time_point{};
    rp::BulkEnvelopeSample envelope;
    envelope.total_queries = n_tokens;
    envelope.cycle_id = cycle_id;
    envelope.session_id = id;
    envelope.layer = layer;
    envelope.wire = RKVA_WIRE_F32;
    rkva_kvarn_bulk::ScratchCall scratch_call(bulk_scratch_, n_tokens, RKVA_WIRE_F32);

    Session* s = find_session(id);
    if (!ready_ || !s || !s->valid || !s->range_configured || s->range_storage != 3 ||
        layer >= s->layer_kv.size() || !out || !lse || !range_attn_kvarn4_prefix_pso_ ||
        bulk_recording_) {
        if (prof) {
            ++profile_->recorder.counters.failed_calls;
        }
        return ::fail(error, "native bulk: invalid session/arguments");
    }
    auto& segments = bulk_scratch_.segments;
    if (segments.capacity() < kTsBulkMaxSegments)
        segments.reserve(kTsBulkMaxSegments);
    if (!rkva_kvarn_bulk::validate(data, bytes, n_tokens, s->range_descriptor, geo_.n_head,
                                   geo_.n_head_kv, s->range_counts[layer],
                                   s->range_record_groups[layer], s->range_ready[layer],
                                   s->prefix_coverage[layer], s->native_generation, segments)) {
        if (prof) {
            ++profile_->recorder.counters.failed_calls;
        }
        return ::fail(error, "native bulk: invalid envelope/coverage/query");
    }
    envelope.segment_count = static_cast<uint32_t>(segments.size());
    if (prof) {
        envelope.cpu_ns[rp::kCpuValidate] = elapsed_ns(t_val_start, prof_clock::now());
    }

    // Admission and all grow-only allocation precede the first native mutation.
    const prof_clock::time_point t_alloc_start =
        prof ? prof_clock::now() : prof_clock::time_point{};
    uint32_t max_queries = 0;
    for (const auto& segment : segments)
        max_queries = std::max(max_queries, segment.n_queries);
    const bool scratch_grew = range_scratch_tokens_ < max_queries;
    if (!ensure_range_scratch(max_queries, error)) {
        if (prof) {
            ++profile_->recorder.counters.failed_calls;
        }
        return false;
    }
    bool alloc_grew = scratch_grew;
    if (bulk_transfers_.size() < segments.size()) {
        alloc_grew = true;
        bulk_transfers_.resize(segments.size());
    }
    for (size_t i = 0; i < segments.size(); ++i) {
        const auto& segment = segments[i];
        auto& transfer = bulk_transfers_[i];
        const uint64_t q_bytes = uint64_t(segment.n_queries) * geo_.n_head * 256 * 4;
        const uint64_t query_bytes = 2 * q_bytes + segment.n_queries * 4;
        const uint64_t output_bytes = q_bytes + uint64_t(segment.n_queries) * geo_.n_head * 4;
        if (transfer.upload.bytes < segment.data_bytes) {
            alloc_grew = true;
            if (!create_buffer(segment.data_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                               D3D12_RESOURCE_STATE_GENERIC_READ, &transfer.upload, error)) {
                if (prof) {
                    ++profile_->recorder.counters.failed_calls;
                }
                return false;
            }
        }
        if (transfer.query.bytes < query_bytes) {
            alloc_grew = true;
            if (!create_buffer(query_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                               D3D12_RESOURCE_STATE_GENERIC_READ, &transfer.query, error)) {
                if (prof) {
                    ++profile_->recorder.counters.failed_calls;
                }
                return false;
            }
        }
        if (transfer.readback.bytes < output_bytes) {
            alloc_grew = true;
            if (!create_buffer(output_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
                               D3D12_RESOURCE_STATE_COPY_DEST, &transfer.readback, error)) {
                if (prof) {
                    ++profile_->recorder.counters.failed_calls;
                }
                return false;
            }
        }
    }
    if (prof) {
        envelope.cpu_ns[rp::kCpuAlloc] = elapsed_ns(t_alloc_start, prof_clock::now());
        envelope.alloc_grew = alloc_grew ? 1 : 0;
    }

    if (!begin(error)) {
        if (prof) {
            ++profile_->recorder.counters.failed_calls;
        }
        return false;
    }

    const bool ts_bulk = prof && timestamp_heap_ && (segments.size() <= kTsBulkMaxSegments);
    if (prof) {
        profile_->bulk.active = true;
        profile_->bulk.segment_index = 0;
        profile_->bulk.segments.clear();
        profile_->bulk.segments.resize(segments.size());
    }
    if (ts_bulk) {
        timestamp(kTsBulkEnvelopeBegin);
    }

    bulk_recording_ = true;
    for (size_t i = 0; i < segments.size(); ++i) {
        if (prof) {
            profile_->bulk.segment_index = static_cast<uint32_t>(i);
        }
        const auto& segment = segments[i];
        auto& transfer = bulk_transfers_[i];
        // Each recorded copy/dispatch references immutable transfer memory. The
        // attention output/partials are shared in strict command-list order.
        std::swap(range_upload_, transfer.upload);
        std::swap(range_query_, transfer.query);
        std::swap(range_readback_, transfer.readback);
        const bool ok =
            upload_kvarn_range(id, layer, segment.upload, data + segment.data_offset,
                               segment.data_bytes, error) &&
            attend_range(id, layer, segment.n_queries,
                         reinterpret_cast<const int32_t*>(data + segment.positions_offset),
                         reinterpret_cast<const float*>(data + segment.query_offset), out, lse,
                         error);
        std::swap(range_upload_, transfer.upload);
        std::swap(range_query_, transfer.query);
        std::swap(range_readback_, transfer.readback);
        if (ts_bulk && ok) {
            const uint32_t base = 1 + uint32_t(i) * kTsBulkSlotsPerSegment;
            timestamp(base + kTsBulkPackEnd);
            timestamp(base + kTsBulkCopyEnd);
        }
        if (!ok) {
            bulk_recording_ = false;
            if (prof) {
                profile_->bulk.active = false;
                ++profile_->recorder.counters.failed_calls;
            }
            s->valid = false;
            // Recorded resource transitions were not executed; do not reuse
            // their host-side state tracking in any other session.
            terminal_failure_ = true;
            return false;
        }
    }

    const uint32_t total_slots =
        2 + static_cast<uint32_t>(segments.size()) * kTsBulkSlotsPerSegment;
    if (ts_bulk) {
        const uint32_t env_end_slot =
            1 + static_cast<uint32_t>(segments.size()) * kTsBulkSlotsPerSegment;
        timestamp(env_end_slot);
        resolve_timestamps(0, total_slots);
    }
    bulk_recording_ = false;

    const prof_clock::time_point t_submit_start =
        prof ? prof_clock::now() : prof_clock::time_point{};
    if (!submit(error)) {
        if (prof) {
            profile_->bulk.active = false;
            ++profile_->recorder.counters.failed_calls;
        }
        s->valid = false;
        return false;
    }
    const uint64_t submit_fence_ns = prof ? elapsed_ns(t_submit_start, prof_clock::now()) : 0;
    if (prof) {
        const uint64_t exec_ns = std::min(last_submit_exec_ns_, submit_fence_ns);
        envelope.cpu_ns[rp::kCpuSubmit] = exec_ns;
        envelope.cpu_ns[rp::kCpuFenceWait] = submit_fence_ns - exec_ns;
        envelope.fence_blocked = last_submit_blocked_ ? 1 : 0;
    }

    const prof_clock::time_point t_rb_total_start =
        prof ? prof_clock::now() : prof_clock::time_point{};
    size_t row = 0;
    for (size_t i = 0; i < segments.size(); ++i) {
        const prof_clock::time_point t_seg_rb = prof ? prof_clock::now() : prof_clock::time_point{};
        const uint64_t heads = uint64_t(segments[i].n_queries) * geo_.n_head;
        const uint64_t q_bytes = heads * 256 * 4;
        const uint64_t output_bytes = q_bytes + heads * 4;
        void* mapped = nullptr;
        D3D12_RANGE range{0, static_cast<SIZE_T>(output_bytes)};
        const HRESULT hr = bulk_transfers_[i].readback.resource->Map(0, &range, &mapped);
        if (FAILED(hr)) {
            if (prof) {
                profile_->bulk.active = false;
                ++profile_->recorder.counters.failed_calls;
            }
            s->valid = false;
            return fail(error, "Map native bulk readback", hr);
        }
        std::memcpy(out + row * geo_.n_head * 256, mapped, q_bytes);
        std::memcpy(lse + row * geo_.n_head, static_cast<const uint8_t*>(mapped) + q_bytes,
                    heads * 4);
        D3D12_RANGE empty{0, 0};
        bulk_transfers_[i].readback.resource->Unmap(0, &empty);
        row += segments[i].n_queries;
        if (prof) {
            profile_->bulk.segments[i].readback_ns = elapsed_ns(t_seg_rb, prof_clock::now());
        }
    }
    if (prof) {
        envelope.cpu_ns[rp::kCpuReadback] = elapsed_ns(t_rb_total_start, prof_clock::now());
    }

    if (prof) {
        finish_bulk_profile(envelope, ts_bulk);
    }
    return true;
}

bool RemoteAttnEngine::attend_native_rows(uint32_t id, uint32_t layer, uint32_t n_tokens,
                                          uint32_t storage, uint64_t row_bytes,
                                          const uint8_t* data, size_t bytes, float* out,
                                          float* lse, std::string* error) {
    struct q4_segment {
        size_t up_off, pos_off, q_off;
        uint32_t up_bytes, append_begin, append_rows, n_queries, wire;
    };
    Session* s = find_session(id);
    if (!ready_ || !s || !s->valid || !s->range_configured ||
        s->range_storage != storage || layer >= s->layer_kv.size() ||
        !data || !out || !lse || bulk_recording_ || !n_tokens || n_tokens > 256) {
        return ::fail(error, "q4 bulk: invalid session/arguments");
    }
    std::vector<q4_segment> segments;
    segments.reserve(32);
    size_t off = sizeof(rkva_kvarn_range_bulk);
    rkva_kvarn_range_bulk header{};
    if (bytes < off) return ::fail(error, "q4 bulk: short header");
    std::memcpy(&header, data, sizeof(header));
    if (header.version != 1 || !header.segment_count || header.segment_count > 32 ||
        header.reserved || header.total_queries != n_tokens)
        return ::fail(error, "q4 bulk: bad header");
    const uint64_t row = row_bytes;
    uint32_t sum_queries = 0;
    for (uint32_t i = 0; i < header.segment_count; ++i) {
        if (bytes - off < sizeof(rkva_kvarn_range_bulk_segment))
            return ::fail(error, "q4 bulk: short segment");
        rkva_kvarn_range_bulk_segment seg{};
        std::memcpy(&seg, data + off, sizeof(seg));
        off += sizeof(seg);
        if (seg.upload_bytes < sizeof(rkva_kvarn_range_upload) || seg.upload_bytes > bytes - off)
            return ::fail(error, "q4 bulk: bad upload size");
        rkva_kvarn_range_upload u{};
        std::memcpy(&u, data + off, sizeof(u));
        const bool upload_ok = storage == RKVA_RANGE_STORAGE_Q8_0
                ? rkva_kvarn_range::valid_q8_upload(u, s->capacity, u.append_begin)
                : rkva_kvarn_range::valid_q4_upload(u, s->capacity, u.append_begin);
        if (!upload_ok ||
            uint64_t(u.append_begin) + u.append_rows != u.live_count ||
            seg.upload_bytes != sizeof(u) + uint64_t(u.append_rows) * row * 2)
            return ::fail(error, "q4 bulk: bad upload header");
        q4_segment qs{off + sizeof(u), 0, 0, seg.upload_bytes, u.append_begin, u.append_rows,
                      seg.n_queries, RKVA_WIRE_F32};
        off += seg.upload_bytes;
        if (bytes - off < sizeof(rkva_range_attn))
            return ::fail(error, "q4 bulk: short attention");
        rkva_range_attn attn{};
        std::memcpy(&attn, data + off, sizeof(attn));
        off += sizeof(attn);
        qs.wire = attn.wire;
        if (!qs.n_queries || qs.n_queries > n_tokens - sum_queries || attn.flags ||
            attn.reserved[0] || attn.reserved[1] ||
            (attn.wire != RKVA_WIRE_F32 && attn.wire != RKVA_WIRE_F16 &&
             attn.wire != RKVA_WIRE_FP8))
            return ::fail(error, "q4 bulk: bad attention");
        const uint64_t elements = uint64_t(qs.n_queries) * geo_.n_head * geo_.head_dim;
        const uint64_t pos_bytes = uint64_t(qs.n_queries) * sizeof(int32_t);
        const uint64_t q_bytes = elements * (attn.wire == RKVA_WIRE_FP8 ? 1u :
                                             attn.wire == RKVA_WIRE_F16 ? 2u : 4u);
        if (elements > (64ull << 20) || bytes - off < pos_bytes + q_bytes)
            return ::fail(error, "q4 bulk: short query payload");
        qs.pos_off = off;
        qs.q_off = off + pos_bytes;
        off += pos_bytes + q_bytes;
        sum_queries += qs.n_queries;
        segments.push_back(qs);
    }
    if (off != bytes || sum_queries != n_tokens)
        return ::fail(error, "q4 bulk: trailing bytes/query total");

    uint32_t max_queries = 0;
    for (const auto& segment : segments)
        max_queries = std::max(max_queries, segment.n_queries);
    if (!ensure_range_scratch(max_queries, error)) return false;
    if (bulk_transfers_.size() < segments.size()) bulk_transfers_.resize(segments.size());
    for (size_t i = 0; i < segments.size(); ++i) {
        auto& transfer = bulk_transfers_[i];
        const uint64_t q_bytes = uint64_t(segments[i].n_queries) * geo_.n_head * 256 * 4;
        const uint64_t query_bytes = 2 * q_bytes + segments[i].n_queries * 4;
        const uint64_t output_bytes = q_bytes + uint64_t(segments[i].n_queries) * geo_.n_head * 4;
        if (transfer.upload.bytes < segments[i].up_bytes &&
            !create_buffer(segments[i].up_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                           D3D12_RESOURCE_STATE_GENERIC_READ, &transfer.upload, error)) return false;
        if (transfer.query.bytes < query_bytes &&
            !create_buffer(query_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                           D3D12_RESOURCE_STATE_GENERIC_READ, &transfer.query, error)) return false;
        if (transfer.readback.bytes < output_bytes &&
            !create_buffer(output_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
                           D3D12_RESOURCE_STATE_COPY_DEST, &transfer.readback, error)) return false;
    }
    if (!begin(error)) return false;
    bulk_recording_ = true;
    std::vector<float> q_f32;
    for (size_t i = 0; i < segments.size(); ++i) {
        const auto& seg = segments[i];
        auto& transfer = bulk_transfers_[i];
        std::swap(range_upload_, transfer.upload);
        std::swap(range_query_, transfer.query);
        std::swap(range_readback_, transfer.readback);
        const uint8_t* up = data + seg.up_off;
        const uint64_t plane = uint64_t(seg.append_rows) * row;
        const uint8_t* kp = up;
        const uint8_t* vp = up + plane;
        const uint64_t elements = uint64_t(seg.n_queries) * geo_.n_head * geo_.head_dim;
        q_f32.resize(elements);
        const uint8_t* qsrc = data + seg.q_off;
        if (seg.wire == RKVA_WIRE_FP8) {
            for (uint64_t e = 0; e < elements; ++e)
                q_f32[e] = rkva_kvarn_bulk::f16_bits_to_f32(
                        rkva_kvarn_bulk::fp8_to_f16_bits(qsrc[e]));
        } else if (seg.wire == RKVA_WIRE_F16) {
            const uint16_t* q16 = reinterpret_cast<const uint16_t*>(qsrc);
            for (uint64_t e = 0; e < elements; ++e)
                q_f32[e] = rkva_kvarn_bulk::f16_bits_to_f32(q16[e]);
        } else {
            std::memcpy(q_f32.data(), qsrc, elements * sizeof(float));
        }
        const bool ok = (seg.append_rows == 0 ||
                         upload_range(id, layer, seg.append_begin, seg.append_rows,
                                      reinterpret_cast<const uint16_t*>(kp),
                                      reinterpret_cast<const uint16_t*>(vp), error)) &&
                        attend_range(id, layer, seg.n_queries,
                                     reinterpret_cast<const int32_t*>(data + seg.pos_off),
                                     q_f32.data(), out, lse, error);
        std::swap(range_upload_, transfer.upload);
        std::swap(range_query_, transfer.query);
        std::swap(range_readback_, transfer.readback);
        if (!ok) {
            bulk_recording_ = false;
            s->valid = false;
            terminal_failure_ = true;
            return false;
        }
    }
    bulk_recording_ = false;
    if (!submit(error)) {
        s->valid = false;
        return false;
    }
    size_t row_out = 0;
    for (size_t i = 0; i < segments.size(); ++i) {
        const uint64_t heads = uint64_t(segments[i].n_queries) * geo_.n_head;
        const uint64_t q_bytes = heads * 256 * 4;
        const uint64_t output_bytes = q_bytes + heads * 4;
        void* mapped = nullptr;
        D3D12_RANGE range{0, static_cast<SIZE_T>(output_bytes)};
        const HRESULT hr = bulk_transfers_[i].readback.resource->Map(0, &range, &mapped);
        if (FAILED(hr)) {
            s->valid = false;
            return fail(error, "Map q4 bulk readback", hr);
        }
        std::memcpy(out + row_out * geo_.n_head * 256, mapped, q_bytes);
        std::memcpy(lse + row_out * geo_.n_head, static_cast<const uint8_t*>(mapped) + q_bytes,
                    heads * 4);
        D3D12_RANGE empty{0, 0};
        bulk_transfers_[i].readback.resource->Unmap(0, &empty);
        row_out += segments[i].n_queries;
    }
    return true;
}

bool RemoteAttnEngine::attend_q4_bulk(uint32_t id, uint32_t layer, uint32_t n_tokens,
                                      const uint8_t* data, size_t bytes, float* out, float* lse,
                                      std::string* error) {
    return attend_native_rows(id, layer, n_tokens, RKVA_RANGE_STORAGE_Q4_0,
                              rkva_kvarn_range::q4_row_bytes(geo_.head_dim, geo_.n_head_kv),
                              data, bytes, out, lse, error);
}

bool RemoteAttnEngine::attend_q8_bulk(uint32_t id, uint32_t layer, uint32_t n_tokens,
                                      const uint8_t* data, size_t bytes, float* out, float* lse,
                                      std::string* error) {
    return attend_native_rows(id, layer, n_tokens, RKVA_RANGE_STORAGE_Q8_0,
                              rkva_kvarn_range::q8_row_bytes(geo_.head_dim, geo_.n_head_kv),
                              data, bytes, out, lse, error);
}

bool RemoteAttnEngine::attend_range(uint32_t id, uint32_t layer, uint32_t n_tokens,
                                    const int32_t* positions, const float* q, float* out,
                                    float* lse, std::string* error) {
    const bool prof = profiling_ && profile_ && !bulk_recording_;
    const bool bulk_prof = profiling_ && profile_ && bulk_recording_ && profile_->bulk.active;
    rp::Sample sample;
    SpanTimer timer;
    timer.start(prof, sample.cpu_ns);
    FailedCallGuard guard{prof ? &profile_->recorder.counters.failed_calls : nullptr};
    Session* s = find_session(id);
    if (!ready_ || !s || !s->valid || !s->range_configured || layer >= s->layer_kv.size() ||
        !n_tokens || !positions || !q || !out || !lse ||
        (s->range_storage == 3 && !s->range_ready[layer])) {
        return ::fail(error, "range attention: invalid arguments/session");
    }
    for (uint32_t t = 0; t < n_tokens; ++t) {
        if (positions[t] < 0) {
            return ::fail(error, "range attention: negative query position");
        }
    }
    // Revision 3: each query reads its own serial prefix. Only a contiguous
    // batch ending at the newest row whose every prefix is still held by the
    // stage slots / 256-row tail ring is accepted; nothing is mutated otherwise.
    const bool prefix =
        s->range_storage == 3 && rkva_kvarn_range::serial_prefix(s->range_descriptor);
    if (prefix && (!range_attn_kvarn4_prefix_pso_ ||
                   !rkva_kvarn_range::valid_prefix_queries(
                       positions, n_tokens, s->range_origin, s->range_counts[layer],
                       s->prefix_coverage[layer], s->precision_tail_tokens))) {
        return ::fail(error, "range attention: serial-prefix batch not contiguous/covered");
    }
    timer.mark(rp::kCpuValidate);
    const bool grow = range_scratch_tokens_ < n_tokens;
    if (!ensure_range_scratch(n_tokens, error)) {
        return false;
    }
    timer.mark(rp::kCpuAlloc);
    const uint64_t q_heads = uint64_t(n_tokens) * geo_.n_head;
    const uint64_t q_bytes = q_heads * geo_.head_dim * 4;
    const uint64_t positions_offset =
        s->range_storage == rp::kStorageKvarn4 ? 2 * q_bytes : q_bytes;
    const uint64_t out_bytes = q_bytes + q_heads * 4;
    const prof_clock::time_point t_map_start =
        bulk_prof ? prof_clock::now() : prof_clock::time_point{};
    void* mapped = nullptr;
    HRESULT hr = S_OK;
    if (!bulk_gpu_query_) {
        hr = range_query_.resource->Map(0, nullptr, &mapped);
        if (FAILED(hr)) {
            return fail(error, "Map range queries", hr);
        }
        std::memcpy(mapped, q, q_bytes);
        if (s->range_storage == 3) {
            float* q_rot = reinterpret_cast<float*>(static_cast<uint8_t*>(mapped) + q_bytes);
            for (uint64_t r = 0; r < q_heads; ++r) {
                rkva_kvarn_range::butterfly_wht256(q + r * 256, q_rot + r * 256);
            }
            std::memcpy(static_cast<uint8_t*>(mapped) + 2 * q_bytes, positions,
                        uint64_t(n_tokens) * 4);
        } else {
            std::memcpy(static_cast<uint8_t*>(mapped) + q_bytes, positions, uint64_t(n_tokens) * 4);
        }
        range_query_.resource->Unmap(0, nullptr);
    }
    timer.mark(rp::kCpuMapCopy);
    if (bulk_prof) {
        profile_->bulk.segments[profile_->bulk.segment_index].map_copy_query_ns +=
            elapsed_ns(t_map_start, prof_clock::now());
    }
    if (!begin(error)) {
        return false;
    }
    const prof_clock::time_point t_rec_start =
        bulk_prof ? prof_clock::now() : prof_clock::time_point{};
    const bool ts = prof && timestamp_heap_;
    if (ts) {
        timestamp(kTsAttnBegin);
    }
    if (bulk_prof && timestamp_heap_) {
        const uint32_t slot =
            1 + profile_->bulk.segment_index * kTsBulkSlotsPerSegment + kTsBulkAttnBegin;
        timestamp(slot);
    }
    const uint32_t active_count = s->range_counts[layer];
    // Shared tail after the readback copy has been recorded.
    auto finish = [&](uint32_t dispatches) -> bool {
        if (ts) {
            timestamp(kTsAttnCopy);
            resolve_timestamps(kTsAttnBegin, 4);
        }
        if (bulk_prof && timestamp_heap_) {
            const uint32_t slot =
                1 + profile_->bulk.segment_index * kTsBulkSlotsPerSegment + kTsBulkAttnCopy;
            timestamp(slot);
        }
        timer.mark(rp::kCpuRecord);
        if (bulk_recording_) {
            if (bulk_prof) {
                auto& seg = profile_->bulk.segments[profile_->bulk.segment_index];
                seg.record_attn_ns += elapsed_ns(t_rec_start, prof_clock::now());
                seg.dispatches = dispatches;
                seg.n_queries = n_tokens;
                seg.active_count = active_count;
                seg.query_bytes = q_bytes + uint64_t(n_tokens) * 4;
                seg.out_bytes = out_bytes;
            }
            return true; // final outer submit/readback owns publication
        }
        if (!submit(error)) {
            s->valid = false;
            return false;
        }
        timer.mark(rp::kCpuFenceWait);
        D3D12_RANGE range{0, static_cast<SIZE_T>(out_bytes)};
        hr = range_readback_.resource->Map(0, &range, &mapped);
        if (FAILED(hr)) {
            return fail(error, "Map range readback", hr);
        }
        std::memcpy(out, mapped, q_bytes);
        std::memcpy(lse, static_cast<uint8_t*>(mapped) + q_bytes, q_heads * 4);
        D3D12_RANGE empty{0, 0};
        range_readback_.resource->Unmap(0, &empty);
        timer.mark(rp::kCpuReadback);
        if (prof) {
            sample.kind = rp::kAttention;
            sample.storage = uint8_t(s->range_storage);
            sample.serial_prefix = prefix;
            sample.mode = uint8_t(q4_kernel_mode_);
            sample.alloc_grew = grow;
            sample.n_tokens = n_tokens;
            sample.rows = active_count;
            sample.bytes = q_bytes + uint64_t(n_tokens) * 4 + out_bytes;
            finish_profile_sample(sample, ts, kTsAttnBegin, 4, 1, dispatches);
            guard.done = true;
        }
        return true;
    };
    const bool optimized = range_attn_q4_opt_pso_ && range_reduce_opt_pso_ &&
                           range_opt_partials_.resource &&
                           rp::optimized_q4_supported(q4_kernel_mode_, s->range_storage, n_tokens,
                                                      geo_.n_head, geo_.n_head_kv, geo_.head_dim);
    if (optimized) {
        // GQA-shared decode/verification. nq1 (and mode 3, or a missing tile PSO):
        // independent per-query workgroups. nq 2..9 in modes 1/2: one workgroup
        // per query tile shares each K/V dequantization across the tile.
        // The bounded workspace covers kOptMaxQueries, including the anchor.
        // split_k == 1 writes OUT/LSE directly.
        uint32_t q_tile = rp::q4_mq_tile(q4_kernel_mode_, n_tokens);
        ID3D12PipelineState* attn_pso = range_attn_q4_opt_pso_.Get();
        if (q_tile && range_attn_q4_mq_pso_[q_tile / 4]) {
            attn_pso = range_attn_q4_mq_pso_[q_tile / 4].Get();
        } else {
            q_tile = 0;
        }
        uint32_t split_k = rp::optimized_split_k(active_count, q4_kernel_mode_);
        uint32_t rows_per_split = rp::split_rows(active_count, split_k);
        if (q_tile) {
            const rp::Q4MqSplit mq = rp::q4_mq_split(active_count, q4_kernel_mode_);
            split_k = mq.split_k;
            rows_per_split = mq.split_rows;
        }
        const uint32_t query_groups = q_tile ? (n_tokens + q_tile - 1) / q_tile : n_tokens;
        const bool direct = split_k == 1;
        Buffer& kv = s->layer_kv[layer];
        transition(kv, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(range_opt_partials_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        transition(range_output_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        const uint64_t q_addr = range_query_.resource->GetGPUVirtualAddress();
        const uint64_t kv_addr = kv.resource->GetGPUVirtualAddress();
        uint32_t scale;
        std::memcpy(&scale, &geo_.kq_scale, sizeof(scale));
        const uint32_t params[] = {n_tokens,       active_count,    geo_.n_head,     geo_.n_head_kv,
                                   geo_.head_dim,  scale,           s->range_origin, split_k,
                                   rows_per_split, direct ? 1u : 0u};
        cmd_->SetComputeRootSignature(range_attn_q4_opt_root_.Get());
        cmd_->SetPipelineState(attn_pso);
        cmd_->SetComputeRoot32BitConstants(0, 10, params, 0);
        cmd_->SetComputeRootShaderResourceView(1, q_addr);
        cmd_->SetComputeRootShaderResourceView(2, kv_addr);
        cmd_->SetComputeRootShaderResourceView(3, kv_addr + s->half_stride);
        cmd_->SetComputeRootShaderResourceView(4, q_addr + positions_offset);
        cmd_->SetComputeRootUnorderedAccessView(
            5, range_opt_partials_.resource->GetGPUVirtualAddress());
        cmd_->SetComputeRootUnorderedAccessView(6, range_output_.resource->GetGPUVirtualAddress());
        cmd_->Dispatch(geo_.n_head_kv, query_groups, split_k);
        if (!direct) {
            uav_barrier(range_opt_partials_);
            transition(range_opt_partials_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            if (ts) {
                timestamp(kTsAttnPass1);
            }
            if (bulk_prof && timestamp_heap_) {
                const uint32_t slot =
                    1 + profile_->bulk.segment_index * kTsBulkSlotsPerSegment + kTsBulkAttnPass1;
                timestamp(slot);
            }
            const uint32_t reduce_params[] = {n_tokens, geo_.n_head, geo_.head_dim, split_k};
            cmd_->SetComputeRootSignature(range_reduce_root_.Get());
            cmd_->SetPipelineState(range_reduce_opt_pso_.Get());
            cmd_->SetComputeRoot32BitConstants(0, 4, reduce_params, 0);
            cmd_->SetComputeRootShaderResourceView(
                1, range_opt_partials_.resource->GetGPUVirtualAddress());
            cmd_->SetComputeRootUnorderedAccessView(2,
                                                    range_output_.resource->GetGPUVirtualAddress());
            cmd_->Dispatch(geo_.n_head, n_tokens, 1);
        }
        uav_barrier(range_output_);
        transition(range_output_, D3D12_RESOURCE_STATE_COPY_SOURCE);
        if (ts) {
            if (direct) {
                timestamp(kTsAttnPass1); // pass1 includes the output barrier; reduce == 0
            }
            timestamp(kTsAttnReduce);
        }
        if (bulk_prof && timestamp_heap_) {
            const uint32_t base = 1 + profile_->bulk.segment_index * kTsBulkSlotsPerSegment;
            if (direct) {
                timestamp(base + kTsBulkAttnPass1);
            }
            timestamp(base + kTsBulkAttnReduce);
        }
        if (prof) {
            sample.optimized = 1;
            sample.direct_output = direct;
            sample.split_k = split_k;
            sample.groups = geo_.n_head_kv * query_groups * split_k;
            sample.q_tile = uint8_t(q_tile);
        }
        if (bulk_prof) {
            auto& seg = profile_->bulk.segments[profile_->bulk.segment_index];
            seg.optimized = 1;
            seg.direct_output = direct ? 1 : 0;
            seg.split_k = split_k;
            seg.groups = geo_.n_head_kv * query_groups * split_k;
            seg.q_tile = uint8_t(q_tile);
        }
        if (!bulk_gpu_query_) {
            cmd_->CopyBufferRegion(range_readback_.resource.Get(), 0, range_output_.resource.Get(),
                                   0, out_bytes);
        }
        return finish(direct ? 1u : 2u);
    }
    // F16 / Q8_0 / native KVarN4 optimized decode/verification kernels: same dispatch shape and
    // OUT/LSE/partials contract as the Q4 branch, storage-specific PSO. KVarN4
    // reads its descriptor-v1 records/stage/tail in place (no materialization).
    // F16/Q8_0 nq 2..9 in modes 1/2 use the multi-query tile PSO when present
    // (same mapping and 64-row split as Q4_0). Native KVarN4 revision 3 nq 2..9
    // in mode 2 uses the 2-query serial-prefix tile PSO when present (same
    // KVarN4 split). nq1, mode 3, KVarN4 revisions 1/2 and a missing tile PSO
    // keep the per-query kernel.
    const uint32_t storage = s->range_storage;
    const bool kvarn = storage == rp::kStorageKvarn4;
    // Revision 3 only ever uses its own serial-prefix PSOs.
    ID3D12PipelineState* fmt_pso = storage >= rp::kStorageCount ? nullptr
                                   : prefix ? range_attn_kvarn4_opt_prefix_pso_.Get()
                                            : range_attn_fmt_opt_pso_[storage].Get();
    const bool fmt_optimized =
        fmt_pso && range_attn_fmt_opt_root_ &&
        (kvarn ? range_reduce_kvarn4_pso_ : range_reduce_opt_pso_) &&
        range_opt_partials_.resource &&
        (kvarn ? rp::optimized_kvarn4_supported(q4_kernel_mode_, storage, n_tokens, geo_.n_head,
                                                geo_.n_head_kv, geo_.head_dim, s->range_origin,
                                                s->precision_tail_tokens)
               : rp::optimized_format_supported(q4_kernel_mode_, storage, n_tokens, geo_.n_head,
                                                geo_.n_head_kv, geo_.head_dim));
    if (fmt_optimized) {
        uint32_t q_tile = 0;
        bool decode_opt = false;
        if (kvarn && prefix) {
            if (n_tokens == 1 && rp::kvarn4_decode_opt(q4_kernel_mode_, wave_lane_min_) &&
                range_attn_kvarn4_decode_prefix_pso_) {
                fmt_pso = range_attn_kvarn4_decode_prefix_pso_.Get();
                decode_opt = true;
            } else {
                q_tile = rp::kvarn4_mq_tile(q4_kernel_mode_, n_tokens, s->precision_tail_tokens);
                ID3D12PipelineState* mq_pso =
                    q_tile == rp::kKvarn4MqTileMax
                        ? range_attn_kvarn4_mq4_pso_.Get()
                        : (q_tile == rp::kKvarn4MqTile ? range_attn_kvarn4_mq_pso_.Get() : nullptr);
                if (mq_pso) {
                    fmt_pso = mq_pso;
                } else {
                    q_tile = 0;
                }
            }
        } else if (!kvarn && !prefix) {
            q_tile = rp::format_mq_tile(q4_kernel_mode_, storage, n_tokens);
            if (q_tile && range_attn_fmt_mq_pso_[storage][q_tile / 4]) {
                fmt_pso = range_attn_fmt_mq_pso_[storage][q_tile / 4].Get();
            } else {
                q_tile = 0;
            }
        }
        uint32_t split_k = rp::optimized_split_k(active_count, q4_kernel_mode_);
        uint32_t rows = rp::split_rows(active_count, split_k);
        if (kvarn) {
            const rp::KvarnSplit ks =
                (n_tokens == 1 && decode_opt)
                    ? rp::kvarn4_decode_split(active_count, q4_kernel_mode_)
                    : (q_tile ? rp::kvarn4_mq_split(active_count, q4_kernel_mode_, q_tile)
                              : rp::kvarn4_split(active_count, q4_kernel_mode_));
            split_k = ks.split_k;
            rows = ks.split_rows;
        } else if (q_tile) {
            const rp::Q4MqSplit mq = rp::q4_mq_split(active_count, q4_kernel_mode_);
            split_k = mq.split_k;
            rows = mq.split_rows;
        }
        const uint32_t query_groups = q_tile ? (n_tokens + q_tile - 1) / q_tile : n_tokens;
        const bool direct = split_k == 1;
        Buffer& kv = s->layer_kv[layer];
        transition(kv, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(range_opt_partials_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        transition(range_output_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        const uint64_t q_addr = range_query_.resource->GetGPUVirtualAddress();
        const uint64_t kv_addr = kv.resource->GetGPUVirtualAddress();
        uint32_t scale;
        std::memcpy(&scale, &geo_.kq_scale, sizeof(scale));
        const uint32_t params[] = {n_tokens,
                                   active_count,
                                   geo_.n_head,
                                   geo_.n_head_kv,
                                   geo_.head_dim,
                                   scale,
                                   s->range_origin,
                                   split_k,
                                   rows,
                                   direct ? 1u : 0u,
                                   kvarn ? uint32_t(s->records_bytes) : 0u,
                                   kvarn ? s->precision_tail_tokens : 0u};
        cmd_->SetComputeRootSignature(range_attn_fmt_opt_root_.Get());
        cmd_->SetPipelineState(fmt_pso);
        cmd_->SetComputeRoot32BitConstants(0, 12, params, 0);
        cmd_->SetComputeRootShaderResourceView(1, q_addr);
        cmd_->SetComputeRootShaderResourceView(2, kv_addr);
        cmd_->SetComputeRootShaderResourceView(3, kv_addr + s->half_stride);
        cmd_->SetComputeRootShaderResourceView(4, q_addr + positions_offset);
        cmd_->SetComputeRootUnorderedAccessView(
            5, range_opt_partials_.resource->GetGPUVirtualAddress());
        cmd_->SetComputeRootUnorderedAccessView(6, range_output_.resource->GetGPUVirtualAddress());
        cmd_->Dispatch(geo_.n_head_kv, query_groups, split_k);
        if (!direct) {
            uav_barrier(range_opt_partials_);
            transition(range_opt_partials_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            if (ts) {
                timestamp(kTsAttnPass1);
            }
            if (bulk_prof && timestamp_heap_) {
                const uint32_t slot =
                    1 + profile_->bulk.segment_index * kTsBulkSlotsPerSegment + kTsBulkAttnPass1;
                timestamp(slot);
            }
            const uint32_t reduce_params[] = {n_tokens, geo_.n_head, geo_.head_dim, split_k};
            cmd_->SetComputeRootSignature(range_reduce_root_.Get());
            ID3D12PipelineState* reduce_pso = (kvarn && range_reduce_kvarn4_pso_)
                                                  ? range_reduce_kvarn4_pso_.Get()
                                                  : range_reduce_opt_pso_.Get();
            cmd_->SetPipelineState(reduce_pso);
            cmd_->SetComputeRoot32BitConstants(0, 4, reduce_params, 0);
            cmd_->SetComputeRootShaderResourceView(
                1, range_opt_partials_.resource->GetGPUVirtualAddress());
            cmd_->SetComputeRootUnorderedAccessView(2,
                                                    range_output_.resource->GetGPUVirtualAddress());
            cmd_->Dispatch(geo_.n_head, n_tokens, 1);
        }
        uav_barrier(range_output_);
        transition(range_output_, D3D12_RESOURCE_STATE_COPY_SOURCE);
        if (ts) {
            if (direct) {
                timestamp(kTsAttnPass1); // pass1 includes the output barrier; reduce == 0
            }
            timestamp(kTsAttnReduce);
        }
        if (bulk_prof && timestamp_heap_) {
            const uint32_t base = 1 + profile_->bulk.segment_index * kTsBulkSlotsPerSegment;
            if (direct) {
                timestamp(base + kTsBulkAttnPass1);
            }
            timestamp(base + kTsBulkAttnReduce);
        }
        if (prof) {
            sample.optimized = 1;
            sample.direct_output = direct;
            sample.split_k = split_k;
            sample.groups = geo_.n_head_kv * query_groups * split_k;
            sample.q_tile = uint8_t(q_tile);
            sample.decode_opt = uint8_t(decode_opt ? 1 : 0);
        }
        if (bulk_prof) {
            auto& seg = profile_->bulk.segments[profile_->bulk.segment_index];
            seg.optimized = 1;
            seg.direct_output = direct ? 1 : 0;
            seg.split_k = split_k;
            seg.groups = geo_.n_head_kv * query_groups * split_k;
            seg.q_tile = uint8_t(q_tile);
            seg.decode_opt = uint8_t(decode_opt ? 1 : 0);
        }
        if (!bulk_gpu_query_) {
            cmd_->CopyBufferRegion(range_readback_.resource.Get(), 0, range_output_.resource.Get(),
                                   0, out_bytes);
        }
        return finish(direct ? 1u : 2u);
    }
    const uint32_t split_k = std::max(1u, std::min(32u, (active_count + 255) / 256));
    Buffer& kv = s->layer_kv[layer];
    transition(kv, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(range_partials_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    transition(range_output_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    const uint64_t q_addr = range_query_.resource->GetGPUVirtualAddress();
    const uint64_t kv_addr = kv.resource->GetGPUVirtualAddress();
    uint32_t scale;
    std::memcpy(&scale, &geo_.kq_scale, sizeof(scale));
    const uint32_t params[] = {n_tokens,
                               active_count,
                               geo_.n_head,
                               geo_.n_head_kv,
                               geo_.head_dim,
                               scale,
                               s->range_origin,
                               split_k,
                               s->range_storage,
                               uint32_t(s->records_bytes),
                               s->precision_tail_tokens};
    cmd_->SetComputeRootSignature(range_attn_root_.Get());
    cmd_->SetPipelineState(prefix ? range_attn_kvarn4_prefix_pso_.Get()
                                  : range_attn_pso_[s->range_storage].Get());
    cmd_->SetComputeRoot32BitConstants(0, 11, params, 0);
    cmd_->SetComputeRootShaderResourceView(1, q_addr);
    cmd_->SetComputeRootShaderResourceView(2, kv_addr);
    cmd_->SetComputeRootShaderResourceView(3, kv_addr + s->half_stride);
    cmd_->SetComputeRootShaderResourceView(4, q_addr + positions_offset);
    cmd_->SetComputeRootUnorderedAccessView(5, range_partials_.resource->GetGPUVirtualAddress());
    cmd_->Dispatch(geo_.n_head, n_tokens, split_k);
    uav_barrier(range_partials_);
    transition(range_partials_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (ts) {
        timestamp(kTsAttnPass1);
    }
    if (bulk_prof && timestamp_heap_) {
        const uint32_t slot =
            1 + profile_->bulk.segment_index * kTsBulkSlotsPerSegment + kTsBulkAttnPass1;
        timestamp(slot);
    }
    const uint32_t reduce_params[] = {n_tokens, geo_.n_head, geo_.head_dim, split_k};
    cmd_->SetComputeRootSignature(range_reduce_root_.Get());
    cmd_->SetPipelineState(range_reduce_pso_.Get());
    cmd_->SetComputeRoot32BitConstants(0, 4, reduce_params, 0);
    cmd_->SetComputeRootShaderResourceView(1, range_partials_.resource->GetGPUVirtualAddress());
    cmd_->SetComputeRootUnorderedAccessView(2, range_output_.resource->GetGPUVirtualAddress());
    cmd_->Dispatch(geo_.n_head, n_tokens, 1);
    uav_barrier(range_output_);
    transition(range_output_, D3D12_RESOURCE_STATE_COPY_SOURCE);
    if (ts) {
        timestamp(kTsAttnReduce);
    }
    if (bulk_prof && timestamp_heap_) {
        const uint32_t slot =
            1 + profile_->bulk.segment_index * kTsBulkSlotsPerSegment + kTsBulkAttnReduce;
        timestamp(slot);
    }
    if (prof) {
        sample.split_k = split_k;
        sample.groups = geo_.n_head * n_tokens * split_k;
    }
    if (bulk_prof) {
        auto& seg = profile_->bulk.segments[profile_->bulk.segment_index];
        seg.optimized = 0;
        seg.direct_output = 0;
        seg.split_k = split_k;
        seg.groups = geo_.n_head * n_tokens * split_k;
        seg.q_tile = 0;
    }
    if (!bulk_gpu_query_) {
        cmd_->CopyBufferRegion(range_readback_.resource.Get(), 0, range_output_.resource.Get(), 0,
                               out_bytes);
    }
    return finish(2);
}

#include "rkva-disk-snapshot.inc"
#include "rkva-kvarn-bulk-f16.inc"
