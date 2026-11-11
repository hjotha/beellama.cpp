// IQ D3D12 matmul conformance harness (WARP, no Xbox needed).
//
// Runs one DXIL matmul kernel over a case file prepared by
// scripts/gen-d3d12-iq-cases.py and compares the GPU output with the
// ggml-dequantizer reference stored in that file. Root signature, constants,
// dispatch and block strides mirror src/bridge/ggml_d3d12.cpp, so a PASS is
// evidence for the checked-in DXIL blob, not for a hand-written emulation.
//
// Build (VS2022 x64 prompt, Windows SDK installed):
//   cl /EHsc /std:c++17 /O2 iq_harness.cpp /Fe:iq_harness.exe d3d12.lib dxgi.lib
//
// Run (DXIL from scripts/compile-iq-shaders.py):
//   iq_harness.exe iq2_s_t64.dxil case_00_iq2_s_17x2560_1.bin iq_tables.bin
//
// Cases with K >= 4096 must use the *_t128 blob (d3d12_mm_threads), the rest
// the *_t64 blob.

#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using Microsoft::WRL::ComPtr;

#define CHECK_HR(expr)                                                                     \
    do {                                                                                   \
        HRESULT hr_ = (expr);                                                              \
        if (FAILED(hr_)) {                                                                 \
            std::printf("FAIL %s hr=0x%08lx line=%d\n", #expr, (unsigned long)hr_,         \
                        __LINE__);                                                         \
            return 1;                                                                      \
        }                                                                                  \
    } while (0)

namespace {

std::vector<std::uint8_t> read_file(const char* path) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::printf("FAIL open %s\n", path);
        std::exit(2);
    }
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
    if (std::fread(data.data(), 1, data.size(), f) != data.size()) {
        std::printf("FAIL read %s\n", path);
        std::exit(2);
    }
    std::fclose(f);
    return data;
}

struct Case {
    std::uint32_t n = 0, k = 0, ncols = 0, row_bytes = 0;
    std::vector<std::uint8_t> weights;
    std::vector<float> x, y_ref;
};

Case load_case(const char* path) {
    const auto raw = read_file(path);
    Case c;
    std::uint32_t magic = 0;
    std::memcpy(&magic, raw.data(), 4);
    if (magic != 0x31435149u) {
        std::printf("FAIL bad case magic\n");
        std::exit(2);
    }
    std::memcpy(&c.n, raw.data() + 4, 4);
    std::memcpy(&c.k, raw.data() + 8, 4);
    std::memcpy(&c.ncols, raw.data() + 12, 4);
    std::memcpy(&c.row_bytes, raw.data() + 16, 4);
    std::size_t off = 20;
    const std::size_t wbytes = static_cast<std::size_t>(c.row_bytes) * c.n;
    c.weights.assign(raw.begin() + off, raw.begin() + off + wbytes);
    off += wbytes;
    const std::size_t xcount = static_cast<std::size_t>(c.k) * c.ncols;
    c.x.resize(xcount);
    std::memcpy(c.x.data(), raw.data() + off, xcount * 4);
    off += xcount * 4;
    const std::size_t ycount = static_cast<std::size_t>(c.n) * c.ncols;
    c.y_ref.resize(ycount);
    std::memcpy(c.y_ref.data(), raw.data() + off, ycount * 4);
    return c;
}

ComPtr<ID3D12Resource> make_buffer(ID3D12Device* device, UINT64 bytes, D3D12_HEAP_TYPE heap,
                                   D3D12_RESOURCE_STATES state, D3D12_RESOURCE_FLAGS flags) {
    D3D12_HEAP_PROPERTIES props = {};
    props.Type = heap;
    props.CreationNodeMask = 1;
    props.VisibleNodeMask = 1;
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    ComPtr<ID3D12Resource> res;
    if (FAILED(device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                               IID_PPV_ARGS(&res))))
        return nullptr;
    return res;
}

struct Staging {
    ComPtr<ID3D12Resource> res;
};

Staging stage(ID3D12Device* device, const void* data, std::size_t size) {
    Staging s;
    s.res = make_buffer(device, size, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ,
                        D3D12_RESOURCE_FLAG_NONE);
    void* p = nullptr;
    if (!s.res || FAILED(s.res->Map(0, nullptr, &p)))
        return {};
    std::memcpy(p, data, size);
    return s;
}

void barrier(ID3D12GraphicsCommandList* cl, ID3D12Resource* res, D3D12_RESOURCE_STATES from,
             D3D12_RESOURCE_STATES to) {
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = res;
    b.Transition.StateBefore = from;
    b.Transition.StateAfter = to;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cl->ResourceBarrier(1, &b);
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::printf("usage: %s <kernel.dxil> <case.bin> <iq_tables.bin>\n", argv[0]);
        return 2;
    }
    const auto dxil = read_file(argv[1]);
    const auto tables = read_file(argv[3]);
    const Case c = load_case(argv[2]);
    if (tables.size() != 11456) {
        std::printf("FAIL table size %zu\n", tables.size());
        return 2;
    }

    ComPtr<IDXGIFactory4> factory;
    CHECK_HR(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
    ComPtr<IDXGIAdapter> warp;
    CHECK_HR(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp)));
    ComPtr<ID3D12Device> device;
    CHECK_HR(D3D12CreateDevice(warp.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)));

    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> queue;
    CHECK_HR(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)));
    ComPtr<ID3D12CommandAllocator> alloc;
    CHECK_HR(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                            IID_PPV_ARGS(&alloc)));
    ComPtr<ID3D12GraphicsCommandList> cl;
    CHECK_HR(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr,
                                       IID_PPV_ARGS(&cl)));

    // Root signature: b0 (8 dwords), t0 weights, u0 out, u1 activations, t1 grids.
    D3D12_ROOT_PARAMETER p[5] = {};
    p[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    p[0].Constants.ShaderRegister = 0;
    p[0].Constants.Num32BitValues = 8;
    p[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    p[1].Descriptor.ShaderRegister = 0;
    p[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    p[2].Descriptor.ShaderRegister = 0;
    p[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    p[3].Descriptor.ShaderRegister = 1;
    p[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    p[4].Descriptor.ShaderRegister = 1;
    for (auto& param : p)
        param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rsd = {};
    rsd.NumParameters = 5;
    rsd.pParameters = p;
    rsd.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
    ComPtr<ID3DBlob> sig, sig_err;
    CHECK_HR(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &sig_err));
    ComPtr<ID3D12RootSignature> root;
    CHECK_HR(device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
                                         IID_PPV_ARGS(&root)));

    D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
    pd.pRootSignature = root.Get();
    pd.CS.pShaderBytecode = dxil.data();
    pd.CS.BytecodeLength = dxil.size();
    ComPtr<ID3D12PipelineState> pso;
    CHECK_HR(device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso)));

    const std::uint64_t wbytes = static_cast<std::uint64_t>(c.row_bytes) * c.n;
    const std::uint64_t xbytes = static_cast<std::uint64_t>(c.k) * c.ncols * 4;
    const std::uint64_t ybytes = static_cast<std::uint64_t>(c.n) * c.ncols * 4;
    ComPtr<ID3D12Resource> wbuf = make_buffer(device.Get(), wbytes + 16, D3D12_HEAP_TYPE_DEFAULT,
                                              D3D12_RESOURCE_STATE_COMMON,
                                              D3D12_RESOURCE_FLAG_NONE);
    ComPtr<ID3D12Resource> xbuf = make_buffer(device.Get(), xbytes, D3D12_HEAP_TYPE_DEFAULT,
                                              D3D12_RESOURCE_STATE_COMMON,
                                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    ComPtr<ID3D12Resource> ybuf = make_buffer(device.Get(), ybytes, D3D12_HEAP_TYPE_DEFAULT,
                                              D3D12_RESOURCE_STATE_COMMON,
                                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    ComPtr<ID3D12Resource> tbuf = make_buffer(device.Get(), tables.size(), D3D12_HEAP_TYPE_DEFAULT,
                                              D3D12_RESOURCE_STATE_COMMON,
                                              D3D12_RESOURCE_FLAG_NONE);
    ComPtr<ID3D12Resource> rbuf = make_buffer(device.Get(), ybytes, D3D12_HEAP_TYPE_READBACK,
                                              D3D12_RESOURCE_STATE_COPY_DEST,
                                              D3D12_RESOURCE_FLAG_NONE);
    if (!wbuf || !xbuf || !ybuf || !tbuf || !rbuf) {
        std::printf("FAIL buffer allocation\n");
        return 1;
    }
    Staging wstage = stage(device.Get(), c.weights.data(), c.weights.size());
    Staging xstage = stage(device.Get(), c.x.data(), xbytes);
    Staging tstage = stage(device.Get(), tables.data(), tables.size());
    if (!wstage.res || !xstage.res || !tstage.res) {
        std::printf("FAIL staging\n");
        return 1;
    }
    cl->CopyBufferRegion(wbuf.Get(), 0, wstage.res.Get(), 0, wbytes);
    cl->CopyBufferRegion(xbuf.Get(), 0, xstage.res.Get(), 0, xbytes);
    cl->CopyBufferRegion(tbuf.Get(), 0, tstage.res.Get(), 0, tables.size());
    barrier(cl.Get(), wbuf.Get(), D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    barrier(cl.Get(), tbuf.Get(), D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    barrier(cl.Get(), xbuf.Get(), D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    barrier(cl.Get(), ybuf.Get(), D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    // Same b0 layout as ggml_d3d12.cpp: n, k, nchunk, w_row_bytes, x_stride,
    // y_stride, ncols, pad1.
    const std::uint32_t constants[8] = {c.n, c.k, c.k / 256u, c.row_bytes, c.k, c.n, c.ncols, 0};
    cl->SetPipelineState(pso.Get());
    cl->SetComputeRootSignature(root.Get());
    cl->SetComputeRoot32BitConstants(0, 8, constants, 0);
    cl->SetComputeRootShaderResourceView(1, wbuf->GetGPUVirtualAddress());
    cl->SetComputeRootUnorderedAccessView(2, ybuf->GetGPUVirtualAddress());
    cl->SetComputeRootUnorderedAccessView(3, xbuf->GetGPUVirtualAddress());
    cl->SetComputeRootShaderResourceView(4, tbuf->GetGPUVirtualAddress());
    cl->Dispatch((c.n + 3) / 4, c.ncols, 1);

    barrier(cl.Get(), ybuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
    cl->CopyBufferRegion(rbuf.Get(), 0, ybuf.Get(), 0, ybytes);
    CHECK_HR(cl->Close());

    ID3D12CommandList* lists[] = {cl.Get()};
    queue->ExecuteCommandLists(1, lists);
    ComPtr<ID3D12Fence> fence;
    CHECK_HR(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    CHECK_HR(queue->Signal(fence.Get(), 1));
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    CHECK_HR(fence->SetEventOnCompletion(1, event));
    WaitForSingleObject(event, INFINITE);
    CloseHandle(event);

    void* mapped = nullptr;
    CHECK_HR(rbuf->Map(0, nullptr, &mapped));
    const auto* got = static_cast<const float*>(mapped);
    double max_ref = 0.0, max_diff = 0.0;
    for (std::size_t i = 0; i < c.y_ref.size(); ++i) {
        max_ref = std::max(max_ref, std::fabs(static_cast<double>(c.y_ref[i])));
        max_diff = std::max(max_diff, std::fabs(static_cast<double>(c.y_ref[i]) - got[i]));
    }
    const double rel = max_ref > 0.0 ? max_diff / max_ref : max_diff;
    std::printf("n=%u k=%u ncols=%u rel_err=%.3e max_ref=%.4f %s\n", c.n, c.k, c.ncols, rel,
                max_ref, rel <= 1e-4 ? "PASS" : "FAIL");
    return rel <= 1e-4 ? 0 : 1;
}
