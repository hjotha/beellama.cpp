// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

// Built only with the llama.cpp backend (it links ggml); the ORT-only UWP
// variant compiles this file to nothing.
#ifdef XLLAMA_USE_LLAMA

    #include "xllama/ggml_d3d12.h"
    #include "xllama/platform.h"

    #include <algorithm>
    #include <array>
    #include <chrono>
    #include <cmath>
    #include <cstdio>
    #include <cstring>
    #include <ctime>
    #include <random>
    #include <set>
    #include <string>
    #include <vector>

    #include "ggml-alloc.h"
    #include "ggml-backend.h"
    #include "ggml-cpu.h"

// Exact CPU reference kernel: ggml-cpu/vec.h declares it inside extern "C".
// Declared at global scope so the call links to the C symbol (a declaration
// inside namespace xllama would look for xllama::ggml_vec_silu_f32).
// Do NOT substitute libm exp.
extern "C" void ggml_vec_silu_f32(const int n, float* y, const float* x);
extern "C" void ggml_vec_swiglu_f32(const int n, float* y, const float* x, const float* g);

namespace xllama {

// --- Pure rules (host-tested) ---

bool d3d12_weight_type_supported(ggml_type t) {
    return t == GGML_TYPE_Q4_0 || t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K ||
           t == GGML_TYPE_Q6_K || t == GGML_TYPE_Q8_0;
}

static bool s_q8_enabled = true;
void d3d12_set_q8_enabled(bool enabled) {
    s_q8_enabled = enabled;
}
bool d3d12_q8_enabled() {
    return s_q8_enabled;
}

static int s_q6_columns = 1;
void d3d12_set_q6_columns(int columns) {
    s_q6_columns = columns == 0 || columns == 2 || columns == 4 ? columns : 1;
}
int d3d12_q6_columns() {
    return s_q6_columns;
}
bool d3d12_q6_allowlisted(int n, int k, int ncols) {
    return n == 248320 && k == 2560 && (ncols == 2 || ncols == 3 || ncols == 5);
}
int d3d12_q6_columns_for(int n, int k, int ncols) {
    if (!d3d12_q6_allowlisted(n, k, ncols))
        return 1;
    return s_q6_columns == 0 ? (ncols == 2 ? 2 : 4) : s_q6_columns;
}

static bool s_gdn_enabled = false;
static int s_gdn_min_tokens = 1;
void d3d12_set_gdn_enabled(bool enabled, int min_tokens) {
    s_gdn_enabled = enabled;
    s_gdn_min_tokens = min_tokens == 2 ? 2 : 1;
}
bool d3d12_gdn_enabled() {
    return s_gdn_enabled;
}

bool d3d12_gdn_supported(const ggml_tensor* op) {
    if (!d3d12_gdn_enabled() || !op || op->op != GGML_OP_GATED_DELTA_NET ||
        op->type != GGML_TYPE_F32 || op->src[6] || op->src[7] || op->src[8] ||
        op->op_params[1] != 0)
        return false;
    const int slots = op->op_params[0];
    if (slots < 1 || slots > 17)
        return false;
    for (int i = 0; i < 6; ++i)
        if (!op->src[i] || op->src[i]->type != GGML_TYPE_F32 || op->src[i]->ne[3] != 1)
            return false;
    const int64_t tokens = op->src[2]->ne[2];
    if (tokens < s_gdn_min_tokens || tokens > 64)
        return false;
    for (int i = 0; i < 3; ++i) {
        const ggml_tensor* t = op->src[i];
        if (t->ne[0] != 128 || t->ne[1] != (i == 2 ? 32 : 16) || t->ne[2] != tokens ||
            !ggml_is_contiguous_rows(t))
            return false;
        for (int j = 1; j < 3; ++j)
            if (t->nb[j] % sizeof(float) != 0 || t->nb[j] / sizeof(float) > UINT32_MAX)
                return false;
    }
    for (int i = 3; i < 5; ++i) {
        const ggml_tensor* t = op->src[i];
        if (t->ne[0] != 1 || t->ne[1] != 32 || t->ne[2] != tokens || !ggml_is_contiguous(t))
            return false;
    }
    const ggml_tensor* s = op->src[5];
    return s->ne[0] == 128 && s->ne[1] == 128 && s->ne[2] == 32 && ggml_is_contiguous(s) &&
           ggml_is_contiguous(op) && op->ne[0] == 4096 && op->ne[1] == tokens + slots * 128 &&
           op->ne[2] == 1 && op->ne[3] == 1;
}

// Host mirror of the HLSL eight-lane reduction and chronological state updates.
// This is a small-op reference check, never a CPU inference implementation.
void d3d12_gdn_emulate(const ggml_tensor* op, float* out) {
    if (!out || !d3d12_gdn_supported(op))
        return;
    const int tokens = static_cast<int>(op->src[2]->ne[2]);
    const int slots = op->op_params[0];
    auto at = [&](int src, int token, int head, int i) {
        const ggml_tensor* t = op->src[src];
        const char* data = static_cast<const char*>(t->data);
        return *reinterpret_cast<const float*>(data + token * t->nb[2] + head * t->nb[1] + i * 4);
    };
    auto reduce = [](float p[8]) {
        for (int stride = 4; stride > 0; stride >>= 1)
            for (int lane = 0; lane < stride; ++lane)
                p[lane] += p[lane + stride];
        return p[0];
    };
    std::vector<float> state(128 * 128);
    const float* initial = static_cast<const float*>(op->src[5]->data);
    const int s_off = 4096 * tokens;
    const float scale = 1.0f / std::sqrt(128.0f);
    for (int head = 0; head < 32; ++head) {
        std::memcpy(state.data(), initial + head * 128 * 128, state.size() * sizeof(float));
        for (int t = 0; t < tokens; ++t) {
            const float decay = std::exp(at(3, t, head, 0));
            const float beta = at(4, t, head, 0);
            for (int col = 0; col < 128; ++col) {
                float kv[8] = {}, attn[8] = {};
                float* s = state.data() + col * 128;
                for (int i = 0; i < 128; ++i) {
                    s[i] *= decay;
                    kv[i % 8] += s[i] * at(1, t, head % 16, i);
                }
                const float delta = (at(2, t, head, col) - reduce(kv)) * beta;
                for (int i = 0; i < 128; ++i) {
                    s[i] += at(1, t, head % 16, i) * delta;
                    attn[i % 8] += s[i] * at(0, t, head % 16, i);
                }
                out[(t * 32 + head) * 128 + col] = reduce(attn) * scale;
            }
            const int slot = tokens - 1 - t;
            if ((slots > 1 && slot < slots) || (slots == 1 && t + 1 == tokens))
                std::memcpy(out + s_off + (slots == 1 ? 0 : slot) * 524288 + head * 128 * 128,
                            state.data(), state.size() * sizeof(float));
        }
    }
}

namespace {

struct GdnEnabledGuard {
    bool previous = d3d12_gdn_enabled();
    GdnEnabledGuard() {
        d3d12_set_gdn_enabled(true);
    }
    ~GdnEnabledGuard() {
        d3d12_set_gdn_enabled(previous);
    }
};

// Only small synthetic ops: the same inputs feed the real CPU kernel and
// either the GPU or its eight-lane host mirror. No model/context is loaded.
D3d12SelftestRow run_gdn_case(ggml_backend_t backend, ggml_backend_buffer_type_t buft,
                              ggml_backend_t cpu, int tokens, int slots, int pad,
                              std::vector<float>* result) {
    D3d12SelftestRow row;
    row.type = "gdn_k" + std::to_string(slots);
    row.n = 128;
    row.k = 32;
    row.ncols = tokens;
    row.pads = std::to_string(pad) + "/" + std::to_string(pad ? pad + 2 : 0) + "/" +
               std::to_string(pad ? pad + 4 : 0); // Q/K/V head padding, floats
    ggml_init_params ip = {};
    ip.mem_size = 16 * ggml_tensor_overhead() + 2 * ggml_graph_overhead();
    ip.no_alloc = true;
    ggml_context* ctx = ggml_init(ip);
    if (!ctx) {
        row.error = "GDN metadata allocation failed";
        return row;
    }
    ggml_tensor* src[6] = {ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 16, tokens, 1),
                           ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 16, tokens, 1),
                           ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 32, tokens, 1),
                           ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 32, tokens, 1),
                           ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 32, tokens, 1),
                           ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 128, 32, 1)};
    for (int i = 0; pad && i < 3; ++i) {
        src[i]->nb[1] = (128 + pad + 2 * i) * sizeof(float);
        src[i]->nb[2] = src[i]->nb[1] * src[i]->ne[1];
        src[i]->nb[3] = src[i]->nb[2] * tokens;
    }
    ggml_tensor* test =
        ggml_gated_delta_net(ctx, src[0], src[1], src[2], src[3], src[4], src[5], slots);
    ggml_tensor* ref =
        ggml_gated_delta_net(ctx, src[0], src[1], src[2], src[3], src[4], src[5], slots);
    ggml_cgraph* gt = ggml_new_graph(ctx);
    ggml_cgraph* gr = ggml_new_graph(ctx);
    ggml_build_forward_expand(gt, test);
    ggml_build_forward_expand(gr, ref);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    auto cleanup = [&] {
        if (buf)
            ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    };
    if (!buf || !d3d12_gdn_supported(test)) {
        row.error = "GDN allocation/contract failed";
        cleanup();
        return row;
    }
    constexpr float sentinel = 1e30f;
    std::array<std::vector<float>, 6> inputs;
    for (int j = 0; j < 6; ++j) {
        inputs[j].assign(ggml_nbytes(src[j]) / sizeof(float), sentinel);
        std::mt19937 rng(1701u + j); // tensor-specific, independent of T/K/padding
        std::uniform_real_distribution<float> uni(-0.05f, 0.05f);
        for (int64_t t = 0; t < src[j]->ne[2]; ++t)
            for (int64_t h = 0; h < src[j]->ne[1]; ++h)
                for (int64_t i = 0; i < src[j]->ne[0]; ++i) {
                    float v = uni(rng);
                    if (j == 3)
                        v = -0.3f - std::fabs(v);
                    if (j == 4)
                        v += 0.5f;
                    inputs[j][(t * src[j]->nb[2] + h * src[j]->nb[1]) / 4 + i] = v;
                }
        ggml_backend_tensor_set(src[j], inputs[j].data(), 0, ggml_nbytes(src[j]));
    }
    std::vector<float> got(ggml_nelements(test), sentinel), want(got);
    ggml_backend_tensor_set(test, got.data(), 0, ggml_nbytes(test));
    ggml_backend_tensor_set(ref, want.data(), 0, ggml_nbytes(ref));
    std::vector<double> cpu_times, gpu_times, wall_times;
    ggml_status status = GGML_STATUS_SUCCESS;
    const int runs = backend ? 4 : 1; // GPU: warmup + three paired samples; host mirror: one check
    for (int run = 0; run < runs; ++run) {
        const auto tc = std::chrono::steady_clock::now();
        status = ggml_backend_graph_compute(cpu, gr);
        const double cpu_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc)
                .count();
        if (status != GGML_STATUS_SUCCESS)
            break;
        const double gpu0 = d3d12_gpu_ms();
        const auto tg = std::chrono::steady_clock::now();
        if (backend)
            status = ggml_backend_graph_compute(backend, gt);
        else
            d3d12_gdn_emulate(test, got.data());
        const double wall_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tg)
                .count();
        if (status != GGML_STATUS_SUCCESS)
            break;
        if (!backend || run > 0) {
            cpu_times.push_back(cpu_ms);
            wall_times.push_back(wall_ms);
            gpu_times.push_back(d3d12_gpu_ms() - gpu0);
        }
    }
    if (status != GGML_STATUS_SUCCESS) {
        row.error = "GDN CPU/GPU graph compute failed";
        cleanup();
        return row;
    }
    row.d3d12_ran = backend != nullptr;
    row.cpu_ms = d3d12_block_median(cpu_times);
    row.wall_ms = d3d12_block_median(wall_times);
    row.gpu_ms = d3d12_block_median(gpu_times);
    row.wall_ms_range = d3d12_block_range(wall_times);
    row.gpu_ms_range = d3d12_block_range(gpu_times);
    if (backend)
        ggml_backend_tensor_get(test, got.data(), 0, ggml_nbytes(test));
    ggml_backend_tensor_get(ref, want.data(), 0, ggml_nbytes(ref));
    const std::size_t valid = 4096u * tokens + 524288u * std::min(tokens, slots);
    bool intact = true;
    double max_ref = 0, max_diff = 0;
    for (std::size_t i = 0; i < valid; ++i) {
        intact &= std::isfinite(got[i]) && std::isfinite(want[i]);
        max_ref = std::max(max_ref, std::fabs(static_cast<double>(want[i])));
        max_diff = std::max(max_diff, std::fabs(static_cast<double>(got[i]) - want[i]));
    }
    for (std::size_t i = valid; i < got.size(); ++i)
        intact &= got[i] == sentinel && want[i] == sentinel; // old unwritten slots are caller-owned
    for (int j = 0; j < 6; ++j) {
        std::vector<float> after(inputs[j].size());
        ggml_backend_tensor_get(src[j], after.data(), 0, ggml_nbytes(src[j]));
        intact &= after == inputs[j]; // inputs and their padding must stay untouched
    }
    row.rel_err = max_ref > 0 ? max_diff / max_ref : max_diff;
    row.ok = intact && row.rel_err <= 1e-4;
    if (!row.ok)
        row.error = "GDN mismatch/input mutation/unwritten snapshot overwrite";
    if (result)
        *result = std::move(got);
    cleanup();
    return row;
}

void run_gdn_cases(ggml_backend_t backend, ggml_backend_buffer_type_t buft,
                   std::vector<D3d12SelftestRow>* out) {
    GdnEnabledGuard guard;
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_threadpool_params tp = ggml_threadpool_params_default(6);
    ggml_threadpool_t pool = ggml_threadpool_new(&tp);
    if (!cpu || !pool) {
        D3d12SelftestRow row;
        row.type = "gdn";
        row.error = "GDN CPU reference backend/threadpool unavailable";
        out->push_back(row);
        if (cpu)
            ggml_backend_free(cpu);
        if (pool)
            ggml_threadpool_free(pool);
        return;
    }
    ggml_backend_cpu_set_n_threads(cpu, 6);
    ggml_backend_cpu_set_threadpool(cpu, pool);
    std::vector<float> baseline;
    auto check = [&](int tokens, int slots, int pad) {
        std::vector<float> got;
        D3d12SelftestRow row = run_gdn_case(backend, buft, cpu, tokens, slots, pad, &got);
        if (row.ok && baseline.empty())
            baseline = got; // first case: T=5, K=5
        if (row.ok && !baseline.empty()) {
            bool same =
                std::equal(got.begin(), got.begin() + 4096 * std::min(tokens, 5), baseline.begin());
            if (tokens <= 5) {
                for (int j = 0; j < std::min(tokens, slots); ++j) {
                    const int state_slot = slots == 1 ? 5 - tokens : 5 - tokens + j;
                    const std::size_t a = 4096u * tokens + 524288u * j;
                    const std::size_t b = 4096u * 5 + 524288u * state_slot;
                    same &=
                        std::equal(got.begin() + a, got.begin() + a + 524288, baseline.begin() + b);
                }
            }
            if (!same) {
                row.ok = false;
                row.error = "GDN prefix/snapshot changed across T/K/padding";
            }
        }
        char msg[320];
        std::snprintf(msg, sizeof(msg),
                      "[xllama] GDN_SELFTEST T=%d K=%d pads=%s gpu=%d cpu6_ms=%.4f wall_ms=%.4f "
                      "gpu_ms=%.4f rel_err=%.8g snapshots=%d ok=%d\n",
                      tokens, slots, row.pads.c_str(), row.d3d12_ran ? 1 : 0, row.cpu_ms,
                      row.wall_ms, row.gpu_ms, row.rel_err, std::min(tokens, slots),
                      row.ok ? 1 : 0);
        log_output(msg);
        out->push_back(std::move(row));
    };
    for (int slots : {5, 1})
        for (int tokens : {5, 1, 2, 3, 4, 64})
            check(tokens, slots, 0);
    check(3, 5, 3);
    check(4, 5, 3);
    ggml_backend_free(cpu);
    ggml_threadpool_free(pool);
}

} // namespace

D3d12Dispatch d3d12_mm_dispatch(std::int64_t n, std::int64_t ncols) {
    D3d12Dispatch d;
    if (n <= 0 || ncols <= 0)
        return d;
    const std::int64_t gx = (n + kD3d12MmvRows - 1) / kD3d12MmvRows;
    d.ok = gx <= kD3d12MaxGroups && ncols <= kD3d12MaxGroups;
    if (d.ok) {
        d.groups_x = static_cast<std::uint32_t>(gx);
        d.groups_y = static_cast<std::uint32_t>(ncols);
    }
    return d;
}

D3d12Dispatch d3d12_mm_dispatch_2col(std::int64_t n, std::int64_t ncols) {
    D3d12Dispatch d;
    if (n <= 0 || ncols <= 0)
        return d;
    const std::int64_t gx = (n + kD3d12MmvRows - 1) / kD3d12MmvRows;
    const std::int64_t gy = (ncols + 1) / 2;
    d.ok = gx <= kD3d12MaxGroups && gy <= kD3d12MaxGroups;
    if (d.ok) {
        d.groups_x = static_cast<std::uint32_t>(gx);
        d.groups_y = static_cast<std::uint32_t>(gy);
    }
    return d;
}

D3d12Dispatch d3d12_mm_dispatch_4col(std::int64_t n, std::int64_t ncols) {
    if (ncols <= 0)
        return {};
    return d3d12_mm_dispatch_2col(n, ncols / 2 + ncols % 2);
}

// Explicit allowlist for variant 2 (plan 004): the rev57 same-run pair
// winners. Five q4_k shapes at B=2/3/5; each won with pair-delta ranges far
// from zero (1.18x-1.59x). Deliberately absent: B=1 (single column, nothing
// to tile), B=4 and prefill widths (unmeasured), 1024x2560 and smaller
// (measured regressions), every other type.
bool d3d12_2col_allowlisted(ggml_type t, int n, int k, int ncols) {
    if (t != GGML_TYPE_Q4_K)
        return false;
    if (ncols != 2 && ncols != 3 && ncols != 5)
        return false;
    static const int kShapes[][2] = {
        {2560, 4096}, {4096, 2560}, {8192, 2560}, {2560, 9216}, {9216, 2560},
    };
    for (const auto& s : kShapes)
        if (n == s[0] && k == s[1])
            return true;
    return false;
}

int d3d12_mm_threads(std::int64_t k) {
    return k / kD3d12Chunk >= kD3d12LongKChunks ? kD3d12MmvThreadsLong : kD3d12MmvThreadsShort;
}

bool d3d12_mm_supported(const D3d12MatmulDesc& d) {
    return d3d12_weight_type_supported(d.src0_type) &&
           (d.src0_type != GGML_TYPE_Q8_0 || d3d12_q8_enabled()) && d.src0_in_weight_buffer &&
           d.src0_contiguous && d.src1_contiguous && d.src1_type == GGML_TYPE_F32 &&
           d.dst_type == GGML_TYPE_F32 && d.ne00 > 0 && d.ne00 % kD3d12Chunk == 0 &&
           d.ne00 == d.ne10 && d.ne02 == 1 && d.ne03 == 1 && d.ne12 == 1 && d.ne13 == 1 &&
           d.ne01 > 0 && (d.ne11 == 0 || d3d12_mm_dispatch(d.ne01, d.ne11).ok);
}

// --- Host emulation of shaders/ggml_d3d12_mmv_*.hlsl ---

namespace {

// Little-endian loads with the shaders' 2-byte-aligned dword trick.
std::uint32_t ld32(const std::uint8_t* w, std::uint32_t a) {
    auto dw = [&](std::uint32_t at) {
        std::uint32_t v;
        std::memcpy(&v, w + at, 4);
        return v;
    };
    const std::uint32_t b = a & ~3u;
    const std::uint32_t lo = dw(b);
    if ((a & 3u) == 0u)
        return lo;
    return (lo >> 16) | (dw(b + 4u) << 16);
}

std::uint32_t ld16(const std::uint8_t* w, std::uint32_t a) {
    std::uint32_t v;
    std::memcpy(&v, w + (a & ~3u), 4);
    return (a & 2u) != 0u ? (v >> 16) : (v & 0xffffu);
}

float h2f(std::uint32_t h) {
    return ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(h));
}

float nib(std::uint32_t word, std::uint32_t i, std::uint32_t shift) {
    return static_cast<float>((word >> (8u * i + shift)) & 0xFu);
}

float sbyte(std::uint32_t word, std::uint32_t i) {
    return static_cast<float>(static_cast<std::int8_t>((word >> (8u * i)) & 0xffu));
}

float bit1(std::uint32_t word, std::uint32_t i, std::uint32_t bit) {
    return static_cast<float>((word >> (8u * i + bit)) & 1u);
}

void q4k_scale_min(std::uint32_t j, const std::uint8_t* sc, float* d, float* m) {
    std::uint32_t dd, mm;
    if (j < 4u) {
        dd = sc[j] & 63u;
        mm = sc[j + 4u] & 63u;
    } else {
        dd = (sc[j + 4u] & 0xFu) | ((sc[j - 4u] >> 6) << 4);
        mm = (sc[j + 4u] >> 4) | ((sc[j] >> 6) << 4);
    }
    *d = static_cast<float>(dd);
    *m = static_cast<float>(mm);
}

// One thread's contribution for one row and one 256-element chunk.
float thread_chunk(ggml_type t, const std::uint8_t* w, std::uint32_t row_off, std::uint32_t blk,
                   std::uint32_t itid, const float* x) {
    const float* xe = x + blk * 256u;
    if (t == GGML_TYPE_Q4_0) {
        const std::uint32_t b = itid >> 1, h = itid & 1u;
        const std::uint32_t e_lo = 32u * b + 8u * h, e_hi = e_lo + 16u;
        const std::uint32_t bb = row_off + (blk * 8u + b) * 18u;
        const float d = h2f(ld16(w, bb));
        const std::uint32_t q0 = ld32(w, bb + 2u + 8u * h), q1 = ld32(w, bb + 6u + 8u * h);
        float dotq = 0.f, sx = 0.f;
        for (std::uint32_t i = 0; i < 4; ++i) {
            dotq += nib(q0, i, 0) * xe[e_lo + i] + nib(q1, i, 0) * xe[e_lo + 4 + i] +
                    nib(q0, i, 4) * xe[e_hi + i] + nib(q1, i, 4) * xe[e_hi + 4 + i];
            sx += xe[e_lo + i] + xe[e_lo + 4 + i] + xe[e_hi + i] + xe[e_hi + 4 + i];
        }
        return d * (dotq - 8.f * sx);
    }
    if (t == GGML_TYPE_Q4_K) {
        const std::uint32_t il = itid >> 2, ir = itid & 3u;
        const std::uint32_t e_lo = il * 64u + ir * 8u, e_hi = e_lo + 32u;
        const std::uint32_t bb = row_off + blk * 144u;
        const float d = h2f(ld16(w, bb)), dmin = h2f(ld16(w, bb + 2u));
        float sc, m;
        q4k_scale_min(2u * il, w + bb + 4u, &sc, &m);
        const float d1 = d * sc, m1 = dmin * m;
        q4k_scale_min(2u * il + 1u, w + bb + 4u, &sc, &m);
        const float d2 = d * sc, m2 = dmin * m;
        const std::uint32_t qa = ld32(w, bb + 16u + il * 32u + ir * 8u);
        const std::uint32_t qb = ld32(w, bb + 20u + il * 32u + ir * 8u);
        float lo = 0.f, hi = 0.f, sxl = 0.f, sxh = 0.f;
        for (std::uint32_t i = 0; i < 4; ++i) {
            lo += nib(qa, i, 0) * xe[e_lo + i] + nib(qb, i, 0) * xe[e_lo + 4 + i];
            hi += nib(qa, i, 4) * xe[e_hi + i] + nib(qb, i, 4) * xe[e_hi + 4 + i];
            sxl += xe[e_lo + i] + xe[e_lo + 4 + i];
            sxh += xe[e_hi + i] + xe[e_hi + 4 + i];
        }
        return d1 * lo - m1 * sxl + d2 * hi - m2 * sxh;
    }
    if (t == GGML_TYPE_Q5_K) {
        // block_q5_K: half d, half dmin, scales[12], qh[32], qs[128]; 176 B. Same
        // lane map as Q4_K plus the high bit, which dequantize_row_q5_K reads
        // from qh[element & 31] with mask 1 << (element >> 5) — the 32 qh bytes
        // serve all four sub-blocks, so one bit index covers a thread's eight
        // weights.
        const std::uint32_t il = itid >> 2, ir = itid & 3u;
        const std::uint32_t e_lo = il * 64u + ir * 8u, e_hi = e_lo + 32u;
        const std::uint32_t bb = row_off + blk * 176u;
        const float d = h2f(ld16(w, bb)), dmin = h2f(ld16(w, bb + 2u));
        float sc, m;
        q4k_scale_min(2u * il, w + bb + 4u, &sc, &m);
        const float d1 = d * sc, m1 = dmin * m;
        q4k_scale_min(2u * il + 1u, w + bb + 4u, &sc, &m);
        const float d2 = d * sc, m2 = dmin * m;
        const std::uint32_t qa = ld32(w, bb + 48u + il * 32u + ir * 8u);
        const std::uint32_t qb = ld32(w, bb + 52u + il * 32u + ir * 8u);
        const std::uint32_t ha = ld32(w, bb + 16u + ir * 8u);
        const std::uint32_t hb = ld32(w, bb + 20u + ir * 8u);
        float lo = 0.f, hi = 0.f, sxl = 0.f, sxh = 0.f;
        for (std::uint32_t i = 0; i < 4; ++i) {
            lo += (nib(qa, i, 0) + 16.f * bit1(ha, i, 2u * il)) * xe[e_lo + i] +
                  (nib(qb, i, 0) + 16.f * bit1(hb, i, 2u * il)) * xe[e_lo + 4 + i];
            hi += (nib(qa, i, 4) + 16.f * bit1(ha, i, 2u * il + 1u)) * xe[e_hi + i] +
                  (nib(qb, i, 4) + 16.f * bit1(hb, i, 2u * il + 1u)) * xe[e_hi + 4 + i];
            sxl += xe[e_lo + i] + xe[e_lo + 4 + i];
            sxh += xe[e_hi + i] + xe[e_hi + 4 + i];
        }
        return d1 * lo - m1 * sxl + d2 * hi - m2 * sxh;
    }
    if (t == GGML_TYPE_Q8_0) {
        const std::uint32_t b = itid >> 1, h = itid & 1u;
        const std::uint32_t e = 32u * b + 16u * h;
        const std::uint32_t bb = row_off + (blk * 8u + b) * 34u;
        const float d = h2f(ld16(w, bb));
        float dotq = 0.f;
        for (std::uint32_t j = 0; j < 4; ++j) {
            const std::uint32_t q = ld32(w, bb + 2u + 16u * h + 4u * j);
            for (std::uint32_t i = 0; i < 4; ++i)
                dotq += sbyte(q, i) * xe[e + 4u * j + i];
        }
        return d * dotq;
    }
    // Q6_K
    const std::uint32_t v = itid >> 3, l0 = 4u * (itid & 7u), is = l0 >> 4;
    const std::uint32_t e = 128u * v + l0;
    const std::uint32_t bb = row_off + blk * 210u;
    const std::uint32_t qla = ld32(w, bb + 64u * v + l0);
    const std::uint32_t qlb = ld32(w, bb + 64u * v + 32u + l0);
    const std::uint32_t qh = ld32(w, bb + 128u + 32u * v + l0);
    const std::uint32_t s0 = ld32(w, bb + 192u + 8u * v), s1 = ld32(w, bb + 196u + 8u * v);
    const float d = h2f(ld16(w, bb + 208u));
    auto q6 = [](std::uint32_t ql, std::uint32_t qhw, std::uint32_t i, std::uint32_t qs,
                 std::uint32_t hs) {
        const std::uint32_t lo4 = (ql >> (8u * i + qs)) & 0xFu;
        const std::uint32_t hi2 = (qhw >> (8u * i + hs)) & 3u;
        return static_cast<float>(static_cast<int>(lo4 | (hi2 << 4))) - 32.f;
    };
    float t1 = 0.f, t2 = 0.f, t3 = 0.f, t4 = 0.f;
    for (std::uint32_t i = 0; i < 4; ++i) {
        t1 += q6(qla, qh, i, 0, 0) * xe[e + i];
        t2 += q6(qlb, qh, i, 0, 2) * xe[e + 32 + i];
        t3 += q6(qla, qh, i, 4, 4) * xe[e + 64 + i];
        t4 += q6(qlb, qh, i, 4, 6) * xe[e + 96 + i];
    }
    return d * (sbyte(s0, is) * t1 + sbyte(s0, is + 2) * t2 + sbyte(s1, is) * t3 +
                sbyte(s1, is + 2) * t4);
}

} // namespace

void d3d12_mmv_emulate(ggml_type t, const std::uint8_t* w, std::size_t w_row_bytes, const float* x,
                       std::size_t x_stride, float* y, std::size_t y_stride, int n, int k,
                       int ncols) {
    if (!w || !x || !y || n <= 0 || k <= 0 || k % kD3d12Chunk != 0 || ncols <= 0 ||
        !d3d12_weight_type_supported(t))
        return;
    const std::uint32_t nchunk = static_cast<std::uint32_t>(k / kD3d12Chunk);
    const std::uint32_t threads = static_cast<std::uint32_t>(d3d12_mm_threads(k));
    const std::uint32_t in_flight = threads / 16u;
    float acc[kD3d12MmvThreadsLong];
    for (int c = 0; c < ncols; ++c) {
        const float* xc = x + static_cast<std::size_t>(c) * x_stride;
        for (int row = 0; row < n; ++row) {
            const std::uint32_t row_off = static_cast<std::uint32_t>(row * w_row_bytes);
            for (std::uint32_t tid = 0; tid < threads; ++tid) {
                float a = 0.f;
                for (std::uint32_t blk = tid >> 4; blk < nchunk; blk += in_flight)
                    a += thread_chunk(t, w, row_off, blk, tid & 15u, xc);
                acc[tid] = a;
            }
            for (int stride = static_cast<int>(threads) / 2; stride > 0; stride >>= 1)
                for (int i = 0; i < stride; ++i)
                    acc[i] += acc[i + stride];
            y[static_cast<std::size_t>(c) * y_stride + row] = acc[0];
        }
    }
}

// --- Bounded weight generation (plan 004 shape-cost bench) ---

void d3d12_q6_tile_emulate(const std::uint8_t* w, std::size_t w_row_bytes, const float* x,
                           std::size_t x_stride, float* y, std::size_t y_stride, int n, int k,
                           int ncols, int columns) {
    if (!w || !x || !y || n <= 0 || k <= 0 || k % kD3d12Chunk != 0 || ncols <= 0 ||
        (columns != 2 && columns != 4))
        return;
    const int threads = d3d12_mm_threads(k);
    const int in_flight = threads / 16;
    for (int col0 = 0; col0 < ncols; col0 += columns) {
        for (int row0 = 0; row0 < n; row0 += kD3d12MmvRows) {
            float acc[4][kD3d12MmvRows][kD3d12MmvThreadsLong] = {};
            for (int tid = 0; tid < threads; ++tid) {
                for (int blk = tid / 16; blk < k / kD3d12Chunk; blk += in_flight) {
                    for (int r = 0; r < kD3d12MmvRows; ++r) {
                        const auto offset =
                            static_cast<std::uint32_t>(std::min(row0 + r, n - 1) * w_row_bytes);
                        for (int c = 0; c < columns && col0 + c < ncols; ++c)
                            acc[c][r][tid] += thread_chunk(GGML_TYPE_Q6_K, w, offset, blk, tid % 16,
                                                           x + (col0 + c) * x_stride);
                    }
                }
            }
            for (int c = 0; c < columns; ++c) {
                for (int stride = threads / 2; stride > 0; stride >>= 1)
                    for (int tid = 0; tid < stride; ++tid)
                        for (int r = 0; r < kD3d12MmvRows; ++r)
                            acc[c][r][tid] += acc[c][r][tid + stride];
                if (col0 + c < ncols)
                    for (int r = 0; r < kD3d12MmvRows && row0 + r < n; ++r)
                        y[(col0 + c) * y_stride + row0 + r] = acc[c][r][0];
            }
        }
    }
}

bool d3d12_gen_weights(ggml_type t, int n, int k, std::mt19937& rng, void* dst) {
    if (!dst || n <= 0 || k <= 0 || !d3d12_weight_type_supported(t))
        return false;
    const std::int64_t blck = ggml_blck_size(t);
    if (blck <= 0 || k % blck != 0)
        return false;
    const std::size_t row_bytes = ggml_row_size(t, k);
    if (row_bytes == 0)
        return false;
    const std::size_t rows = std::max<std::size_t>(
        1, kD3d12GenScratchBytes / (static_cast<std::size_t>(k) * sizeof(float)));
    std::vector<float> src(rows * static_cast<std::size_t>(k));
    auto* out = static_cast<std::uint8_t*>(dst);
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    for (int r = 0; r < n; r += static_cast<int>(rows)) {
        const std::size_t nr = std::min<std::size_t>(rows, static_cast<std::size_t>(n - r));
        const std::size_t ne = nr * static_cast<std::size_t>(k);
        // Row-major draw order, chunk after chunk: the concatenated stream is
        // the same count in the same order as one n*k vector drawn at once.
        for (std::size_t i = 0; i < ne; ++i)
            src[i] = uni(rng);
        // start=0 with the dst pre-advanced to row r: ggml_quantize_chunk
        // writes at dst + (start/n_per_row) * row_size, so this is the same
        // write the one-shot call would make for rows [r, r+nr), and the same
        // flat super-block stream when k is a multiple of the block size.
        const std::size_t want = nr * row_bytes;
        if (ggml_quantize_chunk(t, src.data(), out + static_cast<std::size_t>(r) * row_bytes, 0,
                                static_cast<std::int64_t>(nr), k, nullptr) != want)
            return false;
    }
    return true;
}

bool d3d12_gen_weights_reference(ggml_type t, int n, int k, std::mt19937& rng, void* dst) {
    if (!dst || n <= 0 || k <= 0 || !d3d12_weight_type_supported(t))
        return false;
    const std::int64_t blck = ggml_blck_size(t);
    if (blck <= 0 || k % blck != 0)
        return false;
    const std::size_t row_bytes = ggml_row_size(t, k);
    if (row_bytes == 0)
        return false;
    std::vector<float> src(static_cast<std::size_t>(n) * static_cast<std::size_t>(k));
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    for (float& v : src)
        v = uni(rng);
    return ggml_quantize_chunk(t, src.data(), dst, 0, n, k, nullptr) ==
           static_cast<std::size_t>(n) * row_bytes;
}

double d3d12_block_median(const std::vector<double>& v) {
    if (v.empty())
        return 0.0;
    std::vector<double> s(v);
    std::sort(s.begin(), s.end());
    return s[(s.size() - 1) / 2];
}

double d3d12_block_range(const std::vector<double>& v) {
    if (v.empty())
        return 0.0;
    const auto mm = std::minmax_element(v.begin(), v.end());
    return *mm.second - *mm.first;
}

// --- Selftest CSV ---

const char* d3d12_selftest_csv_header() {
    return "type,n,k,ncols,rel_err,gpu_ms,packed_gbs,ok,d3d12_ran,host,date,error\n";
}

std::string format_d3d12_selftest_row(const D3d12SelftestRow& r, const char* host_label) {
    char date_buf[32];
    std::time_t now = std::time(nullptr);
    std::strftime(date_buf, sizeof(date_buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    std::string err = r.error.empty() ? "-" : r.error;
    for (char& ch : err)
        if (ch == ',' || ch == '\n' || ch == '\r')
            ch = ' ';
    char buf[4096];
    std::snprintf(buf, sizeof(buf), "%s,%d,%d,%d,%.3g,%.4f,%.2f,%d,%d,%s,%s,%s\n", r.type.c_str(),
                  r.n, r.k, r.ncols, r.rel_err, r.gpu_ms, r.packed_gbs, r.ok ? 1 : 0,
                  r.d3d12_ran ? 1 : 0, host_label ? host_label : "unknown", date_buf, err.c_str());
    return buf;
}
const char* d3d12_shapecost_csv_header() {
    return "type,n,k,ncols,rel_err,gpu_ms,wall_ms,gpu_ms_range,wall_ms_range,packed_gbs,weight_"
           "mb,peak_ws_mb,variant,pair,twocol,pads,ok,d3d12_ran,host,date,error\n";
}

std::string format_d3d12_shapecost_row(const D3d12SelftestRow& r, const char* host_label) {
    char date_buf[32];
    std::time_t now = std::time(nullptr);
    std::strftime(date_buf, sizeof(date_buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    std::string err = r.error.empty() ? "-" : r.error;
    for (char& ch : err)
        if (ch == ',' || ch == '\n' || ch == '\r')
            ch = ' ';
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "%s,%d,%d,%d,%.3g,%.4f,%.4f,%.4f,%.4f,%.2f,%.3f,%.1f,%d,%d,%llu,%s,%d,%d,%s,%s,"
                  "%s\n",
                  r.type.c_str(), r.n, r.k, r.ncols, r.rel_err, r.gpu_ms, r.wall_ms, r.gpu_ms_range,
                  r.wall_ms_range, r.packed_gbs, r.weight_mb, r.peak_ws_mb, r.variant, r.pair,
                  static_cast<unsigned long long>(r.twocol), r.pads.c_str(), r.ok ? 1 : 0,
                  r.d3d12_ran ? 1 : 0, host_label ? host_label : "unknown", date_buf, err.c_str());
    return buf;
}

} // namespace xllama

    #if !defined(_WIN32)

namespace xllama {

bool ggml_d3d12_register() {
    return false;
}

void run_d3d12_silu_selftest(std::vector<D3d12SelftestRow>* out);

void run_d3d12_selftest(std::vector<D3d12SelftestRow>* out) {
    if (!out)
        return;
    D3d12SelftestRow r;
    r.type = "-";
    r.error = "d3d12 unavailable on this platform";
    out->push_back(std::move(r));
}

void run_d3d12_shape_cost(std::vector<D3d12SelftestRow>* out) {
    if (!out)
        return;
    D3d12SelftestRow r;
    r.type = "-";
    r.error = "d3d12 unavailable on this platform";
    out->push_back(std::move(r));
}

void run_d3d12_gdn_selftest(std::vector<D3d12SelftestRow>* out) {
    if (out)
        run_gdn_cases(nullptr, ggml_backend_cpu_buffer_type(), out);
}

void d3d12_set_kernel_variant(int) {}
int d3d12_kernel_variant() {
    return 0;
}
void d3d12_set_z0_dispatch_log(bool) {}
void d3d12_set_z0_pin_cpu(bool) {}
void d3d12_set_swiglu_mtp_diag(bool) {}
bool d3d12_swiglu_mtp_diag_enabled() {
    return false;
}
void d3d12_set_swiglu_range_diag(bool) {}
void d3d12_set_island_enabled(bool) {}
static int s_spin_wait_us = -1;
void d3d12_set_spin_wait_us(int spin_us) {
    s_spin_wait_us = spin_us;
}
int d3d12_spin_wait_us() {
    return s_spin_wait_us;
}
bool d3d12_2col_available() {
    return false;
}

bool d3d12_q6_tiled_available(int) {
    return false;
}
std::uint64_t d3d12_q6_tiled_matmuls() {
    return 0;
}
std::uint64_t d3d12_2col_matmuls() {
    return 0;
}
std::string d3d12_2col_info() {
    return "2col unavailable on this platform";
}

// Same names as the Windows build below, so a caller compiles either way; on a CPU
// host there is no d3d12 backend and every count is zero.
std::uint64_t d3d12_graph_calls() {
    return 0;
}
std::uint64_t d3d12_matmul_count() {
    return 0;
}
double d3d12_gpu_ms() {
    return 0.0;
}
double d3d12_wall_ms() {
    return 0.0;
}
// No D3D12 backend on this host: timestamps are "enabled" by contract but no
// query ever runs, so the product default is preserved without state.
void d3d12_set_gpu_timestamps(bool) {}
bool d3d12_gpu_timestamps_enabled() {
    return true;
}
void d3d12_set_silu_test_enabled(bool) {}
bool d3d12_silu_test_enabled() {
    return false;
}
void d3d12_set_swiglu_product_mode(int) {}
int d3d12_swiglu_product_mode() {
    return 0;
}
void d3d12_finalize_swiglu_mode(bool) {}
bool d3d12_swiglu_profile_accepts(bool) {
    return true;
}
void d3d12_set_swiglu_target_layers(int) {}
int d3d12_swiglu_target_layers() {
    return 0;
}
void d3d12_set_shape_log(bool) {}
void d3d12_set_scope(const char*, const char*) {}
void d3d12_get_scope(const char**, const char**) {}
bool d3d12_shape_drain(const char*) {
    return true;
}

} // namespace xllama

    #else // _WIN32 — the backend itself

        #include <atomic>
        #include <chrono>
        #include <cstdint>
        #include <limits>
        #include <mutex>
        #include <random>
        #include <xmmintrin.h>

        #include <dxgi1_4.h>

        #include "d3d12_compute.h"
        #include "ggml-alloc.h"
        #include "ggml-backend-impl.h"
        #include "ggml-backend.h"
        #include "ggml_d3d12_gated_delta_net_dxil.h"
        #include "ggml_d3d12_mmv_q4_0_t128_dxil.h"
        #include "ggml_d3d12_mmv_q4_0_t64_dxil.h"
        #include "ggml_d3d12_mmv_q4_k_t128_2col_dxil.h"
        #include "ggml_d3d12_mmv_q4_k_t128_dxil.h"
        #include "ggml_d3d12_mmv_q4_k_t64_2col_dxil.h"
        #include "ggml_d3d12_mmv_q4_k_t64_dxil.h"
        #include "ggml_d3d12_mmv_q5_k_t128_dxil.h"
        #include "ggml_d3d12_mmv_q5_k_t64_dxil.h"
        #include "ggml_d3d12_mmv_q6_k_t128_2col_dxil.h"
        #include "ggml_d3d12_mmv_q6_k_t128_4col_dxil.h"
        #include "ggml_d3d12_mmv_q6_k_t128_dxil.h"
        #include "ggml_d3d12_mmv_q6_k_t64_2col_dxil.h"
        #include "ggml_d3d12_mmv_q6_k_t64_4col_dxil.h"
        #include "ggml_d3d12_mmv_q6_k_t64_dxil.h"
        #include "ggml_d3d12_mmv_q8_0_t128_dxil.h"
        #include "ggml_d3d12_mmv_q8_0_t64_dxil.h"
        #include "ggml_d3d12_add_dxil.h"
        #include "ggml_d3d12_mul_dxil.h"
        #include "ggml_d3d12_rms_norm_dxil.h"
        #include "ggml_d3d12_silu_dxil.h"
        #include "ggml_d3d12_swiglu_probe_dxil.h"
        #include "xllama/d3d12_dyn.h"
        #include "xllama/platform.h"

namespace xllama {
namespace {

using d3d12c::ComPtr;

enum Pso { kPsoQ40 = 0, kPsoQ4K = 1, kPsoQ5K = 2, kPsoQ6K = 3, kPsoQ80 = 4, kPsoCount = 5 };

int pso_for(ggml_type t) {
    switch (t) {
    case GGML_TYPE_Q4_0:
        return kPsoQ40;
    case GGML_TYPE_Q4_K:
        return kPsoQ4K;
    case GGML_TYPE_Q5_K:
        return kPsoQ5K;
    case GGML_TYPE_Q6_K:
        return kPsoQ6K;
    case GGML_TYPE_Q8_0:
        return kPsoQ80;
    default:
        return -1;
    }
}

// Process-wide D3D12 state behind the ggml device. One queue, serialised.
struct Gpu {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> cl;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12RootSignature> gdn_root;
    ComPtr<ID3D12PipelineState> gdn_pso;
    std::string gdn_error;
    // Selftest-only SILU kernel (plan 006 C4), reusing the matmul root layout.
    ComPtr<ID3D12PipelineState> silu_pso;
    // Selftest-only SWIGLU probe, reusing the GDN root (three UAVs: out/gate/up).
    ComPtr<ID3D12PipelineState> swiglu_probe_pso;
    // Plan 007 bounded chain: RMS_NORM and MUL on the GDN root (16 root
    // constants + seven UAVs). Product graph ops, unlike the probes above.
    ComPtr<ID3D12PipelineState> rms_norm_pso;
    ComPtr<ID3D12PipelineState> mul_pso;
    ComPtr<ID3D12PipelineState> add_pso;
    ComPtr<ID3D12PipelineState> pso[kPsoCount][2]; // [type][0 = 64 threads, 1 = 128]
    // Plan 004 two-column tile experiment: q4_k only, same two widths. Kept
    // beside pso (not inside it) so the gate blobs and their indices never
    // move. Null entries mean the NEW path is unavailable and dispatch falls
    // back to OLD; a paired case then reports an error row, never silent OLD.
    ComPtr<ID3D12PipelineState> pso2col[2];
    ComPtr<ID3D12PipelineState> pso_q6col[2][2]; // [2/4 columns][64/128 threads]
    std::string twocol_error;
    ComPtr<ID3D12QueryHeap> ts;
    ComPtr<ID3D12Resource> ts_rb;
    ComPtr<ID3D12Resource> staging;  // weight upload ring (kStagingBytes)
    ComPtr<ID3D12Resource> readback; // get_tensor on weights (kStagingBytes)
    std::uint8_t* staging_ptr = nullptr;
    std::uint8_t* readback_ptr = nullptr;
    d3d12c::QueueFence fence;
    UINT64 ts_freq = 0;
    LUID luid = {};
    // Valid-sample GPU timing state (plan 006 C8): one exactly-one-path update
    // per graph_compute, so a disabled/failed sample can never re-add the
    // previous call's timing.
    GpuTimingState timing;
    // Per-backend-lifetime counters, logged when the backend is freed.
    std::uint64_t n_calls = 0;
    std::uint64_t n_matmuls = 0;
    std::uint64_t n_gdn = 0;
    // Actual two-column dispatches (plan 004): incremented only when the NEW
    // kernel really runs. Requested variant (the global) is not evidence: B1
    // stays OLD even under request 1, and non-Q4 never dispatches NEW.
    std::uint64_t n_matmuls_2col = 0;
    std::uint64_t n_q6_tiled[2] = {}; // actual 2/4-column Q6 dispatches
    // Product-trial FFN SWIGLU dispatches (plan 006 C4): nonzero is placement
    // evidence that the knob engaged; zero proves the OFF arm stayed on CPU.
    std::uint64_t n_swiglu = 0;
    std::uint64_t n_rms_norm = 0;   // plan 007 bounded chain
    std::uint64_t n_mul = 0;        // plan 007 bounded chain
    std::uint64_t n_add = 0;        // plan 008 island
    double wall_ms = 0.0;
    std::mutex mu;
    bool ok = false;
    // Per-shape histogram switch (plan 003 stage 2). Defaults ON; the bench knob
    // turns it off for the instrumentation-cost arm, because building and logging
    // the aggregation is itself work in the hot path.
    std::string error;
};

// --- Plan 003 stage 2: shape histogram -------------------------------------
// Separate from Gpu's lifetime counters: those are zeroed in backend_free and
// cannot attribute anything to a phase, and they have no shape dimension. These
// are keyed by (context, phase, type, N, K, B, threads) and drained at a phase or
// generation boundary. One aggregation buffer owned by its own mutex, so the hot
// path never nests locks on g.mu and never logs.
struct ShapeAggKey {
    std::string ctx;
    std::string phase;
    int type = 0;
    long long n = 0;
    long long k = 0;
    long long b = 0;
    int thr = 0;
};
std::mutex g_shape_mu;
std::vector<std::pair<ShapeAggKey, long long>> g_shape_agg;
bool g_shape_off = false;
// GPU timestamp queries (plan 006 C8). Default true = historical behavior;
// the knob disables query/resolve/readback entirely for the cost arm. Atomic:
// the setter runs at startup/knob time while decode threads read it inside
// graph_compute under g.mu.
std::atomic<bool> g_ts_enabled{true};
// Selftest-only SILU gate (plan 006 C4). False in every product path.
bool g_silu_test = false;
// Plan 007: the bounded RMS_NORM + MUL chain, off by default until the paired
// same-build A/B (bench_normmul.txt="1" enables) accepts it.
bool g_island_enabled = false;
// Product trial gate (plan 006 C4): FFN split-SWIGLU on D3D12, default OFF.
// Mode 0 off, 1 all FFN, 2 target-context FFN only. Target layer count is
// setup-time (model load) and never changes while graphs run. The mode is
// governed by a process-profile latch: once any MTP-capable context is
// created, FFN stays OFF for the process (fail-closed on conflicting context
// profiles; no multi-context concurrency guarantee is claimed).
int g_swiglu_requested_mode = 0;
int g_swiglu_target_layers = 0;
SwigluModePolicy g_swiglu_policy;
// Diagnostic-only (bench): allow FFN SWIGLU in an MTP context to reproduce the
// C4 divergence with fixed-history dumps. Never on in production.
bool g_swiglu_mtp_diag = false;
// Owner audit: real-range sampling of the product SWIGLU inputs (gate/up).
// Buckets by |x|; sampled after the graph fence so the host-visible values are
// the real kernel inputs, not stale.
bool g_swiglu_range_diag = false;
struct SwigluRangeStats {
    std::uint64_t n = 0, subn = 0, nf = 0, zero = 0;
    std::uint64_t b1 = 0, b10 = 0, b30 = 0, b87 = 0, b126 = 0, b192 = 0, binf = 0;
    double minv = 1e30, maxv = 0.0;
};
SwigluRangeStats g_swiglu_range_gate; // src[0]
SwigluRangeStats g_swiglu_range_up;   // src[1]
std::uint64_t g_swiglu_range_sampled_pairs = 0;  // pairs actually sampled
std::uint64_t g_swiglu_range_skipped_pairs = 0;  // dispatched but not sampled (cap)
// Context/phase tags. Thread-local because decode and prefill run on the calling
// thread, and a tag set around a target call must not leak into the drafter call
// that follows it. Restored by the RAII guard at each call site.
thread_local const char* t_ctx = "target";
thread_local const char* t_phase = "decode";

// Weight upload / readback ring. Mapped for the process lifetime and counted in
// its working set, so it stays small: 64 MiB each put the first GPU-layer
// smoke at 640 MB peak vs 311 MB on the CPU (D2b); 8 MiB costs a few more
// round trips at load only.
constexpr UINT64 kStagingBytes = 8ull << 20;

ComPtr<ID3D12RootSignature> make_root_sig(ID3D12Device* device, std::string* err,
                                          bool gdn = false) {
    D3D12_ROOT_PARAMETER p[8] = {};
    p[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    p[0].Constants.ShaderRegister = 0;
    p[0].Constants.Num32BitValues = gdn ? 16 : 8;
    if (gdn) {
        // All seven GDN tensors live in D3D12_Host/UNORDERED_ACCESS.
        for (int i = 0; i < 7; ++i) {
            p[i + 1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
            p[i + 1].Descriptor.ShaderRegister = i;
        }
    } else {
        p[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV; // t0 weights
        p[1].Descriptor.ShaderRegister = 0;
        p[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; // u0 output
        p[2].Descriptor.ShaderRegister = 0;
        p[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; // u1 activations
        p[3].Descriptor.ShaderRegister = 1;
    }
    for (auto& q : p)
        q.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC desc = {};
    desc.NumParameters = gdn ? 8 : 4;
    desc.pParameters = p;
    auto serialize = d3d12_dyn::SerializeRootSignature();
    if (!serialize) {
        *err = "D3D12SerializeRootSignature not available";
        return {};
    }
    ComPtr<ID3DBlob> sig, blob_err;
    HRESULT hr = serialize(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &blob_err);
    if (FAILED(hr)) {
        *err = d3d12c::hr_message("D3D12SerializeRootSignature", hr);
        return {};
    }
    ComPtr<ID3D12RootSignature> root;
    hr = device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
                                     IID_PPV_ARGS(&root));
    if (FAILED(hr)) {
        *err = d3d12c::hr_message("CreateRootSignature", hr);
        return {};
    }
    return root;
}

bool init_gpu(Gpu& g) {
    std::string& err = g.error;
    g.device = d3d12c::create_device(&err);
    if (!g.device)
        return false;
    D3D12_FEATURE_DATA_ARCHITECTURE1 arch = {};
    if (FAILED(g.device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE1, &arch, sizeof(arch))) ||
        !arch.CacheCoherentUMA) {
        // D3D12_Host relies on a CPU-cached GPU-visible heap (D1: cc_uma=1 on Series S).
        err = "adapter is not cache-coherent UMA";
        return false;
    }
    g.luid = g.device->GetAdapterLuid();
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    HRESULT hr = g.device->CreateCommandQueue(&qd, IID_PPV_ARGS(&g.queue));
    if (FAILED(hr)) {
        err = d3d12c::hr_message("CreateCommandQueue", hr);
        return false;
    }
    hr = g.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&g.alloc));
    if (SUCCEEDED(hr))
        hr = g.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g.alloc.Get(), nullptr,
                                         IID_PPV_ARGS(&g.cl));
    if (FAILED(hr)) {
        err = d3d12c::hr_message("CreateCommandList", hr);
        return false;
    }
    g.cl->Close();
    if (!g.fence.init(g.device.Get(), &err))
        return false;
    g.root = make_root_sig(g.device.Get(), &err);
    if (!g.root)
        return false;
    const void* blobs[kPsoCount][2] = {{kGgmlD3d12MmvQ40T64Dxil, kGgmlD3d12MmvQ40T128Dxil},
                                       {kGgmlD3d12MmvQ4KT64Dxil, kGgmlD3d12MmvQ4KT128Dxil},
                                       {kGgmlD3d12MmvQ5KT64Dxil, kGgmlD3d12MmvQ5KT128Dxil},
                                       {kGgmlD3d12MmvQ6KT64Dxil, kGgmlD3d12MmvQ6KT128Dxil},
                                       {kGgmlD3d12MmvQ80T64Dxil, kGgmlD3d12MmvQ80T128Dxil}};
    const size_t sizes[kPsoCount][2] = {
        {kGgmlD3d12MmvQ40T64DxilSize, kGgmlD3d12MmvQ40T128DxilSize},
        {kGgmlD3d12MmvQ4KT64DxilSize, kGgmlD3d12MmvQ4KT128DxilSize},
        {kGgmlD3d12MmvQ5KT64DxilSize, kGgmlD3d12MmvQ5KT128DxilSize},
        {kGgmlD3d12MmvQ6KT64DxilSize, kGgmlD3d12MmvQ6KT128DxilSize},
        {kGgmlD3d12MmvQ80T64DxilSize, kGgmlD3d12MmvQ80T128DxilSize}};
    for (int i = 0; i < kPsoCount; ++i) {
        for (int wd = 0; wd < 2; ++wd) {
            D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
            pd.pRootSignature = g.root.Get();
            pd.CS.pShaderBytecode = blobs[i][wd];
            pd.CS.BytecodeLength = sizes[i][wd];
            hr = g.device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&g.pso[i][wd]));
            if (FAILED(hr)) {
                err = d3d12c::hr_message("CreateComputePipelineState", hr);
                return false;
            }
        }
    }
    // Optional GDN PSO: failure leaves the existing matmul backend available.
    g.gdn_root = make_root_sig(g.device.Get(), &g.gdn_error, true);
    if (g.gdn_root) {
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
        pd.pRootSignature = g.gdn_root.Get();
        pd.CS = {kGgmlD3d12GatedDeltaNetDxil, kGgmlD3d12GatedDeltaNetDxilSize};
        hr = g.device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&g.gdn_pso));
        if (FAILED(hr))
            g.gdn_error = d3d12c::hr_message("CreateComputePipelineState(GDN)", hr);
    }
    if (!g.gdn_pso)
        log_output("[xllama] d3d12: GDN unavailable: " + g.gdn_error + "\n");
    // Selftest-only SILU PSO (plan 006 C4): reuses the matmul root layout
    // (b0 constants, t0 SRV, u0 UAV). Never dispatched unless the selftest
    // gate is on; no product graph can reach it.
    if (g.root) {
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
        pd.pRootSignature = g.root.Get();
        pd.CS = {kGgmlD3d12SiluDxil, kGgmlD3d12SiluDxilSize};
        hr = g.device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&g.silu_pso));
        if (FAILED(hr))
            log_output("[xllama] d3d12: SILU selftest PSO unavailable\n");
    }
    if (g.gdn_root) {
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
        pd.pRootSignature = g.gdn_root.Get();
        pd.CS = {kGgmlD3d12SwigluProbeDxil, kGgmlD3d12SwigluProbeDxilSize};
        hr = g.device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&g.swiglu_probe_pso));
        if (FAILED(hr))
            log_output("[xllama] d3d12: SWIGLU probe PSO unavailable\n");
            pd.CS = {kGgmlD3d12AddDxil, kGgmlD3d12AddDxilSize};
        hr = g.device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&g.add_pso));
        if (FAILED(hr))
            log_output("[xllama] d3d12: add PSO unavailable\n");
    }
    // Plan 007: the bounded RMS_NORM + MUL chain. Matmul root layout: b0
    // constants + t0 SRV (MUL weight, like the matmul path) + u0/u1 UAVs.
    if (g.root) {
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
        pd.pRootSignature = g.root.Get();
        pd.CS = {kGgmlD3d12RmsNormDxil, kGgmlD3d12RmsNormDxilSize};
        hr = g.device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&g.rms_norm_pso));
        if (FAILED(hr))
            log_output("[xllama] d3d12: rms_norm PSO unavailable\n");
        pd.CS = {kGgmlD3d12MulDxil, kGgmlD3d12MulDxilSize};
        hr = g.device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&g.mul_pso));
        if (FAILED(hr))
            log_output("[xllama] d3d12: mul PSO unavailable\n");
    }

    // Two-column experiment PSOs (plan 004). A failure here must NOT fail the
    // backend: the gate and production path are OLD-only. Record it; dispatch
    // falls back to OLD and paired cases report explicit error rows.
    const void* blobs2col[2] = {kGgmlD3d12MmvQ4K2ColT64Dxil, kGgmlD3d12MmvQ4K2ColT128Dxil};
    const size_t sizes2col[2] = {kGgmlD3d12MmvQ4K2ColT64DxilSize, kGgmlD3d12MmvQ4K2ColT128DxilSize};
    for (int wd = 0; wd < 2; ++wd) {
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
        pd.pRootSignature = g.root.Get();
        pd.CS.pShaderBytecode = blobs2col[wd];
        pd.CS.BytecodeLength = sizes2col[wd];
        hr = g.device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&g.pso2col[wd]));
        if (FAILED(hr) || !g.pso2col[wd]) {
            g.pso2col[wd].Reset();
            g.twocol_error = d3d12c::hr_message("CreateComputePipelineState(2col)", hr);
        }
    }
    const void* q6blobs[2][2] = {{kGgmlD3d12MmvQ6K2ColT64Dxil, kGgmlD3d12MmvQ6K2ColT128Dxil},
                                 {kGgmlD3d12MmvQ6K4ColT64Dxil, kGgmlD3d12MmvQ6K4ColT128Dxil}};
    const size_t q6sizes[2][2] = {
        {kGgmlD3d12MmvQ6K2ColT64DxilSize, kGgmlD3d12MmvQ6K2ColT128DxilSize},
        {kGgmlD3d12MmvQ6K4ColT64DxilSize, kGgmlD3d12MmvQ6K4ColT128DxilSize}};
    for (int cols = 0; cols < 2; ++cols) {
        for (int wide = 0; wide < 2; ++wide) {
            D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
            pd.pRootSignature = g.root.Get();
            pd.CS = {q6blobs[cols][wide], q6sizes[cols][wide]};
            hr = g.device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&g.pso_q6col[cols][wide]));
            if (FAILED(hr)) {
                g.pso_q6col[cols][wide].Reset();
                log_output("[xllama] d3d12: Q6 tile unavailable: " +
                           d3d12c::hr_message("CreateComputePipelineState(Q6 tile)", hr) + "\n");
            }
        }
    }
    if (SUCCEEDED(g.queue->GetTimestampFrequency(&g.ts_freq)) && g.ts_freq > 0) {
        D3D12_QUERY_HEAP_DESC qhd = {};
        qhd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        qhd.Count = 2;
        g.device->CreateQueryHeap(&qhd, IID_PPV_ARGS(&g.ts));
        g.ts_rb = d3d12c::create_buffer(g.device.Get(), 256, D3D12_HEAP_TYPE_READBACK,
                                        D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
                                        "ts_rb", &err);
        if (!g.ts || !g.ts_rb) {
            g.ts.Reset();
            g.ts_rb.Reset();
            err.clear();
        }
    }
    g.staging = d3d12c::create_buffer(g.device.Get(), kStagingBytes, D3D12_HEAP_TYPE_UPLOAD,
                                      D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ,
                                      "staging", &err);
    g.readback = d3d12c::create_buffer(g.device.Get(), kStagingBytes, D3D12_HEAP_TYPE_READBACK,
                                       D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
                                       "readback", &err);
    if (!g.staging || !g.readback)
        return false;
    void* p = nullptr;
    if (FAILED(g.staging->Map(0, nullptr, &p)))
        return false;
    g.staging_ptr = static_cast<std::uint8_t*>(p);
    if (FAILED(g.readback->Map(0, nullptr, &p)))
        return false;
    g.readback_ptr = static_cast<std::uint8_t*>(p);
    return true;
}

Gpu& gpu() {
    static Gpu g;
    static std::once_flag once;
    std::call_once(once, [] { g.ok = init_gpu(g); });
    return g;
}

// Record into the shared list and run it to completion. Caller holds g.mu.
template <typename F> bool run_now(Gpu& g, F&& record) {
    g.alloc->Reset();
    g.cl->Reset(g.alloc.Get(), nullptr);
    record(g.cl.Get());
    g.cl->Close();
    ID3D12CommandList* lists[] = {g.cl.Get()};
    g.queue->ExecuteCommandLists(1, lists);
    return g.fence.signal_and_wait(g.queue.Get(), /*spin=*/true, d3d12_spin_wait_us());
}

// --- Buffers ---

struct BufCtx {
    ComPtr<ID3D12Resource> res;
    D3D12_GPU_VIRTUAL_ADDRESS va = 0;
    std::uint8_t* host = nullptr; // mapped pointer (D3D12_Host only)
    bool is_host = false;
};

void* buf_base(ggml_backend_buffer_t b) {
    auto* c = static_cast<BufCtx*>(b->context);
    return c->is_host ? static_cast<void*>(c->host)
                      : reinterpret_cast<void*>(static_cast<std::uintptr_t>(c->va));
}

void buf_free(ggml_backend_buffer_t b) {
    delete static_cast<BufCtx*>(b->context);
}

std::size_t tensor_offset(const ggml_tensor* t) {
    return static_cast<std::size_t>(static_cast<const std::uint8_t*>(t->data) -
                                    static_cast<const std::uint8_t*>(buf_base(t->buffer)));
}

D3D12_GPU_VIRTUAL_ADDRESS tensor_va(const ggml_tensor* t) {
    return static_cast<BufCtx*>(t->buffer->context)->va + tensor_offset(t);
}

// D3D12_Host: plain memory.
void host_memset(ggml_backend_buffer_t, ggml_tensor* t, uint8_t v, size_t off, size_t size) {
    std::memset(static_cast<std::uint8_t*>(t->data) + off, v, size);
}
void host_set(ggml_backend_buffer_t, ggml_tensor* t, const void* data, size_t off, size_t size) {
    std::memcpy(static_cast<std::uint8_t*>(t->data) + off, data, size);
}
void host_get(ggml_backend_buffer_t, const ggml_tensor* t, void* data, size_t off, size_t size) {
    std::memcpy(data, static_cast<const std::uint8_t*>(t->data) + off, size);
}
bool host_cpy(ggml_backend_buffer_t, const ggml_tensor* src, ggml_tensor* dst) {
    if (!ggml_backend_buffer_is_host(src->buffer))
        return false;
    std::memcpy(dst->data, src->data, ggml_nbytes(src));
    return true;
}
void host_clear(ggml_backend_buffer_t b, uint8_t v) {
    std::memset(static_cast<BufCtx*>(b->context)->host, v, b->size);
}

const ggml_backend_buffer_i kHostBufIface = {
    /* .free_buffer   = */ buf_free,
    /* .get_base      = */ buf_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ host_memset,
    /* .set_tensor    = */ host_set,
    /* .get_tensor    = */ host_get,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ host_cpy,
    /* .clear         = */ host_clear,
    /* .reset         = */ nullptr,
};

// D3D12_Weights: DEFAULT heap, filled through the staging ring. The buffer
// stays in COMMON; buffers promote implicitly to COPY_DEST / shader read and
// decay back at the end of each submission.
void weights_write(ggml_backend_buffer_t b, std::size_t dst_off, const void* data, std::size_t size,
                   int fill) {
    Gpu& g = gpu();
    std::lock_guard<std::mutex> lock(g.mu);
    auto* c = static_cast<BufCtx*>(b->context);
    for (std::size_t done = 0; done < size;) {
        const std::size_t chunk = std::min<std::size_t>(size - done, kStagingBytes);
        if (data)
            std::memcpy(g.staging_ptr, static_cast<const std::uint8_t*>(data) + done, chunk);
        else
            std::memset(g.staging_ptr, fill, chunk);
        run_now(g, [&](ID3D12GraphicsCommandList* cl) {
            cl->CopyBufferRegion(c->res.Get(), dst_off + done, g.staging.Get(), 0, chunk);
        });
        done += chunk;
    }
}
void weights_memset(ggml_backend_buffer_t b, ggml_tensor* t, uint8_t v, size_t off, size_t size) {
    weights_write(b, tensor_offset(t) + off, nullptr, size, v);
}
void weights_set(ggml_backend_buffer_t b, ggml_tensor* t, const void* data, size_t off,
                 size_t size) {
    weights_write(b, tensor_offset(t) + off, data, size, 0);
}
void weights_get(ggml_backend_buffer_t b, const ggml_tensor* t, void* data, size_t off,
                 size_t size) {
    Gpu& g = gpu();
    std::lock_guard<std::mutex> lock(g.mu);
    auto* c = static_cast<BufCtx*>(b->context);
    const std::size_t src_off = tensor_offset(t) + off;
    for (std::size_t done = 0; done < size;) {
        const std::size_t chunk = std::min<std::size_t>(size - done, kStagingBytes);
        run_now(g, [&](ID3D12GraphicsCommandList* cl) {
            cl->CopyBufferRegion(g.readback.Get(), 0, c->res.Get(), src_off + done, chunk);
        });
        std::memcpy(static_cast<std::uint8_t*>(data) + done, g.readback_ptr, chunk);
        done += chunk;
    }
}
void weights_clear(ggml_backend_buffer_t b, uint8_t v) {
    weights_write(b, 0, nullptr, b->size, v);
}

const ggml_backend_buffer_i kWeightsBufIface = {
    /* .free_buffer   = */ buf_free,
    /* .get_base      = */ buf_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ weights_memset,
    /* .set_tensor    = */ weights_set,
    /* .get_tensor    = */ weights_get,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ nullptr,
    /* .clear         = */ weights_clear,
    /* .reset         = */ nullptr,
};

// --- Buffer types ---

ggml_backend_dev_t device_ptr();

const char* host_buft_name(ggml_backend_buffer_type_t) {
    return "D3D12_Host";
}
const char* weights_buft_name(ggml_backend_buffer_type_t) {
    return "D3D12_Weights";
}
size_t buft_alignment(ggml_backend_buffer_type_t) {
    return 256;
}
bool buft_is_host_true(ggml_backend_buffer_type_t) {
    return true;
}
// The Q4_0/Q6_K kernels read whole aligned dwords, up to 2 bytes past a
// tensor's last block; root descriptors have no bounds check, so pad.
size_t weights_alloc_size(ggml_backend_buffer_type_t, const ggml_tensor* t) {
    return ggml_nbytes(t) + 16;
}

ggml_backend_buffer_t alloc_buf(ggml_backend_buffer_type_t buft, size_t size, bool host) {
    Gpu& g = gpu();
    if (!g.ok)
        return nullptr;
    size = std::max<size_t>(size, 256);
    auto* c = new BufCtx;
    c->is_host = host;
    std::string err;
    if (host) {
        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = D3D12_HEAP_TYPE_CUSTOM;
        hp.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
        hp.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
        c->res = d3d12c::create_buffer_props(
            g.device.Get(), size, hp, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "D3D12_Host", &err);
        void* p = nullptr;
        if (c->res && SUCCEEDED(c->res->Map(0, nullptr, &p)))
            c->host = static_cast<std::uint8_t*>(p);
    } else {
        c->res = d3d12c::create_buffer(g.device.Get(), size, D3D12_HEAP_TYPE_DEFAULT,
                                       D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON,
                                       "D3D12_Weights", &err);
    }
    if (!c->res || (host && !c->host)) {
        char msg[256];
        std::snprintf(msg, sizeof(msg), "[xllama] d3d12: allocation of %zu bytes failed: %s\n",
                      size, err.c_str());
        log_output(msg);
        delete c;
        return nullptr;
    }
    c->va = c->res->GetGPUVirtualAddress();
    if (size >= (1u << 20)) {
        char msg[128];
        std::snprintf(msg, sizeof(msg), "[xllama] d3d12: %s buffer %.1f MiB\n",
                      host ? "D3D12_Host" : "D3D12_Weights", static_cast<double>(size) / 1048576.0);
        log_output(msg);
    }
    return ggml_backend_buffer_init(buft, host ? kHostBufIface : kWeightsBufIface, c, size);
}

ggml_backend_buffer_t host_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    return alloc_buf(buft, size, true);
}
ggml_backend_buffer_t weights_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    return alloc_buf(buft, size, false);
}

ggml_backend_buffer_type kHostBuft = {
    /* .iface = */ {
        /* .get_name         = */ host_buft_name,
        /* .alloc_buffer     = */ host_alloc,
        /* .alloc_buffer_n   = */ nullptr,
        /* .get_alignment    = */ buft_alignment,
        /* .get_max_size     = */ nullptr,
        /* .get_alloc_size   = */ nullptr,
        /* .get_alloc_size_n = */ nullptr,
        /* .is_host          = */ buft_is_host_true,
    },
    /* .device  = */ nullptr,
    /* .context = */ nullptr,
};
ggml_backend_buffer_type kWeightsBuft = {
    /* .iface = */ {
        /* .get_name         = */ weights_buft_name,
        /* .alloc_buffer     = */ weights_alloc,
        /* .alloc_buffer_n   = */ nullptr,
        /* .get_alignment    = */ buft_alignment,
        /* .get_max_size     = */ nullptr,
        /* .get_alloc_size   = */ weights_alloc_size,
        /* .get_alloc_size_n = */ nullptr,
        /* .is_host          = */ nullptr,
    },
    /* .device  = */ nullptr,
    /* .context = */ nullptr,
};

bool is_ours(ggml_backend_buffer_type_t buft) {
    return buft == &kHostBuft || buft == &kWeightsBuft;
}

// One shared predicate for every SILU entry point (plan 006 C4): the selftest
// gate + PSO, GGML_OP_UNARY/GGML_UNARY_OP_SILU, f32 src/dst, contiguity, an
// 8-divisible row, a bounded element count, and (when buffers are already
// assigned) this backend's own buffers. dev_supports_op, the op collector and
// the dispatch all use exactly this, so a shape the dispatch would refuse can
// never be claimed by supports_op.
bool d3d12_swiglu_supported(const ggml_tensor* op) {
    if ((!g_silu_test && g_swiglu_policy.mode == 0) || !gpu().swiglu_probe_pso)
        return false;
    if (!op || op->op != GGML_OP_GLU || ggml_get_glu_op(op) != GGML_GLU_OP_SWIGLU)
        return false;
    const ggml_tensor* a = op->src[0];
    const ggml_tensor* b = op->src[1];
    if (!a || !b || op->type != GGML_TYPE_F32 || a->type != GGML_TYPE_F32 ||
        b->type != GGML_TYPE_F32)
        return false;
    if (!ggml_is_contiguous(op) || !ggml_is_contiguous(a) || !ggml_is_contiguous(b))
        return false;
    if (op->ne[0] <= 0 || op->ne[0] % 8 != 0 || a->ne[0] != b->ne[0] || a->ne[1] != b->ne[1])
        return false;
    // Every operand dimension must match the output.
    if (a->ne[0] != op->ne[0] || b->ne[0] != op->ne[0] || a->ne[1] != op->ne[1] ||
        b->ne[1] != op->ne[1] || op->ne[2] != 1 || op->ne[3] != 1)
        return false;
    // Product trial: FFN name/shape lineage only, so non-FFN GLU (attention or
    // draft blocks with different lineage) is never claimed. The selftest path
    // skips this whitelist (it builds its own corpus tensors).
    if (!g_silu_test) {
        const bool lineage = op->name[0] && std::strstr(op->name, "ffn_swiglu") == op->name &&
                             a->name[0] && std::strstr(a->name, "ffn_gate") == a->name &&
                             b->name[0] && std::strstr(b->name, "ffn_up") == b->name &&
                             op->ne[0] == 9216;
        if (!lineage)
            return false;
        if (g_swiglu_policy.mode == 2) {
            // Target-context only: the observed name suffix is the layer index;
            // target layers are 0..n_layer-1, the nextn/draft layer is 32.
            const char* dash = std::strrchr(op->name, '-');
            if (!dash || g_swiglu_target_layers <= 0)
                return false;
            const int layer = std::atoi(dash + 1);
            if (layer < 0 || layer >= g_swiglu_target_layers)
                return false;
        }
    }
    const int64_t n = ggml_nelements(op);
    if (n <= 0 || n > (int64_t{1} << 26) || n % 256 != 0)
        return false;
    if (a->buffer && !is_ours(a->buffer->buft))
        return false;
    if (b->buffer && !is_ours(b->buffer->buft))
        return false;
    if (op->buffer && !is_ours(op->buffer->buft))
        return false;
    return true;
}

// Plan 007 bounded chain predicates. Strict on purpose: anything not provably
// handled stays with the CPU kernel (bit-exactness first, placement second).
// RMS_NORM: f32 contiguous in/out, ne[0] % 256 == 0 so every lane has work in
// the fixed-stride loop shape the kernel assumes for its double tree, rows
// (ne[1]*ne[2]*ne[3]) in [1, 65535], and both tensors in our buffers.
bool d3d12_rms_norm_supported(const ggml_tensor* op) {
    if (!g_island_enabled || !gpu().rms_norm_pso)
        return false;
    if (!op || op->op != GGML_OP_RMS_NORM)
        return false;
    if (op->type != GGML_TYPE_F32)
        return false;
    const ggml_tensor* src = op->src[0];
    if (!src || src->type != GGML_TYPE_F32)
        return false;
    if (!ggml_is_contiguous(op) || !ggml_is_contiguous(src))
        return false;
    if (op->ne[0] <= 0 || op->ne[0] % 256 != 0)
        return false;
    const int64_t rows = op->ne[1] * op->ne[2] * op->ne[3];
    if (rows <= 0 || rows > 65535)
        return false;
    if (src->buffer && !is_ours(src->buffer->buft))
        return false;
    if (op->buffer && !is_ours(op->buffer->buft))
        return false;
    return true;
}

// MUL: broadcast-only 1-D weight form (the norm-weight shape). The scalar
// form is deliberately unsupported: a scalar would have to be read through the
// host on a possibly GPU-produced tensor with no dependency tracking, and the
// weight here is bound as an SRV like the matmul path (weights buffers are not
// UAV-bindable). Everything not proven stays on the CPU kernel.
bool d3d12_mul_supported(const ggml_tensor* op, bool* scalar_form) {
    if (scalar_form)
        *scalar_form = false;
    if (!g_island_enabled || !gpu().mul_pso)
        return false;
    if (!op || op->op != GGML_OP_MUL || op->type != GGML_TYPE_F32)
        return false;
    const ggml_tensor* a = op->src[0];
    const ggml_tensor* b = op->src[1];
    if (!a || !b || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32)
        return false;
    if (!ggml_is_contiguous(op) || !ggml_is_contiguous(a) || !ggml_is_contiguous(b))
        return false;
    if (b->ne[0] != op->ne[0] || b->ne[1] != 1 || b->ne[2] != 1 || b->ne[3] != 1)
        return false;
    const int64_t n = ggml_nelements(op);
    // One group per 256 elements, one dispatch dimension: D3D12 caps a
    // dimension at 65535 groups.
    if (n <= 0 || n % 256 != 0 || n / 256 > 65535)
        return false;
    if (a->buffer && !is_ours(a->buffer->buft))
        return false;
    if (b->buffer && !is_ours(b->buffer->buft))
        return false;
    if (op->buffer && !is_ours(op->buffer->buft))
        return false;
    return true;
}

// ADD (plan 008 island): same-shape f32 elementwise add, contiguous, both
// sources and dst in our buffers, whole groups only. Bit-exact per element.
bool d3d12_add_supported(const ggml_tensor* op) {
    static int s_add_rej = 0;
    auto rej = [&](const char* why) {
        if (g_island_enabled && s_add_rej < 4) {
            ++s_add_rej;
            char lb[320];
            const ggml_tensor* a = op && op->op == GGML_OP_ADD ? op->src[0] : nullptr;
            const ggml_tensor* b = op && op->op == GGML_OP_ADD ? op->src[1] : nullptr;
            std::snprintf(lb, sizeof(lb),
                          "[xllama] d3d12: ADD rejected (%s) dst[%s %lldx%lldx%lldx%lld ov=%d] "
                          "a[%s %lldx%lld ov=%d] b[%s %lldx%lld ov=%d] n=%lld\n",
                          why, op ? ggml_type_name(op->type) : "?",
                          op ? (long long)op->ne[0] : -1, op ? (long long)op->ne[1] : -1,
                          op ? (long long)op->ne[2] : -1, op ? (long long)op->ne[3] : -1,
                          op ? (op->buffer && !is_ours(op->buffer->buft)) : 1,
                          a ? ggml_type_name(a->type) : "?",
                          a ? (long long)a->ne[0] : -1, a ? (long long)a->ne[1] : -1,
                          a ? (a->buffer && !is_ours(a->buffer->buft)) : 1,
                          b ? ggml_type_name(b->type) : "?",
                          b ? (long long)b->ne[0] : -1, b ? (long long)b->ne[1] : -1,
                          b ? (b->buffer && !is_ours(b->buffer->buft)) : 1,
                          op ? (long long)ggml_nelements(op) : -1);
            log_output(lb);
        }
        return false;
    };
    if (!gpu().add_pso)
        return false;
    if (!op || op->op != GGML_OP_ADD || op->type != GGML_TYPE_F32)
        return false;
    const ggml_tensor* a = op->src[0];
    const ggml_tensor* b = op->src[1];
    if (!a || !b || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32)
        return rej("type");
    if (!ggml_is_contiguous(op) || !ggml_is_contiguous(a) || !ggml_is_contiguous(b))
        return rej("contig");
    if (a->ne[0] != op->ne[0] || a->ne[1] != op->ne[1] || a->ne[2] != op->ne[2] ||
        a->ne[3] != op->ne[3])
        return rej("shape_a");
    if (b->ne[0] != op->ne[0] || b->ne[1] != op->ne[1] || b->ne[2] != op->ne[2] ||
        b->ne[3] != op->ne[3])
        return rej("shape_b");
    const int64_t n = ggml_nelements(op);
    if (n <= 0 || n % 256 != 0 || n / 256 > 65535)
        return rej("size");
    if (a->buffer && !is_ours(a->buffer->buft))
        return false;
    if (b->buffer && !is_ours(b->buffer->buft))
        return false;
    if (op->buffer && !is_ours(op->buffer->buft))
        return false;
    return true;
}

bool d3d12_silu_supported(const ggml_tensor* op) {
    if (!g_silu_test || !gpu().silu_pso)
        return false;
    if (!op || op->op != GGML_OP_UNARY || ggml_get_unary_op(op) != GGML_UNARY_OP_SILU)
        return false;
    if (op->type != GGML_TYPE_F32)
        return false;
    const ggml_tensor* src = op->src[0];
    if (!src || src->type != GGML_TYPE_F32)
        return false;
    if (!ggml_is_contiguous(op) || !ggml_is_contiguous(src))
        return false;
    if (op->ne[0] <= 0 || op->ne[0] % 8 != 0)
        return false;
    const int64_t n = ggml_nelements(op);
    if (n <= 0 || n > (int64_t{1} << 26))
        return false;
    // The kernel has no bounds guard: one thread per element, whole groups.
    if (n % 256 != 0)
        return false;
    if (src->buffer && !is_ours(src->buffer->buft))
        return false;
    if (op->buffer && !is_ours(op->buffer->buft))
        return false;
    return true;
}

// Narrow z-0 dispatch record (see header): log-only, default off. The flag
// lives here (ahead of the matmul record path); the setter sits beside the
// other switches further down, in xllama namespace for linkage.
static bool g_z0_dispatch_log = false;
static int g_spin_wait_us = -1; // -1 = historical unbounded spin (see ggml_d3d12.h)
// Narrow z-0 placement pin (see header + setter beside the other switches).
static bool g_z0_pin_cpu = false;

// --- Backend ---

const char* backend_name(ggml_backend_t) {
    return "D3D12";
}
void backend_free(ggml_backend_t b) {
    Gpu& g = gpu();
    {
        std::lock_guard<std::mutex> lock(g.mu);
        char msg[256];
        char gpu_part[128];
        if (g.timing.valid == 0) {
            std::snprintf(gpu_part, sizeof(gpu_part), "timing unavailable: %llu calls",
                          static_cast<unsigned long long>(g.timing.unavailable));
        } else if (g.timing.unavailable > 0) {
            std::snprintf(gpu_part, sizeof(gpu_part),
                          "%.1f ms GPU partial: ts_valid=%llu ts_unavailable=%llu",
                          g.timing.total_ms, static_cast<unsigned long long>(g.timing.valid),
                          static_cast<unsigned long long>(g.timing.unavailable));
        } else {
            std::snprintf(gpu_part, sizeof(gpu_part), "%.1f ms GPU: ts_valid=%llu",
                          g.timing.total_ms, static_cast<unsigned long long>(g.timing.valid));
        }
        std::snprintf(msg, sizeof(msg),
                      "[xllama] d3d12: %llu graph_compute calls, %llu matmuls (%llu 2col), "
                      "%.1f ms wall (%s)\n",
                      static_cast<unsigned long long>(g.n_calls),
                      static_cast<unsigned long long>(g.n_matmuls),
                      static_cast<unsigned long long>(g.n_matmuls_2col), g.wall_ms, gpu_part);
        log_output(msg);
        if (g.n_gdn > 0) {
            std::snprintf(msg, sizeof(msg), "[xllama] d3d12: %llu GATED_DELTA_NET dispatches\n",
                          static_cast<unsigned long long>(g.n_gdn));
            log_output(msg);
        }
        g.n_gdn = 0;
        if (g.n_q6_tiled[0] || g.n_q6_tiled[1]) {
            std::snprintf(msg, sizeof(msg), "[xllama] d3d12: Q6_K tiles 2col=%llu 4col=%llu\n",
                          static_cast<unsigned long long>(g.n_q6_tiled[0]),
                          static_cast<unsigned long long>(g.n_q6_tiled[1]));
            log_output(msg);
        }
        if (g.n_swiglu) {
            std::snprintf(msg, sizeof(msg), "[xllama] d3d12: %llu FFN SWIGLU dispatches\n",
                          static_cast<unsigned long long>(g.n_swiglu));
            log_output(msg);
        }
        if (g.n_swiglu && (g_swiglu_range_gate.n || g_swiglu_range_up.n)) {
            auto log_range = [&](const char* op, const SwigluRangeStats& r) {
                std::snprintf(msg, sizeof(msg),
                              "[xllama] d3d12: SWIGLU %s range n=%llu min=%.3g max=%.3g "
                              "zero=%llu subn=%llu nf=%llu |x| buckets <1:%llu 1-10:%llu "
                              "10-30:%llu 30-87:%llu 87-126:%llu 126-192:%llu >192:%llu "
                              "pairs_sampled=%llu pairs_skipped=%llu\n",
                              op, (unsigned long long)r.n, r.minv, r.maxv,
                              (unsigned long long)r.zero, (unsigned long long)r.subn,
                              (unsigned long long)r.nf, (unsigned long long)r.b1,
                              (unsigned long long)r.b10, (unsigned long long)r.b30,
                              (unsigned long long)r.b87, (unsigned long long)r.b126,
                              (unsigned long long)r.b192, (unsigned long long)r.binf,
                              (unsigned long long)g_swiglu_range_sampled_pairs,
                              (unsigned long long)g_swiglu_range_skipped_pairs);
                log_output(msg);
            };
            log_range("gate", g_swiglu_range_gate);
            log_range("up", g_swiglu_range_up);
            g_swiglu_range_gate = SwigluRangeStats{};
            g_swiglu_range_up = SwigluRangeStats{};
            g_swiglu_range_sampled_pairs = 0;
            g_swiglu_range_skipped_pairs = 0;
        }
        if (g.n_rms_norm || g.n_mul || g.n_add) {
            std::snprintf(msg, sizeof(msg),
                          "[xllama] d3d12: %llu RMS_NORM dispatches, %llu MUL dispatches, "
                          "%llu ADD dispatches\n",
                          static_cast<unsigned long long>(g.n_rms_norm),
                          static_cast<unsigned long long>(g.n_mul),
                          static_cast<unsigned long long>(g.n_add));
            log_output(msg);
        }
        g.n_swiglu = 0;
        g.n_rms_norm = 0;
        g.n_mul = 0;
        g.n_add = 0;
        g.n_q6_tiled[0] = g.n_q6_tiled[1] = 0;
        g.n_calls = g.n_matmuls = g.n_matmuls_2col = 0;
        g.wall_ms = 0.0;
        g.timing.reset();
    }
    delete b;
}

ggml_status backend_graph_compute(ggml_backend_t, ggml_cgraph* cgraph) {
    Gpu& g = gpu();
    std::vector<const ggml_tensor*> ops;
    std::size_t n_mm = 0, n_gdn = 0;
    for (int i = 0; i < ggml_graph_n_nodes(cgraph); ++i) {
        const ggml_tensor* node = ggml_graph_node(cgraph, i);
        if (ggml_is_empty(node) || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0)
            continue;
        switch (node->op) {
        case GGML_OP_ADD:
            GGML_ASSERT(g_island_enabled && d3d12_add_supported(node));
            ops.push_back(node);
            break;
        case GGML_OP_RMS_NORM:
        case GGML_OP_MUL:
            // Plan 007 bounded chain: accepted only by the shared predicates
            // (the scheduler can only assign them here when they pass), so the
            // collector must take them too or the switch aborts.
            GGML_ASSERT(node->op == GGML_OP_RMS_NORM ? d3d12_rms_norm_supported(node)
                                                     : d3d12_mul_supported(node, nullptr));
            ops.push_back(node);
            break;
        case GGML_OP_MUL_MAT:
            ops.push_back(node);
            ++n_mm;
            break;
        case GGML_OP_GATED_DELTA_NET:
            GGML_ASSERT(d3d12_gdn_supported(node) && g.gdn_pso);
            ops.push_back(node);
            ++n_gdn;
            break;
        case GGML_OP_UNARY:
            // Selftest-only SILU (plan 006 C4): the op collector must accept it
            // here too, or the switch aborts before the dispatch branch runs.
            if (d3d12_silu_supported(node))
                ops.push_back(node);
            else
                GGML_ABORT("d3d12: unsupported unary op %s", ggml_op_desc(node));
            break;
        case GGML_OP_GLU:
            if (d3d12_swiglu_supported(node))
                ops.push_back(node);
            else
                GGML_ABORT("d3d12: unsupported glu op %s", ggml_op_desc(node));
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            break;
        default:
            GGML_ABORT("d3d12: unsupported op %s", ggml_op_desc(node));
        }
    }
    if (ops.empty())
        return GGML_STATUS_SUCCESS;

    std::lock_guard<std::mutex> lock(g.mu);
    std::vector<std::pair<const ggml_tensor*, const ggml_tensor*>> swiglu_range_pairs;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ts = g.ts && g.ts_rb && g_ts_enabled;
    g.timing.begin_call();
    const bool ran = run_now(g, [&](ID3D12GraphicsCommandList* cl) {
        cl->SetComputeRootSignature(g.root.Get());
        bool gdn_root_active = false;
        if (ts)
            cl->EndQuery(g.ts.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
        for (const ggml_tensor* node : ops) {
            if (node->op == GGML_OP_GATED_DELTA_NET) {
                const ggml_tensor* q = node->src[0];
                const ggml_tensor* k = node->src[1];
                const ggml_tensor* v = node->src[2];
                std::uint32_t c[16] = {static_cast<std::uint32_t>(v->ne[2]),
                                       static_cast<std::uint32_t>(node->op_params[0]),
                                       static_cast<std::uint32_t>(q->nb[1] / 4),
                                       static_cast<std::uint32_t>(q->nb[2] / 4),
                                       static_cast<std::uint32_t>(k->nb[1] / 4),
                                       static_cast<std::uint32_t>(k->nb[2] / 4),
                                       static_cast<std::uint32_t>(v->nb[1] / 4),
                                       static_cast<std::uint32_t>(v->nb[2] / 4),
                                       static_cast<std::uint32_t>(4096 * v->ne[2]),
                                       524288,
                                       32,
                                       16,
                                       0,
                                       0,
                                       0,
                                       0};
                const float scale = 1.0f / std::sqrt(128.0f);
                std::memcpy(c + 12, &scale, sizeof(scale));
                cl->SetComputeRootSignature(g.gdn_root.Get());
                gdn_root_active = true;
                cl->SetPipelineState(g.gdn_pso.Get());
                cl->SetComputeRoot32BitConstants(0, 16, c, 0);
                for (int i = 0; i < 6; ++i) {
                    GGML_ASSERT(node->src[i]->buffer && node->src[i]->buffer->buft == &kHostBuft);
                    cl->SetComputeRootUnorderedAccessView(i + 1, tensor_va(node->src[i]));
                }
                GGML_ASSERT(node->buffer && node->buffer->buft == &kHostBuft);
                cl->SetComputeRootUnorderedAccessView(7, tensor_va(node));
                cl->Dispatch(32, 1, 16);
                D3D12_RESOURCE_BARRIER b = {};
                b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                cl->ResourceBarrier(1, &b);
                continue;
            }
            if (node->op == GGML_OP_ADD && g_island_enabled && d3d12_add_supported(node)) {
                const std::uint32_t count = static_cast<std::uint32_t>(ggml_nelements(node));
                std::uint32_t c32[16] = {};
                c32[0] = count;
                cl->SetComputeRootSignature(g.gdn_root.Get());
                gdn_root_active = true;
                cl->SetPipelineState(g.add_pso.Get());
                cl->SetComputeRoot32BitConstants(0, 16, c32, 0);
                cl->SetComputeRootUnorderedAccessView(1, tensor_va(node));        // u0 dst
                cl->SetComputeRootUnorderedAccessView(2, tensor_va(node->src[0])); // u1
                cl->SetComputeRootUnorderedAccessView(3, tensor_va(node->src[1])); // u2
                cl->Dispatch(count / 256u, 1, 1);
                ++g.n_add;
                D3D12_RESOURCE_BARRIER b = {};
                b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                cl->ResourceBarrier(1, &b);
                continue;
            }
            if (node->op == GGML_OP_RMS_NORM && d3d12_rms_norm_supported(node)) {
                const ggml_tensor* sx = node->src[0];
                std::uint32_t c32[8] = {};
                const std::uint32_t ne00 = static_cast<std::uint32_t>(node->ne[0]);
                const std::uint32_t rows =
                    static_cast<std::uint32_t>(node->ne[1] * node->ne[2] * node->ne[3]);
                float eps = 0.0f;
                std::memcpy(&eps, node->op_params, sizeof(float));
                std::memcpy(&c32[2], &eps, sizeof(c32[2]));
                c32[0] = ne00;
                c32[1] = rows;
                cl->SetComputeRootSignature(g.root.Get());
                gdn_root_active = false;
                cl->SetPipelineState(g.rms_norm_pso.Get());
                cl->SetComputeRoot32BitConstants(0, 8, c32, 0);
                cl->SetComputeRootUnorderedAccessView(2, tensor_va(node)); // u0 dst
                cl->SetComputeRootUnorderedAccessView(3, tensor_va(sx));   // u1 src
                cl->Dispatch(rows, 1, 1); // one group per row; predicate caps 65535
                ++g.n_rms_norm;
                D3D12_RESOURCE_BARRIER b = {};
                b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                cl->ResourceBarrier(1, &b);
                continue;
            }
            if (node->op == GGML_OP_MUL) {
                // Predicate-gated; the scheduler only assigns it here when true.
                GGML_ASSERT(d3d12_mul_supported(node, nullptr));
                const ggml_tensor* a = node->src[0];
                const ggml_tensor* wt = node->src[1];
                const std::uint32_t count = static_cast<std::uint32_t>(ggml_nelements(node));
                std::uint32_t c32[8] = {};
                c32[0] = count;
                c32[1] = static_cast<std::uint32_t>(node->ne[0]);
                cl->SetComputeRootSignature(g.root.Get());
                gdn_root_active = false;
                cl->SetPipelineState(g.mul_pso.Get());
                cl->SetComputeRoot32BitConstants(0, 8, c32, 0);
                cl->SetComputeRootShaderResourceView(1, tensor_va(wt));   // t0 weight
                cl->SetComputeRootUnorderedAccessView(2, tensor_va(node)); // u0 dst
                cl->SetComputeRootUnorderedAccessView(3, tensor_va(a));    // u1 src
                cl->Dispatch(count / 256u, 1, 1); // predicate: <= 65535 groups
                ++g.n_mul;
                D3D12_RESOURCE_BARRIER b = {};
                b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                cl->ResourceBarrier(1, &b);
                continue;
            }
            if (node->op == GGML_OP_GLU && d3d12_swiglu_supported(node)) {
                // Selftest-only split-SWIGLU probe: gdn_root (u0..u6).
                cl->SetComputeRootSignature(g.gdn_root.Get());
                gdn_root_active = true;
                const ggml_tensor* gate = node->src[0];
                const ggml_tensor* up = node->src[1];
                const std::uint32_t count = static_cast<std::uint32_t>(ggml_nelements(node));
                cl->SetPipelineState(g.swiglu_probe_pso.Get());
                cl->SetComputeRootUnorderedAccessView(1, tensor_va(node)); // u0 output
                cl->SetComputeRootUnorderedAccessView(2, tensor_va(gate)); // u1 gate
                cl->SetComputeRootUnorderedAccessView(3, tensor_va(up));   // u2 up
                cl->Dispatch(count / 256u, 1, 1); // predicate: count % 256 == 0
                ++g.n_swiglu;
                if (g_swiglu_range_diag) {
                    if (swiglu_range_pairs.size() < 32)
                        swiglu_range_pairs.emplace_back(gate, up);
                    else
                        ++g_swiglu_range_skipped_pairs;
                }
                D3D12_RESOURCE_BARRIER b = {};
                b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                cl->ResourceBarrier(1, &b);
                continue;
            }
            if (node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_SILU) {
                if (gdn_root_active) {
                    cl->SetComputeRootSignature(g.root.Get());
                    gdn_root_active = false;
                }
                const ggml_tensor* sx = node->src[0];
                if (!sx || !sx->buffer || !is_ours(sx->buffer->buft) || !node->buffer ||
                    !is_ours(node->buffer->buft)) {
                    log_output("[xllama] d3d12: SILU selftest node buffers not ours; skipped\n");
                    continue;
                }
                const std::uint32_t count = static_cast<std::uint32_t>(ggml_nelements(node));
                cl->SetPipelineState(g.silu_pso.Get());
                // Input and output are UAVs: D3D12_Host tensors live in UAV
                // state, and the matmul path never binds an SRV to activations.
                // Output = u0 (param 2), input = u1 (param 3): the same slots
                // matmul uses for dst and activations.
                cl->SetComputeRootUnorderedAccessView(2, tensor_va(node));
                cl->SetComputeRootUnorderedAccessView(3, tensor_va(sx));
                cl->Dispatch(count / 256u, 1, 1); // predicate: count % 256 == 0
                D3D12_RESOURCE_BARRIER b = {};
                b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                cl->ResourceBarrier(1, &b);
                continue;
            }
            if (gdn_root_active) {
                cl->SetComputeRootSignature(g.root.Get());
                gdn_root_active = false;
            }
            const ggml_tensor* w = node->src[0];
            const ggml_tensor* x = node->src[1];
            GGML_ASSERT(w->buffer && w->buffer->buft == &kWeightsBuft);
            GGML_ASSERT(x->buffer && is_ours(x->buffer->buft));
            GGML_ASSERT(node->buffer && is_ours(node->buffer->buft));
            // Plan 004 two-column tile. Variant 1 (bench experiment) forces NEW
            // wherever applicable; variant 2 (real decode) additionally
            // requires the explicit allowlist, so unmeasured widths (B=4,
            // prefill) and measured regressions stay OLD. B=1 can never take
            // this branch: the B1 path is exactly OLD. ncols rides in constant
            // dword 6 (pad0 renamed ncols in the shared header; one-column
            // kernels never read it).
            const int kvar = d3d12_kernel_variant();
            const int bn = static_cast<int>(node->ne[1]);
            const int wide = d3d12_mm_threads(w->ne[0]) == kD3d12MmvThreadsLong ? 1 : 0;
            const int q6_cols =
                d3d12_q6_columns_for(static_cast<int>(w->ne[1]), static_cast<int>(w->ne[0]), bn);
            const int q6_index = q6_cols == 4 ? 1 : 0;
            const bool q6_tiled =
                q6_cols > 1 && g.pso_q6col[q6_index][wide] && w->type == GGML_TYPE_Q6_K &&
                d3d12_q6_allowlisted(static_cast<int>(w->ne[1]), static_cast<int>(w->ne[0]), bn);
            const bool two_col =
                g.pso2col[0] && g.pso2col[1] && w->type == GGML_TYPE_Q4_K &&
                ((kvar == 1 && bn >= 2) ||
                 (kvar == 2 && d3d12_2col_allowlisted(w->type, static_cast<int>(w->ne[1]),
                                                      static_cast<int>(w->ne[0]), bn)));
            const int columns = q6_tiled ? q6_cols : (two_col ? 2 : 1);
            const D3d12Dispatch d =
                columns == 4 ? d3d12_mm_dispatch_4col(node->ne[0], node->ne[1])
                             : (columns == 2 ? d3d12_mm_dispatch_2col(node->ne[0], node->ne[1])
                                             : d3d12_mm_dispatch(node->ne[0], node->ne[1]));
            const std::uint32_t c[8] = {static_cast<std::uint32_t>(w->ne[1]),
                                        static_cast<std::uint32_t>(w->ne[0]),
                                        static_cast<std::uint32_t>(w->ne[0] / kD3d12Chunk),
                                        static_cast<std::uint32_t>(w->nb[1]),
                                        static_cast<std::uint32_t>(x->nb[1] / sizeof(float)),
                                        static_cast<std::uint32_t>(node->nb[1] / sizeof(float)),
                                        columns > 1 ? static_cast<std::uint32_t>(node->ne[1]) : 0,
                                        0};
            // Narrow dispatch record (see header): matched by node name OR by
            // stable weight identity, so a renamed graph node is still caught
            // (the `via=` field then reveals the transform). Log-only.
            const bool z_by_node = std::strcmp(node->name, "z-0") == 0;
            const bool z_by_weight =
                d3d12_is_z0_weight(w->name, w->ne[0], w->ne[1], w->type == GGML_TYPE_Q4_K);
            if (g_z0_dispatch_log && (z_by_node || z_by_weight)) {
                // Narrow dispatch record: everything that selects the executed
                // kernel for this exact node. Log-only; numerics untouched.
                const bool x_in_weights = x->buffer && x->buffer->buft == &kWeightsBuft;
                const bool y_in_weights = node->buffer && node->buffer->buft == &kWeightsBuft;
                char lb[512];
                std::snprintf(lb, sizeof(lb),
                              "[xllama] d3d12 zdispatch via=%s node=%s w=%s[%s %lldx%lld] "
                              "x=%s[%s] y=[%s %lldx%lld] wbuf=%s xbuf=%s ybuf=%s "
                              "twocol=%d wide=%d threads=%d groups=%ux%u "
                              "c=[%u,%u,%u,%u,%u,%u,%u,%u]\n",
                              z_by_node ? "node-name" : "weight-identity",
                              node->name[0] != '\0' ? node->name : "-",
                              w->name[0] != '\0' ? w->name : "-", ggml_type_name(w->type),
                              static_cast<long long>(w->ne[0]), static_cast<long long>(w->ne[1]),
                              x->name[0] != '\0' ? x->name : "-", ggml_type_name(x->type),
                              ggml_type_name(node->type), static_cast<long long>(node->ne[0]),
                              static_cast<long long>(node->ne[1]), "weights",
                              x_in_weights ? "weights" : "host", y_in_weights ? "weights" : "host",
                              two_col ? 1 : 0, wide, d3d12_mm_threads(w->ne[0]), d.groups_x,
                              d.groups_y, c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);
                log_output(lb);
            }
            if (q6_tiled) {
                cl->SetPipelineState(g.pso_q6col[q6_index][wide].Get());
                ++g.n_q6_tiled[q6_index];
            } else if (two_col)
                cl->SetPipelineState(g.pso2col[wide].Get());
            else
                cl->SetPipelineState(g.pso[pso_for(w->type)][wide].Get());
            if (two_col)
                ++g.n_matmuls_2col;
            cl->SetComputeRoot32BitConstants(0, 8, c, 0);
            cl->SetComputeRootShaderResourceView(1, tensor_va(w));
            cl->SetComputeRootUnorderedAccessView(2, tensor_va(node));
            cl->SetComputeRootUnorderedAccessView(3, tensor_va(x));
            cl->Dispatch(d.groups_x, d.groups_y, 1);
            // A later matmul may read this output (LoRA chains); order them.
            D3D12_RESOURCE_BARRIER b = {};
            b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            cl->ResourceBarrier(1, &b);
        }
        if (ts) {
            cl->EndQuery(g.ts.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
            cl->ResolveQueryData(g.ts.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, g.ts_rb.Get(), 0);
        }
    });
    if (!ran)
        return GGML_STATUS_FAILED;
    if (g_swiglu_range_diag && !swiglu_range_pairs.empty()) {
        auto sample_into = [](SwigluRangeStats& r, const ggml_tensor* t) {
            if (!t || !t->data || !t->buffer || !is_ours(t->buffer->buft))
                return;
            const float* p = static_cast<const float*>(t->data);
            const std::size_t ne = ggml_nelements(t);
            for (std::size_t i = 0; i < ne; ++i) {
                const float v = p[i];
                const float a = std::fabs(v);
                r.n++;
                if (std::isnan(v) || std::isinf(v)) {
                    r.nf++;
                    continue;
                }
                if (a == 0.0f) {
                    r.zero++;
                    continue;
                }
                if (a > r.maxv)
                    r.maxv = a;
                if (a < r.minv)
                    r.minv = a;
                if (a < std::numeric_limits<float>::min())
                    r.subn++;
                if (a < 1.0f)
                    r.b1++;
                else if (a < 10.0f)
                    r.b10++;
                else if (a < 30.0f)
                    r.b30++;
                else if (a < 87.0f)
                    r.b87++;
                else if (a < 126.0f)
                    r.b126++;
                else if (a < 192.0f)
                    r.b192++;
                else
                    r.binf++;
            }
        };
        for (const auto& gu : swiglu_range_pairs) {
            sample_into(g_swiglu_range_gate, gu.first);
            sample_into(g_swiglu_range_up, gu.second);
            ++g_swiglu_range_sampled_pairs;
        }
        swiglu_range_pairs.clear();
    }
    if (ts) {
        void* p = nullptr;
        if (SUCCEEDED(g.ts_rb->Map(0, nullptr, &p))) {
            const auto* t = static_cast<const std::uint64_t*>(p);
            if (g.ts_freq > 0 && t[1] > t[0]) {
                g.timing.add_sample(1000.0 * static_cast<double>(t[1] - t[0]) /
                                    static_cast<double>(g.ts_freq));
            } else {
                // end<=start or zero frequency is not a measurement: report it
                // as unavailable instead of a valid zero.
                g.timing.add_unavailable();
            }
            g.ts_rb->Unmap(0, nullptr);
        } else {
            // A failed readback is "unavailable", never the previous sample.
            g.timing.add_unavailable();
        }
    } else {
        g.timing.add_unavailable();
    }
    ++g.n_calls;
    g.n_matmuls += n_mm;
    g.n_gdn += n_gdn;
    g.wall_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    // Plan 003 stage 2: shape counts into the aggregation buffer, keyed by
    // context + phase + shape. g.mu is already held here, so this only appends to
    // the histogram's OWN mutex-protected buffer — never logs and never takes a
    // second lock on g.mu. Nothing is emitted from the hot path; the boundary call
    // d3d12_shape_drain() logs outside both locks.
    if (!g_shape_off) {
        std::lock_guard<std::mutex> shlock(g_shape_mu);
        const char* ctx = t_ctx;
        const char* phase = t_phase;
        for (const ggml_tensor* node : ops) {
            if (node->op != GGML_OP_MUL_MAT)
                continue;
            const ggml_tensor* w = node->src[0];
            const int type = static_cast<int>(w->type);
            const long long n = static_cast<long long>(w->ne[1]);
            const long long k = static_cast<long long>(w->ne[0]);
            const long long b = static_cast<long long>(node->ne[1]);
            const int thr = d3d12_mm_threads(w->ne[0]);
            bool found = false;
            for (auto& e : g_shape_agg) {
                if (e.first.type == type && e.first.n == n && e.first.k == k && e.first.b == b &&
                    e.first.thr == thr && e.first.ctx == ctx && e.first.phase == phase) {
                    e.second += 1;
                    found = true;
                    break;
                }
            }
            if (!found)
                g_shape_agg.push_back({ShapeAggKey{ctx, phase, type, n, k, b, thr}, 1});
        }
    }
    return GGML_STATUS_SUCCESS;
}

ggml_guid_t backend_guid() {
    static ggml_guid guid = {0x78, 0x6c, 0x6c, 0x61, 0x6d, 0x61, 0x2d, 0x64,
                             0x33, 0x64, 0x31, 0x32, 0x2d, 0x76, 0x30, 0x31};
    return &guid;
}

const ggml_backend_i kBackendIface = {
    /* .get_name            = */ backend_name,
    /* .free                = */ backend_free,
    /* .set_tensor_async    = */ nullptr,
    /* .get_tensor_async    = */ nullptr,
    /* .set_tensor_2d_async = */ nullptr,
    /* .get_tensor_2d_async = */ nullptr,
    /* .cpy_tensor_async    = */ nullptr,
    /* .synchronize         = */ nullptr, // graph_compute completes before returning
    /* .graph_plan_create   = */ nullptr,
    /* .graph_plan_free     = */ nullptr,
    /* .graph_plan_update   = */ nullptr,
    /* .graph_plan_compute  = */ nullptr,
    /* .graph_compute       = */ backend_graph_compute,
    /* .event_record        = */ nullptr,
    /* .event_wait          = */ nullptr,
    /* .graph_optimize      = */ nullptr,
};

// --- Device ---

const char* dev_name(ggml_backend_dev_t) {
    return "D3D12";
}
const char* dev_description(ggml_backend_dev_t) {
    return "xllama D3D12 compute (Q4_0/Q4_K/Q5_K/Q6_K matmul)";
}
void dev_memory(ggml_backend_dev_t, size_t* free, size_t* total) {
    *free = *total = 0;
    Gpu& g = gpu();
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter3> adapter;
    if (!g.ok || FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) ||
        FAILED(factory->EnumAdapterByLuid(g.luid, IID_PPV_ARGS(&adapter))))
        return;
    DXGI_QUERY_VIDEO_MEMORY_INFO info = {};
    if (SUCCEEDED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
        *total = static_cast<size_t>(info.Budget);
        *free = info.Budget > info.CurrentUsage
                    ? static_cast<size_t>(info.Budget - info.CurrentUsage)
                    : 0;
    }
}
enum ggml_backend_dev_type dev_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}
void dev_props(ggml_backend_dev_t dev, ggml_backend_dev_props* props) {
    props->name = dev_name(dev);
    props->description = dev_description(dev);
    props->type = dev_type(dev);
    dev_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {/* async */ false, /* host_buffer */ false, /* buffer_from_host_ptr */ false,
                   /* events */ false, /* mmap_support */ false};
}
ggml_backend_t dev_init_backend(ggml_backend_dev_t dev, const char*) {
    if (!gpu().ok)
        return nullptr;
    return new ggml_backend{backend_guid(), kBackendIface, dev, nullptr};
}
ggml_backend_buffer_type_t dev_buffer_type(ggml_backend_dev_t) {
    return &kHostBuft;
}

bool dev_supports_op(ggml_backend_dev_t, const ggml_tensor* op) {
    switch (op->op) {
    case GGML_OP_ADD:
        return g_island_enabled && d3d12_add_supported(op);
    case GGML_OP_RMS_NORM:
        return d3d12_rms_norm_supported(op);
    case GGML_OP_MUL:
        return d3d12_mul_supported(op, nullptr);
    case GGML_OP_UNARY:
        // Selftest-only SILU (plan 006 C4): one shared predicate with the
        // collector and the dispatch.
        return d3d12_silu_supported(op);
    case GGML_OP_GLU:
        // Selftest-only split SWIGLU probe (plan 006 C4).
        return d3d12_swiglu_supported(op);
    case GGML_OP_GATED_DELTA_NET:
        return gpu().gdn_pso && d3d12_gdn_supported(op);
    case GGML_OP_NONE:
    case GGML_OP_RESHAPE:
    case GGML_OP_VIEW:
    case GGML_OP_PERMUTE:
    case GGML_OP_TRANSPOSE:
        return true;
    case GGML_OP_MUL_MAT: {
        // Narrow z-0 placement pin (first-divergence diagnosis): refuse exactly
        // the z-0 node so it falls back to the CPU kernel path, testing whether
        // the B3/B4 width variation follows placement. Matched by node name OR
        // by stable weight identity (name + shape + type), so a renamed graph
        // node cannot silently dodge it; the hit line says which matcher fired.
        // Opt-in via knob (default off); every other node is untouched.
        if (g_z0_pin_cpu) {
            const ggml_tensor* w0 = op->src[0];
            const bool by_node = std::strcmp(op->name, "z-0") == 0;
            const bool by_weight =
                w0 != nullptr &&
                d3d12_is_z0_weight(w0->name, w0->ne[0], w0->ne[1], w0->type == GGML_TYPE_Q4_K);
            if (by_node || by_weight) {
                char lb[256];
                std::snprintf(
                    lb, sizeof(lb),
                    "[xllama] d3d12: zpin REFUSED node=%s via=%s "
                    "w=%s[%s %lldx%lld]\n",
                    op->name[0] != '\0' ? op->name : "-", by_node ? "node-name" : "weight-identity",
                    w0 != nullptr ? w0->name : "-", w0 != nullptr ? ggml_type_name(w0->type) : "?",
                    w0 != nullptr ? static_cast<long long>(w0->ne[0]) : -1,
                    w0 != nullptr ? static_cast<long long>(w0->ne[1]) : -1);
                log_output(lb);
                return false;
            }
        }
        const ggml_tensor* w = op->src[0];
        const ggml_tensor* x = op->src[1];
        D3d12MatmulDesc d;
        d.src0_type = w->type;
        d.ne00 = w->ne[0];
        d.ne01 = w->ne[1];
        d.ne02 = w->ne[2];
        d.ne03 = w->ne[3];
        d.ne10 = x->ne[0];
        d.ne11 = x->ne[1];
        d.ne12 = x->ne[2];
        d.ne13 = x->ne[3];
        d.src1_type = x->type;
        d.dst_type = op->type;
        d.src0_contiguous = ggml_is_contiguous(w);
        d.src1_contiguous = ggml_is_contiguous(x);
        d.src0_in_weight_buffer = w->buffer && w->buffer->buft == &kWeightsBuft;
        const bool ok = d3d12_mm_supported(d);
        // A refused matmul whose weight already sits in D3D12_Weights means a
        // placement the backend then cannot run — log each distinct shape once
        // (D2b #309; was first-8-only, which hid all but the first shape and
        // made per-op backend attribution impossible).
        static std::set<std::string> logged_shapes;
        if (!ok && w->buffer && w->buffer->buft == &kWeightsBuft) {
            char msg[256];
            std::snprintf(msg, sizeof(msg),
                          "[xllama] d3d12: MUL_MAT refused: %s %lldx%lld x [%lld,%lld,%lld,%lld] "
                          "src1=%s dst=%s\n",
                          ggml_type_name(w->type), static_cast<long long>(w->ne[0]),
                          static_cast<long long>(w->ne[1]), static_cast<long long>(x->ne[0]),
                          static_cast<long long>(x->ne[1]), static_cast<long long>(x->ne[2]),
                          static_cast<long long>(x->ne[3]), ggml_type_name(x->type),
                          ggml_type_name(op->type));
            if (logged_shapes.insert(msg).second)
                log_output(msg);
        }
        return ok;
    }
    default:
        return false;
    }
}
bool dev_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) {
    return is_ours(buft);
}

const ggml_backend_device_i kDeviceIface = {
    /* .get_name             = */ dev_name,
    /* .get_description      = */ dev_description,
    /* .get_memory           = */ dev_memory,
    /* .get_type             = */ dev_type,
    /* .get_props            = */ dev_props,
    /* .init_backend         = */ dev_init_backend,
    /* .get_buffer_type      = */ dev_buffer_type,
    // The default buft doubles as the host buft: llama.cpp then allocates the
    // CPU backend's compute buffer in D3D12_Host, which supports_buft accepts,
    // so the scheduler stops copying every matmul input into a second buffer.
    /* .get_host_buffer_type = */ dev_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ dev_supports_op,
    /* .supports_buft        = */ dev_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

// --- Registry ---

const char* reg_name(ggml_backend_reg_t) {
    return "D3D12";
}
size_t reg_device_count(ggml_backend_reg_t) {
    return 1;
}
ggml_backend_dev_t reg_device(ggml_backend_reg_t, size_t index) {
    GGML_ASSERT(index == 0);
    return device_ptr();
}

// Weights live in the extra buft; llama.cpp lists it after the default one.
ggml_backend_buffer_type_t* dev_extra_bufts(ggml_backend_dev_t) {
    static ggml_backend_buffer_type_t list[] = {&kWeightsBuft, nullptr};
    return list;
}

void* reg_proc_address(ggml_backend_reg_t, const char* name) {
    if (std::strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0)
        return reinterpret_cast<void*>(dev_extra_bufts);
    return nullptr;
}

const ggml_backend_reg_i kRegIface = {
    /* .get_name         = */ reg_name,
    /* .get_device_count = */ reg_device_count,
    /* .get_device       = */ reg_device,
    /* .get_proc_address = */ reg_proc_address,
};

ggml_backend_reg kReg = {
    /* .api_version = */ GGML_BACKEND_API_VERSION,
    /* .iface       = */ kRegIface,
    /* .context     = */ nullptr,
};

ggml_backend_device kDevice = {
    /* .iface   = */ kDeviceIface,
    /* .reg     = */ &kReg,
    /* .context = */ nullptr,
};

ggml_backend_dev_t device_ptr() {
    return &kDevice;
}

} // namespace

bool ggml_d3d12_register() {
    static std::once_flag once;
    static bool registered = false;
    std::call_once(once, [] {
        Gpu& g = gpu();
        if (!g.ok) {
            log_output(("[xllama] d3d12: backend unavailable: " + g.error + "\n").c_str());
            return;
        }
        kHostBuft.device = &kDevice;
        kWeightsBuft.device = &kDevice;
        ggml_backend_register(&kReg);
        registered = true;
    });
    return registered;
}

// --- Selftest ---

namespace {

struct SelftestCase {
    ggml_type type;
    const char* name;
    int n, k, ncols;
};

// RAII variant scope: the bench sets OLD/NEW around timed runs and always
// restores OLD, so a variant never leaks from one case (or into real decode).
struct KernelVariantGuard {
    explicit KernelVariantGuard(int v) {
        d3d12_set_kernel_variant(v);
    }
    ~KernelVariantGuard() {
        d3d12_set_kernel_variant(0);
    }
};

struct Q6ColumnsGuard {
    int previous = d3d12_q6_columns();
    Q6ColumnsGuard() {
        d3d12_set_q6_columns(1);
    }
    ~Q6ColumnsGuard() {
        d3d12_set_q6_columns(previous);
    }
};

// Padding sentinel for the strided gate (plan 004): exact float equality is
// the check, so it must survive memcpy/set/get bit-identically (it does) and
// must never be produced by real computation (1e30 is far outside the
// uniform(-1,1) draws and their products).
constexpr float kPadSentinel = 1e30f;
constexpr std::uint8_t kPadByte = 0xAB;

// Exact-equality padding checks on device readback: any kernel write outside
// the valid region fails the row, even when the numerics match. ggml_nbytes
// covers [0, last valid element] — the FINAL row/column has no padding storage
// (neither host vector nor device buffer holds it), so only rows/cols [0, n)
// resp. [0, ncols) carry checkable padding. The kernel's address range stays
// inside the buffer by construction (col < ncols, row < n); these canaries
// catch stride miscalculations short of an overrun.
bool check_float_pad(const std::vector<float>& v, std::size_t stride, int ncols, int n_valid) {
    for (int c = 0; c + 1 < ncols; ++c)
        for (std::size_t i = static_cast<std::size_t>(n_valid); i < stride; ++i)
            if (v[static_cast<std::size_t>(c) * stride + i] != kPadSentinel)
                return false;
    return true;
}

bool check_byte_pad(const std::vector<std::uint8_t>& q, std::size_t nb1, std::size_t row_bytes,
                    int n) {
    for (int r = 0; r + 1 < n; ++r)
        for (std::size_t i = row_bytes; i < nb1; ++i)
            if (q[static_cast<std::size_t>(r) * nb1 + i] != kPadByte)
                return false;
    return true;
}

// `row_pitch` is the byte distance between weight-row starts in `q` — the
// PACKED row size for contiguous tensors, nb[1] for strided ones. Passing the
// packed size for a strided q misaligns every row past the first: the padded
// gate caught exactly this (rev58, rel_err=1 on both variants while the
// kernels agreed with each other).
double reference_rel_err(ggml_type t, const std::vector<std::uint8_t>& q, std::size_t row_pitch,
                         const std::vector<float>& x, const std::vector<float>& y, int n, int k,
                         int ncols, std::size_t x_stride = 0, std::size_t y_stride = 0) {
    const auto* traits = ggml_get_type_traits(t);
    const std::size_t xs = x_stride == 0 ? static_cast<std::size_t>(k) : x_stride;
    const std::size_t ys = y_stride == 0 ? static_cast<std::size_t>(n) : y_stride;
    std::vector<float> wrow(static_cast<std::size_t>(k));
    double max_ref = 0.0, max_diff = 0.0;
    for (int r = 0; r < n; ++r) {
        traits->to_float(q.data() + static_cast<std::size_t>(r) * row_pitch, wrow.data(), k);
        for (int c = 0; c < ncols; ++c) {
            double acc = 0.0;
            const float* xc = x.data() + static_cast<std::size_t>(c) * xs;
            for (int i = 0; i < k; ++i)
                acc += static_cast<double>(wrow[static_cast<std::size_t>(i)]) * xc[i];
            const double got = y[static_cast<std::size_t>(c) * ys + r];
            max_ref = std::max(max_ref, std::fabs(acc));
            max_diff = std::max(max_diff, std::fabs(acc - got));
        }
    }
    return max_ref > 0.0 ? max_diff / max_ref : max_diff;
}

D3d12SelftestRow run_case(ggml_backend_t backend, const SelftestCase& sc, int repeats) {
    D3d12SelftestRow row;
    row.type = sc.name;
    row.n = sc.n;
    row.k = sc.k;
    row.ncols = sc.ncols;
    KernelVariantGuard vg(0); // gate and single-variant rows always run OLD
    Q6ColumnsGuard q6g;

    ggml_init_params ip = {};
    ip.mem_size = 8 * ggml_tensor_overhead() + ggml_graph_overhead();
    ip.no_alloc = true;
    ggml_context* ctx_w = ggml_init(ip);
    ggml_context* ctx_x = ggml_init(ip);
    ggml_context* ctx_g = ggml_init(ip);
    ggml_tensor* w = ggml_new_tensor_2d(ctx_w, sc.type, sc.k, sc.n);
    ggml_tensor* x = ggml_new_tensor_2d(ctx_x, GGML_TYPE_F32, sc.k, sc.ncols);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, &kWeightsBuft);
    ggml_backend_buffer_t xbuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_x, &kHostBuft);
    ggml_gallocr_t galloc = nullptr;

    auto cleanup = [&] {
        if (galloc)
            ggml_gallocr_free(galloc);
        if (wbuf)
            ggml_backend_buffer_free(wbuf);
        if (xbuf)
            ggml_backend_buffer_free(xbuf);
        ggml_free(ctx_g);
        ggml_free(ctx_x);
        ggml_free(ctx_w);
    };
    if (!wbuf || !xbuf) {
        row.error = "buffer allocation failed";
        cleanup();
        return row;
    }

    // Seed depends on the SHAPE only, never on ncols (B): the microbenchmark needs
    // the same weights and activations across B so a timing difference is the batch
    // width, not different data. A seed that moves with B would compare two
    // different matrices and call it T_target(B).
    std::mt19937 rng(1234u + static_cast<unsigned>(sc.n + sc.k));
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    // Bounded generation: a full n*k fp32 source would be 2542.8 MB for the
    // 248320x2560 Q6_K lm_head. Chunked rows quantize to the same bytes as one
    // call over the whole matrix, and the rng ends where the one-shot path
    // would, so xf below draws the same numbers.
    std::vector<std::uint8_t> q(ggml_nbytes(w));
    if (!d3d12_gen_weights(sc.type, sc.n, sc.k, rng, q.data())) {
        row.error = "weight generation failed";
        cleanup();
        return row;
    }
    row.weight_mb = static_cast<double>(q.size()) / 1e6;
    ggml_backend_tensor_set(w, q.data(), 0, q.size());
    std::vector<float> xf(static_cast<std::size_t>(sc.k) * sc.ncols);
    for (float& v : xf)
        v = uni(rng);
    ggml_backend_tensor_set(x, xf.data(), 0, xf.size() * sizeof(float));

    ggml_tensor* y = ggml_mul_mat(ctx_g, w, x);
    ggml_cgraph* gf = ggml_new_graph(ctx_g);
    ggml_build_forward_expand(gf, y);
    galloc = ggml_gallocr_new(&kHostBuft);
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        row.error = "graph allocation failed";
        cleanup();
        return row;
    }
    // Warm-up, then the timed block: `repeats` consecutive timed runs. The row
    // reports the median with the max-min range, not one exploratory sample.
    // repeats=1 keeps the gate rows on their exact single-sample methodology.
    if (repeats < 1)
        repeats = 1;
    // GPU-time self-test: force real samples (diagnostic path, never product
    // decode) and fail the row if any timed call reports unavailable timing,
    // instead of silently treating it as a measured zero.
    d3d12_set_gpu_timestamps(true);
    const std::uint64_t ts_un0 = gpu().timing.unavailable;
    const std::uint64_t tc0 = d3d12_2col_matmuls();
    ggml_status st = ggml_backend_graph_compute(backend, gf);
    std::vector<double> gpu_t, wall_t;
    for (int i = 0; i < repeats && st == GGML_STATUS_SUCCESS; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        st = ggml_backend_graph_compute(backend, gf);
        if (st != GGML_STATUS_SUCCESS)
            break;
        gpu_t.push_back(gpu().timing.last_ms);
        wall_t.push_back(
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                .count());
    }
    const bool ts_complete = gpu().timing.unavailable == ts_un0;
    row.d3d12_ran = ts_complete && st == GGML_STATUS_SUCCESS && !gpu_t.empty();
    if (!row.d3d12_ran) {
        row.error = ts_complete ? "graph_compute failed" : "gpu timing unavailable";
        cleanup();
        return row;
    }
    row.gpu_ms = d3d12_block_median(gpu_t);
    row.wall_ms = d3d12_block_median(wall_t);
    row.gpu_ms_range = d3d12_block_range(gpu_t);
    row.wall_ms_range = d3d12_block_range(wall_t);
    row.twocol = d3d12_2col_matmuls() - tc0;
    row.packed_gbs = row.gpu_ms > 0.0 ? static_cast<double>(q.size()) / 1e6 / row.gpu_ms : 0.0;
    row.peak_ws_mb = static_cast<double>(peak_working_set_mb());
    std::vector<float> yf(static_cast<std::size_t>(sc.n) * sc.ncols);
    ggml_backend_tensor_get(y, yf.data(), 0, yf.size() * sizeof(float));
    row.rel_err = reference_rel_err(sc.type, q, w->nb[1], xf, yf, sc.n, sc.k, sc.ncols);
    row.ok = row.rel_err <= kD3d12SelftestRelTol;
    if (!row.ok)
        row.error = "mismatch vs ggml dequant reference";
    cleanup();
    return row;
}

// Paired OLD/NEW block (plan 004, q4_k ncols >= 2 only). Setup mirrors
// run_case above (same seed encoding, same tensors); the gate path there is
// intentionally not shared, so it stays byte-identical. Tensors are allocated
// once and every pair runs on them: warmup OLD, warmup NEW, then `pairs`
// pairs with the starting variant alternating per pair and case_index, then
// one untimed verification run per variant for rel_err. Emits 2 rows per pair
// (one timed sample each); the script aggregates pairs and variances.
void run_paired_case(ggml_backend_t backend, ggml_type type, const char* name, int n, int k,
                     int ncols, int pairs, int case_index, std::vector<D3d12SelftestRow>* out,
                     int x_pad = 0, int y_pad = 0, std::size_t w_pad = 0, int q6_columns = 2) {
    if (!out || pairs < 1)
        return;
    KernelVariantGuard vg(0);
    Q6ColumnsGuard q6g;
    const bool q6 = type == GGML_TYPE_Q6_K;
    D3d12SelftestRow base;
    base.type = name ? name : "-";
    base.n = n;
    base.k = k;
    base.ncols = ncols;
    base.pads = std::to_string(x_pad) + "/" + std::to_string(y_pad) + "/" + std::to_string(w_pad);
    auto fail = [&](const char* msg) {
        D3d12SelftestRow r = base;
        r.variant = -1;
        r.error = msg;
        out->push_back(std::move(r));
    };
    if (x_pad < 0 || y_pad < 0) {
        fail("negative padding");
        return;
    }
    if ((type != GGML_TYPE_Q4_K && !q6) || ncols < 2 ||
        (q6 && !d3d12_q6_allowlisted(n, k, ncols))) {
        fail("paired blocks need q4_k B>=2 or the Q6 LM-head allowlist");
        return;
    }
    if (q6 ? !d3d12_q6_tiled_available(q6_columns) : !d3d12_2col_available()) {
        fail("requested tile PSOs unavailable");
        return;
    }

    ggml_init_params ip = {};
    ip.mem_size = 8 * ggml_tensor_overhead() + ggml_graph_overhead();
    ip.no_alloc = true;
    ggml_context* ctx_w = ggml_init(ip);
    ggml_context* ctx_x = ggml_init(ip);
    ggml_context* ctx_g = ggml_init(ip);
    ggml_tensor* w = ggml_new_tensor_2d(ctx_w, type, k, n);
    ggml_tensor* x = ggml_new_tensor_2d(ctx_x, GGML_TYPE_F32, k, ncols);
    // Stride padding (plan 004 padded gate): widened row strides BEFORE any
    // buffer is sized, so ggml_nbytes accounts the padding. supports_op would
    // route strided inputs to CPU in production (it requires contiguity); the
    // bench bypasses it with direct graph_compute like every run_case row, to
    // prove the kernels honor nb01/nb11 regardless.
    const std::size_t row_bytes = ggml_row_size(type, k);
    const std::size_t w_nb1 = row_bytes + w_pad;
    const std::size_t x_stride_f = static_cast<std::size_t>(k) + static_cast<std::size_t>(x_pad);
    const std::size_t y_stride_f = static_cast<std::size_t>(n) + static_cast<std::size_t>(y_pad);
    const bool padded = x_pad != 0 || y_pad != 0 || w_pad != 0;
    w->nb[1] = w_nb1;
    x->nb[1] = x_stride_f * sizeof(float);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, &kWeightsBuft);
    ggml_backend_buffer_t xbuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_x, &kHostBuft);
    ggml_backend_buffer_t ybuf = nullptr;
    ggml_gallocr_t galloc = nullptr;

    auto cleanup = [&] {
        if (galloc)
            ggml_gallocr_free(galloc);
        if (ybuf)
            ggml_backend_buffer_free(ybuf);
        if (wbuf)
            ggml_backend_buffer_free(wbuf);
        if (xbuf)
            ggml_backend_buffer_free(xbuf);
        ggml_free(ctx_g);
        ggml_free(ctx_x);
        ggml_free(ctx_w);
    };
    if (!wbuf || !xbuf) {
        fail("buffer allocation failed");
        cleanup();
        return;
    }

    std::mt19937 rng(1234u + static_cast<unsigned>(n + k));
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    // Same draws as the contiguous case (same seed, same count, same order):
    // generate flat, then lay rows at the strided pitch over a sentinel fill,
    // so any weight-padding byte the kernel touches is visible on readback.
    std::vector<std::uint8_t> qflat(static_cast<std::size_t>(n) * row_bytes);
    if (!d3d12_gen_weights(type, n, k, rng, qflat.data())) {
        fail("weight generation failed");
        cleanup();
        return;
    }
    std::vector<std::uint8_t> q(ggml_nbytes(w), kPadByte);
    for (int r = 0; r < n; ++r)
        std::memcpy(q.data() + static_cast<std::size_t>(r) * w_nb1,
                    qflat.data() + static_cast<std::size_t>(r) * row_bytes, row_bytes);
    ggml_backend_tensor_set(w, q.data(), 0, q.size());
    std::vector<float> xf(ggml_nbytes(x) / sizeof(float), kPadSentinel);
    for (int c = 0; c < ncols; ++c)
        for (int i = 0; i < k; ++i)
            xf[static_cast<std::size_t>(c) * x_stride_f + static_cast<std::size_t>(i)] = uni(rng);
    ggml_backend_tensor_set(x, xf.data(), 0, xf.size() * sizeof(float));

    ggml_tensor* y = ggml_mul_mat(ctx_g, w, x);
    ggml_cgraph* gf = ggml_new_graph(ctx_g);
    ggml_build_forward_expand(gf, y);
    if (!padded) {
        galloc = ggml_gallocr_new(&kHostBuft);
        if (!ggml_gallocr_alloc_graph(galloc, gf)) {
            fail("graph allocation failed");
            cleanup();
            return;
        }
    } else {
        // Padded Y cannot come from gallocr (it packs outputs tightly, like
        // production): assign the padded pitch, then size the buffer from it.
        // The kernel only sees (VA, y_stride), the same mechanism either way.
        y->nb[1] = y_stride_f * sizeof(float);
        ybuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_g, &kHostBuft);
        if (!ybuf) {
            fail("graph allocation failed");
            cleanup();
            return;
        }
    }
    const std::size_t y_floats = ggml_nbytes(y) / sizeof(float);
    const std::size_t x_floats = ggml_nbytes(x) / sizeof(float);
    if (padded) {
        // Device Y padding starts uninitialized: seed it with the sentinel so
        // the check below is meaningful. Untimed; valid outputs are fully
        // overwritten by every compute.
        std::vector<float> yinit(y_floats, kPadSentinel);
        ggml_backend_tensor_set(y, yinit.data(), 0, yinit.size() * sizeof(float));
    }

    auto select_variant = [&](int variant) {
        if (q6)
            d3d12_set_q6_columns(variant == 0 ? 1 : q6_columns);
        else
            d3d12_set_kernel_variant(variant);
    };
    auto tiled_count = [&] { return q6 ? d3d12_q6_tiled_matmuls() : d3d12_2col_matmuls(); };
    // Paired variant timing is a GPU-time self-test: force real samples and
    // reject any call whose timing is unavailable, never a silent zero.
    d3d12_set_gpu_timestamps(true);
    auto run_once = [&](int variant, double* gpu_ms, double* wall_ms) {
        select_variant(variant);
        const std::uint64_t un0 = gpu().timing.unavailable;
        const auto t0 = std::chrono::steady_clock::now();
        const ggml_status st = ggml_backend_graph_compute(backend, gf);
        if (st != GGML_STATUS_SUCCESS)
            return false;
        *gpu_ms = gpu().timing.last_ms;
        *wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                       .count();
        return gpu().timing.unavailable == un0;
    };
    double g0 = 0.0, w0 = 0.0;
    if (!run_once(0, &g0, &w0) || !run_once(1, &g0, &w0)) {
        fail("variant warmup failed");
        cleanup();
        return;
    }
    struct PairT {
        double old_gpu, old_wall, new_gpu, new_wall;
        std::uint64_t old_tc = 0, new_tc = 0;
    };
    std::vector<PairT> pts;
    for (int p = 0; p < pairs; ++p) {
        PairT pt{};
        const int first = (p + case_index) % 2; // alternate the starting variant
        for (int s = 0; s < 2; ++s) {
            const int v = (s == 0) ? first : 1 - first;
            double g = 0.0, wl = 0.0;
            const std::uint64_t tc0 = tiled_count();
            if (!run_once(v, &g, &wl)) {
                fail("paired compute failed");
                cleanup();
                return;
            }
            const std::uint64_t tc = tiled_count() - tc0;
            if ((v == 0 && tc != 0) || (v == 1 && tc != 1)) {
                fail("requested tile did not execute exactly once");
                cleanup();
                return;
            }
            if (v == 0) {
                pt.old_gpu = g;
                pt.old_wall = wl;
                pt.old_tc = tc;
            } else {
                pt.new_gpu = g;
                pt.new_wall = wl;
                pt.new_tc = tc;
            }
        }
        pts.push_back(pt);
    }
    // Untimed verification per variant against the unchanged threshold. The
    // reference dequantizes at the PACKED row pitch (not the strided nb[1])
    // with strided X/Y; padding regions are checked against sentinels on
    // device readback, so any out-of-bounds kernel write fails the row even
    // when the numerics match. Each region reports separately: a combined
    // flag once hid which of X/Y/W moved.
    std::vector<float> yf_old(y_floats, kPadSentinel);
    std::vector<float> yf_new(y_floats, kPadSentinel);
    std::vector<float> xf_old(x_floats, kPadSentinel);
    std::vector<float> xf_new(x_floats, kPadSentinel);
    std::vector<std::uint8_t> q_old(q.size(), 0);
    std::vector<std::uint8_t> q_new(q.size(), 0);
    std::vector<float> yinit(y_floats, kPadSentinel);
    ggml_backend_tensor_set(y, yinit.data(), 0, yinit.size() * sizeof(float));
    select_variant(0);
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fail("OLD verification failed");
        cleanup();
        return;
    }
    ggml_backend_tensor_get(y, yf_old.data(), 0, yf_old.size() * sizeof(float));
    ggml_backend_tensor_get(x, xf_old.data(), 0, xf_old.size() * sizeof(float));
    ggml_backend_tensor_get(w, q_old.data(), 0, q_old.size());
    ggml_backend_tensor_set(y, yinit.data(), 0, yinit.size() * sizeof(float));
    select_variant(1);
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fail("NEW verification failed");
        cleanup();
        return;
    }
    ggml_backend_tensor_get(y, yf_new.data(), 0, yf_new.size() * sizeof(float));
    ggml_backend_tensor_get(x, xf_new.data(), 0, xf_new.size() * sizeof(float));
    ggml_backend_tensor_get(w, q_new.data(), 0, q_new.size());
    const double rel_old =
        reference_rel_err(type, q, w_nb1, xf, yf_old, n, k, ncols, x_stride_f, y_stride_f);
    const double rel_new =
        reference_rel_err(type, q, w_nb1, xf, yf_new, n, k, ncols, x_stride_f, y_stride_f);
    const bool y_old_ok = check_float_pad(yf_old, y_stride_f, ncols, n);
    const bool x_old_ok = check_float_pad(xf_old, x_stride_f, ncols, k);
    const bool w_old_ok = check_byte_pad(q_old, w_nb1, row_bytes, n);
    const bool y_new_ok = check_float_pad(yf_new, y_stride_f, ncols, n);
    const bool x_new_ok = check_float_pad(xf_new, x_stride_f, ncols, k);
    const bool w_new_ok = check_byte_pad(q_new, w_nb1, row_bytes, n);
    bool bit_exact = true;
    if (q6)
        for (int c = 0; c < ncols; ++c)
            bit_exact &= std::memcmp(yf_old.data() + c * y_stride_f, yf_new.data() + c * y_stride_f,
                                     n * sizeof(float)) == 0;

    const double wmb = static_cast<double>(q.size()) / 1e6;
    const double peak = static_cast<double>(peak_working_set_mb());
    for (int p = 0; p < pairs; ++p) {
        D3d12SelftestRow ro = base, rn = base;
        ro.variant = 0;
        ro.pair = p;
        ro.gpu_ms = pts[p].old_gpu;
        ro.wall_ms = pts[p].old_wall;
        ro.twocol = pts[p].old_tc;
        ro.packed_gbs = ro.gpu_ms > 0.0 ? wmb / ro.gpu_ms : 0.0;
        ro.weight_mb = wmb;
        ro.peak_ws_mb = peak;
        ro.rel_err = rel_old;
        ro.ok = rel_old <= kD3d12SelftestRelTol && y_old_ok && x_old_ok && w_old_ok;
        ro.d3d12_ran = true;
        if (rel_old > kD3d12SelftestRelTol)
            ro.error = "mismatch vs ggml dequant reference (OLD)";
        else if (!y_old_ok)
            ro.error = "y padding clobbered (OLD)";
        else if (!x_old_ok)
            ro.error = "x padding clobbered (OLD)";
        else if (!w_old_ok)
            ro.error = "w padding clobbered (OLD)";
        rn.variant = 1;
        rn.pair = p;
        rn.gpu_ms = pts[p].new_gpu;
        rn.wall_ms = pts[p].new_wall;
        rn.twocol = pts[p].new_tc;
        rn.packed_gbs = rn.gpu_ms > 0.0 ? wmb / rn.gpu_ms : 0.0;
        rn.weight_mb = wmb;
        rn.peak_ws_mb = peak;
        rn.rel_err = rel_new;
        rn.ok = rel_new <= kD3d12SelftestRelTol && y_new_ok && x_new_ok && w_new_ok && bit_exact;
        rn.d3d12_ran = true;
        if (rel_new > kD3d12SelftestRelTol)
            rn.error = "mismatch vs ggml dequant reference (NEW)";
        else if (!y_new_ok)
            rn.error = "y padding clobbered (NEW)";
        else if (!x_new_ok)
            rn.error = "x padding clobbered (NEW)";
        else if (!w_new_ok)
            rn.error = "w padding clobbered (NEW)";
        else if (!bit_exact)
            rn.error = "Q6 tiled output differs bitwise from OLD";
        out->push_back(std::move(ro));
        out->push_back(std::move(rn));
    }
    cleanup();
}

} // namespace

// Read-only counter snapshots. A caller takes a delta around one phase; the totals
// are never reset here, so a phase delta stays valid even though backend_free logs
// and zeroes them when the backend goes away.
std::uint64_t d3d12_graph_calls() {
    Gpu& g = gpu();
    std::lock_guard<std::mutex> lock(g.mu);
    return g.n_calls;
}

// Turns the per-shape histogram off/on. Separate from the decode-phase switch:
// the decode timers are cheap chrono reads, this one builds strings in the hot
// path, so they are measured and gated independently.
void d3d12_set_shape_log(bool off) {
    std::lock_guard<std::mutex> shlock(g_shape_mu);
    g_shape_off = off;
    if (off)
        g_shape_agg.clear();
}

void d3d12_set_scope(const char* ctx, const char* phase) {
    t_ctx = ctx ? ctx : "unknown";
    t_phase = phase ? phase : "unknown";
}

void d3d12_get_scope(const char** ctx, const char** phase) {
    if (ctx)
        *ctx = t_ctx;
    if (phase)
        *phase = t_phase;
}

// Drained at a phase or generation boundary, outside the compute lock and outside
// the decode loop. Returns false when the buffer did not fit the cap: that is a
// precondition failure of the ANALYSIS (the histogram is incomplete), not a
// complete histogram with a truncation note.
bool d3d12_shape_drain(const char* label) {
    std::vector<std::pair<ShapeAggKey, long long>> local;
    {
        std::lock_guard<std::mutex> shlock(g_shape_mu);
        local.swap(g_shape_agg);
    }
    if (local.empty())
        return true;
    // Paginated, no arbitrary cap: every key is emitted, across as many records as
    // it takes. Each record carries the key count and the total so a consumer can
    // verify completeness from the lines themselves instead of trusting a cap that
    // happened to be large enough. A truncated record would be an incomplete
    // analysis, so this never truncates — it pages.
    const std::size_t kPerLine = 12;
    const std::size_t n = local.size();
    for (std::size_t first = 0; first < n; first += kPerLine) {
        const std::size_t last = (first + kPerLine < n) ? first + kPerLine : n;
        std::string line = "[xllama] d3d12_shape label=";
        line += label ? label : "-";
        line += " part=" + std::to_string(first / kPerLine + 1) + "/" +
                std::to_string((n + kPerLine - 1) / kPerLine);
        line += " keys_total=" + std::to_string(n);
        for (std::size_t i = first; i < last; ++i) {
            const ShapeAggKey& k = local[i].first;
            line += " | ctx=" + k.ctx + " phase=" + k.phase + " t=" + std::to_string(k.type) +
                    " N=" + std::to_string(k.n) + " K=" + std::to_string(k.k) +
                    " B=" + std::to_string(k.b) + " thr=" + std::to_string(k.thr) +
                    " mm=" + std::to_string(local[i].second);
        }
        line += "\n";
        log_output(line.c_str());
    }
    return true;
}

std::uint64_t d3d12_matmul_count() {
    Gpu& g = gpu();
    std::lock_guard<std::mutex> lock(g.mu);
    return g.n_matmuls;
}

double d3d12_gpu_ms() {
    Gpu& g = gpu();
    std::lock_guard<std::mutex> lock(g.mu);
    return g.timing.total_ms;
}

void d3d12_set_gpu_timestamps(bool enabled) {
    g_ts_enabled = enabled;
}

bool d3d12_gpu_timestamps_enabled() {
    return g_ts_enabled;
}

void d3d12_set_silu_test_enabled(bool enabled) {
    g_silu_test = enabled;
}

bool d3d12_silu_test_enabled() {
    return g_silu_test;
}

void d3d12_set_swiglu_product_mode(int mode) {
    g_swiglu_requested_mode = mode;
    if (!g_swiglu_policy.bound)
        g_swiglu_policy.mode = mode; // pre-bind display; finalize binds later
}

// Binds the profile at the first context; bound calls are no-ops. Conflicting
// contexts are rejected by d3d12_swiglu_profile_accepts BEFORE creation, so
// the capability never changes under a live context.
void d3d12_finalize_swiglu_mode(bool mtp_active) {
    if (g_swiglu_mtp_diag) {
        // Diagnostic override: bind the raw requested mode even in MTP, so the
        // C4 divergence can be reproduced with fixed-history dumps. Never on
        // in production (bench_swiglu_mtp_diag.txt only).
        if (!g_swiglu_policy.bound) {
            g_swiglu_policy.bound = true;
            g_swiglu_policy.mtp_capable = mtp_active;
            g_swiglu_policy.mode = g_swiglu_requested_mode;
            log_output("[xllama] d3d12: FFN SWIGLU profile bound: " +
                       std::string(g_swiglu_policy.mode == 0 ? "off" : "on") +
                       " (mtp_capable=" + (mtp_active ? "1" : "0") + ", DIAG override)\n");
        }
        return;
    }
    const bool was_bound = g_swiglu_policy.bound;
    const bool mtp_before = g_swiglu_policy.mtp_capable;
    g_swiglu_policy.finalize(g_swiglu_requested_mode, mtp_active);
    if (!was_bound)
        log_output("[xllama] d3d12: FFN SWIGLU profile bound: " +
                   std::string(g_swiglu_policy.mode == 0 ? "off" : "on") +
                   " (mtp_capable=" + (mtp_active ? "1" : "0") + ")\n");
    else if (g_swiglu_policy.mtp_capable != mtp_before)
        log_output("[xllama] d3d12: WARNING: MTP profile flag changed after binding\n");
}

// True when a context with |mtp_active| may be created under the current
// knob. Called before context creation; a false result means the caller must
// fail the request and ask for a process restart instead of changing the
// capability under an existing context.
bool d3d12_swiglu_profile_accepts(bool mtp_active) {
    if (g_swiglu_mtp_diag)
        return true; // diagnostic override: FFN allowed in MTP for the repro
    return g_swiglu_policy.accepts(g_swiglu_requested_mode, mtp_active);
}

void d3d12_set_swiglu_mtp_diag(bool on) {
    g_swiglu_mtp_diag = on;
}

void d3d12_set_swiglu_range_diag(bool on) {
    g_swiglu_range_diag = on;
}

bool d3d12_swiglu_mtp_diag_enabled() {
    return g_swiglu_mtp_diag;
}

int d3d12_swiglu_product_mode() {
    return g_swiglu_policy.mode;
}

void d3d12_set_swiglu_target_layers(int n) {
    g_swiglu_target_layers = n;
}

int d3d12_swiglu_target_layers() {
    return g_swiglu_target_layers;
}

double d3d12_wall_ms() {
    Gpu& g = gpu();
    std::lock_guard<std::mutex> lock(g.mu);
    return g.wall_ms;
}

std::uint64_t d3d12_2col_matmuls() {
    Gpu& g = gpu();
    std::lock_guard<std::mutex> lock(g.mu);
    return g.n_matmuls_2col;
}

bool d3d12_q6_tiled_available(int columns) {
    if (columns != 2 && columns != 4)
        return false;
    Gpu& g = gpu();
    const int i = columns == 4 ? 1 : 0;
    return g.ok && g.pso_q6col[i][0] && g.pso_q6col[i][1];
}
std::uint64_t d3d12_q6_tiled_matmuls() {
    Gpu& g = gpu();
    std::lock_guard<std::mutex> lock(g.mu);
    return g.n_q6_tiled[0] + g.n_q6_tiled[1];
}

// Kernel variant switch (plan 004). Set outside graph_compute on the bench
// thread, read inside it on the same thread; plain int like the scope tags'
// contract (set/restore around the call, never leaking across cases).
static int g_kernel_variant = 0;

void d3d12_set_kernel_variant(int v) {
    g_kernel_variant = (v >= 0 && v <= 2) ? v : 0;
}

void d3d12_set_z0_dispatch_log(bool on) {
    g_z0_dispatch_log = on;
}

void d3d12_set_z0_pin_cpu(bool on) {
    g_z0_pin_cpu = on;
}

void d3d12_set_island_enabled(bool on) {
    g_island_enabled = on;
}

void d3d12_set_spin_wait_us(int spin_us) {
    g_spin_wait_us = spin_us;
}

int d3d12_spin_wait_us() {
    return g_spin_wait_us;
}

int d3d12_kernel_variant() {
    return g_kernel_variant;
}

bool d3d12_2col_available() {
    Gpu& g = gpu();
    return g.ok && g.pso2col[0] && g.pso2col[1];
}

std::string d3d12_2col_info() {
    Gpu& g = gpu();
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "q4k-2col blobs t64=%zuB t128=%zuB | root 8dw (ncols in dword 6) | groupshared "
                  "t64=1024B t128=2048B (same red array as OLD) | acc 8 f32/thread (OLD 4) | %s",
                  kGgmlD3d12MmvQ4K2ColT64DxilSize, kGgmlD3d12MmvQ4K2ColT128DxilSize,
                  g.twocol_error.empty() ? (d3d12_2col_available() ? "PSOs ready" : "PSOs missing")
                                         : g.twocol_error.c_str());
    return buf;
}

void run_d3d12_silu_selftest(std::vector<D3d12SelftestRow>* out);

void run_d3d12_selftest(std::vector<D3d12SelftestRow>* out) {
    if (!out)
        return;
    if (!ggml_d3d12_register()) {
        D3d12SelftestRow r;
        r.type = "-";
        r.error = gpu().error.empty() ? "d3d12 backend unavailable" : gpu().error;
        out->push_back(std::move(r));
        return;
    }
    ggml_backend_t backend = dev_init_backend(&kDevice, nullptr);
    // Shapes of the D2 target models; q6_k 2048x11008 has 2-byte-aligned rows.
    const SelftestCase cases[] = {
        {GGML_TYPE_Q4_0, "q4_0", 8192, 2048, 1},
        {GGML_TYPE_Q4_0, "q4_0", 2048, 8192, 1},
        {GGML_TYPE_Q4_K, "q4_k", 11008, 2048, 1},
        {GGML_TYPE_Q4_K, "q4_k", 2048, 11008, 1},
        {GGML_TYPE_Q5_K, "q5_k", 11008, 2048, 1},
        {GGML_TYPE_Q5_K, "q5_k", 2048, 11008, 1},
        {GGML_TYPE_Q6_K, "q6_k", 2048, 11008, 1},
        {GGML_TYPE_Q6_K, "q6_k", 65536, 1024, 1},
        {GGML_TYPE_Q4_0, "q4_0", 1024, 1024, 7},
        {GGML_TYPE_Q4_K, "q4_k", 1024, 1024, 7},
        {GGML_TYPE_Q5_K, "q5_k", 1024, 1024, 7},
        {GGML_TYPE_Q6_K, "q6_k", 1024, 1024, 7},
        {GGML_TYPE_Q4_0, "q4_0", 1024, 1024, 512},
        {GGML_TYPE_Q4_K, "q4_k", 1024, 1024, 512},
        {GGML_TYPE_Q5_K, "q5_k", 1024, 1024, 512},
        {GGML_TYPE_Q6_K, "q6_k", 1024, 1024, 512},
        // MTP eh_proj, including the B3/B4 transition and odd-row tails.
        {GGML_TYPE_Q8_0, "q8_0", 2560, 5120, 1},
        {GGML_TYPE_Q8_0, "q8_0", 2560, 5120, 2},
        {GGML_TYPE_Q8_0, "q8_0", 2560, 5120, 3},
        {GGML_TYPE_Q8_0, "q8_0", 2560, 5120, 4},
        {GGML_TYPE_Q8_0, "q8_0", 2560, 5120, 5},
        {GGML_TYPE_Q8_0, "q8_0", 13, 256, 512},
    };
    for (const auto& sc : cases)
        out->push_back(run_case(backend, sc, 1));

    // Plan 003 stage 2: MATMUL MICROBENCHMARK across batch widths, NOT T_target(B).
    // These are isolated matmuls with a fixed seed independent of B; they say
    // nothing about KV attention, the recurrent state or real decode scheduling.
    // The real T_target(B) is a separate bench (tttarget.flag) on a live context.
    const SelftestCase batch_cases[] = {
        {GGML_TYPE_Q4_K, "q4_k", 11008, 2048, 1}, {GGML_TYPE_Q4_K, "q4_k", 11008, 2048, 2},
        {GGML_TYPE_Q4_K, "q4_k", 11008, 2048, 3}, {GGML_TYPE_Q4_K, "q4_k", 11008, 2048, 5},
        {GGML_TYPE_Q6_K, "q6_k", 2048, 11008, 1}, {GGML_TYPE_Q6_K, "q6_k", 2048, 11008, 2},
        {GGML_TYPE_Q6_K, "q6_k", 2048, 11008, 3}, {GGML_TYPE_Q6_K, "q6_k", 2048, 11008, 5},
        // Owner audit: complete the missing real-format width rows (q5_k and
        // the remaining shapes seen in the workload census).
        {GGML_TYPE_Q5_K, "q5_k", 11008, 2048, 1}, {GGML_TYPE_Q5_K, "q5_k", 11008, 2048, 2},
        {GGML_TYPE_Q5_K, "q5_k", 11008, 2048, 3}, {GGML_TYPE_Q5_K, "q5_k", 11008, 2048, 5},
        {GGML_TYPE_Q5_K, "q5_k", 2048, 11008, 1}, {GGML_TYPE_Q5_K, "q5_k", 2048, 11008, 2},
        {GGML_TYPE_Q5_K, "q5_k", 2048, 11008, 3}, {GGML_TYPE_Q5_K, "q5_k", 2048, 11008, 5},
        {GGML_TYPE_Q4_K, "q4_k", 2048, 11008, 1}, {GGML_TYPE_Q4_K, "q4_k", 2048, 11008, 2},
        {GGML_TYPE_Q4_K, "q4_k", 2048, 11008, 3}, {GGML_TYPE_Q4_K, "q4_k", 2048, 11008, 5},
        {GGML_TYPE_Q6_K, "q6_k", 65536, 1024, 1}, {GGML_TYPE_Q6_K, "q6_k", 65536, 1024, 2},
        {GGML_TYPE_Q6_K, "q6_k", 65536, 1024, 3}, {GGML_TYPE_Q6_K, "q6_k", 65536, 1024, 5},
    };
    for (const auto& sc : batch_cases)
        out->push_back(run_case(backend, sc, 1));
    ggml_backend_free(backend);
    run_d3d12_silu_selftest(out);
}

void run_d3d12_shape_cost(std::vector<D3d12SelftestRow>* out) {
    if (!out)
        return;
    run_d3d12_gdn_selftest(out);
    if (!ggml_d3d12_register()) {
        D3d12SelftestRow r;
        r.type = "-";
        r.error = gpu().error.empty() ? "d3d12 backend unavailable" : gpu().error;
        out->push_back(std::move(r));
        return;
    }
    // The 11 (t, N, K) triples the target saw under label=tttarget in the real
    // run (plan 004): every shape the batch widths must be costed against. The
    // 12th histogram triple (q6_k 8192x2560) appeared only in
    // session_prefill/session_decode, never under tttarget, so it stays out.
    // Sorted by weight bytes, smallest first and the 248320x2560 Q6_K lm_head
    // last, so a kill at any point leaves the cheap shapes already measured.
    // Ordering is by weight bytes computed from ggml_row_size, not by a
    // hand-sorted table: the order claim is checkable, not asserted.
    struct ShapeCostShape {
        ggml_type type;
        const char* name;
        int n, k;
    };
    const ShapeCostShape shapes[] = {
        {GGML_TYPE_Q4_K, "q4_k", 1024, 2560},   {GGML_TYPE_Q4_K, "q4_k", 2560, 4096},
        {GGML_TYPE_Q4_K, "q4_k", 2560, 9216},   {GGML_TYPE_Q4_K, "q4_k", 4096, 2560},
        {GGML_TYPE_Q4_K, "q4_k", 8192, 2560},   {GGML_TYPE_Q4_K, "q4_k", 9216, 2560},
        {GGML_TYPE_Q5_K, "q5_k", 2560, 4096},   {GGML_TYPE_Q5_K, "q5_k", 8192, 2560},
        {GGML_TYPE_Q6_K, "q6_k", 1024, 2560},   {GGML_TYPE_Q6_K, "q6_k", 2560, 9216},
        {GGML_TYPE_Q6_K, "q6_k", 248320, 2560},
    };
    const int widths[] = {1, 2, 3, 5};
    std::vector<SelftestCase> cases;
    for (const auto& s : shapes)
        for (int w : widths)
            cases.push_back({s.type, s.name, s.n, s.k, w});
    std::stable_sort(cases.begin(), cases.end(), [](const SelftestCase& a, const SelftestCase& b) {
        return static_cast<std::int64_t>(a.n) * ggml_row_size(a.type, a.k) <
               static_cast<std::int64_t>(b.n) * ggml_row_size(b.type, b.k);
    });
    ggml_backend_t backend = dev_init_backend(&kDevice, nullptr);
    // Synthetic edge cases first (NOT histogram triples, documented as such):
    // an odd row count (exercises the row-clamp path) and a single-row group,
    // B=1 single-variant plus B=2 paired. Microsecond cheap, and they prove
    // the tail guards before the real shapes run.
    const SelftestCase edges[] = {
        {GGML_TYPE_Q4_K, "q4_k", 1023, 2560, 1},
        {GGML_TYPE_Q4_K, "q4_k", 1023, 2560, 2},
        {GGML_TYPE_Q4_K, "q4_k", 1, 256, 1},
        {GGML_TYPE_Q4_K, "q4_k", 1, 256, 2},
    };
    int case_index = 0;
    for (const auto& sc : edges) {
        if (sc.ncols >= 2)
            run_paired_case(backend, sc.type, sc.name, sc.n, sc.k, sc.ncols, kD3d12ShapeCostPairs,
                            case_index++, out);
        else
            out->push_back(run_case(backend, sc, kD3d12ShapeCostRepeats));
    }
    // Padded-stride gate (plan 004, NOT histogram triples): real strided X/Y
    // rows and weight rows with sentinel padding and odd tails, both variants
    // checked for untouched padding and numerical match. Odd pads catch
    // alignment assumptions; the 9216-row case keeps a B=5 column tail.
    struct PaddedCase {
        ggml_type type;
        const char* name;
        int n, k, ncols, x_pad, y_pad;
        std::size_t w_pad;
    };
    const PaddedCase padded_cases[] = {
        {GGML_TYPE_Q4_K, "q4_k", 2560, 4096, 3, 13, 7, 64},
        {GGML_TYPE_Q4_K, "q4_k", 9216, 2560, 5, 1, 3, 48},
    };
    for (const auto& pc : padded_cases)
        run_paired_case(backend, pc.type, pc.name, pc.n, pc.k, pc.ncols, kD3d12ShapeCostPairs,
                        case_index++, out, pc.x_pad, pc.y_pad, pc.w_pad);
    for (const auto& sc : cases) {
        if (sc.type == GGML_TYPE_Q4_K && sc.ncols >= 2)
            run_paired_case(backend, sc.type, sc.name, sc.n, sc.k, sc.ncols, kD3d12ShapeCostPairs,
                            case_index++, out);
        else
            out->push_back(run_case(backend, sc, kD3d12ShapeCostRepeats));
    }
    // Real measured LM-head shape: same W/X for OLD vs each tile, alternated
    // pairs. Inputs, outputs, pads and the exact OLD column sums are checked.
    for (int columns : {2, 4}) {
        const std::string label = "q6_k_c" + std::to_string(columns);
        for (int width : {2, 3, 5})
            run_paired_case(backend, GGML_TYPE_Q6_K, label.c_str(), 248320, 2560, width,
                            kD3d12ShapeCostPairs, case_index++, out, 0, 0, 0, columns);
        run_paired_case(backend, GGML_TYPE_Q6_K, label.c_str(), 248320, 2560, 5,
                        kD3d12ShapeCostPairs, case_index++, out, 13, 7, 64, columns);
    }
    ggml_backend_free(backend);
}

void run_d3d12_gdn_selftest(std::vector<D3d12SelftestRow>* out) {
    if (!out)
        return;
    if (!ggml_d3d12_register() || !gpu().gdn_pso) {
        D3d12SelftestRow row;
        row.type = "gdn";
        row.error = "D3D12 GDN unavailable: " + gpu().gdn_error;
        out->push_back(std::move(row));
        return;
    }
    ggml_backend_t backend = dev_init_backend(&kDevice, nullptr);
    run_gdn_cases(backend, &kHostBuft, out);
    ggml_backend_free(backend);
}

// Probe metrics (plan 006 C4): ordinary finite-domain and extreme/subnormal
// classes are reported separately, with max ULP / max abs / max relative and
// an ULP histogram, so an edge-dominated max ULP cannot hide the ordinary
// behavior. Non-finite mismatches (NaN-ness, inf-ness, inf sign) are counted
// separately. Pure function over three vectors (in1 null for unary ops).
struct ProbeMetrics {
    bool ok = false;
    std::string report;
};
ProbeMetrics probe_metrics(const std::vector<float>& got, const std::vector<float>& want,
                           const std::vector<float>& in0, const std::vector<float>* in1) {
    auto key = [](float v) {
        std::uint32_t u = 0;
        std::memcpy(&u, &v, sizeof(u));
        return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    };
    const float sub_min = std::numeric_limits<float>::min();
    size_t fin = 0, edge = 0, nf = 0, ulp_max = 0;
    size_t ulp_max_ord = 0;
    size_t hist[6] = {0, 0, 0, 0, 0, 0}; // 0,1,2,3,4-7,8+
    double maxabs = 0.0, maxrel = 0.0;
    double maxabs_ord = 0.0, maxrel_ord = 0.0;
    std::string first;
    for (size_t i = 0; i < got.size(); ++i) {
        const float a = got[i], b = want[i];
        const float in = in0[i];
        const bool fa = std::isfinite(a), fb = std::isfinite(b);
        if (!fa || !fb) {
            const bool pair_ok = (std::isnan(a) == std::isnan(b)) &&
                                 (std::isinf(a) == std::isinf(b)) &&
                                 (!std::isinf(a) || (std::signbit(a) == std::signbit(b)));
            if (!pair_ok)
                ++nf;
            continue;
        }
        const std::uint32_t ka = key(a), kb = key(b);
        if (ka == kb)
            continue;
        const std::uint32_t d = ka > kb ? ka - kb : kb - ka;
        if (d > ulp_max)
            ulp_max = d;
        const int bucket =
            d == 0 ? 0 : (d == 1 ? 1 : (d == 2 ? 2 : (d == 3 ? 3 : (d < 8 ? 4 : 5))));
        const double ad = std::fabs(static_cast<double>(a) - static_cast<double>(b));
        if (ad > maxabs)
            maxabs = ad;
        const double rel =
            std::fabs(static_cast<double>(b)) > 0.0
                ? ad / std::fabs(static_cast<double>(b))
                : 0.0;
        if (rel > maxrel)
            maxrel = rel;
        const bool is_edge = (std::fabs(in) > 30.0f) || (std::fabs(a) < sub_min && a != 0.0f) ||
                             (std::fabs(b) < sub_min && b != 0.0f) || std::fabs(a) > 1e30f ||
                             std::fabs(b) > 1e30f || (in1 && std::fabs((*in1)[i]) > 30.0f);
        if (is_edge)
            ++edge;
        else {
            ++fin;
            ++hist[bucket];
            // Metric fix: the ordered maxima were declared but never updated,
            // so ord_ulp_max/ord_abs/ord_rel always read 0 and looked like a
            // bit-exact ordered region.
            if (d > ulp_max_ord)
                ulp_max_ord = d;
            if (ad > maxabs_ord)
                maxabs_ord = ad;
            if (rel > maxrel_ord)
                maxrel_ord = rel;
        }
        if (first.size() < 160) {
            std::uint32_t ia = 0, ib = 0, ii = 0;
            std::memcpy(&ia, &a, 4);
            std::memcpy(&ib, &b, 4);
            std::memcpy(&ii, &in, 4);
            char t[96];
            std::snprintf(t, sizeof(t), " [i=%llu in=0x%08x got=0x%08x ref=0x%08x]",
                          static_cast<unsigned long long>(i), ii, ia, ib);
            first += t;
        }
    }
    char buf[448];
    std::snprintf(
        buf, sizeof(buf),
        "ord=%llu ord_ulp_max=%llu ord_abs=%.3g ord_rel=%.3g "
        "hist0/1/2/3/4-7/8+=%llu/%llu/%llu/%llu/%llu/%llu edge=%llu nf=%llu "
        "all_ulp_max=%llu all_abs=%.3g all_rel=%.3g",
        static_cast<unsigned long long>(fin), static_cast<unsigned long long>(ulp_max_ord),
        maxabs_ord, maxrel_ord, static_cast<unsigned long long>(hist[0]),
        static_cast<unsigned long long>(hist[1]), static_cast<unsigned long long>(hist[2]),
        static_cast<unsigned long long>(hist[3]), static_cast<unsigned long long>(hist[4]),
        static_cast<unsigned long long>(hist[5]), static_cast<unsigned long long>(edge),
        static_cast<unsigned long long>(nf), static_cast<unsigned long long>(ulp_max), maxabs,
        maxrel);
    ProbeMetrics m;
    // ok = no ordinary mismatch (hist 1..5 all zero), no edge mismatch, no
    // non-finite mismatch. fin counts ALL ordinary elements now.
    m.ok = (hist[1] == 0 && hist[2] == 0 && hist[3] == 0 && hist[4] == 0 && hist[5] == 0) &&
           (edge == 0) && (nf == 0);
    m.report = std::string(buf) + first;
    return m;
}

// Plan 006 C4 selftest: exact SWIGLU/SILU kernel feasibility. CPU reference is
// the same ggml_silu op on the CPU backend (AVX2/FMA path); GPU runs through
// the selftest-gated D3D12 SILU dispatch. Compares per element: bit equality,
// max ULP (finite pairs) and non-finite classification. The row's error string
// carries the counts and the actual MXCSR value, so the reference's FTZ/DAZ
// state is evidence, not an assumption.
// Unfused emulation of the SILU formula: identical coefficients/tree/branches
// as the shader, but every FMA is split into a multiply and an add through
// volatile intermediates. Comparing the GPU result against this reference and
// the fused CPU kernel isolates the fusion semantics of dx.op.tertiary.
float s_unfused_silu(float x) {
    volatile float lo = 0x1.715476p+0f, rr = 0x1.8p23f;
    volatile float c1 = 0x1.7f7d1cp-20f, c2 = 0x1.62e4p-1f;
    volatile float p0 = 0x1.0e4020p-7f, p1 = 0x1.573e2ep-5f, p2 = 0x1.555e66p-3f;
    volatile float p3 = 0x1.fffdb6p-2f, p4 = 0x1.ffffecp-1f;
    const float neg_x = 0.0f - x;
    volatile float z = neg_x * lo;
    z = z + rr;
    volatile float n = z - rr;
    volatile float t = n * c2;
    t = x - t;
    volatile float b = n * c1;
    b = t - b;
    volatile float u = b * b;
    uint32_t e = 0;
    const float zf = z;
    std::memcpy(&e, &zf, sizeof(e));
    e <<= 23;
    float k = 0.0f;
    {
        const uint32_t sm = e + 0x3f800000u;
        std::memcpy(&k, &sm, sizeof(k));
    }
    volatile float t1 = p0 * b;
    t1 = t1 + p1;
    volatile float t2 = p2 * b;
    t2 = t2 + p3;
    volatile float j1 = t1 * u;
    j1 = j1 + t2;
    volatile float t3 = p4 * b;
    volatile float j = j1 * u;
    j = j + t3;
    float ex = 0.0f;
    const float an = std::fabs(n);
    if (an > 126.0f) {
        const uint32_t gg = (n <= 0.0f) ? 0x82000000u : 0u;
        float s1 = 0.0f, s2 = 0.0f;
        {
            const uint32_t b1 = gg + 0x7f000000u;
            std::memcpy(&s1, &b1, sizeof(s1));
        }
        {
            const uint32_t b2 = e - gg;
            std::memcpy(&s2, &b2, sizeof(s2));
        }
        if (an > 192.0f) {
            ex = s1 * s1;
        } else {
            volatile float q = s2 * j;
            q = q + s2;
            ex = q * s1;
        }
    } else {
        volatile float e2 = j * k;
        e2 = e2 + k;
        ex = e2;
    }
    const float e_plus = 1.0f + ex;
    return x / e_plus;
}

D3d12SelftestRow run_silu_case(ggml_backend_t backend, ggml_backend_buffer_type_t buft,
                               ggml_backend_t cpu, int n_embd, int rows,
                               std::vector<float>* result) {
    (void)cpu; // direct kernel reference; kept in the signature for callers
    D3d12SelftestRow row;
    row.type = "silu_avx2";
    row.n = n_embd;
    row.k = rows;
    row.ncols = rows;
    ggml_init_params ip = {};
    ip.mem_size = 16 * ggml_tensor_overhead() + 2 * ggml_graph_overhead();
    ip.no_alloc = true;
    ggml_context* ctx = ggml_init(ip);
    if (!ctx) {
        row.error = "SILU metadata allocation failed";
        return row;
    }
    log_output("[xllama] silu selftest: ctx ready\n");
    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, rows);
    ggml_tensor* test = ggml_silu(ctx, x);
    ggml_cgraph* gt = ggml_new_graph(ctx);
    ggml_build_forward_expand(gt, test);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    auto cleanup = [&] {
        if (buf)
            ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    };
    if (!buf) {
        row.error = "SILU allocation failed";
        cleanup();
        return row;
    }
    // Corpus: deterministic real-range values plus adversarial boundaries
    // (signed zero, subnormals, the polynomial's |n| ~ 126/192 branches,
    // large magnitudes, non-finite).
    std::vector<float> xs(static_cast<size_t>(n_embd) * rows);
    std::mt19937 rng(2026u);
    std::uniform_real_distribution<float> uni(-20.0f, 20.0f);
    const float specials[] = {0.0f,
                              -0.0f,
                              1e-45f,
                              -1e-45f,
                              1e-40f,
                              -1e-40f,
                              87.3f,
                              -87.3f,
                              88.0f,
                              -88.0f,
                              133.0f,
                              -133.0f,
                              134.0f,
                              -134.0f,
                              3.4e38f,
                              -3.4e38f,
                              std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::quiet_NaN()};
    const size_t n_special = sizeof(specials) / sizeof(specials[0]);
    for (size_t i = 0; i < xs.size(); ++i)
        xs[i] = (i % 97 == 0) ? specials[(i / 97) % n_special] : uni(rng);
    log_output("[xllama] silu selftest: buffers allocated, setting inputs\n");
    ggml_backend_tensor_set(x, xs.data(), 0, xs.size() * sizeof(float));
    log_output("[xllama] silu selftest: GPU compute\n");
    const ggml_status st_t = ggml_backend_graph_compute(backend, gt);
    log_output("[xllama] silu selftest: GPU compute rc=" + std::to_string(st_t) + "\n");
    log_output("[xllama] silu selftest: post-GPU reached\n");
    // CPU reference: the exact compiled AVX2/FMA kernel, called directly on
    // this thread. MXCSR is sampled around the call, so the FTZ/DAZ state that
    // produced the reference is evidence, not an assumption.
    std::vector<float> want(xs.size());
    const unsigned csr_before = static_cast<unsigned>(_mm_getcsr());
    log_output("[xllama] silu selftest: CPU direct kernel\n");
    ::ggml_vec_silu_f32(static_cast<int>(xs.size()), want.data(), xs.data());
    const unsigned csr_after = static_cast<unsigned>(_mm_getcsr());
    log_output(
        "[xllama] silu selftest: CPU direct done mxcsr_before=0x" +
        [&] {
            char b[8];
            std::snprintf(b, sizeof(b), "%04x", csr_before);
            return std::string(b);
        }() +
        " after=0x" +
        [&] {
            char b[8];
            std::snprintf(b, sizeof(b), "%04x", csr_after);
            return std::string(b);
        }() +
        "\n");
    if (st_t != GGML_STATUS_SUCCESS) {
        row.error = "SILU GPU graph_compute failed";
        cleanup();
        return row;
    }
    std::vector<float> got(xs.size());
    ggml_backend_tensor_get(test, got.data(), 0, got.size() * sizeof(float));
    auto key = [](float v) {
        std::uint32_t u = 0;
        std::memcpy(&u, &v, sizeof(u));
        return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    };
    // Causal test for the edge mismatches: same formula, every FMA split into
    // mul+add with volatile intermediates (no contraction), so the fused and
    // unfused references differ only in the fusion semantics. Comparing the GPU
    // result against both says whether the Xbox driver lowers dx.op.tertiary
    // (mad()) fused or unfused. Normal path only (|n| <= 126); the edge
    // branches are reported by the fused comparison already.
    std::vector<float> want_unf(xs.size());
    for (size_t i = 0; i < xs.size(); ++i)
        want_unf[i] = s_unfused_silu(xs[i]);
    const ProbeMetrics mu = probe_metrics(got, want_unf, xs, nullptr);
    const ProbeMetrics m = probe_metrics(got, want, xs, nullptr);
    row.rel_err = 0.0;
    row.ok = m.ok;
    row.d3d12_ran = true;
    char note[2048];
    std::snprintf(note, sizeof(note), "mxcsr=0x%04x [fused] ok=%d %s | [unfused] ok=%d %s",
                  static_cast<unsigned>(_mm_getcsr()), m.ok ? 1 : 0, m.report.c_str(),
                  mu.ok ? 1 : 0, mu.report.c_str());
    row.error = note;
    if (result)
        *result = got;
    cleanup();
    return row;
}

// Split-SWIGLU probe: out = silu(gate) * up, gate/up independent corpora.
namespace {
inline uint32_t s_mono_key(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
} // namespace

// Plan 008 selftests. ADD must be bit-exact (one rounded add per element);
// RMS_NORM is tolerance-measured: the GPU uses double lanes + a double tree
// and a precise 1/sqrt, the CPU reference is the exact scalar double ascending
// loop from ggml-cpu, so the row reports the exact-match rate and the max
// relative error/ULP instead of a bitwise promise.
D3d12SelftestRow run_add_case(ggml_backend_t backend, ggml_backend_buffer_type_t buft, int n_embd,
                              int rows) {
    D3d12SelftestRow row;
    row.type = "add_f32";
    row.n = n_embd;
    row.k = rows;
    row.ncols = rows;
    const bool prev_island = g_island_enabled;
    g_island_enabled = true; // exercise the kernel, not the product default
    ggml_init_params ip = {};
    ip.mem_size = 16 * ggml_tensor_overhead() + 2 * ggml_graph_overhead();
    ip.no_alloc = true;
    ggml_context* ctx = ggml_init(ip);
    auto cleanup = [&] {
        g_island_enabled = prev_island;
    };
    if (!ctx) {
        row.error = "add metadata allocation failed";
        cleanup();
        return row;
    }
    ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, rows);
    ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, rows);
    ggml_tensor* t = ggml_add(ctx, a, b);
    ggml_cgraph* gt = ggml_new_graph(ctx);
    ggml_build_forward_expand(gt, t);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) {
        row.error = "add allocation failed";
        ggml_free(ctx);
        cleanup();
        return row;
    }
    const size_t n = static_cast<size_t>(n_embd) * static_cast<size_t>(rows);
    std::vector<float> av(n), bv(n);
    std::mt19937 rng(808u);
    std::uniform_real_distribution<float> uni(-8.0f, 8.0f);
    const float specials[] = {0.0f,
                              -0.0f,
                              1e-45f,
                              -1e-45f,
                              3.4e38f,
                              -3.4e38f,
                              std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::quiet_NaN()};
    const size_t ns = sizeof(specials) / sizeof(specials[0]);
    for (size_t i = 0; i < n; ++i) {
        av[i] = (i % 89 == 0) ? specials[(i / 89) % ns] : uni(rng);
        bv[i] = (i % 97 == 0) ? specials[(i / 97) % ns] : uni(rng);
    }
    ggml_backend_tensor_set(a, av.data(), 0, n * sizeof(float));
    ggml_backend_tensor_set(b, bv.data(), 0, n * sizeof(float));
    const ggml_status st = ggml_backend_graph_compute(backend, gt);
    if (st != GGML_STATUS_SUCCESS) {
        row.error = "add GPU graph_compute failed";
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        cleanup();
        return row;
    }
    std::vector<float> got(n);
    ggml_backend_tensor_get(t, got.data(), 0, n * sizeof(float));
    size_t mismatch = 0, nonfinite_mismatch = 0;
    uint32_t max_ulp = 0;
    double max_rel = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const float ref = av[i] + bv[i];
        if (std::memcmp(&got[i], &ref, sizeof(float)) != 0) {
            ++mismatch;
            if (std::isnan(ref) != std::isnan(got[i]) || std::isinf(ref) != std::isinf(got[i]))
                ++nonfinite_mismatch;
            const uint32_t kg = s_mono_key(got[i]), kr = s_mono_key(ref);
            const uint32_t d = kg > kr ? kg - kr : kr - kg;
            if (d > max_ulp)
                max_ulp = d;
        }
        const double denom = std::max(std::fabs(static_cast<double>(ref)), 1e-30);
        max_rel = std::max(max_rel, std::fabs(static_cast<double>(got[i]) - ref) / denom);
    }
    row.ok = mismatch == 0;
    row.d3d12_ran = true;
    row.rel_err = max_rel;
    char note[192];
    std::snprintf(note, sizeof(note), "bit_mismatch=%zu max_ulp=%u nonfinite_mismatch=%zu n=%zu",
                  mismatch, max_ulp, nonfinite_mismatch, n);
    row.error = note;
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    cleanup();
    return row;
}

D3d12SelftestRow run_rms_norm_case(ggml_backend_t backend, ggml_backend_buffer_type_t buft,
                                   int n_embd, int rows) {
    D3d12SelftestRow row;
    row.type = "rms_norm_f32";
    row.n = n_embd;
    row.k = rows;
    row.ncols = rows;
    const float eps = 1e-5f;
    const bool prev_island = g_island_enabled;
    g_island_enabled = true;
    ggml_init_params ip = {};
    ip.mem_size = 16 * ggml_tensor_overhead() + 2 * ggml_graph_overhead();
    ip.no_alloc = true;
    ggml_context* ctx = ggml_init(ip);
    auto cleanup = [&] {
        g_island_enabled = prev_island;
    };
    if (!ctx) {
        row.error = "rms metadata allocation failed";
        cleanup();
        return row;
    }
    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, rows);
    ggml_tensor* t = ggml_rms_norm(ctx, x, eps);
    ggml_cgraph* gt = ggml_new_graph(ctx);
    ggml_build_forward_expand(gt, t);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) {
        row.error = "rms allocation failed";
        ggml_free(ctx);
        cleanup();
        return row;
    }
    const size_t n = static_cast<size_t>(n_embd) * static_cast<size_t>(rows);
    std::vector<float> xv(n);
    std::mt19937 rng(909u);
    std::uniform_real_distribution<float> uni(-4.0f, 4.0f);
    for (size_t i = 0; i < n; ++i)
        xv[i] = uni(rng);
    // Row 0: all zeros (scale = 1/sqrt(eps)); row 1: all ones; last row: NaN.
    for (int j = 0; j < n_embd; ++j)
        xv[j] = 0.0f;
    for (int j = 0; j < n_embd; ++j)
        xv[n_embd + j] = 1.0f;
    for (int j = 0; j < n_embd; ++j)
        xv[n - n_embd + j] = (j % 7 == 0) ? std::numeric_limits<float>::quiet_NaN() : xv[n - n_embd + j];
    ggml_backend_tensor_set(x, xv.data(), 0, n * sizeof(float));
    const ggml_status st = ggml_backend_graph_compute(backend, gt);
    if (st != GGML_STATUS_SUCCESS) {
        row.error = "rms GPU graph_compute failed";
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        cleanup();
        return row;
    }
    std::vector<float> got(n);
    ggml_backend_tensor_get(t, got.data(), 0, n * sizeof(float));
    size_t exact = 0, mismatch = 0, nonfinite_mismatch = 0;
    uint32_t max_ulp = 0;
    double max_rel = 0.0;
    for (int r = 0; r < rows; ++r) {
        const float* xr = xv.data() + static_cast<size_t>(r) * n_embd;
        const float* gr = got.data() + static_cast<size_t>(r) * n_embd;
        double sum = 0.0;
        for (int j = 0; j < n_embd; ++j)
            sum += static_cast<double>(xr[j]) * static_cast<double>(xr[j]);
        const float mean = static_cast<float>(sum / static_cast<double>(n_embd));
        const float scale = 1.0f / std::sqrt(mean + eps);
        for (int j = 0; j < n_embd; ++j) {
            const float ref = xr[j] * scale;
            const size_t idx = static_cast<size_t>(r) * n_embd + j;
            if (std::memcmp(&got[idx], &ref, sizeof(float)) == 0) {
                ++exact;
            } else {
                ++mismatch;
                if (std::isnan(ref) != std::isnan(got[idx]) || std::isinf(ref) != std::isinf(got[idx]))
                    ++nonfinite_mismatch;
                const uint32_t kg = s_mono_key(got[idx]), kr = s_mono_key(ref);
                const uint32_t d = kg > kr ? kg - kr : kr - kg;
                if (d > max_ulp)
                    max_ulp = d;
            }
            const double denom = std::max(std::fabs(static_cast<double>(ref)), 1e-30);
            max_rel = std::max(max_rel, std::fabs(static_cast<double>(got[idx]) - ref) / denom);
        }
    }
    row.ok = max_rel <= 1e-6 && nonfinite_mismatch == 0;
    row.d3d12_ran = true;
    row.rel_err = max_rel;
    char note[224];
    std::snprintf(note, sizeof(note),
                  "exact=%zu/%zu mismatch=%zu max_rel=%.3g max_ulp=%u nonfinite_mismatch=%zu",
                  exact, n, mismatch, max_rel, max_ulp, nonfinite_mismatch);
    row.error = note;
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    cleanup();
    return row;
}

D3d12SelftestRow run_swiglu_case(ggml_backend_t backend, ggml_backend_buffer_type_t buft,
                                 int n_embd, int rows) {
    D3d12SelftestRow row;
    row.type = "swiglu_avx2";
    row.n = n_embd;
    row.k = rows;
    row.ncols = rows;
    ggml_init_params ip = {};
    ip.mem_size = 16 * ggml_tensor_overhead() + 2 * ggml_graph_overhead();
    ip.no_alloc = true;
    ggml_context* ctx = ggml_init(ip);
    if (!ctx) {
        row.error = "SWIGLU metadata allocation failed";
        return row;
    }
    ggml_tensor* gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, rows);
    ggml_tensor* up = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, rows);
    ggml_tensor* test = ggml_swiglu_split(ctx, gate, up);
    ggml_cgraph* gt = ggml_new_graph(ctx);
    ggml_build_forward_expand(gt, test);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    auto cleanup = [&] {
        if (buf)
            ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    };
    if (!buf) {
        row.error = "SWIGLU allocation failed";
        cleanup();
        return row;
    }
    const size_t n = static_cast<size_t>(n_embd) * rows;
    std::vector<float> g(n), u(n);
    std::mt19937 rng(2027u);
    std::uniform_real_distribution<float> uni(-20.0f, 20.0f);
    std::uniform_real_distribution<float> upu(-2.0f, 2.0f);
    const float specials[] = {0.0f,
                              -0.0f,
                              1e-45f,
                              -1e-45f,
                              1e-40f,
                              -1e-40f,
                              87.3f,
                              -87.3f,
                              88.0f,
                              -88.0f,
                              133.0f,
                              -133.0f,
                              134.0f,
                              -134.0f,
                              3.4e38f,
                              -3.4e38f,
                              std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::quiet_NaN()};
    const size_t ns = sizeof(specials) / sizeof(specials[0]);
    for (size_t i = 0; i < n; ++i) {
        g[i] = (i % 97 == 0) ? specials[(i / 97) % ns] : uni(rng);
        u[i] = (i % 89 == 0) ? specials[(i / 89) % ns] : upu(rng);
    }
    ggml_backend_tensor_set(gate, g.data(), 0, n * sizeof(float));
    ggml_backend_tensor_set(up, u.data(), 0, n * sizeof(float));
    log_output("[xllama] swiglu selftest: GPU compute\n");
    const ggml_status st = ggml_backend_graph_compute(backend, gt);
    log_output("[xllama] swiglu selftest: GPU compute rc=" + std::to_string(st) + "\n");
    if (st != GGML_STATUS_SUCCESS) {
        row.error = "SWIGLU GPU graph_compute failed";
        cleanup();
        return row;
    }
    std::vector<float> got(n), want(n);
    ggml_backend_tensor_get(test, got.data(), 0, n * sizeof(float));
    const unsigned csr = static_cast<unsigned>(_mm_getcsr());
    log_output("[xllama] swiglu selftest: CPU direct kernel\n");
    ::ggml_vec_swiglu_f32(static_cast<int>(n), want.data(), g.data(), u.data());
    std::vector<float> want_unf(n);
    for (size_t i = 0; i < n; ++i)
        want_unf[i] = s_unfused_silu(g[i]) * u[i];
    const ProbeMetrics mu = probe_metrics(got, want_unf, g, &u);
    const ProbeMetrics m = probe_metrics(got, want, g, &u);
    row.ok = m.ok;
    row.d3d12_ran = true;
    char note[2048];
    std::snprintf(note, sizeof(note), "mxcsr=0x%04x [fused] ok=%d %s | [unfused] ok=%d %s", csr,
                  m.ok ? 1 : 0, m.report.c_str(), mu.ok ? 1 : 0, mu.report.c_str());
    row.error = note;
    cleanup();
    return row;
}

void run_d3d12_silu_selftest(std::vector<D3d12SelftestRow>* out) {
    if (!out)
        return;
    if (!ggml_d3d12_register() || !gpu().silu_pso) {
        D3d12SelftestRow row;
        row.type = "silu_avx2";
        row.error = "D3D12 SILU selftest PSO unavailable";
        out->push_back(std::move(row));
        return;
    }
    ggml_backend_t backend = dev_init_backend(&kDevice, nullptr);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!backend || !cpu) {
        D3d12SelftestRow row;
        row.type = "silu_avx2";
        row.error = "SILU backend/CPU init failed";
        out->push_back(std::move(row));
        if (backend)
            ggml_backend_free(backend);
        if (cpu)
            ggml_backend_free(cpu);
        return;
    }
    // Single-threaded reference with an explicit pool, mirroring the GDN
    // selftest: the CPU graph then executes on one worker, and the caller's
    // MXCSR sample is recorded as a separate observation.
    ggml_threadpool_params tp = ggml_threadpool_params_default(1);
    ggml_threadpool_t pool = ggml_threadpool_new(&tp);
    if (!pool) {
        D3d12SelftestRow row;
        row.type = "silu_avx2";
        row.error = "SILU CPU threadpool unavailable";
        out->push_back(std::move(row));
        ggml_backend_free(backend);
        ggml_backend_free(cpu);
        return;
    }
    ggml_backend_cpu_set_n_threads(cpu, 1);
    ggml_backend_cpu_set_threadpool(cpu, pool);
    log_output("[xllama] silu selftest: backends ready, flag on\n");
    d3d12_set_silu_test_enabled(true);
    out->push_back(run_silu_case(backend, &kHostBuft, cpu, 9216, 4, nullptr));
    out->push_back(run_swiglu_case(backend, &kHostBuft, 9216, 4));
    out->push_back(run_add_case(backend, &kHostBuft, 9216, 4));
    out->push_back(run_rms_norm_case(backend, &kHostBuft, 9216, 4));
    d3d12_set_silu_test_enabled(false);
    log_output("[xllama] silu selftest: case done\n");
    ggml_backend_free(backend);
    ggml_backend_free(cpu);
    ggml_threadpool_free(pool);
}

} // namespace xllama

    #endif // _WIN32

#endif // XLLAMA_USE_LLAMA
