#include "ggml.h"
#include "ggml-saw-int4.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "ggml-turboquant-nc.h"
#include <cuda_runtime_api.h>
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

// Only the absence of a usable CUDA driver/device is a skip. Unexpected
// runtime errors and failures after a device is detected remain test errors.
static constexpr int cuda_test_availability(cudaError_t status, int count) {
    return status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ? 77 :
           status != cudaSuccess || count < 0 ? 1 : count == 0 ? 77 : 0;
}
static_assert(cuda_test_availability(cudaSuccess, 1) == 0 &&
              cuda_test_availability(cudaSuccess, 0) == 77 &&
              cuda_test_availability(cudaErrorNoDevice, 0) == 77 &&
              cuda_test_availability(cudaErrorInsufficientDriver, 0) == 77 &&
              cuda_test_availability(cudaErrorUnknown, 0) == 1 &&
              cuda_test_availability(cudaSuccess, -1) == 1, "CUDA test availability policy");

static int failures = 0;
#define CHECK(x) do { if (!(x)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failures; } } while (0)

// Admission checks must reject malformed views before a kernel sees their pointers.
static void test_view_admission(ggml_backend_t backend) {
    constexpr int d = 256;
    ggml_init_params params = { 8*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    for (ggml_type type : { GGML_TYPE_TQNC_K, GGML_TYPE_TQNC_V, GGML_TYPE_SAW_INT4 }) {
        for (ggml_type idx_type : { GGML_TYPE_I32, GGML_TYPE_I64 }) {
            ggml_tensor * base = ggml_new_tensor_2d(ctx, type, d, 8);
            ggml_tensor * sb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, 4);
            ggml_tensor * ib = ggml_new_tensor_1d(ctx, idx_type, 8);
            const size_t row = base->nb[1], ir = ggml_type_size(idx_type);
            ggml_tensor * dst = ggml_view_2d(ctx, base, d, 2, row, row);
            ggml_tensor * src = ggml_view_2d(ctx, sb, d, 2, sb->nb[1], sizeof(float));
            ggml_tensor * ids = ggml_view_1d(ctx, ib, 2, ir);
            const auto supported = [&](ggml_tensor * a, ggml_tensor * b, ggml_tensor * c) {
                auto * op = ggml_set_rows(ctx, a, b, c);
                if (type == GGML_TYPE_SAW_INT4) op->op_params[GGML_SET_ROWS_OP_PARAM_SAW_CHECKED] = 1;
                return ggml_backend_supports_op(backend, op);
            };
            CHECK(supported(dst, src, ids));
            ggml_tensor * padded = ggml_view_2d(ctx, base, d, 2, row*3, row);
            CHECK(supported(padded, src, ids));
            ggml_tensor * outer = ggml_view_2d(ctx, base, d, 6, row, row);
            ggml_tensor * nested = ggml_view_2d(ctx, outer, d, 2, row, row);
            CHECK(nested->view_offs == 2*row);
            CHECK(supported(nested, src, ids));
            CHECK(!supported(ggml_view_2d(ctx, base, d, 2, row, 1), src, ids));
            CHECK(!supported(ggml_view_2d(ctx, base, d, 2, row, 2), src, ids));
            CHECK(!supported(ggml_view_2d(ctx, base, d, 2, row*8, 0), src, ids));
            CHECK(!supported(dst, ggml_view_2d(ctx, sb, d, 2, sb->nb[1], 1), ids));
            CHECK(!supported(dst, ggml_view_2d(ctx, sb, d, 2, sb->nb[1], 2), ids));
            CHECK(!supported(dst, ggml_view_2d(ctx, sb, d, 2, sb->nb[1]*4, 0), ids));
            CHECK(!supported(dst, src, ggml_view_1d(ctx, ib, 2, 1)));
            CHECK(!supported(dst, src, ggml_view_1d(ctx, ib, 2, 2)));
            ggml_tensor * ids_oob = ggml_view_1d(ctx, ib, 2, 0);
            ids_oob->nb[0] = ir*8;
            CHECK(!supported(dst, src, ids_oob));
            ggml_tensor * mismatched = ggml_set_rows(ctx, dst, src, ids);
            if (type == GGML_TYPE_SAW_INT4) mismatched->op_params[GGML_SET_ROWS_OP_PARAM_SAW_CHECKED] = 1;
            --ids->ne[0];
            CHECK(!ggml_backend_supports_op(backend, mismatched));
            ++ids->ne[0];
            if (type != GGML_TYPE_SAW_INT4) {
                ggml_tensor * other = ggml_new_tensor_2d(ctx, type, d, 8);
                ggml_tensor * copy_dst = ggml_view_2d(ctx, other, d, 2, row, row);
                CHECK(ggml_backend_supports_op(backend, ggml_cpy(ctx, dst, copy_dst)));
                ggml_tensor * copy_bad = ggml_view_2d(ctx, base, d, 2, row, 2);
                CHECK(!ggml_backend_supports_op(backend, ggml_cpy(ctx, copy_bad, copy_dst)));
            }
            void * saved = src->data;
            src->data = reinterpret_cast<void *>(uintptr_t(0x1001));
            CHECK(!supported(dst, src, ids));
            src->data = saved;
        }
    }

    constexpr int n = 256;
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, 1, 6, 1);
    ggml_tensor * kb = ggml_new_tensor_4d(ctx, GGML_TYPE_TQNC_K, d, n + 4, 1, 1);
    ggml_tensor * vb = ggml_new_tensor_4d(ctx, GGML_TYPE_TQNC_V, d, n + 4, 1, 1);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n, 1);
    const auto view = [&](ggml_tensor * base, int rows, size_t stride, size_t offset) {
        return ggml_view_4d(ctx, base, d, rows, 1, 1, stride, base->nb[2], base->nb[3], offset);
    };
    ggml_tensor * k = view(kb, n, kb->nb[1], sizeof(ggml_tqnc_key));
    ggml_tensor * v = view(vb, n, vb->nb[1], sizeof(ggml_tqnc_value));
    const auto supported_fa = [&](ggml_tensor * a, ggml_tensor * b, ggml_tensor * m) {
        return ggml_backend_supports_op(backend, ggml_flash_attn_ext(ctx, q, a, b, m, 1.0f/16.0f, 0, 0));
    };
    CHECK(supported_fa(k, v, mask));
    CHECK(!supported_fa(view(kb, n, kb->nb[1], 1), v, mask));
    CHECK(!supported_fa(view(kb, n, kb->nb[1], 2), v, mask));
    CHECK(!supported_fa(k, view(vb, n, vb->nb[1], 1), mask));
    CHECK(!supported_fa(k, view(vb, n, vb->nb[1], 2), mask));
    CHECK(!supported_fa(view(kb, n, kb->nb[1]*2, 0), v, mask));
    CHECK(!supported_fa(k, view(vb, n, vb->nb[1]*2, 0), mask));
    CHECK(!supported_fa(k, view(vb, n - 1, vb->nb[1], 0), mask));
    CHECK(!supported_fa(k, v, ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n - 1, 1)));
    ggml_tensor * vh = ggml_new_tensor_4d(ctx, GGML_TYPE_TQNC_V, d, n, 2, 1);
    CHECK(!supported_fa(k, vh, mask));
    ggml_tensor * overflow = view(kb, n, kb->nb[1], 0);
    overflow->nb[1] = (SIZE_MAX/kb->nb[1])*kb->nb[1];
    CHECK(!supported_fa(overflow, v, mask));
    ggml_free(ctx);
}

// A later SET_ROWS failure must not publish any row of that operation. The
// preceding operation is intentionally allowed to finish; model-level state
// rollback after a multi-layer failure is a separate integration test.
static void test_set_rows_failure(ggml_backend_t backend, ggml_type idx_type, bool view) {
    constexpr int d = 256, rows = 4, batch = 2;
    std::printf("SET_ROWS transaction idx=%s view=%d\n", ggml_type_name(idx_type), int(view));
    ggml_init_params params = { 8*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    const int step = view ? 2 : 1;
    ggml_tensor * kbase = ggml_new_tensor_2d(ctx, GGML_TYPE_TQNC_K, d, rows*step);
    ggml_tensor * vbase = ggml_new_tensor_2d(ctx, GGML_TYPE_TQNC_V, d, rows*step);
    ggml_tensor * ksbase = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, batch*step);
    ggml_tensor * vsbase = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, batch*step);
    ggml_tensor * kdst = view ? ggml_view_2d(ctx, kbase, d, rows, step*kbase->nb[1], 0) : kbase;
    ggml_tensor * vdst = view ? ggml_view_2d(ctx, vbase, d, rows, step*vbase->nb[1], 0) : vbase;
    ggml_tensor * ksrc = view ? ggml_view_2d(ctx, ksbase, d, batch, step*ksbase->nb[1], 0) : ksbase;
    ggml_tensor * vsrc = view ? ggml_view_2d(ctx, vsbase, d, batch, step*vsbase->nb[1], 0) : vsbase;
    ggml_tensor * ki = ggml_new_tensor_1d(ctx, idx_type, batch);
    ggml_tensor * vi = ggml_new_tensor_1d(ctx, idx_type, batch);
    ggml_tensor * kw = ggml_set_rows(ctx, kdst, ksrc, ki);
    ggml_tensor * vw = ggml_set_rows(ctx, vdst, vsrc, vi);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    graph->uid = 1001; // exercise graph compatibility instead of the zero-UID direct fallback
    ggml_build_forward_expand(graph, kw);
    ggml_build_forward_expand(graph, vw);
    CHECK(ggml_backend_supports_op(backend, kw));
    CHECK(ggml_backend_supports_op(backend, vw));
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx, backend));
    CHECK(buffer != nullptr);
    if (!buffer) { ggml_free(ctx); return; }

    std::vector<float> k(d*batch*step, 0.25f), v(d*batch*step, -0.5f);
    std::vector<uint8_t> initial_k(rows*step*sizeof(ggml_tqnc_key), 0x5a);
    std::vector<uint8_t> initial_v(rows*step*sizeof(ggml_tqnc_value), 0xa5);
    std::vector<uint8_t> after_k(initial_k.size()), after_v(initial_v.size());
    int64_t ids[batch] = { 0, 1 };
    const auto set_ids = [&](ggml_tensor * tensor) {
        if (idx_type == GGML_TYPE_I32) {
            int32_t compact[batch] = { int32_t(ids[0]), int32_t(ids[1]) };
            ggml_backend_tensor_set(tensor, compact, 0, sizeof(compact));
        } else {
            ggml_backend_tensor_set(tensor, ids, 0, sizeof(ids));
        }
    };
    ggml_backend_tensor_set(kbase, initial_k.data(), 0, initial_k.size());
    ggml_backend_tensor_set(vbase, initial_v.data(), 0, initial_v.size());
    ggml_backend_tensor_set(ksbase, k.data(), 0, k.size()*sizeof(float));
    ggml_backend_tensor_set(vsbase, v.data(), 0, v.size()*sizeof(float));
    set_ids(ki);
    set_ids(vi);

    // Exercise the graph repeatedly before injection; NC must remain on its
    // synchronized direct path even if the backend normally captures graphs.
    for (int i = 0; i < 8; ++i) CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(vbase, initial_v.data(), 0, initial_v.size());
    std::fill(k.begin(), k.end(), 0.75f);
    v[step*d + 7] = std::numeric_limits<float>::quiet_NaN();
    ggml_backend_tensor_set(ksbase, k.data(), 0, k.size()*sizeof(float));
    ggml_backend_tensor_set(vsbase, v.data(), 0, v.size()*sizeof(float));
    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);
    ggml_backend_tensor_get(kbase, after_k.data(), 0, after_k.size());
    ggml_backend_tensor_get(vbase, after_v.data(), 0, after_v.size());
    CHECK(after_v == initial_v);
    ggml_tqnc_key expected_key;
    CHECK(ggml_tqnc_quantize_key(k.data(), &expected_key));
    CHECK(std::memcmp(after_k.data(), &expected_key, sizeof(expected_key)) == 0);

    // A bad destination index must also leave every V row intact.
    v[step*d + 7] = -0.5f;
    ids[1] = rows;
    ggml_backend_tensor_set(vsbase, v.data(), 0, v.size()*sizeof(float));
    set_ids(vi);
    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);
    ggml_backend_tensor_get(vbase, after_v.data(), 0, after_v.size());
    CHECK(after_v == initial_v);

    // The same backend and graph must still accept a valid operation.
    ids[1] = 1;
    set_ids(vi);
    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(vbase, after_v.data(), 0, after_v.size());
    ggml_tqnc_value expected_value;
    CHECK(ggml_tqnc_quantize_value(v.data(), &expected_value));
    CHECK(std::memcmp(after_v.data(), &expected_value, sizeof(expected_value)) == 0);
    ggml_free(ctx);
}

// A shared index vector is broadcast to independent higher-dimensional streams.
// Duplicate physical destinations fail atomically; the same row number in
// different streams must remain valid, and negative indices must be skipped.
static void test_multistream_rows(ggml_backend_t backend, ggml_type type, ggml_type idx_type) {
    constexpr int d = 512, rows = 4, batch = 4, n2 = 2, n3 = 3;
    std::printf("SET_ROWS broadcast type=%s idx=%s\n", ggml_type_name(type), ggml_type_name(idx_type));
    ggml_init_params params = { 8*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * dst = ggml_new_tensor_4d(ctx, type, d, rows, n2, n3);
    ggml_tensor * src = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, batch, n2, n3);
    ggml_tensor * idx = ggml_new_tensor_3d(ctx, idx_type, batch, 1, 1);
    ggml_tensor * write = ggml_set_rows(ctx, dst, src, idx);
    if (type == GGML_TYPE_SAW_INT4) write->op_params[GGML_SET_ROWS_OP_PARAM_SAW_CHECKED] = 1;
    ggml_cgraph * graph = ggml_new_graph(ctx);
    graph->uid = 1003 + (type == GGML_TYPE_TQNC_V ? 2 : 0) + (idx_type == GGML_TYPE_I64 ? 1 : 0);
    ggml_build_forward_expand(graph, write);
    CHECK(ggml_backend_supports_op(backend, write));
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx, backend));
    CHECK(buffer != nullptr);
    if (!buffer) { ggml_free(ctx); return; }

    if (type == GGML_TYPE_SAW_INT4) {
        write->op_params[GGML_SET_ROWS_OP_PARAM_SAW_CHECKED] = 2;
        CHECK(!ggml_backend_supports_op(backend, write));
        CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);
        write->op_params[GGML_SET_ROWS_OP_PARAM_SAW_CHECKED] = 1;
    }

    const size_t block_bytes = ggml_type_size(type);
    const size_t row_bytes = 2*block_bytes;
    std::vector<float> input(d*batch*n2*n3);
    for (int z = 0; z < n3; ++z) for (int y = 0; y < n2; ++y) for (int r = 0; r < batch; ++r) {
        const size_t start = size_t((z*n2 + y)*batch + r)*d;
        std::fill(input.begin() + start, input.begin() + start + d, float(1 + 10*z + 3*y + r));
    }
    std::vector<uint8_t> initial(rows*n2*n3*row_bytes, 0xa5), actual(initial.size()), expected(initial);
    int64_t ids[batch] = { 0, 1, 1, -1 };
    const auto set_ids = [&]() {
        if (idx_type == GGML_TYPE_I32) {
            int32_t compact[batch];
            for (int i = 0; i < batch; ++i) compact[i] = int32_t(ids[i]);
            ggml_backend_tensor_set(idx, compact, 0, sizeof(compact));
        } else {
            ggml_backend_tensor_set(idx, ids, 0, sizeof(ids));
        }
    };
    ggml_backend_tensor_set(dst, initial.data(), 0, initial.size());
    ggml_backend_tensor_set(src, input.data(), 0, input.size()*sizeof(float));
    set_ids();
    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);
    ggml_backend_tensor_get(dst, actual.data(), 0, actual.size());
    CHECK(actual == initial);

    ids[2] = 2;
    set_ids();
    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(dst, actual.data(), 0, actual.size());
    for (int z = 0; z < n3; ++z) for (int y = 0; y < n2; ++y) for (int r = 0; r < 3; ++r) {
        const size_t input_offset = size_t((z*n2 + y)*batch + r)*d;
        const size_t output_offset = size_t((z*n2 + y)*rows + r)*row_bytes;
        for (int b = 0; b < 2; ++b) {
            if (type == GGML_TYPE_TQNC_K) {
                ggml_tqnc_key block;
                CHECK(ggml_tqnc_quantize_key(input.data() + input_offset + b*256, &block));
                std::memcpy(expected.data() + output_offset + b*block_bytes, &block, block_bytes);
            } else if (type == GGML_TYPE_TQNC_V) {
                ggml_tqnc_value block;
                CHECK(ggml_tqnc_quantize_value(input.data() + input_offset + b*256, &block));
                std::memcpy(expected.data() + output_offset + b*block_bytes, &block, block_bytes);
            } else {
                CHECK(ggml_quantize_chunk(type, input.data() + input_offset + b*256,
                        expected.data() + output_offset + b*block_bytes, 0, 1, 256, nullptr) == block_bytes);
            }
        }
    }
    CHECK(actual == expected);

    ids[2] = rows;
    set_ids();
    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);
    ggml_backend_tensor_get(dst, actual.data(), 0, actual.size());
    CHECK(actual == expected);
    ids[2] = 2;
    set_ids();

    input[size_t((2*n2 + 1)*batch + 1)*d + 13] = std::numeric_limits<float>::quiet_NaN();
    ggml_backend_tensor_set(src, input.data(), 0, input.size()*sizeof(float));
    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);
    ggml_backend_tensor_get(dst, actual.data(), 0, actual.size());
    CHECK(actual == expected);
    input[size_t((2*n2 + 1)*batch + 1)*d + 13] = 18.0f;
    ggml_backend_tensor_set(src, input.data(), 0, input.size()*sizeof(float));
    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    if (type == GGML_TYPE_SAW_INT4) {
        // BDR has a 16384 DC coefficient here, but inverse BDR is the
        // perfectly representable F16 constant 4096.
        std::fill(input.begin(), input.end(), 4096.0f);
        ggml_backend_tensor_set(src, input.data(), 0, input.size()*sizeof(float));
        CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(dst, actual.data(), 0, actual.size());
        block_saw_int4_ggml reference{};
        CHECK(ggml_quantize_chunk(type, input.data(), &reference, 0, 1, 256, nullptr) == sizeof(reference));
        CHECK(std::memcmp(actual.data(), &reference, sizeof(reference)) == 0);
    }
    input[size_t((2*n2 + 1)*batch + 1)*d + 13] = std::numeric_limits<float>::max();
    ggml_backend_tensor_set(src, input.data(), 0, input.size()*sizeof(float));
    ggml_backend_tensor_get(dst, expected.data(), 0, expected.size());
    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);
    ggml_backend_tensor_get(dst, actual.data(), 0, actual.size());
    CHECK(actual == expected);
    ggml_free(ctx);
}

// Corrupted metadata must stop attention before O/LSE are published and leave
// the same CUDA backend usable after the metadata is repaired.
static void test_attention_failure(ggml_backend_t backend) {
    constexpr int d = 256, n = 256, heads = 6;
    ggml_init_params params = { 8*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, 1, heads, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_TQNC_K, d, n, 1, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_TQNC_V, d, n, 1, 1);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n, 1);
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f/16.0f, 0.0f, 0.0f);
    ggml_tensor * lse = ggml_flash_attn_ext_lse_out(ctx, fa);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    graph->uid = 1002;
    ggml_build_forward_expand(graph, lse);
    CHECK(ggml_backend_supports_op(backend, fa));
    CHECK(ggml_backend_supports_op(backend, lse));
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx, backend));
    CHECK(buffer != nullptr);
    if (!buffer) { ggml_free(ctx); return; }

    std::vector<float> qv(d*heads, 0.125f), row(d, 0.5f), out(d*heads, 123.0f), sums(heads, 123.0f);
    std::vector<ggml_fp16_t> masks(n, ggml_fp32_to_fp16(0.0f));
    std::vector<ggml_tqnc_key> keys(n);
    std::vector<ggml_tqnc_value> vals(n);
    for (int i = 0; i < n; ++i) {
        CHECK(ggml_tqnc_quantize_key(row.data(), &keys[i]));
        CHECK(ggml_tqnc_quantize_value(row.data(), &vals[i]));
    }
    const ggml_fp16_t valid_norm = keys[13].norm;
    keys[13].norm = ggml_fp32_to_fp16(std::numeric_limits<float>::infinity());
    ggml_backend_tensor_set(q, qv.data(), 0, qv.size()*sizeof(float));
    ggml_backend_tensor_set(k, keys.data(), 0, keys.size()*sizeof(keys[0]));
    ggml_backend_tensor_set(v, vals.data(), 0, vals.size()*sizeof(vals[0]));
    ggml_backend_tensor_set(mask, masks.data(), 0, masks.size()*sizeof(masks[0]));
    ggml_backend_tensor_set(fa, out.data(), 0, out.size()*sizeof(float));
    ggml_backend_tensor_set(lse, sums.data(), 0, sums.size()*sizeof(float));
    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);
    ggml_backend_tensor_get(fa, out.data(), 0, out.size()*sizeof(float));
    ggml_backend_tensor_get(lse, sums.data(), 0, sums.size()*sizeof(float));
    CHECK(std::all_of(out.begin(), out.end(), [](float x) { return x == 123.0f; }));
    CHECK(std::all_of(sums.begin(), sums.end(), [](float x) { return x == 123.0f; }));

    keys[13].norm = valid_norm;
    ggml_backend_tensor_set(k, keys.data(), 0, keys.size()*sizeof(keys[0]));
    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(fa, out.data(), 0, out.size()*sizeof(float));
    ggml_backend_tensor_get(lse, sums.data(), 0, sums.size()*sizeof(float));
    CHECK(std::all_of(out.begin(), out.end(), [](float x) { return std::isfinite(x); }));
    CHECK(std::all_of(sums.begin(), sums.end(), [](float x) { return std::isfinite(x); }));

    const ggml_fp16_t valid_scale = vals[17].scale;
    vals[17].scale = ggml_fp32_to_fp16(-1.0f);
    std::fill(out.begin(), out.end(), 123.0f);
    std::fill(sums.begin(), sums.end(), 123.0f);
    ggml_backend_tensor_set(v, vals.data(), 0, vals.size()*sizeof(vals[0]));
    ggml_backend_tensor_set(fa, out.data(), 0, out.size()*sizeof(float));
    ggml_backend_tensor_set(lse, sums.data(), 0, sums.size()*sizeof(float));
    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);
    ggml_backend_tensor_get(fa, out.data(), 0, out.size()*sizeof(float));
    ggml_backend_tensor_get(lse, sums.data(), 0, sums.size()*sizeof(float));
    CHECK(std::all_of(out.begin(), out.end(), [](float x) { return x == 123.0f; }));
    CHECK(std::all_of(sums.begin(), sums.end(), [](float x) { return x == 123.0f; }));
    vals[17].scale = valid_scale;
    ggml_backend_tensor_set(v, vals.data(), 0, vals.size()*sizeof(vals[0]));
    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_free(ctx);
}

// An invalid NC body must not publish a tail-merged O or LSE. Exercise both
// the indexed-small (128+query) and padded packed-FA (256+query) tail paths.
static void test_attention_tail_failure(ggml_backend_t backend, int tail, int nq, bool packed, bool saw = false) {
    constexpr int d = 256, n = 256, nqh = 24, nkh = 4;
    const int stride = tail + nq;
    const int history = packed ? tail : stride;
    ggml_init_params params = { 16*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, nq, nqh, 1);
    auto * k = ggml_new_tensor_4d(ctx, saw ? GGML_TYPE_SAW_INT4 : GGML_TYPE_TQNC_K, d, n, nkh, 1);
    auto * v = ggml_new_tensor_4d(ctx, saw ? GGML_TYPE_SAW_INT4 : GGML_TYPE_TQNC_V, d, n, nkh, 1);
    auto * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n, nq);
    auto * kt = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, history, nkh, 1);
    auto * vt = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, history, nkh, 1);
    auto * kc = packed ? ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, nq, nkh, 1) : nullptr;
    auto * vc = packed ? ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, nq, nkh, 1) : nullptr;
    auto * mt = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, stride, nq);
    auto * qo = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, nq, 1);
    auto * rd = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 6 + stride + (packed ? n : 0), 1);
    auto * fa = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f/16.0f, 0.0f, 0.0f);
    if (packed) {
        fa = ggml_kv_tail_attention_merge_segmented(ctx, fa, kt, vt, kc, vc, mt, qo, rd);
        ggml_flash_attn_ext_set_kv_tail_history_slots(fa, history);
    } else {
        fa = ggml_kv_tail_attention_merge(ctx, fa, kt, vt, mt, qo, rd);
    }
    auto * lse = ggml_flash_attn_ext_lse_out(ctx, fa);
    auto * graph = ggml_new_graph(ctx);
    graph->uid = 1100 + tail + nq + int(packed);
    ggml_build_forward_expand(graph, lse);
    CHECK(ggml_backend_supports_op(backend, fa));
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx, backend));
    CHECK(buffer != nullptr);
    if (!buffer) { ggml_free(ctx); return; }

    std::vector<float> query(d*nq*nqh, 0.125f), row(d, 0.5f);
    std::vector<ggml_tqnc_key> keys(n*nkh);
    std::vector<ggml_tqnc_value> values(n*nkh);
    std::vector<block_saw_int4_ggml> saw_keys(n*nkh), saw_values(n*nkh);
    for (size_t i = 0; i < keys.size(); ++i) {
        if (saw) {
            CHECK(ggml_quantize_chunk(GGML_TYPE_SAW_INT4, row.data(), &saw_keys[i], 0, 1, d, nullptr) == sizeof(saw_keys[i]));
            saw_values[i] = saw_keys[i];
        } else {
            CHECK(ggml_tqnc_quantize_key(row.data(), &keys[i]));
            CHECK(ggml_tqnc_quantize_value(row.data(), &values[i]));
        }
    }
    const auto set_keys = [&] {
        if (saw) ggml_backend_tensor_set(k, saw_keys.data(), 0, saw_keys.size()*sizeof(saw_keys[0]));
        else ggml_backend_tensor_set(k, keys.data(), 0, keys.size()*sizeof(keys[0]));
    };
    const auto set_values = [&] {
        if (saw) ggml_backend_tensor_set(v, saw_values.data(), 0, saw_values.size()*sizeof(saw_values[0]));
        else ggml_backend_tensor_set(v, values.data(), 0, values.size()*sizeof(values[0]));
    };
    std::vector<ggml_fp16_t> body_mask(n*nq, ggml_fp32_to_fp16(0.0f));
    std::vector<ggml_fp16_t> tail_mask(stride*nq, ggml_fp32_to_fp16(0.0f));
    std::vector<ggml_fp16_t> tail_data(d*history*nkh, ggml_fp32_to_fp16(0.25f));
    std::vector<ggml_fp16_t> current_data(packed ? d*nq*nkh : 0, ggml_fp32_to_fp16(0.25f));
    std::vector<int32_t> order(nq), desc(6 + stride + (packed ? n : 0), -1);
    for (int i = 0; i < nq; ++i) order[i] = i;
    desc[4] = stride;
    for (int i = 0; i < stride; ++i) desc[6 + i] = i;
    if (packed) { desc[5] = n; for (int i = 0; i < n; ++i) desc[6 + stride + i] = i; }
    ggml_backend_tensor_set(q, query.data(), 0, query.size()*sizeof(float));
    set_keys();
    set_values();
    ggml_backend_tensor_set(mask, body_mask.data(), 0, body_mask.size()*sizeof(body_mask[0]));
    ggml_backend_tensor_set(kt, tail_data.data(), 0, tail_data.size()*sizeof(tail_data[0]));
    ggml_backend_tensor_set(vt, tail_data.data(), 0, tail_data.size()*sizeof(tail_data[0]));
    if (packed) {
        ggml_backend_tensor_set(kc, current_data.data(), 0, current_data.size()*sizeof(current_data[0]));
        ggml_backend_tensor_set(vc, current_data.data(), 0, current_data.size()*sizeof(current_data[0]));
    }
    ggml_backend_tensor_set(mt, tail_mask.data(), 0, tail_mask.size()*sizeof(tail_mask[0]));
    ggml_backend_tensor_set(qo, order.data(), 0, order.size()*sizeof(order[0]));
    ggml_backend_tensor_set(rd, desc.data(), 0, desc.size()*sizeof(desc[0]));
    std::vector<float> out(d*nq*nqh), sums(nq*nqh);
    const auto poison = [&] {
        std::fill(out.begin(), out.end(), 123.0f);
        std::fill(sums.begin(), sums.end(), 123.0f);
        ggml_backend_tensor_set(fa, out.data(), 0, out.size()*sizeof(float));
        ggml_backend_tensor_set(lse, sums.data(), 0, sums.size()*sizeof(float));
    };
    const auto unchanged = [&] {
        ggml_backend_tensor_get(fa, out.data(), 0, out.size()*sizeof(float));
        ggml_backend_tensor_get(lse, sums.data(), 0, sums.size()*sizeof(float));
        CHECK(std::all_of(out.begin(), out.end(), [](float x) { return x == 123.0f; }));
        CHECK(std::all_of(sums.begin(), sums.end(), [](float x) { return x == 123.0f; }));
    };
    const auto old_norm = keys[13].norm;
    const auto old_saw_scale_k = saw_keys[13].scale;
    if (saw) saw_keys[13].scale = std::numeric_limits<float>::infinity();
    else keys[13].norm = ggml_fp32_to_fp16(std::numeric_limits<float>::infinity());
    set_keys();
    poison();
    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);
    unchanged();
    if (saw) saw_keys[13].scale = old_saw_scale_k; else keys[13].norm = old_norm;
    set_keys();
    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    const auto old_scale = values[17].scale;
    const auto old_saw_scale_v = saw_values[17].scale;
    if (saw) saw_values[17].scale = -1.0f; else values[17].scale = ggml_fp32_to_fp16(-1.0f);
    set_values();
    poison();
    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);
    unchanged();
    if (saw) saw_values[17].scale = old_saw_scale_v; else values[17].scale = old_scale;
    set_values();
    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    if (saw) {
        const auto old_k = saw_keys[13], old_v = saw_values[17];
        std::fill(row.begin(), row.end(), 4096.0f);
        CHECK(ggml_quantize_chunk(GGML_TYPE_SAW_INT4, row.data(), &saw_keys[13], 0, 1, d, nullptr) == sizeof(old_k));
        saw_values[17] = saw_keys[13];
        set_keys(); set_values();
        CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        saw_keys[13].zero = std::numeric_limits<float>::quiet_NaN();
        set_keys(); poison();
        CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);
        unchanged();
        saw_keys[13] = old_k; saw_values[17] = old_v;
        set_keys(); set_values();
        CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    }

    // Malformed plan/data must fail before packing or consuming any result.
    const int32_t old_slot = desc[6];
    desc[6] = history + nq + 1;
    ggml_backend_tensor_set(rd, desc.data(), 0, desc.size()*sizeof(desc[0]));
    poison();
    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);
    unchanged();
    desc[6] = old_slot;
    ggml_backend_tensor_set(rd, desc.data(), 0, desc.size()*sizeof(desc[0]));

    if (packed) {
        const int32_t saved_row = desc[6 + stride];
        desc[6 + stride] = n;
        ggml_backend_tensor_set(rd, desc.data(), 0, desc.size()*sizeof(desc[0]));
        poison();
        CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);
        unchanged();
        desc[6 + stride] = saved_row;
        ggml_backend_tensor_set(rd, desc.data(), 0, desc.size()*sizeof(desc[0]));
    }
    const int32_t old_order = order[0];
    order[0] = nq;
    ggml_backend_tensor_set(qo, order.data(), 0, order.size()*sizeof(order[0]));
    poison();
    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);
    unchanged();
    order[0] = old_order;
    ggml_backend_tensor_set(qo, order.data(), 0, order.size()*sizeof(order[0]));

    const auto old_mask = tail_mask[0];
    tail_mask[0] = ggml_fp32_to_fp16(std::numeric_limits<float>::quiet_NaN());
    ggml_backend_tensor_set(mt, tail_mask.data(), 0, tail_mask.size()*sizeof(tail_mask[0]));
    poison();
    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);
    unchanged();
    tail_mask[0] = old_mask;
    ggml_backend_tensor_set(mt, tail_mask.data(), 0, tail_mask.size()*sizeof(tail_mask[0]));

    const float old_query = query[0];
    query[0] = std::numeric_limits<float>::quiet_NaN();
    ggml_backend_tensor_set(q, query.data(), 0, query.size()*sizeof(float));
    poison();
    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);
    unchanged();
    query[0] = old_query;
    ggml_backend_tensor_set(q, query.data(), 0, query.size()*sizeof(float));

    const auto old_tail = tail_data[0];
    tail_data[0] = ggml_fp32_to_fp16(std::numeric_limits<float>::quiet_NaN());
    ggml_backend_tensor_set(kt, tail_data.data(), 0, tail_data.size()*sizeof(tail_data[0]));
    poison();
    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);
    unchanged();
    tail_data[0] = old_tail;
    ggml_backend_tensor_set(kt, tail_data.data(), 0, tail_data.size()*sizeof(tail_data[0]));
    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(fa, out.data(), 0, out.size()*sizeof(float));
    ggml_backend_tensor_get(lse, sums.data(), 0, sums.size()*sizeof(float));
    CHECK(std::all_of(out.begin(), out.end(), [](float x) { return std::isfinite(x); }));
    CHECK(std::all_of(sums.begin(), sums.end(), [](float x) { return std::isfinite(x); }));
    std::printf("%s tail failure/reuse tail=%d nq=%d packed=%d\n", saw ? "SAW" : "NC", tail, nq, int(packed));
    ggml_free(ctx);
}

// Deliberately bypass supports_op: malformed NC input must fail in CUDA dispatch.
static void test_attention_dispatch_guards(ggml_backend_t backend) {
    constexpr int d = 256, n = 256, heads = 6;
    ggml_init_params params = { 8*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, 1, heads, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_TQNC_K, d, n, 1, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_TQNC_V, d, n, 1, 1);
    ggml_tensor * h = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, n, 1, 1);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n, 1);
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f/16.0f, 0, 0);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    graph->uid = 1010;
    ggml_build_forward_expand(graph, fa);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx, backend));
    CHECK(buffer != nullptr);
    if (!buffer) { ggml_free(ctx); return; }

    std::vector<float> query(d*heads, 0.125f), row(d, 0.5f), out(d*heads);
    std::vector<ggml_fp16_t> mask_data(n, ggml_fp32_to_fp16(0.0f));
    std::vector<ggml_tqnc_key> keys(n);
    std::vector<ggml_tqnc_value> values(n);
    for (int i = 0; i < n; ++i) {
        CHECK(ggml_tqnc_quantize_key(row.data(), &keys[i]));
        CHECK(ggml_tqnc_quantize_value(row.data(), &values[i]));
    }
    ggml_backend_tensor_set(q, query.data(), 0, query.size()*sizeof(float));
    ggml_backend_tensor_set(k, keys.data(), 0, keys.size()*sizeof(keys[0]));
    ggml_backend_tensor_set(v, values.data(), 0, values.size()*sizeof(values[0]));
    ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size()*sizeof(mask_data[0]));
    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    const auto reject = [&](ggml_tensor * key, ggml_tensor * value) {
        fa->src[1] = key;
        fa->src[2] = value;
        std::fill(out.begin(), out.end(), 123.0f);
        ggml_backend_tensor_set(fa, out.data(), 0, out.size()*sizeof(float));
        CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);
        ggml_backend_tensor_get(fa, out.data(), 0, out.size()*sizeof(float));
        CHECK(std::all_of(out.begin(), out.end(), [](float x) { return x == 123.0f; }));
    };
    reject(v, k); // both NC types swapped
    reject(v, h); // only V-type NC in the K position
    reject(h, k); // only K-type NC in the V position
    reject(k, h);
    reject(h, v);
    reject(k, k);
    reject(v, v);
    const size_t old_stride = k->nb[1];
    k->nb[1] += 2;
    reject(k, v);
    k->nb[1] = old_stride;
    fa->src[1] = k;
    fa->src[2] = v;
    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_free(ctx);
}

int main() {
    int devices = 0;
    const cudaError_t probe_status = cudaGetDeviceCount(&devices);
    const int availability = cuda_test_availability(probe_status, devices);
    if (availability != 0) {
        std::fprintf(stderr, "%s: CUDA device probe: %s (count=%d)\n",
                     availability == 77 ? "SKIP" : "FAIL", cudaGetErrorString(probe_status), devices);
        return availability;
    }
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "CUDA backend unavailable\n"); return 1; }
    test_view_admission(backend);
    for (ggml_type type : { GGML_TYPE_I32, GGML_TYPE_I64 }) {
        test_set_rows_failure(backend, type, false);
        test_set_rows_failure(backend, type, true);
        test_multistream_rows(backend, GGML_TYPE_TQNC_K, type);
        test_multistream_rows(backend, GGML_TYPE_TQNC_V, type);
        test_multistream_rows(backend, GGML_TYPE_SAW_INT4, type);
    }
    test_attention_failure(backend);
    test_attention_tail_failure(backend, 128, 1, false);
    test_attention_tail_failure(backend, 128, 32, true);
    test_attention_tail_failure(backend, 256, 1, false);
    test_attention_tail_failure(backend, 256, 32, true);
    test_attention_tail_failure(backend, 128, 1, false, true);
    test_attention_tail_failure(backend, 128, 32, true, true);
    test_attention_tail_failure(backend, 256, 1, false, true);
    test_attention_tail_failure(backend, 256, 32, true, true);
    test_attention_dispatch_guards(backend);
    ggml_backend_free(backend);
    std::printf("TurboQuant NC transactional backend: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
