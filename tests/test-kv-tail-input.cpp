#include "llama-kv-cache.h"
#include "llama-kv-cache-kvarn.h"
#include "llama-model.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(bool ok, const char * message) {
    if (!ok) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

static void persistent_stage_descriptor(ggml_backend_t backend, int dimension) {
    constexpr int head_slices_param = 5; // ggml_kvarn_view descriptor ABI
    std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_LLAMA, llama_model_default_params()));
    auto & hp = model->hparams;
    hp = {};
    hp.n_layer_all = 1;
    hp.n_embd = dimension;
    hp.n_ctx_train = 512;
    hp.n_embd_head_k_full = dimension;
    hp.n_embd_head_v_full = dimension;
    hp.n_head_arr[0] = 1;
    hp.n_head_kv_arr[0] = 1;
    model->layers.resize(1);
    llama_kv_cache_kvarn cache(*model, hp, llama_kvarn_params_for_type(LLAMA_KVARN_K4V4_G128),
            false, true, 512, 1, 32, 32);
    auto ctx = ggml_init({1024*1024, nullptr, true});
    auto indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
    auto query = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dimension, 2, 1, 1);
    llama_kv_cache::slot_info slots = {};
    slots.s0 = slots.s1 = 0;
    slots.strm = {0};
    slots.idxs = {{0}};
    ggml_tensor * key_stage = cache.get_materialization_source(0, false);
    ggml_tensor * value_stage = cache.get_materialization_source(0, true);
    check(key_stage->op == GGML_OP_NONE && value_stage->op == GGML_OP_NONE,
            "fixture must read persistent stage, without a KVARN_STORE result");
    check(key_stage->op_params[head_slices_param] == 0 &&
          value_stage->op_params[head_slices_param] == 0,
            "persistent NONE stage must not supply inherited store geometry");
    auto key = cache.view(ctx, key_stage, 0, 1, slots, false, indices);
    auto value = cache.view(ctx, value_stage, 0, 1, slots, true, indices);
    for (auto tensor : {key, value}) {
        auto descriptor = tensor;
        while (descriptor->op == GGML_OP_RESHAPE) { descriptor = descriptor->src[0]; }
        check(descriptor->op == GGML_OP_KVARN_VIEW, "expected native KVarN descriptor");
        check(descriptor->op_params[head_slices_param] == dimension/128,
                "persistent-stage descriptor lost its logical head slices");
        check(tensor->ne[0] == dimension && tensor->ne[1] == 1,
                "persistent-stage logical head geometry differs from cache");
    }
    key = ggml_permute(ctx, key, 0, 2, 1, 3);
    value = ggml_permute(ctx, value, 0, 2, 1, 3);
    auto attention = ggml_flash_attn_ext(ctx, query, key, value, nullptr, 1.0f, 0.0f, 0.0f);
    attention->op_params[GGML_FLASH_ATTN_EXT_OP_PARAM_KVARN_DOMAIN] = GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED;

    // A masked candidate can refer to the absent current segment. It must be
    // skipped BEFORE validating a physical history slot or loading K/V.
    auto tail_key = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, dimension, 1, 1, 1);
    auto tail_value = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, dimension, 1, 1, 1);
    auto tail_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 2, 1, 1);
    auto order = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 1);
    auto desc = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 9, 1);
    attention = ggml_kv_tail_attention_merge(ctx, attention, tail_key, tail_value, tail_mask, order, desc);
    auto lse = ggml_flash_attn_ext_lse_out(ctx, attention);
    auto graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, attention);
    ggml_build_forward_expand(graph, lse);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    check(buffer != nullptr, "persistent-stage test graph allocation failed");
    check(ggml_backend_supports_op(backend, attention), "CPU rejected native persistent-stage attention");
    // Passing explicit mat_idxs makes cache.view a READ_INDIRECT descriptor:
    // nonnegative 0 selects compressed record 0, while this fixture populated
    // the persistent F16 stage. Bind both the cell and its physical stage slot.
    const int64_t first_stage_cell = llama_kvarn_encode_stage_cell(0, 0);
    check(llama_kvarn_decode_cell(first_stage_cell) == 0 &&
          llama_kvarn_decode_stage_slot(first_stage_cell) == 0,
            "persistent-stage fixture must explicitly select F16 stage slot zero");
    ggml_backend_tensor_set(indices, &first_stage_cell, 0, sizeof(first_stage_cell));
    std::vector<float> q(size_t(dimension)*2, 1.0f);
    ggml_backend_tensor_set(query, q.data(), 0, q.size()*sizeof(float));
    std::vector<ggml_fp16_t> zeros(size_t(dimension), ggml_fp32_to_fp16(0.0f));
    std::vector<ggml_fp16_t> v(size_t(dimension), 0);
    for (int i = 0; i < dimension; ++i) { v[size_t(i)] = ggml_fp32_to_fp16(0.01f*(1+i%17)); }
    ggml_backend_tensor_set(key_stage, zeros.data(), 0, zeros.size()*sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(value_stage, v.data(), 0, v.size()*sizeof(ggml_fp16_t));
    std::vector<ggml_fp16_t> poisoned(size_t(dimension), ggml_fp32_to_fp16(NAN));
    ggml_backend_tensor_set(tail_key, poisoned.data(), 0, poisoned.size()*sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(tail_value, poisoned.data(), 0, poisoned.size()*sizeof(ggml_fp16_t));
    std::vector<ggml_fp16_t> masks(6, ggml_fp32_to_fp16(-INFINITY));
    ggml_backend_tensor_set(tail_mask, masks.data(), 0, masks.size()*sizeof(ggml_fp16_t));
    const int32_t query_order[] = {0, 1};
    const int32_t run_desc[] = {0, 0, 2, 1, 1, -1, 3, -1, -1};
    ggml_backend_tensor_set(order, query_order, 0, sizeof(query_order));
    ggml_backend_tensor_set(desc, run_desc, 0, sizeof(run_desc));
    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            "CPU native descriptor or masked absent-current tail computation failed");
    std::vector<float> got(size_t(dimension)*2, 0);
    ggml_backend_tensor_get(attention, got.data(), 0, got.size()*sizeof(float));
    for (size_t i = 0; i < got.size(); ++i) {
        check(std::isfinite(got[i]) && std::abs(got[i]-ggml_fp16_to_fp32(v[i%size_t(dimension)])) <= 1e-6f,
                "native persistent-stage attention did not reproduce its single visible V row");
    }
    float logsum[2];
    ggml_backend_tensor_get(lse, logsum, 0, sizeof(logsum));
    check(logsum[0] == 0.0f && logsum[1] == 0.0f, "masked tail changed single-key LSE");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

static void mapped_local_tail_plan(ggml_backend_t backend) {
    std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_LLAMA, llama_model_default_params()));
    llama_kv_cache cache(*model, model->hparams, GGML_TYPE_F16, GGML_TYPE_F16,
            false, false, true, 512, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
            nullptr, {}, {}, {}, "", 4, 2, GGML_TYPE_F16, 2, true, 2);
    llama_kv_tail_layer_route route = {};
    route.capability.supported = true;
    route.capability.route = LLAMA_KV_TAIL_ROUTE_NATIVE;
    cache.set_tail_routes({route});
    cache.finalize_tail_overlay_metadata();
    cache.set_position_split_p(256);
    llama_pos local_positions[] = {254, 255};
    llama_pos overflow_positions[] = {256, 257};
    int32_t counts[] = {1, 1};
    llama_seq_id owner = 0;
    llama_seq_id * owners[] = {&owner, &owner};
    llama_ubatch batch = {};
    batch.n_tokens = 2; batch.n_pos = 1;
    batch.pos = local_positions; batch.n_seq_id = counts; batch.seq_id = owners;
    llama_kv_cache::slot_info slots = {};
    slots.s0 = slots.s1 = 0; slots.strm = {0}; slots.idxs = {{254, 255}};
    cache.apply_ubatch(slots, batch);
    cache.finish_tail_batch(true, false);
    batch.pos = overflow_positions; slots.idxs = {{256, 257}};
    cache.apply_ubatch(slots, batch);

    auto ctx = ggml_init({1024*1024, nullptr, true});
    // P-aware graph builder uses one effective query for overflow, even when
    // the ubatch contains two queries. The metadata planner must agree.
    const uint32_t stride = cache.get_tail_attention_stride(1);
    auto local = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, 2, 1, 1);
    auto exact = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, stride, 2, 1, 1);
    auto read = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, stride, 2);
    auto body_read = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, stride, 2);
    auto bias_read = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, stride, 2);
    auto order = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 1);
    auto desc = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 6+stride, 1);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    check(buffer != nullptr, "mapped local tail inputs allocation failed");
    auto pruned_ctx = ggml_init({ggml_tensor_overhead()+256, nullptr, true});
    auto pruned_base = ggml_new_tensor_4d(pruned_ctx, GGML_TYPE_F32, 512, 2, 1, 1);
    check(pruned_base->buffer == nullptr, "general mask must remain pruned in the fixture");
    std::vector<float> mask(512, -INFINITY);
    mask[0] = mask[1] = mask[256] = mask[257] = 0;
    ggml_backend_tensor_set(local, mask.data(), 0, mask.size()*sizeof(float));
    const int32_t query_order[] = {0, 1};
    std::vector<int32_t> run_desc(size_t(6+stride), -1);
    run_desc[0] = 0; run_desc[1] = 0; run_desc[2] = 2; run_desc[3] = 1;
    ggml_backend_tensor_set(order, query_order, 0, sizeof(query_order));
    ggml_backend_tensor_set(desc, run_desc.data(), 0, run_desc.size()*sizeof(int32_t));
    std::vector<int64_t> map(256, -1);
    map[0] = 254; map[1] = 255;
    cache.set_input_kq_mask_tail_mapped(local, exact, read, body_read, bias_read, &batch, true, map);
    cache.set_input_tail_body_plan(order, desc, local, &batch, true);
    ggml_backend_tensor_get(desc, run_desc.data(), 0, run_desc.size()*sizeof(int32_t));
    check(run_desc[4] == 2, "overflow Q>1 planner left local tail count uninitialized or selected overflow rows");
    check(run_desc[5] == -1, "non-packed local body descriptor unexpectedly changed body mode");
    std::vector<float> tails(size_t(stride)*2, 0);
    std::vector<int32_t> mapped(size_t(stride)*2, -1);
    ggml_backend_tensor_get(exact, tails.data(), 0, tails.size()*sizeof(float));
    ggml_backend_tensor_get(body_read, mapped.data(), 0, mapped.size()*sizeof(int32_t));
    ggml_backend_tensor_get(local, mask.data(), 0, mask.size()*sizeof(float));
    for (size_t query = 0; query < 2; ++query) {
        int selected = 0;
        for (size_t i = query*stride; i < (query+1)*stride; ++i) {
            if (std::isfinite(tails[i])) {
                ++selected;
                check(mapped[i] == 0 || mapped[i] == 1, "tail fallback row uses physical cell instead of local mapped index");
            }
        }
        check(selected == 2, "local exact mask did not select the two persistent history rows");
        for (size_t i = query*256; i < (query+1)*256; ++i) {
            check(mask[i] == -INFINITY, "tail row remained visible in local body (duplicate attention)");
        }
    }
    cache.finish_tail_batch(true, false);
    ggml_free(pruned_ctx);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

int main() {
    ggml_backend_load_all();
    auto backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    check(backend != nullptr, "CPU backend unavailable");
    persistent_stage_descriptor(backend, 128);
    persistent_stage_descriptor(backend, 256);
    persistent_stage_descriptor(backend, 512);
    mapped_local_tail_plan(backend);
    std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_LLAMA, llama_model_default_params()));
    // No model tensors: exercise the same metadata cache used by structured caches.
    llama_kv_cache cache(*model, model->hparams, GGML_TYPE_F16, GGML_TYPE_F16,
            false, false, true, 32, 2, 1, 0, LLAMA_SWA_TYPE_NONE,
            nullptr, {}, {}, {}, "", 4, 2, GGML_TYPE_F16, 2, true, 0);
    check(cache.has_compact_tail(), "fixture must use a compact tail");
    llama_kv_tail_layer_route route = {};
    route.capability.supported = true;
    route.capability.route = LLAMA_KV_TAIL_ROUTE_NATIVE;
    cache.set_tail_routes({route});
    cache.finalize_tail_overlay_metadata();

    llama_pos positions[] = {0, 1, 2, 3};
    int32_t counts[] = {2, 1, 1, 1};
    llama_seq_id owners[] = {1, 0};
    llama_seq_id * seqs[] = {owners, owners + 1, owners + 1, owners + 1};
    llama_ubatch batch = {};
    batch.n_tokens = 4;
    batch.n_pos = 1;
    batch.pos = positions;
    batch.n_seq_id = counts;
    batch.seq_id = seqs;
    llama_kv_cache::slot_info slots = {};
    slots.strm = {0};
    slots.idxs = {{0, 1, 2, 3}};
    cache.apply_ubatch(slots, batch);

    auto ctx = ggml_init({1024*1024, nullptr, true});
    auto indices = cache.build_input_tail_idxs(ctx, batch);
    check(indices && indices->ne[1] == 2, "fixture must produce two owner levels");
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    check(buffer != nullptr, "input allocation failed");
    cache.set_input_tail_idxs(indices, &batch);
    std::vector<int64_t> got(8);
    ggml_backend_tensor_get(indices, got.data(), 0, got.size()*sizeof(int64_t));
    // Level 1/row 0 and level 0/row 2 reuse slot 0. Only row 2 persists.
    check(got[2] == 0 && got[3] == 1 && got[4] == -1,
            "production input did not retain the globally last slot write");
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] < 0) continue;
        for (size_t j = i + 1; j < got.size(); ++j) {
            check(got[i] != got[j], "production input retained duplicate destinations");
        }
    }
    cache.finish_tail_batch(true, false);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}
