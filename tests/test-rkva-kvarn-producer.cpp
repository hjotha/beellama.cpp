// Native producer/wire regression. Replays actual append packets into an
// independent console-layout image and compares it byte-for-byte with one
// canonical CPU KVarN store over the original projected F32 values.
#include "../src/llama-native-kvarn-rollback.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-remote-attn/rkva-kvarn-producer.h"
#include "../ggml/src/ggml-remote-attn/rkva-kvarn-rewind.h"
#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <limits>

static void test_precision_tail(uint32_t precision_tail, bool compact_stage = false) {
    constexpr uint32_t origin = 1024, capacity = 640, heads = 1, count = 513;
    constexpr size_t width = heads * 256, physical_heads = heads * 2, record = 8960;
    auto * reg = ggml_backend_cpu_reg();
    auto store = reinterpret_cast<ggml_backend_cpu_kvarn_store_direct_v1_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_kvarn_store_direct_v1"));
    assert(store);
    // RS=0 checkpoint fallback restores before the anchor: N128 would need
    // 129 original rows even though only 128 proposals can be rejected.
    assert(llama_native_kvarn_mtp_depth_supported(127));
    assert(!llama_native_kvarn_mtp_depth_supported(128));
    assert(!llama_native_kvarn_mtp_depth_supported(UINT32_MAX));
    const uint64_t generation = rkva_kvarn_new_generation_seed();
    const uint64_t reconfigured = rkva_kvarn_new_generation_seed();
    assert(generation && reconfigured > generation && uint32_t(generation) == 0);
    assert(rkva_kvarn_generation_mutable(generation));
    assert(!rkva_kvarn_generation_mutable(generation | UINT32_MAX));
    std::vector<float> keys(size_t(count) * width), values(keys.size());
    std::vector<int32_t> pos(count);
    std::vector<int64_t> idx(count);
    for (uint32_t t = 0; t < count; ++t) {
        pos[t] = origin + t; idx[t] = pos[t];
        for (size_t d = 0; d < width; ++d) {
            keys[size_t(t) * width + d] = std::sin(float(t * 7 + d * 3) * 0.003f) * 0.7f;
            values[size_t(t) * width + d] = std::cos(float(t * 11 + d * 5) * 0.007f) * 0.9f;
        }
    }
    const size_t global_groups = (origin + capacity + 127) / 128;
    std::vector<uint8_t> expected_rk(global_groups * physical_heads * record), expected_rv(expected_rk.size());
    std::vector<ggml_fp16_t> expected_sk(384 * width), expected_sv(expected_sk.size());
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({16 * ggml_tensor_overhead(), nullptr, true}), ggml_free);
    assert(ctx);
    auto * current = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 128, physical_heads, count);
    auto * indices = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, count);
    indices->data = idx.data();
    auto * stage = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, 128, physical_heads, 384);
    auto * records = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_I8, record, physical_heads, global_groups);
    auto * kstore = ggml_kvarn_store(ctx.get(), current, indices, stage, records, 4, 3, false, 3);
    auto * vstore = ggml_kvarn_store(ctx.get(), current, indices, stage, records, 4, 3, true, 3);
    for (auto * op : {kstore, vstore}) {
        ggml_set_op_params_i32(op, 5, 2);
        ggml_set_op_params_i32(op, 9, 1);
    }
    current->data = keys.data(); stage->data = expected_sk.data(); records->data = expected_rk.data();
    assert(store(kstore));
    current->data = values.data(); stage->data = expected_sv.data(); records->data = expected_rv.data();
    assert(store(vstore));

    rkva_kvarn_producer producer(origin, capacity, heads, 3, store, precision_tail, compact_stage);
    std::vector<uint8_t> image_rk((capacity + 127) / 128 * physical_heads * record), image_rv(image_rk.size());
    std::vector<ggml_fp16_t> image_sk(384 * width), image_sv(image_sk.size());
    std::vector<ggml_fp16_t> image_tk(128 * width), image_tv(image_tk.size());
    std::vector<uint8_t> packet;
    uint32_t previous = 0;
    // The middle append crosses THREE global groups and wraps both stage and
    // precision-tail rings. Reading only the final stage ring would lose rows.
    for (const uint32_t n : {127u, 256u, 130u}) {
        assert(producer.append(keys.data() + size_t(previous) * width,
            values.data() + size_t(previous) * width, pos.data() + previous, n, previous == 0, packet));
        rkva_kvarn_range_upload u;
        assert(packet.size() >= sizeof(u)); std::memcpy(&u, packet.data(), sizeof(u));
        assert(u.live_count == previous + n && u.record_begin == previous / 128);
        assert(u.record_groups == u.live_count / 128 - previous / 128);
        assert(u.flags == ((previous == 0 ? 1u : 0u) |
            (compact_stage ? RKVA_KVARN_UPLOAD_OMIT_STAGE : 0u)));
        size_t offset = sizeof(u);
        const auto take = [&](void * destination, size_t bytes) {
            assert(offset + bytes <= packet.size());
            std::memcpy(destination, packet.data() + offset, bytes); offset += bytes;
        };
        const size_t record_bytes = size_t(u.record_groups) * physical_heads * record;
        take(image_rk.data() + size_t(u.record_begin) * physical_heads * record, record_bytes);
        take(image_rv.data() + size_t(u.record_begin) * physical_heads * record, record_bytes);
        if (!compact_stage) for (auto * image : {&image_sk, &image_sv}) {
            if (u.flags & RKVA_KVARN_UPLOAD_RESET) take(image->data(), image->size() * 2);
            else for (uint32_t t = previous; t < u.live_count; ++t) {
                const uint32_t g = (origin + t) / 128;
                const uint32_t slot = g == 0 ? 0 : 1 + (g - 1) % 2;
                take(image->data() + size_t(slot * 128 + (origin + t) % 128) * width, width * 2);
            }
        }
        if (precision_tail) for (auto * image : {&image_tk, &image_tv}) {
            const uint32_t start = (u.flags & RKVA_KVARN_UPLOAD_RESET) ? u.live_count - std::min(u.live_count, 128u) : previous;
            for (uint32_t t = start; t < u.live_count; ++t) {
                take(image->data() + size_t(t % 128) * width, width * 2);
            }
        }
        assert(offset == packet.size());
        previous += n;
    }
    assert(previous == count && producer.rows() == count);
    if (!compact_stage) assert(image_sk == expected_sk && image_sv == expected_sv);
    const size_t begin = size_t(origin / 128) * physical_heads * record;
    const size_t committed = size_t(count / 128) * physical_heads * record;
    assert(std::memcmp(image_rk.data(), expected_rk.data() + begin, committed) == 0);
    assert(std::memcmp(image_rv.data(), expected_rv.data() + begin, committed) == 0);
    for (uint32_t t = count - std::min(count, precision_tail); t < count; ++t) for (size_t d = 0; d < width; ++d) {
        assert(image_tk[size_t(t % 128) * width + d] == ggml_fp32_to_fp16(keys[size_t(t) * width + d]));
        assert(image_tv[size_t(t % 128) * width + d] == ggml_fp32_to_fp16(values[size_t(t) * width + d]));
    }
    // Compare resealed native bytes against an independent, fresh canonical
    // CPU store over the accepted prefix plus different regenerated F32 rows.
    for (const auto boundary : {std::pair<uint32_t, uint32_t>{129, 127}, {256, 128}, {257, 129}, {385, 257}}) {
        const uint32_t old = boundary.first, keep = boundary.second;
        rkva_kvarn_producer trial(origin, capacity, heads, 3, store, precision_tail, compact_stage);
        for (uint32_t begin = 0; begin < old;) {
            const uint32_t n = std::min(128u, old - begin);
            assert(trial.append(keys.data() + size_t(begin) * width, values.data() + size_t(begin) * width,
                pos.data() + begin, n, begin == 0, packet));
            begin += n;
        }
        assert(!trial.can_rewind(old + 1));
        assert(trial.can_rewind(keep));
        if (old > 128) assert(!trial.can_rewind(old - 129));
        std::vector<uint8_t> repair;
        assert(trial.rewind_tail(keep, repair));
        const uint32_t repair_begin = keep > 128 ? keep - 128 : 0;
        const size_t tail_rows = precision_tail ? std::min(keep, old > 128 ? old - 128 : 0) - repair_begin : 0;
        assert(repair.size() == tail_rows * width * 2 * 2);
        for (uint32_t kind = 0; kind < 2; ++kind) {
            const auto & source = kind ? values : keys;
            for (size_t row = 0; row < tail_rows; ++row) for (size_t d = 0; d < width; ++d) {
                ggml_fp16_t half;
                std::memcpy(&half, repair.data() + (kind * tail_rows * width + row * width + d) * 2, 2);
                assert(half == ggml_fp32_to_fp16(source[(repair_begin + row) * width + d]));
            }
        }
        // The production all-layer wire parser rejects stale expected counts,
        // truncated payloads and distances outside retained originals before
        // exposing a plan. Exercise both precision-tail descriptors.
        {
            const uint32_t tail = precision_tail;
            rkva_kvarn_range_rewind request {generation, origin + keep, 1, tail, 0};
            rkva_kvarn_layer_rewind layer {old, keep};
            std::vector<uint8_t> body(sizeof(layer)); std::memcpy(body.data(), &layer, sizeof(layer));
            if (tail) body.insert(body.end(), repair.begin(), repair.end());
            rkva_kvarn_rewind_plan plan;
            assert(rkva_kvarn_plan_rewind(request, origin, capacity, heads, tail, generation, {old}, {true}, body.data(), body.size(), plan));
            assert(plan.layers.size() == 1 && plan.layers[0].new_live == keep);
            // Same-count ABA after rewind/reappend must not accept an old tail patch.
            assert(!rkva_kvarn_plan_rewind(request, origin, capacity, heads, tail, generation + 2, {old}, {true}, body.data(), body.size(), plan));
            assert(!rkva_kvarn_plan_rewind(request, origin, capacity, heads, tail, reconfigured, {old}, {true}, body.data(), body.size(), plan));
            assert(!rkva_kvarn_plan_rewind(request, origin, capacity, heads, tail, UINT64_MAX, {old}, {true}, body.data(), body.size(), plan));
            assert(!rkva_kvarn_plan_rewind(request, origin, capacity, heads, tail, generation, {old - 1}, {true}, body.data(), body.size(), plan));
            assert(!rkva_kvarn_plan_rewind(request, origin, capacity, heads, tail, generation, {old}, {true}, body.data(), body.size() - 1, plan));
        }
        assert(trial.rewind(keep));
        if (keep > 128 && old - keep == 128) assert(!trial.can_rewind(keep - 128));
        assert(trial.rewind(keep)); // repeated valid anchor
        auto changed_k = keys, changed_v = values;
        for (size_t i = size_t(keep) * width; i < size_t(old) * width; ++i) {
            changed_k[i] = -keys[i] * 0.75f; changed_v[i] = values[i] + 0.25f;
        }
        assert(trial.append(changed_k.data() + size_t(keep) * width, changed_v.data() + size_t(keep) * width,
            pos.data() + keep, old - keep, false, packet));
        std::fill(expected_sk.begin(), expected_sk.end(), 0); std::fill(expected_sv.begin(), expected_sv.end(), 0);
        std::fill(expected_rk.begin(), expected_rk.end(), 0); std::fill(expected_rv.begin(), expected_rv.end(), 0);
        current->ne[2] = old; indices->ne[0] = old;
        current->data = changed_k.data(); stage->data = expected_sk.data(); records->data = expected_rk.data(); assert(store(kstore));
        current->data = changed_v.data(); stage->data = expected_sv.data(); records->data = expected_rv.data(); assert(store(vstore));
        rkva_kvarn_range_upload u; std::memcpy(&u, packet.data(), sizeof(u));
        const size_t bytes = size_t(u.record_groups) * physical_heads * record;
        const size_t global_offset = size_t(origin / 128 + u.record_begin) * physical_heads * record;
        assert(std::memcmp(packet.data() + sizeof(u), expected_rk.data() + global_offset, bytes) == 0);
        assert(std::memcmp(packet.data() + sizeof(u) + bytes, expected_rv.data() + global_offset, bytes) == 0);
        assert(trial.rows() == old);
    }
    packet.clear();
    const auto before = packet;
    int32_t wrong = origin; // a quantized/history bootstrap or rewind is refused
    assert(!producer.append(keys.data(), values.data(), &wrong, 1, false, packet));
    assert(producer.rows() == count && packet == before);
    int32_t next = origin + count;
    keys[0] = std::numeric_limits<float>::quiet_NaN();
    assert(!producer.append(keys.data(), values.data(), &next, 1, false, packet));
    assert(producer.rows() == count && packet == before);
    std::puts("PASS native KVarN4 original-F32 producer, canonical CPU record bytes, wire stage/tail wrap, bounded original-F32 rewind/reseal, stale wire counts, refused replay/NaN");
}

static void test_compact_stage_gates() {
    auto store = reinterpret_cast<ggml_backend_cpu_kvarn_store_direct_v1_t>(
        ggml_backend_reg_get_proc_address(ggml_backend_cpu_reg(),
            "ggml_backend_cpu_kvarn_store_direct_v1"));
    assert(store);
    std::vector<float> k(256, 0.25f), v(256, -0.5f);
    for (uint32_t origin : {0u, 1024u}) {
        for (uint32_t tail : {0u, 128u}) {
            for (bool negotiated : {false, true}) {
                rkva_kvarn_producer producer(origin, 256, 1, 3, store, tail, negotiated);
                const int32_t pos = int32_t(origin);
                std::vector<uint8_t> packet;
                assert(producer.append(k.data(), v.data(), &pos, 1, true, packet));
                rkva_kvarn_range_upload u;
                std::memcpy(&u, packet.data(), sizeof(u));
                const bool compact = negotiated && origin >= 128 && tail == 128;
                assert(bool(u.flags & RKVA_KVARN_UPLOAD_OMIT_STAGE) == compact);
                const size_t row_bytes = 256 * sizeof(ggml_fp16_t);
                assert(packet.size() == sizeof(u) + 2 * row_bytes *
                    ((compact ? 0u : 384u) + (tail ? 1u : 0u)));
            }
        }
    }
    std::puts("PASS compact-stage negotiation fallback, global sink and tail-zero gates");
}

static void test_append_scratch_reuse() {
    auto store = reinterpret_cast<ggml_backend_cpu_kvarn_store_direct_v1_t>(
        ggml_backend_reg_get_proc_address(ggml_backend_cpu_reg(),
            "ggml_backend_cpu_kvarn_store_direct_v1"));
    assert(store);
    constexpr uint32_t origin = 1024, capacity = 1024, width = 256;
    std::vector<float> k(size_t(capacity) * width), v(k.size());
    std::vector<int32_t> positions(capacity);
    for (uint32_t t = 0; t < capacity; ++t) {
        positions[t] = int32_t(origin + t);
        for (uint32_t d = 0; d < width; ++d) {
            k[size_t(t) * width + d] = std::sin(float(t * 7 + d) * 0.01f);
            v[size_t(t) * width + d] = std::cos(float(t + d * 3) * 0.02f);
        }
    }
    for (bool compact : {false, true}) {
        for (const uint32_t shape : {1u, 8u, 9u, 128u, 256u}) {
            rkva_kvarn_producer producer(origin, capacity, 1, 3, store, 128, compact);
            assert(producer.append_scratch_grows() == 0);
            assert(producer.append_scratch_capacity_bytes() == 0);
            std::vector<uint8_t> packet;
            const size_t before_bytes = producer.host_bytes();
            // Warm the maximum production shape before replaying smaller
            // segment/query counts; this is not a whole-path allocation claim.
            assert(producer.append(k.data(), v.data(), positions.data(), 256, true, packet));
            const size_t warm_growth = producer.append_scratch_grows();
            const size_t scratch_bytes = producer.append_scratch_capacity_bytes();
            assert(warm_growth == 2);
            assert(scratch_bytes >= 2 * size_t(256) * width * sizeof(ggml_fp16_t));
            assert(producer.host_bytes() == before_bytes + scratch_bytes);
            uint32_t begin = 256;
            for (uint32_t repeat = 0; repeat < 3; ++repeat) {
                assert(producer.append(k.data() + size_t(begin) * width,
                    v.data() + size_t(begin) * width, positions.data() + begin,
                    shape, false, packet));
                assert(producer.append_scratch_grows() == warm_growth);
                assert(producer.append_scratch_capacity_bytes() == scratch_bytes);
                rkva_kvarn_range_upload header{};
                std::memcpy(&header, packet.data(), sizeof(header));
                assert(header.live_count == begin + shape);
                assert(header.append_rows == shape);
                const size_t records_bytes = size_t(header.record_groups) * 2 * 8960 * 2;
                const size_t new_rows_bytes = size_t(shape) * width * sizeof(ggml_fp16_t);
                assert(packet.size() == sizeof(header) + records_bytes +
                    2 * new_rows_bytes * (compact ? 1 : 2));
                // The original-domain tail is always the last two sections.
                // A smaller shape must not emit a prior larger scratch suffix.
                const size_t tail_offset = packet.size() - 2 * new_rows_bytes;
                for (size_t kind = 0; kind < 2; ++kind) {
                    const auto & source = kind ? v : k;
                    for (size_t e = 0; e < size_t(shape) * width; ++e) {
                        ggml_fp16_t actual;
                        std::memcpy(&actual, packet.data() + tail_offset + kind * new_rows_bytes +
                            e * sizeof(actual), sizeof(actual));
                        assert(actual == ggml_fp32_to_fp16(source[size_t(begin) * width + e]));
                    }
                }
                begin += shape;
            }
            // Rewind can internally append a smaller incomplete group. It must
            // preserve the high-water buffers and produce the same reseal bytes
            // already checked independently by test_precision_tail().
            const uint32_t keep = begin - std::min(shape, 128u);
            assert(producer.rewind(keep));
            assert(producer.append_scratch_grows() == warm_growth);
            assert(producer.append_scratch_capacity_bytes() == scratch_bytes);
            assert(producer.append(k.data() + size_t(keep) * width,
                v.data() + size_t(keep) * width, positions.data() + keep,
                begin - keep, false, packet));
            assert(producer.rows() == begin);
            assert(producer.append_scratch_grows() == warm_growth);
            assert(producer.append_scratch_capacity_bytes() == scratch_bytes);
            std::printf("PASS append scratch compact=%d shape=%u warm_growth=%zu post_warmup_growth=%zu retained_bytes=%zu\n",
                int(compact), shape, warm_growth, producer.append_scratch_grows() - warm_growth, scratch_bytes);
        }
    }
}

int main() {
    test_compact_stage_gates();
    test_precision_tail(0);
    test_precision_tail(128);
    test_precision_tail(128, true);
    test_append_scratch_reuse();
}
