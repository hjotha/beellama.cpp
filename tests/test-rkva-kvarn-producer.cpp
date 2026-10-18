// Native producer/wire regression. Replays actual append packets into an
// independent console-layout image and compares it byte-for-byte with one
// canonical CPU KVarN store over the original projected F32 values.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-remote-attn/rkva-kvarn-producer.h"
#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <limits>

int main() {
    constexpr uint32_t origin = 1024, capacity = 640, heads = 1, count = 513;
    constexpr size_t width = heads * 256, physical_heads = heads * 2, record = 8960;
    auto * reg = ggml_backend_cpu_reg();
    auto store = reinterpret_cast<ggml_backend_cpu_kvarn_store_direct_v1_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_kvarn_store_direct_v1"));
    assert(store);
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

    rkva_kvarn_producer producer(origin, capacity, heads, 3, store);
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
        assert(u.flags == (previous == 0 ? 1u : 0u));
        size_t offset = sizeof(u);
        const auto take = [&](void * destination, size_t bytes) {
            assert(offset + bytes <= packet.size());
            std::memcpy(destination, packet.data() + offset, bytes); offset += bytes;
        };
        const size_t record_bytes = size_t(u.record_groups) * physical_heads * record;
        take(image_rk.data() + size_t(u.record_begin) * physical_heads * record, record_bytes);
        take(image_rv.data() + size_t(u.record_begin) * physical_heads * record, record_bytes);
        for (auto * image : {&image_sk, &image_sv}) {
            if (u.flags) take(image->data(), image->size() * 2);
            else for (uint32_t t = previous; t < u.live_count; ++t) {
                const uint32_t g = (origin + t) / 128;
                const uint32_t slot = g == 0 ? 0 : 1 + (g - 1) % 2;
                take(image->data() + size_t(slot * 128 + (origin + t) % 128) * width, width * 2);
            }
        }
        for (auto * image : {&image_tk, &image_tv}) {
            const uint32_t start = u.flags ? u.live_count - std::min(u.live_count, 128u) : previous;
            for (uint32_t t = start; t < u.live_count; ++t) {
                take(image->data() + size_t(t % 128) * width, width * 2);
            }
        }
        assert(offset == packet.size());
        previous += n;
    }
    assert(previous == count && producer.rows() == count);
    assert(image_sk == expected_sk && image_sv == expected_sv);
    const size_t begin = size_t(origin / 128) * physical_heads * record;
    const size_t committed = size_t(count / 128) * physical_heads * record;
    assert(std::memcmp(image_rk.data(), expected_rk.data() + begin, committed) == 0);
    assert(std::memcmp(image_rv.data(), expected_rv.data() + begin, committed) == 0);
    for (uint32_t t = count - 128; t < count; ++t) for (size_t d = 0; d < width; ++d) {
        assert(image_tk[size_t(t % 128) * width + d] == ggml_fp32_to_fp16(keys[size_t(t) * width + d]));
        assert(image_tv[size_t(t % 128) * width + d] == ggml_fp32_to_fp16(values[size_t(t) * width + d]));
    }
    const auto before = packet;
    int32_t wrong = origin; // a quantized/history bootstrap or rewind is refused
    assert(!producer.append(keys.data(), values.data(), &wrong, 1, false, packet));
    assert(producer.rows() == count && packet == before);
    int32_t next = origin + count;
    keys[0] = std::numeric_limits<float>::quiet_NaN();
    assert(!producer.append(keys.data(), values.data(), &next, 1, false, packet));
    assert(producer.rows() == count && packet == before);
    std::puts("PASS native KVarN4 original-F32 producer, canonical CPU record bytes, wire stage/tail wrap, refused replay/NaN");
}
