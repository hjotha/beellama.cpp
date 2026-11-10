// CPU contracts only. This does not qualify D3D12, DXIL or model quality.
#include "ggml.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-remote-attn/rkva-low4-producer.h"
#include "../tools/xbox/uwp/rkva/rkva-kvarn-range.h"
#include "../tools/xbox/uwp/rkva/rkva-kvarn-bulk.h"
#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <limits>
#include <random>

static void test_format(ggml_type type) {
    const uint32_t storage = type == GGML_TYPE_Q4_1 ? RKVA_RANGE_STORAGE_Q4_1 : RKVA_RANGE_STORAGE_IQ4_NL;
    const uint32_t layout = type == GGML_TYPE_Q4_1 ? RKVA_LAYOUT_Q4_1 : RKVA_LAYOUT_IQ4_NL;
    const uint32_t head_bytes = rkva_low4_producer::record_bytes(type);
    rkva_kvarn_range_config d{256, 512, storage, layout, 1, 0, 256, 2, 4, 4, head_bytes, 0, 0, 0, 0, 0, 0, 1};
    assert(rkva_kvarn_range::valid_low4_descriptor(d, 256));
    for (size_t field = 0; field < sizeof(d) / 4; ++field) {
        auto bad = d; uint32_t word;
        std::memcpy(&word, reinterpret_cast<uint8_t*>(&bad) + field * 4, 4);
        word = UINT32_MAX; std::memcpy(reinterpret_cast<uint8_t*>(&bad) + field * 4, &word, 4);
        assert(!rkva_kvarn_range::valid_low4_descriptor(bad, 256));
    }
    auto alias = d; alias.storage = RKVA_RANGE_STORAGE_Q4_0; alias.layout_version = 4;
    assert(!rkva_kvarn_range::valid_low4_descriptor(alias, 256));
    assert(!rkva_kvarn_range::valid_q4_descriptor(d, 256));
    assert(!rkva_kvarn_range::valid_low4_descriptor(d, 128));
    std::mt19937 rng(0x1a41);
    std::uniform_real_distribution<float> dist(-17.0f, 9.0f);
    for (uint32_t heads : {1u, 4u}) for (uint32_t count : {1u, 2u, 9u, 128u, 256u}) {
        rkva_low4_producer producer(256, 512, heads, type);
        const size_t n = size_t(count) * heads * 256;
        std::vector<float> k(n), v(n);
        for (size_t i = 0; i < n; ++i) { k[i] = dist(rng); v[i] = std::sin(float(i) * .017f) + 3.0f; }
        // Exercise nonzero Q4_1 minima and IQ4_NL codebook, not just zero blocks.
        if (count > 1) for (size_t i = 0; i < 256; ++i) k[i] = -2.75f;
        std::vector<int32_t> positions(count); for (uint32_t i = 0; i < count; ++i) positions[i] = 256 + i;
        std::vector<uint8_t> payload;
        assert(producer.prepare(k.data(), v.data(), positions.data(), count));
        assert(producer.append(k.data(), v.data(), positions.data(), count, true, payload));
        assert(producer.rows() == count);
        rkva_kvarn_range_upload u; std::memcpy(&u, payload.data(), sizeof(u));
        assert(rkva_kvarn_range::valid_low4_upload(u, 512, 0));
        const size_t plane = size_t(count) * heads * head_bytes;
        assert(payload.size() == sizeof(u) + 2 * plane);
        std::vector<uint8_t> expected(plane);
        const uint8_t* data = payload.data() + sizeof(u);
        assert(rkva_kvarn_range::valid_low4_planes(storage, data, data + plane, size_t(count) * heads));
        for (const float* input : {k.data(), v.data()}) {
            assert(ggml_quantize_chunk(type, input, expected.data(), 0, int64_t(count) * heads, 256, nullptr) == plane);
            assert(std::memcmp(expected.data(), data, plane) == 0);
#ifdef RKVA_LOW4_SHADER_HOST
            // Compile and execute the actual scalar HLSL loaders through a tiny
            // ByteAddressBuffer shim. This checks offsets/table/packing only.
            std::vector<float> decoded(n);
            ggml_get_type_traits(type)->to_float(data, decoded.data(), n);
            ByteAddressBuffer buffer{data, plane};
            for (size_t row = 0; row < size_t(count) * heads; ++row) for (uint32_t q = 0; q < 64; ++q) {
                const auto out = type == GGML_TYPE_Q4_1 ? load_q4_1(buffer, row * head_bytes, q) : load_iq4_nl(buffer, row * head_bytes, q);
                const auto base = type == GGML_TYPE_Q4_1 ? quarter_dim_q4_1(q) : quarter_dim_iq4_nl(q);
                for (size_t j = 0; j < 4; ++j) assert(out[j] == decoded[row * 256 + base + j]);
            }
#endif
            data += plane;
        }
        auto corrupted = payload;
        corrupted[sizeof(u)] = 0; corrupted[sizeof(u) + 1] = 0x7c;
        assert(!rkva_kvarn_range::valid_low4_planes(storage, corrupted.data() + sizeof(u),
                corrupted.data() + sizeof(u) + plane, size_t(count) * heads));
        if (type == GGML_TYPE_Q4_1) {
            corrupted = payload; corrupted[sizeof(u) + 2] = 1; corrupted[sizeof(u) + 3] = 0x7c;
            assert(!rkva_kvarn_range::valid_low4_planes(storage, corrupted.data() + sizeof(u),
                    corrupted.data() + sizeof(u) + plane, size_t(count) * heads));
        }
        auto bad = u; bad.append_rows = UINT32_MAX;
        assert(!rkva_kvarn_range::valid_low4_upload(bad, 512, 0));
        bad = u; bad.reserved[1] = 1; assert(!rkva_kvarn_range::valid_low4_upload(bad, 512, 0));
        assert(!producer.rewind(count + 1));
        assert(producer.rewind(0));
        const auto saved = payload;
        k[0] = std::numeric_limits<float>::infinity();
        assert(!producer.append(k.data(), v.data(), positions.data(), count, true, payload));
        assert(payload == saved && producer.rows() == 0);
        k[0] = std::numeric_limits<float>::quiet_NaN();
        assert(!producer.append(k.data(), v.data(), positions.data(), count, true, payload));
        k[0] = 1e30f;
        assert(!producer.append(k.data(), v.data(), positions.data(), count, true, payload));
        k[0] = 0;
        positions[0]++;
        assert(!producer.append(k.data(), v.data(), positions.data(), count, true, payload));
        positions[0]--;
        assert(producer.append(k.data(), v.data(), positions.data(), count, true, payload));
        assert(producer.rewind(count - 1));
        const int32_t last = int32_t(256 + count - 1);
        assert(producer.append(k.data(), v.data(), &last, 1, false, payload));
        std::memcpy(&u, payload.data(), sizeof(u));
        assert(rkva_kvarn_range::valid_low4_upload(u, 512, count - 1));
        assert(u.live_count == count && u.append_rows == 1 && u.append_begin == count - 1);
        assert(!producer.append(k.data(), v.data(), &last, 1, true, payload));
    }
    std::printf("PASS: %s native producer, descriptors, metadata, rollback%s\n", ggml_type_name(type),
#ifdef RKVA_LOW4_SHADER_HOST
        ", actual HLSL scalar loader parity"
#else
        ""
#endif
    );
}
static void test_wire_validation() {
    int32_t pos = 256;
    for (uint32_t wire : {uint32_t(RKVA_WIRE_FP8), uint32_t(RKVA_WIRE_F16), uint32_t(RKVA_WIRE_F32)}) {
        const size_t width = wire == RKVA_WIRE_FP8 ? 1 : wire == RKVA_WIRE_F16 ? 2 : 4;
        std::vector<uint8_t> q(256 * width);
        const auto valid = [&] { return rkva_kvarn_range::valid_low4_queries(wire, q.data(), q.size(),
                    reinterpret_cast<const uint8_t*>(&pos), 1, 1); };
        assert(valid());
        assert(!rkva_kvarn_range::valid_low4_queries(wire, q.data(), q.size()-1,
                    reinterpret_cast<const uint8_t*>(&pos), 1, 1));
        pos = -1; assert(!valid()); pos = 256;
        if (width == 1) q[0] = 0x7c;
        else if (width == 2) { q[0] = 0; q[1] = 0x7c; }
        else { float f = std::numeric_limits<float>::quiet_NaN(); std::memcpy(q.data(), &f, 4); }
        assert(!valid());
    }
    std::vector<float> out(256, 1.f); float lse = 0;
    const auto valid_out = [&] { return rkva_kvarn_bulk::turboquant_nc_output_valid(
        reinterpret_cast<const uint8_t*>(out.data()), reinterpret_cast<const uint8_t*>(&lse),
        reinterpret_cast<const uint8_t*>(&pos), 1, 1, 256, 1, true); };
    assert(valid_out());
    out.back() = std::numeric_limits<float>::quiet_NaN(); assert(!valid_out()); out.back() = 1;
    lse = std::numeric_limits<float>::infinity(); assert(!valid_out());
    lse = -std::numeric_limits<float>::infinity(); assert(!valid_out());
    pos = 255; assert(valid_out()); // empty causal range may have LSE -inf
    std::puts("PASS: native low4 query/position/truncation and CPU output finite guards");
}

static void test_kvarn_vs_qx_separation() {
    // 1. Verify all 6 KVarN 2D WHT + Sinkhorn bit widths (2, 3, 4, 5, 6, 8)
    for (uint32_t bits : {2u, 3u, 4u, 5u, 6u, 8u}) {
        const uint32_t rec_bytes = rkva_kvarn_range::kvarn_record_bytes(bits);
        assert(rec_bytes == bits * 2048u + 768u);
        rkva_kvarn_range_config kd{256, 512, RKVA_RANGE_STORAGE_KVARN4, 3, 128, 128, 256, 2,
                                   bits, bits, rec_bytes, 3, 2, 128, 128, 1, 1, 1};
        assert(rkva_kvarn_range::valid_descriptor(kd, 256));
        assert(!rkva_kvarn_range::valid_q8_descriptor(kd, 256));
        assert(!rkva_kvarn_range::valid_q4_descriptor(kd, 256));
        assert(!rkva_kvarn_range::valid_extended_native_descriptor(kd, 256));
        auto tail_boundary = kd;
        tail_boundary.capacity = 131072;
        tail_boundary.precision_tail_tokens = 65535;
        assert(rkva_kvarn_range::valid_descriptor(tail_boundary, 256));
        tail_boundary.precision_tail_tokens = 65536;
        assert(!rkva_kvarn_range::valid_descriptor(tail_boundary, 256));
        auto bad_rec = kd;
        bad_rec.record_bytes = 272; // q8_0 head_bytes must NOT be accepted as kvarn8!
        assert(!rkva_kvarn_range::valid_descriptor(bad_rec, 256));
    }
    // 2. Verify Q2_0S (80 B/head, layout 20) vs OSCAR_INT2 (96 B/head, layout 7) vs KVarN2 (4864 B/rec, layout 3)
    {
        rkva_low4_producer prod_q2_0s(256, 512, 2, GGML_TYPE_Q2_0S, false);
        rkva_low4_producer prod_oscar(256, 512, 2, GGML_TYPE_Q2_0S, true);
        assert(rkva_low4_producer::record_bytes(GGML_TYPE_Q2_0S, false) == 80u);
        assert(rkva_low4_producer::record_bytes(GGML_TYPE_Q2_0S, true) == 96u);
        assert(prod_q2_0s.row_bytes() == 2u * 80u);
        assert(prod_oscar.row_bytes() == 2u * 96u);
        rkva_kvarn_range_config d_q2_0s{256, 512, RKVA_RANGE_STORAGE_Q2_0S, RKVA_LAYOUT_Q2_0S, 1, 0, 256, 2, 2, 2, 80, 0, 0, 0, 0, 0, 0, 1};
        rkva_kvarn_range_config d_oscar{256, 512, RKVA_RANGE_STORAGE_OSCAR_INT2, 7, 1, 0, 256, 2, 2, 2, 96, 0, 0, 0, 0, 0, 1, 1};
        assert(rkva_kvarn_range::valid_extended_native_descriptor(d_q2_0s, 256));
        assert(!rkva_kvarn_range::valid_oscar_int2_descriptor(d_q2_0s, 256));
        assert(rkva_kvarn_range::valid_oscar_int2_descriptor(d_oscar, 256));
        assert(!rkva_kvarn_range::valid_extended_native_descriptor(d_oscar, 256));
    }
    std::puts("PASS: KVarN 2D (kvarn2/3/4/5/6/8) vs GGML 1D (q2_0s/q3_0/q4_0/q5_0/q6_0/q8_0) vs oscar_int2 strict separation");
}

static void test_extended_producer_bytes() {
    // Compare actual upload bytes to the canonical GGML encoder, with nonzero,
    // mixed-sign data and token/head boundaries. This is CPU codec coverage.
    for (auto type : {GGML_TYPE_Q2_0S, GGML_TYPE_Q2_1, GGML_TYPE_Q3_0, GGML_TYPE_Q3_1,
                      GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q6_0, GGML_TYPE_Q6_1,
                      GGML_TYPE_BF16, GGML_TYPE_F32}) {
        for (uint32_t heads : {1u, 3u}) for (uint32_t count : {1u, 2u, 127u, 128u, 129u}) {
            rkva_low4_producer producer(256, 512, heads, type);
            const size_t n = size_t(count) * heads * 256;
            std::vector<float> k(n), v(n);
            for (size_t i = 0; i < n; ++i) {
                k[i] = std::sin(float(i) * .017f) * 17.f - 2.f;
                v[i] = std::cos(float(i) * .021f) * 9.f + 1.f;
            }
            std::vector<int32_t> positions(count);
            for (uint32_t i = 0; i < count; ++i) positions[i] = 256 + i;
            std::vector<uint8_t> packet;
            assert(producer.prepare(k.data(), v.data(), positions.data(), count));
            assert(producer.append(k.data(), v.data(), positions.data(), count, true, packet));
            const size_t plane = size_t(count) * producer.row_bytes();
            assert(packet.size() == sizeof(rkva_kvarn_range_upload) + 2 * plane);
            size_t offset = sizeof(rkva_kvarn_range_upload);
            for (const auto * input : {k.data(), v.data()}) {
                std::vector<uint8_t> expected(plane);
                assert(ggml_quantize_chunk(type, input, expected.data(), 0, count * heads, 256, nullptr) == plane);
                assert(std::memcmp(packet.data() + offset, expected.data(), plane) == 0);
                offset += plane;
            }
            assert(producer.rewind(0));
            std::vector<uint8_t> repeated;
            assert(producer.append(k.data(), v.data(), positions.data(), count, true, repeated));
            assert(packet == repeated);
        }
    }
    std::puts("PASS: 100 extended-format byte-exact producer/rewind cases");
}

int main() {
    ggml_backend_load_all();
    ggml_context * init = ggml_init({1u << 16, nullptr, true}); assert(init);
    test_extended_producer_bytes();
    test_wire_validation();
    test_format(GGML_TYPE_Q4_1); test_format(GGML_TYPE_IQ4_NL);
    test_kvarn_vs_qx_separation();
    ggml_free(init);
}
