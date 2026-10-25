#include "../uwp/rkva/rkva-kvarn-bulk.h"
#include "../uwp/rkva/rkva-range-profile.h"
#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
using namespace rkva_kvarn_bulk;
namespace {
template <class T> void append(std::vector<uint8_t>& data, const T& value) {
    const size_t offset = data.size();
    data.resize(offset + sizeof(T));
    std::memcpy(data.data() + offset, &value, sizeof(T));
}
rkva_kvarn_range_config descriptor(uint32_t tail) {
    return {96256, 34816, RKVA_RANGE_STORAGE_KVARN4,
            3,     128,   128,
            256,   2,     4,
            4,     8960,  3,
            2,     128,   tail,
            1,     1,     1};
}
std::vector<uint8_t> payload(uint32_t begin, uint32_t count, uint32_t tail,
                             uint32_t wire = RKVA_WIRE_F32, bool compact_stage = false) {
    std::vector<uint8_t> data;
    const uint32_t segments = (count + 7) / 8;
    const size_t q_elem_bytes = (wire == RKVA_WIRE_FP8) ? sizeof(uint8_t)
                                : (wire == RKVA_WIRE_F16 || wire == RKVA_WIRE_SPLITZIP)
                                    ? sizeof(uint16_t)
                                    : sizeof(float);
    append(data, rkva_kvarn_range_bulk{1, segments, count, 0});
    for (uint32_t first = 0; first < count; first += 8) {
        const uint32_t n = std::min(8u, count - first);
        const uint32_t previous = begin + first, live = previous + n;
        const bool reset = previous == 0;
        const uint32_t groups = live / 128 - previous / 128;
        const rkva_kvarn_range_upload u{live,
                                        previous / 128,
                                        groups,
                                        (reset ? 1u : 0u) |
                                            (compact_stage ? RKVA_KVARN_UPLOAD_OMIT_STAGE : 0u),
                                        reset ? 0u : previous,
                                        reset ? 0u : n,
                                        {0, 0}};
        const size_t records = size_t(groups) * 4 * 2 * 8960;
        const size_t rows = (compact_stage ? 0 : (reset ? 384 : n)) +
                            (tail ? (reset ? std::min(live, tail) : n) : 0);
        const size_t bytes = 2 * (records + rows * 4 * 256 * 2);
        append(data, rkva_kvarn_range_bulk_segment{uint32_t(sizeof(u) + bytes), n});
        append(data, u);
        data.resize(data.size() + bytes);
        append(data, rkva_range_attn{wire, 0, {0, 0}});
        for (uint32_t row = 0; row < n; ++row)
            append(data, int32_t(96256 + previous + row));
        data.resize(data.size() + size_t(n) * 24 * 256 * q_elem_bytes);
    }
    return data;
}
bool valid(const std::vector<uint8_t>& data, uint32_t begin, uint32_t count, uint32_t tail,
           uint64_t generation = 1, uint32_t wire = RKVA_WIRE_F32) {
    std::vector<Segment> segments;
    return validate_wire(
        data.data(), data.size(), count, descriptor(tail), 24, 4, begin, begin / 128, begin != 0,
        rkva_kvarn_range::coverage_after_reset(begin, tail), generation, wire, segments);
}
void test_f16_helpers() {
    assert(f32_to_f16_bits(0.0f) == 0x0000u);
    assert(f32_to_f16_bits(-0.0f) == 0x8000u);
    assert(f32_to_f16_bits(1.0f) == 0x3c00u);
    assert(f32_to_f16_bits(-1.0f) == 0xbc00u);
    assert(f16_bits_to_f32(0x3c00u) == 1.0f);
    assert(f16_bits_to_f32(0xbc00u) == -1.0f);
    assert(f16_bits_finite(0x3c00u));
    assert(f16_bits_finite(0x7bffu));  // max finite FP16 (65504.0)
    assert(!f16_bits_finite(0x7c00u)); // +Inf
    assert(!f16_bits_finite(0xfc00u)); // -Inf
    assert(!f16_bits_finite(0x7e00u)); // qNaN
    const uint16_t overflow_bits = f32_to_f16_bits(70000.0f);
    assert(!f16_bits_finite(overflow_bits));
    // Subnormal roundtrip check.
    const uint16_t min_sub = 0x0001u;
    const float sub_f32 = f16_bits_to_f32(min_sub);
    assert(std::isfinite(sub_f32) && sub_f32 > 0.0f);
    assert(f32_to_f16_bits(sub_f32) == min_sub);
}
void test_fp8_contract() {
    // 256 saturation patterns: all 128 positive and 128 negative patterns
    // with exponent 30 and mantissa >= 0x380 (0x7b80..0x7bff and 0xfb80..0xfbff)
    // which previously overflowed to Inf must saturate to max finite FP8 E5M2.
    size_t saturated_count = 0;
    for (uint32_t h = 0x7b80u; h <= 0x7bffu; ++h) {
        assert(f16_to_fp8_bits(uint16_t(h)) == 0x7bu);
        ++saturated_count;
    }
    for (uint32_t h = 0xfb80u; h <= 0xfbffu; ++h) {
        assert(f16_to_fp8_bits(uint16_t(h)) == 0xfbu);
        ++saturated_count;
    }
    assert(saturated_count == 256);

    // Clamping boundaries around 57344.0f (0x7b00 in F16).
    assert(f16_to_fp8_bits(0x7b00u) == 0x7bu);
    assert(f16_to_fp8_bits(0xfb00u) == 0xfbu);
    assert(f16_to_fp8_bits(0x7b01u) == 0x7bu);
    assert(f16_to_fp8_bits(0xfb01u) == 0xfbu);
    assert(f16_to_fp8_bits(0x7b7fu) == 0x7bu);
    assert(f16_to_fp8_bits(0xfb7fu) == 0xfbu);

    // Standard normal values.
    assert(f16_to_fp8_bits(0x3c00u) == 0x3cu); // 1.0f
    assert(f16_to_fp8_bits(0xbc00u) == 0xbcu); // -1.0f
    assert(f16_to_fp8_bits(0x0000u) == 0x00u); // +0.0f
    assert(f16_to_fp8_bits(0x8000u) == 0x80u); // -0.0f

    // Subnormals.
    assert(f16_to_fp8_bits(0x0001u) == 0x00u);
    assert(f16_to_fp8_bits(0x8001u) == 0x80u);
    assert(f16_to_fp8_bits(0x007fu) == 0x00u);
    assert(f16_to_fp8_bits(0x0080u) == 0x00u);
    assert(f16_to_fp8_bits(0x8080u) == 0x80u);
    assert(f16_to_fp8_bits(0x0100u) == 0x01u);
    assert(f16_to_fp8_bits(0x8100u) == 0x81u);
    assert(f16_to_fp8_bits(0x0200u) == 0x02u);
    assert(f16_to_fp8_bits(0x8200u) == 0x82u);
    assert(f16_to_fp8_bits(0x0300u) == 0x03u);
    assert(f16_to_fp8_bits(0x8300u) == 0x83u);

    assert(f16_to_fp8_bits(0x0180u) == 0x02u);
    assert(f16_to_fp8_bits(0x0280u) == 0x02u);
    assert(f16_to_fp8_bits(0x0380u) == 0x04u);
    assert(f16_to_fp8_bits(0x3c80u) == 0x3cu);
    assert(f16_to_fp8_bits(0x3d80u) == 0x3eu);
    assert(f16_to_fp8_bits(0xbc80u) == 0xbcu);

    // Infinities and NaNs.
    assert(f16_to_fp8_bits(0x7c00u) == 0x7cu);
    assert(f16_to_fp8_bits(0xfc00u) == 0xfcu);
    const uint8_t qnan_pos = f16_to_fp8_bits(0x7e00u);
    assert((qnan_pos & 0x7cu) == 0x7cu && (qnan_pos & 3u) != 0 && (qnan_pos & 0x80u) == 0);
    const uint8_t qnan_neg = f16_to_fp8_bits(0xfe00u);
    assert((qnan_neg & 0x7cu) == 0x7cu && (qnan_neg & 3u) != 0 && (qnan_neg & 0x80u) == 0x80u);
    const uint8_t nan_low_pos = f16_to_fp8_bits(0x7c01u);
    assert((nan_low_pos & 0x7cu) == 0x7cu && (nan_low_pos & 3u) != 0);
    const uint8_t nan_low_neg = f16_to_fp8_bits(0xfc01u);
    assert((nan_low_neg & 0x7cu) == 0x7cu && (nan_low_neg & 3u) != 0);

    // FP8 -> FP16 mapping.
    for (uint32_t b = 0; b < 256; ++b) {
        assert(fp8_to_f16_bits(uint8_t(b)) == (uint16_t(b) << 8));
    }
    std::puts("fp8 256 saturation patterns and numerical contract: passed");
}
void test_splitzip() {
    std::vector<uint16_t> src(1024);
    for (size_t i = 0; i < src.size(); ++i) {
        float f = std::sin(float(i) * 0.1f) * 2.5f;
        src[i] = f32_to_f16_bits(f);
    }
    std::vector<uint8_t> compressed;
    size_t sz = splitzip_encode(src.data(), src.size(), compressed);
    assert(sz > 0);
    std::vector<uint16_t> decoded(src.size());
    assert(splitzip_decode(compressed.data(), compressed.size(), decoded.data(), decoded.size()));
    for (size_t i = 0; i < src.size(); ++i) {
        assert(src[i] == decoded[i]);
    }

    // Regression: test all truncated prefixes under AddressSanitizer.
    for (size_t len = 0; len < compressed.size(); ++len) {
        std::vector<uint16_t> out(src.size());
        assert(!splitzip_decode(compressed.data(), len, out.data(), src.size()));
    }

    // Regression: crafted truncated header claiming elements without buffer data.
    rkva_splitzip_header hdr_trunc{};
    hdr_trunc.n_elements = 1000;
    hdr_trunc.n_escapes = 0;
    hdr_trunc.compressed_bytes = sizeof(hdr_trunc);
    std::vector<uint8_t> trunc_buf(sizeof(hdr_trunc));
    std::memcpy(trunc_buf.data(), &hdr_trunc, sizeof(hdr_trunc));
    std::vector<uint16_t> dst1000(1000);
    assert(!splitzip_decode(trunc_buf.data(), trunc_buf.size(), dst1000.data(), 1000));

    // Inconsistent / corrupted header fields:
    auto corrupt_escapes = compressed;
    auto* hdr_e = reinterpret_cast<rkva_splitzip_header*>(corrupt_escapes.data());
    hdr_e->n_escapes = 0xffffffffu;
    assert(!splitzip_decode(corrupt_escapes.data(), corrupt_escapes.size(), decoded.data(),
                            decoded.size()));

    auto corrupt_sz = compressed;
    auto* hdr_s = reinterpret_cast<rkva_splitzip_header*>(corrupt_sz.data());
    hdr_s->compressed_bytes = uint32_t(corrupt_sz.size() - 1);
    assert(!splitzip_decode(corrupt_sz.data(), corrupt_sz.size(), decoded.data(), decoded.size()));
    hdr_s->compressed_bytes = uint32_t(corrupt_sz.size() + 10);
    assert(!splitzip_decode(corrupt_sz.data(), corrupt_sz.size(), decoded.data(), decoded.size()));

    // Element count mismatch:
    assert(
        !splitzip_decode(compressed.data(), compressed.size(), decoded.data(), decoded.size() - 1));
    assert(
        !splitzip_decode(compressed.data(), compressed.size(), decoded.data(), decoded.size() + 1));

    // Null pointer checks:
    assert(!splitzip_decode(nullptr, compressed.size(), decoded.data(), decoded.size()));
    assert(!splitzip_decode(compressed.data(), compressed.size(), nullptr, decoded.size()));

    // Byte-oriented input need not have native struct alignment.
    std::vector<uint8_t> unaligned(compressed.size() + 1);
    std::memcpy(unaligned.data() + 1, compressed.data(), compressed.size());
    assert(
        splitzip_decode(unaligned.data() + 1, compressed.size(), decoded.data(), decoded.size()));
    assert(decoded == src);

    for (size_t count : {size_t(0), size_t(1), size_t(3), size_t(65536)}) {
        std::vector<uint16_t> input(count), output(count);
        for (size_t i = 0; i < count; ++i)
            input[i] = uint16_t(i);
        std::vector<uint8_t> frame;
        splitzip_encode(input.data(), count, frame);
        assert(splitzip_decode(frame.data(), frame.size(), output.data(), count));
        assert(input == output);
    }
    std::puts("splitzip bounds, truncation and corruption safety: passed");
}
void test_compact_stage() {
    using namespace rkva_kvarn_range;
    for (uint32_t begin : {0u, 1u, 120u, 127u, 128u, 255u, 8191u, 16128u}) {
        for (uint32_t count : {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 256u}) {
            for (uint32_t wire :
                 {uint32_t(RKVA_WIRE_F32), uint32_t(RKVA_WIRE_F16), uint32_t(RKVA_WIRE_FP8)}) {
                auto compact = payload(begin, count, 128, wire, true);
                auto full = payload(begin, count, 128, wire);
                assert(valid(compact, begin, count, 128, 1, wire));
                const size_t omitted_rows = begin ? count : 384u + count - std::min(count, 8u);
                assert(full.size() - compact.size() == omitted_rows * 4u * 256u * 2u * 2u);
                compact.pop_back();
                assert(!valid(compact, begin, count, 128, 1, wire));
            }
        }
    }
    // The omitted stage cannot serve tail=0, revision 1/2 or the global sink.
    auto d = descriptor(128);
    assert(compact_stage_supported(d));
    d.origin = 0;
    assert(!compact_stage_supported(d));
    d = descriptor(128);
    d.layout_version = 2;
    assert(!compact_stage_supported(d));
    assert(!compact_stage_supported(descriptor(0)));
    auto data = payload(127, 9, 0, RKVA_WIRE_FP8, true);
    assert(!valid(data, 127, 9, 0, 1, RKVA_WIRE_FP8));
    rkva_kvarn_range_upload u{8, 0, 0, 3, 0, 0, {0, 0}};
    assert(!valid_upload(u, 512, 0, 0, false)); // old/default admission rejects flag
    assert(valid_upload(u, 512, 0, 0, false, true));
    u.flags |= 4;
    assert(!valid_upload(u, 512, 0, 0, false, true));
    std::puts("compact exact-tail upload admission, seal edges and wire byte savings: passed");
}
void test_kvarn4_mq_split() {
    using namespace rkva_rprof;
    assert(kvarn4_mq_split(0, kModeOptBalanced, 4).split_k == 1);
    assert(kvarn4_mq_split(0, kModeOptBalanced, 4).split_rows == 0);
    assert(kvarn4_mq_split(0, kModeOptFine, 2).split_k == 1);
    assert(kvarn4_mq_split(0, kModeOptFine, 2).split_rows == 0);

    for (uint32_t q_tile : {2u, 4u}) {
        for (uint32_t m = 1; m < kModeCount; ++m) {
            for (uint32_t a :
                 {0u, 1u, 127u, 128u, 129u, 1000u, 6144u, 16384u, 24576u, 34816u, 131072u}) {
                const KvarnSplit ks = kvarn4_mq_split(a, m, q_tile);
                assert(ks.split_k >= 1 && ks.split_k <= kOptMaxSplits);
                assert(ks.split_rows % kKvarnTile == 0);
                assert(uint64_t(ks.split_rows) * ks.split_k >= a);
                assert(a == 0 || uint64_t(ks.split_rows) * (ks.split_k - 1) < a);
            }
        }
        // 6k (6144): target_rows=256 -> 24 splits of 256 tokens for both modes
        assert(kvarn4_mq_split(6144, kModeOptBalanced, q_tile).split_k == 24);
        assert(kvarn4_mq_split(6144, kModeOptBalanced, q_tile).split_rows == 256);
        assert(kvarn4_mq_split(6144, kModeOptFine, q_tile).split_k == 24);
        assert(kvarn4_mq_split(6144, kModeOptFine, q_tile).split_rows == 256);

        // 16k (16384 - 112k context point):
        // Balanced (max 32 splits): 32 splits of 512 tokens
        assert(kvarn4_mq_split(16384, kModeOptBalanced, q_tile).split_k == 32);
        assert(kvarn4_mq_split(16384, kModeOptBalanced, q_tile).split_rows == 512);
        // Fine (max 64 splits): 64 splits of 256 tokens, down from legacy fine 256 splits
        assert(kvarn4_mq_split(16384, kModeOptFine, q_tile).split_k == 64);
        assert(kvarn4_mq_split(16384, kModeOptFine, q_tile).split_rows == 256);

        // 35k (34816):
        assert(kvarn4_mq_split(34816, kModeOptBalanced, q_tile).split_k == 32);
        assert(kvarn4_mq_split(34816, kModeOptBalanced, q_tile).split_rows == 1088);
        assert(kvarn4_mq_split(34816, kModeOptFine, q_tile).split_k == 64);
        assert(kvarn4_mq_split(34816, kModeOptFine, q_tile).split_rows == 544);
    }

    // Invalid q_tile or coarse mode falls back to kvarn4_split
    assert(kvarn4_mq_split(16384, kModeOptBalanced, 0).split_k ==
           kvarn4_split(16384, kModeOptBalanced).split_k);
    assert(kvarn4_mq_split(16384, kModeOptBalanced, 1).split_k ==
           kvarn4_split(16384, kModeOptBalanced).split_k);
    assert(kvarn4_mq_split(16384, kModeOptBalanced, 3).split_k ==
           kvarn4_split(16384, kModeOptBalanced).split_k);
    assert(kvarn4_mq_split(16384, kModeOptCoarse, 4).split_k ==
           kvarn4_split(16384, kModeOptCoarse).split_k);
    std::puts("kvarn4_mq_split policy invariants: passed");
}
} // namespace
int main() {
    test_f16_helpers();
    test_fp8_contract();
    test_splitzip();
    test_kvarn4_mq_split();
    test_compact_stage();
    for (uint32_t count : {1u, 8u, 256u}) {
        auto data = payload(0, count, 128, RKVA_WIRE_SPLITZIP);
        uint32_t wire = 99;
        assert(!peek_wire(data.data(), data.size(), &wire));
        assert(!valid(data, 0, count, 128, 1, RKVA_WIRE_SPLITZIP));
    }
    // Every segment's own prefix image remains authoritative across two seals,
    // tail wrap, arbitrary unaligned live reuse, and maximum b/ub=256 for F32,
    // FP16 and FP8 wire formats. SplitZip remains archived, never admitted.
    for (uint32_t tail : {0u, 128u}) {
        for (uint32_t wire :
             {uint32_t(RKVA_WIRE_F32), uint32_t(RKVA_WIRE_F16), uint32_t(RKVA_WIRE_FP8)}) {
            for (uint32_t begin : {0u, 1u, 120u, 127u, 128u, 255u, 8191u}) {
                for (uint32_t count : {1u, 8u, 9u, 64u, 256u}) {
                    auto data = payload(begin, count, tail, wire);
                    uint32_t peeked = 99u;
                    assert(peek_wire(data.data(), data.size(), &peeked) && peeked == wire);
                    assert(valid(data, begin, count, tail, 1, wire));
                    // Wrong expected wire must fail closed.
                    const uint32_t other_wire =
                        (wire == RKVA_WIRE_F16) ? RKVA_WIRE_F32 : RKVA_WIRE_F16;
                    assert(!valid(data, begin, count, tail, 1, other_wire));
                    auto truncated = data;
                    truncated.pop_back();
                    assert(!valid(truncated, begin, count, tail, 1, wire));
                    data.push_back(0);
                    assert(!valid(data, begin, count, tail, 1, wire));
                }
            }
        }
        auto data = payload(120, 256, tail, RKVA_WIRE_F32);
        std::vector<Segment> segments;
        assert(validate(data.data(), data.size(), 256, descriptor(tail), 24, 4, 120, 0, true,
                        rkva_kvarn_range::coverage_after_reset(120, tail), 1, segments));
        // Last-segment corruption is rejected before any first-segment writes.
        int32_t bad_position = -1;
        std::memcpy(data.data() + segments.back().positions_offset, &bad_position, 4);
        assert(!valid(data, 120, 256, tail));
        data = payload(120, 256, tail, RKVA_WIRE_F32);
        float nan = std::numeric_limits<float>::quiet_NaN();
        std::memcpy(data.data() + segments.back().query_offset, &nan, 4);
        assert(!valid(data, 120, 256, tail));
        data = payload(120, 256, tail, RKVA_WIRE_F32);
        assert(!valid(data, 120, 256, tail, UINT64_MAX - 31));
        auto wrong_revision = descriptor(tail);
        wrong_revision.layout_version = 2;
        assert(!validate(data.data(), data.size(), 256, wrong_revision, 24, 4, 120, 0, true,
                         rkva_kvarn_range::coverage_after_reset(120, tail), 1, segments));

        // FP16 bulk admission: NaN/Inf in any segment Q or mixed segment wire is rejected.
        auto data_f16 = payload(120, 256, tail, RKVA_WIRE_F16);
        std::vector<Segment> segs_f16;
        assert(validate_wire(data_f16.data(), data_f16.size(), 256, descriptor(tail), 24, 4, 120, 0,
                             true, rkva_kvarn_range::coverage_after_reset(120, tail), 1,
                             RKVA_WIRE_F16, segs_f16));
        for (uint16_t bad_f16 : {uint16_t(0x7c00u), uint16_t(0xfc00u), uint16_t(0x7e00u)}) {
            auto bad_q = data_f16;
            std::memcpy(bad_q.data() + segs_f16.back().query_offset, &bad_f16, sizeof(bad_f16));
            assert(!valid(bad_q, 120, 256, tail, 1, RKVA_WIRE_F16));
        }
        // Flip the last segment's rkva_range_attn.wire from F16 to F32: must reject mixed wire.
        auto mixed_wire = data_f16;
        const uint32_t wire_f32 = RKVA_WIRE_F32;
        std::memcpy(mixed_wire.data() + segs_f16.back().positions_offset - sizeof(rkva_range_attn),
                    &wire_f32, sizeof(wire_f32));
        assert(!valid(mixed_wire, 120, 256, tail, 1, RKVA_WIRE_F16));
    }
    std::puts("native bulk envelope/coverage/f16 admission: passed");
}
