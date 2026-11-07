#include "../uwp/rkva/rkva-kvarn-bulk.h"
#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <limits>

using namespace rkva_kvarn_bulk;
template<class T> static size_t append(std::vector<uint8_t>& bytes, const T& value) {
    const size_t offset = bytes.size();
    bytes.resize(offset + sizeof(value));
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
    return offset;
}
template<class T> static void put(std::vector<uint8_t>& bytes, size_t offset, const T& value) {
    assert(offset + sizeof(value) <= bytes.size());
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

static void envelope_tests(uint32_t storage, uint32_t layout, size_t row) {
    constexpr uint32_t origin = 256, capacity = 256, heads = 6;
    const uint64_t cap = rkva_e4_layout_capability(storage, layout);
    assert(cap && rkva_e4_layout_allowed(cap, storage, layout));
    assert(!rkva_e4_layout_allowed(RKVA_CAP_KVARN_BULK_E4M3FN, storage, layout));
    assert(!rkva_e4_layout_allowed(~uint64_t(0), storage, layout + 1));
    std::vector<uint8_t> request;
    append(request, rkva_kvarn_range_bulk{1, 2, 2, 0});
    size_t segments[2], uploads[2], planes[2], attention[2], positions[2], query[2];
    for (uint32_t i = 0; i < 2; ++i) {
        segments[i] = append(request, rkva_kvarn_range_bulk_segment{uint32_t(sizeof(rkva_kvarn_range_upload) + 2 * row), 1});
        uploads[i] = append(request, rkva_kvarn_range_upload{i + 1, 0, 0, i == 0 ? RKVA_KVARN_UPLOAD_RESET : 0u, i, 1, {0, 0}});
        planes[i] = request.size(); request.resize(request.size() + 2 * row, 0);
        attention[i] = append(request, rkva_range_attn{RKVA_WIRE_FP8_E4M3FN, 0, {0, 0}});
        positions[i] = append(request, int32_t(origin + i));
        query[i] = request.size(); request.resize(request.size() + heads * 256, 0x38);
    }
    const auto valid = [&](const std::vector<uint8_t>& bytes, uint32_t live = 0) {
        return validate_native_e4(bytes.data(), bytes.size(), 2, storage, heads, 1, capacity, live, true);
    };
    assert(valid(request));
    // Every truncated prefix is rejected; a late malformed segment cannot admit a prefix.
    for (size_t length = 0; length < request.size(); ++length)
        assert(!validate_native_e4(request.data(), length, 2, storage, heads, 1, capacity, 0, true));
    auto bad = request; bad.push_back(0); assert(!valid(bad));
    for (size_t pos : {query[0], query[1]}) {
        for (uint8_t nan : {uint8_t(0x7f), uint8_t(0xff)}) { bad = request; bad[pos] = nan; assert(!valid(bad)); }
        for (uint8_t finite : {uint8_t(0x7e), uint8_t(0xfe), uint8_t(0x80), uint8_t(1)}) { bad = request; bad[pos] = finite; assert(valid(bad)); }
    }
    bad = request; put(bad, positions[1], int32_t(-1)); assert(!valid(bad));
    for (uint32_t wire : {uint32_t(RKVA_WIRE_FP8), uint32_t(RKVA_WIRE_F16), uint32_t(RKVA_WIRE_F32), 99u}) {
        bad = request; put(bad, attention[1], rkva_range_attn{wire, 0, {0, 0}}); assert(!valid(bad));
    }
    bad = request; put(bad, attention[1], rkva_range_attn{RKVA_WIRE_FP8_E4M3FN, 1, {0, 0}}); assert(!valid(bad));
    bad = request; put(bad, segments[1], rkva_kvarn_range_bulk_segment{uint32_t(sizeof(rkva_kvarn_range_upload) + 2 * row - 1), 1}); assert(!valid(bad));
    bad = request; put(bad, uploads[1], rkva_kvarn_range_upload{2, 0, 0, RKVA_KVARN_UPLOAD_RESET, 0, 2, {0, 0}}); assert(!valid(bad));
    bad = request; put(bad, uploads[1], rkva_kvarn_range_upload{3, 0, 0, 0, 2, 1, {0, 0}}); assert(!valid(bad));
    bad = request; put(bad, uploads[0], rkva_kvarn_range_upload{6, 0, 0, 0, 5, 1, {0, 0}});
    put(bad, uploads[1], rkva_kvarn_range_upload{7, 0, 0, 0, 6, 1, {0, 0}});
    assert(valid(bad, 5)); assert(!valid(bad));
    // Invalid metadata in either plane of the last segment fails preflight.
    for (size_t plane : {planes[1], planes[1] + row}) {
        bad = request;
        if (storage == RKVA_RANGE_STORAGE_Q4_0 || rkva_kvarn_range::low4_storage(storage)) put(bad, plane, uint16_t(0x7c00));
        else if (storage == RKVA_RANGE_STORAGE_SAW_INT4) put(bad, plane + 128, std::numeric_limits<float>::infinity());
        else put(bad, plane + 128, uint16_t(0x7c00));
        assert(!valid(bad));
    }
    if (storage == RKVA_RANGE_STORAGE_SAW_INT4) {
        bad = request; put(bad, planes[1] + 128, std::numeric_limits<float>::max());
        put(bad, planes[1] + 132, -2.0f); assert(!valid(bad));
    }
    if (storage == RKVA_RANGE_STORAGE_TURBOQUANT_NC) {
        bad = request; bad[planes[1] + 130] = 1; assert(!valid(bad));
    }
    // All 256 queries, not the separate KVarN single-segment <=9 limit.
    std::vector<uint8_t> large;
    append(large, rkva_kvarn_range_bulk{1, 1, 256, 0});
    append(large, rkva_kvarn_range_bulk_segment{uint32_t(sizeof(rkva_kvarn_range_upload) + 512 * row), 256});
    append(large, rkva_kvarn_range_upload{256, 0, 0, RKVA_KVARN_UPLOAD_RESET, 0, 256, {0, 0}});
    large.resize(large.size() + 512 * row, 0);
    append(large, rkva_range_attn{RKVA_WIRE_FP8_E4M3FN, 0, {0, 0}});
    for (int32_t q = 0; q < 256; ++q) append(large, int32_t(origin + q));
    large.resize(large.size() + 256 * heads * 256, 0x38);
    assert(validate_native_e4(large.data(), large.size(), 256, storage, heads, 1, capacity, 0, true));
}

static void output_tests() {
    constexpr uint32_t nq = 256, heads = 6, dim = 256, origin = 256;
    constexpr size_t width = heads * dim;
    std::vector<float> out(nq * width, 1.0f), lse(nq * heads, 1.0f);
    std::vector<int32_t> positions(nq, origin);
    positions[0] = origin - 1;
    for (size_t e = 0; e < width; ++e) out[e] = -0.0f;
    for (size_t h = 0; h < heads; ++h) lse[h] = -std::numeric_limits<float>::infinity();
    const float values[] = {600.0f, -600.0f, 1.0625f, std::nextafter(1.0625f, 2.0f), std::ldexp(1.0f, -10), std::nextafter(std::ldexp(1.0f, -10), 1.0f)};
    std::copy(std::begin(values), std::end(values), out.begin() + width);
    std::vector<uint8_t> reply;
    const auto pack = [&] { return native_e4_pack_reply(out.data(), lse.data(), positions.data(), nq, heads, dim, origin, reply); };
    assert(pack());
    assert(reply[0] == 0x80);
    const uint8_t expected[] = {0x7e, 0xfe, 0x38, 0x39, 0x00, 0x01};
    assert(std::memcmp(reply.data() + width, expected, sizeof(expected)) == 0);
    assert(ggml_remote_attn_policy::e4_reply_valid(reply.data(), reply.size(), nq, width, heads, positions.data(), origin));
    const auto good = reply;
    for (float invalid : {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        out.back() = invalid; assert(!pack() && reply == good); out.back() = 1.0f;
        lse.back() = invalid; assert(!pack() && reply == good); lse.back() = 1.0f;
    }
    out[0] = 1.0f; assert(!pack() && reply == good); out[0] = 0.0f;
    lse[0] = 0.0f; assert(!pack() && reply == good);
}
int main() {
    envelope_tests(RKVA_RANGE_STORAGE_Q4_0, 4, 144);
    envelope_tests(RKVA_RANGE_STORAGE_SAW_INT4, 6, 136);
    envelope_tests(RKVA_RANGE_STORAGE_TURBOQUANT_NC, 8, 132);
    envelope_tests(RKVA_RANGE_STORAGE_Q4_1, RKVA_LAYOUT_Q4_1, 160);
    envelope_tests(RKVA_RANGE_STORAGE_IQ4_NL, RKVA_LAYOUT_IQ4_NL, 144);
    assert(!rkva_e4_layout_allowed(~uint64_t(0), RKVA_RANGE_STORAGE_Q8_0, 5));
    assert(!rkva_e4_layout_allowed(~uint64_t(0), RKVA_RANGE_STORAGE_Q4_0 | RKVA_RANGE_STORAGE_SAW_INT4, 4));
    output_tests();
    std::puts("Native E4: capabilities, full envelope preflight, metadata, RNE, empty prefixes and N256 output passed");
}
