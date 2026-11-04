#include "../uwp/rkva/rkva-kvarn-bulk.h"
#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

template<class T> static void append(std::vector<uint8_t> & out, const T & value) {
    const size_t offset = out.size();
    out.resize(offset + sizeof(T));
    std::memcpy(out.data() + offset, &value, sizeof(T));
}

int main() {
    using namespace ggml_remote_attn_policy;
    assert(!e4_requested(nullptr, nullptr, nullptr));
    assert(e4_requested("fp8_e4m3fn", nullptr, nullptr));
    assert(e4_requested(nullptr, "fp8_e4m3fn", nullptr));
    assert(e4_requested(nullptr, nullptr, "fp8_e4m3fn"));
    assert(!e4_requested("fp8", "fp8_e4m3fn", "fp8_e4m3fn"));
    assert(!e4_requested(nullptr, "f32", "fp8_e4m3fn"));
    assert(!e4_requested("", "fp8_e4m3fn", nullptr));
    assert(!e4_remote_mode_allowed(true, false, true)); // TCP failure/local Vulkan cannot become E4
    assert(e4_remote_mode_allowed(true, true, true));
    assert(e4_remote_mode_allowed(true, false, false)); // legacy fallback remains available
    assert(e4_remote_mode_allowed(false, false, true)); // unrelated local context

    constexpr uint32_t origin = 128, heads = 2, dim = 256, nq = 2;
    constexpr size_t width = heads * dim, out_bytes = nq * width;
    const int32_t positions[nq] = {127, 128};
    const float neg_inf = -std::numeric_limits<float>::infinity();
    const float pos_inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float good_lse[] = {neg_inf, neg_inf, 0.0f, 1.0f};
    std::vector<uint8_t> reply(out_bytes + nq * heads * sizeof(float), 0);
    reply[0] = 0x80; // negative zero is still zero
    reply[width] = 0x38;
    std::memcpy(reply.data() + out_bytes, good_lse, sizeof(good_lse));
    const auto valid = [&](const std::vector<uint8_t> & r) {
        return e4_reply_valid(r.data(), r.size(), nq, width, heads, positions, origin);
    };
    assert(valid(reply));
    auto bad = reply; bad[0] = 1; assert(!valid(bad)); // empty OUT must be zero
    bad = reply; bad[width] = 0x7f; assert(!valid(bad)); // NaN E4 OUT
    bad = reply; float zero = 0.0f; std::memcpy(bad.data() + out_bytes, &zero, sizeof(zero));
    assert(!valid(bad)); // empty LSE must be -inf
    for (float invalid : {neg_inf, pos_inf, nan}) {
        bad = reply;
        std::memcpy(bad.data() + out_bytes + heads * sizeof(float), &invalid, sizeof(invalid));
        assert(!valid(bad)); // nonempty LSE must be finite
    }
    bad = reply; bad.pop_back(); assert(!valid(bad));
    const int32_t invalid_positions[nq] = {-1, 128};
    assert(!e4_reply_valid(reply.data(), reply.size(), nq, width, heads, invalid_positions, origin));
    assert(!e4_reply_valid(reply.data(), reply.size(), nq, width - 1, heads, positions, origin));

    // The same E4 empty-prefix contract is used after Xbox F16 readback.
    std::vector<uint16_t> f16(nq * width, 0);
    f16[0] = 0x8000; f16[width] = 0x3c00;
    const auto server_valid = [&](const float * lse) {
        return rkva_kvarn_bulk::e4_f16_prefix_results_valid(f16.data(), lse,
                reinterpret_cast<const uint8_t *>(positions), nq, heads, dim, origin);
    };
    assert(server_valid(good_lse));
    f16[0] = 1; assert(!server_valid(good_lse)); f16[0] = 0;
    float bad_lse[] = {neg_inf, neg_inf, neg_inf, 1.0f};
    assert(!server_valid(bad_lse));
    bad_lse[0] = 0.0f; bad_lse[2] = 0.0f; assert(!server_valid(bad_lse));

    // Real descriptor and op22 framing: two queries, one visible row.
    const rkva_kvarn_range_config desc {origin, 512, RKVA_RANGE_STORAGE_KVARN4, 3,
        128, 128, 256, 2, 4, 4, 8960, 3, 2, 128, 128, 1, 1, 1};
    const rkva_kvarn_range_upload upload {1, 0, 0,
        RKVA_KVARN_UPLOAD_RESET | RKVA_KVARN_UPLOAD_OMIT_STAGE, 0, 0, {0, 0}};
    const size_t planes = 2 * 1 * 1 * 256 * sizeof(uint16_t);
    std::vector<uint8_t> request;
    append(request, rkva_kvarn_range_bulk{1, 1, nq, 0});
    append(request, rkva_kvarn_range_bulk_segment{uint32_t(sizeof(upload) + planes), nq});
    append(request, upload);
    request.resize(request.size() + planes, 0);
    append(request, rkva_range_attn{RKVA_WIRE_FP8_E4M3FN, 0, {0, 0}});
    for (int32_t pos : positions) append(request, pos);
    request.resize(request.size() + nq * width, 0);
    std::vector<rkva_kvarn_bulk::Segment> segments;
    assert(rkva_kvarn_bulk::validate_wire(request.data(), request.size(), nq, desc, heads, 1,
            0, 0, false, rkva_kvarn_range::coverage_after_reset(0, 128), 1,
            RKVA_WIRE_FP8_E4M3FN, segments));
    assert(segments.size() == 1 && segments[0].upload.live_count == 1);
    std::puts("E4 policy: explicit fallback, empty-prefix OUT/LSE, invalid responses and real op22 framing passed");
}
