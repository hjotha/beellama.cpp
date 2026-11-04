#undef NDEBUG
#include "rkva-turboquant-nc-producer.h"
#include "rkva-kvarn-bulk.h"
#include "rkva-range-profile.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

// Standalone mode links the real codec with the console's portable conversions.
// CMake defines RKVA_TQNC_USE_GGML_FP16 and links the real ggml-base instead.
#ifndef RKVA_TQNC_USE_GGML_FP16
extern "C" ggml_fp16_t ggml_fp32_to_fp16(float x) { return rkva_kvarn_bulk::f32_to_f16_bits(x); }
extern "C" float ggml_fp16_to_fp32(ggml_fp16_t x) { return rkva_kvarn_bulk::f16_bits_to_f32(x); }
extern "C" void ggml_abort(const char *, int, const char *, ...) { std::abort(); }
#endif

int main() {
    static_assert(rkva_rprof::kStorageTurboquantNC == RKVA_RANGE_STORAGE_TURBOQUANT_NC, "profile storage id");
    assert(std::strcmp(rkva_rprof::storage_name(RKVA_RANGE_STORAGE_TURBOQUANT_NC), "turboquant_nc") == 0);
    assert(rkva_rprof::bulk_upload_copy_regions(RKVA_RANGE_STORAGE_TURBOQUANT_NC, true) == 2);
    rkva_rprof::Header profile{};
    profile.turboquant_nc_available = true;
    profile.turboquant_nc_status = "ready";
    assert(rkva_rprof::Recorder().json(profile).find("\"turboquant_nc\":{\"available\":true") != std::string::npos);
    rkva_rprof::Sample sample{};
    sample.kind = rkva_rprof::kAttention;
    sample.storage = RKVA_RANGE_STORAGE_TURBOQUANT_NC;
    sample.optimized = 1;
    assert(rkva_rprof::bucket_key(sample).find("attention/turboquant_nc/optimized-turboquant_nc/") == 0);
    std::vector<float> output(6 * 256, 0.0f), lse(6, 0.0f);
    int32_t query_position = 256;
    auto output_valid = [&](bool narrow) { return rkva_kvarn_bulk::turboquant_nc_output_valid(
        reinterpret_cast<const uint8_t*>(output.data()), reinterpret_cast<const uint8_t*>(lse.data()),
        reinterpret_cast<const uint8_t*>(&query_position), 1, 6, 256, 1, narrow); };
    assert(output_valid(true));
    output[0] = 70000.0f; assert(!output_valid(true) && output_valid(false)); output[0] = 0.0f;
    output[0] = std::numeric_limits<float>::quiet_NaN(); assert(!output_valid(false)); output[0] = 0.0f;
    lse[0] = -std::numeric_limits<float>::infinity(); assert(!output_valid(true));
    query_position = 255; assert(output_valid(true));
    lse[0] = std::numeric_limits<float>::quiet_NaN(); assert(!output_valid(true));
    const rkva_kvarn_range_config good {256, 8, RKVA_RANGE_STORAGE_TURBOQUANT_NC, 8,
        1, 0, 256, 2, 4, 4, 132, 0, 0, 0, 0, 256, RKVA_DOMAIN_ORIGINAL_V, 1};
    assert(rkva_kvarn_range::valid_turboquant_nc_descriptor(good, 256));
    auto bad = good;
    bad.domain = RKVA_DOMAIN_ROTATED;
    assert(!rkva_kvarn_range::valid_turboquant_nc_descriptor(bad, 256));
    bad = good; bad.record_bytes = 130;
    assert(!rkva_kvarn_range::valid_turboquant_nc_descriptor(bad, 256));
    bad = good; bad.wht_version = 16;
    assert(!rkva_kvarn_range::valid_turboquant_nc_descriptor(bad, 256));
    bad = good; bad.origin = 255;
    assert(!rkva_kvarn_range::valid_turboquant_nc_descriptor(bad, 256));
    bad = good; bad.storage = RKVA_RANGE_STORAGE_SAW_INT4;
    assert(!rkva_kvarn_range::valid_turboquant_nc_descriptor(bad, 256));
    assert(!rkva_kvarn_range::valid_turboquant_nc_descriptor(good, 128));

    constexpr size_t heads = 2, count = 2, dim = 256, row = heads * 132;
    std::vector<float> keys(count * heads * dim), values(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        keys[i] = std::sin(float(i) * 0.037f);
        values[i] = std::cos(float(i) * 0.053f) * 2.0f - 0.3f;
    }
    const int32_t positions[] = {256, 257};
    rkva_turboquant_nc_producer producer(256, 8, heads);
    std::vector<uint8_t> payload;
    assert(producer.append(keys.data(), values.data(), positions, count, true, payload));
    assert(producer.rows() == count && payload.size() == sizeof(rkva_kvarn_range_upload) + 2 * count * row);
    rkva_kvarn_range_upload u{};
    std::memcpy(&u, payload.data(), sizeof(u));
    assert(rkva_kvarn_range::valid_turboquant_nc_upload(u, 8, 0));
    assert(u.flags == RKVA_KVARN_UPLOAD_RESET && u.append_rows == count);
    auto no_reset = u;
    no_reset.flags = 0;
    assert(!rkva_kvarn_range::valid_turboquant_nc_upload(no_reset, 8, 0));
    const size_t off = sizeof(u), plane = count * row;
    for (size_t i = 0; i < count * heads; ++i) {
        ggml_tqnc_key k{};
        ggml_tqnc_value v{};
        assert(ggml_tqnc_quantize_key(keys.data() + i * dim, &k));
        assert(ggml_tqnc_quantize_value(values.data() + i * dim, &v));
        assert(std::memcmp(payload.data() + off + i * 132, &k, sizeof(k)) == 0);
        assert(payload[off + i * 132 + 130] == 0 && payload[off + i * 132 + 131] == 0);
        assert(std::memcmp(payload.data() + off + plane + i * 132, &v, sizeof(v)) == 0);
    }
    auto valid = [&]() { return rkva_kvarn_bulk::turboquant_nc_planes_valid(
        payload.data() + off, payload.data() + off + plane, count * heads); };
    assert(valid());
    auto pristine = payload;
    payload[off + 130] = 1; assert(!valid()); payload = pristine;
    payload[off + 128] = 0; payload[off + 129] = 0x7c; assert(!valid()); payload = pristine;
    payload[off + plane + 128] = 0; payload[off + plane + 129] = 0x80; // negative zero is valid
    assert(valid()); payload = pristine;
    payload[off + plane + 128] = 0; payload[off + plane + 129] = 0xbc; assert(!valid()); payload = pristine;
    payload[off + plane + 130] = 0; payload[off + plane + 131] = 0x7c; assert(!valid()); payload = pristine;
    payload[off + plane + 128] = 0xff; payload[off + plane + 129] = 0x7b; assert(!valid()); payload = pristine;

    const int32_t next[] = {258};
    keys[dim + 17] = std::numeric_limits<float>::quiet_NaN(); // second head, after a row was packed
    assert(!producer.append(keys.data(), values.data(), next, 1, false, payload));
    assert(producer.rows() == count && payload == pristine);
    keys[dim + 17] = 0;
    assert(!producer.append(keys.data(), values.data(), next, 1, true, payload));
    assert(producer.rows() == count && payload == pristine);
    assert(producer.append(keys.data(), values.data(), next, 1, false, payload));
    assert(producer.rows() == 3 && producer.rewind(1) && producer.rows() == 1);
    assert(!producer.rewind(2));
    assert(producer.append(keys.data(), values.data(), positions + 1, 1, false, payload));
    std::memcpy(&u, payload.data(), sizeof(u));
    assert(u.append_begin == 1 && u.live_count == 2 && u.flags == 0 &&
           rkva_kvarn_range::valid_turboquant_nc_upload(u, 8, 1));
    assert(producer.rewind(0));
    assert(producer.append(keys.data(), values.data(), positions, 1, true, payload));
    std::memcpy(&u, payload.data(), sizeof(u));
    assert(u.append_begin == 0 && u.live_count == 1 && u.flags == RKVA_KVARN_UPLOAD_RESET);
    std::puts("PASS: TurboQuant NC descriptor, wire layout, metadata and transactional owner");
}
