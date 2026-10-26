#include "../ggml/src/ggml-mask-pack.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); std::exit(1); } } while (0)

using pack_fn = bool (*)(const void *, size_t, uint32_t *);
static size_t checks = 0;

static uint32_t load_word(const uint8_t * p) {
    uint32_t word;
    std::memcpy(&word, p, sizeof(word));
    return word;
}

static std::vector<uint8_t> guarded_pack(pack_fn pack, const std::vector<uint16_t> & values,
        size_t source_offset, size_t destination_offset, bool expected_valid = true) {
    const size_t source_bytes = values.size()*sizeof(uint16_t);
    std::vector<uint8_t> source(source_bytes + source_offset + 17, 0x69);
    if (source_bytes) std::memcpy(source.data() + source_offset, values.data(), source_bytes);
    const auto source_before = source;
    const size_t output_bytes = ggml_mask_pack_f16_words(values.size())*sizeof(uint32_t);
    std::vector<uint8_t> output(output_bytes + destination_offset + 17, 0xa5);
    const bool valid = pack(source.data() + source_offset, values.size(),
            reinterpret_cast<uint32_t *>(output.data() + destination_offset));
    CHECK(valid == expected_valid);
    CHECK(source == source_before);
    for (size_t i = 0; i < destination_offset; ++i) CHECK(output[i] == 0xa5);
    for (size_t i = destination_offset + output_bytes; i < output.size(); ++i) CHECK(output[i] == 0xa5);
    ++checks;
    return std::vector<uint8_t>(output.begin() + destination_offset, output.begin() + destination_offset + output_bytes);
}

static void check_valid(const std::vector<uint16_t> & values) {
    for (size_t source_offset : {size_t(0), size_t(1), size_t(3), size_t(15)}) {
        for (size_t destination_offset : {size_t(0), size_t(1), size_t(2), size_t(3)}) {
            const auto packed = guarded_pack(ggml_mask_pack_f16, values, source_offset, destination_offset);
            const auto scalar = guarded_pack(ggml_mask_pack_detail::scalar, values, source_offset, destination_offset);
            CHECK(packed == scalar);
#if GGML_MASK_PACK_HAS_SSE2
            CHECK(packed == guarded_pack(ggml_mask_pack_detail::sse2, values, source_offset, destination_offset));
#endif
            for (size_t i = 0; i < values.size(); ++i) {
                // Independent decoder/oracle: no packing helper is reused.
                const uint32_t word = load_word(packed.data() + (i/32)*sizeof(uint32_t));
                const uint16_t decoded = (word & (uint32_t(1) << (i%32))) ? uint16_t(0xfc00) : uint16_t(0);
                CHECK(decoded == values[i]);
            }
            if (values.size()%32) {
                const uint32_t last = load_word(packed.data() + packed.size() - sizeof(uint32_t));
                CHECK((last >> (values.size()%32)) == 0);
            }
        }
    }
}

static void check_known_words() {
    // These literal expected words exercise lane order and the 16/32 boundary.
    for (uint32_t word : {uint32_t(0), uint32_t(0xffffffff), uint32_t(0x80018001),
            uint32_t(0xaaaaaaaa), uint32_t(0x55555555), uint32_t(0x01234567)}) {
        std::vector<uint16_t> values(32);
        for (size_t i = 0; i < values.size(); ++i) values[i] = (word >> i) & 1 ? 0xfc00 : 0;
        auto packed = guarded_pack(ggml_mask_pack_f16, values, 1, 1);
        CHECK(load_word(packed.data()) == word);
    }
    for (size_t lane = 0; lane < 32; ++lane) {
        std::vector<uint16_t> values(32, 0);
        values[lane] = 0xfc00;
        auto packed = guarded_pack(ggml_mask_pack_f16, values, 3, 3);
        CHECK(load_word(packed.data()) == (uint32_t(1) << lane));
    }
}

static void check_lengths_and_patterns() {
    std::mt19937 rng(0x4d41534b);
    for (size_t n : {size_t(0), size_t(1), size_t(15), size_t(16), size_t(17), size_t(31),
            size_t(32), size_t(33), size_t(63), size_t(64), size_t(65), size_t(257), size_t(8192)}) {
        std::vector<uint16_t> values(n, 0);
        check_valid(values);
        std::fill(values.begin(), values.end(), 0xfc00);
        check_valid(values);
        for (size_t i = 0; i < n; ++i) values[i] = (i%7 == 1 || i%11 == 0) ? 0xfc00 : 0;
        check_valid(values); // sparse holes and irregular causal-looking runs
        for (int pass = 0; pass < 8; ++pass) {
            for (auto & value : values) value = (rng() & 1) ? 0xfc00 : 0;
            check_valid(values);
        }
    }
}

static void check_invalid_values() {
    // -0, subnormal, finite, +inf and multiple NaN payloads are all forbidden.
    const std::array<uint16_t, 10> invalid = {
        0x8000, 0x0001, 0x8001, 0x3c00, 0xbc00, 0x7bff, 0x7c00, 0x7c01, 0x7e00, 0xfe01,
    };
    for (size_t n : {size_t(1), size_t(17), size_t(32), size_t(33), size_t(65)}) {
        // Put every invalid pattern in every SIMD lane and every scalar tail.
        for (size_t lane = 0; lane < n; ++lane) {
            for (const auto bad : invalid) {
                std::vector<uint16_t> values(n, 0xfc00);
                values[lane] = bad;
                guarded_pack(ggml_mask_pack_f16, values, 1, 1, false);
                guarded_pack(ggml_mask_pack_detail::scalar, values, 3, 3, false);
#if GGML_MASK_PACK_HAS_SSE2
                guarded_pack(ggml_mask_pack_detail::sse2, values, 15, 2, false);
#endif
            }
        }
    }
    // Exhaust the full F16 raw domain at both SIMD and tail positions.
    std::vector<uint16_t> values(65, 0);
    for (uint32_t raw = 0; raw <= 0xffff; ++raw) {
        values[13] = uint16_t(raw);
        values[64] = uint16_t(raw);
        const bool valid = raw == 0 || raw == 0xfc00;
        uint32_t output[3] = {};
        CHECK(ggml_mask_pack_f16(values.data(), values.size(), output) == valid);
        CHECK(ggml_mask_pack_detail::scalar(values.data(), values.size(), output) == valid);
        ++checks;
    }
}

int main() {
    CHECK(ggml_mask_pack_f16(nullptr, 0, nullptr));
    CHECK(!ggml_mask_pack_f16(nullptr, 1, nullptr));
    uint16_t source = 0;
    uint32_t output = 0;
    CHECK(!ggml_mask_pack_f16(nullptr, 1, &output));
    CHECK(!ggml_mask_pack_f16(&source, 1, nullptr));
    CHECK(ggml_mask_pack_f16_words(0) == 0 && ggml_mask_pack_f16_words(33) == 2);
    CHECK(ggml_mask_pack_f16_words(std::numeric_limits<size_t>::max()) ==
            std::numeric_limits<size_t>::max()/32 + 1);
    CHECK(!ggml_mask_pack_f16(&source, std::numeric_limits<size_t>::max(), &output));
    check_known_words();
    check_lengths_and_patterns();
    check_invalid_values();
    std::printf("F16 binary mask pack: PASS (%zu guarded/domain checks; SSE2=%d)\n", checks, GGML_MASK_PACK_HAS_SSE2);
}
