#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "../ggml/src/ggml-remote-attn/rkva-bulk-segments.h"

// The pre-existing implementation. A segment of 8 must reproduce it exactly.
static std::vector<size_t> reference_segments(size_t queries, bool serial_kvarn,
        bool bootstrap, uint32_t tail) {
    const size_t base = serial_kvarn ? (queries > 9 ? 8 : queries) : 1;
    size_t first = base;
    if (serial_kvarn && bootstrap && tail && tail < first) first = tail;
    const size_t rest = first < base && queries - first > 63 * base ? 9 : base;
    std::vector<size_t> result;
    if (!queries) return result;
    result.push_back(first);
    for (size_t remaining = queries - first; remaining; ) {
        const size_t chunk = std::min(rest, remaining);
        result.push_back(chunk);
        remaining -= chunk;
    }
    return result;
}

static int failures = 0;

static void expect(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

static size_t total(const std::vector<size_t> & v) {
    size_t sum = 0;
    for (size_t n : v) sum += n;
    return sum;
}

int main() {
    // Default width must be bit-for-bit identical to the pre-existing policy.
    for (size_t queries = 1; queries <= 600; ++queries) {
        for (int serial = 0; serial <= 1; ++serial) {
            for (int bootstrap = 0; bootstrap <= 1; ++bootstrap) {
                for (uint32_t tail : {0u, 4u, 8u, 16u, 128u}) {
                    const auto got = rkva_bulk_segments(queries, serial != 0, bootstrap != 0, tail);
                    const auto ref = reference_segments(queries, serial != 0, bootstrap != 0, tail);
                    expect(got == ref, "default segment 8 changed the historical split");
                }
            }
        }
    }

    // Non-serial and small serial batches are unchanged by the option.
    for (size_t segment : {8u, 16u, 32u, 64u, 128u}) {
        for (size_t queries : {1u, 2u, 8u, 9u}) {
            const auto got = rkva_bulk_segments(queries, true, false, 128, segment);
            expect(got.size() == 1 && got[0] == queries, "queries <= 9 must stay a single segment");
            const auto non_serial = rkva_bulk_segments(queries, false, false, 128, segment);
            expect(non_serial.size() == queries, "non-serial split must stay per-query");
        }
    }

    // Wider segments keep the first native segment at 8 and preserve the sum.
    for (size_t segment : {16u, 32u, 64u, 128u}) {
        for (size_t queries : {10u, 11u, 16u, 32u, 128u, 512u, 1024u}) {
            const auto got = rkva_bulk_segments(queries, true, false, 128, segment);
            expect(!got.empty(), "wide split must not be empty");
            expect(got[0] == std::min<size_t>(8, queries), "first native segment must stay at 8");
            expect(total(got) == queries, "wide split must preserve the token count");
            for (size_t n : got) {
                expect(n > 0 && n <= std::max<size_t>(segment, 9), "wide segment out of bounds");
            }
            if (queries >= 64) {
                expect(got.size() < reference_segments(queries, true, false, 128).size(),
                    "wide split must reduce the number of segments");
            }
        }
    }

    // A bootstrap tail smaller than the native first segment still bounds it.
    for (uint32_t tail : {1u, 4u, 7u}) {
        const auto got = rkva_bulk_segments(512, true, true, tail, 32);
        expect(!got.empty() && got[0] == tail, "bootstrap tail must bound the first segment");
        expect(total(got) == 512, "bootstrap split must preserve the token count");
    }

    // A 512-token ubatch with 32-query segments: 8 + 15 x 32 + 24.
    {
        const auto got = rkva_bulk_segments(512, true, false, 128, 32);
        expect(got.size() == 17, "512/32 split size");
        expect(got[0] == 8 && got[1] == 32 && got.back() == 24, "512/32 split shape");
    }

    // The bootstrap parent must stay entirely at the native MQ2 width.
    {
        const auto got = rkva_bulk_segments(512, true, true, 128, 32);
        expect(got.size() == 64, "bootstrap/32 split size");
        for (size_t n : got) expect(n == 8, "bootstrap/32 segment width");
    }

    if (failures) {
        std::fprintf(stderr, "%d bulk-segment checks failed\n", failures);
        return 1;
    }
    std::printf("bulk-segment checks passed\n");
    return 0;
}
