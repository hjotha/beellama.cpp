// Host checks for uwp/rkva/rkva-range-profile.h: kernel-mode policy, split
// math, ring behaviour and JSON report shape. Build and run:
//   g++ -std=c++17 -Wall -Wextra -Werror tests/rkva_range_profile_cpu.cpp -o /tmp/rkva_profile
//   /tmp/rkva_profile > /tmp/rkva_profile.json && python3 -m json.tool /tmp/rkva_profile.json
#include "../uwp/rkva/rkva-bulk-scratch.h"
#include "../uwp/rkva/rkva-range-profile.h"
#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <cstring>

using namespace rkva_rprof;

int main() {
    // Mode validation and names.
    for (uint32_t m = 0; m < kModeCount; ++m) {
        assert(valid_mode(m));
        assert(std::strcmp(mode_name(m), "invalid") != 0);
    }
    assert(!valid_mode(4) && !valid_mode(0xffffffffu));

    // Reference live259 split policy is unchanged.
    assert(reference_split_k(0) == 1 && reference_split_k(1) == 1);
    assert(reference_split_k(256) == 1 && reference_split_k(257) == 2);
    assert(reference_split_k(28672) == 32 && reference_split_k(1u << 20) == 32);

    // Optimized split policies stay within the opt reducer and partial buffer.
    for (uint32_t m = 1; m < kModeCount; ++m) {
        for (uint32_t a : {0u, 1u, 31u, 32u, 33u, 128u, 129u, 512u, 4096u, 28672u, 131072u}) {
            const uint32_t k = optimized_split_k(a, m);
            assert(k >= 1 && k <= kOptMaxSplits && k <= split_policy(m).max_splits);
            const uint32_t rows = split_rows(a, k);
            assert(uint64_t(rows) * k >= a);                // every row is covered
            assert(a == 0 || uint64_t(rows) * (k - 1) < a); // no split starts past the end
        }
    }
    assert(optimized_split_k(1, kModeOptBalanced) == 1); // direct-output path
    assert(optimized_split_k(28672, kModeOptBalanced) == 128);
    assert(optimized_split_k(28672, kModeOptFine) == 256);
    assert(optimized_split_k(28672, kModeOptCoarse) == 56);

    // Per-call gate: qualified Q4 decode shape only.
    assert(optimized_q4_supported(1, 1, 1, 24, 4, 256));
    assert(!optimized_q4_supported(0, 1, 1, 24, 4, 256)); // reference mode
    assert(!optimized_q4_supported(4, 1, 1, 24, 4, 256)); // invalid mode
    assert(!optimized_q4_supported(1, 0, 1, 24, 4, 256)); // F16
    assert(!optimized_q4_supported(1, 2, 1, 24, 4, 256)); // Q8
    assert(!optimized_q4_supported(1, 3, 1, 24, 4, 256)); // KVarN4
    // N proposed tokens require N+1 queries. Cover every bounded verification
    // shape, plus the first unsupported prefill batch.
    for (uint32_t nq = 1; nq <= 9; ++nq) {
        assert(optimized_q4_supported(1, 1, nq, 24, 4, 256));
        assert(optimized_format_supported(1, 0, nq, 24, 4, 256));
        assert(optimized_format_supported(1, 2, nq, 24, 4, 256));
        assert(optimized_kvarn4_supported(1, 3, nq, 24, 4, 256, 102400, 128));
    }
    assert(!optimized_q4_supported(1, 1, 10, 24, 4, 256));
    assert(!optimized_q4_supported(1, 1, 0, 24, 4, 256));
    assert(!optimized_q4_supported(1, 1, 1, 32, 4, 256)); // GQA 8
    assert(!optimized_q4_supported(1, 1, 1, 24, 4, 128)); // head_dim 128
    assert(!optimized_q4_supported(1, 1, 1, 24, 0, 256));

    // Storage-qualified kernel names: Q4 keeps the 261 names, mode 0 is reference.
    for (uint32_t m = 0; m < kModeCount; ++m) {
        assert(std::strcmp(kernel_name(kStorageQ4, m), mode_name(m)) == 0);
        for (uint32_t st = 0; st < kStorageCount; ++st) {
            assert(std::strcmp(kernel_name(st, m), "invalid") != 0);
        }
    }
    assert(std::strcmp(kernel_name(kStorageF16, 0), "reference") == 0);
    assert(std::strcmp(kernel_name(kStorageF16, 1), "f16_gqa_balanced") == 0);
    assert(std::strcmp(kernel_name(kStorageQ8, 2), "q8_gqa_fine") == 0);
    assert(std::strcmp(kernel_name(kStorageKvarn4, 3), "kvarn4_gqa_coarse") == 0);
    assert(std::strcmp(kernel_name(7, 1), "invalid") == 0);

    // F16/Q8 gate: same decode shape, only storages 0 and 2.
    for (uint32_t m = 1; m < kModeCount; ++m) {
        assert(optimized_format_supported(m, 0, 1, 24, 4, 256));
        assert(optimized_format_supported(m, 2, 1, 24, 4, 256));
    }
    assert(!optimized_format_supported(0, 0, 1, 24, 4, 256));
    assert(!optimized_format_supported(4, 2, 1, 24, 4, 256));
    assert(!optimized_format_supported(1, 1, 1, 24, 4, 256)); // Q4 has its own gate
    assert(!optimized_format_supported(1, 3, 1, 24, 4, 256)); // KVarN4 has its own gate
    assert(!optimized_format_supported(1, 0, 10, 24, 4, 256));
    assert(!optimized_format_supported(1, 2, 1, 32, 4, 256));
    assert(!optimized_format_supported(1, 2, 1, 24, 4, 128));

    // KVarN4 gate: descriptor-v1 origin/tail shapes only.
    assert(optimized_kvarn4_supported(1, 3, 1, 24, 4, 256, 102400, 0));
    assert(optimized_kvarn4_supported(2, 3, 1, 24, 4, 256, 102400, 128));
    assert(optimized_kvarn4_supported(3, 3, 1, 24, 4, 256, 0, 128));
    assert(!optimized_kvarn4_supported(0, 3, 1, 24, 4, 256, 102400, 128));
    assert(!optimized_kvarn4_supported(1, 1, 1, 24, 4, 256, 102400, 128));
    assert(!optimized_kvarn4_supported(1, 3, 1, 24, 4, 256, 102464, 128)); // unaligned origin
    assert(!optimized_kvarn4_supported(1, 3, 1, 24, 4, 256, 102400, 64));  // other tail
    assert(!optimized_kvarn4_supported(1, 3, 10, 24, 4, 256, 102400, 128));
    assert(!optimized_kvarn4_supported(1, 3, 1, 24, 4, 128, 102400, 128));
    for (uint32_t m = 1; m < kModeCount; ++m) {
        for (uint32_t a : {0u, 1u, 7u, 8u, 9u, 127u, 128u, 129u, 1000u, 4133u, 28672u, 131072u}) {
            const KvarnSplit ks = kvarn4_split(a, m);
            assert(ks.split_k >= 1 && ks.split_k <= kOptMaxSplits);
            assert(ks.split_rows % kKvarnTile == 0);
            assert(uint64_t(ks.split_rows) * ks.split_k >= a);
            assert(a == 0 || uint64_t(ks.split_rows) * (ks.split_k - 1) < a);
        }
    }
    assert(kvarn4_split(1, 1).split_k == 1);         // direct output
    assert(kvarn4_split(1000, 1).split_rows == 128); // 125 -> 128 rows, 8 splits
    assert(kvarn4_split(1000, 1).split_k == 8);

    // Q4_0 multi-query tile selector: nq 2..9 only, mode 1 -> 4, mode 2 -> 2,
    // mode 3 and nq1 keep the per-query kernel (and its split policy).
    for (uint32_t m = 0; m < kModeCount + 1; ++m) {
        assert(q4_mq_tile(m, 0) == 0 && q4_mq_tile(m, 1) == 0 && q4_mq_tile(m, 10) == 0);
        for (uint32_t nq = 2; nq <= kOptMaxQueries; ++nq) {
            const uint32_t want = m == kModeOptBalanced ? 4u : m == kModeOptFine ? 2u : 0u;
            assert(q4_mq_tile(m, nq) == want);
            assert(want <= kQ4MqTileMax);
        }
    }
    assert(std::strcmp(q4_mq_kernel_name(4), "q4_mq4_balanced") == 0);
    assert(std::strcmp(q4_mq_kernel_name(2), "q4_mq2_fine") == 0);
    assert(std::strcmp(q4_mq_kernel_name(3), "invalid") == 0);
    for (uint32_t m = 1; m < kModeCount; ++m) {
        for (uint32_t a : {0u, 1u, 63u, 64u, 65u, 455u, 4096u, 28672u, 58128u, 58368u, 131072u}) {
            const Q4MqSplit mq = q4_mq_split(a, m);
            assert(mq.split_k >= 1 && mq.split_k <= kOptMaxSplits);
            assert(mq.split_k <= optimized_split_k(a, m)); // fits the bounded partials
            assert(mq.split_rows % kQ4MqRows == 0);
            assert(uint64_t(mq.split_rows) * mq.split_k >= a);
            assert(a == 0 || uint64_t(mq.split_rows) * (mq.split_k - 1) < a);
        }
    }
    assert(q4_mq_split(1, kModeOptBalanced).split_k == 1); // direct output
    assert(q4_mq_split(58128, kModeOptBalanced).split_rows == 512);
    assert(q4_mq_split(58128, kModeOptBalanced).split_k == 114);
    assert(q4_mq_split(58128, kModeOptFine).split_rows == 256);
    assert(q4_mq_split(58128, kModeOptFine).split_k == 228);

    // F16 / Q8_0 multi-query tiles (267): same legacy mode map as Q4_0, only for
    // F16/Q8_0 storage; Q4_0 keeps its own selector and KVarN4 never tiles.
    for (uint32_t m = 0; m < kModeCount + 1; ++m) {
        for (uint32_t nq = 0; nq <= kOptMaxQueries + 1; ++nq) {
            assert(format_mq_tile(m, kStorageF16, nq) == q4_mq_tile(m, nq));
            assert(format_mq_tile(m, kStorageQ8, nq) == q4_mq_tile(m, nq));
            assert(format_mq_tile(m, kStorageQ4, nq) == 0);
            assert(format_mq_tile(m, kStorageKvarn4, nq) == 0);
            assert(format_mq_tile(m, kStorageCount, nq) == 0);
        }
    }
    assert(format_mq_tile(kModeOptCoarse, kStorageQ8, 3) == 0); // per-query baseline
    assert(std::strcmp(format_mq_kernel_name(kStorageF16, 4), "f16_mq4_balanced") == 0);
    assert(std::strcmp(format_mq_kernel_name(kStorageF16, 2), "f16_mq2_fine") == 0);
    assert(std::strcmp(format_mq_kernel_name(kStorageQ8, 4), "q8_mq4_balanced") == 0);
    assert(std::strcmp(format_mq_kernel_name(kStorageQ8, 2), "q8_mq2_fine") == 0);
    assert(std::strcmp(format_mq_kernel_name(kStorageQ8, 3), "invalid") == 0);
    assert(std::strcmp(format_mq_kernel_name(kStorageQ4, 4), "invalid") == 0);
    assert(std::strcmp(format_mq_kernel_name(kStorageKvarn4, 2), "invalid") == 0);
    assert(std::strcmp(mq_kernel_name(kStorageQ4, 4), "q4_mq4_balanced") == 0);
    assert(std::strcmp(mq_kernel_name(kStorageQ8, 2), "q8_mq2_fine") == 0);
    // Native KVarN4 revision-3 multi-query tile (268/MQ4): mode 1 -> 4, mode 2 -> 2,
    // nq 2..9, tail 128 only; the F16/Q8_0 selector and names never return it.
    for (uint32_t m = 0; m < kModeCount; ++m) {
        for (uint32_t nq = 0; nq <= 12; ++nq) {
            const uint32_t want = (nq >= 2 && nq <= kOptMaxQueries) ? (m == kModeOptBalanced ? 4u
                                                                       : m == kModeOptFine   ? 2u
                                                                                             : 0u)
                                                                    : 0u;
            assert(kvarn4_mq_tile(m, nq, 128) == want);
            assert(kvarn4_mq_tile(m, nq, 0) == 0);
            assert(kvarn4_mq_tile(m, nq, 64) == 0);
        }
    }
    assert(std::strcmp(mq_kernel_name(kStorageKvarn4, 2), "kvarn4_mq2_fine") == 0);
    assert(std::strcmp(mq_kernel_name(kStorageKvarn4, 4), "kvarn4_mq4_balanced") == 0);
    assert(std::strcmp(mq_kernel_name(kStorageKvarn4, 0), "invalid") == 0);
    assert(kKvarn4MqWaveLanes == 64);
    assert(kKvarn4MqTileMax == 4);

    // Native KVarN4 wave64 decode kernel and split policy (nq == 1):
    assert(kKvarn4DecodeWaveLanes == 64);
    assert(kvarn4_decode_opt(kModeOptBalanced, 64));
    assert(kvarn4_decode_opt(kModeOptFine, 64));
    assert(!kvarn4_decode_opt(kModeOptCoarse, 64));
    assert(!kvarn4_decode_opt(kModeReference, 64));
    assert(!kvarn4_decode_opt(kModeOptFine, 32)); // requires wave64

    assert(std::strcmp(kvarn4_decode_kernel_name(kModeOptBalanced), "kvarn4_decode_balanced") == 0);
    assert(std::strcmp(kvarn4_decode_kernel_name(kModeOptFine), "kvarn4_decode_fine") == 0);
    assert(std::strcmp(kvarn4_decode_kernel_name(kModeOptCoarse), "kvarn4_gqa_coarse") == 0);
    assert(std::strcmp(kvarn4_decode_kernel_name(0), "invalid") == 0);

    for (uint32_t m = 1; m < kModeCount; ++m) {
        for (uint32_t a :
             {0u, 1u, 127u, 128u, 129u, 1000u, 6144u, 16384u, 24576u, 34816u, 131072u}) {
            const KvarnSplit ks = kvarn4_decode_split(a, m);
            assert(ks.split_k >= 1 && ks.split_k <= kOptMaxSplits);
            if (m != kModeOptCoarse) {
                assert(ks.split_rows % kKvarnGroup == 0); // whole 128-token record groups
            } else {
                assert(ks.split_rows % kKvarnTile == 0);
            }
            assert(uint64_t(ks.split_rows) * ks.split_k >= a);
            assert(a == 0 || uint64_t(ks.split_rows) * (ks.split_k - 1) < a);
        }
    }
    // Remote context decode split counts:
    // 6k (6144):
    assert(kvarn4_decode_split(6144, kModeOptBalanced).split_k == 12);
    assert(kvarn4_decode_split(6144, kModeOptBalanced).split_rows == 512);
    assert(kvarn4_decode_split(6144, kModeOptFine).split_k == 24);
    assert(kvarn4_decode_split(6144, kModeOptFine).split_rows == 256);
    // 16k (16384):
    assert(kvarn4_decode_split(16384, kModeOptBalanced).split_k == 32);
    assert(kvarn4_decode_split(16384, kModeOptBalanced).split_rows == 512);
    assert(kvarn4_decode_split(16384, kModeOptFine).split_k == 64);
    assert(kvarn4_decode_split(16384, kModeOptFine).split_rows == 256);
    // 35k (34816):
    assert(kvarn4_decode_split(34816, kModeOptBalanced).split_k == 31);
    assert(kvarn4_decode_split(34816, kModeOptBalanced).split_rows == 1152);
    assert(kvarn4_decode_split(34816, kModeOptFine).split_k == 55);
    assert(kvarn4_decode_split(34816, kModeOptFine).split_rows == 640);

    // Exhaust the supported context ladder, including both cap transitions.
    for (uint32_t mode : {kModeOptBalanced, kModeOptFine}) {
        for (uint32_t tile : {2u, 4u}) {
            for (uint32_t n = 0; n <= 131072; ++n) {
                const KvarnSplit ks = kvarn4_mq_split(n, mode, tile);
                assert(ks.split_k >= 1 && ks.split_k <= (mode == kModeOptFine ? 64u : 32u));
                assert(ks.split_rows % kKvarnTile == 0);
                if (!n) {
                    assert(ks.split_k == 1 && ks.split_rows == 0);
                    continue;
                }
                assert(uint64_t(ks.split_rows) * ks.split_k >= n);
                assert(uint64_t(ks.split_rows) * (ks.split_k - 1) < n);
                assert(ks.split_k == 1 || n > 256);
            }
        }
    }
    assert(kvarn4_mq_split(257, kModeOptBalanced, 4).split_k == 2);
    assert(kvarn4_mq_split(257, kModeOptBalanced, 4).split_rows == 136);
    assert(kvarn4_mq_split(8193, kModeOptBalanced, 4).split_rows == 264);
    assert(kvarn4_mq_split(16385, kModeOptFine, 2).split_k == 63);
    assert(kvarn4_mq_split(16385, kModeOptFine, 2).split_rows == 264);

    // Multi-query split policy tests (kvarn4_mq_split)
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
        // Remote context multi-query split counts:
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

    // Bucket keys separate warm decode from prefill and cold calls.
    Sample s;
    s.kind = kAttention;
    s.storage = 1;
    s.n_tokens = 1;
    s.rows = 28672;
    assert(bucket_key(s) == "attention/q4_0/reference/nq1/r16k-32k/warm");
    s.optimized = 1;
    s.mode = 2;
    s.alloc_grew = 1;
    s.n_tokens = 256;
    assert(bucket_key(s) == "attention/q4_0/optimized-q4_gqa_fine/nq129+/r16k-32k/cold");
    s.storage = 3;
    s.mode = 1;
    s.alloc_grew = 0;
    s.n_tokens = 1;
    assert(bucket_key(s) == "attention/kvarn4/optimized-kvarn4_gqa_balanced/nq1/r16k-32k/warm");
    s.storage = 2;
    assert(bucket_key(s) == "attention/q8_0/optimized-q8_gqa_balanced/nq1/r16k-32k/warm");
    // Revision-3 serial-prefix calls never share a bucket with resident-image ones.
    s.storage = 3;
    s.serial_prefix = 1;
    s.n_tokens = 3;
    assert(bucket_key(s) ==
           "attention/kvarn4-prefix/optimized-kvarn4_gqa_balanced/nq2-16/r16k-32k/warm");
    s.optimized = 0;
    assert(bucket_key(s) == "attention/kvarn4-prefix/reference/nq2-16/r16k-32k/warm");
    s.optimized = 1;
    s.mode = 2;
    s.q_tile = 2;
    assert(bucket_key(s) ==
           "attention/kvarn4-prefix/optimized-kvarn4_mq2_fine/nq2-16/r16k-32k/warm");
    s.mode = 1;
    s.q_tile = 4;
    assert(bucket_key(s) ==
           "attention/kvarn4-prefix/optimized-kvarn4_mq4_balanced/nq2-16/r16k-32k/warm");
    s.mode = 3;
    s.q_tile = 0;
    assert(bucket_key(s) ==
           "attention/kvarn4-prefix/optimized-kvarn4_gqa_coarse/nq2-16/r16k-32k/warm");
    // Decode specialization bucket (nq == 1, decode_opt == 1):
    s.n_tokens = 1;
    s.q_tile = 0;
    s.decode_opt = 1;
    s.mode = 1;
    assert(bucket_key(s) ==
           "attention/kvarn4-prefix/optimized-kvarn4_decode_balanced/nq1/r16k-32k/warm");
    s.mode = 2;
    assert(bucket_key(s) ==
           "attention/kvarn4-prefix/optimized-kvarn4_decode_fine/nq1/r16k-32k/warm");
    s.decode_opt = 0;
    s.mode = 1;
    s.n_tokens = 3;
    // Multi-query tile calls get their own Q4 kernel bucket.
    s.storage = 1;
    s.serial_prefix = 0;
    s.optimized = 1;
    s.q_tile = 4;
    assert(bucket_key(s) == "attention/q4_0/optimized-q4_mq4_balanced/nq2-16/r16k-32k/warm");
    s.mode = 2;
    s.q_tile = 2;
    assert(bucket_key(s) == "attention/q4_0/optimized-q4_mq2_fine/nq2-16/r16k-32k/warm");
    s.mode = 3;
    s.q_tile = 0;
    assert(bucket_key(s) == "attention/q4_0/optimized-q4_gqa_coarse/nq2-16/r16k-32k/warm");
    // F16/Q8_0 multi-query buckets are storage-qualified (never q4_mq*).
    s.storage = kStorageQ8;
    s.mode = 1;
    s.q_tile = 4;
    assert(bucket_key(s) == "attention/q8_0/optimized-q8_mq4_balanced/nq2-16/r16k-32k/warm");
    s.storage = kStorageF16;
    s.mode = 2;
    s.q_tile = 2;
    assert(bucket_key(s) == "attention/f16/optimized-f16_mq2_fine/nq2-16/r16k-32k/warm");
    s.mode = 3;
    s.q_tile = 0;
    assert(bucket_key(s) == "attention/f16/optimized-f16_gqa_coarse/nq2-16/r16k-32k/warm");

    // Ring: fixed capacity, overwrites oldest, counts everything.
    Recorder r;
    assert(r.capacity() == 0);
    r.record(Sample{}); // unconfigured: counted, not stored
    assert(r.recorded() == 1 && r.stored() == 0);
    r.configure(4);
    r.reset();
    for (uint32_t i = 0; i < 6; ++i) {
        Sample x;
        x.kind = i & 1 ? kAttention : kUpload;
        x.storage = 1;
        x.n_tokens = 1;
        x.rows = 1000 + i;
        x.split_k = 4;
        x.groups = 96;
        x.gpu_valid = 1;
        x.gpu_ticks[0] = 1000 * (i + 1);
        x.gpu_ticks[1] = 10;
        x.gpu_ticks[2] = 5;
        for (uint32_t span = 0; span < kCpuSpanCount; ++span) {
            x.cpu_ns[span] = 1000 * (span + 1);
        }
        r.record(x);
    }
    assert(r.recorded() == 6 && r.stored() == 4 && r.capacity() == 4);
    r.counters.calls[kUpload] = 3;
    r.counters.calls[kAttention] = 3;

    // Scratch shape counters distinguish cold growth from equal/smaller warm calls.
    rkva_kvarn_bulk::Scratch scratch;
    for (unsigned i = 0; i < 3; ++i) {
        rkva_kvarn_bulk::ScratchCall call(scratch, 256, RKVA_WIRE_FP8);
        scratch.output_f16.resize(256 * 24 * 256);
        scratch.lse.resize(256 * 24);
        scratch.segments.clear();
        scratch.segments.resize(32);
    }
    assert(scratch.shapes[2][256].calls == 3);
    assert(scratch.shapes[2][256].growth_calls == 1);
    assert(scratch.shapes[2][256].reallocations == 3);
    {
        rkva_kvarn_bulk::ScratchCall call(scratch, 1, RKVA_WIRE_FP8);
        scratch.output_f16.resize(24 * 256);
        scratch.lse.resize(24);
        scratch.segments.resize(1);
    }
    assert(scratch.shapes[2][1].growth_calls == 0);
    assert(scratch.json().find("\"growth_calls\":0") != std::string::npos);
    const auto capacity_before_reset = scratch.capacities();
    scratch.reset_counters();
    assert(scratch.capacities() == capacity_before_reset);
    assert(scratch.json() == "[]");

    BulkEnvelopeSample b;
    b.cycle_id = 123;
    b.session_id = 7;
    b.layer = 3;
    b.wire = RKVA_WIRE_FP8;
    b.gpu_unpack_ticks = 20;
    b.gpu_pack_ticks = 10;
    b.segment_count = 4;
    b.total_queries = 32;
    b.gpu_valid = 1;
    b.gpu_upload_ticks = 100;
    b.gpu_pass1_ticks = 400;
    b.gpu_reduce_ticks = 50;
    b.gpu_copy_ticks = 30;
    b.gpu_total_ticks = 650;
    b.cpu_ns[kCpuValidate] = 15000;
    b.cpu_ns[kCpuAlloc] = 5000;
    b.cpu_ns[kCpuMapCopy] = 120000;
    b.cpu_ns[kCpuRecord] = 80000;
    b.cpu_ns[kCpuSubmit] = 20000;
    b.cpu_ns[kCpuFenceWait] = 900000;
    b.cpu_ns[kCpuReadback] = 40000;
    b.upload_bytes = 65536;
    b.query_bytes = 32768;
    b.output_bytes = 32768;
    r.record_bulk(b);
    assert(r.bulk_recorded() == 1 && r.bulk_stored() == 1);
    r.counters.bulk_calls = 1;
    r.counters.bulk_segments = 4;
    r.counters.bulk_queries = 32;

    Header h;
    h.profiling = true;
    h.q4_kernel_mode = 1;
    h.gpu_timestamps = true;
    h.gpu_timestamp_frequency = 1000000; // 1 tick == 1 us
    h.gpu_timestamp_status = "ready \"quoted\"\n";
    h.q4_optimized_available = true;
    h.q4_optimized_status = "ready";
    h.optimized_available[kStorageQ4] = true;
    h.optimized_status[kStorageQ4] = "ready";
    h.optimized_status[kStorageKvarn4] = "kvarn4 optimized pipeline state: 0x80070057";
    r.counters.optimized_by_storage[kStorageKvarn4] = 5;
    r.counters.reference_by_storage[kStorageF16] = 2;
    h.wave_lane_min = 32;
    h.wave_lane_max = 64;
    h.kvarn4_prefix_available[0] = true;
    h.kvarn4_prefix_status[0] = "ready";
    h.kvarn4_prefix_status[1] = "kvarn4 optimized serial-prefix pipeline state: 0x80070057";
    h.q4_mq_available[1] = true;
    h.q4_mq_status[0] = "q4 multi-query pipeline state: 0x80070057";
    h.q4_mq_status[1] = "ready";
    r.counters.q4_mq_calls[1] = 7;
    h.fmt_mq_available[kStorageF16][0] = true;
    h.fmt_mq_available[kStorageF16][1] = true;
    h.fmt_mq_status[kStorageF16][0] = "ready";
    h.fmt_mq_status[kStorageF16][1] = "ready";
    h.fmt_mq_available[kStorageQ8][1] = true;
    h.fmt_mq_status[kStorageQ8][0] = "q8 multi-query pipeline state: 0x80070057";
    h.fmt_mq_status[kStorageQ8][1] = "ready";
    r.counters.fmt_mq_calls[kStorageF16][0] = 3;
    r.counters.fmt_mq_calls[kStorageQ8][1] = 11;
    h.kvarn4_mq_status = "requires wave lanes >= 64 (reported min 32)";
    h.kvarn4_mq4_available = true;
    h.kvarn4_mq4_status = "ready";
    h.kvarn4_decode_available = true;
    h.kvarn4_decode_status = "ready";
    r.counters.kvarn4_mq_calls = 13;
    r.counters.kvarn4_mq4_calls = 5;
    r.counters.kvarn4_decode_calls = 7;
    h.memory.app_available = true;
    h.memory.app_usage_bytes = 123456789;
    h.memory.app_limit_bytes = 5ull * 1024 * 1024 * 1024;
    h.memory.gpu_local_available = true;
    h.memory.gpu_local_usage_bytes = 1075838976;
    h.memory.gpu_local_budget_bytes = 3985637376;
    const std::string json = r.json(h);
    assert(json.find("\"app_limit_bytes\":5368709120") != std::string::npos);
    assert(json.find("\"gpu_local_usage_bytes\":1075838976") != std::string::npos);
    assert(json.find("\"kvarn4_mq_qualified_precision_tail_tokens\":[128]") != std::string::npos);
    assert(
        json.find("\"kvarn4_multi_query\":{\"nq_min\":2,\"nq_max\":9,"
                  "\"descriptor_revision\":3,\"required_wave_lanes\":64,\"mode_map\":\"1:"
                  "kvarn4_mq4_balanced 2:kvarn4_mq2_fine 3:per-query "
                  "kvarn4_gqa_coarse\",\"mq2\":{\"available\":false,\"status\":\"requires "
                  "wave lanes >= 64 (reported min 32)\",\"calls\":13},\"mq4\":{\"available\":true,"
                  "\"status\":\"ready\",\"calls\":5}}") != std::string::npos);
    assert(json.find("\"kvarn4_decode\":{\"nq\":1,"
                     "\"descriptor_revision\":3,\"required_wave_lanes\":64,"
                     "\"mode_map\":\"1:kvarn4_decode_balanced 2:kvarn4_decode_fine "
                     "3:per-query kvarn4_gqa_coarse\","
                     "\"available\":true,\"status\":\"ready\",\"calls\":7}") != std::string::npos);
    assert(json.find("\"format_multi_query\":{\"nq_min\":2,\"nq_max\":9,\"tile_rows\":64,"
                     "\"mode_map\":\"1:<fmt>_mq4_balanced 2:<fmt>_mq2_fine 3:per-query "
                     "<fmt>_gqa_coarse\",\"f16\":{\"mq2\":{\"available\":true,\"status\":"
                     "\"ready\",\"calls\":3},\"mq4\":{\"available\":true,\"status\":\"ready\","
                     "\"calls\":0}},\"q8_0\":{\"mq2\":{\"available\":false,\"status\":\"q8 "
                     "multi-query pipeline state: 0x80070057\",\"calls\":0},\"mq4\":{"
                     "\"available\":true,\"status\":\"ready\",\"calls\":11}}}") !=
           std::string::npos);
    assert(json.find("\"q4_multi_query\":{\"nq_min\":2,\"nq_max\":9,\"tile_rows\":64,"
                     "\"mode_map\":\"1:q4_mq4_balanced 2:q4_mq2_fine 3:per-query "
                     "q4_gqa_coarse\",\"mq2\":{\"available\":false,\"status\":\"q4 "
                     "multi-query pipeline state: 0x80070057\",\"calls\":0},\"mq4\":{"
                     "\"available\":true,\"status\":\"ready\",\"calls\":7}}") != std::string::npos);
    assert(json.find("\"kvarn4_serial_prefix\":{\"reference\":{\"available\":true,\"status\":"
                     "\"ready\"},\"optimized\":{\"available\":false,\"status\":\"kvarn4 "
                     "optimized serial-prefix pipeline state: 0x80070057\"}}") !=
           std::string::npos);
    assert(json.find("\"schema\":\"rkva-range-profile-v1\"") != std::string::npos);
    assert(json.find("\"samples_overwritten\":2") != std::string::npos);
    assert(json.find("upload/q4_0/nq1/r257-1k/warm") != std::string::npos);
    assert(json.find("attention/q4_0/reference/nq1/r257-1k/warm") != std::string::npos);
    assert(json.find("\"readback_copy\"") != std::string::npos);
    assert(json.find("\"fast_nq_max\":9") != std::string::npos);
    assert(json.find("\"kernel_mode_scope\":\"legacy name q4_kernel_mode") != std::string::npos);
    assert(json.find("\"q4_0\":{\"available\":true,\"status\":\"ready\",\"optimized_calls\":0") !=
           std::string::npos);
    assert(
        json.find(
            "\"kvarn4\":{\"available\":false,\"status\":\"kvarn4 optimized pipeline "
            "state: 0x80070057\",\"optimized_calls\":5,\"reference_calls\":0,\"fast_nq_max\":0}") !=
        std::string::npos);
    assert(json.find("\"f16\":{\"available\":false,\"status\":\"\",\"optimized_calls\":0,"
                     "\"reference_calls\":2,\"fast_nq_max\":0}") != std::string::npos);
    assert(json.find("\\\"quoted\\\"\\u000a") != std::string::npos);
    assert(json.find("\"bulk\":{") != std::string::npos);
    assert(json.find("\"cycle_id\":123,\"session\":7,\"layer\":3,\"wire\":4,\"nq\":32") !=
           std::string::npos);
    assert(json.find("\"gpu_unpack_wht_us\":{\"n\":1,\"mean\":20.000") != std::string::npos);
    assert(json.find("\"gpu_pack_us\":{\"n\":1,\"mean\":10.000") != std::string::npos);
    assert(json.find("\"p95\":") != std::string::npos);
    assert(json.find("\"bulk_calls\":1") != std::string::npos);
    assert(json.find("\"bulk_segments\":4") != std::string::npos);
    assert(json.find("\"bulk_queries\":32") != std::string::npos);
    assert(json.find("\"admission_cpu_us\":{\"n\":1,\"mean\":15.000") != std::string::npos);
    assert(json.find("\"gpu_total_us\":{\"n\":1,\"mean\":650.000") != std::string::npos);
    assert(json.find("\"queue_wait_us\":{\"n\":1,\"mean\":250.000") != std::string::npos);
    assert(json.find("\"upload_bytes\":{\"n\":1,\"mean\":65536.000") != std::string::npos);
    assert(json.find("\"timing_breakdown_notes\":{") != std::string::npos);
    std::printf("%s\n", json.c_str());

    // A long decode suffix evicts prefill from the global ring but must not
    // erase either independent phase distribution or its correlation IDs.
    Recorder phase_recorder;
    phase_recorder.configure(4);
    phase_recorder.record_bulk(b);
    BulkEnvelopeSample verification = b;
    verification.total_queries = 4;
    verification.gpu_total_ticks = 333;
    phase_recorder.record_bulk(verification);
    BulkEnvelopeSample decode = b;
    decode.total_queries = 1;
    decode.gpu_total_ticks = 2;
    for (unsigned i = 0; i < 20; ++i)
        phase_recorder.record_bulk(decode);
    const std::string phase_json = phase_recorder.json(h);
    const auto prefill_pos = phase_json.find("\"prefill\":{");
    const auto verification_pos = phase_json.find("\"verification\":{");
    assert(prefill_pos != std::string::npos && verification_pos != std::string::npos);
    assert(phase_json.find("\"gpu_total_us\":{\"n\":1,\"mean\":650.000", prefill_pos) !=
           std::string::npos);
    assert(phase_json.find("\"gpu_total_us\":{\"n\":1,\"mean\":333.000", verification_pos) !=
           std::string::npos);
    assert(phase_json.find("\"recent\":[{\"cycle_id\":123", prefill_pos) != std::string::npos);

    // Reset clears samples and counters but keeps the reserved ring.
    r.reset();
    assert(r.recorded() == 0 && r.stored() == 0 && r.capacity() == 4);
    assert(r.bulk_recorded() == 0 && r.bulk_stored() == 0);
    assert(r.counters.calls[kAttention] == 0);
    assert(r.counters.bulk_calls == 0);
    assert(r.counters.kvarn4_decode_calls == 0);
    const std::string empty = Recorder().json(Header{});
    assert(empty.find("\"buckets\":{}") != std::string::npos);
    std::fprintf(stderr, "rkva_range_profile_cpu: ok\n");
    return 0;
}
