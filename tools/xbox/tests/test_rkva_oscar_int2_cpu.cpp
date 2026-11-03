#include "rkva-oscar-int2.h"
#include "rkva-kvarn-range.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

// Standalone C++ validation test for RKVA OSCAR INT2 KV-cache quantization.

static void test_layout_and_sizes() {
    std::cout << "[Test C++ 1] Layout, Sizes & Protocol Descriptors..." << std::endl;
    static_assert(sizeof(block_oscar_int2) == 12, "block_oscar_int2 must be 12 bytes");
    static_assert(sizeof(rkva_oscar_int2_row) == 96, "rkva_oscar_int2_row must be 96 bytes");

    // Check descriptor
    rkva_kvarn_range_config cfg = {};
    cfg.storage = RKVA_RANGE_STORAGE_OSCAR_INT2;
    cfg.layout_version = 7;
    cfg.origin = 0;
    cfg.capacity = 4096;
    cfg.token_group = 1;
    cfg.record_dim = 0;
    cfg.head_dim = 256;
    cfg.head_slices = 2;
    cfg.key_bits = 2;
    cfg.value_bits = 2;
    cfg.record_bytes = 96;
    cfg.stage_groups = 0;
    cfg.tail_groups = 0;
    cfg.sink_tokens = 0;
    cfg.precision_tail_tokens = 0;
    cfg.wht_version = 0;
    cfg.domain = 1;
    cfg.record_policy = 1;

    assert(rkva_kvarn_range::valid_oscar_int2_descriptor(cfg, 256));
    assert(rkva_kvarn_range::oscar_int2_row_bytes(256, 8) == 8 * 96);

    std::cout << "  -> PASS: 96 bytes per 256-dim row verified (vs 136 bytes SAW-INT4, 272 bytes Q8_0, 512 bytes F16)." << std::endl;
}

static void test_dot_product_invariance() {
    std::cout << "[Test C++ 2] Mathematical Invariance Q' @ (K')^T == Q @ K^T..." << std::endl;
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> q(256), k(256);
    for (int i = 0; i < 256; ++i) {
        q[i] = dist(rng);
        k[i] = dist(rng);
    }

    // Generate random orthogonal matrix via Gram-Schmidt
    std::vector<float> M(256 * 256);
    for (int i = 0; i < 256; ++i) {
        for (int j = 0; j < 256; ++j) {
            M[i * 256 + j] = dist(rng);
        }
        for (int prev = 0; prev < i; ++prev) {
            float dot = 0.0f;
            for (int k_idx = 0; k_idx < 256; ++k_idx) {
                dot += M[i * 256 + k_idx] * M[prev * 256 + k_idx];
            }
            for (int k_idx = 0; k_idx < 256; ++k_idx) {
                M[i * 256 + k_idx] -= dot * M[prev * 256 + k_idx];
            }
        }
        float norm = 0.0f;
        for (int k_idx = 0; k_idx < 256; ++k_idx) {
            norm += M[i * 256 + k_idx] * M[i * 256 + k_idx];
        }
        norm = std::sqrt(norm);
        for (int k_idx = 0; k_idx < 256; ++k_idx) {
            M[i * 256 + k_idx] /= norm;
        }
    }

    // Rotate Q' = Q @ M, K' = K @ M
    std::vector<float> q_rot(256), k_rot(256);
    rkva_oscar_int2::rotate_256(q.data(), M.data(), q_rot.data());
    rkva_oscar_int2::rotate_256(k.data(), M.data(), k_rot.data());

    float dot_orig = 0.0f, dot_rot = 0.0f;
    for (int i = 0; i < 256; ++i) {
        dot_orig += q[i] * k[i];
        dot_rot += q_rot[i] * k_rot[i];
    }

    float diff = std::fabs(dot_orig - dot_rot);
    std::cout << "  -> dot_orig=" << dot_orig << ", dot_rot=" << dot_rot << ", diff=" << diff << std::endl;
    assert(diff < 1e-4f);
    std::cout << "  -> PASS: Dot product strictly invariant under orthogonal rotation." << std::endl;
}

static void test_quantization_and_dequantization() {
    std::cout << "[Test C++ 3] Quantization, Lloyd-Max Centroids & Zero-Centering..." << std::endl;
    std::mt19937 rng(12345);
    std::normal_distribution<float> dist(2.5f, 1.2f); // Non-zero mean to test zero-centering

    std::vector<float> src(256);
    for (int i = 0; i < 256; ++i) {
        src[i] = dist(rng);
    }
    // Add large outliers to test outlier clipping
    src[10] = 35.0f;
    src[120] = -40.0f;

    rkva_oscar_int2_row encoded;
    rkva_oscar_int2::quantize_row_oscar_int2_single(src.data(), encoded.blocks, 0.96f);

    std::vector<float> recovered(256);
    rkva_oscar_int2::dequantize_row_oscar_int2_single(encoded.blocks, recovered.data());

    // Compute MSE on non-outlier channels
    float mse = 0.0f;
    int count = 0;
    for (int i = 0; i < 256; ++i) {
        if (i != 10 && i != 120) {
            float err = src[i] - recovered[i];
            mse += err * err;
            count++;
        }
    }
    mse /= static_cast<float>(count);
    std::cout << "  -> Non-outlier channel MSE: " << mse << std::endl;
    assert(mse < 0.25f);
    std::cout << "  -> PASS: Quantization and dequantization accurately recovered." << std::endl;
}

int main() {
    std::cout << "=== RKVA OSCAR INT2 C++ Validation Suite ===" << std::endl;
    test_layout_and_sizes();
    test_dot_product_invariance();
    test_quantization_and_dequantization();
    std::cout << "=== ALL C++ TESTS PASSED ===" << std::endl;
    return 0;
}
