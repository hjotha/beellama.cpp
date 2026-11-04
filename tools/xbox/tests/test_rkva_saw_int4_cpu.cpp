#include "../uwp/rkva/rkva-saw-int4.h"
#include "../uwp/rkva/rkva-kvarn-range.h"
#include "../../ggml/src/ggml-remote-attn/rkva-saw-int4-producer.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

int main() {
    std::cout << "=== Running RKVA SAW-INT4 C++ Unit Tests ===" << std::endl;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // 1. Test Self-Inverseness of bdr16_transform_256
    {
        std::vector<float> orig(256);
        std::vector<float> rot1(256);
        std::vector<float> rot2(256);
        for (int i = 0; i < 256; ++i) orig[i] = dist(rng);

        rkva_saw_int4::bdr16_transform_256(orig.data(), rot1.data());
        rkva_saw_int4::bdr16_transform_256(rot1.data(), rot2.data());

        float max_diff = 0.0f;
        for (int i = 0; i < 256; ++i) {
            float d = std::abs(orig[i] - rot2[i]);
            if (d > max_diff) max_diff = d;
        }
        std::cout << "[Test 1] bdr16_transform_256 Self-Inverse max_diff: " << max_diff << std::endl;
        assert(max_diff < 1e-6f);
    }

    // 2. Test Dot Product Preservation: (Q * R) . (K * R) == Q . K
    {
        std::vector<float> q(256), k(256);
        std::vector<float> q_rot(256), k_rot(256);
        for (int i = 0; i < 256; ++i) {
            q[i] = dist(rng);
            k[i] = dist(rng);
        }

        rkva_saw_int4::bdr16_transform_256(q.data(), q_rot.data());
        rkva_saw_int4::bdr16_transform_256(k.data(), k_rot.data());

        double dot_orig = 0.0;
        double dot_rot  = 0.0;
        for (int i = 0; i < 256; ++i) {
            dot_orig += static_cast<double>(q[i]) * static_cast<double>(k[i]);
            dot_rot  += static_cast<double>(q_rot[i]) * static_cast<double>(k_rot[i]);
        }

        float diff = static_cast<float>(std::abs(dot_orig - dot_rot));
        std::cout << "[Test 2] Dot Product Preserved: orig=" << dot_orig << " rot=" << dot_rot << " diff=" << diff << std::endl;
        assert(diff < 1e-5f);
    }

    // 3. Test Attention Output Des-Rotation Linearity: (Sum P_i * V_rot,i) * R == Sum P_i * V_i
    {
        const int T = 64;
        std::vector<std::vector<float>> V(T, std::vector<float>(256));
        std::vector<std::vector<float>> V_rot(T, std::vector<float>(256));
        std::vector<float> P(T);
        float p_sum = 0.0f;
        for (int t = 0; t < T; ++t) {
            P[t] = std::abs(dist(rng));
            p_sum += P[t];
            for (int i = 0; i < 256; ++i) V[t][i] = dist(rng);
            rkva_saw_int4::bdr16_transform_256(V[t].data(), V_rot[t].data());
        }
        for (int t = 0; t < T; ++t) P[t] /= p_sum;

        std::vector<float> O_orig(256, 0.0f);
        std::vector<float> O_rot(256, 0.0f);
        for (int t = 0; t < T; ++t) {
            for (int i = 0; i < 256; ++i) {
                O_orig[i] += P[t] * V[t][i];
                O_rot[i]  += P[t] * V_rot[t][i];
            }
        }

        std::vector<float> O_rec(256);
        rkva_saw_int4::bdr16_transform_256(O_rot.data(), O_rec.data());

        float max_diff = 0.0f;
        for (int i = 0; i < 256; ++i) {
            float d = std::abs(O_orig[i] - O_rec[i]);
            if (d > max_diff) max_diff = d;
        }
        std::cout << "[Test 3] Output Des-Rotation max_diff: " << max_diff << std::endl;
        assert(max_diff < 1e-6f);
    }

    // 4. Test Quantization + Dequantization Round-Trip
    {
        std::vector<float> src(256);
        for (int i = 0; i < 256; ++i) src[i] = dist(rng);
        // Inject large outlier channels (typical in LLMs)
        src[5] = 25.0f;
        src[48] = -32.0f;
        src[201] = 20.0f;

        block_saw_int4 block;
        rkva_saw_int4::quantize_row_saw_int4_single(src.data(), &block);

        std::vector<float> deq(256);
        rkva_saw_int4::dequantize_row_saw_int4_single(&block, deq.data());

        float mse = 0.0f;
        for (int i = 0; i < 256; ++i) {
            float err = src[i] - deq[i];
            mse += err * err;
        }
        mse /= 256.0f;
        std::cout << "[Test 4] SAW-INT4 MSE with outliers: " << mse << " (scale=" << block.scale << ", zero=" << block.zero << ")" << std::endl;
        assert(mse < 0.25f);
    }

    // 5. Test RKVA Layout-6 Descriptor, Producer, and Upload Validation
    {
        const rkva_kvarn_range_config desc {
            1024u, 512u, RKVA_RANGE_STORAGE_SAW_INT4, 6u,
            1u, 0u, 256u, 2u, 4u, 4u, 136u,
            0u, 0u, 0u, 0u, 16u, 1u, 1u
        };
        assert(rkva_kvarn_range::valid_saw_int4_descriptor(desc, 256));
        assert(rkva_kvarn_range::saw_int4_row_bytes(256, 2) == 272);

        rkva_saw_int4_producer producer(1024, 512, 2);
        std::vector<float> k_in(4 * 2 * 256), v_in(4 * 2 * 256);
        for (size_t i = 0; i < k_in.size(); ++i) {
            k_in[i] = dist(rng);
            v_in[i] = dist(rng);
        }
        std::vector<int32_t> pos = {1024, 1025, 1026, 1027};
        std::vector<uint8_t> payload;
        assert(producer.prepare(k_in.data(), v_in.data(), pos.data(), 4));
        assert(producer.append(k_in.data(), v_in.data(), pos.data(), 4, true, payload));
        assert(producer.rows() == 4);
        assert(payload.size() == sizeof(rkva_kvarn_range_upload) + 2 * 4 * 272);

        rkva_kvarn_range_upload up{};
        std::memcpy(&up, payload.data(), sizeof(up));
        assert(rkva_kvarn_range::valid_saw_int4_upload(up, 512, 0));
        assert(up.live_count == 4 && up.append_begin == 0 && up.append_rows == 4);
        assert(producer.can_rewind(2));
        assert(producer.rewind(2));
        assert(producer.rows() == 2);
        std::cout << "[Test 5] RKVA Layout-6 Descriptor & Producer verified (payload=" << payload.size() << " B)" << std::endl;
    }

    std::cout << "All C++ SAW-INT4 tests passed successfully!" << std::endl;
    return 0;
}
