#include "ggml.h"
#include "llama-kvarn.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    const auto layout = llama_kvarn_make_layout(128, 128, 4, 4);
    const auto record = llama_kvarn_make_record_layout(128, 4, false);
    if (record.record_bytes != 8960 || record.scale_off != 8192 || record.zp_off != 8448 ||
        record.other_off != 8704)
        return 3;
    std::vector<float> k(16384), v(16384), kd(16384), vd(16384);
    for (size_t i = 0; i < k.size(); ++i) {
        k[i] = std::sin(float(i) * 0.017f) * 2 + std::cos(float(i) * 0.003f);
        v[i] = std::cos(float(i) * 0.019f) * 1.4f - std::sin(float(i) * 0.007f);
    }
    std::vector<uint8_t> tile(layout.tile_bytes), packed(17920);
    llama_kvarn_quantize_k_tile(k.data(), 16, 4, layout, tile.data());
    llama_kvarn_quantize_v_tile(v.data(), 16, 4, layout, tile.data());
    llama_kvarn_dequantize_k_tile(tile.data(), 4, layout, kd.data());
    llama_kvarn_dequantize_v_tile(tile.data(), 4, layout, vd.data());
    std::memcpy(packed.data(), tile.data(), 8960);
    auto* vo = packed.data() + 8960;
    std::memcpy(vo, tile.data() + layout.v_payload_off, 8192);
    std::memcpy(vo + 8192, tile.data() + layout.v_s_row_off, 256);
    std::memcpy(vo + 8448, tile.data() + layout.v_zp_off, 256);
    std::memcpy(vo + 8704, tile.data() + layout.v_s_col_off, 256);
    auto decode = [](const uint8_t* b, int i) {
        uint16_t a, z, o;
        int row = i / 128, col = i % 128;
        std::memcpy(&a, b + 8192 + row * 2, 2);
        std::memcpy(&z, b + 8448 + row * 2, 2);
        std::memcpy(&o, b + 8704 + col * 2, 2);
        unsigned q = (b[i / 2] >> ((i & 1) * 4)) & 15;
        return (float(q) * ggml_fp16_to_fp32(a) + ggml_fp16_to_fp32(z)) * ggml_fp16_to_fp32(o);
    };
    float error = 0;
    for (int i = 0; i < 16384; ++i) {
        error = std::max(error, std::abs(decode(packed.data(), i) - kd[i]));
        error = std::max(error, std::abs(decode(vo, i) - vd[i]));
    }
    std::ofstream input(std::string(argv[1]) + "/bee-kvarn4-source.bin", std::ios::binary);
    input.write((char*)k.data(), k.size() * 4);
    input.write((char*)v.data(), v.size() * 4);
    input.close();
    std::ofstream out(std::string(argv[1]) + "/bee-kvarn4-records.bin", std::ios::binary);
    out.write((char*)packed.data(), packed.size());
    out.close();
    std::ofstream ref(std::string(argv[1]) + "/bee-kvarn4-dequant.bin", std::ios::binary);
    ref.write((char*)kd.data(), kd.size() * 4);
    ref.write((char*)vd.data(), vd.size() * 4);
    ref.close();
    std::vector<float> wh(256);
    for (int i = 0; i < 256; ++i)
        wh[i] = std::sin(i * .123f);
    auto original = wh;
    auto wht = [](std::vector<float>& x) {
        llama_kvarn_hadamard_128(x.data());
        llama_kvarn_hadamard_128(x.data() + 128);
        for (int i = 0; i < 128; ++i) {
            float a = x[i], b = x[i + 128];
            x[i] = (a + b) * .7071067811865475f;
            x[i + 128] = (a - b) * .7071067811865475f;
        }
    };
    wht(wh);
    std::ofstream wf(std::string(argv[1]) + "/bee-kvarn4-wht.bin", std::ios::binary);
    wf.write((char*)original.data(), 1024);
    wf.write((char*)wh.data(), 1024);
    wf.close();
    wht(wh);
    float we = 0;
    for (int i = 0; i < 256; ++i)
        we = std::max(we, std::abs(wh[i] - original[i]));
    std::printf(
        "{\"record_bytes\":8960,\"record_decode_max_abs\":%.9g,\"wht_roundtrip_max_abs\":%.9g}\n",
        error, we);
    return error > 1e-6f || we > 1e-5f;
}
