#pragma once
// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Small streaming SHA-256. Shared byte-for-byte with the optional Bee client.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
namespace rkva_snapshot {
using Digest = std::array<uint8_t, 32>;
class Sha256 {
    uint32_t state_[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                          0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint8_t block_[64]{};
    size_t used_ = 0;
    uint64_t bytes_ = 0;
    static uint32_t rotr(uint32_t x, unsigned n) {
        return (x >> n) | (x << (32 - n));
    }
    void compress() {
        static constexpr uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2};
        uint32_t w[64];
        for (size_t i = 0; i < 16; ++i)
            w[i] = uint32_t(block_[4 * i]) << 24 | uint32_t(block_[4 * i + 1]) << 16 |
                   uint32_t(block_[4 * i + 2]) << 8 | block_[4 * i + 3];
        for (size_t i = 16; i < 64; ++i) {
            const auto a = w[i - 15], b = w[i - 2];
            w[i] = w[i - 16] + (rotr(a, 7) ^ rotr(a, 18) ^ (a >> 3)) + w[i - 7] +
                   (rotr(b, 17) ^ rotr(b, 19) ^ (b >> 10));
        }
        uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3], e = state_[4],
                 f = state_[5], g = state_[6], h = state_[7];
        for (size_t i = 0; i < 64; ++i) {
            const uint32_t t1 =
                h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
            const uint32_t t2 =
                (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

  public:
    void update(const void* data, size_t size) {
        const auto* p = static_cast<const uint8_t*>(data);
        bytes_ += size;
        while (size) {
            const size_t n = std::min(size, 64 - used_);
            std::memcpy(block_ + used_, p, n);
            used_ += n;
            p += n;
            size -= n;
            if (used_ == 64) {
                compress();
                used_ = 0;
            }
        }
    }
    Digest finish() {
        const uint64_t bits = bytes_ * 8;
        const uint8_t one = 0x80, zero = 0;
        update(&one, 1);
        while (used_ != 56)
            update(&zero, 1);
        uint8_t tail[8];
        for (size_t i = 0; i < 8; ++i)
            tail[7 - i] = uint8_t(bits >> (8 * i));
        update(tail, 8);
        Digest result{};
        for (size_t i = 0; i < 8; ++i)
            for (size_t j = 0; j < 4; ++j)
                result[4 * i + j] = uint8_t(state_[i] >> (24 - 8 * j));
        return result;
    }
};
inline Digest digest(const void* data, size_t size) {
    Sha256 h;
    h.update(data, size);
    return h.finish();
}
inline std::string hex(const uint8_t* digest) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string s(64, '0');
    for (size_t i = 0; i < 32; ++i) {
        s[2 * i] = digits[digest[i] >> 4];
        s[2 * i + 1] = digits[digest[i] & 15];
    }
    return s;
}
} // namespace rkva_snapshot
