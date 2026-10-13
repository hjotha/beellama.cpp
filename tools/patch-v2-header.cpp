// TEST fixture tool: rewrite KMS2 envelope fields INSIDE the llama state file
// and repair BOTH checksums (header_crc over the 104-byte header and the frame
// tail_crc over the whole frame) so the frame stays internally consistent.
//
// This is what lets a negative test reach the SPECIFIC position_split guard
// instead of being rejected earlier by generic checksum machinery.
//
// layout of a llama state slot file:
//   [0]      llama state file header + tensor directory
//   [qskm]   outer envelope: magic "QSKM" u32, version u32, frame_bytes u64
//   [qskm+16] KMS frame: 104-byte header, blob, descriptors+payloads, cells, tail_crc
#define XXH_INLINE_ALL
#include "../vendor/hash/xxhash/xxhash.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

static const uint32_t KMS2_MAGIC = 0x32534D4Bull; // "KMS2"
static const uint32_t QSKM_MAGIC = 0x4D4B5351ull; // "QSKM"

int main(int argc, char ** argv) {
    if (argc >= 4 && std::strcmp(argv[3], "corrupt") == 0) {
        if (argc < 7) {
            std::fprintf(stderr,
                "usage: %s <in> <out> corrupt <range_id> <layer_index> <byte_offset>\n",
                argv[0]);
            return 2;
        }
    } else if (argc < 8) {
        std::fprintf(stderr,
            "usage: %s <in> <out> <p> <c> <model_hex> <rope_hex> <version>\n"
            "       %s <in> <out> corrupt <range_id> <layer_index> <byte_offset>\n",
            argv[0], argv[0]);
        return 2;
    }
    std::vector<uint8_t> buf{};
    {
        FILE * f = std::fopen(argv[1], "rb");
        if (!f) { std::perror("open"); return 1; }
        std::fseek(f, 0, SEEK_END);
        const long n = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        buf.resize(size_t(n));
        if (std::fread(buf.data(), 1, buf.size(), f) != buf.size()) { std::perror("read"); return 1; }
        std::fclose(f);
    }

    auto rd32 = [&](size_t off) {
        return uint32_t(buf[off]) | (uint32_t(buf[off+1]) << 8) |
               (uint32_t(buf[off+2]) << 16) | (uint32_t(buf[off+3]) << 24);
    };
    auto rd64 = [&](size_t off) {
        uint64_t v = 0;
        for (int i = 7; i >= 0; --i) v = (v << 8) | buf[off + size_t(i)];
        return v;
    };

    // locate the outer envelope and the frame it declares
    size_t qskm = SIZE_MAX;
    for (size_t i = 0; i + 16 <= buf.size(); ++i) {
        if (rd32(i) == QSKM_MAGIC) { qskm = i; break; }
    }
    if (qskm == SIZE_MAX) { std::fprintf(stderr, "QSKM envelope not found\n"); return 1; }
    const uint64_t frame_bytes = rd64(qskm + 8);
    const size_t frame = qskm + 16;
    if (frame + frame_bytes > buf.size()) {
        std::fprintf(stderr, "declared frame exceeds file\n");
        return 1;
    }
    if (rd32(frame) != KMS2_MAGIC) { std::fprintf(stderr, "frame is not KMS2\n"); return 1; }

    // report the untouched checksums so a caller can confirm the fixture is sane
    std::vector<uint8_t> hdr(buf.begin() + frame, buf.begin() + frame + 104);
    std::memset(hdr.data() + 96, 0, 8);
    const uint64_t want_hdr = XXH64(hdr.data(), 104, 0);
    uint64_t have_hdr = 0;
    for (int i = 0; i < 8; ++i) have_hdr |= uint64_t(buf[frame + 96 + size_t(i)]) << (8*i);
    const size_t tail_off = frame + size_t(frame_bytes) - 8;
    uint64_t have_tail = 0;
    for (int i = 0; i < 8; ++i) have_tail |= uint64_t(buf[tail_off + size_t(i)]) << (8*i);
    const uint64_t want_tail = XXH64(buf.data() + frame, size_t(frame_bytes) - 8, 0);
    std::printf("qskm=%zu frame=%zu frame_bytes=%llu header_crc %s tail_crc %s\n",
                qskm, frame, (unsigned long long) frame_bytes,
                want_hdr == have_hdr ? "OK" : "MISMATCH",
                want_tail == have_tail ? "OK" : "MISMATCH");
    if (want_hdr != have_hdr || want_tail != have_tail) {
        std::fprintf(stderr, "refusing to patch an already inconsistent frame\n");
        return 1;
    }

    // ---- payload corruption mode -------------------------------------------
    // usage: <in> <out> corrupt <range_id> <layer_index> <byte_offset>
    // Flips one payload byte inside the chosen RANGE and repairs that payload's
    // own XXH64 plus the frame tail_crc, so the frame stays self-consistent and
    // the ONLY difference from the pristine snapshot is the flipped KV byte.
    // This is what proves the overflow range is genuinely consumed after restore
    // instead of being carried along as inert bytes.
    if (std::strcmp(argv[3], "corrupt") == 0) {
        const uint32_t want_range = uint32_t(std::strtoul(argv[4], nullptr, 10));
        const uint32_t want_layer = uint32_t(std::strtoul(argv[5], nullptr, 10));
        const uint64_t byte_off   = std::strtoull(argv[6], nullptr, 10);
        const uint32_t n_layers2 = rd32(frame + 28);
        const size_t meta_bytes2 = rd64(frame + 40);
        size_t off2 = frame + 104 + meta_bytes2;
        bool done = false;
        for (uint32_t i = 0; i < n_layers2; ++i) {
            const uint32_t layer = rd32(off2);
            const uint32_t rng   = rd32(off2 + 101);
            const uint64_t pbytes = rd64(off2 + 85);
            uint64_t stored_sum = 0;
            for (int k = 0; k < 8; ++k) stored_sum |= uint64_t(buf[off2 + 93 + size_t(k)]) << (8*k);
            const size_t payload = off2 + 113;
            if (rng == want_range && layer == want_layer) {
                if (byte_off >= pbytes) {
                    std::fprintf(stderr, "byte offset beyond payload (%llu >= %llu)\n",
                                 (unsigned long long) byte_off, (unsigned long long) pbytes);
                    return 1;
                }
                buf[payload + size_t(byte_off)] ^= 0xFF;
                const uint64_t fresh = XXH64(buf.data() + payload, size_t(pbytes), 0);
                for (int k = 0; k < 8; ++k) buf[off2 + 93 + size_t(k)] = uint8_t(fresh >> (8*k));
                const uint64_t t2 = XXH64(buf.data() + frame, size_t(frame_bytes) - 8, 0);
                for (int k = 0; k < 8; ++k) buf[tail_off + size_t(k)] = uint8_t(t2 >> (8*k));
                std::printf("corrupted range=%u layer=%u byte=%llu (payload=%llu), "
                            "payload checksum + tail_crc repaired\n",
                            rng, layer, (unsigned long long) byte_off,
                            (unsigned long long) pbytes);
                done = true;
                break;
            }
            off2 += 113 + size_t(pbytes);
        }
        if (!done) {
            std::fprintf(stderr, "range/layer not found\n");
            return 1;
        }
        FILE * cf = std::fopen(argv[2], "wb");
        if (!cf) { std::perror("open out"); return 1; }
        std::fwrite(buf.data(), 1, buf.size(), cf);
        std::fclose(cf);
        return 0;
    }

    // apply the requested mutation
    const uint32_t p   = uint32_t(std::strtoul(argv[3], nullptr, 10));
    const uint32_t c   = uint32_t(std::strtoul(argv[4], nullptr, 10));
    const uint64_t mid = std::strtoull(argv[5], nullptr, 16);
    const uint64_t rid = std::strtoull(argv[6], nullptr, 16);
    const uint32_t ver = uint32_t(std::strtoul(argv[7], nullptr, 10));
    auto put32 = [&](size_t off, uint32_t v) {
        buf[off+0] = uint8_t(v); buf[off+1] = uint8_t(v >> 8);
        buf[off+2] = uint8_t(v >> 16); buf[off+3] = uint8_t(v >> 24);
    };
    auto put64 = [&](size_t off, uint64_t v) {
        for (int i = 0; i < 8; ++i) buf[off + size_t(i)] = uint8_t(v >> (8*i));
    };
    put32(frame + 4,  ver); // version
    put32(frame + 56, p);
    put32(frame + 60, c);
    put64(frame + 64, mid);
    put64(frame + 72, rid);

    // repair header_crc then tail_crc (both cover the mutated bytes)
    std::vector<uint8_t> h2(buf.begin() + frame, buf.begin() + frame + 104);
    std::memset(h2.data() + 96, 0, 8);
    const uint64_t crc2 = XXH64(h2.data(), 104, 0);
    for (int i = 0; i < 8; ++i) buf[frame + 96 + size_t(i)] = uint8_t(crc2 >> (8*i));
    const uint64_t tail2 = XXH64(buf.data() + frame, size_t(frame_bytes) - 8, 0);
    for (int i = 0; i < 8; ++i) buf[tail_off + size_t(i)] = uint8_t(tail2 >> (8*i));

    FILE * out = std::fopen(argv[2], "wb");
    if (!out) { std::perror("open out"); return 1; }
    std::fwrite(buf.data(), 1, buf.size(), out);
    std::fclose(out);
    std::printf("patched -> %s (version=%u P=%u C=%u model=%016llx rope=%016llx) checksums repaired\n",
                argv[2], ver, p, c, (unsigned long long) mid, (unsigned long long) rid);
    return 0;
}