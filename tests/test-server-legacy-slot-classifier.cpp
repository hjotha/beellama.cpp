#include "../tools/server/server-legacy-slot-classifier.h"
#undef NDEBUG
#include <cassert>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <vector>

struct fixture {
    std::vector<uint8_t> bytes;
    void u32(uint32_t n) { for (int i=0; i<4; ++i) bytes.push_back(uint8_t(n >> (8*i))); }
    void u64(uint64_t n) { for (int i=0; i<8; ++i) bytes.push_back(uint8_t(n >> (8*i))); }
    void raw(const std::string & s) { bytes.insert(bytes.end(), s.begin(), s.end()); }
    void blob(const std::string & s) { u64(s.size()); raw(s); }
    void finish() { u64(XXH64(bytes.data(), bytes.size(), server_legacy_slot::detail::checksum_seed)); }
};

static fixture make_fixture(bool checkpoints) {
    fixture f;
    f.raw("LLAMASLT"); f.u32(2); f.u32(1);
    f.u32(1024); f.u32(0); f.u32(0); f.u32(1024);
    for (int i=0; i<5; ++i) f.u64(0);
    f.blob("identity"); f.blob("layout"); f.blob("");
    f.u32(0); f.u32(UINT32_MAX); f.u64(1);
    f.blob("abcd"); f.blob(std::string("\xd8\x3c\x14\xaf\x00\x00\x00\x00", 8) + "state-data"); f.blob(""); f.blob("");
    f.u64(checkpoints ? 1 : 0);
    if (checkpoints) {
        f.u64(1); f.u32(0); f.u32(0); f.u32(0); f.u32(0); f.u32(0);
        f.u64(0); f.u64(0); f.blob("layout"); f.blob("");
        for (int i=0; i<24; ++i) f.u32(0);
        f.u32(0); f.u32(0); f.bytes.push_back(0);
        f.blob("checkpoint"); f.blob(""); f.blob("");
    }
    f.finish();
    return f;
}
static void rewrite_checksum(std::vector<uint8_t> & b) {
    assert(b.size() >= 8);
    const auto hash = XXH64(b.data(), b.size() - 8, server_legacy_slot::detail::checksum_seed);
    for (int i = 0; i < 8; ++i) b[b.size()-8+i] = uint8_t(hash >> (i*8));
}
static void write(const std::filesystem::path & path, const std::vector<uint8_t> & bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char *>(bytes.data()), std::streamsize(bytes.size()));
    assert(out.good());
}
int main(int argc, char ** argv) {
    if (argc == 2) {
        const bool accepted = server_legacy_slot::is_proven_local(argv[1], UINT64_MAX);
        std::puts(accepted ? "LLAMASLT accepted" : "LLAMASLT rejected");
        return accepted ? 0 : 1;
    }
    if (argc == 3 && std::string(argv[1]) == "--sequence") {
        uint32_t tokens = 0;
        const bool accepted = server_legacy_slot::is_local_sequence(argv[2], UINT64_MAX, &tokens);
        std::printf("%s tokens=%u\n", accepted ? "sequence accepted" : "sequence rejected", tokens);
        return accepted ? 0 : 1;
    }
    namespace fs = std::filesystem;
    const auto root = fs::temp_directory_path() / ("beellama-slot-classifier-" + std::to_string(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    assert(fs::create_directory(root));
    const auto path = root / "slot.bin";
    // Sparse file proves ordinary snapshots are rejected before the 2 GiB scan.
    const auto large_unknown = root / "unrelated.snapshot";
    {
        std::ofstream out(large_unknown, std::ios::binary);
        out.write("NOT-A-SLOT!!", 12);
        out.seekp((std::streamoff(2) << 30) + 511);
        out.put('x');
        assert(out.good());
    }
    uint64_t inspected = UINT64_MAX;
    assert(!server_legacy_slot::is_proven_local(large_unknown.string(), UINT64_MAX, &inspected));
    assert(inspected == 12);
    {
        std::fstream out(large_unknown, std::ios::binary | std::ios::in | std::ios::out);
        out.write("LLAMASLT\x03\x00\x00\x00", 12);
        assert(out.good());
    }
    assert(!server_legacy_slot::is_proven_local(large_unknown.string(), UINT64_MAX, &inspected));
    assert(inspected == 12);
    fs::remove(large_unknown);
    auto valid = make_fixture(true).bytes;
    write(path, valid);
    assert(server_legacy_slot::is_proven_local(path.string(), valid.size()));
    uint64_t classified_tokens = 0;
    assert(server_legacy_slot::is_proven_local(path.string(), valid.size(), nullptr, &classified_tokens));
    assert(classified_tokens == 1);
    assert(!server_legacy_slot::is_proven_local(path.string(), valid.size()-1));
    for (size_t i = 0; i < valid.size(); ++i) {
        auto truncated = std::vector<uint8_t>(valid.begin(), valid.begin() + i);
        write(path, truncated);
        assert(!server_legacy_slot::is_proven_local(path.string(), UINT64_MAX));
    }
    auto corrupt = valid; corrupt[80] ^= 1;
    write(path, corrupt); assert(!server_legacy_slot::is_proven_local(path.string(), UINT64_MAX));
    auto spoof = valid; spoof[0] = 'X'; rewrite_checksum(spoof);
    write(path, spoof); assert(!server_legacy_slot::is_proven_local(path.string(), UINT64_MAX));
    spoof = valid; spoof[8] = 3; rewrite_checksum(spoof);
    write(path, spoof); assert(!server_legacy_slot::is_proven_local(path.string(), UINT64_MAX));
    spoof = valid; spoof[12] = 6; rewrite_checksum(spoof);
    write(path, spoof); assert(!server_legacy_slot::is_proven_local(path.string(), UINT64_MAX));
    spoof = make_fixture(false).bytes;
    for (size_t i = spoof.size() - 16; i < spoof.size() - 8; ++i) spoof[i] = 0xff;
    rewrite_checksum(spoof);
    write(path, spoof); assert(!server_legacy_slot::is_proven_local(path.string(), UINT64_MAX));
    spoof = valid;
    auto target_data = std::search(spoof.begin(), spoof.end(), "state-data", "state-data"+10);
    assert(target_data != spoof.end()); *(target_data - 8) ^= 1; rewrite_checksum(spoof);
    write(path, spoof); assert(!server_legacy_slot::is_proven_local(path.string(), UINT64_MAX));
    spoof = valid; for (size_t i = 72; i < 80; ++i) spoof[i] = 0xff; rewrite_checksum(spoof);
    write(path, spoof); assert(!server_legacy_slot::is_proven_local(path.string(), UINT64_MAX));
    spoof = valid; spoof.resize(spoof.size() + 4); rewrite_checksum(spoof);
    write(path, spoof); assert(!server_legacy_slot::is_proven_local(path.string(), UINT64_MAX));
    auto forged = make_fixture(false);
    // A valid checksum does not make an envelope containing native bytes safe.
    auto pos = std::search(forged.bytes.begin(), forged.bytes.end(), "state-data", "state-data"+10);
    assert(pos != forged.bytes.end()); std::copy_n("RKVK", 4, pos); rewrite_checksum(forged.bytes);
    write(path, forged.bytes); assert(!server_legacy_slot::is_proven_local(path.string(), UINT64_MAX));
    // A structurally valid file whose marker crosses the 64 KiB read boundary.
    fixture large;
    large.raw("LLAMASLT"); large.u32(2); large.u32(1);
    for (int i=0; i<4; ++i) large.u32(0);
    for (int i=0; i<5; ++i) large.u64(0);
    large.blob("identity"); large.blob("layout"); large.blob("");
    large.u32(0); large.u32(UINT32_MAX); large.u64(1);
    large.blob("abcd"); large.blob(std::string("\xd8\x3c\x14\xaf\x00\x00\x00\x00", 8) + std::string(65536, 'x'));
    large.blob(""); large.blob(""); large.u64(0); large.finish();
    write(path, large.bytes); assert(server_legacy_slot::is_proven_local(path.string(), UINT64_MAX));
    std::copy_n("RKVK", 4, large.bytes.begin() + 65534); rewrite_checksum(large.bytes);
    write(path, large.bytes); assert(!server_legacy_slot::is_proven_local(path.string(), UINT64_MAX));
    write(path, std::vector<uint8_t>{0x67, 0x67, 0x6d, 0x6c});
    assert(!server_legacy_slot::is_proven_local(path.string(), UINT64_MAX));

    // Raw qsgg sequence snapshots: local only when the complete file has no
    // RKVK trailer. Token count is exposed for sidecar binding.
    auto raw_sequence = [](uint32_t tokens, size_t state_bytes, const std::string & marker) {
        std::vector<uint8_t> bytes;
        auto u32 = [&](uint32_t value) {
            for (int i = 0; i < 4; ++i) bytes.push_back(uint8_t(value >> (8*i)));
        };
        u32(0x67677371u); u32(4); u32(tokens);
        for (uint32_t i = 0; i < tokens; ++i) u32(i);
        bytes.insert(bytes.end(), state_bytes, uint8_t('s'));
        bytes.insert(bytes.end(), marker.begin(), marker.end());
        return bytes;
    };
    uint32_t sequence_tokens = 0;
    auto sequence = raw_sequence(3, 64, "");
    write(path, sequence);
    assert(server_legacy_slot::is_local_sequence(path.string(), UINT64_MAX, &sequence_tokens));
    assert(sequence_tokens == 3);
    sequence = raw_sequence(3, 64, "RKVK");
    write(path, sequence);
    assert(!server_legacy_slot::is_local_sequence(path.string(), UINT64_MAX, &sequence_tokens));
    assert(sequence_tokens == 0);
    sequence = raw_sequence(3, 64, "");
    sequence[0] = 'X';
    write(path, sequence);
    assert(!server_legacy_slot::is_local_sequence(path.string(), UINT64_MAX));
    // Header-only and token-count-overflow shapes fail closed.
    write(path, std::vector<uint8_t>(sequence.begin(), sequence.begin() + 12));
    assert(!server_legacy_slot::is_local_sequence(path.string(), UINT64_MAX));
    sequence = raw_sequence(3, 64, "");
    sequence[8] = 0xff; sequence[9] = 0xff; sequence[10] = 0xff; sequence[11] = 0x7f;
    write(path, sequence);
    assert(!server_legacy_slot::is_local_sequence(path.string(), UINT64_MAX));
#ifndef _WIN32
    const auto link = root / "link";
    fs::create_symlink(path, link);
    assert(!server_legacy_slot::is_proven_local(link.string(), UINT64_MAX));
#endif
    fs::remove_all(root);
    std::puts("legacy LLAMASLT file classification: PASS");
}
