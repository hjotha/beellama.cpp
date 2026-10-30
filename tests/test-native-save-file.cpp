// CPU regression for native slot publication: UTF-8 names, replacement and
// failed-publication preservation. The Windows branch is compiled by MSVC;
// this host test exercises the shared POSIX branch and path handling.
#include "../src/llama-native-save-file.h"
#include "../src/llama-state-framing.h"

#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace fs = std::filesystem;

static void write_file(const fs::path & path, const std::string & contents) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    assert(out && out.write(contents.data(), std::streamsize(contents.size())));
    out.flush();
    assert(out.good());
}

static void write_bytes(const fs::path & path, const std::vector<uint8_t> & contents) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    assert(out && out.write(reinterpret_cast<const char *>(contents.data()),
                            std::streamsize(contents.size())));
    out.flush();
    assert(out.good());
}

static std::string read_file(const fs::path & path) {
    std::ifstream in(path, std::ios::binary);
    assert(in);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

int main() {
    const uint8_t sckp[4] = {0x53, 0x43, 0x4b, 0x50};
    const uint8_t junk[4] = {1, 2, 3, 4};
    assert(llama_state_framing::classify(nullptr, 0) ==
           llama_state_framing::trailing::none);
    assert(llama_state_framing::classify(sckp, sizeof(sckp)) ==
           llama_state_framing::trailing::slot_checkpoint);
    assert(llama_state_framing::classify(junk, sizeof(junk)) ==
           llama_state_framing::trailing::unknown);
    assert(llama_state_framing::classify(sckp, 3) ==
           llama_state_framing::trailing::truncated);

    const fs::path root = fs::temp_directory_path() /
        (std::string("rkva-native-save-") + std::to_string((unsigned long long) ggml_time_us()));
    std::error_code ec;
    fs::create_directories(root, ec);
    assert(!ec);
    const fs::path destination = root / fs::u8path("slot-\xce\xb4-\xe6\x96\x87\xe4\xbb\xb6.bin");
    const auto destination_u8 = destination.u8string();
    const std::string destination_utf8(
        reinterpret_cast<const char *>(destination_u8.data()), destination_u8.size());
    write_file(destination, "old-pair");
    assert(read_file(destination) == "old-pair");

    const std::string temporary = llama_native_save::temporary_path(destination_utf8.c_str());
    write_file(fs::u8path(temporary), "new-pair");
    assert(llama_native_save::publish(temporary, destination_utf8.c_str()));
    assert(!fs::exists(fs::u8path(temporary)));
    assert(read_file(destination) == "new-pair");

    const std::string second = llama_native_save::temporary_path(destination_utf8.c_str());
    write_file(fs::u8path(second), "newer-pair-\xe6\x96\x87\xe4\xbb\xb6");
    assert(llama_native_save::publish(second, destination_utf8.c_str()));
    assert(read_file(destination) == "newer-pair-\xe6\x96\x87\xe4\xbb\xb6");

    // A missing staging file must not destroy the published destination.
    const std::string missing = llama_native_save::temporary_path(destination_utf8.c_str());
    assert(!llama_native_save::publish(missing, destination_utf8.c_str()));
    assert(read_file(destination) == "newer-pair-\xe6\x96\x87\xe4\xbb\xb6");

    // The guard removes an uncommitted staging file but never the destination,
    // including a real UTF-8 path (the Windows branch uses DeleteFileW).
    const std::string guarded = llama_native_save::temporary_path(destination_utf8.c_str());
    {
        write_file(fs::u8path(guarded), "orphan");
        llama_native_save::remove_guard guard{guarded};
    }
    assert(!fs::exists(fs::u8path(guarded)));
    assert(read_file(destination) == "newer-pair-\xe6\x96\x87\xe4\xbb\xb6");

    // The production key helper binds header/tokens/local state through the
    // RKVK end and excludes only the appended SCKP block.
    std::vector<uint8_t> local(37, 0x11);
    std::vector<uint8_t> owner(41, 0x22);
    std::vector<uint8_t> sckp_bytes(12, 0x33);
    std::vector<uint8_t> framed = local;
    framed.insert(framed.end(), owner.begin(), owner.end());
    framed.insert(framed.end(), sckp_bytes.begin(), sckp_bytes.end());
    const fs::path framed_path = root / fs::u8path("framing-\xce\xb4.bin");
    const auto framed_u8 = framed_path.u8string();
    const std::string framed_utf8(
        reinterpret_cast<const char *>(framed_u8.data()), framed_u8.size());
    write_bytes(framed_path, framed);
    const size_t prefix_end = local.size() + owner.size();
    uint8_t good_prefix_key[32] = {};
    {
        llama_file file(framed_utf8.c_str(), "rb");
        assert(llama_native_save::hash_file_prefix(file, prefix_end, good_prefix_key));
    }
    auto mutated_local = local;
    mutated_local[5] ^= 0x01;
    std::vector<uint8_t> mutated = mutated_local;
    mutated.insert(mutated.end(), owner.begin(), owner.end());
    mutated.insert(mutated.end(), sckp_bytes.begin(), sckp_bytes.end());
    write_bytes(framed_path, mutated);
    uint8_t bad_local_key[32] = {};
    {
        llama_file file(framed_utf8.c_str(), "rb");
        assert(llama_native_save::hash_file_prefix(file, prefix_end, bad_local_key));
    }
    assert(std::memcmp(good_prefix_key, bad_local_key, sizeof(good_prefix_key)) != 0);
    // Restoring the local bytes restores the key; SCKP remains excluded.
    write_bytes(framed_path, framed);
    uint8_t restored_prefix_key[32] = {};
    {
        llama_file file(framed_utf8.c_str(), "rb");
        assert(llama_native_save::hash_file_prefix(file, prefix_end, restored_prefix_key));
    }
    assert(std::memcmp(good_prefix_key, restored_prefix_key, sizeof(good_prefix_key)) == 0);
    auto mutated_trailer = owner;
    mutated_trailer[10] ^= 0x01;
    std::vector<uint8_t> mutated_frame = local;
    mutated_frame.insert(mutated_frame.end(), mutated_trailer.begin(), mutated_trailer.end());
    mutated_frame.insert(mutated_frame.end(), sckp_bytes.begin(), sckp_bytes.end());
    write_bytes(framed_path, mutated_frame);
    uint8_t bad_trailer_key[32] = {};
    {
        llama_file file(framed_utf8.c_str(), "rb");
        assert(llama_native_save::hash_file_prefix(file, prefix_end, bad_trailer_key));
    }
    assert(std::memcmp(good_prefix_key, bad_trailer_key, sizeof(good_prefix_key)) != 0);

    fs::remove_all(root, ec);
    assert(!ec);
    std::printf("native save publication CPU regression: ok\n");
    return 0;
}
