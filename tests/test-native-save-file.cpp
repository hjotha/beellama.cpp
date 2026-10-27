// CPU regression for native slot publication: UTF-8 names, replacement and
// failed-publication preservation. The Windows branch is compiled by MSVC;
// this host test exercises the shared POSIX branch and path handling.
#include "../src/llama-native-save-file.h"

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

static std::string read_file(const fs::path & path) {
    std::ifstream in(path, std::ios::binary);
    assert(in);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

int main() {
    const fs::path root = fs::temp_directory_path() /
        (std::string("rkva-native-save-") + std::to_string((unsigned long long) ggml_time_us()));
    std::error_code ec;
    fs::create_directories(root, ec);
    assert(!ec);
    const fs::path destination = root / fs::u8path("slot-\xce\xb4-\xe6\x96\x87\xe4\xbb\xb6.bin");
    write_file(destination, "old-pair");
    assert(read_file(destination) == "old-pair");

    const std::string temporary = llama_native_save::temporary_path(destination.string().c_str());
    write_file(fs::u8path(temporary), "new-pair");
    assert(llama_native_save::publish(temporary, destination.string().c_str()));
    assert(!fs::exists(fs::u8path(temporary)));
    assert(read_file(destination) == "new-pair");

    const std::string second = llama_native_save::temporary_path(destination.string().c_str());
    write_file(fs::u8path(second), "newer-pair-\xe6\x96\x87\xe4\xbb\xb6");
    assert(llama_native_save::publish(second, destination.string().c_str()));
    assert(read_file(destination) == "newer-pair-\xe6\x96\x87\xe4\xbb\xb6");

    // A missing staging file must not destroy the published destination.
    const std::string missing = llama_native_save::temporary_path(destination.string().c_str());
    assert(!llama_native_save::publish(missing, destination.string().c_str()));
    assert(read_file(destination) == "newer-pair-\xe6\x96\x87\xe4\xbb\xb6");

    // The guard removes an uncommitted staging file but never the destination.
    {
        const std::string guarded = llama_native_save::temporary_path(destination.string().c_str());
        write_file(fs::u8path(guarded), "orphan");
        llama_native_save::remove_guard guard{guarded};
    }
    assert(read_file(destination) == "newer-pair-\xe6\x96\x87\xe4\xbb\xb6");

    fs::remove_all(root, ec);
    assert(!ec);
    std::printf("native save publication CPU regression: ok\n");
    return 0;
}
