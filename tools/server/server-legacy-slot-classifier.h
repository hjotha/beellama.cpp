#pragma once

// Conservative recognition of the server's legacy LLAMASLT v2 envelope.
// Only suitable for a trusted local store whose publication lock stays held
// through classification and subsequent replacement/eviction. XXH64 detects
// accidental damage, not forged writers. Concurrent non-cooperating changes
// and filesystem substitution are outside this contract. Raw sequence files,
// other versions, unknown/truncated files and any occurrence of RKVK are
// intentionally unclassified. The RKVK scan is conservative: arbitrary tensor
// bytes may produce false positives. This is not an RKVK parser.

#include "../../vendor/hash/xxhash/xxhash.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>

namespace server_legacy_slot {
namespace detail {
constexpr uint64_t checksum_seed = 0x534c4f5453544154ULL;

class reader {
public:
    reader(std::ifstream & input, uint64_t end, uint64_t * inspected)
        : input_(input), remaining_(end), inspected_(inspected) {}
    bool raw(void * dst, size_t n) {
        if (n > remaining_ || !input_.read(static_cast<char *>(dst), std::streamsize(n))) return false;
        remaining_ -= n;
        if (inspected_) *inspected_ += n;
        return true;
    }
    bool skip(uint64_t n) {
        if (n > remaining_ || n > uint64_t(std::numeric_limits<std::streamoff>::max())) return false;
        input_.seekg(std::streamoff(n), std::ios::cur);
        if (!input_) return false;
        remaining_ -= n;
        return true;
    }
    bool u32(uint32_t & value) {
        std::array<uint8_t, 4> b{};
        if (!raw(b.data(), b.size())) return false;
        value = uint32_t(b[0]) | uint32_t(b[1]) << 8 | uint32_t(b[2]) << 16 | uint32_t(b[3]) << 24;
        return true;
    }
    bool u64(uint64_t & value) {
        std::array<uint8_t, 8> b{};
        if (!raw(b.data(), b.size())) return false;
        value = 0;
        for (size_t i = 0; i < b.size(); ++i) value |= uint64_t(b[i]) << (8*i);
        return true;
    }
    bool blob(uint64_t & size) { return u64(size) && skip(size); }
    uint64_t remaining() const { return remaining_; }
private:
    std::ifstream & input_;
    uint64_t remaining_;
    uint64_t * inspected_;
};
} // namespace detail

// True means structurally valid LLAMASLT v2, checksum verified and no literal
// RKVK anywhere in its payload, under the store trust/locking contract above.
// inspected_bytes counts application-level bytes read (including repeated parse
// reads); it is optional and does not report OS buffering or physical disk I/O.
inline bool is_proven_local(const std::string & path, uint64_t max_bytes, uint64_t * inspected_bytes = nullptr,
        uint64_t * token_count = nullptr) {
    if (token_count) *token_count = 0;
    if (inspected_bytes) *inspected_bytes = 0;
    try {
        namespace fs = std::filesystem;
        std::error_code ec;
        const auto status = fs::symlink_status(fs::u8path(path), ec);
        if (ec || fs::is_symlink(status) || !fs::is_regular_file(status)) return false;
        std::ifstream in(fs::u8path(path), std::ios::binary | std::ios::ate);
        const auto end = in.tellg();
        if (!in || end < std::streamoff(20) || uint64_t(end) > max_bytes) return false;
        const uint64_t payload = uint64_t(end) - 8;
        in.seekg(0);
        std::array<uint8_t, 12> header{};
        if (!in.read(reinterpret_cast<char *>(header.data()), header.size())) return false;
        if (inspected_bytes) *inspected_bytes += header.size();
        if (std::memcmp(header.data(), "LLAMASLT", 8) ||
                header[8] != 2 || header[9] || header[10] || header[11]) return false;
        in.seekg(0);
        auto hash = std::unique_ptr<XXH64_state_t, decltype(&XXH64_freeState)>(XXH64_createState(), XXH64_freeState);
        if (!hash || XXH64_reset(hash.get(), detail::checksum_seed) != XXH_OK) return false;
        std::array<char, 64 * 1024> buffer{};
        unsigned match = 0;
        constexpr char marker[] = "RKVK";
        uint64_t left = payload;
        while (left) {
            const size_t n = size_t(std::min<uint64_t>(left, buffer.size()));
            if (!in.read(buffer.data(), std::streamsize(n)) || XXH64_update(hash.get(), buffer.data(), n) != XXH_OK)
                return false;
            if (inspected_bytes) *inspected_bytes += n;
            for (size_t i = 0; i < n; ++i) {
                const char c = buffer[i];
                match = c == marker[match] ? match + 1 : (c == marker[0] ? 1 : 0);
                if (match == 4) return false;
            }
            left -= n;
        }
        std::array<uint8_t, 8> footer{};
        if (!in.read(reinterpret_cast<char *>(footer.data()), footer.size())) return false;
        if (inspected_bytes) *inspected_bytes += footer.size();
        uint64_t expected = 0;
        for (size_t i = 0; i < footer.size(); ++i) expected |= uint64_t(footer[i]) << (8*i);
        if (XXH64_digest(hash.get()) != expected) return false;
        in.seekg(0);
        detail::reader r(in, payload, inspected_bytes);
        std::array<char, 8> magic{};
        uint32_t version = 0, profile = 0;
        if (!r.raw(magic.data(), magic.size()) || std::memcmp(magic.data(), "LLAMASLT", 8) ||
                !r.u32(version) || version != 2 || !r.u32(profile) || profile > 5 || !r.skip(16 + 40)) return false;
        uint64_t fingerprint = 0, target_layout = 0, draft_layout = 0;
        if (!r.blob(fingerprint) || !r.blob(target_layout) || !r.blob(draft_layout) ||
                (fingerprint && !target_layout) || !r.skip(8)) return false;
        uint64_t tokens = 0, packed = 0, target = 0, draft = 0, spec = 0;
        uint32_t target_magic = 0, target_seq = 0;
        if (!r.u64(tokens) || !tokens || tokens > INT32_MAX || !r.blob(packed) || packed % 4 ||
                !r.u64(target) || target < 8 || !r.u32(target_magic) || target_magic != 0xaf143cd8 ||
                !r.u32(target_seq) || target_seq > INT32_MAX || !r.skip(target - 8) ||
                !r.blob(draft) || !r.blob(spec)) return false;
        uint64_t checkpoints = 0;
        constexpr uint64_t checkpoint_min_bytes = 189;
        if (!r.u64(checkpoints) || checkpoints > 1000000 || checkpoints > r.remaining() / checkpoint_min_bytes)
            return false;
        for (uint64_t i = 0; i < checkpoints; ++i) {
            uint64_t ignored = 0;
            if (!r.skip(44) || !r.blob(ignored) || !r.blob(ignored) || !r.skip(96 + 8 + 1) ||
                    !r.blob(ignored) || !r.blob(ignored) || !r.blob(ignored)) return false;
        }
        if (r.remaining() != 0) return false;
        if (token_count) *token_count = tokens;
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

// Recognizes a raw llama sequence-state file ("qsgg" v4) that has no literal
// RKVK payload anywhere. Native overflow snapshots bind their host prefix
// through an RKVK trailer, so a complete scan for that marker is a conservative
// way to distinguish a local-only sequence from a native host/Xbox pair. The
// raw stream has no envelope checksum verified here; a damaged local file is
// still safe to replace, while false positives only keep it pinned. This is
// not a model/layout compatibility check and grants no retention authority.
inline bool is_local_sequence(const std::string & path, uint64_t max_bytes,
        uint32_t * token_count = nullptr) {
    if (token_count) *token_count = 0;
    try {
        namespace fs = std::filesystem;
        std::error_code ec;
        const auto status = fs::symlink_status(fs::u8path(path), ec);
        if (ec || fs::is_symlink(status) || !fs::is_regular_file(status)) return false;
        const auto size = fs::file_size(fs::u8path(path), ec);
        if (ec || size < 12 || size > max_bytes) return false;
        std::ifstream in(fs::u8path(path), std::ios::binary);
        if (!in) return false;
        std::array<unsigned char, 12> header{};
        if (!in.read(reinterpret_cast<char *>(header.data()), header.size())) return false;
        auto le32 = [](const unsigned char * p) {
            return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
        };
        const uint32_t magic = le32(header.data());
        const uint32_t version = le32(header.data() + 4);
        const uint32_t tokens = le32(header.data() + 8);
        if (magic != 0x67677371u || version != 4 || tokens == 0 || tokens > uint32_t(INT32_MAX)) return false;
        if (uint64_t(tokens) > (size - 12) / sizeof(int32_t)) return false;
        if (uint64_t(tokens) * sizeof(int32_t) + 12 >= size) return false;
        unsigned match = 0;
        constexpr char marker[] = "RKVK";
        std::array<char, 64 * 1024> buffer{};
        uint64_t left = size;
        in.clear();
        in.seekg(0);
        while (left) {
            const size_t n = size_t(std::min<uint64_t>(left, buffer.size()));
            if (!in.read(buffer.data(), std::streamsize(n))) return false;
            for (size_t i = 0; i < n; ++i) {
                const char c = buffer[i];
                match = c == marker[match] ? match + 1 : (c == marker[0] ? 1 : 0);
                if (match == 4) return false;
            }
            left -= n;
        }
        if (token_count) *token_count = tokens;
        return true;
    } catch (const std::exception &) {
        return false;
    }
}
} // namespace server_legacy_slot
