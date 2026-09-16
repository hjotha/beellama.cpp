#include "server-route-state.h"

#include "json.h"
#define XXH_STATIC_LINKING_ONLY
#include "hash/xxhash/xxhash.h"

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <system_error>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <unistd.h>
#else
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#endif

namespace {
using json = common_json;
constexpr size_t buffer_size = 8 * 1024 * 1024;
constexpr size_t max_manifest = 64 * 1024;
constexpr std::array<char, 8> magic = {{'L','L','R','O','U','T','E','1'}};
constexpr size_t footer_size = 24;
constexpr uint32_t canonical_index_block = 256;
constexpr uint32_t lock_stripes = 64;
constexpr uint64_t max_logits_bytes = 8ULL * 1024 * 1024;

void read_exact(std::istream & in, void * data, size_t size) {
    if (size && !in.read(static_cast<char *>(data), size)) {
        throw std::runtime_error("truncated router state file");
    }
}
uint64_t read_u64(const char * p) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) { value |= uint64_t(uint8_t(p[i])) << (8*i); }
    return value;
}
void put_u64(char * p, uint64_t value) {
    for (int i = 0; i < 8; ++i) { p[i] = char(value >> (8*i)); }
}

struct temporary_file {
    std::string path;
    explicit temporary_file(const std::string & target) {
#ifndef _WIN32
        std::string pattern = target + ".tmp-XXXXXX";
        std::vector<char> name(pattern.begin(), pattern.end());
        name.push_back(0);
        const int fd = mkstemp(name.data());
        if (fd < 0) { throw std::runtime_error("cannot create router state temporary file"); }
        path = name.data();
        ::close(fd);
#else
        static std::atomic<uint64_t> nonce{0};
        for (int attempt = 0; attempt < 32; ++attempt) {
            const std::string candidate = target + ".tmp-" + std::to_string((long) _getpid()) + "-" +
                std::to_string(nonce.fetch_add(1, std::memory_order_relaxed));
            const int fd = _open(candidate.c_str(), _O_CREAT | _O_EXCL | _O_BINARY | _O_RDWR,
                                  _S_IREAD | _S_IWRITE);
            if (fd >= 0) {
                _close(fd);
                path = candidate;
                return;
            }
        }
        throw std::runtime_error("cannot create router state temporary file");
#endif
    }
    ~temporary_file() {
        if (!path.empty()) { std::error_code ec; std::filesystem::remove(path, ec); }
    }
};

static void publish_replace(const std::string & temporary, const std::string & destination) {
    std::error_code ec;
    std::filesystem::rename(temporary, destination, ec);
    if (!ec) {
        return;
    }
#ifdef _WIN32
    if (MoveFileExA(temporary.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        return;
    }
#endif
    throw std::runtime_error("router state publication rename failed: " + ec.message());
}

struct publication_backup {
    std::string original;
    std::string backup;
};

static std::string publication_backup_path(const std::string & path) {
    static std::atomic<uint64_t> nonce{0};
    for (;;) {
        const std::string candidate = path + ".tmp-rollback-" +
            std::to_string(nonce.fetch_add(1, std::memory_order_relaxed));
        std::error_code ec;
        if (ec) {
            throw std::runtime_error("router state rollback-name inspection failed: " + ec.message());
        }
        if (!std::filesystem::exists(candidate, ec)) {
            if (ec) {
                throw std::runtime_error("router state rollback-name inspection failed: " + ec.message());
            }
            return candidate;
        }
    }
}
}

size_t server_route_state_save(llama_context * ctx, llama_seq_id seq,
        const llama_tokens & tokens, const server_model_identity & identity,
        const std::string & path, uint64_t max_bytes, uint32_t index_block,
        const std::vector<float> * logits,
        const std::function<bool(uint64_t)> & before_publish,
        const std::function<bool()> & after_publish,
        server_route_state_store_lock * store_lock,
        server_route_state_lease * publication_lock) {
    std::unique_ptr<server_route_state_store_lock> owned_store;
    if (store_lock == nullptr) {
        const std::filesystem::path parent = std::filesystem::path(path).parent_path();
        owned_store = std::make_unique<server_route_state_store_lock>(parent.string());
        store_lock = owned_store.get();
    }
    if (!store_lock->acquired()) {
        throw std::runtime_error("router snapshot store is busy");
    }
    if (publication_lock != nullptr && !publication_lock->acquired()) {
        throw std::runtime_error("router state publication is busy");
    }
    identity.verify_sources();
    const llama_pos position = llama_memory_seq_pos_max(llama_get_memory(ctx), seq);
    if (tokens.empty() || tokens.size() != size_t(position) + 1 || tokens.size() > llama_n_ctx_seq(ctx)) {
        throw std::runtime_error("router state tokens do not cover evaluated target");
    }
    const size_t native_data = llama_state_seq_get_size(ctx, seq);
    uint64_t logits_bytes = 0;
    uint32_t logits_vocab = 0;
    uint64_t logits_checksum = 0;
    if (logits != nullptr && !logits->empty()) {
        if (logits->size() > UINT32_MAX || logits->size() > SIZE_MAX / sizeof(float)) {
            throw std::runtime_error("router snapshot logits payload is too large");
        }
        logits_bytes = uint64_t(logits->size() * sizeof(float));
        logits_vocab = (uint32_t) logits->size();
        logits_checksum = XXH64(logits->data(), (size_t) logits_bytes, 0);
    }
    // Native file replaces the in-memory magic/seq header (8) with file header
    // (12) and token IDs. Reserve the maximum trailer before writing anything.
    // Build this bound with checked additions: an attacker-controlled token
    // count must fail closed instead of wrapping a subtraction near UINT64_MAX.
    const auto checked_add = [](uint64_t a, uint64_t b, uint64_t & result) {
        if (b > UINT64_MAX - a) {
            return false;
        }
        result = a + b;
        return true;
    };
    uint64_t worst_case = native_data;
    const uint64_t token_bytes = tokens.size() > UINT64_MAX / sizeof(llama_token)
        ? UINT64_MAX : uint64_t(tokens.size()) * sizeof(llama_token);
    if (!native_data || token_bytes == UINT64_MAX ||
            !checked_add(worst_case, logits_bytes, worst_case) ||
            !checked_add(worst_case, sizeof(uint32_t), worst_case) ||
            !checked_add(worst_case, token_bytes, worst_case) ||
            !checked_add(worst_case, max_manifest, worst_case) ||
            !checked_add(worst_case, footer_size, worst_case) ||
            worst_case > max_bytes) {
        throw std::runtime_error("router state exceeds disk size budget");
    }
    temporary_file temporary(path);
    const size_t written = llama_state_seq_save_file(ctx, temporary.path.c_str(), seq, tokens.data(), tokens.size());
    if (!written) { throw std::runtime_error("router native state save failed"); }
    XXH64_state_t hash;
    XXH64_reset(&hash, 0);
    {
        std::ifstream input(temporary.path, std::ios::binary);
        std::vector<char> buffer(buffer_size);
        for (size_t left = written; left;) {
            const size_t count = std::min(left, buffer.size());
            read_exact(input, buffer.data(), count);
            XXH64_update(&hash, buffer.data(), count);
            left -= count;
        }
    }
    if (logits_bytes) {
        std::ofstream output(temporary.path, std::ios::binary | std::ios::app);
        if (!output) {
            throw std::runtime_error("router snapshot logits staging failed");
        }
        output.write(reinterpret_cast<const char *>(logits->data()), (std::streamsize) logits_bytes);
        output.flush();
        if (!output.good()) {
            throw std::runtime_error("router snapshot logits write failed");
        }
    }
    const std::string metadata = json{
        {"version", 1}, {"model", identity.fingerprint()}, {"layout", common_prompt_cache_layout(ctx)},
        {"source_ctx", llama_n_ctx_seq(ctx)}, {"position", position}, {"n_tokens", tokens.size()},
        {"state_bytes", written}, {"state_checksum", XXH64_digest(&hash)},
        {"logits_offset", logits_bytes ? written : 0}, {"logits_bytes", logits_bytes},
        {"logits_checksum", logits_checksum}, {"logits_vocab", logits_vocab},
        {"logits_tokens", logits_bytes ? tokens.size() : 0},
        {"index_block", index_block ? index_block : canonical_index_block}
    }.dump();
    if (metadata.size() > max_manifest) { throw std::runtime_error("router manifest exceeds budget"); }
    std::array<char, footer_size> footer;
    put_u64(footer.data(), metadata.size());
    put_u64(footer.data() + 8, XXH64(metadata.data(), metadata.size(), 0));
    std::copy(magic.begin(), magic.end(), footer.begin() + 16);
    {
        std::unique_ptr<FILE, decltype(&std::fclose)> file(ggml_fopen(temporary.path.c_str(), "ab"), &std::fclose);
        if (!file || std::fwrite(metadata.data(), 1, metadata.size(), file.get()) != metadata.size() ||
                std::fwrite(footer.data(), 1, footer.size(), file.get()) != footer.size() || std::fflush(file.get()) != 0) {
            throw std::runtime_error("router manifest write failed");
        }
#ifndef _WIN32
        if (fsync(fileno(file.get())) != 0) { throw std::runtime_error("router state sync failed"); }
#endif
        if (std::fclose(file.release()) != 0) { throw std::runtime_error("router state close failed"); }
    }
    identity.verify_sources();
    if (std::getenv("LLAMA_TEST_ROUTE_STATE_PUBLISH_FAIL")) {
        throw std::runtime_error("injected router state publication failure");
    }
    const size_t total = written + (size_t) logits_bytes + metadata.size() + footer.size();
    if (std::filesystem::file_size(temporary.path) != total || total > max_bytes) {
        throw std::runtime_error("router state file size mismatch");
    }
    if (before_publish && !before_publish(total)) {
        throw std::runtime_error("router snapshot store budget rejected publication");
    }
    // Keep the previous state and both historical auxiliaries out of the way
    // only while the global store lock is held. If retention commit fails, the
    // new pathname is removed and the old complete object is restored. This
    // matters for callers that reuse a requested filename: a failed budget
    // decision must never turn into a lost prior version or a state/.logits
    // mixture. New auto/route names are normally unique, but the transaction
    // remains correct for explicit legacy filenames too.
    std::vector<publication_backup> backups;
    bool rolled_back = false;
    auto restore_backups = [&]() {
        for (auto it = backups.rbegin(); it != backups.rend(); ++it) {
            std::error_code restore_ec;
            std::filesystem::rename(it->backup, it->original, restore_ec);
            if (restore_ec) {
                throw std::runtime_error("router state publication rollback failed: " + restore_ec.message());
            }
        }
    };
    auto rollback = [&]() {
        if (rolled_back) {
            return;
        }
        rolled_back = true;
        std::error_code ec;
        std::filesystem::remove(path, ec);
        restore_backups();
    };

    bool retention_committed = false;
    try {
        for (const std::string & original : {path, path + ".logits", path + ".meta"}) {
            std::error_code ec;
            if (std::filesystem::exists(original, ec)) {
                if (ec) {
                    throw std::runtime_error("router state publication backup inspection failed: " + ec.message());
                }
                publication_backup item{original, publication_backup_path(original)};
                std::filesystem::rename(original, item.backup, ec);
                if (ec) {
                    throw std::runtime_error("router state publication backup failed: " + ec.message());
                }
                backups.push_back(std::move(item));
            } else if (ec) {
                throw std::runtime_error("router state publication backup inspection failed: " + ec.message());
            }
        }
        publish_replace(temporary.path, path);
        temporary.path.clear();
        if (after_publish && !after_publish()) {
            rollback();
            throw std::runtime_error("router snapshot store retention commit failed");
        }
        retention_committed = true;
        for (const auto & item : backups) {
            std::error_code ec;
            std::filesystem::remove(item.backup, ec);
            if (ec) {
                // The new state is already committed, but an old orphaned
                // version must not be misreported as removed or reintroduced
                // into the index. Surface this as an incomplete transaction.
                throw std::runtime_error("router state publication backup cleanup failed: " + ec.message());
            }
        }
    } catch (...) {
        // A failed rename/retention callback must restore any old object that
        // was moved aside. Once retention has committed, cleanup of an old
        // rollback file is housekeeping only: do not undo a valid new commit
        // based on a second unlink failure.
        if (!retention_committed && !rolled_back && !backups.empty()) {
            rollback();
        }
        throw;
    }
    return total;
}

server_route_state_file server_route_state_read(const std::string & path, uint64_t max_bytes,
        uint32_t max_tokens, bool verify_payload) {
    server_route_state_lease reference(path, server_route_state_lock_mode::reference);
    if (!reference.acquired()) {
        throw std::runtime_error("router state is busy");
    }
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    const auto end = input.tellg();
    if (!input || end < std::streamoff(footer_size) || uint64_t(end) > max_bytes) {
        throw std::runtime_error("invalid router state file size");
    }
    server_route_state_file result;
    result.state_path = path;
    result.file_bytes = size_t(end);
    std::array<char, footer_size> footer;
    input.seekg(end - std::streamoff(footer_size));
    read_exact(input, footer.data(), footer.size());
    const uint64_t count = read_u64(footer.data());
    if (!std::equal(magic.begin(), magic.end(), footer.begin() + 16) ||
            count > max_manifest || count > result.file_bytes - footer_size) {
        throw std::runtime_error("invalid router state manifest");
    }
    std::string metadata(size_t(count), '\0');
    input.seekg(end - std::streamoff(footer_size + count));
    read_exact(input, &metadata[0], metadata.size());
    if (XXH64(metadata.data(), metadata.size(), 0) != read_u64(footer.data() + 8)) {
        throw std::runtime_error("router manifest checksum mismatch");
    }
    const auto meta = json::parse(metadata);
    if (meta.at("version") != 1) { throw std::runtime_error("unsupported router state version"); }
    result.state_bytes = meta.at("state_bytes").get<size_t>();
    result.state_checksum = meta.at("state_checksum").get<uint64_t>();
    result.logits_offset = meta.contains("logits_offset") ? meta.at("logits_offset").get<uint64_t>() : 0;
    result.logits_bytes = meta.contains("logits_bytes") ? meta.at("logits_bytes").get<uint64_t>() : 0;
    result.logits_checksum = meta.contains("logits_checksum") ? meta.at("logits_checksum").get<uint64_t>() : 0;
    result.logits_vocab = meta.contains("logits_vocab") ? meta.at("logits_vocab").get<uint32_t>() : 0;
    result.logits_tokens = meta.contains("logits_tokens") ? meta.at("logits_tokens").get<uint32_t>() : 0;
    result.source_ctx = meta.at("source_ctx").get<uint32_t>();
    result.index_block = meta.contains("index_block") ? meta.at("index_block").get<uint32_t>() : 0;
    result.position = meta.at("position").get<llama_pos>();
    result.model = meta.at("model").get<std::string>();
    result.layout = meta.at("layout").get<std::string>();
    const auto n_tokens = meta.at("n_tokens").get<uint64_t>();
    const uint64_t payload_bytes = result.file_bytes - footer_size - count;
    const bool logits_valid = result.logits_bytes == 0
        ? result.logits_offset == 0 && result.logits_checksum == 0 && result.logits_vocab == 0 && result.logits_tokens == 0
        : result.logits_offset == result.state_bytes && result.logits_vocab > 0 &&
            result.logits_tokens == n_tokens && result.logits_bytes == uint64_t(result.logits_vocab) * sizeof(float) &&
            result.logits_bytes <= max_logits_bytes;
    const bool payload_layout = result.logits_bytes <= payload_bytes &&
        result.state_bytes <= payload_bytes - result.logits_bytes &&
        result.state_bytes + result.logits_bytes == payload_bytes;
    if (!payload_layout ||
            result.state_bytes < 12 || !n_tokens || n_tokens > max_tokens || n_tokens > result.source_ctx ||
            !logits_valid ||
            result.position < 0 || uint64_t(result.position) + 1 != n_tokens ||
            result.index_block > (1u << 20) ||
            n_tokens > (result.state_bytes - 12)/sizeof(llama_token) || result.model.empty() || result.layout.empty()) {
        throw std::runtime_error("invalid router state bounds");
    }
    // Index lookups use only this bounded manifest/header/token pass. A full
    // payload checksum is intentionally deferred to the single candidate that
    // will actually be restored (the native streaming loader verifies the same
    // checksum before committing state). Direct readers retain the safe,
    // verifying default.
    if (verify_payload) {
        input.seekg(0);
        XXH64_state_t state_hash;
        XXH64_reset(&state_hash, 0);
        std::vector<char> state_buffer(buffer_size);
        uint64_t state_left = result.state_bytes;
        while (state_left) {
            const size_t chunk = (size_t) std::min<uint64_t>(state_left, state_buffer.size());
            read_exact(input, state_buffer.data(), chunk);
            XXH64_update(&state_hash, state_buffer.data(), chunk);
            state_left -= chunk;
        }
        if (XXH64_digest(&state_hash) != result.state_checksum) {
            throw std::runtime_error("router native state checksum mismatch");
        }
    }
    input.seekg(0);
    uint32_t header[3];
    read_exact(input, header, sizeof(header));
    if (header[0] != LLAMA_STATE_SEQ_MAGIC || header[1] != LLAMA_STATE_SEQ_VERSION || header[2] != n_tokens) {
        throw std::runtime_error("router native header mismatch");
    }
    result.tokens.resize(size_t(n_tokens));
    read_exact(input, result.tokens.data(), result.tokens.size()*sizeof(llama_token));
    return result;
}

bool server_route_state_read_logits(const server_route_state_file & file,
        uint32_t expected_vocab, std::vector<float> & logits) {
    logits.clear();
    if (file.logits_bytes == 0) {
        return false;
    }
    if (!file.logits_vocab || file.logits_vocab != expected_vocab ||
            file.logits_tokens != file.tokens.size() || file.logits_bytes > max_logits_bytes ||
            file.logits_bytes != uint64_t(file.logits_vocab) * sizeof(float)) {
        return false;
    }
    server_route_state_lease reference(file.state_path, server_route_state_lock_mode::reference);
    if (!reference.acquired()) {
        return false;
    }
    try {
        std::ifstream input(file.state_path, std::ios::binary);
        if (!input || file.logits_offset > file.file_bytes ||
                file.logits_bytes > file.file_bytes - file.logits_offset) {
            return false;
        }
        input.seekg((std::streamoff) file.logits_offset);
        logits.resize(file.logits_vocab);
        std::vector<uint8_t> buffer((size_t) std::min<uint64_t>(file.logits_bytes, buffer_size));
        XXH64_state_t hash;
        XXH64_reset(&hash, 0);
        uint64_t left = file.logits_bytes;
        size_t offset = 0;
        while (left) {
            const size_t count = (size_t) std::min<uint64_t>(left, buffer.size());
            input.read(reinterpret_cast<char *>(buffer.data()), (std::streamsize) count);
            if (input.gcount() != (std::streamsize) count) {
                logits.clear();
                return false;
            }
            std::memcpy(reinterpret_cast<uint8_t *>(logits.data()) + offset, buffer.data(), count);
            XXH64_update(&hash, buffer.data(), count);
            offset += count;
            left -= count;
        }
        if (XXH64_digest(&hash) != file.logits_checksum) {
            logits.clear();
            return false;
        }
        return true;
    } catch (const std::exception &) {
        logits.clear();
        return false;
    }
}

bool server_route_state_adopt_native(const std::string & native_path,
        const std::string & destination,
        const std::string & model, const std::string & layout,
        const llama_tokens & tokens, uint32_t source_ctx,
        uint32_t index_block, uint64_t max_bytes,
        const std::string * provenance_json) {
    try {
        if (native_path.empty() || destination.empty() ||
                model.empty() || layout.empty() || tokens.empty() ||
                tokens.size() > UINT32_MAX || !source_ctx || tokens.size() > source_ctx) {
            return false;
        }
        server_route_state_store_lock store(std::filesystem::path(destination).parent_path().string(),
                server_route_state_store_lock_mode::exclusive);
        if (!store.acquired()) {
            return false;
        }
        std::ifstream input(native_path, std::ios::binary | std::ios::ate);
        const auto end = input.tellg();
        if (!input || end < std::streamoff(3*sizeof(uint32_t))) {
            return false;
        }
        const uint64_t native_size = uint64_t(end);
        if (native_size > max_bytes || native_size < 3*sizeof(uint32_t) + tokens.size()*sizeof(llama_token)) {
            return false;
        }
        input.seekg(0);
        uint32_t header[3] = {};
        read_exact(input, header, sizeof(header));
        if (header[0] != LLAMA_STATE_SEQ_MAGIC || header[1] != LLAMA_STATE_SEQ_VERSION ||
                header[2] != tokens.size()) {
            return false;
        }
        input.seekg(0);

        temporary_file temporary(destination);
        std::ofstream output(temporary.path, std::ios::binary | std::ios::trunc);
        if (!output) {
            return false;
        }
        XXH64_state_t hash;
        XXH64_reset(&hash, 0);
        std::vector<char> buffer(buffer_size);
        uint64_t left = native_size;
        while (left) {
            const size_t count = (size_t) std::min<uint64_t>(left, buffer_size);
            input.read(buffer.data(), (std::streamsize) count);
            if (input.gcount() != (std::streamsize) count) {
                return false;
            }
            output.write(buffer.data(), (std::streamsize) count);
            if (!output) {
                return false;
            }
            XXH64_update(&hash, buffer.data(), count);
            left -= count;
        }
        output.close();

        common_json manifest = {
            {"version", 1}, {"model", model}, {"layout", layout},
            {"source_ctx", source_ctx}, {"position", (llama_pos) tokens.size() - 1},
            {"n_tokens", tokens.size()}, {"state_bytes", native_size},
            {"state_checksum", XXH64_digest(&hash)},
            {"index_block", index_block ? index_block : canonical_index_block}
        };
        if (provenance_json != nullptr) {
            manifest["converted"] = json::parse(*provenance_json);
        }
        const std::string metadata = manifest.dump();
        if (metadata.size() > max_manifest || native_size > UINT64_MAX - metadata.size() - footer_size ||
                native_size + metadata.size() + footer_size > max_bytes) {
            return false;
        }
        std::array<char, footer_size> footer;
        put_u64(footer.data(), metadata.size());
        put_u64(footer.data() + 8, XXH64(metadata.data(), metadata.size(), 0));
        std::copy(magic.begin(), magic.end(), footer.begin() + 16);
        {
            std::unique_ptr<FILE, decltype(&std::fclose)> file(ggml_fopen(temporary.path.c_str(), "ab"), &std::fclose);
            if (!file || std::fwrite(metadata.data(), 1, metadata.size(), file.get()) != metadata.size() ||
                    std::fwrite(footer.data(), 1, footer.size(), file.get()) != footer.size() ||
                    std::fflush(file.get()) != 0) {
                return false;
            }
#ifndef _WIN32
            if (fsync(fileno(file.get())) != 0) {
                return false;
            }
#endif
            if (std::fclose(file.release()) != 0) {
                return false;
            }
        }
        publish_replace(temporary.path, destination);
        temporary.path.clear();
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

bool server_route_state_adopt_legacy(const std::string & path,
        const std::string & model, const std::string & layout,
        const llama_tokens & tokens, uint32_t source_ctx,
        uint32_t index_block, uint64_t max_bytes) {
    return server_route_state_adopt_native(path, path, model, layout, tokens,
            source_ctx, index_block, max_bytes, nullptr);
}

bool server_route_state_remove_if_unreferenced(const std::string & path,
        server_route_state_store_lock * store_lock) {
    std::unique_ptr<server_route_state_store_lock> owned_store;
    if (store_lock == nullptr) {
        owned_store = std::make_unique<server_route_state_store_lock>(
                std::filesystem::path(path).parent_path().string(),
                server_route_state_store_lock_mode::exclusive_try);
        store_lock = owned_store.get();
    }
    if (!store_lock->acquired()) {
        return false;
    }
    server_route_state_lease eviction(path, server_route_state_lock_mode::eviction);
    if (!eviction.acquired()) {
        return false;
    }
    std::error_code ec;
    std::filesystem::remove(path, ec);
    if (ec) {
        return false;
    }
    std::filesystem::remove(path + ".logits", ec);
    std::filesystem::remove(path + ".meta", ec);
    return true;
}

server_route_state_store_lock::server_route_state_store_lock(const std::string & directory,
        server_route_state_store_lock_mode mode) {
    const std::filesystem::path dir = directory.empty() ? std::filesystem::path(".") : std::filesystem::path(directory);
    path = (dir / ".llama-kv-snapshot-store.lock").string();
#ifndef _WIN32
    int flags = O_CREAT | O_RDWR;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
    fd = ::open(path.c_str(), flags, S_IRUSR | S_IWUSR);
    if (fd >= 0) {
        const bool exclusive = mode != server_route_state_store_lock_mode::shared;
        const int operation = exclusive ? LOCK_EX : LOCK_SH;
        const int wait = mode == server_route_state_store_lock_mode::exclusive_try ? LOCK_NB : 0;
        if (::flock(fd, operation | wait) != 0) {
            ::close(fd);
            fd = -1;
        }
    }
#else
    HANDLE raw = CreateFileA(path.c_str(), GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (raw != INVALID_HANDLE_VALUE) {
        OVERLAPPED overlapped = {};
        const DWORD flags = (mode != server_route_state_store_lock_mode::shared ? LOCKFILE_EXCLUSIVE_LOCK : 0) |
            (mode == server_route_state_store_lock_mode::exclusive_try ? LOCKFILE_FAIL_IMMEDIATELY : 0);
        if (LockFileEx(raw, flags, 0, MAXDWORD, MAXDWORD, &overlapped)) {
            handle = raw;
        } else {
            CloseHandle(raw);
        }
    }
#endif
    if (!acquired()) {
        path.clear();
    }
}

void server_route_state_store_lock::release() {
#ifndef _WIN32
    if (fd >= 0) {
        ::flock(fd, LOCK_UN);
        ::close(fd);
        fd = -1;
    }
#else
    if (handle != nullptr) {
        HANDLE raw = static_cast<HANDLE>(handle);
        OVERLAPPED overlapped = {};
        UnlockFileEx(raw, 0, MAXDWORD, MAXDWORD, &overlapped);
        CloseHandle(raw);
        handle = nullptr;
    }
#endif
    path.clear();
}

server_route_state_store_lock::~server_route_state_store_lock() {
    release();
}

server_route_state_lease::server_route_state_lease(const std::string & state_path,
        server_route_state_lock_mode mode) {
    if (state_path.empty()) {
        return;
    }
    path = state_path;
    this->mode = mode;
    std::error_code path_ec;
    const std::string lock_key = std::filesystem::absolute(state_path, path_ec).string();
    const uint32_t stripe = (uint32_t) (XXH64(lock_key.data(), lock_key.size(), 0) % lock_stripes);
    const std::filesystem::path parent = std::filesystem::path(state_path).parent_path().empty()
        ? std::filesystem::path(".") : std::filesystem::path(state_path).parent_path();
    const std::string stem = ".llama-kv-snapshot-ref-" + std::to_string(stripe);
    ref_path = (parent / (stem + ".ref")).string();

#ifndef _WIN32
    auto open_and_lock = [](const std::string & lock_file, int operation) -> int {
        int flags = O_CREAT | O_RDWR;
#ifdef O_CLOEXEC
        flags |= O_CLOEXEC;
#endif
        const int fd = ::open(lock_file.c_str(), flags, S_IRUSR | S_IWUSR);
        if (fd < 0 || ::flock(fd, operation | LOCK_NB) != 0) {
            if (fd >= 0) { ::close(fd); }
            return -1;
        }
        return fd;
    };
    if (mode == server_route_state_lock_mode::reference) {
        shared_store = std::make_unique<server_route_state_store_lock>(parent.string(),
                server_route_state_store_lock_mode::shared);
        if (shared_store->acquired()) {
            ref_fd = open_and_lock(ref_path, LOCK_SH);
        }
    } else if (mode == server_route_state_lock_mode::reservation) {
        ref_fd = open_and_lock(ref_path, LOCK_SH);
    } else if (mode == server_route_state_lock_mode::eviction) {
        ref_fd = open_and_lock(ref_path, LOCK_EX);
    } else if (mode == server_route_state_lock_mode::publication) {
        shared_store = std::make_unique<server_route_state_store_lock>(parent.string(),
                server_route_state_store_lock_mode::exclusive);
    }
#else
    auto open_and_lock = [](const std::string & lock_file, bool exclusive) -> void * {
        HANDLE handle = CreateFileA(lock_file.c_str(), GENERIC_READ | GENERIC_WRITE,
                FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (handle == INVALID_HANDLE_VALUE) {
            return nullptr;
        }
        OVERLAPPED overlapped = {};
        const DWORD flags = (exclusive ? LOCKFILE_EXCLUSIVE_LOCK : 0) | LOCKFILE_FAIL_IMMEDIATELY;
        if (!LockFileEx(handle, flags, 0, MAXDWORD, MAXDWORD, &overlapped)) {
            CloseHandle(handle);
            return nullptr;
        }
        return handle;
    };
    if (mode == server_route_state_lock_mode::reference) {
        shared_store = std::make_unique<server_route_state_store_lock>(parent.string(),
                server_route_state_store_lock_mode::shared);
        if (shared_store->acquired()) {
            ref_handle = open_and_lock(ref_path, false);
        }
    } else if (mode == server_route_state_lock_mode::reservation) {
        ref_handle = open_and_lock(ref_path, false);
    } else if (mode == server_route_state_lock_mode::eviction) {
        ref_handle = open_and_lock(ref_path, true);
    } else if (mode == server_route_state_lock_mode::publication) {
        shared_store = std::make_unique<server_route_state_store_lock>(parent.string(),
                server_route_state_store_lock_mode::exclusive);
    }
#endif

    const bool lock_ok = mode == server_route_state_lock_mode::publication
#ifndef _WIN32
        ? shared_store && shared_store->acquired() : mode == server_route_state_lock_mode::reference
            ? shared_store && shared_store->acquired() && ref_fd >= 0
            : ref_fd >= 0;
#else
        ? shared_store && shared_store->acquired() : mode == server_route_state_lock_mode::reference
            ? shared_store && shared_store->acquired() && ref_handle != nullptr
            : ref_handle != nullptr;
#endif
    if (!lock_ok) {
        release();
    }
}

void server_route_state_lease::release() {
#ifndef _WIN32
    if (lock_fd >= 0) {
        ::flock(lock_fd, LOCK_UN);
        ::close(lock_fd);
        lock_fd = -1;
    }
    if (ref_fd >= 0) {
        ::flock(ref_fd, LOCK_UN);
        ::close(ref_fd);
        ref_fd = -1;
    }
    shared_store.reset();
#else
    auto unlock_close = [](void * raw) {
        if (raw == nullptr) { return; }
        HANDLE handle = static_cast<HANDLE>(raw);
        OVERLAPPED overlapped = {};
        UnlockFileEx(handle, 0, MAXDWORD, MAXDWORD, &overlapped);
        CloseHandle(handle);
    };
    unlock_close(lock_handle);
    unlock_close(ref_handle);
    lock_handle = nullptr;
    ref_handle = nullptr;
    shared_store.reset();
#endif
    path.clear();
    lock_path.clear();
    ref_path.clear();
}

server_route_state_lease::~server_route_state_lease() {
    release();
}
