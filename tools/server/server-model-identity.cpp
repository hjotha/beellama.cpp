#include "server-model-identity.h"

#include "ggml.h"
#include "gguf.h"
#include "hash/hash.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <stdexcept>

#ifndef _WIN32
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

using overrides_map = std::map<std::string, const llama_model_kv_override *>;

static void append_u64(std::string & out, uint64_t value) {
    for (int i = 0; i < 8; ++i) { out.push_back(char((value >> (8*i)) & 255)); }
}

static void append_string(std::string & out, const std::string & value) {
    append_u64(out, value.size());
    out += value;
}

static std::string bounded_string(const char * data, size_t capacity) {
    const auto * end = static_cast<const char *>(std::memchr(data, 0, capacity));
    if (!end) { throw std::runtime_error("unterminated model identity override"); }
    return {data, end};
}

static overrides_map first_overrides(const std::vector<llama_model_kv_override> & values) {
    overrides_map result;
    for (const auto & value : values) {
        if (!value.key[0]) { break; } // Same sentinel and first-wins insertion as the loader.
        result.emplace(bounded_string(value.key, sizeof(value.key)), &value);
    }
    return result;
}

static std::string canonical_overrides(const overrides_map & values) {
    std::string result;
    append_u64(result, values.size());
    for (const auto & item : values) {
        append_string(result, item.first);
        const auto & value = *item.second;
        result.push_back(char(value.tag));
        switch (value.tag) {
            case LLAMA_KV_OVERRIDE_TYPE_INT: append_u64(result, uint64_t(value.val_i64)); break;
            case LLAMA_KV_OVERRIDE_TYPE_FLOAT: {
                uint64_t bits;
                static_assert(sizeof(bits) == sizeof(value.val_f64), "unexpected double size");
                std::memcpy(&bits, &value.val_f64, sizeof(bits));
                append_u64(result, bits);
            } break;
            case LLAMA_KV_OVERRIDE_TYPE_BOOL: result.push_back(value.val_bool ? 1 : 0); break;
            case LLAMA_KV_OVERRIDE_TYPE_STR:
                append_string(result, bounded_string(value.val_str, sizeof(value.val_str)));
                break;
            default: throw std::runtime_error("unknown model identity override type");
        }
    }
    return result;
}

static uint16_t split_value(const gguf_context * meta, const char * key,
        const overrides_map & overrides, bool required) {
    const auto found = overrides.find(key);
    if (found != overrides.end() && found->second->tag == LLAMA_KV_OVERRIDE_TYPE_INT) {
        return uint16_t(found->second->val_i64); // Match GGUFMeta integral override conversion.
    }
    // A wrong override tag falls back to the raw metadata, as in the loader.
    const int64_t id = gguf_find_key(meta, key);
    if (id < 0) {
        if (required) { throw std::runtime_error(std::string("missing model identity key: ") + key); }
        return 0;
    }
    if (gguf_get_kv_type(meta, id) != GGUF_TYPE_UINT16) {
        throw std::runtime_error(std::string("invalid model identity key type: ") + key);
    }
    return gguf_get_val_u16(meta, id);
}

#ifndef _WIN32
static std::array<uint64_t, 8> stamp(const struct stat & s) {
#ifdef __APPLE__
    const auto modified = s.st_mtimespec;
    const auto changed = s.st_ctimespec;
#else
    const auto modified = s.st_mtim;
    const auto changed = s.st_ctim;
#endif
    return {{uint64_t(s.st_dev), uint64_t(s.st_ino), uint64_t(s.st_size), uint64_t(s.st_mode),
        uint64_t(modified.tv_sec), uint64_t(modified.tv_nsec), uint64_t(changed.tv_sec), uint64_t(changed.tv_nsec)}};
}

static std::array<uint64_t, 8> descriptor_stamp(FILE * file) {
    struct stat status;
    if (fstat(fileno(file), &status) != 0 || !S_ISREG(status.st_mode)) {
        throw std::runtime_error("cannot inspect the opened model identity source");
    }
    return stamp(status);
}
#endif

} // namespace

server_model_identity::source server_model_identity::capture(const std::string & loader_path) {
#ifdef _WIN32
    (void) loader_path;
    // Second-resolution stat cannot prove stability across hashing/loading.
    // Legacy/default-off loading does not use this adaptive identity helper.
    throw std::runtime_error("adaptive model identity requires native file ID/change-time support on Windows");
#else
    struct stat link_status;
    struct stat file_status;
    if (lstat(loader_path.c_str(), &link_status) != 0 || stat(loader_path.c_str(), &file_status) != 0 ||
            !S_ISREG(file_status.st_mode)) {
        throw std::runtime_error("cannot inspect model identity source: " + loader_path);
    }
    source result;
    result.loader_path = loader_path;
    result.canonical_path = std::filesystem::canonical(loader_path).string();
    result.file = stamp(file_status);
    result.symlink = S_ISLNK(link_status.st_mode);
    result.alias = result.symlink ? stamp(link_status) : result.file;
    return result;
#endif
}

bool server_model_identity::equal(const source & a, const source & b) {
    return a.loader_path == b.loader_path && a.canonical_path == b.canonical_path &&
        a.file == b.file && a.alias == b.alias && a.symlink == b.symlink;
}

void server_model_identity::verify_sources() const {
    if (sources.empty() || digest.empty()) { throw std::runtime_error("model identity was not prepared"); }
    for (const auto & source : sources) {
        if (!equal(source, capture(source.loader_path))) {
            throw std::runtime_error("model identity source changed during hashing/loading: " + source.loader_path);
        }
    }
}

server_model_identity server_model_identity::prepare(const std::string & loader_path,
        const std::vector<llama_model_kv_override> & overrides) {
    server_model_identity result;
    const auto typed = first_overrides(overrides);
    std::string material = "llama-prompt-cache-model-v1";
    append_string(material, canonical_overrides(typed));
    // Keep the loader-visible name. Canonicalizing first changes inferred shard siblings.
    std::vector<std::string> paths {std::filesystem::absolute(loader_path).string()};
    for (size_t index = 0; index < paths.size(); ++index) {
        const source before = capture(paths[index]);
        std::unique_ptr<FILE, int (*)(FILE *)> file(ggml_fopen(paths[index].c_str(), "rb"), &std::fclose);
        if (!file) { throw std::runtime_error("cannot open model identity source: " + paths[index]); }
#ifndef _WIN32
        if (descriptor_stamp(file.get()) != before.file) {
            throw std::runtime_error("model identity source changed while opening: " + paths[index]);
        }
#endif
        std::unique_ptr<gguf_context, decltype(&gguf_free)> meta(
            gguf_init_from_file_ptr(file.get(), {true, nullptr}), &gguf_free);
        if (!meta) { throw std::runtime_error("cannot read model identity GGUF: " + paths[index]); }
        if (index == 0) {
            const uint16_t count = split_value(meta.get(), "split.count", typed, false);
            if (count > 1) {
                if (split_value(meta.get(), "split.no", typed, true) != 0) {
                    throw std::runtime_error("model identity requires the first GGUF shard");
                }
                std::vector<char> prefix(paths.front().size() + 1);
                if (llama_split_prefix(prefix.data(), prefix.size(), paths.front().c_str(), 0, count) <= 0) {
                    throw std::runtime_error("invalid model identity shard filename");
                }
                std::vector<char> path(prefix.size() + 64);
                for (uint16_t i = 1; i < count; ++i) {
                    const int n = llama_split_path(path.data(), path.size(), prefix.data(), i, count);
                    if (n <= 0 || size_t(n) >= path.size()) { throw std::runtime_error("model identity shard path overflow"); }
                    paths.emplace_back(path.data(), n);
                }
            }
            append_u64(material, paths.size());
        } else if (split_value(meta.get(), "split.no", {}, true) != index) {
            throw std::runtime_error("model identity GGUF shard index mismatch");
        }
        if (std::fseek(file.get(), 0, SEEK_SET) != 0) { throw std::runtime_error("cannot rewind model identity source"); }
        const std::string file_hash = hash_sha256_hex(file.get());
#ifndef _WIN32
        if (descriptor_stamp(file.get()) != before.file) {
            throw std::runtime_error("model identity source changed while hashing: " + paths[index]);
        }
#endif
        if (!equal(before, capture(paths[index]))) {
            throw std::runtime_error("model identity source path changed while hashing: " + paths[index]);
        }
        append_u64(material, before.file[2]);
        append_string(material, file_hash);
        result.sources.push_back(before);
    }
    result.digest = hash_sha256_hex(material.data(), material.size());
    result.verify_sources(); // Earlier shards must still match after hashing later shards.
    return result;
}
