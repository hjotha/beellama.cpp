#include "server-model-identity.h"

#include "download.h"
#include "ggml.h"
#include "gguf.h"
#include "hash/hash.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>

#ifdef _WIN32
#define ST_FSEEK64 _fseeki64
using st_off_t = __int64;
#else
#define ST_FSEEK64 fseeko
using st_off_t = off_t;
#endif

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
    if (streamed) {
        return;
    }
    for (const auto & source : sources) {
        if (!equal(source, capture(source.loader_path))) {
            throw std::runtime_error("model identity source changed during hashing/loading: " + source.loader_path);
        }
    }
}

static std::string compute_etag_header_identity(FILE * file, size_t data_offset, uint64_t total_size, const std::string & etag) {
    if (std::fseek(file, 0, SEEK_SET) != 0) {
        throw std::runtime_error("cannot rewind GGUF header for identity hash");
    }
    const size_t header_len = std::min<size_t>(data_offset, size_t(128) * 1024 * 1024);
    std::string buf;
    buf.reserve(header_len + etag.size() + 64);
    append_string(buf, "gguf-etag-identity-v1");
    append_u64(buf, total_size);
    append_string(buf, etag);
    append_u64(buf, header_len);
    const size_t base = buf.size();
    buf.resize(base + header_len);
    if (header_len > 0 && std::fread(&buf[base], 1, header_len, file) != header_len) {
        throw std::runtime_error("cannot read GGUF header for identity hash");
    }
    return hash_sha256_hex(buf.data(), buf.size());
}

static std::string compute_safetensors_head_tail_identity(FILE * file, uint64_t total_size) {
    static_assert(sizeof(st_off_t) >= 8, "64-bit file offset required");
    if (total_size < 8) {
        throw std::runtime_error("truncated safetensors model identity source");
    }
    if (ST_FSEEK64(file, 0, SEEK_SET) != 0) {
        throw std::runtime_error("cannot rewind safetensors model identity source");
    }
    uint8_t hdr_bytes[8];
    if (std::fread(hdr_bytes, 1, sizeof(hdr_bytes), file) != sizeof(hdr_bytes)) {
        throw std::runtime_error("cannot read safetensors header length for identity hash");
    }
    uint64_t header_len = 0;
    for (int i = 0; i < 8; ++i) {
        header_len |= uint64_t(hdr_bytes[i]) << (8 * i);
    }
    if (header_len > total_size - 8) {
        throw std::runtime_error("corrupt safetensors header length for identity hash");
    }
    const uint64_t payload_off = 8 + header_len;
    const size_t hdr_take = static_cast<size_t>(std::min<uint64_t>(header_len, uint64_t(128) * 1024 * 1024));
    const size_t head_take = static_cast<size_t>(std::min<uint64_t>(total_size - payload_off, 65536ULL));
    const size_t tail_take = static_cast<size_t>(std::min<uint64_t>(total_size, 65536ULL));

    std::string buf;
    buf.reserve(hdr_take + head_take + tail_take + 64);
    append_string(buf, "st-head-tail-v1");
    append_u64(buf, total_size);
    append_u64(buf, header_len);
    append_u64(buf, hdr_take);
    {
        const size_t pos = buf.size();
        buf.resize(pos + hdr_take);
        if (hdr_take > 0 && std::fread(&buf[pos], 1, hdr_take, file) != hdr_take) {
            throw std::runtime_error("cannot read safetensors header for identity hash");
        }
    }
    append_u64(buf, head_take);
    if (head_take > 0) {
        if (hdr_take < header_len && ST_FSEEK64(file, st_off_t(payload_off), SEEK_SET) != 0) {
            throw std::runtime_error("cannot seek safetensors payload head for identity hash");
        }
        const size_t pos = buf.size();
        buf.resize(pos + head_take);
        if (std::fread(&buf[pos], 1, head_take, file) != head_take) {
            throw std::runtime_error("cannot read safetensors payload head for identity hash");
        }
    }
    append_u64(buf, tail_take);
    if (tail_take > 0) {
        if (ST_FSEEK64(file, st_off_t(total_size - tail_take), SEEK_SET) != 0) {
            throw std::runtime_error("cannot seek safetensors tail for identity hash");
        }
        const size_t pos = buf.size();
        buf.resize(pos + tail_take);
        if (std::fread(&buf[pos], 1, tail_take, file) != tail_take) {
            throw std::runtime_error("cannot read safetensors tail for identity hash");
        }
    }
    return hash_sha256_hex(buf.data(), buf.size());
}

server_model_identity server_model_identity::prepare(const std::string & loader_path,
        const std::vector<llama_model_kv_override> & overrides) {
    server_model_identity result;
    const auto typed = first_overrides(overrides);
    std::string material = "llama-prompt-cache-model-v1";
    append_string(material, canonical_overrides(typed));

    const auto exl3_metadata = [](const std::string & name) {
        return name == "config.json" || name == "tokenizer.json" || name == "tokenizer_config.json" ||
               name == "model.safetensors.index.json" || name == "quantization_config.json" ||
               name == "chat_template.jinja" || name == "chat_template.json" || name == "generation_config.json";
    };

    std::vector<common_download_stream_file_info> stream_infos;
    if (common_download_get_stream_file_infos(loader_path, stream_infos) && !stream_infos.empty()) {
        result.streamed = true;
        const bool exl3 = !stream_infos.front().rel_path.empty();
        if (exl3) {
            append_string(material, "exl3-etag-identity-v1");
            std::sort(stream_infos.begin(), stream_infos.end(), [](const auto & a, const auto & b) {
                return a.rel_path < b.rel_path;
            });
            std::vector<std::filesystem::path> metadata;
            for (const auto & entry : std::filesystem::directory_iterator(loader_path)) {
                if (entry.is_regular_file() && exl3_metadata(entry.path().filename().string())) {
                    metadata.push_back(entry.path());
                }
            }
            std::sort(metadata.begin(), metadata.end());
            append_u64(material, metadata.size());
            for (const auto & path : metadata) {
                std::unique_ptr<FILE, int (*)(FILE *)> file(ggml_fopen(path.string().c_str(), "rb"), &std::fclose);
                if (!file) {
                    throw std::runtime_error("cannot open streamed EXL3 metadata: " + path.string());
                }
                append_string(material, path.filename().string());
                append_string(material, hash_sha256_hex(file.get()));
            }
        }
        append_u64(material, stream_infos.size());
        for (const auto & sinfo : stream_infos) {
            if (exl3) {
                if (sinfo.etag.empty()) {
                    throw std::runtime_error("missing ETag for streamed EXL3 shard: " + sinfo.rel_path);
                }
                append_string(material, sinfo.rel_path);
                append_u64(material, sinfo.total_size);
                append_string(material, sinfo.etag);
            } else {
                std::unique_ptr<FILE, int (*)(FILE *)> file(ggml_fopen(sinfo.temp_path.c_str(), "rb"), &std::fclose);
                if (!file) {
                    throw std::runtime_error("cannot open streamed model identity source: " + sinfo.temp_path);
                }
                std::unique_ptr<gguf_context, decltype(&gguf_free)> meta(
                    gguf_init_from_file_ptr(file.get(), {true, nullptr}), &gguf_free);
                if (!meta) {
                    throw std::runtime_error("cannot read streamed model identity GGUF: " + sinfo.temp_path);
                }
                const size_t data_offset = gguf_get_data_offset(meta.get());
                const std::string tag = !sinfo.etag.empty() ? sinfo.etag : sinfo.url;
                const std::string file_hash = compute_etag_header_identity(file.get(), data_offset, sinfo.total_size, tag);
                append_u64(material, sinfo.total_size);
                append_string(material, file_hash);
            }
            source src{};
            src.loader_path    = sinfo.temp_path;
            src.canonical_path = sinfo.final_path.empty() ? sinfo.local_path : sinfo.final_path;
            src.symlink        = false;
            result.sources.push_back(std::move(src));
        }
        result.digest = hash_sha256_hex(material.data(), material.size());
        return result;
    }
    const std::filesystem::path abs_dir = std::filesystem::absolute(loader_path);
    std::error_code dir_ec;
    if (std::filesystem::is_directory(abs_dir, dir_ec) && !dir_ec) {
        std::vector<std::filesystem::path> dir_paths;
        bool has_config = false;
        bool has_safetensors = false;
        std::error_code it_ec;
        for (const auto & entry : std::filesystem::directory_iterator(abs_dir, it_ec)) {
            if (it_ec) {
                throw std::runtime_error("cannot enumerate model identity directory: " + abs_dir.string());
            }
            std::error_code reg_ec;
            if (!entry.is_regular_file(reg_ec) || reg_ec) {
                continue;
            }
            const std::string fn = entry.path().filename().string();
            const bool is_json = exl3_metadata(fn);
            const bool is_st = entry.path().extension() == ".safetensors" &&
                               fn.rfind("cal_trace", 0) != 0 && fn.rfind("vision_", 0) != 0;
            if (!is_json && !is_st) {
                continue;
            }
            if (fn == "config.json") {
                has_config = true;
            }
            if (is_st) {
                has_safetensors = true;
            }
            dir_paths.push_back(entry.path());
        }
        if (it_ec || !has_config || !has_safetensors) {
            throw std::runtime_error("cannot inspect model identity directory: " + abs_dir.string());
        }
        std::sort(dir_paths.begin(), dir_paths.end(), [](const std::filesystem::path & a, const std::filesystem::path & b) {
            return a.filename().string() < b.filename().string();
        });
        append_string(material, "exl3-dir-identity-v1");
        append_u64(material, dir_paths.size());
        for (const auto & entry_path : dir_paths) {
            const std::string p = entry_path.string();
            const source before = capture(p);
            std::unique_ptr<FILE, int (*)(FILE *)> file(ggml_fopen(p.c_str(), "rb"), &std::fclose);
            if (!file) {
                throw std::runtime_error("cannot open model identity source: " + p);
            }
#ifndef _WIN32
            if (descriptor_stamp(file.get()) != before.file) {
                throw std::runtime_error("model identity source changed while opening: " + p);
            }
#endif
            const bool is_st = entry_path.extension() == ".safetensors";
            const std::string file_hash = is_st
                ? compute_safetensors_head_tail_identity(file.get(), before.file[2])
                : hash_sha256_hex(file.get());
#ifndef _WIN32
            if (descriptor_stamp(file.get()) != before.file) {
                throw std::runtime_error("model identity source changed while hashing: " + p);
            }
#endif
            if (!equal(before, capture(p))) {
                throw std::runtime_error("model identity source path changed while hashing: " + p);
            }
            append_string(material, entry_path.filename().string());
            append_u64(material, before.file[2]);
            append_string(material, file_hash);
            result.sources.push_back(before);
        }
        result.digest = hash_sha256_hex(material.data(), material.size());
        result.verify_sources();
        return result;
    }
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
        // The full-file SHA-256 dominates startup for large models on spinning
        // disks. Cache it in a sidecar keyed by the complete file stamp (device,
        // inode, size, mode, mtime and ctime with nanoseconds): replacing the
        // model changes the stamp and forces a re-hash. The sidecar is advisory,
        // so a missing or unwritable one only costs the hash again.
        std::string file_hash;
#ifndef _WIN32
        const std::string cache_path = paths[index] + ".identity-sha256";
        {
            std::ifstream cache(cache_path);
            uint64_t cached_stamp[8] = {0};
            bool hit = bool(cache);
            for (int i = 0; hit && i < 8; ++i) { hit = bool(cache >> cached_stamp[i]); }
            if (hit) { hit = bool(cache >> file_hash); }
            if (hit) {
                for (int i = 0; i < 8; ++i) {
                    if (cached_stamp[i] != before.file[size_t(i)]) { hit = false; break; }
                }
            }
            if (hit && (file_hash.size() != 64 ||
                    file_hash.find_first_not_of("0123456789abcdef") != std::string::npos)) {
                hit = false;
            }
            if (!hit) { file_hash.clear(); }
        }
#endif
        if (file_hash.empty()) {
            const size_t data_offset = gguf_get_data_offset(meta.get());
            std::string etag;
            {
                std::ifstream ef(paths[index] + ".etag");
                if (ef) {
                    std::getline(ef, etag);
                }
            }
            if (!etag.empty()) {
                file_hash = compute_etag_header_identity(file.get(), data_offset, before.file[2], etag);
            } else {
                file_hash = hash_sha256_hex(file.get());
            }
#ifndef _WIN32
            std::error_code wec;
            const std::string tmp_path = cache_path + ".tmp";
            {
                std::ofstream out(tmp_path, std::ios::trunc);
                if (out) {
                    for (size_t i = 0; i < before.file.size(); ++i) { out << before.file[i] << ' '; }
                    out << file_hash << '\n';
                }
            }
            std::filesystem::rename(tmp_path, cache_path, wec);
            if (wec) { std::filesystem::remove(tmp_path, wec); }
#endif
        }
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
