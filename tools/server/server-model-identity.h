#pragma once

#include "llama.h"

#include <array>
#include <string>
#include <vector>

// Prepared once after download/path resolution, before loading the resident model.
// Recheck after model loading; do not replace this identity by hashing changed files.
// Effective context/RoPE and native state versions are checked separately on restore.
class server_model_identity {
public:
    static server_model_identity prepare(const std::string & loader_path,
            const std::vector<llama_model_kv_override> & overrides);

    // Throws on an observed source change. Uses metadata only, never hashes again.
    void verify_sources() const;
    const std::string & fingerprint() const { return digest; }
    size_t source_count() const { return sources.size(); }

private:
    struct source {
        std::string loader_path;
        std::string canonical_path;
        std::array<uint64_t, 8> file;
        std::array<uint64_t, 8> alias;
        bool symlink;
    };
    static source capture(const std::string & loader_path);
    static bool equal(const source & a, const source & b);
    std::vector<source> sources;
    std::string digest;
};
