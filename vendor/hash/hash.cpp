#include "hash.h"

#include <array>
#include <stdexcept>

extern "C" {
#include "sha256/sha256.h"
}

static std::string to_hex(const unsigned char * digest, size_t len) {
    static const char hex[] = "0123456789abcdef";

    std::string out;
    out.reserve(2*len);
    for (size_t i = 0; i < len; ++i) {
        out += hex[digest[i] >> 4];
        out += hex[digest[i] & 0xf];
    }
    return out;
}

std::string hash_sha256_hex(const void * data, size_t len) {
    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_hash(digest, (const unsigned char *) data, len);
    return to_hex(digest, SHA256_DIGEST_SIZE);
}


std::string hash_sha256_hex(FILE * file) {
    if (!file) { throw std::runtime_error("cannot hash a null file"); }
    sha256_t state;
    sha256_init(&state);
    std::array<unsigned char, 65536> buffer;
    for (;;) {
        const size_t n = std::fread(buffer.data(), 1, buffer.size(), file);
        sha256_update(&state, buffer.data(), n);
        if (n != buffer.size()) {
            if (std::ferror(file)) { throw std::runtime_error("cannot read file for SHA256"); }
            break;
        }
    }
    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_final(&state, digest);
    return to_hex(digest, SHA256_DIGEST_SIZE);
}
