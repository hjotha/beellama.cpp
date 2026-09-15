#pragma once

// C++ wrapper for the vendored hash functions

#include <cstddef>
#include <cstdio>
#include <string>

// returns the SHA-256 digest as a lowercase hex string
std::string hash_sha256_hex(const void * data, size_t len);

// Hash from the current file position to EOF; the caller owns/rewinds the file.
// Throws on an I/O error, and does not close the file.
std::string hash_sha256_hex(FILE * file);
