#pragma once

#include "llama-io.h"
#include "llama-mmap.h"
#include "ggml-backend.h"

#define XXH_INLINE_ALL
#include "../vendor/hash/xxhash/xxhash.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <vector>

// Payload staging is independent of tensor/file size. Metadata prepared by the
// memory implementation is context-sized; the transfer index is bounded too.
static constexpr size_t LLAMA_STATE_FILE_BUFFER_SIZE = 8 * 1024 * 1024;

class llama_io_read_file_stream final : public llama_io_read_i {
public:
    llama_io_read_file_stream(llama_file & file, size_t size) : file(file), size(size), buffer(LLAMA_STATE_FILE_BUFFER_SIZE) {
        XXH64_reset(&hash, 0);
    }

    void read(void * dst, size_t count) override {
        if (count > size - consumed) { throw std::runtime_error("truncated sequence state file"); }
        file.read_raw(dst, count);
        XXH64_update(&hash, dst, count);
        consumed += count;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t count) override {
        check_span(tensor, offset, count);
        while (count) {
            const size_t chunk = std::min(count, buffer.size());
            operation op { tensor, offset, chunk, file.tell(), 0, false };
            read(buffer.data(), chunk);
            op.checksum = XXH64(buffer.data(), chunk, 0);
            add(op);
            offset += chunk;
            count -= chunk;
        }
    }

    void stage_tensor_set(ggml_tensor * tensor, const void * src, size_t offset, size_t count) override {
        check_span(tensor, offset, count);
        if (count > LLAMA_STATE_FILE_BUFFER_SIZE - staged.size()) {
            throw std::runtime_error("sequence state staging budget exceeded");
        }
        add({ tensor, offset, count, staged.size(), 0, true });
        const auto * bytes = static_cast<const uint8_t *>(src);
        staged.insert(staged.end(), bytes, bytes + count);
    }

    void stage_tensor_clear(ggml_tensor * tensor, size_t offset, size_t count) override {
        check_span(tensor, offset, count);
        add({ tensor, offset, count, 0, 0, true, true });
    }

    void on_commit(std::function<void()> callback) override { callbacks.push_back(std::move(callback)); }

    void commit() override {
        size_t written = 0;
        const char * fault = std::getenv("LLAMA_TEST_STATE_FILE_COMMIT_FAIL_AFTER");
        const size_t fail_after = fault ? std::strtoull(fault, nullptr, 10) : SIZE_MAX;
        for (const auto & op : operations) {
            if (written >= fail_after) { throw std::runtime_error("injected late sequence state I/O failure"); }
            if (op.clear) {
                ggml_backend_tensor_memset(op.tensor, 0, op.offset, op.size);
            } else if (op.owned) {
                ggml_backend_tensor_set(op.tensor, staged.data() + op.file_offset, op.offset, op.size);
            } else {
                file.seek(op.file_offset, SEEK_SET);
                file.read_raw(buffer.data(), op.size);
                // Recheck before writing: the file may change after parse.
                // Late failures must be handled by the empty-destination API.
                if (XXH64(buffer.data(), op.size, 0) != op.checksum) {
                    throw std::runtime_error("sequence state changed during commit");
                }
                ggml_backend_tensor_set(op.tensor, buffer.data(), op.offset, op.size);
            }
            written += op.size;
        }
        for (auto & callback : callbacks) { callback(); }
        cancel();
    }

    void cancel() override { operations.clear(); callbacks.clear(); staged.clear(); }
    size_t n_bytes() override { return consumed; }
    uint64_t checksum() const { return XXH64_digest(&hash); }

private:
    struct operation {
        ggml_tensor * tensor;
        size_t offset;
        size_t size;
        size_t file_offset;
        uint64_t checksum;
        bool owned;
        bool clear = false;
    };
    static void check_span(ggml_tensor * tensor, size_t offset, size_t count) {
        if (!tensor || offset > ggml_nbytes(tensor) || count > ggml_nbytes(tensor) - offset) {
            throw std::runtime_error("invalid sequence state tensor range");
        }
    }
    void add(operation op) {
        if (operations.size() >= LLAMA_STATE_FILE_BUFFER_SIZE / sizeof(operation)) {
            throw std::runtime_error("sequence state index budget exceeded");
        }
        operations.push_back(op);
    }
    llama_file & file;
    size_t size;
    size_t consumed = 0;
    XXH64_state_t hash;
    std::vector<uint8_t> buffer;
    std::vector<uint8_t> staged;
    std::vector<operation> operations;
    std::vector<std::function<void()>> callbacks;
};
