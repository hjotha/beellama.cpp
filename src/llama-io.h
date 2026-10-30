#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <utility>
#include <vector>

struct ggml_tensor;

class llama_io_write_i {
public:
    llama_io_write_i() = default;
    virtual ~llama_io_write_i() = default;

    virtual void write(const void * src, size_t size) = 0;
    virtual void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;
    // Counting writers can answer exact state sizes without reading device
    // tensor payloads. Other writers always return false.
    virtual bool counts_only() const { return false; }

    // bytes written so far
    virtual size_t n_bytes() = 0;

    uint32_t native_expected_tokens() const { return native_expected_tokens_; }
    void set_native_expected_tokens(uint32_t count) { native_expected_tokens_ = count; }

    void write_string(const std::string & str);

private:
    uint32_t native_expected_tokens_ = 0;
};

class llama_io_read_i {
public:
    llama_io_read_i() = default;
    virtual ~llama_io_read_i() = default;

    virtual void read(void * dst, size_t size) = 0;
    // Optional non-consuming lookahead. Readers that cannot inspect without
    // advancing return false; native FULL restore then fails closed when a
    // trailer might be present.
    virtual bool peek(void * dst, size_t size) {
        (void) dst;
        (void) size;
        return false;
    }
    virtual void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // State restore is a transaction. Tensor writes and publication callbacks
    // are prepared while the frame is parsed, then made visible only after the
    // complete frame has validated. Destruction without commit cancels them.
    virtual void stage_tensor_set(ggml_tensor * tensor, const void * src, size_t offset, size_t size) = 0;
    virtual void stage_tensor_clear(ggml_tensor * tensor, size_t offset, size_t size) = 0;
    virtual void on_commit(std::function<void()> callback) = 0;
    virtual void commit() = 0;
    virtual void cancel() = 0;
    virtual void discard() { cancel(); }

    // bytes read so far
    virtual size_t n_bytes() = 0;
    // Optional frame extent. Readers that know their logical end expose the
    // remaining bytes; opaque readers return 0 and cannot host an optional
    // native trailer.
    virtual size_t remaining() const { return 0; }

    // File readers set this only while staging a native FULL memory body.
    // The mixed cache uses it to bind every occupied position to the exact
    // evaluated-token prefix, before the outer RKVK trailer is validated.
    virtual uint32_t native_expected_tokens() const { return native_expected_tokens_; }
    virtual void set_native_expected_tokens(uint32_t count) { native_expected_tokens_ = count; }

    void read_string(std::string & str);

private:
    uint32_t native_expected_tokens_ = 0;
};

// F4 §3.6: replays a short prefix that was already consumed from `inner` and
// then delegates everything else. llama_io_read_i has no peek/seek, so a caller
// that must classify a frame has to read the leading word first; wrapping the
// reader here restores the exact original byte stream without teaching any
// reader about speculative reads.
//
// Only read() is served from the prefix, and only until it drains. Every
// transactional and tensor operation (read_tensor, stage_tensor_set/clear,
// on_commit, commit, cancel) is forwarded to `inner` untouched, so the restore
// transaction still belongs to the original reader. n_bytes() already counts the
// prefix, because it was read from `inner` before this wrapper existed.
class llama_io_read_replay final : public llama_io_read_i {
public:
    llama_io_read_replay(llama_io_read_i & inner, const void * prefix, size_t prefix_size)
        : m_inner(inner) {
        const uint8_t * src = static_cast<const uint8_t *>(prefix);
        m_prefix.assign(src, src + prefix_size);
    }

    void read(void * dst, size_t size) override {
        if (m_offset < m_prefix.size()) {
            const size_t available = m_prefix.size() - m_offset;
            const size_t take = size < available ? size : available;
            memcpy(dst, m_prefix.data() + m_offset, take);
            m_offset += take;
            dst = static_cast<uint8_t *>(dst) + take;
            size -= take;
        }
        if (size > 0) {
            m_inner.read(dst, size);
        }
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        m_inner.read_tensor(tensor, offset, size);
    }
    void stage_tensor_set(ggml_tensor * tensor, const void * src, size_t offset, size_t size) override {
        m_inner.stage_tensor_set(tensor, src, offset, size);
    }
    void stage_tensor_clear(ggml_tensor * tensor, size_t offset, size_t size) override {
        m_inner.stage_tensor_clear(tensor, offset, size);
    }
    void on_commit(std::function<void()> callback) override {
        m_inner.on_commit(std::move(callback));
    }
    void commit() override {
        m_inner.commit();
    }
    void cancel() override {
        m_inner.cancel();
    }
    size_t n_bytes() override {
        return m_inner.n_bytes();
    }
    size_t remaining() const override {
        return m_inner.remaining();
    }
    uint32_t native_expected_tokens() const override { return m_inner.native_expected_tokens(); }
    void set_native_expected_tokens(uint32_t count) override { m_inner.set_native_expected_tokens(count); }
    bool peek(void * dst, size_t size) override {
        if (m_offset + size > m_prefix.size()) {
            return false;
        }
        std::memcpy(dst, m_prefix.data() + m_offset, size);
        return true;
    }

private:
    llama_io_read_i & m_inner;
    std::vector<uint8_t> m_prefix;
    size_t m_offset = 0;
};
