#pragma once

#include <condition_variable>
#include <cstdio>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/statvfs.h>
#include <unistd.h>
#endif
#ifdef __linux__
#include <linux/magic.h>
#include <sys/vfs.h>
#endif

// A failed or saturated queue is unusable: its remote session must be discarded.
class rkva_stage_queue {
public:
    struct frame {
        uint32_t layer;
        uint32_t first_pos;
        uint32_t count;
        std::vector<float> keys;
        std::vector<float> values;
        size_t size_bytes = 0;
        uint64_t file_offset = UINT64_MAX;
        size_t bytes() const { return size_bytes; }
    };
    using send_fn = std::function<bool(const frame &)>;

    rkva_stage_queue(size_t width, size_t max_bytes, send_fn send, uint64_t max_spool_bytes = 0)
        : width_(width), max_bytes_(max_bytes), send_(std::move(send)), max_spool_bytes_(max_spool_bytes) {
#ifndef _WIN32
        if (max_spool_bytes_) spool_ = std::tmpfile();
        if (spool_) {
            const int fd = fileno(spool_);
            const int flags = fcntl(fd, F_GETFD);
            if (flags < 0 || fcntl(fd, F_SETFD, flags | FD_CLOEXEC) != 0) {
                std::fclose(spool_);
                spool_ = nullptr;
            }
        }
#endif
#ifdef __linux__
        if (spool_) {
            struct statfs fs {};
            if (fstatfs(fileno(spool_), &fs) != 0 || fs.f_type == TMPFS_MAGIC || fs.f_type == RAMFS_MAGIC) {
                std::fclose(spool_);
                spool_ = nullptr;
            }
        }
#endif
        try { worker_ = std::thread([this] { run(); }); }
        catch (...) { if (spool_) std::fclose(spool_); throw; }
    }

    rkva_stage_queue(const rkva_stage_queue &) = delete;
    rkva_stage_queue & operator=(const rkva_stage_queue &) = delete;

    ~rkva_stage_queue() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            clear_pending();
        }
        cv_.notify_all();
        worker_.join();
        if (spool_) std::fclose(spool_);
    }

    bool submit(uint32_t layer, uint32_t first_pos, uint32_t count,
                const float * keys, const float * values) {
        if (!keys || !values || !count || count > 512 || !width_ ||
                width_ > std::numeric_limits<size_t>::max() / count / (2 * sizeof(float))) return false;
        const size_t elements = width_ * count;
        const size_t bytes = elements * 2 * sizeof(float);
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_ || failed_) return false;
        try {
            frame f{layer, first_pos, count, {}, {}, bytes, UINT64_MAX};
            if (bytes <= max_bytes_ && pending_bytes_ <= max_bytes_ - bytes) {
                f.keys.assign(keys, keys + elements);
                f.values.assign(values, values + elements);
                pending_bytes_ += bytes;
            } else {
#ifndef _WIN32
                struct statvfs disk {};
                const int fd = spool_ ? fileno(spool_) : -1;
                if (fd < 0 || bytes > max_spool_bytes_ || spool_bytes_ > max_spool_bytes_ - bytes ||
                        spool_bytes_ + bytes > uint64_t(std::numeric_limits<off_t>::max()) ||
                        fstatvfs(fd, &disk) != 0 ||
                        uint64_t(disk.f_bavail) * disk.f_frsize < uint64_t(bytes) + (1ull << 30) ||
                        !write_all(fd, keys, elements * sizeof(float), spool_bytes_) ||
                        !write_all(fd, values, elements * sizeof(float), spool_bytes_ + elements * sizeof(float))) {
                    failed_ = true;
                    clear_pending();
                    cv_.notify_all();
                    return false;
                }
                f.file_offset = spool_bytes_;
                spool_bytes_ += bytes;
#else
                failed_ = true;
                clear_pending();
                cv_.notify_all();
                return false;
#endif
            }
            pending_.push_back(std::move(f));
        } catch (...) {
            failed_ = true;
            clear_pending();
            cv_.notify_all();
            return false;
        }
        cv_.notify_all();
        return true;
    }

    bool drain() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return pending_.empty() && !busy_; });
        return !failed_ && !stopping_;
    }

    uint64_t spooled_bytes() {
        std::lock_guard<std::mutex> lock(mutex_);
        return spool_bytes_;
    }

private:
    void clear_pending() {
        for (const auto & f : pending_) if (f.file_offset == UINT64_MAX) pending_bytes_ -= f.bytes();
        pending_.clear();
    }

#ifndef _WIN32
    static bool write_all(int fd, const void * data, size_t size, uint64_t offset) {
        auto * p = static_cast<const uint8_t *>(data);
        while (size) {
            const ssize_t n = pwrite(fd, p, size, off_t(offset));
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) return false;
            p += n; size -= size_t(n); offset += uint64_t(n);
        }
        return true;
    }
    static bool read_all(int fd, void * data, size_t size, uint64_t offset) {
        auto * p = static_cast<uint8_t *>(data);
        while (size) {
            const ssize_t n = pread(fd, p, size, off_t(offset));
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) return false;
            p += n; size -= size_t(n); offset += uint64_t(n);
        }
        return true;
    }
#endif

    void run() {
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            cv_.wait(lock, [this] { return stopping_ || !pending_.empty(); });
            if (stopping_) return;
            frame f = std::move(pending_.front());
            pending_.pop_front();
            busy_ = true;
            lock.unlock();
            bool ok = false;
            try {
#ifndef _WIN32
                if (f.file_offset != UINT64_MAX) {
                    const size_t elements = f.bytes() / (2 * sizeof(float));
                    f.keys.resize(elements); f.values.resize(elements);
                    ok = read_all(fileno(spool_), f.keys.data(), elements * sizeof(float), f.file_offset) &&
                            read_all(fileno(spool_), f.values.data(), elements * sizeof(float),
                                    f.file_offset + elements * sizeof(float));
                } else
#endif
                { ok = true; }
                ok = ok && send_ && send_(f);
            } catch (...) { ok = false; }
            lock.lock();
            if (f.file_offset == UINT64_MAX) pending_bytes_ -= f.bytes();
            busy_ = false;
            if (!ok) {
                failed_ = true;
                clear_pending();
            }
            cv_.notify_all();
        }
    }

    const size_t width_;
    const size_t max_bytes_;
    send_fn send_;
    const uint64_t max_spool_bytes_;
    uint64_t spool_bytes_ = 0;
    std::FILE * spool_ = nullptr;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<frame> pending_;
    size_t pending_bytes_ = 0;
    bool busy_ = false;
    bool failed_ = false;
    bool stopping_ = false;
    std::thread worker_;
};
