#include "../ggml/src/ggml-remote-attn/rkva-stage-queue.h"

#undef NDEBUG
#include <cassert>
#include <condition_variable>
#include <mutex>
#include <vector>

int main() {
    std::mutex mutex;
    std::condition_variable cv;
    bool started = false, release = false;
    std::vector<uint32_t> order;
    rkva_stage_queue q(2, 64, [&](const rkva_stage_queue::frame & f) {
        std::unique_lock<std::mutex> lock(mutex);
        started = true;
        cv.notify_all();
        cv.wait(lock, [&] { return release; });
        assert(f.keys[0] == 1.0f && f.values[0] == 2.0f);
        order.push_back(f.first_pos);
        return true;
    });
    float k[2] = {1.0f, 1.0f}, v[2] = {2.0f, 2.0f};
    assert(q.submit(0, 10, 1, k, v));
    {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return started; });
    }
    assert(q.submit(0, 11, 1, k, v));
    k[0] = 9.0f; v[0] = 9.0f;
    {
        std::lock_guard<std::mutex> lock(mutex);
        release = true;
    }
    cv.notify_all();
    assert(q.drain());
    assert(order == std::vector<uint32_t>({10, 11}));

    rkva_stage_queue full(2, 8, [](const rkva_stage_queue::frame &) { return true; });
    assert(!full.submit(0, 0, 1, k, v));
    assert(!full.drain());
    assert(!full.submit(0, 0, 1, k, v));

    rkva_stage_queue failed(2, 64, [](const rkva_stage_queue::frame &) { return false; });
    assert(failed.submit(0, 0, 1, k, v));
    assert(!failed.drain());
    assert(!failed.submit(0, 1, 1, k, v));

#ifndef _WIN32
    struct statvfs space {};
    bool disk_backed = true;
#ifdef __linux__
    struct statfs filesystem {};
    disk_backed = statfs("/tmp", &filesystem) == 0 &&
            filesystem.f_type != TMPFS_MAGIC && filesystem.f_type != RAMFS_MAGIC;
#endif
    if (disk_backed && statvfs("/tmp", &space) == 0 && uint64_t(space.f_bavail) * space.f_frsize > (1ull << 30) + 1024) {
        bool spill_started = false, spill_release = false;
        std::vector<uint32_t> spilled_order;
        rkva_stage_queue spill(2, 16, [&](const rkva_stage_queue::frame & f) {
            std::unique_lock<std::mutex> lock(mutex);
            spill_started = true;
            cv.notify_all();
            cv.wait(lock, [&] { return spill_release; });
            assert(f.keys.size() == 2 && f.values.size() == 2);
            assert(f.keys[0] == float(f.first_pos) && f.values[0] == -float(f.first_pos));
            spilled_order.push_back(f.first_pos);
            return true;
        }, 64);
        float sk[2] = {10, 10}, sv[2] = {-10, -10};
        assert(spill.submit(0, 10, 1, sk, sv));
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&] { return spill_started; });
        }
        sk[0] = sk[1] = 11; sv[0] = sv[1] = -11;
        assert(spill.submit(0, 11, 1, sk, sv));
        sk[0] = sk[1] = 12; sv[0] = sv[1] = -12;
        assert(spill.submit(0, 12, 1, sk, sv));
        sk[0] = sv[0] = 99;
        {
            std::lock_guard<std::mutex> lock(mutex);
            spill_release = true;
        }
        cv.notify_all();
        assert(spill.drain());
        assert(spilled_order == std::vector<uint32_t>({10, 11, 12}));
        assert(spill.spooled_bytes() == 32);

        rkva_stage_queue quota(2, 0, [](const rkva_stage_queue::frame &) { return true; }, 16);
        assert(quota.submit(0, 0, 1, sk, sv));
        assert(!quota.submit(0, 1, 1, sk, sv));
        assert(!quota.drain());
    }
#endif
}
