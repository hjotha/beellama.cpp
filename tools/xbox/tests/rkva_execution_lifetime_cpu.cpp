// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "../uwp/rkva/rkva-execution-lifetime.h"
#include <cassert>
#include <cstdio>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
using rkva_execution_lifetime::releasable;
using rkva_execution_lifetime::Snapshot;
int main() {
    const uint64_t max = std::numeric_limits<uint64_t>::max();
    const Snapshot cases[] = {{},
                              {false, false, false, max, max, false},
                              {true, false, true, 7, 7, false},
                              {true, true, false, 7, 7, false},
                              {true, true, true, 7, 6, false},
                              {true, true, true, 7, 7, false},
                              {true, true, true, 7, 8, false},
                              {true, true, true, 7, max, false},
                              {true, true, true, 7, max, true},
                              {true, true, true, 0, 0, false},
                              {true, true, true, max, max - 1, false},
                              {true, true, true, 7, 6, true},
                              {true, true, true, 7, 7, true},
                              {true, true, true, 0, max, true}};
    const bool expected[] = {true,  true, false, false, false, true, true,
                             false, true, false, false, false, true, true};
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i)
        assert(releasable(cases[i]) == expected[i]);
    // Model the listener owner: an uncertain resource remains owned, and a UI
    // reactivation on another thread cannot create a replacement listener.
    unsigned destroyed = 0;
    struct Resource {
        unsigned& n;
        ~Resource() {
            ++n;
        }
    };
    std::unique_ptr<Resource> resource(new Resource{destroyed});
    std::mutex owner;
    std::unique_lock<std::mutex> retained(owner);
    bool replacement_started = false;
    std::thread reactivation([&] {
        std::unique_lock<std::mutex> next(owner, std::try_to_lock);
        replacement_started = next.owns_lock();
    });
    reactivation.join();
    assert(!replacement_started);
    if (releasable(cases[4]))
        resource.reset();
    assert(resource && destroyed == 0);
    if (releasable(cases[5]))
        resource.reset();
    assert(!resource && destroyed == 1);
    retained.unlock();
    std::thread normal_restart([&] {
        std::unique_lock<std::mutex> next(owner, std::try_to_lock);
        replacement_started = next.owns_lock();
    });
    normal_restart.join();
    assert(replacement_started);
    std::puts("execution lifetime: 14 retirement cases and retained-owner lifecycle passed");
}
