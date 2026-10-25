// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/ggml_d3d12.h"
#include "xllama/platform.h"

TEST_CASE("process_cpu_ms is available, monotonic, and delta-shaped") {
    const double a = xllama::process_cpu_ms();
    CHECK(a >= 0.0); // contract: negative only means "unavailable"

    // Burn a little CPU on this thread; the process-wide counter must move.
    volatile double sink = 0.0;
    for (int i = 0; i < 4000000; ++i)
        sink += 0.5;
    (void)sink;

    const double b = xllama::process_cpu_ms();
    CHECK(b >= a);         // monotonic origin on both GetProcessTimes and
                           // CLOCK_PROCESS_CPUTIME_ID
    CHECK(b - a < 5000.0); // a few million adds are milliseconds, not seconds
}

TEST_CASE("thread detection reports a usable, capped count") {
    CHECK(xllama::detect_threads() > 0);
    CHECK(xllama::detect_threads_llama() > 0);
    CHECK(xllama::detect_threads_llama() <= xllama::detect_threads());
}

TEST_CASE("spin-wait knob defaults to historical behavior and round-trips") {
    CHECK(xllama::d3d12_spin_wait_us() == -1); // file absent -> unbounded spin
    xllama::d3d12_set_spin_wait_us(500);
    CHECK(xllama::d3d12_spin_wait_us() == 500);
    xllama::d3d12_set_spin_wait_us(0);
    CHECK(xllama::d3d12_spin_wait_us() == 0);
    xllama::d3d12_set_spin_wait_us(-1);
    CHECK(xllama::d3d12_spin_wait_us() == -1); // restore default
}
