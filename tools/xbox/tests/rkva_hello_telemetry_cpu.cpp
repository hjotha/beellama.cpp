// CPU-only JSON contract; no D3D12 device or Xbox connection.
#include "../uwp/rkva/rkva-hello-telemetry.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>

int main() {
    rkva_hello_telemetry::InitTimings t;
    t.device_us = 11;
    t.queue_us = 12;
    t.roots_us = 13;
    t.required_psos_us = 14;
    t.q4_psos_us = 15;
    t.format_psos_us = 16;
    t.record(rkva_hello_telemetry::kvarn4_mq2_i8, 21, false);
    t.record(rkva_hello_telemetry::q8_mq2_i8, 22, true);
    const auto good = rkva_hello_telemetry::event(true, 7, 8, 100, t, 17, 200, true);
    assert(good.find("\"lazy_int8\":true") != std::string::npos);
    assert(good.find("\"format_init_us\":{") != std::string::npos);
    assert(good.find("\"format_late_us\":{") != std::string::npos);
    assert(t.format_init_us[rkva_hello_telemetry::kvarn4_mq2_i8] == 21);
    assert(t.format_late_us[rkva_hello_telemetry::q8_mq2_i8] == 22);
    const auto failed = rkva_hello_telemetry::event(false, std::numeric_limits<uint64_t>::max(),
                                                    0, 1, {}, 0, 1, false);
    assert(failed.find("\"ok\":false") != std::string::npos);
    assert(failed.find("\"cycle\":18446744073709551615") != std::string::npos);
    std::puts(good.c_str());
    std::puts(failed.c_str());
}
