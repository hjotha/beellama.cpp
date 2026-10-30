#pragma once

#include <cstdint>
#include <array>
#include <cstddef>
#include <string>

namespace rkva_hello_telemetry {

enum FormatPso : size_t {
    f16_opt, q8_opt, kvarn4_opt, kvarn4_prefix, kvarn4_mq2, kvarn4_mq4,
    kvarn4_mq2_i8, kvarn4_decode, kvarn4_rowlane, f16_mq2, f16_mq4,
    q8_mq2, q8_mq4, q8_mq2_i8, format_pso_count
};
inline const char* format_pso_name(size_t i) {
    static const char* const names[] = {"f16_opt", "q8_opt", "kvarn4_opt", "kvarn4_prefix",
        "kvarn4_mq2", "kvarn4_mq4", "kvarn4_mq2_i8", "kvarn4_decode", "kvarn4_rowlane",
        "f16_mq2", "f16_mq4", "q8_mq2", "q8_mq4", "q8_mq2_i8"};
    return names[i];
}

// Device, queue, mandatory roots and mandatory PSOs are disjoint; q4/format
// groups each include any optional root signatures they create. engine_us and
// hello_before_log_us are enclosing spans, not additive to their children.
struct InitTimings {
    uint64_t device_us = 0;
    uint64_t queue_us = 0;
    uint64_t roots_us = 0;
    uint64_t required_psos_us = 0;
    uint64_t q4_psos_us = 0;
    uint64_t format_psos_us = 0;
    std::array<uint64_t, format_pso_count> format_init_us{};
    std::array<uint64_t, format_pso_count> format_init_attempts{};
    std::array<uint64_t, format_pso_count> format_late_us{};
    std::array<uint64_t, format_pso_count> format_late_attempts{};

    void record(FormatPso which, uint64_t us, bool late) {
        (late ? format_late_us : format_init_us)[which] += us;
        ++(late ? format_late_attempts : format_init_attempts)[which];
    }
};

inline std::string event(bool ok, uint64_t cycle, uint64_t evict_us, uint64_t engine_us,
                         const InitTimings& t, uint64_t options_us, uint64_t hello_before_log_us,
                         bool lazy_int8) {
    std::string result = "{\"phase\":\"hello_init\",\"ok\":" + std::string(ok ? "true" : "false") +
           ",\"lazy_int8\":" + std::string(lazy_int8 ? "true" : "false") +
           ",\"cycle\":" + std::to_string(cycle) + ",\"evict_us\":" + std::to_string(evict_us) +
           ",\"engine_us\":" + std::to_string(engine_us) +
           ",\"device_us\":" + std::to_string(t.device_us) +
           ",\"queue_us\":" + std::to_string(t.queue_us) +
           ",\"roots_us\":" + std::to_string(t.roots_us) +
           ",\"required_psos_us\":" + std::to_string(t.required_psos_us) +
           ",\"q4_psos_us\":" + std::to_string(t.q4_psos_us) +
           ",\"format_psos_us\":" + std::to_string(t.format_psos_us) +
           ",\"options_us\":" + std::to_string(options_us) +
           ",\"hello_before_log_us\":" + std::to_string(hello_before_log_us);
    const auto append = [&](const char* label, const std::array<uint64_t, format_pso_count>& values) {
        result += ",\"" + std::string(label) + "\":{";
        for (size_t i = 0; i < values.size(); ++i) {
            if (i) result += ',';
            result += "\"" + std::string(format_pso_name(i)) + "\":" + std::to_string(values[i]);
        }
        result += '}';
    };
    append("format_init_us", t.format_init_us);
    append("format_init_attempts", t.format_init_attempts);
    append("format_late_us", t.format_late_us);
    append("format_late_attempts", t.format_late_attempts);
    return result + "}";
}

} // namespace rkva_hello_telemetry
