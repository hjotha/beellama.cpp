#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "server-gpu-power.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <vector>

namespace {
struct options {
    std::string backend = "Vulkan0";
    std::string type = "q4_0";
    std::string backend_path = "/home/hjotha/beellama.cpp/build-optimized/bin";
    int64_t n_kv = 102400;
    int iterations = 10;
    int warmups = 10;
    int apu_tdp_w = -1;
    int amd_sclk_decode = -1;
    int gpu_power_amd_device = 0;
};

void usage(const char * name) {
    std::cout << "Usage: " << name << " [--backend Vulkan0] [--kv N] [--type q4_0|q5_0|q6_0|q8_0] "
              << "[--iterations N] [--warmups N] [--apu-tdp W] [--amd-sclk-decode MHz] "
              << "[--gpu-power-amd-device N] [--backend-path DIR]\n"
              << "Runs decode-shaped native FLASH_ATTN_EXT (D=256, nq=1, hq=24, hkv=4), no mask/tail.\n";
}

options parse(int argc, char ** argv) {
    options out;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        }
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
        const char * value = argv[++i];
        if (arg == "--backend") out.backend = value;
        else if (arg == "--backend-path") out.backend_path = value;
        else if (arg == "--type") out.type = value;
        else if (arg == "--kv") {
            char * end = nullptr;
            const long long parsed = std::strtoll(value, &end, 10);
            if (!end || *end || parsed < 1 || parsed > (1LL << 22)) {
                throw std::runtime_error("--kv must be in 1..4194304");
            }
            out.n_kv = parsed;
        } else if (arg == "--iterations") {
            char * end = nullptr;
            const long parsed = std::strtol(value, &end, 10);
            if (!end || *end || parsed < 1 || parsed > 100000) {
                throw std::runtime_error("--iterations must be in 1..100000");
            }
            out.iterations = int(parsed);
        } else if (arg == "--warmups") {
            char * end = nullptr;
            const long parsed = std::strtol(value, &end, 10);
            if (!end || *end || parsed < 0 || parsed > 100000) {
                throw std::runtime_error("--warmups must be in 0..100000");
            }
            out.warmups = int(parsed);
        } else if (arg == "--apu-tdp") {
            char * end = nullptr;
            const long parsed = std::strtol(value, &end, 10);
            if (!end || *end || parsed < 1 || parsed > std::numeric_limits<int32_t>::max()) {
                throw std::runtime_error("--apu-tdp must be a positive int32 watt value");
            }
            out.apu_tdp_w = int(parsed);
        } else if (arg == "--amd-sclk-decode") {
            char * end = nullptr;
            const long parsed = std::strtol(value, &end, 10);
            if (!end || *end || parsed < 1 || parsed > std::numeric_limits<int32_t>::max()) {
                throw std::runtime_error("--amd-sclk-decode must be a positive int32 MHz value");
            }
            out.amd_sclk_decode = int(parsed);
        } else if (arg == "--gpu-power-amd-device") {
            char * end = nullptr;
            const long parsed = std::strtol(value, &end, 10);
            if (!end || *end || parsed < 0 || parsed > std::numeric_limits<int32_t>::max()) {
                throw std::runtime_error("--gpu-power-amd-device must be a non-negative int32 index");
            }
            out.gpu_power_amd_device = int(parsed);
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }
    if (out.type != "q4_0" && out.type != "q5_0" && out.type != "q6_0" && out.type != "q8_0") {
        throw std::runtime_error("--type must be q4_0, q5_0, q6_0, or q8_0");
    }
    return out;
}

ggml_type type_from_name(const std::string & name) {
    if (name == "q4_0") return GGML_TYPE_Q4_0;
    if (name == "q5_0") return GGML_TYPE_Q5_0;
    if (name == "q6_0") return GGML_TYPE_Q6_0;
    return GGML_TYPE_Q8_0;
}

float sample(uint64_t i, uint64_t salt) {
    uint64_t x = i + salt + 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return float((x >> 40) * (1.0 / 16777215.0) - 0.5) * 0.8f;
}

std::string json_escape(const std::string & text) {
    std::ostringstream out;
    for (unsigned char c : text) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) out << "\\u00" << "0123456789abcdef"[c >> 4] << "0123456789abcdef"[c & 15];
                else out << char(c);
        }
    }
    return out.str();
}

std::string run_ryzenadj_info() {
    FILE * pipe = popen("/usr/bin/ryzenadj --info", "r");
    if (!pipe) throw std::runtime_error("failed to start fixed /usr/bin/ryzenadj --info readback");
    std::string output;
    char chunk[1024];
    while (std::fgets(chunk, sizeof(chunk), pipe)) output += chunk;
    const int status = pclose(pipe);
    if (status == -1 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        throw std::runtime_error("/usr/bin/ryzenadj --info readback failed");
    }
    return output;
}

std::string amdgpu_card_dir(const server_gpu_power & governor) {
    const std::string & name = governor.amd_device_info().name;
    const size_t start = name.find("/sys/class/drm/card");
    const size_t end = start == std::string::npos ? std::string::npos : name.find(" (sysfs)", start);
    if (start == std::string::npos || end == std::string::npos) {
        throw std::runtime_error("cannot resolve AMD sysfs card path from governor device info");
    }
    return name.substr(start, end - start);
}

struct amd_sysfs_snapshot {
    std::string sclk;
    std::string overdrive;
    std::string performance_level;
};

std::string read_required(const std::string & path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot read AMD governor snapshot: " + path);
    std::ostringstream contents;
    contents << file.rdbuf();
    if (contents.str().empty()) throw std::runtime_error("empty AMD governor snapshot: " + path);
    return contents.str();
}

amd_sysfs_snapshot read_amd_sysfs(const server_gpu_power & governor) {
    const std::string device = amdgpu_card_dir(governor) + "/device/";
    return { read_required(device + "pp_dpm_sclk"), read_required(device + "pp_od_clk_voltage"),
             read_required(device + "power_dpm_force_performance_level") };
}

std::string amd_sysfs_json(const amd_sysfs_snapshot & snapshot) {
    return "{\"pp_dpm_sclk\":\"" + json_escape(snapshot.sclk) +
           "\",\"pp_od_clk_voltage\":\"" + json_escape(snapshot.overdrive) +
           "\",\"power_dpm_force_performance_level\":\"" + json_escape(snapshot.performance_level) + "\"}";
}

void fill_quantized(ggml_backend_t backend, ggml_tensor * tensor, ggml_type type,
                    int64_t n_rows, int64_t width, uint64_t salt) {
    constexpr int64_t chunk_rows = 512;
    const size_t row_bytes = ggml_row_size(type, width);
    std::vector<float> source(size_t(chunk_rows * width));
    std::vector<uint8_t> packed(size_t(chunk_rows) * row_bytes);
    for (int64_t start = 0; start < n_rows; start += chunk_rows) {
        const int64_t count = std::min(chunk_rows, n_rows - start);
        for (int64_t r = 0; r < count; ++r) {
            for (int64_t c = 0; c < width; ++c) {
                const uint64_t index = uint64_t(start + r) * uint64_t(width) + uint64_t(c);
                source[size_t(r * width + c)] = sample(index, salt);
            }
        }
        const size_t written = ggml_quantize_chunk(type, source.data(), packed.data(), 0, count, width, nullptr);
        const size_t expected = size_t(count) * row_bytes;
        if (written != expected) throw std::runtime_error("quantizer returned an unexpected byte count");
        ggml_backend_tensor_set(tensor, packed.data(), size_t(start) * row_bytes, expected);
    }
    ggml_backend_synchronize(backend);
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    return n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) * 0.5;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const options opt = parse(argc, argv);
        ggml_backend_load_all_from_path(opt.backend_path.c_str());
        ggml_backend_dev_t device = ggml_backend_dev_by_name(opt.backend.c_str());
        if (!device) throw std::runtime_error("backend device not found: " + opt.backend);
        ggml_backend_t raw_backend = ggml_backend_dev_init(device, nullptr);
        if (!raw_backend) throw std::runtime_error("failed to initialize backend: " + opt.backend);
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(raw_backend, ggml_backend_free);

        constexpr int64_t d = 256, n_query = 1, n_heads = 24, n_kv_heads = 4;
        if (opt.n_kv > INT64_MAX / n_kv_heads) throw std::runtime_error("--kv is too large");
        ggml_init_params params{ 32 * 1024 * 1024, nullptr, true };
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
        if (!ctx) throw std::runtime_error("ggml_init failed");
        const ggml_type kv_type = type_from_name(opt.type);
        ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, d, n_query, n_heads, 1);
        ggml_tensor * k = ggml_new_tensor_4d(ctx.get(), kv_type, d, opt.n_kv, n_kv_heads, 1);
        ggml_tensor * v = ggml_new_tensor_4d(ctx.get(), kv_type, d, opt.n_kv, n_kv_heads, 1);
        ggml_set_name(q, "query");
        ggml_set_name(k, "key_cache");
        ggml_set_name(v, "value_cache");
        ggml_tensor * out = ggml_flash_attn_ext(ctx.get(), q, k, v, nullptr,
                                                1.0f / std::sqrt(float(d)), 0.0f, 0.0f);
        if (!out || out->op != GGML_OP_FLASH_ATTN_EXT) throw std::runtime_error("failed to build FLASH_ATTN_EXT");
        if (!ggml_prec_set_acc(out, GGML_PREC_F32)) throw std::runtime_error("failed to request F32 accumulation");
        ggml_cgraph * graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, out);
        if (!ggml_backend_dev_supports_op(device, out)) {
            throw std::runtime_error("selected backend does not support this FLASH_ATTN_EXT shape/type");
        }
        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
            ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
        if (!buffer) throw std::runtime_error("backend tensor allocation failed");

        std::vector<float> q_values(size_t(d * n_query * n_heads));
        for (size_t i = 0; i < q_values.size(); ++i) q_values[i] = sample(i, 0x12345678ULL);
        ggml_backend_tensor_set(q, q_values.data(), 0, ggml_nbytes(q));
        fill_quantized(backend.get(), k, kv_type, opt.n_kv * n_kv_heads, d, 0x4b4559ULL);
        fill_quantized(backend.get(), v, kv_type, opt.n_kv * n_kv_heads, d, 0x56414cULL);

        const bool power_requested = opt.apu_tdp_w > 0 || opt.amd_sclk_decode > 0;
        std::unique_ptr<server_gpu_power> governor;
        std::string power_readback, restore_readback;
        std::string amd_before_json = "null", amd_after_json = "null", amd_restored_json = "null";
        if (power_requested) {
            governor = std::make_unique<server_gpu_power>();
            server_gpu_power_config config{};
            config.backend = server_gpu_power_backend_from_string("amdgpu");
            config.apu_tdp_w = opt.apu_tdp_w;
            config.amd_sclk_decode = opt.amd_sclk_decode;
            config.amd_device = opt.gpu_power_amd_device;
            if (!governor->init(config)) {
                governor->shutdown();
                throw std::runtime_error("AMDGPU governor initialization failed");
            }
            governor->update(server_gpu_power_phase::decode);
            if (!governor->enabled()) {
                governor->shutdown();
                throw std::runtime_error("AMDGPU governor disabled while applying decode controls");
            }
        }
        for (int i = 0; i < opt.warmups; ++i) {
            if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("warmup graph compute failed");
            }
            ggml_backend_synchronize(backend.get());
        }
        if (power_requested) {
            power_readback = run_ryzenadj_info();
            amd_before_json = amd_sysfs_json(read_amd_sysfs(*governor));
        }
        std::vector<double> times_ms;
        times_ms.reserve(size_t(opt.iterations));
        for (int i = 0; i < opt.iterations; ++i) {
            ggml_backend_synchronize(backend.get());
            const auto begin = std::chrono::steady_clock::now();
            const ggml_status status = ggml_backend_graph_compute(backend.get(), graph);
            ggml_backend_synchronize(backend.get());
            const auto end = std::chrono::steady_clock::now();
            if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("timed graph compute failed");
            times_ms.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
        }
        if (power_requested) {
            amd_after_json = amd_sysfs_json(read_amd_sysfs(*governor));
            governor->shutdown();
            restore_readback = run_ryzenadj_info();
            amd_restored_json = amd_sysfs_json(read_amd_sysfs(*governor));
        }

        std::vector<float> result(size_t(ggml_nelements(out)));
        ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));
        double sum = 0.0, sum_sq = 0.0;
        bool finite = true;
        for (float x : result) {
            finite = finite && std::isfinite(x);
            sum += x;
            sum_sq += double(x) * x;
        }
        const double mean = sum / result.size();
        const double rms = std::sqrt(sum_sq / result.size());
        if (!finite || !std::isfinite(rms) || rms == 0.0) throw std::runtime_error("attention output is non-finite or zero");

        const double med_ms = median(times_ms);
        const size_t kv_bytes = ggml_nbytes(k) + ggml_nbytes(v);
        const double apparent_gib_s = double(kv_bytes) / (med_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
        std::cout << "{\"benchmark\":\"native_flash_attn_ext_decode\",\"backend\":\""
                  << ggml_backend_dev_name(device) << "\",\"device_description\":\""
                  << ggml_backend_dev_description(device) << "\",\"type\":\"" << opt.type
                  << "\",\"shape\":{\"d\":256,\"n_query\":1,\"n_heads\":24,\"n_kv\":"
                  << opt.n_kv << ",\"n_kv_heads\":4},\"mask\":false,\"tail\":false"
                  << ",\"warmups\":" << opt.warmups << ",\"iterations\":" << opt.iterations
                  << ",\"kv_payload_bytes\":" << kv_bytes
                  << ",\"times_ms\":[";
        for (size_t i = 0; i < times_ms.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << times_ms[i];
        }
        std::cout << "],\"latency_ms\":{\"median\":" << med_ms << ",\"min\":"
                  << *std::min_element(times_ms.begin(), times_ms.end()) << ",\"max\":"
                  << *std::max_element(times_ms.begin(), times_ms.end()) << "}"
                  << ",\"power_controls\":";
        if (!power_requested) {
            std::cout << "null";
        } else {
            std::cout << "{\"requested\":{\"apu_tdp_w\":";
            if (opt.apu_tdp_w > 0) std::cout << opt.apu_tdp_w; else std::cout << "null";
            std::cout << ",\"amd_sclk_decode_mhz\":";
            if (opt.amd_sclk_decode > 0) std::cout << opt.amd_sclk_decode; else std::cout << "null";
            std::cout << ",\"gpu_power_amd_device\":" << opt.gpu_power_amd_device
                      << "},\"governor_enabled_during_measurement\":true"
                      << ",\"ryzenadj_readback_before_timing\":\"" << json_escape(power_readback) << "\""
                      << ",\"amd_sysfs_before_timing\":" << amd_before_json
                      << ",\"amd_sysfs_after_timing\":" << amd_after_json
                      << ",\"ryzenadj_readback_after_restore\":\"" << json_escape(restore_readback) << "\""
                      << ",\"amd_sysfs_after_restore\":" << amd_restored_json << "}";
        }
        std::cout
                  << ",\"apparent_kv_payload_read_GiB_s\":" << apparent_gib_s
                  << ",\"is_pure_dram_bandwidth\":false,\"model_prefill_included\":false"
                  << ",\"end_to_end_tokens_per_second\":null,\"output\":{\"finite\":true,\"mean\":"
                  << mean << ",\"rms\":" << rms << "}}\n";
        ggml_quantize_free();
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "bench-radeon-attention: " << e.what() << '\n';
        return 1;
    }
}
