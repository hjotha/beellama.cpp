// Raw full-vocabulary logit capture for the RKVA quality gate.
//
// This driver is inert by default: without --execute it only prints a preview
// and never initializes a backend, a model, or a network endpoint. --self-test
// is a CPU-only contract check. --execute is reserved for the hardware owner.
//
// Build (CPU host, no CUDA backend compilation):
// Build with g++ -std=c++17 -O2 -Wall -Wextra, the root include directories
// (include, common, ggml/include, ggml/src, vendor, src), and the root
// build-optimized/libllama-common link line. See docs/agent-reports for the
// exact command. No CUDA backend is compiled here.
//
// Output is float32 little-endian, index == token id, exactly 4*vocab bytes per
// step. This file intentionally has no HTTP server, sampler, or model warmup.

#include "llama.h"
#include "llama-cpp.h"
#include "ggml-backend.h"
#include "common.h"
#include "arg.h"
#include "speculative.h"
#include "llama-context.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <iterator>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

extern char ** environ;

namespace {

using json = nlohmann::json;
namespace fs = std::filesystem;
static_assert(sizeof(llama_token) == sizeof(uint32_t), "unexpected token word size");

constexpr uint32_t kBaselineContext = 131072;
constexpr uint32_t kBaselineP       = 102400;
constexpr uint32_t kMaxContext      = 160000;
constexpr uint32_t kBatch           = 1024;
constexpr uint32_t kUbatch          = 512;
constexpr uint32_t kPhysicalUbatch  = 256;
constexpr size_t   kMaxJsonBytes    = 64u * 1024u * 1024u;

struct app_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

[[noreturn]] void fail(const std::string & message) {
    throw app_error(message);
}

void require(bool ok, const std::string & message) {
    if (!ok) fail(message);
}

std::string path_text(const fs::path & path) {
    return path.string();
}

bool is_symlink_path(const fs::path & path) {
    std::error_code ec;
    return fs::is_symlink(fs::symlink_status(path, ec));
}

void require_regular_file(const fs::path & path, const std::string & label) {
    std::error_code ec;
    require(!is_symlink_path(path), label + " must not be a symlink: " + path_text(path));
    require(fs::is_regular_file(path, ec), label + " is not a regular file: " + path_text(path));
}

uint64_t file_size_or_fail(const fs::path & path) {
    std::error_code ec;
    const uint64_t size = fs::file_size(path, ec);
    require(!ec, "cannot stat file: " + path_text(path));
    return size;
}

void require_little_endian() {
    const uint16_t probe = 1;
    require(*reinterpret_cast<const uint8_t *>(&probe) == 1,
            "raw f32/u32 capture requires a little-endian host");
}

std::string read_text(const fs::path & path, size_t max_bytes, const std::string & label) {
    require_regular_file(path, label);
    const uint64_t size = file_size_or_fail(path);
    require(size <= max_bytes, label + " exceeds the bounded read size");
    std::ifstream in(path, std::ios::binary);
    require(bool(in), "cannot open " + label + ": " + path_text(path));
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    require(bool(in) || in.eof(), "short read: " + path_text(path));
    return text;
}

json read_json_file(const fs::path & path, size_t max_bytes, const std::string & label) {
    const std::string text = read_text(path, max_bytes, label);
    try {
        return json::parse(text);
    } catch (const std::exception & e) {
        fail(label + " is not valid JSON: " + e.what());
    }
}

void require_new_path(const fs::path & path) {
    std::error_code ec;
    const fs::file_status status = fs::symlink_status(path, ec);
    require(!fs::is_symlink(status), "refusing symlink path: " + path_text(path));
    require(!fs::exists(status), "refusing to overwrite: " + path_text(path));
}

void ensure_new_directory(const fs::path & path) {
    require_new_path(path);
    const fs::path parent = path.parent_path().empty() ? fs::path(".") : path.parent_path();
    std::error_code ec;
    require(fs::is_directory(parent, ec), "output parent is not a directory: " + path_text(parent));
    require(fs::create_directory(path, ec) && !ec, "cannot create output directory: " + path_text(path));
}

void write_bytes_new(const fs::path & path, const void * data, size_t bytes) {
    require_new_path(path);
    std::ofstream out(path, std::ios::binary | std::ios::out | std::ios::trunc);
    require(bool(out), "cannot create: " + path_text(path));
    if (bytes != 0) {
        out.write(static_cast<const char *>(data), static_cast<std::streamsize>(bytes));
    }
    out.flush();
    require(bool(out), "short write: " + path_text(path));
}

void write_json_new(const fs::path & path, const json & value) {
    const std::string text = value.dump(2) + "\n";
    write_bytes_new(path, text.data(), text.size());
}

// Compact SHA-256; used for step logits and the exact token history.
struct sha256_ctx {
    uint32_t state[8];
    uint64_t bits;
    uint8_t  buffer[64];
    size_t   used;

    sha256_ctx() : state{0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                         0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u}, bits(0), used(0) {}

    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

    void block(const uint8_t * p) {
        static const uint32_t k[64] = {
            0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
            0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
            0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
            0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
            0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
            0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
            0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
            0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (uint32_t(p[4*i]) << 24) | (uint32_t(p[4*i+1]) << 16) |
                   (uint32_t(p[4*i+2]) << 8) | uint32_t(p[4*i+3]);
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3);
            const uint32_t s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
        uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = h + s1 + ch + k[i] + w[i];
            const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = s0 + maj;
            h = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        state[0] += a; state[1] += b; state[2] += c; state[3] += d;
        state[4] += e; state[5] += f; state[6] += g; state[7] += h;
    }

    void update(const void * data, size_t bytes) {
        const uint8_t * p = static_cast<const uint8_t *>(data);
        bits += uint64_t(bytes) * 8u;
        while (bytes != 0) {
            const size_t take = std::min(bytes, sizeof(buffer) - used);
            std::memcpy(buffer + used, p, take);
            used += take;
            p += take;
            bytes -= take;
            if (used == sizeof(buffer)) {
                block(buffer);
                used = 0;
            }
        }
    }

    void finish(uint8_t out[32]) {
        const uint64_t saved = bits;
        const uint8_t pad = 0x80;
        update(&pad, 1);
        const uint8_t zero = 0;
        while (used != 56) update(&zero, 1);
        uint8_t length[8];
        for (int i = 0; i < 8; ++i) length[7-i] = uint8_t(saved >> (8 * i));
        update(length, sizeof(length));
        for (int i = 0; i < 8; ++i) {
            out[4*i]   = uint8_t(state[i] >> 24);
            out[4*i+1] = uint8_t(state[i] >> 16);
            out[4*i+2] = uint8_t(state[i] >> 8);
            out[4*i+3] = uint8_t(state[i]);
        }
    }
};

std::string sha256_digest(sha256_ctx ctx) {
    uint8_t digest[32];
    ctx.finish(digest);
    static const char hex[] = "0123456789abcdef";
    std::string out(64, '0');
    for (int i = 0; i < 32; ++i) {
        out[2*i] = hex[digest[i] >> 4];
        out[2*i+1] = hex[digest[i] & 0x0f];
    }
    return out;
}

std::string sha256_bytes(const void * data, size_t bytes) {
    sha256_ctx ctx;
    ctx.update(data, bytes);
    return sha256_digest(ctx);
}

std::vector<llama_token> parse_token_array(const json & doc, const char * key, size_t max_count, bool required) {
    const json * array = &doc;
    if (doc.is_object()) {
        if (!doc.contains(key)) {
            require(!required, std::string("missing JSON field: ") + key);
            return {};
        }
        array = &doc.at(key);
    }
    require(array->is_array(), std::string(key) + " must be an array");
    require(array->size() <= max_count, std::string(key) + " exceeds the bounded token count");
    std::vector<llama_token> out;
    out.reserve(array->size());
    for (const auto & item : *array) {
        require(item.is_number_integer(), std::string(key) + " contains a non-integer token");
        const int64_t value = item.get<int64_t>();
        require(value >= 0 && value <= std::numeric_limits<int32_t>::max(),
                std::string(key) + " contains an out-of-range token");
        out.push_back(static_cast<llama_token>(value));
    }
    return out;
}

std::vector<llama_token> read_prompt(const fs::path & path) {
    const json doc = read_json_file(path, kMaxJsonBytes, "prompt JSON");
    return parse_token_array(doc, "tokens", kMaxContext, true);
}

std::vector<llama_token> read_teacher(const fs::path & path, size_t steps) {
    if (path.empty()) return {};
    const json doc = read_json_file(path, kMaxJsonBytes, "teacher JSON");
    std::vector<llama_token> ids = parse_token_array(doc, "teacher_ids", 4096, true);
    require(ids.size() >= steps, "teacher JSON has fewer ids than --teacher-steps");
    ids.resize(steps);
    return ids;
}

// server_tokens::deserialize v1 text envelope. A marked envelope never falls
// back to raw ids on malformed input.
std::vector<llama_token> unpack_prefix_tokens(const std::vector<llama_token> & packed, bool & server_v1) {
    require(!packed.empty(), "empty restored token payload");
    server_v1 = packed.front() == LLAMA_TOKEN_NULL;
    if (!server_v1) return packed;
    require(packed.size() >= 4, "truncated server token envelope");
    require(static_cast<uint32_t>(packed[1]) == 1, "unsupported server token state version");
    const size_t count = static_cast<uint32_t>(packed[2]);
    require(count <= packed.size() - 4, "truncated server token IDs");
    require(static_cast<uint32_t>(packed[3 + count]) == 0, "media tokens are unsupported in this prefix");
    require(packed.size() == count + 4, "trailing data in server token envelope");
    return std::vector<llama_token>(packed.begin() + 3, packed.begin() + 3 + count);
}

bool prefix_packed_count_ok(uint32_t p, size_t packed_count) {
    return packed_count == p || packed_count == size_t(p) + 4;
}

// common/arg.cpp::parse_device_list returns {device, nullptr} for one GPU.
// This inspector is deliberately pointer-only so the CPU self-test can prove
// the count/terminator contract with opaque addresses and no backend calls.
struct device_slots {
    size_t non_null = 0;
    bool terminated = false;
    bool leading_null = false;
    bool holes = false;
};

device_slots inspect_device_slots(const std::vector<ggml_backend_dev_t> & devices) {
    device_slots slots;
    for (size_t i = 0; i < devices.size(); ++i) {
        if (devices[i] != nullptr) {
            ++slots.non_null;
            if (i + 1 == devices.size()) slots.terminated = false;
            continue;
        }
        if (i == 0) slots.leading_null = true;
        if (i + 1 == devices.size()) slots.terminated = true;
        else if (devices[i + 1] != nullptr) slots.holes = true;
    }
    return slots;
}

bool one_device_with_terminator(const std::vector<ggml_backend_dev_t> & devices) {
    const device_slots slots = inspect_device_slots(devices);
    return slots.non_null == 1 && slots.terminated && !slots.leading_null && !slots.holes &&
           devices.size() == 2;
}

struct decode_step {
    uint32_t start = 0;
    uint32_t count = 0;
    bool want_logits = false;
    std::string kind;
};

std::vector<decode_step> plan_cadence(uint32_t p, uint32_t length) {
    require(length > p, "prompt must extend beyond the restored prefix");
    require((length - p) % 512u == 0, "prompt tail must be a multiple of 512 tokens");
    std::vector<decode_step> plan;
    uint32_t start = p;
    while (start < length) {
        const uint32_t remaining = length - start;
        uint32_t count = std::min<uint32_t>(kBatch, remaining);
        // Carve the final 512 before choosing the logical parent.
        if (remaining > 512 && count == remaining) count -= 512;
        const bool final = start + count == length;
        if (final && count == 512) {
            plan.push_back({start,        256, false, "seam-head-256"});
            plan.push_back({start + 256,  128, false, "seam-middle-128"});
            plan.push_back({start + 384,  128, true,  "seam-tail-128"});
        } else if (final && count == 256) {
            plan.push_back({start,        128, false, "seam-head-128"});
            plan.push_back({start + 128,  128, true,  "seam-tail-128"});
        } else {
            plan.push_back({start, count, final, final ? "parent-final" : "parent"});
        }
        start += count;
    }
    uint32_t covered = 0;
    for (const auto & step : plan) {
        require(step.start == p + covered, "cadence plan is not contiguous");
        require(step.count > 0 && step.count <= kBatch, "cadence plan has an invalid parent");
        covered += step.count;
    }
    require(covered == length - p, "cadence plan does not cover the prompt tail");
    require(!plan.empty() && plan.back().want_logits && plan.back().count == 128,
            "cadence plan does not end in an exact logits seam");
    return plan;
}

struct cadence_counts {
    size_t api_calls = 0;
    size_t ubatch_512 = 0;
    size_t seam_256 = 0;
    size_t seam_128 = 0;
};

cadence_counts count_cadence(const std::vector<decode_step> & plan) {
    cadence_counts counts;
    counts.api_calls = plan.size();
    for (const auto & step : plan) {
        if (step.count % kUbatch == 0) counts.ubatch_512 += step.count / kUbatch;
        if (step.count == 256) ++counts.seam_256;
        if (step.count == 128) ++counts.seam_128;
    }
    return counts;
}

json cadence_json(const cadence_counts & counts) {
    return {{"api_calls", counts.api_calls}, {"ubatch_512", counts.ubatch_512},
            {"seam_256", counts.seam_256}, {"seam_128", counts.seam_128}};
}

json plan_json(const std::vector<decode_step> & plan) {
    json steps = json::array();
    for (const auto & step : plan) {
        steps.push_back({{"start", step.start}, {"count", step.count},
                         {"want_logits", step.want_logits}, {"kind", step.kind}});
    }
    return steps;
}

uint32_t argmax_lowest(const float * logits, size_t vocab) {
    require(vocab != 0, "empty logits vector");
    uint32_t best = 0;
    for (size_t i = 1; i < vocab; ++i) {
        if (logits[i] > logits[best]) best = static_cast<uint32_t>(i);
    }
    return best;
}

llama_token select_teacher(const std::vector<llama_token> & fixed, size_t step, uint32_t argmax) {
    return step < fixed.size() ? fixed[step] : static_cast<llama_token>(argmax);
}

std::string step_stem(size_t step) {
    std::ostringstream out;
    out << std::setw(4) << std::setfill('0') << step;
    return out.str();
}

json write_step(const fs::path & dir, size_t step, uint32_t vocab, const float * logits, size_t count,
                uint32_t position, llama_token teacher_id, const std::string & prompt_sha,
                const json & cadence) {
    require(count == vocab, "logits row does not match the declared vocabulary");
    for (size_t i = 0; i < count; ++i) {
        require(std::isfinite(logits[i]), "non-finite logit at index " + std::to_string(i));
    }
    const std::string stem = step_stem(step);
    const std::string filename = stem + ".logits.f32.bin";
    const uint32_t argmax = argmax_lowest(logits, count);
    write_bytes_new(dir / filename, logits, count * sizeof(float));
    const std::string digest = sha256_bytes(logits, count * sizeof(float));
    const json meta = {
        {"schema", "rkva-quality-raw-capture-step-v1"},
        {"label", "RAW_CAPTURE_ONLY"},
        {"step", step},
        {"position", position},
        {"token_count", uint64_t(position) + 1},
        {"teacher_id", static_cast<int64_t>(teacher_id)},
        {"prompt_sha256", prompt_sha},
        {"argmax", argmax},
        {"vocab", static_cast<uint64_t>(vocab)},
        {"filename", filename},
        {"bytes", count * sizeof(float)},
        {"finite", true},
        {"sha256", digest},
        {"cadence", cadence},
    };
    write_json_new(dir / (stem + ".json"), meta);
    return meta;
}

std::vector<std::string> read_common_args(const fs::path & path) {
    const json doc = read_json_file(path, 4u * 1024u * 1024u, "common args JSON");
    require(doc.is_array(), "common args JSON must be an array of argv strings");
    require(!doc.empty() && doc.size() <= 4096, "common args JSON has an invalid argv size");
    std::vector<std::string> args;
    args.reserve(doc.size());
    for (const auto & item : doc) {
        require(item.is_string(), "common args JSON must contain only strings");
        const std::string value = item.get<std::string>();
        require(!value.empty() && value.find('\0') == std::string::npos,
                "common args JSON contains an empty or NUL argv entry");
        require(value.size() <= 8192, "common args JSON contains an oversized argv entry");
        args.push_back(value);
    }
    return args;
}

struct options {
    bool help = false;
    bool execute = false;
    bool preview = false;
    bool self_test = false;
    fs::path self_test_json;
    std::string common_args_json;
    std::string prompt_json;
    std::string prefix_snapshot;
    std::string teacher_json;
    fs::path output;
    uint32_t p = 0;
    uint32_t context = 0;
    uint32_t teacher_steps = 0;
};

std::optional<std::string> arg_value(const std::vector<std::string> & argv, const std::string & name) {
    for (size_t i = 0; i < argv.size(); ++i) {
        if (argv[i] == name && i + 1 < argv.size()) return argv[i + 1];
        const std::string prefix = name + "=";
        if (argv[i].compare(0, prefix.size(), prefix) == 0) return argv[i].substr(prefix.size());
    }
    return std::nullopt;
}

bool has_flag(const std::vector<std::string> & argv, const std::string & name) {
    return std::find(argv.begin(), argv.end(), name) != argv.end();
}

void require_arg(const std::vector<std::string> & argv, const std::string & name, const std::string & expected) {
    const auto value = arg_value(argv, name);
    require(value.has_value(), "common args must explicitly set " + name);
    require(*value == expected, "common args require " + name + " " + expected + ", got " + *value);
}

// Lexical contract used by preview and re-checked before parsing. It rejects
// any argv that does not pin geometry, model, device, cache, and wire format.
void validate_common_argv(const std::vector<std::string> & argv, const options & opt,
                          std::string & model_path) {
    require(argv.size() >= 2, "common args must include argv0 and at least one option");
    for (const auto & arg : argv) {
        require(arg != "--help" && arg != "-h" && arg != "--usage" && arg != "--version" &&
                arg != "--list-devices" && arg != "--completion",
                "common args must not request help, version, device listing, or completion");
    }
    require_arg(argv, "--ctx-size", std::to_string(opt.context));
    require_arg(argv, "--batch-size", std::to_string(kBatch));
    require_arg(argv, "--ubatch-size", std::to_string(kUbatch));
    require_arg(argv, "--parallel", "1");
    require_arg(argv, "--gpu-layers", "99");
    require_arg(argv, "--device", "CUDA0");
    require_arg(argv, "--flash-attn", "on");
    require_arg(argv, "--fit", "off");
    require_arg(argv, "--cache-type-k", "kvarn4");
    require_arg(argv, "--cache-type-v", "kvarn4");
    require_arg(argv, "--remote-attn-layers", "full");
    require_arg(argv, "--remote-attn-cache-type-k", "q4_0");
    require_arg(argv, "--remote-attn-cache-type-v", "q4_0");
    require_arg(argv, "--remote-attn-format", "kvarn4");
    require_arg(argv, "--remote-attn-wire-format", "fp8");
    require_arg(argv, "--kv-position-split-p", std::to_string(opt.p));
    require(has_flag(argv, "--no-warmup"), "common args must include --no-warmup");
    require(has_flag(argv, "--no-context-shift"), "common args must include --no-context-shift");
    const auto remote = arg_value(argv, "--remote-attn");
    require(remote.has_value() && !remote->empty(), "common args must set a non-empty --remote-attn");
    const auto spec = arg_value(argv, "--spec-type");
    require(!spec.has_value() || *spec == "none", "common args must not enable speculative decoding");
    const auto spec_n = arg_value(argv, "--spec-draft-n-max");
    require(!spec_n.has_value() || *spec_n == "0", "common args must set --spec-draft-n-max 0");
    const auto tail = arg_value(argv, "--kv-tail-tokens");
    require(!tail.has_value() || *tail == "0", "common args must keep --kv-tail-tokens 0");
    const auto prefill = arg_value(argv, "--remote-attn-prefill");
    require(!prefill.has_value() || *prefill == "remote", "common args must keep remote prefill");
    const auto model = arg_value(argv, "-m").has_value() ? arg_value(argv, "-m") : arg_value(argv, "--model");
    require(model.has_value() && !model->empty(), "common args must set a local -m model path");
    model_path = *model;
    require_regular_file(model_path, "model");
}

struct validated {
    std::vector<llama_token> prompt;
    std::vector<llama_token> teacher;
    std::vector<std::string> common_args;
    std::string model_path;
};

uint32_t parse_u32(const std::string & text, const char * flag) {
    require(!text.empty(), std::string(flag) + " requires an integer");
    size_t used = 0;
    long long value = 0;
    try {
        value = std::stoll(text, &used, 10);
    } catch (...) {
        fail(std::string(flag) + " requires an integer");
    }
    require(used == text.size() && value >= 0 && value <= std::numeric_limits<uint32_t>::max(),
            std::string(flag) + " requires a non-negative 32-bit integer");
    return static_cast<uint32_t>(value);
}

options parse_cli(int argc, char ** argv) {
    options opt;
    auto take = [&](int & i, const char * flag) -> std::string {
        require(i + 1 < argc, std::string(flag) + " requires a value");
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") opt.help = true;
        else if (arg == "--execute") opt.execute = true;
        else if (arg == "--preview") opt.preview = true;
        else if (arg == "--self-test") opt.self_test = true;
        else if (arg == "--self-test-json") opt.self_test_json = take(i, "--self-test-json");
        else if (arg == "--common-args-json") opt.common_args_json = take(i, "--common-args-json");
        else if (arg == "--prompt-json") opt.prompt_json = take(i, "--prompt-json");
        else if (arg == "--prefix-snapshot") opt.prefix_snapshot = take(i, "--prefix-snapshot");
        else if (arg == "--teacher-json") opt.teacher_json = take(i, "--teacher-json");
        else if (arg == "--output") opt.output = take(i, "--output");
        else if (arg == "--p") opt.p = parse_u32(take(i, "--p"), "--p");
        else if (arg == "--context") opt.context = parse_u32(take(i, "--context"), "--context");
        else if (arg == "--teacher-steps") opt.teacher_steps = parse_u32(take(i, "--teacher-steps"), "--teacher-steps");
        else fail("unknown argument: " + arg);
    }
    if (opt.self_test) {
        require(!opt.execute && !opt.preview && opt.common_args_json.empty() && opt.prompt_json.empty() &&
                opt.prefix_snapshot.empty() && opt.teacher_json.empty() && opt.output.empty() &&
                opt.p == 0 && opt.context == 0 && opt.teacher_steps == 0,
                "--self-test must run alone");
    } else if (!opt.help) {
        require(!(opt.execute && opt.preview), "choose --execute or --preview");
        if (!opt.self_test_json.empty()) fail("--self-test-json requires --self-test");
    }
    return opt;
}

void print_usage() {
    std::cout <<
        "rkva_quality_capture - raw full-vocab logits (RAW_CAPTURE_ONLY)\n"
        "\n"
        "  rkva_quality_capture --help\n"
        "  rkva_quality_capture --self-test [--self-test-json FILE]\n"
        "  rkva_quality_capture [--preview] --common-args-json FILE --prompt-json FILE\n"
        "      --prefix-snapshot FILE --p 102400 --context 131072 --teacher-steps 128\n"
        "      [--teacher-json FILE] --output DIR\n"
        "  rkva_quality_capture --execute ... (hardware owner only)\n"
        "\n"
        "Defaults: preview. No backend/model/network is touched without --execute.\n";
}

validated validate_options(const options & opt) {
    require(!opt.common_args_json.empty(), "--common-args-json is required");
    require(!opt.prompt_json.empty(), "--prompt-json is required");
    require(!opt.prefix_snapshot.empty(), "--prefix-snapshot is required");
    require(!opt.output.empty(), "--output is required");
    require(opt.p != 0, "--p is required");
    require(opt.context != 0, "--context is required");
    require(opt.teacher_steps != 0, "--teacher-steps is required");
    require(opt.context <= kMaxContext, "--context exceeds the hard cap of 160000");
    require(opt.context == kBaselineContext, "--context must be exactly 131072 for this capture");
    require(opt.p == kBaselineP, "--p must be exactly 102400 for this capture");
    require(opt.teacher_steps <= 1024, "--teacher-steps is too large");
    require_regular_file(opt.common_args_json, "common args JSON");
    require_regular_file(opt.prompt_json, "prompt JSON");
    require_regular_file(opt.prefix_snapshot, "prefix snapshot");
    if (!opt.teacher_json.empty()) require_regular_file(opt.teacher_json, "teacher JSON");
    const fs::path parent = opt.output.parent_path().empty() ? fs::path(".") : opt.output.parent_path();
    std::error_code ec;
    require(fs::is_directory(parent, ec), "output parent is not a directory: " + path_text(parent));
    require_new_path(opt.output);

    validated out;
    out.common_args = read_common_args(opt.common_args_json);
    validate_common_argv(out.common_args, opt, out.model_path);
    out.prompt = read_prompt(opt.prompt_json);
    const uint32_t length = static_cast<uint32_t>(out.prompt.size());
    require(length >= opt.p + 512, "prompt is shorter than P+512");
    require(length >= opt.p && (length - opt.p) % 512u == 0, "prompt tail must be a multiple of 512");
    require(uint64_t(length) + opt.teacher_steps <= opt.context,
            "prompt length + teacher steps must stay inside the context");
    out.teacher = read_teacher(opt.teacher_json, opt.teacher_steps);
    return out;
}

std::string env_value(const char * name) {
    const char * value = std::getenv(name);
    return value ? std::string(value) : std::string();
}

void require_env(const char * name, const char * expected) {
    const std::string value = env_value(name);
    require(value == expected, std::string("environment requires ") + name + "=" + expected +
            ", got '" + value + "'");
}

void validate_environment() {
    require_env("GGML_RKVA_NATIVE_KVARN4_EXPERIMENTAL", "1");
    require_env("GGML_RKVA_NATIVE_PRODUCER", "cuda");
    require_env("GGML_RKVA_RANGE_STORAGE", "kvarn4");
    require_env("GGML_RKVA_RANGE_KERNEL", "fine");
    require_env("GGML_RKVA_NATIVE_TAIL_TOKENS", "128");
    require_env("GGML_RKVA_PREFILL_QUERY_CHUNK", "8");
    require_env("GGML_RKVA_NATIVE_BULK", "f16");
    require_env("GGML_RKVA_WIRE_FORMAT", "fp8");
    require_env("GGML_RKVA_WIRE_STRICT", "1");
    require_env("GGML_RKVA_ASYNC_LOCAL_REMOTE", "1");
    require_env("GGML_RKVA_CUDA_FP8", "mapped");
    require_env("GGML_RKVA_DISK_SNAPSHOTS", "0");
    require_env("GGML_RKVA_WAVEFRONT_TILE", "256");
    require_env("GGML_RKVA_WAVEFRONT_RPC_BATCH", "1");
    require_env("GGML_RKVA_TCP_PREFETCH", "1");
    require_env("GGML_RKVA_LOCAL_INPUT_REUSE", "0");
    require_env("GGML_RKVA_PROFILE", "1");
    require_env("GGML_KVARN_WINDOW_CHUNK", "4096");
    require_env("GGML_CUDA_GRAPH_RECOVERY_HEADROOM_MB", "128");
    require_env("GGML_CUDA_GRAPH_GUARD_SAMPLE_MS", "10");
    require_env("GGML_CUDA_GRAPH_GUARD_SLACK_MB", "64");
    require_env("LLAMA_N256_LOOKAHEAD", "0");
    require(std::getenv("GGML_CUDA_MASK_PACK") == nullptr,
            "GGML_CUDA_MASK_PACK must be unset (default mask-pack policy)");
}

json environment_json() {
    std::map<std::string, std::string> selected;
    for (char ** item = environ; item && *item; ++item) {
        const std::string entry(*item);
        const size_t eq = entry.find('=');
        if (eq == std::string::npos) continue;
        const std::string name = entry.substr(0, eq);
        if (name.compare(0, 5, "GGML_") == 0 || name.compare(0, 6, "LLAMA_") == 0 ||
                name.compare(0, 5, "CUDA_") == 0 || name.compare(0, 7, "XLLAMA_") == 0) {
            selected[name] = entry.substr(eq + 1);
        }
    }
    json out = json::object();
    for (const auto & item : selected) out[item.first] = item.second;
    return out;
}

common_params parse_common_args(const std::vector<std::string> & args) {
    common_params params;
    std::vector<std::string> storage = args;
    std::vector<char *> argv;
    argv.reserve(storage.size() + 1);
    for (auto & arg : storage) argv.push_back(arg.data());
    argv.push_back(nullptr);
    if (!common_params_parse(static_cast<int>(storage.size()), argv.data(), params, LLAMA_EXAMPLE_SERVER)) {
        fail("common_params_parse rejected the common args");
    }
    // The capture never warms up, never loads the MTP head, and never runs a
    // draft/speculative context. One process, one model, one context.
    params.warmup = false;
    params.load_mtp = false;
    params.speculative.types.assign(1, COMMON_SPECULATIVE_TYPE_NONE);
    params.speculative.draft.n_max = 0;
    params.speculative.draft.mparams = {};
    params.speculative.n_rs_seq_target = 0;
    // Reproduce tools/server/server-context.cpp::server_output_limits: with
    // one slot and no draft the HTTP server asks for exactly one output row.
    // The common default 0 would mean n_batch (1024) logits rows (~970 MiB).
    const common_speculative_output_limits limits = common_speculative_get_output_limits(
            params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));
    params.n_outputs_max = std::max<int32_t>(1, limits.total);
    params.n_outputs_max_per_seq = std::max<int32_t>(1, limits.per_seq);
    // HTTP n_probs uses dense host logits; do not attach a backend sampler.
    params.sampling.backend_sampling = false;
    return params;
}

void validate_params(const common_params & params, const options & opt) {
    require(params.n_ctx == kBaselineContext, "parsed --ctx-size is not the baseline 131072");
    require(params.n_batch == kBatch, "parsed --batch-size is not 1024");
    require(params.n_ubatch == kUbatch, "parsed --ubatch-size is not 512");
    require(params.n_parallel == 1, "parsed --parallel is not 1");
    require(params.n_sequences == 1, "parsed sequence count is not 1");
    require(params.n_gpu_layers == 99, "parsed --gpu-layers is not 99");
    require(params.main_gpu == 0, "parsed --main-gpu is not 0");
    require(params.fit_params == false, "parsed --fit must be off");
    require(params.warmup == false, "parsed warmup must be off");
    require(params.load_mtp == false, "MTP weights must not be loaded");
    require(params.ctx_shift == false, "context shift must be off");
    require(params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_ENABLED, "Flash Attention must be on");
    require(params.cache_type_k == GGML_TYPE_Q4_0 && params.cache_type_v == GGML_TYPE_Q4_0,
            "KVarN4 backing cache must be q4_0/q4_0");
    require(params.cache_kvarn_bits_k == 4 && params.cache_kvarn_bits_v == 4,
            "parsed cache is not KVarN4/KVarN4");
    require(params.kvarn.type == LLAMA_KVARN_K4V4_G128, "parsed KVarN type is not K4V4_G128");
    require(params.kvarn.key_bits == 4 && params.kvarn.value_bits == 4 && params.kvarn.group == 128,
            "parsed KVarN descriptor is not 4/4/g128");
    require(params.kv_tail_tokens == "0", "KV tail tokens must be 0");
    require(params.grp_attn_n == 1, "group attention is unsupported");
    require(params.remote_attn_host.size() != 0, "remote attention host is missing");
    require(params.remote_attn_prefill == "remote", "remote prefill must stay remote");
    require(params.remote_attn_layers == "full", "remote attention must cover full layers");
    require(params.remote_attn_format == "kvarn4", "remote upload format must be kvarn4");
    require(params.remote_attn_wire_format == "fp8", "remote wire format must be fp8");
    require(params.remote_attn_cache_type_k == GGML_TYPE_Q4_0 &&
            params.remote_attn_cache_type_v == GGML_TYPE_Q4_0,
            "remote cache type must be q4_0/q4_0");
    require(params.kv_position_split_p == opt.p, "parsed --kv-position-split-p differs from --p");
    require(params.speculative.types.size() == 1 &&
            params.speculative.types.front() == COMMON_SPECULATIVE_TYPE_NONE,
            "speculative decoding must be disabled");
    require(params.speculative.draft.n_max == 0, "draft n_max must be 0");
    require(params.speculative.need_n_rs_seq() == 0, "n_rs_seq must be 0");
    require(params.speculative.n_rs_seq_target == 0, "n_rs_seq target must be 0");
    require(params.sampling.backend_sampling == false, "backend sampling must be off for dense logits");
    require(params.embedding == false, "embedding mode is outside the capture scope");
    require(params.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED ||
            params.pooling_type == LLAMA_POOLING_TYPE_NONE, "pooling mode is outside the capture scope");
    require(params.n_outputs_max == 1, "output limit total must be pinned to 1");
    require(params.n_outputs_max_per_seq == 1, "output limit per sequence must be pinned to 1");
    require_regular_file(params.model.path, "parsed model");
    require(one_device_with_terminator(params.devices),
            "device list must be exactly one GPU plus a null terminator");
    const ggml_backend_dev_t device = params.devices.front();
    require(device != nullptr, "device list has no GPU");
    require(ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU,
            "CPU device substitution is not allowed");
    const char * device_name = ggml_backend_dev_name(device);
    require(device_name && std::string(device_name) == "CUDA0", "selected device is not CUDA0");
}

json runtime_geometry(llama_context * ctx) {
    const llama_cparams & cparams = ctx->get_cparams();
    return {
        {"n_ctx", llama_n_ctx(ctx)},
        {"n_batch", llama_n_batch(ctx)},
        {"n_ubatch", llama_n_ubatch(ctx)},
        {"n_ubatch_physical", llama_n_ubatch_physical(ctx)},
        {"n_seq_max", llama_n_seq_max(ctx)},
        {"n_rs_seq", llama_n_rs_seq(ctx)},
        {"n_outputs_max", cparams.n_outputs_max},
        {"n_outputs_max_per_seq", cparams.n_outputs_max_per_seq},
    };
}

json model_config(llama_model * model, const common_params & params) {
    char desc[512] = {0};
    const int desc_rc = llama_model_desc(model, desc, sizeof(desc));
    const llama_vocab * vocab = llama_model_get_vocab(model);
    require(vocab != nullptr, "model vocabulary is unavailable");
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    require(n_vocab > 0, "model vocabulary is empty");
    return {
        {"path", params.model.path},
        {"desc", desc_rc > 0 ? std::string(desc) : std::string()},
        {"n_vocab", n_vocab},
        {"n_ctx_train", llama_model_n_ctx_train(model)},
        {"n_embd", llama_model_n_embd(model)},
        {"n_layer", llama_model_n_layer(model)},
        {"n_layer_nextn", llama_model_n_layer_nextn(model)},
        {"n_head", llama_model_n_head(model)},
        {"n_head_kv", llama_model_n_head_kv(model)},
        {"n_params", llama_model_n_params(model)},
        {"size_bytes", llama_model_size(model)},
    };
}

json declared_config(const common_params & params, const options & opt) {
    return {
        {"context", params.n_ctx},
        {"p", opt.p},
        {"batch", params.n_batch},
        {"ubatch", params.n_ubatch},
        {"parallel", params.n_parallel},
        {"gpu_layers", params.n_gpu_layers},
        {"device", "CUDA0"},
        {"flash_attn", "on"},
        {"fit", params.fit_params},
        {"warmup", params.warmup},
        {"load_mtp", params.load_mtp},
        {"n_rs_seq", params.speculative.need_n_rs_seq()},
        {"output_limits", {
            {"total", params.n_outputs_max},
            {"per_seq", params.n_outputs_max_per_seq},
            {"source", "server_output_limits"},
        }},
        {"sampling_backend", params.sampling.backend_sampling},
        {"spec_types", "none"},
        {"cache_k", "kvarn4"},
        {"cache_v", "kvarn4"},
        {"kvarn", {
            {"type", llama_kvarn_type_name(params.kvarn.type)},
            {"key_bits", params.kvarn.key_bits},
            {"value_bits", params.kvarn.value_bits},
            {"group", params.kvarn.group},
        }},
        {"remote", {
            {"host", params.remote_attn_host},
            {"prefill", params.remote_attn_prefill},
            {"layers", params.remote_attn_layers},
            {"format", params.remote_attn_format},
            {"wire_format", params.remote_attn_wire_format},
            {"cache_k", "q4_0"},
            {"cache_v", "q4_0"},
        }},
        {"kv_position_split_p", params.kv_position_split_p},
        {"kv_tail_tokens", params.kv_tail_tokens},
        {"ctx_shift", params.ctx_shift},
        {"grp_attn_n", params.grp_attn_n},
    };
}

struct decode_state {
    llama_context * ctx = nullptr;
    llama_batch batch{};
    std::vector<llama_token> history;
    size_t vocab = 0;
    uint32_t sampled_logits_count = 0;
    uint32_t sampled_candidates_count = 0;
    std::string sampled_candidates_source = "unchecked";
    bool logits_contract_checked = false;

    ~decode_state() {
        if (batch.token != nullptr) llama_batch_free(batch);
    }

    const float * decode(uint32_t start, uint32_t count, bool want_logits) {
        require(count > 0 && count <= kBatch, "decode size is outside the parent bound");
        require(uint64_t(start) + count <= history.size(), "decode range exceeds the token history");
        batch.n_tokens = static_cast<int32_t>(count);
        for (uint32_t i = 0; i < count; ++i) {
            batch.token[i] = history[start + i];
            batch.pos[i] = static_cast<llama_pos>(start + i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = want_logits && i + 1 == count;
        }
        require(llama_decode(ctx, batch) == 0, "llama_decode failed at position " + std::to_string(start));
        llama_synchronize(ctx);
        if (!want_logits) return nullptr;
        const int32_t out_index = static_cast<int32_t>(count - 1);
        // llama_get_logits_ith prefers backend-sampled logits. With
        // backend_sampling=false there must be no sampled logits and no compact
        // candidate set. The public candidates getter may return the
        // full-vocabulary identity fallback (llama-context.cpp), so the
        // authoritative no-candidates signal is candidates_count == 0.
        require(llama_get_sampled_logits_ith(ctx, out_index) == nullptr,
                "backend sampled logits are active; refusing compact logits");
        const uint32_t sampled_count = llama_get_sampled_logits_count_ith(ctx, out_index);
        require(sampled_count == static_cast<uint32_t>(vocab),
                "sampled logits count is not the full vocabulary");
        const uint32_t candidates_count = llama_get_sampled_candidates_count_ith(ctx, out_index);
        require(candidates_count == 0, "backend sampler candidates are active; refusing compact candidates");
        const llama_token * candidates = llama_get_sampled_candidates_ith(ctx, out_index);
        if (candidates != nullptr) {
            require(candidates[0] == 0 && candidates[vocab - 1] == static_cast<llama_token>(vocab - 1),
                    "sampled candidates are not the full-vocab identity fallback");
        }
        sampled_candidates_source = candidates ? "full-vocab-fallback" : "null";
        sampled_logits_count = sampled_count;
        sampled_candidates_count = candidates_count;
        logits_contract_checked = true;
        const float * logits = llama_get_logits_ith(ctx, out_index);
        require(logits != nullptr, "requested last-row logits are missing");
        return logits;
    }
};

json run_capture(const options & opt, const validated & input, common_params & params,
                 size_t & completed_steps) {
    require_little_endian();
    auto init = common_init_from_params(params);
    require(bool(init) && init->model() != nullptr && init->context() != nullptr,
            "common_init_from_params did not produce a model and context");
    llama_model * model = init->model();
    llama_context * ctx = init->context();
    llama_memory_t memory = llama_get_memory(ctx);
    require(memory != nullptr, "context memory is unavailable");

    const json geometry = runtime_geometry(ctx);
    require(geometry.at("n_ctx") == kBaselineContext, "runtime context is not 131072");
    require(geometry.at("n_batch") == kBatch, "runtime batch is not 1024");
    require(geometry.at("n_ubatch") == kUbatch, "runtime ubatch is not 512");
    require(geometry.at("n_ubatch_physical") == kPhysicalUbatch, "runtime physical ubatch is not 256");
    require(geometry.at("n_seq_max") == 1, "runtime sequence count is not 1");
    require(geometry.at("n_rs_seq") == 0, "runtime n_rs_seq is not 0");
    require(geometry.at("n_outputs_max") == 1, "runtime output limit total is not 1");
    require(geometry.at("n_outputs_max_per_seq") == 1, "runtime output limit per sequence is not 1");
    require(llama_memory_seq_pos_max(memory, 0) == -1, "context is not empty before prefix restore");

    const json config = model_config(model, params);
    const size_t vocab = static_cast<size_t>(config.at("n_vocab").get<int32_t>());
    for (llama_token id : input.prompt) require(id < static_cast<llama_token>(vocab), "prompt token is out of vocab");
    for (llama_token id : input.teacher) require(id < static_cast<llama_token>(vocab), "teacher token is out of vocab");

    const std::string prefix_path = path_text(opt.prefix_snapshot);
    const uint64_t prefix_size = file_size_or_fail(opt.prefix_snapshot);
    size_t packed_count = 0;
    const size_t header_read = llama_state_seq_load_file(ctx, prefix_path.c_str(), 0, nullptr, 0, &packed_count);
    require(header_read > 0 && prefix_packed_count_ok(opt.p, packed_count),
            "prefix snapshot has an unexpected packed token count; P-128 is refused in this prototype");
    std::vector<llama_token> packed(packed_count);
    size_t actual_packed = 0;
    const size_t state_read = llama_state_seq_load_file(ctx, prefix_path.c_str(), 0, packed.data(),
                                                        packed.size(), &actual_packed);
    require(state_read > 0 && actual_packed == packed_count, "prefix snapshot load failed or changed size");
    bool server_v1 = false;
    const std::vector<llama_token> restored = unpack_prefix_tokens(packed, server_v1);
    require(restored.size() == opt.p, "restored prefix length is not exactly P");
    require(std::equal(restored.begin(), restored.end(), input.prompt.begin()),
            "restored prefix token IDs differ from the prompt prefix");
    const llama_pos restored_pos_max = llama_memory_seq_pos_max(memory, 0);
    require(restored_pos_max == static_cast<llama_pos>(opt.p) - 1,
            "restored prefix position does not match P-1");
    llama_synchronize(ctx);

    const uint32_t length = static_cast<uint32_t>(input.prompt.size());
    const std::vector<decode_step> plan = plan_cadence(opt.p, length);
    const cadence_counts counts = count_cadence(plan);
    const uint32_t prefill_tail = length - opt.p;
    require(prefill_tail >= 512 && prefill_tail % 512u == 0, "invalid prompt tail");
    uint64_t planned_tokens = 0;
    for (const auto & step : plan) planned_tokens += step.count;
    require(planned_tokens == prefill_tail, "cadence plan does not cover the prompt tail exactly");
    require(counts.api_calls == plan.size(), "cadence api-call count does not match the plan");
    require(uint64_t(counts.ubatch_512) * kUbatch + uint64_t(counts.seam_256) * 256 +
            uint64_t(counts.seam_128) * 128 == prefill_tail,
            "cadence UB512/seam token totals do not cover the prompt tail");

    decode_state decoder;
    decoder.ctx = ctx;
    decoder.vocab = vocab;
    decoder.history = input.prompt;
    decoder.batch = llama_batch_init(kBatch, 0, 1);
    require(decoder.batch.token != nullptr && decoder.batch.pos != nullptr &&
            decoder.batch.n_seq_id != nullptr && decoder.batch.seq_id != nullptr &&
            decoder.batch.logits != nullptr, "llama_batch_init failed");

    json steps = json::array();
    const json prefill_counts = cadence_json(counts);
    const float * logits = nullptr;
    for (const auto & step : plan) {
        const float * row = decoder.decode(step.start, step.count, step.want_logits);
        if (step.want_logits) logits = row;
    }
    require(logits != nullptr, "prefill did not produce the final logits row");
    require(llama_memory_seq_pos_max(memory, 0) == static_cast<llama_pos>(length) - 1,
            "prefill did not leave the prompt at position L-1");

    const std::string teacher_source = input.teacher.empty() ? "argmax" : "fixed-file";
    sha256_ctx prompt_hasher;
    prompt_hasher.update(input.prompt.data(), input.prompt.size() * sizeof(llama_token));
    json prompt_hashes = json::array();
    uint32_t position = length - 1;
    for (size_t step = 0; step < opt.teacher_steps; ++step) {
        require(logits != nullptr, "missing logits row for capture step");
        const uint32_t argmax = argmax_lowest(logits, vocab);
        const llama_token chosen = select_teacher(input.teacher, step, argmax);
        const std::string step_prompt_sha = sha256_digest(prompt_hasher);
        const json meta = write_step(opt.output, step, static_cast<uint32_t>(vocab), logits, vocab,
                                     position, chosen, step_prompt_sha,
                                     step == 0 ? prefill_counts : cadence_json({1, 0, 0, 0}));
        steps.push_back(meta);
        prompt_hashes.push_back(step_prompt_sha);
        completed_steps = step + 1;
        std::fprintf(stderr, "CAPTURE step=%zu position=%u teacher=%lld argmax=%u finite=1\n",
                     step, position, static_cast<long long>(chosen), argmax);
        std::fflush(stderr);
        if (step + 1 == opt.teacher_steps) break;
        decoder.history.push_back(chosen);
        prompt_hasher.update(&chosen, sizeof(chosen));
        position = length + static_cast<uint32_t>(step);
        logits = decoder.decode(position, 1, true);
    }
    require(decoder.history.size() == size_t(length) + opt.teacher_steps - 1,
            "token history length does not match the HTTP teacher cadence");
    require(llama_memory_seq_pos_max(memory, 0) == static_cast<llama_pos>(length + opt.teacher_steps - 2),
            "final state position does not match the HTTP teacher cadence");
    require(decoder.logits_contract_checked,
            "dense-logits contract was never checked before capture");

    write_bytes_new(opt.output / "tokens.u32.bin", decoder.history.data(),
                    decoder.history.size() * sizeof(llama_token));
    const std::string history_sha = sha256_bytes(decoder.history.data(),
                                                 decoder.history.size() * sizeof(llama_token));
    const std::string prompt_sha = sha256_bytes(input.prompt.data(), input.prompt.size() * sizeof(llama_token));

    llama_perf_context_print(ctx);
    return {
        {"schema", "rkva-quality-raw-capture-v1"},
        {"label", "RAW_CAPTURE_ONLY"},
        {"mode", "execute"},
        {"capture", {
            {"vocab", static_cast<uint64_t>(vocab)},
            {"steps", opt.teacher_steps},
            {"prompt_tokens", length},
            {"prefix_tokens", opt.p},
            {"tail_tokens", length - opt.p},
            {"all_logits", true},
            {"logits_format", "float32-le-index-token-id"},
            {"step_logits_bytes", vocab * sizeof(float)},
            {"finite", true},
            {"source", "capi_llama_decode"},
            {"teacher_source", teacher_source},
            {"logits_contract", {
                {"sampled_logits_active", false},
                {"sampled_logits_count", decoder.sampled_logits_count},
                {"sampled_candidates_count", decoder.sampled_candidates_count},
                {"sampled_candidates_source", decoder.sampled_candidates_source},
                {"dense_row_index", "count-1"},
                {"outputs_requested", "last-row-only"},
            }},
        }},
        {"geometry", geometry},
        {"config", config},
        {"declared", declared_config(params, opt)},
        {"environment", environment_json()},
        {"prefix_restore", {
            {"file", prefix_path},
            {"file_size", prefix_size},
            {"packed_words", packed_count},
            {"n_loaded", restored.size()},
            {"token_format", server_v1 ? "server-text-v1" : "legacy-raw-token-ids"},
            {"token_ids_match", true},
            {"pos_max", restored_pos_max},
            {"sha256_source", "root-wrapper"},
            {"sha256_note", "delegated to the root wrapper; the driver never hashes the multi-GB snapshot"},
        }},
        {"teacher", {
            {"source", teacher_source},
            {"steps", opt.teacher_steps},
            {"ids", input.teacher.empty() ? json::array() : json(input.teacher)},
            {"prompt_hashes", prompt_hashes},
        }},
        {"cadence", {
            {"p", opt.p},
            {"length", length},
            {"plan", plan_json(plan)},
            {"ubatch_512", counts.ubatch_512},
            {"seam_256", counts.seam_256},
            {"seam_128", counts.seam_128},
        }},
        {"token_history", {
            {"file", "tokens.u32.bin"},
            {"count", decoder.history.size()},
            {"sha256", history_sha},
        }},
        {"prompt_sha256", prompt_sha},
        {"steps", steps},
        {"system_info", llama_print_system_info()},
    };
}

int run_preview(const options & opt) {
    const validated input = validate_options(opt);
    const uint32_t length = static_cast<uint32_t>(input.prompt.size());
    const auto cadence = plan_cadence(opt.p, length);
    const auto counts = count_cadence(cadence);
    std::cout << "mode: preview\n"
              << "execute: false\n"
              << "starts_child: false\n"
              << "writes_output: false\n"
              << "label: RAW_CAPTURE_ONLY\n"
              << "context: " << opt.context << " (hardcap " << kMaxContext << ")\n"
              << "p: " << opt.p << "\n"
              << "prompt_tokens: " << length << "\n"
              << "tail_tokens: " << length - opt.p << "\n"
              << "teacher_steps: " << opt.teacher_steps << "\n"
              << "teacher_source: " << (input.teacher.empty() ? "argmax" : "fixed-file") << "\n"
              << "prefix_snapshot: " << opt.prefix_snapshot << " ("
              << file_size_or_fail(opt.prefix_snapshot) << " bytes)\n"
              << "output: " << path_text(opt.output) << " (not created)\n"
              << "common_args: " << input.common_args.size() << "\n"
              << "cadence_api_calls: " << counts.api_calls << "\n"
              << "cadence_ubatch_512: " << counts.ubatch_512 << "\n"
              << "cadence_seam_256: " << counts.seam_256 << "\n"
              << "cadence_seam_128: " << counts.seam_128 << "\n";
    return 0;
}

json envelope_case(const std::vector<llama_token> & packed, bool expect_ok) {
    json result = {{"packed", packed}, {"expect_ok", expect_ok}};
    try {
        bool server_v1 = false;
        const auto ids = unpack_prefix_tokens(packed, server_v1);
        result["ok"] = true;
        result["server_v1"] = server_v1;
        result["ids"] = ids;
        result["error"] = nullptr;
    } catch (const std::exception & e) {
        result["ok"] = false;
        result["server_v1"] = nullptr;
        result["ids"] = nullptr;
        result["error"] = e.what();
    }
    return result;
}

int run_self_test(const options & opt) {
    require_little_endian();
    if (!opt.self_test_json.empty()) require_new_path(opt.self_test_json);
    const fs::path temp = fs::temp_directory_path() /
        ("rkva-capture-selftest-" + std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    ensure_new_directory(temp);
    struct cleanup { fs::path path; ~cleanup() { std::error_code ec; fs::remove_all(path, ec); } } guard{temp};

    json report = {{"schema", "rkva-quality-raw-capture-selftest-v1"}, {"label", "RAW_CAPTURE_ONLY"}};
    json envelopes = json::array();
    const llama_token t0 = 11, t1 = 22, t2 = 33;
    envelopes.push_back(envelope_case({LLAMA_TOKEN_NULL, 1, 3, t0, t1, t2, 0}, true));
    envelopes.push_back(envelope_case({t0, t1, t2}, true));
    envelopes.push_back(envelope_case({}, false));
    envelopes.push_back(envelope_case({LLAMA_TOKEN_NULL, 1, 3, t0}, false));
    envelopes.push_back(envelope_case({LLAMA_TOKEN_NULL, 2, 3, t0, t1, t2, 0}, false));
    envelopes.push_back(envelope_case({LLAMA_TOKEN_NULL, 1, 3, t0, t1, t2, 1}, false));
    envelopes.push_back(envelope_case({LLAMA_TOKEN_NULL, 1, 3, t0, t1, t2, 0, 9}, false));
    envelopes.push_back(envelope_case({LLAMA_TOKEN_NULL, 1, 99, t0, t1, t2, 0}, false));
    for (const auto & item : envelopes) {
        const bool ok = item.at("ok").get<bool>();
        require(ok == item.at("expect_ok").get<bool>(), "envelope fixture result mismatch");
        if (ok) require(item.at("ids").size() > 0, "accepted envelope is empty");
    }
    report["envelope"] = envelopes;

    report["prefix_guard"] = {
        {"p", kBaselineP},
        {"packed_p_ok", prefix_packed_count_ok(kBaselineP, kBaselineP)},
        {"packed_p_plus_4_ok", prefix_packed_count_ok(kBaselineP, size_t(kBaselineP) + 4)},
        {"packed_p_minus_128_ok", prefix_packed_count_ok(kBaselineP, kBaselineP - 128)},
        {"packed_p_plus_1_ok", prefix_packed_count_ok(kBaselineP, size_t(kBaselineP) + 1)},
    };
    require(report["prefix_guard"]["packed_p_ok"].get<bool>(), "P guard rejected P");
    require(report["prefix_guard"]["packed_p_plus_4_ok"].get<bool>(), "P guard rejected P+4");
    require(!report["prefix_guard"]["packed_p_minus_128_ok"].get<bool>(), "P-128 must be refused");
    require(!report["prefix_guard"]["packed_p_plus_1_ok"].get<bool>(), "P+1 must be refused");

    static int dummy_device_a = 0;
    static int dummy_device_b = 0;
    const ggml_backend_dev_t dev_a = reinterpret_cast<ggml_backend_dev_t>(&dummy_device_a);
    const ggml_backend_dev_t dev_b = reinterpret_cast<ggml_backend_dev_t>(&dummy_device_b);
    auto device_case = [](const char * name, const std::vector<ggml_backend_dev_t> & devices) {
        const device_slots slots = inspect_device_slots(devices);
        return json{{"name", name}, {"non_null", slots.non_null}, {"terminated", slots.terminated},
                    {"leading_null", slots.leading_null}, {"holes", slots.holes},
                    {"valid", one_device_with_terminator(devices)}};
    };
    json device_cases = json::array();
    device_cases.push_back(device_case("one_gpu_terminated", {dev_a, nullptr}));
    device_cases.push_back(device_case("none", {nullptr}));
    device_cases.push_back(device_case("two_gpus", {dev_a, dev_b, nullptr}));
    device_cases.push_back(device_case("leading_null", {nullptr, dev_a, nullptr}));
    device_cases.push_back(device_case("hole_before_gpu", {dev_a, nullptr, dev_b}));
    device_cases.push_back(device_case("missing_terminator", {dev_a, dev_b}));
    report["device_slots"] = device_cases;
    require(device_cases[0]["valid"].get<bool>(), "valid {CUDA0, null} device list was rejected");
    for (size_t i = 1; i < device_cases.size(); ++i) {
        require(!device_cases[i]["valid"].get<bool>(), "invalid device list was accepted");
    }

    const common_speculative_output_limits output_limits =
        common_speculative_get_output_limits(kBatch, 1, 0);
    report["output_limits"] = {
        {"total", output_limits.total}, {"per_seq", output_limits.per_seq},
        {"expected_total", 1}, {"expected_per_seq", 1},
        {"source", "server_output_limits"},
    };
    require(output_limits.total == 1 && output_limits.per_seq == 1,
            "server output limits are not 1/1 for one slot without a draft");

    json cadences = json::array();
    const std::array<std::pair<uint32_t, uint32_t>, 7> cadence_cases = {{
        {1024, 5120}, {1024, 17408}, {1024, 29184}, {1024, 58368},
        {kBaselineP, 106496}, {kBaselineP, 130560}, {kBaselineP, 159744},
    }};
    const std::map<uint32_t, size_t> expected_ubatch = {
        {4096, 7}, {16384, 31}, {28160, 54}, {57344, 111},
    };
    for (const auto & item : cadence_cases) {
        const auto plan = plan_cadence(item.first, item.second);
        const auto counts = count_cadence(plan);
        const uint32_t tail = item.second - item.first;
        json entry = {
            {"p", item.first}, {"length", item.second}, {"tail", tail},
            {"within_root_cap", item.second <= kMaxContext},
            {"plan", plan_json(plan)}, {"ubatch_512", counts.ubatch_512},
            {"seam_256", counts.seam_256}, {"seam_128", counts.seam_128},
            {"api_calls", counts.api_calls},
        };
        cadences.push_back(entry);
        const auto expected = expected_ubatch.find(tail);
        require(expected != expected_ubatch.end(), "cadence regression tail is missing");
        require(entry["ubatch_512"].get<size_t>() == expected->second,
                "cadence UB512 regression mismatch for tail " + std::to_string(tail));
        require(uint64_t(counts.ubatch_512) * 512 + uint64_t(counts.seam_256) * 256 +
                uint64_t(counts.seam_128) * 128 == tail,
                "cadence token coverage mismatch for tail " + std::to_string(tail));
        require(entry["api_calls"].get<size_t>() == entry["plan"].size(),
                "cadence api-call count mismatch");
    }
    require(cadences[0]["seam_256"].get<size_t>() == 1 && cadences[0]["seam_128"].get<size_t>() == 2,
            "4k cadence must end in 256/128/128");
    require(cadences.back()["tail"].get<uint32_t>() == 57344 &&
            cadences.back()["within_root_cap"].get<bool>(),
            "57k root-cap cadence regression is missing");
    report["cadence"] = cadences;

    const std::array<llama_token, 3> hash_tokens = {1, 2, 3};
    sha256_ctx hash_ctx;
    hash_ctx.update(hash_tokens.data(), hash_tokens.size() * sizeof(llama_token));
    const std::string hash_base = sha256_digest(hash_ctx);
    const llama_token hash_next = 4;
    hash_ctx.update(&hash_next, sizeof(hash_next));
    report["prompt_hash"] = {{"base", hash_base}, {"after_token", sha256_digest(hash_ctx)},
                             {"next_token", hash_next}};

    report["teacher"] = {
        {"fixed", {3, 4}},
        {"candidate_argmax", {0, 0}},
        {"fixed_choice", {select_teacher({3, 4}, 0, 0), select_teacher({3, 4}, 1, 0)}},
        {"argmax_choice", {select_teacher({}, 0, 0), select_teacher({}, 1, 0)}},
        {"tie_argmax", argmax_lowest(std::array<float, 3>{1.0f, 1.0f, 0.0f}.data(), 3)},
    };
    require(report["teacher"]["fixed_choice"] == json({3, 4}), "fixed teacher was replaced by the candidate argmax");
    require(report["teacher"]["argmax_choice"] == json({0, 0}), "argmax teacher fallback changed");
    require(report["teacher"]["tie_argmax"].get<uint32_t>() == 0, "argmax tie is not the lowest id");

    const fs::path writer_dir = temp / "writer";
    ensure_new_directory(writer_dir);
    const uint32_t vocab = 7;
    const std::array<float, 7> row = {0.5f, -1.0f, 2.5f, 0.0f, 1.0f, -0.25f, 0.75f};
    const std::string expected_sha = sha256_bytes(row.data(), row.size() * sizeof(float));
    const json meta = write_step(writer_dir, 0, vocab, row.data(), row.size(), 3, 2, expected_sha, cadence_json({1, 0, 0, 0}));
    for (size_t step = 1; step < 4; ++step) {
        write_step(writer_dir, step, vocab, row.data(), row.size(), 3 + step, 2, expected_sha, cadence_json({1, 0, 0, 0}));
    }
    const json on_disk = read_json_file(writer_dir / "0000.json", 1024u * 1024u, "step metadata");
    require(on_disk.at("filename") == "0000.logits.f32.bin" && on_disk.at("teacher_id") == 2,
            "step metadata on disk does not match the writer row");
    const uint64_t bytes = file_size_or_fail(writer_dir / "0000.logits.f32.bin");
    require(bytes == row.size() * sizeof(float), "writer byte count is not 4*vocab");
    require(meta.at("sha256") == expected_sha, "writer sha does not match the raw row");
    require(meta.at("filename") == "0000.logits.f32.bin", "writer filename mismatch");
    require(meta.at("teacher_id") == 2, "writer teacher metadata mismatch");
    bool overwrite_refused = false;
    try { write_step(writer_dir, 0, vocab, row.data(), row.size(), 3, 2, expected_sha, cadence_json({1, 0, 0, 0})); }
    catch (const app_error &) { overwrite_refused = true; }
    require(overwrite_refused, "writer allowed an overwrite");
    bool nonfinite_rejected = false;
    auto bad = row;
    bad[2] = std::numeric_limits<float>::quiet_NaN();
    try { write_step(writer_dir, 1, vocab, bad.data(), bad.size(), 4, 2, expected_sha, cadence_json({1, 0, 0, 0})); }
    catch (const app_error &) { nonfinite_rejected = true; }
    require(nonfinite_rejected, "writer accepted a non-finite logit");
    bool short_rejected = false;
    try { write_step(writer_dir, 2, vocab, row.data(), row.size() - 1, 5, 2, expected_sha, cadence_json({1, 0, 0, 0})); }
    catch (const app_error &) { short_rejected = true; }
    require(short_rejected, "writer accepted a short logits row");
    json files = json::array();
    for (size_t step = 0; step < 4; ++step) {
        const fs::path file = writer_dir / (step_stem(step) + ".logits.f32.bin");
        files.push_back({{"step", step}, {"filename", file.filename().string()}, {"exists", fs::exists(file)}});
    }
    report["writer"] = {
        {"bytes", bytes}, {"vocab", static_cast<uint64_t>(vocab)}, {"sha256", expected_sha},
        {"overwrite_refused", overwrite_refused}, {"nonfinite_rejected", nonfinite_rejected},
        {"short_rejected", short_rejected}, {"files", files},
    };
    require(std::set<std::string>({files[0]["filename"].get<std::string>(),
            files[1]["filename"].get<std::string>(), files[2]["filename"].get<std::string>(),
            files[3]["filename"].get<std::string>()}).size() == 4, "step filenames are not unique");

    report["scope"] = {
        {"backend_initialized", false}, {"model_loaded", false},
        {"network_started", false}, {"writes_output", false},
    };
    if (!opt.self_test_json.empty()) write_json_new(opt.self_test_json, report);
    std::cout << "SELF-TEST PASS\n";
    return 0;
}

int run_execute(const options & opt) {
    const validated input = validate_options(opt);
    ensure_new_directory(opt.output);
    bool backend_started = false;
    std::string phase = "environment";
    size_t completed_steps = 0;
    try {
        validate_environment();
        phase = "backend-init";
        llama_backend_init();
        backend_started = true;
        phase = "common-args";
        common_params params = parse_common_args(input.common_args);
        validate_params(params, opt);
        phase = "model-and-capture";
        json manifest = run_capture(opt, input, params, completed_steps);
        phase = "finalize";
        if (backend_started) {
            llama_backend_free();
            backend_started = false;
        }
        manifest["result"] = "PASS";
        manifest["finalized"] = {
            {"context_destroyed", true},
            {"profile_flushed", true},
            {"backend_freed", true},
            {"manifest_after_context_destroy", true},
        };
        write_json_new(opt.output / "manifest.json", manifest);
        std::fprintf(stderr, "CAPTURE PASS steps=%zu vocab=%llu output=%s\n",
                     completed_steps, static_cast<unsigned long long>(manifest.at("capture").at("vocab").get<uint64_t>()),
                     path_text(opt.output).c_str());
        std::fflush(stderr);
        return 0;
    } catch (const std::exception & e) {
        if (backend_started) {
            llama_backend_free();
            backend_started = false;
        }
        const json failed = {
            {"schema", "rkva-quality-raw-capture-v1"},
            {"label", "RAW_CAPTURE_ONLY"},
            {"result", "FAILED"},
            {"phase", phase},
            {"error", e.what()},
            {"steps_completed", completed_steps},
            {"output", path_text(opt.output)},
        };
        try { write_json_new(opt.output / "failed.json", failed); } catch (...) {}
        std::fprintf(stderr, "CAPTURE FAILED phase=%s error=%s\n", phase.c_str(), e.what());
        std::fflush(stderr);
        return 3;
    }
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const options opt = parse_cli(argc, argv);
        if (opt.help) {
            print_usage();
            return 0;
        }
        if (opt.self_test) return run_self_test(opt);
        if (opt.execute) return run_execute(opt);
        return run_preview(opt);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
