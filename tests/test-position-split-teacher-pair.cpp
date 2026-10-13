// Explicit real-model structural campaign. One persistent GPU trajectory;
// every CPU/GPU teacher step starts from the same native KV+recurrent prefix.
// Build only: deliberately not registered with default CTest.
#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "llama.h"
#include "llama-ext.h"
#include "../src/llama-model.h"

#include <nlohmann/json.hpp>
#define XXH_INLINE_ALL
#include "../vendor/hash/xxhash/xxhash.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;
namespace fs = std::filesystem;

namespace {

void require(bool ok, const std::string & message) {
    if (!ok) { throw std::runtime_error(message); }
}

std::string hex64(uint64_t value) {
    char text[17];
    std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(value));
    return text;
}

bool sha256_text(const std::string & value) {
    return value.size() == 64 && value.find_first_not_of("0123456789abcdefABCDEF") == std::string::npos;
}

struct options {
    std::string tokens_file;
    std::string token_format = "json";
    std::string output;
    std::string model_sha256;
    std::string build_id;
    std::string prefix_state_in;
    bool prefix_state_out = false;
    bool diagnostic_p_above_target = false;
    std::string checkpoint_method = "full";
    std::string reference = "cpu";
    std::vector<int> starts;
    int count = 32;
    bool verify_full_prefix = false;
    bool pure_cuda_baseline = false;
    bool layered_baseline = false;
    bool fixed_current_kv = false;
    bool quality_only = false;
};

struct kv_callback_state {
    enum class mode { off, capture, replay } current = mode::off;
    std::string error;
    std::set<int> expected_layers;
    std::map<int, std::pair<int64_t, int64_t>> expected_geometry;
    std::map<std::string, std::vector<uint8_t>> captured;
    std::map<std::string, std::array<int64_t, GGML_MAX_DIMS>> shapes;
    std::set<std::string> replayed;
    size_t injected = 0;
};

std::string canonical_tensor_name(const ggml_tensor * tensor) {
    return tensor->name;
}

bool split_kv_tensor_name(const ggml_tensor * tensor, std::string & name, char & role, int & layer) {
    name = canonical_tensor_name(tensor);
    if (name.rfind("Kcur-", 0) == 0) { role = 'K'; }
    else if (name.rfind("Vcur-", 0) == 0) { role = 'V'; }
    else { return false; }
    const std::string layer_text = name.substr(5);
    // Scheduler copies (backend#name#copy) and reference-host clones are not
    // the producer tensors that precede the KV store. Select only canonical
    // Kcur-il / Vcur-il graph names.
    if (layer_text.empty() || layer_text.find_first_not_of("0123456789") != std::string::npos) { return false; }
    try { layer = std::stoi(layer_text); }
    catch (...) { return false; }
    return true;
}

bool paired_kv_eval_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    auto * state = static_cast<kv_callback_state *>(user_data);
    if (state == nullptr || state->current == kv_callback_state::mode::off) { return false; }
    try {
    std::string name;
    char role = 0;
    int layer = -1;
    if (!split_kv_tensor_name(tensor, name, role, layer)) { return false; }
    if (ask) {
        if (layer < 0 || !state->expected_layers.count(layer)) {
            state->error = "unexpected current-KV callback tensor " + name;
        }
        return true;
    }
    if (!state->error.empty()) { return true; }
    if (layer < 0 || !state->expected_layers.count(layer)) {
        state->error = "unexpected current-KV callback tensor " + name;
        return true;
    }
    if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor) || tensor->ne[2] != 1 || tensor->ne[3] != 1) {
        state->error = "current-KV tensor must be contiguous F32 with nq=1: " + name;
        return true;
    }
    const auto expected_shape = state->expected_geometry.find(layer);
    if (expected_shape == state->expected_geometry.end() || tensor->ne[0] != expected_shape->second.first ||
            tensor->ne[1] != expected_shape->second.second) {
        state->error = "unexpected current-KV geometry for " + name;
        return true;
    }
    const size_t bytes = ggml_nbytes(tensor);
    std::array<int64_t, GGML_MAX_DIMS> shape{};
    for (int i = 0; i < GGML_MAX_DIMS; ++i) { shape[size_t(i)] = tensor->ne[i]; }
    if (state->current == kv_callback_state::mode::capture) {
        if (state->captured.count(name)) {
            state->error = "duplicate current-KV capture " + name;
            return true;
        }
        auto & data = state->captured[name];
        data.resize(bytes);
        ggml_backend_tensor_get(tensor, data.data(), 0, bytes);
        state->shapes[name] = shape;
        return true;
    }
    const auto found = state->captured.find(name);
    const auto saved_shape = state->shapes.find(name);
    if (found == state->captured.end() || saved_shape == state->shapes.end() ||
            found->second.size() != bytes || saved_shape->second != shape) {
        state->error = "missing or geometry-mismatched frozen current-KV tensor " + name;
        return true;
    }
    ggml_backend_tensor_set(tensor, found->second.data(), 0, bytes);
    std::vector<uint8_t> check(bytes);
    ggml_backend_tensor_get(tensor, check.data(), 0, bytes);
    if (check != found->second) {
        state->error = "frozen current-KV readback differs bytewise for " + name;
        return true;
    }
    if (!state->replayed.insert(name).second) {
        state->error = "duplicate frozen current-KV injection " + name;
        return true;
    }
    ++state->injected;
    return true;
    } catch (const std::exception & error) {
        state->error = std::string("current-KV callback exception: ")+error.what();
    } catch (...) {
        state->error = "unknown current-KV callback exception";
    }
    return true;
}

void usage() {
    std::fprintf(stderr,
        "Paired native-state structural reference, plus ordinary server CLI model/cache/backend flags.\n"
        "  --paired-token-file FILE          JSON token array / {tokens:[...]} / {prompt:[...]}\n"
        "  --paired-token-format FORMAT      json (default) or little-endian int32\n"
        "  --paired-start N                  first captured prefix length (default 513); repeat for more ranges\n"
        "  --paired-count N                  consecutive teacher steps per range (default 32)\n"
        "  --paired-output-dir DIR           NEW empty output directory\n"
        "  --paired-model-sha256 HASH         actual model SHA256 verified by operator before launch\n"
        "  --paired-build-id HASH             build bundle SHA256 verified by operator before launch\n"
        "  --paired-checkpoint METHOD         full (default): native file restore; partial: diagnostic rollback\n"
        "  --paired-reference ARM             cpu (default) or gpu; gpu is diagnostic, never structural acceptance\n"
        "  --paired-pure-cuda-baseline       require P=0 and remote attention disabled; diagnostic only\n"
        "  --paired-layered-baseline         require P=0 and explicit remote layers; calibrated-baseline diagnostic only\n"
        "  --paired-fixed-current-kv         capture GPU K/V and replay exact bytes into the CPU reference; required for structural protocol eligibility\n"
        "  --paired-quality-only             capture consecutive GPU teacher-forced logits only; no CPU replay or structural/calibration eligibility\n"
        "  --paired-prefix-state-in FILE     diagnostic only: import validated prefix for quality-only or verified full CPU fixed-KV replay\n"
        "  --paired-prefix-state-out         diagnostic: save native prefix before first captured token in output dir\n"
        "  --paired-diagnostic-p105216       allow P=105216 for quality-only ablation, never protocol acceptance\n"
        "  --paired-verify-full-prefix        required for acceptance: exact native full-state file comparison\n"
        "At captured N both arms decode teacher[N-1] from identical GPU prefix N-1.\n"
        "FULL restores the native prefix directly from its bounded file/spool stream.\n"
        "Diagnostic PARTIAL restores in place, then truncates with seq_rm(prefix_count,-1).\n"
        "Reference-step state is discarded; the GPU candidate result alone becomes the next prefix.\n"
        "Native full-state scratch files are reused after their hashes/equality are recorded.\n"
        "Capture directories range-N/<reference-arm> and range-N/gpu work with occupancy-f5/compare_logits.py.\n"
        "PARTIAL is diagnostic only. FULL plus full-prefix verification and >=32 points are required for acceptance.\n");
}

std::vector<llama_token> read_tokens(const options & opt) {
    std::ifstream in(opt.tokens_file, std::ios::binary);
    require(bool(in), "cannot open teacher token file");
    std::vector<llama_token> tokens;
    if (opt.token_format == "json") {
        auto data = json::parse(in);
        if (data.is_object()) { data = data.at(data.contains("tokens") ? "tokens" : "prompt"); }
        require(data.is_array(), "teacher token JSON must contain an array");
        for (const auto & item : data) {
            require(item.is_number_integer(), "teacher tokens must be integer IDs");
            const int64_t value = item.get<int64_t>();
            require(value >= 0 && value <= INT32_MAX, "teacher token outside int32 range");
            tokens.push_back(llama_token(value));
        }
    } else {
        require(opt.token_format == "int32", "unknown teacher token format");
        std::array<unsigned char, 4> bytes;
        while (in.read(reinterpret_cast<char *>(bytes.data()), bytes.size())) {
            const uint32_t value = uint32_t(bytes[0]) | uint32_t(bytes[1]) << 8 |
                                   uint32_t(bytes[2]) << 16 | uint32_t(bytes[3]) << 24;
            require(value <= INT32_MAX, "negative teacher token");
            tokens.push_back(llama_token(value));
        }
        require(in.gcount() == 0, "truncated int32 teacher token file");
    }
    return tokens;
}

void write_json(const fs::path & path, const json & value) {
    const auto temporary = path.string() + ".tmp";
    std::ofstream out(temporary);
    require(bool(out), "cannot open JSON output " + path.string());
    out << value.dump(2) << '\n'; out.close();
    require(bool(out), "JSON write failed " + path.string());
    fs::rename(temporary, path);
}

void write_binary(const fs::path & path, const void * data, size_t bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    require(bool(out), "cannot open binary output " + path.string());
    out.write(static_cast<const char *>(data), std::streamsize(bytes)); out.close();
    require(bool(out), "binary write failed " + path.string());
}

json file_observation(const fs::path & path) {
    std::ifstream in(path, std::ios::binary);
    require(bool(in), "cannot read file " + path.string());
    XXH64_state_t state;
    XXH64_reset(&state, 0);
    std::array<char, 1024*1024> buffer;
    uint64_t bytes = 0;
    while (in.read(buffer.data(), buffer.size()) || in.gcount()) {
        const size_t count = size_t(in.gcount());
        XXH64_update(&state, buffer.data(), count); bytes += count;
    }
    require(in.eof(), "file read failed " + path.string());
    return {{"path", path.string()}, {"bytes", bytes}, {"xxh64", hex64(XXH64_digest(&state))}};
}

// Exact bounded-memory comparison, not merely matching checksums. Digests are
// recorded for replay provenance, while every byte decides prefix equality.
json compare_prefix_files(const fs::path & before, const fs::path & after) {
    std::ifstream a(before, std::ios::binary), b(after, std::ios::binary);
    require(bool(a) && bool(b), "cannot open full-prefix verification files");
    XXH64_state_t ha, hb;
    XXH64_reset(&ha, 0); XXH64_reset(&hb, 0);
    std::array<char, 1024*1024> x, y;
    uint64_t offset = 0;
    bool same = true;
    json first_difference = nullptr;
    for (;;) {
        a.read(x.data(), x.size()); b.read(y.data(), y.size());
        const auto nx = size_t(a.gcount()), ny = size_t(b.gcount());
        XXH64_update(&ha, x.data(), nx); XXH64_update(&hb, y.data(), ny);
        if (nx != ny || std::memcmp(x.data(), y.data(), std::min(nx, ny))) {
            if (same) {
                size_t i = 0;
                while (i < std::min(nx, ny) && x[i] == y[i]) { ++i; }
                first_difference = offset+i;
            }
            same = false;
        }
        offset += std::max(nx, ny);
        if (nx == 0 && ny == 0) {
            require(a.eof() && b.eof(), "full-prefix comparison I/O failed");
            break;
        }
    }
    return {{"equal", same}, {"bytes_compared", offset}, {"first_difference", first_difference},
            {"before_xxh64", hex64(XXH64_digest(&ha))}, {"after_xxh64", hex64(XXH64_digest(&hb))},
            {"method", "exact streaming byte comparison of native full KV/table/recurrent sequence files; no metadata normalization"}};
}

json logits_metrics(const std::vector<float> & reference, const std::vector<float> & candidate,
                    llama_token next_teacher) {
    require(reference.size() == candidate.size() && !reference.empty(), "full-vocabulary size mismatch");
    double mr = -INFINITY, mc = -INFINITY;
    for (size_t i = 0; i < reference.size(); ++i) {
        require(std::isfinite(reference[i]) && std::isfinite(candidate[i]), "nonfinite paired logits");
        mr = std::max(mr, double(reference[i])); mc = std::max(mc, double(candidate[i]));
    }
    double zr = 0, zc = 0;
    for (size_t i = 0; i < reference.size(); ++i) {
        zr += std::exp(double(reference[i])-mr); zc += std::exp(double(candidate[i])-mc);
    }
    const double lr = mr+std::log(zr), lc = mc+std::log(zc);
    double kl = 0, max_abs = 0, max_logp = 0;
    for (size_t i = 0; i < reference.size(); ++i) {
        const double ref_logp = double(reference[i])-lr, cand_logp = double(candidate[i])-lc;
        kl += std::exp(ref_logp)*(ref_logp-cand_logp);
        max_abs = std::max(max_abs, std::abs(double(candidate[i])-reference[i]));
        max_logp = std::max(max_logp, std::abs(cand_logp-ref_logp));
    }
    require(std::isfinite(kl), "nonfinite paired FP64 KL");
    return {{"kl_nat", kl}, {"max_abs_logits", max_abs}, {"max_logp_difference", max_logp},
            {"reference_nll", lr-reference.at(size_t(next_teacher))},
            {"candidate_nll", lc-candidate.at(size_t(next_teacher))}};
}

struct runner {
    options opt;
    common_params params;
    common_init_result_ptr init;
    llama_context * ctx = nullptr;
    llama_batch batch{};
    std::vector<llama_token> tokens;
    int past = 0, vocab = 0;
    json report;
    std::string config_core;
    kv_callback_state kv_callback;

    ~runner() { if (batch.token) { llama_batch_free(batch); } }
    fs::path path(const std::string & name) const { return fs::path(opt.output)/name; }
    void publish() { write_json(path("paired-result.json"), report); }
    std::string arm_classification() const {
        if (opt.quality_only) { return "quality_only_gpu_trajectory"; }
        if (opt.pure_cuda_baseline) { return "pure_cuda_xxl_baseline"; }
        if (opt.layered_baseline) { return "layered_xxxl_baseline"; }
        if (params.kv_position_split_p > 0) { return "position_split_candidate"; }
        return "natural_diagnostic";
    }

    void reference_cpu(bool enabled) {
        common_set_env("LLAMA_OVF_ATTN_REFERENCE_CPU", enabled ? "1" : "");
        common_set_env("LLAMA_OVF_ATTN_REFERENCE_CPU_MIN_POS", "");
    }

    void paired_decode(int begin, int count, const std::string & dump_tag, bool cpu_reference) {
        reference_cpu(cpu_reference);
        llama_f4_set_dump_tag(dump_tag.c_str());
        try {
            decode(begin, count);
        } catch (...) {
            llama_f4_set_dump_tag("");
            reference_cpu(false);
            throw;
        }
        llama_f4_set_dump_tag("");
        reference_cpu(false);
    }

    void decode(int begin, int count) {
        common_batch_clear(batch);
        for (int i = 0; i < count; ++i) { common_batch_add(batch, tokens.at(size_t(begin+i)), begin+i, {0}, i == count-1); }
        const int rc = llama_decode(ctx, batch);
        llama_synchronize(ctx);
        require(rc == 0, "teacher decode failed at position " + std::to_string(begin) + ", rc=" + std::to_string(rc));
        require(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == begin+count-1,
                "teacher decode published an unexpected logical position");
    }

    void gpu_prefix(int target) {
        reference_cpu(false);
        llama_f4_set_dump_tag("");
        if (past == 0 && !opt.prefix_state_in.empty()) {
            const fs::path source(opt.prefix_state_in);
            require(fs::is_regular_file(source), "prefix cache is not a regular file");
            std::vector<llama_token> restored(tokens.size(), 0);
            size_t count = 0;
            const size_t bytes = llama_state_seq_load_file(ctx, source.c_str(), 0,
                    restored.data(), restored.size(), &count);
            require(bytes > 0 && bytes == fs::file_size(source), "prefix cache restore refused or incomplete");
            require(count > 0 && count <= size_t(target) &&
                    std::equal(restored.begin(), restored.begin()+count, tokens.begin()),
                    "prefix cache token IDs/count differ from bound teacher corpus");
            require(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == int(count)-1,
                    "prefix cache restored wrong logical position");
            past = int(count);
            std::fprintf(stderr, "paired: imported %d prefix tokens from %s; replay %d tokens\n",
                    past, source.c_str(), target-past);
        }
        while (past < target) {
            // Imported diagnostic suffixes must follow the captured teacher trajectory, one token per decode.
            int end = std::min(target, past+(opt.prefix_state_in.empty() ? params.n_batch : 1));
            if (past < int(params.kv_position_split_p)) { end = std::min(end, int(params.kv_position_split_p)); }
            decode(past, end-past); past = end;
            if (past%4096 == 0 || past == target) { std::fprintf(stderr, "paired: GPU prefix %d/%d\n", past, target); }
        }
    }

    void full_prefix_file(const fs::path & filename, int token_count = -1) {
        if (token_count < 0) { token_count = past; }
        llama_synchronize(ctx);
        require(llama_state_seq_save_file(ctx, filename.c_str(), 0, tokens.data(), size_t(token_count)) > 0,
                "native full-prefix file save failed");
    }

    std::vector<uint8_t> partial_checkpoint() {
        llama_synchronize(ctx);
        constexpr auto flags = llama_state_seq_flags(LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        require(llama_memory_state_seq_can_save_ext(llama_get_memory(ctx), 0, flags), "native partial checkpoint is unsupported");
        const auto bytes = llama_state_seq_get_size_ext(ctx, 0, flags);
        require(bytes > 0, "empty native partial checkpoint");
        std::vector<uint8_t> checkpoint(bytes);
        require(llama_state_seq_get_data_ext(ctx, checkpoint.data(), checkpoint.size(), 0, flags) == checkpoint.size(),
                "native partial checkpoint export failed");
        return checkpoint;
    }

    void restore_partial(const std::vector<uint8_t> & checkpoint) {
        // Restore first: partial snapshots reference this very context's
        // resident records, owner epoch and prefix rows. PARTIAL_ONLY restores
        // rollback payloads; it does not truncate newly appended attention
        // cells. Match the server checkpoint transaction's caller contract by
        // removing that suffix AFTER restoring the recurrent/stage payloads.
        // The suffix is exactly one token and stays within the rollback window;
        // do not clear the context or invalidate the partial snapshot's owner.
        constexpr auto flags = llama_state_seq_flags(LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        require(llama_memory_state_seq_can_restore_ext(llama_get_memory(ctx), 0, flags), "native partial restore is unsupported");
        llama_synchronize(ctx);
        require(llama_state_seq_set_data_ext(ctx, checkpoint.data(), checkpoint.size(), 0, flags) == checkpoint.size(),
                "same-owner native partial restore refused after one-token CPU append");
        llama_synchronize(ctx);
        require(llama_memory_seq_rm(llama_get_memory(ctx), 0, past, -1),
                "post-checkpoint one-token attention suffix truncation refused");
        llama_synchronize(ctx);
        require(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == past-1, "native partial restore did not restore prefix position");
    }

    void restore_full_file(const fs::path & filename, int token_count = -1) {
        if (token_count < 0) { token_count = past; }
        // The canonical sequence reader consumes the native file and mixed
        // bounded spool. No complete KV payload is copied into a RAM vector.
        std::vector<llama_token> restored_tokens(size_t(token_count), 0);
        size_t count = 0;
        llama_synchronize(ctx);
        const size_t bytes = llama_state_seq_load_file(ctx, filename.c_str(), 0,
                restored_tokens.data(), restored_tokens.size(), &count);
        require(bytes > 0 && bytes == fs::file_size(filename), "same-context native FULL file restore refused or consumed wrong extent");
        llama_synchronize(ctx);
        require(count == size_t(token_count) && std::equal(restored_tokens.begin(), restored_tokens.end(), tokens.begin()),
                "native FULL file restore changed prefix token IDs/count");
        require(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == token_count-1,
                "native FULL file restore did not restore prefix position");
    }

    void check_kv_callback(const std::string & label, bool replay) {
        require(kv_callback.error.empty(), label + ": " + kv_callback.error);
        const size_t expected = kv_callback.expected_layers.size()*2;
        require(kv_callback.captured.size() == expected,
                label + ": expected current K/V for every full-attention layer; captured " +
                std::to_string(kv_callback.captured.size()) + "/" + std::to_string(expected));
        if (replay) {
            require(kv_callback.injected == expected && kv_callback.replayed.size() == expected,
                    label + ": frozen current-KV injection count mismatch");
        }
    }

    json persist_current_kv(const fs::path & directory, int n) {
        const auto out = directory/"current-kv";
        fs::create_directories(out);
        json entries = json::array();
        for (const auto & item : kv_callback.captured) {
            const auto shape_it = kv_callback.shapes.find(item.first);
            require(shape_it != kv_callback.shapes.end(), "missing captured current-KV geometry");
            const auto file = out/(item.first+".f32");
            write_binary(file, item.second.data(), item.second.size());
            entries.push_back({{"name", item.first}, {"path", file.string()}, {"bytes", item.second.size()},
                {"xxh64", hex64(XXH64(item.second.data(), item.second.size(), 0))},
                {"shape", shape_it->second}, {"type", "F32"}, {"contiguous", true}, {"nq", 1}});
        }
        require(entries.size() == kv_callback.expected_layers.size()*2,
                "persisted current-KV set is incomplete");
        write_json(out/"metadata.json", { {"schema", "paired-current-kv-v1"}, {"position", n-1},
            {"source_arm", "gpu"}, {"required_count", kv_callback.expected_layers.size()*2},
            {"captured_count", entries.size()}, {"items", entries},
            {"method", "scheduler eval callback capture after producer and before cache store"} });
        return entries;
    }

    std::vector<float> logits() {
        const auto values = llama_get_logits_ith(ctx, -1);
        require(values != nullptr, "missing full-vocabulary teacher logits");
        std::vector<float> result(values, values+vocab);
        for (auto value : result) { require(std::isfinite(value), "nonfinite full-vocabulary teacher logits"); }
        return result;
    }

    void capture(const fs::path & directory, const std::string & arm, int n,
                 const std::vector<float> & values, const json & prefix_proof,
                 const std::string & checkpoint_hash) {
        const auto stem = directory/("cap."+arm+".n"+std::to_string(n));
        write_binary(stem.string()+".logits", values.data(), values.size()*sizeof(float));
        write_binary(stem.string()+".tokens", tokens.data(), size_t(n)*sizeof(llama_token));
        const json meta = {
            {"schema", "occupancy-native-paired-teacher-v1"}, {"run_id", fs::path(opt.output).filename().string()},
            {"phase", arm}, {"build_id", opt.build_id}, {"compiled_commit", llama_commit()},
            {"model_fingerprint", "sha256:"+opt.model_sha256}, {"model_sha256", opt.model_sha256},
            {"provenance_verification", "model and build hashes supplied by operator after independent verification"},
            {"config_core", config_core}, {"n_ctx", llama_n_ctx(ctx)}, {"split_p", params.kv_position_split_p},
            {"n_vocab", vocab}, {"n_params", llama_model_n_params(init->model())},
            {"tokens_n", n}, {"input_pos", n-1}, {"output_pos", n-1}, {"pos_max", n-1},
            {"logits_count", values.size()}, {"finite", true},
            {"tokens_seq_hash", hex64(XXH64(tokens.data(), size_t(n)*sizeof(llama_token), 0))},
            {"prefix_tokens", n-1}, {"checkpoint_method", opt.quality_only ? "none_quality_trajectory" : opt.checkpoint_method},
            {"checkpoint_xxh64", checkpoint_hash}, {"prefix_proof", prefix_proof},
            {"sampling", opt.quality_only
                ? "GPU-only teacher-forced trajectory; each point decodes exactly token[N-1] from the preceding GPU state"
                : "teacher-forced; selected reference arm and GPU candidate both decode exactly token[N-1], then score ground truth token[N]"},
            {"reference_arm", opt.quality_only ? "gpu_quality_only" : opt.reference},
            {"quality_only", opt.quality_only},
            {"diagnostic_prefix_import", !opt.prefix_state_in.empty()},
            {"quality_trajectory_reference", opt.quality_only ? "GPU-only teacher-forced trajectory; no CPU reference arm" : "paired reference protocol"},
            {"reference_option_ignored_in_quality_only", opt.quality_only ? json(opt.reference) : json(nullptr)},
            {"pure_cuda_baseline", opt.pure_cuda_baseline},
            {"layered_baseline", opt.layered_baseline},
            {"arm_classification", arm_classification()},
            {"fixed_current_kv", opt.fixed_current_kv},
            {"capture_role", opt.quality_only ? "gpu_quality_only" : (arm == "gpu" ? "candidate" : "reference")},
            {"checkpoint_restore_contract", opt.quality_only
                ? "none: quality-only capture follows one persistent GPU trajectory without native save/restore"
                : (opt.checkpoint_method == "full"
                ? "restore native FULL prefix file directly in same context through bounded file/spool API; exact native full-state bytes must match"
                : "diagnostic: restore PARTIAL_ONLY payloads first, then seq_rm(prefix_count,-1); exact native full-state bytes must match")},
            {"state_trajectory", "only GPU teacher-step state advances the persistent prefix"},
            {"cache_source", opt.quality_only ? "live_gpu_trajectory" : "same-context native "+opt.checkpoint_method+" checkpoint"}};
        write_json(stem.string()+".json", meta);
    }

    void range(int start) {
        const auto root = path("range-"+std::to_string(start));
        const std::string reference_phase = opt.quality_only ? "gpu_quality_only" :
            (opt.reference == "cpu" ? "cpu" : "gpu-reference");
        const fs::path reference_dir = opt.quality_only ? root/"gpu" : root/reference_phase;
        fs::create_directories(reference_dir);
        if (!opt.quality_only) { fs::create_directories(root/"gpu"); }
        json range_report = {{"start", start}, {"count", opt.count}, {"complete", false},
                             {"reference_arm", opt.quality_only ? "gpu_quality_only" : opt.reference},
                             {"quality_only", opt.quality_only},
                             {"diagnostic_prefix_import", !opt.prefix_state_in.empty()},
                             {"protocol_acceptance_eligible", false},
                             {"calibration_baseline_eligible", false},
                             {"calibration_candidate_eligible", false},
                             {"pure_cuda_baseline", opt.pure_cuda_baseline},
                             {"layered_baseline", opt.layered_baseline}, {"rows", json::array()}};
        report["ranges"].push_back(range_report);
        auto & current = report["ranges"].back();
        for (int n = start; n < start+opt.count; ++n) {
            gpu_prefix(n-1);
            if (opt.prefix_state_out && n == opt.starts.front()) {
                const fs::path cache = path("prefix-cache.bin");
                require(!fs::exists(cache), "prefix cache output already exists");
                full_prefix_file(cache);
                report["prefix_cache"] = {{"path", cache.string()}, {"tokens", past},
                    {"source", "native pure/split state before first captured teacher token"},
                    {"file", file_observation(cache)}};
                publish();
            }
            if (opt.quality_only) {
                json row = {{"n", n}, {"state", "running"}, {"checkpoint_method", "none_quality_trajectory"},
                            {"prefix_occupancy", past}, {"reference_arm", "gpu_quality_only"}};
                current["rows"].push_back(row); publish();
                const auto began = std::chrono::steady_clock::now();
                decode(past, 1);
                const double gpu_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
                const auto gpu_logits = logits();
                const json proof = {{"verified", false},
                    {"method", "quality-only: no native checkpoint or CPU structural comparison"}};
                capture(root/"gpu", "gpu", n, gpu_logits, proof, "");
                auto & completed = current["rows"].back();
                completed["state"] = "complete";
                completed["gpu_decode_seconds"] = gpu_seconds;
                completed["gpu_logits_count"] = gpu_logits.size();
                completed["quality_capture"] = "range-N/gpu/cap.gpu.nN logits and exact teacher prefix";
                ++past;
                publish();
                std::fprintf(stderr, "paired: N=%d quality_only=gpu prefix_occupancy=%d GPU=%.3fs\n",
                             n, n-1, gpu_seconds);
                continue;
            }
            const auto before = path("native-prefix-before.bin"), after = path("native-prefix-after.bin");
            std::vector<uint8_t> checkpoint;
            std::string checkpoint_hash;
            uint64_t checkpoint_bytes = 0;
            if (opt.checkpoint_method == "full" || opt.verify_full_prefix) { full_prefix_file(before); }
            if (opt.checkpoint_method == "partial") {
                checkpoint = partial_checkpoint();
                checkpoint_hash = hex64(XXH64(checkpoint.data(), checkpoint.size(), 0));
                checkpoint_bytes = checkpoint.size();
            } else {
                const auto saved = file_observation(before);
                checkpoint_hash = saved["xxh64"].get<std::string>();
                checkpoint_bytes = saved["bytes"].get<uint64_t>();
            }
            json row = {{"n", n}, {"state", "running"}, {"checkpoint_method", opt.checkpoint_method},
                        {"checkpoint_bytes", checkpoint_bytes}, {"checkpoint_xxh64", checkpoint_hash}, {"prefix_occupancy", past}};
            current["rows"].push_back(row); publish();
            double reference_seconds = 0;
            double gpu_seconds = 0;
            std::vector<float> reference_logits;
            std::vector<float> gpu_logits;
            json proof = {{"verified", false}, {"method", "full native byte comparison not requested; partial restoration alone is not acceptance evidence"}};
            json kv_proof = nullptr;
            if (opt.fixed_current_kv) {
                require(opt.reference == "cpu", "--paired-fixed-current-kv requires --paired-reference cpu");
                kv_callback.current = kv_callback_state::mode::capture;
                kv_callback.error.clear(); kv_callback.captured.clear(); kv_callback.shapes.clear(); kv_callback.replayed.clear(); kv_callback.injected = 0;
                auto began = std::chrono::steady_clock::now();
                paired_decode(past, 1, "gpu-n"+std::to_string(n), false);
                gpu_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
                kv_callback.current = kv_callback_state::mode::off;
                check_kv_callback("GPU current-KV capture at N="+std::to_string(n), false);
                gpu_logits = logits();
                const json manifest = persist_current_kv(reference_dir, n);
                current["rows"].back()["current_kv_manifest"] = manifest;
                const auto post_gpu = path("native-prefix-post-gpu.bin");
                full_prefix_file(post_gpu, past+1);

                // Restore the exact GPU pre-step prefix before replaying the CPU
                // arm. The native pre/post files retain unnormalized bytes.
                restore_full_file(before);
                if (opt.verify_full_prefix) {
                    const auto restored_pre = path("native-prefix-pre-restored.bin");
                    full_prefix_file(restored_pre);
                    proof = compare_prefix_files(before, restored_pre);
                    proof["verified"] = proof["equal"];
                    require(proof["equal"].get<bool>(), "GPU pre-step native prefix changed during restore at N="+std::to_string(n));
                }
                kv_callback.current = kv_callback_state::mode::replay;
                kv_callback.error.clear(); kv_callback.replayed.clear(); kv_callback.injected = 0;
                began = std::chrono::steady_clock::now();
                paired_decode(past, 1, "reference-n"+std::to_string(n), true);
                reference_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
                kv_callback.current = kv_callback_state::mode::off;
                check_kv_callback("CPU frozen current-KV replay at N="+std::to_string(n), true);
                reference_logits = logits();
                kv_proof = {{"complete", true}, {"required_count", kv_callback.expected_layers.size()*2},
                    {"captured_count", kv_callback.captured.size()}, {"injected_count", kv_callback.injected},
                    {"all_byte_equal", true}, {"tensor_manifest", manifest},
                    {"protocol", "GPU current Kcur/Vcur captured after producer and replayed bytewise into CPU teacher step before cache store"}};
                current["rows"].back()["current_kv_proof"] = kv_proof;

                // CPU reference is disposable. Restore the already-computed
                // GPU post-step state so only that trajectory advances.
                restore_full_file(post_gpu, past+1);
                const auto restored_post = path("native-prefix-post-restored.bin");
                full_prefix_file(restored_post, past+1);
                const json post_proof = compare_prefix_files(post_gpu, restored_post);
                require(post_proof["equal"].get<bool>(), "GPU post-step native state changed during CPU replay/restore at N="+std::to_string(n));
                current["rows"].back()["post_gpu_restore_proof"] = post_proof;
            } else {
                // Natural paired reference remains a numeric diagnostic only;
                // it is not eligible for the frozen-memory structural protocol.
                auto began = std::chrono::steady_clock::now();
                paired_decode(past, 1, "reference-n"+std::to_string(n), opt.reference == "cpu");
                reference_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
                reference_logits = logits();
                capture(reference_dir, reference_phase, n, reference_logits,
                        {{"verified", false}, {"pending_restore", true}}, checkpoint_hash);
                current["rows"].back()["reference_decode_seconds"] = reference_seconds;
                current["rows"].back()["reference_arm"] = opt.reference; publish();
                if (opt.checkpoint_method == "full") { restore_full_file(before); }
                else { restore_partial(checkpoint); }
            }
            if (!opt.fixed_current_kv && opt.verify_full_prefix) {
                full_prefix_file(after);
                proof = compare_prefix_files(before, after);
                proof["verified"] = proof["equal"];
                current["rows"].back()["prefix_proof"] = proof; publish();
                require(proof["equal"].get<bool>(),
                        "full native prefix bytes changed after "+opt.checkpoint_method+" restore at N="+std::to_string(n)+
                        "; inspect metadata/record-boundary mutation; no numerical acceptance is claimed");
            }
            if (!opt.fixed_current_kv) {
                auto began = std::chrono::steady_clock::now();
                paired_decode(past, 1, "gpu-n"+std::to_string(n), false);
                gpu_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
                gpu_logits = logits();
                capture(reference_dir, reference_phase, n, reference_logits, proof, checkpoint_hash);
            } else {
                capture(reference_dir, reference_phase, n, reference_logits, proof, checkpoint_hash);
            }
            capture(root/"gpu", "gpu", n, gpu_logits, proof, checkpoint_hash);
            auto metrics = logits_metrics(reference_logits, gpu_logits, tokens.at(size_t(n)));
            auto & completed = current["rows"].back();
            completed["state"] = "complete";
            completed["prefix_proof"] = proof;
            if (opt.fixed_current_kv) {
                completed["current_kv_proof"] = kv_proof;
                completed["fixed_current_kv"] = true;
            }
            completed["reference_decode_seconds"] = reference_seconds;
            completed["reference_arm"] = opt.reference;
            completed["gpu_decode_seconds"] = gpu_seconds;
            completed["metrics"] = metrics;
            ++past;
            publish();
        std::fprintf(stderr, "paired: N=%d reference=%s prefix_equal=%s checkpoint=%s/%llu B KL=%.9g REF=%.3fs GPU=%.3fs\n",
                    n, opt.reference.c_str(), opt.verify_full_prefix ? "true" : "not_verified", opt.checkpoint_method.c_str(),
                    static_cast<unsigned long long>(checkpoint_bytes),
                    metrics["kl_nat"].get<double>(), reference_seconds, gpu_seconds);
        }
        if (opt.quality_only) {
            bool trajectory_complete = opt.count >= 32 && current["rows"].size() == size_t(opt.count) &&
                std::all_of(current["rows"].begin(), current["rows"].end(), [](const json & row) {
                    return row.value("state", std::string()) == "complete" &&
                        row.value("gpu_logits_count", size_t(0)) > 0;
                });
            for (size_t i = 0; i < current["rows"].size(); ++i) {
                trajectory_complete = trajectory_complete &&
                    current["rows"][i].value("n", -1) == start + int(i);
            }
            current["complete"] = trajectory_complete;
            current["quality_only"] = true;
            current["quality_trajectory_reference"] = "GPU-only teacher-forced trajectory; no CPU reference arm";
            current["quality_trajectory_complete"] = trajectory_complete;
            current["quality_comparison_required"] = true;
            current["structural_gate_evaluated"] = false;
            current["structural_gate_pass"] = nullptr;
            current["protocol_acceptance_eligible"] = false;
            current["fixed_memory_reference_complete"] = false;
            current["calibration_candidate_eligible"] = false;
            current["calibration_baseline_eligible"] = false;
            current["external_calibrated_gate_required"] = false;
            publish();
            return;
        }
        std::vector<double> kls;
        double sum = 0;
        for (const auto & row : current["rows"]) { const double kl = row["metrics"]["kl_nat"]; kls.push_back(kl); sum += kl; }
        std::sort(kls.begin(), kls.end());
        const double mean = sum/kls.size();
        const double p99 = kls[std::min(kls.size()-1, size_t(std::ceil(kls.size()*.99)-1))];
        current["complete"] = true;
        current["mean_kl_nat_per_token"] = mean;
        current["p99_kl_nat_per_token"] = p99;
        current["min_kl_nat_per_token"] = kls.front();
        current["absolute_numeric_diagnostic_pass"] = kls.front() >= -1e-10 && mean <= 1e-4 && p99 <= 1e-3;
        current["structural_gate_evaluated"] = false;
        current["fixed_memory_reference_complete"] = current["complete"].get<bool>() && opt.prefix_state_in.empty() && opt.fixed_current_kv &&
            opt.reference == "cpu" && opt.checkpoint_method == "full" && opt.verify_full_prefix && opt.count >= 32 &&
            std::all_of(current["rows"].begin(), current["rows"].end(), [](const json & row) {
                return row.contains("current_kv_proof") && row["current_kv_proof"]["complete"].get<bool>() &&
                    row["prefix_proof"]["verified"].get<bool>() && row["post_gpu_restore_proof"]["equal"].get<bool>();
            });
        current["external_calibrated_gate_required"] = true;
        current["calibrated_kl_gate"] = "not evaluated here; compare calibrated mean/p99 KL deltas externally";
        current["f16_quality_gate"] = "not evaluated here; requires separate O/LSE/F16 quality evidence";
        current["structural_gate_pass"] = nullptr;
        current["protocol_acceptance_eligible"] = current["fixed_memory_reference_complete"];
        current["calibration_baseline_eligible"] = current["fixed_memory_reference_complete"].get<bool>() &&
            (opt.pure_cuda_baseline || opt.layered_baseline);
        current["calibration_candidate_eligible"] = current["fixed_memory_reference_complete"].get<bool>() &&
            !opt.pure_cuda_baseline && !opt.layered_baseline && params.kv_position_split_p > 0;
        publish();
    }

    int run() {
        const uint32_t endian = 1;
        require(*reinterpret_cast<const uint8_t *>(&endian) == 1, "capture binary format requires little-endian host");
        tokens = read_tokens(opt);
        require(opt.checkpoint_method == "full" || opt.checkpoint_method == "partial", "paired checkpoint must be full or partial");
        require(opt.quality_only || (!opt.prefix_state_out && !opt.diagnostic_p_above_target),
                "prefix cache export and P105216 override are diagnostic quality-only options");
        require(opt.prefix_state_in.empty() || opt.quality_only ||
                (opt.fixed_current_kv && opt.reference == "cpu" &&
                 opt.checkpoint_method == "full" && opt.verify_full_prefix),
                "diagnostic prefix import requires quality-only or fixed-current-KV CPU replay with verified FULL checkpoints");
        require(opt.prefix_state_in.empty() || !opt.prefix_state_out,
                "choose prefix cache import or export, not both");
        require(!(opt.quality_only && opt.fixed_current_kv),
                "--paired-quality-only and --paired-fixed-current-kv are mutually exclusive");
        require(!opt.fixed_current_kv || (opt.reference == "cpu" && opt.checkpoint_method == "full"),
                "--paired-fixed-current-kv requires CPU reference and FULL native checkpoints");
        require(params.n_parallel == 1 && std::all_of(params.speculative.types.begin(), params.speculative.types.end(),
                    [](common_speculative_type type) { return type == COMMON_SPECULATIVE_TYPE_NONE; }),
                "paired reference requires parallel=1 and speculation disabled");
        require(params.n_batch > 0 && params.n_ubatch > 0, "paired reference requires explicit positive batch and ubatch sizes");
        require(!(opt.pure_cuda_baseline && opt.layered_baseline), "choose at most one baseline mode");
        if (opt.pure_cuda_baseline) {
            require(params.kv_position_split_p == 0,
                    "--paired-pure-cuda-baseline requires P=0 (omit --kv-position-split-p)");
            require(params.remote_attn_host.empty(),
                    "--paired-pure-cuda-baseline requires remote attention disabled (omit --remote-attn)");
            require(params.remote_attn_cache_type_k == GGML_TYPE_COUNT &&
                    params.remote_attn_cache_type_v == GGML_TYPE_COUNT,
                    "--paired-pure-cuda-baseline does not allow remote Qx cache overrides");
            bool has_cuda_device = false;
            for (ggml_backend_dev_t dev : params.devices) {
                if (dev == nullptr) { continue; }
                const char * name = ggml_backend_dev_name(dev);
                if (name != nullptr && std::strncmp(name, "Vulkan", 6) == 0) {
                    throw std::runtime_error(
                        "--paired-pure-cuda-baseline requires no Vulkan device in --device");
                }
                has_cuda_device |= name != nullptr && std::strncmp(name, "CUDA", 4) == 0;
            }
            require(has_cuda_device,
                    "--paired-pure-cuda-baseline requires an explicit CUDA device in --device");
            // Context initialization accepts this legacy environment override
            // when P is zero in common_params. Clear it only for this explicit
            // pure-baseline mode so the actual context is guaranteed P=0.
            common_set_env("LLAMA_KV_POSITION_SPLIT_P", "");
        } else if (opt.layered_baseline) {
            require(params.kv_position_split_p == 0,
                    "--paired-layered-baseline requires P=0 (omit --kv-position-split-p)");
            require(!params.remote_attn_host.empty(),
                    "--paired-layered-baseline requires an explicit --remote-attn host");
            require(!params.remote_attn_layers.empty() && params.remote_attn_layers != "auto",
                    "--paired-layered-baseline requires an explicit --remote-attn-layers value");
            common_set_env("LLAMA_KV_POSITION_SPLIT_P", "");
        } else {
            require(params.kv_position_split_p > 0 && params.kv_position_split_p%256 == 0 &&
                    (params.kv_position_split_p <= 102400 ||
                     (opt.diagnostic_p_above_target && opt.quality_only && params.kv_position_split_p == 105216)) &&
                    params.kv_position_split_p < uint32_t(params.n_ctx),
                    "invalid split P/C");
        }
        require(opt.count > 0, "paired count must be positive");
        if (opt.starts.empty()) { opt.starts = {513}; }
        std::sort(opt.starts.begin(), opt.starts.end());
        int previous_end = 0;
        for (int start : opt.starts) {
            require(start > previous_end && start > 1 && int64_t(start)+opt.count <= params.n_ctx &&
                    uint64_t(start)+uint64_t(opt.count)-1 < tokens.size(), "invalid, overlapping, or out-of-capacity teacher range");
            previous_end = start+opt.count-1;
        }
        reference_cpu(false);
        common_set_env("LLAMA_GRAPH_REUSE_DISABLE", "1");
        if (opt.quality_only) {
            common_set_env("F4_TRACE_CAPTURE", "");
            common_set_env("F4_NODE_DUMP", "");
            common_set_env("F4_TEST_LOGIT_CAPTURE", "");
        }
        if (!opt.quality_only) {
            fs::create_directories(path("native-spool"));
            common_set_env("TMPDIR", path("native-spool").string());
        }
        params.fit_params = false;
        params.cb_eval = opt.quality_only ? nullptr : paired_kv_eval_callback;
        params.cb_eval_user_data = opt.quality_only ? nullptr : &kv_callback;
        init = common_init_from_params(params);
        require(init && init->model() && init->context(), "model/context initialization failed; inspect external OOM/driver logs");
        ctx = init->context();
        if (opt.fixed_current_kv) {
            const auto & hparams = init->model()->hparams;
            for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
                if (hparams.has_kv(il) && !hparams.is_recr(il)) {
                    kv_callback.expected_layers.insert(int(il));
                    kv_callback.expected_geometry[int(il)] = {hparams.n_embd_head_k(il), hparams.n_head_kv(il)};
                }
            }
            require(!kv_callback.expected_layers.empty(), "model has no eligible full-attention current-KV layers");
        }
        require(llama_n_ctx(ctx) == uint32_t(params.n_ctx), "effective context capacity changed");
        vocab = llama_vocab_n_tokens(llama_model_get_vocab(init->model()));
        for (auto token : tokens) { require(token >= 0 && token < vocab, "teacher token outside model vocabulary"); }
        batch = llama_batch_init(params.n_batch, 0, 1);
        config_core = "model_sha256="+opt.model_sha256+";n_ctx="+std::to_string(llama_n_ctx(ctx))+
            ";n_batch="+std::to_string(llama_n_batch(ctx))+";n_ubatch="+std::to_string(llama_n_ubatch(ctx))+
            ";n_seq="+std::to_string(llama_n_seq_max(ctx))+";n_params="+std::to_string(llama_model_n_params(init->model()))+
            ";p="+std::to_string(params.kv_position_split_p)+";kv_k="+std::to_string(params.cache_kvarn_bits_k)+
            ";kv_v="+std::to_string(params.cache_kvarn_bits_v)+";overflow_k="+
            std::to_string(params.remote_attn_cache_type_k)+";overflow_v="+std::to_string(params.remote_attn_cache_type_v)+
            ";cache_type_k="+std::to_string(int(params.cache_type_k))+":"+ggml_type_name(params.cache_type_k)+
            ";cache_type_v="+std::to_string(int(params.cache_type_v))+":"+ggml_type_name(params.cache_type_v)+
            ";remote_layers="+params.remote_attn_layers;
        report["configuration"] = {{"config_core", config_core}, {"model", params.model.path},
            {"cache_type_k_numeric", int(params.cache_type_k)}, {"cache_type_k_name", ggml_type_name(params.cache_type_k)},
            {"cache_type_v_numeric", int(params.cache_type_v)}, {"cache_type_v_name", ggml_type_name(params.cache_type_v)},
            {"reference_arm", opt.quality_only ? "gpu_quality_only" : opt.reference}, {"pure_cuda_baseline", opt.pure_cuda_baseline},
            {"quality_only", opt.quality_only},
            {"reference_option_ignored_in_quality_only", opt.quality_only ? json(opt.reference) : json(nullptr)},
            {"layered_baseline", opt.layered_baseline},
            {"arm_classification", arm_classification()},
            {"model_sha256_operator_verified", opt.model_sha256}, {"build_bundle_sha256_operator_verified", opt.build_id},
            {"compiled_commit", llama_commit()}, {"compiled_build", llama_build_info()},
            {"n_rs_seq", llama_n_rs_seq(ctx)}, {"teacher_file", file_observation(opt.tokens_file)},
            {"token_format", opt.token_format}, {"starts", opt.starts}, {"count", opt.count},
            {"checkpoint_method", opt.quality_only ? "none_quality_trajectory" : opt.checkpoint_method},
            {"state_spool_directory", opt.quality_only ? json(nullptr) : json(path("native-spool").string())},
            {"full_prefix_verification", opt.verify_full_prefix}, {"fixed_current_kv", opt.fixed_current_kv}};
        write_json(path("teacher-tokens.json"), tokens);
        publish();
        for (auto start : opt.starts) { range(start); }
        reference_cpu(false);
        if (opt.quality_only) {
            bool quality_complete = opt.count >= 32 && !report["ranges"].empty();
            for (const auto & range_result : report["ranges"]) {
                quality_complete = quality_complete && range_result.value("quality_trajectory_complete", false);
            }
            report["complete"] = quality_complete;
            report["quality_only"] = true;
            report["quality_trajectory_reference"] = "GPU-only teacher-forced trajectory; reference option does not select a CPU arm";
            report["quality_trajectory_complete"] = quality_complete;
            report["quality_comparison_required"] = true;
            report["quality_comparison_scope"] = "compare GPU-only captures to the requested F16/F32 quality reference externally; no acceptance gate is evaluated here";
            report["protocol_acceptance_eligible"] = false;
            report["fixed_memory_reference_complete"] = false;
            report["calibration_candidate_eligible"] = false;
            report["calibration_baseline_eligible"] = false;
            report["structural_gate_evaluated"] = false;
            report["structural_gate_pass"] = nullptr;
            report["external_calibrated_gate_required"] = false;
            report["allocation_outcome"] = quality_complete
                ? "all requested GPU teacher positions decoded once on one persistent trajectory; no native checkpoint or CPU arm was run"
                : "incomplete GPU-only quality trajectory; some requested positions are missing";
            publish();
            return quality_complete ? 0 : 2;
        }
        bool fixed_complete = opt.prefix_state_in.empty() && opt.fixed_current_kv && opt.reference == "cpu" &&
            opt.checkpoint_method == "full" && opt.verify_full_prefix && opt.count >= 32;
        for (const auto & range_result : report["ranges"]) {
            fixed_complete = fixed_complete && range_result["fixed_memory_reference_complete"].get<bool>();
        }
        report["complete"] = true;
        report["fixed_current_kv"] = opt.fixed_current_kv;
        report["fixed_memory_reference_complete"] = fixed_complete;
        report["calibration_baseline_eligible"] = fixed_complete && (opt.pure_cuda_baseline || opt.layered_baseline);
        report["calibration_candidate_eligible"] = fixed_complete && !opt.pure_cuda_baseline &&
            !opt.layered_baseline && params.kv_position_split_p > 0;
        report["external_calibrated_gate_required"] = true;
        report["calibrated_kl_gate"] = "not evaluated here; compare calibrated mean/p99 KL deltas externally";
        report["f16_quality_gate"] = "not evaluated here; requires separate O/LSE/F16 quality evidence";
        report["protocol_acceptance_eligible"] = fixed_complete;
        report["structural_gate_evaluated"] = false;
        report["structural_gate_pass"] = nullptr;
        report["external_comparator_scope"] = opt.prefix_state_in.empty()
            ? "compare_logits.py supplies numerical comparison; acceptance also requires this runner's verified native-prefix protocol result"
            : "diagnostic imported prefix: numerical comparison only; protocol and calibration acceptance remain ineligible";
        report["allocation_outcome"] = "all requested initialization/native restores/reference+GPU decodes returned successfully; driver diagnostics require external logs";
        publish();
        if (fixed_complete) { return 0; }
        return 2;
    }
};

} // namespace

int main(int argc, char ** argv) {
    runner run;
    run.params.n_ctx = 2048;
    run.params.n_batch = 512; run.params.n_ubatch = 256;
    run.params.n_parallel = 1;
    run.params.sampling.temp = 0;
    run.params.speculative.types.clear(); run.params.speculative.draft.n_max = 0;
    std::vector<char *> common_args{argv[0]};
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string flag(argv[i]);
            if (flag == "--paired-help") { usage(); return 0; }
            if (flag == "--paired-verify-full-prefix") { run.opt.verify_full_prefix = true; continue; }
            if (flag == "--paired-pure-cuda-baseline") { run.opt.pure_cuda_baseline = true; continue; }
            if (flag == "--paired-layered-baseline") { run.opt.layered_baseline = true; continue; }
            if (flag == "--paired-fixed-current-kv") { run.opt.fixed_current_kv = true; continue; }
            if (flag == "--paired-quality-only") { run.opt.quality_only = true; continue; }
            if (flag == "--paired-prefix-state-out") { run.opt.prefix_state_out = true; continue; }
            if (flag == "--paired-diagnostic-p105216") { run.opt.diagnostic_p_above_target = true; continue; }
            if (flag.compare(0, 9, "--paired-") != 0) { common_args.push_back(argv[i]); continue; }
            require(i+1 < argc, "missing value for "+flag);
            const std::string value(argv[++i]);
            if (flag == "--paired-token-file") { run.opt.tokens_file = value; }
            else if (flag == "--paired-token-format") { run.opt.token_format = value; }
            else if (flag == "--paired-output-dir") { run.opt.output = value; }
            else if (flag == "--paired-model-sha256") { run.opt.model_sha256 = value; }
            else if (flag == "--paired-build-id") { run.opt.build_id = value; }
            else if (flag == "--paired-prefix-state-in") { run.opt.prefix_state_in = value; }
            else if (flag == "--paired-checkpoint") { run.opt.checkpoint_method = value; }
            else if (flag == "--paired-reference") { run.opt.reference = value; }
            else if (flag == "--paired-start") { run.opt.starts.push_back(std::stoi(value)); }
            else if (flag == "--paired-count") { run.opt.count = std::stoi(value); }
            else { throw std::runtime_error("unknown paired option "+flag); }
        }
        if (run.opt.quality_only && (run.opt.pure_cuda_baseline || run.opt.layered_baseline)) {
            common_set_env("LLAMA_KV_POSITION_SPLIT_P", "");
        }
        common_init();
        common_args.push_back(nullptr);
        if (!common_params_parse(int(common_args.size()-1), common_args.data(), run.params, LLAMA_EXAMPLE_SERVER)) { return 1; }
        require(!run.opt.tokens_file.empty() && !run.opt.output.empty(), "paired token file and new output directory are required; see --paired-help");
        require(run.opt.reference == "cpu" || run.opt.reference == "gpu",
                "paired reference arm must be cpu or gpu");
        require(sha256_text(run.opt.model_sha256) && sha256_text(run.opt.build_id), "provide independently verified 64-hex model and build bundle SHA256 hashes");
        require(!fs::exists(run.opt.output) || fs::is_empty(run.opt.output), "paired output directory must be empty; existing evidence is never overwritten");
        fs::create_directories(run.opt.output);
        run.report = {{"schema", "occupancy-native-paired-teacher-v1"}, {"complete", false},
            {"structural_gate_pass", nullptr}, {"protocol_acceptance_eligible", false},
            {"calibration_baseline_eligible", false}, {"calibration_candidate_eligible", false},
            {"external_calibrated_gate_required", false}, {"ranges", json::array()},
            {"reference_arm", run.opt.quality_only ? "gpu_quality_only" : run.opt.reference},
            {"quality_only", run.opt.quality_only}, {"quality_trajectory_complete", false},
            {"diagnostic_prefix_import", !run.opt.prefix_state_in.empty()},
            {"prefix_import_scope", run.opt.prefix_state_in.empty() ? "none" :
                "diagnostic only: imported native prefix cannot establish protocol or calibration acceptance"},
            {"quality_trajectory_reference", run.opt.quality_only ? "GPU-only teacher-forced trajectory; no CPU reference arm" : "paired or diagnostic protocol"},
            {"quality_comparison_required", run.opt.quality_only},
            {"pure_cuda_baseline", run.opt.pure_cuda_baseline},
            {"layered_baseline", run.opt.layered_baseline},
            {"arm_classification", run.arm_classification()},
            {"structural_gate_evaluated", false},
            {"checkpoint_method", run.opt.quality_only ? "none_quality_trajectory" : run.opt.checkpoint_method},
            {"protocol", run.opt.quality_only
                ? "GPU prefix -> exactly one GPU teacher decode per consecutive position -> capture full-vocabulary logits and teacher prefix -> advance same GPU trajectory; no CPU arm, native save/restore, or structural/calibration eligibility"
                : (run.opt.fixed_current_kv
                ? "GPU prefix -> capture GPU post-RoPE current K/V -> GPU teacher step and save post-N native state -> restore exact pre-N state -> CPU teacher step with byte-exact frozen GPU current K/V before cache store -> restore exact GPU post-N state; only GPU trajectory advances"
                : "diagnostic only: natural selected reference-arm teacher decode and native restore followed by GPU candidate; current K/V may differ downstream")},
            {"fixed_current_kv", run.opt.fixed_current_kv}, {"fixed_memory_reference_complete", false},
            {"allocation_outcome", "incomplete: abort/OOM/driver failure requires external log classification"},
            {"argv", json::array()}};
        for (int i = 0; i < argc; ++i) { run.report["argv"].push_back(argv[i]); }
        run.publish();
        ggml_backend_load_all();
        return run.run();
    } catch (const std::exception & error) {
        std::fprintf(stderr, "paired: FAIL: %s\n", error.what());
        if (!run.opt.output.empty() && !run.report.is_null()) {
            run.report["error"] = error.what();
            try { run.publish(); } catch (...) {}
        }
        return 1;
    }
}
