// Explicit real-model campaign runner. Built, but never registered with CTest:
// a kCapacityC-token prefill must only run when an operator requests it.
#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "llama.h"
#include "llama-ext.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"
#include "llama-kv-cache-kvarn.h"
#include "llama-io.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::ordered_json;
namespace fs = std::filesystem;

namespace {

void require(bool ok, const std::string & what) {
    if (!ok) { throw std::runtime_error(what); }
}

struct options {
    std::string tokens;
    std::string format = "json";
    std::string output;
    std::string work;
    std::string resume;
    std::string mode = "all";
    int cycles = 10;
    int generate = 512;
    int cycle_hold_ms = 0;
    bool discard_verified_scratch = false;
};

void usage() {
    std::fprintf(stderr,
        "Real-model occupancy campaign, with ordinary llama CLI model/cache/backend flags.\n"
        "  --capacity-token-file FILE      JSON token array (or {tokens:[...]}) / LE int32\n"
        "  --capacity-token-format FORMAT  json (default) or int32\n"
        "  --capacity-output FILE          JSON result, updated after every phase\n"
        "  --capacity-work-dir DIR         native snapshots and reference logits on disk\n"
        "  --capacity-resume-file FILE     optional native prefix snapshot from this runner\n"
        "  --capacity-mode MODE            all (default), capacity, cycles, boundary, mismatch, empty-local\n"
        "  --capacity-cycles N             default 10, first cycle is the warmed reference\n"
        "  --capacity-generate N           default 512 (prefix is C-N)\n"
        "  --capacity-cycle-hold-ms N      optional stable telemetry window, 1..60000 ms after every cycle\n"
        "  --capacity-discard-verified-scratch  delete closed, byte-verified runner scratch files\n"
        "Generation is greedy, continues through EOG, and fills every position through C-1.\n"
        "No result from a smaller C or fewer than ten cycles proves the production gate.\n");
    std::fprintf(stderr, "empty-local: isolated attention-only clear at exactP; recurrent bytes preserved.\n"
                        "Requires P>0 and fresh work files; resume/generation/cycle/hold options do not execute.\n");
}

class bounded_state_writer final : public llama_io_write_i {
    std::ofstream out;
    std::vector<uint8_t> scratch = std::vector<uint8_t>(1u << 20);
    size_t written = 0;
public:
    explicit bounded_state_writer(const fs::path & path) : out(path, std::ios::binary) {
        require(bool(out), "cannot open recurrent state output: " + path.string());
    }
    void write(const void * src, size_t size) override {
        require(size <= std::numeric_limits<size_t>::max()-written, "recurrent state byte count overflow");
        for (size_t at = 0; at < size;) {
            const size_t count = std::min(scratch.size(), size-at);
            out.write(static_cast<const char *>(src)+at, std::streamsize(count));
            require(bool(out), "recurrent state write failed"); at += count;
        }
        written += size;
    }
    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        require(offset <= ggml_nbytes(tensor) && size <= ggml_nbytes(tensor)-offset, "recurrent tensor span invalid");
        for (size_t at = 0; at < size;) {
            const size_t count = std::min(scratch.size(), size-at);
            ggml_backend_tensor_get(tensor, scratch.data(), offset+at, count);
            write(scratch.data(), count); at += count;
        }
    }
    size_t n_bytes() override { return written; }
    void finish() { out.close(); require(bool(out), "recurrent state close failed"); }
};

std::vector<llama_token> read_tokens(const options & opt) {
    std::ifstream in(opt.tokens, std::ios::binary);
    require(bool(in), "cannot open token file");
    std::vector<llama_token> tokens;
    if (opt.format == "json") {
        auto data = json::parse(in);
        if (data.is_object()) { data = data.at("tokens"); }
        require(data.is_array(), "token JSON must be an array or an object with tokens");
        for (const auto & value : data) {
            require(value.is_number_integer(), "non-integer token");
            const auto token = value.get<int64_t>();
            require(token >= 0 && token <= INT32_MAX, "token outside int32 range");
            tokens.push_back(llama_token(token));
        }
    } else {
        require(opt.format == "int32", "unknown token format");
        std::array<unsigned char, 4> bytes;
        while (in.read(reinterpret_cast<char *>(bytes.data()), bytes.size())) {
            const uint32_t token = uint32_t(bytes[0]) | uint32_t(bytes[1]) << 8 |
                                   uint32_t(bytes[2]) << 16 | uint32_t(bytes[3]) << 24;
            require(token <= INT32_MAX, "negative token in int32 file");
            tokens.push_back(llama_token(token));
        }
        require(in.gcount() == 0, "truncated int32 token file");
    }
    return tokens;
}

json file_observation(const fs::path & path) {
    std::ifstream in(path, std::ios::binary);
    require(bool(in), "cannot hash " + path.string());
    std::array<char, 65536> buf;
    uint64_t hash = UINT64_C(14695981039346656037);
    uint64_t bytes = 0;
    while (in.read(buf.data(), buf.size()) || in.gcount()) {
        const auto count = size_t(in.gcount());
        bytes += count;
        for (size_t i = 0; i < count; ++i) {
            hash ^= uint8_t(buf[i]);
            hash *= UINT64_C(1099511628211);
        }
    }
    require(in.eof(), "file read failed while hashing");
    return {{"path", path.string()}, {"bytes", bytes}, {"fnv1a64", hash}};
}

// Deliberately exact: hashing alone does not decide transactional intactness.
json compare_files(const fs::path & a, const fs::path & b) {
    std::ifstream left(a, std::ios::binary), right(b, std::ios::binary);
    require(bool(left) && bool(right), "cannot open files for state comparison");
    std::array<char, 65536> x, y;
    uint64_t offset = 0;
    for (;;) {
        left.read(x.data(), x.size()); right.read(y.data(), y.size());
        const size_t nx = size_t(left.gcount()), ny = size_t(right.gcount());
        if (nx != ny) { return {{"equal", false}, {"first_difference", offset}, {"reason", "length"}}; }
        if (std::memcmp(x.data(), y.data(), nx)) {
            size_t i = 0;
            while (i < nx && x[i] == y[i]) { ++i; }
            return {{"equal", false}, {"first_difference", offset + i}, {"reason", "bytes"}};
        }
        offset += nx;
        if (nx == 0) {
            require(left.eof() && right.eof(), "state comparison I/O failed");
            return {{"equal", true}, {"bytes_compared", offset}};
        }
    }
}

struct logits_comparison {
    std::vector<double> kl;
    double max_abs = 0;
    uint64_t unequal = 0;

    void add(const float * current, const std::vector<float> & ref) {
        const size_t n = ref.size();
        bool exact = true;
        double mr = -std::numeric_limits<double>::infinity();
        double mc = mr;
        for (size_t i = 0; i < n; ++i) {
            require(std::isfinite(current[i]) && std::isfinite(ref[i]), "nonfinite logits");
            mr = std::max(mr, double(ref[i])); mc = std::max(mc, double(current[i]));
            max_abs = std::max(max_abs, std::abs(double(current[i]) - ref[i]));
            if (current[i] != ref[i]) { exact = false; ++unequal; }
        }
        if (exact) { kl.push_back(0); return; }
        double zr = 0, zc = 0;
        for (size_t i = 0; i < n; ++i) { zr += std::exp(double(ref[i])-mr); zc += std::exp(double(current[i])-mc); }
        const double lr = mr + std::log(zr), lc = mc + std::log(zc);
        double value = 0;
        for (size_t i = 0; i < n; ++i) {
            value += std::exp(double(ref[i])-lr) * (double(ref[i])-lr-double(current[i])+lc);
        }
        // Keep the signed FP64 result: cancellation may yield a tiny negative,
        // but a substantial negative must not be concealed as perfect parity.
        kl.push_back(value);
    }

    json result() const {
        auto ordered = kl;
        std::sort(ordered.begin(), ordered.end());
        double sum = 0;
        for (auto value : ordered) { sum += value; }
        const double mean = ordered.empty() ? 0 : sum / ordered.size();
        const double p99 = ordered.empty() ? 0 : ordered[std::min(ordered.size()-1, size_t(std::ceil(ordered.size()*.99)-1))];
        const double minimum = ordered.empty() ? 0 : ordered.front();
        return {{"rows", kl.size()}, {"max_abs", max_abs}, {"unequal_values", unequal},
                {"kl_mean_nat", mean}, {"kl_p99_nat", p99}, {"kl_min_nat", minimum},
                {"pass", minimum >= -1e-10 && mean <= 1e-4 && p99 <= 1e-3}};
    }
};

struct runner {
    options opt;
    common_params params;
    common_init_result_ptr init;
    llama_context * ctx = nullptr;
    int vocab = 0;
    int c = 0, p = 0, prefix = 0;
    std::vector<llama_token> input, history;
    json report;
    llama_batch batch{};
    std::map<std::string, std::string> generated_files;
    json scratch_retention = {{"enabled", false}, {"hash_algorithm", "FNV-1a-64 (non-cryptographic)"},
                              {"events", json::array()}, {"retained_files", json::array()}};
    const std::chrono::steady_clock::time_point campaign_started = std::chrono::steady_clock::now();

    ~runner() { if (batch.token) { llama_batch_free(batch); } }

    fs::path file(const std::string & name) const { return fs::path(opt.work) / name; }
    llama_pos attention_position_min() const {
        auto * memory = llama_get_memory(ctx);
        if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(memory)) {
            memory = hybrid->get_mem_attn_base();
        }
        // Hybrid's public minimum also accounts for the current recurrent
        // state. Position-range checks need the attention cell table itself.
        return memory->seq_pos_min(0);
    }
    void publish() {
        const auto temp = opt.output + ".tmp";
        std::ofstream out(temp);
        require(bool(out), "cannot write result JSON");
        out << report.dump(2) << '\n'; out.close();
        require(bool(out), "result JSON write failed");
        fs::rename(temp, opt.output);
    }

    json memory() {
        llama_synchronize(ctx);
        json allocated = json::object(), devices = json::array();
        for (const auto & entry : llama_get_memory_breakdown(ctx)) {
            const auto buft = entry.first;
            const auto dev = ggml_backend_buft_get_device(buft);
            const std::string name = dev ? ggml_backend_dev_name(dev) : ggml_backend_buft_name(buft);
            if (!allocated.contains(name)) { allocated[name] = {{"model", 0}, {"context", 0}, {"compute", 0}, {"total", 0}}; }
            auto & value = allocated[name];
            value["model"] = value["model"].get<uint64_t>() + entry.second.model;
            value["context"] = value["context"].get<uint64_t>() + entry.second.context;
            value["compute"] = value["compute"].get<uint64_t>() + entry.second.compute;
            value["total"] = value["total"].get<uint64_t>() + entry.second.total();
        }
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto dev = ggml_backend_dev_get(i);
            size_t free = 0, total = 0;
            ggml_backend_dev_memory(dev, &free, &total);
            devices.push_back({{"name", ggml_backend_dev_name(dev)}, {"description", ggml_backend_dev_description(dev)},
                               {"free", free}, {"total", total}});
        }
        return {{"llama_accounted_bytes", allocated}, {"device_memory_observation", devices}};
    }

    int decode(const llama_token * tokens, int n, int position) {
        common_batch_clear(batch);
        for (int i = 0; i < n; ++i) { common_batch_add(batch, tokens[i], position+i, {0}, i == n-1); }
        const int rc = llama_decode(ctx, batch);
        llama_synchronize(ctx);
        return rc;
    }

    void save(const fs::path & path) {
        llama_synchronize(ctx);
        require(llama_state_seq_save_file(ctx, path.c_str(), 0, history.data(), history.size()) != 0,
                "native sequence snapshot failed: " + path.string());
        generated_files[path.lexically_normal().string()] = "runner snapshot; retained unless a closed exact comparison permits cleanup";
    }

    void discard_verified(const fs::path & path, const json & proof, const std::string & reason) {
        if (!opt.discard_verified_scratch || !proof.is_object() || !proof.value("equal", false)) { return; }
        const auto key = path.lexically_normal().string();
        const auto found = generated_files.find(key);
        if (found == generated_files.end() || !fs::exists(path)) { return; }
        const auto root = fs::weakly_canonical(opt.work);
        const auto candidate = fs::weakly_canonical(path);
        if (candidate.parent_path() != root ||
                (!opt.resume.empty() && candidate == fs::weakly_canonical(opt.resume))) { return; }
        auto observation = file_observation(path);
        const bool removed = fs::remove(path);
        require(removed, "verified scratch removal failed: " + path.string());
        scratch_retention["events"].push_back({{"path", path.string()}, {"action", "deleted_after_exact_comparison"},
            {"reason", reason}, {"bytes", observation["bytes"]}, {"fnv1a64", observation["fnv1a64"]},
            {"comparison", proof}});
        generated_files.erase(found);
        report["scratch_retention"] = scratch_retention;
        publish();
    }

    void finish_retention_manifest() {
        scratch_retention["enabled"] = opt.discard_verified_scratch;
        scratch_retention["retained_files"] = json::array();
        if (opt.discard_verified_scratch) {
            for (const auto & item : generated_files) {
                const fs::path path(item.first);
                if (fs::exists(path)) {
                    auto observation = file_observation(path);
                    scratch_retention["retained_files"].push_back({{"path", path.string()},
                        {"reason", item.second}, {"bytes", observation["bytes"]}, {"fnv1a64", observation["fnv1a64"]}});
                }
            }
        }
        report["scratch_retention"] = scratch_retention;
    }

    void save_state_bytes(const fs::path & path) {
        llama_synchronize(ctx);
        const size_t size = llama_state_seq_get_size(ctx, 0);
        require(size > 0, "native sequence state has zero byte size");
        std::vector<uint8_t> bytes(size);
        require(llama_state_seq_get_data(ctx, bytes.data(), bytes.size(), 0) == bytes.size(),
                "native sequence state byte capture failed");
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        require(bool(out), "cannot open native sequence state bytes: " + path.string());
        out.write(reinterpret_cast<const char *>(bytes.data()), std::streamsize(bytes.size()));
        out.close();
        require(bool(out), "native sequence state byte write failed: " + path.string());
        generated_files[path.lexically_normal().string()] = "runner state-byte export";
    }

    bool restore(const fs::path & path) {
        std::vector<llama_token> tokens(size_t(c), 0);
        size_t count = 0;
        llama_synchronize(ctx);
        const size_t read = llama_state_seq_load_file(ctx, path.c_str(), 0, tokens.data(), tokens.size(), &count);
        if (!read) { return false; }
        tokens.resize(count); history = std::move(tokens);
        require(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == int64_t(history.size())-1,
                "restored position does not match native snapshot token count");
        return true;
    }

    llama_token greedy() {
        const float * logits = llama_get_logits_ith(ctx, -1);
        require(logits != nullptr, "missing generation logits");
        llama_token best = 0;
        for (int i = 0; i < vocab; ++i) {
            require(std::isfinite(logits[i]), "nonfinite generation logits");
            if (logits[i] > logits[best]) { best = i; }
        }
        return best;
    }

    json generate(int n, llama_token first, const fs::path & logits_path, bool reference) {
        std::ofstream output;
        std::ifstream comparison;
        if (reference) { output.open(logits_path, std::ios::binary); require(bool(output), "cannot open reference logits output"); }
        else { comparison.open(logits_path, std::ios::binary); require(bool(comparison), "cannot open reference logits input"); }
        std::vector<float> ref(size_t(vocab), 0);
        logits_comparison metrics;
        std::vector<llama_token> produced;
        if (reference) { generated_files[logits_path.lexically_normal().string()] = "reference logits retained; no byte-exact comparison permits deletion"; }
        const auto begin = std::chrono::steady_clock::now();
        for (int step = 0; step < n; ++step) {
            const auto token = step == 0 && first >= 0 ? first : greedy();
            require(decode(&token, 1, int(history.size())) == 0, "generation decode failed at position " + std::to_string(history.size()));
            history.push_back(token); produced.push_back(token);
            const auto logits = llama_get_logits_ith(ctx, -1);
            require(logits != nullptr, "missing comparison logits");
            if (reference) {
                for (int i = 0; i < vocab; ++i) { require(std::isfinite(logits[i]), "nonfinite reference logits"); }
                output.write(reinterpret_cast<const char *>(logits), size_t(vocab)*sizeof(float));
                require(bool(output), "reference logits disk write failed");
            } else {
                comparison.read(reinterpret_cast<char *>(ref.data()), ref.size()*sizeof(float));
                require(bool(comparison), "reference logits disk read failed");
                metrics.add(logits, ref);
            }
            if ((step+1) % 64 == 0) { std::fprintf(stderr, "capacity: generation %d/%d occupancy=%zu\n", step+1, n, history.size()); }
        }
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        json result = {{"tokens", produced}, {"count", n}, {"elapsed_s", elapsed}, {"final_occupancy", history.size()}};
        if (!reference) { result["logits_replay"] = metrics.result(); }
        return result;
    }

    void reject_excess() {
        save(file("capacity-before.bin"));
        const auto token = input.front();
        const int rc = decode(&token, 1, c);
        save(file("capacity-after.bin"));
        auto equality = compare_files(file("capacity-before.bin"), file("capacity-after.bin"));
        const bool position = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == c-1;
        report["capacity_rejection"] = {{"decode_rc", rc}, {"state", equality}, {"position_unchanged", position},
                                        {"pass", rc != 0 && equality["equal"].get<bool>() && position}};
        publish();
        require(report["capacity_rejection"]["pass"].get<bool>(), "C+1 rejection mutated destination or succeeded");
        discard_verified(file("capacity-before.bin"), equality, "C+1 rejection comparison closed; retain capacity-after.bin as the representative C-state");
        generated_files[file("capacity-after.bin").lexically_normal().string()] = "representative decoded-capacity C state after byte-exact C+1 rejection proof";
    }

    void prefill() {
        std::vector<int> checkpoints;
        const bool boundary_requested = opt.mode == "all" || opt.mode == "boundary";
        if (p && (boundary_requested || !opt.discard_verified_scratch)) { checkpoints = {p-129, p-128, p-1, p, p+1, p+128}; }
        checkpoints.push_back(prefix);
        std::sort(checkpoints.begin(), checkpoints.end());
        checkpoints.erase(std::unique(checkpoints.begin(), checkpoints.end()), checkpoints.end());
        if (!opt.resume.empty()) {
            require(restore(opt.resume), "resume restore refused");
            require(history.size() == size_t(prefix), "resume snapshot must occupy exactly C-generation positions");
            require(std::equal(history.begin(), history.end(), input.begin()), "resume token prefix differs from token file");
        }
        const auto begin = std::chrono::steady_clock::now();
        while (history.size() < size_t(prefix)) {
            const int position = int(history.size());
            int end = std::min(prefix, position + params.n_batch);
            for (int checkpoint : checkpoints) { if (checkpoint > position) { end = std::min(end, checkpoint); } }
            require(decode(input.data()+position, end-position, position) == 0, "prefill failed at position " + std::to_string(position));
            history.insert(history.end(), input.begin()+position, input.begin()+end);
            if (std::find(checkpoints.begin(), checkpoints.end(), end) != checkpoints.end()) {
                save(file("prefix-" + std::to_string(end) + ".bin"));
            }
            if (end % 4096 == 0 || end == prefix) { std::fprintf(stderr, "capacity: prefill %d/%d\n", end, prefix); }
        }
        // A resumed prefix needs its own persistent file, too.
        save(file("prefix-" + std::to_string(prefix) + ".bin"));
        report["prefill"] = {{"occupancy", prefix}, {"elapsed_s", std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()},
                             {"resumed", !opt.resume.empty()}, {"memory", memory()}};
        publish();
    }

    void boundary() {
        require(p > 128, "boundary probes require a position-split context");
        require(opt.resume.empty(), "boundary mode requires its own prefill checkpoints; resume cannot prove near-P cases");
        report["boundary_replays"] = json::array();
        const uint32_t rollback_reserve = llama_n_rs_seq(ctx);
        report["boundary_rollback_reserve_rows"] = rollback_reserve;
        report["boundary_scratch_policy"] = "source checkpoints retained once per occupancy; boundary-restored.bin, boundary-control.bin, boundary-replay.bin, and logits scratch files are overwritten sequentially";
        for (int at : {p-129, p-128, p-1, p, p+1, p+128}) {
            const auto source = file("prefix-" + std::to_string(at) + ".bin");
            const auto divergent = llama_token((uint64_t(input.at(size_t(at)))+1)%uint64_t(vocab));
            json probe = {{"occupancy", at}, {"divergent_token", divergent}, {"decoded_tokens_per_replay", 4}};
            require(restore(source), "same-P boundary restore failed");
            save(file("boundary-control.bin"));
            probe["same_layout_state"] = compare_files(source, file("boundary-control.bin"));
            require(probe["same_layout_state"]["equal"].get<bool>(), "boundary restore bytes differ; investigate serialization metadata before claiming corruption");
            const auto control = generate(4, divergent, file("boundary-logits.f32"), true);
            save(file("boundary-control.bin"));
            const llama_pos position_min_before_remove = attention_position_min();
            const llama_pos position_max_before_remove = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
            const bool removed = llama_memory_seq_rm(llama_get_memory(ctx), 0, at, -1);
            const bool removal_within_reserve = 4 <= rollback_reserve;
            probe["removal"] = {{"seq_rm_from_position", at}, {"requested_tokens", 4},
                {"rollback_reserve_rows", rollback_reserve}, {"within_reserve", removal_within_reserve},
                {"accepted", removed}, {"position_min_before", position_min_before_remove},
                {"position_max_before", position_max_before_remove}};
            if (!removed) {
                // Refusal is valid outside the recurrent rollback reserve only
                // when the destination is bytewise intact and a FULL source
                // restore/replay remains usable.
                save(file("boundary-restored.bin"));
                probe["removal"]["rejection_intact_state"] =
                    compare_files(file("boundary-control.bin"), file("boundary-restored.bin"));
                probe["removal"]["position_min_after"] = attention_position_min();
                probe["removal"]["position_max_after"] = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
                probe["removal"]["position_unchanged"] =
                    probe["removal"]["position_min_after"].get<int64_t>() == position_min_before_remove &&
                    probe["removal"]["position_max_after"].get<int64_t>() == position_max_before_remove;
                require(restore(source), "full restore after refused boundary removal failed");
                save(file("boundary-restored.bin"));
                probe["removal"]["full_restore_state"] = compare_files(source, file("boundary-restored.bin"));
                probe["removal"]["pass"] = !removal_within_reserve &&
                    probe["removal"]["rejection_intact_state"]["equal"].get<bool>() &&
                    probe["removal"]["position_unchanged"].get<bool>() &&
                    probe["removal"]["full_restore_state"]["equal"].get<bool>();
                if (removal_within_reserve) {
                    probe["removal"]["reason"] = "seq_rm refused despite requested suffix being within the reported recurrent rollback reserve";
                }
            } else {
                history.resize(size_t(at));
                probe["removal"]["position_min_after"] = attention_position_min();
                probe["removal"]["position_max_after"] = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
                save(file("boundary-restored.bin"));
                probe["removal"]["native_state"] = compare_files(source, file("boundary-restored.bin"));
                probe["removal"]["position_matches"] =
                    probe["removal"]["position_min_after"].get<int64_t>() == 0 &&
                    probe["removal"]["position_max_after"].get<int64_t>() == at-1;
                probe["removal"]["pass"] = removal_within_reserve &&
                    probe["removal"]["native_state"]["equal"].get<bool>() &&
                    probe["removal"]["position_matches"].get<bool>();
                if (probe["removal"]["pass"].get<bool>()) {
                    const auto removal_next = generate(1, divergent, file("boundary-removal-logits.f32"), true);
                    save(file("boundary-removal-control.bin"));
                    require(restore(source), "restore for within-reserve removal replay failed");
                    const auto removal_replay = generate(1, divergent, file("boundary-removal-logits.f32"), false);
                    save(file("boundary-removal-replay.bin"));
                    probe["removal"]["next_vocab"] = removal_replay["logits_replay"];
                    probe["removal"]["next_vocab_exact"] =
                        removal_replay["logits_replay"]["unequal_values"].get<uint64_t>() == 0 &&
                        removal_replay["logits_replay"]["max_abs"].get<double>() == 0.0;
                    probe["removal"]["next_tokens_equal"] = removal_next["tokens"] == removal_replay["tokens"];
                    probe["removal"]["next_native_state"] = compare_files(
                        file("boundary-removal-control.bin"), file("boundary-removal-replay.bin"));
                    probe["removal"]["pass"] = probe["removal"]["next_vocab_exact"].get<bool>() &&
                        probe["removal"]["next_tokens_equal"].get<bool>() &&
                        probe["removal"]["next_native_state"]["equal"].get<bool>();
                }
                if (!removal_within_reserve) {
                    probe["removal"]["pass"] = false;
                    probe["removal"]["reason"] = "seq_rm accepted a suffix longer than the reported recurrent rollback reserve";
                }
            }
            report["boundary_replays"].push_back(probe); publish();
            require(probe["removal"]["pass"].get<bool>(),
                    "boundary removal was not reserve-safe or failed its intact-state proof");
            require(restore(source), "second same-P boundary restore failed");
            const auto replay = generate(4, divergent, file("boundary-logits.f32"), false);
            save(file("boundary-replay.bin"));
            probe["tokens_equal"] = control["tokens"] == replay["tokens"];
            probe["final_state"] = compare_files(file("boundary-control.bin"), file("boundary-replay.bin"));
            probe["logits"] = replay["logits_replay"];
            probe["pass"] = probe["tokens_equal"].get<bool>() && probe["final_state"]["equal"].get<bool>() && probe["logits"]["pass"].get<bool>();
            report["boundary_replays"].back() = probe; publish();
            require(probe["pass"].get<bool>(), "near-P divergent restore replay failed");
            json removal_proof = probe["removal"].value("accepted", false)
                ? probe["removal"].value("native_state", json::object())
                : probe["removal"].value("full_restore_state", json::object());
            discard_verified(file("boundary-restored.bin"), removal_proof, "boundary removal and restoration check closed");
            discard_verified(file("boundary-control.bin"), probe["final_state"], "same-occupancy control/replay comparison closed");
            discard_verified(file("boundary-replay.bin"), probe["final_state"], "same-occupancy control/replay comparison closed");
            if (at != p-1 && at != prefix) {
                discard_verified(source, probe["same_layout_state"], "all boundary probes for this source occupancy are closed");
            }
            if (probe["removal"].value("accepted", false)) {
                const auto next_proof = probe["removal"].value("next_native_state", json::object());
                discard_verified(file("boundary-removal-control.bin"), next_proof, "within-reserve removal replay closed");
                discard_verified(file("boundary-removal-replay.bin"), next_proof, "within-reserve removal replay closed");
            }
        }

        // When the context reports a nonzero recurrent rollback reserve, probe
        // a one-token suffix removal at P-1. This is the true within-reserve
        // case for models whose reserve is smaller than the four-token probes.
        const int within_at = p-1;
        const auto within_source = file("prefix-" + std::to_string(within_at) + ".bin");
        const auto within_token = llama_token((uint64_t(input.at(size_t(within_at)))+1)%uint64_t(vocab));
        json within = {{"occupancy", within_at}, {"rollback_reserve_rows", rollback_reserve},
                       {"requested_removal_tokens", 1}, {"within_reserve", rollback_reserve > 0}};
        if (rollback_reserve == 0) {
            within["status"] = "not_applicable_zero_reported_reserve";
            within["pass"] = true; // capability is absent and is explicitly reported
        } else {
            require(restore(within_source), "within-reserve source restore failed");
            const auto control = generate(1, within_token, file("boundary-within-logits.f32"), true);
            save(file("boundary-within-control.bin"));
            const bool removed = llama_memory_seq_rm(llama_get_memory(ctx), 0, within_at, -1);
            within["accepted"] = removed;
            if (!removed) {
                save(file("boundary-restored.bin"));
                within["rejection_intact_state"] = compare_files(
                    file("boundary-within-control.bin"), file("boundary-restored.bin"));
                within["position_unchanged"] =
                    llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == within_at;
                require(restore(within_source), "full restore after within-reserve refusal failed");
                save(file("boundary-restored.bin"));
                within["full_restore_state"] = compare_files(within_source, file("boundary-restored.bin"));
                within["status"] = "rejected_intact_despite_reported_reserve";
                within["pass"] = false;
            } else {
                history.resize(size_t(within_at));
                within["position_min"] = attention_position_min();
                within["position_max"] = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
                save(file("boundary-restored.bin"));
                within["removed_state"] = compare_files(within_source, file("boundary-restored.bin"));
                within["position_matches"] = within["position_min"].get<int64_t>() == 0 &&
                    within["position_max"].get<int64_t>() == within_at-1;
                const bool state_ok = within["removed_state"]["equal"].get<bool>() &&
                    within["position_matches"].get<bool>();
                if (state_ok) {
                    const auto next_control = generate(1, within_token,
                            file("boundary-within-next-logits.f32"), true);
                    save(file("boundary-within-next-control.bin"));
                    require(restore(within_source), "within-reserve next-token replay restore failed");
                    const auto next_replay = generate(1, within_token,
                            file("boundary-within-next-logits.f32"), false);
                    save(file("boundary-within-next-replay.bin"));
                    within["next_tokens_equal"] = next_control["tokens"] == next_replay["tokens"];
                    within["next_vocab"] = next_replay["logits_replay"];
                    within["next_vocab_exact"] = next_replay["logits_replay"]["unequal_values"].get<uint64_t>() == 0 &&
                        next_replay["logits_replay"]["max_abs"].get<double>() == 0.0;
                    within["next_native_state"] = compare_files(file("boundary-within-next-control.bin"),
                                                                  file("boundary-within-next-replay.bin"));
                    within["pass"] = within["next_tokens_equal"].get<bool>() &&
                        within["next_vocab_exact"].get<bool>() && within["next_native_state"]["equal"].get<bool>();
                } else {
                    within["pass"] = false;
                }
                within["status"] = "accepted_exactly_replayed";
            }
        }
        report["boundary_within_reserve_removal"] = within;
        publish();
        require(within["pass"].get<bool>(), "within-reserve boundary removal/replay failed");
        if (within.value("accepted", false)) {
            const auto next_proof = within.value("next_native_state", json::object());
            discard_verified(file("boundary-within-next-control.bin"), next_proof, "one-token within-reserve replay closed");
            discard_verified(file("boundary-within-next-replay.bin"), next_proof, "one-token within-reserve replay closed");
        }

        // Exercise one continuous 129-token decode crossing the placement edge:
        // decode positions P-1 through P+127, ending at occupied extent P+128.
        // The first token is the same explicit deterministic divergence used by
        // the shorter checkpoint probes; all remaining tokens use this runner's
        // greedy sampler and EOG never stops the loop.
        const int crossing_start = p-1;
        const int crossing_tokens = 129;
        const int crossing_end_occupancy = p+128;
        const auto source = file("prefix-" + std::to_string(crossing_start) + ".bin");
        const auto divergent = llama_token((uint64_t(input.at(size_t(crossing_start)))+1)%uint64_t(vocab));
        json crossing = {{"source_occupancy", crossing_start}, {"first_position", crossing_start},
                         {"last_position", crossing_end_occupancy-1}, {"decode_tokens", crossing_tokens},
                         {"expected_final_occupancy", crossing_end_occupancy}, {"divergent_first_token", divergent},
                         {"eog_stops_generation", false}};
        require(restore(source), "P-1 crossing source restore failed");
        save(file("boundary-crossing-control.bin"));
        crossing["source_restore_state"] = compare_files(source, file("boundary-crossing-control.bin"));
        require(crossing["source_restore_state"]["equal"].get<bool>(), "P-1 crossing source restore bytes differ");
        const auto control = generate(crossing_tokens, divergent, file("boundary-crossing-logits.f32"), true);
        save(file("boundary-crossing-control.bin"));
        crossing["reference_generation"] = control;
        crossing["reference_final_occupancy_matches"] = history.size() == size_t(crossing_end_occupancy) &&
            llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == crossing_end_occupancy-1;
        require(crossing["reference_final_occupancy_matches"].get<bool>(),
                "P-1 crossing reference did not end at P+128 occupancy");

        require(restore(source), "P-1 crossing replay restore failed");
        const auto replay = generate(crossing_tokens, divergent, file("boundary-crossing-logits.f32"), false);
        save(file("boundary-crossing-replay.bin"));
        crossing["replay_generation"] = replay;
        crossing["tokens_equal"] = control["tokens"] == replay["tokens"];
        crossing["full_history_final_native_state"] = compare_files(
                file("boundary-crossing-control.bin"), file("boundary-crossing-replay.bin"));
        crossing["logits"] = replay["logits_replay"];
        crossing["replay_final_occupancy_matches"] = history.size() == size_t(crossing_end_occupancy) &&
            llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == crossing_end_occupancy-1;
        crossing["full_history_pass"] = crossing["tokens_equal"].get<bool>() &&
            crossing["full_history_final_native_state"]["equal"].get<bool>() &&
            crossing["logits"]["pass"].get<bool>() &&
            crossing["replay_final_occupancy_matches"].get<bool>();
        report["boundary_crossing"] = crossing; publish();
        require(crossing["full_history_pass"].get<bool>(),
                "129-token P-boundary full-history replay failed");

        const llama_token overflow_probe_token = input.at(size_t(crossing_end_occupancy));
        crossing["overflow_only_probe_token"] = overflow_probe_token;
        // Try to trim away the occupied local [0,P) prefix. Recurrent memory
        // may reject this when P lies outside its exact rollback reserve. A
        // rejection is accepted only with byte-exact intact-state proof plus a
        // full restore and next-vocabulary replay. The empty-local attention
        // path is marked not executed in that case; it is never claimed as a
        // pass merely because the API failed closed.
        const llama_pos full_pos_min = attention_position_min();
        const llama_pos full_pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
        const bool empty_local_removed = llama_memory_seq_rm(llama_get_memory(ctx), 0, 0, p);
        json empty_local = {{"seq_rm_begin", 0}, {"seq_rm_end", p},
            {"rollback_reserve_rows", rollback_reserve}, {"reserve_sufficient_for_prefix_remove", p <= int(rollback_reserve)},
            {"accepted", empty_local_removed}, {"position_min_before", full_pos_min},
            {"position_max_before", full_pos_max}, {"logical_history_tokens", history.size()}};
        if (!empty_local_removed) {
            save(file("boundary-restored.bin"));
            empty_local["rejection_intact_state"] = compare_files(
                file("boundary-crossing-replay.bin"), file("boundary-restored.bin"));
            empty_local["position_min_after"] = attention_position_min();
            empty_local["position_max_after"] = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
            empty_local["position_unchanged"] =
                empty_local["position_min_after"].get<int64_t>() == full_pos_min &&
                empty_local["position_max_after"].get<int64_t>() == full_pos_max;
            empty_local["attention_path_executed"] = false;
            empty_local["attention_path_status"] = "not_executed_recurrent_seq_rm_rejected";
            require(empty_local["rejection_intact_state"]["equal"].get<bool>() &&
                    empty_local["position_unchanged"].get<bool>(),
                    "rejected prefix removal mutated native state or shifted positions");

            // Full restore after rejection, then compare the next complete
            // vocabulary distribution and resulting native state against a
            // second replay from the same restored checkpoint.
            require(restore(file("boundary-crossing-replay.bin")),
                    "full restore after rejected local-prefix removal failed");
            save(file("boundary-restored.bin"));
            empty_local["full_restore_state"] = compare_files(
                file("boundary-crossing-replay.bin"), file("boundary-restored.bin"));
            const auto next_control = generate(1, overflow_probe_token,
                    file("boundary-overflow-next.logits"), true);
            save(file("boundary-crossing-control.bin")); // full reference checkpoint no longer needed
            require(restore(file("boundary-crossing-replay.bin")),
                    "second full restore after rejected local-prefix removal failed");
            const auto next_replay = generate(1, overflow_probe_token,
                    file("boundary-overflow-next.logits"), false);
            save(file("boundary-crossing-replay.bin")); // full replay checkpoint no longer needed
            empty_local["next_tokens_equal"] = next_control["tokens"] == next_replay["tokens"];
            empty_local["next_vocab"] = next_replay["logits_replay"];
            empty_local["next_vocab_exact"] =
                next_replay["logits_replay"]["unequal_values"].get<uint64_t>() == 0 &&
                next_replay["logits_replay"]["max_abs"].get<double>() == 0.0;
            empty_local["post_probe_native_state"] = compare_files(
                file("boundary-crossing-control.bin"), file("boundary-crossing-replay.bin"));
            empty_local["next_position_matches"] =
                attention_position_min() == 0 &&
                llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == crossing_end_occupancy;
            empty_local["pass"] = empty_local["reserve_sufficient_for_prefix_remove"] == false &&
                empty_local["full_restore_state"]["equal"].get<bool>() &&
                empty_local["next_tokens_equal"].get<bool>() && empty_local["next_vocab_exact"].get<bool>() &&
                empty_local["post_probe_native_state"]["equal"].get<bool>() &&
                empty_local["next_position_matches"].get<bool>();
            empty_local["status"] = empty_local["pass"].get<bool>()
                ? "api_rejected_outside_reserve_intact_full_restore_replay_passed"
                : "api_rejection_or_full_restore_replay_check_failed";
        } else {
            empty_local["position_min_after"] = attention_position_min();
            empty_local["position_max_after"] = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
            empty_local["no_position_shift"] = empty_local["position_min_after"].get<int64_t>() == p &&
                empty_local["position_max_after"].get<int64_t>() == crossing_end_occupancy-1 &&
                history.size() == size_t(crossing_end_occupancy);
            if (empty_local["no_position_shift"].get<bool>()) {
                // Save the replay's sparse overflow-only state, then construct
                // the same sparse state from the independent reference history.
                save(file("boundary-crossing-replay.bin"));
                require(restore(file("boundary-crossing-control.bin")),
                        "restore full reference crossing state for empty-local comparison failed");
                const bool ref_removed = llama_memory_seq_rm(llama_get_memory(ctx), 0, 0, p);
                empty_local["reference_trim_accepted"] = ref_removed;
                if (ref_removed) {
                    save(file("boundary-crossing-control.bin"));
                    empty_local["native_state"] = compare_files(
                        file("boundary-crossing-control.bin"), file("boundary-crossing-replay.bin"));
                    empty_local["reference_position_min"] = attention_position_min();
                    empty_local["reference_position_max"] = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
                    require(restore(file("boundary-crossing-control.bin")),
                            "restore reference sparse overflow-only state failed");
                    empty_local["sparse_restore_positions_match"] =
                        attention_position_min() == p &&
                        llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == crossing_end_occupancy-1;
                    const auto next_control = generate(1, overflow_probe_token,
                            file("boundary-overflow-next.logits"), true);
                    save(file("boundary-crossing-control.bin")); // overwrite sparse pre-probe state
                    require(restore(file("boundary-crossing-replay.bin")),
                            "restore replay sparse overflow-only state failed");
                    empty_local["replay_sparse_position_min"] = attention_position_min();
                    empty_local["replay_sparse_position_max"] = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
                    empty_local["replay_sparse_positions_match"] =
                        empty_local["replay_sparse_position_min"].get<int64_t>() == p &&
                        empty_local["replay_sparse_position_max"].get<int64_t>() == crossing_end_occupancy-1;
                    const auto next_replay = generate(1, overflow_probe_token,
                            file("boundary-overflow-next.logits"), false);
                    save(file("boundary-crossing-replay.bin")); // overwrite sparse pre-probe state
                    empty_local["next_tokens_equal"] = next_control["tokens"] == next_replay["tokens"];
                    empty_local["next_vocab"] = next_replay["logits_replay"];
                    empty_local["next_vocab_exact"] =
                        next_replay["logits_replay"]["unequal_values"].get<uint64_t>() == 0 &&
                        next_replay["logits_replay"]["max_abs"].get<double>() == 0.0;
                    empty_local["post_probe_native_state"] = compare_files(
                        file("boundary-crossing-control.bin"), file("boundary-crossing-replay.bin"));
                    empty_local["next_position_matches"] =
                        attention_position_min() == p &&
                        llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == crossing_end_occupancy;
                    empty_local["attention_path_executed"] = true;
                    empty_local["pass"] = empty_local["reserve_sufficient_for_prefix_remove"].get<bool>() &&
                        empty_local["no_position_shift"].get<bool>() &&
                        empty_local["reference_trim_accepted"].get<bool>() &&
                        empty_local["native_state"]["equal"].get<bool>() &&
                        empty_local["reference_position_min"].get<int64_t>() == p &&
                        empty_local["reference_position_max"].get<int64_t>() == crossing_end_occupancy-1 &&
                        empty_local["sparse_restore_positions_match"].get<bool>() &&
                        empty_local["replay_sparse_positions_match"].get<bool>() &&
                        empty_local["next_tokens_equal"].get<bool>() && empty_local["next_vocab_exact"].get<bool>() &&
                        empty_local["post_probe_native_state"]["equal"].get<bool>() &&
                        empty_local["next_position_matches"].get<bool>();
                    empty_local["status"] = "empty_local_overflow_replay_executed";
                } else {
                    empty_local["attention_path_executed"] = false;
                    empty_local["pass"] = false;
                    empty_local["status"] = "reference_trim_refused_after_replay_trim_accepted";
                }
            } else {
                empty_local["attention_path_executed"] = false;
                empty_local["pass"] = false;
                empty_local["status"] = "accepted_removal_shifted_positions_or_history";
            }
        }
        crossing["mandatory_api_rejection_proof"] = empty_local["pass"];
        crossing["empty_local_overflow_attention"] = empty_local;
        crossing["empty_local_attention_executed"] = empty_local.value("attention_path_executed", false);
        crossing["all_requested_checks_executed"] = crossing["empty_local_attention_executed"].get<bool>();
        crossing["pass"] = crossing["full_history_pass"].get<bool>() && empty_local["pass"].get<bool>();
        crossing["mandatory_overflow_api_check_pass"] = empty_local["pass"];
        report["boundary_crossing"] = crossing; publish();
        report["all_requested_checks_executed"] = crossing["all_requested_checks_executed"];
        publish();
        require(crossing["pass"].get<bool>(), "P-prefix removal/full-restore/replay contract failed");
        const auto post_probe_proof = empty_local.value("post_probe_native_state", json::object());
        discard_verified(file("boundary-crossing-control.bin"), post_probe_proof, "129-token crossing and overflow probe closed");
        discard_verified(file("boundary-crossing-replay.bin"), post_probe_proof, "129-token crossing and overflow probe closed");
        if (empty_local.contains("full_restore_state")) {
            discard_verified(file("boundary-restored.bin"), empty_local["full_restore_state"], "rejected local-prefix trim restored exactly");
        }
        discard_verified(source, crossing["source_restore_state"], "P-1 checkpoint no longer needed after crossing completes");
        for (const auto & name : {"boundary-removal-logits.f32", "boundary-within-logits.f32", "boundary-within-next-logits.f32", "boundary-overflow-next.logits"}) {
            const auto path = file(name);
            if (fs::exists(path)) { generated_files[path.lexically_normal().string()] = "logit evidence retained; no byte-exact file comparison authorized deletion"; }
        }
    }

    void cycles() {
        report["cycles"] = json::array();
        json baseline_memory, baseline_generation;
        const auto source = file("prefix-" + std::to_string(prefix) + ".bin");
        // In a full campaign force divergence from the actual capacity run.
        // Standalone cycles use the corpus continuation as the recorded source.
        const auto original = report.contains("capacity_generation")
                ? report["capacity_generation"]["tokens"][0].get<llama_token>()
                : input.at(size_t(prefix));
        const auto divergent = llama_token((uint64_t(original)+1)%uint64_t(vocab));
        report["cycle_divergence"] = {{"original_next_token", original}, {"forced_next_token", divergent},
                                      {"source", report.contains("capacity_generation") ? "actual capacity generation" : "corpus continuation"}};
        for (int cycle = 1; cycle <= opt.cycles; ++cycle) {
            require(restore(source), "cycle prefix restore failed");
            save(file("cycle-restored.bin"));
            auto restored = compare_files(source, file("cycle-restored.bin"));
            require(restored["equal"].get<bool>(), "cycle restored state bytes differ; investigate serialization metadata before claiming corruption");
            auto generation = generate(opt.generate, divergent, file("cycle-reference-logits.f32"), cycle == 1);
            save(file(cycle == 1 ? "cycle-reference.bin" : "cycle-final.bin"));
            auto mem = memory();
            json entry = {{"cycle", cycle}, {"restored_prefix", restored}, {"generation", generation}, {"memory", mem}};
            if (cycle == 1) {
                baseline_memory = mem; baseline_generation = generation;
                entry["reference"] = true;
            } else {
                entry["tokens_equal"] = generation["tokens"] == baseline_generation["tokens"];
                entry["final_state"] = compare_files(file("cycle-reference.bin"), file("cycle-final.bin"));
                entry["pass"] = entry["tokens_equal"].get<bool>() && entry["final_state"]["equal"].get<bool>() && generation["logits_replay"]["pass"].get<bool>();
            }
            const int64_t completed_unix_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
            entry["completed_unix_ms"] = completed_unix_ms;
            entry["elapsed_since_campaign_start_s"] = std::chrono::duration<double>(
                    std::chrono::steady_clock::now()-campaign_started).count();
            entry["cycle_hold_ms"] = opt.cycle_hold_ms;
            report["cycles"].push_back(entry); publish();
            if (cycle > 1) { require(entry["pass"].get<bool>(), "cycle token/logit/native-state replay failed"); }
            discard_verified(file("cycle-restored.bin"), entry["restored_prefix"], "cycle prefix restoration comparison closed");
            if (cycle > 1 && cycle < opt.cycles) {
                discard_verified(file("cycle-final.bin"), entry["final_state"], "nonfinal cycle exact replay completed");
            }
            std::fprintf(stderr, "capacity: completed restore/divergence cycle %d/%d unix_ms=%lld hold_ms=%d\n",
                    cycle, opt.cycles, static_cast<long long>(completed_unix_ms), opt.cycle_hold_ms);
            std::fflush(stderr);
            if (opt.cycle_hold_ms > 0) {
                // Explicit campaign instrumentation: let external per-PID
                // NVML/DRM collectors observe retained allocations at rest.
                // Includes the final cycle; no serving code is affected.
                llama_synchronize(ctx);
                std::this_thread::sleep_for(std::chrono::milliseconds(opt.cycle_hold_ms));
            }
        }
        const auto final = memory();
        json growth = json::object();
        bool accounted_pass = true;
        for (const auto & entry : final["llama_accounted_bytes"].items()) {
            const auto key = entry.key();
            const auto before = baseline_memory["llama_accounted_bytes"].contains(key) ? baseline_memory["llama_accounted_bytes"][key]["total"].get<int64_t>() : 0;
            const auto delta = entry.value()["total"].get<int64_t>() - before;
            growth[key] = {{"retained_growth_bytes", delta}, {"within_64_mib", delta <= 64*1024*1024}};
            accounted_pass = accounted_pass && delta <= 64*1024*1024;
        }
        report["retained_memory"] = {{"baseline_cycle", 1}, {"final_cycle", opt.cycles}, {"accounted_by_backend", growth},
                                     {"accounted_pass", accounted_pass},
                                     {"allocator_pool_gate", "requires external allocator/device retained-memory evidence; llama breakdown excludes unreported backend pools"}};
        publish();
        require(accounted_pass, "accounted retained backend memory grew by more than 64 MiB");
    }

    void mismatch() {
        require(p > 256, "P-mismatch probe requires P>256");
        auto destination = params;
        destination.kv_position_split_p = uint32_t(p-256);
        llama_synchronize(ctx);
        init->release_context(); ctx = nullptr;
        require(init->recreate_context(destination), "cannot create P-mismatch destination");
        ctx = init->context();
        history.clear();
        require(decode(input.data(), 32, 0) == 0, "P-mismatch sentinel prefill failed");
        history.assign(input.begin(), input.begin()+32);
        save(file("mismatch-before.bin"));
        const bool accepted = restore(file("prefix-" + std::to_string(prefix) + ".bin"));
        save(file("mismatch-after.bin"));
        const auto equality = compare_files(file("mismatch-before.bin"), file("mismatch-after.bin"));
        const bool position = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == 31;
        report["p_mismatch"] = {{"source_p", p}, {"destination_p", p-256}, {"accepted", accepted},
                                {"state", equality}, {"position_unchanged", position},
                                {"pass", !accepted && equality["equal"].get<bool>() && position}};
        publish();
        require(report["p_mismatch"]["pass"].get<bool>(), "P-mismatch restore was accepted or mutated destination");
        discard_verified(file("mismatch-before.bin"), equality, "P-mismatch rejection comparison closed");
    }

    void empty_local() {
        require(opt.resume.empty(), "empty-local cannot use an external resume snapshot");
        const char * reference = std::getenv("LLAMA_OVF_ATTN_REFERENCE_CPU");
        require(!reference || std::strcmp(reference, "1") != 0, "empty-local requires GPU attention, without CPU reference override");
        auto * hybrid = dynamic_cast<llama_memory_hybrid *>(llama_get_memory(ctx));
        auto * attn = hybrid ? dynamic_cast<llama_kv_cache_kvarn *>(hybrid->get_mem_attn_base()) : nullptr;
        require(attn && attn->position_split_enabled(), "empty-local requires a split KVarN hybrid context");
        for (const auto * name : {"empty-local-rs-before.bin", "empty-local-rs-after.bin", "empty-local-A.bin",
                "empty-local-B.bin", "empty-local-control.bin", "empty-local-replay.bin", "empty-local-logits.f32"}) {
            require(!fs::exists(file(name)), "empty-local refuses to overwrite prior evidence: " + file(name).string());
        }
        report["production_dimensions"] = false;
        report["remaining_evidence"] = {"this isolated fixture does not qualify production hybrid prefix removal or capacity/quality/performance gates"};
        report["fixture_scope"] = "attention-only clear while preserving recurrent checkpoint; does not authorize/support production hybrid prefix removal";
        auto & proof = report["empty_local_fixture"];
        proof = {{"fixture_scope", report["fixture_scope"]},
            {"prefill_tokens", p}, {"fixed_token_at_p", input.at(size_t(p))}, {"replay_tokens_per_arm", 4},
            {"cpu_reference", false}, {"sparse_restores", json::array()}, {"pass", false}};
        report["configuration"]["inactive_capacity_options"] = {{"generate", opt.generate}, {"cycles", opt.cycles}, {"cycle_hold_ms", opt.cycle_hold_ms}};
        report["configuration"]["capacity_mode"] = "empty-local";
        publish();
        for (int at = 0; at < p;) {
            const int count = std::min(params.n_batch, p-at);
            require(decode(input.data()+at, count, at) == 0, "empty-local prefill failed");
            history.insert(history.end(), input.begin()+at, input.begin()+at+count); at += count;
        }
        require(attn->seq_pos_min(0) == 0 && attn->seq_pos_max(0) == p-1 &&
                hybrid->get_mem_recr()->seq_pos_max(0) == p-1, "exactP attention/recurrent prefill positions differ");
        const auto save_recurrent = [&](const char * name) {
            llama_synchronize(ctx);
            bounded_state_writer writer(file(name)); hybrid->get_mem_recr()->state_write(writer, 0, 0);
            require(writer.n_bytes() > 0, "recurrent checkpoint is empty"); writer.finish();
            generated_files[file(name).lexically_normal().string()] = "attention-only fixture recurrent checkpoint";
            return file_observation(file(name));
        };
        proof["recurrent_before"] = save_recurrent("empty-local-rs-before.bin");
        proof["direct_attention_clear_accepted"] = attn->seq_rm(0, 0, -1);
        proof["recurrent_after"] = save_recurrent("empty-local-rs-after.bin");
        proof["recurrent_bytes"] = compare_files(file("empty-local-rs-before.bin"), file("empty-local-rs-after.bin"));
        proof["attention_empty_after_clear"] = attn->seq_pos_min(0) == -1 && attn->seq_pos_max(0) == -1 &&
                attn->get_metadata_cache()->get_cells(0).get_used() == 0;
        publish();
        require(proof["direct_attention_clear_accepted"].get<bool>() && proof["recurrent_bytes"]["equal"].get<bool>() &&
                proof["attention_empty_after_clear"].get<bool>(), "attention-only clear mutated recurrent bytes or retained attention cells");
        require(decode(input.data()+p, 1, p) == 0, "empty-local absoluteP decode failed"); history.push_back(input[size_t(p)]);
        proof["absolute_p_argmax"] = greedy(); // Also rejects nonfinite first-step full-vocab logits.
        const auto sparse = [&]() {
            const auto & cells = attn->get_metadata_cache()->get_cells(0);
            require(cells.size() == uint32_t(c), "sparse metadata logical capacity differs from C");
            std::vector<uint32_t> local, overflow;
            require(attn->position_split_live_cells(llama_position_split::range_kind::LOCAL, local) &&
                    attn->position_split_live_cells(llama_position_split::range_kind::OVERFLOW, overflow), "sparse range mapping unavailable");
            bool holes = true;
            for (int i = 0; i < p; ++i) { holes = holes && cells.is_empty(uint32_t(i)); }
            const auto * standard = attn->get_standard_cache();
            const bool ok = holes && local.empty() && overflow == std::vector<uint32_t>{uint32_t(p)} && cells.get_used() == 1 &&
                cells.seq_has(uint32_t(p), 0) && cells.pos_get(uint32_t(p)) == p && attn->seq_pos_min(0) == p && attn->seq_pos_max(0) == p &&
                standard && standard->get_payload_row_origin() == uint32_t(p) && standard->get_payload_size() == uint32_t(c-p) &&
                standard->payload_row_for_cell(uint32_t(p)) == 0 && history.size() == size_t(p+1);
            require(ok, "empty-local sparse globalCID/physicalrow/history invariant failed");
            return json{{"pass", ok}, {"local_holes", p}, {"local_live_cells", local.size()}, {"attention_live_cells", cells.get_used()},
                {"attention_min", attn->seq_pos_min(0)}, {"attention_max", attn->seq_pos_max(0)}, {"global_overflow_cell", p},
                {"physical_overflow_row", standard->payload_row_for_cell(uint32_t(p))}, {"physical_capacity_rows", standard->get_payload_size()}, {"token_history_count", history.size()}};
        };
        proof["executed_empty_local_decode"] = sparse(); save(file("empty-local-A.bin"));
        proof["source_native_full"] = file_observation(file("empty-local-A.bin")); publish();
        json control;
        for (int arm = 0; arm < 2; ++arm) {
            require(restore(file("empty-local-A.bin")), "sparse FULL restore refused"); sparse(); save(file("empty-local-B.bin"));
            const auto equality = compare_files(file("empty-local-A.bin"), file("empty-local-B.bin"));
            proof["sparse_restores"].push_back(equality); publish(); require(equality["equal"].get<bool>(), "sparse native FULL bytes changed after restore");
            auto generation = generate(4, input.at(size_t(p+1)), file("empty-local-logits.f32"), arm == 0);
            save(file(arm == 0 ? "empty-local-control.bin" : "empty-local-replay.bin"));
            proof[arm == 0 ? "control" : "replay"] = generation; publish();
            if (arm == 0) { control = generation; }
            else {
                proof["tokens_equal"] = generation["tokens"] == control["tokens"];
                proof["full_vocab_exact"] = generation["logits_replay"]["rows"] == 4 && generation["logits_replay"]["unequal_values"] == 0;
                proof["final_native_full"] = compare_files(file("empty-local-control.bin"), file("empty-local-replay.bin")); publish();
                require(proof["tokens_equal"].get<bool>() && proof["full_vocab_exact"].get<bool>() && proof["final_native_full"]["equal"].get<bool>(), "empty-local replay was not exact");
            }
        }
        proof["pass"] = true; report["completed"] = true; report["all_requested_checks_executed"] = true;
        report["mandatory_executed_checks_pass"] = true; report["executed_checks_pass"] = true; report["all_requested_checks_passed"] = true;
        report["production_dimensions"] = false; report["production_gate_complete"] = false;
        report["allocation_outcome"] = "attention-only fixture decodes and FULL restores returned successfully; external driver logs remain required";
        report["remaining_evidence"] = {"this isolated fixture does not qualify production hybrid prefix removal or capacity/quality/performance gates"};
        discard_verified(file("empty-local-rs-after.bin"), proof["recurrent_bytes"], "recurrent preservation comparison closed");
        discard_verified(file("empty-local-B.bin"), proof["sparse_restores"].back(), "sparse FULL restoration comparison closed");
        discard_verified(file("empty-local-replay.bin"), proof["final_native_full"], "empty-local exact replay comparison closed");
        finish_retention_manifest(); publish();
    }

    void run() {
        input = read_tokens(opt);
        require(params.n_parallel == 1, "runner requires parallel=1");
        require(std::all_of(params.speculative.types.begin(), params.speculative.types.end(),
                    [](common_speculative_type type) { return type == COMMON_SPECULATIVE_TYPE_NONE; }),
                "runner requires speculative decoding disabled");
        require(params.n_ctx > opt.generate && opt.generate > 0 && opt.cycles > 0, "invalid capacity/generation/cycle configuration");
        params.fit_params = false;
        init = common_init_from_params(params);
        require(init && init->model() && init->context(), "model/context initialization failed (inspect logs for OOM)");
        ctx = init->context(); c = int(llama_n_ctx(ctx)); p = int(params.kv_position_split_p); prefix = c-opt.generate;
        require(c == params.n_ctx, "effective C differs from requested C");
        const uint32_t p_target = llama_position_split::kTargetP;
        if (opt.mode == "empty-local") { require(p > 0 && p%256 == 0 && p <= int(p_target) && p+5 <= c, "empty-local requires aligned P>0 up to the target and room for five overflow tokens"); }
        else { require(!p || (p%256 == 0 && p <= int(p_target) && p+128 <= prefix), "P must be 256-aligned up to the target and leave room for boundary probes"); }
        vocab = llama_vocab_n_tokens(llama_model_get_vocab(init->model()));
        if (opt.mode == "empty-local") { require(input.size() > size_t(p+1), "empty-local requires P+2 input tokens"); }
        else { require(input.size() > size_t(prefix), "token file needs at least C-generation+1 tokens (one continuation for divergence)"); }
        for (auto token : input) { require(token >= 0 && token < vocab, "token outside model vocabulary"); }
        batch = llama_batch_init(params.n_batch, 0, 1);
        report["configuration"] = {{"model", params.model.path}, {"commit", llama_commit()}, {"build", llama_build_info()},
                                   {"c", c}, {"p", p}, {"batch", params.n_batch}, {"ubatch", params.n_ubatch},
                                   {"cycle_hold_ms", opt.cycle_hold_ms},
                                   {"recurrent_rollback_rows", llama_n_rs_seq(ctx)}, {"sampling", "greedy; EOG does not stop generation"},
                                   {"token_file", file_observation(opt.tokens)}, {"token_format", opt.format},
                                   {"local_k_bits", params.cache_kvarn_bits_k}, {"local_v_bits", params.cache_kvarn_bits_v},
                                   {"overflow_k", ggml_type_name(params.remote_attn_cache_type_k == GGML_TYPE_COUNT ? params.cache_type_k : params.remote_attn_cache_type_k)},
                                   {"overflow_v", ggml_type_name(params.remote_attn_cache_type_v == GGML_TYPE_COUNT ? params.cache_type_v : params.remote_attn_cache_type_v)}};
        report["initial_memory"] = memory(); publish();
        if (opt.mode == "empty-local") { empty_local(); return; }
        prefill();
        if (opt.mode == "all" || opt.mode == "capacity") {
            // Resume does not restore logits: replay the last prefix token by
            // restoring a prefix with one fewer token would change the fixture.
            require(opt.resume.empty(), "capacity generation needs original prefix logits; use cycles/boundary/mismatch with resume");
            report["capacity_generation"] = generate(opt.generate, -1, file("capacity-logits.f32"), true);
            report["capacity_generation"]["pass"] = history.size() == size_t(c) && llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == c-1;
            report["capacity_generation"]["memory"] = memory(); publish();
            require(report["capacity_generation"]["pass"].get<bool>(), "generation did not fill C positions");
            reject_excess();
        }
        if (opt.mode == "all" || opt.mode == "boundary") { boundary(); }
        if (opt.mode == "all" || opt.mode == "cycles") { cycles(); }
        if (opt.mode == "all" || opt.mode == "mismatch") { mismatch(); }
        if ((opt.mode == "all" || opt.mode == "cycles") && report.contains("cycles") && !report["cycles"].empty()) {
            discard_verified(file("prefix-" + std::to_string(prefix) + ".bin"),
                report["cycles"][0]["restored_prefix"], "all prefix consumers have completed");
        }
        report["completed"] = true;
        const bool all_requested_checks_executed =
            (opt.mode != "all" && opt.mode != "boundary") ||
            report.value("all_requested_checks_executed", false);
        report["all_requested_checks_executed"] = all_requested_checks_executed;
        report["mandatory_executed_checks_pass"] = true;
        report["executed_checks_pass"] = all_requested_checks_executed;
        report["all_requested_checks_passed"] = all_requested_checks_executed &&
            report["mandatory_executed_checks_pass"].get<bool>();
        report["allocation_outcome"] = "initialization and all requested decodes/restores returned successfully; external logs still required for driver diagnostics";
        // Legacy C=131072 keeps reporting production dimensions; the C=204800
        // ceiling (kCapacityC) is the new target instance.
        const bool production_c = c == 131072 || c == int(llama_position_split::kCapacityC);
        report["production_dimensions"] = production_c && p != 0 && opt.generate == 512 && opt.cycles >= 10;
        report["production_gate_complete"] = false;
        report["remaining_evidence"] = {"external backend allocator retained-memory evidence", "model SHA256 and hardware/driver/concurrency provenance", "pure-CUDA import and split-to-CUDA conversion matrix are separate campaigns"};
        finish_retention_manifest();
        publish();
    }
};

} // namespace

int main(int argc, char ** argv) {
    runner run;
    run.params.n_ctx = llama_position_split::kCapacityC;
    run.params.n_parallel = 1;
    run.params.n_batch = 512;
    run.params.n_ubatch = 256;
    run.params.sampling.temp = 0;
    run.params.speculative.types.clear();
    run.params.speculative.draft.n_max = 0;
    std::vector<char *> args{argv[0]};
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string name(argv[i]);
            if (name == "--capacity-help") { usage(); return 0; }
            if (name == "--capacity-discard-verified-scratch") { run.opt.discard_verified_scratch = true; continue; }
            if (name.compare(0, 11, "--capacity-") != 0) { args.push_back(argv[i]); continue; }
            require(i+1 < argc, "missing value for " + name);
            const std::string value(argv[++i]);
            if (name == "--capacity-token-file") { run.opt.tokens = value; }
            else if (name == "--capacity-token-format") { run.opt.format = value; }
            else if (name == "--capacity-output") { run.opt.output = value; }
            else if (name == "--capacity-work-dir") { run.opt.work = value; }
            else if (name == "--capacity-resume-file") { run.opt.resume = value; }
            else if (name == "--capacity-mode") { run.opt.mode = value; }
            else if (name == "--capacity-cycles") { run.opt.cycles = std::stoi(value); }
            else if (name == "--capacity-generate") { run.opt.generate = std::stoi(value); }
            else if (name == "--capacity-cycle-hold-ms") {
                require(!value.empty() && value.find_first_not_of("0123456789") == std::string::npos,
                        "cycle hold must be an integer in 1..60000 milliseconds");
                const auto hold = std::stoull(value);
                require(hold >= 1 && hold <= 60000, "cycle hold must be in 1..60000 milliseconds");
                run.opt.cycle_hold_ms = int(hold);
            }
            else { throw std::runtime_error("unknown runner flag " + name); }
        }
        common_init();
        args.push_back(nullptr);
        // Split placement and remote cache flags are server-scoped common args.
        if (!common_params_parse(int(args.size()-1), args.data(), run.params, LLAMA_EXAMPLE_SERVER)) { return 1; }
        require(!run.opt.tokens.empty() && !run.opt.output.empty() && !run.opt.work.empty(), "token file, output and work directory are required; see --capacity-help");
        require(run.opt.mode == "all" || run.opt.mode == "capacity" || run.opt.mode == "cycles" || run.opt.mode == "boundary" || run.opt.mode == "mismatch" || run.opt.mode == "empty-local", "unknown campaign mode");
        fs::create_directories(run.opt.work);
        if (!fs::path(run.opt.output).parent_path().empty()) { fs::create_directories(fs::path(run.opt.output).parent_path()); }
        run.report = {{"schema", 1}, {"completed", false}, {"executed_checks_pass", false},
                      {"mandatory_executed_checks_pass", false}, {"all_requested_checks_executed", false},
                      {"all_requested_checks_passed", false}, {"production_gate_complete", false},
                      {"cycle_hold_ms", run.opt.cycle_hold_ms},
                      {"allocation_outcome", "incomplete: process termination or abort requires external OOM/driver log classification"}, {"mode", run.opt.mode}};
        run.report["scratch_retention"] = run.scratch_retention;
        run.report["argv"] = json::array();
        for (int i = 0; i < argc; ++i) { run.report["argv"].push_back(argv[i]); }
        run.publish();
        ggml_backend_load_all();
        run.run();
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "capacity: FAIL: %s\n", error.what());
        if (!run.opt.output.empty() && !run.report.is_null()) {
            run.report["error"] = error.what();
            const std::string message(error.what());
            run.report["error_classification"] = message.find("bad_alloc") != std::string::npos ||
                    message.find("out of memory") != std::string::npos || message.find("OOM") != std::string::npos
                    ? "allocation failure or possible OOM; inspect external logs" : "check failure; inspect external logs for OOM/driver faults";
            run.report["executed_checks_pass"] = false;
            // Publish the primary failure even if optional retention hashing
            // encounters another IO error while collecting failed artifacts.
            try { run.publish(); } catch (...) {}
            try { run.finish_retention_manifest(); run.publish(); }
            catch (const std::exception & retention_error) {
                std::fprintf(stderr, "capacity: retention diagnostic failed: %s\n", retention_error.what());
            }
        }
        return 1;
    }
}
