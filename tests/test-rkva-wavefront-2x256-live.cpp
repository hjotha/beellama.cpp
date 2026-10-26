// Explicit live development-prompt qualification, never a benchmark/holdout gate.
// Manual target only: no network or model initialization without --execute.
#include "llama.h"
#include "llama-cpp.h"
#include "ggml-backend.h"
#include "../src/llama-context.h"
#include "../src/llama-memory-hybrid.h"
#include "../src/llama-memory-recurrent.h"
#include "../src/llama-io.h"
#include "../vendor/nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Existing diagnostic hook, verified as an exported C++ symbol in libllama.
// Keep this declaration local to the manual proof driver, not the public API.
void llama_f4_set_dump_tag(const char * tag);

namespace {
using json = nlohmann::json;
namespace fs = std::filesystem;
void require(bool ok, const std::string & message) { if (!ok) throw std::runtime_error(message); }
// Match server_tokens::deserialize's v1 text-only envelope without bringing
// server/mtmd dependencies into this manual CAPI driver. A marked envelope
// must never fall back to the legacy raw-token interpretation on bad input.
std::vector<llama_token> unpack_prefix_tokens(const std::vector<llama_token> & packed, bool & server_v1) {
    static_assert(sizeof(llama_token) == sizeof(uint32_t), "unexpected token word size");
    require(!packed.empty(), "empty restored token payload");
    server_v1 = packed.front() == LLAMA_TOKEN_NULL;
    if (!server_v1) return packed; // legacy CAPI plain token list
    require(packed.size() >= 4, "truncated server token envelope");
    require(static_cast<uint32_t>(packed[1]) == 1, "unsupported server token state version");
    const size_t count = static_cast<uint32_t>(packed[2]);
    require(count <= packed.size() - 4, "truncated server token IDs");
    require(static_cast<uint32_t>(packed[3 + count]) == 0, "media tokens are unsupported in the development prefix");
    require(packed.size() == count + 4, "trailing data in server token envelope");
    return std::vector<llama_token>(packed.begin() + 3, packed.begin() + 3 + count);
}
void set_env(const char * name, const char * value) {
#ifdef _WIN32
    require(_putenv_s(name, value ? value : "") == 0, "environment update failed");
#else
    require((value ? setenv(name, value, 1) : unsetenv(name)) == 0, "environment update failed");
#endif
}
struct env_restore {
    const char * name;
    bool present;
    std::string value;
    explicit env_restore(const char * key) : name(key), present(std::getenv(key) != nullptr),
        value(present ? std::getenv(key) : "") {}
    ~env_restore() {
#ifdef _WIN32
        _putenv_s(name, present ? value.c_str() : "");
#else
        if (present) setenv(name, value.c_str(), 1); else unsetenv(name);
#endif
    }
};
struct backend_lifetime {
    backend_lifetime() { llama_backend_init(); ggml_backend_load_all(); }
    ~backend_lifetime() { llama_backend_free(); }
};
void write_bytes(const fs::path & path, const void * data, size_t size) {
    std::ofstream out(path, std::ios::binary);
    require(bool(out), "cannot open artifact: " + path.string());
    out.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
    out.close();
    require(bool(out), "cannot finish artifact: " + path.string());
}
void write_json(const fs::path & path, const json & value) {
    const auto text = value.dump(2) + "\n";
    write_bytes(path, text.data(), text.size());
}
uint32_t number(const std::string & value, const char * option) {
    size_t used = 0;
    const auto n = std::stoull(value, &used);
    require(used == value.size() && n <= UINT32_MAX && !value.empty() && value[0] != '-',
            std::string("invalid ") + option);
    return static_cast<uint32_t>(n);
}
struct options {
    std::string model, tokens, host;
    std::string off_cuda_fp8, on_cuda_fp8;
    std::string off_tcp_prefetch, on_tcp_prefetch;
    std::string off_local_input_reuse, on_local_input_reuse;
    std::string off_mask_pack, on_mask_pack;
    fs::path output;
    uint32_t p = 512, context = 2048, port = 50054;
    uint32_t parent_tokens = 512;
    uint32_t serial_ubatch = 256;
    uint32_t prefix_probe_tokens = 0;
    bool control_only_pair = false;
    fs::path restore_prefix_slot, followup_tokens_file;
    uint32_t overflow_tokens = 1024;
    bool repeat_prompt = false, serving_prefill = false, serving_checkpoint_seam = false;
    bool skip_hybrid_diagnostics = false;
    std::vector<uint32_t> followup_capture_steps;
};
struct checkpoint {
    std::string label;
    std::vector<float> logits;
    json metadata;
};
using arm_result = std::vector<checkpoint>;
class recurrent_file_writer final : public llama_io_write_i {
    std::ofstream output;
    std::vector<uint8_t> scratch;
    size_t written = 0;
    static constexpr size_t limit = size_t(1) << 30;
public:
    explicit recurrent_file_writer(const fs::path & path)
        : output(path, std::ios::binary), scratch(4 * 1024 * 1024) {
        require(bool(output), "cannot open recurrent state artifact: " + path.string());
    }
    void write(const void * src, size_t bytes) override {
        require(bytes <= limit - written, "recurrent state artifact exceeds 1 GiB");
        if (bytes) output.write(static_cast<const char *>(src), static_cast<std::streamsize>(bytes));
        require(bool(output), "cannot write recurrent state artifact");
        written += bytes;
    }
    void write_tensor(ggml_tensor * tensor, size_t offset, size_t bytes) override {
        require(tensor && offset <= ggml_nbytes(tensor) && bytes <= ggml_nbytes(tensor) - offset,
                "recurrent state tensor span is invalid");
        require(bytes <= limit - written, "recurrent state artifact exceeds 1 GiB");
        // This Qwen35 proof expects F32 convolution/GDN payloads. Equal NaN
        // bit patterns must not turn a poisoned state into a parity pass.
        require(tensor->type == GGML_TYPE_F32 && offset % sizeof(float) == 0 && bytes % sizeof(float) == 0,
                "recurrent proof requires aligned F32 tensor payloads");
        while (bytes) {
            const size_t chunk = std::min(bytes, scratch.size());
            ggml_backend_tensor_get(tensor, scratch.data(), offset, chunk);
            for (size_t word = 0; word < chunk; word += sizeof(float)) {
                float value;
                std::memcpy(&value, scratch.data() + word, sizeof(value));
                require(std::isfinite(value), "recurrent state contains a nonfinite F32 payload value");
            }
            write(scratch.data(), chunk);
            offset += chunk;
            bytes -= chunk;
        }
    }
    size_t n_bytes() override { return written; }
    size_t finish(const fs::path & path) {
        output.close();
        require(bool(output) && fs::file_size(path) == written && written != 0,
                "recurrent state artifact did not close with its complete byte count");
        return written;
    }
};
void dump_state(llama_context * ctx, const fs::path & output, const std::string & stem, json & record,
                bool hybrid_diagnostics = true) {
    // KVarN can require attention data even for PARTIAL_ONLY. Serialize the
    // recurrent owner directly so this strict comparison contains convolution
    // and GDN state plus their own metadata, never attention cache metadata.
    auto * hybrid = dynamic_cast<llama_memory_hybrid *>(llama_get_memory(ctx));
    require(hybrid && hybrid->get_mem_recr(), "pure recurrent proof requires the hybrid recurrent owner");
    const auto recurrent_file = stem + ".recurrent.bin";
    const auto recurrent_path = output / recurrent_file;
    size_t recurrent_bytes = 0;
    try {
        recurrent_file_writer writer(recurrent_path);
        hybrid->get_mem_recr()->state_write(writer, 0, 0);
        recurrent_bytes = writer.finish(recurrent_path);
    } catch (...) {
        std::error_code ignored;
        fs::remove(recurrent_path, ignored);
        throw;
    }
    record["recurrent_file"] = recurrent_file;
    record["recurrent_bytes"] = recurrent_bytes;
    record["recurrent_format"] = "llama_memory_recurrent::state_write(seq_id=0,flags=0); convolution/GDN and recurrent metadata only; exact serialized comparison";
    if (!hybrid_diagnostics) {
        record["hybrid_diagnostics_skipped"] = "explicitly disabled; complete pure recurrent state is still compared";
        return;
    }

    // Public hybrid snapshots are diagnostic artifacts only: PARTIAL_ONLY
    // may still contain KVarN attention data and is not a GDN-only contract.
    for (bool partial : { true, false }) {
        const auto flags = partial ? LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY : llama_state_seq_flags(0);
        const char * kind = partial ? "hybrid_partial_diagnostic" : "hybrid_full_diagnostic";
        try {
            const size_t bytes = llama_state_seq_get_size_ext(ctx, 0, flags);
            record[kind]["bytes"] = bytes;
            if (bytes == 0 || bytes > (size_t(1) << 30)) {
                record[kind]["skipped"] = "empty or larger than 1 GiB";
                continue;
            }
            std::vector<uint8_t> state(bytes);
            require(llama_state_seq_get_data_ext(ctx, state.data(), state.size(), 0, flags) == bytes,
                    "hybrid sequence serialization returned a different size");
            const std::string file = stem + (partial ? ".hybrid.partial.bin" : ".hybrid.full.bin");
            write_bytes(output / file, state.data(), state.size());
            record[kind]["file"] = file;
            record[kind]["comparison"] = "not compared; may include attention cache and configuration-dependent metadata";
        } catch (const std::exception & error) {
            record[kind]["skipped"] = error.what();
        }
    }
}

json compare_recurrent_files(const fs::path & off_file, const fs::path & on_file) {
    const uint64_t off_bytes = fs::file_size(off_file), on_bytes = fs::file_size(on_file);
    std::ifstream off_state(off_file, std::ios::binary), on_state(on_file, std::ios::binary);
    require(bool(off_state) && bool(on_state), "cannot reopen recurrent snapshots");
    std::array<char, 65536> a {}, b {};
    uint64_t state_differences = off_bytes > on_bytes ? off_bytes - on_bytes : on_bytes - off_bytes;
    uint64_t first_difference = off_bytes != on_bytes ? std::min(off_bytes, on_bytes) : UINT64_MAX;
    for (uint64_t offset = 0; offset < std::min(off_bytes, on_bytes);) {
        const auto count = static_cast<std::streamsize>(std::min<uint64_t>(a.size(), std::min(off_bytes, on_bytes) - offset));
        off_state.read(a.data(), count); on_state.read(b.data(), count);
        require(off_state.gcount() == count && on_state.gcount() == count, "recurrent snapshot read truncated");
        for (std::streamsize j = 0; j < count; ++j) {
            if (a[static_cast<size_t>(j)] != b[static_cast<size_t>(j)]) {
                ++state_differences;
                first_difference = std::min(first_difference, offset + static_cast<uint64_t>(j));
            }
        }
        offset += static_cast<uint64_t>(count);
    }
    return {{"recurrent_serialized_bytes_equal", state_differences == 0},
        {"recurrent_different_bytes", state_differences},
        {"recurrent_first_difference", first_difference == UINT64_MAX ? json(nullptr) : json(first_difference)}};
}

arm_result run_arm(llama_model * model, const options & opt, const std::vector<llama_token> & tokens,
                   bool wavefront, json & manifest, const char * arm_name = nullptr) {
    const char * arm = arm_name ? arm_name : (wavefront ? "on" : "off");
    const bool second = std::strcmp(arm, "on") == 0 || std::strcmp(arm, "control-b") == 0;
    const auto & codec = second ? opt.on_cuda_fp8 : opt.off_cuda_fp8;
    if (!codec.empty()) set_env("GGML_RKVA_CUDA_FP8", codec.c_str());
    const auto & prefetch = second ? opt.on_tcp_prefetch : opt.off_tcp_prefetch;
    if (!prefetch.empty()) set_env("GGML_RKVA_TCP_PREFETCH", prefetch.c_str());
    const auto & reuse = second ? opt.on_local_input_reuse : opt.off_local_input_reuse;
    if (!reuse.empty()) set_env("GGML_RKVA_LOCAL_INPUT_REUSE", reuse.c_str());
    const auto & mask_pack = second ? opt.on_mask_pack : opt.off_mask_pack;
    if (!mask_pack.empty()) set_env("GGML_CUDA_MASK_PACK", mask_pack == "default" ? nullptr : mask_pack.c_str());
    const bool enabled = wavefront || opt.parent_tokens == 1024;
    const uint32_t requested_ubatch = opt.parent_tokens == 1024 ? (wavefront ? 1024 : 512) : (wavefront ? 512 : opt.serial_ubatch);
    set_env("GGML_RKVA_WAVEFRONT_TILE", enabled ? "256" : "0");
    const std::string arm_tag = std::string("2xn256-live-") + arm;
    llama_f4_set_dump_tag((arm_tag + "-context-init").c_str());
    std::fprintf(stderr, "LIVE-2XN256 arm=%s context_begin parent_batch=%u configured_ubatch=%u P=%u C=%u\n",
            arm, opt.parent_tokens, requested_ubatch, opt.p, opt.context);
    std::fflush(stderr);
    auto cp = llama_context_default_params();
    cp.n_ctx = opt.context;
    cp.n_batch = opt.parent_tokens;
    cp.n_ubatch = requested_ubatch;
    cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = 8;
    cp.n_rs_seq = 0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.kvarn = llama_kvarn_params_for_type(LLAMA_KVARN_K4V4_G128);
    cp.kv_position_split_p = opt.p;
    cp.remote_attn_host = opt.host.c_str();
    cp.remote_attn_port = static_cast<uint16_t>(opt.port);
    cp.remote_attn_n_layers = 0;
    cp.remote_attn_stats = 1;
    cp.remote_attn_cache_type_k = cp.remote_attn_cache_type_v = GGML_TYPE_Q4_0;
    llama_context_ptr ctx(llama_init_from_model(model, cp));
    require(bool(ctx), std::string("context creation failed for arm ") + arm);
    manifest["arms"][arm]["wavefront_tile"] = enabled ? 256 : 0;
    manifest["arms"][arm]["wavefront_tiles"] = enabled ? requested_ubatch / 256 : 0;
    const char * active_codec = std::getenv("GGML_RKVA_CUDA_FP8");
    manifest["arms"][arm]["cuda_fp8_mode"] = active_codec ? json(active_codec) : json(nullptr);
    const char * active_prefetch = std::getenv("GGML_RKVA_TCP_PREFETCH");
    manifest["arms"][arm]["tcp_prefetch"] = active_prefetch ? json(active_prefetch) : json(nullptr);
    const char * active_reuse = std::getenv("GGML_RKVA_LOCAL_INPUT_REUSE");
    manifest["arms"][arm]["local_input_reuse"] = active_reuse ? json(active_reuse) : json(nullptr);
    const char * active_mask_pack = std::getenv("GGML_CUDA_MASK_PACK");
    manifest["arms"][arm]["mask_pack_environment"] = active_mask_pack ? json(active_mask_pack) : json(nullptr);
    manifest["arms"][arm]["configured_ubatch"] = llama_n_ubatch(ctx.get());
    const auto physical_ubatch = llama_n_ubatch_physical(ctx.get());
    require(llama_n_ubatch(ctx.get()) == requested_ubatch && physical_ubatch == (enabled ? 256u : opt.serial_ubatch),
            "context logical/physical ubatch API does not match the requested 2xN256 arm");
    manifest["arms"][arm]["physical_ubatch"] = physical_ubatch;
    const size_t vocab = static_cast<size_t>(llama_vocab_n_tokens(llama_model_get_vocab(model)));
    arm_result result;
    uint32_t restored_tokens = 0;
    if (!opt.restore_prefix_slot.empty()) {
        size_t packed_count = 0;
        const size_t header_read = llama_state_seq_load_file(ctx.get(), opt.restore_prefix_slot.string().c_str(),
                0, nullptr, 0, &packed_count);
        // Bound allocation before loading: only the two accepted checkpoint
        // lengths, either raw CAPI IDs or the four-word server text envelope.
        require(header_read > 0 && (packed_count == opt.p || packed_count == opt.p - 128 ||
                    packed_count == size_t(opt.p) + 4 || packed_count == size_t(opt.p - 128) + 4),
                "prefix restore has an unexpected packed token count");
        std::vector<llama_token> packed(packed_count);
        size_t actual_packed_count = 0;
        const size_t read = llama_state_seq_load_file(ctx.get(), opt.restore_prefix_slot.string().c_str(),
                0, packed.data(), packed.size(), &actual_packed_count);
        require(read > 0 && actual_packed_count == packed_count, "prefix restore packed token count changed or load failed");
        bool server_v1 = false;
        const auto loaded = unpack_prefix_tokens(packed, server_v1);
        const size_t count = loaded.size();
        require(count == opt.p || count == opt.p - 128,
                "prefix restore must contain exactly P or the production aligned checkpoint P-128 token IDs");
        require(std::equal(loaded.begin(), loaded.end(), tokens.begin()), "restored prefix token IDs differ from development input");
        require(llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0) == int64_t(count) - 1,
                "restored prefix position does not match its actual loaded length");
        restored_tokens = static_cast<uint32_t>(count);
        llama_synchronize(ctx.get());
        const std::string label = "restored-prefix-" + std::to_string(restored_tokens);
        llama_f4_set_dump_tag((arm_tag + "-" + label).c_str());
        json restored = {{"file", opt.restore_prefix_slot.string()}, {"n_loaded", count}, {"n_read", read},
            {"packed_words", packed_count}, {"token_format", server_v1 ? "server-text-v1" : "legacy-raw-token-ids"},
            {"token_ids_match", true}, {"logits_available", false}, {"repair_tokens", opt.p - restored_tokens},
            {"scope", "frozen prefix restore, not a new end-to-end prefill qualification"}};
        dump_state(ctx.get(), opt.output, std::string(arm) + "-" + label, restored, !opt.skip_hybrid_diagnostics);
        manifest["arms"][arm]["prefix_restore"] = std::move(restored);
        write_json(opt.output / "manifest.json", manifest);
        std::fprintf(stderr, "LIVE-2XN256 arm=%s restored_prefix=%u n_read=%zu logits=unavailable\n", arm, restored_tokens, read);
    }
    auto decode = [&](uint32_t start, uint32_t count, const std::string & label, bool capture = true, bool want_logits = true) {
        // Long prefix progress shares a tag, so the bounded opmap does not
        // expand into hundreds of full graphs. Reset at the final prefix and
        // each overflow/followup checkpoint to cover the actual candidate.
        const std::string dump_tag = arm_tag + "-" + (capture ? label : "prefix-progress");
        llama_f4_set_dump_tag(dump_tag.c_str());
        std::fprintf(stderr, "LIVE-2XN256 arm=%s checkpoint=%s start=%u tokens=%u begin\n", arm, label.c_str(), start, count);
        std::fflush(stderr);
        llama_batch batch = llama_batch_init(static_cast<int32_t>(count), 0, 1);
        struct batch_guard { llama_batch & b; ~batch_guard() { llama_batch_free(b); } } guard { batch };
        batch.n_tokens = static_cast<int32_t>(count);
        for (uint32_t i = 0; i < count; ++i) {
            batch.token[i] = tokens[start + i];
            batch.pos[i] = static_cast<llama_pos>(start + i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = want_logits && i + 1 == count;
        }
        require(llama_decode(ctx.get(), batch) == 0, std::string("decode failed: ") + arm + "/" + label);
        llama_synchronize(ctx.get());
        require(!capture || want_logits, "artifact capture requires an actual output row");
        const float * logits = want_logits ? llama_get_logits_ith(ctx.get(), static_cast<int32_t>(count - 1)) : nullptr;
        if (want_logits) {
            require(logits != nullptr, "last-row logits missing");
            require(std::all_of(logits, logits + vocab, [](float x) { return std::isfinite(x); }),
                    std::string("nonfinite logits: ") + arm + "/" + label);
        }
        if (!capture) {
            // Q1 serving always produces and fetches its logits, even when
            // this teacher-forced step does not write a large state artifact.
            const auto argmax = want_logits ? std::max_element(logits, logits + vocab) - logits : -1;
            std::fprintf(stderr, "LIVE-2XN256 arm=%s progress=%s end=%u outputs=%d argmax=%lld (no artifact capture)\n",
                    arm, label.c_str(), start + count, int(want_logits), static_cast<long long>(argmax));
            return;
        }
        checkpoint record;
        record.label = label;
        record.logits.assign(logits, logits + vocab);
        const auto argmax = std::max_element(record.logits.begin(), record.logits.end()) - record.logits.begin();
        const std::string stem = std::string(arm) + "-" + label;
        write_bytes(opt.output / (stem + ".logits.f32.bin"), record.logits.data(), vocab * sizeof(float));
        record.metadata = {{"label", label}, {"f4_tag", dump_tag}, {"start", start}, {"tokens", count}, {"argmax", argmax},
            {"vocab", vocab}, {"logits_file", stem + ".logits.f32.bin"}, {"finite", true}};
        dump_state(ctx.get(), opt.output, stem, record.metadata, !opt.skip_hybrid_diagnostics);
        manifest["arms"][arm]["checkpoints"].push_back(record.metadata);
        write_json(opt.output / "manifest.json", manifest);
        result.push_back(std::move(record));
        std::fprintf(stderr, "LIVE-2XN256 arm=%s checkpoint=%s complete argmax=%lld\n", arm, label.c_str(), static_cast<long long>(argmax));
        std::fflush(stderr);
    };
    if (!opt.restore_prefix_slot.empty() && restored_tokens < opt.p) {
        // The server saves the latest production-aligned checkpoint P-128.
        // Replay only its missing final physical128, with real output logits;
        // the raw pre-replay recurrent snapshot remains a separate comparison.
        const std::string label = "prefix-repair-" + std::to_string(opt.p);
        decode(restored_tokens, opt.p - restored_tokens, label, true, true);
        manifest["arms"][arm]["prefix_restore"]["repair_checkpoint"] = label;
        manifest["arms"][arm]["prefix_restore"]["repair_complete"] = true;
        write_json(opt.output / "manifest.json", manifest);
    }
    const uint32_t prompt_end = opt.prefix_probe_tokens ? opt.prefix_probe_tokens : opt.p + opt.overflow_tokens;
    for (uint32_t start = opt.restore_prefix_slot.empty() ? 0 : opt.p; start < prompt_end;) {
        const bool overflow = start >= opt.p;
        const uint32_t region_end = overflow ? prompt_end : std::min(opt.p, prompt_end);
        const uint32_t remaining = region_end - start;
        uint32_t count = std::min(opt.parent_tokens, remaining);
        // Carve the final512 before choosing a logical parent, so both B1024
        // arms retain production's final256/128/128 seam independently of UB.
        if (opt.serving_checkpoint_seam && remaining > 512 && count == remaining) count -= 512;
        const auto label = !overflow ? "prefix-" + std::to_string(start + count)
            : "overflow-" + std::to_string((start + count - opt.p) / 512);
        const bool final = start + count == prompt_end;
        const bool capture = opt.prefix_probe_tokens ? final :
            !overflow ? start + count == opt.p : opt.serving_prefill ? final : (start < opt.p + 2 * opt.parent_tokens || final);
        const bool want_logits = !opt.serving_prefill || capture;
        const bool final_prefix_parent = !overflow && start + count == std::min(opt.p, prompt_end);
        if (opt.serving_checkpoint_seam && (final_prefix_parent || (overflow && final))) {
            require(count == 512 || (count == 256 && opt.parent_tokens == 256), "invalid final production seam");
            // Match the production B256/physical-N256 checkpoint cadence:
            // 256 + 128 + 128, irrespective of the arm's logical parent size.
            // All parts take the serial fallback; only the final 128 rows
            // request the final sample's logits. No older R3 seam is assumed.
            std::fprintf(stderr, "LIVE-2XN256 arm=%s checkpoint=%s logical_parent_start=%u seam=%s final_sample_end=%u\n",
                    arm, label.c_str(), start, count == 512 ? "256,128,128" : "128,128", start + count);
            if (count == 512) decode(start, 256, label + "-seam-head-256", false, false);
            decode(start + count - 256, 128, label + "-seam-middle-128", false, false);
            decode(start + count - 128, 128, label, capture, want_logits);
        } else {
            decode(start, count, label, capture, want_logits);
        }
        start += count;
    }
    // All followup IDs are fixed before either arm runs. Never substitute an
    // arm's sampled argmax, which would confound state and input divergence.
    if (!opt.prefix_probe_tokens) {
        const uint32_t followups = uint32_t(tokens.size()) - prompt_end;
        for (uint32_t step = 1; step <= followups; ++step) {
            const bool capture = std::binary_search(opt.followup_capture_steps.begin(), opt.followup_capture_steps.end(), step);
            decode(prompt_end + step - 1, 1, "followup-" + std::to_string(step), capture, true);
        }
    }
    ggml_backend_sched_input_cache_stats cache_stats {};
    ggml_backend_sched_get_input_cache_stats(ctx->get_sched(), &cache_stats);
    manifest["arms"][arm]["local_input_cache"] = {
        {"hits", cache_stats.hits}, {"misses", cache_stats.misses},
        {"copied_bytes", cache_stats.copied_bytes}, {"skipped_bytes", cache_stats.skipped_bytes}};
    if (!reuse.empty() && !opt.prefix_probe_tokens) {
        if (reuse == "1") require(cache_stats.hits > 0 && cache_stats.skipped_bytes > 0 &&
                cache_stats.copied_bytes > 0, "local input reuse requested but no actual reuse was proved");
        else require(cache_stats.hits == 0 && cache_stats.skipped_bytes == 0,
                "local input reuse OFF arm unexpectedly reused inputs");
    }
    std::fprintf(stderr, "LIVE-2XN256 arm=%s context_end\n", arm);
    std::fflush(stderr);
    return result;
}
}

int main(int argc, char ** argv) {
    bool execute = false;
    for (int i = 1; i < argc; ++i) execute |= std::string(argv[i]) == "--execute";
    if (!execute || argc < 6) {
        std::printf("Manual development-prompt parity test; no performance or historical holdout claim.\n"
            "Usage: %s MODEL TOKENS_JSON XBOX_HOST OUTPUT_DIR --execute [--p 512] [--ctx 2048] [--port 50054]\n"
            "Diagnostics only: [--prefix-probe-tokens L] [--control-only-pair]\n"
            "Frozen/teacher-forced: [--restore-prefix-slot PATH] [--overflow-tokens N] [--repeat-prompt]\n"
            "  [--serving-prefill] [--serving-checkpoint-seam] [--followup-tokens PATH] [--followup-capture-step N ...]\n"
            "  [--off-cuda-fp8 prefill|mapped --on-cuda-fp8 prefill|mapped] (explicit codec A/B)\n"
            "  [--parent-tokens 1024] (B1024 both arms; control2xN256 versus candidate4xN256)\n"
            "  [--off-tcp-prefetch 0|1 --on-tcp-prefetch 0|1] (use --control-only-pair --parent-tokens 1024 for fixed2xN256)\n"
            "  [--off-local-input-reuse 0|1 --on-local-input-reuse 0|1] (requires --control-only-pair)\n"
            "  [--serial-ubatch 128|256] (serial control pair with parent512)\n"
            "  [--skip-hybrid-diagnostics] (retain full pure recurrent/logit comparisons)\n"
            "Without --execute no model/GPU/network initialization occurs.\n", argv[0]);
        return execute ? 2 : 0;
    }
    options opt;
    bool output_owned = false;
    json manifest = {{"purpose", "development prompt exact 2xN256 parity; no performance claim"}, {"status", "starting"}};
    try {
        opt.model = argv[1]; opt.tokens = argv[2]; opt.host = argv[3]; opt.output = argv[4];
        for (int i = 5; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--execute") continue;
            if (arg == "--control-only-pair") { opt.control_only_pair = true; continue; }
            if (arg == "--repeat-prompt") { opt.repeat_prompt = true; continue; }
            if (arg == "--serving-prefill") { opt.serving_prefill = true; continue; }
            if (arg == "--serving-checkpoint-seam") { opt.serving_checkpoint_seam = true; continue; }
            if (arg == "--skip-hybrid-diagnostics") { opt.skip_hybrid_diagnostics = true; continue; }
            require(i + 1 < argc, "option value missing");
            const std::string text = argv[++i];
            if (arg == "--restore-prefix-slot") { opt.restore_prefix_slot = text; continue; }
            if (arg == "--followup-tokens") { opt.followup_tokens_file = text; continue; }
            if (arg == "--off-mask-pack" || arg == "--on-mask-pack") {
                require(text == "0" || text == "1" || text == "default", "unsupported mask-pack mode");
                (arg == "--off-mask-pack" ? opt.off_mask_pack : opt.on_mask_pack) = text;
                continue;
            }
            if (arg == "--off-local-input-reuse" || arg == "--on-local-input-reuse") {
                require(text == "0" || text == "1", "unsupported local input reuse mode");
                (arg == "--off-local-input-reuse" ? opt.off_local_input_reuse : opt.on_local_input_reuse) = text;
                continue;
            }
            if (arg == "--off-tcp-prefetch" || arg == "--on-tcp-prefetch") {
                require(text == "0" || text == "1", "unsupported TCP prefetch mode");
                (arg == "--off-tcp-prefetch" ? opt.off_tcp_prefetch : opt.on_tcp_prefetch) = text;
                continue;
            }
            if (arg == "--off-cuda-fp8" || arg == "--on-cuda-fp8") {
                require(text == "prefill" || text == "mapped", "unsupported CUDA FP8 mode");
                (arg == "--off-cuda-fp8" ? opt.off_cuda_fp8 : opt.on_cuda_fp8) = text;
                continue;
            }
            const auto value = number(text, arg.c_str());
            if (arg == "--p") opt.p = value;
            else if (arg == "--ctx") opt.context = value;
            else if (arg == "--parent-tokens") opt.parent_tokens = value;
            else if (arg == "--serial-ubatch") opt.serial_ubatch = value;
            else if (arg == "--port") opt.port = value;
            else if (arg == "--overflow-tokens") opt.overflow_tokens = value;
            else if (arg == "--followup-capture-step") opt.followup_capture_steps.push_back(value);
            else if (arg == "--prefix-probe-tokens") {
                require(value > 0, "prefix probe length must be positive");
                opt.prefix_probe_tokens = value;
            }
            else throw std::runtime_error("unknown option: " + arg);
        }
        require(opt.p != 0 && opt.p % 512 == 0, "P must be a positive multiple of 512");
        require(opt.parent_tokens == 256 || opt.parent_tokens == 512 || opt.parent_tokens == 1024, "parent tokens must be256/512/1024");
        require(opt.parent_tokens != 256 || opt.control_only_pair, "B256 requires a serial control pair");
        require(opt.serial_ubatch == 128 || opt.serial_ubatch == 256, "serial ubatch must be128 or256");
        require(opt.serial_ubatch == 256 || (opt.control_only_pair && opt.parent_tokens == 512),
                "serial ubatch128 requires a serial control pair with parent512");
        require(opt.off_cuda_fp8.empty() == opt.on_cuda_fp8.empty(), "codec A/B requires both explicit arm modes");
        require(opt.off_tcp_prefetch.empty() == opt.on_tcp_prefetch.empty(), "TCP prefetch A/B requires both explicit arm modes");
        const bool tcp_pair = !opt.off_tcp_prefetch.empty();
        require(opt.off_local_input_reuse.empty() == opt.on_local_input_reuse.empty(), "local input reuse A/B requires both arm modes");
        const bool reuse_pair = !opt.off_local_input_reuse.empty();
        require(opt.off_mask_pack.empty() == opt.on_mask_pack.empty(), "mask-pack A/B requires both arm modes");
        const bool mask_pair = !opt.off_mask_pack.empty();
        require(!mask_pair || (opt.control_only_pair && opt.off_cuda_fp8 == opt.on_cuda_fp8 &&
                    opt.off_tcp_prefetch == opt.on_tcp_prefetch && opt.off_local_input_reuse == opt.on_local_input_reuse),
                "mask-pack A/B must hold geometry, codec, transport and residency fixed");
        require(!reuse_pair || (opt.control_only_pair && opt.off_cuda_fp8 == opt.on_cuda_fp8 &&
                    opt.off_tcp_prefetch == opt.on_tcp_prefetch),
                "local input reuse A/B must hold control geometry, codec and TCP prefetch fixed");
        require(!tcp_pair || opt.off_cuda_fp8 == opt.on_cuda_fp8,
                "TCP prefetch A/B must keep the CUDA FP8 codec identical");
        require(!tcp_pair || (opt.control_only_pair && opt.parent_tokens == 1024),
                "TCP prefetch A/B requires --control-only-pair --parent-tokens 1024 to hold B1024/UB512/two N256 tiles fixed");
        require(opt.overflow_tokens >= 512 && opt.overflow_tokens % 512 == 0,
                "overflow tokens must be a positive multiple of 512");
        require(opt.prefix_probe_tokens == 0 || (opt.prefix_probe_tokens % 512 == 0 && opt.prefix_probe_tokens <= opt.p),
                "prefix probe length must be a positive multiple of 512 at or below P");
        require(opt.prefix_probe_tokens == 0 || (opt.restore_prefix_slot.empty() && opt.followup_tokens_file.empty()),
                "prefix-only diagnostics cannot combine restored prefixes or followup files");
        if (!opt.restore_prefix_slot.empty()) require(fs::is_regular_file(opt.restore_prefix_slot), "prefix slot file missing");
        const bool diagnostic = opt.prefix_probe_tokens != 0 || (opt.control_only_pair && !tcp_pair && !reuse_pair && !mask_pair);
        require(opt.port > 0 && opt.port <= UINT16_MAX && !opt.host.empty(), "invalid Xbox endpoint");
        require(fs::is_regular_file(opt.model) && fs::is_regular_file(opt.tokens), "model/token file missing");
        auto read_ids = [](const fs::path & path, bool array_only) {
            std::ifstream in(path); json doc; in >> doc;
            const auto & ids = !array_only && doc.is_object() ? doc.at("tokens") : doc;
            require(ids.is_array() && !ids.empty(), "token JSON must contain a nonempty integer array");
            std::vector<llama_token> result;
            for (const auto & id : ids) {
                require(id.is_number_integer(), "token ID is not an integer");
                const auto value = id.get<int64_t>();
                require(value >= 0 && value <= INT32_MAX, "token ID out of range");
                result.push_back(static_cast<llama_token>(value));
            }
            return result;
        };
        const auto input_ids = read_ids(opt.tokens, false);
        const auto forced = opt.followup_tokens_file.empty() ? std::vector<llama_token>{} : read_ids(opt.followup_tokens_file, true);
        const uint64_t prompt_end = uint64_t(opt.p) + opt.overflow_tokens;
        const uint64_t followup_count = forced.empty() ? 1 : forced.size();
        require(prompt_end + followup_count <= opt.context && opt.context <= uint32_t(INT32_MAX),
                "P + overflow + followups exceeds context or the position range");
        const size_t needed_input = opt.prefix_probe_tokens ? opt.prefix_probe_tokens :
            size_t(prompt_end) + (forced.empty() ? 1 : 0);
        require(opt.repeat_prompt || input_ids.size() >= needed_input,
                "development token file is too short; repetition requires explicit --repeat-prompt");
        std::vector<llama_token> tokens;
        tokens.reserve(needed_input + forced.size());
        for (size_t i = 0; i < needed_input; ++i) tokens.push_back(input_ids[i % input_ids.size()]);
        if (!opt.prefix_probe_tokens && !forced.empty()) tokens.insert(tokens.end(), forced.begin(), forced.end());
        if (opt.followup_capture_steps.empty()) opt.followup_capture_steps = {1, uint32_t(followup_count)};
        opt.followup_capture_steps.push_back(uint32_t(followup_count));
        for (auto step : opt.followup_capture_steps) require(step >= 1 && step <= followup_count, "followup capture step is out of range");
        std::sort(opt.followup_capture_steps.begin(), opt.followup_capture_steps.end());
        opt.followup_capture_steps.erase(std::unique(opt.followup_capture_steps.begin(), opt.followup_capture_steps.end()), opt.followup_capture_steps.end());
        require(!fs::exists(opt.output) || fs::is_empty(opt.output), "output directory must be new or empty");
        fs::create_directories(opt.output);
        output_owned = true;
        const uint32_t endian = 1;
        require(*reinterpret_cast<const uint8_t *>(&endian) == 1 && sizeof(float) == 4,
                "raw logits artifacts require little-endian float32");
        manifest["model"] = opt.model; manifest["development_tokens_file"] = opt.tokens;
        manifest["prompt_tokens"] = opt.prefix_probe_tokens ? opt.prefix_probe_tokens : prompt_end;
        manifest["fixed_followup_tokens"] = opt.prefix_probe_tokens ? 0 : followup_count;
        manifest["overflow_tokens"] = opt.overflow_tokens;
        manifest["logical_batch"] = opt.parent_tokens;
        manifest["comparison_geometry"] = opt.control_only_pair ? "repeat the configured control arm" :
            opt.parent_tokens == 1024 ? "B1024 both arms: UB512/two N256 versus UB1024/four N256" : "B512: serial UB256 versus wavefront UB512/two N256";
        manifest["repeat_prompt"] = opt.repeat_prompt;
        manifest["serving_prefill"] = opt.serving_prefill;
        manifest["serving_checkpoint_seam"] = opt.serving_checkpoint_seam;
        if (opt.serving_checkpoint_seam) {
            manifest["final_parent_seam"] = {256, 128, 128};
            manifest["final_parent_seam_outputs"] = {0, 0, 1};
            manifest["seam_scope"] = "last prefix and last overflow logical512; restored prefixes retain the cadence encoded by their source snapshot";
        }
        manifest["followup_tokens_file"] = opt.followup_tokens_file.string();
        manifest["followup_capture_steps"] = opt.followup_capture_steps;
        manifest["frozen_prefix_reuse"] = !opt.restore_prefix_slot.empty();
        manifest["prefix_slot_file"] = opt.restore_prefix_slot.string();
        if (!opt.restore_prefix_slot.empty()) manifest["purpose"] = "frozen-prefix development continuation/teacher-forced parity; not a new prefix or end-to-end prefill qualification";
        manifest["diagnostic_only"] = diagnostic;
        manifest["control_only_pair"] = opt.control_only_pair;
        if (diagnostic) manifest["purpose"] = "prefix/repeatability diagnostic only; not WF 2xN256 qualification or a performance claim";
        manifest["P"] = opt.p; manifest["context"] = opt.context; manifest["host"] = opt.host; manifest["port"] = opt.port;
        manifest["activation_requirement"] = "Inspect arm-scoped logs: WF parent512/tile256 activation in ON overflow; prefix remains N256. Parity alone does not prove activation.";
        if (opt.parent_tokens == 1024) manifest["activation_requirement"] = "Inspect arm-scoped logs: OFF tiles2,parent512 and ON tiles4,parent1024, finalFFN256,max_outstanding2 in both; prefix/tails stay physical256. Parity alone does not prove activation.";
        if (tcp_pair) {
            manifest["comparison_geometry"] = "B1024/UB512/two N256 tiles and codec fixed; explicit per-arm TCP prefetch";
            manifest["activation_requirement"] = "Both arms require tiles2,parent512,finalFFN256,max_outstanding2; prefetch ON must prove max_frames2 and actual TX ahead. Parity alone does not prove overlap.";
        }
        if (reuse_pair) {
            manifest["purpose"] = "fixed-geometry local input reuse exact logits/recurrent parity; not throughput or tier migration qualification";
            manifest["comparison_geometry"] = "fixed control parent/ubatch/codec/transport; explicit per-arm local input reuse";
            manifest["activation_requirement"] = "Inspect per-arm input-cache counters: ON must reuse bytes after actual CUDA copies; equality alone does not prove activation.";
        }
        manifest["local_input_reuse_scope"] = "local I64 maps; serial/two-tile local masks; four-tile prefill masks retain ordinary copies to preserve workspace aliasing; N1 masks remain eligible";
        if (mask_pair) {
            manifest["purpose"] = "fixed-geometry lossless binary-mask transport exact logits/recurrent parity";
            manifest["activation_requirement"] = "Inspect per-arm ggml_cuda_mask_pack counters; DEFAULT must actually pack and OFF must not";
        }
        manifest["skip_hybrid_diagnostics"] = opt.skip_hybrid_diagnostics;
        if (diagnostic) manifest["activation_requirement"] = "Diagnostic only: inspect the configured geometry in each arm; prefix probe deliberately executes no overflow, and control pairs repeat the same geometry.";
        for (const char * key : {"GGML_RKVA_ASYNC_LOCAL_REMOTE", "GGML_RKVA_NATIVE_KVARN4_EXPERIMENTAL", "GGML_RKVA_NATIVE_BULK",
                "GGML_RKVA_NATIVE_BULK_WIRE", "GGML_RKVA_WIRE_FORMAT", "GGML_RKVA_CUDA_FP8", "GGML_RKVA_WAVEFRONT_RPC_BATCH", "GGML_RKVA_TCP_PREFETCH", "GGML_RKVA_LOCAL_INPUT_REUSE", "GGML_CUDA_MASK_PACK",
                "F4_NODE_DUMP", "F4_NODE_DUMP_ROLES", "F4_NODE_DUMP_OPMAP", "F4_NODE_DUMP_AUTOTAG",
                "GGML_CUDA_DISABLE_GRAPHS", "GGML_CUDA_GRAPH_RECOVERY_HEADROOM_MB",
                "GGML_KVARN_WINDOW_CHUNK", "GGML_CUDA_FA_ROUTE_DEBUG"}) {
            const char * value = std::getenv(key);
            manifest["environment"][key] = value ? json(value) : json(nullptr);
        }
        write_json(opt.output / "manifest.json", manifest);
        backend_lifetime backends;
        auto cuda = ggml_backend_dev_by_name("CUDA0");
        require(cuda && ggml_backend_dev_type(cuda) == GGML_BACKEND_DEVICE_TYPE_GPU, "CUDA0 unavailable; refusing CPU substitution");
        ggml_backend_dev_t devices[] = {cuda, nullptr};
        auto mp = llama_model_default_params(); mp.devices = devices; mp.n_gpu_layers = 99;
        llama_model_ptr model(llama_model_load_from_file(opt.model.c_str(), mp));
        require(bool(model), "CUDA model load failed");
        char architecture[64] {};
        llama_model_meta_val_str(model.get(), "general.architecture", architecture, sizeof(architecture));
        require(std::string(architecture) == "qwen35", "live 2xN256 proof requires Qwen35 dense");
        manifest["architecture"] = architecture;
        manifest["model_layers"] = llama_model_n_layer(model.get());
        manifest["n_gpu_layers_requested"] = 99;
        const auto vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        for (auto token : tokens) require(token < vocab, "development token ID exceeds model vocabulary");
        env_restore restore("GGML_RKVA_WAVEFRONT_TILE");
        env_restore codec_restore("GGML_RKVA_CUDA_FP8");
        env_restore prefetch_restore("GGML_RKVA_TCP_PREFETCH");
        env_restore local_input_restore("GGML_RKVA_LOCAL_INPUT_REUSE");
        env_restore mask_pack_restore("GGML_CUDA_MASK_PACK");
        const auto off = run_arm(model.get(), opt, tokens, false, manifest,
                opt.control_only_pair ? "control-a" : "off");
        const auto on = run_arm(model.get(), opt, tokens, !opt.control_only_pair, manifest,
                opt.control_only_pair ? "control-b" : "on");
        require(off.size() == on.size(), "checkpoint count differs");
        require(!off.empty(), "no actual logits checkpoints were captured");
        manifest["n_checkpoints"] = off.size();
        bool exact = true;
        if (!opt.restore_prefix_slot.empty()) {
            const auto & a = manifest["arms"][opt.control_only_pair ? "control-a" : "off"]["prefix_restore"];
            const auto & b = manifest["arms"][opt.control_only_pair ? "control-b" : "on"]["prefix_restore"];
            manifest["prefix_restore_comparison"] = compare_recurrent_files(
                opt.output / a.at("recurrent_file").get<std::string>(),
                opt.output / b.at("recurrent_file").get<std::string>());
            exact &= manifest["prefix_restore_comparison"].at("recurrent_serialized_bytes_equal").get<bool>();
        }
        for (size_t i = 0; i < off.size(); ++i) {
            require(off[i].label == on[i].label && off[i].logits.size() == on[i].logits.size(), "checkpoint layout differs");
            size_t different = 0; double max_abs = 0;
            for (size_t j = 0; j < off[i].logits.size(); ++j) {
                different += std::memcmp(&off[i].logits[j], &on[i].logits[j], sizeof(float)) != 0;
                max_abs = std::max(max_abs, std::abs(double(off[i].logits[j]) - double(on[i].logits[j])));
            }
            const bool argmax_equal = off[i].metadata["argmax"] == on[i].metadata["argmax"];
            const auto state = compare_recurrent_files(
                opt.output / off[i].metadata.at("recurrent_file").get<std::string>(),
                opt.output / on[i].metadata.at("recurrent_file").get<std::string>());
            const bool state_equal = state.at("recurrent_serialized_bytes_equal").get<bool>();
            exact &= different == 0 && argmax_equal && state_equal;
            manifest["comparisons"].push_back({{"label", off[i].label}, {"different_float_words", different},
                {"max_abs_diagnostic_only", max_abs}, {"argmax_equal", argmax_equal}, {"bitexact", different == 0},
                {"recurrent_serialized_bytes_equal", state_equal}, {"recurrent_different_bytes", state.at("recurrent_different_bytes")},
                {"recurrent_first_difference", state.at("recurrent_first_difference")}});
        }
        manifest["status"] = !exact ? "FAIL_logits_or_recurrent_snapshot_differ" :
            diagnostic ? "diagnostic_bitexact_only_not_WF_qualification" :
            !opt.restore_prefix_slot.empty() ? "frozen_prefix_bitexact_continuation_pass_activation_requires_log_verification" :
            "bitexact_parity_pass_activation_requires_log_verification";
        write_json(opt.output / "manifest.json", manifest);
        std::printf("LIVE-2XN256 finite whole-vocab bitexact=%d checkpoints=%zu diagnostic_only=%d; activation requires arm-scoped log verification; no performance claim\n",
                int(exact), off.size(), int(diagnostic));
        return exact ? 0 : 1;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "LIVE-2XN256 FAIL: %s\n", error.what());
        manifest["status"] = "FAIL"; manifest["error"] = error.what();
        if (output_owned && !opt.output.empty() && fs::is_directory(opt.output)) {
            try { write_json(opt.output / "manifest.json", manifest); } catch (...) {}
        }
        return 1;
    }
}
