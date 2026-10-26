// Manual local-tier CAPI qualification. No model/GPU initialization without --execute.
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
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using json = nlohmann::json;
namespace fs = std::filesystem;
using clock_type = std::chrono::steady_clock;
void need(bool ok, const std::string & message) { if (!ok) throw std::runtime_error(message); }
void env(const char * key, const char * value) {
#ifdef _WIN32
    need(_putenv_s(key, value ? value : "") == 0, "environment update failed");
#else
    need((value ? setenv(key, value, 1) : unsetenv(key)) == 0, "environment update failed");
#endif
}
struct env_guard {
    bool present = std::getenv("GGML_CUDA_MASK_PACK") != nullptr;
    std::string value = present ? std::getenv("GGML_CUDA_MASK_PACK") : "";
    ~env_guard() {
#ifdef _WIN32
        _putenv_s("GGML_CUDA_MASK_PACK", present ? value.c_str() : "");
#else
        if (present) setenv("GGML_CUDA_MASK_PACK", value.c_str(), 1); else unsetenv("GGML_CUDA_MASK_PACK");
#endif
    }
};
void bytes(const fs::path & path, const void * data, size_t n) {
    std::ofstream out(path, std::ios::binary);
    need(bool(out), "cannot open " + path.string());
    if (n) out.write(static_cast<const char *>(data), std::streamsize(n));
    out.close(); need(bool(out), "cannot finish " + path.string());
}
void save_json(const fs::path & path, const json & value) { const auto s = value.dump(2) + "\n"; bytes(path, s.data(), s.size()); }
json read_json(const fs::path & path) { std::ifstream in(path); need(bool(in), "cannot read " + path.string()); json j; in >> j; return j; }
std::vector<llama_token> ids(const fs::path & path) {
    const auto doc = read_json(path);
    const auto & a = doc.is_array() ? doc : doc.at("tokens");
    need(a.is_array() && !a.empty(), "token JSON must be a nonempty array or {tokens:[]}");
    std::vector<llama_token> result;
    for (const auto & item : a) {
        need(item.is_number_integer(), "token ID is not an integer");
        const auto n = item.get<int64_t>();
        need(n >= 0 && n <= std::numeric_limits<llama_token>::max(), "invalid token ID");
        result.push_back(llama_token(n));
    }
    return result;
}
uint32_t number(const std::string & s) {
    size_t used = 0; const auto n = std::stoull(s, &used);
    need(!s.empty() && s.front() != '-' && used == s.size() && n <= UINT32_MAX, "invalid integer: " + s);
    return uint32_t(n);
}
struct options {
    fs::path model, tokens, teacher, out, slot;
    uint32_t context = 8192, prefix = 3584, end = 7680, ubatch = 256, batch = 0;
    bool execute = false, repeat = false, quality_only = false, timing_only = false;
};
options parse(int argc, char ** argv) {
    options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--execute") { o.execute = true; continue; }
        if (a == "--repeat-prompt") { o.repeat = true; continue; }
        if (a == "--quality-only") { o.quality_only = true; continue; }
        if (a == "--timing-only") { o.timing_only = true; continue; }
        if (a == "--help") { std::puts("--model GGUF --tokens JSON --teacher-json JSON --out DIR --ctx N --prefix P --prompt-end E --ubatch 128|256 [--batch SAME] [--prefix-slot FILE] [--repeat-prompt] [--quality-only|--timing-only] --execute"); std::exit(0); }
        need(i + 1 < argc, "missing argument for " + a); const std::string v = argv[++i];
        if (a == "--model") o.model = v;
        else if (a == "--tokens" || a == "--token-json") o.tokens = v;
        else if (a == "--teacher-json") o.teacher = v;
        else if (a == "--out" || a == "--output") o.out = v;
        else if (a == "--prefix-slot") o.slot = v;
        else if (a == "--ctx") o.context = number(v);
        else if (a == "--prefix") o.prefix = number(v);
        else if (a == "--prompt-end") o.end = number(v);
        else if (a == "--ubatch") o.ubatch = number(v);
        else if (a == "--batch") o.batch = number(v);
        else throw std::runtime_error("unknown option " + a);
    }
    if (!o.batch) o.batch = o.ubatch;
    need(o.ubatch == 128 || o.ubatch == 256, "ubatch must be 128 or 256");
    need(o.batch == o.ubatch, "batch must equal ubatch");
    need(o.context <= INT32_MAX && o.prefix > 0 && o.prefix%512 == 0 && o.end > o.prefix && uint64_t(o.end) + 256 <= o.context, "require aligned P>0, E>P and E+256<=ctx");
    need(!o.quality_only || !o.timing_only, "quality-only and timing-only conflict");
    if (o.execute) need(!o.model.empty() && !o.tokens.empty() && !o.teacher.empty() && !o.out.empty(), "missing model/tokens/teacher/out");
    if (o.slot.empty()) o.slot = o.out / "prefix.slot";
    return o;
}
struct backend_scope { backend_scope() { llama_backend_init(); ggml_backend_load_all(); } ~backend_scope() { llama_backend_free(); } };
struct batch_scope {
    llama_batch batch;
    explicit batch_scope(uint32_t n) : batch(llama_batch_init(int32_t(n), 0, 1)) { need(batch.token != nullptr, "batch allocation failed"); }
    ~batch_scope() { llama_batch_free(batch); }
};
llama_context_ptr context(llama_model * model, const options & o) {
    auto p = llama_context_default_params();
    p.n_ctx = o.context; p.n_batch = o.batch; p.n_ubatch = o.ubatch; p.n_seq_max = 1;
    p.n_threads = p.n_threads_batch = 8; p.n_rs_seq = 0; p.ctx_type = LLAMA_CONTEXT_TYPE_DEFAULT;
    p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    p.kvarn = llama_kvarn_params_for_type(LLAMA_KVARN_K4V4_G128);
    p.remote_attn_host = nullptr; p.kv_position_split_p = 0;
    llama_context_ptr result(llama_init_from_model(model, p)); need(bool(result), "local CUDA context creation failed");
    need(llama_n_batch(result.get()) == o.batch && llama_n_ubatch(result.get()) == o.ubatch, "context changed requested batch geometry");
    need(!result->get_cparams().remote_attn_position_split, "unexpected remote context");
    return result;
}
json geometry(llama_context * ctx) {
    auto * h = dynamic_cast<llama_memory_hybrid *>(llama_get_memory(ctx));
    need(h && h->get_mem_attn_base() && h->get_mem_recr(), "expected Qwen35 hybrid memory");
    const auto & p = ctx->get_cparams();
    return {{"n_ctx", llama_n_ctx(ctx)}, {"n_ctx_seq", ctx->n_ctx_seq()}, {"kv_capacity", h->get_mem_attn_base()->get_kv_size()},
        {"batch", llama_n_batch(ctx)}, {"ubatch", llama_n_ubatch(ctx)}, {"physical_ubatch", llama_n_ubatch_physical(ctx)},
        {"tail_tokens", p.kv_tail_tokens}, {"tail_type", int(p.kv_tail_type)}, {"n_rs_seq", p.n_rs_seq}};
}
void decode(llama_context * ctx, batch_scope & b, const llama_token * tokens, uint32_t n, uint32_t pos, bool last_output) {
    b.batch.n_tokens = int32_t(n);
    for (uint32_t i = 0; i < n; ++i) {
        b.batch.token[i] = tokens[i]; b.batch.pos[i] = llama_pos(pos + i); b.batch.n_seq_id[i] = 1;
        b.batch.seq_id[i][0] = 0; b.batch.logits[i] = last_output && i + 1 == n;
    }
    need(llama_decode(ctx, b.batch) == 0, "llama_decode failed at " + std::to_string(pos));
}
void prefill(llama_context * ctx, batch_scope & b, const std::vector<llama_token> & tokens, uint32_t begin, uint32_t end, uint32_t ubatch) {
    for (uint32_t p = begin; p < end;) {
        const uint32_t n = std::min(ubatch, end - p);
        decode(ctx, b, tokens.data() + p, n, p, p + n == end); p += n;
    }
    llama_synchronize(ctx);
    need(llama_get_logits_ith(ctx, -1) != nullptr, "prefill has no final logits");
}
const float * logits(llama_context * ctx, size_t vocab) {
    const auto * p = llama_get_logits_ith(ctx, -1); need(p != nullptr, "missing logits");
    for (size_t i = 0; i < vocab; ++i) need(std::isfinite(p[i]), "nonfinite full-vocabulary logits");
    return p;
}
struct recurrent_writer : llama_io_write_i {
    std::ofstream file;
    std::vector<uint8_t> scratch = std::vector<uint8_t>(4u << 20);
    size_t written = 0;
    explicit recurrent_writer(const fs::path & path) : file(path, std::ios::binary) { need(bool(file), "cannot open recurrent snapshot"); }
    void write(const void * p, size_t n) override {
        need(n <= (size_t(1) << 30) - written, "recurrent snapshot exceeds 1 GiB");
        if (n) file.write(static_cast<const char *>(p), std::streamsize(n));
        need(bool(file), "recurrent snapshot write failed"); written += n;
    }
    void write_tensor(ggml_tensor * tensor, size_t offset, size_t n) override {
        need(tensor && tensor->type == GGML_TYPE_F32 && offset%4 == 0 && n%4 == 0 && offset <= ggml_nbytes(tensor) && n <= ggml_nbytes(tensor) - offset, "invalid pure recurrent tensor");
        while (n) {
            const size_t chunk = std::min(n, scratch.size()); ggml_backend_tensor_get(tensor, scratch.data(), offset, chunk);
            for (size_t i = 0; i < chunk; i += sizeof(float)) { float v; std::memcpy(&v, scratch.data() + i, sizeof(v)); need(std::isfinite(v), "nonfinite recurrent tensor"); }
            write(scratch.data(), chunk); offset += chunk; n -= chunk;
        }
    }
    size_t n_bytes() override { return written; }
    size_t finish() { file.close(); need(bool(file) && written > 0, "incomplete recurrent snapshot"); return written; }
};
json checkpoint(llama_context * ctx, const options & o, const std::string & label, uint32_t step, size_t vocab) {
    llama_synchronize(ctx);
    const std::string stem = label + "-step-" + std::to_string(step);
    const auto * p = logits(ctx, vocab); bytes(o.out / (stem + ".logits.f32"), p, vocab*sizeof(float));
    auto * hybrid = dynamic_cast<llama_memory_hybrid *>(llama_get_memory(ctx));
    need(hybrid && hybrid->get_mem_recr(), "missing pure recurrent owner");
    const auto state = o.out / (stem + ".recurrent.bin");
    try { recurrent_writer writer(state); hybrid->get_mem_recr()->state_write(writer, 0, 0); const size_t count = writer.finish(); need(fs::file_size(state) == count, "recurrent artifact byte count differs");
        return {{"step", step}, {"logits_file", stem + ".logits.f32"}, {"logits_bytes", vocab*sizeof(float)}, {"recurrent_file", state.filename().string()}, {"recurrent_bytes", count}};
    } catch (...) { std::error_code ignored; fs::remove(state, ignored); throw; }
}
bool equal_files(const fs::path & a, const fs::path & b) {
    const auto length = fs::file_size(a); if (length != fs::file_size(b)) return false;
    std::ifstream x(a, std::ios::binary), y(b, std::ios::binary); need(bool(x) && bool(y), "cannot compare artifacts");
    std::array<char, 65536> ax {}, by {};
    for (uint64_t offset = 0; offset < length;) {
        const auto n = std::streamsize(std::min<uint64_t>(ax.size(), length - offset));
        x.read(ax.data(), n); y.read(by.data(), n); need(x.gcount() == n && y.gcount() == n, "truncated comparison artifact");
        if (std::memcmp(ax.data(), by.data(), size_t(n)) != 0) return false; offset += uint64_t(n);
    }
    return true;
}
json perf(llama_context * ctx) { const auto p = llama_perf_context(ctx); return {{"prompt_tokens", p.n_p_eval}, {"decode_tokens", p.n_eval}, {"prompt_ms", p.t_p_eval_ms}, {"decode_ms", p.t_eval_ms}, {"reused_graphs", p.n_reused}, {"reallocs_same", p.n_reallocs_same}, {"reallocs_different", p.n_reallocs_diff}}; }
void restore(llama_context * ctx, const options & o, const std::vector<llama_token> & tokens) {
    size_t count = 0;
    need(llama_state_seq_load_file(ctx, o.slot.string().c_str(), 0, nullptr, 0, &count) > 0 && count == o.prefix, "slot does not contain exact P raw IDs");
    std::vector<llama_token> loaded(count); size_t actual = 0;
    need(llama_state_seq_load_file(ctx, o.slot.string().c_str(), 0, loaded.data(), loaded.size(), &actual) > 0 && actual == o.prefix, "prefix restore failed");
    need(std::equal(loaded.begin(), loaded.end(), tokens.begin()), "restored prefix IDs differ");
    need(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == int64_t(o.prefix) - 1, "restored prefix position differs");
    llama_synchronize(ctx);
}
json run_arm(llama_model * model, const options & o, const std::vector<llama_token> & tokens,
        const std::vector<llama_token> & teacher, const json & effective, const std::string & label, bool off, bool quality, size_t vocab) {
    env("GGML_CUDA_MASK_PACK", off ? "0" : nullptr);
    std::fprintf(stderr, "MASK_PACK_MODEL arm=%s phase=%s mode=%s context_begin\n", label.c_str(), quality ? "quality" : "timing", off ? "0" : "UNSET_DEFAULT"); std::fflush(stderr);
    auto ctx = context(model, o); need(geometry(ctx.get()) == effective, "arm geometry differs from canonical prefix"); restore(ctx.get(), o, tokens);
    batch_scope b(o.ubatch); json record = {{"label", label}, {"phase", quality ? "quality" : "timing"}, {"mask_pack", off ? json("0") : json(nullptr)}, {"geometry", geometry(ctx.get())}, {"restored_ids", o.prefix}};
    llama_perf_context_reset(ctx.get()); const auto begin = clock_type::now();
    prefill(ctx.get(), b, tokens, o.prefix, o.end, o.ubatch);
    const auto end = clock_type::now();
    record["prefill_seconds"] = std::chrono::duration<double>(end - begin).count();
    record["prefill_tps"] = double(o.end - o.prefix)/record["prefill_seconds"].get<double>(); record["prefill_perf"] = perf(ctx.get());
    need(geometry(ctx.get()) == effective, "prefill changed effective geometry"); logits(ctx.get(), vocab);
    if (quality) record["checkpoints"].push_back(checkpoint(ctx.get(), o, label, 0, vocab));
    llama_perf_context_reset(ctx.get()); const auto decode_begin = clock_type::now();
    for (uint32_t i = 0; i < 256; ++i) {
        decode(ctx.get(), b, &teacher[i], 1, o.end + i, true); llama_synchronize(ctx.get());
        need(llama_get_logits_ith(ctx.get(), -1) != nullptr, "teacher step has no logits");
        if (quality && (i == 0 || i == 127 || i == 255)) record["checkpoints"].push_back(checkpoint(ctx.get(), o, label, i + 1, vocab));
    }
    const auto decode_end = clock_type::now();
    if (!quality) { record["teacher_forced_decode_seconds"] = std::chrono::duration<double>(decode_end - decode_begin).count(); record["teacher_forced_decode_tps"] = 256.0/record["teacher_forced_decode_seconds"].get<double>(); }
    record["decode_perf"] = perf(ctx.get()); logits(ctx.get(), vocab);
    need(llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0) == int64_t(o.end) + 255, "final teacher position differs");
    ctx.reset(); // finish context/buffer ownership and print pack counters before arm_end
    std::fprintf(stderr, "MASK_PACK_MODEL arm=%s context_end\n", label.c_str()); std::fflush(stderr);
    return record;
}
} // namespace

int main(int argc, char ** argv) {
    options o; json manifest = {{"status", "starting"}, {"scope", "local CUDA Qwen35 dense target only; n_rs_seq=0; no MTP/speculation/RKVA; teacher-forced decode TPS"}};
    try {
        o = parse(argc, argv);
        if (!o.execute) { std::puts("Dry run: no model/GPU initialization. Add --execute; --help lists arguments."); return 0; }
        need(!fs::exists(o.out) || fs::is_empty(o.out), "output directory must be new or empty");
        fs::create_directories(o.out);
        auto tokens = ids(o.tokens); const auto teacher = ids(o.teacher); need(teacher.size() == 256, "teacher JSON must contain exactly 256 fixed IDs");
        if (tokens.size() < o.end) { need(o.repeat, "prompt is short; explicit --repeat-prompt required"); const auto original = tokens; while (tokens.size() < o.end) tokens.insert(tokens.end(), original.begin(), original.end()); }
        tokens.resize(o.end);
        json environment = json::object();
        for (const char * key : {"GGML_CUDA_GRAPH_RECOVERY_HEADROOM_MB", "GGML_KVARN_WINDOW_CHUNK", "GGML_CUDA_DISABLE_GRAPHS", "LLAMA_KVARN_EXPERIMENTAL_TARGET_TAIL", "GGML_RKVA_LOCAL_INPUT_REUSE"}) { const auto * value = std::getenv(key); environment[key] = value ? json(value) : json(nullptr); }
        const json identity = {{"canonical_path", fs::canonical(o.model).string()}, {"bytes", fs::file_size(o.model)}, {"mtime_ticks", fs::last_write_time(o.model).time_since_epoch().count()}};
        const json audit = {{"format", "mask-pack-local-prefix-v1"}, {"model", identity}, {"ctx", o.context}, {"batch", o.batch}, {"ubatch", o.ubatch}, {"prefix", o.prefix}, {"prefix_ids", std::vector<llama_token>(tokens.begin(), tokens.begin() + o.prefix)}, {"environment", environment}, {"canonical_prefix_mask_pack", "0"}};
        manifest["model"] = identity; manifest["environment"] = environment; manifest["prefix_slot"] = o.slot.string(); manifest["teacher_ids"] = teacher;
        manifest["requested"] = {{"ctx", o.context}, {"prefix", o.prefix}, {"prompt_end", o.end}, {"overflow_tokens", o.end - o.prefix}, {"batch", o.batch}, {"ubatch", o.ubatch}, {"repeat_prompt", o.repeat}};
        save_json(o.out / "manifest.json", manifest);
        const char * wf = std::getenv("GGML_RKVA_WAVEFRONT_TILE");
        need(!wf || std::string(wf) == "0", "local-tier driver requires GGML_RKVA_WAVEFRONT_TILE unset or 0");
        env_guard original_mode; backend_scope backends;
        auto cuda = ggml_backend_dev_by_name("CUDA0"); need(cuda && ggml_backend_dev_type(cuda) == GGML_BACKEND_DEVICE_TYPE_GPU, "CUDA0 required; refusing CPU substitution");
        ggml_backend_dev_t devices[] = {cuda, nullptr}; auto mp = llama_model_default_params(); mp.devices = devices; mp.n_gpu_layers = 99;
        llama_model_ptr model(llama_model_load_from_file(o.model.string().c_str(), mp)); need(bool(model), "model load failed");
        char architecture[64] {}; llama_model_meta_val_str(model.get(), "general.architecture", architecture, sizeof(architecture)); need(std::string(architecture) == "qwen35", "requires Qwen35 dense target");
        const size_t vocab = size_t(llama_vocab_n_tokens(llama_model_get_vocab(model.get()))); manifest["vocab"] = vocab;
        for (auto id : tokens) need(size_t(id) < vocab, "prompt token outside vocabulary"); for (auto id : teacher) need(size_t(id) < vocab, "teacher token outside vocabulary");
        const fs::path sidecar = o.slot.string() + ".json";
        need(fs::exists(o.slot) == fs::exists(sidecar), "unknown/incomplete prefix slot: both native file and audited sidecar required");
        json effective;
        if (fs::exists(o.slot)) { const auto saved = read_json(sidecar); need(saved.at("audit") == audit, "prefix slot audit is incompatible"); need(fs::file_size(o.slot) == saved.at("slot_bytes").get<uint64_t>(), "prefix slot byte count changed"); effective = saved.at("effective_geometry"); manifest["prefix_reused"] = true; }
        else {
            env("GGML_CUDA_MASK_PACK", "0"); std::fprintf(stderr, "MASK_PACK_MODEL arm=prefix-build phase=prefix mode=0 context_begin\n");
            auto ctx = context(model.get(), o); batch_scope b(o.ubatch); prefill(ctx.get(), b, tokens, 0, o.prefix, o.ubatch); effective = geometry(ctx.get());
            need(llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0) == int64_t(o.prefix) - 1, "canonical prefix position differs");
            if (!o.slot.parent_path().empty()) fs::create_directories(o.slot.parent_path());
            const fs::path temporary = o.slot.string() + ".building";
            try { need(llama_state_seq_save_file(ctx.get(), temporary.string().c_str(), 0, tokens.data(), o.prefix) > 0, "prefix save failed"); save_json(sidecar.string() + ".building", {{"audit", audit}, {"effective_geometry", effective}, {"slot_bytes", fs::file_size(temporary)}}); fs::rename(temporary, o.slot); fs::rename(sidecar.string() + ".building", sidecar); }
            catch (...) { std::error_code ignored; fs::remove(temporary, ignored); fs::remove(sidecar.string() + ".building", ignored); throw; }
            ctx.reset(); std::fprintf(stderr, "MASK_PACK_MODEL arm=prefix-build context_end\n"); manifest["prefix_reused"] = false;
        }
        manifest["effective_geometry"] = effective;
        if (!o.timing_only) {
            manifest["quality"].push_back(run_arm(model.get(), o, tokens, teacher, effective, "quality-off", true, true, vocab));
            manifest["quality"].push_back(run_arm(model.get(), o, tokens, teacher, effective, "quality-default", false, true, vocab));
            bool equal = true;
            for (size_t i = 0; i < 4; ++i) for (const char * key : {"logits_file", "recurrent_file"}) {
                const bool same = equal_files(o.out / manifest["quality"][0]["checkpoints"][i][key].get<std::string>(), o.out / manifest["quality"][1]["checkpoints"][i][key].get<std::string>());
                manifest["quality_comparisons"].push_back({{"step", manifest["quality"][0]["checkpoints"][i]["step"]}, {"kind", key}, {"byte_exact", same}}); equal &= same;
            }
            manifest["quality_pass"] = equal; save_json(o.out / "manifest.json", manifest); need(equal, "full-vocabulary logits or pure recurrent state differ");
        }
        if (!o.quality_only) for (int arm = 0; arm < 4; ++arm) {
            const bool off = arm == 0 || arm == 3; manifest["timing"].push_back(run_arm(model.get(), o, tokens, teacher, effective, "timing-" + std::to_string(arm) + (off ? "-off" : "-default"), off, false, vocab)); save_json(o.out / "manifest.json", manifest);
        }
        manifest["status"] = "passed"; save_json(o.out / "manifest.json", manifest); return 0;
    } catch (const std::exception & error) {
        manifest["status"] = "failed"; manifest["error"] = error.what();
        std::error_code ignored;
        if (!o.out.empty() && fs::exists(o.out, ignored)) try { save_json(o.out / "manifest.json", manifest); } catch (...) {}
        std::fprintf(stderr, "mask-pack-model-live: %s\n", error.what()); return 1;
    }
}
