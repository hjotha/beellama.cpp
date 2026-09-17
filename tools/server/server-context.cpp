#include "server-context.h"
#include "server-chat.h"
#include "server-adaptive-dm.h"
#include "server-common.h"
#include "server-http.h"
#include "server-loop-guard.h"
#include "server-task.h"
#include "server-queue.h"
#include "server-schema.h"
#include "server-stream.h"
#include "server-gpu-power.h"
#include "server-model-identity.h"
#include "server-route-state.h"

#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "src/llama-ext.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#define XXH_STATIC_LINKING_ONLY
#include "hash/xxhash/xxhash.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <exception>
#include <fstream>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#ifndef _WIN32
#include <unistd.h> // getpid() for per-writer-unique temp filenames (cross-process atomicity)
#else
#include <process.h> // _getpid()
#define getpid _getpid
#endif

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

constexpr int HTTP_POLLING_SECONDS = 1;

static bool server_reasoning_budget_state_is_reasoning(common_reasoning_budget_state state) {
    return state == REASONING_BUDGET_COUNTING ||
           state == REASONING_BUDGET_WAITING_UTF8 ||
           state == REASONING_BUDGET_FORCING;
}

static bool server_accept_info_is_reasoning(const common_sampler_accept_info & info) {
    return server_reasoning_budget_state_is_reasoning(info.reasoning_state_before) ||
           server_reasoning_budget_state_is_reasoning(info.reasoning_state_after);
}

static std::string router_state_dir_from_env() {
    const char * raw = std::getenv("LLAMA_SERVER_ROUTER_STATE_DIR");
    if (raw == nullptr || raw[0] == '\0') {
        return {};
    }
    const std::filesystem::path dir(raw);
    std::error_code ec;
    if (dir.empty() || !std::filesystem::is_directory(dir, ec) || ec) {
        return {};
    }
    std::string result = dir.string();
    if (!result.empty() && result.back() != std::filesystem::path::preferred_separator) {
        result += std::filesystem::path::preferred_separator;
    }
    return result;
}

static std::string server_loop_guard_reason_to_string(const server_loop_guard_result & result) {
    std::string reason = result.kind.empty() ? "unknown" : result.kind;
    if (result.period > 0) {
        reason += string_format(" period=%d", result.period);
    }
    if (result.coverage > 0) {
        reason += string_format(" coverage=%d", result.coverage);
    }
    if (result.score > 0.0f) {
        reason += string_format(" score=%.3f", (double) result.score);
    }
    return reason;
}

static common_speculative_output_limits server_output_limits(const common_params & params) {
    if (params.embedding ||
            (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED && params.pooling_type != LLAMA_POOLING_TYPE_NONE)) {
        return { params.n_batch, 1 };
    }

    auto result = common_speculative_get_output_limits(
            params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));

    result.total   = std::max<int32_t>(1, result.total);
    result.per_seq = std::max<int32_t>(1, result.per_seq);
    return result;
}

// Test-only fault injection; unset in production. Values: long, mtp,
// long+rollback, mtp+rollback, mtp-allocation or mtp-upload.
static bool adaptive_test_fault(const char * phase, common_context_profile profile) {
    const char * raw = std::getenv("LLAMA_TEST_ADAPTIVE_TRANSITION_FAIL");
    if (raw == nullptr) {
        return false;
    }

    const std::string value(raw);
    if (std::string(phase) == "rollback-after-residency") {
        return value.find("rollback") != std::string::npos &&
            value.find("rollback-context") == std::string::npos;
    }
    if (std::string(phase) == "rollback-after-context") {
        return value.find("rollback-context") != std::string::npos;
    }

    if (std::string(phase) != "candidate-after-residency") {
        return false;
    }
    if (profile == COMMON_CONTEXT_PROFILE_MTP_SHORT) {
        return value.find("mtp-short") != std::string::npos;
    }
    if (profile == COMMON_CONTEXT_PROFILE_MTP) {
        return value.find("mtp") != std::string::npos &&
            value.find("mtp-short") == std::string::npos;
    }
    if (profile == COMMON_CONTEXT_PROFILE_XLONG) {
        return value.find("xlong") != std::string::npos;
    }
    if (profile == COMMON_CONTEXT_PROFILE_XXLONG) {
        return value.find("xxlong") != std::string::npos;
    }
    return value.find("long") != std::string::npos;
}

static llama_mtp_weights_fault adaptive_test_mtp_fault(const char * phase, common_context_profile profile) {
    if ((profile != COMMON_CONTEXT_PROFILE_MTP && profile != COMMON_CONTEXT_PROFILE_MTP_SHORT) ||
            std::string(phase) != "candidate") {
        return llama_mtp_weights_fault::none;
    }
    const char * raw = std::getenv("LLAMA_TEST_ADAPTIVE_TRANSITION_FAIL");
    if (raw == nullptr) {
        return llama_mtp_weights_fault::none;
    }
    const std::string value(raw);
    if (value.find("mtp-allocation") != std::string::npos) {
        return llama_mtp_weights_fault::allocation;
    }
    if (profile == COMMON_CONTEXT_PROFILE_MTP_SHORT) {
        if (value.find("mtp-upload") != std::string::npos || value.find("mtp-short") != std::string::npos) {
            return llama_mtp_weights_fault::upload;
        }
    } else {
        if (value.find("mtp-upload") != std::string::npos ||
                (value.find("mtp") != std::string::npos && value.find("mtp-short") == std::string::npos)) {
            return llama_mtp_weights_fault::upload;
        }
    }
    return llama_mtp_weights_fault::none;
}

// synthetic draft verification for benchmarking - accept draft tokens at random instead of by match with the target
// on replay the draft was already accepted before a context checkpoint restore, so repeat the same decisions
static std::vector<llama_token> server_sample_and_accept_synth(
        common_sampler * smpl,
        llama_context * ctx,
        const std::vector<int32_t> & idxs,
        const llama_tokens & draft,
        const std::vector<double> & synth_probs,
        std::mt19937 & rng,
        bool is_replay) {
    GGML_ASSERT(idxs.size() == draft.size() + 1);
    GGML_ASSERT(synth_probs.size() >= draft.size());

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (size_t i = 0; i < draft.size(); ++i) {
        const llama_token id = common_sampler_sample(smpl, ctx, idxs[i]);
        const bool accept = is_replay || dist(rng) < synth_probs[i];
        // do not accept a drafted EOG token - it would end the generation early
        // on replay the last token is from the target and can be EOG, so skip this check
        if (accept && (is_replay || !llama_vocab_is_eog(vocab, draft[i]))) {
            // synthetic draft tokens do not advance grammar or reasoning state
            // the last replay token is from the target and must advance both
            const bool is_replay_target = is_replay && i + 1 == draft.size();
            common_sampler_accept(smpl, draft[i], is_replay_target);
            result.push_back(draft[i]);
            continue;
        }

        common_sampler_accept(smpl, id, true);
        result.push_back(id);
        return result;
    }

    const llama_token id = common_sampler_sample(smpl, ctx, idxs[draft.size()]);
    common_sampler_accept(smpl, id, true);
    result.push_back(id);

    return result;
}

// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_WAIT_OTHER, // after assigning a task, but waiting for parent slot to process prompt
    SLOT_STATE_STARTED,    // after assigning a task and about to process prompt
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

static server_gpu_power_slot_state server_gpu_power_slot_state_from_slot_state(slot_state state) {
    switch (state) {
        case SLOT_STATE_IDLE:              return server_gpu_power_slot_state::idle;
        case SLOT_STATE_WAIT_OTHER:        return server_gpu_power_slot_state::wait_other;
        case SLOT_STATE_STARTED:           return server_gpu_power_slot_state::started;
        case SLOT_STATE_PROCESSING_PROMPT: return server_gpu_power_slot_state::processing_prompt;
        case SLOT_STATE_DONE_PROMPT:       return server_gpu_power_slot_state::done_prompt;
        case SLOT_STATE_GENERATING:        return server_gpu_power_slot_state::generating;
    }

    return server_gpu_power_slot_state::idle;
}

struct server_slot; // forward declaration

struct server_batch {
    llama_batch batch;
    bool batch_rendered = false;

    struct token {
        int32_t id_slot;
        llama_token token;
        llama_pos pos;
        bool output;
        bool is_prompt; // for stats tracking
    };
    std::vector<token> tokens;
    int32_t n_tokens_alloc = 0;
    int32_t n_embd = 0;

    // track if given slot can be batched with slots already in the batch
    server_slot * slot_batched = nullptr;

    // in embd mode, we temporarily swap out the tokens arr and restore it on clear()
    bool has_embd = false;
    llama_token * tokens_ptr = nullptr;
    std::vector<float> embd;

    float  alora_scale       = -1.0f;
    size_t alora_disabled_id = 0;

    server_batch() {
        batch.pos = nullptr; // sentinel: uninitialized batch
    }

    void free() {
        if (batch.pos != nullptr) {
            clear();
            llama_batch_free(batch);
            batch.pos = nullptr;
            tokens.clear();
            tokens.shrink_to_fit();
            n_tokens_alloc = 0;
            n_embd = 0;
            tokens_ptr = nullptr;
        }
    }

    ~server_batch() {
        free();
    }

    void init(int32_t n_tokens_alloc, int32_t n_embd) {
        free();
        this->n_tokens_alloc = n_tokens_alloc;
        this->n_embd = n_embd;
        batch = llama_batch_init(n_tokens_alloc, 0, 1);
        tokens_ptr = batch.token;
        tokens.reserve(n_tokens_alloc);
    }

    bool add(int32_t id_slot, llama_token token, llama_pos pos, bool output, bool is_prompt) {
        GGML_ASSERT(!has_embd); // cannot mix tokens + embd in same batch
        GGML_ASSERT(batch.pos != nullptr);
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, token, pos, output, is_prompt });
        return true;
    }

    bool add(int32_t id_slot, const std::vector<float> & embd_in, llama_pos pos, bool output, bool is_prompt) {
        GGML_ASSERT(batch.pos != nullptr);
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, LLAMA_TOKEN_NULL, pos, output, is_prompt });
        has_embd = true;
        embd.insert(embd.end(), embd_in.begin(), embd_in.end());
        return true;
    }

    void clear() {
        tokens.clear();
        embd.clear();
        common_batch_clear(batch);
        slot_batched      = nullptr;
        alora_scale       = -1.0f;
        alora_disabled_id = 0;
        batch_rendered    = false;
        has_embd          = false;
        if (batch.token == nullptr) {
            batch.token = tokens_ptr;
            batch.embd  = nullptr;
        }
    }

    int32_t size() const {
        return (int32_t)tokens.size();
    }

    void set_output(int32_t idx, bool output) {
        GGML_ASSERT(idx >= 0 && idx < (int32_t)tokens.size());
        tokens[idx].output = output;
    }

    void render() {
        GGML_ASSERT(!batch_rendered);
        GGML_ASSERT(batch.pos != nullptr);
        common_batch_clear(batch);
        for (int32_t i = 0; i < size(); i++) {
            const auto & t = tokens[i];
            common_batch_add(batch, t.token, t.pos, { t.id_slot }, t.output);
        }
        if (has_embd) {
            batch.token = nullptr; // will be restored on clear()
            batch.embd  = embd.data();
        }
        batch_rendered = true;
    }

    llama_batch get_view(int32_t off, int32_t n_tokens) const {
        GGML_ASSERT(batch.pos != nullptr);
        GGML_ASSERT(batch_rendered);
        GGML_ASSERT(off >= 0 && off < size());
        GGML_ASSERT(n_tokens > 0 && off + n_tokens <= size());

        auto * token = batch.token ? batch.token + off          : nullptr;
        auto * embd  = batch.embd  ? batch.embd  + off * n_embd : nullptr;

        llama_batch view = {
            n_tokens,
            token,
            embd,
            batch.pos      + off,
            batch.n_seq_id + off,
            batch.seq_id   + off,
            batch.logits   + off,
        };

        return view;
    }
};

// --- KV restore-reuse: logits sidecar -------------------------------------------
// When a slot's full state is saved to disk (SLOT_SAVE) for a recurrent/hybrid (FULL) model,
// we additionally persist the last decoded token's full-vocab logits in a small sidecar file
// (<state>.logits). On SLOT_RESTORE of an exact-prompt "regenerate" request, those logits let
// the server emit the first token WITHOUT re-decoding into the (un-rewindable) restored
// recurrent state - which would otherwise crash. The sidecar is independent of libllama's
// state-file format (so that format is left untouched) and is purely best-effort: any
// missing/corrupt/vocab-mismatched sidecar degrades gracefully to the existing behavior.
static constexpr uint32_t SLOT_LOGITS_MAGIC   = 0x474C4B4Cu; // "LKLG" (llama kv logits), LE
static constexpr uint32_t SLOT_LOGITS_VERSION = 1u;

static std::string slot_logits_sidecar_path(const std::string & state_filepath) {
    return state_filepath + ".logits";
}

// Best-effort "touch": bump the mtime of a unified snapshot and optional .logits sidecar to now,
// so a snapshot that is REUSED (read/restored) but never rewritten is treated as
// recently-used by the mtime LRU. Without this, the LRU is least-recently-WRITTEN, which would
// evict a hot base snapshot that N forked requests keep restoring from (it never gets rewritten).
// Never throws and never errors out the caller: every failure is swallowed via error_code (the
// file may have been concurrently evicted by another process; that is harmless here). This only
// runs on a successful restore, off the generation hot path.
static void auto_touch_unit(const std::string & state_filepath) {
    const auto now = std::filesystem::file_time_type::clock::now();
    std::error_code ec;
    std::filesystem::last_write_time(state_filepath, now, ec);
    std::filesystem::last_write_time(slot_logits_sidecar_path(state_filepath), now, ec);
}

// Best-effort write of the logits sidecar. Returns the number of bytes written (0 on failure or
// when there is nothing valid to write). Never throws. The file is written to a temp path and
// atomically renamed so a partial/interrupted write can never leave a corrupt sidecar in place.
// Fields are serialized byte-by-byte little-endian (not a raw struct fwrite) for portability;
// the float payload is documented LE-only, matching llama.cpp's native-LE state-file contract.
static size_t slot_logits_write(const std::string & state_filepath,
                                const std::vector<float> & logits,
                                int32_t n_vocab,
                                uint32_t n_tokens) {
    if (logits.empty() || (int32_t) logits.size() != n_vocab || n_vocab <= 0) {
        return 0;
    }
    const std::string sidecar = slot_logits_sidecar_path(state_filepath);
    static std::atomic<uint64_t> s_logits_nonce{0};
    const std::string tmp = sidecar + ".tmp-" + std::to_string((long) getpid()) + "-" +
        std::to_string(s_logits_nonce.fetch_add(1, std::memory_order_relaxed));

    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f) {
        return 0;
    }
    auto put_u32 = [&](uint32_t v) {
        const unsigned char b[4] = {
            (unsigned char)( v        & 0xFF),
            (unsigned char)((v >> 8)  & 0xFF),
            (unsigned char)((v >> 16) & 0xFF),
            (unsigned char)((v >> 24) & 0xFF),
        };
        f.write((const char *) b, 4);
    };
    put_u32(SLOT_LOGITS_MAGIC);
    put_u32(SLOT_LOGITS_VERSION);
    put_u32((uint32_t) n_vocab);
    put_u32(n_tokens);
    f.write((const char *) logits.data(), (std::streamsize) logits.size() * sizeof(float));
    f.flush();
    if (!f.good()) {
        f.close();
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
        return 0;
    }
    f.close();
    std::error_code ec;
    std::filesystem::rename(tmp, sidecar, ec); // atomic replace
    if (ec) {
        std::filesystem::remove(tmp, ec);
        return 0;
    }
    return 16 + logits.size() * sizeof(float);
}

// Read a logits sidecar. Returns true and fills `out` (size n_vocab) iff a valid sidecar exists
// whose vocab matches `expect_n_vocab` AND whose recorded token count matches `expect_n_tokens`
// (the count of the state just restored). The token-count check is AUTHORITATIVE: it binds the
// sidecar to the exact state it was saved against, so a sidecar that was somehow written for a
// different state length can never be reused. Any mismatch / short read / missing file => false
// with `out` cleared, so the caller falls back to existing behavior. Never throws.
static bool slot_logits_read(const std::string & state_filepath,
                             int32_t expect_n_vocab,
                             uint32_t expect_n_tokens,
                             std::vector<float> & out) {
    out.clear();
    if (expect_n_vocab <= 0) {
        return false;
    }
    const std::string sidecar = slot_logits_sidecar_path(state_filepath);
    std::ifstream f(sidecar, std::ios::binary);
    if (!f) {
        return false;
    }
    auto get_u32 = [&](uint32_t & v) -> bool {
        unsigned char b[4];
        f.read((char *) b, 4);
        if (f.gcount() != 4) {
            return false;
        }
        v = (uint32_t) b[0] | ((uint32_t) b[1] << 8) | ((uint32_t) b[2] << 16) | ((uint32_t) b[3] << 24);
        return true;
    };
    uint32_t magic = 0, version = 0, n_vocab = 0, n_tokens = 0;
    if (!get_u32(magic) || !get_u32(version) || !get_u32(n_vocab) || !get_u32(n_tokens)) {
        return false;
    }
    if (magic != SLOT_LOGITS_MAGIC || version != SLOT_LOGITS_VERSION ||
        (int32_t) n_vocab != expect_n_vocab || n_tokens != expect_n_tokens) {
        return false;
    }
    out.resize(n_vocab);
    const std::streamsize want = (std::streamsize) n_vocab * (std::streamsize) sizeof(float);
    f.read((char *) out.data(), want);
    if (f.gcount() != want) {
        out.clear();
        return false;
    }
    bool has_finite = false;
    for (float logit : out) {
        if (std::isnan(logit) || logit == INFINITY) {
            out.clear();
            return false;
        }
        has_finite = has_finite || std::isfinite(logit);
    }
    if (!has_finite) {
        out.clear();
        return false;
    }
    return true;
}

// --- KV restore-reuse: bounded unified snapshot store ----------------------------
// One snapshot = a canonical state file plus its optional <name>.logits sidecar.
// Legacy .meta sidecars are still accounted with their old state during migration,
// but new route and auto snapshots use the same native/footer file.
struct slot_save_unit {
    std::string state_path;
    std::string sidecar_path; // "<state>.logits", "" if none
    std::string meta_path;    // legacy "<state>.meta", "" if none
    uintmax_t   bytes = 0;
    std::filesystem::file_time_type mtime;
};

// A pre-publication budget reservation for the unified store. It holds
// exclusive eviction locks on the exact victims while the state writer holds
// the store lock, so another publisher cannot change the decision and a
// failed publication never evicts an older valid snapshot.
struct unified_snapshot_limit_plan {
    std::string dir;
    std::string just_written;
    int32_t max_count = 0;
    int64_t max_bytes = 0;
    std::vector<slot_save_unit> victims;
    std::vector<std::unique_ptr<server_route_state_lease>> locks;

    bool prepare(uint64_t incoming_bytes) {
        if (max_count <= 0 && max_bytes <= 0) {
            return true;
        }
        const auto reject = [this](const char * reason) {
            SRV_WRN("unified snapshot budget preflight rejected: %s dir=%s\n", reason, dir.c_str());
            return false;
        };
        std::error_code ec;
        std::vector<slot_save_unit> units;
        std::set<std::string> present;
        for (std::filesystem::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
            std::error_code fec;
            if (it->is_regular_file(fec) && !fec) {
                present.insert(it->path().string());
            } else if (fec) {
                return reject("directory entry inspection failed");
            }
        }
        if (ec) {
            return reject("directory enumeration failed");
        }
        ec.clear();
        for (const auto & path : present) {
            if (path.empty()) {
                continue;
            }
            if ((path.size() >= 4 && path.compare(path.size() - 4, 4, ".tmp") == 0) ||
                    path.find(".tmp-") != std::string::npos ||
                    (path.size() >= 6 && path.compare(path.size() - 6, 6, ".lease") == 0) ||
                    (path.size() >= 5 && path.compare(path.size() - 5, 5, ".lock") == 0) ||
                    (path.size() >= 4 && path.compare(path.size() - 4, 4, ".ref") == 0)) {
                continue;
            }
            if (path.size() >= 7 && path.compare(path.size() - 7, 7, ".logits") == 0 &&
                    present.count(path.substr(0, path.size() - 7))) {
                continue;
            }
            if (path.size() >= 5 && path.compare(path.size() - 5, 5, ".meta") == 0 &&
                    present.count(path.substr(0, path.size() - 5))) {
                continue;
            }
            slot_save_unit unit;
            unit.state_path = path;
            unit.bytes = std::filesystem::file_size(path, ec);
            if (ec) {
                return reject("snapshot size inspection failed");
            }
            const std::string logits = path + ".logits";
            if (present.count(logits)) {
                const auto bytes = std::filesystem::file_size(logits, ec);
                if (ec || bytes > UINTMAX_MAX - unit.bytes) {
                    return reject("snapshot logits size inspection failed");
                }
                unit.sidecar_path = logits;
                unit.bytes += bytes;
                ec.clear();
            }
            const std::string meta = path + ".meta";
            if (present.count(meta)) {
                const auto bytes = std::filesystem::file_size(meta, ec);
                if (ec || bytes > UINTMAX_MAX - unit.bytes) {
                    return reject("snapshot metadata size inspection failed");
                }
                unit.meta_path = meta;
                unit.bytes += bytes;
                ec.clear();
            }
            unit.mtime = std::filesystem::last_write_time(path, ec);
            if (ec) {
                return reject("snapshot timestamp inspection failed");
            }
            units.push_back(std::move(unit));
        }

        size_t count = units.size();
        uintmax_t total = 0;
        bool replacing = false;
        uintmax_t replaced_bytes = 0;
        for (const auto & unit : units) {
            if (unit.bytes > UINTMAX_MAX - total) {
                return reject("snapshot byte total overflow");
            }
            total += unit.bytes;
            if (unit.state_path == just_written) {
                replacing = true;
                replaced_bytes = unit.bytes;
            }
        }
        if (replacing) {
            count = count > 0 ? count - 1 : 0;
            total -= std::min(total, replaced_bytes);
        }
        ++count; // the incoming published snapshot
        if (incoming_bytes > UINT64_MAX - total) {
            return reject("incoming snapshot byte total overflow");
        }
        total += incoming_bytes;
        if (max_bytes > 0 && incoming_bytes > (uint64_t) max_bytes) {
            return reject("incoming snapshot exceeds byte cap");
        }

        std::sort(units.begin(), units.end(), [](const slot_save_unit & a, const slot_save_unit & b) {
            return a.mtime < b.mtime;
        });
        for (const auto & unit : units) {
            if ((max_count <= 0 || count <= (size_t) max_count) &&
                    (max_bytes <= 0 || total <= (uintmax_t) max_bytes)) {
                break;
            }
            if (unit.state_path == just_written) {
                continue; // replacement is accounted above, never delete the old inode
            }
            auto lock = std::make_unique<server_route_state_lease>(unit.state_path,
                    server_route_state_lock_mode::eviction);
            if (!lock->acquired()) {
                continue; // live reader/publication: safe protected candidate
            }
            locks.push_back(std::move(lock));
            victims.push_back(unit);
            count = count > 0 ? count - 1 : 0;
            total -= std::min(total, (uintmax_t) unit.bytes);
        }
        const bool fits = (max_count <= 0 || count <= (size_t) max_count) &&
            (max_bytes <= 0 || total <= (uintmax_t) max_bytes);
        if (!fits) {
            victims.clear();
            locks.clear();
            return reject("all remaining candidates are protected or limits cannot fit");
        }
        return true;
    }

    bool commit() noexcept {
        bool ok = true;
        for (const auto & unit : victims) {
            std::error_code ec;
            std::filesystem::remove(unit.state_path, ec);
            if (ec) {
                SRV_WRN("unified snapshot eviction failed for %s: %s\n", unit.state_path.c_str(), ec.message().c_str());
                ok = false;
                continue;
            }
            if (std::filesystem::exists(unit.state_path, ec) || ec) {
                ok = false;
            }
            if (!unit.sidecar_path.empty()) {
                std::filesystem::remove(unit.sidecar_path, ec);
                if (ec) { ok = false; }
            }
            if (!unit.meta_path.empty()) {
                std::filesystem::remove(unit.meta_path, ec);
                if (ec) { ok = false; }
            }
        }
        victims.clear();
        locks.clear();
        return ok;
    }
};

// Enforce --slot-save-max-count / --slot-save-max-bytes over `dir` using LRU-by-mtime eviction.
// `just_written` is the state path that was just saved: it is never evicted, but if it ALONE
// exceeds the byte cap it is deleted (with its sidecar) and `oversized` is set so the caller can
// reject the save rather than evict everything else. Operates strictly within `dir`; uses only
// the error_code std::filesystem overloads so it never throws across the server loop.
//
    // IMPORTANT: when a cap is set, --slot-save-path is treated as a server-owned store - any regular
    // file in it (other than recognized sidecars, temporary files and lock identities) is an eviction
// candidate. Point --slot-save-max-count/-mb at a DEDICATED directory; do not mix unrelated files
// into the slot-save directory. (With both caps explicitly set to zero, nothing
// is ever deleted and the directory is left exactly as before.)
// `just_written` is the exact filepath string the server built as `slot_save_path + filename`;
// directory_iterator(dir) over that same `slot_save_path` yields identically-spelled path strings
// on POSIX (the production target), so raw string equality correctly identifies the just-saved
// unit. (Not used on Windows in practice; if ever needed there, switch to filename comparison.)
static void slot_save_enforce_limits(const std::string & dir,
                                     int32_t max_count, int64_t max_bytes,
                                     const std::string & just_written,
                                     bool & oversized) {
    oversized = false;
    if (max_count <= 0 && max_bytes <= 0) {
        return; // both unlimited
    }

    std::error_code ec;
    std::vector<slot_save_unit> units;
    uintmax_t this_unit_bytes = 0;

    // First pass: enumerate every regular file once and record the full set of paths so we can
    // tell a real sidecar (sibling of a state file we wrote) from a state file a client happened
    // to name "foo.logits". We must NOT blindly skip every "*.logits" - fs_validate_filename
    // allows that suffix, so a state file literally named "foo.logits" would otherwise escape both
    // caps entirely. Only "<X>.logits" where "<X>" also exists is treated as a sidecar.
    std::vector<std::string> all_files;
    {
        std::set<std::string> present;
        for (std::filesystem::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
            std::error_code fec;
            if (!it->is_regular_file(fec) || fec) {
                continue;
            }
            all_files.push_back(it->path().string());
            present.insert(all_files.back());
        }

        for (const std::string & p : all_files) {
            std::error_code fec;
            // In-flight temporary and stable lock-identity files are never counted.
            if ((p.size() >= 4 && p.compare(p.size() - 4, 4, ".tmp") == 0) ||
                    p.find(".tmp-") != std::string::npos ||
                    (p.size() >= 6 && p.compare(p.size() - 6, 6, ".lease") == 0) ||
                    (p.size() >= 5 && (p.compare(p.size() - 5, 5, ".lock") == 0 ||
                                       p.compare(p.size() - 4, 4, ".ref") == 0))) {
                continue;
            }
            // a "<X>.logits" file is a sidecar ONLY when its state file "<X>" is also present;
            // accounted together with that state file below, so skip it here.
            if (p.size() >= 7 && p.compare(p.size() - 7, 7, ".logits") == 0 &&
                present.count(p.substr(0, p.size() - 7))) {
                continue;
            }
            // a "<X>.meta" file is the auto disk cache's tokens+fingerprint sidecar; treat it
            // exactly like ".logits" - accounted with its state file below, reaped if orphaned.
            if (p.size() >= 5 && p.compare(p.size() - 5, 5, ".meta") == 0 &&
                present.count(p.substr(0, p.size() - 5))) {
                continue;
            }
            slot_save_unit u;
            u.state_path = p;
            u.bytes = std::filesystem::file_size(p, fec);
            if (fec) {
                continue;
            }
            const std::string side = p + ".logits";
            if (present.count(side)) {
                const auto sb = std::filesystem::file_size(side, fec);
                if (!fec) {
                    u.sidecar_path = side;
                    u.bytes += sb;
                }
            }
            const std::string meta = p + ".meta";
            if (present.count(meta)) {
                const auto mb = std::filesystem::file_size(meta, fec);
                if (!fec) {
                    u.meta_path = meta;
                    u.bytes += mb;
                }
            }
            u.mtime = std::filesystem::last_write_time(p, fec);
            if (fec) {
                continue;
            }

            if (p == just_written) {
                this_unit_bytes = u.bytes;
            }
            units.push_back(std::move(u));
        }
    }

    // a single snapshot larger than the byte cap is rejected: delete only the just-written unit,
    // do NOT cascade-evict every other (valid) snapshot to make room for something that can't fit.
    // NOTE: intentionally a no-op when max_bytes == 0 (byte cap disabled); in count-only mode an
    // individual snapshot's size is never bounded - only --slot-save-max-mb bounds per-snapshot size.
    if (max_bytes > 0 && this_unit_bytes > (uintmax_t) max_bytes) {
        for (const auto & u : units) {
            if (u.state_path == just_written) {
                server_route_state_remove_if_unreferenced(u.state_path);
                break;
            }
        }
        oversized = true;
        return;
    }

    std::sort(units.begin(), units.end(),
              [](const slot_save_unit & a, const slot_save_unit & b) { return a.mtime < b.mtime; }); // oldest first

    size_t    count = units.size();
    uintmax_t total = 0;
    for (const auto & u : units) {
        total += u.bytes;
    }
    size_t idx = 0;

    auto evict_oldest = [&]() -> bool {
        while (idx < units.size()) {
            const auto & u = units[idx++];
            if (u.state_path == just_written) {
                continue; // the writer owns the newly published snapshot
            }
            // Acquire the same kernel-coordinated publication+reference lock
            // used by readers and writers. A dead process releases its lock;
            // a live reader makes this candidate simply ineligible for now.
            if (!server_route_state_remove_if_unreferenced(u.state_path)) {
                SRV_DBG("unified snapshot eviction skipped while protected: %s\n", u.state_path.c_str());
                continue;
            }
            total -= std::min(total, (uintmax_t) u.bytes);
            count = (count > 0) ? count - 1 : 0;
            return true;
        }
        return false;
    };

    if (max_count > 0) {
        while (count > (size_t) max_count) {
            if (!evict_oldest()) {
                break;
            }
        }
    }
    if (max_bytes > 0) {
        while (total > (uintmax_t) max_bytes) {
            if (!evict_oldest()) {
                break;
            }
        }
    }
    // A live reader/reference can temporarily make the configured budget
    // impossible. Report rejection so the publisher can remove its own new
    // snapshot after releasing its reference, rather than growing forever.
    if ((max_count > 0 && count > (size_t) max_count) ||
            (max_bytes > 0 && total > (uintmax_t) max_bytes)) {
        oversized = true;
    }
}

// ---------------------------------------------------------------------------
// --- Auto disk prompt/KV cache (opt-in: --slot-save-auto) ---
//
// Persists per-slot KV snapshots to disk as the canonical native/footer
// envelope (tokens, identity, layout and optional embedded logits) and indexes
// them by a chained hash over token IDs, so a cold process can reuse a warm
// process's KV with no client/router involvement. Legacy .meta/.logits pairs
// are read only for validated one-time adoption; new saves do not create them.
//
// Design invariants (all must hold; comments below reference them by number):
//   1. Off by default: every hook's FIRST statement is auto_cache_enabled(); when
//      false there is no scan, index, hashing, or allocation - behavior is unchanged.
//   2. Never restore on hash alone: the snapshot's token-ID array must byte-compare
//      equal to the request prefix before any restore (collision-safe).
//   3. Model identity: each snapshot carries a fingerprint (model/vocab/ctx/rope/
//      KV-type/FULL-vs-attention/LoRA); a mismatch refuses the restore.
//   4. Fallback totality: any failure (corrupt file, fp/vocab mismatch, IO error,
//      no match) falls back to a normal prefill - never crash, never wrong output.
//   5. Hot-path purity: the multi-GB save runs only at a prompt branch boundary
//      (before the final prompt token) or slot release/reassign, never between
//      sampled generation tokens; restore happens once before prefill.
//
// Concurrency: all slot work runs on the single server-loop thread, so the index is
// single-threaded and the mutex below is uncontended today; it becomes load-bearing
// only if the save I/O is later moved to a worker thread (do not make save async
// without keeping the mutex honest). Independent of legacy --prompt-cache and the
// in-memory prefix-reuse path; auto-restore fires only when in-memory reuse is poor.
// ---------------------------------------------------------------------------

static constexpr uint32_t SLOT_META_MAGIC   = 0x544D4B4Cu; // "LKMT" (llama kv meta), LE
static constexpr uint32_t SLOT_META_VERSION = 1u;

// Model/quant/context fingerprint that MUST match for a restore to be sound. All
// fields are stable inference-affecting identity captured once at model load and
// compared by exact equality (pure-CPU int compares). See invariant 3. The blob
// produced by llama_state_seq_save_file is only safe to load into a context with
// identical KV geometry - a Q4_0-KV blob loaded into an F16 ctx, or a different
// rope/yarn scale (positions are baked into the saved state), silently corrupts -
// so cache_type_k/v and rope_scale are NOT optional.
struct model_fp {
    uint64_t fp_model      = 0; // hash of llama_model_desc + size + n_params (+ n_embd/n_layer)
    uint32_t fp_n_vocab    = 0;
    uint32_t fp_n_ctx_train= 0;
    uint32_t fp_n_embd     = 0;
    uint32_t fp_n_layer    = 0;
    uint32_t fp_rope_type  = 0;
    uint32_t fp_cache_k    = 0; // ggml_type of K cache (enum int)
    uint32_t fp_cache_v    = 0; // ggml_type of V cache (enum int)
    uint32_t fp_n_ctx      = 0; // effective per-seq n_ctx
    uint32_t fp_kv_full    = 0; // 1 if COMMON_CONTEXT_SEQ_RM_TYPE_FULL else 0
    uint32_t fp_block      = 0; // slot_save_block this snapshot was hashed with
    uint64_t fp_rope_scale = 0; // bit-pattern of effective rope_freq_scale (position-critical)
    // rope_freq_base and ALL YaRN params also bake positions into the saved KV state exactly as
    // rope_freq_scale does - a same-model run differing only in --rope-freq-base or any --yarn-*
    // flag would otherwise pass the fingerprint and silently restore positionally-corrupt state.
    // All are bit-cast (float->u32) into identity; yarn_orig_ctx is an int. "0/negative = use
    // model-trained value" is normalized in auto_compute_fingerprint so equal effective configs match.
    uint64_t fp_rope_base       = 0; // bit-pattern of effective rope_freq_base
    uint32_t fp_yarn_ext        = 0; // bit-pattern of yarn_ext_factor
    uint32_t fp_yarn_attn       = 0; // bit-pattern of yarn_attn_factor
    uint32_t fp_yarn_beta_fast  = 0; // bit-pattern of yarn_beta_fast
    uint32_t fp_yarn_beta_slow  = 0; // bit-pattern of yarn_beta_slow
    uint32_t fp_yarn_orig_ctx   = 0; // yarn_orig_ctx (int)
    uint64_t fp_lora       = 0; // hash of active LoRA-set ids+scales (0 if none)
    // refuse cross-shape restores: 1 if the server was launched with --mmproj (mctx != nullptr),
    // else 0. The auto-cache only ever persists text-only prefixes, but mmproj-aware rope (M-RoPE)
    // and projector wiring CAN alter the text KV layout, so we conservatively REFUSE to cross-load
    // a text-only-server snapshot into an mmproj server (or vice-versa) - they get disjoint stores.
    // Removing this bit later would require proving the text KV layout is identical across the two
    // deployment shapes.
    uint32_t fp_mmproj_loaded   = 0;

    // exact field-by-field equality (C++17: no defaulted operator==). Any difference REFUSES the
    // restore (invariant 3). Note: fp_block is intentionally part of identity - a snapshot hashed
    // with a different block size cannot be longest-prefix-matched against the current index.
    bool operator==(const model_fp & o) const {
        return fp_model == o.fp_model && fp_n_vocab == o.fp_n_vocab &&
               fp_n_ctx_train == o.fp_n_ctx_train && fp_n_embd == o.fp_n_embd &&
               fp_n_layer == o.fp_n_layer && fp_rope_type == o.fp_rope_type &&
               fp_cache_k == o.fp_cache_k && fp_cache_v == o.fp_cache_v &&
               fp_n_ctx == o.fp_n_ctx && fp_kv_full == o.fp_kv_full &&
               fp_block == o.fp_block && fp_rope_scale == o.fp_rope_scale &&
               fp_rope_base == o.fp_rope_base && fp_yarn_ext == o.fp_yarn_ext &&
               fp_yarn_attn == o.fp_yarn_attn && fp_yarn_beta_fast == o.fp_yarn_beta_fast &&
               fp_yarn_beta_slow == o.fp_yarn_beta_slow && fp_yarn_orig_ctx == o.fp_yarn_orig_ctx &&
               fp_lora == o.fp_lora && fp_mmproj_loaded == o.fp_mmproj_loaded;
    }
};

// 64-bit chained block hash over token IDs. Each token folds via FNV-1a then a
// splitmix avalanche; block k's output seeds block k+1, so the hash at every block
// boundary commits to the ENTIRE prefix [0, (k+1)*B). Collision resistance is only
// a candidate-narrowing accelerator: we NEVER trust it alone (invariant 2) - the
// caller byte-verifies tokens before any restore. Block boundaries are the only
// resumable prefix lengths (vLLM-APC / SGLang-radix granularity).
static inline uint64_t auto_hash_mix(uint64_t h, int32_t tok) {
    h ^= (uint64_t) (uint32_t) tok;
    h *= 0x100000001b3ULL;                                  // FNV-1a 64-bit prime
    h ^= h >> 29; h *= 0xbf58476d1ce4e5b9ULL; h ^= h >> 32; // splitmix64 finalize
    return h;
}

// Returns, for each block boundary b in [1 .. n/B], the cumulative chain hash
// committing to tokens[0 .. b*B). out[k] = hash of prefix length (k+1)*B. The
// chain is salted with `salt` (the model fingerprint hash) so two different models
// can never produce the same boundary hash for identical tokens. A trailing
// partial block is NOT a boundary (only whole-block prefixes are index keys).
static std::vector<uint64_t> auto_block_hashes(const llama_tokens & toks, int B, uint64_t salt) {
    std::vector<uint64_t> out;
    if (B <= 0) {
        return out;
    }
    out.reserve(toks.size() / (size_t) B);
    uint64_t h = 0xcbf29ce484222325ULL ^ salt; // FNV offset basis, fingerprint-salted
    for (size_t i = 0; i < toks.size(); ++i) {
        h = auto_hash_mix(h, toks[i]);
        if ((i + 1) % (size_t) B == 0) {
            out.push_back(h);
        }
    }
    return out;
}

static constexpr uint32_t AUTO_INDEX_DEFAULT_BLOCK = 256;

static inline uint32_t auto_index_block(uint32_t block) {
    return block > 0 ? block : AUTO_INDEX_DEFAULT_BLOCK;
}

static inline uint64_t auto_boundary_key(uint32_t block, uint64_t boundary) {
    return auto_hash_mix(boundary, (int32_t) auto_index_block(block));
}

// A snapshot can share a block boundary with other branches.
struct auto_cache_entry {
    std::string state_path; // full canonical unified snapshot path
    uint32_t    n_tokens = 0;
    uint32_t    index_block = AUTO_INDEX_DEFAULT_BLOCK;
    std::string model;
    std::string layout;
};

// Keep all candidates for each boundary; select by verified token prefix.
struct auto_cache_index {
    std::mutex mtx;
    std::unordered_multimap<uint64_t, auto_cache_entry> by_boundary;
    std::unordered_set<std::string> indexed_files;    // state paths already scanned (incremental refresh)
    std::set<uint32_t> index_blocks;                  // blocks found in the shared store
    std::filesystem::file_time_type dir_mtime{};      // dir mtime as of the last scan
    std::chrono::steady_clock::time_point last_refresh{}; // throttle: skip stat storms in a burst
};

// Cross-process refresh throttle: at most one dir-mtime stat per this interval on the hot lookup
// path (a forced refresh on a lookup miss bypasses it). Sub-second so a peer's new snapshot is
// visible within ~1 prefill of being written - effectively immediate from the user's view.
static constexpr int AUTO_REFRESH_MIN_MS = 1000;

// Sidecar path twins for an auto snapshot's state file. `.logits` is the committed
// (byte-identical) regenerate sidecar; `.meta` is the NEW tokens+fingerprint
// sidecar this feature adds so the startup scan / pre-restore verify reads only a
// tiny file, never the multi-GB state.
static std::string slot_meta_sidecar_path(const std::string & state_filepath) {
    return state_filepath + ".meta";
}

// Best-effort atomic write of the .meta sidecar (LE, temp+rename - the exact idiom
// of slot_logits_write). Layout: magic/version, fingerprint fields, tok_count,
// chain_hash, then int32 tokens[tok_count]. Returns true on success. Never throws.
static bool slot_meta_write(const std::string & state_filepath,
                            const model_fp & fp,
                            const llama_tokens & toks,
                            uint64_t chain_hash) {
    const std::string sidecar = slot_meta_sidecar_path(state_filepath);
    const std::string tmp     = sidecar + ".tmp";

    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f) {
        return false;
    }
    auto put_u32 = [&](uint32_t v) {
        const unsigned char b[4] = {
            (unsigned char)( v        & 0xFF),
            (unsigned char)((v >> 8)  & 0xFF),
            (unsigned char)((v >> 16) & 0xFF),
            (unsigned char)((v >> 24) & 0xFF),
        };
        f.write((const char *) b, 4);
    };
    auto put_u64 = [&](uint64_t v) {
        put_u32((uint32_t)(v & 0xFFFFFFFFu));
        put_u32((uint32_t)(v >> 32));
    };
    put_u32(SLOT_META_MAGIC);
    put_u32(SLOT_META_VERSION);
    put_u64(fp.fp_model);
    put_u32(fp.fp_n_vocab);
    put_u32(fp.fp_n_ctx_train);
    put_u32(fp.fp_n_embd);
    put_u32(fp.fp_n_layer);
    put_u32(fp.fp_rope_type);
    put_u32(fp.fp_cache_k);
    put_u32(fp.fp_cache_v);
    put_u32(fp.fp_n_ctx);
    put_u32(fp.fp_kv_full);
    put_u32(fp.fp_block);
    put_u64(fp.fp_rope_scale);
    // rope_freq_base + YaRN fingerprint fields
    put_u64(fp.fp_rope_base);
    put_u32(fp.fp_yarn_ext);
    put_u32(fp.fp_yarn_attn);
    put_u32(fp.fp_yarn_beta_fast);
    put_u32(fp.fp_yarn_beta_slow);
    put_u32(fp.fp_yarn_orig_ctx);
    put_u64(fp.fp_lora);
    // mmproj deployment-shape bit - refuses cross-shape restores.
    put_u32(fp.fp_mmproj_loaded);
    put_u32((uint32_t) toks.size());
    put_u64(chain_hash);
    // token IDs as raw LE int32 (llama_token == int32_t; llama.cpp's on-disk
    // contract is native-LE, matching slot_logits_write's float payload).
    f.write((const char *) toks.data(), (std::streamsize) toks.size() * sizeof(int32_t));
    f.flush();
    if (!f.good()) {
        f.close();
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
        return false;
    }
    f.close();
    std::error_code ec;
    std::filesystem::rename(tmp, sidecar, ec); // atomic replace
    if (ec) {
        std::filesystem::remove(tmp, ec);
        return false;
    }
    return true;
}

// Read a .meta sidecar. Returns true and fills `fp_out` + `toks_out` iff a valid
// sidecar exists. Any short read / bad magic / version mismatch => false with
// outputs cleared (invariant 4). Never throws. Note: `chain_hash` is recorded for
// debuggability but the authority for reuse is always the byte-compared tokens.
static bool slot_meta_read(const std::string & state_filepath,
                           model_fp & fp_out,
                           llama_tokens & toks_out, uint32_t max_tokens) {
    fp_out = model_fp{};
    toks_out.clear();
    const std::string sidecar = slot_meta_sidecar_path(state_filepath);
    std::ifstream f(sidecar, std::ios::binary);
    if (!f) {
        return false;
    }
    auto get_u32 = [&](uint32_t & v) -> bool {
        unsigned char b[4];
        f.read((char *) b, 4);
        if (f.gcount() != 4) {
            return false;
        }
        v = (uint32_t) b[0] | ((uint32_t) b[1] << 8) | ((uint32_t) b[2] << 16) | ((uint32_t) b[3] << 24);
        return true;
    };
    auto get_u64 = [&](uint64_t & v) -> bool {
        uint32_t lo = 0, hi = 0;
        if (!get_u32(lo) || !get_u32(hi)) {
            return false;
        }
        v = (uint64_t) lo | ((uint64_t) hi << 32);
        return true;
    };
    uint32_t magic = 0, version = 0;
    if (!get_u32(magic) || !get_u32(version)) {
        return false;
    }
    if (magic != SLOT_META_MAGIC || version != SLOT_META_VERSION) {
        return false;
    }
    model_fp fp;
    uint32_t tok_count = 0;
    uint64_t chain_hash = 0;
    if (!get_u64(fp.fp_model)         || !get_u32(fp.fp_n_vocab)       || !get_u32(fp.fp_n_ctx_train) ||
        !get_u32(fp.fp_n_embd)        || !get_u32(fp.fp_n_layer)       || !get_u32(fp.fp_rope_type)   ||
        !get_u32(fp.fp_cache_k)       || !get_u32(fp.fp_cache_v)       || !get_u32(fp.fp_n_ctx)       ||
        !get_u32(fp.fp_kv_full)       || !get_u32(fp.fp_block)         || !get_u64(fp.fp_rope_scale)  ||
        // rope_freq_base + YaRN - must be read in the same order slot_meta_write emits.
        !get_u64(fp.fp_rope_base)     || !get_u32(fp.fp_yarn_ext)      || !get_u32(fp.fp_yarn_attn)   ||
        !get_u32(fp.fp_yarn_beta_fast)|| !get_u32(fp.fp_yarn_beta_slow)|| !get_u32(fp.fp_yarn_orig_ctx)||
        // mmproj deployment-shape bit - read in the same order slot_meta_write emits.
        !get_u64(fp.fp_lora)          || !get_u32(fp.fp_mmproj_loaded) ||
        !get_u32(tok_count)           || !get_u64(chain_hash)) {
        return false;
    }
    (void) chain_hash;
    const auto payload_start = f.tellg();
    f.seekg(0, std::ios::end);
    const auto payload_end = f.tellg();
    const std::streamsize want = (std::streamsize) tok_count * sizeof(int32_t);
    if (!f || tok_count > max_tokens || tok_count > fp.fp_n_ctx || tok_count > INT32_MAX || payload_end - payload_start != want) {
        return false;
    }
    f.seekg(payload_start);
    toks_out.resize(tok_count);
    f.read((char *) toks_out.data(), want);
    if (f.gcount() != want) {
        toks_out.clear();
        return false;
    }
    fp_out = fp;
    return true;
}

struct server_slot {
    int id;

    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;

    common_memory mem;

    // multimodal
    mtmd_context * mctx = nullptr;
    mtmd::batch_ptr mbatch = nullptr;

    // speculative decoding
    common_speculative * spec;

    llama_tokens spec_draft;
    std::vector<common_speculative_token_dist> spec_dists;
    llama_tokens spec_prompt;
    std::vector<int32_t> spec_i_batch;
    common_prompt_checkpoint spec_ckpt;
    bool spec_is_replay = false;
    std::mt19937 spec_synth_rng;

    // TODO: move members that belong to the task (such as `generated_text`, `has_new_line`) to task_results_state
    //       see https://github.com/ggml-org/llama.cpp/pull/18283#issuecomment-3710175837
    std::unique_ptr<const server_task> task;
    std::unique_ptr<const server_task> task_prev; // used for debugging

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx   = 0;  // context size per slot
    int32_t n_keep  = 0;
    int32_t i_batch = -1;

int32_t n_prompt_tokens_cache     = 0;
    int32_t n_prompt_tokens_processed = 0;
    int32_t n_prompt_tokens_lcp       = 0;
    int32_t n_prompt_tokens_planned   = 0;
    std::string prompt_cache_source   = "none";
    std::string prompt_cache_reason   = "none";

    // effective generation limit for the current task, -1 means unlimited
    int32_t n_predict_max = -1;
    int32_t n_ctx_reservation = 0;
    int32_t n_kv_reservation = 0;
    bool clear_context_on_release = false;
    // Target-only adaptive snapshots need one real target suffix decode before MTP can resume.
    bool bootstrap_pending = false;

    size_t last_nl_pos = 0;

    std::string  generated_text;
    std::string  debug_generated_text;
    llama_tokens generated_tokens;
    size_t n_sent_text = 0; // number of sent text character (i.e. handle partial UTF-8 on streaming)

    std::vector<completion_token_output> generated_token_probs;

    bool has_next_token = true;
    bool has_new_line   = false;
    bool truncated      = false;
    bool just_restored  = false; // set on disk slot-restore; one-shot, gates restored-slot KV reuse

    // --- KV restore-reuse (logits sidecar) ---
    // Full-vocab logits of this slot's most recently sampled token, captured at sample time
    // (only populated for FULL/recurrent models when --slot-save-path is set). Serialized to the
    // <state>.logits sidecar on SLOT_SAVE so a later exact-prompt "regenerate" can emit the first
    // token WITHOUT re-decoding into the (un-rewindable) restored recurrent state.
    std::vector<float> logits_last;     // size n_vocab when valid, else empty
    // Token count of the prompt-state that `logits_last` corresponds to (i.e. the slot's KV/token
    // length at the moment of capture). Used to BIND the captured distribution to a specific state:
    // a sidecar is only written when this equals the saved snapshot's token_count, so a stale
    // distribution (e.g. left over from a prior task, or skipped on a spec-decode step) can never
    // be serialized against a mismatched state. -1 = no valid capture.
    int32_t logits_last_n_tokens = -1;
    // Logits loaded from a sidecar at SLOT_RESTORE, consumed once by the restore-continue path.
    std::vector<float> restored_logits; // size n_vocab when a valid sidecar was loaded, else empty

    stop_type stop;
    std::string stop_detail;

    server_loop_guard loop_guard;
    int32_t loop_guard_interventions = 0;
    bool loop_guard_triggered = false;
    std::string loop_guard_action;
    std::string loop_guard_reason;
    int32_t reasoning_output_tokens = 0;
    int32_t visible_output_tokens = 0;

    std::string stopping_word;

    // state
    slot_state state = SLOT_STATE_IDLE;

    server_prompt prompt;

    void prompt_reset_after_memory_clear() {
        prompt.clear();
        n_prompt_tokens_cache = 0;
        n_prompt_tokens_processed = 0;
        n_prompt_tokens_lcp = 0;
        n_prompt_tokens_planned = 0;
        prompt_cache_source = "none";
        prompt_cache_reason = "memory_cleared";
        spec_ckpt.clear();
    }

    bool prompt_save(server_prompt_cache & prompt_cache) const {
        if ((task && !task->params.cache_prompt) || (!task && task_prev && !task_prev->params.cache_prompt)) {
            prompt_cache.last_reason = "outgoing task disabled prompt caching";
            return false;
        }
        const bool saved = prompt_cache.save(prompt, ctx_tgt, ctx_dft, spec, id);
        if (!saved) { SLT_TRC(*this, "prompt cache save miss: %s\n", prompt_cache.last_reason.c_str()); }
        return saved;
    }

server_prompt_cache_result prompt_load_result(server_prompt_cache & prompt_cache, const server_tokens & tokens) {
        const auto result = prompt_cache.load(prompt, tokens, ctx_tgt, ctx_dft, spec, id);
        SLT_TRC(*this, "prompt cache load: %s\n", prompt_cache.last_reason.c_str());
        return result;
    }

    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens) {
        const auto result = prompt_load_result(prompt_cache, tokens);
        return result == server_prompt_cache_result::hit || result == server_prompt_cache_result::unchanged;
    }

    void prompt_clear() {
        SLT_TRC(*this, "clearing prompt with %zu tokens\n", prompt.tokens.size());

        mem.seq_rm(id, -1, -1);
        common_speculative_set_state(spec, id, {});

prompt_reset_after_memory_clear();
        just_restored = false;
        restored_logits.clear();
        logits_last.clear();
        logits_last_n_tokens = -1;
        bootstrap_pending = false;
    }

    std::vector<common_adapter_lora_info> lora;
    int32_t alora_invocation_start = -1;

    // sampling
    json json_schema;

    common_sampler_ptr smpl;

    llama_token sampled; // in speculative mode, this is the last accepted token

    // for TTS models, this is the embd generated from prev step, decode this to generate next hidden state
    // corresponding to one token position (size = n_embd)
    std::vector<float> inp_embd;

    server_slot_stats stats;

    // accepted tokens per draft position
    // not in server_slot_stats to avoid copying to every task result
    std::vector<uint64_t> n_accepted_per_pos;

    std::function<void(int /* id_slot */)>   callback_on_release;
    std::function<void(const server_slot &)> callback_on_reset; // called before reset()

    // this is for printing timings with slot progress, not part of metrics
    int64_t t_print_last = 0;
    int32_t n_gen_last = 0;

    // Profit-controller state is retained per slot so measurements remain
    // attributable across compatible requests.
    server_adaptive_dm_state adaptive_dm;
    int64_t adaptive_cycle_start_us = 0;
    float adaptive_draft_ms = 0.0f;
    int32_t adaptive_requested_n_max = 0;

    void reset() {
        SLT_DBG(*this, "%s", "\n");

        spec_is_replay = false;

        n_prompt_tokens_cache = 0;
        n_prompt_tokens_lcp = 0;
        n_prompt_tokens_planned = 0;
        prompt_cache_source = "none";
        prompt_cache_reason = "none";
        bootstrap_pending = false;

        last_nl_pos    = 0;
        generated_text = "";
        has_new_line   = false;
        truncated      = false;
        stop           = STOP_TYPE_NONE;
        stop_detail    = "";
        loop_guard.reset();
        loop_guard_interventions = 0;
        loop_guard_triggered = false;
        loop_guard_action = "";
        loop_guard_reason = "";
        reasoning_output_tokens = 0;
        visible_output_tokens = 0;
        stopping_word  = "";
        n_sent_text    = 0;

        if (can_speculate()) {
            spec_draft.clear();
            spec_dists.clear();
            spec_i_batch.clear();
            spec_ckpt.clear();
        }
        generated_tokens.clear();
        generated_token_probs.clear();
        json_schema = json();

        task_prev = std::move(task);
        task.reset();

        // note: callback_on_reset() must have run before this, see release()
        stats = {};
        n_accepted_per_pos.clear();

        n_predict_max = -1;
        n_ctx_reservation = 0;
        n_kv_reservation = 0;

        llama_set_sampler(ctx_tgt, id, nullptr);

        // clear alora start
        alora_invocation_start = -1;

        // one-shot; never carry restored sidecar logits into a non-restore request.
        // NOTE: logits_last is deliberately NOT cleared here - it is the slot's running
        // "last sampled distribution" and must survive into the idle state so a subsequent
        // SLOT_SAVE can serialize it.
        restored_logits.clear();

        // clear multimodal state
        mbatch.reset();
    }

    void init_sampler() const {
        common_sampler_reset(smpl.get());

        if (!task->need_sampling()) {
            return;
        }

        const int64_t t_start = ggml_time_us();

        int n_text = 0;

        for (int i = 0; i < (int) prompt.tokens.size(); i++) {
            const llama_token id = prompt.tokens[i];

            if (id != LLAMA_TOKEN_NULL) {
                common_sampler_accept(smpl.get(), id, false);
                n_text++;
            }
        }

        SLT_TRC(*this, "init sampler, took %0.2f ms, tokens: text = %d, total = %d\n",
                (ggml_time_us() - t_start) / 1000.0, n_text, (int) prompt.tokens.size());
    }

    bool need_embd() const {
        GGML_ASSERT(task);
        return task->need_embd();
    }

    // if the context does not have a memory module then all embeddings have to be computed within a single ubatch
    // also we cannot split if the pooling would require any past tokens
    // (MTP supports splitting — uses task->need_embd() not need_embd())
    bool can_split() const {
        GGML_ASSERT(task);

        return
            !task->need_embd() ||
            (llama_get_memory(ctx_tgt) && llama_pooling_type(ctx_tgt) == LLAMA_POOLING_TYPE_LAST);
    }

    bool can_batch_with(server_slot & other_slot) const {
        GGML_ASSERT(task);

        return task->type == other_slot.task->type
            && inp_embd.size() == other_slot.inp_embd.size()
            && are_lora_equal(lora, other_slot.lora);
    }

    // returns -1 if the generation is limitless
    int32_t n_remaining() const {
        return n_predict_max == -1 ? -1 : n_predict_max - (int32_t) stats.n_gen;
    }

    bool has_budget() const {
        return n_predict_max == -1 || n_remaining() > 0;
    }

    bool is_processing() const {
        return state != SLOT_STATE_IDLE;
    }

    bool can_speculate() const {
        return !!spec;
    }

    bool uses_dflash() const {
        return task && std::find(
                task->params.speculative.types.begin(),
                task->params.speculative.types.end(),
                COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH) != task->params.speculative.types.end();
    }

    void add_token(const completion_token_output & token) {
        if (!is_processing()) {
            SLT_WRN(*this, "%s", "slot is not processing\n");
            return;
        }

        generated_token_probs.push_back(token);
    }

    int get_n_draft_max() const {
        GGML_ASSERT(task);

        if (!can_speculate()) {
            return 0;
        }

        // determine the max draft that fits the current slot state
        // note: slot.prompt is not yet expanded with the `id` token sampled above
        //       also, need to leave space for 1 extra token to allow context shifts
        int n_draft_max = n_ctx - prompt.n_tokens() - 2;

        if (n_remaining() > 0) {
            n_draft_max = std::min(n_draft_max, n_remaining() - 1);
        }

        if (uses_dflash() && adaptive_dm.dm_adaptive && adaptive_dm.adaptive_n_max >= 0) {
            n_draft_max = std::min(n_draft_max, adaptive_dm.adaptive_n_max);
        }

        SLT_DBG(*this, "max possible draft: %d\n", n_draft_max);

        return n_draft_max;
    }

    // add sampled token of this slot to the batch, optionally add the speculative draft tokens if any
    void handle_last_sampled_token(server_batch & batch) {
        bool add_ok = true;
        if (spec_draft.empty()) {
            // no speculative decoding
            i_batch = batch.size();

            if (!inp_embd.empty()) {
                add_ok &= batch.add(id, inp_embd, prompt.tokens.pos_next(), true, false);
            } else {
                add_ok &= batch.add(id, sampled, prompt.tokens.pos_next(), true, false);
            }

            SLT_DBG(*this, "slot decode token, id=%d, n_ctx = %d, n_tokens = %d, truncated = %d\n",
                    sampled, n_ctx, prompt.n_tokens(), truncated);
        } else {
            SLT_DBG(*this, "generate_draft: id=%d, #tokens=%zu, #draft=%zu, pos_next=%d\n",
                    sampled, prompt.tokens.size(), spec_draft.size(), prompt.tokens.pos_next());

            GGML_ASSERT(spec_i_batch.empty());

            spec_i_batch.push_back(batch.size());
            for (size_t i = 0; i < spec_draft.size(); i++) {
                spec_i_batch.push_back(batch.size() + i + 1);
            }

            auto pos0 = prompt.tokens.pos_next();

            add_ok &= batch.add(id, sampled, pos0++, true, false);
            for (auto token : spec_draft) {
                add_ok &= batch.add(this->id, token, pos0++, true, false);
            }
        }

        GGML_ASSERT(add_ok && "batch must be large enough to hold the sampled and draft tokens");

        prompt.tokens.push_back(sampled);
        prompt.tokens.insert(spec_draft);
    }

    void release() {
        if (is_processing()) {
            GGML_ASSERT(task);

            SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\n", prompt.n_tokens(), truncated);

            t_last_used = ggml_time_us();

            state = SLOT_STATE_IDLE;

            // do not keep context of the child slots - the parent's context is enough
            if (clear_context_on_release || task->is_child()) {
                prompt_clear();
            }

            callback_on_reset(*this);

            reset();

            callback_on_release(id);
        }
    }

size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) {
        GGML_ASSERT(task);

        size_t stop_pos = std::string::npos;

        for (const std::string & word : task->params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                // otherwise, partial stop
                pos = string_find_partial_stop(text, word);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stop           = STOP_TYPE_WORD;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    void print_timings_tg() {
        if (stats.n_gen < 100) {
            return;
        }

        const int64_t t_now = ggml_time_us();

        if (t_now - t_print_last < 3*1000*1000) {
            return;
        }

        const double n_gen_second     = stats.n_gen_tps();
        const double n_gen_second_win = 1e6 / (t_now - t_print_last) * (stats.n_gen - n_gen_last);

        t_print_last = t_now;
        n_gen_last = stats.n_gen;

        SLT_INF(*this, "n_gen = %6d, tg = %6.2f t/s, tg_3s = %6.2f t/s\n", (int) stats.n_gen, n_gen_second, n_gen_second_win);
    }

    void print_timings_pp() const {
        const double t_prompt_total = stats.t_prompt_ms();

        if (t_prompt_total < 3000.0) {
            return;
        }

        const double n_prompt_second = stats.n_prompt_tps();
        const double f_progress = task->n_tokens() > 0 ? (double) prompt.n_tokens() / task->n_tokens() : 0.0;

        SLT_INF(*this, "prompt processing, n_tokens = %6d, progress = %.2f, t = %6.2f s / %.2f tokens per second\n",
                (int) stats.n_prompt_processed, f_progress, t_prompt_total / 1e3, n_prompt_second);
    }

    void print_timings() const {
        const double t_prompt_total = stats.t_prompt_ms();
        const double t_gen_total    = stats.t_gen_ms();

        const double t_prompt        = stats.t_prompt_per_token_ms();
        const double n_prompt_second = stats.n_prompt_tps();

        const double t_gen        = stats.t_gen_per_token_ms();
        const double n_gen_second = stats.n_gen_tps();

        SLT_INF(*this,
                "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_prompt_total, (int) stats.n_prompt_processed, t_prompt, n_prompt_second);

        SLT_INF(*this,
                "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_gen_total, (int) stats.n_gen, t_gen, n_gen_second);

        SLT_INF(*this,
                "      total time = %10.2f ms / %5d tokens\n",
                t_prompt_total + t_gen_total, (int) (stats.n_prompt_processed + stats.n_gen));

        SLT_INF(*this,
                "   graphs reused = %10d\n",
                llama_perf_context(ctx_tgt).n_reused);

        const int32_t n_draft_total       = stats.n_draft_tokens;
        const int32_t n_draft_accepted    = stats.n_draft_accepted;
        const int32_t n_draft_verif_steps = stats.n_draft_verif_steps;

        if (n_draft_total > 0) {
            const float  draft_ratio  = (float) n_draft_accepted / n_draft_total;
            const double mean_acc_len = n_draft_verif_steps > 0 ? 1.0 + (double) n_draft_accepted / (double) n_draft_verif_steps : 1.0;

            std::string acceptance_rates_per_pos;
            if (n_draft_verif_steps > 0) {
                for (size_t i = 0; i < n_accepted_per_pos.size(); ++i) {
                    if (i > 0) {
                        acceptance_rates_per_pos += ", ";
                    }
                    acceptance_rates_per_pos += string_format("%.3f", (double) n_accepted_per_pos[i] / (double) n_draft_verif_steps);
                }
            }

            SLT_INF(*this,
                    "draft acceptance = %0.5f (%5d accepted / %5d generated), mean len = %5.2f\n",
                    draft_ratio, n_draft_accepted, n_draft_total, mean_acc_len);
            SLT_TRC(*this,
                    "     acc per pos = (%s)\n", acceptance_rates_per_pos.c_str());
        }

        common_speculative_print_stats(spec);
    }

    json to_json(bool only_metrics = false) const {
        json res;

        res = {
            {"id",            id},
            {"n_ctx",         n_ctx},
            {"speculative",   can_speculate()},
            {"is_processing", is_processing()},
        };

        const auto & ptask = task ? task : task_prev;

        if (ptask) {
            res["id_task"] = ptask->id;
            res["n_prompt_tokens"]           = (int32_t) prompt.tokens.size();
            res["n_prompt_tokens_processed"] = stats.n_prompt_processed;
            res["n_prompt_tokens_cache"]     = stats.n_prompt_cached;
            res["params"] = ptask->params.to_json(only_metrics);
            res["next_token"] = json::array({
                {
                    {"has_next_token", has_next_token},
                    {"has_new_line",   has_new_line},
                    {"n_remain",       n_remaining()},
                    {"n_decoded",      stats.n_gen},
                }
            });

            if (!only_metrics) {
                res["prompt"] = ctx_tgt ? ptask->tokens.detokenize(ctx_tgt, true) : "";
                res["generated"] = generated_text.empty() ? debug_generated_text : generated_text;
            }
        }

        return res;
    }

    void copy_state_to(server_slot & other) const {
        GGML_ASSERT(state == SLOT_STATE_DONE_PROMPT);

        mem.seq_rm(other.id,     -1, -1);
        mem.seq_cp(id, other.id, -1, -1);

        other.i_batch = i_batch;

other.n_prompt_tokens_cache     = n_prompt_tokens_cache;
        other.n_prompt_tokens_processed = n_prompt_tokens_processed;
        other.n_prompt_tokens_lcp       = n_prompt_tokens_lcp;
        other.n_prompt_tokens_planned   = n_prompt_tokens_planned;
        other.prompt_cache_source       = prompt_cache_source;
        other.prompt_cache_reason       = prompt_cache_reason;
        other.stats = stats;

        other.prompt = prompt.clone();
        other.init_sampler();
    }
};

namespace {

// One explicit snapshot can coexist with the file bytes, decoded vectors,
// rollback copy, checkpoint copies and canonical comparison scratch. Reserve
// the complete peak against the global prompt-cache budget before reading or
// serializing so eviction and failure remain explicit.
static constexpr uint64_t ADAPTIVE_SLOT_WORKING_COPIES = 5;

struct adaptive_slot_cache_reservation {
    server_prompt_cache * cache = nullptr;
    size_t bytes = 0;

    bool acquire(server_prompt_cache * value, size_t amount) {
        if (!value || !value->reserve_transient(amount)) {
            return !value;
        }
        cache = value;
        bytes = amount;
        return true;
    }

    ~adaptive_slot_cache_reservation() {
        if (cache) {
            cache->release_transient(bytes);
        }
    }
};

// Explicit adaptive slot files are deliberately separate from the upstream
// sequence-state format: the latter has no model/configuration identity and
// cannot carry a draft or MTP implementation state.
static constexpr std::array<uint8_t, 8> ADAPTIVE_SLOT_MAGIC = {{
    'L', 'L', 'A', 'M', 'A', 'S', 'L', 'T'
}};
static constexpr uint32_t ADAPTIVE_SLOT_VERSION = 1;

class adaptive_slot_writer {
public:
    void reserve(size_t size) {
        data.reserve(size);
    }

    void raw(const void * ptr, size_t size) {
        if (size == 0) {
            return;
        }
        if (ptr == nullptr) {
            throw std::invalid_argument("adaptive slot writer received a null payload");
        }
        const auto * begin = static_cast<const uint8_t *>(ptr);
        data.insert(data.end(), begin, begin + size);
    }

    void u8(uint8_t value) {
        data.push_back(value);
    }

    void u32(uint32_t value) {
        for (int i = 0; i < 4; ++i) {
            data.push_back((uint8_t) (value >> (8*i)));
        }
    }

    void u64(uint64_t value) {
        for (int i = 0; i < 8; ++i) {
            data.push_back((uint8_t) (value >> (8*i)));
        }
    }

    void i32(int32_t value) { u32((uint32_t) value); }
    void i64(int64_t value) { u64((uint64_t) value); }

    void string(const std::string & value) {
        u64(value.size());
        raw(value.data(), value.size());
    }

    void bytes(const std::vector<uint8_t> & value) {
        u64(value.size());
        raw(value.data(), value.size());
    }

    std::vector<uint8_t> finish() && {
        u64(XXH64(data.data(), data.size(), 0x534c4f5453544154ULL));
        return std::move(data);
    }

private:
    std::vector<uint8_t> data;
};

class adaptive_slot_reader {
public:
    adaptive_slot_reader(const uint8_t * data, size_t size) : data(data), size(size) {}

    void raw(void * dst, size_t count) {
        if (count > remaining()) {
            throw std::runtime_error("truncated adaptive slot snapshot");
        }
        if (count) {
            std::memcpy(dst, data + pos, count);
            pos += count;
        }
    }

    uint8_t u8() {
        uint8_t value = 0;
        raw(&value, sizeof(value));
        return value;
    }

    uint32_t u32() {
        uint32_t value = 0;
        for (int i = 0; i < 4; ++i) {
            value |= (uint32_t) u8() << (8*i);
        }
        return value;
    }

    uint64_t u64() {
        uint64_t value = 0;
        for (int i = 0; i < 8; ++i) {
            value |= (uint64_t) u8() << (8*i);
        }
        return value;
    }

    int32_t i32() { return (int32_t) u32(); }
    int64_t i64() { return (int64_t) u64(); }

    std::string string(const char * label) {
        const uint64_t count = u64();
        if (count > remaining() || count > std::numeric_limits<size_t>::max()) {
            throw std::runtime_error(std::string("invalid adaptive slot ") + label + " length");
        }
        std::string value(reinterpret_cast<const char *>(data + pos), (size_t) count);
        pos += (size_t) count;
        return value;
    }

    std::vector<uint8_t> bytes(const char * label) {
        const uint64_t count = u64();
        if (count > remaining() || count > std::numeric_limits<size_t>::max()) {
            throw std::runtime_error(std::string("invalid adaptive slot ") + label + " length");
        }
        std::vector<uint8_t> value((size_t) count);
        raw(value.data(), value.size());
        return value;
    }

    size_t remaining() const { return size - pos; }
    bool done() const { return pos == size; }

private:
    const uint8_t * data;
    size_t size;
    size_t pos = 0;
};

struct adaptive_slot_checkpoint_blob {
    int64_t n_tokens = 0;
    int32_t id_task = -1;
    llama_pos pos_min = -1;
    llama_pos pos_max = -1;
    uint32_t flags_tgt = LLAMA_STATE_SEQ_FLAGS_NONE;
    uint32_t flags_dft = LLAMA_STATE_SEQ_FLAGS_NONE;
    uint64_t instance_tgt = 0;
    uint64_t instance_dft = 0;
    std::string layout_tgt;
    std::string layout_dft;
    std::array<llama_pos, 4> attention_tgt = {{-1, -1, -1, -1}};
    std::array<llama_pos, 4> attention_dft = {{-1, -1, -1, -1}};
    std::array<llama_pos, 8> retained_tgt = {{-1, -1, -1, -1, -1, -1, -1, -1}};
    std::array<llama_pos, 8> retained_dft = {{-1, -1, -1, -1, -1, -1, -1, -1}};
    uint32_t retained_count_tgt = UINT32_MAX;
    uint32_t retained_count_dft = UINT32_MAX;
    bool draft_base_valid = false;
    std::vector<uint8_t> data_tgt;
    std::vector<uint8_t> data_dft;
    std::vector<uint8_t> data_spec;
};

struct adaptive_slot_snapshot_blob {
    uint32_t profile = COMMON_CONTEXT_PROFILE_LONG;
    int32_t active_ctx = 0;
    int32_t ctx_size_mtp = 0;
    int32_t mtp_max_tokens = 0;
    int32_t long_ctx = 0;
    uint64_t model_instance = 0;
    std::string model_fingerprint;
    std::string layout_tgt;
    std::string layout_dft;
    llama_pos pos_tgt = -1;
    llama_pos pos_dft = -1;
    uint64_t n_tokens = 0;
    std::vector<uint8_t> tokens;
    std::vector<uint8_t> data_tgt;
    std::vector<uint8_t> data_dft;
    std::vector<uint8_t> data_spec;
    std::vector<adaptive_slot_checkpoint_blob> checkpoints;
};

static uint64_t adaptive_slot_serialized_size(const adaptive_slot_snapshot_blob & snapshot);
static uint64_t adaptive_slot_snapshot_memory_bytes(const adaptive_slot_snapshot_blob & snapshot);

static void adaptive_slot_write_checkpoint(adaptive_slot_writer & writer,
        const adaptive_slot_checkpoint_blob & checkpoint) {
    writer.i64(checkpoint.n_tokens);
    writer.i32(checkpoint.id_task);
    writer.i32(checkpoint.pos_min);
    writer.i32(checkpoint.pos_max);
    writer.u32(checkpoint.flags_tgt);
    writer.u32(checkpoint.flags_dft);
    writer.u64(checkpoint.instance_tgt);
    writer.u64(checkpoint.instance_dft);
    writer.string(checkpoint.layout_tgt);
    writer.string(checkpoint.layout_dft);
    for (const llama_pos value : checkpoint.attention_tgt) { writer.i32(value); }
    for (const llama_pos value : checkpoint.attention_dft) { writer.i32(value); }
    for (const llama_pos value : checkpoint.retained_tgt) { writer.i32(value); }
    for (const llama_pos value : checkpoint.retained_dft) { writer.i32(value); }
    writer.u32(checkpoint.retained_count_tgt);
    writer.u32(checkpoint.retained_count_dft);
    writer.u8(checkpoint.draft_base_valid ? 1 : 0);
    writer.bytes(checkpoint.data_tgt);
    writer.bytes(checkpoint.data_dft);
    writer.bytes(checkpoint.data_spec);
}

static adaptive_slot_checkpoint_blob adaptive_slot_read_checkpoint(adaptive_slot_reader & reader) {
    adaptive_slot_checkpoint_blob checkpoint;
    checkpoint.n_tokens = reader.i64();
    checkpoint.id_task = reader.i32();
    checkpoint.pos_min = reader.i32();
    checkpoint.pos_max = reader.i32();
    checkpoint.flags_tgt = reader.u32();
    checkpoint.flags_dft = reader.u32();
    checkpoint.instance_tgt = reader.u64();
    checkpoint.instance_dft = reader.u64();
    checkpoint.layout_tgt = reader.string("checkpoint target layout");
    checkpoint.layout_dft = reader.string("checkpoint draft layout");
    for (auto & value : checkpoint.attention_tgt) { value = reader.i32(); }
    for (auto & value : checkpoint.attention_dft) { value = reader.i32(); }
    for (auto & value : checkpoint.retained_tgt) { value = reader.i32(); }
    for (auto & value : checkpoint.retained_dft) { value = reader.i32(); }
    checkpoint.retained_count_tgt = reader.u32();
    checkpoint.retained_count_dft = reader.u32();
    checkpoint.draft_base_valid = reader.u8() != 0;
    checkpoint.data_tgt = reader.bytes("checkpoint target state");
    checkpoint.data_dft = reader.bytes("checkpoint draft state");
    checkpoint.data_spec = reader.bytes("checkpoint MTP state");
    return checkpoint;
}

static std::vector<uint8_t> adaptive_slot_encode(const adaptive_slot_snapshot_blob & snapshot, size_t max_bytes) {
    const uint64_t wire_size = adaptive_slot_serialized_size(snapshot);
    if (wire_size > max_bytes || wire_size > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("adaptive slot snapshot exceeds the configured state budget (wire=" +
                std::to_string(wire_size) + ", max=" + std::to_string(max_bytes) + ")");
    }
    adaptive_slot_writer writer;
    writer.reserve((size_t) wire_size);
    writer.raw(ADAPTIVE_SLOT_MAGIC.data(), ADAPTIVE_SLOT_MAGIC.size());
    writer.u32(ADAPTIVE_SLOT_VERSION);
    writer.u32(snapshot.profile);
    writer.i32(snapshot.active_ctx);
    writer.i32(snapshot.ctx_size_mtp);
    writer.i32(snapshot.mtp_max_tokens);
    writer.i32(snapshot.long_ctx);
    writer.u64(snapshot.model_instance);
    writer.string(snapshot.model_fingerprint);
    writer.string(snapshot.layout_tgt);
    writer.string(snapshot.layout_dft);
    writer.i32(snapshot.pos_tgt);
    writer.i32(snapshot.pos_dft);
    writer.u64(snapshot.n_tokens);
    writer.bytes(snapshot.tokens);
    writer.bytes(snapshot.data_tgt);
    writer.bytes(snapshot.data_dft);
    writer.bytes(snapshot.data_spec);
    writer.u64(snapshot.checkpoints.size());
    for (const auto & checkpoint : snapshot.checkpoints) {
        adaptive_slot_write_checkpoint(writer, checkpoint);
    }
    auto result = std::move(writer).finish();
    if (result.size() != wire_size) {
        throw std::runtime_error("adaptive slot snapshot size calculation failed");
    }
    return result;
}

static uint64_t adaptive_slot_read_u64_tail(const uint8_t * data) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= (uint64_t) data[i] << (8*i);
    }
    return value;
}

static adaptive_slot_snapshot_blob adaptive_slot_decode(const std::vector<uint8_t> & file,
        size_t max_checkpoints, size_t max_decoded_bytes) {
    constexpr size_t ADAPTIVE_SLOT_MIN_SIZE = ADAPTIVE_SLOT_MAGIC.size() + sizeof(uint32_t) + sizeof(uint64_t);
    if (file.size() < ADAPTIVE_SLOT_MIN_SIZE) {
        throw std::runtime_error("legacy or truncated adaptive slot snapshot");
    }

    if (std::memcmp(file.data(), ADAPTIVE_SLOT_MAGIC.data(), ADAPTIVE_SLOT_MAGIC.size()) != 0) {
        throw std::runtime_error("legacy slot save file lacks adaptive snapshot identity");
    }

    const size_t payload_size = file.size() - sizeof(uint64_t);
    const uint64_t expected = adaptive_slot_read_u64_tail(file.data() + payload_size);
    const uint64_t actual = XXH64(file.data(), payload_size, 0x534c4f5453544154ULL);
    if (actual != expected) {
        throw std::runtime_error("adaptive slot snapshot checksum mismatch");
    }

    adaptive_slot_reader reader(file.data(), payload_size);
    std::array<uint8_t, ADAPTIVE_SLOT_MAGIC.size()> magic = {};
    reader.raw(magic.data(), magic.size());
    if (magic != ADAPTIVE_SLOT_MAGIC) {
        throw std::runtime_error("legacy slot save file lacks adaptive snapshot identity");
    }

    if (reader.u32() != ADAPTIVE_SLOT_VERSION) {
        throw std::runtime_error("unsupported adaptive slot snapshot version");
    }

    adaptive_slot_snapshot_blob snapshot;
    snapshot.profile = reader.u32();
    if (snapshot.profile != COMMON_CONTEXT_PROFILE_MTP &&
            snapshot.profile != COMMON_CONTEXT_PROFILE_LONG &&
            snapshot.profile != COMMON_CONTEXT_PROFILE_MTP_SHORT &&
            snapshot.profile != COMMON_CONTEXT_PROFILE_XLONG &&
            snapshot.profile != COMMON_CONTEXT_PROFILE_XXLONG) {
        throw std::runtime_error("invalid adaptive slot snapshot profile");
    }
    snapshot.active_ctx = reader.i32();
    snapshot.ctx_size_mtp = reader.i32();
    snapshot.mtp_max_tokens = reader.i32();
    snapshot.long_ctx = reader.i32();
    snapshot.model_instance = reader.u64();
    snapshot.model_fingerprint = reader.string("model identity");
    snapshot.layout_tgt = reader.string("target layout");
    snapshot.layout_dft = reader.string("draft layout");
    snapshot.pos_tgt = reader.i32();
    snapshot.pos_dft = reader.i32();
    snapshot.n_tokens = reader.u64();
    if (snapshot.n_tokens > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error("adaptive slot snapshot token count is too large");
    }
    snapshot.tokens = reader.bytes("tokens");
    snapshot.data_tgt = reader.bytes("target state");
    snapshot.data_dft = reader.bytes("draft state");
    snapshot.data_spec = reader.bytes("MTP state");

    const uint64_t n_checkpoints = reader.u64();
    if (n_checkpoints > max_checkpoints || n_checkpoints > reader.remaining() || n_checkpoints > 1000000 ||
            n_checkpoints > max_decoded_bytes / sizeof(adaptive_slot_checkpoint_blob)) {
        throw std::runtime_error("invalid adaptive slot snapshot checkpoint count");
    }
    snapshot.checkpoints.reserve((size_t) n_checkpoints);
    for (uint64_t i = 0; i < n_checkpoints; ++i) {
        snapshot.checkpoints.push_back(adaptive_slot_read_checkpoint(reader));
    }
    if (!reader.done()) {
        throw std::runtime_error("trailing data in adaptive slot snapshot");
    }
    if (adaptive_slot_snapshot_memory_bytes(snapshot) > max_decoded_bytes) {
        throw std::runtime_error("adaptive slot snapshot exceeds the decoded state budget");
    }
    if (snapshot.tokens.size() % sizeof(llama_token) != 0 || snapshot.data_tgt.empty()) {
        throw std::runtime_error("adaptive slot snapshot has invalid target payload");
    }
    if (!snapshot.model_fingerprint.empty() && snapshot.layout_tgt.empty()) {
        throw std::runtime_error("adaptive slot snapshot has incomplete identity");
    }
    return snapshot;
}

static uint64_t adaptive_slot_saturating_add(uint64_t left, uint64_t right) {
    return left > std::numeric_limits<uint64_t>::max() - right
        ? std::numeric_limits<uint64_t>::max() : left + right;
}

static uint64_t adaptive_slot_saturating_mul(uint64_t left, uint64_t right) {
    return left != 0 && right > std::numeric_limits<uint64_t>::max() / left
        ? std::numeric_limits<uint64_t>::max() : left * right;
}

static size_t adaptive_slot_working_bytes(size_t max_file_bytes) {
    const uint64_t total = adaptive_slot_saturating_mul(
        (uint64_t) max_file_bytes, ADAPTIVE_SLOT_WORKING_COPIES);
    return (size_t) std::min<uint64_t>(total, std::numeric_limits<size_t>::max());
}

static uint64_t adaptive_slot_ceil_div(uint64_t value, uint64_t divisor) {
    return value == 0 ? 0 : (value - 1) / divisor + 1;
}

static uint64_t adaptive_slot_ram_budget(int32_t cache_ram_mib) {
    constexpr uint64_t MIB = 1024ULL * 1024ULL;
    constexpr uint64_t FALLBACK = 8ULL * 1024ULL * MIB;

    uint64_t configured = 0;
    if (cache_ram_mib > 0) {
        configured = adaptive_slot_saturating_mul((uint64_t) cache_ram_mib, MIB);
    }

    uint64_t available = FALLBACK;
#ifndef _WIN32
    const long pages = sysconf(_SC_AVPHYS_PAGES);
    const long page_size = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page_size > 0) {
        available = adaptive_slot_saturating_mul((uint64_t) pages, (uint64_t) page_size);
        // Leave one quarter of currently available RAM for the server and allocator.
        available -= available / 4;
    }
#endif
    return configured ? std::min(configured, available) : available;
}

static uint64_t adaptive_slot_vector_storage(const std::vector<uint8_t> & value) {
    return value.capacity();
}

static uint64_t adaptive_slot_string_storage(const std::string & value) {
    return value.capacity();
}

static uint64_t adaptive_slot_snapshot_memory_bytes(const adaptive_slot_snapshot_blob & snapshot) {
    uint64_t total = sizeof(snapshot);
    const auto add = [&total](uint64_t value) {
        total = adaptive_slot_saturating_add(total, value);
    };
    add(adaptive_slot_string_storage(snapshot.model_fingerprint));
    add(adaptive_slot_string_storage(snapshot.layout_tgt));
    add(adaptive_slot_string_storage(snapshot.layout_dft));
    add(adaptive_slot_vector_storage(snapshot.tokens));
    add(adaptive_slot_vector_storage(snapshot.data_tgt));
    add(adaptive_slot_vector_storage(snapshot.data_dft));
    add(adaptive_slot_vector_storage(snapshot.data_spec));
    for (const auto & checkpoint : snapshot.checkpoints) {
        add(sizeof(checkpoint));
        add(adaptive_slot_string_storage(checkpoint.layout_tgt));
        add(adaptive_slot_string_storage(checkpoint.layout_dft));
        add(adaptive_slot_vector_storage(checkpoint.data_tgt));
        add(adaptive_slot_vector_storage(checkpoint.data_dft));
        add(adaptive_slot_vector_storage(checkpoint.data_spec));
    }
    return total;
}

static uint64_t adaptive_slot_wire_bytes(uint64_t size) {
    return adaptive_slot_saturating_add(8, size);
}

static uint64_t adaptive_slot_wire_string(uint64_t size) {
    return adaptive_slot_wire_bytes(size);
}

static uint64_t adaptive_slot_serialized_size(const adaptive_slot_snapshot_blob & snapshot) {
    uint64_t total = 8 + 4 + 4 + 4*4 + 8;
    const auto add = [&total](uint64_t value) {
        total = adaptive_slot_saturating_add(total, value);
    };
    add(adaptive_slot_wire_string(snapshot.model_fingerprint.size()));
    add(adaptive_slot_wire_string(snapshot.layout_tgt.size()));
    add(adaptive_slot_wire_string(snapshot.layout_dft.size()));
    add(4 + 4 + 8);
    add(adaptive_slot_wire_bytes(snapshot.tokens.size()));
    add(adaptive_slot_wire_bytes(snapshot.data_tgt.size()));
    add(adaptive_slot_wire_bytes(snapshot.data_dft.size()));
    add(adaptive_slot_wire_bytes(snapshot.data_spec.size()));
    add(8);
    for (const auto & checkpoint : snapshot.checkpoints) {
        // i64 token count, id/positions, two flag words and two instances.
        add(44);
        add(adaptive_slot_wire_string(checkpoint.layout_tgt.size()));
        add(adaptive_slot_wire_string(checkpoint.layout_dft.size()));
        add(96 + 4 + 4 + 1);
        add(adaptive_slot_wire_bytes(checkpoint.data_tgt.size()));
        add(adaptive_slot_wire_bytes(checkpoint.data_dft.size()));
        add(adaptive_slot_wire_bytes(checkpoint.data_spec.size()));
    }
    return adaptive_slot_saturating_add(total, 8);
}

// The file contains target/draft state plus up to n_ctx_checkpoints copies.  Use
// the live state/KV footprint to derive a conservative bound before allocating
// attacker-controlled bytes; the multiplier covers both profiles and metadata.
static size_t adaptive_slot_max_file_bytes(const server_slot & slot,
        llama_context * ctx_tgt, llama_context * ctx_dft,
        int32_t ctx_size_mtp_short, int32_t ctx_size_mtp, int32_t long_ctx,
        int32_t cache_ram_mib, int32_t n_ctx_checkpoints) {
    const uint64_t max_ctx = (uint64_t) std::max({ctx_size_mtp_short, ctx_size_mtp, long_ctx});
    const uint64_t active_ctx = (uint64_t) std::max(1u, ctx_tgt ? llama_n_ctx_seq(ctx_tgt) : 1u);
    uint64_t context_bytes = 0;
    uint64_t state_bytes = 0;
    for (llama_context * ctx : {ctx_tgt, ctx_dft}) {
        if (!ctx) {
            continue;
        }
        for (const auto & [unused, breakdown] : llama_get_memory_breakdown(ctx)) {
            context_bytes = adaptive_slot_saturating_add(context_bytes, breakdown.context);
        }
        state_bytes = adaptive_slot_saturating_add(
            state_bytes, llama_state_seq_get_size_ext(ctx, slot.id, LLAMA_STATE_SEQ_FLAGS_NONE));
    }

    const uint64_t n_tokens = std::max<uint64_t>(1, slot.prompt.tokens.size());
    uint64_t bytes_per_token = adaptive_slot_ceil_div(state_bytes, n_tokens);
    bytes_per_token = std::max<uint64_t>(bytes_per_token,
        adaptive_slot_ceil_div(context_bytes, active_ctx));
    bytes_per_token = std::max<uint64_t>(bytes_per_token, sizeof(llama_token));

    const uint64_t checkpoint_count = n_ctx_checkpoints > 0 ? (uint64_t) n_ctx_checkpoints : 0;
    const uint64_t state_budget = adaptive_slot_saturating_mul(
        adaptive_slot_saturating_mul(bytes_per_token, max_ctx),
        adaptive_slot_saturating_add(checkpoint_count, 2));
    const uint64_t token_budget = adaptive_slot_saturating_mul(
        max_ctx, sizeof(llama_token) * 16ULL);
    const uint64_t limit = adaptive_slot_saturating_add(
        adaptive_slot_saturating_add(state_budget, token_budget), 64ULL * 1024 * 1024);
    const uint64_t working_budget = adaptive_slot_ram_budget(cache_ram_mib);
    const uint64_t per_file = working_budget / ADAPTIVE_SLOT_WORKING_COPIES;
    return (size_t) std::min<uint64_t>(limit, per_file);
}

static std::vector<uint8_t> adaptive_slot_read_file(const std::string & filepath, size_t max_bytes) {
#ifndef _WIN32
    int flags = O_RDONLY;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
    const int fd = ::open(filepath.c_str(), flags);
    if (fd < 0) {
        throw std::runtime_error("cannot open adaptive slot snapshot");
    }
    struct fd_guard {
        int fd;
        ~fd_guard() { if (fd >= 0) { ::close(fd); } }
    } guard { fd };

    struct stat status = {};
    if (::fstat(fd, &status) != 0 || !S_ISREG(status.st_mode) || status.st_size < 0) {
        throw std::runtime_error("adaptive slot snapshot is not a regular file");
    }
    const uintmax_t file_size = (uintmax_t) status.st_size;
    if (file_size > max_bytes || file_size > (uintmax_t) std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("adaptive slot snapshot exceeds the configured state budget");
    }

    std::vector<uint8_t> file((size_t) file_size);
    size_t offset = 0;
    while (offset < file.size()) {
        const ssize_t count = ::read(fd, file.data() + offset, file.size() - offset);
        if (count > 0) {
            offset += (size_t) count;
        } else if (count < 0 && errno == EINTR) {
            continue;
        } else {
            throw std::runtime_error("cannot read adaptive slot snapshot");
        }
    }
    return file;
#else
    std::error_code status_error;
    const auto status = std::filesystem::status(filepath, status_error);
    if (status_error || !std::filesystem::is_regular_file(status)) {
        throw std::runtime_error("adaptive slot snapshot is not a regular file");
    }
    std::error_code size_error;
    const uintmax_t file_size = std::filesystem::file_size(filepath, size_error);
    if (size_error || file_size > max_bytes || file_size > (uintmax_t) std::numeric_limits<size_t>::max() ||
            file_size > (uintmax_t) std::numeric_limits<std::streamsize>::max()) {
        throw std::runtime_error("adaptive slot snapshot exceeds the configured state budget");
    }

    std::ifstream input(filepath, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open adaptive slot snapshot");
    }
    std::vector<uint8_t> file((size_t) file_size);
    if (!file.empty()) {
        input.read(reinterpret_cast<char *>(file.data()), (std::streamsize) file.size());
    }
    if (!input || input.gcount() != (std::streamsize) file.size()) {
        throw std::runtime_error("cannot read adaptive slot snapshot");
    }
    return file;
#endif
}

static size_t adaptive_slot_write_file(const std::string & filepath, const std::vector<uint8_t> & data) {
    const std::string temporary = filepath + ".tmp-" + std::to_string(ggml_time_us());
    try {
        {
            std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
            if (!output) {
                throw std::runtime_error("cannot create adaptive slot snapshot");
            }
            output.write(reinterpret_cast<const char *>(data.data()), (std::streamsize) data.size());
            output.flush();
            if (!output) {
                throw std::runtime_error("cannot write adaptive slot snapshot");
            }
        }

        std::error_code error;
        std::filesystem::rename(temporary, filepath, error);
        if (error) {
            throw std::runtime_error("cannot publish adaptive slot snapshot: " + error.message());
        }
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
    return data.size();
}

static adaptive_slot_checkpoint_blob adaptive_slot_capture_checkpoint(
        const common_prompt_checkpoint & source,
        llama_context * ctx_dft,
        int64_t evaluated,
        llama_pos pos_tgt) {
    adaptive_slot_checkpoint_blob checkpoint;
    checkpoint.n_tokens = source.n_tokens;
    checkpoint.id_task = source.id_task;
    checkpoint.pos_min = source.pos_min;
    checkpoint.pos_max = source.pos_max;
    checkpoint.flags_tgt = source.flags_tgt;
    checkpoint.flags_dft = source.flags_dft;
    checkpoint.instance_tgt = source.instance_tgt;
    checkpoint.instance_dft = source.instance_dft;
    checkpoint.layout_tgt = source.layout_tgt;
    checkpoint.layout_dft = source.layout_dft;
    checkpoint.attention_tgt = source.attention_tgt;
    checkpoint.attention_dft = source.attention_dft;
    checkpoint.retained_tgt = source.retained_tgt;
    checkpoint.retained_dft = source.retained_dft;
    checkpoint.retained_count_tgt = source.retained_count_tgt;
    checkpoint.retained_count_dft = source.retained_count_dft;
    checkpoint.draft_base_valid = source.draft_base_valid;
    checkpoint.data_tgt = source.data_tgt;

    if (ctx_dft && source.compatible_dft(ctx_dft) && source.n_tokens <= evaluated && source.pos_max <= pos_tgt) {
        checkpoint.data_dft = source.data_dft;
        checkpoint.data_spec = source.data_spec;
    } else {
        checkpoint.layout_dft.clear();
        checkpoint.instance_dft = 0;
        checkpoint.attention_dft = {{-1, -1, -1, -1}};
        checkpoint.retained_dft.fill(-1);
        checkpoint.retained_count_dft = UINT32_MAX;
        checkpoint.draft_base_valid = false;
    }
    return checkpoint;
}

static adaptive_slot_snapshot_blob adaptive_slot_capture(
        const server_slot & slot,
        llama_context * ctx_tgt,
        llama_context * ctx_dft,
        common_speculative * spec,
        const server_model_identity & identity,
        common_context_profile profile,
        int32_t ctx_size_mtp,
        int32_t mtp_max_tokens,
        int32_t long_ctx,
        size_t max_bytes,
        bool include_checkpoints = true) {
    if (!ctx_tgt || slot.prompt.tokens.empty()) {
        throw std::runtime_error("cannot save an empty adaptive slot");
    }

    llama_synchronize(ctx_tgt);
    if (ctx_dft) { llama_synchronize(ctx_dft); }

    adaptive_slot_snapshot_blob snapshot;
    const auto ensure_budget = [&](uint64_t extra) {
        const uint64_t used = adaptive_slot_snapshot_memory_bytes(snapshot);
        if (used > max_bytes || extra > max_bytes - used) {
            throw std::runtime_error("adaptive slot snapshot exceeds the configured state budget (used=" +
                    std::to_string(used) + ", extra=" + std::to_string(extra) +
                    ", max=" + std::to_string(max_bytes) + ")");
        }
    };
    snapshot.profile = (uint32_t) profile;
    snapshot.active_ctx = slot.n_ctx;
    snapshot.ctx_size_mtp = ctx_size_mtp;
    snapshot.mtp_max_tokens = mtp_max_tokens;
    snapshot.long_ctx = long_ctx;
    snapshot.model_fingerprint = identity.fingerprint();
    snapshot.layout_tgt = common_prompt_cache_layout(ctx_tgt);
    snapshot.layout_dft = ctx_dft ? common_prompt_cache_layout(ctx_dft) : "";
    snapshot.model_instance = llama_model_mtp_weights_get_info(llama_get_model(ctx_tgt)).model_instance;

    snapshot.pos_tgt = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);
    if (snapshot.pos_tgt < 0) {
        throw std::runtime_error("adaptive slot target state is empty");
    }
    const size_t evaluated = slot.prompt.tokens.size_up_to_pos(snapshot.pos_tgt + 1);
    if (!evaluated || slot.prompt.tokens.pos_next(evaluated) != snapshot.pos_tgt + 1) {
        throw std::runtime_error("adaptive slot tokens do not cover target state");
    }
    snapshot.n_tokens = evaluated;

    server_tokens tokens = slot.prompt.tokens.clone_for_cache();
    tokens.keep_first(evaluated);
    std::vector<char> packed = tokens.serialize();
    ensure_budget(adaptive_slot_saturating_mul(packed.size(), 2));
    snapshot.tokens.assign(reinterpret_cast<const uint8_t *>(packed.data()),
            reinterpret_cast<const uint8_t *>(packed.data()) + packed.size());
    std::vector<char>().swap(packed);

    const size_t n_tgt = llama_state_seq_get_size_ext(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_NONE);
    ensure_budget(n_tgt);
    snapshot.data_tgt.resize(n_tgt);
    if (!n_tgt || llama_state_seq_get_data_ext(ctx_tgt, snapshot.data_tgt.data(), n_tgt, slot.id,
            LLAMA_STATE_SEQ_FLAGS_NONE) != n_tgt) {
        throw std::runtime_error("adaptive slot target serialization failed");
    }

    if (ctx_dft) {
        snapshot.pos_dft = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), slot.id);
        const size_t n_dft = llama_state_seq_get_size_ext(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_NONE);
        ensure_budget(n_dft);
        snapshot.data_dft.resize(n_dft);
        if (n_dft && llama_state_seq_get_data_ext(ctx_dft, snapshot.data_dft.data(), n_dft, slot.id,
                LLAMA_STATE_SEQ_FLAGS_NONE) != n_dft) {
            throw std::runtime_error("adaptive slot draft serialization failed");
        }
        common_speculative_get_state(spec, slot.id, snapshot.data_spec);
        ensure_budget(0);
    }

    if (include_checkpoints) {
        for (const auto & checkpoint : slot.prompt.checkpoints) {
            if (!checkpoint || !checkpoint->host_only() || !checkpoint->compatible_tgt(ctx_tgt) ||
                    checkpoint->data_tgt.empty() || checkpoint->n_tokens > (int64_t) evaluated ||
                    checkpoint->pos_max > snapshot.pos_tgt) {
                continue;
            }
            const uint64_t checkpoint_bytes = adaptive_slot_saturating_add(
                sizeof(adaptive_slot_checkpoint_blob),
                adaptive_slot_saturating_add(checkpoint->layout_tgt.capacity(), checkpoint->layout_dft.capacity()));
            const uint64_t checkpoint_data = adaptive_slot_saturating_add(
                checkpoint->data_tgt.capacity(), adaptive_slot_saturating_add(
                    checkpoint->data_dft.capacity(), checkpoint->data_spec.capacity()));
            ensure_budget(adaptive_slot_saturating_add(checkpoint_bytes, checkpoint_data));
            snapshot.checkpoints.push_back(adaptive_slot_capture_checkpoint(
                *checkpoint, ctx_dft, evaluated, snapshot.pos_tgt));
            ensure_budget(0);
        }
    }
    return snapshot;
}

static std::shared_ptr<common_prompt_checkpoint> adaptive_slot_make_checkpoint(
        const adaptive_slot_checkpoint_blob & source,
        const adaptive_slot_snapshot_blob & snapshot,
        llama_context * ctx_tgt,
        llama_context * ctx_dft,
        const server_tokens & tokens,
        bool & dropped) {
    const auto valid_bounds = [](const auto & bounds) {
        for (size_t i = 0; i < bounds.size(); i += 2) {
            const auto begin = bounds[i];
            const auto end = bounds[i + 1];
            if ((begin < 0) != (end < 0) || (begin >= 0 && end < begin)) {
                return false;
            }
        }
        return true;
    };
    const bool valid_flags = (source.flags_tgt | source.flags_dft) & ~LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    const bool has_draft = !source.data_dft.empty();
    const bool has_carry = !source.data_spec.empty();
    const bool token_position_match = source.pos_max < std::numeric_limits<llama_pos>::max() &&
        (size_t) source.n_tokens == tokens.size_up_to_pos(source.pos_max + 1);
    if (source.n_tokens <= 0 || source.n_tokens > (int64_t) snapshot.n_tokens || source.pos_min < 0 ||
            source.pos_max < source.pos_min || source.pos_max >= snapshot.pos_tgt || source.data_tgt.empty() ||
            !token_position_match || source.flags_tgt != LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY ||
            valid_flags || source.flags_tgt & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE ||
            source.flags_dft & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE ||
            (has_draft != has_carry) || (!has_draft && (source.flags_dft != LLAMA_STATE_SEQ_FLAGS_NONE ||
                                                        !source.layout_dft.empty() || source.draft_base_valid)) ||
            (has_draft && source.flags_dft != LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) ||
            (source.retained_count_tgt != UINT32_MAX && source.retained_count_tgt > 4) ||
            (source.retained_count_dft != UINT32_MAX && source.retained_count_dft > 4) ||
            !valid_bounds(source.attention_tgt) || !valid_bounds(source.attention_dft) ||
            !valid_bounds(source.retained_tgt) || !valid_bounds(source.retained_dft) ||
            (source.draft_base_valid && (source.data_dft.empty() || source.data_spec.empty())) ||
            !common_prompt_cache_layout_reusable(source.layout_tgt, common_prompt_cache_layout(ctx_tgt))) {
        dropped = true;
        return nullptr;
    }

    auto checkpoint = std::make_shared<common_prompt_checkpoint>();
    checkpoint->n_tokens = source.n_tokens;
    checkpoint->id_task = source.id_task;
    checkpoint->pos_min = source.pos_min;
    checkpoint->pos_max = source.pos_max;
    checkpoint->flags_tgt = source.flags_tgt;
    checkpoint->model_tgt = llama_get_model(ctx_tgt);
    checkpoint->instance_tgt = llama_model_mtp_weights_get_info(checkpoint->model_tgt).model_instance;
    checkpoint->layout_tgt = common_prompt_cache_layout(ctx_tgt);
    checkpoint->attention_tgt = source.attention_tgt;
    checkpoint->retained_tgt = source.retained_tgt;
    checkpoint->retained_count_tgt = source.retained_count_tgt;
    checkpoint->data_tgt = source.data_tgt;

    const bool is_mtp_snapshot = snapshot.profile == COMMON_CONTEXT_PROFILE_MTP ||
                                 snapshot.profile == COMMON_CONTEXT_PROFILE_MTP_SHORT;
    if (ctx_dft && is_mtp_snapshot && has_draft &&
            common_prompt_cache_layout_reusable(source.layout_dft, common_prompt_cache_layout(ctx_dft))) {
        checkpoint->flags_dft = source.flags_dft;
        checkpoint->model_dft = llama_get_model(ctx_dft);
        checkpoint->instance_dft = llama_model_mtp_weights_get_info(checkpoint->model_dft).model_instance;
        checkpoint->layout_dft = common_prompt_cache_layout(ctx_dft);
        checkpoint->attention_dft = source.attention_dft;
        checkpoint->retained_dft = source.retained_dft;
        checkpoint->retained_count_dft = source.retained_count_dft;
        checkpoint->draft_base_valid = source.draft_base_valid;
        checkpoint->data_dft = source.data_dft;
        checkpoint->data_spec = source.data_spec;
    } else if (has_draft) {
        dropped = true;
        return nullptr;
    }
    return checkpoint;
}

static bool adaptive_slot_restore(
        const adaptive_slot_snapshot_blob & snapshot,
        server_slot & slot,
        llama_context * ctx_tgt,
        llama_context * ctx_dft,
        common_speculative * spec,
        std::string & error) {
    if (!ctx_tgt || !common_prompt_cache_layout_reusable(snapshot.layout_tgt, common_prompt_cache_layout(ctx_tgt))) {
        error = "adaptive slot snapshot target layout differs";
        return false;
    }
    if (snapshot.pos_tgt < 0 || snapshot.n_tokens == 0 ||
            snapshot.n_tokens > (uint64_t) llama_n_ctx_seq(ctx_tgt) ||
            snapshot.pos_tgt >= (llama_pos) llama_n_ctx_seq(ctx_tgt)) {
        error = "adaptive slot snapshot does not fit destination context";
        return false;
    }
    if (snapshot.pos_dft < -1 || (ctx_dft && snapshot.pos_dft >= (llama_pos) llama_n_ctx_seq(ctx_dft))) {
        error = "adaptive slot snapshot draft position does not fit destination context";
        return false;
    }
    if (snapshot.tokens.size() % sizeof(llama_token) != 0) {
        error = "adaptive slot snapshot token payload is malformed";
        return false;
    }

    server_prompt candidate;
    try {
        llama_tokens packed(snapshot.tokens.size() / sizeof(llama_token));
        if (!snapshot.tokens.empty()) {
            std::memcpy(packed.data(), snapshot.tokens.data(), snapshot.tokens.size());
        }
        candidate.tokens = server_tokens::deserialize(packed, slot.mctx != nullptr);
        if (candidate.tokens.size() != snapshot.n_tokens || !candidate.tokens.validate(ctx_tgt) ||
                snapshot.pos_tgt == std::numeric_limits<llama_pos>::max() ||
                candidate.tokens.pos_next() != snapshot.pos_tgt + 1) {
            error = "adaptive slot snapshot tokens are invalid";
            return false;
        }

    } catch (const std::bad_alloc &) {
        error = "adaptive slot snapshot allocation failed";
        return false;
    } catch (const std::exception & exception) {
        error = exception.what();
        return false;
    }

    const bool destination_mtp = ctx_dft != nullptr;
    const bool snapshot_mtp = snapshot.profile == COMMON_CONTEXT_PROFILE_MTP ||
                              snapshot.profile == COMMON_CONTEXT_PROFILE_MTP_SHORT;
    if (snapshot_mtp != destination_mtp) {
        error = "adaptive slot snapshot profile does not match destination context";
        return false;
    }
    const bool complete_mtp = destination_mtp && snapshot_mtp &&
        !snapshot.data_dft.empty() && !snapshot.data_spec.empty();
    if (snapshot_mtp) {
        if (snapshot.data_dft.empty() != snapshot.data_spec.empty() ||
                (snapshot.data_dft.empty() && snapshot.pos_dft >= 0) ||
                (complete_mtp && snapshot.pos_dft != snapshot.pos_tgt)) {
            error = "adaptive slot snapshot MTP state is incoherent";
            return false;
        }
    } else if (!snapshot.data_dft.empty() || !snapshot.data_spec.empty() || snapshot.pos_dft >= 0) {
        error = "adaptive slot snapshot long profile carries draft state";
        return false;
    }
    if (destination_mtp && snapshot_mtp && !snapshot.data_dft.empty() &&
            !common_prompt_cache_layout_reusable(snapshot.layout_dft, common_prompt_cache_layout(ctx_dft))) {
        error = "adaptive slot snapshot draft layout differs";
        return false;
    }

    // Only now clear the live sequence. Any malformed file or incompatible
    // metadata has already returned, so a failed set operation leaves no
    // published prompt and no partial cache hit.
    slot.prompt_clear();
    const bool target_ok = llama_state_seq_set_data_ext(ctx_tgt, snapshot.data_tgt.data(), snapshot.data_tgt.size(),
            slot.id, LLAMA_STATE_SEQ_FLAGS_NONE) == snapshot.data_tgt.size() &&
        llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id) == snapshot.pos_tgt;
    bool draft_ok = true;
    if (complete_mtp) {
        draft_ok = llama_state_seq_set_data_ext(ctx_dft, snapshot.data_dft.data(), snapshot.data_dft.size(),
                slot.id, LLAMA_STATE_SEQ_FLAGS_NONE) == snapshot.data_dft.size() &&
            llama_memory_seq_pos_max(llama_get_memory(ctx_dft), slot.id) == snapshot.pos_dft;
    }
    bool carry_ok = true;
    if (destination_mtp) {
        carry_ok = complete_mtp
            ? common_speculative_set_state(spec, slot.id, snapshot.data_spec, snapshot.pos_tgt)
            : common_speculative_set_state(spec, slot.id, {}, -1);
        if (carry_ok && complete_mtp) {
            carry_ok = common_speculative_is_ready(spec, slot.id, snapshot.pos_tgt + 1);
        }
    }
    if (!target_ok || !draft_ok || !carry_ok) {
        slot.prompt_clear();
        error = !target_ok ? "adaptive slot target restore failed" :
            !draft_ok ? "adaptive slot draft restore failed" : "adaptive MTP carry restore failed";
        return false;
    }

    const auto state_matches = [](llama_context * ctx, llama_seq_id seq_id,
            const std::vector<uint8_t> & expected, llama_state_seq_flags flags) {
        const size_t size = llama_state_seq_get_size_ext(ctx, seq_id, flags);
        if (size != expected.size()) {
            return false;
        }
        std::vector<uint8_t> actual(size);
        return llama_state_seq_get_data_ext(ctx, actual.data(), actual.size(), seq_id, flags) == size &&
            actual == expected;
    };
    const auto restore_full_state = [&]() {
        const bool target = llama_state_seq_set_data_ext(ctx_tgt, snapshot.data_tgt.data(), snapshot.data_tgt.size(),
                slot.id, LLAMA_STATE_SEQ_FLAGS_NONE) == snapshot.data_tgt.size() &&
            llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id) == snapshot.pos_tgt;
        const bool draft = !complete_mtp ||
            (llama_state_seq_set_data_ext(ctx_dft, snapshot.data_dft.data(), snapshot.data_dft.size(),
                    slot.id, LLAMA_STATE_SEQ_FLAGS_NONE) == snapshot.data_dft.size() &&
                llama_memory_seq_pos_max(llama_get_memory(ctx_dft), slot.id) == snapshot.pos_dft);
        const bool carry = !destination_mtp ||
            (complete_mtp
                ? (common_speculative_set_state(spec, slot.id, snapshot.data_spec, snapshot.pos_tgt) &&
                   common_speculative_is_ready(spec, slot.id, snapshot.pos_tgt + 1))
                : common_speculative_set_state(spec, slot.id, {}, -1));
        return target && draft && carry;
    };

    bool dropped_checkpoint = false;
    llama_pos previous_pos_max = -1;
    int64_t previous_n_tokens = -1;
    for (const auto & source : snapshot.checkpoints) {
        if (source.n_tokens <= previous_n_tokens || source.pos_max <= previous_pos_max) {
            dropped_checkpoint = true;
            continue;
        }
        previous_n_tokens = source.n_tokens;
        previous_pos_max = source.pos_max;

        auto checkpoint = adaptive_slot_make_checkpoint(
            source, snapshot, ctx_tgt, ctx_dft, candidate.tokens, dropped_checkpoint);
        if (!checkpoint) {
            continue;
        }

        const bool target_restored = checkpoint->restore_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        const bool target_bounds = target_restored &&
            llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id) == checkpoint->pos_min &&
            llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id) == checkpoint->pos_max;
        const bool target_state = target_bounds &&
            state_matches(ctx_tgt, slot.id, checkpoint->data_tgt, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        bool valid = target_state;
        bool draft_restored = true;
        bool draft_bounds = true;
        bool draft_state = true;
        if (valid && ctx_dft) {
            draft_restored = checkpoint->restore_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == common_checkpoint_restore::restored;
            const auto draft_pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_dft), slot.id);
            const auto draft_pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), slot.id);
            // MTP draft state may retain its base from position zero while the
            // target PARTIAL_ONLY state starts at the checkpoint window. The
            // end position must still align exactly; a swapped blob therefore
            // fails even when both checkpoints have the same byte size.
            draft_bounds = draft_restored && draft_pos_min >= 0 && draft_pos_min <= checkpoint->pos_min &&
                draft_pos_max == checkpoint->pos_max;
            draft_state = draft_bounds &&
                state_matches(ctx_dft, slot.id, checkpoint->data_dft, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            valid = draft_state;
            if (!draft_bounds) {
                SLT_WRN(slot, "adaptive checkpoint draft bounds actual=[%d,%d] expected_end=%d start_max=%d\n",
                        draft_pos_min, draft_pos_max, checkpoint->pos_max, checkpoint->pos_min);
            }
        }
        if (valid && spec && !checkpoint->data_spec.empty()) {
            valid = common_speculative_set_state(spec, slot.id, checkpoint->data_spec, checkpoint->pos_max) &&
                common_speculative_is_ready(spec, slot.id, checkpoint->pos_max + 1);
            if (valid) {
                std::vector<uint8_t> actual_spec;
                common_speculative_get_state(spec, slot.id, actual_spec);
                valid = actual_spec == checkpoint->data_spec;
            }
        }

        if (valid) {
            candidate.checkpoints.push_back(std::move(checkpoint));
        } else {
            SLT_WRN(slot, "discarding adaptive checkpoint n_tokens=%" PRId64 " pos=[%d,%d] target=(restore:%d bounds:%d state:%d) draft=(restore:%d bounds:%d state:%d)\n",
                    source.n_tokens, source.pos_min, source.pos_max, target_restored, target_bounds, target_state,
                    draft_restored, draft_bounds, draft_state);
            dropped_checkpoint = true;
        }
        if (!restore_full_state()) {
            slot.prompt_clear();
            error = "adaptive slot checkpoint validation disturbed the restored state";
            return false;
        }
    }
    if (dropped_checkpoint) {
        SLT_WRN(slot, "%s", "adaptive slot snapshot dropped incompatible checkpoints\n");
    }

    slot.prompt = std::move(candidate);
    slot.bootstrap_pending = destination_mtp && !complete_mtp;
    SLT_TRC(slot, "adaptive slot restore published tokens=%zu checkpoints=%zu complete_mtp=%d bootstrap=%d\n",
            slot.prompt.tokens.size(), slot.prompt.checkpoints.size(), complete_mtp, slot.bootstrap_pending);
    if (slot.bootstrap_pending) {
        SLT_INF(slot, "%s", "adaptive slot target restored; MTP bootstrap is required before a cache hit\n");
    }
    return true;
}

} // namespace

// returns 0 on success
// caller need to update prompt.tokens after a successful call to keep track of the processing progress
// note: this is not a member of server_slot because we want to run it inside yield_to_queue
//       slot is passed as const to avoid accidental modification of the slot state
//       some pointers are allowed to be used, they are not used by to_json()
static int process_mtmd_chunk(const server_slot & slot, mtmd::batch_ptr & mbatch, size_t idx, size_t & n_tokens_out) {
    GGML_ASSERT(slot.mctx);
    const auto & mctx = slot.mctx;
    const auto & input_tokens = slot.task->tokens;
    const auto & chunk = input_tokens.find_chunk(idx);
    int32_t res = 0;

    auto try_decode = [&]() -> int32_t {
        if (mbatch) {
            float * embd = mtmd_batch_get_output_embd(mbatch.get(), chunk.get());
            if (embd) {
                void * cb_data = slot.spec;
                static auto cb = [](llama_batch batch, void * user_data) {
                    common_speculative * spec = static_cast<common_speculative *>(user_data);
                    if (!common_speculative_process(spec, batch)) {
                        return 1;
                    }
                    return 0;
                };

                llama_pos new_n_past; // unused for now
                res = mtmd_helper_decode_image_chunk(
                    mctx,
                    slot.ctx_tgt,
                    chunk.get(),
                    embd,
                    slot.prompt.tokens.pos_next(),
                    slot.id,
                    llama_n_batch(slot.ctx_tgt),
                    &new_n_past,
                    cb,
                    cb_data
                );
                if (res != 0) {
                    SLT_ERR(slot, "failed to decode mtmd chunk, idx = %zu, res = %d\n", idx, res);
                    return -1;
                }
                n_tokens_out = mtmd_input_chunk_get_n_tokens(chunk.get());
                return 0; // success
            }
        }
        return 1; // (non-error) need to create & encode batch
    };

    // if the batch is already exist, try searching & encode
    res = try_decode();
    if (res == 0) {
        return 0;
    }
    if (res < 0) {
        // fatal error
        return res;
    }

    // otherwise, the batch is either uninitialized or is used up
    // we need to create & encode a new batch
    mbatch.reset(mtmd_batch_init(mctx));
    res = mtmd_batch_add_chunk(mbatch.get(), chunk.get());
    GGML_ASSERT(res == 0); // we should never have an empty batch

    // try batching as much as possible
    int n_added = 1;
    size_t idx_cur = idx;
    while (res == 0) {
        auto [next_chunk, next_idx] = input_tokens.find_next_media_chunk(idx_cur);
        if (next_chunk == nullptr) {
            break;
        }
        res = mtmd_batch_add_chunk(mbatch.get(), next_chunk->get());
        n_added += (res == 0 ? 1 : 0);
        idx_cur = next_idx;
        SLT_DBG(slot, "try adding media chunk idx = %zu to batch, res = %d\n", next_idx, res);
        // if res != 0, batch is full or chunk is not compatible -> this loop breaks
    }

    // TODO @ngxson : move this log line to debug when it become more stable
    SLT_TRC(slot, "encoding mtmd batch from idx = %zu, n_chunks = %d\n", idx, n_added);

    res = mtmd_batch_encode(mbatch.get());
    if (res != 0) {
        SLT_ERR(slot, "failed to encode mtmd batch for chunk idx = %zu, res = %d\n", idx, res);
        return -1;
    }

    return try_decode();
}

//
// server_context_impl (private implementation)
//

struct server_context_impl {
    friend struct server_context;

public:
    // only use these pointers outside of this class:
    //  - when not in sleeping state
    //  - and, with thread-safe APIs (e.g., tokenizer calls)
    llama_model * model_tgt = nullptr;

    mtmd_context * mctx = nullptr;
    // note: video_params.ffmpeg_bin_dir points into params_base, which outlives this struct
    mtmd_helper_init_opt init_opt = mtmd_helper_init_opt_default();
    const llama_vocab * vocab = nullptr;

    server_queue    queue_tasks;
    server_response queue_results;

    // note: chat_params must not be refreshed upon existing sleeping state
    server_chat_params chat_params;

    server_state_callback_t callback_state = [](server_state, json) -> void {};

    server_context_impl() {
        mtmd_helper_log_set(common_log_default_callback, nullptr);
    }

    ~server_context_impl() {
        gpu_power.shutdown();
        if (!sleeping) {
            // destroy() is already called when entering sleeping state
            // we don't call it again here to avoid double free
            destroy();
        }
    }

    server_metrics get_metrics() const {
        return metrics;
    }

    void reset_metrics_bucket() {
        metrics.reset_bucket();
    }

    server_context_adaptive_status get_adaptive_status() const {
        std::lock_guard<std::mutex> lock(adaptive_status_mutex);
        return adaptive_status_snapshot;
    }

private:
    // note: accessing these fields outside of this class is not thread-safe
    // use server_context methods instead

    common_params params_base;

    server_gpu_power gpu_power;

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    common_init_result_ptr llama_init;

    llama_context * ctx_tgt = nullptr;

    server_batch batch;

    llama_model   * model_dft = nullptr;
    llama_context * ctx_dft   = nullptr;

    // Prepared before loading the resident model; used to reject stale explicit
    // snapshots without hashing a live model again during a transition.
    std::optional<server_model_identity> adaptive_model_identity;

    common_speculative_init_result_ptr spec_init;

    common_context_seq_rm_type ctx_tgt_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    common_context_seq_rm_type ctx_dft_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;

    common_speculative_ptr spec;

    bool add_bos_token = true;

    int32_t n_ctx; // total context for all clients / slots
    int32_t adaptive_long_ctx = 0;
    int32_t adaptive_draft_n_medium = 2;
    int32_t adaptive_draft_n_short  = 4;
    int32_t adaptive_batch_normal   = 256;
    int32_t adaptive_ubatch_normal  = 256;
    ggml_type adaptive_cache_type_k_normal = GGML_TYPE_Q4_0;
    ggml_type adaptive_cache_type_v_normal = GGML_TYPE_Q4_0;
    llama_kvarn_params adaptive_kvarn_normal{};
    int32_t adaptive_cache_kvarn_bits_k_normal = 0;
    int32_t adaptive_cache_kvarn_bits_v_normal = 0;

    int32_t adaptive_batch_xlong    = 64;
    int32_t adaptive_ubatch_xlong   = 64;

    int32_t adaptive_batch_xxlong   = 64;
    int32_t adaptive_ubatch_xxlong  = 64;
    ggml_type adaptive_cache_type_k_xxlong = GGML_TYPE_Q4_0;
    ggml_type adaptive_cache_type_v_xxlong = GGML_TYPE_Q4_0;
    llama_kvarn_params adaptive_kvarn_xxlong{};
    int32_t adaptive_cache_kvarn_bits_k_xxlong = 4;
    int32_t adaptive_cache_kvarn_bits_v_xxlong = 4;

    int32_t adaptive_max_ctx() const {
        if (params_base.ctx_size_xxlong > 0) {
            return params_base.ctx_size_xxlong;
        }
        if (params_base.ctx_size_xlong > 0) {
            return params_base.ctx_size_xlong;
        }
        return adaptive_long_ctx;
    }

    common_context_profile active_context_profile = COMMON_CONTEXT_PROFILE_LONG;
    bool adaptive_context_unavailable = false;
    bool adaptive_context_transitioning = false;

    enum adaptive_status_state : int {
        ADAPTIVE_STATUS_DISABLED = 0,
        ADAPTIVE_STATUS_READY,
        ADAPTIVE_STATUS_TRANSITIONING,
        ADAPTIVE_STATUS_UNAVAILABLE,
    };

    mutable std::mutex adaptive_status_mutex;
    server_context_adaptive_status adaptive_status_snapshot;
    bool elastic_paged_context = false;
    uint32_t paged_admission_blocks = 0;

    // set to llama_model_n_swa(model)
    // if swa_full is enabled, this is set to 0 to simulate a non-SWA model
    int32_t n_swa;

    // slots / clients
    std::vector<server_slot> slots;

    int trace = 0;        // env: LLAMA_TRACE
    int slots_debug = 0;  // env: LLAMA_SERVER_SLOTS_DEBUG
    int slots_n_diff = 0; // env: LLAMA_SERVER_SLOTS_N_DIFF

    int n_empty_consecutive = 0;

    std::unique_ptr<server_prompt_cache> prompt_cache;

    server_metrics metrics;

    // queued prompt stats - llama_decode() is async, so the timing is only valid after a sync
    // note: kept out of server_metrics, which is copied as-is into the task result
    int64_t  t_decode_start  = 0; // start of the last submitted decode
    int64_t  t_prompt_start  = 0; // start of the oldest queued prompt decode
    uint64_t n_prompt_queued = 0;

    json json_ui_settings = json::object();

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;

    std::string model_name; // name of the loaded model, to be used by API
    std::set<std::string> model_aliases; // additional names for the model
    std::set<std::string> model_tags;    // informational tags

    bool sleeping = false;

    // --- auto disk prompt/KV cache (opt-in: --slot-save-auto) ---
    // Default-constructed: empty and untouched when the feature is OFF (invariant 1).
    // `cur_fp` is the live model fingerprint, computed once at load (only when enabled).
    auto_cache_index auto_idx;
    model_fp         cur_fp;

    // The ONE gate for the entire auto disk cache. When false, NO hook below does
    // any work (no scan, no hash, no alloc). This is invariant 1 - the first
    // statement of every auto_* hook is `if (!auto_cache_enabled()) return;`.
    bool auto_cache_enabled() const {
        return params_base.slot_save_auto && !params_base.slot_save_path.empty() &&
            params_base.lora_adapters.empty() && params_base.control_vectors.empty();
    }

    int64_t t_last_load_progress_ms = 0;

    void apply_profile_params(common_context_profile profile) {
        if (profile == COMMON_CONTEXT_PROFILE_MTP_SHORT) {
            params_base.n_ctx = params_base.ctx_size_mtp_short;
            params_base.speculative.draft.n_max = adaptive_draft_n_short;
            params_base.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
            params_base.n_batch = adaptive_batch_normal;
            params_base.n_ubatch = adaptive_ubatch_normal;
            params_base.cache_type_k = adaptive_cache_type_k_normal;
            params_base.cache_type_v = adaptive_cache_type_v_normal;
            params_base.kvarn = adaptive_kvarn_normal;
            params_base.cache_kvarn_bits_k = adaptive_cache_kvarn_bits_k_normal;
            params_base.cache_kvarn_bits_v = adaptive_cache_kvarn_bits_v_normal;
        } else if (profile == COMMON_CONTEXT_PROFILE_MTP) {
            params_base.n_ctx = params_base.ctx_size_mtp;
            params_base.speculative.draft.n_max = adaptive_draft_n_medium;
            params_base.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
            params_base.n_batch = adaptive_batch_normal;
            params_base.n_ubatch = adaptive_ubatch_normal;
            params_base.cache_type_k = adaptive_cache_type_k_normal;
            params_base.cache_type_v = adaptive_cache_type_v_normal;
            params_base.kvarn = adaptive_kvarn_normal;
            params_base.cache_kvarn_bits_k = adaptive_cache_kvarn_bits_k_normal;
            params_base.cache_kvarn_bits_v = adaptive_cache_kvarn_bits_v_normal;
        } else if (profile == COMMON_CONTEXT_PROFILE_LONG) {
            params_base.n_ctx = adaptive_long_ctx;
            params_base.speculative.draft.n_max = 0;
            params_base.speculative.types = {};
            params_base.n_batch = adaptive_batch_normal;
            params_base.n_ubatch = adaptive_ubatch_normal;
            params_base.cache_type_k = adaptive_cache_type_k_normal;
            params_base.cache_type_v = adaptive_cache_type_v_normal;
            params_base.kvarn = adaptive_kvarn_normal;
            params_base.cache_kvarn_bits_k = adaptive_cache_kvarn_bits_k_normal;
            params_base.cache_kvarn_bits_v = adaptive_cache_kvarn_bits_v_normal;
        } else if (profile == COMMON_CONTEXT_PROFILE_XLONG) {
            params_base.n_ctx = params_base.ctx_size_xlong;
            params_base.speculative.draft.n_max = 0;
            params_base.speculative.types = {};
            params_base.n_batch = adaptive_batch_xlong;
            params_base.n_ubatch = adaptive_ubatch_xlong;
            params_base.cache_type_k = adaptive_cache_type_k_normal;
            params_base.cache_type_v = adaptive_cache_type_v_normal;
            params_base.kvarn = adaptive_kvarn_normal;
            params_base.cache_kvarn_bits_k = adaptive_cache_kvarn_bits_k_normal;
            params_base.cache_kvarn_bits_v = adaptive_cache_kvarn_bits_v_normal;
        } else if (profile == COMMON_CONTEXT_PROFILE_XXLONG) {
            params_base.n_ctx = params_base.ctx_size_xxlong;
            params_base.speculative.draft.n_max = 0;
            params_base.speculative.types = {};
            params_base.n_batch = adaptive_batch_xxlong;
            params_base.n_ubatch = adaptive_ubatch_xxlong;
            params_base.cache_type_k = adaptive_cache_type_k_xxlong;
            params_base.cache_type_v = adaptive_cache_type_v_xxlong;
            params_base.kvarn = adaptive_kvarn_xxlong;
            params_base.cache_kvarn_bits_k = adaptive_cache_kvarn_bits_k_xxlong;
            params_base.cache_kvarn_bits_v = adaptive_cache_kvarn_bits_v_xxlong;
        }
        params_base.n_parallel = 1;
        const auto output_limits = server_output_limits(params_base);
        params_base.n_outputs_max = output_limits.total;
        params_base.n_outputs_max_per_seq = output_limits.per_seq;
    }

    static std::string adaptive_status_profile_name(int profile) {
        switch (profile) {
            case COMMON_CONTEXT_PROFILE_MTP_SHORT: return "mtp-short";
            case COMMON_CONTEXT_PROFILE_MTP:       return "mtp";
            case COMMON_CONTEXT_PROFILE_LONG:      return "long";
            case COMMON_CONTEXT_PROFILE_XLONG:     return "xlong";
            case COMMON_CONTEXT_PROFILE_XXLONG:    return "xxlong";
            default:                               return "unknown";
        }
    }

    static std::string adaptive_status_state_name(int state) {
        switch (state) {
            case ADAPTIVE_STATUS_READY:         return "ready";
            case ADAPTIVE_STATUS_TRANSITIONING: return "transitioning";
            case ADAPTIVE_STATUS_UNAVAILABLE:   return "unavailable";
            default:                            return "disabled";
        }
    }

    void publish_adaptive_status() {
        const bool enabled = common_context_is_adaptive(params_base);
        server_context_adaptive_status snapshot;
        snapshot.enabled = enabled;
        snapshot.profile = !enabled ? "disabled" : adaptive_context_unavailable ? "none" : adaptive_status_profile_name(static_cast<int>(active_context_profile));
        snapshot.context_size_long = enabled ? n_ctx_slot() : 0;
        snapshot.context_size = enabled && !adaptive_context_unavailable && ctx_tgt ? active_n_ctx_slot() : 0;
        snapshot.mtp_weights_resident = enabled && model_tgt && llama_model_mtp_weights_get_info(model_tgt).resident;
        const int state = !enabled ? ADAPTIVE_STATUS_DISABLED
            : adaptive_context_unavailable ? ADAPTIVE_STATUS_UNAVAILABLE
            : adaptive_context_transitioning ? ADAPTIVE_STATUS_TRANSITIONING
            : ADAPTIVE_STATUS_READY;
        snapshot.state = adaptive_status_state_name(state);
        std::lock_guard<std::mutex> lock(adaptive_status_mutex);
        adaptive_status_snapshot = std::move(snapshot);
    }

    void destroy() {
        prompt_cache.reset();
        spec.reset();
        spec_init.reset();

        ctx_dft   = nullptr;
        model_dft = nullptr;

        llama_init.reset();

        ctx_tgt = nullptr;
        model_tgt = nullptr;

        mtmd_free(mctx);
        mctx = nullptr;
    }

    // ----- auto disk cache: fingerprint, index, restore, save (all gated by auto_cache_enabled()) -----

    // hash of the active LoRA set (ids/scales) for the fingerprint; 0 when no adapter is active.
    static uint64_t auto_lora_hash(const std::vector<common_adapter_lora_info> & lora) {
        uint64_t h = 0xcbf29ce484222325ULL;
        bool any = false;
        for (const auto & a : lora) {
            if (a.scale == 0.0f) {
                continue; // disabled adapter does not affect inference identity
            }
            any = true;
            for (char c : a.path) {
                h ^= (uint64_t) (unsigned char) c; h *= 0x100000001b3ULL;
            }
            uint32_t sc; std::memcpy(&sc, &a.scale, sizeof(sc));
            h = auto_hash_mix(h, (int32_t) sc);
        }
        return any ? h : 0;
    }

    // Stable model hash used by the unified token-prefix index. Context capacity
    // and block size are intentionally not part of it.
    uint64_t auto_model_hash_base() const {
        char desc[256] = {0};
        llama_model_desc(model_tgt, desc, sizeof(desc));
        uint64_t h = 0xcbf29ce484222325ULL;
        for (const char * p = desc; *p; ++p) {
            h ^= (uint64_t) (unsigned char) *p; h *= 0x100000001b3ULL;
        }
        const uint64_t sz = llama_model_size(model_tgt);
        const uint64_t np = llama_model_n_params(model_tgt);
        h = auto_hash_mix(h, (int32_t) (sz & 0xFFFFFFFFu)); h = auto_hash_mix(h, (int32_t) (sz >> 32));
        h = auto_hash_mix(h, (int32_t) (np & 0xFFFFFFFFu)); h = auto_hash_mix(h, (int32_t) (np >> 32));

        GGML_ASSERT(adaptive_model_identity);
        const std::string identity = adaptive_model_identity->fingerprint() + common_prompt_cache_layout(ctx_tgt);
        for (unsigned char c : identity) {
            h = auto_hash_mix(h, c);
        }
        return h;
    }

    // Legacy .meta files used the context and block as part of fp_model. This
    // exact formula is retained only to authenticate an upgrade into the
    // unified envelope; no new file uses it as its identity.
    uint64_t auto_legacy_model_hash(uint32_t n_ctx_value, uint32_t block) const {
        uint64_t h = auto_model_hash_base();
        h = auto_hash_mix(h, (int32_t) n_ctx_value);
        h = auto_hash_mix(h, (int32_t) block);
        return h;
    }

    bool auto_legacy_fp_compatible(const model_fp & legacy) const {
        if (!legacy.fp_n_ctx || !legacy.fp_block ||
                legacy.fp_model != auto_legacy_model_hash(legacy.fp_n_ctx, legacy.fp_block)) {
            return false;
        }
        // Context capacity and index block are deliberately allowed to differ;
        // all representation/position-affecting fields remain exact matches.
        model_fp expected = cur_fp;
        expected.fp_model = legacy.fp_model;
        expected.fp_n_ctx = legacy.fp_n_ctx;
        expected.fp_block = legacy.fp_block;
        return expected == legacy;
    }

    // Compute the live model fingerprint once at load (invariant 3). Pure-CPU; only called from
    // an auto_cache_enabled() branch so it costs nothing when OFF.
    // See README "Automatic disk prompt cache" for which flags invalidate the cache.
    model_fp auto_compute_fingerprint() const {
        model_fp fp;
        // The shared-store salt deliberately excludes the active context
        // capacity and configured block size: a snapshot made by a 32k/256
        // child must be discoverable by a 97k child, and a 64-sized child may
        // still consume the same target state when it fits.
        fp.fp_model       = auto_model_hash_base();
        fp.fp_n_vocab     = (uint32_t) llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));
        fp.fp_n_ctx_train = (uint32_t) llama_model_n_ctx_train(model_tgt);
        fp.fp_n_embd      = (uint32_t) llama_model_n_embd(model_tgt);
        fp.fp_n_layer     = (uint32_t) llama_model_n_layer(model_tgt);
        fp.fp_rope_type   = (uint32_t) llama_model_rope_type(model_tgt);
        // K/V cache type has no live-ctx getter - capture from the server's own params (the value
        // used to construct ctx_tgt). Blob-layout-critical: a Q4_0-KV blob into an F16 ctx corrupts.
        fp.fp_cache_k     = (uint32_t) params_base.cache_type_k;
        fp.fp_cache_v     = (uint32_t) params_base.cache_type_v;
        fp.fp_n_ctx       = (uint32_t) llama_n_ctx_seq(ctx_tgt);
        fp.fp_kv_full     = (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) ? 1u : 0u;
        fp.fp_block       = (uint32_t) params_base.slot_save_block;
        // effective rope scale (positions are baked into the saved state). rope_freq_scale==0 means
        // "use the model's trained value", so fall back to that for a stable comparison.
        float rs = params_base.rope_freq_scale != 0.0f
                       ? params_base.rope_freq_scale
                       : llama_model_rope_freq_scale_train(model_tgt);
        uint32_t rsb; std::memcpy(&rsb, &rs, sizeof(rsb));
        fp.fp_rope_scale  = (uint64_t) rsb;
        // rope_freq_base + the five YaRN params also bake positions into the saved
        // KV, so they MUST be part of identity. There is no public getter for the model's trained
        // rope base, so we normalize "use-model-default" to a single canonical 0 sentinel: when the
        // operator left the knob at its default (rope_freq_base==0; YaRN floats<0, i.e. -1.0 "auto";
        // yarn_orig_ctx<=0), we store 0. Two runs that both rely on the model default thus match;
        // any explicit override (or two different overrides) yields a different fp and refuses
        // (conservative - a needless miss is safe, a wrong restore is not). yarn_orig_ctx is an int.
        auto bitcast_f = [](float v) -> uint32_t { uint32_t u; std::memcpy(&u, &v, sizeof(u)); return u; };
        auto norm_yarn = [&](float v) -> uint32_t { return v < 0.0f ? 0u : bitcast_f(v); }; // <0 == model default
        const float rb = params_base.rope_freq_base > 0.0f ? params_base.rope_freq_base : 0.0f; // 0 == model default
        uint32_t rbb; std::memcpy(&rbb, &rb, sizeof(rbb));
        fp.fp_rope_base      = (uint64_t) rbb;
        fp.fp_yarn_ext       = norm_yarn(params_base.yarn_ext_factor);
        fp.fp_yarn_attn      = norm_yarn(params_base.yarn_attn_factor);
        fp.fp_yarn_beta_fast = norm_yarn(params_base.yarn_beta_fast);
        fp.fp_yarn_beta_slow = norm_yarn(params_base.yarn_beta_slow);
        fp.fp_yarn_orig_ctx  = params_base.yarn_orig_ctx > 0 ? (uint32_t) params_base.yarn_orig_ctx : 0u;
        // LoRA: fingerprint the global adapter set so a snapshot under adapter A never restores
        // under B. (Per-request adapter overrides additionally gate at the restore hook.)
        fp.fp_lora        = auto_lora_hash(params_base.lora_adapters);
        // deployment-shape bit (invariant 3): text-only server vs --mmproj server get
        // disjoint stores (mmproj-aware rope/projector wiring can change the text KV layout).
        fp.fp_mmproj_loaded = (mctx != nullptr) ? 1u : 0u;
        return fp;
    }

    // Auto-snapshot filename: model-fp prefix lets the startup scan reject foreign-model files by
    // name before opening anything; chain-hash + token-count make it deterministic across processes
    // (a same-prefix save from another process yields the same name -> atomic-rename-idempotent).
    std::string auto_state_filename(uint64_t chain_hash, size_t n_tokens) const {
        char buf[96]; // "auto-" + 16 hex fp + "-" + 16 hex hash + "-" + up to 20-digit count + ".bin" < 96
        snprintf(buf, sizeof(buf), "auto-%016" PRIx64 "-%016" PRIx64 "-%zu.bin",
                 cur_fp.fp_model, chain_hash, n_tokens);
        return (std::filesystem::path(params_base.slot_save_path) / buf).string();
    }

    std::optional<server_route_state_file> auto_find_exact_snapshot(const llama_tokens & tokens) {
        if (!auto_cache_enabled() || !adaptive_model_identity || tokens.empty()) {
            return std::nullopt;
        }
        const uint32_t block = auto_index_block(params_base.slot_save_block);
        const auto bhs = auto_block_hashes(tokens, (int) block, cur_fp.fp_model);
        if (bhs.empty()) {
            return std::nullopt;
        }
        uint64_t full_hash = bhs.back();
        for (size_t i = tokens.size() - tokens.size() % block; i < tokens.size(); ++i) {
            full_hash = auto_hash_mix(full_hash, tokens[i]);
        }
        const std::string expected_path = auto_state_filename(full_hash, tokens.size());
        if (auto file = read_unified_snapshot(expected_path, false); file &&
                file->model == adaptive_model_identity->fingerprint() &&
                common_prompt_cache_layout_reusable(file->layout, common_prompt_cache_layout(ctx_tgt)) &&
                file->tokens == tokens) {
            return file;
        }

        // A snapshot written by another compatible child may use a different
        // filename/block namespace. Reuse it by identity/content, still via
        // the same canonical reader and without writing a route copy.
        std::vector<std::string> indexed;
        {
            std::lock_guard<std::mutex> lk(auto_idx.mtx);
            indexed.assign(auto_idx.indexed_files.begin(), auto_idx.indexed_files.end());
        }
        for (const auto & path : indexed) {
            if (auto file = read_unified_snapshot(path, false); file &&
                    file->model == adaptive_model_identity->fingerprint() &&
                    common_prompt_cache_layout_reusable(file->layout, common_prompt_cache_layout(ctx_tgt)) &&
                    file->tokens == tokens) {
                return file;
            }
        }
        return std::nullopt;
    }

    // Deterministic automatic-store path for an exact token prefix in this
    // child's own layout namespace.
    std::string auto_state_path_for(const llama_tokens & tokens) const {
        if (tokens.empty()) {
            return {};
        }
        const uint32_t block = auto_index_block(params_base.slot_save_block);
        const auto bhs = auto_block_hashes(tokens, (int) block, cur_fp.fp_model);
        if (bhs.empty()) {
            return {};
        }
        uint64_t full_hash = bhs.back();
        for (size_t i = tokens.size() - tokens.size() % block; i < tokens.size(); ++i) {
            full_hash = auto_hash_mix(full_hash, tokens[i]);
        }
        return auto_state_filename(full_hash, tokens.size());
    }

    // Explicit q4_0/q4_0 -> native compact snapshot conversion for the route
    // restore path. The converted snapshot is published in the automatic store
    // under its deterministic name with recorded provenance, so later restores
    // reuse it without converting again. Returns the converted path, empty on
    // any failure. This is lossy and is never reported as a native prefill.
    std::string convert_route_snapshot(const server_route_state_file & source_file,
            server_route_state_lease * source_lease) {
        if (!ctx_tgt || !adaptive_model_identity || source_file.tokens.empty()) {
            return {};
        }
        const std::string target_layout = common_prompt_cache_layout(ctx_tgt);
        std::string source_type_k;
        std::string source_type_v;
        if (!common_prompt_cache_layout_convertible(source_file.layout, target_layout,
                &source_type_k, &source_type_v)) {
            SRV_WRN("route conversion rejected: source layout is not convertible to this child's layout\n  source=%s\n  target=%s\n",
                    source_file.layout.c_str(), target_layout.c_str());
            return {};
        }
        if (params_base.cache_kvarn_bits_k == 0 || params_base.cache_kvarn_bits_v == 0) {
            SRV_WRN("%s", "route conversion rejected: destination is not a KVarN cache\n");
            return {};
        }
        // The source format is validated by the converter itself; only the
        // restricted q4_0/q4_0 shape parses."
        if (!auto_cache_enabled()) {
            SRV_WRN("%s", "route conversion requires the automatic snapshot store (slot-save-auto + slot-save-path)\n");
            return {};
        }
        const std::string converted_path = auto_state_path_for(source_file.tokens);
        if (converted_path.empty()) {
            return {};
        }

        // Reuse a previously converted snapshot whenever it matches exactly.
        if (auto existing = read_unified_snapshot(converted_path, true); existing &&
                existing->model == adaptive_model_identity->fingerprint() &&
                existing->layout == target_layout && existing->tokens == source_file.tokens) {
            SRV_INF("converted route snapshot reused: path=%s tokens=%zu\n",
                    converted_path.c_str(), source_file.tokens.size());
            return converted_path;
        }

        static std::atomic<uint64_t> conversion_nonce{0};
        const std::string native_tmp = converted_path + ".tmp-convert-native-" +
            std::to_string(ggml_time_us()) + "-" +
            std::to_string(conversion_nonce.fetch_add(1, std::memory_order_relaxed));
        llama_tokens tokens(source_file.tokens.size());
        size_t count = 0;
        const int64_t conversion_started = ggml_time_us();
        const size_t converted = llama_state_seq_convert_file(ctx_tgt,
                source_file.state_path.c_str(), 0, source_file.state_bytes, source_file.state_checksum,
                native_tmp.c_str(), tokens.data(), tokens.size(), &count);
        if (!converted || count != source_file.tokens.size() || tokens != source_file.tokens) {
            std::error_code ec;
            std::filesystem::remove(native_tmp, ec);
            SRV_WRN("route conversion failed for %s elapsed_ms=%.3f backend=cpu source=disk\n",
                    source_file.state_path.c_str(), (ggml_time_us() - conversion_started) / 1000.0);
            return {};
        }
        const double conversion_ms = (ggml_time_us() - conversion_started) / 1000.0;
        const std::string target_format = string_format("kvarn_k%dv%d_g128",
                params_base.cache_kvarn_bits_k, params_base.cache_kvarn_bits_v);
        const std::string provenance = common_json{
            {"converter", 1},
            {"source_path", std::filesystem::path(source_file.state_path).filename().string()},
            {"source_checksum", source_file.state_checksum},
            {"source_format", "q4_0/q4_0"},
            {"target_format", target_format},
        }.dump();
        // Publication takes the exclusive store lock, so the shared source
        // reference (which also shares that lock) must be dropped first. The
        // converted native file is already complete and self-contained.
        if (source_lease != nullptr) {
            source_lease->release();
        }
        if (!server_route_state_adopt_native(native_tmp, converted_path,
                adaptive_model_identity->fingerprint(), target_layout, source_file.tokens,
                auto_store_max_tokens(), auto_index_block(params_base.slot_save_block),
                auto_store_max_bytes(), &provenance)) {
            std::error_code ec;
            std::filesystem::remove(native_tmp, ec);
            SRV_WRN("route conversion publication failed for %s\n", converted_path.c_str());
            return {};
        }
        std::error_code ec;
        std::filesystem::remove(native_tmp, ec);
        SRV_INF("route snapshot converted: %s -> %s tokens=%zu bytes=%zu target=%s "
                "conversion_ms=%.3f backend=cpu source=disk layout=%s\n",
                source_file.state_path.c_str(), converted_path.c_str(), source_file.tokens.size(),
                converted, target_format.c_str(), conversion_ms, target_layout.c_str());
        return converted_path;
    }

    uint64_t auto_store_max_bytes() const {
        return params_base.slot_save_max_bytes > 0
            ? uint64_t(params_base.slot_save_max_bytes) : SIZE_MAX;
    }

    uint32_t auto_store_max_tokens() const {
        uint64_t result = std::max<int32_t>(llama_n_ctx_seq(ctx_tgt), adaptive_max_ctx());
        result = std::max<uint64_t>(result, params_base.ctx_size_mtp);
        result = std::max<uint64_t>(result, params_base.ctx_size_mtp_short);
        result = std::max<uint64_t>(result, params_base.ctx_size_xlong);
        result = std::max<uint64_t>(result, params_base.ctx_size_xxlong);
        return result > UINT32_MAX ? UINT32_MAX : (uint32_t) result;
    }

    std::optional<server_route_state_file> read_unified_snapshot(const std::string & path,
            bool verify_payload = true) const {
        try {
            return server_route_state_read(path, auto_store_max_bytes(), auto_store_max_tokens(), verify_payload);
        } catch (const std::exception & error) {
            SRV_WRN("unified snapshot ignored at %s: %s\n", path.c_str(), error.what());
            return std::nullopt;
        }
    }

    // Keep each snapshot: a longer state cannot replace a shorter branch point.
    void auto_index_insert_locked(uint32_t block, uint64_t boundary, const auto_cache_entry & e) {
        auto_idx.by_boundary.emplace(auto_boundary_key(block, boundary), e);
        auto_idx.index_blocks.insert(auto_index_block(block));
    }

    // Read canonical snapshot metadata and skip incompatible files. Caller holds auto_idx.mtx.
    // Both route handoff files and auto-cache files are intentionally discovered here.
    void auto_index_scan_locked() {
        std::error_code mec;
        const auto dmt = std::filesystem::last_write_time(params_base.slot_save_path, mec);
        if (!mec) {
            auto_idx.dir_mtime = dmt; // snapshot the dir mtime we are scanning at
        }
        const std::string current_model = adaptive_model_identity->fingerprint();
        const std::string current_layout = common_prompt_cache_layout(ctx_tgt);
        std::error_code ec;
        for (std::filesystem::directory_iterator it(params_base.slot_save_path, ec), end;
             !ec && it != end; it.increment(ec)) {
            std::error_code fec;
            if (!it->is_regular_file(fec) || fec) {
                continue;
            }
            const std::string p = it->path().string();
            const std::string base = it->path().filename().string();
            if (p.size() < 4 || p.compare(p.size() - 4, 4, ".bin") != 0 ||
                    base.find(".tmp") != std::string::npos || auto_idx.indexed_files.count(p)) {
                continue;
            }
            auto file = read_unified_snapshot(p, false);
            if (!file) {
                // Upgrade the compatible phase-1 auto-cache file in place. The
                // old .meta is read only to authenticate identity/tokens; the
                // native payload is copied in 8 MiB chunks and atomically
                // replaced with the common footer before it can be indexed.
                model_fp legacy_fp;
                llama_tokens legacy_tokens;
                if (slot_meta_read(p, legacy_fp, legacy_tokens, auto_store_max_tokens()) &&
                        auto_legacy_fp_compatible(legacy_fp) &&
                        server_route_state_adopt_legacy(p, current_model, current_layout,
                            legacy_tokens, legacy_fp.fp_n_ctx, legacy_fp.fp_block, auto_store_max_bytes())) {
                    std::error_code legacy_ec;
                    std::filesystem::remove(slot_meta_sidecar_path(p), legacy_ec);
                        file = read_unified_snapshot(p);
                    if (file) {
                        SRV_INF("unified snapshot adopted legacy auto-cache file: %s\n", p.c_str());
                    }
                }
            }
            if (!file) {
                continue; // incompatible/corrupt or incomplete publication
            }
            auto_idx.indexed_files.insert(p);
            if (file->model != current_model) {
                SRV_WRN("unified snapshot ignored at %s: model identity mismatch\n", p.c_str());
                continue; // identity mismatch is a safe miss, not a fallback restore
            }
            if (!common_prompt_cache_layout_reusable(file->layout, current_layout)) {
                SRV_WRN("unified snapshot ignored at %s: KV/attention/RoPE layout mismatch\n", p.c_str());
                continue; // layout mismatch is a safe miss, not a fallback restore
            }
            if (file->tokens.empty()) {
                SRV_WRN("unified snapshot ignored at %s: empty token prefix\n", p.c_str());
                continue;
            }
            const uint32_t block = auto_index_block(file->index_block);
            const auto bhs = auto_block_hashes(file->tokens, (int) block, cur_fp.fp_model);
            auto_cache_entry e{ p, (uint32_t) file->tokens.size(), block, file->model, file->layout };
            for (uint64_t bh : bhs) {
                auto_index_insert_locked(block, bh, e);
            }
        }
        if (ec) {
            SRV_WRN("unified snapshot index scan failed at %s: %s\n",
                    params_base.slot_save_path.c_str(), ec.message().c_str());
        }
    }

    // One-time startup scan: builds the initial index. Invariant 1: only ever called from an
    // auto_cache_enabled() branch.
    void auto_index_scan() {
        std::lock_guard<std::mutex> lk(auto_idx.mtx);
        auto_idx.by_boundary.clear();
        auto_idx.indexed_files.clear();
        auto_idx.index_blocks.clear();
        auto_idx.last_refresh = std::chrono::steady_clock::now();
        auto_index_scan_locked();
    }

    // Cross-process refresh: make snapshots that OTHER processes created visible here WITHOUT a
    // restart. Cheap by design: throttled to at most once per AUTO_REFRESH_MIN_MS, and even then it
    // only does one stat of the dir mtime - a full re-scan happens ONLY when the dir actually changed
    // (a peer create/rename/delete bumps the dir mtime) or when `force` is set (a lookup miss, where
    // we are about to pay a cold prefill anyway so the scan is free in comparison). On a change we
    // also drop entries whose files a peer evicted. CALLER MUST HOLD auto_idx.mtx.
    void auto_index_refresh_locked(bool force) {
        const auto now = std::chrono::steady_clock::now();
        if (!force &&
            now - auto_idx.last_refresh < std::chrono::milliseconds(AUTO_REFRESH_MIN_MS)) {
            return; // throttled: avoid a stat storm during a burst of lookups
        }
        auto_idx.last_refresh = now;
        std::error_code ec;
        const auto dmt = std::filesystem::last_write_time(params_base.slot_save_path, ec);
        if (!ec && dmt == auto_idx.dir_mtime && !force) {
            return; // nothing changed on disk since the last scan
        }
        // a peer changed the dir (or forced): re-scan for NEW files, then reconcile deletions.
        auto_index_scan_locked();
        auto_index_drop_missing_locked();
    }

    // Longest-prefix lookup over the request tokens. The shared index may contain
    // snapshots written with different block sizes, so search every discovered
    // block namespace and then select the deepest verified prefix.
    std::optional<auto_cache_entry> auto_index_lookup(const llama_tokens & req) {
        if (!auto_cache_enabled()) {
            return std::nullopt; // off by default
        }
        std::lock_guard<std::mutex> lk(auto_idx.mtx);
        // Cross-process visibility: cheaply pick up snapshots a peer process created since our last
        // scan (throttled dir-mtime check). Then search; on a MISS, force a re-scan and search again
        // - the force is justified because a miss means we are about to cold-prefill, so the scan
        // cost is negligible against it, and a peer's snapshot written <1s ago (within the throttle
        // window) is still found on this first request rather than only the next one.
        auto_index_refresh_locked(/*force=*/false);
        for (int attempt = 0; attempt < 2; ++attempt) {
            std::optional<auto_cache_entry> best;
            std::unordered_set<std::string> visited_paths;
            size_t boundary_entries = 0;
            size_t best_prefix = 0;
            for (uint32_t block : auto_idx.index_blocks) {
                const auto bhs = auto_block_hashes(req, (int) block, cur_fp.fp_model);
                for (size_t k = bhs.size(); k-- > 0; ) {
                    const auto range = auto_idx.by_boundary.equal_range(auto_boundary_key(block, bhs[k]));
                    for (auto it = range.first; it != range.second; ++it) {
                        ++boundary_entries;
                        if (!visited_paths.insert(it->second.state_path).second) {
                            continue;
                        }
                        auto file = read_unified_snapshot(it->second.state_path, false);
                        if (!file || file->model != adaptive_model_identity->fingerprint() ||
                                !common_prompt_cache_layout_reusable(file->layout, common_prompt_cache_layout(ctx_tgt)) ||
                                auto_index_block(file->index_block) != block) {
                            continue;
                        }
                        // Managed MTP restore must end strictly before the
                        // request. Loading a state that already contains the
                        // complete request would require a recurrent rewind
                        // plan that is not represented by this target-only
                        // envelope. The pre-last prompt snapshot is the safe
                        // candidate; a shorter response snapshot remains
                        // eligible for a genuinely longer continuation.
                        if (ctx_dft && file->tokens.size() >= req.size()) {
                            continue;
                        }
                        size_t prefix = 0;
                        while (prefix < file->tokens.size() && prefix < req.size() &&
                                file->tokens[prefix] == req[prefix]) {
                            ++prefix;
                        }
                        if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_PART &&
                                prefix != file->tokens.size()) {
                            continue;
                        }
                        if (prefix > best_prefix) {
                            best_prefix = prefix;
                            best = it->second;
                        }
                    }
                }
            }
            if (best) {
                SRV_DBG("auto-index lookup visited %zu unique snapshot paths (attempt=%d)\n",
                        visited_paths.size(), attempt + 1);
                if (std::getenv("LLAMA_TEST_SNAPSHOT_LOOKUP_TRACE")) {
                    SRV_INF("auto-index lookup trace: boundary_entries=%zu unique_paths=%zu payload_validation=deferred_to_selected_restore\n",
                            boundary_entries, visited_paths.size());
                }
                return best;
            }
            SRV_DBG("auto-index lookup visited %zu unique snapshot paths (attempt=%d)\n",
                    visited_paths.size(), attempt + 1);
            if (std::getenv("LLAMA_TEST_SNAPSHOT_LOOKUP_TRACE")) {
                SRV_INF("auto-index lookup trace: boundary_entries=%zu unique_paths=%zu payload_validation=deferred_to_selected_restore\n",
                        boundary_entries, visited_paths.size());
            }
            if (attempt == 0) {
                auto_index_refresh_locked(/*force=*/true); // miss -> rescan once before giving up
            }
        }
        return std::nullopt;
    }

    // After an LRU eviction (which deletes files silently - ours OR a peer process's), drop index
    // boundaries pointing at files that no longer exist, and forget them in indexed_files so a future
    // re-create can be re-indexed. Cheap stat per unique path; keeps index <-> disk consistent (invariant 4).
    // A lookup that races an eviction and finds a now-deleted file simply fails the load -> prefill.
    // CALLER MUST HOLD auto_idx.mtx.
    void auto_index_drop_missing_locked() {
        std::unordered_set<std::string> checked_paths;
        std::unordered_set<std::string> missing_paths;
        for (auto it = auto_idx.by_boundary.begin(); it != auto_idx.by_boundary.end(); ) {
            const std::string & path = it->second.state_path;
            if (checked_paths.insert(path).second) {
                std::error_code ec;
                if (!std::filesystem::exists(path, ec) || ec) {
                    missing_paths.insert(path);
                }
            }
            if (missing_paths.count(path)) {
                it = auto_idx.by_boundary.erase(it);
            } else {
                ++it;
            }
        }
        for (auto it = auto_idx.indexed_files.begin(); it != auto_idx.indexed_files.end(); ) {
            std::error_code ec;
            if (!std::filesystem::exists(*it, ec) || ec) {
                it = auto_idx.indexed_files.erase(it);
            } else {
                ++it;
            }
        }
        auto_idx.index_blocks.clear();
        for (const auto & item : auto_idx.by_boundary) {
            auto_idx.index_blocks.insert(auto_index_block(item.second.index_block));
        }
    }

    // Restore a canonical unified snapshot INTO `slot`. The native payload is
    // always installed by the same empty-context streaming reader as route
    // handoff; the footer/token manifest was already inspected by the caller.
    // On any failure the destination is cleared and the caller cold-prefills.
    bool do_slot_restore(server_slot & slot, const server_route_state_file & file,
                         const llama_tokens & expected_tokens) {
        if (file.tokens != expected_tokens || file.tokens.empty() ||
                file.tokens.size() > (size_t) slot.n_ctx || file.state_bytes == 0) {
            slot.prompt_clear();
            return false;
        }
        server_route_state_lease lease(file.state_path);
        if (!lease.acquired()) {
            slot.prompt_clear();
            return false;
        }
        llama_tokens tokens(file.tokens.size());
        size_t token_count = 0;
        const size_t nread = llama_state_seq_load_file_streaming(
            ctx_tgt, file.state_path.c_str(), slot.id, tokens.data(), tokens.size(), &token_count,
            file.state_bytes, file.state_checksum);
        if (nread != file.state_bytes || token_count != file.tokens.size() || tokens != expected_tokens ||
                llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id) != file.position) {
            slot.prompt_clear();
            return false;
        }
        slot.prompt.tokens.clear();
        slot.prompt.tokens.insert(file.tokens);
        slot.just_restored = true;

        // Reconstruct a context checkpoint at the restored position so hybrid/recurrent (and SWA)
        // models - which cannot partially rewind - can reuse this state for the suffix; other
        // models do not need it.
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            const auto ckpt_pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
            const auto ckpt_pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);
            if (ckpt_pos_min >= 0 && params_base.n_ctx_checkpoints > 0) {
                slot.prompt.checkpoints.clear();
                create_checkpoint(slot, 0, ckpt_pos_min, ckpt_pos_max);
            }
        }

        // The restored state's running distribution is unknown; invalidate any stale capture so a
        // later SLOT_SAVE cannot persist a mismatched sidecar.
        slot.logits_last.clear();
        slot.logits_last_n_tokens = -1;

        // Load the regenerate logits sidecar (FULL only) so an exact-prompt regenerate can emit the
        // first token without re-decoding into the restored recurrent state.
        slot.restored_logits.clear();
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));
            const bool embedded = server_route_state_read_logits(file, (uint32_t) nv, slot.restored_logits);
            if (!embedded) {
                // Compatibility for an old sidecar that has not yet been
                // adopted; new saves never create this second state format.
                slot_logits_read(file.state_path, nv, (uint32_t) token_count, slot.restored_logits);
            }
            if (!slot.restored_logits.empty()) {
                SLT_INF(slot, "loaded logits sidecar (%d vocab, %zu tokens) - regenerate fast-path armed\n", nv, token_count);
            }
        }
        return true;
    }

    // AUTO-RESTORE wrapper: byte-verify the candidate's persisted tokens against the request prefix
    // (invariant 2), confirm the fingerprint (invariant 3), then restore. Returns the verified prefix length
    // actually restored, or 0 if nothing was restored (caller keeps the in-memory prefill path).
    // `req` is the full request token-ID array; `n_keep_mem` is the in-memory match to beat.
    int auto_restore_into_slot(server_slot & slot, const auto_cache_entry & cand,
                               const llama_tokens & req, int n_keep_mem) {
        const int64_t restore_started = ggml_time_us();
        const uint64_t workspace = adaptive_slot_saturating_add(
            48ULL * 1024 * 1024,
            adaptive_slot_saturating_mul((uint64_t) llama_n_ctx_seq(ctx_tgt), 512));
        adaptive_slot_cache_reservation reservation;
        if (workspace > adaptive_slot_ram_budget(params_base.cache_ram_mib) ||
                !reservation.acquire(prompt_cache.get(), (size_t) workspace)) {
            SLT_WRN(slot, "auto-restore refused by transient budget: workspace=%" PRIu64 "\n", workspace);
            return 0;
        }
        // Hold both shared stripes from metadata inspection through the native
        // streaming commit. A publisher can replace the pathname only after
        // this operation releases the reference, so checksum mismatch is not
        // the normal result of a legitimate concurrent write.
        server_route_state_lease reference(cand.state_path, server_route_state_lock_mode::reference);
        if (!reference.acquired()) {
            return 0;
        }
        auto file = read_unified_snapshot(cand.state_path, false);
        if (!file || file->model != adaptive_model_identity->fingerprint() ||
                !common_prompt_cache_layout_reusable(file->layout, common_prompt_cache_layout(ctx_tgt))) {
            return 0; // invariant 3/4: incompatible or incomplete publication
        }
        const llama_tokens & disk_toks = file->tokens;
        // byte-verify: longest common prefix of the persisted tokens and the request (invariant 2).
        const size_t lim = std::min(disk_toks.size(), req.size());
        size_t v = 0;
        while (v < lim && disk_toks[v] == req[v]) {
            ++v;
        }
        // Only WHOLE-block prefixes are valid reuse lengths (hash boundaries).
        const int B = (int) auto_index_block(file->index_block);
        int n_keep_disk;
        if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_PART) {
            // a FULL/recurrent/hybrid/SWA state cannot be PARTIALLY rewound -
            // do_slot_restore loads the ENTIRE L-token snapshot, and a later keep_first(n_past<L)
            // would issue a PARTIAL common_context_seq_rm that GGML_ABORTs the server (a FULL model's
            // llama_memory_seq_rm refuses a partial range). So we ONLY auto-restore a FULL snapshot
            // when the request diverges at or beyond the snapshot end (v == disk_toks.size(), i.e.
            // the whole snapshot is a verified prefix of the request). If the request diverges INSIDE
            // the snapshot, refuse and fall back to normal prefill - never restore a FULL snapshot we
            // would have to partially unwind. (No block-boundary clamp for FULL: only the exact whole
            // snapshot is a legal restore length here.)
            if (v != disk_toks.size()) {
                return 0;
            }
            n_keep_disk = (int) disk_toks.size();
        } else {
            // Attention (PART) models support per-token partial seq_rm, so a mid-snapshot divergence
            // is fine: claim the verified prefix clamped down to the last whole block boundary <= v.
            // An exact full-snapshot match keeps the whole snapshot length.
            if (v == disk_toks.size()) {
                n_keep_disk = (int) disk_toks.size();
            } else {
                n_keep_disk = (int) (v - (v % (size_t) B));
            }
        }
        if (n_keep_disk <= 0) {
            return 0;
        }
        // MARGIN gate (invariant 5): only pay a multi-GB load if disk strictly beats the
        // in-memory match by at least one block - never thrash a reload to save a few tokens.
        if (n_keep_disk < n_keep_mem + B) {
            return 0;
        }
        // Clear the slot's resident KV before loading the snapshot (mirror the restore-continue safe
        // fallback): seq removal + token/checkpoint clear so the restore writes into an empty seq.
        slot.prompt_clear();

        if (!do_slot_restore(slot, *file, disk_toks)) {
            // restore failed -> slot seq already cleared by do_slot_restore; caller reprefills (invariant 4).
            SLT_WRN(slot, "auto-restore: candidate %s failed native streaming validation; falling back to cold prefill\n",
                    cand.state_path.c_str());
            return 0;
        }
        // do_slot_restore loaded the snapshot. For FULL models n_keep_disk == snapshot length (gated
        // above), so the existing regenerate / suffix-reuse path takes over with no partial
        // rewind. For attention models the request may diverge inside the snapshot; keep_first(n_past)
        // + a PARTIAL seq_rm then reprefills the divergent tail (supported for PART). The verified
        // prefix is what we claim as reused.
        // [FORK] MTP draft carry: the restored state covers only the target context; the resident
        // MTP carry is stale (or unset). Flag the slot so the fork's next-decode bootstrap path
        // re-syncs the carry from the decoded suffix (same mechanism as a target-only RAM cache hit).
        // NOTE: must be set for PART models too - the production Qwen3.8-27B-RCO runs with MTP and
        // a PART seq-rm type, and an un-synced carry breaks the next speculative process.
        slot.bootstrap_pending = slot.can_speculate();
        slot.prompt_cache_source = "disk";
        slot.prompt_cache_reason = "unified_snapshot_restore";
        // Bump the snapshot's mtime so the LRU treats a reused-but-not-rewritten base snapshot as
        // recently-used (true LRU, not least-recently-written) - critical for the fan-out case where
        // many requests restore one hot base prefix. Best-effort; never errors the restore (invariant 4).
        auto_touch_unit(cand.state_path);
        SLT_INF(slot, "auto-restore: reused %d tokens from disk (in-memory match was %d), bytes=%zu restore_ms=%.3f workspace=%" PRIu64 ", file=%s\n",
                n_keep_disk, n_keep_mem, file->file_bytes, (ggml_time_us() - restore_started) / 1000.0,
                workspace, cand.state_path.c_str());
        return n_keep_disk;
    }

    void auto_capture_prompt_logits(server_slot & slot, int logits_index) {
        if (!auto_cache_enabled() || ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                logits_index < 0) {
            return;
        }
        const float * logits = llama_get_logits_ith(slot.ctx_tgt, logits_index);
        if (!logits) {
            slot.logits_last.clear();
            slot.logits_last_n_tokens = -1;
            return;
        }
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));
        slot.logits_last.assign(logits, logits + n_vocab);
        slot.logits_last_n_tokens = (int32_t) slot.prompt.tokens.size();
    }

    // AUTO-SAVE: persist a slot's KV before it is discarded, keyed by its token-prefix block hash.
    // Route handoff and auto-cache use the same canonical native/footer snapshot. The optional
    // logits sidecar is an auxiliary regenerate fast-path, never an index key or a second state
    // format. Invariant 1: first statement is the gate; invariant 5: only called on slot
    // release/reassign, never during generation.
    void auto_save_slot_if_useful(server_slot & slot, const char * reason = "discard") {
        if (!auto_cache_enabled()) {
            return; // off by default
        }
        if (!adaptive_model_identity) {
            SLT_DBG(slot, "%s", "auto-save skipped: unified snapshot identity is unavailable\n");
            return;
        }
        const int64_t save_started = ggml_time_us();
        const uint64_t workspace = adaptive_slot_saturating_add(
            16ULL * 1024 * 1024,
            adaptive_slot_saturating_mul((uint64_t) llama_n_ctx_seq(ctx_tgt), 16));
        adaptive_slot_cache_reservation reservation;
        if (workspace > adaptive_slot_ram_budget(params_base.cache_ram_mib) ||
                !reservation.acquire(prompt_cache.get(), (size_t) workspace)) {
            SLT_WRN(slot, "auto-save refused by transient budget: workspace=%" PRIu64 "\n", workspace);
            return;
        }
        // exclusions reuse the existing guards. NOTE: an idle slot has already been reset(), so
        // `slot.task` is null here - the just-finished task survives as `slot.task_prev`. Use it for
        // the generative check (COMPLETION/INFILL only). Gate on the PER-REQUEST `has_media()` (not
        // the server-wide has_mtmd/mctx) so an --mmproj server still persists its text-only turns;
        // a turn carrying an image (has_media()==true) is skipped - exactly correct, since token-ids
        // alone cannot identify image content.
        const auto & wtask = slot.task ? slot.task : slot.task_prev;
        if (!wtask || !wtask->need_sampling() || !wtask->params.cache_prompt || slot.prompt.tokens.has_media()) {
            return;
        }
        // The fingerprint captures the GLOBAL LoRA set; refuse to persist a snapshot taken under a
        // per-request adapter override that differs from it (invariant 3). (Conservative: a future version
        // could fold the slot's adapters into the snapshot fingerprint instead.)
        if (!are_lora_equal(slot.lora, params_base.lora_adapters)) {
            return;
        }
        // get_text_tokens() (not get_tokens()): media-safe accessor that never trips the
        // get_tokens() GGML_ASSERT(!has_mtmd) under mmproj. For this no-media prompt (has_media()
        // false, guarded above) it equals the full token-id prefix, so the persisted token stream
        // and the block-hash key are byte-identical to what a text-only server would write.
        const llama_tokens toks = slot.prompt.tokens.get_text_tokens();
        const uint32_t block = auto_index_block(params_base.slot_save_block);
        if (toks.size() < block) {
            return; // < 1 block: not worth a multi-GB write
        }
        const auto bhs = auto_block_hashes(toks, (int) block, cur_fp.fp_model);
        if (bhs.empty()) {
            return;
        }
        uint64_t full_hash = bhs.back();
        for (size_t i = toks.size() - toks.size() % block; i < toks.size(); ++i) {
            full_hash = auto_hash_mix(full_hash, toks[i]);
        }
        const std::string fname = auto_state_filename(full_hash, toks.size());
        const std::string current_model = adaptive_model_identity->fingerprint();
        const std::string current_layout = common_prompt_cache_layout(ctx_tgt);
        std::error_code exists_ec;
        if (std::filesystem::exists(fname, exists_ec)) {
            if (auto existing = read_unified_snapshot(fname); existing &&
                    existing->model == current_model &&
                    common_prompt_cache_layout_reusable(existing->layout, current_layout) &&
                    existing->tokens == toks) {
                return; // deterministic same-prefix publication already exists
            }
        }

        // Serialize budget decisions with every other publisher. The common
        // writer invokes the preflight before exposing the pathname and
        // commits victims while the store lock is still held; no reader pin is
        // needed during that atomic write window.
        {
            server_route_state_store_lock store(params_base.slot_save_path,
                    server_route_state_store_lock_mode::exclusive);
            if (!store.acquired()) {
                SLT_WRN(slot, "auto-save skipped: unified snapshot store is unavailable (%s)\n",
                        params_base.slot_save_path.c_str());
                return;
            }
            auto limit_plan = std::make_shared<unified_snapshot_limit_plan>();
            limit_plan->dir = params_base.slot_save_path;
            limit_plan->just_written = fname;
            limit_plan->max_count = params_base.slot_save_max_count;
            limit_plan->max_bytes = params_base.slot_save_max_bytes;
            const std::vector<float> * logits =
                slot.logits_last_n_tokens == (int32_t) toks.size() && !slot.logits_last.empty()
                    ? &slot.logits_last : nullptr;
            try {
                const size_t nwrite = server_route_state_save(ctx_tgt, slot.id, toks, *adaptive_model_identity,
                        fname, auto_store_max_bytes(), block, logits,
                        [limit_plan](uint64_t incoming) { return limit_plan->prepare(incoming); },
                        [limit_plan]() { return limit_plan->commit(); }, &store, nullptr);
                // Legacy auxiliaries cannot be combined with a new embedded
                // payload. Remove them only after the canonical file committed,
                // while the publication lock still excludes readers.
                std::error_code ec;
                std::filesystem::remove(slot_logits_sidecar_path(fname), ec);
                std::filesystem::remove(slot_meta_sidecar_path(fname), ec);

                {
                    std::lock_guard<std::mutex> lk(auto_idx.mtx);
                    auto_cache_entry e{ fname, (uint32_t) toks.size(), block, current_model, current_layout };
                    if (!auto_idx.indexed_files.count(fname)) {
                        for (uint64_t bh : bhs) {
                            auto_index_insert_locked(block, bh, e);
                        }
                    }
                    auto_idx.indexed_files.insert(fname); // remember our own write
                }
                SLT_INF(slot, "auto-save: persisted unified snapshot reason=%s tokens=%zu bytes=%zu save_ms=%.3f workspace=%" PRIu64 " path=%s\n",
                        reason, toks.size(), nwrite, (ggml_time_us() - save_started) / 1000.0,
                        workspace, fname.c_str());
            } catch (const std::exception & error) {
                SLT_WRN(slot, "auto-save: unified snapshot failed safely: %s\n", error.what());
            }
        }
        // Reconcile index with what the LRU kept (ours or a peer's) AND adopt the post-write dir
        // mtime as our scan baseline - both under ONE lock. Re-baselining here means OUR OWN
        // save+evict does not make the next lookup think a PEER changed the dir (which would force a
        // redundant full re-scan); a real peer write afterwards bumps the mtime again -> still
        // detected. CALLER holds no lock here.
        {
            std::lock_guard<std::mutex> lk(auto_idx.mtx);
            auto_index_drop_missing_locked();
            std::error_code mec;
            const auto dmt = std::filesystem::last_write_time(params_base.slot_save_path, mec);
            if (!mec) {
                auto_idx.dir_mtime = dmt;
            }
        }
    }

    // Capture a prompt branch point before generation appends its response. For
    // managed MTP the pre-last hook below stores N-1, so the final prompt token
    // is a real suffix decode that can bootstrap draft/carry without rewinding
    // a recurrent state. FULL/no-MTP keeps the complete prompt and may embed
    // last-prompt logits in the same canonical file.
    void auto_save_prompt_prefix_checkpoint(server_slot & slot) {
        if (auto_cache_enabled() && ctx_dft && slot.can_speculate()) {
            auto_save_slot_if_useful(slot, "prompt-prefix");
        }
    }

    void auto_save_prompt_checkpoint(server_slot & slot) {
        if (auto_cache_enabled() && (!ctx_dft || !slot.can_speculate())) {
            auto_save_slot_if_useful(slot, "prompt");
        }
    }

    // Persist a just-finished slot's KV to the disk auto cache at task completion, so a cold
    // process can reuse the snapshot from the very first request on (not only when the slot is
    // evicted). Off by default; all correctness gates live in auto_save_slot_if_useful.
    void auto_save_on_completion(server_slot & slot) {
        if (auto_cache_enabled()) {
            auto_save_slot_if_useful(slot);
        }
    }

    void handle_sleeping_state(bool new_state) {
        GGML_ASSERT(sleeping != new_state);
        if (new_state) {
            gpu_power.on_sleeping(true);
            if (callback_state) {
                callback_state(SERVER_STATE_SLEEPING, {});
                // note: for sleeping == false, event is emitted by load_model()
            }
            SRV_INF("%s", "server is entering sleeping state\n");
            destroy();
        } else {
            SRV_INF("%s", "server is exiting sleeping state\n");
            if (!load_model(params_base)) {
                GGML_ABORT("failed to reload model after sleeping");
            }
            if (!gpu_power.init({
                    params_base.gpu_power_prefill,
                    params_base.gpu_power_decode,
                    params_base.gpu_mem_clock_decode,
                    params_base.gpu_mem_clock_prefill,
                    params_base.gpu_power_device,
                })) {
                GGML_ABORT("failed to reinitialize GPU power governor after sleeping");
            }
        }
        sleeping = new_state;
    }

    struct load_progress_data {
        server_context_impl * ctx;
        std::string stage;
        std::vector<std::string> stages;
        int64_t t_last_load_progress_ms = 0;
        load_progress_data(server_context_impl * ctx, const std::string & stage) : ctx(ctx), stage(stage) {}
    };
    static bool load_progress_callback(float progress, void * user_data) {
        auto * d = static_cast<load_progress_data *>(user_data);
        GGML_ASSERT(d);
        // always emit the first and final sample; throttle the rest to one per 200ms
        {
            auto & t_last = d->t_last_load_progress_ms;
            const int64_t t_now = ggml_time_ms();
            const bool first = t_last == 0;
            const bool done  = progress >= 1.0f;
            const bool throttled = !first && !done && (t_now - t_last) < 200;
            if (throttled) {
                return true;
            }
            t_last = t_now;
        }
        if (d->ctx->callback_state) {
            d->ctx->callback_state(SERVER_STATE_LOADING, {
                {"stages", d->stages},
                {"current", d->stage},
                {"value", progress},
            });
        }
        return true;
    }

    // load the model and initialize llama_context
    // this may also be called to resume from sleeping state
    bool load_model(common_params & params) {
        if (const std::string error = common_context_adaptive_normalize(params); !error.empty()) {
            SRV_ERR("invalid adaptive context configuration: %s\n", error.c_str());
            return false;
        }
        if (const std::string error = common_context_prepare_devices(params); !error.empty()) {
            SRV_ERR("invalid adaptive context device selection: %s\n", error.c_str());
            return false;
        }

        load_progress_data load_progress_text  (this, "text_model");
        load_progress_data load_progress_mmproj(this, "mmproj_model");
        load_progress_data load_progress_spec  (this, "spec_model");

        const bool is_resume = sleeping;
        const bool adaptive = common_context_is_adaptive(params);
        const int32_t requested_long_ctx = adaptive_long_ctx > 0 ? adaptive_long_ctx : params.n_ctx;

        std::optional<server_model_identity> prepared_identity;
        if (adaptive || params.slot_save_auto || std::getenv("LLAMA_SERVER_ROUTER_STATE")) {
            adaptive_model_identity.reset();
            try {
                prepared_identity.emplace(server_model_identity::prepare(params.model.path, params.kv_overrides));
            } catch (const std::exception & error) {
                SRV_ERR("failed to prepare adaptive model identity: %s\n", error.what());
                return false;
            }
        } else {
            adaptive_model_identity.reset();
        }

        params_base = params;
        if (adaptive) {
            adaptive_draft_n_medium = params.speculative.draft.n_max;
            adaptive_draft_n_short  = params.spec_draft_n_max_short;
            adaptive_batch_normal   = params.n_batch;
            adaptive_ubatch_normal  = params.n_ubatch;
            adaptive_cache_type_k_normal = params.cache_type_k;
            adaptive_cache_type_v_normal = params.cache_type_v;
            adaptive_kvarn_normal   = params.kvarn;
            adaptive_cache_kvarn_bits_k_normal = params.cache_kvarn_bits_k;
            adaptive_cache_kvarn_bits_v_normal = params.cache_kvarn_bits_v;

            adaptive_batch_xlong    = params.batch_size_xlong > 0 ? params.batch_size_xlong : 64;
            adaptive_ubatch_xlong   = params.ubatch_size_xlong > 0 ? params.ubatch_size_xlong : 64;

            adaptive_batch_xxlong   = params.batch_size_xxlong > 0 ? params.batch_size_xxlong : 64;
            adaptive_ubatch_xxlong  = params.ubatch_size_xxlong > 0 ? params.ubatch_size_xxlong : 64;
            adaptive_cache_type_k_xxlong = params.cache_type_k_xxlong;
            adaptive_cache_type_v_xxlong = params.cache_type_v_xxlong;
            adaptive_kvarn_xxlong   = params.kvarn_xxlong;
            adaptive_cache_kvarn_bits_k_xxlong = params.cache_kvarn_bits_k_xxlong;
            adaptive_cache_kvarn_bits_v_xxlong = params.cache_kvarn_bits_v_xxlong;

            if (params.ctx_size_xxlong > 0 && adaptive_cache_kvarn_bits_k_xxlong == 0) {
                adaptive_cache_kvarn_bits_k_xxlong = 4;
                adaptive_cache_kvarn_bits_v_xxlong = 4;
                adaptive_cache_type_k_xxlong = GGML_TYPE_Q4_0;
                adaptive_cache_type_v_xxlong = GGML_TYPE_Q4_0;
                adaptive_kvarn_xxlong.type = llama_kvarn_type_from_name("kvarn_k4v4_g128");
            }

            if (params.ctx_size_mtp_short > 0) {
                apply_profile_params(COMMON_CONTEXT_PROFILE_MTP_SHORT);
                active_context_profile = COMMON_CONTEXT_PROFILE_MTP_SHORT;
            } else {
                apply_profile_params(COMMON_CONTEXT_PROFILE_MTP);
                active_context_profile = COMMON_CONTEXT_PROFILE_MTP;
            }
        } else {
            active_context_profile = COMMON_CONTEXT_PROFILE_LONG;
            const auto output_limits = server_output_limits(params_base);
            params_base.n_outputs_max = output_limits.total;
            params_base.n_outputs_max_per_seq = output_limits.per_seq;
        }

        const bool has_mmproj = !params.mmproj.path.empty();
        const bool has_draft = params.speculative.has_dft();
        const bool spec_mtp = std::find(params_base.speculative.types.begin(),
                                        params_base.speculative.types.end(),
                                        COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params_base.speculative.types.end();
        const bool has_spec = has_draft || spec_mtp;

        if (callback_state) {
            std::vector<std::string> stages = {"text_model"};
            if (has_spec) {
                stages.push_back("spec_model");
            }
            if (has_mmproj) {
                stages.push_back("mmproj_model");
            }
            load_progress_text.stages   = stages;
            load_progress_mmproj.stages = stages;
            load_progress_spec.stages   = stages;

            // trigger 0% progress
            load_progress_callback(0.0f, &load_progress_text);
        }


        SRV_INF("loading model '%s'\n", params.model.get_name().c_str());
        SRV_TRC("local path '%s'\n", params.model.path.c_str());

        std::string & mmproj_path = params_base.mmproj.path;
        mtmd_context_params mparams = mtmd_context_params_default();
        if (has_mmproj) {
            mparams.use_gpu          = params_base.mmproj_use_gpu;
            mparams.device           = params_base.mmproj_device;
            mparams.print_timings    = false;
            mparams.n_threads        = params_base.cpuparams.n_threads;
            mparams.flash_attn_type  = params_base.flash_attn_type;
            mparams.warmup           = params_base.warmup;
            mparams.image_min_tokens = params_base.image_min_tokens;
            mparams.image_max_tokens = params_base.image_max_tokens;
            mparams.batch_max_tokens = params_base.mtmd_batch_max_tokens;
            mparams.media_marker     = get_media_marker();
            // progress callback
            mparams.progress_callback           = load_progress_callback;
            mparams.progress_callback_user_data = &load_progress_mmproj;
        }

        // optionally get the memory usage of mmproj
        if (has_mmproj && params_base.fit_params) {
            int64_t t_start = ggml_time_us();
            auto mmproj_mem = mtmd_get_memory_usage(mmproj_path.c_str(), mparams);
            int64_t t_elapsed = ggml_time_us() - t_start;
            if (!mmproj_mem.empty()) {
                size_t total = 0;
                for (auto & [dev, size] : mmproj_mem) {
                    total += size;
                }
                SRV_TRC("[mtmd] estimated worst-case memory usage of mmproj is %.2f MiB (took %.2f ms)\n", total / (1024.0 * 1024.0), t_elapsed / 1000.0);
                GGML_ASSERT(!params_base.fit_params_target.empty());
                for (auto & [dev, size] : mmproj_mem) {
                    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                        if (ggml_backend_dev_get(i) == dev) {
                            if (i < params_base.fit_params_target.size()) {
                                SRV_DBG("[mtmd] adding %.2f MiB to fit_params_target for device %s\n", size / (1024.0 * 1024.0), ggml_backend_dev_name(dev));
                                params_base.fit_params_target[i] += size;
                            }
                            break;
                        }
                    }
                }
            } else {
                SRV_ERR("%s", "[mtmd] failed to get memory usage of mmproj\n");
            }
        }

        // note: the draft / MTP context is fitted together with the target model, see common_fit_extra_model

        // attach a progress callback
        {
            params_base.load_progress_callback = load_progress_callback;
            params_base.load_progress_callback_user_data = &load_progress_text;
        }

        llama_init = common_init_from_params(params_base);

        model_tgt = llama_init->model();
        ctx_tgt   = llama_init->context();

        if (model_tgt == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", params_base.model.path.c_str());
            return false;
        }

        if (ctx_tgt == nullptr) {
            SRV_ERR("failed to create_context with model '%s'\n", params_base.model.path.c_str());
            return false;
        }

        if (prepared_identity) {
            try {
                prepared_identity->verify_sources();
            } catch (const std::exception & error) {
                SRV_ERR("adaptive model identity changed during load: %s\n", error.what());
                destroy();
                return false;
            }
            adaptive_model_identity = std::move(prepared_identity);
        }

        vocab = llama_model_get_vocab(model_tgt);

        n_ctx = llama_n_ctx(ctx_tgt);
        const bool has_spec_mtp = std::find(params_base.speculative.types.begin(), params_base.speculative.types.end(),
                COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params_base.speculative.types.end();
        const int32_t n_ctx_train = llama_model_n_ctx_train(model_tgt);
        const int32_t requested_long_ctx_effective = requested_long_ctx > 0 ? requested_long_ctx : n_ctx_train;
        // The model's post-load training context is the only valid long ceiling.
        // Reject a larger declaration instead of silently serving a smaller one.
        adaptive_long_ctx = common_context_is_adaptive(params_base)
            ? std::min(requested_long_ctx_effective, n_ctx_train)
            : n_ctx;
        params_base.ctx_size_long = adaptive_long_ctx;
        if (common_context_is_adaptive(params_base) && requested_long_ctx_effective > n_ctx_train) {
            SRV_ERR("adaptive context long size (%d) exceeds model training context (%d)\n",
                    requested_long_ctx_effective, n_ctx_train);
            destroy();
            return false;
        }
        if (common_context_is_adaptive(params_base) && adaptive_long_ctx <= 0) {
            SRV_ERR("%s", "adaptive context requires a positive long context size\n");
            return false;
        }
        if (const std::string error = common_context_adaptive_error(params_base, adaptive_long_ctx); !error.empty()) {
            SRV_ERR("invalid effective adaptive context configuration: %s\n", error.c_str());
            return false;
        }
        // MTP's draft context uses the same dynamic paged block geometry as the
        // target.  The shared admission budget therefore covers both caches;
        // other speculative backends remain on the conservative path.
        elastic_paged_context = params_base.kv_paged && params_base.kv_paged_dynamic && params_base.n_parallel > 1 &&
            (!has_spec || has_spec_mtp);
        paged_admission_blocks = elastic_paged_context ? params_base.n_gpu_blocks_admission : 0;

        add_bos_token = llama_vocab_get_add_bos(vocab);

        if (has_spec) {
            // spec_mtp doesn't use load a model internally, so we report 0.0 and 1.0 manually
            load_progress_callback(0.0f, &load_progress_spec);
            load_progress_spec.t_last_load_progress_ms = 0;  // reset so internal cbs aren't delayed

            {
                common_params params_dft = common_base_params_to_speculative(params_base);

                // progress callback
                params_dft.load_progress_callback           = load_progress_callback;
                params_dft.load_progress_callback_user_data = &load_progress_spec;

                spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
                model_dft = spec_init->model();
                ctx_dft   = spec_init->context();

                if (has_draft && model_dft == nullptr) {
                    SRV_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
                    return false;
                }

                if (ctx_dft == nullptr) {
                    SRV_ERR("%s", "failed to create MTP context\n");
                    return false;
                }

                params_base.speculative.draft.ctx_tgt = ctx_tgt;
                params_base.speculative.draft.ctx_dft = ctx_dft;
            }

            load_progress_callback(1.0f, &load_progress_spec);
        }

        if (has_mmproj) {
            if (callback_state) {
                callback_state(SERVER_STATE_LOADING, {{"stage", "mmproj_model"}});
            }

            if (!is_resume) {
                mtmd_helper_log_set(common_log_default_callback, nullptr);
            }

            mctx = mtmd_init_from_file(mmproj_path.c_str(), model_tgt, mparams);
            if (mctx == nullptr) {
                SRV_ERR("failed to load multimodal model, '%s'\n", mmproj_path.c_str());
                return false;
            }
            SRV_INF("loaded multimodal model, '%s'\n", mmproj_path.c_str());

            init_opt.video_params.fps_target = params_base.video_fps;
            init_opt.video_params.timestamp_interval_ms = params_base.video_timestamp_interval_ms;
            init_opt.video_params.ffmpeg_bin_dir = params_base.video_ffmpeg_bin_dir.empty()
                                ? nullptr : params_base.video_ffmpeg_bin_dir.c_str();

            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by multimodal, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by multimodal, it will be disabled");
            }
        }

        if (!llama_memory_can_shift(llama_get_memory(ctx_tgt))) {
            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by this context, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by this context, it will be disabled");
            }
        }

        if (llama_model_n_swa(model_tgt) == 0) {
            if (params_base.swa_full) {
                params_base.swa_full = false;
                SRV_WRN("%s\n", "swa_full is not supported by this model, it will be disabled");
            }
        }

        n_swa = params_base.swa_full ? 0 : llama_model_n_swa(model_tgt);

        // Necessary similarity of prompt for slot selection
        slot_prompt_similarity = params_base.slot_prompt_similarity;

        int n_ctx_slot_value = llama_n_ctx_seq(ctx_tgt);
        if (elastic_paged_context) {
            n_ctx_slot_value = n_ctx;
        }
        {
            // Keep the fork's elastic paged slot size, while applying upstream's
            // per-slot cap diagnostics and limit when requested.
            const int n_ctx_seq = llama_n_ctx_seq(ctx_tgt);

            if (params_base.kv_unified_per_slot > 0) {
                if (n_ctx_seq > params_base.kv_unified_per_slot) {
                    SRV_INF("capping per-slot context (%d) to --kv-unified-per-slot (%d)\n",
                            n_ctx_seq, params_base.kv_unified_per_slot);
                } else if (params_base.kv_unified_per_slot > n_ctx_seq) {
                    // cap is above the per-slot pool capacity, so it can never bind
                    SRV_WRN(
                        "--kv-unified-per-slot (%d) exceeds the per-slot pool capacity (%d) - cap has no effect, "
                        "slots are limited to %d (raise the KV pool with -c, or unset -c to size it to "
                        "n_parallel * kv_unified_per_slot)\n",
                        params_base.kv_unified_per_slot, n_ctx_seq, n_ctx_seq);
                }
            }

            const int n_ctx_capped = params_base.kv_unified_per_slot > 0 ?
                std::min(n_ctx_seq, params_base.kv_unified_per_slot) : n_ctx_seq;
            if (elastic_paged_context) {
                if (params_base.kv_unified_per_slot > 0) {
                    n_ctx_slot_value = std::min(n_ctx_slot_value, params_base.kv_unified_per_slot);
                }
            } else {
                n_ctx_slot_value = std::min(n_ctx_slot_value, n_ctx_capped);
            }

            if (n_ctx_slot_value > n_ctx_train) {
                SRV_WRN("the slot context (%d) exceeds the training context of the model (%d) - capping\n",
                        n_ctx_slot_value, n_ctx_train);
            }
        }
        if (n_ctx_slot_value > n_ctx_train) {
            n_ctx_slot_value = n_ctx_train;
        }

        slots.clear();

        ctx_tgt_seq_rm_type = common_context_can_seq_rm(ctx_tgt);
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            SRV_WRN("%s", "speculative decoding not supported by this context\n");
        }

        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            SRV_TRC("%s", "speculative decoding will use checkpoints\n");
        }

        // setup slots
        SRV_INF("initializing, n_slots = %d, n_ctx_slot = %d, kv_unified = '%s', elastic_paged_context = %s\n",
                params_base.n_parallel, n_ctx_slot_value, params_base.kv_unified ? "true" : "false", elastic_paged_context ? "true" : "false");

        // initialize slots
        for (int i = 0; i < params_base.n_parallel; i++) {
            slots.emplace_back();
        }

        // try speculative decoding
        if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            try {
                spec.reset(common_speculative_init(params_base.speculative, params_base.n_parallel));
            } catch (const std::exception & e) {
                SRV_ERR("failed to initialize speculative decoding context: %s\n", e.what());
                if (params_base.speculative.has_synth()) {
                    return false;
                }
            }
        }

        if (ctx_dft) {
            ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft);
        }

        if (spec) {
            SRV_TRC("%s", "speculative decoding context initialized\n");
        } else {
            spec_init.reset();
            ctx_dft   = nullptr;
            model_dft = nullptr;
        }

        if (!spec && params_base.speculative.has_synth()) {
            SRV_ERR("%s", "synthetic acceptance requires an initialized speculative decoding context\n");
            return false;
        }

        for (int i = 0; i < params_base.n_parallel; i++) {
            server_slot & slot = slots[i];

            slot.id      = i;
            slot.ctx_tgt = ctx_tgt;
            slot.ctx_dft = ctx_dft;
            slot.mem.init(ctx_tgt, ctx_dft);
            slot.spec    = spec.get();
            slot.n_ctx   = n_ctx_slot_value;
            slot.clear_context_on_release = elastic_paged_context;

            slot.mctx                   = mctx;
            slot.prompt.tokens.has_mtmd = mctx != nullptr;

            SLT_TRC(slot, "new slot, n_ctx = %d\n", slot.n_ctx);

            slot.callback_on_release = [this](int id_slot) {
                queue_tasks.pop_deferred_task(id_slot);
            };

            slot.callback_on_reset = [this](const server_slot & slot) {
                // flush the generated token stats before reset()
                if (slot.stats.n_gen > 0) {
                    metrics_on_prediction(slot);
                }
            };

            slot.reset();
        }

        {
            const char * LLAMA_TRACE = getenv("LLAMA_TRACE");
            trace = LLAMA_TRACE ? atoi(LLAMA_TRACE) : 0;

            if (trace) {
                SRV_WRN("LLAMA_TRACE = %d\n", trace);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_DEBUG = getenv("LLAMA_SERVER_SLOTS_DEBUG");
            slots_debug = LLAMA_SERVER_SLOTS_DEBUG ? atoi(LLAMA_SERVER_SLOTS_DEBUG) : 0;

            if (slots_debug) {
                SRV_WRN("LLAMA_SERVER_SLOTS_DEBUG = %d\n", slots_debug);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_N_DIFF = getenv("LLAMA_SERVER_SLOTS_N_DIFF");
            slots_n_diff = LLAMA_SERVER_SLOTS_N_DIFF ? atoi(LLAMA_SERVER_SLOTS_N_DIFF) : 0;

            if (slots_n_diff) {
                SRV_WRN("LLAMA_SERVER_SLOTS_N_DIFF = %d\n", slots_n_diff);
            }
        }

        // the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const int32_t n_batch = std::max({params_base.n_batch, (int32_t) llama_n_batch(ctx_tgt), params_base.n_parallel});
            const int32_t n_embd  = llama_model_n_embd_inp(model_tgt);
            batch.init(n_batch, n_embd);
        }

        if (params_base.cache_ram_mib != 0) {
            if (params_base.cache_ram_mib < 0) {
                SRV_TRC("prompt cache is enabled, size limit: %s\n", "no limit");
            } else {
                SRV_TRC("prompt cache is enabled, size limit: %d MiB\n", params_base.cache_ram_mib);
            }
            SRV_TRC("%s", "use `--cache-ram 0` to disable the prompt cache\n");

            prompt_cache = std::make_unique<server_prompt_cache>(params_base.cache_ram_mib,
                common_context_is_adaptive(params_base) && adaptive_max_ctx() > 0 ? adaptive_max_ctx() : n_ctx);
        } else {
            SRV_TRC("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
        }
        SRV_TRC("%s", "for more info see https://github.com/ggml-org/llama.cpp/pull/16391\n");

        if (params_base.n_ctx_checkpoints > 0) {
            SRV_TRC("context checkpoints enabled, max = %d, min spacing = %d\n",
                    params_base.n_ctx_checkpoints, params_base.checkpoint_min_step);
        } else {
            SRV_TRC("%s", "context checkpoints disabled\n");
        }

        if (!params_base.model_alias.empty()) {
            // backward compat: use first alias as model name
            model_name = *params_base.model_alias.begin();
        } else if (!params_base.model.get_name().empty()) {
            model_name = params_base.model.get_name();
        } else {
            // fallback: derive model name from file name
            auto model_path = std::filesystem::path(params_base.model.path);
            model_name = model_path.filename().string();
        }

        model_aliases = params_base.model_alias;
        model_tags    = params_base.model_tags;

        // Propagate public defaults back to the HTTP layer, but keep the
        // adaptive ceiling visible there while params_base describes the
        // currently allocated profile.
        params = params_base;
        if (common_context_is_adaptive(params_base)) {
            params.n_ctx = adaptive_max_ctx();
            params.speculative.draft.ctx_tgt = nullptr;
            params.speculative.draft.ctx_dft = nullptr;
        }

        // AUTO disk prompt/KV cache (invariant 1): compute the model fingerprint and build the
        // longest-prefix index ONCE, header-only - but ONLY when the feature is enabled. When OFF
        // this is a single boolean test and nothing else (no fingerprint, no scan, no allocation).
        if (params_base.slot_save_auto && !auto_cache_enabled()) {
            SRV_WRN("%s", "auto disk prompt cache disabled: adapter/control-vector content identity is unavailable\n");
        }
        if (auto_cache_enabled()) {
            cur_fp = auto_compute_fingerprint();
            auto_index_scan();
            SRV_INF("auto disk prompt cache enabled: indexed %zu prefix boundaries from %s (block=%d)\n",
                    auto_idx.by_boundary.size(), params_base.slot_save_path.c_str(), params_base.slot_save_block);
        }

        if (!is_resume) {
            const bool ok = init();
            if (ok) {
                publish_adaptive_status();
            }
            return ok;
        }

        publish_adaptive_status();
        if (callback_state) {
            callback_state(SERVER_STATE_READY, {});
        }

        return true;
    }

    // unlike load_model(), this is only called once during initialization
    bool init() {
        GGML_ASSERT(ctx_tgt   != nullptr);
        GGML_ASSERT(model_tgt != nullptr);

        GGML_ASSERT(!sleeping);

        if (!gpu_power.init({
                params_base.gpu_power_prefill,
                params_base.gpu_power_decode,
                params_base.gpu_mem_clock_decode,
                params_base.gpu_mem_clock_prefill,
                params_base.gpu_power_device,
            })) {
            return false;
        }

        // wiring up server queues
        queue_tasks.on_new_task([this](server_task && task, bool is_yielding) {
            return process_single_task(std::move(task), is_yielding);
        });
        queue_tasks.on_update_slots([this]() {
            update_slots();
        });
        queue_tasks.on_sleeping_state([this](bool sleeping) {
            handle_sleeping_state(sleeping);
        });

        metrics.init();

        if (params_base.cache_idle_slots && prompt_cache) {
            if (params_base.cache_ram_mib == 0) {
                SRV_WRN("%s", "--cache-idle-slots requires --cache-ram, disabling\n");
                params_base.cache_idle_slots = false;
            } else {
                if (params_base.kv_unified) {
                    SRV_TRC("%s", "idle slots will be saved to prompt cache and cleared upon starting a new task\n");
                } else {
                    // without a unified KV cache, clearing a slot frees no reusable room, so we only
                    // publish a RAM-cache copy of idle slots (their KV stays in VRAM) [TAG_IDLE_SLOT_CLEAR]
                    SRV_TRC("%s", "idle slots will be saved to prompt cache upon starting a new task\n");
                }
                SRV_DBG("%s", "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__\n");
            }
        }

        {
            const std::string & cfg = params_base.ui_config_json;
            if (!cfg.empty()) {
                try {
                    json json_settings = json::parse(cfg);
                    json_ui_settings = json_settings;
                } catch (const std::exception & e) {
                    SRV_ERR("%s: failed to parse UI config: %s\n", __func__, e.what());
                    return false;
                }
            }
        }

        // populate chat template params
        {
            common_chat_templates_ptr chat_templates;
            bool enable_thinking = false;

            try {
                chat_templates = common_chat_templates_init(model_tgt, params_base.chat_template);

                SRV_TRC("%s: chat template, example_format: '%s'\n", __func__,
                    common_chat_format_example(chat_templates.get(), params_base.use_jinja, params_base.default_template_kwargs).c_str());

                // thinking is enabled if:
                // 1. It's not explicitly disabled via --reasoning off
                // 2. The chat template supports it
                const bool template_supports_thinking = params_base.use_jinja && common_chat_templates_support_enable_thinking(chat_templates.get());
                enable_thinking = params_base.enable_reasoning != 0 && template_supports_thinking;
                SRV_TRC("%s: chat template, thinking = %d\n", __func__, enable_thinking);
            } catch (const std::exception & e) {
                SRV_ERR("%s: chat template parsing error: %s\n", __func__, e.what());
                SRV_ERR("%s: please consider disabling jinja via --no-jinja, or use a custom chat template via --chat-template\n", __func__);
                SRV_ERR("%s: for example: --no-jinja --chat-template chatml\n", __func__);
                return false;
            }

            // IMPORTANT: chat_params is reused across sleeping / resuming states,
            //            never store llama_context/llama_model pointers in chat_params,
            //            as they may be invalidated after sleeping
            chat_params = {
                /* use_jinja             */ params_base.use_jinja,
                /* prefill_assistant     */ params_base.prefill_assistant,
                /* reasoning_format      */ params_base.reasoning_format,
                /* chat_template_kwargs  */ params_base.default_template_kwargs,
                /* tmpls                 */ std::move(chat_templates),
                /* allow_image           */ mctx ? mtmd_support_vision(mctx) : false,
                /* allow_audio           */ mctx ? mtmd_support_audio (mctx) : false,
                /* allow_video           */ mctx ? mtmd_helper_support_video(mctx) : false,
                /* enable_thinking       */ enable_thinking,
                /* reasoning_budget      */ params_base.sampling.reasoning_budget_tokens,
                /* reasoning_budget_msg  */ params_base.sampling.reasoning_budget_message,
                /* media_path            */ params_base.media_path,
                /* force_pure_content    */ params_base.force_pure_content_parser
            };

            {
                auto caps = common_chat_templates_get_caps(chat_params.tmpls.get());
                auto it = params_base.default_template_kwargs.find("preserve_reasoning");
                bool supported = caps.at("supports_preserve_reasoning");
                bool specified = params_base.preserve_reasoning_specified;
                // note: the kwarg is enabled by default if not specified explicitly, so check the value
                bool enabled = it != params_base.default_template_kwargs.end() && it->second == "true";
                if (supported) {
                    SRV_TRC("preserve_reasoning kwarg: %s\n",
                            it == params_base.default_template_kwargs.end() ? "unset (template default)" : it->second.c_str());
                } else {
                    SRV_TRC("%s", "preserve_reasoning kwarg: not supported by template\n");
                }
                if (supported && !specified) {
                    SRV_WRN("%s", "chat template supports preserving reasoning, it is enabled by default (may use more tokens, disable via --no-reasoning-preserve)\n");
                }
                if (supported && !enabled) {
                    SRV_INF("%s", "chat template supports preserving reasoning, consider enabling it via --reasoning-preserve\n");
                }
                if (!supported && specified && enabled) {
                    SRV_WRN("%s", "chat template does NOT support preserving reasoning, --reasoning-preserve has no effect\n");
                }
            }
        }

        return true;
    }

    server_slot * get_slot_by_id(int id_slot) {
        // note: allow id_slot to be out of bounds (wrap around)
        id_slot = id_slot % slots.size();

        for (server_slot & slot : slots) {
            if (slot.id == id_slot) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_slot_by_cmpl_id(const std::string & cmpl_id) {
        if (cmpl_id.empty()) {
            return nullptr;
        }

        for (server_slot & slot : slots) {
            if (slot.is_processing() && slot.task && slot.task->params.oaicompat_cmpl_id == cmpl_id) {
                return &slot;
            }
        }

        return nullptr;
    }

    int64_t get_context_reservation(const server_task & task) const {
        const int32_t n_predict = task.params.n_predict != -1 ? task.params.n_predict : params_base.n_predict;

        if (n_predict < 0) {
            return n_ctx;
        }

        return (int64_t) task.n_tokens() + n_predict;
    }

    bool has_context_reservation(const server_task & task) const {
        int64_t n_reserved = get_context_reservation(task);
        int32_t n_active = 0;

        for (const server_slot & slot : slots) {
            if (slot.is_processing()) {
                n_reserved += slot.n_ctx_reservation;
                ++n_active;
            }
        }

        if (n_reserved > n_ctx) {
            return false;
        }

        if (n_active == 0 || paged_admission_blocks == 0) {
            return true;
        }

        const uint64_t n_blocks = (n_reserved + params_base.block_size - 1) / params_base.block_size;
        return n_blocks <= paged_admission_blocks;
    }

    bool select_adaptive_context(server_task & task) {
        if (!common_context_is_adaptive(params_base)) {
            return true;
        }

        if (adaptive_context_unavailable) {
            send_error(task, "adaptive context is unavailable after a failed profile transition", ERROR_TYPE_UNAVAILABLE);
            return false;
        }

        if (task.is_parent()) {
            send_error(task, "adaptive context supports one completion per request", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        try {
            task.context_budget = common_context_budget_for_task(
                params_base, task.n_tokens(), task.params.n_predict, task.need_sampling());
        } catch (const std::exception & error) {
            send_error(task, string_format("invalid adaptive context budget: %s", error.what()), ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        const int32_t max_allowed_ctx = adaptive_max_ctx();
        if (max_allowed_ctx <= 0) {
            send_error(task, "adaptive context has no positive maximum context", ERROR_TYPE_SERVER);
            return false;
        }
        if (task.context_budget.total_tokens > max_allowed_ctx) {
            send_error(
                task.id,
                string_format("request (%lld tokens plus output) exceeds the shared context size (%d tokens)",
                    (long long) task.context_budget.total_tokens, max_allowed_ctx),
                ERROR_TYPE_EXCEED_CONTEXT_SIZE,
                task.n_tokens(),
                max_allowed_ctx);
            return false;
        }

        task.context_profile = common_context_profile_for_budget(
            params_base, task.context_budget.total_tokens);
        SRV_INF("adaptive context task id = %d: %lld prompt + %lld output = %lld -> %s\n",
            task.id,
            (long long) task.context_budget.prompt_tokens,
            (long long) task.context_budget.output_reserve,
            (long long) task.context_budget.total_tokens,
            adaptive_status_profile_name((int) task.context_profile).c_str());
        return true;
    }

    bool adaptive_slots_idle() const {
        return std::none_of(slots.begin(), slots.end(), [](const server_slot & slot) {
            return slot.is_processing();
        });
    }

    void unbind_slots_from_context() {
        for (auto & slot : slots) {
            slot.ctx_tgt = nullptr;
            slot.ctx_dft = nullptr;
            slot.mem.init(nullptr, nullptr);
            slot.spec = nullptr;
            slot.n_ctx = 0;
            slot.bootstrap_pending = false;
        }
    }

    void enter_adaptive_unavailable() {
        spec.reset();
        spec_init.reset();
        params_base.speculative.draft.ctx_tgt = nullptr;
        params_base.speculative.draft.ctx_dft = nullptr;
        ctx_dft = nullptr;
        model_dft = nullptr;
        if (llama_init && llama_init->context()) {
            llama_init->release_context();
        }
        ctx_tgt = nullptr;
        n_ctx = 0;
        unbind_slots_from_context();
        adaptive_context_unavailable = true;
        publish_adaptive_status();
        SRV_ERR("%s", "adaptive context is unavailable after a failed transition; rejecting inference tasks\n");
    }

    void bind_slots_to_active_context() {
        if (auto_cache_enabled()) {
            cur_fp = auto_compute_fingerprint();
            auto_index_scan();
        }
        const int slot_ctx = active_n_ctx_slot();
        for (auto & slot : slots) {
            slot.ctx_tgt = ctx_tgt;
            slot.ctx_dft = ctx_dft;
            slot.mem.init(ctx_tgt, ctx_dft);
            slot.spec = spec.get();
            slot.n_ctx = slot_ctx;
            slot.bootstrap_pending = false;
            slot.clear_context_on_release = elastic_paged_context;
        }
    }

    bool rebuild_active_speculation(common_context_profile profile) {
        spec.reset();
        spec_init.reset();
        params_base.speculative.draft.ctx_tgt = nullptr;
        params_base.speculative.draft.ctx_dft = nullptr;
        ctx_dft = nullptr;
        model_dft = nullptr;

        const bool use_mtp = (profile == COMMON_CONTEXT_PROFILE_MTP || profile == COMMON_CONTEXT_PROFILE_MTP_SHORT);
        if (!use_mtp) {
            ctx_tgt_seq_rm_type = common_context_can_seq_rm(ctx_tgt);
            ctx_dft_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
            bind_slots_to_active_context();
            return true;
        }

        ctx_tgt_seq_rm_type = common_context_can_seq_rm(ctx_tgt);
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            return false;
        }

        try {
            common_params params_dft = common_base_params_to_speculative(params_base);
            spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
            model_dft = spec_init->model();
            ctx_dft = spec_init->context();
            if (!ctx_dft) {
                return false;
            }
            params_base.speculative.draft.ctx_tgt = ctx_tgt;
            params_base.speculative.draft.ctx_dft = ctx_dft;
            spec.reset(common_speculative_init(params_base.speculative, params_base.n_parallel));
            if (!spec) {
                return false;
            }
            ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft);
        } catch (const std::exception & error) {
            SRV_ERR("adaptive speculative rebuild failed: %s\n", error.what());
            return false;
        }
        bind_slots_to_active_context();
        return true;
    }

    bool switch_adaptive_context(common_context_profile requested) {
        if (!common_context_is_adaptive(params_base)) {
            return true;
        }
        if (adaptive_context_unavailable) {
            return false;
        }
        if (requested == active_context_profile) {
            return true;
        }
        if (!adaptive_slots_idle()) {
            return false;
        }

        const int64_t transition_start_us = ggml_time_us();
        adaptive_context_transitioning = true;
        publish_adaptive_status();
        struct transition_status_guard {
            server_context_impl * impl;
            ~transition_status_guard() {
                impl->adaptive_context_transitioning = false;
                impl->publish_adaptive_status();
            }
        } transition_guard{this};

        const common_context_profile old_profile = active_context_profile;
        common_params old_params = params_base;
        old_params.speculative.draft.ctx_tgt = nullptr;
        old_params.speculative.draft.ctx_dft = nullptr;

        // Publish every reusable idle state before destroying either context.
        if (prompt_cache) {
            for (auto & slot : slots) {
                if (!slot.prompt.tokens.empty()) {
                    slot.prompt_save(*prompt_cache);
                }
            }
            prompt_cache->update();
        }
        for (auto & slot : slots) {
            slot.prompt_clear();
            slot.reset();
        }

        const auto discard_context = [&]() {
            spec.reset();
            spec_init.reset();
            params_base.speculative.draft.ctx_tgt = nullptr;
            params_base.speculative.draft.ctx_dft = nullptr;
            ctx_dft = nullptr;
            model_dft = nullptr;
            if (llama_init && llama_init->context()) {
                llama_init->release_context();
            }
            ctx_tgt = nullptr;
            unbind_slots_from_context();
        };

        const auto restore_old = [&]() -> bool {
            active_context_profile = old_profile;
            params_base = old_params;
            params_base.speculative.draft.ctx_tgt = nullptr;
            params_base.speculative.draft.ctx_dft = nullptr;
            apply_profile_params(old_profile);
            const bool old_resident = (old_profile == COMMON_CONTEXT_PROFILE_MTP_SHORT || old_profile == COMMON_CONTEXT_PROFILE_MTP);
            if (!llama_model_mtp_weights_set_resident(model_tgt,
                    old_resident, adaptive_test_mtp_fault("rollback", old_profile))) {
                ctx_tgt = nullptr;
                unbind_slots_from_context();
                return false;
            }
            if (adaptive_test_fault("rollback-after-residency", old_profile)) {
                SRV_WRN("%s", "adaptive test fault: rollback after MTP residency\n");
                return false;
            }
            if (!llama_init->recreate_context(params_base)) {
                ctx_tgt = nullptr;
                unbind_slots_from_context();
                return false;
            }
            ctx_tgt = llama_init->context();
            if (adaptive_test_fault("rollback-after-context", old_profile)) {
                SRV_WRN("%s", "adaptive test fault: rollback after context recreation\n");
                discard_context();
                return false;
            }
            if (ctx_tgt == nullptr || !rebuild_active_speculation(old_profile)) {
                discard_context();
                return false;
            }
            n_ctx = llama_n_ctx(ctx_tgt);
            adaptive_context_unavailable = false;
            return true;
        };

        discard_context();

        apply_profile_params(requested);

        const bool resident = (requested == COMMON_CONTEXT_PROFILE_MTP_SHORT || requested == COMMON_CONTEXT_PROFILE_MTP);
        const auto req_name = adaptive_status_profile_name((int) requested);
        if (!llama_model_mtp_weights_set_resident(model_tgt, resident,
                adaptive_test_mtp_fault("candidate", requested))) {
            SRV_WRN("adaptive test fault: candidate %s MTP residency/upload\n",
                    req_name.c_str());
            SRV_ERR("adaptive context transition to %s failed; attempting rollback\n",
                    req_name.c_str());
            if (!restore_old()) {
                enter_adaptive_unavailable();
            }
            return false;
        }
        if (adaptive_test_fault("candidate-after-residency", requested)) {
            SRV_WRN("adaptive test fault: candidate %s after MTP residency\n",
                    req_name.c_str());
            SRV_ERR("adaptive context transition to %s failed; attempting rollback\n",
                    req_name.c_str());
            if (!restore_old()) {
                enter_adaptive_unavailable();
            }
            return false;
        }
        if (!llama_init->recreate_context(params_base)) {
            SRV_ERR("adaptive context transition to %s failed; attempting rollback\n",
                    req_name.c_str());
            if (!restore_old()) {
                enter_adaptive_unavailable();
            }
            return false;
        }
        ctx_tgt = llama_init->context();
        if (ctx_tgt == nullptr || !rebuild_active_speculation(requested)) {
            SRV_ERR("%s", "adaptive context transition built an unusable speculative profile; attempting rollback\n");
            discard_context();
            if (!restore_old()) {
                enter_adaptive_unavailable();
            }
            return false;
        }
        active_context_profile = requested;
        adaptive_context_unavailable = false;
        n_ctx = llama_n_ctx(ctx_tgt);
        const int32_t target_batch = (int32_t) llama_n_batch(ctx_tgt);
        if (target_batch > batch.n_tokens_alloc) {
            batch.init(std::max(target_batch, batch.n_tokens_alloc), llama_model_n_embd_inp(model_tgt));
        }
        const auto mtp_info = llama_model_mtp_weights_get_info(model_tgt);
        SRV_INF("adaptive context transition complete: %s, n_ctx=%d, transition_ms=%.3f, MTP_GPU=%zu, MTP_HOST=%zu, MTP_ALLOC=%zu, model_instance=%" PRIu64
                ", model_loads=%" PRIu64 ", main_gpu_upload_bytes=%" PRIu64
                ", mtp_gpu_upload_bytes=%" PRIu64 ", mtp_reloads=%" PRIu64 "\n",
                adaptive_status_profile_name((int) requested).c_str(), n_ctx,
                (ggml_time_us() - transition_start_us) / 1000.0,
                mtp_info.gpu_allocated_bytes, mtp_info.host_bytes, mtp_info.allocated_bytes,
                mtp_info.model_instance, mtp_info.model_load_count, mtp_info.main_gpu_upload_bytes,
                mtp_info.mtp_gpu_upload_bytes, mtp_info.mtp_reloads);
        return true;
    }

    server_slot * get_available_slot(const server_task & task) {
        if (elastic_paged_context && !has_context_reservation(task)) {
            SRV_DBG("%s", "deferring task: paged KV reservation exceeds an active admission limit\n");
            return nullptr;
        }

        server_slot * ret = nullptr;

        bool update_cache = false;

        // if a specific slot is requested, use it (still goes through cache update logic below)
        if (task.id_slot != -1) {
            ret = get_slot_by_id(task.id_slot);
            if (ret) {
                SLT_INF(*ret, "selected slot by id (%d)\n", task.id_slot);
                // Unified idle-slot caching clears the device sequence after
                // saving it. An explicit id must therefore restore its RAM
                // entry before normal LCP trimming; otherwise the slot is
                // silently reprocessed from zero on every alternation.
                update_cache = prompt_cache && prompt_cache->n_tokens() > 0 && ret->prompt.tokens.empty();
            }
        }

        // find the slot that has at least n% prompt similarity
        if (slot_prompt_similarity != 0.0f) {
            float f_sim_best = 0;
            size_t restorable_best = 0;

            for (server_slot & slot : slots) {
                if (task.id_slot != -1 && slot.id != task.id_slot) {
                    continue;
                }

                // skip the slot if it is not available
                if (slot.is_processing()) {
                    SLT_TRC(slot, " - skipping, is_processing = %d\n", slot.is_processing());
                    continue;
                }

                const auto & tokens = slot.prompt.tokens;

                // skip the slot if it does not contains cached tokens
                if (tokens.empty()) {
                    SLT_TRC(slot, "%s", " - skipping, slot is empty\n");
                    continue;
                }

                const auto reuse = server_prompt_plan_reuse(
                        slot.prompt, task.tokens, prompt_reuse_alignment(),
                        prompt_live_native_restorable(slot, task.tokens), false);
                const size_t lcp_len = reuse.lexical_tokens;
                const size_t restorable = reuse.restorable_tokens;
                const float f_sim_cur = float(restorable) / task.tokens.size();

                SLT_TRC(slot,
                        " - checking restorable sim = %.3f (%zu restorable, %zu lexical/%zu) > %.3f\n",
                        f_sim_cur, restorable, lcp_len, task.tokens.size(), slot_prompt_similarity);

                // select the current slot if the criteria match
                if ((restorable > restorable_best ||
                     (restorable == restorable_best && f_sim_cur > f_sim_best)) &&
                        f_sim_cur > slot_prompt_similarity) {
                    f_sim_best = f_sim_cur;
                    restorable_best = restorable;

                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                const float f_keep = (f_sim_best*task.tokens.size()) / ret->prompt.tokens.size();

                if (task.id_slot == -1) {
                    SLT_INF(*ret,
                            "selected slot by restorable prefix, n_restorable = %zu, f_sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                            restorable_best, f_sim_best, slot_prompt_similarity, f_keep);
                }

                // if we are about to lose a large portion of the existing context - save it in the prompt cache
                if (f_keep < 0.5f) {
                    update_cache = true;
                }
            }
        }

        // find the slot that has been least recently used
        if (ret == nullptr) {
            int64_t t_last = -1;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                // select the current slot if the criteria match
                if (!ret || slot.t_last_used <= t_last) {
                    t_last = slot.t_last_used;
                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                SLT_INF(*ret, "selected slot by LRU, t_last = %" PRId64 "\n", t_last);

                update_cache = true;
            }
        }

        if (ret) {
            // Second auto-save site for when cache_idle_slots is OFF (the idle-flush path in
            // process_single_task -> auto_save never runs). Here get_available_slot just picked `ret`
            // for a new task and `update_cache` signals its prior KV is about to be discarded, so we
            // persist it before the prompt_save/prompt_load below overwrites it. Mutually exclusive
            // with the primary site via !cache_idle_slots, so no double-save. Reads `update_cache`
            // BEFORE the `&& prompt_cache` narrowing so disk save works without --cache-ram. The
            // callee carries all correctness gates; `ret` is idle so this never stalls generation.
            if (auto_cache_enabled() && !params_base.cache_idle_slots && update_cache) {
                auto_save_slot_if_useful(*ret);
            }

            update_cache = update_cache && prompt_cache;

            // cache prompts only for completion tasks
            update_cache = update_cache && task.type == SERVER_TASK_TYPE_COMPLETION;
            update_cache = update_cache && task.params.cache_prompt;

            if (update_cache) {
                SRV_TRC("%s", "updating prompt cache\n");

                const int64_t t_start = ggml_time_us();

                ret->prompt_save(*prompt_cache);

if (task.params.cache_prompt) {
                    const auto cache_result = ret->prompt_load_result(*prompt_cache, task.tokens);
                    if (cache_result == server_prompt_cache_result::needs_bootstrap) {
                        ret->bootstrap_pending = true;
                        SLT_INF(*ret, "%s", "target-only cache hit requires MTP bootstrap suffix\n");
                    } else if (cache_result != server_prompt_cache_result::hit &&
                               cache_result != server_prompt_cache_result::unchanged) {
                        ret->prompt_clear();
                    }
                }

                prompt_cache->update();

                SRV_TRC("prompt cache update took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
            }
        }

        return ret;
    }

    // return true if at least one slot has been cleared
    // TODO: improve logic
    //       - smarter decision which slot to clear (LRU or longest prompt?)
    //       - move slot to level 2 cache instead of removing?
    //       - instead of purging, try to store and resume later?
    bool try_clear_idle_slots() {
        bool res = false;

        if (!params_base.kv_unified) {
            return res;
        }

        for (auto & slot : slots) {
            if (slot.is_processing()) {
                continue;
            }

            if (slot.prompt.n_tokens() > 0) {
                SRV_WRN("purging slot %d with %zu tokens\n", slot.id, slot.prompt.tokens.size());

                slot.prompt_clear();

                res = true;

                // clear slots one by one
                break;
            }
        }

        return res;
    }

    std::vector<common_adapter_lora_info> construct_lora_list(const std::map<int, float> & config) const {
        std::vector<common_adapter_lora_info> output = params_base.lora_adapters; // copy
        for (size_t i = 0; i < output.size(); ++i) {
            auto it = config.find(i);
            if (it != config.end()) {
                output[i].scale = it->second;
            } else {
                output[i].scale = 0.0f;
            }
        }
        return output;
    }

    bool launch_slot_with_task(server_slot & slot, server_task && task) {
        struct bootstrap_guard {
            server_slot & slot;
            bool committed = false;
            ~bootstrap_guard() {
                if (!committed) {
                    slot.bootstrap_pending = false;
                }
            }
        } bootstrap_state{slot};

        // A new task is being assigned to this slot: its prompt may differ from whatever produced
        // the slot's current `logits_last` (even at the same token count). Invalidate the capture so
        // a SLOT_SAVE issued on the new prompt can never serialize a stale, mismatched distribution.
        // The stamp is re-established only by a real decode of the new prompt (the capture point).
        slot.logits_last.clear();
        slot.logits_last_n_tokens = -1;

        // process per-request lora adapters
        if (!task.params.lora.empty()) {
            auto task_loras = construct_lora_list(task.params.lora);
            if (!are_lora_equal(task_loras, slot.lora)) {
                // if lora has changed, check to see if the cache should be cleared
                if (lora_should_clear_cache(slot.lora, task_loras)) {
                    SLT_TRC(slot, "clearing cache for lora change. %zu loras -> %zu loras\n", slot.lora.size(), task.params.lora.size());
                    slot.prompt.clear();
                } else {
                    SLT_TRC(slot, "keeping cache for alora. %zu target loras\n", task_loras.size());
                }
                slot.lora = task_loras;
            }
        } else {
            slot.lora = params_base.lora_adapters;
        }

        // if using alora, make sure it's only a single one requested and active
        size_t alora_invocation_start = task.tokens.size();
        if (lora_all_alora(slot.lora)) {
            const auto & enabled_ids = lora_get_enabled_ids(slot.lora);
            // TODO: This will error out if a user requests two aloras, but only
            // provides the activation string for one. We could, instead search
            // for all requested alora activation strings and then either keep
            // only the last one, or reject if multiple are found.
            if (enabled_ids.size() != 1) {
                send_error(task, "Cannot run multiple aLoRAs in a single request", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            const auto & lora = slot.lora[enabled_ids[0]].ptr;

            // get the pointer and count for the invocation tokens
            const uint64_t      n_invocation_tokens = llama_adapter_get_alora_n_invocation_tokens(lora);
            const llama_token * invocation_tokens   = llama_adapter_get_alora_invocation_tokens  (lora);

            // scan backwards through the prompt tokens to find the last
            // occurrence of the invocation sequence
            int match_idx = static_cast<int>(n_invocation_tokens) - 1;
            for (int i = task.tokens.size() - 1; i >= 0; --i) {
                // the token in this position matches the next token to find in
                // the invocation sequence
                if (task.tokens[i] == invocation_tokens[match_idx]) {
                    // if it's a full match, we've found the start
                    if (match_idx == 0) {
                        alora_invocation_start = i;
                        break;
                    }
                    // otherwise, check the next token in the sequence
                    --match_idx;
                } else {
                    // no match in this position, so start looking over again
                    match_idx = static_cast<int>(n_invocation_tokens) - 1;
                }
            }

            // if the activation string is not found, disable the alora
            if (alora_invocation_start == task.tokens.size()) {
                SLT_DBG(slot, "alora %zu requested, but not found. deactivating\n", enabled_ids[0]);
                slot.lora[enabled_ids[0]].scale = 0.0f;
            } else {
                SLT_DBG(slot, "alora %zu activated starting at %zu\n", enabled_ids[0], alora_invocation_start);
                slot.alora_invocation_start = alora_invocation_start;
            }
        }

        if (!task.tokens.validate(ctx_tgt)) {
            send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        SLT_DBG(slot, "launching slot : %s\n", safe_json_to_str(slot.to_json()).c_str());

        // initialize samplers
        if (task.need_sampling()) {
            try {
                slot.smpl.reset(common_sampler_init(model_tgt, task.params.sampling));
            } catch (std::exception & e) {
                std::string err_msg = std::string("Failed to initialize samplers: ") + e.what();
                send_error(task, err_msg, ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            const bool need_pre_sample_logits = task.params.sampling.n_probs > 0 && !task.params.post_sampling_probs;

            bool use_backend_sampling = task.params.sampling.backend_sampling;

            // TODO: getting pre sampling logits is not yet supported with backend sampling
            use_backend_sampling &= !need_pre_sample_logits;

            // TODO: tmp until backend sampling is fully implemented
            if (use_backend_sampling) {
                llama_set_sampler(ctx_tgt, slot.id, common_sampler_get(slot.smpl.get()));
            } else {
                llama_set_sampler(ctx_tgt, slot.id, nullptr);
            }

            SLT_TRC(slot, "sampler chain: %s\n", common_sampler_print(slot.smpl.get()).c_str());
            SLT_TRC(slot, "sampler params: \n%s\n", task.params.sampling.print().c_str());

            if (spec && !common_speculative_get_synth_probs(spec.get()).empty()) {
                const uint32_t seed = task.params.sampling.seed == LLAMA_DEFAULT_SEED
                    ? std::random_device{}()
                    : task.params.sampling.seed;
                slot.spec_synth_rng.seed(seed);
            }
        } else {
            slot.smpl.reset();
        }

slot.loop_guard.configure(task.params.reasoning_loop_guard);

        const bool task_uses_dflash = std::find(
                task.params.speculative.types.begin(),
                task.params.speculative.types.end(),
                COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH) != task.params.speculative.types.end();
        if (slot.can_speculate() && task_uses_dflash) {
            slot.adaptive_dm.configure(task.params.speculative);
            const int base_n_max = common_speculative_n_max(&task.params.speculative);
            slot.adaptive_dm.reset_profit_if_config_changed(
                    task.params.speculative, base_n_max, slot.prompt.n_tokens(), &task.params.sampling);
            slot.adaptive_dm.reset_request_state();
            if (slot.adaptive_dm.dm_adaptive) {
                slot.adaptive_dm.apply_profit_recommendation(
                        slot.adaptive_dm.decide_profit_n_max(base_n_max));
            } else {
                slot.adaptive_dm.adaptive_n_max = -1;
            }
        }

        // the per-request limit takes priority over the global one
        slot.n_predict_max = task.params.n_predict != -1 ? task.params.n_predict : params_base.n_predict;

        if (params_base.kv_paged && params_base.kv_paged_dynamic) {
            int64_t request_tokens = task.n_tokens();
            if (slot.n_predict_max > 0) {
                request_tokens += slot.n_predict_max;
            }

            // The paged pool budget is n_ctx: prompt + predicted output must
            // fit or the KV growth would OOM mid-prefill and abort the whole
            // process (ggml_abort on CUDA alloc failure). Fail cleanly with a
            // context error so the client can compact instead of crash-looping.
            if (request_tokens > n_ctx) {
                slot.n_kv_reservation = 0;
                send_error(
                    task.id,
                    string_format("request (%lld tokens) exceeds the available context size (%d tokens), compact the context and retry",
                        (long long) request_tokens, n_ctx),
                    ERROR_TYPE_EXCEED_CONTEXT_SIZE,
                    (int32_t) task.n_tokens(),
                    n_ctx);
                return false;
            }

            slot.n_kv_reservation = std::min<int64_t>(request_tokens, n_ctx);

            int64_t total_tokens = slot.n_kv_reservation;
            for (const server_slot & active_slot : slots) {
                if (&active_slot != &slot && active_slot.is_processing()) {
                    total_tokens += active_slot.n_kv_reservation;
                }
            }
            total_tokens = std::min<int64_t>(total_tokens, n_ctx);

            SLT_INF(slot, "reserving paged KV for %lld tokens\n", (long long) total_tokens);
            if (!llama_memory_reserve(llama_get_memory(ctx_tgt), (uint32_t) total_tokens)) {
                send_error(task, "Failed to reserve paged KV cache.", ERROR_TYPE_SERVER);
                slot.n_kv_reservation = 0;
                return false;
            }
        }

        slot.n_ctx_reservation = elastic_paged_context ? get_context_reservation(task) : 0;

        slot.task = std::make_unique<const server_task>(std::move(task));

        slot.state = slot.task->is_child()
            ? SLOT_STATE_WAIT_OTHER // wait for the parent to process prompt
            : SLOT_STATE_STARTED;

        // reset server kill-switch counter
        n_empty_consecutive = 0;

        SLT_INF(slot, "processing task, is_child = %d\n", slot.task->is_child());
        bootstrap_state.committed = true;
        return true;
    }

    bool loop_guard_accept_enabled(const server_slot & slot) const {
        return slot.task &&
               slot.smpl &&
               slot.task->params.reasoning_loop_guard.mode != COMMON_REASONING_LOOP_GUARD_OFF &&
               slot.task->params.sampling.reasoning_budget_tracking;
    }

    bool handle_loop_guard_accept(server_slot & slot, const common_sampler_accept_info & info) {
        if (!loop_guard_accept_enabled(slot) || !info.is_generated) {
            return true;
        }

        if (!server_accept_info_is_reasoning(info)) {
            slot.visible_output_tokens++;
            return true;
        }

        slot.reasoning_output_tokens++;

        const bool forcing_reasoning_end = info.reasoning_state_before == REASONING_BUDGET_FORCING ||
                                           info.reasoning_state_after  == REASONING_BUDGET_FORCING;
        if (forcing_reasoning_end) {
            return true;
        }

        slot.loop_guard.accept(info.token, SERVER_LOOP_REGION_REASONING);

        const bool token_is_eog = llama_vocab_is_eog(vocab, info.token);
        if (!slot.loop_guard.should_check(SERVER_LOOP_REGION_REASONING, token_is_eog, forcing_reasoning_end)) {
            return true;
        }

        const auto check = slot.loop_guard.check(SERVER_LOOP_REGION_REASONING);
        if (!check.triggered) {
            return true;
        }

        slot.loop_guard_triggered = true;
        slot.loop_guard_reason = server_loop_guard_reason_to_string(check);

        const auto & params = slot.task->params.reasoning_loop_guard;
        if (params.mode == COMMON_REASONING_LOOP_GUARD_FORCE_CLOSE &&
                slot.loop_guard_interventions < params.interventions_max &&
                common_sampler_force_reasoning_end(slot.smpl.get())) {
            slot.loop_guard_interventions++;
            slot.loop_guard_action = "force-close";
            SLT_WRN(slot, "reasoning loop guard force-closing hidden reasoning: %s\n", slot.loop_guard_reason.c_str());
            return false;
        }

        slot.loop_guard_action = "stop";
        slot.stop = STOP_TYPE_LIMIT;
        slot.stop_detail = "reasoning_loop_guard";
        slot.has_next_token = false;
        SLT_WRN(slot, "reasoning loop guard stopping generation: %s\n", slot.loop_guard_reason.c_str());
        return false;
    }

    common_sampler_accept_callback make_loop_guard_accept_callback(server_slot & slot) {
        if (!loop_guard_accept_enabled(slot)) {
            return {};
        }

        return [this, &slot](const common_sampler_accept_info & info) {
            return handle_loop_guard_accept(slot, info);
        };
    }

    void apply_adaptive_profit_decision(server_slot & slot) {
        if (!slot.uses_dflash() || !slot.adaptive_dm.dm_adaptive) {
            return;
        }

        const int base_n_max = common_speculative_n_max(&slot.task->params.speculative);
        const int previous_n_max = slot.adaptive_dm.adaptive_n_max;
        const int recommended_n_max = slot.adaptive_dm.decide_profit_n_max(base_n_max);
        slot.adaptive_dm.apply_profit_recommendation(recommended_n_max);

        if (slot.adaptive_dm.adaptive_n_max != previous_n_max) {
            SLT_INF(slot, "adaptive draft-max profit: %d -> %d (score=%.2f)\n",
                    previous_n_max,
                    slot.adaptive_dm.adaptive_n_max,
                    (double) slot.adaptive_dm.profit_current_score);
        }
    }

    bool process_token(completion_token_output & result, server_slot & slot) {
        const bool stopped_before_process = slot.stop != STOP_TYPE_NONE && !slot.has_next_token;

        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = result.text_to_send;
        slot.sampled = result.tok;

        slot.generated_text += token_str;
        if (slot.task->params.return_tokens) {
            slot.generated_tokens.push_back(result.tok);
        }
        slot.has_next_token = !stopped_before_process;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = validate_utf8(slot.generated_text) < slot.generated_text.size();

        // search stop word and delete it
        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());

            const std::string str_test = slot.generated_text.substr(pos);
            bool send_text = true;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                slot.generated_text.erase(
                    slot.generated_text.begin() + pos + stop_pos,
                    slot.generated_text.end());
                pos = std::min(slot.n_sent_text, slot.generated_text.size());
            } else if (slot.has_next_token && !llama_vocab_is_eog(vocab, result.tok) ) {
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }

            // check if there is any token to predict
            if (send_text) {
                // no send the stop word in the response
                result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            } else {
                result.text_to_send = "";
            }

            slot.add_token(result);
            if (slot.task->params.stream) {
                send_partial_response(slot, result, false);
            }
        }

        if (incomplete && !stopped_before_process) {
            slot.has_next_token = true;
        }

        // if context shifting is disabled, make sure that we don't run out of context
        if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.stop_detail    = "context_limit";
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, n_gen = %d, n_ctx = %d\n",
                    slot.prompt.n_tokens(), slot.task->n_tokens(), (int) slot.stats.n_gen, slot.n_ctx);
        }

        // check the limits
        if (slot.stats.n_gen > 0 && slot.has_next_token && !slot.has_budget()) {
            slot.stop           = STOP_TYPE_LIMIT;
            slot.stop_detail    = "token_limit";
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by limit, n_gen = %d, n_predict = %d\n", (int) slot.stats.n_gen, slot.task->params.n_predict);
        }

        if (slot.has_new_line) {
            // require that each new line has a whitespace prefix (i.e. indentation) of at least slot.params.n_indent
            if (slot.task->params.n_indent > 0) {
                // check the current indentation
                // TODO: improve by not doing it more than once for each new line
                if (slot.last_nl_pos > 0) {
                    size_t pos = slot.last_nl_pos;

                    int n_indent = 0;
                    while (pos < slot.generated_text.size() && (slot.generated_text[pos] == ' ' || slot.generated_text[pos] == '\t')) {
                        n_indent++;
                        pos++;
                    }

                    if (pos < slot.generated_text.size() && n_indent < slot.task->params.n_indent) {
                        slot.stop           = STOP_TYPE_LIMIT;
                        slot.stop_detail    = "indentation_limit";
                        slot.has_next_token = false;

                        // cut the last line
                        slot.generated_text.erase(pos, std::string::npos);

                        SLT_DBG(slot, "stopped by indentation limit, n_gen = %d, n_indent = %d\n", (int) slot.stats.n_gen, n_indent);
                    }
                }

                // find the next new line
                {
                    const size_t pos = slot.generated_text.find('\n', slot.last_nl_pos);

                    if (pos != std::string::npos) {
                        slot.last_nl_pos = pos + 1;
                    }
                }
            }
        }

        // check if there is a new line in the generated text
        if (result.text_to_send.find('\n') != std::string::npos) {
            slot.has_new_line = true;

            // if we have seen a new line, we stop after a certain time limit, but only upon another new line
            if (slot.task->params.t_max_predict_ms > 0 && slot.stats.t_gen_ms() > slot.task->params.t_max_predict_ms) {
                slot.stop           = STOP_TYPE_LIMIT;
                slot.stop_detail    = "time_limit";
                slot.has_next_token = false;

                SLT_DBG(slot, "stopped by time limit, n_gen = %d, t_max_predict_ms = %d ms\n", (int) slot.stats.n_gen, (int) slot.task->params.t_max_predict_ms);
            }
        }

        if (llama_vocab_is_eog(vocab, result.tok)) {
            slot.stop           = STOP_TYPE_EOS;
            slot.stop_detail    = "eos";
            slot.has_next_token = false;

            SLT_DBG(slot, "%s", "stopped by EOS\n");
        }

        SLT_DBG(slot, "n_gen = %d, n_remaining = %d, next token: %5d '%s'\n", (int) slot.stats.n_gen, slot.n_remaining(), result.tok, token_str.c_str());

        return slot.has_next_token; // continue
    }

    void populate_token_probs(const server_slot & slot, completion_token_output & result, bool post_sampling, bool special, int idx) const {
        const size_t n_probs_request = slot.task->params.sampling.n_probs;

        if (post_sampling) {
            const auto * cur_p = common_sampler_get_candidates(slot.smpl.get(), true);
            const size_t max_probs = cur_p->size;
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                if (cur_p->data[i].id == result.tok) {
                    result.prob = cur_p->data[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                // Some samplers do return 0.0 probabilities, others don't.
                // Filter 0.0 probailities, to ensure the behavior is consistent.
                if (cur_p->data[i].p == 0.0) {
                    break;
                }

                result.probs.push_back({
                    cur_p->data[i].id,
                    common_token_to_piece(ctx_tgt, cur_p->data[i].id, special),
                    cur_p->data[i].p
                });
            }
        } else {
            std::vector<llama_token_data> cur = get_token_probabilities(ctx_tgt, idx, n_probs_request);
            const size_t max_probs = cur.size();
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                // set probability for sampled token
                if (cur[i].id == result.tok) {
                    result.prob = cur[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({
                    cur[i].id,
                    common_token_to_piece(ctx_tgt, cur[i].id, special),
                    cur[i].p
                });
            }
        }
    }

    void send_error(const server_task & task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type);
    }

    void send_error(const server_slot & slot, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(slot.task->id, error, type, slot.task->n_tokens(), slot.n_ctx);
    }

    void send_error(const int id_task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER, const int32_t n_prompt_tokens = 0, const int32_t n_ctx = 0) {
        SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

        if (type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
            GGML_ASSERT(n_ctx > 0 && n_prompt_tokens >= 0);
        }

        auto res = std::make_unique<server_task_result_error>();
        res->id              = id_task;
        res->err_type        = type;
        res->err_msg         = error;
        res->n_prompt_tokens = n_prompt_tokens;
        res->n_ctx           = n_ctx;

        queue_results.send(std::move(res));
    }

    void send_partial_response(server_slot & slot, const completion_token_output & tkn, bool is_progress, bool is_begin = false) {
        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id    = slot.task->id;
        res->index = slot.task->index;

        if (is_progress) {
            res->is_progress        = true;
            res->progress.total     = slot.task->n_tokens();
            res->progress.cache     = slot.stats.n_prompt_cached;
            res->progress.processed = slot.prompt.tokens.size();
            res->progress.time_ms   = slot.stats.t_elapsed_us() / 1000;
        }
        if (is_begin) {
            res->is_begin = true;
        } else {
            res->content = tkn.text_to_send;
            res->tokens  = { tkn.tok };
        }

        res->n_decoded             = slot.stats.n_gen;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.stats.n_prompt_cached;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            res->prob_output = tkn; // copy the token probs
        }

        // populate timings if this is final response or timings_per_token is enabled
        if (slot.stop != STOP_TYPE_NONE || slot.task->params.timings_per_token) {
            res->stats = slot.stats;
        }

        queue_results.send(std::move(res));
    }

    void send_final_response(server_slot & slot) {
        auto res = std::make_unique<server_task_result_cmpl_final>();

        res->id      = slot.task->id;
        res->id_slot = slot.id;

        res->index = slot.task->index;

        // keep copy of last generated text for debugging purposes
        if (slots_debug) {
            slot.debug_generated_text = slot.generated_text;
        }

        // in stream mode, content and tokens are already in last partial chunk
        if (slot.task->params.stream) {
            res->content     = "";
            res->tokens      = llama_tokens{};
        } else {
            res->content     = std::move(slot.generated_text);
            res->tokens      = std::move(slot.generated_tokens);
        }
        res->stats           = slot.stats;
        res->prompt          = slot.task->tokens.detokenize(ctx_tgt, true);
        res->response_fields = std::move(slot.task->params.response_fields);

        res->truncated             = slot.truncated;
        res->n_decoded             = slot.stats.n_gen;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.stats.n_prompt_cached;
        res->n_tokens_cached       = slot.prompt.n_tokens();
        res->has_new_line          = slot.has_new_line;
        res->stopping_word         = slot.stopping_word;
        res->stop                  = slot.stop;
        res->stop_detail           = slot.stop_detail;
        res->reasoning_output_tokens = slot.reasoning_output_tokens;
        res->visible_output_tokens   = slot.visible_output_tokens;
        res->loop_guard_triggered    = slot.loop_guard_triggered;
        res->loop_guard_action       = slot.loop_guard_action;
        res->loop_guard_reason       = slot.loop_guard_reason;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->stream            = slot.task->params.stream;
        res->include_usage     = slot.task->params.include_usage;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            if (!slot.task->params.stream && slot.stop == STOP_TYPE_WORD) {
                const llama_tokens stop_word_toks = common_tokenize(ctx_tgt, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end() - safe_offset);
            } else {
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end());
            }
        }

        res->generation_params = slot.task->params; // copy the parameters

        queue_results.send(std::move(res));
    }

    void send_embedding(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_embd>();
        res->id        = slot.task->id;
        res->index     = slot.task->index;
        res->n_tokens  = slot.task->n_tokens();
        res->res_type  = slot.task->params.res_type;

        const int n_embd_out = llama_model_n_embd_out(model_tgt);

        std::vector<float> embd_res(n_embd_out, 0.0f);

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = nullptr;
            if (llama_pooling_type(slot.ctx_tgt) == LLAMA_POOLING_TYPE_NONE) {
                embd = llama_get_embeddings_ith(slot.ctx_tgt, i);
            } else {
                embd = llama_get_embeddings_seq(slot.ctx_tgt, batch.seq_id[i][0]);
            }

            if (embd == nullptr) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->embedding.push_back(std::vector<float>(n_embd_out, 0.0f));
                continue;
            }

            // normalize only when there is pooling
            if (llama_pooling_type(slot.ctx_tgt) != LLAMA_POOLING_TYPE_NONE) {
                common_embd_normalize(embd, embd_res.data(), n_embd_out, slot.task->params.embd_normalize);
                res->embedding.push_back(embd_res);
                break;
            }

            res->embedding.emplace_back(embd, embd + n_embd_out);
        }

        SLT_DBG(slot, "%s", "sending embeddings\n");

        queue_results.send(std::move(res));
    }

    void send_rerank(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_rerank>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = llama_get_embeddings_seq(ctx_tgt, batch.seq_id[i][0]);
            if (embd == NULL) {
                embd = llama_get_embeddings_ith(ctx_tgt, i);
            }

            if (embd == NULL) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->score = -1e6;
                continue;
            }

            res->score = embd[0];
        }

        SLT_DBG(slot, "sending rerank result, res.score = %f\n", res->score);

        queue_results.send(std::move(res));
    }

    //
    // Functions to process the task
    //

    // tokenize the input if it's set by CLI, return false on error
    bool tokenize_cli_input(server_task & task) {
        try {
            auto & prompt = task.cli_prompt;
            if (mctx != nullptr) {
                task.tokens = process_mtmd_prompt(mctx, prompt, task.cli_files, init_opt);
            } else {
                task.tokens = std::move(tokenize_input_prompts(vocab, mctx, prompt, true, true, init_opt)[0]);
            }
            task.cli_prompt.clear();
            task.cli_files.clear();
        } catch (const std::exception & e) {
            send_error(task, std::string("Failed to format input: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        return true;
    }

    std::vector<server_slot *> get_free_slots(size_t n_slots_needed, int exclude_id_slot) {
        std::vector<server_slot *> free_slots;
        for (auto & slot : slots) {
            if (!slot.is_processing() && slot.id != exclude_id_slot) {
                free_slots.push_back(&slot);
            }
            if (free_slots.size() >= n_slots_needed) {
                break;
            }
        }
        return free_slots;
    }

    // KVarN publishes and restores only complete descriptor groups. Standard
    // and recurrent caches keep the upstream scheduling cadence.
    int32_t prompt_reuse_alignment() const {
        const int32_t kvarn_group = params_base.kvarn.type != LLAMA_KVARN_TYPE_DISABLED ?
                params_base.kvarn.group : 0;
        GGML_ASSERT(kvarn_group >= 0);
        return server_prompt_reuse_alignment(kvarn_group);
    }

    bool prompt_reuse_boundary_is_stable(int64_t n_tokens) const {
        const int32_t alignment = prompt_reuse_alignment();
        return n_tokens >= 0 && n_tokens%alignment == 0;
    }

    size_t prompt_live_native_restorable(
            const server_slot & slot,
            const server_tokens & requested) const {
        const size_t lcp_len = slot.prompt.tokens.get_common_prefix(requested);
        if (lcp_len == 0) {
            return 0;
        }
        if (lcp_len == slot.prompt.tokens.size()) {
            return lcp_len;
        }
        const llama_pos requested_p0 = slot.prompt.tokens.pos_next(lcp_len);
        llama_pos target_p0 = requested_p0;
        llama_pos target_p1 = -1;
        llama_pos draft_p0 = requested_p0;
        llama_pos draft_p1 = -1;
        const bool target_planned = llama_memory_seq_rm_plan(
                llama_get_memory(slot.ctx_tgt), slot.id, requested_p0, -1,
                &target_p0, &target_p1);
        const bool draft_planned = !slot.ctx_dft || llama_memory_seq_rm_plan(
                llama_get_memory(slot.ctx_dft), slot.id, requested_p0, -1,
                &draft_p0, &draft_p1);
        if (!target_planned || !draft_planned || target_p1 >= 0 ||
                (slot.ctx_dft && draft_p1 >= 0)) {
            return 0;
        }
        const llama_pos common_p0 = slot.ctx_dft ? std::min(target_p0, draft_p0) : target_p0;
        return common_p0 > 0 && common_p0 <= requested_p0 ?
                slot.prompt.tokens.size_up_to_pos(common_p0) : 0;
    }

    // launch multiple slots for parent + child tasks
    bool launch_slots_with_parent_task(server_slot & parent_slot, std::vector<server_slot *> & child_slots, server_task && parent_task) {
        GGML_ASSERT(!parent_slot.is_processing());
        GGML_ASSERT(parent_task.is_parent());
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());

        int id_parent = parent_task.id;

        SRV_TRC("launching slots for parent task id_task = %d with %zu child tasks\n", id_parent, parent_task.child_tasks.size());

        // to be called in case of failure to release all launched slots
        auto release_slots = [this, id_parent]() {
            for (auto & slot : slots) {
                if (slot.is_processing() && (
                        slot.task->id == id_parent ||
                        slot.task->id_parent == id_parent
                )) {
                    slot.release();
                }
            }
        };

        // launch all child tasks first
        size_t idx = 0;
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());
        for (auto * slot : child_slots) {
            int id_child = parent_task.child_tasks[idx].id;
            if (!launch_slot_with_task(*slot, std::move(parent_task.child_tasks[idx]))) {
                SRV_ERR("failed to launch slot with child task, id_task = %d\n", id_child);
                release_slots();
                return false;
            }
            idx++;
        }

        // finally, launch the parent task
        if (!launch_slot_with_task(parent_slot, std::move(parent_task))) {
            SRV_ERR("failed to launch slot with task, id_task = %d\n", id_parent);
            release_slots();
            return false;
        }

        return true;
    }

// n_tokens_cur: the number of tokens added to the batch for the current slot
    void create_checkpoint(server_slot & slot, const int64_t n_tokens_cur, llama_pos pos_min, llama_pos pos_max) {
        const int id_task = slot.task ? slot.task->id : -1;

        // evict checkpoints within min-step of a previous checkpoint, unless they were
        // created by the current task
        // only when the list is full, otherwise short prompts keep just the oldest checkpoint
        int64_t last = -1;
        for (auto it = slot.prompt.checkpoints.begin();
                slot.prompt.checkpoints.size() + 1 >= (size_t) params_base.n_ctx_checkpoints &&
                it != slot.prompt.checkpoints.end(); ) {
            if ((*it)->id_task != id_task && last >= 0 && (*it)->n_tokens <= last + params_base.checkpoint_min_step) {
                SLT_TRC(slot, "erasing context checkpoint too close to an earlier one (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                        (*it)->pos_min, (*it)->pos_max, (*it)->n_tokens, (float) (*it)->size() / 1024 / 1024);

                it = slot.prompt.checkpoints.erase(it);
                continue;
            }

            last = (*it)->n_tokens;
            ++it;
        }

        while (slot.prompt.checkpoints.size() >= (size_t) params_base.n_ctx_checkpoints) {
            // make room for the new checkpoint, if needed
            const auto & cur = *slot.prompt.checkpoints.front();

            SLT_WRN(slot, "erasing old context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                    cur.pos_min, cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);

            slot.prompt.checkpoints.erase(slot.prompt.checkpoints.begin());
        }

// replace an existing checkpoint at the same n_tokens instead of appending a duplicate
        {
            const int64_t n_tokens_new = slot.prompt.n_tokens() - n_tokens_cur;
            for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end(); ) {
                if ((*it)->n_tokens == n_tokens_new) {
                    SLT_TRC(slot, "superseding context checkpoint at n_tokens = %" PRId64 "\n", (*it)->n_tokens);
                    it = slot.prompt.checkpoints.erase(it);
                } else {
                    ++it;
                }
            }
        }

        auto checkpoint = std::make_shared<common_prompt_checkpoint>();
        auto & cur = *checkpoint;

        cur.id_task = id_task;

        // [TAG_CHECKPOINTS_FIX_POS_MIN]
        // TODO: here we incorrectly deterimne that the saved checkpoint data covers the [pos_min, pos_max] range
        //       this is not true for SWA models: https://github.com/ggml-org/llama.cpp/pull/24411#issuecomment-4677983225
        cur.update_pos(slot.prompt.n_tokens() - n_tokens_cur, pos_min, pos_max);

        cur.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        cur.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        // stash the draft's speculative state with the checkpoint
        cur.update_spec(spec.get(), slot.id);

        slot.prompt.checkpoints.push_back(std::move(checkpoint));

        SLT_TRC(slot,
                "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                (int) slot.prompt.checkpoints.size(), params_base.n_ctx_checkpoints, cur.pos_min,
                cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);
    }

    bool restore_checkpoint_transaction(
            server_slot & slot,
            const common_prompt_checkpoint & checkpoint,
            llama_context * target,
            llama_context * draft,
            bool restore_target,
            bool restore_draft,
            bool restore_speculative) {
        const bool restored = server_prompt_restore_transaction(
                target, draft, spec.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY,
                { checkpoint.data_tgt.data(), checkpoint.data_tgt.size() },
                { checkpoint.data_dft.data(), checkpoint.data_dft.size() },
                { checkpoint.data_spec.data(), checkpoint.data_spec.size() },
                restore_target, restore_draft, restore_speculative);
        if (!restored) {
            SLT_WRN(slot, "%s", "checkpoint restore preparation failed; destination is unchanged\n");
        }
        return restored;
    }

    void route_slot_state(const server_task & task, server_slot & slot, bool save) {
        const int64_t started = ggml_time_us();
        bool restored_memory = false;
        try {
            // KVarN children are valid transfer endpoints for the fixed no-MTP
            // long tier. Their native state restores through the same bounded
            // streaming path; a q4_0 source is converted explicitly first.
            if (!adaptive_model_identity || params_base.n_parallel != 1 || mctx ||
                    !params_base.lora_adapters.empty() || !slot.lora.empty() ||
                    !params_base.control_vectors.empty() || params_base.kv_paged ||
                    params_base.kv_tail_tokens != "0") {
                throw std::runtime_error("router streaming state requires identified, unmodified text model and standard KV");
            }
            if (!save && task.slot_action.route_target_no_mtp && (ctx_dft || common_context_is_adaptive(params_base))) {
                throw std::runtime_error("router target-only restore requires a fixed no-MTP child");
            }
            adaptive_model_identity->verify_sources();
            // Account transfer/index/staging, token vectors and context-sized
            // metadata. Disk payload bytes are not a RAM reservation. Keep the
            // explicit adaptive snapshot's separate five-copy guard unchanged.
            const uint64_t workspace = (save ? 16ULL : 48ULL)*1024*1024 +
                    uint64_t(llama_n_ctx_seq(ctx_tgt))*(save ? 16 : 512);
            adaptive_slot_cache_reservation reservation;
            if (workspace > adaptive_slot_ram_budget(params_base.cache_ram_mib) ||
                    !reservation.acquire(prompt_cache.get(), size_t(workspace))) {
                throw std::runtime_error("router streaming workspace exceeds RAM budget");
            }
            const uint64_t max_bytes = params_base.slot_save_max_bytes > 0 ?
                    uint64_t(params_base.slot_save_max_bytes) : SIZE_MAX;
            const std::string route_store = std::filesystem::path(task.slot_action.filepath).parent_path().string();
            if (route_store.empty()) {
                throw std::runtime_error("router state directory is unavailable");
            }
            size_t bytes = 0;
            size_t count = 0;
            std::string result_filename = task.slot_action.filename;
            if (save) {
                llama_synchronize(ctx_tgt);
                const auto position = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);
                const size_t evaluated = slot.prompt.tokens.size_up_to_pos(position + 1);
                if (position < 0 || slot.prompt.tokens.has_media() ||
                        slot.prompt.tokens.pos_next(evaluated) != position + 1) {
                    throw std::runtime_error("router source has no complete evaluated text prefix");
                }
                auto tokens = slot.prompt.tokens.get_text_tokens();
                tokens.resize(evaluated);
                if (task.slot_action.route_state_reuse) {
                    if (auto existing = auto_find_exact_snapshot(tokens)) {
                        // The index/identity probe above is metadata-only.
                        // Verify the one selected object once, immediately
                        // before announcing reuse; the restore loader is not
                        // involved because this path only hands the object to
                        // another child.
                        auto verified = read_unified_snapshot(existing->state_path, true);
                        if (!verified || verified->model != adaptive_model_identity->fingerprint() ||
                                !common_prompt_cache_layout_reusable(verified->layout, common_prompt_cache_layout(ctx_tgt)) ||
                                verified->tokens != tokens) {
                            verified.reset();
                        }
                        if (!verified) {
                            // Treat an invalid selected object as a miss and
                            // create the normal route snapshot under the same
                            // pre-publication budget transaction.
                        } else {
                        server_route_state_lease reference(existing->state_path,
                                server_route_state_lock_mode::reference);
                        if (!reference.acquired()) {
                            throw std::runtime_error("compatible unified snapshot is busy");
                        }
                        result_filename = std::filesystem::path(verified->state_path).filename().string();
                        bytes = verified->file_bytes;
                        count = verified->tokens.size();
                        SLT_INF(slot, "router snapshot reused: path=%s checksum=%" PRIu64 " inode-independent=atomic\n",
                                verified->state_path.c_str(), verified->state_checksum);
                        }
                    }
                }
                if (bytes == 0) {
                server_route_state_store_lock store(route_store, server_route_state_store_lock_mode::exclusive);
                if (!store.acquired()) {
                    throw std::runtime_error("router snapshot store is busy");
                }
                auto limit_plan = std::make_shared<unified_snapshot_limit_plan>();
                limit_plan->dir = route_store;
                limit_plan->just_written = task.slot_action.filepath;
                limit_plan->max_count = params_base.slot_save_max_count;
                limit_plan->max_bytes = params_base.slot_save_max_bytes;
                bytes = server_route_state_save(ctx_tgt, slot.id, tokens, *adaptive_model_identity,
                        task.slot_action.filepath, max_bytes, 0, nullptr,
                        [limit_plan](uint64_t incoming) { return limit_plan->prepare(incoming); },
                        [limit_plan]() { return limit_plan->commit(); }, &store, nullptr);
                count = tokens.size();
                }
            } else {
                // A nonempty target is not silently erased. The router should
                // restore immediately after loading the new child.
                if (!slot.prompt.tokens.empty() || llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id) >= 0) {
                    throw std::runtime_error("router streaming restore destination is not empty");
                }
                for (uint32_t seq = 0; seq < llama_n_seq_max(ctx_tgt); ++seq) {
                    if (llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), seq) >= 0) {
                        throw std::runtime_error("router streaming restore context is not empty");
                    }
                }
                server_route_state_lease reference(task.slot_action.filepath,
                        server_route_state_lock_mode::reference);
                if (!reference.acquired()) {
                    throw std::runtime_error("router state is busy");
                }
                auto file = server_route_state_read(task.slot_action.filepath, max_bytes, slot.n_ctx, false);
                if (file.model != adaptive_model_identity->fingerprint()) {
                    throw std::runtime_error("router state model identity differs");
                }
                std::unique_ptr<server_route_state_lease> converted_reference;
                std::string restore_path = task.slot_action.filepath;
                if (file.layout != common_prompt_cache_layout(ctx_tgt)) {
                    // Explicit, restricted q4_0 -> native compact conversion.
                    // The converted snapshot is persisted with provenance and
                    // reused on later restores; this is a lossy conversion, not
                    // a native prefill.
                    const std::string converted = convert_route_snapshot(file, &reference);
                    if (converted.empty()) {
                        throw std::runtime_error("router state layout differs and no conversion is available");
                    }
                    converted_reference = std::make_unique<server_route_state_lease>(
                            converted, server_route_state_lock_mode::reference);
                    if (!converted_reference->acquired()) {
                        throw std::runtime_error("converted router state is busy");
                    }
                    file = server_route_state_read(converted, max_bytes, slot.n_ctx, false);
                    if (file.model != adaptive_model_identity->fingerprint() ||
                            file.layout != common_prompt_cache_layout(ctx_tgt)) {
                        throw std::runtime_error("converted router state identity or layout differs");
                    }
                    restore_path = converted;
                }
                if (file.tokens.size() > llama_n_ctx_seq(ctx_tgt)) {
                    throw std::runtime_error("router state tokens do not fit destination context");
                }
                server_prompt candidate;
                candidate.tokens.insert(file.tokens);
                if (!candidate.tokens.validate(ctx_tgt) || candidate.tokens.pos_next() != file.position + 1) {
                    throw std::runtime_error("router state tokens are invalid");
                }
                llama_tokens loaded(file.tokens.size());
                const size_t read = llama_state_seq_load_file_streaming(ctx_tgt, restore_path.c_str(),
                        slot.id, loaded.data(), loaded.size(), &count, file.state_bytes, file.state_checksum);
                restored_memory = read != 0;
                if (read != file.state_bytes || count != file.tokens.size() || loaded != file.tokens ||
                        llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id) != file.position) {
                    throw std::runtime_error("router native streaming restore failed");
                }
                // Whole evaluated prefix + a new suffix needs no rewind, even
                // for recurrent models. Normal prompt checkpoints are created
                // during the suffix and remain available for subsequent turns.
                slot.prompt = std::move(candidate);
                slot.just_restored = true;
                const auto target_mtp = llama_model_mtp_weights_get_info(llama_get_model(ctx_tgt));
                slot.bootstrap_pending = ctx_dft && target_mtp.managed;
                if (slot.bootstrap_pending) {
                    SLT_INF(slot, "%s", "unified target-only snapshot restored; MTP bootstrap required before generation\n");
                }
                slot.prompt_cache_source = "disk";
                slot.prompt_cache_reason = "unified_snapshot_restore";
                slot.logits_last.clear();
                slot.logits_last_n_tokens = -1;
                slot.restored_logits.clear();
                bytes = file.file_bytes;
            }
            SLT_INF(slot, "router streaming %s: tokens=%zu bytes=%zu buffer=8388608 workspace=%" PRIu64 "\n",
                    save ? "save" : "restore", count, bytes, workspace);
            auto result = std::make_unique<server_task_result_slot_save_load>();
            result->id = task.id;
            result->id_slot = slot.id;
            result->filename = result_filename;
            result->is_save = save;
            result->n_tokens = count;
            result->n_bytes = bytes;
            result->t_ms = (ggml_time_us() - started)/1000.0;
            queue_results.send(std::move(result));
        } catch (const std::exception & error) {
            if (restored_memory) { slot.prompt_clear(); }
            send_error(task, std::string("Unable to transfer router state: ") + error.what(), ERROR_TYPE_SERVER);
        }
    }

    // returns false to decline the task, it is offered again after the decode is done
    bool process_single_task(server_task && task, bool is_yielding) {
        // while yielding, an encode / decode is running and only reading the server state is safe
        if (is_yielding && task.type != SERVER_TASK_TYPE_METRICS && task.type != SERVER_TASK_TYPE_SLOT_GET) {
            SRV_DBG("decoding, decline task, id_task = %d\n", task.id);
            return false;
        }

        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                {
                    // special case: if input is provided via CLI, tokenize it first
                    // otherwise, no need to tokenize as it's already done inside the HTTP thread
                    if (task.cli) {
                        if (!tokenize_cli_input(task)) {
                            break;
                        }
                    }

                    const int id_task = task.id;

                    if (!select_adaptive_context(task)) {
                        break;
                    }

                    if (common_context_is_adaptive(params_base) &&
                            task.context_profile != active_context_profile) {
                        if (!adaptive_slots_idle()) {
                            SRV_DBG("adaptive profile transition waits for idle slots, id_task = %d\n", id_task);
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                        if (!switch_adaptive_context(task.context_profile)) {
                            send_error(task, "adaptive context profile transition failed", ERROR_TYPE_SERVER);
                            break;
                        }
                    }

                    if (elastic_paged_context && get_context_reservation(task) > n_ctx) {
                        send_error(task.id,
                                   string_format("request (%d tokens plus output) exceeds the shared context size (%d tokens)",
                                                 task.n_tokens(), n_ctx),
                                   ERROR_TYPE_EXCEED_CONTEXT_SIZE,
                                   task.n_tokens(), n_ctx);
                        break;
                    }

                    server_slot * slot = get_available_slot(task);

                    //
                    // slot scheduling logic
                    //

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (task.is_parent()) {
                        // try getting free slots for all child tasks
                        size_t n_child_tasks = task.child_tasks.size();
                        std::vector<server_slot *> child_slots = get_free_slots(n_child_tasks, slot->id);
                        if (child_slots.size() < n_child_tasks) {
                            SRV_DBG("not enough free slots for child tasks, n_free = %zu, n_children = %zu, defer task, id_task = %d\n", child_slots.size(), n_child_tasks, id_task);
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                        if (!launch_slots_with_parent_task(*slot, child_slots, std::move(task))) {
                            SRV_ERR("failed to launch slot with parent task, id_task = %d\n", id_task);
                            break; // drop the task
                        }
                    } else if (!launch_slot_with_task(*slot, std::move(task))) {
                        SRV_ERR("failed to launch slot with task, id_task = %d\n", id_task);
                        break; // drop the task
                    }

                    if (params_base.cache_idle_slots && prompt_cache) {
                        for (auto & slot : slots) {
                            if (!slot.is_processing()) {
                                // Auto-save (write path, invariant 5): persist this slot's KV to disk
                                // BEFORE the in-memory cache drops it. This runs at task-launch time for
                                // idle slots (off the launching task's generation hot path), reusing the
                                // exact SLOT_SAVE machinery. Gate the CALL SITE so the entire call frame
                                // is elided when OFF (this is the dominant idle-slot-flush path under
                                // cache_idle_slots=true). The callee keeps its own first-statement gate
                                // as defense-in-depth.
                                if (auto_cache_enabled()) {
                                    auto_save_slot_if_useful(slot);
                                }
                                SLT_TRC(slot, "%s", "saving idle slot to prompt cache\n");

                                const bool saved = slot.prompt_save(*prompt_cache);
                                if (saved) {
                                    SLT_DBG(slot, "%s", "__TEST_TAG_CACHE_IDLE_SLOT__\n");
                                    prompt_cache->update();
                                }

                                // A unified sequence may be cleared only after its complete
                                // target/draft state was durably admitted to the RAM cache.
                                // KVarN deliberately rejects a per-sequence snapshot while
                                // another logical sequence owns the shared stream.
                                if (params_base.kv_unified && saved) {
                                    // [TAG_IDLE_SLOT_CLEAR]
                                    slot.prompt_clear();
                                }
                            }
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    // release slot linked with the task id
                    for (auto & slot : slots) {
                        if (slot.task && slot.task->id == task.id_target) {
                            slot.release();
                            break;
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CONTROL:
                {
                    auto res = std::make_unique<server_task_result_control>();
                    res->id = task.id;

                    server_slot * slot = get_slot_by_cmpl_id(task.params.control_cmpl_id);
                    if (slot == nullptr) {
                        SRV_WRN("control %s on unknown completion id=%s, no live slot\n",
                                task.params.control_action.c_str(), task.params.control_cmpl_id.c_str());
                        res->success = false;
                        res->message = "no active completion for this id";
                        queue_results.send(std::move(res));
                        break;
                    }

                    if (task.params.control_action == "reasoning_end") {
                        // the budget sampler only exists when reasoning control was armed
                        if (!slot->task->params.sampling.reasoning_control) {
                            res->success = false;
                            res->message = "reasoning control not enabled for this completion";
                            queue_results.send(std::move(res));
                            break;
                        }
                        // act on the live slot mid generation, never defer
                        res->success = common_sampler_reasoning_budget_force(slot->smpl.get());
                        if (!res->success) {
                            res->message = "reasoning is not active";
                        }
                    } else {
                        res->success = false;
                        res->message = "unknown control action";
                    }

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                    // do nothing
                } break;
            case SERVER_TASK_TYPE_METRICS:
                {
                    int n_processing_slots = 0;

                    for (server_slot & slot : slots) {
                        if (slot.is_processing()) {
                            n_processing_slots++;
                        }
                    }
                    SRV_DBG("n_processing_slots = %d\n", n_processing_slots);

                    auto res = std::make_unique<server_task_result_metrics>();
                    res->id                  = task.id;
                    res->n_processing_slots  = n_processing_slots;
                    res->n_tasks_deferred    = queue_tasks.queue_tasks_deferred_size();
res->metrics             = metrics;

                    if (task.metrics_reset_bucket) {
                        metrics.reset_bucket();
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_GET:
                {
                    json slots_data = json::array();
                    const auto adaptive_status = get_adaptive_status();

                    int n_idle_slots = 0;

                    for (server_slot & slot : slots) {
                        if (!slot.is_processing()) {
                            n_idle_slots++;
                        }

                        json slot_data = slot.to_json(slots_debug == 0);
                        if (adaptive_status.enabled) {
                            slot_data["adaptive_context"] = json {
                                {"enabled",              adaptive_status.enabled},
                                {"profile",              adaptive_status.profile},
                                {"state",                adaptive_status.state},
                                {"context_size",         adaptive_status.context_size},
                                {"context_size_long",    adaptive_status.context_size_long},
                                {"mtp_weights_resident", adaptive_status.mtp_weights_resident},
                            };
                        }
                        slots_data.push_back(std::move(slot_data));
                    }
                    SRV_DBG("n_idle_slots = %d\n", n_idle_slots);

                    auto res = std::make_unique<server_task_result_slots>();
                    res->id           = task.id;
                    res->slots_data   = std::move(slots_data);
                    res->n_idle_slots = n_idle_slots;

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_SAVE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // The streaming route format is selected by the target-only
                    // marker. The automatic-cache reuse marker only selects it
                    // when that cache is actually enabled; otherwise the legacy
                    // slot API contract is preserved for speculative targets.
                    if (task.slot_action.route_target_no_mtp ||
                            (task.slot_action.route_state_reuse && auto_cache_enabled())) {
                        route_slot_state(task, *slot, true);
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;
                    const std::string slot_store = task.slot_action.route_state_transfer
                        ? std::filesystem::path(filepath).parent_path().string()
                        : params_base.slot_save_path;
                    if (slot_store.empty()) {
                        send_error(task, "Unable to save slot: slot state directory is unavailable", ERROR_TYPE_SERVER);
                        break;
                    }

                    if (common_context_is_adaptive(params_base)) {
                        try {
                            if (!adaptive_model_identity) {
                                throw std::runtime_error("adaptive model identity is unavailable");
                            }
                            const size_t max_file_bytes = adaptive_slot_max_file_bytes(
                                *slot, ctx_tgt, ctx_dft, params_base.ctx_size_mtp_short, params_base.ctx_size_mtp,
                                adaptive_max_ctx(), params_base.cache_ram_mib, params_base.n_ctx_checkpoints);
                            adaptive_slot_cache_reservation reservation;
                            if (!reservation.acquire(prompt_cache.get(), adaptive_slot_working_bytes(max_file_bytes))) {
                                throw std::runtime_error("adaptive slot snapshot exceeds the global RAM cache budget");
                            }
                            const auto snapshot = adaptive_slot_capture(
                                *slot, ctx_tgt, ctx_dft, spec.get(), *adaptive_model_identity,
                                active_context_profile, params_base.ctx_size_mtp,
                                params_base.mtp_max_tokens, adaptive_long_ctx, max_file_bytes);
                            const auto encoded = adaptive_slot_encode(snapshot, max_file_bytes);
                            const size_t nwrite = adaptive_slot_write_file(filepath, encoded);
                            bool oversized = false;
                            slot_save_enforce_limits(slot_store, params_base.slot_save_max_count,
                                    params_base.slot_save_max_bytes, filepath, oversized);
                            if (oversized) {
                                throw std::runtime_error("slot snapshot exceeds --slot-save-max-mb; save rejected");
                            }

                            const int64_t t_end = ggml_time_us();
                            const double t_save_ms = (t_end - t_start) / 1000.0;
                            auto res = std::make_unique<server_task_result_slot_save_load>();
                            res->id       = task.id;
                            res->id_slot  = id_slot;
                            res->filename = filename;
                            res->is_save  = true;
                            res->n_tokens = snapshot.n_tokens;
                            res->n_bytes  = nwrite;
                            res->t_ms     = t_save_ms;
                            queue_results.send(std::move(res));
                        } catch (const std::bad_alloc &) {
                            send_error(task, "Unable to save slot: adaptive snapshot allocation failed", ERROR_TYPE_SERVER);
                        } catch (const std::exception & err) {
                            send_error(task, std::string("Unable to save slot: ") + err.what(), ERROR_TYPE_SERVER);
                        }
                        break;
                    }

                    std::vector<char> packed;
                    try {
                        packed = slot->prompt.tokens.serialize();
                    } catch (const std::exception & err) {
                        send_error(task, err.what(), ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }

                    GGML_ASSERT(packed.size() % sizeof(llama_token) == 0);
                    const size_t nwrite = llama_state_seq_save_file(
                        ctx_tgt, filepath.c_str(), slot->id,
                        reinterpret_cast<const llama_token *>(packed.data()), packed.size() / sizeof(llama_token));
                    if (nwrite == 0) {
                        send_error(task, "Unable to save slot", ERROR_TYPE_SERVER);
                        break;
                    }

                    // persist this slot's last-token logits as a sidecar (FULL/recurrent
                    // only). Best-effort - a missing/failed sidecar simply disables the regenerate
                    // fast-path for this snapshot. NOT folded into res->n_bytes (that contract stays
                    // "state-file bytes only").
                    //
                    // CRITICAL consistency guard: only write the sidecar when the captured logits
                    // provably belong to the EXACT state being saved, i.e. logits_last_n_tokens ==
                    // token_count. This blocks every stale-logits path (restore-then-save with no
                    // intervening decode; a spec-decode step that skipped the capture; a distribution
                    // left over from a prior task on this slot object) from persisting a sidecar that
                    // does not match the saved state - which would otherwise emit a wrong first token
                    // on a later regenerate with nothing to catch it.
                    const int32_t n_slot_tokens = (int32_t) slot->prompt.tokens.size();
                    std::error_code sidecar_ec;
                    std::filesystem::remove(slot_logits_sidecar_path(filepath), sidecar_ec);
                    if (nwrite > 0 && ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
                        if (slot->logits_last_n_tokens == n_slot_tokens && !slot->logits_last.empty()) {
                            const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));
                            const size_t nwrite_logits =
                                slot_logits_write(filepath, slot->logits_last, nv, (uint32_t) n_slot_tokens);
                            if (nwrite_logits == 0) {
                                SLT_WRN(*slot, "%s", "failed to write logits sidecar; regenerate fast-path disabled for this snapshot\n");
                            }
                        } else {
                            SLT_DBG(*slot, "no matching captured logits for this state (stamp=%d, token_count=%d); sidecar omitted\n",
                                    slot->logits_last_n_tokens, n_slot_tokens);
                        }
                    }

                    // enforce the bounded slot-save store (LRU by mtime). If this single
                    // snapshot exceeds the byte cap, reject the save instead of evicting everything.
                    if (nwrite > 0 &&
                        (params_base.slot_save_max_count > 0 || params_base.slot_save_max_bytes > 0)) {
                        bool oversized = false;
                        slot_save_enforce_limits(slot_store,
                                                 params_base.slot_save_max_count,
                                                 params_base.slot_save_max_bytes,
                                                 filepath, oversized);
                        if (oversized) {
                            send_error(task,
                                       "slot snapshot exceeds --slot-save-max-mb; save rejected",
                                       ERROR_TYPE_INVALID_REQUEST);
                            break;
                        }
                    }

                    const int64_t t_end = ggml_time_us();
                    const double t_save_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = true;
                    res->n_tokens = slot->prompt.tokens.size();
                    res->n_bytes  = nwrite;
                    res->t_ms     = t_save_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_RESTORE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (task.slot_action.route_target_no_mtp ||
                            (task.slot_action.route_state_reuse && auto_cache_enabled())) {
                        route_slot_state(task, *slot, false);
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    if (common_context_is_adaptive(params_base)) {
                        const common_context_profile old_profile = active_context_profile;
                        bool mutation_started = false;
                        std::optional<adaptive_slot_snapshot_blob> previous_snapshot;

                        auto rollback = [&]() -> bool {
                            if (!mutation_started) {
                                return true;
                            }
                            if (active_context_profile != old_profile && !switch_adaptive_context(old_profile)) {
                                return false;
                            }
                            if (adaptive_context_unavailable || !ctx_tgt) {
                                return false;
                            }
                            slot = get_slot_by_id(id_slot);
                            if (!slot) {
                                return false;
                            }
                            if (!previous_snapshot) {
                                slot->prompt_clear();
                                return true;
                            }
                            std::string rollback_error;
                            if (!adaptive_slot_restore(*previous_snapshot, *slot, ctx_tgt, ctx_dft,
                                    spec.get(), rollback_error)) {
                                SRV_ERR("adaptive slot rollback failed: %s\n", rollback_error.c_str());
                                return false;
                            }
                            return true;
                        };

                        auto fail_restore = [&](const std::string & reason, error_type type) {
                            if (!rollback()) {
                                enter_adaptive_unavailable();
                                send_error(task,
                                    "Unable to restore slot: " + reason + "; adaptive context is unavailable after rollback failure",
                                    ERROR_TYPE_UNAVAILABLE);
                                return;
                            }
                            send_error(task, "Unable to restore slot: " + reason, type);
                        };
                        adaptive_slot_cache_reservation reservation;
                        try {
                            if (!adaptive_model_identity) {
                                throw std::runtime_error("adaptive model identity is unavailable");
                            }

                            const size_t max_file_bytes = adaptive_slot_max_file_bytes(
                                *slot, ctx_tgt, ctx_dft, params_base.ctx_size_mtp_short, params_base.ctx_size_mtp,
                                adaptive_max_ctx(), params_base.cache_ram_mib, params_base.n_ctx_checkpoints);
                            if (!reservation.acquire(prompt_cache.get(), adaptive_slot_working_bytes(max_file_bytes))) {
                                throw std::runtime_error("adaptive slot snapshot exceeds the global RAM cache budget");
                            }
                            const auto file = adaptive_slot_read_file(filepath, max_file_bytes);
                            const auto snapshot = adaptive_slot_decode(file,
                                params_base.n_ctx_checkpoints > 0 ? (size_t) params_base.n_ctx_checkpoints : 0,
                                max_file_bytes);
                            if (snapshot.model_fingerprint.empty() ||
                                    snapshot.model_fingerprint != adaptive_model_identity->fingerprint()) {
                                throw std::runtime_error("adaptive slot snapshot model identity differs");
                            }
                            if (snapshot.ctx_size_mtp != params_base.ctx_size_mtp ||
                                    snapshot.mtp_max_tokens != params_base.mtp_max_tokens ||
                                    snapshot.long_ctx != adaptive_long_ctx) {
                                throw std::runtime_error("adaptive slot snapshot context limits differ");
                            }
                            const int raw_ctx = snapshot.profile == COMMON_CONTEXT_PROFILE_MTP_SHORT
                                ? params_base.ctx_size_mtp_short
                                : (snapshot.profile == COMMON_CONTEXT_PROFILE_MTP ? params_base.ctx_size_mtp
                                : (snapshot.profile == COMMON_CONTEXT_PROFILE_XLONG ? params_base.ctx_size_xlong
                                : (snapshot.profile == COMMON_CONTEXT_PROFILE_XXLONG ? params_base.ctx_size_xxlong : adaptive_long_ctx)));
                            const int padded_ctx = GGML_PAD(raw_ctx, 256);
                            int expected_ctx = std::min(padded_ctx, llama_model_n_ctx_train(model_tgt));
                            if (params_base.kv_unified_per_slot > 0) {
                                expected_ctx = std::min(expected_ctx, params_base.kv_unified_per_slot);
                            }
                            if (snapshot.active_ctx != expected_ctx) {
                                throw std::runtime_error("adaptive slot snapshot active context differs");
                            }

                            if (!slot->prompt.tokens.empty()) {
                                previous_snapshot = adaptive_slot_capture(
                                    *slot, ctx_tgt, ctx_dft, spec.get(), *adaptive_model_identity,
                                    old_profile, params_base.ctx_size_mtp, params_base.mtp_max_tokens,
                                    adaptive_long_ctx, max_file_bytes, false);
                            }

                            if (snapshot.profile != (uint32_t) active_context_profile) {
                                mutation_started = true;
                                if (!switch_adaptive_context((common_context_profile) snapshot.profile)) {
                                    throw std::runtime_error("adaptive context profile transition failed");
                                }
                                // The transition rebinds every slot to the new context.
                                slot = get_slot_by_id(id_slot);
                                if (!slot) {
                                    throw std::runtime_error("adaptive slot disappeared during profile transition");
                                }
                            }

                            mutation_started = true;
                            std::string restore_error;
                            if (!adaptive_slot_restore(snapshot, *slot, ctx_tgt, ctx_dft, spec.get(), restore_error)) {
                                throw std::runtime_error(restore_error);
                            }

                            const int64_t t_end = ggml_time_us();
                            const double t_restore_ms = (t_end - t_start) / 1000.0;
                            auto res = std::make_unique<server_task_result_slot_save_load>();
                            res->id       = task.id;
                            res->id_slot  = id_slot;
                            res->filename = filename;
                            res->is_save  = false;
                            res->n_tokens = snapshot.n_tokens;
                            res->n_bytes  = file.size();
                            res->t_ms     = t_restore_ms;
                            queue_results.send(std::move(res));
                        } catch (const std::bad_alloc &) {
                            fail_restore("adaptive snapshot allocation failed", ERROR_TYPE_INVALID_REQUEST);
                        } catch (const std::exception & err) {
                            fail_restore(err.what(), ERROR_TYPE_INVALID_REQUEST);
                        }
                        break;
                    }

                    size_t nread = 0;
                    try {
                        size_t n_packed = 0;
                        llama_tokens packed;
                        nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->id, nullptr, 0, &n_packed);
                        if (nread != 0) {
                            packed.resize(std::max<size_t>(1, n_packed));
                            nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->id, packed.data(), packed.size(), &n_packed);
                        }
                        if (nread == 0) {
                            throw std::runtime_error("No available space in KV cache or invalid slot save file");
                        }
                        packed.resize(n_packed);

                        server_tokens restored = server_tokens::deserialize(packed, mctx != nullptr);

                        if (restored.size() > (size_t) slot->n_ctx) {
                            throw std::runtime_error("Restored prompt does not fit in the slot context");
                        }

                        if (!restored.validate(ctx_tgt)) {
                            throw std::runtime_error("Invalid tokens in slot save file");
                        }

                        slot->prompt.clear();
                        slot->prompt.tokens = std::move(restored);
                    } catch (const std::exception & err) {
                        slot->prompt_clear();
                        send_error(task, std::string("Unable to restore slot: ") + err.what(), ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }

                    // [PR-24003] Reconstruct a context checkpoint at the restored position so the
                    // prompt-cache reuse path can reuse this state on the next matching request.
                    // Hybrid/recurrent (and SWA) models cannot partially rewind their memory, so
                    // without a checkpoint the matcher forces a full re-prefill; other models do not
                    // need it. Gates restored-slot KV reuse on just_restored.
                    slot->just_restored = true;
                    // Unmanaged MTP keeps the legacy draft catch-up path and
                    // does not support the resident-weight bootstrap API.
                    // Only managed target-only state needs that bootstrap.
                    slot->bootstrap_pending = slot->can_speculate() &&
                        llama_model_mtp_weights_get_info(llama_get_model(ctx_tgt)).managed;
                    if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
                        const auto ckpt_pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot->id);
                        const auto ckpt_pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot->id);
                        if (ckpt_pos_min >= 0 && params_base.n_ctx_checkpoints > 0) {
                            slot->prompt.checkpoints.clear();
                            create_checkpoint(*slot, 0, ckpt_pos_min, ckpt_pos_max);
                        }
                    }

                    // The restored state's freshly-sampled "running" distribution is unknown: the
                    // logits in `logits_last` (if any) belong to a PRIOR generation on this slot
                    // object, NOT to the restored state. Invalidate them so a subsequent SLOT_SAVE
                    // (e.g. a restore -> re-checkpoint with no intervening decode) cannot persist a
                    // stale, mismatched sidecar. (Belt-and-suspenders: the SLOT_SAVE stamp check
                    // already blocks this, since the stamp no longer equals the restored length.)
                    slot->logits_last.clear();
                    slot->logits_last_n_tokens = -1;

                    // Feature A/B: load the logits sidecar (if any) so an exact-prompt regenerate
                    // can emit the first token without re-decoding into the restored recurrent state.
                    // The token-count check binds the sidecar to exactly this restored state.
                    // On any failure restored_logits stays empty and Feature B falls back safely.
                    slot->restored_logits.clear();
                    if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
                        const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));
                        if (slot_logits_read(filepath, nv, (uint32_t) slot->prompt.tokens.size(), slot->restored_logits)) {
                            SLT_INF(*slot, "loaded logits sidecar (%d vocab, %zu tokens) - regenerate fast-path armed\n", nv, slot->prompt.tokens.size());
                        }
                    }

                    const int64_t t_end = ggml_time_us();
                    const double t_restore_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = false;
                    res->n_tokens = slot->prompt.tokens.size();
                    res->n_bytes  = nread;
                    res->t_ms     = t_restore_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_ERASE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // Erase token cache
                    const size_t n_erased = slot->prompt.tokens.size();

                    slot->prompt_clear();

                    auto res = std::make_unique<server_task_result_slot_erase>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->n_erased = n_erased;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_GET_LORA:
                {
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    auto & loras = params_base.lora_adapters;
                    auto res = std::make_unique<server_task_result_get_lora>();
                    res->id = task.id;
                    for (size_t i = 0; i < loras.size(); ++i) {
                        auto & lora = loras[i];
                        std::string alora_invocation_string = "";
                        const uint64_t n_alora_tokens = llama_adapter_get_alora_n_invocation_tokens(lora.ptr);
                        llama_tokens alora_invocation_tokens;
                        if (n_alora_tokens) {
                            const llama_token * alora_tokens = llama_adapter_get_alora_invocation_tokens(lora.ptr);
                            for (uint64_t j = 0; j < n_alora_tokens; ++j) {
                                alora_invocation_string += common_token_to_piece(vocab, alora_tokens[j]);
                                alora_invocation_tokens.push_back(alora_tokens[j]);
                            }
                        }
                        res->loras.push_back(server_task_result_get_lora::lora{
                            lora,
                            alora_invocation_string,
                            alora_invocation_tokens,
                        });
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SET_LORA:
                {
                    auto new_loras = construct_lora_list(task.set_lora);
                    // logging
                    for (size_t i = 0; i < new_loras.size(); ++i) {
                        SRV_TRC("set lora adapter idx=%zu scale=%f\n", i, new_loras[i].scale);
                    }
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    params_base.lora_adapters = new_loras;
                    auto res = std::make_unique<server_task_result_apply_lora>();
                    res->id = task.id;
                    queue_results.send(std::move(res));
                } break;
        }

        return true;
    }

    void iterate(std::vector<server_slot> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(slot);
            } catch (const std::exception & e) {
                SLT_ERR(slot, "got exception: %s\n", e.what());
                send_error(slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }

    void iterate(std::vector<server_slot *> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(*slot);
            } catch (const std::exception & e) {
                SLT_ERR(*slot, "got exception: %s\n", e.what());
                send_error(*slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot->release();
            }
        }
    }

    void abort_all_slots(const std::string & reason) {
        for (auto & slot : slots) {
            if (slot.is_processing()) {
                send_error(slot, reason, ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }

    // @ngxson : for debugging only
    int64_t t_pre_decode  = 0;
    int64_t t_decode      = 0;
    int64_t t_post_decode = 0;
    int64_t t_sampl       = 0;
    int64_t n_pre_decode  = 0;
    int64_t n_decode      = 0;
    int64_t n_post_decode = 0;
    int64_t n_sampl       = 0;
// #define DEBUG_TIMINGS
#ifdef DEBUG_TIMINGS
    struct scoped_timer {
        int64_t & t;
        int64_t & n;
        int64_t t_start;
        scoped_timer(int64_t & t_, int64_t & n_) : t(t_), n(n_) {
            t_start = ggml_time_us();
        }
        ~scoped_timer() {
            t += ggml_time_us() - t_start;
            n++;
        }
    };
#else
    struct scoped_timer {
        scoped_timer(int64_t &, int64_t &) {}
        ~scoped_timer() {}
    };
#endif

    void update_gpu_power_phase() {
        if (!gpu_power.enabled()) {
            return;
        }

        server_gpu_power_phase_arbitrator arbitrator;
        for (const auto & slot : slots) {
            arbitrator.observe(server_gpu_power_slot_state_from_slot_state(slot.state));
        }
        gpu_power.update(arbitrator.phase());
    }

    void update_slots() {
        update_gpu_power_phase();

#ifdef DEBUG_TIMINGS
        static int64_t t_prev = 0;
        int64_t t_start = ggml_time_us();
        if (t_start - t_prev > 5 * 1000 * 1000) { // every 5 seconds
            t_prev = t_start;
            SRV_INF("n_pre_decode      = %" PRId64 "\n", n_pre_decode);
            SRV_INF("avg t_pre_decode  = %f ms\n", (double) t_pre_decode / n_pre_decode / 1000.0);
            SRV_INF("avg t_decode      = %f ms\n", (double) t_decode / n_decode / 1000.0);
            SRV_INF("avg t_post_decode = %f ms\n", (double) t_post_decode / n_post_decode / 1000.0);
            SRV_INF("avg t_sampl       = %f ms\n", (double) t_sampl / n_sampl / 1000.0);
        }
#endif

        // check if all slots are idle
        {
            bool all_idle = true;

            for (auto & slot : slots) {
                if (slot.is_processing()) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                SRV_TRC("%s", "all slots are idle\n");

                metrics_flush_idle();

                return; // skip further processing

            } else {
                SRV_DBG("%s", "posting NEXT_RESPONSE\n");

                server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
                task.id = queue_tasks.get_new_id();
                queue_tasks.post(std::move(task));
            }
        }

        try {
            scoped_timer t(t_pre_decode, n_pre_decode);
            pre_decode();
            batch.render();
        } catch (const std::exception & e) {
            SRV_ERR("pre_decode() failed: %s\n", e.what());
            abort_all_slots("pre_decode() failed: " + std::string(e.what()));

            // the batch is half-built and not rendered, skip now to avoid UB
            return;
        }

        GGML_ASSERT(batch.slot_batched || batch.size() == 0);

        if (batch.slot_batched) {
            auto & slot_batched      = batch.slot_batched;
            auto & alora_scale       = batch.alora_scale;
            auto & alora_disabled_id = batch.alora_disabled_id;

            // TODO @ngxson : alora handling is too messy, need to refactor it to be more clear and maintainable
            // apply lora, only need to do it once per batch
            common_set_adapter_lora(ctx_tgt, slot_batched->lora);

            // if the lora is temporarily disabled for an alora, re-enable it
            // for next time
            if (alora_scale > 0.0f) {
                SRV_DBG("re-enabling alora with scale %f\n", alora_scale);
                slot_batched->lora[alora_disabled_id].scale = alora_scale;
            }

            llama_set_embeddings(ctx_tgt, slot_batched->need_embd());
        }

        llama_batch batch_view;
        int32_t off_next = 0;
        int32_t n_batch = llama_n_batch(ctx_tgt);
        for (int32_t off = 0; off < batch.size(); off = off_next) {
            const int32_t n_tokens = std::min(n_batch, batch.size() - off);
            try {
                scoped_timer t(t_decode, n_decode);
                // TODO @ngxson : maybe handle n_batch == 1 here instead of inside decode()

                batch_view = batch.get_view(off, n_tokens);
                bool ok = decode(n_batch, off, batch_view);
#ifdef DEBUG_TIMINGS
                llama_synchronize(ctx_tgt);
#endif

                if (ok) {
                    // move the head of the batch forward with the number of tokens we just processed
                    off_next = off + n_tokens;

                    // on successful decode, restore the original batch size
                    n_batch = llama_n_batch(ctx_tgt);
                } else {
                    // try again with the updated n_batch
                    continue;
                }
            } catch (const std::exception & e) {
                SRV_ERR("decode() failed: %s\n", e.what());
                abort_all_slots("decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }

            try {
                scoped_timer t(t_post_decode, n_post_decode);
                post_decode(n_tokens, off, batch_view);
            } catch (const std::exception & e) {
                SRV_ERR("post_decode() failed: %s\n", e.what());
                abort_all_slots("post_decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }
        }
    }

    void pre_decode() {
        // apply context-shift if needed
        // TODO: simplify and improve
        iterate(slots, [&](server_slot & slot) {
            if (slot.state == SLOT_STATE_GENERATING && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
                if (!params_base.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in process_token()
                    send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                if (mctx) {
                    // we should never reach this because params_base.ctx_shift is automatically disabled if mmproj is loaded
                    // we don't support ctx_shift because an image chunk may contains multiple tokens
                    GGML_ABORT("not supported by multimodal");
                }

                if (slot.task->is_parent() || slot.task->is_child()) {
                    send_error(slot, "context shift cannot be used for shared prompt", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                // Shift context
                int n_keep = slot.task->params.n_keep < 0 ? slot.task->n_tokens() : slot.task->params.n_keep;

                if (add_bos_token) {
                    n_keep += 1;
                }

                n_keep = std::min(slot.n_ctx - 4, n_keep);

                const int n_left    = slot.prompt.n_tokens() - n_keep;
                int       n_discard = slot.task->params.n_discard ? slot.task->params.n_discard : (n_left / 2);

                // ref: https://github.com/ggml-org/llama.cpp/pull/24786
                n_discard = std::clamp(n_discard, 0, std::max(0, n_left - 1));

                SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", n_keep, n_left, n_discard);

                slot.mem.seq_rm (slot.id, n_keep            , n_keep + n_discard);
                slot.mem.seq_add(slot.id, n_keep + n_discard, slot.prompt.tokens.pos_next(), -n_discard);

                // add generated tokens to cache
                // ref: https://github.com/ggml-org/llama.cpp/pull/16818#discussion_r2473269481
                {
                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                    llama_tokens new_tokens = slot.prompt.tokens.get_tokens(); // copy
                    for (size_t i = n_keep + n_discard; i < new_tokens.size(); i++) {
                        new_tokens[i - n_discard] = new_tokens[i];
                    }

                    new_tokens.resize(slot.prompt.tokens.size() - n_discard);

                    slot.prompt.clear();
                    slot.prompt.tokens.insert(new_tokens);
                }

                slot.truncated = true;
            }
        });

        // start populating the batch for this iteration
        batch.clear();

        // track if given slot can be batched with slots already in the batch
        auto & slot_batched = batch.slot_batched;

        std::vector<server_slot *> generating;
        std::vector<server_slot *> drafting;

        // determine which slots are generating and drafting
        iterate(slots, [&](server_slot & slot) {
            if (slot.state != SLOT_STATE_GENERATING) {
                return;
            }

            // check if we can batch this slot with the previous one
            if (!slot_batched) {
                slot_batched = &slot;
            } else if (!slot_batched->can_batch_with(slot)) {
                return;
            }

            generating.push_back(&slot);

            if (slot.uses_dflash() && slot.adaptive_dm.dm_adaptive) {
                slot.adaptive_cycle_start_us = ggml_time_us();
                slot.adaptive_draft_ms = 0.0f;
                slot.adaptive_requested_n_max = 0;
            }

            if (spec) {
                common_speculative_get_draft_params(spec.get(), slot.id).drafting = false;

                const bool use_ckpt_tgt = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
                const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

                const int n_draft_max = slot.get_n_draft_max();

                if (n_draft_max > 0) {
                    if (slot.uses_dflash() && slot.adaptive_dm.dm_adaptive) {
                        slot.adaptive_requested_n_max = n_draft_max;
                    }
                    GGML_ASSERT(slot.can_speculate());

                    if (!slot.spec_draft.empty()) {
                        // we have a previous (partial) draft to reuse
                        if (use_ckpt_tgt) {
                            GGML_ASSERT(!slot.spec_ckpt.empty());
                        }
                    } else {
                        GGML_ASSERT(slot.spec_i_batch.empty());

                        slot.spec_ckpt.update_pos(
                                slot.prompt.n_tokens(),
                                llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id),
                                llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id));
                        slot.spec_ckpt.data_spec.clear();
                        common_speculative_get_state(spec.get(), slot.id, slot.spec_ckpt.data_spec);

                        if (use_ckpt_dft) {
                            const auto capture = slot.spec_ckpt.update_dft(
                                    ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                            if (!capture.ok()) {
                                SLT_WRN(slot, "failed to capture draft checkpoint (status = %d, bytes = %zu)\n",
                                        int(capture.status), capture.bytes);
                                slot.spec_ckpt.clear();
                                return;
                            }
                        }

                        slot.spec_prompt = slot.prompt.tokens.get_text_tokens();

                        common_speculative_get_draft_params(spec.get(), slot.id) = {
                            /* .drafting = */ true,
                            /* .n_max    = */ n_draft_max,
                            /* .pos0     = */ slot.prompt.tokens.pos_next(),
                            /* .id_last  = */ slot.sampled,
                            /* .prompt   = */ &slot.spec_prompt,
                            /* .result   = */ &slot.spec_draft,
                            /* .dists    = */ &slot.spec_dists,
                            /* .temperature = */ slot.task->params.sampling.temp,
                            /* .seed     = */ common_sampler_get_seed(slot.smpl.get()),
                        };

                        drafting.push_back(&slot);
                    }
                }
            }
        });

        // generate the actual drafts (if any)
if (!drafting.empty()) {
            const int64_t t_draft_start = ggml_time_us();
            queue_tasks.yield_to_queue([&]() {
                common_speculative_draft(spec.get());
            });
            const float shared_draft_ms = (ggml_time_us() - t_draft_start) / 1000.0f;
            size_t drafted_tokens_total = 0;
            for (const auto * slot : drafting) {
                drafted_tokens_total += slot->spec_draft.size();
            }
            for (auto * slot : drafting) {
                if (slot->uses_dflash() && slot->adaptive_dm.dm_adaptive) {
                    // Upstream drafts the cohort in one batched call. Attribute that
                    // shared wall time by actual drafted-token work so a shallow
                    // adaptive DFlash slot is not charged the same as a deep one.
                    slot->adaptive_draft_ms = drafted_tokens_total > 0
                        ? shared_draft_ms * (float) slot->spec_draft.size() / (float) drafted_tokens_total
                        : 0.0f;
                }
            }
        }

        // make checkpoints if needed
        iterate(drafting, [&](server_slot & slot) {
            auto & draft = slot.spec_draft;
            auto & ckpt  = slot.spec_ckpt;

            slot.stats.n_draft_tokens += draft.size();

            // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
            const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

            if (ctx_dft) {
                if (use_ckpt_dft) {
                    if (!restore_checkpoint_transaction(
                                slot, ckpt, nullptr, ctx_dft,
                                false, true, false)) {
                        draft.clear();
                        return;
                    }
                }

                if (!llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, ckpt.pos_max + 1, -1)) {
                    GGML_ABORT("failed to remove sequence %d\n", slot.id);
                }
            }

            if (!draft.empty()) {
                const bool use_ckpt_tgt = server_speculative_rollback_requires_checkpoint(
                        ctx_tgt_seq_rm_type, common_context_seq_rm_max_rollback(ctx_tgt), draft.size());

                const bool use_ckpt_dft = server_speculative_rollback_requires_checkpoint(
                        ctx_dft_seq_rm_type, common_context_seq_rm_max_rollback(ctx_dft), draft.size());

                if (use_ckpt_tgt) {
                    //const int64_t t_start = ggml_time_us();

                    const auto capture = ckpt.update_tgt(
                            ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                    if (!capture.ok()) {
                        SLT_WRN(slot, "failed to capture target speculative checkpoint (status = %d, bytes = %zu)\n",
                                int(capture.status), capture.bytes);
                        ckpt.clear();
                        draft.clear();
                        return;
                    }

                    //const int64_t t_total = ggml_time_us() - t_start;
                    //printf("checkpoint total: %f ms\n", t_total / 1000.0);

                    SLT_DBG(slot, "created speculative checkpoint (pos_min = %d, pos_max = %d, n_tokens = %d, size = %.3f MiB, draft = %.3f MiB)\n",
                            ckpt.pos_min, ckpt.pos_max, slot.prompt.n_tokens(),
                            (float) ckpt.size() / 1024 / 1024,
                            (float) ckpt.data_dft.size() / 1024 / 1024);
                }

                if (use_ckpt_dft) {
                    const auto capture = ckpt.update_dft(
                            ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                    if (!capture.ok()) {
                        SLT_WRN(slot, "failed to capture draft speculative checkpoint (status = %d, bytes = %zu)\n",
                                int(capture.status), capture.bytes);
                        ckpt.clear();
                        draft.clear();
                        return;
                    }
                }
            }
        });

        // update the batch with the sampled/drafted tokens
        iterate(generating, [&](server_slot & slot) {
            slot.handle_last_sampled_token(batch);
        });

        // process in chunks of params.n_batch
        int32_t n_batch  = llama_n_batch(ctx_tgt);
        int32_t n_ubatch = llama_n_ubatch(ctx_tgt);
        // The first real target suffix after a target-only snapshot must fit the
        // dense NextN output buffer. Restore the normal physical batch afterward.
        if (std::any_of(slots.begin(), slots.end(), [](const server_slot & slot) {
                    return slot.bootstrap_pending;
                })) {
            n_batch = std::min(n_batch, n_ubatch);
        }

        auto & alora_scale       = batch.alora_scale;
        auto & alora_disabled_id = batch.alora_disabled_id;

        // next, batch any pending prompts without exceeding n_batch
        if (params_base.cont_batching || batch.size() == 0) {
            bool add_ok = true; // false means the batch is full, skip remaining slots

            iterate(slots, [&](server_slot & slot) {
                if (!add_ok || batch.size() >= n_batch) {
                    return; // batch is full, skip remaining slots
                }

                if (!slot.is_processing()) {
                    return;
                }

                // check if we can batch this slot with the previous one
                if (slot_batched && !slot_batched->can_batch_with(slot)) {
                    return;
                }

                // check if this is a child slot
                if (slot.state == SLOT_STATE_WAIT_OTHER) {
                    SLT_DBG(slot, "%s", "waiting for parent slot to complete\n");
                    return;
                }

                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_STARTED) {
                    const auto & input_tokens = slot.task->tokens;
                    const bool prompt_just_started = slot.state == SLOT_STATE_STARTED;

                    // used to determine the number of tokens added to the batch for the current slot
                    const auto n_tokens_prev = batch.size();

                    // TODO: maybe move branch to outside of this loop in the future
                    if (slot.state == SLOT_STATE_STARTED) {
                        slot.stats.update_prompt_start();

                        slot.state = SLOT_STATE_PROCESSING_PROMPT;

                        SLT_TRC(slot, "new prompt, n_ctx_slot = %d, n_keep = %d, task.n_tokens = %d\n",
                                slot.n_ctx, slot.task->params.n_keep, slot.task->n_tokens());

                        // print prompt tokens (for debugging)
                        /*if (1) {
                            // first 16 tokens (avoid flooding logs)
                            for (int i = 0; i < std::min<int>(16, input_tokens.size()); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        } else {
                            // all
                            for (int i = 0; i < (int) input_tokens.size(); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        }*/

                        // keep track how many tokens we can reuse from the previous state
                        int n_past = 0;

                        // empty prompt passed -> release the slot and send empty response
                        if (input_tokens.empty()) {
                            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

                            slot.print_timings();
                            send_final_response(slot);
                            slot.release();

                            return;
                        }

                        // TODO: support memory-less logits computation
                        if (slot.task->need_logits() && !llama_get_memory(ctx_tgt)) {
                            send_error(slot, "the current context does not logits computation. skipping", ERROR_TYPE_SERVER);
                            slot.release();
                            return;
                        }

                        if (!slot.can_split()) {
                            if (slot.task->n_tokens() > n_ubatch) {
                                send_error(slot,
                                           string_format(
                                               "input (%d tokens) is too large to process. increase the physical batch "
                                               "size (current batch size: %d)",
                                               slot.task->n_tokens(), n_ubatch),
                                           ERROR_TYPE_SERVER);
                                slot.release();
                                return;
                            }

                            if (slot.task->n_tokens() > slot.n_ctx) {
                                send_error(
                                    slot,
                                    string_format(
                                        "input (%d tokens) is larger than the max context size (%d tokens). skipping",
                                        slot.task->n_tokens(), slot.n_ctx),
                                    ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }
                        } else {
                            if (slot.task->n_tokens() >= slot.n_ctx) {
                                send_error(slot,
                                           string_format("request (%d tokens) exceeds the available context size (%d "
                                                         "tokens), try increasing it",
                                                         slot.task->n_tokens(), slot.n_ctx),
                                           ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }

                            if (slot.task->params.cache_prompt) {
                                // reuse any previously computed tokens that are common with the new prompt
                                n_past = slot.prompt.tokens.get_common_prefix(input_tokens);
                                slot.n_prompt_tokens_lcp = n_past;
                                if (n_past > 0 && slot.prompt_cache_source == "none") {
                                    slot.prompt_cache_source = "live";
                                }
                                slot.prompt_cache_reason = n_past > 0 ? "candidate" : "no_common_prefix";

                                SLT_TRC(slot, "adaptive cache common prefix=%d slot_tokens=%zu task_tokens=%zu bootstrap=%d\n",
                                        n_past, slot.prompt.tokens.size(), input_tokens.size(), slot.bootstrap_pending);

                                // ===== AUTO-RESTORE: cold/cross-process KV reuse from disk (opt-in) ==========
                                // If the in-memory match (n_past) is POOR and the disk index holds a snapshot
                                // whose persisted tokens are a verified, fingerprint-matching, longer prefix of
                                // this request, restore it INTO the slot and RECOMPUTE n_past so all downstream
                                // machinery runs unchanged - agnostic to HOW the tokens arrived. Gated on the
                                // PER-REQUEST has_media() (not the server-wide has_mtmd) so an --mmproj server
                                // still caches its text-only turns: a no-media prompt has no NULL placeholders,
                                // so get_text_tokens() equals the full token-id prefix and does not trip the
                                // get_tokens() GGML_ASSERT(!has_mtmd); a turn carrying an image (and every turn
                                // after it) is skipped. auto_restore_into_slot byte-verifies tokens + fingerprint
                                // and falls back to a normal prefill on any mismatch/failure (invariants 2/3/4).
                                // media-prefix caching intentionally unsupported: token-ids cannot identify image content.
                                if (auto_cache_enabled()
                                        && slot.task->need_sampling()        // generative only (not embed/rerank)
                                        && !input_tokens.has_media()         // no media in this request
                                        && slot.alora_invocation_start <= 0      // aLoRA caching bound (mirror below)
                                        && are_lora_equal(slot.lora, params_base.lora_adapters)) { // fp captures global LoRA (invariant 3)
                                    // get_text_tokens() (not get_tokens()): media-safe accessor that never asserts
                                    // under has_mtmd and, for this no-media prompt, equals the full token-id prefix.
                                    const llama_tokens req = input_tokens.get_text_tokens();
                                    if (auto cand = auto_index_lookup(req)) {
                                        // auto_restore_into_slot may CLEAR the slot
                                        // (KV seq + prompt.tokens) and then have do_slot_restore FAIL
                                        // (corrupt/short .bin, KV-capacity exceeded, racing LRU eviction
                                        // deleting the file mid-read). In that case the slot tokens are now
                                        // empty. We therefore RECOMPUTE n_past UNCONDITIONALLY after any
                                        // attempt - not only on success - so a cleared-but-failed restore
                                        // falls back to n_past=0 (clean cold prefill) instead of carrying a
                                        // stale n_keep_mem>0 into keep_first() on an empty token vector
                                        // (which would GGML_ASSERT/abort). The recompute is harmless on the
                                        // early-return-before-clear paths (margin/fp/verify rejects): those
                                        // leave prompt.tokens untouched, so the LCP is identical to before.
                                        auto_restore_into_slot(slot, *cand, req, (int) n_past);
                                        n_past = slot.prompt.tokens.get_common_prefix(input_tokens);
                                    }
                                }
                                // ===== end AUTO-RESTORE =====================================================

                                // if there is an alora invoked, don't cache after the invocation start
                                if (slot.alora_invocation_start > 0) {
                                    SLT_DBG(slot, "only caching to alora invocation start (n_past = %d, alora_invocation_start = %d)\n", n_past, slot.alora_invocation_start);
                                    n_past = std::min(n_past, slot.alora_invocation_start - 1);
                                }

                                const auto n_cache_reuse = slot.task->params.n_cache_reuse;

                                const bool can_cache_reuse =
                                    llama_memory_can_shift(llama_get_memory(ctx_tgt)) &&
                                    !slot.prompt.tokens.has_mtmd;

                                if (!can_cache_reuse && n_cache_reuse > 0) {
                                    SLT_WRN(slot, "cache reuse is not supported - ignoring n_cache_reuse = %d\n", n_cache_reuse);
                                }

                                // reuse chunks from the cached prompt by shifting their KV cache in the new position
                                if (can_cache_reuse && n_cache_reuse > 0) {
                                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                                    size_t head_c = n_past; // cache
                                    size_t head_p = n_past; // current prompt

                                    if (mctx) {
                                        // we should never reach this
                                        GGML_ABORT("not supported by multimodal");
                                    }

                                    SLT_DBG(slot, "trying to reuse chunks with size > %d, n_past = %d\n", n_cache_reuse, n_past);

                                    while (head_c < slot.prompt.tokens.size() &&
                                           head_p < input_tokens.size()) {

                                        size_t n_match = 0;
                                        while (head_c + n_match < slot.prompt.tokens.size() &&
                                               head_p + n_match < input_tokens.size()       &&
                                               slot.prompt.tokens[head_c + n_match] == input_tokens[head_p + n_match]) {
                                            n_match++;
                                        }

                                        if (n_match >= (size_t) n_cache_reuse) {
                                            SLT_TRC(slot, "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> [%zu, %zu)\n", n_match, head_c, head_c + n_match, head_p, head_p + n_match);
                                            //for (size_t i = head_p; i < head_p + n_match; i++) {
                                            //    SLT_DBG(slot, "cache token %3zu: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx_tgt, prompt_tokens[i]).c_str());
                                            //}

                                            const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;

                                            slot.mem.seq_rm (slot.id, head_p, head_c);
                                            slot.mem.seq_add(slot.id, head_c, head_c + n_match, kv_shift);

                                            for (size_t i = 0; i < n_match; i++) {
                                                slot.prompt.tokens.set_token(head_p + i, slot.prompt.tokens[head_c + i]);
                                                n_past++;
                                            }

                                            head_c += n_match;
                                            head_p += n_match;
                                        } else {
                                            head_c += 1;
                                        }
                                    }

                                    SLT_DBG(slot, "after context reuse, new n_past = %d\n", n_past);
                                }
                            } else {
                                // if we don't cache the prompt, we have to remove all previous tokens
                                n_past = 0;
                                slot.n_prompt_tokens_lcp = 0;
                                slot.prompt_cache_source = "none";
                                slot.prompt_cache_reason = "disabled";
                                common_speculative_set_state(spec.get(), slot.id, {});
                            }

                            llama_pos pos_next = slot.prompt.tokens.pos_next(n_past);

                            // ref: https://github.com/ggml-org/llama.cpp/pull/24110
                            const bool has_new_tokens = (n_past < slot.task->n_tokens());

                            // the largest pos_min required for a checkpoint to be useful
                            const auto pos_min_thold = std::max(0, pos_next - n_swa - (has_new_tokens ? 0 : 1));

                            // ===== restore-continue (regenerate fast-path) ==============================
                            // A just-restored FULL/recurrent slot receiving the EXACT restored tokens (no
                            // suffix) cannot rewind its memory: the normal path would re-decode into the
                            // already-occupied sequence and crash (the pure-recurrent gate at the relaxed
                            // checkpoint predicate below is FALSE for this case, so just_restored would never
                            // be consumed and [TAG_PROMPT_LOGITS] would decrement n_past and re-decode into
                            // the occupied sequence). Detect that case here, BEFORE the n_past>0 guard, and:
                            //   - if we have the saved next-token logits: emit the first token with NO decode,
                            //     keeping the full restored state, then continue normal autoregression;
                            //   - otherwise: fall back to a SAFE clear-then-reprefill (never crash).
                            // Gated so it is unreachable for non-recurrent models, with-suffix requests,
                            // non-generative slots, and multimodal (already excluded by check_no_mtmd at
                            // save/restore). With-suffix restore reuse is left entirely to the unchanged
                            // relaxed-predicate path below.
                            // NOTE: cache_prompt==false sets n_past=0 above, so this gate (which
                            // requires n_past == task->n_tokens()) is naturally not entered for a
                            // no-cache request; that case safely takes the normal full-clear +
                            // reprefill path (common_context_seq_rm at [p0,-1) empties the restored
                            // sequence first), so regenerate degrades to a cold reprefill, never a crash.
                            // No `n_past < n_ctx` clause: the fast path emits with NO decode so it
                            // needs no free context slot; a full-n_ctx no-suffix restore is handled
                            // here rather than falling through to a zero-token-added crash window.
                            if (slot.just_restored &&
                                ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL &&
                                slot.task->need_sampling() &&
                                slot.alora_invocation_start <= 0 &&
                                n_past == slot.task->n_tokens() &&
                                n_past == (int) slot.prompt.n_tokens()) {

                                slot.just_restored = false; // one-shot consume (this path owns it)

                                if (!slot.restored_logits.empty() && !slot.can_speculate() &&
                                        (slot.task->params.sampling.n_probs == 0 || slot.task->params.post_sampling_probs)) {
                                    // --- fast path: emit first token from saved logits, no decode ---
                                    slot.stats.n_prompt_cached     = n_past; // entire prompt "reused"
                                    slot.stats.n_prompt_processed  = 0;      // prompt_n = 0 => observable reuse signal

                                    // prime the sampler over the full restored prompt (penalties/grammar
                                    // history), exactly as the normal DONE_PROMPT transition (init_sampler) would.
                                    slot.stats.n_gen = 0;
                                    slot.init_sampler();

                                    slot.state   = SLOT_STATE_GENERATING;
                                    slot.i_batch = -1;

                                    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));
                                    const llama_token id = common_sampler_sample_from_logits(
                                            slot.smpl.get(), slot.restored_logits.data(), nv, /*grammar_first=*/false);
                                    slot.restored_logits.clear(); // consumed

                                    common_sampler_accept(slot.smpl.get(), id, true);

                                    // mirror the generation accounting from the normal sample path
                                    // (fork stores timings in slot.stats: t_prompt_last marks the end of
                                    // prompt processing - which here includes the disk restore -, then
                                    // t_gen_last marks the start of generation; the first token is free).
                                    slot.stats.n_gen = 1;
                                    if (slot.stats.is_set()) {
                                        slot.stats.update_prompt_last();
                                        slot.stats.update_gen_last();
                                        metrics.add_prompt(slot.stats.n_prompt_processed, (uint64_t) (slot.stats.t_prompt_ms() * 1000));
                                    }

                                    if (slot.task->params.stream) {
                                        // mirror the normal prompt-start streaming signal exactly so a
                                        // return_progress client still gets its initial 0% progress event
                                        if (slot.task->params.return_progress) {
                                            send_partial_response(slot, {}, true);
                                        } else {
                                            // signal HTTP to send the headers (200 status)
                                            send_partial_response(slot, {}, false, true);
                                        }
                                    }

                                    completion_token_output result;
                                    result.tok  = id;
                                    // inline of the accept_special_token lambda (defined later in this method,
                                    // out of scope here): keep special tokens iff the server allows specials
                                    // or the request explicitly preserves this token.
                                    const bool keep_special =
                                        params_base.special ||
                                        slot.task->params.sampling.preserved_tokens.find(result.tok) !=
                                            slot.task->params.sampling.preserved_tokens.end();
                                    result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, keep_special);
                                    result.prob         = 1.0f;

                                    // First-token logprobs (n_probs>0): the post-sampling variant reads the
                                    // candidate set (cur_p), which common_sampler_sample_from_logits leaves
                                    // populated - so we can serve it exactly as the normal path does. The
                                    // pre-sampling variant reads raw ctx logits at a decode index we bypass
                                    // here; idx=-1 is passed but populate_token_probs() only uses idx in that
                                    // branch, so we restrict the call to post_sampling to stay correct.
                                    if (slot.task->params.sampling.n_probs > 0 && slot.task->params.post_sampling_probs) {
                                        populate_token_probs(slot, result, /*post_sampling=*/true, params_base.special, /*idx=*/-1);
                                    }

                                    if (!process_token(result, slot)) {
                                        slot.print_timings();
                                        send_final_response(slot);
                                        auto_save_on_completion(slot);
                                        slot.release();
                                    }

                                    SLT_INF(slot, "%s", "restore-continue: emitted first token from saved logits (prompt_n=0)\n");
                                    // all prompt-batch building for this slot is skipped: the restored KV
                                    // already holds the full prompt and the first token was just emitted
                                    // (this code runs inside the iterate(slots) lambda of update_slots)
                                    return;
                                }

                                // --- safe fallback: no valid sidecar -> clear restored seq, then reprefill ---
                                // FULL models support full-sequence removal; clearing first guarantees the
                                // subsequent reprefill writes into an EMPTY sequence instead of re-decoding
                                // into the already-occupied restored state (which is the crash being fixed).
                                SLT_WRN(slot, "%s", "restore-continue: saved logits unavailable or incompatible; clearing restored state and re-prefilling\n");
                                slot.prompt_clear();
                                n_past   = 0;
                                pos_next = 0; // mirror the do_reset path; keep the stale full-length value from leaking into the checkpoint-erase loop below
                                // fall through to the normal guard below with an empty sequence (safe)
                            }
                            // ===== end restore-continue =================================================

                            if (n_past > 0 && n_past <= slot.prompt.n_tokens()) {
                                const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                                if (pos_min == -1) {
                                    SLT_ERR(slot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d\n", n_past, (int) slot.prompt.tokens.size(), slot.id, pos_min);
                                    GGML_ABORT("pos_min == -1, but n_past > 0 - should not happen: https://github.com/ggml-org/llama.cpp/pull/13833#discussion_r2116181237");
                                }

                                // when the prompt prefix does not match, print the tokens around the mismatch
                                // this is useful for debugging prompt caching
                                if (slots_debug) {
                                    const int np0 = std::max<int>(n_past - slots_n_diff, 0);
                                    const int np1 = std::min<int>(n_past + slots_n_diff + 2, std::min(slot.prompt.tokens.size(), slot.task->tokens.size()));

                                    std::stringstream ss0;
                                    std::stringstream ss1;

                                    std::stringstream st0;
                                    std::stringstream st1;

                                    ss0 << "old: ... ";
                                    ss1 << "new: ... ";

                                    for (int i = np0; i < np1; i++) {
                                        if (i == n_past) {
                                            ss0 << " | ";
                                            ss1 << " | ";
                                        }

                                        {
                                            const auto token = slot.prompt.tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss0 << piece;
                                            st0 << std::setw(8) << token;
                                        }

                                        {
                                            const auto token = slot.task->tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss1 << piece;
                                            st1 << std::setw(8) << token;
                                        }
                                    }

                                    SLT_WRN(slot, "%s\n", ss0.str().c_str());
                                    SLT_WRN(slot, "%s\n", ss1.str().c_str());

                                    SLT_WRN(slot, "%s\n", st0.str().c_str());
                                    SLT_WRN(slot, "%s\n", st1.str().c_str());
                                }

                                if (pos_min >= pos_min_thold) {
                                    // search for a context checkpoint
                                    // reuse a tail checkpoint (e.g. restored from disk) when genuinely-new tokens follow,
                                    // which supply the required logits so the >=1-token guarantee still holds
                                    const bool slot_was_restored = slot.just_restored; slot.just_restored = false;
                                    const bool has_new_suffix = (size_t) slot.task->n_tokens() > (size_t) n_past;
                                    const auto it = std::find_if(
                                        slot.prompt.checkpoints.rbegin(),
                                        slot.prompt.checkpoints.rend(),
                                        [&](const auto & cur) {
                                            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
                                            SLT_TRC(slot, "checking checkpoint with [%d, %d] against %d...\n", cur->pos_min, cur->pos_max, pos_min_thold);
                                            // workaround for [TAG_CHECKPOINTS_FIX_POS_MIN]
                                            if (cur->pos_max > pos_next) {
                                                return false;
                                            }
                                            // [PR-24003] reuse a tail checkpoint (e.g. restored from disk) when genuinely-new tokens follow,
                                            // which supply the required logits so the >=1-token guarantee still holds
                                            return cur->pos_min == 0 || cur->pos_min < pos_min_thold || (slot_was_restored && has_new_suffix && cur->pos_min == pos_min_thold);
                                        }
                                    );

                                    bool do_reset = it == slot.prompt.checkpoints.rend();

                                    if (!do_reset) {
                                        // PARTIAL_ONLY leaves future attention KV present until suffix removal below.
                                        const auto & checkpoint = **it;
                                        do_reset = !checkpoint.restore_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                                        const auto draft_restore = do_reset ? common_checkpoint_restore::failed :
                                            checkpoint.restore_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                                        if (draft_restore == common_checkpoint_restore::missing_base) {
                                            SLT_WRN(slot, "%s", "checkpoint requires MTP bootstrap; cold fallback until bootstrap is connected\n");
                                        }
                                        do_reset = do_reset || draft_restore != common_checkpoint_restore::restored ||
                                            (spec && !checkpoint.data_spec.empty() &&
                                             !common_speculative_set_state(spec.get(), slot.id, checkpoint.data_spec, checkpoint.pos_max));
                                        if (!do_reset) {
                                            if (checkpoint.data_spec.empty()) { common_speculative_set_state(spec.get(), slot.id, {}); }
                                            pos_next = std::min(pos_next, std::max(checkpoint.pos_min + 1, checkpoint.pos_max));
                                            n_past = std::min(slot.prompt.tokens.size_up_to_pos(pos_next), (size_t) checkpoint.n_tokens);
                                            SLT_TRC(slot, "restored context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_past = %d, size = %.3f MiB)\n",
                                                    checkpoint.pos_min, checkpoint.pos_max, checkpoint.n_tokens, n_past, (float) checkpoint.size() / 1024 / 1024);
                                        } else {
                                            SLT_WRN(slot, "%s", "checkpoint restore failed; clearing target, draft and carry\n");
                                        }
                                    }

                                    if (do_reset) {
                                        if (slot_was_restored) {
                                            SLT_WRN(slot, "%s", "target-only snapshot restore discarded; no valid MTP/checkpoint suffix was available, falling back to cold prefill\n");
                                        }
                                        slot.mem.seq_rm(slot.id, -1, -1);
                                        common_speculative_set_state(spec.get(), slot.id, {});
                                        SLT_TRC(slot, "forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see %s)\n",
                                                "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");
                                        pos_next = 0;
                                        n_past = 0;
                                    }
                                }
                            }

                            {
                                // erase any checkpoints with pos_max > pos_next
                                for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {
                                    const auto & cur = *it;
                                    if (cur->pos_max > pos_next) {
                                        SLT_TRC(slot, "erased invalidated context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_swa = %d, pos_next = %d, size = %.3f MiB)\n", cur->pos_min, cur->pos_max, cur->n_tokens, n_swa, pos_next, (float) cur->size() / 1024 / 1024);
                                        it = slot.prompt.checkpoints.erase(it);
                                    } else {
                                        ++it;
                                    }
                                }
                            }
                        }

                        slot.n_prompt_tokens_planned = n_past;

                        // [TAG_PROMPT_LOGITS]
                        if (n_past == slot.task->n_tokens() && n_past > 0) {
                            SLT_WRN(slot, "need to evaluate at least 1 token for each active slot (n_past = %d, task.n_tokens() = %d)\n", n_past, slot.task->n_tokens());
                            n_past--;
                            SLT_WRN(slot, "n_past was set to %d\n", n_past);
                        }

                        slot.stats.n_prompt_cached    = n_past;
                        slot.stats.n_prompt_processed = 0;

                        metrics.add_prompt_cached(n_past);

                        slot.prompt.tokens.keep_first(n_past);

const int32_t snapkv_prefill_end = slot.task->n_tokens();
                        if (getenv("LLAMA_SNAPKV_DEBUG")) {
                            fprintf(stderr, "SNAPKVDBG server_prefill_hint seq=%d expected_prefill_end=%d\n",
                                    slot.id, snapkv_prefill_end);
                        }
                        llama_set_snapkv_prefill_end(ctx_tgt, slot.id, snapkv_prefill_end);
                        llama_set_snapkv_prefill_end(ctx_dft, slot.id, snapkv_prefill_end);

                        // this is to signal the client that the request has started processing
                        if (slot.task->params.stream) {
                            if (slot.task->params.return_progress) {
                                // send initial 0% progress update if needed
                                send_partial_response(slot, {}, true);
                            } else {
                                // otherwise, for streaming without progress, signal HTTP to send the headers (i.e. 200 status)
                                send_partial_response(slot, {}, false, true);
                            }
                        }
                    } // end of SLOT_STATE_STARTED

                    if (!slot.can_split()) {
                        // cannot fit the prompt in the current batch - will try next iter
                        if (batch.size() + slot.task->n_tokens() > n_batch) {
                            return;
                        }
                    }

                    // note: the prompt timing is advanced in post_decode(), so it does not cover
                    //       the tokens added to the batch below
                    slot.print_timings_pp();

                    // truncate any tokens that are beyond n_past for this slot
                    const llama_pos p0 = slot.prompt.tokens.pos_next();

                    SLT_TRC(slot, "cached n_tokens = %d, memory_seq_rm [%d, end)\n", slot.prompt.n_tokens(), p0);

                    llama_pos planned_p0 = p0;
                    const auto normalize_p0 = [&](llama_pos value) {
                        return value > 0
                                ? slot.prompt.tokens.pos_next(slot.prompt.tokens.size_up_to_pos(value))
                                : value;
                    };
                    const auto seq_rm_result = slot.mem.seq_rm_suffix(
                            slot.id, p0, normalize_p0, planned_p0);
                    if (seq_rm_result == COMMON_MEMORY_SEQ_RM_MUTATION_FAILED) {
                        slot.prompt_reset_after_memory_clear();
                        send_error(slot, "internal prompt-cache rollback failure", ERROR_TYPE_SERVER);
                        slot.release();
                        return;
                    }
                    if (seq_rm_result == COMMON_MEMORY_SEQ_RM_FULL_REPROCESS) {
                        const int32_t lexical_lcp = slot.n_prompt_tokens_lcp;
                        slot.prompt_reset_after_memory_clear();
                        slot.n_prompt_tokens_lcp = lexical_lcp;
                        slot.prompt_cache_reason = "suffix_removal_requires_reprocess";
                        if (!prompt_just_started) {
                            slot.state = SLOT_STATE_STARTED;
                            return;
                        }
                    } else if (prompt_just_started) {
                        const size_t planned_n_past = slot.prompt.tokens.size_up_to_pos(planned_p0);
                        if (int32_t(planned_n_past) < slot.n_prompt_tokens_cache) {
                            slot.n_prompt_tokens_planned = std::min(
                                    slot.n_prompt_tokens_planned, int32_t(planned_n_past));
                        }
                        slot.n_prompt_tokens_cache = int32_t(planned_n_past);
                        slot.prompt.tokens.keep_first(planned_n_past);
                        for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {
                            if ((*it)->pos_max >= planned_p0) {
                                it = slot.prompt.checkpoints.erase(it);
                            } else {
                                ++it;
                            }
                        }
                    }
                    if (slot.n_prompt_tokens_cache > 0) {
                        slot.prompt_cache_reason = "committed";
                    }

                    // Signal streaming clients only after rollback planning and
                    // mutation have completed, so an internal failure is still
                    // returned as an HTTP server error.
                    if (prompt_just_started && slot.task->params.stream) {
                        if (slot.task->params.return_progress) {
                            send_partial_response(slot, {}, true);
                        } else {
                            send_partial_response(slot, {}, false, true);
                        }
                    }

                    // If using an alora, there may be uncached tokens that come
                    // before the invocation sequence. When this happens, the
                    // tokens before the invocation sequence need to be
                    // processed without the adapter in a separate batch, then
                    // the adapter needs to be enabled for the remaining tokens.
                    if (lora_all_alora(slot.lora) && slot.alora_invocation_start - 1 > slot.prompt.n_tokens()) {
                        SLT_DBG(slot, "processing pre-alora tokens without the adapter (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                        const auto & enabled_loras = lora_get_enabled_ids(slot.lora);
                        GGML_ASSERT(enabled_loras.size() == 1);
                        alora_scale = slot.lora[enabled_loras[0]].scale;
                        slot.lora[enabled_loras[0]].scale = 0.0f;
                        alora_disabled_id = enabled_loras[0];
                    }

                    bool do_checkpoint = params_base.n_ctx_checkpoints > 0;

                    // make checkpoints only for completion tasks
                    do_checkpoint = do_checkpoint && slot.task->type == SERVER_TASK_TYPE_COMPLETION;

                    // make a checkpoint of the parts of the memory that cannot be rolled back.
                    // checkpoints are created only if:
                    // - the model does not support partial sequence removal
                    // - the model uses SWA (and we are not using `swa_full`)
                    // - the model supports partial sequence removal but only up to a fixed bound
                    do_checkpoint = do_checkpoint && (
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS ||
                            n_swa > 0);

                    bool has_mtmd = false;

                    // check if we should process the mtmd chunk
                    while (true) {
                        auto cur_token_idx = slot.prompt.n_tokens();
                        if (
                            cur_token_idx >= slot.task->n_tokens() ||
                            input_tokens[cur_token_idx] != LLAMA_TOKEN_NULL // encountered a text token
                        ) {
                            break;
                        }

                        // process the mtmd chunk
                        // note: it submits its own decode, potentially be async
                        //       so the timing is queued and flushed on the next sync
                        metrics_pre_decode();

                        // encode on the worker thread, so we can still handle metrics tasks
                        size_t n_tokens_out = 0;
                        int32_t res = 0;
                        queue_tasks.yield_to_queue([&]() {
                            res = process_mtmd_chunk(slot, slot.mbatch, cur_token_idx, n_tokens_out);
                        });

                        if (res != 0) {
                            SLT_ERR(slot, "failed to process mtmd chunk, res = %d\n", res);
                            send_error(slot, "failed to process mtmd chunk", ERROR_TYPE_SERVER);
                            slot.release();
                            return; // the slot is done, skip it entirely
                        }

                        metrics_queue_prompt(n_tokens_out);
                        slot.stats.n_prompt_processed += n_tokens_out;
                        slot.stats.update_prompt_last();

                        // add the mtmd chunk to cache
                        {
                            const auto & chunk = input_tokens.find_chunk(cur_token_idx);
                            // the chunk is already in the KV cache at this point, so we don't need to keep its data around
                            slot.prompt.tokens.push_back_placeholder(chunk.get());
                        }

                        has_mtmd = true;
                    }

                    const auto & spans = slot.task->params.message_spans;
                    const auto last_user_pos = spans.last_user_message_pos();

                    // add prompt tokens for processing in the current batch
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.size() < n_batch) {
                        // Managed MTP's original-prompt disk hit needs a
                        // persisted state at N-1. If this slot already added
                        // prompt work to the current batch, finish that batch
                        // first; the next pass can save the evaluated prefix
                        // before adding the final token.
                        if (auto_cache_enabled() && ctx_dft && slot.can_speculate() &&
                                slot.prompt.n_tokens() + 1 == slot.task->n_tokens() &&
                                batch.size() > n_tokens_prev) {
                            break;
                        }
                        // get next token to process
                        llama_token cur_tok = input_tokens[slot.prompt.n_tokens()];
                        if (cur_tok == LLAMA_TOKEN_NULL) {
                            break; // end of text chunk
                        }

                        // if this is an alora request with pre-invocation
                        // tokens that are not cached, we need to stop filling
                        // this batch at those pre-invocation tokens.
                        if (alora_scale > 0 && slot.prompt.n_tokens() == slot.alora_invocation_start - 1) {
                            SLT_DBG(slot, "stop prompt batch filling at (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                            break;
                        }

                        // embedding requires all tokens in the batch to be output;
                        // MTP also wants logits at every prompt position so the
                        // streaming hook can mirror t_h_nextn into ctx_dft.
                        if (auto_cache_enabled() && ctx_dft && slot.can_speculate() &&
                                slot.prompt.n_tokens() + 1 == slot.task->n_tokens()) {
                            auto_save_prompt_prefix_checkpoint(slot);
                        }

                        if (!batch.add(slot.id,
                                cur_tok,
                                /* pos       = */ slot.prompt.tokens.pos_next(),
                                /* output    = */ slot.need_embd(),
                                /* is_prompt = */ true)) {
                            add_ok = false;
                            break;
                        }
                        slot.prompt.tokens.push_back(cur_tok);

                        // break at the last user message, or at user messages at least min step past the last checkpoint
                        if (do_checkpoint && spans.is_user_start(slot.prompt.n_tokens())) {
                            const auto pos = slot.prompt.n_tokens();
                            const auto & checkpoints = slot.prompt.checkpoints;

if (pos == last_user_pos || checkpoints.empty() || pos > checkpoints.back()->n_tokens + params_base.checkpoint_min_step) {
                                break;
                            }
                        }

                        // process the last few tokens of the prompt separately in order to allow for a checkpoint to be created.
                        // create checkpoints that many tokens before the end of the prompt:
                        //  - 4 + n_ubatch
                        //  - 4
                        // ref: https://github.com/ggml-org/llama.cpp/pull/20288
                        if (do_checkpoint) {
                            static const int checkpoint_offsets[] = {4 + n_ubatch, 4};

                            bool should_break = false;
                            for (int offset : checkpoint_offsets) {
                                const int n_last = std::min(n_batch, offset);
                                const int32_t alignment = prompt_reuse_alignment();
                                const int64_t checkpoint_boundary = server_prompt_checkpoint_boundary(
                                        slot.task->n_tokens(), n_last, alignment);
                                if (checkpoint_boundary > 0 && slot.prompt.n_tokens() == checkpoint_boundary) {
                                    should_break = true;
                                    break;
                                }
                            }
                            if (should_break) {
                                break;
                            }
                        }
                    }

                    // the number of tokens added to the batch for the current slot
                    const auto n_tokens_cur = batch.size() - n_tokens_prev;

                    const auto n_tokens_start = slot.prompt.n_tokens() - n_tokens_cur;

                    const bool near_prompt_end = slot.task->n_tokens() < slot.prompt.n_tokens() + n_ubatch;

                    const bool is_user_start = spans.is_user_start(n_tokens_start);
                    const bool is_last_user_message = n_tokens_start == last_user_pos;
                    // entire prompt has been processed
                    if (slot.prompt.n_tokens() == slot.task->n_tokens()) {
                        slot.state = SLOT_STATE_DONE_PROMPT;

                        GGML_ASSERT(batch.size() > 0);

                        // extract the logits only for the last token
                        batch.set_output(batch.size() - 1, true);

                        slot.stats.n_gen = 0;
                        slot.i_batch     = batch.size() - 1;

                        slot.init_sampler();
                    } else {
                        // skip ordinary mid-prompt checkpoints, unless the batch starts a user
                        // message or we are near the end of the prompt
                        if (!is_user_start && !near_prompt_end) {
                            do_checkpoint = false;
                        }
                    }

                    const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                    const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);

                    // nothing to checkpoint yet
                    // TODO: is this check needed?
                    if (do_checkpoint && pos_min < 0) {
                        do_checkpoint = false;
                    }

                    // do not checkpoint after mtmd chunks
                    do_checkpoint = do_checkpoint && !has_mtmd;
                    do_checkpoint = do_checkpoint && n_tokens_cur > 0;

                    // no need to create checkpoints that are too close together, unless it's the last user message
                    do_checkpoint = do_checkpoint && (
                            slot.prompt.checkpoints.empty() ||
                            is_last_user_message || near_prompt_end ||
                            n_tokens_start > slot.prompt.checkpoints.back()->n_tokens + params_base.checkpoint_min_step);
                    SLT_DBG(slot, "main/do_checkpoint = %s, pos_min = %d, pos_max = %d\n", do_checkpoint ? "yes" : "no", pos_min, pos_max);

                    // note: we create the checkpoint before calling llama_decode(), so the current batch is not
                    //       yet processed and therefore it is not part of the checkpoint.
                    if (do_checkpoint) {
                        create_checkpoint(slot, n_tokens_cur, pos_min, pos_max);
                    }
                }

                if (!slot_batched) {
                    slot_batched = &slot;
                }
            });
        }
    }

    // returns true = success ; false = retry with smaller batch size
    // throw std::runtime_error on fatal error
    bool decode(int32_t & n_batch, int32_t off, llama_batch & batch_view) {
        SRV_DBG("n_batch (effective) = %d, off = %d\n", n_batch, off);

        metrics_pre_decode();

        if (batch.size() == 0) {
            SRV_WRN("%s", "no tokens to decode\n");

            if (++n_empty_consecutive > 3) {
                GGML_ABORT("fatal error - please provide logs and repro in %s\n", "https://github.com/ggml-org/llama.cpp/pull/20277");
            }

            return true; // nothing to decode
        } else {
            n_empty_consecutive = 0;
        }

        // TODO @ngxson : dft model may have different n_embd than the tgt model, so we check & reject if that's the case
        // this case is not currently used by any models, but may need to be supported in the future
        if (spec && batch.has_embd) {
            if (llama_model_n_embd_inp(model_dft) != llama_model_n_embd_inp(model_tgt)) {
                SRV_ERR("%s", "unsupported batch.has_embd + spec case\n");
                throw std::runtime_error("unsupported batch.has_embd + spec case");
            }
        }

        bool has_output = false;
        for (int i = off; i < off + batch_view.n_tokens; ++i) {
            has_output |= batch.tokens[i].output;
        }

        // yield to the queue, so we can still handle metrics tasks while decoding
        // note: the sync is done here too, so that the wait is also covered by the yield
        int ret = 0;
        queue_tasks.yield_to_queue([&]() {
            ret = llama_decode(ctx_tgt, batch_view);
            if (ret == 0 && has_output) {
                llama_synchronize(ctx_tgt);
            }
        });

        if (ret != 0) {
            std::string err;

            if (n_batch == 1 && ret == 1) {
                // TODO: try to terminate only the largest active slot/sequence and continue with the rest
                //       need to remove the tokens from the current batch too
                err = "Context size has been exceeded.";
            }

            if (ret == -1) {
                err = "Invalid input batch.";
            }

            if (ret < -1) {
                // TODO: update slot state based on llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
                err = "Compute error.";
            }

            // TODO: handle ret == 2 (abort) when we start aborting

            if (err.empty() && !try_clear_idle_slots()) {
                const int32_t n_batch_next = n_batch / 2;
                for (const auto & slot : slots) {
                    if (!slot.spec_i_batch.empty() &&
                            slot.spec_i_batch.front() >= off &&
                            slot.spec_i_batch.front() < off + n_batch_next &&
                            slot.spec_i_batch.back() >= off + n_batch_next) {
                        err = "Context size has been exceeded.";
                        break;
                    }
                }
                n_batch = n_batch_next;
            }

            if (!err.empty()) {
                SRV_ERR("%s off = %d, n_batch = %d, ret = %d\n", err.c_str(), off, n_batch, ret);

                for (auto & slot : slots) {
                    if (slot.is_processing()) {
                        send_error(slot, err);
                        slot.release();

                        // note: it's complicated to keep track of how much of the current batch has been
                        //       processed before the error occurred, so we simply clear the entire context
                        slot.prompt_clear();
                    }
                }

                // stop, do not retry with smaller batch size
                throw std::runtime_error(err);
            }

            SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size, off = %d, n_batch = %d, ret = %d\n", off, n_batch, ret);

            return false; // retry with the updated n_batch
        } else {
            // success, apply batch metrics
            metrics_post_decode(off, batch_view.n_tokens, has_output);
        }

        // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
        //       for now, always re-evaluate for simplicity
        //       ref: https://github.com/ggml-org/llama.cpp/pull/22728#issuecomment-4400925384
        bool bootstrapped = false;
        if (spec) {
            for (auto & slot : slots) {
                if (!slot.bootstrap_pending || !slot.is_processing()) {
                    continue;
                }
                const uint64_t decode_id = llama_get_nextn_decode_id(ctx_tgt);
                if (!common_speculative_bootstrap(spec.get(), batch_view, decode_id)) {
                    SLT_ERR(slot, "%s", "target-only cache hit could not bootstrap MTP from the decoded suffix\n");
                    slot.bootstrap_pending = false;
                    slot.prompt_clear();
                    throw std::runtime_error("failed to bootstrap MTP from target-only prompt cache");
                }
                slot.bootstrap_pending = false;
                slot.just_restored = false;
                bootstrapped = true;
                SLT_INF(slot, "target-only MTP bootstrap accepted: decoded_suffix=%d\n", batch_view.n_tokens);
                break;
            }
        }

        if (spec && !bootstrapped) {
            bool ok = true;
            queue_tasks.yield_to_queue([&]() {
                ok = common_speculative_process(spec.get(), batch_view);
            });

            if (!ok) {
                SRV_ERR("%s", "failed to process speculative batch\n");

                // TODO: handle error
                for (auto & slot : slots) {
                    if (slot.is_processing()) {
                        slot.prompt_clear();
                    }
                }
                throw std::runtime_error("failed to process speculative batch");
            }
        }

        // handle `n_cmpl > 1` tasks - when the main prompt is processed, activate all child tasks too
        for (auto & slot : slots) {
            if (slot.state == SLOT_STATE_DONE_PROMPT && slot.task->is_parent()) {
                std::vector<server_slot *> children;
                for (auto & other : slots) {
                    if (other.state == SLOT_STATE_WAIT_OTHER && slot.task->id == other.task->id_parent) {
                        children.push_back(&other);
                    }
                }

                // all children slots should already launched by launch_slots_with_parent_task()
                // copy state to the child slots
                for (auto & child : children) {
                    SLT_TRC(slot, " - copying state to child %d\n", child->id);

                    GGML_ASSERT(child->state == SLOT_STATE_WAIT_OTHER);

                    slot.copy_state_to(*child);
                    child->state = SLOT_STATE_DONE_PROMPT;
                }
            }
        }

        return true;
    }

    void post_decode(int32_t n_batch_tokens, int32_t off, llama_batch & batch_view) {
        // for checking if a given batch index is inside batch_view
        auto is_inside_view = [&](int32_t idx) {
            return idx >= off && idx < off + n_batch_tokens;
        };

        // TODO @ngxson : it's tricky to make sub-batch compatible with common_sampler_sample_and_accept_n,
        // so for now we will throw an error in this case: https://github.com/ggml-org/llama.cpp/issues/24840
        iterate(slots, [&](server_slot & slot) {
            for (auto & i : slot.spec_i_batch) {
                if (!is_inside_view(i)) {
                    throw std::runtime_error(string_format("speculative batch index %d is not inside the current sub-batch [%d, %d)", i, off, off + n_batch_tokens));
                }
            }
        });

        auto accept_special_token = [&](server_slot & slot, llama_token token) {
            return params_base.special ||
                slot.task->params.sampling.preserved_tokens.find(token) != slot.task->params.sampling.preserved_tokens.end();
        };

        iterate(slots, [&](server_slot & slot) {
            // optionally send prompt processing progress
            if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->params.stream && slot.task->params.return_progress) {
                    send_partial_response(slot, {}, true);
                }
            }

            if (!is_inside_view(slot.i_batch)) {
                // the required token not in this sub-batch, skip
                return;
            }

            if (slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->type == SERVER_TASK_TYPE_EMBEDDING) {
                    // prompt evaluated for embedding
                    send_embedding(slot, batch_view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                if (slot.task->type == SERVER_TASK_TYPE_RERANK) {
                    send_rerank(slot, batch_view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                GGML_ASSERT(slot.task->need_sampling());

                // prompt evaluated for next-token prediction
                auto_capture_prompt_logits(slot, slot.i_batch - off);
                auto_save_prompt_checkpoint(slot);
                slot.state = SLOT_STATE_GENERATING;

                if (slot.can_speculate()) {
                    common_speculative_begin(spec.get(), slot.id, slot.prompt.tokens.get_text_tokens());
                }
            } else if (slot.state != SLOT_STATE_GENERATING) {
                return;
            }

            if (slot.can_speculate() && !slot.spec_draft.empty()) {
                return; // sample using speculative decoding
            }

            // shifted according to the current sub-batch
            const int tok_idx = slot.i_batch - off;

            llama_token id;
            {
                scoped_timer timer(t_sampl, n_sampl);
                id = common_sampler_sample(slot.smpl.get(), slot.ctx_tgt, tok_idx);
            }

            // capture this slot's last-token full-vocab logits for a possible disk
            // save. Cost: one ~n_vocab*4-byte copy per decoded token, incurred ONLY on
            // FULL/recurrent models AND only when slot saving is enabled (--slot-save-path set);
            // attention models and servers without slot-save pay nothing at all. The copy is
            // unavoidable for correctness: ctx logits are overwritten by the next slot's decode,
            // so a lazy read at SLOT_SAVE would be wrong under --parallel>1. Captured per-slot
            // from this slot's own tok_idx so it is correct for any N/interleave (never read
            // from the shared ctx at save time). common_sampler_sample already synchronized the
            // context above, so llama_get_logits_ith is valid here.
            if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL && !params_base.slot_save_path.empty()) {
                const float * lg = llama_get_logits_ith(slot.ctx_tgt, tok_idx);
                if (lg) {
                    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));
                    slot.logits_last.assign(lg, lg + nv);
                    // Stamp the capture with the token count of the state it corresponds to.
                    // At this point process_token() has NOT yet appended the just-sampled token,
                    // so prompt.tokens.size() is exactly the length whose final token produced
                    // these logits - i.e. it matches the token_count a SLOT_SAVE would record.
                    slot.logits_last_n_tokens = (int32_t) slot.prompt.tokens.size();
                } else {
                    // capture failed -> invalidate so a later save never serializes stale logits
                    slot.logits_last.clear();
                    slot.logits_last_n_tokens = -1;
                }
            }

            slot.i_batch = -1;

            const auto accept_info = common_sampler_accept_with_info(slot.smpl.get(), id, true);
            (void) handle_loop_guard_accept(slot, accept_info);

            // here we have synchronized the llama_context (due to the sampling above), so we can do time measurement
            const int64_t t_now = ggml_time_us();

            slot.stats.n_gen += 1;

            if (slot.stats.n_gen == 1) {
                slot.stats.update_prompt_last();
                slot.t_print_last = t_now;
                slot.n_gen_last = 0;
            }

            slot.stats.update_gen_last();

            completion_token_output result;
            result.tok          = id;
            result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
            result.prob         = 1.0f; // TODO: set it here instead of doing inside populate_token_probs

            if (slot.task->params.sampling.n_probs > 0) {
                populate_token_probs(slot, result, slot.task->params.post_sampling_probs, params_base.special, tok_idx);
            }

            if (!process_token(result, slot)) {
                // release slot because of stop condition
                slot.print_timings();
                send_final_response(slot);
                auto_save_on_completion(slot);
                slot.release();

                return;
            }

            slot.print_timings_tg();
        });

        // speculative decoding - main model sample and accept
        iterate(slots, [&](server_slot & slot) {
            if (slot.state != SLOT_STATE_GENERATING || !slot.can_speculate() ||
                    slot.spec_draft.empty() || slot.spec_i_batch.empty()) {
                return;
            }

            // save the original draft size
            const size_t n_draft = slot.spec_draft.size();

            GGML_ASSERT(n_draft > 0);

            // verify and try to accept the draft
            const int64_t t_verify_start = ggml_time_us();
            {
                common_sampler_ptr smpl_save(common_sampler_clone(slot.smpl.get()));

                // The loop guard observes sampler acceptance. A checkpoint restore discards
                // every speculative acceptance from this pass, so its token tail, telemetry,
                // and any force-close/stop action must be restored with the sampler.
                const server_loop_guard loop_guard_save = slot.loop_guard;
                const int32_t loop_guard_interventions_save = slot.loop_guard_interventions;
                const bool loop_guard_triggered_save = slot.loop_guard_triggered;
                const std::string loop_guard_action_save = slot.loop_guard_action;
                const std::string loop_guard_reason_save = slot.loop_guard_reason;
                const int32_t reasoning_output_tokens_save = slot.reasoning_output_tokens;
                const int32_t visible_output_tokens_save = slot.visible_output_tokens;
                const bool has_next_token_save = slot.has_next_token;
                const stop_type stop_save = slot.stop;
                const std::string stop_detail_save = slot.stop_detail;

                GGML_ASSERT(slot.spec_i_batch.size() == n_draft + 1);
                const auto & synth_probs = common_speculative_get_synth_probs(spec.get());
                const bool can_rollback =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_PART ||
                    (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && n_draft <= llama_n_rs_seq(ctx_tgt));
                auto accepted = synth_probs.empty()
                    ? (can_rollback && slot.task->params.sampling.temp > 0.0f &&
                       slot.spec_dists.size() == slot.spec_draft.size()
                        ? common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft, slot.spec_dists)
                        : common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft))
                    : server_sample_and_accept_synth(
                            slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft,
                            synth_probs, slot.spec_synth_rng, slot.spec_is_replay);
                slot.spec_i_batch.clear();

                GGML_ASSERT(accepted.size() >= 1);

                const uint32_t n_rollback = slot.spec_draft.size() + 1 - accepted.size();

                const bool use_ckpt_tgt = server_speculative_rollback_requires_checkpoint(
                        ctx_tgt_seq_rm_type, common_context_seq_rm_max_rollback(ctx_tgt), n_rollback);

                // check for partial draft acceptance
                if (n_rollback > 0) {
                    if (use_ckpt_tgt) {
                        if (trace > 0) {
                            SLT_INF(slot, "accepted %2zu/%2zu draft tokens (restore checkpoint)\n", accepted.size() - 1, slot.spec_draft.size());
                        }

                        // partial acceptance is not supported by the context -> truncate the draft and restore the state
                        slot.spec_is_replay = true;
                        slot.spec_draft = std::move(accepted);
                        slot.spec_dists.clear();

                        const auto & ckpt = slot.spec_ckpt;

                        SLT_DBG(slot, "restoring speculative checkpoint (pos_min = %d, pos_max = %d, size = %zu)\n", ckpt.pos_min, ckpt.pos_max, ckpt.size());

                        if (!restore_checkpoint_transaction(
                                    slot, ckpt, slot.ctx_tgt, slot.ctx_dft,
                                    true, slot.ctx_dft != nullptr, true)) {
                            SLT_ERR(slot, "%s", "failed to restore speculative checkpoint transaction\n");
                            slot.release();
                            return;
                        }

                        slot.mem.seq_rm(slot.id, ckpt.pos_max + 1, -1);

                        slot.prompt.tokens.keep_first(ckpt.n_tokens);
                        common_sampler_copy(smpl_save.get(), slot.smpl.get());
                        slot.loop_guard = loop_guard_save;
                        slot.loop_guard_interventions = loop_guard_interventions_save;
                        slot.loop_guard_triggered = loop_guard_triggered_save;
                        slot.loop_guard_action = loop_guard_action_save;
                        slot.loop_guard_reason = loop_guard_reason_save;
                        slot.reasoning_output_tokens = reasoning_output_tokens_save;
                        slot.visible_output_tokens = visible_output_tokens_save;
                        slot.has_next_token = has_next_token_save;
                        slot.stop = stop_save;
                        slot.stop_detail = stop_detail_save;

                        return;
                    }
                }

                if (trace > 0) {
                    SLT_INF(slot, "accepted %2zu/%2zu draft tokens\n", accepted.size() - 1, n_draft);
                }

                common_speculative_accept(spec.get(), slot.id, accepted.size() - 1);

                slot.spec_draft = std::move(accepted);
                slot.spec_dists.clear();
            }

const int64_t t_now = ggml_time_us();
            const float verify_ms = (float) (t_now - t_verify_start) / 1000.0f;

            const auto ids = std::move(slot.spec_draft);

            size_t n_accepted = ids.size() - 1;
            if (slot.spec_is_replay && n_accepted > 0) {
                n_accepted--;
            }
            slot.spec_is_replay = false;

            slot.stats.update_gen_last();

            // update how many tokens out of those tested were accepted
            slot.stats.n_draft_accepted += n_accepted;
            slot.stats.n_draft_verif_steps += 1;

if (slot.uses_dflash() && slot.adaptive_dm.dm_adaptive && slot.adaptive_cycle_start_us > 0) {
                const int n_accepted = std::max(0, (int) ids.size() - 1);
                const float cycle_ms = std::max(0.001f,
                        (float) (t_now - slot.adaptive_cycle_start_us) / 1000.0f);
                slot.adaptive_dm.observe_profit_acceptance((int) n_draft, n_accepted);
                slot.adaptive_dm.observe_profit_timing(
                        slot.adaptive_requested_n_max > 0
                            ? slot.adaptive_requested_n_max
                            : (int) n_draft,
                        (int) n_draft,
                        n_accepted,
                        slot.adaptive_draft_ms,
                        verify_ms,
                        0.0f,
                        cycle_ms);
                apply_adaptive_profit_decision(slot);
            }

            auto & n_accepted_per_pos = slot.n_accepted_per_pos;
            if (n_accepted_per_pos.empty()) {
                n_accepted_per_pos.resize(common_speculative_n_max(spec.get()), 0);
            }
            for (size_t i = 0; i < n_accepted && i < n_accepted_per_pos.size(); ++i) {
                n_accepted_per_pos[i]++;
            }

            // add accepted tokens to the prompt
            slot.prompt.tokens.keep_first(slot.prompt.n_tokens() - n_draft);
            slot.prompt.tokens.insert({ids.begin(), ids.end() - 1});

            slot.sampled = ids.back(); // last accepted token
            SLT_DBG(slot, "add accepted tokens: sampled=%d, ids.size=%zu, n_draft=%zu\n", slot.sampled, ids.size(), n_draft);

            slot.mem.seq_rm(slot.id, slot.prompt.tokens.pos_next(), -1);

            for (size_t i = 0; i < ids.size(); ++i) {
                completion_token_output result;

                result.tok          = ids[i];
                result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                result.prob         = 1.0f; // set later

                // TODO: set result.probs

                slot.stats.n_gen += 1;

                if (!process_token(result, slot)) {
                    slot.print_timings();
                    send_final_response(slot);
                    auto_save_on_completion(slot);
                    slot.release();

                    return;
                }
            }

            slot.print_timings_tg();

            SLT_DBG(slot, "accepted %d/%d draft tokens, new n_tokens = %d\n", (int) n_accepted, (int) n_draft, slot.prompt.n_tokens());
        });
    }

    // context size of a single slot, capped by --kv-unified-per-slot and by the training context of the model
    int active_n_ctx_slot() const {
        int res = llama_n_ctx_seq(ctx_tgt);

        if (params_base.kv_unified_per_slot > 0) {
            res = std::min(res, params_base.kv_unified_per_slot);
        }

        return std::min(res, llama_model_n_ctx_train(model_tgt));
    }

    int n_ctx_slot() const {
        if (common_context_is_adaptive(params_base) && adaptive_long_ctx > 0) {
            return adaptive_max_ctx();
        }
        return active_n_ctx_slot();
    }

    server_response_reader get_response_reader() {
        return server_response_reader(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
    }

    //
    // metrics helpers
    //

    // call before submitting a decode, so that the queued prompt stats can be timed
    void metrics_pre_decode() {
        t_decode_start = ggml_time_us();
    }

    // the batch is submitted, but its compute may not be done yet
    void metrics_queue_prompt(uint64_t n_tokens) {
        if (n_tokens == 0) {
            return;
        }
        if (n_prompt_queued == 0) {
            t_prompt_start = t_decode_start;
        }
        n_prompt_queued += n_tokens;
    }

    // call only after the context is synchronized, otherwise the time is meaningless
    void metrics_flush_prompt() {
        if (n_prompt_queued == 0) {
            return;
        }
        metrics.add_prompt(n_prompt_queued, ggml_time_us() - t_prompt_start);
        n_prompt_queued = 0;
    }

    // has_output is computed by the caller, which also already synchronized the context if it is set
    void metrics_post_decode(int32_t off, int32_t n_tokens, bool has_output) {
        metrics.n_decode++;
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                metrics.n_busy_slots++;
            }
            metrics.n_tokens_max = std::max(metrics.n_tokens_max, (uint64_t) slot.prompt.n_tokens());
        }

        // apply enqueued prompt tokens stats
        // note: a slot can be released before we get here, which clears its stats
        //       the tokens were still computed, counted in the global metrics, not in slot
        uint64_t n_prompt_tokens = 0;

        for (int i = off; i < off + n_tokens; ++i) {
            const auto & t = batch.tokens[i];

            if (!t.is_prompt) {
                continue; // generated tokens are handled after sampling
            }

            n_prompt_tokens++;

            auto & slot = slots[t.id_slot];
            if (slot.stats.is_set()) {
                slot.stats.n_prompt_processed++;
            }
        }

        metrics_queue_prompt(n_prompt_tokens);

        if (has_output) {
            // the context is already synchronized, so the timings are correct
            metrics_flush_prompt();
        }

        // advance the prompt timing of the slots that had tokens in this batch
        // note: a second pass, it must run after the sync to reflect the compute
        const int64_t t_now = ggml_time_us();
        for (int i = off; i < off + n_tokens; ++i) {
            const auto & t = batch.tokens[i];
            auto & slot = slots[t.id_slot];
            if (t.is_prompt && slot.stats.is_set()) {
                slot.stats.set_prompt_last(t_now);
            }
        }
    }

    // flush any queued prompt metrics if all slots are now idle
    void metrics_flush_idle() {
        if (n_prompt_queued == 0) {
            return;
        }

        llama_synchronize(ctx_tgt);
        metrics_flush_prompt();
    }

    void metrics_on_prediction(const server_slot & slot) {
        const uint64_t t_us    = slot.stats.t_gen_us();
        const uint64_t n       = slot.stats.n_gen;
        const uint64_t n_steps = slot.stats.n_gen_steps();

        metrics.predict       .add(n, n_steps, t_us);
        metrics.predict_bucket.add(n, n_steps, t_us);

        metrics.n_draft_tokens      += slot.stats.n_draft_tokens;
        metrics.n_draft_accepted    += slot.stats.n_draft_accepted;
        metrics.n_draft_verif_steps += slot.stats.n_draft_verif_steps;

        auto & dst = metrics.n_accepted_per_pos;
        const auto & src = slot.n_accepted_per_pos;

        if (dst.size() < src.size()) {
            dst.resize(src.size(), 0);
        }
        for (size_t i = 0; i < src.size(); i++) {
            dst[i] += src[i];
        }
    }
};

//
// server_context (public API)
//

server_context::server_context() : impl(new server_context_impl()) {}
server_context::~server_context() = default;

bool server_context::load_model(common_params & params) {
    return impl->load_model(params);
}

void server_context::start_loop() {
    auto & params = impl->params_base;
    impl->queue_tasks.start_loop(params.sleep_idle_seconds * 1000);
    impl->gpu_power.shutdown();
}

void server_context::terminate() {
    impl->queue_tasks.terminate();
}

llama_context * server_context::get_llama_context() const {
    return impl->ctx_tgt;
}

server_response_reader server_context::get_response_reader() {
    return impl->get_response_reader();
}

server_context_meta server_context::get_meta() const {
    auto bos_id = llama_vocab_bos(impl->vocab);
    auto eos_id = llama_vocab_eos(impl->vocab);
    auto bos_token_str = bos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, bos_id, true) : "";
    auto eos_token_str = eos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, eos_id, true) : "";

    const char * ftype_name = llama_ftype_name(llama_model_ftype(impl->model_tgt));

    return server_context_meta {
        /* build_info             */ std::string(llama_build_info()),
        /* model_name             */ impl->model_name,
        /* model_aliases          */ impl->model_aliases,
        /* model_tags             */ impl->model_tags,
        /* model_path             */ impl->params_base.model.path,
        /* has_mtmd               */ impl->mctx != nullptr,
        /* has_inp_image          */ impl->chat_params.allow_image,
        /* has_inp_audio          */ impl->chat_params.allow_audio,
        /* has_inp_video          */ impl->chat_params.allow_video,
        /* json_ui_settings       */ impl->json_ui_settings,
        /* slot_n_ctx             */ impl->n_ctx_slot(),
        /* pooling_type           */ llama_pooling_type(impl->ctx_tgt),

        /* chat_params            */ impl->chat_params,
        /* chat_template_caps     */ common_chat_templates_get_caps(impl->chat_params.tmpls.get()),

        /* bos_token_str          */ bos_token_str,
        /* eos_token_str          */ eos_token_str,
        /* fim_pre_token          */ llama_vocab_fim_pre(impl->vocab),
        /* fim_sub_token          */ llama_vocab_fim_suf(impl->vocab),
        /* fim_mid_token          */ llama_vocab_fim_mid(impl->vocab),
        /* fim_pad_token          */ llama_vocab_fim_pad(impl->vocab),
        /* fim_rep_token          */ llama_vocab_fim_rep(impl->vocab),
        /* fim_sep_token          */ llama_vocab_fim_sep(impl->vocab),

        /* logit_bias_eog         */ impl->params_base.sampling.logit_bias_eog,

        /* model_vocab_type       */ llama_vocab_type(impl->vocab),
        /* model_vocab_n_tokens   */ llama_vocab_n_tokens(impl->vocab),
        /* model_n_ctx_train      */ llama_model_n_ctx_train(impl->model_tgt),
        /* model_n_embd_inp       */ llama_model_n_embd(impl->model_tgt),
        /* model_n_params         */ llama_model_n_params(impl->model_tgt),
        /* model_size             */ llama_model_size(impl->model_tgt),
        /* model_ftype            */ ftype_name,
    };
}

server_context_adaptive_status server_context::get_adaptive_status() const {
    return impl->get_adaptive_status();
}

// generator-like API for HTTP response generation
// may have bypass_sleep = true if the task does not use ctx_server
struct server_res_generator : server_res_spipe {
    server_response_reader rd;
    server_res_generator(server_queue & queue_tasks, server_response & queue_results, int sleep_idle_seconds, bool bypass_sleep = false)
            : rd(queue_tasks, queue_results, HTTP_POLLING_SECONDS) {
        // fast path in case sleeping is disabled
        bypass_sleep |= sleep_idle_seconds < 0;
        if (!bypass_sleep) {
            queue_tasks.wait_until_no_sleep();
        }
    }
    void ok(const json & response_data) {
        status = 200;
        data = safe_json_to_str(response_data);
    }
    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data = safe_json_to_str({{ "error", error_data }});
    }
};

void server_context::set_state_callback(server_state_callback_t callback) {
    impl->callback_state = std::move(callback);
}

//
// server_routes
//

std::unique_ptr<server_res_generator> server_routes::handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type) {
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    auto res = create_response();
    auto completion_id = gen_chatcmplid();
    auto & rd = res->rd;
    auto & params = this->params;

    res->set_req(&req); // will also set spipe if needed

    int32_t sse_ping_interval = params.sse_ping_interval;

    try {
        std::vector<server_task> tasks;

        const auto & prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        if (!params.path_prompts_log_dir.empty()) {
            const auto file_path = std::filesystem::path(params.path_prompts_log_dir) / string_format("%012" PRId64 ".txt", ggml_time_ms());
            std::ofstream f(file_path);
            if (f) {
                f << (prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());
            } else {
                SRV_ERR("failed to create %s\n", file_path.string().c_str());
            }
        }

        // process prompt
        std::vector<server_tokens> inputs;

        if (res_type != TASK_RESPONSE_TYPE_NONE && ctx_server.mctx != nullptr) {
            // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
            inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files, ctx_server.init_opt));
        } else {
            // Everything else, including multimodal completions.
            inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true, ctx_server.init_opt);
        }

        // tasks.reserve(inputs.size()); // TODO: this is inaccurate due to child tasks

        // message delimiters for checkpointing
        json delims = json_value(data, "message_delimiters", json::array());
        auto delimiters = common_chat_msg_delimiters_parse(delims);
        delimiters.tokenize(ctx_server.vocab);

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = rd.get_new_id();

            task.tokens = std::move(inputs[i]);
            task.params = server_schema::eval_llama_cmpl_schema(
                    ctx_server.vocab,
                    params,
                    meta->logit_bias_eog,
                    data);

            task.params.message_spans = task.tokens.find_message_spans(delimiters);

            task.id_slot = json_value(data, "id_slot", -1);
            sse_ping_interval = task.params.sse_ping_interval;

            // OAI-compat
            task.params.res_type          = res_type;
            task.params.oaicompat_cmpl_id = completion_id;
            task.params.oaicompat_model   = meta->model_name;

            // prepare child tasks
            if (task.params.n_cmpl > 1) {
                int n_children = task.params.n_cmpl - 1;
                for (int j = 0; j < n_children; j++) {
                    task.add_child(task.id, rd.get_new_id());
                }
            }

            tasks.push_back(std::move(task));
        }

        rd.post_tasks(std::move(tasks));
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool stream = json_value(data, "stream", false);

    if (!stream) {
        // non-stream, wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            json arr = json::array();
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(res.get()) != nullptr);
                arr.push_back(res->to_json());
            }
            GGML_ASSERT(!arr.empty() && "empty results");
            if (arr.size() == 1) {
                // if single request, return single object instead of array
                res->ok(arr[0]);
            } else if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT || res_type == TASK_RESPONSE_TYPE_OAI_CMPL) {
                // if multiple results in OAI format, we need to re-format them
                json & choices = arr[0]["choices"];
                for (size_t i = 1; i < arr.size(); i++) {
                    choices.push_back(std::move(arr[i]["choices"][0]));
                }
                res->ok(arr[0]);
            } else {
                // multi-results, non-OAI compat
                res->ok(arr);
            }
        }
    } else {
        // in streaming mode, the first error must be treated as non-stream response
        // this is to match the OAI API behavior
        // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
        auto first_result = rd.next(req.should_stop);
        if (first_result == nullptr) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }

        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr ||
            dynamic_cast<server_task_result_cmpl_final*>  (first_result.get()) != nullptr
        );

        // next responses are streamed
        // to be sent immediately
        json first_result_json = first_result->to_json();
        if (first_result_json.is_null()) {
            res->data = ""; // simply send HTTP headers and status code
        } else if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
            res->data = format_anthropic_sse(first_result_json);
        } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
            res->data = format_oai_resp_sse(first_result_json);
        } else {
            res->data = format_oai_sse(first_result_json);
        }
        res->status = 200;
        res->content_type = "text/event-stream";
        res->set_next([res_this = res.get(), res_type, sse_ping_interval](std::string & output) -> bool {
            static auto format_error = [](task_response_type res_type, const json & res_json) {
                if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                    return format_anthropic_sse({
                        {"event", "error"},
                        {"data", res_json},
                    });
                } else {
                    return format_oai_sse(json {{ "error", res_json }});
                }
            };

            auto effective_should_stop = [&res_this]() {
                return res_this->should_stop();
            };

            try {
                if (effective_should_stop()) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    return false; // should_stop condition met
                }

                if (!res_this->data.empty()) {
                    // flush the first chunk
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }

                server_response_reader & rd = res_this->rd;

                // check if there is more data
                if (!rd.has_next()) {
                    switch (res_type) {
                        case TASK_RESPONSE_TYPE_NONE:
                        case TASK_RESPONSE_TYPE_OAI_RESP:
                        case TASK_RESPONSE_TYPE_ANTHROPIC:
                            output = "";
                            break;

                        default:
                            output = "data: [DONE]\n\n";
                            break;
                    }
                    SRV_DBG("%s", "all results received, terminating stream\n");
                    return false; // no more data, terminate
                }

                // receive subsequent results
                bool timeout = false;
                int64_t start_time = ggml_time_ms();
                auto result = rd.next([&timeout, &start_time, sse_ping_interval, &effective_should_stop]() {
                    if (effective_should_stop()) {
                        return true; // should_stop condition met
                    } else if (sse_ping_interval > 0 && ggml_time_ms() - start_time > (int64_t)sse_ping_interval * 1000) {
                        timeout = true;
                        return true; // timeout
                    }
                    return false;
                });

                if (timeout) {
                    // some clients may time out (e.g. undici) will time out if no data is received for a while, so we need to send a ping to keep the connection alive
                    SRV_DBG("%s", "sending SSE ping\n");
                    output = ":\n\n";
                    return true;
                }

                if (result == nullptr) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    GGML_ASSERT(effective_should_stop());
                    return false; // should_stop condition met
                }

                // send the results
                if (result->is_error()) {
                    json res_json = result->to_json();
                    output = format_error(res_type, res_json);
                    SRV_DBG("%s", "error received during streaming, terminating stream\n");
                    return false; // terminate on error
                } else {
                    GGML_ASSERT(
                        dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
                        || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                    );
                    json res_json = result->to_json();
                    // A progress-only partial and the explicit begin marker can
                    // intentionally have no wire payload.  Do not serialize
                    // that internal sentinel as an OpenAI `data: null` event;
                    // keep the stream open so the next real result is sent.
                    if (res_json.is_null()) {
                        output.clear();
                        return true;
                    } else if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                        output = format_anthropic_sse(res_json);
                    } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                        output = format_oai_resp_sse(res_json);
                    } else {
                        output = format_oai_sse(res_json);
                    }
                }

                // has next data, continue
                return true;

            } catch (const std::exception & e) {
                json error_json = format_error_response(e.what(), ERROR_TYPE_SERVER);
                output = format_error(res_type, error_json);

                // terminate on exception
                return false;
            }
        });
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::create_response(bool bypass_sleep) {
    return std::make_unique<server_res_generator>(queue_tasks, queue_results, params.sleep_idle_seconds, bypass_sleep);
}

server_routes::server_routes(const common_params & params, server_context & ctx_server)
        : params(params),
          ctx_server(*ctx_server.impl),
          queue_tasks(ctx_server.impl->queue_tasks),
          queue_results(ctx_server.impl->queue_results) {
    init_routes();

    // note: this must be registered before load_model()
    //       so that on sleep phase, the callback is called before ctx is destroyed
    queue_tasks.on_sleeping_state([this](bool is_sleeping) {
        update_cached_responses(is_sleeping);
    });
}

static json adaptive_status_to_json(const server_context_adaptive_status & status) {
    return json {
        {"enabled",              status.enabled},
        {"profile",              status.profile},
        {"state",                status.state},
        {"context_size",         status.context_size},
        {"context_size_long",    status.context_size_long},
        {"mtp_weights_resident", status.mtp_weights_resident},
    };
}

static json get_res_model_info(const server_context_meta & meta, const server_context_adaptive_status & status) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    json result = {
        {"id",       meta.model_name},
        {"aliases",  meta.model_aliases},
        {"tags",     meta.model_tags},
        {"object",   "model"},
        {"created",  std::time(0)},
        {"owned_by", "llamacpp"},
        {"meta",     {
            {"vocab_type",  meta.model_vocab_type},
            {"n_vocab",     meta.model_vocab_n_tokens},
            {"n_ctx",       meta.slot_n_ctx},
            {"n_ctx_train", meta.model_n_ctx_train},
            {"n_embd",      meta.model_n_embd_inp},
            {"n_params",    meta.model_n_params},
            {"size",        meta.model_size},
            {"ftype",       meta.model_ftype},
        }},
    };
    if (status.enabled) {
        result["adaptive_context"] = adaptive_status_to_json(status);
    }
    return result;
}

static json get_res_models(const server_context_meta & meta, const common_params & params,
                           const server_context_adaptive_status & status) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    json result = json{
        {"models", json::array({
            format_codex_model_entry(meta.model_name, params.n_ctx, meta.has_mtmd),
        })},
        {"object", "list"},
        {"data", json::array({
            get_res_model_info(meta, status),
        })},
    };
    if (status.enabled) {
        result["adaptive_context"] = adaptive_status_to_json(status);
    }
    return result;
}

static json get_res_props(const server_context_meta & meta, const common_params & params, bool is_sleeping,
                          const server_context_adaptive_status & status) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    task_params tparams;
    tparams.sampling = params.sampling;
    json default_generation_settings_for_props = json {
        { "params", tparams.to_json(true) },
        { "n_ctx",  meta.slot_n_ctx },
    };

    std::string tmpl_default = common_chat_templates_source(meta.chat_params.tmpls.get(), "");
    std::string tmpl_tools   = common_chat_templates_source(meta.chat_params.tmpls.get(), "tool_use");

    json props = {
        { "default_generation_settings", default_generation_settings_for_props },
        { "total_slots",                 params.n_parallel },
        { "model_alias",                 meta.model_name },
        { "model_ftype",                 meta.model_ftype },
        { "model_path",                  meta.model_path },
        { "modalities",                  json {
            {"vision", meta.has_inp_image},
            {"video",  meta.has_inp_video},
            {"audio",  meta.has_inp_audio},
        } },
        { "media_marker",                get_media_marker() },
        { "endpoint_slots",              params.endpoint_slots },
        { "endpoint_props",              params.endpoint_props },
        { "endpoint_metrics",            params.endpoint_metrics },
        { "ui",                          params.ui },
        { "ui_settings",                 meta.json_ui_settings },
        { "chat_template",               tmpl_default },
        { "chat_template_caps",          meta.chat_template_caps },
        { "bos_token",                   meta.bos_token_str },
        { "eos_token",                   meta.eos_token_str },
        { "build_info",                  meta.build_info },
        { "is_sleeping",                 is_sleeping },
        { "cors_proxy_enabled",          params.ui_mcp_proxy },
    };
    if (status.enabled) {
        props["adaptive_context"] = adaptive_status_to_json(status);
    }
    if (params.use_jinja) {
        if (!tmpl_tools.empty()) {
            props["chat_template_tool_use"] = tmpl_tools;
        }
    }

    return props;
}

json server_routes::get_model_info() const {
    return get_res_model_info(*meta, ctx_server.get_adaptive_status());
}

void server_routes::init_routes() {
    // IMPORTANT: all lambda functions must start with create_response()
    // this is to ensure that the server_res_generator can handle sleeping case correctly

    this->get_health = [this](const server_http_req &) {
        // error and loading states are handled by middleware
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        res->ok({{"status", "ok"}});
        return res;
    };

    this->get_metrics = [this](const server_http_req & req) {
        auto res = create_response(true);
        if (!params.endpoint_metrics) {
            res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // render response using cached_metrics
        auto use_cached_metrics = [&]() {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->headers["Process-Start-Time-Unix"] = std::to_string(cached_metrics.t_start);
            server_task_result_metrics tmp;
            tmp.metrics = cached_metrics;
            res->content_type = "text/plain; version=0.0.4";
            res->status = 200;
            res->data = tmp.to_metrics();
            // the gauges are averaged over the window between two scrapes
            cached_metrics.reset_bucket();
            should_reset_buckets = true;
        };

        if (queue_tasks.is_sleeping()) {
            use_cached_metrics();

        } else {
            // request slots data using task queue
            {
                server_task task(SERVER_TASK_TYPE_METRICS);
                task.id = res->rd.get_new_id();
                // the gauges are averaged over the window between two scrapes
                task.metrics_reset_bucket = true;
                res->rd.post_task(std::move(task), true); // high-priority task
            }

            // a task posted right before sleeping is never processed, do not wait for it
            auto result = res->rd.next([&]{
                return req.should_stop() || queue_tasks.is_sleeping();
            });
            if (!result) {
                if (!req.should_stop()) {
                    use_cached_metrics();
                }
                return res;
            }

            if (result->is_error()) {
                res->error(result->to_json());
                return res;
            }

            auto res_task = dynamic_cast<server_task_result_metrics*>(result.get());
            GGML_ASSERT(res_task != nullptr);

            res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->metrics.t_start);
            res->content_type = "text/plain; version=0.0.4";
            res->status = 200;
            res->data = res_task->to_metrics();
        }

        return res;
    };

    this->get_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_SLOT_GET);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        auto * res_task = dynamic_cast<server_task_result_slots*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // optionally return "fail_on_no_slot" error
        if (!req.get_param("fail_on_no_slot").empty()) {
            if (res_task->n_idle_slots == 0) {
                res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        res->ok(res_task->to_json());
        return res;
    };

    this->post_slots = [this](const server_http_req & req) {
        auto res = create_response();
        const std::string action = req.get_param("action");
        json request_data = json::object();
        if (action == "save" || action == "restore" || (action == "erase" && !req.body.empty())) {
            request_data = json::parse(req.body);
        }
        const bool route_target_no_mtp = request_data.value("route_target_no_mtp", false);
        const bool route_state_transfer = request_data.value("route_state_transfer", false);
        const bool route_available = route_state_transfer && !router_state_dir_from_env().empty();
        if ((route_state_transfer || route_target_no_mtp) && !route_available) {
            res->error(format_error_response("Router state directory is unavailable", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        if (params.slot_save_path.empty() && !route_available) {
            res->error(format_error_response("This server does not support slots action. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::string id_slot_str = req.get_param("id_slot");

        int id_slot;
        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        if (action == "save") {
            return handle_slots_save(req, id_slot);
        }
        if (action == "restore") {
            return handle_slots_restore(req, id_slot);
        }
        if (action == "erase") {
            return handle_slots_erase(req, id_slot);
        }

        res->error(format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        return res;
    };

    this->get_props = [this](const server_http_req &) {
        auto res = create_response(true);
        // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
        if (queue_tasks.is_sleeping()) {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->ok(cached_props);
        } else {
            res->ok(get_res_props(*meta, params, false, ctx_server.get_adaptive_status()));
        }
        return res;
    };

    this->post_props = [this](const server_http_req &) {
        auto res = create_response();
        if (!params.endpoint_props) {
            res->error(format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        // update any props here

        res->ok({{ "success", true }});
        return res;
    };

    this->post_infill = [this](const server_http_req & req) {
        auto res = create_response();
        // check model compatibility
        std::string err;
        if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res->error(format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // validate input
        json data = json::parse(req.body);
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            // prompt is optional
            res->error(format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_prefix")) {
            res->error(format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_suffix")) {
            res->error(format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            // input_extra is optional
            res->error(format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res->error(format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            // filename is optional
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res->error(format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        data["input_extra"] = input_extra; // default to empty array if it's not exist

        std::string prompt = json_value(data, "prompt", std::string());
        std::vector<server_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, false, true, ctx_server.init_opt);
        SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
        data["prompt"] = format_prompt_infill(
            ctx_server.vocab,
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            params.n_batch,
            params.n_predict,
            meta->slot_n_ctx,
            params.spm_infill,
            tokenized_prompts[0].get_tokens() // TODO: this could maybe be multimodal.
        );

        std::vector<raw_buffer> files; // dummy
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            TASK_RESPONSE_TYPE_NONE); // infill is not OAI compatible
    };

    this->post_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_NONE);
    };

    this->post_completions_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_OAI_CMPL);
    };

    this->post_chat_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_chat_completions_tok = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, ctx_server.init_opt, req, TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_control = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        const std::string cmpl_id = json_value(body, "id", std::string());
        const std::string action  = json_value(body, "action", std::string());
        if (cmpl_id.empty()) {
            res->error(format_error_response("missing completion id", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (action != "reasoning_end") {
            res->error(format_error_response("unknown control action", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_CONTROL);
            task.id              = rd.get_new_id();
            task.params.control_cmpl_id = cmpl_id;
            task.params.control_action  = action;
            rd.post_task(std::move(task));
        }

        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        res->ok(result->to_json());
        return res;
    };

    this->post_responses_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_responses_to_chatcmpl(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: OpenAI Responses -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_responses_tok_oai = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, ctx_server.init_opt, req, TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_transcriptions_oai = [this](const server_http_req & req) {
        auto res = create_response();

        if (!meta->has_mtmd || !meta->chat_params.allow_audio) {
            res->error(format_error_response("The current model does not support audio input.", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::vector<raw_buffer> files;
        json body = convert_transcriptions_to_chatcmpl(
            json::parse(req.body),
            meta->chat_params.tmpls.get(),
            req.files,
            files);
        SRV_DBG("%s\n", "Request converted: OpenAI Transcriptions -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_ASR);
    };

    this->post_anthropic_messages = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    this->post_anthropic_count_tokens = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, ctx_server.init_opt, req, TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    // same with handle_chat_completions, but without inference part
    this->post_apply_template = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy, unused
        json body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        res->ok({{ "prompt", std::move(data.at("prompt")) }});
        return res;
    };

    this->get_models = [this](const server_http_req &) {
        auto res = create_response(true);
        // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
        if (queue_tasks.is_sleeping()) {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->ok(cached_models);
        } else {
            res->ok(get_res_models(*meta, params, ctx_server.get_adaptive_status()));
        }
        return res;
    };

    this->post_tokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, parse_special);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.vocab, token);
                    json piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        res->ok(json{{"tokens", std::move(tokens_response)}});
        return res;
    };

    this->post_detokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens").get<llama_tokens>();
            content = tokens_to_str(ctx_server.vocab, tokens);
        }

        res->ok(json{{"content", std::move(content)}});
        return res;
    };

    this->post_embeddings = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_NONE);
    };

    this->post_embeddings_oai = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_OAI_EMBD);
    };

    this->post_rerank = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.embedding || params.pooling_type != LLAMA_POOLING_TYPE_RANK) {
            res->error(format_error_response("This server does not support reranking. Start it with `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);

        // if true, use TEI API format, otherwise use Jina API format
        // Jina: https://jina.ai/reranker/
        // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
        bool is_tei_format = body.contains("texts");

        json query;
        if (body.count("query") == 1) {
            query = body.at("query");
            if (!query.is_string()) {
                res->error(format_error_response("\"query\" must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        } else {
            res->error(format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::vector<std::string> documents = json_value(body, "documents",
                                             json_value(body, "texts", std::vector<std::string>()));
        if (documents.empty()) {
            res->error(format_error_response("\"documents\" must be a non-empty string array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        int top_n = json_value(body, "top_n", (int)documents.size());

        // create and queue the task
        json responses = json::array();
        auto & rd = res->rd;
        {
            std::vector<server_task> tasks;
            tasks.reserve(documents.size());
            for (size_t i = 0; i < documents.size(); i++) {
                auto tmp = format_prompt_rerank(ctx_server.model_tgt, ctx_server.vocab, ctx_server.mctx, query, documents[i], ctx_server.init_opt);
                server_task task = server_task(SERVER_TASK_TYPE_RERANK);
                task.id     = rd.get_new_id();
                task.tokens = std::move(tmp);
                tasks.push_back(std::move(task));
            }
            rd.post_tasks(std::move(tasks));
        }

        // wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);

        // collect results
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_rerank*>(res.get()) != nullptr);
                responses.push_back(res->to_json());
            }
        }

        // write JSON response
        json root = format_response_rerank(
            body,
            meta->model_name,
            responses,
            is_tei_format,
            documents,
            top_n);

        res->ok(root);
        return res;
    };

    this->get_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_GET_LORA);
            task.id = rd.get_new_id();
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_get_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    this->post_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.is_array()) {
            res->error(format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_SET_LORA);
            task.id = rd.get_new_id();
            task.set_lora = parse_lora_request(body);
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_apply_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_save(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    const bool route_state_transfer = request_data.value("route_state_transfer", false);
    const std::string slot_path = route_state_transfer ? router_state_dir_from_env() : params.slot_save_path;
    if (slot_path.empty()) {
        res->error(format_error_response("Router state directory is unavailable", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }
    std::string filepath = slot_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        task.slot_action.route_state_transfer = route_state_transfer;
        task.slot_action.route_target_no_mtp = request_data.value("route_target_no_mtp", false);
        task.slot_action.route_state_reuse = request_data.value("route_state_reuse", false);
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_restore(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    const bool route_state_transfer = request_data.value("route_state_transfer", false);
    const std::string slot_path = route_state_transfer ? router_state_dir_from_env() : params.slot_save_path;
    if (slot_path.empty()) {
        res->error(format_error_response("Router state directory is unavailable", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }
    std::string filepath = slot_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        task.slot_action.route_state_transfer = route_state_transfer;
        task.slot_action.route_target_no_mtp = request_data.value("route_target_no_mtp", false);
        task.slot_action.route_state_reuse = request_data.value("route_state_reuse", false);
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_erase(const server_http_req & req, int id_slot) {
    auto res = create_response();
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_erase*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_embeddings_impl(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    if (!params.embedding) {
        res->error(format_error_response("This server does not support embeddings. Start it with `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    if (res_type != TASK_RESPONSE_TYPE_NONE && meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
        res->error(format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const json body = json::parse(req.body);

    // for the shape of input/content, see tokenize_input_prompts()
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        res_type = TASK_RESPONSE_TYPE_NONE; // "content" field is not OAI compatible
        prompt = body.at("content");
    } else {
        res->error(format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0) {
        const std::string & format = body.at("encoding_format");
        if (format == "base64") {
            use_base64 = true;
        } else if (format != "float") {
            res->error(format_error_response("The format to return the embeddings in. Can be either float or base64", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    auto tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true, ctx_server.init_opt);
    for (const auto & tokens : tokenized_prompts) {
        // this check is necessary for models that do not add BOS token to the input
        if (tokens.empty()) {
            res->error(format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    int embd_normalize = params.embd_normalize;
    if (body.count("embd_normalize") != 0) {
        embd_normalize = body.at("embd_normalize").get<int>();
        if (meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
            SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", meta->pooling_type);
        }
    }

    // create and queue the task
    json responses = json::array();
    auto & rd = res->rd;
    {
        std::vector<server_task> tasks;
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

            task.id     = rd.get_new_id();
            task.tokens = std::move(tokenized_prompts[i]);

            // OAI-compat
            task.params.res_type = res_type;
            task.params.embd_normalize = embd_normalize;

            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // collect results
    if (all_results.is_terminated) {
        return res; // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<server_task_result_embd*>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }
    }

    // write JSON response
    json root = res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? format_embeddings_response_oaicompat(body, meta->model_name, responses, use_base64)
        : json(responses);
    res->ok(root);
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_count_tokens(const llama_vocab * vocab, mtmd_context * mctx, const mtmd_helper_init_opt & init_opt, const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    std::vector<raw_buffer> files;
    json body = json::parse(req.body);
    bool is_oai = false;

    switch (res_type) {
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            {
                is_oai = true;
            } break;
        case TASK_RESPONSE_TYPE_OAI_RESP:
            {
                is_oai = true;
                body = server_chat_convert_responses_to_chatcmpl(body);
            } break;
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            {
                body = server_chat_convert_anthropic_to_oai(body);
            } break;
        default:
            res->error(format_error_response("invalid res_type", ERROR_TYPE_INVALID_REQUEST));
            return res;
    }

    json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
    json prompt = body_parsed.at("prompt");
    // SRV_DBG("prompt = %s\n", prompt.dump().c_str());

    // TODO @ngxson : refactor this code block, move this to server-common and reuse it in other places
    size_t n_tokens;
    if (mctx != nullptr) {
        if (!prompt.is_string()) {
            throw std::runtime_error("for mtmd, input prompt must be a string.");
        }
        n_tokens = process_mtmd_prompt(mctx, prompt.get<std::string>(), files, init_opt, true).size();
    } else {
        n_tokens = tokenize_mixed(vocab, prompt, true, true).size();
    }

    json response = {{"input_tokens", static_cast<int64_t>(n_tokens)}};
    if (is_oai) {
        response["object"] = "response.input_tokens";
    }
    res->ok(response);
    return res;
}

void server_routes::update_cached_responses(bool is_sleeping) {
    // caller is task_queue, so ctx_server can be accessed without holding locks
    std::unique_lock<std::mutex> lock(mutex_cache);

    if (is_sleeping) {
        const auto status = ctx_server.get_adaptive_status();
        cached_models  = get_res_models(*meta, params, status);
        cached_props   = get_res_props(*meta, params, true, status);
        cached_metrics = ctx_server.get_metrics();

        should_reset_buckets = false;

        SRV_DBG("%s\n", "cached responses updated");

    } else if (should_reset_buckets) {
        // a scrape during sleep already reported these buckets
        ctx_server.reset_metrics_bucket();

        should_reset_buckets = false;
    }
}
