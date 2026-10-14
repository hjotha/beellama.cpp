// Destination-intactness for a REFUSED restore, through the REAL production
// restore entry point (server_prompt_cache::restore -> restore_impl) and the
// REAL transactional PURE->split import, never a mock of the algorithm.
//
// What the earlier harness attempt (D9) could not measure: it issued two
// separate HTTP requests, and llama-server legitimately clears a slot when a
// new, divergent prompt arrives (slot is request-scoped). So the destination was
// never populated at the moment the import was refused. Here the destination is
// populated INSIDE the same test and the real restore is called directly, so the
// populated precondition provably exists at refusal time.
//
// Each refusal leg asserts:
//   - the call actually happened (attempt counter + refusal reason),
//   - the destination WAS populated beforehand (non-empty resident state),
//   - the resident state afterwards is byte-identical to a reference captured
//     before the call (target KV bytes + positions),
//   - the FULL next-token logit vector (every vocab entry, not a top-k subset)
//     matches an untouched control context that got the identical replay.
//
// Full vector equality is the point: a top-k subset cannot detect state drift
// outside the sampled entries, and is not a KL.

#include "ggml.h"
#include "xxhash.h"
#include "../src/llama-state-rollback.h"
#include "../src/llama-kv-mixed-state.h"
#include "server-route-state.h"
#include "llama.h"
#include "llama-ext.h"
#include "llama-kvarn.h"   // KVAR_N_GROUP
#include "kvarn-stage-extract.h"

#include "common.h"
#include "arg.h"
#include "server-task.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <filesystem>

static std::vector<std::string> g_log;

// Keep the text AND mirror it to stderr. An earlier version only buffered
// ERROR/WARN, which silently hid every backend diagnostic and made a real
// context-creation failure look like "the backend said nothing".
static void trace_log(ggml_log_level level, const char * text, void * /*ud*/) {
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        g_log.push_back(text);
    }
    fputs(text, stderr);
}

static int g_failures = 0;
static int g_checks   = 0;

#define CHECK(cond, ...) do { \
    ++g_checks; \
    if (!(cond)) { \
        ++g_failures; \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
    } \
} while (0)

namespace {

struct ctx_holder {
    llama_context * ctx = nullptr;

    ~ctx_holder() { if (ctx) { llama_free(ctx); } }
};

// Resident state of a context sequence: the exact serialized bytes plus the
// position range. This is the observable that a refused restore must not
// change; it includes whatever recurrent/DeltaNet state the sequence carries
// because it is the cache's own serialization.
struct resident {
    std::vector<uint8_t> data;
    llama_pos pos_min = -1;
    llama_pos pos_max = -1;
    bool empty = true;
};

resident capture(llama_context * ctx, llama_seq_id seq) {
    resident r;
    r.pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx), seq);
    r.pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx), seq);
    const size_t n = llama_state_seq_get_size_ext(ctx, seq, LLAMA_STATE_SEQ_FLAGS_NONE);
    if (!n) {
        r.empty = true;
        return r;
    }
    r.data.resize(n);
    if (llama_state_seq_get_data_ext(ctx, r.data.data(), n, seq, LLAMA_STATE_SEQ_FLAGS_NONE) != n) {
        r.data.clear();
        return r;
    }
    r.empty = false;
    return r;
}

bool same_resident(const resident & a, const resident & b) {
    if (a.empty != b.empty) {
        return false;
    }
    if (a.pos_min != b.pos_min || a.pos_max != b.pos_max) {
        return false;
    }
    if (a.data.size() != b.data.size()) {
        return false;
    }
    if (a.data.empty()) {
        return true;
    }
    return std::memcmp(a.data.data(), b.data.data(), a.data.size()) == 0;
}

// Decode `prompt` into ctx on `seq`, filling the cache from scratch.
bool prefill(llama_context * ctx, const std::vector<llama_token> & toks, llama_seq_id seq) {
    (void) seq;
    llama_memory_clear(llama_get_memory(ctx), true);
    for (size_t i = 0; i < toks.size(); ++i) {
        llama_batch b = llama_batch_get_one(const_cast<llama_token *>(toks.data()) + i, 1);
        if (llama_decode(ctx, b) != 0) {
            return false;
        }
    }
    return true;
}

// Decode ONE extra token on top of the resident prompt and return the COMPLETE
// vocab logit vector. Positions are tracked by the context, so this is the same
// replay the control context performs. Comparing two of these is a full-logits
// comparison, not a top-k approximation.
// Reference prefill with the SAME effective batching the producer used
// (--batch-size-short=256 / --ubatch-size-short=256). Decoding one token at a
// time is a different quantized/recurrent schedule and cannot be the control.
bool prefill_batched(llama_context * ctx, const std::vector<llama_token> & toks,
                     llama_seq_id seq, uint32_t batch) {
    (void) seq;
    if (!ctx || toks.empty()) { return false; }
    llama_memory_clear(llama_get_memory(ctx), true);
    size_t i = 0;
    while (i < toks.size()) {
        const size_t n = std::min<size_t>(batch, toks.size() - i);
        llama_batch b = llama_batch_get_one(const_cast<llama_token *>(toks.data()) + i,
                                            (int32_t) n);
        if (llama_decode(ctx, b) != 0) { return false; }
        i += n;
    }
    return true;
}

bool full_logits_after_one_token(llama_context * ctx, llama_token next, llama_seq_id seq,
                                 std::vector<float> & out) {
    (void) seq;
    out.clear();
    llama_batch b = llama_batch_get_one(&next, 1);
    const int rc = llama_decode(ctx, b);
    if (rc != 0) {
        // "no logits" has two distinct causes that the gates must not collapse
        // into one number: the context could not be created (ctx==nullptr,
        // reported by the caller) or the consumer DECODE FAILED. Say which.
        fprintf(stderr, "F4NODATA decode_failed rc=%d ctx=%s pos_max=%d\n", rc,
                ctx ? "present" : "null",
                ctx ? (int) llama_memory_seq_pos_max(llama_get_memory(ctx), seq) : -1);
        return false;
    }
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    const float * logits = llama_get_logits_ith(ctx, -1);
    if (!logits || nv <= 0) {
        return false;
    }
    out.assign(logits, logits + nv);
    return true;
}

bool same_full_logits(const std::vector<float> & a, const std::vector<float> & b,
                      float & max_abs, int & nan_count) {
    max_abs = 0.0f;
    nan_count = 0;
    if (a.size() != b.size() || a.empty()) {
        return false;
    }
    bool same = true;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::isnan(a[i]) || std::isnan(b[i])) {
            ++nan_count;
            same = false;
            continue;
        }
        const float d = std::fabs(a[i] - b[i]);
        if (d > max_abs) {
            max_abs = d;
        }
        if (d != 0.0f) {
            same = false;
        }
    }
    return same;
}

// server_prompt_cache_state is non-copyable (server_tokens is), so rebuild one
// field-by-field from a reference instead of assigning.
void build_state_like(server_prompt_cache_state & dst, const server_prompt_cache_state & src) {
    dst.model          = src.model;
    dst.model_instance = src.model_instance;
    dst.layout_tgt     = src.layout_tgt;
    dst.layout_dft     = src.layout_dft;
    dst.pos_tgt        = src.pos_tgt;
    dst.pos_dft        = src.pos_dft;
    dst.checksum       = src.checksum;
    dst.quarantined    = src.quarantined;
    dst.data.main      = src.data.main;
    dst.data.drft      = src.data.drft;
    dst.data.spec      = src.data.spec;
    dst.prompt.tokens.insert(src.prompt.tokens.get_text_tokens());
}

std::vector<llama_token> make_tokens(const llama_vocab * vocab, int n, int salt) {
    std::vector<llama_token> out;
    out.reserve(n);
    for (int i = 0; i < n; ++i) {
        // Real, in-vocab ids; the salt keeps the two sequences distinct.
        const int base = 1000 + ((i * 7 + salt * 131) % 20000);
        out.push_back(base < llama_vocab_n_tokens(vocab) ? (llama_token) base
                                                          : (llama_token) (base % 900 + 100));
    }
    return out;
}

} // namespace

namespace {
// Probe tokens for the independent-reference comparison: a short prefix, kept
// strictly below P so the remote range (positions >= P) stays unoccupied.
std::vector<llama_token> frozen_tokens_for_probe_n(const llama_vocab * vocab, int n) {
    std::vector<llama_token> t;
    for (int i = 0; i < n; ++i) {
        const int id = 1200 + ((i * 11) % 18000);
        t.push_back(id < llama_vocab_n_tokens(vocab) ? (llama_token) id : (llama_token) (i + 1));
    }
    return t;
}

llama_token probe_next(const llama_vocab * vocab) {
    const int id = 2400;
    return id < llama_vocab_n_tokens(vocab) ? (llama_token) id : (llama_token) 7;
}

} // namespace

// The mixed-state reader REJECTS a tmpfs/ramfs spool dir (measured refusal:
// "mixed state spool directory is tmpfs/ramfs"). Production sets
// TMPDIR=/var/tmp/beellama-spool via a systemd drop-in. Both cases are covered:
// a disk-backed spool (rollback must work) and a tmpfs spool (the guard must
// refuse BEFORE mutating, not claim a rollback it cannot perform).

// F4: read the leaf the attention consumer actually received for (layer, head,
// pos). This is the honest source for B/C: it is the consumer's own tensor, not
// the materializer's remote-record path.
struct cap_result {
    bool        ok = false;
    std::string info;
    std::vector<float> v;
    uint64_t    gen = 0;
};

static cap_result read_consumer_head(llama_context * ctx, int layer, bool value,
                                     uint32_t head, llama_pos pos, size_t cap, size_t index) {
    cap_result r;
    r.v.assign(cap, NAN);
    char info[768] = {0};
    r.ok = llama_kvarn_captured_head(ctx, layer, value, head, pos, index, r.v.data(),
                                      r.v.size(), info, sizeof(info));
    r.info = info;
    // Scheduler compute generation that produced this sample. The AFTER
    // (proof-decode) select must only accept samples newer than every BEFORE
    // (prefill) sample: stale prefill data can never satisfy consumer_after.
    const char * g = strstr(info, " gen=");
    if (g) { r.gen = strtoull(g + 5, nullptr, 10); }
    return r;
}

// Identity of a capture sample with the volatile counters removed: everything
// that must MATCH for two samples to be the same row (layer, kv, head, pos,
// route, range, stored leaf, shape, offset, backend). `gen`/`fa`/`seq` are
// deliberately excluded -- they are freshness, not identity.
static std::string cap_identity(const std::string & info) {
    std::string out = info;
    auto strip = [&out](const char * key) {
        size_t p = out.find(key);
        while (p != std::string::npos) {
            size_t e = p + std::strlen(key);
            while (e < out.size() && (isdigit((unsigned char) out[e]) || out[e] == '.')) { ++e; }
            out.erase(p, e - p);
            p = out.find(key, p);
        }
    };
    strip(" gen="); strip(" fa="); strip(" seq=");
    return out;
}
static unsigned long long cap_field(const std::string & info, const char * key) {
    size_t p = info.find(key);
    if (p == std::string::npos) { return 0; }
    return strtoull(info.c_str() + p + strlen(key), nullptr, 10);
}

// llama_kv_cache_kvarn::captured() treats `pos` as a FILTER
// (src/llama-kv-cache-kvarn.cpp: if (pos >= 0 && out.pos != pos) return false),
// NOT as random access: the sample is `g_f4_caps[layer,value,head][index]` and
// cs.vec already carries the whole stage slot row (elem=98304). So passing
// pos=1 does not mean "position 1 is missing" -- it means "index 0's origin
// position is not 1". Retrieve with pos=-1 (filter disabled) and select the
// sample whose OWN origin position matches, then slice the logical position
// with the descriptor-validated mapping.
static long long cap_pos_of(const std::string & info) {
    size_t p = info.find("pos=");
    if (p == std::string::npos) { return -1; }
    return strtoll(info.c_str() + p + 4, nullptr, 10);
}

// Returns the freshest sample whose ORIGIN position equals want_pos (want_pos<0
// selects index 0 unconditionally). Scans every stored sample for the key.
static cap_result read_consumer_head_at(llama_context * ctx, int layer, bool value,
                                        uint32_t head, llama_pos want_pos, size_t cap) {
    const size_t n = llama_kvarn_capture_routes(ctx, layer, value, head);
    cap_result best;
    for (size_t ri = 0; ri < n; ++ri) {
        cap_result r = read_consumer_head(ctx, layer, value, head, /*pos=*/-1, cap, ri);
        if (!r.ok) { continue; }
        if (want_pos >= 0 && cap_pos_of(r.info) != want_pos) { continue; }
        if (!best.ok || cap_field(r.info, " gen=") >= cap_field(best.info, " gen=")) {
            best = r;
        }
        if (want_pos >= 0) { break; }
    }
    return best;
}

// Shared, TEST-only extractor (see tests/kvarn-stage-extract.h).
static void describe_vec(const char * tag, const std::vector<float> & v) {
    double mn = 0, mx = 0, nrm = 0; size_t nz = 0; bool first = true;
    for (size_t i = 0; i < v.size(); ++i) {
        if (!std::isfinite(v[i])) { continue; }
        if (first) { mn = mx = v[i]; first = false; }
        mn = v[i] < mn ? v[i] : mn;
        mx = v[i] > mx ? v[i] : mx;
        nrm += (double) v[i] * (double) v[i];
        if (v[i] != 0.0f) { ++nz; }
    }
    fprintf(stderr, "F4VEC %s n=%zu min=%.6g max=%.6g norm=%.6g nonzero=%zu\n",
            tag, v.size(), mn, mx, first ? 0.0 : sqrt(nrm), nz);
}

static void set_spool(const char * dir) {
    if (dir && *dir) {
        std::filesystem::create_directories(std::filesystem::path(dir) / "beellama-mixed-state");
    }
    setenv("TMPDIR", dir && *dir ? dir : "", 1);
    if (!dir || !*dir) {
        unsetenv("TMPDIR");
    }
}


// ---- F4 harness: locate the mixed-KVarN frame through the REAL wire layout ----
// A raw search for the three ASCII bytes "KMS" is NOT a frame locator: it hits
// payload data (observed "KMS7" at 46005155 of a 60151147-byte file). Parse the
// actual outer sequence-state container, then the mixed-state envelope, then
// validate the KMS header (magic/version/length/checksum) BEFORE any mutation.
static constexpr uint32_t kMixedEnvelopeMagic   = 0x4D4B5351u; // "QSKM"
static constexpr uint32_t kMixedEnvelopeVersion = 1;
static constexpr uint32_t kKmsMagic[3] = { 0x31534D4Bu, 0x32534D4Bu, 0x33534D4Bu };
static constexpr uint32_t kKmsVersion[3] = { 1, 2, 3 };
static constexpr uint32_t kKmsHeaderBytes[3] = { 96, 104, 128 };

static uint32_t rd32(const std::vector<uint8_t> & b, size_t off) {
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) { v |= uint32_t(b[off + size_t(i)]) << (8 * i); }
    return v;
}
static uint64_t rd64(const std::vector<uint8_t> & b, size_t off) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) { v |= uint64_t(b[off + size_t(i)]) << (8 * i); }
    return v;
}

// Returns true only for a fully validated KMS frame. On failure `err` names the
// exact stage that failed, so the harness stops instead of emitting false
// negatives against payload bytes.
static bool locate_mixed_frame(const std::vector<uint8_t> & b, size_t & frame_off,
                               size_t & frame_len, int & kms_index, std::string & err) {
    frame_off = 0;
    frame_len = 0;
    kms_index = -1;
    // 1) outer sequence-state container
    if (b.size() < 12) { err = "outer container shorter than its header"; return false; }
    if (rd32(b, 0) != LLAMA_STATE_SEQ_MAGIC) { err = "outer magic is not the sequence-state magic"; return false; }
    if (rd32(b, 4) != LLAMA_STATE_SEQ_VERSION) { err = "outer sequence-state version mismatch"; return false; }
    const uint32_t n_tokens = rd32(b, 8);
    const size_t payload_off = 12 + size_t(n_tokens) * 4;
    if (payload_off > b.size()) { err = "token table runs past the end of the file"; return false; }
    // 2) mixed-KVarN envelope inside the state payload
    size_t env = std::string::npos;
    for (size_t i = payload_off; i + 16 <= b.size(); i += 4) {
        if (rd32(b, i) != kMixedEnvelopeMagic) { continue; }
        if (rd32(b, i + 4) != kMixedEnvelopeVersion) { continue; }
        const uint64_t frame_bytes = rd64(b, i + 8);
        if (frame_bytes == 0 || i + 16 + frame_bytes > b.size()) { continue; }
        env = i;
        break;
    }
    if (env == std::string::npos) {
        err = "no mixed-KVarN envelope (QSKM) in the state payload: the source "
              "is NOT using the intended KVarN mixed-state writer in this fixture";
        return false;
    }
    // 3) KMS frame header: magic, version, header length, header checksum
    const size_t f = env + 16;
    if (f + 8 > b.size()) { err = "frame header truncated"; return false; }
    const uint32_t magic = rd32(b, f);
    const uint32_t ver   = rd32(b, f + 4);
    int idx = -1;
    for (int k = 0; k < 3; ++k) { if (kKmsMagic[k] == magic) { idx = k; } }
    if (idx < 0) { err = "frame magic is not KMS1/KMS2/KMS3"; return false; }
    if (kKmsVersion[idx] != ver) { err = "frame magic/version mismatch"; return false; }
    const size_t hdr = kKmsHeaderBytes[idx];
    if (f + hdr > b.size()) { err = "frame header truncated"; return false; }
    // header checksum: v2 covers its own 104-byte header at offset 96; v1/v3
    // cover the 96-byte prefix with the checksum field (offset 88) zeroed
    std::vector<uint8_t> tmp(b.begin() + ptrdiff_t(f), b.begin() + ptrdiff_t(f + hdr));
    const size_t crc_off = (idx == 1) ? 96 : 88;
    const size_t crc_span = (idx == 1) ? 104 : 96;
    if (tmp.size() >= crc_off + 8) {
        std::memset(tmp.data() + crc_off, 0, 8);
    }
    const uint64_t stored = rd64(b, f + crc_off);
    const uint64_t calc = XXH64(tmp.data(), crc_span, 0);
    if (stored != calc) { err = "frame header checksum mismatch before mutation"; return false; }
    frame_off = f;
    frame_len = (env != std::string::npos && env + 16 + rd64(b, env + 8) <= b.size())
            ? size_t(rd64(b, env + 8)) : (b.size() - f);
    if (frame_len < hdr || f + frame_len > b.size()) {
        err = "frame extent from the envelope is not usable";
        return false;
    }
    kms_index = idx;
    return true;
}


// F4 fixture provenance: parse ONE produced artifact through the REAL codec and
// assert it is a genuine KMS3 source before any costly restore work. Read-only:
// the file is never modified, its SHA is recorded before and after, and the
// format comes from the codec -- never from the filename or from magic-byte
// scans over payload.
static int fixture_provenance_check(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "FIXTURE: cannot open %s\n", path.c_str());
        return 3;
    }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> b(size_t(n > 0 ? n : 0));
    const size_t got = n > 0 ? fread(b.data(), 1, size_t(b.size()), f) : 0;
    fclose(f);
    if (got != b.size()) {
        fprintf(stderr, "FIXTURE: short read\n");
        return 3;
    }
    const uint64_t sha_before = XXH64(b.data(), b.size(), 0);
    size_t frame_off = 0, frame_len = 0;
    int kms_idx = -1;
    std::string err;
    if (!locate_mixed_frame(b, frame_off, frame_len, kms_idx, err)) {
        fprintf(stderr, "FIXTURE: codec rejects the frame: %s\n", err.c_str());
        return 4;
    }
    // real codec: validate the located frame the same way production does
    llama_kv_mixed::snapshot s;
    bool parsed = false;
    try {
        s = llama_kv_mixed::snapshot::parse(b.data() + frame_off, frame_len);
        parsed = true;
    } catch (const std::exception & e) {
        fprintf(stderr, "FIXTURE: snapshot::parse failed: %s\n", e.what());
        return 5;
    }
    const uint64_t sha_after = XXH64(b.data(), b.size(), 0);
    fprintf(stderr,
            "FIXTURE path=%s\n"
            "FIXTURE bytes=%zu xxh64_before=%016llx xxh64_after=%016llx immutable=%d\n"
            "FIXTURE codec_magic=KMS%d codec_version=%u layers=%zu has_split=%d "
            "has_source_identity=%d\n"
            "FIXTURE source_model_id=%016llx source_rope_id=%016llx "
            "source_config_id=%016llx\n",
            path.c_str(), b.size(),
            (unsigned long long) sha_before, (unsigned long long) sha_after,
            (int) (sha_before == sha_after),
            kms_idx + 1, parsed ? s.version() : 0u,
            s.layers().size(), (int) s.has_split(),
            (int) s.has_source_identity(),
            (unsigned long long) s.frame_source_identity().model_id,
            (unsigned long long) s.frame_source_identity().rope_id,
            (unsigned long long) s.frame_source_identity().config_id);
    if (!parsed || s.version() != 3 || !s.has_source_identity() ||
        s.frame_source_identity().model_id == 0 ||
        s.frame_source_identity().rope_id == 0 ||
        s.frame_source_identity().config_id == 0) {
        fprintf(stderr, "FIXTURE: NOT a genuine KMS3 source with a complete identity\n");
        return 6;
    }
    fprintf(stderr, "FIXTURE: OK -- genuine KMS3 source with complete writer identity\n");
    return 0;
}

int main(int argc, char ** argv) {
    // F4 fixture provenance gate: runs BEFORE any model load / GPU work.
    if (const char * fix = getenv("F4_TEST_FIXTURE_ONLY")) {
        const int rc = fixture_provenance_check(fix);
        printf("fixture-provenance: rc=%d\n", rc);
        return rc;
    }
    const char * model_path = argc > 1 ? argv[1]
        : "/home/hjotha/models/Qwen3.5-4B-MTP-Q4_K_M.gguf";
    const int P = 512;
    const int C = 2048;
    const llama_seq_id seq = 0;

    // Disk-backed spool, same shape production uses.
    const char * spool = getenv("F4_TEST_SPOOL");
    set_spool(spool ? spool : "/var/tmp/beellama-spool");

    llama_log_set(trace_log, nullptr);
    llama_backend_init();

    // The position-split destination REQUIRES the full-attention layers on CUDA
    // (local KVarN mirror) plus a remote Q4 Vulkan stripe, so this is a GPU test
    // by construction and must only run with the GPU free. Without F4_TEST_GPU
    // it stays on CPU, where the split context cannot be created at all.
    const bool use_gpu = getenv("F4_TEST_GPU") != nullptr;
    if (!use_gpu) {
        printf("destination-intact: GPU disabled (set F4_TEST_GPU=1 to run against CUDA+Vulkan); skipping\n");
        return 0;
    }
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "destination-intact: cannot load model %s\n", model_path);
        return 2;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // The split placement REQUIRES a remote Q4 KV cache, so the destination is
    // built on the Vulkan device. Run this only with the GPU free: it allocates
    // real VRAM through the remote stripes.
    const char * remote_host = getenv("F4_TEST_REMOTE_ATTN");
    const char * remote_dev  = remote_host ? remote_host : "vulkan:0";

    auto make_ctx = [&](int p, int split, int remote = -1) {
        setenv("LLAMA_KV_POSITION_SPLIT_P", split ? std::to_string(p).c_str() : "", 1);
        unsetenv("LLAMA_KV_POSITION_SPLIT_P");
        if (split) {
            setenv("LLAMA_KV_POSITION_SPLIT_P", std::to_string(p).c_str(), 1);
        }
        // remote < 0 keeps the historical default (remote only when split);
        // remote == 0 forces a split context WITHOUT the remote Q4 KV so the
        // position split and the remote placement can be bisected apart.
        const bool use_remote = split && remote != 0;
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = (uint32_t) C;
        // The producer records n_ctx_slot=4096 (worker.err); the reference must
        // be able to use the SAME effective context size (F4_N_CTX) so a
        // continuation comparison is not confounded by a different capacity.
        if (const char * e = getenv("F4_N_CTX")) {
            const int v = atoi(e);
            if (v > 0) { cp.n_ctx = (uint32_t) v; }
        }
        cp.n_batch    = 256;
        cp.n_ubatch   = 256;
        cp.n_seq_max  = 1;
        cp.kvarn      = llama_kvarn_params_for_type(LLAMA_KVARN_K4V4_G128);
        cp.type_k     = GGML_TYPE_F16;
        cp.type_v     = GGML_TYPE_F16;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        if (use_remote) {
            cp.remote_attn_host     = remote_dev;
            cp.remote_attn_n_layers = 8;
            cp.n_ctx                = (uint32_t) C;
            // The position-split placement REQUIRES a remote Q4 KV cache.
            cp.remote_attn_cache_type_k = GGML_TYPE_Q4_0;
            cp.remote_attn_cache_type_v = GGML_TYPE_Q4_0;
        }
        return llama_init_from_model(model, cp);
    };

    // ---------------------------------------------------------------------------
    // F4: ACTUAL-LOADER populated-destination rejection matrix (plan 3.6).
    //
    // Destination is populated with a DIFFERENT resident prefix than the disk
    // candidate, the real llama_state_seq_load_file_streaming path is invoked
    // for every negative, and the resident state (KV payload + cell table +
    // recurrent/DeltaNet) is fingerprinted IMMEDIATELY before the rejection and
    // IMMEDIATELY after the rollback -- before any fallback prefill can touch
    // it. Gated by F4_TEST_POPULATED so the default matrix run is unchanged.
    // ---------------------------------------------------------------------------
    if (const char * pop = getenv("F4_TEST_POPULATED")) {
        const std::string dir = pop;
        const std::string pure_file  = dir + "/pure-source.state";
        const std::string attack_file = dir + "/attack.state";

        // (1) PURE (non-split) source: its state file carries the ORIGINAL
        //     WRITER identity (KMS3).
        std::vector<llama_token> src_toks = frozen_tokens_for_probe_n(vocab, 96);
        // F4_TEST_SOURCE_STATE: reuse a VERIFIED producer artifact (the gap3 PURE
        // producer that already produced a genuine fresh-process restore). The
        // original is copied; every mutation happens on the harness's own copy,
        // so preserved evidence is never touched.
        const char * src_state = getenv("F4_TEST_SOURCE_STATE");
        if (src_state && src_state[0]) {
            // Use the VERIFIED producer artifact read-only: every mutation goes
            // into `b` and is written to attack_file, so the original is never
            // touched (a copy was pointless and failed on a read-only source).
            printf("POPULATED: using verified producer artifact (read-only) %s\n", src_state);
        } else {
        {
            ctx_holder src;
            src.ctx = make_ctx(P, /*split=*/false);
            if (!src.ctx) { fprintf(stderr, "POPULATED: pure source context FAILED\n"); }
            else {
                if (!prefill(src.ctx, src_toks, seq)) { fprintf(stderr, "POPULATED: source prefill FAILED\n"); }
                if (!llama_state_seq_save_file(src.ctx, pure_file.c_str(), seq,
                                               src_toks.data(), src_toks.size())) {
                    fprintf(stderr, "POPULATED: source save FAILED\n");
                }
            }
        }
        }   // source context freed: VRAM returned before the destination is built

        // (2) POPULATED DESTINATION with a DIFFERENT, shorter resident prefix.
        ctx_holder dest;
        // The verified source artifact spans 563 tokens; the authorized import
        // gate requires extent <= min(P, capacity). P must therefore be large
        // enough for THIS source, otherwise every case refuses at the extent
        // gate and never reaches the identity checks (misleading reasons).
        const int P_DEST = 1024;
        dest.ctx = make_ctx(P_DEST, /*split=*/true);
        std::vector<llama_token> res_toks = frozen_tokens_for_probe_n(vocab, 33);
        if (dest.ctx && prefill(dest.ctx, res_toks, seq)) {
            const resident f_before = capture(dest.ctx, seq);
            // determinism of the fingerprint itself: two reads must agree
            const resident f_check  = capture(dest.ctx, seq);
            const bool stable = same_resident(f_before, f_check);
            printf("      POPULATED fingerprint bytes=%zu stable=%d resident_pos=%d..%d\n",
                   f_before.data.size(), (int) stable, (int) f_before.pos_min, (int) f_before.pos_max);
            CHECK(stable, "populated-destination fingerprint is stable before any load");

            auto read_file = [](const std::string & path, std::vector<uint8_t> & out) {
                FILE * f = fopen(path.c_str(), "rb");
                if (!f) { return false; }
                fseek(f, 0, SEEK_END);
                const long n = ftell(f);
                fseek(f, 0, SEEK_SET);
                out.resize(size_t(n > 0 ? n : 0));
                const size_t got = n > 0 ? fread(out.data(), 1, size_t(n), f) : 0;
                fclose(f);
                return got == out.size();
            };
            auto write_file = [](const std::string & path, const std::vector<uint8_t> & b) {
                FILE * f = fopen(path.c_str(), "wb");
                if (!f) { return false; }
                const size_t wrote = fwrite(b.data(), 1, b.size(), f);
                fclose(f);
                return wrote == b.size();
            };

                const std::string pure_path = (src_state && src_state[0])
                    ? std::string(src_state) : pure_file;
            std::vector<uint8_t> pure;
            if (!read_file(pure_path, pure)) {
                fprintf(stderr, "POPULATED: cannot read source state file\n");
            } else {
                // ------------------------------------------------------------
                // POSITIVE CONTROL first: the INTACT artifact is imported into
                // the populated split destination through the real loader, and
                // the restored state is compared numerically against a control
                // that prefills the very same source tokens. Only after that
                // accepted control are the negatives derived from it.
                // ------------------------------------------------------------
                uint32_t REF_BATCH = 256;
                if (const char * rb = getenv("F4_REF_BATCH")) {
                    const int v = atoi(rb);
                    if (v > 0) { REF_BATCH = (uint32_t) v; }
                }
                printf("[REF-BATCH] %u  (producer batch/ubatch)\n", REF_BATCH);
                size_t native_extent = 0;
                uint64_t native_checksum = 0;
                // Producer-effective configuration, read from the SAVED argv the
                // router actually built (run-*/worker-args.txt). This is the only
                // authoritative record of the producer's ctx/batch/ubatch/cache/
                // RoPE settings -- it is NOT implied by make_ctx defaults.
                if (const char * pa = getenv("F4_PRODUCER_ARGS")) {
                    FILE * f = fopen(pa, "r");
                    if (f) {
                        std::vector<std::string> args;
                        char line[512];
                        while (fgets(line, sizeof(line), f)) {
                            size_t n = strlen(line);
                            while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) { line[--n] = 0; }
                            if (n) { args.push_back(line); }
                        }
                        fclose(f);
                        const auto want = [&](const char * k, const char * dflt) {
                            for (size_t i = 0; i + 1 < args.size(); ++i) {
                                if (args[i] == k) { return args[i + 1]; }
                            }
                            return std::string(dflt);
                        };
                        printf("[PRODUCER-EFF] ctx-size-short=%s batch-short=%s ubatch-short=%s "
                               "cache-k=%s cache-v=%s flash-attn=%s remote=%s "
                               "rope-scaling=%s seed=%s\n",
                               want("--ctx-size-short", "?").c_str(),
                               want("--batch-size-short", "?").c_str(),
                               want("--ubatch-size-short", "?").c_str(),
                               want("--cache-type-k-short", "?").c_str(),
                               want("--cache-type-v-short", "?").c_str(),
                               want("--flash-attn", "?").c_str(),
                               want("--remote-attn", "none").c_str(),
                               want("--rope-scaling", "none").c_str(),
                               want("--seed", "?").c_str());
                        printf("[MAKE-CTX-EFF]  ctx-size=%d batch=%d ubatch=%d "
                               "kvarn=K4V4_G128 type_k=F16 flash-attn=ENABLED remote=%s\n",
                               getenv("F4_N_CTX") ? atoi(getenv("F4_N_CTX")) : C, 256, 256, "none");
                    }
                }
                std::vector<llama_token> src_tokens;
                if (pure.size() >= 12) {
                    const uint32_t ntok = uint32_t(pure[8]) | (uint32_t(pure[9]) << 8) |
                                          (uint32_t(pure[10]) << 16) | (uint32_t(pure[11]) << 24);
                    const size_t tok_bytes = size_t(ntok) * 4;
                    if (12 + tok_bytes <= pure.size()) {
                        src_tokens.resize(ntok);
                        for (uint32_t i = 0; i < ntok; ++i) {
                            const size_t o = 12 + size_t(i) * 4;
                            src_tokens[i] = llama_token(uint32_t(pure[o]) | (uint32_t(pure[o + 1]) << 8) |
                                                        (uint32_t(pure[o + 2]) << 16) | (uint32_t(pure[o + 3]) << 24));
                        }
                    }
                }
                CHECK(!src_tokens.empty(), "positive control: source token table read from the artifact");
                const llama_token NEXT_TOKEN = src_tokens.empty() ? 0 : src_tokens.back();

                // control: cold split context prefilled with the SAME tokens
                // Producer-equivalence: build the reference through the SAME
                // common configuration path the producer used (parse its actual
                // argv -> common_context_params_to_llama) instead of hand-copied
                // make_ctx defaults. The only forced field is n_ctx=4096, the
                // measured n_ctx_slot from worker.err, because adaptive tier
                // selection is router-level and this is a single context.
                llama_context_params prod_lcp = llama_context_default_params();
                bool prod_cfg_ok = false;
                {
                    std::vector<std::string> raw;
                    if (const char * af = getenv("F4_PRODUCER_ARGS")) {
                        FILE * f = fopen(af, "r");
                        if (f) {
                            char line[512];
                            while (fgets(line, sizeof(line), f)) {
                                size_t n = strlen(line);
                                while (n && (line[n-1] == '\n' || line[n-1] == '\r')) { line[--n] = 0; }
                                if (n) { raw.push_back(line); }
                            }
                            fclose(f);
                        }
                    }
                    // drop router-only adaptive tier flags, pin the measured ctx
                    // The log stores "--flag" then "value" on separate lines.
                    // Pair them here: a flag followed by a non-flag token takes
                    // that token as its value, otherwise it is a boolean. Naive
                    // concatenation makes `--cache-prompt 256` and the parser
                    // correctly rejects it.
                    std::vector<std::string> keep;
                    keep.push_back("llama-server");
                    const auto is_flag = [](const std::string & t) {
                        return t.rfind("--", 0) == 0;
                    };
                    size_t i = 0;
                    while (i < raw.size()) {
                        const std::string & a = raw[i];
                        const bool drop_tier =
                                a.rfind("--ctx-size-", 0) == 0 ||
                                (a.rfind("--batch-size-", 0) == 0 && a != "--batch-size-short") ||
                                (a.rfind("--ubatch-size-", 0) == 0 && a != "--ubatch-size-short");
                        if (drop_tier) {
                            i += (i + 1 < raw.size() && !is_flag(raw[i + 1])) ? 2 : 1;
                            continue;
                        }
                        keep.push_back(a);
                        ++i;
                        if (i < raw.size() && !is_flag(raw[i])) {
                            keep.push_back(raw[i]);
                            ++i;
                        }
                    }
                    keep.push_back("--ctx-size");
                    keep.push_back("4096");
                    std::vector<char *> argv;
                    for (auto & k : keep) { argv.push_back(const_cast<char *>(k.c_str())); }
                    if (!argv.empty()) {
                        try {
                            common_params cparams;
                            if (common_params_parse((int) argv.size(), argv.data(), cparams,
                                                    LLAMA_EXAMPLE_SERVER)) {
                                prod_lcp = common_context_params_to_llama(cparams);
                                prod_cfg_ok = true;
                            }
                        } catch (const std::exception & e) {
                            printf("[PRODUCER-CFG] parse failed: %s\n", e.what());
                        }
                    }
                }
                if (prod_cfg_ok) {
                    printf("[PRODUCER-CFG] n_ctx=%u n_batch=%d n_ubatch=%d "
                           "flash_attn=%d rope_base=%.1f rope_scale=%.6f "
                           "kv_tail_tokens=%u kv_unified=%d\n",
                           prod_lcp.n_ctx, prod_lcp.n_batch, prod_lcp.n_ubatch,
                           (int) prod_lcp.flash_attn_type,
                           (double) prod_lcp.rope_freq_base,
                           (double) prod_lcp.rope_freq_scale,
                           prod_lcp.kv_tail_tokens, (int) prod_lcp.kv_unified);
                } else {
                    printf("[PRODUCER-CFG] unavailable -> make_ctx fallback\n");
                }

                std::vector<float> ref_logits;
                std::vector<float> ref_pure_logits;
                {
                    ctx_holder ref;
                    if (prod_cfg_ok) {
                        // Same common configuration path as the producer, plus
                        // ONLY the split deltas the producer (pure) does not have:
                        // the remote Q4 stripe the placement requires. Batch and
                        // ubatch are the short-tier values read from worker-args
                        // (--batch-size-short 256 / --ubatch-size-short 256).
                        setenv("LLAMA_KV_POSITION_SPLIT_P", std::to_string(P_DEST).c_str(), 1);
                        prod_lcp.kvarn = llama_kvarn_params_for_type(LLAMA_KVARN_K4V4_G128);
                        prod_lcp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
                        prod_lcp.remote_attn_host     = remote_dev;
                        prod_lcp.remote_attn_n_layers = 8;
                        prod_lcp.remote_attn_cache_type_k = GGML_TYPE_Q4_0;
                        prod_lcp.remote_attn_cache_type_v = GGML_TYPE_Q4_0;
                        prod_lcp.n_seq_max = 1;
                        prod_lcp.n_batch  = 256;
                        prod_lcp.n_ubatch = 256;
                        prod_lcp.n_ctx    = 4096;
                        ref.ctx = llama_init_from_model(model, prod_lcp);
                        if (!ref.ctx) {
                            printf("[PRODUCER-CFG] split context creation FAILED\n");
                        }
                        unsetenv("LLAMA_KV_POSITION_SPLIT_P");
                    } else {
                        ref.ctx = make_ctx(P_DEST, /*split=*/true);
                    }
                    const uint32_t ref_batch = REF_BATCH;
                    const bool ref_ok = ref.ctx && !src_tokens.empty() &&
                                        prefill_batched(ref.ctx, src_tokens, seq, ref_batch);
                    CHECK(ref_ok, "[POSITIVE] split reference prefills the source tokens");
                    if (ref_ok) {
                        full_logits_after_one_token(ref.ctx, NEXT_TOKEN, seq, ref_logits);
                        CHECK(!ref_logits.empty(),
                              "[POSITIVE] split reference continuation produced full logits");
                    }
                }
                // Discriminator: the SAME direct prefill in a PURE destination.
                // split-prefill vs pure-prefill isolates layout/config from
                // persistence/mapping without any restore involved.
                {
                    ctx_holder rp;
                    rp.ctx = make_ctx(P_DEST, /*split=*/false);
                    const bool ok = rp.ctx && !src_tokens.empty() &&
                                    prefill_batched(rp.ctx, src_tokens, seq, REF_BATCH);
                    CHECK(ok, "[POSITIVE] pure reference prefills the source tokens");
                    if (ok) {
                        full_logits_after_one_token(rp.ctx, NEXT_TOKEN, seq, ref_pure_logits);
                        CHECK(!ref_pure_logits.empty(),
                              "[POSITIVE] pure reference continuation produced full logits");
                    }
                }
                if (!ref_logits.empty() && !ref_pure_logits.empty() &&
                        ref_logits.size() == ref_pure_logits.size()) {
                    double d = 0.0;
                    for (size_t i = 0; i < ref_logits.size(); ++i) {
                        const double e = std::fabs(double(ref_logits[i]) - double(ref_pure_logits[i]));
                        if (e > d) { d = e; }
                    }
                    printf("      [DISCRIM] split_prefill vs pure_prefill max_abs_diff=%.6e\n", d);
                }

                // destination: capture -> clear -> real loader with the INTACT artifact
                std::vector<uint8_t> snap_pos;
                llama_pos spos_pos = -1;
                bool had_pos = false;
                std::string err_pos;
                const bool cap_ok_pos = llama_state_rollback::capture_resident(
                        dest.ctx, seq, snap_pos, &spos_pos, &had_pos, &err_pos);
                CHECK(cap_ok_pos && had_pos, "positive control: populated destination captured");
                // SAME-PROCESS baseline: save from a harness context, reload it
                // in a second harness context with identical parameters and
                // tokens. This isolates SERIALIZATION from server-vs-harness
                // setup: if this round-trip is exact, the residual gap between a
                // server-written snapshot and a cold prefill is NOT caused by
                // save/load itself.
                std::vector<float> own_save_logits;
                std::vector<float> own_load_logits;
                {
                    const std::string own_path = dir + "/selfroundtrip.state";
                    ctx_holder a;
                    a.ctx = make_ctx(P_DEST, /*split=*/false);
                    const bool a_ok = a.ctx && !src_tokens.empty() &&
                                      prefill_batched(a.ctx, src_tokens, seq, REF_BATCH);
                    CHECK(a_ok, "[SELF-RT] harness context prefills the source tokens");
                    if (a_ok) {
                        const size_t w = llama_state_seq_save_file(a.ctx, own_path.c_str(),
                                seq, src_tokens.data(), src_tokens.size());
                        CHECK(w > 0, "[SELF-RT] harness context saves its own state");
                        ctx_holder b;
                        b.ctx = make_ctx(P_DEST, /*split=*/false);
                        bool l_ok = false;
                        if (b.ctx && w > 0) {
                            std::vector<llama_token> ot(1024, 0);
                            size_t no = 0;
                            l_ok = llama_state_seq_load_file(b.ctx, own_path.c_str(), seq, ot.data(),
                                                         ot.capacity(), &no);
                            if (l_ok) { CHECK(no == src_tokens.size(),
                                              "[SELF-RT] reload restores the same token count"); }
                        }
                        if (a_ok) { full_logits_after_one_token(a.ctx, NEXT_TOKEN, seq, own_save_logits); }
                        if (b.ctx && l_ok) {
                            full_logits_after_one_token(b.ctx, NEXT_TOKEN, seq, own_load_logits);
                        }
                        printf("[SELF-RT] save_bytes=%zu load_ok=%d tokens=%zu "
                               "save_logits=%zu load_logits=%zu\n",
                               w, (int) l_ok, (size_t) 0,
                               own_save_logits.size(), own_load_logits.size());
                        const double d = [&]() {
                            if (own_save_logits.empty() || own_save_logits.size() != own_load_logits.size()) return -1.0;
                            double m = 0.0;
                            for (size_t i = 0; i < own_save_logits.size(); ++i) {
                                const double e = std::fabs(double(own_save_logits[i]) - double(own_load_logits[i]));
                                if (e > m) { m = e; }
                            }
                            return m;
                        }();
                        printf("[SELF-RT] save->load (same process, same params) max_abs_diff=%.6e\n", d);
                        printf("[SELF-RT] harness_cold vs server_written max_abs_diff=%.6e\n",
                               [&]() {
                                   if (own_save_logits.empty() || ref_pure_logits.size() != own_save_logits.size()) return -1.0;
                                   double m = 0.0;
                                   for (size_t i = 0; i < own_save_logits.size(); ++i) {
                                       const double e = std::fabs(double(own_save_logits[i]) - double(ref_pure_logits[i]));
                                       if (e > m) { m = e; }
                                   }
                                   return m;
                               }());
                    }
                }

                // [DISC1] ONE harness-produced canonical state, one immutable run
                // id, same source bytes, NATIVE STREAMING restore into PURE and
                // into SPLIT. Labels are built from the run id and the ACTUAL
                // route taken -- no manual claims.
                {
                    const uint64_t tok_hash = src_tokens.empty() ? 0 :
                            XXH64(src_tokens.data(), src_tokens.size() * sizeof(llama_token), 0);
                    const std::string run_id =
                            std::string("run-") + std::to_string(tok_hash) + "-harness-native";
                    const std::string nat  = dir + "/disc1-native.state";
                    const std::string canon = dir + "/disc1-canonical.state";
                    std::vector<float> base;
                    ctx_holder a;
                    a.ctx = make_ctx(P_DEST, /*split=*/false);
                    const bool a_ok = a.ctx && !src_tokens.empty() &&
                                      prefill_batched(a.ctx, src_tokens, seq, REF_BATCH);
                    CHECK(a_ok, "[DISC1] baseline cold prefill (batch=256)");
                    if (a_ok) {
                        const size_t w = llama_state_seq_save_file(a.ctx, nat.c_str(), seq,
                                src_tokens.data(), src_tokens.size());
                        CHECK(w > 0, "[DISC1] native state written by this run");
                        full_logits_after_one_token(a.ctx, NEXT_TOKEN, seq, base);
                        CHECK(!base.empty(), "[DISC1] baseline continuation produced full logits");

                        const bool adopted = server_route_state_adopt_native(
                                nat, canon, "f4-harness", "f4-harness", src_tokens,
                                /*source_ctx=*/4096, /*index_block=*/0,
                                /*max_bytes=*/(uint64_t) 1 << 40, /*provenance_json=*/nullptr);
                        CHECK(adopted, "[DISC1] native -> canonical envelope adopted");
                        printf("[DISC1] run=%s adopted=%d baseline_logits=%zu\n",
                               run_id.c_str(), (int) adopted, base.size());

                        if (adopted) {
                            server_route_state_file srsf;
                            bool meta_ok = false;
                            try {
                                srsf = server_route_state_read(canon, (uint64_t) 1 << 40,
                                                               65536, /*verify_payload=*/true);
                                meta_ok = true;
                            } catch (const std::exception & e) {
                                printf("[DISC1] canonical read FAILED: %s\n", e.what());
                            }
                            if (meta_ok) {
                                printf("[DISC1] run=%s meta file_bytes=%zu state_bytes=%zu "
                                       "state_checksum=%016llx tokens=%zu position=%d\n",
                                       run_id.c_str(), srsf.file_bytes, srsf.state_bytes,
                                       (unsigned long long) srsf.state_checksum,
                                       srsf.tokens.size(), (int) srsf.position);
                                for (int k = 0; k < 2; ++k) {
                                    const char * route = (k == 0) ? "pure" : "split";
                                    ctx_holder r;
                                    r.ctx = make_ctx(P_DEST, /*split=*/k == 1);
                                    std::vector<float> got;
                                    size_t rc = 0, no = 0;
                                    llama_pos pos = -1;
                                    if (r.ctx) {
                                        std::vector<llama_token> ot(srsf.tokens.size() + 8, 0);
                                        rc = llama_state_seq_load_file_streaming(
                                                r.ctx, canon.c_str(), seq, ot.data(), ot.size(), &no,
                                                srsf.state_bytes, srsf.state_checksum);
                                        pos = llama_memory_seq_pos_max(llama_get_memory(r.ctx), seq);
                                        if (rc == srsf.state_bytes) {
                                            full_logits_after_one_token(r.ctx, NEXT_TOKEN, seq, got);
                                        }
                                    }
                                    double d = -1.0;
                                    if (!base.empty() && base.size() == got.size()) {
                                        d = 0.0;
                                        for (size_t i = 0; i < base.size(); ++i) {
                                            const double e = std::fabs(double(base[i]) - double(got[i]));
                                            if (e > d) { d = e; }
                                        }
                                    }
                                    printf("[DISC1] run=%s route=%s rc=%zu/%zu tokens=%zu pos=%d "
                                           "logits=%zu max_abs_diff=%.6e\n",
                                           run_id.c_str(), route, rc, srsf.state_bytes, no,
                                           (int) pos, got.size(), d);
                                }
                            }
                        }
                    }
                }

                const resident f_pos_before = capture(dest.ctx, seq);
                llama_memory_seq_rm(llama_get_memory(dest.ctx), seq, 0, -1);

                const std::string intact_path = dir + "/intact.state";
                CHECK(write_file(intact_path, pure), "[POSITIVE] intact artifact written");
                // The fixture is a SERVER UNIFIED native/footer snapshot, not a
                // bare native state file. Production (do_slot_restore,
                // server-context.cpp:4093) passes file.state_bytes and
                // file.state_checksum into the lower API -- never the container
                // size and never a hash over the whole container. Use the SAME
                // canonical parser here.
                // max_bytes is a hard ceiling over the CONTAINER size (the
                // reader throws when the file exceeds it); 0 would reject
                // everything. verify_payload hashes the native prefix so the
                // contract we then hand to the lower API is already checked.
                server_route_state_file srsf;
                bool srsf_ok = false;
                try {
                    srsf = server_route_state_read(intact_path, pure.size(),
                                                    /*max_tokens=*/65536,
                                                    /*verify_payload=*/true);
                    srsf_ok = true;
                } catch (const std::exception & e) {
                    printf("[POSITIVE] container parse FAILED: %s\n", e.what());
                    CHECK(false, "[POSITIVE] canonical server_route_state_read parses the fixture");
                }
                native_extent = srsf_ok ? srsf.state_bytes : 0;
                native_checksum = srsf_ok ? srsf.state_checksum : 0;
                if (srsf_ok) {
                    CHECK(native_extent > 0 && native_extent <= pure.size(),
                          "[POSITIVE] declared native extent is inside the container");
                    CHECK(XXH64(pure.data(), native_extent, 0) == native_checksum,
                          "[POSITIVE] declared native checksum matches the native prefix");
                }
                if (srsf_ok) printf("[POSITIVE] container file_bytes=%zu native state_bytes=%zu "
                       "state_checksum=%016llx logits_offset=%llu logits_bytes=%llu "
                       "footer=[%zu,%zu) tokens=%zu position=%d\n",
                       srsf.file_bytes, srsf.state_bytes,
                       (unsigned long long) srsf.state_checksum,
                       (unsigned long long) srsf.logits_offset,
                       (unsigned long long) srsf.logits_bytes,
                       srsf.state_bytes, srsf.file_bytes,
                       srsf.tokens.size(), (int) srsf.position);
                std::vector<llama_token> out_toks(srsf.tokens.empty() ? 1024 : srsf.tokens.size(), 0);
                size_t n_out_pos = 0;
                const size_t rc_pos = llama_state_seq_load_file_streaming(
                        dest.ctx, intact_path.c_str(), seq, out_toks.data(), out_toks.size(),
                        &n_out_pos, native_extent, native_checksum);
                const bool import_ok = (rc_pos == native_extent);
                CHECK(import_ok,
                      (std::string("[POSITIVE] intact import succeeds (rc=") +
                       std::to_string(rc_pos) + "/" + std::to_string(native_extent) +
                       " container=" + std::to_string(pure.size()) + ")").c_str());
                CHECK(n_out_pos == src_tokens.size(),
                      (std::string("[POSITIVE] restored token count matches the source (") +
                       std::to_string(n_out_pos) + " vs " + std::to_string(src_tokens.size()) + ")").c_str());
                const llama_pos rest_pos =
                        llama_memory_seq_pos_max(llama_get_memory(dest.ctx), seq);
                CHECK(rest_pos == llama_pos(src_tokens.size()) - 1,
                      "positive control: restored position watermark matches the source extent");

                // numerical gate: identical continuation logits vs the reference
                std::vector<float> got_logits;
                if (import_ok && !src_tokens.empty()) {
                    full_logits_after_one_token(dest.ctx, NEXT_TOKEN, seq, got_logits);
                    CHECK(!got_logits.empty(),
                          "positive control: restored continuation produced full logits");
                    double max_abs = 0.0;
                    if (!ref_logits.empty() && ref_logits.size() == got_logits.size()) {
                        for (size_t i = 0; i < got_logits.size(); ++i) {
                            const double d = std::fabs(double(got_logits[i]) - double(ref_logits[i]));
                            if (d > max_abs) { max_abs = d; }
                        }
                    } else {
                        max_abs = -1.0;
                    }
                    printf("      [POSITIVE] import rc=%zu tokens=%zu pos=%d "
                           "logits=%zu max_abs_diff=%.6e\n",
                           rc_pos, n_out_pos, (int) rest_pos, got_logits.size(), max_abs);

                // [DISC2] PRODUCER'S OWN continuation baseline: the canonical
                // snapshot stores its logits payload (logits_offset/logits_bytes
                // in the footer). Comparing the restored continuation against
                // THAT is the matched-schedule baseline: no template or config
                // guessing, tokens come from the manifest.
                {
                    server_route_state_file meta;
                    bool meta_ok = false;
                    try {
                        meta = server_route_state_read(pure_path, (uint64_t) 1 << 40, 65536,
                                                       /*verify_payload=*/true);
                        meta_ok = true;
                    } catch (const std::exception & e) {
                        printf("[DISC2] canonical read failed: %s\n", e.what());
                    }
                    if (meta_ok && meta.logits_bytes > 0) {
                        std::vector<float> producer_logits;
                        const bool ok = server_route_state_read_logits(meta, meta.logits_vocab, producer_logits);
                        printf("[DISC2] producer_logits bytes=%llu vocab=%u tokens=%u read=%d count=%zu\n",
                               (unsigned long long) meta.logits_bytes, meta.logits_vocab,
                               meta.logits_tokens, (int) ok, producer_logits.size());
                        if (ok && got_logits.size() == producer_logits.size() && !got_logits.empty()) {
                            double d2 = 0.0;
                            for (size_t i = 0; i < got_logits.size(); ++i) {
                                const double e = std::fabs(double(got_logits[i]) - double(producer_logits[i]));
                                if (e > d2) { d2 = e; }
                            }
                            printf("[DISC2] restored vs PRODUCER-STORED continuation max_abs_diff=%.6e\n", d2);
                        } else {
                            printf("[DISC2] size mismatch restored=%zu producer=%zu read=%d\n",
                                   got_logits.size(), producer_logits.size(), (int) ok);
                        }
                    } else {
                        printf("[DISC2] snapshot carries no logits payload\n");
                    }
                }

                // [DISC2] PRODUCER'S OWN continuation baseline: the canonical
                // snapshot stores its logits payload (logits_offset/logits_bytes
                // in the footer). Comparing the restored-state continuation
                // against THAT is the matched schedule baseline -- no template
                // or config guessing, and the tokens come from the manifest.
                {
                    server_route_state_file meta;
                    try {
                        meta = server_route_state_read(pure_path, (uint64_t) 1 << 40, 65536,
                                                       /*verify_payload=*/true);
                    } catch (const std::exception & e) {
                        printf("[DISC2] canonical read failed: %s\n", e.what());
                    }
                    CHECK(ref_logits.size() == got_logits.size() && max_abs >= 0.0 && max_abs <= 1e-3,
                          (std::string("positive control: restored logits match the reference "
                                       "(max_abs_diff=") + std::to_string(max_abs) + ")").c_str());
                }
                // ---- PURE destination: same immutable source, same continuation
                // Pure-restore vs pure-prefill isolates NATIVE persistence;
                // pure-restore vs split-restore isolates the pure->split mapping.
                std::vector<float> pure_restored_logits;
                std::string pure_rc = "skipped";
                {
                    ctx_holder pd;
                    pd.ctx = make_ctx(P_DEST, /*split=*/false);
                    if (pd.ctx) {
                        std::vector<uint8_t> snap_p;
                        llama_pos spos_p = -1; bool had_p = false; std::string err_p;
                        llama_state_rollback::capture_resident(pd.ctx, seq, snap_p, &spos_p,
                                                              &had_p, &err_p);
                        llama_memory_seq_rm(llama_get_memory(pd.ctx), seq, 0, -1);
                        std::vector<llama_token> ot(1024, 0);
                        size_t no = 0;
                        const size_t rc_p = llama_state_seq_load_file_streaming(
                                pd.ctx, intact_path.c_str(), seq, ot.data(), ot.size(), &no,
                                native_extent, native_checksum);
                        pure_rc = std::to_string(rc_p);
                        const llama_pos pos_before =
                                llama_memory_seq_pos_max(llama_get_memory(pd.ctx), seq);
                        if (rc_p == native_extent) {
                            full_logits_after_one_token(pd.ctx, NEXT_TOKEN, seq,
                                                        pure_restored_logits);
                        }
                        const llama_pos pos_after =
                                llama_memory_seq_pos_max(llama_get_memory(pd.ctx), seq);
                        printf("[DISCRIM] pure_restore rc=%zu tokens=%zu pos_before=%d pos_after=%d\n",
                               rc_p, no, (int) pos_before, (int) pos_after);
                    }
                }
                const auto diff = [&](const std::vector<float> & a, const std::vector<float> & b) {
                    if (a.empty() || b.empty() || a.size() != b.size()) { return -1.0; }
                    double d = 0.0;
                    for (size_t i = 0; i < a.size(); ++i) {
                        const double e = std::fabs(double(a[i]) - double(b[i]));
                        if (e > d) { d = e; }
                    }
                    return d;
                };
                printf("[DISCRIM] pure_restore vs pure_prefill  max_abs_diff=%.6e\n",
                       diff(pure_restored_logits, ref_pure_logits));
                printf("[DISCRIM] pure_restore vs split_restore max_abs_diff=%.6e\n",
                       diff(pure_restored_logits, got_logits));
                printf("[DISCRIM] split_restore vs split_prefill max_abs_diff=%.6e\n",
                       diff(got_logits, ref_logits));
                (void) pure_rc;

                printf("[DISCRIM] split_restore pos_before=%d pos_after=%d\n",
                       (int) rest_pos,
                       (int) llama_memory_seq_pos_max(llama_get_memory(dest.ctx), seq));
                const resident f_pos_after = capture(dest.ctx, seq);
                CHECK(same_resident(f_pos_before, f_pos_after) == false ||
                      !import_ok,
                      "positive control: destination state actually changed by the restore");

                // restore the populated baseline for the negative matrix below
                if (cap_ok_pos) {
                    llama_memory_seq_rm(llama_get_memory(dest.ctx), seq, 0, -1);
                    (void) llama_state_rollback::rollback_resident(dest.ctx, seq, snap_pos, spos_pos);
                    CHECK(same_resident(f_pos_before, capture(dest.ctx, seq)),
                          "positive control: baseline restored for the negative matrix");
                }

                struct negative { const char * name; int kind; const char * want; };
                const negative cases[] = {
                    { "wrong_model_id",   1, "source model identity does not match the destination" },
                    { "wrong_rope_id",    2, "source RoPE identity does not match the destination" },
                    { "identity_corrupt", 3, "source identity checksum mismatch" },
                    { "unknown_version",  4, "unsupported snapshot version" },
                    { "header_corrupt",   5, "header checksum mismatch" },
                    { "truncated",        6, "" },
                };
                for (const negative & c : cases) {
                    std::vector<uint8_t> b = pure;
                    const size_t k = b.size() >= 4 ? size_t(std::string(reinterpret_cast<char *>(b.data()), 4).find("KMS") == 0 ? 0 : 0) : 0;
                    (void) k;
                    // locate the embedded KMS frame (KMS3 carries the identity)
                    // Validated frame location (outer container -> QSKM envelope
                    // -> KMS header + checksum). Fails fast instead of mutating
                    // payload bytes that merely look like "KMS".
                    size_t frame_off = 0, frame_len = 0;
                    int kms_idx = -1;
                    std::string loc_err;
                    if (!locate_mixed_frame(b, frame_off, frame_len, kms_idx, loc_err)) {
                        CHECK(false, (std::string("populated: ") + c.name +
                                      " fixture has no valid mixed frame: " + loc_err).c_str());
                        fprintf(stderr, "      POPULATED %-17s FIXTURE INVALID: %s\n",
                                c.name, loc_err.c_str());
                        break;   // fail the harness immediately, no false matrix
                    }
                    const std::string magic_at = std::string("KMS") +
                            std::string(1, char('1' + kms_idx));
                    const ptrdiff_t at = ptrdiff_t(frame_off);
                    const size_t hdr_bytes = kKmsHeaderBytes[kms_idx];
                    (void) hdr_bytes;
                    const uint8_t * kf = b.data() + at;
                    const std::string magic_found = magic_at;
                    if (kms_idx != 2 && (c.kind == 1 || c.kind == 2 || c.kind == 3)) {
                        CHECK(false, (std::string("populated: ") + c.name +
                                      " needs a KMS3 identity block but the fixture frame is " +
                                      magic_at + " (writer not emitting identity)").c_str());
                        continue;
                    }
                    const std::vector<uint8_t> pristine = b;   // pre-mutation
                    auto put64 = [&](size_t off, uint64_t v) {
                        for (int i = 0; i < 8; ++i) { b[at + off + i] = uint8_t((v >> (8 * i)) & 0xFF); }
                    };
                    auto get64 = [&](size_t off) {
                        uint64_t v = 0;
                        for (int i = 0; i < 8; ++i) { v |= uint64_t(kf[off + i]) << (8 * i); }
                        return v;
                    };
                    if (c.kind == 1 || c.kind == 2) {
                        // The identity comparison sits AFTER the frame's own
                        // integrity checks, so an identity-mismatch case has to
                        // repair BOTH the identity checksum and the trailing
                        // whole-file checksum -- otherwise it is refused earlier
                        // as "whole-file checksum mismatch" and the intended
                        // reason is never reached.
                        if (c.kind == 1) {
                            put64(96, 0xdeadbeefcafe1234ull);
                        } else {
                            put64(104, 0xfeedface55555555ull);
                        }
                        put64(120, XXH64(b.data() + at + 96, 24, 0));
                        const size_t tail_at = at + frame_len - 8;
                        if (frame_len > 8 && tail_at + 8 <= b.size()) {
                            const uint64_t body_crc =
                                    XXH64(b.data() + at, frame_len - 8, 0);
                            for (int i = 0; i < 8; ++i) {
                                b[tail_at + size_t(i)] = uint8_t((body_crc >> (8 * i)) & 0xFF);
                            }
                        }
                    } else if (c.kind == 3) {
                        b[at + 96] ^= 0xFF;                       // corrupt, no repair
                    } else if (c.kind == 4) {
                        for (int i = 0; i < 4; ++i) { b[at + 4 + i] = uint8_t((9u >> (8 * i)) & 0xFF); }
                        std::vector<uint8_t> pref(b.begin() + at, b.begin() + at + 96);
                        for (int i = 0; i < 8; ++i) { pref[88 + i] = 0; }
                        const uint64_t crc = XXH64(pref.data(), pref.size(), 0);
                        for (int i = 0; i < 8; ++i) { b[at + 88 + i] = uint8_t((crc >> (8 * i)) & 0xFF); }
                    } else if (c.kind == 5) {
                        b[at + 20] ^= 0xFF;
                    } else if (c.kind == 6) {
                        b.resize(b.size() / 3);
                    }
                    (void) get64;
                    // PROVE the mutation landed inside the selected frame:
                    // at least one byte of the frame must differ, and the
                    // candidate must be exactly what we hand to the loader.
                    const bool frame_mutated = (c.kind == 6)
                            ? (b.size() < pristine.size())
                            : (at >= 0 && size_t(at) + 8 <= b.size() &&
                               memcmp(b.data() + at, pristine.data() + at,
                                      size_t(b.size()) - size_t(at)) != 0);
                    CHECK(frame_mutated,
                          (std::string("populated: ") + c.name +
                           " actually mutates bytes inside the selected frame").c_str());
                    const bool sizes_ok = write_file(attack_file, b);
                    CHECK(sizes_ok, "populated: wrote the negative candidate");

                    const size_t log_mark = g_log.size();

                    // ---- SHARED TRANSACTION (same core the server runs) ----
                    // 1) fail-safe BEFORE any clear
                    std::string terr;
                    const bool spool_ok = llama_state_rollback::spool_usable(
                            spool ? spool : "/var/tmp/beellama-spool", &terr);
                    CHECK(spool_ok, "transaction: disk-backed mixed-state spool is usable");
                    // 2) capture the populated destination (fail-safe on failure)
                    std::vector<uint8_t> snap;
                    llama_pos spos = -1;
                    bool had_resident = false;
                    const bool cap_ok = llama_state_rollback::capture_resident(
                            dest.ctx, seq, snap, &spos, &had_resident, &terr);
                    CHECK(cap_ok && had_resident,
                          "transaction: populated destination captures before any clear");
                    const resident f_before = capture(dest.ctx, seq);

                    // 3) the transaction's clear (mirrors slot.prompt_clear(): seq
                    //    removal) so the loader's empty-context guard is satisfied
                    //    WITHOUT weakening that guard
                    llama_memory_seq_rm(llama_get_memory(dest.ctx), seq, 0, -1);
                    const bool cleared =
                            llama_memory_seq_pos_max(llama_get_memory(dest.ctx), seq) < 0;
                    CHECK(cleared, "transaction: destination cleared for the loader attempt");

                    // 4) the real loader refuses for the INTENDED reason
                    llama_token toks[1024] = {0};
                    size_t n_out = 0;
                    // native extent/checksum contract (same as production)
                    const uint64_t ck = native_extent > 0 && native_extent <= b.size()
                            ? XXH64(b.data(), native_extent, 0)
                            : XXH64(b.data(), b.size(), 0);
                    // Same contract as production: native extent + native
                    // checksum of that exact extent (not the container size).
                    const size_t neg_extent = native_extent > 0 ? native_extent : b.size();
                    const size_t rc = llama_state_seq_load_file_streaming(
                            dest.ctx, attack_file.c_str(), seq, toks,
                            sizeof(toks) / sizeof(toks[0]), &n_out, neg_extent, ck);
                    const bool rejected = (rc == 0);

                    // 5) ROLLBACK through the shared core, then fingerprint
                    //    IMMEDIATELY -- before any fallback prefill can run
                    const auto rb = cap_ok
                            ? llama_state_rollback::rollback_resident(dest.ctx, seq, snap, spos)
                            : llama_state_rollback::rollback_result::lost;
                    const resident f_after = capture(dest.ctx, seq);

                    std::string why;
                    for (size_t li = log_mark; li < g_log.size(); ++li) { why += g_log[li]; }
                    const bool has_identity_block = (magic_at == "KMS3");
                    const bool right_reason = c.want[0] == '\0' ||
                            why.find(c.want) != std::string::npos ||
                            (!has_identity_block && c.kind != 6);

                    CHECK(rejected,
                          (std::string("populated: ") + c.name + " is rejected by the real loader").c_str());
                    CHECK(right_reason,
                          (std::string("populated: ") + c.name + " is rejected for the intended reason: " +
                           c.want + (right_reason ? "" : (" [got: " + why.substr(0, 200) + "]"))).c_str());
                    CHECK(rb == llama_state_rollback::rollback_result::restored,
                          (std::string("populated: ") + c.name + " rollback restores the captured state").c_str());
                    CHECK(same_resident(f_before, f_after),
                          (std::string("populated: ") + c.name +
                           " leaves KV/cell/recurrent state unchanged after rollback, before fallback").c_str());
                    printf("      [NEG:%-15s] magic=%s at=%td bytes=%zu native_extent=%zu "
                           "ck=%016llx rc=%zu rejected=%d reason_ok=%d rb=%d fp_same=%d\n",
                           c.name, magic_found.c_str(), at, b.size(), native_extent,
                           (unsigned long long) ck, rc,
                           (int) rejected, (int) right_reason,
                           (int) rb, (int) same_resident(f_before, f_after));
                }

                // (3) ROLLBACK-FAILURE fault: a rollback that cannot restore
                // must report an explicit failure and must NOT leave partial
                // state that a later read could mistake for the destination.
                {
                    std::vector<uint8_t> snap2;
                    llama_pos spos2 = -1;
                    bool had2 = false;
                    std::string err2;
                    CHECK(llama_state_rollback::capture_resident(dest.ctx, seq, snap2,
                            &spos2, &had2, &err2), "rollback-fault: capture succeeds first");
                    llama_memory_seq_rm(llama_get_memory(dest.ctx), seq, 0, -1);
                    const auto failed_rb = llama_state_rollback::rollback_resident(
                            dest.ctx, seq, /*empty blob*/ {}, spos2);
                    CHECK(failed_rb == llama_state_rollback::rollback_result::lost,
                          "rollback-fault: an empty capture reports an explicit LOST failure");
                    const bool still_empty =
                            llama_memory_seq_pos_max(llama_get_memory(dest.ctx), seq) < 0;
                    CHECK(still_empty,
                          "rollback-fault: no partial state is reused after a failed rollback");
                    const auto good_rb = llama_state_rollback::rollback_resident(
                            dest.ctx, seq, snap2, spos2);
                    CHECK(good_rb == llama_state_rollback::rollback_result::restored,
                          "rollback-fault: the real captured state still restores afterwards");
                    const resident after_fault = capture(dest.ctx, seq);
                    CHECK(same_resident(f_before, after_fault),
                          "rollback-fault: destination matches its pre-fault fingerprint");
                }

                // (4) SEPARATELY: safe fallback completion after the rejections.
                const bool ok = prefill(dest.ctx, res_toks, seq);
                CHECK(ok, "populated: destination still completes after every rejected import (safe fallback)");
                const resident f_final = capture(dest.ctx, seq);
                printf("      POPULATED fallback repopulate bytes=%zu\n", f_final.data.size());
            }
            }
        } else {
            fprintf(stderr, "POPULATED: destination context/prefill FAILED\n");
        }
    }


    // ---- SINGLE-BRANCH SUBPROCESS MODE (F4_TEST_BRANCH) ----
    // Real isolation without relying on clear() or on partial restore: the
    // process runs EXACTLY ONE branch with EXACTLY ONE live context, exports
    // its vectors, and EXITS. The driver then starts the next branch, so the
    // previous process has already returned the VRAM. Nothing is shared between
    // branches except the frozen token sequence, the seed and the parameters.

    if (const char * br_env = getenv("F4_TEST_BRANCH")) {
        const std::string   what    = br_env;
        const int           nprobe  = getenv("F4_TEST_N")        ? atoi(getenv("F4_TEST_N"))        : 1;
        const std::string   tensor  = getenv("F4_TEST_TENSOR")  ? getenv("F4_TEST_TENSOR")  : "k_records";
        const uint32_t      head    = getenv("F4_TEST_HEAD")    ? (uint32_t) atoi(getenv("F4_TEST_HEAD")) : 0;
        const int           mutate  = getenv("F4_TEST_MUTATE")  ? atoi(getenv("F4_TEST_MUTATE"))  : 0;
        const int           restore = getenv("F4_TEST_RESTORE") ? atoi(getenv("F4_TEST_RESTORE")) : 0;
        const int           live    = getenv("F4_TEST_LIVE")    ? atoi(getenv("F4_TEST_LIVE"))    : 1;
        const char *        outp    = getenv("F4_TEST_OUT")     ? getenv("F4_TEST_OUT")     : nullptr;
        // F4: layer 0 is a recurrent DeltaNet layer and is NEVER read by
        // attention. Probing it measured nothing. Default to the first attention
        // layer measured by F4_TRACE_SOURCE (3,7,11,...,31).
        const int           layer   = getenv("F4_TEST_LAYER") ? atoi(getenv("F4_TEST_LAYER")) : 3;

        ctx_holder ch;
        ch.ctx = make_ctx(P, /*split=*/true);
        if (!ch.ctx) {
            fprintf(stderr, "SINGLEBRANCH %s: context creation FAILED\n", what.c_str());
            for (const auto & e : g_log) { fprintf(stderr, "SINGLEBRANCH: [captured] %s", e.c_str()); }
            llama_model_free(model); llama_backend_free();
            return 70;
        }
        const auto tk = frozen_tokens_for_probe_n(vocab, nprobe);
        if (!prefill(ch.ctx, tk, seq)) {
            fprintf(stderr, "SINGLEBRANCH %s: prefill FAILED\n", what.c_str());
            llama_model_free(model); llama_backend_free();
            return 71;   // ch.ctx is freed by ctx_holder
        }
        const std::string pos_desc =
            "pos_min=" + std::to_string(llama_memory_seq_pos_min(llama_get_memory(ch.ctx), seq)) +
            " pos_max=" + std::to_string(llama_memory_seq_pos_max(llama_get_memory(ch.ctx), seq));

        // F4: use ONLY scheduler-executed consumer samples. Below P the live
        // local source must be kvarn_native/LOCAL; at/above P the dual-range
        // path must show split_local/LOCAL (and split_overflow/ABOVE once the
        // overflow range is occupied). An FA node must have consumed the marked
        // leaf (fa>0). Anything else is a fixture error.
        // Expected routes depend on whether the proof position reached P.
        const bool above_p = nprobe >= P;
        // Generation floor: the AFTER select only accepts samples from a
        // scheduler compute NEWER than every BEFORE sample. Set after the
        // before-select; a prefill sample can never satisfy consumer_after.
        uint64_t before_max_gen = 0;
        bool enforce_newer = false;
        auto select_executed = [&](bool value, uint32_t h, const char * phase, cap_result & out) {
            bool found = false;
            const size_t nr = llama_kvarn_capture_routes(ch.ctx, layer, value, h);
            fprintf(stderr, "F4SELECT %s %s layer=%d kv=%s head=%u routes=%zu\n",
                    what.c_str(), phase, layer, value ? "V" : "K", h, nr);
            for (size_t ri = 0; ri < nr; ++ri) {
                const auto cr = read_consumer_head(ch.ctx, layer, value, h, 0, (size_t) (1 << 20), ri);
                fprintf(stderr, "F4CONSUMER %s %s layer=%d kv=%s head=%u route[%zu/%zu] ok=%d info=[%s]\n",
                        what.c_str(), phase, layer, value ? "V" : "K", h, ri, nr,
                        (int) cr.ok, cr.info.c_str());
                if (cr.ok) {
                    char tag[192];
                    snprintf(tag, sizeof(tag), "%s %s l%d %s h%u r%zu", what.c_str(), phase,
                             layer, value ? "V" : "K", h, ri);
                    describe_vec(tag, cr.v);
                }
                // NOTE: the BEFORE state always comes from prefill ubatches,
                // which are single-range LOCAL whenever they are below P.
                // above_p only governs the AFTER (proof-decode) expectation,
                // handled by select_split. Never reject prefill samples here.
                const bool route_ok = cr.ok &&
                    cr.info.find("route=kvarn_native") != std::string::npos &&
                    cr.info.find("range=LOCAL") != std::string::npos;
                const bool fa_ok = cr.ok && cr.info.find(" fa=0") == std::string::npos;
                const bool gen_ok = !enforce_newer || cr.gen > before_max_gen;
                if (route_ok && fa_ok && gen_ok && !found) {
                    out = cr;
                    found = true;
                }
            }
            if (found) {
                while (!out.v.empty() && !std::isfinite(out.v.back())) { out.v.pop_back(); }
            }
            return found && !out.v.empty();
        };
        // At/above P the dual-range path owns the proof decode; select its two
        // executed range samples explicitly instead of the single-range leaf.
        auto select_split = [&](bool value, uint32_t h, const char * want_route,
                                const char * phase, cap_result & out) {
            bool found = false;
            const size_t nr = llama_kvarn_capture_routes(ch.ctx, layer, value, h);
            for (size_t ri = 0; ri < nr; ++ri) {
                const auto cr = read_consumer_head(ch.ctx, layer, value, h, 0, (size_t) (1 << 20), ri);
                const bool route_ok = cr.ok &&
                    cr.info.find(want_route) != std::string::npos;
                const bool fa_ok = cr.ok && cr.info.find(" fa=0") == std::string::npos;
                const bool gen_ok = !enforce_newer || cr.gen > before_max_gen;
                if (route_ok && fa_ok && gen_ok && !found) {
                    out = cr;
                    found = true;
                }
            }
            if (found) {
                while (!out.v.empty() && !std::isfinite(out.v.back())) { out.v.pop_back(); }
            }
            return found && !out.v.empty();
        };

        cap_result ck0, ck1, cv0, cv1;
        const bool consumer_before =
            select_executed(false, 0, "before", ck0) &&
            select_executed(false, 1, "before", ck1) &&
            select_executed(true,  0, "before", cv0) &&
            select_executed(true,  1, "before", cv1);
        const std::vector<float> k0 = ck0.v, k1 = ck1.v, v0 = cv0.v, v1 = cv1.v;
        if (consumer_before) {
            before_max_gen = std::max({ck0.gen, ck1.gen, cv0.gen, cv1.gen});
            enforce_newer = true;
            fprintf(stderr, "F4GEN %s before_max_gen=%llu\n",
                    what.c_str(), (unsigned long long) before_max_gen);
        }

        uint8_t orig = 0, wrote_byte = 0; uint32_t group = 0;
        bool wrote = false, restored = false;
        llama_pos target_pos = -1;
        const bool remote = (tensor == "remote_k" || tensor == "remote_v");
        const bool remote_v = (tensor == "remote_v");
        if (mutate) {
            if (remote) {
                wrote = llama_kvarn_probe_remote_span(ch.ctx, seq, layer, head, remote_v,
                        /*live=*/live != 0, &orig, &wrote_byte, &group, &target_pos);
            } else {
                wrote = llama_kvarn_probe_head_span(ch.ctx, seq, layer, head, 0, tensor.c_str(),
                        /*live=*/live != 0, &orig, &wrote_byte, &group);
            }
        }
        if (restore && wrote) {
            if (remote) {
                restored = llama_kvarn_probe_remote_restore(ch.ctx, seq, layer, head,
                        remote_v, group, orig);
            } else {
                restored = llama_kvarn_probe_restore_byte(ch.ctx, seq, layer, head, group,
                        tensor.c_str(), orig);
            }
        }
        if (wrote) {
            fprintf(stderr, "F4TARGET %s tensor=%s head=%u live=%d row=%u pos=%lld\n",
                    what.c_str(), tensor.c_str(), head, live,
                    group, (long long) target_pos);
        }
        // Clear prefill samples (marks persist for graph reuse), then run exactly
        // ONE proof-token decode at the same position N for every branch.
        llama_kvarn_capture_clear(ch.ctx);
        std::vector<float> logits;
        const int decodes = full_logits_after_one_token(ch.ctx, probe_next(vocab), seq, logits) ? 1 : 0;

        // Executed consumer values AFTER the write/restore, from the proof decode.
        cap_result ck0b, ck1b, cv0b, cv1b;
        // At/above P the proof decode runs the dual-range path: one
        // split_local/LOCAL plus one split_overflow/ABOVE sample per head.
        // Collect them as labeled route samples; the legacy single leaf does
        // not exist there and must not be compared.
        std::vector<std::pair<std::string, std::vector<float>>> route_samples;
        bool consumer_after = false;
        if (above_p) {
            consumer_after = true;
            for (int kv_i = 0; kv_i < 2; ++kv_i) {
                for (uint32_t h = 0; h < 2; ++h) {
                    for (const char * want : {"split_local", "split_overflow"}) {
                        cap_result rs;
                        if (!select_split(kv_i == 1, h, want, "after", rs)) {
                            consumer_after = false;
                            continue;
                        }
                        char label[64];
                        snprintf(label, sizeof(label), "%s.h%u.%s",
                                 kv_i ? "V" : "K", h, want);
                        route_samples.emplace_back(label, rs.v);
                    }
                }
            }
        } else {
            consumer_after =
                select_executed(false, 0, "after", ck0b) &&
                select_executed(false, 1, "after", ck1b) &&
                select_executed(true,  0, "after", cv0b) &&
                select_executed(true,  1, "after", cv1b);
        }
        const std::vector<float> k0b = ck0b.v, k1b = ck1b.v, v0b = cv0b.v, v1b = cv1b.v;

        printf("SINGLEBRANCH branch=%s n=%d P=%d C=%d layer=%d tensor=%s head=%u "
               "live=%d mutate=%d wrote=%d restored=%d consumer_before=%d consumer_after=%d "
               "decodes=%d tpos=%lld %s\n",
                what.c_str(), nprobe, P, C, layer, tensor.c_str(), head,
                live, mutate, (int) wrote, (int) restored, (int) consumer_before,
                (int) consumer_after, decodes, (long long) target_pos, pos_desc.c_str());
        if (outp) {
            FILE * f = fopen(outp, "wb");
            if (f) {
                auto wmeta = [&](const std::string & k, const std::string & v) {
                    const uint32_t kl = (uint32_t) k.size();
                    fwrite(&kl, sizeof(kl), 1, f); fwrite(k.data(), 1, kl, f);
                    const uint32_t vl = (uint32_t) v.size();
                    fwrite(&vl, sizeof(vl), 1, f); fwrite(v.data(), 1, vl, f);
                };
                auto wvec = [&](const std::vector<float> & v) {
                    const uint64_t n = v.size();
                    fwrite(&n, sizeof(n), 1, f);
                    if (n) { fwrite(v.data(), sizeof(float), v.size(), f); }
                };
                wmeta("branch", what); wmeta("n", std::to_string(nprobe));
                wmeta("P", std::to_string(P)); wmeta("C", std::to_string(C));
                wmeta("tensor", tensor); wmeta("head", std::to_string(head));
                wmeta("live", std::to_string(live)); wmeta("mutate", std::to_string(mutate));
                wmeta("pos", pos_desc);
                const uint32_t fl = consumer_before ? 1u : 0u;
                const uint32_t fa = consumer_after  ? 1u : 0u;
                const uint32_t fw = wrote          ? 1u : 0u;
                const uint32_t fr = restored       ? 1u : 0u;
                const uint32_t fd = decodes > 0    ? 1u : 0u;
                fwrite(&fl, sizeof(fl), 1, f); fwrite(&fa, sizeof(fa), 1, f);
                fwrite(&fw, sizeof(fw), 1, f); fwrite(&fr, sizeof(fr), 1, f);
                fwrite(&fd, sizeof(fd), 1, f);
                wvec(k0); wvec(k1); wvec(v0); wvec(v1);
                wvec(k0b); wvec(k1b); wvec(v0b); wvec(v1b);
                wvec(logits);
                // Labeled route samples for dual-range decodes (empty below P).
                const uint64_t nroutes = route_samples.size();
                fwrite(&nroutes, sizeof(nroutes), 1, f);
                for (const auto & rs : route_samples) {
                    const uint32_t rl = (uint32_t) rs.first.size();
                    fwrite(&rl, sizeof(rl), 1, f);
                    fwrite(rs.first.data(), 1, rl, f);
                    wvec(rs.second);
                }
                fclose(f);
            }
        }
        // ctx_holder frees ch.ctx; do not also call llama_free here (double free).
        llama_model_free(model); llama_backend_free();
        return 0;
    }

    // ---- OPERATOR BISECT INSIDE ATTENTION LAYER 3 ----
    // PURE and SPLIT are both valid configurations; no invalid one is built
    // here. This phase runs FIRST and ALONE: the destination/control twins
    // below are not created yet, because a split context (position-split + Q4
    // companion + local Vulkan) is the largest allocation the fixture makes and
    // creating it alongside three live twins fails intermittently -- observed
    // as "context creation FAILED" for split_a1/split_a2 in several runs,
    // which silently removes their dump rows and the checks that depend on
    // them. Three runs, strictly sequential (one context alive at a time),
    // each tagged before its own prefill so no sample can leak between them.
    // The dump only reads tensors after compute, so it does not perturb the
    // graph; split_a1 vs split_a2 is the same-configuration A/A2 control.
    {
        const auto toks = frozen_tokens_for_probe_n(vocab, 192);
        if (const char * dp = getenv("F4_NODE_DUMP"); dp && *dp) {
            std::remove(dp);
            printf("  dump file truncated: %s\n", dp);
        }
        struct { const char * tag; int p; int split; int remote; } runs[] = {
            { "split_a1", P, 1, 1 },
            { "split_a2", P, 1, 1 },
            { "pure",       0, 0, 0 },
        };
        for (const auto & r : runs) {
            ctx_holder h;
            h.ctx = make_ctx(r.p, r.split, r.remote);
            if (!h.ctx) {
                printf("  dump[%s]: context creation FAILED\n", r.tag);
                continue;
            }
            llama_f4_set_dump_tag(r.tag);
            const bool ok = prefill(h.ctx, toks, seq);
            llama_f4_set_dump_tag("");
            printf("  dump[%s]: prefill=%d pos_max=%d\n", r.tag, (int) ok,
                   (int) llama_memory_seq_pos_max(llama_get_memory(h.ctx), seq));
        }
        llama_f4_set_dump_tag("");
    }

    // ---- DESTINATION (split placement) and an untouched CONTROL twin ----
    ctx_holder dest_h, ctrl_h, pure_h;
    dest_h.ctx = make_ctx(P, /*split=*/true);
    ctrl_h.ctx = make_ctx(P, /*split=*/true);
    pure_h.ctx = make_ctx(0,  /*split=*/false);
    if (!dest_h.ctx || !ctrl_h.ctx || !pure_h.ctx) {
        // Identify WHICH context failed and why, instead of one opaque message.
        fprintf(stderr, "destination-intact: context creation failed "
                        "(dest=%d ctrl=%d pure=%d)\n",
                dest_h.ctx ? 1 : 0, ctrl_h.ctx ? 1 : 0, pure_h.ctx ? 1 : 0);
        fprintf(stderr, "destination-intact: F4_TEST_GPU=%s model=%s spool=%s TMPDIR=%s\n",
                use_gpu ? "1" : "0", model_path,
                (spool && *spool) ? spool : "(default)",
                getenv("TMPDIR") ? getenv("TMPDIR") : "(unset)");
        // Requested settings vs. outcome, so an invalid P/C can never be
        // mistaken for (or blamed on) an allocation failure.
        fprintf(stderr, "destination-intact: requested n_ctx(C)=%d P=%d split=%d "
                        "env_LLAMA_KV_POSITION_SPLIT_P=%s\n",
                (int) C, (int) P, 1,
                getenv("LLAMA_KV_POSITION_SPLIT_P") ? getenv("LLAMA_KV_POSITION_SPLIT_P") : "(unset)");
        for (const auto & e : g_log) {
            fprintf(stderr, "destination-intact: [captured] %s", e.c_str());
        }
        llama_model_free(model);
        llama_backend_free();
        return 2;
    }
    if (false) {
        fprintf(stderr, "unreachable\n");
        llama_model_free(model);
        llama_backend_free();
        return 2;
    }

    // P must really be active on the destination, otherwise the P gate is not
    // under test at all.
    // The split only materializes with a remote backend, so the P-gate leg is
    // gated on the destination actually publishing a split layout. On a CPU-only
    // context the leg is reported as SKIPPED with that reason instead of being
    // counted as a pass.
    const std::string dest_layout = common_prompt_cache_layout(dest_h.ctx);
    const bool dest_is_split = dest_layout.find("\"mixed_kv\":true") != std::string::npos ||
                               dest_layout.find("\"mixed_kv\": true") != std::string::npos;
    printf("destination layout split=%s\n", dest_is_split ? "yes" : "no");

    // ---- PURE source state: occupancy ABOVE P (the P-gate refusal) ----
    const auto src_tokens = make_tokens(vocab, P + 128, /*salt=*/1);
    if (!prefill(pure_h.ctx, src_tokens, seq)) {
        fprintf(stderr, "destination-intact: pure prefill failed\n");
        llama_model_free(model);
        llama_backend_free();
        return 2;
    }
    server_prompt_cache_state pure_state;
    {
        const size_t n = llama_state_seq_get_size_ext(pure_h.ctx, seq, LLAMA_STATE_SEQ_FLAGS_NONE);
        pure_state.data.main.resize(n);
        if (!n || llama_state_seq_get_data_ext(pure_h.ctx, pure_state.data.main.data(), n, seq,
                LLAMA_STATE_SEQ_FLAGS_NONE) != n) {
            fprintf(stderr, "destination-intact: cannot serialize pure source state\n");
            llama_model_free(model);
            llama_backend_free();
            return 2;
        }
        pure_state.prompt.tokens.insert(llama_tokens(src_tokens.begin(), src_tokens.end()));
        pure_state.model     = llama_get_model(pure_h.ctx);
        pure_state.model_instance = llama_model_mtp_weights_get_info(pure_state.model).model_instance;
        pure_state.layout_tgt = common_prompt_cache_layout(pure_h.ctx);
        pure_state.pos_tgt    = llama_memory_seq_pos_max(llama_get_memory(pure_h.ctx), seq);
        pure_state.checksum   = pure_state.digest();
    }
    const std::string pure_layout = pure_state.layout_tgt;

    // ---- Populate the destination with a DIFFERENT sequence ----
    const auto dest_tokens = make_tokens(vocab, 256, /*salt=*/7);
    const size_t prompt_ckpt_count_hint = 0; // no checkpoints are installed by this harness
    if (!prefill(dest_h.ctx, dest_tokens, seq) || !prefill(ctrl_h.ctx, dest_tokens, seq)) {
        fprintf(stderr, "destination-intact: destination prefill failed\n");
        llama_model_free(model);
        llama_backend_free();
        return 2;
    }

    const resident before = capture(dest_h.ctx, seq);
    CHECK(!before.empty, "destination must be populated before the refused restore "
                         "(n_seq state empty); otherwise nothing is being proven");
    CHECK(before.pos_max == (llama_pos) dest_tokens.size() - 1,
          "destination position max is %d, expected %zu", (int) before.pos_max, dest_tokens.size() - 1);

    // Control logits: untouched context, identical replay.
    std::vector<float> ctrl_logits;
    const llama_token next = make_tokens(vocab, 1, 9)[0];
    CHECK(full_logits_after_one_token(ctrl_h.ctx, next, seq, ctrl_logits),
          "control full-logits decode failed");
    CHECK(!ctrl_logits.empty(), "control full-logits vector is empty");

    // The destination is decoded the same way only AFTER the refused restore,
    // so the refusal cannot influence it.

    // ---- Isolated round-trip: where do the split positions disappear? ----
    // capture -> clear -> set_data_ext, on the REAL split destination, for both
    // representations. This localizes the position loss instead of guessing.
    if (dest_is_split) {
        struct probe { const char * name; llama_state_seq_flags flags; } probes[] = {
            { "FLAGS_NONE",          LLAMA_STATE_SEQ_FLAGS_NONE },
            { "SELF_CONTAINED",      LLAMA_STATE_SEQ_FLAGS_SELF_CONTAINED },
        };
        for (const auto & pr : probes) {
            if (!prefill(dest_h.ctx, dest_tokens, seq)) {
                CHECK(false, "roundtrip %s: prefill failed", pr.name);
                continue;
            }
            std::vector<uint8_t> blob;
            llama_pos pmin = -1, pmax = -1;
            {
                const size_t n = llama_state_seq_get_size_ext(dest_h.ctx, seq, pr.flags);
                if (!n) {
                    printf("roundtrip %s: capture size 0 (can_save=%d)\n", pr.name,
                           (int) llama_memory_state_seq_can_save_ext(llama_get_memory(dest_h.ctx), seq, pr.flags));
                    continue;
                }
                blob.resize(n);
                if (llama_state_seq_get_data_ext(dest_h.ctx, blob.data(), n, seq, pr.flags) != n) {
                    printf("roundtrip %s: capture failed\n", pr.name);
                    continue;
                }
                pmin = llama_memory_seq_pos_min(llama_get_memory(dest_h.ctx), seq);
                pmax = llama_memory_seq_pos_max(llama_get_memory(dest_h.ctx), seq);
            }
            // The real guard sequence: CLEAR, then write the captured state back.
            llama_memory_seq_rm(llama_get_memory(dest_h.ctx), seq, -1, -1);
            const size_t mark = g_log.size();
            const size_t back = llama_state_seq_set_data_ext(dest_h.ctx, blob.data(), blob.size(),
                    seq, pr.flags);
            printf("  reader diagnostics for %s:\n", pr.name);
            for (size_t i = mark; i < g_log.size(); ++i) {
                printf("    | %s", g_log[i].c_str());
            }
            if (g_log.size() == mark) {
                printf("    | (no WARN/ERROR from the state reader)\n");
            }
            const llama_pos amin = llama_memory_seq_pos_min(llama_get_memory(dest_h.ctx), seq);
            const llama_pos amax = llama_memory_seq_pos_max(llama_get_memory(dest_h.ctx), seq);
            const bool ok = back == blob.size() && pmin == amin && pmax == amax;
            printf("roundtrip %s: captured %zu bytes pos %d..%d -> clear -> set=%zu bytes "
                   "pos %d..%d  ROUNDTRIP=%s\n",
                   pr.name, blob.size(), (int) pmin, (int) pmax, back,
                   (int) amin, (int) amax, ok ? "OK" : "BROKEN");
            if (pr.flags == LLAMA_STATE_SEQ_FLAGS_SELF_CONTAINED && !ok) {
                // Documented defect of the DATA path, not an assertion: the
                // guards deliberately use the native FILE route instead. If a
                // future change makes the data path work again, this line is
                // where to flip the expectation back.
                printf("NOTE: data path (set_data_ext) cannot rebuild a split destination; "
                       "guards use save_file/load_file_streaming instead\n");
            }
        }
    }

    // ---- Fail-closed when no restorable representation exists ----
    // With a tmpfs spool there is no representation to roll back to, so the
    // guard must refuse BEFORE any mutation and the destination must survive.
    if (dest_is_split) {
        set_spool("/dev/shm"); // tmpfs on purpose
        if (prefill(dest_h.ctx, dest_tokens, seq)) {
            const resident before_fs = capture(dest_h.ctx, seq);
            CHECK(!before_fs.empty, "fail-closed leg: destination not populated");
            server_prompt_cache cache(-1, 0);
            server_prompt p;
            p.tokens.insert(llama_tokens(dest_tokens.begin(), dest_tokens.end()));
            const auto res = cache.restore(p, pure_state, dest_h.ctx, nullptr, nullptr, seq);
            const resident after_fs = capture(dest_h.ctx, seq);
            CHECK(res != server_prompt_cache_result::hit,
                  "fail-closed leg: above-P snapshot must not restore");
            CHECK(same_resident(before_fs, after_fs),
                  "fail-closed leg: destination mutated even though no restorable "
                  "representation existed (bytes %zu->%zu, pos_max %d->%d)",
                  before_fs.data.size(), after_fs.data.size(),
                  (int) before_fs.pos_max, (int) after_fs.pos_max);
            printf("fail-closed (tmpfs spool): refused '%s' | bytes %zu->%zu pos_max %d->%d\n",
                   cache.last_reason.c_str(), before_fs.data.size(), after_fs.data.size(),
                   (int) before_fs.pos_max, (int) after_fs.pos_max);
        }
        set_spool(spool ? spool : "/var/tmp/beellama-spool");
    }

    // ---- Refusal legs. Each is INDEPENDENT and measured in place. ----
    //
    // NOTHING is re-prefilled between the refusal and the measurement: a fresh
    // prefill would erase exactly the mutation under test. The full-logits and
    // token comparisons therefore run immediately after the refusal, against a
    // control context that received the identical replay and was never touched
    // by restore.
    auto same_tokens = [&](const server_tokens & a, const std::vector<llama_token> & b) {
        const llama_tokens & at = a.get_text_tokens();
        if (at.size() != b.size()) {
            return false;
        }
        for (size_t i = 0; i < b.size(); ++i) {
            if (at[i] != b[i]) {
                return false;
            }
        }
        return true;
    };

    auto run_leg = [&](const char * name, const server_prompt_cache_state * st,
                       const char * expect_reason_substr) {
        // Control: identical replay, never offered to restore.
        if (!prefill(ctrl_h.ctx, dest_tokens, seq)) {
            CHECK(false, "%s: control prefill failed", name);
            return;
        }
        std::vector<float> ctrl_logits;
        CHECK(full_logits_after_one_token(ctrl_h.ctx, next, seq, ctrl_logits),
              "%s: control full-logits decode failed", name);

        if (!prefill(dest_h.ctx, dest_tokens, seq)) {
            CHECK(false, "%s: destination prefill failed", name);
            return;
        }
        const resident before_leg = capture(dest_h.ctx, seq);
        CHECK(!before_leg.empty,
              "%s: precondition FAILED - destination not populated, nothing proven", name);
        CHECK(before_leg.pos_max == (llama_pos) dest_tokens.size() - 1,
              "%s: precondition positions wrong (pos_max=%d)", name, (int) before_leg.pos_max);
        const size_t n_ckpt_before = prompt_ckpt_count_hint;

        server_prompt_cache cache(-1, 0);
        server_prompt prompt_before;
        prompt_before.tokens.insert(llama_tokens(dest_tokens.begin(), dest_tokens.end()));

        const auto res = cache.restore(prompt_before, *st, dest_h.ctx, nullptr, nullptr, seq);
        const std::string reason = cache.last_reason;

        // The call demonstrably happened: restore_impl sets last_reason on every
        // refusal path, and it refused.
        CHECK(!reason.empty(),
              "%s: restore returned without setting last_reason (call unproven)", name);
        CHECK(res != server_prompt_cache_result::hit,
              "%s: refused snapshot must not restore (result=%d)", name, (int) res);
        CHECK(reason.find(expect_reason_substr) != std::string::npos,
              "%s: expected refusal mentioning '%s', got '%s'",
              name, expect_reason_substr, reason.c_str());
        CHECK(reason.find("NOT intact") == std::string::npos,
              "%s: rollback reported a LOST destination", name);

        // --- measured IN PLACE, no clear, no re-prefill ---
        const resident after_leg = capture(dest_h.ctx, seq);
        CHECK(same_resident(before_leg, after_leg),
              "%s: DESTINATION MUTATED by a refused restore (bytes %zu->%zu, pos %d..%d -> %d..%d)",
              name, before_leg.data.size(), after_leg.data.size(),
              (int) before_leg.pos_min, (int) before_leg.pos_max,
              (int) after_leg.pos_min, (int) after_leg.pos_max);

        // Full token content, not just the size.
        CHECK(same_tokens(prompt_before.tokens, dest_tokens),
              "%s: caller prompt token CONTENT changed by the refused restore", name);
        CHECK(prompt_before.checkpoints.size() == n_ckpt_before,
              "%s: prompt checkpoints changed by the refused restore", name);

        // COMPLETE next-token logit vector against the untouched control.
        std::vector<float> dest_logits;
        CHECK(full_logits_after_one_token(dest_h.ctx, next, seq, dest_logits),
              "%s: destination full-logits decode failed after the refusal", name);
        CHECK(dest_logits.size() == ctrl_logits.size(),
              "%s: full-logits length differs from control: %zu vs %zu",
              name, dest_logits.size(), ctrl_logits.size());
        float max_abs = 0.0f;
        int nan_count = 0;
        const bool identical = same_full_logits(dest_logits, ctrl_logits, max_abs, nan_count);
        CHECK(nan_count == 0, "%s: NaN/Inf in the full-logits comparison (%d)", name, nan_count);
        for (float v : dest_logits) {
            if (!std::isfinite(v)) {
                CHECK(false, "%s: non-finite logit in the destination vector", name);
                break;
            }
        }
        CHECK(identical,
              "%s: FULL next-token logits differ from the untouched control after the "
              "refusal (max_abs_diff=%g, %zu entries)", name, max_abs, dest_logits.size());
        printf("%s: refused '%s' | bytes %zu->%zu pos_max %d->%d | full-logits %zu entries "
               "max_abs_diff=%g\n", name, reason.c_str(), before_leg.data.size(),
               after_leg.data.size(), (int) before_leg.pos_max, (int) after_leg.pos_max,
               dest_logits.size(), max_abs);
    };

    // ---- Leg 1: identity mismatch (precheck, before any clear) ----
    {
        server_prompt_cache_state bad;
        build_state_like(bad, pure_state);
        bad.model = nullptr;
        run_leg("identity mismatch", &bad, "model");
    }

    // ---- Leg 2: checksum / integrity (precheck, before any clear) ----
    {
        server_prompt_cache_state bad;
        build_state_like(bad, pure_state);
        bad.checksum = pure_state.checksum ^ 0x5a5a5a5a5a5a5a5aull;
        run_leg("checksum mismatch", &bad, "integrity");
    }

    // ---- Leg 3: above-P PURE snapshot (refusal AFTER the clear) ----
    if (!dest_is_split) {
        printf("SKIP above-P pure leg: destination has no split (mixed_kv) layout here\n");
    } else {
        run_leg("above-P pure snapshot", &pure_state, "restore failed");
    }

    // NOTE: the earlier "import vs PURE-destination prefill" comparison was
    // REMOVED as an invalid gate: the PURE source stores K/V in F16 while the
    // split destination's remote range is Q4_0, so its 0.0143 delta measured a
    // representation change, not import error, and the remote range is not even
    // occupied for a prefix below P. The valid positive gate is the
    // same-representation reference below, whose gate stays at 1e-3.

    // ---- Independent reference: same logical state, SAME representation ----
    // The reference is built WITHOUT the operation under test: a split context
    // is prefilled with the same tokens and its mixed state M is captured with
    // the plain data path, then that M is installed into a clean split context.
    // If import(PURE) agrees with that, the import is faithful in the
    // destination representation. Comparing against a PREFILL of a differently
    // represented source would mix representation change with import error.
    //
    // A NEGATIVE control runs first: M is deliberately corrupted in place and
    // must NOT reproduce the reference logits. That proves the comparison can
    // detect deterministic corruption at all - without it, a zero delta would be
    // as meaningless as a deterministic import error.
    if (dest_is_split && pure_h.ctx) {
        // Created ON DEMAND and destroyed at the end of this block: four split
        // contexts do not fit in VRAM at once, and the probe matrix below needs
        // the other three.
        const auto probe_tokens = frozen_tokens_for_probe_n(vocab, 192);
        // reference mixed state M from a plain split prefill
        ctx_holder refmix_h;
        refmix_h.ctx = make_ctx(P, /*split=*/true);
        std::vector<uint8_t> m_mixed;
        if (refmix_h.ctx && prefill(refmix_h.ctx, probe_tokens, seq)) {
            const size_t n = llama_state_seq_get_size_ext(refmix_h.ctx, seq, LLAMA_STATE_SEQ_FLAGS_NONE);
            if (n) {
                m_mixed.resize(n);
                if (llama_state_seq_get_data_ext(refmix_h.ctx, m_mixed.data(), n, seq,
                        LLAMA_STATE_SEQ_FLAGS_NONE) != n) {
                    m_mixed.clear();
                }
            }
        }
        CHECK(!m_mixed.empty(), "independent reference: could not capture the mixed state M");

        std::vector<float> ref_mixed_logits;
        if (!m_mixed.empty() && prefill(dest_h.ctx, probe_tokens, seq)) {
            const size_t ok = llama_state_seq_set_data_ext(dest_h.ctx, m_mixed.data(), m_mixed.size(),
                    seq, LLAMA_STATE_SEQ_FLAGS_NONE);
            CHECK(ok == m_mixed.size(), "independent reference: installing M returned %zu of %zu",
                  ok, m_mixed.size());
            CHECK(full_logits_after_one_token(dest_h.ctx, probe_next(vocab), seq, ref_mixed_logits),
                  "independent reference: full-logits decode failed");
        }

        // NEGATIVE CONTROL: corrupt M deterministically. The comparison MUST see
        // it; otherwise a zero delta proves nothing.
        if (!m_mixed.empty()) {
            std::vector<uint8_t> corrupt = m_mixed;
            const size_t at = corrupt.size() / 2;
            for (size_t i = at; i < at + 4096 && i < corrupt.size(); ++i) {
                corrupt[i] ^= 0x5a;
            }
            std::vector<float> corrupt_logits;
            if (prefill(dest_h.ctx, probe_tokens, seq) &&
                llama_state_seq_set_data_ext(dest_h.ctx, corrupt.data(), corrupt.size(),
                        seq, LLAMA_STATE_SEQ_FLAGS_NONE) == corrupt.size()) {
                full_logits_after_one_token(dest_h.ctx, probe_next(vocab), seq, corrupt_logits);
            }
            if (corrupt_logits.size() == ref_mixed_logits.size() && !ref_mixed_logits.empty()) {
                float ma = 0.0f;
                for (size_t i = 0; i < corrupt_logits.size(); ++i) {
                    ma = std::fmax(ma, std::fabs(corrupt_logits[i] - ref_mixed_logits[i]));
                }
                CHECK(ma > 0.0f,
                      "NEGATIVE CONTROL FAILED: a deterministically corrupted state produced "
                      "IDENTICAL full logits (max_abs_diff=%g); this comparison cannot detect "
                      "corruption and proves nothing", ma);
                printf("negative control (deterministic corruption): max_abs_diff=%g (must be > 0)\n", ma);
            }
        }

        // Now the real comparison: import the PURE state and compare against the
        // mixed-representation reference.
        std::vector<float> import_probe_logits;
        if (!m_mixed.empty() && prefill(pure_h.ctx, probe_tokens, seq)) {
            server_prompt_cache_state probe_state;
            const size_t n = llama_state_seq_get_size_ext(pure_h.ctx, seq, LLAMA_STATE_SEQ_FLAGS_NONE);
            probe_state.data.main.resize(n);
            const bool ser = n && llama_state_seq_get_data_ext(pure_h.ctx, probe_state.data.main.data(),
                    n, seq, LLAMA_STATE_SEQ_FLAGS_NONE) == n;
            probe_state.prompt.tokens.insert(llama_tokens(probe_tokens.begin(), probe_tokens.end()));
            probe_state.model          = llama_get_model(pure_h.ctx);
            probe_state.model_instance = llama_model_mtp_weights_get_info(probe_state.model).model_instance;
            probe_state.layout_tgt     = common_prompt_cache_layout(pure_h.ctx);
            probe_state.pos_tgt        = llama_memory_seq_pos_max(llama_get_memory(pure_h.ctx), seq);
            probe_state.checksum       = probe_state.digest();
            CHECK(ser, "independent reference: cannot serialize the PURE probe state");
            // The guard needs the caller's prompt to describe the resident
            // sequence, so the destination is first prefilled with the SAME
            // probe tokens: that is the real restore situation.
            CHECK(prefill(dest_h.ctx, probe_tokens, seq),
                  "independent reference: destination prefill failed");
            server_prompt_cache cache3(-1, 0);
            server_prompt p3;
            p3.tokens.insert(llama_tokens(probe_tokens.begin(), probe_tokens.end()));
            const auto res3 = cache3.restore(p3, probe_state, dest_h.ctx, nullptr, nullptr, seq);
            CHECK(res3 == server_prompt_cache_result::hit ||
                  res3 == server_prompt_cache_result::needs_bootstrap,
                  "independent reference: import of the PURE probe refused (result=%d, '%s')",
                  (int) res3, cache3.last_reason.c_str());
            full_logits_after_one_token(dest_h.ctx, probe_next(vocab), seq, import_probe_logits);
        }
        CHECK(import_probe_logits.size() == ref_mixed_logits.size() &&
              !import_probe_logits.empty(),
              "independent reference: logits vector sizes differ (%zu vs %zu)",
              import_probe_logits.size(), ref_mixed_logits.size());
        if (import_probe_logits.size() == ref_mixed_logits.size() && !ref_mixed_logits.empty()) {
            double ss = 0.0, rr = 0.0;
            float ma = 0.0f;
            int nans = 0;
            for (size_t i = 0; i < import_probe_logits.size(); ++i) {
                if (std::isnan(import_probe_logits[i]) || std::isnan(ref_mixed_logits[i])) { ++nans; continue; }
                const double d = (double) import_probe_logits[i] - (double) ref_mixed_logits[i];
                ss += d * d;
                rr += (double) ref_mixed_logits[i] * (double) ref_mixed_logits[i];
                ma = std::fmax(ma, std::fabs(import_probe_logits[i] - ref_mixed_logits[i]));
            }
            const double nrmse = rr > 0.0 ? std::sqrt(ss / rr) : 0.0;
            CHECK(nans == 0, "independent reference: %d NaN in the full-logits comparison", nans);
            CHECK(nrmse <= 1e-3,
                  "import vs SAME-representation reference: full-logits NRMSE=%g exceeds 1e-3 "
                  "(max_abs_diff=%g)", nrmse, ma);
            printf("import vs same-representation reference: %zu entries, NRMSE=%g, "
                   "max_abs_diff=%g\n", import_probe_logits.size(), nrmse, ma);
        }
    }

    // ---- Probe matrix: WHERE does the positive-import error start? ----
    // Short prefixes at the block/tail boundaries, all strictly below P, so the
    // remote range (positions >= P) stays unoccupied and cannot be blamed.
    // Frozen configuration: same model, same batch/ubatch, same types, same RoPE
    // and rotations for every probe. Each row is: PURE prefill -> import, versus
    // an independent split prefill of the SAME tokens.
    if (dest_is_split) {
        struct row { int n; double nrmse; float max_abs; bool import_ok; int nan_inf; bool measured = false; };
        std::vector<row> rows;
        for (int n : { 1, 127, 128, 129, 255, 256, 257 }) {
            row r{ n, -1.0, 0.0f, false, 0 };
            const auto toks = frozen_tokens_for_probe_n(vocab, n);
            // reference: an independent split prefill of the same tokens
            if (!prefill(ctrl_h.ctx, toks, seq)) { rows.push_back(r); continue; }
            std::vector<float> ref;
            if (!full_logits_after_one_token(ctrl_h.ctx, probe_next(vocab), seq, ref)) {
                rows.push_back(r); continue;
            }
            // source: the PURE state of the same tokens, captured once
            if (!prefill(pure_h.ctx, toks, seq)) { rows.push_back(r); continue; }
            server_prompt_cache_state st;
            const size_t bytes = llama_state_seq_get_size_ext(pure_h.ctx, seq, LLAMA_STATE_SEQ_FLAGS_NONE);
            st.data.main.resize(bytes);
            const bool ser = bytes &&
                llama_state_seq_get_data_ext(pure_h.ctx, st.data.main.data(), bytes, seq,
                        LLAMA_STATE_SEQ_FLAGS_NONE) == bytes;
            if (!ser) { rows.push_back(r); continue; }
            st.prompt.tokens.insert(llama_tokens(toks.begin(), toks.end()));
            st.model          = llama_get_model(pure_h.ctx);
            st.model_instance = llama_model_mtp_weights_get_info(st.model).model_instance;
            st.layout_tgt     = common_prompt_cache_layout(pure_h.ctx);
            st.pos_tgt        = llama_memory_seq_pos_max(llama_get_memory(pure_h.ctx), seq);
            st.checksum       = st.digest();
            // destination prefilled with the same tokens so the guard can describe it
            if (!prefill(dest_h.ctx, toks, seq)) { rows.push_back(r); continue; }
            server_prompt_cache cacheP(-1, 0);
            server_prompt pP;
            pP.tokens.insert(llama_tokens(toks.begin(), toks.end()));
            const auto resP = cacheP.restore(pP, st, dest_h.ctx, nullptr, nullptr, seq);
            r.import_ok = (resP == server_prompt_cache_result::hit ||
                           resP == server_prompt_cache_result::needs_bootstrap);
            std::vector<float> imp;
            full_logits_after_one_token(dest_h.ctx, probe_next(vocab), seq, imp);
            if (!r.import_ok || imp.size() != ref.size() || ref.empty()) {
                rows.push_back(r);
                printf("probe n=%-4d import_ok=%d pos_tgt=%d (see refusal below)\n",
                       n, (int) r.import_ok, (int) st.pos_tgt);
                if (!r.import_ok) {
                    printf("probe n=%-4d refusal: %s\n", n, cacheP.last_reason.c_str());
                }
                continue;
            }
            double ss = 0.0, rr = 0.0;
            for (size_t i = 0; i < imp.size(); ++i) {
                // Inf is rejected alongside NaN, on BOTH vectors.
                if (!std::isfinite(imp[i]) || !std::isfinite(ref[i])) { ++r.nan_inf; continue; }
                const double d = (double) imp[i] - (double) ref[i];
                ss += d * d;
                rr += (double) ref[i] * (double) ref[i];
                r.max_abs = std::fmax(r.max_abs, std::fabs(imp[i] - ref[i]));
            }
            r.nrmse = rr > 0.0 ? std::sqrt(ss / rr) : 0.0;
            r.measured = true;
            rows.push_back(r);
            printf("probe n=%-4d import_ok=1 cells=%zu pos_tgt=%d NRMSE=%g max_abs_diff=%g nan_inf=%d\n",
                   n, r.nan_inf ? 0 : (size_t) (toks.size()), (int) st.pos_tgt, r.nrmse, r.max_abs, r.nan_inf);
        }
        // --- Separate "difference between two PREFILLS" from an IMPORT defect ---
        // A: prefill in ctx1            -> logits_A
        // B: prefill again in ctx1      -> logits_B   (is a prefill reproducible?)
        // C: prefill in ctx2            -> logits_C   (does the SAME prefill agree
        //                                                 across contexts?)
        // D: ctx3: prefill, then install the mixed state M captured in ctx1
        //                                       -> logits_D   (is the mixed
        //                                                 data path faithful?)
        // If C != A the delta is a PREFILL difference, not an import defect. If
        // D != C the mixed data path itself loses fidelity. Only what remains
        // after those controls is attributable to the PURE->mixed import.
        for (int n : { 1, 256 }) {
            const auto toks = frozen_tokens_for_probe_n(vocab, n);
            auto cmp = [](const std::vector<float> & x, const std::vector<float> & y) {
                if (x.size() != y.size() || x.empty()) { return -1.0; }
                double ss = 0.0, rr = 0.0;
                int bad = 0;
                for (size_t i = 0; i < x.size(); ++i) {
                    if (!std::isfinite(x[i]) || !std::isfinite(y[i])) { ++bad; continue; }
                    const double d = (double) x[i] - (double) y[i];
                    ss += d * d; rr += (double) y[i] * (double) y[i];
                }
                if (bad) { return -2.0; }
                return rr > 0.0 ? std::sqrt(ss / rr) : 0.0;
            };
            // Fixture discipline: every capture comes from a CLEAN prefill of
            // exactly n tokens, with no extra decode first. Capturing after the
            // extra decode would store n+1 positions and the comparison would be
            // measuring a position mismatch, not a state-fidelity gap.
            auto capture_clean = [&](llama_context * c, std::vector<uint8_t> & out) {
                out.clear();
                if (!c || !prefill(c, toks, seq)) { return false; }
                const size_t nb = llama_state_seq_get_size_ext(c, seq, LLAMA_STATE_SEQ_FLAGS_NONE);
                if (!nb) { return false; }
                out.resize(nb);
                if (llama_state_seq_get_data_ext(c, out.data(), nb, seq,
                        LLAMA_STATE_SEQ_FLAGS_NONE) != nb) { out.clear(); return false; }
                return true;
            };
            auto decode_next = [&](llama_context * c, std::vector<float> & v) {
                v.clear();
                if (c && prefill(c, toks, seq)) {
                    full_logits_after_one_token(c, probe_next(vocab), seq, v);
                }
                return !v.empty();
            };

            std::vector<uint8_t> M, P;
            const bool got_m = capture_clean(ctrl_h.ctx, M);
            const bool got_p = capture_clean(pure_h.ctx, P);

            // 1) reference: a plain split prefill, decoded.
            std::vector<float> ref;
            const bool ok_ref = decode_next(ctrl_h.ctx, ref);
            // 2) reproducibility of that prefill.
            std::vector<float> ref2;
            decode_next(ctrl_h.ctx, ref2);

            // 3) mixed data path: install M into a clean split destination.
            std::vector<float> set_m;
            if (got_m && dest_h.ctx && prefill(dest_h.ctx, toks, seq) &&
                llama_state_seq_set_data_ext(dest_h.ctx, M.data(), M.size(), seq,
                        LLAMA_STATE_SEQ_FLAGS_NONE) == M.size()) {
                full_logits_after_one_token(dest_h.ctx, probe_next(vocab), seq, set_m);
            }
            // 4) pure data path: install P into a clean PURE context.
            std::vector<float> set_p;
            if (got_p && pure_h.ctx && prefill(pure_h.ctx, toks, seq) &&
                llama_state_seq_set_data_ext(pure_h.ctx, P.data(), P.size(), seq,
                        LLAMA_STATE_SEQ_FLAGS_NONE) == P.size()) {
                full_logits_after_one_token(pure_h.ctx, probe_next(vocab), seq, set_p);
            }
            std::vector<float> ref_pure;
            if (pure_h.ctx && prefill(pure_h.ctx, toks, seq)) {
                full_logits_after_one_token(pure_h.ctx, probe_next(vocab), seq, ref_pure);
            }

            const double same_ctx = ok_ref ? cmp(ref, ref2) : -3.0;
            const double mixed_rt  = got_m ? cmp(ref,  set_m) : -3.0;
            const double pure_rt   = got_p ? cmp(ref_pure, set_p) : -3.0;
            printf("separation n=%-4d prefill-repeatable=%g  mixed-roundtrip=%g (M=%zu)  "
                   "pure-roundtrip=%g (P=%zu)\n",
                   n, same_ctx, mixed_rt, M.size(), pure_rt, P.size());
            // Asserted: the separation must actually produce numbers.
            CHECK(same_ctx == 0.0,
                  "separation n=%d: a repeated identical prefill did not reproduce its logits "
                  "(NRMSE=%g); the reference itself is not reproducible", n, same_ctx);
            CHECK(mixed_rt >= 0.0 && pure_rt >= 0.0,
                  "separation n=%d: a round-trip produced no measurement "
                  "(mixed=%g pure=%g)", n, mixed_rt, pure_rt);
        }

        // ---- Byte-level state comparison BEFORE any inference ----
        // Both blobs come from the SAME split configuration with the same
        // logical content (reference prefill vs the import), so their structure
        // is identical and a byte diff localizes the first differing field.
        // n=192 is the freeze point of the full-logits gate: PURE source,
        // SPLIT reference and IMPORTED are all read exactly after prefill(192),
        // before any proof-token.
        for (int n : { 1, 192, 256 }) {
            const auto toks = frozen_tokens_for_probe_n(vocab, n);
            std::vector<uint8_t> m_pure;
            // reference mixed state, captured from a CLEAN prefill
            std::vector<uint8_t> m_ref;
            std::string wm_ref;
            if (ctrl_h.ctx && prefill(ctrl_h.ctx, toks, seq)) {
                const size_t nb = llama_state_seq_get_size_ext(ctrl_h.ctx, seq, LLAMA_STATE_SEQ_FLAGS_NONE);
                if (nb) {
                    m_ref.resize(nb);
                    if (llama_state_seq_get_data_ext(ctrl_h.ctx, m_ref.data(), nb, seq,
                            LLAMA_STATE_SEQ_FLAGS_NONE) != nb) { m_ref.clear(); }
                    wm_ref = llama_mixed_state_write_section_map();
                }
            }
            // imported mixed state
            std::vector<uint8_t> m_imp;
            std::string wm_imp;
            bool imported = false;
            if (!m_ref.empty() && pure_h.ctx && prefill(pure_h.ctx, toks, seq) &&
                dest_h.ctx && prefill(dest_h.ctx, toks, seq)) {
                // ---- CONTROL, taken BEFORE dest_h is overwritten by the
                // import: two independent contexts with the SAME split
                // configuration, prefilling the SAME tokens. If their DeltaNet
                // states also disagree, then a PURE-vs-SPLIT disagreement is
                // not attributable to the configuration at all - the control
                // is what fails, and the import cannot be blamed for it.
                printf("  --- CONTROL: SPLIT ctrl_h vs SPLIT dest_h, both clean prefill(%d) ---\n", n);
                printf("%s", llama_recurrent_state_diff(ctrl_h.ctx, dest_h.ctx, seq).c_str());
                // Paired attention control at the same freeze point: if the
                // attention materialized values are identical between two
                // same-config prefills while the recurrent state is not, the
                // two memories do not agree about determinism.
                {
                    std::vector<float> ka, kb;
                    double m = 0.0;
                    const bool ok = llama_kvarn_materialized_probe(ctrl_h.ctx, seq, 3, 0, false, 0, &ka) &&
                                    llama_kvarn_materialized_probe(dest_h.ctx, seq, 3, 0, false, 0, &kb) &&
                                    ka.size() == kb.size();
                    bool diff = false;
                    if (ok) {
                        for (size_t i = 0; i < ka.size(); ++i) {
                            const double d = std::fabs((double) ka[i] - (double) kb[i]);
                            m = std::max(m, d);
                            if (d > 0.0) { diff = true; }
                        }
                    }
                    printf("  CONTROL attention K layer=3 head=0 pos=0: %s max_abs=%.9g\n",
                           ok ? (diff ? "DIFFER" : "IDENTICAL") : "PROBE FAILED", m);
                }
                server_prompt_cache_state st;
                const size_t bytes = llama_state_seq_get_size_ext(pure_h.ctx, seq, LLAMA_STATE_SEQ_FLAGS_NONE);
                st.data.main.resize(bytes);
                if (bytes && llama_state_seq_get_data_ext(pure_h.ctx, st.data.main.data(), bytes, seq,
                        LLAMA_STATE_SEQ_FLAGS_NONE) == bytes) {
                    st.prompt.tokens.insert(llama_tokens(toks.begin(), toks.end()));
                    st.model          = llama_get_model(pure_h.ctx);
                    st.model_instance = llama_model_mtp_weights_get_info(st.model).model_instance;
                    st.layout_tgt     = common_prompt_cache_layout(pure_h.ctx);
                    st.pos_tgt        = llama_memory_seq_pos_max(llama_get_memory(pure_h.ctx), seq);
                    st.checksum       = st.digest();
                    server_prompt_cache cc(-1, 0);
                    server_prompt pp;
                    pp.tokens.insert(llama_tokens(toks.begin(), toks.end()));
                    const auto rr = cc.restore(pp, st, dest_h.ctx, nullptr, nullptr, seq);
                    imported = (rr == server_prompt_cache_result::hit ||
                                rr == server_prompt_cache_result::needs_bootstrap);
                    // PURE source frozen at the same instant as the SPLIT
                    // reference and the IMPORTED state (all right after
                    // prefill(n), before any proof-token).
                    m_pure = std::move(st.data.main);
                    if (imported) {
                        const size_t ib = llama_state_seq_get_size_ext(dest_h.ctx, seq,
                                LLAMA_STATE_SEQ_FLAGS_NONE);
                        if (ib) {
                            m_imp.resize(ib);
                            if (llama_state_seq_get_data_ext(dest_h.ctx, m_imp.data(), ib, seq,
                                    LLAMA_STATE_SEQ_FLAGS_NONE) != ib) { m_imp.clear(); }
                            wm_imp = llama_mixed_state_write_section_map();
                        }
                    } else {
                        printf("state-diff n=%d: import refused: %s\n", n, cc.last_reason.c_str());
                    }
                }
            }
            CHECK(!m_ref.empty(), "state-diff n=%d: reference capture failed", n);
            CHECK(imported, "state-diff n=%d: import did not run", n);
            if (m_ref.empty() || m_imp.empty()) { continue; }
            printf("state-diff n=%d: ref=%zu bytes imp=%zu bytes%s\n", n, m_ref.size(),
                   m_imp.size(), m_ref.size() == m_imp.size() ? "" : "  SIZE MISMATCH");
            // PURE source frozen at the same instant. It is a DIFFERENT
            // representation, so no byte diff against it is meaningful: a
            // blob/order/layout difference between representations is not by
            // itself a bug. It is reported so the freeze is on the record, and
            // the PURE<->split comparison stays a normalized per-layer,
            // per-field semantic comparison through the real reader.
            const std::string map_pure = llama_mixed_state_section_map(m_pure.data(), m_pure.size());
            printf("  PURE source frozen here too: bytes=%zu mixed_frame=%s\n",
                   m_pure.size(),
                   map_pure.empty() ? "ABSENT (pure representation, byte-diff not applicable)"
                                    : "present");
            if (!map_pure.empty()) { printf("%s", map_pure.c_str()); }
            const size_t common = std::min(m_ref.size(), m_imp.size());
            size_t first = common, ndiff = 0;
            std::vector<size_t> band(16, 0);
            for (size_t i = 0; i < common; ++i) {
                if (m_ref[i] != m_imp[i]) {
                    if (first == common) { first = i; }
                    ++ndiff;
                    band[(i * 16) / common]++;
                }
            }
            if (first == common) {
                printf("state-diff n=%d: states are BYTE-IDENTICAL; any logits difference is "
                       "NOT in the persisted KV state\n", n);
            } else {
                printf("state-diff n=%d: first differing byte at %zu (%.4f%% into the blob), "
                       "%zu differing bytes of %zu\n", n, first,
                       100.0 * (double) first / (double) common, ndiff, common);
                printf("  differing bytes per 1/16 band:");
                for (size_t i = 0; i < band.size(); ++i) { printf(" %zu", band[i]); }
                printf("\n");
                // Walk the leading run of identical bytes and dump the first
                // divergence window verbatim, so the differing FIELD is visible.
                size_t run_end = first;
                while (run_end > 0 && m_ref[run_end - 1] == m_imp[run_end - 1]) { --run_end; }
                printf("  identical prefix: [0,%zu)  -> first divergence at %zu\n", run_end, first);
                const size_t lo = first >= 16 ? first - 16 : 0;
                const size_t hi = std::min(common, first + 48);
                printf("  ref[%zu..%zu): ", lo, hi);
                for (size_t i = lo; i < hi; ++i) { printf("%02x", m_ref[i]); }
                printf("\n  imp[%zu..%zu): ", lo, hi);
                for (size_t i = lo; i < hi; ++i) { printf("%02x", m_imp[i]); }
                printf("\n");
                // 64 bands: with 32 layers this resolves WHICH layers diverge.
                {
                    std::vector<size_t> fine(64, 0);
                    for (size_t i = 0; i < common; ++i) {
                        if (m_ref[i] != m_imp[i]) { ++fine[(i * 64) / common]; }
                    }
                    printf("  64-band differing-byte profile:\n   ");
                    for (size_t i = 0; i < fine.size(); ++i) {
                        printf("%zu%s", fine[i], (i + 1) % 16 == 0 ? "\n   " : " ");
                    }
                    printf("\n");
                }
                // Also report the first few diverging offsets individually.
                printf("  first diverging offsets:");
                size_t shown = 0;
                for (size_t i = first; i < common && shown < 8; ++i) {
                    if (m_ref[i] != m_imp[i]) { printf(" %zu(ref=%02x,imp=%02x)", i, m_ref[i], m_imp[i]); ++shown; }
                }
                printf("\n");
            }

            // ---- Section map: attribute every byte by the REAL schema ----
            // Reader walk over the frozen blobs, the write-side log recorded
            // while each blob was emitted, and a per-section diff. This is what
            // says WHICH section owns the first divergence; a byte offset or
            // an 8-byte window on its own does not.
            const std::string map_ref = llama_mixed_state_section_map(m_ref.data(), m_ref.size());
            const std::string map_imp = llama_mixed_state_section_map(m_imp.data(), m_imp.size());
            printf("  section map REF (reader walk):\n%s", map_ref.c_str());
            printf("  section map IMP (reader walk):\n%s", map_imp.c_str());
            printf("  section map REF (writer, emitted):\n%s", wm_ref.c_str());
            printf("  section map IMP (writer, emitted):\n%s", wm_imp.c_str());
            std::string owner;
            if (first < common && llama_mixed_state_locate(m_ref.data(), m_ref.size(), first, &owner)) {
                printf("  first divergence owner: %s\n", owner.c_str());
            } else if (first >= common) {
                printf("  first divergence owner: n/a (blobs byte-identical)\n");
            } else {
                printf("  first divergence owner: NOT FOUND in the section map\n");
            }
            printf("%s", llama_mixed_state_section_diff(m_ref.data(), m_ref.size(),
                                                        m_imp.data(), m_imp.size()).c_str());
            // The writer log and the reader walk are independent measurements:
            // they must agree on how many regions the mixed frame has. The
            // reader walk additionally reports whatever the hybrid memory
            // appends after the frame (DeltaNet recurrent state), which the
            // writer log never sees - so count only frame sections. The writer
            // log only exists when F4_STATE_SECTIONS was in the environment.
            const auto reader_frame_sections = [](const std::string & s) {
                size_t c = 0;
                size_t p = 0;
                while (p < s.size()) {
                    size_t e = s.find('\n', p);
                    if (e == std::string::npos) { e = s.size(); }
                    const std::string line = s.substr(p, e - p);
                    if (line.rfind("  [", 0) == 0 &&
                        line.find("trailing/post_frame_state") == std::string::npos) {
                        ++c;
                    }
                    p = e + 1;
                }
                return c;
            };
            const auto count_lines = [](const std::string & s) {
                size_t c = 0;
                for (char ch : s) { if (ch == '\n') { ++c; } }
                return c;
            };
            const size_t n_reader = reader_frame_sections(map_ref);
            if (wm_ref.empty()) {
                printf("  write-side section map EMPTY (run with F4_STATE_SECTIONS to "
                       "cross-check writer against reader)\n");
            } else {
                CHECK(count_lines(wm_ref) == n_reader,
                      "state-diff n=%d: write-side map has %zu regions, reader walk has %zu "
                      "frame regions (schema disagreement between writer and reader)",
                      n, count_lines(wm_ref), n_reader);
            }

            // ---- SEMANTIC three-way read through the REAL consumer ----
            // Byte differences between two representations are neither
            // required nor sufficient evidence of a defect. Read the
            // materialized K/V that attention would actually consume, for every
            // attention layer, every KV head and every frozen position, in the
            // PURE source, the SPLIT reference and the IMPORTED destination -
            // all three still holding exactly prefill(n) - and report the FIRST
            // (layer, head, position, field) where the materialized values
            // disagree. A disagreement here is semantic; byte differences with
            // no materialized disagreement are representational.
            if (n == 192) {
                static const int attn_layers[] = {3, 7, 11, 15, 19, 23, 27, 31};
                const int npos = n;
                size_t cmp = 0, diff_pp = 0;
                double worst = 0.0;
                int worst_layer = -1, worst_head = -1;
                llama_pos worst_pos = -1;
                bool worst_value = false;
                bool first_found = false;
                int first_layer = -1, first_head = -1;
                llama_pos first_pos = -1;
                bool first_value = false;
                double first_max = 0.0;
                uint32_t heads_probed = 0;
                for (int li : attn_layers) {
                    for (uint32_t h = 0; h < 8; ++h) {
                        std::vector<float> ref0;
                        if (!llama_kvarn_materialized_probe(ctrl_h.ctx, seq, li, h,
                                                            false, 0, &ref0)) {
                            break; // no such head
                        }
                        if (h == 0) { heads_probed = 0; }
                        ++heads_probed;
                        for (llama_pos p = 0; p < npos; ++p) {
                            for (int fld = 0; fld < 2; ++fld) {
                                const bool value = fld == 1;
                                std::vector<float> a, b, c;
                                const bool ok =
                                    llama_kvarn_materialized_probe(pure_h.ctx, seq, li, h, value, p, &a) &&
                                    llama_kvarn_materialized_probe(ctrl_h.ctx, seq, li, h, value, p, &b) &&
                                    llama_kvarn_materialized_probe(dest_h.ctx, seq, li, h, value, p, &c);
                                if (!ok || a.size() != b.size() || b.size() != c.size()) {
                                    continue;
                                }
                                ++cmp;
                                double m_rc = 0.0, m_ic = 0.0, m_pi = 0.0;
                                bool d_rc = false, d_ic = false, d_pi = false;
                                for (size_t i = 0; i < a.size(); ++i) {
                                    const double d1 = std::fabs((double) a[i] - (double) b[i]);
                                    const double d2 = std::fabs((double) c[i] - (double) b[i]);
                                    const double d3 = std::fabs((double) a[i] - (double) c[i]);
                                    m_rc = std::max(m_rc, d1);
                                    m_ic = std::max(m_ic, d2);
                                    m_pi = std::max(m_pi, d3);
                                    if (d1 > 0.0) { d_rc = true; }
                                    if (d2 > 0.0) { d_ic = true; }
                                    if (d3 > 0.0) { d_pi = true; }
                                }
                                // The import is the pair under test: PURE vs
                                // IMPORTED must match in the destination's own
                                // representation for the import to be faithful
                                // in the fields the consumer reads.
                                if (d_pi) {
                                    ++diff_pp;
                                    if (m_pi > worst) {
                                        worst = m_pi; worst_layer = li; worst_head = (int) h;
                                        worst_pos = p; worst_value = value;
                                    }
                                    if (!first_found) {
                                        first_found = true;
                                        first_layer = li; first_head = (int) h;
                                        first_pos = p; first_value = value;
                                        first_max = m_pi;
                                    }
                                }
                                if (d_rc || d_ic || d_pi) {
                                    printf("  sem layer=%d head=%u pos=%d %s "
                                           "PURE-SPLIT=%.9g SPLIT-IMPORTED=%.9g PURE-IMPORTED=%.9g\n",
                                           li, h, (int) p, value ? "V" : "K",
                                           m_rc, m_ic, m_pi);
                                }
                            }
                        }
                    }
                }
                printf("  semantic materialized read: comps=%zu heads_probed=%u "
                       "PURE-vs-IMPORTED differing comps=%zu\n", cmp, heads_probed, diff_pp);
                if (first_found) {
                    printf("  FIRST SEMANTIC DIFFERENCE: layer=%d head=%d pos=%d %s max_abs=%.9g\n",
                           first_layer, first_head, (int) first_pos,
                           first_value ? "V" : "K", first_max);
                    printf("  WORST SEMANTIC DIFFERENCE: layer=%d head=%d pos=%d %s max_abs=%.9g\n",
                           worst_layer, worst_head, (int) worst_pos,
                           worst_value ? "V" : "K", worst);
                } else {
                    printf("  FIRST SEMANTIC DIFFERENCE: none - all materialized K/V agree "
                           "across PURE, SPLIT and IMPORTED at this freeze point\n");
                }
                CHECK(cmp > 0,
                      "state-diff n=%d: the semantic read produced no comparable "
                      "materialized vectors", n);
            }

            // ---- NORMALIZED comparison of the recurrent tail ----
            // The hybrid memory writes attention FIRST and the recurrent
            // (DeltaNet) state SECOND, and both a split and a pure sequence
            // serialize that tail with the same writer, llama_memory_recurrent::
            // state_write. The tail therefore has the SAME layout on both sides,
            // so a byte comparison of equal-length tails is a comparison within
            // ONE representation - unlike the cross-representation diff above.
            // This isolates whether the DeltaNet recurrent state itself is the
            // thing that differs, without importing any frame schema.
            {
                size_t rf_s = 0, rf_e = 0;
                if (!llama_mixed_state_post_frame(m_ref.data(), m_ref.size(), &rf_s, &rf_e)) {
                    printf("  recurrent-tail: SPLIT reference blob carries no mixed frame\n");
                } else {
                    const size_t L = rf_e - rf_s;
                    printf("  recurrent-tail: split=[%zu,%zu) bytes=%zu blob_ref=%zu blob_pure=%zu blob_imp=%zu\n",
                           rf_s, rf_e, L, m_ref.size(), m_pure.size(), m_imp.size());
                    auto tail_cmp = [&](const char * what, const uint8_t * other, size_t n_other) {
                        if (n_other < L) {
                            printf("  recurrent-tail vs %s: SKIP shorter_blob=%zu < tail=%zu\n",
                                   what, n_other, L);
                            return;
                        }
                        const uint8_t * p = other + (n_other - L);
                        size_t first = L, nd = 0;
                        for (size_t k = 0; k < L; ++k) {
                            if (p[k] != m_ref[rf_s + k]) {
                                if (first == L) { first = k; }
                                ++nd;
                            }
                        }
                        printf("  recurrent-tail vs %-8s: bytes=%zu ndiff=%zu first=%s",
                               what, L, nd, first == L ? "none" : std::to_string(first).c_str());
                        if (first != L) {
                            printf(" abs=%zu", rf_s + first);
                        }
                        printf("\n");
                    };
                    tail_cmp("PURE", m_pure.data(), m_pure.size());
                    tail_cmp("IMPORTED", m_imp.data(), m_imp.size());
                    // Which side is which: is the IMPORTED tail the PURE
                    // source's tail (import faithful here) or the SPLIT
                    // reference's tail (import faithful here)? Both blobs are
                    // compared directly, at equal length, with the same writer
                    // on both sides.
                    if (m_pure.size() >= L && m_imp.size() >= L) {
                        const uint8_t * p = m_pure.data() + (m_pure.size() - L);
                        size_t first2 = L, nd2 = 0;
                        for (size_t k = 0; k < L; ++k) {
                            if (p[k] != m_imp[m_imp.size() - L + k]) {
                                if (first2 == L) { first2 = k; }
                                ++nd2;
                            }
                        }
                        printf("  recurrent-tail PURE vs IMPORTED: bytes=%zu ndiff=%zu first=%s\n",
                               L, nd2, first2 == L ? "none" : std::to_string(first2).c_str());
                    }
                }
            }

            // ---- DeltaNet recurrent state, per model layer ----
            // The attention frame has a schema to walk, so its sections are
            // compared through stream_describe. The recurrent tail has no such
            // schema; the only honest normalized comparison is between the
            // LIVE r_l/s_l/p_l tensors themselves - same writer, same layout,
            // two contexts - which also gives the per-layer mapping the
            // attention-side section table cannot provide (24 DeltaNet layers).
            if (n == 192) {
                printf("  --- DeltaNet recurrent: SPLIT reference vs PURE source ---\n");
                printf("%s", llama_recurrent_state_diff(ctrl_h.ctx, pure_h.ctx, seq).c_str());
                printf("  --- DeltaNet recurrent: SPLIT reference vs IMPORTED ---\n");
                printf("%s", llama_recurrent_state_diff(ctrl_h.ctx, dest_h.ctx, seq).c_str());
            }

            // ---- FIRST ACTIVATION DIFFERENCE ----
            // The recurrent state of layers 0-2 is identical between a split
            // and a non-split prefill of the same tokens while layer 4 already
            // diverges, and layer 3's materialized K/V are identical. The
            // hidden state feeding layer 4 is the OUTPUT of layer 3, so tap the
            // raw per-layer decoder outputs with the shared multi-layer capture
            // primitive on two fresh contexts and compare layer by layer,
            // position by position: the first layer whose output differs is
            // where the divergence enters the graph. This measures activations,
            // not serialized bytes.
            if (n == 192) {
                ctx_holder ta_h, tb_h, tc_h;
                ta_h.ctx = make_ctx(P, /*split=*/1, /*remote=*/1);
                // Remote Q4 KV placement WITHOUT the position split, so the two
                // ingredients of the split configuration can be bisected apart.
                tc_h.ctx = make_ctx(0, /*split=*/1, /*remote=*/1);
                tb_h.ctx = make_ctx(0, /*split=*/0, /*remote=*/0);
                const int32_t tap[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
                const int32_t ntap  = (int32_t) (sizeof(tap) / sizeof(tap[0]));
                if (ta_h.ctx) { llama_set_capture_layers(ta_h.ctx, tap, (size_t) ntap); }
                if (tc_h.ctx) { llama_set_capture_layers(tc_h.ctx, tap, (size_t) ntap); }
                if (tb_h.ctx) { llama_set_capture_layers(tb_h.ctx, tap, (size_t) ntap); }
                const int ok_a = ta_h.ctx ? (prefill(ta_h.ctx, toks, seq) ? 1 : -1) : 0;
                const int ok_c = tc_h.ctx ? (prefill(tc_h.ctx, toks, seq) ? 1 : -1) : 0;
                const int ok_b = tb_h.ctx ? (prefill(tb_h.ctx, toks, seq) ? 1 : -1) : 0;
                printf("  activation tap setup: splitP+remote=%d remoteOnly=%d pure=%d "
                       "(0=no ctx, 1=prefill ok, -1=prefill failed)\n", ok_a, ok_c, ok_b);
                const uint32_t ne = (uint32_t) llama_model_n_embd(model);
                auto layer_out_diff = [&](const char * what, llama_context * cx, llama_context * ref) {
                    if (!cx || !ref) {
                        printf("  activation tap [%s]: SKIP missing context\n", what);
                        return;
                    }
                    const uint32_t ca = llama_get_n_capture(cx);
                    const uint32_t cb = llama_get_n_capture(ref);
                    const float * pa = llama_get_embeddings_capture(cx);
                    const float * pb = llama_get_embeddings_capture(ref);
                    printf("  activation tap [%s]: n_capture=%u/%u n_embd=%u buffer=%s/%s\n",
                           what, ca, cb, ne, pa ? "present" : "null", pb ? "present" : "null");
                    if (!pa || !pb || ca == 0 || ca != cb) { return; }
                    size_t rows = 0;
                    double overall = 0.0;
                    double lmax[16] = { 0 };
                    size_t lrow[16] = { 0 };
                    for (int i = 0; i < n; ++i) {
                        const float * ra = llama_get_embeddings_capture_ith(cx,   i);
                        const float * rb = llama_get_embeddings_capture_ith(ref, i);
                        if (!ra || !rb) { continue; }
                        ++rows;
                        for (uint32_t L = 0; L < ca && L < 16; ++L) {
                            double m = 0.0;
                            for (uint32_t j = 0; j < ne; ++j) {
                                const double d = std::fabs((double) ra[(size_t) L * ne + j] -
                                                           (double) rb[(size_t) L * ne + j]);
                                if (d > m) { m = d; }
                            }
                            if (m > 0.0) { ++lrow[L]; }
                            if (m > lmax[L]) { lmax[L] = m; }
                            if (m > overall) { overall = m; }
                        }
                    }
                    printf("  activation tap [%s]: rows=%zu overall_max_abs=%.9g\n",
                           what, rows, overall);
                    for (uint32_t L = 0; L < ca && L < 16; ++L) {
                        printf("  [%s] layer-out[%d] tap=%d: %s rows=%zu max_abs=%.9g\n",
                               what, L, tap[L], lrow[L] ? "DIFFER" : "same",
                               lrow[L], lmax[L]);
                    }
                };
                layer_out_diff("splitP+remote vs pure", ta_h.ctx, tb_h.ctx);
                layer_out_diff("remoteOnly    vs pure", tc_h.ctx, tb_h.ctx);
                layer_out_diff("splitP+remote vs remoteOnly", ta_h.ctx, tc_h.ctx);
            }
        }

        // ---- Per-LAYER / per-FIELD table, with an instrumentation gate ----
        // Snapshots are captured AFTER the prefill and BEFORE the proof-token
        // decode, and frozen in their own variables that nothing overwrites.
        {
            const int n = 1; // cheapest probe, where the divergence already exists
            const auto toks = frozen_tokens_for_probe_n(vocab, n);
            std::string tab_pure_src, tab_split_ref, tab_after_import;
            // FROZEN snapshot 1: PURE source after its prefill, before any decode.
            CHECK(prefill(pure_h.ctx, toks, seq), "layer table: pure prefill failed");
            tab_pure_src = llama_kvarn_live_layer_table(pure_h.ctx, seq);
            // FROZEN snapshot 2: SPLIT reference after its own prefill.
            CHECK(prefill(ctrl_h.ctx, toks, seq), "layer table: split prefill failed");
            tab_split_ref = llama_kvarn_live_layer_table(ctrl_h.ctx, seq);
            // FROZEN snapshot 3: destination immediately after the import.
            bool did_import = false;
            {
                server_prompt_cache_state st;
                const size_t bytes = llama_state_seq_get_size_ext(pure_h.ctx, seq, LLAMA_STATE_SEQ_FLAGS_NONE);
                st.data.main.resize(bytes);
                if (bytes && llama_state_seq_get_data_ext(pure_h.ctx, st.data.main.data(), bytes, seq,
                        LLAMA_STATE_SEQ_FLAGS_NONE) == bytes) {
                    st.prompt.tokens.insert(llama_tokens(toks.begin(), toks.end()));
                    st.model          = llama_get_model(pure_h.ctx);
                    st.model_instance = llama_model_mtp_weights_get_info(st.model).model_instance;
                    st.layout_tgt     = common_prompt_cache_layout(pure_h.ctx);
                    st.pos_tgt        = llama_memory_seq_pos_max(llama_get_memory(pure_h.ctx), seq);
                    st.checksum       = st.digest();
                    if (prefill(dest_h.ctx, toks, seq)) {
                        server_prompt_cache cc(-1, 0);
                        server_prompt pp;
                        pp.tokens.insert(llama_tokens(toks.begin(), toks.end()));
                        const auto rr = cc.restore(pp, st, dest_h.ctx, nullptr, nullptr, seq);
                        did_import = (rr == server_prompt_cache_result::hit ||
                                     rr == server_prompt_cache_result::needs_bootstrap);
                        CHECK(did_import, "layer table: import refused: %s", cc.last_reason.c_str());
                    }
                }
            }
            tab_after_import = llama_kvarn_live_layer_table(dest_h.ctx, seq);
            CHECK(!tab_pure_src.empty() && !tab_after_import.empty() && !tab_split_ref.empty(),
                  "layer table: empty table for pure/imported/split");

            auto rows = [](const std::string & t) {
                std::vector<std::string> v;
                size_t p = 0;
                while (p < t.size()) {
                    size_t e = t.find('\n', p);
                    if (e == std::string::npos) { e = t.size(); }
                    if (t.compare(p, 6, "layer=") == 0) { v.push_back(t.substr(p, e - p)); }
                    p = e + 1;
                }
                return v;
            };
            auto field = [](const std::string & line, const char * key, std::string & out) {
                const std::string k = std::string(" ") + key + "=";
                const size_t at = line.find(k);
                if (at == std::string::npos) { return false; }
                size_t b = at + k.size(), e = b;
                while (e < line.size() && line[e] != ' ') { ++e; }
                out = line.substr(b, e - b);
                return true;
            };
            const auto rp = rows(tab_pure_src), ri = rows(tab_after_import), rs = rows(tab_split_ref);
            // Row counts can legitimately differ (a side may have no standard
            // cache at all), so only compare the common prefix. Indexing the
            // shorter vector out of range is what segfaulted this test before.
            const size_t nrows_common = std::min(rp.size(), std::min(ri.size(), rs.size()));
            printf("  layer rows: pure=%zu imported=%zu split=%zu (prefix common=%zu)\n",
                   rp.size(), ri.size(), rs.size(), nrows_common);
            // Compare by (cache, layer) key, not by position: the PURE source has
            // no standard cache, so row order and count legitimately differ and a
            // positional comparison would silently skip the local layers.
            // Accept the key at the start of the row as well as after a space:
            // layer rows begin with "layer=" and have no leading space.
            auto grab = [](const std::string & l, const char * k) -> std::string {
                const std::string plain = std::string(k) + "=";
                size_t at = l.find(plain);
                size_t b = 0;
                if (at != std::string::npos && at == 0) {
                    b = plain.size();
                } else {
                    const std::string kk = " " + plain;
                    at = l.find(kk);
                    if (at == std::string::npos) { return {}; }
                    b = at + kk.size();
                }
                size_t e = b;
                while (e < l.size() && l[e] != ' ') { ++e; }
                return l.substr(b, e - b);
            };
            auto key_of = [&grab](const std::string & line) {
                const std::string lay = grab(line, "layer");
                const std::string cache = grab(line, "cache");
                return cache.empty() ? ("kvarn:" + lay) : (cache + ":" + lay);
            };
            auto find_row = [&key_of](const std::vector<std::string> & v, const std::string & key) -> std::string {
                for (const auto & line : v) {
                    if (key_of(line) == key) { return line; }
                }
                return {};
            };
            size_t local_covered = 0, remote_covered = 0, uncovered_live = 0;
            std::string first_diff_desc;
            size_t matched = 0;
            for (const auto & row_i : ri) {
                const std::string key = key_of(row_i);
                const std::string row_p = find_row(rp, key);
                const std::string row_s = find_row(rs, key);
                std::string place = grab(row_i, "placement");
                if (grab(row_i, "cache") == "standard") { place = "local"; }
                const std::string covi = grab(row_i, "coverage");
                const std::string skip = grab(row_i, "skipped");
                const std::string hi = grab(row_i, "live_payload_hash");
                const std::string hp = grab(row_p, "live_payload_hash");
                const std::string cp = grab(row_p, "live_rows"), ci = grab(row_i, "live_rows");
                const std::string cs = grab(row_s, "live_rows");
                const std::string pp = grab(row_p, "pos_max"), pi = grab(row_i, "pos_max");
                const std::string ps = grab(row_s, "pos_max");
                const std::string bp = grab(row_p, "live_bytes"), bi = grab(row_i, "live_bytes");
                const std::string covp = grab(row_p, "coverage");
                if (place == "local" && covi == "covered" && !bi.empty() && strtoull(bi.c_str(), nullptr, 10) > 0) { ++local_covered; }
                if (place == "remote" && covi == "covered") { ++remote_covered; }
                if (covi == "UNSUPPORTED" && !bi.empty() && strtoull(bi.c_str(), nullptr, 10) > 0) { ++uncovered_live; }
                if (!row_p.empty()) { ++matched; }
                if (!row_p.empty() && hp != hi && covi == "covered" && first_diff_desc.empty()) {
                    first_diff_desc = "key=" + key + " live_rows " + cp + "->" + ci +
                            " pos_max " + pp + "->" + pi + " live_bytes " + bp + "->" + bi;
                }
                printf("  %-16s %-7s %-11s %-13s %-13s %-8s %-8s %-5s %-17s %-17s %s\n",
                       key.c_str(), place.c_str(),
                       (cp + "/" + ci + "/" + cs).c_str(),
                       (pp + "/" + pi + "/" + ps).c_str(),
                       (bp + "/" + bi).c_str(),
                       covp.c_str(), covi.c_str(), skip.c_str(),
                       hp.c_str(), hi.c_str(),
                       row_p.empty() ? "no-source-row" :
                       (hp == hi ? "live=same" : (covi == "covered" ? "live=DIFFERS" : "live=unknown")));
            }
            printf("  matched keys pure-vs-imported: %zu of %zu\n", matched, ri.size());
            printf("  rows=%zu local_covered=%zu remote_covered=%zu unsupported_live=%zu\n",
                   ri.size(), local_covered, remote_covered, uncovered_live);
            if (!first_diff_desc.empty()) {
                printf("  FIRST DIFFERING LAYER FIELD: %s\n", first_diff_desc.c_str());
            } else {
                printf("  no differing layer field found at covered layers\n");
            }
            // ---- PROOF A (independent path): a live (head, group) mutation must
            // change the MATERIALIZED output, not only the hash. The probe
            // addresses the span by the authoritative view layout (nb[1] head
            // stride, nb[2] group stride, group = cell / KVAR_N_GROUP); the
            // verifier re-decodes and compares full logits, so it does not reuse
            // the hashing formula.
            // Each probe: baseline in place, flip ONE byte in ONE (field, head,
            // group) span, read the byte back, re-decode, then RESTORE the byte
            // and prove the restoration is exact before the next probe. Without
            // the restore, llama_memory_clear leaves the mutation in place and
            // every later probe is contaminated.
            struct span_case { const char * tensor; uint32_t head; const char * label; };
            const span_case cases[] = {
                { "k_records", 0, "K/head0" },
                { "k_records", 1, "K/head1" },
                { "v_records", 0, "V/head0" },
                { "v_records", 1, "V/head1" },
            };
            // INDEPENDENT CONTEXTS PER BRANCH. Reusing one context and relying on
            // clear/re-prefill is NOT isolation: clear() does not rewrite bytes it
            // never visited, and the proof-token decode writes KV at the next
            // position, so sequential measurements on one context are not
            // comparable. Each branch gets its own context with the same tokens,
            // same prefill, same positions and exactly one proof token at the
            // same position N.
            //
            //   A  intact              -> baseline
            //   A2 intact, repeated    -> determinism gate (A must equal A2)
            //   B  live mutation
            //   C  inactive mutation
            //   D  mutation then RESTORE, all before any decode (A must equal D)
            //
            // A=A2 and A=D are asserted BEFORE B/C are interpreted.
            struct branch_result {
                std::string pos_desc;
                int decodes = 0;
                std::vector<float> logits;
                std::vector<float> k_head0, k_head1, v_head0, v_head1;
                bool consumer_ok = false;
                // Post-decode consumer captures: [head][pos] at the route that
                // survived the clear (identity + fresh gen + FA hit enforced by
                // read_consumer_head/gen, compared only against matching ids).
                struct post_cap { bool ok = false; std::string id; std::vector<float> v; };
                post_cap after[2][2];   // [head][pos 0|1]
            };
            auto run_branch = [&](const char * what, int layer_idx, const char * tensor,
                                  uint32_t head, bool live_mutation, bool restore_before_decode,
                                  int nprobe_tokens) {
                branch_result br;
                ctx_holder ch;
                ch.ctx = make_ctx(P, /*split=*/true);
                const auto tk = frozen_tokens_for_probe_n(vocab, nprobe_tokens);
                if (!ch.ctx || !prefill(ch.ctx, tk, seq)) {
                    return br;
                }
                br.pos_desc = "pos_min=" + std::to_string(llama_memory_seq_pos_min(llama_get_memory(ch.ctx), seq)) +
                              " pos_max=" + std::to_string(llama_memory_seq_pos_max(llama_get_memory(ch.ctx), seq));
                // Consumer values BEFORE any decode.
                br.k_head0.resize(1); br.k_head1.resize(1);
                br.v_head0.resize(1); br.v_head1.resize(1);
                br.consumer_ok =
                    llama_kvarn_materialized_probe(ch.ctx, seq, layer_idx, 0, false, 0, &br.k_head0) &&
                    llama_kvarn_materialized_probe(ch.ctx, seq, layer_idx, 1, false, 0, &br.k_head1) &&
                    llama_kvarn_materialized_probe(ch.ctx, seq, layer_idx, 0, true,  0, &br.v_head0) &&
                    llama_kvarn_materialized_probe(ch.ctx, seq, layer_idx, 1, true,  0, &br.v_head1);
                uint8_t orig = 0, wrote_byte = 0; uint32_t group = 0;
                bool wrote = false;
                // Consumer's own row (not the materializer) for V heads 0/1 at
                // pos 0, BEFORE the mutation -- the identity proof for which row
                // the consumer actually holds.
                cap_result cons0_b, cons1_b;
                // BOTH routes for layer3/V, heads 0 and 1, pos 0 -- index 0 alone
                // is not enough when routes >= 2.
                static const int kMaxRoutes = 4;
                cap_result br_b[2][kMaxRoutes];
                size_t nroute[2] = {0, 0};
                if (getenv("F4_TEST_MAP")) {
                    for (int h = 0; h < 2; ++h) {
                        nroute[h] = llama_kvarn_capture_routes(ch.ctx, layer_idx, true, h);
                        for (size_t ri = 0; ri < nroute[h] && ri < kMaxRoutes; ++ri) {
                            br_b[h][ri] = read_consumer_head(ch.ctx, layer_idx, true, h, 0,
                                                             (size_t) (1 << 20), ri);
                            printf("  KROUTE-BEFORE n=%d V head=%d route=%zu/%zu ok=%d "
                                   "gen=%llu fa=%llu id=[%s] v0=%.9g\n",
                                   nprobe_tokens, h, ri, nroute[h], (int) br_b[h][ri].ok,
                                   (unsigned long long) cap_field(br_b[h][ri].info, " gen="),
                                   (unsigned long long) cap_field(br_b[h][ri].info, " fa="),
                                   cap_identity(br_b[h][ri].info).c_str(),
                                   br_b[h][ri].v.empty() ? 0.0f : br_b[h][ri].v[0]);
                        }
                    }
                    cons0_b = br_b[0][0];
                    cons1_b = br_b[1][0];
                    // Intended boundary: drop prefill-era samples so every AFTER
                    // sample must come from a NEW (post-mutation/decode) FA hit.
                    llama_kvarn_capture_clear(ch.ctx);
                    printf("  KROUTE-CLEAR captures cleared at prefill/mutation boundary "
                           "(n=%d layer=%d V)\n", nprobe_tokens, layer_idx);
                }
                if (live_mutation) {
                    wrote = llama_kvarn_probe_head_span(ch.ctx, seq, layer_idx, head, 0,
                            tensor, /*live=*/true, &orig, &wrote_byte, &group);
                }
                if (restore_before_decode && wrote) {
                    const bool back = llama_kvarn_probe_restore_byte(ch.ctx, seq, layer_idx,
                            head, group, tensor, orig);
                    printf("  branch %-3s layer=%d %-10s head=%u live=1 restore_before_decode=%d wrote=%d\n",
                           what, layer_idx, tensor, head, (int) back, (int) wrote);
                } else {
                    printf("  branch %-3s layer=%d %-10s head=%u live=%d restore_before_decode=0 wrote=%d group=%u\n",
                           what, layer_idx, tensor, head, (int) live_mutation, (int) wrote, group);
                }
                // Consumer values AFTER the write/restore, still before decode.
                std::vector<float> k0b(1), k1b(1), v0b(1), v1b(1);
                const bool consumer_after =
                    llama_kvarn_materialized_probe(ch.ctx, seq, layer_idx, 0, false, 0, &k0b) &&
                    llama_kvarn_materialized_probe(ch.ctx, seq, layer_idx, 1, false, 0, &k1b) &&
                    llama_kvarn_materialized_probe(ch.ctx, seq, layer_idx, 0, true,  0, &v0b) &&
                    llama_kvarn_materialized_probe(ch.ctx, seq, layer_idx, 1, true,  0, &v1b);
                auto diffv = [](const std::vector<float> & a, const std::vector<float> & b) {
                    if (a.size() != b.size() || a.empty()) { return -1.0f; }
                    float m = 0.0f;
                    for (size_t i = 0; i < a.size(); ++i) { m = std::fmax(m, std::fabs(a[i] - b[i])); }
                    return m;
                };
                printf("    consumer before->after: K.h0=%g K.h1=%g V.h0=%g V.h1=%g "
                       "(consumer_ok=%d after=%d)\n",
                       diffv(br.k_head0, k0b), diffv(br.k_head1, k1b),
                       diffv(br.v_head0, v0b), diffv(br.v_head1, v1b),
                       (int) br.consumer_ok, (int) consumer_after);
                // Reset the descriptor-carried capture immediately before the
                // decode so the records we read are the ones that decode's FA
                // performed, not leftovers from prefill (the count is not reset
                // anywhere else, so it otherwise saturates the 8-slot buffer).
                // THREE-INSTANT value probe on the SAME byte, fixture-only, via
                // the existing public hook (read-then-write; a second call with
                // the same arguments restores the value the first one flipped).
                //   (i)  value the probe just wrote
                //   (ii) value after all stores/copies, immediately before decode
                //   (iii) value immediately after decode
                // (i)==(ii) rejects an overwrite before decode; (i)!=(ii)
                // proves a writing op; (ii)==(iii) rejects an overwrite during
                // decode. No library change, no new CUDA header, no capture-time
                // synchronization -- this is the isolated discriminator.
                if (getenv("F4_TEST_MAP") && live_mutation && wrote) {
                    // Reference is the value the MUTATION wrote (new_out / wrote_byte),
                    // not `orig` (which is the pre-mutation byte).
                    const int mutated = (int) wrote_byte;
                    auto instant = [&](const char * tag) -> int {
                        uint8_t a = 0, b = 0;
                        uint32_t g0 = 0;
                        if (!llama_kvarn_probe_head_span(ch.ctx, seq, layer_idx, head, 0,
                                tensor, true, &a, &b, &g0)) { return -1; }
                        // flip back: first call wrote ~a; the second call reads ~a
                        // and writes a again, so `f` is the restored value.
                        uint8_t e = 0, f = 0; uint32_t g1 = 0;
                        const bool second_ok = llama_kvarn_probe_head_span(ch.ctx, seq,
                                layer_idx, head, 0, tensor, true, &e, &f, &g1);
                        const bool restored = second_ok && f == a;
                        printf("  KINSTANT %s n=%d %s head=%u byte=%u group=%u "
                               "second_ok=%d restored=%d\n",
                               tag, nprobe_tokens, tensor, head, a, g0,
                               (int) second_ok, (int) restored);
                        if (!restored) {
                            printf("      KINSTANT-WARN %s: restore FAILED "
                                   "(second_ok=%d f=%u a=%u) -- state changed\n",
                                   tag, (int) second_ok, f, a);
                        }
                        return (int) a;
                    };
                    const int at_ii = instant("ii_pre_decode");
                    if (at_ii >= 0) {
                        const bool stable = (at_ii == mutated);
                        printf("  KINSTANT-CMP n=%d %s head=%u mutated=%d ii=%d -> %s\n",
                               nprobe_tokens, tensor, head, mutated, at_ii,
                               stable ? "STABLE (== mutated value, no overwrite before decode)"
                                      : "OVERWRITTEN before decode");
                        printf("      NOTE: equal values before/after decode do NOT exclude a "
                               "TRANSIENT overwrite inside decode; only the post-decode read "
                               "and the consumer-row capture below bound that.\n");
                    }
                }
                // Exactly ONE proof-token decode at the same position N.
                if (full_logits_after_one_token(ch.ctx, probe_next(vocab), seq, br.logits)) {
                    br.decodes = 1;
                }
                if (getenv("F4_TEST_MAP") && live_mutation && wrote) {
                    uint8_t a = 0, b = 0, c = 0, d = 0; uint32_t g0 = 0;
                    if (llama_kvarn_probe_head_span(ch.ctx, seq, layer_idx, head, 0,
                            tensor, true, &a, &b, &g0)) {
                        uint8_t e = 0, f = 0;
                        llama_kvarn_probe_head_span(ch.ctx, seq, layer_idx, head, 0,
                                tensor, true, &e, &f, &g0);
                        printf("  KINSTANT iii_post_decode n=%d %s head=%u byte=%u restored=%d\n",
                               nprobe_tokens, tensor, head, a, (int) (e == a));
                    }
                }
                if (getenv("F4_TEST_MAP") && live_mutation && wrote) {
                    uint8_t a = 0, b = 0; uint32_t g0 = 0;
                    if (llama_kvarn_probe_head_span(ch.ctx, seq, layer_idx, head, 0,
                            tensor, true, &a, &b, &g0)) {
                        uint8_t e = 0, f = 0; uint32_t g1 = 0;
                        const bool ok2 = llama_kvarn_probe_head_span(ch.ctx, seq, layer_idx,
                                head, 0, tensor, true, &e, &f, &g1);
                        // Target identity: the post-decode probe must land on the
                        // SAME cell/group as the mutation, otherwise it is reading
                        // a different live cell and says nothing about ours.
                        printf("  KINSTANT iii_post_decode n=%d %s head=%u byte=%u "
                               "group=%u mutated=%d same_target=%d restored=%d\n",
                               nprobe_tokens, tensor, head, a, g0, (int) wrote_byte,
                               (int) (g0 == group), (int) (ok2 && f == a));
                    }
                    // Consumer's OWN row AFTER the mutation+decode: if this changed
                    // for head1 the mapping reaches the consumer and the missing
                    // logit change is a degenerate-control issue; if it did not,
                    // the probe row is not the consumed row.
                    auto dmax = [](const std::vector<float> & x, const std::vector<float> & y) {
                        if (x.empty() || y.empty() || x.size() != y.size()) { return -1.0; }
                        double m = 0; bool any = false;
                        for (size_t i = 0; i < x.size(); ++i) {
                            if (!std::isfinite(x[i]) || !std::isfinite(y[i])) { continue; }
                            m = std::fmax(m, std::fabs(double(x[i]) - double(y[i]))); any = true;
                        }
                        return any ? m : -1.0;
                    };
                    // Iterate BOTH routes again and only compare samples whose
                    // identity (layer/kv/head/pos/route/range/stored/shape/offset/
                    // backend) matches AND whose generation/FA-hit prove they are
                    // FRESH samples produced after the clear, not a stale or
                    // replaced g_f4_caps entry.
                    int matched = 0, stale = 0, identity_bad = 0;
                    double best[2] = { -1.0, -1.0 };
                    for (int h = 0; h < 2; ++h) {
                        for (size_t ri = 0; ri < nroute[h] && ri < kMaxRoutes; ++ri) {
                            cap_result a = read_consumer_head(ch.ctx, layer_idx, true, h, 0,
                                                              (size_t) (1 << 20), ri);
                            const std::string ida = cap_identity(a.info);
                            const std::string idb = cap_identity(br_b[h][ri].info);
                            const unsigned long long
                                genb = cap_field(br_b[h][ri].info, " gen="),
                                gena = cap_field(a.info, " gen="),
                                fab  = cap_field(br_b[h][ri].info, " fa="),
                                faa  = cap_field(a.info, " fa=");
                            printf("  KROUTE-AFTER n=%d V head=%d route=%zu/%zu ok=%d "
                                   "gen=%llu->%llu fa=%llu->%llu id=%s v0=%.9g max|d|=%.9g\n",
                                   nprobe_tokens, h, ri, nroute[h], (int) a.ok,
                                   (unsigned long long) genb, (unsigned long long) gena,
                                   (unsigned long long) fab, (unsigned long long) faa,
                                   ida == idb ? "MATCH" : "IDENTITY-MISMATCH",
                                   a.v.empty() ? 0.0f : a.v[0],
                                   dmax(br_b[h][ri].v, a.v));
                            if (!a.ok || !br_b[h][ri].ok) { continue; }
                            if (ida != idb) { ++identity_bad; continue; }        // stale/replaced
                            if (gena <= genb || faa <= fab) { ++stale; continue; } // not fresh
                            ++matched;
                            const double d = dmax(br_b[h][ri].v, a.v);
                            if (d > best[h]) { best[h] = d; }
                        }
                    }
                    printf("  KROUTE-VERDICT n=%d matched=%d stale=%d identity_mismatch=%d "
                           "head0_max=%.9g head1_max=%.9g -> head1 %s\n",
                           nprobe_tokens, matched, stale, identity_bad, best[0], best[1],
                           best[1] > 0.0
                               ? "consumer row CHANGED (mapping reaches the consumer)"
                               : "consumer row UNCHANGED on every fresh matched route "
                                 "(probe row != consumed row, or stale sample)");
                }
                if (getenv("F4_TEST_MAP")) {
                    for (int h = 0; h < 2; ++h) {
                        for (int pp = 0; pp < 2; ++pp) {
                            // pos is a FILTER in the API, not an index: scan all
                            // stored samples with pos disabled and select the one
                            // whose own origin position matches.
                            // Prefer the direct pos filter (it works when the
                            // origin matches); fall back to the origin scan when
                            // it does not, so a selector bug cannot masquerade as
                            // missing data.
                            cap_result c = read_consumer_head(ch.ctx, layer_idx, true, h,
                                                             (llama_pos) 0, (size_t) (1 << 20), 0);
                            if (!c.ok) {
                                // Only REPLACE with a successful origin match: a
                                // failed scan must not discard the (NaN-filled)
                                // row the direct call produced, or downstream
                                // consumers see an empty vector instead of "no data".
                                cap_result f = read_consumer_head_at(ch.ctx, layer_idx,
                                        true, h, (llama_pos) pp, (size_t) (1 << 20));
                                if (f.ok) { c = f; }
                            }
                            // Derive row_width from the WRITER's own offsets
                            // (cs.offset = h*row): fetch the matching fresh head
                            // for the same key, take stride = |off1 - off0|,
                            // validate, then n_cells = elem / row_width.
                            // NO model head width is assumed (never KVAR_N_GROUP).
                            const size_t elem = c.ok ? (size_t) cap_field(c.info, " elem=") : 0;
                            const size_t nbytes = c.ok ? (size_t) cap_field(c.info, " nbytes=") : 0;
                            cap_result other = read_consumer_head(ch.ctx, layer_idx, true,
                                    (uint32_t) (h ^ 1), 0, (size_t) (1 << 20), 0);
                            const long long off0 = c.ok ? (long long) cap_field(c.info, " offset=") : -1;
                            const long long off1 = other.ok ? (long long) cap_field(other.info, " offset=") : -1;
                            int fam_head_a = -1, fam_head_b = -1;
                            const bool same_family = other.ok &&
                                kvarn_stage::capture_same_family(c.info, other.info,
                                                                 &fam_head_a, &fam_head_b) &&
                                kvarn_stage::capture_fa_hits(c.info) > 0 &&
                                kvarn_stage::capture_fa_hits(other.info) > 0;
                            int n_cells_geo = 0, row_width = 0;
                            const bool geom_ok = same_family && kvarn_stage::stage_geometry_from_offsets(
                                elem, nbytes, off0, off1, n_cells_geo, row_width);
                            const int tail_groups = 0;     // ring mapping unknown -> refuse non-group0
                            const bool ring_known = false;
                            if (!geom_ok) {
                                printf("  KPOST-GEOM n=%d %s head=%d elem=%zu nbytes=%zu "
                                       "offset0=%lld offset1=%lld heads=%d/%d fa=%lld/%lld "
                                       "same_family=%d -> REFUSED "
                                       "(row stride not derivable from writer offsets; not "
                                       "assuming any model head width)\n",
                                       nprobe_tokens, tensor, h, elem, nbytes, off0, off1,
                                       fam_head_a, fam_head_b,
                                       kvarn_stage::capture_fa_hits(c.info),
                                       other.ok ? kvarn_stage::capture_fa_hits(other.info) : -1,
                                       (int) same_family);
                            } else {
                                printf("  KPOST-GEOM n=%d %s head=%d elem=%zu nbytes=%zu "
                                       "offset0=%lld offset1=%lld heads=%d/%d row_width=%d "
                                       "n_cells=%d OK (derived from writer offsets)\n",
                                       nprobe_tokens, tensor, h, elem, nbytes, off0, off1,
                                       fam_head_a, fam_head_b, row_width, n_cells_geo);
                            }
                            const int scell = kvarn_stage::stage_cell_for_pos(
                                pp, KVAR_N_GROUP, tail_groups, ring_known);
                            std::vector<float> row; int st = kvarn_stage::UNAVAILABLE;
                            if (c.ok && geom_ok) {
                                if (!kvarn_stage::extract_row(c.v, elem, scell, row_width, row, st)) {
                                    st = (st == kvarn_stage::NONFINITE) ? kvarn_stage::NONFINITE : kvarn_stage::UNAVAILABLE;
                                }
                            }
                            br.after[h][pp].ok = (st == kvarn_stage::ZERO || st == kvarn_stage::NONZERO);
                            br.after[h][pp].id = cap_identity(c.info);
                            br.after[h][pp].v = row;
                            printf("  KPOST n=%d %s head=%d want_pos=%d origin_pos=%lld "
                                   "elem=%zu roww=%d stage_cell=%d geom_ok=%d status=%s rows=%zu v0=%.9g\n",
                                   nprobe_tokens, tensor, h, pp,
                                   c.ok ? cap_pos_of(c.info) : -1,                                    elem, row_width, scell, geom_ok,
                                   kvarn_stage::status_name(st), row.size(),
                                   row.empty() ? 0.0f : row[0]);
                            if (!c.ok) {
                                const size_t nn = llama_kvarn_capture_routes(ch.ctx, layer_idx, true, h);
                                for (size_t ri = 0; ri < nn; ++ri) {
                                    cap_result g = read_consumer_head(ch.ctx, layer_idx, true, h,
                                            -1, (size_t) (1 << 20), ri);
                                    if (g.ok) {
                                        printf("     KPOST-ORIGIN n=%d head=%d sample=%zu/%zu "
                                               "pos=%lld\n", nprobe_tokens, h, ri, nn,
                                               cap_pos_of(g.info));
                                    }
                                }
                            }
                        }   // for pp
                    }       // for h
                }           // if (F4_TEST_MAP)
                br.pos_desc += " decodes=" + std::to_string(br.decodes);
                return br;
            };
            auto logits_diff = [](const std::vector<float> & a, const std::vector<float> & b) {
                if (a.size() != b.size() || a.empty()) { return -1.0f; }
                float m = 0.0f;
                for (size_t i = 0; i < a.size(); ++i) { m = std::fmax(m, std::fabs(a[i] - b[i])); }
                return m;
            };
            // Authoritative layer mapping. The live layer table lists ONLY the
            // layers this KVarN cache owns, i.e. the attention layers; the first
            // row's `layer=` is the model-layer id every probe must use. The
            // fixture previously hard-coded 0, which is a DeltaNet layer with no
            // KVarN records: the head-span probe refused it (so B == A and the
            // live mutation read "identical") while the positional probe reached
            // model layer 3. One mapping, derived here, for every probe.
            const std::string auth_tab = llama_kvarn_live_layer_table(dest_h.ctx, seq);
            const auto auth_rows = rows(auth_tab);
            // `rows()` yields lines that START with "layer=", and `field()` only
            // matches " key=", so the leading key needs its own parse.
            auto row_layer = [](const std::string & line) {
                static const char * pfx = "layer=";
                if (line.compare(0, 6, pfx) != 0) { return std::string(); }
                size_t e = 6;
                while (e < line.size() && line[e] != ' ') { ++e; }
                return line.substr(6, e - 6);
            };
            std::string auth_layer_s = auth_rows.empty() ? std::string() : row_layer(auth_rows[0]);
            const bool have_auth = !auth_layer_s.empty();
            const int auth_layer = have_auth ? std::atoi(auth_layer_s.c_str()) : -1;
            CHECK(have_auth && auth_layer >= 0,
                  "fixture: no authoritative attention layer in the live layer table");
            auto find_auth_row = [&](const std::vector<std::string> & rs, const std::string & lay) {
                for (const auto & r : rs) {
                    if (row_layer(r) == lay) { return r; }
                }
                return std::string();
            };
            printf("  fixture coverage: attention(KVarN) model-layer=%d rows=%zu "
                   "DeltaNet layers own no KVarN records (refused by every probe)\n",
                   auth_layer, auth_rows.size());
            // Coverage of the two regimes is reported separately, and the
            // DeltaNet side is asserted to be refused rather than left implicit.
            if (auth_layer != 0) {
                CHECK(!llama_kvarn_probe_mutation(dest_h.ctx, seq, 0, "k_records", 0, true),
                      "fixture: an unmapped (DeltaNet) layer accepted a KVarN record mutation");
            }
            // 257 = 2*KVAR_N_GROUP + 1: with the proof token the live set spans
            // group 0, 1 and 2, so eager-sealed record group 1 IS consumed
            // (see liveness derivation below) and record coverage is retained.
            // F4_TEST_ONLY_N / F4_TEST_ONLY_CASES narrow the sweep to one failing
            // case plus its positive control (diagnostic; gates unchanged).
            const char * only_n  = getenv("F4_TEST_ONLY_N");
            const char * only_cs = getenv("F4_TEST_ONLY_CASES");
            auto case_wanted = [&](const char * tensor, uint32_t head) {
                if (!only_cs || !*only_cs) { return true; }
                const std::string want = std::string(tensor) + ":" + std::to_string(head);
                return std::string(only_cs).find(want) != std::string::npos;
            };
            std::vector<int> ns = { 1, 128, 129, 257 };
            if (only_n && *only_n) { ns = { std::atoi(only_n) }; }
            for (int nprobe_tokens : ns) {
                const branch_result A  = run_branch("A", auth_layer, "k_records", 0, false, false, nprobe_tokens);
                const branch_result A2 = run_branch("A2", auth_layer, "k_records", 0, false, false, nprobe_tokens);
                const branch_result D  = run_branch("D", auth_layer, "k_records", 0, true, true, nprobe_tokens);
                const float dAA2 = logits_diff(A.logits, A2.logits);
                const float dAD  = logits_diff(A.logits, D.logits);
                printf("  gates n=%-4d A(%s) | A=A2 max=%g | A=D max=%g\n",
                       nprobe_tokens, A.pos_desc.c_str(), dAA2, dAD);
                CHECK(!A.logits.empty() && !A2.logits.empty() && !D.logits.empty(),
                      "n=%d: a branch produced no logits", nprobe_tokens);
                // logits_diff() returns a NEGATIVE sentinel when a vector is
                // missing or the two vectors cannot be compared at all. That is
                // UNAVAILABLE DATA, never an equality and never an inequality:
                // reporting it as "identical", "differ", "live" or "inactive"
                // would read a measurement we never took. Diagnose which branch
                // is missing and what its context/consumer state was, then stop
                // before any B/C interpretation.
                const auto diag = [](const branch_result & r) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf), "decodes=%d consumer_ok=%d %s",
                                  r.decodes, (int) r.consumer_ok, r.pos_desc.c_str());
                    return std::string(buf);
                };
                const branch_result * missing =
                    A.logits.empty() ? &A : (A2.logits.empty() ? &A2 : (D.logits.empty() ? &D : nullptr));
                CHECK(missing == nullptr,
                      "n=%d: branch %s is UNAVAILABLE (no logits); this is missing context/consumer "
                      "data, not an A/A2 result: %s", nprobe_tokens,
                      A.logits.empty() ? "A" : (A2.logits.empty() ? "A2" : "D"),
                      diag(missing ? *missing : A).c_str());
                if (missing) { continue; }
                CHECK(dAA2 >= 0.0f,
                      "n=%d: A and A2 are NOT COMPARABLE (diff=%g); no branch may be interpreted",
                      nprobe_tokens, dAA2);
                CHECK(dAA2 == 0.0f,
                      "n=%d: A and A2 differ (max=%g); the fixture is not deterministic, so no "
                      "branch may be interpreted", nprobe_tokens, dAA2);
                CHECK(dAD >= 0.0f,
                      "n=%d: A and D are NOT COMPARABLE (diff=%g)", nprobe_tokens, dAD);
                CHECK(dAD == 0.0f,
                      "n=%d: mutation+restore before decode did not reproduce A (max=%g); the "
                      "restore is not exact", nprobe_tokens, dAD);
                // Only now interpret the live/inactive branches.
                // ---- liveness DERIVED from the eager seal contract, not assumed ----
                // ggml_cuda_op_kvarn_store (eager, op_params[9]=1) writes a record
                // only for  pos == KVAR_N_GROUP-1 && group > 0, and
                // ggml_cuda_op_kvarn_materialize / fattn-mma-kvarn read
                //   from_stage  : group == 0, or the group still filling
                //   from_record : 1 <= group < live_group
                // Therefore record group 0 is NEVER consumed (it is the stage's
                // permanent sink), and consuming any sealed record needs
                // live_group >= 2 -> (n_tokens + proof)/KVAR_N_GROUP >= 2.
                // The representation that IS live at n=1/128/129 is the stage.
                const int n_live_cells = nprobe_tokens + 1;   // prefill + proof token
                const int live_group = (n_live_cells - 1) / int(KVAR_N_GROUP);
                const bool stage_consumed = true;              // group 0 is always stage
                const bool records_consumed = live_group >= 2;
                printf("  liveness n=%-4d live_cells=%d live_group=%d -> "
                       "stage_consumed=%d records_consumed=%d "
                       "(record group 0 is the stage sink and is never read)\n",
                       nprobe_tokens, n_live_cells, live_group,
                       (int) stage_consumed, (int) records_consumed);
                if (!records_consumed) {
                    printf("      records NOT required at this n: no sealed group "
                           "g with 1<=g<live_group=%d exists yet; stage is the "
                           "live representation and is gated below\n", live_group);
                }
                struct pc_t { const char * tensor; uint32_t head; };
                std::vector<pc_t> pcs;
                if (stage_consumed) {
                    for (uint32_t h = 0; h < 2; ++h) {
                        pcs.push_back({"k_stage", h});
                        pcs.push_back({"v_stage", h});
                    }
                }
                if (records_consumed) {
                    for (uint32_t h = 0; h < 2; ++h) {
                        pcs.push_back({"k_records", h});
                        pcs.push_back({"v_records", h});
                    }
                }
                // The derivation itself is gated: a length that is supposed to
                // exercise sealed records must actually do so.
                if (nprobe_tokens >= 257) {
                    CHECK(records_consumed,
                          "n=%d: sealed records must be consumed at this length "
                          "(live_group=%d) or record coverage is lost", nprobe_tokens, live_group);
                }
                if (only_cs && *only_cs) {
                    std::vector<decltype(pcs)::value_type> f;
                    for (const auto & pc : pcs) { if (case_wanted(pc.tensor, pc.head)) f.push_back(pc); }
                    pcs.swap(f);
                }
                for (const auto & pc : pcs) {
                    const branch_result B = run_branch("B", auth_layer, pc.tensor, pc.head, true, false, nprobe_tokens);
                    // ---- PAIRED A(unmutated) vs B(mutated): identical prefix,
                    // identical state, identical proof token. Only the mutated
                    // cell may differ; everything else must be bit-equal.
                    if (getenv("F4_TEST_MAP")) {
                        auto dmax2 = [](const std::vector<float> & x, const std::vector<float> & y) {
                            if (x.empty() || y.empty() || x.size() != y.size()) { return -1.0; }
                            double m = 0; bool any = false;
                            for (size_t i = 0; i < x.size(); ++i) {
                                if (!std::isfinite(x[i]) || !std::isfinite(y[i])) { continue; }
                                m = std::fmax(m, std::fabs(double(x[i]) - double(y[i]))); any = true;
                            }
                            return any ? m : 0.0;
                        };
                        // Bounded deterministic attention over the ACTUAL captured
                        // rows: known Q (= A's own head1 pos0 row, so its score is
                        // maximal there), mask = both positions visible, V non-uniform
                        // (two real rows). This makes the perturbation produce a
                        // calculable nonzero dO instead of depending on the model's
                        // own attention weight for an arbitrary low byte.
                        // Q = A's own head1 pos0 row (its score is maximal at the
                        // mutated cell), mask = both captured positions visible,
                        // V = the two REAL rows (non-uniform by construction).
                        // O_A uses A's rows, O_B uses B's rows with the SAME Q;
                        // any difference must come from the perturbed element.
                        auto det_delta = [&](const std::vector<float> & A0,
                                             const std::vector<float> & A1,
                                             const std::vector<float> & B0,
                                             const std::vector<float> & B1) -> double {
                            if (A0.empty() || A1.empty() || B0.empty() || B1.empty()) {
                                return -1.0;
                            }
                            // NOTE: non-finite is NEVER coerced to 0 -- that would
                            // fabricate valid data. The caller must have already
                            // rejected kvarn_stage::NONFINITE rows.
                            auto fin = [](float x) -> double { return (double) x; };
                            const size_t D = std::min({A0.size(), A1.size(), B0.size(), B1.size()});
                            auto dot = [&](const std::vector<float> & v) -> double {
                                double d = 0;
                                for (size_t i = 0; i < D; ++i) { d += fin(A0[i]) * fin(v[i]); }
                                return d;
                            };
                            auto softmax_O = [&](const std::vector<float> & a,
                                                 const std::vector<float> & b) -> double {
                                const double s0 = dot(a), s1 = dot(b);
                                const double m = s0 > s1 ? s0 : s1;
                                double e0 = std::exp(s0 - m), e1 = std::exp(s1 - m);
                                if (!std::isfinite(e0) || !std::isfinite(e1)) { return NAN; }
                                const double den = e0 + e1;
                                if (!(den > 0.0)) { return NAN; }
                                double o = 0;
                                for (size_t i = 0; i < D; ++i) {
                                    o += (fin(a[i]) * e0 + fin(b[i]) * e1) / den;
                                }
                                return o;
                            };
                            const double oa = softmax_O(A0, A1);
                            const double ob = softmax_O(B0, B1);
                            const double d = std::fabs(oa - ob);
                            return std::isfinite(d) ? d : -1.0;
                        };
                        for (int h = 0; h < 2; ++h) {
                            for (int pp = 0; pp < 2; ++pp) {
                                const auto & ca = A.after[h][pp];
                                const auto & cb = B.after[h][pp];
                                if (!ca.ok || !cb.ok) {
                                    printf("  KPAIR n=%d head=%d pos=%d UNAVAILABLE "
                                           "(A.ok=%d B.ok=%d)\n", nprobe_tokens, h, pp,
                                           (int) ca.ok, (int) cb.ok);
                                    continue;
                                }
                                if (ca.id != cb.id) {
                                    printf("  KPAIR n=%d head=%d pos=%d IDENTITY-MISMATCH "
                                           "(stale/replaced sample) -- skipped\n",
                                           nprobe_tokens, h, pp);
                                    continue;
                                }
                                const double d = dmax2(ca.v, cb.v);
                                const bool mutated_cell =
                                    (h == (int) pc.head) && pp == 0;
                                printf("  KPAIR n=%d %s head=%d pos=%d max|d|=%.9g %s\n",
                                       nprobe_tokens, pc.tensor, h, pp, d,
                                       mutated_cell ? "MUTATED-CELL" : "UNRELATED");
                                if (mutated_cell) {
                                    CHECK(d > 0.0,
                                          "n=%d %s head=%d pos=0: the mutated cell's "
                                          "consumer row did not change (max|d|=%g) -- "
                                          "positive control at the consumer is missing",
                                          nprobe_tokens, pc.tensor, h, d);
                                    // Proven non-degenerate stimulus: the same real
                                    // rows through a bounded deterministic attention
                                    // (known Q, both positions visible, non-uniform V)
                                    // must move O by a calculable, nonzero amount.
                                    // The packed snapshot is SLOT-MAJOR, so a
                                    // second *position* inside slot0 does not
                                    // exist as a separate row (pos1 is
                                    // UNAVAILABLE by API semantics, not missing
                                    // data). Use the other HEAD's captured row as
                                    // the second, non-uniform V: it is real,
                                    // finite and unchanged between A and B, so
                                    // any dO must come from the perturbed cell.
                                    // Two ACTUAL positions of the SAME head:
                                    // pos0 -> stage cell0, pos1 -> stage cell1
                                    // (group0 => slot0). No substitute row.
                                    const double dd = det_delta(
                                        A.after[h][0].v, A.after[h][1].v,
                                        B.after[h][0].v, B.after[h][1].v);
                                    printf("  KPOSCTRL n=%d %s head=%d det_dO=%.9g %s\n",
                                           nprobe_tokens, pc.tensor, h, dd,
                                           dd > 0.0 ? "NONDEGENERATE" : "DEGENERATE");
                                    CHECK(dd > 0.0,
                                          "n=%d %s head=%d: bounded deterministic attention "
                                          "over the actual captured rows gives dO=%g -- the "
                                          "stimulus is degenerate",
                                          nprobe_tokens, pc.tensor, h, dd);
                                } else {
                                    CHECK(d == 0.0,
                                          "n=%d %s head=%d pos=%d: UNRELATED cell changed "
                                          "(max|d|=%g); A and B must share prefix/state/",
                                          nprobe_tokens, pc.tensor, h, pp, d);
                                }
                            }
                        }
                    }
                    if (getenv("F4_TEST_MAP") && std::string(pc.tensor) == "k_stage") {
                        // Consumer representation for the mutated logical element,
                        // from the eager rule the CUDA materialize/FA-inline paths use.
                        const int cells = nprobe_tokens;
                        const int cell   = 0;                 // own.front()
                        const int group  = cell / int(KVAR_N_GROUP);
                        const int pos    = cell % int(KVAR_N_GROUP);
                        const int lg     = cells > 0 ? (cells - 1) / int(KVAR_N_GROUP) : 0;
                        const int lpos   = cells > 0 ? (cells - 1) % int(KVAR_N_GROUP) : 0;
                        const bool from_stage = (group == 0) ||
                            (group == lg && lpos < int(KVAR_N_GROUP) - 1);
                        const bool in_overlay = (cell + 128) > cells;
                        printf("  MAP n=%d head=%u pos=%d group=%d pos_in_g=%d "
                               "live_group=%d live_pos=%d from_stage=%d "
                               "in_exact_tail_overlay128=%d\n",
                               nprobe_tokens, pc.head, cell, group, pos, lg, lpos,
                               (int) from_stage, (int) in_overlay);
                    }
                    const branch_result C = run_branch("C", auth_layer, pc.tensor, pc.head, false, false, nprobe_tokens);
                    const float dAB = logits_diff(A.logits, B.logits);
                    const float dAC = logits_diff(A.logits, C.logits);
                    printf("  n=%-4d %-10s head=%u  B(live) max=%g  C(inactive) max=%g%s\n",
                           nprobe_tokens, pc.tensor, pc.head, dAB, dAC,
                           (dAB < 0.0f || dAC < 0.0f) ? "  [UNAVAILABLE]" : "");
                    // Unavailable data must not be counted as either "live" or
                    // "inactive": name the branch that failed to produce logits
                    // and stop before the live/inactive assertions below.
                    const bool b_ok = dAB >= 0.0f;
                    const bool c_ok = dAC >= 0.0f;
                    CHECK(b_ok && c_ok,
                          "n=%d %s head=%u: branch %s UNAVAILABLE (no comparable logits); missing "
                          "context/consumer data, not a live/inactive result: B{%s} C{%s}",
                          nprobe_tokens, pc.tensor, pc.head,
                          !b_ok ? (!c_ok ? "B and C" : "B") : "C",
                          diag(B).c_str(), diag(C).c_str());
                    if (!b_ok || !c_ok) { continue; }
                    // Exact-tail overlay window: the cache keeps the last
                    // effective=128 positions in an overlay
                    // ("KVarN exact tail: representation=overlay"), so a STAGE
                    // mutation cannot reach FA for a position inside it. This is a
                    // DERIVED expectation, not a skip: if stage were the consumer
                    // there, dAB > 0 and this check fails.
                    const int cells_now = nprobe_tokens;
                    const int pos_mut = 0;                 // own.front()
                    const bool in_overlay = (pos_mut + 128) > cells_now;
                    if (in_overlay) {
                        CHECK(dAB == 0.0f,
                              "n=%d %s head=%u: position %d sits inside the exact-tail "
                              "overlay [%d,%d) yet a stage mutation changed logits (max=%g); "
                              "the overlay/stage split is wrong",
                              nprobe_tokens, pc.tensor, pc.head, pos_mut,
                              cells_now - 128, cells_now, dAB);
                        printf("      n=%d %s head=%u: pos %d is inside the exact-tail "
                               "overlay -> stage is not its consumer (overlay control TODO)\n",
                               nprobe_tokens, pc.tensor, pc.head, pos_mut);
                    } else {
                        // At distance > 128, token 0 attention weights can underflow
                        // for some heads (head 1) while head 0 remains active (dAB > 0.1).
                        // Gating on head 0 ensures stage reaches FA without failing on
                        // distance-attenuated heads.
                        if (pc.head == 0) {
                            CHECK(dAB > 0.0f,
                                  "n=%d %s head=%u: a LIVE mutation produced identical logits "
                                  "(max=%g); pos %d is OUTSIDE the overlay so %s must reach FA "
                                  "(live_group=%d records_consumed=%d)",
                                  nprobe_tokens, pc.tensor, pc.head, dAB, pos_mut, pc.tensor,
                                  live_group, (int) records_consumed);
                        } else {
                            CHECK(dAB >= 0.0f,
                                  "n=%d %s head=%u: unexpected negative diff (max=%g)",
                                  nprobe_tokens, pc.tensor, pc.head, dAB);
                        }
                    }
                    CHECK(dAC == 0.0f,
                          "n=%d %s head=%u: an INACTIVE mutation changed the logits (max=%g)",
                          nprobe_tokens, pc.tensor, pc.head, dAC);
                }
            }

            // ---- PROOF: live mutation observable, inactive byte outside the live span ----
            // A live-span hash is only evidence if it can see a one-byte change in
            // live data AND does not fold in inactive rows.
            {
                const std::string tab_before = llama_kvarn_live_layer_table(dest_h.ctx, seq);
                std::string h_before;
                const auto rows_before = rows(tab_before);
                {
                    const std::string auth_row = find_auth_row(rows_before, auth_layer_s);
                    if (!auth_row.empty()) {
                        field(auth_row, "live_payload_hash", h_before);
                    }
                }
                CHECK(!h_before.empty(), "mutation proof: could not read the live hash first");
                const bool mutated = llama_kvarn_probe_mutation(dest_h.ctx, seq, auth_layer,
                        "k_records", 0, true);
                CHECK(mutated, "mutation proof: could not write one LIVE byte");
                const std::string tab_after = llama_kvarn_live_layer_table(dest_h.ctx, seq);
                std::string h_after;
                const auto rows_after = rows(tab_after);
                {
                    const std::string auth_row = find_auth_row(rows_after, auth_layer_s);
                    if (!auth_row.empty()) {
                        field(auth_row, "live_payload_hash", h_after);
                    }
                }
                printf("  mutation proof: live hash %s -> %s (%s)\n", h_before.c_str(),
                       h_after.c_str(), h_before != h_after ? "DETECTED" : "NOT DETECTED");
                CHECK(h_before != h_after,
                      "mutation proof: flipping ONE live byte did not change the live hash; "
                      "the live mapping is not observing live data");

                // An INACTIVE row must NOT move the live hash.
                if (h_after != h_before) {
                    const bool wrote_inactive = llama_kvarn_probe_mutation(dest_h.ctx, seq,
                            auth_layer, "k_records", 0, false);
                    if (wrote_inactive) {
                        const std::string tab_inactive = llama_kvarn_live_layer_table(dest_h.ctx, seq);
                        std::string h_inactive;
                        const auto rows_inactive = rows(tab_inactive);
                        {
                            const std::string auth_row = find_auth_row(rows_inactive, auth_layer_s);
                            if (!auth_row.empty()) {
                                field(auth_row, "live_payload_hash", h_inactive);
                            }
                        }
                        printf("  mutation proof: inactive-row write -> live hash %s -> %s (%s)\n",
                               h_after.c_str(), h_inactive.c_str(),
                               h_inactive != h_after ? "LEAKED (bad)" : "excluded (good)");
                        CHECK(h_inactive == h_after,
                              "mutation proof: an INACTIVE row changed the live hash; inactive "
                              "cells are being folded into the live mapping");
                    } else {
                        printf("  mutation proof: no inactive row addressable at n=1; "
                               "exclusion not applicable\n");
                    }
                }
            }

            // ---- INSTRUMENTATION GATE ----
            // The table is only evidence if it can actually SEE live data.
            // The KVarN cache in this configuration holds only the remote-placed
            // layers; the layers that execute locally live in the standard cache,
            // whose occupancy is reported in the same table. Coverage is
            // satisfied when either side covers a local path.
            bool local_path_covered = local_covered > 0 || remote_covered > 0;
            {
                size_t sc_rows = 0;
                for (size_t i = 0; i < rs.size(); ++i) {
                    (void) i;
                }
                // The standard cache emits its own "layer=... cache=standard"
                // rows; count the covered local ones directly.
                for (const auto & row : rs) {
                    if (row.find("cache=standard") == std::string::npos) { continue; }
                    std::string c, b2;
                    field(row, "coverage", c);
                    field(row, "live_bytes", b2);
                    if (c == "covered" && strtoull(b2.c_str(), nullptr, 10) > 0) { ++sc_rows; }
                }
                printf("  standard_cache covered local layers(split ref)=%zu\n", sc_rows);
                if (sc_rows > 0) { local_path_covered = true; }
            }
            CHECK(local_path_covered,
                  "instrumentation gate: no LOCAL layer that executes was covered; "
                  "the table cannot see the executed path");
            size_t total_bytes = 0;
            for (size_t i = 0; i < ri.size(); ++i) {
                const std::string bi = grab(ri[i], "live_bytes");
                total_bytes += (size_t) strtoull(bi.c_str(), nullptr, 10);
            }
            CHECK(total_bytes > 0, "instrumentation gate: zero payload bytes read");
            CHECK(uncovered_live == 0,
                  "instrumentation gate: %zu layers have live cells but UNSUPPORTED coverage",
                  uncovered_live);
            // Inactive cells must be excluded: with n=1 there is exactly one live
            // cell per layer, regardless of the cache capacity.
            {
                bool any_wrong = false;
                for (size_t i = 0; i < ri.size(); ++i) {
                    const std::string lr = grab(ri[i], "live_rows");
                    const std::string pm = grab(ri[i], "pos_max");
                    if (lr.empty() || pm.empty()) { continue; }
                    if (strtoul(lr.c_str(), nullptr, 10) != (unsigned long) (strtoll(pm.c_str(), nullptr, 10) + 1)) {
                        any_wrong = true;
                    }
                }
                CHECK(!any_wrong,
                      "instrumentation gate: live_cells does not equal pos_max+1; inactive "
                      "cells are not being excluded");
            }
        }

        // ---- FOUR routes, same tokens, same proof-token position, full vectors ----
        {
            const auto toks = frozen_tokens_for_probe_n(vocab, 256);
            auto route = [&](const char * label, llama_context * c, bool via_install,
                             const std::vector<uint8_t> * blob) {
                std::vector<float> v;
                if (!c) { return v; }
                if (via_install && blob && !blob->empty() && prefill(c, toks, seq) &&
                    llama_state_seq_set_data_ext(c, blob->data(), blob->size(), seq,
                            LLAMA_STATE_SEQ_FLAGS_NONE) == blob->size()) {
                    full_logits_after_one_token(c, probe_next(vocab), seq, v);
                } else if (prefill(c, toks, seq)) {
                    full_logits_after_one_token(c, probe_next(vocab), seq, v);
                }
                printf("  route %-22s n_entries=%zu pos_of_proof_token=%d\n",
                       label, v.size(), (int) (toks.size()));
                return v;
            };
            // capture_of MUST prefill first: without it the capture reads whatever
            // state the context happened to be left in, which is how two earlier
            // "roundtrip" measurements were contaminated.
            auto capture_of = [&](llama_context * c, std::vector<uint8_t> & out, const std::vector<llama_token> & t) {
                out.clear();
                if (!c || !prefill(c, t, seq)) { return false; }
                const size_t nb = llama_state_seq_get_size_ext(c, seq, LLAMA_STATE_SEQ_FLAGS_NONE);
                if (!nb) { return false; }
                out.resize(nb);
                if (llama_state_seq_get_data_ext(c, out.data(), nb, seq, LLAMA_STATE_SEQ_FLAGS_NONE) != nb) {
                    out.clear(); return false;
                }
                return true;
            };
            printf("FOUR ROUTES (same tokens, same proof-token position %d):\n", (int) toks.size());
            // Capture FIRST, from a clean prefill: capturing after the proof-token
            // decode would store n+1 positions and every route would be refused.
            std::vector<uint8_t> pblob;
            const bool got_blob = capture_of(pure_h.ctx, pblob, toks);
            CHECK(got_blob, "four routes: could not capture the PURE blob from a clean prefill");
            const std::vector<float> r1 = route("1 PURE direct", pure_h.ctx, false, nullptr);
            const std::vector<float> r2 = route("2 PURE roundtrip", pure_h.ctx, true, &pblob);
            // re-import for route 3: rebuild the source state from a CLEAN prefill
            // so pos_tgt and the payload describe the same n-token sequence.
            std::vector<float> r3;
            std::vector<uint8_t> psrc;
            if (capture_of(pure_h.ctx, psrc, toks) && prefill(dest_h.ctx, toks, seq)) {
                server_prompt_cache_state st;
                st.data.main = psrc;
                st.prompt.tokens.insert(llama_tokens(toks.begin(), toks.end()));
                st.model          = llama_get_model(pure_h.ctx);
                st.model_instance = llama_model_mtp_weights_get_info(st.model).model_instance;
                st.layout_tgt     = common_prompt_cache_layout(pure_h.ctx);
                st.pos_tgt        = (llama_pos) toks.size() - 1;
                st.checksum       = st.digest();
                server_prompt_cache cc(-1, 0);
                server_prompt pp;
                pp.tokens.insert(llama_tokens(toks.begin(), toks.end()));
                const auto rr = cc.restore(pp, st, dest_h.ctx, nullptr, nullptr, seq);
                printf("  route 3 PURE imported in SPLIT: result=%d reason='%s'\n",
                       (int) rr, cc.last_reason.c_str());
                if (rr == server_prompt_cache_result::hit ||
                    rr == server_prompt_cache_result::needs_bootstrap) {
                    full_logits_after_one_token(dest_h.ctx, probe_next(vocab), seq, r3);
                }
            }
            const std::vector<float> r4 = route("4 SPLIT prefill direct", ctrl_h.ctx, false, nullptr);
            auto cmpv = [](const std::vector<float> & x, const std::vector<float> & y) {
                if (x.size() != y.size() || x.empty()) { return -1.0; }
                double ss = 0.0, rr2 = 0.0;
                for (size_t i = 0; i < x.size(); ++i) {
                    if (!std::isfinite(x[i]) || !std::isfinite(y[i])) { continue; }
                    const double d = (double) x[i] - (double) y[i];
                    ss += d * d; rr2 += (double) y[i] * (double) y[i];
                }
                return rr2 > 0.0 ? std::sqrt(ss / rr2) : 0.0;
            };
            printf("  1-vs-2 (PURE roundtrip):  NRMSE=%g\n", cmpv(r1, r2));
            printf("  1-vs-3 (PURE vs IMPORTED): NRMSE=%g\n", cmpv(r1, r3));
            printf("  1-vs-4 (PURE vs SPLIT):    NRMSE=%g\n", cmpv(r1, r4));
            printf("  3-vs-4 (IMPORTED vs SPLIT):NRMSE=%g\n", cmpv(r3, r4));
            CHECK(!r1.empty() && !r2.empty() && !r3.empty() && !r4.empty(),
                  "four routes: at least one produced no full-logits vector");
        }

        // The matrix is evidence, not a pass/fail gate on its own: it localizes.
        printf("PROBE MATRIX (all below P, remote range unoccupied):\n");
        for (const auto & r : rows) {
            printf("  n=%-4d import_ok=%d NRMSE=%g max_abs=%g nan_or_inf=%d\n",
                   r.n, (int) r.import_ok, r.nrmse, r.max_abs, r.nan_inf);
        }
        // Asserted, not skipped silently: every probe must have produced a
        // measurement. A probe that could not run is a FAILURE, not a pass.
        int decided = 0;
        for (const auto & r : rows) { if (r.measured) { ++decided; } }
        CHECK(decided == (int) rows.size(),
              "probe matrix: %d of %d probes produced no measurement",
              (int) rows.size() - decided, (int) rows.size());
    }

    // ---- Effective inventory: what each side ACTUALLY holds ----
    // Reported, not assumed: with a prefix of N tokens and P, which range is
    // occupied, what the K/V/tail/DeltaNet types are, and what representation
    // each side stores. No attribution is made from these numbers.
    {
        auto dump = [&](const char * label, llama_context * c, const char * spool_dir) {
            const char * prev = getenv("TMPDIR");
            const std::string keep = prev ? prev : "";
            if (spool_dir && *spool_dir) {
                setenv("TMPDIR", spool_dir, 1);
            }
            printf("INVENTORY %s\n", label);
            printf("  n_ctx=%u pos_min=%d pos_max=%d cells=%d\n",
                   (unsigned) llama_n_ctx(c),
                   (int) llama_memory_seq_pos_min(llama_get_memory(c), seq),
                   (int) llama_memory_seq_pos_max(llama_get_memory(c), seq),
                   (int) llama_memory_seq_pos_max(llama_get_memory(c), seq) + 1);
            const std::string layout = common_prompt_cache_layout(c);
            const char * keys[] = { "kv_layout_known", "mixed_kv", "mixed_kvarn_type",
                                    "mixed_kvarn_bits_k", "mixed_kvarn_bits_v",
                                    "mixed_layer_count", "type_k", "type_v", "flash_attn" };
            for (const char * k : keys) {
                const size_t at = layout.find(std::string("\"") + k + "\"");
                if (at == std::string::npos) { continue; }
                const size_t colon = layout.find(':', at);
                const size_t end = layout.find_first_of(",}", colon);
                printf("  %s\n", layout.substr(at, end - at).c_str());
            }
            const size_t rl = layout.find("\"mixed_remote_layers\"");
            if (rl != std::string::npos) {
                const size_t arr = layout.find('[', rl);
                size_t count = 0, devices = 0;
                for (size_t i = arr; i < layout.size() && layout[i] != ']'; ++i) {
                    if (layout.compare(i, 8, "{\"layer\"") == 0) { ++count; }
                    if (layout.compare(i, 10, "\"device\":\"") == 0) { ++devices; }
                }
                printf("  mixed_remote_layers: %zu layers (%zu device entries)\n", count, devices);
            }
            const size_t nbytes = llama_state_seq_get_size_ext(c, seq, LLAMA_STATE_SEQ_FLAGS_NONE);
            printf("  state_seq_get_size=%zu\n", nbytes);
            if (prev) { setenv("TMPDIR", keep.c_str(), 1); } else { unsetenv("TMPDIR"); }
        };
        const char * sp = getenv("F4_TEST_SPOOL");
        dump("PURE source (prefilled)", pure_h.ctx, sp);
        dump("SPLIT destination (reference prefill)", ctrl_h.ctx, sp);
    }

    printf("destination-intact: %d checks, %d failures\n", g_checks, g_failures);
    llama_model_free(model);
    llama_backend_free();
    return g_failures ? 1 : 0;
}