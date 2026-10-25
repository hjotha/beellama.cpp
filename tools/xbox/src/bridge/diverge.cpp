// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// First-numeric-divergence measurement (plan 004). See include/xllama/diverge.h.
//
// Model/prefix construction mirrors replay.cpp (same fixture, same flags,
// same teacher-forced prefix asserts) so the B3/B4 blocks start from the
// identical state the replay branches use. This file is intentionally
// self-contained: replay.cpp stays byte-stable for the A/B/C/D/E evidence.
#include "xllama/diverge.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "llama_gpu.h"
#include "xllama/chat_prompt.h"
#include "xllama/path_utils.h"
#include "xllama/platform.h"
#include "xllama/tttarget.h" // fnv1a64_file (same model-hash log line as replay)

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "decode_loop.h" // clamp_speculative_n_rs_seq (same call the session makes)

namespace xllama {
namespace {

uint64_t fnv1a(const void* data, size_t len) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < len; ++i) {
        h ^= static_cast<uint64_t>(p[i]);
        h *= 1099511628211ull;
    }
    return h;
}

int raw_argmax(const float* logits, int32_t n) {
    int best = 0;
    for (int32_t i = 1; i < n; ++i)
        if (logits[i] > logits[best])
            best = i;
    return best;
}

// Per-block capture state, pointed to by cb_eval_user_data. The callback is
// installed on every diverge context but stays dormant (active=false) during
// prefix decodes; it counts and captures only around the block decode.
//
// Memory discipline (Xbox OOM-safe): raw tensor bytes are NEVER retained.
// Each captured node keeps only first-3-row slices (hash + F32 floats) for
// candidate token axes; the raw copy is freed before the callback returns.
struct BlockCap {
    bool active = false;
    bool narrow = false;  // NARROW-TO-Z0: capture only diverge_want_narrow names
    int n_tokens = 0;     // block width (3 or 4)
    int node_idx = -1;    // increments on every ask=true while active
    int pending_idx = -1; // node index the ask=false tensor refers to
    bool read_fail = false;
    size_t asked = 0;       // ask=true count (diagnostic: graph size sanity)
    size_t captured = 0;    // stored nodes
    size_t skipped_big = 0; // over the per-tensor byte cap
    size_t n_bighash = 0;   // hash-only slices over the float cap
    size_t slice_bytes = 0; // retained slice payload (accounting)
    // Online-pairing mode (B4 runs): each captured node is paired immediately
    // against ref_nodes[paired] and only verdict rows are kept, so peak
    // retention is one run's slices, never two.
    bool pair_mode = false;
    const std::vector<DivergeCapNode>* ref_nodes = nullptr;
    std::vector<DivergeTensorRow>* out_rows = nullptr;
    int pair_rep = 0;
    std::string* pair_fail = nullptr;
    size_t paired = 0;
    int z0_hits = 0; // narrow mode: z-0 nodes seen (must be exactly 1 per run)
    std::vector<DivergeCapNode> nodes;
};

constexpr size_t kMaxCaptureBytes = 16ull * 1024ull * 1024ull;
// Slices above this keep a hash only (no floats): bounds retained memory
// while still detecting every divergence. First-divergence maxabs then comes
// from a small tensor, or is reported hash-only with a targeted follow-up.
constexpr size_t kMaxSliceFloatBytes = 1ull * 1024ull * 1024ull;
// Global retained-slice budget per block run (seatbelt only: expected totals
// are tens of MB). Tripping it skews capture symmetrically in practice; the
// node-count check below fails loudly if runs ever diverge.
constexpr size_t kMaxTotalSliceBytes = 512ull * 1024ull * 1024ull;

bool tensor_maybe_token_axis(const ggml_tensor* t) {
    // Union rule: capture tensors with a 3- or 4-extent axis in EITHER run,
    // so static dim-3/dim-4 tensors (weights, kernel histories) are captured
    // in both runs and skipped symmetrically as static, instead of skewing
    // the per-run node counts.
    for (int a = 0; a < 4; ++a)
        if (t->ne[a] == 3 || t->ne[a] == 4)
            return true;
    return false;
}

bool diverge_eval_cb(ggml_tensor* t, bool ask, void* user_data) {
    auto* cap = static_cast<BlockCap*>(user_data);
    if (!cap || !cap->active)
        return false; // dormant: no capture, scheduler runs full-graph
    if (ask) {
        ++cap->node_idx;
        ++cap->asked;
        cap->pending_idx = cap->node_idx;
        // Ask for the tensor only when it can carry token rows; the data is
        // read synchronously at ask=false before later nodes can reuse memory.
        // Narrow mode instead takes exactly the z-0 neighborhood by name.
        if (cap->narrow) {
            if (!diverge_want_narrow(t->name))
                return false;
        } else if (!tensor_maybe_token_axis(t))
            return false;
        if (t->data == nullptr)
            return false; // unallocated view: never asked-for below
        if (ggml_nbytes(t) > kMaxCaptureBytes) {
            ++cap->skipped_big;
            return false;
        }
        if (cap->slice_bytes > kMaxTotalSliceBytes)
            return false; // global seatbelt; node counts stay aligned
        return true;
    }
    // ask=false: node computed (scheduler synchronized). Slice immediately
    // and free the raw copy before returning: raw bytes never accumulate.
    if (t->data == nullptr) {
        cap->read_fail = true;
        return true;
    }
    const size_t nbytes = ggml_nbytes(t);
    std::vector<uint8_t> raw(nbytes);
    if (nbytes > 0)
        ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
    const size_t esz = ggml_type_size(t->type);
    DivergeCapNode node;
    node.index = cap->pending_idx;
    node.name = t->name;
    node.op = static_cast<int>(t->op);
    node.op_name = ggml_op_name(t->op);
    for (int a = 0; a < 4; ++a)
        node.ne[a] = t->ne[a];
    node.type = static_cast<int>(t->type);
    const bool is_f32 = (t->type == GGML_TYPE_F32);
    int64_t ne[4] = {t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
    size_t nb[4] = {t->nb[0], t->nb[1], t->nb[2], t->nb[3]};
    // Candidate token axes in this run; store one slice per singleton plus
    // the full-combo slice, so pair-time lookup by rule mask always hits.
    int cand_mask = 0;
    for (int a = 0; a < 4; ++a)
        if (t->ne[a] == 3 || t->ne[a] == 4)
            cand_mask |= (1 << a);
    auto store_slice = [&](int mask, int64_t nstart) {
        const std::vector<size_t> offs = diverge_slice_offsets(ne, nb, esz, mask, nstart, 3);
        if (offs.empty())
            return;
        DivergeCapNode::AxisSlice sl;
        sl.axes = mask;
        sl.nstart = nstart;
        std::vector<uint8_t> sbytes;
        sbytes.reserve(offs.size() * esz);
        for (size_t off : offs) {
            if (off + esz > raw.size())
                return; // out of bounds: skip this mask, keep others
            sbytes.insert(sbytes.end(), raw.begin() + off, raw.begin() + off + esz);
        }
        if (sbytes.empty())
            return;
        sl.hash = diverge_fnv(sbytes.data(), sbytes.size());
        // Shifted windows (nstart 1, B4 only) are hash-first: floats only
        // when small, since their maxabs only matters for snapshot-family
        // magnitude reporting, never for detection.
        const size_t float_cap = (nstart == 0) ? kMaxSliceFloatBytes : 256ull * 1024ull;
        if (is_f32 && sbytes.size() % sizeof(float) == 0 && sbytes.size() <= float_cap) {
            sl.floats.resize(sbytes.size() / sizeof(float));
            memcpy(sl.floats.data(), sbytes.data(), sbytes.size());
        } else if (is_f32) {
            ++cap->n_bighash;
        }
        cap->slice_bytes += sbytes.size() + sl.floats.size() * sizeof(float);
        node.slices.push_back(std::move(sl));
    };
    for (int a = 0; a < 4; ++a)
        if (cand_mask & (1 << a))
            store_slice(1 << a, 0);
    int pop = 0;
    for (int m = cand_mask; m; m &= m - 1)
        ++pop;
    if (pop > 1)
        store_slice(cand_mask, 0);
    // B4 runs additionally keep the one-position-shifted window so snapshot
    // slots (reverse-chronological) compare at the same logical positions.
    // B3 never needs it (it is always the reference side).
    if (cap->n_tokens == 4) {
        for (int a = 0; a < 4; ++a)
            if (cand_mask & (1 << a))
                store_slice(1 << a, 1);
        if (pop > 1)
            store_slice(cand_mask, 1);
    }
    ++cap->captured;
    if (cap->narrow && std::strcmp(node.name.c_str(), "z-0") == 0) {
        // Source identities from tensor METADATA only (no src data reads:
        // graph memory may already be reused for this node's output).
        // Buffer-type names name the owning backend's arena directly.
        ++cap->z0_hits;
        const ggml_tensor* s0 = (t->op == GGML_OP_MUL_MAT) ? t->src[0] : nullptr;
        const ggml_tensor* s1 = (t->op == GGML_OP_MUL_MAT) ? t->src[1] : nullptr;
        auto buft_name = [](const ggml_backend_buffer_t buf) -> const char* {
            if (buf == nullptr)
                return "-";
            const char* n = ggml_backend_buft_name(ggml_backend_buffer_get_type(buf));
            return (n != nullptr && n[0] != '\0') ? n : "?";
        };
        int64_t zne[4] = {node.ne[0], node.ne[1], node.ne[2], node.ne[3]};
        int64_t s0ne[4] = {0, 0, 0, 0}, s1ne[4] = {0, 0, 0, 0};
        const char *s0n = "-", *s1n = "-", *s0t = "?", *s1t = "?";
        if (s0 != nullptr) {
            for (int a = 0; a < 4; ++a)
                s0ne[a] = s0->ne[a];
            s0n = s0->name[0] != '\0' ? s0->name : "-";
            s0t = ggml_type_name(s0->type);
        }
        if (s1 != nullptr) {
            for (int a = 0; a < 4; ++a)
                s1ne[a] = s1->ne[a];
            s1n = s1->name[0] != '\0' ? s1->name : "-";
            s1t = ggml_type_name(s1->type);
        }
        // Buffer arenas via public getters (no struct access): which backend
        // owns each tensor at compute time.
        // Reuse/overwrite test: hash src[1]'s bytes AS READ NOW (at z-0
        // compute time). The separately captured norm node holds the same
        // tensor's bytes as read at ITS completion; equality means the
        // activation survived intact, difference proves buffer overwrite or
        // aliasing between production and consumption. Reads are safe:
        // worst case is recycled-but-mapped bytes, which is the signal.
        uint64_t src_now_hash = 0;
        uint64_t src_first3_hash = 0;
        bool src_read_ok = false;
        if (s1 != nullptr && s1->buffer != nullptr) {
            const size_t snbytes = ggml_nbytes(s1);
            if (snbytes > 0 && snbytes <= kMaxCaptureBytes) {
                std::vector<uint8_t> sraw(snbytes);
                ggml_backend_tensor_get(s1, sraw.data(), 0, snbytes);
                src_now_hash = diverge_fnv(sraw.data(), sraw.size());
                int s_axis = -1;
                for (int a = 0; a < 4; ++a)
                    if (s1->ne[a] == (cap->n_tokens == 4 ? 4 : 3))
                        s_axis = a;
                if (s_axis >= 0) {
                    int64_t sne[4] = {s1->ne[0], s1->ne[1], s1->ne[2], s1->ne[3]};
                    size_t snb[4] = {s1->nb[0], s1->nb[1], s1->nb[2], s1->nb[3]};
                    const size_t sesz = ggml_type_size(s1->type);
                    const std::vector<size_t> soffs =
                        diverge_slice_offsets(sne, snb, sesz, (1 << s_axis), 0, 3);
                    std::vector<uint8_t> sfirst;
                    bool soob = false;
                    for (size_t off : soffs) {
                        if (off + sesz > sraw.size()) {
                            soob = true;
                            break;
                        }
                        sfirst.insert(sfirst.end(), sraw.begin() + off, sraw.begin() + off + sesz);
                    }
                    if (!soob && !sfirst.empty()) {
                        src_first3_hash = diverge_fnv(sfirst.data(), sfirst.size());
                        src_read_ok = true;
                    }
                }
            }
        }
        char sxb[160];
        snprintf(sxb, sizeof(sxb), "srcx read=%d full=%016llx first3=%016llx", src_read_ok ? 1 : 0,
                 static_cast<unsigned long long>(src_now_hash),
                 static_cast<unsigned long long>(src_first3_hash));
        log_output(("[xllama] diverge zsrc idx=" + std::to_string(node.index) +
                    " z=" + diverge_tensor_id("z-0", zne, ggml_type_name(t->type)) +
                    " src0=" + diverge_tensor_id(s0n, s0ne, s0t) +
                    " src1=" + diverge_tensor_id(s1n, s1ne, s1t) +
                    " buf=" + std::string(buft_name(t->buffer)) + "/" +
                    std::string(buft_name(s0 != nullptr ? s0->buffer : nullptr)) + "/" +
                    std::string(buft_name(s1 != nullptr ? s1->buffer : nullptr)) + " " + sxb + "\n")
                       .c_str());
    }
    if (cap->pair_mode) {
        // Online pairing: emit verdict rows immediately and drop the node,
        // so B4 slices never accumulate next to the retained B3 set.
        if (!cap->ref_nodes || !cap->out_rows || cap->paired >= cap->ref_nodes->size()) {
            if (cap->pair_fail && cap->pair_fail->empty())
                *cap->pair_fail = "diverge online pairing overran reference";
            return true;
        }
        const DivergeCapNode& ref = (*cap->ref_nodes)[cap->paired];
        if (node.index != ref.index) {
            if (cap->pair_fail && cap->pair_fail->empty())
                *cap->pair_fail = "diverge online pairing index drift";
            return true;
        }
        diverge_emit_pair(cap->pair_rep, ref, node, cap->out_rows);
        ++cap->paired;
        return true;
    }
    cap->nodes.push_back(std::move(node));
    return true; // never abort the graph from the diagnostic
}

} // namespace

uint64_t diverge_fnv(const void* data, size_t len) {
    return fnv1a(data, len);
}

int diverge_token_axes(const int64_t n3[4], const int64_t n4[4]) {
    int mask = 0;
    for (int a = 0; a < 4; ++a) {
        if (n3[a] == n4[a])
            continue; // static axis (any value, including static 3 or 4)
        if (n3[a] == 3 && n4[a] == 4) {
            mask |= (1 << a);
            continue;
        }
        return -1; // a non-token axis depends on width: genuine drift
    }
    if (mask == 0)
        return -2; // static tensor
    return mask;
}

std::vector<size_t> diverge_slice_offsets(const int64_t ne[4], const size_t nb[4], size_t esz,
                                          int axes_mask, int64_t nstart, int64_t nkeep) {
    std::vector<size_t> offs;
    if (axes_mask <= 0 || axes_mask > 15 || nstart < 0 || nkeep <= 0 || esz == 0)
        return offs;
    // Walk the full tensor in linear-offset order, keeping elements whose
    // index along axis is < nkeep. Index reconstruction from linear offset
    // uses strides: fastest axis first.
    const size_t total = static_cast<size_t>(ne[0]) * static_cast<size_t>(ne[1]) *
                         static_cast<size_t>(ne[2]) * static_cast<size_t>(ne[3]);
    if (total == 0)
        return offs;
    // Stride-sorted axes (ascending nb) for odometer walk.
    int order[4] = {0, 1, 2, 3};
    std::sort(order, order + 4, [&](int a, int b) { return nb[a] < nb[b]; });
    int64_t idx[4] = {0, 0, 0, 0};
    for (size_t lin = 0; lin < total; ++lin) {
        bool keep = true;
        for (int a = 0; a < 4; ++a)
            if ((axes_mask & (1 << a)) && (idx[a] < nstart || idx[a] >= nstart + nkeep))
                keep = false;
        if (keep) {
            const size_t off =
                static_cast<size_t>(idx[0]) * nb[0] + static_cast<size_t>(idx[1]) * nb[1] +
                static_cast<size_t>(idx[2]) * nb[2] + static_cast<size_t>(idx[3]) * nb[3];
            offs.push_back(off);
            if (offs.size() * esz > kMaxCaptureBytes * 4)
                break; // pathological: caller treats short list as skip
        }
        for (int k = 0; k < 4; ++k) {
            const int a = order[k];
            if (++idx[a] < ne[a])
                break;
            idx[a] = 0;
        }
    }
    return offs;
}

double diverge_maxabs_f32(const float* a, const float* b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs(static_cast<double>(a[i]) - b[i]);
        if (d > m)
            m = d;
        if (!std::isfinite(d))
            return std::numeric_limits<double>::infinity();
    }
    return m;
}

bool diverge_want_narrow(const char* name) {
    if (name == nullptr)
        return false;
    return std::strcmp(name, "z-0") == 0 || std::strcmp(name, "norm-0") == 0 ||
           std::strcmp(name, "attn_norm-0") == 0;
}

std::string diverge_tensor_id(const char* name, const int64_t ne[4], const char* type) {
    char lb[256];
    snprintf(lb, sizeof(lb), "%s[%s %lldx%lldx%lldx%lld]", name ? name : "-", type ? type : "?",
             static_cast<long long>(ne[0]), static_cast<long long>(ne[1]),
             static_cast<long long>(ne[2]), static_cast<long long>(ne[3]));
    return std::string(lb);
}

DivergeCmp compare_diverge_rows(const DivergeTensorRow& a, const DivergeTensorRow& b) {
    DivergeCmp c;
    if (a.slice.size() != b.slice.size() || a.slice.empty())
        return c;
    c.hash_match = (diverge_fnv(a.slice.data(), a.slice.size() * sizeof(float)) ==
                    diverge_fnv(b.slice.data(), b.slice.size() * sizeof(float)));
    c.maxabs = diverge_maxabs_f32(a.slice.data(), b.slice.data(), a.slice.size());
    return c;
}

// Pair one B3/B4 capture node into two export rows (b3 then b4) carrying the
// unshifted + shifted measurements. Structural/slice problems are marked as
// skips (the exporter fails loudly on those); value mismatches are evidence.
// Pure over the node structs: host-testable with hand-built fixtures. Used
// both by the online B4 pairing (bounded memory) and nowhere else.
void diverge_emit_pair(int rep, const DivergeCapNode& n3, const DivergeCapNode& n4,
                       std::vector<DivergeTensorRow>* out) {
    const int64_t ne3[4] = {n3.ne[0], n3.ne[1], n3.ne[2], n3.ne[3]};
    const int64_t ne4[4] = {n4.ne[0], n4.ne[1], n4.ne[2], n4.ne[3]};
    const int mask = diverge_token_axes(ne3, ne4);
    const bool struct_ok = (n3.op == n4.op) && (n3.type == n4.type) && (n3.name == n4.name);
    auto find_slice = [&](const DivergeCapNode& nd, int64_t nstart, uint64_t* hash,
                          std::vector<float>* floats) {
        for (const auto& sl : nd.slices) {
            if (sl.axes != mask || sl.nstart != nstart)
                continue;
            *hash = sl.hash;
            *floats = sl.floats;
            return true;
        }
        return false;
    };
    auto fill = [&](const DivergeCapNode& nd, const char* branch, DivergeTensorRow* row) {
        row->branch = branch;
        row->rep = rep;
        row->index = nd.index;
        row->name = nd.name;
        row->op = nd.op;
        row->op_name = nd.op_name;
        for (int a = 0; a < 4; ++a)
            row->ne[a] = nd.ne[a];
        row->type = nd.type;
        row->axes = mask;
        row->hash = 0;
        row->hash_alt = 0;
        row->maxabs = -1.0;
        row->maxabs_alt = -1.0;
        if (!struct_ok || mask < 0) {
            row->skip_why = !struct_ok ? "structural" : mask == -2 ? "static" : "shape-drift";
            return;
        }
        // Slices were extracted inline during capture (raw freed).
        uint64_t h = 0;
        std::vector<float> f;
        if (!find_slice(nd, 0, &h, &f)) {
            row->skip_why = "slice-missing";
            return;
        }
        row->hash = h;
        row->slice = std::move(f);
    };
    DivergeTensorRow r3, r4;
    fill(n3, "b3", &r3);
    fill(n4, "b4", &r4);
    r3.hash_alt = r3.hash; // B3 side: shifted window is the same slice
    // Shifted window (B4 rows 1..3, same logical positions as B3 rows 0..2
    // for reverse-chronological snapshot slots). This site ships measurements
    // only; the shared helper computes the tiered verdict (host-tested).
    // Payload freed right after.
    if (r3.skip_why.empty() && r4.skip_why.empty()) {
        for (const auto& sl : n4.slices) {
            if (sl.axes != mask || sl.nstart != 1)
                continue;
            r4.hash_alt = sl.hash;
            if (!r3.slice.empty() && r3.slice.size() == sl.floats.size() && !sl.floats.empty())
                r4.maxabs_alt =
                    diverge_maxabs_f32(r3.slice.data(), sl.floats.data(), r3.slice.size());
            break;
        }
        if (!r3.slice.empty() && r3.slice.size() == r4.slice.size()) {
            const double mu = diverge_maxabs_f32(r3.slice.data(), r4.slice.data(), r3.slice.size());
            r3.maxabs = mu;
            r4.maxabs = mu;
        }
    }
    r3.slice.clear();
    r3.slice.shrink_to_fit();
    r4.slice.clear();
    r4.slice.shrink_to_fit();
    out->push_back(std::move(r3));
    out->push_back(std::move(r4));
}

int diverge_tier_by_name(const std::string& op_name) {
    // Tier-1: per-token/causal compute ops whose same-token rows are directly
    // comparable. Everything else (state presentation, recurrent fused ops,
    // scatters, custom/MTP ops, anything unlisted) is Tier-2: diagnostic
    // rows only, never ranked. Conservative default is deliberate.
    static const char* const kTier1[] = {
        "MUL_MAT", "MUL", "ADD", "RMS_NORM", "UNARY", "GLU", "ROPE", "SCALE",
    };
    for (const char* n : kTier1)
        if (op_name == n)
            return 1;
    return 2;
}
std::string diverge_shape_string(const int64_t ne[4], int type) {
    char sb[128];
    snprintf(sb, sizeof(sb), "\"[%lld,%lld,%lld,%lld]/t%d\"", static_cast<long long>(ne[0]),
             static_cast<long long>(ne[1]), static_cast<long long>(ne[2]),
             static_cast<long long>(ne[3]), type);
    return std::string(sb);
}

std::string diverge_row_string(const DivergeTensorRow& r) {
    char lb[320];
    snprintf(lb, sizeof(lb), "%s idx=%d name=%s op=%s ne=[%lld,%lld,%lld,%lld]/t%d skip=%s",
             r.branch.c_str(), r.index, r.name.empty() ? "-" : r.name.c_str(),
             r.op_name.empty() ? "?" : r.op_name.c_str(), static_cast<long long>(r.ne[0]),
             static_cast<long long>(r.ne[1]), static_cast<long long>(r.ne[2]),
             static_cast<long long>(r.ne[3]), r.type,
             r.skip_why.empty() ? "-" : r.skip_why.c_str());
    return std::string(lb);
}

std::string diverge_csv_line(const DivergePairOut& p) {
    const std::string nm = p.name.empty() ? "-" : p.name;
    const std::string op = p.op_name.empty() ? "?" : p.op_name;
    const std::string note = p.note.empty() ? "-" : p.note;
    char lb[640];
    snprintf(lb, sizeof(lb), "%d,%d,%s,%s,%s,%s,%d,%.6g,%s\n", p.rep, p.index, nm.c_str(),
             op.c_str(), diverge_shape_string(p.ne_a, p.type).c_str(),
             diverge_shape_string(p.ne_b, p.type).c_str(), p.match, p.maxabs, note.c_str());
    return std::string(lb);
}

DivergeExport diverge_pair_rows(const std::vector<DivergeTensorRow>& rows) {
    DivergeExport ex;
    if (rows.size() % 4 != 0 || rows.empty()) {
        ex.fail_why = "diverge row count is not 4-balanced";
        return ex;
    }
    const size_t n = rows.size() / 4;
    for (int rep = 0; rep < 2 && ex.fail_why.empty(); ++rep) {
        const size_t base = static_cast<size_t>(rep) * 2 * n;
        DivergeRepSummary summary;
        summary.rep = rep;
        summary.n_tensors = n;
        auto mark_fail = [&](const std::string& why, size_t i) {
            ex.fail_why = why;
            ex.fail_pair = static_cast<int>(i);
            ex.fail_a = rows[base + 2 * i];
            ex.fail_b = rows[base + 2 * i + 1];
        };
        for (size_t i = 0; i < n; ++i) {
            const auto& a = rows[base + 2 * i];
            const auto& b = rows[base + 2 * i + 1];
            // Captured indices are sparse (static tensors are never asked
            // for), so only cross-branch index equality is required here,
            // never index==i.
            if (a.branch != "b3" || b.branch != "b4" || a.rep != rep || b.rep != rep ||
                a.index != b.index) {
                mark_fail("diverge pairing broken", i);
                break;
            }
            DivergePairOut p;
            p.rep = rep;
            p.pair = static_cast<int>(i);
            p.index = a.index;
            p.name = a.name;
            p.op_name = a.op_name;
            for (int k = 0; k < 4; ++k) {
                p.ne_a[k] = a.ne[k];
                p.ne_b[k] = b.ne[k];
            }
            p.type = a.type;
            if (!a.skip_why.empty() || !b.skip_why.empty()) {
                if (a.skip_why != b.skip_why) {
                    mark_fail("diverge asymmetric skip", i);
                    break;
                }
                if (a.skip_why == "shape-drift" || a.skip_why == "structural" ||
                    a.skip_why == "slice-missing") {
                    mark_fail(std::string("diverge ") + a.skip_why, i);
                    break;
                }
                ++summary.n_skipped;
                p.match = 1;
                p.maxabs = -1.0;
                p.note = std::string("skipped:") + a.skip_why;
                ex.pairs.push_back(std::move(p));
                continue;
            }
            if (a.op != b.op || a.type != b.type || a.name != b.name) {
                mark_fail("diverge structural drift within rep", i);
                break;
            }
            // Tiered verdict (provenance, never values): Tier-1 per-token
            // compute is judged on the unshifted (same-token) alignment
            // only; Tier-2 (state-family/uncertain) emits both alignments as
            // diagnostic rows and never ranks. The hash recheck is
            // independent (stored per-row hashes); maxabs consistency guards
            // device packaging.
            if (a.maxabs != b.maxabs) {
                mark_fail("diverge stored verdict mismatch", i);
                break;
            }
            const int tier = diverge_tier_by_name(a.op_name);
            if (tier == 1) {
                p.match = (a.hash == b.hash) ? 1 : 0;
                p.maxabs = a.maxabs;
                p.note = "";
                if (p.match)
                    ++summary.n_matched;
                else if (summary.first_diff < 0)
                    summary.first_diff = static_cast<int>(i);
                ex.pairs.push_back(std::move(p));
                continue;
            }
            // Tier-2 diagnostic pair: unshifted row first, shifted row second.
            DivergePairOut q = p;
            p.match = (a.hash == b.hash) ? 1 : 0;
            p.maxabs = a.maxabs;
            p.note = "tier2:unshifted";
            if (p.match)
                ++summary.n_matched;
            ex.pairs.push_back(std::move(p));
            q.match = (a.hash == b.hash_alt) ? 1 : 0;
            q.maxabs = b.maxabs_alt;
            q.note = "tier2:shifted";
            if (q.match)
                ++summary.n_matched;
            ex.pairs.push_back(std::move(q));
        }
        ex.reps.push_back(summary);
    }
    return ex;
}

void run_diverge_measure(const std::string& model_name, const std::string& prompt_text,
                         const std::vector<int32_t>& ctx_ids, const std::vector<int32_t>& feed_ids,
                         const std::vector<int32_t>& known_next, int gpu_layers, int n_ctx,
                         int n_threads, int rs_seq, bool nextn_on,
                         std::vector<DivergeTensorRow>* out, std::string* fail_why) {
    if (!out || !fail_why)
        return;
    auto fail = [&](const std::string& msg) {
        *fail_why = msg;
        log_output(("[xllama] diverge FAIL: " + msg + "\n").c_str());
    };
    // Same fixture contract as the replay branches: ctx 23 committed ids,
    // feed 4 ids (B3 = first three, 4th rejected), known 4 sequential argmax.
    if (ctx_ids.size() != 23 || feed_ids.size() != 4 || known_next.size() != 4) {
        fail("diverge files must hold: ctx 23 ids, feed 4 ids, known 4 ids");
        return;
    }

    const std::string model_dir = resolve_model_path(model_name);
    const std::string abs_model_path = first_gguf_in_dir(model_dir);
    if (abs_model_path.empty()) {
        fail("no .gguf in model dir: " + model_dir);
        return;
    }
    std::uint64_t gguf_bytes = 0;
    const std::string gguf_hash = fnv1a64_file(abs_model_path, &gguf_bytes);
    {
        char lb[256];
        snprintf(lb, sizeof(lb), "[xllama] diverge model=%s gguf=%s bytes=%llu gpu_layers=%d\n",
                 abs_model_path.c_str(), gguf_hash.c_str(),
                 static_cast<unsigned long long>(gguf_bytes), gpu_layers);
        log_output(lb);
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.load_mtp = true; // same model object as the failing runs
    const int applied = apply_gguf_gpu_layers(gpu_layers, mparams);
    llama_model* model = llama_model_load_from_file(abs_model_path.c_str(), mparams);
    if (!model) {
        fail("model load failed: " + abs_model_path);
        return;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    const ChatFormat fmt = chat_format_for(model_name);
    const std::string full_prompt =
        fmt.render_prompt("You are a helpful AI assistant.", {}, prompt_text);
    int32_t n_tok =
        llama_tokenize(vocab, full_prompt.c_str(), static_cast<int32_t>(full_prompt.size()),
                       nullptr, 0, true, false);
    if (n_tok >= 0) {
        fail("prompt tokenize size query failed");
        llama_model_free(model);
        return;
    }
    std::vector<llama_token> prompt_ids(static_cast<size_t>(-n_tok));
    n_tok = llama_tokenize(vocab, full_prompt.c_str(), static_cast<int32_t>(full_prompt.size()),
                           prompt_ids.data(), static_cast<int32_t>(prompt_ids.size()), true, false);
    if (n_tok <= 0) {
        fail("prompt tokenization failed");
        llama_model_free(model);
        return;
    }
    prompt_ids.resize(static_cast<size_t>(n_tok));
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    if (prompt_ids.size() != 119 || ctx_ids.size() != 23) {
        fail("diverge prefix mismatch: want prompt 119 + ctx 23");
        llama_model_free(model);
        return;
    }
    const llama_pos p_block = static_cast<llama_pos>(prompt_ids.size() + ctx_ids.size());
    if (p_block != 142) {
        fail("diverge block position is not the observed 142");
        llama_model_free(model);
        return;
    }

    const int n_thr = n_threads > 0 ? n_threads : detect_threads_llama();

    auto single_decode = [&](llama_context* ctx, llama_token tok, llama_pos pos,
                             std::vector<float>* logits_out) {
        llama_batch b = llama_batch_init(1, 0, 1);
        b.token[0] = tok;
        b.pos[0] = pos;
        b.n_seq_id[0] = 1;
        b.seq_id[0][0] = 0;
        b.logits[0] = 1;
        b.n_tokens = 1;
        const int rc = llama_decode(ctx, b);
        llama_batch_free(b);
        if (rc != 0)
            return false;
        const float* lg = llama_get_logits_ith(ctx, 0);
        if (!lg)
            return false;
        logits_out->assign(lg, lg + n_vocab);
        return true;
    };

    auto block_decode = [&](llama_context* ctx, const int32_t* toks, int n,
                            std::vector<std::vector<float>>* rows_out) {
        llama_batch b = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; ++i) {
            b.token[i] = toks[i];
            b.pos[i] = p_block + i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = 1;
        }
        b.n_tokens = n;
        const int rc = llama_decode(ctx, b);
        llama_batch_free(b);
        if (rc != 0)
            return false;
        rows_out->clear();
        for (int i = 0; i < n; ++i) {
            const float* lg = llama_get_logits_ith(ctx, i);
            if (!lg)
                return false;
            rows_out->emplace_back(lg, lg + n_vocab);
        }
        return true;
    };

    // Teacher-forced prefix identical to the replay branches; every argmax
    // asserted against the known sequential trajectory.
    auto build_prefix = [&](llama_context* ctx, std::string* why) {
        llama_batch pre = llama_batch_init(static_cast<int32_t>(prompt_ids.size()), 0, 1);
        for (size_t i = 0; i < prompt_ids.size(); ++i) {
            pre.token[i] = prompt_ids[i];
            pre.pos[i] = static_cast<llama_pos>(i);
            pre.n_seq_id[i] = 1;
            pre.seq_id[i][0] = 0;
            pre.logits[i] = 0;
        }
        pre.n_tokens = static_cast<int32_t>(prompt_ids.size());
        if (llama_decode(ctx, pre) != 0) {
            *why = "prefix prefill failed";
            llama_batch_free(pre);
            return false;
        }
        llama_batch_free(pre);
        for (size_t i = 0; i < ctx_ids.size(); ++i) {
            const llama_pos pos = static_cast<llama_pos>(prompt_ids.size() + i);
            std::vector<float> lg;
            if (!single_decode(ctx, ctx_ids[i], pos, &lg)) {
                *why = "prefix force-decode failed";
                return false;
            }
            const int want = (i + 1 < ctx_ids.size()) ? ctx_ids[i + 1] : feed_ids[0];
            if (raw_argmax(lg.data(), n_vocab) != want) {
                *why = "prefix diverged (staging error, not evidence)";
                return false;
            }
        }
        return true;
    };

    for (int rep = 0; rep < 2; ++rep) {
        // Four runs per rep on fresh contexts: B3 and B4, each once without
        // the callback (contamination control) and once with it (capture).
        // Callback-off runs execute the normal full-graph schedule.
        struct RunOut {
            std::vector<std::vector<float>> rows;
            std::vector<DivergeCapNode> nodes;
            int node_count = 0;
        };
        RunOut b3off, b3on, b4off, b4on;
        const int32_t b3toks[3] = {feed_ids[0], feed_ids[1], feed_ids[2]};
        const int32_t b4toks[4] = {feed_ids[0], feed_ids[1], feed_ids[2], feed_ids[3]};

        auto run_once = [&](const int32_t* toks, int n, bool capture, RunOut* ro,
                            const char* what) {
            BlockCap cap;
            cap.n_tokens = n;
            llama_context_params cp = llama_context_default_params();
            cp.n_ctx = static_cast<std::uint32_t>(n_ctx > 0 ? n_ctx : 2048);
            cp.n_threads = n_thr;
            cp.n_threads_batch = n_thr;
            clamp_speculative_n_rs_seq(cp, static_cast<std::uint32_t>(rs_seq < 0 ? 0 : rs_seq));
            apply_gguf_gpu_context(applied, cp);
            cp.cb_eval = diverge_eval_cb;
            cp.cb_eval_user_data = &cap;
            llama_context* ctx = llama_init_from_model(model, cp);
            if (!ctx) {
                fail(std::string(what) + ": context creation failed");
                return;
            }
            {
                char plb[160];
                snprintf(plb, sizeof(plb), "[xllama] diverge %s rep=%d ctx-ready n=%d capture=%d\n",
                         what, rep, n, capture ? 1 : 0);
                log_output(plb);
            }
            std::string why;
            if (!build_prefix(ctx, &why)) {
                fail(std::string(what) + " prefix: " + why);
                llama_free(ctx);
                return;
            }
            cap.active = capture; // callback live only around the block decode
            if (!block_decode(ctx, toks, n, &ro->rows)) {
                fail(std::string(what) + ": block decode failed");
                cap.active = false;
                llama_free(ctx);
                return;
            }
            cap.active = false;
            if (cap.read_fail) {
                fail(std::string(what) + ": tensor read failed during capture");
                llama_free(ctx);
                return;
            }
            // B3 rows must reproduce the known sequential steps (staging
            // guard, same as the replay branches); B4 row 3 is the rejected
            // token (argmax 567 live) and is recorded, never asserted.
            const int n_assert = n == 3 ? 3 : 2;
            for (int i = 0; i < n_assert; ++i) {
                if (raw_argmax(ro->rows[static_cast<size_t>(i)].data(), n_vocab) !=
                    known_next[static_cast<size_t>(i)]) {
                    fail(std::string(what) + ": block row diverged (staging error)");
                    llama_free(ctx);
                    return;
                }
            }
            ro->nodes = std::move(cap.nodes);
            ro->node_count = cap.node_idx + 1;
            {
                // Progress boundary: the next run's silence is diagnosable.
                char plb[256];
                snprintf(plb, sizeof(plb),
                         "[xllama] diverge %s rep=%d done rows=%d node_count=%d asked=%d "
                         "captured=%d skipped_big=%d bighash=%d slice_kb=%d\n",
                         what, rep, static_cast<int>(ro->rows.size()), ro->node_count,
                         static_cast<int>(cap.asked), static_cast<int>(cap.captured),
                         static_cast<int>(cap.skipped_big), static_cast<int>(cap.n_bighash),
                         static_cast<int>(cap.slice_bytes / 1024));
                log_output(plb);
            }
            llama_free(ctx);
        };

        run_once(b3toks, 3, false, &b3off, "diverge B3 off");
        if (!fail_why->empty()) {
            llama_model_free(model);
            return;
        }
        run_once(b3toks, 3, true, &b3on, "diverge B3 on");
        if (!fail_why->empty()) {
            llama_model_free(model);
            return;
        }
        run_once(b4toks, 4, false, &b4off, "diverge B4 off");
        if (!fail_why->empty()) {
            llama_model_free(model);
            return;
        }
        // B4 capture run with ONLINE pairing: each node pairs immediately
        // against the retained B3 set via diverge_emit_pair, so B4 slices
        // never accumulate next to it (peak = one run). Same interleaved
        // rows land in out in the same order post-hoc pairing produced.
        {
            BlockCap cap;
            cap.n_tokens = 4;
            llama_context_params cp = llama_context_default_params();
            cp.n_ctx = static_cast<std::uint32_t>(n_ctx > 0 ? n_ctx : 2048);
            cp.n_threads = n_thr;
            cp.n_threads_batch = n_thr;
            clamp_speculative_n_rs_seq(cp, static_cast<std::uint32_t>(rs_seq < 0 ? 0 : rs_seq));
            apply_gguf_gpu_context(applied, cp);
            cp.cb_eval = diverge_eval_cb;
            cp.cb_eval_user_data = &cap;
            llama_context* ctx = llama_init_from_model(model, cp);
            if (!ctx) {
                fail("diverge B4 on: context creation failed");
                llama_model_free(model);
                return;
            }
            {
                char plb[160];
                snprintf(plb, sizeof(plb),
                         "[xllama] diverge B4 on rep=%d ctx-ready n=4 capture=1\n", rep);
                log_output(plb);
            }
            std::string why;
            if (!build_prefix(ctx, &why)) {
                fail(std::string("diverge B4 on prefix: ") + why);
                llama_free(ctx);
                llama_model_free(model);
                return;
            }
            cap.pair_mode = true;
            cap.ref_nodes = &b3on.nodes;
            cap.out_rows = out;
            cap.pair_rep = rep;
            std::string pair_fail;
            cap.pair_fail = &pair_fail;
            const size_t out_mark = out->size();
            cap.active = true; // callback live only around the block decode
            if (!block_decode(ctx, b4toks, 4, &b4on.rows)) {
                fail("diverge B4 on: block decode failed");
                cap.active = false;
                llama_free(ctx);
                llama_model_free(model);
                return;
            }
            cap.active = false;
            if (cap.read_fail) {
                fail("diverge B4 on: tensor read failed during capture");
                llama_free(ctx);
                llama_model_free(model);
                return;
            }
            for (int i = 0; i < 2; ++i) {
                if (raw_argmax(b4on.rows[static_cast<size_t>(i)].data(), n_vocab) !=
                    known_next[static_cast<size_t>(i)]) {
                    fail("diverge B4 on: block row diverged (staging error)");
                    llama_free(ctx);
                    llama_model_free(model);
                    return;
                }
            }
            b4on.node_count = cap.node_idx + 1;
            {
                char plb[256];
                snprintf(plb, sizeof(plb),
                         "[xllama] diverge B4 on rep=%d done rows=%d node_count=%d asked=%d "
                         "captured=%d paired=%d slice_kb=%d\n",
                         rep, static_cast<int>(b4on.rows.size()), b4on.node_count,
                         static_cast<int>(cap.asked), static_cast<int>(cap.captured),
                         static_cast<int>(cap.paired), static_cast<int>(cap.slice_bytes / 1024));
                log_output(plb);
            }
            if (!pair_fail.empty() || cap.paired != b3on.nodes.size()) {
                fail(pair_fail.empty() ? "diverge online pairing count drift" : pair_fail);
                out->resize(out_mark);
                llama_free(ctx);
                llama_model_free(model);
                return;
            }
            // Pairing consumed the B3 reference set: free it now.
            b3on.nodes.clear();
            b3on.nodes.shrink_to_fit();
            llama_free(ctx);
        }

        // Contamination control: callback scheduling must not move values.
        // Compare full logit rows off-vs-on for both widths (bit-exact).
        auto rows_equal = [&](const std::vector<std::vector<float>>& a,
                              const std::vector<std::vector<float>>& b) {
            if (a.size() != b.size())
                return false;
            for (size_t i = 0; i < a.size(); ++i) {
                if (a[i].size() != b[i].size())
                    return false;
                if (diverge_fnv(a[i].data(), a[i].size() * sizeof(float)) !=
                    diverge_fnv(b[i].data(), b[i].size() * sizeof(float)))
                    return false;
            }
            return true;
        };
        if (!rows_equal(b3off.rows, b3on.rows) || !rows_equal(b4off.rows, b4on.rows)) {
            fail("callback scheduling changed block rows: capture contaminated, discarded");
            llama_model_free(model);
            return;
        }
        if (b3on.node_count != b4on.node_count) {
            char lb[160];
            snprintf(lb, sizeof(lb), "graph node count differs B3=%d B4=%d: structural",
                     b3on.node_count, b4on.node_count);
            fail(lb);
            llama_model_free(model);
            return;
        }

        // (Pairing already done online during the B4 run above; B3 slices freed.)
    }
    llama_model_free(model);
}

void run_znarrow_measure(const std::string& model_name, const std::string& prompt_text,
                         const std::vector<int32_t>& ctx_ids, const std::vector<int32_t>& feed_ids,
                         const std::vector<int32_t>& known_next, int gpu_layers, int n_ctx,
                         int n_threads, int rs_seq, bool nextn_on,
                         std::vector<DivergeTensorRow>* out, std::string* fail_why) {
    // NARROW-TO-Z0: same fixture/flags/prefix as run_diverge_measure (kept
    // deliberately duplicated so the broad path stays byte-stable), but the
    // capture filter takes only the z-0 neighborhood by name plus src-meta
    // logging. Bounded by construction: ≤3 small tensors per run.
    if (!out || !fail_why)
        return;
    auto fail = [&](const std::string& msg) {
        *fail_why = msg;
        log_output(("[xllama] znarrow FAIL: " + msg + "\n").c_str());
    };
    if (ctx_ids.size() != 23 || feed_ids.size() != 4 || known_next.size() != 4) {
        fail("znarrow files must hold: ctx 23 ids, feed 4 ids, known 4 ids");
        return;
    }

    const std::string model_dir = resolve_model_path(model_name);
    const std::string abs_model_path = first_gguf_in_dir(model_dir);
    if (abs_model_path.empty()) {
        fail("no .gguf in model dir: " + model_dir);
        return;
    }
    std::uint64_t gguf_bytes = 0;
    const std::string gguf_hash = fnv1a64_file(abs_model_path, &gguf_bytes);
    {
        char lb[256];
        snprintf(lb, sizeof(lb), "[xllama] znarrow model=%s gguf=%s bytes=%llu gpu_layers=%d\n",
                 abs_model_path.c_str(), gguf_hash.c_str(),
                 static_cast<unsigned long long>(gguf_bytes), gpu_layers);
        log_output(lb);
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.load_mtp = true; // same model object as the failing runs
    const int applied = apply_gguf_gpu_layers(gpu_layers, mparams);
    llama_model* model = llama_model_load_from_file(abs_model_path.c_str(), mparams);
    if (!model) {
        fail("model load failed: " + abs_model_path);
        return;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    const ChatFormat fmt = chat_format_for(model_name);
    const std::string full_prompt =
        fmt.render_prompt("You are a helpful AI assistant.", {}, prompt_text);
    int32_t n_tok =
        llama_tokenize(vocab, full_prompt.c_str(), static_cast<int32_t>(full_prompt.size()),
                       nullptr, 0, true, false);
    if (n_tok >= 0) {
        fail("prompt tokenize size query failed");
        llama_model_free(model);
        return;
    }
    std::vector<llama_token> prompt_ids(static_cast<size_t>(-n_tok));
    n_tok = llama_tokenize(vocab, full_prompt.c_str(), static_cast<int32_t>(full_prompt.size()),
                           prompt_ids.data(), static_cast<int32_t>(prompt_ids.size()), true, false);
    if (n_tok <= 0) {
        fail("prompt tokenization failed");
        llama_model_free(model);
        return;
    }
    prompt_ids.resize(static_cast<size_t>(n_tok));
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    if (prompt_ids.size() != 119 || ctx_ids.size() != 23) {
        fail("znarrow prefix mismatch: want prompt 119 + ctx 23");
        llama_model_free(model);
        return;
    }
    const llama_pos p_block = static_cast<llama_pos>(prompt_ids.size() + ctx_ids.size());
    if (p_block != 142) {
        fail("znarrow block position is not the observed 142");
        llama_model_free(model);
        return;
    }

    const int n_thr = n_threads > 0 ? n_threads : detect_threads_llama();

    auto single_decode = [&](llama_context* ctx, llama_token tok, llama_pos pos,
                             std::vector<float>* logits_out) {
        llama_batch b = llama_batch_init(1, 0, 1);
        b.token[0] = tok;
        b.pos[0] = pos;
        b.n_seq_id[0] = 1;
        b.seq_id[0][0] = 0;
        b.logits[0] = 1;
        b.n_tokens = 1;
        const int rc = llama_decode(ctx, b);
        llama_batch_free(b);
        if (rc != 0)
            return false;
        const float* lg = llama_get_logits_ith(ctx, 0);
        if (!lg)
            return false;
        logits_out->assign(lg, lg + n_vocab);
        return true;
    };

    auto block_decode = [&](llama_context* ctx, const int32_t* toks, int n,
                            std::vector<std::vector<float>>* rows_out) {
        llama_batch b = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; ++i) {
            b.token[i] = toks[i];
            b.pos[i] = p_block + i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = 1;
        }
        b.n_tokens = n;
        const int rc = llama_decode(ctx, b);
        llama_batch_free(b);
        if (rc != 0)
            return false;
        rows_out->clear();
        for (int i = 0; i < n; ++i) {
            const float* lg = llama_get_logits_ith(ctx, i);
            if (!lg)
                return false;
            rows_out->emplace_back(lg, lg + n_vocab);
        }
        return true;
    };

    auto build_prefix = [&](llama_context* ctx, std::string* why) {
        llama_batch pre = llama_batch_init(static_cast<int32_t>(prompt_ids.size()), 0, 1);
        for (size_t i = 0; i < prompt_ids.size(); ++i) {
            pre.token[i] = prompt_ids[i];
            pre.pos[i] = static_cast<llama_pos>(i);
            pre.n_seq_id[i] = 1;
            pre.seq_id[i][0] = 0;
            pre.logits[i] = 0;
        }
        pre.n_tokens = static_cast<int32_t>(prompt_ids.size());
        if (llama_decode(ctx, pre) != 0) {
            *why = "prefix prefill failed";
            llama_batch_free(pre);
            return false;
        }
        llama_batch_free(pre);
        for (size_t i = 0; i < ctx_ids.size(); ++i) {
            const llama_pos pos = static_cast<llama_pos>(prompt_ids.size() + i);
            std::vector<float> lg;
            if (!single_decode(ctx, ctx_ids[i], pos, &lg)) {
                *why = "prefix force-decode failed";
                return false;
            }
            const int want = (i + 1 < ctx_ids.size()) ? ctx_ids[i + 1] : feed_ids[0];
            if (raw_argmax(lg.data(), n_vocab) != want) {
                *why = "prefix diverged (staging error, not evidence)";
                return false;
            }
        }
        return true;
    };

    struct NarrowOut {
        std::vector<std::vector<float>> rows;
        std::vector<DivergeCapNode> nodes;
        int node_count = 0;
    };

    for (int rep = 0; rep < 2; ++rep) {
        NarrowOut b3off, b3on, b4off, b4on;
        const int32_t b3toks[3] = {feed_ids[0], feed_ids[1], feed_ids[2]};
        const int32_t b4toks[4] = {feed_ids[0], feed_ids[1], feed_ids[2], feed_ids[3]};

        auto run_once = [&](const int32_t* toks, int n, bool capture, NarrowOut* ro,
                            const char* what) {
            BlockCap cap;
            cap.n_tokens = n;
            cap.narrow = true; // name-filtered capture (z-0 neighborhood only)
            llama_context_params cp = llama_context_default_params();
            cp.n_ctx = static_cast<std::uint32_t>(n_ctx > 0 ? n_ctx : 2048);
            cp.n_threads = n_thr;
            cp.n_threads_batch = n_thr;
            clamp_speculative_n_rs_seq(cp, static_cast<std::uint32_t>(rs_seq < 0 ? 0 : rs_seq));
            apply_gguf_gpu_context(applied, cp);
            cp.cb_eval = diverge_eval_cb;
            cp.cb_eval_user_data = &cap;
            llama_context* ctx = llama_init_from_model(model, cp);
            if (!ctx) {
                fail(std::string(what) + ": context creation failed");
                return;
            }
            {
                char plb[160];
                snprintf(plb, sizeof(plb), "[xllama] znarrow %s rep=%d ctx-ready n=%d capture=%d\n",
                         what, rep, n, capture ? 1 : 0);
                log_output(plb);
            }
            std::string why;
            if (!build_prefix(ctx, &why)) {
                fail(std::string(what) + " prefix: " + why);
                llama_free(ctx);
                return;
            }
            cap.active = capture; // callback live only around the block decode
            if (!block_decode(ctx, toks, n, &ro->rows)) {
                fail(std::string(what) + ": block decode failed");
                cap.active = false;
                llama_free(ctx);
                return;
            }
            cap.active = false;
            if (cap.read_fail) {
                fail(std::string(what) + ": tensor read failed during capture");
                llama_free(ctx);
                return;
            }
            const int n_assert = n == 3 ? 3 : 2;
            for (int i = 0; i < n_assert; ++i) {
                if (raw_argmax(ro->rows[static_cast<size_t>(i)].data(), n_vocab) !=
                    known_next[static_cast<size_t>(i)]) {
                    fail(std::string(what) + ": block row diverged (staging error)");
                    llama_free(ctx);
                    return;
                }
            }
            if (capture && cap.z0_hits != 1) {
                char lb[128];
                snprintf(lb, sizeof(lb), "z-0 seen %d times, want exactly 1", cap.z0_hits);
                fail(std::string(what) + ": " + lb);
                llama_free(ctx);
                return;
            }
            ro->nodes = std::move(cap.nodes);
            ro->node_count = cap.node_idx + 1;
            {
                char plb[256];
                snprintf(plb, sizeof(plb),
                         "[xllama] znarrow %s rep=%d done rows=%d node_count=%d asked=%d "
                         "captured=%d\n",
                         what, rep, static_cast<int>(ro->rows.size()), ro->node_count,
                         static_cast<int>(cap.asked), static_cast<int>(cap.captured));
                log_output(plb);
            }
            llama_free(ctx);
        };

        run_once(b3toks, 3, false, &b3off, "znarrow B3 off");
        if (!fail_why->empty()) {
            llama_model_free(model);
            return;
        }
        run_once(b3toks, 3, true, &b3on, "znarrow B3 on");
        if (!fail_why->empty()) {
            llama_model_free(model);
            return;
        }
        run_once(b4toks, 4, false, &b4off, "znarrow B4 off");
        if (!fail_why->empty()) {
            llama_model_free(model);
            return;
        }
        run_once(b4toks, 4, true, &b4on, "znarrow B4 on");
        if (!fail_why->empty()) {
            llama_model_free(model);
            return;
        }

        // Contamination control: callback scheduling must not move values.
        auto rows_equal = [&](const std::vector<std::vector<float>>& a,
                              const std::vector<std::vector<float>>& b) {
            if (a.size() != b.size())
                return false;
            for (size_t i = 0; i < a.size(); ++i) {
                if (a[i].size() != b[i].size())
                    return false;
                if (diverge_fnv(a[i].data(), a[i].size() * sizeof(float)) !=
                    diverge_fnv(b[i].data(), b[i].size() * sizeof(float)))
                    return false;
            }
            return true;
        };
        if (!rows_equal(b3off.rows, b3on.rows) || !rows_equal(b4off.rows, b4on.rows)) {
            fail("callback scheduling changed block rows: capture contaminated, discarded");
            llama_model_free(model);
            return;
        }
        if (b3on.nodes.size() != b4on.nodes.size()) {
            fail("narrow capture node count differs B3 vs B4 within rep");
            llama_model_free(model);
            return;
        }
        // Post-hoc pairing is trivially bounded here (<=3 small nodes); the
        // tested shared pair function is reused verbatim.
        for (size_t i = 0; i < b3on.nodes.size(); ++i)
            diverge_emit_pair(rep, b3on.nodes[i], b4on.nodes[i], out);
    }
    llama_model_free(model);
}

} // namespace xllama
