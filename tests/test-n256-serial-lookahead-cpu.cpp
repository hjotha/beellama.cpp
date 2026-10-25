// CPU harness: two real N256 ubatches, serial vs lookahead, tensor-by-tensor.
//
// REAL: ggml CPU backend, F32 tensors, ggml_mul_mat/add/scale/silu/cpy, output
// buffers with slot generations/lifetime, bitwise memcmp of every captured
// node and committed state/prefix.
// SYNTHETIC/FIXTURE: deterministic weights/inputs, simplified attention and a
// GDN-like recurrent state (not the real Qwen3.8 GDN kernel/model), and a
// logical event scheduler; CPU compute is synchronous, so this proves ordering
// and tensor equality for the fixture, NOT wall-clock overlap, real model
// quality, GDN correctness, TPS or GPU behavior.
//
// Layers: L0 full/remote KV, L1 full/remote KV, L2 recurrent/horizontal state.
// Ubatch U0 is committed/published before U1 may read its state or publish U1.
// Serial: U0 L0..L2, then U1 L0..L2.
// Lookahead: legal pairs (U1 Attn(L), U0 QKV(L+1)) for L0 and L1, with all
// dependencies checked before issue/complete.
//
// No model file, GPU, RPC, server, Xbox or benchmark. V2 FAIL, P 96256/102400
// and trim NO-GO are untouched by this fixture.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kN = 256;      // tokens per ubatch
constexpr int kE = 64;       // input/embedding width (uniform fixture)
constexpr int kD = 64;       // qkv/merge/ffn width (one GEMM N=256)
constexpr int kLayers = 3;   // L0/L1 full, L2 recurrent
constexpr int kUb = 2;       // U0/U1
constexpr int kKinds = 5;    // QKV, ATTN, MERGE, FFN, STATE
constexpr int kNodes = kLayers * kUb * kKinds;
constexpr int kSlots = 8;

int failures = 0;

void check(bool ok, const std::string & name) {
    std::printf("%s: %s\n", name.c_str(), ok ? "OK" : "FAIL");
    if (!ok) ++failures;
}

void fatal(const std::string & msg) {
    std::printf("FATAL: %s\n", msg.c_str());
    ++failures;
}

enum class Kind { Qkv = 0, Attn = 1, Merge = 2, Ffn = 3, State = 4 };

const char * kind_name(Kind k) {
    switch (k) {
        case Kind::Qkv: return "qkv";
        case Kind::Attn: return "attn";
        case Kind::Merge: return "merge";
        case Kind::Ffn: return "ffn";
        case Kind::State: return "state";
    }
    return "?";
}

bool is_full(int layer) { return layer < 2; }

int nid(Kind k, int layer, int ub) {
    return (layer * kUb + ub) * kKinds + static_cast<int>(k);
}

struct Dep {
    int id = -1;
    bool slot = false; // true: consumes the producer's real output slot
};

struct Node {
    int id = -1;
    Kind kind = Kind::Qkv;
    int layer = 0;
    int ub = 0;
    int ndeps = 0;
    Dep deps[3];
};

std::array<Node, kNodes> make_nodes() {
    std::array<Node, kNodes> n{};
    for (int l = 0; l < kLayers; ++l) {
        for (int u = 0; u < kUb; ++u) {
            for (int k = 0; k < kKinds; ++k) {
                Node & x = n[nid(static_cast<Kind>(k), l, u)];
                x.id = nid(static_cast<Kind>(k), l, u);
                x.kind = static_cast<Kind>(k);
                x.layer = l;
                x.ub = u;
                switch (x.kind) {
                    case Kind::Qkv:
                        if (l > 0) x.deps[x.ndeps++] = {nid(Kind::Ffn, l - 1, u), true};
                        break;
                    case Kind::Attn:
                        x.deps[x.ndeps++] = {nid(Kind::Qkv, l, u), true};
                        if (u == 1) x.deps[x.ndeps++] = {nid(Kind::State, l, 0), false};
                        break;
                    case Kind::Merge:
                        x.deps[x.ndeps++] = {nid(Kind::Qkv, l, u), true};
                        x.deps[x.ndeps++] = {nid(Kind::Attn, l, u), true};
                        if (u == 1 && is_full(l)) x.deps[x.ndeps++] = {nid(Kind::Merge, l, 0), false};
                        break;
                    case Kind::Ffn:
                        x.deps[x.ndeps++] = {nid(Kind::Merge, l, u), true};
                        break;
                    case Kind::State:
                        x.deps[x.ndeps++] = {nid(Kind::Qkv, l, u), true};
                        if (u == 1) x.deps[x.ndeps++] = {nid(Kind::State, l, 0), false};
                        break;
                }
            }
        }
    }
    return n;
}

struct Handle {
    int idx = -1;
    uint64_t gen = 0;
};

struct SlotPool {
    struct Slot {
        bool live = false;
        uint64_t gen = 0;
        int owner = -1;
        int remaining = 0;
    };
    std::array<Slot, kSlots> slots{};
    ggml_tensor * tensors[kSlots] = {};
    int allocs = 0;
    int reuses = 0;

    int alloc(int owner, int consumers, std::string & err) {
        for (int i = 0; i < kSlots; ++i) {
            if (!slots[i].live) {
                slots[i].live = true;
                if (slots[i].gen > 0) ++reuses;
                ++allocs;
                ++slots[i].gen;
                slots[i].owner = owner;
                slots[i].remaining = consumers;
                return i;
            }
        }
        err = "no free slot";
        return -1;
    }

    bool consume(const Handle & h, int consumer, std::string & err) {
        if (h.idx < 0 || h.idx >= kSlots || !slots[h.idx].live) {
            err = "consume of non-live slot";
            return false;
        }
        if (slots[h.idx].gen != h.gen) {
            err = "stale generation";
            return false;
        }
        if (slots[h.idx].remaining <= 0) {
            err = "consume after last consumer";
            return false;
        }
        if (slots[h.idx].owner == consumer) {
            err = "self alias";
            return false;
        }
        --slots[h.idx].remaining;
        if (slots[h.idx].remaining == 0) {
            slots[h.idx].live = false;
            slots[h.idx].owner = -1;
        }
        return true;
    }

    void release(int idx) {
        if (idx < 0 || idx >= kSlots) return;
        slots[idx].live = false;
        slots[idx].owner = -1;
        slots[idx].remaining = 0;
    }

    int live_count() const {
        int c = 0;
        for (const auto & s : slots) c += s.live ? 1 : 0;
        return c;
    }
};

std::vector<float> make_data(size_t n, int seed) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        const int q = static_cast<int>((i * 37 + seed * 101) % 199) - 99;
        v[i] = 0.001f * static_cast<float>(q);
    }
    return v;
}

void set_vec(ggml_tensor * t, const std::vector<float> & v) {
    std::memcpy(t->data, v.data(), v.size() * sizeof(float));
}

std::vector<float> get_vec(ggml_tensor * t) {
    std::vector<float> v(static_cast<size_t>(ggml_nelements(t)));
    std::memcpy(v.data(), t->data, v.size() * sizeof(float));
    return v;
}

bool same_vec(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

struct Run {
    ggml_context * ctx = nullptr;
    ggml_tensor * wqkv = nullptr;
    ggml_tensor * wffn = nullptr;
    ggml_tensor * x[kUb] = {};
    ggml_tensor * zero = nullptr;
    ggml_tensor * prefix = nullptr;
    ggml_tensor * merge_prefix = nullptr;
    ggml_tensor * state_prefix = nullptr;
    SlotPool pool;
    ggml_tensor * slot_t[kSlots] = {};
    Handle slot[kNodes];
    std::array<std::array<std::vector<float>, kUb>, kLayers> state{};
    std::array<std::vector<float>, kLayers> merge0{};
    bool state_pub[kLayers][kUb] = {};
    bool merge_pub[kLayers] = {};
    std::array<std::vector<float>, kNodes> out{};
    bool issued[kNodes] = {};
    bool done[kNodes] = {};
    std::vector<std::string> events;
    int pairs = 0;
};

const std::array<Node, kNodes> & nodes() {
    static const std::array<Node, kNodes> n = make_nodes();
    return n;
}

std::array<int, kNodes> make_consumers() {
    std::array<int, kNodes> c{};
    for (const Node & n : nodes()) {
        for (int i = 0; i < n.ndeps; ++i) {
            if (n.deps[i].slot) ++c[n.deps[i].id];
        }
    }
    return c;
}

const std::array<int, kNodes> & consumers() {
    static const std::array<int, kNodes> c = make_consumers();
    return c;
}

bool init_run(Run & r, ggml_backend_t backend, int x1_seed) {
    ggml_init_params p = {
        /* .mem_size   = */ 256ull * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc    = */ false,
    };
    r.ctx = ggml_init(p);
    if (!r.ctx) return false;

    r.wqkv = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F32, kE, kD);
    r.wffn = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F32, kD, kD);
    r.x[0] = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F32, kE, kN);
    r.x[1] = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F32, kE, kN);
    r.zero = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F32, kD, kN);
    r.prefix = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F32, kD, kN);
    r.merge_prefix = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F32, kD, kN);
    r.state_prefix = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F32, kD, kN);
    for (int i = 0; i < kSlots; ++i) {
        r.slot_t[i] = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F32, kD, kN);
    }

    set_vec(r.wqkv, make_data(kE * kD, 3));
    set_vec(r.wffn, make_data(kD * kD, 5));
    set_vec(r.x[0], make_data(kE * kN, 7));
    set_vec(r.x[1], make_data(kE * kN, x1_seed));
    set_vec(r.zero, std::vector<float>(kD * kN, 0.0f));
    set_vec(r.prefix, std::vector<float>(kD * kN, 0.0f));
    set_vec(r.merge_prefix, std::vector<float>(kD * kN, 0.0f));
    set_vec(r.state_prefix, std::vector<float>(kD * kN, 0.0f));

    for (int l = 0; l < kLayers; ++l) {
        for (int u = 0; u < kUb; ++u) r.state[l][u].assign(kD * kN, 0.0f);
        r.merge0[l].assign(kD * kN, 0.0f);
    }
    return true;
}

void free_run(Run & r) {
    if (r.ctx) ggml_free(r.ctx);
    r = Run{};
}

bool deps_ok(const Run & r, const Node & n, std::string & why) {
    for (int i = 0; i < n.ndeps; ++i) {
        const int d = n.deps[i].id;
        if (!r.done[d]) {
            why = "dependency not done: " + std::to_string(d);
            return false;
        }
    }
    if (n.ub == 1) {
        if (n.kind == Kind::Attn && !r.state_pub[n.layer][0]) {
            why = "U1 attn before U0 state publish";
            return false;
        }
        if (n.kind == Kind::Merge && is_full(n.layer) && !r.merge_pub[n.layer]) {
            why = "U1 full merge before U0 merge publish";
            return false;
        }
        if (n.kind == Kind::State && !r.state_pub[n.layer][0]) {
            why = "U1 state before U0 state publish";
            return false;
        }
    }
    return true;
}

ggml_tensor * dep_slot(const Run & r, const Node & n, Kind kind) {
    for (int i = 0; i < n.ndeps; ++i) {
        const Node & d = nodes()[n.deps[i].id];
        if (d.kind == kind) {
            if (!n.deps[i].slot) return nullptr;
            return r.slot_t[r.slot[d.id].idx];
        }
    }
    return nullptr;
}

bool compute_node(Run & r, ggml_backend_t backend, const Node & n) {
    ggml_context * ctx = r.ctx;
    ggml_tensor * out = r.slot_t[r.slot[n.id].idx];
    ggml_tensor * result = nullptr;

    switch (n.kind) {
        case Kind::Qkv: {
            ggml_tensor * in = n.layer > 0
                ? r.slot_t[r.slot[nid(Kind::Ffn, n.layer - 1, n.ub)].idx]
                : r.x[n.ub];
            result = ggml_mul_mat(ctx, r.wqkv, in);
            break;
        }
        case Kind::Attn: {
            ggml_tensor * qkv = dep_slot(r, n, Kind::Qkv);
            if (!qkv) return false;
            if (n.ub == 1) {
                set_vec(r.prefix, r.state[n.layer][0]);
            } else {
                set_vec(r.prefix, std::vector<float>(kD * kN, 0.0f));
            }
            result = ggml_scale(ctx, ggml_add(ctx, qkv, r.prefix), 0.5f);
            break;
        }
        case Kind::Merge: {
            ggml_tensor * qkv = dep_slot(r, n, Kind::Qkv);
            ggml_tensor * attn = dep_slot(r, n, Kind::Attn);
            if (!qkv || !attn) return false;
            ggml_tensor * base = qkv;
            if (n.ub == 1 && is_full(n.layer)) {
                set_vec(r.merge_prefix, r.merge0[n.layer]);
                base = r.merge_prefix;
            }
            result = ggml_scale(ctx, ggml_add(ctx, attn, base), 0.5f);
            break;
        }
        case Kind::Ffn: {
            ggml_tensor * merge = dep_slot(r, n, Kind::Merge);
            if (!merge) return false;
            result = ggml_silu(ctx, ggml_mul_mat(ctx, r.wffn, merge));
            break;
        }
        case Kind::State: {
            ggml_tensor * qkv = dep_slot(r, n, Kind::Qkv);
            if (!qkv) return false;
            if (n.ub == 1) {
                set_vec(r.state_prefix, r.state[n.layer][0]);
            } else {
                set_vec(r.state_prefix, std::vector<float>(kD * kN, 0.0f));
            }
            result = ggml_scale(ctx, ggml_add(ctx, r.state_prefix, qkv), 0.5f);
            break;
        }
    }
    if (!result) return false;

    ggml_tensor * cpy = ggml_cpy(ctx, result, out);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cpy);
    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::printf("compute %s L%d U%d failed: %s\n", kind_name(n.kind), n.layer, n.ub,
                    ggml_status_to_string(status));
        return false;
    }
    r.out[n.id] = get_vec(out);
    return true;
}

bool issue_node(Run & r, const Node & n, bool lookahead_pair) {
    if (r.issued[n.id] || r.done[n.id]) {
        std::printf("issue duplicate node %d\n", n.id);
        return false;
    }
    std::string why;
    if (!deps_ok(r, n, why)) {
        std::printf("issue %s L%d U%d rejected: %s\n", kind_name(n.kind), n.layer, n.ub, why.c_str());
        return false;
    }
    std::string err;
    const int idx = r.pool.alloc(n.id, consumers()[n.id], err);
    if (idx < 0) {
        std::printf("issue %s L%d U%d: %s\n", kind_name(n.kind), n.layer, n.ub, err.c_str());
        return false;
    }
    r.slot[n.id] = Handle{idx, r.pool.slots[idx].gen};
    r.issued[n.id] = true;
    r.events.push_back(std::string("issue ") + (lookahead_pair ? "pair " : "") +
                       kind_name(n.kind) + " L" + std::to_string(n.layer) +
                       " U" + std::to_string(n.ub));
    return true;
}

bool complete_node(Run & r, ggml_backend_t backend, const Node & n) {
    if (!r.issued[n.id] || r.done[n.id]) {
        std::printf("complete without issue node %d\n", n.id);
        return false;
    }
    if (!compute_node(r, backend, n)) return false;

    if (n.kind == Kind::State) {
        r.state[n.layer][n.ub] = r.out[n.id];
        if (n.ub == 1 && !r.state_pub[n.layer][0]) {
            std::printf("state U1 published before U0 L%d\n", n.layer);
            return false;
        }
        r.state_pub[n.layer][n.ub] = true;
    }
    if (n.kind == Kind::Merge && n.ub == 0) {
        r.merge0[n.layer] = r.out[n.id];
        r.merge_pub[n.layer] = true;
    }

    // Consume only real data-slot edges; state/merge-prefix edges are logical
    // ordering edges serviced through host->tensor prefix copies.
    for (int i = 0; i < n.ndeps; ++i) {
        if (!n.deps[i].slot) continue;
        std::string err;
        if (!r.pool.consume(r.slot[n.deps[i].id], n.id, err)) {
            std::printf("consume %s L%d U%d -> %d failed: %s\n", kind_name(n.kind), n.layer, n.ub,
                        n.deps[i].id, err.c_str());
            return false;
        }
    }
    r.done[n.id] = true;
    r.events.push_back(std::string("complete ") + kind_name(n.kind) + " L" +
                       std::to_string(n.layer) + " U" + std::to_string(n.ub));
    if (consumers()[n.id] == 0) {
        r.pool.release(r.slot[n.id].idx);
        r.slot[n.id] = Handle{};
    }
    return true;
}

struct Step {
    int a = -1;
    int b = -1; // >=0: legal lookahead pair, issue both before completing
};

std::vector<Step> serial_steps() {
    std::vector<Step> s;
    for (int u = 0; u < kUb; ++u) {
        for (int l = 0; l < kLayers; ++l) {
            for (int k = 0; k < kKinds; ++k) s.push_back({nid(static_cast<Kind>(k), l, u), -1});
        }
    }
    return s;
}

std::vector<Step> lookahead_steps() {
    std::vector<Step> s;
    auto add = [&](Kind k, int l, int u) { s.push_back({nid(k, l, u), -1}); };
    auto pair = [&](Kind ka, int la, int ua, Kind kb, int lb, int ub) {
        s.push_back({nid(ka, la, ua), nid(kb, lb, ub)});
    };
    // U0 L0 establishes the first committed/published prefix.
    add(Kind::Qkv, 0, 0); add(Kind::Attn, 0, 0); add(Kind::Merge, 0, 0);
    add(Kind::Ffn, 0, 0); add(Kind::State, 0, 0);
    add(Kind::Qkv, 0, 1);
    // Legal pair 1: remote U1 L0 Attn while GPU U0 L1 QKV runs.
    pair(Kind::Attn, 0, 1, Kind::Qkv, 1, 0);
    add(Kind::Attn, 1, 0); add(Kind::Merge, 1, 0); add(Kind::Ffn, 1, 0); add(Kind::State, 1, 0);
    add(Kind::Merge, 0, 1); add(Kind::Ffn, 0, 1); add(Kind::State, 0, 1);
    add(Kind::Qkv, 1, 1);
    // Legal pair 2: remote U1 L1 Attn while GPU U0 L2 QKV runs.
    pair(Kind::Attn, 1, 1, Kind::Qkv, 2, 0);
    add(Kind::Attn, 2, 0); add(Kind::Merge, 2, 0); add(Kind::Ffn, 2, 0); add(Kind::State, 2, 0);
    add(Kind::Merge, 1, 1); add(Kind::Ffn, 1, 1); add(Kind::State, 1, 1);
    add(Kind::Qkv, 2, 1); add(Kind::Attn, 2, 1); add(Kind::Merge, 2, 1);
    add(Kind::Ffn, 2, 1); add(Kind::State, 2, 1);
    return s;
}

bool run_schedule(Run & r, ggml_backend_t backend, bool lookahead, int x1_seed) {
    if (!init_run(r, backend, x1_seed)) {
        fatal("init_run failed");
        return false;
    }
    const auto steps = lookahead ? lookahead_steps() : serial_steps();
    for (const Step & s : steps) {
        if (s.b >= 0) {
            if (!issue_node(r, nodes()[s.a], true)) return false;
            if (!issue_node(r, nodes()[s.b], true)) return false;
            // Both are issued before either completes: logical overlap.
            if (!complete_node(r, backend, nodes()[s.a])) return false;
            if (!complete_node(r, backend, nodes()[s.b])) return false;
            ++r.pairs;
        } else {
            if (!issue_node(r, nodes()[s.a], false)) return false;
            if (!complete_node(r, backend, nodes()[s.a])) return false;
        }
    }
    int done = 0;
    for (int i = 0; i < kNodes; ++i) done += r.done[i] ? 1 : 0;
    check(done == kNodes, (lookahead ? "lookahead" : "serial") + std::string(": all nodes completed"));
    check(r.pool.live_count() == 0, (lookahead ? "lookahead" : "serial") + std::string(": all slots released"));
    check(r.pool.allocs >= kNodes, (lookahead ? "lookahead" : "serial") + std::string(": every node allocated a slot"));
    check(r.pool.reuses > 0, (lookahead ? "lookahead" : "serial") + std::string(": slots actually reused"));
    return done == kNodes && r.pool.live_count() == 0;
}

bool compare_runs(const Run & a, const Run & b, const char * label) {
    bool ok = true;
    int mismatches = 0;
    for (int id = 0; id < kNodes; ++id) {
        if (!same_vec(a.out[id], b.out[id])) {
            ++mismatches;
            if (mismatches <= 5) {
                const Node & n = nodes()[id];
                std::printf("MISMATCH %s %s L%d U%d\n", label, kind_name(n.kind), n.layer, n.ub);
            }
            ok = false;
        }
    }
    for (int l = 0; l < kLayers; ++l) {
        for (int u = 0; u < kUb; ++u) {
            if (!same_vec(a.state[l][u], b.state[l][u])) {
                std::printf("MISMATCH %s state L%d U%d\n", label, l, u);
                ok = false;
            }
        }
        if (!same_vec(a.merge0[l], b.merge0[l])) {
            std::printf("MISMATCH %s merge0 L%d\n", label, l);
            ok = false;
        }
    }
    check(ok, std::string(label) + ": tensors/state bitwise equal");
    check(mismatches == 0, std::string(label) + ": zero tensor mismatches");
    return ok;
}

bool all_finite(const Run & r) {
    for (int id = 0; id < kNodes; ++id) {
        for (const float v : r.out[id]) {
            if (!std::isfinite(v)) {
                std::printf("non-finite output node %d\n", id);
                return false;
            }
        }
    }
    return true;
}

bool check_perturbation(const Run & base, const Run & pert) {
    bool u0_same = true;
    bool u1_diff = false;
    for (int id = 0; id < kNodes; ++id) {
        const Node & n = nodes()[id];
        if (n.ub == 0) {
            if (!same_vec(base.out[id], pert.out[id])) u0_same = false;
        } else if (!same_vec(base.out[id], pert.out[id])) {
            u1_diff = true;
        }
    }
    check(u0_same, "perturb U1: U0 outputs unchanged");
    check(u1_diff, "perturb U1: at least one U1 output changed");
    for (int l = 0; l < kLayers; ++l) {
        check(same_vec(base.state[l][0], pert.state[l][0]), "perturb U1: U0 state unchanged L" + std::to_string(l));
    }
    return u0_same && u1_diff;
}

bool check_failure_rollback() {
    struct S {
        uint64_t epoch = 1;
        bool appended = false;
        bool committed = false;
        bool published = false;
        bool retired = false;
        bool next_allowed = true;
        int publications = 0;
        std::vector<float> data;

        bool append(uint64_t g) {
            if (retired || g != epoch || appended) return false;
            appended = true;
            return true;
        }
        bool commit(uint64_t g, const std::vector<float> & v, bool u0_committed) {
            if (retired || g != epoch || !appended || committed) return false;
            if (!u0_committed) return false;
            committed = true;
            appended = false; // allow the next ubatch's append in this test
            data = v;
            return true;
        }
        bool publish(uint64_t g) {
            if (retired || g != epoch || !committed || published) return false;
            published = true;
            ++publications;
            return true;
        }
        void fail() {
            retired = true;
            next_allowed = false;
            appended = false;
            committed = false;
            published = false;
            ++epoch;
        }
    };
    S s;
    const std::vector<float> u0 = make_data(kD * kN, 11);
    const std::vector<float> u1 = make_data(kD * kN, 13);
    check(s.append(1), "rollback: append U0");
    check(s.commit(1, u0, true), "rollback: commit U0");
    check(s.publish(1), "rollback: publish U0");
    check(s.append(1), "rollback: append U1");
    S before_u0;
    check(before_u0.append(1), "rollback: fresh append");
    check(!before_u0.commit(1, u1, false), "rollback: U1 commit before U0 refused");
    check(!before_u0.publish(1), "rollback: publish before commit refused");
    const std::vector<float> before = s.data;
    s.fail();
    check(s.data == before, "rollback: committed U0 data unchanged");
    check(!s.commit(1, u1, true), "rollback: stale U1 commit refused");
    check(!s.publish(1), "rollback: stale U1 publish refused");
    check(s.publications == 1, "rollback: only U0 publication");
    check(!s.next_allowed, "rollback: next request refused");
    return true;
}

void check_slot_generation(Run & r) {
    std::string err;
    const int idx = r.pool.alloc(9999, 0, err);
    check(idx >= 0, std::string("slot: alloc after run: ") + err);
    if (idx < 0) return;
    const Handle h{idx, r.pool.slots[idx].gen};
    r.pool.release(idx);
    std::string err2;
    check(!r.pool.consume(h, 10000, err2), "slot: stale consume refused");
    const int idx2 = r.pool.alloc(9998, 0, err);
    check(idx2 >= 0, "slot: realloc after release");
    check(idx2 == idx, "slot: same index reused");
    if (idx2 >= 0) r.pool.release(idx2);
}

} // namespace

int main() {
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        std::printf("FATAL: CPU backend unavailable\n");
        return 1;
    }

    Run serial;
    Run look;
    Run serial_pert;

    const bool s_ok = run_schedule(serial, backend, false, 7);
    const bool l_ok = run_schedule(look, backend, true, 7);
    if (!s_ok || !l_ok) {
        free_run(serial); free_run(look); ggml_backend_free(backend);
        return 1;
    }

    check(serial.pairs == 0, "serial: no lookahead pair");
    check(look.pairs == 2, "lookahead: two legal pairs");
    check(all_finite(serial), "serial: finite outputs");
    check(all_finite(look), "lookahead: finite outputs");
    compare_runs(serial, look, "serial vs lookahead");

    // Check the two specific overlap pairs are actually issued before either
    // member completes in the lookahead event sequence.
    int pair_issuances = 0;
    for (const std::string & e : look.events) {
        if (e.rfind("issue pair ", 0) == 0) ++pair_issuances;
    }
    check(pair_issuances == 4, "lookahead: two pairs issued (4 pair-issue events)");

    const bool pert_ok = run_schedule(serial_pert, backend, false, 11);
    if (pert_ok) {
        check_perturbation(serial, serial_pert);
        free_run(serial_pert);
    } else {
        check(false, "perturbation: serial run initialized");
    }

    check_failure_rollback();
    check_slot_generation(serial);

    free_run(serial);
    free_run(look);
    ggml_backend_free(backend);

    std::printf("%s: %d failure(s)\n", failures == 0 ? "OK" : "FAILED", failures);
    return failures == 0 ? 0 : 1;
}
