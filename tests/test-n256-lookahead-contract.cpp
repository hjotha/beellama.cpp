// Model-level CPU contract fixture (v2) for the two-ubatch N256 lookahead.
//
// This test does NOT run a model, GPU, RPC, Xbox or benchmark, and it is NOT a
// P0-D revalidation. It freezes a model-level dependency/shape/state contract:
//   - U0/U1 stay original N=256 ubatches (N512 FFN/attention refused);
//   - Attn is the remote attention for full layers and the local recurrent
//     attention for recurrent layers; Merge is the local branch + merge/
//     Proj_O/residual; FFN consumes Merge, not Attn directly;
//   - full layers: Attn(L,U1) needs the remote-KV state of U0 at layer L;
//     Merge(L,U1) needs Merge(L,U0) (local branch/prefix continuity);
//   - recurrent layers: Attn(L,U1) needs the horizontal recurrent state of U0
//     at layer L (a remote KV commit does not prove recurrence continuity);
//   - GPU QKV(L+1,U0) depends on FFN(L,U0); Attn/FFN already depend on the
//     previous attention, so the old direct QKV<-Attn edge is removed;
//   - a legal lookahead overlaps full remote Attn(L,U1) with GPU QKV(L+1,U0)
//     only when their buffers are separate and the U0 prefix/tail are not
//     reused early; durations 2/1 are logical, not a TPS prediction;
//   - U1 perturbation is executed twice with different U1 inputs and the U0
//     output/state snapshot must not change;
//   - a post-append failure retires the state, advances the epoch and refuses
//     stale/late commits; the publication sentinel stays untouched (model);
//   - runtime P=96.256, historical declared 102.400 (not used), C=131.072,
//     tail=128; the tail starts at absolute P+384 = offset 128 inside U1;
//     near C, a pair starting at C-512 is allowed and a later start refused.
//
// Real tensors/dispatches, real ACK/OUT/LSE, epoch-tagged buffers and the
// production P0-D retirement are the next discriminants, not this fixture.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

constexpr int kLayers = 3;
constexpr int kUb = 2;
constexpr int kN = 256;
constexpr int kKinds = 5;
constexpr int kNodes = kLayers * kUb * kKinds;

constexpr int32_t kPRuntime = 96256;   // effective kv-position-split-p
constexpr int32_t kPDeclared = 102400; // historical V2 declaration (not used)
constexpr int32_t kC = 131072;
constexpr int32_t kTail = 128;

// L0/L1 are full-attention (remote) layers; L2 is recurrent (local).
enum class Kind { Qkv = 0, Attn = 1, Merge = 2, Ffn = 3, State = 4 };
enum class LayerType { FullRemote, Recurrent };
constexpr LayerType kLayerType[kLayers] = {
    LayerType::FullRemote, LayerType::FullRemote, LayerType::Recurrent,
};
enum class Res { Gpu, Remote };
enum class Violation { None, Node, Duplicate, Time, Dependency, Resource, Missing };

struct Node {
    Kind kind;
    int layer;
    int ub;
};

constexpr bool is_full_layer(int layer) {
    return kLayerType[layer] == LayerType::FullRemote;
}

const char * state_semantics(int layer) {
    // Full layers commit remote KV; recurrent layers publish a local
    // horizontal state. The two must not be conflated.
    return is_full_layer(layer) ? "remote-KV" : "recurrent";
}

int nid(Kind kind, int layer, int ub) {
    return (layer * kUb + ub) * kKinds + static_cast<int>(kind);
}

Node node_of(int id) {
    const int kind = id % kKinds;
    const int lu = id / kKinds;
    return Node{static_cast<Kind>(kind), lu / kUb, lu % kUb};
}

const char * kind_name(Kind kind) {
    switch (kind) {
        case Kind::Qkv: return "qkv";
        case Kind::Attn: return "attn";
        case Kind::Merge: return "merge";
        case Kind::Ffn: return "ffn";
        case Kind::State: return "state";
    }
    return "?";
}

Res resource_of(const Node & n) {
    switch (n.kind) {
        case Kind::Qkv:
        case Kind::Merge:
        case Kind::Ffn:
            return Res::Gpu;
        case Kind::Attn:
            return is_full_layer(n.layer) ? Res::Remote : Res::Gpu;
        case Kind::State:
            return is_full_layer(n.layer) ? Res::Remote : Res::Gpu;
    }
    return Res::Gpu;
}

int duration(Kind kind) {
    // Logical example only: remote attention is the long side of the overlap.
    return kind == Kind::Attn ? 2 : 1;
}

// Model-level shape check: no tensors/dispatches are examined here.
bool shape_ok(Kind kind, int n_tokens) {
    (void) kind;
    return n_tokens == kN;
}

std::vector<int> deps_of(int id) {
    const Node n = node_of(id);
    std::vector<int> deps;
    switch (n.kind) {
        case Kind::Qkv:
            if (n.layer > 0) {
                // FFN already depends on Merge -> Attn; no direct Attn edge.
                deps.push_back(nid(Kind::Ffn, n.layer - 1, n.ub));
            }
            break;
        case Kind::Attn:
            deps.push_back(nid(Kind::Qkv, n.layer, n.ub));
            if (n.ub == 1) {
                // Remote KV prefix (full) or horizontal recurrent state (recurrent).
                deps.push_back(nid(Kind::State, n.layer, 0));
            }
            break;
        case Kind::Merge:
            // Local branch + merge/Proj_O/residual. U1 full also needs the
            // U0 local branch/prefix completion.
            deps.push_back(nid(Kind::Qkv, n.layer, n.ub));
            deps.push_back(nid(Kind::Attn, n.layer, n.ub));
            if (n.ub == 1 && is_full_layer(n.layer)) {
                deps.push_back(nid(Kind::Merge, n.layer, 0));
            }
            break;
        case Kind::Ffn:
            deps.push_back(nid(Kind::Merge, n.layer, n.ub));
            break;
        case Kind::State:
            deps.push_back(nid(Kind::Attn, n.layer, n.ub));
            break;
    }
    return deps;
}

struct Step {
    int node;
    int start;
    int end;
};

struct CheckResult {
    bool ok = false;
    Violation violation = Violation::None;
    std::string why;
};

CheckResult check_schedule(std::vector<Step> steps) {
    std::sort(steps.begin(), steps.end(), [](const Step & a, const Step & b) {
        return a.start != b.start ? a.start < b.start : a.end < b.end;
    });
    std::vector<int> end_at(kNodes, -1);
    std::vector<bool> seen(kNodes, false);
    // Pass 1: identity, time and interval invariants.
    for (const Step & s : steps) {
        if (s.node < 0 || s.node >= kNodes) {
            return {false, Violation::Node, "node out of range"};
        }
        if (seen[s.node]) {
            return {false, Violation::Duplicate, "duplicate node " + std::to_string(s.node)};
        }
        const Node n = node_of(s.node);
        if (s.start < 0) {
            return {false, Violation::Time,
                    "negative start for " + std::string(kind_name(n.kind)) +
                    " L" + std::to_string(n.layer) + " U" + std::to_string(n.ub)};
        }
        if (s.end != s.start + duration(n.kind)) {
            return {false, Violation::Time,
                    "end != start + duration for " + std::string(kind_name(n.kind)) +
                    " L" + std::to_string(n.layer) + " U" + std::to_string(n.ub)};
        }
        seen[s.node] = true;
        end_at[s.node] = s.end;
    }
    int count = 0;
    for (const bool b : seen) count += b ? 1 : 0;
    if (count != kNodes) {
        return {false, Violation::Missing,
                "schedule missing nodes (" + std::to_string(count) + "/" + std::to_string(kNodes) + ")"};
    }
    // Pass 2: all dependencies before any resource conflict, so a negative
    // schedule cannot be "proved" by a mere resource overlap.
    for (const Step & s : steps) {
        const Node n = node_of(s.node);
        for (const int d : deps_of(s.node)) {
            if (end_at[d] < 0 || end_at[d] > s.start) {
                return {false, Violation::Dependency,
                        "dependency not complete: " + std::string(kind_name(n.kind)) +
                        " L" + std::to_string(n.layer) +
                        " U" + std::to_string(n.ub) +
                        " needs " + std::string(kind_name(node_of(d).kind)) +
                        " L" + std::to_string(node_of(d).layer) +
                        " U" + std::to_string(node_of(d).ub)};
            }
        }
    }
    // Pass 3: exclusive resource intervals.
    for (size_t i = 0; i < steps.size(); ++i) {
        for (size_t j = i + 1; j < steps.size(); ++j) {
            const Node a = node_of(steps[i].node);
            const Node b = node_of(steps[j].node);
            if (resource_of(a) != resource_of(b)) continue;
            if (steps[i].start < steps[j].end && steps[j].start < steps[i].end) {
                return {false, Violation::Resource,
                        "resource overlap: " + std::string(kind_name(a.kind)) +
                        " and " + std::string(kind_name(b.kind))};
            }
        }
    }
    return {true, Violation::None, ""};
}

std::vector<Step> serial_schedule() {
    std::vector<Step> steps;
    int t = 0;
    for (int u = 0; u < kUb; ++u) {
        for (int l = 0; l < kLayers; ++l) {
            for (const Kind k : {Kind::Qkv, Kind::Attn, Kind::Merge, Kind::Ffn, Kind::State}) {
                const int id = nid(k, l, u);
                steps.push_back(Step{id, t, t + duration(k)});
                t += duration(k);
            }
        }
    }
    return steps;
}

std::vector<Step> lookahead_schedule() {
    std::vector<Step> steps;
    std::vector<int> dep_end(kNodes, -1);
    std::vector<bool> done(kNodes, false);
    int t = 0;
    int remaining = kNodes;
    while (remaining > 0 && t < kNodes * 8) {
        for (const Res r : {Res::Gpu, Res::Remote}) {
            bool free = true;
            for (const Step & s : steps) {
                if (resource_of(node_of(s.node)) == r && s.end > t) {
                    free = false;
                    break;
                }
            }
            if (!free) continue;
            int best = -1;
            int best_score = 1 << 30;
            for (int i = 0; i < kNodes; ++i) {
                if (done[i] || resource_of(node_of(i)) != r) continue;
                bool ready = true;
                for (const int d : deps_of(i)) {
                    if (dep_end[d] < 0 || dep_end[d] > t) {
                        ready = false;
                        break;
                    }
                }
                if (!ready) continue;
                const Node n = node_of(i);
                const int kind_score = (n.kind == Kind::Qkv || n.kind == Kind::Attn) ? 0 : 1;
                const int score = kind_score * 1000 + n.layer * 10 + n.ub;
                if (score < best_score) {
                    best_score = score;
                    best = i;
                }
            }
            if (best >= 0) {
                const int len = duration(node_of(best).kind);
                steps.push_back(Step{best, t, t + len});
                dep_end[best] = t + len;
                done[best] = true;
                --remaining;
            }
        }
        ++t;
    }
    return steps;
}

const Step * find_step(const std::vector<Step> & steps, int node) {
    for (const Step & s : steps) {
        if (s.node == node) return &s;
    }
    return nullptr;
}

bool overlap_pair(const std::vector<Step> & steps, int layer) {
    if (!is_full_layer(layer) || layer + 1 >= kLayers) return false;
    const Step * a = find_step(steps, nid(Kind::Attn, layer, 1));
    const Step * b = find_step(steps, nid(Kind::Qkv, layer + 1, 0));
    if (!a || !b) return false;
    return std::max(a->start, b->start) < std::min(a->end, b->end);
}

bool starts_before_dep_completes(const std::vector<Step> & steps, int node, int dep) {
    const Step * a = find_step(steps, node);
    const Step * d = find_step(steps, dep);
    return a && d && a->start < d->end;
}

bool has_dep(int node, int dep) {
    const std::vector<int> deps = deps_of(node);
    return std::find(deps.begin(), deps.end(), dep) != deps.end();
}

// Buffers: the full U0 State is the committed prefix; the full U1 State is the
// tail region; every other output has its own identifier. The overlap pair
// may only overlap if these buffers are separate.
struct BufferId {
    uint64_t id;
    const char * role;
};

BufferId buffer_of(const Node & n) {
    if (n.kind == Kind::State && is_full_layer(n.layer)) {
        if (n.ub == 0) return BufferId{100 + static_cast<uint64_t>(n.layer), "u0-prefix"};
        return BufferId{200 + static_cast<uint64_t>(n.layer), "tail"};
    }
    return BufferId{1000 + static_cast<uint64_t>(nid(n.kind, n.layer, n.ub)), "out"};
}

bool buffers_ok(std::string & why) {
    // No node writes over an input it still depends on.
    for (int i = 0; i < kNodes; ++i) {
        const Node n = node_of(i);
        const BufferId out = buffer_of(n);
        for (const int d : deps_of(i)) {
            const BufferId in = buffer_of(node_of(d));
            if (out.id == in.id) {
                why = "alias: " + std::string(kind_name(n.kind)) +
                      " L" + std::to_string(n.layer) + " U" + std::to_string(n.ub) +
                      " writes its input " + std::string(kind_name(node_of(d).kind));
                return false;
            }
        }
    }
    // The overlapped pair must use separate outputs and must not reuse the
    // committed U0 prefix or the tail.
    for (int l = 0; l + 1 < kLayers; ++l) {
        if (!is_full_layer(l)) continue;
        const BufferId attn_out = buffer_of(Node{Kind::Attn, l, 1});
        const BufferId qkv_out = buffer_of(Node{Kind::Qkv, l + 1, 0});
        const BufferId prefix = buffer_of(Node{Kind::State, l, 0});
        const BufferId tail = buffer_of(Node{Kind::State, l, 1});
        const BufferId attn_in = buffer_of(Node{Kind::Qkv, l, 1});
        const BufferId qkv_in = buffer_of(Node{Kind::Ffn, l, 0});
        if (attn_out.id == qkv_out.id) {
            why = "overlap pair shares one output buffer at L" + std::to_string(l);
            return false;
        }
        for (const BufferId x : {attn_out, qkv_out, attn_in, qkv_in}) {
            if (x.id == prefix.id || x.id == tail.id) {
                why = "overlap pair reuses prefix/tail buffer at L" + std::to_string(l);
                return false;
            }
        }
    }
    return true;
}

uint64_t mix(uint64_t a, uint64_t b) {
    return a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2));
}

struct SimResult {
    std::array<std::array<uint64_t, kUb>, kLayers> qkv{};
    std::array<std::array<uint64_t, kUb>, kLayers> attn{};
    std::array<std::array<uint64_t, kUb>, kLayers> merge{};
    std::array<std::array<uint64_t, kUb>, kLayers> ffn{};
    std::array<std::array<uint64_t, kUb>, kLayers> state{};
    std::array<int32_t, kUb> rows{};
    std::array<int, kUb> publishes{};
};

// Executes the model twice with a different U1 input; the U0 branch is fixed.
SimResult simulate(uint64_t u1_tokens) {
    SimResult r;
    const uint64_t tokens[kUb] = {0x55aa55aa55aa55aaULL, u1_tokens};
    for (int u = 0; u < kUb; ++u) {
        for (int l = 0; l < kLayers; ++l) {
            const uint64_t in = l == 0 ? tokens[u] : r.ffn[l - 1][u];
            r.qkv[l][u] = mix(in, 0x1000 + static_cast<uint64_t>(l * kUb + u));
            const uint64_t prefix = (u == 1) ? r.state[l][0] : 0;
            const uint64_t local_prefix = (u == 1 && is_full_layer(l)) ? r.merge[l][0] : 0;
            r.attn[l][u] = mix(r.qkv[l][u], prefix ^ 0x2000);
            r.merge[l][u] = mix(r.attn[l][u], r.qkv[l][u] ^ local_prefix ^ 0x3000);
            r.ffn[l][u] = mix(r.merge[l][u], 0x4000 + static_cast<uint64_t>(l));
            r.state[l][u] = mix(r.attn[l][u], 0x5000 + static_cast<uint64_t>(l));
        }
        r.rows[u] = kPRuntime + (u + 1) * kN;
        r.publishes[u] = 1;
    }
    return r;
}

struct RemoteState {
    std::array<bool, kUb> appended{};
    std::array<bool, kUb> committed{};
    std::array<int32_t, kUb> rows{};
    uint64_t epoch = 1;
    bool retired = false;
    int published = 0;

    bool append(int ub, uint64_t e) {
        if (retired || e != epoch) return false;
        appended[ub] = true;
        return true;
    }

    bool commit(int ub, int32_t row_end, uint64_t e) {
        if (retired || e != epoch || !appended[ub]) return false;
        if (ub == 1 && !committed[0]) return false;
        committed[ub] = true;
        rows[ub] = row_end;
        ++published;
        return true;
    }

    void fail_after_append() {
        retired = true;
        appended = {};
        committed = {};
        rows = {};
        ++epoch;
    }

    bool next_request_allowed() const { return !retired; }
};

int failures = 0;

void check(bool ok, const char * name) {
    std::printf("%s: %s\n", name, ok ? "OK" : "FAIL");
    if (!ok) ++failures;
}

} // namespace

int main() {
    // --- layer semantics: full remote vs recurrent local -------------------
    check(is_full_layer(0) && is_full_layer(1) && !is_full_layer(2),
          "layer mix full/full/recurrent");
    check(resource_of(Node{Kind::Attn, 0, 1}) == Res::Remote, "full Attn is remote");
    check(resource_of(Node{Kind::State, 0, 0}) == Res::Remote, "full State is remote KV");
    check(resource_of(Node{Kind::Attn, 2, 1}) == Res::Gpu, "recurrent Attn is local");
    check(resource_of(Node{Kind::State, 2, 0}) == Res::Gpu, "recurrent State is local");
    check(std::string(state_semantics(0)) == "remote-KV", "full layer state is remote KV");
    check(std::string(state_semantics(2)) == "recurrent", "recurrent layer state is horizontal");

    // --- explicit model-level DAG edges (review point 1) ------------------
    check(has_dep(nid(Kind::Ffn, 0, 0), nid(Kind::Merge, 0, 0)),
          "DAG: FFN consumes Merge, not raw remote Attn");
    check(has_dep(nid(Kind::Merge, 0, 0), nid(Kind::Attn, 0, 0)),
          "DAG: Merge consumes remote Attn");
    check(has_dep(nid(Kind::Merge, 0, 0), nid(Kind::Qkv, 0, 0)),
          "DAG: Merge consumes local QKV (local branch/Proj_O/residual)");
    check(has_dep(nid(Kind::Attn, 0, 1), nid(Kind::State, 0, 0)),
          "DAG: full U1 Attn needs the U0 remote KV");
    check(has_dep(nid(Kind::Merge, 0, 1), nid(Kind::Merge, 0, 0)),
          "DAG: full U1 Merge needs the U0 local branch/prefix");
    check(has_dep(nid(Kind::Qkv, 1, 0), nid(Kind::Ffn, 0, 0)),
          "DAG: GPU next-layer U0 needs FFN U0");
    check(!has_dep(nid(Kind::Qkv, 1, 0), nid(Kind::Attn, 0, 0)),
          "DAG: redundant direct QKV<-Attn edge is removed");
    check(has_dep(nid(Kind::Attn, 2, 1), nid(Kind::State, 2, 0)),
          "DAG: recurrent U1 Attn needs the U0 horizontal state");

    // --- model-level shapes: original N256 only ----------------------------
    check(shape_ok(Kind::Qkv, 256), "shape QKV N256");
    check(shape_ok(Kind::Attn, 256), "shape ATTN N256");
    check(shape_ok(Kind::Merge, 256), "shape MERGE N256");
    check(shape_ok(Kind::Ffn, 256), "shape FFN N256");
    check(!shape_ok(Kind::Ffn, 512), "shape FFN N512 rejected");
    check(!shape_ok(Kind::Attn, 512), "shape ATTN N512 rejected");

    // --- dependency graph / serial vs lookahead ----------------------------
    const std::vector<Step> serial = serial_schedule();
    const std::vector<Step> lookahead = lookahead_schedule();
    const CheckResult serial_r = check_schedule(serial);
    const CheckResult lookahead_r = check_schedule(lookahead);
    check(serial_r.ok, "serial schedule dependencies/resources/time");
    if (!serial_r.ok) std::printf("  serial: %s\n", serial_r.why.c_str());
    check(lookahead_r.ok, "lookahead schedule dependencies/resources/time");
    if (!lookahead_r.ok) std::printf("  lookahead: %s\n", lookahead_r.why.c_str());
    check(overlap_pair(lookahead, 0), "lookahead overlaps Attn(L0,U1) with QKV(L1,U0)");
    check(overlap_pair(lookahead, 1), "lookahead overlaps Attn(L1,U1) with QKV(L2,U0)");
    check(!overlap_pair(serial, 0), "serial does not overlap the L0 pair");
    check(!overlap_pair(serial, 1), "serial does not overlap the L1 pair");

    std::string bwhy;
    check(buffers_ok(bwhy), "overlap pair uses separate prefix/tail buffers");
    if (!bwhy.empty()) std::printf("  buffers: %s\n", bwhy.c_str());

    // --- negative tests must prove the intended dependency edge -----------
    {
        std::vector<Step> bad = lookahead;
        Step * attn1 = const_cast<Step *>(find_step(bad, nid(Kind::Attn, 0, 1)));
        const Step * state0 = find_step(bad, nid(Kind::State, 0, 0));
        check(attn1 && state0, "negative full-prefix nodes present");
        if (attn1 && state0) {
            attn1->start = state0->start;
            attn1->end = attn1->start + duration(Kind::Attn);
            check(starts_before_dep_completes(bad, nid(Kind::Attn, 0, 1), nid(Kind::State, 0, 0)),
                  "negative setup: U1 Attn starts before U0 remote KV completes");
            const CheckResult r = check_schedule(bad);
            check(!r.ok && r.violation == Violation::Dependency,
                  "negative: U1 full Attn before U0 remote KV (dependency, not overlap)");
            check(r.why.find("state L0 U0") != std::string::npos,
                  "negative names the U0 remote-KV edge");
        }
    }
    {
        std::vector<Step> bad = lookahead;
        Step * qkv10 = const_cast<Step *>(find_step(bad, nid(Kind::Qkv, 1, 0)));
        const Step * ffn00 = find_step(bad, nid(Kind::Ffn, 0, 0));
        check(qkv10 && ffn00, "negative QKV/FFN nodes present");
        if (qkv10 && ffn00) {
            qkv10->start = ffn00->start;
            qkv10->end = qkv10->start + duration(Kind::Qkv);
            check(starts_before_dep_completes(bad, nid(Kind::Qkv, 1, 0), nid(Kind::Ffn, 0, 0)),
                  "negative setup: GPU next-layer U0 starts before FFN completes");
            const CheckResult r = check_schedule(bad);
            check(!r.ok && r.violation == Violation::Dependency,
                  "negative: GPU next-layer U0 before FFN (dependency)");
            check(r.why.find("ffn L0 U0") != std::string::npos,
                  "negative names the FFN edge");
        }
    }
    {
        std::vector<Step> bad = lookahead;
        Step * merge1 = const_cast<Step *>(find_step(bad, nid(Kind::Merge, 0, 1)));
        const Step * merge0 = find_step(bad, nid(Kind::Merge, 0, 0));
        check(merge1 && merge0, "negative local-merge nodes present");
        if (merge1 && merge0) {
            merge1->start = merge0->start;
            merge1->end = merge1->start + duration(Kind::Merge);
            check(starts_before_dep_completes(bad, nid(Kind::Merge, 0, 1), nid(Kind::Merge, 0, 0)),
                  "negative setup: U1 local merge starts before U0 local branch completes");
            const CheckResult r = check_schedule(bad);
            check(!r.ok && r.violation == Violation::Dependency,
                  "negative: local-prefix violation rejected as dependency (not resource)");
        }
    }
    {
        std::vector<Step> bad = lookahead;
        Step * attn21 = const_cast<Step *>(find_step(bad, nid(Kind::Attn, 2, 1)));
        const Step * state20 = find_step(bad, nid(Kind::State, 2, 0));
        check(attn21 && state20, "negative recurrent nodes present");
        if (attn21 && state20) {
            attn21->start = state20->start;
            attn21->end = attn21->start + duration(Kind::Attn);
            check(starts_before_dep_completes(bad, nid(Kind::Attn, 2, 1), nid(Kind::State, 2, 0)),
                  "negative setup: recurrent U1 Attn starts before U0 state completes");
            const CheckResult r = check_schedule(bad);
            check(!r.ok && r.violation == Violation::Dependency,
                  "negative: recurrent U1 before U0 horizontal state (dependency)");
            check(r.why.find("state L2 U0") != std::string::npos,
                  "negative names the recurrent horizontal edge");
        }
    }
    {
        std::vector<Step> bad = lookahead;
        Step * attn00 = const_cast<Step *>(find_step(bad, nid(Kind::Attn, 0, 0)));
        check(attn00 != nullptr, "negative time node present");
        if (attn00) {
            attn00->end += 1;
            const CheckResult r = check_schedule(bad);
            check(!r.ok && r.violation == Violation::Time,
                  "validator rejects end != start + duration");
        }
    }
    {
        std::vector<Step> bad = lookahead;
        Step * attn00 = const_cast<Step *>(find_step(bad, nid(Kind::Attn, 0, 0)));
        check(attn00 != nullptr, "negative start node present");
        if (attn00) {
            attn00->start = -1;
            attn00->end = attn00->start + duration(Kind::Attn);
            const CheckResult r = check_schedule(bad);
            check(!r.ok && r.violation == Violation::Time,
                  "validator rejects negative start");
        }
    }
    {
        std::vector<Step> bad = serial;
        Step * q0 = const_cast<Step *>(find_step(bad, nid(Kind::Qkv, 0, 0)));
        Step * q1 = const_cast<Step *>(find_step(bad, nid(Kind::Qkv, 0, 1)));
        check(q0 && q1, "negative resource nodes present");
        if (q0 && q1) {
            q1->start = q0->start;
            q1->end = q0->end;
            const CheckResult r = check_schedule(bad);
            check(!r.ok && r.violation == Violation::Resource,
                  "validator rejects GPU interval overlap");
        }
    }

    // --- real two-run U1 perturbation; U0 snapshot must stay invariant ----
    {
        const SimResult a = simulate(0xaaaaaaaaaaaaaaaaULL);
        const SimResult b = simulate(0xbbbbbbbbbbbbbbbbULL);
        bool u0_same = a.rows[0] == b.rows[0] && a.publishes[0] == b.publishes[0];
        for (int l = 0; l < kLayers; ++l) {
            u0_same = u0_same &&
                      a.qkv[l][0] == b.qkv[l][0] &&
                      a.attn[l][0] == b.attn[l][0] &&
                      a.merge[l][0] == b.merge[l][0] &&
                      a.ffn[l][0] == b.ffn[l][0] &&
                      a.state[l][0] == b.state[l][0];
        }
        check(u0_same, "simulation: U1 perturbation leaves the whole U0 snapshot unchanged");
        const int last = kLayers - 1;
        check(a.ffn[last][1] != b.ffn[last][1] ||
              a.attn[last][1] != b.attn[last][1] ||
              a.merge[last][1] != b.merge[last][1] ||
              a.state[last][1] != b.state[last][1],
              "simulation: U1 perturbation changes the U1 output/state");
        check(a.rows[0] == kPRuntime + kN && a.rows[1] == kPRuntime + 2 * kN,
              "simulation rows: U0 ends P+256, U1 ends P+512");
        check(a.publishes[0] == 1 && b.publishes[0] == 1,
              "simulation: U0 published once per run, not re-uploaded by U1");
    }

    // --- model-level epoch/sentinel retirement ----------------------------
    {
        RemoteState st;
        check(!st.commit(1, kPRuntime + 2 * kN, 1), "state: U1 commit before U0 refused");
        check(st.append(0, 1), "state: U0 append epoch 1");
        check(st.commit(0, kPRuntime + kN, 1), "state: U0 commit epoch 1");
        check(st.append(1, 1), "state: U1 append epoch 1");
        check(st.commit(1, kPRuntime + 2 * kN, 1), "state: U1 commit after U0");
        check(st.published == 2 && st.rows[0] == kPRuntime + kN,
              "state: U0/U1 publication and rows");
    }
    {
        RemoteState st;
        check(st.append(0, 1) && st.commit(0, kPRuntime + kN, 1) && st.append(1, 1),
              "failure state: U0 committed, U1 appended");
        st.fail_after_append();
        check(st.epoch == 2 && st.retired, "failure advances epoch and retires");
        check(!st.commit(1, kPRuntime + 2 * kN, 1),
              "failure refuses stale-epoch U1 commit after append");
        check(!st.commit(1, kPRuntime + 2 * kN, 2),
              "failure refuses post-retirement U1 commit");
        check(st.published == 1, "failure: no U1 publication sentinel");
        check(!st.next_request_allowed(), "failure rejects next request");
        check(!st.committed[0] && !st.committed[1], "failure clears published prefix");
    }

    // --- P provenance, tail and near-C ------------------------------------
    check(kPRuntime == 96256, "runtime P=96256");
    check(kPDeclared == 102400 && kPDeclared != kPRuntime,
          "declared P=102400 preserved as historical, not used");
    check(kPRuntime + kN == kPRuntime + 256, "U1 starts 256 tokens after U0");
    check(kPRuntime + 2 * kN <= kC, "two N256 ubatches fit in C=131072");
    check(kTail == 128, "tail 128");
    const int32_t tail_abs = kPRuntime + 2 * kN - kTail;
    const int32_t tail_offset_u1 = tail_abs - (kPRuntime + kN);
    check(tail_abs == kPRuntime + 384, "tail absolute start is P+384");
    check(tail_offset_u1 == 128, "tail offset inside U1 is 128 (not 384)");
    check(tail_abs == kPRuntime + kN + kTail, "tail = U1 start + 128");
    const auto pair_fits = [](int32_t start) {
        return start >= 0 && start + 2 * kN <= kC;
    };
    check(pair_fits(kC - 2 * kN), "near-C: pair starting at C-512 allowed");
    check(!pair_fits(kC - kN), "near-C: pair starting at C-256 refused");
    check(!pair_fits(kC - 2 * kN + 1), "near-C: pair starting at C-511 refused");

    std::printf("%s: %d failure(s)\n", failures == 0 ? "OK" : "FAILED", failures);
    return failures == 0 ? 0 : 1;
}
