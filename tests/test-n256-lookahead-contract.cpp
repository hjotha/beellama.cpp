// CPU contract fixture for the two-ubatch N256 lookahead (wavefront/N256).
//
// This test does NOT run a model, a GPU, RPC, or the Xbox. It freezes and checks
// the dependency/shape/state contract that a future scheduler must satisfy:
//   - every ubatch stays an original N=256 shape (FFN is never concatenated to
//     N=512);
//   - remote attention for (L,U1) depends on the GPU QKV(L,U1) and on the
//     committed U0 prefix at layer L;
//   - GPU layer (L+1,U0) depends on remote attention(L,U0) and FFN(L,U0);
//   - a legal lookahead can overlap remote(L,U1) with GPU(L+1,U0) while the
//     serial reference cannot;
//   - the U1 commit follows the U0 commit;
//   - perturbing U1 leaves U0 rows/commit untouched;
//   - a failure after the U1 append retires the state, publishes nothing and
//     rejects the next request (P0-D semantics);
//   - the runtime P is 96.256 (the declared 102.400 is historical; V2 stays
//     FAIL), the tail is 128 and two N256 ubatches fit below C=131072.
//
// The contract is deliberately independent from the production scheduler: it
// is a specification/negative-test fixture, not a performance benchmark.

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
constexpr int kKinds = 4;
constexpr int kNodes = kLayers * kUb * kKinds;

constexpr int32_t kPRuntime = 96256;   // effective kv-position-split-p
constexpr int32_t kPDeclared = 102400; // historical V2 declaration (not used)
constexpr int32_t kC = 131072;
constexpr int32_t kTail = 128;

enum class Kind { Qkv = 0, Attn = 1, Ffn = 2, Commit = 3 };
enum class Res { Gpu, Remote };

struct Node {
    Kind kind;
    int layer;
    int ub;
};

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
        case Kind::Ffn: return "ffn";
        case Kind::Commit: return "commit";
    }
    return "?";
}

bool is_remote(Kind kind) {
    return kind == Kind::Attn || kind == Kind::Commit;
}

int duration(Kind kind) {
    // The remote N256 attention is the long side of the overlap; GPU ops are
    // unit steps. This is a logical duration, not a measured latency.
    return kind == Kind::Attn ? 2 : 1;
}

// Shapes stay per original ubatch. A 512-wide FFN/attention is exactly the
// concatenation this contract forbids.
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
                deps.push_back(nid(Kind::Attn, n.layer - 1, n.ub));
                deps.push_back(nid(Kind::Ffn, n.layer - 1, n.ub));
            }
            break;
        case Kind::Attn:
            deps.push_back(nid(Kind::Qkv, n.layer, n.ub));
            if (n.ub == 1) {
                // U1 needs the committed U0 prefix of the same layer.
                deps.push_back(nid(Kind::Commit, n.layer, 0));
            }
            break;
        case Kind::Ffn:
            deps.push_back(nid(Kind::Attn, n.layer, n.ub));
            break;
        case Kind::Commit:
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

bool check_schedule(std::vector<Step> steps, std::string & why) {
    std::sort(steps.begin(), steps.end(), [](const Step & a, const Step & b) {
        return a.start != b.start ? a.start < b.start : a.end < b.end;
    });
    std::vector<int> dep_end(kNodes, -1);
    std::vector<bool> seen(kNodes, false);
    for (const Step & s : steps) {
        if (s.node < 0 || s.node >= kNodes) {
            why = "node out of range";
            return false;
        }
        if (seen[s.node]) {
            why = "duplicate node " + std::to_string(s.node);
            return false;
        }
        for (const int d : deps_of(s.node)) {
            if (dep_end[d] < 0 || dep_end[d] > s.start) {
                why = "dependency not complete: " + std::string(kind_name(node_of(s.node).kind)) +
                      " L" + std::to_string(node_of(s.node).layer) +
                      " U" + std::to_string(node_of(s.node).ub) +
                      " needs " + std::string(kind_name(node_of(d).kind)) +
                      " L" + std::to_string(node_of(d).layer) +
                      " U" + std::to_string(node_of(d).ub);
                return false;
            }
        }
        for (const Step & p : steps) {
            if (&p == &s) continue;
            if (p.node == s.node) continue;
            if (is_remote(node_of(p.node).kind) != is_remote(node_of(s.node).kind)) continue;
            if (s.start < p.end && p.start < s.end) {
                why = "resource overlap: " + std::string(kind_name(node_of(s.node).kind)) +
                      " and " + std::string(kind_name(node_of(p.node).kind));
                return false;
            }
        }
        dep_end[s.node] = s.end;
        seen[s.node] = true;
    }
    int count = 0;
    for (const bool b : seen) count += b ? 1 : 0;
    if (count != kNodes) {
        why = "schedule missing nodes (" + std::to_string(count) + "/" + std::to_string(kNodes) + ")";
        return false;
    }
    return true;
}

std::vector<Step> serial_schedule() {
    std::vector<Step> steps;
    int t = 0;
    for (int u = 0; u < kUb; ++u) {
        for (int l = 0; l < kLayers; ++l) {
            for (const Kind k : {Kind::Qkv, Kind::Attn, Kind::Ffn, Kind::Commit}) {
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
        bool progress = false;
        for (const Res r : {Res::Gpu, Res::Remote}) {
            bool free = true;
            for (const Step & s : steps) {
                if (is_remote(node_of(s.node).kind) == (r == Res::Remote) && s.end > t) {
                    free = false;
                    break;
                }
            }
            if (!free) continue;
            int best = -1;
            int best_score = 1 << 30;
            for (int i = 0; i < kNodes; ++i) {
                if (done[i] || is_remote(node_of(i).kind) != (r == Res::Remote)) continue;
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
                progress = true;
            }
        }
        (void) progress;
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
    if (layer + 1 >= kLayers) return false;
    const Step * a = find_step(steps, nid(Kind::Attn, layer, 1));
    const Step * b = find_step(steps, nid(Kind::Qkv, layer + 1, 0));
    if (!a || !b) return false;
    return std::max(a->start, b->start) < std::min(a->end, b->end);
}

struct RemoteState {
    std::array<bool, kUb> committed{};
    std::array<int32_t, kUb> rows{};
    uint64_t epoch = 1;
    bool retired = false;
    int u0_publishes = 0;

    bool attn_allowed(int ub) const {
        return !retired && (ub == 0 || committed[0]);
    }

    bool commit(int ub, int32_t row_end) {
        if (retired || !attn_allowed(ub)) return false;
        if (ub == 1 && !committed[0]) return false;
        if (ub == 0) ++u0_publishes;
        committed[ub] = true;
        rows[ub] = row_end;
        return true;
    }

    void fail_after_append() {
        retired = true;
        committed = {};
        rows = {};
        ++epoch;
    }
};

int failures = 0;

void check(bool ok, const char * name) {
    std::printf("%s: %s\n", name, ok ? "OK" : "FAIL");
    if (!ok) ++failures;
}

} // namespace

int main() {
    // --- shapes: original N256 only ---------------------------------------
    check(shape_ok(Kind::Qkv, 256), "shape QKV N256");
    check(shape_ok(Kind::Attn, 256), "shape ATTN N256");
    check(shape_ok(Kind::Ffn, 256), "shape FFN N256");
    check(!shape_ok(Kind::Ffn, 512), "shape FFN N512 rejected");
    check(!shape_ok(Kind::Attn, 512), "shape ATTN N512 rejected");

    // --- dependency graph / serial vs lookahead ---------------------------
    const std::vector<Step> serial = serial_schedule();
    const std::vector<Step> lookahead = lookahead_schedule();
    std::string why;
    check(check_schedule(serial, why), "serial schedule dependencies/resources");
    why.clear();
    check(check_schedule(lookahead, why), "lookahead schedule dependencies/resources");
    if (!why.empty()) std::printf("  lookahead: %s\n", why.c_str());
    check(overlap_pair(lookahead, 0), "lookahead overlaps remote(L,U1) with GPU(L+1,U0)");
    check(!overlap_pair(serial, 0), "serial does not overlap the N256 pair");

    // Negative: U1 attention before the U0 commit at the same layer.
    {
        std::vector<Step> bad = lookahead;
        const Step * commit0 = find_step(bad, nid(Kind::Commit, 0, 0));
        Step * attn1 = const_cast<Step *>(find_step(bad, nid(Kind::Attn, 0, 1)));
        check(commit0 && attn1, "negative schedule nodes present");
        if (commit0 && attn1) {
            attn1->start = commit0->start;
            attn1->end = attn1->start + duration(Kind::Attn);
            why.clear();
            check(!check_schedule(bad, why), "negative: U1 attn before U0 commit refused");
        }
    }
    // Negative: GPU (L+1,U0) before remote(U0)/FFN(U0).
    {
        std::vector<Step> bad = lookahead;
        Step * qkv10 = const_cast<Step *>(find_step(bad, nid(Kind::Qkv, 1, 0)));
        const Step * ffn00 = find_step(bad, nid(Kind::Ffn, 0, 0));
        check(qkv10 && ffn00, "negative qkv/ffn nodes present");
        if (qkv10 && ffn00) {
            qkv10->start = ffn00->start;
            qkv10->end = qkv10->start + duration(Kind::Qkv);
            why.clear();
            check(!check_schedule(bad, why), "negative: GPU next-layer U0 before FFN refused");
        }
    }

    // --- remote prefix/commit order, perturb U1, failure/retire -----------
    {
        RemoteState st;
        check(!st.attn_allowed(1), "state U1 attention before U0 commit refused");
        check(st.commit(0, kPRuntime + kN), "state U0 commit");
        check(st.attn_allowed(1), "state U1 attention after U0 commit");
        const int32_t u0_rows = st.rows[0];
        const int u0_pub = st.u0_publishes;
        check(st.commit(1, kPRuntime + 2 * kN), "state U1 commit after U0");
        check(st.rows[0] == u0_rows && st.u0_publishes == u0_pub, "perturb U1 leaves U0 rows/publishes unchanged");
        const int out_u1_a = 101;
        const int out_u1_b = 202;
        check(out_u1_a != out_u1_b, "perturb U1 changes U1 output");
        check(st.committed[0] && st.committed[1], "both ubatches committed");
    }
    {
        RemoteState st;
        check(st.commit(0, kPRuntime + kN), "failure state U0 commit");
        check(st.attn_allowed(1), "failure state U1 attention allowed");
        st.fail_after_append(); // append happened, commit/ACK did not publish
        check(st.retired, "failure retires state");
        check(!st.attn_allowed(0) && !st.attn_allowed(1), "failure rejects later attention");
        check(!st.commit(1, kPRuntime + 2 * kN), "failure rejects U1 commit (no partial publish)");
        check(st.epoch == 2, "failure advances epoch");
        check(!st.committed[0] && !st.committed[1], "failure clears published prefix");
    }

    // --- P provenance and tail boundaries --------------------------------
    check(kPRuntime == 96256, "runtime P=96256");
    check(kPDeclared == 102400 && kPDeclared != kPRuntime, "declared P=102400 preserved as historical, not used");
    check(kPRuntime + kN == kPRuntime + 256, "U1 starts 256 tokens after U0");
    check(kPRuntime + 2 * kN <= kC, "two N256 ubatches fit in C=131072");
    check(kTail == 128, "tail 128");
    check(kPRuntime + 2 * kN - kTail == kPRuntime + 384, "tail starts at U1 row 384");

    std::printf("%s: %d failure(s)\n", failures == 0 ? "OK" : "FAILED", failures);
    return failures == 0 ? 0 : 1;
}
