// CPU prototype: two-ubatch N256 lookahead executor/scheduler.
//
// Model-level only. This test runs NO model, GPU, Xbox, RPC or benchmark and
// does not inspect real tensors/dispatches. It executes the same simulated
// operators as tests/test-n256-lookahead-contract.cpp through a small
// discrete-event executor, with:
//   - serial vs lookahead schedules (completion events, GPU/remote resources);
//   - per-layer KV/recurrent state (append -> commit -> publication), epochs;
//   - buffers with slot generation and lifetime (last consumer releases);
//   - adversarial alias, duplicate commit, late/stale completion and failures
//     between append/commit/publication;
//   - a U1-before-U0 negative that appends U1 first, so the rejection is
//     specifically the ordering (Order), not a missing append (NoAppend).
//
// V2 FAIL, declared P=102400 vs runtime P=96256 and trim NO-GO are untouched.
// Real graph/tensor inspection (layout, strides, dispatch) and GPU work remain
// the next discriminants, not this prototype.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <queue>
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

enum class Kind { Qkv = 0, Attn = 1, Merge = 2, Ffn = 3, State = 4 };
enum class LayerType { FullRemote, Recurrent };
constexpr LayerType kLayerType[kLayers] = {
    LayerType::FullRemote, LayerType::FullRemote, LayerType::Recurrent,
};
enum class Res { Gpu, Remote };

struct Node {
    Kind kind;
    int layer;
    int ub;
};

constexpr bool is_full_layer(int layer) {
    return kLayerType[layer] == LayerType::FullRemote;
}

const char * state_semantics(int layer) {
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
    // Logical example only; no TPS or latency inference.
    return kind == Kind::Attn ? 2 : 1;
}

bool shape_ok(Kind kind, int n_tokens) {
    (void) kind;
    return n_tokens == kN;
}

std::vector<int> deps_of(int id) {
    const Node n = node_of(id);
    std::vector<int> deps;
    switch (n.kind) {
        case Kind::Qkv:
            if (n.layer > 0) deps.push_back(nid(Kind::Ffn, n.layer - 1, n.ub));
            break;
        case Kind::Attn:
            deps.push_back(nid(Kind::Qkv, n.layer, n.ub));
            if (n.ub == 1) deps.push_back(nid(Kind::State, n.layer, 0));
            break;
        case Kind::Merge:
            deps.push_back(nid(Kind::Qkv, n.layer, n.ub));
            deps.push_back(nid(Kind::Attn, n.layer, n.ub));
            if (n.ub == 1 && is_full_layer(n.layer)) deps.push_back(nid(Kind::Merge, n.layer, 0));
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

int dependents_of(int node) {
    int count = 0;
    for (int i = 0; i < kNodes; ++i) {
        for (const int d : deps_of(i)) {
            if (d == node) {
                ++count;
                break;
            }
        }
    }
    return count;
}

uint64_t mix(uint64_t a, uint64_t b) {
    return a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2));
}

enum class StateError {
    None, Retired, StaleEpoch, NoAppend, Order, Duplicate, NotCommitted, AlreadyPublished,
};

const char * state_error_name(StateError e) {
    switch (e) {
        case StateError::None: return "none";
        case StateError::Retired: return "retired";
        case StateError::StaleEpoch: return "stale-epoch";
        case StateError::NoAppend: return "no-append";
        case StateError::Order: return "order";
        case StateError::Duplicate: return "duplicate";
        case StateError::NotCommitted: return "not-committed";
        case StateError::AlreadyPublished: return "already-published";
    }
    return "?";
}

struct LayerKv {
    uint64_t epoch = 1;
    bool retired = false;
    std::array<bool, kUb> appended{};
    std::array<bool, kUb> committed{};
    std::array<bool, kUb> published{};
    std::array<int32_t, kUb> rows{};
    std::array<uint64_t, kUb> values{};
    int publications = 0;

    StateError append(int ub, uint64_t gen) {
        if (retired) return StateError::Retired;
        if (gen != epoch) return StateError::StaleEpoch;
        if (appended[ub]) return StateError::Duplicate;
        appended[ub] = true;
        return StateError::None;
    }

    StateError commit(int ub, int32_t row_end, uint64_t value, uint64_t gen) {
        if (retired) return StateError::Retired;
        if (gen != epoch) return StateError::StaleEpoch;
        if (!appended[ub]) return StateError::NoAppend;
        if (ub == 1 && !committed[0]) return StateError::Order;
        if (committed[ub]) return StateError::Duplicate;
        committed[ub] = true;
        rows[ub] = row_end;
        values[ub] = value;
        return StateError::None;
    }

    StateError publish(int ub, uint64_t gen) {
        if (retired) return StateError::Retired;
        if (gen != epoch) return StateError::StaleEpoch;
        if (!committed[ub]) return StateError::NotCommitted;
        if (published[ub]) return StateError::AlreadyPublished;
        published[ub] = true;
        ++publications;
        return StateError::None;
    }

    void fail() {
        retired = true;
        appended = {};
        committed = {};
        published = {};
        rows = {};
        values = {};
        ++epoch;
    }

    bool next_request_allowed() const { return !retired; }
};

bool same_kv(const LayerKv & a, const LayerKv & b) {
    return a.epoch == b.epoch && a.retired == b.retired &&
           a.appended == b.appended && a.committed == b.committed &&
           a.published == b.published && a.rows == b.rows &&
           a.values == b.values && a.publications == b.publications;
}

struct Snapshot {
    std::array<uint64_t, kNodes> out{};
    std::array<LayerKv, kLayers> kv{};
};

bool same_snapshot(const Snapshot & a, const Snapshot & b) {
    if (a.out != b.out) return false;
    for (int l = 0; l < kLayers; ++l) {
        if (!same_kv(a.kv[l], b.kv[l])) return false;
    }
    return true;
}

// Buffer slots carry a generation. A live slot cannot be reallocated (alias
// refused); a consumer/release with an old generation is stale; the last
// consumer frees the slot. A leaf output is released by its producer.
class BufferPool {
public:
    static constexpr int kSlots = 128;
    struct Slot {
        bool live = false;
        uint64_t gen = 0;
        int owner = -1;
        int remaining = 0;
        uint64_t value = 0;
    };

    int alloc(int owner, int consumers, std::string & why) {
        int idx = forced_ >= 0 ? forced_ : first_free();
        forced_ = -1;
        if (idx < 0 || idx >= kSlots) {
            why = "no free buffer slot";
            return -1;
        }
        if (slots_[idx].live) {
            why = "alias: buffer slot " + std::to_string(idx) +
                  " still live (owner " + std::to_string(slots_[idx].owner) + ")";
            return -1;
        }
        slots_[idx].live = true;
        ++slots_[idx].gen;
        slots_[idx].owner = owner;
        slots_[idx].remaining = consumers;
        slots_[idx].value = 0;
        return idx;
    }

    bool publish(int idx, uint64_t gen, uint64_t value, std::string & why) {
        if (idx < 0 || idx >= kSlots) {
            why = "publish: bad buffer slot";
            return false;
        }
        if (!slots_[idx].live) {
            why = "publish: buffer is not live";
            return false;
        }
        if (slots_[idx].gen != gen) {
            why = "publish: stale buffer generation";
            return false;
        }
        slots_[idx].value = value;
        return true;
    }

    bool consume(int idx, uint64_t gen, int owner, std::string & why) {
        (void) owner;
        if (idx < 0 || idx >= kSlots) {
            why = "consume: bad buffer slot";
            return false;
        }
        if (!slots_[idx].live) {
            why = "consume: buffer already free";
            return false;
        }
        if (slots_[idx].gen != gen) {
            why = "consume: stale buffer generation";
            return false;
        }
        if (slots_[idx].remaining <= 0) {
            why = "consume: consumer count underflow";
            return false;
        }
        --slots_[idx].remaining;
        if (slots_[idx].remaining == 0) {
            slots_[idx].live = false;
            slots_[idx].owner = -1;
        }
        return true;
    }

    bool release(int idx, uint64_t gen, std::string & why) {
        if (idx < 0 || idx >= kSlots) {
            why = "release: bad buffer slot";
            return false;
        }
        if (!slots_[idx].live) {
            why = "release: buffer already free";
            return false;
        }
        if (slots_[idx].gen != gen) {
            why = "release: stale buffer generation";
            return false;
        }
        slots_[idx].live = false;
        slots_[idx].owner = -1;
        slots_[idx].remaining = 0;
        return true;
    }

    void force(int idx) { forced_ = idx; }

    int live_count() const {
        int count = 0;
        for (const Slot & s : slots_) count += s.live ? 1 : 0;
        return count;
    }

    const Slot & slot(int idx) const { return slots_[idx]; }

private:
    int first_free() const {
        for (int i = 0; i < kSlots; ++i) {
            if (!slots_[i].live) return i;
        }
        return -1;
    }

    std::array<Slot, kSlots> slots_{};
    int forced_ = -1;
};

// Reference semantics: the same operators in the serial order (U0 all layers,
// then U1), mirroring the model contract. No scheduler, state epochs or
// buffers here.
Snapshot reference_run(uint64_t u1_tokens) {
    Snapshot s;
    const uint64_t tokens[kUb] = {0x55aa55aa55aa55aaULL, u1_tokens};
    for (int u = 0; u < kUb; ++u) {
        for (int l = 0; l < kLayers; ++l) {
            const uint64_t in = l == 0 ? tokens[u] : s.out[nid(Kind::Ffn, l - 1, u)];
            const uint64_t qkv = mix(in, 0x1000 + static_cast<uint64_t>(l * kUb + u));
            const uint64_t prefix = (u == 1) ? s.kv[l].values[0] : 0;
            const uint64_t local = (u == 1 && is_full_layer(l)) ? s.out[nid(Kind::Merge, l, 0)] : 0;
            const uint64_t attn = mix(qkv, prefix ^ 0x2000);
            const uint64_t merge = mix(attn, qkv ^ local ^ 0x3000);
            const uint64_t ffn = mix(merge, 0x4000 + static_cast<uint64_t>(l));
            const uint64_t state = mix(attn, 0x5000 + static_cast<uint64_t>(l));
            s.out[nid(Kind::Qkv, l, u)] = qkv;
            s.out[nid(Kind::Attn, l, u)] = attn;
            s.out[nid(Kind::Merge, l, u)] = merge;
            s.out[nid(Kind::Ffn, l, u)] = ffn;
            s.out[nid(Kind::State, l, u)] = state;
            const uint64_t gen = s.kv[l].epoch;
            const int32_t row_end = kPRuntime + (u + 1) * kN;
            const StateError a = s.kv[l].append(u, gen);
            const StateError c = s.kv[l].commit(u, row_end, state, gen);
            const StateError p = s.kv[l].publish(u, gen);
            if (a != StateError::None || c != StateError::None || p != StateError::None) {
                std::printf("reference state error at L%d U%d\n", l, u);
            }
        }
    }
    return s;
}

class Executor {
public:
    enum class Mode { Serial, Lookahead };
    enum class FailPhase { None, AfterAppend, AfterCommit, AfterPublish };
    struct Interval {
        int node;
        int start;
        int end;
    };
    struct OpState {
        enum Status { Pending = 0, Running = 1, Done = 2, Failed = 3 };
        int status = Pending;
        int start = -1;
        int end = -1;
        uint64_t out = 0;
        int out_slot = -1;
        uint64_t out_gen = 0;
        std::array<int, 3> in_slot{-1, -1, -1};
        std::array<uint64_t, 3> in_gen{};
        std::array<uint64_t, 3> in_val{};
        uint64_t event_gen = 0;
        int deps_count = 0;
    };

    Executor(Mode mode, uint64_t u1_tokens, bool inject_alias = false,
             int fail_layer = -1, int fail_ub = -1,
             FailPhase fail_phase = FailPhase::None)
        : mode_(mode), inject_alias_(inject_alias), fail_layer_(fail_layer),
          fail_ub_(fail_ub), fail_phase_(fail_phase) {
        tokens_[0] = 0x55aa55aa55aa55aaULL;
        tokens_[1] = u1_tokens;
    }

    bool run(std::string & why) {
        while (done_ + failed_ < kNodes) {
            if (mode_ == Mode::Serial) {
                if (!lane_busy_) {
                    const int n = pick_ready(-1);
                    if (n >= 0 && !start(n, now_, why)) return false;
                }
            } else {
                for (int r = 0; r < 2; ++r) {
                    if (res_busy_[r]) continue;
                    const int n = pick_ready(r);
                    if (n >= 0 && !start(n, now_, why)) return false;
                }
            }
            if (events_.empty()) {
                why = "deadlock: no completion event";
                return false;
            }
            now_ = events_.top().time;
            while (!events_.empty() && events_.top().time == now_) {
                const Event e = events_.top();
                events_.pop();
                if (!complete(e, why)) return false;
            }
        }
        return true;
    }

    Snapshot snapshot() const {
        Snapshot s;
        for (int i = 0; i < kNodes; ++i) s.out[i] = ops_[i].out;
        s.kv = kv_;
        return s;
    }

    const std::array<OpState, kNodes> & ops() const { return ops_; }
    const std::array<LayerKv, kLayers> & kv() const { return kv_; }
    const BufferPool & pool() const { return pool_; }
    const std::vector<Interval> & intervals() const { return intervals_; }
    int events_processed() const { return events_processed_; }
    int late_events() const { return late_events_; }
    int alias_rejected() const { return alias_rejected_; }

    bool overlap_pair(int layer) const {
        if (!is_full_layer(layer) || layer + 1 >= kLayers) return false;
        const int a = nid(Kind::Attn, layer, 1);
        const int b = nid(Kind::Qkv, layer + 1, 0);
        const Interval * ia = nullptr;
        const Interval * ib = nullptr;
        for (const Interval & iv : intervals_) {
            if (iv.node == a) ia = &iv;
            if (iv.node == b) ib = &iv;
        }
        if (!ia || !ib) return false;
        return std::max(ia->start, ib->start) < std::min(ia->end, ib->end);
    }

    void inject_late_event(int node, uint64_t stale_gen) {
        events_.push(Event{now_, ++seq_, node, stale_gen});
    }

    bool drain_late(std::string & why) {
        while (!events_.empty()) {
            const Event e = events_.top();
            events_.pop();
            if (!complete(e, why)) return false;
        }
        return true;
    }

private:
    struct Event {
        int time;
        uint64_t seq;
        int node;
        uint64_t gen;
    };

    struct EventGreater {
        bool operator()(const Event & a, const Event & b) const {
            return a.time != b.time ? a.time > b.time : a.seq > b.seq;
        }
    };

    Mode mode_;
    bool inject_alias_;
    int fail_layer_;
    int fail_ub_;
    FailPhase fail_phase_;
    uint64_t tokens_[kUb]{};
    std::array<OpState, kNodes> ops_{};
    std::array<LayerKv, kLayers> kv_{};
    BufferPool pool_;
    std::vector<Interval> intervals_;
    std::priority_queue<Event, std::vector<Event>, EventGreater> events_;
    int now_ = 0;
    uint64_t seq_ = 0;
    bool lane_busy_ = false;
    std::array<bool, 2> res_busy_{};
    int done_ = 0;
    int failed_ = 0;
    int events_processed_ = 0;
    int late_events_ = 0;
    int alias_rejected_ = 0;

    int res_index(const Node & n) const {
        return resource_of(n) == Res::Remote ? 1 : 0;
    }

    void free_res(const Node & n) {
        if (mode_ == Mode::Serial) {
            lane_busy_ = false;
        } else {
            res_busy_[res_index(n)] = false;
        }
    }

    bool ready(int node) const {
        if (ops_[node].status != OpState::Pending) return false;
        for (const int d : deps_of(node)) {
            if (ops_[d].status != OpState::Done) return false;
        }
        return true;
    }

    int pick_ready(int res) const {
        int best = -1;
        int best_score = 1 << 30;
        for (int i = 0; i < kNodes; ++i) {
            const Node n = node_of(i);
            if (ops_[i].status != OpState::Pending) continue;
            if (mode_ == Mode::Lookahead && res_index(n) != res) continue;
            if (!ready(i)) continue;
            int score;
            if (mode_ == Mode::Serial) {
                score = n.ub * 1000 + n.layer * 10 + static_cast<int>(n.kind);
            } else {
                const int ks = (n.kind == Kind::Qkv || n.kind == Kind::Attn) ? 0 : 1;
                score = ks * 1000 + n.layer * 10 + n.ub;
            }
            if (score < best_score) {
                best_score = score;
                best = i;
            }
        }
        return best;
    }

    bool start(int node, int now, std::string & why) {
        const Node n = node_of(node);
        OpState & op = ops_[node];
        if (op.status != OpState::Pending) return true;
        const std::vector<int> deps = deps_of(node);
        op.deps_count = static_cast<int>(deps.size());
        for (size_t i = 0; i < deps.size(); ++i) {
            const OpState & d = ops_[deps[i]];
            op.in_slot[i] = d.out_slot;
            op.in_gen[i] = d.out_gen;
            op.in_val[i] = d.out;
        }
        const int consumers = dependents_of(node);
        if (inject_alias_ && node == nid(Kind::Qkv, 1, 0)) {
            const OpState & first = ops_[nid(Kind::Attn, 0, 1)];
            if (first.out_slot >= 0) {
                pool_.force(first.out_slot);
                const int bad = pool_.alloc(node, consumers, why);
                if (bad < 0) {
                    ++alias_rejected_;
                    op.status = OpState::Failed;
                    ++failed_;
                    why = "alias injected: " + why;
                    return false;
                }
            }
        }
        const int slot = pool_.alloc(node, consumers, why);
        if (slot < 0) {
            ++alias_rejected_;
            op.status = OpState::Failed;
            ++failed_;
            return false;
        }
        op.out_slot = slot;
        op.out_gen = pool_.slot(slot).gen;
        op.start = now;
        op.end = now + duration(n.kind);
        op.status = OpState::Running;
        op.event_gen = kv_[n.layer].epoch;
        intervals_.push_back(Interval{node, now, op.end});
        events_.push(Event{op.end, ++seq_, node, op.event_gen});
        if (mode_ == Mode::Serial) {
            lane_busy_ = true;
        } else {
            res_busy_[res_index(n)] = true;
        }
        return true;
    }

    uint64_t compute(const Node & n, const OpState & op) const {
        switch (n.kind) {
            case Kind::Qkv:
                if (n.layer == 0) {
                    return mix(tokens_[n.ub], 0x1000 + static_cast<uint64_t>(n.layer * kUb + n.ub));
                }
                return mix(op.in_val[0], 0x1000 + static_cast<uint64_t>(n.layer * kUb + n.ub));
            case Kind::Attn:
                return mix(op.in_val[0], (n.ub == 1 ? op.in_val[1] : 0) ^ 0x2000);
            case Kind::Merge:
                return mix(op.in_val[1],
                           op.in_val[0] ^
                               ((n.ub == 1 && is_full_layer(n.layer)) ? op.in_val[2] : 0) ^
                               0x3000);
            case Kind::Ffn:
                return mix(op.in_val[0], 0x4000 + static_cast<uint64_t>(n.layer));
            case Kind::State:
                return mix(op.in_val[0], 0x5000 + static_cast<uint64_t>(n.layer));
        }
        return 0;
    }

    bool complete(const Event & e, std::string & why) {
        ++events_processed_;
        if (e.node < 0 || e.node >= kNodes) {
            why = "bad event node";
            return false;
        }
        OpState & op = ops_[e.node];
        const Node n = node_of(e.node);
        if (op.status != OpState::Running) {
            ++late_events_;
            return true;
        }
        if (e.gen != op.event_gen || e.gen != kv_[n.layer].epoch) {
            ++late_events_;
            op.status = OpState::Failed;
            ++failed_;
            free_res(n);
            return true;
        }
        const uint64_t out = compute(n, op);
        op.out = out;
        if (!pool_.publish(op.out_slot, op.out_gen, out, why)) {
            op.status = OpState::Failed;
            ++failed_;
            free_res(n);
            return false;
        }
        if (n.kind == Kind::State) {
            LayerKv & layer = kv_[n.layer];
            StateError er = layer.append(n.ub, e.gen);
            if (er != StateError::None) {
                op.status = OpState::Failed;
                ++failed_;
                free_res(n);
                why = std::string("state append: ") + state_error_name(er);
                return true;
            }
            const bool fail_here = (fail_layer_ == n.layer && fail_ub_ == n.ub);
            if (fail_here && fail_phase_ == FailPhase::AfterAppend) {
                layer.fail();
                op.status = OpState::Failed;
                ++failed_;
                free_res(n);
                why = "simulated failure after append";
                return true;
            }
            const int32_t row_end = kPRuntime + (n.ub + 1) * kN;
            er = layer.commit(n.ub, row_end, out, e.gen);
            if (er != StateError::None) {
                op.status = OpState::Failed;
                ++failed_;
                free_res(n);
                why = std::string("state commit: ") + state_error_name(er);
                return true;
            }
            if (fail_here && fail_phase_ == FailPhase::AfterCommit) {
                layer.fail();
                op.status = OpState::Failed;
                ++failed_;
                free_res(n);
                why = "simulated failure after commit";
                return true;
            }
            er = layer.publish(n.ub, e.gen);
            if (er != StateError::None) {
                op.status = OpState::Failed;
                ++failed_;
                free_res(n);
                why = std::string("state publish: ") + state_error_name(er);
                return true;
            }
            if (fail_here && fail_phase_ == FailPhase::AfterPublish) {
                layer.fail();
                op.status = OpState::Failed;
                ++failed_;
                free_res(n);
                why = "simulated failure after publication";
                return true;
            }
        }
        op.status = OpState::Done;
        ++done_;
        const std::vector<int> deps = deps_of(e.node);
        for (size_t i = 0; i < deps.size(); ++i) {
            if (!pool_.consume(op.in_slot[i], op.in_gen[i], e.node, why)) {
                op.status = OpState::Failed;
                ++failed_;
                free_res(n);
                return false;
            }
        }
        if (dependents_of(e.node) == 0) {
            if (!pool_.release(op.out_slot, op.out_gen, why)) {
                op.status = OpState::Failed;
                ++failed_;
                free_res(n);
                return false;
            }
        }
        free_res(n);
        return true;
    }
};

int failures = 0;

void check(bool ok, const char * name) {
    std::printf("%s: %s\n", name, ok ? "OK" : "FAIL");
    if (!ok) ++failures;
}

} // namespace

int main() {
    // --- model-level shape/layer invariants -------------------------------
    check(is_full_layer(0) && is_full_layer(1) && !is_full_layer(2),
          "layer mix full/full/recurrent");
    check(std::string(state_semantics(0)) == "remote-KV" &&
          std::string(state_semantics(2)) == "recurrent",
          "full=remote KV, recurrent=horizontal state");
    check(shape_ok(Kind::Qkv, 256) && shape_ok(Kind::Attn, 256) &&
          shape_ok(Kind::Merge, 256) && shape_ok(Kind::Ffn, 256),
          "model shapes stay N256");
    check(!shape_ok(Kind::Ffn, 512), "N512 FFN refused");

    // --- U1-before-U0: append first, so the failure is ordering ----------
    {
        LayerKv kv;
        check(kv.commit(1, kPRuntime + 2 * kN, 0x1111, 1) == StateError::NoAppend,
              "U1 commit with no append is NoAppend");
        check(kv.append(1, 1) == StateError::None, "U1 append succeeds before U0");
        check(kv.commit(1, kPRuntime + 2 * kN, 0x1111, 1) == StateError::Order,
              "U1 append then commit before U0 reports Order (not NoAppend)");
        check(!kv.committed[1], "U1 remains uncommitted after Order");
        check(kv.append(0, 1) == StateError::None, "U0 append");
        check(kv.commit(0, kPRuntime + kN, 0x2222, 1) == StateError::None, "U0 commit");
        check(kv.commit(1, kPRuntime + 2 * kN, 0x1111, 1) == StateError::None,
              "U1 commit after U0 succeeds");
        check(kv.commit(1, kPRuntime + 2 * kN, 0x1111, 1) == StateError::Duplicate,
              "duplicate U1 commit refused");
        check(kv.publish(1, 1) == StateError::None, "U1 publication");
        check(kv.publish(1, 1) == StateError::AlreadyPublished,
              "duplicate U1 publication refused");
        check(kv.publications == 1, "publication sentinel counted once");
    }

    // --- state failures between append/commit/publication ----------------
    {
        LayerKv kv;
        check(kv.append(0, 1) == StateError::None, "fail state: U0 append");
        check(kv.commit(0, kPRuntime + kN, 0x2222, 1) == StateError::None, "fail state: U0 commit");
        check(kv.publish(0, 1) == StateError::None, "fail state: U0 publish");
        check(kv.append(1, 1) == StateError::None, "fail state: U1 append");
        kv.fail();
        check(kv.retired && kv.epoch == 2, "after-append failure retires/advances epoch");
        check(kv.commit(1, kPRuntime + 2 * kN, 0x1111, 1) == StateError::Retired,
              "after-append failure refuses late U1 commit");
        check(kv.publish(1, 2) == StateError::Retired,
              "after-append failure refuses post-retirement U1 publish");
        check(kv.publications == 1, "after-append failure keeps only U0 publication");
        check(!kv.next_request_allowed(), "after-append failure rejects next request");
    }
    {
        LayerKv kv;
        check(kv.append(0, 1) == StateError::None && kv.commit(0, kPRuntime + kN, 0x2222, 1) == StateError::None &&
              kv.publish(0, 1) == StateError::None, "fail-commit state: U0 published");
        check(kv.append(1, 1) == StateError::None, "fail-commit state: U1 append");
        check(kv.commit(1, kPRuntime + 2 * kN, 0x1111, 1) == StateError::None,
              "fail-commit state: U1 commit");
        kv.fail();
        check(kv.retired && kv.publications == 1,
              "after-commit failure: U1 was not published");
        check(kv.publish(1, 2) == StateError::Retired,
              "after-commit failure refuses publication");
        check(!kv.published[1], "after-commit failure leaves U1 unpublished");
    }
    {
        LayerKv kv;
        check(kv.append(0, 1) == StateError::None && kv.commit(0, kPRuntime + kN, 0x2222, 1) == StateError::None &&
              kv.publish(0, 1) == StateError::None, "fail-publish state: U0 published");
        check(kv.append(1, 1) == StateError::None, "fail-publish state: U1 append");
        check(kv.commit(1, kPRuntime + 2 * kN, 0x1111, 1) == StateError::None,
              "fail-publish state: U1 commit");
        check(kv.publish(1, 1) == StateError::None, "fail-publish state: U1 publish");
        check(kv.publications == 2, "fail-publish state: two publications before failure");
        kv.fail();
        check(kv.retired && kv.publications == 2,
              "after-publish failure preserves publication count");
        check(!kv.next_request_allowed(), "after-publish failure rejects next request");
    }
    {
        LayerKv kv;
        kv.epoch = 3;
        check(kv.append(1, 1) == StateError::StaleEpoch, "stale epoch append refused");
        check(kv.commit(1, kPRuntime + 2 * kN, 0x1111, 2) == StateError::StaleEpoch,
              "stale epoch commit refused");
    }

    // --- buffer generations/lifetime/alias --------------------------------
    {
        BufferPool pool;
        std::string why;
        const int s0 = pool.alloc(0, 1, why);
        check(s0 >= 0, "buffer: first allocation");
        const uint64_t g0 = pool.slot(s0).gen;
        check(pool.publish(s0, g0, 42, why), "buffer: publish current generation");
        const int s1 = pool.alloc(1, 1, why);
        check(s1 >= 0 && s1 != s0, "buffer: second allocation is distinct");
        pool.force(s0);
        const int alias = pool.alloc(2, 1, why);
        check(alias < 0 && why.find("alias") != std::string::npos,
              "buffer: forced live slot refused as alias");
        check(!pool.consume(s0, g0 + 1, 0, why), "buffer: stale generation consume refused");
        check(pool.consume(s0, g0, 0, why), "buffer: last consumer releases");
        check(pool.live_count() == 1, "buffer: only s1 remains live");
        check(pool.consume(s1, pool.slot(s1).gen, 1, why), "buffer: s1 consumed/released");
        check(!pool.consume(s1, pool.slot(s1).gen, 1, why),
              "buffer: consume after free refused");
        check(pool.live_count() == 0, "buffer: all slots released");
        const int s2 = pool.alloc(3, 0, why);
        check(s2 == s0 && pool.slot(s2).gen == g0 + 1,
              "buffer: reuse increments generation");
        check(pool.release(s2, pool.slot(s2).gen, why), "buffer: leaf release");
    }

    // --- serial vs lookahead: complete state must match ------------------
    const uint64_t u1 = 0xaaaaaaaaaaaaaaaaULL;
    const Snapshot ref = reference_run(u1);
    check(ref.kv[0].rows[0] == kPRuntime + kN &&
          ref.kv[0].rows[1] == kPRuntime + 2 * kN,
          "reference rows: U0=P+256, U1=P+512");
    check(ref.kv[0].publications == 2 && ref.kv[2].publications == 2,
          "reference publications: two per layer");
    {
        std::string why;
        Executor serial(Executor::Mode::Serial, u1);
        const bool ok = serial.run(why);
        check(ok, "serial executor completes");
        if (!ok) std::printf("  serial: %s\n", why.c_str());
        const Snapshot snap = serial.snapshot();
        check(same_snapshot(snap, ref), "serial full state equals reference");
        check(!serial.overlap_pair(0) && !serial.overlap_pair(1),
              "serial never overlaps the N256 pair");
        check(serial.events_processed() == kNodes, "serial processes one completion per node");
        check(serial.pool().live_count() == 0, "serial releases all buffers");
        check(serial.ops()[nid(Kind::Attn, 0, 1)].out_slot !=
              serial.ops()[nid(Kind::Qkv, 1, 0)].out_slot,
              "serial overlap pair uses distinct buffers");
    }
    {
        std::string why;
        Executor look(Executor::Mode::Lookahead, u1);
        const bool ok = look.run(why);
        check(ok, "lookahead executor completes");
        if (!ok) std::printf("  lookahead: %s\n", why.c_str());
        const Snapshot snap = look.snapshot();
        check(same_snapshot(snap, ref), "lookahead full state equals reference");
        check(same_snapshot(snap, reference_run(u1)), "lookahead state stable across runs");
        check(look.overlap_pair(0) && look.overlap_pair(1),
              "lookahead overlaps the two legal pairs");
        check(look.events_processed() == kNodes, "lookahead processes one completion per node");
        check(look.pool().live_count() == 0, "lookahead releases all buffers");
        check(look.ops()[nid(Kind::Attn, 0, 1)].out_slot !=
              look.ops()[nid(Kind::Qkv, 1, 0)].out_slot,
              "lookahead L0 pair uses distinct buffers");
        check(look.ops()[nid(Kind::Attn, 1, 1)].out_slot !=
              look.ops()[nid(Kind::Qkv, 2, 0)].out_slot,
              "lookahead L1 pair uses distinct buffers");
        check(look.ops()[nid(Kind::State, 0, 0)].out_slot !=
              look.ops()[nid(Kind::State, 0, 1)].out_slot,
              "prefix and tail use distinct buffers");
    }

    // --- failures between append/commit/publication in the executor ------
    {
        std::string why;
        Executor f(Executor::Mode::Serial, u1, false, 0, 1, Executor::FailPhase::AfterAppend);
        check(f.run(why), "executor failure-after-append finishes");
        check(f.kv()[0].retired, "executor failure-after-append retires layer");
        check(f.kv()[0].publications == 1 && !f.kv()[0].published[1],
              "executor failure-after-append publishes only U0");
        check(!f.kv()[0].next_request_allowed(), "executor failure-after-append rejects next");
        f.inject_late_event(nid(Kind::State, 0, 1), 1);
        check(f.drain_late(why), "late completion drained");
        check(f.late_events() >= 1, "late completion after failure is detected");
    }
    {
        std::string why;
        Executor f(Executor::Mode::Serial, u1, false, 0, 1, Executor::FailPhase::AfterCommit);
        check(f.run(why), "executor failure-after-commit finishes");
        check(f.kv()[0].retired && f.kv()[0].publications == 1,
              "executor failure-after-commit: U1 unpublished");
        check(!f.kv()[0].published[1], "executor failure-after-commit keeps U1 unpublished");
    }
    {
        std::string why;
        Executor f(Executor::Mode::Serial, u1, false, 0, 1, Executor::FailPhase::AfterPublish);
        check(f.run(why), "executor failure-after-publication finishes");
        check(f.kv()[0].retired && f.kv()[0].publications == 2,
              "executor failure-after-publication preserves both publications");
        check(!f.kv()[0].next_request_allowed(), "executor failure-after-publication rejects next");
    }

    // --- alias injection and late buffer generation ----------------------
    {
        std::string why;
        Executor alias(Executor::Mode::Lookahead, u1, true);
        const bool ok = alias.run(why);
        check(!ok, "executor refuses injected alias");
        check(alias.alias_rejected() >= 1 && why.find("alias") != std::string::npos,
              "alias rejection is explicit");
    }

    // --- P provenance and boundaries -------------------------------------
    check(kPRuntime == 96256, "runtime P=96256");
    check(kPDeclared == 102400 && kPDeclared != kPRuntime,
          "declared P=102400 preserved as historical, not used");
    check(kTail == 128, "tail 128");
    check(kPRuntime + 2 * kN <= kC, "two N256 ubatches fit in C=131072");
    const int32_t tail_abs = kPRuntime + 2 * kN - kTail;
    check(tail_abs == kPRuntime + 384, "tail absolute start P+384");
    check(tail_abs - (kPRuntime + kN) == kTail, "tail offset inside U1 is 128");
    const auto pair_fits = [](int32_t start) {
        return start >= 0 && start + 2 * kN <= kC;
    };
    check(pair_fits(kC - 2 * kN), "near-C: pair at C-512 allowed");
    check(!pair_fits(kC - kN), "near-C: pair at C-256 refused");

    std::printf("%s: %d failure(s)\n", failures == 0 ? "OK" : "FAILED", failures);
    return failures == 0 ? 0 : 1;
}
