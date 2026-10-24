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

// Default logical durations per kind (QKV, ATTN, MERGE, FFN, STATE).
// These are scheduling fixtures, not latency/TPS estimates.
constexpr std::array<int, kKinds> kDefaultDurations{{1, 2, 1, 1, 1}};

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
        uint64_t consumed = 0; // bitmask of consumer owners (node ids)
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
        slots_[idx].consumed = 0;
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
        if (owner >= 0 && owner < 64) {
            const uint64_t bit = 1ULL << owner;
            if (slots_[idx].consumed & bit) {
                why = "consume: duplicate consumer " + std::to_string(owner);
                return false;
            }
            slots_[idx].consumed |= bit;
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

    bool read(int idx, uint64_t gen, uint64_t & value, std::string & why) const {
        if (idx < 0 || idx >= kSlots) {
            why = "read: bad buffer slot";
            return false;
        }
        if (!slots_[idx].live) {
            why = "read: buffer already free";
            return false;
        }
        if (slots_[idx].gen != gen) {
            why = "read: stale buffer generation";
            return false;
        }
        value = slots_[idx].value;
        return true;
    }

    bool tamper(int idx, uint64_t gen, uint64_t value, std::string & why) {
        if (idx < 0 || idx >= kSlots) {
            why = "tamper: bad buffer slot";
            return false;
        }
        if (!slots_[idx].live) {
            why = "tamper: buffer not live";
            return false;
        }
        if (slots_[idx].gen != gen) {
            why = "tamper: stale buffer generation";
            return false;
        }
        slots_[idx].value = value;
        return true;
    }

    bool force_stale(int idx, std::string & why) {
        if (idx < 0 || idx >= kSlots || !slots_[idx].live) {
            why = "force_stale: buffer not live";
            return false;
        }
        ++slots_[idx].gen;
        return true;
    }

    void release_all() {
        for (Slot & s : slots_) {
            if (s.live) {
                s.live = false;
                s.owner = -1;
                s.remaining = 0;
            }
        }
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

enum class RunOutcome { Completed, SimulatedFailure, Error };

struct RunResult {
    RunOutcome outcome;
    std::string why;
};

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
        enum Status { Pending = 0, Running = 1, Done = 2, Failed = 3, Cancelled = 4 };
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
             FailPhase fail_phase = FailPhase::None,
             int tamper_input_node = -1, uint64_t tamper_value = 0,
             int stale_input_node = -1, int input_alias_node = -1,
             int late_during_node = -1, int duplicate_completion_node = -1)
        : mode_(mode), inject_alias_(inject_alias), fail_layer_(fail_layer),
          fail_ub_(fail_ub), fail_phase_(fail_phase),
          tamper_input_node_(tamper_input_node), tamper_value_(tamper_value),
          stale_input_node_(stale_input_node), input_alias_node_(input_alias_node),
          late_during_node_(late_during_node),
          duplicate_completion_node_(duplicate_completion_node) {
        tokens_[0] = 0x55aa55aa55aa55aaULL;
        tokens_[1] = u1_tokens;
    }

    RunResult run() {
        if (terminal_) return {terminal_outcome_, failure_why_};
        std::string why;
        while (done_ + failed_ < kNodes) {
            if (terminal_) return {terminal_outcome_, failure_why_};
            if (mode_ == Mode::Serial) {
                if (!lane_busy_) {
                    const int n = pick_ready(-1);
                    if (n >= 0 && !start(n, now_, why)) {
                        terminate_error(why);
                        return {terminal_outcome_, failure_why_};
                    }
                }
            } else {
                for (int r = 0; r < 2; ++r) {
                    if (res_busy_[r]) continue;
                    const int n = pick_ready(r);
                    if (n >= 0 && !start(n, now_, why)) {
                        terminate_error(why);
                        return {terminal_outcome_, failure_why_};
                    }
                }
            }
            if (events_.empty()) {
                terminate_error("deadlock: no completion event");
                return {terminal_outcome_, failure_why_};
            }
            now_ = events_.top().time;
            while (!events_.empty() && events_.top().time == now_) {
                const Event e = events_.top();
                events_.pop();
                if (!complete(e, why)) {
                    terminate_error(why);
                    return {terminal_outcome_, failure_why_};
                }
                if (terminal_) return {terminal_outcome_, failure_why_};
            }
        }
        return {RunOutcome::Completed, ""};
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

    bool resource_free() const {
        if (mode_ == Mode::Serial) return !lane_busy_;
        return !res_busy_[0] && !res_busy_[1];
    }

    bool set_durations(const std::array<int, kKinds> & durations) {
        for (const int d : durations) {
            if (d <= 0) return false;
        }
        durations_ = durations;
        return true;
    }

    void inject_old_event_at(int old_node, int at_node) {
        late_old_node_ = old_node;
        late_old_at_ = at_node;
    }

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
    std::array<int, kKinds> durations_{kDefaultDurations};
    int tamper_input_node_;
    uint64_t tamper_value_;
    int stale_input_node_;
    int input_alias_node_;
    int late_during_node_;
    int duplicate_completion_node_;
    int late_old_node_ = -1;
    int late_old_at_ = -1;
    bool terminal_ = false;
    RunOutcome terminal_outcome_ = RunOutcome::Completed;
    std::string failure_why_;
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
            if (tamper_input_node_ == node && i == 0) {
                if (!pool_.tamper(d.out_slot, d.out_gen, tamper_value_, why)) {
                    op.status = OpState::Failed;
                    ++failed_;
                    why = "tamper input: " + why;
                    return false;
                }
            }
            if (stale_input_node_ == node && i == 0) {
                if (!pool_.force_stale(d.out_slot, why)) {
                    op.status = OpState::Failed;
                    ++failed_;
                    why = "stale input injection: " + why;
                    return false;
                }
            }
            // Inputs are read by {slot, generation}; the dependent never uses
            // a copied producer output. A stale/free/aliased input fails before
            // any dependent publication.
            if (!pool_.read(d.out_slot, d.out_gen, op.in_val[i], why)) {
                op.status = OpState::Failed;
                ++failed_;
                why = "input handle: " + why;
                return false;
            }
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
        if (input_alias_node_ == node && !deps.empty()) {
            pool_.force(op.in_slot[0]);
            const int bad = pool_.alloc(node, consumers, why);
            if (bad < 0) {
                ++alias_rejected_;
                op.status = OpState::Failed;
                ++failed_;
                why = "input alias: " + why;
                return false;
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
        op.end = now + durations_[static_cast<int>(n.kind)];
        op.status = OpState::Running;
        op.event_gen = kv_[n.layer].epoch;
        intervals_.push_back(Interval{node, now, op.end});
        events_.push(Event{op.end, ++seq_, node, op.event_gen});
        if (late_during_node_ == node) {
            // A stale completion arrives while the operation is still RUNNING.
            events_.push(Event{now, ++seq_, node, op.event_gen + 1});
        }
        if (duplicate_completion_node_ == node) {
            // A second completion with the right generation must not double-commit.
            events_.push(Event{op.end, ++seq_, node, op.event_gen});
        }
        if (late_old_at_ == node && late_old_node_ >= 0) {
            // Old completion from a previous op while the newer op occupies the
            // reused slot; it must be ignored without touching the new value.
            events_.push(Event{now_, ++seq_, late_old_node_, ops_[late_old_node_].event_gen});
        }
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

    void terminate(bool simulated, const std::string & why) {
        terminal_ = true;
        terminal_outcome_ = simulated ? RunOutcome::SimulatedFailure : RunOutcome::Error;
        failure_why_ = why;
        for (OpState & op : ops_) {
            if (op.status == OpState::Running) op.status = OpState::Cancelled;
        }
        events_ = decltype(events_)();
        pool_.release_all();
        lane_busy_ = false;
        res_busy_ = {false, false};
    }

    void terminate_failure(const std::string & why) { terminate(true, why); }
    void terminate_error(const std::string & why) { terminate(false, why); }

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
            why = "stale completion while RUNNING for " + std::string(kind_name(n.kind)) +
                  " L" + std::to_string(n.layer) + " U" + std::to_string(n.ub);
            terminate_error(why);
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
                why = std::string("state append: ") + state_error_name(er);
                terminate_error(why);
                return true;
            }
            const bool fail_here = (fail_layer_ == n.layer && fail_ub_ == n.ub);
            if (fail_here && fail_phase_ == FailPhase::AfterAppend) {
                layer.fail();
                op.status = OpState::Failed;
                ++failed_;
                why = "simulated failure after append";
                terminate_failure(why);
                return true;
            }
            const int32_t row_end = kPRuntime + (n.ub + 1) * kN;
            er = layer.commit(n.ub, row_end, out, e.gen);
            if (er != StateError::None) {
                op.status = OpState::Failed;
                ++failed_;
                why = std::string("state commit: ") + state_error_name(er);
                terminate_error(why);
                return true;
            }
            if (fail_here && fail_phase_ == FailPhase::AfterCommit) {
                layer.fail();
                op.status = OpState::Failed;
                ++failed_;
                why = "simulated failure after commit";
                terminate_failure(why);
                return true;
            }
            er = layer.publish(n.ub, e.gen);
            if (er != StateError::None) {
                op.status = OpState::Failed;
                ++failed_;
                why = std::string("state publish: ") + state_error_name(er);
                terminate_error(why);
                return true;
            }
            if (fail_here && fail_phase_ == FailPhase::AfterPublish) {
                layer.fail();
                op.status = OpState::Failed;
                ++failed_;
                why = "simulated failure after publication";
                terminate_failure(why);
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

    // --- buffer consumer identity / reuse / handle reads ------------------
    {
        BufferPool pool;
        std::string why;
        const int s = pool.alloc(0, 2, why);
        const uint64_t g = pool.slot(s).gen;
        check(pool.consume(s, g, 7, why), "buffer: first of two consumers");
        check(!pool.consume(s, g, 7, why), "buffer: duplicate consumer refused");
        check(pool.slot(s).remaining == 1, "buffer: duplicate consumer does not decrement");
        check(pool.consume(s, g, 8, why), "buffer: second consumer releases");
        check(!pool.slot(s).live, "buffer: two consumers release once");
    }
    {
        BufferPool pool;
        std::string why;
        const int s = pool.alloc(0, 0, why);
        const uint64_t g = pool.slot(s).gen;
        check(pool.publish(s, g, 1, why), "buffer: publish before reuse");
        check(pool.release(s, g, why), "buffer: release before reuse");
        const int s2 = pool.alloc(1, 0, why);
        check(s2 == s && pool.slot(s2).gen == g + 1, "buffer: reuse increments generation");
        check(!pool.publish(s, g, 2, why), "buffer: old publication after reuse refused");
        check(!pool.consume(s, g, 7, why), "buffer: old consume after reuse refused");
        check(!pool.release(s, g, why), "buffer: old release after reuse refused");
    }
    {
        BufferPool pool;
        std::string why;
        const int s = pool.alloc(0, 1, why);
        const uint64_t g = pool.slot(s).gen;
        uint64_t v = 0;
        check(pool.publish(s, g, 42, why), "buffer: publish read handle");
        check(pool.read(s, g, v, why) && v == 42, "buffer: read current handle");
        check(pool.tamper(s, g, 99, why), "buffer: tamper current handle");
        check(pool.read(s, g, v, why) && v == 99, "buffer: tampered value visible");
        check(pool.force_stale(s, why), "buffer: force stale generation");
        check(!pool.read(s, g, v, why), "buffer: stale handle read refused");
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
        Executor serial(Executor::Mode::Serial, u1);
        const RunResult sr = serial.run();
        check(sr.outcome == RunOutcome::Completed, "serial executor completes");
        if (sr.outcome != RunOutcome::Completed) std::printf("  serial: %s\n", sr.why.c_str());
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
        Executor look(Executor::Mode::Lookahead, u1);
        const RunResult lr = look.run();
        check(lr.outcome == RunOutcome::Completed, "lookahead executor completes");
        if (lr.outcome != RunOutcome::Completed) std::printf("  lookahead: %s\n", lr.why.c_str());
        const Snapshot snap = look.snapshot();
        check(same_snapshot(snap, ref), "lookahead full state equals reference");
        Executor look2(Executor::Mode::Lookahead, u1);
        const RunResult lr2 = look2.run();
        check(lr2.outcome == RunOutcome::Completed, "second lookahead executor completes");
        check(same_snapshot(snap, look2.snapshot()),
              "two lookahead instances produce the same complete state");
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

    // --- variable durations / simultaneous completions -------------------
    {
        Executor serial(Executor::Mode::Serial, u1);
        serial.set_durations({{1, 3, 2, 1, 1}});
        const RunResult r = serial.run();
        check(r.outcome == RunOutcome::Completed, "variable-duration serial completes");
        check(same_snapshot(serial.snapshot(), ref),
              "variable-duration serial full state equals reference");
        check(serial.pool().live_count() == 0, "variable-duration serial releases buffers");
    }
    {
        Executor look(Executor::Mode::Lookahead, u1);
        look.set_durations({{1, 3, 2, 1, 1}});
        const RunResult r = look.run();
        check(r.outcome == RunOutcome::Completed, "variable-duration lookahead completes");
        check(same_snapshot(look.snapshot(), ref),
              "variable-duration lookahead full state equals reference");
        check(look.overlap_pair(0) && look.overlap_pair(1),
              "variable durations preserve both legal overlaps");
        check(look.events_processed() == kNodes && look.pool().live_count() == 0,
              "variable durations: one event per node and buffers released");
    }
    {
        Executor serial(Executor::Mode::Serial, u1);
        serial.set_durations({{1, 1, 1, 1, 1}});
        const RunResult r = serial.run();
        check(r.outcome == RunOutcome::Completed, "unit-duration serial completes");
        check(same_snapshot(serial.snapshot(), ref),
              "unit-duration serial full state equals reference");
        check(serial.pool().live_count() == 0, "unit-duration serial releases buffers");
    }
    {
        Executor look(Executor::Mode::Lookahead, u1);
        look.set_durations({{1, 1, 1, 1, 1}});
        const RunResult r = look.run();
        check(r.outcome == RunOutcome::Completed, "simultaneous-completion lookahead completes");
        check(same_snapshot(look.snapshot(), ref),
              "simultaneous completions preserve the complete state");
        // With all-unit durations the lookahead may be merely adjacent; the
        // complete state and event ordering are the discriminants here. The
        // sequence tie-break is deterministic but not separately observed.
        int max_same_end = 0;
        for (int i = 0; i < kNodes; ++i) {
            int same = 0;
            for (int j = 0; j < kNodes; ++j) {
                if (look.ops()[j].end == look.ops()[i].end) ++same;
            }
            max_same_end = std::max(max_same_end, same);
        }
        check(max_same_end >= 2,
              "unit durations: two completions share a timestamp (seq tie-break)");
        check(look.events_processed() == kNodes,
              "simultaneous completions: one event per node");
        check(look.pool().live_count() == 0, "simultaneous completions release buffers");
    }
    {
        Executor f(Executor::Mode::Lookahead, u1, false, 0, 1, Executor::FailPhase::AfterCommit);
        f.set_durations({{1, 1, 1, 3, 2}});
        const RunResult r = f.run();
        check(r.outcome == RunOutcome::SimulatedFailure, "in-flight failure is terminal");
        check(f.pool().live_count() == 0 && f.resource_free(),
              "in-flight failure cleans buffers/resources");
        check(!f.kv()[0].published[1], "in-flight failure keeps U1 unpublished");
        bool cancelled = false;
        for (const auto & op : f.ops()) {
            cancelled = cancelled || op.status == Executor::OpState::Cancelled;
        }
        check(cancelled, "in-flight failure has a RUNNING peer op");
        const size_t n = f.intervals().size();
        const Snapshot before = f.snapshot();
        const RunResult r2 = f.run();
        check(r2.outcome == RunOutcome::SimulatedFailure, "in-flight failure sticky");
        check(f.intervals().size() == n && same_snapshot(f.snapshot(), before),
              "in-flight failure does not dispatch or change state");
    }

    // --- old completion against a reused buffer slot ---------------------
    {
        BufferPool pool;
        std::string why;
        const int old_slot = pool.alloc(0, 1, why);
        const uint64_t old_gen = pool.slot(old_slot).gen;
        check(pool.publish(old_slot, old_gen, 0xaaa, why), "slot reuse: old operation publishes");
        check(pool.consume(old_slot, old_gen, 0, why), "slot reuse: old consumer releases");
        const int new_slot = pool.alloc(1, 0, why);
        const uint64_t new_gen = pool.slot(new_slot).gen;
        check(new_slot == old_slot && new_gen == old_gen + 1, "slot reuse: new operation reuses slot");
        check(pool.publish(new_slot, new_gen, 0xbbb, why), "slot reuse: new operation publishes");
        uint64_t value = 0;
        check(pool.read(new_slot, new_gen, value, why) && value == 0xbbb,
              "slot reuse: new operation value readable");
        check(!pool.publish(old_slot, old_gen, 0xccc, why),
              "slot reuse: old completion publication refused");
        check(!pool.consume(old_slot, old_gen, 0, why),
              "slot reuse: old completion consume refused");
        check(pool.read(new_slot, new_gen, value, why) && value == 0xbbb,
              "slot reuse: new operation value intact");
    }
    {
        Executor bad(Executor::Mode::Serial, u1);
        check(!bad.set_durations({{0, 1, 1, 1, 1}}), "non-positive duration refused");
        check(bad.set_durations({{1, 2, 1, 1, 1}}), "positive durations accepted");
    }
    {
        Executor base(Executor::Mode::Serial, u1);
        const RunResult br = base.run();
        check(br.outcome == RunOutcome::Completed, "slot reuse executor baseline completes");
        int a = -1;
        int b = -1;
        for (int i = 0; i < kNodes && a < 0; ++i) {
            if (base.ops()[i].out_slot < 0) continue;
            for (int j = i + 1; j < kNodes; ++j) {
                if (base.ops()[j].out_slot == base.ops()[i].out_slot) {
                    a = i;
                    b = j;
                    break;
                }
            }
        }
        check(a >= 0 && b > a, "slot reuse: executor reuses an output slot");
        if (a >= 0 && b > a) {
            Executor re(Executor::Mode::Serial, u1);
            re.inject_old_event_at(a, b);
            const RunResult rr = re.run();
            check(rr.outcome == RunOutcome::Completed,
                  "slot reuse: old event does not abort the new run");
            check(re.late_events() >= 1, "slot reuse: old completion is late");
            check(re.ops()[b].out_slot == re.ops()[a].out_slot,
                  "slot reuse: newer op occupies the reused slot");
            check(re.ops()[b].out == base.ops()[b].out,
                  "slot reuse: newer op value intact");
            check(same_snapshot(re.snapshot(), ref), "slot reuse: complete state intact");
            check(re.pool().live_count() == 0 && re.resource_free(),
                  "slot reuse: buffers/resources released");
            check(re.ops()[a].status == Executor::OpState::Done &&
                  re.ops()[b].status == Executor::OpState::Done,
                  "slot reuse: both operations complete");
        }
    }

    // --- failures between append/commit/publication in the executor ------
    {
        struct Case {
            const char * name;
            Executor::Mode mode;
            int ub;
        };
        const Case cases[] = {
            {"U0 serial", Executor::Mode::Serial, 0},
            {"U1 serial", Executor::Mode::Serial, 1},
            {"U0 lookahead", Executor::Mode::Lookahead, 0},
            {"U1 lookahead", Executor::Mode::Lookahead, 1},
        };
        for (const Case & c : cases) {
            Executor f(c.mode, u1, false, 0, c.ub, Executor::FailPhase::AfterAppend);
            if (c.mode == Executor::Mode::Lookahead) {
                f.set_durations({{1, 1, 1, 3, 2}});
            }
            const RunResult r = f.run();
            check(r.outcome == RunOutcome::SimulatedFailure,
                  (std::string(c.name) + ": terminal simulated failure").c_str());
            check(f.kv()[0].retired, (std::string(c.name) + ": layer retired").c_str());
            check(f.pool().live_count() == 0, (std::string(c.name) + ": buffers released").c_str());
            check(f.resource_free(), (std::string(c.name) + ": resources released").c_str());
            const size_t n_intervals = f.intervals().size();
            const Snapshot before = f.snapshot();
            const RunResult r2 = f.run();
            check(r2.outcome == r.outcome, (std::string(c.name) + ": sticky terminal").c_str());
            check(f.intervals().size() == n_intervals,
                  (std::string(c.name) + ": no dispatch after terminal").c_str());
            check(same_snapshot(f.snapshot(), before),
                  (std::string(c.name) + ": state unchanged after terminal").c_str());
            if (c.ub == 0) {
                check(f.ops()[nid(Kind::Attn, 0, 1)].status != Executor::OpState::Done,
                      "U0 failure invalidates the U1 predecessor");
            }
            if (c.mode == Executor::Mode::Lookahead) {
                bool cancelled = false;
                for (const auto & op : f.ops()) {
                    cancelled = cancelled || op.status == Executor::OpState::Cancelled;
                }
                check(cancelled, "lookahead failure had an in-flight op");
            }
        }
    }
    {
        Executor f(Executor::Mode::Serial, u1, false, 0, 1, Executor::FailPhase::AfterCommit);
        const RunResult r = f.run();
        check(r.outcome == RunOutcome::SimulatedFailure, "failure-after-commit terminal");
        check(f.kv()[0].retired && f.kv()[0].publications == 1,
              "failure-after-commit: U1 unpublished");
        check(!f.kv()[0].published[1], "failure-after-commit keeps U1 unpublished");
        check(f.pool().live_count() == 0, "failure-after-commit releases buffers");
        check(f.resource_free(), "failure-after-commit releases resources");
        const size_t n = f.intervals().size();
        const Snapshot before = f.snapshot();
        const RunResult r2 = f.run();
        check(r2.outcome == RunOutcome::SimulatedFailure, "failure-after-commit sticky");
        check(f.intervals().size() == n && same_snapshot(f.snapshot(), before),
              "failure-after-commit does not dispatch or change state");
    }
    {
        Executor f(Executor::Mode::Serial, u1, false, 0, 1, Executor::FailPhase::AfterPublish);
        const RunResult r = f.run();
        check(r.outcome == RunOutcome::SimulatedFailure, "failure-after-publication terminal");
        check(f.kv()[0].retired && f.kv()[0].publications == 2,
              "failure-after-publication preserves both publications");
        check(f.pool().live_count() == 0, "failure-after-publication releases buffers");
        check(f.resource_free(), "failure-after-publication releases resources");
        const size_t n = f.intervals().size();
        const Snapshot before = f.snapshot();
        const RunResult r2 = f.run();
        check(r2.outcome == RunOutcome::SimulatedFailure, "failure-after-publication sticky");
        check(f.intervals().size() == n && same_snapshot(f.snapshot(), before),
              "failure-after-publication does not dispatch or change state");
    }
    {
        std::string why;
        Executor f(Executor::Mode::Serial, u1, false, 0, 1, Executor::FailPhase::AfterAppend);
        const RunResult r = f.run();
        check(r.outcome == RunOutcome::SimulatedFailure, "post-failure late test terminal");
        f.inject_late_event(nid(Kind::State, 0, 1), 1);
        check(f.drain_late(why), "late completion drained after failure");
        check(f.late_events() >= 1, "late completion after failure is detected");
        check(f.pool().live_count() == 0, "late completion after failure keeps buffers released");
        const size_t n = f.intervals().size();
        const Snapshot before = f.snapshot();
        const RunResult r2 = f.run();
        check(r2.outcome == RunOutcome::SimulatedFailure, "post-failure terminal remains sticky");
        check(f.intervals().size() == n && same_snapshot(f.snapshot(), before),
              "post-failure late event does not change state");
    }

    // --- alias / late-running / duplicate completion ---------------------
    {
        Executor alias(Executor::Mode::Lookahead, u1, true);
        const RunResult r = alias.run();
        check(r.outcome == RunOutcome::Error, "executor refuses injected alias");
        check(alias.alias_rejected() >= 1 && r.why.find("alias") != std::string::npos,
              "alias rejection is explicit");
        check(alias.pool().live_count() == 0, "alias error cleans buffers");
        check(alias.resource_free(), "alias error frees resources");
        const size_t n = alias.intervals().size();
        const Snapshot before = alias.snapshot();
        const RunResult r2 = alias.run();
        check(r2.outcome == RunOutcome::Error, "alias terminal is sticky");
        check(alias.intervals().size() == n && same_snapshot(alias.snapshot(), before),
              "alias terminal does not dispatch or change state");
    }
    {
        Executor late(Executor::Mode::Serial, u1, false, -1, -1, Executor::FailPhase::None,
                      -1, 0, -1, -1, nid(Kind::Qkv, 0, 0));
        const RunResult r = late.run();
        check(r.outcome == RunOutcome::Error, "late event while RUNNING terminates as Error");
        check(r.why.find("stale") != std::string::npos, "late event error names stale completion");
        check(late.late_events() >= 1, "late event while RUNNING is detected");
        check(late.ops()[nid(Kind::Qkv, 0, 0)].status == Executor::OpState::Failed,
              "late completion fails the current op");
        check(late.resource_free(), "late completion releases the resource");
        check(late.pool().live_count() == 0, "late completion cleans buffers");
        const size_t n = late.intervals().size();
        const Snapshot before = late.snapshot();
        const RunResult r2 = late.run();
        check(r2.outcome == RunOutcome::Error, "late terminal is sticky");
        check(late.intervals().size() == n && same_snapshot(late.snapshot(), before),
              "late terminal does not dispatch or change state");
    }
    {
        Executor dup(Executor::Mode::Serial, u1, false, -1, -1, Executor::FailPhase::None,
                     -1, 0, -1, -1, -1, nid(Kind::Qkv, 0, 0));
        const RunResult r = dup.run();
        check(r.outcome == RunOutcome::Completed, "duplicate completion run completes");
        check(dup.late_events() >= 1, "duplicate completion is detected as late");
        check(same_snapshot(dup.snapshot(), ref),
              "duplicate completion does not change the complete state");
        check(dup.pool().live_count() == 0, "duplicate completion releases buffers");
    }

    // --- inputs are read through pool handles ----------------------------
    {
        Executor t(Executor::Mode::Serial, u1, false, -1, -1, Executor::FailPhase::None,
                   nid(Kind::Qkv, 1, 0), 0xdeadbeefULL);
        const RunResult r = t.run();
        check(r.outcome == RunOutcome::Completed, "tamper run completes");
        check(t.snapshot().out[nid(Kind::Qkv, 1, 0)] != ref.out[nid(Kind::Qkv, 1, 0)],
              "pool content feeds the dependent computation");
        check(!same_snapshot(t.snapshot(), ref), "tampered pool state differs from reference");
    }
    {
        Executor st(Executor::Mode::Serial, u1, false, -1, -1, Executor::FailPhase::None,
                    -1, 0, nid(Kind::Qkv, 1, 0));
        const RunResult r = st.run();
        check(r.outcome == RunOutcome::Error, "stale input handle rejected as error");
        check(r.why.find("stale") != std::string::npos, "stale input error names generation");
        check(st.ops()[nid(Kind::Qkv, 1, 0)].status != Executor::OpState::Done,
              "stale input prevents dependent publication");
        check(st.pool().live_count() == 0, "stale input cleans buffers");
        check(st.resource_free(), "stale input frees resources");
        const size_t n = st.intervals().size();
        const Snapshot before = st.snapshot();
        const RunResult r2 = st.run();
        check(r2.outcome == RunOutcome::Error, "stale input terminal is sticky");
        check(st.intervals().size() == n && same_snapshot(st.snapshot(), before),
              "stale input terminal does not dispatch or change state");
    }
    {
        Executor ia(Executor::Mode::Serial, u1, false, -1, -1, Executor::FailPhase::None,
                    -1, 0, -1, nid(Kind::Qkv, 1, 0));
        const RunResult r = ia.run();
        check(r.outcome == RunOutcome::Error, "input alias rejected as error");
        check(ia.alias_rejected() >= 1 && r.why.find("alias") != std::string::npos,
              "input alias rejection is explicit");
        check(ia.ops()[nid(Kind::Qkv, 1, 0)].status != Executor::OpState::Done,
              "input alias prevents dependent publication");
        check(ia.pool().live_count() == 0, "input alias cleans buffers");
        check(ia.resource_free(), "input alias frees resources");
        const size_t n = ia.intervals().size();
        const Snapshot before = ia.snapshot();
        const RunResult r2 = ia.run();
        check(r2.outcome == RunOutcome::Error, "input alias terminal is sticky");
        check(ia.intervals().size() == n && same_snapshot(ia.snapshot(), before),
              "input alias terminal does not dispatch or change state");
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
