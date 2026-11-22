#include "llama-moe-cache.h"

#include "llama-impl.h"
#include "llama-memory.h"
#include "llama-model.h"

#include "ggml-cpp.h"
#include "ggml-remote-attn-trace.h"

#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {

// SLRU of the experts of a group of layers, the slot of each expert is kept in the slot map of its layer
struct moe_cache_slru {
    int32_t n_expert = 0;
    int32_t n_slots  = 0;

    std::vector<int32_t *> slot_map; // [n_layer] data of the slot maps, -1 if the expert is not cached
    std::vector<int32_t>   key_of;   // [n_slots] il*n_expert + expert, -1 if empty

    // probation and protected LRU lists; hits promote slots to the protected half
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    int32_t head[2] = { -1, -1 };
    int32_t tail[2] = { -1, -1 };
    std::vector<bool> is_protected;
    int32_t n_protected = 0;

    std::vector<uint32_t> seen; // [n_expert]
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;

    // number of slots that hold an expert of each layer, kept in sync by plan and plan_partial
    std::vector<int32_t> n_cached;

    // slots guaranteed to each layer by the partial prefill admission; 0 for uncached layers
    std::vector<int32_t> quota;

    // slots that still hold no expert; each filled slot leaves it for good
    std::vector<int32_t> free_slots;
    std::vector<int32_t> free_pos;

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        head[0] = 0;
        tail[0] = n_slots - 1;
        is_protected.assign(n_slots, false);
        seen.assign(n_expert, 0);
        n_cached.assign(n_layer, 0);
        quota.assign(n_layer, 0);
        free_slots.resize(n_slots);
        free_pos.assign(n_slots, -1);
        for (int32_t s = 0; s < n_slots; ++s) {
            // take_free_slot pops the back, so hand out slots in ascending order and let
            // consecutive experts land in consecutive slots
            free_slots[s] = n_slots - 1 - s;
            free_pos[free_slots[s]] = s;
        }
    }

    void remove_free(int32_t s) {
        const int32_t pos = free_pos[s];
        if (pos < 0) {
            return;
        }
        const int32_t last = free_slots.back();
        free_slots[pos] = last;
        free_pos[last]  = pos;
        free_slots.pop_back();
        free_pos[s] = -1;
    }

    // drop the key of slot s, if any, and empty the slot
    void evict(int32_t s) {
        if (key_of[s] < 0) {
            return;
        }
        const int32_t old_il = key_of[s] / n_expert;
        slot_map[old_il][key_of[s] % n_expert] = -1;
        GGML_ASSERT(n_cached[old_il] > 0);
        n_cached[old_il]--;
        key_of[s] = -1;
    }

    // move slot s to the most recently used end of a segment
    void move(int32_t s, bool protect) {
        const bool from = is_protected[s];
        if (from == protect && s == tail[from]) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head[from] = next[s];
        }
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else {
            tail[from] = prev[s];
        }

        prev[s] = tail[protect];
        next[s] = -1;
        if (tail[protect] >= 0) {
            next[tail[protect]] = s;
        } else {
            head[protect] = s;
        }
        tail[protect] = s;
        is_protected[s] = protect;
        n_protected += int(protect) - int(from);
    }

    void touch(int32_t s) {
        move(s, true);
        if (n_protected > n_slots/2) {
            move(head[1], false);
        }
    }

    int32_t victim(int32_t il) const {
        for (int segment = 0; segment < 2; ++segment) {
            for (int32_t s = head[segment]; s >= 0; s = next[s]) {
                const int32_t key = key_of[s];
                // keep every expert selected by this request, including earlier fills
                if (key < 0 || key / n_expert != il || seen[key % n_expert] != seen_gen) {
                    return s;
                }
            }
        }
        GGML_ABORT("no SLRU victim");
    }

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    // give a slot to each expert selected by ids in layer il, evicting from probation first
    // returns false if the ids select more distinct experts than there are slots
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, std::vector<fill> & fills, size_t & n_hit) {
        fills.clear();
        n_hit = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) n_slots) {
            return false;
        }

        int32_t * slots = slot_map[il];

        // promote hits before choosing victims for misses
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                touch(slots[e]);
                n_hit++;
            }
        }
        // sorted misses usually get consecutive slots, so the uploads can be merged
        std::sort(uniq.begin(), uniq.end());
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                continue;
            }
            const int32_t s = victim(il);
            evict(s);
            remove_free(s);
            key_of[s] = il*n_expert + e;
            slots[e] = s;
            n_cached[il]++;
            move(s, false);
            fills.push_back({ e, s });
        }
        return true;
    }

    // Give a slot to as many experts selected by ids in layer il as fit in quota, evicting
    // first the experts of the same layer that this request does not use. Experts that do
    // not fit stay out of the cache and are copied directly by the caller, so this never
    // fails. The per-layer quota keeps a stable resident subset across large batches even
    // when one request selects more experts than the whole group can hold.
    void plan_partial(int32_t il, const int32_t * ids, size_t n_ids, int32_t quota,
            std::vector<fill> & fills, size_t & n_hit, size_t & n_bypass) {
        fills.clear();
        n_hit = 0;
        n_bypass = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }

        int32_t * slots = slot_map[il];
        if (slots == nullptr) {
            n_bypass = uniq.size();
            return;
        }

        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                touch(slots[e]);
                n_hit++;
            }
        }
        // sorted misses usually get consecutive slots, so the uploads can be merged
        std::sort(uniq.begin(), uniq.end());

        int32_t n_res = n_cached[il];
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                continue;
            }
            int32_t s = -1;
            if (n_res >= quota) {
                // the layer is full: replace one of its experts that this request does not use
                if (n_res == (int32_t) (n_hit + fills.size())) {
                    // every resident expert is selected, so there is nothing to replace
                    n_bypass++;
                    continue;
                }
                s = victim_of_layer(il);
            } else {
                s = take_free_slot();
                if (s < 0) {
                    s = victim_of_other_layer(il);
                }
            }
            if (s < 0) {
                n_bypass++;
                continue;
            }
            evict(s);
            remove_free(s);
            key_of[s] = il*n_expert + e;
            slots[e] = s;
            n_cached[il]++;
            // a replacement inside the layer leaves n_cached[il] unchanged
            n_res = n_cached[il];
            move(s, false);
            fills.push_back({ e, s });
        }
    }

    int32_t take_free_slot() const {
        return free_slots.empty() ? -1 : free_slots.back();
    }

    // a slot of layer il that holds an expert this request does not select, probation first
    int32_t victim_of_layer(int32_t il) const {
        for (int segment = 0; segment < 2; ++segment) {
            for (int32_t s = head[segment]; s >= 0; s = next[s]) {
                const int32_t key = key_of[s];
                if (key >= 0 && key / n_expert == il && seen[key % n_expert] != seen_gen) {
                    return s;
                }
            }
        }
        return -1;
    }

    // while a layer is below its quota the cache may still be full because another layer
    // holds more than its share, e.g. after small batches used the whole group
    int32_t victim_of_other_layer(int32_t il) const {
        // First pass: a layer that holds more than its guaranteed quota is hoarding slots.
        // Second pass: any other layer, for the case where every layer is at its share.
        for (int pass = 0; pass < 2; ++pass) {
            for (int segment = 0; segment < 2; ++segment) {
                for (int32_t s = head[segment]; s >= 0; s = next[s]) {
                    const int32_t key = key_of[s];
                    if (key < 0 || key / n_expert == il) {
                        continue;
                    }
                    const int32_t other = key / n_expert;
                    if (pass == 0 && (other >= (int32_t) quota.size() || n_cached[other] <= quota[other])) {
                        continue;
                    }
                    return s;
                }
            }
        }
        return -1;
    }
};

static constexpr bool moe_cache_admit(int64_t n_tokens, int64_t n_expert_used, int64_t n_expert,
        size_t n_layers, int64_t n_slots, bool prefill) {
    return n_tokens > 0 && std::min(n_tokens*n_expert_used, n_expert) <= n_slots &&
        (n_tokens <= 32 || (prefill && (int64_t) n_layers*n_expert <= n_slots));
}

// Large batches cannot use the slot map of a partial group: every expert they select must
// have a slot, and a full ubatch selects more experts than the group holds. They can still
// fill and reuse the slots through the scheduler copy path, one expert tensor at a time.
static constexpr bool moe_cache_admit_partial_prefill(int64_t n_tokens, int64_t n_expert,
        size_t n_layers, int64_t n_slots, bool enabled) {
    return enabled && n_tokens > 32 && n_slots > 0 && n_slots < (int64_t) n_layers*n_expert;
}

// the number of group slots a layer may keep for large batches; the remainder goes to the
// first layers, so the quotas always add up to n_slots
static constexpr int32_t moe_cache_partial_quota(int64_t n_slots, size_t n_layers, size_t layer_index) {
    return (int32_t) (n_slots/n_layers + (layer_index < (size_t) (n_slots % n_layers) ? 1 : 0));
}

static constexpr bool moe_cache_device_map(bool enabled, bool cuda, int64_t slots, int64_t required,
        size_t spare_bytes, size_t map_bytes) {
    return enabled && cuda && slots == required && map_bytes <= spare_bytes;
}

static_assert(moe_cache_device_map(true, true, 256, 256, 1024, 1024), "complete CUDA group can be immutable");
static_assert(!moe_cache_device_map(true, true, 255, 256, 1024, 1024), "partial groups need eviction");
static_assert(!moe_cache_device_map(true, false, 256, 256, 1024, 1024), "non-CUDA groups use the existing path");
static_assert(!moe_cache_device_map(false, true, 256, 256, 1024, 1024), "device maps are opt-in");
static_assert(!moe_cache_device_map(true, true, 256, 256, 1023, 1024), "the full group also needs map space");

static_assert(moe_cache_admit(32, 8, 128, 2, 128, false), "small batch can use a partial group");
static_assert(!moe_cache_admit(32, 8, 128, 2, 127, true), "selected experts must fit");
static_assert(!moe_cache_admit(33, 8, 128, 2, 255, true), "large batch bypasses a partial group");
static_assert(!moe_cache_admit(33, 8, 128, 2, 256, false), "large batch stays opt-in");
static_assert(moe_cache_admit(1024, 8, 128, 2, 256, true), "large batch can use a full group");

static_assert(moe_cache_admit_partial_prefill(33, 128, 2, 255, true), "large batch can fill a partial group");
static_assert(!moe_cache_admit_partial_prefill(33, 128, 2, 255, false), "partial prefill is opt-in");
static_assert(!moe_cache_admit_partial_prefill(32, 128, 2, 255, true), "small batches use the slot map");
static_assert(!moe_cache_admit_partial_prefill(33, 128, 2, 256, true), "full groups use the slot map");

static_assert(moe_cache_partial_quota(255, 2, 0) == 128, "one extra slot goes to the first layer");
static_assert(moe_cache_partial_quota(255, 2, 1) == 127, "the quotas add up to the group slots");
static_assert(moe_cache_partial_quota(254, 2, 0) == 127, "even quotas are exact");
static_assert(moe_cache_partial_quota(254, 2, 1) == 127, "even quotas are exact");
static_assert(moe_cache_partial_quota(1, 4, 0) == 1, "one slot can go to a single layer");
static_assert(moe_cache_partial_quota(1, 4, 3) == 0, "the remaining layers get no large-batch slots");

// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
            return false;
        }
    }
    return true;
}

static bool llama_moe_cache_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

}

struct llama_moe_cache::impl {
    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device {
        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;
        size_t host_bytes = 0; // host experts of the layers it caches
        double split = 0.0;    // share of the budget

        // banks and their views
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;
    };

    // layers of the same device with the same expert tensor layout share the banks and the LRU of a group
    struct group {
        int32_t id;                       // device
        std::vector<ggml_tensor *> ref;   // expert tensors of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        bool device_map = false;
        mutable bool logged_prefill = false;
        mutable bool logged_partial = false;
        moe_cache_slru slru;
    };

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached
        int32_t quota = 0;                  // slots reserved for large batches of a partial group
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert]
        int32_t first_slot = -1;           // immutable contiguous range, if present
        std::vector<ggml_tensor *> experts; // host expert tensors, in the order of the banks
    };

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
    };

    struct stats {
        size_t hits   = 0;
        size_t misses = 0;
        size_t bytes  = 0;
    };

    int32_t n_expert_used;

    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;
    stats stats_copy;  // experts copied from the cache for large batches

    // partial prefill: large batches that fill and reuse the slots of a partial group
    struct prefill_stats {
        size_t batches     = 0;
        size_t hits        = 0; // selected experts already resident
        size_t fills       = 0; // selected experts admitted to the bank
        size_t bypass      = 0; // selected experts that did not fit
        size_t fill_bytes  = 0; // bank uploads
        size_t dst_bytes   = 0; // direct uploads to the scheduler copy
    };
    prefill_stats stats_prefill;

    uint32_t batch_gen = 0;
    std::vector<uint32_t> layer_batch; // [n_layer] generation when prepare_layer last ran

    std::vector<moe_cache_slru::fill> partial_fills;

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, int32_t> layer_of; // slot map -> layer

    std::vector<int32_t> ids;
    std::vector<moe_cache_slru::fill> fills;

    const bool prefill = llama_moe_cache::prefill_enabled();

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    // views used by copy_experts
    ggml_context_ptr ctx_views;

    impl(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size) :
            n_expert_used(model.hparams.n_expert_used_max()),
            layer_batch(model.layers.size(), 0),
            layers(model.layers.size()) {
        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("MoE cache requires a GPU backend");
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }

        // only cache layers that keep all of their experts in host memory, on the device the layer is assigned to
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
                continue;
            }
            const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return ggml_backend_get_device(d.backend) == model.dev_layer(il); });
            if (it_dev == devices.end()) {
                continue;
            }
            const int32_t id = (int32_t) (it_dev - devices.begin());
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && llama_moe_cache_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
            }
            it->layers.push_back(il);
            for (const ggml_tensor * t : experts) {
                it->host_bytes         += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }

        // one extra slot at the end, CUDA MMQ can read past the last expert
        auto alloc_size = [&](const group & g, int32_t n_slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD(t->nb[2]*(n_slots + 1), alignment);
            }
            return res;
        };

        // the budget is split among the devices with host experts like the layers, by the tensor split or by default by free memory
        const float * tensor_split = model.tensor_split();
        const bool split_by_free = tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
        double split_sum = 0.0;
        for (device & d : devices) {
            if (d.host_bytes == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            if (split_by_free) {
                size_t free;
                size_t total;
                ggml_backend_dev_memory(dev, &free, &total);
                d.split = (double) free;
            } else {
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                GGML_ASSERT(it != model.devices.end());
                d.split = (double) tensor_split[it - model.devices.begin()];
            }
            split_sum += d.split;
        }
        if (split_sum == 0.0) {
            // the devices do not report their free memory
            for (device & d : devices) {
                d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                split_sum += d.split;
            }
        }

        // within a device the budget is split by the size of the experts, so each group caches the same fraction of its experts
        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;
        const char * map_env = std::getenv("LLAMA_MOE_MAP_DEVICE");
        const bool map_enabled = map_env && std::strcmp(map_env, "1") == 0;
        for (group & g : groups) {
            const device & d = devices[g.id];
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  budget    = (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes);
            const int32_t max_slots = g.layers.size()*n_expert;
            while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= budget) {
                g.n_slots++;
            }
            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            for (size_t i = 0; i < g.layers.size(); ++i) {
                layers[g.layers[i]].quota = moe_cache_partial_quota(g.n_slots, g.layers.size(), i);
            }
            const auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(d.backend));
            const bool cuda = reg && std::strcmp(ggml_backend_reg_name(reg), "CUDA") == 0;
            const size_t map_bytes = g.layers.size()*GGML_PAD(n_expert*sizeof(int32_t), ggml_backend_buft_get_alignment(d.buft));
            g.device_map = moe_cache_device_map(map_enabled, cuda, g.n_slots, max_slots,
                    budget - alloc_size(g, g.n_slots), map_bytes);
            if (g.device_map) {
                n_tensors[g.id] += g.layers.size();
                devices[g.id].buf_size += map_bytes;
            } else {
                g.slru.init(model.layers.size(), n_expert, g.n_slots);
                for (size_t li = 0; li < g.layers.size(); ++li) {
                    g.slru.quota[g.layers[li]] = layers[g.layers[li]].quota;
                }
                n_tensors_host += g.layers.size();
            }
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size());
        }
        if (std::all_of(n_tensors.begin(), n_tensors.end(), [](size_t n) { return n == 0; })) {
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
        }

        auto init_ctx = [](size_t n_tensors) {
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr res(ggml_init(params));
            if (!res) {
                throw std::runtime_error("failed to create the MoE cache context");
            }
            return res;
        };
        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] > 0) {
                devices[id].ctx = init_ctx(n_tensors[id]);
            }
        }
        if (n_tensors_host) {
            ctx_host = init_ctx(n_tensors_host);
        }
        ctx_views = init_ctx(2);

        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = devices[g.id].ctx.get();
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + 1);
                GGML_ASSERT(bank->nb[2] == t->nb[2]);
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            int32_t first_slot = 0;
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig      = (int32_t) ig;
                l.experts = llama_moe_cache_layer_experts(model.layers[il]);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], g.n_slots, bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                }
                l.slot_map = ggml_new_tensor_2d(g.device_map ? ctx : ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                if (g.device_map) {
                    l.first_slot = first_slot;
                    first_slot += l.slot_map->ne[1];
                } else {
                    layer_of[l.slot_map] = il;
                    buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
                }
            }
            devices[g.id].buf_size += alloc_size(g, g.n_slots);
        }

        // with no_alloc, dummy buffers are attached so the memory use can still be measured, see llama_context::memory_breakdown
        for (device & d : devices) {
            if (!d.ctx) {
                continue;
            }
            d.buf = llama_memory_bufs::alloc_buf(d.ctx.get(), d.buft, no_alloc, "MoE cache");
            if (!no_alloc) {
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
        }
        if (ctx_host) {
            buf_host = llama_memory_bufs::alloc_buf(ctx_host.get(), buft_host, no_alloc, "MoE cache slot maps");
            ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            if (!no_alloc) {
                ggml_backend_buffer_clear(buf_host.get(), 0xff);
                buf_host_size = ggml_backend_buffer_get_size(buf_host.get());
            }
        }

        for (group & g : groups) {
            if (no_alloc || g.n_slots == 0) {
                continue;
            }
            if (!g.device_map) {
                for (int32_t il : g.layers) {
                    g.slru.slot_map[il] = (int32_t *) layers[il].slot_map->data;
                }
                continue;
            }

            // Publish before graph reservation. No slot in this group is ever reused.
            ggml_backend_t backend = devices[g.id].backend;
            {
                llama_moe_upload upload(backend);
                for (int32_t il : g.layers) {
                    const layer & l = layers[il];
                    for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                        const ggml_tensor * w = l.experts[ip];
                        upload.set(g.banks[ip], w->data, l.first_slot*w->nb[2], ggml_nbytes(w));
                    }
                }
            }
            ggml_backend_synchronize(backend);
            std::vector<int32_t> map(g.ref[0]->ne[2]);
            for (int32_t il : g.layers) {
                const layer & l = layers[il];
                for (size_t e = 0; e < map.size(); ++e) {
                    map[e] = l.first_slot + e;
                }
                ggml_backend_tensor_set(l.slot_map, map.data(), 0, ggml_nbytes(l.slot_map));
            }
            LLAMA_LOG_INFO("%s: immutable MoE map on %s, %d slots, %zu weight bytes and %zu map bytes published\n",
                    __func__, ggml_backend_name(backend), g.n_slots, g.host_bytes, g.n_slots*sizeof(int32_t));
        }

        // Keep lookups and expert matrix multiplies on the bank's device.
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts\n", __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0);
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%)\n", __func__,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]));
                }
            }
        }
    }

    ggml_backend_t backend(int32_t il) const {
        return devices[groups[layers[il].ig].id].backend;
    }

    ~impl() {
        log_stats();
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];
        const group & g = groups[l.ig];

        const int64_t n_expert = l.slot_map->ne[1];
        if (!moe_cache_admit(n_tokens, n_expert_used, n_expert, g.layers.size(), g.n_slots, prefill)) {
            return nullptr;
        }
        if (n_tokens > 32 && !g.logged_prefill) {
            g.logged_prefill = true;
            LLAMA_LOG_INFO("%s: large batch of %" PRId64 " tokens uses the MoE cache of group %d: %zu layers, %" PRId64 " experts, %d slots\n",
                __func__, n_tokens, l.ig, g.layers.size(), n_expert, g.n_slots);
        }

        return l.slot_map;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
    }

    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return 0;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        if (backend != devices[g.id].backend) {
            return 0;
        }

        // this path only reads the cache, it never fills slots, so the experts used in generation stay in it
        int64_t n;
        int64_t slot;
        if (g.device_map) {
            slot = layers[b.il].first_slot + e;
            n = last - e + 1;
        } else {
            const int32_t * slots = g.slru.slot_map[b.il];
            if (slots == nullptr || slots[e] < 0) {
                return 0;
            }
            slot = slots[e];
            n = 1;
            while (e + n <= last && slots[e + n] == slot + n) {
                n++;
            }
        }

        ggml_tensor * bank = g.banks[b.ip];
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slot*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        stats_copy.hits  += n;
        stats_copy.bytes += ggml_nbytes(src_view);

        return n;
    }

    void begin_batch() {
        if (++batch_gen == 0) {
            std::fill(layer_batch.begin(), layer_batch.end(), 0);
            batch_gen = 1;
        }
    }

    // For a large batch of a partial group, keep as many of the selected experts of layer il
    // in the bank as its quota allows. Fills every expert tensor of the layer at once, so the
    // scheduler copies that follow only need to move resident rows from the bank to their dst.
    bool prepare_layer(ggml_backend_t backend, const ggml_tensor * w,
            const int32_t * ids, size_t n_ids, int64_t n_tokens) {
        if (!llama_moe_cache::partial_prefill_enabled()) {
            return false;
        }
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return false;
        }
        const int32_t il = it->second.il;
        const layer & l = layers[il];
        if (l.ig < 0) {
            return false;
        }
        group & g = groups[l.ig];
        if (g.n_slots == 0 || g.device_map || g.slru.slot_map[il] == nullptr) {
            return false;
        }
        if (!moe_cache_admit_partial_prefill(n_tokens, g.ref[0]->ne[2], g.layers.size(), g.n_slots, true)) {
            return false;
        }
        if (backend != devices[g.id].backend) {
            return false;
        }
        if (layer_batch[il] == batch_gen) {
            return true;
        }
        layer_batch[il] = batch_gen;

        size_t n_hit = 0;
        size_t n_bypass = 0;
        g.slru.plan_partial(il, ids, n_ids, l.quota, partial_fills, n_hit, n_bypass);
        stats_prefill.batches++;
        stats_prefill.hits   += n_hit;
        stats_prefill.bypass += n_bypass;
        LLAMA_LOG_DEBUG("%s: partial prefill layer %d: hits = %zu, fills = %zu, bypass = %zu, quota = %d\n",
                __func__, il, n_hit, partial_fills.size(), n_bypass, l.quota);

        if (partial_fills.empty()) {
            return true;
        }

        // a slot holds the same expert of every tensor of the layer, upload them together;
        // consecutive experts in consecutive slots go up in a single copy
        llama_moe_upload upload(backend);
        for (size_t i = 0; i < partial_fills.size();) {
            size_t n = 1;
            while (i + n < partial_fills.size() &&
                    partial_fills[i + n].expert == partial_fills[i].expert + (int32_t) n &&
                    partial_fills[i + n].slot   == partial_fills[i].slot   + (int32_t) n) {
                n++;
            }
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                const ggml_tensor * src  = l.experts[ip];
                ggml_tensor       * bank = g.banks[ip];
                const size_t expert_size = src->nb[2];
                upload.set(bank, (const uint8_t *) src->data + partial_fills[i].expert*expert_size,
                        partial_fills[i].slot*expert_size, n*expert_size);
                stats_prefill.fill_bytes += n*expert_size;
            }
            i += n;
        }
        stats_prefill.fills += partial_fills.size();

        if (!g.logged_partial) {
            g.logged_partial = true;
            LLAMA_LOG_INFO("%s: large batch of %" PRId64 " tokens fills the partial MoE cache of group %d: "
                    "%zu layers, %" PRId64 " experts, %d slots, quota %d per layer\n",
                    __func__, n_tokens, l.ig, g.layers.size(), (int64_t) g.ref[0]->ne[2], g.n_slots, l.quota);
        }
        return true;
    }

    // direct host-to-device bytes for the scheduler copy of a cached expert tensor
    void note_copy_upload(size_t bytes) {
        stats_prefill.dst_bytes += bytes;
    }

    llama_moe_cache_stats get_stats() const {
        llama_moe_cache_stats res;
        res.large_batches = stats_prefill.batches;
        res.large_hits    = stats_prefill.hits;
        res.large_fills   = stats_prefill.fills;
        res.large_bypass  = stats_prefill.bypass;
        res.large_fill_bytes = stats_prefill.fill_bytes;
        res.large_copy_bytes = stats_prefill.dst_bytes;
        return res;
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }
        const int32_t il = it->second;
        const layer & l = layers[il];
        group & g = groups[l.ig];

        GGML_ASSERT(backend == devices[g.id].backend);

        // the get_rows that looks up the slots of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
        GGML_ASSERT(lookup != nullptr);

        // the selected experts must be computed in an earlier split
        // the scheduler starts a new split at the lookup because it reads a host weight, but only if the split already has inputs
        const ggml_tensor * sel = lookup->src[1];
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(sel));

        ids.resize(ggml_nelements(sel));
        {
            ggml_remote_attn_trace_scope route("strata_moe_route", sel);
            route.enabled &= llama_prefill_profile_enabled();
            if (route.enabled) {
                route.event.nq = sel->ne[1];
                route.event.bytes = ggml_nbytes(sel);
                route.event.flags = 1; // bank fill
            }
            ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));
            ggml_backend_synchronize(backend);
        }

        size_t n_hit = 0;
        if (!g.slru.plan(il, ids.data(), ids.size(), fills, n_hit)) {
            // get_slot_map enables the cache only when the experts selected by one layer fit in the
            // slots of the group, so a plan that does not fit means the decision and the selection
            // disagree; the graph is already built, abort instead of changing it here
            GGML_ABORT("the MoE cache cannot plan layer %d: %d slots, %zu layers, %d experts, %zu selected ids",
                il, g.n_slots, g.layers.size(), (int) g.ref[0]->ne[2], ids.size());
        }

        // upload the missing experts, consecutive experts going to consecutive slots are uploaded together
        size_t bytes = 0;
        llama_moe_upload upload(backend);
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            for (size_t i = 0; i < fills.size();) {
                size_t n = 1;
                while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                    n++;
                }
                upload.set(bank, (const uint8_t *) w->data + fills[i].expert*expert_size, fills[i].slot*expert_size, n*expert_size);
                bytes += n*expert_size;
                i += n;
            }
        }

        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += fills.size();
        st.bytes  += bytes;

        // the next copy synchronizes the backend before it changes the slot map again
        ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));

        return true;
    }

    void log_stats() const {
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu, hit rate = %.2f%%, uploaded = %.2f MiB\n",
                name, st.hits, st.misses, 100.0*st.hits/n, st.bytes/1024.0/1024.0);
        };
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }
        if (stats_prefill.batches > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: partial prefill: batches = %zu, hits = %zu, fills = %zu, bypass = %zu, "
                    "uploaded = %.2f MiB (%.2f MiB fills, %.2f MiB copy misses)\n",
                    stats_prefill.batches, stats_prefill.hits, stats_prefill.fills, stats_prefill.bypass,
                    (stats_prefill.fill_bytes + stats_prefill.dst_bytes)/1024.0/1024.0,
                    stats_prefill.fill_bytes/1024.0/1024.0, stats_prefill.dst_bytes/1024.0/1024.0);
        }
    }
};

llama_moe_upload::llama_moe_upload(ggml_backend_t backend) : backend(backend) {
    static const bool enabled = [] {
        const char * env = std::getenv("LLAMA_MOE_UPLOAD_OVERLAP");
        return env && std::strcmp(env, "1") == 0;
    }();
    if (enabled) {
        const auto dev = ggml_backend_get_device(backend);
        const auto reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg) {
            upload = reinterpret_cast<upload_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_moe_upload"));
        }
    }
}

llama_moe_upload::~llama_moe_upload() {
    if (pending) {
        upload(backend, nullptr, nullptr, 0, 0, false);
    }
}

void llama_moe_upload::set(ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_remote_attn_trace_scope timing("strata_moe_upload", tensor);
    timing.enabled &= llama_prefill_profile_enabled();
    if (timing.enabled) {
        timing.event.bytes = size;
        timing.event.flags = (upload ? 1u : 0u) | (ggml_backend_buffer_is_host(tensor->buffer) ? 2u : 0u);
    }
    if (upload) {
        upload(backend, tensor, data, offset, size, !pending);
        pending = true;
    } else {
        ggml_backend_tensor_set_async(backend, tensor, data, offset, size);
    }
}

bool llama_moe_cache::prefill_enabled() {
    const char * env = std::getenv("LLAMA_MOE_CACHE_PREFILL");
    return env && std::strcmp(env, "1") == 0;
}

bool llama_moe_cache::partial_prefill_enabled() {
    const char * env = std::getenv("LLAMA_MOE_CACHE_PARTIAL_PREFILL");
    return env && std::strcmp(env, "1") == 0;
}

llama_moe_cache::llama_moe_cache(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size) :
    pimpl(new impl(model, no_alloc, backends, bufts, size)) {
}

llama_moe_cache::~llama_moe_cache() = default;

ggml_backend_t llama_moe_cache::backend(int32_t il) const {
    return pimpl->backend(il);
}

ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
    return pimpl->get_slot_map(il, n_tokens, n_expert_used);
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return pimpl->copy(backend, src, dst, graph);
}

int64_t llama_moe_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
    return pimpl->copy_experts(backend, w, dst, e, last);
}

void llama_moe_cache::begin_batch() {
    pimpl->begin_batch();
}

bool llama_moe_cache::prepare_layer(ggml_backend_t backend, const ggml_tensor * w,
        const int32_t * ids, size_t n_ids, int64_t n_tokens) {
    return pimpl->prepare_layer(backend, w, ids, n_ids, n_tokens);
}

void llama_moe_cache::note_copy_upload(size_t bytes) {
    pimpl->note_copy_upload(bytes);
}

llama_moe_cache_stats llama_moe_cache::get_stats() const {
    return pimpl->get_stats();
}

// unit-test hooks for the partial planner, not used by the cache itself
bool llama_moe_cache_test_admit_partial(int64_t n_tokens, int64_t n_expert, size_t n_layers,
        int64_t n_slots, bool enabled) {
    return moe_cache_admit_partial_prefill(n_tokens, n_expert, n_layers, n_slots, enabled);
}

std::vector<llama_moe_cache_test_batch> llama_moe_cache_test_plan_partial(
        int32_t n_expert, int32_t n_slots, int32_t quota,
        const std::vector<std::vector<int32_t>> & batches) {
    moe_cache_slru slru;
    slru.init(1, n_expert, n_slots);
    std::vector<int32_t> map(n_expert, -1);
    slru.slot_map[0] = map.data();

    std::vector<llama_moe_cache_test_batch> res;
    res.reserve(batches.size());
    for (const auto & ids : batches) {
        std::vector<moe_cache_slru::fill> fills;
        size_t hits = 0;
        size_t bypass = 0;
        slru.plan_partial(0, ids.data(), ids.size(), quota, fills, hits, bypass);

        std::vector<uint32_t> seen(n_expert, 0);
        size_t used = 0;
        for (int32_t e : ids) {
            if (!seen[e]) {
                seen[e] = 1;
                used++;
            }
        }

        llama_moe_cache_test_batch b;
        b.used   = used;
        b.hits   = hits;
        b.fills  = fills.size();
        b.bypass = bypass;
        b.slots  = map;
        res.push_back(std::move(b));
    }
    return res;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.buf) {
            res[ggml_backend_buffer_get_type(d.buf.get())] += d.buf_size;
        }
    }
    if (pimpl->buf_host) {
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += pimpl->buf_host_size;
    }
    return res;
}
