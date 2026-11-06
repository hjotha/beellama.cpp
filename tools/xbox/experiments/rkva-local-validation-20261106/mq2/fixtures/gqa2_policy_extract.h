#include <cstdint>
namespace policy {
constexpr uint32_t kOptGqa=6,kStorageKvarn4=3,kModeOptFine=2,kKvarn4MqTile=2,kOptMaxQueries=9,kOptHeadDim=256,kKvarn4MqWaveLanes=64;
// Native KVarN4 descriptor v1 shapes the optimized kernel reads directly:
// 128-aligned origin and a 0 or 128 token precision tail.
constexpr uint32_t kKvarnGroup = 128;
constexpr uint32_t kKvarnTile = 8;

// Opt-in GQA2 candidate (rkva_range_attn_kvarn4_mq_gqa2.hlsl): the same global
// 24Q/4KV shape and [split][query * n_head + head][258] partials, but each
// group computes only two local Q heads. The host dispatches three consecutive
// x groups per KV head (DispatchX == 3 * n_head_kv) so every global head is
// still written exactly once. It reloads each K/V row for three groups instead
// of one; profitability is UNKNOWN and this remains opt-in.
constexpr uint32_t kKvarn4Gqa2HeadsPerGroup = 2;
constexpr uint32_t kKvarn4Gqa2GroupsPerKvHead = kOptGqa / kKvarn4Gqa2HeadsPerGroup;

inline uint32_t kvarn4_gqa2_dispatch_x(uint32_t n_head_kv) {
    return kKvarn4Gqa2GroupsPerKvHead * n_head_kv;
}

// Fail-closed selection for the GQA2 candidate. Requires the opt-in arm, no
// INT8 arm (the two specializations are mutually exclusive), descriptor
// revision 3, native KVarN4, the fine 2-query tile, nq 2..9, head_dim 256,
// GQA 6, a 128-token exact tail, fixed wave64 and the dedicated PSO. Every
// other call keeps the qualified default kernel.
inline bool kvarn4_gqa2_supported(bool enabled, bool i8_enabled, uint32_t mode,
                                  uint32_t descriptor_revision, uint32_t storage, uint32_t q_tile,
                                  uint32_t n_tokens, uint32_t n_head, uint32_t n_head_kv,
                                  uint32_t head_dim, uint32_t precision_tail_tokens,
                                  uint32_t wave_lane_min, uint32_t wave_lane_max,
                                  bool pso_available) {
    return enabled && !i8_enabled && descriptor_revision == 3 && storage == kStorageKvarn4 &&
           mode == kModeOptFine && q_tile == kKvarn4MqTile && n_tokens >= 2 &&
           n_tokens <= kOptMaxQueries && head_dim == kOptHeadDim && n_head_kv > 0 &&
           n_head == kOptGqa * n_head_kv && precision_tail_tokens == kKvarnGroup &&
           wave_lane_min == kKvarn4MqWaveLanes && wave_lane_max == kKvarn4MqWaveLanes &&
           pso_available;
}

// Engine selection plan. `dispatch_x` is the x dimension the engine must use:
// three groups per KV head only when the candidate was selected, otherwise the
// unchanged n_head_kv. `kvarn4_gqa2_groups` yields the matching pass-1 group
// count for samples and bulk segment profiles.
struct Kvarn4Gqa2Plan {
    bool selected = false;
    uint32_t dispatch_x = 0;
};

inline Kvarn4Gqa2Plan kvarn4_gqa2_plan(bool enabled, bool i8_enabled, uint32_t mode,
                                       uint32_t descriptor_revision, uint32_t storage,
                                       uint32_t q_tile, uint32_t n_tokens, uint32_t n_head,
                                       uint32_t n_head_kv, uint32_t head_dim,
                                       uint32_t precision_tail_tokens, uint32_t wave_lane_min,
                                       uint32_t wave_lane_max, bool pso_available) {
    const bool selected = kvarn4_gqa2_supported(
        enabled, i8_enabled, mode, descriptor_revision, storage, q_tile, n_tokens, n_head,
        n_head_kv, head_dim, precision_tail_tokens, wave_lane_min, wave_lane_max, pso_available);
    return {selected, selected ? kvarn4_gqa2_dispatch_x(n_head_kv) : n_head_kv};
}

inline uint32_t kvarn4_gqa2_groups(const Kvarn4Gqa2Plan& plan, uint32_t query_groups,
                                   uint32_t split_k) {
    return plan.dispatch_x * query_groups * split_k;
}

// LocalState arming: `rkva-mq-gqa2.flag` is opt-in and default off. If the
// INT8 flag is also present the pair is ambiguous, so GQA2 stays off and the
// profile reports the conflict; INT8 keeps its own existing selection.
enum class MqGqa2Arm : uint8_t { kOff = 0, kOn = 1, kConflict = 2 };

inline MqGqa2Arm mq_gqa2_arm(bool gqa2_flag, bool i8_flag) {
    if (!gqa2_flag) {
        return MqGqa2Arm::kOff;
    }
    return i8_flag ? MqGqa2Arm::kConflict : MqGqa2Arm::kOn;
}

inline const char* mq_gqa2_arm_reason(MqGqa2Arm arm) {
    switch (arm) {
    case MqGqa2Arm::kOn:
        return "enabled";
    case MqGqa2Arm::kConflict:
        return "disabled: conflicts with rkva-mq-i8.flag (fail-closed)";
    default:
        return "disabled (rkva-mq-gqa2.flag absent)";
    }
}

}
