// GGML llama-kv-cache Hadamard rotation for a 128-element attention head.
// transform_width=128 rotates the whole Q/K head; transform_width=64 rotates
// each half of the V/output head independently. The orthonormal matrix is
// involutive, so the same kernel undoes the V rotation after attention.
// b0={batch,heads,transform_width,0}; t0/input and u0/output are contiguous
// F32 [batch,heads,128]. Dispatch(heads,batch,1).
cbuffer Params : register(b0) {
    uint batch;
    uint heads;
    uint transform_width;
    uint reserved;
};
ByteAddressBuffer X : register(t0);
RWByteAddressBuffer Y : register(u0);

groupshared float values[128];

[numthreads(128,1,1)]
void CSHadamard(uint lane : SV_GroupIndex, uint3 group : SV_GroupID) {
    if (group.x >= heads || group.y >= batch ||
        (transform_width != 64u && transform_width != 128u)) return;
    const uint index = (group.y * heads + group.x) * 128u + lane;
    // GGML FWHT scales each input before the butterfly passes. Moving this
    // multiply after the passes changes F32 rounding near Q4/Q8 thresholds.
    const float factor = transform_width == 64u ? 0.125f : 0.08838834764831845f;
    values[lane] = asfloat(X.Load(index * 4u)) * factor;
    GroupMemoryBarrierWithGroupSync();
    [loop] for (uint stride = 1u; stride < transform_width; stride <<= 1u) {
        const float own = values[lane];
        const float other = values[lane ^ stride];
        // Both reads must complete before any lane overwrites its value.
        GroupMemoryBarrierWithGroupSync();
        values[lane] = (lane & stride) != 0u ? other - own : own + other;
        GroupMemoryBarrierWithGroupSync();
    }
    Y.Store(index * 4u, asuint(values[lane]));
}
