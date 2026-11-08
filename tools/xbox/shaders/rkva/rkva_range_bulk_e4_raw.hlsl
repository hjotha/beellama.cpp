// Native-row E4M3FN codecs. Unlike KVarN, OUT is rounded directly from F32.
// Same root constants and compact readback/status layout as the E5 codec.
cbuffer Params : register(b0) { uint q_pairs; uint aux_words; uint reserved0; uint reserved1; };
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);

uint e4_round_shift(uint value, uint shift) {
    const uint low = value & ((1u << shift) - 1u);
    const uint half = 1u << (shift - 1u);
    const uint base = value >> shift;
    return base + ((low > half || (low == half && (base & 1u))) ? 1u : 0u);
}
uint e4_from_f32_bits(uint bits) {
    const uint mag = bits & 0x7fffffffu;
    const uint sign = (bits >> 24u) & 0x80u;
    if (mag >= 0x43e00000u) return sign | 0x7eu;
    if (mag <= 0x3a800000u) return sign;
    const uint exponent = mag >> 23u;
    const uint mantissa = mag & 0x7fffffu;
    return sign | (mag < 0x3c800000u
        ? e4_round_shift(mantissa | 0x800000u, 141u - exponent)
        : ((exponent - 120u) << 3u) + e4_round_shift(mantissa, 20u));
}
uint e4_to_f32_bits(uint code) {
    uint bits = (code & 0x80u) << 24u;
    const uint exponent = (code & 0x7fu) >> 3u;
    const uint mantissa = code & 7u;
    if (exponent != 0u) bits |= ((exponent + 120u) << 23u) | (mantissa << 20u);
    else if (mantissa != 0u) {
        const uint power = mantissa >= 4u ? 2u : (mantissa >= 2u ? 1u : 0u);
        bits |= ((power + 118u) << 23u) | ((mantissa << (23u - power)) & 0x7fffffu);
    }
    return bits;
}
[numthreads(256, 1, 1)]
void CSBulkUnpackQE4Raw(uint3 tid : SV_DispatchThreadID) {
    const uint idx = tid.x;
    if (idx < q_pairs) {
        const uint packed = src.Load((idx / 2u) * 4u) >> ((idx & 1u) * 16u);
        dst.Store(idx * 8u, e4_to_f32_bits(packed & 0xffu));
        dst.Store(idx * 8u + 4u, e4_to_f32_bits((packed >> 8u) & 0xffu));
    }
    if (idx < aux_words) dst.Store((q_pairs * 2u + idx) * 4u, src.Load(q_pairs * 2u + idx * 4u));
}
groupshared uint gs_bad;
[numthreads(256, 1, 1)]
void CSBulkPackOutE4Raw(uint lane : SV_GroupIndex, uint3 group : SV_GroupID) {
    const uint idx = group.x * 256u + lane;
    const uint elements = q_pairs * 2u;
    if (lane == 0u) gs_bad = 0u;
    GroupMemoryBarrierWithGroupSync();
    uint bad = 0u;
    if (idx < elements / 4u) {
        const uint4 values = src.Load4(idx * 16u);
        const uint lse = src.Load((elements + idx / 64u) * 4u); // D=256
        uint packed = 0u;
        [unroll] for (uint j = 0u; j < 4u; ++j) {
            if ((values[j] & 0x7fffffffu) >= 0x7f800000u) bad |= 1u;
            // Preserve exact-zero empty-prefix validation before lossy packing.
            if (lse == 0xff800000u && (values[j] & 0x7fffffffu) != 0u) bad |= 4u;
            packed |= e4_from_f32_bits(values[j]) << (j * 8u);
        }
        dst.Store(idx * 4u, packed);
    }
    if (idx < aux_words) {
        const uint lse = src.Load((elements + idx) * 4u);
        if ((lse & 0x7f800000u) == 0x7f800000u) bad |= 2u;
        dst.Store(elements + idx * 4u, lse);
    }
    if (bad != 0u) InterlockedOr(gs_bad, bad);
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u) dst.Store(elements + aux_words * 4u + group.x * 4u, gs_bad);
}
