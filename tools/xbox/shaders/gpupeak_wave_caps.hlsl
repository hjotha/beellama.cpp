// PSO capability probes only. No 16-bit types or vendor extensions.
// The UAV stores keep the tested operations live in DXIL. Creating these PSOs
// does NOT prove execution correctness or speed; gpupeak reports that explicitly.
cbuffer Params : register(b0) {
    uint iters, groups_x, groups_y, seed;
};
RWStructuredBuffer<uint> Sink : register(u0);

[numthreads(64, 1, 1)]
void CSWaveCaps(uint tid : SV_GroupIndex) {
    uint lane = WaveGetLaneIndex();
    float x = float(lane + seed);
#if defined(GPUPK_WAVE_MULTI_PREFIX)
    uint bits = 0xffffu << (lane & 16u);
    uint4 mask = lane < 32 ? uint4(bits, 0, 0, 0) : uint4(0, bits, 0, 0);
    Sink[tid] = asuint(WaveMultiPrefixSum(x, mask) + x);
#elif defined(GPUPK_WAVE_MATCH)
    uint4 mask = WaveMatch(lane / 16);
    Sink[tid] = mask.x ^ mask.y ^ mask.z ^ mask.w;
#elif defined(GPUPK_WAVE_QUAD)
    Sink[tid] = asuint(QuadReadAcrossX(x)) ^ asuint(QuadReadAcrossY(x)) ^
                asuint(QuadReadAcrossDiagonal(x));
#else
    // Same SM 6.4 operations as the RKVA candidate, with constant broadcasts.
    float p = WavePrefixSum(x) + x;
    float a = WaveReadLaneAt(p, 15), b = WaveReadLaneAt(p, 31);
    float c = WaveReadLaneAt(p, 47), d = WaveReadLaneAt(p, 63);
    Sink[tid] = asuint(a) ^ asuint(b - a) ^ asuint(c - b) ^ asuint(d - c);
#endif
}
