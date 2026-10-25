// DFlash2 build_dflash2_conv (llama.cpp/src/models/dflash.cpp).
// All buffers contain contiguous F32 in GGML axis order (ne[0] is fastest).
// hidden/output: [5120, n_tokens]
// dynamic:       [320 groups, 2 taps, 2 sides, n_tokens], projection output
// base:          [5120 channels, 2 taps, 2 sides], unmodified GGUF F32 bytes
// n_tokens must be divisible by n_blocks; each block is padded independently.
// Dispatch ceil(5120 * n_tokens / 64) groups for side 0 or 1.
cbuffer Params : register(b0) {
    uint n_tokens;
    uint n_blocks;
    uint side;
    uint reserved;
};

ByteAddressBuffer hidden : register(t0);
ByteAddressBuffer dynamic_coeff : register(t1);
ByteAddressBuffer base_weight : register(t2);
RWByteAddressBuffer output : register(u0);

static const uint DFLASH_HIDDEN = 5120;
static const uint DFLASH_GROUP = 16;
static const uint DFLASH_GROUPS = 320;
static const uint DFLASH_TAPS = 2;

[numthreads(64, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    const uint element = tid.x;
    if (n_blocks == 0 || side >= 2 || element >= n_tokens * DFLASH_HIDDEN) return;

    const uint token = element / DFLASH_HIDDEN;
    const uint channel = element % DFLASH_HIDDEN;
    const uint group = channel / DFLASH_GROUP;
    const uint block_size = n_tokens / n_blocks;
    if (block_size == 0) return;

    const uint coeff_base = token * (2 * DFLASH_TAPS * DFLASH_GROUPS)
                          + side * (DFLASH_TAPS * DFLASH_GROUPS) + group;
    const uint weight_base = side * (DFLASH_TAPS * DFLASH_HIDDEN) + channel;
    const float current = asfloat(hidden.Load(element * 4));
    const float weight0 = asfloat(base_weight.Load(weight_base * 4))
                        + asfloat(dynamic_coeff.Load(coeff_base * 4));
    // GGML materializes each multiply before adding the two taps.
    precise float result = weight0 * current;

    // ggml_concat([zeros, previous], axis=token) resets at *every* block.
    if (token % block_size != 0) {
        const float previous = asfloat(hidden.Load((element - DFLASH_HIDDEN) * 4));
        const float weight1 = asfloat(base_weight.Load((weight_base + DFLASH_HIDDEN) * 4))
                            + asfloat(dynamic_coeff.Load((coeff_base + DFLASH_GROUPS) * 4));
        result += weight1 * previous;
    }
    output.Store(element * 4, asuint(result));
}
