// Shared integer E4M3FN conversion; callers reject non-finite inputs.
uint e4m3fn_to_half(uint code) {
    const uint sign = (code & 0x80u) << 8;
    const uint magnitude = code & 0x7fu;
    const uint exponent = magnitude >> 3;
    const uint mantissa = magnitude & 7u;
    if (exponent == 0) {
        if (mantissa == 0) return sign;
        const uint power = mantissa >= 4 ? 2u : (mantissa >= 2 ? 1u : 0u);
        return sign | ((power + 6u) << 10) | ((mantissa << (10u - power)) & 0x3ffu);
    }
    return sign | ((exponent + 8u) << 10) | (mantissa << 7);
}

uint e4m3fn_from_half(uint h) {
    const uint sign = (h & 0x8000u) >> 8;
    const uint magnitude = h & 0x7fffu;
    if (magnitude >= 0x5f00u) return sign | 0x7eu;
    if (magnitude <= 0x1400u) return sign;
    const uint exponent = magnitude >> 10;
    const uint mantissa = magnitude & 0x3ffu;
    const uint shift = magnitude < 0x2400u ? 16u - exponent : 7u;
    const uint value = magnitude < 0x2400u ? (mantissa | 0x400u) : mantissa;
    const uint base = value >> shift;
    const uint remainder = value & ((1u << shift) - 1u);
    const uint halfway = 1u << (shift - 1u);
    const uint rounded = base + ((remainder > halfway || (remainder == halfway && (base & 1u))) ? 1u : 0u);
    return sign | (magnitude < 0x2400u ? rounded : ((exponent - 8u) << 3) + rounded);
}

