#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

// CPU float32 model of the shader. Build without fast-math or contraction.
struct Split { float maximum, den; std::vector<float> v; };
struct Result { std::vector<float> out; float lse; };
static float bits_float(uint32_t x) { float f; std::memcpy(&f, &x, 4); return f; }
static uint32_t bits(float f) { uint32_t x; std::memcpy(&x, &f, 4); return x; }
static constexpr uint32_t qnan = 0x7fc00000u;

static Result reduce(const std::vector<Split>& p, unsigned dim, bool paired) {
    unsigned splits = std::min<unsigned>(p.size(), 256);
    bool invalid = p.size() > 256;
    float maximum = -3.402823466e38f;
    for (unsigned s = 0; s < splits; ++s) {
        float den = p[s].den;
        if (den == 0) continue;
        if (!std::isfinite(den) || den < 0 || !std::isfinite(p[s].maximum)) invalid = true;
        else maximum = std::max(maximum, p[s].maximum);
    }
    Result r{std::vector<float>(dim), 0};
    auto poison = [&] { std::fill(r.out.begin(), r.out.end(), bits_float(qnan)); r.lse = bits_float(qnan); };
    if (invalid) { poison(); return r; }
    std::vector<float> w(splits);
    for (unsigned s = 0; s < splits; ++s)
        w[s] = p[s].den != 0 ? std::exp(p[s].maximum - maximum) : 0;
    float sum = 0;
    for (unsigned s = 0; s < splits; ++s)
        if (p[s].den != 0) sum += p[s].den * w[s];
    float inv = sum == 0 ? 0 : 1 / sum;
    if (!std::isfinite(sum) || sum < 0 || !std::isfinite(inv)) invalid = true;
    r.lse = sum == 0 ? -std::numeric_limits<float>::infinity() : maximum + std::log(sum);
    if (sum != 0 && !std::isfinite(r.lse)) invalid = true;
    for (unsigned lane = 0; lane < 128; ++lane) {
        bool invalid_numerator = false;
        if (paired && dim == 256) {
            float a = 0, b = 0;
            for (unsigned s = 0; s < splits; ++s) if (p[s].den != 0) {
                float x = p[s].v[lane], y = p[s].v[lane + 128];
                invalid_numerator = invalid_numerator || !std::isfinite(x) || !std::isfinite(y);
                a += x * w[s];
                b += y * w[s];
            }
            invalid_numerator = invalid_numerator || !std::isfinite(a * inv) || !std::isfinite(b * inv);
            r.out[lane] = a * inv;
            r.out[lane + 128] = b * inv;
        } else {
            for (unsigned d = lane; d < dim; d += 128) {
                float a = 0;
                for (unsigned s = 0; s < splits; ++s) if (p[s].den != 0) {
                    float value = p[s].v[d];
                    invalid_numerator = invalid_numerator || !std::isfinite(value);
                    a += value * w[s];
                }
                invalid_numerator = invalid_numerator || !std::isfinite(a * inv);
                r.out[d] = a * inv;
            }
        }
        invalid = invalid || invalid_numerator;
    }
    if (invalid) poison();
    return r;
}
static void require(bool b, const std::string& msg) {
    if (!b) { std::cerr << "FAIL: " << msg << '\n'; std::exit(1); }
}
static void check(const std::vector<Split>& p, unsigned dim, bool poison_expected, const std::string& name) {
    Result a = reduce(p, dim, false), b = reduce(p, dim, true);
    require(bits(a.lse) == bits(b.lse), name + " lse bit mismatch");
    for (unsigned d = 0; d < dim; ++d) {
        require(bits(a.out[d]) == bits(b.out[d]), name + " output bit mismatch at " + std::to_string(d));
        if (poison_expected) require(bits(a.out[d]) == qnan, name + " failed to poison dimension " + std::to_string(d));
        else require(std::isfinite(a.out[d]), name + " unexpected nonfinite output");
    }
    if (poison_expected) require(bits(a.lse) == qnan, name + " failed to poison lse");
    else require(std::isfinite(a.lse) || (std::isinf(a.lse) && a.lse < 0), name + " unexpected nonfinite lse");
}
static Split split(unsigned dim, float den=1, float max=0, float v=1) { return {max, den, std::vector<float>(dim, v)}; }
int main() {
    static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559, "IEEE float32 required");
    std::mt19937 rng(0x726b7661);
    std::uniform_real_distribution<float> value(-5.f, 5.f), maxdist(-30.f, 30.f), den(0.01f, 5.f);
    unsigned checks = 0;
    for (unsigned dim : {64u, 128u, 255u, 256u, 257u}) {
        for (unsigned n : {0u, 1u, 2u, 127u, 128u, 129u, 256u}) {
            for (int trial=0; trial<4; ++trial) {
                std::vector<Split> p;
                for (unsigned s=0; s<n; ++s) {
                    Split x = split(dim, s%7 == 0 ? 0.f : den(rng), maxdist(rng), 0);
                    for (auto& v : x.v) v = value(rng);
                    p.push_back(x);
                }
                check(p, dim, false, "random dim="+std::to_string(dim)+" splits="+std::to_string(n)); ++checks;
            }
        }
    }
    // Independently known arithmetic: same maxima and denominator 1 yield
    // equal weights and the arithmetic mean for each dimension.
    auto known = std::vector<Split>{split(256, 1, 0, 1), split(256, 1, 0, -3)};
    known[0].v[128] = 5; known[1].v[128] = 1;
    check(known, 256, false, "known equal weights"); ++checks;
    Result known_result = reduce(known, 256, true);
    require(bits(known_result.out[0]) == bits(-1.f), "known first dimension");
    require(bits(known_result.out[128]) == bits(3.f), "known paired dimension");
    require(bits(known_result.lse) == bits(std::log(2.f)), "known lse");
    const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
    const float large = std::numeric_limits<float>::max();
    const float tiny = std::numeric_limits<float>::denorm_min();
    auto valid = std::vector<Split>{split(256), split(256)};
    auto p = valid;
    p[0].den = 0; p[0].maximum = nan; std::fill(p[0].v.begin(), p[0].v.end(), inf);
    check(p, 256, false, "empty split ignores invalid data"); ++checks;
    p = {split(256, 0, nan, inf), split(256, -0.f, inf, nan)};
    check(p, 256, false, "all empty"); ++checks;
    Result empty = reduce(p, 256, true);
    require(bits(empty.lse) == 0xff800000u, "empty lse is negative infinity");
    for (float v: empty.out) require(bits(v) == 0u, "empty output is positive zero");
    for (float bad : {nan, inf, -inf, -1.f}) {
        p = valid; p[1].den = bad; check(p, 256, true, "invalid denominator"); ++checks;
    }
    for (float bad : {nan, inf, -inf}) {
        p = valid; p[0].maximum = bad; check(p, 256, true, "invalid maximum"); ++checks;
        for (unsigned d : {0u, 1u, 127u, 128u, 129u, 255u}) {
            p = valid; p[0].v[d] = bad; check(p, 256, true, "invalid numerator"); ++checks;
            p = {split(256, 1, 0), split(256, 1, -1000)}; p[1].v[d] = bad;
            check(p, 256, true, "invalid numerator with underflowed weight"); ++checks;
        }
    }
    p = {split(256, large), split(256, large)}; check(p, 256, true, "sum overflow"); ++checks;
    p = {split(256, tiny)}; check(p, 256, true, "inverse overflow"); ++checks;
    p = {split(256, 1e-38f, 0, 4)}; check(p, 256, true, "normalized numerator overflow"); ++checks;
    p = {split(256, 1, 0, large), split(256, 1, 0, large)}; check(p, 256, true, "numerator overflow"); ++checks;
    p = {split(256, 2, large)}; check(p, 256, false, "extreme finite maximum"); ++checks;
    p = std::vector<Split>(257, split(256)); check(p, 256, true, "too many splits"); ++checks;
    std::cout << "PASS: " << checks << " CPU parity and invalid-propagation cases\n";
}
