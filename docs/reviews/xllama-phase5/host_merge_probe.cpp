#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <initializer_list>
#include <utility>

static void run(float a, float b, bool fixed) {
    ggml_context * ctx = ggml_init({1u << 20, nullptr, false});
    if (!ctx) std::exit(2);
    auto scalar = [&](float value) {
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_set_f32(t, value); return t;
    };
    auto * l1 = scalar(a); auto * l2 = scalar(b);
    auto * o1 = scalar(1.0f); auto * o2 = scalar(3.0f);
    ggml_tensor * maximum;
    if (!fixed) {
        auto * c1 = ggml_clamp(ctx, l1, -1e6f, 1e30f);
        auto * c2 = ggml_clamp(ctx, l2, -1e6f, 1e30f);
        maximum = ggml_scale(ctx, ggml_add(ctx, ggml_add(ctx, c1, c2),
            ggml_abs(ctx, ggml_sub(ctx, c1, c2))), 0.5f);
    } else {
        // Both input LSE values are finite in the RPC admission contract.
        auto * choose1 = ggml_step(ctx, ggml_sub(ctx, l1, l2));
        auto * choose2 = ggml_scale_bias(ctx, choose1, -1.0f, 1.0f);
        maximum = ggml_add(ctx, ggml_mul(ctx, choose1, l1), ggml_mul(ctx, choose2, l2));
    }
    auto * w1 = ggml_exp(ctx, ggml_sub(ctx, l1, maximum));
    auto * w2 = ggml_exp(ctx, ggml_sub(ctx, l2, maximum));
    auto * numerator = ggml_add(ctx, ggml_mul(ctx, o1, w1), ggml_mul(ctx, o2, w2));
    auto * denominator = ggml_add(ctx, w1, w2);
    auto * result = ggml_div(ctx, numerator, denominator);
    auto * graph = ggml_new_graph(ctx); ggml_build_forward_expand(graph, result);
    if (ggml_graph_compute_with_ctx(ctx, graph, 1) != GGML_STATUS_SUCCESS) std::exit(3);
    float out = ggml_get_f32_1d(result, 0);
    double m = std::fmax(double(a), double(b));
    double wa = std::exp(double(a)-m), wb = std::exp(double(b)-m);
    double expected = (wa + 3*wb)/(wa+wb);
    std::printf("%s LSE=(%.9g,%.9g) max=%.9g weights=(%.9g,%.9g) OUT=%.9g expected=%.9g\n",
        fixed ? "fixed" : "current", a,b,ggml_get_f32_1d(maximum,0),ggml_get_f32_1d(w1,0),ggml_get_f32_1d(w2,0),out,expected);
    if (fixed && (!std::isfinite(out) || std::abs(double(out)-expected) > 1e-6)) std::exit(4);
    if (!fixed && a==b && std::abs(a)>1e6f && std::isfinite(out)) std::exit(5);
    ggml_free(ctx);
}
int main() {
    const float hi = std::numeric_limits<float>::max();
    for (auto pair : {std::pair<float,float>{0,0}, {-2097152,-2097152}, {-2097152,0}, {1e31f,1e31f}, {hi,-hi}, {-hi,hi}}) {
        run(pair.first,pair.second,false); run(pair.first,pair.second,true);
    }
}
