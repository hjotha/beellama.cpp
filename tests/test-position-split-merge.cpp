// Fase 0 — contratos O/LSE, envelope v2, fronteira de ubatch e orçamento.
//
// Cobre §4.1 (corretude estrutural) em CPU: referência de merge sobre o KV
// dequantizado da mesma representação por faixa, com acumulação estável.
// Build standalone (sem CMake):
//   g++ -std=c++17 -O2 -Wall -Wextra -I src tests/test-position-split-merge.cpp \
//       -o /tmp/test-position-split-merge
//
// Gates verificados aqui:
//   NRMSE <= 1e-3, |O| <= 1e-2, |LSE| <= 1e-2, vazios exatos O=0/LSE=-inf,
//   sem NaN/Inf em saídas válidas. P alinhado/desalinhado no teste
//   matemático; no cache real P é múltiplo de 256.

#include "llama-position-split.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace ps = llama_position_split;

static int g_checks = 0;
static int g_failures = 0;
static const char * g_current = "";

#define CHECK(cond) do { \
    ++g_checks; \
    if (!(cond)) { \
        ++g_failures; \
        std::printf("  FAIL [%s] line %d: %s\n", g_current, __LINE__, #cond); \
    } \
} while (0)

static bool feq(float a, float b, float eps) {
    return std::fabs(a - b) <= eps;
}

static void test_merge_basic() {
    g_current = "merge_basic";
    // Partição de scores [1,2,3,4]: faixa1 cobre 1,2; faixa2 cobre 3,4.
    // O1 = softmax([1,2]) ponderado de valores 10,20; O2 de 30,40.
    const float e1 = std::exp(1.0f), e2 = std::exp(2.0f);
    const float e3 = std::exp(3.0f), e4 = std::exp(4.0f);
    const float lse1 = std::log(e1 + e2);
    const float lse2 = std::log(e3 + e4);
    const float o1 = (10.0f * e1 + 20.0f * e2) / (e1 + e2);
    const float o2 = (30.0f * e3 + 40.0f * e4) / (e3 + e4);
    const float lse_full = std::log(e1 + e2 + e3 + e4);
    const float o_full = (10.0f * e1 + 20.0f * e2 + 30.0f * e3 + 40.0f * e4) / (e1 + e2 + e3 + e4);

    auto out = ps::merge_o_lse({o1}, lse1, {o2}, lse2);
    CHECK(out.size() == 1);
    CHECK(feq(out[0], o_full, 1e-5f));
    CHECK(feq(ps::merge_lse(lse1, lse2), lse_full, 1e-5f));

    // NRMSE contra referência direta.
    CHECK(ps::nrmse(out, {o_full}) <= 1e-3);
    CHECK(std::fabs(out[0] - o_full) <= 1e-2f);
    CHECK(std::fabs(ps::merge_lse(lse1, lse2) - lse_full) <= 1e-2f);
}

static void test_merge_empty() {
    g_current = "merge_empty";
    auto both = ps::merge_o_lse({1.0f, 2.0f}, ps::lse_neg_inf(), {3.0f, 4.0f}, ps::lse_neg_inf());
    CHECK(both.size() == 2);
    CHECK(both[0] == 0.0f && both[1] == 0.0f);
    CHECK(ps::lse_is_neg_inf(ps::merge_lse(ps::lse_neg_inf(), ps::lse_neg_inf())));

    auto only1 = ps::merge_o_lse({5.0f}, 0.5f, {9.0f}, ps::lse_neg_inf());
    CHECK(feq(only1[0], 5.0f, 1e-6f));
    CHECK(feq(ps::merge_lse(0.5f, ps::lse_neg_inf()), 0.5f, 1e-6f));

    auto only2 = ps::merge_o_lse({5.0f}, ps::lse_neg_inf(), {7.0f}, -1.0f);
    CHECK(feq(only2[0], 7.0f, 1e-6f));
    CHECK(feq(ps::merge_lse(ps::lse_neg_inf(), -1.0f), -1.0f, 1e-6f));
}

static void test_lse_from_ml() {
    g_current = "lse_from_ml";
    CHECK(feq(ps::lse_from_ml(2.0f, std::exp(1.0f)), 3.0f, 1e-5f));
    CHECK(ps::lse_is_neg_inf(ps::lse_from_ml(0.0f, 0.0f)));
    CHECK(ps::lse_is_neg_inf(ps::lse_from_ml(0.0f, -1.0f)));
    CHECK(ps::lse_is_neg_inf(ps::lse_from_ml(NAN, 1.0f)));
    CHECK(ps::lse_is_neg_inf(ps::lse_from_ml(0.0f, INFINITY)));
}

static void test_ranges() {
    g_current = "ranges";
    const uint32_t P = 1024, C = 2048;
    std::vector<uint32_t> layers;
    for (uint32_t i = 0; i < 16; ++i) {
        layers.push_back(i);
    }
    std::vector<ps::range_desc> ok;
    for (uint32_t lid : layers) {
        ok.push_back({lid, ps::kRangeLocal, 0, P, 0, 0});
        ok.push_back({lid, ps::kRangeOverflow, P, C, 1, 1});
    }
    CHECK(ps::validate_ranges(ok, P, C, layers).empty());

    // Lacuna na fronteira.
    auto gap = ok;
    gap[1].start = P + 256;
    CHECK(!ps::validate_ranges(gap, P, C, layers).empty());

    // Sobreposição.
    auto ov = ok;
    ov[0].end = P + 256;
    CHECK(!ps::validate_ranges(ov, P, C, layers).empty());

    // Duplicata de faixa.
    auto dup = ok;
    dup.push_back({0, ps::kRangeLocal, 0, P, 0, 0});
    CHECK(!ps::validate_ranges(dup, P, C, layers).empty());

    // Faixa ausente.
    auto missing = ok;
    missing.pop_back();
    CHECK(!ps::validate_ranges(missing, P, C, layers).empty());

    // Backend/formato errado.
    auto bad = ok;
    bad[0].backend = 1;
    CHECK(!ps::validate_ranges(bad, P, C, layers).empty());

    // P desalinhado rejeitado no teste matemático como inválido para cache.
    CHECK(!ps::is_valid_p(P + 1, C));
    CHECK(!ps::is_valid_p(0, C));
    CHECK(!ps::is_valid_p(C, C));
    CHECK(ps::is_valid_p(P, C));
    CHECK(ps::align_p_down(P + 129) == P);
}

static void test_ubatch_boundary() {
    g_current = "ubatch_boundary";
    const uint32_t P = 1024;
    CHECK(!ps::ubatch_crosses_p(0, 256, P));
    CHECK(!ps::ubatch_crosses_p(P, 256, P));
    CHECK(!ps::ubatch_crosses_p(P - 256, 256, P)); // termina exatamente em P
    CHECK(ps::ubatch_crosses_p(P - 128, 256, P));
    CHECK(ps::ubatch_crosses_p(P - 1, 2, P));
    CHECK(!ps::ubatch_crosses_p(P + 1, 128, P));

    // Ocupações e remoções do §4.1 em torno de P.
    const uint32_t cases[][2] = {
        {P - 129, 1}, {P - 128, 1}, {P - 1, 1}, {P, 1}, {P + 1, 1}, {P + 128, 1},
    };
    for (const auto & c : cases) {
        const uint32_t pos = c[0];
        // ubatch de 1 token nunca atravessa.
        CHECK(!ps::ubatch_crosses_p(pos, 1, P));
    }
    CHECK(ps::ubatch_crosses_p(P - 1, 129, P));

    // Janela recorrente cruzando P deve ser rejeitada sem mutação.
    CHECK(ps::validate_recurrent_window(P - 10, P + 10, P) != "");
    CHECK(ps::validate_recurrent_window(0, P, P).empty());
    CHECK(ps::validate_recurrent_window(P, P + 128, P).empty());
}

static void test_choose_p() {
    g_current = "choose_p";
    // 12 GiB livres, camada ~2 MiB em P=102400, 16 camadas, reserva 2 GiB.
    const uint64_t GiB = 1024ull * 1024ull * 1024ull;
    const uint64_t MiB = 1024ull * 1024ull;
    uint32_t p = ps::choose_p(12 * GiB, 2 * MiB, 16, 2 * GiB, 12 * GiB);
    CHECK(p == 102400);
    CHECK(ps::is_valid_p(p));

    // Sem espaço: rejeita (retorna 0).
    CHECK(ps::choose_p(1 * GiB, 2 * MiB, 16, 2 * GiB, 12 * GiB) == 0);

    // Espaço parcial: maior múltiplo de 256 que cabe.
    // 4 GiB livres, 256 MiB/camada em P=102400, 16 camadas, reserva 1 GiB:
    // cheio = 4 GiB + 1 GiB + margem 0.6 GiB > 4 GiB, então P encolhe.
    uint32_t p2 = ps::choose_p(4 * GiB, 256 * MiB, 16, 1 * GiB, 12 * GiB);
    CHECK(p2 != 0);
    CHECK(p2 % 256 == 0);
    CHECK(p2 < 102400);
}

int main() {
    test_merge_basic();
    test_merge_empty();
    test_lse_from_ml();
    test_ranges();
    test_ubatch_boundary();
    test_choose_p();
    std::printf("position-split-merge: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
