#pragma once

// Position-split KV placement contracts (Fase 0).
//
// Plano: docs/occupancy-placement-plan-20261002.md §§3.1-3.8, 4.1.
// Um único perfil C=131072 em que, para todas as camadas full-attention,
// posições [0,P) ficam na CUDA (KVarN4) e [P,C) na Vulkan0 (Q4 overflow).
// A atenção por camada é merge(FA_cuda, FA_vulkan) por log-sum-exp.
//
// Este header contém apenas contratos CPU/testáveis, sem dependência de
// backend: escolha de P, validação de faixas, fronteira de ubatch, contrato
// O/LSE e protótipo de merge. Integração com cache/grafo/scheduler vem nas
// Fases 1-3; snapshots v2 na Fase 4.
//
// Convenções:
//   - P múltiplo de 256 e do grupo KVarN de 128 tokens; C = 131072.
//   - LSE = log(sum(exp(score))) em ln, após scale/máscara/bias/softcap.
//   - Faixa vazia ou query inteiramente mascarada: O = 0, LSE = -inf.
//   - Merge ocorre na CUDA no produto; este protótipo é a referência.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace llama_position_split {

inline constexpr uint32_t kCapacityC        = 131072;
inline constexpr uint32_t kTargetP          = 102400;
inline constexpr uint32_t kAlignTokens       = 256;
inline constexpr uint32_t kKvarnGroup       = 128;
inline constexpr uint32_t kMaxLayers        = 4096;

// Envelope misto v2 (proposta Fase 0; wire format definitivo na Fase 4).
// v1 existente usa magic "KMS1"/version 1 com um descritor por layer_id.
// v2 exige dois descritores por camada full: (layer_id, range_id) com
// [start,end) == [0,P) local e [P,C) overflow, sem lacunas/sobreposições.
inline constexpr uint32_t kMixedStateV1      = 1;
inline constexpr uint32_t kMixedStateV2      = 2;
inline constexpr uint32_t kRangeLocal        = 0;
inline constexpr uint32_t kRangeOverflow     = 1;

inline float lse_neg_inf() {
    return -std::numeric_limits<float>::infinity();
}

inline bool lse_is_neg_inf(float v) {
    return std::isinf(v) && v < 0.0f;
}

// Alinha P para baixo ao múltiplo de 256 (que também é múltiplo de 128).
inline uint32_t align_p_down(uint32_t p) {
    return (p / kAlignTokens) * kAlignTokens;
}

inline bool is_valid_p(uint32_t p, uint32_t c = kCapacityC) {
    if (p == 0 || p >= c) {
        return false;
    }
    if (p % kAlignTokens != 0) {
        return false;
    }
    if (p % kKvarnGroup != 0) {
        return false;
    }
    return true;
}

// Descritor de faixa (layer_id, range_id) para o envelope v2.
struct range_desc {
    uint32_t layer_id  = 0;
    uint32_t range_id  = 0; // 0 = local [0,P), 1 = overflow [P,C)
    uint32_t start     = 0; // inclusive
    uint32_t end       = 0; // exclusive
    uint8_t  backend  = 0; // 0 = CUDA, 1 = Vulkan0 (convenção deste header)
    uint8_t  format   = 0; // 0 = KVarN4, 1 = Q4 (convenção deste header)
};

// Valida exatamente as faixas [0,P) e [P,C) para um conjunto de camadas.
// Retorna "" se válido, ou motivo explícito (fail-closed).
inline std::string validate_ranges(const std::vector<range_desc> & descs,
                                   uint32_t p, uint32_t c,
                                   const std::vector<uint32_t> & full_layers) {
    if (!is_valid_p(p, c)) {
        return "P inválido (exige múltiplo de 256, 0 < P < C)";
    }
    if (descs.size() != full_layers.size() * 2) {
        return "número de descritores != 2 por camada full";
    }
    for (uint32_t lid : full_layers) {
        const range_desc * local = nullptr;
        const range_desc * ovf   = nullptr;
        for (const auto & d : descs) {
            if (d.layer_id != lid) {
                continue;
            }
            if (d.range_id == kRangeLocal) {
                if (local) {
                    return "duplicata de faixa local";
                }
                local = &d;
            } else if (d.range_id == kRangeOverflow) {
                if (ovf) {
                    return "duplicata de faixa overflow";
                }
                ovf = &d;
            } else {
                return "range_id desconhecido";
            }
        }
        if (!local || !ovf) {
            return "faixa ausente (local e overflow obrigatórios)";
        }
        if (local->start != 0 || local->end != p) {
            return "faixa local deve ser [0,P)";
        }
        if (ovf->start != p || ovf->end != c) {
            return "faixa overflow deve ser [P,C)";
        }
        if (local->backend != 0 || local->format != 0) {
            return "faixa local deve ser CUDA/KVarN4";
        }
        if (ovf->backend != 1 || ovf->format != 1) {
            return "faixa overflow deve ser Vulkan0/Q4";
        }
        if (local->end != ovf->start) {
            return "lacuna/sobreposição na fronteira P";
        }
    }
    return "";
}

// Fronteira de ubatch: um ubatch nunca atravessa P (§3.2).
// pos_start = posição absoluta do primeiro token do ubatch.
inline bool ubatch_crosses_p(uint32_t pos_start, uint32_t n_tokens, uint32_t p) {
    if (n_tokens == 0) {
        return false;
    }
    const uint64_t end = uint64_t(pos_start) + uint64_t(n_tokens);
    return pos_start < p && end > p;
}

// Validação da janela protegida recorrente (1 + n_rs_seq deve ficar junta).
// Se a janela [win_start, win_end) atravessa P, a versão inicial rejeita o
// prepare antes de qualquer mutação (retorna motivo; não divide a janela).
inline std::string validate_recurrent_window(uint32_t win_start, uint32_t win_end, uint32_t p) {
    if (win_end <= win_start) {
        return "";
    }
    if (win_start < p && win_end > p) {
        return "janela recorrente atravessa P: rejeitar prepare sem mutação";
    }
    return "";
}

// Conversão (m,l) -> LSE: lse = m + log(l), com l = denom (>= 0).
// Se l <= 0 ou não-finito, a query é vazia/mascarada: LSE = -inf.
inline float lse_from_ml(float m, float denom) {
    if (!(denom > 0.0f) || !std::isfinite(m) || !std::isfinite(denom)) {
        return lse_neg_inf();
    }
    return m + std::log(denom);
}

// Protótipo CPU do merge por log-sum-exp (§3.3):
//   M = max(lse1, lse2)
//   o = (o1*e^{lse1-M} + o2*e^{lse2-M}) / (e^{lse1-M} + e^{lse2-M})
// Regras de faixa vazia (§3.4):
//   - ambas vazias (-inf): O = 0.
//   - só uma válida: retorna a saída válida.
//   - nenhuma saída válida pode conter NaN/Inf (checagem do chamador).
inline std::vector<float> merge_o_lse(const std::vector<float> & o1, float lse1,
                                      const std::vector<float> & o2, float lse2) {
    const size_t d = o1.size() > o2.size() ? o1.size() : o2.size();
    std::vector<float> out(d, 0.0f);
    const bool v1 = !lse_is_neg_inf(lse1);
    const bool v2 = !lse_is_neg_inf(lse2);
    if (!v1 && !v2) {
        return out; // O = 0
    }
    if (v1 && !v2) {
        for (size_t i = 0; i < d; ++i) {
            out[i] = i < o1.size() ? o1[i] : 0.0f;
        }
        return out;
    }
    if (v2 && !v1) {
        for (size_t i = 0; i < d; ++i) {
            out[i] = i < o2.size() ? o2[i] : 0.0f;
        }
        return out;
    }
    const float m = lse1 > lse2 ? lse1 : lse2;
    const float w1 = std::exp(lse1 - m);
    const float w2 = std::exp(lse2 - m);
    const float denom = w1 + w2;
    for (size_t i = 0; i < d; ++i) {
        const float a = i < o1.size() ? o1[i] : 0.0f;
        const float b = i < o2.size() ? o2[i] : 0.0f;
        out[i] = (a * w1 + b * w2) / denom;
    }
    return out;
}

inline float merge_lse(float lse1, float lse2) {
    const bool v1 = !lse_is_neg_inf(lse1);
    const bool v2 = !lse_is_neg_inf(lse2);
    if (!v1) {
        return lse2;
    }
    if (!v2) {
        return lse1;
    }
    const float m = lse1 > lse2 ? lse1 : lse2;
    return m + std::log(std::exp(lse1 - m) + std::exp(lse2 - m));
}

// NRMSE = RMS(O - ref) / max(RMS(ref), 1e-6). Gate §4.1: <= 1e-3.
inline double nrmse(const std::vector<float> & o, const std::vector<float> & ref) {
    if (o.size() != ref.size() || o.empty()) {
        return std::numeric_limits<double>::infinity();
    }
    double se = 0.0;
    double sr = 0.0;
    for (size_t i = 0; i < o.size(); ++i) {
        const double d = double(o[i]) - double(ref[i]);
        se += d * d;
        sr += double(ref[i]) * double(ref[i]);
    }
    const double rms_e = std::sqrt(se / double(o.size()));
    double rms_r = std::sqrt(sr / double(ref.size()));
    if (rms_r < 1e-6) {
        rms_r = 1e-6;
    }
    return rms_e / rms_r;
}

// Escolha de P (§3.8): maior P múltiplo de 256 até 102400 tal que
// 16 * local_bytes(P) + reservas caiba na CUDA, com margem
// max(512 MiB, 5% VRAM). Versão CPU/testável; a integração real mede o pico
// com grafos reservados de prefill/decode, restore e passagem por P.
inline uint32_t choose_p(uint64_t cuda_free_bytes, uint64_t local_bytes_per_layer_at_target,
                         uint32_t n_full_layers, uint64_t reserve_bytes,
                         uint64_t vram_total_bytes, uint32_t p_max = kTargetP) {
    const uint64_t margin = vram_total_bytes / 20 > 512ull * 1024ull * 1024ull
        ? vram_total_bytes / 20
        : 512ull * 1024ull * 1024ull;
    uint32_t p = align_p_down(p_max);
    const double bytes_per_token_layer =
        double(local_bytes_per_layer_at_target) / double(p_max == 0 ? 1 : p_max);
    while (p >= kAlignTokens) {
        const uint64_t local_total =
            uint64_t(double(n_full_layers) * bytes_per_token_layer * double(p));
        if (local_total + reserve_bytes + margin <= cuda_free_bytes) {
            return p;
        }
        if (p == kAlignTokens) {
            break;
        }
        p -= kAlignTokens;
    }
    return 0; // nenhum P suportado: rejeitar configuração
}

} // namespace llama_position_split
