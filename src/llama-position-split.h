#pragma once

// Position-split KV placement contracts (Fase 0).
//
// Plano: docs/occupancy-placement-plan-20261002.md §§3.1-3.8, 4.1.
// Um perfil de contexto em que, para todas as camadas full-attention,
// posições [0,P) ficam na CUDA (KVarN4) e [P,C) na Vulkan0 (Q4 overflow).
// A atenção por camada é merge(FA_cuda, FA_vulkan) por log-sum-exp.
// C = 204800 é o teto do contrato (kCapacityC); instâncias menores, como
// C = 131072, continuam válidas — o runtime dimensiona as faixas [0,P) e
// [P,C) pelo n_ctx efetivo do perfil, não pela constante.
//
// Este header contém apenas contratos CPU/testáveis, sem dependência de
// backend: escolha de P, validação de faixas, fronteira de ubatch, contrato
// O/LSE e protótipo de merge. Integração com cache/grafo/scheduler vem nas
// Fases 1-3; snapshots v2 na Fase 4.
//
// Convenções:
//   - P múltiplo de 256 e do grupo KVarN de 128 tokens; C = 204800 (teto).
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

// llama_pos is int32_t in the public API (include/llama.h). The alias is local
// to this backend-free contract header so it compiles standalone; an identical
// typedef in a wider scope is legal when both are visible.
using llama_pos = int32_t;

inline constexpr uint32_t kCapacityC        = 204800;
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
    uint8_t  backend  = 0; // 0 = CUDA, 1 = overflow companion (Vulkan or RPC host)
    uint8_t  format   = 0; // 0 = KVarN4, 1 = Q4_0, 2 = Q8_0 (logical contract only)
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
        if (ovf->backend != 1 || (ovf->format != 1 && ovf->format != 2)) {
            return "faixa overflow deve ser um companion Q4_0 ou Q8_0";
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

// ---------------------------------------------------------------------------
// MTP especulativo sobre o split por posição (peça (d), fase 1)
// ---------------------------------------------------------------------------
//
// O ubatch de verify é o token âncora mais até `n_draft` tokens de draft. Com
// n_draft <= n_rs_seq esse ubatch é exatamente a janela protegida (1 +
// n_rs_seq), que divide_ubatches_at_p rejeita quando atravessa P. O fallback
// determinístico da fase 1 é nunca gerar draft cujo verify atravesse P: o
// passo que cruzaria decodifica só a âncora (não-especulativo) e o draft
// volta quando a própria âncora já está em [P, C). Só a string exata "1"
// habilita; ausente/qualquer outro valor mantém o guard fail-closed.
inline bool mtp_position_split_env_enabled(const char * value) {
    return value != nullptr && value[0] == '1' && value[1] == '\0';
}

// The LLAMA_KV_POSITION_SPLIT_P env fallback applies only to the plain target
// context. The MTP draft context (and auxiliary/draft contexts that carry
// ctx_other) has its P deliberately zeroed by
// common_base_params_to_speculative; resurrecting it from the env there makes
// llama_model::create_memory reject the context
// ("position-split KV não suporta contexto MTP", src/llama-model.cpp:2861).
inline bool position_split_env_p_fallback_applies(bool is_auxiliary_context) {
    return !is_auxiliary_context;
}

// Limite de draft para a âncora em `anchor_pos` com fronteira P: o verify
// [anchor_pos, anchor_pos + n_draft] não pode cruzar P, logo
// n_draft <= P - anchor_pos - 1 quando a âncora está abaixo de P. Âncora
// negativa (não deveria ocorrer) falha fechada em 0; âncora >= P não é
// limitada. P == 0 desliga o limite (modo atual).
inline int64_t spec_draft_limit_at_p(int64_t anchor_pos, uint32_t p) {
    if (p == 0) {
        return std::numeric_limits<int64_t>::max();
    }
    if (anchor_pos < 0) {
        return 0;
    }
    if (uint64_t(anchor_pos) >= p) {
        return std::numeric_limits<int64_t>::max();
    }
    return int64_t(p) - anchor_pos - 1;
}

// Atribuição de faixa por posição (Fase 2b): cada posição absoluta pertence
// a exatamente uma faixa. Local [0,P) vive na CUDA (KVarN4); overflow [P,C)
// na Vulkan0 (Q4). `index_in_range` é o índice dentro da faixa (pos no local,
// pos-P no overflow), pronto para indexar células da faixa. Posições fora de
// [0,C), ou qualquer P inválido (incluindo P == 0, modo desligado), resultam
// em INVALID: fail-closed, nunca adivinhar faixa.
enum class range_kind : uint8_t {
    LOCAL    = 0,
    OVERFLOW = 1,
    INVALID  = 2,
};

struct range_pos {
    range_kind kind = range_kind::INVALID;
    uint32_t index_in_range = 0;
};

inline range_pos range_for_pos(int64_t pos, uint32_t p, uint32_t c = kCapacityC) {
    range_pos out;
    if (!is_valid_p(p, c) || pos < 0 || uint64_t(pos) >= c) {
        return out;
    }
    if (uint64_t(pos) < p) {
        out.kind = range_kind::LOCAL;
        out.index_in_range = uint32_t(pos);
    } else {
        out.kind = range_kind::OVERFLOW;
        out.index_in_range = uint32_t(uint64_t(pos) - p);
    }
    return out;
}

// Divide um intervalo contíguo [pos_start, pos_start + n_tokens) nas contagens
// por faixa, para occupancy/seq_rm/rollback (§4.1: P-129, P-128, P-1, P, P+1,
// P+128). Falha fechada (false + motivo) em intervalo vazio, início negativo
// ou qualquer token além de C. Um intervalo que não cruza P deixa a outra
// contagem em zero; isso não é erro.
inline bool split_span_at_p(int64_t pos_start, uint64_t n_tokens, uint32_t p, uint32_t c,
                            uint32_t & local_count, uint32_t & overflow_count,
                            std::string & error) {
    local_count = 0;
    overflow_count = 0;
    error.clear();
    if (!is_valid_p(p, c)) {
        error = "P inválido (exige múltiplo de 256, 0 < P < C)";
        return false;
    }
    if (n_tokens == 0) {
        error = "intervalo vazio não tem faixa";
        return false;
    }
    if (pos_start < 0) {
        error = "posição inicial negativa";
        return false;
    }
    const uint64_t end = uint64_t(pos_start) + n_tokens;
    if (end > c) {
        error = "intervalo excede a capacidade C";
        return false;
    }
    const uint64_t local_end = end < p ? end : p;
    if (local_end > uint64_t(pos_start)) {
        local_count = uint32_t(local_end - uint64_t(pos_start));
    }
    const uint64_t ovf_start = uint64_t(pos_start) > p ? uint64_t(pos_start) : p;
    if (end > ovf_start) {
        overflow_count = uint32_t(end - ovf_start);
    }
    if (uint64_t(local_count) + uint64_t(overflow_count) != n_tokens) {
        error = "partição interna inconsistente";
        return false;
    }
    return true;
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

// Classificação de remoção/rollback por faixa (§§3.1-3.2, Fase 2b).
// Uma remoção [p0,p1) toca no máximo as duas faixas; o chamador (cache)
// decide representabilidade KVarN/checkpoint a partir daqui, mas a
// classificação em si é total e fail-closed: intervalo vazio, invertido ou
// além de C nunca produz um plano parcial.
enum class rm_span_kind : uint8_t {
    LOCAL_ONLY    = 0, // [p0,p1) ⊆ [0,P)
    OVERFLOW_ONLY = 1, // [p0,p1) ⊆ [P,C)
    CROSSING      = 2, // p0 < P <= p1: overflow inteiro + sufixo local
    INVALID       = 3,
};

// A remoção do llama.cpp usa p1 negativo como "até o fim do contexto".
// Normalizar esse sentinela para C antes da classificação/planejamento; p1
// negativo nunca significa uma posição negativa real aqui.
inline int64_t normalize_removal_end(int64_t p1, uint32_t c) {
    return p1 < 0 ? int64_t(c) : p1;
}

inline rm_span_kind classify_removal(int64_t p0, int64_t p1, uint32_t p, uint32_t c = kCapacityC) {
    const int64_t end = normalize_removal_end(p1, c);
    if (!is_valid_p(p, c) || end <= p0 || p0 < 0 || uint64_t(end) > c) {
        return rm_span_kind::INVALID;
    }
    if (uint64_t(end) <= p) {
        return rm_span_kind::LOCAL_ONLY;
    }
    if (uint64_t(p0) >= p) {
        return rm_span_kind::OVERFLOW_ONLY;
    }
    return rm_span_kind::CROSSING;
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

// ---------------------------------------------------------------------------
// Remoção/rollback transacional (Fase 2b, §§3.1-3.2 e 4.1)
// ---------------------------------------------------------------------------

// Uma remoção [p0,p1) é convertida em passos por faixa antes de qualquer
// mutação: o chamador valida o plano inteiro e só então aplica. Nada é
// escrito durante o planejamento, e um plano inválido não deixa estado
// parcial para o chamarista desfazer.
enum class removal_disposition : uint8_t {
    APPLY            = 0, // representável in-place nas faixas envolvidas
    NEEDS_CHECKPOINT = 1, // fora da reserva exata: exige checkpoint/replay
    REJECT           = 2, // intervalo ou P inválido
};

// Passo de remoção dentro de uma faixa: `start_index`/`n_tokens` são índices
// intra-faixa (posição - P no overflow), nunca absolutos.
struct range_removal_step {
    uint32_t range_id     = kRangeLocal;
    uint32_t start_index  = 0;
    uint32_t n_tokens     = 0;
    // true quando o passo remove um sufixo da faixa (exige truncamento
    // monotônico do anel); false quando remove do meio (replay).
    bool suffix = false;
};

struct removal_plan {
    std::vector<range_removal_step> steps;
    rm_span_kind shape = rm_span_kind::INVALID;
};

// Planeja a remoção [p0,p1) nas faixas de um contexto (C) com fronteira P.
// p1 negativo significa C, como na API de seq_rm do llama.cpp.
// `exact_reserve` é o sufixo representável in-place por faixa (tail/stage
// exatos); fora dele o plano exige checkpoint/replay. Retorna APPLY,
// NEEDS_CHECKPOINT ou REJECT e, em caso de sucesso, o plano completo.
inline removal_disposition plan_removal(
        int64_t p0, int64_t p1, uint32_t p, uint32_t c,
        uint32_t exact_reserve, uint32_t & exact_reserve_clamped,
        removal_plan & out) {
    out.steps.clear();
    out.shape = rm_span_kind::INVALID;
    if (!is_valid_p(p, c)) {
        return removal_disposition::REJECT;
    }
    exact_reserve_clamped = exact_reserve > c ? c : exact_reserve;
    const rm_span_kind shape = classify_removal(p0, p1, p, c);
    if (shape == rm_span_kind::INVALID) {
        return removal_disposition::REJECT;
    }
    out.shape = shape;

    // Divisão canônica do intervalo nas duas faixas.
    const uint64_t end = uint64_t(normalize_removal_end(p1, c));
    const uint64_t local_end = end < p ? end : p;
    const uint64_t ovf_start = uint64_t(p0) > p ? uint64_t(p0) : p;
    bool needs_checkpoint = false;

    if (local_end > uint64_t(p0)) {
        range_removal_step step;
        step.range_id    = kRangeLocal;
        step.start_index = uint32_t(p0);
        step.n_tokens    = uint32_t(local_end - uint64_t(p0));
        step.suffix      = local_end == p; // termina em P => sufixo da faixa local
        out.steps.push_back(step);
    }
    if (end > ovf_start) {
        range_removal_step step;
        step.range_id    = kRangeOverflow;
        step.start_index = uint32_t(ovf_start - p);
        step.n_tokens    = uint32_t(end - ovf_start);
        step.suffix      = end == c; // termina em C => sufixo da faixa overflow
        out.steps.push_back(step);
    }

    // Representabilidade: um passo é in-place apenas se cabe inteiro na
    // reserva exata da sua faixa; senão depende de checkpoint/replay.
    for (const auto & step : out.steps) {
        const uint32_t reserve = exact_reserve_clamped;
        const uint32_t need = step.suffix ? step.n_tokens : step.start_index + step.n_tokens;
        if (need > reserve) {
            needs_checkpoint = true;
        }
    }
    return needs_checkpoint ? removal_disposition::NEEDS_CHECKPOINT
                            : removal_disposition::APPLY;
}


// Tail extent for a graph (plan 3.1/3.4). The local tail carries the
// in-flight rows ONLY when this graph's ubatch writes the local range; a
// position-split OVERFLOW ubatch stores Q4 overflow keys, so its local tail is
// read purely as history and must not be sized with its query count (that would
// promise q_max current rows the graph deliberately does not bind and trip
// fattn-tail.cuh's mt->ne[0] <= kt->ne[1] + kt_current->ne[1]).
//
// Returns the query count to feed llama_kv_cache::get_tail_attention_stride():
//   - P-mode off            -> q_max (unchanged legacy behaviour)
//   - empty ubatch          -> q_max (nothing to classify)
//   - any token >= P        -> 1 (configured persistent-tail extent only)
//   - all tokens < P        -> number of local tokens
inline int64_t tail_extent_queries(
        bool        position_split_enabled,
        uint32_t    p,
        const llama_pos * pos,
        uint32_t    n_tokens,
        int64_t     q_max) {
    if (!position_split_enabled || p == 0 || pos == nullptr || n_tokens == 0) {
        return q_max;
    }
    int64_t local = 0;
    for (uint32_t i = 0; i < n_tokens; ++i) {
        if (pos[i] >= llama_pos(p)) {
            return 1;
        }
        ++local;
    }
    return local;
}

} // namespace llama_position_split
