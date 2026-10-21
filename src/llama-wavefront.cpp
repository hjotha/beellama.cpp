#include "llama-wavefront.h"
#include "llama-memory-hybrid.h"
#include "llama-position-split.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <limits>
#include <stdexcept>

namespace llama_wavefront {

uint32_t parse_serving_rpc_batch(const char * value) {
    if (!value || std::strcmp(value, "1") == 0) return 1;
    if (std::strcmp(value, "4") == 0) return 4;
    throw std::invalid_argument("GGML_RKVA_WAVEFRONT_RPC_BATCH expects 1 or 4");
}

uint32_t parse_serving_tile_size(const char * value) {
    if (!value || std::strcmp(value, "0") == 0) return 0;
    if (std::strcmp(value, "64") == 0) return 64;
    if (std::strcmp(value, "128") == 0) return 128;
    throw std::invalid_argument("GGML_RKVA_WAVEFRONT_TILE expects 0, 64 or 128");
}

std::vector<serving_step> build_serving_schedule(uint32_t groups, uint32_t tiles, uint32_t rpc_batch) {
    std::vector<serving_step> steps;
    if (rpc_batch != 1 && rpc_batch != 4) throw std::invalid_argument("invalid serving RPC batch");
    if (!groups || !tiles) return steps;
    if (uint64_t(groups)*tiles > UINT32_MAX/2) throw std::invalid_argument("wavefront schedule too large");
    if (rpc_batch == 4) {
        // Conservative complete-cell diagonals: all preceding wave outputs
        // are available before a new cohort, and every cohort has distinct
        // remote layer IDs. No future Q is inferred from a later same-token layer.
        for (uint64_t wave = 0; wave < uint64_t(groups) + tiles - 1; ++wave) {
            std::vector<serving_step> ready;
            for (uint32_t g = 0; g < groups; ++g) {
                if (wave >= g && wave - g < tiles) {
                    ready.push_back({ serving_step::phase::PREPARE, g, uint32_t(wave - g) });
                }
            }
            for (size_t begin = 0; begin < ready.size(); begin += 4) {
                const size_t end = std::min(ready.size(), begin + 4);
                for (size_t j = begin; j < end; ++j) steps.push_back(ready[j]);
                for (size_t j = begin; j < end; ++j) {
                    steps.push_back({ serving_step::phase::COMPLETE, ready[j].group, ready[j].tile });
                }
            }
        }
        return steps;
    }

    std::vector<bool> prepared(size_t(groups)*tiles, false), complete(prepared.size(), false);
    std::deque<uint32_t> pending;
    uint32_t completed = 0;
    while (completed < prepared.size()) {
        bool added = false;
        if (pending.size() < 2) {
            // Prefer advancing a ready earlier tile to the next layer group.
            // A horizontal predecessor must have committed its recurrent state
            // and submitted its KV append; its remote OUT need not be ready.
            for (uint32_t g = groups; g-- > 0 && !added;) {
                for (uint32_t t = 0; t < tiles && !added; ++t) {
                    const uint32_t id = g*tiles + t;
                    if (!prepared[id] && (g == 0 || complete[id - tiles]) &&
                            (t == 0 || prepared[id - 1])) {
                        prepared[id] = true;
                        pending.push_back(id);
                        steps.push_back({ serving_step::phase::PREPARE, g, t });
                        added = true;
                    }
                }
            }
        }
        if (!added) {
            if (pending.empty()) throw std::logic_error("wavefront serving schedule stalled");
            const uint32_t id = pending.front(); pending.pop_front();
            complete[id] = true;
            ++completed;
            steps.push_back({ serving_step::phase::COMPLETE, id/tiles, id%tiles });
        }
    }
    return steps;
}

bool validate_tile_size(uint32_t tile_size) {
    if (tile_size < kMinTileSize || tile_size > kMaxTileSize) {
        return false;
    }
    // Exige potência de 2 (16, 32, 64, 128, 256, 512)
    return (tile_size & (tile_size - 1)) == 0;
}

wavefront_schedule build_schedule(uint32_t n_layers, uint32_t n_tiles) {
    wavefront_schedule sched;
    sched.n_layers = n_layers;
    sched.n_tiles  = n_tiles;

    if (n_layers == 0 || n_tiles == 0) {
        return sched;
    }

    sched.n_waves = n_layers + n_tiles - 1;
    sched.total_tasks = n_layers * n_tiles;
    sched.max_concurrency = std::min(n_layers, n_tiles);
    sched.waves.resize(sched.n_waves);

    for (uint32_t w = 0; w < sched.n_waves; ++w) {
        // Para uma onda w fixa: w = l + t => l = w - t
        // Como 0 <= t <= n_tiles - 1, temos: w - (n_tiles - 1) <= l <= w
        // E também 0 <= l <= n_layers - 1
        const int32_t l_min = std::max<int32_t>(0, (int32_t) w - (int32_t)(n_tiles - 1));
        const int32_t l_max = std::min<int32_t>((int32_t) n_layers - 1, (int32_t) w);

        for (int32_t l = l_min; l <= l_max; ++l) {
            const uint32_t t = uint32_t(w - l);

            wavefront_coord coord;
            coord.layer   = uint32_t(l);
            coord.tile    = t;
            coord.wave    = w;
            coord.task_id = uint32_t(l) * n_tiles + t;

            coord.dep_layer_prev = (l > 0) ? int32_t((l - 1) * n_tiles + t) : -1;
            coord.dep_tile_prev  = (t > 0) ? int32_t(l * n_tiles + (t - 1)) : -1;

            sched.waves[w].push_back(coord);
        }
    }

    return sched;
}

bool partition_ubatch_causal_tiles(
    const llama_ubatch & ubatch,
    uint32_t tile_size,
    uint32_t n_layers,
    uint32_t p,
    uint32_t n_keep_tail,
    wavefront_plan & plan,
    std::string & error) {

    plan = wavefront_plan{};
    error.clear();

    if (!validate_tile_size(tile_size)) {
        error = "tamanho de tile inválido (exige potência de 2 entre 16 e 512, tipicamente 64 ou 128)";
        return false;
    }

    if (ubatch.n_tokens == 0) {
        error = "ubatch vazio não possui tokens para particionar";
        return false;
    }

    if (ubatch.pos == nullptr || ubatch.n_pos < 1) {
        error = "ubatch sem posições temporais não pode ser particionado causalmente";
        return false;
    }

    if (ubatch.embd != nullptr) {
        error = "ubatch de embeddings não é suportado para prefill causal em wavefront; apenas tokens";
        return false;
    }

    if (ubatch.n_seqs != 1) {
        error = "apenas ubatch de sequência única é suportado para particionamento causal em wavefront";
        return false;
    }

    // Validação de posições temporais não negativas e estritamente crescentes (P1)
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.pos[i] < 0) {
            error = "posições temporais negativas não são suportadas";
            return false;
        }
        if (i > 0 && ubatch.pos[i] <= ubatch.pos[i - 1]) {
            error = "tokens no ubatch não estão ordenados monotonicamente de forma estritamente crescente";
            return false;
        }
    }

    // Validação de janela protegida recorrente (P1 Sol review)
    const uint32_t protected_tokens = std::min(n_keep_tail, ubatch.n_tokens);
    if (protected_tokens > tile_size) {
        error = "janela protegida recorrente excede o tamanho máximo do tile";
        return false;
    }
    const uint32_t rec_start = ubatch.n_tokens - protected_tokens;

    // Validação da fronteira P
    if (p > 0) {
        if (!llama_position_split::is_valid_p(p)) {
            error = "fronteira P inválida (exige múltiplo de 256, 0 < P < C)";
            return false;
        }

        // Validação da janela protegida recorrente contra P
        if (protected_tokens > 0) {
            const uint64_t win_begin = (uint64_t)ubatch.pos[rec_start];
            const uint64_t win_end   = (uint64_t)ubatch.pos[ubatch.n_tokens - 1] + 1;
            if (win_begin < uint64_t(p) && win_end > uint64_t(p)) {
                error = "janela protegida recorrente cruza P: rejeitar prepare sem mutação";
                return false;
            }
        }
    }

    // Invariante de particionamento causal:
    // 1. Cada tile tem no máximo tile_size tokens.
    // 2. Nenhum tile cruza P (divide exatamente em P).
    // 3. A janela protegida recorrente (últimos protected_tokens) fica indivisível no tile final.
    wavefront_plan local_plan;
    uint32_t curr = 0;

    while (curr < ubatch.n_tokens) {
        uint32_t cand = curr + std::min(tile_size, ubatch.n_tokens - curr);

        // Se o tile cruza P, divide exatamente em P
        if (p > 0 && ubatch.pos[curr] < llama_pos(p) && ubatch.pos[cand - 1] >= llama_pos(p)) {
            uint32_t split_p = curr;
            while (split_p < cand && ubatch.pos[split_p] < llama_pos(p)) {
                ++split_p;
            }
            cand = split_p;
        }

        // Se curr está antes da janela protegida, cand não pode cortar dentro da janela protegida.
        // Se cand já atinge o final do ubatch (cand == ubatch.n_tokens), a janela já está indivisível no tile final.
        if (protected_tokens > 0 && curr < rec_start && cand > rec_start && cand < ubatch.n_tokens) {
            cand = rec_start;
        }

        if (cand <= curr) {
            error = "falha interna no avanço do cursor de particionamento de tiles";
            return false;
        }

        std::vector<int32_t> sel;
        sel.reserve(cand - curr);
        for (uint32_t i = curr; i < cand; ++i) {
            sel.push_back(int32_t(i));
        }

        llama_ubatch sub_ub = llama_ubatch_subset(ubatch, sel);

        causal_tile ct;
        ct.tile_id      = uint32_t(local_plan.tiles.size());
        ct.token_offset = curr;
        ct.n_tokens     = cand - curr;
        ct.pos_min      = ubatch.pos[curr];
        ct.pos_max      = ubatch.pos[cand - 1];

        if (p > 0 && ubatch.pos[curr] >= llama_pos(p)) {
            ct.range = llama_position_split::range_kind::OVERFLOW;
        } else {
            ct.range = llama_position_split::range_kind::LOCAL;
        }

        ct.is_protected_window = (protected_tokens > 0 && cand == ubatch.n_tokens);

        local_plan.tiles.push_back(ct);
        local_plan.ubatch_tiles.push_back(std::move(sub_ub));

        curr = cand;
    }

    local_plan.tile_size   = tile_size;
    local_plan.n_tiles     = uint32_t(local_plan.tiles.size());
    local_plan.n_layers    = n_layers;
    local_plan.p           = p;
    local_plan.n_keep_tail = n_keep_tail;
    local_plan.schedule    = build_schedule(n_layers, local_plan.n_tiles);

    plan = std::move(local_plan);
    return true;
}

bool validate_topological_order(
    const wavefront_schedule & sched,
    const std::vector<uint32_t> & execution_order,
    std::string & error) {

    error.clear();
    if (sched.n_layers == 0 || sched.n_tiles == 0) {
        error = "grade de wavefront vazia (n_layers ou n_tiles zero)";
        return false;
    }

    if (sched.total_tasks != sched.n_layers * sched.n_tiles) {
        error = "total_tasks inconsistente com n_layers * n_tiles";
        return false;
    }

    const uint32_t expected_waves = sched.n_layers + sched.n_tiles - 1;
    if (sched.n_waves != expected_waves || sched.waves.size() != expected_waves) {
        error = "n_waves inconsistente no schedule";
        return false;
    }

    if (execution_order.size() != sched.total_tasks) {
        error = "tamanho da ordem de execução (" + std::to_string(execution_order.size()) +
                ") difere do total de tarefas (" + std::to_string(sched.total_tasks) + ")";
        return false;
    }

    // Validação da cobertura da grade e consistência de coordenadas (P2)
    std::vector<bool> task_defined(sched.total_tasks, false);
    for (uint32_t w = 0; w < sched.waves.size(); ++w) {
        for (const auto & coord : sched.waves[w]) {
            if (coord.layer >= sched.n_layers || coord.tile >= sched.n_tiles) {
                error = "coordenada fora dos limites na onda " + std::to_string(w);
                return false;
            }
            if (coord.wave != coord.layer + coord.tile || coord.wave != w) {
                error = "índice de onda inconsistente na coordenada";
                return false;
            }
            const uint32_t expected_task_id = coord.layer * sched.n_tiles + coord.tile;
            if (coord.task_id != expected_task_id) {
                error = "task_id inconsistente na coordenada";
                return false;
            }
            if (task_defined[coord.task_id]) {
                error = "tarefa duplicada no schedule: " + std::to_string(coord.task_id);
                return false;
            }
            task_defined[coord.task_id] = true;
        }
    }

    for (uint32_t i = 0; i < sched.total_tasks; ++i) {
        if (!task_defined[i]) {
            error = "schedule incompleto: tarefa ausente " + std::to_string(i);
            return false;
        }
    }

    std::vector<bool> executed(sched.total_tasks, false);

    for (size_t i = 0; i < execution_order.size(); ++i) {
        const uint32_t tid = execution_order[i];
        if (tid >= sched.total_tasks) {
            error = "task_id " + std::to_string(tid) + " fora dos limites no passo " + std::to_string(i);
            return false;
        }
        if (executed[tid]) {
            error = "tarefa duplicada no passo " + std::to_string(i) + ": task_id " + std::to_string(tid);
            return false;
        }

        const uint32_t l = tid / sched.n_tiles;
        const uint32_t t = tid % sched.n_tiles;

        // Dependência vertical canônica (Camada anterior: (l - 1, t))
        if (l > 0) {
            const uint32_t pred_v = (l - 1) * sched.n_tiles + t;
            if (!executed[pred_v]) {
                error = "violação de dependência vertical: tarefa (" + std::to_string(l) +
                        "," + std::to_string(t) + ") executada antes de (" +
                        std::to_string(l - 1) + "," + std::to_string(t) + ")";
                return false;
            }
        }

        // Dependência horizontal canônica (Tile causal anterior: (l, t - 1))
        if (t > 0) {
            const uint32_t pred_h = l * sched.n_tiles + (t - 1);
            if (!executed[pred_h]) {
                error = "violação de dependência causal horizontal: tarefa (" + std::to_string(l) +
                        "," + std::to_string(t) + ") executada antes de (" +
                        std::to_string(l) + "," + std::to_string(t - 1) + ")";
                return false;
            }
        }

        executed[tid] = true;
    }

    return true;
}

bool generate_causal_mask(
    const std::vector<causal_tile> & tiles,
    uint32_t total_tokens,
    std::vector<float> & mask_out,
    std::string & error) {

    error.clear();
    if (total_tokens == 0 || total_tokens > 8192) {
        error = "total_tokens inválido (deve ser > 0 e <= 8192 no gerador de máscara causal de teste)";
        return false;
    }
    if (tiles.empty()) {
        error = "vetor de tiles vazio";
        return false;
    }

    uint32_t accounted = 0;
    for (size_t i = 0; i < tiles.size(); ++i) {
        const auto & t = tiles[i];
        if (t.tile_id != i) {
            error = "tile_id inconsistente no índice " + std::to_string(i);
            return false;
        }
        if (t.n_tokens == 0) {
            error = "tile " + std::to_string(i) + " com n_tokens zero";
            return false;
        }
        if (t.token_offset != accounted) {
            error = "lacuna ou sobreposição de token_offset no tile " + std::to_string(i);
            return false;
        }
        accounted += t.n_tokens;
    }
    if (accounted != total_tokens) {
        error = "soma dos tokens dos tiles (" + std::to_string(accounted) +
                ") != total_tokens (" + std::to_string(total_tokens) + ")";
        return false;
    }

    mask_out.assign(size_t(total_tokens) * size_t(total_tokens), 0.0f);
    const float kNegInf = -std::numeric_limits<float>::infinity();

    for (const auto & t_q : tiles) {
        for (uint32_t lq = 0; lq < t_q.n_tokens; ++lq) {
            const uint32_t q_idx = t_q.token_offset + lq;

            for (const auto & t_k : tiles) {
                for (uint32_t lk = 0; lk < t_k.n_tokens; ++lk) {
                    const uint32_t k_idx = t_k.token_offset + lk;
                    const size_t cell = size_t(q_idx) * size_t(total_tokens) + size_t(k_idx);

                    if (t_k.tile_id > t_q.tile_id) {
                        // Tile futuro: estritamente proibido
                        mask_out[cell] = kNegInf;
                    } else if (t_k.tile_id < t_q.tile_id) {
                        // Tile passado: completamente visível
                        mask_out[cell] = 0.0f;
                    } else {
                        // Mesmo tile: causalidade triangular inferior
                        mask_out[cell] = (lk <= lq) ? 0.0f : kNegInf;
                    }
                }
            }
        }
    }

    return true;
}

void progressive_tracker::init(uint32_t layers, uint32_t tiles) {
    n_layers = layers;
    n_tiles  = tiles;
    n_waves  = (layers > 0 && tiles > 0) ? (layers + tiles - 1) : 0;
    current_wave = 0;
    tile_completed.assign(tiles, false);
    tokens_completed = 0;
}

uint32_t progressive_tracker::wave_for_tile_completion(uint32_t tile_idx) const {
    if (n_layers == 0) return 0;
    return (n_layers - 1) + tile_idx;
}

bool progressive_tracker::step_wave(
    uint32_t wave_idx,
    const std::vector<causal_tile> & tiles,
    std::vector<uint32_t> & newly_completed_tiles) {

    newly_completed_tiles.clear();

    if (n_layers == 0 || n_tiles == 0) {
        return false;
    }
    if (tiles.size() != n_tiles) {
        return false;
    }
    if (wave_idx >= n_waves) {
        return false;
    }
    // Exige progressão sequencial monotônica de ondas (P2)
    if (wave_idx != current_wave) {
        return false;
    }

    // Um tile T completa na onda W quando atinge a camada final (L - 1):
    // W = (L - 1) + T => T = W - (L - 1)
    if (wave_idx >= n_layers - 1) {
        const uint32_t t = wave_idx - (n_layers - 1);
        if (t < n_tiles && !tile_completed[t]) {
            tile_completed[t] = true;
            newly_completed_tiles.push_back(t);
            tokens_completed += tiles[t].n_tokens;
        }
    }

    current_wave++;
    return true;
}

bool progressive_tracker::is_tile_ready(uint32_t tile_idx) const {
    if (tile_idx >= tile_completed.size()) return false;
    return tile_completed[tile_idx];
}

bool progressive_tracker::all_completed() const {
    if (tile_completed.empty()) return false;
    for (bool b : tile_completed) {
        if (!b) return false;
    }
    return true;
}

void kv_cache_tracker::init(uint32_t layers, uint32_t p_val, uint32_t hist_local, uint32_t hist_remote) {
    n_layers = layers;
    p = p_val;
    initial_history_local = hist_local;
    initial_history_remote = hist_remote;
    committed_rows_local.assign(layers, 0);
    committed_rows_remote.assign(layers, 0);
    last_committed_tile.assign(layers, -1);
    kvarn_group_sealed.assign(layers, (hist_local % llama_position_split::kKvarnGroup == 0));
}

bool kv_cache_tracker::can_layer_attend_tile(uint32_t layer, const causal_tile & tile, std::string & error) const {
    error.clear();
    if (layer >= n_layers) {
        error = "camada " + std::to_string(layer) + " excede n_layers " + std::to_string(n_layers);
        return false;
    }

    // Para tile 0, o cache inicial está disponível
    if (tile.tile_id == 0) {
        return true;
    }

    // Para tile T > 0, o tile T-1 deve ter sido comitado nesta camada
    const int32_t required_prev = int32_t(tile.tile_id - 1);
    if (last_committed_tile[layer] < required_prev) {
        error = "camada " + std::to_string(layer) + " tentou atender ao tile " +
                std::to_string(tile.tile_id) + " antes do commit do tile " +
                std::to_string(required_prev) + " no cache KV";
        return false;
    }

    return true;
}

bool kv_cache_tracker::record_tile_kv_commit(uint32_t layer, const causal_tile & tile, std::string & error) {
    error.clear();
    if (layer >= n_layers) {
        error = "camada " + std::to_string(layer) + " fora dos limites";
        return false;
    }

    // Commit estritamente sequencial
    const int32_t expected = last_committed_tile[layer] + 1;
    if (int32_t(tile.tile_id) != expected) {
        error = "camada " + std::to_string(layer) + " tentou commit fora de ordem: tile " +
                std::to_string(tile.tile_id) + " (esperado " + std::to_string(expected) + ")";
        return false;
    }

    last_committed_tile[layer] = int32_t(tile.tile_id);

    if (tile.range == llama_position_split::range_kind::LOCAL) {
        committed_rows_local[layer] += tile.n_tokens;
        // KVarN sela a cada 128 linhas absolutas (histórico + novo) (P2)
        const uint64_t total_local = uint64_t(initial_history_local) + uint64_t(committed_rows_local[layer]);
        kvarn_group_sealed[layer] = (total_local % llama_position_split::kKvarnGroup == 0);
    } else {
        committed_rows_remote[layer] += tile.n_tokens;
    }

    return true;
}

ggml_cgraph * build_wavefront_computation_graph(
    ggml_context * ctx,
    const wavefront_graph_spec & spec,
    const wavefront_schedule & sched,
    const std::vector<ggml_tensor *> & tile_inputs,
    std::vector<ggml_tensor *> & tile_outputs,
    const std::vector<ggml_tensor *> & weights_q,
    const std::vector<ggml_tensor *> & weights_k,
    const std::vector<ggml_tensor *> & weights_v,
    const std::vector<ggml_tensor *> & weights_o,
    const std::vector<ggml_tensor *> & masks) {

    tile_outputs.clear();

    if (!ctx) {
        return nullptr;
    }

    if (spec.n_layers == 0 || spec.n_tiles == 0 || spec.tile_size == 0 || spec.d_model <= 0) {
        return nullptr;
    }

    if (sched.n_layers != spec.n_layers || sched.n_tiles != spec.n_tiles) {
        return nullptr;
    }

    if (tile_inputs.size() != spec.n_tiles) {
        return nullptr;
    }
    for (uint32_t t = 0; t < spec.n_tiles; ++t) {
        if (!tile_inputs[t]) return nullptr;
        if (tile_inputs[t]->ne[0] != spec.d_model) return nullptr;
        if (tile_inputs[t]->ne[1] <= 0 || tile_inputs[t]->ne[1] > (int64_t)spec.tile_size) return nullptr;
    }

    if (weights_q.size() != spec.n_layers || weights_k.size() != spec.n_layers ||
        weights_v.size() != spec.n_layers || weights_o.size() != spec.n_layers) {
        return nullptr;
    }
    for (uint32_t l = 0; l < spec.n_layers; ++l) {
        if (!weights_q[l] || !weights_k[l] || !weights_v[l] || !weights_o[l]) return nullptr;
        if (weights_q[l]->ne[0] != spec.d_model || weights_q[l]->ne[1] != spec.d_model) return nullptr;
        if (weights_k[l]->ne[0] != spec.d_model || weights_k[l]->ne[1] != spec.d_model) return nullptr;
        if (weights_v[l]->ne[0] != spec.d_model || weights_v[l]->ne[1] != spec.d_model) return nullptr;
        if (weights_o[l]->ne[0] != spec.d_model || weights_o[l]->ne[1] != spec.d_model) return nullptr;
    }

    // Validação estrita de máscaras (P1: máscara ausente ou null falha fechada)
    const bool per_task_masks = (masks.size() == spec.n_layers * spec.n_tiles);
    const bool per_tile_masks = (masks.size() == spec.n_tiles);
    if (!per_task_masks && !per_tile_masks) {
        return nullptr;
    }
    for (const auto * m : masks) {
        if (!m) return nullptr;
    }

    // Dimensionamento dinâmico e verificado do grafo GGML (P2)
    const size_t estimated_nodes = std::max<size_t>(
        (size_t)GGML_DEFAULT_GRAPH_SIZE,
        (size_t)spec.n_layers * ((size_t)spec.n_tiles * 16 + 8) + (size_t)spec.n_tiles * 4 + 64);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, estimated_nodes, false);
    if (!gf) {
        return nullptr;
    }

    tile_outputs.resize(spec.n_tiles, nullptr);

    // Grid de ativações: h[layer][tile]
    // h[0][t] é a entrada tile_inputs[t]
    std::vector<std::vector<ggml_tensor *>> h(spec.n_layers + 1, std::vector<ggml_tensor *>(spec.n_tiles, nullptr));
    for (uint32_t t = 0; t < spec.n_tiles; ++t) {
        h[0][t] = tile_inputs[t];
    }

    // Cache acumulado de chaves e valores: kv_k[layer][tile], kv_v[layer][tile]
    std::vector<std::vector<ggml_tensor *>> kv_k(spec.n_layers, std::vector<ggml_tensor *>(spec.n_tiles, nullptr));
    std::vector<std::vector<ggml_tensor *>> kv_v(spec.n_layers, std::vector<ggml_tensor *>(spec.n_tiles, nullptr));

    // Execução em ondas W = 0 .. n_waves - 1
    for (uint32_t w = 0; w < sched.n_waves; ++w) {
        for (const auto & coord : sched.waves[w]) {
            const uint32_t l = coord.layer;
            const uint32_t t = coord.tile;

            ggml_tensor * cur_in = h[l][t];
            if (!cur_in) return nullptr;

            // Projeções Q, K, V
            ggml_tensor * q = ggml_mul_mat(ctx, weights_q[l], cur_in);
            ggml_tensor * k = ggml_mul_mat(ctx, weights_k[l], cur_in);
            ggml_tensor * v = ggml_mul_mat(ctx, weights_v[l], cur_in);

            // Acumulação causal de KV cache na camada l até o tile t
            ggml_tensor * k_att = k;
            ggml_tensor * v_att = v;

            if (t > 0) {
                if (!kv_k[l][t - 1] || !kv_v[l][t - 1]) return nullptr;
                k_att = ggml_concat(ctx, kv_k[l][t - 1], k, 1);
                v_att = ggml_concat(ctx, kv_v[l][t - 1], v, 1);
            }
            kv_k[l][t] = k_att;
            kv_v[l][t] = v_att;

            // Atenção Q @ K^T / sqrt(D) + Mask
            // k_att ne = [D, N_accum], q ne = [D, N_q]
            // mul_mat(k_att, q) -> ne = [N_accum, N_q]
            ggml_tensor * kq = ggml_mul_mat(ctx, k_att, q);
            const float scale = 1.0f / std::sqrt(float(spec.d_model));

            // Máscara correspondente ao tile t
            const uint32_t mask_idx = per_task_masks ? (l * spec.n_tiles + t) : t;
            ggml_tensor * mask = masks[mask_idx];
            if (!mask) {
                return nullptr;
            }
            if (mask->ne[0] != k_att->ne[1] || mask->ne[1] != cur_in->ne[1]) {
                // Dimensões da máscara incompatíveis com [N_accum, N_q]
                return nullptr;
            }

            ggml_tensor * kq_soft = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);

            // kq_soft ne = [N_accum, N_q]
            // v_att ne = [D, N_accum]
            // v_att_t = cont(transpose(v_att)) ne = [N_accum, D]
            // mul_mat(v_att_t, kq_soft) -> ne = [D, N_q]
            ggml_tensor * v_att_t = ggml_cont(ctx, ggml_transpose(ctx, v_att));
            ggml_tensor * attn_out = ggml_mul_mat(ctx, v_att_t, kq_soft);

            // Projeção de saída e residual
            ggml_tensor * o_proj = ggml_mul_mat(ctx, weights_o[l], attn_out);
            ggml_tensor * cur_out = ggml_add(ctx, cur_in, o_proj);

            h[l + 1][t] = cur_out;
            ggml_build_forward_expand(gf, cur_out);
        }
    }

    for (uint32_t t = 0; t < spec.n_tiles; ++t) {
        tile_outputs[t] = h[spec.n_layers][t];
    }

    return gf;
}

} // namespace llama_wavefront
