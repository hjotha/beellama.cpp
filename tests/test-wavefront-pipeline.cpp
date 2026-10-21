// Teste de validação da FASE 7: Wavefront Causal de Tiles 64/128 e Serviço Progressivo de Prefill.
//
// Referência: docs/hybrid-attention-master-review.md §§2.3, 4.4, 5.3.

#include "llama-wavefront.h"
#include "llama-memory-hybrid.h"
#include "llama-position-split.h"
#include "llama-graph.h"
#include "llama-model.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <stdexcept>
#include <limits>
#include <random>
#include <string>
#include <vector>

static int g_checks = 0;
static int g_failures = 0;
static std::string g_case;

#define CHECK(cond) do { \
    ++g_checks; \
    if (!(cond)) { \
        ++g_failures; \
        std::printf("  FAIL [%s] linha %d: %s\n", g_case.c_str(), __LINE__, #cond); \
    } \
} while (0)

#define CHECK_TRUE(cond, msg) do { \
    ++g_checks; \
    if (!(cond)) { \
        ++g_failures; \
        std::printf("  FAIL [%s] linha %d: %s (%s)\n", g_case.c_str(), __LINE__, msg, #cond); \
    } \
} while (0)

static llama_ubatch make_test_ubatch(uint32_t n_tokens, llama_pos pos0, uint32_t n_keep_tail) {
    llama_ubatch ub;
    ub.b_equal_seqs = 1;
    ub.n_tokens     = n_tokens;
    ub.n_seq_tokens = n_tokens;
    ub.n_seqs       = 1;
    ub.n_seqs_unq   = 1;
    ub.n_pos        = 1;

    auto data = std::make_shared<llama_ubatch::data_t>();
    data->token.resize(n_tokens);
    data->pos.resize(n_tokens);
    data->n_seq_id.assign(n_tokens, 1);
    data->seq_id.resize(n_tokens);
    data->output.assign(n_tokens, 0);

    const llama_seq_id kSeq = 5;
    data->seq_id_unq = { kSeq };
    data->seq_idx.assign(LLAMA_MAX_SEQ, -1);
    data->seq_idx[kSeq] = 0;
    data->seq_id_data.assign(n_tokens, kSeq);

    for (uint32_t i = 0; i < n_tokens; ++i) {
        data->token[i] = llama_token(2000 + i);
        data->pos[i]   = pos0 + llama_pos(i);
        data->output[i] = (n_keep_tail > 0 && i + n_keep_tail >= n_tokens) ? 1 : 0;
        data->seq_id[i] = data->seq_id_data.data() + i;
    }

    ub.data       = data;
    ub.token      = data->token.data();
    ub.embd       = nullptr;
    ub.pos        = data->pos.data();
    ub.n_seq_id   = data->n_seq_id.data();
    ub.seq_id     = data->seq_id.data();
    ub.seq_id_unq = data->seq_id_unq.data();
    ub.seq_idx    = data->seq_idx.data();
    ub.output     = data->output.data();
    return ub;
}

static llama_ubatch make_test_ubatch_custom_pos(const std::vector<llama_pos> & positions, uint32_t n_keep_tail) {
    const uint32_t n_tokens = uint32_t(positions.size());
    llama_ubatch ub;
    ub.b_equal_seqs = 1;
    ub.n_tokens     = n_tokens;
    ub.n_seq_tokens = n_tokens;
    ub.n_seqs       = 1;
    ub.n_seqs_unq   = 1;
    ub.n_pos        = 1;

    auto data = std::make_shared<llama_ubatch::data_t>();
    data->token.resize(n_tokens);
    data->pos = positions;
    data->n_seq_id.assign(n_tokens, 1);
    data->seq_id.resize(n_tokens);
    data->output.assign(n_tokens, 0);

    const llama_seq_id kSeq = 5;
    data->seq_id_unq = { kSeq };
    data->seq_idx.assign(LLAMA_MAX_SEQ, -1);
    data->seq_idx[kSeq] = 0;
    data->seq_id_data.assign(n_tokens, kSeq);

    for (uint32_t i = 0; i < n_tokens; ++i) {
        data->token[i] = llama_token(2000 + i);
        data->output[i] = (n_keep_tail > 0 && i + n_keep_tail >= n_tokens) ? 1 : 0;
        data->seq_id[i] = data->seq_id_data.data() + i;
    }

    ub.data       = data;
    ub.token      = data->token.data();
    ub.embd       = nullptr;
    ub.pos        = data->pos.data();
    ub.n_seq_id   = data->n_seq_id.data();
    ub.seq_id     = data->seq_id.data();
    ub.seq_id_unq = data->seq_id_unq.data();
    ub.seq_idx    = data->seq_idx.data();
    ub.output     = data->output.data();
    return ub;
}

// 1. Validação de tamanho de tile
static void test_tile_size_validation() {
    g_case = "tile_size_validation";
    CHECK(llama_wavefront::validate_tile_size(16));
    CHECK(llama_wavefront::validate_tile_size(32));
    CHECK(llama_wavefront::validate_tile_size(64));
    CHECK(llama_wavefront::validate_tile_size(128));
    CHECK(llama_wavefront::validate_tile_size(256));
    CHECK(llama_wavefront::validate_tile_size(512));

    CHECK(!llama_wavefront::validate_tile_size(0));
    CHECK(!llama_wavefront::validate_tile_size(8));
    CHECK(!llama_wavefront::validate_tile_size(50));
    CHECK(!llama_wavefront::validate_tile_size(100));
    CHECK(!llama_wavefront::validate_tile_size(130));
    CHECK(!llama_wavefront::validate_tile_size(1024));
}

// 2. Propriedades matemáticas do escalonamento Wavefront 2D
static void test_wavefront_schedule_properties() {
    g_case = "schedule_properties";
    const uint32_t L_cases[] = {1, 2, 4, 16, 28, 32};
    const uint32_t K_cases[] = {1, 2, 4, 8};

    for (uint32_t L : L_cases) {
        for (uint32_t K : K_cases) {
            auto sched = llama_wavefront::build_schedule(L, K);
            CHECK(sched.n_layers == L);
            CHECK(sched.n_tiles == K);
            CHECK(sched.n_waves == L + K - 1);
            CHECK(sched.total_tasks == L * K);
            CHECK(sched.max_concurrency == std::min(L, K));
            CHECK(sched.waves.size() == sched.n_waves);

            uint32_t task_count = 0;
            std::vector<bool> seen(sched.total_tasks, false);

            for (uint32_t w = 0; w < sched.n_waves; ++w) {
                const auto & wave = sched.waves[w];
                CHECK(!wave.empty());
                CHECK(wave.size() <= sched.max_concurrency);

                for (const auto & coord : wave) {
                    CHECK(coord.wave == w);
                    CHECK(coord.layer + coord.tile == w);
                    CHECK(coord.task_id < sched.total_tasks);
                    CHECK(!seen[coord.task_id]);
                    seen[coord.task_id] = true;
                    ++task_count;

                    // Dependência vertical
                    if (coord.layer > 0) {
                        CHECK(coord.dep_layer_prev >= 0);
                        const uint32_t expected_prev_l = (coord.layer - 1) * K + coord.tile;
                        CHECK(uint32_t(coord.dep_layer_prev) == expected_prev_l);
                        // A camada anterior está na onda w - 1
                        CHECK((coord.layer - 1) + coord.tile == w - 1);
                    } else {
                        CHECK(coord.dep_layer_prev == -1);
                    }

                    // Dependência horizontal
                    if (coord.tile > 0) {
                        CHECK(coord.dep_tile_prev >= 0);
                        const uint32_t expected_prev_t = coord.layer * K + (coord.tile - 1);
                        CHECK(uint32_t(coord.dep_tile_prev) == expected_prev_t);
                        // O tile anterior está na onda w - 1
                        CHECK(coord.layer + (coord.tile - 1) == w - 1);
                    } else {
                        CHECK(coord.dep_tile_prev == -1);
                    }
                }
            }
            CHECK(task_count == L * K);
        }
    }
}

// 3. Validação de ordenação topológica
static void test_topological_order_validation() {
    g_case = "topological_order_validation";
    const uint32_t L = 4;
    const uint32_t K = 4;
    auto sched = llama_wavefront::build_schedule(L, K);

    // Ordem estrita por ondas (válida)
    std::vector<uint32_t> valid_order;
    for (const auto & wave : sched.waves) {
        for (const auto & coord : wave) {
            valid_order.push_back(coord.task_id);
        }
    }
    std::string err;
    CHECK(llama_wavefront::validate_topological_order(sched, valid_order, err));
    CHECK(err.empty());

    // Inversão vertical: executar (1, 0) [task_id = 1 * K + 0] antes de (0, 0) [task_id = 0]
    std::vector<uint32_t> invalid_vertical = valid_order;
    auto it_v0 = std::find(invalid_vertical.begin(), invalid_vertical.end(), 0 * K + 0);
    auto it_v1 = std::find(invalid_vertical.begin(), invalid_vertical.end(), 1 * K + 0);
    CHECK(it_v0 != invalid_vertical.end() && it_v1 != invalid_vertical.end());
    std::swap(*it_v0, *it_v1);
    CHECK(!llama_wavefront::validate_topological_order(sched, invalid_vertical, err));
    CHECK(!err.empty());

    // Inversão horizontal: executar (0, 1) antes de (0, 0)
    std::vector<uint32_t> invalid_horizontal = valid_order;
    auto it_00 = std::find(invalid_horizontal.begin(), invalid_horizontal.end(), 0 * K + 0);
    auto it_01 = std::find(invalid_horizontal.begin(), invalid_horizontal.end(), 0 * K + 1);
    CHECK(it_00 != invalid_horizontal.end() && it_01 != invalid_horizontal.end());
    std::swap(*it_00, *it_01);
    CHECK(!llama_wavefront::validate_topological_order(sched, invalid_horizontal, err));
    CHECK(!err.empty());

    // Tarefa duplicada
    std::vector<uint32_t> duplicate_order = valid_order;
    duplicate_order.back() = duplicate_order.front();
    CHECK(!llama_wavefront::validate_topological_order(sched, duplicate_order, err));

    // Tamanho incompleto
    std::vector<uint32_t> incomplete_order = valid_order;
    incomplete_order.pop_back();
    CHECK(!llama_wavefront::validate_topological_order(sched, incomplete_order, err));
}

// 4. Particionamento básico em tiles 64 e 128
static void test_partitioning_basic_64_128() {
    g_case = "partitioning_basic_64_128";
    const uint32_t N = 256;
    auto ub = make_test_ubatch(N, 100, 0);

    // Particionamento 128 (exatamente 2 tiles)
    {
        llama_wavefront::wavefront_plan plan;
        std::string err;
        CHECK(llama_wavefront::partition_ubatch_causal_tiles(ub, 128, 4, 0, 0, plan, err));
        CHECK(err.empty());
        CHECK(plan.n_tiles == 2);
        CHECK(plan.tiles.size() == 2);
        CHECK(plan.ubatch_tiles.size() == 2);

        CHECK(plan.tiles[0].n_tokens == 128);
        CHECK(plan.tiles[0].pos_min == 100);
        CHECK(plan.tiles[0].pos_max == 227);
        CHECK(plan.ubatch_tiles[0].n_tokens == 128);

        CHECK(plan.tiles[1].n_tokens == 128);
        CHECK(plan.tiles[1].pos_min == 228);
        CHECK(plan.tiles[1].pos_max == 355);
        CHECK(plan.ubatch_tiles[1].n_tokens == 128);

        CHECK(plan.schedule.n_waves == 4 + 2 - 1);
    }

    // Particionamento 64 (exatamente 4 tiles)
    {
        llama_wavefront::wavefront_plan plan;
        std::string err;
        CHECK(llama_wavefront::partition_ubatch_causal_tiles(ub, 64, 4, 0, 0, plan, err));
        CHECK(err.empty());
        CHECK(plan.n_tiles == 4);
        CHECK(plan.tiles.size() == 4);
        for (uint32_t t = 0; t < 4; ++t) {
            CHECK(plan.tiles[t].n_tokens == 64);
            CHECK(plan.tiles[t].token_offset == t * 64);
            CHECK(plan.tiles[t].pos_min == llama_pos(100 + t * 64));
            CHECK(plan.tiles[t].pos_max == llama_pos(100 + (t + 1) * 64 - 1));
            CHECK(plan.ubatch_tiles[t].n_tokens == 64);
        }
        CHECK(plan.schedule.n_waves == 4 + 4 - 1);
    }
}

// 5. Particionamento com sobras (unaligned tail)
static void test_partitioning_unaligned_tail() {
    g_case = "partitioning_unaligned_tail";
    const uint32_t N = 300;
    auto ub = make_test_ubatch(N, 0, 0);

    llama_wavefront::wavefront_plan plan;
    std::string err;
    CHECK(llama_wavefront::partition_ubatch_causal_tiles(ub, 128, 4, 0, 0, plan, err));
    CHECK(err.empty());
    CHECK(plan.n_tiles == 3);
    CHECK(plan.tiles[0].n_tokens == 128);
    CHECK(plan.tiles[1].n_tokens == 128);
    CHECK(plan.tiles[2].n_tokens == 44);

    uint32_t sum = 0;
    for (const auto & t : plan.tiles) sum += t.n_tokens;
    CHECK(sum == N);
}

// 6. Invariantes da fronteira de particionamento P
static void test_position_split_boundary_p_invariants() {
    g_case = "boundary_p_invariants";
    const uint32_t P = 512;
    const uint32_t N = 300;
    // [400, 700) atravessa P = 512
    auto ub = make_test_ubatch(N, 400, 0);

    llama_wavefront::wavefront_plan plan;
    std::string err;
    CHECK(llama_wavefront::partition_ubatch_causal_tiles(ub, 128, 4, P, 0, plan, err));
    CHECK(err.empty());

    // Invariante essencial: NENHUM TILE PODE CRUZAR P
    for (const auto & t : plan.tiles) {
        const bool entirely_local = (t.pos_max < llama_pos(P));
        const bool entirely_ovf   = (t.pos_min >= llama_pos(P));
        CHECK(entirely_local || entirely_ovf);
        if (entirely_local) {
            CHECK(t.range == llama_position_split::range_kind::LOCAL);
        } else {
            CHECK(t.range == llama_position_split::range_kind::OVERFLOW);
        }
    }

    // Tile 0 deve cobrir [400, 512) (112 tokens, limitado por P)
    CHECK(plan.tiles[0].n_tokens == P - 400);
    CHECK(plan.tiles[0].pos_min == 400);
    CHECK(plan.tiles[0].pos_max == 511);
    CHECK(plan.tiles[0].range == llama_position_split::range_kind::LOCAL);

    // Tiles subsequentes começam em P
    CHECK(plan.tiles[1].pos_min == llama_pos(P));
    CHECK(plan.tiles[1].range == llama_position_split::range_kind::OVERFLOW);
}

// 7. Preservação da janela protegida recorrente (1 + n_rs_seq)
static void test_recurrent_protected_window_preservation() {
    g_case = "recurrent_protected_window";
    const uint32_t P = 512;
    const uint32_t NKEEP = 4; // 1 + n_rs_seq
    const uint32_t N = 256;

    // Caso A: janela protegida bem comportada dentro do intervalo
    {
        auto ub = make_test_ubatch(N, 0, NKEEP);
        llama_wavefront::wavefront_plan plan;
        std::string err;
        CHECK(llama_wavefront::partition_ubatch_causal_tiles(ub, 128, 4, P, NKEEP, plan, err));
        CHECK(err.empty());
        CHECK(plan.tiles.back().is_protected_window);
        CHECK(plan.tiles.back().n_tokens >= NKEEP);
    }

    // Caso B: janela protegida cruzando P deve falhar fechada
    {
        // Janela de tamanho 4 cruzando P: posições 510, 511, 512, 513
        auto ub_cross = make_test_ubatch(4, 510, 4);
        llama_wavefront::wavefront_plan plan;
        std::string err;
        CHECK(!llama_wavefront::partition_ubatch_causal_tiles(ub_cross, 128, 4, P, 4, plan, err));
        CHECK(!err.empty());
    }

    // Caso C (Sol critique): N=256, T=128, keep=129 (keep > tile_size) deve falhar fechada
    {
        auto ub = make_test_ubatch(256, 0, 129);
        llama_wavefront::wavefront_plan plan;
        std::string err;
        CHECK(!llama_wavefront::partition_ubatch_causal_tiles(ub, 128, 4, 0, 129, plan, err));
        CHECK(!err.empty());
        CHECK(plan.tiles.empty());
    }

    // Caso D (Sol critique): N=256, T=128, keep=256 deve falhar fechada
    {
        auto ub = make_test_ubatch(256, 0, 256);
        llama_wavefront::wavefront_plan plan;
        std::string err;
        CHECK(!llama_wavefront::partition_ubatch_causal_tiles(ub, 128, 4, 0, 256, plan, err));
        CHECK(!err.empty());
        CHECK(plan.tiles.empty());
    }

    // Caso E (Sol critique): N=32, pos=500..531, P=512, keep=64 (N < keep, janela cruza P) deve falhar fechada
    {
        auto ub = make_test_ubatch(32, 500, 64);
        llama_wavefront::wavefront_plan plan;
        std::string err;
        CHECK(!llama_wavefront::partition_ubatch_causal_tiles(ub, 128, 4, P, 64, plan, err));
        CHECK(!err.empty());
        CHECK(plan.tiles.empty());
    }

    // Caso F (Sol critique): N=256, T=128, keep=4 NÃO deve fragmentar em 128/124/4, e sim 128/128!
    {
        auto ub = make_test_ubatch(256, 0, 4);
        llama_wavefront::wavefront_plan plan;
        std::string err;
        CHECK(llama_wavefront::partition_ubatch_causal_tiles(ub, 128, 4, 0, 4, plan, err));
        CHECK(err.empty());
        CHECK(plan.n_tiles == 2);
        CHECK(plan.tiles.size() == 2);
        CHECK(plan.tiles[0].n_tokens == 128);
        CHECK(plan.tiles[1].n_tokens == 128);
        CHECK(!plan.tiles[0].is_protected_window);
        CHECK(plan.tiles[1].is_protected_window);
    }

    // Caso G (Sol critique P1): Posições repetidas [0, 0] devem falhar fechadas
    {
        auto ub_dup = make_test_ubatch_custom_pos({0, 0}, 0);
        llama_wavefront::wavefront_plan plan;
        std::string err;
        CHECK(!llama_wavefront::partition_ubatch_causal_tiles(ub_dup, 128, 4, 0, 0, plan, err));
        CHECK(!err.empty());
    }

    // Caso H (Sol critique P1): Posições negativas [-1, 0] devem falhar fechadas
    {
        auto ub_neg = make_test_ubatch_custom_pos({-1, 0}, 0);
        llama_wavefront::wavefront_plan plan;
        std::string err;
        CHECK(!llama_wavefront::partition_ubatch_causal_tiles(ub_neg, 128, 4, 0, 0, plan, err));
        CHECK(!err.empty());
    }
}

// 8. Construção e validação da matriz de máscara causal
static void test_causal_mask_matrix_generation() {
    g_case = "causal_mask_matrix";
    const uint32_t N = 256;
    auto ub = make_test_ubatch(N, 0, 0);

    llama_wavefront::wavefront_plan plan;
    std::string err;
    CHECK(llama_wavefront::partition_ubatch_causal_tiles(ub, 64, 4, 0, 0, plan, err));

    std::vector<float> mask;
    CHECK(llama_wavefront::generate_causal_mask(plan.tiles, N, mask, err));
    CHECK(err.empty());
    CHECK(mask.size() == size_t(N) * size_t(N));

    // Verificação amostral e exaustiva das regras de máscara
    for (uint32_t q = 0; q < N; ++q) {
        for (uint32_t k = 0; k < N; ++k) {
            const float val = mask[size_t(q) * size_t(N) + size_t(k)];
            if (k <= q) {
                CHECK(val == 0.0f);
            } else {
                CHECK(std::isinf(val) && val < 0.0f); // -inf
            }
        }
    }
}

// 9. Rastreador de estado do KV cache e selamento KVarN
static void test_kv_cache_tracker_and_kvarn_sealing() {
    g_case = "kv_cache_tracker_kvarn_sealing";
    llama_wavefront::kv_cache_tracker tracker;
    tracker.init(4, 512);

    llama_wavefront::causal_tile t0;
    t0.tile_id = 0;
    t0.n_tokens = 64;
    t0.range = llama_position_split::range_kind::LOCAL;

    llama_wavefront::causal_tile t1;
    t1.tile_id = 1;
    t1.n_tokens = 64;
    t1.range = llama_position_split::range_kind::LOCAL;

    std::string err;
    // Camada 0 pode atender ao tile 0 inicialmente
    CHECK(tracker.can_layer_attend_tile(0, t0, err));
    // Mas NÃO pode atender ao tile 1 antes do commit do tile 0
    CHECK(!tracker.can_layer_attend_tile(0, t1, err));

    // Commit do tile 0 (64 tokens): grupo KVarN de 128 não está selado
    CHECK(tracker.record_tile_kv_commit(0, t0, err));
    CHECK(!tracker.kvarn_group_sealed[0]);
    CHECK(tracker.committed_rows_local[0] == 64);

    // Agora a camada 0 pode atender ao tile 1
    CHECK(tracker.can_layer_attend_tile(0, t1, err));

    // Commit do tile 1 (mais 64 tokens = 128 tokens): grupo KVarN SELADO!
    CHECK(tracker.record_tile_kv_commit(0, t1, err));
    CHECK(tracker.kvarn_group_sealed[0]);
    CHECK(tracker.committed_rows_local[0] == 128);

    // Tentativa de commit fora de ordem deve falhar fechada
    llama_wavefront::causal_tile t3;
    t3.tile_id = 3;
    t3.n_tokens = 64;
    CHECK(!tracker.record_tile_kv_commit(0, t3, err));

    // Teste de histórico pré-existente e selamento KVarN (P2 Sol critique)
    {
        llama_wavefront::kv_cache_tracker tr_hist;
        // Histórico de 192 linhas locais (grupo incompleto: 192 % 128 == 64)
        tr_hist.init(4, 512, 192, 0);
        CHECK(!tr_hist.kvarn_group_sealed[0]);

        // Commit de mais 64 tokens -> 192 + 64 = 256 (256 % 128 == 0) -> GRUPO SELADO!
        llama_wavefront::causal_tile t_hist0;
        t_hist0.tile_id = 0;
        t_hist0.n_tokens = 64;
        t_hist0.range = llama_position_split::range_kind::LOCAL;
        CHECK(tr_hist.record_tile_kv_commit(0, t_hist0, err));
        CHECK(tr_hist.kvarn_group_sealed[0]);
    }
}

// 10. Rastreador de serviço progressivo (Time-to-First-Token)
static void test_progressive_prefill_tracker() {
    g_case = "progressive_prefill_tracker";
    const uint32_t L = 4;
    const uint32_t K = 4;
    llama_wavefront::progressive_tracker tracker;
    tracker.init(L, K);

    std::vector<llama_wavefront::causal_tile> tiles(K);
    for (uint32_t t = 0; t < K; ++t) {
        tiles[t].tile_id = t;
        tiles[t].n_tokens = 64;
    }

    CHECK(tracker.wave_for_tile_completion(0) == L - 1); // Onda 3
    CHECK(tracker.wave_for_tile_completion(K - 1) == L + K - 2); // Onda 6

    std::vector<uint32_t> completed;

    // Ondas 0, 1, 2: nenhum tile completo
    for (uint32_t w = 0; w < L - 1; ++w) {
        CHECK(tracker.step_wave(w, tiles, completed));
        CHECK(completed.empty());
        CHECK(!tracker.is_tile_ready(0));
    }

    // Onda 3 (L - 1): Tile 0 completo! Serviço progressivo iniciado!
    CHECK(tracker.step_wave(3, tiles, completed));
    CHECK(completed.size() == 1 && completed[0] == 0);
    CHECK(tracker.is_tile_ready(0));
    CHECK(!tracker.is_tile_ready(1));
    CHECK(tracker.tokens_completed == 64);

    // Onda 4: Tile 1 completo!
    CHECK(tracker.step_wave(4, tiles, completed));
    CHECK(completed.size() == 1 && completed[0] == 1);
    CHECK(tracker.is_tile_ready(1));
    CHECK(tracker.tokens_completed == 128);

    // Onda 5: Tile 2 completo!
    CHECK(tracker.step_wave(5, tiles, completed));
    CHECK(completed.size() == 1 && completed[0] == 2);
    CHECK(tracker.is_tile_ready(2));
    CHECK(tracker.tokens_completed == 192);

    // Onda 6: Tile 3 completo! Todos concluídos.
    CHECK(tracker.step_wave(6, tiles, completed));
    CHECK(completed.size() == 1 && completed[0] == 3);
    CHECK(tracker.all_completed());
    CHECK(tracker.tokens_completed == 256);

    // Validação de erro com chamada fora de ordem ou tiles incompletos (P2 Sol critique)
    {
        llama_wavefront::progressive_tracker tr_bad;
        tr_bad.init(4, 4);
        std::vector<uint32_t> newly;
        // Chamar onda 3 de imediato deve falhar fechada
        CHECK(!tr_bad.step_wave(3, tiles, newly));
        // Chamar com vetor de tiles incompleto deve falhar fechada
        std::vector<llama_wavefront::causal_tile> short_tiles(2);
        CHECK(!tr_bad.step_wave(0, short_tiles, newly));
    }
}

// 11. Protótipo de grafo de computação GGML em Wavefront
static void test_ggml_wavefront_computation_graph() {
    g_case = "ggml_wavefront_graph";

    llama_wavefront::wavefront_graph_spec spec;
    spec.n_layers  = 2;
    spec.n_tiles   = 2;
    spec.tile_size = 64;
    spec.d_model   = 64;

    const size_t mem_size = 64 * 1024 * 1024;
    struct ggml_init_params params = {
        /* .mem_size   = */ mem_size,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };

    struct ggml_context * ctx = ggml_init(params);
    CHECK(ctx != nullptr);

    auto sched = llama_wavefront::build_schedule(spec.n_layers, spec.n_tiles);

    // Entradas dos tiles: X_0 e X_1
    std::vector<ggml_tensor *> tile_inputs(spec.n_tiles);
    for (uint32_t t = 0; t < spec.n_tiles; ++t) {
        tile_inputs[t] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, spec.d_model, spec.tile_size);
        float * data = (float *) tile_inputs[t]->data;
        for (int64_t i = 0; i < spec.d_model * spec.tile_size; ++i) {
            data[i] = 0.05f * float((i % 17) + 1);
        }
    }

    // Pesos por camada
    std::vector<ggml_tensor *> wq(spec.n_layers);
    std::vector<ggml_tensor *> wk(spec.n_layers);
    std::vector<ggml_tensor *> wv(spec.n_layers);
    std::vector<ggml_tensor *> wo(spec.n_layers);

    for (uint32_t l = 0; l < spec.n_layers; ++l) {
        wq[l] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, spec.d_model, spec.d_model);
        wk[l] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, spec.d_model, spec.d_model);
        wv[l] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, spec.d_model, spec.d_model);
        wo[l] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, spec.d_model, spec.d_model);

        // Inicialização com valores determinísticos pequenos para estabilidade
        auto init_weight = [](ggml_tensor * t, float scale) {
            float * d = (float *) t->data;
            const int64_t n = ggml_nelements(t);
            for (int64_t i = 0; i < n; ++i) {
                d[i] = ((i % 13) - 6) * scale;
            }
        };
        init_weight(wq[l], 0.01f);
        init_weight(wk[l], 0.01f);
        init_weight(wv[l], 0.01f);
        init_weight(wo[l], 0.01f);
    }

    // Máscaras causais para cada (layer, tile)
    // Para tile 0: n_accum = 64, n_q = 64 -> causal triangular inferior
    // Para tile 1: n_accum = 128, n_q = 64 -> colunas 0..63 são 0.0, colunas 64..127 causais
    std::vector<ggml_tensor *> masks(spec.n_layers * spec.n_tiles);
    const float kNegInf = -std::numeric_limits<float>::infinity();

    for (uint32_t l = 0; l < spec.n_layers; ++l) {
        // Tile 0
        {
            const uint32_t idx = l * spec.n_tiles + 0;
            masks[idx] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, spec.tile_size, spec.tile_size);
            float * d = (float *) masks[idx]->data;
            for (uint32_t q = 0; q < spec.tile_size; ++q) {
                for (uint32_t k = 0; k < spec.tile_size; ++k) {
                    d[q * spec.tile_size + k] = (k <= q) ? 0.0f : kNegInf;
                }
            }
        }
        // Tile 1
        {
            const uint32_t idx = l * spec.n_tiles + 1;
            const uint32_t n_accum = spec.tile_size * 2;
            masks[idx] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_accum, spec.tile_size);
            float * d = (float *) masks[idx]->data;
            for (uint32_t q = 0; q < spec.tile_size; ++q) {
                for (uint32_t k = 0; k < n_accum; ++k) {
                    if (k < spec.tile_size) {
                        d[q * n_accum + k] = 0.0f; // Tile anterior totalmente visível
                    } else {
                        const uint32_t k_local = k - spec.tile_size;
                        d[q * n_accum + k] = (k_local <= q) ? 0.0f : kNegInf;
                    }
                }
            }
        }
    }

    std::vector<ggml_tensor *> tile_outputs;
    struct ggml_cgraph * gf = llama_wavefront::build_wavefront_computation_graph(
        ctx, spec, sched, tile_inputs, tile_outputs, wq, wk, wv, wo, masks);

    CHECK(gf != nullptr);
    CHECK(tile_outputs.size() == spec.n_tiles);
    for (uint32_t t = 0; t < spec.n_tiles; ++t) {
        CHECK(tile_outputs[t] != nullptr);
    }

    // Execução do grafo na CPU com checagem do status (P2 Sol critique)
    enum ggml_status status = ggml_graph_compute_with_ctx(ctx, gf, 1);
    CHECK(status == GGML_STATUS_SUCCESS);

    // Validação de saída: tensores computados sem NaN ou Inf
    for (uint32_t t = 0; t < spec.n_tiles; ++t) {
        const float * out = (const float *) tile_outputs[t]->data;
        const int64_t n_elem = ggml_nelements(tile_outputs[t]);
        CHECK(n_elem == spec.d_model * spec.tile_size);
        for (int64_t i = 0; i < n_elem; ++i) {
            CHECK(!std::isnan(out[i]));
            CHECK(!std::isinf(out[i]));
        }
    }

    ggml_free(ctx);
}

// 12. Teste de fail-closed e perturbação de futuro na máscara causal (P1 Sol critique)
static void test_causal_mask_leakage_and_future_perturbation() {
    g_case = "causal_mask_leakage_and_future_perturbation";

    llama_wavefront::wavefront_graph_spec spec;
    spec.n_layers  = 1;
    spec.n_tiles   = 1;
    spec.tile_size = 16;
    spec.d_model   = 4;

    auto sched = llama_wavefront::build_schedule(spec.n_layers, spec.n_tiles);

    struct ggml_init_params params = {
        /* .mem_size   = */ 16 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    CHECK(ctx != nullptr);

    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, spec.d_model, spec.tile_size);
    float * inp_data = (float *) inp->data;
    for (int64_t i = 0; i < ggml_nelements(inp); ++i) {
        inp_data[i] = 1.0f + 0.1f * float(i % 17);
    }

    std::vector<ggml_tensor *> wq(1), wk(1), wv(1), wo(1);
    wq[0] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, spec.d_model, spec.d_model);
    wk[0] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, spec.d_model, spec.d_model);
    wv[0] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, spec.d_model, spec.d_model);
    wo[0] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, spec.d_model, spec.d_model);

    auto init_identity = [](ggml_tensor * t) {
        float * d = (float *) t->data;
        const int64_t n = t->ne[0];
        for (int64_t r = 0; r < n; ++r) {
            for (int64_t c = 0; c < n; ++c) {
                d[r * n + c] = (r == c) ? 1.0f : 0.0f;
            }
        }
    };
    init_identity(wq[0]);
    init_identity(wk[0]);
    init_identity(wv[0]);
    init_identity(wo[0]);

    // Máscara causal triangular inferior
    const float kNegInf = -std::numeric_limits<float>::infinity();
    ggml_tensor * mask_causal = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, spec.tile_size, spec.tile_size);
    float * m_data = (float *) mask_causal->data;
    for (uint32_t q = 0; q < spec.tile_size; ++q) {
        for (uint32_t k = 0; k < spec.tile_size; ++k) {
            m_data[q * spec.tile_size + k] = (k <= q) ? 0.0f : kNegInf;
        }
    }

    // 1. Validação fail-closed: builder com máscara vazia deve retornar nullptr
    std::vector<ggml_tensor *> out_fail;
    CHECK(llama_wavefront::build_wavefront_computation_graph(
        ctx, spec, sched, {inp}, out_fail, wq, wk, wv, wo, {}) == nullptr);

    // 2. Validação fail-closed: builder com máscara nullptr deve retornar nullptr
    CHECK(llama_wavefront::build_wavefront_computation_graph(
        ctx, spec, sched, {inp}, out_fail, wq, wk, wv, wo, {nullptr}) == nullptr);

    // 3. Validação fail-closed: builder com máscara de shape incompatível deve retornar nullptr
    ggml_tensor * bad_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, spec.tile_size + 4, spec.tile_size);
    CHECK(llama_wavefront::build_wavefront_computation_graph(
        ctx, spec, sched, {inp}, out_fail, wq, wk, wv, wo, {bad_mask}) == nullptr);

    // 4. Teste de perturbação de futuro: token 0 deve ser ESTRITAMENTE INVARIANTE à alteração de tokens futuros
    std::vector<ggml_tensor *> out_base;
    struct ggml_cgraph * gf_base = llama_wavefront::build_wavefront_computation_graph(
        ctx, spec, sched, {inp}, out_base, wq, wk, wv, wo, {mask_causal});
    CHECK(gf_base != nullptr);
    CHECK(ggml_graph_compute_with_ctx(ctx, gf_base, 1) == GGML_STATUS_SUCCESS);

    // Salva o vetor de saída do token 0
    std::vector<float> token0_out_base(spec.d_model);
    const float * out0_ptr = (const float *) out_base[0]->data;
    for (int64_t i = 0; i < spec.d_model; ++i) {
        token0_out_base[i] = out0_ptr[i];
    }
    // Salva também o token 15 (último token)
    std::vector<float> token15_out_base(spec.d_model);
    for (int64_t i = 0; i < spec.d_model; ++i) {
        token15_out_base[i] = out0_ptr[(spec.tile_size - 1) * spec.d_model + i];
    }

    // Perturba radicalmente o token 15 na entrada (futuro em relação a 0..14)
    for (int64_t i = 0; i < spec.d_model; ++i) {
        inp_data[(spec.tile_size - 1) * spec.d_model + i] += 50.0f;
    }

    // Recomputa o grafo
    CHECK(ggml_graph_compute_with_ctx(ctx, gf_base, 1) == GGML_STATUS_SUCCESS);

    // Token 0 NÃO PODE ter mudado nem 1 epsilon (ausência estrita de vazamento causal futuro)
    for (int64_t i = 0; i < spec.d_model; ++i) {
        CHECK(std::abs(out0_ptr[i] - token0_out_base[i]) < 1e-6f);
    }

    // Mas o token 15 DEVE ter mudado (confirmando que a computação de fato ocorreu)
    bool token15_changed = false;
    for (int64_t i = 0; i < spec.d_model; ++i) {
        if (std::abs(out0_ptr[(spec.tile_size - 1) * spec.d_model + i] - token15_out_base[i]) > 1.0f) {
            token15_changed = true;
        }
    }
    CHECK(token15_changed);

    ggml_free(ctx);
}

// 13. Comparação de equivalência algébrica com referência causal monolítica (Sol critique)
static void test_monolithic_reference_equivalence() {
    g_case = "monolithic_reference_equivalence";

    const uint32_t L = 1;
    const uint32_t K = 2;
    const uint32_t tile_size = 16;
    const uint32_t N = K * tile_size; // 32 tokens
    const int64_t  D = 8;

    llama_wavefront::wavefront_graph_spec spec;
    spec.n_layers  = L;
    spec.n_tiles   = K;
    spec.tile_size = tile_size;
    spec.d_model   = D;

    const size_t mem_size = 32 * 1024 * 1024;
    struct ggml_init_params params = {
        /* .mem_size   = */ mem_size,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };

    struct ggml_context * ctx = ggml_init(params);
    CHECK(ctx != nullptr);

    // Entrada global X [D, N]
    ggml_tensor * X_mono = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, N);
    float * x_ptr = (float *) X_mono->data;
    for (int64_t i = 0; i < D * N; ++i) {
        x_ptr[i] = 0.02f * float((i % 23) + 1);
    }

    // Pesos W_q, W_k, W_v, W_o
    ggml_tensor * wq = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, D);
    ggml_tensor * wk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, D);
    ggml_tensor * wv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, D);
    ggml_tensor * wo = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, D);

    auto init_weights = [](ggml_tensor * t, float scale) {
        float * d = (float *) t->data;
        for (int64_t i = 0; i < ggml_nelements(t); ++i) {
            d[i] = ((i % 11) - 5) * scale;
        }
    };
    init_weights(wq, 0.05f);
    init_weights(wk, 0.05f);
    init_weights(wv, 0.05f);
    init_weights(wo, 0.05f);

    // 1. Grafo monolítico de referência para todos os N tokens
    const float kNegInf = -std::numeric_limits<float>::infinity();
    ggml_tensor * mask_mono = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, N);
    float * m_mono_ptr = (float *) mask_mono->data;
    for (uint32_t q = 0; q < N; ++q) {
        for (uint32_t k = 0; k < N; ++k) {
            m_mono_ptr[q * N + k] = (k <= q) ? 0.0f : kNegInf;
        }
    }

    ggml_tensor * q_mono = ggml_mul_mat(ctx, wq, X_mono);
    ggml_tensor * k_mono = ggml_mul_mat(ctx, wk, X_mono);
    ggml_tensor * v_mono = ggml_mul_mat(ctx, wv, X_mono);
    ggml_tensor * kq_mono = ggml_mul_mat(ctx, k_mono, q_mono);
    const float scale = 1.0f / std::sqrt(float(D));
    ggml_tensor * kq_mono_soft = ggml_soft_max_ext(ctx, kq_mono, mask_mono, scale, 0.0f);
    ggml_tensor * v_mono_t = ggml_cont(ctx, ggml_transpose(ctx, v_mono));
    ggml_tensor * attn_mono = ggml_mul_mat(ctx, v_mono_t, kq_mono_soft);
    ggml_tensor * o_proj_mono = ggml_mul_mat(ctx, wo, attn_mono);
    ggml_tensor * out_mono = ggml_add(ctx, X_mono, o_proj_mono);

    // 2. Grafo em wavefront particionado em 2 tiles de 16 tokens
    auto sched = llama_wavefront::build_schedule(L, K);

    std::vector<ggml_tensor *> tile_inputs(K);
    // Tile 0: tokens 0..15
    tile_inputs[0] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, tile_size);
    float * t0_in = (float *) tile_inputs[0]->data;
    for (int64_t i = 0; i < D * tile_size; ++i) {
        t0_in[i] = x_ptr[i];
    }
    // Tile 1: tokens 16..31
    tile_inputs[1] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, tile_size);
    float * t1_in = (float *) tile_inputs[1]->data;
    for (int64_t i = 0; i < D * tile_size; ++i) {
        t1_in[i] = x_ptr[D * tile_size + i];
    }

    // Máscaras para wavefront:
    // Tile 0: [16, 16] causal
    // Tile 1: [32, 16] (0..15 visíveis, 16..31 causais)
    std::vector<ggml_tensor *> wf_masks(2);
    wf_masks[0] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, tile_size, tile_size);
    float * m0 = (float *) wf_masks[0]->data;
    for (uint32_t q = 0; q < tile_size; ++q) {
        for (uint32_t k = 0; k < tile_size; ++k) {
            m0[q * tile_size + k] = (k <= q) ? 0.0f : kNegInf;
        }
    }

    wf_masks[1] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, tile_size);
    float * m1 = (float *) wf_masks[1]->data;
    for (uint32_t q = 0; q < tile_size; ++q) {
        for (uint32_t k = 0; k < N; ++k) {
            if (k < tile_size) {
                m1[q * N + k] = 0.0f;
            } else {
                const uint32_t k_local = k - tile_size;
                m1[q * N + k] = (k_local <= q) ? 0.0f : kNegInf;
            }
        }
    }

    std::vector<ggml_tensor *> wf_outputs;
    struct ggml_cgraph * gf_wf = llama_wavefront::build_wavefront_computation_graph(
        ctx, spec, sched, tile_inputs, wf_outputs, {wq}, {wk}, {wv}, {wo}, wf_masks);
    CHECK(gf_wf != nullptr);

    // Expandir ambos no grafo para executar no mesmo contexto
    struct ggml_cgraph * gf_all = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(gf_all, out_mono);
    for (auto * out_t : wf_outputs) {
        ggml_build_forward_expand(gf_all, out_t);
    }

    CHECK(ggml_graph_compute_with_ctx(ctx, gf_all, 1) == GGML_STATUS_SUCCESS);

    // 3. Comparação de exatidão algébrica/numérica entre Wavefront e Monolítico
    const float * mono_res = (const float *) out_mono->data;
    const float * t0_res   = (const float *) wf_outputs[0]->data;
    const float * t1_res   = (const float *) wf_outputs[1]->data;

    // Comparar Tile 0 com os primeiros 16 tokens
    float max_diff_t0 = 0.0f;
    for (int64_t i = 0; i < D * tile_size; ++i) {
        const float diff = std::abs(mono_res[i] - t0_res[i]);
        if (diff > max_diff_t0) max_diff_t0 = diff;
    }
    CHECK(max_diff_t0 < 1e-4f);

    // Comparar Tile 1 com os tokens 16..31
    float max_diff_t1 = 0.0f;
    for (int64_t i = 0; i < D * tile_size; ++i) {
        const float diff = std::abs(mono_res[D * tile_size + i] - t1_res[i]);
        if (diff > max_diff_t1) max_diff_t1 = diff;
    }
    CHECK(max_diff_t1 < 1e-4f);

    ggml_free(ctx);
}

// 14. Dimensionamento e capacidade do grafo GGML para shapes realistas (P2 Sol critique)
static void test_graph_capacity_and_scaling() {
    g_case = "graph_capacity_and_scaling";

    llama_wavefront::wavefront_graph_spec spec;
    spec.n_layers  = 48;
    spec.n_tiles   = 4;
    spec.tile_size = 16;
    spec.d_model   = 16;

    const size_t mem_size = 128 * 1024 * 1024;
    struct ggml_init_params params = {
        /* .mem_size   = */ mem_size,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    CHECK(ctx != nullptr);

    auto sched = llama_wavefront::build_schedule(spec.n_layers, spec.n_tiles);

    std::vector<ggml_tensor *> tile_inputs(spec.n_tiles);
    for (uint32_t t = 0; t < spec.n_tiles; ++t) {
        tile_inputs[t] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, spec.d_model, spec.tile_size);
    }

    std::vector<ggml_tensor *> wq(spec.n_layers);
    std::vector<ggml_tensor *> wk(spec.n_layers);
    std::vector<ggml_tensor *> wv(spec.n_layers);
    std::vector<ggml_tensor *> wo(spec.n_layers);
    for (uint32_t l = 0; l < spec.n_layers; ++l) {
        wq[l] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, spec.d_model, spec.d_model);
        wk[l] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, spec.d_model, spec.d_model);
        wv[l] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, spec.d_model, spec.d_model);
        wo[l] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, spec.d_model, spec.d_model);
    }

    // 4 máscaras por tile
    std::vector<ggml_tensor *> masks(spec.n_tiles);
    for (uint32_t t = 0; t < spec.n_tiles; ++t) {
        const uint32_t n_accum = (t + 1) * spec.tile_size;
        masks[t] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_accum, spec.tile_size);
    }

    std::vector<ggml_tensor *> tile_outputs;
    struct ggml_cgraph * gf = llama_wavefront::build_wavefront_computation_graph(
        ctx, spec, sched, tile_inputs, tile_outputs, wq, wk, wv, wo, masks);

    CHECK(gf != nullptr);
    CHECK(ggml_graph_size(gf) >= 2208); // P2: L48/K4 requer mais que o default GGML de 2048
    CHECK(ggml_graph_n_nodes(gf) > 0);
    CHECK(tile_outputs.size() == spec.n_tiles);

    ggml_free(ctx);
}

static void test_serving_policy_and_two_slot_schedule() {
    g_case = "real_serving_schedule";
    CHECK(llama_wavefront::parse_serving_tile_size(nullptr) == 0);
    CHECK(llama_wavefront::parse_serving_tile_size("0") == 0);
    CHECK(llama_wavefront::parse_serving_tile_size("64") == 64);
    CHECK(llama_wavefront::parse_serving_tile_size("128") == 128);
    CHECK(llama_wavefront::parse_serving_rpc_batch(nullptr) == 1);
    CHECK(llama_wavefront::parse_serving_rpc_batch("4") == 4);
    for (const char * invalid : { "", "1", "32", "256", "64x", " 64", "-64" }) {
        bool threw = false;
        try { llama_wavefront::parse_serving_tile_size(invalid); } catch (const std::invalid_argument &) { threw = true; }
        CHECK(threw);
    }
    for (uint32_t batch : { 1u, 4u }) {
        for (uint32_t groups : { 1u, 3u, 16u }) {
            for (uint32_t tiles : { 2u, 3u, 4u }) {
                const auto execution = llama_wavefront::build_serving_schedule(groups, tiles, batch);
                CHECK(execution.size() == 2*groups*tiles);
                std::vector<bool> prepared(groups*tiles, false), complete(prepared.size(), false);
                std::deque<uint32_t> outstanding;
                uint32_t peak = 0, envelopes = 0;
                for (size_t si = 0; si < execution.size(); ++si) {
                    const auto & step = execution[si];
                    const uint32_t id = step.group*tiles + step.tile;
                    CHECK(step.group < groups && step.tile < tiles);
                    if (step.action == llama_wavefront::serving_step::phase::PREPARE) {
                        CHECK(!prepared[id]);
                        CHECK(step.group == 0 || complete[id - tiles]);
                        CHECK(step.tile == 0 || prepared[id - 1]);
                        CHECK(outstanding.size() < (batch == 4 ? 4u : 2u));
                        if (batch == 4) {
                            for (uint32_t active : outstanding) CHECK(active/tiles != step.group);
                            if (si == 0 || execution[si - 1].action == llama_wavefront::serving_step::phase::COMPLETE) {
                                CHECK(outstanding.empty());
                                ++envelopes;
                            }
                        } else ++envelopes;
                        outstanding.push_back(id);
                        prepared[id] = true;
                        peak = std::max<uint32_t>(peak, uint32_t(outstanding.size()));
                    } else {
                        CHECK(!outstanding.empty() && outstanding.front() == id);
                        CHECK(prepared[id] && !complete[id]);
                        outstanding.pop_front();
                        complete[id] = true;
                    }
                }
                CHECK(outstanding.empty());
                CHECK(std::all_of(complete.begin(), complete.end(), [](bool done) { return done; }));
                CHECK(peak <= (batch == 4 ? 4u : 2u));
                if (groups == 16 && tiles == 4) CHECK(envelopes == (batch == 4 ? 19u : 64u));
                if (groups == 16 && tiles == 2) CHECK(envelopes == (batch == 4 ? 17u : 32u));
            }
        }
    }
    llm_graph_params p{};
    p.wavefront_tile_size = 64;
    p.ubatch = make_test_ubatch(256, 102400, 0);
    auto q = p;
    CHECK(p.allow_reuse(q));
    q.wavefront_tile_size = 128;
    CHECK(!p.allow_reuse(q));
    q = p; q.ubatch = make_test_ubatch(256, 102656, 0);
    CHECK(p.allow_reuse(q)); // data changed; no position is baked into tile metadata shapes
    q = p; q.cparams.rkva_wavefront_rpc_batch = 4;
    CHECK(!p.allow_reuse(q));
}

static void test_reused_parent_prefix_input() {
    g_case = "reused_serving_prefix_graph";
    constexpr uint32_t P = 1024, CAP = 3072, N = 256, TILE = 64;
    auto * cpu = ggml_backend_cpu_init();
    auto * ctx = ggml_init({ 1u << 20, nullptr, true });
    CHECK(cpu && ctx);
    auto * parent = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, CAP);
    ggml_set_input(parent);
    auto * gf = ggml_new_graph_custom(ctx, 256, false);
    std::vector<std::unique_ptr<llm_graph_input_wavefront_prefix>> adapters;
    std::vector<ggml_tensor *> outputs;
    for (uint32_t t = 0; t < N/TILE; ++t) {
        auto * child = ggml_dup_tensor(ctx, parent);
        ggml_set_input(child);
        adapters.push_back(std::make_unique<llm_graph_input_wavefront_prefix>(parent, child, P, t*TILE, TILE));
        auto * out = ggml_cast(ctx, child, GGML_TYPE_F32);
        ggml_build_forward_expand(gf, out);
        outputs.push_back(out);
    }
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    CHECK(buffer != nullptr);
    // Same graph, same INPUT tensors, two adjacent parent ubatches followed
    // by a shorter restored prefix. Poison old bytes to detect stale exposure.
    for (uint32_t old_history : { 512u, 768u, 256u }) {
        auto ub = make_test_ubatch(N, llama_pos(P + old_history), 0);
        std::vector<int32_t> ids(CAP, -1);
        for (uint32_t i = 0; i < old_history + N; ++i) ids[i] = int32_t(P + i);
        ggml_backend_tensor_set(parent, ids.data(), 0, ids.size()*sizeof(int32_t));
        llm_graph_params params{};
        params.ubatch = ub;
        params.wavefront_tile_size = TILE;
        params.cparams.kv_position_split_p = P;
        for (auto & adapter : adapters) {
            CHECK(adapter->can_reuse(params));
            adapter->set_input(&ub);
        }
        CHECK(ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS);
        for (uint32_t t = 0; t < N/TILE; ++t) {
            std::vector<float> out(CAP);
            ggml_backend_tensor_get(outputs[t], out.data(), 0, out.size()*sizeof(float));
            const uint32_t visible = old_history + (t + 1)*TILE;
            for (uint32_t i = 0; i < CAP; ++i) CHECK(out[i] == (i < visible ? float(P + i) : -1.0f));
        }
    }
    // A malformed validated-list fixture must fail before any RPC can run.
    auto ub = make_test_ubatch(N, P + 256, 0);
    std::vector<int32_t> malformed(CAP, -1);
    for (uint32_t i = 0; i < 512; ++i) malformed[i] = int32_t(P + i);
    malformed[10] += 1;
    ggml_backend_tensor_set(parent, malformed.data(), 0, malformed.size()*sizeof(int32_t));
    bool threw = false;
    try { adapters[0]->set_input(&ub); } catch (const std::runtime_error &) { threw = true; }
    CHECK(threw);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(cpu);
}

static void test_real_hybrid_reuse_with_unallocated_general_mask() {
    g_case = "real_hybrid_reuse_unused_general_mask";
    constexpr uint32_t N = 256, P = 1024, C = 4096;
    // No weights or tensor state are needed to exercise the real hybrid
    // ownership/head/rs_z checks. The recurrent memory and hybrid context are
    // production objects; only the KV live extent/tail policy is a test double.
    struct no_weight_model final : llama_model {
        no_weight_model() : llama_model(llama_model_default_params()) {}
        void load_stats(llama_model_loader &) override {}
        void load_hparams(llama_model_loader &) override {}
        void load_vocab(llama_model_loader &) override {}
        bool load_tensors(llama_model_loader &) override { return true; }
        void load_arch_hparams(llama_model_loader &) override {}
        void load_arch_tensors(llama_model_loader &) override {}
        std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params &) const override { return {}; }
    } model;
    struct kv_extent final : llama_kv_cache_context {
        uint32_t rows;
        llama_ubatch ub;
        kv_extent(uint32_t rows, const llama_ubatch & ub) :
            llama_kv_cache_context(LLAMA_MEMORY_STATUS_SUCCESS), rows(rows), ub(ub) {}
        bool apply() override { return true; }
        const llama_ubatch & get_ubatch() const override { return ub; }
        uint32_t get_n_kv() const override { return rows; }
        uint32_t get_tail_slots() const override { return 0; }
        ggml_type get_tail_type() const override { return GGML_TYPE_F16; }
        uint32_t get_tail_tokens() const override { return 0; }
        uint32_t get_tail_arena_stride() const override { return 0; }
        uint32_t get_tail_attention_stride(uint32_t = 0) const override { return 0; }
        uint32_t get_tail_body_execution_stride() const override { return 0; }
        uint32_t get_tail_rollback_tokens() const override { return 0; }
        bool has_compact_tail() const override { return false; }
        llama_kv_tail_storage_kind get_tail_storage_kind() const override { return LLAMA_KV_TAIL_STORAGE_DISABLED; }
        bool can_pack_tail_body(const llama_ubatch &) const override { return false; }
    };
    const auto make_ub = [](llama_pos pos) {
        auto ub = make_test_ubatch(N, pos, 0);
        for (uint32_t i = 0; i < N; ++i) ub.seq_id[i][0] = 0;
        ub.seq_id_unq[0] = 0;
        ub.seq_idx[5] = -1;
        ub.seq_idx[0] = 0;
        return ub;
    };
    auto recurrent = std::make_unique<llama_memory_recurrent>(model, GGML_TYPE_F32, GGML_TYPE_F32,
            false, 1, 1, 0, nullptr);
    llama_memory_hybrid memory(model, nullptr, std::move(recurrent));
    const auto context = [&](const llama_ubatch & ub, uint32_t rows) {
        return std::make_unique<llama_memory_hybrid_context>(&memory,
                std::make_unique<kv_extent>(rows, ub), std::vector<llama_ubatch>{ub});
    };
    auto warm = context(make_ub(P - N), P);
    CHECK(warm->apply());
    auto first_ub = make_ub(P);
    auto first = context(first_ub, P + N);
    CHECK(first->apply());
    auto * cpu = ggml_backend_cpu_init();
    auto * ctx = ggml_init({ 1u << 20, nullptr, true });
    CHECK(cpu && ctx);
    llama_cparams cp{};
    cp.causal_attn = true;
    cp.remote_attn_position_split = true;
    cp.kv_position_split_p = P;
    cp.n_ctx_seq = C;
    cp.n_rs_seq = 0;
    auto attn = std::make_unique<llm_graph_input_attn_kv>(model.hparams, cp, first->get_attn());
    auto * attn_ptr = attn.get();
    attn->self_k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, N);
    attn->self_kq_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, P + N, N, 1, 1);
    attn->self_kq_mask_cnv = attn->self_kq_mask;
    attn->self_kq_mask_local = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, P, N, 1, 1);
    attn->self_kq_mask_local_cnv = attn->self_kq_mask_local;
    attn->self_overflow_cell_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, C - P);
    ggml_set_input(attn->self_kq_mask_local);
    ggml_set_input(attn->self_overflow_cell_idxs);
    auto * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, ggml_scale(ctx, attn->self_kq_mask_local, 1.0f));
    ggml_build_forward_expand(graph, attn->self_overflow_cell_idxs);
    ggml_backend_t backends[] = {cpu};
    auto * sched = ggml_backend_sched_new(backends, nullptr, 1, 64, false, true);
    CHECK(ggml_backend_sched_alloc_graph(sched, graph));
    CHECK(attn->self_kq_mask->buffer == nullptr); // genuinely pruned by the scheduler
    CHECK(attn->self_kq_mask_local->buffer != nullptr);
    CHECK(attn->self_overflow_cell_idxs->buffer != nullptr);
    auto rs = std::make_unique<llm_graph_input_rs>(first->get_recr());
    rs->s_copy = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, first->get_recr()->get_n_rs());
    rs->s_copy_main = ggml_view_1d(ctx, rs->s_copy, 1, 0);
    rs->s_copy_extra = ggml_view_1d(ctx, rs->s_copy, 0, sizeof(int32_t));
    rs->head = first->get_recr()->get_head();
    rs->rs_z = first->get_recr()->get_rs_z();
    llm_graph_input_mem_hybrid input(cp, std::move(attn), std::move(rs), first.get());
    auto next_ub = make_ub(P + N);
    auto next = context(next_ub, P + 2*N);
    CHECK(next->apply());
    llm_graph_params params{};
    params.arch = LLM_ARCH_QWEN35;
    params.cparams = cp;
    params.ubatch = next_ub;
    params.mctx = next.get();
    params.backend_remote = cpu; // opaque identity only; this test never runs an RPC
    params.wavefront_tile_size = 64;
    CHECK(input.inp_rs->head == next->get_recr()->get_head());
    CHECK(input.inp_rs->rs_z == next->get_recr()->get_rs_z());
    CHECK(input.can_reuse(params)); // old implementation fails here solely on general.ne0
    auto changed = params;
    changed.wavefront_tile_size = 128;
    CHECK(input.can_reuse(changed));
    changed = params; changed.wavefront_tile_size = 0;
    CHECK(!input.can_reuse(changed)); // legacy still checks the complete-cache extent
    changed = params; changed.cparams.remote_attn_position_split = false;
    CHECK(!input.can_reuse(changed));
    changed = params; changed.arch = LLM_ARCH_QWEN35MOE;
    CHECK(!input.can_reuse(changed));
    changed = params; changed.cparams.kv_position_split_p += 256;
    CHECK(!input.can_reuse(changed));
    changed = params; changed.ubatch.n_tokens /= 2;
    CHECK(!input.can_reuse(changed));
    // A changed prefix capacity must not be hidden by the unused general mask.
    changed = params; changed.cparams.n_ctx_seq += 256;
    CHECK(!input.can_reuse(changed));
    // A real allocated general mask remains a topology dependency.
    auto * mask_buffer = ggml_backend_alloc_buffer(cpu, ggml_nbytes(attn_ptr->self_kq_mask));
    CHECK(mask_buffer != nullptr);
    CHECK(ggml_backend_tensor_alloc(mask_buffer, attn_ptr->self_kq_mask,
            ggml_backend_buffer_get_base(mask_buffer)) == GGML_STATUS_SUCCESS);
    CHECK(!input.can_reuse(params));
    changed = params; changed.mctx = first.get();
    CHECK(input.can_reuse(changed)); // the allocated mask matches the old KV extent
    ggml_backend_buffer_free(mask_buffer);
    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_free(cpu);
}

struct recurrent_fixture_result {
    std::vector<float> output;
    std::vector<float> conv_state;
    std::vector<float> gdn_state;
};

// Actual CPU SSM_CONV + GATED_DELTA_NET operators with a nonzero initial
// history and a shared persistent destination. The next tile consumes CPY's
// RESULT, as the serving model does; a bare view of the cache is insufficient.
static recurrent_fixture_result run_recurrent_commit_fixture(uint32_t tile_size, bool perturb_future) {
    constexpr int64_t T = 256, D = 16, HV = 2, C = D*(2 + HV), K = 4;
    auto * ctx = ggml_init({ 32u << 20, nullptr, false });
    CHECK(ctx != nullptr);
    auto * gf = ggml_new_graph_custom(ctx, 4096, false);
    auto * raw = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, T, 1);
    auto * kernel = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, C);
    auto * conv_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K - 1, C, 1);
    auto * state_cache = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, D, HV, 1);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t c = 0; c < C; ++c) {
            static_cast<float *>(raw->data)[t*C + c] =
                    0.1f*std::sin(float(7*t + 3*c + 1)*0.043f) + (perturb_future && t >= 128 ? 0.7f : 0.0f);
        }
    }
    for (int64_t i = 0; i < ggml_nelements(kernel); ++i) static_cast<float *>(kernel->data)[i] = float(i%7 - 3)*0.07f;
    for (int64_t i = 0; i < ggml_nelements(conv_cache); ++i) static_cast<float *>(conv_cache->data)[i] = float(i%11 - 5)*0.01f;
    for (int64_t i = 0; i < ggml_nelements(state_cache); ++i) static_cast<float *>(state_cache->data)[i] = float(i%13 - 6)*0.005f;
    ggml_tensor * conv_commit = conv_cache;
    ggml_tensor * gdn_commit = state_cache;
    ggml_tensor * joined = nullptr;
    for (int64_t begin = 0; begin < T; begin += tile_size) {
        const int64_t count = std::min<int64_t>(tile_size, T - begin);
        auto * tile = ggml_view_3d(ctx, raw, C, count, 1, raw->nb[1], raw->nb[2], size_t(begin)*raw->nb[1]);
        auto * input = ggml_concat(ctx, ggml_reshape_3d(ctx, conv_commit, K - 1, C, 1), ggml_transpose(ctx, tile), 0);
        auto * last = ggml_view_3d(ctx, input, K - 1, C, 1, input->nb[1], input->nb[2], size_t(count)*input->nb[0]);
        conv_commit = ggml_cpy(ctx, last, conv_cache);
        ggml_build_forward_expand(gf, conv_commit);
        auto * mixed = ggml_silu(ctx, ggml_ssm_conv(ctx, input, kernel));
        auto component = [&](int64_t dim, int64_t heads, int64_t offset) {
            return ggml_view_4d(ctx, mixed, dim, heads, count, 1,
                    size_t(dim)*sizeof(float), mixed->nb[1], mixed->nb[2], size_t(offset)*sizeof(float));
        };
        auto * q = ggml_l2_norm(ctx, component(D, 1, 0), 1e-6f);
        auto * k = ggml_l2_norm(ctx, component(D, 1, D), 1e-6f);
        auto * v = component(D, HV, 2*D);
        auto * gates = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, HV, count, 1);
        auto * beta = ggml_dup_tensor(ctx, gates);
        for (int64_t i = 0; i < ggml_nelements(gates); ++i) {
            static_cast<float *>(gates->data)[i] = -0.05f;
            static_cast<float *>(beta->data)[i] = 0.25f;
        }
        auto * result = ggml_gated_delta_net(ctx, q, k, v, gates, beta,
                ggml_reshape_4d(ctx, gdn_commit, D, D, HV, 1), 1);
        auto * output = ggml_view_2d(ctx, result, D*HV, count, size_t(D*HV)*sizeof(float), 0);
        auto * next_state = ggml_view_4d(ctx, result, D, D, HV, 1,
                size_t(D)*sizeof(float), size_t(D*D)*sizeof(float), size_t(D*D*HV)*sizeof(float),
                size_t(D*HV*count)*sizeof(float));
        gdn_commit = ggml_cpy(ctx, next_state, state_cache);
        ggml_build_forward_expand(gf, gdn_commit);
        joined = joined ? ggml_concat(ctx, joined, output, 1) : output;
    }
    ggml_build_forward_expand(gf, joined);
    CHECK(ggml_graph_compute_with_ctx(ctx, gf, 1) == GGML_STATUS_SUCCESS);
    recurrent_fixture_result result;
    const auto copy = [](ggml_tensor * tensor) {
        const auto * values = static_cast<const float *>(tensor->data);
        return std::vector<float>(values, values + ggml_nelements(tensor));
    };
    result.output = copy(joined);
    result.conv_state = copy(conv_cache);
    result.gdn_state = copy(state_cache);
    ggml_free(ctx);
    return result;
}

static void test_production_recurrent_commit_causality() {
    g_case = "production_conv_gdn_commit_chain";
    const auto full = run_recurrent_commit_fixture(256, false);
    for (uint32_t tile : { 64u, 128u }) {
        const auto tiled = run_recurrent_commit_fixture(tile, false);
        const auto future = run_recurrent_commit_fixture(tile, true);
        const auto compare = [](const std::vector<float> & a, const std::vector<float> & b, size_t n) {
            CHECK(a.size() == b.size());
            for (size_t i = 0; i < n; ++i) {
                CHECK(std::isfinite(a[i]) && std::isfinite(b[i]));
                CHECK(std::abs(a[i] - b[i]) <= 2e-5f);
            }
        };
        compare(full.output, tiled.output, full.output.size());
        compare(full.conv_state, tiled.conv_state, full.conv_state.size());
        compare(full.gdn_state, tiled.gdn_state, full.gdn_state.size());
        compare(tiled.output, future.output, size_t(128*16*2));
        bool changed = false;
        for (size_t i = 128*16*2; i < tiled.output.size(); ++i) changed |= std::abs(tiled.output[i] - future.output[i]) > 1e-4f;
        CHECK(changed);
    }
}

int main() {
    std::printf("[test-wavefront-pipeline] Iniciando bateria de testes da Fase 7...\n");

    test_serving_policy_and_two_slot_schedule();
    test_reused_parent_prefix_input();
    test_real_hybrid_reuse_with_unallocated_general_mask();
    test_production_recurrent_commit_causality();
    test_tile_size_validation();
    test_wavefront_schedule_properties();
    test_topological_order_validation();
    test_partitioning_basic_64_128();
    test_partitioning_unaligned_tail();
    test_position_split_boundary_p_invariants();
    test_recurrent_protected_window_preservation();
    test_causal_mask_matrix_generation();
    test_kv_cache_tracker_and_kvarn_sealing();
    test_progressive_prefill_tracker();
    test_ggml_wavefront_computation_graph();
    test_causal_mask_leakage_and_future_perturbation();
    test_monolithic_reference_equivalence();
    test_graph_capacity_and_scaling();

    std::printf("[test-wavefront-pipeline] Total de checagens: %d, Falhas: %d\n", g_checks, g_failures);
    if (g_failures > 0) {
        std::fprintf(stderr, "[test-wavefront-pipeline] FALHA: %d checagens falharam!\n", g_failures);
        return 1;
    }

    std::printf("[test-wavefront-pipeline] SUCESSO: Todas as checagens passaram com 100%% de conformidade.\n");
    return 0;
}
