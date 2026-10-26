#pragma once

// Wavefront Causal Pipelining and Progressive Prefill (Fase 7).
//
// Documentação: docs/hybrid-attention-master-review.md §§2.3, 4.4, 5.3.
//
// Em prefill longo de lotes causais, a dependência temporal impede que camadas
// futuras de um token sejam calculadas antes da camada atual concluir o merge
// e FFN. No entanto, particionando o ubatch em tiles causais no tempo
// (ex: blocos de 64 ou 128 tokens), o pipelining em wavefront (efeito cascata)
// permite que a Camada L processe o Tile T+1 enquanto a Camada L+1 processa
// o Tile T.
//
// Invariantes fundamentais:
// 1. Causalidade estrita: qualquer query em um tile T só atende a tokens em
//    posições <= sua própria posição. Tokens em tiles futuros T' > T têm peso
//    estritamente nulo (-inf na máscara de atenção).
// 2. Fronteira P: nenhum tile pode cruzar a fronteira de particionamento P.
//    Tiles com posições < P residem em range LOCAL (KVarN4 CUDA). Tiles com
//    posições >= P residem em range OVERFLOW (remote/companion RKVA).
// 3. Janela protegida recorrente: as (1 + n_rs_seq) posições finais devem
//    permanecer indivisíveis em um único tile final para manter válidos os
//    snapshots de rollback.
// 4. Ordenação topológica: no grafo de computação, nó (L, T) depende
//    estritamente de (L-1, T) [ativação vertical] e de (L, T-1) [KV/recorrente
//    horizontal]. Todas as dependências pertencem a ondas anteriores (W < L+T).
// 5. Serviço progressivo e overlap de camadas: o Tile 0 completa a Camada L-1
//    na onda W = L-1, habilitando progresso e overlap contínuo entre
//    computação e transferência de camadas. Contudo, o primeiro token de resposta
//    gerado depende estritamente do término do último tile (K-1) na onda W = L+K-2.
//    O pipeline em wavefront reduz o tempo exposto por overlap (profundidade teórica
//    L+K-1 vs serial L*K com trabalho total L*K), mas NÃO reduz o TTFT da resposta
//    completa para O(L).

#include "llama.h"
#include "llama-batch.h"
#include "llama-position-split.h"

#include <cstdint>
#include <string>
#include <vector>
#include <memory>

struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;

namespace llama_wavefront {

// Serving MVP policy. This selects a topology; it is not a claim of a
// throughput gain or a reduction of the per-token remote dependency depth.
uint32_t parse_serving_rpc_batch(const char * value); // unset/1 or 4; throws on invalid values
uint32_t parse_serving_tile_size(const char * value); // unset/0, 64, 128 or 256; throws on invalid values
struct serving_step {
    enum class phase { PREPARE, COMPLETE } action;
    uint32_t group;
    uint32_t tile;
};
std::vector<serving_step> build_serving_schedule(uint32_t groups, uint32_t tiles, uint32_t rpc_batch = 1);


inline constexpr uint32_t kDefaultTileSize = 128; // Alinhado ao grupo KVarN
inline constexpr uint32_t kSmallTileSize   = 64;  // Alinhado a warp tiles
inline constexpr uint32_t kMinTileSize     = 16;
inline constexpr uint32_t kMaxTileSize     = 512;

// Metadados de um tile causal no tempo
struct causal_tile {
    uint32_t tile_id = 0;              // Índice sequencial 0 .. n_tiles - 1
    uint32_t token_offset = 0;        // Offset inicial de token no ubatch pai
    uint32_t n_tokens = 0;            // Quantidade de tokens neste tile
    llama_pos pos_min = 0;            // Menor posição temporal (inclusiva)
    llama_pos pos_max = 0;            // Maior posição temporal (inclusiva)
    llama_position_split::range_kind range = llama_position_split::range_kind::LOCAL;
    bool is_protected_window = false; // Contém a janela protegida recorrente (1 + n_rs_seq)
};

// Coordenada de execução na grade 2D (Camada L, Tile T)
struct wavefront_coord {
    uint32_t layer = 0;               // Índice da camada 0 .. L - 1
    uint32_t tile = 0;                // Índice do tile 0 .. K - 1
    uint32_t wave = 0;                // Índice da onda = layer + tile
    uint32_t task_id = 0;             // Identificador único = layer * n_tiles + tile

    int32_t dep_layer_prev = -1;      // Predecessor vertical (layer - 1, tile), -1 se layer == 0
    int32_t dep_tile_prev  = -1;      // Predecessor horizontal (layer, tile - 1), -1 se tile == 0
};

// Escalonamento em ondas da grade 2D (L x K)
struct wavefront_schedule {
    uint32_t n_layers = 0;
    uint32_t n_tiles = 0;
    uint32_t n_waves = 0;             // n_layers + n_tiles - 1
    uint32_t total_tasks = 0;         // n_layers * n_tiles
    uint32_t max_concurrency = 0;     // min(n_layers, n_tiles)
    std::vector<std::vector<wavefront_coord>> waves; // waves[w] = tarefas da onda w
};

// Plano completo de particionamento e wavefront de um ubatch
struct wavefront_plan {
    uint32_t tile_size = kDefaultTileSize;
    uint32_t n_tiles = 0;
    uint32_t n_layers = 0;
    uint32_t p = 0;                   // Fronteira de particionamento por posição
    uint32_t n_keep_tail = 0;         // Janela protegida recorrente
    std::vector<causal_tile> tiles;
    std::vector<llama_ubatch> ubatch_tiles;
    wavefront_schedule schedule;
};

// Exact production shape guards for opt-in parent512/1024, physical tile256.
bool is_serving_n256_plan(const wavefront_plan & plan);
bool is_serving_2x256_plan(const wavefront_plan & plan);

// Rastreador de serviço progressivo de prefill
struct progressive_tracker {
    uint32_t n_layers = 0;
    uint32_t n_tiles = 0;
    uint32_t n_waves = 0;
    uint32_t current_wave = 0;
    std::vector<bool> tile_completed; // true quando o tile completou a camada L-1
    uint32_t tokens_completed = 0;

    void init(uint32_t layers, uint32_t tiles);
    bool step_wave(uint32_t wave_idx, const std::vector<causal_tile> & tiles,
                   std::vector<uint32_t> & newly_completed_tiles);
    bool is_tile_ready(uint32_t tile_idx) const;
    bool all_completed() const;
    uint32_t wave_for_tile_completion(uint32_t tile_idx) const;
};

// Validação de tamanho de tile
bool validate_tile_size(uint32_t tile_size);

// Particiona um ubatch em tiles causais respeitando máscaras, posições, P e janela recorrente
bool partition_ubatch_causal_tiles(
    const llama_ubatch & ubatch,
    uint32_t tile_size,
    uint32_t n_layers,
    uint32_t p,
    uint32_t n_keep_tail,
    wavefront_plan & plan,
    std::string & error);

// Constrói o escalonamento em ondas 2D (Wave = L + T)
wavefront_schedule build_schedule(uint32_t n_layers, uint32_t n_tiles);

// Valida a ordenação topológica de uma lista de execução de tarefas
bool validate_topological_order(
    const wavefront_schedule & sched,
    const std::vector<uint32_t> & execution_order,
    std::string & error);

// Constrói e valida a matriz de máscara causal para a composição de tiles
bool generate_causal_mask(
    const std::vector<causal_tile> & tiles,
    uint32_t total_tokens,
    std::vector<float> & mask_out, // size: total_tokens * total_tokens, formato row=q, col=k
    std::string & error);

// Rastreador de estado do KV cache (KVarN local e overflow remoto)
struct kv_cache_tracker {
    uint32_t n_layers = 0;
    uint32_t p = 0;
    uint32_t initial_history_local = 0;
    uint32_t initial_history_remote = 0;
    std::vector<uint32_t> committed_rows_local;    // Por camada: linhas salvas < P neste lote
    std::vector<uint32_t> committed_rows_remote;   // Por camada: linhas salvas >= P neste lote
    std::vector<int32_t>  last_committed_tile;     // Por camada: índice do último tile salvo
    std::vector<bool>     kvarn_group_sealed;      // Por camada: grupo de 128 linhas selado

    void init(uint32_t layers, uint32_t p_val, uint32_t hist_local = 0, uint32_t hist_remote = 0);
    bool record_tile_kv_commit(uint32_t layer, const causal_tile & tile, std::string & error);
    bool can_layer_attend_tile(uint32_t layer, const causal_tile & tile, std::string & error) const;
};

// Especificação e execução de protótipo de grafo de computação GGML
struct wavefront_graph_spec {
    uint32_t n_layers = 2;
    uint32_t n_tiles = 2;
    uint32_t tile_size = 64;
    int64_t  d_model = 64;
};

// Constrói um grafo GGML em wavefront e executa a computação progressiva
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
    const std::vector<ggml_tensor *> & masks);

} // namespace llama_wavefront
