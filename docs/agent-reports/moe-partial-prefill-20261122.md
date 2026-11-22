# MoE cache: prefill parcial em grupos que não cabem na VRAM

Data: 2026-11-22 · branch `feat/moe-partial` (de `1fb324283`).
Alvo: Qwen3.8-Flash-Next IQ3_S (~95 GB) em A100-40G, onde só ~35 GB de experts
cabem no banco do cache e ~60 GB ficam no host.

Flag nova, default-off: `LLAMA_MOE_CACHE_PARTIAL_PREFILL=1`.

## 1. Problema

O cache de experts (`llama_moe_cache`) tem dois caminhos:

1. **slot map** para lotes pequenos e para grupos completos: o grafo lê o
   `slot_map` no host, o callback do scheduler preenche o banco e o MMQ lê o
   banco por índice de slot (`src/llama-graph.cpp:2743`,
   `src/llama-graph.cpp:3619-3658`).
2. **bypass** para lotes grandes que não usam o slot map: o grafo lê os experts
   do host e o scheduler copia só os experts usados para um `dst` de dispositivo
   (`src/llama-context.cpp:6302-6419`). O `copy_experts` já copia D2D os experts
   residentes (`src/llama-moe-cache.cpp:778-818`).

O caminho 1 não serve para um grupo parcial com `n_tokens > 32`: o MMQ precisa
que **todos** os experts selecionados naquele lote estejam em slots, e um ubatch
de prefill seleciona praticamente todos os 512 experts de cada camada. O
`moe_cache_admit` atual rejeita o grupo parcial nesse caso
(`src/llama-moe-cache.cpp:306-309`):

```cpp
return n_tokens > 0 && std::min(n_tokens*n_expert_used, n_expert) <= n_slots &&
    (n_tokens <= 32 || (prefill && (int64_t) n_layers*n_expert <= n_slots));
```

O caminho 2 então só aproveita o que o decode/small-batch deixou no banco. Num
prefill puro o banco começa vazio e **nunca é preenchido**, então os ~60 GB de
experts host são reenviados H2D a cada ubatch.

## 2. Desenho

### 2.1 Admissão

`moe_cache_admit` continua decidindo apenas o uso do **slot map**. A admissão
nova é separada porque o grupo parcial não pode entrar no grafo com slot map:

```cpp
static constexpr bool moe_cache_admit_partial_prefill(
        int64_t n_tokens, int64_t n_expert, size_t n_layers, int64_t n_slots, bool enabled) {
    return enabled && n_tokens > 32 && n_slots > 0 &&
        n_slots < (int64_t) n_layers*n_expert;
}
```

`src/llama-moe-cache.cpp:315-319`, com `static_assert`s em
`src/llama-moe-cache.cpp:343-346` cobrindo flag on/off, `n_tokens` 32/33, grupo
parcial e grupo completo. A flag é lida em
`llama_moe_cache::partial_prefill_enabled()` (`src/llama-moe-cache.cpp:1068-1071`).

### 2.2 Quota por camada

O grupo tem `n_slots` slots compartilhados por `L` camadas. Para evitar que um
scan completo de prefill faça o LRU expulsar exatamente o que o próximo ubatch
vai usar (scan cíclico = 0% de hit), o caminho parcial reserva uma cota estável:

```
quota[il] = n_slots/L + (índice da camada < n_slots % L ? 1 : 0)
```

`moe_cache_partial_quota` em `src/llama-moe-cache.cpp:322-325`, atribuída no
construtor em `src/llama-moe-cache.cpp:585`. A soma das cotas é sempre
`n_slots`. Quando todos os experts de uma camada são selecionados, os
residentes viram hits e os ausentes são bypassados sem evicção — o conjunto
residente fica estável entre ubatches.

### 2.3 Política `plan_partial`

`src/llama-moe-cache.cpp:205-272`:

- deduplica os ids (ids repetidos contam uma vez, como no caminho pequeno);
- promove os hits no SLRU;
- para cada miss, enquanto `n_cached[il] < quota[il]`, usa um slot livre; se não
  houver, evita um slot de outra camada (só acontece se o small path encheu o
  grupo fora da cota);
- com a cota cheia, evita um expert **da mesma camada** que este request não
  seleciona; se todos os residentes da camada estão selecionados, não há vítima
  e o miss é bypassado;
- misses bypassados continuam sendo copiados H2D direto para o `dst` pelo
  caminho existente. Não há mudança no grafo nem na ordem de soma.

O livro-caixa `n_cached` e a lista de slots livres são mantidos também pelo
`plan` do caminho pequeno (`src/llama-moe-cache.cpp:153-198`), então os dois
caminhos compartilham o mesmo SLRU sem corromper as contagens.

### 2.4 Preenchimento e upload

`impl::prepare_layer` (`src/llama-moe-cache.cpp:831-903`) roda no callback do
scheduler, uma vez por camada por grafo:

1. `begin_batch()` é chamado no início de cada `graph_compute`
   (`src/llama-context.cpp:6285-6288`); `layer_batch[il]` evita replanejar a
   mesma camada nas três cópias (gate/up/down).
2. `plan_partial` decide os fills e atualiza o slot map do host.
3. Para cada fill, sobe **todos** os tensores da camada para o mesmo slot, num
   único `llama_moe_upload` (`src/llama-moe-cache.cpp:876-899`). Runs de experts
   consecutivos em slots consecutivos viram um só upload. Isso mantém a
   invariante de que um slot `(il, expert)` é o mesmo expert em gate, up e down.
4. O destrutor de `llama_moe_upload` faz o join (stream de upload com
   `LLAMA_MOE_UPLOAD_OVERLAP=1` ou o stream principal sem a flag); só depois o
   `copy_experts` faz o D2D banco→`dst`.
5. Os misses que não couberam são copiados H2D direto para o `dst` pelo loop
   existente de `sched_copy_experts`.

O upload do banco usa o mesmo `llama_moe_upload` do caminho de bypass
(`src/llama-moe-cache.cpp:1048-1067`), portanto `LLAMA_MOE_UPLOAD_OVERLAP=1`,
`LLAMA_MOE_PINNED_RING=1` e o ring pinned do CUDA valem sem código novo
(`ggml/src/ggml-cuda/ggml-cuda.cu:3534-3627`).

### 2.5 Ordem e segurança

- **Flag off:** `prepare_layer` retorna na primeira linha; `plan`, `copy` e
  `copy_experts` são os mesmos; os `upload.set` do `dst` têm os mesmos offsets e
  tamanhos. `begin_batch`, `ids_flat` e os contadores são os únicos acréscimos.
- **Slot evicted antes do D2D:** o `evict` limpa `slot_map[il][expert]` antes de
  reutilizar o slot. Um `copy_experts` posterior vê `-1` e faz upload direto, em
  vez de ler conteúdo errado.
- **D2D em voo vs. fill posterior:** com overlap, o primeiro upload de cada
  escopo grava um evento no stream principal e o stream de upload espera por
  ele; o stream principal já contém os D2D anteriores. Sem overlap, tudo está no
  stream principal. Um fill nunca sobrescreve um slot antes do D2D que o lê.
- **`no_alloc`:** se `g.slru.slot_map[il] == nullptr`, `prepare_layer` retorna
  false (`src/llama-moe-cache.cpp:846`); nenhum eval acontece nesse modo.
- **`LLAMA_MOE_MAP_DEVICE=1`:** `prepare_layer` só atua em grupos sem device map
  (`src/llama-moe-cache.cpp:846`); grupos integrais imutáveis continuam iguais.

### 2.6 Diff conceitual

```diff
+ if (partial_flag && n_tokens > 32 && group is partial && slot_map != nullptr) {
+     // once per layer per graph
+     plan_partial(layer, ids, quota[layer]);       // hits, fills, bypass
+     upload_all_tensors(layer, fills);             // H2D to the bank, overlap/ring
+ }
  for each used expert run:
      n = copy_experts(bank -> dst);                // D2D residents (unchanged)
      upload(dst, host -> dst);                     // H2D bypass (unchanged)
```

Sem a flag, o bloco novo não executa.

## 3. Arquivos e linhas

| arquivo | linhas | mudança |
|---|---:|---|
| `src/llama-moe-cache.cpp` | 40-95 | contagem `n_cached`, lista de slots livres, `evict` |
| `src/llama-moe-cache.cpp` | 153-198 | `plan` mantém as contagens novas |
| `src/llama-moe-cache.cpp` | 205-303 | `plan_partial`, vítimas por camada/fora da cota |
| `src/llama-moe-cache.cpp` | 306-353 | admissão parcial e `static_assert`s |
| `src/llama-moe-cache.cpp` | 585 | cotas por camada no construtor |
| `src/llama-moe-cache.cpp` | 821-918 | `begin_batch`, `prepare_layer`, contadores |
| `src/llama-moe-cache.cpp` | 1068-1071 | leitura da flag |
| `src/llama-moe-cache.h` | 28-73 | stats, flag, `prepare_layer` |
| `src/llama-moe-cache.h` | 92-105 | hooks de teste da política |
| `src/llama-context.cpp` | 6285-6412 | `begin_batch`, `ids_flat`, `prepare_layer`, contagem |
| `src/llama-context.h` | 333, 518-528 | acesso de diagnóstico, `ids_flat` |
| `tests/test-llama-archs.cpp` | 1920-2241 | modo `--test-moe-cache-partial` |
| `tests/CMakeLists.txt` | 670-675 | ctest `test-moe-cache-partial` |

## 4. Testes

### 4.1 Build CPU

```bash
cmake -S . -B build-cpu -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
  -DPython3_EXECUTABLE=/usr/bin/python3
cmake --build build-cpu -j2 --target test-llama-archs test-strata-opt
```

Saída: build concluído; avisos só de arquivos não tocados por este patch
(`common/speculative.cpp`, `src/llama-kv-cache-kvarn.cpp`, `common/common.cpp`,
`common/exl3-dir.cpp`), nenhum em `llama-moe-cache.cpp`/`llama-context.cpp`.

### 4.2 ctest

```bash
ctest --test-dir build-cpu -R 'llama-archs|strata|moe' --output-on-failure
```

```
1/4 Test #106: test-moe-cache-partial ...........   Passed    0.16 sec
2/4 Test #107: test-strata-trace ................   Passed    0.93 sec
3/4 Test #108: test-strata-prefill ..............   Passed    0.99 sec
4/4 Test #283: test-strata-opt ..................   Passed    0.04 sec

100% tests passed out of 4
```

No build CPU (sem backend GPU) o `test-moe-cache-partial` roda os testes de
política e imprime:

```
test_moe_cache_partial: planner checks done, no GPU backend, skipping the MoE cache data path
```

Os casos CPU cobrem: admissão flag on/off, 32/33 tokens, grupo parcial/completo,
ids repetidos, expert ausente/evicção, contadores `hit+fill+bypass == used` e a
invariante `fills <= used - hits` (o total H2D do caminho com cache é
`fills + (used - hits - fills) = used - hits <= used`).

### 4.3 Regressão deepseek2

```bash
./build-cpu/bin/test-llama-archs -a deepseek2 -s 12345
```

```
|       deepseek2|AMD Ryzen Z1 Extreme (CPU)|   MoE|  OK (0.00e+00)|       OK|  OK (7.68e-10)|
Summary: all 1 test(s) passed
```

### 4.4 Caminho de dados em GPU (Vulkan, validação extra)

Como o host não pode compilar CUDA, o caminho real de `sched_copy_experts` +
`copy_experts` foi validado num build **Vulkan** (`-DGGML_VULKAN=ON
-DGGML_CUDA=OFF`, `test-llama-archs`), na iGPU AMD RADV Phoenix. O mesmo
`src/llama-context.cpp` e o mesmo `src/llama-moe-cache.cpp` são usados; só o
backend de dispositivo muda.

```bash
cmake -S . -B build-vulkan -DGGML_VULKAN=ON -DGGML_CUDA=OFF -DGGML_NATIVE=ON \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_UI=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan -j2 --target test-llama-archs test-strata-opt
./build-vulkan/bin/test-llama-archs --test-moe-cache-partial -s 12345
```

Fixture deepseek2: 1 camada MoE, 64 experts, top-2, experts host, cache de
2 MiB = **6 slots** (9,4% do grupo). Saída real:

```
prepare_layer: partial prefill layer 1: hits = 0, fills = 2, bypass = 0, quota = 6
prepare_layer: partial prefill layer 1: hits = 2, fills = 0, bypass = 0, quota = 6
test_moe_cache_partial: router=biased NMSE ref/off=0.000e+00 ref/on=0.000e+00 off/on=0.000e+00 | off copy=1.12 MiB | on hits=2 fills=2 bypass=0 fill=0.56 MiB copy=0.00 MiB

prepare_layer: partial prefill layer 1: hits = 0, fills = 6, bypass = 3, quota = 6
prepare_layer: partial prefill layer 1: hits = 6, fills = 0, bypass = 2, quota = 6
test_moe_cache_partial: router=random NMSE ref/off=0.000e+00 ref/on=0.000e+00 off/on=0.000e+00 | off copy=4.80 MiB | on hits=6 fills=6 bypass=5 fill=1.69 MiB copy=1.42 MiB

test_moe_cache_partial: small batch NMSE=0.000e+00, partial batches=0
test_moe_cache_partial: two sequences NMSE=0.000e+00, hits=2 fills=2 bypass=0
test_moe_cache_partial: prefill+decode NMSE ref/on=1.526e-16 ref/off=1.526e-16 off/on=0.000e+00, hits=0 fills=2 bypass=0
```

Leitura:

- **NMSE 0** nos casos enviesado e aleatório nas três comparações (sem cache,
  flag off, flag on), e também NMSE 0 no lote pequeno e nas duas sequências
  contra o caminho sem cache. Os ids de roteamento e a ordem de soma são os
  mesmos; só a origem das linhas de peso muda.
- **Aleatório/parcial:** o primeiro ubatch de 64 tokens usou 9 experts
  distintos: 6 preencheram o banco e 3 foram bypassados; o segundo usou 8, com
  **6 hits** e 2 bypass. O conjunto residente ficou estável (quota 6).
- **Bytes H2D:** flag off 4,80 MiB; flag on 1,69 MiB (fills) + 1,42 MiB (misses
  diretos) = **3,11 MiB ≤ 4,80 MiB**. No caso enviesado: 1,12 MiB → 0,56 MiB.
- **Lote pequeno:** `partial batches=0`; continua no slot map, NMSE 0.
- **Duas sequências:** NMSE 0 e o caminho parcial ativo.
- **Prefill + decode no mesmo contexto:** o lote de 64 tokens preenche o grupo e o
  lote seguinte de 2 tokens volta ao slot map; NMSE 1,5e-16 contra o caminho sem
  cache, e `off/on=0` (a flag não alterou os logits).

O teste também passou com `LLAMA_MOE_UPLOAD_OVERLAP=1
LLAMA_MOE_PINNED_RING=1` (no Vulkan o proc CUDA não existe, então o upload cai
no `set_async`; a fiação é a mesma do caminho existente).

Regressão da arquitetura no mesmo build Vulkan:

```
|       deepseek2|AMD Ryzen Z1 Extreme (RADV PHOENIX) (Vulkan0)|   MoE|  OK (9.30e-08)| OK| OK (0.00e+00)|
|       deepseek2|            NVIDIA GeForce RTX 4070 (Vulkan1)|   MoE|  OK (5.07e-07)| OK| OK (0.00e+00)|
|       deepseek2|                   AMD Ryzen Z1 Extreme (CPU)|   MoE|  OK (0.00e+00)| OK| OK (7.68e-10)|
|       deepseek2|                                    MoE cache|   MoE|  OK (7.52e-10)| OK| OK (0.00e+00)|
|       deepseek2|                       MoE cache, layer split|   MoE|  OK (7.52e-10)| OK| OK (0.00e+00)|
Summary: all 6 test(s) passed
```

O `ctest` do build Vulkan (`-R 'llama-archs|strata|moe'`) também roda
`test-strata-trace` e `test-strata-prefill`, que falharam por
`vk::Device::allocateMemory: ErrorOutOfDeviceMemory` no Vulkan1 (RTX 4070 já
com 11,8 GB de 12,2 GB usados por outro processo). Não é regressão deste patch:
os dois testes passam no build CPU e o `test-moe-cache-partial` passa isolado no
Vulkan. Não foi feito A/B contra o binário pré-patch de
`/home/hjotha/beellama.cpp/build` porque ele carrega `libllama` de
`build-optimized` e falha com `undefined symbol:
_ZN17llama_model_saverC1E8llm_archP12gguf_context`; o checkout principal não foi
modificado.


## 5. O que ficou de fora

- **Build CUDA/A100 não executado** (regra do host: 14 GB de RAM, build sm_80
  caro). Não há número de performance deste patch; o ganho deve ser medido pelo
  pai numa A100.
- **Grupo integral sem `LLAMA_MOE_CACHE_PREFILL=1`**: a flag nova é só para
  grupos parciais. Grupo integral continua usando o slot map com
  `LLAMA_MOE_CACHE_PREFILL=1` ou o bypass antigo sem ela.
- **`LLAMA_MOE_MAP_DEVICE=1`**: não se aplica a grupo parcial (o mapa é imutável
  e exige `n_slots == n_layers*n_expert`).
- **`no_alloc`**: o guard foi adicionado, mas não há eval nesse modo.
- **Performance**: os contadores são bytes lógicos de H2D/D2D; não medem PCIe,
  overlap real nem t/s.
- **Escolha do subconjunto**: o preenchimento inicial é determinístico por id
  (os primeiros `quota` experts selecionados, em ordem crescente). Não pondera
  a frequência de roteamento; num prompt de domínio específico, uma política
  por frequência pode render mais hits. Fica para o A/B na A100.

## 6. Próximo passo

Na A100-40G, com IQ3_S e `-ub 2048`, rodar o mesmo prompt duas vezes (primeiro
ubatch frio, segundo quente) comparando:

```bash
LLAMA_MOE_CACHE_PARTIAL_PREFILL=0 ...
LLAMA_MOE_CACHE_PARTIAL_PREFILL=1 ...
LLAMA_MOE_CACHE_PARTIAL_PREFILL=1 LLAMA_MOE_UPLOAD_OVERLAP=1 ...
LLAMA_MOE_CACHE_PARTIAL_PREFILL=1 LLAMA_MOE_UPLOAD_OVERLAP=1 LLAMA_MOE_PINNED_RING=1 ...
```

Critérios: mesmos ids/logits (NMSE <= 1e-6 contra o caminho sem cache), H2D por
ubatch caindo de ~50 GB para ~(60/95)·50 GB depois do primeiro ubatch, VRAM
dentro do orçamento e decode pós-prefill >= 97% do baseline. O log
`llama_moe_cache: partial prefill: ... uploaded = ...` e a linha
`large batch ... fills the partial MoE cache` confirmam o caminho.
