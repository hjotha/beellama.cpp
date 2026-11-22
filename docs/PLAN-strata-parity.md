# Plano: paridade Strata ↔ beellama (offload de experts)

Status: **P0/P1 implementados nesta branch** (`feat/strata-parity-moe-autocache`).
Referência lida: `github.com/Niko1221/Strata` (MIT) — `docs/HOW_IT_WORKS.md`,
`docs/DETAILS.md` (Memory modes, Expert cache budget, Draft layer), `engine/src/core/*`,
mais o GGUF `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` (2 shards; só o shard 1 precisa
residir).

## O que o beellama JÁ tem (verificado no código)

- `llama_moe_cache` (`src/llama-moe-cache.{h,cpp}`, 698 linhas): banco de experts por
  device, **slot map** por camada, **SLRU** (probation/protected), upload assíncrono no
  miss (`ggml_backend_tensor_set_async`), bypass para lotes grandes, orçamento por device
  dividido como o `--tensor-split`.
- `--n-cpu-moe`, `--cpu-moe`, `--n-cpu-ffn`, `-ot` por padrão de tensor.
- `--fit/-fitt/-fitc`: mede a memória livre por device e move camadas/experts para host.
- `--hf-stream-vram`, KVarN4 + tiers de contexto, `--remote-attn`, snapshots, EXL3.
- MTP com perfis adaptativos e `--spec-type ngram-*` (base do prompt-lookup).

## O que falta para igualar a Strata (gaps reais)

| # | Gap | Strata | beellama hoje | Esforço |
|---|---|---|---|---|
| G1 | Tamanho do cache automático | `--expert-cache auto`: lê VRAM livre, dimensiona e encolhe se não couber | só valor fixo em MiB | **P1 (esta branch)** |
| G2 | Experts em RAM **pinned** | arena mlock/`cudaHostAlloc`, `STRATA_RESIDENT_PIN`, `--pcie-frac` | tensores host em buffer CPU paginável | P2 (médio) |
| G3 | **Compute da CPU no miss** | CPU calcula o expert que não está na GPU, **em paralelo** com a GPU | miss é **upload** por PCIe e o cálculo é na GPU | P3 (grande) |
| G4 | Perfil adaptativo por conversa | `--adapt-every`, decay, swap de pares; `--adapt-async` | SLRU estático por grupo | P3 |
| G5 | Modos de pouca RAM | `--mmap-experts`, `--resident-experts`, `--resident-budget-gib` | só `mmap` genérico do GGUF | P4 (médio) |
| G6 | KV com janela residente | `--kv-streaming`, `--kv-resident 32768`, `--kv-grow` | tiers por faixa de contexto + remote-attn | P4 (médio) |
| G7 | Prefetch da próxima camada | "experts of the next layer stream over PCIe while attention runs" | não existe | P5 (médio) |
| G8 | Draft por subconjunto de vocabulário | `--draft-vocab en\|latin\|cyrillic`, `--spec-min-p` | `--spec-draft-p-min` + ngram | P5 (pequeno) |
| G9 | Planner de instalação | escolhe low-RAM, `--vram-reserve-mib`, `--calibrate` | `--fit` | P2 |

## P1 — cache de experts com dimensionamento automático (implementado)

- `--moe-cache-mib -1` (ou `auto`) passa um sentinela; `llama-context` resolve **depois**
  dos pesos carregados:
  1. soma os bytes dos experts **host-residentes** (checa `ggml_backend_buft_is_host`);
  2. soma a VRAM livre de cada device e reserva 1,5 GiB de margem;
  3. `cache = min(livre - margem, bytes_host)`; abaixo de 256 MiB, desliga. A resolução
     acontece **depois** do primeiro `sched_reserve()`, então a VRAM medida já desconta os
     buffers de compute; com cache, um segundo `sched_reserve()` reserva o grafo de novo;
  4. loga o plano (`MoE cache auto: ...`).
- Guardas: nunca resolve em contexto `no_alloc` (medição do `--fit`) nem quando não há
  expert em host (evita o `throw` do construtor com `n_tensors_host == 0`).
- Critério de aceite: com modelo MoE e experts em host, o log mostra o plano, o cache
  entra em uso (hit rate no fim) e o servidor responde; com modelo denso, loga
  "nenhum expert em host" e segue sem cache.
- **Verificado em 2026-11-20** (RTX 4070, OLMoE-1B-7B q2_k, `-cmoe -ngl 99`):
  `MoE cache auto: 2224.00 MiB for 2224.00 MiB of host experts (100.0% of the experts
  can be cached)` → `CUDA0 MoE cache size = 2224.00 MiB` → hit rate 68,76% em
  `ubatch <= 8` (372 hits/169 misses) e 27,22% em lote grande. Modelo denso
  (`stories15M`, `-ngl 0`): `no host-resident MoE experts, cache disabled`, sem crash.
  `test-arg-parser` cobre `-1`, valor fixo e `-2` rejeitado.

## P2 — pinned + planner (feito, com limite conhecido)

1. ~~`--vram-reserve-mib N`~~ (default 1536) usado pelo auto-size do P1. **Feito e testado**
   (reserve 20000 MiB desliga o cache: "only 0.00 MiB ... free for the cache").
2. ~~`--moe-host-pinned` / `--no-moe-host-pinned`~~ (default: ligado). Os overrides de expert
   (`-cmoe`, `-ncmoe`, `--fit`) que apontam para o buffer CPU passam a apontar para o
   host buffer do device (`CUDA_Host`, `cudaMallocHost`), e o loader deixa de rebaixá-los
   para pageable quando o mmap está ligado (`llama_model_params::keep_host_pinned`).
   **Verificado**: `CUDA_Host model buffer size = 2256.24 MiB` com a flag e `CPU_Mapped`
   com `--no-moe-host-pinned`; decode 34,3/34,4/34,3 t/s vs 33,9/33,8/33,9 t/s
   (OLMoE-1B-7B q2_k, `-cmoe --moe-cache-mib 1024`, 4070) → **+1,5%**.
3. **Limite**: o host buffer CUDA tem `max_size` de 4 GiB por alocação
   (`ggml-cuda.cu:1688-1691`) e o loader aloca todos os experts num buffer só. Em modelos
   com >4 GiB de experts em host (Flash-Next no 40G: ~17 GiB) a alocação cai para pageable.
   Para pinar nessa escala é preciso uma arena em blocos (`cudaHostRegister` sobre ranges),
   que é o mesmo trabalho de P3/P4.
4. `--moe-plan`: substituído pelo log `MoE cache plan: <buft> = N MiB` no start.

## P3 — o item grande (não é port de arquivo)

Para o miss ser **computado na CPU em paralelo** em vez de enviado por PCIe é preciso:
- dividir `ids` em `hit`/`miss` no host (o slot map já é copiado para o host no callback);
- dois `MUL_MAT_ID` (banco da GPU e tensor host) + soma dos parciais;
- executar o subgrafo da CPU **concorrentemente** com o da GPU.

O scheduler do ggml é sequencial por grafo (só há pipeline com múltiplos devices), então
isso é uma mudança de motor, não uma flag. Estimativa: 3–4 semanas de trabalho dedicado,
com A/B em A100-40G. **Não portar o `engine/` da Strata**: são ~211k LOC de um motor
próprio; o caminho é implementar os mecanismos no ggml.

## P4/P5 — resumo

- G4 (perfil adaptativo): trocar o SLRU por contadores com decay + janela de swap, e um
  `--adapt-async` usando um thread auxiliar entre janelas de decode.
- G5/G6: reaproveitar `TENSOR_READ_LAZY` (já usado no PLE) para expert arena mmap e para
  janela KV residente; `--kv-resident` já é quase o `TIER_*_CTX` existente.
- G7: prefetch é um `ggml_backend_tensor_set_async` especulativo da próxima camada,
  agendado antes da atenção — baixo risco, ganho depende do hit rate.
- G8: `--spec-draft-p-min` já cobre o gate; falta o subconjunto de vocabulário do draft.

## Ordem e critérios

1. P1 (feito) → build + `test-moe-*` + smoke real.
2. P2 → A/B de tps com pinned; só mergear com ganho >= 3% ou empate e sem regressão.
3. P3 → protótipo atrás de `--moe-cpu-compute`; aceite = hit rate baixo (< 50%) sem queda
   de decode vs upload, em A100-40G com IQ3_S.
4. P4/P5 → por último; cada um com teste próprio e métrica no `/metrics`.

## Status 2026-11-21 (fase 1/2)

Estado no commit `128411b6f` da branch `feat/strata-parity`. A A100-40G está medindo
performance; esta seção registra só o que já foi validado e não antecipa números.

### Fase 1 — QSA esparsa com KV Q8_0 (implementada)

- `src/llama-graph.cpp`: o corpo KV local, com `q8_0/q8_0`, sem tail e fora do KVarN,
  passa a habilitar o caminho esparso e a propagar `n_kv_max`. Antes só F16 entrava;
  `q8_0` caía na conversão densa para F16 e varria o KV inteiro.
- `ggml/src/ggml-cuda/fattn.cu`: dispatch `q8_direct` restrito a D=256, `ncols2=8`,
  Ampere+ e `ncols1=1|8`. O helper `ggml_cuda_flash_attn_ext_uses_sparse_q8` espelha o
  switch de `ncols1`/`ncols2` no cálculo do buffer, para que os fallbacks densos
  (`ncols1=2/4`, GQA desalinhado) mantenham o buffer de conversão F16.
- `ggml/src/ggml-cuda/fattn-mma-f16.cuh`: o gather da linha esparsa dequantiza 8 valores
  `q8_0` por chunk (`dequantize_V_q8_0<half, 8>`) direto para a tile, de forma síncrona
  (sem `cp.async`); instanciado em D=256, `ncols1=1` e `8`, `ncols2=8`, Ampere+.

**Teste**: 28/28 testes da matriz QSA em `test-backend-ops` passaram no CUDA local
(RTX 4070, build sm_80), incluindo KV 4096/16384/32768, `nq` 1/5/512, `nq=2/3/4`,
`nq=13`, sem máscara e tail 1–3. O caso GQA ratio 2 (rota densa, `ncols2=2`) foi
validado separadamente, 1/1.

**Medição na A100-40G** (3 runs, `b2048/ub1024`, cache 6144, KV q8_0):

| config | short tg | pp4k | decode 4k | pp28k | decode 28k |
|---|---:|---:|---:|---:|---:|
| base-q8 (release) | 55,73 | 1.484,5 | 46,04 | 1.513,9 | 39,62 |
| QSA q8 (branch) | 55,59 | 1.485,8 | 46,40 | 1.516,4 | **43,53** |

- Prefill em 28k: **+0,2%** (dentro do ruído); decode após 28k: **+9,9%**.
- O alvo de **+25% no prefill longo não foi atingido**. O cache de experts é bypassado
  em lotes > 32 tokens nesse benchmark. O volume de referência é ~39,7 GB de
  experts enviados no prefill de 28k. O A/B de residência e o trace devem medir quanto
  desse H2D fica no caminho crítico.

### Correção do baseline (`MEASURED.md`, 21/11)

A tabela de 21/11 foi medida com **F16** (default), não `q8_0`; a recipe
`a100-40g/qwen38-fn-q20-beellama` usa `q8_0`. O baseline `q8_0` medido na A100
(`b2048/ub1024/cache6144`, 2 runs da sessão anterior):

| métrica | valor |
|---|---:|
| decode curto | 54,09 t/s |
| prefill 4k | 1.505,5 t/s |
| prefill ~28k | 1.528,8 t/s |
| decode pós-28k | 38,90 t/s |

O baseline antigo de ~1.175 t/s (prefill) era `ub512`; não é comparável ao `ub1024`.

### Fase 2 — MTP com sidecar (sem reconverter os 66 GB)

O GGUF Q2_0 publicado não tem `blk.48.nextn.*`, então não há cabeça MTP no alvo. O sidecar
pronto é `unsloth/Qwen3.8-Flash-Next-GGUF` →
`MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf` (1.907.151.936 B, sha256
`f521868a9e143718bef513772f6e04d9642551e362cf2439636d2abdbd149dfc`), carregado com
`-md <sidecar> --spec-type draft-mtp` (o auto-discovery não varre `MTP/`, então `-md` é
obrigatório).

- O sidecar tem `compress_ratios[48]=0`; o patch de 1 byte `0→4` **falha ao carregar**
  (metadata inválida para o loader), então o arquivo original é o caminho.
- O cache de experts compartilhado entre contextos foi **removido** da integração por
  risco de corrida entre streams (não validado). Com `-md`, o draft não compartilha o
  cache do alvo.

**Medição na A100-40G** (1 run, cache auto `-1` + reserve 4096, n-max 2):

| config | VRAM | pp4k | decode 4k | pp28k | decode 28k | aceitação |
|---|---:|---:|---:|---:|---:|---:|
| sem spec | 37.550 | 1.484,5 | 46,04 | 1.513,9 | 39,62 | — |
| MTP sidecar | 39.052 | 1.227,2 | **61,47** | 1.293,0 | **58,86** | **80,7%** |

- Decode: **+33%** após 4k e **+49%** após 28k; curto 53,96 (sem ganho).
- Prefill: **-17% / -15%** — o alvo calcula o hidden NextN de todos os tokens do prompt
  para o catch-up do draft. O custo de extração, transferência, execução e espera ainda
  precisa ser separado em braços com orçamento de cache pareado.
- Cache fixo 2048/4096 = 40.404 MiB e OOM no primeiro request; n-max 4 falha no load;
  auto + reserve 4096 funciona em 39.052 MiB.
- A recipe fica `experimental` até o caminho da recipe (zip clean) ser verificado; os
  números acima usam o build `parity2`.

### Fase 0 — instrumentação

Os scripts (`profile_prefill.sh`, `ablate_lazy.py`, `summarize_nsys.py`) estão em
`collabosm/tools/profiling/`, validados em dry-run e com SQLite sintético. O achado que
muda a ordem: o cache de experts é bypassado em lotes > 32 tokens, então o prefill de
28k envia ~39,7 GB de experts do host. As fases 3 e 4 continuam **gated por evidência**;
o primeiro experimento é permitir o cache no prefill quando o working set do grupo cabe
(protótipo em andamento), seguido do overlap H2D se o profiling confirmar.

### Build A100 (nota curta)

O build CUDA de referência agora é feito na `.57` (gokaya) em chroot Ubuntu 24.04
(glibc 2.39, OpenSSL 3.0, GCC 13.3), para casar com a VM. Binários linkados contra a
glibc 2.43 do host não carregam na VM.

```bash
cmake -S <src> -B <build> -DGGML_CUDA=ON -DGGML_CUDA_FA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined
cmake --build <build> -j
```

- usar diretório de build **novo**; nunca empacotar zip de build incremental
  (`docs/FN40-CRASH-ROOTCAUSE.md`);
- o CMake 3.28 do Ubuntu 24.04 não adiciona o include do CCCL: passe
  `-DCMAKE_CUDA_FLAGS=-I/opt/cuda/targets/x86_64-linux/include/cccl` e garanta que
  `/usr/include/cccl` exista dentro do chroot (o toolkit do host aponta para lá);
- `-Wl,--allow-shlib-undefined` porque o CUDA do chroot traz libcublas 13.8;
- antes de subir o zip, fazer o smoke de load com os dois headers + `truncate`
  (mesmo doc).

## Revisão do plano Kimi (Codex, 2026-11-21)

Veredito: o plano segue útil como diagnóstico e controle de regressão, mas a ordem de implementação
mudou. Relatório completo em `docs/agent-reports/kimi-plan-review-20261121.md`.

1. **Prefill 28k primeiro**: parear `LLAMA_MOE_CACHE_PREFILL=0/1` sem MTP, Q8/Q8, cache fixo 6144,
   b2048/ub1024, `-ncmoe 2`, mmap/lazy on; confirmar 100% dos slots, o log de lote grande, H2D
   frio/repetido, +10% de pp28k e decode pós-28k ≥97% do braço pareado.
2. **Overlap depois**: `LLAMA_MOE_UPLOAD_OVERLAP=0/1` com cache 1024 (misses) e 6144 + prefill
   (residente), medindo H2D por stream e espera de evento separadamente do throughput, graphs on/off.
3. **Decode**: repetir QSA e MTP em braços pareados; isolar `LLAMA_MTP_RESERVE_WEIGHTS` e
   `LLAMA_MTP_KV_STORE_ONLY`; manter o sidecar original e n-max 2.
4. **Fora da fila imediata**: port Q8, reformulação GDN, MTP embutido neste GGUF, n-max 4, patch
   ratio 4, cache compartilhado do draft, CPU no miss e prefetch SLRU. DFlash2 fica em trilha separada.

## Oportunidades adicionais (Strata × beellama, 2026-11-21)

Manter a sequência revisada. Primeiro confirmar a residência de experts no prefill de 28k. Depois medir o overlap com misses residuais. Parear o MTP antes de otimizar o caminho especulativo. As propostas novas entram quando o perfil mostrar custo residual. A [revisão de código](agent-reports/strata-code-review-20261121.md) traz as referências em Strata `3142b66` e Bee `3d0f99732`, além das diferenças entre o benchmark e o clone.

A Strata pré-compilada subiu de 2.394,7 para 4.629,3 t/s entre 4k e 28k com `--prefill auto`, INT8 e cache de experts de 24.576 slots reportados. No Bee com QSA Q8_0, foram 1.485,8 e 1.516,4 t/s com b2048/ub1024. Chunks maiores, reaproveitamento de pesos e agrupamento de experts são hipóteses verificáveis para a escala da Strata. O binário histórico não tem trace de dispatch disponível. Em decode, os 85,9/67,4/75,0 t/s da Strata incluem uma configuração com DFlash2. Os 58,86 t/s do Bee MTP pós-28k vieram de uma execução separada com auto-cache e reserva de 4096 MiB. A comparação ainda mistura estratégias especulativas e orçamentos.

**Protocolo dos próximos A/B.** Usar a mesma revisão e GPU, os mesmos dois shards, prompts e tokens reais, contexto 32768, Q8/Q8, b2048/ub1024, `-ncmoe 2`, mmap/lazy on, greedy e `cache_prompt=false`. Para os itens de prefill, começar sem MTP. Rodar pelo menos três pares alternados com profiler desligado para throughput, e sessões separadas com trace para atribuição. Registrar bytes e tempo descoberto de H2D/D2H por fase e camada, primeiro pedido e repetição, TTFT, pico em MiB e flags/placement efetivos. Para aceitar uma otimização nova, exigir que a métrica do item melhore além da dispersão dos pares, pp4k/pp28k e decode curto/longo ≥97% nos casos fora do alvo, nenhum OOM e correção conforme a fixture indicada. S, M e L são ordens de esforço. Ganhos descritos são mecanismos ou metas de aceite para A/B futuro.

### Prefill

| Prioridade | Oportunidade e status | Evidência | Ganho esperado | Esforço | Risco | Teste e aceite |
|---|---|---|---|---|---|---|
| 1 | Residência de experts no pp28k · coberto, implementado sob flag | ~39,7 GB de H2D de referência. Bee `src/llama-moe-cache.cpp:495`. Strata exclui residentes da lista em `engine/src/prefill/prefill.cpp:4622`. | Remover fills recorrentes em grupos completos. | S | Capacidade/padding do banco, mudança no grafo. | `LLAMA_MOE_CACHE_PREFILL=0/1`, cache fixo 6144, verificar 100% dos slots, log de lote grande, primeira carga de até um working set e H2D de pesos próximo de zero nas repetições para os grupos integrais. Exigir pp28k ≥110% e decode pós-28k ≥97%. Exercitar ub32/33, grupo parcial e NMSE ≤1e-4 na fixture. |
| 2 | Overlap de upload · coberto, implementado sob flag. Staging pinned limitado · novo, condicionado | Strata ring + thread emissora `engine/src/prefill/prefill.cpp:4660,4683`. Bee H2D direto e espera por split em `ggml/src/ggml-cuda/ggml-cuda.cu:3524`, host buffer com teto em `:1688`. | Recuperar apenas espera de cópia descoberta e bloqueio de submissão. | S para o A/B, M para staging | Reuso antes do DMA, RAM pinned e graphs. | Parear `LLAMA_MOE_UPLOAD_OVERLAP=0/1` com 1024 MiB e misses, mais controle residente 6144, graphs on/off. Exigir sobreposição no trace ou ≥3% nas medianas com misses. Se pageable ainda bloquear a submissão, testar ring pequeno com o mesmo volume transferido, menos tempo descoberto e pp28k acima do ruído. Testar wrap e fallback pageable. |
| 3 | Mapa imutável para grupo integral · novo | Strata agrupa com endereços conhecidos em `engine/src/prefill/prefill.cpp:5592,5712`. Callback Bee ainda baixa IDs, sincroniza e sobe o mapa com todos os experts residentes (`src/llama-moe-cache.cpp:588,624`). | Remover ida ao host por camada no quente. Reutiliza o banco e MMQ atuais. | M | Publicação prematura, lookup obsoleto e bind de graph. | Preencher e publicar pesos/mapa uma vez no device para grupos completos. Comparar cache já aquecido com callback atual. Exigir zero D2H/sync de roteamento e zero H2D recorrente de pesos nesses grupos, mesmos IDs/logits greedy na fixture e pp28k acima da dispersão. Registrar tempo/VRAM adicionais a frio. |
| 4 · se MMID aparecer no trace | Reusar agrupamento e entrada MMQ entre projeções compatíveis · novo | Strata agrupa no GPU e prepara entrada conjunta (`engine/src/prefill/moe_fused.cu:744`, `engine/src/prefill/prefill.cpp:6322`, trecho MMQ legado). Bee já deduplica por token dentro de cada chamada (`ggml/src/ggml-cuda/mmq.cu:393`) e suporta gate/up empacotado (`src/models/qwen4exp.cpp:1312`). | Menos agrupamentos e quantizações repetidas. | M | Strides, tipo, padding e ordem das linhas. | Identificar primeiro os tipos e as projeções realmente separadas no GGUF. Reusar mapa e Q8 de gate/up de layout igual. Para down, reusar só ordenação onde compatível. Testar roteamento uniforme e enviesado, grupo frio/quente. Exigir menos launches/tempo de preparo, pp28k acima do ruído e paridade de IDs e NMSE na fixture MoE. |
| 4 · se indexer aparecer no trace | Limites de visibilidade compactos no indexer · novo | Strata limita o score por sub-batelada (`engine/src/prefill/prefill.cpp:5299`). Bee já calcula `[p0,p0+nv)` e expande máscara de pools (`src/llama-memory-hybrid-idx.cpp:1031`), com scorer tiled de quatro cabeças (`ggml/src/ggml-cuda/lightning-indexer.cu:700`). | Menos preenchimento/upload da máscara e tiles totalmente inválidos. | M | Limites causais, padding ou sequência errados. | Carregar dois I32 por query na rota qualificada e preservar a aritmética e -inf fora do intervalo. A/B em 4k/28k, prefixo e duas sequências. Exigir mesmos scores válidos/IDs, menos bytes H2D da máscara e menor tempo combinado de `set_input` + indexer. Comparar pp28k e exigir pico de VRAM ≤ controle. |
| 4 · se máscara/FA aparecer no trace | IDs de QSA direto à FA · novo. Gather esparso Q8 já implementado | Strata fornece `sel_ids` à atenção (`engine/src/prefill/prefill.cpp:5320,5485`). Bee expande seleção em máscara e a varre para compactar por tile (`src/models/qwen4exp.cpp:994`, `ggml/src/ggml-cuda/fattn.cu:14`). | Menos trânsito de máscara densa e um passo de compactação mais curto. | L | Visibilidade por query, overflow da união, cauda e fallback. | Rota inicial CUDA Q8 local sm80, sem KVarN nem tail KV explícita. Emitir IDs/contagem e pertinência por query para tiles 1/8. Exigir os mesmos conjuntos válidos por query, fallback no overflow e NMSE da matriz QSA em KV 4k/16k/32k, incluindo cauda incompleta, GQA fallback e duas sequências. Medir menos bytes/tempo de seleção→FA e pico ≤ controle. |
| 4 · se preparação GDN aparecer no trace | Fundir L2 de Q/K no conv+SiLU · novo detalhe da investigação GDN já prevista | Strata CONV-L2 (`engine/src/prefill/kernels.cu:3001`). Bee conv+SiLU já fundidos (`ggml/src/ggml-cuda/ggml-cuda.cu:5931`), L2 separado (`src/models/qwen4exp.cpp:1281`), scan MMA disponível. | Eliminar materializações e launches de L2 de Q/K. | M | Arredondamento e estado recorrente. | Isolar epílogo mantendo V/estado/dispatch. Comparar split de ubatches, duas sequências, rollback e replay. Exigir estado/logits dentro da tolerância das fixtures GDN e `GATED_DELTA_NET_CACHE_FUSION` de `test-backend-ops`, menor tempo/bytes na preparação e pp28k fora da dispersão. |

### Decode

| Prioridade | Oportunidade e status | Evidência | Ganho esperado | Esforço | Risco | Teste e aceite |
|---|---|---|---|---|---|---|
| 3 · após o A/B de overlap | MTP sidecar n-max 2 pareado · coberto, reserva e store-only implementados sob flags | 80,7%, 39.052 MiB e prefill −17%/−15% no ensaio isolado. Strata DFlash2 usa taps (`engine/src/program/generate.cpp:23190`). Bee catch-up (`common/speculative.cpp:4394`). | Confirmar ganho líquido de decode e localizar o custo de catch-up. | S | OOM, atribuir ganho ao draft quando placement/cache mudou. | Comparar sem spec e MTP com cache efetivo do alvo controlado, medindo placement, ubatch do draft e pico. No MTP isolar primeiro `LLAMA_MTP_RESERVE_WEIGHTS` e depois `LLAMA_MTP_KV_STORE_ONLY`. Exigir load/primeiro pedido sem OOM, saída greedy equivalente, ganho de decode além do ruído em 4k/28k e, para store-only, catch-up menor, aceitação ±1 p.p. e decode ≥97% do braço MTP completo. Aplicar o teto de 40,9 GB decimais do [A/B MTP](agent-reports/codex-rest-of-plan-20261121.md#ab-de-hardware), medido em pico e reportado também em MiB (~39.005 MiB). |
| 5 · se houver lacunas de submissão | Estabilizar replay CUDA nas larguras reais de verificação · novo | Strata seleciona graph por largura (`engine/src/core/verify.cpp:3626`). Bee invalida por sync no MMID ou propriedades/pool (`ggml/src/ggml-cuda/ggml-cuda.cu:3729,6711`). | Menos launches e idle de host, se replay estiver sendo perdido. | M | Captura com parâmetros ou endereços obsoletos. | Separar métricas CUDA de `graphs reused` ggml. Em 1/2/3 linhas e curto/28k, parear graphs on/off e contar replay, capture e motivos de execução direta. Só alterar o motivo dominante. Exigir menos capturas ou gaps, menor p95 entre tokens, tokens iguais e stress de mudanças de forma/rollback sem replay obsoleto. |
| 5 · se sobrar espera NextN | Hidden do MTP entre alvo e sidecar via D2D · novo | Strata MTP lê residuais no device (`engine/src/core/mtp.cpp:1648`). Bee extrai para host e reenvia estado (`src/llama-context.cpp:5485`, `src/llama-graph.cpp:325`). | Menos D2H/H2D e espera em catch-up e decode, se ambos os inputs estiverem no mesmo device. | L | Vida útil de buffers, carry, ordem por sequência e rollback. | Depois do store-only, medir bytes NextN e espera por fase. Prototipar staging D2D só no mesmo CUDA device e em sequência única contígua. Exigir remoção do tráfego host dessas linhas no trace, catch-up ou latência por rodada menor, logits/hidden dentro da tolerância de `test-mtp-qwen4exp-store-only`, carry e rollback/replay corretos, aceitação ±1 p.p. |

### Não fazer agora

- **Portar o motor, o KV INT8 ou o layer-major completo.** O KV usa outra representação. O escalonamento layer-major muda a vida de residuais e checkpoints. Primeiro testar chunks maiores no sweep já previsto com cache/placement pareados. Revisitar a reordenação apenas com H2D recorrente material em grupos parciais.
- **Emprestar slots do cache ao scratch.** A Strata recarrega os slots antes da verificação. Medir a vida dos temporários de QSA/MMQ no alocador existente antes de introduzir refill no TTFT e movimentar endereços capturados.
- **Prefetch SLRU especulativo ou CPU no miss.** O stream longo da Strata pode antecipar uma lista conhecida ou podá-la depois do router. O router de outra camada não prevê os IDs daqui. Uma nova partição CPU/GPU depende de misses residuais e de vantagem medida sobre upload + GPU.
- **Trocar a matemática do score ou do scan por hipótese.** TF32 compensado, indexer em dois passes, heurísticas que descartam experts e reformulação de GDN ficam fora do primeiro teste. Preservar seleção e estado torna possível atribuir ganhos às mudanças de tráfego.
- **Misturar DFlash2 ao A/B MTP.** Manter a trilha DFlash2 separada. O sidecar Q4_K_M original e n-max 2 são o caminho medido. As variantes n-max 4 e ratio alterado falharam.

## A/B das flags na A100-40G (2026-11-21)

Engine `parity5` (`feat/strata-opt` @ `f5b72f42a`), 2 runs por config, Q2_0, Q8_0/Q8_0,
b2048/ub1024, `-ncmoe 2`, cache 6144 MiB salvo indicação.

| config | short tg | pp4k | pp28k | decode 28k |
|---|---:|---:|---:|---:|
| base-q8 (release) | 48,3* | 1.410,8 | 1.511,3 | 39,63 |
| p0, prefill off | 56,34 | 1.486,1 | 1.514,0 | 43,55 |
| p1, `LLAMA_MOE_CACHE_PREFILL=1` | 56,76 | 1.556,8 | **1.567,4** | 44,11 |
| m1, p1 + `LLAMA_MOE_MAP_DEVICE=1` | **59,72** | 1.565,4 | 1.574,0 | 44,40 |
| MTP, `LLAMA_MTP_RESERVE_WEIGHTS=0` | 55,58 | 1.231,3 | 1.292,2 | 53,14 |
| MTP, `LLAMA_MTP_RESERVE_WEIGHTS=1` | **59,95** | 1.234,6 | 1.292,3 | **57,70** |
| MTP, reserve 1 + store-only | 56,80 | 1.241,9 | 1.311,6 | 58,53 |
| overlap + ring 0/1 (cache 1024) | 56,6–56,8 | 1.500–1.502 | 1.536–1.538 | ~44,1 |
| QSA indexer bounds 0/1 | 59,8/59,8 | 1.564/1.580 | 1.573/1.568 | 44,4/44,3 |
| GDN conv+L2 0/1 | 59,6/60,6 | 1.564/1.567 | 1.573/1.576 | 44,3/44,3 |

\* short da base pegou a primeira execução fria; a segunda foi 56,9.

**Veredito**
- `LLAMA_MOE_CACHE_PREFILL`: **+3,5%** em 28k, +4,8% em 4k, decode estável. Real, mas abaixo do
  limiar de +10% do experimento: com `-ncmoe 2` só duas camadas (~1,5 GB) estão no host. A recipe
  base passa a habilitar a flag.
- `LLAMA_MOE_MAP_DEVICE`: +0,4% prefill, **+5,2% no decode curto**, +0,7% em 28k. Marginal; fica
  desligado nas recipes.
- `LLAMA_MTP_RESERVE_WEIGHTS`: **+8,6% no decode 28k** (53,14 → 57,70) e +7,9% no curto. A recipe
  MTP habilita.
- `LLAMA_MTP_KV_STORE_ONLY`: +1% (ruído); desligado.
- `LLAMA_MOE_UPLOAD_OVERLAP`/`LLAMA_MOE_PINNED_RING`: sem efeito mensurável com cache 1024.
- `LLAMA_QSA_INDEXER_BOUNDS` e `LLAMA_GDN_CONV_L2`: sem efeito mensurável no Flash-Next.
- MTP vs sem spec: decode **+31%** em 28k, **+37%** em 4k, curto +5,6%; prefill **-17%**.
  VRAM 39.052 MiB pronto / 39.740 pico.

A lacuna de prefill para a Strata (4.629 vs 1.567) **não** é explicada pelo H2D das duas camadas
host nem por essas flags. O próximo passo é atribuição por nsys/scheduler
(`GGML_SCHED_SPLIT_TIMING`, `LLAMA_GRAPH_PREP_PROFILE`), não mais flags default-off. Recipes usam
o zip `parity5`; a base com `LLAMA_MOE_CACHE_PREFILL=1` e a MTP com `LLAMA_MTP_RESERVE_WEIGHTS=1`.

## DFlash2, perfil de prefill e PLE (A100, 2026-11-22)

Engine `parity6` (`feat/strata-opt` @ `9f59e300f`), Q2_0, Q8_0/Q8_0, b2048/ub1024,
`-ncmoe 2`, cache 2048 MiB (host-side, não muda VRAM), **`--parallel 1`**, 2 runs por config.
Detalhes e comandos em [strata-a100-dflash2-20261122.md](agent-reports/strata-a100-dflash2-20261122.md).

| config | short tg | pp4k | pp4k+dec tg | pp28k | pp28k+dec tg | VRAM pico |
|---|---:|---:|---:|---:|---:|---:|
| m1-np1 (sem spec) | 60,15 | 1.559,4 | 46,93 | 1.575,1 | 45,09 | 37.838 |
| DFlash2 Q8_0, n-max 7 | 66,83 | 1.341,3 | **89,91** | 1.393,0 | **84,17** | 39.624 |
| DFlash2 Q8_0, n-max 4 | 59,28 | 1.353,5 | 90,33 | 1.399,1 | 79,31 | 39.286 |

**Veredito**
- DFlash2 Q8_0 é a melhor estratégia de decode medida no alvo real: **+91,5%** em 4k e **+86,7%**
  em 28k sobre o alvo sem spec, contra +38%/+33% do MTP sidecar (64,8/58,5 t/s medidos em rodada
  separada, mesmo alvo e GPU). Aceitação implícita ≥1,87 token/passo em n-max 7.
- Custo: prefill **-14,0%** em 4k e **-11,6%** em 28k (o draft também processa o prompt), e +1,7 GiB
  de VRAM. Fica em trilha separada do MTP, como já previsto.
- `--parallel 1` é requisito: com 4 slots o estado recorrente GDN do alvo cresce de 112,6 para
  3.602,3 MiB e o drafter não cabe.
- `--moe-cache-mib` 6144/2048 dão o mesmo VRAM (cache é host-side); a diferença entre 6144 e 2048
  não aparece no throughput medido.
- **Critério de corretude:** "texto idêntico ao sem spec" não vale neste alvo. Só trocar
  `-ub 1024` → `-ub 512` no `m1` sem spec já muda o greedy no 5º token; o greedy não é invariante à
  geometria do batch com Q2_0 + KV q8_0. O aceite correto é top-1 no mesmo batch, NMSE do verify
  contra single-token e aceitação/throughput, como no MTP. Os taps são bit-exatos no alvo (teste
  CPU) e um erro neles reduziria aceitação, não mudaria o argmax do alvo.

**Atribuição do prefill.** `LLAMA_PREFILL_PROFILE=1` só emite com `llama-server -lv 4`; em `-lv 3`
o filtro do server descarta as linhas `LLAMA_LOG_INFO`. Com `-lv 4`, 37 ubatches / 33.177 tokens:
parede do scheduler 15,669 s, dos quais **15,159 s (96,7%) são submissão host** (`CLASS/mixed`),
2,614 s de preparação de grafo e 2,714 s de espera de sync; cópias 0,181 s e 1,595 GB de H2D.
Todos os 74 splits caem em `mixed`, então a separação GDN/QSA/MoE por tempo exige trace de kernels.
O perfil não muda o throughput (pp4k 1.543,8; pp28k 1.574,2).

**PLE.** `LLAMA_PLE_PREFETCH=1` fica default-off: no A/B frio válido
(`posix_fadvise(DONTNEED)`; `/proc/sys/vm/drop_caches` é read-only no container do Colab, carga
sobe de 12,7 para 26,0 s) deu +0,07% (1.550,44 → 1.551,49 t/s), dentro da dispersão de 3,2 t/s.

**Próximo alvo de prefill**, na ordem: reduzir submissões host por token (splits/ubatch), reusar o
plano de grafo entre ubatches de mesma forma (2,6 s de `prep`), e só então otimizar kernels por
classe com trace.

## Prefill: ubatch e teto de compute (A100, 2026-11-22, `feat/strata-prefill`)

Medição com o mesmo alvo/engine, `-b/-ub` variando e o resto fixo (2 runs, pp28k estável):

| config | pp4k | pp28k | pp28k+dec tg | VRAM pronto |
|---|---:|---:|---:|---:|
| `-b 2048 -ub 1024` | 1.449,7\* | 1.575,7 | 40,75 | 37.098 |
| `-b 2048 -ub 2048` | 1.762,9 | **1.831,7** | 44,24 | 37.528 |
| `-b 4096 -ub 4096` | erro 500 | erro 500 | — | 38.382 |

\* primeira execução fria; a segunda deu 1.551,9.

Modelo de dois pontos: **181 ms fixos por ubatch** + **457 µs/token (~2.190 t/s
assintótico)**. Em `-ub 1024` o custo fixo responde por 5,1 s dos 18,1 s do pp28k; por isso
`-ub 2048` rende +16% e o teto com ubatch maior é ~2,0-2,2k t/s. A Strata (4.629 t/s em 28k) está
~2,1× acima desse teto: a diferença restante é compute por token, não agendamento.

`-ub 4096` carrega mas devolve `500 Compute error` em todo request (não é OOM; pico 38.456 MiB) —
limite de geometria/kernel a investigar só se o teto de ubatch importar.

A leitura anterior de "96,7% host" fica corrigida: `device_host_us` inclui sincronizações internas
do nó e, principalmente, **back-pressure da fila CUDA** (o host bloqueia no launch quando a GPU está
atrás). O perfil por op (`LLAMA_OP_PROFILE=1`) mostra `submit_ms` crescendo com nós de `TOP_K`,
`FLASH_ATTN_EXT` e `MUL_MAT` a ~1 ms/nó, sem sync no código do top-k: é o dispositivo que manda.
Conclusão: o prefill é **compute-bound** no teto de ~2,19k t/s; o que resta contra a Strata são
kernels por token, não agendamento.

Os dois patches default-on (poda da busca de fusão em `graph_optimize` e mapa de consumidores GB10
só em DGX Spark) deram **empate técnico** no A/B `parity6` × `parity7` (`-ub 2048`, 3 runs:
1.830,2 vs 1.830,5 t/s no pp28k; 1.766,2 vs 1.768,4 no pp4k) com saída bit-idêntica; ficam como
remoção de trabalho impossível, não como ganho. Detalhes em
[strata-prefill-ub-20261122.md](agent-reports/strata-prefill-ub-20261122.md).
