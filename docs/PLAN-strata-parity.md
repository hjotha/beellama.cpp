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
   (`ggml-cuda.cu:1681-1685`) e o loader aloca todos os experts num buffer só. Em modelos
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
  em lotes > 32 tokens, então o prefill de 28k envia ~39,7 GB de experts do host; esse
  H2D domina o custo e mascara o ganho da atenção esparsa.

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
  para o draft fazer o catch-up uma vez; esse é o custo.
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
