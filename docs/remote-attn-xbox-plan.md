# Plano: Xbox Series X como backend remoto de KV cache + attention (KVarN4)

**Data:** 2026-09-29
**Repositórios:** `beellama.cpp` (host, gokaya/.57) + `xllama-dflash-xbox` worktree (Xbox/UWP, HPBEATS/.193)
**Status:** plano aprovado pelo usuário; implementação pendente

## Decisões do usuário (2026-09-29)

1. **KVarN4 direto** — sem fase intermediária F16. O lado Xbox já nasce com store comprimido
   (Hadamard + Sinkhorn + records) e attention nativa sobre records. Paridade numérica
   kernel-level continua sendo o primeiro gate antes da integração ponta-a-ponta.
2. **Git:** commitar primeiro o trabalho spec-verify existente (validado na tarefa anterior)
   como commit próprio na branch atual; criar `feat/remote-attn-xbox` a partir desse HEAD;
   commits lógicos pequenos. **Sem merge no main.**
3. **Produção:** parar/restaurar `llama-server` automaticamente nas janelas de teste
   (padrão `dflash-gpu-test.sh`, trap EXIT), avisando cada parada/restauração.

## Objetivo

Qwen3.8-27B (arch `qwen35`, 64 layers, 16 full-attention `il=3,7,...,63`, GQA 24Q/4KV,
head_dim 256, gate pós-attention) com **pesos inteiros na RTX 4070 12GB** e o KV das
full-attention layers (KVarN4 + precision tail) **residente e computado no Xbox Series X**
(~5GB, D3D12/HLSL nativo, RDNA2 52 CUs wave32). O Xbox NÃO é RAM remota: recebe
Q/K_new/V_new por layer e devolve somente o attention output. DeltaNet state, FFN,
projeções, RoPE, gate, sampling: tudo local.

## Diagnóstico da auditoria (código real)

- Ponto de integração: `src/models/qwen35.cpp:398-400` (`build_attn` da full-attention layer).
  Q/K já têm QG-split, Q/K-norm e MRoPE aplicados; gate sigmoid (`qwen35.cpp:403-407`) e
  `wo` (`409-411`) permanecem locais. Variante MTP: `qwen35.cpp:700-733`.
- KVarN: records por grupo de 128 tokens (`KVAR_N_GROUP=128`); tile =
  `k_payload | k_s_col(u16) | k_zp(u16) | k_s_row(u16) | v_payload | v_s_col | v_s_row | v_zp`,
  scales/zp F16, quantização Sinkhorn (`llama-kvarn.cpp:595-649`, kernels `ggml-cuda/kvarn.cu`).
  head_dim 256 → 2 slices×128 (`llama-kvarn.cpp:651-657`). Stage F16 do grupo incompleto +
  `k_tail/v_tail` exatos; política de tail em `llama-kv-cache-kvarn.h:30-49` (tail_groups=2,
  intrinsic=min(128,window)). Attention nativa sobre records: `fattn-kvarn-*` (CUDA) e
  `ggml-vulkan/vulkan-shaders/kvarn_flash_attn.comp` (spec do port HLSL); store:
  `kvarn_store.comp`; Hadamard: `kvarn_wht*.comp`.
- Precedente de pinagem de nó: `ggml_backend_sched_set_tensor_backend` (`llama-graph.cpp:3505`).
  Precedente de ops custom: `GGML_OP_KVARN_{WHT,STORE,VIEW,MATERIALIZE}` (`ggml.h:619-624`).
- `layer_filter_cb` na construção do cache (`llama-model.cpp:3225-3244`) permite não alocar
  KV host das layers remotas.
- DFlash RPC existente (reuso): protocolo DFL2 v2 em `common/dflash-remote.h`, servidor UWP
  `uwp/dflash-rpc/dedicated.cpp` (porta 50053, single-client, framing 24B/56B com timings),
  engine D3D12 `dflash-gpu-engine.*`, KV persistente com trim/reset `dflash-gpu-kv.*`,
  attention HLSL bit-fiel `shaders/dflash/dflash_attention.hlsl` (exp UCRT, sinks, Q4_0),
  Hadamard HLSL `dflash_hadamard128.hlsl`. RTT vazio medido: mediana 560µs / p95 681µs.
  Ciclo de vida do app: threads detach por serviço (`App.cpp Run()`).

## Arquitetura

```
CUDA (4070)                       RPC RKVA v1 (TCP persistente :50054)        Xbox (D3D12, app HJotha.XboxDFlashRpc)
norm → Wqg/Wk/Wv → Q/K-norm → MRoPE
   │
   [GGML_OP_REMOTE_ATTN pinado no backend "remote-attn" via sched]
   │  Q,K_new,V_new,mask,positions ──► HELLO/CREATE_SESSION/ATTN_DECODE/
   │                                   ATTN_PREFILL/TRIM/REWIND/RESET/
   │                                   DESTROY_SESSION/PING/GET_STATS ──►  append K/V no stage F16 (por sessão/layer)
   │                                                                       grupo completo → KVarN store HLSL (Hadamard+Sinkhorn+pack)
   ◄── attention output F16 [n_head*256, n_tokens] ──────────────────────  flash-attn HLSL sobre records + tail exato (GQA 24/4, causal)
   │
sigmoid(gate) ⊙ out → wo → residual → FFN (local)
```

### Host (beellama.cpp)

1. **Op novo** `GGML_OP_REMOTE_ATTN` (registro no padrão KVARN_*): src0=Q, src1=K_new,
   src2=V_new, src3=kq_mask, src4=positions (i32); op_params: layer_id, n_head, n_head_kv,
   n_embd_head, kq_scale, domain, flags. Saída `[n_head*n_embd_head, n_tokens]` F32/F16
   (mesmo contrato do kqv_out de `build_attn`).
2. **Backend ggml dedicado** `ggml/src/ggml-remote-attn/`: device/buffer_type/buffer mínimos,
   buffers host pinned permanentes por conexão (zero malloc por token), compute callback =
   empacotar frame → send → recv → escrever output. Só declara suporte ao op REMOTE_ATTN.
   Conexão persistente TCP_NODELAY + MSG_MORE (estilo `dflash_remote::client`), 1 nó por
   layer por decode (16 RTTs sequenciais/token — inerente à dependência entre layers).
3. **llama-context/cparams:** `llama_context_params.remote_attn{host,port,prefill_mode}` →
   cria backend, adiciona à lista do sched, expõe em `llm_graph_context` (como `backend_cpu`).
   `qwen35.cpp:398`: se remoto ativo para a layer → constrói nó REMOTE_ATTN e pinagem
   `ggml_backend_sched_set_tensor_backend(sched, node, backend_remote)`; senão caminho atual
   intacto. KV host das layers remotas não é alocado (`layer_filter_cb`); bookkeeping de
   posições/trim/rewind/reset por (sessão, sequence) num manager leve que espelha as ops de
   memória (seq_rm/seq_cp/context-shift) em chamadas TRIM/REWIND/RESET.
4. **CLI (common/arg.cpp):** `--remote-attn HOST:PORT`, `--remote-attn-layers full`,
   `--remote-attn-prefill remote|migrate`, `--remote-attn-stats`. `--cache-type-k/v` e
   `--kv-tail-*` continuam fonte de verdade; geometria e tail vão no HELLO. Sem as flags,
   comportamento bitwise intacto.
5. **MTP/DFlash:** primeiro target normal; draft cache continua independente; configuração
   speculative incompatível → falha explícita (nunca resultado silenciosamente errado).

### Protocolo RKVA v1 (header compartilhado, magic próprio, porta 50054)

- Header request/response no padrão DFL2 (magic/version/op/cycle_id/n_tokens/payload_bytes;
  resposta com status + timestamps do Xbox: receive/prepare/compute/response µs).
- HELLO negocia: versão, geometria (n_layer_remote, n_head, n_head_kv, head_dim, max_ctx),
  cache_type_k/v (kvarn4), tail (tokens/tipo), ops suportadas.
- ATTN_DECODE (n_tokens=1, caminho quente): pacote único Q/K/V/positions → output.
  Wire em F16 (Q ~12KB, K/V ~4KB, out ~12KB por layer).
- ATTN_PREFILL: batched, mensagens grandes, chunking configurável.
- TRIM/REWIND/RESET/DESTROY_SESSION/PING/GET_STATS.
- Sem checksum no caminho quente (validação pesada só em modo debug). Sessão inválida após
  reconnect → falha explícita ou reconstrução segura.

### Xbox (worktree `xllama-dflash-xbox` na 193, UWP `DFlashRpc.vcxproj`)

- `remote-attn-protocol.h` (espelho do header host), `remote-attn.{h,cpp}` (listener próprio
  em thread detach, single-client, jsonl logging), `remote-attn-engine.{h,cpp}` (D3D12:
  device/queue/allocators/heaps/fences/upload-ring/readback pré-alocados por sessão;
  nada criado por token), shaders em `shaders/dflash/` (ou dir novo `shaders/rkva/`):
  - `rkva_kvarn_store.hlsl` — port de `kvarn_store.comp` (Hadamard 128, Sinkhorn, pack
    record com scales/zp u16); reusa padrões de `dflash_hadamard128.hlsl` e quantização
    GGML-exata já provada no console.
  - `rkva_flash_attn.hlsl` — port de `kvarn_flash_attn.comp`: causal, GQA 24/4, head 256
    (2 slices×128), leitura linear coalesced dos records (evitar regime latency-bound ~20GB/s;
    alvo: streaming 145-165GB/s), merge do tail exato, exp/softmax fiéis ao ggml
    (reaproveitar `ggml_exp_neg`/`ggml_float.hlsli`), wave32-safe (sem WaveReadLaneAt lane>31).
  - `rkva_materialize.hlsl` — fallback/corretude (port de `kvarn_materialize.comp`), usado
    só em testes de paridade e prefill se necessário; nunca no decode normal.
- Memória por sessão (kvarn4, 16 layers): ~1,1-1,2KB/posição → 128K ≈ 2,4GB; 256K ≈ 4,7GB
  (teto ~5GB: 256K é a meta limite, validar na escada).
- Build/deploy: `build-dflash-rpc.cmd` → signtool (thumbprint
  `CD6D10EC8332955BC5C1074BC20A297D0625F1F4`) → `xbox_portal.py --upload` →
  start/stop via `/api/taskmanager/app`; bump de versão no `Package.appxmanifest`.
  Serviço DFL2 (50053) intocado.

## Ordem de implementação e gates

1. **Git:** commit spec-verify pendente (commit próprio) → branch `feat/remote-attn-xbox`.
2. **Protocolo:** header RKVA compartilhado (host `common/remote-attn-remote.h` + espelho Xbox).
3. **Paridade de kernel primeiro (gate 1):** harness no Xbox (op de diagnóstico estilo
   probe 13-17) que recebe records/stage/tail/Q sintéticos ou capturados do host e devolve
   attention output; comparar contra `fattn-kvarn` CUDA com as mesmas entradas:
   erros quantificados (max abs, cosine, NMSE) por configuração (grupo cheio, fronteira de
   tile, tail parcial, sinks, head 256, GQA). Critério: dentro da tolerância do fattn local
   vs referência F32 (documentar números).
4. **Store parity (gate 2):** mesmo harness para `rkva_kvarn_store.hlsl` vs `kvarn.cu`
   (records bit-identicos ou dentro de tolerância documentada — alvo: bit-idêntico, como o
   pack Q4_0 já conseguiu).
5. **Integração host:** op + backend + graph surgery + manager de sessões + args.
6. **Engine Xbox de produção:** sessões persistentes, decode/prefill, trim/rewind/reset,
   stats.
7. **Corretude ponta-a-ponta (gate 3):** token-identity vs baseline local (temp 0, mesma
   seed, mesmos cache types) em 128/512/2K/4K; testes de cache vazio, primeiro token,
   boundary de tile, precision tail, trim, rewind, reset, slot reuse, multi-layer,
   prefill+decode.
8. **Escada de contexto:** 4K→16K→32K→64K→88K→128K→192K→256K, parando em limite de
   memória ou bug reproduzível; por nível: VRAM host/Xbox, prefill/decode tok/s, ms/token,
   RPC MB/s e bytes/token, RTT avg/p50/p95, kernel Xbox ms, RPC calls/token.
9. **Benchmark A/B(/C):** A=IQ3_XXS-mtp KV local vs B=KV/attention Xbox vs
   C=(se disponível) quant atual; custo em TPS de liberar a VRAM, sem esconder perdas em
   médias.
10. **Otimização:** só com benchmark antes/depois — RTT, sincronizações CUDA, fences D3D12,
    staging copies, kernel decode, serialização, prefill batching; `--remote-attn-prefill
    migrate` (construir/quantizar records no host e migrar comprimido) se o prefill remoto
    via Gigabit ficar caro — espaço arquitetural já previsto no protocolo.

## Profiling

`--remote-attn-stats` (sem recompilar): calls, bytes_tx/rx, avg bytes/token, rpc_ms,
xbox_store_ms, xbox_attn_ms, host_staging_ms, total_ms, p50/p95; por layer em verbose.
Memória: host (weights/KV/compute buffers/pico prefill/steady decode) e Xbox (KV persistente,
tail, scratch, upload/readback, pico commitado vs teto 5GB).

## Critérios de sucesso (MVP)

- Pesos target ~inteiros na 4070; KV das 16 full-attention layers só no Xbox; attention
  executada no Xbox sobre KV comprimido; sem transmitir KV histórico por token.
- Output = baseline dentro da tolerância esperada do KVarN4; 32K e 64K estáveis; 128K testado.
- Profiling claro; falha de conexão não corrompe estado silenciosamente; baseline sem
  `--remote-attn` intacto.
- Metas seguintes: 256K kvarn4 se couber em ~5GB; decode utilizável via Gigabit;
  MTP/DFlash sobre target KV remoto.

## Riscos conhecidos

- **16 RTTs sequenciais/token** (~9-10ms de rede só de latência) + transferências: decode
  remoto será mais lento que local em contexto curto; o ganho é habilitar contextos que NÃO
  cabem na 4070 (128K+ com pesos IQ3) e liberar VRAM.
- Regime latency-bound do console (20GB/s) se o kernel de attention não ler records de forma
  linear/coalesced → projetar layout de leitura para streaming (145-165GB/s).
- Paridade numérica Sinkhorn/Hadamard em wave32/RDNA2: mitigada pelos gates 1-2 antes da
  integração e pelos precedentes bit-exatos do DFlash.
- Teto de 5GB do Xbox com 256K (≈4,7GB estimado) — validar na escada; 128K (≈2,4GB) folgado.
- VRAM da 4070 ocupada pelo server de produção — janelas de teste com stop/restore (decisão 3).

## Não fazer (do pedido original)

Não pagear pesos pela Ethernet; não devolver KV histórico ao CUDA; não mover DeltaNet state;
não replicar o modelo no Xbox; não materializar KVarN→F16 no decode normal; não criar conexão
por RPC/token; não malloc por token; não alterar quantização do target; não sacrificar
correctness; não mergear no main automaticamente.
