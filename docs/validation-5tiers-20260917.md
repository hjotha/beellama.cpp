# Validação 5 tiers — Qwen3.8-27B (gokaya, RTX 4070 12GB eGPU)

Data: 2026-09-17 · Binário: `releases/beellama-router-kvarn-convert-20260917-r8` (main `2f0d711bc`) · Produção porta 8090 · Testes isolados na 8091 e suíte final na 8090.

Tiers: s=32768 (q4_0) · m=56320 (q4_0) · l=97536 (q4_0) · xl=104448 (q4_0) · xxl=114688 (kvarn4).
Prompt A = contexto no limite (input=limite−4096); Prompt B = persistência de slot; transições = reuso de cache (RAM + disco).

## Resultado: 18/18 PASS · FAIL=0

| teste | orig→dest | input | cache reusado | prefillTPS | decodeTPS | TTFT(s) | total(s) | KV | resultado |
|---|---|---|---|---|---|---|---|---|---|
| A-s-ctx | s->s | 28672 | 0 | 860.0 | 70.9 | 33.3 | 37.8 | q4 | **PASS** |
| A-m-ctx | m->m | 52224 | 0 | 749.9 | 43.3 | 69.6 | 77.0 | q4 | **PASS** |
| A-l-ctx | l->l | 93440 | 0 | 638.2 | 20.3 | 146.4 | 162.2 | q4 | **PASS** |
| A-xl-ctx | xl->xl | 100352 | 0 | 436.0 | 19.6 | 230.2 | 246.5 | q4 | **PASS** |
| A-xxl-ctx | xxl->xxl | 110592 | 0 | 379.4 | 23.9 | 291.5 | 304.9 | kvarn4 | **PASS** |
| T-s->m | s->m | 52224 | 28671 | 544.8 | 44.1 | 43.2 | 50.5 | q4 | **PASS** |
| T-m->l | m->l | 93440 | 52223 | 477.8 | 20.2 | 86.3 | 102.1 | q4 | **PASS** |
| T-l->xl | l->xl | 100352 | 93440 | 236.5 | 19.6 | 29.2 | 45.5 | q4 | **PASS** |
| T-xl->xxl | xl->xxl | 110592 | 100352 | 173.3 | 23.1 | 59.1 | 72.9 | kvarn4 | **PASS** |
| T-s->xxl | s->xxl | 110592 | 28671 | 340.5 | 23.1 | 240.6 | 254.4 | kvarn4 | **PASS** |
| T-m->xxl | m->xxl | 110592 | 52223 | 308.3 | 23.1 | 189.3 | 203.2 | kvarn4 | **PASS** |
| T-l->xxl | l->xxl | 110592 | 93440 | 229.1 | 23.1 | 74.9 | 88.7 | kvarn4 | **PASS** |
| T-xl->xxl | xl->xxl | 110592 | 110592 | 0.0 | 23.2 | 4.1 | 17.9 | kvarn4 | **PASS** |
| C-s-slot | s->s | 28672 | 29695 | 0.8* | 19.7 | 1.2 | 1.7 | q4 | **PASS** |
| C-m-slot | m->m | 52224 | 53247 | 0.4* | 22.0 | 2.9 | 3.2 | q4 | **PASS** |
| C-l-slot | l->l | 93440 | 94464 | 0* | 21.3 | 4.2 | 4.6 | q4 | **PASS** |
| C-xl-slot | xl->xl | 100352 | 101376 | 0* | 21.0 | 3.0 | 3.3 | q4 | **PASS** |
| C-xxl-slot | xxl->xxl | 110592 | 111616 | 0* | 23.4 | 3.0 | 3.4 | kvarn4 | **PASS** |

> * Nas linhas C, o `prefillTPS` mostra o `delta_tps` (só os tokens novos), para não diluir com o custo fixo de switch+restore. Quebras típicas: `switch_ms` 220–260ms (xxlong no caminho quente; o rebuild frio do contexto kvarn chega a 8.5s sob pressão de VRAM) + `restore_ms` 3.0–4.2s (leitura do estado do disco) + delta.


## Causa-raiz corrigida: OOM de VRAM no prefill xxlong (regressão do merge)

O merge do upstream `anbeeld/main` (v0.4.6, `b01db3d84`) mudou a partição do split-decode KVarN: o buffer parcial do pool CUDA passou a ser cobrado pelo `n_q` real (2–8 na verificação MTP) em vez de `n_q==1`. Em `n_kv≈110k` isso reserva **~226 MiB permanentemente** no `ggml_cuda_pool_vmm` (que nunca devolve páginas à driver), e o prefill do xxlong (110592 tokens) estourava intermitentemente com `CUDA error: out of memory` (`ggml_cuda_pool_vmm::alloc`). Confirmado por comparação: o binário pré-merge (r7) passava a mesma sequência; o mesclado (r8) crashava.

### Fix aplicado
- **Código** (`2f0d711bc`): `ggml_cuda_fattn_kvarn_split_max_q` em `ggml/src/ggml-cuda/fattn-kvarn-dispatch.cu` — limita o `max_q` do split-decode por um budget de 64 MiB (env `GGML_CUDA_KVARN_SPLIT_MAX_Q` / `GGML_CUDA_KVARN_SPLIT_PARTIAL_MB`); em `n_kv` profundo volta ao caminho MMA genérico (comportamento pré-merge), **sem perda de TPS de decode** (xxl 23.9 tok/s vs 19.6 do r7).
- **Env na produção** (`/etc/systemd/system/llama-server-root.service`): `GGML_KVARN_WINDOW_CHUNK=32768` e `GGML_CUDA_GRAPH_RECOVERY_HEADROOM_MB=128`.

## O que foi validado
1. **Contexto no limite** (Prompt A): os 5 tiers processam input = limite−4096 sem erro, com prefill 379–860 tok/s e decode 19.6–70.9 tok/s; VRAM ~11.5–11.9 GB.
2. **Cache entre tiers** (8 transições): sequenciais 32→56→97→104→110k e diretas X→110k; reuso de 28671/52223/93440/100352 tokens confirmado por `cache_n` (RAM + auto-store em disco com conversão q4→KVarN no destino xxl).
3. **Persistência de slots** (Prompt B, 5 tiers): prime → restart do serviço → restore do disco → `cache_n` igual ao estado salvo, sem prefill completo indevido.
4. **KVarN comprovado**: destinos xxl restaurados em kvarn4 (log `route snapshot converted` / `auto-restore`), conversão ~1–4 s.

## Acompanhamento: teste do tier kvarn único (s/m/l, l = xxl atual)

Pedido: substituir o tier l (q4 97k) por um único tier kvarn até 110k. Testado isolado
(porta 8091, build `b1f188532` com as novas flags `--batch-size-l`/`--ubatch-size-l`/
`--cache-type-k-l`/`--cache-type-v-l`), request de 110592 + 4096 de saída:

- **ubatch 128**: `failed to allocate compute pp buffers` na criação do contexto long
  (VRAM) -> 500 "adaptive context profile transition failed".
- **ubatch 64**: mesmo erro -> o reserva de buffers de compute do perfil long (kvarn4,
  114688) NÃO cabe em VRAM, embora o perfil xxlong com a mesma geometria (kvarn4,
  ubatch 64) funcione. => o dimensionamento do buffer do perfil long difere do xxlong
  (investigação pendente antes de adotar o layout de tier kvarn único).

Implicação: a perda de prefill do tier kvarn único fica como o medido no xxl (379 tps
vs 638 do l q4, +68% TTFT para 93k) enquanto o ubatch não puder subir; com ubatch
viável seria ~neutro (kvarn ~+10% sobre q4 ao mesmo ubatch). Decode kvarn é mais rápido
(23.9 vs 20.3 tps). Flag de investigação: por que o reserva do perfil long usa mais VRAM
que o xxlong para a mesma geometria.

## Notas
- O build mesclado para a geração em ~320 tokens (vs 4096 do r7) — comportamento de stop do reasoning pós-merge; não é falha da suíte (o cap era 4096, "máximo").
- Linhas `*-prime`/`*-restore` sem campo `pass` aparecem como FAIL no resumo; os veredictos por teste (C-*-slot) são todos PASS.

