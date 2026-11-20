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
  3. `cache = min(livre - margem, bytes_host)`; abaixo de 256 MiB, desliga;
  4. loga o plano (`MoE cache auto: ...`).
- Guardas: nunca resolve em contexto `no_alloc` (medição do `--fit`) nem quando não há
  expert em host (evita o `throw` do construtor com `n_tensors_host == 0`).
- Critério de aceite: com modelo MoE e experts em host, o log mostra o plano, o cache
  entra em uso (hit rate no fim) e o servidor responde; com modelo denso, loga
  "nenhum expert em host" e segue sem cache.

## P2 — pinned + planner (próximo)

1. `--moe-host-pinned on|off` (default `on` quando o device expõe host buffer type):
   forçar os tensores `ffn_*_exps` host para `ggml_backend_dev_host_buffer_type(dev)`.
2. `--vram-reserve-mib N` (default 1536) usado pelo P1.
3. `--moe-plan`: imprime plano (bytes host, slots, cobertura %, VRAM livre) e sai.
4. Teste: `llama-bench` com/sem pinned num MoE que force host.

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
