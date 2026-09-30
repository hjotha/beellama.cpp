# BeeLLaMA — Local-Split Attention 4070 + Radeon 780M

## Plano inicial com Qwen3.8-27B IQ3_XXS

> **Status da Execução:** Em andamento (Compilação concluída, Profiling Fase 0 ativo)
> **Branch:** `integrate-pending-dflash-and-adaptive-mtp`
> **Dispositivos:** RTX 4070 12GB (CUDA) + Radeon 780M (Vulkan:0 / RADV Phoenix)
> **Configuração APU/Radeon:** TDP = 20W (`apu-tdp 20`), Clock da APU = 2700 MHz (`pclockmax 2700`)

### Tabela de Progresso das Fases
| Fase | Descrição | Status | Resultados / Notas |
|---|---|---|---|
| **Fase 0** | Profiling baseline (4070 pura, IQ3_XXS) | ✅ **Concluída** | 24K: 852 t/s, 38K: 798 t/s, 56K: 714 t/s, 72K: 666 t/s, 102K: 87 t/s (VRAM limit @ 102K), Decode: 37.2 t/s |
| **Fase 1** | Backend local-attn Vulkan (In-process) | ✅ **Concluída** | `ggml-local-split` implementado (ring buffer host-pinned 64B align), APU TDP 20W / 2700MHz, `test-local-split-attn` 100% OK |
| **Fase 2** | 1–2 layers remotas (Validação ring buffer) | ✅ **Concluída** | Decode estável: 1 layer = 34.5 tok/s, 2 layers = 34.1 tok/s. Latência por boundary = 1.11 ms |
| **Fase 3** | Placement variável (Curva TPS × N layers) | ✅ **Concluída** | Curva completa (16/0 -> 0/16): 16L=36.9 t/s, 4L=33.5 t/s, 8L=33.7 t/s, 16L offload=33.3 t/s (~183 µs/layer, queda total <10%) |
| **Fase 4** | Auto placement baseado em VRAM | ✅ **Concluída** | Auto-placement dinâmico via `--remote-attn-layers auto` e `--remote-attn-cuda-reserve`. Validado: 4K = 16 locais / 0 remotas; 102.4K = 2 locais / 14 remotas (176 t/s prefill, 30.6 tok/s decode, zero VRAM thrashing) |
| **Fase 5** | Prefill migration assíncrona | ⏳ Planejada | Pipeline assíncrona chunk 512 |
| **Fase 6** | Prompt cache awareness | ⏳ Planejada | Direct KV placement no cache |
| **Fase 7** | MTP / DFlash2 batching ($N > 1$) | ✅ **Concluída** | MTP validado com sucesso (`Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf`). 4K: **47.7 tok/s** (83.3% acceptance rate, speedup de +47% vs autoregressivo); 56.3K: **33.9 tok/s** com 13L offloaded no target e 16L no draft context |
| **Fase 8** | Migração e validação IQ3_S | ⏳ Planejada | Alvo após caracterização do IQ3_XXS |

**Objetivo:** usar a RTX 4070 como GPU principal do modelo e a Radeon 780M como acelerador auxiliar de KV cache + attention, mantendo o máximo possível de attention local na 4070 e enviando para a 780M apenas o overflow necessário.

O primeiro alvo será o **Qwen3.8-27B GSQ-RCO IQ3_XXS**, não o IQ3_S. O IQ3_XXS oferece mais folga de VRAM e permite validar corretamente arquitetura, sincronização, staging, cache placement e desempenho antes de tentar ocupar quase toda a VRAM da 4070 com pesos.

---

# 1. Arquitetura alvo

```text
                         Qwen3.8-27B IQ3_XXS

                                TOKEN
                                  │
                                  ▼
                       RTX 4070 / CUDA
                  ┌─────────────────────────┐
                  │ weights                 │
                  │ DeltaNet                │
                  │ FFN / MLP               │
                  │ RMSNorm                 │
                  │ Q/K/V projections       │
                  │ Wo / gate               │
                  │ sampling                │
                  │ prefill                 │
                  └─────────────────────────┘
                                  │
                         full-attention layer
                            │             │
                         LOCAL          REMOTE
                            │             │
                            ▼             ▼
                     RTX 4070 KV      host-pinned
                       KVarN4          ring buffer
                            │             │
                            ▼             ▼
                    CUDA attention   Radeon 780M
                                     Vulkan KVarN4
                                     Flash Attention
                                          │
                                          ▼
                                  attention output
                                          │
                                          ▼
                                      RTX 4070
```

A Radeon 780M não será tratada como "RAM externa".

Ela deve armazenar KV e executar a attention correspondente.

A RTX 4070 continuará executando todas as outras operações do modelo.

---

# 2. Estratégia principal: split por attention layer

O Qwen3.8-27B tem 64 layers, mas apenas aproximadamente 16 usam full attention tradicional.

Não dividir inicialmente uma mesma attention entre duas GPUs.

Em vez disso:

```text
attention layer X → inteiramente CUDA
attention layer Y → inteiramente Vulkan/780M
```

Isso evita:

- combinação distribuída de softmax na primeira versão;
- sincronizações adicionais dentro de uma mesma layer;
- transferência de KV histórico;
- complexidade desnecessária.

O placement será determinado pela VRAM disponível na 4070 após carregar o modelo e seus buffers obrigatórios.

---

# 3. Por que começar com IQ3_XXS

O IQ3_XXS já funciona no hardware atual e ocupa significativamente menos VRAM que IQ3_S.

Isso permite reservar deliberadamente parte da VRAM da 4070 para KV local.

Objetivo inicial:

```text
RTX 4070
├─ todos os pesos IQ3_XXS
├─ DeltaNet/recurrent state
├─ compute buffers
├─ KV KVarN4 de várias full-attention layers
└─ staging mínimo

Radeon 780M
├─ KV KVarN4 das full-attention layers restantes
├─ precision tail remoto
├─ Vulkan Flash Attention
└─ buffers persistentes
```

Primeiro provar que o mecanismo funciona e medir o ganho real.

Só depois subir para IQ3_S.

---

# 4. Meta de placement automático

No startup, medir a VRAM CUDA realmente livre depois de carregar:

- pesos;
- CUDA context;
- graph buffers;
- DeltaNet state;
- scratch obrigatório;
- speculative state, se habilitado.

Reservar margem de segurança configurável.

Exemplo conceitual:

```text
CUDA free after model:            1100 MiB
safe reserve for compute:          350 MiB
usable KV budget:                  750 MiB
ctx:                               65536
KVarN4 per full-attn layer:         XX MiB
local full-attn layers:              N
remote full-attn layers:          16-N
```

Flags sugeridas:

```text
--split-attn-device vulkan:0
--split-attn-auto
--split-attn-cuda-reserve 350M
```

Também permitir override manual:

```text
--split-attn-local-layers 8
```

---

# 5. Regra de placement

Primeira versão:

- manter as primeiras N full-attention layers localmente ou;
- testar colocar as layers de maior custo/localidade mais favorável.

Não assumir que primeiras ou últimas layers são melhores sem benchmark.

Adicionar modo:

```text
--split-attn-placement sequential
--split-attn-placement benchmark
```

No modo benchmark, testar rapidamente algumas distribuições e escolher a melhor.

---

# 6. Transporte CUDA ↔ Vulkan

Não assumir interoperabilidade direta NVIDIA↔AMD via external-memory GPU-to-GPU.

Primeira implementação segura:

```text
RTX 4070 CUDA
    │
    │ cudaMemcpyAsync
    ▼
pinned host memory
    │
    │ Vulkan host-visible/imported buffer
    ▼
Radeon 780M
```

Usar buffers persistentes e pré-alocados.

Evitar qualquer malloc/free por token.

---

# 7. Ring buffer persistente

Criar ring buffers para:

```text
Q
K_new
V_new
attention output
metadata
```

Testar double buffering primeiro.

Se houver stalls:

```text
triple buffering
```

Estrutura conceitual:

```text
slot 0 → CUDA produz / Vulkan consome
slot 1 → CUDA produz / Vulkan consome
slot 2 → opcional
```

Cada slot deve ter:

- host pointer;
- CUDA registration;
- Vulkan buffer;
- offsets fixos;
- fence/event próprio.

---

# 8. Sincronização

Esse é um dos principais riscos de performance.

Não usar sincronização global do device a cada layer.

Evitar:

```text
cudaDeviceSynchronize()
vkQueueWaitIdle()
```

no hot path.

Preferir:

```text
CUDA events
Vulkan fences/semaphores
per-slot state
poll/wait apenas no recurso necessário
```

Instrumentar separadamente:

```text
cuda_to_host_us
vulkan_wait_us
vulkan_kernel_us
host_to_cuda_us
sync_total_us
```

---

# 9. Decode — caminho principal

Para uma full-attention layer remota:

```text
4070:
  norm
  Q projection
  K projection
  V projection
  transforms necessários

       ↓

host-pinned ring
  Q
  K_new
  V_new
  position
  sequence metadata

       ↓

780M:
  append K/V ao KVarN
  precision-tail handling
  attention sobre KV histórico
  escreve attention output

       ↓

4070:
  Wo
  gate
  residual
  próxima operação
```

O KV histórico nunca deve voltar para a 4070.

---

# 10. KVarN

Primeiro alvo:

```text
--cache-type-k kvarn4
--cache-type-v kvarn4
```

Reutilizar o máximo possível dos shaders já existentes no BeeLLaMA:

```text
kvarn_store.comp
kvarn_flash_attn.comp
kvarn_wht.comp
```

Não criar formato incompatível.

O layout CUDA e Vulkan deve permanecer semanticamente equivalente.

Validar:

- scales;
- records;
- WHT/Hadamard;
- sinks;
- precision tail;
- positions;
- sequence IDs;
- trim;
- rewind.

---

# 11. Prefill

O prefill continuará inicialmente 100% na RTX 4070.

Isso é especialmente importante porque o sistema atual já atinge aproximadamente:

```text
~800 tok/s prefill
```

Não sacrificar esse caminho.

O fluxo será:

```text
4070 prefill local
      │
      ├─ KV das layers locais permanece CUDA
      │
      └─ KV das layers remotas é migrado para a 780M
```

---

# 12. Migração de KV durante prefill

Não enviar Q/K/V de cada token via split-attention durante prefill se não for necessário.

Gerar KVarN localmente e migrar em blocos.

Testar chunks:

```text
128
256
512
1024
2048 tokens
```

Flag:

```text
--split-attn-migrate-chunk 512
```

Implementar pipeline assíncrono:

```text
CUDA calcula chunk N+1
        │
        └───────────── paralelamente
                      │
          migra KVarN chunk N
                      │
                      ▼
                    780M
```

Medir quanto da migração fica oculto pelo próprio prefill.

---

# 13. Prompt cache

Cache hit muda completamente a importância do prefill.

O prompt cache deve guardar, quando possível, a representação já adequada ao placement final.

Objetivo:

```text
cache hit
  ├─ local attention KV → CUDA
  └─ remote attention KV → RAM/780M
```

Evitar:

```text
SSD
→ CUDA
→ materializar
→ re-quantizar
→ migrar para Vulkan
```

se o cache já puder ser carregado diretamente como KVarN.

Adicionar métricas:

```text
prompt_cache_hit
cache_load_ms
cache_to_cuda_ms
cache_to_vulkan_ms
cache_reformat_ms
```

---

# 14. MTP / DFlash2

Depois que decode normal estiver correto, testar speculative.

O objetivo é fazer remote attention aceitar múltiplos query tokens numa única submissão.

Exemplo:

```text
n_tokens = 1
n_tokens = 2
n_tokens = 4
n_tokens = 8
```

Isso pode amortizar fortemente:

- sincronização;
- command submission;
- CUDA↔host transfer;
- Vulkan dispatch;
- retorno.

Para MTP `n_max=4`, idealmente:

```text
4 queries
→ 1 transferência
→ 1 ou poucos dispatches Vulkan
→ 1 retorno
```

em vez de quatro operações independentes.

---

# 15. Não quantizar transporte prematuramente

FP16 para Q/K/V/output deve ser o baseline.

A largura de banda necessária é pequena comparada à capacidade do link da eGPU.

O gargalo provável será:

- sincronização;
- dispatch Vulkan;
- memória da 780M;
- número de boundaries por target pass.

Só testar FP8/Q8 para transporte se profiling mostrar que PCIe/USB4 realmente virou gargalo.

---

# 16. Métricas obrigatórias

Por token:

```text
local_full_attn_layers
remote_full_attn_layers
cuda_attn_ms
remote_attn_ms
cuda_to_host_us
host_to_vulkan_us
vulkan_store_us
vulkan_attention_us
vulkan_to_host_us
host_to_cuda_us
cross_device_sync_us
bytes_tx
bytes_rx
```

Globais:

```text
prefill tok/s
decode tok/s
prompt cache hit rate
CUDA VRAM peak
CUDA VRAM steady
780M memory usage
system RAM usage
```

---

# 17. Baselines obrigatórios

## A — atual

```text
IQ3_XXS
4070 somente
KVarN local
```

Registrar:

```text
prefill tok/s
decode tok/s
VRAM
max context
```

## B — split forçado pequeno

```text
IQ3_XXS
14 full-attn layers CUDA
2 full-attn layers 780M
```

Objetivo: validar plumbing com penalidade mínima.

## C — split médio

```text
8 CUDA
8 Radeon
```

## D — quase tudo remoto

```text
2 CUDA
14 Radeon
```

## E — tudo remoto

```text
0 CUDA
16 Radeon
```

Esse último é importante como referência, mesmo que não seja o modo recomendado.

---

# 18. Escada de contexto

Testar:

```text
4K
8K
16K
32K
40K
56K
64K
84K
88K
96K
128K
```

Depois, se estável:

```text
192K
256K
```

Registrar automaticamente para cada ponto:

```text
context
prefill tok/s
decode tok/s
CUDA VRAM
780M memory
RAM
local layers
remote layers
cross-device ms/token
```

---

# 19. Critério de placement automático

Escolher o máximo de layers locais possível sem colocar o CUDA no limite.

Pseudocódigo:

```text
load model
measure free CUDA VRAM
reserve safety margin
estimate KV bytes per full-attn layer for requested context
N = floor(usable_cuda_kv / bytes_per_layer)
N = clamp(N, 0, 16)
assign N layers CUDA
assign 16-N layers Radeon
```

Adicionar safety margin configurável.

Nunca operar com menos que uma margem mínima de memória apenas para aumentar N.

---

# 20. Placement baseado em profiling

Segunda etapa.

Nem toda distribuição de N layers precisa ter o mesmo custo.

Testar se é melhor:

```text
primeiras N locais
últimas N locais
alternadas
blocos contíguos
```

Como cada remote layer gera uma boundary, pode existir diferença real devido ao scheduler e à posição das full-attention layers.

Criar benchmark rápido no startup opcional:

```text
--split-attn-autotune
```

---

# 21. Etapa futura: attention particionada por contexto

Não implementar inicialmente.

Depois do split por layers estar estável, considerar:

```text
4070 → precision tail / KV recente
780M  → KVarN body / KV antigo
```

Cada GPU calcula estatísticas parciais de attention e o resultado é combinado por online softmax numericamente estável.

Potencial vantagem:

- 4070 continua participando mesmo quando não cabe uma layer inteira de KV;
- tail recente permanece extremamente rápido;
- 780M processa apenas histórico antigo.

Mas isso deve ser fase posterior por aumentar bastante a complexidade.

---

# 22. Testes de correctness

Comparar sempre contra execução 100% local na 4070.

Configuração:

```text
temperature = 0
same seed
same prompt
same KVarN settings
```

Testar:

1. 1 token;
2. 128 tokens;
3. 512 tokens;
4. 4K;
5. tile boundary;
6. precision tail boundary;
7. trim;
8. rewind;
9. slot reset;
10. prompt cache restore;
11. MTP disabled;
12. MTP enabled;
13. DFlash disabled;
14. DFlash enabled.

Comparar:

```text
logits
selected token
top-k
K/V checksums where useful
per-layer output error
```

---

# 23. Fail-safe

Se Vulkan/780M falhar:

- nunca continuar com KV inconsistente;
- abortar a sequence ou;
- reconstruir/migrar para CUDA apenas se houver memória suficiente.

Não fazer fallback silencioso incorreto.

---

# 24. Fases de implementação

## Fase 0 — Profiling baseline

Sem alterar execução. Modelo: `Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf` (9.39 GiB), 100% CUDA na RTX 4070 (12 GB), `ngl 99`, `ctk kvarn4`, `ctv kvarn4`, `b/ub 256`, MTP OFF.

### Resultados Medidos (Empírico — 30/09/2026):
| Contexto / Tier | Teste | Prefill / Decode (tok/s) | VRAM Ocupada | Status / Observações |
|---|---|---|---|---|
| **4.096 (4K)** | `pp4096` / `tg128` | **912.46 tok/s** / **36.95 tok/s** | ~10.2 GiB | Prefill acima da meta (~800 tok/s) |
| **24.576 (Tier S)** | `pp24576` | **852.40 tok/s** | ~10.6 GiB | Excelente estabilidade |
| **38.912 (Tier M)** | `pp38912` | **798.71 tok/s** | ~10.9 GiB | Operação 100% em VRAM rápida |
| **56.320 (Tier L)** | `pp56320` | **714.80 tok/s** | ~11.3 GiB | Alta performance contínua |
| **72.704 (Tier XL)** | `pp72704` | **666.41 tok/s** | ~11.6 GiB | Limiar superior confortável na 4070 |
| **102.400 (Tier XXL)**| `pp102400` | **86.98 tok/s** | **11.875 MiB** (28 MiB livres) | **Saturação de VRAM** (Queda abrupta por thrashing) |
| **Decode Global** | `tg128` | **37.20 tok/s** | 11.875 MiB | Decode puro sem MTP |

### Diagnóstico Crucial da Fase 0:
1. **O Teto da 4070 isolada é 72.7K:** A RTX 4070 mantém de 852 a 666 tok/s de prefill até 72.7K.
2. **Colapso em 102.4K por falta de VRAM:** Em 102.4K tokens, restam apenas 28 MiB livres na 4070, fazendo o prefill despencar de 666 t/s para 86.98 t/s.
3. **Justificativa Comprovada do Local-Split:** O offload de KV para a Radeon 780M (Vulkan) manterá a 4070 na faixa de folga de VRAM (<11 GiB), preservando os ~700–850 tok/s de prefill mesmo em 102K, 128K e 200K.

### Entregável

Tabela baseline completa (Concluída).

---

## Fase 1 — Backend local-attn Vulkan

Criar abstração in-process equivalente à interface remota existente.

Sem socket.

Operações mínimas:

```text
create_session
append_kv
compute_attn
trim
rewind
reset
destroy
stats
```

### Entregável

Uma attention layer executada na Radeon com resultado equivalente ao CUDA.

---

## Fase 2 — 1–2 layers remotas

Manter 14–15 attention layers locais.

Objetivo:

- validar ring buffer;
- validar sync;
- medir custo real por boundary.

### Entregável

Decode estável com penalidade quantificada.

---

## Fase 3 — Placement variável

Testar:

```text
16/0
14/2
12/4
8/8
4/12
0/16
```

### Entregável

Curva TPS × número de remote layers.

---

## Fase 4 — Auto placement

Escolher N automaticamente de acordo com contexto e VRAM.

### Entregável

Startup log com decisão automática.

---

## Fase 5 — Prefill migration

Manter prefill CUDA rápido e migrar KV remoto em chunks.

### Entregável

Queda de prefill mínima em relação aos ~800 tok/s atuais.

---

## Fase 6 — Prompt cache awareness

Carregar KV diretamente no device correto.

### Entregável

Cache hit sem reconstrução desnecessária.

---

## Fase 7 — MTP / DFlash batching

Permitir n_tokens > 1 por dispatch remoto.

### Entregável

Redução perceptível de cross-device overhead por token aceito.

---

## Fase 8 — IQ3_S

Somente depois que o IQ3_XXS estiver caracterizado.

Trocar pesos para IQ3_S.

Repetir automaticamente:

```text
VRAM baseline
auto placement
context ladder
prefill
decode
MTP/DFlash
```

O IQ3_S será tratado como evolução da mesma arquitetura, não como uma implementação separada.

---

# 25. Metas iniciais de performance com IQ3_XXS

Esses números são metas de engenharia, não resultados garantidos.

## 32K

```text
prefill: manter próximo de ~800 tok/s em cache miss normal
decode: alvo >= 45 tok/s se a maioria das attentions ficar CUDA
```

## 64K

```text
decode: alvo >= 35–45 tok/s
```

## 88K

```text
decode: alvo >= 30–40 tok/s
```

## 128K

```text
decode: alvo >= 25–35 tok/s
```

A primeira pergunta que o profiling deve responder é:

> quantas full-attention layers conseguimos manter na 4070 em cada contexto sem prejudicar estabilidade?

Essa resposta determinará o teto real do sistema.

---

# 26. Métrica mais importante: custo por remote layer

Depois da Fase 2, calcular:

```text
cost_per_remote_layer_us
```

Separar em:

```text
transfer_us
sync_us
vulkan_dispatch_us
kvarn_attention_us
return_us
```

Com isso podemos estimar rapidamente qualquer placement.

Exemplo:

```text
baseline token:               19.0 ms
remote-layer overhead:         0.28 ms
8 remote layers:              +2.24 ms
estimated total:              21.24 ms
estimated decode:             47.1 tok/s
```

Esse modelo precisa ser confrontado com benchmark real para detectar efeitos não lineares.

---

# 27. O que NÃO fazer no primeiro protótipo

- Não dividir uma mesma attention entre CUDA e Vulkan.
- Não implementar FP8 de transporte antes de medir.
- Não portar DeltaNet para 780M.
- Não mover FFN para 780M.
- Não mover pesos do target para RAM.
- Não usar KV histórico via PCIe diretamente pela 4070.
- Não usar TCP/local sockets entre as duas GPUs.
- Não fazer malloc/free por token.
- Não usar sync global por layer.
- Não começar com IQ3_S.

---

# 28. Primeiro experimento prático

Configuração inicial recomendada:

```text
Model: Qwen3.8-27B GSQ-RCO IQ3_XXS
Target GPU: RTX 4070
Aux GPU: Radeon 780M / Vulkan
KV: KVarN4 K + V
Context: 32K
MTP: OFF
DFlash: OFF
Prefill: CUDA
Remote attention layers: 2
```

Objetivo:

1. validar correctness;
2. medir overhead de exatamente 2 boundaries;
3. medir bandwidth real;
4. medir latência real CUDA↔host↔Vulkan;
5. medir kernel KVarN da 780M;
6. medir TPS vs baseline.

Depois:

```text
2 → 4 → 8 → 12 → 16 remote layers
```

Só então ativar MTP/DFlash.

---

# 29. Resultado final desejado

Queremos que o BeeLLaMA transforme automaticamente o sistema em algo semelhante a:

```text
RTX 4070 12 GB
────────────────────────────
Qwen3.8-27B weights
DeltaNet
compute buffers
máximo possível de KV/attention local

Radeon 780M
────────────────────────────
overflow de KVarN KV
attention das layers remotas
prompt-cache KV remoto
```

O objetivo não é simplesmente aumentar capacidade.

O objetivo é maximizar simultaneamente:

1. qualidade da quantização;
2. contexto disponível;
3. decode TPS;
4. prefill TPS;
5. aproveitamento dos dois GPUs;
6. estabilidade.

Primeiro provar tudo com **IQ3_XXS**.

Depois migrar para **IQ3_S** usando os números medidos para decidir exatamente quanto KV precisa ficar na 780M.
