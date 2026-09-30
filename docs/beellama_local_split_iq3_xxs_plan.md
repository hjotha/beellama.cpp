# BeeLLaMA — Local-Split Attention 4070 + Radeon 780M

## Correção de validade — 2026-09-30

Os resultados e marcações de conclusão abaixo são históricos e **não validam a
implementação local-split original** (`49f273af9`). O operador local copiava Q/K/V,
zerava a saída com `memset` e retornava sucesso sem computar atenção no Vulkan.
Um teste isolado da biblioteca utilizada pela produção confirmou a falha: com um
único token Q=K=0 e V=1, todos os 6144 elementos de saída foram 0 em vez de 1.
O teste anterior exercitava apenas inicialização e não conferia os valores.
Consequentemente, medições de throughput, aceitação MTP, memória e persistência
feitas com aquele caminho não demonstram inferência correta.

A correção utiliza os caches nativos `llama_kv_cache`/`llama_kv_cache_kvarn` com
placement por camada e o scheduler GGML para executar a atenção no backend Vulkan.
Pesos, projeções Q/K/V, RoPE, gate, FFN e estado recorrente permanecem no dispositivo
do modelo. O próprio cache gerencia records, staging, tail, rollback e save/restore.
O antigo operador de transporte local falha explicitamente se for chamado.
O protocolo TCP RKVA do Xbox continua separado desse caminho local.

O protótipo opt-in da Fase 5 agora mantém o prefill na CUDA e espelha records
KVarN selados para Vulkan em background, com handoff explícito do cache antes
do decode. O caminho nativo continua executando atenção Vulkan com KV Vulkan
durante decode; não reutilizar slots produzidos pelo caminho antigo que zerava
a atenção.

O protótipo requer KVarN, uma sequência e nenhuma SWA. Suporta MTP standalone:
somente o cache alvo migra e o contexto draft mantém a própria rota estática.
Com `remote-attn-layers=auto`, migra uma layer se o KV alvo completo cabe no
orçamento seguro da CUDA; quando não cabe, mantém o placement remoto estático
para o perfil. Shadow e outros modos especulativos são recusados. O espelho
Vulkan continua duplicando os records durante o prefill, então a feature não
libera VRAM CUDA nem aumenta por si só o limite de contexto; ainda não está
habilitada no preset de produção.

## Diagnóstico de produção — 2026-09-30

O perfil que forçava `remote-attn-layers = 16` executava o núcleo de attention de
**todas** as 16 camadas full-attention na Radeon também durante o prefill. Isso
contraria a seção 11 do plano, que mantém o prefill na 4070, e pula a migração
assíncrona de KV da Fase 5.

No pedido de produção de hoje, o prefill caiu progressivamente de 106.48 t/s em
512 tokens para 20.30 t/s em 6.847 tokens. Em seguida o kernel registrou dois
timeouts na fila `comp_1.2.0`, resetou o compute queue e a inferência falhou com
`decode() failed: vk::Queue::submit: ErrorDeviceLost`. A próxima tarefa falhou
com `a compact KV tail batch transaction is already pending`. A unit foi parada
para evitar mais requests nesse estado.

A segunda falha era um bug de limpeza: `process_ubatch()` abre a transação do
tail em `mctx->apply()`, mas chamava `graph_compute_finish()` só nos retornos
normais. A exceção de submit Vulkan saltava essa limpeza e deixava a transação
pendente para o request seguinte. Um guard agora finaliza a transação como falha
em qualquer saída excepcional; quando o grafo já começou, o cache invalida o
payload afetado. Isso não recupera o device Vulkan perdido, então o backend ainda
precisa ser recriado antes de aceitar novo trabalho após `ErrorDeviceLost`.

Os ~800–900 t/s anteriores não são uma baseline válida: o backend original
`ggml_local_split_exec` copiava Q/K/V, preenchia a saída com zeros via `memset`
e retornava sucesso sem calcular attention. O benchmark pulava o trabalho caro.

Os presets Qwen anteriores davam como base S=24.576, M=40.960, L=56.320,
XL=63.488 e XXL=102.400, com `batch/ubatch=256`. Revalidei esses patamares com
o modelo IQ3_XXS usado na produção, auto-placement e MTP por tier:

- 40.960/MTP4/tail2.048: prompt de 28.881 tokens a 751.79 t/s, decode 47.72 t/s.
- 56.320/MTP2/tail4.096: prompt de 28.881 tokens a 727.40 t/s, decode 46.86 t/s.
- 63.488/MTP2/tail2.048: prompt de 7.881 tokens a 809.98 t/s, decode 49.79 t/s.
- 102.400/target-only/tail2.048: prompt de 7.881 tokens a 861.29 t/s, decode
  30.88 t/s; 16 camadas permaneceram na 4070 e restaram 25 MiB livres.

MTP4 em 40.960 com tail4.096 falhou por falta de VRAM na alocação de 55.62 MiB
do kernel de atenção draft; com tail2.048 concluiu. MTP2 em 63.488 também só
concluiu com tail2.048. Com reserva de 700 MiB no XXL, uma camada passou à
Radeon e o teste caiu a 260.89/9.81 t/s; o preset usa reserva350M e mantém as
camadas locais até o limite de 102.400. A tentativa de 204.800 com 9 camadas na
Radeon foi interrompida quando o prefill desceu a 54.12 t/s em 3.840 tokens.
Esse tamanho segue sem validação segura.

`--no-offload-rs` continua opcional: manter RS na CPU mediu 154.28/8.34 t/s no
perfil curto, contra 775.44/53.19 t/s com RS na GPU. A Fase 5 tem agora um
protótipo experimental: prefill CUDA, cópia assíncrona dos records KVarN
selados e handoff do payload/tail para decode Vulkan. Sua validação de
throughput e os limites atuais estão registrados abaixo.

### Como remover o gargalo da Radeon

O ensaio controlado usou o mesmo modelo IQ3_XXS, contexto 102.400, tail FP16 de
2.048, `batch/ubatch=256` e prompt de 7.681 tokens. Com zero camadas remotas o
prefill mediu 847.95 t/s e decode 30.85 t/s. Com uma camada full-attention na
Radeon, usando o store workspace para ubatch256, mediu 254.84 t/s e decode
11.94 t/s. Um ensaio anterior da mesma classe marcou 270.02/11.93 t/s. A
diferença entre os dois resultados remotos não foi isolada em uma variável; use
254.84 t/s como a medição do código atual. Uma única fronteira CUDA↔Vulkan já
reduziu o prefill em aproximadamente 70%. Portanto, a queda não vem só de
enviar as 16 camadas: o caminho atual de atenção remota é caro mesmo com uma
camada.

Estratégias para remover esse gargalo, em ordem de investigação:

1. **Manter atenção local na 4070 quando o objetivo for throughput.** É o
   fallback atual, validado até 102.400 com os tiers Qwen configurados. O teste
   passou, mas esse contexto deixou pouca VRAM livre; não extrapolar para 204.800.
2. **Separar o tempo da atenção Vulkan do tempo de transferência.** Contar bytes
   Q/K/V CUDA→Vulkan e saída Vulkan→CUDA por ubatch, medir KVarN store e atenção
   por camada. O scheduler tenta `cpy_tensor_async` no backend de destino, mas
   `ggml_backend_vk_cpy_tensor_async` rejeita origem CUDA; `ggml_backend_sched`
   então sincroniza os dois backends e cai na cópia genérica via host. A nova
   sonda `GGML_BACKEND_COPY_PROFILE=1` registra bytes e tempos separados de sync
   da origem, sync do destino e cópia genérica para medir essa hipótese junto
   com o kernel Radeon. Combine-a com os timestamps Vulkan existentes, ativados
   por `GGML_VK_PERF_LOGGER=1`: o tempo `src_sync` também pode incluir trabalho
   Vulkan pendente (atenção/store/WHT), não apenas transporte.
3. **Reduzir custo de fronteira.** Adicionar staging host pinned e cópias
   assíncronas com eventos, agrupar Q/K/V e saída por ubatch e evitar
   sincronização por tensor. Validar 0/1/2 camadas com prompt, contexto, cache,
   clocks e `batch/ubatch` idênticos antes de liberar placement remoto.
4. **Implementar a Fase 5 real.** Gerar e quantizar KV no caminho CUDA e migrar
   records KVarN em chunks de 128–2.048 tokens, publicando cada chunk só após o
   fence de conclusão. O próximo chunk de prefill pode sobrepor a transferência
   anterior. A atenção do chunk seguinte precisa continuar vendo todo o prefixo;
   resolver isso sem manter duas cópias completas é a parte crítica. Cópia
   integral ao fim do prefill serve como baseline de correção, não como solução
   final: pode duplicar até ~1,8 GiB de KV.
5. **Falhar fechado.** Cancelamento, falha de cópia, fence ou device loss devem
   invalidar/resetar as duas cópias e impedir decode de estado parcial. Auto-
   placement não deve enviar camadas à Radeon só porque a conta de VRAM diz que
   cabem: o teste mostrou uma redução de 3,3× com uma camada.

Os resultados antigos de ubatch 256/512/1024 não provam migração: ubatch apenas
divide a execução do prompt. O código não copiava records KVarN entre devices.

O protótipo da Fase 5 usa cada grupo eager de 128 tokens como unidade de cópia,
uma fila assíncrona limitada a quatro jobs e staging host pinned de 4 MiB por
device. A próxima cópia pode sobrepor a computação CUDA do ubatch seguinte; o
handoff final sincroniza a fila e copia stage/tail. Ao voltar do decode Vulkan
para um novo prefill CUDA, sincroniza o contexto e atualiza o espelho antes de
trocar o owner. Uma falha invalida o espelho e mantém o owner ativo; não tenta
decodificar de um mirror parcial. A fila permanece desabilitada até recriar o
contexto depois de um erro de transferência. No placement `auto`, se a estimativa
da cache completa excede o orçamento seguro, o contexto volta ao placement
remoto estático em vez de falhar durante a transição adaptativa.

Esse primeiro corte ainda conserva as duas alocações dos records e exige que o
KV usado no prefill caiba na CUDA. O placement `auto` apenas ativa a migração
quando essa condição é satisfeita; o fallback estático conserva os tiers maiores.
Para liberar VRAM e suportar contextos maiores, falta migrar ownership sem
manter buffers completos duplicados, incluindo records, stage, tail e os
caminhos de restore/rollback.

Na mesma revisão, a atenção Vulkan KVarN foi otimizada e medida isoladamente na
Radeon: prefill256 caiu de 536.94 para 497.04 ms (7.4%), decode1 de 16.913 para
10.153 ms (40.0%) e verify4 de 38.312 para 32.106 ms (16.2%). O shader agrupa
reduções GQA, mantém Q em registradores, ignora tokens mascarados antes da
descompressão e distribui o tail entre splits. A variante wave64 dedicada foi
descartada por regressão. Isso reduz custo de atenção; não remove o custo de
fronteira CUDA↔Vulkan nem substitui placement conservador.

Validação integrada em 2026-09-30: build de `llama-server` e `test-kvarn`,
suíte KVarN completa em CUDA, 13 casos de paridade da atenção em Vulkan0/RADV e
Vulkan1/NVIDIA e paridade do store Vulkan com ubatch256 passaram. Duas requests
consecutivas com 4.216 tokens, ctx40960, KVarN4, tail2048 e batch/ubatch256
completaram; cada handoff de prompt registrou `prefill-migrate decode handoff
result=0`. O prefill mediu 825.31 e 847.77 t/s; o decode Vulkan, 16.05 e 17.16
t/s. Isso valida ida e volta entre requests no smoke test sem MTP, mas não
qualifica o preset de produção. Com MTP4 e uma layer explícita, duas requests
de 1.355 tokens completaram com 4/4 drafts aceitos e handoff `result=0`
(prefill 644.19/677.86 t/s; decode 28.73/38.35 t/s). Com `auto`, ctx24576
completou duas requests em migração (prefill 653.61/733.58 t/s; decode
34.21/42.41 t/s). Em ctx40960, a estimativa foi 852.0 MiB contra orçamento
767.5 MiB e o placement remoto estático foi usado com MTP ativo.

O perfil Vulkan instrumentado antes do novo split-K, com reserva700M (13 camadas
locais/3 Vulkan), contexto102.400, tail2.048, `batch/ubatch=256` e prompt7.681,
registrou 23.42 s acumulados em `FLASH_ATTN_EXT`, 1.20 s em KVarN store e 1.22 s
em cópias genéricas CUDA→Vulkan para 2.17 GiB de dados. As cópias Vulkan→CUDA
somaram 0.38 s. `src_sync` somou 1.23 s, mas inclui espera por trabalho CUDA já
enfileirado e não é tempo de transporte puro. O Vulkan timestamp logger força
sincronizações e esse ensaio serve para atribuir custo, não throughput absoluto.

O KVarN store agora permite a rota com workspace em ubatch256 (dois grupos
completos), com teste de paridade contra o oráculo CPU e comparação com a rota
monolítica. O teste confirma a correção, mas ainda falta uma medição A/B isolada
sem profiler para provar ganho de throughput. O ajuste de split-K recomendado
como linha de investigação permanece fora do código: o kernel ainda percorre
sequencialmente o eixo de tokens por grupo, e a divisão atual é guiada por
ocupação. É preciso separar ganho de paralelismo de custo de combinar splits e
do tratamento do tail antes de mudar essa política.

### Auditoria das outras fases ainda citadas pelo transcript

- **Fase 6 — cache em disco:** existe no código atual como `--slot-save-auto`,
  com fingerprint de modelo/KV/RoPE e verificação byte a byte dos tokens antes
  de restaurar (`tools/server/server-context.cpp`, bloco Auto disk prompt/KV
  cache). Validado em 2026-10-01 com o preset Qwen de produção: após salvar um
  prefixo de 6.846 tokens, reiniciar `llama-server-root.service` restaurou
  6.846 tokens do disco e reprocessou 1 token (`cache_source=disk`). Os dois
  snapshots de teste foram removidos depois da validação.
- **Fase 7 — batch remoto/DFlash:** o protocolo RKVA já distingue prefill
  batched (`n_tokens >= 1`) de decode (`n_tokens == 1`) e o caminho DFlash tem
  contexto/draft model separado. O MTP do Qwen de produção completou o smoke
  (21/24 tokens propostos aceitos); com uma camada e handoff Vulkan, duas
  requests manuais aceitaram 4/4. Falta medir DFlash remoto. Isso não torna a
  Fase 5 existente.
- **Fase 8 — IQ3_S:** o arquivo de modelo de 12.120.016.896 bytes existe, mas a
  caracterização no hardware atual ainda não foi feita. Não há baseline validada
  de placement, contexto, prefill, decode ou MTP/DFlash para esse modelo.

## Plano inicial com Qwen3.8-27B IQ3_XXS (registro histórico invalidado)

> **Status da Execução:** Em andamento (Compilação concluída, Profiling Fase 0 ativo)
> **Branch:** `integrate-pending-dflash-and-adaptive-mtp`
> **Dispositivos:** RTX 4070 12GB (CUDA) + Radeon 780M (Vulkan:0 / RADV Phoenix)
> **Configuração APU/Radeon:** TDP = 20W (`apu-tdp 20`), Clock da APU = 2700 MHz (`pclockmax 2700`)

### Tabela de Progresso das Fases

Os status e números nesta tabela foram escritos antes da correção de validade
acima. As fases 0–3 e as medições remotas da fase 4 usaram o backend que
retornava atenção zerada; ficam registrados apenas como histórico e não como
resultado aprovado.

| Fase | Descrição | Status | Resultados / Notas |
|---|---|---|---|
| **Fase 0** | Profiling baseline (4070 pura, IQ3_XXS) | ⚠️ **Histórico inválido para comparar com o caminho corrigido** | Os números antigos foram obtidos com a atenção local-split que não calculava o resultado. Baseline atual válida: 102.4K, IQ3_XXS, b/ub256, tail2.048: 847.95 t/s prefill e 30.85 t/s decode. |
| **Fase 1** | Backend local-attn Vulkan (In-process) | ❌ **Abandonada** | `ggml-local-split` copiava Q/K/V e retornava saída zerada; substituído pela atenção nativa Vulkan via scheduler e KVarN. |
| **Fase 2** | 1–2 layers remotas (Validação ring buffer) | ❌ **Medições invalidadas** | O ring buffer antigo não executava atenção correta. A rota atual de uma camada remota mediu 254.84/11.94 t/s em teste controlado. |
| **Fase 3** | Placement variável (Curva TPS × N layers) | ❌ **Medições invalidadas** | Curva antiga não representa custo de atenção válido. Instrumentação atual identificou atenção Vulkan como o maior custo medido. |
| **Fase 4** | Auto placement baseado em VRAM | 🟡 **Parcialmente revalidada** | Estimativa agora inclui records, staging e tail por camada. Contexto de 102.4K foi validado com 16 camadas locais e reserva de 350M; placement remoto não é recomendado para throughput. |
| **Fase 5** | Prefill migration assíncrona | 🟡 **Protótipo opt-in** | Prefill CUDA, cópia em background por grupo KVarN de 128 tokens e handoff para decode Vulkan. MTP standalone e `auto` com fallback remoto foram testados. Ainda duplica records CUDA/Vulkan e não aumenta o teto de contexto; não ativado no preset de produção. |
| **Fase 6** | Prompt cache & Session lifecycle | ✅ **Round-trip validado** | O preset Qwen de produção salvou e, após reiniciar a unit, restaurou 6.846 tokens do disco e reprocessou 1 (`cache_source=disk`). |
| **Fase 7** | MTP / Multi-Token Batching ($N > 1$) | 🟡 **MTP validado; DFlash remoto pendente** | A geração Qwen de produção aceitou 21/24 tokens propostos; duas requests manuais com uma camada e handoff Vulkan aceitaram 4/4. Falta medir DFlash remoto. |
| **Fase 8** | Caracterização e Validação | 🟡 **Parcialmente concluída** | IQ3_XXS foi validado até 102.4K no caminho atual; IQ3_S permanece sem caracterização. |

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
