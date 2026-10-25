# Plano de implementação: DFlash2 na Radeon para o tier XXL

Data: 2026-10-23.
Estado: planejamento; implementação e benchmarks novos ainda não executados.
Base inspecionada: `main` em `ef120ee6f`; servidor observado inicialmente com
build `b12448-ec6995e6c`.

## 1. Objetivo e decisão recomendada

Manter MTP nos tiers atuais S/M/L/XL e usar DFlash2 na Radeon integrada como
draft exclusivo do tier XXL, onde MTP está desligado por limite de VRAM.
O alvo continua na RTX 4070 eGPU, com contexto máximo configurado de 102.400
tokens, KV KVarN4 e verificação especulativa pelas APIs upstream existentes.

Começar pelo split local compacto já implementado, depois adaptar as melhorias
recentes do RPC Xbox para uma ponte local CUDA/host/Vulkan: buffers persistentes,
staging pinned, conversão antes do D2H e dependências por evento. Não adicionar
TCP entre GPUs no mesmo computador. F16 é o primeiro transporte reduzido;
FP8 E5M2 é uma variante posterior condicionada a ganho líquido e aceitação.

A viabilidade funcional tem precedentes locais. O ganho no modelo atual em
73–102K ainda é hipótese: comparar com XXL sem draft, não com MTP de tier menor.
Este plano não autoriza executar automaticamente os runners históricos que
param serviços, nem representa implantação ou mudança do preset de produção.

## 2. Ambiente e configuração de referência

Estado observado durante o estudo; confirmar novamente antes de implementar
ou medir, pois há outras tarefas trabalhando no repositório e nos serviços.

| Item | Referência observada |
| --- | --- |
| Host | gokaya |
| Alvo | `/home/hjotha/models/Huihui-Qwen3.8-27B-abliterated-Swift-1.5-GSQ-RCO-IQ3_XXS-mtp.gguf` |
| Draft inicial | `/home/hjotha/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf` |
| Preset | `/home/hjotha/router-production.ini` |
| Binário | `/home/hjotha/beellama.cpp/build-optimized/bin/llama-server` |
| CUDA | RTX 4070 12 GB, `CUDA0`, ligada por ponte Thunderbolt 3 |
| Radeon | Ryzen Z1 Extreme / RADV PHOENIX, `Vulkan0`, RAM compartilhada |
| Build | CUDA e Vulkan habilitados; HIP desligado |
| Concorrência | Um slot, um modelo ativo |
| Alvo | Flash Attention on; fit off; gpu-layers 99; context shift off |
| Sampling | temp 1.0, top-p 0.95, top-k 20, min-p 0 |
| Penalidades | repeat 1.1, frequency 0.3, presence 0 |
| Reasoning | low; loop guard force-close |

O link reportou x4 e geração reduzida em repouso. Isso não mede a banda efetiva
sob carga nem demonstra um teto de transferência fim a fim. Medir bytes e
latências reais; não derivar TPS apenas da largura PCIe nominal.

| Tier atual | Contexto | Batch/ubatch | MTP n_max |
| --- | ---: | ---: | ---: |
| S | 24.576 | 256/256 | 6 |
| M | 38.912 | 256/256 | 4 |
| L | 56.320 | 256/256 | 2 |
| XL | 72.704 | 128/128 | 2 |
| XXL | 102.400 | 256/256 | 0 |

O XXL deve continuar sem attention do alvo na Radeon neste primeiro trabalho.
O perfil XXXL, Xbox e overflow de attention ficam fora da primeira integração.

## 3. O que já existe e onde alterar

| Arquivo | Papel no trabalho |
| --- | --- |
| `common/arg.cpp`, `common/common.h` | Flags/preset e parâmetros do draft/transporte |
| `common/speculative.cpp` | Split local, captura/injeção, draft, seletor e lifecycle |
| `src/models/dflash.cpp` | Encoder de features, transformer e saída hidden sem logits |
| `src/llama-context.cpp`, `include/llama.h` | Extração de taps e eventual API interna de transporte |
| `tools/server/server-context.cpp` | Política por tier, rebuild e restore |
| `ggml/src/ggml-cuda/remote-attn-cuda.cu` | Base de codec CUDA, eventos, scratch e pinned |
| `ggml/include/ggml-remote-attn-cuda.h` | Contrato das interfaces CUDA existentes |
| `ggml/src/ggml-remote-attn/ggml-remote-attn.cpp` | Referência de slots, pools, tickets/epochs e trace |
| `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | Upload/readback, host importado e sincronização Vulkan |

O split atual é selecionado por `--spec-draft-local-split`, com
`--spec-type draft-dflash` e dispositivo draft `Vulkan0`. Ele mantém transformer
e KV no draft, e head/seletor junto do alvo. `dflash_split` retorna hidden
normalizado sem transferir logits do vocabulário inteiro.

O roteamento adaptativo ainda não seleciona DFlash por tier:
`apply_profile_params()` força MTP se `draft_n > 0`, e nenhum draft caso zero;
`rebuild_active_speculation()` também trata essa decisão como `use_mtp`.
Portanto, aumentar `spec-draft-n-max-xxl` não implementa a proposta.

Preservar os contratos upstream de sampler, verificação, checkpoint e rollback.
Não restaurar verifier privado, ring/tape removido ou reduced-verifier antigo.

## 4. Geometria real, banda e memória

Metadados lidos no GGUF DFlash2 Q4_K_M local:

- Cinco camadas; embedding 5.120; taps `[6, 20, 34, 48, 62]`.
- Entrada do encoder: 25.600 valores por token.
- Bloco treinado: oito posições; até sete propostas por bloco.
- Seletor top-k 16, rank 256.
- Todas as camadas usam SWA de 2.048 tokens; oito KV heads, head dimension 128.
- Comprimento de contexto declarado: 262.144, distinto da janela física de KV.

| Payload lógico, sem framing e metadados | F32 | F16 | FP8 |
| --- | ---: | ---: | ---: |
| Features por token processado | 100 KiB | 50 KiB | 25 KiB |
| Hidden de oito posições | 160 KiB | 80 KiB | 40 KiB |

Logits F32 de oito posições e vocabulário 248.320 ocupariam 7,58 MiB.
O split compacto já evita esse retorno, aproximadamente 48 vezes maior que
os hidden states. Contabilizar também tokens rejeitados, embeddings e replay.

Enviar features F32 de 102.400 tokens acumula 10.000 MiB de tráfego lógico;
uma janela de 2.048 acumula 200 MiB. Isso é volume calculado, não medição de
latência, memória residente ou economia já implementada.

O draft Q4_K_M ocupa 1.143.006.816 bytes em arquivo. No placement atual, cerca
de 68,91 MiB de pesos do seletor ficam na CUDA, reutilizando o head do alvo.
O KV F16 para cinco camadas e 2.304 posições tem payload de 45 MiB, sem incluir
scratch, padding adicional, estados, pesos ou overhead do backend. Confirmar
`swa_full=false` e a capacidade realmente alocada; não criar KV denso de 102K.

Planejar margem CUDA para seletor, logits, outputs de taps, rollback recorrente,
verificação em bloco, CUDA Graphs e pico de prefill. A API RPC v2 usa scratch
mínimo de 4 MiB por contexto de codec; pools e duas filas não são gratuitos.
Não presumir que retirar MTP garante espaço para todos esses componentes.

## 5. Reaproveitamento das otimizações Xbox

| Commits recentes | Reaproveitamento proposto | Limite |
| --- | --- | --- |
| `f7730f5a3` | Submit/espera separados e transporte F16 agrupado | Overlap de attention local/remota tem dependências diferentes do draft |
| `6521463d1` | Evitar ida/volta CPU ao produtor CUDA; conversões vetorizadas como referência | D2D ali não é NVIDIA→AMD direto; codec CPU após D2H não reduz banda da eGPU |
| `aaa63758a` | Pools de capacidade reutilizável | `writev`/QUICKACK são específicos de TCP |
| `ff687c6f1` | Codec FP8 CUDA, pinned, identidade de jobs e ownership | Shaders Xbox não executam diretamente no Vulkan; wavefront é scaffold sem caller no serving |
| `7500b991b` | Eventos do produtor, retirement, tracing e lifetime de scratch | Codec CUDA já teve regressão; comparar ponta a ponta |
| `3e551c8a7`, `9c9a884cc` | Separação entre representação e transporte, deltas e fronteiras confirmadas | Produtores Q4/Q8 de KV remoto não são o payload de features DFlash |

A interface CUDA v2 conclui pack/unpack e DMA antes de retornar, inclusive em
erro. Reutilizá-la num worker não a transforma automaticamente em API totalmente
assíncrona. O unpack aceita ausência de LSE com `n_head=0`, mas ainda exige
integração ao contrato de hidden do DFlash.

Vulkan não aceita origem CUDA no caminho de cópia assíncrona existente.
`cudaHostAllocPortable` também não torna o buffer automaticamente importável no
Vulkan. Verificar `VK_EXT_external_memory_host`, alinhamentos, coerência e fences;
se não houver compartilhamento compatível, usar staging explícito e medir a cópia.

## 6. Arquitetura proposta

```text
4070: target → cinco taps → gather/pack F16 antes do readback
                                  ↓
                    slots host pinned persistentes
                                  ↓
Radeon: unpack → encoder → atualização KV SWA → transformer DFlash2
                                  ↓
                 hidden F16 → staging → unpack CUDA
                                  ↓
4070: head + seletor → verificação upstream → prefixo confirmado
```

Primeiro estabelecer a referência com F32 no split atual. A ponte F16 deve
interceptar tensores enquanto ainda residem no backend produtor. Não usar o
getter F32 como origem e depois recodificar: ele já transferiu os bytes pela eGPU.
Avaliar gather dos taps num único payload, respeitando o lifetime de cada tensor
e as dependências do grafo; não reter ponteiros de scratch que o scheduler reutiliza.

Usar fila pequena e limitada, inicialmente dois slots, com um dono do contexto
draft. Cada job identifica request, geração/epoch, ticket, posições e fronteiras
tentativa/confirmada. Não executar operações concorrentes no mesmo contexto
llama sem contrato explícito. Consumidor só usa dados após conclusão/validação.
Cancel/reset/restore/troca de tier drenam ou cancelam jobs antes de liberar recursos.

Dependências impedem esconder todo o draft atrás da verificação. No prefill,
investigar pipeline entre injeção do chunk anterior e computação target do próximo.
No decode, sobrepor apenas trabalho comprovadamente independente. A previsão
de âncora/bloco futuro (`local-prefetch`) permanece desligada no primeiro candidato.

## 7. Bootstrap, janela SWA e persistência

Em prefill frio, começar alimentando o draft pelo caminho existente. Depois
qualificar alimentação somente do sufixo exigido pela janela, com margem para
âncora, bloco, máscara, conv e rollback. Derivar essa margem do código; não fixar
2.048 como corte sem testar a fronteira exata.

As features target já incorporam o contexto completo. O KV salvo do target não
contém automaticamente esses taps; os últimos IDs de tokens sozinhos não
reconstroem suas ativações, sobretudo com estado recorrente do Qwen.

Para ativação após restore ou mudança de tier, escolher e testar uma estratégia:

1. Capturar sufixo necessário no prefill/replay que o target já executa.
2. Persistir estado auxiliar/janela de features com identidade e formato próprios.
3. Fazer replay a partir de checkpoint target compatível anterior ao sufixo.

Não impor captura contínua nos tiers MTP antes de medir seu custo. Se faltarem
features válidas, continuar sem draft ou executar reconstrução explícita; nunca
usar estado auxiliar desalinhado. Persistência deve incluir identidade de modelo,
taps, posições, codec, precisão e fronteira confirmada.

## 8. Etapas de implementação e gates

### Etapa 0 — Baseline e instrumentação

- Confirmar estado Git, serviço e outros ensaios ativos; preservar alterações alheias.
- Registrar commits, diff, hashes de binário/modelos, drivers e comandos exatos.
- Medir XXL fixo sem draft e split DFlash F32 existente, sem alterar produção.
- Separar captura, D2H, rearranjo, H2D Vulkan, injeção, transformer, retorno,
  seletor e verificação. Incluir waits dos getters; não atribuir residual a kernel.
- Contar bytes, alocações após warmup, picos CUDA/RAM e CUDA Graphs efetivos.

Gate: execução correta e dados completos. Se o split F32 não couber, registrar
o ponto de OOM e avaliar placement alternativo antes de reduzir contexto do alvo.

### Etapa 1 — Lifecycle e buffers persistentes

- Extrair/reutilizar abstrações pequenas de slot, staging e trace, sem copiar o
  backend RPC inteiro nem criar dependência de socket no DFlash.
- Reutilizar batches, buffers e capacidade; garantir leases/lifetime por job.
- Testar cancelamento, erro, drain e reuso antes de adicionar overlap.

Gate: mesmo contrato F32 e ausência de estado contaminado; memória limitada.

### Etapa 2 — Ponte F16 e redução de cópias

- Pack/gather CUDA antes do D2H; consumo Vulkan sem roundtrip F32 extra.
- Retorno hidden convertido no produtor Vulkan e consumo pelo seletor CUDA.
- Manter F32 selecionável para A/B; rejeitar formatos incompatíveis claramente.
- Distinguir payload físico transferido de volume lógico e cópias internas.

Gate: bytes principais reduzidos pela metade, conformidade numérica do codec,
aceitação medida e resultado ponta a ponta sem regressão fora da dispersão.

### Etapa 3 — Overlap limitado e bootstrap por sufixo

- Producer-events e espera no consumidor, sem retirar sincronização necessária.
- Pipeline de injeção/preparação durante prefill com backpressure.
- Alimentação SWA por sufixo somente após validar seu estado contra caminho completo.
- Aplicar tickets/epochs, fronteiras imutáveis e retirement em falhas.

Gate: ganho exposto em trace e no tempo total; invariantes de rollback preservados.

### Etapa 4 — Seleção por tier

- Separar política `tipo de draft` de `n_max`; permitir MTP, DFlash e nenhum draft.
- Definir parâmetros/modelo/device/cache do DFlash XXL sem herdar opções MTP
  incompatíveis, inclusive o threshold de confiança atual de 0,70 sem avaliação.
- Criar/destruir recursos por transição, preservando política de unload MTP.
- Atualizar sizing de outputs, checkpoints e janela recorrente para verificação.
- Integrar restore, retorno XXL→XL e falha de bootstrap sem estado parcial.
- Manter preset atual como default; nomes de flags/INI novos serão definidos
  conforme o padrão existente e documentados somente após implementação.

Gate: S/M/L/XL preservam comportamento e desempenho; XXL ativa DFlash somente
quando seus recursos e estado estiverem válidos; falhas têm fallback explícito.

### Etapa 5 — FP8 e alternativas condicionais

- Adaptar codec E5M2 CUDA e implementar equivalente Vulkan com contrato comum de
  rounding, saturação, NaN/Inf, dimensões e validação.
- Comparar F32/F16/FP8 para aceitação, tempos e memória; não assumir que ganho
  de banda compensa conversão ou pressão de CUDA Graphs.
- Se banda ainda dominar e houver VRAM, testar encoder `fc` + norm na CUDA:
  reduz features de 25.600 para 5.120 valores/token, mas acrescenta cerca de
  70,31 MiB de pesos Q4, scratch e computação CUDA.
- Se seletor CUDA impedir o XXL, testar head/seletor replicados na Radeon.
  O head atual tem 644,14 MiB e o embedding 388,38 MiB; não duplicar embedding
  inteiro sem necessidade, pois âncora e MASK podem continuar sendo fornecidos.
- Q2/Q3 do draft são candidatos opcionais; Q4 permanece referência inicial.

Gate: promover somente variante com ganho líquido e memória/aceitação adequadas.

## 9. Validação funcional e de desempenho

Testes novos devem exercitar comportamento real, não apenas presença de strings:

- F32/F16 e depois FP8: tamanhos ímpares, alinhamentos, bounds, extremos,
  arredondamento, NaN/Inf e erro antes de publicar saída parcial.
- Slots: cancel/reset com job pendente, epoch antiga, reuso, erro de backend,
  retirement antes de liberar staging e sequência de posições.
- Draft: aceitação total/parcial/zero, bônus, trim, janela SWA, conv, repetição
  após cancelamento, restore e entrada/saída do XXL.
- Comparar estado da injeção completa e por sufixo; validar rollback target
  recorrente pelas APIs upstream. Não relaxar tolerância para esconder falha.
- Greedy como diagnóstico; sampling real temp 1.0/top-p 0.95/top-k 20 e
  penalidades atuais como critério de serving. Registrar divergências; diferenças
  numéricas entre trajetórias não permitem alegar identidade exata sem evidência.

Escada: smoke curto, cerca de 74K, 90K e próximo do teto XXL. O limite de
102.400 é contexto total: reservar espaço para resposta e verificação, não
enviar prompt de 102.400 esperando ainda gerar uma resposta dentro dele.

Comparar: A=XXL sem draft; B=split F32; C=pooled/F16; D=overlap;
E=FP8 opcional. Alterar um fator por experimento, com prompts preservados de
código, texto e repetição. Repetição isolada não representa uso geral.

Usar pares alternados após warmup, inicialmente cinco pares por cenário-chave;
expandir apenas se a variância impedir decisão. Registrar mediana, dispersão,
p95, prefill frio, restore, late prefill e decode separadamente. Não somar
intervalos inclusivos/sobrepostos nem confundir volume evitado com tempo poupado.

Métricas obrigatórias: tokens confirmados/s, avanço por rodada incluindo bônus,
aceitação por posição, custo do draft/seletor/verificador, bytes físicos em
ambas direções, sincronização exposta, memória pico/steady, clocks/potência
observados e estado efetivo dos CUDA Graphs. APU/CPU compartilham orçamento:
registrar interferência térmica/energética sem atribuir causa só por correlação.

Condição de ganho: `tempo_total_da_rodada / tokens_confirmados` menor que o
tempo/token do baseline. Não há promessa de multiplicador de TPS neste plano.

## 10. Evidências históricas e limites

As medições de setembro usaram outro target, 16K, KV q4_0 e greedy. Não validam
o Huihui/Swift atual com KVarN4 em 100K. O híbrido simultâneo MTP+DFlash teve
resultado negativo e não é o objetivo desta implementação.

- `docs/mtp-dflash2-hybrid-plan.md`
- `docs/benchmarks/mtp-dflash2-etapa0-20260929-122824/README.md`
- `docs/benchmarks/mtp-dflash2-etapa0-q2-20260929/README.md`
- `docs/benchmarks/mtp-dflash2-shadow-corrected-20260929/README.md`
- `docs/hybrid-attention-master-review.md`
- `docs/beellama-args.md` — contrato e política do codec CUDA RPC.

Usar os timers corrigidos: draft + retorno mediu aproximadamente 38–52 ms e
seletor + retorno 3–4 ms na bateria de observação histórica. Os antigos tempos
de kernel aparentes omitiam waits; não reutilizá-los como custo puro de compute.

## 11. Entregas e conclusão do trabalho futuro

1. Baseline auditável com modelos atuais e breakdown de custo/memória.
2. Ponte local F16 com pooling, eventos e testes de lifecycle/codec.
3. Bootstrap SWA e política DFlash XXL com transições/restore verificadas.
4. Relatório A/B com decisão de manter, ajustar ou descartar cada otimização.
5. FP8 e placements alternativos somente se justificados pelas etapas anteriores.
6. Documentação de flags reais e promoção explícita após os gates de serving.

Desenvolver em branch dedicada no checkout principal, conforme a exceção do
projeto para BeeLlama; não criar worktree. Preservar alterações concorrentes.
Antes de qualquer teste que use as GPUs ou altere serviço, conferir operações
ativas e coordenar a janela; registrar parada/restauração e saúde real quando
ocorrerem. Este documento não executou build, teste, benchmark ou deploy.
