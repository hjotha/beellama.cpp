# 002 — Fechar a regressão CUDA FP8 e reconciliar a review de performance

Data: 2026-10-20. Destinatários: AGY na GOKAYA → Codex Sol, mesma sessão `01a1583b-529b-7160-9cff-5d320ac39c24`, esforço xhigh.
Pedido explícito do usuário: analisar a regressão, conferir tudo da review anterior, encaminhar ao Sol correções viáveis e novas melhorias de performance. Executar implementação e validação, não apenas outra review.

## Estado e limites

- Bee: `/home/hjotha/beellama.cpp`, HEAD `ff687c6f1199913c7b0e06cfccae3ba0a0c95465`, branch `fix/rkva-serving-payload-20261020`, alterações não commitadas do Build 276.
- Xbox: `/home/hjotha/worktrees/xllama-rkva-serving-payload-20261020`, HEAD `86957662985742665ef1f88e51e567a26d1ddb0f`, mesma branch. Checkout principal Xbox apenas leitura.
- Reusar esses locais e preservar o diff existente. Bee tem exceção explícita de worktree; não abrir outro checkout. Não fazer reset/checkout destrutivo. Antes de editar, salvar hashes/diff da árvore suja; HEAD sozinho não identifica o candidato.
- Esta revisão alterou somente este documento e seu índice. Não executou novos testes nem benchmarks.
- Review original: `/home/hjotha/beellama-gate/20261020/review-otimizacoes-codex-gpt61-master.md`, complementada pelos `codex-review-phase*.md` da mesma pasta.
- Evidência anterior: `/home/hjotha/build/rkva-serving276-evidence/report.json`, `verification.json`, patches R1/R2 e manifests. Isso prova execuções anteriores, não os novos candidatos.
- Não promover candidato regressivo. Respeitar AGENTS dos projetos, preservar serviço/usuários ativos e não modificar rede ou máquinas fora do fluxo existente GOKAYA/HPBEATS/Xbox. Publicação/integracão somente conforme política aplicável e gates completos; não confundir instalar pacote de teste com homologar produção.

## Resultado e correções de interpretação

Comparação mais limpa disponível: MESMO Build 276, compact-stage e FP8 no fio, alternando codec host. Ainda só uma execução por variante:

| Contexto | CPU prefill final | CUDA prefill final | CPU decode | CUDA decode |
|---|---:|---:|---:|---:|
| 102400 | 117.6711 | 90.9953 | 18.5789 | 17.9849 |
| 112640 | 44.9628 | 46.2224 | 17.5475 | 14.3408 |

Unidade t/s. Preservar baseline 276 CPU (`GGML_RKVA_CUDA_FP8=0`) com compact-stage. Não apresentar desabilitar CUDA como solução definitiva: investigar, corrigir e comparar. Caso CUDA não vença após correção, manter opção experimental explícita e baseline qualificada, com conclusão honesta.

- TX caiu 4093932012 → 2991419884 bytes no ensaio inteiro; RX permaneceu 1642250284. Não é por ubatch nem prova de teto da rede.
- O pack CUDA é chamado UMA vez para `nq*out_width` antes do loop de segmentos (`ggml-remote-attn.cpp:1521–1540`). 3072 packs/unpacks agregam prefill+decode. Não repetir explicação de um pack CUDA por segmento de 8 queries como fato.
- 6467223552 bytes F32 evitados foram contados em CADA direção Q e OUT, não medição de tempo economizado.
- `xbox_prepare_us` inclui `xbox_compute_us`; RTT menos prepare não é rede pura. Separar trechos exclusivos e sobreposição.

## Reconciliação de todas as fases

| Fase | Estado realmente verificado | O que falta fechar |
|---|---|---|
| 0 profile | Parcial | JSON GPU FP8 tem profiling=true, samples_recorded=0, timestamp_resolves=0, bulk_calls=0, mas submits=fence_blocking_waits=3072. Falta cadeia host/GPU/RPC por fase/token/lote. |
| 1 codecs/contrato | Correções presentes | RNE/saturação, bounds, negociação estrita, SplitZip desativado; sanitizers/conformidade anteriores positivos. Gates de logits/KL/contexto longo/MTP não se inferem de unit tests. |
| 2 pool/cópias | Parcial no caminho completo | Slots host persistentes e stage F32 condicional presentes; ainda há vetores temporários por RPC no Xbox e falta prova pós-warmup. |
| 3 FP8 GPU | Parcial | Hook host/pinned integrado, mas performance regressiva. Xbox ainda converte FP8→F16 na CPU e OUT F16→FP8 na CPU. |
| 4 MQ/splits/Q-WHT | Implementação presente | Qrot/layout e tuning presentes; falta qualificar matriz física completa e ganhos A/B repetidos. |
| 5 inverse-WHT/reducer | Presente no RPC/shader | Correções de den/NaN e LSE RPC presentes. Outro merge position-split ainda tem clamp defeituoso; item separado abaixo. |
| 6 fila/ownership/TCP | Implementação presente | Dois slots, tickets/epoch e shutdown/close corrigidos; testes anteriores positivos. Completar matriz física de cancel/reset/restore conforme alterações feitas. |
| 7 wavefront | Scaffold/testes somente | `src/llama-wavefront.cpp/.h` não têm caller no runtime real. Não está implementado no serving/modelo. |

O Sol deve atualizar esta matriz com código, comando, saída e artefato de cada gate; não marcar fase inteira como concluída só por corrigir seu scaffold.

## P0 — Medir antes de atribuir causa (esforço M, risco baixo)

1. Instrumentar de verdade `uwp/rkva/rkva-kvarn-bulk-f16.inc` e o dispatch de `RKVA_KVARN_RANGE_BULK_ATTN`. Reusar profile/queries existentes. Guardar timestamps correlacionados sem sincronização GPU extra no caminho medido. Provar bulk_calls/queries, amostras e resolves não zero no FP8 físico.
2. Host: registrar por request/ubatch/token/layer/ticket/epoch os intervalos de preparação K/V, espera do produtor, pack Q, DMA D2H, fila RPC, send/recv, unpack/H2D, WAIT exposto e computação local. Separar prefill e decode; reportar mediana/p95, bytes e contagens. Não somar intervalos inclusivos/sobrepostos. Medir overhead profile on/off.
3. Manifesto de clocks/power reais: logs CPU e CUDA registram falha de lock da memória e power limit NVIDIA por `Insufficient Permissions`. Config INI não prova aplicação. Registrar clocks/temperatura/VRAM/governor reais e controlar condições entre variantes; não mudar potência para fabricar ganho.
4. Comparações devem manter modelo/token IDs/snapshot/seed, P=96256, C=131072, b=ub=256, wire FP8 e compact-stage. Separar cold restore, prompt inteiro, late prefill e decode. Os 63 intervalos de decode não incluem o primeiro token.

## P1 — Resolver regressão pelo caminho crítico (esforço M/L, risco alto, evidência de custo alta/causalidade ainda não medida)

**Hipótese A: colocação do op altera CUDA Graphs.**
- `ggml/src/ggml-cuda/ggml-cuda.cu:2824` desativa captura quando o subgrafo contém `GGML_OP_REMOTE_ATTN`; o caminho CPU colocava a orquestração fora desses trechos.
- NÃO remover esse guard: socket, mutex e sincronização host não podem entrar em captura.
- Medir quais subgrafos perderam captura/replay, quantidade real de launches, graph captures/replays por backend e tempo CPU de enqueue. O contador `graphs reused` do servidor não prova replay CUDA.
- Controlar também VRAM: logs reais em `/media/hjotha/GAMES/llama-slot-cache/candidate-276-fp8/server.log` e `candidate-276-cpu-codec-fp8/server.log` mostram ambos desativando graphs por pressão (17.44/17.38 MiB livres <18), liberando 75/109 captures. Logo a hipótese de graphs não está provada nem sozinha explica o decode.
- A/B diagnóstico pequeno CPU/CUDA com `GGML_CUDA_DISABLE_GRAPHS=1` e sem a variável (não usar `=0`, presença já desabilita, `common.cuh:1400`), observar estado efetivo e margem real. Se confirmado, manter orquestração fora dos trechos capturáveis com a menor mudança segura, preservando Q/OUT em VRAM e dependências; não reescrever scheduler inteiro.

**Hipótese B: espera host impede overlap.**
- `ggml-cuda.cu:2244`: `cudaStreamSynchronize(ctx.stream())` executa antes de submit E WAIT, seguido de evento.
- `remote-attn-cuda.cu:213,277,322`: pack/unpack sempre concluem via `cudaStreamSynchronize`; pack é chamado no submit antes de lançar trabalho local posterior.
- K/V ainda faz `ggml_backend_tensor_get` duas vezes (`ggml-remote-attn.cpp:1483–1487`); medir esse custo antes de eliminar cópia necessária ao produtor/rollback.
- Trace producer-ready→pack/DMA→socket e local attention; só então reduzir espera usando evento/lifetime por slot. TCP só pode ler dados prontos e consumidor só pode ver OUT validado. Preservar erro numérico, poison e retirement de falhas; não liberar buffer/epoch antes de DMA terminar.
- IMPORTANTE: R2 já tentou pular sync apenas no WAIT e trocar piso de scratch 4MiB por granularidade 1MiB, em conjunto. Piorou e foi revertido. Ver diff do patch final versus `r2/beellama-complete.patch`. Não repetir como novidade; isolar fatores somente com nova evidência.
- GPU/CPU decode e prefill podem ter break-even distintos. Não criar heurística de threshold antes de medir, nem fallback de wire silencioso. Se ambos os lados não vencerem, relatar limitação e manter seleção explícita qualificada.

## P2 — Novas economias concretas de trabalho (esforço S/M, risco baixo/médio)

1. **Alocações temporárias por RPC no Xbox:** `uwp/rkva/remote-attn.cpp:667–673` cria e redimensiona `temp_out_f16` e `temp_lse` em toda chamada; `rkva-kvarn-bulk-f16.inc:59` cria `segments`. Reusar scratch do dono serial existente com limites, sem estado global concorrente novo. Contar alocações pós-warmup por forma. É evidência concreta de que “zero alocações” ainda não está fechado; ganho precisa ser medido.
2. **Zerar e recopiá-los sem necessidade no host:** `rkva_byte_buffer::clear/resize` (`ggml-remote-attn.cpp:103–109,650+`) zera áreas novamente a cada uso; montagem do envelope cresce/zera e depois sobrescreve com memcpy, query pinned é copiada para payload. Medir bytes zerados/copiados e tempo. Eliminar somente áreas comprovadamente totalmente escritas antes do envio; manter padding determinístico e teste contra vazamento de bytes não inicializados. Não substituir por um transport complexo se não der ganho.
3. **Concluir FP8 direto no Xbox:** `rkva-kvarn-bulk-f16.inc:151–154` expande query para F16 na CPU e `remote-attn.cpp:694–700` converte saída F16 para FP8 na CPU. Implementar PSO direto somente após o profile localizar custo relevante, reaproveitando upload/readback e LSE F32. Preservar a semântica numérica efetiva do OUT; conversão F32→FP8 direta não é automaticamente byte-equivalente a F32→F16→FP8. Se alterar contrato, negociar/revalidar, não mudar silenciosamente. Medir envelope inteiro e TPS, não só microkernel.

## P3 — Fechar pendências originais sem inventar ganho

- Reavaliar MQ2/MQ4/splits por nq/contexto com perfil funcionando e um fator por experimento. Usar oracles existentes, causal masks, tail/domain, GQA e limites de nq. Ganho de kernel isolado não basta.
- Wavefront fase7: está ausente no runtime, não apenas desativado. Depois de estabilizar P0–P2, implementar integração mínima viável no modelo real se o trace provar trabalho independente que compense tiles 64/128; incluir FFN/RoPE/GQA/estado recorrente/KV/rewind. Não inventar prefetch de Q da próxima camada do mesmo token, que depende do resultado atual. Se inviável/sem benefício, registrar bloqueio técnico e artefatos; não declarar concluído por teste sintético.
- Correção separada: `src/llama-graph.cpp:6246–6257`, merge position-split não RPC, ainda usa clamp[-1e6,1e30]. O RPC foi corrigido em :5233. Testar LSE finito igual -2097152 (peso antigo zera) e extremos positivos, preservar semântica de vazio/NaN por caminho. Não atribuir isso à regressão CUDA medida; é alcance incompleto da correção anterior.

## Verificação e critérios de entrega

Comandos existentes identificados nos manifests, a executar SOBRE O NOVO CÓDIGO, reconstruindo antes:

```bash
cd /home/hjotha/beellama.cpp
cmake --build build --target llama-server test-remote-attn-range test-remote-attn-cuda test-rkva-kvarn-producer test-rkva-kvarn-gpu-producer test-wavefront-pipeline -j 6
GGML_RKVA_GPU_TEST=1 ctest --test-dir build --output-on-failure -R '^test-(remote-attn-(range|cuda)|rkva-kvarn(-gpu)?-producer|wavefront-pipeline)$'
/opt/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 99 build/bin/test-remote-attn-cuda
python3 tests/test-cuda-graph-source-properties-static.py
```

Verificar existência dos targets antes de construir; ajustar nomes apenas conforme CMake real. Reusar suites ASan/UBSan/TSan e oracles/produtor→Xbox descritos em `verification.json`; reconstruir binários temporários, não executar stale e chamar de validação nova. Shader alterado exige DXC + UWP Release HPBEATS + hashes de pacote/DXIL e teste no Xbox; deploy de teste preserva LocalState.

- Gates de sincronização: testes com trabalho CUDA local entre submit/WAIT, concorrência controlada, buffer overwrite após retirement, cancel/disconnect/reset/restore com trabalho pendente, sem aceitar saída inválida ou epoch antigo.
- Gates numéricos: conformidade FP8, NaN/Inf/overflow, LSE/OUT oracle e precisão de contexto longo. Se tocar MTP/wavefront, validar rejeição/rewind e tokens aceitos/s, não somente decode sem especulação.
- Benchmark driver existente: `/home/hjotha/beellama-gate/20261020/run_bench_276_cpu_codec.py` e `run_bench_276.py`. Não sobrescrever pastas históricas: adaptar saída para novo ID único, arquivar source diff/hash, controles/manifest e todas as repetições.
- Diagnóstico inicial enxuto; candidato final em pelo menos 5 pares A/B alternados depois de warmup conforme review original, ou reportar precisamente impossibilidade/variância. Mediana/dispersão/diferença, ms/token por fase, late/total prefill, TTFT/restore, decode e wall. Não escolher somente melhor rodada.
- Concluir regressão resolvida apenas com correção + evidência de não regressão nos dois contextos e ganho acima de ruído no alvo. Não prometer percentual antecipado. Se inviável, explicar causa medida, manter baseline CPU e documentar descartes.
- Entrega final deve separar implementado/testado/instalado/integrado/homologado e fases ainda abertas, com caminhos de logs e resultados reais. AGY deve acompanhar o Sol e consolidar, sem encerrar após apenas encaminhar o prompt.

## Encaminhamento

AGY: leia integralmente este steer e retome o Sol na sessão indicada usando o wrapper existente `/home/hjotha/beellama-gate/20261020/codex_review_sol.sh` (lock já evita corrida). Passe este arquivo como prompt e use saída/log novos, identificados por esta revisão. Não iniciar dois Sols no mesmo checkout. Confirmar início pelo processo e transcript, não apenas pela intenção. Usuário autorizou explicitamente encaminhar e implementar; não pedir nova confirmação rotineira.
