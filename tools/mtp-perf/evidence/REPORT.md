# Revisão 2 — somente código e testes CPU — 2026-11-16

## Resultado

Corrigidas as seis lacunas no clone /home/hjotha/beellama-tps-model-20261116,
branch experiment/mtp-tps-model-20261116, base
19f92e9dff3757683a09367785b71e1be8d7f6d0.

1. Default 4k/8k/16k/24k/32k; plan valida contagem independente/rank do fit e
   holdouts antes de execução. Run exige janela autorizada e um ponto explícito.
   Nenhuma expansão automática de matriz.
2. Identidade obrigatória nas rows e JSON/CSV: hardware/GPU/driver, modelo,
   binário/build, backend/kernel/KV/graphs, sampling, B/UB/pmin, corpus, geração e
   parallel. Rejeita ausências/misturas/digests/slices incompatíveis. Synthetic é
   um caminho explícito. Hash de modelo não foi executado nesta etapa.
3. Caminho físico implementado por família com roofline de bytes pesos/KV/FLOPs,
   eficiências limitadas, custos host/launch/comunicação separados. Retorna TPS,
   breakdown, envelope de sensibilidade, domínio e proveniência. Prior-transfer
   distingue-se de target-calibrated; calibração exige probes identificáveis e
   evidência de hardware/workload/domínio. Coeficientes empíricos são locais.
4. Sobrevivência por posição e bonus observado em ciclos elegíveis; actual
   committed/cycle usa todos os ciclos. Posição não é inventada para replay/stop;
   exclusões são registradas. Parser MTPSTATS aproveita os contadores existentes,
   explicitamente cumulativos incluindo warmup. Pmin não é probabilidade de aceite.
5. Contexto prompt e contexto medido são distintos; anchors brutos e média/min/max
   preservados. Coverage exige bootstrap único fora do ciclo, IDs consistentes,
   tempo e contagens compatíveis. TPS de ciclo e ponta-a-ponta têm denominadores.
6. Extrapolação é rejeitada por padrão. Holdout de borda a identifica explicitamente.
   Testes cobrem identidade, plano, troca fictícia de hardware, host, BW/compute,
   sobrevivência/cap, domínio, faixas e prior versus calibração.

## Verificações realmente executadas

- 24 testes CPU aprovados em revision-tests.txt.
- compileall Python e git diff --check aprovados.
- CLI plan executado: evidence/plan-revision/plan.json, desenho identificável,
  memory_qualified=false.
- Teste da barreira CLI: chamadas de GPU/comandos externos substituídas por mocks
  que falham se invocados; run sem autorização/ponto foi recusado antes delas.
- Nenhuma medição/inferência GPU, build CUDA, profiling, consulta nova de GPU,
  parada/reconfiguração de serviço ou leitura/hash do modelo de 10GB.
- Nenhum commit, push ou merge. Checkout original não editado.

Os valores dos testes físicos são fixtures fictícias. Mesmo o teste de
target_calibrated leva data_kind=synthetic. Não há previsão numérica para 4070/T4,
erro real estimado, novo resultado TPS ou tier validado.

## Entregáveis

bench.py, model.py, identity.py, acceptance.py, physical.py, test_model.py,
test_revision.py, README.md. Patch C++ opt-in original de 7 linhas mantido,
ainda não compilado. server-instrumentation.patch permanece no pacote.
REPORT-initial.md/baseline.json/guard-real.json guardam a etapa anterior.

## Riscos e limitações remanescentes

- Identidade de kernel/rotas e completude do manifesto de bibliotecas dependem de
  qualificação externa. Hash do manifesto + verificação de artefatos não prova
  que toda biblioteca dinâmica foi inventariada.
- Hardware_id de runner usa hostname/arquitetura, com GPU UUID/driver separados;
  clocks/power/CPU drift demandam telemetria e repetição.
- Runner fixa graph-off via GGML_CUDA_DISABLE_GRAPHS=1; não é equivalente à produção
  nem valida graph-on/recapture. Essa escolha está na identidade e README.
- Bootstrap/coverage e integração de traces devem ser confirmados no primeiro
  run real; agora falham de modo conservador quando a convenção não bate.
- Custo físico soma caminho serial. Sobreposição, waves/SM/shape e atenção/KVarN
  requerem famílias/domínios qualificados; não há eficiência universal.
- Classificação de probes memory-bound/compute-bound deve ser sustentada por
  profiling real. Calibração recusa dados insuficientes; isso não prova que
  poucos probes bastam para erro pequeno.
- Envelope é sensibilidade dos parâmetros fornecidos, não IC estatístico;
  erro de modelo/bytes/FLOPs e correlações podem excedê-lo.
- Sobrevivência elegível exclui stops/replays; contadores cumulativos não podem
  ser atribuídos a requisições sem snapshots/deltas apropriados.
- Nenhuma transição KV, threshold N/B/UB, zero OOM ou precisão cross-arquitetura
  validada. Sem autorização atual para sair de CPU-only.

Próxima etapa continua dependendo de janela do Henrique. Primeiro build isolado
e qualificação curta; só depois medições e expansão progressiva. Não parar
produção por iniciativa do agente.

## Context clarification — 2026-11-17 (documentation only)

Objective: maximize effective TPS per dynamic context tier with separately
qualified N/B/UB choices, not maximize VRAM occupancy. p_min per tier already
exists in serving and will be tuned later. Current fixed controls are for
identification; joint tier optimization remains unimplemented and unvalidated.
The Colab 16 GB approximate 100 MB headroom is not a validated safe reserve or
production instruction. See README for candidate comparison and transition
requirements. This addendum changes documentation only: no GPU work, service
change or numerical performance claim. The revision2 archive predates this
addendum and its original hash remains unchanged.
