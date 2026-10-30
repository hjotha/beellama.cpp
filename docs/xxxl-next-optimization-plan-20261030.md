# Plano de otimização adicional do XXXL

Data: 30/10/2026. Escopo desta entrega: análise do código, branches, clones e evidências existentes; nenhuma nova medição de inferência, build ou alteração de produção.

## 1. Objetivo e restrições

Minimizar o tempo de request do Qwen3.8-27B entre o início do XXXL e o contexto máximo, preservando qualidade, KVarN4 e estabilidade. Não confundir ganho em kernel com ganho no request nem ganho de cache com aceleração de prefill novo.

Invariantes para a campanha:

- XXL local até 102400; XXXL a partir do orçamento de 102401 tokens de prompt + saída. P=102400. Não variar P independentemente do teto local/limiar remoto nesta campanha.
- XXL B512/UB256; referência XXXL B1024/UB512, 2×N256, N=0, native KVarN4, tail128, bulk FP8, kernel fine, produtor CUDA e TCP prefetch.
- Guard de graphs em 18 MiB; procurar margem real acima disso, não apenas uma leitura favorável após a execução. Não enfraquecer o guard para obter TPS.
- Preservar snapshots, backups e trabalho alheio. Aumentar quota ou remover snapshots não equivale a implementar um ciclo de vida correto.
- CUDA: produção parada, build isolado e `-j10`, sem inferência simultânea. Restaurar serviço/flags em `finally`, inclusive se o teste falhar. UWP/MSVC e execução no Xbox têm validações próprias.
- Todas as mudanças são opt-in inicialmente. Parâmetros de tiers locais não devem mudar silenciosamente por otimização do XXXL.

## 2. Estado confirmado

O checkout está em `e8a1380e2485ae27220001337ed5982122fde803`. Produção usa o release baseado em `724613730` com a alteração de restore direto do servidor, não uma build integral desse main. Essa diferença precisa permanecer explícita em qualquer comparação.

Configuração verificada: `/home/hjotha/router-production.ini`, SHA-256 `227f2e50e5b25e6e0e982eba645aa57b1c10d79a8a959b8a2618bdb9ab8437ad`. Estão ativos:

- Cache automático em `/home/hjotha/llama-slot-cache/auto-qwen-p102400/`, bloco 98304 e snapshot comprovado de 102399 tokens.
- `LLAMA_ADAPTIVE_DIRECT_PURE_DISK_RESTORE=1`.
- `GGML_CUDA_GRAPH_MIN_STABLE_CALLS=256`, janela KVarN 4096 e wavefront256.
- Xbox `GianlucaMazza.xllama_1.6.0.411_x64__pj67f1fcj4n14`; INT8 desligado, rowlane sem opt-out e GQA2 desligado. Quatro entradas no store de snapshots nativos, confirmadas por leitura do Device Portal.

No request isolado de 158720 + 4096 com restore direto:

| Parcela contabilizada | Tempo | Fração do wall |
|---|---:|---:|
| Prefill de 56321 tokens após cache de 102399 | 406,69 s | 58,9% |
| Decode de 4096 tokens | 251,40 s | 36,4% |
| Restante aritmético | 32,76 s | 4,7% |
| Total | 690,85 s | 100% |

O restante não é uma atribuição causal exclusiva a rede, restore ou inicialização. A análise deve decompor os spans antes de atribuí-lo. O mínimo amostrado de VRAM foi 48 MiB, sem disparo do guard, com 4186 reutilizações de graph.

Sensibilidade, não previsão: reduzir em 10% o tempo de prefill desse caso economizaria aproximadamente 40,7 s; reduzir em 10% o decode economizaria 25,1 s, se o resto permanecesse igual. Isso prioriza trabalho de kernel/pipeline para requests grandes, em vez de perseguir apenas mais um segundo de restore.

Já obtivemos cerca de 7–9 s por request removendo o estágio local intermediário de restore. Não contabilizar esse ganho outra vez.

Fonte: [relatório da campanha](/home/hjotha/beellama-benchmark-results/xxxl-cache-110k-20261029/resultado-otimizacao-20261030.md), JSONs `summary.json`, `vram.csv` e logs no mesmo diretório.

## 3. O que pode ser reaproveitado

| Implementação/experimento | Estado da evidência | Reaproveitamento proposto |
|---|---|---|
| Bulk, FP8, prefetch de duas mensagens, rowlane | Já implementados; retirar bulk/prefetch piorou fortemente os testes | Manter como controle; reutilizar contadores, controle FIFO e testes de cancelamento |
| Restore direto de prefixo puro | Já integrado e ativo; saída idêntica nos pares medidos | Base da campanha, não nova hipótese |
| Snapshot nativo FULL do Xbox | No main e validado em build isolada; não no release integral de produção | Prioridade alta para prefixos repetidos, após fechar quota, pareamento e compatibilidade de release |
| GQA2 | Clone separado, fora do main; a rodada posterior r3 confirmou 53248 chamadas no Xbox, prefill +1,98%, mas falha determinística de qualidade (top-1 muda no passo 3, KL 1,11) | Reaproveitar o repro, captura RAW, shader e política para localizar o defeito; não ativar nem repetir a campanha de performance antes de corrigi-lo |
| MQ4/balanced | Mais lento que fine nos requests atuais; também altera política de split | Reaproveitar infraestrutura de tiles e reducer para variar uma dimensão por vez; não reativar balanced inteiro |
| Quatro tiles N256 | Qualidade histórica pareada passou; prefill +0,103%, sem ganho estabelecido | Reutilizar planner, lifetimes e testes; quatro tiles não significam quatro requests GPU simultâneos |
| N512/local split N256 | Implementado opt-in; N512 pressionou graphs e a tentativa de split local ficou 5,9% mais lenta, com +29 MiB de buffer CUDA | Reutilizar contratos, fallback e captura; só revisitar com desenho de lifetime/workspace diferente |
| Local input reuse | Prova histórica de cópias removidas; modo geral sem ganho total atual. O clone N1/decode-only também foi testado: bit-exato, ganho médio de decode de apenas +0,057% em 129024, sem benefício consistente | Reaproveitar comparação e invalidação para um experimento prefill-only; não apresentar decode-only como ideia nova |
| INT8 prefix | Ganho de prefill, mas gate recente falhou | Reutilizar harness teacher-forced, diagnóstico de lanes e packing; corrigir erro numérico antes de novos benchmarks de promoção |
| Técnicas Q8 de representação exata da query/reducer | No código; específicas de Q8 | Estudar a técnica mantendo KV KVarN4. Não copiar seleção/layout Q8 nem presumir que a query após WHT seja representável exatamente |
| Scratch/OOM da worktree anterior | Alterações e testes não integrados | Reaproveitar preflight de workspace e testes de pressão como proteção/admissão; não contar como aceleração comprovada |
| Worktrees de native save/preflight/erase | Contêm estados parciais de trabalho; boa parte já incorporada ou superada no main | Extrair somente deltas ainda necessários após comparar staged, unstaged e main; nunca mergear em bloco |
| Lookahead N256 do branch `feat/n256-graph-tensor-contract` | Código restringe a 32 camadas e UB256; modelo atual tem 64 camadas e XXXL usa UB512 | Reaproveitar contratos/testes CPU de dependências e falhas; exige port real, não ativação por variável |
| Rescale-skip | Revertido após resultado histórico exato, mas 1,67% mais lento | Reutilizar teste de exatidão e modelo de custo; não restaurar variante rejeitada sem hipótese nova |

GQA2 está em `/home/hjotha/beellama-agents/gqa2-prefill`, branch `perf/rkva-gqa2-prefill-20261027`, HEAD `582cc8d8e`. Há commits de integração e correções `e2368e95d`, `9c1729ef9`, `582cc8d8e`, além do protótipo `dbbb9f205`. A presença desse clone não deve ser inferida apenas da lista de branches do checkout principal.

O GQA2 divide as seis cabeças GQA em grupos de duas. Reduz estado por grupo, mas triplica grupos e releituras/dequantização de K/V. DXIL menor e LDS menor são apenas propriedades do artefato, não prova de ocupação ou TPS melhores.

**Correção de cronologia:** a documentação dentro do clone GQA2 ainda diz “offline only”, mas os artefatos posteriores de `/home/hjotha/.codex/artifacts/rkva-optimization-20261027-r3/` registram a execução e a rejeição. Esses resultados prevalecem. No mesmo relatório r3, o snapshot nativo ainda estava bloqueado; esse bloqueio é anterior aos consertos incorporados no main e aos roundtrips positivos de 29–30/10. Não transplantar o status antigo para o recurso atual.

Inventário dos branches visíveis ainda não integrados no checkout principal:

- `feat/rkva-wavefront-batching-20261021`: experimento antigo de mapped/wavefront/cohort; várias capacidades foram implementadas depois por outra linha. Comparar conteúdo, não reintroduzir o branch inteiro. Cohort4 continua sem qualificação para ativação automática.
- `feat/n256-graph-tensor-contract`: guardas de 32 camadas/UB256, incompatíveis com o caminho atual.
- `experiment/rkva-q8-decode-split-20261028`: apesar do nome, o delta não integrado `7f15edbb1` é principalmente uma correção de truncamento/orçamento de contexto no servidor. Não é um novo kernel Q8 nem um ganho de TPS demonstrado; só recuperar se um teste de fronteira reproduzir problema ainda presente no main.
- Os clones privados GQA2 e `rkva-decode-reuse` precisam entrar no inventário: não aparecem necessariamente no `git branch` do checkout principal.

## 4. Plano priorizado

### P0 — baseline reproduzível e atribuição atual do caminho crítico

**Antes de alterar execução:** fixar modelo, SHA de cada biblioteca carregada, pacote Xbox, flags, configuração e hashes dos prompts/snapshots. Separar o release de produção, o main e builds isoladas que contêm patches experimentais. Não atribuir ganhos a uma variável quando a build também mudou.

Criar quatro cenários independentes:

1. Contexto já no XXXL, prefixo vivo: mede trabalho de inferência sem transição.
2. Processo novo com snapshot local 102399: inclui seleção, restore e sufixo remoto.
3. Snapshot nativo pareado de prefixo maior: inclui seleção + restore host/Xbox + continuação.
4. Sem cache compatível: medição eventual do custo total, sem misturar com o prefill incremental.

Instrumentar janelas de overflow de aproximadamente 1k, 8k, 28k e 56k. Medir na 4070 FA local, matvec, GDN/conv, cópias e custo de graphs; no Xbox, pass1, reducer, decode rowlane, fila e fence; no transporte, espera exposta e bytes. Correlacionar por processo/sessão/ciclo/camada e separar clocks CPU/GPU. Nunca somar containers, filhos e trabalho sobreposto.

Reutilizar `rkva-latency-audit.py`, `analyze-rkva-decode-trace.py`, `bench-rkva-range.py`, raw-capture e CUPTI. Perf final com trace desligado. O material antigo teve correções importantes de clocks e atribuição; não reaplicar números de 26/10 como perfil atual.

**Entrega:** decomposição atual por tamanho/fase, latência por request e detecção de gargalo. A região de cerca de 25–30 s entre troca de perfil e disponibilidade do backend deve ser decomposta em inicialização host, alocações, configuração de sessão Xbox e PSOs. Ainda não há prova de que todo esse intervalo seja do Xbox.

### P1A — prefill novo: otimizar pass1 do Xbox preservando KVarN4

É a frente com maior potencial para prompts que não compartilham um prefixo remoto já salvo. Os perfis históricos do kernel MQ2 colocavam cerca de 97–98% do tempo de GPU do segmento no pass1; isso precisa ser reconfirmado na build atual e em 56k linhas remotas.

Sequência:

1. Variar split do MQ2/fine conforme linhas remotas sem trocar simultaneamente para MQ4. O código atual limita a política fine a 64 splits; alterar esse limite exige auditar buffers, reducer, limites do shader e protocolo. Começar por valores já suportados antes de ampliar limites. A mudança de split altera a ordem de somas de ponto flutuante: continuar usando KVarN4 não garante equivalência bit a bit.
2. Examinar cargas/dequantização, endereçamento e redução do MQ2 original, uma transformação por vez. Confirmar registradores/spills no hardware antes de atribuir o gargalo a ocupação. Preservar ordem aritmética sempre que o candidato se declarar exato.
3. Comparar O/LSE, logits completos e estado recorrente antes de medir TPS; depois executar ABBA sem trace nas três faixas. Usar os testes já existentes para seams de 128 tokens, queries ímpares e tail128.
4. Em frente secundária e limitada, recuperar o repro GQA2 `gqa2-raw-129024-r2` e localizar a primeira divergência por camada/posição, antes de ela aparecer no passo 3 do decode. Investigar mapeamento de cabeças, layout de partials, reducer, indexação de tail e alterações de ordem numérica como hipóteses, não causas comprovadas.
5. Só depois de corrigir a causa e passar OFF/OFF, ON/ON e OFF/ON, rebasear o conjunto GQA2 em worktree nova e repetir hardware. Manter arrays padrão byte a byte, INT8 OFF, contadores `mq2_gqa2` e grupos reais. Cobrir nq1/2..9/10, descritores, PSO/wave indisponível e conflito de flags.

**Critério de avanço:** redução repetida de tempo do request, além do kernel, sem erro de qualidade e sem sacrificar a margem CUDA. **Limitar o investimento em GQA2:** o ganho histórico foi pequeno (~2%) e a falha grande; abandonar a variante se a correção retirar o benefício ou não houver uma causa isolável. O que se reaproveita primeiro é seu repro e a infraestrutura, não seu veredito.

### P1B — prefixos repetidos: completar o snapshot nativo e seu ciclo de vida

Já existe prova isolada de save/restore de 110000 tokens e continuação igual. Na comparação alinhada, seleção + restore + request levou 63,82 s contra 80,51 s; save inicial de 14,11 s não faz parte desses 63,82 s e precisa ser amortizado. Não aplicar esse ganho a prompts novos ou a prefixes diferentes.

Sequência:

1. Primeiro verificar a cobertura do cache local para um **prompt novo que já entra diretamente no XXXL**: `slot-save-auto` não implica que o prefixo em P será gravado antes de existir overflow. Instrumentar esse caso e, se necessário, acrescentar checkpoint/publicação local em P antes do primeiro append remoto. Reutilizar o escritor de snapshot puro, salvar o estado recorrente correto e medir o custo adicional de I/O/RAM. Não prometer cache de um prefixo que nunca foi publicado.
2. Para o passo FULL nativo, produzir release consolidado a partir do main, com as capacidades usadas em produção (incluindo RPC/Vulkan), hashes e baseline próprio. Não instalar diretamente uma build de teste que omita backends ou contenha patches CUDA não integrados.
3. Qualificar primeiro a API explícita `native_stream` em 110k, 130k e perto do máximo, no mesmo P. Comparar estado e logits com um contexto vivo construído com os mesmos batches. O prefill único e o particionado já divergiram sem snapshot; não usar essa diferença como falso teste de corrupção.
4. Resolver quota e ciclo de vida pareado antes de automatizar: inventário host/Xbox, referências/leases, publicação em duas fases, falha de save sem apagar o par anterior, exclusão somente de pares sem usuários e recuperação após restart. O store atual tem teto de quatro entradas/8 GiB e limite por snapshot de 4 GiB; não basta trocar constantes.
5. Integrar seleção automática por prefixo exato, modelo/layout e geração, com admissão de RAM e fallback local. O código atual deliberadamente não faz auto-save de overflow nativo.
6. Medir hit-rate e custo de save + restore. Usar a decisão `k × (T_recomputar − T_reutilizar) > T_save` por ponto; não salvar cada comprimento por padrão. Priorizar pontos realmente recorrentes nas tarefas de código/agente.

**Entrega:** cache nativo opt-in com limites, rollback e métricas de economia líquida. **Parar** se a restauração/cópia superar a recomputação na faixa ou se o pareamento não sobreviver às falhas simuladas.

### P2 — decode e memória da 4070: acelerar o caminho realmente exposto

Os perfis antigos mostravam trabalho local CUDA relevante fora da FA e espera remota parcialmente sobreposta. Não presumir que todo o decode está limitado pelo Xbox.

1. Reutilizar a medição por subgrafo para localizar kernels locais dominantes (matvec/GDN/FA); otimizar/fundir apenas spans que estejam no caminho crítico atual.
2. Separar a política de captura por contexto/fase. Hoje o valor 256 é global e custou cerca de 3,4% no decode do XXL medido. Tornar o controle específico do XXXL e das formas instáveis de prefill, preservando o aquecimento padrão dos tiers locais.
3. Acrescentar orçamento/admissão de memória para graphs: contabilizar instâncias/capturas e preferir liberar formas antigas não referenciadas antes de desabilitar tudo. O código já tem sweep de graphs inativos, mas o caminho de pressão destrói todas as instâncias. Qualquer mudança precisa respeitar a vida dos buffers capturados, sincronização e descarte seguro.
4. Reaproveitar os testes de pressão e planejadores da worktree OOM. Workspace stream-K e scratch remoto devem caber junto com graphs; uma recuperação que evita OOM não é automaticamente uma aceleração.
5. Para input reuse, testar versão **somente no prefill** e/ou eliminar comparações de bytes comprovadamente desnecessárias por versão/epoch. O modo geral já teve perda de decode, e a variante decode-only `4b13ace53` foi bit-exata mas rejeitada por ganho inconsistente em testes reais. Remover a verificação de bytes sem provar todas as invalidações não é aceitável.
6. Auditar a leitura repetida de IDs de cells em `rkva_run_range`: só substituir varredura completa por validação incremental se append, rewind, erase, RESET, restore, cancelamento e troca de sessão invalidarem a mesma geração. Essa frente é subordinada à medição do custo exposto.

**Entrega:** menor tempo local/exposto, zero OOM e política que não desacelera os tiers locais para proteger o XXXL.

### P2 — transição e inicialização: eliminar trabalho duplicado sem dois KV residentes

O restore direto já removeu uma inicialização local e uma leitura do snapshot. O próximo passo é identificar o custo restante, não repetir essa alteração.

Depois da atribuição P0, testar separadamente:

- Cache de metadados e PSOs compatíveis, mantendo a sessão lógica e o conteúdo KV separados.
- Reaproveitamento de recursos vazios do backend quando geometria, modelo e geração permitirem. RESET/HELLO/cancelamento precisam continuar invalidando todo estado publicado.
- Inicialização e readback/restore sobrepostos somente onde não compartilham buffers e não elevam o pico de RAM/VRAM.

Não manter simultaneamente contexts completos do XXL e XXXL na 4070. Não declarar ganho de request por mover uma etapa para antes do cronômetro. Medir tanto o primeiro request após troca quanto uma sequência real de alternância entre tiers.

Esta frente sobe de prioridade se a carga real alterna muito entre tiers ou se concentra perto de 102k, onde o custo fixo pesa mais. Para requests já quentes perto de 160k, priorizar pass1/decode.

### P3 — pipeline e transporte: aprofundar apenas se houver espera exposta

Reutilizar o pipeline de duas mensagens já qualificado. Quatro tiles N256 não foram mais rápidos porque o limite de mensagens em voo permaneceu dois; isso não prova nem que janela maior ajudaria nem que seria segura.

Se P0 confirmar espera exposta suficiente, avaliar uma variável por vez:

- Janela limitada de frames preparados de 2 para 3/4, independente do tamanho de parent/UB. Medir posse de buffers, memória, backpressure e ganho real.
- Agrupamento de segmentos num único command list/dispatch com a mesma semântica causal. Cada query deve enxergar apenas seu prefixo correto. Não aumentar `PREFILL_QUERY_CHUNK` acima dos valores aceitos nem ativar cohort4 por variável sem suporte e capacidade negociados.
- Reutilizar planner de quatro tiles, testes FIFO, mensagens parciais, cancelamento/EOF, generation e ACK em ordem. Após falha, nenhum trabalho futuro pode ser publicado.

Não priorizar vectored-send ou coalescing de NIC: o ABBA antigo de resposta vetorizada não mostrou ganho claro; reduzir rx-usecs de 15000 para 1000 teve resultados agregados praticamente nulos. Só revisitar rede com evidência nova de espera/jitter, sem atribuir causalidade ao fato de a interface ser USB.

### P4 — INT8 ou técnicas Q8: pesquisa condicionada à qualidade

INT8 não está aprovado: a amostra recente reduziu o prefill, mas divergiu em geração e o smoke teacher-forced falhou (PPL ratio 1,08267 e KL médio 0,0026475). Não relaxar o gate nem promover com base em uma continuação que coincidiu.

Reaproveitar:

- Gate teacher-forced de vocabulário completo, captura de O/LSE/estado, comparação OFF/ON e diagnóstico de redução por lane.
- Técnicas de decomposição exata da query e fallback F32 do caminho Q8, **sem mudar o KV para Q8**. A query rotacionada por WHT e os fatores afins do KVarN4 tornam a transferência não trivial; provar elegibilidade e exatidão por bloco antes de medir ganho.
- Alternativamente, medir correção residual/mista restrita a blocos qualificados. O custo da correção pode eliminar o ganho de INT8.

MTP não é prioridade com P=102400: o N=1 medido falhou no gate de memória. Offload extra ou draft externo seria outra arquitetura e outro estudo, não ajuste barato de p_min.

## 5. Protocolo de avaliação e decisão

- Prompts: código, agente com ferramentas e pelo menos retrieval/documento; mesmos tokens, sementes e batches dentro de cada par. Testar temperatura zero e os parâmetros reais de produção.
- Pontos: fronteira 102400/102401; 110k, 130k, 150k e 158720. Saídas 256 e 1024 na triagem; finalista obrigatório em 158720 + 4096.
- Separar cache frio, local, nativo e prefixo vivo. Verificar `cache_n`, `prompt_n`, bytes restaurados, hash e rota efetiva. Reutilizar o snapshot local em todos os braços compatíveis; nada de refazer 102k a cada experimento.
- Primeiro, triagem curta com uma alteração; depois pelo menos três blocos ABBA para candidatos promissores, alternando ordem e monitorando clocks, temperatura, RAM/swap e fila. Não tratar uma diferença de 1% em uma execução como descoberta.
- Critério sugerido de promoção: economia mediana de pelo menos 3% ou 5 s, maior que a dispersão pareada, sem regressão material nos demais cenários. É um gate proposto para a campanha, não uma propriedade do código. Para estimar p95, ampliar a amostra; não calcular confiança de cauda a partir de quatro execuções.
- Exatidão: para mudanças declaradas exatas, comparar O/LSE onde aplicável, estado recorrente, logits/probabilidades completas e continuações. Tokens greedy iguais são evidência complementar. Mesma segmentação de prefill é obrigatória.
- Aproximações: manter ao menos os gates existentes (PPL ratio <=1,01, KL médio <=0,001, KL p99 <=0,01, delta NLL por caso <=0,02), tarefas semânticas e controle de determinismo. Um smoke falho bloqueia promoção; um smoke aprovado não qualifica toda a faixa.
- VRAM: amostrar e usar guard/contadores de alocação para capturar transientes; meta >18 MiB também nos picos, sem desativação por pressão. Não chamar o mínimo amostrado a cada segundo de mínimo absoluto.
- Interromper o braço em OOM, corrupção, timeout persistente, restauração incorreta, quota/owner inconsistente ou regressão clara. Restaurar serviço e flags; não continuar qualificando um processo contaminado.
- Probes RKVA que abrem nova conexão podem resetar engine/sessão: executá-los somente com o servidor de inferência parado e o console reservado para o teste. Leitura do Device Portal não substitui profiling de inferência.

## 6. Ordem de execução recomendada

1. P0: baseline/proveniência e perfil atual de prefill, decode e inicialização.
2. P1A: testar split MQ2 sem trocar o tile e otimizações exatas do pass1 que o perfil justificar. Nenhum ganho de TPS é presumido.
3. P1B: consolidar e qualificar snapshots nativos para prefixos repetidos, com quota pareada; essa frente pode vir primeiro se o hit-rate real for alto.
4. P2: política de graphs por fase/contexto, depois kernels locais e validação incremental que tenham custo exposto medido.
5. P3: profundidade do pipeline/agrupamento apenas se o perfil justificar; não repetir os ajustes de rede já neutros.
6. GQA2 e P4/INT8: primeiro localizar/corrigir as divergências com os repros existentes; manter desligados até passar qualidade e tempo total.

O primeiro lote deve entregar **um perfil atual e uma comparação controlada da política de split do MQ2**, não repetir a ativação de GQA2 que já falhou. Para carga com alto reuso de prefixo, entregar em paralelo lógico a admissão/gestão de pares de snapshots nativos. Os checkpoints e ferramentas existentes reduzem o custo da campanha.

## 7. Evidências e arquivos de partida

- [Campanha recente](/home/hjotha/beellama-benchmark-results/xxxl-cache-110k-20261029/resultado-otimizacao-20261030.md) e subdiretórios no mesmo local.
- [GQA2 fora do main](/home/hjotha/beellama-agents/gqa2-prefill/docs/rkva-prefill-gqa2.md), [integração offline](/home/hjotha/beellama-agents/gqa2-prefill/docs/agent-reports/rkva-mq-gqa2-integration.md) e [notas de revisão](/home/hjotha/.codex/artifacts/rkva-optimization-20261027-r2/gqa2-review-notes.md).
- Evidência posterior que prevalece sobre esses relatórios offline: [qualificação GQA2](/home/hjotha/.codex/artifacts/rkva-optimization-20261027-r3/gqa2-qualification-20261027.json), [veredito de decode-only reuse](/home/hjotha/.codex/artifacts/rkva-optimization-20261027-r3/decode-reuse-evaluation-20261027.json), [consolidação r3](/home/hjotha/.codex/artifacts/rkva-optimization-20261027-r3/CONTINUATION-REPORT.md) e [split local N512](/home/hjotha/.codex/artifacts/rkva-optimization-20261027-r3/wf512-local-split-result.json).
- [Wavefront, prefetch e input reuse](rkva-2xn256-status.md).
- [Pass1 e custo de instruções](agent-reports/a2-pass1-kernel.md), [revisão INT8](agent-reports/astra-int8-review.md), [gate de qualidade](agent-reports/rkva-quality-gate.md).
- [Caminho de decode](rkva-decode-critical-path.md), [auditoria de latência](agent-reports/rkva-latency-audit.md). Números históricos não substituem novo perfil.
- [Snapshot nativo](xbox-native-overflow-snapshots.md), `tools/xbox/uwp/rkva/rkva-disk-snapshot.inc`, `tools/server/server-context.cpp`, `src/llama-context.cpp`.
- [Coalescing sem ganho](/home/hjotha/.codex/artifacts/rkva-priorities-20261026/nic-coalesce-r1/analysis.json) e [ABBA de transporte](/home/hjotha/.codex/artifacts/rkva-priorities-20261026/transport-abba-r1/status.json).
- Worktree experimental de OOM: `/home/hjotha/worktrees/beellama-xxxl-oom-20261029`; preservar e revisar os deltas, não aplicar em bloco.

Plano salvo em worktree dedicada, sem commit/merge/deploy nesta etapa de análise.
