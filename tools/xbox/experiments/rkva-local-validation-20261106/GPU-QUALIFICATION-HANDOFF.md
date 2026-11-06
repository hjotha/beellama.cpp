# Handoff: qualificação MQ2 cap96 e próximas frentes RTX 4070/Xbox

Preparado em 2026-11-06. Este documento é um plano, não resultado de GPU.
Destino: execução local no ambiente RTX 4070/Xbox autorizado, após conferir o commit publicado.
Repositório: `hjotha/beellama.cpp`; branch autorizado: `astra/xbox-rkva-local-validation-20261106`.
Último head informado antes do novo pacote: `e5b969ed9935abde9be9593e429ab47b48003cf7`;
main de referência informado: `7310f5d7e`. Registrar o SHA efetivamente recebido, não presumir que esses sejam o novo head.

## 1. Escopo e autorização

Escopo técnico: preparar e publicar o pacote experimental no branch indicado, e continuar a qualificação no ambiente RTX 4070/Xbox, incluindo possíveis promoções condicionadas aos gates e à autorização operacional existente.

- Continuar a tarefa existente e os testes autorizados no ambiente RTX 4070/Xbox. Possível promoção depende dos gates abaixo e do escopo operacional já aprovado; não significa licença para interromper trabalho concorrente ou tomar console ocupado.
- Publicação de código permanece no branch acima. Não inferir autorização para merge em main, force-push, exclusão de histórico, ampliação de acesso ou mudanças de segurança.
- Ler `AGENTS.md`, `tools/xbox/AGENTS.md`, `tools/xbox/README.md` e instruções locais relevantes. Não criar git worktree, nested checkout ou patch de core durante build. Preservar alterações de terceiros; clone separado se necessário.
- Verificar janela/lease de teste existente antes de tocar serviços/console. Usar somente endpoints e identidades locais já autorizados. Se não houver janela, preparar builds e registrar bloqueio.
- Build/signing pode reutilizar o certificado existente no fluxo aprovado. Não criar certificados/credenciais, não alterar identidade/publisher, não desinstalar app; preservar LocalState, modelos e snapshots. Não iniciar implantação apenas para testar build.
- Não afrouxar guards de memória, shape, codec, tail, fallback, wave ou PSO. Não desativar checks para obter um número.

## 2. Perfil congelado e interpretação da evidência

Baseline operacional informado: pacote Xbox 1005, comportamento idêntico ao 939; RTX 4070 + Xbox, KVarN4, tail128, E4M3FN, produtor CUDA, prefetch2, wavefront 2×256, P=107008, contexto=166400, batch=1024, ubatch=512, sem MTP. Confirmar hashes do executável/pacote/perfil e a equivalência relevante no ambiente, em vez de presumir identidade binária entre números de pacote.

Controles históricos 130560/158720: prefill 205,94/143,74 tokens/s e decode 17,32/15,94 tokens/s. Há uma execução por perna, não distribuição, P50/P95 ou ganho repetível.

- Pass1 `nq8 fine`: 21760 rows, 2,230 ms; 49920 rows, 4,816 ms. Reducer: 0,075/0,076 ms.
- Oito RPCs de cauda `nq128`, mesmo payload: GPU 39,55→81,56 ms e RPC 60,58→102,73 ms. GPU é componente/observação dentro do fluxo RPC, não parcela independente para somar ao RTT.
- Prefetch ON/OFF/ON: 205,94/150,75/205,31 tokens/s, mesmos bytes/tokens. Manter prefetch2 no A/B cap64/cap96; não repetir o estudo misturando causas.
- Nsight decode: matvec 59,2% do tempo de kernels, KVarN 33,4%; maior kernel individual `fattn_kvarn_decode_mma`. Captura usou contexto 132096 por VRAM, portanto não qualifica produção 166400 nem equivale a frações do tempo total do request.
- Menor memória livre normal observada: 43 MiB. Nsight completo abortou com 15 MiB; guard foi mantido. Esses valores são observações, não novos limites autorizados.

As instruções antigas do pacote que mandam começar por Q_rot ou testar MTP não são a prioridade desta campanha: perfil atual é sem MTP e o primeiro candidato é somente split MQ2. Q_rot sem ganho e com diferença, WHT sem ganho, GQA2 falhou qualidade. Não reativar cegamente.

## 3. Candidato exato e riscos

Comparar somente limite fino MQ2 cap64 vs cap96, opt-in com default OFF; MQ4, coarse, tiles, prefetch e demais gates devem ficar iguais. Flag definida no candidato: `RKVA_EXPERIMENTAL_MQ2_FINE_SPLIT96`, default `0`, somente fine/q_tile2. Confirmar no patch recebido o ponto de definição e propagação ao build; não presumir que seja variável de ambiente ou CLI. A publicação contém apenas pacote experimental: presença do patch no Git não altera o aplicativo nem confirma opt-in efetivo.

| Rows | Baseline | Candidato esperado | Interpretação |
|---|---|---|---|
| ≤16384 | Mesma política | Mesma política | Controle sem alteração |
| 16385 | 63×264 | 65×256 | Primeira fronteira relevante |
| 21760 | 64×344 | 85×256 | +32,8125% de grupos, não 96 splits |
| 24576 | 64×384 | 96×256 | Fronteira de alinhamento |
| 24577 | Verificar política | 94×264 | Split count não monotônico |
| 49920 | 64×784 | 96×520 | +50% de grupos |

Span nominal pode incluir padding/clipping; verificar cobertura e ausência de trailing split vazio com a política efetiva. Cap96 não significa sempre 96 splits nem split span96.

A análise CPU anterior indica buffer de partials já dimensionado para 256 splits, mas isso não prova compatibilidade de toda a cadeia. Verificar alocação, constantes de dispatch/reducer, limites PSO, número de partials realmente escritos/lidos e fallback no código recebido. Mais partials aumentam tráfego e custo de redução mesmo sem crescer a alocação. Ordem de redução pode alterar OUT/LSE. Não usar PASS CPU como passe GPU.

Histórico a consultar antes do tuning: `docs/rkva-2xn256-status.md`, `docs/rkva-final-optimizations.md`, histórico Git da política e resultados locais da campanha atual. O status 2xn256 registra redução de barreiras de pacote295 com replay igual, mas MQ2 tardio 2785,531→2786,489 µs e sem ganho TPS estabelecido. Menos barreiras/grupos mais curtos não bastam como evidência de velocidade.

O pacote [MQ2 split96](mq2-split96/README.md) e seu [patch](mq2-split96/mq2-fine-split96.patch) incluem testes CPU que cobrem C++ default/0/1, negativos e OUT/LSE sintético pós-decode, com reducer/WHT e tolerâncias existentes versus softmax FP64. Não cobrem bytes KVarN, engine, HLSL ou Xbox. A reprodução final confirmou 20 de 36 casos CPU não bit-exact entre cap64 e cap96, embora ambos passem as tolerâncias. Isso não qualifica codec, QK, engine ou GPU e não prevê aprovação do replay byte-exact. Registrar resultados da reprodução do pacote recebido separadamente dos modelos históricos.

## 4. Matriz curta, em ordem de custo

| Etapa | Cobertura e prova necessária | Gate |
|---|---|---|
| Proveniência | SHA novo, diff, flag OFF por padrão, lista de arquivos, ferramentas, baseline rollback | Não misturar main/artefato antigo/candidato |
| CPU | Suíte recebida e regressões de política; flag ausente/OFF/ON, MQ4/coarse/fallback idênticos | PASS local é somente CPU |
| Build | Windows MSVC/UWP e host; DXC se shader/payload recompilado, hashes e defines | Não confundir fixture DXIL com app executado |
| GPU smoke | Confirmar seleção real cap64/cap96 e geometria, mesmos requests; finitude e oráculo | Falha encerra desempenho/promoção |
| GPU bordas | nq2..9, query tile incompleto, rows 16383/16384/16385 e 24575/24576/24577, 21760/49920; não múltiplos8/128 | Sem perda/duplicação de rows ou query fantasma |
| Contrato | F32/F16/E5M2/E4M3FN, tails0/128; origens0 e não zero128-alinhadas; causal vazio/curto/completo; rewind/reappend/ring wrap; negativos de shape/codec | Caminhos não suportados preservam fallback; gates do probe intactos |
| Tempo GPU | Pass1/reducer/readback/atenção total, iguais payloads e PSO confirmado | Ganho não pode desaparecer na redução/RPC |
| Request real | Perfil congelado em contexto166400, controles130560/158720 e carga/prompts fixos | Correção/qualidade e ganho repetível total |
| Rollback | Voltar cap64/baseline1005 e repetir smoke/health/request de controle | Baseline recuperado e artefatos preservados |

Não chamar a matriz completa se apenas um subconjunto foi executado. Priorizar E4M3FN/tail128 atual após smoke, mas declarar cobertura faltante. Uma redução de contexto ou batch por falta de memória cria outra configuração; não qualifica produção.

## 5. Comandos verificados e limites

Os comandos abaixo existem no pacote inspecionado; verificar novamente no SHA recebido. São inspeção/self-test/plano, sem contato com console:

```sh
git status --short
git branch --show-current
git rev-parse HEAD
git diff --check
git log -8 --oneline
python3 tools/xbox/tests/rkva_kvarn_serial_prefix_probe.py --self-test
python3 tools/xbox/tests/rkva_kvarn_bulk_replay.py
```

O replay bulk acima sem `--execute` imprime o plano. O replay existente usa F32 e modos MQ4/MQ2; não cobre sozinho E4M3FN nem prova que atingiu buckets onde cap96 muda a política.

Depois da janela autorizada, baseline/candidato selecionados e verificados, usar os probes existentes com host explícito e outputs novos. Exemplo condicional em shell POSIX:

```sh
# Definir RKVA_TEST_HOST somente para o listener autorizado.
# Definir OUT como novo diretório local de artefatos desta rodada.
mkdir -p "$OUT"
python3 tools/xbox/tests/rkva_kvarn_bulk_replay.py \
  --execute --host "$RKVA_TEST_HOST" --output "$OUT/bulk-A.json"
# Selecionar candidato pela interface real documentada no patch; confirmar nos logs.
python3 tools/xbox/tests/rkva_kvarn_bulk_replay.py \
  --execute --host "$RKVA_TEST_HOST" --reference "$OUT/bulk-A.json" \
  --output "$OUT/bulk-B.json"
```

Nunca interpretar esse exemplo como mecanismo de seleção do candidato. O comparador exige hashes idênticos de requests e respostas de atenção. Se OUT/LSE mudarem pela ordem de redução, isso é falha do gate de replay, mesmo que o erro numérico seja pequeno; registrar as duas avaliações separadamente, não mudar o comparador para fazê-lo passar. Gates numéricos herdados: OUT `abs(delta) <= 3e-4 + 3e-4*abs(ref)`, LSE `abs(delta) <= 3e-4 + 3e-5*abs(ref)`, finitude, além de qualquer gate atual mais estrito. Não substituir gates de qualidade do modelo por esses limites.

Para build, seguir o wrapper/configuração real de `tools/xbox/README.md`, SDK instalado, revisão de pacote válida maior que a instalada e certificado existente. O exemplo antigo de revisão287 não serve para baseline1005. Não copiar build CUDA sm86 para RTX4070; preservar a toolchain/arquitetura qualificada do host, verificar hardware e flags reais. Não há comando novo de flag ou promoção inventado aqui.

## 6. Medição e decisão

O `qualification/qualify_abba.py` anterior é orientado a decode + total; sua aprovação ou reprovação não é um gate genérico de prefill. Nesta campanha, declarar antes das medições o objetivo de ganho de prefill e os limites aceitáveis de regressão de decode e total, preservando todos os gates de correção. Registrar esses critérios no manifesto do experimento e avaliar cada métrica no seu escopo. Não mudar scripts ou gates existentes para forçar PASS, nem escolher limites depois de ver os resultados.

1. Registrar configuração completa e diferenças entre braços antes de executar. Mesmo modelo/hash, prompt/tokenização, seeds/sampling, número de tokens, cache/prefixo, resets, codecs, P/contexto, b/ub, producer, prefetch, clocks/temperatura e carga concorrente.
2. Aquecer separadamente. Usar seis blocos A/B/B/A completos como padrão, alinhado ao qualificador já publicado, sob a mesma carga e se os gates permitirem. Menos blocos constituem evidência parcial, não qualificação completa. Preservar todas as amostras, ordem e motivos de descarte; não descartar regressões por conveniência. Repetir mais se houver drift/ruído.
3. Reportar mediana, amplitude/dispersão e diferenças pareadas. Com poucas amostras, P95 é descritivo instável, não garantia de cauda. Separar prefill, decode, TTFT/request total e bytes/tokens; não somar spans sobrepostos de GPU/RPC.
4. Exigir ganho repetível no request real e ausência de regressão material em correção, qualidade, memória, estabilidade e fases relevantes. Resultado dentro do ruído ou ganho somente de microbenchmark = inconclusivo, sem promoção.
5. Promoção só após revisão dos artefatos e rollback qualificado, no destino operacional já autorizado, mantendo caminho OFF/baseline. Sem ampliar escopo ou mudar defaults globalmente apenas porque um shape melhorou. Se alvo ou janela da promoção não estiverem claros, registrar bloqueio em vez de escolher um novo alvo.

Rejeitar/abortar diante de OUT/LSE inválidos, replay/qualidade falhos, fallback silencioso, seleção errada, OOM/guard, regressão repetível, dados insuficientes ou configurações não equivalentes. Restaurar pelo procedimento de rollback existente; nunca apagar dados para liberar memória nem reduzir guard.

## 7. Outras frentes, sem misturar experimentos

- **Shader MQ2:** somente depois de caracterizar cap64/96. Hipóteses: reuso KV entre queries e redução de dequant redundante. Ainda precisam de implementação revisada e medidas de registradores, LDS, ocupação, spills, cargas, partials e reducer. Não alegar que o modelo CPU MQ2 rowlane já é shader. Um candidato por vez com o mesmo protocolo.
- **CUDA matvec/KVarN:** frente independente de decode. Começar por timestamps/telemetria existente e capturas curtas de baixo overhead; registrar overhead contra execução sem profiler. Separar famílias, shapes e peso no request. Não repetir Nsight completo quando viola margem; contexto132096 fica evidência diagnóstica separada. Não converter porcentagens de kernels em promessa de ganho total.
- **Q8 histórico:** `da14105a7` e paired `ebc793d6` já integrados ao main segundo a revisão recebida; verificar ancestralidade/diff antes de reaplicar. São KV Q8 Xbox, não quantização de pesos CUDA nem KVarN4 do perfil atual. Opt-in e INT8 aproximado exigem campanha própria. Não cherry-pick duplicado nem ativar no braço cap96.
- **WHT/Q_rot/GQA2:** resultados desfavoráveis acima permanecem; só reabrir com hipótese nova e evidência específica, nunca como tuning automático.

## 8. Relatório conclusivo obrigatório

Salvar junto aos artefatos um relatório com:

- Commit/base/diff, hashes de fonte, binários e DXIL, pacote instalado, versões MSVC/DXC/CUDA/driver e flags exatas; comandos completos sem segredos.
- Manifesto dos braços A/B com todas as diferenças e mecanismo real da flag; counters/logs provando seleção e splits reais por bucket.
- Matriz de etapas: PASS / FAIL / NÃO EXECUTADO, limites de cobertura e motivo; separar CPU, DXIL, pacote Windows, GPU Xbox, RTX4070 e serving.
- Requests/payloads, hashes, OUT/LSE máximos e pior caso, replay, qualidade/logits/KLD conforme protocolo existente com batch/ubatch iguais; divergência de tokens é investigada, nunca ignorada.
- Todas as amostras e ordem ABBA, métricas por fase e end-to-end, memória mínima, clocks/temperatura, contexto efetivo, exceções/guards e overhead do profiler.
- Decisão por frente: rejeitado / inconclusivo / qualificado experimental / promovido. “Promovido” somente com alvo, instante, revisão efetiva e validação posterior; ausência de execução não é aprovação.
- Estado final do serviço/console, rollback testado, baseline preservado e próximos passos concretos. Não declarar trabalho completo enquanto a qualificação hardware estiver pendente.

Fontes locais consultadas para preparar este plano: instruções root/Xbox e README, pacote `tools/xbox/experiments/rkva-local-validation-20261106`, probe `rkva_kvarn_bulk_replay.py`, `docs/rkva-2xn256-status.md`, `docs/rkva-final-optimizations.md`, análise isolada de geometria MQ2. Números da campanha atual foram fornecidos pelo coordenador e precisam ser ligados aos artefatos originais no ambiente de teste no relatório final.
