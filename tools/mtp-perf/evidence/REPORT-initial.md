# Resultado — 2026-11-16

Implementação experimental isolada, ainda NÃO calibrada em hardware.

## Baseline antes de editar
Observação inicial 2026-11-16T17:51:58+01:00: gokaya, Ryzen Z1 Extreme,
RTX 4070 UUID GPU-14e0b235-c9a7-3721-e8c8-bff50a6d79c3, driver 620.70.06.
12282 MiB total, 11691 usados, 235 livres (nvidia-smi reporta memória reservada
separadamente; não substituir free por total-used). GPU 0%, SM 210 MHz,
memória 405 MHz, 7.93 W / limite 200 W: estado ocioso, não clocks de benchmark.
Linux 7.3.3-1-cachyos-deckify. Perfil de energia custom.
Repo original 19f92e9dff3757683a09367785b71e1be8d7f6d0,
branch investigate/cuda-pv16-deepseek-20261116, oito arquivos já modificados.
main aponta para o mesmo commit; nenhum desses arquivos foi editado por este trabalho.

Produção router PID119138 release kvarn-prefill-dff4c2db5-20261113 porta8090;
filho PID119180 release rkva-parallel-182df85bc-20261112 porta46779,
manifest commit182df85bcac807e722503d312dc18b745b56860f.
Modelo Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf,
10442827776 bytes, quantização indicada pelo nome; hash completo não calculado
nesta janela. KV alvo/draft kvarn4, draft-mtp, FA on, GPU layers99, parallel1,
fit off, sem context shift. Flags completas e manifests estão em baseline.json.
N atual short6/medium4/long2/xlong+0; p-min .80/.70/.65/.50 (não uniforme).
B/UB short2048/256, medium1152/384, long2048/256; demais no baseline.
CUDA graphs têm flags de recuperação/headroom no processo, mas o estado efetivo
por shape não foi verificado em logs de startup ou replay; não afirmar graphs ativos.
Métricas read-only do filho: requests_processing0 / requests_deferred0.
Router /metrics retornou400; consulta direta ao filho funcionou.
Servidor ocioso permanece com VRAM alocada; isso NÃO autoriza desalocar produção.

## Entrega
Branch experiment/mtp-tps-model-20261116 em clone local separado
/home/hjotha/beellama-tps-model-20261116. Sem worktree, merge, push, deploy ou
instalação. Sem commit novo: AGENTS.md exige pedido explícito de commit.
Arquivos tools/mtp-perf/{bench.py,model.py,test_model.py,README.md} e alteração
opt-in de 7 linhas em tools/server/server-context.cpp.

Modelo: latência não-negativa com termos constante/N/contexto/interação,
N0 próprio; sobrevivência geométrica ajustável para tokens efetivos; função roofline
com taxas sustentadas como entradas. Não usa números T4 como treino.
Holdout por N/contexto, rejeição de desenho sem rank e regimes misturados,
CSV/JSON, mediana/faixa/MAD. Os intervalos são dispersão descritiva, não IC.
Harness recusa GPU com processos compute e porta ocupada, monitora memória,
encerra só seu processo, separa prefill/TTFT/TPS e exige prompt exato + geração
completa. Warmup separado, mínimo3 repetições, N0/1/2/4/6/8, B/UB e p-min fixos.

## Evidência real e lacunas
- 9 testes CPU aprovados (tests.txt), py_compile e git diff --check aprovados.
- guard-real.json: recusa real com PID119180 / free235 MiB, antes de inferência.
- Nenhum benchmark de inferência executado, nenhuma medição TPS nova.
- Nenhum build C++/CUDA executado; patch ainda não compilado em hardware.
- Nsight Systems2026.3.2 e Compute2026.3.1 disponíveis, nenhum perfil coletado.
- DRAM/L2/SM/kernel/launch/sync: sem contadores medidos.
- Erro preditivo real: indisponível. Testes sintéticos não validam arquitetura.
- Não há thresholds N/B/UB validados ou tiers recomendados para ativação.
- Busca adaptativa/pruning automático e bootstrap de IC ainda não implementados;
  protocolo manual documentado, seleção explícita --ns/--contexts disponível.
- Atenção/KV está dentro do tempo target; separação GPU exige profiler.
- Transições reais KV/tier e zero OOM não testados. Monitor amostrado pode perder
  pico breve; não constitui prova de segurança. Não mirar100 MiB livres.

## Próximo passo concreto
Janela autorizada em que a produção libere a GPU, sem o agente parar o serviço
por conta própria. Build isolado, qualificação curta N0/N2/N6 a4k, revisão de
VRAM/graphs, expansão gradual para8k/16k/24k/32k com >=3 repetições e corpus
fixo. Depois holdouts N1/N4/N8, perfis short/long, confirmação de cruzamentos
previstos com pontos intermediários. Só então B/UB e 64k/100k se memória permitir.
Mudança de GPU conserva protocolo/contagem/forma física, recalibra custos e
valida aceitação; custo mínimo de calibração é hipótese a testar, não conclusão.

Nota de coleta: /proc de produção teve erro de permissão; baseline.json preserva
o erro e captura args via ps separadamente. Ambiente CUDA não foi legível.
O manifest do router não estava disponível; seu commit não foi confirmado.
