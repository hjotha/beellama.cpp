# Observador DFlash2 corrigido — 2026-09-29

Branch `feat/dflash2-shadow-observation`, base `6281c6b56` com o diff preservado em cada execução. Estes são testes de observação: nenhum token DFlash entra na resposta.

## Correções verificadas

- Tokens de referência vêm da aceitação definitiva do servidor, com o bônus real; batches enviados à verificação não são usados como tokens confirmados.
- Um resultado concluído aguarda commit e prazo da sua observação; workers rápidos não desaparecem da amostra.
- A decisão é registrada antes do próximo MTP, mesmo nos ciclos sem lançamento por throttle; conclusão é timestampada pelo próprio worker.
- Rejeições e replay cortam features provisórias e a KV auxiliar; só uma fronteira alinhada pode iniciar outra previsão.
- Request/epoch/job identificam cada resultado. Restore sem prefixo auxiliar completo suspende o auxiliar até novo prefill frio.
- Cancelamento do draft primário e fim de requisição classificam o trabalho pendente; contadores reconciliam lançamentos, observações e cancelamentos.
- Staging/injeção respeitam a capacidade real de ubatch; falhas do worker são distintas de proposta vazia por confiança.
- Timers de decode/seletor incluem os getters que já sincronizavam as GPUs; nenhum sincronismo extra foi inserido para medir as etapas.

## Configuração da nova bateria

Mesmo binário em todos os casos; alvo Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf em CUDA0/RTX 4070 eGPU; auxiliar Qwen3.8-27B-DFlash2-Q4_K_M.gguf em Vulkan0; MTP n=4/p_min=0,70; DFlash n=7; ctx 16384; b/ub 64; parallel 1; KV q4_0; temperature 0/seed 42; 192 tokens gerados. APU 20 W e solicitação SCLK 2700 MHz em todos os casos, inclusive MTP isolado. O indicador de clock de GPU ocupada pode ser nulo quando a Radeon não executa kernels.

CPU explicitamente fixada em powersave/EPP power e faixa 419175–3301000 kHz em todas as 16 políticas. Readback conferido antes e depois de cada requisição; APERF/MPERF e hwmon registrados. A política encontrada antes do teste foi restaurada integralmente no final, mesmo sendo diferente dessa configuração experimental.

Aquecimento de 64 tokens, 3 repetições curtas e 2 longas por modo. O longo tem 13850 tokens de entrada. Bateria sequencial por modo, com repetição final dos curtos do MTP isolado; não é uma ordem aleatória/intercalada de todos os casos.

| Modo | Repetição tok/s | Código tok/s | Longo tok/s | Diferença vs MTP |
|---|---:|---:|---:|---|
| mtp-n4 | 87.80 | 61.03 | 59.06 | +0.0%, +0.0%, +0.0% |
| shadow-sync | 85.05 | 58.89 | 56.99 | -3.1%, -3.5%, -3.5% |
| shadow-every1 | 60.94 | 43.18 | 43.47 | -30.6%, -29.3%, -26.4% |
| shadow-every4 | 82.38 | 56.80 | 55.37 | -6.2%, -6.9%, -6.2% |
| mtp-n4-repeat | 86.94 | 60.69 | — | -1.0%, -0.6% |

Todas as 38 respostas medidas coincidiram com o hash de referência por cenário. O baseline final confirma deriva inferior a 1% nos prompts curtos. O `smoke-v2/` é um canário intermediário; os resultados principais usam a versão final em `paired-v1/`. Scripts e dados completos estão em `paired-v1/`; cada modo tem comando e ambiente explícitos, respostas, log, políticas e telemetria.

## Contadores corrigidos

A tabela inclui os blocos do aquecimento. Observações são apenas jobs com conclusão, commit definitivo e decisão identificada. O fim da requisição cancela os jobs que já não teriam uma rodada seguinte. Tokens utilizáveis são propostas potenciais, limitadas à capacidade do MTP/verificador (até quatro aqui); não são tokens já aceitos nem ganho de throughput.

| Cadência | Lançados | Observados | Cancelados | Prontos no prazo | Prefixo coincide | Pronto + coincide + sufixo | Tokens candidatos |
|---|---:|---:|---:|---:|---:|---:|---:|
| shadow-every1 | 414 | 405 | 9 | 383 (94.57%) | 285 (70.37%) | 273 (67.41%) | 736 |
| shadow-every4 | 104 | 102 | 2 | 78 (76.47%) | 69 (67.65%) | 53 (51.96%) | 130 |

Todos os lançamentos fecharam como observados ou cancelados: pending=0, errors=0, skipped_invalid=0. O analisador valida os IDs, os tokens comparados, o bônus, o prazo e a contabilidade. `EVERY=4` é amostragem periódica, não aleatória; suas taxas não devem ser extrapoladas como amostra IID.

## Tempos corrigidos

Médias por bloco, incluindo aquecimento. São latências de parede das operações e seus retornos necessários; não constituem tempos de kernel puros. Injeção/cópia no thread principal incluem prefill e estão preservadas como totais em `observer-summary.json`, sem dividi-las pelo número de blocos para criar uma falsa latência por chamada.

| Cadência | Worker ms | Decode + retorno dos hidden rows ms | Seletor + retorno da lattice ms | Outros dentro do worker ms |
|---|---:|---:|---:|---:|
| shadow-every1 | 55.298 | 51.665 | 3.299 | 0.335 |
| shadow-every4 | 41.635 | 37.896 | 3.598 | 0.141 |

O residual antigo de ~35 ms não era prova de preparo/amostragem: incluía sincronização não atribuída. Agora o residual é inferior a 0,4 ms por bloco nesta bateria. A observação contínua ainda interfere bastante no primário; PPT e frequência média de CPUs se correlacionam com a interferência, mas esses dados não isolam a causa nem o clock específico do thread MTP.

## Validação e próximos passos

- Build incremental CUDA/Vulkan via plugins existentes e quatro CTest selecionados aprovados: observação, candidato de pipeline, DFlash upstream e colocação do seletor.
- `lifecycle-v1/`: batch 64/ubatch 32, prefill de 13850 tokens, reuse de cache com apenas 4 tokens reprocessados e o mesmo hash, cancelamento de stream após 4 eventos e próxima requisição sem contaminação.
- O auxiliar fica suspenso quando o alvo restaura um prefixo que o auxiliar não possui. A requisição fria seguinte reativa a observação.
- Produção, CPU e limites de GPU/APU restaurados e conferidos ao final.

A coincidência de prefixos deixou de ser a objeção principal: ela era medida incorretamente. Ainda não foi demonstrado ganho ativo. O próximo experimento deve comparar o custo/avanço de sufixos curtos com um novo MTP, e pode investigar lançar o auxiliar antes do MTP atual para ampliar sua janela (hoje ele ainda começa depois desse draft). No EVERY=4, 16 prefixos corretos chegaram tarde (atraso mediano 4,693 ms, máximo 16,388 ms); antecipar o lançamento é uma hipótese concreta a medir. A aceitação dos tokens restantes ainda não foi medida pelo verificador híbrido. O modo ativo continua condicionado à validação do estado e ao ganho líquido, e árvores de verificação permanecem uma etapa posterior.

## Reprodução

Executar `python3 runner.py paired-v2` para uma nova bateria em diretório novo; `python3 lifecycle.py lifecycle-v2` para o canário de ciclo de vida. A senha é solicitada via getpass e não é persistida. Esses runners param e restauram os serviços. `python3 analyze.py paired-v1` reanalisa os artefatos sem executar inferência.
