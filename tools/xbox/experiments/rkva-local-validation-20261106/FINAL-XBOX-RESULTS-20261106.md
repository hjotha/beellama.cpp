# Resultado final externo: WHT e Q_rot, 2026-11-06

## Decisão para o próximo agente

**Não promover WHT nem Q_rot. Não repetir os mesmos candidatos como se ainda
aguardassem uma primeira qualificação.** Uma nova investigação precisa de uma
hipótese diferente, evidência de que ela resolve o bloqueio e autorização para
qualquer nova execução remota. Não afrouxe gates para aceitar os resultados.

- WHT passou os probes de corretude descritos, mas não mostrou ganho justificável.
- Q_rot passou tolerâncias seriais, mas falhou igualdade bulk, divergiu na
  geração e não mostrou ganho mensurável.
- Baseline foi restaurado segundo o relatório final; nada foi integrado ao main.
- LDS wave64 continua um candidato independente sem testes GPU. Estes resultados
  KVarN4 não qualificam LDS Q4/Q8/F16/SAW.

## Fonte e limites

Resumo público sanitizado de `NOTAS.md` e do transcript técnico, consultados em
2026-11-06 às **16:52:32 UTC**. A resposta final do agente executor é de
**16:44:46 UTC**. Os dados abaixo são relatos verificados nesses artefatos,
transcritos e arredondados. Não são novos testes no dot, logs brutos anexados,
medições independentes ou atestação do estado atual do console.

Campanha relatada: contexto 8k, P4096, 1024 tokens de saída, KVarN4, produtor
CUDA, E4M3FN e prefetch; mesmo prompt, seed, parâmetros e binário host entre
braços. Base host `7310f5d7e`; pacote de proposta `2616d383`. Identificadores
abreviados são preservados como relatados, não expandidos por inferência.
Não se presume que a cobertura de todos os formatos, contextos e qualidade
esteja completa. Não há novos hashes de artefatos inventados neste resumo.

## WHT ping-pong: correto nos probes, sem ganho justificável

Corretude relatada: 498 checagens seriais, tails 0/128; 696 registros bulk com
requests e bytes OUT/LSE idênticos ao baseline. Os 1024 tokens foram idênticos
nos quatro braços. Isso não prova cobertura geral de qualidade.

Valores transcritos de `NOTAS.md`: throughput em tokens/s, wall em segundos.

| Braço | Prefill total t/s | Trecho remoto t/s | Decode t/s | Wall s |
| --- | ---: | ---: | ---: | ---: |
| A1 | 465.13 | 318.97 | 21.38 | 65.48 |
| B1 | 464.90 | 317.44 | 21.38 | 65.47 |
| B2 | 465.73 | 318.71 | 21.38 | 65.44 |
| A2 | 462.30 | 316.68 | 21.39 | 65.56 |

Média remota relatada: baseline 317,83 t/s, candidato 318,08 t/s, +0,08%.
O relato considera o efeito dentro do ruído, sem ganho decode/total que
justifique manter o candidato. WHT não promovido. Um único bloco não permite
concluir estatisticamente ausência de ganho ou equivalência.

## Q_rot: gate exato falhou, sem ganho mensurável

Build Final MSVC/MSIX 1004, DXIL esperado no executável e reducer WHT baseline
foram confirmados no relatório externo. Não são builds executados nesta
publicação. As 498 checagens seriais passaram; bucket
`optimized-kvarn4_decode_rowlane`, nq1. Nas 19 verificações finas nq1, máximos
relatados versus oráculo: OUT 6,73e-5 e LSE 9,63e-6. Tolerância não prova
igualdade exata com baseline.

Bulk F32: **8 de 696 respostas divergem em bytes**, todas nq1, com requests
idênticos. Serial e bulk de cada candidato são iguais entre si, mas não ao
baseline. O gate de igualdade bulk falhou. A causa exata não está estabelecida
neste resumo; não atribuir a falha a uma operação sem evidência adicional.

| Braço | Prefill total t/s | Trecho remoto t/s | Decode t/s | Wall s |
| --- | ---: | ---: | ---: | ---: |
| A1 | 462.30 | 316.68 | 21.39 | 65.56 |
| B1 | 465.43 | 318.71 | 21.33 | 65.56 |
| B2 | 456.02 | 308.79 | 21.36 | 65.86 |
| A2 | 464.60 | 317.95 | 21.42 | 65.41 |

A1 reutiliza o último A2 de WHT. B1/B2 são Q_rot; A2 é uma nova execução do
baseline após rollback. Decode médio relatado: 21,40 t/s baseline versus
21,35 t/s candidato; delta relatado -0,25%, dentro do ruído. Não recalcular
precisão ausente a partir das células arredondadas, nem chamar isso de piora
estatística comprovada.

B1/B2 geram tokens idênticos entre si, mas divergem de A a partir do token 23;
A1/A2 são iguais. Tokens diferentes isoladamente não provam perda de qualidade.
A divergência muda o workload end-to-end após esse ponto e confunde a comparação
de desempenho. Falha do gate exato e ausência de ganho mensurável bastam para
**não promover** Q_rot.

## Limites do ABBA e do analisador

Há apenas um bloco ABBA transcrito para cada candidato, com A1 reutilizado em
Q_rot. As tabelas não incluem manifesto bruto completo, hashes verificáveis por
corrida ou réplicas independentes suficientes. Não transformá-las em entradas
supostamente completas para `qualification/qualify_abba.py`, não inventar
hashes/amostras e não publicar p-valores ou intervalos não calculados a partir
de dados adequados. Throughput t/s não é latência ms. O +0,08% remoto WHT não é
a métrica percentual de latência usada pelo analisador.

## Outros candidatos e estado final relatado

- GQA2: histórico de cerca de 2% em prefill, mas KLD 1,11 e top1 divergente em
  1 de 16 passos. Configuração diferente da campanha atual, não repetido aqui.
  Não promover por tolerância parcial CPU.
- MQ2: modelo CPU, sem shader integrado, PSO ou medição de TPS.
- LDS wave64: novo experimento inerte com testes host; DXC, LDS compilada, PSO,
  corretude e desempenho GPU continuam pendentes. Seguir seus gates próprios.

Estado final **relatado naquele momento**: Xbox no pacote 1005, com executável
byte a byte idêntico ao baseline 939; aplicativo e porta RKVA ativos. Serviço de
produção ativo, `/health` OK e `/metrics` HTTP 200; serviço de vídeo reativado.
Nada integrado ao main. Esta atualização documental não consultou nem alterou
novamente o console, serviços ou produção; não garante que esse estado persista.

## Validação desta atualização documental

No ambiente local isolado de publicação: 10 testes ABBA e 6 testes LDS
reexecutados; integridade do pacote verificada após regenerar o manifesto.
Nenhum código de candidato foi alterado. Nenhum novo teste DXC/DXIL, Windows,
GPU, console ou produção foi executado por esta atualização.
