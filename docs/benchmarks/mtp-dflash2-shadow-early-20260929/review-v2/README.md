# Revisão do lançamento antecipado — resultado controlado

Fonte: commit `3d4128354` mais o diff registrado em `git.txt`. Branch `feat/dflash2-shadow-observation`. O modo continua somente observacional.

## Correções

- `GGML_DFLASH_SHADOW_EVERY=0` volta a significar apenas sincronização, inclusive com EARLY=1. O commit revisado o convertia incorretamente para 1.
- Valores negativos, inválidos ou fora do intervalo são rejeitados; EARLY aceita 0 ou 1.
- Os modos early/late contam a cadência antes do MTP em todas as tentativas. Propostas vazias não deslocam a seleção de ciclos.
- Cancelamentos do primário registram a identidade do trabalho, permitindo reconciliar cada lançamento.
- Runner passa a fixar e conferir CPU por requisição, registrar o diff e todos os binários precarregados, usar diretório novo/porta 59589/lease de benchmark, e preservar logs em texto não ignorado pelo Git.

## Validação

Build e quatro testes selecionados passaram. O teste de ambiente/cadência cobre zero, limites, entradas inválidas e alinhamento na presença de propostas vazias. A bateria real inclui sync-only com early ligado e MTP p_min=1 para exercitar cancelamento sem espera.

Todos os casos usam contexto 16384, batch/ubatch 64, um slot, KV q4_0, alvo Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf/CUDA0 e auxiliar DFlash2-Q4_K_M/Vulkan0; MTP n=4, auxiliar n=7; temperature 0, seed 42, 192 tokens. APU 20 W, SCLK solicitado 2700 MHz, CPU powersave/EPP power e faixa 419175–3301000 kHz. Políticas conferidas antes/depois das requisições.

## Throughput observado

| Modo | Repetição tok/s | Código tok/s | Longo tok/s |
|---|---:|---:|---:|
| early-off | 83.13 | 57.22 | 55.28 |
| early-on | 83.05 | 56.84 | 54.77 |

Medianas de três execuções curtas e duas longas. A ordem é sequencial off/on. Os hashes coincidiram em todas as requisições pareadas; as diferenças de throughput ficaram abaixo de 1% nesta bateria. Isso não prova custo zero em outras cargas e não mede ganho de um híbrido ativo.

## Comparação das mesmas âncoras

O analisador cruzou request ordinal, posição e token âncora: 101 observações em comum, sem observações exclusivas de um dos modos. As sequências confirmadas pelo alvo são idênticas em cada par. Contadores incluem aquecimento.

| Métrica nas 101 âncoras | early-off | early-on |
|---|---:|---:|
| Prontos no prazo | 87 | 97 |
| Prefixo coincide | 76 | 76 |
| Pronto + coincide + sufixo | 67 | 73 |
| Tokens candidatos | 172 | 194 |

Atrasos caíram de 14 para 4. Oportunidades passaram de 67/101 (66,3%) para 73/101 (72,3%). A melhora é de prontidão; os candidatos continuam sem entrar na resposta. Excluindo aquecimento, oportunidades são 65/96 e 70/96.

## Regressões exercitadas

- Sync-only: zero lançamentos e zero erros com EVERY=0/EARLY=1.
- Primário sem proposta: 223 lançamentos, 221 cancelados pelo primário, duas observações, nenhum pendente e nenhum erro. Os hashes coincidem com o MTP p_min=1 isolado. Esse é um canário de cancelamento; o custo de trabalhos desperdiçados continua existindo.
- Off: 103 = 101 observações + 2 cancelamentos. On: 117 = 101 + 16, dos quais 14 por primário abortado. Todos os IDs foram reconciliados.
- Serviços de produção restaurados e /health OK; CPU restaurada exatamente ao snapshot inicial. A rodada anterior `review-v1/` sofreu interferência externa, foi interrompida e está excluída destas conclusões.

## Artefatos

`summary.json`, `results.json`, `review-summary.json`, comandos e ambientes completos, hashes de binários incluindo libllama-server-impl.so, snapshots da CPU, respostas, telemetria compacta e logs completos `*.server-output.txt`. O analisador tolera prefixos do logger em linhas SHADOWv2.

Reanálise sem inferência: `python3 ../analyze_review.py review-v2` a partir deste diretório. Nova execução: a partir da pasta mãe, `python3 runner.py review-v3`. A senha é solicitada por getpass; configurações e produção são restauradas ao final.
