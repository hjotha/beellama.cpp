# Qualificação ABBA offline (sem execução remota)

Gate orientado a candidatos de **decode**: exige melhora em decode e total.
Não serve para decidir promoção de MQ2 ou outros candidatos só de prefill sem
um protocolo próprio pré-definido.

Este analisador só lê JSON local. Não compila, publica, liga flags, acessa Xbox
nem decide promoção. Use Python 3 (stdlib), sem instalar dependências:

```
cd qualification
python3 -m unittest -v
python3 qualify_abba.py synthetic-example.json > resultado.json
```

`synthetic-example.json` e `synthetic-result.json` são **fixtures sintéticas**,
não medições. Até um ganho sintético grande termina `not_qualified`. Não mude
`data_kind` para obter aprovação. O teste unitário do caminho `measured` muda
o rótulo somente em memória para testar código, nunca gera evidência real.

## Contrato de coleta

Copie o formato da fixture, substituindo-o por registros realmente coletados.
Use tempos de latência positivos em milissegundos, sem converter throughput em
latência sem documentar a transformação. Cada corrida contém:

- sequência estritamente crescente, ID único, variante A/B e warmup explícito;
- workload completo idêntico: hashes SHA256 de modelos, requests e tokens;
  hardware, comando, configuração (contexto, codecs, batch/ubatch, splits,
  threads, scheduler, network, MTP), sampling e política de clocks/temperatura;
- artefato exato da variante: commit, hashes source/binary/PSO, compilador,
  flags de compilação e feature flags. Mesma variante não pode trocar artefato;
- hash do log bruto, com timings kernel, remote, prefill, decode e total.

Registre listas ordenadas de modelos/shaders e seus hashes em manifests canônicos
quando houver múltiplos arquivos; o hash desse manifest entra no campo SHA256.
A/B devem ter mesmos compiler flags; a única diferença experimental permitida
fica explicitamente nas feature flags/artefatos A/B. Audite manualmente essa
diferença: o analisador não sabe quais flags são equivalentes semanticamente.
Não misture modelos, codecs, shapes ou configurações em um relatório. Faça um
arquivo por workload. Preserve logs, manifests e medições ambientais fora do
Git quando forem privados. Nunca inclua tokens de autenticação ou secrets.

Warmup deve preceder todas as medições e não entra nas estatísticas. Colete
blocos completos A/B/B/A consecutivos; não escolha somente rodadas favoráveis.
Registre falhas, repetições e exclusões, com motivos, antes da análise. Interrupção
ou troca de configuração invalida a campanha: não reordene/ampute logs para
satisfazer o parser. A integridade do arquivo de entrada fica no relatório.

Os hashes são **declarações**, não atestação: verificar os bytes dos arquivos,
logs, PSOs executados, ordem real e controles ambientais ainda é obrigatório.
Não reutilize o hash de um log para fingir réplicas independentes. Um log agregado
pode conter várias corridas, mas cada ID deve apontar a uma observação distinta.

## Estatísticas e decisão conservadora

Cada bloco usa média dos dois A e média dos dois B. Redução percentual de
latência = 100 × (A − B) / A; positivo é melhor. Isso NÃO é ganho percentual de
throughput. Relata medianas, MAD, mínimo, máximo e n das latências A/B e dos
ganhos por bloco, além de todos os ganhos por bloco e dispersão intrabloco.

São necessárias pelo menos seis réplicas **de blocos**, não seis requisições do
mesmo bloco. O limiar conservador é max(0,5 ponto percentual, mediana da amplitude
intrabloco A/B normalizada por A, 3 × MAD dos ganhos). Ganhos abaixo desse limiar
são inconclusivos. O teste de sinais unilateral conta blocos acima do limiar;
p < 0,025 é um filtro adicional, não intervalo de confiança. O limiar depende
dos próprios dados: este p é apenas diagnóstico, não um teste confirmatório
calibrado. Autocorrelação, variação térmica, seleção de candidatos e múltiplas
comparações exigem revisão, novas campanhas independentes e protocolo pré-fixado.
Seis blocos não são garantia de poder estatístico.

Kernel e remote são reportados separadamente de decode e total. Ambos decode e
total precisam mostrar sinal de melhora; regressão sinalizada em qualquer fase
bloqueia elegibilidade. Isto é propositalmente conservador para candidatos de
decode: ganhos só de prefill requerem protocolo próprio, não flexibilização
retroativa deste gate. Resultado inconclusivo não prova igualdade, ausência de
regressão nem ausência de ganho. Nenhum resultado promove automaticamente.

## Corretude precede desempenho

Todos os gates precisam ser registrados (`pass`, `fail`, `not_run`), com hash da
evidência para pass/fail: serial, bulk bitwise, matriz de formatos/shapes,
qualidade em contexto longo, serving/MTP. Performance nunca substitui um gate.
Gates incompletos/falhos impedem elegibilidade mesmo com grande ganho.

- CPU/FP64/OUT/LSE: testes locais necessários, não qualificação GPU ou de modelo.
- DXIL/MSVC: compilação não prova execução do PSO correto.
- Serial/bulk: igualdade byte a byte no caminho real, não só tolerância CPU.
- Matriz: codecs/tails/origins/splits/rewind/ring wrap, probes adversariais e
  controles negativos definidos no README principal.
- Qualidade: KLD/top1/perplexidade e serving/MTP no contexto e cadence de destino,
  com o mesmo batch/ubatch. Defina critérios antes da coleta. Nunca promova
  apenas porque diferenças parciais de atenção cabem na tolerância CPU.

Mesmo `eligible_for_human_review_only` exige conferir logs originais, cobertura,
independência, ruído, identidade do PSO, efeitos de sistema e decisão humana.
Nenhuma mudança em produção está incluída neste pacote.

## Evidência recente, somente relato de transcript

Fonte: transcript técnico da máquina .57, consultado em 2026-11-06 às 16:32 UTC. Não contém série bruta
neste pacote, portanto NÃO foi convertido em fixtures ou reanalisado.

- 16:17:27: WHT passou probes seriais e 696 registros bulk byte a byte.
- 16:22:58: ABBA terminou com +0,08% médio remoto relatado, sem ganho decode/total.
  Baseline mantido; WHT descartado por falta de ganho. Não reinterpretamos o
  sinal ou denominador desse percentual como a métrica de latência desta ferramenta.
- Q_rot compilava separadamente; WHT original havia sido restaurado.
- GQA2 histórico em 129k: cerca de 2% prefill, mas KLD 1,11 e top1 divergente.
  Pequenas diferenças parciais não justificam promoção.

Esses relatos não são logs brutos, nem novos testes desta máquina. A conclusão
WHT continua 'não promover'; não é evidência de falha de corretude dos probes.
Esse era o estado Q_rot às 16:32 UTC; a atualização abaixo registra a falha
posterior do gate bulk e substitui esse estado preliminar.

### Retrospectiva adicional: NOTAS.md consultado às ~16:35 UTC

Observações do transcript técnico na .57, consultado em 2026-11-06, **não**
validações novas no dot: WHT compilado em MSIX 1001 Final/MSVC com DXIL candidato
confirmado; 498 checagens seriais (tails 0/128); 696 requests e bytes OUT/LSE
iguais no bulk; 1024 tokens idênticos nos quatro braços. DXIL: 20→13 barreiras
relatadas. Caso: 8k, P4096, output1024, KVarN4/E4M3FN/CUDA/prefetch, base
`7310f5d7e`. Gates restantes de cobertura/qualidade não são presumidos completos.

Valores arredondados transcritos, preservados sem conversão ou recomputação:

| Braço | Prefill total t/s | Trecho remoto t/s | Decode t/s | Wall s |
| --- | ---: | ---: | ---: | ---: |
| A1 | 465.13 | 318.97 | 21.38 | 65.48 |
| B1 | 464.90 | 317.44 | 21.38 | 65.47 |
| B2 | 465.73 | 318.71 | 21.38 | 65.44 |
| A2 | 462.30 | 316.68 | 21.39 | 65.56 |

Unidades confirmadas no cabeçalho de NOTAS.md: throughput em tokens/s e wall
em segundos. Não copiar throughput para campos `*_ms` de latência. Baseline
939/restaurado 1002 usavam o mesmo executável; candidato WHT era 1001.
Há apenas um bloco ABBA nessa transcrição. Não é série bruta completa, não contém
hashes verificáveis por corrida e não satisfaz o contrato do analisador. O ganho
remoto médio relatado de +0,08% está no ruído; baseline mantido e WHT não promovido.
Os 1024 tokens iguais nesse caso não provam cobertura geral de qualidade.

GQA2 histórico: divergência top1 em 1 de 16 passos forçados e KLD 1,11; duas
tentativas de correção falharam segundo o mesmo repasse. Não é candidato
qualificado. Qualquer gate de qualidade incompleto também impede promoção.

### Atualização Q_rot: transcript consultado às 16:38:27 UTC

Fonte: transcript técnico da máquina .57, consultado em 2026-11-06. Estes são
resultados relatados da execução remota, não novos testes no dot nem logs brutos:

- 16:33:13: build Q_rot completo.
- 16:34:22: pacote com DXIL esperado e reducer original no Xbox.
- 16:37:29: probes seriais passaram nas tolerâncias, mas **8 das 696 respostas
  bulk diferiram em bytes**. O gate de igualdade bulk falhou e bloqueia promoção.

A causa das diferenças e o resultado final de desempenho ainda não foram
determinados neste relato. Medição isolada poderia continuar para diagnóstico,
sem promover o candidato. Tolerância serial não substitui igualdade byte a byte;
nenhuma alteração de gate foi feita para aceitar essas diferenças. Não foram
criadas amostras, hashes ou resultados do analisador a partir desse resumo.
