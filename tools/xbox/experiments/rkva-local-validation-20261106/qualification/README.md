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

## Resultado final externo, não entrada do analisador

Leia [FINAL-XBOX-RESULTS-20261106.md](../FINAL-XBOX-RESULTS-20261106.md).
Fonte: `NOTAS.md` e transcript técnico consultados em 2026-11-06 às 16:52:32 UTC;
resposta final às 16:44:46 UTC. Esse resultado substitui os relatos preliminares.
Não houve novos testes remotos nesta atualização documental.

- WHT: 498 checagens seriais e 696 bulk com requests e OUT/LSE idênticos;
  1024 tokens iguais nos quatro braços. +0,08% de throughput remoto médio
  relatado, no ruído; sem ganho justificável. Não promover.
- Q_rot: build Final/MSVC/MSIX 1004 e DXIL esperado relatados; serial passou,
  mas 8/696 respostas bulk F32 divergem em bytes, todas nq1, com requests iguais.
  B1/B2 geram tokens iguais entre si, mas divergem do baseline a partir do token
  23. Falha do gate exato e ausência de ganho: não promover.
- Decode Q_rot médio relatado: 21,40 t/s baseline e 21,35 t/s candidato,
  delta relatado -0,25%. Não é demonstração estatística de regressão.
- Baseline restaurado segundo o relatório final, sem integração ao main.
- GQA2 histórico não qualificado por qualidade; não repetido nesta campanha.
  MQ2 permanece modelo CPU. LDS wave64 continua sem validação GPU.

As tabelas do relatório têm somente **um bloco ABBA por candidato**, valores
arredondados e nenhum manifesto bruto completo por corrida. Q_rot reutiliza
como A1 o A2 final de WHT; a geração diferente a partir do token 23 confunde a
comparabilidade do workload end-to-end. Tokens diferentes, isoladamente, não
provam perda de qualidade. Não invente hashes, réplicas, intervalos ou p-valores.

Não passe essas tabelas por `qualify_abba.py` como evidência completa e não
copie throughput t/s para campos de latência `*_ms`. O percentual remoto WHT
é throughput relatado, não redução de latência deste analisador. Os gates e a
exigência de novas campanhas independentes continuam inalterados.
