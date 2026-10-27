# QA semântica INT8 RKVA — entrega offline (2026-10-26)

Branch: `perf/rkva-quality-gate-20261026` (clone
`/home/hjotha/beellama-agents/quality-gate`, base `4bf192bb2`).

**Esta entrega não executou hardware.** Nenhum `llama-server`, CUDA, Xbox,
`systemctl`, build UWP/MSVC, push ou merge foi usado. O único artefato de
execução foi o preview inerte da ferramenta (nenhum filho, nenhum arquivo de
saída). O gate de hardware é do root; a revisão obrigatória antes de integrar é
do Astra high.

## 1. Arquivos entregues

| Arquivo | Conteúdo |
| --- | --- |
| `tools/xbox/scripts/rkva-quality-gate.py` | Ferramenta de QA teacher-forced, inerte por padrão, `--execute` explícito |
| `tools/xbox/tests/test_rkva_quality_gate.py` | 82 testes offline com processos/deploy/HTTP falsos |
| `docs/agent-reports/rkva-quality-gate.md` | Este relatório |

Nenhum arquivo C++/produção, tier, preset, modelo, cache de slot ou configuração
foi alterado. O clone de integração antigo não é referenciado.

## 2. Uso e defaults

```bash
# Preview (padrão): não inicia processo, não fala com o Xbox, não escreve saída.
nice -n 19 python3 tools/xbox/scripts/rkva-quality-gate.py \
  --model <target.gguf> \
  --prompt-tokens /home/hjotha/beellama-gate/20261017/tokens-prompt120k-t2.json \
  --prefix-snapshot /home/hjotha/llama-slot-cache/rkva-2xn256-dev/a7e8962f16a8/prod-prefix-p102400-c131072-n256-w4096.bin \
  --output <run-dir>
```

O preview imprime `mode: preview`, `starts_child: false`, `writes_output: false`,
o plano de casos com hashes, o gate congelado, o contrato do prefixo e o
`command_preview`. A flag `--execute` é a única forma de iniciar um servidor.

Defaults que a ferramenta **exige**: C131072, P102400, B1024, UB512,
wavefront 2xN256, CUDA FP8 `mapped`, range kernel `fine`, TCP prefetch ligado,
janela KVarN 4096, guard headroom 128 MiB, `GGML_CUDA_GRAPH_GUARD_SAMPLE_MS=10`,
`GGML_CUDA_GRAPH_GUARD_SLACK_MB=64`, input reuse legado OFF e mask-pack no
default (variável removida). `GGML_CUDA_GRAPH_*` é validado contra
`ggml/src/ggml-cuda/common.cuh` e um valor herdado mais estrito é recusado, nunca
enfraquecido.

Contexto é limitado a `(0, 160000]`; 160001 é erro de configuração. O perfil
congelado mantém o limite de 131072. Um snapshot C131072 **não** é compatível
com C160000: o contrato do snapshot é mantido e a ferramenta recusa o mismatch,
sem editar metas e sem aceitar prefill frio. O primeiro QA roda só em C131072;
160k fica para um teste isolado do root, com snapshot compatível próprio.

## 3. Protocolo teacher-forced

Dois braços, sempre em servidor privado novo (novo `HELLO` no Xbox):

- **OFF**: rowlane ON, flag `rkva-mq-i8.flag` ausente.
- **ON**: rowlane ON, `rkva-mq-i8.flag` presente.

Cada braço restaura **apenas** o prefixo local P congelado e reavalia a cauda.
A primeira requisição de cada braço é a ingestão obrigatória da cauda
multiquery; `n_predict=1` isolado não é tratado como prova de multiquery.

O body de cada passo é `/completion` com os IDs exatos, `n_predict=1`,
`n_probs=n_vocab`, `post_sampling_probs=false`, `cache_prompt=true`, `seed=1234`,
`temperature=0`, `id_slot=0`, `stream=false`. A distribuição inteira é validada:

- todos os IDs do vocabulário presentes e únicos;
- `id` inteiro no range, `logprob` finito;
- `std::numeric_limits<float>::lowest()` (≈ `-3.4e38`, sentinela de
  `logarithm(0)` em `tools/server/server-task.cpp`) vira probabilidade zero;
- massa normalizada próxima de 1 (tolerância configurável e congelada);
- NaN/Inf, ID repetido, ID faltante, `null`, logprob positivo, resposta truncada
  ou `predicted_n != 1` são fatais.

O baseline (OFF) cria a sequência teacher. Em cada passo o próximo token é o
`top1` do baseline; **apenas** essa sequência é usada no braço ON. O top1 do
candidato nunca vira teacher. Os hashes dos prompts de cada passo são gravados e
comparados entre os braços; históricos diferentes são recusados.

Não há `erase` entre passos e não há rewind de um token para o prefixo. O próximo
prompt inclui o token teacher. O passo 0 precisa reavaliar a cauda real a partir
do snapshot (`cache_n <= P`, `prompt_n > 0`) e os passos seguintes precisam ser
exatamente `cache_n == len(prompt anterior)` e `prompt_n == 1`; replay/rewind
não é aceito. Os dois braços precisam ter `cache_n`/`prompt_n` idênticos.

## 4. Casos e gates

Seis caudas determinísticas independentes, geradas por `/tokenize` do mesmo
binário: `portuguese`, `documents`, `code`, `numbers`, `retrieval`, `factual`.
Só o sufixo depois de P é tokenizado; o prefixo compartilhado nunca é alterado.
Os comprimentos padrão são P+4096 (`106496`), P+16384 (`118784`) e `130560`.
`--smoke` reduz para um caso em 106496 com 2 passos; `--cases`, `--lengths`,
`--teacher-steps` e `--cases-json` são os seletores.

Retrieval/factual têm três chaves únicas (início/meio/fim). A chave final fica a
exatamente 128 tokens do fim da cauda (limite 128±1), com a pergunta no fim.
`--cases-json` aceita corpus real, `answer` e/ou `answer_ids`; o comprimento
tokenizado é verificado e nunca é truncado/preenchido em silêncio.

Gates pré-declarados (configuráveis, congelados no `binding.json` antes de
qualquer filho; nunca relaxados automaticamente):

| Gate | Limite |
| --- | --- |
| Razão de perplexidade teacher por estrato | ≤ 1.01 |
| KL mean (OFF‖ON) | ≤ 0.001 nat |
| KL p99 | ≤ 0.01 nat |
| Delta NLL por caso | ≤ 0.02 nat |
| Ativação ON (`mq2_i8.calls > 0`, `available`, `exact=false`) | obrigatória |
| Ativação OFF (`mq2_i8.calls == 0`, mq2 fine > 0) | obrigatória |
| rowlane/decode calls > 0 nos dois braços | obrigatório |
| FP8 GPU bulk > 0 e fallback CPU == 0 | obrigatório |
| WF2xN256 e TCP overlap | obrigatório |
| Retrieval baseline acertado que falha no candidato | falha |
| Baseline retrieval informativo | obrigatório |
| Divergência top1 com margem alta | NEEDS_REVIEW |

O relatório final traz acurácia absoluta por braço, `top1_agreement`, margens
top1–top2 e contagem de divergências de alta confiança. Não existem step probs
capturados em história gerada pelo candidato: toda distribuição é teacher-forced
na sequência do baseline. Um caso de retrieval cujo baseline falhou é registrado
como não informativo e não qualifica a qualidade do candidato. Um PASS é rotulado
`QUALIFIED_SCOPE_ONLY` e sempre carrega `int8_default_authorized: false`; ele não
autoriza INT8 como default global. Uma matriz sem retrieval informativo, sem
métrica finita, sem ativação positiva ou sem o controle de determinismo não
passa.

O controle de determinismo (`--determinism-control auto`) roda um processo OFF
novo para o caso longo (≥130560), compara a sequência teacher e o top1 e grava a
maior diferença de probabilidade. `off` explícito impede PASS; `--smoke` não
exige o controle e nunca qualifica a matriz.

## 5. Lifecycle, Xbox e flags

- Cada braço usa um `llama-server` privado, filho da ferramenta, em porta
  loopback reservada; `start_new_session=True`; `SIGTERM` e `wait`; exit
  diferente de 0 é falha. Timeout de startup, request e deploy é explícito.
  Servidor iniciado e não confirmado é terminado no próprio `__enter__`.
- O guard de host é `fail-closed`: processos `llama-server`, `llama-bench`,
  `llama-perplexity`, `ninja`, `cmake`, `ctest`, prefixos `test-rkva`/
  `bench-rkva` e scripts RKVA ativos recusam o `--execute`. Texto de agente não
  conta; o próprio processo/ancestrais são ignorados.
- `deploy.sh pfn` descobre o pacote; o prefixo de identidade precisa ser
  `GianlucaMazza.xllama_` e múltiplos pacotes registrados são recusados. O app
  nunca é iniciado, parado ou sondado. Credenciais vêm de
  `~/.config/xllama/xbox-env`; valores não são impressos nem gravados. O
  `XBOX_IP` do arquivo é sobrescrito pelo host de `--xbox-host` para garantir o
  mesmo console do modelo.
- `deploy.sh` tem stdout interno completo para `json.loads`; as cópias de
  auditoria são redigidas e limitadas, então um `list-localstate` grande não é
  truncado antes do parse e a senha não vaza nos logs.
- As flags originais são lidas por `list-localstate` antes de qualquer mudança.
  `rkva-decode-rowlane-off.flag` é removido (rowlane ON) e o estado original é
  restaurado em `finally`. `rkva-mq-i8.flag` é removido no braço OFF e criado no
  ON; ao final volta ao estado original, qualquer que seja. `rkva-mq-gqa2.flag`
  também é capturada: todo braço INT8/rowlane existente a remove, o estado
  original é restaurado no mesmo `finally` e um mapa antigo de duas flags é
  migrado lendo o estado atual. Arm com INT8+GQA2 juntos é rejeitado antes de
  tocar no console. Delete 404 é idempotente; 403/transporte/erro real é fatal. A
  restauração é verificada por uma nova listagem e gravada em
  `flags-restored.json`.
- A ferramenta nunca chama `start-app`/`stop-app`/`systemctl`, não altera tier ou
  configuração de produção e não usa GPU diretamente.

## 6. Streaming, memória e SHA

Cada passo é gravado imediatamente como `array('f')` little-endian em um fluxo
gzip, com header `steps/vocab` e SHA do arquivo + SHA por passo no
`accounting.json`. O `run_arm` não mantém a matriz completa, não guarda as
distribuições anteriores e descarta o JSON HTTP bruto depois de alinhar por ID.
O `compute_case_result` abre um `DistributionReader` bounded para OFF e outro
para ON, lê um par por passo e calcula KL/NLL/top1 incrementalmente com
acumuladores float64 (`math.fsum`); as matrizes completas nunca são carregadas.
O reader valida header, tamanho comprimido, truncamento, passos/vocab errados,
dados extras e reutiliza um único buffer por leitor. Retrieval usa os IDs
teacher e o top1 por passo, não matrizes duplicadas.

O `binding.json` inclui as bibliotecas LLAMA/GGML **ativas** resolvidas por
`ldd`/DT_NEEDED (não todas as versões antigas no diretório), com SHA de conteúdo.
`LibraryGuard` confere o conjunto ativo e os `loaded_libraries` de `/proc` no
startup e no pós-arm/controle; usa `stat` e só rehasha um arquivo cujo stat
mudou, evitando hash de gigabytes por passo N1. `libraries.json` por braço e o
`final.json` vinculam o conjunto esperado; mutação de lib, caminho carregado
fora do binding ou rebuild de `.so` invalidam resume/PASS.

## 6.1. Prefixo, retomada e limitação de FULL overflow

O prefixo existente é reusado por
`H.load_prefix_snapshot` + `H.slot_operation('restore')`, com SHA/modelo/P/C/
batch/ambiente do contrato. Um novo P100k só é criado com
`--allow-new-prefix` explícito; sem snapshot e sem a flag, a ferramenta recusa.
Nunca sobrescreve meta ou sidecars.

A saída é retomável por caso completo: `binding.json` fixa
fonte/binário/modelo/snapshot/geometria/gates/plano; `complete.json` guarda o SHA
dos artefatos; braços e casos incompletos são refeitos, PASS/FAIL ficam
persistidos e corrompidos são recusados. Um FAIL só é refeito com
`--retry-failed`. Cada caso grava `distributions.bin.gz` (float32 gzip),
`accounting.json`, `requests.json`, `teacher.json`/`profile.json` e um resumo
por passo; não há JSON de 10 GB nem TPS tratado como benchmark.

**FULL slot com overflow não é suportado hoje.** A ferramenta não tenta salvar
overflow nem reporta isso como PASS. O código atual recusa o save FULL nativo em
`src/llama-kv-cache-kvarn.cpp:5045-5050`, recusa o restore de overflow de
standard-cache em `:5399-5405`, mantém op 17/18 nativos desabilitados e não tem
codec para owner RAM records/stage/tail. Por isso só o P local existente é
restaurado; a cauda avaliada é sempre reavaliada. No primeiro passo a ferramenta
exige `cache_n <= P` e `prompt_n > 0`; um snapshot que já trouxesse a cauda
falha em vez de virar PASS. O root pode auditar itens adicionais fora deste
escopo.

A opção `--verify-overflow-save-refusal` está **desabilitada**. Ela falha na
validação, sem abrir child, sem tocar flags e sem API nova; não existe
qualificação de FULL overflow nesta matriz. O codec nativo continua
`unsupported`, e o root/Astra usam um probe manual separado (dois braços, guard
FULL exato). Nada aqui é um FEATURE/false pass de overflow restore.

## 7. Validação offline executada (comandos reais)

Todos os comandos abaixo rodaram neste clone, em CPU, com `nice -n 19`. Não
houve chamada a modelo, GPU, Xbox, `llama-server` ou rede.

```bash
nice -n 19 env PYTHONPYCACHEPREFIX=/tmp/opencode/pycache \
  python3 -m py_compile \
  tools/xbox/scripts/rkva-quality-gate.py \
  tools/xbox/tests/test_rkva_quality_gate.py
# exit 0

nice -n 19 env PYTHONPYCACHEPREFIX=/tmp/opencode/pycache \
  PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s tools/xbox/tests -p 'test_rkva_quality_gate.py' -v
# Ran 82 tests in 3.368s
# OK

nice -n 19 env PYTHONPYCACHEPREFIX=/tmp/opencode/pycache \
  PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s tools/xbox/tests -p 'test_bench_rkva_range.py'
# Ran 48 tests in 0.044s
# OK

nice -n 19 env PYTHONPYCACHEPREFIX=/tmp/opencode/pycache \
  PYTHONDONTWRITEBYTECODE=1 python3 -c \
  "import ast,pathlib; [ast.parse(pathlib.Path(p).read_text(), p) for p in \
  ['tools/xbox/scripts/rkva-quality-gate.py', \
   'tools/xbox/tests/test_rkva_quality_gate.py']]; print('AST_OK')"
# AST_OK

git diff --check
# exit 0
```

Cobertura dos testes offline inclui (82 testes): ordem arbitrária de IDs e sentinela zero;
ID faltante/duplicado/NaN/fora do vocabulário; KL/NLL golden e distribuições
iguais; massas e gates ausentes; ratio/KL por caso e por estrato; retrieval
informativo e regressão; divergência de alta confiança como NEEDS_REVIEW; flags
originais restauradas (ON e opt-out de rowlane), delete 404, falha real de
delete/upload, PFN e redação de segredo; startup timeout, exit não-zero, request
timeout e SIGTERM; binding de fonte/binário/modelo/prefixo/gate/contexto;
limite 160000, seed 1234 e ausência do literal de tier; ativação INT8 positiva obrigatória;
gerador determinístico com comprimento exato e chave final a 128 tokens;
corpus externo com `answer_ids`; retomada de tokens/casos/FAIL, ordem de finalização do perfil, controle de determinismo, preflight smoke, streaming bounded, library binding/loaded guard, opção overflow desabilitada, execução completa de um caso com teacher e preview sem filho/escrita.

Preview final verificado (mesmo binário/script, sem `--execute`):

```bash
nice -n 19 python3 tools/xbox/scripts/rkva-quality-gate.py \
  --model /tmp/opencode/qg-final-preview/model.gguf \
  --prompt-tokens /tmp/opencode/qg-final-preview/tokens.json \
  --prefix-snapshot /tmp/opencode/qg-final-preview/missing.bin \
  --output /tmp/opencode/qg-final-preview/out
# preview_rc=0; no_output_dir
# status= PREVIEW_ONLY; starts_child=False; writes_output=False
# lengths=[106496, 118784, 130560]; cases=18
# guard sample_ms=10 slack_mb=64 headroom_mib=128
# GGML_RKVA_CUDA_FP8=mapped; GGML_RKVA_LOCAL_INPUT_REUSE=0
```

O preview também listou `cmake`/`ninja` do build `-j10` do root como processos
ativos; ele **não** recusa no preview, mas o `--execute` seria recusado pelo
guard enquanto esse build estiver vivo.

## 8. Correções da pré-review Astra

- Finalização do perfil movida para **depois** do `with ServerSession` e do
  restore de flags: perfil ausente, exit != 0 ou falha de cleanup impedem
  marker PASS. Há testes com `finalize_arm_dir` real (perfil ausente e execução
  completa de caso), além do teste de ordem.
- `run_determinism_control` testado com assinaturas reais, servidor OFF novo e
  `run_arm` real; o antigo caminho de 5 argumentos não existe mais.
- `DeployClient` preserva stdout completo interno para JSON e trunca/redige
  somente auditoria; teste com JSON >20 KB e flags no início/meio/fim.
- Distribuições passaram a streaming gzip por passo; reader bounded e teste de
  128 passos com vocabulário 20k prova memória limitada (não 10 IDs).
- Preflight smoke separado OFF/ON, 2 passos, 106496, antes da matriz; perfil
  final por braço/caso ocorre após o destructor. O proof grava model/binary e,
  no `execute`, também PFN/app/flags originais. `--smoke` é o próprio smoke.
- Biblioteca: binding agora inclui SHA das libs LLAMA/GGML ativas; guard
  pré/pós-arm, startup/final loaded e controle impede PASS/resume após rebuild.
  Testes cobrem stub igual com lib mudada, loaded path desconhecido e mutação
  durante arm sem marker.
- `--verify-overflow-save-refusal` foi **desabilitada** na validação; não abre
  child/flags, não cria API nova e não qualifica FULL overflow. O root/Astra já
  têm probe manual separado.
- Retrieval usa IDs teacher/top1; baseline que falhou é registrado como não
  informativo e não qualifica qualidade. Não há step probs de história gerada
  pelo candidato (protocolo é teacher-forced).

## 9. O que NÃO foi validado

- Nenhuma distribuição real do modelo, nenhum `llama-server`, CUDA, Xbox, HELLO
  real, perfil RPC real, PSO INT8 real ou snapshot real foi lido/executado.
- Nenhuma conclusão de qualidade, ativação INT8, TPS, ganho ou regressão de
  hardware é afirmada aqui. `--execute` **não** foi rodado.
- A matriz de seis caudas × três comprimentos × 128 passos é deliberadamente
  cara; o root deve começar por `--smoke` e por um caso/estrato, conforme o
  orçamento de GPU.
- O resultado de hardware só vale após o root confirmar: build terminar (guard
  livre), app `GianlucaMazza.xllama` exato, HELLO novo por braço, snapshot
  C131072 verificado, perfil RPC com ativação positiva, flags restauradas e
  nenhum servidor órfão.

## 10. Próximos passos do root e revisão Astra

1. Aguardar o build `-j10` terminar; rodar primeiro o preview acima e conferir
   `case_plan`, `binding`, `gates` e `command_preview`.
2. Rodar `--smoke` (1 caso/106496, 2 passos) em C131072, com o snapshot
   existente; conferir ativação ON/OFF, rowlane, bulk FP8, WF2, TCP e flags.
3. Só então rodar a matriz seletiva (`--cases`, `--lengths`) e o controle de
   determinismo; manter um único job de GPU por vez.
4. Astra high revisa antes de integrar: protocolo teacher (mesma sequência),
   gates conservadores, limite FULL overflow, contrato do snapshot, restauração
   de flags e o fato de PASS ser `QUALIFIED_SCOPE_ONLY`/sem default global.

O relatório não deve ser lido como aprovação de hardware. Ele documenta apenas a
ferramenta e a validação offline desta entrega.

O Astra high ainda precisa revisar o diff **depois** destes fixes; esta entrega
não afirma aprovação final do script nem do hardware. O root continua sendo o
dono da execução e da decisão de gate.
