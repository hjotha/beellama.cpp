# RKVA: candidatos isolados e validação CPU

Pacote de pesquisa, **sem ativar otimizações em produção**. Preparado em 2026-11-06
para o agente local continuar a qualificação de DXIL, Windows e Xbox. Não é um
benchmark do console e não demonstra aceleração.

## Estado e proveniência

- Base do branch de publicação: `e64ad7c43ee6f2ae6ced1d1a47ba462d7751594e` (`main`).
- Fixtures dos candidatos: snapshot `4bad0205c1658ac1aa67b358b89884c2e9cddf4c`.
- Modelos CPU herdados: `ff008e07eea8e471ebe247ad3ef5dfce35b828bf`.
- GQA2: branch histórico `582cc8d8e70216e0ba697e735fbe3c6817704f9a`.
- As fixtures e seus hashes ficam nos subdiretórios. Testar uma fixture antiga não
  valida todas as mudanças do main atual.
- Os patches Q_rot e WHT passaram `git apply --check` sobre a base de publicação.
  Isso não é teste de compilação ou runtime. Eles são entregues como arquivos de
  patch; nenhum shader, header DXIL ou PSO de produção foi modificado.

| Candidato | O que existe | Resultado CPU histórico | O que falta |
| --- | --- | --- | --- |
| Q_rot reuse | Patch HLSL opt-in, flag OFF; modelos e fixtures | 44 casos completos, OUT/LSE bitwise A/B | Relato externo: compilou/executou; bloqueado por 8/696 bulk divergentes |
| WHT ping-pong | Patch HLSL do reducer; modelo de sincronização | 2.469 vetores, 94 casos; ASan/UBSan | Relato externo: corretude passou; ABBA +0,08%, sem ganho real |
| GQA2 | Patch parcial de shader, teste de layout e política | 6 testes; 5.598.720 combinações | Falha histórica de qualidade em 129k; não promover por tolerância CPU |
| LDS wave64 rowlane | Patch opt-in Q4/Q8/F16/SAW, sem integração | 6 testes host nesta atualização | DXIL/LDS compilada, PSO, corretude GPU e desempenho |
| MQ2 rowlane | Modelo executável de atenção, **não um shader** | 40 cenários / 104 execuções contra FP64 | Implementação HLSL, registradores, PSO e GPU |

Gates originais mantidos: OUT `abs(delta) <= 3e-4 + 3e-4 * abs(ref)`;
LSE `abs(delta) <= 3e-4 + 3e-5 * abs(ref)`, além de finitude. Os máximos históricos
são Q_rot OUT `6.1851e-5`/LSE `9.5979e-6`, MQ2 OUT `1.8272e-4`/LSE `1.4576e-5`.
A igualdade bitwise de Q_rot é entre os dois modelos FP32, não entre FP32 e FP64.
Veja os READMEs e JSONs de cada candidato para a cobertura exata e os controles
negativos. F32/F16/E5M2/E4M3FN não formam uma matriz cartesiana completa de shapes.

## Instruções para o agente local

### 1. Preservar trabalhos em andamento

Leia `AGENTS.md`, `tools/xbox/AGENTS.md` e `tools/xbox/README.md`.
Não troque o branch de um checkout em uso, não interrompa processos e não use
`git worktree` neste repositório. Use um clone separado se necessário:

```sh
git clone --branch astra/xbox-rkva-local-validation-20261106 --single-branch \
  https://github.com/hjotha/beellama.cpp.git beellama-rkva-validation
cd beellama-rkva-validation
git status --short
git rev-parse HEAD
```

O procedimento abaixo não autoriza instalar ou desinstalar o aplicativo,
interromper servidores, mudar produção ou tomar uma sessão de console em uso.
Qualificação ao vivo exige uma janela/lease de console já autorizada.

### 2. Reproduzir as validações CPU

Ambiente de referência: Linux x86_64, Python 3.12, NumPy 2.3.5, g++ 14.2, `cpp`,
`patch` e Bash. Use dependências de confiança já disponíveis; se faltarem,
registre o bloqueio e peça a preparação do ambiente. Não instale dependências
automaticamente nem altere o Python global.

```sh
P=tools/xbox/experiments/rkva-local-validation-20261106
bash "$P/qrot/run.sh"
bash "$P/wht/run_all.sh"
bash "$P/mq2/run_all.sh"
```

Cada subdiretório explica onde deixa os resultados novos. Resultados históricos
não devem ser sobrescritos para aparentar uma nova execução. Falhas de sanitizer,
compilação ou dependência devem ser registradas como falhas ou etapas não rodadas.
Não compare tempos destes modelos CPU com tempos do Xbox.

### 3. Compilar primeiro as fixtures com o DXC real

Com DXC de origem oficial já instalado, sem mudar seus flags para mascarar erros:

```sh
DXC=dxc bash "$P/compile-fixtures-dxil.sh"
```

O script compila somente as fixtures em uma pasta local `.build/dxil`, não
substitui os headers gerados do aplicativo. Compila Q_rot OFF/ON com rowlane
OFF/ON, e reducer baseline/candidato, com `cs_6_0`, entradas e defines equivalentes
aos alvos existentes. Registra versão, comandos, DXIL e disassembly. **Este script
não foi executado com DXC no ambiente CPU de origem.**

Audite o DXIL: endereços e dimensão de Q_rot, recursos/root signature, ausência
de WHT no setup de Q do candidato, barreiras e bancos de LDS, duas normalizações
separadas e ordem aritmética. `reduce_kvarn4` não usa `-Gis` na base registrada;
não introduza `-Gis` somente no candidato e chame isso de comparação equivalente.
Se testar flags adicionais, rode baseline e candidato com os mesmos flags e
declare uma experiência separada.

### 4. Integrar um candidato por vez, somente no clone de teste

Q_rot está bloqueado pela divergência bulk descrita na atualização abaixo.
O procedimento seguinte é referência para investigação, não recomendação de
promoção. Antes de aplicar qualquer patch, confirme que não há alteração local:

```sh
git apply --check "$P/qrot/qrot-reuse-opt-in.patch"
# Aplicar apenas no clone experimental, após revisar o patch:
git apply "$P/qrot/qrot-reuse-opt-in.patch"
```

O patch sozinho **não ativa** Q_rot: a flag `RKVA_DECODE_QROT_REUSE` permanece 0,
sem wiring de PSO. Acrescente uma variante de compilação e seleção de PSO
explicitamente experimental, mantendo baseline e fallback. Não sobrescreva um
header existente com bytes ON mantendo metadados de baseline. Registre source
SHA, defines, DXC e SHA do DXIL; confirme pelo log/counter qual PSO foi usado.
Valide o contrato do produtor: planos Q original/Q_rot/positions, offsets,
codecs, padding de head e uso de Q original no tail. A flag fica restrita ao
decode serial-prefix já qualificado, sem ampliar o gate nq1/GQA6.

Para WHT, use outro branch de teste ou remova somente suas mudanças anteriores
de forma revisada; não use `reset --hard` sobre trabalho desconhecido:

```sh
git apply --check "$P/wht/candidate.patch"
git apply "$P/wht/candidate.patch"
bash tools/xbox/scripts/compile-rkva-range-shaders.sh reduce_kvarn4
```

A WHT mantém H128 seguido de H2 e duas normalizações. Não misture redução de
barreiras com normalização fundida ou wave shuffle dependente da lane.
GQA2 ainda requer portar a integração histórica manualmente e revisar seus
gates; não force hunks. MQ2 ainda requer criar um shader, não apenas ligar flag.

Compile o aplicativo com a toolchain Windows prevista em `tools/xbox/README.md`.
Exemplo, após escolher SDK instalado e revisão de pacote válida:

```powershell
pwsh -File scripts/build-xbox.ps1 -Configuration Release -Backend llamacpp `
  -WindowsSdkVersion $env:RKVA_TEST_WINDOWS_SDK `
  -BuildRevision $env:RKVA_TEST_BUILD_REVISION -MaxParallel 10
```

As duas variáveis são entradas explícitas do ambiente local, não valores
descobertos automaticamente. Preserve identidade e certificado existente;
nunca coloque certificados, modelos, MSIX ou secrets no Git. Build de pacote
não prova que o candidato foi executado no console.

### 5. Qualificar no Xbox antes de medir velocidade

Primeiro rode o replay offline atual (não toca o console):

```sh
python3 tools/xbox/tests/rkva_kvarn_serial_prefix_probe.py --self-test
```

Somente depois de uma janela de teste autorizada, com baseline/candidato
identificados e sem outra carga concorrente, rode os probes reais existentes.
Os comandos abaixo são **etapas futuras condicionais**, não executadas nesta
campanha. `RKVA_TEST_HOST` é o listener autorizado, nunca um endereço embutido.

```sh
mkdir -p "$P/.build/console"
python3 tools/xbox/tests/rkva_kvarn_serial_prefix_probe.py \
  --execute --host "$RKVA_TEST_HOST" --full \
  --output "$P/.build/console/serial-A.json"
python3 tools/xbox/tests/rkva_kvarn_bulk_replay.py \
  --execute --host "$RKVA_TEST_HOST" \
  --output "$P/.build/console/bulk-A.json"
# Após selecionar e verificar o candidato, com os mesmos requests:
python3 tools/xbox/tests/rkva_kvarn_serial_prefix_probe.py \
  --execute --host "$RKVA_TEST_HOST" --full \
  --output "$P/.build/console/serial-B.json"
python3 tools/xbox/tests/rkva_kvarn_bulk_replay.py \
  --execute --host "$RKVA_TEST_HOST" \
  --reference "$P/.build/console/bulk-A.json" \
  --output "$P/.build/console/bulk-B.json"
```

O bulk replay exige igualdade dos hashes de requests e respostas de atenção;
não afrouxe esse gate. Os probes não cobrem automaticamente toda a matriz de
codecs e novos caminhos. Complete F32, F16, E5M2, E4M3FN; tails 0/128; origens
0/72704/102400; fronteiras 127/128/129 e 255/256/257; splits 448/512 e não
alinhados; rewind/reappend, ring wrap, queries incompletas e contexts longos.
Teste NaN/Inf conforme contrato, finitude e OUT/LSE contra o oráculo original.

### 6. Medir ganho sem confundir causas

- Q_rot: decode nq1. Há aproximadamente 6→12 KiB de loads de Q por grupo no
  protótipo direto, embora cache possa absorver a repetição. Não é medida DRAM.
- WHT: 14→7 barreiras internas, mas +1 KiB LDS. Meça tempo absoluto do reducer
  e impacto na ocupação, não apenas conte barreiras.
- GQA2: LDS 12.384→4.128 bytes, mas triplica grupos e leitura/dequantização KV.
- MQ2: splits de 512 aumentam compartilhamento no modelo; medição depende de
  registradores, ocupação e resíduos. Geometria não é benchmark.

Use A/B/B/A repetido, aquecimento separado, mesmos modelos, prompts, seeds,
contexto, codecs, batch/ubatch e clock/temperatura. Registre tempo de upload,
atenção, reducer, RTT e total, P50/P95 e dispersão; separe prefill de decode.
Confira counters e hashes de PSO em cada rodada, não apenas o nome do perfil.
Depois dos kernels, repita serving/MTP e qualidade com a mesma configuração.
Não promova regressão, efeito dentro do ruído ou melhora de uma fase que piore
o total. Não faça merge/deploy automático ao terminar estes testes.

## Resultado esperado do agente local

Entregue commit testado, source/DXIL hashes, versão e flags do DXC, build MSVC,
hardware e configuração, gates numéricos, requests exatos, ABBA e counters,
falhas/reverts e decisão por candidato. Separe claramente: passou em CPU,
compilou DXIL, compilou pacote, executou no Xbox, melhorou desempenho.

## Atualização local de qualificação (2026-11-06)

Veja [qualification/README.md](qualification/README.md) para o analisador ABBA
conservador, testes e fixtures explicitamente sintéticas. Esta extensão foi
preparada e testada em cópia isolada; a publicação contém apenas artefatos
experimentais, sem alterar Xbox ou produção.
O relato novo de WHT passou corretude serial/bulk, mas não mostrou ganho em
decode/total; baseline mantido e candidato descartado por desempenho. Esse
relato atualiza o estado histórico da tabela acima, sem substituir logs brutos.
Atualização do transcript consultado às 16:38:27 UTC: Q_rot compilou e executou
no Xbox, mas 8/696 respostas bulk diferiram em bytes. Apesar de passar as
tolerâncias seriais, falhou o gate de igualdade bulk e não pode ser promovido.
Causa e ganho final ainda não determinados. Não promover GQA2 por tolerância
CPU: o relato histórico em 129k inclui KLD 1,11 e top1 divergente.


## Nova prioridade experimental: LDS rowlane wave64

Veja [lds-wave64/README.md](lds-wave64/README.md) e o
[patch inerte](lds-wave64/candidate.patch). O experimento é independente de
WHT e Q_rot e não altera shaders de produção. Não é KVarN4.

1. No clone limpo, valide o manifesto e os testes novos:

   ```sh
   cd tools/xbox/experiments/rkva-local-validation-20261106
   sha256sum -c SHA256SUMS-PACKAGE
   python3 -m unittest discover -s qualification -v
   python3 lds-wave64/test_lds.py
   python3 qualification/qualify_abba.py qualification/synthetic-example.json
   ```

   Esperado: 10 testes ABBA + 6 LDS passam; dados sintéticos resultam sempre
   `not_qualified`. GNU cpp/CPU não qualificam DXC, DXIL ou GPU.
2. Compile baseline e candidato isolados com o DXC exato já usado no build
   local (registre versão e hash do executável), perfil `cs_6_0`, entradas e
   defines da tabela LDS. Reconfira o script de produção no commit testado.
   Não acrescente `-Gis` ou outros flags apenas a um lado. Guarde DXIL e
   disassembly em `.build/`, sem substituir headers gerados de produção.
   O script `compile-fixtures-dxil.sh` antigo não cobre LDS; use o roteiro LDS.
3. Compare LDS declarada por entry no DXIL e recursos, não apenas os tamanhos
   declarados em HLSL. Alocação física, registradores e ocupação exigem o
   compilador/profiler do alvo quando disponível; DXIL não os mede. Se não houver redução real de
   LDS, pare esta hipótese e registre o resultado. Não alegue ganho de ocupação
   sem medição. Q4 rowlane pode eliminar arrays de staging já no baseline.
4. Somente após esse gate, prepare PSO/payload experimental separado no clone
   de teste e confirme o gate host `wave_lane_min_ == 64 && wave_lane_max_ == 64`.
   Restrinja a Q4/Q8/F16/SAW rowlane, D256/GQA6/128 threads. Não aplique define
   global, não altere generic/wave32, fallback, KVarN4 ou outros formatos.
   Registre hash do PSO efetivamente selecionado; o define não força wave64.
5. Uma execução real requer janela de console autorizada. Use probes e oráculos
   adequados a cada formato; os probes KVarN4 acima não qualificam Q4/Q8/F16/SAW.
   Confirme OUT/LSE, finitude, igualdade bulk onde exigida, tails, splits,
   origens, rewind/reappend, ring wrap e qualidade/serving/MTP. Pare em qualquer
   falha e mantenha baseline. Não afrouxe tolerâncias para passar.
6. Só depois, execute ABBA repetido e isolado por formato/workload seguindo
   `qualification/README.md`. Não combine LDS com WHT/Q_rot. O analisador é um
   filtro conservador de decode e total, não prova estatística confirmatória
   nem gate genérico de prefill. Não faça install/deploy/merge automático.

Resultado desta publicação: somente testes CPU/estrutura e integridade. DXC,
DXIL LDS, Windows/MSVC, PSO e GPU desse candidato ainda não foram executados.
