# MQ2/GQA2: investigação CPU, reprodução de shaders e plano para localizar a divergência

**Data da evidência: 30/10/2026. Estado: diagnóstico sintético concluído; divergência original do modelo no Xbox não reproduzida; nenhuma correção de produção demonstrada.**

Este documento transfere os resultados para o próximo agente de engenharia. Não autoriza executar trabalho em máquinas, publicar alterações ou relaxar gates: essas ações continuam dependentes do escopo concedido pelo proprietário. As investigações descritas ocorreram em Linux x86-64/CPU, sem executar ou alterar o Xbox ou máquinas do proprietário.

## 1. Resposta curta e prioridade

Há duas questões diferentes:

1. **MQ2 cap32/cap128:** mudar o particionamento em splits muda a associação das operações FP32. O simulador e a execução de HLSL adaptado mostraram pequenas diferenças numéricas com entradas sintéticas iguais. Isso não prova que o arredondamento observado explique a falha de qualidade do modelo.
2. **GQA2:** trocar seis heads locais por três grupos de dois heads não produziu divergência nos modelos CPU nem na execução lógica128 com o mesmo cap64. Isso não absolve o caminho real de duas waves64, o compilador/driver ou o estado real da aplicação.

**Próximo passo de maior valor:** localizar a primeira operação divergente do caso original com teacher forcing e capturar suas entradas físicas, parciais e saída. Não começar alterando tolerâncias, forçando igualdade artificial no simulador ou promovendo o candidato NaN como fix do Xbox.

## 2. Proveniência fixada

Repositório principal: `https://github.com/hjotha/beellama.cpp`. Os hashes abaixo são os snapshots efetivamente analisados, não uma afirmação sobre a ponta atual das branches.

| Referência | Commit |
|---|---|
| main observada no início | `8d181aeb40c9be2af85df15a7b0065c3e3d244fa` |
| MQ2 baseline/cap64 | `e8a1380e2485ae27220001337ed5982122fde803` |
| `perf/mq2-split-20261030` / cap32 | `f3fb8a933c9bf800434f03125cea1c9abf99f157` |
| `perf/mq2-cap128-20261030` / cap128 | `4a9eea29ef49ecaab3e72a476a1cd1a576a79efa` |
| `perf/rkva-gqa2-prefill-20261027` | `582cc8d8e70216e0ba697e735fbe3c6817704f9a` |
| Base da integração GQA2 | `e3fedf8d17a8224fc4b4669533fee45aff7427d6` |

As três variantes MQ2 têm o mesmo HLSL; o cap altera o particionamento no host. Fontes relevantes:

- `tools/xbox/shaders/rkva/rkva_range_attn_kvarn4_mq.hlsl`
- `tools/xbox/shaders/rkva/rkva_range_attn_kvarn4_mq_gqa2.hlsl`
- `tools/xbox/shaders/rkva/rkva_range_reduce_opt.hlsl`
- `tools/xbox/uwp/rkva/remote-attn-engine.cpp` e políticas de dispatch/split relacionadas.
- `src/llama-context.cpp`, hooks `F4_NODE_DUMP*` existentes no snapshot main analisado.

O repositório separado do aplicativo é `hjotha/xllama-qwen35-xbox`, não `hjotha/xllama`. Não inferir ausência de branches a partir do nome errado. Para este diagnóstico os shaders foram obtidos de `tools/xbox` no fork beellama.cpp.

## 3. Investigação inicial: simuladores, layout e DXIL

### 3.1 MQ2

O oracle anterior `rkva_kvarn4_mq_split_model.py` é útil matematicamente, mas não reproduz a ordem de execução MQ: usa residual antes de records, lotes de oito tokens e somas/BLAS NumPy. O shader processa records primeiro, dois meios-tiles de quatro tokens com rescale próprio, depois residual não-tail e tail.

Foi preparado um simulador FP32 separado com essa ordem, propriedade por wave, janelas causais, body rotacionado/tail original, merge e reducer. Ele encontrou diferenças de aproximadamente `1e-9` entre caps com Q padrão; testes mais agudos chegaram a aproximadamente `5.36e-7`. O controle F64 reduziu as diferenças a aproximadamente `3.94e-18`. A cobertura simulada de linhas foi exatamente uma vez e os gates existentes do oracle permaneceram inalterados.

Limites: um head isolado por caso, ordem de soma escolhida, funções NumPy em vez das intrínsecas RDNA2, dados sintéticos e fontes compartilhadas com partes do oracle. O teste não certifica loads físicos, LDS entre heads, FMA/FTZ/transcendentais ou qualidade do modelo.

Arquivos diagnósticos preparados nessa etapa, não presumir que já existam em qualquer branch remota:

- `tools/xbox/tests/rkva_mq2_fp32_repro.py`
- `tools/xbox/tests/test_rkva_mq2_fp32_repro.py`
- `tools/xbox/docs/rkva-mq2-fp32-diagnostic.md`

### 3.2 GQA2

Auditoria do mapeamento: com `DispatchX=3*n_head_kv`, `kv_head=x/3`, `head_base=2*(x%3)` e head final `kv_head*6+head_base+g=2*x+g`. Foram examinados Q loads, stores de parciais/saídas, NaN paths, LDS e scratch WHT. O candidato mantém GQA global6, especializa heads locais2 e exclui INT8/QT4 nessa rota.

O modelo FP32 comparou seis heads versus três grupos de dois, exigindo bits iguais em scores, estados acumulados, parciais e WHT. Casos longos 102400/112640/130048 passaram com cap64. Controle negativo que mudou associação de dot4 detectou uma diferença de 1 ULP, comprovando sensibilidade do teste, não comportamento real da GPU.

Arquivos preparados: `tools/xbox/tests/test_rkva_mq_gqa2_fp32.py`, `tools/xbox/tests/rkva_gqa2_dxil_compare.py` e relatório correspondente em `docs/agent-reports/`.

### 3.3 Inspeção do DXIL realmente embutido

Os blobs foram extraídos e desmontados, não recompilados para essa comparação. Ambos declaram `dxcoob 1.8.2502.11 (239921522)`, `cs_6_0`, 128 threads por grupo.

| Blob | Bytes | SHA256 |
|---|---:|---|
| MQ2 | 118452 | `9d81c3beba4e36756c89c6bc70c5807cedb990be67fba683fc714ff7a1c5b431` |
| GQA2 | 47924 | `b26dd85cb5512225dcb10333fc53a7a7b52676ccac2d480ed237fdcf92565a99` |
| Reducer KVarN4 | — | `82474320bf2dc8d4d8ec87a021dba342a82e934c00543dcd5511106d586c94ed` |

Blocos por head de record half-tile, residual e tail coincidiram após normalização de nomes SSA, preservando opcodes, flags, constantes e dependências locais. Isso não valida correspondência global de endereços/heads/rows. `WaveActiveSum` e dot4 continuam intrínsecas; a ordem final na GPU não é determinada pelo dump.

Detalhes importantes para qualquer replay:

- WHT compilado usa `(x +/- y) * 0.0624999962747097`, bits F32 `0x3d7fffff`, não exatamente 1/16.
- Reducer normaliza numerador pelo denominador **antes** do WHT.
- Exp usa exp2 com fator F32 `0x3fb8aa3b`; log usa log2 com fator `0x3f317218`.
- Ausência de FMad explícito não prova ausência de contração no backend. Pressão de registradores/driver é hipótese, não causa demonstrada.

## 4. Execução de HLSL em CPU

### Ambiente e barreira real

Linux x86-64, AMD EPYC 9V74, Mesa Lavapipe 25.0.7, LLVM 19.1.7, DXC 1.9.2609.5, g++ 14.2.0, Python 3.12.14 e NumPy 2.3.5. Dependências foram usadas isoladamente, sem mudança de sistema.

O shader original requer subgroup64..128. Lavapipe nativo padrão expôs subgroup8 e acionou corretamente o guard, produzindo poison NaN. `LP_NATIVE_VECTOR_WIDTH=2048` anunciou64 mas falhou no smoke: índices repetidos0..15 e soma136 em vez de2080. **Esse override foi descartado; não é suporte wave64.**

Foi criado adaptador explicitamente experimental de **uma wave lógica128**, via shared memory e barreiras de workgroup. Corpo HLSL original incluído por wrappers; APIs de subgroup substituídas. O guard não foi removido. Escopo não-INT8/QT2. Os pontos de coletivas foram auditados quanto à uniformidade.

A versão otimizada mantém a árvore de soma stride64→1, mas thread0 calcula a árvore entre três barreiras. Foi comparada bit a bit à implementação anterior em fixtures pequenas. A configuração final usa subgroup nativo4 (`LP_NATIVE_VECTOR_WIDTH=128` bits), `GALLIVM_PERF=nopt` e quatro threads. Esses parâmetros são locais ao processo.

Tentativas anteriores de MQ terminaram SIGKILL/137 perto de 4.9 GiB RSS. OOM é suspeita, não diagnóstico comprovado. Outputs antigos herdados de diretórios com poison foram excluídos; somente execuções terminadas com métricas e hashes entram nos resultados abaixo.

### Matriz final: 12 execuções válidas

Cada tamanho abaixo executou MQ cap32, MQ cap64, MQ cap128 e GQA cap64. As contagens são **linhas KV ativas sintéticas**, não contexto total do modelo/serving.

| KV ativas | MQ64 vs GQA64 | Máximo ΔO cap32 vs64 | Máximo ΔO cap128 vs64 |
|---:|---|---:|---:|
| 102400 | Parciais/O/LSE bit idênticos | 9.8953e-9 | 6.7521e-9 |
| 112640 | Parciais/O/LSE bit idênticos | 9.1968e-9 | 7.9162e-9 |
| 130048 | Parciais/O/LSE bit idênticos | 9.5897e-9 | 7.4506e-9 |

Máximo ΔLSE entre caps: `1.9073486e-6`. Contra oracle FP64 independente dos bytes: máximo absoluto O `1.0156559e-8`; LSE `1.7368958e-6`. Todos outputs finitos, buffers somente leitura intactos e canários de fim de alocação preservados. Canários não provam ausência de leitura fora da faixa.

Geometria: origin128, nq3, 24 Q heads/4 KV heads, dimensão256, janelas `[active-1,active,active]`, tail128 em ring256, query fantasma no último tile QT2. Layout físico de KVarN4: record8960 bytes, 8192 bytes de nibbles mais três conjuntos128F16; stage384rowsF16. Q original/rotacionada, positions e cbuffer48 bytes foram fornecidos explicitamente.

Controles: smoke nativo e lógico; igualdade adaptador antigo/novo em256/2048; repetição GQA2048; 1 versus4 threads; edge257 com origin0/tail0/janelas0,256,257 e split vazio; saída direta256 com split_k1. Na rota direta MQ/GQA foram idênticos, erro O máximo `1.18905e-7` e LSE `6.97596e-7`.

**Limitação principal:** uma wave lógica128 por workgroup não exerce o caminho de duas waves64 e seu merge. Não substituir isso por uma alegação de teste wave64/Xbox. Implementar duas waves lógicas com barreiras de workgroup exige nova análise: coletivas por wave sob controle divergente podem causar deadlock ou alterar semântica.

## 5. Candidato NaN: defeito separado de portabilidade SPIR-V

Injetar NaN em Q/Qrot do head0/query0 revelou que a rota SPIR-V adaptada original convertia estado inválido em vazio: denominador0, O0 e LSE−inf. Outros heads continuaram iguais.

Microshader mostrou soma preservando NaN e votação detectando inválido, porém `x != 0` compilou para `OpFOrdNotEqual`, falso para NaN. O merge descartava a wave inválida. O DXIL inspecionado usa `fcmp fast une`, unordered. Portanto o comportamento não demonstra o mesmo bug no Xbox.

`candidate_unordered_merge.patch` altera cinco gates em uma cópia isolada: `x != 0` → `!(x == 0)`, nas condições de denominador/peso do merge e LSE direto. Nesta compilação isso gerou comparações unordered.

Validação realizada:

- Parciais finitos MQ/GQA2048 bit idênticos ao original.
- Saídas diretas finitas256 bit idênticas ao original.
- NaN propagado em parciais, saída direta e parciais→reducer; outros heads preservados.

**Não realizada:** matriz longa com o candidato; reducer dos controles finitos2048 do candidato; validação DXIL/Xbox. O patch não foi aplicado ao código de produção. Deve continuar separado até evidência causal e testes suficientes; não usá-lo como resposta automática à divergência original.

## 6. Artefatos e como reproduzir

Pacote técnico portátil: `astra-mq2-gqa2-cpu-replay-20261030.zip`, 33126 bytes, SHA256 `960e0000461c0f6807c75a0d48a7205b6544c744ebe4a5c03b0d31edbb16a4f0`. O pacote contém fontes, metadados, resultados compactos e documentação, sem runtimes, modelos ou buffers KV grandes. Os fontes e resultados compactos acompanham este relatório no subdiretório [`replay/`](replay/). Não presumir que os scripts existam nas branches históricas anteriores a este handoff.

Arquivos-chave em `replay/` (também presentes no pacote):

- `PORTABLE.md`, `RESULTADOS.md`, `CAPTURA_NECESSARIA.md`.
- `replay.cpp`, `build_portable.py`, `long_fixtures.py`, `run_case.py`, `long_check.py`, `summarize_matrix.py`.
- `logical128.hlsl`, `logical128-3barrier.hlsl`, smokes e `nan-semantics.hlsl`.
- `matrix-results.json`, `mq-cap-source-provenance.json`, hashes de identidade MQ/reducer.
- `direct-result.json`, `negative-result.json`, `candidate-result.json`, `candidate_unordered_merge.patch`.

Com dependências presentes e clones independentes dos commits fixados (respeitar AGENTS.md; não criar worktrees se proibido):

```sh
# A partir da raiz do repositório do handoff:
cd docs/diagnostics/mq2-gqa2-20261030/replay
# Substituir caminhos pelas dependências locais.
python build_portable.py --mq-repo /caminho/mq2 --gqa-repo /caminho/gqa2 \
  --dxc /caminho/dxc --vulkan-include /caminho/include
export VK_DRIVER_FILES=/caminho/lvp_icd.json
export LD_LIBRARY_PATH=/caminho/bibliotecas
python long_fixtures.py fixture2048 --active 2048 --windows 2047,2048,2048
REPLAY_THREADS=4 python run_case.py fixture2048 64 mq_cap_source-3barrier mq64
REPLAY_THREADS=4 python run_case.py fixture2048 64 mq_gqa2-3barrier gqa64
```

O builder verifica SHA256 dos HLSL e recusa fonte não fixada. Cada execução requer diretório de saída novo. Conferir smokes antes de interpretar resultados em outro driver. Os comandos são roteiro, não registro de novas execuções após este relatório. Tempos/RSS incluem compilação e emulação; não são TPS do Xbox.

## 7. Plano objetivo para resolver o problema original

### A. Fixar e reproduzir o gate, sem mudar sua exigência

1. Registrar hashes do modelo e cache/prefix snapshot, tokens exatos do prompt e teacher forcing, seed, contexto, B/UB, modo, flags e política de split.
2. Fixar commits, revisão do app, PSO/DXIL exato, compiler flags, GPU/driver e wave size observada. Não confundir versão do DXC com driver.
3. Executar baseline e candidato com tokens forçados idênticos desde step0. Comparar logits completos desde o início, não só top1 do step3. Repetir cada configuração para identificar não determinismo.
4. Preservar artefatos originais imutáveis e declarar FAIL, erro de execução e teste não realizado separadamente. Registrar margens top1/top2, KL, max/RMS/erro relativo e thresholds já existentes.

### B. Encontrar a primeira operação, não o primeiro sintoma final

1. Comparar hashes/saídas por camada no primeiro passo que diverge. Os hooks `F4_NODE_DUMP`, `F4_NODE_DUMP_ROLES`, `F4_NODE_DUMP_LAYERS`, `F4_NODE_DUMP_AUTOTAG`, `F4_NODE_DUMP_FULL` e `F4_NODE_DUMP_FULL_EVALS` existem no snapshot examinado de `llama-context.cpp`; confirmar semântica atual antes de usar. O default observado seleciona layers2,3: não assumir cobertura integral. `nobuffer` não é igualdade.
2. Na primeira chamada de atenção diferente, guardar **o mesmo estado físico**: Q/Qrot F32, posições absolutas, cbuffer/descritores, offsets/tamanhos/hashes e bytes referenciados de KVarN4/stage/tail. Não requantizar nem reconstruir um prefixo apenas semelhante.
3. Registrar n_query, active_count, n_head/n_head_kv/head_dim, kq_scale, origin, split_k, split_rows, direct_output, records_bytes e precision_tail_tokens, além da geometria do dispatch.
4. Capturar parciais depois de pass1 e antes do reducer; depois O/LSE final. Para nq2×24heads×64splits, parciais são aproximadamente3 MiB e O/LSE48 KiB. Preferir uma operação isolada a dumps de todos os caches do modelo.

### C. Árvore de decisão da investigação

- **Entradas já diferentes:** recuar à camada/operação anterior; não atribuir ao kernel de atenção local.
- **Entradas iguais, parciais diferentes:** localizar primeiro `[split][query][head]` e comparar row coverage/row-kind, endereços/heads, QK antes/depois de wave sum, exp, max/den online, body/tail, estado por wave e merge. Priorizar explicitamente duas waves64 e fronteiras causal/split/stage/tail.
- **Parciais iguais, O/LSE diferentes:** isolar reducer, ordem de normalização, WHT, binding/lifetime/sincronização.
- **O/LSE iguais, logits diferentes:** avançar a operadores downstream/estado recorrente; não modificar atenção para esconder o problema.
- **Variação entre repetições:** investigar corrida, sincronização, lifetime e leitura não inicializada antes de explicar tudo por arredondamento.

Uma redução de modelo para um replay da chamada com bytes capturados permite avançar sem repetir todo o LLM. Para hipótese de driver/registradores, comparar disassembly/ISA ou captura GPU quando disponível, mantendo compilação e entrada fixadas.

### D. Corrigir somente uma causa sustentada e testar regressão

1. Escrever um caso mínimo que falhe antes do patch, idealmente a captura real reduzida, e verificar o ponto causal.
2. Fazer alteração mínima e separar correção de indexação/sincronização, mudança de estabilidade numérica e candidato de portabilidade NaN. Não misturar os três em um único resultado.
3. Reexecutar exatamente o caso falho, os controles finitos/vazios/NaN, tail0/128, origin0/>0, queries ímpares/phantom, splits vazios, saída direta e caminho duas waves64. Cobertura de wave128 é complementar.
4. Executar a matriz longa novamente se o patch afetar merge/reducer; preservar comparações baseline/candidato e hashes. Para o candidato SPIR-V, esta etapa ainda falta.
5. Retornar ao gate completo original de16 passos e aos contextos-alvo com mesmos tokens e critérios. Só então medir prefill/decode no hardware e avaliar regressão de desempenho.
6. Entregar diff, commit, evidência antes/depois, resultados e limites. Não declarar o problema resolvido apenas porque o cap64 sintético coincide ou porque os testes CPU passam. Não promover/deployar sem escopo autorizado.

## 8. Definição de pronto para o próximo agente

Uma solução convincente tem: primeira divergência localizada, entrada reproduzível, mecanismo causal explicado, teste que falhava antes e passa depois, gate real aprovado sem tolerância relaxada, e desempenho medido separadamente. Se a captura real continuar indisponível, entregar essa falta explicitamente; o resultado máximo desta rodada permanece diagnóstico sintético e candidato de portabilidade isolado.
