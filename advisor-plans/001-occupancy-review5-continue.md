# Revisão 5 e plano de continuação — occupancy position-split

Data: 2026-10-04. Alvo: GOKAYA, 192.168.1.57.
Checkout: `/home/hjotha/beellama.cpp`; branch `feat/occupancy-position-split`.
Planejado contra HEAD `dfd98368f`, inicialmente limpo e alinhado ao acompanhamento remoto.
Sessão OpenCode: `ses_eff002d1affeoqzr9houpYK01p`, tmux `0:0.0`.
Prioridade P1; esforço grande; risco alto se integrar sem corrigir os testes.

## Ordem de execução do usuário

O usuário pediu revisão, um plano para continuar e steer, e acrescentou:
“Verifique tb o plano do steer anterior que vc deu pra ele”.
Continua vigente a ordem de implementar e validar o plano inteiro sem encerrar após etapas parciais.

Retome a execução agora. Este arquivo é um roteiro de reparação e continuação,
não uma substituição ou redução de escopo do plano original:
`/home/hjotha/beellama.cpp/docs/occupancy-placement-plan-20261002.md`.
Leia também o steer anterior real:
`/tmp/occupancy-review4-20261003/review-steer.md`.
O checklist persistente é `docs/occupancy-progress-20261003.md`.

Não responder apenas com reconhecimento, um novo plano, commit ou resumo do que
falta. Ao terminar um gate, atualizar o checklist e executar o próximo.
Depois de compactar, reler estes documentos e continuar pelo primeiro item
aberto. Erro de código, teste vermelho, comando malformado e build demorado são
trabalho implementável, não bloqueio externo nem motivo para finalizar.
Se uma hipótese falhar, corrigir a hipótese e prosseguir; não repetir a mesma
tentativa sem nova evidência. Limite real de provedor/ferramenta exige registrar
erro exato e ponto de retomada; nunca inventar continuidade em background.

## Escopo, limites e estado inicial

- Trabalhar no checkout e branch existentes: BeeLlama não usa worktree.
- Preservar mudanças concorrentes; verificar `git status --short --branch`
  antes de editar. Não resetar nem trocar a branch debaixo de outro agente.
- Não mergear `main` nem promover permanentemente candidato reprovado.
  Continuar todas as implementações, testes e canários autorizados na branch.
- Não trocar o modelo selecionado pelo usuário.
- Produção observada: `llama-server-root.service`, `failed`, `MainPID=0`.
  ExecStart aponta para `build-optimized/bin/llama-server`, que contém candidato.
  O binário original foi sobrescrito: reiniciar esse caminho não restaura o
  original. Preparar baseline conhecido separado, com bibliotecas/config/hashes,
  e chamá-lo explicitamente de baseline reconstruído. Preservar modelos e caches.
- Revisão focada no delta `638c565b8..dfd98368f`, testes e consumidores imediatos.
  Não houve auditoria integral do repositório, campanha 4B/27B ou correção de fonte
  nesta revisão. Os achados estáticos não são apresentados como falhas dinâmicas
  reproduzidas quando isso não ocorreu.

## Reconciliação do steer anterior

| Ordem da revisão 4 | Estado atual conferido | Ação |
|---|---|---|
| Preservar as raízes O e LSE no grafo | Correção preservada; não reabrir a hipótese de O zerado pelo kernel sem evidência | Manter |
| Separar loops de O/LSE, sentinel, finitude e erro absoluto | Implementados parcialmente na matriz; nem todas as formas impõem o gate FP64 | Completar os gates de todas as formas obrigatórias |
| Referência dos mesmos bytes KV e telemetria | Implementação e medições adicionadas | Preservar e completar cobertura das rotas efetivas |
| Body+tail, tail-only, Q>=512, D256/GQA24:4 | Casos adicionados, mas tail continua aberto/recusado | Fechar F1 de verdade |
| Dividir lotes comuns em P antes de prepare | Helper e chamada existem; novo defeito de ponteiros e contrato de falha | Corrigir antes da integração |
| F2 cache/store/tabela/rollback atômico | Não implementado no produto | Implementar |
| F3 integração CUDA+Vulkan com 4B | Só um protótipo CPU+Vulkan; merge com erros concretos | Corrigir protótipo e integrar |
| F4 snapshots/importação XXL | Pendente | Implementar e testar em processo novo |
| F5 campanha 27B/orçamento | Pendente | Executar após gates anteriores |
| F7 baseline/canário/rollback | Baseline separado ainda não comprovado; serviço parado | Preparar e validar |
| Checklist e diagnóstico honesto | Checklist existe, mas contém causalidade incorreta e aceites excessivos | Corrigir junto dos reparos |
| Preservar exit real e não rodar binário após falha de build | Descumprido no último probe: compilação falhou e binário anterior foi executado | Usar dependências sequenciais e exit real |
| Só encerrar com todo escopo autorizado concluído | Descumprido: resposta final às 00:24:55 de 04/10 com F2b–F7 pendentes | Retomar e seguir a sequência completa |

O steer anterior não esqueceu F2–F7 e já proibia usar recusas/known-open como
conclusão. O problema atual é a execução incompleta e novos defeitos no teste,
não falta de um objetivo. Não redesenhar o plano inteiro nem apagar sua história.

## Achados confirmados e correções mínimas

### R1 — P1, introduzido: ponteiros inválidos no splitter

`src/llama-memory-hybrid.cpp:132-158`, função `llama_ubatch_subset`:

```cpp
data->seq_id[i] = data->seq_id_data.data() + data->seq_id_data.size();
for (int32_t s = 0; s < n_ids; ++s) {
    data->seq_id_data.push_back(src.seq_id[t][s]);
}
```

`seq_id_data` está vazio no primeiro token e cresce sem reserva: publica ponteiro
antes da primeira alocação e depois mantém ponteiros invalidáveis por realocação.
Consumidores reais dereferenciam esses ponteiros em
`src/llama-memory-recurrent.cpp:612`, `672`, `712` e `749`.
Os 995 checks não cobrem isso: `check_ubatch_equal` dereferencia IDs apenas no
pass-through (`tests/test-position-split-ubatch.cpp:96-109`); o caso dividido
(`112-145`) verifica tokens/posições/flags, não os IDs das metades.

Correção mínima: preencher integralmente `seq_id_data` primeiro e só então
reconstruir os ponteiros por offsets. Reusar o padrão existente de
`src/llama-batch.cpp:796-811`; não criar um segundo mecanismo de ownership.
Teste vermelho primeiro: dividir `[400,700)` em P=512, verificar todos os IDs
em ambas as metades, após destruir a entrada e após movimentar os ubatches.
Usar sequência não zero para evitar falso sucesso por memória zerada; ASan/UBSan
no alvo CPU ajuda a detectar uso após liberação, sem rebuild CUDA completo.
Confiança alta por inspeção do produtor e consumidores; crash não reproduzido
por esta revisão.

### R2 — P2, introduzido: saída parcial na rejeição

`src/llama-memory-hybrid.h:21-24` promete `out` vazio em qualquer falha.
`divide_ubatches_at_p` faz `out.push_back` durante o loop e retorna false nos
itens seguintes sem limpar o prefixo já publicado (`.cpp:186-255`).
O chamador atual descarta a variável local em caso de erro, portanto isso não
prova mutação do KV; é violação do contrato do helper e lacuna do teste.
Produzir em vetor temporário e mover para `out` apenas ao concluir, ou limpar
em todos os caminhos de erro. Testar lote válido seguido de lote rejeitado.
Não confundir esse reparo com a atomicidade futura de KV + DeltaNet em F2b.

### R3 — P1, introduzido: o diagnóstico de defeito CPU usa layout incorreto

`tests/test-position-split-merge-graph.cpp:65-67,87-88,105-106` interpreta host
como `[token][head][D]`. Os tensores criados em `242-244` são contíguos
`ne={D,n_tokens,n_heads,1}`; o upload linear em `345-349` entrega outro layout.
O mesmo problema aparece para Q quando n_q>1.

O probe `/tmp/opencode/sweep.cpp:18-20,43-48,64-66` repete o erro.
Com NKH=2, V[t,h,d]=t gravado por token/head e lido por head/token produz
EXATAMENTE as médias que foram atribuídas a um kernel que leria metade das chaves:

| NKV | Média pretendida | Média da head 0 no buffer realmente enviado |
|---|---:|---:|
|128|63.5|31.5|
|256|127.5|63.5|
|512|255.5|127.5|
|1024|511.5|255.5|

Esta revisão calculou essas médias diretamente dos índices, sem modificar o
kernel. Logo a evidência apresentada NÃO isola um defeito de cobertura da CPU.
Corrigir fill/upload com strides reais, ou empacotar os vetores na ordem dos
tensores; usar a mesma representação e máscara no oracle FP64. Preservar o
layout de saída correto `{D,n_q_heads,n_q,n_batch}`, confirmado em
`ggml/src/ggml.c:5647-5649`; não consertar o output para combinar com um input errado.
No caso causal o oracle atual também soma todas as chaves (`82-93`), enquanto
o overflow GPU limita `t <= iq + n_kv_ovf - n_q` (`355-359`). Unificar a máscara.
Corrigir a conclusão em `docs/occupancy-progress-20261003.md:414-435`.

### R4 — P1, introduzido: views e broadcasting errados no merge

LSE tem `ne={heads,queries,batch}`, confirmado em `ggml/src/ggml.c:5718-5721`.
O teste usa `lse->nb[1]` como stride de head e `nb[2]` como stride de query
(`tests/test-position-split-merge-graph.cpp:298-306`). Os corretos são nb[0]
e nb[1]. Para H=8/Q=4, 32 floats de LSE são vistos com offsets até 152 floats:
há leitura fora da região lógica do tensor e potencial acesso a memória alheia.

Para pesos contíguos, preferir `ggml_reshape_4d(ctx,e,1,H,Q,B)`, validando
contiguidade e dimensões. Se usar view, obter strides do próprio tensor de pesos,
não de uma suposição sobre o LSE. O denominador também deve ser visto em
`{1,H,Q,B}` antes de dividir O: em `312-313` ele é passado como `{H,Q,B,1}`,
espalhando pesos pelos eixos errados. Testar pesos distintos por head/query.

### R5 — P1/P2, introduzido: vazio, Q4 e backends não estão comprovados

- Em `284-314`, clamp(-inf,-1e30) faz ambas as faixas vazias terem e1=e2=1,
  não peso zero. O LSE merged fica finito (-1e30), violando -inf exato.
  A fórmula `(a+b)/2+abs(a-b)/2` ainda sofre cancelamento quando um lado é esse
  sentinel e o outro é um LSE normal; não é um máximo numericamente confiável.
  Remover o atalho: tratar validade/vazio explicitamente antes das exponenciais,
  reutilizando operações existentes adequadas; nenhuma tolerância relaxada.
- O caso “empty overflow” em `602-626` apenas não cria a segunda faixa; não
  testa uma segunda FA executada com máscara toda -inf, nem ambas vazias.
- Todas as chamadas atuais usam `run_merge(cpu,vk,cpu,...,GGML_TYPE_F16)`.
  A documentação diz overflow Q4 e merge pronto, mas o teste executa F16/F16 e
  merge CPU. Faltam Q4 real e integração CUDA/KVarN + Vulkan/Q4.
- `quantize_range`, linha 177, avança uma linha por `tr->type_size` (um bloco)
  em vez de `ggml_row_size(type,D)` (todos os blocos da linha). Isso corromperá
  o oracle Q4 quando o caso realmente for ativado. Corrigir antes de medir Q4.
- Contar um nó FA abaixo de P não prova ausência de execução/cópias Vulkan:
  conferir backend efetivo e callbacks/telemetria do scheduler no caminho real.

### R6 — P1, introduzido/pendência: teste verde com gates vermelhos

Reexecutado nesta revisão, binário existente:

```text
timeout 90s /home/hjotha/beellama.cpp/build-optimized/bin/test-position-split-merge-graph
merge Q=1 nrmse=2.092e+00 max_abs=4.575e-01
merge Q=2 nrmse=2.766e+00 max_abs=4.581e-01
merge Q=4 nrmse=2.811e+00 max_abs=4.623e-01
merged lse max_abs=5.782e-02
below P nrmse=1.044e+00
position-split-merge-graph: 7201 checks, 0 failures, 6 known-open
exit 0
```

`return g_failures == 0 ? 0 : 1` (`635`) ignora `g_open_issues`.
Há ainda `return 0` quando backend requerido falta (`457-462`).
Na matriz CUDA, `tests/test-kvarn.cpp:6243-6248,6515-6525` separa known-open
e exige apenas `failures.empty()`. Isso serve para diagnóstico, não aceite.
Não apresentar ctest 8/8 como prontidão. Criar gate obrigatório que falhe para
qualquer requisito aberto, backend ausente ou forma obrigatória não executada.
Se preservar modo diagnóstico, separá-lo explicitamente do gate de aprovação.

O teste de merge O não aplica os limites e imprime resultado; uma contagem alta
de finitude/tamanho não substitui NRMSE/max_abs/LSE contra o oracle correto.

### R7 — P2: build falho seguido de execução de binário antigo

Histórico OpenCode às 00:23:57: o compile de `/tmp/opencode/sweep.cpp` falhou
com `fa_out was not declared in this scope`, mas o comando prosseguiu no loop
executando o binário anterior. Esta falha também foi advertida na revisão 4.
Não usar pipeline com `head`/`grep` para decidir se compilou. Compilar, verificar
exit e só então executar. Registrar comando, SHA do fonte, data/hash do binário
e log quando usar um probe fora do CMake. Preferir alvo já registrado no CMake.

### R8 — P1, introduzido: layouts de produção não herdam o gate FP64 prometido

`tests/test-kvarn.cpp:6265-6266` marca Q4/Q8 de produção com
`fp64_reference=false`. Em `6330-6337` calcula `fp64_np`, mas não o compara
numericamente; em `6406-6411` o loop do oracle é desabilitado por `gate_fp64`.
Comparar O/LSE permutado versus não permutado verifica paridade de layout, mas
não prova corretude de ambos. Exigir o gate FP64 para a forma não permutada
equivalente e o mapeamento por strides da forma de produção, incluindo NaN/Inf.

Em `6359-6363`, comparar apenas `isinf` aceita +inf e -inf como equivalentes;
`std::max` com erro NaN também pode manter zero. Em `6439-6442`, linha mascarada
falha só se LSE for finito, aceitando +inf/NaN em vez de exigir -inf. Exigir
`isinf(x) && x<0` e O exatamente zero por linha vazia; linhas válidas devem
ter ambos valores finitos. Criar testes negativos que injetem NaN/+inf/sentinel
no resultado validado para provar que o próprio verificador rejeita.

### R9 — contrato de suporte CUDA precisa de prova reproduzível por forma

Inspeção: `ggml/src/ggml-cuda/fattn-kvarn-dispatch.cu:1075-1108` aceita por
capacidades amplas; `967-972` recusa MMA com LSE quando o subcaminho windowed
não se aplica; `ggml/src/ggml-cuda/fattn.cu:963-966` aborta se o dispatch falha.
Validar que todo supports_op=true para as formas/flags testadas tem executor
O/LSE correspondente, e que a recusa aparece antes do compute quando necessária.
Reusar os predicados existentes; só extrair helper comum se isso eliminar a
divergência, sem redesenhar o dispatcher.

O auxiliar executou o binário avulso `build-optimized/bin/test-position-split-lse-cu-kvarn`
e observou exit 255 com `unsupported KVarN CUDA FlashAttention route` em
`fattn.cu:966`. Ele NÃO está registrado no CMake atual. A reprodução desse
binário prova que ele não é uma validação utilizável, mas não basta para atribuir
o abort ao fonte atual: reconstruir pela receita rastreada e identificar forma,
flags e bibliotecas antes de concluir causa. Não confundir essa observação com
o conhecido abort amplo `fattn.cu:595` de hsk=320.

## Validação adicional nesta revisão: test-kvarn atual

Executado `timeout 90s ./build-optimized/bin/test-kvarn` no checkout, exit 0.
O binário tem mtime 2026-10-03 23:32:32+0200; a libggml-cuda.so tem mtime
23:37:14+0200. O binário avulso cu-kvarn é de 18:10:07+0200, mais antigo.

```text
LSE d256-q4-prod   nrmse=0 maxO=0 maxLSE=0 rows=0 masked=0
LSE d256-q8-prod   nrmse=0 maxO=0 maxLSE=0 rows=0 masked=0
LSE d128-q256-allmsk nrmse=0 attach=nan maxO=0 maxLSE=0 rows=0 masked=6144
LSE d256-q1-tail   nrmse=1.000e+00 maxO=6.451e+04 maxLSE=1.620e+04
LSE d256-q256-tail nrmse=1.680e-03 maxO=4.161e-03 maxLSE=7.084e-03
LSE d256-q512-tail nrmse=1.000e+00 maxO=3.117e+04 maxLSE=2.257e+04
LSE parity GPU: 18 shapes evaluated, 0 failing, 7 known-open
test-kvarn: all tests OK
```

São sete violações em TRÊS formas tail, não sete formas. O gate NRMSE Q256-tail
excedeu 1e-3; Q1/Q512 também excederam os gates de O/LSE em muitas ordens de
grandeza. Resultados tail diferem da rodada anterior, portanto preservar seeds,
repetir caso mínimo e verificar inicialização, limites e lifetime antes de
atribuir todo o desvio à precisão. A revisão não isolou a causa desse tail.
O `attach=nan` é uma métrica de comparação inválida que passou pelo verificador,
não prova sozinho que a saída O do kernel contém NaN. Corrigir finitude de
ambos os lados e classificação mascarada na comparação.

## Sequência obrigatória de execução

### 0. Retomar contexto e preparar rollback observável

Executar no checkout:

```bash
git status --short --branch
git diff --stat dfd98368f..HEAD -- src ggml tests docs
systemctl show llama-server-root.service -p ActiveState -p MainPID -p ExecStart
```

Se houve drift, ler o diff e adaptar este plano preservando mudanças. Não parar
automaticamente por drift. Atualizar o checklist para remover falsas conclusões:
F1 tail aberto, F2a precisa reparo, F3 é protótipo reprovado, F2b/F4/F5/F7 pendentes.
Preparar baseline conhecido em diretório separado de build, preservando o build
incremental candidato; documentar origem e hashes. Não reiniciar produção usando
o candidato por engano. A autonomia operacional já autorizada não autoriza merge
de main nem transformar candidato reprovado em produção permanente.

### 1. Corrigir R1/R2 e tornar o teste de ubatch representativo

Aplicar o padrão de preenchimento-antes-de-ponteiros existente no allocator.
Adicionar testes das metades com seq_id não zero, lifetime, rejeição após prefixo
válido e chamada real de prepare que consome os IDs. Manter proteção da janela
recorrente e o split antes do prepare. Não alargar este reparo para refatoração.

```bash
cmake --build build-optimized --target test-position-split-ubatch -j 4
build-optimized/bin/test-position-split-ubatch
```

Esperado após reparo: exit 0, nenhum ID inválido, nenhuma falha, saída vazia em
rejeição. Fazer o teste novo falhar contra o código anterior antes de aceitar.
Em seguida executar o passo 2; não finalizar.

### 2. Fechar o protótipo de merge e seu oracle (R3–R6)

Ordem pequena de diagnóstico: uma FA CPU com K=0/V=rampa e vários heads;
input/output lidos por strides; oracle FP64 com máscara idêntica; duas faixas
CPU; CPU+Vulkan F16; Q4 com row_size corrigido; weights/denominator por head/query;
uma/ambas faixas inteiramente mascaradas. Só depois culpar kernel por resíduo.
Preservar o contrato de output no domínio V original e corrigir a documentação.

```bash
cmake --build build-optimized --target test-position-split-merge-graph -j 4
build-optimized/bin/test-position-split-merge-graph
```

Esperado: exit 0 SOMENTE com NRMSE<=1e-3, max_abs(O)<=1e-2,
max_abs(LSE finito)<=1e-2, O=0/LSE=-inf exatos em vazios, zero gates abertos.
Exercitar Q1/Q2/Q256 e GQA24:4/D256; não limitar o teste à forma D128/Q1-4.
Adquirir backend pelo identificador correto e falhar se o requerido não existir.

### 3. Concluir F1 CUDA tail e cobertura efetiva

Preservar correções úteis de `a8ad7084c` e fechar body+tail e tail-only, com O e
LSE finais e a mesma representação KV/tail na referência FP64. Auditar offsets,
metadata, domínio V, buffer capacity/lifetime e strides com casos mínimos antes
de expandir a matriz. Recusa temporária protege, mas não entrega rota obrigatória.
Gates não podem ser desativados para layout de produção; confirmar rota real de
decode/windowed/fallback e forma Q>=512, não apenas imprimir contadores.

```bash
cmake --build build-optimized --target test-kvarn test-position-split-lse-cu test-position-split-lse-vk test-position-split-lse-sched test-position-split-lse-alloc -j 4
build-optimized/bin/test-kvarn
ctest --test-dir build-optimized -R position-split --output-on-failure
```

Esperado: gates obrigatórios todos exercitados e aprovados, zero requisitos em
known-open; ctest sozinho ainda não comprova isso no código inicial. O arquivo
`build-optimized/bin/test-position-split-lse-cu-kvarn` existe, mas não é alvo
registrado no CMake atual: não tratá-lo como artefato reproduzível sem sua receita.
Criar/reusar opção mínima de filtro em `test-kvarn` se acelerar diagnóstico;
registrar a receita, sem multiplicar probes não rastreados.

Regressões depois de corrigir os kernels: clone/allocator/scheduler, test-kvarn,
FA CPU/Vulkan/CUDA com backend exato. Isolar falha preexistente pela mesma forma
em baseline conhecido, sem rotular requisito obrigatório como fora de escopo.

### 4. Concluir F2b: cache por posição e transação KV + DeltaNet

Arquivos: `src/llama-kv-cache-kvarn.*`, cache padrão/mixed,
`src/llama-memory-hybrid.*`, `src/llama-context.cpp`, `src/llama-cparams.h`,
`src/llama-model.cpp`, stores em `src/llama-graph.cpp` e testes mixed existentes.

Implementar duas faixas por camada full, uma tabela lógica; local KVarN4 CUDA
capacidade P; overflow Q4 Vulkan capacidade C-P; tail/stage somente local.
P fixo alinhado a 256, uma sequência, sem SWA/context shift, MTP0.
Prepare/seq_rm/truncate/rollback validam todas as partes antes de publicar;
falha da segunda faixa deixa KV/tabela/DeltaNet iguais ao estado anterior.
Fora da reserva exata usar checkpoint/replay existente ou rejeição sem mutação.
Não anunciar “placement enabled” enquanto P só dividir ubatches.

Ampliar os testes existentes e registrá-los no CMake:

```bash
cmake --build build-optimized -j 4
ctest --test-dir build-optimized -R 'position-split|kv-mixed|state-convert' --output-on-failure
```

Esperado: casos reais P-129/P-128/P-1/P/P+1/P+128, remoção/rollback,
prepare injetado falhando, janela protegida, embd_all/prefill/decode, sem mutação
parcial. Comparar KV lógico e estado recorrente, não só retorno do helper.

### 5. Concluir F3: integração real 4B

Ligar cache/store e duas FAs em `src/llama-graph.cpp::build_attn` e
`src/models/qwen35.cpp::build_layer_attn`. Abaixo de P, caminho CUDA atual sem
execução/cópias de atenção Vulkan. Acima, CUDA/KVarN + Vulkan/Q4 + merge CUDA,
RoPE/posições absolutas, máscara overflow offset P, gate/wo CUDA.

Modelo existente: `/home/hjotha/models/Qwen3.5-4B-Q4_K_M.gguf`.
Executar P=512 e 1024, gerar de P-1 a P+128 e prefill cruzando P.
Registrar teste de integração CMake como `test-position-split-integration-4b`
(novo teste a implementar; esse nome NÃO existe hoje). Depois:

```bash
ctest --test-dir build-optimized -R '^test-position-split-integration-4b$' --output-on-failure --no-tests=error
```

Exigir próximos logits e DeltaNet contra referência do mesmo KV por faixa:
KL FP64 média<=1e-4 nat/token e p99<=1e-3, além dos gates O/LSE anteriores.
Teste prova backend/cópias executados, não mera contagem de nós.

### 6. Concluir F4: snapshots, importação XXL e disco

Arquivos: `src/llama-kv-mixed-state.*`, serialização KVarN, mixed-io/handoff,
`tools/server/server-mixed-kv-handoff.*`, `server-context.cpp`, índice de disco.
Confirmar versão livre; envelope position_split com P/C/ocupação/modelo/RoPE,
descritores (layer,range), limites/dimensões/strides/tamanho/integridade.
Writer/reader/spool/handoff/índice precisam concordar. Manter v1 legado; rejeitar
versão desconhecida e estado inválido antes de mutação. Publicação atômica em disco.

Estender `test-kv-mixed-state`, stream/io/placement/handoff e state-convert.
Adicionar e registrar `test-position-split-disk-integration` (novo): processo novo,
somente snapshot XXL em disco, sem residente, provar importação/hit/tokens
reutilizados e ausência de prefill do prefixo. P diferente vira miss; corrupção,
truncamento, faixa duplicada, capacidade excessiva e falha de restore preservam
destino e snapshot original. Validar sufixo divergente/DeltaNet perto de P.
Split->CUDA só se n<=min(P,capacidade); overflow ocupado rejeita.

```bash
ctest --test-dir build-optimized -R 'kv-mixed|state-convert|position-split-disk-integration' --output-on-failure --no-tests=error
```

Esperado: testes registrados e aprovados, rollback exato, TMPDIR em disco.

### 7. Concluir F5: orçamento e campanha 27B

Modelo observado no plano: `/home/hjotha/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf`.
Verificar arquivo/hash antes da campanha; MTP0. Escolher maior P alinhado <=102400
que caiba com pesos+KV+stage/tail+rollback+DeltaNet+grafos pp/tg+O/LSE+scratch+
cópias+restore simultâneos. Medir execução/reserva, não só idle após carregar.
Folga CUDA>=max(512MiB,5% VRAM), MemAvailable>=2GiB, sem swap/OOM, picos Vulkan/GTT.

Reutilizar harness de benchmark existente após localizá-lo; registrar comando
exato antes de executar, não inventar opções de CLI. Baseline/candidato com
hash de modelo/corpus/prompt, seed, sampling, -b/-ub, hardware/config idênticos.
Um aquecimento descartado + cinco A/B alternados, separando frio/quente/restore.

- Abaixo de P (incluindo ~96k quando couber): regressão<=5% prefill/TTFT e p95
  decode versus XXL.
- 105000/115000/130560: prefill/TTFT>=10% menor versus XXXL por camada;
  regressão p95 decode<=5%. P alterado: incluir P+256.
- Mesma referência F16: KL média<=baseline+0.001 nat/token,
  perplexidade<=1.01*baseline por comprimento; não confundir com gate estrutural.
- Ocupar C=131072; excesso rejeita limpo; gerar 130560+512; dez ciclos de
  restore/divergência/geração, memória retida<=64MiB/backend acima do primeiro
  ciclo aquecido, zero corrupção/OOM/erro driver.

Gate reprovado: corrigir e repetir o cenário afetado, preservando limiar original.
F6 concorrência é OPCIONAL; não impede entrega sequencial aprovada. Só abrir se
F0–F5 verdes e ganho mediano>=5% ponta a ponta em dois comprimentos acima de P.

### 8. Concluir F7 autorizado e entregar evidência

Separar artefatos candidato/baseline, registrar binários+bibliotecas+config+hashes,
comandos, canário isolado e rollback. Restaurar baseline conhecido autorizado,
verificar processo/bibliotecas/config efetivos, health, métricas e prompt real.
Não chamá-lo de binário original perdido. Não mergear main nem promover
permanentemente candidato sem a autorização correspondente.

## Critério único de término

Todos os itens obrigatórios implementados e validados, nenhum gate estrutural
aberto, 4B integrado, snapshots/disco/DeltaNet aprovados, campanha 27B dentro dos
limites, baseline e canário/rollback verificáveis. A promoção ainda não autorizada
fica separada desse escopo. Não terminar novamente com “faltam F2b/F3/F4/F5”.

Guardar por item: estado, commit/diff, comando, log, exit, métricas e próximo
passo. Commit/build/ctest isolado não é critério de conclusão. Não flexibilizar
tolerância, trocar caso difícil por recusa nem rotular defeito implementável de
bloqueio externo. Executar uma compilação incremental por vez, sem sobrescrever
baseline; alterações em testes C++ não exigem recompilar todas as variantes CUDA.

## Manutenção e achados rejeitados

- Rejeitado como causa comprovada: “CPU FA lê só metade das chaves”. O input do
  probe fornece exatamente as médias observadas por erro de layout.
- Rejeitado como aceite: “7201 checks/0 falhas” e “ctest 8/8”. Gates numéricos
  obrigatórios não participam do exit atual.
- Não reabrir consertos de O/LSE anteriores sem reprodução; preservar raízes do
  grafo, saída LSE explícita, correções de clone/allocator e precisão Vulkan.
- Mudanças futuras no layout O/LSE/tail devem atualizar views, oracle, telemetria
  e integração conjuntamente. Usar padrões existentes antes de criar abstrações.
