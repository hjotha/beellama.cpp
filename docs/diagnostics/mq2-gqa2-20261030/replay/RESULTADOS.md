# Resultados do executor CPU — 30/10/2026

## Conclusão

O corpo dos shaders foi executado em CPU através de HLSL → SPIR-V → Lavapipe, usando um **adaptador explícito de wave lógica 128**. Nas três fixtures longas, MQ2 cap64 e GQA2 cap64 produziram parciais, O e LSE idênticos bit a bit. Alterar o cap de MQ2 produziu pequenas diferenças FP32 reais neste executor.

**Isso não reproduz nem valida o caminho de duas waves64 e sua mesclagem no Xbox. Não foi reproduzido o gate original de 16 passos do modelo, e nenhuma correção desse problema foi demonstrada ou aplicada.**

## Matriz com entradas finitas

Todas as 12 execuções abaixo terminaram com outputs finitos, canários intactos e buffers de entrada preservados. Os números são linhas KV ativas sintéticas, não contexto total do modelo.

| Linhas KV ativas | MQ64 vs GQA64: bits diferentes | Máximo ΔO cap32 vs64 | Máximo ΔO cap128 vs64 |
|---:|---:|---:|---:|
| 102400 | 0, incluindo parciais | 9,8953e-9 | 6,7521e-9 |
| 112640 | 0, incluindo parciais | 9,1968e-9 | 7,9162e-9 |
| 130048 | 0, incluindo parciais | 9,5897e-9 | 7,4506e-9 |

Máximo ΔLSE entre caps: **1,90735e-6**. Contra o oracle FP64 independente de bytes: máximo absoluto de O **1,01566e-8** e de LSE **1,73690e-6** nas 12 execuções. Esses valores não são um gate de qualidade do LLM.

Geometria comum: `origin=128`, `n_query=3`, janelas causais `[active-1, active, active]`, tail de 128 tokens em ring de 256, 24 heads Q, 4 heads KV e dimensão 256. Queries originais/rotacionadas, registros KVarN4, stage, tail, posições, cbuffer e reducer usam seus layouts reais. Há uma query fantasma no último tile QT2. A semente, os hashes, os splits, os tempos e a memória por execução constam nos JSON.

Configuração final comum: Lavapipe nativo com 4 lanes (`LP_NATIVE_VECTOR_WIDTH=128` bits), wave lógica 128 implementada por memória compartilhada, adaptador de três barreiras, LLVM `nopt` e quatro threads de trabalho. O cap muda apenas o particionamento host. Os três commits MQ2 têm o mesmo HLSL; o baseline de seis heads do ramo GQA gera o mesmo SPIR-V que o cap-source. O reducer WHT também gera binário idêntico entre as fontes dos dois ramos.

## Controles adicionais

- Smokes nativo e lógico verificaram lanes, soma e votação. O adaptador lógico somou 1…128 em 8256 e detectou condição verdadeira no lane 127.
- Os adaptadores antigo e de três barreiras produziram resultados bit a bit iguais em 256 e 2048 linhas KV, em ambas as arms.
- A repetição GQA de 2048 linhas no mesmo setting foi bit a bit idêntica. A troca de uma para quatro threads preservou os resultados MQ/GQA.
- O caso de 257 linhas, `origin=0`, tail desativado, janelas `[0,256,257]` e split extra vazio verificou stage, causalidade, query fantasma, O zero/LSE −infinito para query vazia e denominador zero no split vazio.
- A rota direta (`split_k=1`, `direct_output=1`) foi executada com 256 linhas: MQ/GQA bit idênticos, máximo erro O de 1,18905e-7 e LSE de 6,97596e-7 contra o oracle.

## Teste negativo: falha original e candidato separado

A injeção de NaN em 512 valores F32 de Q/Qrot do head 0/query 0 **falhou na expectativa de propagação** da rota SPIR-V original adaptada: o head virou estado vazio, com O zero e LSE −infinito. Os demais heads permaneceram bit idênticos. A falha está preservada em `negative-result.json`; ela não foi contada como PASS.

Um microshader isolou a causa nesta rota: a soma preserva NaN e a votação reconhece estado inválido, mas `x != 0` foi compilado como `OpFOrdNotEqual`, retornando falso para NaN. Isso permite descartar a wave marcada inválida no merge. O DXIL previamente inspecionado usa uma comparação unordered (`fcmp fast une`), portanto este resultado não comprova o mesmo defeito no Xbox.

Foi preparado **somente um candidato isolado de portabilidade**, em `candidate_unordered_merge.patch`: cinco gates de merge/LSE usam `!(x == 0)`, que nesta compilação gerou comparações unordered. Ele preservou identidade bit a bit nos **parciais** dos controles finitos MQ/GQA de 2048 linhas e nas **saídas diretas** de 256 linhas. Não foi reexecutada a matriz longa com o candidato, nem o reducer dos controles finitos de 2048 linhas do candidato. No negativo, passou a propagar NaN em O e LSE, tanto com reducer quanto na rota direta, mantendo os demais heads inalterados. Não foi aplicado ao fork, comitado, publicado ou validado no Xbox; não é apresentado como correção da divergência original do modelo.

## Limites e próximos dados necessários

A wave lógica 128 contém apenas uma wave por workgroup. Ela não exercita a divisão/mesclagem de duas waves64, a geração DXIL usada no console, os efeitos do compilador/driver Xbox ou os tensores reais que provocaram a divergência. `CAPTURA_NECESSARIA.md` descreve a captura mínima ainda ausente.

Tempos/RSS documentam o executor CPU e incluem custos de compilação/execução; **não são benchmarks do Xbox**. Nenhuma alteração foi feita em máquinas do usuário, no aplicativo de produção ou no repositório original.
