# Revisão crítica do protótipo wavefront da Fase 7

Parecer: **homologação da Fase 7 pendente**. O escalonamento abstrato por antidiagonais está correto para a grade de dependências descrita. A implementação atual é um protótipo CPU de planejamento e atenção simplificada, sem integração ao runtime de inferência, e contém falhas de causalidade e de preservação da janela recorrente. A conclusão de testes com zero falhas não estabelece as garantias de serving declaradas.

Foram examinados os arquivos atuais do checkout, com alterações locais, na branch `docs/hybrid-attention-master-review`, base `aaa63758a968577c593654d27b80de748dd15c6a`. Não houve alteração dos fontes de implementação, commit, deploy ou benchmark GPU nesta revisão.

## Achados que bloqueiam a homologação

### P1 A janela recorrente protegida pode ser dividida com retorno de sucesso

Referências: [particionamento](../../src/llama-wavefront.cpp#L115), especialmente o cálculo de `rec_start` na linha134 e a limitação de `cand` nas linhas139–152.

O algoritmo evita invadir o início da janela protegida, mas não garante que o restante caiba num único tile. Há contraexemplos com posições não negativas e contíguas:

| N | Início | Tile | P | n_keep_tail | Resultado atual |
| ---: | ---: | ---: | ---: | ---: | --- |
| 256 | 0 | 128 | 0 | 129 | Sucesso, tiles127/128/1; janela dividida, nenhum tile protegido |
| 256 | 0 | 128 | 0 | 256 | Sucesso, tiles128/128; toda a janela dividida |
| 32 | 500 | 128 | 512 | 64 | Sucesso, tiles12/20; a janela efetiva cruza P |

O terceiro caso ocorre porque a validação de cruzamento de P só roda quando `N>=n_keep_tail`. Se N é menor, todos os tokens presentes pertencem à janela protegida e precisam permanecer juntos.

Correção recomendada: calcular `protected_tokens=min(N,n_keep_tail)`, validar essa janela também contra P e resolver explicitamente o conflito entre atomicidade e limite do tile. Para o contrato atual de tiles de no máximo T, rejeitar `protected_tokens>T` com erro explícito é a opção mais simples. Uma extensão que permita o último tile maior precisa auditar os limites de ubatch, RPC e rollback; não pode ser um crescimento silencioso.

Também há uma fragmentação evitável: N256,T128,keep4 produz128/124/4, embora128/128 já preserve os quatro tokens finais juntos. O clamp para `rec_start` só é necessário quando `cand<N` e o corte dividiria a janela. O último tile pode conter tokens anteriores à janela protegida.

Um núcleo de correção para esse contrato é:

```cpp
const uint32_t protected_tokens = std::min(n_keep_tail, ubatch.n_tokens);
if (protected_tokens > tile_size) {
    error = "protected recurrent window exceeds tile size";
    return false;
}
const uint32_t rec_start = ubatch.n_tokens - protected_tokens;
// Validar [pos[rec_start], uint64_t(pos[last]) + 1) contra P quando
// protected_tokens != 0, mesmo que n_keep_tail > ubatch.n_tokens.

uint32_t cand = curr + std::min(tile_size, ubatch.n_tokens - curr);
if (protected_tokens && curr < rec_start &&
    cand > rec_start && cand < ubatch.n_tokens) {
    cand = rec_start;
}
```

Manter a validação de P antes de construir qualquer subconjunto. Validar posições antes de convertê-las para unsigned; o `pos[last]+1` atual também deve usar aritmética alargada para evitar overflow assinado no limite de llama_pos. Ao retornar erro, limpar o plano inteiro ou publicar um plano temporário somente no sucesso: atualmente vetores são limpos, mas metadados e schedule de uma chamada anterior podem permanecer.

### P1 Máscara ausente habilita atenção ao futuro dentro do tile

Referência: [builder, linhas485–488](../../src/llama-wavefront.cpp#L485).

Um vetor `masks` curto ou uma entrada null vira `nullptr` em `ggml_soft_max_ext`. Isso significa softmax sem máscara. O concat exclui tiles posteriores, mas não exclui os tokens futuros que já estão dentro do tile atual.

O problema foi confirmado com uma camada, um tile de16 tokens, dimensão1, pesos iguais a1, primeiro token1 e token final alterado de2 para4:

```text
Sem máscara: primeira saída 1.72577047 -> 4.10042
Com máscara causal: primeira saída 2 -> 2
```

Todas essas saídas são finitas. Portanto, o teste atual de ausência de NaN/Inf não detecta esse vazamento causal.

O builder precisa exigir máscaras presentes e corretamente dimensionadas por tarefa ou construí-las internamente. Para o protótipo atual, a forma é `[tokens_acumulados, tokens_query]`, com passado visível e triangular causal no tile atual. Validar apenas o tamanho do vetor não estabelece que o conteúdo seja causal; uma fonte única de construção de máscaras reduz esse risco. Adicionar testes de máscara ausente, null, shape incorreta e perturbação de futuro dentro do mesmo tile.

### P1 A conclusão do tile0 do prompt não estabelece TTFT da resposta

Referência: [afirmação no header, linhas27–28](../../src/llama-wavefront.h#L27).

Para um prompt dividido em K tiles, o primeiro token da resposta depende dos logits da última posição do prompt, no tile K−1. O resultado do tile0 prevê a continuação daquele prefixo; essa continuação já faz parte do restante do prompt conhecido. Não é uma resposta condicionada ao prompt inteiro. Pelo mesmo motivo, concluir o primeiro tile do prompt não autoriza verificar como resposta um draft MTP condicionado ao prompt completo.

No DAG idealizado, o tile0 chega à última camada na onda L−1 e o último tile chega na onda L+K−2. Com células de custo unitário e recursos independentes suficientes, o trabalho total é LK e a profundidade é L+K−1. O índice da onda não mede tempo de parede.

No executor CPU atual, as células são expandidas em um único grafo e os nós são executados serialmente. Para L>=K, a ordem usada faz o tile0 terminar depois de `LK−K(K−1)/2` células. Exemplo: L32,K4 significa122 células antes de terminar o tile0, não32. O total continua128 células.

Retirar a promessa `TTFT O(L*K) -> O(L)`. O benefício a demonstrar é redução de tempo exposto por overlap útil entre recursos, mantendo a geração condicionada a todo o prompt. Conclusões intermediárias podem alimentar progresso ou consumidores explícitos de logits do prompt, mas não equivalem ao primeiro token gerado da resposta.

### P1 A execução híbrida e o serviço progressivo ainda não estão integrados

Referências: [builder](../../src/llama-wavefront.cpp#L425), [execução no teste](../../tests/test-wavefront-pipeline.cpp#L565).

As referências ao módulo estão restritas aos próprios arquivos e aos testes. Não há chamadas no contexto de inferência, nos builders de modelos, no servidor ou no caminho RKVA. `ggml_build_forward_expand` acrescenta nós a um grafo; não despacha uma onda, não cria streams, não espera eventos e não notifica consumidores progressivos.

O protótipo contém projeções Q/K/V, atenção de uma cabeça e projeção/residual. Não contém norm, RoPE, GQA, gates, FFN, estado recorrente, histórico anterior ao ubatch, merge local/remoto por LSE, KVarN nativo ou transporte. `tile.range` é metadado: não escolhe backend nem executa rede.

O hardware também acrescenta restrições além dos dois predecessores da grade: uma GPU CUDA, engine remoto ordenado, allocator, produtor de cache e buffers têm ownership e capacidade limitados. Tarefas matematicamente independentes precisam ser desmembradas em operações locais/remotas e ter lifetimes próprios para permitir overlap sem races.

O estágio atual pode servir como scaffold e oracle simplificado após as correções. Para concluir a Fase 7 são necessários construção do grafo real por tile, integração com memória/modelo, dispatch por disponibilidade, eventos por operação, joins corretos, backpressure, cancelamento/drain e um consumidor real de resultados. Um trace deve mostrar RPC pendente coexistindo com trabalho CUDA independente, seguido de ganho ponta a ponta.

### P1 Posições iguais são aceitas apesar do contrato estrito

Referências: [validação de posições](../../src/llama-wavefront.cpp#L99), [máscara gerada](../../src/llama-wavefront.cpp#L292), [máscara upstream](../../src/llama-kv-cache.cpp#L4077).

O teste usa `<`, permitindo posições temporais repetidas. A máscara gerada usa índice de token e tile. Para posições `[0,0]` da mesma sequência, a regra upstream mascara `p_key>p_query`; sem componentes espaciais adicionais, a primeira query pode enxergar ambas as chaves. O triângulo por índice do protótipo mascara a segunda. Se posições iguais atravessarem tiles, a suposta independência de tiles futuros também deixa de corresponder àquela máscara por posição.

Para esta primeira implementação, exigir posições temporais não negativas e estritamente crescentes é necessário para justificar a máscara por índice. Para o caminho RKVA serial-prefix, exigir também contiguidade e conexão com o histórico por camada. Se forem admitidas posições repetidas, sparse ou regras M-RoPE, a máscara e a partição precisam implementar a semântica correspondente; `n_pos>=1` não prova isso.

## Outros problemas de robustez e dimensionamento

### P2 O validador topológico confia nos dados que deveria validar

Referências: [task_map](../../src/llama-wavefront.cpp#L216) e acessos a predecessores nas linhas238/248.

`coord.task_id` é usado como índice antes de verificar seus limites. Os predecessores armazenados também são indexados sem validação. Coordenadas ausentes deixam entradas default sem dependências, permitindo que um schedule incompleto pareça válido. O builder não chama esse validador e não verifica compatibilidade entre spec, schedule, vetores de pesos e inputs.

Validar dimensões e aritmética antes de alocar: L+K−1, L*K, task_ids e representação int32 dos predecessores. Verificar que a grade está coberta uma vez, com layer/tile/wave/task_id consistentes, e derivar as dependências canônicas de `(L,T)` em vez de aceitar links arbitrários. Validar todos os índices antes de acessá-los. Esses riscos pertencem hoje a uma API interna do protótipo; não foi identificada exposição remota desse caminho.

O teste denominado inversão vertical troca os dois primeiros elementos. Na ordem criada pelo builder, eles são `(0,0)` e `(0,1)`: esse teste repete uma inversão horizontal. Para testar a vertical, localizar explicitamente task0 e taskK e trocá-los.

### P2 O grafo tem limite default insuficiente para shapes realistas

Referência: [ggml_new_graph na linha437](../../src/llama-wavefront.cpp#L437). O default GGML é2048 nós.

O primeiro tile por camada acrescenta aproximadamente10 nós, e cada tile seguinte12, incluindo concat e transformação de V. Para o grafo apresentado, isso dá `L*(12K−2)`: L48,K4 exige2208 nós; L64,K4 exige2944. O teste2×2 não cobre essa condição.

Usar a política de dimensionamento real de grafos ou `ggml_new_graph_custom` com capacidade calculada e verificada, incluindo dependências adicionais dos inputs/weights. Não basta aumentar um número fixo. O tamanho da arena de tensores também precisa acompanhar os limites aceitos.

### P2 Trackers lógicos não confirmam estado real do cache ou da GPU

`progressive_tracker::step_wave` aceita um índice de onda sem evidência de execução. Depois de `init(4,4)`, uma primeira chamada `step_wave(3)` já marca tile0 pronto. Chamadas fora de ordem ou com vetor de tiles incompleto não são rejeitadas de forma consistente. O contrato precisa ser explícito: somente o scheduler dono pode registrar completion após todos os eventos necessários. Não permitir callbacks concorrentes escrevendo nesses vetores sem serialização.

O tracker KV começa sem histórico e deduz selamento por `novas_linhas%128`. Com histórico0..191 e um tile192..255, um grupo físico termina, mas o tracker vê apenas64 linhas novas e indica não selado. Também não modela cell mapping, sink, stage/tail, geração, seq_id ou confirmação remota; selamento remoto não é atualizado.

Esses trackers podem verificar uma agenda abstrata. Não devem autorizar reads de records, snapshots, publicação de resultados ou rollback. Essa autoridade deve vir das implementações reais de memória e dos eventos/ACKs correspondentes.

### P2 Máscara e concat precisam de limites de protótipo

`generate_causal_mask` verifica apenas a soma de `n_tokens`. Isso não garante offsets em bounds, ausência de sobreposição, ordenação de tile_ids ou cobertura de todos os índices. Validar essas propriedades antes de escrever a máscara e antes de alocá-la. Em tamanho de contexto131072, uma matriz F32 N×N ocuparia64GiB; limitar a função ao ubatch de teste e evitar essa materialização na produção.

Repetir concat de todo o prefixo KV e `cont(transpose(V_prefix))` cria tráfego proporcional à soma dos comprimentos dos prefixos, O(L*T*D*K²). No protótipo com alocação imediata na arena, os buffers intermediários também ficam materializados. Um allocator com reutilização pode reduzir o pico, mas não elimina o custo de cópia. O caminho produtivo deve referenciar o cache existente e usar atenção nativa, sem reconstruir seu histórico a cada tile.

## O que a prova matemática estabelece

Para uma grade não vazia, com dependências completas exatamente `(l−1,t)` e `(l,t−1)`, ambos os predecessores têm onda `l+t−1`. A agenda por antidiagonais é uma ordenação topológica válida, e a largura máxima dessa grade é `min(L,K)`.

Para o bloco de atenção simplificado, com pesos iguais aos da referência, inputs distintos por tile, máscara causal correta e K/V de todo o prefixo visível, uma indução na grade prova equivalência algébrica com atenção causal monolítica. As dimensões de QK e PV no builder estão corretas para esse caso de uma cabeça. Projeções tokenwise e residual preservam a equivalência.

Essa prova deve ser ampliada para norm/RoPE/GQA/FFN e para as transições reais de estado. Ela não estabelece identidade de bits de matmuls com shapes diferentes, nem invariância da representação KVarN quando mudam cadência de store, selamento e tail. “Tile128” também não significa, sozinho, “um grupo selado”: dependem do offset inicial, histórico, mapeamento de células e descriptor.

`n_seqs==1` significa um conjunto de sequências, não necessariamente um único seq_id. O slicing existente preserva os IDs e seu ownership, e não foi identificado defeito nessa cópia. Adicionar um teste com IDs compartilhados `{5,7}`, `n_pos>1`, `seq_id_unq`, `seq_idx` e liberação do ubatch pai antes dos filhos, dentro da política de suporte escolhida.

## Evidência de testes e lacunas

Resultados reproduzidos:

- Binário existente:94.626 checks, zero falhas.
- `ctest --test-dir build-optimized -R 'test-position-split|test-wavefront' --output-on-failure`:13/13 aprovados.
- Recompilação dos fontes wavefront atuais e do helper subset real em um harness CPU isolado com ASan/UBSan: suíte original aprovada e contraexemplos acima confirmados.
- Perturbação de token futuro: vazamento na ausência de máscara e invariância com a máscara causal explícita.

Os testes adicionais e logs estão em `/tmp/beellama-phase7-review`; os binários usados não alteram a implementação do repositório. O helper subset foi compilado a partir do corpo real do fonte, e as operações GGML foram ligadas às bibliotecas CPU usadas na revisão anterior.

Dos checks atuais,65.536 correspondem à matriz triangular256×256 e16.384 à finitude dos outputs do pequeno grafo: aproximadamente86,6% do total estão nessas duas classes. A contagem não mede cobertura do runtime híbrido.

O teste do grafo ignora o status de `ggml_graph_compute_with_ctx`, usa entradas idênticas nos tiles e não compara com uma referência monolítica. Conservar o smoke test, mas exigir também:

1. Inputs distintos e seeded; comparação por camada e de logits contra referência monolítica/F64, incluindo o histórico anterior ao ubatch.
2. Perturbação de tokens futuros dentro e fora do tile, verificando invariância de todas as saídas anteriores.
3. Máscaras ausentes/null/erradas; posições repetidas/negativas/sparse conforme o contrato escolhido.
4. Janela protegida com keep0,1,T−1,T,T+1,N,N+1 e cruzamento de P, incluindo N<keep.
5. Tiles64/128, restos, P−1/P/P+1 e início de histórico desalinhado em grupos de128.
6. Schedule vazio, incompleto, duplicado, incoerente com spec ou com índices inválidos, rejeitado antes de qualquer acesso.
7. Construção com dimensões realistas, como64 camadas e4 tiles, e checagem do status de execução.
8. Paridade de estado/cache e rollback após abort, inclusive com RPC pendente, além de paridade local-only, CUDA/KVarN e RKVA.
9. Traces de streams/eventos e completions que mostrem overlap útil; depois comparação de prefill, TTFT da resposta completa, memória, decode e aceitação MTP.

As correções locais de particionamento e máscara devem preceder a integração. A homologação do subsistema exige os gates de runtime e equivalência acima; a suíte atual qualifica somente o scaffold e seus casos cobertos.
