# Review da Fase 5: inverse WHT após split-reduce e LSE

Parecer: **o reducer está coerente para partials válidos no domínio esperado, mas a Fase 5 integrada permanece sem homologação**. Há um novo achado concreto no merge de LSE do host e um problema de classificação de estados não finitos. A alegação de ausência de perda de precisão em F32 e de redução do tráfego de partials precisa ser corrigida.

Os hashes dos fontes XLLAMA examinados são os mesmos da revisão da Fase 4B. Portanto, permanecem os bloqueadores de C++, buffers/bindings, LDS e compatibilidade entre PSO/reducer descritos em [review da Fase 4B](../xllama-phase4b/review.md). Esta revisão aprofunda o contrato numérico e a interação com o host, sem repetir todos esses achados.

Escopo: XLLAMA `main`, HEAD `3c657399733a18559548cca0cf103dfe6a273cba`, com alterações locais; BeeLlama.cpp na branch `docs/hybrid-attention-master-review`. Fontes de implementação preservados. Documentação/propostas neste diretório, experimentos em `/tmp/xllama-phase5-review`; sem commit, merge, push, build UWP ou deploy. Manifesto: [sources.json](sources.json).

## P1 — O merge do host produz NaN com LSE válidos e finitos

Referência: `src/llama-graph.cpp:5231–5246`, função `build_attn_position_split_rpc`.

O grafo calcula um suposto máximo usando cópias dos LSE limitadas a `[-1e6,1e30]`, mas calcula `exp(lse-mx)` com os LSE originais. A afirmação no comentário de que duas faixas não vazias garantem denominador positivo é falsa para essa construção.

Reprodução com as operações GGML reais, backend CPU:

```text
LSE1=LSE2=-2097152
mx=-1000000
w1=w2=0
OUT=NaN, esperado=2 para O1=1 e O2=3
```

Esse LSE não exige entradas não finitas nem Q F32 extremo. Uma única componente Q=4096, representável em FP8 E5M2, K=-8192, representável em F16 original do tail, e scale=1/16 produzem score=-2097152. Para uma única chave, esse score é o próprio LSE. Faixas com várias chaves de score semelhante também podem produzir LSE abaixo do limite artificial. A validação atual não impõe LSE>=-1e6.

Também reproduzi overflow dos pesos com LSE iguais a 1e31. Esse segundo exemplo pertence ao contrato F32 amplo; o exemplo negativo já demonstra o defeito com operandos FP8/F16.

A proposta [host-finite-lse-max.patch](host-finite-lse-max.patch) substitui o máximo limitado por seleção de uma das duas entradas finitas:

```cpp
ggml_tensor * choose1 = ggml_step(ctx0, ggml_sub(ctx0, lse1, lse2));
ggml_tensor * choose2 = ggml_scale_bias(ctx0, choose1, -1.0f, 1.0f);
ggml_tensor * mx = ggml_add(ctx0,
        ggml_mul(ctx0, choose1, lse1), ggml_mul(ctx0, choose2, lse2));
```

Para LSE finitos, a seleção preserva exatamente o valor escolhido e ao menos um peso é 1. A diferença entre finitos extremos pode transbordar para Inf, mas seu sinal ainda seleciona a entrada correta; o probe também cobriu esses casos. Não basta remover o clamp e manter `(a+b+abs(a-b))/2`, pois somas/diferenças grandes podem transbordar nessa identidade.

O patch é restrito ao merge RPC que admite duas faixas não vazias e LSE finitos. **Não reutilizar esse snippet como tratamento de -Inf para faixas vazias:** multiplicar zero por -Inf produz NaN. Os demais merges do arquivo que possuem construção semelhante devem ser auditados conforme seu próprio contrato.

Validação da proposta: seis pares de LSE executados nas versões atual e corrigida, com O1=1/O2=3, incluindo limites F32; a versão corrigida retornou os valores esperados. `git apply --check` passou. Apliquei o patch somente a uma cópia temporária de `llama-graph.cpp`; seu syntax check C++ passou. Não alterei o fonte de serving nem executei o grafo corrigido em CUDA. Evidências: [host_merge_probe.cpp](host_merge_probe.cpp), [host-merge.log](host-merge.log), [host-patch-syntax.log](host-patch-syntax.log).

## P2 — NaN no denominador é tratado como split/wave vazio

Referências XLLAMA: `rkva_range_reduce_opt.hlsl:37–47`, `rkva_range_attn_kvarn4_opt.hlsl:499–508`, `rkva_range_attn_kvarn4_mq.hlsl:534–545`.

Os merges usam apenas `den>0` para escolher estados participantes. Se den for NaN, a comparação é falsa e o estado desaparece. Um split válido sobrevivente pode produzir OUT/LSE completamente finitos, aceitos pelo host, embora parte da atenção tenha sido descartada.

A emulação confirmou esse comportamento com um split válido e outro com denominador/numerador NaN. Não executei esse caso em GPU. Há um caminho aritmético para esse estado com Q F32 finito extremo, que o protocolo admite: o produto escalar transborda para+Inf e `exp(Inf-Inf)` gera NaN. Esse é um problema de robustez para o caminho F32 extremo, não uma falha esperada do workload FP8 usual.

O estado vazio deve ser identificado especificamente por den==0. Um estado não vazio exige máximo finito e denominador positivo finito; estados inválidos precisam propagar erro/poison. Essa checagem deve existir antes de qualquer merge de waves ou splits: validar somente no reducer é insuficiente se pass1 já descartou a wave inválida e publicou um partial aparentemente vazio. Os numeradores não finitos dos estados participantes também precisam preservar a falha.

Isso não foi introduzido exclusivamente pela WHT pós-reduce; é um risco residual do contrato que a homologação numérica deve contemplar.

## Normalização correta da WHT

A fórmula do pedido mistura a matriz não normalizada com a transformada ortonormal. Seja U a Hadamard não normalizada de ordem 256:

```text
UᵀU = 256 I
U⁻¹ = U / 256
```

O código usa R=U/16, portanto:

```text
Rᵀ = R⁻¹ = R
R² = I
```

Os fatores 1/√128 e 1/√2 do shader implementam essa segunda convenção. **Não é necessário adicionar uma divisão por 256.** A implementação normalizada está correta; a documentação matemática deve explicitar qual matriz está sendo usada.

## Contrato dos partials e do reducer

Para scores a_i já escalados, cada split não vazio representa:

```text
m_s = max a_i
d_s = Σ exp(a_i-m_s)
B_s = Σ_body exp(a_i-m_s) V_rot_i
T_s = Σ_tail exp(a_i-m_s) V_original_i
P_s = B_s + R(T_s)
```

O reducer calcula M=max(m_s), w_s=exp(m_s-M), D=Σw_s d_s e publica:

```text
OUT = R(Σ w_s P_s / D)
LSE = M + ln(D)
```

Em aritmética exata, a linearidade de R recupera `Σw_s(R(B_s)+T_s)/D`. Dividir pelo denominador antes da WHT é válido. Os partials devem continuar sendo numeradores, não saídas já normalizadas. O tail original precisa ser transformado antes de ser misturado ao body rotacionado.

O diff do reducer sozinho não implementa esse contrato: ele depende dos pass1 atualizados e de os DXIL efetivamente carregados representarem os mesmos domínios. O opt legado desatualizado identificado na Fase 4B continua sendo bloqueador para split_k>1.

## Acumulação F32: estável no domínio admitido, mas não exata

Com estados finitos e válidos, w_s está em [0,1], e o split que contém M tem peso 1. Como um split não vazio tem d_s>=1, D>=1. Para as capacidades atuais, a soma dos denominadores é pequena diante do limite F32. Não há cancelamento nessa soma positiva. Pesos de splits muito abaixo de M podem subfluir para zero, descartando contribuições numericamente pequenas.

Os numeradores têm sinais mistos e podem sofrer cancelamento. O teste sintético de três splits com médias rotacionadas `(1e8,1,-1e8)` e d_s=128 perdeu a contribuição intermediária em F32: OUT[0]=0, enquanto a referência em precisão alta dá 0.020833333. Esse teste não demonstra uma regressão exclusiva da Fase 5; demonstra que F32 não garante “sem perda de precisão”.

Como referência de escala, o fator clássico da soma sequencial é γ63≈3.7551e-6 para 64 termos e γ255≈1.5199e-5 para 256 termos, além dos erros de produtos, exp, quantização e butterfly. O erro relativo de uma componente cancelada pode ser arbitrariamente alto. Esses fatores não são um orçamento formal do DXIL compilado, que ainda exige qualificação.

A WHT é bem condicionada na norma 2, mas muda a ordem do arredondamento; o tail passa por duas transformações no caminho split. Avaliar erro absoluto, erro relativo com piso e norma do vetor. Os casos de cancelamento não devem depender apenas de erro relativo por componente.

## Split_k=1, split_k=64 e splits parciais/vazios

- A engine usa direct output quando S=1 e não chama o reducer. Pass1 deve publicar `R(B)+T`, já normalizado, com LSE. Não aplicar outra WHT nessa saída original.
- Para S=64, os 64 estados cabem nos arrays de 256. Threads com lane>=64 não inicializam entradas extras, mas os loops também não as leem.
- Para S=256, cada thread inicializa duas entradas, separadas por 128; as barreiras publicam os dados antes do uso. Não encontrei corrida interna nova no reducer.
- Um split vazio tem d_s=0 e não participa do máximo, denominador ou numerador. Um último split curto é tratado pelo mesmo contrato; a normalização usa seu denominador real, não uma contagem presumida de linhas.
- Uma query inteiramente vazia resulta em OUT=0 e LSE=-Inf. A emulação cobriu S1/64/256, inclusive numeradores inutilizados preenchidos com NaN, que não devem ser lidos.

O layout continua `[split][query_head][head_dim+2]` em F32. Para KVarN, head_dim256 é uma pré-condição real. As leituras lane/lane+128 e stores de OUT cobrem exatamente 256 dimensões. O workspace otimizado de 256×9×24×258×4 tem 57.065.472 bytes; o recurso de referência dimensionado para32 splits é separado e não limita esse caminho.

As barreiras de leitura antes de escrita em cada estágio da WHT do reducer estão corretas. A corrida apontada na Fase 4B é no reuso de LDS entre queries do produtor MQ, não nesta WHT final.

## LSE e merge da camada

LSE está em log natural e usa os scores já multiplicados por `kq_scale`. HLSL `exp` e `log` são base-e, conforme a [documentação de exp](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-exp) e [log](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-log). Não aplicar a WHT ao LSE, nem adicionar/remover ln256.

O store `(total*head_dim+qh)*4` mantém LSE depois de todos os valores de OUT, como esperado pelo wire e pelo decoder host. A mesma somaD normaliza OUT e calcula LSE; isso é necessário para o merge de duas atenções já normalizadas. O bloco LSE do código é comum às duas especializações do reducer. Na emulação, os LSE ficaram bit a bit iguais entre os dois ramos para os mesmos partials.

LSE F32 também tem quantização: em torno de 2^21, o espaçamento representável pode ser 0.25. Nos testes de scores com grande offset, o erro absoluto chegou a0.1062102667, embora o erro relativo fosse pequeno. Isso não foi introduzido pela WHT nem pelo novo merge; decorre da representação F32 de M+lnD. O patch de máximo evita NaN, mas não recupera bits já perdidos no LSE recebido.

Há uma assimetria de admissão que deve permanecer explícita. Xbox `valid_prefix_queries` permite queries iniciais abaixo de origin e, portanto, vazias; por exemplo origin128/live1 com posições127/128. O batch inteiro não é vazio. BeeLlama exige posições>=range_origin e live>0; por isso suas queries nativas admitidas não são vazias, e o decoder rejeita -Inf de LSE. Isso é coerente no caminho Bee atual. Se essa admissão for ampliada, alterar conjuntamente decoder e merge para tratar o sentinel de vazio; não trocar apenas `isfinite` por uma aceitação indiscriminada de infinitos.

## Desempenho e tráfego

A quantidade de dados de partials não diminuiu: continuam 256 valores e 2 escalares F32 por cabeça/query/split. Alterar o domínio desses valores não reduz seus bytes em GDDR6. O benefício demonstrável pelo código é a redução de butterflies e sincronizações LDS em pass1. Ganho de tráfego global requer outra alteração de layout/compressão ou medição específica.

Sem tail, S transformações inversas em pass1 viram uma no reducer: a redução líquida é S-1. Com tail, os splits que o contêm ainda precisam transformar T para o domínio rotacionado. O ganho depende de quantos splits tocam o tail e do custo do reducer. Não extrapolar uma contagem de chamadas diretamente para ganho de TPS.

## Validação executada e limites

Os fontes XLLAMA não mudaram desde os testes ASan/UBSan e oracle `--quick` da Fase 4B; reutilizei essa evidência, sem repetir a mesma suíte. O oracle do projeto ainda usa `R(body)+tail` por split e reducer comum, portanto sua métrica 1.64e-7 não qualifica a Fase 5 nova.

Executei testes adicionais de domínio/redução e o grafo real do merge host:

| Verificação | Resultado |
|---|---|
| Modelo novo versus referência F64 |49 casos; S1/2/3/32/64/128/256; body, tail, misto, último vazio, spread de scores e offsets grandes |
| Maior erro absoluto de OUT nesses casos |7.152557373e-7 |
| Maior erro absoluto de LSE com scores usuais |6.045115217e-7 |
| Maior erro absoluto de LSE com offset ±2^21 |0.1062102667 |
| Query inteiramente vazia |O=0, LSE=-Inf em S1/64/256 |
| Contrato inválido com den=NaN |Estado descartado; OUT/LSE sobreviventes finitos |
| Cancelamento sintético F32 |Contribuição intermediária perdida, conforme descrito acima |
| Merge GGML atual |NaN reproduzido para LSE finitos baixos/altos |
| Seleção corrigida de máximo |Seis pares do probe passaram, incluindo extremos F32 |
| Patch host |apply-check e syntax check da cópia temporária aprovados |

Evidências: [reducer_probe.py](reducer_probe.py), [reducer-probe.log](reducer-probe.log), [host_merge_probe.cpp](host_merge_probe.cpp), [host-merge.log](host-merge.log). A emulação não reproduz execução concorrente HLSL, precisão específica das instruções exp/log do Xbox ou eventuais contrações/reassociações do DXC.

O probe host foi compilado com ASan/UBSan e ligado às bibliotecas GGML CPU já disponíveis da auditoria. Isso não é uma build completa nem uma execução CUDA do produto. Não executei DXC, MSVC/UWP ou GPU no console nesta etapa.

## Antes da homologação

1. Resolver os bloqueadores integrados da Fase 4B, principalmente o domínio do PSO legado e a barreira do produtor MQ.
2. Corrigir o máximo do merge RPC e qualificar sua execução CUDA com os LSE extremos, além dos casos usuais.
3. Propagar estados não finitos desde o primeiro merge de waves, em vez de tratá-los como vazios.
4. Atualizar o oracle para produzir partials rotacionados e executar a WHT final, com testes direct/split, tail-only, splits vazios, cancelamento e LSE.
5. Comparar O/LSE em GPU antes do pack wire, depois validar F16/FP8 end-to-end e medir pass1/reduce/LDS/bytes separadamente.

