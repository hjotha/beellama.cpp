# Review da Fase 4B: Q WHT hoisting e reducer KVarN4

Parecer: **não homologar a implementação atual**. O hoisting de Q é matematicamente válido, e o novo contrato de partials também é válido em aritmética exata. Há, contudo, bloqueadores de compilação, bindings, capacidade/produção dos buffers bulk e sincronização LDS. Os testes CPU existentes passam, mas não cobrem esses contratos novos.

Escopo: fontes locais do XLLAMA em `/home/hjotha/xllama-qwen35-xbox`, branch `main`, HEAD `3c657399733a18559548cca0cf103dfe6a273cba`, com alterações locais do usuário. Examinei também os arquivos modificados não incluídos integralmente no diff da mensagem: unpack bulk F16, reducer, shader opt/decode, producer CPU e headers DXIL. Não editei os fontes XLLAMA nem criei worktree para a inspeção de leitura. A documentação ficou no checkout BeeLlama.cpp, branch `docs/hybrid-attention-master-review`. Não houve commit, merge, push, build UWP ou deploy. Hashes: [sources.json](sources.json).

## P1 — O ramo Q4 não compila: kvarn é usado antes de sua declaração

Referências: `uwp/rkva/remote-attn-engine.cpp:2706`, declaração de `kvarn` na linha 2762.

A seleção do novo reducer foi inserida também no primeiro ramo `if (optimized)`, que atende Q4. A variável local `kvarn` só é declarada depois desse ramo. Não há membro ou declaração anterior que a forneça. Compilar HLSL não detecta esse erro no C++ da engine.

Esse ramo Q4 deve continuar selecionando diretamente `range_reduce_opt_pso_`. A seleção de `range_reduce_kvarn4_pso_` pertence ao ramo posterior de formatos, onde `kvarn` está definido. O patch proposto contém essa correção; a build MSVC/UWP ainda é necessária.

## P1 — Posições gravadas no offset novo são lidas pelo binding antigo

Referências: `remote-attn-engine.cpp:2564`, bindings nas linhas 2846 e 2940.

Para KVarN, o upload CPU agora tem o layout:

```text
[0, Q)       Q original F32
[Q, 2Q)      Q rotacionada F32
[2Q, 2Q+P)   posições I32
Q = n_tokens * n_head * 256 * sizeof(float)
P = n_tokens * sizeof(int32_t)
```

O root parameter 4 de posições ainda recebe `q_addr + q_bytes`. Portanto, o shader interpreta os primeiros bits de Q rotacionada como posições. Isso afeta MQ, opt/decode single-query e fallback de referência quando recebem o layout CPU novo.

Um exemplo do probe: Q com todos os elementos iguais a1 tem `Q_rot[0]=15.999999046`, cujo padrão interpretado como I32 é 1098907647. O shader pode limitar essa posição artificial a todo o cache ativo, dando à primeira query acesso a tokens posteriores ao seu prefixo. Componentes rotacionados iguais a zero ou negativos podem, em vez disso, tornar a janela vazia. A validação CPU da posição real não protege contra esse binding errado.

Centralizar o offset de posições por layout e usá-lo em todos os bindings. Para F16/Q8/Q4 convencionais, o offset continua Q. Os offsets Q e2Q preservam os alinhamentos do formato: o problema é a região selecionada.

Também há uma imprecisão no resumo do pedido: Q rotacionada não ganhou um registro HLSL separado `t4`. MQ lê a segunda região do mesmo `queries : register(t0)`; `positions` é `t3`, ligado pelo root parameter 4.

## P1 — Bulk F16/FP8 GPU não aloca nem produz o layout novo

Referências: `uwp/rkva/rkva-kvarn-bulk-f16.inc:95`, `:175–196`; `shaders/rkva/rkva_range_bulk_f16.hlsl:70–82`.

`transfer.query_f32_default` ainda é alocado com `q_f32_bytes + pos_bytes`. Porém o unpack ganhou um store em `(q_pairs*4 + idx)*4`, que corresponde a `2*q_f32_bytes + idx*4`. Esse store está fora do recurso. O MQ também passa a ler a segunda região de Q, que não cabe nesse recurso.

Para o formato do oracle, `nq=1`, `n_head=24`, D256:

```text
Q bytes                         24576
Width do recurso atual          24580
Primeiro store novo de posição  49152
Capacidade necessária           49156
```

`try_create_buffer` arredonda Width para quatro bytes, não para uma capacidade que cubra esse novo layout. O probe conferiu o mesmo problema para nq1,2,3,4,5,8,9. Essa é uma prova dos offsets/capacidades no código; não executei o store inválido em D3D12 no Xbox.

Aumentar o buffer resolve somente a capacidade. O unpack atual expande F16 para Q original e duplica posições; **ele não calcula WHT(Q)**. `attend_range` pula a preparação CPU quando `bulk_gpu_query_` está ativo. Assim, não há produtor da região que o novo MQ consome. Isso inclui FP8, que nesse caminho é expandido para F16 antes do unpack GPU.

É necessário implementar o produtor GPU de Q rotacionada, preferencialmente fundido ao unpack por cabeça, ou um passe WHT com recursos/transições corretos. Confirmar a conclusão de UAV antes do uso como SRV. Duplicar posições no novo offset não substitui esse produtor. Se o caminho ainda não puder produzir o layout, ele deve selecionar uma alternativa compatível e explícita.

## P1 — O fallback CPU do bulk F16/FP8 também mantém a alocação antiga

Referências: `rkva-kvarn-bulk-f16.inc:117–121`, `:244–252`, preparação de Q em `remote-attn-engine.cpp:2558–2565`.

Quando a conversão GPU não está disponível, `transfer.query` ainda tem Q+P bytes. Esse recurso é trocado para `range_query_` e `attend_range` escreve nele duas regiões de Q mais posições. O resultado é uma escrita CPU além do Width do recurso mapeado.

O crescimento de `range_query_` em `ensure_range_scratch` não corrige os recursos próprios de cada transferência. O scratch já foi preparado para `max_queries` antes do loop; o caminho posterior pode retornar cedo pela contagem de tokens, enquanto o recurso menor foi trocado para seu lugar.

Esse fallback precisa de `2*q_f32_bytes + pos_bytes`. O bulk F32 teve sua alocação equivalente atualizada, mas o include de F16 ficou inconsistente. Portanto, desligar o caminho GPU sozinho não torna o código atual seguro.

## P1 — Falta uma barreira antes do reuso de LDS nos splits sem tail

Referência: `shaders/rkva/rkva_range_attn_kvarn4_mq.hlsl:519–556`, ramo599–604.

Cada query publica `gs_max`, `gs_den` e `gs_vec`, sincroniza e lê as contribuições de todas as waves. No ramo `direct_output==0 && !has_tail`, o código agora apenas grava os partials. Uma wave pode avançar para a query seguinte e sobrescrever esses arrays enquanto outra ainda lê a query anterior. A WHT incondicional removida fornecia a barreira de aposentadoria dessas leituras.

Esse caminho é comum: muitos splits de uma atenção com precision-tail 128 não contêm nenhuma linha de tail. Em wave64, o grupo de 128 threads contém duas waves e a diferença de progresso entre elas importa. MQ4 também pode atravessar a fronteira entre pares.

Correção mínima, dentro do ramo `!has_tail`, antes dos stores:

```cpp
// Retire LDS reads before the next query reuses gs_* storage.
GroupMemoryBarrierWithGroupSync();
```

As condições relevantes são uniformes por grupo, portanto essa barreira não fica em um desvio divergente por lane. Outra solução é restaurar uma barreira uniforme antes da escolha direct/split. A semântica da barreira e a exigência de controle uniforme constam da [documentação HLSL da Microsoft](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/groupmemorybarrierwithgroupsync).

Não encontrei erro equivalente na inicialização nova de Q2/Q3: o clamp de queries fantasmas mantém os endereços válidos; os dois blocos de 1536 floats cabem em LDS; a barreira da linha 346 publica os dados antes do consumo.

## P1 — O DXIL opt legado não foi atualizado junto com o domínio dos partials

Referências: `shaders/generated/rkva_range_attn_kvarn4_opt_dxil.h:3`, `remote-attn-engine.cpp:785`, `:2864`.

O header de opt sem serial-prefix continua byte a byte igual ao HEAD anterior. Seu hash de fonte embutido é `976d634828a4a359da4e0437eca2e3a8c52fb25d3bcabcbc129ea4752bda7bd5`; o fonte atual é `c89306bd2372c44310c91794d70aa939e2c30e9312479e240b0897831f3d3a54`. Os headers de opt_prefix, decode_prefix, MQ2/MQ4 e do novo reducer declaram o hash atual dos respectivos fontes.

O contrato antigo de opt produzia `H(body)+tail`, no domínio original, e era combinado com reducer comum. A engine passou a selecionar o reducer WHT para todos os KVarN otimizados, incluindo o binário legado que não foi regenerado junto com a mudança. Para split_k>1, aplicar WHT a um partial já no domínio original muda a saída. Um contraexemplo do probe é um partial original e0: deveria manter O[0]=1, enquanto H(e0)[0]≈0.0625 e o erro máximo é0.9375.

Regenerar o target `attn_kvarn4_opt` junto com os demais targets afetados, ou manter explicitamente o reducer comum para o binário legado. O domínio precisa ser propriedade do PSO efetivamente carregado, não apenas da classe de storage. Os hashes são evidência de sincronização de fontes/artefatos; não executei ou desmontei esse DXIL com DXC para certificar sua implementação em GPU.

## Equivalência matemática

Se H é a WHT256 normalizada, então Hᵀ=H e H²=I em aritmética exata. Para a chave reconstruída do domínio KVarN, k̂=H·k_rot:

```text
qᵀ k̂ = qᵀ H k_rot = (Hq)ᵀ k_rot
```

Não há fator adicional de 256 ou 16. A implementação CPU usa as mesmas sete etapas H128 por metade, normalização 1/√128 e mistura H2 com 1/√2 do shader. A igualdade refere-se aos K/V representados pelos records/stage; não desfaz a quantização do KVarN.

O tail original precisa continuar sendo pontuado com Q original. O shader MQ preserva esse caminho. Para numeradores B no domínio rotacionado e T no domínio original, pesos de merge w_s e denominador D, o contrato novo é válido:

```text
P_s = B_s + H(T_s)
OUT = H(Σ_s w_s P_s / D)
    = Σ_s w_s (H(B_s) + T_s) / D
```

O caminho direct mantém `H(B)+T`. O reducer novo aplica a WHT normalizada depois do merge dos partials; suas barreiras internas e layout `(256+2)` por cabeça/split estão coerentes. O workspace otimizado é dimensionado por `kOptMaxSplits=256`, `kOptMaxQueries=9`; splits64 não exigem ampliar o buffer de referência de32, porque são recursos distintos.

Essa transformação não garante identidade bit a bit em FP32: H(H(T)) acumula erro, e mover a transformação depois do merge/normalização muda a ordem de arredondamento. Validar com tolerância absoluta e relativa, incluindo cancelamento e componentes próximas de zero. Não usar “100% de precisão” como sinônimo de equivalência FP32 bitwise.

## P2 — Decode e outros modos precisam de seleção de layout mais precisa

Opt single-query e decode continuam calculando WHT(Q) no shader (`rkva_range_attn_kvarn4_opt.hlsl:319`, `:600`). O upload CPU agora calcula e envia Q rotacionada para todo KVarN, mesmo quando esse dado não é consumido. Há trabalho CPU e tráfego de upload adicionais nesses modos sem remoção da WHT GPU correspondente. O fallback de referência também não consome o hoisting.

Resolver primeiro qual kernel/PSO será usado e escolher o layout a partir dessa decisão. Isso permite restringir o hoisting a consumidores qualificados e deixar decode/referência com seu contrato original, ou estender explicitamente esses shaders para consumir a rotação pré-computada.

Para F16/Q8/Q4 convencionais, a cópia de posições permanece emQ e o reducer deve continuar sem WHT. A alocação geral foi dobrada mesmo nesses formatos, embora a segunda região não seja usada. Além disso, a criação do reducer KVarN foi acoplada ao sucesso do pipeline Q4: falhar criando esse PSO agora reseta também os PSOs Q4/reducer comum. Separar a capacidade opcional evita perder otimizações de formatos não relacionados.

Os contadores de profiling devem distinguir bytes wire, bytes de Q original e bytes de Q rotacionada enviados à GPU. As expressões antigas com apenasQ+P não descrevem integralmente o upload CPU KVarN novo.

## Cobertura real dos testes

Recompilei e executei os dois testes C++ indicados com `-fsanitize=address,undefined`: ambos passaram. O modelo Python `--quick` também passou e reproduziu `worst relative |O-oracle| 1.64e-07`. Evidências: [bulk-cpu.log](bulk-cpu.log), [profile-cpu.json](profile-cpu.json), [oracle-quick.log](oracle-quick.log).

Esse modelo, porém, ainda monta `acc = butterfly_wht(bm) + tm` na linha 596 e chama `m4.reduce_splits` comum na linha 604. Ele não exercita o novo `body+H(tail)` e reducer WHT, não constrói os bindings D3D12, não aloca os buffers da engine e não executa o compartilhamento LDS entre queries. Portanto, o resultado informado é real, mas não qualifica a alteração nova.

Acrescentei um probe independente da auditoria, sem modificar os testes do projeto: [domain_layout_probe.py](domain_layout_probe.py), wrapper do helper real em [wht_wrapper.cpp](wht_wrapper.cpp), resultados em [domain-layout.log](domain-layout.log).

Resultados:

- Helper CPU versus emulação F32 do shader: 512 vetores, 131.072 componentes bit a bit iguais, no build host usado.
- Novo contrato de partials versus contrato anterior: 16 casos sintéticos de body-only, tail-only, misto e cancelamento, com 2/32/64/256 splits; maior diferença absoluta 2.384185791e-7.
- Contraexemplo de reducer em domínio errado: erro máximo 0.9375.
- Capacidades e offsets bulk: inconsistências confirmadas para nq1/2/3/4/5/8/9.
- Binding antigo de posições: leitura dos bits de Q rotacionada confirmada no exemplo controlado.

Os casos sintéticos validam a identidade numérica isolada; não substituem a atenção real no shader. Não executei DXC, MSVC/UWP, GPU D3D12 ou deploy no Xbox nesta auditoria.

Comandos executados, na raiz do XLLAMA:

```bash
c++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer tests/rkva_kvarn_bulk_cpu.cpp -o /tmp/xllama-phase4b-review/bulk-cpu
/tmp/xllama-phase4b-review/bulk-cpu
c++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer tests/rkva_range_profile_cpu.cpp -o /tmp/xllama-phase4b-review/profile-cpu
/tmp/xllama-phase4b-review/profile-cpu
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 tests/rkva_range_kvarn4_opt_model.py --quick
c++ -std=c++17 -O2 -fPIC -shared /tmp/xllama-phase4b-review/wht_wrapper.cpp -o /tmp/xllama-phase4b-review/wht.so
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 /tmp/xllama-phase4b-review/domain_layout_probe.py
git apply --check /home/hjotha/beellama.cpp/docs/reviews/xllama-phase4b/partial-fixes.patch
```

## Patch proposto e critérios para homologação

[partial-fixes.patch](partial-fixes.patch) corrige a referência C++ fora de escopo, os bindings de posições, as capacidades bulk F16 e a barreira LDS. `git apply --check` passou contra os fontes examinados. Não foi aplicado, compilado em UWP ou empacotado.

**O patch é parcial:** não implementa o produtor GPU de Q rotacionada, não atualiza DXIL e não adapta o oracle. Não deve ser tratado como uma correção suficiente para deploy.

Antes da homologação:

1. Fechar o contrato completo de cada produtor/consumidor: CPU, bulk F32, bulk F16/FP8 GPU e fallback CPU, incluindo capacidade, região de posições, domínio dos partials e PSO correspondente.
2. Implementar o produtor GPU ou uma alternativa compatível; corrigir barreiras e regenerar todos os DXIL afetados, incluindo opt legado. Compilar a engine UWP para capturar erros C++.
3. Atualizar o oracle para o contrato novo e testar nq1..9, tiles2/4, direct e splits2/32/64, descriptors1/2/3, tail0/128, body-only/tail-only/misto, grupos parcialmente ocupados e fallback sem PSO MQ.
4. Executar no console a cadeia completa com F32/F16/FP8, conferir O/LSE contra referência, validar ausência de acessos fora de recursos e exercitar repetidamente splits sem tail para expor a corrida de LDS.
5. Medir separadamente WHT CPU/upload, pass1, reduce e total. A remoção da WHT por split tem mérito, mas o ganho líquido depende do custo de preparação e dos caminhos efetivamente selecionados.

