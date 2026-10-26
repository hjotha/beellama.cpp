# Revisão final do INT8 RKVA — 2026-10-26

**Ainda existe uma hipótese realista de desempenho: `WavePrefixSum` seguido
de broadcasts com índices constantes. Entreguei esse braço para medição.
Não existe nesta entrega um substituto simultaneamente qualificado em
velocidade e qualidade para a produção. O Q8 continua sendo um bloqueio de
qualificação, independentemente da redução.**

Portanto, eu não descartaria INT8 por uma suposta impossibilidade de reduzir
16 lanes corretamente. Também não promoveria este INT8 como equivalente ao
MQ2 atual. “Correto” abaixo distingue semântica D3D, concordância com a
aproximação Q8, equivalência ao oráculo F32 e qualificação no console.

Trabalho somente no clone/branch solicitados. Nenhum acesso ao Xbox, 4070,
llama-server ou systemctl; nenhum build CUDA/MSVC/UWP, push ou merge.
DXC somente pelo `.oc-artifacts/dxc/dxc193.py`, cujo diretório remoto é
`C:\Users\hjotha\build\oc-int8-codex-20261026\`. Não consultei/pollei o console.

## Evidência e método

Aceito como medidos os 11,09 TFLOP/s F32, 37,6 TOPS INT8, rejeição de F16
nativo/dot2, 99,93 ns/linha de MQ2, 69,26 ns do XOR defeituoso e 112,01 ns do
masked correto em relação ao Q8. Não os reestimei a partir do pico teórico.
O ganho do XOR defeituoso não serve como demonstração de ganho útil.

Copiei `hw/quality/serial-i8masked-302.json` para `.oc-artifacts/`;
SHA256 `9961049108a669cf9d409ac5bb1c25068fc680543bc89ee132b1b0269c13cc29`.
Li os quatro `hw/sweep/i8m-{off,on}-r{1,2}.json`: plano de 28.672 linhas,
nq=256, seed=4242, wire=fp8, mode=2, tail=128, segmentos de 8 queries.
São microprobes sintéticos, sem arquivo de modelo, prompt ou sampling;
não constituem benchmark de geração. O commit do pacote medido não está
identificado nesses planos, portanto não atribuo esses tempos ao novo código.

Usei `rkva_dxil_loop_cost.py` sobre a assembly do DXC autorizado. Corrigi
o contador para reconhecer prefix/quad/LDS e permitir `--header`/`--rows`:
o seletor automático baseado em dot4 escolhia o loop residual no bisect F32,
porque seu loop rápido usa multiplicações escalares. Isto não era custo
válido para o fast path F32. As contagens são **proxies estáticos DXIL**, com
wave64 estimada: não são ISA, não medem spills, bancos LDS, ocupação, latência
de barreira nem todos os caminhos dinâmicos. Prefix de wave inteira é
estimado como seis movimentos e seis somas; não sabemos seu lowering Xbox.

| Fast path compilado | Instr./linha | Redução/movimentos por linha | Barreiras no loop rápido |
|---|---:|---:|---:|
| MQ2 de produção | 512,0 | 144,0 | 0 |
| INT8 XOR, defeituoso no Xbox | 331,9 | 24,0 + 12 somas explícitas | 0 |
| INT8 masked | 453,4 | 144,0, além das máscaras | 0 |
| **INT8 prefix, entregue** | **355,4** | **48,0 + 12 somas/subtrações explícitas** | **0** |
| F32 16 dims/lane + prefix, bisect compilado | 650,9 | 48,0 + 12 somas/subtrações explícitas | 0 |

Para o último: `.oc-artifacts/dxc/mq2_f32_prefix.txt --header=2318 --rows=8`.
Seu hot loop também tem 15,5 loads de buffer por linha, contra 3,5 do INT8.

O custo do prefix relativo ao masked é 355,375/453,375 = 0,78384. A hipótese
linear calibrada no **INT8 correto medido**, não no XOR, dá
`112,01 × 0,78384 = 87,80 ns/linha`, aproximadamente 12,1% abaixo de 99,93.
O orçamento para empatar é 404,48 unidades/linha: há margem de 49,1 unidades,
ou 13,8% sobre a estimativa do candidato. Não é intervalo de confiança nem
previsão garantida. O próprio masked, mais lento que MQ2 apesar de menor
contagem, demonstra a importância de custos que o contador não representa.
Não reaproveito os −30,7% do braço defeituoso.

## Formas de redução ou de evitá-la

As famílias relevantes são coletivas de wave, leituras de lanes, memória
compartilhada/global e mudança da distribuição de trabalho. Não há uma lista
finita de todos os shaders possíveis. A tabela cobre essas famílias e as
alternativas pedidas. Para alternativas não implementadas, dou **piso ou
estimativa analítica**, derivada da mesma contagem, não uma compilação fictícia.
No layout INT8 atual o trabalho comum sem redução custa aproximadamente
**296 unidades/linha**; há três conjuntos de quatro scores por linha
(`12 query-heads / 4 linhas`). QK/PV/softmax e montagem dos nibbles permanecem.

| Alternativa | Estimativa instr./linha e sincronização | Registradores/LDS e decisão |
|---|---|---|
| Quatro `WaveActiveSum` mascarados | 453,4 compilado; 0 barreiras | Estado atual; válido e medido, 112,01 ns. Não ganha. |
| **Prefix inteiro + diferenças** | **355,4 compilado; 0 barreiras** | Quatro endpoints e alguns temporários, nenhuma LDS adicional. Melhor próximo experimento. |
| `WaveMultiPrefixSum` com quatro máscaras fixas | Piso ≈335: `296 + 3×(8 + 1 + 4)`; scan de seis estágios levaria ≈347; lowering serializado pode ser muito pior. 0 barreiras | Máscara uint4, prefix e endpoints. Pode ganhar; depende de aceitação e lowering SM6.5. Sem necessidade de WaveMatch. |
| `WaveMatch(row)` + MultiPrefix | Piso acima + custo desconhecido do match, amortizado fora do loop | Máscara fixa já resolve o layout. Match é útil como sonda, não traz benefício aqui. |
| Quads + LDS para completar 16 lanes | Piso ≈344: `296 + 3×(2 reads quad + 2 adds + 1 store + 4 loads LDS + 3 adds + 4 broadcasts)`; endereçamento/barreiras vêm além | Pode ganhar se quads forem qualificados e trocas forem agrupadas; não é melhoria demonstrada. Ver detalhes abaixo. |
| Butterfly inteiramente via LDS | Piso ≈344: quatro estágios de store/load/add + quatro broadcasts, três conjuntos/linha; **até 24 barreiras/linha** no esquema ingênuo | 512 B de scratch escalar para 128 threads; barreiras e endereços destroem o piso. Não escolher. |
| LDS: cada lane lê as 16 parciais da sua linha | Piso ≈404: `296 + 3×(1 store + 16 loads + 15 adds + 4 broadcasts)` antes dos endereços | Redução redundante nas 16 lanes. Quase esgota a margem antes da sincronização. Não escolher. |
| **Transpose-reduce LDS em lote de 16 linhas** | Piso de transporte/redução ≈20,8/linha, total comum ≈317; endereços, retiling e PV adicionais não incluídos; duas barreiras por lote, 0,125/linha | Ainda plausível. 24 KiB para scores de duas waves, reutilizando scratch de outra fase; caso contrário excede 32 KiB com o merge. Mais invasivo que prefix. |
| Uma linha completa por lane, 64 linhas/wave | QK: `12×64 dot4 / 64 linhas = 12 dot4/linha`; redução QK zero, mas 12 broadcasts de score/linha ou transpose para PV | Não se obtém speedup adicional de dot4: o atual já custa 12/linha. Requer redistribuir PV e Q; detalhes abaixo. Não é um drop-in. |
| F32, 16 dims/lane, decoder exato atual + prefix | **650,9 compilado**, 0 barreiras; escala linear masked daria ≈160,8 ns | Não ganha nessa implementação. O aumento de loads e decode supera a economia de redução. |
| F32 fatorado/hoisted, 16 dims/lane | Piso QK de 48 FMAs/linha em vez de 12 dp4a, pelo menos +36, antes de conversões/constantes; ordem de 400+ é plausível, não medida | 192 coeficientes Q por lane para 12 heads, contra 48 words Q8. Spills/ocupação são risco sério; não demonstrado inviável, nem candidato pronto. |
| Gather com apenas índices uniformes | 64 broadcasts + 60 somas por conjunto de quatro scores: ≈668 total/linha | Sem LDS, sem índice variável, mas desperdiça paralelismo. Não ganha. |
| `WaveActiveSum` dentro de quatro braços divergentes | Piso ≈404 se cada grupo ativo reduz em quatro estágios; pode aproximar masked ou pior se usar seis estágios/serialização | Legal para lanes ativas, com reconvergência antes dos broadcasts. Sem barreira/LDS, mas quatro braços serializados e custo EXEC. Não assume redução de 16 lanes a preço de uma wave inteira. |
| UAV + segunda dispatch | Pelo menos três stores de parciais e três loads por linha, além de redução, endereços, tráfego e dispatch/UAV barrier | Troca dependência de wave por memória global e outro pass. Poderia ser um redesign matricial, não há expectativa justificada para este pass curto. |
| Atomics/LDS ou CAS float | Pelo menos três atomics de parciais/linha, **16 concorrentes por score**, limpeza, leitura e sincronização adicionais; custo de retries não limitado | Não há soma F32 atômica comum nesse contrato SM6.4. CAS implementável, caro; inteiros precisam respeitar escalas diferentes dos slices. Sem ganho defensável. |

O estado persistente do INT8 já representa cerca de 144 escalares por lane:
48 words Q8, 12 escalas, 12 constantes, 48 acumuladores PV e 24 max/den.
Somam-se K, V, índices e temporários. Isso é inventário semântico, **não
contagem de VGPR físicos**: só ISA/estatísticas do driver podem demonstrar
alocação, spills e ocupação. O prefix não acrescenta arrays persistentes;
os braços LDS e de linha inteira têm as pressões adicionais indicadas acima.

**Contrato de wave.** `WaveReadLaneAt` com índice variável é permitido pelo
D3D/HLSL, desde que a origem esteja ativa. Logo o XOR não é proibido pela
linguagem; aceito o defeito confirmado no console como restrição deste
lowering. O candidato usa somente índices literais `15,31,47,63` e mantém
todas as 64 lanes ativas no fast path. A guarda wave64 e `act[j]` uniforme
garantem isso. [Contrato oficial](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/wavereadlaneat).

**Prefix.** Se `P(l)=sum(val[0..l])`, os scores são `P15`, `P31-P15`,
`P47-P31`, `P63-P47`. `WavePrefixSum` é exclusivo, portanto o shader soma
`val` antes dos broadcasts. As duas reduções mascaradas de máximo de Q são
mantidas fora do hot loop; não ficou um XOR escondido na quantização.
Há cancelamento F32 entre prefixos: não é bit-exato, nem seguro para toda
distribuição de magnitudes imaginável (por exemplo, 1 após 1e20). Nos fixtures
auditados ficou na escala normal de arredondamento F32. O intrinsic existe
desde SM6.0; não exige que o console aceite SM6.5.
[Semântica de prefix](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/waveprefixsum).

**MultiPrefix.** Para lane `l`, `bits=0xffff<<(l&16)`;
`mask=(bits,0,0,0)` se `l<32`, senão `(0,bits,0,0)`.
As quatro máscaras particionam a wave, e são idênticas dentro de cada grupo.
Após `p=WaveMultiPrefixSum(val,mask)+val`, os endpoints constantes já são
os quatro scores: não há subtração entre grupos. Não há `MultiPrefixMax`:
o máximo de quantização ainda precisa de outra solução. SM6.5, não SM6.6,
introduziu MultiPrefix e Match. Sua aceitação UWP continua desconhecida.
[Especificação Microsoft](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_ShaderModel6_5.html).

**Quads.** Dois estágios X/Y bastam para quatro lanes; diagonal é redundante
na butterfly, ou pode participar de um gather de três vizinhos. Isso cobre
xor 1/2, não xor 4/8. Compute SM6.0–6.5 não promete a organização que esse
algoritmo pressupõe. SM6.6 define o layout 1D para `numthreads(128,1,1)`.
Uma aceitação PSO não prova que o driver Xbox faça o shuffle corretamente.
[QuadReadLaneAt](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/quadreadlaneat),
[layout SM6.6](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_Derivatives.html).

Para completar xor 4/8, é melhor gravar uma soma por quad e ler as quatro
somas que fazer dois estágios LDS separados. No esquema escalar há duas
barreiras por query-head/half-tile, seis por linha. Agrupar os 12 heads em
scratch de 3 KiB permite duas barreiras por half-tile, 0,5 por linha; o total
de LDS passa de 12.384 B para 15.456 B, se não houver reutilização.
Não basta inserir uma barreira no loop existente: as duas waves podem ter
números diferentes de tiles na borda. É preciso um loop uniforme de grupo,
mascarando tiles vazios. Memória de wave não dispensa a sincronização LDS
documentada. [Barreiras em fluxo divergente são indefinidas](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/groupmemorybarrierwithgroupsync).

**Transpose em lote.** Dezesseis linhas × 16 parciais × 12 heads × 4 B =
12 KiB por wave; duas waves precisam de 24 KiB. Cada lane produz parciais
de quatro linhas e grava. Depois, cada lane consumidora soma as 16 parciais
de um par (linha,head). São 192 scores por wave, em três lotes de 64 lanes:
`3×(16 loads+15 adds)/16 = 5,8125` instruções/linha, mais três stores e
12 broadcasts de scores/linha. É isso que origina o piso 20,8, **não uma
redução gratuita de 16 scores**. Duplicar o scratch junto aos 12.384 B do
merge ultrapassaria 32 KiB; reutilizá-lo entre fases cabe, porém pode reduzir
ocupação. Índices LDS com padding/bancos, dois words K por dimensão em vez
de um, uniformização das bordas e carga/PV dos 16 V ainda precisam ser pagos.
Não descartei esta família. Preferi entregar primeiro o prefix, que altera
somente a redução e permite um A/B mais conclusivo.

**Linha inteira por lane.** Q8 pré-carregado para 12 heads ocupa
`12×256/4=768` words/lane, antes de PV/softmax. Q F32 ocuparia 3.072 floats.
Em LDS, Q8 ocupa 3 KiB e F32 12 KiB por grupo, mais escalas/constantes;
isso economiza registradores mas exige 768 loads de words por lane para
12 scores (12 loads amortizados por linha, antes das outras operações).
Processar um head de cada vez reduz estado, porém perde a reutilização de
K/V entre heads e precisa de outro esquema para PV. Guardar uma saída inteira
por lane acrescentaria 256 acumuladores por head e várias saídas parciais a
combinar. O layout transposto K favorece loads de posições vizinhas, mas essa
vantagem não elimina os custos de Q/PV. O teto INT8 não basta para escolhê-lo.

**Demais variantes.** Prefix inteiro dos acumuladores INT32 evita cancelamento
na soma inteira, mas são oito somas de slice: `other` e escala Q diferem por
slice. São oito endpoints/conversões/combinações, não um único score inteiro.
Empacotar quatro somas em um INT32/INT64 não cabe sem prova de limites:
um slice já pode ter `128×126×15=241920` em módulo, e escalas são F32.
Combinações lineares de quatro WaveActiveSum continuam custando quatro
reduções. CAS float não elimina contenção e atomics não implicam barreira.
[InterlockedAdd](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/interlockedadd).
Wave32/WaveSize exigiria outro layout e qualificação SM6.6/capabilities;
não altera o fato medido de wave64 nem contorna a guarda atual.
[WaveSize](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_WaveSize.html).
Não encontrei prova de disponibilidade de intrinsics AMD/GDK específicos
no Xbox **UWP** desta aplicação; não são uma saída assumida. F16 nativo
continua excluído pela rejeição medida. Duplo Q8/resíduo em F32 ou mais bits
poderiam diminuir erro, mas acrescentam dot/estado/quantização e precisam
de avaliação própria; não são correções exatas e gratuitas deste Q8.

## Qualidade: bloqueio para promoção

O novo `rkva_mq_i8_oracle_scale.py` reproduz o RNG seed=20261266, os limites,
append e rewind do probe, sem rede. São 62 casos (excluído somente capacity
edge), dos quais 39 exercitam o Q8. Calcula magnitude do **oráculo**, não
usa a saída errada do shader como denominador, e modela a aproximação Q8
apenas nas linhas compartilhadas do par MQ2. O JSON do hardware só contém
máximos: não permite recuperar erro relativo componente a componente no GPU.

Nos 39 casos afetados:

| Medida | Intervalo |
|---|---:|
| RMS do OUT do oráculo | 0,9074–1,9331 |
| Mediana de `abs(OUT)` por caso | 0,2075–0,4845 |
| Percentil 95 de `abs(OUT)` por caso | 1,5747–3,8518 |
| Maior `abs(OUT)` por caso | 23,35–68,72 |
| Erro relativo L2 Q8 modelado | **0,6297%–1,3951%** |
| Máximo absoluto de erro Q8 modelado | **0,1986–0,9994** |
| Máximo absoluto de erro LSE Q8 modelado | 0,0600–0,1378 |

A comparação entre os **máximos** modelados e os máximos do JSON medido
difere no máximo em `6,66e-5` OUT e `5,36e-6` LSE. Portanto há suporte muito
forte para atribuir o residual à aproximação Q8, não a um novo erro de layout.
O arquivo fornecido tem inclusive um caso além do resumo 0,86: decode append,
live=262/nq=5, OUT=0,99939075 medido, 0,99939875 modelado; RMS do oráculo
1,70767 e mediana absoluta 0,44649. Não é um valor invisível na escala típica.
O máximo de erro/RMS não é erro relativo L2; não confundo essas métricas.

Exemplo reprodutível: boundary 256/nq2, RMS=1,38931, mediana absoluta=0,22842,
erro OUT=0,19857, LSE=0,11086, erro relativo L2=0,6297%. A existência de
outliers de OUT de 48 não torna um erro de 0,20 em outros componentes aceitável.
Esse conjunto é sintético e adversarial em escala/sinais/outliers, não uma
amostra de ativações reais do Qwen. O gate anterior de erro incremental versus
cache de 4 bits é informativo, mas não substitui um teste de logits/KLD.

**Decisão:** inaceitável como implementação numericamente equivalente;
bloqueado para promoção a padrão. Pode continuar como experimento explicitamente
aproximado. Não há evidência suficiente para declarar que perde qualidade de
texto de forma intolerável, nem para declarar que o erro é inofensivo.

Para logits, se `δo` é o erro de atenção, a projeção produz `W_O δo` (e pode
haver gating); os blocos seguintes produzem aproximadamente `J δo` localmente.
Sem esses pesos, ativações e Jacobiano, não existe número honesto de erro
nos logits. Nem RMSNorm garante apagá-lo. Um limite seria
`||δlogits||∞ <= ||J||₂→∞ ||δo||₂`; se a margem entre dois logits é menor
que duas vezes um limite de perturbação, o argmax pode mudar. A medição que
falta é A/B de logits/KLD, perplexidade e aceitação especulativa com modelo,
prompt e sampling fixos. Esta entrega não executa tais medições.

## Entrega e validação

`RKVA_MQ_I8_PREFIX_REDUCE=1` é opt-in, default 0 no fonte. O header INT8
commitado foi gerado com **1**, registrado em `Additional compiler options`.
A flag de runtime já existente `rkva-mq-i8.flag` continua sendo a seleção.
Regerar sem esse define volta ao XOR original defeituoso; não é rollback
seguro. Desligar a flag volta ao MQ2 de produção.

O interpretador agora executa `wavePrefixOp` com uma árvore de scan F32,
captura as três subtrações **reais do DXIL**, e pode rejeitar qualquer
`WaveReadLaneAt` com índice não uniforme. Uma mutação da primeira subtração
foi corretamente rejeitada. Resultados:

| Assembly interpretada | Scores | Erro máximo |
|---|---:|---:|
| INT8 prefix vs álgebra Q8 | 8.832 | 8,34e-6 |
| F32 prefix vs score de referência F32 | 8.832 | 2,38e-6 |
| Masked anterior, regressão do interpretador | 8.832 | 9,30e-6 |

Limite: scores do fast path, live=256/384, KV heads 0/3, duas queries e duas
waves. Não valida numeradores PV, merge/WHT final, todas as bordas ou ISA
do driver. Em particular, prefix aprovado na CPU **não prova execução
correta no Xbox**, assim como o teste anterior não provou o XOR.

MQ2/MQ4 foram recompilados com o DXC autorizado e comparados byte a byte
com os payloads anteriores; seus headers commitados não foram alterados.
O payload INT8 do header também é idêntico ao binário interpretado.

| Payload | Bytes | SHA256 |
|---|---:|---|
| MQ2 exato | 118452 | `9d81c3beba4e36756c89c6bc70c5807cedb990be67fba683fc714ff7a1c5b431` |
| MQ4 exato | 236896 | `ef2878458bd37b6f702fec19f36e674a836803d300915cf2604bfd801dfb48bb` |
| **INT8 prefix entregue** | **140356** | `a2b66069596fe3b8111dd19d99b4b65a652b2b2e376f44014015ade13f2c2d3f` |

Também passaram lane model (8 casos), verificação de hashes de todos os
headers gpupeak e `git diff --check`. Não foi realizado build Windows.

Reprodução local, sem console:

```bash
OPENBLAS_NUM_THREADS=1 python3 tools/xbox/tests/rkva_mq_i8_oracle_scale.py \
  --hardware .oc-artifacts/serial-i8masked-302.json \
  --output .oc-artifacts/astra-oracle-scale.json
DXC="$PWD/.oc-artifacts/dxc/dxc193.py" \
RKVA_EXTRA_DEFINES='-D RKVA_MQ_I8_PREFIX_REDUCE=1' \
  bash tools/xbox/scripts/compile-rkva-range-shaders.sh attn_kvarn4_mq2_i8
.oc-artifacts/dxc/dxc193.py -Fc -T cs_6_4 -E CSRangeAttnKvarn4Mq \
  -D RKVA_MQ_QT=2 -D RKVA_MQ_I8_PREFIX_REDUCE=1 \
  tools/xbox/shaders/rkva/rkva_range_attn_kvarn4_mq_i8.hlsl \
  -Fo .oc-artifacts/dxc/mq2_i8_prefix.dxil
OPENBLAS_NUM_THREADS=1 python3 tools/xbox/tests/rkva_mq_i8_dxil_model.py \
  .oc-artifacts/dxc/mq2_i8_prefix.txt --require-uniform-reads
python3 tools/xbox/tests/rkva_dxil_loop_cost.py .oc-artifacts/dxc/mq2_i8_prefix.txt
```

## Provas mínimas SM6.5/6.6 e próximo passo do Claude

`gpupeak_wave_caps.hlsl` e quatro headers novos estão integrados à criação
de PSOs do gpupeak. São kernels mínimos com store UAV, sem tipos F16 nativos,
para não misturar a rejeição já conhecida com a capability de wave:

| Nome no JSON | Profile/operação | DXIL bytes |
|---|---|---:|
| `wave_prefix64` | cs_6_4, PrefixSum + quatro broadcasts constantes | 3636 |
| `wave_multi65` | cs_6_5, MultiPrefixSum com máscaras fixas | 3532 |
| `wave_match65` | cs_6_5, WaveMatch | 3156 |
| `wave_quad66` | cs_6_6, QuadRead X/Y/Diagonal | 3728 |

O código registra uma linha `suite=capability` com
`variant=pso_accepted_not_executed` ou `pso_rejected`; falha inclui HRESULT
em `pso_errors`. **Não faz dispatch dessas sondas** e não inventa throughput
ou checksum. Uma aceitação responde à pergunta de PSO, não à correção
numérica da operação, nem à disponibilidade de todo SM6.5/6.6.
Os targets, defines e hashes estão em `gpupeak-shaders.json`.

```bash
python3 tools/xbox/scripts/compile-gpupeak-shaders.py \
  --dxc "$PWD/.oc-artifacts/dxc/dxc193.py" \
  --dxc-version 'Windows SDK 10.0.26100.0 x64/dxc.exe via isolated dxc193.py; version not queried' \
  --only wave_prefix64 wave_multi65 wave_match65 wave_quad66
```

O Claude deve recompilar o pacote no seu fluxo e:

1. Com INT8 prefix selecionado, rodar o mesmo serial-prefix probe, seed
   20261266, mode=2, tail=128, `--full`. A tolerância F32 original deve
   continuar reprovando Q8; não alterá-la para fabricar aprovação. Procurar
   concordância por caso com o masked/oráculo Q8 e ausência de erros de
   dezenas, NaN ou divergências novas. Capacity edge continua obrigatório.
2. Se concordar, repetir OFF/ON nas mesmas condições do sweep 28.672/nq256,
   confirmando bucket `kvarn4_mq2_i8`. **99,93 ns é o alvo; 87,80 é somente
   hipótese**. Se não ganhar, o prefix não salvou o desempenho nesta máquina.
3. No gpupeak opt-in já existente, ler as quatro linhas de capability.
   MultiPrefix aceito permitiria um braço futuro com menos cancelamento;
   ainda exigiria probe numérico. Quad aceito também exige testar o mapa e
   a execução antes de construir um kernel LDS dependente dele.
4. Mesmo que prefix ganhe, manter INT8 fora do padrão até qualificar qualidade
   com modelo real. Se equivalência ao oráculo atual for requisito rígido,
   **este Q8 está reprovado** e o resultado de velocidade não muda a decisão.

Não há evidência para encerrar todas as possibilidades de INT8; há evidência
suficiente para encerrar a alegação de que o braço atual já é uma otimização
correta e qualificada de produção.

## Commits da entrega

- `a1000026c`: braço prefix opt-in, interpretador e auditoria de escala Q8.
- `abee01d7f`: header INT8 prefix com define e proveniência registrados.
- `e29a144fe`: quatro provas de aceitação PSO no gpupeak, fontes e headers.
- Este relatório é registrado em commit documental separado.
