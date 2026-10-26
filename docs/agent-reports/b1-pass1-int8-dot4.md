# Relatório B1 — `pass1` de prefill com INT8 `dot4` (`attn_kvarn4_mq2_i8`)

Branch: `perf/rkva-pass1-int8-dot4-20261026`, clone
`/home/hjotha/beellama-agents/int8-pass1`, base = integração
`integ/rkva-xbox-agents-20261026` (main `5322461e5` + A1 + A2 + A3).

Nada foi executado em hardware: sem Xbox, sem RTX 4070, sem `llama-server`, sem
CUDA. Tudo abaixo é compilação DXC isolada na máquina 193 (diretório próprio
`C:\Users\hjotha\build\oc-int8-pass1-20261026\`), contagem de instruções sobre o
DXIL, emulação NumPy e um TU de teste CPU compilado com `g++` neste host.

## 0.1 Correção pós-hardware (pacote 297) — leia primeiro

O primeiro build rodou no console e **estava numericamente quebrado**, embora
rápido: o PSO `attn_kvarn4_mq2_i8` foi selecionado e deu `pass1` **−30,7%**
(100,0 → 69,3 ns/linha), mas o oráculo NumPy do probe reprovou **40 de 40** casos
`ptt=128` (os `ptt=0` passam porque não usam o tile mq2), todos no bucket
`kvarn4_mq2_i8`, com `out_max_abs` 31,9–77,9 e `lse_max_abs` 414–831. O braço OFF
deu `out_max_abs` 1e-5..5e-5 e `lse_max_abs` 2e-6..5e-6.

**Causa raiz:** a constante por slice `C_s` era reduzida sobre as 8 lanes do
slice *antes* de entrar na soma, então cada uma das 8 lanes somava o `C_s`
inteiro e a redução de 16 lanes colhia `8*C_0 + 8*C_1` em vez de `C_0 + C_1` —
sete constantes a mais por slice. Não é erro de quantização: é um offset
aditivo no score, e é por isso que o `lse` errava por centenas enquanto a
direção da saída se mantinha.

**Por que passou:** `rkva_mq_i8_quality.py` valida a *álgebra* (cada score como
uma soma de 128 dims com `C_s` somado uma vez). Ela não consegue ver como o HLSL
distribui essa álgebra entre lanes. A emulação era do algoritmo, não do
programa — exatamente o risco que o relatório §7 não listou.

**Correção:** `zs`, `qsum` e `sum(Q8)` passam a ficar como parciais por lane; só
`mx` (que define `sq_s`, o multiplicador do dot parcial de cada lane) continua
reduzido no slice. A forma correta é também mais barata: saem 2 reduções × 3
passos por (query, head) do quantizador. DXIL novo: **141 612 B**
(sha256 `85b2e376a9ddd807…`, era 143 620 B).

**Prova (`tools/xbox/tests/rkva_mq_i8_lane_model.py`)**: emulador NumPy que
executa o HLSL **lane a lane** — 2 waves de 64 lanes, mapeamento 16 dims/16
lanes, quantizador por grupo, `dot4add_i8packed` como int8 COM sinal × int8 com
sinal e wrap int32, redução `lane ^ {8,4,2,1}`, broadcast
`WaveReadLaneAt(val, t*16)`, e a aplicação por lane de `q8scale`/`q8const`/
`other`. Ele reproduz a falha e valida a correção em 8 casos:

```
$ python3 tools/xbox/tests/rkva_mq_i8_lane_model.py
tight live   256 fast=[0,128)  algebra out 5.53e-03 lse 3.67e-03
    shipped  "slice"        out    8.471 lse    71.93 bias -6.29e+00   (  1531x algebra)
    fix      per_lane       out 5.53e-03 lse 3.67e-03 bias -1.30e-03   (  1.00x algebra)
    fix      slice_split    out 5.53e-03 lse 3.67e-03
model live  1920 fast=[0,1792) algebra out 5.54e-02 lse 6.46e-02
    shipped  "slice"        out   23.394 lse   184.61 bias +1.82e-01   (   422x algebra)
    fix      per_lane       out 5.54e-02 lse 6.46e-02 bias +1.97e-04   (  1.00x algebra)
    fix      slice_split    out 5.54e-02 lse 6.46e-02
rkva_mq_i8_lane_model: 8 cases, PASS
```

O modelo também confere que a forma "reduz e divide por 8" (`slice_split`) é
numericamente equivalente à forma por lane — o que fecha o diagnóstico — e que
toda linha rápida é visitada **exatamente uma vez** pelas duas waves.

Antes de confiar no modelo, ele foi cruzado contra a álgebra já validada: o modo
`per_lane` reproduz `rkva_mq_i8_quality.i8_scores` com rms 4,3e-7 (só
arredondamento F32), e o modo `slice` diverge dela com rms 30,7 e viés −10,5.
O primeiro rascunho do emulador tinha um bug próprio (montava os dwords de
nibble com os dims da lane 0); foi esse cruzamento que o expôs, não a inspeção.

`mq2`/`mq4` continuam **byte-idênticos** (`9d81c3beba4e3675…` /
`ef2878458bd37b6f…`) e a contagem de instruções por linha não mudou (331,9),
porque o quantizador fica fora do laço medido. Pendente: recompilar o pacote e
rodar oráculo + sweep no console.

## 0.2 Segunda rodada de hardware (pacote 299) — defeito NÃO resolvido

O pacote 299 (correção da constante por slice) melhorou e **continua quebrado**:
o oráculo passa todos os casos cujo tile rápido de registros fica vazio
(`live <= 129`) e reprova os 40 casos `ptt=128` que entram nele — `out_max_abs`
15,7–69,3 e `lse_max_abs` 12,4–50,5 (antes 31,9–77,9 / 414–831), sempre no bucket
`kvarn4_mq2_i8`.

**Não encontrei o defeito nesta rodada.** O que consegui descartar com evidência:

1. **O caminho de score INT8 está provadamente certo.** O emulador lane a lane
   reproduz a álgebra de 128 dims com 1,8e-7 sobre os registros *do próprio
   probe*, e a álgebra reproduz o decode do registro com 0,016 de rms em scores
   de rms 2,3 (0,7%).
2. **A reestruturação fora do tile está certa.** Os casos `live <= 129` passam:
   neles `fast_hi <= fast_lo` e o bloco INT8 nunca executa — então remover a Q
   rotacionada residente, o laço por grupo e o reload direto de Q no laço
   residual estão validados em hardware.
3. **O artefato compilado é o fonte corrigido**: 229 `waveReadLaneAt` = 192 do
   tile + 36 de um único reduce de 3 passos no quantizador + 1, ou seja só `mx`
   continua reduzido.
4. **Não é multiplicidade da constante.** Varredura k = 1,5…8 (tabela em
   `docs/rkva-prefill-int8-dot4.md`) dá lse 13→246 e out saturado em 48,3; o
   console dá 15,70/16,06.
5. **Não é slice errado** para `other`/`scale`/`zero`/nibbles, nem base de dim
   errada para a Q, nem redução de 64 lanes, nem broadcast da lane 0, nem linha
   trocada no PV (tabela idem).

O modelo tem histórico preditivo: para o defeito *anterior* (constante ×8) ele
previu 59,2/260 em `boundary 256 nq=9` onde o console mediu 54,1/414. Agora ele
prevê 0,199/0,111 onde o console mede 15,70/16,06 — um fator 79 que nenhuma
perturbação de score que eu consiga construir reproduz.

**Bisect de uma rodada** (entregue): reconstruir o *mesmo* PSO com o mapeamento
16 dims/16 lanes, o laço por grupo, o caminho V/PV e o softmax intactos, mas com
score exato em F32 (`RKVA_MQ_I8_F32_PROBE=1`). Se ficar exato no console
(~1e-5), o mapeamento e a reestruturação estão certos e o problema é a
aritmética int8; se reproduzir 15–70, o mapeamento ou o PV desse bloco é o
culpado. Qualquer um dos dois resultados corta o espaço de busca pela metade.
Comando em `docs/rkva-prefill-int8-dot4.md` §"One-run bisect".

**Recomendação:** não ligar `rkva-mq-i8.flag` em build de usuário até fechar
isto. O ganho medido (−30,7% em `pass1`) é real, mas a variante não está pronta.

## 0. Veredito curto

* A variante existe e compila: PSO separado `attn_kvarn4_mq2_i8`, `cs_6_4`,
  143 620 B, flag `rkva-mq-i8.flag` default OFF.
* Custo medido no modelo de issue por linha de KV: **512,0 → 331,9
  instruções/lane (−35,2%)**.
* Exatidão: **não é bit-exata**. O erro que ela adiciona é **6,2–7,6% do erro
  de score que o próprio KVarN4 4-bit já introduz**; erro total de score cresce
  ≤1,2%, e saída/LSE/pesos de softmax ficam dentro de 1,15× do caminho de
  produção em 10 casos (PASS). Isso valia para a **álgebra**; o primeiro build
  no console reprovou por um bug de distribuição lane↔slice, corrigido em §0.1 e
  agora coberto por um emulador lane a lane.
* O ganho **não** vem do `dot4` em si (medido: o MAC é o mesmo), vem de o int8
  permitir 16 dims/lane, que encolhe o disco de redução cross-lane de 64 para 16
  lanes. Essa é a conclusão estrutural mais importante da tarefa.
* Pendente de hardware: criação real do PSO no console, VGPR/ocupação, `pass1_us`
  e erro numérico medidos — script inerte entregue.

## 1. Caminho quente e álgebra exata (Passo 1)

O `pass1` de prefill em produção é o tile de 2 queries
(`RKVA_MQ_QT=2` → PSO `attn_kvarn4_mq2`, bucket `kvarn4_mq2_fine`), 97,8% do
tempo de GPU por segmento de 8 queries segundo o relatório A2. `numthreads(128)`
= 2 waves de 64; `u = lane` cobre 4 dims; redução `WaveActiveSum` de 64 lanes por
score.

### Fórmula exata de K/V dequantizado

Layout do registro revision 3 (8960 B por (grupo, kv_head, slice)), confirmado no
shader (`load_record4` / laço rápido) e no store (`rkva_store_kvarn4.hlsl`):

```
K: nibbles [dim][pos]   em base+0 .. 8191   (nibble(d,p) em byte dim*64 + p/2)
   scale_k[dim]         em base+8192        (F16, por dim = por linha do plano)
   zero_k[dim]          em base+8448        (F16, por dim)
   other_k[pos]         em base+8704        (F16, por posição = por coluna)
V: nibbles [pos][dim]   em base+0 .. 8191
   scale_v[pos], zero_v[pos]  (por posição)
   other_v[dim]                (por dim)
   -> V(d,p) = (nibble * scale_v[p] + zero_v[p]) * other_v[d]

K(d,p) = (nibble(d,p) * scale_k[d] + zero_k[d]) * other_k[p]         (*)
```

Slice `s = dim >> 7` (0 ou 1) seleciona o blob; **`other` é por slice**, porque
`rkva_store_kvarn4.hlsl` grava `best_col` do próprio slice
(`o0 = ggml_half_bits(sh_best_col[r0])`). Isso é decisivo: não dá para fatorar um
`other` único para os dois slices.

Estágio: linhas F16 do grupo vivo, já rotacionadas (`load_stage4`).
Tail: anel ORIGINAL-F16 de 256 linhas (`load_tail4`), domínio original.
Os dois continuam no caminho F32 exato — a variante não os toca.

### Dobrar a escala de coluna na Q e quantizar para int8

`scale_k` é por (grupo, dim). Uma vez por (query, head, grupo) e por slice:

```
Qr      = WHT(Q)                                  (já pré-computada pelo host)
Q'      = Qr * scale_k[d]                         F32
sq_s    = max_{d in s} |Q'| / 126                 (126, não 127: |Q'| <= 126*sq_s
                                                   garante que o arredondamento
                                                   nunca precise de clamp)
Q8      = clamp(round(Q' / sq_s), -127, 127)      int8
Z_s     = sum_{d in s} Qr[d] * zero_k[d]          F32, com a Q NÃO quantizada
Qsum_s  = sum_{d in s} Q'
```

Substituindo em (*):

```
score = kq * sum_s other_s[p] * ( sq_s * sum_{d in s} Q8[d]*nibble(d,p)
                                + C_s )
C_s   = Z_s + 8*Qsum_s - 8*sq_s*sum_{d in s} Q8
```

Por que `C_s` tem esse formato: o resíduo do arredondamento é
`δ_d = Q'_d − sq_s*Q8_d`, e

```
sum_d Q'_d*nibble_d = sq_s*sum_d Q8_d*nibble_d + sum_d δ_d*nibble_d
                    = sq_s*sum_d Q8_d*nibble_d + 8*sum_d δ_d + sum_d δ_d*(nibble_d − 8)
```

`8*sum δ_d = 8*(Qsum_s − sq_s*sum Q8)` é **exato** e entra em `C_s`; só sobra
`sum δ_d*(nibble_d − 8)`, cuja variância é ~3,6× menor que `sum δ_d*nibble_d`
porque o nibble tem média ≈7,5 e não 0. Medido: reduz o erro adicionado de
0,12–0,24× para 0,062–0,076× do erro 4-bit (ver §3).

### `dot4add_i8packed` + termo de zero

Nibble ∈ [0,15] já é um lane int8 não-negativo válido, então **não há termo de
offset**: `dot4add_i8packed(k8, q8, acc)` acumula 4 MACs int8 por instrução em
int32, exato. Pior caso de score: `256*15*126 = 483 840`, muito dentro de int32
(`check_dp4a` do gate prova a cadeia exata e o limite). O termo de zero `Z_s`
continua F32 e nunca é quantizado; só ele é somado via `C_s`.

### Tail / estágio F16 e softmax

Tail e estágio **não** entram na variante: continuam `load_tail4`/`load_stage4` +
`WaveActiveSum(dot(qa,k))` em F32, como no kernel qualificado. O softmax também é
o mesmo código (meia-linha de 4 posições, `maximum`/`denom`/`body`), assim como
o WHT de corpo, o merge por wave, os splits e o layout de parciais
`[split][query*n_head + head][258]`. O host não muda.

### O que a medição derrubou (importante)

A ideia ingênua "trocar `dot4` por `dot4add_i8packed` no layout atual" **não
ganha nada**: ambos aposentam 4 MACs por lane por instrução, então as mesmas 12
instruções por lane cobrem os mesmos 96 scores, e o layout do registro
(`[dim][pos]` para K, nibble 2-por-byte) obriga uma transposição de ~14 ops por
4 dims — mais caro que o dequant F32 que substitui. O ganho real só aparece
quando o int8 permite **16 dims por lane**: aí a redução cai de 6 para 4 passos
de butterfly e quatro linhas de KV reduzem no mesmo fluxo de instruções.

## 2. Variante implementada (Passo 3)

`RKVA_MQ_I8=1` (wrapper `rkva_range_attn_kvarn4_mq_i8.hlsl`) → PSO separado
**`attn_kvarn4_mq2_i8`**, `cs_6_4`, default OFF atrás de
`rkva-mq-i8.flag`.

Mapeamento: `lane & 15` = slot de dim, `lane / 16` = linha de KV do half tile.
Cada lane cobre 16 dims consecutivas, que ficam dentro de um único slice de 128
dims (16 divide 128), então `scale`/`zero`/`other` vêm de um único registro. As
quatro lane-groups de uma wave64 pontuam quatro linhas de KV ao mesmo tempo:

* empacotamento dos nibbles: 4× `int8x4` por lane por linha (11 ops por 4 dims),
  compartilhado pelos 12 (query, head) pares;
* 4 `dp4a` + redução de 16 lanes (4 passos) + 4 broadcasts por (query, head);
* quantização da Q movida para um laço externo por grupo (128 linhas), fora do
  laço quente: substituí-la por constantes muda o laço medido em 37 de 4 019
  instruções.

Seleção no engine: `remote-attn.cpp` lê o flag no HELLO,
`RemoteAttnEngine::set_mq_i8()`, e `attend_range()` prefere o PSO i8 ao
`attn_kvarn4_mq2` quando o flag está ligado. Bucket próprio
`…optimized-kvarn4_mq2_i8/…`, contador `kvarn4_multi_query.mq2_i8.calls` e
`"exact":false` no JSON de perfil.

### Byte-identidade dos alvos existentes (obrigatória)

Payload DXIL (não o comentário do header), HEAD vs regenerado com o mesmo DXC:

```
rkva_range_attn_kvarn4_mq2_dxil.h  bytes 118452  sha256 9d81c3beba4e36756c89c6bc70c5807cedb990be67fba683fc714ff7a1c5b431
rkva_range_attn_kvarn4_mq4_dxil.h  bytes 236896  sha256 ef2878458bd37b6f702fec19f36e674a836803d300915cf2604bfd801dfb48bb
rkva_range_attn_kvarn4_mq2_i8_dxil.h bytes 143620 sha256 e0e47250c2c9b6e4fbff01fdfb9860391aafc1a916ab55e05e99122e4af2eabc
```

Os dois alvos qualificados continuam idênticos; `mq2` e `mq4` foram regenerados
e comparados byte a byte (`identical= True`).

### Contagem de instruções por linha

`tools/xbox/tests/rkva_dxil_loop_cost.py` (modelo de issue RDNA2 wave64, uma
iteração = tile de 8 linhas de KV, 2 queries × 6 heads):

```
== mq2 baseline
   per KV row:  TOTAL 512.0   fma_class 237.0   reduce 144.0   sfu 15.0   int 89.1
   useful FMA per row (QK+PV)  96.0  => 40.5% of fma_class, 18.8% of TOTAL

== mq2_i8
   loop header %4419  body blocks 61  dot-units/iter 96 (dot4 0, dp4a 96)  rows/iter 8
   per KV row:  TOTAL 331.9   fma_class 198.2   reduce 24.0   sfu 15.0   int 81.6
   useful FMA per row (QK+PV)  96.0  => 48.4% of fma_class, 28.9% of TOTAL
```

Decomposição do −180,1 instr/linha:

| item | baseline | i8 | delta |
| --- | ---: | ---: | ---: |
| reduce cross-lane | 144,0 | 24,0 | **−120,0** |
| fma class (dot + dequant + PV) | 237,0 | 198,2 | −38,8 |
| int/misc | 89,1 | 81,6 | −7,5 |
| cbuf/constantes | 12,1 | 0,25 | −11,9 |
| loads | 2,25 | 3,5 | +1,25 |
| **TOTAL** | **512,0** | **331,9** | **−180,1 (−35,2%)** |

A redução responde por dois terços do ganho. A queda de `cbuf` é efeito colateral
da reestruturação laço-por-grupo (o compilador hoistou as cargas de constante
para fora do laço interno) e está contada separadamente para não inflar a
atribuição ao `dp4a`.

Ferramenta: precisei estender `rkva_dxil_loop_cost.py` — ela só conhecia
`dx.op.dot4.f32`, e o DXC emite `dx.op.dot4AddPacked.i32`; além disso a detecção
de laço natural era frouxa (sem reachability reversa) e escolhia o laço
envolvente. Agora conta `dot4AddPacked`, `waveReadLaneAt` (1 slot) e
`rawBufferLoad/Store` (SM 6.4 troca `bufferLoad` por `rawBufferLoad`), e escolhe
o menor corpo que cobre o tile inteiro.

### Medições de apoio

* Build de medição sem o quantizador (`RKVA_MQ_I8_MEASURE_NOQUANT=1`, nunca
  embarcado): laço medido muda 4 019 → 3 982 instruções, provando que a
  quantização por grupo está fora do laço quente.
* A elisão de rescale do A2 foi deliberadamente **não** combinada: com os dois
  braços de softmax presentes o laço estático sobe de 331,9 para 502,4
  instr/linha (medido), mais do que a elisão pode economizar.

## 3. Emulação de qualidade / gate (Passo 2)

`tools/xbox/tests/rkva_mq_i8_quality.py` (NumPy, CPU). Três caminhos sobre o
mesmo store sintético-realista:

* **GOLD** — atenção float64 sobre K/V originais arredondados para F16 (o que um
  cache F16 exato devolveria).
* **REF** — aritmética do kernel embarcado: decode `precise`
  (`float(q)*scale + zero`, depois `*other`), dot F32 com butterfly de 64 lanes,
  softmax online em meias-linhas de 4, corpo pelo WHT, tail F16 no domínio
  original.
* **I8** — a variante, com dot int32 exato por slice e `C_s` do §1.

Duas distribuições de store: `tight` (escala por linha 10^±0,4) e `model` (a
mesma distribuição do oráculo do repo, `10^U(-1.5,0.7)`, linhas com sinal e 3 dims
outlier ×6), prefixos 256/384/640/1920/4096, 8 queries cada. Métrica de registro:
`add/ref` = erro do arredondamento Q8 dividido pelo erro de score que o 4-bit já
introduz.

```
$ nice -n 19 python3 tools/xbox/tests/rkva_mq_i8_quality.py --json .oc-artifacts/i8-quality.json
tight live   256 dc=True  add/ref 0.076 total/ref 0.999 | rel_o_rms 9.844e-02->9.648e-02  1-cos 6.235e-03->5.960e-03  lse 1.395e-02->1.388e-02  wL1 5.522e-01->5.522e-01  PASS
model live   256 dc=True  add/ref 0.076 total/ref 1.012 | rel_o_rms 1.925e-01->1.896e-01  1-cos 2.453e-02->2.393e-02  lse 1.523e-01->1.491e-01  wL1 5.649e-01->5.653e-01  PASS
tight live   384 dc=True  add/ref 0.070 total/ref 1.002 | rel_o_rms 1.855e-01->1.854e-01  1-cos 1.321e-02->1.331e-02  lse 5.921e-02->5.855e-02  wL1 2.546e-01->2.540e-01  PASS
model live   384 dc=True  add/ref 0.071 total/ref 1.003 | rel_o_rms 2.325e-01->2.240e-01  1-cos 1.629e-02->1.683e-02  lse 2.622e-01->2.573e-01  wL1 3.970e-01->3.978e-01  PASS
tight live   640 dc=True  add/ref 0.062 total/ref 1.002 | rel_o_rms 2.050e-01->2.051e-01  1-cos 1.746e-02->1.766e-02  lse 3.179e-02->3.206e-02  wL1 2.085e-01->2.084e-01  PASS
model live   640 dc=True  add/ref 0.066 total/ref 1.003 | rel_o_rms 2.574e-01->2.540e-01  1-cos 1.032e-02->1.042e-02  lse 9.716e-02->9.752e-02  wL1 5.136e-01->5.137e-01  PASS
tight live  1920 dc=True  add/ref 0.071 total/ref 1.003 | rel_o_rms 1.822e-01->1.843e-01  1-cos 1.624e-02->1.672e-02  lse 2.678e-02->2.643e-02  wL1 1.373e-01->1.386e-01  PASS
model live  1920 dc=True  add/ref 0.069 total/ref 1.004 | rel_o_rms 1.735e-01->1.698e-01  1-cos 1.749e-02->1.617e-02  lse 3.461e-01->3.328e-01  wL1 1.142e-01->1.113e-01  PASS
tight live  4096 dc=True  add/ref 0.074 total/ref 1.003 | rel_o_rms 2.082e-01->2.068e-01  1-cos 1.767e-02->1.753e-02  lse 1.780e-02->1.762e-02  wL1 1.289e-01->1.295e-01  PASS
model live  4096 dc=True  add/ref 0.074 total/ref 1.003 | rel_o_rms 2.091e-01->2.131e-01  1-cos 1.346e-02->1.372e-02  lse 2.629e-01->2.666e-01  wL1 2.119e-01->2.151e-01  PASS
rkva_mq_i8_quality: 10 gated cases (dc exact, 8 queries each), PASS
```

As linhas `dc=False` (sem o termo `C_s`) rodam no mesmo harness e mostram
`add/ref 0.120 .. 0.202` — é a evidência de que o termo de DC vale ~2,5×.

### Critério de aceitação proposto (e por que passa)

O critério não é "erro absoluto pequeno" — o cache já é 4-bit, então a régua tem
de ser o erro que o produto já aceita:

1. **primário** `add/ref ≤ 0,35` — o arredondamento Q8 adiciona no máximo um
   terço do erro de score que o KVarN4 4-bit já introduz. Medido: 0,062–0,076,
   ou seja ~5× de folga.
2. **total** `score_ratio ≤ 1,15` — erro total de score (variante vs F16) não
   cresce mais de 15%. Medido: 0,999–1,012.
3. saída, `max|O|`, `1−cos`, `max|ΔLSE|` e L1 dos pesos de softmax ≤1,15× do REF.
   Todos passam; vários ficam *melhores* que o REF por ruído de amostragem
   (o que é esperado: a perturbação é ~7% do erro já existente).

O harness também prova, independentemente:
`check_dp4a` — cadeia de 4 MACs int8 = soma int64, 64 trials, e o pior score
(483 840) cabe em int32.

## 4. Validação offline executada

1. **DXIL na 193** (`dxc.exe` 1.8.2502.11, Windows SDK 10.0.26100.0). Baseline
   reproduzido byte a byte (sha256 `9d81c3be…`), alvo novo compilado
   (`cs_6_4`, 143 620 B), contagens acima.
2. **Gate de qualidade** — 10 casos, PASS (saída literal em §3).
3. **Teste CPU do perfil/JSON** (TU único, `g++ -O1`, `nice -n 19 ionice -c3`),
   incluindo o bucket novo e o campo `mq2_i8` do JSON:

```
$ g++ -std=c++17 -Wall -Wextra -Werror -O1 tools/xbox/tests/rkva_range_profile_cpu.cpp -o .oc-artifacts/bin/rkva_range_profile_cpu
$ .oc-artifacts/bin/rkva_range_profile_cpu > .oc-artifacts/bin/profile.json
rkva_range_profile_cpu: ok
$ python3 -m json.tool .oc-artifacts/bin/profile.json > /dev/null && echo JSON_OK
JSON_OK
```

   JSON contém `"mq2_i8":{"available":true,"status":"ready","calls":11,"exact":false}`
   e o bucket `…/optimized-kvarn4_mq2_i8/…` é coberto por asserção nova.
4. **Script de hardware em modo preview** (sem `--execute`), exit 4 — §5.

## 5. Script para a janela de hardware (Passo 4)

`tools/xbox/scripts/run-rkva-prefill-int8-ab.py`:

* **Inerte por padrão**: preview JSON, exit 4, não cria processo nem artefato.
  O preview funciona mesmo com build alheio rodando (só o `--execute` recusa).
* **Recusa** (exit 2) se `pgrep -af 'llama-server|bench-rkva|test-rkva|ninja|cmake --build|rkva_kvarn_serial'`
  achar processo alheio, ou se faltar argumento/binário.
* Não instala nada: pressupõe o pacote com `attn_kvarn4_mq2_i8` já instalado pelo
  Claude; usa `--enable-cmd`/`--disable-cmd` para criar/remover
  `rkva-mq-i8.flag` e **sempre restaura OFF** no `finally`.
* Sequência por braço: `rkva_kvarn_serial_prefix_probe.py --execute --modes 2
  --precision-tails 128` (erro OUT/LSE contra o oráculo NumPy — o gate numérico,
  porque a variante não é exata e um replay por hash falharia por construção),
  depois `bench-rkva-range.py --execute` (fine + FP8 mapped + tcp-prefetch +
  profile).
* Coleta: `pass1_us` mean/p50/p95 e `split_k`/`active_rows` dos buckets
  `…mq2_fine…` vs `…mq2_i8…`, `kvarn4_multi_query.mq2_i8.{available,calls}`, os
  256 IDs gerados dos dois braços (reportados, **não** exigidos iguais), e a
  razão `out_error_ratio`/`lse_error_ratio`.
* Status `pass` só se: o braço ON usou o kernel (`calls > 0`), os dois buckets
  existem, a sonda mediu nos dois braços, `out_error_ratio ≤ 1,15` e
  `pass1_ratio < 1,0`. Tudo em JSON (`--json`).

Preview real executado (com o build CUDA do Claude ativo no host, que o script
detectou e registrou):

```
$ python3 tools/xbox/scripts/run-rkva-prefill-int8-ab.py --json .oc-artifacts/ab-int8-preview.json
exit=4
mode preview | status preview | foreign 3
arms ['off', 'on', 'restore']
"kernel_on": "attn_kvarn4_mq2_i8 / bucket attention/kvarn4-prefix/optimized-kvarn4_mq2_i8/nq2-16/r16k-32k/warm",
"kernel_off": "attn_kvarn4_mq2 / bucket attention/kvarn4-prefix/optimized-kvarn4_mq2_fine/nq2-16/r16k-32k/warm",
"exact": false,
```

## 6. Estimativa de ganho (ESTIMATIVA, não medida)

Base: `pass1` = 2 725,7 µs/segmento (bucket A2, 4 112 amostras), 97,8% do
segmento, kernel limitado por issue a 86% de eficiência, 150 t/s de prefill XXXL.

| cenário | instr/linha | `pass1`/segmento | envelope 256q×1 camada | prefill XXXL |
| --- | ---: | ---: | ---: | ---: |
| atual (medido A2) | 512,0 | 2 725,7 µs | 65,3 ms (briefing) | ~150 t/s |
| variante (derivado) | 331,9 (−35,2%) | ≈1 800 µs | ≈43 ms | ≈215–230 t/s |

Ganho estimado **−30% a −35% no `pass1`**, **+35% a +50% no prefill XXXL**, com
`reduce`/`unpack`/`pack` e o resto do pipeline inalterados. É derivação de
contagem de instruções; **não há medição de hardware nesta entrega**. Se o
kernel deixar de ser puramente issue-bound ao cair para 332 instr/linha (por
exemplo se a ocupação despencar por VGPR), o ganho real será menor.

## 7. Pendências de hardware / riscos

Pendente medir:

1. **Criação do PSO `attn_kvarn4_mq2_i8` no console.** A sonda A1 aceitou um PSO
   `dot4_i8` isolado, mas este PSO completo (143 620 B de DXIL, SM 6.4) nunca foi
   criado. Se `CreateComputePipelineState` falhar, o engine reporta
   `mq2_i8.status` e cai no `attn_kvarn4_mq2` qualificado — falha fechada.
2. `pass1_us` OFF vs ON e `out_error_ratio` (script §5) — confirma ou refuta a
   estimativa de §6.
3. **VGPR/ocupação reais.** `q8[12]` custa 48 VGPRs (compensado por não manter a
   Q F32 rotacionada residente, −48), então a pressão arquitetural fica perto da
   do kernel atual, mas isso é estimativa: a alocação final é do compilador do
   driver. Se a ocupação cair, o próximo passo é processar os 12 (query, head)
   pares em dois lotes de 6, aceitando repack do K (14 ops/linha a mais).
4. Paridade de logits/IDs no mesmo prefixo congelado (o script compara IDs e
   reporta a primeira divergência; não exige igualdade, porque a variante não é
   exata).
5. Recompilar o pacote com o DXIL corrigido (141 612 B, sha256
   `85b2e376a9ddd807…`) e repetir `rkva_kvarn_serial_prefix_probe.py --modes 2
   --full`: esperado `out_max_abs` na ordem de 1e-2 a 1e-1 e `lse_max_abs` de
   1e-2, contra 1e-5..5e-5 do braço OFF — a diferença é o erro de quantização Q8,
   não um defeito. O `pass1` deve continuar ≈ −30%.

Riscos:

* **Um modelo da álgebra não qualifica um kernel distribuído por lanes.** O
  defeito de `C_s` passou por todos os gates offline do B1 e só caiu no console.
  `rkva_mq_i8_lane_model.py` é agora parte do gate, não um extra.
* **Não é bit-exata.** Não pode ser default. O flag, o bucket próprio e o
  `"exact":false` no JSON existem para que nenhum número seja atribuído por
  engano ao caminho qualificado.
* O flag é lido no HELLO a partir de `g_snapshot_local_folder`; se a pasta
  estiver vazia nesse ponto o flag não é visto (mesmo comportamento de
  `rkva-fp8-gpu-off.flag`). Validar no primeiro run.
* O app UWP **não foi compilado** (sem MSVC nesta máquina, build UWP do Claude
  rodando na 193). As mudanças de C++ espelham o padrão do A2 (PSO opcional +
  status + contador + JSON) e o `BulkSegmentProfile` recebeu o campo `int8_dot`,
  mas **compilação do app é pendência**.
* A variante exige exatamente 64 lanes por wave (uma lane-group de 16 por linha
  de KV do half tile). Outros tamanhos envenenam a saída com NaN, como o gate de
  shape já fazia.

## 8. O que não fiz (e por quê)

* Não toquei em `ggml/`, `src/`, `common/`, nem no core CUDA — o diff é só
  app/shader/perfil/docs.
* Não rodei build completo, CUDA, `llama-server`, Xbox ou 4070.
* **Não fiz o PV em int8.** O eixo de soma do PV é a posição, e no registro V o
  dword é `[pos][dim]` — o eixo de 4 bytes do `dp4a` cairia sobre dims, não sobre
  posições. Transpor custaria mais que os 384 fma que substituiria, e quantizar
  os pesos de softmax em int8 (escala por (query,head,tile)) adicionaria erro
  direto na saída, com risco bem maior que o do QK. Registrado como rejeitado,
  com a álgebra acima.
* Não combinei a elisão de rescale do A2 (§2, medido: 331,9 → 502,4 estático).
* Não promovi nada a default.

## 9. Comandos de reprodução

```bash
# DXIL + assembly de qualquer alvo (usa DXC da 193 num diretório próprio)
python3 .oc-artifacts/dxc/dxc193.py -T cs_6_4 -E CSRangeAttnKvarn4Mq -D RKVA_MQ_QT=2 -Fc \
  tools/xbox/shaders/rkva/rkva_range_attn_kvarn4_mq_i8.hlsl -Fo .oc-artifacts/dxc/mq2_i8.dxil

# regenerar o header embarcado com o caminho documentado
DXC=.oc-artifacts/dxc/dxc193.py bash tools/xbox/scripts/compile-rkva-range-shaders.sh attn_kvarn4_mq2_i8

# custo por linha de KV do laço quente (baseline vs variante)
python3 tools/xbox/tests/rkva_dxil_loop_cost.py .oc-artifacts/dxc/mq2_chk.txt .oc-artifacts/dxc/mq2_i8.txt

# gate de qualidade INT8 (álgebra)
python3 tools/xbox/tests/rkva_mq_i8_quality.py --json .oc-artifacts/i8-quality.json

# gate do HLSL lane a lane (reproduz a falha do pacote 297 e valida a correção)
python3 tools/xbox/tests/rkva_mq_i8_lane_model.py --json .oc-artifacts/lane-model.json

# teste CPU de perfil/JSON
g++ -std=c++17 -Wall -Wextra -Werror -O1 tools/xbox/tests/rkva_range_profile_cpu.cpp -o /tmp/rkva_profile
/tmp/rkva_profile > /tmp/profile.json && python3 -m json.tool /tmp/profile.json > /dev/null

# A/B de hardware (inerte sem --execute)
python3 tools/xbox/scripts/run-rkva-prefill-int8-ab.py --help
```
