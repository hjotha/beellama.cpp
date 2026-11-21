> Historical QSA Q8 audit. Its findings were applied (alloc-size helper, Ampere gate, tile asserts, nq/ratio tests).
>
> Historical agent report (2026-11-21). Current status and numbers: docs/PLAN-strata-parity.md.

# Auditoria read-only — QSA esparsa com KV Q8_0 (Strata)

- Worktree: `/home/hjotha/worktrees/beellama-strata-parity`, branch `feat/strata-parity`
- Commit auditado: `5f46a18da` (`feat(cuda): dequantize QSA sparse attention from the Q8_0 KV cache`)
- Diff: `git diff a7a77084a..HEAD -- src/llama-graph.cpp ggml/src/ggml-cuda/{fattn.cu,fattn-mma-f16.cuh} tests/test-backend-ops.cpp`
- Nada foi editado/compilado/executado na GPU. Evidência de compilação apenas observada em artefatos existentes (ver §4, item 1).

## 1. Veredito

**Nenhum BLOCKER.** O loader Q8 está correto em offsets, blocos, strides, barreiras e seleção de kernel;
não achei leitura fora do bloco/array nem caminho em que o helper diga `true` e o kernel denso rode sem
buffer F16 (o sentido perigoso). Achei **um MAJOR** de contrato/alocação (o helper não espelha o dispatcher
para `Q->ne[1] ∈ {2,3,4}`; benigno em correção, caro em memória) e lacunas de teste concretas.

## 2. Respostas diretas

### Q1 — Offsets/unidades: OK, sem OOB

- K: `elem_offset = 2*k0_start` (elementos). `k0_start ∈ {0, nbatch_K2, 2*nbatch_K2, …}` e o macro de
  config tem `static_assert(nbatch_K2 % 4 == 0)` (`fattn-mma-f16.cuh:31`), logo `elem_offset % 8 == 0`.
- V: `elem_offset = i0_start = 2*nbatch_V2*i`, com `nbatch_V2 % 4 == 0` → também múltiplo de 8.
- Chunk: `k*8` elementos → `i0 = elem_offset + k*8` é **sempre múltiplo de 8**.
  `ib = i0/32`, `iqs = i0%32 ∈ {0,8,16,24}`; `ne=8` lê `qs[iqs..iqs+7]`, que nunca cruza o bloco de 32
  (pior caso `iqs=24` termina exatamente no último byte do bloco: `ib*34+34`). `x[ib].d` em `ib*34`.
- Limite superior: `i0_max = DKQ-8` (ou `DV-8`) → `ib ≤ 7` para D=256, e a linha tem exatamente 8 blocos
  (272 B). Para D=256/QSA os configs Ampere/Turing/CDNA/RDNA têm `nbatch_K2 = nbatch_V2 = 128`, então na
  prática `elem_offset == 0`; a fórmula continua correta para configs menores.
- `dequantize_V_q8_0<half,8>` (`fattn-common.cuh:1108`) tem `static_assert(ne%2==0)` e faz
  `ggml_cuda_memcpy_1<8,2>(qs, x[ib].qs+iqs)`: `src` é alinhado a 16 B (nb11=272) e `ib*34+2+iqs` é par → ok.
- Sem risco de ler fora: os índices vêm da compactação, que só emite `i < mask->ne[0] == K->ne[1]`, e
  `n_kv_max` da lista é `min(K->ne[1], ncols1*n_kv_max)` em `launch_fattn` (`fattn-common.cuh:1660`).

### Q2 — Layout/strides: OK

- `llama_kv_cache::get_k/get_v` criam a view `[hd, n_kv, n_head_kv, n_seq]` com
  `nb[1] = ggml_row_size(type, hd)` = `8*34 = 272` para Q8_0/D=256 (`src/llama-kv-cache.cpp:3352-3400`).
  `stride_K = nb11` e `stride_V = nb21` em **bytes** e `src = (char*)KV + i_KV*stride` batem com a cache.
- `v_trans` é `!cparams.flash_attn` (`src/llama-model.cpp:3721/3995/4273`), então com FA a V já está no layout
  `[D, n_kv, …]` e `nb21` é o passo de token. O caminho `v_trans` do `build_attn_mha` (transpose) não é
  alcançável com FA ligado.
- `V_is_K_view` é `constexpr = DKQ == 576` (`fattn-mma-f16.cuh:2334`), então o `static_assert` em 2171 é
  vacuamente verdadeiro para a instanciação Q8 (DKQ=256); ver 3.4 (último item).
- `K_h2/V_h2` são bases de linha (sem pré-offset), por isso `elem_offset` é aplicado uma única vez. Correto.
- Bônus: `dequantize_V_q8_0<half,8>` e `dequantize_block_q8_0_f16` usam a mesma conta (`HMUL2(d, q)`), então
  os tiles K/V são **bit-idênticos** ao caminho denso com conversão F16.

### Q3 — Alocação vs kernel: existe divergência (MAJOR, benigna), sem falso positivo

Caminhos conferidos: `ncols1=1/2/4/8`, `turing_mma_available`, `highest_compiled_arch==TURING`, `use_gqa_opt`
(mask/max_bias/K%256/nb%16), batch-invariant, `force_vector_kernel`, Ada, `Q->ne[1]==1`, `logit_softcap`.

- **Sem falso positivo**: `uses_sparse_q8 == true` só ocorre quando (a) `ne1<=1` e o dispatcher pega o braço
  esparso narrow, ou (b) `ne1>=5` e o dispatcher pega `case<8,8>` esparso. Nos dois a execução também
  recalcula `use_sparse_q8` localmente (`fattn-mma-f16.cuh:2364-2386`) e chama `launch_fattn(need_f16=false)`.
  `ncols1=2/4`, `TURING`, `softcap!=0`, `ratio<=4`, `K%256!=0`, sem mask e `logit_softcap` caem todos em
  kernel denso **e** com `need_f16=true`. OK.
- **Falso negativo (MAJOR)**: para `Q->ne[1] ∈ {2,3,4}` o helper devolve `false` (linhas 252/255-257) mas o
  dispatcher escolhe `case<1,8>` esparso Q8, porque o braço esparso narrow vem **antes** da escolha densa de
  `ncols1` (`fattn.cu:263-300`; condição em `fattn.cu:271`). Consequência: aloca buffer F16 que nunca é usado; em Ada, `ne1==2` ainda cai
  em VEC e perde a esparsidade. Detalhe e patch no §3.

### Q4 — Seleção: OK

`can_sparse_q8` (`fattn-mma-f16.cuh:2319-2320`) = `DKQ==256 && DV==256 && ncols2==8 && ncols1∈{1,8}` cobre
exatamente as duas entradas de `may_use_sparse` para 256 (`fattn-mma-f16.cuh:2070`) e exatamente as duas
instâncias declaradas por `DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256,256,8)` e `(…,64)`.
`launch_fattn(need_f16_K=false, need_f16_V=false)` é seguro: K_data/V_data continuam apontando para a cache
Q8, `get_f16_extra_data(dst,false,false).end == dst + nbytes(dst)` e o kernel só escreve em `dst` (+ pools).

### Q5 — Loader/nstages: OK

- `nstages` device (`fattn-mma-f16.cuh:378-384`) força `1` quando `use_sparse && target>1`; logo as chamadas
  dentro de `if constexpr (nstages > 1)` (linhas 684 e 1492) são descartadas na instanciação Q8.
- `use_cp_async` só é `true` para `type_KV == F16` (linhas 712 e 1085), então
  `static_assert(!is_q8_0 || !use_cp_async)` é satisfeito em todas as instanciações.
- Host (`ggml_cuda_fattn_mma_get_nstages`, sem `use_sparse`) pode devolver 2 enquanto o device usa 1 → o host
  reserva shared memory de 2 estágios e o kernel usa 1 (over-reserva, sem corrupção). Pré-existente para F16.

### Q6 — Sincronização: OK

- K: `__syncthreads()` em `fattn-mma-f16.cuh:722` cobre todas as escritas do dequant antes do `ldmatrix`.
- V: `__syncthreads()` em 1094-1095 (imediatamente após o load e antes do laço `mma`).
- Escritas são 4×`half2` = 16 B em slots `(i,k)` disjuntos; o swizzle só permuta unidades de 16 B
  (`mma.cuh:786-806`, `shift=2` para `half2`), então a bijeção/ alinhamento de 16 B do caminho denso é
  preservada. Não há corrida entre chunks adjacentes.

### Q7 — Testes: cobertura real, mas com lacunas

Os 27 casos novos: 18 do laço `kv × nq × n_kv_max` (9 densos com `n_kv_max=0`; com `2048`, esparsos em
`(4096,nq=1)`, `(16384,nq=1)`, `(32768,nq=1)`, `(32768,nq=5)`, `(32768,nq=512)` e densos na fronteira
`min_kv=8*2048+1=16385` em `(16384,nq=5)` e nos `nq=512` de 4096/16384), 3 de `nq=2/3/4` (esparsos narrow,
que exercitam justamente o falso negativo), 3 sem máscara (densos) e 3 de tail (densos, ver 3.3).
Somam-se aos 4 casos Q8 QSA `nq=1` já existentes na base (`kv=4096/16384/32768` com `n_kv_max=2048`, todos
esparsos agora, + controle `n_kv_max=0`). O wide esparso tem só 2 casos (`nq=5` e `nq=512` em 32768).

## 3. Achados por severidade

### BLOCKER
Nenhum.

### MAJOR

**3.1 `ggml_cuda_flash_attn_ext_uses_sparse_q8` diverge do dispatcher para `Q->ne[1] ∈ {2,3,4}`**
`ggml/src/ggml-cuda/fattn.cu:249-259`

```cpp
if (turing_mma_available(cc) && Q->ne[1] <= 8/8) { return narrow; }
if (Q->ne[1] <= 16/8) { return false; }            // ne1 == 2
if (Q->ne[1] <= 32/8 || …TURING) { return false; } // ne1 == 3,4
return wide;
```

`ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1` testa o braço esparso **antes** da escolha densa:

```cpp
if (!(has_wide_sparse && Q->ne[1] > 32/ncols2) && shall_use_sparse(cc, dst, 1, ncols2))
    case<1, ncols2>;   // esparso (Q8 quando K/V são Q8_0)
```

Para `ne1<=4`, `has_wide_sparse && ne1>4` é falso, então o braço é `shall(1,8)` puro. Se ele for verdadeiro
(ex.: KV ≥ `max(4096, min(ne1,8)*n_kv_max+1)`), o kernel executado é o Q8 esparso `ncols1=1`, mas o helper
devolveu `false` → o grafo aloca os buffers F16 de K e V que `launch_fattn` nunca vai usar
(`fattn-mma-f16.cuh:2425` passa `!use_sparse_q8`). Não é corrupção (a execução é auto-consistente), mas:
- desperdiça `2 * ne0 * ne1 * ne2 * ne3 * 2 B` ≈ **64 MiB @32k / 256 MiB @128k** (n_head_kv=2);
- em Ada, `ne1==2` continua caindo em VEC (`fattn.cu:1176-1183`) e a QSA esparsa nunca é usada;
- quebra o contrato documentado no comentário da própria função ("exactly when the Q8_0 sparse kernel is
  going to run"), o que é exatamente o item 3 do pedido de revisão.

Correção (3 linhas, mesma semântica do dispatcher):

```cpp
    // switch_ncols1: the sparse narrow gather is tried first for every tile <= 32/ncols2.
    if (Q->ne[1] <= 32/ncols2) {
        return narrow;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) {
        return false;   // dispatcher falls back to case<4,8> (dense)
    }
    return wide;
```

(`turing_mma_available(cc)` é sempre verdadeiro aqui porque a função já retornou para `cc < AMPERE`.)

Nota: hoje o teste `nq=2/3/4` passa com o código atual (só mede resultado, não memória), então é preciso um
teste/checagem de alocação para travar a regressão (ver §4, item 4).

### MINOR

**3.2 Em Turing o kernel Q8 esparso roda apesar do gate "Ampere+"**
`ggml/src/ggml-cuda/fattn.cu:171-181` + `fattn-mma-f16.cuh:2370-2376`

`q8_direct` devolve `false` para `cc < AMPERE`, mas isso só troca a heurística (`min_kv` 2x em vez de 1x).
Como `shall_use_sparse` no fim só exige `turing_mma_available(cc)`, um sm_75 com D=256/GQA>4/`ne1` 2..4
(o único caso em que o helper não bloqueia antes; `ne1=1` vira VEC) executa `case<1,8>` Q8 com
`need_f16=false`. O loader é arch-independente, então provavelmente funciona, mas é caminho não qualificado e
contradiz o comentário "only on Ampere+". Sugestão: adicionar `cc >= GGML_CUDA_CC_AMPERE` à condição que
escolhe o kernel Q8 em `ggml_cuda_flash_attn_ext_mma_f16_case` (e aí o gate do helper fica coerente), ou
remover o gate do helper e documentar que o narrow é arch-independente.

**3.3 Os casos de KV tail não exercitam corpo esparso**
`tests/test-backend-ops.cpp:13787-13792`

`nq=5`, `kv=4096`, `n_kv_max=2048` → `ncols1=8` → `min_kv = 8*2048+1 = 16385 > 4096` → o corpo é **denso**.
Além disso `test-backend-ops` não passa por `llama-graph.cpp`, então o gate `!k_tail` não é testado de fato
(o próprio commit não tem teste do gate). Correção barata: usar `nq=1` (braço narrow, `min_kv=4096`) ou
`kv=32768` com `n_tail>0`; um caso assim falharia se alguém removesse o `!k_tail`/`tail_route==NONE`.

**3.4 Lacunas de cobertura restantes** (todas em `tests/test-backend-ops.cpp:13761-13793`)

- `mask->ne[1] > Q->ne[1]` (linhas de máscara sobrando): `shall_use_sparse` permite (`>=`) e é o caso de
  máscara com padding; o harness sempre cria `mask->ne[1] == nb`.
- `K->ne[1] % 256 != 0` com D=256/Q8 (força `ncols2=1` + conversão densa) — só existe o caso análogo em
  D=128/96.
- `gqa_ratio <= 4` com D=256/Q8 (`ncols2=4/2/1`, denso).
- `n_kv_max` não múltiplo de `nbatch_fa` (ex.: 2051) e `n_kv_max > K->ne[1]`.
- `logit_softcap != 0` / `max_bias != 0` com `n_kv_max > 0` (deve virar denso mesmo com a dica setada).
- Wide com tile cheio **e** parcial (ex.: `nq=13`); hoje só `nq=5` (1 tile parcial) e `nq=512` (múltiplo de 8).
- `v_is_view_of_k=true` com Q8: hoje não quebra (o kernel lê V por `nb21`), mas o `static_assert` de
  `V_is_K_view` é compile-time `DKQ==576`; um teste serve de documentação de que o caminho é rejeitado antes.

**3.5 Custo do caso `(kv=32768, nq=512)`**
`tests/test-backend-ops.cpp:13763-13769`

`op_flops = 2*24*512*512*32768 ≈ 4.1e11` por backend, e o caso existe 2× (`n_kv_max=0/2048`). Na referência
CPU isso pode custar dezenas de segundos a minutos por caso. Não é bug; se o tempo da suíte importar, reduza
`nq` (ex.: `nq=64`) ou `kv` para a checagem wide (o wide já é coberto por `nq=5`).

### NOTE

**4.1 Shared memory host/device (pré-existente).** `nstages` host (`fattn-mma-f16.cuh:2328`) não conhece
`use_sparse`, então reserva o layout de 2 estágios para um kernel de 1 estágio (~2× over-reserva, sem perda
de ocupação porque `__launch_bounds__(...,2)` já limita). Vale um comentário para não parecer bug.

**4.2 Commit empacotado.** `5f46a18da` também traz MTP (`common.cpp`, `llama-context.cpp`, `qwen35.cpp`,
`qwen4exp.cpp`, `server-context.cpp`, `test-llama-archs.cpp`, `tests/CMakeLists.txt`). Nada a ver com QSA;
atrapalha bisect e mistura riscos. Para o plano Strata convém separar.

**4.3 Callers diretos de `build_attn_mha` não passam pelo gate de tipo novo.**
`src/models/deepseek41.cpp:743` e `src/models/deepseek4.cpp:773` passam `n_kv_max` direto. Se o head dim
desses modelos for 256/256 com GQA>4 e a cache for Q8_0, eles passam a usar o kernel Q8 novo sem o gate
`sparse_kv_type` do `llama-graph.cpp`. É provavelmente desejável, mas amplia o raio de validação além da QSA.

**4.4 Vulkan.** A mudança em `llama-graph.cpp:5121-5126` também habilita a esparsa Q8 no Vulkan
(`ggml-vulkan.cpp:9863-9880`, `min_ratio=16` para quantizado). Fora do escopo CUDA, mas se Vulkan for
suportado nesse fork, os mesmos 27 casos passam a exercitar esse caminho.

**4.5 Invariante implícita do dequant.** `dequantize_V_q8_0<half,8>` exige `i0 % 8 == 0` para não cruzar o
bloco de 32. Hoje isso decorre de `nbatch_K2 % 4 == 0`/`nbatch_V2 % 4 == 0` (static_assert do macro de
config). Sugiro um `static_assert`/comentário no call site (`fattn-mma-f16.cuh:488-494`) porque a ligação é
indireta e uma futura config com `nbatch_* % 4 == 2` passaria a ler 4 bytes do bloco seguinte (valor errado,
sem OOB).

**4.6 Heurística de profundidade do Q8.** `min_kv = max(4096, n_gather+1)` (`fattn.cu:185`) usa
`n_gather = ncols1*n_kv_max`; para `ncols1=8` a lista é a união de 8 queries e o ganho real depende de as
seleções se sobreporem. É só performance, mas o comentário "1x gather bound is already favorable" não vale
para o tile wide (o caminho F16 usa 2x para o narrow também). Medir no A100 antes de fixar.

## 4. O que falta validar na A100 (não há blocker, mas nada em runtime foi provado)

1. **Build sm_80 completo** (já em andamento em `build-a100`, sem `-Werror`): confirmar link de
   `libggml-cuda.so` e ausência de warnings novos em `fattn.cu`/`fattn-mma-f16.cuh`. Evidência parcial já
   observada: `fattn.cu.o` (16:14), `fattn-mma-f16-instance-ncols1_{1,2,4,8}-ncols2_8.cu.o` (16:16-16:17)
   gerados a partir deste HEAD, ou seja os `static_assert`s e o dequant compilam.
2. **Provar qual kernel rodou**, não só que o resultado bate. Sem instrumentação o teste passa mesmo se a
   heurística recusar a esparsa. Rodar `nsys profile --stats=true build-a100/bin/test-backend-ops -o FLASH_ATTN_EXT`
   e conferir o símbolo `flash_attn_ext_f16<256,256,1,8,...,Q8_0,Q8_0,true>` /
   `<256,256,8,8,...,Q8_0,Q8_0,true>`, ou um print temporário atrás de env em
   `ggml_cuda_flash_attn_ext_mma_f16_case` (`DKQ,DV,ncols1,ncols2,use_sparse,use_sparse_q8,ne1,ne11,n_kv_max`).
3. **Casos que precisam rodar esparsos** (além dos 4 Q8 QSA `nq=1` pré-existentes):
   `kv=4096/nq=1` e `kv=32768/nq=1` (narrow, fronteira `>=4096`), `kv=32768/nq=5` e `kv=32768/nq=512`
   (wide), `kv=16384/nq=5` (wide denso na fronteira 16385), `nq=2/3/4` (falso negativo 3.1: resultado ok +
   conferir que a alocação não explodiu).
4. **Alocação**: para os casos esparsos, confirmar que `ggml_cuda_flash_attn_ext_get_alloc_size` devolve
   `nbytes(dst)` (sem buffers F16) e que não há `GGML_ASSERT(f16_extra.K != 0)`.
5. **Fim a fim QSA**: `qwen4exp`/Qwen QSA com `-ctk q8_0 -ctv q8_0 -fa on`, comparar logits/perplexity com
   `-ctk f16` e com o build anterior; conferir no `F4_TAIL_TRACE` que `n_kv` (op param 4) chega igual a
   `n_sel`, e que com KV tail ativo o bound é 0 (gate `!k_tail`).
6. **CUDA graphs ligados** (`GGML_CUDA_USE_GRAPHS` está ON nesse build): a esparsa Q8 faz alocação de pool
   (`KV_max`) dentro do capture, igual à F16 já usada; validar replay repetido sem erro e sem crescimento.
7. **Regressão nas arquiteturas não cobertas pelo build sm_80**: Turing (`highest_compiled_arch==TURING`,
   caso 3.2) e Ada (`ne1<=2`, VEC) não são exercitáveis na A100; se essas GPUs importam, precisa de build
   separado.
8. **Cartão de memória**: medir o compute buffer do grafo com `nq=2..4` antes/depois do patch 3.1 para
   confirmar a economia (esperado ~64 MiB @32k com n_head_kv=2).

## 5. Referências rápidas

| Item | Local |
|---|---|
| Gate de tipo no grafo | `src/llama-graph.cpp:5121-5126` |
| Heurística `q8_direct` / `min_kv` | `ggml/src/ggml-cuda/fattn.cu:169-188` |
| Helpers de alocação/seleção | `ggml/src/ggml-cuda/fattn.cu:200-260`, `1176`, `1188`, `1320-1328` |
| Dispatcher `ncols2`/`ncols1` | `ggml/src/ggml-cuda/fattn.cu:263-300`, `302-392` |
| Loader (`elem_offset`, dequant) | `ggml/src/ggml-cuda/fattn-mma-f16.cuh:393-517` |
| Chamadas K/V | `…fattn-mma-f16.cuh:713-714`, `1086-1087`, `684`, `1492` |
| `is_q8_kv` / `stride_*` / asserts | `…fattn-mma-f16.cuh:2168-2182` |
| `can_sparse_q8` e seleção | `…fattn-mma-f16.cuh:2319-2386`, `2425-2426` |
| `may_use_sparse` | `…fattn-mma-f16.cuh:2070-2076` |
| Dequant Q8_0 | `ggml/src/ggml-cuda/fattn-common.cuh:1108-1136` |
| Bloco Q8_0 (34 B) | `ggml/src/ggml-common.h:371-376` |
| View da KV cache | `src/llama-kv-cache.cpp:3352-3400` |
| Testes novos | `tests/test-backend-ops.cpp:13761-13793` |
| Testes Q8 QSA pré-existentes | `tests/test-backend-ops.cpp:14522-14530` |
