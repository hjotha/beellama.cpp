# REPORT — B2: corrigir a sonda de teto do Xbox (gpupeak) e o roofline

Branch: `fix/xbox-ceiling-probe-20261026`, clone
`/home/hjotha/beellama-agents/probe-fixes`, base
`9a46583cd` (integração `integ/rkva-xbox-agents-20261026`).
Nenhum C++ do app foi compilado (MSVC é do Claude) e nada foi executado no
Xbox. A única compilação externa foi DXC na 193, no diretório isolado
`C:\Users\hjotha\build\oc-probe-fixes-20261026\`. Os testes Python e a
compilação de sintaxe dos arquivos portáveis rodaram em Linux.

## 1. Causa real do `pattern checksum mismatch`

O `stream_read` que rodou no Series X não era o `stream_read` do fonte
commitado. O header DXIL embutido estava velho:

```
manifest antigo  stream_read/stream_write/stream_copy source_sha256
                 8309057ac7bc1c0399c866d8c34257989c174233f3aabb09fe01caff37a90945
HEAD fonte       tools/xbox/shaders/gpupeak_stream.hlsl
                 3921403c9279a9e13755a398623fa7807eba37875414883c635ea354ad2f114b
```

O fonte em `HEAD` tem `if (mode == 1u || acc == 0x9e3779b9u)`, mas o DXIL
antigo (extraído do header commitado e desmontado com `dxc -dumpbin` na 193)
só testava o sentinela:

```
; stream_read_old.ll (header commitado)
%36 = icmp eq i32 %30, -1640531527
...
%42 = mul i32 %41, %7       ; groups_x * gid.y
%43 = add i32 %42, %6       ; + gid.x
call void @dx.op.bufferStore... ; só no caso do sentinela
```

Com `mode=1` o host esperava que a passada de validação sempre publicasse o
acumulador, mas o kernel antigo quase nunca entrava no `if`: o sink ficava com
o marcador deixado pelo `CSWrite` (`Sink[0] = 1`) ou com lixo, e o XOR do host
necessariamente divergia. O `gpubw` independente validava a leitura porque ele
tem pattern e shader próprios; ele não exercita este caminho.

O DXIL novo, regenerado do fonte atual, contém o teste que faltava e publica o
acumulador por thread:

```
; stream_read_new.ll
%39 = icmp eq i32 %38, 1          ; mode == 1
%40 = icmp eq i32 %36, -1640531527
%41 = or i1 %40, %39
...
%44 = mul i32 %43, %11            ; groups_x * (threadId.y << 8)
%45 = add i32 %44, %4             ; + threadId.x  (slot por thread)
```

Além do header velho, havia dois defeitos latentes no mesmo caminho:

1. **Corrida no sink por grupo.** O `CSRead` antigo fazia
   `Sink[flat_group] = acc` com 256 threads escrevendo no mesmo slot sem
   átomo. O slot guardava o acumulador de uma thread arbitrária, não o XOR do
   grupo. Para o plano de 256 MiB cada thread possui 32 vetores (par), então
   todos os acumuladores eram zero por coincidência; para qualquer contagem
   ímpar o erro aparece. A correção publica um slot por thread
   (`Sink[flat_thread]`) e o host dobra `groups_x * groups_y * 256` slots.
   O sink de 2 MiB (`8192 * 64` words) tem exatamente 524.288 words, o mesmo
   número de threads do plano de 2048 grupos; há `static_assert` para isso.
2. **Cópia de um recurso UAV sem transição.** O readback fazia
   `CopyResource(sink_readback, sink)` com `sink` ainda em
   `UNORDERED_ACCESS`. A cópia só é definida com a origem em `COPY_SOURCE`;
   sem a transição o readback é indefinido (e foi uma segunda fonte de lixo).
   Agora há `UNORDERED_ACCESS -> COPY_SOURCE -> UNORDERED_ACCESS` e
   `CopyBufferRegion` só do intervalo dobrado. O mesmo erro existia no
   readback de 4 bytes da suíte de latência e também foi corrigido.

### Modelo CPU do kernel

`tools/xbox/tests/gpupeak_stream_model.py` reproduz o `CSWrite` (pattern por
vetor), o `CSRead` (XOR por thread) e as duas variantes de sink. Saída literal
para as formas de teste:

```
schema gpupeak-stream-model-v1
512 vectors/thread 1 expected 512 fixed 512 old_first 259 old_last 771
1000 vectors/thread 2 expected 1000 fixed 1000 old_first 0 old_last 490
4097 vectors/thread 9 expected 4097 fixed 4097 old_first 4097 old_last 0
16384 vectors/thread 8 expected 0 fixed 0 old_first 0 old_last 0
65536 vectors/thread 1 expected 65536 fixed 65536 old_first 256 old_last 65792
16777216 vectors/thread 32 expected 0 fixed 0 old_first 0 old_last 0
```

`fixed` é o XOR dos slots por thread e bate com `gpupeak_pattern_xor()` em
todas as formas, inclusive o plano real de 256 MiB (16.777.216 vetores,
2048 grupos, 524.288 threads). A forma 512/2 grupos mostra a corrida antiga
falhando; a forma real mostra por que ela passava despercebida.

## 2. Roofline com falha parcial

`tests/rkva_ceiling_roofline.py` não aborta mais quando o checksum de stream
falha ou falta. O que mudou:

* `parse_gpupeak_json()` só levanta erro para schema inválido, D3D12 que não
  rodou ou ausência de `timestamp_frequency`. Checksum ausente/falho vira
  warning e `stream_verified=False`.
* `build_result()` mantém ALU/latência/ocupação; a banda não verificada vai
  para `ceiling_measured.stream_observed_gbs` (observação) e
  `bandwidth_gbs` fica **0**, com `bandwidth_verified=false`, `partial=true` e
  a lista de `warnings`. Os pontos de pass1 continuam sendo calculados contra
  o teto de FP32 medido, com `bandwidth_verified=false`.
* Corrigi um bug pré-existente que apareceu ao validar o clock: `best()`
  escolhia a menor `gpu_seconds`, então para o sweep de ALU pegava o dispatch
  de 52 grupos (mais curto) em vez do de 1664 (maior taxa). Isso dava
  `fma_f32 = 1,7868 TFLOP/s` e `dot4_i8 = 4.040,8 GOPS` no JSON r2; o valor
  real é o pico de 1664 grupos. Novo `best_throughput()` seleciona por
  FLOP/OPS por segundo.

Com o JSON real `tools/xbox/docs/results/gpupeak-series-x-20261026-r2.json` e
`points=[]`, a saída offline é:

```
partial=True warnings=1
bandwidth_gbs=0.0 verified=False observed_read=280.3579585450884
fp32_tflops=11.0943 dot4_i8_tops=37.5750
derived_clock_mhz=1666.81
empty_dispatch_gpu_us=3.008
dual_cu=52.0 agree=1.0
```

O warning é `stream checksum failed: read/268435456_validate (pattern checksum
mismatch); copy/268435456_validate (copied pattern checksum mismatch)` — o dado
histórico não tem os campos `checksum`/`expected_checksum`, então não há
detalhe hexadecimal nele; runs novos gravam os dois valores.

## 3. Clock: derivação correta e remoção da cadeia dependente

Removi do C++ `gpupeak_implied_clock_mhz()`, `catalog_frequency_mhz`,
`implied_clock_*` e o bloco `"derived"` do JSON. A cadeia FMA dependente
continua sendo medida como latência (`latency_fma`), mas não infere clock: a
premissa de 4 ciclos não vale para wave64.

A derivação agora é no Python, a partir do pico FP32 medido:

```
clock_mhz = fp32_tflops * 1e12 / (52 * 128) / 1e6
```

Com os 11,0943 TFLOP/s do r2: **1666,81 MHz derivados**. O JSON carrega
`derived_shader_clock_mhz`, `derived_from_fp32_tflops`, `assumed_dual_cu=52`,
`assumed_flops_per_dual_cu_clock=128`, `nominal_clock_mhz=1825` e a nota
"Not a measured clock". O valor é derivado, não medido.

## 4. `gpubw`: host do CSV vem do adapter

`GpubwResult` ganhou `adapter`. No Windows, `measure_gpubw()` consulta o
adapter pelo LUID do device (`IDXGIFactory4::EnumAdapterByLuid` +
`GetDesc1`) e grava a descrição. `gpubw_host_label()` sanitiza para CSV
(substitui vírgula/quebra/espaço por `_`, cai para `unknown` se vazio), e
`inference-bridge.cpp` agora chama `format_gpubw_row(r, nullptr)` em vez do
fixo `"xbox-series-s"`. O CSV de 2026-10-26 continua histórico e com o rótulo
antigo; runs novos gravam o adapter real (`SraKmd_arden` no Series X).

## 5. `min16float` (FP16 relaxado)

Novo alvo `fma_f16_min16`: `CSFmaF16Min16`, `min16float2`, perfil `cs_6_0`,
sem `-enable-16bit-types`, 4 cadeias. Ele é um dado, não uma promessa de
throughput empacotado: o DXIL pode ser promovido. O `dxc -Fc` na 193 mostra
que este DXC **não** promoveu as operações:

```
%38 = fadd fast half %16, %30
%55 = fmul fast half %46, %30
%57 = fadd fast half %55, %16
...
target datalayout = "...-f16:32-f32:32..."
```

O `f16:32` indica half legalizado em registrador de 32 bits; as operações são
`half`. O que o run no console vai dizer é se o PSO `cs_6_0` com esse DXIL é
aceito (os `float16_t` nativos com `-enable-16bit-types` foram rejeitados com
`E_INVALIDARG`). O resultado entra em `ceiling_measured.fp16_min16_tflops` e
`checks.fma_f16_min16_accepted`; se o PSO for rejeitado, a linha não existe e
o campo fica 0.

## 6. Probe Q4: flags de profiling

`tools/xbox/tests/rkva_range_q4_kernel_probe.py` agora usa
`PROFILE_ENABLE | PROFILE_RESET` em todas as chamadas que precedem medição:

```
113:        profile(client, PROFILE_ENABLE | PROFILE_RESET, 0)
126:                    profile(client, PROFILE_ENABLE | PROFILE_RESET, mode)
144:            profile(client, PROFILE_ENABLE | PROFILE_RESET, mode)
150:            full = profile(client, PROFILE_REPORT_ONLY, 0)
152:        profile(client, PROFILE_RESET, 0)      # cleanup: desliga
158:            profile(client, 0, 0)               # best effort: desliga
```

O loop de decode em si já usava `ENABLE|RESET` no base; a inconsistência
estava na configuração de modo da paridade, que usava só `RESET` e desligava o
profiling. `RESET` sozinho continua sendo o cleanup final, onde desligar é o
objetivo.

## 7. DXIL regenerado (DXC 1.8.2502.11, 193)

```
$ python3 tools/xbox/scripts/compile-gpupeak-shaders.py \
    --ssh-host hjotha@192.168.1.193 \
    --remote-root 'C:/Users/hjotha/build/oc-probe-fixes-20261026'
dxc: dxcompiler.dll: 1.8 - 1.8.2502.11 (239921522); dxil.dll: 1.8(1.8.2502.11)
stream_read      3768 bytes
stream_write     3668 bytes
stream_copy      3904 bytes
fma_f32          3992 bytes
fma_f16          3944 bytes
fma_f16_min16    3944 bytes
dot2_f16         4032 bytes
dot4_i8          3784 bytes
latency_fma      3748 bytes
cu_probe_32      4140 bytes
cu_probe_16      4140 bytes
empty            3344 bytes
```

Comparação de hash com o manifesto anterior:

```
cu_probe_16    SAME     4140
cu_probe_32    SAME     4140
dot2_f16       SAME     4032
dot4_i8        SAME     3784
empty          SAME     3344
fma_f16        SAME     3944
fma_f16_min16  NEW      3944
fma_f32        SAME     3992
latency_fma    SAME     3748
stream_copy    CHANGED  3980 -> 3904  (fix do sink/validação)
stream_read    CHANGED  3816 -> 3768  (fix do mode==1 + sink por thread)
stream_write   CHANGED  3664 -> 3668  (header velho regenerado)
```

O `stream_write` antigo desmonta para o mesmo pattern
`(threadId.x+1, groupId.x+1, n_vec, mode)` com o mesmo grid-stride; a diferença
de 4 bytes é metadado do header que não era reproduzível a partir do fonte
`HEAD` (source hash 8309… vs 3921…). O corpo do `CSWrite` não foi alterado.
`--check` agora falha se um header committed não corresponder ao hash do fonte:

```
$ python3 tools/xbox/scripts/compile-gpupeak-shaders.py --check
stream_read    ok af927db141454e30
...
empty          ok a9dac8c6b8188592
```

## 8. Validação offline executada

```
$ python3 -m unittest tools/xbox/tests/test_rkva_ceiling_roofline.py \
    tools/xbox/tests/test_gpupeak_stream_model.py
----------------------------------------------------------------------
Ran 26 tests in 1.202s

OK
```

```
$ g++ -std=c++17 -Wall -Wextra -fsyntax-only -Itools/xbox/include \
    tools/xbox/src/bridge/gpupeak.cpp
$ g++ -std=c++17 -Wall -Wextra -fsyntax-only -Itools/xbox/include \
    tools/xbox/src/bridge/gpubw.cpp
syntax-ok
```

Harness Linux dos helpers puros (`gpupeak_pattern_xor`, JSON com
`checksum`/`expected_checksum`, ausência de `implied_clock`,
`gpubw_host_label`):

```
$ g++ -std=c++17 -Wall -Wextra -Itools/xbox/include /tmp/b2_harness.cpp \
    tools/xbox/src/bridge/gpupeak.cpp tools/xbox/src/bridge/gpubw.cpp \
    -o /tmp/b2_harness && /tmp/b2_harness
b2-harness-ok
```

O script novo é inerte por padrão; o plano foi verificado e o modo
`--execute --gpupeak-result <json>` foi exercitado com o JSON r2 (sem Xbox),
gerando o mesmo `partial=True` e `derived_shader_clock_mhz=1666.81`.

## 9. Pendência de hardware (Claude)

O C++ do app (MSVC/UWP) não foi compilado por mim. Depois do build, rodar:

```
export XBOX_IP=... XBOX_USER=... XBOX_PASS=...
tools/xbox/scripts/run-gpupeak-probe.py --execute --host "$XBOX_IP" \
    --output .oc-artifacts/gpupeak-probe-result.json
```

O JSON terá `checks.stream_checksum_verified=true` se a correção funcionou,
`checks.fma_f16_min16_accepted` para a variante relaxada e
`checks.derived_shader_clock_mhz` para o clock derivado. Se o checksum ainda
falhar, os campos `checksum`/`expected_checksum` das linhas
`*_validate` dão o valor observado e o esperado. O script não instala nem
compila nada; só usa o pacote já instalado.

## 10. Riscos e estimativas

* O `min16float` pode ser aceito e ainda assim executar a taxa de FP32; o
  número de TFLOP/s não prova empacotamento. É um dado de aceitação de PSO.
* O readback do sink agora é determinístico, mas a primeira execução no
  console ainda é a prova final; não há como validar D3D12/UWP no host Linux.
* A correção do `best_throughput` muda números históricos: o r2 passa de
  1,7868 para 11,0943 TFLOP/s em FP32 e de 4.040,8 para 37.575,0 GOPS em
  INT8. Os dois valores conferem com a leitura manual do relatório A1.
* Estimativa: com o header regenerado, a validação de stream deve passar sem
  alterar as taxas medidas (write 299,6 GB/s, read 280,4 GB/s, copy
  149,3 GB/s por direção no r2); a mudança é de correção, não de desempenho.
* `stream_write` teve o header regenerado e o hash mudou; o IR desmontado é o
  mesmo pattern/grid-stride. Se o revisor exigir hash idêntico para esse alvo,
  a alternativa é manter um header comprovadamente não reproduzível do fonte,
  que é justamente o defeito que causou o bug.

## 11. Arquivos tocados

* `tools/xbox/shaders/gpupeak_stream.hlsl` — sink por thread; sem corrida.
* `tools/xbox/src/bridge/gpupeak.cpp` — validação, transições, checksums no
  JSON, `fma_f16_min16`, remoção do clock implícito.
* `tools/xbox/include/xllama/gpupeak.h` — contrato do sink, checksums, campos
  removidos, FLOP da variante min16.
* `tools/xbox/shaders/gpupeak_alu.hlsl` — `CSFmaF16Min16`.
* `tools/xbox/scripts/compile-gpupeak-shaders.py` — alvo min16 e `--check`.
* `tools/xbox/shaders/generated/gpupeak_*` — DXIL regenerado; novo header
  `gpupeak_fma_f16_min16_dxil.h`.
* `tools/xbox/tests/rkva_ceiling_roofline.py` — falha parcial, clock derivado,
  `best_throughput`.
* `tools/xbox/tests/gpupeak_stream_model.py` +
  `test_gpupeak_stream_model.py` — modelo CPU e testes.
* `tools/xbox/tests/test_rkva_ceiling_roofline.py` — 26 testes.
* `tools/xbox/src/bridge/gpubw.cpp`, `include/xllama/gpubw.h`,
  `uwp/inference-bridge.cpp` — host pelo adapter.
* `tools/xbox/tests/rkva_range_q4_kernel_probe.py` — flags de profiling.
* `tools/xbox/scripts/run-gpupeak-probe.py` — runner inerte/`--execute`.
* `tools/xbox/docs/gpu-ceiling.md` — documentação atualizada.
* `tools/xbox/uwp/rkva/remote-attn-engine.cpp` — commit separado apenas com
  `uint8_t rescale_skip = 0;` em `BulkSegmentProfile`, a correção C2039 que o
  integrador aplicou no clone; o campo já era lido/escrito pelo caminho A2.
