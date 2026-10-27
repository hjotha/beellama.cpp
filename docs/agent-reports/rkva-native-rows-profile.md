# RKVA native-rows (op 22) — profiling, correções da revisão Astra

Branch limpa `perf/rkva-native-rows-profile-clean-20261027`, criada a partir do
`main` local do root (`e079799c1`, fetch read-only de `/home/hjotha/beellama.cpp`)
e com **apenas** o cherry-pick de `3f92cb1dc` (a correção de profiling). O
protótipo vetorizado `222fb5538` (flag de resposta, sender/cursor) **não** está
na linhagem: a ABBA do protótipo mediu −0,192% decode e −0,254% prefill, ficou
OFF e não promovida, e nada dele entra neste diff. O root está em smoke RAW com
source freeze; nada em `/home/hjotha/beellama.cpp` foi escrito.

Branches originais preservadas sem alteração:

- `perf/rkva-native-rows-profile-20261027` = `3f92cb1dc` (pré-correções Astra).
- `perf/rkva-vectored-response-20261027` = `222fb5538`.
- `main` (clone de trabalho) = `4bf192bb2`.

Nada de GPU/shader/protocolo/matemática/validador foi tocado.

## Achado do estudo que permanece válido

O estudo aprovado de 6 braços em formato nativo mediu Q8 descriptor 5 / op 22
com 2304 submits reais na GPU e, ainda assim, `attention`/`upload`/`bulk`/
queries zerados e `buckets` vazio. Este trabalho **não** reabilita aquele
resultado: aquele estudo continua **UNSUPPORTED**, e os tempos de 18.743
permanecem não qualificados. O codec GPU continua em 0 (verdadeiro), e o
validador de formato que exige FP8 na GPU segue reprovando o codec CPU.

Causa confirmada no código (linhas do commit base):

- `attend_q8_bulk()`/`attend_q4_bulk()` chamam `attend_native_rows()`
  (`remote-attn-engine.cpp:2675` e `:2845` no commit base).
- `attend_native_rows()` ligava `bulk_recording_ = true`, mas nunca ligava
  `profile_->bulk.active` nem dimensionava `profile_->bulk.segments`, e nunca
  chamava `finish_bulk_profile()`.
- `upload_range()` (`:2195`) e `attend_range()` (`:2856`) desligam o profiling
  individual quando `bulk_recording_`, e o agregador bulk exige
  `profile_->bulk.active`; com `active == false` **tudo** era descartado. Os
  submits continuavam sendo contados porque `Counters::submits` é global.
- Codec CPU real: `attend_native_rows()` expande Q FP8→F32 na CPU (antes de
  `attend_range`) e `remote-attn.cpp` empacota OUT F32→F16/FP8 na CPU, antes do
  codec GPU de KVarN (`attend_kvarn_bulk_fp8`). Nada disso era medido, e o
  contador de fallback FP8 de KVarN também não era (corretamente) tocado.

## Correções pedidas pela revisão Astra

1. **Correlação de ciclo real.** `attend_native_rows()`, `attend_q4_bulk()` e
   `attend_q8_bulk()` agora recebem `uint64_t cycle_id` **sem default** (o
   default `0` mascarava o bug) e gravam `envelope.cycle_id = cycle_id`. Os dois
   call sites do transporte (Q4 e Q8) forwardsam `req.cycle_id`. Nenhuma
   mudança de protocolo/wire: `cycle_id` já vinha no `rkva_request`.
2. **`BulkWindowGuard` armado só depois da posse.** O guard era construído antes
   da validação; uma chamada recusada (`bulk_recording_` já ativo) destruía a
   janela de outra chamada. Agora ele é construído depois de `begin()`, do
   `profile_->bulk.active = true`/dimensionamento dos segmentos e imediatamente
   antes de `bulk_recording_ = true`; caminhos de recusa retornam antes de o
   guard existir e não tocam flags alheias. Todos os erros depois de abrir a
   janela continuam limpos pelo RAII.
3. **Copy regions reais por segmento.** `upload_range()` nativo grava **dois**
   `CopyBufferRegion` (planos K e V) e `finish_bulk_profile()` contava 1. A
   regra agora é storage-aware via
   `rp::bulk_upload_copy_regions(storage, has_upload)`: Q4_0/Q8_0 com upload = 2,
   KVarN4 = 1 (contabilidade histórica preservada) e segmento query-only = 0.
   Nenhuma cópia, submissão ou espera adicional.

## O que mudou (arquivos)

### `tools/xbox/uwp/rkva/rkva-range-profile.h` (aditivo + regras testáveis)

- `Sample`: `q_unpack_ns`, `q_unpack_bytes` (só rotas nativas; 0 no wire F32 e
  nas rotas decodificadas na GPU).
- `Counters`: `native_rows_calls`, `cpu_q_unpack_calls/bytes/ns`.
- `BulkEnvelopeSample`: `storage`, `serial_prefix` com defaults
  `kStorageKvarn4`/`1` (schema KVarN preservado quando o chamador não escreve).
- `count_attention_sample()`: regra única de contadores por storage, extraída do
  corpo que já existia em `finish_profile_sample()`. Os contadores
  `kvarn4_*`/`optimized_by_storage[kvarn4]` exigem storage KVarN4 (e prefixo
  serial para as especializações de kernel), então Q4_0/Q8_0 nativo nunca os
  incrementa.
- `bulk_upload_copy_regions()`: regra pura e host-testável da correção 3.
- JSON (aditivo): em `bulk`, `native_rows_calls` e
  `native_rows_cpu_q_unpack{calls,bytes,duration_us}`; em
  `recent_envelopes[]`, `storage` e `serial_prefix` (e o `cycle_id` real); em
  buckets de atenção, `cpu_q_unpack_us` e `q_unpack_bytes`. `schema` continua
  `rkva-range-profile-v1`.

### `tools/xbox/uwp/rkva/remote-attn-engine.h`

- `attend_q4_bulk`, `attend_q8_bulk` e `attend_native_rows` com
  `uint64_t cycle_id` obrigatório.

### `tools/xbox/uwp/rkva/remote-attn-engine.cpp`

- `BulkSegmentProfile`: `storage`, `serial_prefix`, `wire`, `has_upload`,
  `q_unpack_ns`, `q_unpack_bytes` (defaults = envelope KVarN atual).
- `attend_native_rows()`: envelope real (`storage` = Q4_0/Q8_0,
  `serial_prefix = 0`, `wire` real do primeiro segmento, `cycle_id` do request),
  ativação/dimensionamento do bulk existente, spans de validação/alocação/
  submit/fence/readback, timestamps **nas posições já existentes**
  (`kTsBulkEnvelopeBegin`, `PackEnd`, `CopyEnd`), resolução antes do submit
  único que já existia e `finish_bulk_profile()` **depois** do fence/readback
  existentes. `BulkWindowGuard` armado apenas após a posse da janela.
- `upload_range()`: preenche `map_copy_upload_ns`, `record_upload_ns`,
  `upload_bytes` (2 planos), `append_rows`, `live_count`, `has_upload`,
  `storage`, `serial_prefix` e os dois timestamps bulk.
- `attend_range()`: grava `storage`/`serial_prefix` reais do segmento no caminho
  bulk (KVarN4: 3/1; nativo: 1|2/0), sem alterar a escolha de kernel.
- `finish_bulk_profile()`: deixa de hardcodar storage 3/prefixo serial. Samples
  de upload/atenção usam `seg.storage`/`seg.serial_prefix`; contadores por
  storage vêm de `count_attention_sample()`; `fp8_gpu_calls`/
  `fp8_cpu_fallback_calls` só contam em rota KVarN; `native_rows_calls` conta
  envelopes nativos; segmento query-only não gera Sample de upload nem copy
  region; copy regions de upload vêm de `bulk_upload_copy_regions()`; Q-unpack
  acumula em `cpu_q_unpack_*`. Nenhum `submit`, espera ou timestamp nessa
  função.
- `attend_kvarn_bulk()`: inicializa explicitamente `storage=3`/`serial=1`
  (o caminho compacto em `rkva-kvarn-bulk-f16.inc` herda os defaults e não foi
  editado).

### `tools/xbox/uwp/rkva/remote-attn.cpp` (telemetria do codec CPU)

- `remote_state`: `native_rows_pack_calls/bytes/us` e
  `native_rows_pack_{fp8,f16,f32}_calls`.
- No ramo layout 4/5: forwardsa `req.cycle_id` nas duas chamadas e mede o
  empacotamento OUT na CPU (após a chamada única ao engine, antes do
  `return RKVA_OK`). Não escreve em `bulk_output_transcode_us` nem em
  `fp8_cpu_fallback_calls`.
- `rpc_profile_json()`: objeto aditivo `native_rows_cpu_out_pack`.
- Reset em `HELLO` e em `RKVA_PROFILE_RESET`.
- Nenhum código do protótipo vetorizado (flag de resposta, sender ou cursor)
  existe neste arquivo nem no diff da branch (verificado por grep).

### `tools/xbox/tests/test_rkva_native_rows_profile.py`

- Fixture funcional: compila o `rkva-range-profile.h` real com `g++`, injeta os
  Samples que `finish_bulk_profile()` compõe para Q8_0/Q4_0 nativo e KVarN4 e
  valida buckets, isolamento de contadores, `native_rows_*`, `cpu_q_unpack`,
  `cycle_id` no envelope e a regra de copy regions (2/2/1/1/0).
- Guard: extrai a definição **real** de `BulkWindowGuard` do engine e compila
  um fixture que prova (a) janela alheia intacta quando o guard não é
  construído, (b) limpeza da janela própria no sucesso/erro e (c) janela
  profiling-off.
- Estrutural (corpo de função balanceado): ordem `bulk.active` → guard →
  `bulk_recording_` → upload/atenção → submit → finalize; guard depois da
  validação de recusa; assinaturas com `cycle_id` sem default; forwarding nos
  dois call sites; `rp::bulk_upload_copy_regions` no agregador; upload com
  exatamente 2 `CopyBufferRegion`.

### Desvio de ownership (autorizado nesta rodada)

`rkva-range-profile.h` foi explicitamente liberado para tipos/regras de
profile. As edições são aditivas e os defaults preservam o schema KVarN.
`rkva-kvarn-bulk-f16.inc`, validadores, shaders e protocolo ficaram intactos.

## O que NÃO é afirmado

- O validador de formato continua reprovando Q8 nativo com codec CPU; este
  commit não qualifica GPU codec e não muda o veredito UNSUPPORTED do estudo.
- Não há ganho de desempenho medido; nada foi executado em GPU/Xbox.
- `profile OFF`: sem timestamps, alocação, submissão ou espera extra.
- Segmentos bulk continuam reportando `alloc_grew=0` no Sample (convenção já
  existente); o `envelope.alloc_grew` guarda o valor real.
- Wire F16 nativo também faz expansão Q na CPU e é contado; F32 é memcpy e
  reporta 0.

## Validação CPU executada (saída real)

```
$ python3 tools/xbox/tests/test_rkva_native_rows_profile.py
Ran 18 tests in 0.419s
OK
```

```
$ g++ -std=c++17 -Wall -Wextra -Werror tools/xbox/tests/rkva_range_profile_cpu.cpp -o /tmp/opencode/rkva-native-rows/rkva_profile
$ /tmp/opencode/rkva-native-rows/rkva_profile > /tmp/opencode/rkva-native-rows/profile2.json
$ python3 -m json.tool /tmp/opencode/rkva-native-rows/profile2.json > /dev/null
rkva_range_profile_cpu: ok
header harness OK
```

```
$ g++ -std=c++17 -Wall -Wextra -Werror tools/xbox/tests/rkva_kvarn_bulk_cpu.cpp -o /tmp/opencode/rkva-native-rows/kvarn_bulk_cpu
$ /tmp/opencode/rkva-native-rows/kvarn_bulk_cpu
fp8 256 saturation patterns and numerical contract: passed
splitzip bounds, truncation and corruption safety: passed
kvarn4_mq_split policy invariants: passed
compact exact-tail upload admission, seal edges and wire byte savings: passed
native bulk envelope/coverage/f16 admission: passed
```

```
$ python3 tools/xbox/tests/test_rkva_kvarn_bulk_profiling.py
Ran 6 tests in 0.002s
OK
```

`clang-format --dry-run --Werror` não aponta violação nas linhas adicionadas dos
quatro arquivos C++ (restam violações pré-existentes em linhas não tocadas) e
`git diff --check` está limpo.

## Pendente (exige MSVC/Xbox/hardware; não fabricado)

1. Compilar UWP com MSVC na máquina de build configurada (`tools/xbox/README.md`)
   — engine/transporte são D3D12/WinSock e não compilam no host Linux; nenhum
   stub foi usado para fingir compilação.
2. Em console, profiling ON, repetir op 22 Q8 descriptor 5 e conferir:
   `counters.upload_calls`/`attention_calls`/`bulk_calls`/`bulk_queries` não
   zerados; `bulk.native_rows_calls == bulk_calls`;
   `recent_envelopes[].cycle_id` igual ao `cycle_id` do request;
   buckets `upload/q8_0/...` e `attention/q8_0/{reference|optimized-...}`;
   `native_rows_cpu_q_unpack` coerente com `wire_fp8_calls`;
   `fp8_gpu_calls == 0`; `fp8_cpu_fallback_calls == 0`; `copy_regions`
   acrescido de 2 por upload nativo (1 por atenção e 1 por upload KVarN,
   como antes).
3. Teste de recusa: chamar op 22 enquanto outro envelope está em gravação e
   confirmar que o retorno antecipado não altera `bulk_recording_`/
   `profile_->bulk.active` da chamada em curso.
4. Conferir que os buckets/contadores KVarN (F32/F16/FP8) e `submits` não
   mudaram e que o validador de formato segue reprovando o codec CPU.
