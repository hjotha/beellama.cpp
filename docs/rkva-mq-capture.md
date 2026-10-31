# RKVA MQ2 capture (diagnostic, default off)

Put `rkva-mq-capture.cfg` in Xbox LocalState before HELLO. Its entire content is
three decimal integers: `remote_layer first_position segment`. It captures the
first qualifying revision-3 KVarN4 fine MQ2 dispatch with exactly that remote
layer, first absolute query position and zero-based bulk segment. For a serial
request segment is 0. It makes one attempt per engine; remove or move the output
before rearming. INT8 is excluded. The normal route and log stay unchanged when
the file is absent.

`remote_layer` is not the model layer ID: the graph counts earlier model layers
for which `has_kv(l) && !is_recr(l)` (`src/llama-graph.cpp`,
`ggml_remote_attn_range`). The result becomes the request `layer_id`, which the
server passes to `attend_range`. For a model whose full-attention layers are
3, 7, ..., 63, model layer 3 maps to remote 0, layer 7 maps to remote 1, etc.
Verify the model's actual layer pattern before choosing the ordinal.

After the existing successful submit/fence, `rkva-mq-capture.bin` contains the
packed little-endian `CaptureHeader` from `remote-attn-engine.h`, followed by
four consecutive byte ranges with the recorded lengths: query buffer (Q F32,
Qrot F32, then i32 positions); the complete physical layer buffer (K plane then
V plane, each records, stage, physical tail); pass-1 partials; and F32 output
then F32 LSE. A final SHA-256 covers the entire header and payload. The header
records the actual root constants, geometry, descriptor, extents, kernel DXIL
SHA-256, wave limits, dispatch dimensions, generation, coverage and layout.
Kernel 1 is MQ2 and 2 is GQA2. `wire=UINT32_MAX` means the serial route's wire
was not available at the engine; it does not imply F32 on the wire. Direct
output has zero partial bytes. More than 512 MiB payload is refused. An existing
output or temporary file is not overwritten.

The readback contains full physical KV planes, including unused or stale slots.
Different bytes in unused slots do not prove a kernel input difference. Copying
them can expose prior session data; handle the file as sensitive. Copies and
resource barriers add GPU work and disk I/O after the fence, so this is not a
timing-neutral observation. Compare capture-off and capture-on results and
prove equality of the consumed input bytes before attributing a difference to
a shader. Identical prompt tokens or a seeded generator do not establish that.
Use the same fixed teacher and capture the packed wire separately if needed;
this file contains actual post-unpack values and pre-pack output, not the wire
payload. Never use F4_NODE_DUMP as a neutral scheduler trace: its callback can
split graph execution.

The CPU tests check selection, limits and serialization. The source-order test
checks copy placement and absence of an extra submit/wait in that path; it does
not prove D3D12 state transitions, resource lifetime, or behavior on Xbox.

`python3 tools/xbox/tests/rkva_mq_capture_compare.py first.bin second.bin`
validates both headers, extents and SHA-256 and prints only metadata and equality
results, never captured values. It compares Q, Qrot, positions, partials, O,
LSE and each physical KV region independently. Until the consumed physical spans
are proven, unequal KV planes are explicitly inconclusive, even when only
padding differs. Equal full planes plus equal descriptor and constants prove
their physical contents match; matching Q/Qrot/positions is reported separately.
