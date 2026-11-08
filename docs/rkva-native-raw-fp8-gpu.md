# Native-row FP8 GPU codec candidate

Local candidate based on `15cf144a94d721298704a9fdcefa1d275346cd56`.
This is implementation and host validation, **not Xbox runtime qualification**.

## Scope and preserved contracts

New E4 raw and LOW4 E5 routes require `RKVA_EXPERIMENTAL_NATIVE_FP8_GPU=1`
(default **0**). Default behavior stays on CPU codecs for these new routes until
Windows/Xbox qualification. Existing E5 Q4/Q8/SAW/NC behavior is unchanged.

- Q4_0, SAW-INT4, TurboQuant NC, Q4_1 and IQ4_NL can use the new raw E4M3FN
  query expansion/output pack PSOs. Query expansion writes raw F32 plus positions.
  Format-specific attention still performs its own required transforms.
- Q4_1/IQ4_NL can also use the existing raw E5 codec. Their experimental LOW4
  build/availability gates remain unchanged.
- E4 output is direct F32 RNE with finite saturation at 448. There is **no F16
  intermediate**, so finite F32 larger than 65504 remains valid. KVarN E4's
  existing F16-intermediate contract and generated shaders are unchanged.
- All E4 non-finite OUT is rejected. The shader checks exact zero before packing
  when LSE is negative infinity; the CPU validates every LSE against original
  query positions. A tiny nonzero empty-prefix OUT cannot disappear by rounding.
- Existing `rkva-q4-fp8-gpu-off.flag` disables these native 4-bit GPU codecs.
  Disabled flag or missing/failed PSOs retains the existing CPU codec entry.
  An error after GPU dispatch invalidates the session and returns failure; no
  silent CPU retry occurs. `native_e4_gpu` profile status is separate from KVarN.
- This is not a CPU-free pipeline: upload, validation, orchestration, network,
  status validation and readback remain host work.

## Reproduce host checks

From repository root (GCC/Python required):

```sh
ASAN_OPTIONS=detect_leaks=0 python3 tools/xbox/tests/rkva_native_e4_raw_verify.py
DXC=/path/to/dxc bash tools/xbox/scripts/compile-rkva-range-shaders.sh \
  bulk_unpack_q_e4m3fn_raw bulk_pack_out_e4m3fn_raw
for entry in CSBulkUnpackQE4Raw CSBulkPackOutE4Raw; do
  dxc -T cs_6_0 -E "$entry" tools/xbox/shaders/rkva/rkva_range_bulk_e4_raw.hlsl -Fo "/tmp/$entry.dxil"
  dxv "/tmp/$entry.dxil"
done
for test in rkva_e4m3_cpu rkva_e4m3_policy_cpu rkva_native_e4_cpu rkva_bulk_fp8_cpu; do
  g++ -std=c++17 -O2 -fsanitize=address,undefined -fno-omit-frame-pointer \
    -Iggml/include -Iggml/src "tools/xbox/tests/$test.cpp" -o "/tmp/$test"
  ASAN_OPTIONS=detect_leaks=0 "/tmp/$test"
done
```

Leak checking is disabled only because LeakSanitizer cannot operate under this
host's ptrace environment. ASan/UBSan remain enabled. The new test compiles the
actual HLSL scalar integer helpers as C++ and compares them to the existing wire
oracle: all 254 finite E4 codes, all adjacent positive midpoint triplets with
both signs, boundary/non-finite inputs and one million deterministic F32 bit
patterns. This does not execute HLSL/DXIL dispatch, memory barriers or D3D12.

## Required hardware qualification before performance claims

1. Build Windows UWP following `tools/xbox/README.md` with these exact sources;
   set `RKVA_EXPERIMENTAL_NATIVE_FP8_GPU=1` for the candidate build and,
   for LOW4, also explicitly enable its existing experimental macro. Repeat the
   admission check with the new macro omitted/default 0.
   Package compilation is separate from deployment and requires authorization.
2. On an authorized Xbox test session, test each listed native layout over E4
   and E5 with GPU codec enabled and disabled, using identical payloads, model,
   positions, context, batches and sampling settings. Require valid OUT/LSE and
   E4 byte equality to CPU packing of the same F32 attention results. Include
   nq=1,2,3,9,256, GQA, append-only/decode, multiple segments and empty prefixes.
3. Exercise NaN/Inf Q, invalid metadata, mixed wires, truncated segments,
   malformed positions, OUT NaN/Inf and empty-prefix nonzero OUT fault injection.
   Invalid input must fail before mutation; execution errors must not retry CPU.
4. Remove either generated raw E4 header in a separate qualification build and
   inject PSO-creation failure; availability must be false and CPU codec retained.
   Assert LOW4 remains unavailable in the default build with its gate disabled.
5. Record `native_e4_gpu`, wire/storage and `fp8_gpu` telemetry with the exact
   commit, executable/package hash and device. Profile dispatches and bytes to
   distinguish codec improvement from attention or network costs. No throughput
   improvement is established by host tests or shader compilation.
