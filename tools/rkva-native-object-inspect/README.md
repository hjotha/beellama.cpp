# Read-only RKVA native v2 local-object inspector

This is a standalone Linux/C++17 verifier for a **local directory fixture** of
an Xbox native v2 object. It has no network, Xbox, GPU, server, registry or
mutation/GC integration. Build and run it with:

```sh
cmake -S tools/rkva-native-object-inspect -B /tmp/rkva-native-object-inspect
cmake --build /tmp/rkva-native-object-inspect
ctest --test-dir /tmp/rkva-native-object-inspect --output-on-failure
/tmp/rkva-native-object-inspect/rkva-native-object-inspect OBJECT_DIRECTORY EXPECTED.json
```

The tool is also built by the repository tools tree on Linux:

```sh
cmake -S . -B build-cpu -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_NATIVE=ON \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --target rkva-native-object-inspect test-rkva-native-object-inspect
ctest --test-dir build-cpu -R test-rkva-native-object-inspect --output-on-failure
```

`EXPECTED.json` must supply identity from a separate trusted source. The
expected key is 64 lowercase hex digits and must also be the directory's final
component. `live_rows` and `saved_generation` must describe the original saved
generation, not a later resident generation. Geometry and descriptor field
names are those of `rkva_native_snapshot::Geometry` and `Descriptor`. For a
small synthetic fixture the structure is:

```json
{
  "key": "abababababababababababababababababababababababababababababababab",
  "live_rows": 513,
  "saved_generation": 30064771073,
  "geometry": {"n_head": 8, "n_head_kv": 2, "head_dim": 256, "head_slices": 2, "max_ctx": 4096, "n_layers": 1},
  "descriptor": {"origin": 1024, "capacity": 1024, "storage": 4, "layout_version": 3,
    "token_group": 128, "record_dim": 128, "head_dim": 256, "head_slices": 2,
    "key_bits": 4, "value_bits": 4, "record_bytes": 8960, "stage_groups": 3,
    "tail_groups": 2, "sink_tokens": 128, "precision_tail_tokens": 128,
    "wht_version": 1, "domain": 1, "record_policy": 1}
}
```

The verifier uses `rkva_native_snapshot::decode`, requires canonical
byte-for-byte re-encoding, then calls `validate` with the external key, complete
geometry/descriptor, live rows and saved generation. It requires precisely
`manifest-v2.bin` and the manifest-derived `n-<index>.kv` files, disallowing
extras, alternate spellings, symlinks and hardlinks. Every extent and SHA-256
is checked; chunk I/O uses 64 KiB, with manifest and entry counts bounded by
the shared codec. File and directory identity, size, mtime and ctime are
rechecked after reading. Results are published only after all checks succeed.

The re-encoding check also rejects an inconsistency in the duplicated layer
count that the current shared decoder alone overwrites rather than compares;
this tool does not modify the shared codec. The tests include a recomputed
manifest checksum for this case and for a wrong embedded key.

**The caller must hold an external lock or reader pin and ensure an immutable
object throughout verification and subsequent use.** File-stat rechecks are
best-effort race detection, not authentication or proof against all concurrent
or adversarial writers, replacement of trusted ancestors, stale copies or
changes immediately after inspection. An object successfully checked in a
local fixture does not prove that a peer contains it, that its bytes are durable,
that its data has the intended semantic values, or that it can safely be
collected. SHA-256 detects corruption but is not an authenticated signature.
This tool must not authorize GC or peer COMMIT on its own.

Inputs are opened with read-only Linux descriptors and `O_NOFOLLOW`; the tool
tries `O_NOATIME` for files where permitted, falling back to normal reads when
not permitted. A filesystem may update access times. Parent path components
must be trusted. An interrupted or uncoordinated writer must be handled by the
caller. Windows/UWP and Xbox filesystem behavior are not implemented or
qualified.

Tests create their own temporary directories and synthetic canonical chunks;
they cover CLI and API success, wrong key/geometry/rows/generation, altered or
truncated manifests/chunks, duplicate-layer forgery, extra/missing files,
symlinks, hardlinks, size/entry limits and deterministic mutation/replacement
injected during inspection. These are local process tests, not a peer,
concurrency-exclusion, network or power-loss qualification.
