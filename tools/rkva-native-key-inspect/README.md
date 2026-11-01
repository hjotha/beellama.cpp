# Read-only native key inspector

Standalone Linux/C++17 forensic inspector for the **single explicitly
qualified Qwen35 profile** with model fingerprint
`e2c19a6f1b6193fbab0a2e446b8ee6dd72cd62191ba7158c29c61b40ddbebd5c`,
64 layers, 48 recurrent layers, F32 R/S rows, 16 native KVarN4 layers and
P=102400/C=162816. Unknown model, schema, geometry or adjunct fails closed.
The recurrent part of the format does not encode its layer-presence mask or
model geometry, so the profile is intentionally fixed rather than guessed.
A manifest fingerprint and noncryptographic checksums are integrity evidence,
not authenticated model identity.

It validates the bounded unified footer and state XXH64, QSKM length and the
existing KMS2 parser/checksums, expected complete native prefix cells, the
profile's exact recurrent row schema, RKVK header/derived sizes/SHA-256 and
rollback-position rings. If present, the structured SCKP appendix is bounded
and parsed after RKVK; unknown/truncated trailing state is rejected. The key is
SHA-256 of byte zero through the **end of RKVK**, excluding SCKP, logits and
route footer. The whole-file SHA-256 is returned separately. It does not test
logits continuation, semantic tensor values or peer availability, and is not a
GC, deletion or retention authority.

```sh
cmake -S tools/rkva-native-key-inspect -B /tmp/rkva-native-key-inspect-build
cmake --build /tmp/rkva-native-key-inspect-build
/tmp/rkva-native-key-inspect-build/rkva-native-key-inspect /absolute/path/to/snapshot
```

The tool is also built by the repository tools tree on Linux when OpenSSL is
available (otherwise it is skipped with a status message):

```sh
cmake -S . -B build-cpu -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_NATIVE=ON \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --target rkva-native-key-inspect test-rkva-native-key-inspect
```

The regression binary takes an externally qualified snapshot path and is
therefore built but not registered with CTest:

```sh
build-cpu/bin/test-rkva-native-key-inspect /absolute/path/to/qualified/snapshot
```

The library offers `inspect(path, result, error)` and `inspect_fd(fd, result,
error)`. `inspect_fd` requires a read-only descriptor and duplicates it without
closing the caller's descriptor. A caller must hold its own trusted store
lock/reader lease and preserve inode immutability for this operation and any
subsequent use. Size, mtime and ctime are rechecked; they do not protect against
an adversarial writer or concurrent truncation of an mmap (which may SIGBUS).
No file locks are acquired or created. The inputs are opened read-only; no
llama context, device, network or Xbox is opened.

The optional orchestrator injection in the companion worktree is a **source
preflight only**. It leaves the existing peer verification and full-file
publication contract in place. A valid native host file alone cannot commit:
actual peer object verification, native-format publication/restore and held
reader pins are not implemented. It must not be enabled as a production native
snapshot lifecycle or used to authorize deletion.
