# BeeLlama Xbox

The Xbox Series S|X Dev Mode app lives in this repository. It compiles the
BeeLlama core in the repository root: there is no nested `llama.cpp` checkout,
submodule update, or build-time core patch step.

The frontend was imported from `hjotha/xllama-qwen35-xbox` at `41b8f956`.
Its MIT notices and `xllama` C++/WinRT namespace remain intact. The existing
`GianlucaMazza.xllama` package identity and signing publisher are retained so
an update can preserve installed application data. The display name is
**BeeLlama Xbox**. Never uninstall the existing app just to update it: uninstall
removes its LocalState, including provisioned models and saved sessions.

## Windows build

Install Visual Studio 2022 with the UWP C++ workload, C++/WinRT, NuGet and a
Windows SDK. From the BeeLlama repository root in an x64 developer environment:

```powershell
pwsh -File scripts/build-xbox.ps1 -Configuration Release -Backend llamacpp -WindowsSdkVersion 10.0.26100.0 -BuildRevision 287 -MaxParallel 10
```

`llamacpp` is the default and uses the root BeeLlama core for GGUF inference,
including Qwen3.5. The same app's `-Backend unified` option additionally enables
ORT/DirectML models; its external NuGet/vendor dependencies remain optional.
`-Iteration` keeps optimized AVX2 inference while disabling LTCG for incremental
work. Omitting it produces the full release optimization mode. Select the SDK
installed on the actual build host and a package revision newer than the
installed package. No model or private signing certificate is committed.

For the configured Windows build host, use the existing desktop signing session
through the dedicated task (the old xllama task is left intact):

```powershell
pwsh -File tools/xbox/scripts/run-xbox-build.ps1 -BuildRevision 287 -TimeoutSeconds 7200
```

This creates/reuses `BeeLlamaXboxBuild`, builds in the current BeeLlama checkout,
reuses its signing certificate, waits on the completion event, and copies the
finalized Xbox-compatible MSIX under the build work directory.
Use `-Final` for the separate full-release cache. Build output and the result
marker are under `C:\Users\hjotha\build\beellama-xbox` by default.

All build paths use the same package finalizer: it normalizes the console's
VCLibs minimum-version dependency, repacks, signs, verifies the expected signer
and rechecks the actual packaged manifest. Final packages, matching public
certificates and SHA256 files are under `tools/xbox/uwp/AppPackages/Finalized`.
The configured-host wrapper copies the verified package to `beellama-xbox.msix`
in its build work directory. Reuse the existing certificate for in-place updates.
Private certificates stay outside Git.

A manually dispatched **Xbox UWP** GitHub Actions workflow builds the same
sources on Windows and uploads only finalized artifacts. Set its required
`build_revision` input newer than the installed package; a new workflow's run
counter is not the existing app's version counter. A package build alone does
not establish runtime correctness on the console.

## Linux host checks

The host frontend links the same `llama` target as the rest of BeeLlama:

```sh
cmake -S . -B build-xbox-host -DLLAMA_BUILD_XBOX_HOST=ON -DXLLAMA_BUILD_TESTS=ON -DGGML_CUDA=OFF -DGGML_VULKAN=OFF
cmake --build build-xbox-host --target xllama-cli xllama-tests -j10
ctest --test-dir build-xbox-host -R '^xllama-tests$' --output-on-failure
```

Console, D3D12 hardware and model-dependent tests are separate from these host
checks. The app can import GGUF models using its existing model provisioning
and Device Portal flows; see [device-portal.md](docs/device-portal.md) and
[uwp-constraints.md](docs/uwp-constraints.md).

## Where to continue RKVA

- `uwp/rkva/`: the shipping app's D3D12 remote-attention service.
- `src/bridge/ggml_d3d12.cpp`: app-local D3D12 tensor backend.
- `shaders/rkva/` and `shaders/generated/`: HLSL and compiled shader payloads.
- Repository root `ggml/src/ggml-remote-attn/` and `src/`: host protocol,
  scheduler, cache and model changes.
- `uwp/BeeLlama.props`: one core source root for MSBuild.
- `uwp/ggml-uwp.vcxproj`: AppContainer build of that core.

The original worktrees remain historical references; this app does not depend
on their paths or build caches. The earlier standalone DFlash research app is
not a second application in this import. Current RKVA includes the newer bulk,
Q4/Q8, range and lifetime handling; older 193 branch trees must not replace it.
The 193 TV dashboard was merged separately, preserving current API/session
behavior. Source provenance is recorded in `import-provenance.json`.
