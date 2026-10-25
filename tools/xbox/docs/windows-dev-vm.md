# Windows UWP Development VM

> This reference was imported from xllama. For the integrated BeeLlama layout,
> current build commands and the single core source root, use [BeeLlama Xbox](../README.md).
> Historical worktree and benchmark paths describe the earlier project.

The Xbox/UWP package cannot be built in a Linux Docker container. The local
Docker daemon is a Linux container host, while Windows containers require a
Windows host. Build and package xllama in a Windows VM or use the existing
`build-uwp` GitHub Actions workflow pinned to `windows-2022`.

## Host preflight

On the Arch host:

```bash
./scripts/check-uwp-host.sh
```

Expected host tools:

- Docker can stay Linux-only; it is useful for other work, not UWP packaging.
- QEMU/libvirt provide the Windows VM runtime.
- The user should be in `kvm` and `libvirt`.
- Start libvirt before creating or running the VM:

```bash
sudo systemctl enable --now libvirtd
```

Use at least 8 GB RAM and 70 GB disk for the Windows VM. More disk is better
because Visual Studio, Windows SDKs, and package artifacts are large.

## VM setup

Use either a Microsoft Windows developer VM, if downloads are available, or a
regular Windows 11 x64 ISO/evaluation install. The Microsoft developer VM is
preferred when available because it already includes Visual Studio 2022 with
UWP workloads, but Microsoft may temporarily disable those downloads.

For a manual Windows 11 VM:

1. Create a VM with `virt-manager` or `virt-install`.
2. Enable Developer Mode in Windows settings.
3. Install Git for Windows and PowerShell.
4. Open an elevated PowerShell prompt.
5. Clone this repository with submodules.
6. Run the Windows UWP setup check:

```powershell
.\scripts\setup-windows-uwp-dev.ps1 -Install
```

If Visual Studio is already installed, omit `-Install` to verify only:

```powershell
.\scripts\setup-windows-uwp-dev.ps1
```

The check must find:

- Visual Studio 2022 Build Tools or Community
- MSBuild
- Windows SDK tools: `MakeAppx.exe` and `signtool.exe`
- `nuget.exe`

## Build

Inside the Windows VM:

```powershell
git submodule update --init --recursive
.\scripts\build-uwp.ps1 -Configuration Release -Platform x64
```

The build script:

1. Restores NuGet packages (`nuget restore`).
2. Builds with MSBuild (`-Backend unified` matches the shipping artifact;
   `llamacpp` is the bench-only lane and the script default is an ORT-only local build).
3. Signs the package with the test certificate.

No model is packaged: distribution artifacts are prepared separately and must be self-contained (`scripts/merge_onnx_external_data.py`, `docs/uwp-constraints.md §8`).

The package output is under:

```text
uwp\AppPackages\
```

The final artifact set must include:

- the main `.msix`
- dependency `.appx` files from `Dependencies\x64`
- `uwp\xllama-test.cer`

## Incremental Xbox iteration on the Windows build host

The configured `.193` host reuses the `XllamaMtpBuild` interactive scheduled
task and its existing signing certificate. From the dedicated build worktree:

```powershell
Set-Location C:\Users\hjotha\worktrees\xllama-mtp-fast
.\scripts\run-xbox-build.ps1 -BuildRevision 126
.\scripts\run-xbox-build.ps1 -BuildRevision 127 -Final -TimeoutSeconds 7200
```

The first command builds incrementally. The inference/ggml sources retain
`/O2` and AVX2; UI glue uses its existing WinRT precompiled header and skips
the optimizer. Whole-program optimization and LTCG are disabled. Objects
and outputs use separate `Release-iteration` directories, NuGet packages
are reused, and unchanged CRT DLLs keep their timestamps. The wrapper fixes
the console's VCLibs dependency and signs the resulting package once.
Use a revision greater than the last generated package; R125 was the last
package generated during this campaign, and R124 is the installed delivery.

`-Final` builds optimized Release with LTCG. Its cache is separate and reusable;
the first final build compiles the native library in that mode. An explicit
`build-uwp.ps1 -Clean` cleans the solution before building the native library
and app, so the app build cannot remove a library that was just produced.
Final intermediates use a project prefix ending in `-final`, so external
source paths containing `..\` cannot resolve to the iteration objects.
Both commands select the SDK already installed on the host, wait for the
completion event, report the package hash, and refuse to replace an active
build. Transfer only files whose content changed and verify their hashes
before invoking the build, so unchanged inputs retain their timestamps.

The initial cold iteration cache took 1,445 seconds on `.193`; subsequent
signed packages with source changes took 75–186 seconds in the 2026-10-07
campaign. The [R120 build log](../bench/results/fast-mtp-20261007/r120-build.log)
records the mode, SDK, elapsed time and signed package hash.
Recreating the iteration PCH/UI cache took 279 seconds (R121); changing the
native attention kernel and its startup control took 270 seconds (R123), with
one native compile command. These changes reused the remaining native objects.
After the final R124 build, a source-unchanged iteration R125 took 41 seconds
and changed no native or app objects in either cache. R125 was archived for
the cache check; the console delivery remains the optimized R124 package.

## Deploy and diagnose

From the Linux host, after copying the artifacts back:

```bash
source ~/.config/xllama/xbox-env
./scripts/deploy.sh path/to/xllama_*.msix
./scripts/deploy.sh diagnose-startup
```

The startup log should show:

```text
[xllama] App::App()
[xllama] App::App() complete
[xllama] App::OnLaunched
[xllama] building MainPageController
[xllama] MainPageController built
[xllama] MainPageController init done
[xllama] Window.Content set
[xllama] Window activated
```

If the app crashes before printing any log entry, collect a minidump via WDP:

```bash
curl --basic -u "${XBOX_USER}:${XBOX_PASS}" -k -sS \
     "https://${XBOX_IP}:11443/api/debug/dump/usermode/dumps"
```

Use `./scripts/deploy.sh diagnose-startup` for process state, logs and minidumps;
Device Portal details are in `docs/device-portal.md`.
