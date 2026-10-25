#Requires -Version 5.1
# BeeLlama Xbox build for the configured Windows desktop signing session.
# Invoke through run-xbox-build.ps1 to reuse that interactive certificate context.
# Portable Windows builds use the repository-root scripts/build-xbox.ps1 entry.
param(
    [Parameter(Mandatory)][ValidateRange(1,65535)][int]$BuildRevision,
    [switch]$Final,
    [ValidateSet("llamacpp", "unified")][string]$Backend = "llamacpp",
    [string]$WorkRoot = "C:\Users\hjotha\build\beellama-xbox"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
$work = $WorkRoot
$sdk  = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64"
$log = "$work\build-local.log"

New-Item -ItemType Directory -Force -Path $work | Out-Null
function Log($m) { Add-Content -Path $log -Value ((Get-Date -Format "HH:mm:ss") + " " + $m) }
function Write-Exit($result) {
    Set-Content -Path "$work\exit.tmp" -Value $result
    Move-Item -Path "$work\exit.tmp" -Destination "$work\exit.txt" -Force
}

$buildLock = New-Object System.Threading.Mutex($false, "Local\BeeLlamaXboxBuild")
if (-not $buildLock.WaitOne(0)) { throw "Another xllama Xbox build owns the cache." }
try {
Set-Content -Path $log -Value ("=== START " + (Get-Date -Format s) +
    " session=" + (Get-Process -Id $PID).SessionId + " user=" + (whoami) + " ===")
Set-Location $repo
Remove-Item "$work\exit.txt" -Force -EA SilentlyContinue
$buildStarted = Get-Date
Log ("mode=" + $(if ($Final) { "final" } else { "iteration" }))

# The app compiles the repository core directly; no patching or submodule step.
Log "1/4 BeeLlama core: native AppContainer compatibility"

# 2. Build. -Backend unified compiles src/bridge/ggml_d3d12.cpp (the Q5_K kernel).
# pwsh, not powershell: build-uwp.ps1 tests $IsWindows, which PS 5.1 lacks
# (CI runs pwsh too).
# git safe.directory: the llama.cpp submodule is owned by hpbeats\hjotha but
# this build runs as hpbeats\hjoth. GenerateGgmlVersion.cmake shells out to
# git, which refuses foreign-owned repos ("dubious ownership"). The env form
# is honored without touching any global config.
$env:GIT_CONFIG_COUNT = "1"
$env:GIT_CONFIG_KEY_0 = "safe.directory"
$env:GIT_CONFIG_VALUE_0 = (Split-Path (Split-Path $repo -Parent) -Parent)
# MSVC env: a bare scheduled task has no compiler on PATH (TRK0005).
# Import the VS2022 x64 developer environment, like ilammy/msvc-dev-cmd on CI.
$vsDevCmd = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
$vsEnv = cmd /c "`"$vsDevCmd`" -arch=x64 -no_logo >nul 2>&1 && set" 2>$null
foreach ($line in $vsEnv) {
    if ($line -match '^(PATH|INCLUDE|LIB|LIBPATH)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
    }
}
Log ("VS dev env: cl.exe na PATH = " + [bool](Get-Command cl.exe -EA SilentlyContinue))
# C1033 guard: the ggml-uwp PDB goes to uwp\x64\Release\ggml-uwp\, which no
# target creates (cl.exe does not create /Fd directories). A directory wiped
# by cleanup -- or never created -- fails every compile with C1033.
foreach ($d in @("uwp\x64\Release\ggml-uwp", "uwp\xllama\x64\Release",
                 "uwp\ggml-uwp\x64\Release")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $repo $d) | Out-Null
}
Log "dirs de saida garantidos"
Log "2/4 build-uwp.ps1 -Backend $Backend -BuildRevision $BuildRevision"
$modeArgs = @()
if (-not $Final) { $modeArgs += "-Iteration" }
$modeArgs += @("-WindowsSdkVersion", (Split-Path (Split-Path $sdk -Parent) -Leaf))
Log ("build flags=" + ($modeArgs -join ' '))
$pwsh = "C:\Users\hjotha\scoop\shims\pwsh.exe"
& $pwsh -ExecutionPolicy Bypass -NoProfile -File "$repo\scripts\build-uwp.ps1" `
    -Configuration Release -Platform x64 -Backend $Backend -BuildRevision $BuildRevision @modeArgs `
    > "$work\msbuild.out" 2> "$work\msbuild.err"
$buildExit = $LASTEXITCODE
Log "build exit=$buildExit"
Get-Content "$work\msbuild.err" -Tail 25 -EA SilentlyContinue | ForEach-Object { Log ("ERR " + $_) }
if ($buildExit -ne 0) { Log "ABORT no build"; Write-Exit "build:$buildExit"; exit 1 }

# build-uwp.ps1 owns dependency normalization, signing and verification for
# both local and CI builds. Consume only its freshly finalized artifact.
$msix = Get-ChildItem "$repo\uwp\AppPackages\Finalized" -Filter "beellama-xbox-*-$Backend.msix" |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $msix -or $msix.LastWriteTime -lt $buildStarted) {
    Log "ABORT: no fresh finalized package was produced"
    Write-Exit "stale-package"
    exit 1
}
Log "finalized-msix=$($msix.FullName) size=$($msix.Length)"
$fixed = Join-Path $work 'beellama-xbox.msix'
Copy-Item -LiteralPath $msix.FullName -Destination $fixed -Force
if ((Get-FileHash -LiteralPath $fixed -Algorithm SHA256).Hash -ne
    (Get-FileHash -LiteralPath $msix.FullName -Algorithm SHA256).Hash) { throw 'Finalized package copy hash mismatch.' }
Log "sha256-fixed=$((Get-FileHash $fixed -Algorithm SHA256).Hash)"
Set-Content "$work\fixed-msix-path.txt" $fixed
Log ("elapsed_seconds=" + [math]::Round(((Get-Date)-$buildStarted).TotalSeconds,2))
Log "=== DONE ==="
Write-Exit "0"

} catch {
    Log ("ABORT: " + $_.Exception.Message)
    Write-Exit "exception"
    exit 1
} finally {
    $buildLock.ReleaseMutex()
    $buildLock.Dispose()
}
