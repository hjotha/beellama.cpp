#Requires -Version 5.1
# No build, signing, deployment, or legacy-package inputs. Run on a VS host.
param(
    [string]$BuildScriptPath = (Join-Path $PSScriptRoot '..\scripts\build-uwp.ps1'),
    [string]$FinalizeScriptPath = (Join-Path $PSScriptRoot '..\scripts\Finalize-XboxPackage.ps1')
)
$ErrorActionPreference = 'Stop'
foreach ($path in @($BuildScriptPath, $FinalizeScriptPath)) {
    $tokens = $null; $errors = $null
    $ast = [Management.Automation.Language.Parser]::ParseFile($path, [ref]$tokens, [ref]$errors)
    if ($errors.Count) { throw "PowerShell parse errors in ${path}: $errors" }
    foreach ($name in @('Find-XboxCrtRedist', 'Get-XboxCrtHashes')) {
        $definition = $ast.Find({ param($node)
            $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name
        }, $true)
        if ($definition) { . ([scriptblock]::Create($definition.Extent.Text)) }
    }
}
function Require-Failure([scriptblock]$Action, [string]$Message) {
    $rejected = $false
    try { $null = & $Action } catch { $rejected = $true }
    if (-not $rejected) { throw "Negative control accepted: $Message" }
}
$CrtDlls = @('MSVCP140.dll', 'MSVCP140_1.dll', 'VCRUNTIME140.dll', 'VCRUNTIME140_1.dll')
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$roots = @(& $vswhere -all -products '*' -property installationPath)
if ($LASTEXITCODE -ne 0) { throw 'VS enumeration failed.' }
$source = Find-XboxCrtRedist $roots
$expected = Get-XboxCrtHashes $source
Write-Host "CRT test installed source: $source"
foreach ($dll in $CrtDlls) { Write-Host "$dll SHA256=$($expected[$dll])" }
$temp = Join-Path ([IO.Path]::GetTempPath()) ('beellama-crt-test-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $temp | Out-Null
try {
    Require-Failure { Find-XboxCrtRedist @($temp) } 'no installed CRT payload'
    $fakeVs = Join-Path $temp 'VS'
    $onecore = Join-Path $fakeVs 'VC\Redist\MSVC\14.44.35112\onecore\x64\Microsoft.VC143.CRT'
    New-Item -ItemType Directory -Force -Path $onecore | Out-Null
    foreach ($dll in $CrtDlls) { Copy-Item -LiteralPath (Join-Path $source $dll) -Destination $onecore }
    Require-Failure { Find-XboxCrtRedist @($fakeVs) } 'onecore payload cannot substitute desktop CRT'
    $standard = Join-Path $fakeVs 'VC\Redist\MSVC\14.44.35112\x64\Microsoft.VC143.CRT'
    New-Item -ItemType Directory -Force -Path $standard | Out-Null
    foreach ($dll in $CrtDlls) { Copy-Item -LiteralPath (Join-Path $source $dll) -Destination $standard }
    $incomplete = Join-Path $fakeVs 'VC\Redist\MSVC\99.99.99999\x64\Microsoft.VC143.CRT'
    New-Item -ItemType Directory -Force -Path $incomplete | Out-Null
    Copy-Item -LiteralPath (Join-Path $source $CrtDlls[0]) -Destination $incomplete
    if ((Find-XboxCrtRedist @($fakeVs)) -ne $standard) { throw 'Discovery selected an incomplete or non-desktop payload.' }

    foreach ($dll in $CrtDlls) { Copy-Item -LiteralPath (Join-Path $source $dll) -Destination $temp }
    $null = Get-XboxCrtHashes $temp $expected
    foreach ($dll in $CrtDlls) {
        $path = Join-Path $temp $dll
        Remove-Item -LiteralPath $path
        Require-Failure { Get-XboxCrtHashes $temp $expected } "missing $dll"
        Copy-Item -LiteralPath (Join-Path $source $dll) -Destination $path
    }
    $path = Join-Path $temp $CrtDlls[0]
    $bytes = [IO.File]::ReadAllBytes($path)
    $original = [byte[]]$bytes.Clone()
    $pe = [BitConverter]::ToUInt32($bytes, 0x3C)
    $bytes[$pe + 4] = 0x4C; $bytes[$pe + 5] = 0x01 # x86 instead of AMD64
    [IO.File]::WriteAllBytes($path, $bytes)
    Require-Failure { Get-XboxCrtHashes $temp } 'wrong architecture'
    [IO.File]::WriteAllBytes($path, $original)
    $stream = [IO.File]::Open($path, [IO.FileMode]::Append)
    try { $stream.WriteByte(0x7F) } finally { $stream.Dispose() }
    Require-Failure { Get-XboxCrtHashes $temp $expected } 'changed DLL hash'
    [IO.File]::WriteAllBytes($path, $original)
    # The signed finalizer verifies its unpacked output using this same helper.
    # Exercise archive round-trip as a content gate, without invoking a compiler.
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $payload = Join-Path $temp 'payload'
    New-Item -ItemType Directory -Path $payload | Out-Null
    foreach ($dll in $CrtDlls) { Copy-Item -LiteralPath (Join-Path $temp $dll) -Destination $payload }
    $archive = Join-Path $temp 'runtime.msix'
    [IO.Compression.ZipFile]::CreateFromDirectory($payload, $archive)
    $verified = Join-Path $temp 'verified'
    [IO.Compression.ZipFile]::ExtractToDirectory($archive, $verified)
    $null = Get-XboxCrtHashes $verified $expected
    $finalizer = [IO.File]::ReadAllText($FinalizeScriptPath)
    if ($finalizer.IndexOf('Get-XboxCrtHashes $verifyDir $crtHashes') -lt 0 -or
        $finalizer.IndexOf('Get-XboxCrtHashes $verifyDir $crtHashes') -gt $finalizer.IndexOf('Move-Item -LiteralPath $candidate')) {
        throw 'Finalized CRT verification must precede package publication.'
    }
    Write-Host 'PASS: installed discovery, exact hashes, four missing-DLL refusals, wrong architecture, changed hash, archive round-trip, final publication gate.'
} finally { Remove-Item -LiteralPath $temp -Recurse -Force }
