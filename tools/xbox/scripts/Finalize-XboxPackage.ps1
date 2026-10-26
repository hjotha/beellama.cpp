#Requires -Version 5.1
# Produce the same Xbox-compatible, signed package for local builds and CI.
param(
    [Parameter(Mandatory)][string]$PackagePath,
    [Parameter(Mandatory)][string]$PfxPath,
    [Parameter(Mandatory)][string]$CerPath,
    [Parameter(Mandatory)][ValidateSet('ort', 'llamacpp', 'unified')][string]$Backend,
    [string]$CertificatePassword = 'xllama-test',
    [string]$WindowsSdkVersion = '',
    [string]$OutputDirectory = '',
    [string]$CrtDirectory = ''
)
# The app imports these desktop /MD names even in the llama-only lane.
# Validate root-level x64 PE payloads, not a dependency package or an old MSIX.
function Get-XboxCrtHashes([string]$Directory, [hashtable]$ExpectedHashes = @{}) {
    $hashes = @{}
    foreach ($dll in @('MSVCP140.dll', 'MSVCP140_1.dll', 'VCRUNTIME140.dll', 'VCRUNTIME140_1.dll')) {
        $path = Join-Path $Directory $dll
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required app-local CRT missing: $path" }
        $stream = [IO.File]::OpenRead($path)
        $reader = New-Object IO.BinaryReader($stream)
        try {
            if ($stream.Length -lt 64 -or $reader.ReadUInt16() -ne 0x5A4D) { throw "CRT is not a PE DLL: $path" }
            $stream.Position = 0x3C
            $pe = $reader.ReadUInt32()
            if ($pe -gt $stream.Length - 24) { throw "CRT PE header is truncated: $path" }
            $stream.Position = $pe
            if ($reader.ReadUInt32() -ne 0x4550 -or $reader.ReadUInt16() -ne 0x8664) { throw "CRT must be an x64 PE DLL: $path" }
            $stream.Position = $pe + 22
            if (($reader.ReadUInt16() -band 0x2000) -eq 0) { throw "CRT PE payload is not a DLL: $path" }
        } finally { $reader.Dispose() }
        $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
        if ($ExpectedHashes.ContainsKey($dll) -and $ExpectedHashes[$dll] -ne $hash) { throw "CRT hash mismatch: $dll" }
        $hashes[$dll] = $hash
    }
    return $hashes
}
$ErrorActionPreference = 'Stop'
if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) { throw 'Xbox package finalization requires Windows SDK tools.' }
foreach ($required in @($PackagePath, $PfxPath, $CerPath)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Required package/signing input missing: $required" }
}
$PackagePath = (Resolve-Path -LiteralPath $PackagePath).Path
$PfxPath = (Resolve-Path -LiteralPath $PfxPath).Path
$CerPath = (Resolve-Path -LiteralPath $CerPath).Path
if (-not $OutputDirectory) { $OutputDirectory = Join-Path (Split-Path $PSScriptRoot -Parent) 'uwp\AppPackages\Finalized' }
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$OutputDirectory = (Resolve-Path -LiteralPath $OutputDirectory).Path
$sdkBin = Join-Path ${env:ProgramFiles(x86)} 'Windows Kits\10\bin'
if ($WindowsSdkVersion) {
    $sdkTools = Join-Path $sdkBin "$WindowsSdkVersion\x64"
} else {
    $sdk = Get-ChildItem -LiteralPath $sdkBin -Directory |
        Where-Object { $_.Name -match '^\d+\.\d+\.\d+\.\d+$' -and
            (Test-Path (Join-Path $_.FullName 'x64\makeappx.exe')) -and
            (Test-Path (Join-Path $_.FullName 'x64\signtool.exe')) } |
        Sort-Object { [version]$_.Name } -Descending | Select-Object -First 1
    if (-not $sdk) { throw 'No installed x64 Windows SDK packaging tools were found.' }
    $sdkTools = Join-Path $sdk.FullName 'x64'
}
$makeappx = Join-Path $sdkTools 'makeappx.exe'
$signtool = Join-Path $sdkTools 'signtool.exe'
foreach ($tool in @($makeappx, $signtool)) {
    if (-not (Test-Path -LiteralPath $tool -PathType Leaf)) { throw "Windows SDK tool missing: $tool" }
}
$temp = Join-Path $OutputDirectory ('.finalize-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $temp | Out-Null
# Start-Process captures native stderr without PS 5.1 ErrorActionPreference=Stop
# turning an expected signtool untrusted-root exit into a terminating exception.
function Invoke-PackageTool([string]$Tool, [string[]]$ToolArguments, [string]$Step) {
    $quoted = @($ToolArguments | ForEach-Object { '"' + ($_ -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"' })
    $stdout = Join-Path $temp "$Step.out"
    $stderr = Join-Path $temp "$Step.err"
    $process = Start-Process -FilePath $Tool -ArgumentList ($quoted -join ' ') -Wait -PassThru -NoNewWindow `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $lines = @(Get-Content -LiteralPath $stdout; Get-Content -LiteralPath $stderr)
    $lines | Select-Object -Last 8 | ForEach-Object { Write-Host $_ }
    return [pscustomobject]@{ ExitCode = $process.ExitCode; Text = ($lines -join "`n") }
}
function Assert-ToolSuccess($Result, [string]$Step) {
    if ($Result.ExitCode -ne 0) { throw "$Step failed (exit $($Result.ExitCode)): $($Result.Text)" }
}
try {
    Write-Host "Xbox finalizer SDK: $sdkTools"
    $unpack = Join-Path $temp 'unpacked'
    Assert-ToolSuccess (Invoke-PackageTool $makeappx @('unpack', '/p', $PackagePath, '/d', $unpack) 'unpack') 'makeappx unpack'
    if ($CrtDirectory) {
        $crtHashes = Get-XboxCrtHashes $CrtDirectory
        foreach ($dll in $crtHashes.Keys) {
            Copy-Item -LiteralPath (Join-Path $CrtDirectory $dll) -Destination (Join-Path $unpack $dll) -Force
        }
        $null = Get-XboxCrtHashes $unpack $crtHashes
    } else {
        $crtHashes = Get-XboxCrtHashes $unpack
    }
    $manifest = Join-Path $unpack 'AppxManifest.xml'
    $utf8 = New-Object System.Text.UTF8Encoding($false, $true)
    $beforeBytes = [IO.File]::ReadAllBytes($manifest)
    $before = $utf8.GetString($beforeBytes)
    # Match the dependency tag independently of attribute order. Replace only
    # the version digits: UTF-8/BOM, whitespace and all other bytes survive.
    $tagPattern = [regex]'(?s)<(?:\w+:)?PackageDependency\b[^>]*>'
    $vcTags = @($tagPattern.Matches($before) | Where-Object { $_.Value -match '\bName\s*=\s*"Microsoft\.VCLibs\.140\.00"' })
    if ($vcTags.Count -ne 1) { throw 'Expected exactly one Microsoft.VCLibs.140.00 package dependency.' }
    $versionPattern = [regex]'(\bMinVersion\s*=\s*")[0-9]+(?:\.[0-9]+){3}(")'
    if ($versionPattern.Matches($vcTags[0].Value).Count -ne 1) { throw 'VCLibs MinVersion is missing or ambiguous.' }
    $newTag = $versionPattern.Replace($vcTags[0].Value, '${1}14.0.0.0${2}', 1)
    $after = $before.Substring(0, $vcTags[0].Index) + $newTag + $before.Substring($vcTags[0].Index + $vcTags[0].Length)
    [IO.File]::WriteAllBytes($manifest, $utf8.GetBytes($after))
    [xml]$manifestXml = $after.TrimStart([char]0xFEFF)
    $identity = $manifestXml.SelectSingleNode("/*[local-name()='Package']/*[local-name()='Identity']")
    $dependency = $manifestXml.SelectSingleNode("/*[local-name()='Package']/*[local-name()='Dependencies']/*[local-name()='PackageDependency'][@Name='Microsoft.VCLibs.140.00']")
    if (-not $identity -or $identity.Version -notmatch '^\d+\.\d+\.\d+\.\d+$' -or $dependency.MinVersion -ne '14.0.0.0') {
        throw 'Final manifest identity or VCLibs dependency validation failed.'
    }
    $expectedCert = New-Object System.Security.Cryptography.X509Certificates.X509Certificate2($CerPath)
    $now = Get-Date
    if ($now -lt $expectedCert.NotBefore -or $now -gt $expectedCert.NotAfter) { throw 'Expected signing certificate is outside its validity period.' }
    if ($identity.Publisher -ne $expectedCert.Subject) { throw 'Package publisher differs from the expected signing certificate.' }
    $stem = "beellama-xbox-$($identity.Version)-$Backend"
    $candidate = Join-Path $temp "$stem.msix"
    # The old signature no longer authenticates the modified manifest.
    Remove-Item -LiteralPath (Join-Path $unpack 'AppxSignature.p7x') -Force -ErrorAction SilentlyContinue
    Assert-ToolSuccess (Invoke-PackageTool $makeappx @('pack', '/d', $unpack, '/p', $candidate, '/o') 'pack') 'makeappx pack'
    Assert-ToolSuccess (Invoke-PackageTool $signtool @('sign', '/f', $PfxPath, '/p', $CertificatePassword, '/fd', 'sha256', $candidate) 'sign') 'signtool sign'
    $verifiedHash = (Get-FileHash -LiteralPath $candidate -Algorithm SHA256).Hash
    $verification = Invoke-PackageTool $signtool @('verify', '/pa', '/v', $candidate) 'verify'
    $auth = Get-AuthenticodeSignature -FilePath $candidate
    if (-not $auth.SignerCertificate -or $auth.SignerCertificate.Thumbprint -ne $expectedCert.Thumbprint -or
        @('HashMismatch', 'NotSigned', 'NotSupportedFileFormat', 'Incompatible') -contains ([string]$auth.Status)) {
        throw "Package signature or signer mismatch (Authenticode status: $($auth.Status))."
    }
    if ($verification.ExitCode -ne 0) {
        $untrustedRoot = $verification.Text -match 'terminated in a root' -and
            $verification.Text -match 'certificate which is not trusted by the trust provider'
        $selfSigned = $expectedCert.Subject -eq $expectedCert.Issuer
        $statusAllowed = @('Valid', 'UnknownError', 'NotTrusted') -contains ([string]$auth.Status)
        if ($verification.ExitCode -ne 1 -or -not $untrustedRoot -or -not $selfSigned -or -not $statusAllowed -or
            $verification.Text -match 'hash mismatch|bad digest|not digitally signed|no signature found') {
            throw "Package verification failed (exit $($verification.ExitCode)): $($verification.Text)"
        }
        Write-Host 'Signature verified with the expected self-signed certificate; build-host root trust is absent.'
    }
    $verifyDir = Join-Path $temp 'verified'
    Assert-ToolSuccess (Invoke-PackageTool $makeappx @('unpack', '/p', $candidate, '/d', $verifyDir) 'verify-manifest') 'final package unpack'
    # Check the signed, re-unpacked artifact before publishing any destination.
    $null = Get-XboxCrtHashes $verifyDir $crtHashes
    foreach ($dll in ($crtHashes.Keys | Sort-Object)) { Write-Host "Final CRT $dll SHA256=$($crtHashes[$dll])" }
    $actualBytes = [IO.File]::ReadAllBytes((Join-Path $verifyDir 'AppxManifest.xml'))
    if ([Convert]::ToBase64String($actualBytes) -ne [Convert]::ToBase64String($utf8.GetBytes($after)) -or
        (Get-FileHash -LiteralPath $candidate -Algorithm SHA256).Hash -ne $verifiedHash) {
        throw 'Final package manifest or package hash changed during verification.'
    }
    $destination = Join-Path $OutputDirectory "$stem.msix"
    Move-Item -LiteralPath $candidate -Destination $destination -Force
    Copy-Item -LiteralPath $CerPath -Destination (Join-Path $OutputDirectory "$stem.cer") -Force
    Set-Content -LiteralPath (Join-Path $OutputDirectory "$stem.sha256") -Value "$verifiedHash  $stem.msix" -Encoding ASCII
    $dependencies = Join-Path (Split-Path $PackagePath -Parent) 'Dependencies'
    if (Test-Path -LiteralPath $dependencies) { Copy-Item -LiteralPath $dependencies -Destination $OutputDirectory -Recurse -Force }
    Write-Host "Final manifest: version=$($identity.Version) publisher=$($identity.Publisher) VCLibs.MinVersion=$($dependency.MinVersion)"
    Write-Host "Final package SHA256: $verifiedHash"
    Write-Output $destination
} finally {
    Remove-Item -LiteralPath $temp -Recurse -Force -ErrorAction SilentlyContinue
}
