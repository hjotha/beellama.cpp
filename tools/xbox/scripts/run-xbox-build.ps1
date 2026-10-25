#Requires -Version 5.1
# Reuse the existing interactive task/certificate context, with an event wait.
param(
    [Parameter(Mandatory)][ValidateRange(1,65535)][int]$BuildRevision,
    [switch]$Final,
    [ValidateSet("llamacpp", "unified")][string]$Backend = "llamacpp",
    [int]$TimeoutSeconds = 1800,
    [string]$WorkRoot = "C:\Users\hjotha\build\beellama-xbox"
)
$ErrorActionPreference = "Stop"
$service = New-Object -ComObject Schedule.Service
$service.Connect()
$folder = $service.GetFolder('\')
try { $task = $folder.GetTask('BeeLlamaXboxBuild') }
catch {
    # Reuse the desktop signing principal without changing the old app task.
    $template = $folder.GetTask('XllamaMtpBuild')
    $null = $folder.RegisterTaskDefinition('BeeLlamaXboxBuild', $template.Definition, 6, $null, $null, 3, $null)
    $task = $folder.GetTask('BeeLlamaXboxBuild')
}
if ($task.State -eq 2 -or $task.State -eq 4) { throw "An existing Xbox build is queued or running; preserve it." }
$definition = $task.Definition
$builder = Join-Path $PSScriptRoot 'build-xbox-local.ps1'
$definition.Actions.Item(1).WorkingDirectory = (Split-Path $PSScriptRoot -Parent)
$definition.Actions.Item(1).Arguments = '-File "{0}" -BuildRevision {1} -WorkRoot "{2}" -Backend {4}{3}' -f `
    $builder, $BuildRevision, $WorkRoot, $(if ($Final) { ' -Final' } else { '' }), $Backend
$null = $folder.RegisterTaskDefinition('BeeLlamaXboxBuild', $definition, 4, $null, $null, 3, $null)
New-Item -ItemType Directory -Force -Path $WorkRoot | Out-Null
$marker = Join-Path $WorkRoot 'exit.txt'
Remove-Item $marker -Force -ErrorAction SilentlyContinue
$watch = New-Object IO.FileSystemWatcher($WorkRoot, 'exit.txt')
$watch.EnableRaisingEvents = $true
$started = Get-Date
try {
    $run = $folder.GetTask('BeeLlamaXboxBuild').Run($null)
    Write-Output "Started revision $BuildRevision, instance=$($run.InstanceGuid)"
    $deadline = $started.AddSeconds($TimeoutSeconds)
    while (-not (Test-Path $marker)) {
        $remaining = [int][Math]::Max(1, ($deadline - (Get-Date)).TotalMilliseconds)
        $event = $watch.WaitForChanged([IO.WatcherChangeTypes]::Created -bor [IO.WatcherChangeTypes]::Changed -bor [IO.WatcherChangeTypes]::Renamed, $remaining)
        if ($event.TimedOut) { throw "Build completion timeout; the running build was preserved." }
    }
    $result = (Get-Content $marker -Raw).Trim()
    Get-Content (Join-Path $WorkRoot 'build-local.log') -Tail 24
    Write-Output ("Total seconds: " + [Math]::Round(((Get-Date) - $started).TotalSeconds, 2))
    if ($result -ne '0') { throw "Xbox build failed: $result" }
} finally {
    $watch.Dispose()
}
