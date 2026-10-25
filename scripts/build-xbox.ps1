#Requires -Version 5.1
# Build the Xbox front end against this repository's BeeLlama core.
# Examples:
#   pwsh -File scripts/build-xbox.ps1 -Configuration Release -WindowsSdkVersion 10.0.26100.0
#   pwsh -File scripts/build-xbox.ps1 -Backend unified -Iteration
& (Join-Path $PSScriptRoot '..\tools\xbox\scripts\build-uwp.ps1') @args
exit $LASTEXITCODE
