param([switch]$SkipTiming)

. (Join-Path $PSScriptRoot 'common.ps1')
Invoke-EviPatchRunner -RunnerArgs @('validate')
Invoke-EviPatchRunner -RunnerArgs @('smoke')
if (-not $SkipTiming) {
    Invoke-EviPatchRunner -RunnerArgs @('timing')
}
