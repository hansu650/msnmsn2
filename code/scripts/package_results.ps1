. (Join-Path $PSScriptRoot 'common.ps1')
Invoke-EviPatchRunner -RunnerArgs @('aggregate')
Invoke-EviPatchRunner -RunnerArgs @('package')
