. (Join-Path $PSScriptRoot 'common.ps1')
Invoke-EviPatchRunner -RunnerArgs @('validate')
Invoke-EviPatchRunner -RunnerArgs @('stage-a')
