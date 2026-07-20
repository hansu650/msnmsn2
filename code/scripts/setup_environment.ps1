param([switch]$InstallMissing)

. (Join-Path $PSScriptRoot 'common.ps1')
$context = Initialize-EviPatchEnvironment
$requirements = Assert-EviPatchPath -Path (Join-Path $context.Root 'code\requirements.txt')

& $context.Python -m pip check
if ($LASTEXITCODE -ne 0 -and -not $InstallMissing) {
    throw 'pip check failed. Re-run with -InstallMissing only if the failure is a genuinely missing project dependency.'
}
if ($InstallMissing) {
    & $context.Python -m pip install -r $requirements
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    & $context.Python -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'pip check still fails after dependency installation.' }
}

& $context.Python -c "import sys, torch, torchvision; print(sys.version); print('torch='+torch.__version__); print('torchvision='+torchvision.__version__); print('cuda_available='+str(torch.cuda.is_available())); print('gpu='+(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'))"
if ($LASTEXITCODE -ne 0) { throw 'Python/CUDA verification failed.' }
Invoke-EviPatchRunner -RunnerArgs @('validate')
