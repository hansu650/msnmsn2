. (Join-Path $PSScriptRoot 'common.ps1')
$context = Initialize-EviPatchEnvironment
$apn = Assert-EviPatchPath -Path (Join-Path $context.Root 'vendor\APN')
$patch = Assert-EviPatchPath -Path (Join-Path $context.Root 'patches\apn_evipatch.patch')
$expected = 'f0d6eeb7a2ee2d7c76475bf725b7ea25f98af3f4'

if (-not (Test-Path -LiteralPath (Join-Path $apn '.git'))) { throw "Pinned APN checkout is missing: $apn" }
if (-not (Test-Path -LiteralPath $patch -PathType Leaf)) { throw "EviPatch patch is missing: $patch" }
$actual = (& git -C $apn rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $actual -ne $expected) { throw "APN commit mismatch: expected $expected, got $actual" }

& git -C $apn apply --reverse --check $patch 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host 'EviPatch APN patch is already applied.'
    exit 0
}
& git -C $apn apply --check $patch
if ($LASTEXITCODE -ne 0) { throw 'Patch cannot be applied cleanly; preserve and inspect any unrelated APN changes.' }
& git -C $apn apply $patch
if ($LASTEXITCODE -ne 0) { throw 'Patch application failed.' }
& git -C $apn diff --check
if ($LASTEXITCODE -ne 0) { throw 'Patched APN failed git diff --check.' }
Write-Host "Applied EviPatch to APN commit $expected"
