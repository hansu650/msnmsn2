Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-EviPatchProjectRoot {
    $root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
    if (-not (Test-Path -LiteralPath (Join-Path $root 'code\configs\stage_a.json'))) {
        throw "EviPatch project marker is missing under $root"
    }
    return $root.TrimEnd([IO.Path]::DirectorySeparatorChar)
}

function Assert-EviPatchPath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [switch]$AllowRoot
    )
    $root = Get-EviPatchProjectRoot
    $full = [IO.Path]::GetFullPath($Path)
    $prefix = $root + [IO.Path]::DirectorySeparatorChar
    $inside = $full.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)
    if (-not $inside -and -not ($AllowRoot -and $full.Equals($root, [StringComparison]::OrdinalIgnoreCase))) {
        throw "Path escapes EviPatch project root: $full"
    }
    return $full
}

function Initialize-EviPatchEnvironment {
    $root = Assert-EviPatchPath -Path (Get-EviPatchProjectRoot) -AllowRoot
    $python = Assert-EviPatchPath -Path (Join-Path $root '.conda\envs\evipatch\python.exe')
    $source = Assert-EviPatchPath -Path (Join-Path $root 'code\src')
    $tsdm = Assert-EviPatchPath -Path (Join-Path $root 'data\tsdm')
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        throw "Existing EviPatch Python is missing: $python"
    }
    if (-not (Test-Path -LiteralPath $tsdm)) {
        New-Item -ItemType Directory -Path $tsdm | Out-Null
    }
    $env:PYTHONPATH = if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) { $source } else { "$source$([IO.Path]::PathSeparator)$env:PYTHONPATH" }
    $env:EVIPATCH_PROJECT_ROOT = $root
    $env:EVIPATCH_TSDM_ROOT = $tsdm
    return @{ Root = $root; Python = $python; Source = $source; Tsdm = $tsdm }
}

function Invoke-EviPatchRunner {
    param([Parameter(Mandatory = $true)][string[]]$RunnerArgs)
    $context = Initialize-EviPatchEnvironment
    & $context.Python -m evipatch.runner @RunnerArgs
    if ($LASTEXITCODE -ne 0) {
        throw "EviPatch runner failed with exit code $LASTEXITCODE"
    }
}
