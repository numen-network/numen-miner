param(
    [ValidateRange(0, 1024)]
    [int]$Device = 0,
    [ValidateRange(10, 86400)]
    [int]$Duration = 60,
    [ValidateRange(32, 1000000)]
    [int]$RetentionCount = 512
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$miner = Join-Path $root 'target\release\numen-miner.exe'
$harness = Join-Path $root 'target\release\numen-harness.exe'
if (-not (Test-Path -LiteralPath $miner) -or -not (Test-Path -LiteralPath $harness)) {
    throw 'Build numen-miner and numen-harness in release mode before running this script.'
}
if (Get-Process -Name 'numen-miner' -ErrorAction SilentlyContinue) {
    throw 'A numen-miner process is already running. Stop it before benchmarking.'
}

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$result = Join-Path $root "benchmark-results\$stamp-gpu$Device"
New-Item -ItemType Directory -Path $result | Out-Null
$benchmark = Join-Path $result 'benchmark.json'
$dump = Join-Path $result 'features.txt'
$effective = Join-Path $result 'effective.txt'

$previousLog = $env:RUST_LOG
try {
    $env:RUST_LOG = 'info'
    & $miner bench --devices $Device --duration $Duration --report-interval 10 --json $benchmark
    if ($LASTEXITCODE -ne 0) { throw "Miner benchmark failed with exit code $LASTEXITCODE." }
    if (Get-Process -Name 'numen-miner' -ErrorAction SilentlyContinue) {
        throw 'The benchmark left a numen-miner process running.'
    }
    & $miner dump --device $Device --count $RetentionCount --out $dump
    if ($LASTEXITCODE -ne 0) { throw "GPU dump failed with exit code $LASTEXITCODE." }
    & $harness gpu-power --from $dump --benchmark $benchmark --device "gpu$Device" 2>&1 |
        Tee-Object -FilePath $effective
    if ($LASTEXITCODE -ne 0) { throw "Effective-power grading failed with exit code $LASTEXITCODE." }
} finally {
    $env:RUST_LOG = $previousLog
}

Write-Host "Benchmark artifacts: $result"
