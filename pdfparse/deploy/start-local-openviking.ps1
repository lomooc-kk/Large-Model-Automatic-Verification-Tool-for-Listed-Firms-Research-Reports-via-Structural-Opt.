[CmdletBinding()]
param(
    [ValidateRange(5, 600)]
    [int]$StartupTimeoutSeconds = 120,
    [ValidateRange(1, 4)]
    [int]$EmbeddingThreads = 4
)

$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$runtimeDir = Join-Path $repoRoot 'data\local-openviking'
$pythonExe = Join-Path $repoRoot '.venv-openviking\Scripts\python.exe'
$embeddingScript = Join-Path $PSScriptRoot 'local_embedding_server.py'
$modelDir = Join-Path $runtimeDir 'models\bge-small-zh-v1.5'
$configPath = Join-Path $runtimeDir 'ov.conf'
$statePath = Join-Path $runtimeDir 'processes.json'
$bootstrap = 'from openviking_cli.server_bootstrap import main; main()'

function Write-JsonFile($Path, $Value) {
    [IO.File]::WriteAllText($Path, ($Value | ConvertTo-Json -Depth 20), [Text.UTF8Encoding]::new($false))
}

function Quote-Argument([string]$Value) {
    if ($Value.Contains('"') -or $Value.EndsWith('\')) { throw 'Unexpected executable argument syntax.' }
    return '"' + $Value + '"'
}

function Test-RoleCommand([string]$Role, [string]$CommandLine) {
    if (-not $CommandLine) { return $false }
    if ($Role -eq 'embedding') {
        return $CommandLine.Contains($embeddingScript) -and $CommandLine.Contains($modelDir) -and
            $CommandLine.Contains('--host 127.0.0.1') -and $CommandLine.Contains('--port 1934')
    }
    return $Role -eq 'openviking' -and $CommandLine.Contains($bootstrap) -and
        $CommandLine.Contains($configPath) -and $CommandLine.Contains('--host 127.0.0.1') -and
        $CommandLine.Contains('--port 1933')
}

function Save-ProcessTree([string]$Role, [int]$RootProcessId) {
    # Windows virtual-environment launchers can have a separate Python child.
    # Record the launcher and only its matching descendants, never port owners
    # discovered by name or unrelated processes that happen to use Python.
    $snapshot = @(Get-CimInstance Win32_Process)
    $known = [Collections.Generic.HashSet[int]]::new()
    [void]$known.Add($RootProcessId)
    do {
        $changed = $false
        foreach ($item in $snapshot) {
            if ($known.Contains([int]$item.ParentProcessId) -and $known.Add([int]$item.ProcessId)) {
                $changed = $true
            }
        }
    } while ($changed)
    foreach ($item in $snapshot) {
        if (-not $known.Contains([int]$item.ProcessId) -or
            -not (Test-RoleCommand $Role ([string]$item.CommandLine))) { continue }
        if (@($script:state.processes | Where-Object { $_.pid -eq $item.ProcessId }).Count) { continue }
        $script:state.processes += [ordered]@{
            role = $Role; pid = [int]$item.ProcessId
            creation_time_utc = $item.CreationDate.ToUniversalTime().ToString('o')
            executable_path = [string]$item.ExecutablePath
            command_line = [string]$item.CommandLine
        }
    }
    Write-JsonFile $statePath $script:state
}

function Wait-Service([string]$Role, [int]$Port, [int]$RootProcessId) {
    $until = [DateTime]::UtcNow.AddSeconds($StartupTimeoutSeconds)
    while ([DateTime]::UtcNow -lt $until) {
        Save-ProcessTree $Role $RootProcessId
        $alive = @($script:state.processes | Where-Object {
            $_.role -eq $Role -and (Get-Process -Id $_.pid -ErrorAction SilentlyContinue)
        })
        if (-not $alive.Count) { throw "$Role exited during startup. Inspect the log files in $runtimeDir" }
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2
            $healthy = $health.status -in @('ok', 'healthy')
            if ($Role -eq 'embedding') {
                $healthy = $healthy -and $health.model -eq 'bge-small-zh-v1.5-onnx' -and $health.dimension -eq 512
            }
            if ($healthy) {
                $listeners = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction Stop)
                $ownedIds = @($alive | ForEach-Object { $_.pid })
                if (@($listeners | Where-Object { $_.OwningProcess -notin $ownedIds }).Count) {
                    throw "Port $Port belongs to a process outside this launch."
                }
                if ($listeners.Count) { return }
            }
        } catch {
            if ($_.Exception.Message -like 'Port * belongs*') { throw }
        }
        Start-Sleep -Milliseconds 300
    }
    throw "$Role did not become healthy within $StartupTimeoutSeconds seconds. Inspect $runtimeDir"
}

foreach ($required in @($pythonExe, $embeddingScript)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Required file is missing: $required" }
}
if (-not (Test-Path -LiteralPath $modelDir -PathType Container)) { throw "Local embedding model is missing: $modelDir" }
foreach ($port in @(1933, 1934)) {
    $listeners = [Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners()
    if (@($listeners | Where-Object { $_.Port -eq $port }).Count) {
        throw "Port $port is already occupied. This script will not adopt or stop its owner. Use the companion stop script for a recorded previous launch."
    }
}
if (Test-Path -LiteralPath $statePath) {
    $previous = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    foreach ($record in @($previous.processes)) {
        if (Get-Process -Id $record.pid -ErrorAction SilentlyContinue) {
            throw 'A recorded PID is still active. Run stop-local-openviking.ps1 before starting again.'
        }
    }
}

[void][IO.Directory]::CreateDirectory($runtimeDir)
$configuration = [ordered]@{
    server = [ordered]@{ host = '127.0.0.1'; port = 1933; workers = 1; auth_mode = 'dev' }
    storage = [ordered]@{
        workspace = (Join-Path $runtimeDir 'workspace').Replace('\', '/')
        vectordb = [ordered]@{ name = 'context'; backend = 'local' }
        agfs = [ordered]@{ backend = 'local' }
    }
    embedding = [ordered]@{
        max_concurrent = 1; max_retries = 0; text_source = 'content_only'; max_input_tokens = 512
        dense = [ordered]@{
            provider = 'openai'; api_base = 'http://127.0.0.1:1934/v1'
            model = 'bge-small-zh-v1.5-onnx'; dimension = 512; encoding_format = 'float'; batch_size = 8
        }
    }
}
if (Test-Path -LiteralPath $configPath) {
    $existing = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json | ConvertTo-Json -Depth 20 -Compress
    $expected = $configuration | ConvertTo-Json -Depth 20 -Compress
    if ($existing -cne $expected) {
        throw "Existing $configPath differs from this script's local-only configuration. It has been preserved; review it before replacing it manually."
    }
} else {
    Write-JsonFile $configPath $configuration
}

$script:state = [ordered]@{
    schema_version = 1; repository = $repoRoot; python_path = $pythonExe
    config_path = $configPath; created_at_utc = [DateTime]::UtcNow.ToString('o'); processes = @()
}
Write-JsonFile $statePath $script:state
$stamp = [DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss-fff')
$started = @()
try {
    $embeddingArgs = (Quote-Argument $embeddingScript) + ' --host 127.0.0.1 --port 1934 --model-dir ' +
        (Quote-Argument $modelDir) + " --threads $EmbeddingThreads"
    $embeddingProcess = Start-Process -FilePath $pythonExe -ArgumentList $embeddingArgs -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $runtimeDir "embedding-$stamp.stdout.log") `
        -RedirectStandardError (Join-Path $runtimeDir "embedding-$stamp.stderr.log")
    $started += @{ role = 'embedding'; id = $embeddingProcess.Id }
    Save-ProcessTree 'embedding' $embeddingProcess.Id
    Wait-Service 'embedding' 1934 $embeddingProcess.Id

    $serverArgs = '-c ' + (Quote-Argument $bootstrap) + ' --config ' + (Quote-Argument $configPath) +
        ' --host 127.0.0.1 --port 1933 --workers 1'
    $serverProcess = Start-Process -FilePath $pythonExe -ArgumentList $serverArgs -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $runtimeDir "openviking-$stamp.stdout.log") `
        -RedirectStandardError (Join-Path $runtimeDir "openviking-$stamp.stderr.log")
    $started += @{ role = 'openviking'; id = $serverProcess.Id }
    Save-ProcessTree 'openviking' $serverProcess.Id
    Wait-Service 'openviking' 1933 $serverProcess.Id
    Write-Output 'OpenViking: http://127.0.0.1:1933 (local development mode; no authentication).'
    Write-Output 'Embedding: http://127.0.0.1:1934/v1 (local CPU model, dimension 512).'
    Write-Output "PID records and logs: $runtimeDir"
    Write-Output 'Health checks passed. Use vectors_only ingestion and /find; indexing and retrieval still require a separate smoke test.'
} catch {
    $failure = $_
    foreach ($launched in $started) {
        try { Save-ProcessTree $launched.role $launched.id } catch { Write-Warning 'Could not refresh a process record during cleanup.' }
    }
    try { & (Join-Path $PSScriptRoot 'stop-local-openviking.ps1') } catch { Write-Warning 'Cleanup could not verify every process; inspect processes.json before stopping anything manually.' }
    throw $failure
}
