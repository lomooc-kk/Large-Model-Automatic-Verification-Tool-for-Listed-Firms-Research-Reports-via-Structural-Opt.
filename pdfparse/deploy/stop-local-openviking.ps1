[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$runtimeDir = Join-Path $repoRoot 'data\local-openviking'
$statePath = Join-Path $runtimeDir 'processes.json'
$pythonExe = Join-Path $repoRoot '.venv-openviking\Scripts\python.exe'
$embeddingScript = Join-Path $PSScriptRoot 'local_embedding_server.py'
$modelDir = Join-Path $runtimeDir 'models\bge-small-zh-v1.5'
$configPath = Join-Path $runtimeDir 'ov.conf'
$bootstrap = 'from openviking_cli.server_bootstrap import main; main()'

if (-not (Test-Path -LiteralPath $statePath -PathType Leaf)) {
    Write-Output 'No recorded local OpenViking launch. No processes were stopped.'
    return
}
$state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
if ($state.schema_version -ne 1 -or $state.repository -cne $repoRoot -or
    $state.config_path -cne $configPath -or $state.python_path -cne $pythonExe) {
    throw 'Process record identity does not match this repository; no processes were stopped.'
}
$unverified = $false
# Stop OpenViking before embedding; stop Python children before their launchers.
$records = @($state.processes | Sort-Object @{Expression = { if ($_.role -eq 'openviking') { 0 } else { 1 } }},
    @{Expression = { $_.creation_time_utc }; Descending = $true})
foreach ($record in $records) {
    if ($record.pid -isnot [int] -and $record.pid -isnot [long]) { $unverified = $true; continue }
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $($record.pid)"
    if (-not $process) { continue }
    try {
        # PowerShell 7 parses ISO JSON values into DateTime, while Windows
        # PowerShell keeps strings. Compare UTC instants, never culture text.
        $recordedCreation = if ($record.creation_time_utc -is [DateTime]) {
            $record.creation_time_utc.ToUniversalTime()
        } else {
            [DateTime]::Parse([string]$record.creation_time_utc,
                [Globalization.CultureInfo]::InvariantCulture,
                [Globalization.DateTimeStyles]::RoundtripKind).ToUniversalTime()
        }
    } catch {
        Write-Warning "PID $($record.pid) has an invalid recorded creation time; it was preserved."
        $unverified = $true
        continue
    }
    $command = [string]$process.CommandLine
    $belongs = $false
    if ($record.role -eq 'embedding') {
        $belongs = $command.Contains($embeddingScript) -and $command.Contains($modelDir) -and
            $command.Contains('--host 127.0.0.1') -and $command.Contains('--port 1934')
    } elseif ($record.role -eq 'openviking') {
        $belongs = $command.Contains($bootstrap) -and $command.Contains($configPath) -and
            $command.Contains('--host 127.0.0.1') -and $command.Contains('--port 1933')
    }
    if (-not $belongs -or $command -cne $record.command_line -or
        [string]$process.ExecutablePath -cne $record.executable_path -or
        $process.CreationDate.ToUniversalTime().Ticks -ne $recordedCreation.Ticks) {
        Write-Warning "PID $($record.pid) no longer matches the recorded command and creation time; it was preserved."
        $unverified = $true
        continue
    }
    Stop-Process -Id $record.pid -ErrorAction Stop
    Wait-Process -Id $record.pid -Timeout 10 -ErrorAction SilentlyContinue
    Write-Output "Stopped recorded $($record.role) process $($record.pid)."
}
if ($unverified) {
    throw 'Some recorded identities could not be verified. The PID record was preserved; no unverified process was stopped.'
}
Remove-Item -LiteralPath $statePath
Write-Output 'Recorded services stopped. Local configuration, model files, logs, and indexed data were preserved.'
