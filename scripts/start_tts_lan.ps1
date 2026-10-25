<#
.SYNOPSIS
    Start the LAN split-TTS stack: Qwen3-TTS on the 2080 Ti + a reverse tunnel to the Pi.

.DESCRIPTION
    Two processes, one lifetime:

      1. uvicorn / scripts/tts_server.py holding Qwen3-TTS-12Hz-0.6B-Base open
         on the RTX 2080 Ti, listening on 127.0.0.1:$Port.
      2. An SSH reverse tunnel ``-R $Port:127.0.0.1:$Port`` pushed to the Pi, so
         the Pi can call the model at http://127.0.0.1:$Port.

    Why a tunnel instead of just binding 0.0.0.0: inbound TCP to this PC is
    blocked by Windows Firewall and we have no admin rights to open the port.
    An *outbound* SSH connection is never blocked, and the reverse forward lands
    on the Pi's loopback only -- nothing is exposed to the LAN.

    The GPU half is restarted when it dies; the tunnel reconnects every 5 s.

.PARAMETER Install
    Register a logon scheduled task that runs this script hidden, then exit.

.PARAMETER Port
    TCP port used on both ends. Default 8765.

.PARAMETER Dtype
    float32 (default, stable on Turing) or bfloat16. Do NOT use float16: the
    top-k compare kernel asserts on overflow in the 2080 Ti's fp16 path.

.EXAMPLE
    .\scripts\start_tts_lan.ps1
    .\scripts\start_tts_lan.ps1 -Install
#>
[CmdletBinding()]
param(
    [switch]$Install,
    [int]$Port = 8765,
    [string]$Dtype = 'float32',
    [string]$PiHost = 'pi@192.168.100.16',
    [string]$SshKey = (Join-Path $HOME '.ssh\pi_ed25519'),
    [string]$RefAudio = 'clone_source/clip0002.wav',
    [string]$RefText = '开始计时三十秒，别走神。'
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot          # repo root
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$ServerScript = Join-Path $Root 'scripts\tts_server.py'
$LogDir = Join-Path $Root 'data\_logs'

if ($Install) {
    # powershell.exe, not pwsh: this box ships Windows PowerShell 5.1 only.
    $action = New-ScheduledTaskAction -Execute 'powershell.exe' `
        -Argument "-NoProfile -WindowStyle Hidden -File `"$PSCommandPath`"" `
        -WorkingDirectory $Root
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
    $settings = New-ScheduledTaskSettingsSet -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
        -ExecutionTimeLimit (New-TimeSpan -Days 365) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    Register-ScheduledTask -TaskName 'dfy-tts-lan' -Action $action -Trigger $trigger `
        -Settings $settings -Description 'dfy-tts: Qwen3-TTS on 2080 Ti + tunnel to Pi' -Force | Out-Null
    Write-Host "registered scheduled task 'dfy-tts-lan' (runs at logon)" -ForegroundColor Green
    return
}

if (-not (Test-Path $Python)) { throw "venv python not found: $Python" }
if (-not (Test-Path $ServerScript)) { throw "server script not found: $ServerScript" }
if (-not (Test-Path $SshKey)) { throw "ssh key not found: $SshKey (run ssh-keygen, then append it to the Pi's authorized_keys)" }

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# The Base checkpoint has no hub access here; resolve the local snapshot directly.
# Pointing at the snapshot also skips the transformers mistral-regex network
# probe, which fails hard in offline mode.
$hubBase = Join-Path $Root '.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-0.6B-Base\snapshots'
$snapshot = Get-ChildItem -Path $hubBase -Directory -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $snapshot) { throw "no Qwen3-TTS-0.6B-Base snapshot under $hubBase" }

$env:HF_HOME = Join-Path $Root '.cache\huggingface'

$serverArgs = @(
    $ServerScript,
    '--host', '127.0.0.1',
    '--port', "$Port",
    '--dtype', $Dtype,
    '--model', $snapshot.FullName,
    '--ref-root', (Join-Path $Root 'data'),
    '--default-ref', $RefAudio,
    '--default-ref-text', $RefText
)

$serverLog = Join-Path $LogDir 'tts_server.log'
$tunnelLog = Join-Path $LogDir 'tunnel.log'

function Start-GpuServer {
    Write-Host "[start] Qwen3-TTS server on 127.0.0.1:$Port ($Dtype)" -ForegroundColor Cyan
    $p = Start-Process -FilePath $Python -ArgumentList $serverArgs `
        -RedirectStandardOutput $serverLog -RedirectStandardError "$serverLog.err" `
        -WindowStyle Hidden -PassThru
    # Wait for the model to answer, not merely for the port to open.
    # curl.exe, not Invoke-RestMethod: a system-wide proxy is configured on this
    # box and Windows PowerShell's WebRequest honours it even for 127.0.0.1.
    foreach ($i in 1..90) {
        Start-Sleep -Milliseconds 1000
        if ($p.HasExited) { throw "server exited with $($p.ExitCode); see $serverLog.err" }
        $body = curl.exe -s -m 2 "http://127.0.0.1:$Port/health" 2>$null
        if ($LASTEXITCODE -eq 0 -and $body) {
            Write-Host "[start] model ready (pid $($p.Id))" -ForegroundColor Green
            return $p
        }
    }
    throw "server did not become healthy in 90 s; see $serverLog"
}

function Start-Tunnel {
    Write-Host "[start] reverse tunnel 127.0.0.1:$Port -> $PiHost" -ForegroundColor Cyan
    $args = @(
        '-N', '-i', $SshKey,
        '-o', 'ExitOnForwardFailure=yes',
        '-o', 'StrictHostKeyChecking=accept-new',
        '-o', 'ServerAliveInterval=30',
        '-o', 'ServerAliveCountMax=3',
        '-R', "127.0.0.1:$Port`:127.0.0.1:$Port",
        $PiHost
    )
    Start-Process -FilePath 'ssh.exe' -ArgumentList $args `
        -RedirectStandardError $tunnelLog -WindowStyle Hidden -PassThru
}

$server = Start-GpuServer
try {
    $tunnel = Start-Tunnel
    Write-Host ""
    Write-Host "Pi side:  python tts_client.py --server http://127.0.0.1:$Port '你好'" -ForegroundColor Yellow
    Write-Host "Ctrl-C stops the tunnel and the GPU server." -ForegroundColor DarkGray
    # Supervise both halves forever: either one dying is recoverable and must not
    # take the other down with it.
    while ($true) {
        Start-Sleep -Seconds 5
        if ($server.HasExited) {
            Write-Host "[warn] GPU server exited ($($server.ExitCode)); restarting" -ForegroundColor Yellow
            Start-Sleep -Seconds 2
            $server = Start-GpuServer
        }
        if ($tunnel.HasExited) {
            Write-Host "[warn] tunnel died (exit $($tunnel.ExitCode)); reconnecting" -ForegroundColor Yellow
            Start-Sleep -Seconds 5
            $tunnel = Start-Tunnel
        }
    }
} finally {
    foreach ($p in @($server, $tunnel)) {
        if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    }
}
