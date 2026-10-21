# Ming Voice launcher: starts the WSL server in its own window (closing that window stops the server),
# waits until the models are loaded, then opens the app in the default browser.
#
# Settings (environment variables, optional):
#   MING_WSL_DISTRO  WSL distribution that holds the project      (default: Ubuntu)
#   MING_WSL_DIR     project directory inside that distribution   (default: the folder this script is in,
#                    if it lives on \\wsl$ / \\wsl.localhost; otherwise ~/ming-voice)
#   MING_PORT        server port                                  (default: 8000)
#   -Phone           also start phone access (windows\lan_proxy.py: https on port 8443 of this PC)
param([switch]$Phone)
$distro = if ($env:MING_WSL_DISTRO) { $env:MING_WSL_DISTRO } else { 'Ubuntu' }
$port = if ($env:MING_PORT) { $env:MING_PORT } else { '8000' }
$dir = $env:MING_WSL_DIR
if (-not $dir) {
    $here = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
    if ($here -match '^\\\\wsl(\.localhost|\$)\\([^\\]+)(\\.*)$') {
        $distro = $Matches[2]
        $dir = $Matches[3] -replace '\\', '/'
    } else {
        $dir = '~/ming-voice'
    }
}
$url = "http://localhost:$port"
$health = "$url/api/health"

function Get-Health {
    try { return Invoke-RestMethod -Uri $health -TimeoutSec 2 } catch { return $null }
}

function Start-PhoneAccess {
    $proxy = Join-Path $PSScriptRoot 'lan_proxy.py'
    $info = Join-Path (Split-Path -Parent $PSScriptRoot) '.lan_info.json'
    if (Test-Path $info) {  # already running? (it refreshes the file every 10 s)
        try { $i = Get-Content $info -Raw | ConvertFrom-Json; if (([DateTimeOffset]::UtcNow.ToUnixTimeSeconds() - $i.heartbeat) -lt 30) { return } } catch {}
    }
    $py = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $py) { $py = (Get-Command py -ErrorAction SilentlyContinue).Source }
    if (-not $py) { Write-Host 'Phone access needs Python for Windows (python.org).'; return }
    Start-Process -FilePath $py -ArgumentList @("`"$proxy`"") -WindowStyle Minimized
    Write-Host 'Phone access started (https, port 8443). Press Phone in the app for the QR code.'
}

$h = Get-Health
if ($Phone) { Start-PhoneAccess }
if ($h -and $h.ready) { Start-Process $url; exit 0 }

if (-not $h) {
    $cmd = "PORT=$port $dir/run.sh 2>&1 | grep --line-buffered -vE 'Warning|warn\(|bos_token|eos_token|scaled_dot|theta is None' | tee $dir/server.log"
    Start-Process -FilePath 'wsl.exe' -ArgumentList @('-d', $distro, '--', 'bash', '-lc', "`"$cmd`"") -WindowStyle Normal
}

Write-Host "Starting Ming Voice ($distro`:$dir) - loading the models onto the GPU (usually under a minute)..."
$t0 = Get-Date
while ($true) {
    Start-Sleep -Seconds 3
    $h = Get-Health
    if ($h -and $h.ready) { break }
    $s = [int]((Get-Date) - $t0).TotalSeconds
    Write-Host -NoNewline "`r  waiting... $s s "
    if ($s -gt 900) {
        Write-Host "`nThe server did not become ready. Check the server window for errors."
        Read-Host 'Press Enter to close'
        exit 1
    }
}
Write-Host "`nReady - opening $url"
Start-Process $url
