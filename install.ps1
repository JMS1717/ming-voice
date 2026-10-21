# Ming Voice installer for Windows 11: sets up WSL2 + Ubuntu 24.04 if needed, gets the code into Ubuntu and runs
# install.sh there (ROCm on AMD, Python env, model, Desktop shortcut). Safe to run again.
#   irm https://raw.githubusercontent.com/JMS1717/ming-voice/main/install.ps1 | iex
# or double-click Install.cmd in a downloaded copy.
function Install-MingVoice {
    $repo = 'https://github.com/JMS1717/ming-voice.git'
    function Say($t) { Write-Host "`n==> $t" -ForegroundColor Cyan }

    $gpus = @(Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name })
    if (-not ($gpus -match 'Radeon|AMD|NVIDIA')) {
        Write-Warning "No AMD or NVIDIA GPU found ($($gpus -join ', ')). Ming Voice needs a GPU with 16 GB+ VRAM."
        if ((Read-Host 'Continue anyway? (y/N)') -ne 'y') { return }
    }

    # an Ubuntu 24.04 WSL distribution: an existing one, or install it
    $names = @()
    try { $names = @((wsl.exe -l -q) -replace "`0", '' | ForEach-Object { $_.Trim() } | Where-Object { $_ }) } catch {}
    $distro = $null
    foreach ($n in @('Ubuntu-24.04', 'Ubuntu') + $names) {
        if ($names -notcontains $n) { continue }
        $os = (wsl.exe -d $n --exec cat /etc/os-release 2>$null) -join "`n"
        if ($os -match 'VERSION_ID="24\.04"') { $distro = $n; break }
    }
    if (-not $distro) {
        Say 'Installing WSL and Ubuntu 24.04 (Windows may ask for permission, and may need a restart)'
        wsl.exe --install -d Ubuntu-24.04
        Write-Host ''
        Write-Host 'When Ubuntu opens, pick a Linux username and password. If Windows asks you to restart, do that.'
        Write-Host 'Then run this installer again to finish.' -ForegroundColor Yellow
        return
    }
    Say "Using WSL distribution '$distro'"

    # the code goes in the Linux home folder (much faster than /mnt/c), then install.sh does the rest
    $sh = 'set -e; command -v git >/dev/null || { sudo apt-get update -qq; sudo apt-get install -y -qq git; }; ' +
          'if [ -d ~/ming-voice/.git ]; then git -C ~/ming-voice pull --ff-only; else git clone ' + $repo + ' ~/ming-voice; fi; ' +
          'MING_FROM_WINDOWS=1 bash ~/ming-voice/install.sh'
    wsl.exe -d $distro --cd ~ --exec bash -lc $sh  # (--exec: no outer shell re-parsing the command)
    if ($LASTEXITCODE -ne 0) { Write-Host "`nThe installer stopped with an error (see above). Fix it and run this again." -ForegroundColor Red; return }
    # ROCm was just installed: restart the distribution so the new GPU group membership applies
    wsl.exe -d $distro --cd ~ --exec test -f ming-voice/.wsl_restart_needed
    if ($LASTEXITCODE -eq 0) {
        wsl.exe -d $distro --cd ~ --exec rm -f ming-voice/.wsl_restart_needed
        wsl.exe --terminate $distro | Out-Null
        Say 'Restarted WSL so the GPU permissions apply - you can start Ming Voice now'
    }
}
Install-MingVoice
