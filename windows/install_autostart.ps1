# Start Ming Voice when you sign in to Windows (for an always-on assistant, e.g. with the wake word):
# puts a shortcut to "Ming Voice (phone).cmd" (or "Ming Voice.cmd" with -NoPhone) in your Startup folder.
# Nothing system-wide: it only affects your account, and -Remove (or deleting the shortcut) undoes it.
#
#   powershell -ExecutionPolicy Bypass -File windows\install_autostart.ps1            # install
#   powershell -ExecutionPolicy Bypass -File windows\install_autostart.ps1 -Remove    # remove
param([switch]$Remove, [switch]$NoPhone)
$startup = [Environment]::GetFolderPath('Startup')
$lnk = Join-Path $startup 'Ming Voice.lnk'
if ($Remove) {
    if (Test-Path $lnk) { Remove-Item $lnk; Write-Host "Removed $lnk" } else { Write-Host 'Not installed.' }
    exit 0
}
$target = Join-Path $PSScriptRoot $(if ($NoPhone) { 'Ming Voice.cmd' } else { 'Ming Voice (phone).cmd' })
$s = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk)
$s.TargetPath = $target
$s.WorkingDirectory = $PSScriptRoot
$s.WindowStyle = 7  # minimized
$s.Description = 'Start Ming Voice'
$s.Save()
Write-Host "Installed: $lnk -> $target"
Write-Host 'At sign-in the server starts and the app opens in the browser; press Start once there (browsers need a click before they use the microphone).'
