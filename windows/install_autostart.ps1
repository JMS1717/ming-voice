# Start Ming Voice when you sign in to Windows: puts a shortcut in your Startup folder to
#   "Ming Voice (always on).cmd"  (default) server + phone access + the background client (this PC's mic and
#                                 speakers, wake word "Hey Ming"); no browser
#   "Ming Voice (phone).cmd"      with -Browser: server + phone access, and the app opens in the browser
# Nothing system-wide: it only affects your account, and -Remove (or deleting the shortcut) undoes it.
#
#   powershell -ExecutionPolicy Bypass -File windows\install_autostart.ps1            # install
#   powershell -ExecutionPolicy Bypass -File windows\install_autostart.ps1 -Remove    # remove
param([switch]$Remove, [switch]$Browser)
$startup = [Environment]::GetFolderPath('Startup')
$lnk = Join-Path $startup 'Ming Voice.lnk'
if ($Remove) {
    if (Test-Path $lnk) { Remove-Item $lnk; Write-Host "Removed $lnk" } else { Write-Host 'Not installed.' }
    exit 0
}
$target = Join-Path $PSScriptRoot $(if ($Browser) { 'Ming Voice (phone).cmd' } else { 'Ming Voice (always on).cmd' })
$s = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk)
$s.TargetPath = $target
$s.WorkingDirectory = $PSScriptRoot
$s.WindowStyle = 7  # minimized
$s.Description = 'Start Ming Voice'
$s.Save()
Write-Host "Installed: $lnk -> $target"
if ($Browser) {
    Write-Host 'At sign-in the server starts and the app opens in the browser; press Start once there (browsers need a click before they use the microphone).'
} else {
    Write-Host 'At sign-in the server starts and Ming listens for "Hey Ming" through this PC''s microphone.'
}
