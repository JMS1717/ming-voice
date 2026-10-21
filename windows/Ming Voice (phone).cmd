@echo off
rem Like "Ming Voice.cmd", plus phone access: https on port 8443 of this PC for devices on the same network.
rem Press "Phone" in the app for the QR code.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_ming_voice.ps1" -Phone
