@echo off
rem Always-on assistant: the server, phone access, and a background client that uses this PC's microphone
rem and speakers (say "Hey Ming, ..."). No browser needed. Stop the client: python windows\ming_client.py --stop
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_ming_voice.ps1" -Phone -Headless
