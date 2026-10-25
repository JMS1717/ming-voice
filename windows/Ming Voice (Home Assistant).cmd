@echo off
rem Ming Voice for Home Assistant: the server plus the Wyoming bridge on port 10300 of this PC (no browser).
rem In Home Assistant: Settings - Devices and services - Add integration - Wyoming Protocol, this PC's address,
rem port 10300. Then choose Ming for speech-to-text, text-to-speech and/or the conversation agent of a voice
rem assistant (Settings - Voice assistants).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_ming_voice.ps1" -Wyoming -NoBrowser
