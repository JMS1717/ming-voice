@echo off
rem Double-click to install Ming Voice (sets up WSL Ubuntu, ROCm on AMD, the Python env and the model).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
pause
