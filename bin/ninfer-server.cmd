@echo off
rem Thin shim so `ninfer-server` works from cmd/PowerShell like llama-server does.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0ninfer-server.ps1" %*
