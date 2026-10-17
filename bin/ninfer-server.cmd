@echo off
rem Thin shim so `ninfer-server` works from cmd/PowerShell like llama-server does.
rem All flags are forwarded verbatim; the tuned defaults (--prefill-chunk 8192,
rem --kv-capacity 262144, MTP K=3, ...) are injected by ninfer-server.ps1.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0ninfer-server.ps1" %*
