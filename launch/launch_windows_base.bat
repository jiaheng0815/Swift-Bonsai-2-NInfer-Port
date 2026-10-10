@echo off
setlocal
rem ============================================================
rem  Baseline Ternary Bonsai 2 27B on NInfer -- Windows / RTX 5060 Ti 16GB / sm_120a
rem
rem  Tuned defaults (measured, see docs/TUNING.md):
rem    262144 ctx / nvfp4 KV / prefill-chunk 8192 / MTP K=3 / vision on
rem  Client must send model id: bonsai2-27b
rem  One engine per GPU: run stop_server.bat before starting another.
rem ============================================================
set "ROOT=%~dp0.."

set "PREFILL=8192"
set "DRAFT=3"
set "CTX=262144"
set "KVTYPE=nvfp4"
set "MODELID=bonsai2-27b"
set "PORT=8080"

set "ARTIFACT=%ROOT%\artifacts\bonsai2-27b-pruned.ninfer"
set "EXE=%ROOT%\build\b1\apps\ninfer-serve.exe"

if not exist "%ARTIFACT%" (
  echo [ERROR] model not found: %ARTIFACT%
  echo         See docs\CONVERSION_GUIDE.md -- build it first.
  pause & exit /b 1
)
if not exist "%EXE%" (
  echo [ERROR] engine not built: %EXE%
  echo         Run engine\build_windows.bat first.
  pause & exit /b 1
)

rem one engine per GPU
tasklist /FI "IMAGENAME eq ninfer-serve.exe" 2>nul | find /i "ninfer-serve.exe" >nul
if not errorlevel 1 (
  echo [ERROR] ninfer-serve.exe is already running. Run stop_server.bat first.
  pause & exit /b 1
)

echo loading model ... (first run ~10s)
echo endpoint: http://127.0.0.1:%PORT%/v1/models
echo after start, check the "capacity" line: free should be ^>= 400 MiB
echo close this window to stop the server (or run stop_server.bat)
echo.

"%EXE%" "%ARTIFACT%" ^
  --host 127.0.0.1 --port %PORT% ^
  --model-id %MODELID% ^
  --max-context %CTX% --kv-capacity %CTX% --kv-dtype %KVTYPE% ^
  --prefill-chunk %PREFILL% ^
  --max-concurrency 1 ^
  --spec mtp --draft-tokens %DRAFT% ^
  --lm-head-draft ^
  --device-state-slots 1 --host-state-slots 0 --host-kv-mib 0 ^
  --max-private-continuations 1 --max-shared-prefixes 0 ^
  --vision ^
  --cors

echo.
echo server exited.
pause
