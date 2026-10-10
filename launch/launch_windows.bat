@echo off
setlocal
rem ============================================================
rem  Swift Bonsai 2 27B on NInfer  —  Windows / RTX 5060 Ti 16GB / sm_120a
rem
rem  参数为实测最优（见 docs/TUNING.md）：
rem    262,144 上下文 / nvfp4 KV / prefill-chunk 8192 / MTP K=3 / 视觉开启
rem  实测：预填充 654 tok/s（20k 级）、解码 64 tok/s（散文）、启动后余量 500+ MiB
rem
rem  客户端 model 字段必须写：swift-bonsai2-27b
rem  一张卡只跑一个引擎；启动前请先 stop_server.bat
rem ============================================================
set "ROOT=%~dp0.."

set "PREFILL=8192"
set "DRAFT=3"
set "CTX=262144"
set "KVTYPE=nvfp4"
set "MODELID=swift-bonsai2-27b"
set "PORT=8080"

set "ARTIFACT=%ROOT%\artifacts\swift-bonsai2-27b-pruned.ninfer"
set "EXE=%ROOT%\build\b1\apps\ninfer-serve.exe"

if not exist "%ARTIFACT%" (
  echo [ERROR] artifact not found: %ARTIFACT%
  echo         先生成制品：见 docs/CONVERSION_GUIDE.md
  pause & exit /b 1
)
if not exist "%EXE%" (
  echo [ERROR] engine not built: %EXE%
  echo         先编译引擎：engine\build_windows.bat
  pause & exit /b 1
)

rem 一张卡只跑一个引擎
tasklist /FI "IMAGENAME eq ninfer-serve.exe" 2>nul | find /i "ninfer-serve.exe" >nul
if not errorlevel 1 (
  echo [ERROR] ninfer-serve.exe is already running. Run stop_server.bat first.
  pause & exit /b 1
)

echo loading model ...
echo endpoint: http://127.0.0.1:%PORT%/v1/models   (OpenAI compatible)
echo 启动后确认 capacity 行的 free 值 ^>= 400 MiB
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
