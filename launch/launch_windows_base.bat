@echo off
setlocal
rem ============================================================
rem  基线 Ternary Bonsai 2 27B（PrismML）on NInfer — 与 Swift 版同参数
rem
rem  想做 A/B 对比（思考 token / PPL）时用本脚本起基线、
rem  launch_windows.bat 起 Swift —— 两次分别跑 bench\swift_bench.py。
rem
rem  客户端 model 字段必须写：bonsai2-27b
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

if not exist "%ARTIFACT%" ( echo [ERROR] artifact not found: %ARTIFACT% & pause & exit /b 1 )
if not exist "%EXE%"      ( echo [ERROR] engine not built: %EXE% & pause & exit /b 1 )

tasklist /FI "IMAGENAME eq ninfer-serve.exe" 2>nul | find /i "ninfer-serve.exe" >nul
if not errorlevel 1 (
  echo [ERROR] ninfer-serve.exe is already running. Run stop_server.bat first.
  pause & exit /b 1
)

echo loading model ...
echo endpoint: http://127.0.0.1:%PORT%/v1/models   (model id: %MODELID%)
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
