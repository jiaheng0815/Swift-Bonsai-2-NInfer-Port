@echo off
setlocal
rem Start the server in the background (log to a file, no console window).
rem Stop with stop_server.bat
set "ROOT=%~dp0.."
set "MODELID=%~1"
if "%MODELID%"=="" set "MODELID=swift-bonsai2-27b"
if /i "%MODELID%"=="swift-bonsai2-27b" (
  set "ARTIFACT=%ROOT%\artifacts\swift-bonsai2-27b-pruned.ninfer"
) else (
  set "ARTIFACT=%ROOT%\artifacts\bonsai2-27b-pruned.ninfer"
)
set "LOG=%ROOT%\logs\serve-%MODELID%.log"
"%ROOT%\build\b1\apps\ninfer-serve.exe" "%ARTIFACT%" ^
  --host 127.0.0.1 --port 8080 --model-id %MODELID% ^
  --max-context 262144 --kv-capacity 262144 --kv-dtype nvfp4 ^
  --prefill-chunk 8192 --max-concurrency 1 ^
  --spec mtp --draft-tokens 3 --lm-head-draft ^
  --device-state-slots 1 --host-state-slots 0 --host-kv-mib 0 ^
  --max-private-continuations 1 --max-shared-prefixes 0 ^
  --vision --cors > "%LOG%" 2>&1
