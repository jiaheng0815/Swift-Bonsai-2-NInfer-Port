@echo off
setlocal
rem Perplexity sanity check. Verdict: single digit = OK, hundreds = broken.
rem Measured here: Swift 4.9912 / baseline 4.9895 (same corpus, fp8 KV).
set "ROOT=%~dp0.."
set "ARTIFACT=%ROOT%\artifacts\swift-bonsai2-27b-pruned.ninfer"
if not "%~1"=="" set "ARTIFACT=%~1"
"%ROOT%\build\b1\apps\ninfer-perplexity.exe" "%ARTIFACT%" ^
  --text "%ROOT%\eval\ppl_sample.txt" --context 512 --stride 256 --kv-dtype fp8
echo.
pause
