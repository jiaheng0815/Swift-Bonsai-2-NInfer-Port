@echo off
setlocal
rem ============================================================
rem  PPL 正确性验收（怀疑权重跑偏时用）
rem  判据：个位数 ~ 十几 = 正确；几百 = 错
rem  本机实测：Swift 4.9912 / 基线 4.9895（同一语料、fp8 KV）
rem ============================================================
set "ROOT=%~dp0.."
set "ARTIFACT=%ROOT%\artifacts\swift-bonsai2-27b-pruned.ninfer"
if not "%~1"=="" set "ARTIFACT=%~1"

"%ROOT%\build\b1\apps\ninfer-perplexity.exe" "%ARTIFACT%" ^
  --text "%ROOT%\eval\ppl_sample.txt" --context 512 --stride 256 --kv-dtype fp8
echo.
pause
