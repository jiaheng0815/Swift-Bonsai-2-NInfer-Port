@echo off
setlocal
rem 停止 NInfer 服务并确认显存回落（回到 ~1 GB 基线才算干净）
taskkill /IM ninfer-serve.exe /F >nul 2>&1
if errorlevel 1 (
  echo no ninfer-serve.exe was running.
) else (
  echo stopped ninfer-serve.exe
)
timeout /t 2 >nul
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
pause
