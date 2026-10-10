@echo off
setlocal
rem Stop the NInfer server and confirm VRAM is released (back to ~1 GB baseline).
taskkill /IM ninfer-serve.exe /F >nul 2>&1
if errorlevel 1 (
  echo no ninfer-serve.exe was running.
) else (
  echo stopped ninfer-serve.exe
)
timeout /t 2 >nul
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
pause
