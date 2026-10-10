@echo off
setlocal enabledelayedexpansion
rem Rebuild the engine (only needed after changing C++ sources).
rem Toolchain: VS Build Tools 2022 (MSVC 14.51) + CUDA 13.3 + Ninja.
call "F:\VSBuildTools2022\VC\Auxiliary\Build\vcvars64.bat" >nul
set "VSLANG=1033"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
set "PATH=%PATH%;E:\cmake\bin;E:\python\Scripts"

cmake -S D:\ai-ninfer\src\cranebw -B D:\ai-ninfer\build\b1 -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CUDA_ARCHITECTURES=120a ^
  -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/nvcc.exe" ^
  -DNINFER_BUILD_APPS=ON ^
  -DNINFER_BUILD_BENCHMARKS=OFF
if errorlevel 1 ( echo CONFIGURE FAILED & pause & exit /b 1 )

set "PREV="
for /L %%R in (1,1,40) do (
  ninja -C D:\ai-ninfer\build\b1 -j 8 -k 0
  set "RC=!ERRORLEVEL!"
  echo ROUND %%R RC=!RC!
  if "!RC!"=="0" ( echo BUILD OK & goto :done )
  if "!RC!"=="!PREV!" ( echo STALLED at RC=!RC! & goto :done )
  set "PREV=!RC!"
)
:done
pause
