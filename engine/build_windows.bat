@echo off
setlocal enabledelayedexpansion
rem Rebuild the engine (only needed after changing C++ sources).
rem Toolchain: VS Build Tools 2022 (MSVC 14.51) + CUDA 13.3 + Ninja.
rem The defaults below are this machine's paths; override VCVARS / CUDA_PATH /
rem CMAKE_BIN (environment variables) when building elsewhere.
set "ROOT=%~dp0.."
if not defined VCVARS    set "VCVARS=F:\VSBuildTools2022\VC\Auxiliary\Build\vcvars64.bat"
if not defined CUDA_PATH set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
if not defined CMAKE_BIN set "CMAKE_BIN=E:\cmake\bin"

call "%VCVARS%" >nul
set "VSLANG=1033"
set "PATH=%PATH%;%CMAKE_BIN%;E:\python\Scripts"

cmake -S "%ROOT%\src\cranebw" -B "%ROOT%\build\b1" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CUDA_ARCHITECTURES=120a ^
  -DCMAKE_CUDA_COMPILER="%CUDA_PATH%/bin/nvcc.exe" ^
  -DNINFER_BUILD_APPS=ON ^
  -DNINFER_BUILD_BENCHMARKS=OFF
if errorlevel 1 ( echo CONFIGURE FAILED & pause & exit /b 1 )

set "PREV="
for /L %%R in (1,1,40) do (
  ninja -C "%ROOT%\build\b1" -j 8 -k 0
  set "RC=!ERRORLEVEL!"
  echo ROUND %%R RC=!RC!
  if "!RC!"=="0" ( echo BUILD OK & goto :done )
  if "!RC!"=="!PREV!" ( echo STALLED at RC=!RC! & goto :done )
  set "PREV=!RC!"
)
:done
pause
