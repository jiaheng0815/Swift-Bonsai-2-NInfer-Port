@echo off
setlocal
rem 终端多轮聊天（先起服务）。用法：
rem   chat.bat                     -> 基线 bonsai2-27b
rem   chat.bat swift-bonsai2-27b   -> Swift 版
chcp 65001 >nul
cd /d "%~dp0"
python chat.py %1
pause
