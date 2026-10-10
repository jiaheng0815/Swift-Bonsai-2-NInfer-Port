@echo off
setlocal
rem Terminal chat (start the server first). Usage:
rem   chat.bat                     -> swift-bonsai2-27b
rem   chat.bat bonsai2-27b         -> baseline
chcp 65001 >nul
cd /d "%~dp0"
python chat.py %1
pause
