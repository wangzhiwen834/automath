@echo off
cd /d "%~dp0"
echo 正在停止后端(:8000)与前端(:3000)服务...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000" ^| findstr LISTENING') do (
  taskkill /F /PID %%a >nul 2>&1 && echo   已停止 PID %%a (端口8000)
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":3000" ^| findstr LISTENING') do (
  taskkill /F /PID %%a >nul 2>&1 && echo   已停止 PID %%a (端口3000)
)
echo 完成。
timeout /t 2 /nobreak >nul
