@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=gbk

echo ============================================================
echo    数学建模多Agent系统 - 一键启动
echo ============================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
  echo [错误] 未找到 python，请先安装 Python 3.10+
  pause
  exit /b 1
)
where npm >nul 2>&1
if errorlevel 1 (
  echo [错误] 未找到 npm，请先安装 Node.js 18+
  pause
  exit /b 1
)

echo [0/4] 清理可能占用的 8000/3000 端口...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000" ^| findstr LISTENING') do taskkill /F /PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":3000" ^| findstr LISTENING') do taskkill /F /PID %%a >nul 2>&1

echo [1/4] 检查后端依赖...
python -c "import fastapi,uvicorn,openai,anthropic,langgraph,xhtml2pdf,multipart,yaml" >nul 2>&1
if errorlevel 1 (
  echo       首次启动，正在安装后端依赖（阿里云镜像，约1-3分钟）...
  python -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
  if errorlevel 1 (
    echo [错误] 后端依赖安装失败
    pause
    exit /b 1
  )
) else (
  echo       后端依赖已就绪
)

echo [2/4] 检查前端依赖...
if not exist "frontend\node_modules" (
  echo       首次启动，正在安装前端依赖（约1-2分钟）...
  pushd frontend
  call npm install
  if errorlevel 1 (
    echo [错误] 前端依赖安装失败
    popd
    pause
    exit /b 1
  )
  popd
) else (
  echo       前端依赖已就绪
)

echo [3/4] 检查 .env 配置...
if not exist ".env" (
  if exist ".env.example" copy ".env.example" ".env" >nul
  echo       已从模板创建 .env
  echo       请编辑填入至少一个模型的 API key（如 ZHIPU_API_KEY）后保存，再重新运行本脚本
  notepad ".env"
  pause
  exit /b 0
)
echo       .env 已存在

echo [4/4] 启动后端与前端服务...
start "后端 Backend :8000" cmd /k "cd /d %~dp0 && set PYTHONIOENCODING=gbk && python -m uvicorn app.server.main:app --port 8000 --log-level info"
start "前端 Frontend :3000" cmd /k "cd /d %~dp0\frontend && npm run dev"

echo.
echo       等待服务就绪...
timeout /t 8 /nobreak >nul
start http://localhost:3000

echo.
echo ============================================================
echo   启动完成！浏览器已打开  http://localhost:3000
echo   后端 API 文档            http://localhost:8000/docs
echo   停止服务：关闭弹出的两个窗口，或运行 stop.bat
echo ============================================================
echo.
pause
