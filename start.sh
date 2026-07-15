#!/usr/bin/env bash
# 数学建模多Agent系统 一键启动 (bash 版，适用于 git bash / WSL / Linux / macOS)
set -e
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8

echo "============================================================"
echo "  数学建模多Agent系统  -  一键启动"
echo "============================================================"

# 检查环境
command -v python >/dev/null || { echo "[错误] 未找到 python"; exit 1; }
command -v npm >/dev/null    || { echo "[错误] 未找到 npm"; exit 1; }

# [1/4] 后端依赖
echo "[1/4] 检查后端依赖..."
if ! python -c "import fastapi,uvicorn,openai,anthropic,langgraph,xhtml2pdf,multipart,yaml" >/dev/null 2>&1; then
  echo "      首次启动，安装后端依赖（阿里云镜像）..."
  python -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
else
  echo "      后端依赖已就绪"
fi

# [2/4] 前端依赖
echo "[2/4] 检查前端依赖..."
if [ ! -d frontend/node_modules ]; then
  echo "      首次启动，安装前端依赖..."
  (cd frontend && npm install)
else
  echo "      前端依赖已就绪"
fi

# [3/4] .env
echo "[3/4] 检查 .env 配置..."
if [ ! -f .env ]; then
  [ -f .env.example ] && cp .env.example .env
  echo "      已从模板创建 .env，请编辑填入 API key 后重新运行"
  exit 0
fi
echo "      .env 已存在"

# [4/4] 启动
echo "[4/4] 启动服务..."
python -m uvicorn app.server.main:app --port 8000 --log-level info &
BACKEND_PID=$!
(cd frontend && npm run dev) &
FRONTEND_PID=$!

echo ""
echo "  启动完成："
echo "    前端  http://localhost:3000"
echo "    后端  http://localhost:8000/docs"
echo "    按 Ctrl+C 停止"
echo ""
wait
