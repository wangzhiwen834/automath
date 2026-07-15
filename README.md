# 数学建模多Agent自动化系统

多 Agent 协作的数学建模自动化工具：输入竞赛题目 → 自动完成 **问题分析 → 建模 → 求解(写代码+执行) → 论文写作 → 审查**，产出论文、代码、图表、PDF。

## 特性

- **多模型可切换**：DeepSeek / 通义千问 / GLM / Claude，配置文件注册，运行时切换
- **三种模式**：A 全自动 / B 人在回路 / C 混合（关键节点确认）
- **5 个 Agent**：分析、建模、求解(真实执行代码+图表)、写作、审查(评分+回退)
- **文件化存储**：无数据库，每个任务一个目录，可复盘可恢复
- **前端网页**：实时流式输出、流水线看板、图表展示、PDF 导出、数据集上传

## 一键启动

**Windows**：双击 `start.bat`
**git bash / WSL / Linux / macOS**：`bash start.sh`

首次启动会自动安装前后端依赖；之后直接启动。停止服务运行 `stop.bat`（或关闭弹出的窗口）。

启动后：
- 前端：http://localhost:3000
- 后端 API 文档：http://localhost:8000/docs

## 配置

1. 复制 `.env.example` 为 `.env`，填入至少一个模型的 API key（首次启动若无 .env 会自动创建并打开）
2. `config.yaml` 注册模型、配置各 Agent 默认模型、求解器超时等

## 任务目录结构

```
workspace/tasks/<任务ID>/
  meta.json          任务元信息
  state.json         编排状态（可恢复）
  input/problem.txt  原始题目
  data/              上传的数据集
  artifacts/
    analysis.md      分析师产物
    model.md         建模产物
    solution/        solve.py + output.txt + status.json
    figures/         求解生成的图表
    paper.md         论文
    review.md        审查报告
  logs/              每个 Agent 的流式日志(JSONL)
```

## 技术栈

- 后端：Python + FastAPI + WebSocket
- 前端：Next.js 16 + React + Tailwind v4
- LLM：OpenAI 兼容接口(国产三家) + Anthropic
