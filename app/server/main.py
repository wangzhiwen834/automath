"""FastAPI 后端：REST 接口 + WebSocket 实时推送。

启动: uvicorn app.server.main:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, FileResponse, Response

from app.config import get_settings
from app.storage import get_store
from .runner import TaskRunner
from .schemas import TaskCreate, TaskResume


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 记录主事件循环，供后台线程投递事件
    TaskRunner.get().set_loop(asyncio.get_running_loop())
    yield


app = FastAPI(title="数学建模多Agent系统", version="0.1.0", lifespan=lifespan)

# 允许前端跨域（Next.js 默认 3000）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _store():
    return get_store()


def _runner():
    return TaskRunner.get()


# ====================================================================
# 模型
# ====================================================================
@app.get("/api/models")
def list_models():
    """列出所有可用模型，供前端选择。"""
    return {"models": get_settings().list_models(), "default": get_settings().default_model}


# ====================================================================
# 任务 CRUD
# ====================================================================
@app.get("/api/tasks")
def list_tasks():
    """任务列表（供首页展示）。"""
    return {"tasks": _store().list_tasks()}


@app.post("/api/tasks")
def create_task(body: TaskCreate):
    """创建任务。"""
    task = _store().create_task(
        title=body.title,
        problem_text=body.problem_text,
        mode=body.mode,
        problem_type=body.problem_type,
        agent_models=body.agent_models,
    )
    return {"task_id": task.meta.task_id, "meta": task.meta.model_dump()}


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str):
    """任务详情：meta + state + 是否运行中。"""
    if not _store().exists(task_id):
        raise HTTPException(404, "任务不存在")
    task = _store().load(task_id)
    return {
        "meta": task.meta.model_dump(),
        "state": task.state.model_dump(),
        "running": _runner().is_running(task_id),
    }


@app.delete("/api/tasks/{task_id}")
def delete_task(task_id: str):
    """删除任务（删目录）。"""
    import shutil
    path = _store().task_path(task_id)
    if not path.exists():
        raise HTTPException(404, "任务不存在")
    shutil.rmtree(path)
    return {"ok": True}


# ====================================================================
# 产物 & 日志
# ====================================================================
ARTIFACT_MAP = {
    "analyst": "artifacts/analysis.md",
    "modeler": "artifacts/model.md",
    "solver": "artifacts/solution/output.txt",
    "writer": "artifacts/paper.md",
    "reviewer": "artifacts/review.md",
}


@app.get("/api/tasks/{task_id}/artifacts/{agent}")
def get_artifact(task_id: str, agent: str):
    """获取某 Agent 的产物文本。"""
    if agent not in ARTIFACT_MAP:
        raise HTTPException(400, f"未知 agent: {agent}")
    if not _store().exists(task_id):
        raise HTTPException(404, "任务不存在")
    content = _store().read_artifact(task_id, ARTIFACT_MAP[agent])
    return PlainTextResponse(content, media_type="text/plain; charset=utf-8")


@app.get("/api/tasks/{task_id}/solution-output")
def get_solution_output(task_id: str):
    """获取求解执行输出。"""
    if not _store().exists(task_id):
        raise HTTPException(404, "任务不存在")
    out = _store().read_artifact(task_id, "artifacts/solution/output.txt")
    status_path = _store().task_path(task_id) / "artifacts" / "solution" / "status.json"
    status = None
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
    return {"output": out, "status": status}


@app.get("/api/tasks/{task_id}/logs/{agent}")
def get_logs(task_id: str, agent: str):
    """获取某 Agent 的历史日志事件（用于回放）。"""
    if not _store().exists(task_id):
        raise HTTPException(404, "任务不存在")
    return {"events": _store().read_log(task_id, agent)}


@app.get("/api/tasks/{task_id}/problem")
def get_problem(task_id: str):
    if not _store().exists(task_id):
        raise HTTPException(404, "任务不存在")
    return PlainTextResponse(_store().read_problem(task_id),
                             media_type="text/plain; charset=utf-8")


# ====================================================================
# 数据集上传 & 图表
# ====================================================================
@app.post("/api/tasks/{task_id}/upload")
async def upload_data(task_id: str, files: list[UploadFile] = File(...)):
    """上传数据集文件，存到任务的 data/ 目录。"""
    if not _store().exists(task_id):
        raise HTTPException(404, "任务不存在")
    data_dir = _store().data_dir(task_id)
    saved = []
    for f in files:
        # 防路径穿越
        safe_name = f.filename.replace("/", "_").replace("\\", "_")
        dest = data_dir / safe_name
        content = await f.read()
        dest.write_bytes(content)
        saved.append(safe_name)
    return {"saved": saved, "all_files": _store().list_data_files(task_id)}


@app.get("/api/tasks/{task_id}/data-files")
def list_data_files(task_id: str):
    if not _store().exists(task_id):
        raise HTTPException(404, "任务不存在")
    return {"files": _store().list_data_files(task_id)}


@app.get("/api/tasks/{task_id}/figures")
def list_figures(task_id: str):
    """列出求解师生成的图表文件名。"""
    if not _store().exists(task_id):
        raise HTTPException(404, "任务不存在")
    return {"figures": _store().list_figures(task_id)}


@app.get("/api/tasks/{task_id}/figures/{name}")
def get_figure(task_id: str, name: str):
    """获取单张图表图片。"""
    if not _store().exists(task_id):
        raise HTTPException(404, "任务不存在")
    path = _store().figures_dir(task_id) / name
    if not path.exists():
        raise HTTPException(404, "图片不存在")
    return FileResponse(path)


# ====================================================================
# 论文 PDF 导出
# ====================================================================
@app.get("/api/tasks/{task_id}/paper.html")
def paper_html(task_id: str):
    """论文 HTML（图表以 base64 内嵌），供在线预览或浏览器打印为 PDF。"""
    html = _render_paper_html(task_id)
    return Response(html, media_type="text/html; charset=utf-8")


@app.get("/api/tasks/{task_id}/paper.pdf")
def paper_pdf(task_id: str):
    """论文导出为 PDF（服务端渲染，图表内嵌）。"""
    if not _store().exists(task_id):
        raise HTTPException(404, "任务不存在")
    html = _render_paper_html(task_id)
    try:
        from xhtml2pdf import pisa
        import io
        buf = io.BytesIO()
        pisa_status = pisa.CreatePDF(io.StringIO(html), dest=buf, encoding="utf-8")
        if pisa_status.err:
            raise HTTPException(500, f"PDF 渲染出错: {pisa_status.err}")
        return Response(buf.getvalue(), media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="paper_{task_id}.pdf"'})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"PDF 导出失败（{e}）；可改用 /paper.html 在浏览器打印为 PDF")


def _render_paper_html(task_id: str) -> str:
    """把 paper.md 渲染成 HTML，图表转 base64 内嵌，公式用 MathJax。"""
    import base64
    import markdown as md
    if not _store().exists(task_id):
        raise HTTPException(404, "任务不存在")
    paper = _store().read_artifact(task_id, "artifacts/paper.md")
    if not paper:
        raise HTTPException(404, "论文尚未生成")

    # 把 figures/xxx.png 引用替换成 base64 data URI
    import re as _re
    fig_dir = _store().figures_dir(task_id)

    def _repl(m):
        fname = m.group(1).split("/")[-1]
        fp = fig_dir / fname
        if fp.exists():
            b64 = base64.b64encode(fp.read_bytes()).decode()
            return f'src="data:image/png;base64,{b64}"'
        return m.group(0)

    html_body = md.markdown(paper, extensions=["tables", "fenced_code"])
    # 匹配 HTML img 的 src="figures/xxx.png"（python-markdown 已把 ![]() 转成 <img>）
    html_body = _re.sub(r'src="(figures/[^"]+)"', _repl, html_body)

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>论文</title>
<script>MathJax = {{tex: {{inlineMath: [['$','$'],['\\\\(','\\\\)']],
displayMath: [['$$','$$'],['\\\\[','\\\\]']]}}}};</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
body {{ font-family: 'SimSun','Times New Roman',serif; line-height:1.8; max-width:780px; margin:40px auto; padding:0 20px; }}
h1 {{ font-size:20px; text-align:center; }}
h2 {{ font-size:16px; }} h3 {{ font-size:14px; }}
img {{ display:block; margin:12px auto; }}
table {{ border-collapse:collapse; width:100%; }}
td,th {{ border:1px solid #999; padding:6px; }}
code,pre {{ background:#f5f5f5; font-family:Consolas,monospace; }}
pre {{ padding:10px; overflow-x:auto; }}
</style></head><body>{html_body}</body></html>"""



# ====================================================================
# 运行 & 恢复
# ====================================================================
@app.post("/api/tasks/{task_id}/run")
def run_task(task_id: str):
    """启动任务（后台线程跑编排器）。"""
    if not _store().exists(task_id):
        raise HTTPException(404, "任务不存在")
    if _runner().is_running(task_id):
        raise HTTPException(409, "任务已在运行")
    ok = _runner().start(task_id)
    return {"ok": ok, "running": _runner().is_running(task_id)}


@app.post("/api/tasks/{task_id}/resume")
def resume_task(task_id: str, body: TaskResume):
    """从检查点恢复（approve 继续 / modify 带反馈重做）。"""
    if not _store().exists(task_id):
        raise HTTPException(404, "任务不存在")
    if _runner().is_running(task_id):
        raise HTTPException(409, "任务已在运行")
    ok = _runner().resume(task_id, body.decision, body.feedback)
    return {"ok": ok, "running": _runner().is_running(task_id)}


# ====================================================================
# WebSocket：实时事件流
# ====================================================================
@app.websocket("/ws/tasks/{task_id}")
async def task_ws(websocket: WebSocket, task_id: str):
    """订阅任务事件流。连接后先推送各 Agent 历史日志（回放），再实时推送。"""
    await websocket.accept()
    if not _store().exists(task_id):
        await websocket.send_json({"type": "error", "msg": "任务不存在"})
        await websocket.close()
        return

    runner = _runner()
    queue = runner.subscribe(task_id)

    try:
        # 1) 回放已有日志（每个 agent 的历史事件）
        for agent in ["analyst", "modeler", "solver", "writer", "reviewer"]:
            for ev in _store().read_log(task_id, agent):
                await websocket.send_json({"type": "replay", "agent": agent, **ev})

        # 2) 推送当前状态快照
        task = _store().load(task_id)
        await websocket.send_json({"type": "snapshot",
                                    "state": task.state.model_dump()})

        # 3) 实时消费队列（30s 无事件发心跳保活，不断连）
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30)
            except asyncio.TimeoutError:
                try:
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    break
                continue
            await websocket.send_json(event)
            if event.get("type") in ("completed", "failed"):
                # 末尾再推一次最终状态
                await websocket.send_json({"type": "final",
                                            "state": _store().load(task_id).state.model_dump()})
    except WebSocketDisconnect:
        pass
    finally:
        runner.unsubscribe(task_id, queue)
