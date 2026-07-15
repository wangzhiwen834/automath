"""任务存储：纯文件操作，无数据库。

目录结构（每个任务一个文件夹）:
  <workspace>/tasks/<task_id>/
    meta.json          任务元信息
    state.json         编排状态
    input/problem.txt  原始题目
    artifacts/         各 Agent 产物
      analysis.md
      model.md
      solution/
        solve.py
        output.txt
      paper.md
      figures/
    logs/              每个 Agent 的流式日志(JSONL)

所有写操作原子化（先写临时文件再替换），避免崩溃损坏 JSON。
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from app.config import get_settings
from .models import (
    AgentName,
    AGENT_ORDER,
    HistoryEvent,
    ProblemType,
    RunMode,
    Task,
    TaskMeta,
    TaskState,
    TaskStatus,
)

# 各 Agent 的产物文件名
ARTIFACT_NAMES = {
    AgentName.ANALYST: "analysis.md",
    AgentName.MODELER: "model.md",
    AgentName.SOLVER: "solution",     # 目录，内含 solve.py + output
    AgentName.WRITER: "paper.md",
    AgentName.REVIEWER: "review.md",
}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _new_task_id() -> str:
    """时间戳 + 短 uuid，可排序且唯一。"""
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


class TaskStore:
    """任务文件存储。"""

    def __init__(self, workspace_dir: Path | None = None) -> None:
        self.workspace: Path = workspace_dir or get_settings().workspace_dir
        self.tasks_dir: Path = self.workspace / "tasks"
        self.tasks_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # 路径辅助
    # ----------------------------------------------------------
    def task_path(self, task_id: str) -> Path:
        return self.tasks_dir / task_id

    def _meta_path(self, task_id: str) -> Path:
        return self.task_path(task_id) / "meta.json"

    def _state_path(self, task_id: str) -> Path:
        return self.task_path(task_id) / "state.json"

    def _artifacts_dir(self, task_id: str) -> Path:
        d = self.task_path(task_id) / "artifacts"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _logs_dir(self, task_id: str) -> Path:
        d = self.task_path(task_id) / "logs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _solution_dir(self, task_id: str) -> Path:
        d = self._artifacts_dir(task_id) / "solution"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def figures_dir(self, task_id: str) -> Path:
        """图表输出目录（求解师生成的图存在这里）。"""
        d = self._artifacts_dir(task_id) / "figures"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def data_dir(self, task_id: str) -> Path:
        """上传数据集目录。"""
        d = self.task_path(task_id) / "data"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def list_figures(self, task_id: str) -> list[str]:
        d = self.figures_dir(task_id)
        return sorted(f.name for f in d.iterdir() if f.is_file())

    def list_data_files(self, task_id: str) -> list[str]:
        d = self.data_dir(task_id)
        return sorted(f.name for f in d.iterdir() if f.is_file())

    # ----------------------------------------------------------
    # 原子写
    # ----------------------------------------------------------
    @staticmethod
    def _atomic_write_json(path: Path, data: dict) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    @staticmethod
    def _atomic_write_text(path: Path, text: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)

    # ----------------------------------------------------------
    # 创建任务
    # ----------------------------------------------------------
    def create_task(
        self,
        title: str,
        problem_text: str,
        mode: RunMode | str = RunMode.AUTO,
        problem_type: ProblemType | str = ProblemType.UNKNOWN,
        agent_models: dict[str, str] | None = None,
    ) -> Task:
        """创建任务：建目录、写题目、初始化 meta/state。"""
        task_id = _new_task_id()
        task_dir = self.task_path(task_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "input").mkdir(exist_ok=True)
        self._artifacts_dir(task_id)
        self._logs_dir(task_id)
        self.figures_dir(task_id)
        self.data_dir(task_id)

        # 写题目
        problem_path = task_dir / "input" / "problem.txt"
        self._atomic_write_text(problem_path, problem_text)

        # 合并 Agent 模型配置：任务级覆盖全局默认
        settings = get_settings()
        merged_models = dict(settings.agent_models)
        if agent_models:
            merged_models.update(agent_models)

        now = _now()
        meta = TaskMeta(
            task_id=task_id,
            title=title,
            problem_type=ProblemType(problem_type) if isinstance(problem_type, str) else problem_type,
            mode=RunMode(mode) if isinstance(mode, str) else mode,
            problem_path="input/problem.txt",
            agent_models=merged_models,
            created_at=now,
            updated_at=now,
        )

        # 初始化各 Agent 记录
        state = TaskState(status=TaskStatus.CREATED)
        for agent in AGENT_ORDER:
            rec = state.get_agent(agent)
            rec.model = merged_models.get(agent.value)
        state.history.append(HistoryEvent(
            timestamp=now, action="created", detail=f"任务创建，模式={meta.mode.value}"
        ))

        task = Task(meta=meta, state=state, dir=str(task_dir))
        self.save(task)
        return task

    # ----------------------------------------------------------
    # 读写
    # ----------------------------------------------------------
    def save(self, task: Task) -> None:
        """持久化 meta + state。"""
        task.meta.updated_at = _now()
        self._atomic_write_json(self._meta_path(task.meta.task_id), task.meta.model_dump())
        self._atomic_write_json(self._state_path(task.meta.task_id), task.state.model_dump())

    def load(self, task_id: str) -> Task:
        """从磁盘加载任务（用于恢复）。"""
        with open(self._meta_path(task_id), "r", encoding="utf-8") as f:
            meta = TaskMeta(**json.load(f))
        with open(self._state_path(task_id), "r", encoding="utf-8") as f:
            state = TaskState(**json.load(f))
        return Task(meta=meta, state=state, dir=str(self.task_path(task_id)))

    def exists(self, task_id: str) -> bool:
        return self._meta_path(task_id).exists()

    # ----------------------------------------------------------
    # 列表（供前端）
    # ----------------------------------------------------------
    def list_tasks(self) -> list[dict[str, Any]]:
        """列出所有任务摘要，按创建时间倒序。"""
        out: list[dict[str, Any]] = []
        for d in self.tasks_dir.iterdir():
            if not d.is_dir():
                continue
            meta_p = d / "meta.json"
            state_p = d / "state.json"
            if not meta_p.exists():
                continue
            try:
                with open(meta_p, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                status = "unknown"
                current = None
                if state_p.exists():
                    with open(state_p, "r", encoding="utf-8") as f:
                        state = json.load(f)
                    status = state.get("status", "unknown")
                    current = state.get("current_agent")
                out.append({
                    "task_id": meta["task_id"],
                    "title": meta["title"],
                    "mode": meta["mode"],
                    "problem_type": meta["problem_type"],
                    "status": status,
                    "current_agent": current,
                    "created_at": meta["created_at"],
                    "updated_at": meta.get("updated_at"),
                })
            except (json.JSONDecodeError, KeyError):
                continue
        out.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return out

    # ----------------------------------------------------------
    # 产物读写
    # ----------------------------------------------------------
    def write_artifact(self, task_id: str, agent: AgentName, content: str) -> str:
        """写 Agent 产物，返回相对任务目录的路径。"""
        name = ARTIFACT_NAMES[agent]
        if agent == AgentName.SOLVER:
            # solver 产物是目录，content 写到 solve.py
            sol_dir = self._solution_dir(task_id)
            path = sol_dir / "solve.py"
            self._atomic_write_text(path, content)
            return "artifacts/solution/solve.py"
        else:
            path = self._artifacts_dir(task_id) / name
            self._atomic_write_text(path, content)
            return f"artifacts/{name}"

    def read_artifact(self, task_id: str, rel_path: str) -> str:
        path = self.task_path(task_id) / rel_path
        if not path.exists():
            return ""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def read_problem(self, task_id: str) -> str:
        return self.read_artifact(task_id, "input/problem.txt")

    def solution_dir(self, task_id: str) -> Path:
        return self._solution_dir(task_id)

    def write_solution_output(self, task_id: str, stdout: str, stderr: str = "") -> None:
        sol = self._solution_dir(task_id)
        self._atomic_write_text(sol / "output.txt", stdout)
        if stderr:
            self._atomic_write_text(sol / "stderr.txt", stderr)

    # ----------------------------------------------------------
    # 日志（JSONL，每行一个事件，供前端流式回放）
    # ----------------------------------------------------------
    def append_log(self, task_id: str, agent: str, event: dict[str, Any]) -> str:
        """追加一条日志事件，返回日志文件相对路径。"""
        log_dir = self._logs_dir(task_id)
        path = log_dir / f"{agent}.jsonl"
        line = json.dumps({**event, "ts": _now()}, ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        return f"logs/{agent}.jsonl"

    def read_log(self, task_id: str, agent: str) -> list[dict[str, Any]]:
        path = self.task_path(task_id) / "logs" / f"{agent}.jsonl"
        if not path.exists():
            return []
        events = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return events

    # ----------------------------------------------------------
    # 状态变更辅助
    # ----------------------------------------------------------
    def add_history(self, task: Task, action: str, agent: str | None = None, detail: str | None = None) -> None:
        task.state.history.append(HistoryEvent(
            timestamp=_now(), agent=agent, action=action, detail=detail
        ))

    def update(self, task_id: str, mutator: Callable[[Task], None]) -> Task:
        """加载 → 修改 → 保存的便捷封装。"""
        task = self.load(task_id)
        mutator(task)
        self.save(task)
        return task


_store: TaskStore | None = None


def get_store() -> TaskStore:
    global _store
    if _store is None:
        _store = TaskStore()
    return _store
