"""编排器：把 5 个 Agent 串成状态机，实现三种模式 + 审查回退循环。

不用 LangGraph：直接基于 TaskStore 的文件状态机，暂停/恢复/回退逻辑透明可控。

流程:
  analyst → modeler → solver → writer → reviewer
                ↑                          │
                └──── 审查不通过回退 ────────┘

三种模式的检查点（"在某 Agent 完成后暂停"）:
  A 自动:        无检查点，一路到底
  B 人在回路:    analyst/modeler/solver/writer 后都暂停
  C 混合:        modeler / writer 后暂停（关键决策点）

暂停时任务状态 PAUSED + waiting_for_human=True，由 resume() 唤醒。
"""
from __future__ import annotations

from typing import Any, Callable
import json

from app.agents import AGENTS
from app.agents.base import RunContext
from app.config import get_settings
from app.storage import (
    AGENT_ORDER,
    AgentName,
    AgentStatus,
    RunMode,
    Task,
    TaskStatus,
    get_store,
)

# 各模式的检查点：这些 Agent 完成后暂停（reviewer 终态单独处理，不在此列）
CHECKPOINTS: dict[RunMode, set[AgentName]] = {
    RunMode.AUTO: set(),
    RunMode.INTERACTIVE: {AgentName.ANALYST, AgentName.MODELER, AgentName.SOLVER, AgentName.WRITER},
    RunMode.HYBRID: {AgentName.MODELER, AgentName.WRITER},
}

# 事件回调：每发一个 dict 事件（供 WebSocket 推送）
EventSink = Callable[[dict[str, Any]], None]


class Orchestrator:
    def __init__(self, task_id: str, event_sink: EventSink | None = None) -> None:
        self.task_id = task_id
        self.store = get_store()
        self.event_sink: EventSink = event_sink or (lambda e: None)

    def _emit(self, event: dict[str, Any]) -> None:
        self.event_sink(event)

    # ----------------------------------------------------------
    # 上下文构建：从磁盘读当前所有已完成产物（每次跑 Agent 前重建，拿最新）
    # ----------------------------------------------------------
    def _build_ctx(self, task: Task, review_feedback: str | None = None) -> RunContext:
        ctx = RunContext(task=task, store=self.store)
        for agent in AGENT_ORDER:
            rec = task.state.agents.get(agent.value)
            if rec and rec.artifact_path:
                ctx.artifacts[agent.value] = self.store.read_artifact(
                    self.task_id, rec.artifact_path
                )
        # 求解输出 + 执行状态
        sol_dir = self.store.task_path(self.task_id) / "artifacts" / "solution"
        out_path = sol_dir / "output.txt"
        if out_path.exists():
            ctx.solution_stdout = out_path.read_text(encoding="utf-8")
        status_path = sol_dir / "status.json"
        if status_path.exists():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
                ctx.solution_executed = status.get("executed")
                ctx.solution_error = status.get("error")
            except json.JSONDecodeError:
                pass
        if review_feedback:
            ctx.review_feedback = review_feedback
        # 图表与上传数据文件
        ctx.figures = self.store.list_figures(self.task_id)
        ctx.data_files = self.store.list_data_files(self.task_id)
        return ctx

    # ----------------------------------------------------------
    # 主入口：从当前状态跑到暂停/完成
    # ----------------------------------------------------------
    def run(self) -> Task:
        task = self.store.load(self.task_id)

        # 已在等人工 → 调 resume，不是 run
        if task.state.waiting_for_human:
            return task
        if task.state.status == TaskStatus.COMPLETED:
            return task

        task.state.status = TaskStatus.RUNNING
        task.state.waiting_for_human = False
        # 取出挂起的人工反馈（resume('modify') 时设置），只给下一个 Agent
        next_feedback = task.state.human_feedback
        if next_feedback:
            task.state.human_feedback = None
        self.store.save(task)
        self._emit({"type": "run_started", "task_id": self.task_id,
                    "mode": task.meta.mode.value})

        max_retries = get_settings().reviewer_config.get("max_retries", 2)

        # 起始位置
        current = task.state.current_agent or AGENT_ORDER[0]
        try:
            idx = AGENT_ORDER.index(current)
        except ValueError:
            idx = 0

        while idx < len(AGENT_ORDER):
            agent_name = AGENT_ORDER[idx]
            # 消融：可跳过总结师(写作直接读求解原始输出，编造风险更高)
            if (agent_name == AgentName.SUMMARIZER
                    and get_settings().pipeline_config.get("skip_summarizer", False)):
                rec = task.state.get_agent(agent_name)
                rec.status = AgentStatus.SKIPPED
                self.store.add_history(task, "skipped", agent=agent_name.value, detail="消融跳过")
                self.store.save(task)
                self._emit({"type": "agent_skipped", "agent": agent_name.value})
                idx += 1
                task.state.current_agent = (
                    AGENT_ORDER[idx] if idx < len(AGENT_ORDER) else None
                )
                continue
            fb = next_feedback
            next_feedback = None  # 只消费一次

            ctx = self._build_ctx(task, fb)
            agent_cls = AGENTS[agent_name.value]
            agent = agent_cls(task, self.store)

            self._emit({"type": "agent_start", "agent": agent_name.value,
                        "model": task.meta.agent_models.get(agent_name.value)})

            def sink(chunk: str, _an: AgentName = agent_name) -> None:
                self._emit({"type": "delta", "agent": _an.value, "text": chunk})

            result = agent.run(ctx, stream_callback=sink)
            task = self.store.load(self.task_id)  # agent 已落盘，重载拿最新

            self._emit({
                "type": "agent_done", "agent": agent_name.value,
                "success": result.success, "summary": result.summary,
            })

            if not result.success:
                task.state.status = TaskStatus.FAILED
                self.store.save(task)
                self._emit({"type": "failed", "agent": agent_name.value,
                            "error": result.error})
                return task

            # ---------- 审查师：决定通过/回退 ----------
            if agent_name == AgentName.REVIEWER:
                passed = result.extra.get("passed")
                score = result.extra.get("score")
                rollback_to = result.extra.get("rollback_to")
                review = result.extra.get("review", {})
                self._emit({"type": "review", "score": score, "passed": passed,
                            "rollback_to": rollback_to})

                if passed:
                    return self._finish(task, score=score, note=None)

                # 不通过：不再自动回退重做，直接结束并输出当前结果。
                # 审查意见/issues/建议回退的 agent 仍记录在 review.md 供人工参考。
                note = "审查未通过"
                if rollback_to:
                    note += f"，建议回退到 {rollback_to}（如需改进请手动重做）"
                return self._finish(task, score=score, note=note)

            # ---------- 普通 Agent：检查点判断 ----------
            checkpoints = CHECKPOINTS.get(task.meta.mode, set())
            if agent_name in checkpoints:
                next_idx = idx + 1
                task.state.current_agent = (
                    AGENT_ORDER[next_idx] if next_idx < len(AGENT_ORDER) else None
                )
                task.state.status = TaskStatus.PAUSED
                task.state.waiting_for_human = True
                self.store.save(task)
                self._emit({
                    "type": "paused", "after": agent_name.value,
                    "next": task.state.current_agent.value if task.state.current_agent else None,
                })
                return task

            idx += 1
            task.state.current_agent = (
                AGENT_ORDER[idx] if idx < len(AGENT_ORDER) else None
            )

        return self._finish(task, score=None, note=None)

    def _finish(self, task: Task, score: int | None, note: str | None) -> Task:
        task.state.status = TaskStatus.COMPLETED
        task.state.current_agent = None
        task.state.waiting_for_human = False
        self.store.save(task)
        self._emit({"type": "completed", "score": score, "note": note})
        return task

    # ----------------------------------------------------------
    # 恢复：从检查点暂停状态继续
    #   decision: "approve" 继续 / "modify" 带反馈重做刚完成的 Agent
    # ----------------------------------------------------------
    def resume(self, decision: str, feedback: str | None = None) -> Task:
        task = self.store.load(self.task_id)
        if not task.state.waiting_for_human:
            self._emit({"type": "warning", "msg": "任务未处于暂停状态"})
            return task

        task.state.waiting_for_human = False
        task.state.human_decision = decision
        current = task.state.current_agent

        if decision == "approve":
            task.state.human_feedback = None
            # current_agent 已是下一个待跑 Agent，直接 run

        elif decision in ("modify", "reject"):
            # 回退到刚完成的 Agent（current 的前一个）重做
            if current is None:
                task.state.human_feedback = feedback
            else:
                cur_idx = AGENT_ORDER.index(current)
                pred_idx = max(0, cur_idx - 1)
                task.state.current_agent = AGENT_ORDER[pred_idx]
                task.state.human_feedback = feedback
            self._emit({"type": "modify", "redo": task.state.current_agent.value})
        else:
            raise ValueError(f"未知决策: {decision}（应为 approve/modify/reject）")

        self.store.save(task)
        return self.run()
