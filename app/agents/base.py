"""Agent 基类：统一处理 LLM 调用、流式日志、产物落盘、状态更新。

每个子类只需实现:
  - system_prompt: 系统提示词
  - build_user_prompt(ctx): 根据上下文构造用户消息
  - postprocess(ctx, text): 对 LLM 输出做后处理（默认直接落盘）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from app.llm import get_llm
from app.llm.provider import Message
from app.storage import AgentName, AgentStatus, Task, TaskStore, get_store


@dataclass
class RunContext:
    """单次 Agent 运行的上下文：任务 + 存储 + 已有产物。"""
    task: Task
    store: TaskStore
    # 各 Agent 已完成的产物文本（key = agent 名）
    artifacts: dict[str, str] = field(default_factory=dict)
    # solver 的执行输出（供 writer/reviewer 引用）
    solution_stdout: str | None = None
    solution_stderr: str | None = None
    # solver 是否真正执行成功（审查师据此硬否决）
    solution_executed: bool | None = None
    solution_error: str | None = None
    # 审查反馈（回退重做时携带）
    review_feedback: str | None = None
    # 求解师生成的图表文件名（供写作师嵌入论文）
    figures: list[str] = field(default_factory=list)
    # 用户上传的数据文件名（供求解师读取）
    data_files: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    agent: AgentName
    success: bool
    artifact_path: str | None = None
    summary: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


# 流式回调类型：每收到一段文本就调用，用于 WebSocket 实时推送
StreamCallback = Callable[[str], None]


class BaseAgent:
    name: AgentName

    def __init__(self, task: Task, store: TaskStore | None = None) -> None:
        self.task = task
        self.store = store or get_store()
        model_key = task.meta.agent_models.get(self.name.value)
        self.llm = get_llm(model_key)

    # ----------------------------------------------------------
    # 子类实现
    # ----------------------------------------------------------
    @property
    def system_prompt(self) -> str:
        raise NotImplementedError

    def build_user_prompt(self, ctx: RunContext) -> str:
        raise NotImplementedError

    def postprocess(self, ctx: RunContext, text: str) -> tuple[str, str, dict]:
        """后处理 LLM 输出。

        返回: (落盘文本, 一句话摘要, 额外信息)
        默认：直接落盘，摘要取首行。
        """
        summary = text.strip().split("\n", 1)[0][:80]
        return text, summary, {}

    # ----------------------------------------------------------
    # 模板方法
    # ----------------------------------------------------------
    def run(
        self,
        ctx: RunContext,
        stream_callback: StreamCallback | None = None,
    ) -> AgentResult:
        """执行 Agent：构建消息 → 流式调用 → 后处理 → 落盘。"""
        task_id = self.task.meta.task_id
        tid = self.name.value

        # 标记运行中
        rec = self.task.state.get_agent(self.name)
        rec.status = AgentStatus.RUNNING
        from datetime import datetime
        rec.started_at = datetime.now().isoformat(timespec="seconds")
        self.store.add_history(self.task, "started", agent=tid)
        self.store.save(self.task)

        self.store.append_log(task_id, tid, {"type": "start", "model": rec.model})

        try:
            messages = [
                Message("system", self.system_prompt),
                Message("user", self.build_user_prompt(ctx)),
            ]
            self.store.append_log(task_id, tid, {"type": "messages", "messages": [
                {"role": m["role"], "content": m["content"]} for m in messages
            ]})

            # 流式调用，边收边写日志（合并小块，避免日志过多）
            full_text = self._stream_and_log(ctx, messages, stream_callback)

            # 后处理
            artifact_text, summary, extra = self.postprocess(ctx, full_text)

            # 落盘
            artifact_path = self.store.write_artifact(task_id, self.name, artifact_text)

            # 更新记录
            rec.status = AgentStatus.DONE
            rec.finished_at = datetime.now().isoformat(timespec="seconds")
            rec.artifact_path = artifact_path
            rec.summary = summary
            self.store.add_history(self.task, "done", agent=tid, detail=summary)
            self.store.save(self.task)

            self.store.append_log(task_id, tid, {
                "type": "done", "summary": summary,
                "artifact_path": artifact_path, "extra": extra,
            })

            return AgentResult(
                agent=self.name, success=True,
                artifact_path=artifact_path, summary=summary, extra=extra,
            )

        except Exception as e:
            rec.status = AgentStatus.FAILED
            rec.error = str(e)
            rec.finished_at = datetime.now().isoformat(timespec="seconds")
            self.store.add_history(self.task, "failed", agent=tid, detail=str(e))
            self.store.save(self.task)
            self.store.append_log(task_id, tid, {"type": "error", "error": str(e)})
            return AgentResult(agent=self.name, success=False, error=str(e))

    def _stream_and_log(
        self,
        ctx: RunContext,
        messages: list[Message],
        stream_callback: StreamCallback | None,
    ) -> str:
        """流式拉取 LLM 输出，合并成块写日志，并回调推送。"""
        task_id = self.task.meta.task_id
        tid = self.name.value
        buf = ""
        full = ""
        FLUSH_SIZE = 120  # 每累积这么多字符写一条日志

        for chunk in self.llm.stream(messages):
            full += chunk
            buf += chunk
            if stream_callback:
                stream_callback(chunk)
            if len(buf) >= FLUSH_SIZE:
                self.store.append_log(task_id, tid, {"type": "delta", "text": buf})
                buf = ""
        if buf:
            self.store.append_log(task_id, tid, {"type": "delta", "text": buf})
        return full

    # ----------------------------------------------------------
    # 上下文辅助：读取上游产物
    # ----------------------------------------------------------
    def _prior(self, ctx: RunContext, agent: AgentName) -> str:
        return ctx.artifacts.get(agent.value, "")

    def _problem(self, ctx: RunContext) -> str:
        return self.store.read_problem(self.task.meta.task_id)
