"""任务运行器：后台线程跑编排器，发布订阅把事件实时推给 WebSocket。

编排器+Agent 是同步阻塞调用（LLM stream），所以放在独立线程，
通过 call_soon_threadsafe 把事件投递到 asyncio.Queue，WebSocket 消费。
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any


class TaskRunner:
    _instance: "TaskRunner | None" = None

    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue]] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    @classmethod
    def get(cls) -> "TaskRunner":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    # ---------------- 订阅 ----------------
    def subscribe(self, task_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._subscribers.setdefault(task_id, set()).add(q)
        return q

    def unsubscribe(self, task_id: str, q: asyncio.Queue) -> None:
        with self._lock:
            self._subscribers.get(task_id, set()).discard(q)

    def _publish(self, task_id: str, event: dict[str, Any]) -> None:
        """从编排器线程调用：把事件投递到该任务的所有订阅队列。"""
        with self._lock:
            subs = list(self._subscribers.get(task_id, set()))
        if self._loop is None:
            return
        for q in subs:
            try:
                self._loop.call_soon_threadsafe(q.put_nowait, event)
            except RuntimeError:
                pass  # loop 已关闭

    # ---------------- 运行/恢复 ----------------
    def is_running(self, task_id: str) -> bool:
        with self._lock:
            t = self._threads.get(task_id)
            return t is not None and t.is_alive()

    def _spawn(self, task_id: str, target) -> bool:
        with self._lock:
            t = self._threads.get(task_id)
            if t and t.is_alive():
                return False  # 已在运行
            t = threading.Thread(target=target, daemon=True)
            self._threads[task_id] = t
        t.start()
        return True

    def _handle_orchestrator_failure(self, task_id: str, err: BaseException) -> None:
        """orch.run()/resume() 抛异常时的兜底：落盘 FAILED 状态 + 发事件。

        否则 task.state 会永久停在 RUNNING（orchestrator 已 save 了 RUNNING，
        但异常发生在 agent.run() 之前，没人把状态改成 FAILED），前端永远显示"运行中"。
        """
        from app.storage import get_store, TaskStatus, AgentStatus, AGENT_ORDER
        try:
            store = get_store()
            task = store.load(task_id)
            if task.state.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                task.state.status = TaskStatus.FAILED
                # 标记当前卡在 RUNNING 的 agent 为失败
                for agent_name in AGENT_ORDER:
                    rec = task.state.agents.get(agent_name.value)
                    if rec and rec.status == AgentStatus.RUNNING:
                        rec.status = AgentStatus.FAILED
                        rec.error = str(err)
                store.add_history(task, "failed", detail=f"orchestrator 异常: {err}")
                store.save(task)
        except Exception:
            # 兜底自身不能再抛，否则线程静默死亡
            pass
        self._publish(task_id, {"type": "failed", "agent": "orchestrator",
                                "error": str(err)})

    def start(self, task_id: str) -> bool:
        from app.orchestrator import Orchestrator

        def target() -> None:
            try:
                orch = Orchestrator(task_id, event_sink=lambda e: self._publish(task_id, e))
                orch.run()
            except Exception as e:
                self._handle_orchestrator_failure(task_id, e)

        return self._spawn(task_id, target)

    def resume(self, task_id: str, decision: str, feedback: str | None) -> bool:
        from app.orchestrator import Orchestrator

        def target() -> None:
            try:
                orch = Orchestrator(task_id, event_sink=lambda e: self._publish(task_id, e))
                orch.resume(decision, feedback)
            except Exception as e:
                self._handle_orchestrator_failure(task_id, e)

        return self._spawn(task_id, target)
