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

    def start(self, task_id: str) -> bool:
        from app.orchestrator import Orchestrator

        def target() -> None:
            orch = Orchestrator(task_id, event_sink=lambda e: self._publish(task_id, e))
            try:
                orch.run()
            except Exception as e:
                self._publish(task_id, {"type": "failed", "agent": "orchestrator",
                                        "error": str(e)})

        return self._spawn(task_id, target)

    def resume(self, task_id: str, decision: str, feedback: str | None) -> bool:
        from app.orchestrator import Orchestrator

        def target() -> None:
            orch = Orchestrator(task_id, event_sink=lambda e: self._publish(task_id, e))
            try:
                orch.resume(decision, feedback)
            except Exception as e:
                self._publish(task_id, {"type": "failed", "agent": "orchestrator",
                                        "error": str(e)})

        return self._spawn(task_id, target)
