"""测试共用 fixture：FakeLLM + 临时 store/task。"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest

from app.llm.provider import Message
from app.storage import TaskStore


class FakeLLM:
    """可脚本化的 LLM 替身。responses 可为 list[str] 或 callable(messages)->str。"""

    def __init__(self, responses):
        self._responses = list(responses) if isinstance(responses, list) else responses
        self.calls: list[list[Message]] = []

    def _next(self, messages):
        self.calls.append(messages)
        if callable(self._responses):
            return self._responses(messages)
        if not self._responses:
            raise AssertionError("FakeLLM 响应已耗尽")
        return self._responses.pop(0)

    def chat(self, messages, *, temperature=None, max_tokens=None) -> str:
        return self._next(messages)

    def stream(self, messages, *, temperature=None, max_tokens=None):
        yield self._next(messages)


@pytest.fixture
def make_store_task(tmp_path):
    """返回工厂 (problem_text="题目") -> (store, task)。"""
    from app.storage import RunMode

    def _make(problem_text: str = "这是一道测试题目。"):
        store = TaskStore(workspace_dir=tmp_path)
        task = store.create_task(
            title="测试任务", problem_text=problem_text, mode=RunMode.AUTO,
        )
        return store, task
    return _make
