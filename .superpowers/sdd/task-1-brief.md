### Task 1: 测试脚手架 + pytest 依赖 + 项目基线入库

**Files:**
- Modify: `requirements.txt`
- Create: `tests/__init__.py`, `tests/conftest.py`

**Interfaces:**
- Produces: `FakeLLM`（`chat`/`stream` 接口，记录调用），`make_store_task(tmp_path)` fixture（返回 `(store, task)`，task 含题目文本）。

- [ ] **Step 1: 确认 frontend 嵌套仓库处理（需用户确认）**

当前 `frontend/` 自带 `.git`，会让 `git add` 把它当作 gitlink 而非普通文件。为统一为一个 GitHub 仓库，需移除 `frontend/.git`（会丢弃 frontend 的本地 git 历史，看上去是脚手架生成的极简历史）。向用户确认后再执行：
```bash
# 经用户确认后执行
rm -rf frontend/.git
```

- [ ] **Step 2: 把现有项目作为基线提交并推送**

```bash
cd "C:/Users/11422/Desktop/claudeWORKSPACE/modeling-agent"
git add -A
git status --short          # 确认 .env 未被加入（应被 .gitignore 忽略）
git commit -m "chore: 项目基线入库（求解器/写作器重设计前）"
git push origin main
```
Expected: 一个新提交推送成功；`.env` 不在提交中。

- [ ] **Step 3: 加 pytest 依赖**

修改 `requirements.txt`，在末尾追加：
```
# Testing
pytest>=8.0.0
```

- [ ] **Step 4: 创建测试包与 conftest**

`tests/__init__.py`（空文件即可）：

```python
```

`tests/conftest.py`：

```python
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
```

- [ ] **Step 5: 写一个冒烟测试验证 fixture 可用**

`tests/test_smoke.py`：

```python
def test_fake_llm_and_store(make_store_task):
    store, task = make_store_task("题目X")
    assert task.meta.title == "测试任务"
    from tests.conftest import FakeLLM
    llm = FakeLLM(["hello"])
    from app.llm.provider import Message
    assert llm.chat([Message("user", "hi")]) == "hello"
    assert len(llm.calls) == 1
```

- [ ] **Step 6: 运行测试**

Run: `python -m pytest tests/test_smoke.py -v`
Expected: PASS（若缺少 pytest，先 `pip install pytest`）。

- [ ] **Step 7: 提交并推送**

```bash
git add requirements.txt tests/
git commit -m "test: 测试脚手架 + pytest 依赖 + FakeLLM/fixture"
git push origin main
```

---

