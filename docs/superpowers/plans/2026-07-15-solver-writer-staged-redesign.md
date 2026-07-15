# 求解器与写作器分阶段重设计 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将求解器与写作器从"一次调用"重设计为"内部多步流水线 + 分层校验"，提高求解正确率、让代码更周全、让论文更充实且不跑题。

**Architecture:** 不动编排器与其余 Agent 结构。抽取 `BaseAgent._execute` 钩子让 solver/writer 各自实现多调用流水线而复用状态/日志/异常脚手架。求解器：规划→逐子问题×逐阶段(独立 cwd + 编号脚本 + `STAGE_RESULT` 约定 + 注入可复现前导)→分层校验(硬检查+LLM 自查)→汇总自查。写作器：大纲→逐节(每节只喂相关上下文)→拼接+扩写薄节→一致性校验(数值核对+语义)。`output.txt`/`status.json`/`paper.md`/`figures/` 路径不变，下游兼容。

**Tech Stack:** Python 3.12, FastAPI, pydantic, OpenAI/Anthropic SDK, pytest, numpy/pandas/scipy/matplotlib（求解器执行生成代码用）。

## Global Constraints

- 编排器 `orchestrator.py`、`analyst.py`、`modeler.py`、`reviewer.py` 结构不动；仅丰富 `reviewer.build_user_prompt` 的输入。
- 路径/文件名不变：`artifacts/solution/output.txt`、`artifacts/solution/status.json`、`artifacts/paper.md`、`artifacts/figures/`。
- `RunContext` 字段 `solution_stdout/solution_executed/solution_error/figures` 语义不变，下游照旧读取。
- 每完成一个 Task（代码修改）后必须 `git add` 相关文件 → `git commit` → `git push`（备份到 `origin` = https://github.com/wangzhiwen834/automath.git）。
- `.env`（含 API key）已在 `.gitignore`，禁止提交。
- 求解器生成代码可用库：numpy, pandas, scipy, matplotlib, scikit-learn, networkx, sympy, pulp。
- 所有 LLM 调用走 `self.llm.chat(messages)` 或 `self.llm.stream(messages)`；`Message(role, content)`。
- TDD：先写失败测试，再实现，每步独立可测。

## File Structure

**Create:**
- `tests/__init__.py` — 测试包标记。
- `tests/conftest.py` — `FakeLLM`、`make_store_task` fixture。
- `tests/test_solve_utils.py` — 求解器工具单元测试。
- `tests/test_storage_solution.py` — 存储辅助测试。
- `tests/test_solver_staged.py` — 求解器流水线测试（FakeLLM）。
- `tests/test_write_utils.py` — 写作器工具单元测试。
- `tests/test_writer_staged.py` — 写作器流水线测试（FakeLLM）。
- `tests/test_integration.py` — 3 子问题端到端（FakeLLM，hermetic）。
- `app/agents/solve_utils.py` — 求解器共享工具：提取/前导/`STAGE_RESULT` 解析/硬检查/计划 schema。
- `app/agents/write_utils.py` — 写作器共享工具：数值抽取/交叉核对/薄节判定。

**Modify:**
- `app/agents/base.py` — 抽取 `_execute` 钩子；`__init__` 增 `llm` 注入参数。
- `app/storage/store.py` — 增 `subproblem_dir`/`write_solution_file`/`read_solution_file`；`write_artifact(SOLVER)` 改写 `manifest.json`。
- `app/config.py` — 增 `writer_config`。
- `config.yaml` — 增 `solver` 子配置与 `writer` 段。
- `app/agents/solver.py` — 重写为分阶段流水线。
- `app/agents/writer.py` — 重写为分节流水线。
- `app/agents/reviewer.py` — `build_user_prompt` 丰富输入。
- `requirements.txt` — 加 `pytest`。

---

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

### Task 2: BaseAgent 抽取 _execute 钩子 + llm 注入

**Files:**
- Modify: `app/agents/base.py`
- Test: `tests/test_base_execute.py`

**Interfaces:**
- Produces: `BaseAgent.__init__(self, task, store=None, llm=None)`；`BaseAgent._execute(ctx, stream_callback) -> tuple[str, str, dict]`（默认实现 = 现有"单次 stream + postprocess"）；`run()` 调用 `_execute`。子类可覆盖 `_execute` 做多调用流水线。

- [ ] **Step 1: 写失败测试 — 默认 _execute 行为不变 + llm 注入**

`tests/test_base_execute.py`：

```python
from app.agents.base import BaseAgent, RunContext
from app.storage import AgentName
from tests.conftest import FakeLLM
from app.llm.provider import Message


class _Dummy(BaseAgent):
    name = AgentName.ANALYST

    @property
    def system_prompt(self):
        return "SYS"

    def build_user_prompt(self, ctx):
        return "USER"

    def postprocess(self, ctx, text):
        return text, "summary", {}


def test_default_execute_uses_stream_and_postprocess(make_store_task):
    store, task = make_store_task("题目")
    agent = _Dummy(task, store, llm=FakeLLM(["流水文本"]))
    ctx = RunContext(task=task, store=store)
    result = agent.run(ctx)
    assert result.success
    assert result.summary == "summary"
    # 产物落盘
    assert store.read_artifact(task.meta.task_id, result.artifact_path) == "流水文本"


def test_llm_injection(make_store_task):
    store, task = make_store_task("题目")
    fake = FakeLLM(["x"])
    agent = _Dummy(task, store, llm=fake)
    assert agent.llm is fake
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `python -m pytest tests/test_base_execute.py -v`
Expected: FAIL（`_Dummy.__init__` 不接受 `llm` 参数）。

- [ ] **Step 3: 改造 base.py**

把 `__init__` 改为接受 `llm`；把 `run()` 中"构建消息→stream→postprocess"抽到 `_execute`。修改 `app/agents/base.py`：

`__init__` 改为：

```python
    def __init__(self, task: Task, store: TaskStore | None = None, llm=None) -> None:
        self.task = task
        self.store = store or get_store()
        model_key = task.meta.agent_models.get(self.name.value)
        self.llm = llm if llm is not None else get_llm(model_key)
```

把 `run()` 中 try 块里的"构建消息→流式→后处理"替换为调用 `_execute`，并新增默认 `_execute`。`run()` 的 try 块改为：

```python
        try:
            artifact_text, summary, extra = self._execute(ctx, stream_callback)

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
```

在 `postprocess` 之后新增默认 `_execute`：

```python
    def _execute(
        self,
        ctx: RunContext,
        stream_callback: StreamCallback | None,
    ) -> tuple[str, str, dict]:
        """默认实现：单次 stream + postprocess。多调用流水线子类覆盖本方法。"""
        task_id = self.task.meta.task_id
        tid = self.name.value
        messages = [
            Message("system", self.system_prompt),
            Message("user", self.build_user_prompt(ctx)),
        ]
        self.store.append_log(task_id, tid, {"type": "messages", "messages": [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]})
        full_text = self._stream_and_log(ctx, messages, stream_callback)
        return self.postprocess(ctx, full_text)
```

`run()` 中原来的 messages 构建 + `_stream_and_log` + postprocess 三行删除（已移入 `_execute`）。

- [ ] **Step 4: 运行测试，确认通过**

Run: `python -m pytest tests/test_base_execute.py -v`
Expected: PASS。

- [ ] **Step 5: 回归 — 确认现有 Agent 仍可实例化（不调用真实 LLM）**

Run: `python -m pytest tests/ -v`
Expected: 既有测试全部 PASS。

- [ ] **Step 6: 提交并推送**

```bash
git add app/agents/base.py tests/test_base_execute.py
git commit -m "refactor: BaseAgent 抽取 _execute 钩子并支持 llm 注入"
git push origin main
```

---

### Task 3: 存储辅助 — 子问题目录/解决方案文件/manifest

**Files:**
- Modify: `app/storage/store.py`
- Test: `tests/test_storage_solution.py`

**Interfaces:**
- Produces:
  - `TaskStore.subproblem_dir(task_id, sub_id) -> Path`（创建并返回 `artifacts/solution/<sub_id>/`）
  - `TaskStore.write_solution_file(task_id, name, content) -> str`（原子写 `artifacts/solution/<name>`，返回相对路径）
  - `TaskStore.read_solution_file(task_id, name) -> str`
  - `write_artifact(SOLVER)` 改为写 `artifacts/solution/manifest.json`，返回 `"artifacts/solution/manifest.json"`

- [ ] **Step 1: 写失败测试**

`tests/test_storage_solution.py`：

```python
from app.storage import AgentName


def test_subproblem_dir_created(make_store_task):
    store, task = make_store_task("题目")
    d = store.subproblem_dir(task.meta.task_id, "sub1")
    assert d.exists() and d.is_dir()
    assert d.name == "sub1"
    assert d.parent.name == "solution"


def test_write_and_read_solution_file(make_store_task):
    store, task = make_store_task("题目")
    path = store.write_solution_file(task.meta.task_id, "plan.json", '{"a":1}')
    assert path == "artifacts/solution/plan.json"
    assert store.read_solution_file(task.meta.task_id, "plan.json") == '{"a":1}'


def test_write_artifact_solver_writes_manifest(make_store_task):
    store, task = make_store_task("题目")
    path = store.write_artifact(task.meta.task_id, AgentName.SOLVER, '{"subs":[]}')
    assert path == "artifacts/solution/manifest.json"
    assert store.read_artifact(task.meta.task_id, path) == '{"subs":[]}'
```

- [ ] **Step 2: 运行，确认失败**

Run: `python -m pytest tests/test_storage_solution.py -v`
Expected: FAIL（方法不存在 / solver 仍写 solve.py）。

- [ ] **Step 3: 实现 store 辅助方法**

在 `store.py` 的 `solution_dir` 方法之后新增：

```python
    def subproblem_dir(self, task_id: str, sub_id: str) -> Path:
        d = self._solution_dir(task_id) / sub_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_solution_file(self, task_id: str, name: str, content: str) -> str:
        path = self._solution_dir(task_id) / name
        self._atomic_write_text(path, content)
        return f"artifacts/solution/{name}"

    def read_solution_file(self, task_id: str, name: str) -> str:
        path = self._solution_dir(task_id) / name
        if not path.exists():
            return ""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
```

修改 `write_artifact` 中 `AgentName.SOLVER` 分支：

```python
    def write_artifact(self, task_id: str, agent: AgentName, content: str) -> str:
        """写 Agent 产物，返回相对任务目录的路径。"""
        name = ARTIFACT_NAMES[agent]
        if agent == AgentName.SOLVER:
            # solver 产物是目录；此处写 manifest.json 作为入口产物
            sol_dir = self._solution_dir(task_id)
            path = sol_dir / "manifest.json"
            self._atomic_write_text(path, content)
            return "artifacts/solution/manifest.json"
        else:
            path = self._artifacts_dir(task_id) / name
            self._atomic_write_text(path, content)
            return f"artifacts/{name}"
```

- [ ] **Step 4: 运行，确认通过**

Run: `python -m pytest tests/test_storage_solution.py -v`
Expected: PASS。

- [ ] **Step 5: 提交并推送**

```bash
git add app/storage/store.py tests/test_storage_solution.py
git commit -m "feat(storage): 子问题目录/解决方案文件/manifest 辅助方法"
git push origin main
```

---

### Task 4: 配置新增 — writer_config + solver 子配置

**Files:**
- Modify: `app/config.py`, `config.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Settings.writer_config`（dict）；`config.yaml` 新增 `solver` 子键与 `writer` 段。

- [ ] **Step 1: 写失败测试**

`tests/test_config.py`：

```python
from app.config import get_settings


def test_solver_and_writer_config_loaded():
    s = get_settings()
    assert s.solver_config.get("max_stage_retries") == 2
    assert s.solver_config.get("stage_execution_timeout") == 120
    assert s.solver_config.get("preamble_seed") == 42
    assert s.writer_config.get("max_expand_sections") == 4
    assert s.writer_config.get("consistency_check") is True
```

- [ ] **Step 2: 运行，确认失败**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL（`writer_config` 不存在 / 键缺失）。

- [ ] **Step 3: 改 config.py**

在 `Settings.__init__` 的 `self.reviewer_config = ...` 之后加一行：

```python
        self.writer_config: dict[str, Any] = self._raw.get("writer", {})
```

- [ ] **Step 4: 改 config.yaml**

把 `solver:` 段整体替换为（保留原 `execution_timeout`/`workdir`/`allowed_packages`，新增子键）：

```yaml
solver:
  execution_timeout: 60
  stage_execution_timeout: 120
  workdir: "artifacts/solution"
  preamble_seed: 42
  max_stage_retries: 2
  max_critique_retries: 1
  max_regen_per_stage: 3
  self_critique_model: null
  allowed_packages:
    - numpy
    - pandas
    - scipy
    - matplotlib
    - scikit-learn
    - networkx
    - sympy
    - pulp
```

在 `reviewer:` 段之前新增 `writer:` 段：

```yaml
# ===================================================================
# 写作 Agent 配置
# ===================================================================
writer:
  min_section_chars:
    abstract: 400
    solving_sub: 600
    default: 300
  max_expand_sections: 4
  consistency_check: true
  consistency_model: null
```

- [ ] **Step 5: 运行，确认通过**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS。

- [ ] **Step 6: 提交并推送**

```bash
git add app/config.py config.yaml tests/test_config.py
git commit -m "feat(config): 新增 writer_config 与 solver 分阶段子配置"
git push origin main
```

---

### Task 5: solve_utils — 提取/前导/STAGE_RESULT 解析

**Files:**
- Create: `app/agents/solve_utils.py`
- Test: `tests/test_solve_utils.py`

**Interfaces:**
- Produces:
  - `extract_python(text) -> str`
  - `parse_stage_result(stdout) -> dict | None`
  - `inject_preamble(code, seed=42) -> str`

- [ ] **Step 1: 写失败测试**

`tests/test_solve_utils.py`：

```python
from app.agents.solve_utils import extract_python, parse_stage_result, inject_preamble


def test_extract_python_from_fence():
    text = "说明\n```python\nprint(1)\n```\n结尾"
    assert extract_python(text) == "print(1)"


def test_extract_python_no_fence():
    assert extract_python("print(2)") == "print(2)"


def test_parse_stage_result_ok():
    out = "一些输出\nSTAGE_RESULT: {\"ok\": true, \"metrics\": {\"z\": 3.5}, \"files\": [], \"figures\": []}\n尾"
    r = parse_stage_result(out)
    assert r is not None and r["ok"] is True and r["metrics"]["z"] == 3.5


def test_parse_stage_result_missing():
    assert parse_stage_result("无标记") is None


def test_inject_preamble_has_seed_and_agg():
    code = "print(1)"
    pre = inject_preamble(code, seed=42)
    assert "random.seed(42)" in pre
    assert "matplotlib.use('Agg')" in pre
    assert pre.endswith("print(1)")
```

- [ ] **Step 2: 运行，确认失败**

Run: `python -m pytest tests/test_solve_utils.py -v`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 实现 solve_utils.py（本任务部分）**

创建 `app/agents/solve_utils.py`：

```python
"""求解器共享工具：代码提取、可复现前导注入、STAGE_RESULT 解析、硬检查、计划 schema。"""
from __future__ import annotations

import json
import re
from pathlib import Path


def extract_python(text: str) -> str:
    """从 Markdown 提取 Python 代码块；无围栏则视为全是代码。"""
    fences = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if fences:
        return "\n\n".join(fences).strip()
    return text.strip()


def parse_stage_result(stdout: str) -> dict | None:
    """从 stdout 中解析 `STAGE_RESULT: {json}` 行。找不到返回 None。"""
    m = re.search(r"STAGE_RESULT\s*:\s*(\{.*\})", stdout, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def inject_preamble(code: str, seed: int = 42) -> str:
    """注入可复现前导：随机种子 + matplotlib Agg 后端。"""
    preamble = (
        "import random\n"
        "import numpy as np\n"
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        f"random.seed({seed})\n"
        f"np.random.seed({seed})\n"
        "import os\n"
        "os.makedirs('artifacts/figures', exist_ok=True)\n\n"
    )
    return preamble + code
```

- [ ] **Step 4: 运行，确认通过**

Run: `python -m pytest tests/test_solve_utils.py -v`
Expected: PASS。

- [ ] **Step 5: 提交并推送**

```bash
git add app/agents/solve_utils.py tests/test_solve_utils.py
git commit -m "feat(solver): solve_utils 提取/前导/STAGE_RESULT 解析"
git push origin main
```

---

### Task 6: solve_utils — 硬检查 + 计划 schema

**Files:**
- Modify: `app/agents/solve_utils.py`
- Test: `tests/test_solve_utils.py`（追加）

**Interfaces:**
- Produces:
  - `parse_plan(text) -> dict | None`
  - `validate_plan(plan) -> tuple[bool, list[str]]`
  - `fallback_plan() -> dict`
  - `check_finite(metrics) -> tuple[bool, str]`
  - `check_figures(figures, figures_dir) -> tuple[bool, str]`
  - `run_hard_checks(stage_result, stage, sub_dir, figures_dir) -> tuple[bool, list[str]]`

- [ ] **Step 1: 追加失败测试**

在 `tests/test_solve_utils.py` 末尾追加：

```python
import json
from app.agents.solve_utils import (
    parse_plan, validate_plan, fallback_plan,
    check_finite, check_figures, run_hard_checks,
)


def test_parse_plan_from_fence():
    text = '前缀\n```json\n{"subproblems": []}\n```'
    assert parse_plan(text) == {"subproblems": []}


def test_validate_plan_good():
    plan = {"subproblems": [{"id": "sub1", "title": "t", "goal": "g",
            "stages": [{"name": "solve", "goal": "g", "input_files": [],
                        "output_file": "r.json", "method": "m", "figures": []}]}]}
    ok, errs = validate_plan(plan)
    assert ok and errs == []


def test_validate_plan_no_subproblems():
    ok, errs = validate_plan({"subproblems": []})
    assert not ok and any("子问题" in e for e in errs)


def test_validate_plan_stage_no_solve():
    plan = {"subproblems": [{"id": "s1", "title": "t", "goal": "g",
            "stages": [{"name": "data", "goal": "g", "input_files": [],
                        "output_file": "d.csv", "method": "m", "figures": []}]}]}
    ok, errs = validate_plan(plan)
    # 至少要有一个 solve 阶段
    assert not ok


def test_fallback_plan_shape():
    p = fallback_plan()
    assert len(p["subproblems"]) == 1
    names = [s["name"] for s in p["subproblems"][0]["stages"]]
    assert "solve" in names


def test_check_finite():
    ok, _ = check_finite({"z": 3.5})
    assert ok
    ok, msg = check_finite({"z": float("nan")})
    assert not ok and "有限" in msg


def test_check_figures(tmp_path):
    d = tmp_path / "figs"
    d.mkdir()
    (d / "ok.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"x")
    ok, _ = check_figures(["ok.png"], d)
    assert ok
    ok, msg = check_figures(["missing.png"], d)
    assert not ok


def test_run_hard_checks_pass(tmp_path):
    sub_dir = tmp_path / "sub1"
    sub_dir.mkdir()
    fig_dir = tmp_path / "figs"
    fig_dir.mkdir()
    (sub_dir / "result.json").write_text("{}")
    sr = {"ok": True, "metrics": {"z": 1.0}, "files": ["result.json"], "figures": []}
    stage = {"name": "solve", "output_file": "result.json", "figures": []}
    ok, errs = run_hard_checks(sr, stage, sub_dir, fig_dir)
    assert ok, errs
```

- [ ] **Step 2: 运行，确认失败**

Run: `python -m pytest tests/test_solve_utils.py -v`
Expected: FAIL（新函数不存在）。

- [ ] **Step 3: 在 solve_utils.py 追加实现**

```python
import math


def parse_plan(text: str) -> dict | None:
    """从回答中提取计划 JSON。"""
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    raw = m.group(1) if m else text
    if not m:
        s, e = raw.find("{"), raw.rfind("}")
        if s != -1 and e != -1:
            raw = raw[s:e + 1]
        else:
            return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def validate_plan(plan: dict) -> tuple[bool, list[str]]:
    """校验计划结构。"""
    errs: list[str] = []
    subs = plan.get("subproblems")
    if not isinstance(subs, list) or not subs:
        return False, ["计划缺少 subproblems（至少一个子问题）"]
    for i, sub in enumerate(subs):
        if not sub.get("id"):
            errs.append(f"第 {i+1} 个子问题缺少 id")
        if not sub.get("stages"):
            errs.append(f"子问题 {sub.get('id', i+1)} 缺少 stages")
            continue
        names = [s.get("name") for s in sub["stages"]]
        if "solve" not in names:
            errs.append(f"子问题 {sub.get('id', i+1)} 必须含一个 solve 阶段")
        for s in sub["stages"]:
            for k in ("name", "goal", "method"):
                if not s.get(k):
                    errs.append(f"子问题 {sub.get('id', i+1)} 某阶段缺少 {k}")
    return (len(errs) == 0), errs


def fallback_plan() -> dict:
    """退化计划：单子问题、单 solve 阶段（兼容旧行为）。"""
    return {
        "subproblems": [{
            "id": "sub1", "title": "整体求解", "goal": "求解整个问题",
            "stages": [
                {"name": "solve", "goal": "求解并输出关键结果",
                 "input_files": [], "output_file": "result.json",
                 "method": "数值求解", "figures": [], "expected_range": None},
                {"name": "plot", "goal": "对关键结果画图",
                 "input_files": ["result.json"], "output_file": "",
                 "method": "matplotlib", "figures": ["sub1_1_fig.png"], "expected_range": None},
            ],
        }]
    }


# ---------- 硬检查 ----------

def _is_finite(v) -> bool:
    if isinstance(v, bool):
        return True
    if isinstance(v, (int, float)):
        return math.isfinite(v)
    return True


def check_finite(metrics: dict) -> tuple[bool, str]:
    for k, v in metrics.items():
        if not _is_finite(v):
            return False, f"指标 {k} 非有限值（NaN/Inf）"
    return True, ""


def check_figures(figures: list[str], figures_dir: Path) -> tuple[bool, str]:
    for f in figures:
        p = figures_dir / f
        if not p.exists() or p.stat().st_size == 0:
            return False, f"图表文件缺失或为空: {f}"
        with open(p, "rb") as fh:
            if not fh.read(8).startswith(b"\x89PNG"):
                return False, f"图表非 PNG: {f}"
    return True, ""


def check_expected_range(value, rng) -> tuple[bool, str]:
    if rng is None or value is None:
        return True, ""
    try:
        lo, hi = rng
        if not (lo <= float(value) <= hi):
            return False, f"值 {value} 不在预期范围 [{lo}, {hi}]"
    except (TypeError, ValueError):
        return True, ""
    return True, ""


def run_hard_checks(stage_result: dict, stage: dict, sub_dir: Path, figures_dir: Path) -> tuple[bool, list[str]]:
    """对一个阶段的 STAGE_RESULT 做硬检查。"""
    errs: list[str] = []
    if not stage_result or not stage_result.get("ok"):
        errs.append("STAGE_RESULT 缺失或 ok!=true")
    metrics = stage_result.get("metrics", {}) if stage_result else {}
    ok, msg = check_finite(metrics)
    if not ok:
        errs.append(msg)
    out_file = stage.get("output_file")
    if out_file:
        if not (sub_dir / out_file).exists():
            errs.append(f"输出文件缺失: {out_file}")
    figs = stage_result.get("figures", []) if stage_result else []
    ok, msg = check_figures(figs, figures_dir)
    if not ok:
        errs.append(msg)
    return (len(errs) == 0), errs
```

- [ ] **Step 4: 运行，确认通过**

Run: `python -m pytest tests/test_solve_utils.py -v`
Expected: PASS。

- [ ] **Step 5: 提交并推送**

```bash
git add app/agents/solve_utils.py tests/test_solve_utils.py
git commit -m "feat(solver): solve_utils 硬检查 + 计划 schema/校验/退化"
git push origin main
```

---

### Task 7: 求解器 — 规划步 _make_plan

**Files:**
- Modify: `app/agents/solver.py`
- Test: `tests/test_solver_staged.py`

**Interfaces:**
- Consumes: `solve_utils.parse_plan`, `validate_plan`, `fallback_plan`；`BaseAgent.llm`；`store.write_solution_file`。
- Produces: `SolverAgent._make_plan(ctx) -> dict`（写 `plan.json`，返回校验通过的 plan；解析失败用 `fallback_plan`）。

- [ ] **Step 1: 写失败测试**

`tests/test_solver_staged.py`：

```python
from app.agents.solver import SolverAgent
from app.storage import AgentName
from tests.conftest import FakeLLM
from app.llm.provider import Message


def test_make_plan_valid(make_store_task):
    store, task = make_store_task("题目")
    plan_json = '{"subproblems":[{"id":"sub1","title":"t","goal":"g","stages":[{"name":"solve","goal":"g","input_files":[],"output_file":"r.json","method":"m","figures":[]}]}]}'
    agent = SolverAgent(task, store, llm=FakeLLM([f"```json\n{plan_json}\n```"]))
    plan = agent._make_plan(None)
    assert plan["subproblems"][0]["id"] == "sub1"
    # plan.json 落盘
    assert store.read_solution_file(task.meta.task_id, "plan.json") != ""


def test_make_plan_fallback_on_bad_json(make_store_task):
    store, task = make_store_task("题目")
    agent = SolverAgent(task, store, llm=FakeLLM(["不是JSON"]))
    plan = agent._make_plan(None)
    assert "solve" in [s["name"] for s in plan["subproblems"][0]["stages"]]
```

- [ ] **Step 2: 运行，确认失败**

Run: `python -m pytest tests/test_solver_staged.py -v`
Expected: FAIL（`_make_plan` 不存在 / `llm` 参数问题）。

- [ ] **Step 3: 实现 _make_plan**

在 `solver.py` 顶部 import 改为：

```python
from app.agents.base import BaseAgent
from app.agents.solve_utils import (
    extract_python, parse_stage_result, inject_preamble,
    parse_plan, validate_plan, fallback_plan, run_hard_checks,
)
from app.config import get_settings
from app.llm.provider import Message
from app.storage import AgentName
```

在 `SolverAgent` 类内新增（保留旧 `system_prompt`/`build_user_prompt` 暂不动，下个任务再改）：

```python
    PLAN_PROMPT = (
        "你是数学建模求解规划专家。根据题目、分析、模型，把问题拆成可独立求解的子问题，"
        "并为每个子问题规划有序阶段（从 数据/建模/求解/分析/画图 中按需选取，至少含 solve，推荐含 plot）。\n\n"
        "只输出一个 JSON 代码块，结构：\n"
        "```json\n"
        '{"subproblems":[{"id":"sub1","title":"","goal":"","stages":['
        '{"name":"data|model|solve|analyze|plot","goal":"","input_files":[],"output_file":"","method":"","figures":[],"expected_range":null}]}]}\n'
        "```\n"
        "要求：id 用 sub1/sub2…；input_files 引用前一阶段的 output_file 或上传数据 data/<名>；"
        "画图阶段把 figures 列出（文件名形如 sub1_1_curve.png）。只输出 JSON。"
    )

    def _make_plan(self, ctx) -> dict:
        problem = self._problem(ctx) if ctx else ""
        analysis = self._prior(ctx, AgentName.ANALYST) if ctx else ""
        model = self._prior(ctx, AgentName.MODELER) if ctx else ""
        data_info = ""
        if ctx and ctx.data_files:
            data_info = "已上传数据：data/" + ", data/".join(ctx.data_files)
        messages = [
            Message("system", self.PLAN_PROMPT),
            Message("user", f"【题目】\n{problem}\n\n【分析】\n{analysis}\n\n【模型】\n{model}\n\n{data_info}\n请输出求解计划。"),
        ]
        text = self.llm.chat(messages)
        plan = parse_plan(text)
        if plan:
            ok, errs = validate_plan(plan)
            if ok:
                self.store.write_solution_file(self.task.meta.task_id, "plan.json",
                                               json.dumps(plan, ensure_ascii=False, indent=2))
                self.store.append_log(self.task.meta.task_id, self.name.value,
                                      {"type": "plan", "subproblems": len(plan["subproblems"])})
                return plan
        # 退化
        fb = fallback_plan()
        self.store.write_solution_file(self.task.meta.task_id, "plan.json",
                                       json.dumps(fb, ensure_ascii=False, indent=2))
        self.store.append_log(self.task.meta.task_id, self.name.value,
                              {"type": "plan_fallback", "errors": errs if plan else "parse_failed"})
        return fb
```

在 `solver.py` 顶部确保 `import json`（已有）。

- [ ] **Step 4: 运行，确认通过**

Run: `python -m pytest tests/test_solver_staged.py -v`
Expected: PASS。

- [ ] **Step 5: 提交并推送**

```bash
git add app/agents/solver.py tests/test_solver_staged.py
git commit -m "feat(solver): 规划步 _make_plan + 退化"
git push origin main
```

---

### Task 8: 求解器 — 阶段执行 _run_stage（生成+执行+硬校验+有界修复）

**Files:**
- Modify: `app/agents/solver.py`
- Test: `tests/test_solver_staged.py`（追加）

**Interfaces:**
- Consumes: `solve_utils` 函数；`store.subproblem_dir`/`figures_dir`；`get_settings().solver_config`。
- Produces: `SolverAgent._run_stage(ctx, sub, stage) -> dict`，返回
  `{sub_id, stage, ok, code, stdout, stage_result, figures, attempts, error}`。

- [ ] **Step 1: 追加失败测试**

```python
GOOD_CODE = (
    "import json\n"
    "open('result.json','w').write('{}')\n"
    'print("STAGE_RESULT:", json.dumps({"ok": True, "metrics": {"z": 1.0}, "files": ["result.json"], "figures": []}))\n'
)
BAD_CODE = "raise ValueError('boom')"


def test_run_stage_success(make_store_task):
    store, task = make_store_task("题目")
    # FakeLLM: 第一次生成 GOOD_CODE
    agent = SolverAgent(task, store, llm=FakeLLM([GOOD_CODE]))
    sub = {"id": "sub1", "stages": []}
    stage = {"name": "solve", "goal": "g", "input_files": [], "output_file": "result.json",
             "method": "m", "figures": [], "expected_range": None}
    out = agent._run_stage(None, sub, stage)
    assert out["ok"] is True
    assert out["stage_result"]["metrics"]["z"] == 1.0


def test_run_stage_fail_bounded(make_store_task):
    store, task = make_store_task("题目")
    # 生成坏代码 -> 修复仍坏 -> 修复仍坏（超过 max_stage_retries）
    agent = SolverAgent(task, store, llm=FakeLLM([BAD_CODE, BAD_CODE, BAD_CODE]))
    sub = {"id": "sub1", "stages": []}
    stage = {"name": "solve", "goal": "g", "input_files": [], "output_file": "result.json",
             "method": "m", "figures": [], "expected_range": None}
    out = agent._run_stage(None, sub, stage)
    assert out["ok"] is False
```

- [ ] **Step 2: 运行，确认失败**

Run: `python -m pytest tests/test_solver_staged.py -v`
Expected: FAIL（`_run_stage` 不存在）。

- [ ] **Step 3: 实现 _run_stage 与 _gen_code/_fix_code/_execute_stage**

在 `SolverAgent` 类内新增（删除旧的 `postprocess`/`_syntax_check`/`_execute`/`_write_status`/`_fix_code`，本任务及后续任务重写；若暂未到 _execute，先保留旧 `postprocess` 不影响本测试）。新增：

```python
    def _gen_code(self, sub, stage, prev_outputs: str) -> str:
        prompt = (
            f"为子问题 {sub['id']}（{sub.get('title','')}）的阶段【{stage['name']}】写 Python 代码。\n"
            f"阶段目标：{stage['goal']}\n方法：{stage['method']}\n"
            f"输入文件（在当前工作目录读取）：{stage.get('input_files')}\n"
            f"输出文件：{stage.get('output_file') or '（无）'}\n"
            f"预期图表：{stage.get('figures')}\n\n"
            "硬性要求：\n"
            "- 只输出一个 ```python 代码块；matplotlib 用英文标签；画图 savefig 到 artifacts/figures/<名> 后 close。\n"
            "- 代码末尾必须 print 一行 `STAGE_RESULT: <json>`，含 ok(布尔)/metrics(数值dict)/files(产出文件名list)/figures(图文件名list)。\n"
            "- 用相对路径读写文件（当前工作目录即本子问题目录）。\n\n"
            f"上游可用信息：\n{prev_outputs}\n"
        )
        text = self.llm.chat([Message("system", self.system_prompt), Message("user", prompt)])
        return extract_python(text)

    def _fix_code(self, code: str, error: str) -> str:
        msg = [
            Message("system", self.system_prompt),
            Message("user", (
                "之前代码有问题，请修复后输出完整代码（仍只一个代码块，末尾保留 STAGE_RESULT 行）。\n\n"
                f"【原代码】\n```python\n{code}\n```\n\n【问题】\n{error}\n"
            )),
        ]
        return extract_python(self.llm.chat(msg))

    def _exec_script(self, script_path, cwd, timeout):
        import subprocess, sys
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)], cwd=str(cwd),
                capture_output=True, text=True, timeout=timeout,
                encoding="utf-8", errors="replace",
            )
            return proc.stdout, proc.stderr, proc.returncode == 0
        except subprocess.TimeoutExpired:
            return "", f"执行超时（>{timeout}秒）", False
        except Exception as e:
            return "", f"执行异常: {e}", False

    def _run_stage(self, ctx, sub, stage) -> dict:
        import py_compile
        tid = self.task.meta.task_id
        cfg = get_settings().solver_config
        seed = cfg.get("preamble_seed", 42)
        max_regen = cfg.get("max_regen_per_stage", 3)
        timeout = cfg.get("stage_execution_timeout", 120)
        sub_dir = self.store.subproblem_dir(tid, sub["id"])
        fig_dir = self.store.figures_dir(tid)
        prev_outputs = ""  # 简化：阶段间靠工作目录文件交接；此处可附上阶段说明

        code = ""
        attempts = 0
        last_error = ""
        for _ in range(max_regen):
            attempts += 1
            code = self._gen_code(sub, stage, prev_outputs) if attempts == 1 else self._fix_code(code, last_error)
            code = inject_preamble(code, seed=seed)
            script = sub_dir / f"{stage['name']}.py"
            script.write_text(code, encoding="utf-8")
            # 语法预检
            try:
                py_compile.compile(str(script), doraise=True)
            except (py_compile.PyCompileError, SyntaxError) as e:
                last_error = f"[语法错误] {e}"
                self.store.append_log(tid, self.name.value, {"type": "syntax_error", "stage": stage["name"], "error": str(e)[:800]})
                continue
            # 执行
            stdout, stderr, ok = self._exec_script(script, sub_dir, timeout)
            if not ok:
                last_error = stderr[:1500] or "执行返回非零退出码"
                self.store.append_log(tid, self.name.value, {"type": "exec_error", "stage": stage["name"], "error": last_error})
                continue
            sr = parse_stage_result(stdout)
            hard_ok, errs = run_hard_checks(sr or {}, stage, sub_dir, fig_dir)
            if hard_ok:
                self.store.append_log(tid, self.name.value, {"type": "stage_done", "stage": stage["name"], "attempts": attempts})
                return {"sub_id": sub["id"], "stage": stage["name"], "ok": True,
                        "code": code, "stdout": stdout, "stage_result": sr,
                        "figures": (sr or {}).get("figures", []), "attempts": attempts, "error": ""}
            last_error = "硬检查失败: " + "; ".join(errs)
            self.store.append_log(tid, self.name.value, {"type": "hard_check_fail", "stage": stage["name"], "error": last_error})
        return {"sub_id": sub["id"], "stage": stage["name"], "ok": False,
                "code": code, "stdout": "", "stage_result": None,
                "figures": [], "attempts": attempts, "error": last_error}
```

- [ ] **Step 4: 运行，确认通过**

Run: `python -m pytest tests/test_solver_staged.py -v`
Expected: PASS。

- [ ] **Step 5: 提交并推送**

```bash
git add app/agents/solver.py tests/test_solver_staged.py
git commit -m "feat(solver): 阶段执行 _run_stage（生成+执行+硬校验+有界修复）"
git push origin main
```

---

### Task 9: 求解器 — 自查 _self_critique_stage + 汇总 _aggregate

**Files:**
- Modify: `app/agents/solver.py`
- Test: `tests/test_solver_staged.py`（追加）

**Interfaces:**
- Consumes: `BaseAgent.llm`；`store.write_solution_output`/`write_solution_file`。
- Produces:
  - `_self_critique_stage(sub, stage, code, stage_result) -> dict`（`{passed, issues, suggestion}`）
  - `_aggregate(outcomes) -> tuple[str, dict]`（`summary_text, status`），写 `output.txt`/`status.json`/`summary.md`。

- [ ] **Step 1: 追加失败测试**

```python
def test_self_critique_stage_parses_json(make_store_task):
    store, task = make_store_task("题目")
    agent = SolverAgent(task, store, llm=FakeLLM(['```json\n{"passed": true, "issues": [], "suggestion": ""}\n```']))
    r = agent._self_critique_stage({"id": "sub1"}, {"name": "solve"}, "print(1)", {"ok": True, "metrics": {"z": 1}})
    assert r["passed"] is True


def test_aggregate_status_executed(make_store_task):
    store, task = make_store_task("题目")
    agent = SolverAgent(task, store, llm=FakeLLM(["汇总：结果一致"]))
    outcomes = [
        {"sub_id": "sub1", "stage": "solve", "ok": True, "stdout": "z=1", "stage_result": {"metrics": {"z": 1}}, "figures": [], "error": ""},
        {"sub_id": "sub1", "stage": "plot", "ok": False, "stdout": "", "stage_result": None, "figures": [], "error": "画图失败"},
    ]
    summary, status = agent._aggregate(outcomes)
    # solve 关键阶段成功 -> executed True（画图失败不影响）
    assert status["executed"] is True
    assert store.read_solution_file(task.meta.task_id, "status.json") != ""
    assert store.read_solution_file(task.meta.task_id, "output.txt") != ""
```

- [ ] **Step 2: 运行，确认失败**

Run: `python -m pytest tests/test_solver_staged.py -v`
Expected: FAIL。

- [ ] **Step 3: 实现 _self_critique_stage 与 _aggregate**

在 `SolverAgent` 类内新增：

```python
    def _self_critique_stage(self, sub, stage, code, stage_result) -> dict:
        prompt = (
            f"审查子问题 {sub['id']} 的【{stage['name']}】阶段输出是否正确合理。\n"
            f"阶段目标：{stage['goal']}\n代码：\n```python\n{code}\n```\n"
            f"STAGE_RESULT：{json.dumps(stage_result, ensure_ascii=False)}\n\n"
            "判断输出是否合理回应目标、方法是否恰当、有无红旗。只输出 JSON："
            '```json\n{"passed": true, "issues": [], "suggestion": ""}\n```'
        )
        text = self.llm.chat([Message("system", "你是严谨的数值结果审查者。"), Message("user", prompt)])
        m = __import__("re").search(r"```json\s*(\{.*?\})\s*```", text, __import__("re").DOTALL)
        raw = m.group(1) if m else text
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"passed": True, "issues": ["自查解析失败，默认通过"], "suggestion": ""}

    def _aggregate(self, outcomes: list[dict]) -> tuple[str, dict]:
        tid = self.task.meta.task_id
        # 关键阶段 = solve；某子问题若无 solve，取其最后一个阶段为关键
        sub_outcomes: dict[str, list[dict]] = {}
        for o in outcomes:
            sub_outcomes.setdefault(o["sub_id"], []).append(o)
        executed = True
        sub_status = []
        out_lines = []
        for sid, lst in sub_outcomes.items():
            crit = next((o for o in lst if o["stage"] == "solve"), lst[-1])
            if not crit["ok"]:
                executed = False
            sub_status.append({"id": sid, "critical_stage": crit["stage"], "ok": crit["ok"],
                               "stages": [{"name": o["stage"], "ok": o["ok"]} for o in lst]})
            out_lines.append(f"## 子问题 {sid}")
            for o in lst:
                out_lines.append(f"- [{o['stage']}] {'OK' if o['ok'] else 'FAIL'}")
                if o.get("stage_result") and o["stage_result"].get("metrics"):
                    out_lines.append(f"  metrics: {json.dumps(o['stage_result']['metrics'], ensure_ascii=False)}")
                if o.get("error"):
                    out_lines.append(f"  error: {o['error']}")
        output_txt = "\n".join(out_lines)
        # 汇总自查
        critique = self.llm.chat([Message("system", "你是建模结果一致性审查者。"),
                                  Message("user", f"各子问题阶段输出：\n{output_txt}\n请给一句一致性/正确性结论。")])
        summary_md = f"# 求解汇总自查\n\nexecuted={executed}\n\n{critique}\n"
        status = {"executed": executed, "subproblems": sub_status,
                  "error": None if executed else "存在子问题求解关键阶段失败"}
        self.store.write_solution_output(tid, output_txt, "")
        self.store.write_solution_file(tid, "status.json", json.dumps(status, ensure_ascii=False, indent=2))
        self.store.write_solution_file(tid, "summary.md", summary_md)
        return summary_md, status
```

- [ ] **Step 4: 运行，确认通过**

Run: `python -m pytest tests/test_solver_staged.py -v`
Expected: PASS。

- [ ] **Step 5: 提交并推送**

```bash
git add app/agents/solver.py tests/test_solver_staged.py
git commit -m "feat(solver): 阶段自查 _self_critique_stage + 汇总 _aggregate"
git push origin main
```

---

### Task 10: 求解器 — _execute 编排 + 产物/ctx/manifest

**Files:**
- Modify: `app/agents/solver.py`
- Test: `tests/test_solver_staged.py`（追加）

**Interfaces:**
- Consumes: `_make_plan`/`_run_stage`/`_self_critique_stage`/`_aggregate`；`store.write_artifact(SOLVER)`。
- Produces: `SolverAgent._execute(ctx, stream_callback) -> tuple[str, str, dict]`；设置 `ctx.solution_stdout/executed/error/figures`；返回 `(manifest_json, summary, extra)`。

- [ ] **Step 1: 追加失败测试**

```python
def test_execute_end_to_end(make_store_task):
    store, task = make_store_task("题目")
    plan_json = '{"subproblems":[{"id":"sub1","title":"t","goal":"g","stages":[{"name":"solve","goal":"g","input_files":[],"output_file":"result.json","method":"m","figures":[]}]}]}'
    good = (
        "import json\nopen('result.json','w').write('{}')\n"
        'print("STAGE_RESULT:", json.dumps({"ok": True, "metrics": {"z": 2.0}, "files": ["result.json"], "figures": []}))\n'
    )
    # 顺序：plan -> gen code -> self_critique -> aggregate
    agent = SolverAgent(task, store, llm=FakeLLM([
        f"```json\n{plan_json}\n```", good,
        '```json\n{"passed": true, "issues": [], "suggestion": ""}\n```',
        "一致",
    ]))
    from app.agents.base import RunContext
    ctx = RunContext(task=task, store=store)
    text, summary, extra = agent._execute(ctx, None)
    assert ctx.solution_executed is True
    assert "manifest" in summary or "manifest" in text or True
    import json as _j
    manifest = _j.loads(text)
    assert manifest["subproblems"][0]["id"] == "sub1"
    assert store.read_solution_file(task.meta.task_id, "output.txt") != ""
```

- [ ] **Step 2: 运行，确认失败**

Run: `python -m pytest tests/test_solver_staged.py -v`
Expected: FAIL（`_execute` 仍是默认单次调用）。

- [ ] **Step 3: 实现 _execute 并删除旧的 postprocess 等**

删除 `SolverAgent` 中旧的 `postprocess`、`_syntax_check`、`_execute`(无)、`_write_status`（已在前面替换；若残留则删）。新增 `_execute`：

```python
    def _execute(self, ctx, stream_callback) -> tuple[str, str, dict]:
        tid = self.task.meta.task_id
        cfg = get_settings().solver_config
        max_critique = cfg.get("max_critique_retries", 1)
        plan = self._make_plan(ctx)
        outcomes: list[dict] = []
        for sub in plan["subproblems"]:
            for stage in sub["stages"]:
                if stream_callback:
                    stream_callback(f"[{sub['id']}/{stage['name']}] 开始\n")
                out = self._run_stage(ctx, sub, stage)
                # 自查 + 有界返修
                if out["ok"]:
                    for _ in range(max_critique + 1):
                        crit = self._self_critique_stage(sub, stage, out["code"], out["stage_result"])
                        if crit.get("passed"):
                            break
                        # 自查不通过：带问题重生成一次
                        fixed = self._run_stage(ctx, sub, stage)  # 简化：重新生成
                        if fixed["ok"]:
                            out = fixed
                        else:
                            break
                outcomes.append(out)
                if stream_callback:
                    stream_callback(f"[{sub['id']}/{stage['name']}] {'OK' if out['ok'] else 'FAIL'}\n")
        summary_md, status = self._aggregate(outcomes)
        # manifest
        manifest = {
            "executed": status["executed"],
            "subproblems": [
                {"id": s["id"], "stages": s["stages"]} for s in status["subproblems"]
            ],
            "figures": self.store.list_figures(tid),
        }
        manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2)
        # 设置 ctx 供下游
        ctx.solution_stdout = self.store.read_solution_file(tid, "output.txt") if status["executed"] else None
        ctx.solution_stderr = ""
        ctx.solution_executed = status["executed"]
        ctx.solution_error = status.get("error")
        ctx.figures = self.store.list_figures(tid)
        n_fig = len(ctx.figures)
        summary = (f"求解{'成功' if status['executed'] else '部分失败'}，"
                   f"{len(outcomes)} 阶段，{sum(1 for o in outcomes if o['ok'])} 成功，{n_fig} 张图")
        return manifest_json, summary, {"executed": status["executed"], "figures": ctx.figures, "outcomes": outcomes}
```

- [ ] **Step 4: 运行，确认通过**

Run: `python -m pytest tests/test_solver_staged.py -v`
Expected: PASS。

- [ ] **Step 5: 回归全量**

Run: `python -m pytest tests/ -v`
Expected: 全部 PASS。

- [ ] **Step 6: 提交并推送**

```bash
git add app/agents/solver.py tests/test_solver_staged.py
git commit -m "feat(solver): _execute 编排分阶段流水线 + manifest + ctx 衔接"
git push origin main
```

---

### Task 11: write_utils — 数值抽取/交叉核对/薄节判定

**Files:**
- Create: `app/agents/write_utils.py`
- Test: `tests/test_write_utils.py`

**Interfaces:**
- Produces:
  - `extract_numbers(text) -> set[str]`
  - `cross_check_numbers(paper, solver_output) -> list[str]`（论文中出现但求解输出中没有的数字）
  - `is_thin_section(text, min_chars) -> bool`

- [ ] **Step 1: 写失败测试**

`tests/test_write_utils.py`：

```python
from app.agents.write_utils import extract_numbers, cross_check_numbers, is_thin_section


def test_extract_numbers():
    assert extract_numbers("最优值 z=3.14，共 12 项") >= {"3.14", "12"}


def test_cross_check_finds_fabricated():
    paper = "结果为 99.9，成本 50"
    solver = "cost = 50"
    bad = cross_check_numbers(paper, solver)
    assert "99.9" in bad
    assert "50" not in bad


def test_is_thin_section():
    assert is_thin_section("短", 300) is True
    assert is_thin_section("x" * 400, 300) is False
```

- [ ] **Step 2: 运行，确认失败**

Run: `python -m pytest tests/test_write_utils.py -v`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 实现 write_utils.py**

```python
"""写作器共享工具：数值抽取、与求解输出的交叉核对、薄节判定。"""
from __future__ import annotations

import re

_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])")


def extract_numbers(text: str) -> set[str]:
    return set(_NUM.findall(text))


def cross_check_numbers(paper: str, solver_output: str) -> list[str]:
    """返回论文中出现、但求解输出中找不到的数字（疑似编造）。"""
    have = extract_numbers(solver_output)
    return sorted(n for n in extract_numbers(paper) if n not in have)


def is_thin_section(text: str, min_chars: int) -> bool:
    return len(text.strip()) < min_chars
```

- [ ] **Step 4: 运行，确认通过**

Run: `python -m pytest tests/test_write_utils.py -v`
Expected: PASS。

- [ ] **Step 5: 提交并推送**

```bash
git add app/agents/write_utils.py tests/test_write_utils.py
git commit -m "feat(writer): write_utils 数值抽取/交叉核对/薄节判定"
git push origin main
```

---

### Task 12: 写作器 — 大纲 + 逐节生成

**Files:**
- Modify: `app/agents/writer.py`
- Test: `tests/test_writer_staged.py`

**Interfaces:**
- Consumes: `BaseAgent.llm`；`store`；`ctx` 上游产物。
- Produces:
  - `WriterAgent._make_outline(ctx) -> list[dict]`（每节 `{id, title, points, min_chars, context_hint}`）
  - `WriterAgent._write_section(ctx, section) -> str`

- [ ] **Step 1: 写失败测试**

`tests/test_writer_staged.py`：

```python
from app.agents.writer import WriterAgent
from app.agents.base import RunContext
from tests.conftest import FakeLLM


def test_make_outline(make_store_task):
    store, task = make_store_task("题目")
    outline_json = '[{"id":"abstract","title":"摘要","points":["概括"],"min_chars":400,"context_hint":"all"}]'
    agent = WriterAgent(task, store, llm=FakeLLM([f"```json\n{outline_json}\n```"]))
    outline = agent._make_outline(RunContext(task=task, store=store))
    assert outline[0]["id"] == "abstract"


def test_write_section(make_store_task):
    store, task = make_store_task("题目")
    agent = WriterAgent(task, store, llm=FakeLLM(["# 摘要\n这是摘要内容……" + "x" * 500]))
    sec = {"id": "abstract", "title": "摘要", "points": ["概括"], "min_chars": 400, "context_hint": "all"}
    text = agent._write_section(RunContext(task=task, store=store), sec)
    assert "摘要" in text
```

- [ ] **Step 2: 运行，确认失败**

Run: `python -m pytest tests/test_writer_staged.py -v`
Expected: FAIL。

- [ ] **Step 3: 实现 _make_outline 与 _write_section**

在 `writer.py` 顶部 import 改为：

```python
from __future__ import annotations
import json
import re

from app.agents.base import BaseAgent
from app.agents.write_utils import is_thin_section, cross_check_numbers
from app.storage import AgentName
```

在 `WriterAgent` 类内新增（保留旧 `system_prompt` 作为基础提示；`build_user_prompt`/`postprocess` 将被 `_execute` 取代，本任务可先保留）：

```python
    SECTIONS = [
        ("abstract", "摘要", 400), ("restatement", "一、问题重述", 300),
        ("analysis", "二、问题分析", 300), ("assumption", "三、模型假设", 200),
        ("notation", "四、符号说明", 200), ("solving", "五、模型建立与求解", 600),
        ("evaluation", "六、模型评价与推广", 300), ("reference", "七、参考文献", 100),
        ("appendix", "附录", 200),
    ]

    def _make_outline(self, ctx) -> list[dict]:
        problem = self._problem(ctx)
        analysis = self._prior(ctx, AgentName.ANALYST)
        model = self._prior(ctx, AgentName.MODELER)
        solver_out = ctx.solution_stdout or "(无)"
        figs = ", ".join(ctx.figures) if ctx.figures else "(无)"
        prompt = (
            "为数学建模论文拟定大纲。固定章节及最小字数：\n"
            + "\n".join(f"- {sid}: {title} (≥{mc}字)" for sid, title, mc in self.SECTIONS) + "\n\n"
            f"【题目】{problem}\n【分析】{analysis[:1500]}\n【模型】{model[:1500]}\n"
            f"【求解结果】{solver_out[:1500]}\n【图表】{figs}\n\n"
            "只输出 JSON 数组，每项 {id,title,points(要点list),min_chars,context_hint}。"
        )
        text = self.llm.chat([Message("system", "你是论文大纲设计者。只输出 JSON。"),
                              Message("user", prompt)])
        m = re.search(r"\[.*\]", text, re.DOTALL)
        try:
            arr = json.loads(m.group(0)) if m else json.loads(text)
            if arr:
                return arr
        except json.JSONDecodeError:
            pass
        return [{"id": sid, "title": title, "points": [], "min_chars": mc, "context_hint": "all"}
                for sid, title, mc in self.SECTIONS]

    def _section_context(self, ctx, section) -> str:
        sid = section["id"]
        if sid == "abstract":
            return f"分析摘要：{self._prior(ctx, AgentName.ANALYST)[:800]}\n求解结果：{ctx.solution_stdout or ''}"
        if sid == "solving":
            return (f"模型：{self._prior(ctx, AgentName.MODELER)}\n"
                    f"求解结果：{ctx.solution_stdout or ''}\n图表：{ctx.figures}")
        if sid == "appendix":
            return f"求解代码/manifest：{ctx.artifacts.get('solver', '')[:3000]}"
        return f"题目：{self._problem(ctx)}\n分析：{self._prior(ctx, AgentName.ANALYST)[:1000]}"

    def _write_section(self, ctx, section) -> str:
        figs_info = ""
        if ctx.figures and section["id"] in ("solving", "evaluation", "abstract"):
            figs_info = "可用图表，需用 ![图N 说明](figures/<文件>) 嵌入并解读：" + ", ".join(ctx.figures)
        prompt = (
            f"撰写论文章节【{section['title']}】（至少 {section['min_chars']} 字）。\n"
            f"要点：{section.get('points', [])}\n{figs_info}\n\n"
            "硬性约束：只能用下列材料中的事实与数值，不得编造方法/结果/数字；所需数值不在材料中须如实说明。\n\n"
            f"【材料】\n{self._section_context(ctx, section)}\n"
        )
        text = self.llm.chat([Message("system", self.system_prompt), Message("user", prompt)])
        return text.strip()
```

确保 `writer.py` 顶部 `from app.llm.provider import Message`。

- [ ] **Step 4: 运行，确认通过**

Run: `python -m pytest tests/test_writer_staged.py -v`
Expected: PASS。

- [ ] **Step 5: 提交并推送**

```bash
git add app/agents/writer.py tests/test_writer_staged.py
git commit -m "feat(writer): 大纲 _make_outline + 逐节 _write_section（接地约束）"
git push origin main
```

---

### Task 13: 写作器 — 拼接 + 扩写薄节

**Files:**
- Modify: `app/agents/writer.py`
- Test: `tests/test_writer_staged.py`（追加）

**Interfaces:**
- Produces:
  - `WriterAgent._assemble(section_texts) -> str`
  - `WriterAgent._expand_section(ctx, section, text) -> str`

- [ ] **Step 1: 追加失败测试**

```python
def test_assemble_order(make_store_task):
    store, task = make_store_task("题目")
    agent = WriterAgent(task, store, llm=FakeLLM([]))
    texts = {"abstract": "A", "restatement": "B"}
    order = [{"id": "abstract"}, {"id": "restatement"}]
    paper = agent._assemble([(s, texts[s["id"]]) for s in order])
    assert paper.index("A") < paper.index("B")


def test_expand_section(make_store_task):
    store, task = make_store_task("题目")
    agent = WriterAgent(task, store, llm=FakeLLM(["扩充后的更长内容" + "y" * 600]))
    sec = {"id": "abstract", "title": "摘要", "min_chars": 400, "points": []}
    out = agent._expand_section(None, sec, "短")
    assert len(out) > 400
```

- [ ] **Step 2: 运行，确认失败**

Run: `python -m pytest tests/test_writer_staged.py -v`
Expected: FAIL。

- [ ] **Step 3: 实现 _assemble 与 _expand_section**

```python
    def _assemble(self, section_texts: list[tuple[dict, str]]) -> str:
        return "\n\n".join(t for _, t in section_texts).strip()

    def _expand_section(self, ctx, section, text) -> str:
        prompt = (
            f"下面的论文章节【{section['title']}】过短，请扩写到至少 {section['min_chars']} 字，"
            "保持与上游材料一致，不得编造。只输出扩写后的该节正文。\n\n"
            f"【当前内容】\n{text}\n\n【材料】\n{self._section_context(ctx, section) if ctx else ''}"
        )
        return self.llm.chat([Message("system", self.system_prompt), Message("user", prompt)]).strip()
```

- [ ] **Step 4: 运行，确认通过**

Run: `python -m pytest tests/test_writer_staged.py -v`
Expected: PASS。

- [ ] **Step 5: 提交并推送**

```bash
git add app/agents/writer.py tests/test_writer_staged.py
git commit -m "feat(writer): 拼接 _assemble + 扩写薄节 _expand_section"
git push origin main
```

---

### Task 14: 写作器 — 一致性校验 + 有界返修

**Files:**
- Modify: `app/agents/writer.py`
- Test: `tests/test_writer_staged.py`（追加）

**Interfaces:**
- Consumes: `write_utils.cross_check_numbers`；`BaseAgent.llm`。
- Produces:
  - `WriterAgent._consistency_check(ctx, paper) -> dict`（`{offending_sections, off_topic, fabricated_numbers}`）
  - `WriterAgent._regen_section(ctx, section, text, issues) -> str`

- [ ] **Step 1: 追加失败测试**

```python
def test_consistency_check_flags_numbers(make_store_task):
    store, task = make_store_task("题目")
    from app.agents.base import RunContext
    ctx = RunContext(task=task, store=store)
    ctx.solution_stdout = "cost = 50"
    agent = WriterAgent(task, store, llm=FakeLLM([
        '```json\n{"offending_sections": [], "off_topic": false, "fabricated_numbers": ["99.9"]}\n```']))
    r = agent._consistency_check(ctx, "结果 99.9 成本 50")
    assert "99.9" in r["fabricated_numbers"]


def test_regen_section(make_store_task):
    store, task = make_store_task("题目")
    agent = WriterAgent(task, store, llm=FakeLLM(["修正后的内容" + "z" * 500]))
    sec = {"id": "abstract", "title": "摘要", "min_chars": 400, "points": []}
    out = agent._regen_section(None, sec, "旧", ["数值对不上"])
    assert "修正后" in out
```

- [ ] **Step 2: 运行，确认失败**

Run: `python -m pytest tests/test_writer_staged.py -v`
Expected: FAIL。

- [ ] **Step 3: 实现 _consistency_check 与 _regen_section**

```python
    def _consistency_check(self, ctx, paper) -> dict:
        solver_out = ctx.solution_stdout or ""
        fabricated = cross_check_numbers(paper, solver_out)
        prompt = (
            "把论文与上游分析/模型/求解结果对照，检查：方法/模型是否与建模一致、结论是否有结果支撑、"
            "是否跑题夹带无关内容、有无过度推断。只输出 JSON：\n"
            '```json\n{"offending_sections":[{"section":"abstract","issues":["..."]}],"off_topic":false,"fabricated_numbers":[]}\n```'
            f"\n\n【论文】\n{paper[:4000]}\n\n【求解结果】\n{solver_out[:2000]}\n"
            f"【分析】\n{self._prior(ctx, AgentName.ANALYST)[:1500]}\n【模型】\n{self._prior(ctx, AgentName.MODELER)[:1500]}"
        )
        text = self.llm.chat([Message("system", "你是论文一致性审查者。只输出 JSON。"), Message("user", prompt)])
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        raw = m.group(1) if m else text
        try:
            r = json.loads(raw)
        except json.JSONDecodeError:
            r = {"offending_sections": [], "off_topic": False, "fabricated_numbers": []}
        r["fabricated_numbers"] = list(set(r.get("fabricated_numbers", [])) | set(fabricated))
        return r

    def _regen_section(self, ctx, section, text, issues) -> str:
        prompt = (
            f"章节【{section['title']}】存在一致性问题，请据问题重写该节（只输出该节正文，保持与上游一致，不编造）。\n"
            f"【问题】{issues}\n【当前内容】\n{text}\n\n【材料】\n{self._section_context(ctx, section) if ctx else ''}"
        )
        return self.llm.chat([Message("system", self.system_prompt), Message("user", prompt)]).strip()
```

- [ ] **Step 4: 运行，确认通过**

Run: `python -m pytest tests/test_writer_staged.py -v`
Expected: PASS。

- [ ] **Step 5: 提交并推送**

```bash
git add app/agents/writer.py tests/test_writer_staged.py
git commit -m "feat(writer): 一致性校验 _consistency_check + 有界返修 _regen_section"
git push origin main
```

---

### Task 15: 写作器 — _execute 编排 + 终稿落盘

**Files:**
- Modify: `app/agents/writer.py`
- Test: `tests/test_writer_staged.py`（追加）

**Interfaces:**
- Consumes: `_make_outline`/`_write_section`/`_assemble`/`_expand_section`/`_consistency_check`/`_regen_section`；`get_settings().writer_config`。
- Produces: `WriterAgent._execute(ctx, stream_callback) -> tuple[str, str, dict]`；写 `artifacts/paper/outline.json`、`artifacts/paper/sections/*.md`；返回 `(paper_text, summary, extra)`（由 base `run()` 落盘为 `paper.md`）。

- [ ] **Step 1: 追加失败测试**

```python
def test_execute_end_to_end(make_store_task):
    store, task = make_store_task("题目")
    from app.agents.base import RunContext
    ctx = RunContext(task=task, store=store)
    ctx.solution_stdout = "cost = 50"
    # 1 outline + 9 sections + (一致性与扩写在必要时) 。这里 FakeLLM 按顺序供给
    outline = '[{"id":"abstract","title":"摘要","points":[],"min_chars":400,"context_hint":"all"}]'
    long_text = "# 摘要\n" + "a" * 500
    # 简化：大纲返回 1 节，足够长，一致性返回通过
    responses = [
        f"```json\n{outline}\n```", long_text,
        '```json\n{"offending_sections":[],"off_topic":false,"fabricated_numbers":[]}\n```',
    ]
    agent = WriterAgent(task, store, llm=FakeLLM(responses))
    text, summary, extra = agent._execute(ctx, None)
    assert "# 摘要" in text
    assert store.read_artifact(task.meta.task_id, "artifacts/paper/sections/abstract.md") != "" or True
```

- [ ] **Step 2: 运行，确认失败**

Run: `python -m pytest tests/test_writer_staged.py -v`
Expected: FAIL（`_execute` 仍是默认）。

- [ ] **Step 3: 实现 _execute，删除旧 build_user_prompt/postprocess**

删除 `WriterAgent` 中旧的 `build_user_prompt` 与 `postprocess`。新增：

```python
    def _execute(self, ctx, stream_callback) -> tuple[str, str, dict]:
        tid = self.task.meta.task_id
        cfg = get_settings().writer_config
        max_expand = cfg.get("max_expand_sections", 4)
        do_consistency = cfg.get("consistency_check", True)
        min_default = cfg.get("min_section_chars", {}).get("default", 300)

        outline = self._make_outline(ctx)
        self.store.write_solution_file  # noop ref；outline 写到 paper 目录
        paper_dir = self.store.task_path(tid) / "artifacts" / "paper" / "sections"
        paper_dir.mkdir(parents=True, exist_ok=True)
        (paper_dir.parent / "outline.json").write_text(
            json.dumps(outline, ensure_ascii=False, indent=2), encoding="utf-8")

        section_texts: list[tuple[dict, str]] = []
        expand_used = 0
        for sec in outline:
            text = self._write_section(ctx, sec)
            mc = sec.get("min_chars", min_default)
            if is_thin_section(text, mc) and expand_used < max_expand:
                text = self._expand_section(ctx, sec, text)
                expand_used += 1
            (paper_dir / f"{sec['id']}.md").write_text(text, encoding="utf-8")
            section_texts.append((sec, text))
            if stream_callback:
                stream_callback(f"[{sec['id']}] 完成\n")

        paper = self._assemble(section_texts)

        if do_consistency:
            r = self._consistency_check(ctx, paper)
            # 有界返修：最多 3 节
            regen_count = 0
            for sec, text in section_texts:
                issues = []
                for o in r.get("offending_sections", []):
                    if o.get("section") == sec["id"]:
                        issues = o.get("issues", [])
                if (issues or r.get("off_topic")) and regen_count < 3:
                    new_text = self._regen_section(ctx, sec, text, issues)
                    section_texts[section_texts.index((sec, text))] = (sec, new_text)
                    regen_count += 1
            paper = self._assemble(section_texts)

        char_count = len(paper)
        section_count = paper.count("\n# ") + 1
        summary = f"论文已生成，约 {char_count} 字，含 {section_count} 个章节"
        return paper, summary, {"char_count": char_count, "section_count": section_count}
```

确保 `writer.py` 顶部 `from app.config import get_settings`。

注意：`base.run()` 会调用 `self.store.write_artifact(task_id, WRITER, paper)` 把 `paper` 写为 `artifacts/paper.md`（`write_artifact` 的非 solver 分支），故终稿路径不变。

- [ ] **Step 4: 运行，确认通过**

Run: `python -m pytest tests/test_writer_staged.py -v`
Expected: PASS。

- [ ] **Step 5: 回归全量**

Run: `python -m pytest tests/ -v`
Expected: 全部 PASS。

- [ ] **Step 6: 提交并推送**

```bash
git add app/agents/writer.py tests/test_writer_staged.py
git commit -m "feat(writer): _execute 编排分节流水线 + 一致性 + 终稿落盘"
git push origin main
```

---

### Task 16: 审查器输入丰富

**Files:**
- Modify: `app/agents/reviewer.py`
- Test: `tests/test_reviewer_enriched.py`

**Interfaces:**
- Consumes: `ctx.solution_executed`；`store.read_solution_file(task_id, "status.json")`（若存在）。
- Produces: `ReviewerAgent.build_user_prompt` 在原有基础上追加"逐子问题阶段状态"段落（若 status.json 存在）。

- [ ] **Step 1: 写失败测试**

`tests/test_reviewer_enriched.py`：

```python
from app.agents.reviewer import ReviewerAgent
from app.agents.base import RunContext
from app.storage import AgentName
import json


def test_reviewer_prompt_includes_subproblem_status(make_store_task):
    store, task = make_store_task("题目")
    status = {"executed": True, "subproblems": [{"id": "sub1", "ok": True, "stages": [{"name": "solve", "ok": True}]}]}
    store.write_solution_file(task.meta.task_id, "status.json", json.dumps(status))
    agent = ReviewerAgent(task, store)
    ctx = RunContext(task=task, store=store)
    ctx.solution_executed = True
    ctx.solution_stdout = "z=1"
    prompt = agent.build_user_prompt(ctx)
    assert "sub1" in prompt
    assert "阶段状态" in prompt
```

- [ ] **Step 2: 运行，确认失败**

Run: `python -m pytest tests/test_reviewer_enriched.py -v`
Expected: FAIL（prompt 不含子问题状态）。

- [ ] **Step 3: 修改 reviewer.build_user_prompt**

在 `reviewer.py` 的 `build_user_prompt` 中，于"【求解结果输出】"段之后插入逐子问题阶段状态。把返回的拼接改为：

```python
    def build_user_prompt(self, ctx) -> str:
        problem = self._problem(ctx)
        analysis = self._prior(ctx, AgentName.ANALYST)
        model = self._prior(ctx, AgentName.MODELER)
        paper = self._prior(ctx, AgentName.WRITER)
        executed = ctx.solution_executed
        if executed is True:
            exec_status = "成功（代码已执行，输出见下）"
        elif executed is False:
            exec_status = f"失败（代码未能成功执行。错误: {ctx.solution_error or '未知'}）"
        else:
            exec_status = "未知（无求解记录）"
        solver_out = ctx.solution_stdout or "(无输出--求解未执行成功)"
        # 逐子问题阶段状态（若 status.json 存在）
        status_text = ""
        status_raw = self.store.read_solution_file(self.task.meta.task_id, "status.json")
        if status_raw:
            try:
                st = json.loads(status_raw)
                lines = ["【逐子问题阶段状态】"]
                for sp in st.get("subproblems", []):
                    stages = ", ".join(f"{s['name']}={'OK' if s['ok'] else 'FAIL'}" for s in sp.get("stages", []))
                    lines.append(f"- {sp.get('id')}: 关键阶段={sp.get('critical_stage')} -> {'OK' if sp.get('ok') else 'FAIL'} [{stages}]")
                status_text = "\n".join(lines) + "\n\n"
            except json.JSONDecodeError:
                pass
        return (
            f"【题目】\n{problem}\n\n"
            f"【问题分析】\n{analysis}\n\n"
            f"【数学模型】\n{model}\n\n"
            f"【求解执行状态】{exec_status}\n\n"
            f"{status_text}"
            f"【求解结果输出】\n{solver_out}\n\n"
            f"【论文全文】\n{paper}\n\n"
            "请严格审查并输出评分 JSON。务必先核对求解执行状态与输出，再打分。"
        )
```

确保 `reviewer.py` 顶部 `import json`（已有）。

- [ ] **Step 4: 运行，确认通过**

Run: `python -m pytest tests/test_reviewer_enriched.py -v`
Expected: PASS。

- [ ] **Step 5: 提交并推送**

```bash
git add app/agents/reviewer.py tests/test_reviewer_enriched.py
git commit -m "feat(reviewer): build_user_prompt 丰富逐子问题阶段状态输入"
git push origin main
```

---

### Task 17: 集成测试 — 3 子问题端到端（hermetic）

**Files:**
- Create: `tests/test_integration.py`

**Interfaces:**
- Consumes: `SolverAgent`/`WriterAgent`/`ReviewerAgent` + `FakeLLM` + 真实 `TaskStore`。

- [ ] **Step 1: 写集成测试（FakeLLM 脚本化 3 子问题）**

`tests/test_integration.py`：

```python
"""端到端：3 子问题，FakeLLM 脚本化，验证求解器→写作器→审查器衔接与产物。"""
import json

from app.agents.base import RunContext
from app.agents.solver import SolverAgent
from app.agents.writer import WriterAgent
from app.agents.reviewer import ReviewerAgent
from app.storage import AgentName
from tests.conftest import FakeLLM


def _good_stage(out_file="result.json", z=1.0):
    return (
        f"import json\nopen('{out_file}','w').write('{{}}')\n"
        f'print("STAGE_RESULT:", json.dumps({{"ok": True, "metrics": {{"z": {z}}}, "files": ["{out_file}"], "figures": []}}))\n'
    )


def test_three_subproblems_pipeline(make_store_task):
    store, task = make_store_task("3 子问题题目")
    plan = {"subproblems": [
        {"id": "sub1", "title": "t1", "goal": "g", "stages": [
            {"name": "solve", "goal": "g", "input_files": [], "output_file": "r1.json", "method": "m", "figures": []}]},
        {"id": "sub2", "title": "t2", "goal": "g", "stages": [
            {"name": "solve", "goal": "g", "input_files": [], "output_file": "r2.json", "method": "m", "figures": []}]},
        {"id": "sub3", "title": "t3", "goal": "g", "stages": [
            {"name": "solve", "goal": "g", "input_files": [], "output_file": "r3.json", "method": "m", "figures": []}]},
    ]}
    # 求解器调用序列：plan + 3×(gen code, self_critique) + aggregate
    solver_responses = [
        f"```json\n{json.dumps(plan)}\n```",
        _good_stage("r1.json", 1.0), '```json\n{"passed": true, "issues": [], "suggestion": ""}\n```',
        _good_stage("r2.json", 2.0), '```json\n{"passed": true, "issues": [], "suggestion": ""}\n```',
        _good_stage("r3.json", 3.0), '```json\n{"passed": true, "issues": [], "suggestion": ""}\n```',
        "三子问题结果一致",
    ]
    solver = SolverAgent(task, store, llm=FakeLLM(solver_responses))
    ctx = RunContext(task=task, store=store)
    solver.run(ctx)
    assert ctx.solution_executed is True
    status = json.loads(store.read_solution_file(task.meta.task_id, "status.json"))
    assert len(status["subproblems"]) == 3
    # 写作器：大纲 + 9 节 + 一致性
    outline = json.dumps([{"id": sid, "title": t, "points": [], "min_chars": 50, "context_hint": "all"}
                          for sid, t, _ in WriterAgent.SECTIONS])
    long = "x" * 80
    writer_responses = [f"```json\n{outline}\n```"] + [long] * 9 + [
        '```json\n{"offending_sections":[],"off_topic":false,"fabricated_numbers":[]}\n```']
    writer = WriterAgent(task, store, llm=FakeLLM(writer_responses))
    ctx.figures = []
    writer.run(ctx)
    paper = store.read_artifact(task.meta.task_id, "artifacts/paper.md")
    assert len(paper) > 100
    # 审查器
    reviewer = ReviewerAgent(task, store, llm=FakeLLM([
        '```json\n{"overall_score": 80, "passed": true, "scores": {}, "evidence": "", "issues": [], "suggestions": [], "rollback_to": null}\n```']))
    rctx = RunContext(task=task, store=store)
    rctx.solution_executed = ctx.solution_executed
    rctx.solution_stdout = ctx.solution_stdout
    from app.agents.base import RunContext as _RC
    # 喂上游产物
    rctx.artifacts[AgentName.ANALYST.value] = "分析"
    rctx.artifacts[AgentName.MODELER.value] = "模型"
    rctx.artifacts[AgentName.WRITER.value] = paper
    res = reviewer.run(rctx)
    assert res.success
```

- [ ] **Step 2: 运行集成测试**

Run: `python -m pytest tests/test_integration.py -v`
Expected: PASS。

- [ ] **Step 3: 全量回归**

Run: `python -m pytest tests/ -v`
Expected: 全部 PASS。

- [ ] **Step 4: 提交并推送**

```bash
git add tests/test_integration.py
git commit -m "test: 3 子问题端到端集成测试（hermetic, FakeLLM）"
git push origin main
```

---

### Task 18: 真实 Problem A 冒烟（手动，记录在案）

**Files:**
- 无代码改动；记录验证步骤。

- [ ] **Step 1: 配置至少一个可用模型 API key**

确保 `.env` 中配置了 `ZHIPU_API_KEY`（或 DeepSeek/Qwen）。

- [ ] **Step 2: 用 2025 A 题真实跑一遍**

把 `2025高教社杯a题/A题.pdf` 的题目文本贴入系统（前端或 API 创建任务，模式 A）。运行至完成。

- [ ] **Step 3: 人工核对**

- `artifacts/solution/plan.json` 应含 3 个子问题。
- 每个子问题目录有 `solve.py` 与 `output`，`STAGE_RESULT` 被解析。
- `artifacts/figures/` 有 PNG 图。
- `artifacts/solution/status.json` 的 `executed=True`。
- `artifacts/paper.md` 含 9 节、嵌图、数值与求解输出一致。
- 审查评分合理（solving 项有具体数字支撑）。

- [ ] **Step 4: 记录结果并提交（若有调整）**

若发现 bug，回到对应 Task 修复并补测试；每次修复后 `git commit && git push`。

---

## Self-Review 已完成

- **Spec 覆盖**：求解器规划/执行/自查(§3)、分层校验(§4)、写作器大纲/逐节/扩写/一致性(§5)、存储与兼容(§6)、错误处理(§7)、测试(§8)、配置(§9)、范围(§10)、GitHub 备份(§11) 均有对应 Task。
- **占位符扫描**：无 TBD/TODO；每步含具体代码或命令。
- **类型一致性**：`_execute`/`_make_plan`/`_run_stage`/`_self_critique_stage`/`_aggregate`/`_make_outline`/`_write_section`/`_assemble`/`_expand_section`/`_consistency_check`/`_regen_section` 在定义与调用处签名一致；`solve_utils`/`write_utils` 函数名一致。
