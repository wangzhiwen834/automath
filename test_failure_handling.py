#!/usr/bin/env python3
"""验证 _handle_orchestrator_failure：orch.run() 早期异常时 task 应落盘 FAILED，而非卡在 RUNNING。
不依赖 LLM/API，直接单元测试。"""
import sys
import io
import traceback

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, "app")

from app.storage import get_store, AGENT_ORDER, TaskStatus, AgentStatus
from app.server.runner import TaskRunner


def make_zombie_task():
    """创建一个模拟"卡死"的任务：status=RUNNING，analyst=RUNNING，无日志。"""
    store = get_store()
    task = store.create_task(
        title="僵尸任务测试",
        problem_text="测试用题目",
        mode="auto",
        problem_type="optimization",
        agent_models={a.value: "glm-4.7" for a in AGENT_ORDER},
    )
    # 模拟 orchestrator 已 save 了 RUNNING，但 agent.run() 还没真正开始
    task = store.load(task.meta.task_id)
    task.state.status = TaskStatus.RUNNING
    task.state.current_agent = AGENT_ORDER[0]
    analyst_rec = task.state.get_agent(AGENT_ORDER[0])
    analyst_rec.status = AgentStatus.RUNNING
    store.save(task)
    return task.meta.task_id


def test_handle_failure():
    task_id = make_zombie_task()
    store = get_store()

    before = store.load(task_id)
    print(f"[BEFORE] status={before.state.status}, analyst={before.state.agents['analyst'].status}")

    # 模拟 orch.run() 早期抛异常（_build_ctx / get_llm 等）
    runner = TaskRunner.get()
    runner._handle_orchestrator_failure(task_id, RuntimeError("模拟 _build_ctx 读文件失败"))

    after = store.load(task_id)
    print(f"[AFTER]  status={after.state.status}, analyst={after.state.agents['analyst'].status}")
    print(f"[AFTER]  analyst.error={after.state.agents['analyst'].error!r}")

    ok = (
        after.state.status == TaskStatus.FAILED
        and after.state.agents["analyst"].status == AgentStatus.FAILED
        and "模拟" in (after.state.agents["analyst"].error or "")
    )
    print(f"[RESULT] {'PASS' if ok else 'FAIL'}")
    return ok


def test_target_full_path():
    """完整路径：monkeypatch orch.run 抛异常，确认 target() 落盘 FAILED。"""
    task_id = make_zombie_task()
    store = get_store()
    runner = TaskRunner.get()

    # monkeypatch Orchestrator.run 抛异常
    import app.orchestrator as orch_mod
    original_run = orch_mod.Orchestrator.run

    def boom(self):
        raise RuntimeError("模拟 orch.run 早期异常（agent.run 之前）")

    orch_mod.Orchestrator.run = boom
    try:
        started = runner.start(task_id)
        print(f"[SPAWN] thread started={started}")
        # 等线程结束
        t = runner._threads.get(task_id)
        if t:
            t.join(timeout=10)
        after = store.load(task_id)
        print(f"[AFTER]  status={after.state.status}")
        ok = after.state.status == TaskStatus.FAILED
        print(f"[RESULT] {'PASS' if ok else 'FAIL'}")
        return ok
    finally:
        orch_mod.Orchestrator.run = original_run


def test_constructor_failure():
    """Orchestrator 构造抛异常（在 try 之内）也应落盘 FAILED，而非静默卡死。"""
    task_id = make_zombie_task()
    store = get_store()
    runner = TaskRunner.get()

    import app.orchestrator as orch_mod
    original_init = orch_mod.Orchestrator.__init__

    def bad_init(self, *a, **kw):
        raise RuntimeError("模拟 Orchestrator 构造失败（如 get_store 异常）")

    orch_mod.Orchestrator.__init__ = bad_init
    try:
        started = runner.start(task_id)
        print(f"[SPAWN] thread started={started}")
        t = runner._threads.get(task_id)
        if t:
            t.join(timeout=10)
        after = store.load(task_id)
        print(f"[AFTER]  status={after.state.status}")
        ok = after.state.status == TaskStatus.FAILED
        print(f"[RESULT] {'PASS' if ok else 'FAIL'}")
        return ok
    finally:
        orch_mod.Orchestrator.__init__ = original_init


if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: _handle_orchestrator_failure (unit)")
    print("=" * 60)
    r1 = test_handle_failure()

    print()
    print("=" * 60)
    print("Test 2: target() full path (monkeypatch orch.run)")
    print("=" * 60)
    r2 = test_target_full_path()

    print()
    print("=" * 60)
    print("Test 3: Orchestrator constructor failure")
    print("=" * 60)
    r3 = test_constructor_failure()

    print()
    print("=" * 60)
    print(f"OVERALL: {'PASS' if (r1 and r2 and r3) else 'FAIL'}")
    print("=" * 60)
    sys.exit(0 if (r1 and r2 and r3) else 1)