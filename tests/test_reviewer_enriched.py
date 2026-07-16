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


def test_reviewer_prompt_malformed_status_safe(make_store_task):
    """畸形但合法 JSON 的 status.json（stage 缺 name/ok）不得抛异常，且提示词保留正常段落。

    旧代码用 s['name']/s['ok'] 直接取键 -> 此处会抛 KeyError 逃出 build_user_prompt。
    新代码用 .get() 兜底 + 宽 except，故不抛异常。
    """
    store, task = make_store_task("题目")
    # stage dict 缺失 name 与 ok 键 -> 旧代码会抛 KeyError
    malformed = {"executed": True, "subproblems": [{"id": "sub1", "stages": [{"foo": "bar"}]}]}
    store.write_solution_file(task.meta.task_id, "status.json", json.dumps(malformed))
    agent = ReviewerAgent(task, store)
    ctx = RunContext(task=task, store=store)
    ctx.solution_executed = True
    ctx.solution_stdout = "z=1"
    # 不得抛异常（旧代码在此抛 KeyError）
    prompt = agent.build_user_prompt(ctx)
    # 提示词仍包含正常段落
    assert "【题目】" in prompt
    assert "【论文全文】" in prompt


def test_reviewer_prompt_missing_status_safe(make_store_task):
    """status.json 缺失时 build_user_prompt 安全返回，提示词包含正常段落。"""
    store, task = make_store_task("题目")
    agent = ReviewerAgent(task, store)
    ctx = RunContext(task=task, store=store)
    ctx.solution_executed = True
    ctx.solution_stdout = "z=1"
    prompt = agent.build_user_prompt(ctx)
    assert "【题目】" in prompt
    assert "【论文全文】" in prompt
    assert "逐子问题阶段状态" not in prompt