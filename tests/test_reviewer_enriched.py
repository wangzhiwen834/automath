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