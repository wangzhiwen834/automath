"""求解器分阶段流水线测试（Task 7+）。"""
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
