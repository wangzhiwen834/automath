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
    assert "成功" in summary  # solution_executed=True 时汇总标记成功
    import json as _j
    manifest = _j.loads(text)
    assert manifest["subproblems"][0]["id"] == "sub1"
    assert store.read_solution_file(task.meta.task_id, "output.txt") != ""


def test_execute_self_critique_retry_feeds_issues(make_store_task):
    """自查不通过 -> 带提示重生成：验证 issues 通过 hint 回填到 _gen_code 调用。

    覆盖 _execute 中 `for _ in range(max_critique + 1)` 的重试分支：
    首次自查 failed -> _run_stage(..., hint="自查不通过：" + issues) -> _gen_code(hint=...)
    -> 重生成代码成功 -> 二次自查 passed。
    """
    store, task = make_store_task("题目")
    plan_json = '{"subproblems":[{"id":"sub1","title":"t","goal":"g","stages":[{"name":"solve","goal":"g","input_files":[],"output_file":"result.json","method":"m","figures":[]}]}]}'
    # FakeLLM 响应序列（顺序对应 _execute 的 LLM 调用）：
    # 0 plan
    # 1 GOOD_CODE（首次 _gen_code，无 hint）
    # 2 自查不通过（passed:false, issues=["结果方向不对"]）
    # 3 GOOD_CODE（带 hint 重生成 _gen_code —— 被断言的调用）
    # 4 自查通过（passed:true）
    # 5 aggregate 汇总
    agent = SolverAgent(task, store, llm=FakeLLM([
        f"```json\n{plan_json}\n```",
        GOOD_CODE,
        '```json\n{"passed": false, "issues": ["结果方向不对"], "suggestion": ""}\n```',
        GOOD_CODE,
        '```json\n{"passed": true, "issues": [], "suggestion": ""}\n```',
        "汇总：结果一致",
    ]))
    from app.agents.base import RunContext
    ctx = RunContext(task=task, store=store)
    agent._execute(ctx, None)
    # 重试后整体成功
    assert ctx.solution_executed is True
    # 关键：hint 真正回填到重生成 _gen_code 的 user 消息
    # 恰好一次调用的 user content 同时含 "自查不通过" 与 issue 文本
    hits = [
        c for c in agent.llm.calls
        if any(m.get("role") == "user" and "自查不通过" in m.get("content", "")
               and "结果方向不对" in m.get("content", "") for m in c)
    ]
    assert len(hits) == 1, (
        f"期望恰好 1 次带 hint 的 _gen_code 调用，实际 {len(hits)}；"
        f"calls={[[m.get('content','')[:60] for m in c] for c in agent.llm.calls]}"
    )
    # 第 4 次 LLM 调用（索引 3）正是带 hint 的重生成 _gen_code
    regen_user = agent.llm.calls[3][1]["content"]
    assert "自查不通过" in regen_user and "结果方向不对" in regen_user
