"""端到端：3 子问题，FakeLLM 脚本化，验证求解器->写作器->审查器衔接与产物。"""
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
    # 喂上游产物
    rctx.artifacts[AgentName.ANALYST.value] = "分析"
    rctx.artifacts[AgentName.MODELER.value] = "模型"
    rctx.artifacts[AgentName.WRITER.value] = paper
    res = reviewer.run(rctx)
    assert res.success
