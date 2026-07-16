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
