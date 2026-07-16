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
