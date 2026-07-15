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
