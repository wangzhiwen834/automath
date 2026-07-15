def test_fake_llm_and_store(make_store_task):
    store, task = make_store_task("题目X")
    assert task.meta.title == "测试任务"
    from tests.conftest import FakeLLM
    llm = FakeLLM(["hello"])
    from app.llm.provider import Message
    assert llm.chat([Message("user", "hi")]) == "hello"
    assert len(llm.calls) == 1
