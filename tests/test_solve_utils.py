from app.agents.solve_utils import extract_python, parse_stage_result, inject_preamble


def test_extract_python_from_fence():
    text = "说明\n```python\nprint(1)\n```\n结尾"
    assert extract_python(text) == "print(1)"


def test_extract_python_no_fence():
    assert extract_python("print(2)") == "print(2)"


def test_parse_stage_result_ok():
    out = "一些输出\nSTAGE_RESULT: {\"ok\": true, \"metrics\": {\"z\": 3.5}, \"files\": [], \"figures\": []}\n尾"
    r = parse_stage_result(out)
    assert r is not None and r["ok"] is True and r["metrics"]["z"] == 3.5


def test_parse_stage_result_missing():
    assert parse_stage_result("无标记") is None


def test_inject_preamble_has_seed_and_agg():
    code = "print(1)"
    pre = inject_preamble(code, seed=42)
    assert "random.seed(42)" in pre
    assert "matplotlib.use('Agg')" in pre
    assert pre.endswith("print(1)")