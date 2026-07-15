from app.agents.solve_utils import extract_python, parse_stage_result, inject_preamble
import json
from app.agents.solve_utils import (
    parse_plan, validate_plan, fallback_plan,
    check_finite, check_figures, run_hard_checks,
)


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


def test_parse_plan_from_fence():
    text = '前缀\n```json\n{"subproblems": []}\n```'
    assert parse_plan(text) == {"subproblems": []}


def test_validate_plan_good():
    plan = {"subproblems": [{"id": "sub1", "title": "t", "goal": "g",
            "stages": [{"name": "solve", "goal": "g", "input_files": [],
                        "output_file": "r.json", "method": "m", "figures": []}]}]}
    ok, errs = validate_plan(plan)
    assert ok and errs == []


def test_validate_plan_no_subproblems():
    ok, errs = validate_plan({"subproblems": []})
    assert not ok and any("子问题" in e for e in errs)


def test_validate_plan_stage_no_solve():
    plan = {"subproblems": [{"id": "s1", "title": "t", "goal": "g",
            "stages": [{"name": "data", "goal": "g", "input_files": [],
                        "output_file": "d.csv", "method": "m", "figures": []}]}]}
    ok, errs = validate_plan(plan)
    # 至少要有一个 solve 阶段
    assert not ok


def test_fallback_plan_shape():
    p = fallback_plan()
    assert len(p["subproblems"]) == 1
    names = [s["name"] for s in p["subproblems"][0]["stages"]]
    assert "solve" in names


def test_check_finite():
    ok, _ = check_finite({"z": 3.5})
    assert ok
    ok, msg = check_finite({"z": float("nan")})
    assert not ok and "有限" in msg


def test_check_figures(tmp_path):
    d = tmp_path / "figs"
    d.mkdir()
    (d / "ok.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"x")
    ok, _ = check_figures(["ok.png"], d)
    assert ok
    ok, msg = check_figures(["missing.png"], d)
    assert not ok


def test_run_hard_checks_pass(tmp_path):
    sub_dir = tmp_path / "sub1"
    sub_dir.mkdir()
    fig_dir = tmp_path / "figs"
    fig_dir.mkdir()
    (sub_dir / "result.json").write_text("{}")
    sr = {"ok": True, "metrics": {"z": 1.0}, "files": ["result.json"], "figures": []}
    stage = {"name": "solve", "output_file": "result.json", "figures": []}
    ok, errs = run_hard_checks(sr, stage, sub_dir, fig_dir)
    assert ok, errs