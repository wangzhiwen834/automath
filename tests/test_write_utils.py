from app.agents.write_utils import extract_numbers, cross_check_numbers, is_thin_section


def test_extract_numbers():
    assert extract_numbers("最优值 z=3.14，共 12 项") >= {"3.14", "12"}


def test_cross_check_finds_fabricated():
    paper = "结果为 99.9，成本 50"
    solver = "cost = 50"
    bad = cross_check_numbers(paper, solver)
    assert "99.9" in bad
    assert "50" not in bad


def test_is_thin_section():
    assert is_thin_section("短", 300) is True
    assert is_thin_section("x" * 400, 300) is False