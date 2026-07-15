"""写作器共享工具：数值抽取、与求解输出的交叉核对、薄节判定。"""
from __future__ import annotations

import re

_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])")


def extract_numbers(text: str) -> set[str]:
    return set(_NUM.findall(text))


def cross_check_numbers(paper: str, solver_output: str) -> list[str]:
    """返回论文中出现、但求解输出中找不到的数字（疑似编造）。"""
    have = extract_numbers(solver_output)
    return sorted(n for n in extract_numbers(paper) if n not in have)


def is_thin_section(text: str, min_chars: int) -> bool:
    return len(text.strip()) < min_chars