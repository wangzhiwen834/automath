"""求解器共享工具：代码提取、可复现前导注入、STAGE_RESULT 解析、硬检查、计划 schema。"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path


def extract_python(text: str) -> str:
    """从 Markdown 提取 Python 代码块；无围栏则视为全是代码。"""
    fences = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if fences:
        return "\n\n".join(fences).strip()
    return text.strip()


def parse_stage_result(stdout: str) -> dict | None:
    """从 stdout 中解析 `STAGE_RESULT: {json}` 行。找不到返回 None。"""
    m = re.search(r"STAGE_RESULT\s*:\s*(\{.*\})", stdout, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def inject_preamble(code: str, seed: int = 42) -> str:
    """注入可复现前导：随机种子 + matplotlib Agg 后端。"""
    preamble = (
        "import random\n"
        "import numpy as np\n"
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        f"random.seed({seed})\n"
        f"np.random.seed({seed})\n"
        "import os\n"
        "os.makedirs('artifacts/figures', exist_ok=True)\n\n"
    )
    return preamble + code


def parse_plan(text: str) -> dict | None:
    """从回答中提取计划 JSON。"""
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    raw = m.group(1) if m else text
    if not m:
        s, e = raw.find("{"), raw.rfind("}")
        if s != -1 and e != -1:
            raw = raw[s:e + 1]
        else:
            return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def validate_plan(plan: dict) -> tuple[bool, list[str]]:
    """校验计划结构。"""
    errs: list[str] = []
    subs = plan.get("subproblems")
    if not isinstance(subs, list) or not subs:
        return False, ["计划缺少 subproblems（至少一个子问题）"]
    for i, sub in enumerate(subs):
        if not sub.get("id"):
            errs.append(f"第 {i+1} 个子问题缺少 id")
        if not sub.get("stages"):
            errs.append(f"子问题 {sub.get('id', i+1)} 缺少 stages")
            continue
        names = [s.get("name") for s in sub["stages"]]
        if "solve" not in names:
            errs.append(f"子问题 {sub.get('id', i+1)} 必须含一个 solve 阶段")
        for s in sub["stages"]:
            for k in ("name", "goal", "method"):
                if not s.get(k):
                    errs.append(f"子问题 {sub.get('id', i+1)} 某阶段缺少 {k}")
    return (len(errs) == 0), errs


def fallback_plan() -> dict:
    """退化计划：单子问题、单 solve 阶段（兼容旧行为）。"""
    return {
        "subproblems": [{
            "id": "sub1", "title": "整体求解", "goal": "求解整个问题",
            "stages": [
                {"name": "solve", "goal": "求解并输出关键结果",
                 "input_files": [], "output_file": "result.json",
                 "method": "数值求解", "figures": [], "expected_range": None},
                {"name": "plot", "goal": "对关键结果画图",
                 "input_files": ["result.json"], "output_file": "",
                 "method": "matplotlib", "figures": ["sub1_1_fig.png"], "expected_range": None},
            ],
        }]
    }


# ---------- 硬检查 ----------

def _is_finite(v) -> bool:
    if isinstance(v, bool):
        return True
    if isinstance(v, (int, float)):
        return math.isfinite(v)
    return True


def check_finite(metrics: dict) -> tuple[bool, str]:
    for k, v in metrics.items():
        if not _is_finite(v):
            return False, f"指标 {k} 非有限值（NaN/Inf）"
    return True, ""


def check_figures(figures: list[str], figures_dir: Path) -> tuple[bool, str]:
    for f in figures:
        p = figures_dir / f
        if not p.exists() or p.stat().st_size == 0:
            return False, f"图表文件缺失或为空: {f}"
        with open(p, "rb") as fh:
            if not fh.read(8).startswith(b"\x89PNG"):
                return False, f"图表非 PNG: {f}"
    return True, ""


def check_expected_range(value, rng) -> tuple[bool, str]:
    if rng is None or value is None:
        return True, ""
    try:
        lo, hi = rng
        if not (lo <= float(value) <= hi):
            return False, f"值 {value} 不在预期范围 [{lo}, {hi}]"
    except (TypeError, ValueError):
        return True, ""
    return True, ""


def run_hard_checks(stage_result: dict, stage: dict, sub_dir: Path, figures_dir: Path) -> tuple[bool, list[str]]:
    """对一个阶段的 STAGE_RESULT 做硬检查。"""
    errs: list[str] = []
    if not stage_result or not stage_result.get("ok"):
        errs.append("STAGE_RESULT 缺失或 ok!=true")
    metrics = stage_result.get("metrics", {}) if stage_result else {}
    ok, msg = check_finite(metrics)
    if not ok:
        errs.append(msg)
    out_file = stage.get("output_file")
    if out_file:
        if not (sub_dir / out_file).exists():
            errs.append(f"输出文件缺失: {out_file}")
    figs = stage_result.get("figures", []) if stage_result else []
    ok, msg = check_figures(figs, figures_dir)
    if not ok:
        errs.append(msg)
    return (len(errs) == 0), errs