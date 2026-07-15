"""求解器共享工具：代码提取、可复现前导注入、STAGE_RESULT 解析、硬检查、计划 schema。"""
from __future__ import annotations

import json
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