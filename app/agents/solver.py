"""求解师 Agent：根据模型 → 生成 Python 求解代码 → 实际执行 → 捕获结果。

加固点:
1. py_compile 语法预检：语法错直接反馈修复（不走完整执行，省时）。
2. 执行 → 失败 → 反馈修复，循环最多 3 次。
3. 写 status.json 记录执行是否成功，供审查师硬否决。
4. prompt 限制代码长度、禁用画图、避免中文字符串编码问题。

⚠️ 安全：直接 subprocess 执行 LLM 生成代码，生产环境建议 Docker 沙箱。
"""
from __future__ import annotations

import json
import py_compile
import subprocess
import sys

from app.agents.base import BaseAgent
from app.agents.solve_utils import (
    extract_python, parse_stage_result, inject_preamble,
    parse_plan, validate_plan, fallback_plan, run_hard_checks,
)
from app.config import get_settings
from app.llm.provider import Message
from app.storage import AgentName


class SolverAgent(BaseAgent):
    name = AgentName.SOLVER

    PLAN_PROMPT = (
        "你是数学建模求解规划专家。根据题目、分析、模型，把问题拆成可独立求解的子问题，"
        "并为每个子问题规划有序阶段（从 数据/建模/求解/分析/画图 中按需选取，至少含 solve，推荐含 plot）。\n\n"
        "只输出一个 JSON 代码块，结构：\n"
        "```json\n"
        '{"subproblems":[{"id":"sub1","title":"","goal":"","stages":['
        '{"name":"data|model|solve|analyze|plot","goal":"","input_files":[],"output_file":"","method":"","figures":[],"expected_range":null}]}]}\n'
        "```\n"
        "要求：id 用 sub1/sub2…；input_files 引用前一阶段的 output_file 或上传数据 data/<名>；"
        "画图阶段把 figures 列出（文件名形如 sub1_1_curve.png）。只输出 JSON。"
    )

    def _make_plan(self, ctx) -> dict:
        problem = self._problem(ctx) if ctx else ""
        analysis = self._prior(ctx, AgentName.ANALYST) if ctx else ""
        model = self._prior(ctx, AgentName.MODELER) if ctx else ""
        data_info = ""
        if ctx and ctx.data_files:
            data_info = "已上传数据：data/" + ", data/".join(ctx.data_files)
        messages = [
            Message("system", self.PLAN_PROMPT),
            Message("user", f"【题目】\n{problem}\n\n【分析】\n{analysis}\n\n【模型】\n{model}\n\n{data_info}\n请输出求解计划。"),
        ]
        text = self.llm.chat(messages)
        plan = parse_plan(text)
        if plan:
            ok, errs = validate_plan(plan)
            if ok:
                self.store.write_solution_file(self.task.meta.task_id, "plan.json",
                                               json.dumps(plan, ensure_ascii=False, indent=2))
                self.store.append_log(self.task.meta.task_id, self.name.value,
                                      {"type": "plan", "subproblems": len(plan["subproblems"])})
                return plan
        # 退化
        fb = fallback_plan()
        self.store.write_solution_file(self.task.meta.task_id, "plan.json",
                                       json.dumps(fb, ensure_ascii=False, indent=2))
        self.store.append_log(self.task.meta.task_id, self.name.value,
                              {"type": "plan_fallback", "errors": errs if plan else "parse_failed"})
        return fb

    @property
    def system_prompt(self) -> str:
        return (
            "你是一名数值计算与 Python 编程专家。\n"
            "根据数学模型编写可运行的 Python 求解代码。\n\n"
            "硬性要求（违反将导致执行失败）：\n"
            "- 代码控制在 200 行以内，专注数值求解。\n"
            "- 只输出一个完整的 Python 代码块（```python 围栏），代码块外不要写文字。\n"
            "- 所有字符串字面量（print/label/注释）引号必须闭合；matplotlib 的标题/标签一律用英文，避免中文编码报错。\n"
            "- 可用库：numpy, pandas, scipy, scikit-learn, networkx, sympy, pulp, matplotlib。\n"
            "- 用 print() 输出关键数值结果，每项带标注，如 print('optimal_cost =', Z)。\n"
            "- 顶部用 try/except 包住主逻辑，出错时 print 错误信息而非崩溃。\n\n"
            "图表要求（重要）：\n"
            "- 对关键结果画图，保存为 PNG 到 'artifacts/figures/' 目录（相对工作目录）。\n"
            "- 保存前先确保目录存在：import os; os.makedirs('artifacts/figures', exist_ok=True)\n"
            "- 用英文文件名，如 plt.savefig('artifacts/figures/fig1_curve.png', dpi=150, bbox_inches='tight'); plt.close()\n"
            "- 每张图画完立即 savefig 并 close，不要 plt.show()。\n"
            "- 代码末尾用 print('FIGURES:', <生成的文件名列表>) 列出所有生成的图。\n\n"
            "数据文件：若提供了上传数据文件，用相对路径 'data/<文件名>' 读取（如 pd.read_csv('data/sales.csv')）。"
        )

    def build_user_prompt(self, ctx) -> str:
        model = self._prior(ctx, AgentName.MODELER)
        analysis = self._prior(ctx, AgentName.ANALYST)
        data_info = ""
        if ctx.data_files:
            data_info = "【已上传数据文件】（用相对路径 data/<文件名> 读取）：\n" + "\n".join(ctx.data_files) + "\n\n"
        prefix = ""
        if ctx.review_feedback:
            prefix = f"【审查反馈，请据此重写求解代码】\n{ctx.review_feedback}\n\n"
        return (
            f"{prefix}{data_info}"
            f"【问题分析】\n{analysis}\n\n"
            f"【数学模型】\n{model}\n\n"
            "请编写 Python 求解代码（含图表生成）。"
        )

    # ----------------------------------------------------------
    # 后处理：提取 → 语法预检 → 执行 → 失败修复，写 status.json
    # ----------------------------------------------------------
    def postprocess(self, ctx, text: str) -> tuple[str, str, dict]:
        code = extract_python(text)
        max_attempts = 3
        stdout, stderr, executed = "", "", False
        fix_attempts = 0
        last_error = ""

        for attempt in range(max_attempts):
            # 1) 语法预检
            syn_ok, syn_err = self._syntax_check(code)
            if not syn_ok:
                last_error = f"[语法错误] {syn_err}"
                self.store.append_log(
                    self.task.meta.task_id, self.name.value,
                    {"type": "syntax_error", "attempt": attempt + 1, "error": syn_err[:1500]},
                )
                if attempt < max_attempts - 1:
                    fix_attempts += 1
                    code = self._fix_code(code, last_error)
                    continue
                break

            # 2) 执行
            stdout, stderr, executed = self._execute(code)
            if executed:
                last_error = ""
                break
            last_error = stderr[:2000] or "执行返回非零退出码"
            self.store.append_log(
                self.task.meta.task_id, self.name.value,
                {"type": "exec_error", "attempt": attempt + 1, "error": last_error},
            )
            if attempt < max_attempts - 1:
                fix_attempts += 1
                code = self._fix_code(code, last_error)

        # 落盘代码、输出、状态
        self.store.write_artifact(self.task.meta.task_id, self.name, code)
        self.store.write_solution_output(self.task.meta.task_id, stdout, stderr)
        self._write_status(executed, last_error)

        # 塞进 ctx 供下游
        ctx.solution_stdout = stdout if executed else None
        ctx.solution_stderr = stderr
        ctx.solution_executed = executed
        ctx.solution_error = last_error if not executed else None
        ctx.figures = self.store.list_figures(self.task.meta.task_id)

        if executed:
            n_fig = len(ctx.figures)
            summary = f"求解执行成功，输出 {len(stdout.splitlines())} 行结果，生成 {n_fig} 张图"
        else:
            summary = f"求解代码执行失败（{fix_attempts} 次修复未成功）"
        return code, summary, {
            "stdout": stdout, "stderr": stderr,
            "executed": executed, "fix_attempts": fix_attempts,
            "error": last_error, "figures": ctx.figures,
        }

    # ----------------------------------------------------------
    def _syntax_check(self, code: str) -> tuple[bool, str | None]:
        sol_dir = self.store.solution_dir(self.task.meta.task_id)
        script = sol_dir / "solve.py"
        with open(script, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            py_compile.compile(str(script), doraise=True)
            return True, None
        except py_compile.PyCompileError as e:
            return False, str(e)
        except SyntaxError as e:
            return False, f"{e.msg} (line {e.lineno})"

    def _execute(self, code: str) -> tuple[str, str, bool]:
        # cwd = 任务根目录，使 'artifacts/figures/' 与 'data/' 相对路径生效
        sol_dir = self.store.solution_dir(self.task.meta.task_id)
        script = sol_dir / "solve.py"
        with open(script, "w", encoding="utf-8") as f:
            f.write(code)
        cwd = str(self.store.task_path(self.task.meta.task_id))
        timeout = get_settings().solver_config.get("execution_timeout", 60)
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                cwd=cwd,
                capture_output=True, text=True, timeout=timeout,
                encoding="utf-8", errors="replace",
            )
            return proc.stdout, proc.stderr, proc.returncode == 0
        except subprocess.TimeoutExpired:
            return "", f"执行超时（>{timeout}秒）", False
        except Exception as e:
            return "", f"执行异常: {e}", False

    def _write_status(self, executed: bool, error: str) -> None:
        sol_dir = self.store.solution_dir(self.task.meta.task_id)
        status = {"executed": executed, "error": error or None}
        with open(sol_dir / "status.json", "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, indent=2)

    def _fix_code(self, code: str, error: str) -> str:
        messages = [
            Message("system", self.system_prompt),
            Message("user", (
                "你之前生成的代码运行报错了，请修复后重新输出完整代码（仍只输出一个代码块）。\n\n"
                f"【原代码】\n```python\n{code}\n```\n\n"
                f"【报错信息】\n{error}\n\n"
                "请输出修复后的完整 Python 代码。"
            )),
        ]
        resp = self.llm.chat(messages)
        return extract_python(resp)
