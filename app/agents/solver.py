"""求解师 Agent：分阶段生成 Python 求解代码并实际执行、硬校验、有界修复。

每个子问题的每个阶段独立走 _run_stage 流水线：
生成代码 -> 注入可复现前导 -> py_compile 语法预检 -> subprocess 执行 ->
解析 STAGE_RESULT -> 硬检查（有限值/输出文件/图表）；失败则反馈修复，最多
max_regen_per_stage 次。

⚠️ 安全：直接 subprocess 执行 LLM 生成代码，生产环境建议 Docker 沙箱。
"""
from __future__ import annotations

import json
import py_compile
import re
import subprocess
import sys

from app.agents.base import BaseAgent
from app.agents.solve_utils import (
    extract_python, parse_stage_result, inject_preamble,
    parse_plan, validate_plan, fallback_plan, run_hard_checks, extract_script_error,
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
            "- 必须严格使用原题与模型中给定的目标函数、变量、约束，禁止替换为其他函数或编造问题参数；代码实现的目标函数必须与模型完全一致，跑题将被判失败。\n"
            "- 代码控制在 200 行以内，专注数值求解。\n"
            "- 只输出一个完整的 Python 代码块（```python 围栏），代码块外不要写文字。\n"
            "- 所有字符串字面量（print/label/注释）引号必须闭合；matplotlib 的标题/标签一律用英文，避免中文编码报错。\n"
            "- 可用库：numpy, pandas, scipy, scikit-learn, networkx, sympy, pulp, matplotlib。\n"
            "- 用 print() 输出关键数值结果，每项带标注，如 print('optimal_cost =', Z)。\n"
            "- 顶部用 try/except 包住主逻辑，出错时 print 错误信息而非崩溃。\n\n"
            "图表要求（重要）：\n"
            "- plot 阶段必须绘制实际数据图（曲线/柱状/散点/热力图等），禁止用 plt.text/ax.text 只在图中间显示标量结果或 metrics 文字。若上游结果只有标量值（如最优解/最优成本），必须重新生成画图数据（如目标函数随变量变化的曲线、参数扫描、迭代收敛过程），画出有信息量的图并在图上标注关键点（最优点、极值等）。\n"
            "- 对关键结果画图，保存为 PNG 到 'artifacts/figures/' 目录（相对工作目录）。\n"
            "- 保存前先确保目录存在：import os; os.makedirs('artifacts/figures', exist_ok=True)\n"
            "- 用英文文件名，如 plt.savefig('artifacts/figures/fig1_curve.png', dpi=150, bbox_inches='tight'); plt.close()\n"
            "- 每张图画完立即 savefig 并 close，不要 plt.show()。\n"
            "- 代码末尾必须 print 一行 `STAGE_RESULT: <json>`，含 ok(布尔)/metrics(数值dict)/files(产出文件名list)/figures(图文件名list，用纯文件名不带路径)。这是框架判定阶段成败的唯一依据，缺失即判 FAIL。\n\n"
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
    # 阶段执行流水线：生成 -> 语法预检 -> 执行 -> 硬校验 -> 有界修复
    # ----------------------------------------------------------
    def _gen_code(self, ctx, sub, stage, prev_outputs: str, hint: str = "") -> str:
        problem = self._problem(ctx)
        model = self._prior(ctx, AgentName.MODELER)
        prompt = (
            f"为子问题 {sub['id']}（{sub.get('title','')}）的阶段【{stage['name']}】写 Python 代码。\n"
            f"阶段目标：{stage['goal']}\n方法：{stage['method']}\n"
            f"输入文件（在当前工作目录读取）：{stage.get('input_files')}\n"
            f"输出文件：{stage.get('output_file') or '（无）'}\n"
            f"预期图表：{stage.get('figures')}\n\n"
            f"【原题】\n{problem}\n\n"
            f"【模型核心】（目标函数/变量/约束必须严格据此实现，禁止替换或编造）\n{model[:2000]}\n\n"
            "硬性要求：\n"
            "- 代码实现的目标函数、变量、约束必须与【模型核心】完全一致，禁止替换为其他函数或编造问题参数（跑题将直接判失败）。\n"
            "- 只输出一个 ```python 代码块；matplotlib 用英文标签；画图 savefig 到 artifacts/figures/<名> 后 close。\n"
            "- 代码末尾必须 print 一行 `STAGE_RESULT: <json>`，含 ok(布尔)/metrics(数值dict)/files(产出文件名list)/figures(图文件名list)。\n"
            "- 用相对路径读写文件（当前工作目录即本子问题目录）。\n\n"
            f"上游可用信息：\n{prev_outputs}\n"
        )
        if stage.get("name") == "plot":
            prompt += (
                "【plot 阶段特别要求】必须绘制实际数据图（曲线/柱状/散点/热力图等），"
                "禁止用 plt.text/ax.text 只在图中间显示标量结果或 metrics 文字。"
                "若上游结果只有标量值，应重新计算生成画图所需的数据序列"
                "（如目标函数随变量变化的曲线、参数扫描、迭代收敛过程），"
                "画出有信息量的图并在图上标注关键点。\n"
            )
        if hint:
            prompt += f"\n\n【特别提示】{hint}"
        text = self.llm.chat([Message("system", self.system_prompt), Message("user", prompt)])
        return extract_python(text)

    def _fix_code(self, ctx, code: str, error: str) -> str:
        problem = self._problem(ctx)
        model = self._prior(ctx, AgentName.MODELER)
        msg = [
            Message("system", self.system_prompt),
            Message("user", (
                "之前代码有问题，请修复后输出完整代码（仍只一个代码块，末尾保留 STAGE_RESULT 行）。\n"
                "修复时必须确保目标函数/变量/约束与原题和模型完全一致，不得替换或编造。\n\n"
                f"【原题】\n{problem}\n\n【模型核心】\n{model[:2000]}\n\n"
                f"【原代码】\n```python\n{code}\n```\n\n【问题】\n{error}\n"
            )),
        ]
        return extract_python(self.llm.chat(msg))

    def _exec_script(self, script_path, cwd, timeout):
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)], cwd=str(cwd),
                capture_output=True, text=True, timeout=timeout,
                encoding="utf-8", errors="replace",
            )
            return proc.stdout, proc.stderr, proc.returncode == 0
        except subprocess.TimeoutExpired:
            return "", f"执行超时（>{timeout}秒）", False
        except Exception as e:
            return "", f"执行异常: {e}", False

    def _collect_figures(self, sub_dir: Path, fig_dir: Path) -> None:
        """脚本以 sub_dir 为工作目录，按 prompt 把图保存到 sub_dir/artifacts/figures/。
        把这些图复制到任务级 fig_dir，供 check_figures / list_figures / writer 统一索引。
        （脚本 cwd 是子问题目录，而框架在任务级 figures_dir 索引图，需对齐。）
        """
        import shutil
        src = sub_dir / "artifacts" / "figures"
        if not src.is_dir():
            return
        for f in src.iterdir():
            if f.is_file():
                shutil.copy2(f, fig_dir / f.name)

    def _run_stage(self, ctx, sub, stage, hint: str = "") -> dict:
        tid = self.task.meta.task_id
        cfg = get_settings().solver_config
        seed = cfg.get("preamble_seed", 42)
        max_regen = cfg.get("max_regen_per_stage", 3)
        timeout = cfg.get("stage_execution_timeout", 120)
        sub_dir = self.store.subproblem_dir(tid, sub["id"])
        fig_dir = self.store.figures_dir(tid)
        prev_outputs = ""  # 简化：阶段间靠工作目录文件交接；此处可附上阶段说明

        code = ""
        attempts = 0
        last_error = ""
        for _ in range(max_regen):
            attempts += 1
            code = self._gen_code(ctx, sub, stage, prev_outputs, hint=hint) if attempts == 1 else self._fix_code(ctx, code, last_error)
            code = inject_preamble(code, seed=seed)
            script = sub_dir / f"{stage['name']}.py"
            script.write_text(code, encoding="utf-8")
            # 语法预检
            try:
                py_compile.compile(str(script), doraise=True)
            except (py_compile.PyCompileError, SyntaxError) as e:
                last_error = f"[语法错误] {e}"
                self.store.append_log(tid, self.name.value, {"type": "syntax_error", "stage": stage["name"], "error": str(e)[:800]})
                continue
            # 执行
            stdout, stderr, ok = self._exec_script(script, sub_dir, timeout)
            if not ok:
                last_error = stderr[:1500] or "执行返回非零退出码"
                self.store.append_log(tid, self.name.value, {"type": "exec_error", "stage": stage["name"], "error": last_error})
                continue
            # 脚本以 sub_dir 为 cwd，按 prompt 把图存到了 sub_dir/artifacts/figures/；
            # 搬到任务级 fig_dir，供 check_figures / list_figures / writer 统一索引
            self._collect_figures(sub_dir, fig_dir)
            sr = parse_stage_result(stdout)
            hard_ok, errs = run_hard_checks(sr or {}, stage, sub_dir, fig_dir)
            if hard_ok:
                self.store.append_log(tid, self.name.value, {"type": "stage_done", "stage": stage["name"], "attempts": attempts})
                return {"sub_id": sub["id"], "stage": stage["name"], "ok": True,
                        "code": code, "stdout": stdout, "stage_result": sr,
                        "figures": (sr or {}).get("figures", []), "attempts": attempts, "error": ""}
            last_error = "硬检查失败: " + "; ".join(errs)
            # 附上脚本实际输出错误，帮助 _fix_code 对症修复：
            # 脚本常走 except 分支打印真实 Error/Traceback，但硬检查只看到
            # "STAGE_RESULT 缺失" 这种表面描述，导致 LLM 反复无效重试。
            script_err = extract_script_error(stdout)
            if script_err:
                last_error += " | 脚本输出: " + script_err
            self.store.append_log(tid, self.name.value, {"type": "hard_check_fail", "stage": stage["name"], "error": last_error})
        return {"sub_id": sub["id"], "stage": stage["name"], "ok": False,
                "code": code, "stdout": "", "stage_result": None,
                "figures": [], "attempts": attempts, "error": last_error}

    def _self_critique_stage(self, sub, stage, code, stage_result) -> dict:
        prompt = (
            f"审查子问题 {sub['id']} 的【{stage['name']}】阶段输出是否正确合理。\n"
            f"阶段目标：{stage.get('goal')}\n代码：\n```python\n{code}\n```\n"
            f"STAGE_RESULT：{json.dumps(stage_result, ensure_ascii=False)}\n\n"
            "判断输出是否合理回应目标、方法是否恰当、有无红旗。只输出 JSON："
            '```json\n{"passed": true, "issues": [], "suggestion": ""}\n```'
        )
        text = self.llm.chat([Message("system", "你是严谨的数值结果审查者。"), Message("user", prompt)])
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        raw = m.group(1) if m else text
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"passed": True, "issues": ["自查解析失败，默认通过"], "suggestion": ""}

    def _aggregate(self, outcomes: list[dict]) -> tuple[str, dict]:
        tid = self.task.meta.task_id
        # 关键阶段 = solve；某子问题若无 solve，取其最后一个阶段为关键
        sub_outcomes: dict[str, list[dict]] = {}
        for o in outcomes:
            sub_outcomes.setdefault(o["sub_id"], []).append(o)
        executed = True
        sub_status = []
        out_lines = []
        summary_lines = []
        for sid, lst in sub_outcomes.items():
            crit = next((o for o in lst if o["stage"] == "solve"), lst[-1])
            if not crit["ok"]:
                executed = False
            sub_status.append({"id": sid, "critical_stage": crit["stage"], "ok": crit["ok"],
                               "stages": [{"name": o["stage"], "ok": o["ok"]} for o in lst]})
            out_lines.append(f"## 子问题 {sid}")
            summary_lines.append(f"## 子问题 {sid}")
            for o in lst:
                st_line = f"- [{o['stage']}] {'OK' if o['ok'] else 'FAIL'}"
                out_lines.append(st_line)
                summary_lines.append(st_line)
                if o.get("stage_result") and o["stage_result"].get("metrics"):
                    m_line = f"  metrics: {json.dumps(o['stage_result']['metrics'], ensure_ascii=False)}"
                    out_lines.append(m_line)
                    summary_lines.append(m_line)
                if o.get("error"):
                    e_line = f"  error: {o['error']}"
                    out_lines.append(e_line)
                    summary_lines.append(e_line)
                # 完整代码写入 output.txt 供前端展示
                if o.get("code"):
                    out_lines.append(f"  代码 ({o['stage']}.py):")
                    out_lines.append(o["code"])
        output_txt = "\n".join(out_lines)
        # 汇总自查（用不含代码的摘要给 LLM，省 token）
        summary_text = "\n".join(summary_lines)
        critique = self.llm.chat([Message("system", "你是建模结果一致性审查者。"),
                                  Message("user", f"各子问题阶段输出：\n{summary_text}\n请给一句一致性/正确性结论。")])
        summary_md = f"# 求解汇总自查\n\nexecuted={executed}\n\n{critique}\n"
        status = {"executed": executed, "subproblems": sub_status,
                  "error": None if executed else "存在子问题求解关键阶段失败"}
        self.store.write_solution_output(tid, output_txt, "")
        self.store.write_solution_file(tid, "status.json", json.dumps(status, ensure_ascii=False, indent=2))
        self.store.write_solution_file(tid, "summary.md", summary_md)
        return summary_md, status

    # ----------------------------------------------------------
    # 编排：计划 -> 逐阶段执行 + 自查有界返修 -> 汇总 -> manifest + ctx
    # ----------------------------------------------------------
    def _emit_progress(self, tid: str, stream_callback, text: str) -> None:
        """推流进度并记日志，保证前端实时看到 + 刷新回放一致。"""
        if stream_callback:
            stream_callback(text)
        self.store.append_log(tid, self.name.value, {"type": "delta", "text": text})

    def _execute(self, ctx, stream_callback) -> tuple[str, str, dict]:
        """求解入口：按 solver.architecture 分派。
        - staged(默认)：规划 + 逐子问题逐阶段 + 分层校验 + 汇总自查
        - monolithic(消融基线)：一次生成单体脚本，无分阶段校验/自查
        """
        cfg = get_settings().solver_config
        if cfg.get("architecture", "staged") == "monolithic":
            return self._execute_monolithic(ctx, stream_callback)
        return self._execute_staged(ctx, stream_callback)

    # ----------------------------------------------------------
    # 单体求解(消融基线)：一次生成一个脚本，无规划/分层校验/自查
    # ----------------------------------------------------------
    def _gen_monolithic_code(self, ctx) -> str:
        return extract_python(self.llm.chat([Message("system", self.system_prompt),
                                             Message("user", self.build_user_prompt(ctx))]))

    def _execute_monolithic(self, ctx, stream_callback) -> tuple[str, str, dict]:
        tid = self.task.meta.task_id
        cfg = get_settings().solver_config
        timeout = cfg.get("stage_execution_timeout", 120)
        seed = cfg.get("preamble_seed", 42)
        sub_dir = self.store.subproblem_dir(tid, "sub1")
        fig_dir = self.store.figures_dir(tid)

        self._emit_progress(tid, stream_callback, "[monolithic] 生成单体求解脚本...\n")
        code = inject_preamble(self._gen_monolithic_code(ctx), seed=seed)
        script = sub_dir / "solve.py"
        script.write_text(code, encoding="utf-8")

        executed = False
        stdout = ""
        try:
            py_compile.compile(str(script), doraise=True)
        except (py_compile.PyCompileError, SyntaxError) as e:
            stdout = f"[语法错误] {e}"
            self.store.append_log(tid, self.name.value, {"type": "syntax_error", "error": str(e)[:800]})
        else:
            out, stderr, ok = self._exec_script(script, sub_dir, timeout)
            stdout = out
            if not ok:
                self.store.append_log(tid, self.name.value, {"type": "exec_error", "error": (stderr or "执行返回非零退出码")[:1500]})
            else:
                self._collect_figures(sub_dir, fig_dir)
                sr = parse_stage_result(stdout)
                # 有 STAGE_RESULT 时以其 ok 为准；无则按退出码 0 判定
                executed = True if sr is None else bool(sr.get("ok", True))

        self._emit_progress(tid, stream_callback, f"--- solve.py ---\n{code}\n")
        shown = stdout[:1000] + ("\n... (执行输出截断)" if len(stdout) > 1000 else "")
        self._emit_progress(tid, stream_callback, f"--- 执行输出 ---\n{shown}\n")
        self._emit_progress(tid, stream_callback, f"[monolithic] {'OK' if executed else 'FAIL'}\n\n")

        status = {"executed": executed,
                  "subproblems": [{"id": "sub1", "critical_stage": "solve", "ok": executed,
                                   "stages": [{"name": "solve", "ok": executed}]}],
                  "error": None if executed else "单体脚本执行失败"}
        self.store.write_solution_output(tid, stdout, "")
        self.store.write_solution_file(tid, "status.json", json.dumps(status, ensure_ascii=False, indent=2))
        self.store.write_solution_file(tid, "summary.md", f"# 单体求解\n\nexecuted={executed}\n")

        ctx.solution_stdout = stdout if executed else None
        ctx.solution_stderr = ""
        ctx.solution_executed = executed
        ctx.solution_error = status.get("error")
        ctx.figures = self.store.list_figures(tid)

        manifest = {"executed": executed, "subproblems": status["subproblems"], "figures": ctx.figures}
        summary = f"单体求解{'成功' if executed else '失败'}，{len(ctx.figures)} 张图"
        return json.dumps(manifest, ensure_ascii=False, indent=2), summary, {"executed": executed, "figures": ctx.figures}

    # ----------------------------------------------------------
    # 编排：计划 -> 逐阶段执行 + 自查有界返修 -> 汇总 -> manifest + ctx
    # ----------------------------------------------------------
    def _execute_staged(self, ctx, stream_callback) -> tuple[str, str, dict]:
        tid = self.task.meta.task_id
        cfg = get_settings().solver_config
        max_critique = cfg.get("max_critique_retries", 1)
        self._emit_progress(tid, stream_callback, "正在生成求解计划...\n")
        plan = self._make_plan(ctx)
        n_stages = sum(len(s["stages"]) for s in plan["subproblems"])
        self._emit_progress(tid, stream_callback,
                            f"计划：{len(plan['subproblems'])} 个子问题，{n_stages} 个阶段\n\n")
        outcomes: list[dict] = []
        for sub in plan["subproblems"]:
            for stage in sub["stages"]:
                self._emit_progress(tid, stream_callback, f"[{sub['id']}/{stage['name']}] 开始\n")
                out = self._run_stage(ctx, sub, stage)
                # 自查 + 有界返修（self_critique_enabled=false 时跳过，E3 消融用）
                if out["ok"] and cfg.get("self_critique_enabled", True):
                    for _ in range(max_critique + 1):
                        crit = self._self_critique_stage(sub, stage, out["code"], out["stage_result"])
                        if crit.get("passed"):
                            break
                        # 自查不通过：把问题回填 -> 带提示重生成一次
                        fixed = self._run_stage(
                            ctx, sub, stage,
                            hint="自查不通过：" + "; ".join(crit.get("issues", [])),
                        )
                        if fixed["ok"]:
                            out = fixed
                        else:
                            break
                outcomes.append(out)
                # 推送生成的代码 + 执行输出，让前端看到求解过程
                if out.get("code"):
                    self._emit_progress(tid, stream_callback,
                                        f"--- {sub['id']}/{stage['name']}.py ---\n{out['code']}\n")
                if out.get("stdout"):
                    stdout = out["stdout"]
                    if len(stdout) > 1000:
                        stdout = stdout[:1000] + "\n... (执行输出截断)"
                    self._emit_progress(tid, stream_callback, f"--- 执行输出 ---\n{stdout}\n")
                self._emit_progress(tid, stream_callback,
                                    f"[{sub['id']}/{stage['name']}] {'OK' if out['ok'] else 'FAIL'}\n\n")
        summary_md, status = self._aggregate(outcomes)
        # manifest
        manifest = {
            "executed": status["executed"],
            "subproblems": [
                {"id": s["id"], "stages": s["stages"]} for s in status["subproblems"]
            ],
            "figures": self.store.list_figures(tid),
        }
        manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2)
        # 设置 ctx 供下游
        ctx.solution_stdout = self.store.read_solution_file(tid, "output.txt") if status["executed"] else None
        ctx.solution_stderr = ""
        ctx.solution_executed = status["executed"]
        ctx.solution_error = status.get("error")
        ctx.figures = self.store.list_figures(tid)
        n_fig = len(ctx.figures)
        summary = (f"求解{'成功' if status['executed'] else '部分失败'}，"
                   f"{len(outcomes)} 阶段，{sum(1 for o in outcomes if o['ok'])} 成功，{n_fig} 张图")
        return manifest_json, summary, {"executed": status["executed"], "figures": ctx.figures, "outcomes": outcomes}
