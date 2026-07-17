"""总结师 Agent：综合分析+模型+求解结果，提炼事实清单(facts.md)，作为写作权威数据源。

位于 solver 与 writer 之间。把散落的求解结果(含代码的 output.txt)提炼成干净的
结构化事实，并做跨环节一致性对齐，避免 writer 凭空编造数值、极大极小混淆等。
"""
from __future__ import annotations

import json

from app.agents.base import BaseAgent
from app.storage import AgentName


class SummarizerAgent(BaseAgent):
    name = AgentName.SUMMARIZER

    @property
    def system_prompt(self) -> str:
        return (
            "你是数学建模事实清单总结师。任务：综合问题分析、数学模型、求解结果，"
            "提炼一份严谨的事实清单，作为论文写作的唯一权威数据源。\n\n"
            "关键要求（违反将直接导致论文编造，必须严格执行）：\n"
            "- 所有数值必须来自【求解结果输出】中的 STAGE_RESULT metrics，禁止编造或推测。"
            "材料中没有的数值如实标注\"未求解/无数据\"，绝不自行补充。\n"
            "- 忽略求解代码本身，只提炼结果数值与结论。\n"
            "- 对每个子问题提炼：所用方法、关键数值表（变量=值，带单位）、驻点/极值/最优解、最终结论。\n"
            "- 极大值/极小值必须严格区分，依据求解结果中的实际数值判定，不得混淆。\n"
            "- 列出所有已生成的图表（文件名 + 内容说明）；未生成的图不得列入，更不得在结论中论述不存在的图。\n"
            "- 跨环节一致性检查：明确标注 analyst/modeler/solver 之间的矛盾"
            "（如驻点数量不一致、极大极小混淆、函数形式不符、结论无求解支撑），用 ⚠️ 标记。\n\n"
            "输出 Markdown 格式的事实清单，包含以下一级标题：\n"
            "# 一、问题核心\n# 二、模型核心（变量/目标函数/约束/解法）\n"
            "# 三、求解结果（按子问题，含数值表）\n# 四、图表清单\n# 五、一致性标注\n\n"
            "回答的第一行请用一句话概括核心结论（将作为摘要展示），从第二行起再展开。"
        )

    def build_user_prompt(self, ctx) -> str:
        tid = self.task.meta.task_id
        analysis = self._prior(ctx, AgentName.ANALYST)
        model = self._prior(ctx, AgentName.MODELER)
        solver_out = ctx.solution_stdout or "(无求解输出)"
        figures = ctx.figures or []
        # 逐子问题阶段状态（让总结师知道哪些阶段真的跑通了）
        status_text = ""
        status_raw = self.store.read_solution_file(tid, "status.json")
        if status_raw:
            try:
                st = json.loads(status_raw)
                lines = ["【逐子问题阶段状态】"]
                for sp in st.get("subproblems", []):
                    stages = ", ".join(
                        f"{s.get('name', '?')}={'OK' if s.get('ok') else 'FAIL'}"
                        for s in sp.get("stages", []))
                    lines.append(f"- {sp.get('id')}: 关键阶段={sp.get('critical_stage')} -> "
                                 f"{'OK' if sp.get('ok') else 'FAIL'} [{stages}]")
                status_text = "\n".join(lines) + "\n\n"
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
        fig_text = "无" if not figures else ", ".join(figures)
        return (
            f"【问题分析】\n{analysis}\n\n"
            f"【数学模型】\n{model}\n\n"
            f"{status_text}"
            f"【求解结果输出】（注意：忽略其中代码，只提炼 STAGE_RESULT 的 metrics 数值与结论）\n{solver_out}\n\n"
            f"【已生成图表】{fig_text}\n\n"
            "请提炼事实清单。"
        )
