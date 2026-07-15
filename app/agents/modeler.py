"""建模师 Agent：根据分析报告 → 建立数学模型（变量、目标函数、约束、求解思路）。"""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.storage import AgentName


class ModelerAgent(BaseAgent):
    name = AgentName.MODELER

    @property
    def system_prompt(self) -> str:
        return (
            "你是一名数学建模专家，擅长将分析结论转化为严格的数学模型。\n"
            "你的任务是基于问题分析报告，建立完整、可求解的数学模型。\n\n"
            "输出要求（Markdown 格式）：\n"
            "1. **模型假设**：复用并补充分析阶段假设。\n"
            "2. **符号说明**：完整列出模型用到的所有符号、单位、含义。\n"
            "3. **模型建立**：对每个子问题，分别给出：\n"
            "   - 决策变量/状态变量\n"
            "   - 目标函数（明确的数学表达式）\n"
            "   - 约束条件（明确的数学表达式）\n"
            "   - 模型类型（如 MILP / NLP / ODE / ARIMA / AHP 等）\n"
            "4. **求解思路**：说明用什么算法/方法求解，推荐 Python 库（如 pulp/scipy/sklearn/statsmodels）。\n"
            "5. **模型评价**：模型的优缺点、适用性说明。\n\n"
            "数学公式用 LaTeX 行内 $...$ 或行间 $$...$$ 表示。务必给出可被代码直接实现的明确表达。\n\n"
            "回答的第一行请用一句话概括你建立的模型类型与核心思路（将作为摘要展示）。"
        )

    def build_user_prompt(self, ctx) -> str:
        problem = self._problem(ctx)
        analysis = self._prior(ctx, AgentName.ANALYST)
        prefix = ""
        if ctx.review_feedback:
            prefix = f"【审查反馈，请据此改进模型】\n{ctx.review_feedback}\n\n"
        return (
            f"{prefix}"
            f"【题目】\n{problem}\n\n"
            f"【问题分析报告】\n{analysis}\n\n"
            "请基于上述分析建立数学模型。"
        )
