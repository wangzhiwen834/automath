"""分析师 Agent：读题目 → 拆解子问题、判定题型、列假设与符号。"""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.storage import AgentName


class AnalystAgent(BaseAgent):
    name = AgentName.ANALYST

    @property
    def system_prompt(self) -> str:
        return (
            "你是一名全国大学生数学建模竞赛（高教社杯）资深指导教师。\n"
            "你的任务是对题目进行深入分析，输出结构化的分析报告，为后续建模做准备。\n\n"
            "输出要求（Markdown 格式）：\n"
            "1. **题目背景**：简述问题背景与目标。\n"
            "2. **题型判定**：判断属于优化类/评价类/统计预测类/机理建模类/图论网络类中的哪一类（可多类），并说明依据。\n"
            "3. **问题拆解**：把大问题拆成若干可独立求解的子问题，编号列出。\n"
            "4. **关键假设**：列出建模所需的合理假设（逐条编号），并说明合理性。\n"
            "5. **符号说明**：预定义会用到的关键符号及其含义。\n"
            "6. **数据需求**：指出需要哪些数据，若题目未提供需说明如何获取或构造。\n"
            "7. **建模思路**：对每个子问题给出拟采用的模型方法（如线性规划/ARIMA/AHP/微分方程等）。\n\n"
            "务必具体、可执行，避免空话。\n\n"
            "回答的第一行请用一句话概括你的核心结论（将作为摘要展示），从第二行起再展开。"
        )

    def build_user_prompt(self, ctx) -> str:
        problem = self._problem(ctx)
        if ctx.review_feedback:
            return (
                f"【原题目】\n{problem}\n\n"
                f"【审查反馈，请据此改进分析】\n{ctx.review_feedback}\n\n"
                "请重新输出完整的分析报告。"
            )
        return f"【题目】\n{problem}\n\n请对上述题目进行分析。"
