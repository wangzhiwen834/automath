"""写作师 Agent：综合所有上游产物 → 生成完整竞赛论文 Markdown。"""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.storage import AgentName


class WriterAgent(BaseAgent):
    name = AgentName.WRITER

    @property
    def system_prompt(self) -> str:
        return (
            "你是一名数学建模竞赛论文写作专家，熟悉高教社杯论文格式与评分要点。\n"
            "你的任务是综合分析、模型、求解结果，撰写一篇结构完整的竞赛论文。\n\n"
            "论文必须包含以下章节（Markdown 一级标题）：\n"
            "# 摘要（含关键词，500字左右，概括问题、方法、结果、结论）\n"
            "# 一、问题重述\n"
            "# 二、问题分析\n"
            "# 三、模型假设\n"
            "# 四、符号说明\n"
            "# 五、模型建立与求解（分子问题，含公式、算法步骤、结果）\n"
            "# 六、模型评价与推广\n"
            "# 七、参考文献\n"
            "# 附录（关键代码）\n\n"
            "要求：\n"
            "- 公式用 LaTeX；数值结果要具体（来自求解结果）。\n"
            "- 摘要要突出创新点与主要结论。\n"
            "- 语言学术、严谨、流畅。\n\n"
            "图表：若提供了【生成的图表】列表，必须在论文相应位置用 Markdown 图片语法嵌入：\n"
            "  ![图N 说明](figures/<文件名>)\n"
            "并在正文中引用、解读每张图。图片路径用 figures/<文件名>（相对论文文件）。不要凭空引用不存在的图。"
        )

    def build_user_prompt(self, ctx) -> str:
        problem = self._problem(ctx)
        analysis = self._prior(ctx, AgentName.ANALYST)
        model = self._prior(ctx, AgentName.MODELER)
        solver_out = ctx.solution_stdout or "(无求解输出)"
        solver_code_path = self.task.state.agents.get("solver").artifact_path if self.task.state.agents.get("solver") else None
        solver_code = ""
        if solver_code_path:
            solver_code = self.store.read_artifact(self.task.meta.task_id, solver_code_path)

        figures_info = ""
        if ctx.figures:
            figures_info = "【求解生成的图表】（必须在论文中嵌入对应位置）：\n" + "\n".join(
                f"- figures/{f}" for f in ctx.figures
            ) + "\n\n"

        prefix = ""
        if ctx.review_feedback:
            prefix = f"【审查反馈，请据此改进论文】\n{ctx.review_feedback}\n\n"

        return (
            f"{prefix}{figures_info}"
            f"【题目】\n{problem}\n\n"
            f"【问题分析】\n{analysis}\n\n"
            f"【数学模型】\n{model}\n\n"
            f"【求解代码】\n```python\n{solver_code}\n```\n\n"
            f"【求解结果（程序输出）】\n{solver_out}\n\n"
            "请基于以上材料撰写完整论文。"
        )

    def postprocess(self, ctx, text: str) -> tuple[str, str, dict]:
        # 摘要取论文字数，不污染论文正文
        char_count = len(text)
        section_count = text.count("\n# ")
        summary = f"论文已生成，约 {char_count} 字，含 {section_count} 个章节"
        return text, summary, {"char_count": char_count, "section_count": section_count}

