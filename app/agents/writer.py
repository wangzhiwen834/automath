"""写作师 Agent：综合所有上游产物 → 生成完整竞赛论文 Markdown。"""
from __future__ import annotations

import json
import re

from app.agents.base import BaseAgent
from app.agents.write_utils import is_thin_section, cross_check_numbers
from app.config import get_settings
from app.llm.provider import Message
from app.storage import AgentName


class WriterAgent(BaseAgent):
    name = AgentName.WRITER

    SECTIONS = [
        ("abstract", "摘要", 400), ("restatement", "一、问题重述", 300),
        ("analysis", "二、问题分析", 300), ("assumption", "三、模型假设", 200),
        ("notation", "四、符号说明", 200), ("solving", "五、模型建立与求解", 600),
        ("evaluation", "六、模型评价与推广", 300), ("reference", "七、参考文献", 100),
        ("appendix", "附录", 200),
    ]

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

    def _execute(self, ctx, stream_callback) -> tuple[str, str, dict]:
        tid = self.task.meta.task_id
        cfg = get_settings().writer_config
        max_expand = cfg.get("max_expand_sections", 4)
        do_consistency = cfg.get("consistency_check", True)
        min_default = cfg.get("min_section_chars", {}).get("default", 300)

        outline = self._make_outline(ctx)
        paper_dir = self.store.task_path(tid) / "artifacts" / "paper" / "sections"
        paper_dir.mkdir(parents=True, exist_ok=True)
        (paper_dir.parent / "outline.json").write_text(
            json.dumps(outline, ensure_ascii=False, indent=2), encoding="utf-8")

        section_texts: list[tuple[dict, str]] = []
        expand_used = 0
        for sec in outline:
            text = self._write_section(ctx, sec)
            mc = sec.get("min_chars", min_default)
            if is_thin_section(text, mc) and expand_used < max_expand:
                text = self._expand_section(ctx, sec, text)
                expand_used += 1
            (paper_dir / f"{sec['id']}.md").write_text(text, encoding="utf-8")
            section_texts.append((sec, text))
            if stream_callback:
                stream_callback(f"[{sec['id']}] 完成\n")

        paper = self._assemble(section_texts)

        if do_consistency:
            r = self._consistency_check(ctx, paper)
            # 有界返修：最多 3 节
            regen_count = 0
            for sec, text in section_texts:
                issues = []
                for o in r.get("offending_sections", []):
                    if o.get("section") == sec["id"]:
                        issues = o.get("issues", [])
                if (issues or r.get("off_topic")) and regen_count < 3:
                    new_text = self._regen_section(ctx, sec, text, issues)
                    section_texts[section_texts.index((sec, text))] = (sec, new_text)
                    regen_count += 1
            paper = self._assemble(section_texts)

        char_count = len(paper)
        section_count = paper.count("\n# ") + 1
        summary = f"论文已生成，约 {char_count} 字，含 {section_count} 个章节"
        return paper, summary, {"char_count": char_count, "section_count": section_count}

    # ----------------------------------------------------------
    # 分节流水线（Task 12+）：大纲 + 逐节生成
    # ----------------------------------------------------------
    def _make_outline(self, ctx) -> list[dict]:
        problem = self._problem(ctx)
        analysis = self._prior(ctx, AgentName.ANALYST)
        model = self._prior(ctx, AgentName.MODELER)
        solver_out = ctx.solution_stdout or "(无)"
        figs = ", ".join(ctx.figures) if ctx.figures else "(无)"
        prompt = (
            "为数学建模论文拟定大纲。固定章节及最小字数：\n"
            + "\n".join(f"- {sid}: {title} (≥{mc}字)" for sid, title, mc in self.SECTIONS) + "\n\n"
            f"【题目】{problem}\n【分析】{analysis[:1500]}\n【模型】{model[:1500]}\n"
            f"【求解结果】{solver_out[:1500]}\n【图表】{figs}\n\n"
            "只输出 JSON 数组，每项 {id,title,points(要点list),min_chars,context_hint}。"
        )
        text = self.llm.chat([Message("system", "你是论文大纲设计者。只输出 JSON。"),
                              Message("user", prompt)])
        m = re.search(r"\[.*\]", text, re.DOTALL)
        try:
            arr = json.loads(m.group(0)) if m else json.loads(text)
            if arr:
                return arr
        except json.JSONDecodeError:
            pass
        return [{"id": sid, "title": title, "points": [], "min_chars": mc, "context_hint": "all"}
                for sid, title, mc in self.SECTIONS]

    def _section_context(self, ctx, section) -> str:
        sid = section["id"]
        if sid == "abstract":
            return f"分析摘要：{self._prior(ctx, AgentName.ANALYST)[:800]}\n求解结果：{ctx.solution_stdout or ''}"
        if sid == "solving":
            return (f"模型：{self._prior(ctx, AgentName.MODELER)}\n"
                    f"求解结果：{ctx.solution_stdout or ''}\n图表：{ctx.figures}")
        if sid == "appendix":
            return f"求解代码/manifest：{ctx.artifacts.get('solver', '')[:3000]}"
        return f"题目：{self._problem(ctx)}\n分析：{self._prior(ctx, AgentName.ANALYST)[:1000]}"

    def _write_section(self, ctx, section) -> str:
        figs_info = ""
        if ctx.figures and section["id"] in ("solving", "evaluation", "abstract"):
            figs_info = "可用图表，需用 ![图N 说明](figures/<文件>) 嵌入并解读：" + ", ".join(ctx.figures)
        prompt = (
            f"撰写论文章节【{section['title']}】（至少 {section['min_chars']} 字）。\n"
            f"要点：{section.get('points', [])}\n{figs_info}\n\n"
            "硬性约束：只能用下列材料中的事实与数值，不得编造方法/结果/数字；所需数值不在材料中须如实说明。\n\n"
            f"【材料】\n{self._section_context(ctx, section)}\n"
        )
        text = self.llm.chat([Message("system", self.system_prompt), Message("user", prompt)])
        return text.strip()

    def _assemble(self, section_texts: list[tuple[dict, str]]) -> str:
        return "\n\n".join(t for _, t in section_texts).strip()

    def _expand_section(self, ctx, section, text) -> str:
        prompt = (
            f"下面的论文章节【{section['title']}】过短，请扩写到至少 {section['min_chars']} 字，"
            "保持与上游材料一致，不得编造。只输出扩写后的该节正文。\n\n"
            f"【当前内容】\n{text}\n\n【材料】\n{self._section_context(ctx, section) if ctx else ''}"
        )
        return self.llm.chat([Message("system", self.system_prompt), Message("user", prompt)]).strip()

    # ----------------------------------------------------------
    # 一致性校验 + 有界返修（Task 14）
    # ----------------------------------------------------------
    def _consistency_check(self, ctx, paper) -> dict:
        solver_out = ctx.solution_stdout or ""
        fabricated = cross_check_numbers(paper, solver_out)
        prompt = (
            "把论文与上游分析/模型/求解结果对照，检查：方法/模型是否与建模一致、结论是否有结果支撑、"
            "是否跑题夹带无关内容、有无过度推断。只输出 JSON：\n"
            '```json\n{"offending_sections":[{"section":"abstract","issues":["..."]}],"off_topic":false,"fabricated_numbers":[]}\n```'
            f"\n\n【论文】\n{paper[:4000]}\n\n【求解结果】\n{solver_out[:2000]}\n"
            f"【分析】\n{self._prior(ctx, AgentName.ANALYST)[:1500]}\n【模型】\n{self._prior(ctx, AgentName.MODELER)[:1500]}"
        )
        text = self.llm.chat([Message("system", "你是论文一致性审查者。只输出 JSON。"), Message("user", prompt)])
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        raw = m.group(1) if m else text
        try:
            r = json.loads(raw)
        except json.JSONDecodeError:
            r = {"offending_sections": [], "off_topic": False, "fabricated_numbers": []}
        r["fabricated_numbers"] = list(set(r.get("fabricated_numbers", [])) | set(fabricated))
        return r

    def _regen_section(self, ctx, section, text, issues) -> str:
        prompt = (
            f"章节【{section['title']}】存在一致性问题，请据问题重写该节（只输出该节正文，保持与上游一致，不编造）。\n"
            f"【问题】{issues}\n【当前内容】\n{text}\n\n【材料】\n{self._section_context(ctx, section) if ctx else ''}"
        )
        return self.llm.chat([Message("system", self.system_prompt), Message("user", prompt)]).strip()

