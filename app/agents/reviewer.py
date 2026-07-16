"""审查师 Agent：对全流程产物打分（0-100），不通过则指明回退到哪个 Agent。

输出结构化 JSON，由 postprocess 解析，供编排器决定是否回退。
"""
from __future__ import annotations

import json
import re

from app.agents.base import BaseAgent
from app.storage import AgentName


class ReviewerAgent(BaseAgent):
    name = AgentName.REVIEWER

    @property
    def system_prompt(self) -> str:
        return (
            "你是一名严格的数学建模竞赛评审专家。\n"
            "审查整篇建模论文及相关产物，给出评分与改进意见。\n\n"
            "评分维度（每项 0-100）：\n"
            "- problem_understanding：问题理解是否准确\n"
            "- modeling：模型是否合理、数学表达是否严谨\n"
            "- solving：求解是否正确、结果是否可信\n"
            "- writing：论文结构、表达、规范\n"
            "- rigor：假设合理性、结论可靠性\n\n"
            "★★★ 硬性规则（必须执行，不可通融）：\n"
            "1. 若【求解执行状态】为失败或无输出，solving 项直接给 ≤30 分，"
            "passed 必须为 false，rollback_to 必须为 \"solver\"。理由：代码没跑通=没有真实结果，论文数值不可信。\n"
            "2. 打分必须基于证据：solving 的分数取决于【求解结果输出】中是否有具体数字。"
            "若论文里的数值在求解输出中找不到对应，视为编造，solving 扣分。\n"
            "3. 不要被论文的流畅程度欺骗——重点核对结论是否有真实求解支撑。\n"
            "4. 总分 overall_score 取各维度加权（solving 权重更高，占 30%）。通过线 75。\n\n"
            "只输出一个 JSON 代码块（```json 围栏），格式：\n"
            "```json\n"
            "{\n"
            '  "overall_score": 85,\n'
            '  "scores": {"problem_understanding": 90, "modeling": 85, "solving": 80, "writing": 85, "rigor": 85},\n'
            '  "passed": true,\n'
            '  "evidence": "引用求解输出中的具体数字作为评分依据",\n'
            '  "issues": ["问题描述1"],\n'
            '  "suggestions": ["改进建议1"],\n'
            '  "rollback_to": null\n'
            "}\n"
            "```\n"
            "rollback_to 取值: analyst/modeler/solver/writer/null。只输出 JSON。"
        )

    def build_user_prompt(self, ctx) -> str:
        problem = self._problem(ctx)
        analysis = self._prior(ctx, AgentName.ANALYST)
        model = self._prior(ctx, AgentName.MODELER)
        paper = self._prior(ctx, AgentName.WRITER)
        executed = ctx.solution_executed
        if executed is True:
            exec_status = "成功（代码已执行，输出见下）"
        elif executed is False:
            exec_status = f"失败（代码未能成功执行。错误: {ctx.solution_error or '未知'}）"
        else:
            exec_status = "未知（无求解记录）"
        solver_out = ctx.solution_stdout or "(无输出——求解未执行成功)"
        # 逐子问题阶段状态（若 status.json 存在）
        status_text = ""
        status_raw = self.store.read_solution_file(self.task.meta.task_id, "status.json")
        if status_raw:
            try:
                st = json.loads(status_raw)
                lines = ["【逐子问题阶段状态】"]
                for sp in st.get("subproblems", []):
                    stages = ", ".join(f"{s['name']}={'OK' if s['ok'] else 'FAIL'}" for s in sp.get("stages", []))
                    lines.append(f"- {sp.get('id')}: 关键阶段={sp.get('critical_stage')} -> {'OK' if sp.get('ok') else 'FAIL'} [{stages}]")
                status_text = "\n".join(lines) + "\n\n"
            except json.JSONDecodeError:
                pass
        return (
            f"【题目】\n{problem}\n\n"
            f"【问题分析】\n{analysis}\n\n"
            f"【数学模型】\n{model}\n\n"
            f"【求解执行状态】{exec_status}\n\n"
            f"{status_text}"
            f"【求解结果输出】\n{solver_out}\n\n"
            f"【论文全文】\n{paper}\n\n"
            "请严格审查并输出评分 JSON。务必先核对求解执行状态与输出，再打分。"
        )

    def postprocess(self, ctx, text: str) -> tuple[str, str, dict]:
        review = self._parse_review(text)
        # 把结构化结果也写成可读 Markdown 落盘
        md = self._to_markdown(review)
        score = review.get("overall_score", 0)
        passed = bool(review.get("passed", score >= 75))
        rollback = review.get("rollback_to")

        # 同步到 AgentRecord
        rec = self.task.state.get_agent(self.name)
        rec.review_score = score
        rec.review_passed = passed

        summary = f"评分 {score}/100，{'通过' if passed else '不通过'}"
        if not passed and rollback:
            summary += f"，回退到 {rollback}"
        return md, summary, {
            "review": review, "score": score,
            "passed": passed, "rollback_to": rollback,
        }

    def _parse_review(self, text: str) -> dict:
        """从回答中提取 JSON。"""
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        raw = m.group(1) if m else text
        # 兜底：找第一个 { ... }
        if not m:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                raw = raw[start:end + 1]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {
                "overall_score": 0, "passed": False,
                "issues": ["审查输出解析失败"], "rollback_to": "writer",
            }

    @staticmethod
    def _to_markdown(review: dict) -> str:
        lines = ["# 审查报告\n"]
        lines.append(f"**总分**: {review.get('overall_score', 'N/A')}/100\n")
        lines.append(f"**结论**: {'通过 ✓' if review.get('passed') else '不通过 ✗'}\n")
        scores = review.get("scores", {})
        if scores:
            lines.append("## 分项评分\n")
            for k, v in scores.items():
                lines.append(f"- {k}: {v}")
            lines.append("")
        if review.get("issues"):
            lines.append("## 存在问题\n")
            for i in review["issues"]:
                lines.append(f"- {i}")
            lines.append("")
        if review.get("suggestions"):
            lines.append("## 改进建议\n")
            for s in review["suggestions"]:
                lines.append(f"- {s}")
            lines.append("")
        if review.get("rollback_to"):
            lines.append(f"\n> 建议回退到 Agent: **{review['rollback_to']}**\n")
        return "\n".join(lines)
