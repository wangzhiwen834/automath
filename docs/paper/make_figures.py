"""论文配图生成脚本：用 matplotlib（求解器同款方式）绘制论文中的概念图。

生成:
  figures/fig1_architecture.png   系统总体架构
  figures/fig2_solver_flow.png    分阶段求解器 + 分层校验流程
  figures/fig3_defense.png        防编造三重防线

运行: python docs/paper/make_figures.py
中文字体用 SimHei（与项目 PDF 导出一致）。
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager

# ---------- 中文字体 ----------
for fp in ["C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/msyh.ttc"]:
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=fp).get_name()]
        break
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

C_BLUE = "#2E5C8A"
C_BLUE_L = "#D6E4F0"
C_GREEN = "#3A7D44"
C_GREEN_L = "#D9EAD3"
C_ORANGE = "#B5651D"
C_ORANGE_L = "#FCE5CD"
C_GRAY = "#666666"
C_RED = "#A33"


def box(ax, x, y, w, h, text, fc, ec, fontsize=11, weight="normal", tc="#111"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       fc=fc, ec=ec, lw=1.5)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=tc, wrap=True)
    return (x + w / 2, y, x + w / 2, y + h)  # bottom-center, top-center


def varrow(ax, x, y1, y2, color=C_GRAY):
    ax.add_patch(FancyArrowPatch((x, y1), (x, y2), arrowstyle="-|>",
                                 mutation_scale=14, color=color, lw=1.4))


def harrow(ax, x1, x2, y, color=C_GRAY, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>",
                                 mutation_scale=14, color=color, lw=1.4, linestyle=ls))


# ====================================================================
# 图1 系统总体架构
# ====================================================================
def fig1_architecture():
    fig, ax = plt.subplots(figsize=(10, 11.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 14); ax.axis("off")

    ax.text(5, 13.6, "图1  系统总体架构", ha="center", fontsize=14, fontweight="bold")

    # 层（自上而下）
    box(ax, 1.0, 12.2, 8.0, 1.0,
        "前端 (Next.js)\n实时流式输出 · 流水线看板 · 图表展示 · PDF/DOCX 导出",
        C_BLUE_L, C_BLUE, fontsize=10)
    varrow(ax, 5, 12.2, 11.5)
    box(ax, 1.0, 10.5, 8.0, 1.0,
        "FastAPI 后端 + TaskRunner\n后台线程跑编排器 · pub-sub 推事件到 WebSocket",
        C_BLUE_L, C_BLUE, fontsize=10)
    varrow(ax, 5, 10.5, 9.8)

    # 编排器层 + 流水线
    box(ax, 0.6, 8.7, 8.8, 1.1,
        "Orchestrator 编排器（文件状态机 · 三种人在回路模式 · 审查回退）",
        C_GREEN_L, C_GREEN, fontsize=11, weight="bold")
    # 流水线 6 个小盒
    agents = ["分析", "建模", "求解", "总结", "写作", "审查"]
    x0, y0, w, gap = 0.8, 7.2, 1.28, 0.16
    centers = []
    for i, a in enumerate(agents):
        x = x0 + i * (w + gap)
        fc = C_ORANGE_L if a != "审查" else "#F4CCCC"
        ec = C_ORANGE if a != "审查" else C_RED
        box(ax, x, y0, w, 0.8, a, fc, ec, fontsize=11, weight="bold")
        centers.append((x, x + w, y0 + 0.4))
        if i < len(agents) - 1:
            harrow(ax, x + w, x + w + gap, y0 + 0.4)
    # 回退弧线：审查 -> 建模（虚线）
    ax.add_patch(FancyArrowPatch((centers[5][1], y0 + 0.8), (centers[1][0], y0 + 0.8),
                                 connectionstyle="arc3,rad=-0.35", arrowstyle="-|>",
                                 mutation_scale=14, color=C_RED, lw=1.4, linestyle="--"))
    ax.text(5, 6.75, "审查不通过 → 回退重做（虚线）", ha="center", fontsize=9, color=C_RED)
    varrow(ax, 5, 7.2, 6.5)

    box(ax, 1.0, 5.4, 8.0, 1.1,
        "BaseAgent 基类\n统一：系统提示 · 流式日志 · 产物落盘 · 状态更新",
        "#FFF2CC", "#BF9000", fontsize=10)
    varrow(ax, 5, 5.4, 4.7)

    # LLM 抽象层（两 provider）
    box(ax, 1.0, 4.2, 8.0, 0.5, "LLM 抽象层（统一 chat/stream 接口）", C_BLUE_L, C_BLUE,
        fontsize=10, weight="bold")
    box(ax, 1.0, 3.2, 3.8, 0.9, "OpenAI 兼容\nDeepSeek · 通义千问 · GLM", C_BLUE_L, C_BLUE, fontsize=9.5)
    box(ax, 5.2, 3.2, 3.8, 0.9, "Anthropic\nClaude", C_BLUE_L, C_BLUE, fontsize=9.5)
    varrow(ax, 5, 3.2, 2.6)

    box(ax, 1.0, 1.4, 8.0, 1.2,
        "TaskStore 文件化存储（无数据库）\n每任务一目录 · meta/state JSON · 原子写 · 崩溃可恢复 · JSONL 日志",
        C_GREEN_L, C_GREEN, fontsize=10)

    plt.tight_layout()
    plt.savefig(OUT / "fig1_architecture.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("fig1_architecture.png")


# ====================================================================
# 图2 分阶段求解器 + 分层校验流程
# ====================================================================
def fig2_solver_flow():
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10); ax.set_ylim(0, 14.5); ax.axis("off")
    ax.text(5, 14.1, "图2  分阶段求解器与分层校验流程", ha="center", fontsize=14, fontweight="bold")

    cx = 5  # 主链中心 x
    # 主链节点 (y, text, fc, ec)
    nodes = [
        (12.9, "规划步 plan.json\n（子问题 × 阶段：data/model/solve/analyze/plot）", C_GREEN_L, C_GREEN),
        (11.7, "生成代码", "#FFF2CC", "#BF9000"),
        (10.7, "注入可复现前导（seed=42 · matplotlib Agg）", C_BLUE_L, C_BLUE),
        (9.7, "py_compile 语法预检", C_BLUE_L, C_BLUE),
        (8.7, "subprocess 执行", C_BLUE_L, C_BLUE),
        (7.7, "解析 STAGE_RESULT", C_BLUE_L, C_BLUE),
        (6.7, "第一层：程序化硬检查\n（退出码·ok·有限值·输出文件·PNG·范围）", C_ORANGE_L, C_ORANGE),
        (5.4, "第二层：LLM 自查（语义·红旗·一致性）", C_ORANGE_L, C_ORANGE),
        (4.2, "下一阶段 / 下一子问题", C_GREEN_L, C_GREEN),
        (3.0, "汇总自查（跨子问题一致性）", C_GREEN_L, C_GREEN),
        (1.7, "status.json · output.txt · summary.md", C_BLUE_L, C_BLUE),
    ]
    bw, bh = 7.6, 0.95
    pos = {}
    for y, text, fc, ec in nodes:
        box(ax, cx - bw / 2, y, bw, bh, text, fc, ec, fontsize=9.5,
            weight="bold" if "层" in text or "规划" in text or "汇总" in text else "normal")
        pos[text.split("\n")[0][:6]] = y
        # 向下箭头到下一个
    # 简化：用文本 key 顺序连箭头
    keys = [n[1].split("\n")[0][:6] for n in nodes]
    ys = [n[0] for n in nodes]
    hs = [bh] * len(nodes)
    for i in range(len(nodes) - 1):
        varrow(ax, cx, ys[i], ys[i + 1] + hs[i] - hs[i + 1] + (hs[i + 1] - hs[i]) * 0 + 0, ) if False else None
        varrow(ax, cx, ys[i], ys[i + 1] + hs[i + 1])
    # 修正：上一行连接到下一个顶部
    # 重画箭头（清除并精确）
    # （上面已画足够；微调略）

    # 有界返修侧支：硬检查失败 -> 回到"生成代码"
    ax.add_patch(FancyArrowPatch((cx - bw / 2, 6.7 + bh / 2), (cx - bw / 2, 11.7 + bh / 2),
                                 connectionstyle="arc3,rad=0.5", arrowstyle="-|>",
                                 mutation_scale=14, color=C_RED, lw=1.4, linestyle="--"))
    ax.text(0.2, 9.2, "硬检查失败\n带错误回填\n有界返修\n(<=max_regen)", ha="left", va="center",
            fontsize=8.5, color=C_RED)
    # 自查不过 -> 回到"生成代码"
    ax.add_patch(FancyArrowPatch((cx + bw / 2, 5.4 + bh / 2), (cx + bw / 2, 11.7 + bh / 2),
                                 connectionstyle="arc3,rad=-0.5", arrowstyle="-|>",
                                 mutation_scale=14, color=C_RED, lw=1.4, linestyle="--"))
    ax.text(9.0, 8.4, "自查不过\n带issues回填\n有界返修\n(<=max_critique)", ha="left", va="center",
            fontsize=8.5, color=C_RED)

    ax.text(5, 0.7, "单阶段耗尽不中断整任务，其他子问题继续；executed = 所有关键(solve)阶段成功",
            ha="center", fontsize=9, color=C_GRAY, style="italic")
    plt.tight_layout()
    plt.savefig(OUT / "fig2_solver_flow.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("fig2_solver_flow.png")


# ====================================================================
# 图3 防编造三重防线
# ====================================================================
def fig3_defense():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
    ax.text(6, 6.6, "图3  防编造三重防线", ha="center", fontsize=14, fontweight="bold")

    cols = [
        (0.4, "产稿前接地", "总结师 · 事实清单 facts.md",
         "· 数值须来自 STAGE_RESULT\n· 极大/极小严格区分\n· 未生成的图不得列入\n· 跨环节矛盾 [!] 标注", C_GREEN_L, C_GREEN),
        (4.3, "产稿前一致性", "写作器 · 一致性校验",
         "· 程序化数值交叉核对\n  (论文数字须在事实清单中)\n· LLM 语义核对\n  (跑题/编造/幻图)\n· 命中节有界返修", C_ORANGE_L, C_ORANGE),
        (8.2, "产稿后核对", "审查师 · 硬规则评分",
         "· 求解未执行→solving<=30\n  回退 solver\n· 编造/极值混淆/幻图\n  →回退 writer\n· 以事实清单为准核对", "#F4CCCC", C_RED),
    ]
    w = 3.4; h = 4.0; y = 1.6
    centers = []
    for x, stage, who, body, fc, ec in cols:
        # 顶部标签
        box(ax, x, y + h, w, 0.7, stage, fc, ec, fontsize=11, weight="bold")
        box(ax, x, y, w, h, who + "\n\n" + body, fc, ec, fontsize=9.5)
        centers.append(x + w)
    # 顶部流程箭头：求解 -> 总结 -> 写作 -> 审查
    ax.text(centers[0] - w / 2, y + h + 1.05, "求解输出", ha="center", fontsize=9, color=C_GRAY)
    harrow(ax, centers[0] - w / 2 + 0.9, centers[1] - w / 2 - 0.1, y + h + 0.9, color=C_GRAY)
    harrow(ax, centers[1] - w / 2 + 0.9, centers[2] - w / 2 - 0.1, y + h + 0.9, color=C_GRAY)
    ax.text(centers[2] + 0.2, y + h + 1.05, "论文", ha="center", fontsize=9, color=C_GRAY)

    ax.text(6, 0.7, "产稿前接地 → 产稿前一致性校验 → 产稿后事实核对：数值编造被三道独立检查拦截",
            ha="center", fontsize=9.5, color=C_GRAY, style="italic")
    plt.tight_layout()
    plt.savefig(OUT / "fig3_defense.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("fig3_defense.png")


if __name__ == "__main__":
    fig1_architecture()
    fig2_solver_flow()
    fig3_defense()
    print("全部图已生成于", OUT)
