"""端到端运行器：读题目文件 → 创建任务 → 编排器跑完整流程。

用法:
  python scripts/run_task.py <题目文件> [auto|interactive|hybrid]

默认 auto 模式。会消耗较多 token（5 个 Agent 全跑）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.storage import ProblemType, RunMode, get_store
from app.orchestrator import Orchestrator

PROBLEM_TYPE_HINTS = {
    "optimization": ProblemType.OPTIMIZATION,
    "evaluation": ProblemType.EVALUATION,
    "statistics": ProblemType.STATISTICS,
    "mechanism": ProblemType.MECHANISM,
    "graph": ProblemType.GRAPH,
}


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python scripts/run_task.py <题目文件> [auto|interactive|hybrid] [题型]")
        print("      题型: optimization/evaluation/statistics/mechanism/graph")
        sys.exit(1)

    problem_file = Path(sys.argv[1])
    mode = RunMode(sys.argv[2]) if len(sys.argv) > 2 else RunMode.AUTO
    ptype = PROBLEM_TYPE_HINTS.get(sys.argv[3], ProblemType.UNKNOWN) if len(sys.argv) > 3 else ProblemType.UNKNOWN

    problem_text = problem_file.read_text(encoding="utf-8")
    title = problem_file.stem

    store = get_store()
    task = store.create_task(
        title=title, problem_text=problem_text,
        mode=mode, problem_type=ptype,
    )
    print(f"任务已创建: {task.meta.task_id}")
    print(f"模式: {mode.value}  题型: {ptype.value}")
    print(f"目录: {task.dir}\n")

    # 事件回调：实时打印
    def on_event(ev: dict) -> None:
        t = ev.get("type")
        if t == "agent_start":
            print(f"\n{'='*60}\n▶ [{ev['agent']}] 开始（模型: {ev.get('model')}）")
        elif t == "agent_done":
            print(f"\n✔ [{ev['agent']}] 完成: {ev['summary']}")
        elif t == "paused":
            print(f"\n⏸ 暂停（{ev['after']}完成后）→ 下一个: {ev['next']}")
        elif t == "review":
            print(f"\n🔍 审查: 评分={ev['score']} 通过={ev['passed']} 回退={ev['rollback_to']}")
        elif t == "rollback":
            print(f"\n↩ 回退到 [{ev['to']}]（第 {ev['round']} 轮重试）")
        elif t == "completed":
            print(f"\n{'='*60}\n🏁 完成！评分={ev.get('score')} {ev.get('note','') or ''}")
        elif t == "failed":
            print(f"\n✗ 失败: {ev}")

    orch = Orchestrator(task.meta.task_id, event_sink=on_event)
    final = orch.run()

    # 输出结果摘要
    print(f"\n{'='*60}")
    print(f"最终状态: {final.state.status.value}")
    print(f"审查轮次: {final.state.review_round}")
    for agent in ["analyst", "modeler", "solver", "writer", "reviewer"]:
        rec = final.state.agents.get(agent)
        if rec:
            mark = "✓" if rec.status.value == "done" else rec.status.value
            print(f"  {mark} {agent:<10} {rec.summary or ''}")

    paper = store.read_artifact(final.meta.task_id, "artifacts/paper.md")
    print(f"\n论文长度: {len(paper)} 字符")
    print(f"任务目录: {final.dir}")


if __name__ == "__main__":
    main()
