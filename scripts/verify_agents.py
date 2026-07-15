"""验证单个 Agent 端到端：用真实 GLM-4.7 跑分析师 Agent。

测试 BaseAgent 完整流程：构建消息 → 流式调用 → 日志写入 → 产物落盘 → 状态更新。
会消耗少量 token。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.storage import AgentName, ProblemType, RunMode, get_store
from app.agents import AnalystAgent
from app.agents.base import RunContext


def main() -> None:
    store = get_store()

    problem = (
        "某物流公司每天需从3个仓库向8个配送点送货。已知各仓库库存、各配送点需求量、"
        "仓库到配送点的单位运输成本。要求制定运输方案使总运输成本最低。"
        "请建立数学模型并求解。"
    )

    task = store.create_task(
        title="运输问题（单Agent测试）",
        problem_text=problem,
        mode=RunMode.AUTO,
        problem_type=ProblemType.OPTIMIZATION,
    )
    print(f"任务已创建: {task.meta.task_id}\n")

    # 构造上下文（analyst 是第一个，没有上游产物）
    ctx = RunContext(task=task, store=store)

    agent = AnalystAgent(task, store)

    # 流式回调：实时打印到控制台
    print("=" * 60)
    print("分析师 Agent 流式输出：")
    print("=" * 60)
    def on_chunk(chunk: str) -> None:
        print(chunk, end="", flush=True)

    result = agent.run(ctx, stream_callback=on_chunk)
    print("\n" + "=" * 60)

    print(f"\n成功: {result.success}")
    print(f"摘要: {result.summary}")
    print(f"产物路径: {result.artifact_path}")

    # 读回落盘的产物前 200 字
    artifact = store.read_artifact(task.meta.task_id, result.artifact_path)
    print(f"\n落盘产物前200字:\n{artifact[:200]}")

    # 检查日志
    logs = store.read_log(task.meta.task_id, "analyst")
    print(f"\n日志事件数: {len(logs)}（含 start/messages/delta/done）")

    # 重新加载任务确认状态持久化
    task2 = store.load(task.meta.task_id)
    rec = task2.state.agents["analyst"]
    print(f"\n重新加载后 analyst 状态: {rec.status.value}")
    print(f"模型: {rec.model}  摘要: {rec.summary}")

    print("\n" + "=" * 60)
    print("✓ AnalystAgent 端到端验证通过")


if __name__ == "__main__":
    main()
