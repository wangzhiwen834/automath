"""验证任务存储层：创建 → 状态变更 → 保存 → 重新加载 → 列表 → 产物读写。

纯文件操作，不调用 LLM，不消耗 token。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.storage import (
    AgentName,
    AgentStatus,
    ProblemType,
    RunMode,
    TaskStatus,
    get_store,
)


def main() -> None:
    store = get_store()
    print(f"工作目录: {store.workspace}\n")

    # 1. 创建任务（模式 C 混合）
    problem = """2023年高教社杯 C 题（示例截取）：
某电商平台有多种生鲜商品，需根据历史销量数据预测未来一周销量，
并制定补货与定价策略，目标是最大化利润同时降低损耗。
请建立数学模型完成：1) 销量预测 2) 补货决策 3) 定价优化。"""

    task = store.create_task(
        title="生鲜电商销量预测与补货定价（示例）",
        problem_text=problem,
        mode=RunMode.HYBRID,
        problem_type=ProblemType.STATISTICS,
    )
    tid = task.meta.task_id
    print(f"[1] 创建任务成功: {tid}")
    print(f"    模式={task.meta.mode.value}  题型={task.meta.problem_type.value}")
    print(f"    各Agent模型: {task.meta.agent_models}")
    print(f"    目录: {task.dir}\n")

    # 2. 模拟分析师 Agent 产出
    rel = store.write_artifact(tid, AgentName.ANALYST, "# 问题分析\n\n本题分为预测、补货、定价三子问题...")
    rec = task.state.get_agent(AgentName.ANALYST)
    rec.status = AgentStatus.DONE
    rec.artifact_path = rel
    rec.summary = "拆解为预测/补货/定价三子问题"
    store.add_history(task, "done", agent="analyst", detail="分析完成")
    task.state.current_agent = AgentName.MODELER
    task.state.status = TaskStatus.PAUSED  # 混合模式在 modeler 前暂停
    task.state.waiting_for_human = True
    store.save(task)
    print(f"[2] 写入分析师产物 → {rel}")
    print(f"    状态改为 PAUSED（等待人工确认进入建模）\n")

    # 3. 从磁盘重新加载（模拟恢复）
    task2 = store.load(tid)
    assert task2.state.status == TaskStatus.PAUSED
    assert task2.state.agents["analyst"].status == "done"
    print(f"[3] 重新加载成功，状态={task2.state.status.value}")
    print(f"    历史事件数: {len(task2.state.history)}")
    print(f"    analyst摘要: {task2.state.agents['analyst'].summary}\n")

    # 4. 读回产物
    content = store.read_artifact(tid, rel)
    print(f"[4] 读回产物前30字: {content[:30]}...\n")

    # 5. 列表
    tasks = store.list_tasks()
    print(f"[5] 任务列表（共{len(tasks)}个）:")
    for t in tasks[:5]:
        print(f"    {t['task_id']}  {t['status']:<10}  {t['title']}")

    # 6. 日志追加与读取
    log_rel = store.append_log(tid, "analyst", {"type": "token", "text": "分析中..."})
    store.append_log(tid, "analyst", {"type": "token", "text": "完成。"})
    events = store.read_log(tid, "analyst")
    print(f"\n[6] 日志写入 {log_rel}，读回 {len(events)} 条事件\n")

    print("=" * 50)
    print("✓ 存储层全部验证通过")


if __name__ == "__main__":
    main()
