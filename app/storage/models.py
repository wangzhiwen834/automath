"""任务数据模型：全部基于 pydantic，可直接 dump 成 meta.json / state.json。

无数据库：每个任务一个目录，状态全在 JSON 文件里，可随时恢复。
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ====================================================================
# 枚举
# ====================================================================
class TaskStatus(str, Enum):
    CREATED = "created"        # 刚创建，未启动
    RUNNING = "running"        # 执行中
    PAUSED = "paused"          # 等待人工确认（模式 B/C 的检查点）
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunMode(str, Enum):
    """三种自动化模式。"""
    AUTO = "auto"              # A: 全自动
    INTERACTIVE = "interactive"  # B: 人在回路，每个 Agent 后暂停
    HYBRID = "hybrid"          # C: 混合，关键节点（建模方案/最终论文）暂停


class ProblemType(str, Enum):
    """CUMCM 常见题型。"""
    OPTIMIZATION = "optimization"      # 优化类（线性/整数/非线性规划）
    EVALUATION = "evaluation"          # 评价类（AHP/TOPSIS/模糊综合）
    STATISTICS = "statistics"          # 统计预测类（回归/时间序列/聚类）
    MECHANISM = "mechanism"            # 机理建模/微分方程
    GRAPH = "graph"                    # 图论网络类
    UNKNOWN = "unknown"                # 待分析


class AgentName(str, Enum):
    ANALYST = "analyst"        # 问题分析
    MODELER = "modeler"        # 建模
    SOLVER = "solver"          # 求解（写代码+执行）
    WRITER = "writer"          # 写论文
    REVIEWER = "reviewer"      # 审查


class AgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


# 流程顺序（审查会回退，但主线顺序固定）
AGENT_ORDER = [
    AgentName.ANALYST,
    AgentName.MODELER,
    AgentName.SOLVER,
    AgentName.WRITER,
    AgentName.REVIEWER,
]

# 模式 C 的检查点：在这些 Agent 完成后暂停等待人工确认
HYBRID_CHECKPOINTS = {AgentName.MODELER, AgentName.WRITER}
# 模式 B 的检查点：所有 Agent 完成后都暂停
INTERACTIVE_CHECKPOINTS = set(AGENT_ORDER)


# ====================================================================
# Agent 执行记录
# ====================================================================
class AgentRecord(BaseModel):
    agent: AgentName
    status: AgentStatus = AgentStatus.PENDING
    model: str | None = None              # 实际使用的模型 key
    started_at: str | None = None
    finished_at: str | None = None
    artifact_path: str | None = None      # 相对任务目录的产物路径
    log_path: str | None = None
    retry_count: int = 0
    review_score: int | None = None       # 审查打分 0-100
    review_passed: bool | None = None
    summary: str | None = None            # 一句话摘要，供前端展示
    error: str | None = None


# ====================================================================
# 任务状态（state.json）
# ====================================================================
class HistoryEvent(BaseModel):
    timestamp: str
    agent: str | None = None
    action: str                            # started/done/failed/paused/resumed/review/rollback...
    detail: str | None = None


class TaskState(BaseModel):
    """编排状态：当前进度、各 Agent 记录、历史、审查回合。"""
    status: TaskStatus = TaskStatus.CREATED
    current_agent: AgentName | None = None
    agents: dict[str, AgentRecord] = Field(default_factory=dict)
    history: list[HistoryEvent] = Field(default_factory=list)
    review_round: int = 0                  # 审查回退次数
    waiting_for_human: bool = False        # 是否停在检查点等人工
    human_decision: str | None = None      # approve/reject/modify
    human_feedback: str | None = None
    # 审查不通过时要回退到的 Agent
    rollback_to: AgentName | None = None

    def get_agent(self, name: AgentName | str) -> AgentRecord:
        key = name.value if isinstance(name, AgentName) else name
        if key not in self.agents:
            self.agents[key] = AgentRecord(agent=AgentName(key))
        return self.agents[key]


# ====================================================================
# 任务元信息（meta.json）—— 创建后基本不变
# ====================================================================
class TaskMeta(BaseModel):
    task_id: str
    title: str
    problem_type: ProblemType = ProblemType.UNKNOWN
    mode: RunMode = RunMode.AUTO
    problem_path: str = "input/problem.txt"  # 相对任务目录
    # 各 Agent 使用的模型（覆盖全局默认）
    agent_models: dict[str, str] = Field(default_factory=dict)
    created_at: str
    updated_at: str


# ====================================================================
# Task：meta + state 的组合，加上目录路径
# ====================================================================
class Task(BaseModel):
    meta: TaskMeta
    state: TaskState
    # 任务目录绝对路径（运行时填充，不持久化的字段用 exclude）
    dir: str | None = None

    model_config = {"arbitrary_types_allowed": True}

    @property
    def task_dir(self) -> "Any":
        from pathlib import Path
        return Path(self.dir) if self.dir else None
