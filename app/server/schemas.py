"""API 请求/响应模型。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class TaskCreate(BaseModel):
    title: str
    problem_text: str
    mode: str = "auto"            # auto / interactive / hybrid
    problem_type: str = "unknown"  # optimization/evaluation/statistics/mechanism/graph/unknown
    agent_models: dict[str, str] | None = None  # 可选：覆盖各 Agent 模型


class TaskResume(BaseModel):
    decision: str = Field(..., description="approve / modify / reject")
    feedback: str | None = None


class TaskRun(BaseModel):
    pass
