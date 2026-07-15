from .base import BaseAgent, RunContext, AgentResult
from .analyst import AnalystAgent
from .modeler import ModelerAgent
from .solver import SolverAgent
from .writer import WriterAgent
from .reviewer import ReviewerAgent

AGENTS = {
    "analyst": AnalystAgent,
    "modeler": ModelerAgent,
    "solver": SolverAgent,
    "writer": WriterAgent,
    "reviewer": ReviewerAgent,
}

__all__ = [
    "BaseAgent",
    "RunContext",
    "AgentResult",
    "AnalystAgent",
    "ModelerAgent",
    "SolverAgent",
    "WriterAgent",
    "ReviewerAgent",
    "AGENTS",
]
