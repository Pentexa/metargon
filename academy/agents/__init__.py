"""Academy agents."""
from .memory import CompositionalMemory, Hypothesis
from .metargon_agent import MetargonAcademyAgent
from .baselines import (
    RandomPolicy, EpisodicPolicy, QLearningPolicy, BayesPlannerPolicy,
    OraclePolicy,
)

__all__ = [
    "CompositionalMemory", "Hypothesis",
    "MetargonAcademyAgent",
    "RandomPolicy", "EpisodicPolicy", "QLearningPolicy",
    "BayesPlannerPolicy", "OraclePolicy",
]
