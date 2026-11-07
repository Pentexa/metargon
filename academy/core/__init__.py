"""Academy core interfaces."""
from .interfaces import (
    Environment, TaskGenerator, StudentAgent, CurriculumManager,
    ExperienceMemory, Experience, TaskSpec,
)

__all__ = [
    "Environment", "TaskGenerator", "StudentAgent", "CurriculumManager",
    "ExperienceMemory", "Experience", "TaskSpec",
]
