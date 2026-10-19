"""TinyAgent: an AI agent built from scratch in plain Python.

Ported from the notebooks of "An Illustrated Guide to AI Agents"
by Maarten Grootendorst and Jay Alammar.
"""

from tinyagent.agent import TinyAgent
from tinyagent.display import Display, RichDisplay, SilentDisplay
from tinyagent.llm import LLM, EmbeddingModel, Response
from tinyagent.memory import (
    Memory,
    MultimodalMemory,
    RAGMemory,
    SummarizationMemory,
    TrimmingMemory,
)
from tinyagent.planning import NativeReAct, PlanAndSolve, ReAct, Reflexion
from tinyagent.tools import NativeTools, Skills, Tools, tool_to_schema
from tinyagent.trajectory import Step, Trajectory

__version__ = "0.1.0"

__all__ = [
    "TinyAgent",
    "LLM",
    "EmbeddingModel",
    "Response",
    "Memory",
    "TrimmingMemory",
    "SummarizationMemory",
    "RAGMemory",
    "MultimodalMemory",
    "ReAct",
    "NativeReAct",
    "PlanAndSolve",
    "Reflexion",
    "Tools",
    "NativeTools",
    "Skills",
    "tool_to_schema",
    "Display",
    "RichDisplay",
    "SilentDisplay",
    "Trajectory",
    "Step",
]
