"""OpenCog autonomous multi-agent orchestration for vLLM.

This module provides an OpenCog-based framework for orchestrating multiple
vLLM agents with high-throughput and memory-efficient inference capabilities.
"""

from vllm.opencog.agents import Agent, AgentRegistry
from vllm.opencog.atomspace import AtomSpace, Atom, Link, Node
from vllm.opencog.orchestration import Orchestrator, AgentPool
from vllm.opencog.reasoning import GoalPlanner, PatternMatcher

__all__ = [
    "Agent",
    "AgentRegistry",
    "AtomSpace",
    "Atom",
    "Link",
    "Node",
    "Orchestrator",
    "AgentPool",
    "GoalPlanner",
    "PatternMatcher",
]
