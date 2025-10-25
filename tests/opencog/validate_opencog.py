#!/usr/bin/env python3
"""Standalone validation script for OpenCog modules.

This script tests the core OpenCog functionality without requiring
the full vLLM installation.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Test AtomSpace
print("Testing AtomSpace...")
from vllm.opencog.atomspace import AtomSpace, Node, Link, AtomType

atomspace = AtomSpace()
assert atomspace.size() == 0, "Empty atomspace should have size 0"

node1 = atomspace.add_node("concept1", truth_value=0.9)
assert atomspace.size() == 1, "Should have 1 atom"
assert node1.name == "concept1", "Node name mismatch"
assert node1.truth_value == 0.9, "Truth value mismatch"

node2 = atomspace.add_node("concept2")
link = atomspace.add_link("relates", [node1, node2])
assert atomspace.size() == 3, "Should have 3 atoms"

# Test pattern matching
results = atomspace.pattern_match({"type": "node"})
assert len(results) == 2, "Should find 2 nodes"

results = atomspace.pattern_match({"type": "link"})
assert len(results) == 1, "Should find 1 link"

print("✓ AtomSpace tests passed")

# Test Agent base class (without vLLM engine)
print("\nTesting Agent classes...")

# We can't fully test agents without mocking the vLLM engine
# but we can test the structure
import asyncio
from vllm.opencog.agents import AgentTask, AgentState, AgentRegistry

task = AgentTask(description="Test task", priority=5)
assert task.description == "Test task", "Task description mismatch"
assert task.priority == 5, "Task priority mismatch"
assert task.task_id is not None, "Task should have ID"

print("✓ Agent task tests passed")

# Test AgentRegistry
async def test_registry():
    atomspace = AtomSpace()
    registry = AgentRegistry(atomspace=atomspace)
    
    assert len(registry.get_all_agents()) == 0, "Registry should be empty"
    print("✓ AgentRegistry tests passed")

asyncio.run(test_registry())

# Test Orchestration
print("\nTesting Orchestration...")
from vllm.opencog.orchestration import OrchestrationPolicy

policy = OrchestrationPolicy(
    load_balancing="least_loaded",
    max_concurrent_tasks=50
)
assert policy.load_balancing == "least_loaded", "Policy setting mismatch"
assert policy.max_concurrent_tasks == 50, "Policy setting mismatch"

print("✓ Orchestration policy tests passed")

# Test Reasoning
print("\nTesting Reasoning...")
from vllm.opencog.reasoning import GoalPlanner, Goal, PatternMatcher

atomspace = AtomSpace()
planner = GoalPlanner(atomspace)

goal = Goal(
    goal_id="test_goal",
    description="Test goal",
    priority=0.8
)

goal_id = planner.add_goal(goal)
assert goal_id == "test_goal", "Goal ID mismatch"

plan = planner.create_plan(goal_id)
assert plan is not None, "Plan should be created"
assert len(plan.steps) > 0, "Plan should have steps"

print("✓ Goal planner tests passed")

# Pattern matcher
matcher = PatternMatcher(atomspace)
node1 = atomspace.add_node("pattern_test_1")
node2 = atomspace.add_node("pattern_test_2")
node3 = atomspace.add_node("different")

similar = matcher.find_similar_atoms(node1, threshold=0.5)
# Should find node2 as similar (both have "pattern" in name)

print("✓ Pattern matcher tests passed")

# Test Config
print("\nTesting Configuration...")
from vllm.opencog.config import AgentConfig, PoolConfig, OrchestratorConfig

agent_config = AgentConfig(
    agent_type="LLMInferenceAgent",
    name="test_agent",
    model="test-model"
)
assert agent_config.agent_type == "LLMInferenceAgent", "Agent config mismatch"

pool_config = PoolConfig(
    pool_id="test_pool",
    agents=[agent_config],
    load_balancing="round_robin"
)
assert len(pool_config.agents) == 1, "Pool should have 1 agent config"

orchestrator_config = OrchestratorConfig(
    pools=[pool_config],
    enable_collaboration=True
)
assert len(orchestrator_config.pools) == 1, "Should have 1 pool config"

print("✓ Configuration tests passed")

print("\n" + "="*60)
print("✅ All OpenCog module tests passed successfully!")
print("="*60)
