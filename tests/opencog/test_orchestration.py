"""Tests for orchestration components."""

import asyncio
import pytest
from vllm.opencog.agents import Agent, AgentTask, AgentState
from vllm.opencog.atomspace import AtomSpace
from vllm.opencog.orchestration import (
    Orchestrator,
    AgentPool,
    OrchestrationPolicy,
)


class TestAgent(Agent):
    """Test agent implementation."""
    
    async def process_task(self, task: AgentTask):
        await asyncio.sleep(0.01)
        return f"Result: {task.description}"


class TestOrchestrationPolicy:
    """Tests for OrchestrationPolicy."""
    
    def test_default_policy(self):
        """Test default policy creation."""
        policy = OrchestrationPolicy()
        
        assert policy.load_balancing == "round_robin"
        assert policy.max_concurrent_tasks == 100
        assert policy.enable_collaboration is True
    
    def test_custom_policy(self):
        """Test custom policy configuration."""
        policy = OrchestrationPolicy(
            load_balancing="least_loaded",
            max_concurrent_tasks=50,
            task_timeout=600.0
        )
        
        assert policy.load_balancing == "least_loaded"
        assert policy.max_concurrent_tasks == 50
        assert policy.task_timeout == 600.0


class TestAgentPool:
    """Tests for AgentPool."""
    
    @pytest.mark.asyncio
    async def test_pool_creation(self):
        """Test pool creation."""
        from vllm.opencog.agents import AgentRegistry
        
        registry = AgentRegistry()
        pool = AgentPool(registry=registry, pool_size=3)
        
        assert pool.pool_size == 3
        assert pool._running is False
    
    @pytest.mark.asyncio
    async def test_start_stop_pool(self):
        """Test starting and stopping pool."""
        from vllm.opencog.agents import AgentRegistry
        
        registry = AgentRegistry()
        agents = [TestAgent(name=f"Agent-{i}") for i in range(2)]
        
        for agent in agents:
            await registry.register_agent(agent)
        
        pool = AgentPool(registry=registry, pool_size=2)
        
        await pool.start()
        assert pool._running is True
        
        await pool.stop()
        assert pool._running is False
    
    @pytest.mark.asyncio
    async def test_submit_task_round_robin(self):
        """Test task submission with round-robin."""
        from vllm.opencog.agents import AgentRegistry
        
        registry = AgentRegistry()
        agents = [TestAgent(name=f"Agent-{i}") for i in range(3)]
        
        for agent in agents:
            await registry.register_agent(agent)
        
        policy = OrchestrationPolicy(load_balancing="round_robin")
        pool = AgentPool(registry=registry, pool_size=3, policy=policy)
        
        await pool.start()
        
        # Submit tasks
        task1 = AgentTask(description="Task1")
        task2 = AgentTask(description="Task2")
        
        await pool.submit_task(task1)
        await pool.submit_task(task2)
        
        await pool.stop()
    
    @pytest.mark.asyncio
    async def test_pool_metrics(self):
        """Test pool metrics."""
        from vllm.opencog.agents import AgentRegistry
        
        registry = AgentRegistry()
        agents = [TestAgent(name=f"Agent-{i}") for i in range(2)]
        
        for agent in agents:
            await registry.register_agent(agent)
        
        pool = AgentPool(registry=registry, pool_size=2)
        metrics = pool.get_pool_metrics()
        
        assert metrics["pool_size"] == 2
        assert "total_tasks_completed" in metrics
        assert "agent_states" in metrics


class TestOrchestrator:
    """Tests for Orchestrator."""
    
    @pytest.mark.asyncio
    async def test_orchestrator_creation(self):
        """Test orchestrator initialization."""
        atomspace = AtomSpace()
        orchestrator = Orchestrator(atomspace=atomspace)
        
        assert orchestrator.atomspace == atomspace
        assert len(orchestrator.pools) == 0
    
    @pytest.mark.asyncio
    async def test_create_pool(self):
        """Test creating an agent pool."""
        orchestrator = Orchestrator()
        agents = [TestAgent(name=f"Agent-{i}") for i in range(3)]
        
        pool = await orchestrator.create_pool("test_pool", agents)
        
        assert "test_pool" in orchestrator.pools
        assert pool.pool_size == 3
    
    @pytest.mark.asyncio
    async def test_create_duplicate_pool(self):
        """Test creating a pool with duplicate ID."""
        orchestrator = Orchestrator()
        agents1 = [TestAgent(name="Agent-1")]
        agents2 = [TestAgent(name="Agent-2")]
        
        await orchestrator.create_pool("pool1", agents1)
        
        with pytest.raises(ValueError, match="already exists"):
            await orchestrator.create_pool("pool1", agents2)
    
    @pytest.mark.asyncio
    async def test_start_stop_orchestrator(self):
        """Test starting and stopping orchestrator."""
        orchestrator = Orchestrator()
        agents = [TestAgent(name=f"Agent-{i}") for i in range(2)]
        
        await orchestrator.create_pool("pool1", agents)
        
        await orchestrator.start()
        assert orchestrator._running is True
        
        await orchestrator.stop()
        assert orchestrator._running is False
    
    @pytest.mark.asyncio
    async def test_submit_task_to_pool(self):
        """Test submitting task to specific pool."""
        orchestrator = Orchestrator()
        agents = [TestAgent(name=f"Agent-{i}") for i in range(2)]
        
        await orchestrator.create_pool("pool1", agents)
        await orchestrator.start()
        
        task = AgentTask(description="Test task")
        task_id = await orchestrator.submit_task(task, pool_id="pool1")
        
        assert task_id == task.task_id
        
        await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_submit_task_to_nonexistent_pool(self):
        """Test submitting to non-existent pool."""
        orchestrator = Orchestrator()
        
        task = AgentTask(description="Test")
        
        with pytest.raises(ValueError, match="not found"):
            await orchestrator.submit_task(task, pool_id="nonexistent")
    
    @pytest.mark.asyncio
    async def test_submit_task_without_pool_id(self):
        """Test submitting task without specifying pool."""
        orchestrator = Orchestrator()
        agents = [TestAgent(name="Agent-1")]
        
        await orchestrator.create_pool("pool1", agents)
        await orchestrator.start()
        
        task = AgentTask(description="Test")
        task_id = await orchestrator.submit_task(task)
        
        assert task_id is not None
        
        await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_orchestrator_metrics(self):
        """Test orchestrator metrics."""
        orchestrator = Orchestrator()
        agents1 = [TestAgent(name=f"A-{i}") for i in range(2)]
        agents2 = [TestAgent(name=f"B-{i}") for i in range(3)]
        
        await orchestrator.create_pool("pool1", agents1)
        await orchestrator.create_pool("pool2", agents2)
        
        metrics = orchestrator.get_metrics()
        
        assert metrics["num_pools"] == 2
        assert metrics["total_agents"] == 5
        assert "pool1" in metrics["pools"]
        assert "pool2" in metrics["pools"]
    
    @pytest.mark.asyncio
    async def test_shared_atomspace_across_pools(self):
        """Test that all pools share the orchestrator's atomspace."""
        atomspace = AtomSpace()
        orchestrator = Orchestrator(atomspace=atomspace)
        
        agents1 = [TestAgent(name="A1")]
        agents2 = [TestAgent(name="A2")]
        
        await orchestrator.create_pool("pool1", agents1)
        await orchestrator.create_pool("pool2", agents2)
        
        # All agents should share the same atomspace
        assert agents1[0].atomspace == atomspace
        assert agents2[0].atomspace == atomspace
