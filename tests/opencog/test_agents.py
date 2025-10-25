"""Tests for Agent and AgentRegistry."""

import asyncio
import pytest
from vllm.opencog.agents import Agent, AgentState, AgentTask, AgentRegistry
from vllm.opencog.atomspace import AtomSpace


class TestAgent(Agent):
    """Concrete test agent implementation."""
    
    async def process_task(self, task: AgentTask):
        """Simple test task processing."""
        # Simulate some work
        await asyncio.sleep(0.01)
        return f"Processed: {task.description}"


class TestAgentTask:
    """Tests for AgentTask."""
    
    def test_task_creation(self):
        """Test task creation with defaults."""
        task = AgentTask(description="Test task")
        
        assert task.description == "Test task"
        assert task.task_id is not None
        assert task.priority == 0
        assert task.result is None
    
    def test_task_with_data(self):
        """Test task with input data."""
        task = AgentTask(
            description="Process data",
            input_data={"key": "value"},
            priority=5
        )
        
        assert task.input_data["key"] == "value"
        assert task.priority == 5


class TestAgentClass:
    """Tests for Agent base class."""
    
    def test_agent_creation(self):
        """Test agent initialization."""
        atomspace = AtomSpace()
        agent = TestAgent(name="TestAgent", atomspace=atomspace)
        
        assert agent.name == "TestAgent"
        assert agent.state == AgentState.IDLE
        assert agent.atomspace == atomspace
    
    def test_agent_registration_in_atomspace(self):
        """Test that agent registers itself in atomspace."""
        atomspace = AtomSpace()
        agent = TestAgent(atomspace=atomspace)
        
        # Check agent node exists
        agent_nodes = [
            n for n in atomspace.get_all_nodes()
            if n.name.startswith("agent:")
        ]
        assert len(agent_nodes) >= 1
    
    @pytest.mark.asyncio
    async def test_submit_task(self):
        """Test task submission."""
        agent = TestAgent()
        task = AgentTask(description="Test")
        
        task_id = await agent.submit_task(task)
        assert task_id == task.task_id
        assert agent.task_queue.qsize() == 1
    
    @pytest.mark.asyncio
    async def test_process_task(self):
        """Test task processing."""
        agent = TestAgent()
        task = AgentTask(description="Test task")
        
        result = await agent.process_task(task)
        assert result.startswith("Processed:")
    
    @pytest.mark.asyncio
    async def test_agent_run_loop(self):
        """Test agent main loop."""
        atomspace = AtomSpace()
        agent = TestAgent(atomspace=atomspace)
        
        # Start agent in background
        run_task = asyncio.create_task(agent.run())
        
        # Submit a task
        task = AgentTask(description="Loop test")
        await agent.submit_task(task)
        
        # Wait a bit for processing
        await asyncio.sleep(0.1)
        
        # Shutdown agent
        agent.shutdown()
        
        # Wait for run loop to complete
        await asyncio.sleep(0.1)
        if not run_task.done():
            run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass
        
        assert agent.state == AgentState.TERMINATED
    
    def test_agent_metrics(self):
        """Test agent metrics."""
        agent = TestAgent()
        metrics = agent.get_metrics()
        
        assert "tasks_completed" in metrics
        assert "tasks_failed" in metrics
        assert "state" in metrics
        assert metrics["state"] == AgentState.IDLE.value
    
    def test_shutdown(self):
        """Test agent shutdown."""
        agent = TestAgent()
        agent.shutdown()
        
        assert agent.state == AgentState.TERMINATED


class TestAgentRegistry:
    """Tests for AgentRegistry."""
    
    @pytest.mark.asyncio
    async def test_registry_creation(self):
        """Test registry initialization."""
        atomspace = AtomSpace()
        registry = AgentRegistry(atomspace=atomspace)
        
        assert registry.atomspace == atomspace
        assert len(registry.get_all_agents()) == 0
    
    @pytest.mark.asyncio
    async def test_register_agent(self):
        """Test agent registration."""
        registry = AgentRegistry()
        agent = TestAgent(name="Agent1")
        
        agent_id = await registry.register_agent(agent)
        
        assert agent_id == agent.agent_id
        assert len(registry.get_all_agents()) == 1
    
    @pytest.mark.asyncio
    async def test_register_duplicate_agent(self):
        """Test registering the same agent twice."""
        registry = AgentRegistry()
        agent = TestAgent(name="Agent1")
        
        await registry.register_agent(agent)
        agent_id2 = await registry.register_agent(agent)
        
        # Should return same ID, not create duplicate
        assert agent_id2 == agent.agent_id
        assert len(registry.get_all_agents()) == 1
    
    @pytest.mark.asyncio
    async def test_unregister_agent(self):
        """Test agent unregistration."""
        registry = AgentRegistry()
        agent = TestAgent(name="Agent1")
        
        agent_id = await registry.register_agent(agent)
        await registry.unregister_agent(agent_id)
        
        assert len(registry.get_all_agents()) == 0
        assert agent.state == AgentState.TERMINATED
    
    @pytest.mark.asyncio
    async def test_get_agent(self):
        """Test retrieving agent by ID."""
        registry = AgentRegistry()
        agent = TestAgent(name="Agent1")
        
        agent_id = await registry.register_agent(agent)
        retrieved = registry.get_agent(agent_id)
        
        assert retrieved == agent
    
    @pytest.mark.asyncio
    async def test_get_idle_agents(self):
        """Test getting idle agents."""
        registry = AgentRegistry()
        
        agent1 = TestAgent(name="Agent1")
        agent2 = TestAgent(name="Agent2")
        agent2.state = AgentState.PROCESSING
        
        await registry.register_agent(agent1)
        await registry.register_agent(agent2)
        
        idle_agents = registry.get_idle_agents()
        
        assert len(idle_agents) == 1
        assert idle_agents[0] == agent1
    
    @pytest.mark.asyncio
    async def test_shutdown_all(self):
        """Test shutting down all agents."""
        registry = AgentRegistry()
        
        agent1 = TestAgent(name="Agent1")
        agent2 = TestAgent(name="Agent2")
        
        await registry.register_agent(agent1)
        await registry.register_agent(agent2)
        
        await registry.shutdown_all()
        
        assert len(registry.get_all_agents()) == 0
        assert agent1.state == AgentState.TERMINATED
        assert agent2.state == AgentState.TERMINATED
    
    @pytest.mark.asyncio
    async def test_shared_atomspace(self):
        """Test that agents share the same atomspace."""
        atomspace = AtomSpace()
        registry = AgentRegistry(atomspace=atomspace)
        
        agent1 = TestAgent(name="Agent1")
        agent2 = TestAgent(name="Agent2")
        
        await registry.register_agent(agent1)
        await registry.register_agent(agent2)
        
        # Both agents should share the registry's atomspace
        assert agent1.atomspace == atomspace
        assert agent2.atomspace == atomspace
        assert agent1.atomspace == agent2.atomspace
