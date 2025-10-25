"""Agent framework for autonomous vLLM orchestration.

Provides base classes and utilities for creating autonomous agents that
can interact with vLLM engines and collaborate through the AtomSpace.
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.opencog.atomspace import AtomSpace, Node
from vllm.sampling_params import SamplingParams

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """States an agent can be in."""
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING = "waiting"
    ERROR = "error"
    TERMINATED = "terminated"


@dataclass
class AgentTask:
    """Task representation for agent processing."""
    
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    input_data: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None


class Agent(ABC):
    """Base class for autonomous agents.
    
    Agents can process tasks, interact with vLLM engines, and share
    knowledge through the AtomSpace.
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        atomspace: Optional[AtomSpace] = None,
        engine: Optional[AsyncLLMEngine] = None,
    ):
        """Initialize an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name
            atomspace: Shared knowledge space
            engine: vLLM engine for inference
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.name = name or f"Agent-{self.agent_id[:8]}"
        self.atomspace = atomspace or AtomSpace()
        self.engine = engine
        self.state = AgentState.IDLE
        self.task_queue: asyncio.Queue[AgentTask] = asyncio.Queue()
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_processing_time": 0.0,
        }
        
        # Register agent in atomspace
        self._register_in_atomspace()
        
        logger.info(f"Agent {self.name} ({self.agent_id}) initialized")
    
    def _register_in_atomspace(self):
        """Register this agent in the AtomSpace."""
        agent_node = self.atomspace.add_node(
            f"agent:{self.agent_id}",
            metadata={
                "name": self.name,
                "type": self.__class__.__name__,
                "state": self.state.value,
            }
        )
        return agent_node
    
    async def submit_task(self, task: AgentTask) -> str:
        """Submit a task to this agent's queue.
        
        Args:
            task: The task to process
            
        Returns:
            Task ID
        """
        await self.task_queue.put(task)
        logger.debug(f"Agent {self.name} received task {task.task_id}")
        return task.task_id
    
    @abstractmethod
    async def process_task(self, task: AgentTask) -> Any:
        """Process a task.
        
        Args:
            task: The task to process
            
        Returns:
            Processing result
        """
        pass
    
    async def run(self):
        """Main agent loop - continuously process tasks from queue."""
        logger.info(f"Agent {self.name} starting main loop")
        
        while self.state != AgentState.TERMINATED:
            try:
                # Get task from queue with timeout
                task = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=1.0
                )
                
                self.state = AgentState.PROCESSING
                logger.debug(f"Agent {self.name} processing task {task.task_id}")
                
                # Process the task
                import time
                start_time = time.time()
                
                try:
                    result = await self.process_task(task)
                    task.result = result
                    self.metrics["tasks_completed"] += 1
                    
                except Exception as e:
                    logger.error(f"Agent {self.name} task failed: {e}")
                    task.error = str(e)
                    self.metrics["tasks_failed"] += 1
                    self.state = AgentState.ERROR
                
                finally:
                    processing_time = time.time() - start_time
                    self.metrics["total_processing_time"] += processing_time
                    self.state = AgentState.IDLE
                    
                    # Store task result in atomspace
                    self._store_task_result(task)
                
            except asyncio.TimeoutError:
                # No tasks available, continue waiting
                if self.state != AgentState.ERROR:
                    self.state = AgentState.IDLE
                continue
                
            except Exception as e:
                logger.error(f"Agent {self.name} error in main loop: {e}")
                self.state = AgentState.ERROR
    
    def _store_task_result(self, task: AgentTask):
        """Store task result in the AtomSpace."""
        task_node = self.atomspace.add_node(
            f"task:{task.task_id}",
            metadata={
                "description": task.description,
                "agent_id": self.agent_id,
                "result": task.result,
                "error": task.error,
                "priority": task.priority,
            }
        )
        
        # Link agent to task
        agent_node = self.atomspace.get_atom(
            f"agent:{self.agent_id}",
            self.atomspace.get_all_nodes().__iter__().__next__().atom_type
        )
        if agent_node:
            self.atomspace.add_link(
                "processed",
                [agent_node, task_node]
            )
    
    async def generate_text(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None
    ) -> str:
        """Generate text using the vLLM engine.
        
        Args:
            prompt: Input prompt
            sampling_params: Generation parameters
            
        Returns:
            Generated text
        """
        if not self.engine:
            raise RuntimeError(f"Agent {self.name} has no vLLM engine")
        
        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=512
            )
        
        # Generate with the engine
        request_id = str(uuid.uuid4())
        results = []
        
        async for output in self.engine.generate(
            prompt,
            sampling_params,
            request_id
        ):
            results.append(output)
        
        if results:
            final_output = results[-1]
            return final_output.outputs[0].text
        
        return ""
    
    def shutdown(self):
        """Shutdown the agent."""
        logger.info(f"Agent {self.name} shutting down")
        self.state = AgentState.TERMINATED
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        return {
            **self.metrics,
            "state": self.state.value,
            "queue_size": self.task_queue.qsize(),
        }


class AgentRegistry:
    """Registry for managing multiple agents."""
    
    def __init__(self, atomspace: Optional[AtomSpace] = None):
        """Initialize the registry.
        
        Args:
            atomspace: Shared knowledge space for all agents
        """
        self.atomspace = atomspace or AtomSpace()
        self._agents: Dict[str, Agent] = {}
        self._lock = asyncio.Lock()
    
    async def register_agent(self, agent: Agent) -> str:
        """Register an agent.
        
        Args:
            agent: The agent to register
            
        Returns:
            Agent ID
        """
        async with self._lock:
            if agent.agent_id in self._agents:
                logger.warning(f"Agent {agent.agent_id} already registered")
                return agent.agent_id
            
            # Share atomspace with agent
            agent.atomspace = self.atomspace
            agent._register_in_atomspace()
            
            self._agents[agent.agent_id] = agent
            logger.info(f"Registered agent {agent.name} ({agent.agent_id})")
            
            return agent.agent_id
    
    async def unregister_agent(self, agent_id: str):
        """Unregister an agent.
        
        Args:
            agent_id: ID of the agent to remove
        """
        async with self._lock:
            if agent_id in self._agents:
                agent = self._agents[agent_id]
                agent.shutdown()
                del self._agents[agent_id]
                logger.info(f"Unregistered agent {agent_id}")
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent if found, None otherwise
        """
        return self._agents.get(agent_id)
    
    def get_all_agents(self) -> List[Agent]:
        """Get all registered agents."""
        return list(self._agents.values())
    
    def get_idle_agents(self) -> List[Agent]:
        """Get all idle agents."""
        return [
            agent for agent in self._agents.values()
            if agent.state == AgentState.IDLE
        ]
    
    async def shutdown_all(self):
        """Shutdown all agents."""
        async with self._lock:
            for agent in self._agents.values():
                agent.shutdown()
            self._agents.clear()
            logger.info("All agents shut down")


__all__ = ["Agent", "AgentState", "AgentTask", "AgentRegistry"]
