"""Orchestration layer for multi-agent coordination.

Provides orchestrator and agent pool for managing task distribution,
load balancing, and agent collaboration.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from vllm.opencog.agents import Agent, AgentRegistry, AgentTask, AgentState
from vllm.opencog.atomspace import AtomSpace

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationPolicy:
    """Configuration for orchestration behavior."""
    
    load_balancing: str = "round_robin"  # round_robin, least_loaded, priority
    max_concurrent_tasks: int = 100
    task_timeout: float = 300.0  # seconds
    enable_collaboration: bool = True
    enable_learning: bool = True


class AgentPool:
    """Pool of agents for high-throughput task processing.
    
    The pool manages agent lifecycle, distributes tasks efficiently,
    and monitors agent performance.
    """
    
    def __init__(
        self,
        registry: AgentRegistry,
        pool_size: int = 4,
        policy: Optional[OrchestrationPolicy] = None
    ):
        """Initialize agent pool.
        
        Args:
            registry: Agent registry
            pool_size: Number of agents in pool
            policy: Orchestration policy
        """
        self.registry = registry
        self.pool_size = pool_size
        self.policy = policy or OrchestrationPolicy()
        self._running = False
        self._agent_tasks: List[asyncio.Task] = []
        self._round_robin_index = 0
    
    async def start(self):
        """Start the agent pool and all agents."""
        if self._running:
            logger.warning("Agent pool already running")
            return
        
        self._running = True
        
        # Start all registered agents
        agents = self.registry.get_all_agents()
        for agent in agents:
            task = asyncio.create_task(agent.run())
            self._agent_tasks.append(task)
        
        logger.info(f"Agent pool started with {len(agents)} agents")
    
    async def stop(self):
        """Stop the agent pool and all agents."""
        if not self._running:
            return
        
        self._running = False
        
        # Shutdown all agents
        await self.registry.shutdown_all()
        
        # Cancel all agent tasks
        for task in self._agent_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._agent_tasks, return_exceptions=True)
        self._agent_tasks.clear()
        
        logger.info("Agent pool stopped")
    
    async def submit_task(self, task: AgentTask) -> str:
        """Submit a task to be processed by an agent in the pool.
        
        Args:
            task: Task to process
            
        Returns:
            Task ID
        """
        agent = self._select_agent()
        
        if agent is None:
            raise RuntimeError("No available agents in pool")
        
        await agent.submit_task(task)
        logger.debug(f"Task {task.task_id} submitted to agent {agent.name}")
        
        return task.task_id
    
    def _select_agent(self) -> Optional[Agent]:
        """Select an agent based on the load balancing policy.
        
        Returns:
            Selected agent or None if no agents available
        """
        agents = self.registry.get_all_agents()
        
        if not agents:
            return None
        
        if self.policy.load_balancing == "round_robin":
            agent = agents[self._round_robin_index % len(agents)]
            self._round_robin_index += 1
            return agent
        
        elif self.policy.load_balancing == "least_loaded":
            # Select agent with smallest queue
            return min(agents, key=lambda a: a.task_queue.qsize())
        
        else:
            # Default to first available idle agent
            idle_agents = self.registry.get_idle_agents()
            if idle_agents:
                return idle_agents[0]
            return agents[0]
    
    def get_pool_metrics(self) -> Dict[str, Any]:
        """Get metrics for the entire pool.
        
        Returns:
            Dictionary of pool metrics
        """
        agents = self.registry.get_all_agents()
        
        total_tasks_completed = sum(
            a.metrics["tasks_completed"] for a in agents
        )
        total_tasks_failed = sum(
            a.metrics["tasks_failed"] for a in agents
        )
        total_processing_time = sum(
            a.metrics["total_processing_time"] for a in agents
        )
        
        return {
            "pool_size": len(agents),
            "total_tasks_completed": total_tasks_completed,
            "total_tasks_failed": total_tasks_failed,
            "total_processing_time": total_processing_time,
            "avg_processing_time": (
                total_processing_time / total_tasks_completed
                if total_tasks_completed > 0 else 0.0
            ),
            "agent_states": {
                state.value: sum(
                    1 for a in agents if a.state == state
                )
                for state in AgentState
            },
        }


class Orchestrator:
    """High-level orchestrator for autonomous multi-agent system.
    
    Coordinates multiple agent pools, manages shared knowledge through
    AtomSpace, and implements autonomous reasoning and adaptation.
    """
    
    def __init__(
        self,
        atomspace: Optional[AtomSpace] = None,
        policy: Optional[OrchestrationPolicy] = None
    ):
        """Initialize orchestrator.
        
        Args:
            atomspace: Shared knowledge space
            policy: Orchestration policy
        """
        self.atomspace = atomspace or AtomSpace()
        self.policy = policy or OrchestrationPolicy()
        self.registry = AgentRegistry(atomspace=self.atomspace)
        self.pools: Dict[str, AgentPool] = {}
        self._running = False
        
        logger.info("Orchestrator initialized")
    
    async def create_pool(
        self,
        pool_id: str,
        agents: List[Agent],
        policy: Optional[OrchestrationPolicy] = None
    ) -> AgentPool:
        """Create a new agent pool.
        
        Args:
            pool_id: Unique identifier for the pool
            agents: List of agents to add to pool
            policy: Pool-specific policy (defaults to orchestrator policy)
            
        Returns:
            Created agent pool
        """
        if pool_id in self.pools:
            raise ValueError(f"Pool {pool_id} already exists")
        
        # Register all agents
        for agent in agents:
            await self.registry.register_agent(agent)
        
        # Create pool
        pool = AgentPool(
            registry=self.registry,
            pool_size=len(agents),
            policy=policy or self.policy
        )
        
        self.pools[pool_id] = pool
        logger.info(f"Created pool {pool_id} with {len(agents)} agents")
        
        return pool
    
    async def start(self):
        """Start the orchestrator and all pools."""
        if self._running:
            logger.warning("Orchestrator already running")
            return
        
        self._running = True
        
        # Start all pools
        for pool_id, pool in self.pools.items():
            await pool.start()
            logger.info(f"Started pool {pool_id}")
        
        logger.info("Orchestrator started")
    
    async def stop(self):
        """Stop the orchestrator and all pools."""
        if not self._running:
            return
        
        self._running = False
        
        # Stop all pools
        for pool_id, pool in self.pools.items():
            await pool.stop()
            logger.info(f"Stopped pool {pool_id}")
        
        logger.info("Orchestrator stopped")
    
    async def submit_task(
        self,
        task: AgentTask,
        pool_id: Optional[str] = None
    ) -> str:
        """Submit a task to be processed.
        
        Args:
            task: Task to process
            pool_id: Specific pool to submit to (optional)
            
        Returns:
            Task ID
        """
        if pool_id:
            if pool_id not in self.pools:
                raise ValueError(f"Pool {pool_id} not found")
            pool = self.pools[pool_id]
        else:
            # Select first available pool
            if not self.pools:
                raise RuntimeError("No pools available")
            pool = next(iter(self.pools.values()))
        
        return await pool.submit_task(task)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics.
        
        Returns:
            Dictionary of metrics
        """
        pool_metrics = {
            pool_id: pool.get_pool_metrics()
            for pool_id, pool in self.pools.items()
        }
        
        total_completed = sum(
            m["total_tasks_completed"] for m in pool_metrics.values()
        )
        total_failed = sum(
            m["total_tasks_failed"] for m in pool_metrics.values()
        )
        
        return {
            "running": self._running,
            "num_pools": len(self.pools),
            "total_agents": sum(
                m["pool_size"] for m in pool_metrics.values()
            ),
            "total_tasks_completed": total_completed,
            "total_tasks_failed": total_failed,
            "atomspace_size": self.atomspace.size(),
            "pools": pool_metrics,
        }


__all__ = ["Orchestrator", "AgentPool", "OrchestrationPolicy"]
