#!/usr/bin/env python3
"""Example usage of OpenCog multi-agent orchestration with vLLM.

This example demonstrates:
1. Creating an orchestrator with multiple agent pools
2. Submitting tasks to agents
3. Monitoring agent performance
4. Using the AtomSpace for knowledge sharing
"""

import asyncio
import logging
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.opencog import (
    Orchestrator,
    AtomSpace,
    AgentRegistry,
    AgentTask,
)
from vllm.opencog.agents.concrete import (
    LLMInferenceAgent,
    SummarizationAgent,
    QuestionAnsweringAgent,
)
from vllm.opencog.orchestration import OrchestrationPolicy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main demonstration function."""
    
    logger.info("Initializing OpenCog Multi-Agent Orchestration System")
    
    # Create shared AtomSpace for knowledge representation
    atomspace = AtomSpace()
    
    # Initialize orchestrator
    policy = OrchestrationPolicy(
        load_balancing="least_loaded",
        max_concurrent_tasks=100,
        enable_collaboration=True,
        enable_learning=True,
    )
    
    orchestrator = Orchestrator(atomspace=atomspace, policy=policy)
    
    # Note: In production, you would initialize actual vLLM engines
    # For this example, we'll create agents without engines to demonstrate structure
    logger.info("Creating agent pools...")
    
    # Create inference pool
    inference_agents = [
        LLMInferenceAgent(
            name=f"InferenceAgent-{i}",
            atomspace=atomspace,
            engine=None  # In production: AsyncLLMEngine(...)
        )
        for i in range(2)
    ]
    
    await orchestrator.create_pool(
        pool_id="inference_pool",
        agents=inference_agents,
        policy=policy
    )
    
    # Create specialized pool
    specialized_agents = [
        SummarizationAgent(
            name="Summarizer-1",
            atomspace=atomspace,
            engine=None
        ),
        QuestionAnsweringAgent(
            name="QA-Agent-1",
            atomspace=atomspace,
            engine=None
        ),
    ]
    
    await orchestrator.create_pool(
        pool_id="specialized_pool",
        agents=specialized_agents,
        policy=policy
    )
    
    # Start orchestrator
    logger.info("Starting orchestrator...")
    await orchestrator.start()
    
    # Submit sample tasks (without actual vLLM engines, these won't generate real text)
    logger.info("Submitting tasks...")
    
    # Task 1: Text generation
    task1 = AgentTask(
        description="Generate creative story",
        input_data={
            "prompt": "Once upon a time in a land of AI agents...",
            "sampling_params": {
                "temperature": 0.8,
                "max_tokens": 256
            }
        },
        priority=1
    )
    
    # Task 2: Summarization
    task2 = AgentTask(
        description="Summarize document",
        input_data={
            "text": "This is a long document that needs to be summarized. " * 20
        },
        priority=2
    )
    
    # Task 3: Question answering
    task3 = AgentTask(
        description="Answer question",
        input_data={
            "question": "What is OpenCog?",
            "context": "OpenCog is a project that aims to build an open-source artificial general intelligence framework."
        },
        priority=1
    )
    
    # Submit tasks to appropriate pools
    try:
        task1_id = await orchestrator.submit_task(task1, pool_id="inference_pool")
        logger.info(f"Submitted task 1: {task1_id}")
        
        task2_id = await orchestrator.submit_task(task2, pool_id="specialized_pool")
        logger.info(f"Submitted task 2: {task2_id}")
        
        task3_id = await orchestrator.submit_task(task3, pool_id="specialized_pool")
        logger.info(f"Submitted task 3: {task3_id}")
        
    except Exception as e:
        logger.error(f"Error submitting tasks: {e}")
    
    # Allow some time for processing (in production with real engines)
    await asyncio.sleep(2)
    
    # Get and display metrics
    logger.info("Retrieving orchestrator metrics...")
    metrics = orchestrator.get_metrics()
    
    logger.info("=== Orchestrator Metrics ===")
    logger.info(f"Running: {metrics['running']}")
    logger.info(f"Number of pools: {metrics['num_pools']}")
    logger.info(f"Total agents: {metrics['total_agents']}")
    logger.info(f"AtomSpace size: {metrics['atomspace_size']} atoms")
    
    for pool_id, pool_metrics in metrics['pools'].items():
        logger.info(f"\n--- Pool: {pool_id} ---")
        logger.info(f"Pool size: {pool_metrics['pool_size']}")
        logger.info(f"Tasks completed: {pool_metrics['total_tasks_completed']}")
        logger.info(f"Tasks failed: {pool_metrics['total_tasks_failed']}")
        logger.info(f"Agent states: {pool_metrics['agent_states']}")
    
    # Demonstrate AtomSpace knowledge sharing
    logger.info("\n=== AtomSpace Knowledge ===")
    all_nodes = atomspace.get_all_nodes()
    logger.info(f"Total nodes: {len(all_nodes)}")
    
    # Show some agent nodes
    agent_nodes = [n for n in all_nodes if n.name.startswith("agent:")]
    logger.info(f"Agent nodes: {len(agent_nodes)}")
    for node in agent_nodes[:3]:
        logger.info(f"  - {node.name}: {node.metadata}")
    
    # Cleanup
    logger.info("\nShutting down orchestrator...")
    await orchestrator.stop()
    
    logger.info("Example completed successfully!")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
