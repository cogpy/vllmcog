# OpenCog Autonomous Multi-Agent vLLM Orchestration

This module provides an OpenCog-based framework for orchestrating multiple vLLM agents with high-throughput and memory-efficient inference capabilities.

## Overview

The OpenCog orchestration system enables:

- **Autonomous Multi-Agent Architecture**: Multiple specialized agents working collaboratively
- **High-Throughput Inference**: Efficient task distribution and parallel processing
- **Memory-Efficient Serving**: Shared knowledge representation through AtomSpace
- **Intelligent Coordination**: Goal-oriented planning and pattern recognition
- **Scalable Agent Pools**: Dynamic load balancing and resource management

## Architecture

### Core Components

1. **AtomSpace** (`vllm.opencog.atomspace`)
   - Hypergraph knowledge representation system
   - Stores relationships between concepts (Nodes and Links)
   - Enables pattern matching and reasoning
   - Shared across all agents for knowledge exchange

2. **Agents** (`vllm.opencog.agents`)
   - Base `Agent` class for autonomous task processing
   - Concrete implementations:
     - `LLMInferenceAgent`: General text generation
     - `SummarizationAgent`: Text summarization
     - `QuestionAnsweringAgent`: Q&A tasks
     - `CoordinatorAgent`: Multi-step task coordination
   - `AgentRegistry`: Central agent management

3. **Orchestration** (`vllm.opencog.orchestration`)
   - `Orchestrator`: High-level multi-pool coordinator
   - `AgentPool`: Pool of agents with load balancing
   - `OrchestrationPolicy`: Configuration for orchestration behavior

4. **Reasoning** (`vllm.opencog.reasoning`)
   - `GoalPlanner`: Decomposes goals into executable plans
   - `PatternMatcher`: Identifies patterns and similarities
   - Supports autonomous learning and adaptation

## Quick Start

### Installation

The OpenCog orchestration module is included with vLLM:

```bash
pip install vllm
```

### Basic Usage

```python
import asyncio
from vllm.opencog import Orchestrator, AtomSpace
from vllm.opencog.agents.concrete import LLMInferenceAgent
from vllm.opencog.orchestration import OrchestrationPolicy

async def main():
    # Create shared knowledge space
    atomspace = AtomSpace()
    
    # Initialize orchestrator
    orchestrator = Orchestrator(
        atomspace=atomspace,
        policy=OrchestrationPolicy(
            load_balancing="least_loaded",
            enable_collaboration=True
        )
    )
    
    # Create agents
    agents = [
        LLMInferenceAgent(name=f"Agent-{i}", atomspace=atomspace)
        for i in range(4)
    ]
    
    # Create agent pool
    await orchestrator.create_pool("main_pool", agents)
    
    # Start orchestrator
    await orchestrator.start()
    
    # Submit tasks
    from vllm.opencog.agents import AgentTask
    
    task = AgentTask(
        description="Generate text",
        input_data={"prompt": "Hello, world!"}
    )
    
    task_id = await orchestrator.submit_task(task)
    
    # Get metrics
    metrics = orchestrator.get_metrics()
    print(f"Tasks completed: {metrics['total_tasks_completed']}")
    
    # Cleanup
    await orchestrator.stop()

asyncio.run(main())
```

### Configuration-Based Setup

Use YAML configuration for declarative orchestration:

```yaml
# config.yaml
enable_collaboration: true
enable_learning: true

pools:
  - pool_id: "inference_pool"
    load_balancing: "least_loaded"
    agents:
      - agent_type: "LLMInferenceAgent"
        name: "agent_1"
        model: "meta-llama/Llama-3.1-8B"
```

Load and use:

```python
from vllm.opencog.config import ConfigLoader

config = ConfigLoader.load_from_file("config.yaml")
# Use config to initialize orchestrator
```

## Examples

See the `examples/opencog/` directory for complete examples:

- `basic_orchestration.py`: Simple multi-agent setup
- `orchestration_config.yaml`: Example configuration file

## Features

### High-Throughput Inference

- **Parallel Processing**: Multiple agents process tasks concurrently
- **Load Balancing**: Distributes tasks based on agent availability
  - Round-robin
  - Least-loaded
  - Priority-based
- **Continuous Batching**: Agents can process multiple requests efficiently

### Memory-Efficient Design

- **Shared AtomSpace**: Single knowledge base shared across all agents
- **Reference Counting**: Efficient memory management for atoms
- **Pattern Deduplication**: Avoids storing redundant information

### Autonomous Capabilities

- **Goal-Oriented Planning**: Automatically decomposes complex goals
- **Pattern Recognition**: Learns from experience and identifies patterns
- **Self-Optimization**: Adapts behavior based on performance metrics

### Monitoring and Metrics

Track orchestration performance:

```python
metrics = orchestrator.get_metrics()
# {
#   'running': True,
#   'num_pools': 2,
#   'total_agents': 6,
#   'total_tasks_completed': 150,
#   'atomspace_size': 450,
#   'pools': {...}
# }
```

## API Reference

### Orchestrator

```python
class Orchestrator:
    async def create_pool(pool_id, agents, policy=None)
    async def start()
    async def stop()
    async def submit_task(task, pool_id=None)
    def get_metrics()
```

### Agent

```python
class Agent:
    async def submit_task(task)
    async def process_task(task)  # Override in subclasses
    async def generate_text(prompt, sampling_params)
    def shutdown()
    def get_metrics()
```

### AtomSpace

```python
class AtomSpace:
    def add_node(name, **kwargs)
    def add_link(name, outgoing, **kwargs)
    def get_atom(name, atom_type)
    def pattern_match(pattern)
    def size()
```

## Advanced Topics

### Custom Agent Types

Create specialized agents by subclassing `Agent`:

```python
from vllm.opencog.agents import Agent, AgentTask

class CustomAgent(Agent):
    async def process_task(self, task: AgentTask):
        # Your custom logic here
        prompt = task.input_data["prompt"]
        result = await self.generate_text(prompt)
        return result
```

### Goal Planning

Use the GoalPlanner for complex multi-step tasks:

```python
from vllm.opencog.reasoning import GoalPlanner, Goal

planner = GoalPlanner(atomspace)

goal = Goal(
    goal_id="complex_task",
    description="Analyze and summarize documents",
    priority=0.8
)

planner.add_goal(goal)
plan = planner.create_plan(goal.goal_id)
```

### Pattern Matching

Leverage the PatternMatcher for intelligent behavior:

```python
from vllm.opencog.reasoning import PatternMatcher

matcher = PatternMatcher(atomspace)

# Find similar concepts
similar = matcher.find_similar_atoms(atom, threshold=0.7)

# Learn patterns
matcher.learn_pattern("question_pattern", example_atoms)
```

## Performance Considerations

- **Agent Pool Size**: Balance between parallelism and resource usage
- **Load Balancing Strategy**: Choose based on workload characteristics
- **Task Timeout**: Set appropriate timeouts for your use case
- **AtomSpace Size**: Monitor and clean up periodically if needed

## Contributing

Contributions are welcome! Areas for enhancement:

- Additional agent types for specialized tasks
- Advanced planning algorithms
- Distributed AtomSpace for multi-node setups
- Enhanced pattern matching capabilities
- Integration with external knowledge bases

## License

Apache 2.0 - See LICENSE file for details
