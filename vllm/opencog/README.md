# OpenCog Multi-Agent Orchestration for vLLM

This directory contains the OpenCog autonomous multi-agent orchestration system integrated with vLLM for high-throughput and memory-efficient LLM inference.

## Overview

The OpenCog orchestration module extends vLLM with autonomous multi-agent capabilities, enabling:

- **Distributed Agent Architecture**: Multiple specialized agents working collaboratively
- **High-Throughput Serving**: Efficient parallel processing and load balancing
- **Shared Knowledge**: Memory-efficient AtomSpace for inter-agent communication
- **Autonomous Reasoning**: Goal-oriented planning and pattern recognition
- **Flexible Configuration**: YAML-based declarative setup

## Quick Start

### Basic Usage

```python
import asyncio
from vllm.opencog import Orchestrator, AtomSpace
from vllm.opencog.agents.concrete import LLMInferenceAgent
from vllm.opencog.agents import AgentTask

async def main():
    # Initialize orchestrator with shared knowledge space
    atomspace = AtomSpace()
    orchestrator = Orchestrator(atomspace=atomspace)
    
    # Create agents (in production, provide vLLM engines)
    agents = [
        LLMInferenceAgent(name=f"Agent-{i}", atomspace=atomspace)
        for i in range(4)
    ]
    
    # Create and start agent pool
    await orchestrator.create_pool("main_pool", agents)
    await orchestrator.start()
    
    # Submit inference task
    task = AgentTask(
        description="Generate text",
        input_data={
            "prompt": "Explain quantum computing in simple terms",
            "sampling_params": {"temperature": 0.7, "max_tokens": 512}
        }
    )
    
    task_id = await orchestrator.submit_task(task)
    
    # Monitor performance
    metrics = orchestrator.get_metrics()
    print(f"Tasks completed: {metrics['total_tasks_completed']}")
    
    await orchestrator.stop()

asyncio.run(main())
```

### Configuration-Based Setup

Create a YAML configuration file:

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
      - agent_type: "LLMInferenceAgent"
        name: "agent_2"
        model: "meta-llama/Llama-3.1-8B"
```

## Architecture

### Core Components

1. **AtomSpace** (`vllm.opencog.atomspace`)
   - Hypergraph knowledge representation
   - Thread-safe atom storage
   - Pattern matching and querying

2. **Agents** (`vllm.opencog.agents`)
   - Base Agent class with vLLM integration
   - Concrete agent types for specialized tasks
   - AgentRegistry for lifecycle management

3. **Orchestration** (`vllm.opencog.orchestration`)
   - Orchestrator for multi-pool coordination
   - AgentPool with load balancing
   - Task distribution and metrics

4. **Reasoning** (`vllm.opencog.reasoning`)
   - GoalPlanner for task decomposition
   - PatternMatcher for learning
   - Autonomous decision making

## Features

### Load Balancing Strategies

- **Round-robin**: Distributes tasks evenly across agents
- **Least-loaded**: Routes to agent with smallest queue
- **Priority-based**: Processes high-priority tasks first

### Knowledge Sharing

All agents share a common AtomSpace, enabling:
- Collaborative learning from past tasks
- Pattern recognition across agent experiences
- Efficient memory usage through shared representation

### Monitoring

Real-time metrics available:
- Tasks completed/failed per agent and pool
- Average processing time
- Agent states (idle, processing, waiting, error)
- AtomSpace knowledge base size

## Agent Types

### LLMInferenceAgent
General-purpose text generation with customizable sampling parameters.

### SummarizationAgent
Specialized for document summarization with optimized prompts.

### QuestionAnsweringAgent
Handles Q&A tasks with optional context.

### CoordinatorAgent
Coordinates multi-step tasks across other agents.

## Performance

### High-Throughput Design
- Asynchronous task processing
- Parallel agent execution
- Non-blocking I/O operations
- Continuous batching via vLLM

### Memory Efficiency
- Single shared AtomSpace
- Reference-counted atoms
- No data duplication across agents

## Examples

See `examples/opencog/` for:
- `basic_orchestration.py` - Complete working example
- `orchestration_config.yaml` - Sample configuration
- `README.md` - Detailed documentation

## Testing

Run validation tests:

```bash
cd tests/opencog
python comprehensive_validation.py
```

Run unit tests (requires pytest):

```bash
pytest tests/opencog/
```

## API Reference

### Orchestrator

```python
orchestrator = Orchestrator(atomspace=None, policy=None)
await orchestrator.create_pool(pool_id, agents, policy=None)
await orchestrator.start()
await orchestrator.stop()
await orchestrator.submit_task(task, pool_id=None)
metrics = orchestrator.get_metrics()
```

### Agent

```python
agent = Agent(agent_id=None, name=None, atomspace=None, engine=None)
await agent.submit_task(task)
await agent.process_task(task)  # Override in subclasses
result = await agent.generate_text(prompt, sampling_params)
agent.shutdown()
metrics = agent.get_metrics()
```

### AtomSpace

```python
atomspace = AtomSpace()
node = atomspace.add_node(name, **kwargs)
link = atomspace.add_link(name, outgoing, **kwargs)
atom = atomspace.get_atom(name, atom_type)
results = atomspace.pattern_match(pattern)
```

## Advanced Usage

### Custom Agent Types

```python
from vllm.opencog.agents import Agent, AgentTask

class CustomAgent(Agent):
    async def process_task(self, task: AgentTask):
        # Your custom logic
        prompt = task.input_data["prompt"]
        result = await self.generate_text(prompt)
        
        # Store in AtomSpace
        self.atomspace.add_node(
            f"result:{task.task_id}",
            metadata={"result": result}
        )
        
        return result
```

### Goal-Oriented Planning

```python
from vllm.opencog.reasoning import GoalPlanner, Goal

planner = GoalPlanner(atomspace)

goal = Goal(
    goal_id="analyze_docs",
    description="Analyze and summarize documents",
    priority=0.8
)

planner.add_goal(goal)
plan = planner.create_plan(goal.goal_id)

# Execute plan steps
while True:
    action = planner.get_next_action(plan.plan_id)
    if not action:
        break
    # Execute action...
    planner.mark_step_completed(plan.plan_id)
```

## Contributing

Contributions welcome! Areas for enhancement:
- Additional specialized agent types
- Advanced planning algorithms
- Distributed AtomSpace for multi-node deployments
- Enhanced pattern matching
- Integration with external knowledge bases

## License

Apache 2.0 - See LICENSE for details

## Citation

If you use this orchestration system in your research, please cite the vLLM paper:

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```
