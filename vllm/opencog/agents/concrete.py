"""Concrete agent implementations for common tasks."""

import logging
from typing import Any, Dict

from vllm.opencog.agents import Agent, AgentTask
from vllm.sampling_params import SamplingParams

logger = logging.getLogger(__name__)


class LLMInferenceAgent(Agent):
    """Agent specialized for LLM text generation tasks."""
    
    async def process_task(self, task: AgentTask) -> Any:
        """Process a text generation task.
        
        Args:
            task: Task containing 'prompt' and optional 'sampling_params'
            
        Returns:
            Generated text
        """
        prompt = task.input_data.get("prompt", "")
        
        if not prompt:
            raise ValueError("Task must contain 'prompt' in input_data")
        
        # Get sampling params from task or use defaults
        sampling_params_dict = task.input_data.get("sampling_params", {})
        sampling_params = SamplingParams(**sampling_params_dict)
        
        logger.info(f"Agent {self.name} generating text for prompt: {prompt[:50]}...")
        
        # Generate text using vLLM engine
        result = await self.generate_text(prompt, sampling_params)
        
        # Store result in atomspace
        self.atomspace.add_node(
            f"result:{task.task_id}",
            metadata={
                "prompt": prompt,
                "result": result,
                "agent_id": self.agent_id,
            }
        )
        
        return result


class SummarizationAgent(Agent):
    """Agent specialized for text summarization."""
    
    async def process_task(self, task: AgentTask) -> Any:
        """Process a summarization task.
        
        Args:
            task: Task containing 'text' to summarize
            
        Returns:
            Summary text
        """
        text = task.input_data.get("text", "")
        
        if not text:
            raise ValueError("Task must contain 'text' in input_data")
        
        # Create summarization prompt
        prompt = f"Please provide a concise summary of the following text:\n\n{text}\n\nSummary:"
        
        sampling_params = SamplingParams(
            temperature=0.5,
            top_p=0.9,
            max_tokens=256
        )
        
        logger.info(f"Agent {self.name} summarizing text of length {len(text)}")
        
        # Generate summary
        summary = await self.generate_text(prompt, sampling_params)
        
        # Store result
        self.atomspace.add_node(
            f"summary:{task.task_id}",
            metadata={
                "original_length": len(text),
                "summary_length": len(summary),
                "summary": summary,
            }
        )
        
        return summary


class QuestionAnsweringAgent(Agent):
    """Agent specialized for question answering."""
    
    async def process_task(self, task: AgentTask) -> Any:
        """Process a question answering task.
        
        Args:
            task: Task containing 'question' and optional 'context'
            
        Returns:
            Answer text
        """
        question = task.input_data.get("question", "")
        context = task.input_data.get("context", "")
        
        if not question:
            raise ValueError("Task must contain 'question' in input_data")
        
        # Create QA prompt
        if context:
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            prompt = f"Question: {question}\n\nAnswer:"
        
        sampling_params = SamplingParams(
            temperature=0.3,
            top_p=0.9,
            max_tokens=512
        )
        
        logger.info(f"Agent {self.name} answering question: {question[:50]}...")
        
        # Generate answer
        answer = await self.generate_text(prompt, sampling_params)
        
        # Store QA pair in atomspace
        question_node = self.atomspace.add_node(f"question:{task.task_id}")
        answer_node = self.atomspace.add_node(
            f"answer:{task.task_id}",
            metadata={"answer": answer, "question": question}
        )
        self.atomspace.add_link("answers", [answer_node, question_node])
        
        return answer


class CoordinatorAgent(Agent):
    """Agent that coordinates other agents for complex multi-step tasks."""
    
    async def process_task(self, task: AgentTask) -> Any:
        """Process a coordination task by delegating to sub-agents.
        
        Args:
            task: Task with 'subtasks' list
            
        Returns:
            Aggregated results from subtasks
        """
        subtasks = task.input_data.get("subtasks", [])
        
        if not subtasks:
            raise ValueError("Task must contain 'subtasks' in input_data")
        
        logger.info(f"Agent {self.name} coordinating {len(subtasks)} subtasks")
        
        # In a real implementation, this would delegate to other agents
        # For now, we'll just track the coordination
        results = []
        
        for i, subtask_data in enumerate(subtasks):
            # Create subtask node in atomspace
            subtask_node = self.atomspace.add_node(
                f"subtask:{task.task_id}:{i}",
                metadata=subtask_data
            )
            results.append(f"Coordinated subtask {i+1}/{len(subtasks)}")
        
        coordination_result = {
            "subtask_count": len(subtasks),
            "results": results,
        }
        
        return coordination_result


__all__ = [
    "LLMInferenceAgent",
    "SummarizationAgent", 
    "QuestionAnsweringAgent",
    "CoordinatorAgent",
]
