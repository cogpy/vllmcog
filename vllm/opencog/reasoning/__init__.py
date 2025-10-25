"""Reasoning and planning capabilities for autonomous agents.

Provides goal-oriented planning, pattern matching, and learning
mechanisms for intelligent agent behavior.
"""

import logging
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field

from vllm.opencog.atomspace import AtomSpace, Atom, Node, Link

logger = logging.getLogger(__name__)


@dataclass
class Goal:
    """Represents a goal for autonomous planning."""
    
    goal_id: str
    description: str
    conditions: Dict[str, Any] = field(default_factory=dict)
    priority: float = 0.5
    completed: bool = False
    sub_goals: List['Goal'] = field(default_factory=list)


@dataclass
class Plan:
    """Represents a plan to achieve a goal."""
    
    plan_id: str
    goal: Goal
    steps: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    estimated_cost: float = 0.0


class GoalPlanner:
    """Goal-oriented planner for autonomous agents.
    
    Decomposes high-level goals into executable sub-tasks and
    creates plans using knowledge from the AtomSpace.
    """
    
    def __init__(self, atomspace: AtomSpace):
        """Initialize the planner.
        
        Args:
            atomspace: Knowledge space for planning
        """
        self.atomspace = atomspace
        self.goals: Dict[str, Goal] = {}
        self.plans: Dict[str, Plan] = {}
    
    def add_goal(self, goal: Goal) -> str:
        """Add a goal to the planner.
        
        Args:
            goal: Goal to add
            
        Returns:
            Goal ID
        """
        self.goals[goal.goal_id] = goal
        
        # Store goal in atomspace
        goal_node = self.atomspace.add_node(
            f"goal:{goal.goal_id}",
            metadata={
                "description": goal.description,
                "priority": goal.priority,
                "completed": goal.completed,
            }
        )
        
        logger.debug(f"Added goal: {goal.description}")
        return goal.goal_id
    
    def create_plan(self, goal_id: str) -> Optional[Plan]:
        """Create a plan to achieve a goal.
        
        Args:
            goal_id: ID of the goal to plan for
            
        Returns:
            Created plan or None if goal not found
        """
        goal = self.goals.get(goal_id)
        if not goal:
            logger.warning(f"Goal {goal_id} not found")
            return None
        
        # Simple planning: decompose goal into steps
        steps = self._decompose_goal(goal)
        
        plan = Plan(
            plan_id=f"plan_{goal_id}",
            goal=goal,
            steps=steps,
            estimated_cost=len(steps) * 1.0,
        )
        
        self.plans[plan.plan_id] = plan
        
        # Store plan in atomspace
        plan_node = self.atomspace.add_node(
            f"plan:{plan.plan_id}",
            metadata={
                "goal_id": goal_id,
                "steps": steps,
                "cost": plan.estimated_cost,
            }
        )
        
        # Link plan to goal
        from vllm.opencog.atomspace import AtomType
        goal_node = self.atomspace.get_atom(
            f"goal:{goal_id}",
            AtomType.NODE
        )
        if goal_node and plan_node:
            self.atomspace.add_link("achieves", [plan_node, goal_node])
        
        logger.info(f"Created plan with {len(steps)} steps for goal: {goal.description}")
        return plan
    
    def _decompose_goal(self, goal: Goal) -> List[str]:
        """Decompose a goal into executable steps.
        
        Args:
            goal: Goal to decompose
            
        Returns:
            List of step descriptions
        """
        # Simple heuristic decomposition
        # In a real system, this would use more sophisticated planning
        steps = []
        
        # If goal has sub-goals, plan for them
        if goal.sub_goals:
            for sub_goal in goal.sub_goals:
                steps.append(f"Achieve sub-goal: {sub_goal.description}")
        else:
            # Create basic steps based on goal description
            steps.append(f"Analyze: {goal.description}")
            steps.append(f"Execute: {goal.description}")
            steps.append(f"Verify: {goal.description}")
        
        return steps
    
    def get_next_action(self, plan_id: str) -> Optional[str]:
        """Get the next action from a plan.
        
        Args:
            plan_id: ID of the plan
            
        Returns:
            Next action or None if plan complete
        """
        plan = self.plans.get(plan_id)
        if not plan or not plan.steps:
            return None
        
        # Return first uncompleted step
        return plan.steps[0] if plan.steps else None
    
    def mark_step_completed(self, plan_id: str):
        """Mark the current step as completed.
        
        Args:
            plan_id: ID of the plan
        """
        plan = self.plans.get(plan_id)
        if plan and plan.steps:
            completed_step = plan.steps.pop(0)
            logger.debug(f"Completed step: {completed_step}")
            
            if not plan.steps:
                plan.goal.completed = True
                logger.info(f"Goal completed: {plan.goal.description}")


class PatternMatcher:
    """Pattern matching and recognition in the AtomSpace.
    
    Identifies patterns, similarities, and relationships to support
    learning and knowledge discovery.
    """
    
    def __init__(self, atomspace: AtomSpace):
        """Initialize the pattern matcher.
        
        Args:
            atomspace: Knowledge space to match patterns in
        """
        self.atomspace = atomspace
        self.learned_patterns: List[Dict[str, Any]] = []
    
    def find_similar_atoms(
        self,
        atom: Atom,
        threshold: float = 0.7
    ) -> List[Atom]:
        """Find atoms similar to the given atom.
        
        Args:
            atom: Reference atom
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of similar atoms
        """
        similar = []
        
        # Get all atoms of the same type
        if atom.atom_type.value == "node":
            candidates = self.atomspace.get_all_nodes()
        else:
            candidates = self.atomspace.get_all_links()
        
        for candidate in candidates:
            if candidate == atom:
                continue
            
            # Simple similarity: name overlap and truth value proximity
            similarity = self._compute_similarity(atom, candidate)
            
            if similarity >= threshold:
                similar.append(candidate)
        
        return similar
    
    def _compute_similarity(self, atom1: Atom, atom2: Atom) -> float:
        """Compute similarity between two atoms.
        
        Args:
            atom1: First atom
            atom2: Second atom
            
        Returns:
            Similarity score (0-1)
        """
        # Name similarity (simple word overlap)
        words1 = set(atom1.name.lower().split(':')[-1].split('_'))
        words2 = set(atom2.name.lower().split(':')[-1].split('_'))
        
        if not words1 or not words2:
            name_sim = 0.0
        else:
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            name_sim = intersection / union if union > 0 else 0.0
        
        # Truth value similarity
        truth_diff = abs(atom1.truth_value - atom2.truth_value)
        truth_sim = 1.0 - truth_diff
        
        # Combined similarity
        return (name_sim * 0.7 + truth_sim * 0.3)
    
    def learn_pattern(self, pattern_description: str, examples: List[Atom]):
        """Learn a pattern from examples.
        
        Args:
            pattern_description: Description of the pattern
            examples: Example atoms exhibiting the pattern
        """
        # Extract common features from examples
        common_features = self._extract_common_features(examples)
        
        pattern = {
            "description": pattern_description,
            "features": common_features,
            "example_count": len(examples),
        }
        
        self.learned_patterns.append(pattern)
        
        # Store pattern in atomspace
        pattern_node = self.atomspace.add_node(
            f"pattern:{pattern_description}",
            metadata=pattern
        )
        
        logger.info(f"Learned pattern: {pattern_description}")
    
    def _extract_common_features(self, atoms: List[Atom]) -> Dict[str, Any]:
        """Extract common features from a list of atoms.
        
        Args:
            atoms: List of atoms
            
        Returns:
            Dictionary of common features
        """
        if not atoms:
            return {}
        
        # Simple feature extraction
        features = {
            "count": len(atoms),
            "avg_truth_value": sum(a.truth_value for a in atoms) / len(atoms),
            "types": list(set(a.atom_type.value for a in atoms)),
        }
        
        return features
    
    def match_pattern(self, atom: Atom) -> List[Dict[str, Any]]:
        """Match an atom against learned patterns.
        
        Args:
            atom: Atom to match
            
        Returns:
            List of matching patterns with confidence scores
        """
        matches = []
        
        for pattern in self.learned_patterns:
            # Simple matching: check if atom's features align with pattern
            confidence = self._compute_pattern_match_confidence(
                atom,
                pattern
            )
            
            if confidence > 0.5:
                matches.append({
                    "pattern": pattern,
                    "confidence": confidence,
                })
        
        return sorted(matches, key=lambda x: x["confidence"], reverse=True)
    
    def _compute_pattern_match_confidence(
        self,
        atom: Atom,
        pattern: Dict[str, Any]
    ) -> float:
        """Compute confidence that an atom matches a pattern.
        
        Args:
            atom: Atom to check
            pattern: Pattern to match against
            
        Returns:
            Confidence score (0-1)
        """
        features = pattern.get("features", {})
        
        if not features:
            return 0.0
        
        # Check type match
        type_match = atom.atom_type.value in features.get("types", [])
        
        # Check truth value proximity
        avg_truth = features.get("avg_truth_value", 0.5)
        truth_proximity = 1.0 - abs(atom.truth_value - avg_truth)
        
        # Combined confidence
        confidence = (
            (1.0 if type_match else 0.0) * 0.6 +
            truth_proximity * 0.4
        )
        
        return confidence


__all__ = ["GoalPlanner", "PatternMatcher", "Goal", "Plan"]
