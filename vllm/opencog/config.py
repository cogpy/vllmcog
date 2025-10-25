"""Configuration management for OpenCog orchestration."""

import yaml
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    
    agent_type: str
    name: Optional[str] = None
    model: Optional[str] = None
    sampling_params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PoolConfig:
    """Configuration for an agent pool."""
    
    pool_id: str
    agents: List[AgentConfig] = field(default_factory=list)
    load_balancing: str = "round_robin"
    max_concurrent_tasks: int = 100


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    
    pools: List[PoolConfig] = field(default_factory=list)
    enable_collaboration: bool = True
    enable_learning: bool = True
    task_timeout: float = 300.0
    metrics_enabled: bool = True


class ConfigLoader:
    """Loader for YAML configuration files."""
    
    @staticmethod
    def load_from_file(filepath: str) -> OrchestratorConfig:
        """Load configuration from a YAML file.
        
        Args:
            filepath: Path to YAML config file
            
        Returns:
            Orchestrator configuration
        """
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return ConfigLoader.from_dict(config_dict)
    
    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> OrchestratorConfig:
        """Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Orchestrator configuration
        """
        pools = []
        
        for pool_data in config_dict.get("pools", []):
            agents = []
            for agent_data in pool_data.get("agents", []):
                agents.append(AgentConfig(**agent_data))
            
            pool_config = PoolConfig(
                pool_id=pool_data["pool_id"],
                agents=agents,
                load_balancing=pool_data.get("load_balancing", "round_robin"),
                max_concurrent_tasks=pool_data.get("max_concurrent_tasks", 100),
            )
            pools.append(pool_config)
        
        return OrchestratorConfig(
            pools=pools,
            enable_collaboration=config_dict.get("enable_collaboration", True),
            enable_learning=config_dict.get("enable_learning", True),
            task_timeout=config_dict.get("task_timeout", 300.0),
            metrics_enabled=config_dict.get("metrics_enabled", True),
        )
    
    @staticmethod
    def save_to_file(config: OrchestratorConfig, filepath: str):
        """Save configuration to a YAML file.
        
        Args:
            config: Configuration to save
            filepath: Output file path
        """
        config_dict = asdict(config)
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)


__all__ = [
    "AgentConfig",
    "PoolConfig",
    "OrchestratorConfig",
    "ConfigLoader",
]
