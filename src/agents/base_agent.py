from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from crewai import Agent, Task, Crew
from pydantic import BaseModel

class BaseAgentConfig(BaseModel):
    """Base configuration for agent initialization"""
    role: str
    goal: str
    backstory: str
    verbose: bool = True

class BaseNeurologyAgent(ABC):
    """
    Abstract base class for neurology agents.
    """
    def __init__(self, config: Dict[str, Any], llm: Any):
        """
        Initialize the base neurology agent.
        
        Args:
            config: Agent configuration dictionary
            llm: Language model to use for the agent
        """
        self.config = config
        self.llm = llm
        self.agent = self._create_agent()
        
    def _create_agent(self) -> Agent:
        """
        Create a CrewAI agent instance.
        
        Returns:
            CrewAI Agent instance
        """
        return Agent(
            role=self.config.get('role', ''),
            goal=self.config.get('goal', ''),
            backstory=self.config.get('backstory', ''),
            llm=self.llm,
            verbose=self.config.get('verbose', True),
            allow_delegation=self.config.get('allow_delegation', False),
            tools=self._get_tools()
        )
    
    @abstractmethod
    def _get_tools(self) -> List[Any]:
        """
        Get the list of tools used by this agent.
        
        Returns:
            List of tools for the agent
        """
        pass