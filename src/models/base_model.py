from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class BaseModel(ABC):
    """
    Abstract base class for language models.
    """
    @abstractmethod
    def query(self, prompt: str, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Query the model with a prompt.
        
        Args:
            prompt: The text prompt to send to the model
            temperature: Sampling temperature (0.0 to 1.0)
            
        Returns:
            Dictionary containing the model's response and metadata
        """
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> Dict[str, Any]:
        """
        Send a chat-style query to the model.
        
        Args:
            messages: List of message dictionaries (role and content)
            temperature: Sampling temperature (0.0 to 1.0)
            
        Returns:
            Dictionary containing the model's response and metadata
        """
        pass