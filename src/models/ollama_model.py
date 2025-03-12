from typing import Dict, Any, List, Optional
import ollama
from .base_model import BaseModel

class OllamaModel(BaseModel):
    """
    Implementation of the BaseModel for Ollama models.
    """
    def __init__(self, model_name: str, host: str = "http://localhost:11434"):
        """
        Initialize the Ollama model.
        
        Args:
            model_name: Name of the Ollama model to use
            host: Host URL for the Ollama API
        """
        self.model_name = model_name
        self.host = host
        self.client = ollama.Client(host=host)
    
    def query(self, prompt: str, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Send a single prompt to the model.
        """
        response = self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature}
        )
        
        return {
            "text": response.message.content,
            "model": response.model,
            "usage": {
                "prompt_eval_count": response.prompt_eval_count,
                "prompt_eval_duration": response.prompt_eval_duration,
                "eval_count": response.eval_count,
                "eval_duration": response.eval_duration
            },
            "raw_response": response
        }
    
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> Dict[str, Any]:
        """
        Send a chat-style query with multiple messages.
        """
        response = self.client.chat(
            model=self.model_name,
            messages=messages,
            options={"temperature": temperature}
        )
        
        return {
            "text": response.message.content,
            "model": response.model,
            "usage": {
                "prompt_eval_count": response.prompt_eval_count,
                "prompt_eval_duration": response.prompt_eval_duration,
                "eval_count": response.eval_count,
                "eval_duration": response.eval_duration
            },
            "raw_response": response
        }