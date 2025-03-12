from typing import Dict, Any, List, Optional
from .base_model import BaseModel
from openai import AzureOpenAI

class AzureOpenAIModel(BaseModel):
    """
    Implementation of the BaseModel for Azure OpenAI models.
    """
    def __init__(self, 
                 endpoint: str, 
                 api_key: str, 
                 api_version: str,
                 deployment: str):
        """
        Initialize the Azure OpenAI model.
        
        Args:
            endpoint: Azure endpoint URL
            api_key: Azure OpenAI API key
            api_version: API version to use
            deployment: Model deployment name
        """
        self.endpoint = endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.deployment = deployment
        
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version
        )
    
    def query(self, prompt: str, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Send a single prompt to the model.
        """
        response = self.client.chat.completions.create(
            model=self.deployment,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        return {
            "text": response.choices[0].message.content,
            "model": response.model,
            "usage": response.usage,
            "raw_response": response
        }
    
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> Dict[str, Any]:
        """
        Send a chat-style query with multiple messages.
        """
        response = self.client.chat.completions.create(
            model=self.deployment,
            temperature=temperature,
            messages=messages
        )
        
        return {
            "text": response.choices[0].message.content,
            "model": response.model,
            "usage": response.usage,
            "raw_response": response
        }