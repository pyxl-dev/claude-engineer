"""
OpenRouter client implementation for OpenAI API compatibility.
This module provides a configured OpenAI client that uses OpenRouter as the backend.
"""
import os
from typing import Optional
from openai import OpenAI
from dataclasses import dataclass

@dataclass
class OpenRouterConfig:
    """Configuration for OpenRouter client"""
    api_key: str
    site_url: Optional[str] = None
    app_name: Optional[str] = None

class OpenRouterClient:
    """Client for interacting with OpenRouter API"""
    
    def __init__(self, config: OpenRouterConfig):
        """
        Initialize OpenRouter client with configuration
        
        Args:
            config: OpenRouterConfig instance with API key and optional metadata
        """
        self.config = config
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.api_key
        )
        
    def get_completion(self, prompt: str, model: str = "openai/gpt-3.5-turbo") -> str:
        """
        Get completion from OpenRouter
        
        Args:
            prompt: The prompt to send
            model: Model identifier to use
            
        Returns:
            The completion text
        """
        headers = {}
        if self.config.site_url:
            headers["HTTP-Referer"] = self.config.site_url
        if self.config.app_name:
            headers["X-Title"] = self.config.app_name
            
        completion = self.client.chat.completions.create(
            extra_headers=headers,
            model=model,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        return completion.choices[0].message.content

def create_client(
    api_key: Optional[str] = None,
    site_url: Optional[str] = None,
    app_name: Optional[str] = None
) -> OpenRouterClient:
    """
    Create an OpenRouter client with the given configuration
    
    Args:
        api_key: OpenRouter API key, defaults to OPENROUTER_API_KEY env var
        site_url: Optional site URL for rankings
        app_name: Optional app name for rankings
        
    Returns:
        Configured OpenRouterClient instance
    """
    config = OpenRouterConfig(
        api_key=api_key or os.getenv("OPENROUTER_API_KEY", ""),
        site_url=site_url,
        app_name=app_name
    )
    
    if not config.api_key:
        raise ValueError("OpenRouter API key must be provided or set in OPENROUTER_API_KEY environment variable")
        
    return OpenRouterClient(config)
