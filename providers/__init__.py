"""
Providers package for external service integrations
"""

from .openrouter_client import create_client, OpenRouterClient, OpenRouterConfig

__all__ = ['create_client', 'OpenRouterClient', 'OpenRouterConfig']
