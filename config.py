from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
    
    # Provider Configuration
    PROVIDER = os.getenv('PROVIDER', 'anthropic').lower()  # 'anthropic' or 'openrouter'
    
    # Model Configuration
    ANTHROPIC_MODEL = os.getenv('ANTHROPIC_MODEL', 'claude-3-opus-20240229')
    OPENROUTER_MODEL = os.getenv('OPENROUTER_MODEL', 'openai/gpt-4-turbo-preview')
    
    # OpenRouter Fallback Models (in order of preference)
    OPENROUTER_FALLBACK_MODELS = [
        "openai/gpt-4-turbo-preview",
        "anthropic/claude-3-opus",
        "openai/gpt-3.5-turbo",
        "google/gemini-pro",
        "meta-llama/llama-2-70b-chat",
    ]
    
    # OpenRouter specific settings
    OPENROUTER_SITE_URL = os.getenv('OPENROUTER_SITE_URL', '')
    OPENROUTER_APP_NAME = os.getenv('OPENROUTER_APP_NAME', 'CE3')
    
    # Common Configuration
    MAX_TOKENS = 8000
    MAX_CONVERSATION_TOKENS = 200000  # Maximum tokens per conversation

    # Paths
    BASE_DIR = Path(__file__).parent
    TOOLS_DIR = BASE_DIR / "tools"
    PROMPTS_DIR = BASE_DIR / "prompts"

    # Assistant Configuration
    ENABLE_THINKING = True
    SHOW_TOOL_USAGE = True
    DEFAULT_TEMPERATURE = 0.7
