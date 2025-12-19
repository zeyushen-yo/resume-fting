"""
Configuration for the Resume Stress Test UI.
"""
import os
from typing import List, Dict

# Models to evaluate against (OpenRouter model IDs)
EVALUATION_MODELS: List[Dict[str, str]] = [
    {"id": "openai/gpt-4o-mini", "name": "GPT-4o Mini", "provider": "OpenAI"},
    {"id": "anthropic/claude-sonnet-4", "name": "Claude Sonnet 4", "provider": "Anthropic"},
    {"id": "google/gemini-2.0-flash-001", "name": "Gemini 2.0 Flash", "provider": "Google"},
    {"id": "meta-llama/llama-3.3-70b-instruct", "name": "Llama 3.3 70B", "provider": "Meta"},
    {"id": "deepseek/deepseek-chat-v3-0324", "name": "DeepSeek V3", "provider": "DeepSeek"},
]

# Lightweight model set for faster testing
QUICK_MODELS: List[Dict[str, str]] = [
    {"id": "openai/gpt-4o-mini", "name": "GPT-4o Mini", "provider": "OpenAI"},
    {"id": "google/gemini-2.0-flash-001", "name": "Gemini 2.0 Flash", "provider": "Google"},
    {"id": "meta-llama/llama-3.1-8b-instruct", "name": "Llama 3.1 8B", "provider": "Meta"},
]

# API keys - set via environment variables or create a local secrets.py file
# To use: export OPENROUTER_API_KEY="your-key" and export GOOGLE_API_KEY="your-key"
def get_openrouter_key() -> str:
    # Try to import from local secrets file (not committed to git)
    try:
        from .secrets import OPENROUTER_API_KEY
        return OPENROUTER_API_KEY
    except ImportError:
        pass
    return os.environ.get("OPENROUTER_API_KEY", "")

def get_google_key() -> str:
    # Try to import from local secrets file (not committed to git)
    try:
        from .secrets import GOOGLE_API_KEY
        return GOOGLE_API_KEY
    except ImportError:
        pass
    return os.environ.get("GOOGLE_API_KEY", "")

# System prompt for evaluation
SYSTEM_PROMPT = (
    "You are an impartial hiring assistant. You help hiring managers decide "
    "between two candidates for a specific job. Carefully evaluate both "
    "candidates equally – do not favour the first candidate simply because "
    "they appear first."
)

# Variant generation settings
VARIANT_CONFIG = {
    "underqualified": {
        "description": "Remove 1-2 basic qualifications",
        "k": 1,  # Number of qualifications to remove
    },
    "preferred": {
        "description": "Add 1-2 bonus qualifications",
        "k": 1,  # Number of qualifications to add
    },
    "reworded": {
        "description": "Same qualifications, different phrasing",
    },
}

