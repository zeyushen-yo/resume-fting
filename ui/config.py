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

# Bypass mode password - allows users to use our API key without providing their own
BYPASS_PASSWORD = "123456"

def get_openrouter_key() -> str:
    """Get OpenRouter API key from environment or fallback."""
    return os.environ.get("OPENROUTER_API_KEY", _FALLBACK_OPENROUTER_KEY)

def get_google_key() -> str:
    """Get Google API key from environment or fallback."""
    return os.environ.get("GOOGLE_API_KEY", _FALLBACK_GOOGLE_KEY)

def check_bypass_password(password: str) -> bool:
    """Check if the provided password matches the bypass password."""
    return password == BYPASS_PASSWORD

def get_bypass_api_config():
    """Get the API configuration for bypass mode (using our OpenRouter key)."""
    return {
        "api_base": "https://openrouter.ai/api",
        "api_key": get_openrouter_key(),
    }

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

# Embedding model for skill overlap checking (via OpenRouter)
EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
SIMILARITY_THRESHOLD = 0.8  # Cosine similarity threshold for considering skills as overlapping