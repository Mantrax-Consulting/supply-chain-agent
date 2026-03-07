"""
config.py — Central Configuration for the Supply Chain Agent
=============================================================
This file is your single control panel for:
  1. Which LLM provider to use
  2. API keys and credentials
  3. Agent behaviour settings

Workshop Note:
  Changing LLM_PROVIDER is all you need to switch between models.
  The agent itself never needs to change — that's the point.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # Reads from your .env file

# ─── LLM Configuration ────────────────────────────────────────────────────────
# Change LLM_PROVIDER to switch between models.
# "openai"    → Uses GPT models
# "anthropic" → Uses Claude models  
# "google"    → Uses Gemini models

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

MODEL_NAMES = {
    "openai":    os.getenv("OPENAI_MODEL",    "gpt-4o-mini"),
    "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
    "google":    os.getenv("GOOGLE_MODEL",    "gemini-1.5-flash"),
}

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# ─── API Keys ─────────────────────────────────────────────────────────────────
# These are loaded from your .env file — NEVER hardcode keys in source code.
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY    = os.getenv("GOOGLE_API_KEY")

# Google Search (required for all providers)
GOOGLE_CSE_ID     = os.getenv("GOOGLE_CSE_ID")
GOOGLE_CSE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY")

# ─── Observability (LangSmith) ────────────────────────────────────────────────
# LangSmith traces every LLM call, tool use, and token count.
# Sign up free at: https://smith.langchain.com
LANGCHAIN_TRACING_V2   = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_API_KEY      = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT      = os.getenv("LANGCHAIN_PROJECT", "supply-chain-monitor")
LANGCHAIN_ENDPOINT     = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# ─── Agent Behaviour ─────────────────────────────────────────────────────────
MAX_AGENT_ITERATIONS  = int(os.getenv("MAX_ITERATIONS", "6"))
REQUIRE_HUMAN_APPROVAL = os.getenv("REQUIRE_APPROVAL", "true").lower() == "true"

# ─── Alert Settings ───────────────────────────────────────────────────────────
ALERT_EMAIL       = os.getenv("ALERT_EMAIL", "operations@yourcompany.com")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")

# ─── Validation ───────────────────────────────────────────────────────────────

def validate_config():
    """Call this at startup to catch missing environment variables early."""
    errors = []

    if not GOOGLE_CSE_ID:
        errors.append("GOOGLE_CSE_ID is missing — required for web search")
    if not GOOGLE_CSE_API_KEY:
        errors.append("GOOGLE_CSE_API_KEY is missing — required for web search")

    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is missing — required for OpenAI provider")
    if LLM_PROVIDER == "anthropic" and not ANTHROPIC_API_KEY:
        errors.append("ANTHROPIC_API_KEY is missing — required for Anthropic provider")
    if LLM_PROVIDER == "google" and not GOOGLE_API_KEY:
        errors.append("GOOGLE_API_KEY is missing — required for Google provider")

    if errors:
        print("\n⚠️  CONFIGURATION ERRORS:")
        for e in errors:
            print(f"   ✗ {e}")
        print("\n   See .env.example for the required variables.\n")
        return False

    print(f"✅  Config OK — Provider: {LLM_PROVIDER}, Model: {MODEL_NAMES[LLM_PROVIDER]}")
    return True


if __name__ == "__main__":
    validate_config()
