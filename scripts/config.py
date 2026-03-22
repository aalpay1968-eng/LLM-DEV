# scripts/config.py — Proje sabitleri ve API havuzu
import os

GITHUB_USER = "aalpay1968-eng"
GITHUB_REPO = "LLM-DEV"
HF_USER = "aalpay1968"
HF_REPO_CHAT = f"{HF_USER}/target-llm-chat"
HF_REPO_PHASE1 = f"{HF_USER}/target-llm-phase1"
HF_REPO_PHASE2 = f"{HF_USER}/target-llm-phase2"
HF_REPO_DEMO = f"{HF_USER}/target-llm-demo"
GIT_EMAIL = "aalpay1968@gmail.com"
GIT_NAME = "Antigravity Bot"

DEVELOPER_APIS = [
    {
        "name": "xai",
        "base_url": "https://api.x.ai/v1",
        "model": "grok-3-mini-fast",
        "key_secret": "XAI_API_KEY",
        "extra_params": {},
    },
    {
        "name": "cerebras",
        "base_url": "https://api.cerebras.ai/v1",
        "model": "qwen-3-32b",
        "key_secret": "CEREBRAS_API_KEY",
        "extra_params": {},
    },
    {
        "name": "groq",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "deepseek-r1-distill-llama-70b",
        "key_secret": "GROQ_API_KEY",
        "extra_params": {},
    },
    {
        "name": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "deepseek/deepseek-r1:free",
        "key_secret": "OPENROUTER_KEY",
        "extra_params": {},
    },
    {
        "name": "google_ai_studio",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.0-flash",
        "key_secret": "GOOGLE_AI_KEY",
        "extra_params": {},
    },
]
