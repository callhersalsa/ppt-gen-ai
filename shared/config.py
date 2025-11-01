"""
config.py

This module defines configuration settings for the application.

Purpose:
- Centralizes all environment-specific or application-wide settings
- Helps manage constants, API keys, database URIs, debug flags, etc.
- Makes it easy to switch between development, testing, and production environments
"""

import os
import logging

from pathlib import Path
from dotenv import load_dotenv

# Configuring Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
# Point to the .env file in the parent directory (one level up from app/)
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
logger.info("Load .env variable config")

class Config:
    # SearXNG Configuration
    SEARXNG_HOST = os.getenv("SEARXNG_HOST", "searxng")
    SEARXNG_PORT = int(os.getenv("SEARXNG_PORT", "8080"))
    SEARXNG_BASE_PATH = os.getenv("SEARXNG_BASE_PATH", "/search")
    SEARXNG_API_BASE = f"http://{SEARXNG_HOST}:{SEARXNG_PORT}{SEARXNG_BASE_PATH}"

    # API Service Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "6789"))

    # Crawler Configuration
    DEFAULT_SEARCH_LIMIT = int(os.getenv("DEFAULT_SEARCH_LIMIT", "10"))
    CONTENT_FILTER_THRESHOLD = float(os.getenv("CONTENT_FILTER_THRESHOLD", "0.6"))
    WORD_COUNT_THRESHOLD = int(os.getenv("WORD_COUNT_THRESHOLD", "10"))

    # Search engine configuration
    DISABLED_ENGINES = os.getenv(
        "DISABLED_ENGINES",
        "wikipedia__general,wikipedia,wikidata,currency__general,wikidata__general,baidu__general,lingva__general,qwant__general,startpage__general,dictzone__general,mymemory translated__general"
    )
    ENABLED_ENGINES = os.getenv("ENABLED_ENGINES", "google__general,bing__general,yahoo__general,yandex__general,duckduckgo__general,brave__general")
    
    # Social media domains to exclude during crawling
    SOCIAL_MEDIA_DOMAINS = os.getenv("SOCIAL_MEDIA_DOMAINS", 
        "facebook.com,x.com,instagram.com,linkedin.com,youtube.com,pinterest.com,tiktok.com,snapchat.com,reddit.com,whatsapp.com,wechat.com,qq.com,telegram.org,vk.com,line.me"
    ).split(',')

    # API key security
    ACCESS_KEY = os.getenv('X_API_KEY')

    # LLM Token
    # CUSTOM_LLM_URL = os.getenv("CUSTOM_LLM_URL")
    # CUSTOM_LLM_CHAT_ENDPOINT = os.getenv("CUSTOM_LLM_CHAT_ENDPOINT")
    # CUSTOM_LLM_TOKEN = os.getenv("CUSTOM_LLM_TOKEN")
    # CUSTOM_LLM_CHAT_URL = f"{CUSTOM_LLM_URL}{CUSTOM_LLM_CHAT_ENDPOINT}{CUSTOM_LLM_TOKEN}"

    # LLM (Gemma via Ollama)
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama3-70b-8192")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Embedding model
    EMBEDDING_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
    # GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Export configuration information function
def get_config_info():
    """Returns a dictionary of current configuration information

    Returns:
        dict: A dictionary containing all configuration parameters
    """
    return {
        "searxng": {
            "host": Config.SEARXNG_HOST,
            "port": Config.SEARXNG_PORT,
            "base_path": Config.SEARXNG_BASE_PATH,
            "api_base": Config.SEARXNG_API_BASE
        },
        "api": {
            "host": Config.API_HOST,
            "port": Config.API_PORT
        },
        "crawler": {
            "default_search_limit": Config.DEFAULT_SEARCH_LIMIT,
            "content_filter_threshold": Config.CONTENT_FILTER_THRESHOLD,
            "word_count_threshold": Config.WORD_COUNT_THRESHOLD,
            "social_media_domains": Config.SOCIAL_MEDIA_DOMAINS
        },
        "search_engines": {
            "disabled": Config.DISABLED_ENGINES,
            "enabled": Config.ENABLED_ENGINES
        },
        "llm": {
        #    "url": "***REDACTED***",
        #    "chat_endpoint": "***REDACTED***",
        #    "token": "***REDACTED***",
            "llm_provider": Config.LLM_PROVIDER,
            "llm": Config.LLM_MODEL_NAME,
            "api_key": "***REDACTED***",
        },
        "embedding": {
            "model": Config.EMBEDDING_MODEL_NAME,
            "dimension": Config.EMBEDDING_DIM
        },
        "X_API": {
            "x_api_key": "***REDACTED***"
        }
    }
