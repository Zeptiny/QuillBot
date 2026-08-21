"""Centralized configuration — single source of truth for all env vars."""

import os
import sys
from typing import Final

from dotenv import load_dotenv

load_dotenv()

# --- Required ---
BOT_TOKEN: Final[str | None] = os.getenv('BOT_TOKEN')
OPENROUTER_API_KEY: Final[str | None] = os.getenv('OPENROUTER_API_KEY')
TAVILY_API_KEY: Final[str | None] = os.getenv('TAVILY_API_KEY')

# --- LLM (OpenAI-compatible) ---
# OPENAI_* is the canonical name for any OpenAI-compatible endpoint.
# OPENROUTER_* and LLM_* are kept as aliases for backward compatibility.
OPENAI_API_KEY: Final[str | None] = (
    os.getenv('OPENAI_API_KEY') or os.getenv('LLM_API_KEY') or OPENROUTER_API_KEY
)
OPENAI_BASE_URL: Final[str] = (
    os.getenv('OPENAI_BASE_URL')
    or os.getenv('LLM_BASE_URL')
    or os.getenv('OPENROUTER_BASE_URL')
    or 'https://openrouter.ai/api/v1'
).rstrip('/')
# Aliases for code that still imports the old names
LLM_API_KEY: Final[str | None] = OPENAI_API_KEY
LLM_BASE_URL: Final[str] = OPENAI_BASE_URL

# --- AI Models ---
CHAT_MODEL: Final[str] = os.getenv('CHAT_MODEL', 'qwen/qwen3.6-plus')
SPARK_MODEL: Final[str] = os.getenv('SPARK_MODEL', 'google/gemini-2.5-pro')
EMBEDDING_MODEL: Final[str] = os.getenv('EMBEDDING_MODEL', 'qwen/qwen3-embedding-8b')
RERANK_MODEL: Final[str] = os.getenv('RERANK_MODEL', 'cohere/rerank-4-fast')

# --- Embeddings (local vs remote) ---
EMBEDDING_PROVIDER: Final[str] = os.getenv('EMBEDDING_PROVIDER', 'openai').strip().lower()
LOCAL_EMBEDDING_MODEL: Final[str] = os.getenv(
    'LOCAL_EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'
)
LOCAL_EMBEDDING_DEVICE: Final[str] = os.getenv('LOCAL_EMBEDDING_DEVICE', 'cpu')
RERANK_ENABLED: Final[bool] = os.getenv('RERANK_ENABLED', 'true').strip().lower() in ('1', 'true', 'yes')
def _is_openrouter_url(url: str) -> bool:
    return 'openrouter.ai' in url
RERANK_AVAILABLE: Final[bool] = RERANK_ENABLED and _is_openrouter_url(OPENAI_BASE_URL)
RERANK_PROVIDER: Final[str] = os.getenv('RERANK_PROVIDER', 'auto').strip().lower()
LOCAL_RERANK_MODEL: Final[str] = os.getenv(
    'LOCAL_RERANK_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2'
)
LOCAL_RERANK_DEVICE: Final[str] = os.getenv('LOCAL_RERANK_DEVICE', LOCAL_EMBEDDING_DEVICE)
HISTORY_RERANK_ENABLED: Final[bool] = os.getenv('HISTORY_RERANK_ENABLED', 'true').strip().lower() in ('1', 'true', 'yes')
HISTORY_RERANK_PROVIDER: Final[str] = os.getenv('HISTORY_RERANK_PROVIDER', RERANK_PROVIDER).strip().lower()
HISTORY_RERANK_MODEL: Final[str] = os.getenv('HISTORY_RERANK_MODEL', LOCAL_RERANK_MODEL)
HISTORY_TIME_DECAY_LAMBDA: Final[float] = float(os.getenv('HISTORY_TIME_DECAY_LAMBDA', '0.0'))
HISTORY_HYBRID_WEIGHT_SEMANTIC: Final[float] = float(os.getenv('HISTORY_HYBRID_WEIGHT_SEMANTIC', '0.65'))
HISTORY_HYBRID_WEIGHT_KEYWORD: Final[float] = float(os.getenv('HISTORY_HYBRID_WEIGHT_KEYWORD', '0.35'))
HISTORY_RRF_K: Final[int] = int(os.getenv('HISTORY_RRF_K', '60'))

# --- Chat Mention ---
CHAT_MENTION_ENABLED: Final[bool] = os.getenv('CHAT_MENTION_ENABLED', 'true').strip().lower() in ('1', 'true', 'yes')

# --- Conversations (multi-turn chat memory) ---
CONVERSATIONS_DB_PATH: Final[str] = os.getenv('CONVERSATIONS_DB_PATH', 'data/conversations.db')
CONVERSATIONS_TTL_SECONDS: Final[float] = float(os.getenv('CONVERSATIONS_TTL_SECONDS', '1800'))
CONVERSATIONS_MAX_STORED: Final[int] = int(os.getenv('CONVERSATIONS_MAX_STORED', '200'))
CONVERSATIONS_MAX_TURNS: Final[int] = int(os.getenv('CONVERSATIONS_MAX_TURNS', '24'))
CONVERSATIONS_HISTORY_TURNS: Final[int] = int(os.getenv('CONVERSATIONS_HISTORY_TURNS', '16'))

# --- History RAG (entire server) ---
HISTORY_ENABLED: Final[bool] = os.getenv('HISTORY_ENABLED', 'true').strip().lower() in ('1', 'true', 'yes')
HISTORY_VECTOR_STORE_DIR: Final[str] = os.getenv('HISTORY_VECTOR_STORE_DIR', 'data/history')
HISTORY_DB_PATH: Final[str] = os.getenv('HISTORY_DB_PATH', os.path.join(HISTORY_VECTOR_STORE_DIR, 'history.db'))
HISTORY_WINDOW_SIZE: Final[int] = int(os.getenv('HISTORY_WINDOW_SIZE', '5'))
HISTORY_WINDOW_OVERLAP: Final[int] = int(os.getenv('HISTORY_WINDOW_OVERLAP', '1'))
HISTORY_BACKFILL_LIMIT: Final[int | None] = int(v) if (v := os.getenv('HISTORY_BACKFILL_LIMIT', '').strip()) else None
HISTORY_MAX_MSG_LENGTH: Final[int] = int(os.getenv('HISTORY_MAX_MSG_LENGTH', '800'))
HISTORY_EXCLUDE_BOTS: Final[bool] = os.getenv('HISTORY_EXCLUDE_BOTS', 'true').strip().lower() in ('1', 'true', 'yes')
HISTORY_INGEST_BATCH_SIZE: Final[int] = int(os.getenv('HISTORY_INGEST_BATCH_SIZE', '10'))
HISTORY_INGEST_FLUSH_SECONDS: Final[float] = float(os.getenv('HISTORY_INGEST_FLUSH_SECONDS', '2.0'))
HISTORY_SNAPSHOT_INTERVAL: Final[float] = float(os.getenv('HISTORY_SNAPSHOT_INTERVAL', '300'))
HISTORY_QUERY_CACHE_SIZE: Final[int] = int(os.getenv('HISTORY_QUERY_CACHE_SIZE', '200'))

# --- Lore Encyclopedia (server history, inside jokes, glossary) ---
LORE_ENABLED: Final[bool] = os.getenv('LORE_ENABLED', 'true').strip().lower() in ('1', 'true', 'yes')
LORE_DB_PATH: Final[str] = os.getenv('LORE_DB_PATH', 'data/lore.db')
LORE_LOG_CHANNEL_ID: Final[int | None] = int(v) if (v := os.getenv('LORE_LOG_CHANNEL_ID', '').strip()) else None
LORE_BOT_WRITE_LIMIT: Final[int] = int(os.getenv('LORE_BOT_WRITE_LIMIT', '10'))

# --- Rate Limiting (per-user) ---
COOLDOWN_RATE: Final[int] = int(os.getenv('COOLDOWN_RATE', '1'))
COOLDOWN_PER: Final[float] = float(os.getenv('COOLDOWN_PER', '30'))

# --- Feature Flags ---
WEB_SEARCH_ENABLED: Final[bool] = os.getenv('WEB_SEARCH_ENABLED', 'true').strip().lower() in ('1', 'true', 'yes')
TAVILY_AVAILABLE: Final[bool] = bool(TAVILY_API_KEY) and WEB_SEARCH_ENABLED

# --- Logging ---
LOG_LEVEL: Final[str] = os.getenv('LOG_LEVEL', 'INFO').upper()

# --- Docs / RAG ---
DOCS_REPO: Final[str] = 'MinersRefuge/docs'
DOCS_BRANCH: Final[str] = 'main'
DOCS_BASE_URL: Final[str] = 'https://docs.minersrefuge.com.br'
GITHUB_RAW: Final[str] = f'https://raw.githubusercontent.com/{DOCS_REPO}/{DOCS_BRANCH}'
GITHUB_API: Final[str] = f'https://api.github.com/repos/{DOCS_REPO}'
VECTOR_STORE_PATH: Final[str] = os.getenv('VECTOR_STORE_PATH', 'data/vectors.json')
REINDEX_INTERVAL_HOURS: Final[int] = int(os.getenv('REINDEX_INTERVAL_HOURS', '6'))

# --- Content Limits ---
MAX_CONTENT_SIZE: Final[int] = 5 * 1024 * 1024  # 5 MB
MAX_LOG_CONTEXT: Final[int] = 12000  # Max characters sent to the LLM

# --- Documentation Sources ---
# Each entry must have: repo, branch, base_url, label
# Optional:
#   summary         — path to SUMMARY.md (mdBook); absent means GitHub tree API is used
#   path_prefix     — only index files whose path starts with this prefix (tree discovery)
#   url_strip_prefix— strip this from the repo path before building the docs website URL
#   max_files       — maximum number of files to index per source (default 200)
DOC_SOURCES: Final[list[dict]] = [
    {
        'repo': 'MinersRefuge/docs',
        'branch': 'main',
        'base_url': 'https://docs.minersrefuge.com.br',
        'label': "Miners' Refuge",
        'summary': 'SUMMARY.md',
    },
    {
        'repo': 'PaperMC/docs',
        'branch': 'main',
        'base_url': 'https://docs.papermc.io',
        'label': 'PaperMC',
        'path_prefix': 'src/content/docs/paper/admin/',
        'url_strip_prefix': 'src/content/docs/',
        'max_files': 400,
    },
    {
        'repo': 'PurpurMC/PurpurDocs',
        'branch': 'main',
        'base_url': 'https://purpurmc.org/docs/purpur/',
        'label': 'PurpurMC',
        'path_prefix': 'mkdocs/purpur/',
        'url_strip_prefix': 'mkdocs/purpur/',
        'max_files': 400,
    },
    {
        'repo': 'lucko/spark-docs',
        'branch': 'master',
        'base_url': 'https://spark.lucko.me/docs/',
        'label': 'Spark',
        'path_prefix': 'docs/',
        'url_strip_prefix': 'docs/',
        'max_files': 400,
    },
]


def validate_config() -> None:
    """Validate required configuration at startup. Exits on failure."""
    if not BOT_TOKEN:
        print('FATAL: BOT_TOKEN environment variable is not set. Cannot start.', file=sys.stderr)
        sys.exit(1)
    if WEB_SEARCH_ENABLED and not TAVILY_API_KEY:
        print(
            'WARNING: WEB_SEARCH_ENABLED is true but TAVILY_API_KEY is not set. '
            'Web search will be disabled.',
            file=sys.stderr,
        )
