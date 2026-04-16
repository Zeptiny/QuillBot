"""Centralized configuration — single source of truth for all env vars."""

import os
import sys
from typing import Final

from dotenv import load_dotenv

load_dotenv()

# --- Required ---
BOT_TOKEN: Final[str | None] = os.getenv('BOT_TOKEN')
OPENROUTER_API_KEY: Final[str | None] = os.getenv('OPENROUTER_API_KEY')

# --- AI Models ---
CHAT_MODEL: Final[str] = os.getenv('CHAT_MODEL', 'google/gemini-2.5-flash-lite')
EMBEDDING_MODEL: Final[str] = os.getenv('EMBEDDING_MODEL', 'qwen/qwen3-embedding-8b')
RERANK_MODEL: Final[str] = os.getenv('RERANK_MODEL', 'cohere/rerank-4-fast')

# --- Rate Limiting (per-user) ---
COOLDOWN_RATE: Final[int] = int(os.getenv('COOLDOWN_RATE', '1'))
COOLDOWN_PER: Final[float] = float(os.getenv('COOLDOWN_PER', '30'))

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
        'path_prefix': 'docs/paper/admin',
        'url_strip_prefix': 'docs/',
        'max_files': 30,
    },
]


def validate_config() -> None:
    """Validate required configuration at startup. Exits on failure."""
    if not BOT_TOKEN:
        print('FATAL: BOT_TOKEN environment variable is not set. Cannot start.', file=sys.stderr)
        sys.exit(1)
