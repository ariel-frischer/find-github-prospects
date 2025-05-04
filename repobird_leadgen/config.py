import os
from typing import Final

from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN: Final[str | None] = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN missing – create a PAT and add to .env")

# Use string defaults for os.getenv, then convert type
CONCURRENCY: Final[int] = int(os.getenv("CONCURRENCY", "20"))
CACHE_DIR: Final[str] = os.getenv("CACHE_DIR", "cache")
OUTPUT_DIR: Final[str] = os.getenv("OUTPUT_DIR", "output")

# LLM model specifically for the main enrichment process
ENRICHER_LLM_MODEL: Final[str] = os.getenv(
    "ENRICHER_LLM_MODEL", "openrouter/google/gemini-2.5-pro-preview-03-25"
)

# LLM model specifically for URL summarization
SUMMARIZER_LLM_MODEL: Final[str] = os.getenv(
    "SUMMARIZER_LLM_MODEL", "openrouter/google/gemini-2.5-flash-preview"
)
