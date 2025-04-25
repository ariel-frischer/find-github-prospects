import os
from dotenv import load_dotenv
from typing import Final

load_dotenv()

GITHUB_TOKEN: Final[str | None] = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN missing – create a PAT and add to .env")

CONCURRENCY: Final[int] = int(os.getenv("CONCURRENCY", 20))
CACHE_DIR: Final[str] = os.getenv("CACHE_DIR", "cache")
OUTPUT_DIR: Final[str] = os.getenv("OUTPUT_DIR", "output")
