# RepoBird LeadGen

**Automated discovery of open‑source repositories with “good‑first‑issue” or
“help‑wanted” labels, plus maintainer contact extraction and prospect
summaries.**

## Features
1. GitHub code‑search with rich filters (stars, language, freshness).
2. Parallel metadata fetch (AIO/threads) for up to 1000 repos per run.
3. Heuristic contact scraping from:
   * Owner profile → public email / blog URL / Twitter.
   * Repository `README`, `package.json`, `AUTHORS`, commit history.
4. Generates **CSV / JSONL / Markdown** summary files ready for outreach.
5. Extensible agent hooks (e.g., OpenAI for quality ranking, email drafting).

## Quick Start
```bash
# 1 – clone & install
pip install -r requirements.txt
cp .env.example .env  # add GITHUB_TOKEN

# 2 – run a full pipeline (50 targets)
python -m repobird_leadgen.cli full --max-results 50 --language python

# 3 – inspect output
cat output/prospects_2025‑04‑24.md
```

## CLI Overview
| Command                   | Purpose                                   |
|---------------------------|-------------------------------------------|
| `search`                  | Fetch raw repo list and save to cache.    |
| `enrich`                  | Add contacts + metrics to cached repos.   |
| `full`                    | Convenience: `search` → `enrich`.         |

## Configuration (.env)
| Variable        | Description                     |
|-----------------|---------------------------------|
| `GITHUB_TOKEN`  | Fine‑grained PAT w/ repo read   |
| `CONCURRENCY`   | (opt) #threads for GitHub calls |

## Extending Beyond 30 Repos
Adjust `--max-results` flag; the code automatically paginates and uses
GitHub Search API rate limits (30 req/min) with back‑off.
