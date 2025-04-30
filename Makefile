# Install development dependencies
install-dev:
	uv pip install -e .[dev]

# Run ruff checks and fixes
fix:
	ruff check --fix .
	ruff format .

# Define shell and command prefix to run within the activated venv
SHELL := /bin/bash
VENV_RUN := bash -c 'source .venv/bin/activate && PYTHONPATH=. exec "$$@"' --

# Run tests using pytest
test:
	$(VENV_RUN) pytest tests/

# Install Playwright browser binaries (run after install-dev)
install-browsers:
	@echo "Ensuring Playwright browsers are installed using venv's playwright..."
	@$(VENV_RUN) playwright install # Use the venv playwright explicitly
	# Note: If the above fails due to missing OS dependencies, you might need:
	# $(VENV_RUN) playwright install --with-deps
	# This attempts to install OS dependencies (may require sudo) but might fail on unsupported OS.

# Run an example search command (using API checker)
run:
	@echo "Running example search (API, last 10 days, min 200 stars)..."
	@$(VENV_RUN) repobird-leadgen search --label "good first issue" --language python --max-results 10 --recent-days 10 --min-stars 200

run-browser:
	@echo "Running example search (Browser, label 'good first issue', lang python, max 10 results, recent 10 days, min 100 stars, max issue age 30 days, max 0 linked PRs)..."
	@$(VENV_RUN) repobird-leadgen search --label "good first issue" --language python --max-results 100 --recent-days 60 --min-stars 100 --max-issue-age-days 60 --max-linked-prs 0 --use-browser-checker

# Update a specific cache file with missing issue numbers (requires label)
# Example: make update-cache CACHE_FILE=cache/raw_repos_label_good_first_issue_lang_python_stars_200_days_10.jsonl LABEL="good first issue"
update-cache: install-browsers
	@echo "Updating cache file: $(CACHE_FILE) for label: $(LABEL)"
	@if [ -z "$(CACHE_FILE)" ] || [ -z "$(LABEL)" ]; then \
		echo "[Error] CACHE_FILE and LABEL must be set. Example:"; \
		echo "make update-cache CACHE_FILE=path/to/your.jsonl LABEL=\"your label\""; \
		exit 1; \
	fi
	@$(VENV_RUN) python scripts/update_cache_issue_numbers.py "$(CACHE_FILE)" --label "$(LABEL)"

# Run aider with linting
aider:
	@aider --no-auto-commit --lint-cmd scripts/lint.sh --auto-lint

# Run aider with linting and testing
aider-test:
	@aider --no-auto-commit --lint-cmd scripts/lint.sh --auto-lint --test-cmd 'make test' --auto-test

# Run the enrichment process on a specific cache file
# Usage: make enrich FILE=cache/your_file.jsonl [ARGS="--concurrency 5"]
enrich:
	@echo "Running enrichment command on file: $(FILE)"
	@if [ -z "$(FILE)" ]; then \
		echo "[Error] FILE argument must be set. Example:"; \
		echo "make enrich FILE=path/to/your.jsonl"; \
		exit 1; \
	fi
	$(VENV_RUN) repobird-leadgen enrich "$(FILE)" $(ARGS)

# Interactively review issues in a cache file
# Usage:
#   make review                 # CLI will prompt for file selection from cache/
#   make review ARGS="cache/your_file.jsonl" # Review specific file
#   make review ARGS="--open"   # Prompt for file, auto-open URLs
#   make review ARGS="cache/your_file.jsonl --open" # Specific file + auto-open
review:
	@echo "Running review command..."
	$(VENV_RUN) repobird-leadgen review $(ARGS)

# Run the post-processing filter on an enriched file
# Usage: make post-process FILE=output/enriched_*.jsonl
post-process:
	@echo "Running post-processing filter on file: $(FILE)"
	@if [ -z "$(FILE)" ]; then \
		echo "[Error] FILE argument must be set. Example:"; \
		echo "make post-process FILE=output/enriched_your_file.jsonl"; \
		exit 1; \
	fi
	$(VENV_RUN) repobird-leadgen post-process "$(FILE)"

# Run the PR filter script on an enriched/filtered file (modifies in-place)
# Usage: make filter-prs FILE=output/enriched_*.jsonl
# Usage: make filter-prs FILE=output/filtered_*.jsonl
filter-prs:
	@echo "Running PR filter script on file (will modify in-place): $(FILE)"
	@if [ -z "$(FILE)" ]; then \
		echo "[Error] FILE argument must be set. Example:"; \
		echo "make filter-prs FILE=output/your_file.jsonl"; \
		exit 1; \
	fi
	$(VENV_RUN) python scripts/filter_issues_by_prs.py "$(FILE)"

.PHONY: install-dev fix test install-browsers run run-browser update-cache aider aider-test enrich review post-process filter-prs
