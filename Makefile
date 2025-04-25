# Install development dependencies
install-dev:
	uv pip install -e .[dev]

# Run ruff checks and fixes
fix:
	ruff check --fix .
	ruff format .

# Run tests using pytest
test:
	PYTHONPATH=. pytest tests/

# Install Playwright browser binaries (run after install-dev)
install-browsers:
	@echo "Ensuring Playwright browsers are installed using venv's playwright..."
	@.venv/bin/playwright install # Use the venv playwright explicitly
	# Note: If the above fails due to missing OS dependencies, you might need:
	# .venv/bin/playwright install --with-deps
	# This attempts to install OS dependencies (may require sudo) but might fail on unsupported OS.

# Run an example search command (using API checker)
run:
	@echo "Running example search (API, last 10 days, min 200 stars)..."
	@.venv/bin/repobird-leadgen search --label "good first issue" --language python --max-results 10 --recent-days 10 --min-stars 200

# Run an example search command using the browser checker
run-browser: install-browsers
	@echo "Running example search (Browser, last 10 days, min 200 stars)..."
	@.venv/bin/repobird-leadgen search --label "good first issue" --language python --max-results 10 --recent-days 10 --min-stars 200 --use-browser-checker

# Update a specific cache file with missing issue numbers (requires label)
# Example: make update-cache CACHE_FILE=cache/raw_repos_label_good_first_issue_lang_python_stars_200_days_10.jsonl LABEL="good first issue"
update-cache: install-browsers
	@echo "Updating cache file: $(CACHE_FILE) for label: $(LABEL)"
	@if [ -z "$(CACHE_FILE)" ] || [ -z "$(LABEL)" ]; then \
		echo "[Error] CACHE_FILE and LABEL must be set. Example:"; \
		echo "make update-cache CACHE_FILE=path/to/your.jsonl LABEL=\"your label\""; \
		exit 1; \
	fi
	@.venv/bin/python scripts/update_cache_issue_numbers.py "$(CACHE_FILE)" --label "$(LABEL)"

.PHONY: install-dev fix test install-browsers run run-browser update-cache
