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

# Run an example search command
run:
	@echo "Running example search (last 10 days, min 200 stars)..."
	@.venv/bin/repobird-leadgen search --label "good first issue" --language python --max-results 10 --recent-days 10 --min-stars 200

.PHONY: install-dev fix test run-search-example
