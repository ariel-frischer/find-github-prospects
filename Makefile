# Install development dependencies
install-dev:
	uv pip install -e .[dev]

# Run ruff checks and fixes
fix:
	ruff check --fix .
	ruff format .

# Run tests using pytest
test:
	pytest tests/

.PHONY: install-dev fix test
