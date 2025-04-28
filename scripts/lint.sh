#!/bin/bash
# Lint script to run ruff and mypy on a single file

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <filename>"
    exit 1
fi

FILE="$1"

# Skip non-Python files
if [[ "$FILE" != *.py ]]; then
    echo "Skipping non-Python file: $FILE"
    exit 0
fi

ruff check --select I --fix "$FILE"
ruff check --fix "$FILE"
ruff format "$FILE"
# mypy "$FILE"

