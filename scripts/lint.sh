#!/bin/bash
set -e
echo "Running ruff check..."
uv run ruff check . --fix

echo "Running ruff format..."
uv run ruff format .

echo "Running mypy..."
uv run mypy src/ --config-file=pyproject.toml