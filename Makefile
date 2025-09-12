# Makefile for REER × DSPy × MLX Social Posting Pack

.PHONY: help install install-dev setup lint format test test-unit test-integration test-e2e coverage clean docs serve-docs build check pre-commit

# Default target
help:
	@echo "REER × DSPy × MLX Social Posting Pack"
	@echo "Available targets:"
	@echo "  install       - Install production dependencies"
	@echo "  install-dev   - Install development dependencies"
	@echo "  setup         - Complete development setup"
	@echo "  lint          - Run linting checks"
	@echo "  format        - Format code with black and ruff"
	@echo "  test          - Run all tests"
	@echo "  test-unit     - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-e2e      - Run end-to-end tests only"
	@echo "  coverage      - Run tests with coverage report"
	@echo "  clean         - Clean up build artifacts"
	@echo "  docs          - Build documentation"
	@echo "  serve-docs    - Serve documentation locally"
	@echo "  build         - Build package"
	@echo "  check         - Run all checks (lint, format, test)"
	@echo "  pre-commit    - Install pre-commit hooks"

# Installation targets
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pip install -r requirements-social.txt
	pip install -e .

setup: install-dev pre-commit
	@echo "✅ Development environment setup complete!"

# Code quality targets
lint:
	ruff check --select E,W,F,I,B,C4,UP \
	  --ignore E501,B008,C901,PLR,PL,SIM,TRY,ARG,TRY301,PLR2004,B007,B904 \
	  core config tools validate_dspy_implementation.py
	mypy --explicit-package-bases --namespace-packages core config tools

format:
	black .
	ruff format .
	ruff check --fix .

# Testing targets
test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-e2e:
	pytest tests/e2e/ -v

coverage:
	coverage run -m pytest tests/
	coverage report --show-missing
	coverage html

# Documentation targets
docs:
	@echo "Building documentation..."
	# Add documentation build commands here

serve-docs:
	@echo "Serving documentation locally..."
	# Add documentation serve commands here

# Build targets
build:
	python -m build

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Git hooks
pre-commit:
	pre-commit install
	pre-commit install --hook-type commit-msg
	pre-commit install --hook-type pre-push

# Check all
check: lint test
	@echo "✅ All checks passed!"
