# Makefile for mini-vllm development

.PHONY: install lint format test ci clean help check-quality

PYTHON := python3
PIP := pip

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies (including dev tools)"
	@echo "  make check-quality - Run quick code quality verification"
	@echo "  make lint         - Run code quality checks (ruff, mypy)"
	@echo "  make format       - Auto-format code (ruff, black)"
	@echo "  make test         - Run tests"
	@echo "  make ci           - Run full CI workflow locally (lint + test)"
	@echo "  make clean        - Clean up build artifacts and cache"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]" || $(PIP) install -e .
	$(PIP) install ruff black mypy types-PyYAML types-tqdm pytest pytest-cov
	# Install torch CPU if not present (optional suggestion)
	# $(PIP) install torch --index-url https://download.pytorch.org/whl/cpu

format:
	@echo "Formatting with ruff and black..."
	ruff format .
	ruff check --fix .

lint:
	@echo "Running ruff linter..."
	ruff check .
	@echo "Running mypy type checker..."
	mypy minivllm

test:
	@echo "Running tests..."
	$(PYTHON) -m pytest tests/ -v --cov=minivllm --cov-report=term-missing || $(PYTHON) test_code_quality.py

ci: check-quality lint test

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .coverage htmlcov .ruff_cache
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
