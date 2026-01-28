# Makefile for mini-vllm development

.PHONY: install lint format test ci clean help

PYTHON := python3
PIP := pip

help:
	@echo "Available commands:"
	@echo "  make install   - Install dependencies (including dev tools)"
	@echo "  make lint      - Run code quality checks (isort, yapf, flake8, mypy)"
	@echo "  make format    - Auto-format code (isort, yapf)"
	@echo "  make test      - Run tests"
	@echo "  make ci        - Run full CI workflow locally (lint + test)"
	@echo "  make clean     - Clean up build artifacts and cache"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]" || $(PIP) install -e .
	$(PIP) install flake8 isort yapf mypy types-PyYAML types-tqdm pytest pytest-cov
	# Install torch CPU if not present (optional suggestion)
	# $(PIP) install torch --index-url https://download.pytorch.org/whl/cpu

format:
	isort .
	yapf -i -r .

lint:
	@echo "Running isort check..."
	isort . --check --diff
	@echo "Running yapf check..."
	yapf -r . --diff
	@echo "Running flake8..."
	flake8 .
	@echo "Running mypy..."
	mypy minivllm

test:
	$(PYTHON) tests/run_tests.py --coverage

ci: lint test

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .coverage htmlcov
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
