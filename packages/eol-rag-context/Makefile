# Makefile for EOL RAG Context
# Provides convenient commands for development and testing

.PHONY: help install install-dev install-uv test test-unit test-integration coverage lint format clean redis-start redis-stop setup-test

# Variables
PYTHON := python3
UV := uv
VENV := .venv
REDIS_STACK := redis-stack-server

# Default target
help:
	@echo "EOL RAG Context - Development Commands"
	@echo "======================================"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install package and dependencies"
	@echo "  make install-dev    Install with development dependencies"
	@echo "  make install-uv     Install using uv (fast)"
	@echo "  make setup-test     Setup complete test environment"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run all tests"
	@echo "  make test-unit      Run unit tests only"
	@echo "  make test-integration Run integration tests only"
	@echo "  make coverage       Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           Run linters (ruff, mypy)"
	@echo "  make format         Format code with black"
	@echo ""
	@echo "Redis:"
	@echo "  make redis-start    Start Redis Stack Server"
	@echo "  make redis-stop     Stop Redis Stack Server"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          Remove build artifacts and cache"

# Installation targets
install:
	@echo "Installing eol-rag-context..."
	@if command -v uv >/dev/null 2>&1; then \
		echo "Using uv for fast installation..."; \
		uv pip install -e .; \
	else \
		echo "Using pip for installation..."; \
		$(PYTHON) -m pip install -e .; \
	fi

install-dev:
	@echo "Installing eol-rag-context with dev dependencies..."
	@if command -v uv >/dev/null 2>&1; then \
		echo "Using uv for fast installation..."; \
		uv pip install -e ".[dev]"; \
	else \
		echo "Using pip for installation..."; \
		$(PYTHON) -m pip install -e ".[dev]"; \
	fi

install-uv:
	@echo "Installing with uv (ultra-fast)..."
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "Installing uv first..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	uv venv $(VENV)
	uv pip install -e ".[dev]"
	@echo "Activate virtual environment with: source $(VENV)/bin/activate"

# Testing targets
test: redis-start
	@echo "Running all tests..."
	@source $(VENV)/bin/activate && \
		PYTHONPATH=src:$$PYTHONPATH pytest tests/ -v

test-unit:
	@echo "Running unit tests..."
	@source $(VENV)/bin/activate && \
		PYTHONPATH=src:$$PYTHONPATH pytest tests/ \
		-v --ignore=tests/integration/

test-integration: redis-start
	@echo "Running integration tests..."
	@source $(VENV)/bin/activate && \
		PYTHONPATH=src:$$PYTHONPATH pytest tests/integration/ -v

coverage: redis-start
	@echo "Running tests with coverage..."
	@source $(VENV)/bin/activate && \
		PYTHONPATH=src:$$PYTHONPATH pytest tests/ \
		--cov=eol.rag_context \
		--cov-report=term-missing \
		--cov-report=html:coverage/html \
		--cov-report=xml:coverage/coverage.xml
	@echo "Coverage report: coverage/html/index.html"

# Code quality targets
lint:
	@echo "Running linters..."
	@source $(VENV)/bin/activate && \
		ruff check src/ tests/ && \
		mypy src/eol/rag_context --ignore-missing-imports

format:
	@echo "Formatting code..."
	@source $(VENV)/bin/activate && \
		black src/ tests/ && \
		ruff check --fix src/ tests/

# Redis management
redis-start:
	@echo "Starting Redis Stack Server..."
	@if ! redis-cli ping >/dev/null 2>&1; then \
		if command -v $(REDIS_STACK) >/dev/null 2>&1; then \
			$(REDIS_STACK) --daemonize yes; \
			sleep 2; \
			echo "Redis Stack Server started"; \
		else \
			echo "Warning: Redis Stack not installed. Install with:"; \
			echo "  brew install --cask redis-stack-server"; \
		fi \
	else \
		echo "Redis is already running"; \
	fi

redis-stop:
	@echo "Stopping Redis..."
	@redis-cli shutdown 2>/dev/null || true
	@echo "Redis stopped"

# Setup complete test environment
setup-test:
	@echo "Setting up test environment..."
	@./setup_test_environment.sh

# Cleanup
clean:
	@echo "Cleaning up..."
	@rm -rf build/ dist/ *.egg-info
	@rm -rf coverage/ htmlcov/ .coverage .coverage.*
	@rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@echo "Cleanup complete"

# Development workflow shortcuts
.PHONY: dev
dev: install-dev redis-start
	@echo "Development environment ready!"
	@echo "Run 'make test' to run tests"
	@echo "Run 'make coverage' to check coverage"

.PHONY: ci
ci: lint test coverage
	@echo "CI checks complete!"

.PHONY: all
all: clean install-dev lint format test coverage
	@echo "Full build and test complete!"