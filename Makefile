# PropelAI Makefile
# Common commands for development, testing, and deployment

.PHONY: help install install-dev run dev test test-cov lint format typecheck security clean docker-build docker-up docker-down db-migrate db-upgrade pre-commit docs

# Default target
help:
	@echo "PropelAI - Autonomous Proposal Operating System"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Development:"
	@echo "  install       Install production dependencies"
	@echo "  install-dev   Install all dependencies (including dev)"
	@echo "  run           Run the API server"
	@echo "  dev           Run the API server with auto-reload"
	@echo ""
	@echo "Testing:"
	@echo "  test          Run all tests"
	@echo "  test-cov      Run tests with coverage report"
	@echo "  test-unit     Run unit tests only"
	@echo "  test-int      Run integration tests only"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint          Run all linters (ruff, flake8)"
	@echo "  format        Format code with black and isort"
	@echo "  typecheck     Run mypy type checker"
	@echo "  security      Run security checks (bandit, safety)"
	@echo "  pre-commit    Run all pre-commit hooks"
	@echo "  check         Run all quality checks (format, lint, typecheck, security)"
	@echo ""
	@echo "Database:"
	@echo "  db-migrate    Create a new database migration"
	@echo "  db-upgrade    Apply database migrations"
	@echo "  db-reset      Reset database (WARNING: destroys data)"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build  Build Docker images"
	@echo "  docker-up     Start Docker containers"
	@echo "  docker-down   Stop Docker containers"
	@echo "  docker-logs   View Docker container logs"
	@echo ""
	@echo "Documentation:"
	@echo "  docs          Build documentation"
	@echo "  docs-serve    Serve documentation locally"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean         Remove build artifacts and caches"

# =============================================================================
# Installation
# =============================================================================

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

# =============================================================================
# Running
# =============================================================================

run:
	python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

dev:
	python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# =============================================================================
# Testing
# =============================================================================

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=api --cov=agents --cov-report=html --cov-report=term-missing

test-unit:
	pytest tests/ -v -m "unit"

test-int:
	pytest tests/ -v -m "integration"

test-fast:
	pytest tests/ -v -m "not slow" -x

# =============================================================================
# Code Quality
# =============================================================================

lint:
	ruff check api/ agents/ tests/
	flake8 api/ agents/ tests/ --max-line-length=100 --ignore=E501,W503

format:
	black api/ agents/ tests/
	isort api/ agents/ tests/

format-check:
	black --check api/ agents/ tests/
	isort --check-only api/ agents/ tests/

typecheck:
	mypy api/ agents/ --ignore-missing-imports

security:
	bandit -r api/ agents/ -c pyproject.toml
	safety check -r requirements.txt

pre-commit:
	pre-commit run --all-files

check: format-check lint typecheck security
	@echo "All quality checks passed!"

# =============================================================================
# Database
# =============================================================================

db-migrate:
	@read -p "Enter migration message: " msg; \
	alembic revision --autogenerate -m "$$msg"

db-upgrade:
	alembic upgrade head

db-downgrade:
	alembic downgrade -1

db-reset:
	@echo "WARNING: This will destroy all data!"
	@read -p "Are you sure? [y/N] " confirm; \
	if [ "$$confirm" = "y" ]; then \
		alembic downgrade base; \
		alembic upgrade head; \
	fi

# =============================================================================
# Docker
# =============================================================================

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-clean:
	docker-compose down -v --rmi local

# =============================================================================
# Documentation
# =============================================================================

docs:
	mkdocs build

docs-serve:
	mkdocs serve

# =============================================================================
# Cleanup
# =============================================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .eggs/ 2>/dev/null || true
	@echo "Cleaned up build artifacts and caches"

# =============================================================================
# Release
# =============================================================================

version:
	@python -c "import toml; print(toml.load('pyproject.toml')['project']['version'])"

bump-patch:
	@echo "Bumping patch version..."
	@python -c "import toml; \
		data = toml.load('pyproject.toml'); \
		v = data['project']['version'].split('.'); \
		v[2] = str(int(v[2]) + 1); \
		data['project']['version'] = '.'.join(v); \
		toml.dump(data, open('pyproject.toml', 'w')); \
		print(f\"Version bumped to {data['project']['version']}\")"

bump-minor:
	@echo "Bumping minor version..."
	@python -c "import toml; \
		data = toml.load('pyproject.toml'); \
		v = data['project']['version'].split('.'); \
		v[1] = str(int(v[1]) + 1); v[2] = '0'; \
		data['project']['version'] = '.'.join(v); \
		toml.dump(data, open('pyproject.toml', 'w')); \
		print(f\"Version bumped to {data['project']['version']}\")"

bump-major:
	@echo "Bumping major version..."
	@python -c "import toml; \
		data = toml.load('pyproject.toml'); \
		v = data['project']['version'].split('.'); \
		v[0] = str(int(v[0]) + 1); v[1] = '0'; v[2] = '0'; \
		data['project']['version'] = '.'.join(v); \
		toml.dump(data, open('pyproject.toml', 'w')); \
		print(f\"Version bumped to {data['project']['version']}\")"
