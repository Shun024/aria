.PHONY: install test lint typecheck ci clean

install:
	python3.11 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e ".[dev]"

test:
	.venv/bin/pytest tests/ -v

test-unit:
	.venv/bin/pytest tests/unit/ -v

test-integration:
	.venv/bin/pytest tests/integration/ -v

test-cov:
	.venv/bin/pytest tests/ --cov=aria --cov-report=html

lint:
	.venv/bin/ruff check aria/ tests/

lint-fix:
	.venv/bin/ruff check --fix aria/ tests/

typecheck:
	.venv/bin/mypy aria/

ci: lint typecheck test

kafka-up:
	docker-compose up -d zookeeper kafka kafka-ui

kafka-down:
	docker-compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .coverage htmlcov/