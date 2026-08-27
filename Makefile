.PHONY: install dev test test-unit test-integration lint format run clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=src/aion_knowledge

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

lint:
	ruff check src/aion_knowledge/
	ruff check tests/

format:
	ruff format src/aion_knowledge/
	ruff format tests/

run:
	uvicorn aion_knowledge.api.app:app --reload

clean:
	rm -rf build/ dist/ *.egg-info/ __pycache__/ .pytest_cache/ .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
