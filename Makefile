# IoMT IDS Makefile

.PHONY: help install test train clean docs

help: ## Show this help message
	@echo "IoMT IDS - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install pytest black flake8 mypy

test: ## Run tests
	python -m pytest tests/ -v

test-unit: ## Run unit tests
	python -m pytest tests/unit/ -v

test-integration: ## Run integration tests
	python -m pytest tests/integration/ -v

train: ## Train the model
	python scripts/train.py

train-advanced: ## Train with hyperparameter optimization
	python scripts/train_advanced.py

predict: ## Run inference
	python scripts/predict.py

explain: ## Generate XAI explanations
	python scripts/explain.py

api: ## Start API server
	python service/api/main.py

stream: ## Start stream processor
	python service/stream_processor/main.py

notebook: ## Start Jupyter notebook
	jupyter notebook

clean: ## Clean temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/

clean-data: ## Clean processed data
	rm -rf data/processed/*.parquet
	rm -rf data/interim/*

clean-models: ## Clean model artifacts
	rm -rf artifacts/models/*
	rm -rf artifacts/results/*

clean-all: clean clean-data clean-models ## Clean everything

format: ## Format code with black
	black src/ scripts/ tests/
	black --check src/ scripts/ tests/

lint: ## Lint code with flake8
	flake8 src/ scripts/ tests/

type-check: ## Type check with mypy
	mypy src/ scripts/

docs: ## Generate documentation
	python -m sphinx docs/ docs/_build/

docker-build: ## Build Docker image
	docker build -t iomt-ids .

docker-run: ## Run Docker container
	docker run -p 8000:8000 -v $(PWD)/data:/app/data iomt-ids

setup: install ## Setup development environment
	pre-commit install

pre-commit: ## Run pre-commit hooks
	pre-commit run --all-files

all: clean install test train ## Run full pipeline

help-dev: ## Show development help
	@echo "Development Commands:"
	@echo "  make install-dev    - Install development dependencies"
	@echo "  make format         - Format code"
	@echo "  make lint           - Lint code"
	@echo "  make type-check     - Type check"
	@echo "  make pre-commit     - Run pre-commit hooks"









