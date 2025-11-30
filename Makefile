.PHONY: install dev-install test lint format run-experiment clean help

# Default Python interpreter
PYTHON ?= python

# Default experiment configuration
CONFIG ?= config.yaml
SEED ?= 42

help:
	@echo "Quadcopter Tracking Research - Available Commands"
	@echo "================================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install      - Install package and dependencies"
	@echo "  make dev-install  - Install with development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test         - Run test suite"
	@echo "  make lint         - Run linter (ruff)"
	@echo "  make format       - Auto-format code (ruff)"
	@echo ""
	@echo "Experiments:"
	@echo "  make run-experiment           - Run experiment with default config"
	@echo "  make run-experiment CONFIG=x  - Run experiment with custom config"
	@echo "  make run-experiment SEED=x    - Run experiment with custom seed"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean        - Remove build artifacts and cache"
	@echo ""
	@echo "Examples:"
	@echo "  make run-experiment CONFIG=configs/lqr.yaml SEED=123"

install:
	$(PYTHON) -m pip install -e .

dev-install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check src/ tests/

format:
	$(PYTHON) -m ruff format src/ tests/
	$(PYTHON) -m ruff check --fix src/ tests/

run-experiment:
	@echo "Running experiment with config=$(CONFIG), seed=$(SEED)"
	$(PYTHON) -c "from src.utils import load_config; from src.env import QuadcopterEnv; \
		config = load_config(); \
		print('Loaded configuration:'); \
		print(f'  Seed: {config[\"seed\"]}'); \
		print(f'  Episode length: {config[\"episode_length\"]}s'); \
		print(f'  Target radius: {config[\"target\"][\"radius_requirement\"]}m'); \
		print(f'  Motion type: {config[\"target\"][\"motion_type\"]}'); \
		env = QuadcopterEnv(config); \
		obs = env.reset(seed=$(SEED)); \
		print('Environment initialized successfully!')"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
