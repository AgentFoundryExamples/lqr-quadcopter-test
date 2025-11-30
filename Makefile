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

# Note: CONFIG and SEED are validated by Python code, not shell interpolation
# to prevent command injection vulnerabilities
run-experiment:
	@echo "Running experiment with config=$(CONFIG), seed=$(SEED)"
	@$(PYTHON) -c "\
from pathlib import Path; \
from quadcopter_tracking.utils import load_config; \
from quadcopter_tracking.env import QuadcopterEnv; \
import sys; \
seed_str = '$(SEED)'; \
seed = int(seed_str) if seed_str.lstrip('-').isdigit() else (print('Error: SEED must be an integer', file=sys.stderr) or sys.exit(1)); \
config_path = '$(CONFIG)'; \
config_file = Path(config_path) if config_path != 'config.yaml' or Path(config_path).exists() else None; \
config = load_config(config_path=config_file); \
print('Loaded configuration:'); \
print(f'  Seed: {config[\"seed\"]}'); \
print(f'  Episode length: {config[\"episode_length\"]}s'); \
print(f'  Target radius: {config[\"target\"][\"radius_requirement\"]}m'); \
print(f'  Motion type: {config[\"target\"][\"motion_type\"]}'); \
env = QuadcopterEnv(config); \
obs = env.reset(seed=seed); \
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
