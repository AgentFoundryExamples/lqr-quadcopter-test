.PHONY: install dev-install test lint format run-experiment train-deep train-pid train-lqr train-riccati-lqr eval-deep eval-pid eval-lqr eval-riccati-lqr eval-baseline-stationary eval-baseline-circular compare-controllers generate-comparison-report clean help

# Default Python interpreter
PYTHON ?= python

# Default experiment configuration
CONFIG ?= config.yaml
SEED ?= 42
EPOCHS ?= 10
EPISODES ?= 5

# Default motion type for evaluation
MOTION_TYPE ?= circular

# Comparison configuration
COMPARISON_CONFIG ?= experiments/configs/comparison_default.yaml

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
	@echo "=== WORKFLOW 1: Baseline PID/LQR Evaluation ==="
	@echo "  make eval-pid                      - Evaluate PID controller"
	@echo "  make eval-lqr                      - Evaluate LQR controller"
	@echo "  make eval-riccati-lqr              - Evaluate Riccati-LQR controller"
	@echo "  make eval-baseline-stationary      - Evaluate PID+LQR on stationary target"
	@echo "  make eval-baseline-circular        - Evaluate PID+LQR on circular target"
	@echo ""
	@echo "=== WORKFLOW 2: Deep Controller Training ==="
	@echo "  make train-deep                    - Train deep controller"
	@echo "  make train-deep EPOCHS=100         - Train with custom epochs"
	@echo "  make eval-deep                     - Evaluate trained deep controller"
	@echo ""
	@echo "=== WORKFLOW 3: Controller Comparison ==="
	@echo "  make compare-controllers           - Run comparison across controllers"
	@echo "  make generate-comparison-report    - Generate side-by-side metrics report"
	@echo ""
	@echo "Legacy/Other Commands:"
	@echo "  make train-pid              - Run PID controller evaluation loop"
	@echo "  make train-lqr              - Run LQR controller evaluation loop"
	@echo "  make train-riccati-lqr      - Run Riccati-LQR controller evaluation loop"
	@echo "  make run-experiment         - Run experiment with custom config"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean        - Remove build artifacts and cache"
	@echo ""
	@echo "Examples:"
	@echo "  make train-deep EPOCHS=50 SEED=123"
	@echo "  make eval-pid EPISODES=20"
	@echo "  make compare-controllers MOTION_TYPE=stationary"
	@echo "  make run-experiment CONFIG=experiments/configs/training_fast.yaml"

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

# Training commands for different controllers
train-deep:
	$(PYTHON) -m quadcopter_tracking.train --controller deep --epochs $(EPOCHS) --episodes-per-epoch $(EPISODES) --seed $(SEED)

train-pid:
	$(PYTHON) -m quadcopter_tracking.train --controller pid --epochs $(EPOCHS) --episodes-per-epoch $(EPISODES) --seed $(SEED)

train-lqr:
	$(PYTHON) -m quadcopter_tracking.train --controller lqr --epochs $(EPOCHS) --episodes-per-epoch $(EPISODES) --seed $(SEED)

train-riccati-lqr:
	$(PYTHON) -m quadcopter_tracking.train --controller riccati_lqr --epochs $(EPOCHS) --episodes-per-epoch $(EPISODES) --seed $(SEED)

# Evaluation commands
eval-deep:
	$(PYTHON) -m quadcopter_tracking.eval --controller deep --episodes $(EPISODES) --seed $(SEED)

eval-pid:
	$(PYTHON) -m quadcopter_tracking.eval --controller pid --episodes $(EPISODES) --seed $(SEED)

eval-lqr:
	$(PYTHON) -m quadcopter_tracking.eval --controller lqr --episodes $(EPISODES) --seed $(SEED)

eval-riccati-lqr:
	$(PYTHON) -m quadcopter_tracking.eval --controller riccati_lqr --episodes $(EPISODES) --seed $(SEED)

# =============================================================================
# WORKFLOW 1: Baseline PID/LQR Evaluation
# =============================================================================
# Evaluate classical controllers on predefined baseline configurations
# These targets use YAML configs to specify documented controller gains

eval-baseline-stationary:
	@echo "=== Evaluating PID on stationary target ==="
	$(PYTHON) -m quadcopter_tracking.eval --controller pid --config experiments/configs/eval_stationary_baseline.yaml --episodes $(EPISODES) --seed $(SEED) --output-dir reports/baseline_stationary_pid
	@echo ""
	@echo "=== Evaluating LQR on stationary target ==="
	$(PYTHON) -m quadcopter_tracking.eval --controller lqr --config experiments/configs/eval_stationary_baseline.yaml --episodes $(EPISODES) --seed $(SEED) --output-dir reports/baseline_stationary_lqr
	@echo ""
	@echo "Baseline evaluation complete. Results in reports/baseline_stationary_*/"

eval-baseline-circular:
	@echo "=== Evaluating PID on circular target ==="
	$(PYTHON) -m quadcopter_tracking.eval --controller pid --config experiments/configs/eval_circular_baseline.yaml --episodes $(EPISODES) --seed $(SEED) --output-dir reports/baseline_circular_pid
	@echo ""
	@echo "=== Evaluating LQR on circular target ==="
	$(PYTHON) -m quadcopter_tracking.eval --controller lqr --config experiments/configs/eval_circular_baseline.yaml --episodes $(EPISODES) --seed $(SEED) --output-dir reports/baseline_circular_lqr
	@echo ""
	@echo "Baseline evaluation complete. Results in reports/baseline_circular_*/"

# =============================================================================
# WORKFLOW 3: Controller Comparison
# =============================================================================
# Compare multiple controllers and generate side-by-side metrics
# Note: Using - prefix to continue even if individual evaluations fail criteria

compare-controllers:
	@echo "=== Running Controller Comparison ==="
	@echo "Evaluating PID controller..."
	-@$(PYTHON) -m quadcopter_tracking.eval --controller pid --motion-type $(MOTION_TYPE) --episodes $(EPISODES) --seed $(SEED) --output-dir reports/comparison/pid
	@echo ""
	@echo "Evaluating LQR controller..."
	-@$(PYTHON) -m quadcopter_tracking.eval --controller lqr --motion-type $(MOTION_TYPE) --episodes $(EPISODES) --seed $(SEED) --output-dir reports/comparison/lqr
	@echo ""
	@echo "Controller evaluation complete. Run 'make generate-comparison-report' to generate summary."

generate-comparison-report:
	@echo "=== Generating Comparison Report ==="
	@$(PYTHON) scripts/generate_comparison_report.py --report-dir reports/comparison

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
