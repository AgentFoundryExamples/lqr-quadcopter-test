# Configuration Files

This directory contains YAML configuration files for training, evaluation, and tuning quadcopter controllers.

## Directory Structure

```
configs/
├── README.md           # This file - config index and migration guide
├── training/           # Deep controller training configs
│   ├── training_default.yaml      # Standard training (100 epochs)
│   ├── training_fast.yaml         # Quick testing (20 epochs)
│   ├── training_large.yaml        # Extended training (500 epochs)
│   ├── training_imitation.yaml    # Imitation learning from PID/LQR
│   ├── diagnostics_stationary.yaml # Training diagnostics on stationary targets
│   └── diagnostics_linear.yaml    # Training diagnostics on linear targets
├── evaluation/         # Controller evaluation configs
│   ├── eval_stationary_baseline.yaml  # Stationary target baseline
│   ├── eval_linear_baseline.yaml      # Linear motion baseline
│   ├── eval_circular_baseline.yaml    # Circular motion baseline
│   ├── eval_riccati_lqr_baseline.yaml # Riccati-LQR specific baseline
│   └── comparison_default.yaml        # Side-by-side controller comparison
└── tuning/             # Auto-tuning configs (grid/random/CMA-ES)
    ├── tuning_pid.yaml            # PID tuning (stationary)
    ├── tuning_pid_linear.yaml     # PID tuning for linear motion
    ├── tuning_lqr.yaml            # LQR tuning (stationary)
    ├── tuning_lqr_linear.yaml     # LQR tuning for linear motion
    ├── tuning_riccati.yaml        # Riccati-LQR tuning (stationary)
    ├── tuning_riccati_linear.yaml # Riccati-LQR tuning for linear motion
    └── tuning_cma_es.yaml         # CMA-ES optimization example
```

## Configuration Index

### Training Configurations

| Config | Controller | Motion | Use Case | Training Time |
|--------|------------|--------|----------|---------------|
| `training/training_default.yaml` | Deep | Circular | Standard training | ~15 min |
| `training/training_fast.yaml` | Deep | Circular | Quick testing | ~2 min |
| `training/training_large.yaml` | Deep | Circular | Extended training | ~1+ hour |
| `training/training_imitation.yaml` | Deep | Configurable | Learn from PID/LQR | ~10 min |
| `training/diagnostics_stationary.yaml` | Deep | Stationary | Training analysis | ~5 min |
| `training/diagnostics_linear.yaml` | Deep | Linear | Training analysis | ~5 min |

### Evaluation Configurations

| Config | Controller | Motion | Expected On-Target |
|--------|------------|--------|-------------------|
| `evaluation/eval_stationary_baseline.yaml` | PID/LQR | Stationary | >80% |
| `evaluation/eval_linear_baseline.yaml` | PID/LQR/Riccati | Linear | 70-90% |
| `evaluation/eval_circular_baseline.yaml` | PID/LQR | Circular | 70-90% |
| `evaluation/eval_riccati_lqr_baseline.yaml` | Riccati-LQR | Configurable | Varies |
| `evaluation/comparison_default.yaml` | Multiple | Configurable | N/A |

### Tuning Configurations

| Config | Controller | Motion | Strategy | Iterations |
|--------|------------|--------|----------|------------|
| `tuning/tuning_pid.yaml` | PID | Stationary | Random | 50 |
| `tuning/tuning_pid_linear.yaml` | PID | Linear | Random | 50 |
| `tuning/tuning_lqr.yaml` | LQR | Stationary | Random | 50 |
| `tuning/tuning_lqr_linear.yaml` | LQR | Linear | Random | 50 |
| `tuning/tuning_riccati.yaml` | Riccati-LQR | Stationary | Random | 50 |
| `tuning/tuning_riccati_linear.yaml` | Riccati-LQR | Linear | Random | 50 |
| `tuning/tuning_cma_es.yaml` | PID | Stationary | CMA-ES | 100 |

## Controller Capability Matrix

| Controller | Training | Feedforward | DARE Solver | Imitation Teacher | Recommended Use |
|------------|----------|-------------|-------------|-------------------|-----------------|
| **PID** | No | ✅ Optional | N/A | ✅ Yes | Stationary/slow targets |
| **LQR** | No | ✅ Optional | N/A | ✅ Yes | Quick prototyping |
| **Riccati-LQR** | No | ✅ Optional | ✅ Yes | ✅ Preferred | Research baselines, imitation |
| **Deep** | ✅ Yes | Learned | N/A | N/A | Complex/nonlinear scenarios |

### Feedforward Support

All classical controllers (PID, LQR, Riccati-LQR) support optional feedforward for improved tracking of moving targets:

```yaml
pid:
  feedforward_enabled: true
  ff_velocity_gain: [0.1, 0.1, 0.1]
  ff_acceleration_gain: [0.05, 0.05, 0.0]
```

**When to enable feedforward:**
- Moving targets (linear, circular, sinusoidal)
- High-speed tracking scenarios
- Reducing phase lag

**When to keep disabled:**
- Stationary targets (default, no benefit)
- Initial tuning/debugging

### CMA-ES Auto-Tuning

CMA-ES (Covariance Matrix Adaptation Evolution Strategy) provides adaptive black-box optimization:

```bash
# CMA-ES tuning example
python scripts/controller_autotune.py \
    --config experiments/configs/tuning/tuning_cma_es.yaml
```

**CMA-ES vs. Random/Grid:**
- **Random**: Fast exploration, good for large spaces
- **Grid**: Exhaustive coverage, small spaces only
- **CMA-ES**: Adaptive, best for high-dimensional optimization (>3 params)

## Migration Guide (v0.4+)

Configuration files have been reorganized into subdirectories. Update your scripts and commands using this mapping:

### Old → New Path Mapping

| Old Path | New Path |
|----------|----------|
| `experiments/configs/training_default.yaml` | `experiments/configs/training/training_default.yaml` |
| `experiments/configs/training_fast.yaml` | `experiments/configs/training/training_fast.yaml` |
| `experiments/configs/training_large.yaml` | `experiments/configs/training/training_large.yaml` |
| `experiments/configs/training_imitation.yaml` | `experiments/configs/training/training_imitation.yaml` |
| `experiments/configs/diagnostics_stationary.yaml` | `experiments/configs/training/diagnostics_stationary.yaml` |
| `experiments/configs/diagnostics_linear.yaml` | `experiments/configs/training/diagnostics_linear.yaml` |
| `experiments/configs/eval_stationary_baseline.yaml` | `experiments/configs/evaluation/eval_stationary_baseline.yaml` |
| `experiments/configs/eval_linear_baseline.yaml` | `experiments/configs/evaluation/eval_linear_baseline.yaml` |
| `experiments/configs/eval_circular_baseline.yaml` | `experiments/configs/evaluation/eval_circular_baseline.yaml` |
| `experiments/configs/eval_riccati_lqr_baseline.yaml` | `experiments/configs/evaluation/eval_riccati_lqr_baseline.yaml` |
| `experiments/configs/comparison_default.yaml` | `experiments/configs/evaluation/comparison_default.yaml` |
| `experiments/configs/tuning_pid.yaml` | `experiments/configs/tuning/tuning_pid.yaml` |
| `experiments/configs/tuning_pid_linear.yaml` | `experiments/configs/tuning/tuning_pid_linear.yaml` |
| `experiments/configs/tuning_lqr.yaml` | `experiments/configs/tuning/tuning_lqr.yaml` |
| `experiments/configs/tuning_lqr_linear.yaml` | `experiments/configs/tuning/tuning_lqr_linear.yaml` |
| `experiments/configs/tuning_riccati.yaml` | `experiments/configs/tuning/tuning_riccati.yaml` |
| `experiments/configs/tuning_riccati_linear.yaml` | `experiments/configs/tuning/tuning_riccati_linear.yaml` |
| `experiments/configs/tuning_cma_es.yaml` | `experiments/configs/tuning/tuning_cma_es.yaml` |

### Updating Commands

**Training:**
```bash
# Old
python -m quadcopter_tracking.train --config experiments/configs/training_default.yaml

# New
python -m quadcopter_tracking.train --config experiments/configs/training/training_default.yaml
```

**Evaluation:**
```bash
# Old
python -m quadcopter_tracking.eval --config experiments/configs/eval_stationary_baseline.yaml

# New
python -m quadcopter_tracking.eval --config experiments/configs/evaluation/eval_stationary_baseline.yaml
```

**Tuning:**
```bash
# Old
python scripts/controller_autotune.py --config experiments/configs/tuning_pid_linear.yaml

# New
python scripts/controller_autotune.py --config experiments/configs/tuning/tuning_pid_linear.yaml
```

### Makefile Targets (Unchanged)

The Makefile targets remain the same and have been updated to use the new paths internally:

```bash
make train-deep EPOCHS=100
make eval-baseline-stationary
make tune-pid-linear
```

## ENU Coordinate Frame

All configurations assume **ENU (East-North-Up)** coordinate conventions:

| Axis | Direction | Control Mapping |
|------|-----------|-----------------|
| X | East | +pitch_rate → +X velocity |
| Y | North | +roll_rate → −Y velocity |
| Z | Up | +thrust → +Z acceleration |

> ⚠️ **Warning:** Do NOT mix with NED or other frames. See [docs/architecture.md](../../docs/architecture.md) for details.

## Quick Start Examples

### 1. Establish Baseline Performance

```bash
# Evaluate PID/LQR on stationary target
make eval-baseline-stationary EPISODES=10

# View results
cat reports/baseline_stationary_pid/metrics.json
```

### 2. Train Deep Controller

```bash
# Quick test run
python -m quadcopter_tracking.train \
    --config experiments/configs/training/training_fast.yaml

# Standard training
python -m quadcopter_tracking.train \
    --config experiments/configs/training/training_default.yaml
```

### 3. Tune → Train → Evaluate

```bash
# Step 1: Tune gains for linear motion
make tune-pid-linear TUNING_ITERATIONS=50

# Step 2: Copy best gains to training config
cat reports/tuning/tuning_pid_*_best_config.json
# Then update experiments/configs/training/training_imitation.yaml

# Step 3: Train with tuned supervisor
python -m quadcopter_tracking.train \
    --config experiments/configs/training/training_imitation.yaml

# Step 4: Evaluate
make eval-baseline-linear EPISODES=10
```

### 4. CMA-ES Auto-Tuning

```bash
# Run CMA-ES optimization
python scripts/controller_autotune.py \
    --config experiments/configs/tuning/tuning_cma_es.yaml \
    --max-iterations 100

# Resume interrupted run
python scripts/controller_autotune.py \
    --resume reports/tuning/tuning_*_results.json \
    --max-iterations 200
```

## Creating Custom Configurations

1. Copy an existing config from the appropriate subdirectory
2. Modify parameters as needed
3. Save in the same subdirectory with a descriptive name

**Naming conventions:**
- Training: `training_<description>.yaml`
- Evaluation: `eval_<motion>_<description>.yaml`
- Tuning: `tuning_<controller>_<motion>.yaml`

## See Also

- [README.md](../../README.md) - Main project documentation
- [docs/training.md](../../docs/training.md) - Training pipeline details
- [docs/architecture.md](../../docs/architecture.md) - Controller architecture
- [docs/results.md](../../docs/results.md) - Evaluation and results interpretation
