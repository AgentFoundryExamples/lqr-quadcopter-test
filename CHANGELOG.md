# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-11-30

### Added

- **Quadcopter Simulation Environment**: Complete simulation environment with configurable dynamics, target motion patterns (linear, circular, sinusoidal, figure8, stationary), and RK4/Euler integration.
- **LQR Controller**: Linear Quadratic Regulator implementation for baseline tracking performance.
- **PID Controller**: Proportional-Integral-Derivative controller for comparison studies.
- **Deep Learning Controller**: Neural network-based controller with PyTorch training pipeline.
- **Training Pipeline**: Full training system with configurable loss functions (position, velocity, control effort weights), checkpointing, curriculum learning, and NaN recovery.
- **Evaluation Framework**: Comprehensive evaluation with success criteria (≥80% on-target ratio, ≥30s duration, ≤0.5m radius), metrics computation, and visualization plots.
- **Success Criteria**:
  - Episode duration ≥ 30 seconds
  - On-target ratio ≥ 80%
  - Target radius ≤ 0.5 meters
- **Configuration System**: Environment variables via `.env`, YAML config files, and programmatic configuration with sensible defaults.
- **Documentation**: Architecture guide, environment documentation, training guide, and evaluation results documentation.
- **Test Suite**: 107 unit and integration tests covering environment, controllers, training, and evaluation.
- **Development Tools**: Makefile with install, test, lint, and format commands.

### Technical Details

- Python 3.10+ support
- PyTorch 2.0+ for neural network controllers
- NumPy, Matplotlib, PyYAML dependencies
- Reproducible experiments via seed control
- CPU and GPU execution support

### Known Limitations

- Perfect target information (no sensor noise)
- Smooth target motion only (differentiable trajectories)
- Idealized quadcopter dynamics (no disturbances)
- Single target tracking only

[0.1.0]: https://github.com/AgentFoundryExamples/lqr-quadcopter-test/releases/tag/v0.1.0
