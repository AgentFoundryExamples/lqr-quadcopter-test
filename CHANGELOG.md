# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-11-30

### Added

- **Training Diagnostics System**: Comprehensive diagnostic tools for analyzing training behavior, including step-level observation/action logging, gradient statistics tracking, epoch-level metric aggregation, and diagnostic plots (loss curves, tracking error, gradient norms, on-target ratio).
- **Classical Controller Support**: Full integration of PID and LQR controllers for evaluation workflows, providing reliable baseline performance for comparison studies.
- **Imitation Learning Mode**: New training mode enabling supervised learning from classical controllers (PID/LQR supervisors), with configurable imitation weight and tracking weight parameters.
- **Reward-Weighted Training**: Alternative training signal with configurable reward weight for shaped reward incorporation.
- **Reproducible Workflows**: Three documented end-to-end workflows:
  1. Baseline PID/LQR Evaluation
  2. Deep Controller Training
  3. Controller Comparison with automated report generation
- **Makefile Enhancements**: New targets for baseline evaluation (`eval-baseline-stationary`, `eval-baseline-circular`), controller comparison (`compare-controllers`, `generate-comparison-report`), and workflow-specific commands.
- **Configuration Presets**: Additional YAML configurations including `diagnostics_stationary.yaml`, `diagnostics_linear.yaml`, `training_imitation.yaml`, and `comparison_default.yaml`.
- **Controller Comparison Framework**: Automated comparison pipeline generating ranked `comparison_summary.json` with success rate, on-target ratio, and tracking error metrics.
- **Workflow Diagram**: Mermaid diagram documenting the relationship between the three core workflows.

### Changed

- **README Reorganization**: Restructured documentation to highlight v0.2 workflows, controller selection, and diagnostic capabilities prominently.
- **Documentation Updates**: Enhanced `docs/training.md` with diagnostics section, `docs/results.md` with training diagnostics findings and imitation learning results.
- **Training Pipeline**: Extended trainer to support multiple training modes (`tracking`, `imitation`, `reward_weighted`) with mode-specific loss computation.

### Fixed

- **Training Signal Clarity**: Diagnostic findings now documented to help users understand and remediate training regression issues.
- **Evaluation Output Structure**: Standardized output directory structure for comparison workflows.

### Known Issues

- Deep controller training exhibits regression after initial epochs (see `docs/results.md#training-diagnostics-results`). Classical controllers (PID/LQR) are recommended for production use.
- Experiment tracking integrations (WandB, MLflow) remain as placeholders only.
- Transfer learning is not yet supported via the Trainer class.

### Migration from v0.1

- **No breaking changes**: All v0.1 configurations and checkpoints remain compatible.
- **New recommended workflow**: Users should review the three documented workflows in `README.md#reproducible-workflows-v02` for best practices.
- **Diagnostic configs**: Consider using diagnostic configurations (`experiments/configs/diagnostics_*.yaml`) when troubleshooting training issues.

[0.2.0]: https://github.com/AgentFoundryExamples/lqr-quadcopter-test/releases/tag/v0.2.0

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
