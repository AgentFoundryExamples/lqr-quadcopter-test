# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-12-01

### Added

- **Riccati-LQR Controller**: New `riccati_lqr` controller that solves the discrete-time algebraic Riccati equation (DARE) to compute mathematically optimal feedback gains. Suitable as a strong teacher for deep imitation learning and as an optimal baseline for control research. See [docs/architecture.md](docs/architecture.md) for configuration details.

- **Controller Auto-Tuning Framework**: Automated PID and LQR gain optimization via `scripts/controller_autotune.py`. Features grid and random search strategies, deterministic seeding for reproducibility, graceful interruption with partial results saved, and resume capability. Configurable via `TUNING_OUTPUT_DIR` environment variable. See [docs/training.md](docs/training.md#pid-auto-tuning) for usage instructions.

- **Feedforward Support**: Optional velocity and acceleration feedforward for PID and LQR controllers to improve tracking of moving targets. Disabled by default to preserve baseline behavior. Enable with `feedforward_enabled: true` in YAML configuration.

- **Environment Variable Documentation**: Added `CONTROLLER_CONFIG_ROOT` placeholder to `.env.example` for users who want to customize controller configuration paths without modifying code. Use `TUNING_OUTPUT_DIR` for tuning results directory.

- **Release Validation Workflow**: Added guidance in `docs/results.md` for regenerating baseline metrics and plots using Make targets (`make eval-baseline-stationary`, `make eval-baseline-circular`) when preparing releases.

- **Roadmap Update**: Marked Riccati milestone as complete in `ROADMAP.md`, including DARE-based optimal control, auto-tuning framework, and documentation consolidation. Updated next focus areas to emphasize observation noise and state estimation (v0.4+).

### Changed

- **Default Controller Gains**: PID and LQR gains are now experimentally validated for stable tracking across stationary, linear, and circular target scenarios. XY gains are intentionally small (kp=0.01) to prevent actuator saturation—position errors in meters map directly to angular rates in rad/s.

- **Stationary Target Default**: Target motion type now defaults to `stationary` for predictable baseline evaluations, ensuring PID/LQR controllers achieve >80% on-target ratio out of the box.

- **Documentation Consolidation**: README, docs/results.md, and docs/training.md updated to clearly reference tuning workflows, new controller options, and expected baseline performance metrics.

### Migration Notes

#### From v0.2.x

- **No Breaking Changes**: All v0.2.x configurations and checkpoints remain compatible.

- **Riccati-LQR Controller**: To use the new Riccati-LQR controller, add the `riccati_lqr` block to your YAML configuration:

  ```yaml
  controller: riccati_lqr

  riccati_lqr:
    dt: 0.01                          # Must match your simulation timestep
    mass: 1.0                         # Must match your quadcopter mass
    gravity: 9.81                     # Must match your gravity setting
    q_pos: [0.0001, 0.0001, 16.0]     # Position cost weights [x, y, z]
    q_vel: [0.0036, 0.0036, 4.0]      # Velocity cost weights [vx, vy, vz]
    r_controls: [1.0, 1.0, 1.0, 1.0]  # Control cost [thrust, roll, pitch, yaw]
    fallback_on_failure: true         # Recommended: fall back to heuristic LQR on solver failure
  ```

- **Recommended Actions**:
  1. Review the new Riccati-LQR controller if you need mathematically optimal feedback gains.
  2. Consider using auto-tuning (`scripts/controller_autotune.py`) to optimize PID/LQR gains for your specific scenario.
  3. If using custom mass/gravity values, ensure controller initialization passes these values explicitly.

- **Verification**: Run baseline evaluations to confirm expected performance:

  ```bash
  # Verify version
  pip show quadcopter-tracking | grep Version

  # Run baseline evaluations
  make eval-baseline-stationary EPISODES=10
  make eval-baseline-circular EPISODES=10

  # Test Riccati-LQR controller
  python -m quadcopter_tracking.eval --controller riccati_lqr --episodes 5
  ```

- **Related Issues**: This release consolidates work from the following improvements:
  - PID/LQR default gain tuning and hover thrust feedforward
  - Riccati-LQR controller implementation
  - Auto-tuning framework for controller gain optimization
  - Documentation and environment variable standardization

[0.3.0]: https://github.com/AgentFoundryExamples/lqr-quadcopter-test/releases/tag/v0.3.0

## [0.2.1] - 2025-12-01

### Fixed

- **Hover Thrust Feedforward**: PID and LQR controllers now correctly include hover thrust (`mass × gravity`) as a baseline feedforward term. At zero tracking error (quadcopter at target), controllers output the exact hover thrust required to maintain altitude (~9.81N for default 1kg mass).

- **Sign Convention Verification**: Added comprehensive regression tests verifying controller output signs match environment dynamics:
  - Positive X error → positive pitch_rate (produces +X velocity)
  - Positive Y error → negative roll_rate (produces -Y velocity, per environment convention)

- **Stationary Target Default**: Target motion type now defaults to `stationary` for predictable baseline evaluations. This ensures PID/LQR controllers achieve >80% on-target ratio out of the box.

### Added

- **Hover Thrust Integration Tests**: New `TestHoverThrustIntegration` test class with parametrized tests covering:
  - PID/LQR hover thrust accuracy (within 0.5N tolerance)
  - Mass/gravity scaling verification
  - Zero-thrust regression guards
  - Multi-step stability tests

- **Axis Sign Convention Tests**: New `TestAxisSignConventions` test class verifying:
  - Environment dynamics (pitch→X, roll→Y)
  - Controller output sign conventions
  - Convergence to stationary targets
  - Initial acceleration direction toward target

### Changed

- **Documentation Updates**: README and docs/results.md now include:
  - Hover test verification commands
  - Expected success criteria (>80% on-target for stationary)
  - Guidance for users with custom mass/gravity configurations

### Migration from v0.2.0

- **Action Required for Custom Configs**: If you have custom configurations with non-default mass or gravity values, ensure your controller initialization passes these values explicitly:
  ```python
  controller = PIDController(config={"mass": your_mass, "gravity": your_gravity})
  controller = LQRController(config={"mass": your_mass, "gravity": your_gravity})
  ```
  Controllers now use these values to compute `hover_thrust = mass × gravity`.

- **DeepTrackingPolicy Unchanged**: The neural network controller (`DeepTrackingPolicy`) was not modified in this release. It continues to learn thrust values from training data rather than using explicit hover feedforward.

- **Verification**: Run hover tests to confirm correct behavior:
  ```bash
  python -m pytest tests/test_env_dynamics.py::TestHoverThrustIntegration -v
  ```

[0.2.1]: https://github.com/AgentFoundryExamples/lqr-quadcopter-test/releases/tag/v0.2.1

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
