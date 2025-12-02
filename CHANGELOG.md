# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2025-12-02

### Added

- **LQI Controller (Linear Quadratic Integral)**: New `lqi` controller type extending Riccati-LQR with integral action for zero steady-state tracking error. Key features:
  - 9-state augmented system (position, velocity, integral of position error)
  - DARE-solved optimal gains for the augmented system decomposed into K_pd (4×6) and K_i (4×3)
  - Configurable `q_int` weights for tuning integral aggressiveness
  - Anti-windup protection with `integral_limit` clamping
  - Zero-error threshold (`integral_zero_threshold`) to prevent drift when on-target
  - `reset_integral_state()` method for explicit integral reset between episodes or mode changes

  See [docs/architecture.md](docs/architecture.md#lqi-mode-with-integral-action) for implementation details.

- **LQI Configuration Parameters**: New YAML configuration options for LQI mode:
  ```yaml
  riccati_lqr:
    use_lqi: true                     # Enable LQI mode
    q_int: [0.01, 0.01, 0.1]          # Integral cost weights [ix, iy, iz]
    integral_limit: 10.0              # Anti-windup clamp
    integral_zero_threshold: 0.01    # Pause integration when error < threshold
  ```

- **LQI Evaluation Support**: The evaluation pipeline now supports `--controller lqi` for direct LQI evaluation:
  ```bash
  python -m quadcopter_tracking.eval --controller lqi --episodes 10
  ```

- **LQI Performance Metrics**: Added expected performance baselines for LQI:
  - Stationary targets: >80% on-target ratio (verified)
  - Linear targets: 75-95% on-target ratio (improved steady-state vs Riccati-LQR)
  
  See [docs/results.md](docs/results.md#lqi-controller-linear-quadratic-integral) for detailed performance analysis.

- **LQI Diagnostics**: New diagnostic methods for LQI controllers:
  - `is_lqi_mode()` - Check if LQI mode is active
  - `get_integral_state()` - Access current integral state [ix, iy, iz]
  - `get_pd_gains()` / `get_integral_gains()` - Access decomposed gain matrices
  - Extended `get_control_components()` output with integral state and K_i gains

- **LQI Testing**: Comprehensive test coverage for LQI mode including:
  - Augmented system matrix construction
  - DARE solver integration with 9-state system
  - Anti-windup clamping behavior
  - Zero-threshold pause logic
  - Integration with existing evaluation workflows

### Changed

- **Documentation Updates**: README, docs/architecture.md, and docs/results.md updated with:
  - LQI mode documentation and configuration examples
  - Comparison table: LQI vs Riccati-LQR vs PID with integral
  - Tuning guidance for `q_int` weights
  - Performance expectations by motion type

- **Controller Selection Guide**: Updated controller selection flowchart and comparison matrix to include LQI controller option.

- **Expected Performance Tables**: Updated baseline performance tables to include LQI controller metrics alongside PID, LQR, and Riccati-LQR.

### Migration Notes

#### From v0.4.x

- **No Breaking Changes**: All v0.4.x configurations and checkpoints remain compatible.

- **LQI Mode**: To use the new LQI controller, add LQI-specific parameters to your `riccati_lqr` configuration block:

  ```yaml
  riccati_lqr:
    dt: 0.01                          # Must match your simulation timestep
    use_lqi: true                     # Enable LQI mode
    q_pos: [0.0001, 0.0001, 16.0]     # Position cost weights [x, y, z]
    q_vel: [0.0036, 0.0036, 4.0]      # Velocity cost weights [vx, vy, vz]
    q_int: [0.001, 0.001, 0.01]       # Integral cost weights [ix, iy, iz]
    r_controls: [1.0, 1.0, 1.0, 1.0]  # Control cost [thrust, roll, pitch, yaw]
    integral_limit: 10.0              # Anti-windup clamp
    integral_zero_threshold: 0.01    # Freeze integral below this error
    fallback_on_failure: true         # Fall back to heuristic LQR on solver failure
  ```

- **When to Use LQI vs Riccati-LQR**:

  | Scenario | Riccati-LQR | LQI |
  |----------|-------------|-----|
  | Stationary target hovering | Good | Excellent (zero steady-state error) |
  | Fast-moving targets | Recommended | Avoid (integral windup risk) |
  | Constant velocity tracking | Good | Better (eliminates bias) |
  | Short episodes (< 5s) | Recommended | Avoid (insufficient integration time) |

- **Tuning q_int (Integral Cost Weights)**:
  - Start with conservative values: `[0.001, 0.001, 0.01]`
  - Z-axis typically needs higher weight for tight altitude tracking
  - Reduce if you observe oscillation or overshoot
  - Increase if steady-state error persists

- **Recommended Actions**:
  1. Review the LQI controller documentation if you need zero steady-state tracking error.
  2. For stationary or slow-moving targets, consider enabling LQI mode.
  3. Use `integral_limit` to prevent windup during large transients.
  4. Call `controller.reset_integral_state()` explicitly when switching target modes.

- **Verification**: Run baseline evaluations to confirm expected performance:

  ```bash
  # Verify version
  pip show quadcopter-tracking | grep Version
  # Should show: Version: 0.5.0

  # Test LQI controller on stationary target
  python -m quadcopter_tracking.eval --controller lqi --episodes 5

  # Run baseline evaluations
  make eval-baseline-stationary EPISODES=10
  ```

- **Release Validation**: When preparing releases or regenerating baseline results, follow these steps to regenerate docs and plots:

  ```bash
  # 1. Regenerate all baseline metrics
  make eval-baseline-stationary EPISODES=50
  make eval-baseline-linear EPISODES=50
  make eval-baseline-circular EPISODES=50

  # 2. Run controller comparison
  make compare-controllers EPISODES=50 MOTION_TYPE=stationary

  # 3. Generate comparison report
  make generate-comparison-report

  # 4. Verify hover thrust tests
  python -m pytest tests/test_env_dynamics.py::TestHoverThrustIntegration -v
  python -m pytest tests/test_env_dynamics.py::TestAxisSignConventions -v
  ```

  Results are saved to `reports/` and should be reviewed before tagging a release.

[0.5.0]: https://github.com/AgentFoundryExamples/lqr-quadcopter-test/releases/tag/v0.5.0

## [0.4.0] - 2025-12-02

### Added

- **Configuration File Reorganization**: Configuration files have been reorganized into subdirectories for better discoverability:
  - `experiments/configs/training/` - Deep controller training configs
  - `experiments/configs/evaluation/` - Controller evaluation configs
  - `experiments/configs/tuning/` - Auto-tuning configs (grid/random/CMA-ES)
  
  See [experiments/configs/README.md](experiments/configs/README.md) for the complete configuration index.

- **Migration Guide**: Comprehensive old → new path mapping documentation in `experiments/configs/README.md`. If you use an old config path, you will receive a clear "file not found" error with guidance on the new path.

- **Controller Capability Matrix**: Added to `experiments/configs/README.md` documenting which controllers support training, feedforward, DARE solver, and imitation teaching.

- **CMA-ES Auto-Tuning Enhancements**: CMA-ES (Covariance Matrix Adaptation Evolution Strategy) support with:
  - Checkpoint and resume capability for interrupted tuning runs
  - Configurable population size and initial sigma
  - Adaptive optimization for high-dimensional parameter spaces

- **ENU Coordinate Frame Documentation**: Standardized ENU (East-North-Up) coordinate frame documentation across all configuration files and documentation. Added axis conventions and control sign mappings to `experiments/configs/README.md`.

### Changed

- **Config Path Structure**: All configuration files now reside in subdirectories. The Makefile targets have been updated internally but remain unchanged from a user perspective.

- **`.env.example` Updates**: Added new environment variables for tuning and CMA-ES configuration.

- **Documentation Updates**: README, docs/results.md, and experiments/configs/README.md updated to reference new config paths and provide quick-start examples.

### Migration Notes

#### From v0.3.x

- **Config Path Changes**: Configuration files have moved to subdirectories. Update your commands:

  | Old Path | New Path |
  |----------|----------|
  | `experiments/configs/training_default.yaml` | `experiments/configs/training/training_default.yaml` |
  | `experiments/configs/eval_stationary_baseline.yaml` | `experiments/configs/evaluation/eval_stationary_baseline.yaml` |
  | `experiments/configs/tuning_pid_linear.yaml` | `experiments/configs/tuning/tuning_pid_linear.yaml` |

  See [experiments/configs/README.md](experiments/configs/README.md) for the complete mapping.

- **Makefile Targets Unchanged**: All `make` commands (e.g., `make train-deep`, `make eval-baseline-stationary`, `make tune-pid-linear`) continue to work without modification.

- **Environment Variables**: Copy [.env.example](.env.example) to `.env` and configure as needed. Key new variables:

  ```bash
  # Tuning output directory (default: reports/tuning)
  TUNING_OUTPUT_DIR=reports/tuning

  # CMA-ES optimization settings
  TUNING_STRATEGY=cma_es
  TUNING_CMA_SIGMA0=0.3
  # TUNING_CMA_POPSIZE=  # Leave unset for auto-calculation

  # Feedforward for moving targets (disabled by default)
  QUADCOPTER_FEEDFORWARD_ENABLED=false
  QUADCOPTER_FF_VELOCITY_GAIN=0.0,0.0,0.0
  QUADCOPTER_FF_ACCELERATION_GAIN=0.0,0.0,0.0
  ```

- **No Breaking Changes to APIs**: All Python APIs, checkpoints, and controller configurations remain compatible.

- **Verification**: Run baseline evaluations to confirm expected performance:

  ```bash
  # Verify version
  pip show quadcopter-tracking | grep Version
  # Should show: Version: 0.4.0

  # Test with new config paths
  python -m quadcopter_tracking.eval \
      --config experiments/configs/evaluation/eval_stationary_baseline.yaml \
      --episodes 5

  # Run baseline evaluations
  make eval-baseline-stationary EPISODES=10
  ```

- **Legacy Config Migration**: If you have custom configurations:
  1. Move them to the appropriate subdirectory (`training/`, `evaluation/`, or `tuning/`)
  2. Update any scripts that reference the old paths
  3. Follow naming conventions: `training_*.yaml`, `eval_*.yaml`, `tuning_*.yaml`

#### From v0.2.x

All v0.3.x migration notes apply. Additionally:
- Review the Riccati-LQR controller if you need mathematically optimal feedback gains
- Consider using auto-tuning for gain optimization

[0.4.0]: https://github.com/AgentFoundryExamples/lqr-quadcopter-test/releases/tag/v0.4.0

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
