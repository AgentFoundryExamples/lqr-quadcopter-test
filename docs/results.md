# Evaluation Results Documentation

This document describes the evaluation framework for quadcopter tracking controllers and how to interpret results.

## Overview

The evaluation pipeline assesses controller performance against defined success criteria:

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Episode Duration | ≥ 30 seconds | Minimum time for valid evaluation |
| On-Target Ratio | ≥ 80% | Fraction of time within target radius |
| Target Radius | ≤ 0.5 meters | Distance threshold for "on-target" |

## Running Evaluations

### Basic Evaluation

```bash
# Evaluate deep learning controller
python -m quadcopter_tracking.eval --controller deep --checkpoint checkpoints/best.pt

# Evaluate with custom parameters
python -m quadcopter_tracking.eval \
    --controller deep \
    --checkpoint checkpoints/model.pt \
    --episodes 20 \
    --motion-type circular \
    --target-radius 0.5

# Evaluate classical controllers
python -m quadcopter_tracking.eval --controller lqr --episodes 10
python -m quadcopter_tracking.eval --controller pid --episodes 10
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--controller` | `deep` | Controller type: `deep`, `lqr`, `pid` |
| `--checkpoint` | None | Path to checkpoint file for neural controllers |
| `--episodes` | 10 | Number of evaluation episodes |
| `--seed` | 42 | Base random seed for reproducibility |
| `--motion-type` | `circular` | Target motion pattern |
| `--episode-length` | 30.0 | Episode duration in seconds |
| `--target-radius` | 0.5 | On-target threshold in meters |
| `--output-dir` | `reports` | Output directory for results |
| `--no-plots` | False | Skip generating visualization plots |

### Hyperparameter Sweeps

Run systematic evaluation across multiple configurations:

```bash
python -m quadcopter_tracking.eval --sweep configs/sweep.yaml
```

Sweep configuration example (`configs/sweep.yaml`):

```yaml
configurations:
  - name: "baseline"
    controller_type: deep
    checkpoint_path: checkpoints/baseline.pt
    num_episodes: 10
    seed: 42

  - name: "improved_v1"
    controller_type: deep
    checkpoint_path: checkpoints/improved_v1.pt
    num_episodes: 10
    seed: 42

  - name: "lqr_baseline"
    controller_type: lqr
    num_episodes: 10
    seed: 42
```

## Output Files

### Reports Directory Structure

```
reports/
├── metrics.json               # Detailed metrics in JSON format
├── eval_*_report.txt          # Human-readable summary
├── plots/
│   ├── position_tracking_best.png   # Quadcopter vs target for best episode
│   ├── tracking_error_best.png      # Error over time for best episode
│   ├── position_tracking_worst.png  # Quadcopter vs target for worst episode
│   └── tracking_error_worst.png     # Error over time for worst episode
└── sweeps/
    └── sweep_results.json           # Ranked hyperparameter results
```

### Metrics JSON Format

```json
{
  "total_episodes": 10,
  "successful_episodes": 8,
  "success_rate": 0.8,
  "mean_on_target_ratio": 0.85,
  "std_on_target_ratio": 0.05,
  "mean_tracking_error": 0.32,
  "std_tracking_error": 0.08,
  "mean_control_effort": 12.5,
  "best_episode_idx": 3,
  "worst_episode_idx": 7,
  "meets_criteria": true,
  "episode_metrics": [...]
}
```

## Interpreting Results

### Success Criteria

An evaluation **passes** when:
1. Mean on-target ratio across episodes ≥ 80%
2. Episodes run for ≥ 30 seconds
3. Target radius threshold is met consistently

### Key Metrics

| Metric | Good Range | Description |
|--------|------------|-------------|
| On-Target Ratio | > 80% | Primary success metric |
| Mean Tracking Error | < 0.5m | Average distance to target |
| RMS Tracking Error | < 0.6m | Root mean square error |
| Control Effort | Lower is better | Efficiency metric |
| Overshoot Count | < 5/episode | Stability indicator |

### Visualization Plots

#### Position Tracking Plot

Shows X, Y, Z positions over time:
- **Blue line**: Quadcopter trajectory
- **Red dashed line**: Target trajectory
- **Green shaded region**: Target radius band

Ideal behavior: Blue line stays within green band.

#### Tracking Error Plot

Shows distance to target over time:
- **Blue line**: Tracking error (distance)
- **Green dots**: On-target samples
- **Red dashed line**: Target radius threshold

Ideal behavior: Error stays below red threshold.

## Programmatic Usage

```python
from quadcopter_tracking.eval import Evaluator, load_controller
from quadcopter_tracking.env import EnvConfig
from quadcopter_tracking.utils.metrics import SuccessCriteria

# Load controller
controller = load_controller(
    controller_type="deep",
    checkpoint_path="checkpoints/best.pt",
)

# Configure evaluation
env_config = EnvConfig()
env_config.target.motion_type = "circular"
env_config.simulation.max_episode_time = 30.0

criteria = SuccessCriteria(
    min_on_target_ratio=0.8,
    min_episode_duration=30.0,
    target_radius=0.5,
)

# Run evaluation
evaluator = Evaluator(
    controller=controller,
    env_config=env_config,
    criteria=criteria,
    output_dir="reports",
)

summary = evaluator.evaluate(num_episodes=10)

# Generate outputs
evaluator.save_report(summary)
evaluator.generate_all_plots()

# Check results
if summary.meets_criteria:
    print("Evaluation PASSED!")
else:
    print(f"Evaluation FAILED: {summary.mean_on_target_ratio:.1%} on-target")
```

## Troubleshooting

### Low On-Target Ratio

Possible causes:
- Learning rate too high/low during training
- Insufficient training epochs
- Incorrect loss weights
- Target motion too aggressive

Solutions:
- Retrain with adjusted hyperparameters
- Use curriculum learning
- Increase position weight in loss function

### High Control Effort

Indicates aggressive/jerky control:
- Increase `control_weight` in training
- Use smoother activation functions
- Enable action rate limiting

### Numerical Instability

If evaluation crashes with NaN:
- Check checkpoint integrity
- Reduce episode length for testing
- Verify environment bounds

### Slow Evaluation

For faster iteration:
- Reduce `--episodes` count
- Use `--max-steps` to limit episode length
- Skip plots with `--no-plots`

## Comparing Controllers

To systematically compare controllers:

1. Create sweep configuration with all controllers
2. Use identical seeds and episode counts
3. Compare mean on-target ratios and success rates
4. Review plots for qualitative differences

Example comparison workflow:

```bash
# Create comparison sweep
cat > configs/comparison.yaml << EOF
configurations:
  - name: deep_v1
    controller_type: deep
    checkpoint_path: checkpoints/v1.pt
    num_episodes: 20
    seed: 42
  - name: deep_v2
    controller_type: deep
    checkpoint_path: checkpoints/v2.pt
    num_episodes: 20
    seed: 42
  - name: lqr
    controller_type: lqr
    num_episodes: 20
    seed: 42
EOF

# Run comparison
python -m quadcopter_tracking.eval --sweep configs/comparison.yaml --output-dir reports/comparison

# Review results
cat reports/comparison/sweep_results.json
```

## Extending Evaluation

### Imperfect Information Scenarios

The current evaluation assumes perfect target information. To extend to imperfect information:

1. **Add Observation Noise**: Modify the environment to add Gaussian noise to target position/velocity observations
2. **State Estimation**: Implement a Kalman filter or other state estimator in the controller
3. **Adjust Success Criteria**: Consider relaxing thresholds when noise is present

Example configuration for future noise scenarios:

```yaml
# Future extension (not yet implemented)
observation_noise:
  enabled: true
  position_stddev: 0.1  # meters
  velocity_stddev: 0.05  # m/s
```

### Alternative Controller Evaluation

To evaluate custom controllers:

1. Create a controller class inheriting from `BaseController`
2. Implement `compute_action(observation) -> action`
3. Register with `load_controller()` or pass directly to `Evaluator`

```python
from quadcopter_tracking.controllers import BaseController
from quadcopter_tracking.eval import Evaluator

class MyController(BaseController):
    def compute_action(self, observation):
        # Custom control logic
        return {"thrust": 10.0, "roll_rate": 0.0, ...}

evaluator = Evaluator(controller=MyController())
summary = evaluator.evaluate(num_episodes=10)
```

### Validating Generated Plots

Ensure plot assets are generated correctly:

1. Check `reports/plots/` directory after evaluation
2. Verify PNG files exist: `position_tracking_*.png`, `tracking_error_*.png`
3. For headless systems, ensure `MPLBACKEND=Agg` is set

```bash
# Validate plot generation
ls -la reports/plots/*.png
```

## Release Validation

When preparing a release, run full evaluation to document achieved metrics:

```bash
# Run comprehensive evaluation
python -m quadcopter_tracking.eval \
    --controller lqr \
    --episodes 50 \
    --seed 42 \
    --output-dir reports/release_validation

# Document metrics for release notes
cat reports/release_validation/metrics.json | python -m json.tool
```
