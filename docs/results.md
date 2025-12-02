# Evaluation Results Documentation (v0.4.0)

This document describes the evaluation framework for quadcopter tracking controllers and how to interpret results.

> **v0.4.0 Update:** This version introduces configuration file reorganization into training/evaluation/tuning subdirectories. See [experiments/configs/README.md](../experiments/configs/README.md) for the migration guide. Previous versions' documentation for Riccati-LQR, auto-tuning, and feedforward remains applicable.

## Overview

The evaluation pipeline assesses controller performance against defined success criteria:

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Episode Duration | ≥ 30 seconds | Minimum time for valid evaluation |
| On-Target Ratio | ≥ 80% | Fraction of time within target radius |
| Target Radius | ≤ 0.5 meters | Distance threshold for "on-target" |

### Expected Performance by Motion Type

| Motion Type | Controller | Expected On-Target Ratio |
|-------------|------------|-------------------------|
| Stationary | PID/LQR | >80% (verified) |
| Linear | PID/LQR | 70-90% |
| Circular | PID/LQR | 70-90% |
| Sinusoidal | PID/LQR | 60-80% |

**Note:** Stationary target is now the default motion type. PID and LQR controllers are expected to achieve the 80% success threshold reliably on stationary targets.

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

### Tuning Terminates Early or Returns Empty Results

**Symptoms:**
- Tuning script exits before max iterations
- `*_results.json` file is empty or missing
- Best config file not generated

**Causes and Solutions:**

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Empty results file | No valid configurations evaluated | Check error messages, widen search range |
| Partial results | Interrupted without Ctrl+C cleanup | Resume with `--resume` flag |
| All configs fail | Environment misconfiguration | Verify environment works: `make eval-pid EPISODES=1` |
| Solver failures only | Invalid Q/R matrices | Narrow search ranges, validate matrices |

**Recovery steps:**

```bash
# Check if partial results exist
ls -la reports/tuning/tuning_*_results.json

# Resume from partial results
python scripts/controller_autotune.py \
    --resume reports/tuning/tuning_*_results.json \
    --max-iterations 100

# If no results, verify environment first
python -m quadcopter_tracking.eval --controller pid --episodes 1
```

### CPU-Only Feasibility and Expected Slowdown

The following table shows expected runtimes on CPU-only machines:

| Task | GPU Time | CPU Time | Slowdown Factor |
|------|----------|----------|-----------------|
| PID evaluation (10 episodes) | ~3s | ~5s | 1.7x |
| Riccati-LQR evaluation (10 episodes) | ~5s | ~10s | 2x |
| PID tuning (50 iterations) | ~5 min | ~15 min | 3x |
| Deep training (100 epochs) | ~2 min | ~15 min | 7.5x |
| Deep training (500 epochs) | ~10 min | ~75 min | 7.5x |

**Recommendations for CPU-only machines:**
- Use `training_fast.yaml` configuration for initial experiments
- Reduce `episodes_per_epoch` to 3-5
- Use smaller networks: `hidden_sizes: [32, 32]`
- Consider overnight training for large experiments

```bash
# Force CPU and optimize for limited resources
export CUDA_VISIBLE_DEVICES=""
python -m quadcopter_tracking.train \
    --config experiments/configs/training/training_fast.yaml \
    --hidden-sizes 32 32 \
    --episodes-per-epoch 3
```

### Headless Server Operation

For servers without display (CI/CD, cloud instances, SSH sessions):

**Required setup:**

```bash
# Set matplotlib backend before any imports
export MPLBACKEND=Agg

# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx

# Verify headless operation
python -c "import matplotlib; matplotlib.use('Agg'); print('Headless mode OK')"
```

**Skip plot generation for faster runs:**

```bash
python -m quadcopter_tracking.eval --controller pid --episodes 10 --no-plots
```

**Common headless issues:**

| Error | Solution |
|-------|----------|
| `_tkinter.TclError: couldn't connect to display` | Set `MPLBACKEND=Agg` |
| `libGL error: No matching fbConfigs` | Install `libgl1-mesa-glx` |
| `cannot open display` | Ensure no GUI code runs, use `--no-plots` |

### Switching Coordinate Frames for External Simulators

When integrating with other simulators (ROS, Gazebo, AirSim):

**ENU (this project) vs NED conversion:**

```python
def enu_to_ned(position, velocity):
    """Convert ENU coordinates to NED."""
    # ENU: X=East, Y=North, Z=Up
    # NED: X=North, Y=East, Z=Down
    pos_ned = np.array([position[1], position[0], -position[2]])
    vel_ned = np.array([velocity[1], velocity[0], -velocity[2]])
    return pos_ned, vel_ned

def ned_to_enu(position, velocity):
    """Convert NED coordinates to ENU."""
    pos_enu = np.array([position[1], position[0], -position[2]])
    vel_enu = np.array([velocity[1], velocity[0], -velocity[2]])
    return pos_enu, vel_enu
```

**Control output conversion:**

```python
def convert_action_enu_to_ned(action):
    """Convert ENU control to NED convention."""
    return {
        "thrust": action["thrust"],
        "roll_rate": -action["roll_rate"],   # Flip roll sign
        "pitch_rate": -action["pitch_rate"], # Flip pitch sign
        "yaw_rate": -action["yaw_rate"],     # Flip yaw sign
    }
```

**Validation steps:**
1. Apply small test commands in each axis
2. Verify direction matches expected behavior
3. Test with stationary target before motion

### Mixing Stale Tuning Outputs with New Environment Parameters

> ⚠️ **Critical Warning**: Using tuning results from different environment configurations can cause instability or poor performance.

**Safe practices:**

1. **Always re-tune after changing:**
   - Mass or gravity values
   - Simulation timestep (dt)
   - Max thrust or rate limits
   - Target radius requirement

2. **Version your tuning outputs:**
   ```bash
   # Before changing environment
   mv reports/tuning reports/tuning_mass1.0_dt0.01
   
   # After changing environment
   mkdir reports/tuning
   # Re-run tuning with new parameters
   ```

3. **Document environment in config:**
   ```yaml
   # tuning_config.yaml
   # Environment: mass=1.5, gravity=9.81, dt=0.01
   controller: pid
   # ... tuning parameters
   ```

4. **Validate before deploying:**
   ```bash
   # After loading old tuning results
   python -m quadcopter_tracking.eval \
       --controller pid \
       --episodes 5 \
       --motion-type stationary
   # Verify >80% on-target before using on moving targets
   ```

## Comparing Controllers

To systematically compare controllers:

1. Create sweep configuration with all controllers
2. Use identical seeds and episode counts
3. Compare mean on-target ratios and success rates
4. Review plots for qualitative differences

### Quick Comparison Workflow

Use the Makefile targets for easy comparison:

```bash
# Step 1: Run evaluation for each controller
make compare-controllers EPISODES=10 MOTION_TYPE=circular

# Step 2: Generate comparison report
make generate-comparison-report

# Step 3: View results
cat reports/comparison/comparison_summary.json
```

### Manual Comparison Workflow

For more control over the comparison:

```bash
# Evaluate each controller with identical parameters
python -m quadcopter_tracking.eval \
    --controller pid \
    --episodes 20 \
    --seed 42 \
    --motion-type circular \
    --output-dir reports/comparison/pid

python -m quadcopter_tracking.eval \
    --controller lqr \
    --episodes 20 \
    --seed 42 \
    --motion-type circular \
    --output-dir reports/comparison/lqr

# Deep controller (requires trained checkpoint)
python -m quadcopter_tracking.eval \
    --controller deep \
    --checkpoint checkpoints/train_xxx_best.pt \
    --episodes 20 \
    --seed 42 \
    --motion-type circular \
    --output-dir reports/comparison/deep
```

### Using Comparison Config File

```bash
# Create comparison sweep
cat > experiments/configs/my_comparison.yaml << EOF
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
python -m quadcopter_tracking.eval --sweep experiments/configs/my_comparison.yaml --output-dir reports/comparison

# Review results
cat reports/comparison/sweep_results.json
```

### Comparison Report Format

The comparison report (`comparison_summary.json`) contains:

```json
{
  "rankings": [
    {
      "controller": "lqr",
      "success_rate": 0.9,
      "mean_on_target_ratio": 0.87,
      "mean_tracking_error": 0.32
    },
    {
      "controller": "pid",
      "success_rate": 0.85,
      "mean_on_target_ratio": 0.82,
      "mean_tracking_error": 0.41
    }
  ]
}
```

Controllers are ranked by:
1. **Success rate** (primary) - higher is better
2. **Mean on-target ratio** (secondary) - higher is better
3. **Mean tracking error** (tertiary) - lower is better

## Interpreting Plots

### Position Tracking Plots

The position tracking plot (`position_tracking_*.png`) shows the quadcopter and target trajectories over time.

```
What to Look For:
┌─────────────────────────────────────────┐
│ X Position vs Time                      │
│   - Blue line: Quadcopter trajectory    │
│   - Red dashed: Target trajectory       │
│   - Green band: Target radius (±0.5m)   │
│                                         │
│ Good: Blue stays within green band      │
│ Bad: Blue line diverges from red        │
└─────────────────────────────────────────┘
```

**Good Performance Indicators:**
- Blue line closely follows red dashed line
- Minimal overshoot when target changes direction
- Quick convergence to target after initial error

**Poor Performance Indicators:**
- Blue line consistently outside green band
- Large oscillations around target
- Growing divergence over time

### Tracking Error Plots

The tracking error plot (`tracking_error_*.png`) shows the distance to target over time.

```
What to Look For:
┌─────────────────────────────────────────┐
│ Tracking Error (meters) vs Time         │
│   - Blue line: Euclidean distance       │
│   - Red dashed: Target radius threshold │
│   - Green dots: On-target samples       │
│                                         │
│ Good: Error mostly below threshold      │
│ Bad: Error consistently above threshold │
└─────────────────────────────────────────┘
```

**Metric Interpretation:**

| Error Pattern | Interpretation | Suggested Action |
|--------------|----------------|------------------|
| Low, stable | Excellent tracking | None needed |
| High initial, then low | Good convergence | Acceptable |
| Oscillating | Underdamped response | Increase damping gains |
| Steadily increasing | Divergence | Check controller config |
| Periodic spikes | Motion pattern difficulty | Tune for motion type |

### Comparison Chart

When comparing controllers, plot metrics side-by-side:

```
Controller Comparison Example:

                 PID     LQR     Deep
Success Rate     85%     90%     70%
On-Target        82%     87%     65%
Mean Error      0.41m   0.32m   0.52m
Control Effort   12.3    10.8    15.2

Interpretation:
- LQR has best overall performance
- PID is close second, more robust
- Deep controller needs more training
```

## Extending Evaluation

### Imperfect Information Scenarios

The current evaluation assumes perfect target information. Future releases will add support for imperfect information scenarios including observation noise and state estimation.

See [ROADMAP.md](../ROADMAP.md) for detailed design proposals on:
- Observation noise configuration
- State estimation (Kalman filter) integration
- Adjusted success criteria for noisy environments

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

## Tune → Train → Evaluate Workflow

This section describes the complete pipeline for optimizing controller performance on moving targets.

### Overview

The workflow chains three stages:
1. **Tune**: Find optimal controller gains for a specific motion pattern
2. **Train**: Use tuned controller as supervisor for imitation learning
3. **Evaluate**: Assess the trained policy on the target motion

### Step 1: Tune Controller Gains

Use the auto-tuning framework to find optimal gains for your target motion:

```bash
# Tune PID for linear motion
make tune-pid-linear TUNING_ITERATIONS=50

# Tune LQR for linear motion  
make tune-lqr-linear TUNING_ITERATIONS=30

# Tune Riccati-LQR for linear motion (optimal DARE-solved gains)
make tune-riccati-linear TUNING_ITERATIONS=30
```

**Configuration files:**
- `experiments/configs/tuning/tuning_pid_linear.yaml` - PID tuning for linear motion
- `experiments/configs/tuning/tuning_lqr_linear.yaml` - LQR tuning for linear motion
- `experiments/configs/tuning/tuning_riccati_linear.yaml` - Riccati-LQR tuning for linear motion

For other motion patterns (circular, sinusoidal, figure8):
1. Copy the appropriate `tuning_*_linear.yaml` config
2. Change `target_motion_type` to your desired pattern
3. Run with `--config path/to/your/config.yaml`

### Step 2: Apply Tuned Gains to Training

After tuning completes, the best configuration is saved to `reports/tuning/`:

```bash
# View the best configuration
cat reports/tuning/tuning_pid_*_best_config.json
```

Copy the best gains to your training configuration:

1. Open `experiments/configs/training/training_imitation.yaml`
2. Update the `pid`, `lqr`, or `riccati_lqr` section with tuned values
3. Set `supervisor_controller` to match your tuned controller type
4. Set `target_motion_type: linear` (or your target motion)

Example for tuned PID gains:
```yaml
# In training_imitation.yaml
supervisor_controller: pid

pid:
  kp_pos: [0.015, 0.015, 4.5]  # Copy from best_config.json
  ki_pos: [0.0, 0.0, 0.0]
  kd_pos: [0.08, 0.08, 2.2]
  integral_limit: 0.0

target_motion_type: linear  # Match tuning motion type
```

### Step 3: Train with Imitation Learning

Train a deep controller using the tuned classical controller as supervisor:

```bash
python -m quadcopter_tracking.train \
    --config experiments/configs/training/training_imitation.yaml
```

The deep controller learns to mimic the tuned supervisor's behavior.

### Step 4: Evaluate Trained Policy

Evaluate the trained policy on the same motion type used for tuning:

```bash
# Using Makefile target
make eval-baseline-linear EPISODES=10

# Or with specific checkpoint
python -m quadcopter_tracking.eval \
    --controller deep \
    --checkpoint checkpoints/imitation/train_*_best.pt \
    --config experiments/configs/evaluation/eval_linear_baseline.yaml
```

### Complete Example

```bash
# Full pipeline for linear motion
make tune-pid-linear TUNING_ITERATIONS=50

# Review and copy best gains to training_imitation.yaml
cat reports/tuning/tuning_pid_*_best_config.json

# Train (after updating training_imitation.yaml)
python -m quadcopter_tracking.train \
    --config experiments/configs/training/training_imitation.yaml

# Evaluate
make eval-baseline-linear EPISODES=20
```

### Resource-Limited Machines

For machines with limited compute:

```bash
# Reduce tuning iterations
make tune-pid-linear TUNING_ITERATIONS=20

# Or edit config files to reduce:
# - evaluation_episodes: 3     (fewer episodes per config)
# - evaluation_horizon: 1500   (shorter episodes)
# - episode_length: 15.0       (shorter duration)
```

### Tips for Best Results

1. **Start with stationary targets** to establish baseline gains, then tune for motion
2. **Use Riccati-LQR as supervisor** for optimal imitation learning signal
3. **Match motion types** between tuning, training, and evaluation
4. **Monitor imitation loss** during training - it should decrease steadily
5. **Compare against baselines** using `make compare-controllers`

## Mixed-Mode Training Strategies

Mixed-mode training combines multiple loss components or supervisors to achieve better generalization. This section covers advanced training configurations.

### Training Modes Overview

| Mode | Primary Signal | Secondary Signal | Use Case |
|------|----------------|------------------|----------|
| `tracking` | Position/velocity error | Control penalty | Pure RL-style learning |
| `imitation` | Supervisor action MSE | Tracking loss | Learning from classical controllers |
| `reward_weighted` | Supervisor-relative control | Tracking loss | Balanced guidance |

### Mixed Imitation + Tracking

Combine imitation learning with tracking loss for policies that can exceed supervisor performance:

```yaml
# experiments/configs/training/training_mixed.yaml
training_mode: imitation
supervisor_controller: riccati_lqr

# Balance imitation vs tracking
imitation_weight: 1.5    # How much to match supervisor
tracking_weight: 0.8     # How much to minimize error directly

# Supervisor configuration (tuned gains)
riccati_lqr:
  dt: 0.01
  q_pos: [0.00012, 0.00012, 16.5]
  q_vel: [0.0035, 0.0035, 4.2]
  r_controls: [1.0, 1.0, 1.0, 1.0]
```

### Progressive Supervisor Transition

Start with strong imitation, then reduce weight to allow policy improvement:

```yaml
# Stage 1: Strong imitation (epochs 0-25)
training_mode: imitation
imitation_weight: 2.0
tracking_weight: 0.3

# Stage 2: Balanced (epochs 25-50) - modify config and resume
imitation_weight: 1.0
tracking_weight: 1.0

# Stage 3: Tracking-focused (epochs 50+) - modify config and resume
imitation_weight: 0.3
tracking_weight: 2.0
```

```bash
# Run staged training
python -m quadcopter_tracking.train --config stage1.yaml --epochs 25
python -m quadcopter_tracking.train --config stage2.yaml --resume checkpoints/*_epoch0025.pt --epochs 50
python -m quadcopter_tracking.train --config stage3.yaml --resume checkpoints/*_epoch0050.pt --epochs 100
```

### Multi-Supervisor Ensemble

Train with different supervisors for different motion types:

```bash
# Train with PID on stationary
python -m quadcopter_tracking.train \
    --config experiments/configs/training/training_imitation_stationary.yaml \
    --epochs 30

# Continue with Riccati-LQR on linear motion
python -m quadcopter_tracking.train \
    --config experiments/configs/training/training_imitation_linear.yaml \
    --resume checkpoints/stationary/*_best.pt \
    --epochs 60
```

### Evaluation Checkpoint Strategy

Track training progress with periodic evaluation checkpoints:

```mermaid
flowchart LR
    A[Epoch 0] --> B[Eval: Baseline]
    B --> C[Epochs 1-25]
    C --> D[Checkpoint 25]
    D --> E[Eval: Compare to Baseline]
    E --> F{Improved?}
    F -->|Yes| G[Continue Training]
    F -->|No| H[Adjust Config]
    G --> I[Epochs 26-50]
    I --> J[Checkpoint 50]
    J --> K[Eval: Compare to Best]
```

**Checkpoint evaluation workflow:**

```bash
# After each checkpoint, evaluate and compare
for checkpoint in checkpoints/train_*_epoch*.pt; do
    epoch=$(echo $checkpoint | grep -oP 'epoch\K\d+')
    python -m quadcopter_tracking.eval \
        --controller deep \
        --checkpoint $checkpoint \
        --episodes 5 \
        --output-dir reports/checkpoints/epoch${epoch}
done

# Compare checkpoints
python scripts/generate_comparison_report.py \
    --report-dir reports/checkpoints \
    --output reports/checkpoint_comparison.json
```

### Early Stopping Criteria

Stop training when performance plateaus:

| Metric | Early Stop Threshold | Action |
|--------|---------------------|--------|
| Loss increasing for 5+ epochs | Stop, load best | Reduce learning rate, resume |
| On-target ratio > 80% | Consider stopping | Evaluate on harder motion |
| Tracking error < 0.3m | Excellent | Save and evaluate thoroughly |

### Troubleshooting Mixed Training

**Imitation loss decreases but tracking error increases:**
- Supervisor may be suboptimal for the scenario
- Reduce `imitation_weight`, increase `tracking_weight`
- Try different supervisor (Riccati-LQR vs PID)

**Training oscillates between modes:**
- Loss weights may be fighting each other
- Use more gradual weight transitions
- Increase batch size for stability

**Supervisor creates bad habits:**
- Supervisor may have limitations (e.g., no feedforward)
- Use Riccati-LQR with feedforward as supervisor
- Reduce imitation weight earlier in training

## Release Validation

When preparing a release, run a full evaluation to document achieved metrics and regenerate baseline plots:

### Step 1: Regenerate Baseline Metrics

Run the baseline evaluation Make targets to generate updated metrics and plots:

```bash
# Regenerate stationary target baseline metrics
make eval-baseline-stationary EPISODES=50

# Regenerate circular target baseline metrics
make eval-baseline-circular EPISODES=50

# Regenerate linear target baseline metrics
make eval-baseline-linear EPISODES=50

# Results are saved to:
# - reports/baseline_stationary_pid/
# - reports/baseline_stationary_lqr/
# - reports/baseline_circular_pid/
# - reports/baseline_circular_lqr/
# - reports/baseline_linear_pid/
# - reports/baseline_linear_lqr/
# - reports/baseline_linear_riccati/
```

### Step 2: Run Controller Comparison

Generate a side-by-side comparison report:

```bash
# Run comparison for all controllers
make compare-controllers EPISODES=50 MOTION_TYPE=stationary

# Generate summary report
make generate-comparison-report

# View ranked results
cat reports/comparison/comparison_summary.json | python -m json.tool
```

### Step 3: Document Release Metrics

For each controller, record the following metrics in release notes:

```bash
# Document LQR metrics
python -m quadcopter_tracking.eval \
    --controller lqr \
    --episodes 50 \
    --seed 42 \
    --output-dir reports/release_validation

# View metrics summary
cat reports/release_validation/metrics.json | python -m json.tool
```

### Step 4: Verify Hover Tests

Ensure all hover thrust integration tests pass:

```bash
python -m pytest tests/test_env_dynamics.py::TestHoverThrustIntegration -v
python -m pytest tests/test_env_dynamics.py::TestAxisSignConventions -v
```

### Expected Baseline Performance (v0.4.0)

| Motion Type | Controller | Expected On-Target Ratio |
|-------------|------------|-------------------------|
| Stationary | PID/LQR | >80% (verified) |
| Stationary | Riccati-LQR | >80% (verified) |
| Linear | PID/LQR | 70-90% |
| Circular | PID/LQR | 70-90% |

## Training Diagnostics Results

This section documents the findings from diagnostic experiments analyzing the deep controller training regression.

### Experiment Configuration

Two diagnostic experiments were run to analyze training behavior:

1. **Stationary Target**: Simplest tracking scenario with a fixed target position
2. **Linear Target**: Moving target with constant velocity

Both experiments used:
- 20 epochs, 5 episodes per epoch
- Network: 2 hidden layers (64, 64 neurons)
- Learning rate: 0.001 (Adam optimizer)
- Batch size: 16

### Stationary Target Results

| Epoch | Loss | On-Target Ratio | Tracking Error (m) |
|-------|------|-----------------|-------------------|
| 0 | 931.89 | 9.2% | 70.26 |
| 2 | 575.00 | 14.2% | 60.34 |
| 5 | 1242.13 | 7.3% | 89.64 |
| 10 | 1613.99 | 4.7% | 96.40 |
| 15 | 2395.14 | 6.5% | 98.37 |
| 19 | 1753.26 | 3.6% | 99.95 |

**Key Observations:**
- Loss initially decreases (931 → 575) then increases dramatically
- Best performance at epoch 2 (lowest loss)
- Tracking error increases from ~60m to ~100m over training
- On-target ratio remains low (3-14%), never approaching 80% target

### Linear Target Results

| Epoch | Loss | On-Target Ratio | Tracking Error (m) |
|-------|------|-----------------|-------------------|
| 0 | 499.30 | 4.1% | 52.60 |
| 4 | 1208.60 | 8.5% | 82.99 |
| 10 | 1346.26 | 4.7% | 96.99 |
| 15 | 2310.18 | 2.5% | 101.74 |
| 19 | 2417.92 | 3.1% | 100.86 |

**Key Observations:**
- Best loss at epoch 0 (initial weights)
- Loss increases 5x over training (499 → 2418)
- Tracking error roughly doubles (53m → 101m)
- On-target ratio stays below 10%

### Identified Failure Modes

Based on diagnostic analysis, the following issues contribute to training regression:

1. **Loss Function Misalignment**
   - The tracking loss penalizes position/velocity error but doesn't incentivize getting closer to target
   - Control penalty may dominate when large corrections are needed
   - Gradient signal doesn't clearly indicate desired improvement direction

2. **Action Scaling Issues**
   - Action magnitudes (~40-50 units) suggest thrust/rate commands may be saturated
   - Bounded sigmoid outputs may limit controller expressiveness
   - Initial random weights may produce reasonable baseline performance that training disrupts

3. **Reward Signal Problems**
   - Reward weight defaults to 0.0, so reward shaping is unused
   - Negative distance reward alone may provide weak learning signal
   - No bonus for being on-target during training updates

4. **Data Flow Concerns**
   - Feature extraction may lose relevant state information
   - Batch sampling may not capture temporal dependencies
   - Episode-level updates may average out important corrections

### Diagnostic Metrics Summary

The diagnostics system captures the following per-epoch metrics:

| Metric | Purpose | Observed Range |
|--------|---------|---------------|
| mean_gradient_norm | Training signal strength | 0.99-1.0 (clipped) |
| action_magnitude_mean | Control output magnitude | 40-50 units |
| observation_range | Input feature spread | [-35, 25] |
| num_nan_gradients | Numerical stability | 0 (stable) |

Gradient norms being clipped to ~1.0 suggests the raw gradients are larger, which may indicate:
- Loss landscape is steep
- Learning rate may be too high for the loss scale
- Network architecture may need adjustment

### Recommended Next Steps

Based on diagnostic findings, the following remediation areas are recommended:

1. **Loss Function Tuning**
   - Increase reward_weight to > 0 to incorporate shaped rewards
   - Adjust position_weight relative to control_weight
   - Consider Huber loss for more robust gradients

2. **Learning Rate Schedule**
   - Start with lower learning rate (1e-4 instead of 1e-3)
   - Implement learning rate decay
   - Consider warmup period

3. **Curriculum Learning**
   - Enable `use_curriculum: true`
   - Start with smaller episode lengths
   - Gradually increase target motion complexity

4. **Architecture Modifications**
   - Experiment with deeper/wider networks
   - Consider residual connections
   - Try different activation functions (tanh, leaky_relu)

5. **Training Procedure**
   - Increase batch size for more stable gradients
   - Reduce episodes_per_epoch for more frequent updates
   - Add validation episodes for early stopping

## Imitation Learning Mode Results

As part of implementing learning signals (Issue: Effective Learning Signals), imitation learning mode was added to provide supervisory signals from classical controllers.

### Imitation Mode Configuration

```yaml
training_mode: imitation
supervisor_controller: pid
imitation_weight: 2.0
tracking_weight: 0.5
```

### Stationary Target Experiment

| Epoch | Loss | Tracking Error (m) | On-Target Ratio |
|-------|------|-------------------|-----------------|
| 0 | 731.62 | 49.68 | 10.2% |
| 1 | 624.91 | 42.19 | 7.1% |
| 2 | 650.17 | 56.64 | 17.4% |
| 5 | 394.58 | 49.40 | 10.6% |
| 8 | 1588.48 | 101.24 | 7.5% |

**Observations:**
- Initial epochs (0-5) show loss decreasing from 731 to 394
- Best tracking error achieved: 42.19m at epoch 1
- Training becomes unstable after epoch 5, similar to baseline
- Imitation provides clearer gradient signal in early epochs

### Key Findings

1. **Imitation provides early improvement**: Loss decreases 46% in first 5 epochs (731 → 394)
2. **Training instability persists**: Without additional regularization, training still diverges
3. **Supervisor quality matters**: PID controller provides smooth but bounded performance ceiling

### Training Mode Comparison

| Mode | Initial Loss | Best Loss | Epochs to Best |
|------|-------------|-----------|----------------|
| Tracking (baseline) | ~930 | ~575 | 2 |
| Imitation (PID) | ~730 | ~395 | 5 |

### Recommendations for Imitation Mode

1. **Use early stopping**: Monitor loss and stop when improvement plateaus
2. **Lower learning rate**: Use 0.0001-0.0005 for more stable learning
3. **Adjust imitation weight**: Higher values (2-5) for strict imitation, lower (0.5-1) for flexibility
4. **Combine with curriculum**: Start with stationary targets, progress to moving targets

### Configuration Files

The following preset configurations are available in `experiments/configs/training/`:

| Config | Mode | Use Case |
|--------|------|----------|
| `training_default.yaml` | tracking | Baseline training |
| `training_fast.yaml` | tracking | Quick testing |
| `training_large.yaml` | tracking | Complex tasks |
| `training_imitation.yaml` | imitation | Supervised learning |
| `diagnostics_stationary.yaml` | tracking | Diagnostic with stationary target |
| `diagnostics_linear.yaml` | tracking | Diagnostic with linear target |

See [docs/training.md](training.md) for detailed configuration options and [experiments/configs/README.md](../experiments/configs/README.md) for the complete config index.

## Hover Thrust Verification Tests

Integration tests verify that PID and LQR controllers output correct hover thrust when the quadcopter is at the target position with zero velocity.

> **v0.2.1 Critical Fix:** Controllers now correctly include hover thrust feedforward (`hover_thrust = mass × gravity`). At zero tracking error, controllers output ~9.81N for default configuration, enabling stable hover and >80% on-target ratio for stationary targets.

### Running Hover Tests

```bash
# Run all hover thrust integration tests
python -m pytest tests/test_env_dynamics.py::TestHoverThrustIntegration -v

# Run specific controller tests
python -m pytest tests/test_env_dynamics.py::TestHoverThrustIntegration::test_pid_hover_thrust_integration -v
python -m pytest tests/test_env_dynamics.py::TestHoverThrustIntegration::test_lqr_hover_thrust_integration -v

# Run parametrized tests for various mass/gravity configurations
python -m pytest tests/test_env_dynamics.py::TestHoverThrustIntegration -v -k "parametrized"

# Run axis sign convention tests
python -m pytest tests/test_env_dynamics.py::TestAxisSignConventions -v
```

### Test Coverage

| Test Category | Description | Tolerance |
|--------------|-------------|-----------|
| Thrust Accuracy | Hover thrust within expected baseline | ±0.5N |
| No Torques | Roll/pitch/yaw rates at equilibrium | <0.01 rad/s |
| Mass/Gravity Scaling | Various mass/gravity configurations | ±0.5N |
| Regression Guards | Detect zero-thrust failures | >1.0N |
| Multi-Step Stability | Thrust stability over time | ±0.5N |
| Sign Conventions | Controller outputs match environment dynamics | Directional |

### Expected Hover Thrust

The expected hover thrust is calculated as:

```
hover_thrust = mass × gravity
```

Default configuration (mass=1.0kg, gravity=9.81m/s²) expects ~9.81N thrust at hover equilibrium.

### Custom Mass/Gravity Configuration

If using custom mass or gravity values, pass them explicitly to controller constructors:

```python
# For custom quadcopter with 1.5kg mass
controller = PIDController(config={"mass": 1.5, "gravity": 9.81})
# Expected hover thrust: 1.5 × 9.81 = 14.715N

# For different planet/environment gravity
controller = LQRController(config={"mass": 1.0, "gravity": 3.72})  # Mars gravity
# Expected hover thrust: 1.0 × 3.72 = 3.72N
```

⚠️ **Important**: Controllers compute `hover_thrust = mass × gravity` internally. If your environment uses different physics parameters, ensure controller and environment configs match.

### Test Helper Functions

The test suite provides shared helper functions for constructing hover test scenarios:

- `create_hover_observation()`: Creates zero-error observation dictionary
- `create_hover_env_config()`: Creates environment config for stationary hover testing

These helpers ensure consistent test setup across different hover verification tests.

### Test Characteristics

- **Deterministic**: Seeded random state ensures reproducible results
- **Fast**: Each test completes in <1s
- **CI-Ready**: Suitable for continuous integration workflows
- **Parameterized**: Tests cover multiple mass/gravity configurations

## v0.4.0 Migration Notes

### Upgrading from v0.3.x

1. **Config Path Changes**: Configuration files have been reorganized into subdirectories. Update your commands:
   - Training configs: `experiments/configs/training/`
   - Evaluation configs: `experiments/configs/evaluation/`
   - Tuning configs: `experiments/configs/tuning/`

   See [experiments/configs/README.md](../experiments/configs/README.md) for the complete old → new path mapping.

2. **Makefile Targets Unchanged**: All `make` commands continue to work without modification.

3. **Environment Variables**: Review [.env.example](../.env.example) for new variables:
   - `TUNING_OUTPUT_DIR` - Customize tuning output directory
   - CMA-ES settings (`TUNING_CMA_SIGMA0`, `TUNING_CMA_POPSIZE`)
   - Feedforward configuration variables

4. **Verification**: Run baseline evaluations with new config paths:
   ```bash
   python -m quadcopter_tracking.eval \
       --config experiments/configs/evaluation/eval_stationary_baseline.yaml \
       --episodes 5
   ```

## v0.3.0 Migration Notes

### Upgrading from v0.2.x

1. **No Breaking Changes**: All v0.2.x configurations and checkpoints remain compatible. No migration required.

2. **New Riccati-LQR Controller**: A new `riccati_lqr` controller type is available that solves the DARE for optimal feedback gains. Use with:
   ```bash
   python -m quadcopter_tracking.eval --controller riccati_lqr --episodes 10
   ```

3. **Auto-Tuning Framework**: Use `scripts/controller_autotune.py` to optimize PID/LQR gains for your specific scenario:
   ```bash
   python scripts/controller_autotune.py --controller pid --max-iterations 50
   ```

4. **Feedforward Support**: Enable feedforward for improved moving target tracking:
   ```yaml
   pid:
     feedforward_enabled: true
     ff_velocity_gain: [0.1, 0.1, 0.1]
     ff_acceleration_gain: [0.1, 0.1, 0.0]  # For circular/sinusoidal targets
   ```

   > **Note**: Velocity feedforward now correctly integrates with the D term,
   > avoiding the double-counting bug that previously degraded tracking.
   > Acceleration feedforward is recommended for circular/sinusoidal targets.

5. **Verification**: Run baseline evaluations to confirm expected performance:
   ```bash
   make eval-baseline-stationary EPISODES=10
   make eval-baseline-circular EPISODES=10
   ```

## Feedforward Troubleshooting

### Feedforward Configuration Guide

The feedforward system uses ENU-aligned velocities and accelerations:

| Parameter | Purpose | Recommended Value |
|-----------|---------|-------------------|
| `feedforward_enabled` | Master toggle | `true` for moving targets |
| `ff_velocity_gain` | Scales target velocity contribution | `[0.0-0.3, 0.0-0.3, 0.0-0.3]` |
| `ff_acceleration_gain` | Scales target acceleration | `[0.05-0.2, 0.05-0.2, 0.0]` |
| `ff_max_velocity` | Clamps velocity for stability | `10.0` (default) |
| `ff_max_acceleration` | Clamps acceleration for stability | `5.0` (default) |

### When to Use Feedforward

| Target Motion | Velocity FF | Acceleration FF | Expected Improvement |
|---------------|-------------|-----------------|---------------------|
| Stationary | Unnecessary | Unnecessary | None (already optimal) |
| Linear | Optional | Not needed | Minimal (D term handles it) |
| Circular | Optional | **Recommended** | 20-30% error reduction |
| Sinusoidal | Optional | **Recommended** | 15-25% error reduction |
| Figure-8 | Optional | **Recommended** | 10-20% error reduction |

### Feedforward Best Practices

1. **Start with feedforward disabled** to establish a baseline
2. **Enable acceleration feedforward first** for circular/oscillatory targets
3. **Use small velocity FF gains** (0.0-0.1) to avoid instability
4. **Set Z-axis acceleration gain to 0** (vertical thrust already handled by altitude control)

### Diagnosing Feedforward Issues

If feedforward degrades tracking:

1. **Check gain magnitudes**: Reduce gains if tracking oscillates
2. **Verify target acceleration**: Use `env.step()` info to confirm acceleration data exists
3. **Review diagnostics**: Use `controller.get_control_components()` to inspect FF contributions

```python
# Debug feedforward contributions
action = controller.compute_action(obs)
components = controller.get_control_components()
print(f"FF velocity term: {components['ff_velocity_term']}")
print(f"FF acceleration term: {components['ff_acceleration_term']}")

# For Riccati-LQR, also check saturation
if hasattr(controller, 'get_saturation_count'):
    print(f"Saturation count: {controller.get_saturation_count()}")
    print(f"Is saturated: {components.get('is_saturated', False)}")
```

### Riccati-LQR Feedforward Notes

The Riccati-LQR controller supports the same feedforward schema as PID and LQR
controllers. Because Riccati-LQR computes mathematically optimal feedback gains,
feedforward may provide smaller improvements compared to heuristic controllers:

- **For circular/sinusoidal targets**: Acceleration feedforward is still
  recommended and typically provides 15-25% tracking error reduction.

- **Saturation tracking**: Riccati-LQR tracks when feedforward actions are
  clamped to actuator limits. Use `controller.get_saturation_count()` to
  monitor saturation events.

- **DARE interactions**: Very aggressive feedforward gains could theoretically
  interact with the DARE solution, but in practice the feedforward is additive
  to the feedback and does not affect the underlying optimal gains.

Example configuration for circular target tracking:

```yaml
riccati_lqr:
  dt: 0.01
  feedforward_enabled: true
  ff_velocity_gain: [0.0, 0.0, 0.0]       # Not needed for circular
  ff_acceleration_gain: [0.1, 0.1, 0.0]   # Anticipate centripetal accel
  ff_max_velocity: 10.0
  ff_max_acceleration: 5.0
```

## v0.2.1 Migration Notes

### Upgrading from v0.2.0

1. **Custom Mass/Gravity Users**: If you have custom configurations with non-default mass or gravity values, ensure your controller initialization passes these values explicitly. Controllers now use these values to compute hover thrust.

2. **DeepTrackingPolicy Unchanged**: The neural network controller was not modified. It continues to learn thrust values from training data rather than using explicit hover feedforward.

3. **Verification**: Run hover tests to confirm correct behavior:
   ```bash
   python -m pytest tests/test_env_dynamics.py::TestHoverThrustIntegration -v
   ```

4. **Expected Results**: PID and LQR controllers should now achieve >80% on-target ratio for stationary targets with default configuration.

## v0.2 Migration Notes

### Upgrading from v0.1

If you are upgrading from v0.1, note the following:

1. **Checkpoint Compatibility**: All v0.1 checkpoints remain compatible with v0.2. No migration required.

2. **New Workflows**: v0.2 introduces three documented reproducible workflows. Review the [README.md](../README.md#reproducible-workflows-v02) section for best practices.

3. **Diagnostic Configurations**: New diagnostic configuration files are available in `experiments/configs/`. Consider using these when troubleshooting training issues.

4. **Comparison Framework**: The new controller comparison framework generates standardized reports. Use `make compare-controllers` and `make generate-comparison-report` for systematic comparisons.

5. **Configuration Files**: New YAML configurations added:
   - `diagnostics_stationary.yaml` - For training diagnostics on stationary targets
   - `diagnostics_linear.yaml` - For training diagnostics on linear targets
   - `training_imitation.yaml` - For imitation learning from classical controllers
   - `comparison_default.yaml` - For controller comparison workflows

### Breaking Changes

None. v0.2 maintains full backward compatibility with v0.1.

### Recommended Actions

- Review training diagnostics results if experiencing training regression
- Consider using imitation learning mode for more stable early training
- Use the new comparison framework for systematic controller evaluation
