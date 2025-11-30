# Training Quadcopter Controllers

This document describes the training pipeline for quadcopter tracking controllers.

## Overview

The training system provides:

- **Unified CLI** supporting deep learning, PID, and LQR controllers
- **Neural network controllers** mapping observations to bounded actions
- **Classical controllers** (PID, LQR) for comparison benchmarks
- **Configurable loss functions** with various error metrics and weighting
- **Episode-based training loop** with gradient descent optimization (deep only)
- **Experiment tracking** with CSV/JSON logs and checkpointing
- **Reproducibility** via seed control and checkpoint recovery

## Controller Selection

The `--controller` flag lets you choose between controller types:

| Controller | Description | Training | Checkpoints |
|------------|-------------|----------|-------------|
| `deep` | Neural network policy (default) | Yes | Yes |
| `pid` | PID controller | No (eval only) | No |
| `lqr` | LQR controller | No (eval only) | No |

```bash
# Train deep controller
python -m quadcopter_tracking.train --controller deep --epochs 100

# Run PID evaluation (no training)
python -m quadcopter_tracking.train --controller pid --epochs 10

# Run LQR evaluation (no training)
python -m quadcopter_tracking.train --controller lqr --epochs 10
```

## Quick Start

### Train with Default Configuration

```bash
# Using make (recommended)
cd lqr-quadcopter-test
make dev-install
python -m quadcopter_tracking.train --controller deep --epochs 100 --seed 42

# Or with a config file
python -m quadcopter_tracking.train --config experiments/configs/training_default.yaml
```

### Train with Custom Parameters

```bash
python -m quadcopter_tracking.train \
    --controller deep \
    --epochs 200 \
    --lr 0.0005 \
    --hidden-sizes 128 128 64 \
    --motion-type circular \
    --checkpoint-dir checkpoints/experiment1
```

### Using Makefile Commands

```bash
# Train deep controller
make train-deep EPOCHS=100 SEED=42

# Run classical controller evaluations
make train-pid EPOCHS=10
make train-lqr EPOCHS=10
```

## Controller Architecture

### Policy Network

The `PolicyNetwork` class implements a feedforward neural network:

```
Input (18 features) → Hidden layers → Output (4 actions)
```

**Input features (18 dimensions):**
- Position error (3): target position - quadcopter position
- Velocity error (3): target velocity - quadcopter velocity
- Quadcopter attitude (3): roll, pitch, yaw
- Angular velocity (3): p, q, r
- Target position (3): absolute target position
- Target velocity (3): absolute target velocity

**Output actions (4 dimensions, bounded):**
- Thrust: [0, max_thrust] N
- Roll rate: [-max_rate, max_rate] rad/s
- Pitch rate: [-max_rate, max_rate] rad/s
- Yaw rate: [-max_rate, max_rate] rad/s

### Configurable Architecture

```python
from quadcopter_tracking.controllers import DeepTrackingPolicy

controller = DeepTrackingPolicy(config={
    "hidden_sizes": [128, 128, 64],  # Three hidden layers
    "activation": "leaky_relu",      # Options: relu, tanh, elu, leaky_relu
    "output_bounds": {
        "thrust": (0.0, 25.0),       # Custom bounds
        "roll_rate": (-5.0, 5.0),
        "pitch_rate": (-5.0, 5.0),
        "yaw_rate": (-3.0, 3.0),
    },
})
```

## Loss Functions

### Tracking Loss

The primary loss combines position error, velocity error, and control effort:

```
L_total = w_pos * L_pos + w_vel * L_vel + w_ctrl * L_ctrl
```

Where:
- `L_pos` = weighted position error norm
- `L_vel` = weighted velocity error norm
- `L_ctrl` = control effort penalty

**Error norm options:**
- `l2`: Quadratic loss (default)
- `l1`: Absolute value loss
- `huber`: Smooth transition (robust to outliers)

### Weight Matrices

Configure weight matrices for different error components:

```python
from quadcopter_tracking.utils import TrackingLoss
import numpy as np

# Emphasize z-axis tracking
pos_weight = np.diag([1.0, 1.0, 2.0])  # Higher weight on z

loss = TrackingLoss(
    position_weight=pos_weight,
    velocity_weight=0.1,
    control_weight=0.01,
    error_type="l2",
)
```

### Reward Shaping

Optional reward-based loss component for reinforcement learning:

```python
from quadcopter_tracking.utils import RewardShapingLoss

reward_loss = RewardShapingLoss(
    target_radius=0.5,      # On-target threshold
    on_target_bonus=1.0,    # Reward for being on-target
    distance_penalty=1.0,   # Penalty coefficient
    smoothing="exp",        # Options: none, exp, sigmoid
)
```

## Training Configuration

### YAML Configuration File

```yaml
# experiments/configs/my_config.yaml

# Controller selection (deep, lqr, pid)
controller: deep

epochs: 200
episodes_per_epoch: 10
batch_size: 32

# Optimizer (deep controller only)
learning_rate: 0.001
optimizer: adam  # adam, sgd, adamw
weight_decay: 0.0
grad_clip: 1.0

# Network (deep controller only)
hidden_sizes: [64, 64]
activation: relu

# Loss weights
position_weight: 1.0
velocity_weight: 0.1
control_weight: 0.01
error_type: l2

# Environment
env_seed: 42
target_motion_type: circular
episode_length: 30.0
target_radius: 0.5

# Checkpointing
checkpoint_dir: checkpoints
checkpoint_interval: 10
save_best: true

# Logging
log_dir: experiments/logs
log_interval: 1
```

### Command Line Arguments

All configuration options can be overridden via CLI:

```bash
python -m quadcopter_tracking.train \
    --config experiments/configs/training_default.yaml \
    --epochs 500 \
    --lr 0.0001 \
    --optimizer adamw \
    --hidden-sizes 128 128 \
    --motion-type figure8 \
    --seed 123
```

## Training Loop

### Episode-Based Training

1. **Reset** environment with episode-specific seed
2. **Collect** trajectory data (observations, actions, rewards)
3. **Compute** loss from batch of collected samples
4. **Update** network via gradient descent
5. **Log** metrics and save checkpoints

### Batch Processing

Training processes episodes in batches:

```python
# Each epoch
for episode in range(episodes_per_epoch):
    # Run episode, collect data
    for step in range(max_steps):
        action = controller.compute_action(obs)
        obs, reward, done, info = env.step(action)
        episode_data.append(...)
    
    # Sample batch and update
    batch = sample(episode_data, batch_size)
    loss = compute_loss(batch)
    optimizer.step()
```

## Curriculum Learning

Enable progressive difficulty for stable training:

```yaml
use_curriculum: true
curriculum_start_difficulty: 0.3
curriculum_end_difficulty: 1.0
```

Difficulty interpolates linearly from start to end over training epochs.

## Checkpointing and Recovery

### Automatic Checkpointing

Checkpoints are saved at configurable intervals:

```
checkpoints/
├── train_20240101_120000_42_epoch0010.pt
├── train_20240101_120000_42_epoch0020.pt
├── train_20240101_120000_42_best.pt
└── train_20240101_120000_42_final.pt
```

### Resume Training

Resume from a checkpoint:

```bash
python -m quadcopter_tracking.train \
    --config experiments/configs/training_default.yaml \
    --resume checkpoints/train_20240101_120000_42_epoch0050.pt
```

### NaN Recovery

Training automatically recovers from gradient explosion:

1. Detects NaN/Inf in loss
2. Reduces learning rate
3. Loads best checkpoint
4. Continues training

Configure recovery:

```yaml
nan_recovery_attempts: 3
lr_reduction_factor: 0.5
```

## Experiment Tracking

### Log Files

Training creates JSON and CSV logs:

```
experiments/logs/
├── train_20240101_120000_42_config.yaml
├── train_20240101_120000_42_log.json
└── train_20240101_120000_42_log.csv
```

### Logged Metrics

Each epoch records:
- `total`: Combined loss
- `position`: Position error component
- `velocity`: Velocity error component
- `control`: Control effort component
- `mean_reward`: Average episode reward
- `mean_on_target_ratio`: Time spent within target radius
- `mean_tracking_error`: Average distance to target

### Analyzing Results

```python
import json
import pandas as pd

# Load JSON log
with open("experiments/logs/train_xxx_log.json") as f:
    log = json.load(f)

# Or use CSV
df = pd.read_csv("experiments/logs/train_xxx_log.csv")
df.plot(x="epoch", y=["total", "mean_tracking_error"])
```

## Adding Alternative Controllers

### Step 1: Create Controller Class

```python
# src/quadcopter_tracking/controllers/my_controller.py
from . import BaseController

class MyController(BaseController):
    def __init__(self, config=None):
        super().__init__(name="my_controller", config=config)
        # Initialize your model
    
    def compute_action(self, observation):
        # Extract features
        quad = observation["quadcopter"]
        target = observation["target"]
        
        # Compute action
        return {
            "thrust": ...,
            "roll_rate": ...,
            "pitch_rate": ...,
            "yaw_rate": ...,
        }
```

### Step 2: Export in Package

```python
# src/quadcopter_tracking/controllers/__init__.py
from .my_controller import MyController
__all__.append("MyController")
```

### Step 3: Create Training Config

```yaml
# experiments/configs/my_controller.yaml
controller_type: my_controller
epochs: 100
# ... other parameters
```

## Hyperparameter Tuning

### Recommended Starting Points

| Parameter | Range | Notes |
|-----------|-------|-------|
| learning_rate | 1e-4 to 1e-2 | Lower for larger networks |
| hidden_sizes | [32,32] to [256,256,128] | Match task complexity |
| batch_size | 16 to 128 | Larger = more stable |
| position_weight | 0.5 to 5.0 | Increase for tighter tracking |
| control_weight | 0.001 to 0.1 | Lower = more aggressive |

### Tuning Strategy

1. Start with default config
2. Run short training (20-50 epochs)
3. Check for:
   - NaN losses → reduce learning rate
   - Slow convergence → increase learning rate
   - High tracking error → increase position_weight
   - Jerky control → increase control_weight
4. Iterate and compare experiments

## GPU Training

PyTorch automatically uses GPU if available:

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU
python -m quadcopter_tracking.train --device cpu

# Specify GPU
CUDA_VISIBLE_DEVICES=0 python -m quadcopter_tracking.train
```

## Troubleshooting

### NaN Losses

- Reduce learning rate
- Enable gradient clipping: `--grad-clip 1.0`
- Try smaller network
- Check for invalid environment observations

### Poor Convergence

- Increase training epochs
- Try different optimizer (adamw often works well)
- Enable curriculum learning
- Increase batch size

### Memory Issues

- Reduce batch size
- Use smaller network
- Enable gradient checkpointing (advanced)

### Slow Training

- Reduce max_steps_per_episode
- Fewer episodes_per_epoch
- Use GPU if available
- Profile with `torch.profiler`

## Training Diagnostics

The training pipeline includes an opt-in diagnostics system for analyzing training behavior and identifying failure modes.

### Enabling Diagnostics

Enable diagnostics via command line:

```bash
python -m quadcopter_tracking.train \
    --epochs 50 \
    --diagnostics \
    --diagnostics-log-interval 1
```

Or via configuration file:

```yaml
# experiments/configs/my_config.yaml
diagnostics_enabled: true
diagnostics_log_observations: true
diagnostics_log_actions: true
diagnostics_log_gradients: true
diagnostics_log_interval: 10
diagnostics_generate_plots: true
```

### Diagnostic Output Files

When enabled, diagnostics creates the following files in `<log_dir>/diagnostics/`:

| File | Contents |
|------|----------|
| `*_step_diagnostics.json` | Per-step observation, action, loss, gradient stats |
| `*_epoch_diagnostics.json` | Aggregated per-epoch metrics |
| `*_epoch_diagnostics.csv` | CSV format for easy analysis |
| `*_diagnostics_summary.json` | Summary with identified issues |
| `*_loss_trajectory.png` | Loss over epochs plot |
| `*_error_trajectory.png` | Tracking error over epochs plot |
| `*_gradient_trajectory.png` | Gradient norm over epochs plot |
| `*_on_target_trajectory.png` | On-target ratio over epochs plot |

### Key Diagnostic Metrics

| Metric | Description | Healthy Range |
|--------|-------------|---------------|
| `mean_gradient_norm` | Gradient magnitude | 0.1 - 10.0 |
| `action_magnitude_mean` | Control output scale | Match actuator limits |
| `observation_range` | Input feature bounds | Reasonably bounded |
| `num_nan_gradients` | NaN gradient count | 0 |
| `num_inf_gradients` | Inf gradient count | 0 |

### Interpreting Diagnostics

**Gradient Norms:**
- Very small (< 1e-7): Vanishing gradients, network stuck
- Very large (> 100): Exploding gradients, reduce learning rate
- Near clip threshold: Training may be fighting gradient clipping

**Loss Trajectory:**
- Steady decrease: Healthy training
- Increasing loss: Learning rate too high or loss misaligned
- Flat loss: Network not learning, check data flow

**Action Statistics:**
- Saturated actions: Bounds too tight or network over-confident
- Very small actions: Network under-confident or initialization issue
- High variance: Unstable policy

### Example Diagnostic Config

```yaml
# experiments/configs/diagnostics_experiment.yaml
epochs: 20
episodes_per_epoch: 5
max_steps_per_episode: 500
batch_size: 16

learning_rate: 0.001
optimizer: adam
hidden_sizes: [64, 64]

target_motion_type: stationary
episode_length: 10.0

# Enable all diagnostics
diagnostics_enabled: true
diagnostics_log_observations: true
diagnostics_log_actions: true
diagnostics_log_gradients: true
diagnostics_log_interval: 1
diagnostics_generate_plots: true
```

### Programmatic Access

```python
from quadcopter_tracking.utils import Diagnostics, DiagnosticsConfig

# Create diagnostics manager
config = DiagnosticsConfig(
    enabled=True,
    log_interval=5,
    generate_plots=True,
)
diag = Diagnostics(config=config, output_dir="my_diagnostics")

# In training loop
diag.log_step(epoch=0, step=i, observation=obs, action=action, losses=losses)

# At epoch end
epoch_diag = diag.log_epoch(epoch=0, metrics=epoch_metrics)

# Save all outputs
diag.save_summary()
diag.generate_plots()
```

### Known Training Issues

Based on diagnostic analysis, the current deep controller training exhibits:

1. **Training Regression**: Loss increases after initial epochs
2. **Tracking Error Growth**: Error increases from ~50m to ~100m over training
3. **Low On-Target Ratio**: Never exceeds 15%, far below 80% target

See [docs/results.md](results.md#training-diagnostics-results) for detailed analysis and recommended remediation.

## API Reference

### DeepTrackingPolicy

```python
class DeepTrackingPolicy(BaseController):
    def __init__(self, config=None, device=None)
    def compute_action(self, observation: dict) -> dict
    def train_mode() -> None
    def eval_mode() -> None
    def get_parameters() -> list[Parameter]
    def save_checkpoint(path, metadata=None) -> None
    def load_checkpoint(path) -> dict
```

### TrainingConfig

```python
class TrainingConfig:
    epochs: int
    episodes_per_epoch: int
    learning_rate: float
    hidden_sizes: list[int]
    # ... see source for full list
    
    @classmethod
    def from_file(path) -> TrainingConfig
    def to_dict() -> dict
```

### Trainer

```python
class Trainer:
    def __init__(self, config: TrainingConfig)
    def train() -> dict  # Returns summary
    def _train_epoch() -> dict
    def _save_checkpoint(epoch, metrics, is_best=False)
```

## Extending to Alternative Controllers

### Implementing Custom Learning Algorithms

To implement alternative learning approaches (e.g., PPO, SAC, model-based RL):

1. **Create a new controller class** inheriting from `BaseController`
2. **Implement the training loop** with your algorithm
3. **Use the existing loss functions** or define custom ones

```python
from quadcopter_tracking.controllers import BaseController
from quadcopter_tracking.env import QuadcopterEnv

class CustomRLController(BaseController):
    def __init__(self, config=None):
        super().__init__(name="custom_rl", config=config)
        # Initialize your policy, value networks, etc.
    
    def compute_action(self, observation):
        # Your policy inference
        pass
    
    def train_step(self, batch):
        # Your training update
        pass
```

### Imperfect Information Controllers

Handling partial observability (e.g., with noisy sensor data) is a planned future enhancement. This will likely involve state estimation techniques like Kalman filters and recurrent neural network architectures.

See [ROADMAP.md](../ROADMAP.md) for detailed design proposals and pseudocode.

### Transfer Learning

**Current Limitation**: The Trainer class creates its own controller internally, which does not directly support transfer learning from pretrained weights.

#### Current Workaround

For inference with pretrained weights, use `DeepTrackingPolicy.load_checkpoint()`:

```python
from quadcopter_tracking.controllers import DeepTrackingPolicy

# Load pretrained weights for inference/evaluation
controller = DeepTrackingPolicy()
controller.load_checkpoint("checkpoints/pretrained.pt")

# Use controller for evaluation
from quadcopter_tracking.eval import Evaluator
evaluator = Evaluator(controller=controller)
summary = evaluator.evaluate(num_episodes=10)
```

True transfer learning support (loading pretrained weights for fine-tuning) is planned for a future release. See [ROADMAP.md](../ROADMAP.md) for the proposed API design.

## Classical Controller Tuning

This section provides guidance for tuning the PID and LQR classical controllers.

### PID Controller

The PID controller uses separate gains for each axis (X, Y, Z):

```python
from quadcopter_tracking.controllers import PIDController

pid = PIDController(config={
    "kp_pos": [2.0, 2.0, 4.0],   # Proportional gains [x, y, z]
    "ki_pos": [0.1, 0.1, 0.2],   # Integral gains [x, y, z]
    "kd_pos": [1.5, 1.5, 2.0],   # Derivative gains [x, y, z]
    "integral_limit": 5.0,       # Windup prevention clamp
})
```

#### Gain Tuning Guidelines

| Parameter | Effect of Increase | Typical Range |
|-----------|-------------------|---------------|
| `kp_pos` | Faster response, more overshoot | 0.5 - 5.0 |
| `ki_pos` | Eliminates steady-state error, causes oscillation | 0.01 - 0.5 |
| `kd_pos` | More damping, reduces overshoot | 0.5 - 3.0 |
| `integral_limit` | Higher limit = more correction, but risk of windup | 2.0 - 10.0 |

#### Tuning Strategy

1. **Start with proportional only**: Set `ki_pos = [0, 0, 0]`, tune `kp_pos` until system responds without excessive oscillation
2. **Add derivative**: Increase `kd_pos` to reduce overshoot and oscillation
3. **Add integral**: Slowly increase `ki_pos` to eliminate steady-state error
4. **Adjust windup limit**: If tracking large errors, increase `integral_limit`

#### Common Issues

- **Oscillation**: Reduce `kp_pos` or increase `kd_pos`
- **Slow response**: Increase `kp_pos`
- **Overshoot**: Increase `kd_pos` or reduce `kp_pos`
- **Steady-state error**: Increase `ki_pos`
- **Windup (runaway after large error)**: Reduce `integral_limit`

#### Example Configurations

**Stationary Target (conservative)**:
```yaml
pid:
  kp_pos: [1.5, 1.5, 3.0]
  ki_pos: [0.05, 0.05, 0.1]
  kd_pos: [1.0, 1.0, 1.5]
  integral_limit: 3.0
```

**Dynamic Tracking (aggressive)**:
```yaml
pid:
  kp_pos: [3.0, 3.0, 5.0]
  ki_pos: [0.15, 0.15, 0.3]
  kd_pos: [2.0, 2.0, 2.5]
  integral_limit: 8.0
```

### LQR Controller

The LQR controller computes optimal feedback gains from cost weights:

```python
from quadcopter_tracking.controllers import LQRController

lqr = LQRController(config={
    "q_pos": [10.0, 10.0, 20.0],  # Position error cost [x, y, z]
    "q_vel": [5.0, 5.0, 10.0],   # Velocity error cost [vx, vy, vz]
    "r_thrust": 0.1,              # Thrust control cost
    "r_rate": 1.0,                # Angular rate control cost
})
```

#### Cost Weight Guidelines

| Parameter | Effect of Increase | Typical Range |
|-----------|-------------------|---------------|
| `q_pos` | Tighter position tracking | 1.0 - 50.0 |
| `q_vel` | More velocity damping | 1.0 - 20.0 |
| `r_thrust` | Smoother thrust changes | 0.01 - 1.0 |
| `r_rate` | Smoother attitude rate changes | 0.1 - 5.0 |

#### Tuning Strategy

1. **Start with balanced weights**: Use defaults
2. **Adjust tracking tightness**: Increase `q_pos` for tighter tracking
3. **Add damping**: Increase `q_vel` if too oscillatory
4. **Control smoothness**: Increase `r_*` weights for smoother control

#### Pre-computed K Matrix

For advanced users, you can provide a pre-computed feedback gain matrix:

```python
import numpy as np

# Custom 4x6 gain matrix
K = np.array([
    [0, 0, 14.14, 0, 0, 7.42],      # Z error -> thrust
    [0, 3.16, 0, 0, 2.35, 0],       # Y error -> roll rate
    [-3.16, 0, 0, -2.35, 0, 0],     # X error -> pitch rate
    [0, 0, 0, 0, 0, 0],             # yaw (unused)
])

lqr = LQRController(config={"K": K.tolist()})
```

#### Operating Envelope

The LQR linearization is valid for:
- Attitude angles < 30 degrees
- Velocities < 5 m/s
- Position errors < 10 m

Outside this envelope, the controller may perform poorly.

### Comparing Controllers

Run evaluation with different controllers:

```bash
# Evaluate PID
python -m quadcopter_tracking.eval --controller pid --episodes 10 --motion-type stationary

# Evaluate LQR
python -m quadcopter_tracking.eval --controller lqr --episodes 10 --motion-type stationary

# Evaluate Deep
python -m quadcopter_tracking.eval --controller deep --checkpoint checkpoints/best.pt
```

### Configuration via YAML

Add controller configurations to your experiment YAML files:

```yaml
# experiments/configs/my_experiment.yaml

# ... other settings ...

# PID gains for this experiment
pid:
  kp_pos: [2.0, 2.0, 4.0]
  ki_pos: [0.1, 0.1, 0.2]
  kd_pos: [1.5, 1.5, 2.0]
  integral_limit: 5.0

# LQR weights for this experiment
lqr:
  q_pos: [10.0, 10.0, 20.0]
  q_vel: [5.0, 5.0, 10.0]
  r_thrust: 0.1
  r_rate: 1.0
```
