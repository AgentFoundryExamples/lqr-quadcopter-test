# Training Deep Learning Controllers

This document describes the deep learning training pipeline for quadcopter tracking controllers.

## Overview

The training system provides:

- **Neural network controllers** mapping observations to bounded actions
- **Configurable loss functions** with various error metrics and weighting
- **Episode-based training loop** with gradient descent optimization
- **Experiment tracking** with CSV/JSON logs and checkpointing
- **Reproducibility** via seed control and checkpoint recovery

## Quick Start

### Train with Default Configuration

```bash
# Using make (recommended)
cd lqr-quadcopter-test
make dev-install
python -m quadcopter_tracking.train --epochs 100 --seed 42

# Or with a config file
python -m quadcopter_tracking.train --config experiments/configs/training_default.yaml
```

### Train with Custom Parameters

```bash
python -m quadcopter_tracking.train \
    --epochs 200 \
    --lr 0.0005 \
    --hidden-sizes 128 128 64 \
    --motion-type circular \
    --checkpoint-dir checkpoints/experiment1
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
epochs: 200
episodes_per_epoch: 10
batch_size: 32

# Optimizer
learning_rate: 0.001
optimizer: adam  # adam, sgd, adamw
weight_decay: 0.0
grad_clip: 1.0

# Network
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
