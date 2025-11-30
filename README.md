# Quadcopter Target Tracking Research

A Python-based research repository for quadcopter target-tracking studies with LQR and ML-based controllers.

## Project Goals

This project provides a simulation environment and controller implementations for studying quadcopter target-tracking problems. The primary objectives are:

1. **Evaluate classical control methods** (LQR, PID) for target tracking
2. **Train and compare learning-based approaches** using deep reinforcement learning
3. **Establish reproducible benchmarks** for tracking performance
4. **Document research findings** systematically

## Assumptions

The current implementation makes the following simplifying assumptions:

- **Perfect target information**: The quadcopter has exact knowledge of target position and velocity (no sensor noise)
- **Smooth target motion**: Target trajectories are differentiable and continuous
- **Idealized dynamics**: Simplified quadcopter model without disturbances
- **3D tracking**: Full spatial tracking problem in three dimensions

These assumptions will be relaxed in future iterations as the research progresses.

## Success Criteria

A tracking episode is considered **successful** when:

| Metric | Threshold |
|--------|-----------|
| Episode duration | ≥ 30 seconds |
| On-target ratio | ≥ 80% |
| Target radius | ≤ 0.5 meters |

**Definition**: The quadcopter is "on-target" when its position is within 0.5 meters of the target position. Success requires maintaining this proximity for at least 80% of episodes lasting 30 seconds or more.

## Installation

### Requirements
- Python 3.10+
- Linux (recommended), macOS, or Windows

### Setup

```bash
# Clone the repository
git clone https://github.com/AgentFoundryExamples/lqr-quadcopter-test.git
cd lqr-quadcopter-test

# Install with dependencies
make install

# Or install with development dependencies
make dev-install
```

### GPU Support

PyTorch is included for future ML-based controllers. For GPU support:
- Install CUDA toolkit (11.8+ recommended)
- PyTorch will automatically use GPU if available

For CPU-only machines:
- No additional setup required
- Set `CUDA_VISIBLE_DEVICES=""` to force CPU usage

## Quick Start

### Run an Experiment

```bash
# Run with default configuration
make run-experiment

# Run with custom seed
make run-experiment SEED=123

# Run with custom config file
make run-experiment CONFIG=configs/circular.yaml
```

### Configuration

Copy the example environment file and customize:

```bash
cp .env.example .env
# Edit .env with your preferred settings
```

Key configuration parameters:
- `QUADCOPTER_SEED`: Random seed for reproducibility
- `QUADCOPTER_EPISODE_LENGTH`: Duration of tracking episodes (seconds)
- `QUADCOPTER_TARGET_RADIUS`: On-target threshold (meters)
- `QUADCOPTER_TARGET_MOTION_TYPE`: Target motion pattern (linear, circular, sinusoidal)

See [.env.example](.env.example) for all available options.

## Project Structure

```
├── src/
│   ├── env/           # Environment simulation
│   ├── controllers/   # Controller implementations (LQR, PID, Neural)
│   ├── utils/         # Shared utilities and loss functions
│   └── train.py       # Deep learning training script
├── experiments/
│   └── configs/       # Training configuration files
├── docs/              # Documentation
│   ├── architecture.md
│   ├── environment.md
│   └── training.md
├── tests/             # Test suite
├── pyproject.toml     # Python package configuration
├── Makefile           # CLI commands
├── .env.example       # Environment variable template
└── README.md          # This file
```

See [docs/architecture.md](docs/architecture.md) for detailed architecture documentation.

## Environment Usage

The simulation environment provides a realistic quadcopter dynamics model with target tracking capabilities.

### Basic Example

```python
from quadcopter_tracking.env import QuadcopterEnv

# Create environment
env = QuadcopterEnv()
obs = env.reset(seed=42)

# Run simulation
done = False
while not done:
    # Get state information
    quad_pos = obs["quadcopter"]["position"]
    target_pos = obs["target"]["position"]
    
    # Compute control action
    error = target_pos - quad_pos
    action = {
        "thrust": 9.81 + error[2] * 2.0,
        "roll_rate": error[1] * 0.5,
        "pitch_rate": -error[0] * 0.5,
        "yaw_rate": 0.0,
    }
    
    obs, reward, done, info = env.step(action)

print(f"Success: {info['success']}")
print(f"On-target ratio: {info['on_target_ratio']:.1%}")
```

### State Vector

| Component | Variables | Units |
|-----------|-----------|-------|
| Position | x, y, z | meters |
| Velocity | vx, vy, vz | m/s |
| Attitude | roll, pitch, yaw | radians |
| Angular rate | p, q, r | rad/s |

### Target Motion Patterns

- **linear**: Constant velocity in random direction
- **circular**: Orbital motion in horizontal plane
- **sinusoidal**: Multi-axis oscillation
- **figure8**: Lemniscate trajectory
- **stationary**: Fixed position (hover reference)

### Configuration

```python
from quadcopter_tracking.env import QuadcopterEnv, EnvConfig

config = EnvConfig()
config.target.motion_type = "circular"
config.target.radius = 3.0
config.simulation.max_episode_time = 60.0

env = QuadcopterEnv(config=config)
```

See [docs/environment.md](docs/environment.md) for complete documentation.

## Training Deep Learning Controllers

The project includes a complete training pipeline for neural network controllers.

### Quick Start Training

```bash
# Train with default configuration
python -m quadcopter_tracking.train --epochs 100 --seed 42

# Train with config file
python -m quadcopter_tracking.train --config experiments/configs/training_default.yaml

# Resume training from checkpoint
python -m quadcopter_tracking.train --resume checkpoints/train_xxx_epoch0050.pt
```

### Training Configuration

Configure training via YAML files or command line:

```bash
python -m quadcopter_tracking.train \
    --epochs 200 \
    --lr 0.001 \
    --hidden-sizes 128 128 \
    --motion-type circular \
    --checkpoint-dir checkpoints/my_experiment
```

See example configurations in `experiments/configs/`:
- `training_default.yaml`: Standard training setup
- `training_fast.yaml`: Quick testing configuration
- `training_large.yaml`: Extended training with larger network

### Using Trained Controllers

```python
from quadcopter_tracking.controllers import DeepTrackingPolicy
from quadcopter_tracking.env import QuadcopterEnv

# Load trained controller
controller = DeepTrackingPolicy(config={
    "checkpoint_path": "checkpoints/best_model.pt"
})

# Run in environment
env = QuadcopterEnv()
obs = env.reset(seed=42)

done = False
while not done:
    action = controller.compute_action(obs)
    obs, reward, done, info = env.step(action)

print(f"On-target ratio: {info['on_target_ratio']:.1%}")
```

See [docs/training.md](docs/training.md) for complete training documentation.

## Evaluating Controllers

The evaluation pipeline assesses controller performance against success criteria.

### Quick Start Evaluation

```bash
# Evaluate a trained deep learning controller
python -m quadcopter_tracking.eval --checkpoint checkpoints/best.pt --episodes 10

# Evaluate with specific motion type
python -m quadcopter_tracking.eval \
    --controller deep \
    --checkpoint checkpoints/model.pt \
    --motion-type circular \
    --episodes 20

# Evaluate classical controllers
python -m quadcopter_tracking.eval --controller lqr --episodes 10
```

### Success Criteria

| Metric | Threshold |
|--------|-----------|
| Episode Duration | ≥ 30 seconds |
| On-Target Ratio | ≥ 80% |
| Target Radius | ≤ 0.5 meters |

### Output

Evaluation generates:
- `reports/metrics.json` - Detailed metrics
- `reports/plots/position_tracking.png` - Trajectory visualization
- `reports/plots/tracking_error.png` - Error analysis

See [docs/results.md](docs/results.md) for complete evaluation documentation.

## Development

```bash
# Run tests
make test

# Run linter
make lint

# Auto-format code
make format

# Clean build artifacts
make clean
```

## Workflow Overview

1. **Configure**: Set experiment parameters via `.env` or config file
2. **Train**: Train controllers with `python -m quadcopter_tracking.train`
3. **Evaluate**: Assess performance with `python -m quadcopter_tracking.eval`
4. **Analyze**: Review metrics in `reports/` and generated plots
5. **Iterate**: Modify controllers or parameters and repeat

## System Dependencies

Most dependencies are pure Python. If you encounter issues:

### Headless Linux
```bash
# For matplotlib rendering
apt install libgl1-mesa-glx
export MPLBACKEND=Agg
```

### Missing Environment Variables
The configuration system falls back to sensible defaults. See [.env.example](.env.example) for all variables and their defaults.



# Permanents (License, Contributing, Author)

Do not change any of the below sections

## License

This project is distributed under the MIT license - see the LICENSE file for details.

## Contributing

Feel free to submit issues and enhancement requests!

## Author

Created by Agent Foundry and John Brosnihan
