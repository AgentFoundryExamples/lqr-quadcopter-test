# Quadcopter Target Tracking Architecture

This document describes the architecture and design principles for the quadcopter target-tracking research repository.

## Overview

The project is structured around a clean separation between **environment simulation** and **controller implementation**, enabling:

1. Independent development and testing of controllers
2. Fair comparison across different control strategies
3. Reproducible experiments with configurable parameters

## Package Structure

```
src/
├── env/           # Environment simulation package
├── controllers/   # Controller implementations package
└── utils/         # Shared utilities package
```

### Environment Package (`src/env/`)

**Responsibility**: Simulate quadcopter dynamics, target motion, and provide observations/rewards.

The environment package owns:
- Quadcopter physics model (simplified or detailed)
- Target trajectory generation
- State observation interface
- Reward computation for tracking performance
- Episode termination logic

**Key Interfaces**:
- `reset(seed) -> observation`: Initialize episode, return initial state
- `step(action) -> (observation, reward, done, info)`: Execute action, return results

**Design Decisions**:
- Environment is the single source of truth for state
- Observations are dictionaries for flexibility
- Reward design focuses on tracking accuracy within radius threshold

### Controllers Package (`src/controllers/`)

**Responsibility**: Transform observations into control actions.

The controllers package owns:
- Control algorithm implementations (LQR, PID, neural)
- Controller-specific state (e.g., integral error for PID)
- Gain tuning interfaces

**Key Interfaces**:
- `compute_action(observation) -> action`: Main control computation
- `reset()`: Clear controller state between episodes

**Design Decisions**:
- Controllers are observation-to-action transformations
- Stateless where possible (LQR), stateful when needed (PID integral)
- Common base class for consistent interface

### Utilities Package (`src/utils/`)

**Responsibility**: Shared functionality used by both environment and controllers.

The utilities package owns:
- Configuration loading (YAML/JSON + env vars)
- Data logging for experiments
- Plotting and visualization
- Common math utilities

**Design Decisions**:
- Config loader provides defaults, file overrides, then env var overrides
- Logging captures sufficient data for post-hoc analysis
- Plotting provides standardized visualizations

## Classical Controller Pipeline

The repository provides two classical controllers (PID and LQR) that share a common pipeline for processing observations and generating control actions.

### Observation Processing

Both classical controllers extract state information from the observation dictionary:

```python
observation = {
    "quadcopter": {
        "position": [x, y, z],           # meters
        "velocity": [vx, vy, vz],        # m/s
        "attitude": [roll, pitch, yaw],  # radians
        "angular_velocity": [p, q, r],   # rad/s
    },
    "target": {
        "position": [x, y, z],           # meters
        "velocity": [vx, vy, vz],        # m/s
    },
    "time": float,                       # seconds
}
```

### Control Output Mapping

Both controllers output a dictionary with the following keys:

```python
action = {
    "thrust": float,      # Total thrust in Newtons [0, max_thrust]
    "roll_rate": float,   # Desired roll rate in rad/s [-max_rate, max_rate]
    "pitch_rate": float,  # Desired pitch rate in rad/s [-max_rate, max_rate]
    "yaw_rate": float,    # Desired yaw rate in rad/s [-max_rate, max_rate]
}
```

### PID Controller Pipeline

The PID controller uses a cascaded structure for position tracking:

```
                    Position Error              PID Terms
Target Position ──┬─────────────────►  ┌─────────────────┐
                  │                    │  P: kp * error  │
Quad Position   ──┴─────►  error ────► │  I: ki * ∫error │ ──► Desired Accel
                                       │  D: kd * ḋerror │
Target Velocity ──┬─────────────────►  └─────────────────┘
                  │                            │
Quad Velocity   ──┴─────► vel_error ──────────┘

Desired Accel ──► Control Mapping ──► [thrust, roll_rate, pitch_rate, yaw_rate]
```

**Control Mapping:**
- Z-axis acceleration → thrust adjustment above hover
- X-axis acceleration → pitch rate (negative sign: pitch forward to move +X)
- Y-axis acceleration → roll rate (positive: roll right to move +Y)
- Yaw rate → zero (no heading tracking)

**Key Features:**
- Integral windup prevention via configurable clamp
- Per-axis gain tuning (3D arrays)
- Automatic hover thrust calculation from mass/gravity

### LQR Controller Pipeline

The LQR controller uses a pre-computed feedback gain matrix:

```
                State Error Vector (6D)
Target State ──┬──────────────────────────────────────────┐
               │                                          │
Quad State   ──┴──► [pos_error(3), vel_error(3)] ──────► │
                                                          ▼
                                            ┌─────────────────────┐
                                            │   u = K @ state     │
                                            │   K: 4x6 gain matrix│
                                            └──────────┬──────────┘
                                                       │
                                                       ▼
                              [thrust_adj, roll_rate, pitch_rate, yaw_rate]
```

**Gain Computation:**
For the linearized hover dynamics, LQR gains are computed from cost matrices:
- `Q`: State cost matrix (6x6 diagonal)
- `R`: Control cost matrix (4x4 diagonal)

The feedback law minimizes: `J = ∫(x'Qx + u'Ru) dt`

**Operating Envelope:**
The LQR linearization assumes:
- Small attitude angles (< 30 degrees from hover)
- Moderate velocities (< 5 m/s)
- Position errors up to ±10 meters

For aggressive maneuvers outside this envelope, performance degrades.

### Metrics and Logging Integration

Classical controllers integrate with the metrics infrastructure:

```python
from quadcopter_tracking.controllers import PIDController
from quadcopter_tracking.eval import Evaluator

# Create controller and evaluator
controller = PIDController(config={"kp_pos": [2.0, 2.0, 4.0]})
evaluator = Evaluator(controller=controller)

# Run evaluation - generates metrics and plots
summary = evaluator.evaluate(num_episodes=10)
evaluator.save_report(summary)
evaluator.generate_all_plots(summary)
```

The evaluation pipeline records:
- Tracking error at each timestep
- On-target ratio (time within target radius)
- Control effort (action magnitudes)
- Overshoot events
- Action violations (clipping events)

## Environment-Controller Separation

### Boundary Definition

```
┌─────────────────────────────────────────────────────────────────┐
│                        EXPERIMENT LOOP                          │
│                                                                 │
│   ┌─────────────┐         observation         ┌─────────────┐  │
│   │             │ ─────────────────────────▶  │             │  │
│   │ Environment │                             │ Controller  │  │
│   │             │ ◀─────────────────────────  │             │  │
│   └─────────────┘          action             └─────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### What Crosses the Boundary

**Environment → Controller (Observation)**:
- Quadcopter position, velocity, orientation
- Target position, velocity
- Time remaining in episode

**Controller → Environment (Action)**:
- Thrust command
- Roll, pitch, yaw rate commands

### What Stays Within Each Side

**Environment Only**:
- Physics integration
- Collision detection
- Reward computation
- Random target trajectory generation

**Controller Only**:
- Control gain computation
- Reference trajectory planning
- Internal state estimation

## Configuration System

### Priority Order
1. Environment variables (highest)
2. Config file (YAML/JSON)
3. Default values (lowest)

### Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed` | 42 | Random seed for reproducibility |
| `episode_length` | 30.0 | Episode duration (seconds) |
| `target.radius_requirement` | 0.5 | On-target threshold (meters) |
| `target.motion_type` | linear | Target motion pattern |
| `success_criteria.min_on_target_ratio` | 0.8 | Required on-target time fraction |

### Classical Controller Configuration

Controller gains are configured via YAML:

```yaml
# PID Controller
pid:
  kp_pos: [2.0, 2.0, 4.0]   # Proportional gains [x, y, z]
  ki_pos: [0.1, 0.1, 0.2]   # Integral gains [x, y, z]
  kd_pos: [1.5, 1.5, 2.0]   # Derivative gains [x, y, z]
  integral_limit: 5.0        # Windup prevention

# LQR Controller
lqr:
  q_pos: [10.0, 10.0, 20.0]  # Position cost weights
  q_vel: [5.0, 5.0, 10.0]    # Velocity cost weights
  r_thrust: 0.1               # Thrust control cost
  r_rate: 1.0                 # Rate control cost
```

## Success Criteria

A tracking episode is considered **successful** when:

1. Episode runs for at least **30 seconds**
2. Quadcopter maintains position within **0.5 meters** of target for **≥80%** of the episode

This criteria ensures:
- Controllers handle sustained tracking (not just instantaneous)
- Some tolerance for transient errors (20% off-target allowed)
- Clear, measurable benchmark for comparison

## Controller Comparison Guide

| Controller | Strengths | Weaknesses | Best For |
|------------|-----------|------------|----------|
| **PID** | Simple, intuitive tuning, integral eliminates steady-state error | Can oscillate, requires manual tuning | Stationary targets, slow motion |
| **LQR** | Optimal for linearized system, systematic gain design | Assumes linear dynamics, no integral action | Moderate velocities, known dynamics |
| **Deep** | Can learn complex mappings, adapts to nonlinearities | Requires training data, black-box | Complex motion, nonlinear scenarios |

## Future Iteration Workflow

### Phase 1: Baseline Implementation (Current)
- Basic environment with linear target motion
- LQR controller implementation
- Validation of success criteria computation

### Phase 2: Controller Comparison
- Implement PID controller
- Run comparative experiments
- Analyze performance differences

### Phase 3: Advanced Scenarios
- Add non-linear target motion
- Introduce observation noise
- Test controller robustness

### Phase 4: Learning-Based Controllers
- Implement neural network controller
- Design training pipeline
- Compare learned vs. classical controllers

## Assumptions and Limitations

### Current Assumptions
- **Perfect state information**: No sensor noise
- **Smooth target motion**: Differentiable trajectories
- **Idealized quadcopter**: Simplified dynamics model
- **3D tracking**: Full spatial tracking problem

### Known Limitations
- No disturbance modeling (wind, etc.)
- Single target only
- No obstacle avoidance
- Deterministic environment (given seed)

### Classical Controller Limitations
- **PID**: Performance degrades for fast-moving targets; integral windup during large transients
- **LQR**: Linearization breaks down for large attitude angles (> 30°) or high velocities

## Development Guidelines

### Adding a New Controller
1. Create class inheriting from `BaseController`
2. Implement `compute_action()` method
3. Add to `__all__` in controllers package
4. Create experiment config for new controller

### Adding Target Motion Pattern
1. Add motion type to `TargetMotion` class
2. Implement `get_position()` for new pattern
3. Document parameters in `.env.example`
4. Add config validation

### Running Experiments
```bash
# Basic experiment
make run-experiment

# Custom configuration
make run-experiment CONFIG=configs/circular.yaml SEED=123

# With environment overrides
QUADCOPTER_SEED=456 make run-experiment
```

### Evaluating Classical Controllers
```bash
# Evaluate PID controller
python -m quadcopter_tracking.eval --controller pid --episodes 10

# Evaluate LQR controller
python -m quadcopter_tracking.eval --controller lqr --motion-type stationary
```

## Installation Notes

### System Dependencies
The project primarily uses pure Python packages. However:
- **PyTorch**: May require CUDA toolkit for GPU support
- **Matplotlib**: Requires display backend or headless configuration

### CPU Fallback
For GPU-less machines:
- PyTorch automatically falls back to CPU
- Set `CUDA_VISIBLE_DEVICES=""` to force CPU usage
- Performance will be reduced but functionality preserved

### Common Setup Issues
- Missing `libgl1-mesa-glx` on headless Linux: `apt install libgl1-mesa-glx`
- Matplotlib display issues: Set `MPLBACKEND=Agg` for headless rendering
