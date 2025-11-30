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

## Success Criteria

A tracking episode is considered **successful** when:

1. Episode runs for at least **30 seconds**
2. Quadcopter maintains position within **0.5 meters** of target for **≥80%** of the episode

This criteria ensures:
- Controllers handle sustained tracking (not just instantaneous)
- Some tolerance for transient errors (20% off-target allowed)
- Clear, measurable benchmark for comparison

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
