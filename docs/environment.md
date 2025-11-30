# Quadcopter Dynamics Environment

This document describes the quadcopter simulation environment for target-tracking studies.

## Overview

The environment provides a realistic quadcopter simulation with:

- **Quadcopter dynamics**: Forces, torques, actuator limits, and numerical integration
- **Target motion generation**: Smooth trajectories with multiple motion patterns
- **State observation interface**: Separate quadcopter and target state dictionaries
- **Action validation**: Graceful clipping with violation logging
- **Time series recording**: Complete history for analysis and plotting

## State Vector Layout

The quadcopter state consists of 12 variables:

| Index | Variable | Description | Units |
|-------|----------|-------------|-------|
| 0-2 | x, y, z | Position | meters |
| 3-5 | vx, vy, vz | Velocity | m/s |
| 6-8 | φ, θ, ψ | Attitude (roll, pitch, yaw) | radians |
| 9-11 | p, q, r | Angular velocity | rad/s |

### Coordinate System

- **World frame**: North-East-Down (NED) with Z pointing up
- **Body frame**: X forward, Y right, Z up
- **Euler angles**: ZYX convention (yaw-pitch-roll)

## Action Space

Actions control the quadcopter through:

| Action | Description | Bounds | Units |
|--------|-------------|--------|-------|
| thrust | Total thrust | [0, max_thrust] | N |
| roll_rate | Desired roll rate | [-max_rate, max_rate] | rad/s |
| pitch_rate | Desired pitch rate | [-max_rate, max_rate] | rad/s |
| yaw_rate | Desired yaw rate | [-max_rate, max_rate] | rad/s |

Actions can be provided as a dictionary:

```python
action = {
    "thrust": 10.0,
    "roll_rate": 0.1,
    "pitch_rate": -0.1,
    "yaw_rate": 0.0,
}
```

Or as a numpy array:

```python
action = np.array([10.0, 0.1, -0.1, 0.0])  # [thrust, roll, pitch, yaw]
```

### Action Validation

The environment validates all actions and handles violations gracefully:

- **Out-of-bounds values**: Clipped to valid range with warning logged
- **NaN/Inf values**: Replaced with zeros with warning logged
- **Violations recorded**: Available via `env.get_action_violations()`

## Target Motion

The target generates smooth trajectories with perfect information (position, velocity, acceleration) provided to the controller.

### Motion Types

| Type | Description | Parameters |
|------|-------------|------------|
| `linear` | Constant velocity motion | speed, direction |
| `circular` | Orbital motion in horizontal plane | radius, speed, center |
| `sinusoidal` | Multi-axis sinusoidal oscillation | amplitude, frequency |
| `figure8` | Lemniscate (figure-8) pattern | scale, speed |
| `stationary` | Fixed position (hover reference) | position |

### Target State

```python
target_state = {
    "position": np.array([x, y, z]),      # meters
    "velocity": np.array([vx, vy, vz]),   # m/s
    "acceleration": np.array([ax, ay, az])  # m/s^2 (clamped)
}
```

### Reproducibility

Target trajectories are deterministic given a seed:

```python
env.reset(seed=42)  # Same seed = same trajectory
```

## Environment API

### Initialization

```python
from quadcopter_tracking.env import QuadcopterEnv, EnvConfig

# Default configuration
env = QuadcopterEnv()

# Custom configuration dictionary
config = {
    "seed": 42,
    "episode_length": 30.0,
    "target": {"motion_type": "circular", "radius": 3.0},
}
env = QuadcopterEnv(config=config)

# Using EnvConfig dataclass
config = EnvConfig()
config.target.motion_type = "sinusoidal"
config.simulation.dt = 0.02
env = QuadcopterEnv(config=config)
```

### Reset

```python
observation = env.reset(seed=42)
# Returns initial observation dictionary
```

### Step

```python
observation, reward, done, info = env.step(action)
```

Returns:
- `observation`: Dict with `quadcopter`, `target`, and `time` keys
- `reward`: Negative tracking error (encourages proximity)
- `done`: True when episode terminates
- `info`: Additional metrics (tracking_error, on_target, etc.)

### Observation Structure

```python
observation = {
    "quadcopter": {
        "position": np.array([x, y, z]),
        "velocity": np.array([vx, vy, vz]),
        "attitude": np.array([roll, pitch, yaw]),
        "angular_velocity": np.array([p, q, r]),
    },
    "target": {
        "position": np.array([x, y, z]),
        "velocity": np.array([vx, vy, vz]),
        "acceleration": np.array([ax, ay, az]),
    },
    "time": float,
}
```

### Info Dictionary

```python
info = {
    "time": 10.5,               # Current simulation time
    "step": 1050,               # Step count
    "tracking_error": 0.3,      # Distance to target (m)
    "on_target": True,          # Within target radius?
    "on_target_ratio": 0.85,    # Fraction of time on target
    "action_violations": 0,     # Count of clipped actions
}
```

On episode termination:

```python
info["termination_reason"] = "time_limit"  # or "position_bounds"
info["episode_length"] = 30.0
info["success"] = True  # Met success criteria?
```

## Physical Parameters

### Default Quadcopter Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| mass | 1.0 kg | Quadcopter mass |
| gravity | 9.81 m/s² | Gravitational acceleration |
| max_thrust | 20.0 N | Maximum total thrust |
| max_angular_rate | 3.0 rad/s | Maximum angular rate |
| drag_coeff_linear | 0.1 | Linear drag coefficient |
| drag_coeff_angular | 0.01 | Angular drag coefficient |

### Simulation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| dt | 0.01 s | Simulation timestep |
| max_episode_time | 30.0 s | Maximum episode duration |
| integrator | "rk4" | Integration method |
| max_velocity | 50.0 m/s | Velocity clipping |
| max_position | 1000.0 m | Position bound |

## Numerical Integration

The environment supports two integration methods:

### Euler (First-order)

```python
config.simulation.integrator = "euler"
```

Simple but may be unstable with large timesteps.

### RK4 (Fourth-order Runge-Kutta)

```python
config.simulation.integrator = "rk4"
```

Default method. More accurate and stable for longer simulations.

### Stability Considerations

- **Timestep**: Use dt ≤ 0.01s for stable simulation
- **State constraints**: Velocities and attitudes are automatically clipped
- **NaN detection**: Episode terminates if numerical instability detected

## Controller Integration

Controllers receive observations and produce actions:

```python
class MyController:
    def compute_action(self, observation):
        quad = observation["quadcopter"]
        target = observation["target"]
        
        # Compute control based on error
        error = target["position"] - quad["position"]
        
        return {
            "thrust": ...,
            "roll_rate": ...,
            "pitch_rate": ...,
            "yaw_rate": ...,
        }
```

### Hover Action Helper

For testing and initialization:

```python
hover_action = env.hover_action(mass=1.0, gravity=9.81)
# Returns action that maintains hover at current position
```

## Success Criteria

An episode is successful when:

1. **Duration**: Episode runs for at least 30 seconds
2. **On-target ratio**: ≥80% of time within 0.5m of target

Access success evaluation:

```python
observation, reward, done, info = env.step(action)
if done:
    print(f"Success: {info['success']}")
    print(f"On-target ratio: {info['on_target_ratio']:.1%}")
```

## Data Recording

### Time Series History

```python
# After episode
history = env.get_history()

for entry in history:
    print(f"Time: {entry['time']:.2f}s")
    print(f"Position: {entry['quadcopter_position']}")
    print(f"Tracking error: {entry['tracking_error']:.3f}m")
```

### Action Violations

```python
violations = env.get_action_violations()
for v in violations:
    print(f"Step {v['step']}: {v['violations']}")
```

## Configuration

### Via Dictionary

```python
config = {
    "seed": 42,
    "episode_length": 60.0,
    "dt": 0.02,
    "quadcopter": {
        "mass": 1.5,
        "max_thrust": 30.0,
    },
    "target": {
        "motion_type": "circular",
        "radius": 5.0,
        "speed": 2.0,
    },
    "success_criteria": {
        "min_on_target_ratio": 0.9,
        "target_radius": 0.3,
    },
}
env = QuadcopterEnv(config=config)
```

### Via EnvConfig Dataclass

```python
from quadcopter_tracking.env import EnvConfig, SimulationParams, TargetParams

config = EnvConfig()
config.simulation = SimulationParams(dt=0.02, max_episode_time=60.0)
config.target = TargetParams(motion_type="figure8", amplitude=3.0)
env = QuadcopterEnv(config=config)
```

## Edge Cases

### Aggressive Target Motion

The target motion generator enforces `max_acceleration` to maintain smooth trajectories:

```python
config.target.max_acceleration = 5.0  # m/s^2 limit
```

### Invalid Actions

```python
# NaN values are replaced with zeros
action = np.array([float('nan'), 0, 0, 0])
obs, _, _, _ = env.step(action)  # Continues without crashing

# Extreme values are clipped
action = {"thrust": 1000}  # Clipped to max_thrust
```

### Long Episodes

For episodes longer than 30 seconds:

- RK4 integration maintains stability
- State constraints prevent unbounded growth
- Automatic termination if numerical issues detected

## Example Usage

### Basic Tracking Loop

```python
from quadcopter_tracking.env import QuadcopterEnv

env = QuadcopterEnv()
obs = env.reset(seed=42)

done = False
total_reward = 0

while not done:
    # Simple proportional controller
    quad_pos = obs["quadcopter"]["position"]
    target_pos = obs["target"]["position"]
    error = target_pos - quad_pos
    
    action = {
        "thrust": 9.81 + error[2] * 2.0,
        "roll_rate": error[1] * 0.5,
        "pitch_rate": -error[0] * 0.5,
        "yaw_rate": 0.0,
    }
    
    obs, reward, done, info = env.step(action)
    total_reward += reward

print(f"Episode complete!")
print(f"Duration: {info['episode_length']:.1f}s")
print(f"On-target ratio: {info['on_target_ratio']:.1%}")
print(f"Success: {info['success']}")
```

### Custom Configuration

```python
from quadcopter_tracking.env import QuadcopterEnv, EnvConfig

config = EnvConfig()
config.target.motion_type = "figure8"
config.simulation.max_episode_time = 60.0
config.success_criteria.min_on_target_ratio = 0.9

env = QuadcopterEnv(config=config)
obs = env.reset(seed=123)
```

## Module Reference

### Environment Classes

- `QuadcopterEnv`: Main simulation environment
- `TargetMotion`: Target trajectory generator
- `EnvConfig`: Configuration dataclass

### Motion Patterns

- `LinearMotion`: Constant velocity
- `CircularMotion`: Orbital motion
- `SinusoidalMotion`: Oscillatory motion
- `Figure8Motion`: Lemniscate pattern
- `StationaryMotion`: Fixed position

### Configuration Classes

- `QuadcopterParams`: Physical parameters
- `SimulationParams`: Integration settings
- `TargetParams`: Motion configuration
- `SuccessCriteria`: Success thresholds
- `LoggingParams`: Data recording settings

## CPU Execution

### Performance Expectations

The simulation environment runs entirely on CPU and does not require GPU resources. Typical performance:

- **Simulation speed**: Environment stepping runs at approximately 10,000+ steps/second on modern CPUs
- **Episode runtime**: A 30-second episode (3,000 steps at dt=0.01) completes in under 1 second
- **Memory usage**: Minimal memory footprint (~100MB including dependencies)

### GPU vs CPU for Training

For neural network controllers:

- **Training**: GPU recommended for faster convergence (10-50x speedup)
- **Inference/Evaluation**: CPU is sufficient for real-time control at dt=0.01s
- **Force CPU**: Set `CUDA_VISIBLE_DEVICES=""` environment variable

```bash
# Force CPU execution
export CUDA_VISIBLE_DEVICES=""
python -m quadcopter_tracking.train --epochs 10

# Or inline
CUDA_VISIBLE_DEVICES="" python -m quadcopter_tracking.eval --controller deep
```

### Headless Environments

For servers without display:

```bash
# Set matplotlib backend before running
export MPLBACKEND=Agg

# Install system dependencies if needed (Debian/Ubuntu)
apt install -y libgl1-mesa-glx
```
