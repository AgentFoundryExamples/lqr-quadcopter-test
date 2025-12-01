"""
Quadcopter Tracking Environment Package

This package provides simulation environments for quadcopter target-tracking studies.
The environment models quadcopter dynamics, target motion, and observation/reward
mechanisms suitable for evaluating tracking controllers.

Coordinate Frame (ENU - East-North-Up):
    - X-axis: East (positive direction)
    - Y-axis: North (positive direction)
    - Z-axis: Up (positive direction, gravity acts in -Z)

Key Assumptions:
- Perfect target information (no sensor noise for initial experiments)
- Smooth target motion (differentiable trajectories)
- 3D state space with position and velocity

State Vector Layout:
    Position:     [x, y, z]           - meters (ENU frame)
    Velocity:     [vx, vy, vz]        - m/s (ENU frame)
    Attitude:     [phi, theta, psi]   - radians (roll, pitch, yaw)
    Angular rate: [p, q, r]           - rad/s

Action Space:
    thrust: Total thrust (N), clipped to [0, max_thrust]
    roll_rate: Desired roll rate (rad/s), clipped to [-max, max]
    pitch_rate: Desired pitch rate (rad/s), clipped to [-max, max]
    yaw_rate: Desired yaw rate (rad/s), clipped to [-max, max]

Sign Conventions (ENU):
    +pitch_rate → +X velocity (pitching nose up accelerates forward/east)
    +roll_rate → -Y velocity (rolling right accelerates left/south)
    +thrust → +Z acceleration (upward force)

Motion Patterns:
- linear: Constant velocity in random direction
- circular: Orbital motion in horizontal plane
- sinusoidal: Multi-axis sinusoidal oscillation
- figure8: Lemniscate (figure-8) trajectory
- stationary: Fixed position (hovering reference)

Future Extensions:
- Sensor noise modeling
- Partial observability
- Multi-target tracking
"""

from .config import (
    EnvConfig,
    LoggingParams,
    QuadcopterParams,
    SimulationParams,
    SuccessCriteria,
    TargetParams,
)
from .quadcopter_env import QuadcopterEnv
from .target_motion import (
    CircularMotion,
    Figure8Motion,
    LinearMotion,
    SinusoidalMotion,
    StationaryMotion,
    TargetMotion,
)

__all__ = [
    # Main classes
    "QuadcopterEnv",
    "TargetMotion",
    # Configuration
    "EnvConfig",
    "QuadcopterParams",
    "SimulationParams",
    "TargetParams",
    "SuccessCriteria",
    "LoggingParams",
    # Motion patterns
    "LinearMotion",
    "CircularMotion",
    "SinusoidalMotion",
    "Figure8Motion",
    "StationaryMotion",
]
