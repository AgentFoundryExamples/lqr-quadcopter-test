"""
Quadcopter Controllers Package

This package provides tracking controllers for quadcopter target following.
Controllers receive environment observations and produce control actions
to minimize tracking error.

Controller Types:
- LQR (Linear Quadratic Regulator): Classic optimal control
- PID: Proportional-Integral-Derivative control
- Neural: ML-based deep learning controllers

Design Philosophy:
- Controllers are stateless transformations where possible
- Clear interface between observation input and action output
- Modular design for easy comparison and benchmarking
"""

import numpy as np

from .base import BaseController
from .deep_tracking_policy import DeepTrackingPolicy, PolicyNetwork

__all__ = [
    "BaseController",
    "LQRController",
    "PIDController",
    "DeepTrackingPolicy",
    "PolicyNetwork",
]


def _validate_observation(observation: dict) -> None:
    """
    Validate that observation contains required keys.

    Args:
        observation: Environment observation dictionary.

    Raises:
        KeyError: If required keys are missing.
    """
    required_keys = ["quadcopter", "target"]
    for key in required_keys:
        if key not in observation:
            raise KeyError(f"Observation missing required key: '{key}'")

    quad_keys = ["position", "velocity", "attitude", "angular_velocity"]
    for key in quad_keys:
        if key not in observation["quadcopter"]:
            raise KeyError(f"Observation['quadcopter'] missing required key: '{key}'")

    target_keys = ["position", "velocity"]
    for key in target_keys:
        if key not in observation["target"]:
            raise KeyError(f"Observation['target'] missing required key: '{key}'")


class PIDController(BaseController):
    """
    PID controller for quadcopter tracking.

    Classic proportional-integral-derivative control for position tracking.
    Maintains integral error state for steady-state error elimination.

    The controller computes position error (target - quadcopter), uses the
    derivative of position error (approximated by velocity error) for the D term,
    and accumulates integral error for the I term with windup prevention.

    Control mapping:
    - Z-axis position error maps to thrust adjustment
    - X-axis position error maps to pitch rate (pitching forward moves +X)
    - Y-axis position error maps to roll rate (rolling right moves +Y)
    - Yaw rate is set to zero (no heading tracking in this implementation)

    Attributes:
        kp_pos (ndarray): Proportional gains for position [x, y, z].
        ki_pos (ndarray): Integral gains for position [x, y, z].
        kd_pos (ndarray): Derivative gains for position [x, y, z].
        integral_error (ndarray): Accumulated integral error.
        integral_limit (float): Maximum integral error magnitude for windup prevention.
        max_thrust (float): Maximum thrust output.
        min_thrust (float): Minimum thrust output.
        max_rate (float): Maximum angular rate output.
        hover_thrust (float): Thrust required to hover (mass * gravity).
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize PID controller.

        Args:
            config: Configuration with PID gain parameters:
                - kp_pos: Proportional gains [x, y, z] or scalar
                  (default: [2.0, 2.0, 4.0])
                - ki_pos: Integral gains [x, y, z] or scalar
                  (default: [0.1, 0.1, 0.2])
                - kd_pos: Derivative gains [x, y, z] or scalar
                  (default: [1.5, 1.5, 2.0])
                - integral_limit: Max integral magnitude (default: 5.0)
                - max_thrust: Max thrust in N (default: 20.0)
                - min_thrust: Min thrust in N (default: 0.0)
                - max_rate: Max angular rate in rad/s (default: 3.0)
                - mass: Quadcopter mass in kg (default: 1.0)
                - gravity: Gravitational acceleration (default: 9.81)
        """
        super().__init__(name="pid", config=config)
        config = config or {}

        # Position PID gains - can be scalar or array
        kp = config.get("kp_pos", config.get("kp", [2.0, 2.0, 4.0]))
        ki = config.get("ki_pos", config.get("ki", [0.1, 0.1, 0.2]))
        kd = config.get("kd_pos", config.get("kd", [1.5, 1.5, 2.0]))

        # Convert scalars to arrays using helper
        def _ensure_array(value, size=3):
            if hasattr(value, "__len__"):
                return np.array(value)
            return np.array([value] * size)

        self.kp_pos = _ensure_array(kp)
        self.ki_pos = _ensure_array(ki)
        self.kd_pos = _ensure_array(kd)

        # Integral windup prevention
        self.integral_limit = config.get("integral_limit", 5.0)

        # Output limits
        self.max_thrust = config.get("max_thrust", 20.0)
        self.min_thrust = config.get("min_thrust", 0.0)
        self.max_rate = config.get("max_rate", 3.0)

        # Physical parameters for hover thrust calculation
        mass = config.get("mass", 1.0)
        gravity = config.get("gravity", 9.81)
        self.hover_thrust = mass * gravity

        # State variables
        self.integral_error = np.zeros(3)
        self._last_time: float | None = None

    def compute_action(self, observation: dict) -> dict:
        """
        Compute PID control action.

        Args:
            observation: Environment observation with quadcopter and target state.

        Returns:
            Action dictionary with thrust, roll_rate, pitch_rate, yaw_rate.

        Raises:
            KeyError: If observation is missing required keys.
        """
        _validate_observation(observation)

        quad = observation["quadcopter"]
        target = observation["target"]

        # Compute position error (target - current)
        quad_pos = np.array(quad["position"])
        target_pos = np.array(target["position"])
        pos_error = target_pos - quad_pos

        # Compute velocity error (for derivative term)
        quad_vel = np.array(quad["velocity"])
        target_vel = np.array(target["velocity"])
        vel_error = target_vel - quad_vel

        # Update integral error with windup clamping (scaled by dt)
        current_time = observation.get("time", 0.0)
        dt = (current_time - self._last_time) if self._last_time is not None else 0.0
        self._last_time = current_time

        if dt > 0:
            self.integral_error += pos_error * dt
            # Clamp each component independently
            self.integral_error = np.clip(
                self.integral_error, -self.integral_limit, self.integral_limit
            )

        # Compute PID terms
        p_term = self.kp_pos * pos_error
        i_term = self.ki_pos * self.integral_error
        d_term = self.kd_pos * vel_error

        # Total desired correction (scaled by gains, not true acceleration)
        desired_correction = p_term + i_term + d_term

        # Map to control outputs:
        # Z-axis: thrust = hover_thrust + z_correction (direct addition in Newtons)
        thrust = self.hover_thrust + desired_correction[2]
        thrust = float(np.clip(thrust, self.min_thrust, self.max_thrust))

        # X-axis: positive position error -> pitch forward (negative pitch rate)
        # Pitching nose down (negative pitch) accelerates forward (+X)
        pitch_rate = -desired_correction[0]
        pitch_rate = float(np.clip(pitch_rate, -self.max_rate, self.max_rate))

        # Y-axis: positive position error -> roll right (positive roll rate)
        # Rolling right (positive roll) accelerates right (+Y)
        roll_rate = desired_correction[1]
        roll_rate = float(np.clip(roll_rate, -self.max_rate, self.max_rate))

        # No yaw tracking in this basic implementation
        yaw_rate = 0.0

        return {
            "thrust": thrust,
            "roll_rate": roll_rate,
            "pitch_rate": pitch_rate,
            "yaw_rate": yaw_rate,
        }

    def reset(self) -> None:
        """Reset integral error and time state."""
        self.integral_error = np.zeros(3)
        self._last_time = None


class LQRController(BaseController):
    """
    Linear Quadratic Regulator controller for quadcopter tracking.

    Uses pre-computed feedback gains to map state errors to control actions.
    The LQR approach minimizes a quadratic cost function:
        J = integral(x'Qx + u'Ru) dt

    This implementation uses simplified linearized dynamics around hover.
    The controller is most effective for:
    - Small attitude angles (< 30 degrees)
    - Moderate velocities (< 5 m/s)
    - Stationary or slowly moving targets

    Operating Envelope:
    - Position errors up to ±10 meters
    - Velocity errors up to ±5 m/s
    - Attitude angles < 30 degrees from hover

    For aggressive maneuvers outside this envelope, the linearization
    assumptions break down and performance degrades.

    State vector (6 dimensions):
        [x_error, y_error, z_error, vx_error, vy_error, vz_error]

    Control vector (4 dimensions):
        [thrust, roll_rate, pitch_rate, yaw_rate]

    Attributes:
        K (ndarray): Pre-computed feedback gain matrix (4x6).
        max_thrust (float): Maximum thrust output.
        min_thrust (float): Minimum thrust output.
        max_rate (float): Maximum angular rate output.
        hover_thrust (float): Thrust required to hover.
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize LQR controller.

        Args:
            config: Configuration with LQR parameters:
                - K: Feedback gain matrix (4x6) or None to use defaults
                - q_pos: Position state cost weights [x, y, z] (default: [10, 10, 20])
                - q_vel: Velocity state cost weights [vx, vy, vz] (default: [5, 5, 10])
                - r_thrust: Thrust control cost (default: 0.1)
                - r_rate: Rate control cost (default: 1.0)
                - max_thrust: Max thrust in N (default: 20.0)
                - min_thrust: Min thrust in N (default: 0.0)
                - max_rate: Max angular rate in rad/s (default: 3.0)
                - mass: Quadcopter mass in kg (default: 1.0)
                - gravity: Gravitational acceleration (default: 9.81)
        """
        super().__init__(name="lqr", config=config)
        config = config or {}

        # Output limits
        self.max_thrust = config.get("max_thrust", 20.0)
        self.min_thrust = config.get("min_thrust", 0.0)
        self.max_rate = config.get("max_rate", 3.0)

        # Physical parameters
        mass = config.get("mass", 1.0)
        gravity = config.get("gravity", 9.81)
        self.hover_thrust = mass * gravity

        # Check for pre-defined K matrix
        if "K" in config and config["K"] is not None:
            self.K = np.array(config["K"])
            if self.K.shape != (4, 6):
                raise ValueError(
                    f"K matrix must have shape (4, 6), got {self.K.shape}"
                )
        else:
            # Compute feedback gains from Q and R weights
            self.K = self._compute_gains(config)

    def _compute_gains(self, config: dict) -> np.ndarray:
        """
        Compute LQR feedback gains from cost weights.

        For the simplified linearized hover dynamics, we use a diagonal
        approximation where each axis is treated independently.

        Args:
            config: Configuration dictionary with cost weights.

        Returns:
            Feedback gain matrix K (4x6).
        """
        # Cost weights
        q_pos = np.array(config.get("q_pos", [10.0, 10.0, 20.0]))
        q_vel = np.array(config.get("q_vel", [5.0, 5.0, 10.0]))
        r_thrust = config.get("r_thrust", 0.1)
        r_rate = config.get("r_rate", 1.0)

        # For each axis, we treat the dynamics as a double integrator:
        #   x'' = u (position, velocity, input)
        # The LQR optimal gains for a 1D double integrator are:
        #   K_pos = sqrt(q_pos / r)
        #   K_vel = sqrt(2*sqrt(q_pos/r) + q_vel/r)
        # We apply this formula per-axis to build the 4x6 gain matrix.

        # Initialize gain matrix: K maps [pos_err, vel_err] to [thrust, rates]
        K = np.zeros((4, 6))

        # Z-axis -> thrust (row 0)
        # Position error gain
        K[0, 2] = np.sqrt(q_pos[2] / r_thrust)
        # Velocity error gain
        K[0, 5] = np.sqrt(2 * np.sqrt(q_pos[2] / r_thrust) + q_vel[2] / r_thrust)

        # Y-axis -> roll rate (row 1)
        K[1, 1] = np.sqrt(q_pos[1] / r_rate)
        K[1, 4] = np.sqrt(2 * np.sqrt(q_pos[1] / r_rate) + q_vel[1] / r_rate)

        # X-axis -> pitch rate (row 2, note sign flip)
        K[2, 0] = -np.sqrt(q_pos[0] / r_rate)
        K[2, 3] = -np.sqrt(2 * np.sqrt(q_pos[0] / r_rate) + q_vel[0] / r_rate)

        # Yaw rate (row 3) - no yaw tracking
        # K[3, :] = 0 (already initialized to zeros)

        return K

    def compute_action(self, observation: dict) -> dict:
        """
        Compute LQR control action.

        Args:
            observation: Environment observation with quadcopter and target state.

        Returns:
            Action dictionary with thrust, roll_rate, pitch_rate, yaw_rate.

        Raises:
            KeyError: If observation is missing required keys.
        """
        _validate_observation(observation)

        quad = observation["quadcopter"]
        target = observation["target"]

        # Compute state error vector [pos_error, vel_error]
        quad_pos = np.array(quad["position"])
        target_pos = np.array(target["position"])
        pos_error = target_pos - quad_pos

        quad_vel = np.array(quad["velocity"])
        target_vel = np.array(target["velocity"])
        vel_error = target_vel - quad_vel

        # Build state error vector (6D)
        state_error = np.concatenate([pos_error, vel_error])

        # Compute control: u = K @ state_error
        u = self.K @ state_error

        # Extract and bound outputs
        # u[0] is thrust adjustment relative to hover
        thrust = self.hover_thrust + u[0]
        thrust = float(np.clip(thrust, self.min_thrust, self.max_thrust))

        roll_rate = float(np.clip(u[1], -self.max_rate, self.max_rate))
        pitch_rate = float(np.clip(u[2], -self.max_rate, self.max_rate))
        yaw_rate = float(np.clip(u[3], -self.max_rate, self.max_rate))

        return {
            "thrust": thrust,
            "roll_rate": roll_rate,
            "pitch_rate": pitch_rate,
            "yaw_rate": yaw_rate,
        }

    def reset(self) -> None:
        """Reset controller state (no-op for LQR as it's stateless)."""
        pass
