"""
Quadcopter Controllers Package

This package provides tracking controllers for quadcopter target following.
Controllers receive environment observations and produce control actions
to minimize tracking error.

Controller Types:
- LQR (Linear Quadratic Regulator): Classic optimal control with heuristic gains
- PID: Proportional-Integral-Derivative control
- Riccati-LQR: True LQR with DARE-solved optimal gains (requires scipy)
- Neural: ML-based deep learning controllers

Action Schema:
    All controllers must return an action dictionary with the following keys:
    - thrust: Total thrust in Newtons [0, max_thrust], default max: 20.0
    - roll_rate: Desired roll rate in rad/s [-max_rate, max_rate], default max: 3.0
    - pitch_rate: Desired pitch rate in rad/s [-max_rate, max_rate], default max: 3.0
    - yaw_rate: Desired yaw rate in rad/s [-max_rate, max_rate], default max: 3.0

Sign Conventions (ENU coordinate system):
    - +pitch_rate → +X velocity (pitching nose up accelerates forward/east)
    - +roll_rate → -Y velocity (rolling right accelerates left/south)
    - +thrust → +Z acceleration (upward force)

Design Philosophy:
- Controllers are stateless transformations where possible
- Clear interface between observation input and action output
- Modular design for easy comparison and benchmarking
- Consistent action schema across all controller types
"""

import numpy as np

from .base import (
    ACTION_KEYS,
    DEFAULT_ACTION_LIMITS,
    ActionLimits,
    BaseController,
    validate_action,
)
from .deep_tracking_policy import DeepTrackingPolicy, PolicyNetwork
from .riccati_lqr import RiccatiLQRController
from .tuning import (
    ControllerTuner,
    GainSearchSpace,
    TuningConfig,
    TuningResult,
)

__all__ = [
    "BaseController",
    "LQRController",
    "PIDController",
    "RiccatiLQRController",
    "DeepTrackingPolicy",
    "PolicyNetwork",
    "ACTION_KEYS",
    "ActionLimits",
    "DEFAULT_ACTION_LIMITS",
    "validate_action",
    "ControllerTuner",
    "GainSearchSpace",
    "TuningConfig",
    "TuningResult",
    "VALID_CONTROLLER_TYPES",
]

# Valid controller type names for config validation
# Used by train.py, eval.py, and other entry points
VALID_CONTROLLER_TYPES = ("deep", "lqr", "pid", "riccati_lqr")


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


def _ensure_array(value, size: int = 3) -> np.ndarray:
    """
    Convert a scalar or sequence to a numpy array.

    Args:
        value: Scalar or sequence of values.
        size: Expected size of the output array.

    Returns:
        Numpy array of the specified size.
    """
    if hasattr(value, "__len__"):
        return np.array(value)
    return np.array([value] * size)


class PIDController(BaseController):
    """
    PID controller for quadcopter tracking with optional feedforward support.

    Classic proportional-integral-derivative control for position tracking.
    Maintains integral error state for steady-state error elimination.
    Optionally incorporates target velocity and acceleration feedforward
    terms for improved tracking of moving targets.

    The controller computes position error (target - quadcopter), uses the
    derivative of position error (approximated by velocity error) for the D term,
    and accumulates integral error for the I term with windup prevention.

    Control mapping:
    - Z-axis position error maps to thrust adjustment
    - X-axis position error maps to pitch rate (pitching forward moves +X)
    - Y-axis position error maps to roll rate (rolling right moves +Y)
    - Yaw rate is set to zero (no heading tracking in this implementation)

    Feedforward (optional):
    - Velocity feedforward: Scales target velocity to anticipate motion
    - Acceleration feedforward: Scales target acceleration for dynamic tracking
    - Both default to disabled (gains = 0) to preserve baseline behavior

    Attributes:
        kp_pos (ndarray): Proportional gains for position [x, y, z].
        ki_pos (ndarray): Integral gains for position [x, y, z].
        kd_pos (ndarray): Derivative gains for position [x, y, z].
        ff_velocity_gain (ndarray): Feedforward gains for target velocity.
        ff_acceleration_gain (ndarray): Feedforward gains for target acceleration.
        feedforward_enabled (bool): Whether feedforward is enabled.
        ff_max_velocity (float): Max velocity for feedforward clamping.
        ff_max_acceleration (float): Max accel for feedforward clamping.
        integral_error (ndarray): Accumulated integral error.
        integral_limit (float): Max integral magnitude for windup prevention.
        max_thrust (float): Maximum thrust output.
        min_thrust (float): Minimum thrust output.
        max_rate (float): Maximum angular rate output.
        hover_thrust (float): Thrust required to hover (mass * gravity).
        last_control_components (dict | None): P/I/D/FF terms for diagnostics.
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize PID controller.

        The default gains are tuned for stable tracking across stationary, linear,
        and circular target scenarios. XY gains are intentionally much smaller than
        Z gains because position errors (in meters) are mapped to angular rates
        (in rad/s). A 1-meter XY error with kp=0.01 produces only 0.01 rad/s of
        pitch/roll rate, preventing actuator saturation while still converging.

        Feedforward is disabled by default (gains = 0) to preserve baseline
        behavior. Enable by setting feedforward_enabled=True and nonzero gains.

        Args:
            config: Configuration with PID gain parameters:
                - kp_pos: Proportional gains [x, y, z] or scalar
                  (default: [0.01, 0.01, 4.0])
                - ki_pos: Integral gains [x, y, z] or scalar
                  (default: [0.0, 0.0, 0.0])
                - kd_pos: Derivative gains [x, y, z] or scalar
                  (default: [0.06, 0.06, 2.0])
                - integral_limit: Max integral magnitude (default: 0.0)
                - ff_velocity_gain: Feedforward gains for target velocity
                  [x, y, z] or scalar (default: [0.0, 0.0, 0.0] - disabled)
                - ff_acceleration_gain: Feedforward gains for target accel
                  [x, y, z] or scalar (default: [0.0, 0.0, 0.0] - disabled)
                - feedforward_enabled: Master toggle for FF (default: False)
                - ff_max_velocity: Max velocity for clamping (default: 10.0)
                - ff_max_acceleration: Max accel for clamping (default: 5.0)
                - max_thrust: Max thrust in N (default: 20.0)
                - min_thrust: Min thrust in N (default: 0.0)
                - max_rate: Max angular rate in rad/s (default: 3.0)
                - mass: Quadcopter mass in kg (default: 1.0)
                - gravity: Gravitational acceleration (default: 9.81)
        """
        config = config or {}

        # Physical parameters for hover thrust calculation
        mass = config.get("mass", 1.0)
        gravity = config.get("gravity", 9.81)

        super().__init__(name="pid", config=config, mass=mass, gravity=gravity)

        # Position PID gains - can be scalar or array
        # XY gains are small because meter→rad/s mapping would otherwise saturate
        kp = config.get("kp_pos", config.get("kp", [0.01, 0.01, 4.0]))
        ki = config.get("ki_pos", config.get("ki", [0.0, 0.0, 0.0]))
        kd = config.get("kd_pos", config.get("kd", [0.06, 0.06, 2.0]))

        # Convert scalars to arrays using module-level helper
        self.kp_pos = _ensure_array(kp)
        self.ki_pos = _ensure_array(ki)
        self.kd_pos = _ensure_array(kd)

        # Feedforward configuration
        # Default to disabled (gains = 0) to preserve baseline behavior
        self.feedforward_enabled = config.get("feedforward_enabled", False)
        ff_vel = config.get("ff_velocity_gain", [0.0, 0.0, 0.0])
        ff_acc = config.get("ff_acceleration_gain", [0.0, 0.0, 0.0])
        self.ff_velocity_gain = _ensure_array(ff_vel)
        self.ff_acceleration_gain = _ensure_array(ff_acc)

        # Feedforward clamping limits (to prevent oscillation from noisy inputs)
        self.ff_max_velocity = config.get("ff_max_velocity", 10.0)
        self.ff_max_acceleration = config.get("ff_max_acceleration", 5.0)

        # Integral windup prevention
        # Default 0.0 for XY to avoid windup; users can tune for bias rejection
        self.integral_limit = config.get("integral_limit", 0.0)

        # Output limits
        self.max_thrust = config.get("max_thrust", 20.0)
        self.min_thrust = config.get("min_thrust", 0.0)
        self.max_rate = config.get("max_rate", 3.0)

        # Hover thrust from base class params
        self.hover_thrust = self.mass * self.gravity

        # State variables
        self.integral_error = np.zeros(3)
        self._last_time: float | None = None

        # Diagnostics: store last computed control components for logging/plotting
        self.last_control_components: dict | None = None

    def compute_action(self, observation: dict) -> dict:
        """
        Compute PID control action with optional feedforward.

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

        # Compute feedforward terms (if enabled)
        ff_vel_term = np.zeros(3)
        ff_acc_term = np.zeros(3)

        if self.feedforward_enabled:
            # Velocity feedforward: scale target velocity
            ff_target_vel = target_vel.copy()
            # Clamp velocity magnitude to prevent oscillation from noisy inputs
            vel_mag = np.linalg.norm(ff_target_vel)
            if vel_mag > self.ff_max_velocity and vel_mag > 0:
                ff_target_vel = ff_target_vel / vel_mag * self.ff_max_velocity
            ff_vel_term = self.ff_velocity_gain * ff_target_vel

            # Acceleration feedforward: scale target acceleration (if available)
            target_acc = target.get("acceleration", None)
            if target_acc is not None:
                ff_target_acc = np.array(target_acc)
                # Clamp acceleration magnitude to prevent oscillation
                acc_mag = np.linalg.norm(ff_target_acc)
                if acc_mag > self.ff_max_acceleration and acc_mag > 0:
                    ff_target_acc = ff_target_acc / acc_mag * self.ff_max_acceleration
                ff_acc_term = self.ff_acceleration_gain * ff_target_acc

        # Total desired correction (scaled by gains, not true acceleration)
        # Combine P + I + D + FF_velocity + FF_acceleration
        desired_correction = p_term + i_term + d_term + ff_vel_term + ff_acc_term

        # Store control components for diagnostics
        self.last_control_components = {
            "p_term": p_term.copy(),
            "i_term": i_term.copy(),
            "d_term": d_term.copy(),
            "ff_velocity_term": ff_vel_term.copy(),
            "ff_acceleration_term": ff_acc_term.copy(),
            "total_correction": desired_correction.copy(),
        }

        # Map to control outputs:
        # Z-axis: thrust = hover_thrust + z_correction (direct addition in Newtons)
        thrust = self.hover_thrust + desired_correction[2]
        thrust = float(np.clip(thrust, self.min_thrust, self.max_thrust))

        # X-axis: positive position error -> positive pitch rate
        # Environment dynamics: +pitch_rate produces +X velocity
        pitch_rate = desired_correction[0]
        pitch_rate = float(np.clip(pitch_rate, -self.max_rate, self.max_rate))

        # Y-axis: positive position error -> negative roll rate
        # Environment dynamics: +roll_rate produces -Y velocity
        roll_rate = -desired_correction[1]
        roll_rate = float(np.clip(roll_rate, -self.max_rate, self.max_rate))

        # No yaw tracking in this basic implementation
        yaw_rate = 0.0

        return {
            "thrust": thrust,
            "roll_rate": roll_rate,
            "pitch_rate": pitch_rate,
            "yaw_rate": yaw_rate,
        }

    def get_control_components(self) -> dict | None:
        """
        Get the last computed control term components for diagnostics.

        Returns:
            Dictionary with P, I, D, FF_velocity, FF_acceleration terms,
            or None if compute_action hasn't been called yet.
        """
        return self.last_control_components

    def reset(self) -> None:
        """Reset integral error, time state, and diagnostics."""
        self.integral_error = np.zeros(3)
        self._last_time = None
        self.last_control_components = None


class LQRController(BaseController):
    """
    Linear Quadratic Regulator controller for quadcopter tracking with
    optional feedforward support.

    Uses pre-computed feedback gains to map state errors to control actions.
    The LQR approach minimizes a quadratic cost function:
        J = integral(x'Qx + u'Ru) dt

    Optionally incorporates target velocity and acceleration feedforward
    terms for improved tracking of moving targets.

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

    Feedforward (optional):
    - Velocity feedforward: Scales target velocity to anticipate motion
    - Acceleration feedforward: Scales target acceleration for dynamic tracking
    - Both default to disabled (gains = 0) to preserve baseline behavior

    Attributes:
        K (ndarray): Pre-computed feedback gain matrix (4x6).
        ff_velocity_gain (ndarray): Feedforward gains for target velocity.
        ff_acceleration_gain (ndarray): Feedforward gains for target accel.
        feedforward_enabled (bool): Whether feedforward is enabled.
        ff_max_velocity (float): Max velocity for feedforward clamping.
        ff_max_acceleration (float): Max accel for feedforward clamping.
        max_thrust (float): Maximum thrust output.
        min_thrust (float): Minimum thrust output.
        max_rate (float): Maximum angular rate output.
        hover_thrust (float): Thrust required to hover.
        last_control_components (dict | None): Feedback/FF terms for diagnostics.
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize LQR controller.

        The default cost weights are tuned to produce feedback gains consistent
        with the validated PID defaults. XY position cost weights are much smaller
        than Z because the resulting gains map meter errors to rad/s rates, and
        large gains would saturate the actuators. The r_rate cost is higher to
        further reduce XY aggressiveness.

        Feedforward is disabled by default (gains = 0) to preserve baseline
        behavior. Enable by setting feedforward_enabled=True and nonzero gains.

        Args:
            config: Configuration with LQR parameters:
                - K: Feedback gain matrix (4x6) or None to use defaults
                - q_pos: Position cost [x, y, z] (default: [0.0001, 0.0001, 16.0])
                - q_vel: Velocity cost [vx, vy, vz] (default: [0.0036, 0.0036, 4.0])
                - r_thrust: Thrust control cost (default: 1.0)
                - r_rate: Rate control cost (default: 1.0)
                - ff_velocity_gain: Feedforward gains for target velocity
                  [x, y, z] or scalar (default: [0.0, 0.0, 0.0] - disabled)
                - ff_acceleration_gain: Feedforward gains for target accel
                  [x, y, z] or scalar (default: [0.0, 0.0, 0.0] - disabled)
                - feedforward_enabled: Master toggle for FF (default: False)
                - ff_max_velocity: Max velocity for clamping (default: 10.0)
                - ff_max_acceleration: Max accel for clamping (default: 5.0)
                - max_thrust: Max thrust in N (default: 20.0)
                - min_thrust: Min thrust in N (default: 0.0)
                - max_rate: Max angular rate in rad/s (default: 3.0)
                - mass: Quadcopter mass in kg (default: 1.0)
                - gravity: Gravitational acceleration (default: 9.81)
        """
        config = config or {}

        # Physical parameters
        mass = config.get("mass", 1.0)
        gravity = config.get("gravity", 9.81)

        super().__init__(name="lqr", config=config, mass=mass, gravity=gravity)

        # Output limits
        self.max_thrust = config.get("max_thrust", 20.0)
        self.min_thrust = config.get("min_thrust", 0.0)
        self.max_rate = config.get("max_rate", 3.0)

        # Hover thrust from base class params
        self.hover_thrust = self.mass * self.gravity

        # Feedforward configuration
        # Default to disabled (gains = 0) to preserve baseline behavior
        self.feedforward_enabled = config.get("feedforward_enabled", False)
        ff_vel = config.get("ff_velocity_gain", [0.0, 0.0, 0.0])
        ff_acc = config.get("ff_acceleration_gain", [0.0, 0.0, 0.0])
        self.ff_velocity_gain = _ensure_array(ff_vel)
        self.ff_acceleration_gain = _ensure_array(ff_acc)

        # Feedforward clamping limits (to prevent oscillation from noisy inputs)
        self.ff_max_velocity = config.get("ff_max_velocity", 10.0)
        self.ff_max_acceleration = config.get("ff_max_acceleration", 5.0)

        # Diagnostics: store last computed control components for logging/plotting
        self.last_control_components: dict | None = None

        # Check for pre-defined K matrix
        if "K" in config and config["K"] is not None:
            self.K = np.array(config["K"])
            if self.K.shape != (4, 6):
                raise ValueError(f"K matrix must have shape (4, 6), got {self.K.shape}")
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
        # XY position costs are small to avoid saturation (meter→rad/s mapping)
        # Z position cost is higher for tight altitude tracking
        q_pos = np.array(config.get("q_pos", [0.0001, 0.0001, 16.0]))
        q_vel = np.array(config.get("q_vel", [0.0036, 0.0036, 4.0]))
        r_thrust = config.get("r_thrust", 1.0)
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
        # Environment dynamics: +roll_rate produces -Y velocity
        # Therefore: +Y error needs -roll_rate, so gains are negative
        K[1, 1] = -np.sqrt(q_pos[1] / r_rate)
        K[1, 4] = -np.sqrt(2 * np.sqrt(q_pos[1] / r_rate) + q_vel[1] / r_rate)

        # X-axis -> pitch rate (row 2)
        # Environment dynamics: +pitch_rate produces +X velocity
        # Therefore: +X error needs +pitch_rate, so gains are positive
        K[2, 0] = np.sqrt(q_pos[0] / r_rate)
        K[2, 3] = np.sqrt(2 * np.sqrt(q_pos[0] / r_rate) + q_vel[0] / r_rate)

        # Yaw rate (row 3) - no yaw tracking
        # K[3, :] = 0 (already initialized to zeros)

        return K

    def compute_action(self, observation: dict) -> dict:
        """
        Compute LQR control action with optional feedforward.

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

        # Compute feedback control: u = K @ state_error
        u_feedback = self.K @ state_error

        # Compute feedforward terms (if enabled)
        ff_vel_term = np.zeros(3)
        ff_acc_term = np.zeros(3)

        if self.feedforward_enabled:
            # Velocity feedforward: scale target velocity
            ff_target_vel = target_vel.copy()
            # Clamp velocity magnitude to prevent oscillation from noisy inputs
            vel_mag = np.linalg.norm(ff_target_vel)
            if vel_mag > self.ff_max_velocity and vel_mag > 0:
                ff_target_vel = ff_target_vel / vel_mag * self.ff_max_velocity
            ff_vel_term = self.ff_velocity_gain * ff_target_vel

            # Acceleration feedforward: scale target acceleration (if available)
            target_acc = target.get("acceleration", None)
            if target_acc is not None:
                ff_target_acc = np.array(target_acc)
                # Clamp acceleration magnitude to prevent oscillation
                acc_mag = np.linalg.norm(ff_target_acc)
                if acc_mag > self.ff_max_acceleration and acc_mag > 0:
                    ff_target_acc = ff_target_acc / acc_mag * self.ff_max_acceleration
                ff_acc_term = self.ff_acceleration_gain * ff_target_acc

        # Store control components for diagnostics
        self.last_control_components = {
            "feedback_u": u_feedback.copy(),
            "ff_velocity_term": ff_vel_term.copy(),
            "ff_acceleration_term": ff_acc_term.copy(),
        }

        # Combine feedback with feedforward
        # Map feedforward to control outputs same way as PID:
        # Z -> thrust, X -> pitch, Y -> -roll
        ff_thrust = ff_vel_term[2] + ff_acc_term[2]
        ff_pitch = ff_vel_term[0] + ff_acc_term[0]
        ff_roll = -(ff_vel_term[1] + ff_acc_term[1])

        # Extract and bound outputs
        # u_feedback[0] is thrust adjustment relative to hover
        thrust = self.hover_thrust + u_feedback[0] + ff_thrust
        thrust = float(np.clip(thrust, self.min_thrust, self.max_thrust))

        roll_rate = float(
            np.clip(u_feedback[1] + ff_roll, -self.max_rate, self.max_rate)
        )
        pitch_rate = float(
            np.clip(u_feedback[2] + ff_pitch, -self.max_rate, self.max_rate)
        )
        yaw_rate = float(np.clip(u_feedback[3], -self.max_rate, self.max_rate))

        return {
            "thrust": thrust,
            "roll_rate": roll_rate,
            "pitch_rate": pitch_rate,
            "yaw_rate": yaw_rate,
        }

    def get_control_components(self) -> dict | None:
        """
        Get the last computed control term components for diagnostics.

        Returns:
            Dictionary with feedback and FF terms,
            or None if compute_action hasn't been called yet.
        """
        return self.last_control_components

    def reset(self) -> None:
        """Reset controller state (no-op for LQR as it's mostly stateless)."""
        self.last_control_components = None
