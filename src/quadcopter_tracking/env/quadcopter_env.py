"""
Quadcopter Dynamics Environment

Provides a simulation environment for quadcopter target tracking with:
- Realistic physics (forces, torques, actuator limits)
- Numerical integration (Euler, RK4)
- Target motion generation
- Action validation and clipping
- Time series recording

State Vector Layout:
    Position:     [x, y, z]           - meters
    Velocity:     [vx, vy, vz]        - m/s
    Attitude:     [phi, theta, psi]   - radians (roll, pitch, yaw)
    Angular rate: [p, q, r]           - rad/s

Action Space:
    thrust: Total thrust (N), clipped to [0, max_thrust]
    roll_rate: Desired roll rate (rad/s), clipped to [-max, max]
    pitch_rate: Desired pitch rate (rad/s), clipped to [-max, max]
    yaw_rate: Desired yaw rate (rad/s), clipped to [-max, max]
"""

import logging
import math

import numpy as np

from .config import EnvConfig
from .target_motion import TargetMotion

logger = logging.getLogger(__name__)


class QuadcopterEnv:
    """
    Simulation environment for quadcopter target tracking.

    This environment provides:
    - Quadcopter dynamics modeling with RK4 integration
    - Target trajectory generation
    - State observation interface
    - Reward computation for tracking performance
    - Action validation with graceful clipping

    Attributes:
        config: Environment configuration.
        state: Current quadcopter state dictionary.
        target: Target motion generator.
        time: Current simulation time.
    """

    # State vector indices
    POS_X, POS_Y, POS_Z = 0, 1, 2
    VEL_X, VEL_Y, VEL_Z = 3, 4, 5
    ROLL, PITCH, YAW = 6, 7, 8
    ROLL_RATE, PITCH_RATE, YAW_RATE = 9, 10, 11

    STATE_DIM = 12
    ACTION_DIM = 4

    def __init__(self, config: dict | EnvConfig | None = None):
        """
        Initialize the quadcopter tracking environment.

        Args:
            config: Configuration dictionary or EnvConfig instance.
                   If dict, will be converted via EnvConfig.from_dict().
                   If None, default configuration is used.
        """
        if config is None:
            self.config = EnvConfig()
        elif isinstance(config, dict):
            self.config = EnvConfig.from_dict(config)
        else:
            self.config = config

        # Initialize target motion generator
        self.target = TargetMotion(
            params=self.config.target,
            seed=self.config.seed,
        )

        # State variables
        self._state_vector = np.zeros(self.STATE_DIM)
        self._time = 0.0
        self._step_count = 0
        self._initialized = False

        # Recording for time series
        self._history: list[dict] = []
        self._action_violations: list[dict] = []

        # Tracking metrics
        self._on_target_count = 0
        self._total_steps = 0

        # Random number generator
        self._rng = np.random.default_rng(self.config.seed)

    def reset(self, seed: int | None = None) -> dict:
        """
        Reset the environment to initial state.

        Args:
            seed: Optional random seed for reproducibility.

        Returns:
            Initial observation dictionary containing quadcopter
            state and target state.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.target = TargetMotion(
                params=self.config.target,
                seed=seed,
            )

        # Reset target
        self.target.reset(seed=seed)

        # Initialize quadcopter state near target initial position
        target_state = self.target.get_state(0.0)
        initial_pos = target_state["position"]

        # Small random offset from target
        offset = self._rng.uniform(-0.5, 0.5, 3)
        self._state_vector = np.zeros(self.STATE_DIM)
        self._state_vector[self.POS_X:self.POS_Z + 1] = initial_pos + offset

        # Reset time and counters
        self._time = 0.0
        self._step_count = 0
        self._on_target_count = 0
        self._total_steps = 0
        self._history = []
        self._action_violations = []
        self._initialized = True

        return self._get_observation()

    def step(self, action: dict | np.ndarray) -> tuple[dict, float, bool, dict]:
        """
        Execute one environment step.

        Args:
            action: Control action for the quadcopter.
                   If dict: {thrust, roll_rate, pitch_rate, yaw_rate}
                   If ndarray: [thrust, roll_rate, pitch_rate, yaw_rate]

        Returns:
            Tuple of (observation, reward, done, info).
            - observation: State dict with quadcopter and target info
            - reward: Tracking reward (negative distance)
            - done: Whether episode has ended
            - info: Additional information dict
        """
        if not self._initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Parse and validate action
        action_vec, violations = self._parse_and_validate_action(action)

        if violations:
            self._action_violations.append({
                "step": self._step_count,
                "time": self._time,
                "violations": violations,
            })

        # Integrate dynamics
        dt = self.config.simulation.dt
        self._state_vector = self._integrate(self._state_vector, action_vec, dt)

        # Apply state constraints
        self._state_vector = self._apply_state_constraints(self._state_vector)

        # Update time
        self._time += dt
        self._step_count += 1
        self._total_steps += 1

        # Get observation
        observation = self._get_observation()

        # Compute reward
        reward = self._compute_reward(observation)

        # Check termination
        done, termination_reason = self._check_termination()

        # Track on-target time
        tracking_error = self._compute_tracking_error(observation)
        if tracking_error <= self.config.success_criteria.target_radius:
            self._on_target_count += 1

        # Build info dict
        info = {
            "time": self._time,
            "step": self._step_count,
            "tracking_error": tracking_error,
            "on_target": tracking_error <= self.config.success_criteria.target_radius,
            "on_target_ratio": (
                self._on_target_count / self._total_steps
                if self._total_steps > 0 else 0.0
            ),
            "action_violations": len(self._action_violations),
        }

        if done:
            info["termination_reason"] = termination_reason
            info["episode_length"] = self._time
            info["success"] = self._evaluate_success()

        # Record history
        if self.config.logging.enabled:
            self._record_step(observation, action_vec, reward, info)

        return observation, reward, done, info

    def _parse_and_validate_action(
        self, action: dict | np.ndarray
    ) -> tuple[np.ndarray, list[str]]:
        """
        Parse action input and validate/clip to bounds.

        Args:
            action: Input action (dict or array).

        Returns:
            Tuple of (action_vector, list of violations).
        """
        violations = []

        # Parse action format
        if isinstance(action, dict):
            action_vec = np.array([
                action.get("thrust", 0.0),
                action.get("roll_rate", 0.0),
                action.get("pitch_rate", 0.0),
                action.get("yaw_rate", 0.0),
            ])
        else:
            action_vec = np.asarray(action, dtype=np.float64)
            if action_vec.shape != (4,):
                raise ValueError(
                    f"Action array must have shape (4,), got {action_vec.shape}"
                )

        # Check for NaN/Inf
        if not np.all(np.isfinite(action_vec)):
            violations.append("Action contains NaN or Inf values")
            logger.warning("Action contains NaN or Inf, replacing with zeros")
            action_vec = np.nan_to_num(action_vec, nan=0.0, posinf=0.0, neginf=0.0)

        # Validate and clip thrust
        thrust = action_vec[0]
        min_thrust = self.config.quadcopter.min_thrust
        max_thrust = self.config.quadcopter.max_thrust

        if thrust < min_thrust:
            violations.append(f"Thrust {thrust:.2f} below min {min_thrust:.2f}")
            action_vec[0] = min_thrust
        elif thrust > max_thrust:
            violations.append(f"Thrust {thrust:.2f} above max {max_thrust:.2f}")
            action_vec[0] = max_thrust

        # Validate and clip angular rates
        max_rate = self.config.quadcopter.max_angular_rate
        rate_names = ["roll_rate", "pitch_rate", "yaw_rate"]

        for i, name in enumerate(rate_names, start=1):
            rate = action_vec[i]
            if abs(rate) > max_rate:
                violations.append(
                    f"{name} {rate:.2f} exceeds limit {max_rate:.2f}"
                )
                action_vec[i] = np.clip(rate, -max_rate, max_rate)

        return action_vec, violations

    def _integrate(
        self, state: np.ndarray, action: np.ndarray, dt: float
    ) -> np.ndarray:
        """
        Integrate dynamics forward by dt.

        Args:
            state: Current state vector.
            action: Action vector [thrust, roll_rate, pitch_rate, yaw_rate].
            dt: Time step.

        Returns:
            New state vector.
        """
        if self.config.simulation.integrator == "euler":
            return self._euler_step(state, action, dt)
        else:  # RK4
            return self._rk4_step(state, action, dt)

    def _euler_step(
        self, state: np.ndarray, action: np.ndarray, dt: float
    ) -> np.ndarray:
        """Euler integration step."""
        deriv = self._compute_derivatives(state, action)
        return state + deriv * dt

    def _rk4_step(
        self, state: np.ndarray, action: np.ndarray, dt: float
    ) -> np.ndarray:
        """4th-order Runge-Kutta integration step."""
        k1 = self._compute_derivatives(state, action)
        k2 = self._compute_derivatives(state + 0.5 * dt * k1, action)
        k3 = self._compute_derivatives(state + 0.5 * dt * k2, action)
        k4 = self._compute_derivatives(state + dt * k3, action)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _compute_derivatives(
        self, state: np.ndarray, action: np.ndarray
    ) -> np.ndarray:
        """
        Compute state derivatives based on quadcopter dynamics.

        The quadcopter model uses:
        - Position derivatives = velocity
        - Velocity derivatives from forces (thrust, gravity, drag)
        - Attitude derivatives = angular rates (simplified)
        - Angular rate derivatives from torque model (rate control)

        Args:
            state: Current state vector.
            action: Action vector [thrust, roll_rate, pitch_rate, yaw_rate].

        Returns:
            State derivative vector.
        """
        # Unpack state
        vel = state[self.VEL_X:self.VEL_Z + 1]
        attitude = state[self.ROLL:self.YAW + 1]
        ang_vel = state[self.ROLL_RATE:self.YAW_RATE + 1]

        phi, theta, psi = attitude  # roll, pitch, yaw
        thrust = action[0]
        desired_rates = action[1:4]

        # Physical parameters
        mass = self.config.quadcopter.mass
        gravity = self.config.quadcopter.gravity
        drag_linear = self.config.quadcopter.drag_coeff_linear
        drag_angular = self.config.quadcopter.drag_coeff_angular

        # Initialize derivative vector
        deriv = np.zeros(self.STATE_DIM)

        # Position derivative = velocity
        deriv[self.POS_X:self.POS_Z + 1] = vel

        # Compute rotation matrix from body to world frame
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        c_theta, s_theta = np.cos(theta), np.sin(theta)
        c_psi, s_psi = np.cos(psi), np.sin(psi)

        # Thrust vector in body frame points up (z-axis)
        # Transform to world frame
        thrust_body = np.array([0, 0, thrust])

        # Rotation matrix (ZYX Euler angles)
        R = np.array([
            [c_psi * c_theta,
             c_psi * s_theta * s_phi - s_psi * c_phi,
             c_psi * s_theta * c_phi + s_psi * s_phi],
            [s_psi * c_theta,
             s_psi * s_theta * s_phi + c_psi * c_phi,
             s_psi * s_theta * c_phi - c_psi * s_phi],
            [-s_theta,
             c_theta * s_phi,
             c_theta * c_phi],
        ])

        thrust_world = R @ thrust_body

        # Gravity force
        gravity_force = np.array([0, 0, -mass * gravity])

        # Linear drag force
        drag_force = -drag_linear * vel

        # Total acceleration
        total_force = thrust_world + gravity_force + drag_force
        accel = total_force / mass

        deriv[self.VEL_X:self.VEL_Z + 1] = accel

        # Attitude derivative (using body rates)
        # Simplification: Direct assignment of angular velocity to Euler rate.
        # This is accurate for small attitude angles. For large angles, a proper
        # kinematic transformation (involving sin/cos of angles) would be needed.
        # The attitude clipping in _apply_state_constraints() helps keep angles
        # within a range where this approximation remains reasonable.
        deriv[self.ROLL:self.YAW + 1] = ang_vel

        # Angular rate derivative (simple rate control model)
        # Model angular acceleration as proportional to rate error
        rate_time_constant = 0.1  # seconds
        rate_error = desired_rates - ang_vel
        ang_accel = rate_error / rate_time_constant

        # Add angular drag
        ang_accel -= drag_angular * ang_vel

        deriv[self.ROLL_RATE:self.YAW_RATE + 1] = ang_accel

        return deriv

    def _apply_state_constraints(self, state: np.ndarray) -> np.ndarray:
        """
        Apply state constraints to ensure numerical stability.

        Args:
            state: State vector.

        Returns:
            Constrained state vector.
        """
        result = state.copy()

        # Clip velocities
        max_vel = self.config.simulation.max_velocity
        vel = result[self.VEL_X:self.VEL_Z + 1]
        vel_mag = np.linalg.norm(vel)
        if vel_mag > max_vel:
            result[self.VEL_X:self.VEL_Z + 1] = vel / vel_mag * max_vel

        # Clip angular velocities
        max_ang_vel = self.config.simulation.max_angular_velocity
        result[self.ROLL_RATE:self.YAW_RATE + 1] = np.clip(
            result[self.ROLL_RATE:self.YAW_RATE + 1],
            -max_ang_vel,
            max_ang_vel,
        )

        # Normalize angles to [-pi, pi]
        for i in range(self.ROLL, self.YAW + 1):
            result[i] = self._normalize_angle(result[i])

        # Clip attitude to prevent extreme orientations
        max_tilt = math.pi / 3  # 60 degrees
        result[self.ROLL] = np.clip(result[self.ROLL], -max_tilt, max_tilt)
        result[self.PITCH] = np.clip(result[self.PITCH], -max_tilt, max_tilt)

        return result

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def _get_observation(self) -> dict:
        """
        Build observation dictionary.

        Returns:
            Observation dict with quadcopter and target state.
        """
        # Quadcopter state
        quad_state = {
            "position": self._state_vector[self.POS_X:self.POS_Z + 1].copy(),
            "velocity": self._state_vector[self.VEL_X:self.VEL_Z + 1].copy(),
            "attitude": self._state_vector[self.ROLL:self.YAW + 1].copy(),
            "angular_velocity": self._state_vector[
                self.ROLL_RATE:self.YAW_RATE + 1
            ].copy(),
        }

        # Target state
        target_state = self.target.get_state(self._time)

        return {
            "quadcopter": quad_state,
            "target": target_state,
            "time": self._time,
        }

    def _compute_tracking_error(self, observation: dict) -> float:
        """Compute distance from quadcopter to target."""
        quad_pos = observation["quadcopter"]["position"]
        target_pos = observation["target"]["position"]
        return float(np.linalg.norm(quad_pos - target_pos))

    def _compute_reward(self, observation: dict) -> float:
        """
        Compute reward for current state.

        Reward is negative distance to target (encourages tracking).
        """
        error = self._compute_tracking_error(observation)
        return -error

    def _check_termination(self) -> tuple[bool, str]:
        """
        Check if episode should terminate.

        Returns:
            Tuple of (done, reason).
        """
        # Time limit
        if self._time >= self.config.simulation.max_episode_time:
            return True, "time_limit"

        # Position bounds
        pos = self._state_vector[self.POS_X:self.POS_Z + 1]
        max_pos = self.config.simulation.max_position
        if np.any(np.abs(pos) > max_pos):
            return True, "position_bounds"

        # Check for numerical instability
        if not np.all(np.isfinite(self._state_vector)):
            logger.warning("Numerical instability detected in state")
            return True, "numerical_instability"

        return False, ""

    def _evaluate_success(self) -> bool:
        """
        Evaluate if episode was successful based on criteria.

        Returns:
            True if success criteria met.
        """
        # Check minimum duration
        if self._time < self.config.success_criteria.min_episode_duration:
            return False

        # Check on-target ratio
        if self._total_steps > 0:
            ratio = self._on_target_count / self._total_steps
        else:
            ratio = 0.0
        return ratio >= self.config.success_criteria.min_on_target_ratio

    def _record_step(
        self,
        observation: dict,
        action: np.ndarray,
        reward: float,
        info: dict,
    ) -> None:
        """Record step data for time series."""
        if self._step_count % self.config.logging.log_interval == 0:
            self._history.append({
                "time": self._time,
                "step": self._step_count,
                "quadcopter_position": observation["quadcopter"]["position"].tolist(),
                "quadcopter_velocity": observation["quadcopter"]["velocity"].tolist(),
                "quadcopter_attitude": observation["quadcopter"]["attitude"].tolist(),
                "target_position": observation["target"]["position"].tolist(),
                "target_velocity": observation["target"]["velocity"].tolist(),
                "action": action.tolist(),
                "reward": reward,
                "tracking_error": info["tracking_error"],
                "on_target": info["on_target"],
            })

    def get_history(self) -> list[dict]:
        """
        Get recorded time series data.

        Returns:
            List of recorded step dictionaries.
        """
        return self._history.copy()

    def get_action_violations(self) -> list[dict]:
        """
        Get logged action violations.

        Returns:
            List of violation records.
        """
        return self._action_violations.copy()

    def render(self, mode: str = "dict") -> dict | None:
        """
        Render current state.

        Args:
            mode: Render mode ('dict' returns state dictionary).

        Returns:
            State dictionary if mode='dict'.
        """
        if mode == "dict":
            return self._get_observation()
        return None

    @property
    def state(self) -> dict:
        """Get current state as dictionary."""
        return self._get_observation()

    @property
    def time(self) -> float:
        """Get current simulation time."""
        return self._time

    @property
    def dt(self) -> float:
        """Get simulation timestep."""
        return self.config.simulation.dt

    @property
    def is_initialized(self) -> bool:
        """Check if environment has been initialized."""
        return self._initialized

    def get_state_vector(self) -> np.ndarray:
        """Get raw state vector (for advanced use)."""
        return self._state_vector.copy()

    def set_state_vector(self, state: np.ndarray) -> None:
        """Set raw state vector (for advanced use)."""
        if state.shape != (self.STATE_DIM,):
            raise ValueError(
                f"State must have shape ({self.STATE_DIM},), got {state.shape}"
            )
        self._state_vector = state.copy()

    @staticmethod
    def hover_action(mass: float = 1.0, gravity: float = 9.81) -> dict:
        """
        Get action to maintain hover.

        Args:
            mass: Quadcopter mass in kg.
            gravity: Gravitational acceleration.

        Returns:
            Action dictionary for hover.
        """
        return {
            "thrust": mass * gravity,
            "roll_rate": 0.0,
            "pitch_rate": 0.0,
            "yaw_rate": 0.0,
        }
