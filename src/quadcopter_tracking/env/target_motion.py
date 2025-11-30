"""
Target Motion Generation Module

Provides smooth trajectory generation for target tracking scenarios.
Supports multiple motion patterns with reproducible seeding.
"""

import math
from typing import Protocol

import numpy as np

from .config import TargetParams


class MotionPattern(Protocol):
    """Protocol for motion pattern implementations."""

    def get_state(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get target state at time t.

        Returns:
            Tuple of (position, velocity, acceleration) as numpy arrays.
        """
        ...


class LinearMotion:
    """Linear motion in a specified direction."""

    def __init__(
        self,
        start: np.ndarray,
        direction: np.ndarray,
        speed: float,
    ):
        """
        Initialize linear motion.

        Args:
            start: Starting position [x, y, z].
            direction: Unit direction vector.
            speed: Speed in m/s.
        """
        self.start = start.copy()
        self.direction = direction / np.linalg.norm(direction)
        self.speed = speed
        self.velocity = self.direction * self.speed

    def get_state(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get position, velocity, and acceleration at time t."""
        position = self.start + self.velocity * t
        velocity = self.velocity.copy()
        acceleration = np.zeros(3)
        return position, velocity, acceleration


class CircularMotion:
    """Circular/orbital motion in horizontal plane."""

    def __init__(
        self,
        center: np.ndarray,
        radius: float,
        speed: float,
        initial_angle: float = 0.0,
    ):
        """
        Initialize circular motion.

        Args:
            center: Center point [x, y, z].
            radius: Orbit radius in meters.
            speed: Linear speed in m/s.
            initial_angle: Starting angle in radians.
        """
        self.center = center.copy()
        self.radius = radius
        self.speed = speed
        self.omega = speed / radius  # angular velocity
        self.initial_angle = initial_angle

    def get_state(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get position, velocity, and acceleration at time t."""
        angle = self.initial_angle + self.omega * t

        # Position
        position = np.array(
            [
                self.center[0] + self.radius * np.cos(angle),
                self.center[1] + self.radius * np.sin(angle),
                self.center[2],
            ]
        )

        # Velocity (tangential)
        velocity = np.array(
            [
                -self.radius * self.omega * np.sin(angle),
                self.radius * self.omega * np.cos(angle),
                0.0,
            ]
        )

        # Acceleration (centripetal)
        acceleration = np.array(
            [
                -self.radius * self.omega**2 * np.cos(angle),
                -self.radius * self.omega**2 * np.sin(angle),
                0.0,
            ]
        )

        return position, velocity, acceleration


class SinusoidalMotion:
    """Sinusoidal motion with configurable amplitude and frequency."""

    def __init__(
        self,
        center: np.ndarray,
        amplitude: np.ndarray,
        frequency: np.ndarray,
        phase: np.ndarray = None,
    ):
        """
        Initialize sinusoidal motion.

        Args:
            center: Center/mean position [x, y, z].
            amplitude: Amplitude for each axis [ax, ay, az].
            frequency: Frequency for each axis in Hz [fx, fy, fz].
            phase: Initial phase for each axis in radians.
        """
        self.center = center.copy()
        self.amplitude = amplitude.copy()
        self.omega = 2 * np.pi * frequency  # angular frequency
        self.phase = phase if phase is not None else np.zeros(3)

    def get_state(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get position, velocity, and acceleration at time t."""
        theta = self.omega * t + self.phase

        position = self.center + self.amplitude * np.sin(theta)
        velocity = self.amplitude * self.omega * np.cos(theta)
        acceleration = -self.amplitude * self.omega**2 * np.sin(theta)

        return position, velocity, acceleration


class Figure8Motion:
    """Figure-8 (lemniscate) motion pattern."""

    def __init__(
        self,
        center: np.ndarray,
        scale: float,
        speed: float,
    ):
        """
        Initialize figure-8 motion.

        Args:
            center: Center position [x, y, z].
            scale: Size scale factor in meters.
            speed: Base speed parameter.
        """
        self.center = center.copy()
        self.scale = scale
        self.omega = speed / scale  # angular parameter rate

    def get_state(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get position, velocity, and acceleration at time t."""
        theta = self.omega * t

        # Lemniscate of Bernoulli parametrization
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        denom = 1 + sin_t**2

        position = np.array(
            [
                self.center[0] + self.scale * cos_t / denom,
                self.center[1] + self.scale * sin_t * cos_t / denom,
                self.center[2],
            ]
        )

        # Compute velocity analytically
        d_cos = -sin_t * self.omega
        d_sin = cos_t * self.omega
        d_denom = 2 * sin_t * d_sin

        dx = (d_cos * denom - cos_t * d_denom) / denom**2
        dy = (
            (d_sin * cos_t + sin_t * d_cos) * denom - sin_t * cos_t * d_denom
        ) / denom**2

        velocity = np.array(
            [
                self.scale * dx,
                self.scale * dy,
                0.0,
            ]
        )

        # Numerical approximation for acceleration
        # The analytic second derivative of the lemniscate is complex.
        # Using numerical differentiation with a small fixed step is acceptable
        # here because: (1) the acceleration is only used for limiting/smoothing,
        # (2) TargetMotion.get_state() clamps acceleration to max_acceleration,
        # and (3) the primary outputs (position, velocity) are computed exactly.
        dt = 1e-6
        theta_plus = self.omega * (t + dt)
        cos_tp = np.cos(theta_plus)
        sin_tp = np.sin(theta_plus)
        denom_p = 1 + sin_tp**2

        pos_plus = np.array(
            [
                self.center[0] + self.scale * cos_tp / denom_p,
                self.center[1] + self.scale * sin_tp * cos_tp / denom_p,
                self.center[2],
            ]
        )
        vel_plus = (pos_plus - position) / dt
        acceleration = (vel_plus - velocity) / dt

        return position, velocity, acceleration


class StationaryMotion:
    """Stationary target (hovering reference)."""

    def __init__(self, position: np.ndarray):
        """
        Initialize stationary target.

        Args:
            position: Fixed position [x, y, z].
        """
        self.position = position.copy()

    def get_state(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get position, velocity, and acceleration at time t."""
        return self.position.copy(), np.zeros(3), np.zeros(3)


class TargetMotion:
    """
    Target motion generator for tracking scenarios.

    Supports various motion patterns including linear, circular,
    sinusoidal, and figure-8 trajectories with smooth continuous motion.

    Attributes:
        motion_type: Type of motion pattern.
        params: Motion parameters.
        rng: Random number generator for reproducibility.
    """

    VALID_MOTION_TYPES = {
        "linear",
        "circular",
        "sinusoidal",
        "figure8",
        "stationary",
    }

    def __init__(
        self,
        params: TargetParams | None = None,
        seed: int | None = None,
    ):
        """
        Initialize target motion generator.

        Args:
            params: Target motion parameters.
            seed: Random seed for reproducibility.
        """
        self.params = params or TargetParams()
        self.rng = np.random.default_rng(seed)
        self._pattern: MotionPattern | None = None
        self._time = 0.0
        self._last_position = None
        self._last_velocity = None

    def reset(self, seed: int | None = None) -> None:
        """
        Reset the motion generator.

        Args:
            seed: Optional new seed for reproducibility.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._time = 0.0
        self._last_position = None
        self._last_velocity = None
        self._pattern = self._create_pattern()

    def _create_pattern(self) -> MotionPattern:
        """Create motion pattern based on parameters."""
        motion_type = self.params.motion_type.lower()

        if motion_type not in self.VALID_MOTION_TYPES:
            raise ValueError(
                f"Invalid motion type: {motion_type}. "
                f"Valid types: {self.VALID_MOTION_TYPES}"
            )

        center = np.array(self.params.center)

        if motion_type == "linear":
            # Random initial direction
            direction = self.rng.standard_normal(3)
            direction /= np.linalg.norm(direction)
            return LinearMotion(
                start=center,
                direction=direction,
                speed=self.params.speed,
            )

        elif motion_type == "circular":
            initial_angle = self.rng.uniform(0, 2 * math.pi)
            return CircularMotion(
                center=center,
                radius=self.params.radius,
                speed=self.params.speed,
                initial_angle=initial_angle,
            )

        elif motion_type == "sinusoidal":
            # Use amplitude for all axes, with different frequencies
            amplitude = np.array(
                [
                    self.params.amplitude,
                    self.params.amplitude * 0.5,
                    self.params.amplitude * 0.25,
                ]
            )
            frequency = np.array(
                [
                    self.params.frequency,
                    self.params.frequency * 1.3,
                    self.params.frequency * 0.7,
                ]
            )
            phase = self.rng.uniform(0, 2 * math.pi, 3)
            return SinusoidalMotion(
                center=center,
                amplitude=amplitude,
                frequency=frequency,
                phase=phase,
            )

        elif motion_type == "figure8":
            return Figure8Motion(
                center=center,
                scale=self.params.amplitude,
                speed=self.params.speed,
            )

        else:  # stationary
            return StationaryMotion(position=center)

    def get_position(self, time: float) -> tuple[float, float, float]:
        """
        Get target position at specified time.

        Args:
            time: Simulation time in seconds.

        Returns:
            Tuple of (x, y, z) position coordinates.
        """
        if self._pattern is None:
            self.reset()

        position, _, _ = self._pattern.get_state(time)
        return tuple(position)

    def get_state(self, time: float) -> dict:
        """
        Get complete target state at specified time.

        Args:
            time: Simulation time in seconds.

        Returns:
            Dictionary with position, velocity, acceleration.
        """
        if self._pattern is None:
            self.reset()

        position, velocity, acceleration = self._pattern.get_state(time)

        # Enforce maximum acceleration for smoothness
        accel_mag = np.linalg.norm(acceleration)
        if accel_mag > self.params.max_acceleration:
            acceleration = acceleration / accel_mag * self.params.max_acceleration

        return {
            "position": position.copy(),
            "velocity": velocity.copy(),
            "acceleration": acceleration.copy(),
        }

    def step(self, dt: float) -> dict:
        """
        Advance time and get new target state.

        Args:
            dt: Time step in seconds.

        Returns:
            Dictionary with position, velocity, acceleration.
        """
        self._time += dt
        return self.get_state(self._time)

    @property
    def current_time(self) -> float:
        """Get current simulation time."""
        return self._time
