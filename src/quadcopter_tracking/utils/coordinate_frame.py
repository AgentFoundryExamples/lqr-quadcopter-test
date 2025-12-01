"""
ENU Coordinate Frame Utilities

This module provides utilities and assertions for the ENU (East-North-Up)
coordinate frame used throughout the quadcopter tracking project.

Coordinate Frame Convention:
    - X-axis: East (positive direction)
    - Y-axis: North (positive direction)
    - Z-axis: Up (positive direction)

Sign Conventions (ENU):
    - +pitch_rate → +X velocity (pitching nose up accelerates forward/east)
    - +roll_rate → -Y velocity (rolling right accelerates left/south)
    - +thrust → +Z acceleration (upward force)
    - Gravity acts in the -Z direction

State Vector Layout:
    Position:     [x, y, z]           - meters (East, North, Up)
    Velocity:     [vx, vy, vz]        - m/s
    Attitude:     [phi, theta, psi]   - radians (roll, pitch, yaw)
    Angular rate: [p, q, r]           - rad/s

Control Mapping:
    - Z-axis position error → thrust adjustment
    - X-axis position error → pitch rate (+X error → +pitch_rate)
    - Y-axis position error → roll rate (+Y error → -roll_rate)

This module is the single source of truth for coordinate frame conventions.
Controllers, environments, and tests should import from here to ensure
consistency throughout the codebase.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

# ==============================================================================
# ENU Frame Constants
# ==============================================================================

# Axis indices for position/velocity vectors
AXIS_X = 0  # East
AXIS_Y = 1  # North
AXIS_Z = 2  # Up

# Gravity direction in ENU frame (negative Z)
GRAVITY_DIRECTION_ENU = np.array([0.0, 0.0, -1.0])

# Thrust direction in body frame (positive Z / up)
THRUST_DIRECTION_BODY = np.array([0.0, 0.0, 1.0])

# Control-to-motion sign mappings (ENU convention)
# These define the sign relationship between control outputs and resulting motion
PITCH_RATE_TO_X_VEL_SIGN = +1.0  # +pitch_rate → +X velocity
ROLL_RATE_TO_Y_VEL_SIGN = -1.0  # +roll_rate → -Y velocity
THRUST_TO_Z_ACCEL_SIGN = +1.0  # +thrust → +Z acceleration

# ==============================================================================
# Validation Tolerance Constants
# ==============================================================================

# Dot product tolerance for direction checks (cosine of ~8 degrees).
# This allows small numerical deviations while catching major frame errors.
# A value of 0.99 means vectors must be within ~8 degrees of expected direction.
DIRECTION_DOT_PRODUCT_TOLERANCE = 0.99

# Minimum vector magnitude to consider non-zero.
# Vectors with magnitude below this are treated as zero (no direction check).
ZERO_MAGNITUDE_THRESHOLD = 1e-10

# Default position error threshold for control sign checks (meters).
# Only errors larger than this trigger sign convention validation.
# This avoids false positives from noise or small oscillations during hover.
# Value of 0.5m chosen as typical tracking precision threshold.
DEFAULT_CONTROL_SIGN_ERROR_THRESHOLD = 0.5


# ==============================================================================
# Frame Identification
# ==============================================================================

@dataclass(frozen=True)
class CoordinateFrame:
    """Immutable descriptor for a coordinate frame convention."""

    name: str
    x_direction: str
    y_direction: str
    z_direction: str
    gravity_axis: Literal["x", "y", "z"]
    gravity_sign: Literal["+", "-"]

    def __str__(self) -> str:
        return (
            f"{self.name}: X={self.x_direction}, Y={self.y_direction}, "
            f"Z={self.z_direction}"
        )


# Canonical ENU frame used throughout this project
ENU_FRAME = CoordinateFrame(
    name="ENU",
    x_direction="East",
    y_direction="North",
    z_direction="Up",
    gravity_axis="z",
    gravity_sign="-",
)

# NED frame (for reference - NOT used in this project)
NED_FRAME = CoordinateFrame(
    name="NED",
    x_direction="North",
    y_direction="East",
    z_direction="Down",
    gravity_axis="z",
    gravity_sign="+",
)


def get_current_frame() -> CoordinateFrame:
    """
    Return the coordinate frame used by this project.

    The project exclusively uses ENU (East-North-Up) convention.
    This function provides a single point of access for frame information.

    Returns:
        CoordinateFrame descriptor for ENU.
    """
    return ENU_FRAME


# ==============================================================================
# ENU Frame Assertions
# ==============================================================================


class ENUFrameError(ValueError):
    """Exception raised when ENU frame conventions are violated."""

    pass


def assert_gravity_direction_enu(gravity_vector: np.ndarray) -> None:
    """
    Assert that gravity points in the correct direction for ENU frame.

    In ENU, gravity should point in the -Z direction (downward).

    Args:
        gravity_vector: 3D gravity vector to check.

    Raises:
        ENUFrameError: If gravity does not point in -Z direction.
        ValueError: If gravity_vector is not 3D.
    """
    gravity_vector = np.asarray(gravity_vector)
    if gravity_vector.shape != (3,):
        raise ValueError(f"Gravity vector must be 3D, got shape {gravity_vector.shape}")

    # Normalize for direction check
    magnitude = np.linalg.norm(gravity_vector)
    if magnitude < ZERO_MAGNITUDE_THRESHOLD:
        raise ENUFrameError("Gravity vector has zero magnitude")

    normalized = gravity_vector / magnitude

    # Check that gravity points in -Z direction
    expected = GRAVITY_DIRECTION_ENU
    dot_product = np.dot(normalized, expected)

    if dot_product < DIRECTION_DOT_PRODUCT_TOLERANCE:
        raise ENUFrameError(
            f"ENU frame violation: Gravity should point in -Z direction "
            f"(expected ~[0, 0, -1], got {normalized}). "
            f"Dot product with expected direction: {dot_product:.4f}"
        )


def assert_thrust_direction_enu(thrust_body: np.ndarray) -> None:
    """
    Assert that thrust points in the correct direction in body frame for ENU.

    In ENU with standard quadcopter convention, thrust should point in +Z
    direction in the body frame (upward when level).

    Args:
        thrust_body: 3D thrust vector in body frame.

    Raises:
        ENUFrameError: If thrust does not point in +Z direction.
        ValueError: If thrust_body is not 3D.
    """
    thrust_body = np.asarray(thrust_body)
    if thrust_body.shape != (3,):
        raise ValueError(f"Thrust vector must be 3D, got shape {thrust_body.shape}")

    magnitude = np.linalg.norm(thrust_body)
    if magnitude < ZERO_MAGNITUDE_THRESHOLD:
        return  # Zero thrust is acceptable

    normalized = thrust_body / magnitude
    expected = THRUST_DIRECTION_BODY
    dot_product = np.dot(normalized, expected)

    if dot_product < DIRECTION_DOT_PRODUCT_TOLERANCE:
        raise ENUFrameError(
            f"ENU frame violation: Thrust in body frame should point in +Z direction "
            f"(expected ~[0, 0, 1], got {normalized}). "
            f"Dot product with expected direction: {dot_product:.4f}"
        )


def assert_control_signs_enu(
    pos_error: np.ndarray,
    pitch_rate: float,
    roll_rate: float,
    error_threshold: float = DEFAULT_CONTROL_SIGN_ERROR_THRESHOLD,
) -> None:
    """
    Assert that control outputs have correct signs for ENU frame.

    Validates the control-to-motion sign conventions:
    - +X error → +pitch_rate (to accelerate forward/east)
    - +Y error → -roll_rate (to accelerate left/north; rolling right goes south)

    Args:
        pos_error: Position error [x, y, z] in meters (target - current).
        pitch_rate: Pitch rate control output in rad/s.
        roll_rate: Roll rate control output in rad/s.
        error_threshold: Minimum position error magnitude to check (meters).
            Errors below this threshold are ignored to avoid triggering on
            noise or small oscillations. Default: 0.5m (typical tracking
            precision threshold).

    Raises:
        ENUFrameError: If control signs violate ENU conventions.
        ValueError: If pos_error is not 3D.

    Note:
        Zero control outputs (pitch_rate=0 or roll_rate=0) for significant
        errors are treated as violations, as the controller should be
        actively correcting the error.
    """
    pos_error = np.asarray(pos_error)
    if pos_error.shape != (3,):
        raise ValueError(f"Position error must be 3D, got shape {pos_error.shape}")

    # Check X-axis: +X error should produce +pitch_rate
    if pos_error[AXIS_X] > error_threshold:
        if pitch_rate <= 0:
            raise ENUFrameError(
                f"ENU sign violation: +X error ({pos_error[AXIS_X]:.2f}m) "
                f"should produce +pitch_rate, but got pitch_rate={pitch_rate:.4f}. "
                "Check controller X-axis to pitch_rate mapping."
            )
    elif pos_error[AXIS_X] < -error_threshold:
        if pitch_rate >= 0:
            raise ENUFrameError(
                f"ENU sign violation: -X error ({pos_error[AXIS_X]:.2f}m) "
                f"should produce -pitch_rate, but got pitch_rate={pitch_rate:.4f}. "
                "Check controller X-axis to pitch_rate mapping."
            )

    # Check Y-axis: +Y error should produce -roll_rate
    if pos_error[AXIS_Y] > error_threshold:
        if roll_rate >= 0:
            raise ENUFrameError(
                f"ENU sign violation: +Y error ({pos_error[AXIS_Y]:.2f}m) "
                f"should produce -roll_rate, but got roll_rate={roll_rate:.4f}. "
                "Check controller Y-axis to roll_rate mapping."
            )
    elif pos_error[AXIS_Y] < -error_threshold:
        if roll_rate <= 0:
            raise ENUFrameError(
                f"ENU sign violation: -Y error ({pos_error[AXIS_Y]:.2f}m) "
                f"should produce +roll_rate, but got roll_rate={roll_rate:.4f}. "
                "Check controller Y-axis to roll_rate mapping."
            )


def assert_z_up(position: np.ndarray, min_altitude: float = -100.0) -> None:
    """
    Assert that Z-axis represents altitude in ENU frame.

    In ENU, Z is "up", so typical quadcopter altitudes should be positive
    or at least above a reasonable minimum (e.g., ground level).

    Args:
        position: 3D position vector [x, y, z] in meters.
        min_altitude: Minimum expected altitude in meters (default: -100m).

    Raises:
        ENUFrameError: If altitude is below minimum.
        ValueError: If position is not 3D.
    """
    position = np.asarray(position)
    if position.shape != (3,):
        raise ValueError(f"Position must be 3D, got shape {position.shape}")

    altitude = position[AXIS_Z]
    if altitude < min_altitude:
        raise ENUFrameError(
            f"ENU frame check: Z-axis (altitude) is {altitude:.2f}m, "
            f"below minimum {min_altitude:.2f}m. "
            "This may indicate Z-axis is inverted (using NED instead of ENU)."
        )


# ==============================================================================
# Frame Validation Helpers
# ==============================================================================


def validate_observation_frame(observation: dict) -> bool:
    """
    Validate that an observation dictionary follows ENU conventions.

    Performs basic sanity checks on observation data:
    1. Target altitude (Z) should typically be positive (above ground)
    2. Quadcopter altitude should be reasonable

    Args:
        observation: Environment observation dictionary.

    Returns:
        True if observation passes ENU validation checks.

    Raises:
        ENUFrameError: If observation violates ENU conventions.
        KeyError: If required keys are missing.
    """
    quad_pos = np.asarray(observation["quadcopter"]["position"])
    target_pos = np.asarray(observation["target"]["position"])

    # Check altitudes are reasonable for ENU (Z-up)
    # Typical flying altitudes are positive
    if target_pos[AXIS_Z] < -10.0:
        raise ENUFrameError(
            f"Target Z position ({target_pos[AXIS_Z]:.2f}m) is very negative. "
            "In ENU, Z should be 'up' with positive altitudes for flying targets. "
            "Check if coordinate frame is NED instead of ENU."
        )

    if quad_pos[AXIS_Z] < -50.0:
        raise ENUFrameError(
            f"Quadcopter Z position ({quad_pos[AXIS_Z]:.2f}m) is very negative. "
            "In ENU, Z should be 'up'. Check coordinate frame convention."
        )

    return True


def compute_expected_hover_thrust(mass: float, gravity: float = 9.81) -> float:
    """
    Compute expected hover thrust for ENU frame.

    In ENU, gravity acts downward (-Z), so thrust to hover equals mg upward.

    Args:
        mass: Quadcopter mass in kg.
        gravity: Gravitational acceleration magnitude in m/s² (positive).

    Returns:
        Hover thrust in Newtons.

    Raises:
        ValueError: If mass or gravity is negative.
    """
    if mass < 0:
        raise ValueError(f"Mass must be non-negative, got {mass}")
    if gravity < 0:
        raise ValueError(f"Gravity magnitude must be non-negative, got {gravity}")

    return mass * gravity


def compute_position_error_enu(
    quadcopter_pos: np.ndarray,
    target_pos: np.ndarray,
) -> np.ndarray:
    """
    Compute position error in ENU frame.

    Error convention: target - current
    Positive error means target is ahead of quadcopter in that axis.

    Args:
        quadcopter_pos: Current quadcopter position [x, y, z] in meters.
        target_pos: Target position [x, y, z] in meters.

    Returns:
        Position error [x_error, y_error, z_error] in meters.
    """
    quadcopter_pos = np.asarray(quadcopter_pos)
    target_pos = np.asarray(target_pos)

    if quadcopter_pos.shape != (3,) or target_pos.shape != (3,):
        raise ValueError("Both positions must be 3D vectors")

    return target_pos - quadcopter_pos
