"""
Base Controller Module

Provides the abstract base class for all quadcopter controllers.

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

    Controllers use these conventions to correctly map position errors to
    control outputs:
    - +X error (target ahead in X) → +pitch_rate output
    - +Y error (target ahead in Y) → -roll_rate output
    - +Z error (target above) → +thrust adjustment
"""

from dataclasses import dataclass

# Canonical action keys expected in all controller outputs
ACTION_KEYS = ("thrust", "roll_rate", "pitch_rate", "yaw_rate")


@dataclass
class ActionLimits:
    """Default action limits shared across all controllers."""

    min_thrust: float = 0.0
    max_thrust: float = 20.0
    max_rate: float = 3.0

    def clip_action(self, action: dict) -> dict:
        """
        Clip action values to valid ranges using numpy.

        Args:
            action: Action dictionary with control commands.

        Returns:
            Clipped action dictionary.
        """
        import numpy as np

        return {
            "thrust": np.clip(action["thrust"], self.min_thrust, self.max_thrust),
            "roll_rate": np.clip(action["roll_rate"], -self.max_rate, self.max_rate),
            "pitch_rate": np.clip(action["pitch_rate"], -self.max_rate, self.max_rate),
            "yaw_rate": np.clip(action["yaw_rate"], -self.max_rate, self.max_rate),
        }


# Default action limits instance
DEFAULT_ACTION_LIMITS = ActionLimits()


def validate_action(action: dict) -> None:
    """
    Validate that an action dictionary has the required schema.

    Args:
        action: Action dictionary to validate.

    Raises:
        KeyError: If required keys are missing.
        TypeError: If values are not numeric.
    """
    for key in ACTION_KEYS:
        if key not in action:
            raise KeyError(f"Action missing required key: '{key}'")
        if not isinstance(action[key], (int, float)):
            raise TypeError(
                f"Action['{key}'] must be numeric, got {type(action[key]).__name__}"
            )


class BaseController:
    """
    Abstract base class for quadcopter tracking controllers.

    All controllers should inherit from this class and implement
    the compute_action method.

    Attributes:
        name (str): Controller identifier for logging/comparison.
        config (dict): Controller-specific configuration.
        mass (float): Quadcopter mass in kg (default: 1.0).
        gravity (float): Gravitational acceleration in m/s² (default: 9.81).
    """

    def __init__(
        self,
        name: str = "base",
        config: dict | None = None,
        mass: float = 1.0,
        gravity: float = 9.81,
    ):
        """
        Initialize the controller.

        Args:
            name: Human-readable controller name.
            config: Controller configuration parameters.
            mass: Quadcopter mass in kg (default: 1.0).
            gravity: Gravitational acceleration in m/s² (default: 9.81).
        """
        self.name = name
        self.config = config or {}
        self.mass = mass
        self.gravity = gravity

    def compute_action(self, observation: dict) -> dict:
        """
        Compute control action from current observation.

        Args:
            observation: Environment observation containing state information.

        Returns:
            Action dictionary with control commands.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement compute_action")

    def reset(self) -> None:
        """Reset controller state (for stateful controllers)."""
        pass
