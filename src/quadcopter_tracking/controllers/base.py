"""
Base Controller Module

Provides the abstract base class for all quadcopter controllers.
"""


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
