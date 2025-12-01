"""
Riccati-LQR Controller Module

Implements a true LQR controller that solves the discrete-time algebraic Riccati
equation (DARE) to compute optimal feedback gains K, rather than using heuristic
approximations.

The DARE solver computes the steady-state solution P to:
    A'PA - P - A'PB(R + B'PB)^{-1}B'PA + Q = 0

The optimal feedback gain is then:
    K = (R + B'PB)^{-1}B'PA

This provides a mathematically optimal solution for the linearized quadcopter
dynamics around hover, making it suitable as a strong teacher for deep imitation
controllers.

Dependencies:
    - scipy.linalg.solve_discrete_are for DARE solving

Usage:
    controller = RiccatiLQRController(config={
        'dt': 0.01,  # Required: simulation timestep
        'Q': Q_matrix,  # Optional: state cost (6x6)
        'R': R_matrix,  # Optional: control cost (4x4)
    })
    action = controller.compute_action(observation)
"""

import logging

import numpy as np

from .base import BaseController

logger = logging.getLogger(__name__)


def _is_positive_semidefinite(matrix: np.ndarray, name: str = "matrix") -> bool:
    """
    Check if a matrix is positive semi-definite.

    A matrix is positive semi-definite if all eigenvalues are >= 0.

    Args:
        matrix: Square matrix to check.
        name: Name for error messages.

    Returns:
        True if positive semi-definite, False otherwise.
    """
    if matrix.shape[0] != matrix.shape[1]:
        return False

    # Check symmetry
    if not np.allclose(matrix, matrix.T, atol=1e-8):
        logger.warning("%s is not symmetric", name)
        return False

    # Check eigenvalues
    eigenvalues = np.linalg.eigvalsh(matrix)
    if np.any(eigenvalues < -1e-10):
        logger.warning("%s has negative eigenvalues: %s", name, eigenvalues)
        return False

    return True


def _is_positive_definite(matrix: np.ndarray, name: str = "matrix") -> bool:
    """
    Check if a matrix is positive definite.

    A matrix is positive definite if all eigenvalues are > 0.

    Args:
        matrix: Square matrix to check.
        name: Name for error messages.

    Returns:
        True if positive definite, False otherwise.
    """
    if matrix.shape[0] != matrix.shape[1]:
        return False

    # Check symmetry
    if not np.allclose(matrix, matrix.T, atol=1e-8):
        logger.warning("%s is not symmetric", name)
        return False

    # Check eigenvalues are strictly positive
    eigenvalues = np.linalg.eigvalsh(matrix)
    if np.any(eigenvalues <= 1e-10):
        logger.warning(
            "%s is not positive definite, eigenvalues: %s", name, eigenvalues
        )
        return False

    return True


def solve_dare(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the discrete-time algebraic Riccati equation (DARE).

    Computes the solution P and optimal feedback gain K for:
        A'PA - P - A'PB(R + B'PB)^{-1}B'PA + Q = 0
        K = (R + B'PB)^{-1}B'PA

    Args:
        A: State transition matrix (n x n).
        B: Control input matrix (n x m).
        Q: State cost matrix (n x n), must be positive semi-definite.
        R: Control cost matrix (m x m), must be positive definite.

    Returns:
        Tuple of (P, K) where P is the solution matrix and K is the feedback gain.

    Raises:
        ValueError: If matrices have incompatible dimensions or invalid properties.
        RuntimeError: If DARE solver fails to converge.
    """
    try:
        from scipy.linalg import solve_discrete_are
    except ImportError as e:
        raise ImportError(
            "scipy is required for RiccatiLQRController. "
            "Install it with: pip install scipy>=1.11.0"
        ) from e

    # Validate dimensions
    n = A.shape[0]
    m = B.shape[1]

    if A.shape != (n, n):
        raise ValueError(f"A must be square, got shape {A.shape}")
    if B.shape != (n, m):
        raise ValueError(f"B must have shape ({n}, m), got {B.shape}")
    if Q.shape != (n, n):
        raise ValueError(f"Q must have shape ({n}, {n}), got {Q.shape}")
    if R.shape != (m, m):
        raise ValueError(f"R must have shape ({m}, {m}), got {R.shape}")

    # Validate Q is positive semi-definite
    if not _is_positive_semidefinite(Q, "Q"):
        raise ValueError("Q matrix must be positive semi-definite")

    # Validate R is positive definite
    if not _is_positive_definite(R, "R"):
        raise ValueError("R matrix must be positive definite")

    # Solve DARE
    try:
        P = solve_discrete_are(A, B, Q, R)
    except Exception as e:
        raise RuntimeError(f"DARE solver failed: {e}") from e

    # Compute optimal gain K = (R + B'PB)^{-1} B'PA
    BtP = B.T @ P
    K = np.linalg.solve(R + BtP @ B, BtP @ A)

    return P, K


def build_linearized_system(
    dt: float,
    mass: float = 1.0,
    gravity: float = 9.81,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the discrete-time linearized quadcopter system matrices A and B.

    The state vector is [x, y, z, vx, vy, vz] (6 dimensions).
    The control vector is [thrust_delta, roll_rate, pitch_rate, yaw_rate]
    (4 dimensions).

    The linearization is around hover where:
    - thrust_delta = thrust - hover_thrust (deviation from hover)
    - Small angle assumption for roll/pitch rates

    Dynamics (continuous-time):
        x_dot = vx
        y_dot = vy
        z_dot = vz
        vx_dot = gravity * pitch  (small angle, pitch ~ pitch_rate * dt integrated)
        vy_dot = -gravity * roll  (small angle, roll ~ roll_rate * dt integrated)
        vz_dot = thrust_delta / mass

    For discrete-time with timestep dt, using Euler approximation:
        A_d = I + A_c * dt
        B_d = B_c * dt

    Args:
        dt: Simulation timestep in seconds.
        mass: Quadcopter mass in kg.
        gravity: Gravitational acceleration in m/s^2.

    Returns:
        Tuple of (A, B) discrete-time system matrices.
    """
    # State dimension: [x, y, z, vx, vy, vz]
    n = 6
    # Control dimension: [thrust_delta, roll_rate, pitch_rate, yaw_rate]
    m = 4

    # Continuous-time A matrix (position-velocity kinematics)
    A_c = np.zeros((n, n))
    A_c[0, 3] = 1.0  # x_dot = vx
    A_c[1, 4] = 1.0  # y_dot = vy
    A_c[2, 5] = 1.0  # z_dot = vz

    # Continuous-time B matrix
    # Maps control inputs to state derivatives
    #
    # For the quadcopter dynamics around hover:
    # - vx_dot = g * pitch (pitch angle)
    # - vy_dot = -g * roll (roll angle)
    # - vz_dot = thrust_delta / mass
    #
    # Since we control roll_rate and pitch_rate (not angles directly), we use
    # a simplified model assuming the inner attitude loop is fast enough that
    # the effective relationship is:
    # - vx_dot ≈ g * pitch_rate (treating pitch_rate as proxy for pitch angle)
    # - vy_dot ≈ -g * roll_rate (treating roll_rate as proxy for roll angle)
    #
    B_c = np.zeros((n, m))
    # thrust_delta affects vz_dot: vz_dot = thrust_delta / mass
    B_c[5, 0] = 1.0 / mass
    # roll_rate affects vy_dot: vy_dot = -gravity * (roll_rate as proxy)
    B_c[4, 1] = -gravity
    # pitch_rate affects vx_dot: vx_dot = gravity * (pitch_rate as proxy)
    B_c[3, 2] = gravity
    # yaw_rate doesn't affect position/velocity in this linearization

    # Discrete-time approximation using Euler method:
    # A_d = I + A_c * dt
    # B_d = B_c * dt
    A = np.eye(n) + A_c * dt
    B = B_c * dt

    return A, B


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


class RiccatiLQRController(BaseController):
    """
    Riccati-LQR Controller for quadcopter tracking.

    Solves the discrete-time algebraic Riccati equation (DARE) to compute
    optimal feedback gains for the linearized quadcopter dynamics around hover.
    This provides a mathematically rigorous LQR solution rather than heuristic
    gains.

    The controller is suitable for:
    - Serving as a strong teacher for deep imitation learning
    - Validating control performance against optimal baselines
    - Research on LQR-based quadcopter control

    State vector (6 dimensions):
        [x_error, y_error, z_error, vx_error, vy_error, vz_error]

    Control vector (4 dimensions):
        [thrust, roll_rate, pitch_rate, yaw_rate]

    Attributes:
        dt (float): Simulation timestep.
        A (ndarray): Discrete-time state transition matrix (6x6).
        B (ndarray): Discrete-time control input matrix (6x4).
        Q (ndarray): State cost matrix (6x6).
        R (ndarray): Control cost matrix (4x4).
        K (ndarray): Optimal feedback gain matrix (4x6).
        P (ndarray): DARE solution matrix (6x6).
        hover_thrust (float): Thrust required to hover (mass * gravity).
        fallback_controller: Heuristic LQR controller used on solver failure.

    Raises:
        ValueError: If Q/R matrices are invalid (not positive semi-definite/definite).
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize Riccati-LQR controller.

        Args:
            config: Configuration dictionary with parameters:
                - dt: Simulation timestep in seconds (required, default: 0.01)
                - mass: Quadcopter mass in kg (default: 1.0)
                - gravity: Gravitational acceleration (default: 9.81)
                - Q: State cost matrix (6x6) or diagonal weights as list
                - R: Control cost matrix (4x4) or diagonal weights as list
                - q_pos: Position cost weights [x, y, z] (alternative to Q)
                - q_vel: Velocity cost weights [vx, vy, vz] (alternative to Q)
                - r_controls: Control weights [thrust, roll, pitch, yaw]
                - max_thrust: Maximum thrust in N (default: 20.0)
                - min_thrust: Minimum thrust in N (default: 0.0)
                - max_rate: Maximum angular rate in rad/s (default: 3.0)
                - fallback_on_failure: Fall back to heuristic LQR (default: True)

        Raises:
            ValueError: If required parameters are missing or matrices are invalid.
        """
        config = config or {}

        # Physical parameters
        mass = config.get("mass", 1.0)
        gravity = config.get("gravity", 9.81)
        self.dt = config.get("dt", 0.01)

        super().__init__(name="riccati_lqr", config=config, mass=mass, gravity=gravity)

        # Output limits
        self.max_thrust = config.get("max_thrust", 20.0)
        self.min_thrust = config.get("min_thrust", 0.0)
        self.max_rate = config.get("max_rate", 3.0)

        # Hover thrust
        self.hover_thrust = self.mass * self.gravity

        # Build linearized system matrices
        self.A, self.B = build_linearized_system(
            dt=self.dt, mass=self.mass, gravity=self.gravity
        )

        # Build cost matrices Q and R
        self.Q = self._build_Q_matrix(config)
        self.R = self._build_R_matrix(config)

        # Fallback controller for solver failure
        self.fallback_on_failure = config.get("fallback_on_failure", True)
        self.fallback_controller = None
        self._using_fallback = False

        # Solve DARE and compute optimal gain
        self.P = None
        self.K = None
        self._solve_riccati()

        # Diagnostics
        self.last_control_components: dict | None = None

    def _build_Q_matrix(self, config: dict) -> np.ndarray:
        """
        Build the state cost matrix Q from configuration.

        Supports either a full 6x6 matrix or separate position/velocity weights.

        Args:
            config: Configuration dictionary.

        Returns:
            6x6 state cost matrix Q.
        """
        if "Q" in config and config["Q"] is not None:
            Q = np.array(config["Q"])
            if Q.shape != (6, 6):
                raise ValueError(f"Q matrix must have shape (6, 6), got {Q.shape}")
            return Q

        # Build from separate weights
        # Default weights matched to heuristic LQR for consistency:
        # XY position costs are small because meter→rad/s mapping saturates actuators
        # Z position cost is higher for tight altitude tracking
        q_pos = np.array(config.get("q_pos", [0.0001, 0.0001, 16.0]))
        q_vel = np.array(config.get("q_vel", [0.0036, 0.0036, 4.0]))

        if len(q_pos) != 3:
            raise ValueError(f"q_pos must have 3 elements, got {len(q_pos)}")
        if len(q_vel) != 3:
            raise ValueError(f"q_vel must have 3 elements, got {len(q_vel)}")

        Q = np.diag(np.concatenate([q_pos, q_vel]))
        return Q

    def _build_R_matrix(self, config: dict) -> np.ndarray:
        """
        Build the control cost matrix R from configuration.

        Supports either a full 4x4 matrix or diagonal weights.

        Args:
            config: Configuration dictionary.

        Returns:
            4x4 control cost matrix R.
        """
        if "R" in config and config["R"] is not None:
            R = np.array(config["R"])
            if R.shape != (4, 4):
                raise ValueError(f"R matrix must have shape (4, 4), got {R.shape}")
            return R

        # Build from diagonal weights
        # [thrust_delta, roll_rate, pitch_rate, yaw_rate]
        r_controls = np.array(config.get("r_controls", [1.0, 1.0, 1.0, 1.0]))

        if len(r_controls) != 4:
            raise ValueError(f"r_controls must have 4 elements, got {len(r_controls)}")

        R = np.diag(r_controls)
        return R

    def _solve_riccati(self) -> None:
        """
        Solve the DARE and compute optimal feedback gain K.

        If the solver fails and fallback is enabled, creates a heuristic LQR
        controller as a backup.
        """
        try:
            self.P, self.K = solve_dare(self.A, self.B, self.Q, self.R)
            self._using_fallback = False
            logger.info("DARE solved successfully, K shape: %s", self.K.shape)
        except (ValueError, RuntimeError, ImportError) as e:
            logger.warning("DARE solver failed: %s", e)

            if self.fallback_on_failure:
                logger.warning("Falling back to heuristic LQR controller")
                self._create_fallback_controller()
                self._using_fallback = True
            else:
                raise

    def _create_fallback_controller(self) -> None:
        """
        Create a heuristic LQR controller as fallback.

        Uses the existing LQRController implementation with matched parameters.
        """
        # Import here to avoid circular imports
        from . import LQRController

        # Extract diagonal weights from Q and R for heuristic LQR
        q_pos = [self.Q[0, 0], self.Q[1, 1], self.Q[2, 2]]
        q_vel = [self.Q[3, 3], self.Q[4, 4], self.Q[5, 5]]

        fallback_config = {
            "mass": self.mass,
            "gravity": self.gravity,
            "q_pos": q_pos,
            "q_vel": q_vel,
            "r_thrust": self.R[0, 0],
            "r_rate": self.R[1, 1],  # Use roll rate cost as representative
            "max_thrust": self.max_thrust,
            "min_thrust": self.min_thrust,
            "max_rate": self.max_rate,
        }

        self.fallback_controller = LQRController(config=fallback_config)
        logger.info("Fallback LQR controller created")

    def compute_action(self, observation: dict) -> dict:
        """
        Compute Riccati-LQR control action.

        If the DARE solver failed and fallback is enabled, delegates to the
        heuristic LQR controller.

        Args:
            observation: Environment observation with quadcopter and target state.

        Returns:
            Action dictionary with thrust, roll_rate, pitch_rate, yaw_rate.

        Raises:
            KeyError: If observation is missing required keys.
        """
        if self._using_fallback and self.fallback_controller is not None:
            return self.fallback_controller.compute_action(observation)

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
        u = self.K @ state_error

        # Store control components for diagnostics
        self.last_control_components = {
            "state_error": state_error.copy(),
            "feedback_u": u.copy(),
            "K_matrix": self.K.copy(),
        }

        # u[0] is thrust_delta (deviation from hover)
        thrust = self.hover_thrust + u[0]
        thrust = float(np.clip(thrust, self.min_thrust, self.max_thrust))

        # u[1] is roll_rate, u[2] is pitch_rate, u[3] is yaw_rate
        roll_rate = float(np.clip(u[1], -self.max_rate, self.max_rate))
        pitch_rate = float(np.clip(u[2], -self.max_rate, self.max_rate))
        yaw_rate = float(np.clip(u[3], -self.max_rate, self.max_rate))

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
            Dictionary with state_error, feedback_u, and K_matrix,
            or None if compute_action hasn't been called yet.
        """
        return self.last_control_components

    def is_using_fallback(self) -> bool:
        """
        Check if the controller is using the fallback heuristic LQR.

        Returns:
            True if using fallback, False if using true Riccati solution.
        """
        return self._using_fallback

    def get_gain_matrix(self) -> np.ndarray | None:
        """
        Get the computed feedback gain matrix K.

        Returns:
            The 4x6 feedback gain matrix, or None if using fallback.
        """
        return self.K if not self._using_fallback else None

    def get_riccati_solution(self) -> np.ndarray | None:
        """
        Get the DARE solution matrix P.

        Returns:
            The 6x6 DARE solution matrix, or None if using fallback.
        """
        return self.P if not self._using_fallback else None

    def reset(self) -> None:
        """Reset controller state (no-op for Riccati-LQR as it's stateless)."""
        self.last_control_components = None
        if self.fallback_controller is not None:
            self.fallback_controller.reset()
