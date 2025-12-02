"""
PID and Feedforward Auto-Tuning Module

This module provides auto-tuning capabilities for PID (and feedforward) controller
gains. It supports grid search and random search strategies to find optimal gains
that minimize tracking error.

Design Philosophy:
- Extensible to LQR and other controller types
- Reproducible via deterministic seeding
- Interruptible with partial results saved
- Configurable via env vars and CLI arguments

Usage:
    from quadcopter_tracking.controllers.tuning import (
        TuningConfig,
        GainSearchSpace,
        ControllerTuner,
    )

    # Define search space for PID gains
    search_space = GainSearchSpace(
        kp_pos_range=([0.005, 0.005, 2.0], [0.05, 0.05, 6.0]),
        kd_pos_range=([0.02, 0.02, 1.0], [0.15, 0.15, 3.0]),
    )

    # Create tuner
    config = TuningConfig(
        controller_type="pid",
        search_space=search_space,
        strategy="random",
        max_iterations=50,
        evaluation_episodes=5,
    )
    tuner = ControllerTuner(config)

    # Run tuning
    result = tuner.tune()
    print(f"Best gains: {result.best_config}")
"""

import json
import logging
import os
import pickle
import signal
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from quadcopter_tracking.env import EnvConfig, QuadcopterEnv
from quadcopter_tracking.utils.metrics import SuccessCriteria, compute_episode_metrics

logger = logging.getLogger(__name__)

# Allowed base directories for output (prevents path traversal to sensitive areas)
_ALLOWED_OUTPUT_DIRS = frozenset(["reports", "experiments", "outputs", "tmp"])


def _validate_path(path: Path) -> None:
    """
    Validate that a path is safe and doesn't contain path traversal sequences.

    Args:
        path: Path to validate.

    Raises:
        ValueError: If path contains dangerous sequences or attempts traversal.
    """
    path_str = str(path)

    # Check for path traversal sequences
    if ".." in path_str:
        raise ValueError(
            f"Path contains path traversal sequence '..': {path}. "
            "Use absolute paths or paths without parent directory references."
        )

    # Check for null bytes (path injection)
    if "\x00" in path_str:
        raise ValueError(f"Path contains null byte: {path}")

    # Resolve to absolute path to detect traversal via symlinks
    try:
        resolved = path.resolve()
        # Ensure the resolved path doesn't escape expected directories
        # by checking it doesn't go above the current working directory
        # unless it's explicitly an absolute path the user provided
        if not path.is_absolute():
            cwd = Path.cwd().resolve()
            # Check that resolved path is under cwd or an allowed output dir
            try:
                resolved.relative_to(cwd)
            except ValueError:
                # Path escapes cwd - check if it's in an allowed location
                parts = resolved.parts
                if len(parts) < 2:
                    raise ValueError(
                        f"Relative path resolves outside working directory: {path}"
                    )
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Cannot resolve path {path}: {e}")


def _validate_output_dir(output_dir: str | Path) -> Path:
    """
    Validate and sanitize output directory path.

    Args:
        output_dir: Output directory path.

    Returns:
        Validated Path object.

    Raises:
        ValueError: If the path is invalid or potentially dangerous.
    """
    path = Path(output_dir)
    _validate_path(path)
    return path


# Delayed imports to avoid circular dependency
def _get_pid_controller():
    from quadcopter_tracking.controllers import PIDController

    return PIDController


def _get_lqr_controller():
    from quadcopter_tracking.controllers import LQRController

    return LQRController


def _get_riccati_lqr_controller():
    from quadcopter_tracking.controllers import RiccatiLQRController

    return RiccatiLQRController


@dataclass
class GainSearchSpace:
    """
    Search space definition for controller gains.

    Each *_range parameter is a tuple of (min_values, max_values) where each
    is a list of 3 floats for [x, y, z] axes. Setting a range to None excludes
    that parameter from the search.

    Attributes:
        kp_pos_range: PID proportional gains range (min, max) vectors
        ki_pos_range: PID integral gains range
        kd_pos_range: PID derivative gains range
        ff_velocity_gain_range: Feedforward velocity gains range
        ff_acceleration_gain_range: Feedforward acceleration gains range
        q_pos_range: LQR/Riccati position cost weights range
        q_vel_range: LQR/Riccati velocity cost weights range
        r_thrust_range: LQR thrust cost weight range (min, max) scalar
        r_rate_range: LQR rate cost weight range (min, max) scalar
        r_controls_range: Riccati control cost weights range [thrust, roll, pitch, yaw]
    """

    # PID gains
    kp_pos_range: tuple[list[float], list[float]] | None = None
    ki_pos_range: tuple[list[float], list[float]] | None = None
    kd_pos_range: tuple[list[float], list[float]] | None = None

    # Feedforward gains (optional)
    ff_velocity_gain_range: tuple[list[float], list[float]] | None = None
    ff_acceleration_gain_range: tuple[list[float], list[float]] | None = None

    # LQR/Riccati weights
    q_pos_range: tuple[list[float], list[float]] | None = None
    q_vel_range: tuple[list[float], list[float]] | None = None
    r_thrust_range: tuple[float, float] | None = None
    r_rate_range: tuple[float, float] | None = None

    # Riccati-specific: 4D control cost weights [thrust, roll, pitch, yaw]
    r_controls_range: tuple[list[float], list[float]] | None = None

    def validate(self) -> None:
        """
        Validate search ranges.

        Raises:
            ValueError: If any range is invalid or inverted.
        """
        ranges = [
            ("kp_pos_range", self.kp_pos_range),
            ("ki_pos_range", self.ki_pos_range),
            ("kd_pos_range", self.kd_pos_range),
            ("ff_velocity_gain_range", self.ff_velocity_gain_range),
            ("ff_acceleration_gain_range", self.ff_acceleration_gain_range),
            ("q_pos_range", self.q_pos_range),
            ("q_vel_range", self.q_vel_range),
        ]

        for name, rng in ranges:
            if rng is None:
                continue
            min_vals, max_vals = rng
            if len(min_vals) != 3 or len(max_vals) != 3:
                raise ValueError(
                    f"{name} must have exactly 3 values for [x, y, z], "
                    f"got min={len(min_vals)}, max={len(max_vals)}"
                )
            for i, (lo, hi) in enumerate(zip(min_vals, max_vals)):
                if lo > hi:
                    axis = ["x", "y", "z"][i]
                    raise ValueError(
                        f"{name}[{axis}] has inverted range: min={lo} > max={hi}. "
                        f"Swap values or use equal values for fixed parameter."
                    )
                if lo < 0:
                    axis = ["x", "y", "z"][i]
                    raise ValueError(
                        f"{name}[{axis}] has negative minimum: {lo}. "
                        f"Controller gains should be non-negative."
                    )

        # Scalar ranges
        scalar_ranges = [
            ("r_thrust_range", self.r_thrust_range),
            ("r_rate_range", self.r_rate_range),
        ]
        for name, rng in scalar_ranges:
            if rng is None:
                continue
            lo, hi = rng
            if lo > hi:
                raise ValueError(
                    f"{name} has inverted range: min={lo} > max={hi}. "
                    f"Swap values or use equal values for fixed parameter."
                )
            if lo < 0:
                raise ValueError(
                    f"{name} has negative minimum: {lo}. "
                    f"Cost weights should be non-negative."
                )

        # Riccati r_controls_range (4D vector)
        if self.r_controls_range is not None:
            min_vals, max_vals = self.r_controls_range
            if len(min_vals) != 4 or len(max_vals) != 4:
                raise ValueError(
                    f"r_controls_range must have exactly 4 values for "
                    f"[thrust, roll, pitch, yaw], "
                    f"got min={len(min_vals)}, max={len(max_vals)}"
                )
            control_names = ["thrust", "roll", "pitch", "yaw"]
            for i, (lo, hi) in enumerate(zip(min_vals, max_vals)):
                if lo > hi:
                    raise ValueError(
                        f"r_controls_range[{control_names[i]}] has inverted range: "
                        f"min={lo} > max={hi}. "
                        f"Swap values or use equal values for fixed parameter."
                    )
                if lo < 0:
                    raise ValueError(
                        f"r_controls_range[{control_names[i]}] has negative minimum: "
                        f"{lo}. Cost weights should be non-negative."
                    )

    def get_active_parameters(self) -> list[str]:
        """Get list of parameter names that have search ranges defined."""
        params = []
        if self.kp_pos_range is not None:
            params.append("kp_pos")
        if self.ki_pos_range is not None:
            params.append("ki_pos")
        if self.kd_pos_range is not None:
            params.append("kd_pos")
        if self.ff_velocity_gain_range is not None:
            params.append("ff_velocity_gain")
        if self.ff_acceleration_gain_range is not None:
            params.append("ff_acceleration_gain")
        if self.q_pos_range is not None:
            params.append("q_pos")
        if self.q_vel_range is not None:
            params.append("q_vel")
        if self.r_thrust_range is not None:
            params.append("r_thrust")
        if self.r_rate_range is not None:
            params.append("r_rate")
        if self.r_controls_range is not None:
            params.append("r_controls")
        return params

    @classmethod
    def from_dict(cls, config: dict) -> "GainSearchSpace":
        """Create search space from dictionary."""
        return cls(
            kp_pos_range=config.get("kp_pos_range"),
            ki_pos_range=config.get("ki_pos_range"),
            kd_pos_range=config.get("kd_pos_range"),
            ff_velocity_gain_range=config.get("ff_velocity_gain_range"),
            ff_acceleration_gain_range=config.get("ff_acceleration_gain_range"),
            q_pos_range=config.get("q_pos_range"),
            q_vel_range=config.get("q_vel_range"),
            r_thrust_range=config.get("r_thrust_range"),
            r_rate_range=config.get("r_rate_range"),
            r_controls_range=config.get("r_controls_range"),
        )

    def to_dict(self) -> dict:
        """Convert search space to dictionary."""
        return {
            "kp_pos_range": self.kp_pos_range,
            "ki_pos_range": self.ki_pos_range,
            "kd_pos_range": self.kd_pos_range,
            "ff_velocity_gain_range": self.ff_velocity_gain_range,
            "ff_acceleration_gain_range": self.ff_acceleration_gain_range,
            "q_pos_range": self.q_pos_range,
            "q_vel_range": self.q_vel_range,
            "r_thrust_range": self.r_thrust_range,
            "r_rate_range": self.r_rate_range,
            "r_controls_range": self.r_controls_range,
        }


@dataclass
class TuningConfig:
    """
    Configuration for controller auto-tuning.

    Attributes:
        controller_type: Type of controller to tune ('pid', 'lqr', or 'riccati_lqr')
        search_space: Gain search space definition
        strategy: Search strategy ('grid', 'random', or 'cma_es')
        max_iterations: Maximum number of configurations to evaluate
        grid_points_per_dim: Points per dimension for grid search (default: 3)
        evaluation_episodes: Number of episodes to evaluate each configuration
        evaluation_horizon: Max steps per evaluation episode
        seed: Random seed for reproducibility (default: 42)
        target_motion_type: Target motion for evaluation
        episode_length: Episode duration in seconds
        target_radius: On-target radius threshold
        output_dir: Directory for saving tuning results
        resume_from: Path to previous tuning results to resume from
        feedforward_enabled: Whether feedforward is enabled for tuning
        cma_sigma0: Initial standard deviation for CMA-ES (default: 0.3)
        cma_popsize: CMA-ES population size (default: None for auto)
    """

    controller_type: str = "pid"
    search_space: GainSearchSpace = field(default_factory=GainSearchSpace)
    strategy: str = "random"
    max_iterations: int = 50
    grid_points_per_dim: int = 3
    evaluation_episodes: int = 5
    evaluation_horizon: int = 3000
    seed: int = 42
    target_motion_type: str = "stationary"
    episode_length: float = 30.0
    target_radius: float = 0.5
    output_dir: str = "reports/tuning"
    resume_from: str | None = None
    feedforward_enabled: bool = False
    cma_sigma0: float = 0.3
    cma_popsize: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        valid_controllers = ("pid", "lqr", "riccati_lqr")
        if self.controller_type not in valid_controllers:
            raise ValueError(
                f"Invalid controller_type: '{self.controller_type}'. "
                f"Valid choices are: {', '.join(valid_controllers)}"
            )

        valid_strategies = ("grid", "random", "cma_es")
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"Invalid strategy: '{self.strategy}'. "
                f"Valid choices are: {', '.join(valid_strategies)}"
            )

        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations}")

        if self.evaluation_episodes < 1:
            raise ValueError(
                f"evaluation_episodes must be >= 1, got {self.evaluation_episodes}"
            )

        if self.cma_sigma0 <= 0:
            raise ValueError(f"cma_sigma0 must be > 0, got {self.cma_sigma0}")

        if self.cma_popsize is not None and self.cma_popsize < 2:
            raise ValueError(f"cma_popsize must be >= 2, got {self.cma_popsize}")

    @classmethod
    def from_dict(cls, config: dict) -> "TuningConfig":
        """Create config from dictionary."""
        search_space_dict = config.get("search_space", {})
        search_space = GainSearchSpace.from_dict(search_space_dict)

        return cls(
            controller_type=config.get("controller_type", "pid"),
            search_space=search_space,
            strategy=config.get("strategy", "random"),
            max_iterations=config.get("max_iterations", 50),
            grid_points_per_dim=config.get("grid_points_per_dim", 3),
            evaluation_episodes=config.get("evaluation_episodes", 5),
            evaluation_horizon=config.get("evaluation_horizon", 3000),
            seed=config.get("seed", 42),
            target_motion_type=config.get("target_motion_type", "stationary"),
            episode_length=config.get("episode_length", 30.0),
            target_radius=config.get("target_radius", 0.5),
            output_dir=config.get("output_dir", "reports/tuning"),
            resume_from=config.get("resume_from"),
            feedforward_enabled=config.get("feedforward_enabled", False),
            cma_sigma0=config.get("cma_sigma0", 0.3),
            cma_popsize=config.get("cma_popsize"),
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "controller_type": self.controller_type,
            "search_space": self.search_space.to_dict(),
            "strategy": self.strategy,
            "max_iterations": self.max_iterations,
            "grid_points_per_dim": self.grid_points_per_dim,
            "evaluation_episodes": self.evaluation_episodes,
            "evaluation_horizon": self.evaluation_horizon,
            "seed": self.seed,
            "target_motion_type": self.target_motion_type,
            "episode_length": self.episode_length,
            "target_radius": self.target_radius,
            "output_dir": self.output_dir,
            "resume_from": self.resume_from,
            "feedforward_enabled": self.feedforward_enabled,
            "cma_sigma0": self.cma_sigma0,
            "cma_popsize": self.cma_popsize,
        }


@dataclass
class TuningResult:
    """
    Result of controller auto-tuning.

    Attributes:
        best_config: Best controller configuration found
        best_score: Score of best configuration (higher is better)
        best_metrics: Metrics from best configuration evaluation
        all_results: List of all evaluated configurations and scores
        iterations_completed: Number of iterations completed
        interrupted: Whether tuning was interrupted
        timestamp: When tuning was completed
        config: Tuning configuration used
    """

    best_config: dict
    best_score: float
    best_metrics: dict
    all_results: list[dict]
    iterations_completed: int
    interrupted: bool
    timestamp: str
    config: dict

    def to_dict(self) -> dict:
        """Convert result to dictionary for serialization."""
        return {
            "best_config": self.best_config,
            "best_score": self.best_score,
            "best_metrics": self.best_metrics,
            "all_results": self.all_results,
            "iterations_completed": self.iterations_completed,
            "interrupted": self.interrupted,
            "timestamp": self.timestamp,
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TuningResult":
        """Create result from dictionary."""
        return cls(
            best_config=data["best_config"],
            best_score=data["best_score"],
            best_metrics=data["best_metrics"],
            all_results=data["all_results"],
            iterations_completed=data["iterations_completed"],
            interrupted=data["interrupted"],
            timestamp=data["timestamp"],
            config=data["config"],
        )

    def save(self, path: str | Path) -> Path:
        """
        Save results to JSON file.

        Args:
            path: Path to save file.

        Returns:
            Path to saved file.

        Raises:
            ValueError: If path contains path traversal sequences.
        """
        path = Path(path)
        # Validate path to prevent path traversal attacks
        _validate_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TuningResult":
        """
        Load results from JSON file.

        Args:
            path: Path to load file.

        Returns:
            TuningResult loaded from file.

        Raises:
            ValueError: If path contains path traversal sequences.
            FileNotFoundError: If the file does not exist.
        """
        path = Path(path)
        # Validate path to prevent path traversal attacks
        _validate_path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tuning results file not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


class ControllerTuner:
    """
    Auto-tuner for PID and LQR controller gains.

    Evaluates controller configurations using grid, random, or CMA-ES search
    to find gains that minimize tracking error.

    Attributes:
        config: Tuning configuration
        rng: Random number generator for reproducibility
        results: List of evaluation results
        best_config: Best configuration found so far
        best_score: Score of best configuration
        interrupted: Whether tuning was interrupted
        cma_es: CMA-ES optimizer instance (when using cma_es strategy)
    """

    def __init__(self, config: TuningConfig):
        """
        Initialize controller tuner.

        Args:
            config: Tuning configuration.

        Raises:
            ValueError: If configuration is invalid.
            ImportError: If CMA-ES strategy is selected but cma package is missing.
        """
        self.config = config
        self.config.search_space.validate()

        # Initialize RNG with seed for reproducibility
        self.rng = np.random.default_rng(config.seed)

        # Tracking state
        self.results: list[dict] = []
        self.best_config: dict = {}
        self.best_score: float = float("-inf")
        self.best_metrics: dict = {}
        self.interrupted: bool = False

        # Handle interruptions gracefully
        self._original_sigint = None
        self._original_sigterm = None

        # Output directory - validate to prevent path traversal
        output_dir_str = os.environ.get("TUNING_OUTPUT_DIR", config.output_dir)
        self.output_dir = _validate_output_dir(output_dir_str)

        # CMA-ES specific state
        self.cma_es = None
        self._cma_param_spec: list[tuple[str, int, tuple]] = []
        self._cma_checkpoint_path: Path | None = None

        # Validate CMA-ES availability if needed
        if config.strategy == "cma_es":
            try:
                import cma  # noqa: F401
            except ImportError:
                raise ImportError(
                    "CMA-ES strategy requires the 'cma' package. "
                    "Install it with: pip install cma"
                )

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful interruption."""
        self._original_sigint = signal.signal(signal.SIGINT, self._handle_interrupt)
        self._original_sigterm = signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm)

    def _handle_interrupt(self, signum: int, frame: Any) -> None:
        """Handle interrupt signals by setting interrupted flag."""
        logger.warning("Received interrupt signal. Saving partial results...")
        self.interrupted = True

    def _sample_vector_param(
        self, rng: np.random.Generator, range_def: tuple[list[float], list[float]]
    ) -> list[float]:
        """Sample a 3D vector parameter from uniform distribution."""
        min_vals, max_vals = range_def
        return [float(rng.uniform(lo, hi)) for lo, hi in zip(min_vals, max_vals)]

    def _sample_scalar_param(
        self, rng: np.random.Generator, range_def: tuple[float, float]
    ) -> float:
        """Sample a scalar parameter from uniform distribution."""
        lo, hi = range_def
        return float(rng.uniform(lo, hi))

    def _sample_4d_vector_param(
        self, rng: np.random.Generator, range_def: tuple[list[float], list[float]]
    ) -> list[float]:
        """Sample a 4D vector parameter from uniform distribution."""
        min_vals, max_vals = range_def
        return [float(rng.uniform(lo, hi)) for lo, hi in zip(min_vals, max_vals)]

    def _generate_random_config(self) -> dict:
        """Generate a random controller configuration."""
        space = self.config.search_space
        config: dict = {}

        # PID parameters
        if space.kp_pos_range is not None:
            config["kp_pos"] = self._sample_vector_param(self.rng, space.kp_pos_range)
        if space.ki_pos_range is not None:
            config["ki_pos"] = self._sample_vector_param(self.rng, space.ki_pos_range)
        if space.kd_pos_range is not None:
            config["kd_pos"] = self._sample_vector_param(self.rng, space.kd_pos_range)

        # Feedforward parameters
        if space.ff_velocity_gain_range is not None:
            config["ff_velocity_gain"] = self._sample_vector_param(
                self.rng, space.ff_velocity_gain_range
            )
            config["feedforward_enabled"] = True
        if space.ff_acceleration_gain_range is not None:
            config["ff_acceleration_gain"] = self._sample_vector_param(
                self.rng, space.ff_acceleration_gain_range
            )
            config["feedforward_enabled"] = True

        # LQR/Riccati parameters
        if space.q_pos_range is not None:
            config["q_pos"] = self._sample_vector_param(self.rng, space.q_pos_range)
        if space.q_vel_range is not None:
            config["q_vel"] = self._sample_vector_param(self.rng, space.q_vel_range)
        if space.r_thrust_range is not None:
            config["r_thrust"] = self._sample_scalar_param(
                self.rng, space.r_thrust_range
            )
        if space.r_rate_range is not None:
            config["r_rate"] = self._sample_scalar_param(self.rng, space.r_rate_range)

        # Riccati-specific: r_controls (4D vector)
        if space.r_controls_range is not None:
            config["r_controls"] = self._sample_4d_vector_param(
                self.rng, space.r_controls_range
            )

        return config

    def _generate_grid_configs(self) -> list[dict]:
        """Generate grid of controller configurations."""
        space = self.config.search_space
        n = self.config.grid_points_per_dim

        # Build parameter grids
        param_grids: dict[str, list[list[float]] | list[float]] = {}

        def make_vector_grid(
            range_def: tuple[list[float], list[float]],
        ) -> list[list[float]]:
            """Create grid points for a vector parameter (any dimension)."""
            min_vals, max_vals = range_def
            grids = []
            for lo, hi in zip(min_vals, max_vals):
                if np.isclose(lo, hi):
                    grids.append([lo])
                else:
                    grids.append(list(np.linspace(lo, hi, n)))

            # Safeguard against combinatorial explosion for high-dimensional vectors
            num_combinations = np.prod([len(g) for g in grids])
            if num_combinations > 1024:  # A reasonable limit
                logger.warning(
                    "Grid search for a vector parameter would generate "
                    "%d combinations, which may be excessive. Consider "
                    "reducing grid_points_per_dim or using random search.",
                    num_combinations,
                )

            # Cartesian product of axis grids
            return [list(combo) for combo in product(*grids)]

        def make_scalar_grid(range_def: tuple[float, float]) -> list[float]:
            """Create grid points for a scalar parameter."""
            lo, hi = range_def
            if np.isclose(lo, hi):
                return [lo]
            return list(np.linspace(lo, hi, n))

        # Build grids for each active parameter
        if space.kp_pos_range is not None:
            param_grids["kp_pos"] = make_vector_grid(space.kp_pos_range)
        if space.ki_pos_range is not None:
            param_grids["ki_pos"] = make_vector_grid(space.ki_pos_range)
        if space.kd_pos_range is not None:
            param_grids["kd_pos"] = make_vector_grid(space.kd_pos_range)
        if space.ff_velocity_gain_range is not None:
            param_grids["ff_velocity_gain"] = make_vector_grid(
                space.ff_velocity_gain_range
            )
        if space.ff_acceleration_gain_range is not None:
            param_grids["ff_acceleration_gain"] = make_vector_grid(
                space.ff_acceleration_gain_range
            )
        if space.q_pos_range is not None:
            param_grids["q_pos"] = make_vector_grid(space.q_pos_range)
        if space.q_vel_range is not None:
            param_grids["q_vel"] = make_vector_grid(space.q_vel_range)
        if space.r_thrust_range is not None:
            param_grids["r_thrust"] = make_scalar_grid(space.r_thrust_range)
        if space.r_rate_range is not None:
            param_grids["r_rate"] = make_scalar_grid(space.r_rate_range)
        # Riccati-specific: r_controls (4D vector)
        if space.r_controls_range is not None:
            param_grids["r_controls"] = make_vector_grid(space.r_controls_range)

        if not param_grids:
            return []

        # Generate all combinations
        param_names = list(param_grids.keys())
        param_values = [param_grids[name] for name in param_names]

        configs = []
        for combo in product(*param_values):
            config = {}
            for name, value in zip(param_names, combo):
                config[name] = value
            # Add feedforward_enabled if feedforward gains are being tuned
            if "ff_velocity_gain" in config or "ff_acceleration_gain" in config:
                config["feedforward_enabled"] = True
            configs.append(config)

        return configs

    def _create_controller(self, controller_config: dict):
        """Create controller with given configuration."""
        if self.config.controller_type == "pid":
            PIDController = _get_pid_controller()
            return PIDController(config=controller_config)
        elif self.config.controller_type == "lqr":
            LQRController = _get_lqr_controller()
            return LQRController(config=controller_config)
        elif self.config.controller_type == "riccati_lqr":
            RiccatiLQRController = _get_riccati_lqr_controller()
            return RiccatiLQRController(config=controller_config)
        else:
            raise ValueError(f"Unknown controller type: {self.config.controller_type}")

    def _evaluate_config(self, controller_config: dict) -> tuple[float, dict]:
        """
        Evaluate a controller configuration.

        Args:
            controller_config: Controller configuration to evaluate.

        Returns:
            Tuple of (score, metrics) where score is higher-is-better.
        """
        # Create controller
        controller = self._create_controller(controller_config)

        # Create environment
        env_config = EnvConfig()
        env_config.simulation.max_episode_time = self.config.episode_length
        env_config.target.motion_type = self.config.target_motion_type
        env_config.success_criteria.target_radius = self.config.target_radius

        criteria = SuccessCriteria(
            min_on_target_ratio=0.8,
            min_episode_duration=self.config.episode_length,
            target_radius=self.config.target_radius,
        )

        # Run evaluation episodes
        all_metrics = []
        for ep in range(self.config.evaluation_episodes):
            env = QuadcopterEnv(config=env_config)
            episode_seed = self.config.seed + ep
            obs = env.reset(seed=episode_seed)

            # Reset controller state
            controller.reset()

            episode_data = []
            done = False
            step = 0

            while not done and step < self.config.evaluation_horizon:
                action = controller.compute_action(obs)
                next_obs, reward, done, info = env.step(action)

                episode_data.append(
                    {
                        "time": info.get("time", step * env.dt),
                        "quadcopter_position": obs["quadcopter"]["position"].tolist(),
                        "target_position": obs["target"]["position"].tolist(),
                        "action": [
                            action["thrust"],
                            action["roll_rate"],
                            action["pitch_rate"],
                            action["yaw_rate"],
                        ],
                        "tracking_error": info.get("tracking_error", 0.0),
                        "on_target": info.get("on_target", False),
                    }
                )

                obs = next_obs
                step += 1

            # Compute metrics for this episode
            metrics = compute_episode_metrics(episode_data, criteria, info)
            all_metrics.append(metrics)

        # Aggregate metrics across episodes
        mean_on_target = np.mean([m.on_target_ratio for m in all_metrics])
        mean_error = np.mean([m.mean_tracking_error for m in all_metrics])
        success_rate = np.mean([float(m.success) for m in all_metrics])

        # Score: prioritize on-target ratio, penalize error
        # Higher score is better
        score = mean_on_target - 0.1 * mean_error

        metrics_dict = {
            "mean_on_target_ratio": float(mean_on_target),
            "mean_tracking_error": float(mean_error),
            "success_rate": float(success_rate),
            "episodes_evaluated": self.config.evaluation_episodes,
        }

        return score, metrics_dict

    def tune(self) -> TuningResult:
        """
        Run controller auto-tuning.

        Returns:
            TuningResult with best configuration and all results.
        """
        logger.info(
            "Starting %s auto-tuning with %s strategy",
            self.config.controller_type.upper(),
            self.config.strategy,
        )
        logger.info(
            "Search space: %s", self.config.search_space.get_active_parameters()
        )
        logger.info("Max iterations: %d", self.config.max_iterations)

        # Resume from previous results if specified
        if self.config.resume_from:
            self._resume_from_file(self.config.resume_from)

        # Setup signal handlers for graceful interruption
        self._setup_signal_handlers()

        try:
            if self.config.strategy == "grid":
                self._run_grid_search()
            elif self.config.strategy == "cma_es":
                self._run_cma_es_search()
            else:
                self._run_random_search()
        finally:
            self._restore_signal_handlers()
            # Save CMA-ES checkpoint on exit if applicable
            if self.config.strategy == "cma_es" and self.cma_es is not None:
                self._save_cma_checkpoint()

        # Create result
        result = TuningResult(
            best_config=self.best_config,
            best_score=self.best_score,
            best_metrics=self.best_metrics,
            all_results=self.results,
            iterations_completed=len(self.results),
            interrupted=self.interrupted,
            timestamp=datetime.now(timezone.utc).isoformat(),
            config=self.config.to_dict(),
        )

        # Save results
        self._save_results(result)

        return result

    def _resume_from_file(self, path: str) -> None:
        """
        Resume tuning from previous results file.

        Args:
            path: Path to previous results file.

        Raises:
            ValueError: If the path is invalid or contains traversal sequences.
            FileNotFoundError: If the file does not exist.
        """
        # Validate path before loading
        _validate_path(Path(path))
        logger.info("Resuming from %s", path)
        prev_result = TuningResult.load(path)

        self.results = prev_result.all_results
        self.best_config = prev_result.best_config
        self.best_score = prev_result.best_score
        self.best_metrics = prev_result.best_metrics

        logger.info(
            "Loaded %d previous results, best score: %.4f",
            len(self.results),
            self.best_score,
        )

    def _run_grid_search(self) -> None:
        """Run grid search over parameter space."""
        configs = self._generate_grid_configs()
        total = min(len(configs), self.config.max_iterations)

        logger.info(
            "Grid search: %d total configurations, evaluating %d",
            len(configs),
            total,
        )

        for i, config in enumerate(configs[:total]):
            if self.interrupted:
                logger.info("Interrupted after %d iterations", i)
                break

            logger.info(
                "Iteration %d/%d: evaluating %s",
                i + 1,
                total,
                self._format_config_summary(config),
            )

            score, metrics = self._evaluate_config(config)
            self._record_result(config, score, metrics)

            logger.info(
                "  Score: %.4f, On-target: %.1f%%, Error: %.3fm",
                score,
                metrics["mean_on_target_ratio"] * 100,
                metrics["mean_tracking_error"],
            )

    def _run_random_search(self) -> None:
        """Run random search over parameter space."""
        start_idx = len(self.results)
        remaining = self.config.max_iterations - start_idx

        logger.info("Random search: %d iterations remaining", remaining)

        for i in range(remaining):
            if self.interrupted:
                logger.info("Interrupted after %d iterations", start_idx + i)
                break

            config = self._generate_random_config()

            logger.info(
                "Iteration %d/%d: evaluating %s",
                start_idx + i + 1,
                self.config.max_iterations,
                self._format_config_summary(config),
            )

            score, metrics = self._evaluate_config(config)
            self._record_result(config, score, metrics)

            logger.info(
                "  Score: %.4f, On-target: %.1f%%, Error: %.3fm",
                score,
                metrics["mean_on_target_ratio"] * 100,
                metrics["mean_tracking_error"],
            )

    def _record_result(self, config: dict, score: float, metrics: dict) -> None:
        """Record evaluation result and update best."""
        result = {
            "config": config,
            "score": score,
            "metrics": metrics,
        }
        self.results.append(result)

        if score > self.best_score:
            self.best_score = score
            self.best_config = config
            self.best_metrics = metrics
            logger.info("  ** New best! Score: %.4f **", score)

    def _format_config_summary(self, config: dict) -> str:
        """Format configuration for logging."""
        parts = []
        for key, value in config.items():
            if isinstance(value, list):
                formatted = "[" + ", ".join(f"{v:.3f}" for v in value) + "]"
            elif isinstance(value, float):
                formatted = f"{value:.3f}"
            else:
                formatted = str(value)
            parts.append(f"{key}={formatted}")
        return ", ".join(parts[:3]) + ("..." if len(parts) > 3 else "")

    def _save_results(self, result: TuningResult) -> None:
        """Save tuning results to files."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        base_name = f"tuning_{self.config.controller_type}_{timestamp}"

        # Save full results
        results_path = self.output_dir / f"{base_name}_results.json"
        result.save(results_path)
        logger.info("Saved full results to %s", results_path)

        # Save best config as standalone file for easy loading
        best_config_path = self.output_dir / f"{base_name}_best_config.json"
        with open(best_config_path, "w") as f:
            json.dump(
                {
                    "controller_type": self.config.controller_type,
                    self.config.controller_type: result.best_config,
                    "metrics": result.best_metrics,
                    "score": result.best_score,
                },
                f,
                indent=2,
            )
        logger.info("Saved best config to %s", best_config_path)

        # Log summary
        logger.info(
            "\n=== TUNING COMPLETE ===\n"
            "Controller: %s\n"
            "Strategy: %s\n"
            "Iterations: %d\n"
            "Best Score: %.4f\n"
            "Best On-Target: %.1f%%\n"
            "Best Error: %.3fm\n"
            "Best Config: %s\n"
            "Results saved to: %s",
            self.config.controller_type.upper(),
            self.config.strategy,
            result.iterations_completed,
            result.best_score,
            result.best_metrics.get("mean_on_target_ratio", 0) * 100,
            result.best_metrics.get("mean_tracking_error", 0),
            result.best_config,
            self.output_dir,
        )

    # =========================================================================
    # CMA-ES Specific Methods
    # =========================================================================

    def _build_cma_param_spec(self) -> list[tuple[str, int, tuple]]:
        """
        Build parameter specification for CMA-ES optimization.

        Returns a list of tuples (param_name, dimension, (min_bounds, max_bounds)).
        This defines the mapping between the flat CMA-ES vector and controller config.

        Returns:
            List of (name, dimension, (min_list, max_list)) tuples.
        """
        space = self.config.search_space
        spec: list[tuple[str, int, tuple]] = []

        # PID parameters (3D vectors)
        if space.kp_pos_range is not None:
            spec.append(("kp_pos", 3, space.kp_pos_range))
        if space.ki_pos_range is not None:
            spec.append(("ki_pos", 3, space.ki_pos_range))
        if space.kd_pos_range is not None:
            spec.append(("kd_pos", 3, space.kd_pos_range))

        # Feedforward parameters (3D vectors)
        if space.ff_velocity_gain_range is not None:
            spec.append(("ff_velocity_gain", 3, space.ff_velocity_gain_range))
        if space.ff_acceleration_gain_range is not None:
            spec.append(("ff_acceleration_gain", 3, space.ff_acceleration_gain_range))

        # LQR/Riccati parameters (3D vectors)
        if space.q_pos_range is not None:
            spec.append(("q_pos", 3, space.q_pos_range))
        if space.q_vel_range is not None:
            spec.append(("q_vel", 3, space.q_vel_range))

        # Scalar LQR parameters
        if space.r_thrust_range is not None:
            spec.append(("r_thrust", 1, space.r_thrust_range))
        if space.r_rate_range is not None:
            spec.append(("r_rate", 1, space.r_rate_range))

        # Riccati-specific (4D vector)
        if space.r_controls_range is not None:
            spec.append(("r_controls", 4, space.r_controls_range))

        return spec

    def _get_cma_bounds(self) -> tuple[list[float], list[float]]:
        """
        Get lower and upper bounds for all CMA-ES parameters.

        Returns:
            Tuple of (lower_bounds, upper_bounds) as flat lists.
        """
        lower: list[float] = []
        upper: list[float] = []

        for name, dim, bounds in self._cma_param_spec:
            if dim == 1:
                # Scalar parameter
                lower.append(bounds[0])
                upper.append(bounds[1])
            else:
                # Vector parameter
                lower.extend(bounds[0])
                upper.extend(bounds[1])

        return lower, upper

    def _get_cma_x0(self) -> list[float]:
        """
        Get initial point for CMA-ES.

        Uses geometric mean for log-scale parameters (e.g., Q/R weights) and
        arithmetic mean for others.

        Returns:
            Initial parameter vector for the search.
        """
        lower, upper = self._get_cma_bounds()
        x0 = []
        idx = 0
        log_scale_params = {"q_pos", "q_vel", "r_controls"}

        for name, dim, _ in self._cma_param_spec:
            is_log_scale = name in log_scale_params
            for i in range(dim):
                lo = lower[idx]
                hi = upper[idx]
                if is_log_scale and lo > 0 and hi > 0:
                    # Geometric mean for log-scale parameters
                    x0.append(np.sqrt(lo * hi))
                else:
                    # Arithmetic mean for linear-scale parameters
                    x0.append((lo + hi) / 2.0)
                idx += 1
        return x0

    def _vector_to_config(self, x: list[float]) -> dict:
        """
        Convert flat CMA-ES parameter vector to controller configuration.

        Args:
            x: Flat parameter vector from CMA-ES.

        Returns:
            Controller configuration dictionary.
        """
        config: dict = {}
        idx = 0

        for name, dim, _ in self._cma_param_spec:
            if dim == 1:
                config[name] = float(x[idx])
                idx += 1
            else:
                config[name] = [float(x[idx + i]) for i in range(dim)]
                idx += dim

        # Add feedforward_enabled flag if feedforward gains are present
        if "ff_velocity_gain" in config or "ff_acceleration_gain" in config:
            config["feedforward_enabled"] = True

        return config

    def _cma_objective(self, x: list[float]) -> float:
        """
        CMA-ES objective function (minimization).

        CMA-ES minimizes, but we want to maximize score, so we return -score.

        Args:
            x: Parameter vector from CMA-ES.

        Returns:
            Negative score (for minimization).
        """
        config = self._vector_to_config(x)
        score, metrics = self._evaluate_config(config)
        self._record_result(config, score, metrics)

        logger.info(
            "  Score: %.4f, On-target: %.1f%%, Error: %.3fm",
            score,
            metrics["mean_on_target_ratio"] * 100,
            metrics["mean_tracking_error"],
        )

        # Return negative score since CMA-ES minimizes
        return -score

    def _run_cma_es_search(self) -> None:
        """Run CMA-ES optimization over parameter space."""
        import cma

        # Build parameter specification
        self._cma_param_spec = self._build_cma_param_spec()

        if not self._cma_param_spec:
            logger.warning("No parameters to tune. CMA-ES requires at least one range.")
            return

        # Get bounds and initial point
        lower, upper = self._get_cma_bounds()
        x0 = self._get_cma_x0()
        dim = len(x0)

        logger.info(
            "CMA-ES search: %d parameters, sigma0=%.3f", dim, self.config.cma_sigma0
        )

        # Calculate scaled sigma based on parameter ranges
        # Use sigma0 as a fraction of the range
        ranges = [hi - lo for lo, hi in zip(lower, upper) if hi > lo]
        if not ranges:
            # All parameters are fixed, though this should be caught earlier.
            # Set a small default sigma to avoid errors.
            sigma0 = 1e-5
        else:
            # Use geometric mean of ranges to be robust to different scales
            sigma0 = self.config.cma_sigma0 * float(np.exp(np.mean(np.log(ranges))))

        # CMA-ES options
        opts: dict[str, Any] = {
            "seed": self.config.seed,
            "bounds": [lower, upper],
            "maxfevals": self.config.max_iterations,
            "verbose": -9,  # Suppress CMA-ES logging (we do our own)
            "tolfun": 1e-11,  # Very tight tolerance (we use maxfevals as main limit)
            "tolx": 1e-12,
        }

        if self.config.cma_popsize is not None:
            opts["popsize"] = self.config.cma_popsize

        # Try to resume from checkpoint
        checkpoint_loaded = False
        if self.config.resume_from:
            resume_path = Path(self.config.resume_from)
            checkpoint_path = resume_path.parent / "cma_checkpoint.pkl"
            if checkpoint_path.exists():
                try:
                    with open(checkpoint_path, "rb") as f:
                        self.cma_es = pickle.load(f)
                    logger.info("Resumed CMA-ES state from %s", checkpoint_path)
                    checkpoint_loaded = True
                except Exception as e:
                    logger.warning("Failed to load CMA-ES checkpoint: %s", e)

        if not checkpoint_loaded:
            self.cma_es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

        # Set checkpoint path for saving
        self._cma_checkpoint_path = self.output_dir / "cma_checkpoint.pkl"

        # Run optimization
        # num_evals tracks total objective evaluations (max_iterations is the budget)
        num_evals = len(self.results)
        while not self.cma_es.stop() and num_evals < self.config.max_iterations:
            if self.interrupted:
                logger.info("CMA-ES interrupted after %d evaluations", num_evals)
                break

            # Ask for new solutions
            solutions = self.cma_es.ask()

            # Evaluate solutions
            fitness_values = []
            for i, x in enumerate(solutions):
                if self.interrupted:
                    break

                logger.info(
                    "Evaluation %d/%d: evaluating %s",
                    num_evals + 1,
                    self.config.max_iterations,
                    self._format_config_summary(self._vector_to_config(x)),
                )

                fitness = self._cma_objective(x)
                fitness_values.append(fitness)
                num_evals += 1

                if num_evals >= self.config.max_iterations:
                    break

            # Tell CMA-ES about the fitness values.
            # CMA-ES requires at least mu solutions (typically popsize // 2).
            # If fewer, skip the tell - the results are still recorded.
            mu = self.cma_es.sp.weights.mu
            if len(fitness_values) >= mu:
                self.cma_es.tell(solutions[:len(fitness_values)], fitness_values)
            elif len(fitness_values) > 0:
                # Not enough solutions for CMA-ES update, but we have results
                logger.debug(
                    "Skipping CMA-ES update: %d solutions < mu=%d",
                    len(fitness_values),
                    mu,
                )

        # Log CMA-ES statistics
        if self.cma_es is not None:
            logger.info(
                "CMA-ES completed: %d evaluations, best fitness: %.4f",
                self.cma_es.result.evaluations,
                -self.best_score,  # Convert back to positive
            )

    def _save_cma_checkpoint(self) -> None:
        """Save CMA-ES state for resumption."""
        if self.cma_es is None or self._cma_checkpoint_path is None:
            return

        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            # Use pickle for CMA-ES state
            with open(self._cma_checkpoint_path, "wb") as f:
                pickle.dump(self.cma_es, f)
            logger.info("Saved CMA-ES checkpoint to %s", self._cma_checkpoint_path)
        except Exception as e:
            logger.warning("Failed to save CMA-ES checkpoint: %s", e)
