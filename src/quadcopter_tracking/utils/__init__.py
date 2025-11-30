"""
Quadcopter Tracking Utilities Package

This package provides shared utilities for the quadcopter tracking project:
- Configuration loading (YAML/JSON with environment variable overrides)
- Data logging for experiment tracking
- Plotting utilities for visualization
- Loss functions for training
- Common math/helper functions

Design Philosophy:
- Utilities are stateless where possible
- Configuration supports both file-based and environment variable sources
- Logging captures enough data for post-hoc analysis
"""

import datetime
import json
import logging
import os
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

from .losses import (
    CombinedLoss,
    LossLogger,
    RewardShapingLoss,
    TrackingLoss,
    create_loss_from_config,
)
from .metrics import (
    EpisodeMetrics,
    EvaluationSummary,
    SuccessCriteria,
    compute_control_effort,
    compute_episode_metrics,
    compute_evaluation_summary,
    compute_on_target_ratio,
    compute_tracking_error,
    detect_overshoots,
    format_metrics_report,
)

__all__ = [
    "load_config",
    "DataLogger",
    "Plotter",
    "get_default_config",
    "TrackingLoss",
    "RewardShapingLoss",
    "CombinedLoss",
    "LossLogger",
    "create_loss_from_config",
    # Metrics
    "EpisodeMetrics",
    "EvaluationSummary",
    "SuccessCriteria",
    "compute_tracking_error",
    "compute_on_target_ratio",
    "compute_control_effort",
    "detect_overshoots",
    "compute_episode_metrics",
    "compute_evaluation_summary",
    "format_metrics_report",
]

logger = logging.getLogger(__name__)


def get_default_config() -> dict:
    """
    Get default configuration values.

    Returns sensible defaults for all configuration parameters.
    These defaults are used when no configuration file is provided
    or when specific values are missing.

    Returns:
        Dictionary with default configuration values.
    """
    return {
        "seed": 42,
        "episode_length": 30.0,  # seconds
        "dt": 0.01,  # simulation timestep in seconds
        "target": {
            "radius_requirement": 0.5,  # meters - on-target threshold
            "motion_type": "linear",
            "speed": 1.0,  # meters/second
            "amplitude": 2.0,  # meters for oscillatory motion
            "frequency": 0.5,  # Hz for oscillatory motion
        },
        "quadcopter": {
            "mass": 1.0,  # kg
            "max_thrust": 20.0,  # N
            "max_angular_rate": 3.0,  # rad/s
        },
        "success_criteria": {
            "min_on_target_ratio": 0.8,  # 80% on-target requirement
            "min_episode_duration": 30.0,  # seconds
        },
        "logging": {
            "enabled": True,
            "output_dir": "experiments",
            "log_interval": 10,  # steps between log entries
        },
    }


def load_config(
    config_path: str | Path | None = None,
    load_env: bool = True,
) -> dict:
    """
    Load configuration from file with environment variable overrides.

    Configuration loading follows this priority (highest to lowest):
    1. Environment variables (from .env file or system)
    2. Config file (YAML or JSON)
    3. Default values

    Environment variables override config file values using a naming convention:
    - QUADCOPTER_SEED -> config["seed"]
    - QUADCOPTER_EPISODE_LENGTH -> config["episode_length"]
    - QUADCOPTER_TARGET_RADIUS -> config["target"]["radius_requirement"]
    - QUADCOPTER_TARGET_MOTION_TYPE -> config["target"]["motion_type"]
    - QUADCOPTER_TARGET_SPEED -> config["target"]["speed"]

    Args:
        config_path: Path to YAML or JSON configuration file.
                    If None, only defaults and env vars are used.
        load_env: Whether to load .env file and apply env var overrides.

    Returns:
        Merged configuration dictionary.

    Raises:
        FileNotFoundError: If config_path is specified but file doesn't exist.
        PermissionError: If config file cannot be read.
        ValueError: If config file format is unsupported or malformed.
    """
    # Start with defaults
    config = get_default_config()

    # Load from file if provided
    if config_path is not None:
        config_path = Path(config_path)

        # Validate file exists and is readable
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        if not config_path.is_file():
            raise ValueError(f"Configuration path is not a file: {config_path}")

        try:
            with open(config_path) as f:
                if config_path.suffix in (".yaml", ".yml"):
                    file_config = yaml.safe_load(f)
                elif config_path.suffix == ".json":
                    file_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {config_path.suffix}")
        except PermissionError as e:
            raise PermissionError(
                f"Cannot read configuration file: {config_path}"
            ) from e
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ValueError(f"Malformed configuration file: {config_path}") from e

        if file_config:
            config = _deep_merge(config, file_config)

    # Apply environment variable overrides
    if load_env:
        load_dotenv()
        config = _apply_env_overrides(config)

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries.

    Args:
        base: Base dictionary.
        override: Override dictionary (values take precedence).

    Returns:
        Merged dictionary.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _json_serializer(obj):
    """
    Custom JSON serializer for objects not serializable by default json.dump.

    Handles common types found in experiment logging:
    - numpy arrays -> lists
    - datetime objects -> ISO format strings
    - Path objects -> strings

    Args:
        obj: Object to serialize.

    Returns:
        JSON-serializable representation.

    Raises:
        TypeError: If object type is not supported.
    """
    # Handle numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # Handle datetime objects
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    # Handle Path objects
    if isinstance(obj, Path):
        return str(obj)
    # Raise for unsupported types to catch serialization issues
    raise TypeError("Object is not JSON serializable")


def _apply_env_overrides(config: dict) -> dict:
    """
    Apply environment variable overrides to configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Configuration with env var overrides applied.
    """
    # Simple scalar overrides
    env_mappings = {
        "QUADCOPTER_SEED": ("seed", int),
        "QUADCOPTER_EPISODE_LENGTH": ("episode_length", float),
        "QUADCOPTER_DT": ("dt", float),
    }

    for env_var, (config_key, type_fn) in env_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            try:
                config[config_key] = type_fn(value)
            except ValueError:
                logger.warning(
                    "Invalid value for %s: '%s', using default", env_var, value
                )

    # Nested target overrides
    target_env_mappings = {
        "QUADCOPTER_TARGET_RADIUS": ("radius_requirement", float),
        "QUADCOPTER_TARGET_MOTION_TYPE": ("motion_type", str),
        "QUADCOPTER_TARGET_SPEED": ("speed", float),
        "QUADCOPTER_TARGET_AMPLITUDE": ("amplitude", float),
        "QUADCOPTER_TARGET_FREQUENCY": ("frequency", float),
    }

    for env_var, (config_key, type_fn) in target_env_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            try:
                config["target"][config_key] = type_fn(value)
            except ValueError:
                logger.warning(
                    "Invalid value for %s: '%s', using default", env_var, value
                )

    return config


class DataLogger:
    """
    Data logging utility for experiment tracking.

    Logs environment state, actions, and metrics at configurable intervals
    for post-experiment analysis.

    Attributes:
        output_dir (Path): Directory for log files.
        experiment_name (str): Name of current experiment.
        log_interval (int): Steps between log entries.
    """

    def __init__(
        self,
        output_dir: str | Path = "experiments",
        experiment_name: str | None = None,
        log_interval: int = 10,
    ):
        """
        Initialize data logger.

        Args:
            output_dir: Directory for log output.
            experiment_name: Name for this experiment (auto-generated if None).
            log_interval: Number of steps between log entries.
        """
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name or f"exp_{id(self)}"
        self.log_interval = log_interval
        self.data = []
        self._step_count = 0

    def log(self, state: dict, action: dict, reward: float, info: dict) -> None:
        """
        Log a single step's data.

        Args:
            state: Environment state observation.
            action: Action taken.
            reward: Reward received.
            info: Additional info dictionary.
        """
        self._step_count += 1
        if self._step_count % self.log_interval == 0:
            self.data.append(
                {
                    "step": self._step_count,
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "info": info,
                }
            )

    def save(self) -> Path:
        """
        Save logged data to file.

        Returns:
            Path to saved log file.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.output_dir / f"{self.experiment_name}.json"
        with open(log_path, "w") as f:
            json.dump(self.data, f, indent=2, default=_json_serializer)
        return log_path

    def reset(self) -> None:
        """Reset logger state for new experiment."""
        self.data = []
        self._step_count = 0


class Plotter:
    """
    Plotting utility for experiment visualization.

    Provides standardized plots for tracking experiments:
    - Trajectory plots (quadcopter vs target)
    - Tracking error over time
    - On-target ratio visualization
    - Control effort analysis

    Attributes:
        figsize (tuple): Default figure size.
        style (str): Matplotlib style to use.
    """

    def __init__(self, figsize: tuple[int, int] = (10, 6), style: str = "default"):
        """
        Initialize plotter.

        Args:
            figsize: Default figure size (width, height) in inches.
            style: Matplotlib style name.
        """
        self.figsize = figsize
        self.style = style

    def plot_trajectory(self, data: dict, save_path: str | Path | None = None):
        """
        Plot quadcopter and target trajectories.

        Args:
            data: Dictionary containing trajectory data.
            save_path: Optional path to save figure.
        """
        # Placeholder implementation
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_title("Quadcopter Target Tracking Trajectory")
        ax.legend()
        ax.grid(True)

        if save_path:
            fig.savefig(save_path)
        return fig, ax

    def plot_tracking_error(self, data: dict, save_path: str | Path | None = None):
        """
        Plot tracking error over time.

        Args:
            data: Dictionary containing error data.
            save_path: Optional path to save figure.
        """
        # Placeholder implementation
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Tracking Error (m)")
        ax.set_title("Tracking Error Over Time")
        ax.grid(True)

        if save_path:
            fig.savefig(save_path)
        return fig, ax
