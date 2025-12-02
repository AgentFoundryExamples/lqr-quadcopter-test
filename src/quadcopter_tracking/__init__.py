"""
Quadcopter Tracking Research Package

A Python-based research repository for quadcopter target-tracking studies
with LQR and ML-based controllers.

Subpackages:
- env: Simulation environment for quadcopter dynamics
- controllers: Control algorithm implementations (LQR, PID, neural)
- utils: Configuration, logging, plotting, and metrics utilities
- eval: Controller evaluation pipeline
"""

import importlib.metadata

try:
    # Retrieve the version from installed package metadata
    __version__ = importlib.metadata.version("quadcopter-tracking")
except importlib.metadata.PackageNotFoundError:
    # Fallback for when the package is not installed, e.g., in an editable install
    __version__ = "0.0.0-dev"

from quadcopter_tracking.controllers import (
    BaseController,
    LQRController,
    PIDController,
)
from quadcopter_tracking.env import QuadcopterEnv, TargetMotion
from quadcopter_tracking.utils import (
    DataLogger,
    EpisodeMetrics,
    EvaluationSummary,
    Plotter,
    SuccessCriteria,
    compute_episode_metrics,
    compute_evaluation_summary,
    get_default_config,
    load_config,
)

__all__ = [
    "QuadcopterEnv",
    "TargetMotion",
    "BaseController",
    "LQRController",
    "PIDController",
    "load_config",
    "get_default_config",
    "DataLogger",
    "Plotter",
    # Metrics
    "EpisodeMetrics",
    "EvaluationSummary",
    "SuccessCriteria",
    "compute_episode_metrics",
    "compute_evaluation_summary",
]
