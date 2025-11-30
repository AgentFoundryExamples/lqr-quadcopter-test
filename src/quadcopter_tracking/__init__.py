"""
Quadcopter Tracking Research Package

A Python-based research repository for quadcopter target-tracking studies
with LQR and ML-based controllers.

Subpackages:
- env: Simulation environment for quadcopter dynamics
- controllers: Control algorithm implementations (LQR, PID, neural)
- utils: Configuration, logging, and plotting utilities
"""

__version__ = "0.1.0"

from quadcopter_tracking.controllers import (
    BaseController,
    LQRController,
    PIDController,
)
from quadcopter_tracking.env import QuadcopterEnv, TargetMotion
from quadcopter_tracking.utils import (
    DataLogger,
    Plotter,
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
]
