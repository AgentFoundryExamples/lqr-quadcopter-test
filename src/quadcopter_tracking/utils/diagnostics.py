"""
Diagnostics Module for Training Analysis

This module provides opt-in diagnostic instrumentation for the deep learning
training pipeline. It enables structured logging of observations, actions,
gradients, and metrics without altering default training behavior.

Design Philosophy:
- Feature flags to enable/disable diagnostics with zero overhead when disabled
- Graceful handling of NaN/inf values to surface root causes
- Configurable throttling to prevent log explosion on long runs
- CPU-compatible (headless environments supported)

Usage:
    from quadcopter_tracking.utils.diagnostics import DiagnosticsConfig, Diagnostics

    config = DiagnosticsConfig(
        enabled=True,
        log_observations=True,
        log_gradients=True,
        log_interval=10,
    )
    diag = Diagnostics(config, output_dir="experiments/diagnostics")

    # In training loop:
    diag.log_step(epoch, step, observation, action, loss, gradients)
    diag.log_epoch(epoch, metrics)

    # At end:
    diag.save_summary()
"""

import csv
import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticsConfig:
    """Configuration for training diagnostics.

    Attributes:
        enabled: Master switch for all diagnostics.
        log_observations: Log observation features at each step.
        log_actions: Log action outputs at each step.
        log_gradients: Log gradient norms and statistics.
        log_losses: Log individual loss components.
        log_metrics: Log per-epoch metrics.
        log_interval: Steps between logged entries (throttling).
        max_entries_per_epoch: Maximum entries to log per epoch.
        output_dir: Directory for diagnostic output files.
        generate_plots: Whether to generate diagnostic plots.
        headless: Force headless mode (disable interactive plots).
    """

    enabled: bool = False
    log_observations: bool = True
    log_actions: bool = True
    log_gradients: bool = True
    log_losses: bool = True
    log_metrics: bool = True
    log_interval: int = 10
    max_entries_per_epoch: int = 1000
    output_dir: str = "experiments/diagnostics"
    generate_plots: bool = True
    headless: bool = True

    @classmethod
    def from_dict(cls, config_dict: dict) -> "DiagnosticsConfig":
        """Create config from dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "enabled": self.enabled,
            "log_observations": self.log_observations,
            "log_actions": self.log_actions,
            "log_gradients": self.log_gradients,
            "log_losses": self.log_losses,
            "log_metrics": self.log_metrics,
            "log_interval": self.log_interval,
            "max_entries_per_epoch": self.max_entries_per_epoch,
            "output_dir": self.output_dir,
            "generate_plots": self.generate_plots,
            "headless": self.headless,
        }


@dataclass
class GradientStats:
    """Statistics for gradient analysis.

    Attributes:
        norm: L2 norm of all gradients.
        max_abs: Maximum absolute gradient value.
        min_abs: Minimum absolute gradient value.
        mean: Mean gradient value.
        std: Standard deviation of gradients.
        num_nan: Count of NaN values.
        num_inf: Count of Inf values.
        per_layer: Per-layer gradient norms.
    """

    norm: float = 0.0
    max_abs: float = 0.0
    min_abs: float = float("inf")
    mean: float = 0.0
    std: float = 0.0
    num_nan: int = 0
    num_inf: int = 0
    per_layer: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "norm": _safe_float(self.norm),
            "max_abs": _safe_float(self.max_abs),
            "min_abs": _safe_float(self.min_abs),
            "mean": _safe_float(self.mean),
            "std": _safe_float(self.std),
            "num_nan": self.num_nan,
            "num_inf": self.num_inf,
            "per_layer": {k: _safe_float(v) for k, v in self.per_layer.items()},
        }


@dataclass
class StepDiagnostics:
    """Diagnostics for a single training step.

    Attributes:
        epoch: Current epoch number.
        step: Step within epoch.
        observation_stats: Statistics for observation features.
        action_stats: Statistics for action outputs.
        loss_components: Individual loss values.
        gradient_stats: Gradient statistics.
        tracking_error: Current tracking error.
        on_target: Whether quadcopter is on target.
    """

    epoch: int
    step: int
    observation_stats: dict[str, float] = field(default_factory=dict)
    action_stats: dict[str, float] = field(default_factory=dict)
    loss_components: dict[str, float] = field(default_factory=dict)
    gradient_stats: GradientStats | None = None
    tracking_error: float = 0.0
    on_target: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "epoch": self.epoch,
            "step": self.step,
            "observation_stats": {
                k: _safe_float(v) for k, v in self.observation_stats.items()
            },
            "action_stats": {k: _safe_float(v) for k, v in self.action_stats.items()},
            "loss_components": {
                k: _safe_float(v) for k, v in self.loss_components.items()
            },
            "gradient_stats": (
                self.gradient_stats.to_dict() if self.gradient_stats else None
            ),
            "tracking_error": _safe_float(self.tracking_error),
            "on_target": self.on_target,
        }


@dataclass
class EpochDiagnostics:
    """Diagnostics for a training epoch.

    Attributes:
        epoch: Epoch number.
        mean_loss: Mean total loss.
        mean_tracking_error: Mean tracking error.
        mean_on_target_ratio: Mean on-target ratio.
        mean_gradient_norm: Mean gradient norm.
        max_gradient_norm: Maximum gradient norm.
        action_magnitude_mean: Mean action magnitude.
        action_magnitude_std: Std of action magnitude.
        observation_range: Range of observation values.
        num_nan_gradients: Count of steps with NaN gradients.
        num_inf_gradients: Count of steps with Inf gradients.
        loss_breakdown: Mean of each loss component.
    """

    epoch: int
    mean_loss: float = 0.0
    mean_tracking_error: float = 0.0
    mean_on_target_ratio: float = 0.0
    mean_gradient_norm: float = 0.0
    max_gradient_norm: float = 0.0
    action_magnitude_mean: float = 0.0
    action_magnitude_std: float = 0.0
    observation_range: tuple[float, float] = (0.0, 0.0)
    num_nan_gradients: int = 0
    num_inf_gradients: int = 0
    loss_breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "epoch": self.epoch,
            "mean_loss": _safe_float(self.mean_loss),
            "mean_tracking_error": _safe_float(self.mean_tracking_error),
            "mean_on_target_ratio": _safe_float(self.mean_on_target_ratio),
            "mean_gradient_norm": _safe_float(self.mean_gradient_norm),
            "max_gradient_norm": _safe_float(self.max_gradient_norm),
            "action_magnitude_mean": _safe_float(self.action_magnitude_mean),
            "action_magnitude_std": _safe_float(self.action_magnitude_std),
            "observation_range": (
                _safe_float(self.observation_range[0]),
                _safe_float(self.observation_range[1]),
            ),
            "num_nan_gradients": self.num_nan_gradients,
            "num_inf_gradients": self.num_inf_gradients,
            "loss_breakdown": {
                k: _safe_float(v) for k, v in self.loss_breakdown.items()
            },
        }


def _safe_float(value: float | int | np.floating) -> float:
    """Convert value to float, handling NaN/Inf gracefully.

    Args:
        value: Value to convert.

    Returns:
        Float value, with NaN/Inf converted to string markers.
    """
    if isinstance(value, (np.floating, np.integer)):
        value = float(value)
    if math.isnan(value):
        return float("nan")
    if math.isinf(value):
        return float("inf") if value > 0 else float("-inf")
    return float(value)


def compute_gradient_stats(
    parameters: list[torch.nn.Parameter], named: bool = False
) -> GradientStats:
    """Compute gradient statistics from model parameters.

    Args:
        parameters: List of model parameters.
        named: Whether parameters are (name, param) tuples.

    Returns:
        GradientStats object with computed statistics.
    """
    stats = GradientStats()
    all_grads = []
    per_layer = {}

    params_iter = parameters if not named else parameters

    for item in params_iter:
        if named:
            name, param = item
        else:
            param = item
            name = f"param_{len(per_layer)}"

        if param.grad is None:
            continue

        grad = param.grad.detach()
        grad_flat = grad.flatten()

        # Count NaN/Inf
        num_nan = torch.isnan(grad_flat).sum().item()
        num_inf = torch.isinf(grad_flat).sum().item()
        stats.num_nan += num_nan
        stats.num_inf += num_inf

        # Filter out NaN/Inf for statistics
        valid_grads = grad_flat[torch.isfinite(grad_flat)]
        if len(valid_grads) > 0:
            all_grads.append(valid_grads)
            per_layer[name] = torch.norm(valid_grads).item()

    if all_grads:
        all_grads_cat = torch.cat(all_grads)
        stats.norm = torch.norm(all_grads_cat).item()
        stats.max_abs = torch.max(torch.abs(all_grads_cat)).item()
        stats.min_abs = torch.min(torch.abs(all_grads_cat)).item()
        stats.mean = torch.mean(all_grads_cat).item()
        stats.std = torch.std(all_grads_cat).item()
        stats.per_layer = per_layer

    return stats


def compute_observation_stats(observation: np.ndarray | torch.Tensor) -> dict:
    """Compute statistics for observation features.

    Args:
        observation: Observation array or tensor.

    Returns:
        Dictionary of statistics.
    """
    if isinstance(observation, torch.Tensor):
        obs = observation.detach().cpu().numpy()
    else:
        obs = np.asarray(observation)

    obs_flat = obs.flatten()

    # Handle NaN/Inf
    num_nan = np.sum(np.isnan(obs_flat))
    num_inf = np.sum(np.isinf(obs_flat))
    valid_obs = obs_flat[np.isfinite(obs_flat)]

    if len(valid_obs) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "num_nan": int(num_nan),
            "num_inf": int(num_inf),
        }

    return {
        "mean": float(np.mean(valid_obs)),
        "std": float(np.std(valid_obs)),
        "min": float(np.min(valid_obs)),
        "max": float(np.max(valid_obs)),
        "num_nan": int(num_nan),
        "num_inf": int(num_inf),
    }


def compute_action_stats(action: dict | np.ndarray | torch.Tensor) -> dict:
    """Compute statistics for action outputs.

    Args:
        action: Action dictionary, array, or tensor.

    Returns:
        Dictionary of statistics.
    """
    if isinstance(action, dict):
        action_values = np.array(
            [
                action.get("thrust", 0.0),
                action.get("roll_rate", 0.0),
                action.get("pitch_rate", 0.0),
                action.get("yaw_rate", 0.0),
            ]
        )
    elif isinstance(action, torch.Tensor):
        action_values = action.detach().cpu().numpy()
    else:
        action_values = np.asarray(action)

    action_flat = action_values.flatten()

    return {
        "thrust_mean": float(action_flat[0]) if len(action_flat) > 0 else 0.0,
        "rate_magnitude": float(np.linalg.norm(action_flat[1:4]))
        if len(action_flat) > 3
        else 0.0,
        "total_magnitude": float(np.linalg.norm(action_flat)),
        "min": float(np.min(action_flat)),
        "max": float(np.max(action_flat)),
    }


class Diagnostics:
    """Diagnostics manager for training analysis.

    Collects, aggregates, and exports diagnostic information during training.
    Designed to have minimal overhead when disabled.

    Attributes:
        config: Diagnostics configuration.
        output_dir: Output directory for diagnostic files.
        step_log: List of step diagnostics.
        epoch_log: List of epoch diagnostics.
    """

    def __init__(
        self,
        config: DiagnosticsConfig | None = None,
        output_dir: str | Path | None = None,
    ):
        """Initialize diagnostics manager.

        Args:
            config: Diagnostics configuration.
            output_dir: Override for output directory.
        """
        self.config = config or DiagnosticsConfig()
        self.output_dir = Path(output_dir or self.config.output_dir)

        # Storage
        self.step_log: list[StepDiagnostics] = []
        self.epoch_log: list[EpochDiagnostics] = []
        self._current_epoch_steps: list[StepDiagnostics] = []
        self._step_counter = 0
        self._epoch_entry_count = 0

        # Create output directory if enabled
        if self.config.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set matplotlib backend for headless mode
        if self.config.headless:
            os.environ.setdefault("MPLBACKEND", "Agg")

    @property
    def enabled(self) -> bool:
        """Check if diagnostics are enabled."""
        return self.config.enabled

    def log_step(
        self,
        epoch: int,
        step: int,
        observation: np.ndarray | torch.Tensor | None = None,
        action: dict | np.ndarray | torch.Tensor | None = None,
        losses: dict[str, float | torch.Tensor] | None = None,
        parameters: list[torch.nn.Parameter] | None = None,
        tracking_error: float = 0.0,
        on_target: bool = False,
    ) -> StepDiagnostics | None:
        """Log diagnostics for a single training step.

        Args:
            epoch: Current epoch number.
            step: Step within epoch.
            observation: Observation features.
            action: Action output.
            losses: Loss component values.
            parameters: Model parameters (for gradient stats).
            tracking_error: Current tracking error.
            on_target: Whether on target.

        Returns:
            StepDiagnostics if logged, None if skipped.
        """
        if not self.config.enabled:
            return None

        self._step_counter += 1

        # Check throttling
        if self._step_counter % self.config.log_interval != 0:
            return None

        # Check max entries
        if self._epoch_entry_count >= self.config.max_entries_per_epoch:
            return None

        self._epoch_entry_count += 1

        # Compute stats
        diag = StepDiagnostics(epoch=epoch, step=step)

        if observation is not None and self.config.log_observations:
            diag.observation_stats = compute_observation_stats(observation)

        if action is not None and self.config.log_actions:
            diag.action_stats = compute_action_stats(action)

        if losses is not None and self.config.log_losses:
            diag.loss_components = {
                k: v.item() if isinstance(v, torch.Tensor) else float(v)
                for k, v in losses.items()
            }

        if parameters is not None and self.config.log_gradients:
            diag.gradient_stats = compute_gradient_stats(parameters)

        diag.tracking_error = tracking_error
        diag.on_target = on_target

        self._current_epoch_steps.append(diag)
        self.step_log.append(diag)

        return diag

    def log_epoch(
        self, epoch: int, metrics: dict[str, Any] | None = None
    ) -> EpochDiagnostics:
        """Log diagnostics for an epoch and aggregate step data.

        Args:
            epoch: Epoch number.
            metrics: Optional epoch-level metrics.

        Returns:
            EpochDiagnostics object.
        """
        diag = EpochDiagnostics(epoch=epoch)

        if not self.config.enabled:
            self._reset_epoch()
            return diag

        # Aggregate step data
        if self._current_epoch_steps:
            losses = [
                s.loss_components.get("total", 0.0) for s in self._current_epoch_steps
            ]
            errors = [s.tracking_error for s in self._current_epoch_steps]
            on_targets = [s.on_target for s in self._current_epoch_steps]

            diag.mean_loss = np.mean(losses) if losses else 0.0
            diag.mean_tracking_error = np.mean(errors) if errors else 0.0
            diag.mean_on_target_ratio = np.mean(on_targets) if on_targets else 0.0

            # Gradient stats
            grad_norms = [
                s.gradient_stats.norm
                for s in self._current_epoch_steps
                if s.gradient_stats
            ]
            if grad_norms:
                diag.mean_gradient_norm = np.mean(grad_norms)
                diag.max_gradient_norm = np.max(grad_norms)

            # Count NaN/Inf gradients
            diag.num_nan_gradients = sum(
                1
                for s in self._current_epoch_steps
                if s.gradient_stats and s.gradient_stats.num_nan > 0
            )
            diag.num_inf_gradients = sum(
                1
                for s in self._current_epoch_steps
                if s.gradient_stats and s.gradient_stats.num_inf > 0
            )

            # Action stats
            action_mags = [
                s.action_stats.get("total_magnitude", 0.0)
                for s in self._current_epoch_steps
                if s.action_stats
            ]
            if action_mags:
                diag.action_magnitude_mean = np.mean(action_mags)
                diag.action_magnitude_std = np.std(action_mags)

            # Observation range
            obs_mins = [
                s.observation_stats.get("min", 0.0)
                for s in self._current_epoch_steps
                if s.observation_stats
            ]
            obs_maxs = [
                s.observation_stats.get("max", 0.0)
                for s in self._current_epoch_steps
                if s.observation_stats
            ]
            if obs_mins and obs_maxs:
                diag.observation_range = (np.min(obs_mins), np.max(obs_maxs))

            # Loss breakdown
            loss_keys = set()
            for s in self._current_epoch_steps:
                loss_keys.update(s.loss_components.keys())
            for key in loss_keys:
                values = [
                    s.loss_components.get(key, 0.0) for s in self._current_epoch_steps
                ]
                diag.loss_breakdown[key] = np.mean(values)

        # Add external metrics
        if metrics and self.config.log_metrics:
            if "mean_tracking_error" in metrics:
                diag.mean_tracking_error = metrics["mean_tracking_error"]
            if "mean_on_target_ratio" in metrics:
                diag.mean_on_target_ratio = metrics["mean_on_target_ratio"]

        self.epoch_log.append(diag)
        self._reset_epoch()

        return diag

    def _reset_epoch(self) -> None:
        """Reset epoch-level counters."""
        self._current_epoch_steps = []
        self._step_counter = 0
        self._epoch_entry_count = 0

    def save_step_log(self, filename: str = "step_diagnostics.json") -> Path | None:
        """Save step-level diagnostics to JSON file.

        Args:
            filename: Output filename.

        Returns:
            Path to saved file, or None if disabled.
        """
        if not self.config.enabled or not self.step_log:
            return None

        path = self.output_dir / filename
        data = [s.to_dict() for s in self.step_log]

        with open(path, "w") as f:
            json.dump(data, f, indent=2, allow_nan=True)

        logger.info("Saved step diagnostics to %s", path)
        return path

    def save_epoch_log(self, filename: str = "epoch_diagnostics.json") -> Path | None:
        """Save epoch-level diagnostics to JSON file.

        Args:
            filename: Output filename.

        Returns:
            Path to saved file, or None if disabled.
        """
        if not self.config.enabled or not self.epoch_log:
            return None

        path = self.output_dir / filename
        data = [e.to_dict() for e in self.epoch_log]

        with open(path, "w") as f:
            json.dump(data, f, indent=2, allow_nan=True)

        logger.info("Saved epoch diagnostics to %s", path)
        return path

    def save_epoch_csv(self, filename: str = "epoch_diagnostics.csv") -> Path | None:
        """Save epoch-level diagnostics to CSV file.

        Args:
            filename: Output filename.

        Returns:
            Path to saved file, or None if disabled.
        """
        if not self.config.enabled or not self.epoch_log:
            return None

        path = self.output_dir / filename
        fieldnames = [
            "epoch",
            "mean_loss",
            "mean_tracking_error",
            "mean_on_target_ratio",
            "mean_gradient_norm",
            "max_gradient_norm",
            "action_magnitude_mean",
            "action_magnitude_std",
            "num_nan_gradients",
            "num_inf_gradients",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for e in self.epoch_log:
                row = e.to_dict()
                writer.writerow(row)

        logger.info("Saved epoch CSV to %s", path)
        return path

    def save_summary(self, filename: str = "diagnostics_summary.json") -> Path | None:
        """Save summary of all diagnostics.

        Args:
            filename: Output filename.

        Returns:
            Path to saved file, or None if disabled.
        """
        if not self.config.enabled:
            return None

        path = self.output_dir / filename

        summary = {
            "config": self.config.to_dict(),
            "num_epochs": len(self.epoch_log),
            "num_steps_logged": len(self.step_log),
        }

        if self.epoch_log:
            summary["final_epoch"] = self.epoch_log[-1].to_dict()
            summary["mean_loss_trajectory"] = [e.mean_loss for e in self.epoch_log]
            summary["mean_error_trajectory"] = [
                e.mean_tracking_error for e in self.epoch_log
            ]
            summary["gradient_norm_trajectory"] = [
                e.mean_gradient_norm for e in self.epoch_log
            ]

            # Identify potential issues
            issues = []
            for e in self.epoch_log:
                if e.num_nan_gradients > 0:
                    issues.append(
                        f"Epoch {e.epoch}: {e.num_nan_gradients} NaN gradients"
                    )
                if e.num_inf_gradients > 0:
                    issues.append(
                        f"Epoch {e.epoch}: {e.num_inf_gradients} Inf gradients"
                    )
                if e.mean_gradient_norm < 1e-7:
                    issues.append(
                        f"Epoch {e.epoch}: Very small gradient norm "
                        f"({e.mean_gradient_norm:.2e})"
                    )
                if e.mean_gradient_norm > 100:
                    issues.append(
                        f"Epoch {e.epoch}: Large gradient norm "
                        f"({e.mean_gradient_norm:.2f})"
                    )

            summary["identified_issues"] = issues

        with open(path, "w") as f:
            json.dump(summary, f, indent=2, allow_nan=True)

        logger.info("Saved diagnostics summary to %s", path)
        return path

    def generate_plots(self, prefix: str = "diag") -> list[Path]:
        """Generate diagnostic plots if enabled.

        Args:
            prefix: Filename prefix for plots.

        Returns:
            List of paths to generated plots.
        """
        if not self.config.enabled or not self.config.generate_plots:
            return []

        if not self.epoch_log:
            return []

        try:
            import matplotlib

            if self.config.headless:
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, skipping plots")
            return []

        paths = []
        epochs = [e.epoch for e in self.epoch_log]

        # Loss trajectory plot
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            losses = [e.mean_loss for e in self.epoch_log]
            ax.plot(epochs, losses, "b-", label="Mean Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Training Loss Trajectory")
            ax.legend()
            ax.grid(True, alpha=0.3)
            path = self.output_dir / f"{prefix}_loss_trajectory.png"
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)
        except Exception as e:
            logger.warning("Failed to generate loss plot: %s", e)

        # Tracking error trajectory plot
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            errors = [e.mean_tracking_error for e in self.epoch_log]
            ax.plot(epochs, errors, "r-", label="Mean Tracking Error")
            ax.axhline(y=0.5, color="g", linestyle="--", label="Target Radius (0.5m)")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Tracking Error (m)")
            ax.set_title("Tracking Error Trajectory")
            ax.legend()
            ax.grid(True, alpha=0.3)
            path = self.output_dir / f"{prefix}_error_trajectory.png"
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)
        except Exception as e:
            logger.warning("Failed to generate error plot: %s", e)

        # Gradient norm trajectory plot
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            grad_norms = [e.mean_gradient_norm for e in self.epoch_log]
            ax.plot(epochs, grad_norms, "g-", label="Mean Gradient Norm")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Gradient Norm")
            ax.set_title("Gradient Norm Trajectory")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale("log")
            path = self.output_dir / f"{prefix}_gradient_trajectory.png"
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)
        except Exception as e:
            logger.warning("Failed to generate gradient plot: %s", e)

        # On-target ratio trajectory plot
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            on_target = [e.mean_on_target_ratio for e in self.epoch_log]
            ax.plot(epochs, on_target, "m-", label="On-Target Ratio")
            ax.axhline(
                y=0.8, color="g", linestyle="--", label="Success Threshold (80%)"
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("On-Target Ratio")
            ax.set_title("On-Target Ratio Trajectory")
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)
            path = self.output_dir / f"{prefix}_on_target_trajectory.png"
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)
        except Exception as e:
            logger.warning("Failed to generate on-target plot: %s", e)

        logger.info("Generated %d diagnostic plots in %s", len(paths), self.output_dir)
        return paths

    def reset(self) -> None:
        """Reset all diagnostics state."""
        self.step_log = []
        self.epoch_log = []
        self._current_epoch_steps = []
        self._step_counter = 0
        self._epoch_entry_count = 0
