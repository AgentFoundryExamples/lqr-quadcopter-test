"""
Loss Functions for Quadcopter Tracking Training

This module provides configurable loss functions for training deep learning
controllers. Supports various error formulations and weighting matrices
for comparing different optimization objectives.

Design Philosophy:
- Modular loss components that can be combined
- Configurable weighting matrices for different error types
- Logging hooks for performance comparison
- Numerical stability with NaN detection
"""

import logging
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TrackingLoss(nn.Module):
    """
    Configurable tracking loss for quadcopter control.

    Combines position error, velocity error, and control effort into
    a weighted loss function for training.

    Attributes:
        position_weight: Weight matrix for position error (3x3).
        velocity_weight: Weight matrix for velocity error (3x3).
        control_weight: Weight matrix for control effort (4x4).
        error_type: Type of error norm ('l2', 'l1', 'huber').
    """

    def __init__(
        self,
        position_weight: np.ndarray | torch.Tensor | float = 1.0,
        velocity_weight: np.ndarray | torch.Tensor | float = 0.1,
        control_weight: np.ndarray | torch.Tensor | float = 0.01,
        error_type: Literal["l2", "l1", "huber"] = "l2",
        huber_delta: float = 1.0,
        device: str | torch.device = "cpu",
    ):
        """
        Initialize tracking loss.

        Args:
            position_weight: Weight for position error. Scalar or 3x3 matrix.
            velocity_weight: Weight for velocity error. Scalar or 3x3 matrix.
            control_weight: Weight for control effort. Scalar or 4x4 matrix.
            error_type: Error norm type ('l2', 'l1', 'huber').
            huber_delta: Delta parameter for Huber loss.
            device: Torch device for computation.
        """
        super().__init__()

        self.error_type = error_type
        self.huber_delta = huber_delta
        self.device = torch.device(device)

        # Convert weights to tensors
        self.register_buffer(
            "position_weight",
            self._to_weight_matrix(position_weight, 3),
        )
        self.register_buffer(
            "velocity_weight",
            self._to_weight_matrix(velocity_weight, 3),
        )
        self.register_buffer(
            "control_weight",
            self._to_weight_matrix(control_weight, 4),
        )

    def _to_weight_matrix(
        self, weight: np.ndarray | torch.Tensor | float, dim: int
    ) -> torch.Tensor:
        """Convert weight specification to matrix tensor."""
        if isinstance(weight, (int, float)):
            return torch.eye(dim, dtype=torch.float32, device=self.device) * weight
        elif isinstance(weight, np.ndarray):
            if weight.shape == ():
                return (
                    torch.eye(dim, dtype=torch.float32, device=self.device)
                    * float(weight)
                )
            elif weight.shape == (dim,):
                return torch.diag(
                    torch.tensor(weight, dtype=torch.float32, device=self.device)
                )
            elif weight.shape == (dim, dim):
                return torch.tensor(
                    weight, dtype=torch.float32, device=self.device
                )
            else:
                raise ValueError(
                    f"Weight shape {weight.shape} incompatible with dim {dim}"
                )
        elif isinstance(weight, torch.Tensor):
            if weight.dim() == 0:
                return (
                    torch.eye(dim, dtype=torch.float32, device=self.device)
                    * weight.item()
                )
            elif weight.shape == (dim,):
                return torch.diag(weight.float().to(self.device))
            elif weight.shape == (dim, dim):
                return weight.float().to(self.device)
            else:
                raise ValueError(
                    f"Weight shape {weight.shape} incompatible with dim {dim}"
                )
        else:
            raise TypeError(f"Unsupported weight type: {type(weight)}")

    def forward(
        self,
        position_error: torch.Tensor,
        velocity_error: torch.Tensor,
        action: torch.Tensor,
        action_target: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute tracking loss.

        Args:
            position_error: Position error tensor (batch_size, 3).
            velocity_error: Velocity error tensor (batch_size, 3).
            action: Controller action tensor (batch_size, 4).
            action_target: Optional target action for imitation (batch_size, 4).

        Returns:
            Dictionary with loss components and total loss:
                - total: Combined weighted loss
                - position: Position error component
                - velocity: Velocity error component
                - control: Control effort component
        """
        # Compute weighted errors
        pos_loss = self._compute_weighted_error(position_error, self.position_weight)
        vel_loss = self._compute_weighted_error(velocity_error, self.velocity_weight)

        if action_target is not None:
            # Imitation learning: penalize deviation from target action
            control_error = action - action_target
            ctrl_loss = self._compute_weighted_error(control_error, self.control_weight)
        else:
            # Reinforcement learning: penalize control magnitude
            ctrl_loss = self._compute_weighted_error(action, self.control_weight)

        total = pos_loss + vel_loss + ctrl_loss

        # Check for NaN and replace with a large finite value to signal issues
        # while maintaining gradient flow
        if torch.isnan(total):
            logger.warning("NaN detected in loss computation")
            # Use a large value that still allows gradient computation
            total = (
                torch.nan_to_num(pos_loss, nan=10.0)
                + torch.nan_to_num(vel_loss, nan=10.0)
                + torch.nan_to_num(ctrl_loss, nan=10.0)
            )

        return {
            "total": total,
            "position": pos_loss,
            "velocity": vel_loss,
            "control": ctrl_loss,
        }

    def _compute_weighted_error(
        self, error: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        """Compute weighted error norm."""
        # error: (batch_size, dim)
        # weight: (dim, dim)

        # Compute x^T W x for each sample
        weighted = torch.matmul(error, weight)  # (batch_size, dim)

        if self.error_type == "l2":
            # Quadratic loss: mean of x^T W x
            loss = (weighted * error).sum(dim=-1).mean()
        elif self.error_type == "l1":
            # L1 loss: mean of |Wx|
            loss = torch.abs(weighted).sum(dim=-1).mean()
        elif self.error_type == "huber":
            # Huber loss
            weighted_norm = weighted.norm(dim=-1)
            delta = self.huber_delta
            huber = torch.where(
                weighted_norm <= delta,
                0.5 * weighted_norm**2,
                delta * (weighted_norm - 0.5 * delta),
            )
            loss = huber.mean()
        else:
            raise ValueError(f"Unknown error type: {self.error_type}")

        return loss


class RewardShapingLoss(nn.Module):
    """
    Reward-shaping loss for policy gradient training.

    Transforms negative tracking error into training signal with
    configurable shaping functions.

    Attributes:
        target_radius: On-target radius threshold.
        on_target_bonus: Bonus reward for being on-target.
        distance_penalty: Penalty coefficient for distance.
    """

    def __init__(
        self,
        target_radius: float = 0.5,
        on_target_bonus: float = 1.0,
        distance_penalty: float = 1.0,
        smoothing: Literal["none", "exp", "sigmoid"] = "none",
        device: str | torch.device = "cpu",
    ):
        """
        Initialize reward shaping loss.

        Args:
            target_radius: Radius for on-target determination.
            on_target_bonus: Bonus for being within target radius.
            distance_penalty: Coefficient for distance-based penalty.
            smoothing: Smoothing function for reward ('none', 'exp', 'sigmoid').
            device: Torch device.
        """
        super().__init__()
        self.target_radius = target_radius
        self.on_target_bonus = on_target_bonus
        self.distance_penalty = distance_penalty
        self.smoothing = smoothing
        self.device = torch.device(device)

    def forward(self, tracking_error: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Compute shaped reward from tracking error.

        Args:
            tracking_error: Distance to target (batch_size,) or scalar.

        Returns:
            Dictionary with reward components:
                - reward: Shaped reward signal
                - on_target: Boolean mask for on-target samples
        """
        if tracking_error.dim() == 0:
            tracking_error = tracking_error.unsqueeze(0)

        # On-target bonus
        on_target = tracking_error <= self.target_radius
        bonus = on_target.float() * self.on_target_bonus

        # Distance penalty with optional smoothing
        if self.smoothing == "none":
            penalty = tracking_error * self.distance_penalty
        elif self.smoothing == "exp":
            penalty = (1 - torch.exp(-tracking_error)) * self.distance_penalty
        elif self.smoothing == "sigmoid":
            penalty = (
                torch.sigmoid(tracking_error - self.target_radius)
                * self.distance_penalty
            )
        else:
            raise ValueError(f"Unknown smoothing: {self.smoothing}")

        reward = bonus - penalty

        return {
            "reward": reward.mean(),
            "on_target": on_target,
            "bonus": bonus.mean(),
            "penalty": penalty.mean(),
        }


class CombinedLoss(nn.Module):
    """
    Combined loss aggregating multiple loss components.

    Allows mixing tracking loss with reward shaping and optional
    auxiliary losses.
    """

    def __init__(
        self,
        tracking_loss: TrackingLoss | None = None,
        reward_loss: RewardShapingLoss | None = None,
        tracking_weight: float = 1.0,
        reward_weight: float = 0.0,
        device: str | torch.device = "cpu",
    ):
        """
        Initialize combined loss.

        Args:
            tracking_loss: Tracking loss component.
            reward_loss: Reward shaping loss component.
            tracking_weight: Weight for tracking loss.
            reward_weight: Weight for reward loss (negative reward = loss).
            device: Torch device.
        """
        super().__init__()
        self.device = torch.device(device)

        self.tracking_loss = tracking_loss or TrackingLoss(device=device)
        self.reward_loss = reward_loss or RewardShapingLoss(device=device)

        self.tracking_weight = tracking_weight
        self.reward_weight = reward_weight

    def forward(
        self,
        position_error: torch.Tensor,
        velocity_error: torch.Tensor,
        action: torch.Tensor,
        tracking_error: torch.Tensor,
        action_target: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            position_error: Position error (batch_size, 3).
            velocity_error: Velocity error (batch_size, 3).
            action: Controller action (batch_size, 4).
            tracking_error: Scalar distance (batch_size,).
            action_target: Optional imitation target.

        Returns:
            Dictionary with all loss components.
        """
        tracking_out = self.tracking_loss(
            position_error, velocity_error, action, action_target
        )
        reward_out = self.reward_loss(tracking_error)

        # Combine: tracking loss + (- reward * weight) since we minimize
        total = (
            self.tracking_weight * tracking_out["total"]
            - self.reward_weight * reward_out["reward"]
        )

        return {
            "total": total,
            "tracking_total": tracking_out["total"],
            "tracking_position": tracking_out["position"],
            "tracking_velocity": tracking_out["velocity"],
            "tracking_control": tracking_out["control"],
            "reward": reward_out["reward"],
            "on_target_ratio": reward_out["on_target"].float().mean(),
        }


def create_loss_from_config(config: dict) -> CombinedLoss:
    """
    Create loss function from configuration dictionary.

    Args:
        config: Configuration with keys:
            - position_weight: Weight for position error
            - velocity_weight: Weight for velocity error
            - control_weight: Weight for control effort
            - error_type: 'l2', 'l1', or 'huber'
            - target_radius: On-target threshold
            - on_target_bonus: Bonus for being on-target
            - tracking_weight: Weight for tracking loss component
            - reward_weight: Weight for reward shaping component
            - device: Torch device

    Returns:
        Configured CombinedLoss instance.
    """
    device = config.get("device", "cpu")

    tracking_loss = TrackingLoss(
        position_weight=config.get("position_weight", 1.0),
        velocity_weight=config.get("velocity_weight", 0.1),
        control_weight=config.get("control_weight", 0.01),
        error_type=config.get("error_type", "l2"),
        huber_delta=config.get("huber_delta", 1.0),
        device=device,
    )

    reward_loss = RewardShapingLoss(
        target_radius=config.get("target_radius", 0.5),
        on_target_bonus=config.get("on_target_bonus", 1.0),
        distance_penalty=config.get("distance_penalty", 1.0),
        smoothing=config.get("smoothing", "none"),
        device=device,
    )

    return CombinedLoss(
        tracking_loss=tracking_loss,
        reward_loss=reward_loss,
        tracking_weight=config.get("tracking_weight", 1.0),
        reward_weight=config.get("reward_weight", 0.0),
        device=device,
    )


class LossLogger:
    """
    Logger for tracking loss components over training.

    Accumulates loss values and provides statistics for monitoring
    and comparison.
    """

    def __init__(self):
        """Initialize loss logger."""
        self.history: list[dict[str, float]] = []
        self.current_epoch: list[dict[str, float]] = []

    def log(self, losses: dict[str, torch.Tensor | float]) -> None:
        """
        Log loss values from a training step.

        Args:
            losses: Dictionary of loss component values.
        """
        entry = {}
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                entry[key] = value.detach().cpu().item()
            else:
                entry[key] = float(value)
        self.current_epoch.append(entry)

    def end_epoch(self) -> dict[str, float]:
        """
        End current epoch and compute statistics.

        Returns:
            Dictionary of mean loss values for the epoch.
        """
        if not self.current_epoch:
            return {}

        # Compute means
        keys = self.current_epoch[0].keys()
        means = {}
        for key in keys:
            values = [entry[key] for entry in self.current_epoch]
            means[key] = sum(values) / len(values)

        self.history.append(means)
        self.current_epoch = []

        return means

    def get_history(self) -> list[dict[str, float]]:
        """Get complete loss history."""
        return self.history.copy()

    def get_best_epoch(self, metric: str = "total") -> tuple[int, float]:
        """
        Get epoch with best (lowest) metric value.

        Args:
            metric: Metric name to minimize.

        Returns:
            Tuple of (epoch_index, metric_value).
        """
        if not self.history:
            return -1, float("inf")

        values = [epoch.get(metric, float("inf")) for epoch in self.history]
        best_idx = int(np.argmin(values))
        return best_idx, values[best_idx]

    def reset(self) -> None:
        """Reset logger state."""
        self.history = []
        self.current_epoch = []
