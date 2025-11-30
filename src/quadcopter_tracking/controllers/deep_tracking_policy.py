"""
Deep Learning Controller for Quadcopter Target Tracking

This module provides neural network-based policy controllers for quadcopter tracking.
Controllers learn to map state+target information to thrust/torque commands via
gradient-descent training.

Design Philosophy:
- Configurable network architectures (hidden sizes, activations)
- Bounded outputs compatible with quadcopter dynamics
- Compatible with existing BaseController interface
- Support for both inference and training modes
"""

import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from .base import BaseController

logger = logging.getLogger(__name__)


class PolicyNetwork(nn.Module):
    """
    Neural network policy for quadcopter control.

    Maps observation features to bounded control actions using a configurable
    feedforward architecture.

    Attributes:
        hidden_sizes: List of hidden layer dimensions.
        activation: Activation function name.
        output_bounds: Dictionary mapping action names to (min, max) tuples.
    """

    def __init__(
        self,
        input_dim: int = 18,
        hidden_sizes: list[int] | None = None,
        activation: Literal["relu", "tanh", "elu", "leaky_relu"] = "relu",
        output_bounds: dict[str, tuple[float, float]] | None = None,
    ):
        """
        Initialize the policy network.

        Args:
            input_dim: Dimension of input features.
            hidden_sizes: List of hidden layer sizes. Default [64, 64].
            activation: Activation function ('relu', 'tanh', 'elu', 'leaky_relu').
            output_bounds: Dict mapping action names to (min, max) bounds.
        """
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [64, 64]

        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.activation_name = activation
        self.output_dim = 4  # thrust, roll_rate, pitch_rate, yaw_rate

        # Default action bounds (from QuadcopterParams defaults)
        if output_bounds is None:
            output_bounds = {
                "thrust": (0.0, 20.0),
                "roll_rate": (-3.0, 3.0),
                "pitch_rate": (-3.0, 3.0),
                "yaw_rate": (-3.0, 3.0),
            }
        self.output_bounds = output_bounds

        # Build network layers
        activation_fn = self._get_activation(activation)

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_fn)
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, self.output_dim)

        # Initialize weights
        self._init_weights()

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(0.1),
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}. Valid: {list(activations)}")
        return activations[name]

    def _init_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Action tensor of shape (batch_size, 4) with bounded outputs.
        """
        features = self.feature_extractor(x)
        raw_output = self.output_layer(features)

        # Apply sigmoid and scale to bounds
        bounded_output = self._apply_bounds(raw_output)
        return bounded_output

    def _apply_bounds(self, raw: torch.Tensor) -> torch.Tensor:
        """Apply output bounds using sigmoid scaling."""
        # Use sigmoid to get values in [0, 1], then scale to bounds
        sigmoid_out = torch.sigmoid(raw)

        bounds_list = [
            self.output_bounds["thrust"],
            self.output_bounds["roll_rate"],
            self.output_bounds["pitch_rate"],
            self.output_bounds["yaw_rate"],
        ]

        result = torch.zeros_like(raw)
        for i, (low, high) in enumerate(bounds_list):
            result[:, i] = sigmoid_out[:, i] * (high - low) + low

        return result


class DeepTrackingPolicy(BaseController):
    """
    Deep learning-based tracking controller.

    Wraps PolicyNetwork to provide BaseController interface for quadcopter
    tracking. Extracts features from observation dictionaries and produces
    bounded control actions.

    Attributes:
        network: The underlying PolicyNetwork.
        device: Torch device for computation.
    """

    def __init__(
        self,
        config: dict | None = None,
        device: str | None = None,
    ):
        """
        Initialize deep tracking policy.

        Args:
            config: Controller configuration with keys:
                - hidden_sizes: List of hidden layer dimensions
                - activation: Activation function name
                - output_bounds: Dict of action bounds
                - checkpoint_path: Optional path to load weights
            device: Torch device ('cpu', 'cuda', or None for auto-detect).
        """
        super().__init__(name="deep_tracking", config=config)

        config = config or {}

        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Extract network configuration
        hidden_sizes = config.get("hidden_sizes", [64, 64])
        activation = config.get("activation", "relu")
        output_bounds = config.get("output_bounds", None)

        # Create network
        self.network = PolicyNetwork(
            input_dim=18,  # 12 quad state + 6 target state
            hidden_sizes=hidden_sizes,
            activation=activation,
            output_bounds=output_bounds,
        ).to(self.device)

        # Load checkpoint if provided
        checkpoint_path = config.get("checkpoint_path")
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        # Set to eval mode by default
        self.network.eval()

    def compute_action(self, observation: dict) -> dict:
        """
        Compute control action from observation.

        Args:
            observation: Environment observation with 'quadcopter' and 'target' keys.

        Returns:
            Action dictionary with thrust, roll_rate, pitch_rate, yaw_rate.
        """
        # Extract features
        features = self._extract_features(observation)

        # Convert to tensor
        features_tensor = torch.tensor(
            features, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        # Forward pass
        with torch.no_grad():
            action_tensor = self.network(features_tensor)

        # Convert to numpy
        action_array = action_tensor.cpu().numpy().squeeze()

        return {
            "thrust": float(action_array[0]),
            "roll_rate": float(action_array[1]),
            "pitch_rate": float(action_array[2]),
            "yaw_rate": float(action_array[3]),
        }

    def _extract_features(self, observation: dict) -> np.ndarray:
        """
        Extract feature vector from observation dictionary.

        Feature layout (18 dimensions):
            [0:3]   - Position error (target - quad)
            [3:6]   - Velocity error (target - quad)
            [6:9]   - Quadcopter attitude (roll, pitch, yaw)
            [9:12]  - Quadcopter angular velocity
            [12:15] - Target position
            [15:18] - Target velocity

        Args:
            observation: Environment observation dictionary.

        Returns:
            Feature vector as numpy array of shape (18,).
        """
        quad = observation["quadcopter"]
        target = observation["target"]

        # Position and velocity errors
        pos_error = np.array(target["position"]) - np.array(quad["position"])
        vel_error = np.array(target["velocity"]) - np.array(quad["velocity"])

        # Quadcopter state
        attitude = np.array(quad["attitude"])
        angular_vel = np.array(quad["angular_velocity"])

        # Target state
        target_pos = np.array(target["position"])
        target_vel = np.array(target["velocity"])

        # Concatenate features
        features = np.concatenate(
            [
                pos_error,
                vel_error,
                attitude,
                angular_vel,
                target_pos,
                target_vel,
            ]
        )

        return features.astype(np.float32)

    def reset(self) -> None:
        """Reset controller state (no-op for feedforward network)."""
        pass

    def train_mode(self) -> None:
        """Set network to training mode."""
        self.network.train()

    def eval_mode(self) -> None:
        """Set network to evaluation mode."""
        self.network.eval()

    def get_parameters(self) -> list[torch.nn.Parameter]:
        """Get network parameters for optimization."""
        return list(self.network.parameters())

    def save_checkpoint(
        self,
        path: str | Path,
        metadata: dict | None = None,
    ) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint file.
            metadata: Optional metadata to include (e.g., training info).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.network.state_dict(),
            "config": self.config,
            "input_dim": self.network.input_dim,
            "hidden_sizes": self.network.hidden_sizes,
            "activation": self.network.activation_name,
            "output_bounds": self.network.output_bounds,
        }

        if metadata:
            checkpoint["metadata"] = metadata

        torch.save(checkpoint, path)
        logger.info("Saved checkpoint to %s", path)

    def load_checkpoint(self, path: str | Path) -> dict:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint file.

        Returns:
            Metadata from checkpoint if present.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Validate architecture compatibility
        if checkpoint.get("input_dim") != self.network.input_dim:
            raise ValueError(
                f"Input dim mismatch: checkpoint has {checkpoint.get('input_dim')}, "
                f"network has {self.network.input_dim}"
            )

        if checkpoint.get("hidden_sizes") != self.network.hidden_sizes:
            logger.warning(
                "Hidden sizes mismatch: checkpoint %s, network %s",
                checkpoint.get("hidden_sizes"),
                self.network.hidden_sizes,
            )

        self.network.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Loaded checkpoint from %s", path)

        return checkpoint.get("metadata", {})

    def to_onnx(self, path: str | Path) -> None:
        """
        Export model to ONNX format.

        Args:
            path: Path for ONNX file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        dummy_input = torch.zeros(1, self.network.input_dim, device=self.device)

        torch.onnx.export(
            self.network,
            dummy_input,
            path,
            input_names=["observation"],
            output_names=["action"],
            dynamic_axes={
                "observation": {0: "batch_size"},
                "action": {0: "batch_size"},
            },
        )
        logger.info("Exported ONNX model to %s", path)


def create_controller_from_config(config_path: str | Path) -> DeepTrackingPolicy:
    """
    Create DeepTrackingPolicy from JSON configuration file.

    Args:
        config_path: Path to JSON configuration file.

    Returns:
        Configured DeepTrackingPolicy instance.
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        config = json.load(f)

    return DeepTrackingPolicy(config=config)
