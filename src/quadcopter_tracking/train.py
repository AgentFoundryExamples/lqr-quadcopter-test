#!/usr/bin/env python3
"""
Training Script for Quadcopter Controllers

This script provides a complete training pipeline for controllers:
- Supports deep learning, PID, and LQR controllers via --controller flag
- Configurable via YAML/JSON files or CLI arguments
- Episode-based gradient descent for deep controllers
- Evaluation-only mode for classical controllers (PID, LQR)
- Experiment tracking with CSV/JSON logs and checkpointing
- NaN detection and optimizer recovery
- Support for curriculum learning and adaptive weighting
- Opt-in diagnostics for training analysis

Usage:
    python -m quadcopter_tracking.train --config configs/training.yaml
    python -m quadcopter_tracking.train --epochs 100 --lr 0.001 --seed 42
    python -m quadcopter_tracking.train --controller pid --episodes-per-epoch 5
    python -m quadcopter_tracking.train --controller lqr --motion-type stationary
    python -m quadcopter_tracking.train --epochs 50 --diagnostics  # Enable diagnostics
"""

import argparse
import csv
import datetime
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml

from quadcopter_tracking.controllers import (
    BaseController,
    DeepTrackingPolicy,
    LQRController,
    PIDController,
)
from quadcopter_tracking.env import EnvConfig, QuadcopterEnv
from quadcopter_tracking.utils.diagnostics import Diagnostics, DiagnosticsConfig
from quadcopter_tracking.utils.losses import (
    LossLogger,
    create_loss_from_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class TrainingConfig:
    """Configuration class for training parameters."""

    # Valid training modes
    VALID_TRAINING_MODES = ("tracking", "imitation", "reward_weighted")

    def __init__(
        self,
        # Controller type
        controller: str = "deep",
        # Training parameters
        epochs: int = 100,
        episodes_per_epoch: int = 10,
        max_steps_per_episode: int = 3000,
        batch_size: int = 32,
        # Optimizer parameters
        learning_rate: float = 0.001,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        grad_clip: float = 1.0,
        # Network parameters
        hidden_sizes: list[int] | None = None,
        activation: str = "relu",
        # Loss parameters
        position_weight: float = 1.0,
        velocity_weight: float = 0.1,
        control_weight: float = 0.01,
        error_type: str = "l2",
        tracking_weight: float = 1.0,
        reward_weight: float = 0.0,
        # Training mode parameters
        training_mode: str = "tracking",
        supervisor_controller: str = "pid",
        imitation_weight: float = 1.0,
        supervisor_blend_ratio: float = 0.0,
        # Environment parameters
        env_seed: int = 42,
        target_motion_type: str = "circular",
        episode_length: float = 30.0,
        target_radius: float = 0.5,
        # Checkpointing
        checkpoint_dir: str = "checkpoints",
        checkpoint_interval: int = 10,
        save_best: bool = True,
        # Logging
        log_dir: str = "experiments/logs",
        log_interval: int = 1,
        # Curriculum learning
        use_curriculum: bool = False,
        curriculum_start_difficulty: float = 0.5,
        curriculum_end_difficulty: float = 1.0,
        # Recovery
        nan_recovery_attempts: int = 3,
        lr_reduction_factor: float = 0.5,
        # Device
        device: str | None = None,
        # Diagnostics (opt-in)
        diagnostics_enabled: bool = False,
        diagnostics_log_observations: bool = True,
        diagnostics_log_actions: bool = True,
        diagnostics_log_gradients: bool = True,
        diagnostics_log_interval: int = 10,
        diagnostics_output_dir: str | None = None,
        diagnostics_generate_plots: bool = True,
    ):
        # Validate and set controller type
        valid_controllers = ("deep", "lqr", "pid")
        if controller not in valid_controllers:
            raise ValueError(
                f"Invalid controller type: '{controller}'. "
                f"Valid choices are: {', '.join(valid_controllers)}"
            )
        self.controller = controller

        self.epochs = epochs
        self.episodes_per_epoch = episodes_per_epoch
        self.max_steps_per_episode = max_steps_per_episode
        self.batch_size = batch_size

        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

        self.hidden_sizes = hidden_sizes or [64, 64]
        self.activation = activation

        self.position_weight = position_weight
        self.velocity_weight = velocity_weight
        self.control_weight = control_weight
        self.error_type = error_type
        self.tracking_weight = tracking_weight
        self.reward_weight = reward_weight

        # Validate and set training mode
        if training_mode not in self.VALID_TRAINING_MODES:
            raise ValueError(
                f"Invalid training_mode: '{training_mode}'. "
                f"Valid choices are: {', '.join(self.VALID_TRAINING_MODES)}"
            )
        self.training_mode = training_mode

        # Validate supervisor controller (used in imitation mode)
        valid_supervisors = ("pid", "lqr")
        if supervisor_controller not in valid_supervisors:
            raise ValueError(
                f"Invalid supervisor_controller: '{supervisor_controller}'. "
                f"Valid choices are: {', '.join(valid_supervisors)}"
            )
        self.supervisor_controller = supervisor_controller
        self.imitation_weight = imitation_weight
        self.supervisor_blend_ratio = supervisor_blend_ratio

        self.env_seed = env_seed
        self.target_motion_type = target_motion_type
        self.episode_length = episode_length
        self.target_radius = target_radius

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.save_best = save_best

        self.log_dir = log_dir
        self.log_interval = log_interval

        self.use_curriculum = use_curriculum
        self.curriculum_start_difficulty = curriculum_start_difficulty
        self.curriculum_end_difficulty = curriculum_end_difficulty

        self.nan_recovery_attempts = nan_recovery_attempts
        self.lr_reduction_factor = lr_reduction_factor

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Diagnostics configuration
        self.diagnostics_enabled = diagnostics_enabled
        self.diagnostics_log_observations = diagnostics_log_observations
        self.diagnostics_log_actions = diagnostics_log_actions
        self.diagnostics_log_gradients = diagnostics_log_gradients
        self.diagnostics_log_interval = diagnostics_log_interval
        self.diagnostics_output_dir = diagnostics_output_dir
        self.diagnostics_generate_plots = diagnostics_generate_plots

    @classmethod
    def from_dict(cls, config_dict: dict) -> "TrainingConfig":
        """Create config from dictionary."""
        valid_keys = cls.__init__.__code__.co_varnames
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        instance = cls(**filtered)
        instance.full_config = config_dict  # Store the original config
        return instance

    @classmethod
    def from_file(cls, path: str | Path) -> "TrainingConfig":
        """Load config from YAML or JSON file."""
        path = Path(path)
        with open(path) as f:
            if path.suffix in (".yaml", ".yml"):
                config_dict = yaml.safe_load(f)
            elif path.suffix == ".json":
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")
        return cls.from_dict(config_dict or {})

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "controller": self.controller,
            "epochs": self.epochs,
            "episodes_per_epoch": self.episodes_per_epoch,
            "max_steps_per_episode": self.max_steps_per_episode,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
            "hidden_sizes": self.hidden_sizes,
            "activation": self.activation,
            "position_weight": self.position_weight,
            "velocity_weight": self.velocity_weight,
            "control_weight": self.control_weight,
            "error_type": self.error_type,
            "tracking_weight": self.tracking_weight,
            "reward_weight": self.reward_weight,
            "training_mode": self.training_mode,
            "supervisor_controller": self.supervisor_controller,
            "imitation_weight": self.imitation_weight,
            "supervisor_blend_ratio": self.supervisor_blend_ratio,
            "env_seed": self.env_seed,
            "target_motion_type": self.target_motion_type,
            "episode_length": self.episode_length,
            "target_radius": self.target_radius,
            "checkpoint_dir": self.checkpoint_dir,
            "checkpoint_interval": self.checkpoint_interval,
            "save_best": self.save_best,
            "log_dir": self.log_dir,
            "log_interval": self.log_interval,
            "use_curriculum": self.use_curriculum,
            "curriculum_start_difficulty": self.curriculum_start_difficulty,
            "curriculum_end_difficulty": self.curriculum_end_difficulty,
            "nan_recovery_attempts": self.nan_recovery_attempts,
            "lr_reduction_factor": self.lr_reduction_factor,
            "device": self.device,
            "diagnostics_enabled": self.diagnostics_enabled,
            "diagnostics_log_observations": self.diagnostics_log_observations,
            "diagnostics_log_actions": self.diagnostics_log_actions,
            "diagnostics_log_gradients": self.diagnostics_log_gradients,
            "diagnostics_log_interval": self.diagnostics_log_interval,
            "diagnostics_output_dir": self.diagnostics_output_dir,
            "diagnostics_generate_plots": self.diagnostics_generate_plots,
        }


class Trainer:
    """
    Training loop for quadcopter controllers.

    For deep learning controllers: handles episode collection, gradient computation,
    optimization, checkpointing, logging, and optional diagnostics.

    For classical controllers (PID, LQR): runs evaluation episodes to generate
    metrics and logs, but skips training-specific steps like gradient updates
    and checkpointing.
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer.

        Args:
            config: Training configuration.
        """
        self.config = config
        self.full_config = getattr(config, "full_config", {})
        self.device = torch.device(config.device)
        self.is_deep_controller = config.controller == "deep"

        # Create controller based on type
        self.controller = self._create_controller()

        # Warn about incompatible config options for classical controllers
        if not self.is_deep_controller:
            self._warn_incompatible_options()

        # Create environment
        env_config = EnvConfig()
        env_config.seed = config.env_seed
        env_config.simulation.max_episode_time = config.episode_length
        env_config.target.motion_type = config.target_motion_type
        env_config.success_criteria.target_radius = config.target_radius
        self.env = QuadcopterEnv(config=env_config)

        # Create loss function (used for deep training and metrics logging)
        loss_config = {
            "position_weight": config.position_weight,
            "velocity_weight": config.velocity_weight,
            "control_weight": config.control_weight,
            "error_type": config.error_type,
            "target_radius": config.target_radius,
            "tracking_weight": config.tracking_weight,
            "reward_weight": config.reward_weight,
            "device": config.device,
        }
        self.loss_fn = create_loss_from_config(loss_config)

        # Create optimizer (only for deep controllers)
        if self.is_deep_controller:
            self.optimizer = self._create_optimizer()
        else:
            self.optimizer = None

        # Setup logging
        self.loss_logger = LossLogger()
        self.experiment_log: list[dict] = []

        # Setup directories
        self.checkpoint_dir = Path(config.checkpoint_dir)
        if self.is_deep_controller:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.best_loss = float("inf")
        self.nan_recovery_count = 0

        # Generate experiment ID with microseconds for uniqueness
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%d_%H%M%S_%f"
        )
        self.experiment_id = f"train_{config.controller}_{timestamp}_{config.env_seed}"

        # Step counter for diagnostics
        self._step_counter = 0

        # Setup diagnostics (opt-in)
        diag_output_dir = config.diagnostics_output_dir or str(
            self.log_dir / "diagnostics"
        )
        # Only log gradients for deep controllers (classical controllers don't train)
        log_gradients = config.diagnostics_log_gradients and self.is_deep_controller
        self.diagnostics = Diagnostics(
            config=DiagnosticsConfig(
                enabled=config.diagnostics_enabled,
                log_observations=config.diagnostics_log_observations,
                log_actions=config.diagnostics_log_actions,
                log_gradients=log_gradients,
                log_interval=config.diagnostics_log_interval,
                output_dir=diag_output_dir,
                generate_plots=config.diagnostics_generate_plots,
                headless=True,
            )
        )
        if config.diagnostics_enabled:
            logger.info("Diagnostics enabled, output: %s", diag_output_dir)

        # Create supervisor controller for imitation mode
        self.supervisor = None
        if self.is_deep_controller and self._uses_supervisor_mode:
            self.supervisor = self._create_supervisor()
            logger.info(
                "Supervisor controller (%s) created for %s mode",
                config.supervisor_controller,
                config.training_mode,
            )

    @property
    def _uses_supervisor_mode(self) -> bool:
        """Check if training mode requires a supervisor controller."""
        return self.config.training_mode in ("imitation", "reward_weighted")

    def _create_controller(self) -> BaseController:
        """Create controller based on config type."""
        config = self.config

        if config.controller == "deep":
            controller_config = {
                "hidden_sizes": config.hidden_sizes,
                "activation": config.activation,
                "output_bounds": {
                    "thrust": (0.0, 20.0),
                    "roll_rate": (-3.0, 3.0),
                    "pitch_rate": (-3.0, 3.0),
                    "yaw_rate": (-3.0, 3.0),
                },
            }
            return DeepTrackingPolicy(config=controller_config, device=config.device)
        elif config.controller == "pid":
            return PIDController(config={})
        elif config.controller == "lqr":
            return LQRController(config={})
        else:
            raise ValueError(f"Unknown controller type: {config.controller}")

    def _create_supervisor(self) -> BaseController:
        """Create supervisor controller for imitation/reward-weighted modes."""
        config = self.config
        supervisor_type = config.supervisor_controller

        # Get the supervisor-specific config from the full config dictionary
        supervisor_config = self.full_config.get(supervisor_type, {})

        if supervisor_type == "pid":
            return PIDController(config=supervisor_config)
        elif supervisor_type == "lqr":
            return LQRController(config=supervisor_config)
        else:
            raise ValueError(
                f"Unknown supervisor controller type: {supervisor_type}"
            )

    def _warn_incompatible_options(self) -> None:
        """Warn about options that don't apply to classical controllers."""
        config = self.config
        warnings = []

        # Deep-specific options that are ignored for classical controllers
        if config.learning_rate != 0.001:
            warnings.append("learning_rate (classical controllers don't train)")
        if config.optimizer != "adam":
            warnings.append("optimizer (classical controllers don't train)")
        if config.hidden_sizes != [64, 64]:
            warnings.append("hidden_sizes (classical controllers have no network)")
        if config.grad_clip != 1.0:
            warnings.append("grad_clip (classical controllers don't use gradients)")
        if config.use_curriculum:
            warnings.append("use_curriculum (classical controllers don't train)")
        if config.checkpoint_interval != 10:
            warnings.append(
                "checkpoint_interval (classical controllers don't save checkpoints)"
            )

        if warnings:
            logger.warning(
                "The following config options are ignored for %s controller: %s",
                config.controller,
                ", ".join(warnings),
            )

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        params = self.controller.get_parameters()

        if self.config.optimizer.lower() == "adam":
            return optim.Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer.lower() == "sgd":
            return optim.SGD(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9,
            )
        elif self.config.optimizer.lower() == "adamw":
            return optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def train(self) -> dict:
        """
        Run the training loop.

        For deep controllers: runs full gradient-based training.
        For classical controllers: runs evaluation episodes without training.

        Returns:
            Dictionary with training results and final metrics.
        """
        logger.info(
            "Starting %s with config: %s",
            "training" if self.is_deep_controller else "evaluation",
            self.experiment_id,
        )
        logger.info("Controller: %s", self.config.controller)
        logger.info("Device: %s", self.device)
        logger.info("Epochs: %d", self.config.epochs)

        # Save initial config
        self._save_config()

        # Set controller mode (only deep controllers have train_mode)
        if self.is_deep_controller:
            self.controller.train_mode()

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch

            # Get curriculum difficulty if enabled (deep only)
            difficulty = self._get_curriculum_difficulty(epoch)

            # Run training/evaluation epoch
            if self.is_deep_controller:
                epoch_metrics = self._train_epoch(difficulty)
            else:
                epoch_metrics = self._evaluate_epoch()

            # End epoch logging
            epoch_losses = self.loss_logger.end_epoch()
            epoch_metrics.update(epoch_losses)

            # Log diagnostics for the epoch
            self.diagnostics.log_epoch(epoch, epoch_metrics)

            # Log progress
            if epoch % self.config.log_interval == 0:
                self._log_epoch(epoch, epoch_metrics)

            # Save checkpoint (deep controllers only)
            if self.is_deep_controller:
                if (epoch + 1) % self.config.checkpoint_interval == 0:
                    self._save_checkpoint(epoch, epoch_metrics)

                # Save best model
                current_loss = epoch_metrics.get("total", float("inf"))
                if self.config.save_best and current_loss < self.best_loss:
                    self.best_loss = epoch_metrics["total"]
                    self._save_checkpoint(epoch, epoch_metrics, is_best=True)

                # Check for training failure
                if self._check_training_failure(epoch_metrics):
                    if not self._recover_from_nan():
                        logger.error("Training failed, could not recover")
                        break

        # Save final model and logs (deep controllers only save checkpoints)
        if self.is_deep_controller:
            self._save_checkpoint(self.current_epoch, {}, is_final=True)
        self._save_experiment_log()

        # Save diagnostics if enabled
        self._save_diagnostics()

        return self._get_training_summary()

    def _evaluate_epoch(self) -> dict:
        """
        Run evaluation-only epoch for classical controllers.

        Collects metrics without performing any training updates.
        """
        epoch_rewards = []
        epoch_on_target_ratios = []
        epoch_tracking_errors = []

        for episode in range(self.config.episodes_per_epoch):
            # Reset environment with episode-specific seed
            episode_seed = self.config.env_seed + self.current_epoch * 1000 + episode
            obs = self.env.reset(seed=episode_seed)

            # Reset controller state (important for PID integral error)
            self.controller.reset()

            episode_data = []
            done = False
            step = 0

            while not done and step < self.config.max_steps_per_episode:
                # Compute action using classical controller
                action = self.controller.compute_action(obs)

                # Step environment
                next_obs, reward, done, info = self.env.step(action)

                # Store step data for metrics computation
                episode_data.append(
                    {
                        "obs": obs,
                        "next_obs": next_obs,
                        "action": action,
                        "reward": reward,
                        "info": info,
                    }
                )

                obs = next_obs
                step += 1

            # Compute episode metrics
            epoch_rewards.append(sum(d["reward"] for d in episode_data))
            if info:
                epoch_on_target_ratios.append(info.get("on_target_ratio", 0.0))
                epoch_tracking_errors.append(info.get("tracking_error", 0.0))

            # Log placeholder loss values for consistency
            mean_error_for_log = (
                np.mean(epoch_tracking_errors) if epoch_tracking_errors else 0.0
            )
            self.loss_logger.log(
                {
                    "total": mean_error_for_log,
                    "position": mean_error_for_log,
                    "velocity": 0.0,
                    "control": 0.0,
                }
            )

        # Compute metrics
        mean_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
        mean_on_target = (
            np.mean(epoch_on_target_ratios) if epoch_on_target_ratios else 0.0
        )
        mean_error = np.mean(epoch_tracking_errors) if epoch_tracking_errors else 0.0

        return {
            "mean_reward": mean_reward,
            "mean_on_target_ratio": mean_on_target,
            "mean_tracking_error": mean_error,
            "difficulty": 1.0,
        }

    def _train_epoch(self, difficulty: float = 1.0) -> dict:
        """Run a single training epoch."""
        epoch_rewards = []
        epoch_on_target_ratios = []
        epoch_tracking_errors = []

        for episode in range(self.config.episodes_per_epoch):
            # Reset environment with episode-specific seed
            episode_seed = self.config.env_seed + self.current_epoch * 1000 + episode
            obs = self.env.reset(seed=episode_seed)

            # Reset supervisor state for each episode (important for PID integral)
            if self.supervisor is not None:
                self.supervisor.reset()

            episode_data = []
            done = False
            step = 0

            while not done and step < self.config.max_steps_per_episode:
                # Compute action from deep policy
                features = self.controller._extract_features(obs)
                features_tensor = torch.tensor(
                    features, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                features_tensor.requires_grad_(True)

                action_tensor = self.controller.network(features_tensor)

                # Get supervisor action for imitation learning
                supervisor_action = None
                if self.supervisor is not None:
                    supervisor_action_dict = self.supervisor.compute_action(obs)
                    supervisor_action = torch.tensor(
                        [
                            supervisor_action_dict["thrust"],
                            supervisor_action_dict["roll_rate"],
                            supervisor_action_dict["pitch_rate"],
                            supervisor_action_dict["yaw_rate"],
                        ],
                        dtype=torch.float32,
                        device=self.device,
                    ).unsqueeze(0)

                # Convert to action dict
                action_array = action_tensor.detach().cpu().numpy().squeeze()
                action = {
                    "thrust": float(action_array[0]),
                    "roll_rate": float(action_array[1]),
                    "pitch_rate": float(action_array[2]),
                    "yaw_rate": float(action_array[3]),
                }

                # Step environment
                next_obs, reward, done, info = self.env.step(action)

                # Store step data for batch training
                episode_data.append(
                    {
                        "features": features_tensor,
                        "action": action_tensor,
                        "supervisor_action": supervisor_action,
                        "obs": obs,
                        "next_obs": next_obs,
                        "reward": reward,
                        "info": info,
                    }
                )

                obs = next_obs
                step += 1

            # Compute episode metrics
            if info:
                epoch_rewards.append(sum(d["reward"] for d in episode_data))
                epoch_on_target_ratios.append(info.get("on_target_ratio", 0.0))
                epoch_tracking_errors.append(info.get("tracking_error", 0.0))

            # Update network with episode data
            if episode_data:
                self._step_counter += 1
                self._update_from_episode(episode_data)

        # Compute metrics
        mean_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
        mean_on_target = (
            np.mean(epoch_on_target_ratios) if epoch_on_target_ratios else 0.0
        )
        mean_error = np.mean(epoch_tracking_errors) if epoch_tracking_errors else 0.0

        return {
            "mean_reward": mean_reward,
            "mean_on_target_ratio": mean_on_target,
            "mean_tracking_error": mean_error,
            "difficulty": difficulty,
        }

    def _update_from_episode(self, episode_data: list[dict]) -> None:
        """Update network parameters from episode data using vectorized batching."""
        if not episode_data:
            return

        batch_size = min(self.config.batch_size, len(episode_data))
        indices = np.random.choice(len(episode_data), batch_size, replace=False)
        batch_data = [episode_data[i] for i in indices]

        # Collate batch data into tensors for vectorized processing
        features_batch = torch.cat([d["features"] for d in batch_data], dim=0)

        obs_batch = [d["obs"] for d in batch_data]
        quad_pos = torch.from_numpy(
            np.array([o["quadcopter"]["position"] for o in obs_batch], dtype=np.float32)
        ).to(self.device)
        quad_vel = torch.from_numpy(
            np.array([o["quadcopter"]["velocity"] for o in obs_batch], dtype=np.float32)
        ).to(self.device)
        target_pos = torch.from_numpy(
            np.array([o["target"]["position"] for o in obs_batch], dtype=np.float32)
        ).to(self.device)
        target_vel = torch.from_numpy(
            np.array([o["target"]["velocity"] for o in obs_batch], dtype=np.float32)
        ).to(self.device)

        # Single forward pass for the entire batch
        actions_batch = self.controller.network(features_batch)

        # Compute errors for the batch
        pos_error = target_pos - quad_pos
        vel_error = target_vel - quad_vel
        tracking_error = torch.norm(target_pos - quad_pos, dim=1)

        # Get supervisor actions for imitation/reward-weighted modes
        action_target = None
        if self.config.training_mode in ("imitation", "reward_weighted"):
            supervisor_actions = [d.get("supervisor_action") for d in batch_data]
            if supervisor_actions[0] is not None:
                action_target = torch.cat(supervisor_actions, dim=0)

        # Compute loss based on training mode
        if self.config.training_mode == "imitation" and action_target is not None:
            # Pure imitation loss: match supervisor actions
            imitation_loss = F.mse_loss(actions_batch, action_target)
            # Compute standard loss for logging
            tracking_losses = self.loss_fn(
                pos_error, vel_error, actions_batch, tracking_error
            )
            # Combine: imitation_weight * imitation + tracking_weight * tracking
            total_loss = (
                self.config.imitation_weight * imitation_loss
                + self.config.tracking_weight * tracking_losses["total"]
            )
            # Override total in losses dict for logging
            losses = {
                **tracking_losses,
                "imitation": imitation_loss,
                "total": total_loss,
            }
        elif self.config.training_mode == "reward_weighted":
            # Reward-weighted mode: standard loss with supervisor hint
            # Pass supervisor actions to loss function for control term
            losses = self.loss_fn(
                pos_error, vel_error, actions_batch, tracking_error, action_target
            )
            total_loss = losses["total"]
        else:
            # Default tracking mode
            losses = self.loss_fn(pos_error, vel_error, actions_batch, tracking_error)
            total_loss = losses["total"]

        # Log losses
        self.loss_logger.log(losses)

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.controller.get_parameters(), self.config.grad_clip
            )

        # Log diagnostics (after backward, before optimizer step)
        if self.diagnostics.enabled:
            # Use mean tracking error and on-target for the batch
            mean_tracking_error = tracking_error.mean().item()
            # Consider batch "on target" if majority (>50%) of samples are within radius
            on_target_threshold = 0.5
            mean_on_target = (
                tracking_error <= self.config.target_radius
            ).float().mean().item() > on_target_threshold

            losses_dict = {
                k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in losses.items()
            }
            self.diagnostics.log_step(
                epoch=self.current_epoch,
                step=self._step_counter,
                observation=features_batch,
                action=actions_batch,
                losses=losses_dict,
                parameters=self.controller.get_parameters(),
                tracking_error=mean_tracking_error,
                on_target=mean_on_target,
            )

        # Optimizer step
        self.optimizer.step()

    def _get_curriculum_difficulty(self, epoch: int) -> float:
        """Compute curriculum difficulty for current epoch."""
        if not self.config.use_curriculum:
            return 1.0

        progress = epoch / max(1, self.config.epochs - 1)
        difficulty = self.config.curriculum_start_difficulty + progress * (
            self.config.curriculum_end_difficulty
            - self.config.curriculum_start_difficulty
        )
        return difficulty

    def _check_training_failure(self, metrics: dict) -> bool:
        """Check if training has failed (NaN loss)."""
        total = metrics.get("total", 0.0)
        return math.isnan(total) or math.isinf(total)

    def _recover_from_nan(self) -> bool:
        """Attempt to recover from NaN gradient."""
        self.nan_recovery_count += 1

        if self.nan_recovery_count > self.config.nan_recovery_attempts:
            return False

        logger.warning(
            "NaN detected, attempting recovery (attempt %d/%d)",
            self.nan_recovery_count,
            self.config.nan_recovery_attempts,
        )

        # Reduce learning rate for the next optimizer
        self.config.learning_rate *= self.config.lr_reduction_factor
        new_lr = self.config.learning_rate
        logger.info("Reduced learning rate to %f", new_lr)

        # Re-create the optimizer to reset its internal state (may be corrupted by NaNs)
        self.optimizer = self._create_optimizer()
        logger.info("Optimizer state has been reset")

        # Try to load best checkpoint
        best_path = self.checkpoint_dir / f"{self.experiment_id}_best.pt"
        if best_path.exists():
            self.controller.load_checkpoint(best_path)
            logger.info("Loaded best checkpoint for recovery")
        else:
            logger.warning(
                "No best checkpoint found to recover from. "
                "Continuing with current model."
            )

        return True

    def _log_epoch(self, epoch: int, metrics: dict) -> None:
        """Log epoch metrics."""
        controller_label = self.config.controller.upper()
        log_msg = (
            f"[{controller_label}] Epoch {epoch:4d} | "
            f"Loss: {metrics.get('total', 0.0):.4f} | "
            f"Reward: {metrics.get('mean_reward', 0.0):.2f} | "
            f"On-target: {metrics.get('mean_on_target_ratio', 0.0):.1%} | "
            f"Error: {metrics.get('mean_tracking_error', 0.0):.3f}m"
        )
        logger.info(log_msg)

        # Store for experiment log (include controller type for comparison)
        self.experiment_log.append(
            {
                "epoch": epoch,
                "controller": self.config.controller,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                **metrics,
            }
        )

    def _save_checkpoint(
        self, epoch: int, metrics: dict, is_best: bool = False, is_final: bool = False
    ) -> None:
        """Save model checkpoint."""
        if is_best:
            filename = f"{self.experiment_id}_best.pt"
        elif is_final:
            filename = f"{self.experiment_id}_final.pt"
        else:
            filename = f"{self.experiment_id}_epoch{epoch:04d}.pt"

        path = self.checkpoint_dir / filename

        metadata = {
            "epoch": epoch,
            "metrics": metrics,
            "config": self.config.to_dict(),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "experiment_id": self.experiment_id,
        }

        self.controller.save_checkpoint(path, metadata=metadata)
        logger.info("Saved checkpoint: %s", path)

    def _save_config(self) -> None:
        """Save training configuration."""
        config_path = self.log_dir / f"{self.experiment_id}_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self.config.to_dict(), f, default_flow_style=False)
        logger.info("Saved config: %s", config_path)

    def _save_experiment_log(self) -> None:
        """Save experiment log to CSV and JSON."""
        # JSON log
        json_path = self.log_dir / f"{self.experiment_id}_log.json"
        with open(json_path, "w") as f:
            json.dump(self.experiment_log, f, indent=2)

        # CSV log
        csv_path = self.log_dir / f"{self.experiment_id}_log.csv"
        if self.experiment_log:
            keys = self.experiment_log[0].keys()
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.experiment_log)

        logger.info("Saved experiment logs to %s", self.log_dir)

    def _save_diagnostics(self) -> None:
        """Save diagnostics if enabled."""
        if not self.diagnostics.enabled:
            return

        # Save all diagnostic files
        self.diagnostics.save_step_log(f"{self.experiment_id}_step_diagnostics.json")
        self.diagnostics.save_epoch_log(f"{self.experiment_id}_epoch_diagnostics.json")
        self.diagnostics.save_epoch_csv(f"{self.experiment_id}_epoch_diagnostics.csv")
        self.diagnostics.save_summary(f"{self.experiment_id}_diagnostics_summary.json")

        # Generate plots
        if self.config.diagnostics_generate_plots:
            self.diagnostics.generate_plots(prefix=self.experiment_id)

        logger.info("Saved diagnostics to %s", self.diagnostics.output_dir)

    def _get_training_summary(self) -> dict:
        """Get summary of training/evaluation results."""
        best_epoch, best_loss = self.loss_logger.get_best_epoch("total")

        summary = {
            "experiment_id": self.experiment_id,
            "controller": self.config.controller,
            "epochs_completed": self.current_epoch + 1,
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            "final_loss": (
                self.experiment_log[-1].get("total", float("inf"))
                if self.experiment_log
                else float("inf")
            ),
            "log_dir": str(self.log_dir),
        }

        # Add deep-specific fields only for deep controllers
        if self.is_deep_controller:
            summary["nan_recoveries"] = self.nan_recovery_count
            summary["checkpoint_dir"] = str(self.checkpoint_dir)

        return summary


def load_checkpoint_and_resume(trainer: Trainer, checkpoint_path: str | Path) -> int:
    """
    Load checkpoint and prepare to resume training.

    Args:
        trainer: Trainer instance.
        checkpoint_path: Path to checkpoint file.

    Returns:
        Epoch number to resume from.
    """
    metadata = trainer.controller.load_checkpoint(checkpoint_path)
    start_epoch = metadata.get("epoch", 0) + 1
    trainer.current_epoch = start_epoch
    logger.info("Resuming training from epoch %d", start_epoch)
    return start_epoch


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train or evaluate quadcopter tracking controllers"
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML/JSON configuration file",
    )

    # Controller selection
    parser.add_argument(
        "--controller",
        type=str,
        choices=["deep", "lqr", "pid"],
        default=None,
        help="Controller type to train/evaluate (default: deep)",
    )

    # Training parameters
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--episodes-per-epoch", type=int, help="Episodes per epoch")
    parser.add_argument("--batch-size", type=int, help="Batch size for updates")
    parser.add_argument("--lr", type=float, dest="learning_rate", help="Learning rate")
    parser.add_argument(
        "--optimizer", type=str, choices=["adam", "sgd", "adamw"], help="Optimizer"
    )

    # Network parameters
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        help="Hidden layer sizes (e.g., 64 64)",
    )
    parser.add_argument(
        "--activation",
        type=str,
        choices=["relu", "tanh", "elu", "leaky_relu"],
        help="Activation function",
    )

    # Training mode
    parser.add_argument(
        "--training-mode",
        type=str,
        choices=["tracking", "imitation", "reward_weighted"],
        help="Training mode: tracking (default), imitation, or reward_weighted",
    )
    parser.add_argument(
        "--supervisor-controller",
        type=str,
        choices=["pid", "lqr"],
        help="Supervisor controller for imitation/reward-weighted modes",
    )
    parser.add_argument(
        "--imitation-weight",
        type=float,
        help="Weight for imitation loss in imitation mode",
    )

    # Environment
    parser.add_argument("--seed", type=int, dest="env_seed", help="Random seed")
    parser.add_argument(
        "--motion-type",
        type=str,
        dest="target_motion_type",
        choices=["linear", "circular", "sinusoidal", "figure8", "stationary"],
        help="Target motion type",
    )
    parser.add_argument(
        "--episode-length", type=float, help="Episode duration in seconds"
    )

    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, help="Checkpoint directory")
    parser.add_argument(
        "--checkpoint-interval", type=int, help="Epochs between checkpoints"
    )
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")

    # Logging
    parser.add_argument("--log-dir", type=str, help="Log directory")
    parser.add_argument("--log-interval", type=int, help="Epochs between logs")

    # Device
    parser.add_argument(
        "--device", type=str, choices=["cpu", "cuda"], help="Device to use"
    )

    # Diagnostics
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        dest="diagnostics_enabled",
        help="Enable training diagnostics",
    )
    parser.add_argument(
        "--diagnostics-output-dir",
        type=str,
        help="Output directory for diagnostics",
    )
    parser.add_argument(
        "--diagnostics-log-interval",
        type=int,
        help="Steps between diagnostic log entries",
    )
    parser.add_argument(
        "--no-diagnostics-plots",
        action="store_false",
        dest="diagnostics_generate_plots",
        default=None,
        help="Disable diagnostic plot generation",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Load config from file if provided
    if args.config:
        config = TrainingConfig.from_file(args.config)
    else:
        config = TrainingConfig()

    # Override with command-line arguments
    for key, value in vars(args).items():
        if value is not None and key != "config" and key != "resume":
            setattr(config, key, value)

    # Create trainer
    try:
        trainer = Trainer(config)
    except ValueError as e:
        logger.error("Configuration error: %s", e)
        return 1

    # Resume from checkpoint if specified (deep controllers only)
    if args.resume:
        if config.controller != "deep":
            logger.warning(
                "Ignoring --resume flag for %s controller (classical controllers "
                "don't support checkpoint resumption)",
                config.controller,
            )
        else:
            load_checkpoint_and_resume(trainer, args.resume)

    # Run training/evaluation
    try:
        summary = trainer.train()
        mode = "Training" if trainer.is_deep_controller else "Evaluation"
        logger.info("%s complete!", mode)
        logger.info("Summary: %s", json.dumps(summary, indent=2))
        return 0
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        if trainer.is_deep_controller:
            trainer._save_checkpoint(trainer.current_epoch, {}, is_final=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
