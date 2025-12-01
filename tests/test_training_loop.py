"""Tests for the training loop and deep learning controller pipeline."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from quadcopter_tracking.controllers import DeepTrackingPolicy, PolicyNetwork
from quadcopter_tracking.env import QuadcopterEnv
from quadcopter_tracking.train import Trainer, TrainingConfig
from quadcopter_tracking.utils.losses import (
    CombinedLoss,
    LossLogger,
    RewardShapingLoss,
    TrackingLoss,
    create_loss_from_config,
)


class TestPolicyNetwork:
    """Tests for the PolicyNetwork class."""

    def test_network_initialization(self):
        """Test network creates with default parameters."""
        network = PolicyNetwork()
        assert network.input_dim == 18
        assert network.output_dim == 4
        assert network.hidden_sizes == [64, 64]

    def test_network_custom_architecture(self):
        """Test network with custom hidden sizes."""
        network = PolicyNetwork(
            input_dim=18,
            hidden_sizes=[32, 64, 32],
            activation="tanh",
        )
        assert network.hidden_sizes == [32, 64, 32]

    def test_network_forward_shape(self):
        """Test forward pass produces correct output shape."""
        network = PolicyNetwork(input_dim=18, hidden_sizes=[32, 32])
        x = torch.randn(8, 18)  # batch of 8
        out = network(x)
        assert out.shape == (8, 4)

    def test_network_output_bounds(self):
        """Test that outputs are bounded correctly."""
        bounds = {
            "thrust": (0.0, 20.0),
            "roll_rate": (-3.0, 3.0),
            "pitch_rate": (-3.0, 3.0),
            "yaw_rate": (-3.0, 3.0),
        }
        network = PolicyNetwork(output_bounds=bounds)
        x = torch.randn(100, 18)  # many samples
        out = network(x)

        # Check bounds
        assert (out[:, 0] >= 0.0).all()
        assert (out[:, 0] <= 20.0).all()
        assert (out[:, 1:] >= -3.0).all()
        assert (out[:, 1:] <= 3.0).all()

    def test_network_activations(self):
        """Test different activation functions."""
        activations = ["relu", "tanh", "elu", "leaky_relu"]
        for activation in activations:
            network = PolicyNetwork(activation=activation)
            x = torch.randn(4, 18)
            out = network(x)
            assert out.shape == (4, 4)

    def test_network_invalid_activation(self):
        """Test that invalid activation raises error."""
        with pytest.raises(ValueError):
            PolicyNetwork(activation="invalid")


class TestDeepTrackingPolicy:
    """Tests for the DeepTrackingPolicy controller."""

    def test_controller_initialization(self):
        """Test controller creates correctly."""
        controller = DeepTrackingPolicy()
        assert controller.name == "deep_tracking"
        assert controller.network is not None

    def test_controller_compute_action(self):
        """Test action computation from observation."""
        controller = DeepTrackingPolicy(device="cpu")
        env = QuadcopterEnv()
        obs = env.reset(seed=42)

        action = controller.compute_action(obs)

        assert "thrust" in action
        assert "roll_rate" in action
        assert "pitch_rate" in action
        assert "yaw_rate" in action
        assert isinstance(action["thrust"], float)

    def test_controller_action_bounds(self):
        """Test that computed actions are within bounds."""
        controller = DeepTrackingPolicy(device="cpu")
        env = QuadcopterEnv()

        for seed in range(10):
            obs = env.reset(seed=seed)
            action = controller.compute_action(obs)

            assert 0.0 <= action["thrust"] <= 20.0
            assert -3.0 <= action["roll_rate"] <= 3.0
            assert -3.0 <= action["pitch_rate"] <= 3.0
            assert -3.0 <= action["yaw_rate"] <= 3.0

    def test_controller_feature_extraction(self):
        """Test feature extraction from observation."""
        controller = DeepTrackingPolicy(device="cpu")
        env = QuadcopterEnv()
        obs = env.reset(seed=42)

        features = controller._extract_features(obs)

        assert features.shape == (18,)
        assert features.dtype == np.float32
        assert np.all(np.isfinite(features))

    def test_controller_checkpoint_roundtrip(self):
        """Test saving and loading checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"

            # Create and save
            controller1 = DeepTrackingPolicy(device="cpu")
            controller1.save_checkpoint(path, metadata={"test": "value"})

            # Load into new controller
            config = {"hidden_sizes": [64, 64], "activation": "relu"}
            controller2 = DeepTrackingPolicy(config=config, device="cpu")
            metadata = controller2.load_checkpoint(path)

            assert metadata["test"] == "value"

            # Verify same outputs
            env = QuadcopterEnv()
            obs = env.reset(seed=42)
            action1 = controller1.compute_action(obs)
            action2 = controller2.compute_action(obs)

            assert np.allclose(action1["thrust"], action2["thrust"])

    def test_controller_train_eval_modes(self):
        """Test switching between train and eval modes."""
        controller = DeepTrackingPolicy(device="cpu")

        controller.train_mode()
        assert controller.network.training

        controller.eval_mode()
        assert not controller.network.training

    def test_controller_with_environment_step(self):
        """Test controller works in environment loop."""
        controller = DeepTrackingPolicy(device="cpu")
        env = QuadcopterEnv()
        obs = env.reset(seed=42)

        # Run a few steps
        for _ in range(10):
            action = controller.compute_action(obs)
            obs, reward, done, info = env.step(action)

            if done:
                break

        # Should complete without errors
        assert obs is not None


class TestTrackingLoss:
    """Tests for tracking loss functions."""

    def test_tracking_loss_initialization(self):
        """Test loss function creates correctly."""
        loss = TrackingLoss()
        assert loss.error_type == "l2"

    def test_tracking_loss_forward(self):
        """Test loss computation."""
        loss = TrackingLoss(device="cpu")

        pos_error = torch.randn(8, 3, requires_grad=True)
        vel_error = torch.randn(8, 3, requires_grad=True)
        action = torch.randn(8, 4, requires_grad=True)

        result = loss(pos_error, vel_error, action)

        assert "total" in result
        assert "position" in result
        assert "velocity" in result
        assert "control" in result
        assert result["total"].requires_grad

    def test_tracking_loss_weight_matrix(self):
        """Test custom weight matrices."""
        pos_weight = np.diag([2.0, 2.0, 1.0])
        loss = TrackingLoss(position_weight=pos_weight, device="cpu")

        pos_error = torch.ones(4, 3)
        vel_error = torch.zeros(4, 3)
        action = torch.zeros(4, 4)

        result = loss(pos_error, vel_error, action)
        assert result["position"].item() > 0

    def test_tracking_loss_error_types(self):
        """Test different error norms."""
        for error_type in ["l2", "l1", "huber"]:
            loss = TrackingLoss(error_type=error_type, device="cpu")
            pos_error = torch.randn(4, 3)
            vel_error = torch.randn(4, 3)
            action = torch.randn(4, 4)

            result = loss(pos_error, vel_error, action)
            assert torch.isfinite(result["total"])

    def test_reward_shaping_loss(self):
        """Test reward shaping computation."""
        loss = RewardShapingLoss(target_radius=0.5, device="cpu")

        # On target
        error_close = torch.tensor([0.3])
        result_close = loss(error_close)
        assert result_close["on_target"].all()

        # Off target
        error_far = torch.tensor([2.0])
        result_far = loss(error_far)
        assert not result_far["on_target"].any()


class TestCombinedLoss:
    """Tests for combined loss function."""

    def test_combined_loss_creation(self):
        """Test combined loss creates correctly."""
        loss = CombinedLoss(device="cpu")
        assert loss.tracking_loss is not None
        assert loss.reward_loss is not None

    def test_combined_loss_forward(self):
        """Test combined loss computation."""
        loss = CombinedLoss(
            tracking_weight=1.0,
            reward_weight=0.1,
            device="cpu",
        )

        result = loss(
            position_error=torch.randn(4, 3),
            velocity_error=torch.randn(4, 3),
            action=torch.randn(4, 4),
            tracking_error=torch.rand(4),
        )

        assert "total" in result
        assert "tracking_total" in result
        assert "reward" in result
        assert "on_target_ratio" in result

    def test_create_loss_from_config(self):
        """Test loss creation from config dict."""
        config = {
            "position_weight": 2.0,
            "velocity_weight": 0.5,
            "error_type": "huber",
            "target_radius": 0.3,
            "device": "cpu",
        }

        loss = create_loss_from_config(config)
        assert isinstance(loss, CombinedLoss)


class TestLossLogger:
    """Tests for loss logging."""

    def test_logger_accumulation(self):
        """Test loss value accumulation."""
        logger = LossLogger()

        for i in range(10):
            logger.log({"total": float(i), "position": float(i * 2)})

        means = logger.end_epoch()

        assert means["total"] == 4.5  # mean of 0-9
        assert means["position"] == 9.0

    def test_logger_history(self):
        """Test history tracking across epochs."""
        log = LossLogger()

        for epoch in range(3):
            for step in range(5):
                log.log({"total": float(epoch + step)})
            log.end_epoch()

        history = log.get_history()
        assert len(history) == 3

    def test_logger_best_epoch(self):
        """Test finding best epoch."""
        log = LossLogger()

        # Epoch 0: mean = 5
        for _ in range(5):
            log.log({"total": 5.0})
        log.end_epoch()

        # Epoch 1: mean = 2
        for _ in range(5):
            log.log({"total": 2.0})
        log.end_epoch()

        # Epoch 2: mean = 8
        for _ in range(5):
            log.log({"total": 8.0})
        log.end_epoch()

        best_idx, best_val = log.get_best_epoch("total")
        assert best_idx == 1
        assert best_val == 2.0


class TestTrainingConfig:
    """Tests for training configuration."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = TrainingConfig()
        assert config.epochs == 100
        assert config.learning_rate == 0.001
        assert config.hidden_sizes == [64, 64]
        assert config.controller == "deep"  # Default controller

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "epochs": 50,
            "learning_rate": 0.0001,
            "hidden_sizes": [128, 128],
        }
        config = TrainingConfig.from_dict(config_dict)

        assert config.epochs == 50
        assert config.learning_rate == 0.0001
        assert config.hidden_sizes == [128, 128]

    def test_config_to_dict(self):
        """Test serializing config to dictionary."""
        config = TrainingConfig(epochs=25, env_seed=123)
        config_dict = config.to_dict()

        assert config_dict["epochs"] == 25
        assert config_dict["env_seed"] == 123
        assert config_dict["controller"] == "deep"

    def test_config_from_file(self):
        """Test loading config from YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("epochs: 10\nlearning_rate: 0.01\n")
            f.flush()

            config = TrainingConfig.from_file(f.name)
            assert config.epochs == 10
            assert config.learning_rate == 0.01

    def test_config_controller_deep(self):
        """Test controller selection for deep."""
        config = TrainingConfig(controller="deep")
        assert config.controller == "deep"

    def test_config_controller_pid(self):
        """Test controller selection for pid."""
        config = TrainingConfig(controller="pid")
        assert config.controller == "pid"

    def test_config_controller_lqr(self):
        """Test controller selection for lqr."""
        config = TrainingConfig(controller="lqr")
        assert config.controller == "lqr"

    def test_config_invalid_controller(self):
        """Test error on invalid controller type."""
        with pytest.raises(ValueError, match="Invalid controller type"):
            TrainingConfig(controller="invalid")

    def test_config_controller_from_yaml(self):
        """Test loading controller selection from YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("controller: lqr\nepochs: 5\n")
            f.flush()

            config = TrainingConfig.from_file(f.name)
            assert config.controller == "lqr"
            assert config.epochs == 5


class TestTrainer:
    """Tests for the Trainer class."""

    @pytest.fixture
    def fast_config(self, tmp_path):
        """Create fast training config for tests."""
        return TrainingConfig(
            epochs=2,
            episodes_per_epoch=2,
            max_steps_per_episode=100,
            batch_size=8,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            log_dir=str(tmp_path / "logs"),
            device="cpu",
        )

    def test_trainer_initialization(self, fast_config):
        """Test trainer creates correctly."""
        trainer = Trainer(fast_config)

        assert trainer.controller is not None
        assert trainer.env is not None
        assert trainer.loss_fn is not None
        assert trainer.optimizer is not None

    def test_trainer_single_epoch(self, fast_config):
        """Test running a single training epoch."""
        trainer = Trainer(fast_config)
        trainer.controller.train_mode()

        metrics = trainer._train_epoch()

        assert "mean_reward" in metrics
        assert "mean_on_target_ratio" in metrics
        assert "mean_tracking_error" in metrics

    def test_trainer_curriculum_difficulty(self, fast_config):
        """Test curriculum difficulty calculation."""
        fast_config.use_curriculum = True
        fast_config.curriculum_start_difficulty = 0.2
        fast_config.curriculum_end_difficulty = 1.0
        fast_config.epochs = 10

        trainer = Trainer(fast_config)

        # First epoch
        diff_0 = trainer._get_curriculum_difficulty(0)
        assert abs(diff_0 - 0.2) < 0.01

        # Last epoch
        diff_9 = trainer._get_curriculum_difficulty(9)
        assert abs(diff_9 - 1.0) < 0.01

        # Middle epoch
        diff_4 = trainer._get_curriculum_difficulty(4)
        assert 0.4 < diff_4 < 0.7

    def test_trainer_checkpoint_save_load(self, fast_config):
        """Test checkpoint save and load."""
        trainer = Trainer(fast_config)
        trainer.current_epoch = 5

        # Save checkpoint
        trainer._save_checkpoint(5, {"test_metric": 0.5})

        # Verify file exists
        checkpoint_path = (
            Path(fast_config.checkpoint_dir) / f"{trainer.experiment_id}_epoch0005.pt"
        )
        assert checkpoint_path.exists()

    def test_trainer_nan_recovery(self, fast_config):
        """Test NaN recovery mechanism."""
        trainer = Trainer(fast_config)

        # First recovery should succeed
        assert trainer._recover_from_nan()
        assert trainer.nan_recovery_count == 1

        # Exhaust recovery attempts
        fast_config.nan_recovery_attempts = 1
        trainer.nan_recovery_count = 1
        assert not trainer._recover_from_nan()

    def test_training_data_flow(self, fast_config):
        """Integration test for training data flow."""
        trainer = Trainer(fast_config)

        # Run minimal training
        summary = trainer.train()

        assert "experiment_id" in summary
        assert "epochs_completed" in summary
        assert summary["epochs_completed"] == 2
        assert "best_loss" in summary
        assert "log_dir" in summary

    def test_optimizer_types(self, fast_config):
        """Test different optimizer types."""
        for opt_name in ["adam", "sgd", "adamw"]:
            fast_config.optimizer = opt_name
            trainer = Trainer(fast_config)
            assert trainer.optimizer is not None


class TestTrainingModes:
    """Tests for configurable training modes."""

    @pytest.fixture
    def mode_config(self, tmp_path):
        """Create config for training mode tests."""
        return {
            "epochs": 2,
            "episodes_per_epoch": 2,
            "max_steps_per_episode": 50,
            "batch_size": 8,
            "hidden_sizes": [16, 16],
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "log_dir": str(tmp_path / "logs"),
            "device": "cpu",
        }

    def test_tracking_mode_default(self, mode_config, tmp_path):
        """Test that tracking mode is the default."""
        config = TrainingConfig.from_dict(mode_config)
        assert config.training_mode == "tracking"

    def test_imitation_mode_config(self, mode_config, tmp_path):
        """Test imitation mode configuration."""
        mode_config["training_mode"] = "imitation"
        mode_config["supervisor_controller"] = "pid"
        mode_config["imitation_weight"] = 2.0

        config = TrainingConfig.from_dict(mode_config)
        assert config.training_mode == "imitation"
        assert config.supervisor_controller == "pid"
        assert config.imitation_weight == 2.0

    def test_reward_weighted_mode_config(self, mode_config, tmp_path):
        """Test reward-weighted mode configuration."""
        mode_config["training_mode"] = "reward_weighted"
        mode_config["supervisor_controller"] = "lqr"

        config = TrainingConfig.from_dict(mode_config)
        assert config.training_mode == "reward_weighted"
        assert config.supervisor_controller == "lqr"

    def test_invalid_training_mode(self, mode_config):
        """Test error on invalid training mode."""
        mode_config["training_mode"] = "invalid_mode"
        with pytest.raises(ValueError, match="Invalid training_mode"):
            TrainingConfig.from_dict(mode_config)

    def test_invalid_supervisor_controller(self, mode_config):
        """Test error on invalid supervisor controller."""
        mode_config["supervisor_controller"] = "invalid_supervisor"
        with pytest.raises(ValueError, match="Invalid supervisor_controller"):
            TrainingConfig.from_dict(mode_config)

    def test_trainer_creates_supervisor_for_imitation(self, mode_config, tmp_path):
        """Test that trainer creates supervisor controller in imitation mode."""
        mode_config["training_mode"] = "imitation"
        mode_config["supervisor_controller"] = "pid"

        config = TrainingConfig.from_dict(mode_config)
        trainer = Trainer(config)

        assert trainer.supervisor is not None
        assert trainer.supervisor.name == "pid"

    def test_trainer_creates_supervisor_for_reward_weighted(
        self, mode_config, tmp_path
    ):
        """Test that trainer creates supervisor in reward_weighted mode."""
        mode_config["training_mode"] = "reward_weighted"
        mode_config["supervisor_controller"] = "lqr"

        config = TrainingConfig.from_dict(mode_config)
        trainer = Trainer(config)

        assert trainer.supervisor is not None
        assert trainer.supervisor.name == "lqr"

    def test_trainer_no_supervisor_for_tracking(self, mode_config, tmp_path):
        """Test that no supervisor is created in default tracking mode."""
        mode_config["training_mode"] = "tracking"

        config = TrainingConfig.from_dict(mode_config)
        trainer = Trainer(config)

        assert trainer.supervisor is None

    def test_imitation_mode_training(self, mode_config, tmp_path):
        """Test that imitation mode training runs without error."""
        mode_config["training_mode"] = "imitation"
        mode_config["supervisor_controller"] = "pid"
        mode_config["target_motion_type"] = "stationary"

        config = TrainingConfig.from_dict(mode_config)
        trainer = Trainer(config)
        summary = trainer.train()

        assert summary["epochs_completed"] == 2
        assert "best_loss" in summary

    def test_reward_weighted_mode_training(self, mode_config, tmp_path):
        """Test that reward-weighted mode training runs without error."""
        mode_config["training_mode"] = "reward_weighted"
        mode_config["supervisor_controller"] = "lqr"
        mode_config["target_motion_type"] = "stationary"

        config = TrainingConfig.from_dict(mode_config)
        trainer = Trainer(config)
        summary = trainer.train()

        assert summary["epochs_completed"] == 2
        assert "best_loss" in summary

    def test_imitation_mode_logs_imitation_loss(self, mode_config, tmp_path):
        """Test that imitation mode logs imitation loss component."""
        mode_config["training_mode"] = "imitation"
        mode_config["supervisor_controller"] = "pid"
        mode_config["target_motion_type"] = "stationary"

        config = TrainingConfig.from_dict(mode_config)
        trainer = Trainer(config)
        trainer.train()

        # Check that experiment log contains metrics
        assert len(trainer.experiment_log) > 0

    def test_config_to_dict_includes_training_mode(self, mode_config, tmp_path):
        """Test that to_dict includes training mode parameters."""
        mode_config["training_mode"] = "imitation"
        mode_config["supervisor_controller"] = "pid"
        mode_config["imitation_weight"] = 1.5

        config = TrainingConfig.from_dict(mode_config)
        config_dict = config.to_dict()

        assert config_dict["training_mode"] == "imitation"
        assert config_dict["supervisor_controller"] == "pid"
        assert config_dict["imitation_weight"] == 1.5

    def test_supervisor_receives_full_config(self, mode_config, tmp_path):
        """Test that supervisor controller receives config from YAML."""
        # Add custom PID gains to the config
        mode_config["training_mode"] = "imitation"
        mode_config["supervisor_controller"] = "pid"
        mode_config["pid"] = {
            "kp_pos": [3.0, 3.0, 5.0],
            "ki_pos": [0.2, 0.2, 0.3],
            "kd_pos": [2.0, 2.0, 2.5],
        }

        config = TrainingConfig.from_dict(mode_config)
        trainer = Trainer(config)

        # Verify the full_config was preserved
        assert hasattr(config, "full_config")
        assert "pid" in config.full_config
        assert config.full_config["pid"]["kp_pos"] == [3.0, 3.0, 5.0]

        # Verify trainer has access to full_config
        assert trainer.full_config.get("pid") is not None


class TestControllerSelection:
    """Tests for controller selection in training."""

    @pytest.fixture
    def base_config(self, tmp_path):
        """Create base config for controller tests."""
        return {
            "epochs": 2,
            "episodes_per_epoch": 2,
            "max_steps_per_episode": 100,
            "batch_size": 8,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "log_dir": str(tmp_path / "logs"),
            "device": "cpu",
        }

    def test_trainer_with_pid_controller(self, base_config, tmp_path):
        """Test training/evaluation with PID controller."""
        base_config["controller"] = "pid"
        config = TrainingConfig.from_dict(base_config)
        trainer = Trainer(config)

        assert trainer.controller.name == "pid"
        assert not trainer.is_deep_controller
        assert trainer.optimizer is None

        # Run evaluation (not training)
        summary = trainer.train()

        assert summary["controller"] == "pid"
        assert summary["epochs_completed"] == 2
        # No checkpoint_dir for classical controllers
        assert "checkpoint_dir" not in summary

    def test_trainer_with_lqr_controller(self, base_config, tmp_path):
        """Test training/evaluation with LQR controller."""
        base_config["controller"] = "lqr"
        config = TrainingConfig.from_dict(base_config)
        trainer = Trainer(config)

        assert trainer.controller.name == "lqr"
        assert not trainer.is_deep_controller
        assert trainer.optimizer is None

        # Run evaluation (not training)
        summary = trainer.train()

        assert summary["controller"] == "lqr"
        assert summary["epochs_completed"] == 2

    def test_trainer_with_deep_controller(self, base_config, tmp_path):
        """Test training with deep controller."""
        base_config["controller"] = "deep"
        config = TrainingConfig.from_dict(base_config)
        trainer = Trainer(config)

        assert trainer.controller.name == "deep_tracking"
        assert trainer.is_deep_controller
        assert trainer.optimizer is not None

        # Run training
        summary = trainer.train()

        assert summary["controller"] == "deep"
        assert summary["epochs_completed"] == 2
        assert "checkpoint_dir" in summary
        assert "nan_recoveries" in summary

    def test_classical_controller_no_checkpoint(self, base_config, tmp_path):
        """Test that classical controllers don't create checkpoints."""
        base_config["controller"] = "pid"
        config = TrainingConfig.from_dict(base_config)
        trainer = Trainer(config)

        # Run evaluation
        trainer.train()

        # Checkpoint directory should not be created for classical controllers
        checkpoint_dir = tmp_path / "checkpoints"
        assert not checkpoint_dir.exists()

    def test_experiment_id_includes_controller_type(self, base_config, tmp_path):
        """Test that experiment ID includes controller type."""
        for controller_type in ["deep", "pid", "lqr"]:
            base_config["controller"] = controller_type
            config = TrainingConfig.from_dict(base_config)
            trainer = Trainer(config)

            assert f"train_{controller_type}_" in trainer.experiment_id

    def test_log_epoch_includes_controller_label(self, base_config, tmp_path):
        """Test that epoch logs include controller type."""
        base_config["controller"] = "pid"
        config = TrainingConfig.from_dict(base_config)
        trainer = Trainer(config)

        # Run evaluation to populate experiment log
        trainer.train()

        # Check that logs include controller type
        assert len(trainer.experiment_log) > 0
        for entry in trainer.experiment_log:
            assert entry["controller"] == "pid"

    def test_pid_receives_config_from_yaml(self, base_config, tmp_path):
        """Test that PID controller receives gains from full_config."""
        base_config["controller"] = "pid"
        base_config["pid"] = {
            "kp_pos": [0.02, 0.02, 5.0],
            "ki_pos": [0.0, 0.0, 0.0],
            "kd_pos": [0.08, 0.08, 2.5],
        }
        config = TrainingConfig.from_dict(base_config)
        trainer = Trainer(config)

        # Verify PID controller received custom gains
        import numpy as np

        np.testing.assert_array_almost_equal(
            trainer.controller.kp_pos, [0.02, 0.02, 5.0]
        )
        np.testing.assert_array_almost_equal(
            trainer.controller.kd_pos, [0.08, 0.08, 2.5]
        )

    def test_lqr_receives_config_from_yaml(self, base_config, tmp_path):
        """Test that LQR controller receives weights from full_config."""
        base_config["controller"] = "lqr"
        base_config["lqr"] = {
            "q_pos": [0.001, 0.001, 20.0],
            "q_vel": [0.01, 0.01, 5.0],
            "r_thrust": 2.0,
            "r_rate": 0.5,
        }
        config = TrainingConfig.from_dict(base_config)
        trainer = Trainer(config)

        # Verify LQR controller was created with a K matrix
        assert trainer.controller.K is not None
        assert trainer.controller.K.shape == (4, 6)

    def test_classical_controller_uses_defaults_without_yaml_config(
        self, base_config, tmp_path
    ):
        """Test that controllers use defaults when no YAML config provided."""
        import numpy as np

        base_config["controller"] = "pid"
        # No 'pid' section in config
        config = TrainingConfig.from_dict(base_config)
        trainer = Trainer(config)

        # Verify default gains were used
        np.testing.assert_array_almost_equal(
            trainer.controller.kp_pos, [0.01, 0.01, 4.0]
        )


class TestIntegration:
    """Integration tests for the complete training pipeline."""

    def test_end_to_end_training(self, tmp_path):
        """Test complete training pipeline end-to-end."""
        config = TrainingConfig(
            epochs=3,
            episodes_per_epoch=2,
            max_steps_per_episode=50,
            batch_size=4,
            hidden_sizes=[16, 16],
            checkpoint_dir=str(tmp_path / "e2e_checkpoints"),
            log_dir=str(tmp_path / "e2e_logs"),
            device="cpu",
        )

        trainer = Trainer(config)
        summary = trainer.train()

        # Verify training completed
        assert summary["epochs_completed"] == 3

        # Verify logs were created
        log_dir = Path(config.log_dir)
        assert log_dir.exists()

        # Verify checkpoint was saved
        checkpoint_dir = Path(config.checkpoint_dir)
        assert checkpoint_dir.exists()

    def test_controller_improves_with_training(self, tmp_path):
        """Test that controller behavior changes with training."""
        config = TrainingConfig(
            epochs=5,
            episodes_per_epoch=3,
            max_steps_per_episode=100,
            batch_size=8,
            hidden_sizes=[32, 32],
            checkpoint_dir=str(tmp_path / "improvement_checkpoints"),
            log_dir=str(tmp_path / "improvement_logs"),
            device="cpu",
        )

        trainer = Trainer(config)

        # Get initial action for reference observation
        env = QuadcopterEnv()
        obs = env.reset(seed=42)
        initial_action = trainer.controller.compute_action(obs)

        # Train
        trainer.train()

        # Get action after training
        final_action = trainer.controller.compute_action(obs)

        # Actions should be different (network has been updated)
        # Note: Not necessarily "better", just different
        assert not np.allclose(
            list(initial_action.values()),
            list(final_action.values()),
            rtol=0.01,
        )

    def test_training_with_different_motion_types(self, tmp_path):
        """Test training with various target motion types."""
        motion_types = ["stationary", "linear", "circular"]

        for motion_type in motion_types:
            config = TrainingConfig(
                epochs=2,
                episodes_per_epoch=2,
                max_steps_per_episode=50,
                target_motion_type=motion_type,
                checkpoint_dir=str(tmp_path / f"motion_{motion_type}"),
                log_dir=str(tmp_path / f"logs_{motion_type}"),
                device="cpu",
            )

            trainer = Trainer(config)
            summary = trainer.train()
            assert summary["epochs_completed"] == 2


class TestDiagnostics:
    """Tests for training diagnostics."""

    def test_diagnostics_config_defaults(self):
        """Test DiagnosticsConfig default values."""
        from quadcopter_tracking.utils.diagnostics import DiagnosticsConfig

        config = DiagnosticsConfig()
        assert config.enabled is False
        assert config.log_observations is True
        assert config.log_actions is True
        assert config.log_gradients is True
        assert config.log_interval == 10

    def test_diagnostics_config_from_dict(self):
        """Test creating DiagnosticsConfig from dictionary."""
        from quadcopter_tracking.utils.diagnostics import DiagnosticsConfig

        config_dict = {
            "enabled": True,
            "log_observations": False,
            "log_interval": 5,
        }
        config = DiagnosticsConfig.from_dict(config_dict)

        assert config.enabled is True
        assert config.log_observations is False
        assert config.log_interval == 5

    def test_diagnostics_disabled_by_default(self, tmp_path):
        """Test diagnostics are disabled by default."""
        from quadcopter_tracking.utils.diagnostics import Diagnostics, DiagnosticsConfig

        config = DiagnosticsConfig(enabled=False)
        diag = Diagnostics(config=config, output_dir=tmp_path)

        assert diag.enabled is False

        # Logging should be no-op when disabled
        result = diag.log_step(epoch=0, step=0)
        assert result is None

    def test_diagnostics_enabled_logging(self, tmp_path):
        """Test diagnostics logging when enabled."""
        from quadcopter_tracking.utils.diagnostics import Diagnostics, DiagnosticsConfig

        config = DiagnosticsConfig(
            enabled=True,
            log_interval=1,  # Log every step
            output_dir=str(tmp_path),
        )
        diag = Diagnostics(config=config)

        assert diag.enabled is True

        # Log some steps
        for step in range(5):
            action = {
                "thrust": 10.0,
                "roll_rate": 0.0,
                "pitch_rate": 0.0,
                "yaw_rate": 0.0,
            }
            diag.log_step(
                epoch=0,
                step=step,
                observation=np.random.randn(18),
                action=action,
                losses={"total": 1.0, "position": 0.5, "velocity": 0.3},
                tracking_error=0.4,
                on_target=True,
            )

        # End epoch
        epoch_diag = diag.log_epoch(epoch=0)

        assert len(diag.step_log) == 5
        assert len(diag.epoch_log) == 1
        assert epoch_diag.epoch == 0

    def test_diagnostics_gradient_stats(self):
        """Test gradient statistics computation."""
        from quadcopter_tracking.utils.diagnostics import compute_gradient_stats

        # Create a simple model
        model = torch.nn.Linear(10, 5)
        x = torch.randn(4, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()

        stats = compute_gradient_stats(model.parameters())

        assert stats.norm > 0
        assert stats.num_nan == 0
        assert stats.num_inf == 0

    def test_diagnostics_observation_stats(self):
        """Test observation statistics computation."""
        from quadcopter_tracking.utils.diagnostics import compute_observation_stats

        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = compute_observation_stats(obs)

        assert stats["mean"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["num_nan"] == 0
        assert stats["num_inf"] == 0

    def test_diagnostics_observation_stats_handles_nan(self):
        """Test observation stats handle NaN gracefully."""
        from quadcopter_tracking.utils.diagnostics import compute_observation_stats

        obs = np.array([1.0, float("nan"), 3.0])
        stats = compute_observation_stats(obs)

        assert stats["num_nan"] == 1
        # Mean should be computed from valid values only
        assert stats["mean"] == 2.0

    def test_diagnostics_action_stats(self):
        """Test action statistics computation."""
        from quadcopter_tracking.utils.diagnostics import compute_action_stats

        action = {
            "thrust": 10.0,
            "roll_rate": 0.5,
            "pitch_rate": -0.5,
            "yaw_rate": 0.0,
        }
        stats = compute_action_stats(action)

        assert stats["thrust_mean"] == 10.0
        assert stats["total_magnitude"] > 0

    def test_diagnostics_save_files(self, tmp_path):
        """Test diagnostics file saving."""
        from quadcopter_tracking.utils.diagnostics import Diagnostics, DiagnosticsConfig

        config = DiagnosticsConfig(
            enabled=True,
            log_interval=1,
            output_dir=str(tmp_path),
            generate_plots=False,  # Skip plots for this test
        )
        diag = Diagnostics(config=config)

        # Log some data
        for epoch in range(2):
            for step in range(3):
                diag.log_step(
                    epoch=epoch,
                    step=step,
                    observation=np.random.randn(18),
                    losses={"total": float(epoch + step)},
                    tracking_error=0.5,
                    on_target=True,
                )
            diag.log_epoch(epoch=epoch)

        # Save files
        step_path = diag.save_step_log()
        epoch_path = diag.save_epoch_log()
        csv_path = diag.save_epoch_csv()
        summary_path = diag.save_summary()

        assert step_path is not None and step_path.exists()
        assert epoch_path is not None and epoch_path.exists()
        assert csv_path is not None and csv_path.exists()
        assert summary_path is not None and summary_path.exists()

    def test_training_with_diagnostics_enabled(self, tmp_path):
        """Test training with diagnostics enabled."""
        config = TrainingConfig(
            epochs=2,
            episodes_per_epoch=2,
            max_steps_per_episode=50,
            batch_size=4,
            hidden_sizes=[16, 16],
            checkpoint_dir=str(tmp_path / "checkpoints"),
            log_dir=str(tmp_path / "logs"),
            device="cpu",
            diagnostics_enabled=True,
            diagnostics_log_interval=1,
            diagnostics_generate_plots=False,
        )

        trainer = Trainer(config)
        summary = trainer.train()

        assert summary["epochs_completed"] == 2

        # Check diagnostics were saved
        diag_dir = Path(config.log_dir) / "diagnostics"
        assert diag_dir.exists()

    def test_training_with_diagnostics_disabled(self, tmp_path):
        """Test training with diagnostics disabled doesn't create files."""
        config = TrainingConfig(
            epochs=2,
            episodes_per_epoch=2,
            max_steps_per_episode=50,
            batch_size=4,
            hidden_sizes=[16, 16],
            checkpoint_dir=str(tmp_path / "checkpoints"),
            log_dir=str(tmp_path / "logs"),
            device="cpu",
            diagnostics_enabled=False,
        )

        trainer = Trainer(config)
        summary = trainer.train()

        assert summary["epochs_completed"] == 2

        # Diagnostics dir should not have diagnostic files
        diag_dir = Path(config.log_dir) / "diagnostics"
        # The directory may or may not exist, but should be empty or not have many files
        if diag_dir.exists():
            # If it exists, should have no or minimal files
            files = list(diag_dir.glob("*diagnostics*.json"))
            assert len(files) == 0

    def test_diagnostics_throttling(self, tmp_path):
        """Test diagnostics respects log_interval for throttling."""
        from quadcopter_tracking.utils.diagnostics import Diagnostics, DiagnosticsConfig

        config = DiagnosticsConfig(
            enabled=True,
            log_interval=5,  # Only log every 5th step
            output_dir=str(tmp_path),
        )
        diag = Diagnostics(config=config)

        # Log 20 steps
        for step in range(20):
            diag.log_step(
                epoch=0,
                step=step,
                observation=np.random.randn(18),
                losses={"total": 1.0},
            )

        # With log_interval=5, logs at internal step counts 5, 10, 15, 20
        # (every 5th internal step counter increment)
        assert len(diag.step_log) == 4

    def test_diagnostics_max_entries_per_epoch(self, tmp_path):
        """Test diagnostics respects max_entries_per_epoch."""
        from quadcopter_tracking.utils.diagnostics import Diagnostics, DiagnosticsConfig

        config = DiagnosticsConfig(
            enabled=True,
            log_interval=1,
            max_entries_per_epoch=5,
            output_dir=str(tmp_path),
        )
        diag = Diagnostics(config=config)

        # Try to log 100 steps
        for step in range(100):
            diag.log_step(
                epoch=0,
                step=step,
                observation=np.random.randn(18),
                losses={"total": 1.0},
            )

        # Should only have max_entries_per_epoch entries
        assert len(diag.step_log) == 5
