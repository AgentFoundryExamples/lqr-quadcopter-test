"""Tests for evaluation pipeline and metrics."""

import numpy as np
import pytest

from quadcopter_tracking.controllers import DeepTrackingPolicy
from quadcopter_tracking.env import EnvConfig
from quadcopter_tracking.eval import Evaluator, load_controller
from quadcopter_tracking.utils.metrics import (
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


class TestMetrics:
    """Tests for metrics computation functions."""

    def test_compute_tracking_error_basic(self):
        """Test tracking error computation."""
        quad_pos = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        target_pos = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

        errors = compute_tracking_error(quad_pos, target_pos)

        assert errors.shape == (3,)
        assert np.isclose(errors[0], 0.0)
        assert np.isclose(errors[1], 1.0)
        assert np.isclose(errors[2], 2.0)

    def test_compute_tracking_error_3d(self):
        """Test tracking error in 3D."""
        quad_pos = np.array([[1, 1, 1]])
        target_pos = np.array([[0, 0, 0]])

        errors = compute_tracking_error(quad_pos, target_pos)

        expected = np.sqrt(3)  # sqrt(1^2 + 1^2 + 1^2)
        assert np.isclose(errors[0], expected)

    def test_compute_tracking_error_shape_mismatch(self):
        """Test error on shape mismatch."""
        quad_pos = np.array([[0, 0, 0]])
        target_pos = np.array([[0, 0, 0], [1, 1, 1]])

        with pytest.raises(ValueError):
            compute_tracking_error(quad_pos, target_pos)

    def test_compute_on_target_ratio_all_on(self):
        """Test on-target ratio when always on target."""
        errors = np.array([0.1, 0.2, 0.3, 0.4])
        target_radius = 0.5

        ratio = compute_on_target_ratio(errors, target_radius)

        assert ratio == 1.0

    def test_compute_on_target_ratio_all_off(self):
        """Test on-target ratio when never on target."""
        errors = np.array([1.0, 2.0, 3.0, 4.0])
        target_radius = 0.5

        ratio = compute_on_target_ratio(errors, target_radius)

        assert ratio == 0.0

    def test_compute_on_target_ratio_mixed(self):
        """Test on-target ratio with mixed results."""
        errors = np.array([0.1, 0.6, 0.2, 0.8])  # 2 on, 2 off
        target_radius = 0.5

        ratio = compute_on_target_ratio(errors, target_radius)

        assert ratio == 0.5

    def test_compute_on_target_ratio_empty(self):
        """Test on-target ratio with empty array."""
        errors = np.array([])
        target_radius = 0.5

        ratio = compute_on_target_ratio(errors, target_radius)

        assert ratio == 0.0

    def test_compute_control_effort_basic(self):
        """Test control effort computation."""
        actions = np.array([[10, 0, 0, 0], [10, 1, 0, 0]])

        total, mean = compute_control_effort(actions)

        assert total > 0
        assert mean > 0
        assert np.isclose(mean, total / 2)

    def test_compute_control_effort_empty(self):
        """Test control effort with empty actions."""
        actions = np.array([]).reshape(0, 4)

        total, mean = compute_control_effort(actions)

        assert total == 0.0
        assert mean == 0.0

    def test_detect_overshoots_none(self):
        """Test overshoot detection with no overshoots."""
        # Always on target
        errors = np.ones(100) * 0.3
        target_radius = 0.5

        count, max_overshoot = detect_overshoots(errors, target_radius)

        assert count == 0
        assert max_overshoot == 0.0

    def test_detect_overshoots_single(self):
        """Test overshoot detection with single overshoot."""
        # Start on target, go off, return
        errors = np.concatenate(
            [
                np.ones(30) * 0.3,  # On target
                np.ones(20) * 0.8,  # Off target (overshoot)
                np.ones(30) * 0.3,  # Back on target
            ]
        )
        target_radius = 0.5

        count, max_overshoot = detect_overshoots(errors, target_radius)

        assert count >= 1
        assert max_overshoot > 0.0

    def test_detect_overshoots_short_sequence(self):
        """Test overshoot detection with short sequence."""
        errors = np.array([0.3, 0.8])
        target_radius = 0.5

        count, max_overshoot = detect_overshoots(errors, target_radius)

        # Should handle gracefully
        assert count >= 0


class TestEpisodeMetrics:
    """Tests for episode metrics computation."""

    def test_compute_episode_metrics_basic(self):
        """Test basic episode metrics computation."""
        episode_data = []
        for i in range(100):
            episode_data.append(
                {
                    "time": i * 0.01,
                    "quadcopter_position": [0, 0, 0],
                    "target_position": [0.3, 0, 0],  # 0.3m away
                    "action": [10, 0, 0, 0],
                }
            )

        metrics = compute_episode_metrics(episode_data)

        assert isinstance(metrics, EpisodeMetrics)
        assert metrics.episode_duration > 0
        assert 0 <= metrics.on_target_ratio <= 1
        assert metrics.mean_tracking_error > 0

    def test_compute_episode_metrics_empty(self):
        """Test metrics with empty episode."""
        metrics = compute_episode_metrics([])

        assert not metrics.success
        assert metrics.termination_reason == "no_data"

    def test_compute_episode_metrics_success(self):
        """Test successful episode detection."""
        # Create episode data that should pass success criteria
        episode_data = []
        for i in range(3001):  # Just over 30 seconds at 100Hz
            episode_data.append(
                {
                    "time": i * 0.01,
                    "quadcopter_position": [0, 0, 0],
                    "target_position": [0.1, 0, 0],  # Within 0.5m radius
                    "action": [10, 0, 0, 0],
                }
            )

        criteria = SuccessCriteria(
            min_on_target_ratio=0.8,
            min_episode_duration=30.0,
            target_radius=0.5,
        )

        metrics = compute_episode_metrics(episode_data, criteria)

        assert metrics.success
        assert metrics.on_target_ratio == 1.0

    def test_compute_episode_metrics_failure_duration(self):
        """Test failure due to short duration."""
        episode_data = []
        for i in range(100):  # Only 1 second
            episode_data.append(
                {
                    "time": i * 0.01,
                    "quadcopter_position": [0, 0, 0],
                    "target_position": [0.1, 0, 0],
                    "action": [10, 0, 0, 0],
                }
            )

        criteria = SuccessCriteria(min_episode_duration=30.0)
        metrics = compute_episode_metrics(episode_data, criteria)

        assert not metrics.success

    def test_episode_metrics_to_dict(self):
        """Test metrics serialization."""
        metrics = EpisodeMetrics(
            episode_duration=30.0,
            on_target_ratio=0.85,
            mean_tracking_error=0.3,
        )

        d = metrics.to_dict()

        assert d["episode_duration"] == 30.0
        assert d["on_target_ratio"] == 0.85


class TestEvaluationSummary:
    """Tests for evaluation summary computation."""

    def test_compute_evaluation_summary_basic(self):
        """Test summary computation."""
        metrics_list = [
            EpisodeMetrics(on_target_ratio=0.9, mean_tracking_error=0.2, success=True),
            EpisodeMetrics(on_target_ratio=0.7, mean_tracking_error=0.4, success=False),
            EpisodeMetrics(on_target_ratio=0.8, mean_tracking_error=0.3, success=True),
        ]

        summary = compute_evaluation_summary(metrics_list)

        assert summary.total_episodes == 3
        assert summary.successful_episodes == 2
        assert np.isclose(summary.success_rate, 2 / 3)
        assert np.isclose(summary.mean_on_target_ratio, 0.8)

    def test_compute_evaluation_summary_empty(self):
        """Test summary with no episodes."""
        summary = compute_evaluation_summary([])

        assert summary.total_episodes == 0
        assert summary.successful_episodes == 0

    def test_compute_evaluation_summary_best_worst(self):
        """Test best/worst episode detection."""
        metrics_list = [
            EpisodeMetrics(on_target_ratio=0.5),  # Worst (idx 0)
            EpisodeMetrics(on_target_ratio=0.9),  # Best (idx 1)
            EpisodeMetrics(on_target_ratio=0.7),
        ]

        summary = compute_evaluation_summary(metrics_list)

        assert summary.best_episode_idx == 1
        assert summary.worst_episode_idx == 0

    def test_evaluation_summary_to_dict(self):
        """Test summary serialization."""
        summary = EvaluationSummary(
            total_episodes=10,
            mean_on_target_ratio=0.85,
            meets_criteria=True,
        )

        d = summary.to_dict()

        assert d["total_episodes"] == 10
        assert d["meets_criteria"] is True

    def test_format_metrics_report(self):
        """Test report formatting."""
        metrics_list = [
            EpisodeMetrics(on_target_ratio=0.85, mean_tracking_error=0.3),
        ]
        summary = compute_evaluation_summary(metrics_list)

        report = format_metrics_report(summary)

        assert "EVALUATION SUMMARY" in report
        assert "85.0%" in report or "85%" in report


class TestSuccessCriteria:
    """Tests for success criteria configuration."""

    def test_default_criteria(self):
        """Test default success criteria."""
        criteria = SuccessCriteria()

        assert criteria.min_on_target_ratio == 0.8
        assert criteria.min_episode_duration == 30.0
        assert criteria.target_radius == 0.5

    def test_custom_criteria(self):
        """Test custom success criteria."""
        criteria = SuccessCriteria(
            min_on_target_ratio=0.9,
            min_episode_duration=60.0,
            target_radius=0.3,
        )

        assert criteria.min_on_target_ratio == 0.9


class TestEvaluator:
    """Tests for the Evaluator class."""

    @pytest.fixture
    def evaluator(self, tmp_path):
        """Create evaluator with test controller."""
        controller = DeepTrackingPolicy(device="cpu")
        env_config = EnvConfig()
        env_config.simulation.max_episode_time = 2.0  # Short episodes for testing

        return Evaluator(
            controller=controller,
            env_config=env_config,
            output_dir=tmp_path / "reports",
        )

    def test_evaluator_initialization(self, evaluator):
        """Test evaluator creates correctly."""
        assert evaluator.controller is not None
        assert evaluator.output_dir.exists()

    def test_evaluator_run_episode(self, evaluator):
        """Test running a single episode."""
        episode_data, info = evaluator.run_episode(seed=42, max_steps=100)

        assert len(episode_data) > 0
        assert "time" in episode_data[0]
        assert "quadcopter_position" in episode_data[0]
        assert "tracking_error" in episode_data[0]

    def test_evaluator_evaluate(self, evaluator):
        """Test full evaluation."""
        summary = evaluator.evaluate(num_episodes=2, verbose=False)

        assert isinstance(summary, EvaluationSummary)
        assert summary.total_episodes == 2
        assert len(evaluator.episode_data_list) == 2

    def test_evaluator_save_report(self, evaluator, tmp_path):
        """Test saving evaluation report."""
        summary = evaluator.evaluate(num_episodes=1, verbose=False)
        saved = evaluator.save_report(summary)

        assert "metrics" in saved
        assert saved["metrics"].exists()

    def test_evaluator_plot_trajectory(self, evaluator, tmp_path):
        """Test trajectory plotting."""
        evaluator.evaluate(num_episodes=1, verbose=False)

        plot_path = tmp_path / "trajectory.png"
        fig, ax = evaluator.plot_trajectory(episode_idx=0, save_path=plot_path)

        assert plot_path.exists()
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_evaluator_plot_tracking_error(self, evaluator, tmp_path):
        """Test tracking error plotting."""
        evaluator.evaluate(num_episodes=1, verbose=False)

        plot_path = tmp_path / "error.png"
        fig, ax = evaluator.plot_tracking_error(episode_idx=0, save_path=plot_path)

        assert plot_path.exists()
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_evaluator_generate_all_plots(self, evaluator):
        """Test generating all plots."""
        summary = evaluator.evaluate(num_episodes=1, verbose=False)
        saved_plots = evaluator.generate_all_plots(summary)

        assert len(saved_plots) >= 1
        for path in saved_plots:
            assert path.exists()


class TestLoadController:
    """Tests for controller loading."""

    def test_load_deep_controller(self):
        """Test loading deep learning controller."""
        controller = load_controller("deep")

        assert controller is not None
        assert controller.name == "deep_tracking"

    def test_load_lqr_controller(self):
        """Test loading LQR controller."""
        controller = load_controller("lqr")

        assert controller is not None
        assert controller.name == "lqr"

    def test_load_pid_controller(self):
        """Test loading PID controller."""
        controller = load_controller("pid")

        assert controller is not None
        assert controller.name == "pid"

    def test_load_riccati_lqr_controller(self):
        """Test loading Riccati-LQR controller."""
        controller = load_controller("riccati_lqr")

        assert controller is not None
        assert controller.name == "riccati_lqr"

    def test_load_deep_from_checkpoint(self, tmp_path):
        """Test loading deep controller from checkpoint."""
        # Create and save a controller
        controller1 = DeepTrackingPolicy(device="cpu")
        checkpoint_path = tmp_path / "test.pt"
        controller1.save_checkpoint(checkpoint_path)

        # Load it back
        controller2 = load_controller(
            "deep",
            checkpoint_path=checkpoint_path,
        )

        assert controller2 is not None

    def test_load_unknown_controller(self):
        """Test error on unknown controller type."""
        with pytest.raises(ValueError):
            load_controller("unknown_type")


class TestIntegration:
    """Integration tests for evaluation pipeline."""

    def test_end_to_end_evaluation(self, tmp_path):
        """Test complete evaluation pipeline."""
        # Create controller
        controller = DeepTrackingPolicy(device="cpu")

        # Setup evaluator with short episodes
        env_config = EnvConfig()
        env_config.simulation.max_episode_time = 2.0
        env_config.target.motion_type = "stationary"

        criteria = SuccessCriteria(
            min_episode_duration=1.0,  # Achievable in short test
        )

        evaluator = Evaluator(
            controller=controller,
            env_config=env_config,
            criteria=criteria,
            output_dir=tmp_path / "reports",
        )

        # Run evaluation
        summary = evaluator.evaluate(num_episodes=3, verbose=False)

        # Save outputs
        evaluator.save_report(summary)
        evaluator.generate_all_plots(summary)

        # Verify outputs
        assert (tmp_path / "reports" / "metrics.json").exists()
        assert (tmp_path / "reports" / "plots" / "position_tracking_best.png").exists()

    def test_evaluation_with_different_motion_types(self, tmp_path):
        """Test evaluation with various target motions."""
        motion_types = ["stationary", "linear", "circular"]

        for motion_type in motion_types:
            controller = DeepTrackingPolicy(device="cpu")
            env_config = EnvConfig()
            env_config.simulation.max_episode_time = 1.0
            env_config.target.motion_type = motion_type

            evaluator = Evaluator(
                controller=controller,
                env_config=env_config,
                output_dir=tmp_path / f"reports_{motion_type}",
            )

            summary = evaluator.evaluate(num_episodes=1, verbose=False)
            assert summary.total_episodes == 1


class TestControllerSelectionEval:
    """Tests for controller selection in evaluation pipeline."""

    def test_eval_with_pid_controller(self, tmp_path):
        """Test evaluation with PID controller via load_controller."""
        from quadcopter_tracking.controllers import PIDController

        controller = load_controller("pid")

        assert isinstance(controller, PIDController)
        assert controller.name == "pid"

        # Run short evaluation
        env_config = EnvConfig()
        env_config.simulation.max_episode_time = 1.0

        evaluator = Evaluator(
            controller=controller,
            env_config=env_config,
            output_dir=tmp_path / "reports_pid",
        )

        summary = evaluator.evaluate(num_episodes=2, verbose=False)
        assert summary.total_episodes == 2

    def test_eval_with_lqr_controller(self, tmp_path):
        """Test evaluation with LQR controller via load_controller."""
        from quadcopter_tracking.controllers import LQRController

        controller = load_controller("lqr")

        assert isinstance(controller, LQRController)
        assert controller.name == "lqr"

        # Run short evaluation
        env_config = EnvConfig()
        env_config.simulation.max_episode_time = 1.0

        evaluator = Evaluator(
            controller=controller,
            env_config=env_config,
            output_dir=tmp_path / "reports_lqr",
        )

        summary = evaluator.evaluate(num_episodes=2, verbose=False)
        assert summary.total_episodes == 2

    def test_eval_with_riccati_lqr_controller(self, tmp_path):
        """Test evaluation with Riccati-LQR controller via load_controller."""
        from quadcopter_tracking.controllers import RiccatiLQRController

        controller = load_controller("riccati_lqr")

        assert isinstance(controller, RiccatiLQRController)
        assert controller.name == "riccati_lqr"

        # Run short evaluation
        env_config = EnvConfig()
        env_config.simulation.max_episode_time = 1.0

        evaluator = Evaluator(
            controller=controller,
            env_config=env_config,
            output_dir=tmp_path / "reports_riccati_lqr",
        )

        summary = evaluator.evaluate(num_episodes=2, verbose=False)
        assert summary.total_episodes == 2

    def test_eval_with_deep_controller(self, tmp_path):
        """Test evaluation with deep controller via load_controller."""
        controller = load_controller("deep")

        assert isinstance(controller, DeepTrackingPolicy)
        assert controller.name == "deep_tracking"

        # Run short evaluation
        env_config = EnvConfig()
        env_config.simulation.max_episode_time = 1.0

        evaluator = Evaluator(
            controller=controller,
            env_config=env_config,
            output_dir=tmp_path / "reports_deep",
        )

        summary = evaluator.evaluate(num_episodes=2, verbose=False)
        assert summary.total_episodes == 2

    def test_evaluator_labels_controller_type(self, tmp_path):
        """Test that evaluator correctly identifies controller type."""
        for controller_type, expected_name in [
            ("pid", "pid"),
            ("lqr", "lqr"),
            ("riccati_lqr", "riccati_lqr"),
            ("deep", "deep_tracking"),
        ]:
            controller = load_controller(controller_type)
            env_config = EnvConfig()
            env_config.simulation.max_episode_time = 1.0

            evaluator = Evaluator(
                controller=controller,
                env_config=env_config,
                output_dir=tmp_path / f"reports_{controller_type}",
            )

            assert evaluator.controller.name == expected_name

    def test_all_controllers_generate_comparable_metrics(self, tmp_path):
        """Test that all controllers generate consistent metric formats."""
        for controller_type in ["pid", "lqr", "riccati_lqr", "deep"]:
            controller = load_controller(controller_type)
            env_config = EnvConfig()
            env_config.simulation.max_episode_time = 1.0

            evaluator = Evaluator(
                controller=controller,
                env_config=env_config,
                output_dir=tmp_path / f"reports_{controller_type}",
            )

            summary = evaluator.evaluate(num_episodes=1, verbose=False)

            # All controllers should produce comparable metrics
            assert hasattr(summary, "total_episodes")
            assert hasattr(summary, "mean_on_target_ratio")
            assert hasattr(summary, "mean_tracking_error")
            assert hasattr(summary, "success_rate")
            assert 0 <= summary.mean_on_target_ratio <= 1
            assert summary.mean_tracking_error >= 0


class TestActionSchema:
    """Tests for action schema validation and consistency."""

    def test_action_keys_constant(self):
        """Test that ACTION_KEYS contains the expected keys."""
        from quadcopter_tracking.controllers import ACTION_KEYS

        assert ACTION_KEYS == ("thrust", "roll_rate", "pitch_rate", "yaw_rate")

    def test_validate_action_valid(self):
        """Test validate_action accepts valid action dictionaries."""
        from quadcopter_tracking.controllers import validate_action

        valid_action = {
            "thrust": 10.0,
            "roll_rate": 0.5,
            "pitch_rate": -0.5,
            "yaw_rate": 0.0,
        }
        # Should not raise
        validate_action(valid_action)

    def test_validate_action_missing_key(self):
        """Test validate_action raises on missing keys."""
        from quadcopter_tracking.controllers import validate_action

        incomplete_action = {
            "thrust": 10.0,
            "roll_rate": 0.5,
            # Missing pitch_rate and yaw_rate
        }
        with pytest.raises(KeyError, match="pitch_rate"):
            validate_action(incomplete_action)

    def test_validate_action_wrong_type(self):
        """Test validate_action raises on wrong value types."""
        from quadcopter_tracking.controllers import validate_action

        invalid_action = {
            "thrust": "not a number",
            "roll_rate": 0.5,
            "pitch_rate": -0.5,
            "yaw_rate": 0.0,
        }
        with pytest.raises(TypeError, match="must be numeric"):
            validate_action(invalid_action)

    def test_action_limits_clip_thrust(self):
        """Test ActionLimits clips thrust correctly."""
        from quadcopter_tracking.controllers import ActionLimits

        limits = ActionLimits(min_thrust=0.0, max_thrust=20.0, max_rate=3.0)
        action = {
            "thrust": 25.0,  # Above max
            "roll_rate": 0.0,
            "pitch_rate": 0.0,
            "yaw_rate": 0.0,
        }
        clipped = limits.clip_action(action)
        assert clipped["thrust"] == 20.0

        action["thrust"] = -5.0  # Below min
        clipped = limits.clip_action(action)
        assert clipped["thrust"] == 0.0

    def test_action_limits_clip_rates(self):
        """Test ActionLimits clips rates correctly."""
        from quadcopter_tracking.controllers import ActionLimits

        limits = ActionLimits(max_rate=3.0)
        action = {
            "thrust": 10.0,
            "roll_rate": 5.0,  # Above max
            "pitch_rate": -5.0,  # Below min
            "yaw_rate": 2.0,  # Within bounds
        }
        clipped = limits.clip_action(action)
        assert clipped["roll_rate"] == 3.0
        assert clipped["pitch_rate"] == -3.0
        assert clipped["yaw_rate"] == 2.0

    def test_all_controllers_produce_valid_action_schema(self, tmp_path):
        """Test that all controllers output valid action dictionaries."""
        from quadcopter_tracking.controllers import validate_action
        from quadcopter_tracking.env import QuadcopterEnv

        env = QuadcopterEnv()
        obs = env.reset(seed=42)

        for controller_type in ["pid", "lqr", "riccati_lqr", "deep"]:
            controller = load_controller(controller_type)
            action = controller.compute_action(obs)

            # Should not raise
            validate_action(action)

            # Check action keys
            assert "thrust" in action
            assert "roll_rate" in action
            assert "pitch_rate" in action
            assert "yaw_rate" in action


class TestControllerConfigPropagation:
    """Tests for controller configuration propagation from YAML."""

    def test_load_pid_with_custom_config(self):
        """Test that PID controller receives custom config."""
        config = {
            "kp_pos": [0.02, 0.02, 5.0],
            "ki_pos": [0.001, 0.001, 0.01],
            "kd_pos": [0.1, 0.1, 3.0],
        }
        controller = load_controller("pid", config=config)

        # Verify custom gains were applied
        np.testing.assert_array_almost_equal(controller.kp_pos, [0.02, 0.02, 5.0])
        np.testing.assert_array_almost_equal(controller.ki_pos, [0.001, 0.001, 0.01])
        np.testing.assert_array_almost_equal(controller.kd_pos, [0.1, 0.1, 3.0])

    def test_load_lqr_with_custom_config(self):
        """Test that LQR controller receives custom config."""
        config = {
            "q_pos": [0.001, 0.001, 20.0],
            "q_vel": [0.01, 0.01, 5.0],
            "r_thrust": 2.0,
            "r_rate": 0.5,
        }
        controller = load_controller("lqr", config=config)

        # LQR should have computed K matrix from these weights
        assert controller.K is not None
        assert controller.K.shape == (4, 6)

    def test_load_controller_with_empty_config_uses_defaults(self):
        """Test that controllers use defaults when config is empty."""
        pid = load_controller("pid", config={})
        lqr = load_controller("lqr", config={})
        riccati_lqr = load_controller("riccati_lqr", config={})

        # Check PID uses default gains
        np.testing.assert_array_almost_equal(pid.kp_pos, [0.01, 0.01, 4.0])

        # Check LQR has valid K matrix
        assert lqr.K is not None
        assert lqr.K.shape == (4, 6)

        # Check Riccati-LQR has valid K matrix
        assert riccati_lqr.K is not None
        assert riccati_lqr.K.shape == (4, 6)

    def test_controller_config_propagates_mass_gravity(self):
        """Test that mass/gravity from config reaches controllers."""
        config = {"mass": 1.5, "gravity": 10.0}
        pid = load_controller("pid", config=config)
        lqr = load_controller("lqr", config=config)
        riccati_lqr = load_controller("riccati_lqr", config=config)

        # All should have correct hover thrust
        expected_hover = 1.5 * 10.0
        assert pid.hover_thrust == expected_hover
        assert lqr.hover_thrust == expected_hover
        assert riccati_lqr.hover_thrust == expected_hover

    def test_load_riccati_lqr_with_custom_config(self):
        """Test that Riccati-LQR controller receives custom config."""
        config = {
            "dt": 0.02,
            "q_pos": [0.001, 0.001, 20.0],
            "q_vel": [0.01, 0.01, 5.0],
            "r_controls": [2.0, 1.0, 1.0, 1.0],
        }
        controller = load_controller("riccati_lqr", config=config)

        # Riccati-LQR should have computed K matrix from these weights
        assert controller.K is not None
        assert controller.K.shape == (4, 6)
        assert controller.dt == 0.02


class TestRiccatiControllerSelection:
    """Tests for Riccati-LQR controller selection and config validation."""

    def test_riccati_config_from_yaml_structure(self, tmp_path):
        """Test that Riccati-LQR config follows same schema as PID/LQR."""
        # Config structure matching PID/LQR canonical schema
        config = {
            "dt": 0.01,
            "mass": 1.0,
            "gravity": 9.81,
            "q_pos": [0.0001, 0.0001, 16.0],
            "q_vel": [0.0036, 0.0036, 4.0],
            "r_controls": [1.0, 1.0, 1.0, 1.0],
            "max_thrust": 20.0,
            "min_thrust": 0.0,
            "max_rate": 3.0,
        }
        controller = load_controller("riccati_lqr", config=config)

        assert controller.name == "riccati_lqr"
        assert controller.max_thrust == 20.0
        assert controller.max_rate == 3.0

    def test_riccati_missing_matrices_uses_defaults(self):
        """Test that missing Riccati matrices yield default values, not errors."""
        # Empty config should use defaults, not raise errors
        controller = load_controller("riccati_lqr", config={})

        # Should have valid K matrix from default Q/R
        assert controller.K is not None
        assert controller.K.shape == (4, 6)
        # Should not be using fallback (DARE should solve with defaults)
        assert not controller.is_using_fallback()

    def test_riccati_invalid_q_matrix_raises_validation_error(self):
        """Test that invalid Q matrix produces validation error."""
        # Negative weights would make Q not positive semi-definite
        config = {
            "Q": [
                [-1, 0, 0, 0, 0, 0],  # Negative eigenvalue
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            "fallback_on_failure": False,  # Don't use fallback
        }
        with pytest.raises(ValueError, match="positive semi-definite"):
            load_controller("riccati_lqr", config=config)

    def test_riccati_with_fallback_handles_solver_failure(self):
        """Test that fallback is used when DARE solver fails."""
        # Invalid R matrix (not positive definite) but with fallback enabled
        config = {
            "R": [
                [0, 0, 0, 0],  # Not positive definite (zero eigenvalue)
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            "fallback_on_failure": True,
        }
        controller = load_controller("riccati_lqr", config=config)

        # Should be using fallback
        assert controller.is_using_fallback()

    def test_riccati_produces_valid_actions(self, tmp_path):
        """Test that Riccati-LQR controller produces valid actions."""
        from quadcopter_tracking.controllers import validate_action
        from quadcopter_tracking.env import QuadcopterEnv

        controller = load_controller("riccati_lqr")
        env = QuadcopterEnv()
        obs = env.reset(seed=42)

        action = controller.compute_action(obs)

        # Should not raise
        validate_action(action)

        # Should be within bounds
        assert 0.0 <= action["thrust"] <= 20.0
        assert -3.0 <= action["roll_rate"] <= 3.0
        assert -3.0 <= action["pitch_rate"] <= 3.0
        assert -3.0 <= action["yaw_rate"] <= 3.0
