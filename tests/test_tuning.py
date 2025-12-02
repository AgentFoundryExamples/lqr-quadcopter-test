"""Tests for the controller auto-tuning module."""

import json

import pytest

from quadcopter_tracking.controllers.tuning import (
    ControllerTuner,
    GainSearchSpace,
    TuningConfig,
    TuningResult,
)


class TestGainSearchSpace:
    """Tests for GainSearchSpace validation and configuration."""

    def test_empty_search_space(self):
        """Test that empty search space is valid."""
        space = GainSearchSpace()
        space.validate()  # Should not raise
        assert space.get_active_parameters() == []

    def test_valid_kp_range(self):
        """Test valid proportional gain range."""
        space = GainSearchSpace(kp_pos_range=([0.005, 0.005, 2.0], [0.05, 0.05, 6.0]))
        space.validate()  # Should not raise
        assert "kp_pos" in space.get_active_parameters()

    def test_inverted_range_raises(self):
        """Test that inverted ranges raise ValueError."""
        space = GainSearchSpace(
            kp_pos_range=([0.1, 0.1, 6.0], [0.05, 0.05, 2.0])  # min > max
        )
        with pytest.raises(ValueError, match="inverted range"):
            space.validate()

    def test_negative_range_raises(self):
        """Test that negative ranges raise ValueError."""
        space = GainSearchSpace(kp_pos_range=([-0.1, 0.005, 2.0], [0.05, 0.05, 6.0]))
        with pytest.raises(ValueError, match="negative minimum"):
            space.validate()

    def test_wrong_dimension_raises(self):
        """Test that wrong dimension raises ValueError."""
        space = GainSearchSpace(
            kp_pos_range=([0.005, 0.005], [0.05, 0.05])  # Only 2 values
        )
        with pytest.raises(ValueError, match="exactly 3 values"):
            space.validate()

    def test_equal_min_max_valid(self):
        """Test that equal min/max (fixed parameter) is valid."""
        space = GainSearchSpace(
            kp_pos_range=([0.01, 0.01, 4.0], [0.01, 0.01, 4.0])  # Fixed values
        )
        space.validate()  # Should not raise

    def test_scalar_range_validation(self):
        """Test scalar range validation for LQR parameters."""
        space = GainSearchSpace(
            r_thrust_range=(0.5, 2.0),
            r_rate_range=(0.5, 2.0),
        )
        space.validate()  # Should not raise
        assert "r_thrust" in space.get_active_parameters()
        assert "r_rate" in space.get_active_parameters()

    def test_inverted_scalar_range_raises(self):
        """Test that inverted scalar ranges raise ValueError."""
        space = GainSearchSpace(r_thrust_range=(2.0, 0.5))
        with pytest.raises(ValueError, match="inverted range"):
            space.validate()

    def test_negative_scalar_range_raises(self):
        """Test that negative scalar ranges raise ValueError."""
        space = GainSearchSpace(r_thrust_range=(-1.0, 2.0))
        with pytest.raises(ValueError, match="negative minimum"):
            space.validate()

    def test_from_dict(self):
        """Test creating search space from dictionary."""
        config = {
            "kp_pos_range": ([0.005, 0.005, 2.0], [0.05, 0.05, 6.0]),
            "kd_pos_range": ([0.02, 0.02, 1.0], [0.15, 0.15, 3.0]),
        }
        space = GainSearchSpace.from_dict(config)
        assert space.kp_pos_range == ([0.005, 0.005, 2.0], [0.05, 0.05, 6.0])
        assert space.kd_pos_range == ([0.02, 0.02, 1.0], [0.15, 0.15, 3.0])

    def test_to_dict(self):
        """Test converting search space to dictionary."""
        space = GainSearchSpace(kp_pos_range=([0.005, 0.005, 2.0], [0.05, 0.05, 6.0]))
        d = space.to_dict()
        assert d["kp_pos_range"] == ([0.005, 0.005, 2.0], [0.05, 0.05, 6.0])
        assert d["ki_pos_range"] is None

    def test_get_active_parameters(self):
        """Test getting list of active parameters."""
        space = GainSearchSpace(
            kp_pos_range=([0.005, 0.005, 2.0], [0.05, 0.05, 6.0]),
            kd_pos_range=([0.02, 0.02, 1.0], [0.15, 0.15, 3.0]),
            ff_velocity_gain_range=([0.0, 0.0, 0.0], [0.2, 0.2, 0.1]),
        )
        params = space.get_active_parameters()
        assert "kp_pos" in params
        assert "kd_pos" in params
        assert "ff_velocity_gain" in params
        assert "ki_pos" not in params


class TestTuningConfig:
    """Tests for TuningConfig validation and configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TuningConfig()
        assert config.controller_type == "pid"
        assert config.strategy == "random"
        assert config.max_iterations == 50
        assert config.seed == 42

    def test_invalid_controller_type_raises(self):
        """Test that invalid controller type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid controller_type"):
            TuningConfig(controller_type="invalid")

    def test_invalid_strategy_raises(self):
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="Invalid strategy"):
            TuningConfig(strategy="invalid")

    def test_zero_iterations_raises(self):
        """Test that zero iterations raises ValueError."""
        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            TuningConfig(max_iterations=0)

    def test_zero_episodes_raises(self):
        """Test that zero episodes raises ValueError."""
        with pytest.raises(ValueError, match="evaluation_episodes must be >= 1"):
            TuningConfig(evaluation_episodes=0)

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "controller_type": "lqr",
            "strategy": "grid",
            "max_iterations": 100,
            "seed": 123,
            "search_space": {
                "q_pos_range": ([0.0001, 0.0001, 10.0], [0.001, 0.001, 25.0]),
            },
        }
        config = TuningConfig.from_dict(config_dict)
        assert config.controller_type == "lqr"
        assert config.strategy == "grid"
        assert config.max_iterations == 100
        assert config.seed == 123
        assert config.search_space.q_pos_range is not None

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = TuningConfig(
            controller_type="pid",
            strategy="random",
            max_iterations=25,
        )
        d = config.to_dict()
        assert d["controller_type"] == "pid"
        assert d["strategy"] == "random"
        assert d["max_iterations"] == 25


class TestTuningResult:
    """Tests for TuningResult serialization."""

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = TuningResult(
            best_config={"kp_pos": [0.01, 0.01, 4.0]},
            best_score=0.85,
            best_metrics={"mean_on_target_ratio": 0.85, "mean_tracking_error": 0.3},
            all_results=[
                {"config": {"kp_pos": [0.01, 0.01, 4.0]}, "score": 0.85},
            ],
            iterations_completed=10,
            interrupted=False,
            timestamp="2024-01-01T12:00:00Z",
            config={"controller_type": "pid"},
        )
        d = result.to_dict()
        assert d["best_score"] == 0.85
        assert d["iterations_completed"] == 10
        assert d["interrupted"] is False

    def test_from_dict(self):
        """Test creating result from dictionary."""
        data = {
            "best_config": {"kp_pos": [0.01, 0.01, 4.0]},
            "best_score": 0.85,
            "best_metrics": {"mean_on_target_ratio": 0.85},
            "all_results": [],
            "iterations_completed": 10,
            "interrupted": False,
            "timestamp": "2024-01-01T12:00:00Z",
            "config": {},
        }
        result = TuningResult.from_dict(data)
        assert result.best_score == 0.85
        assert result.best_config["kp_pos"] == [0.01, 0.01, 4.0]

    def test_save_and_load(self, tmp_path):
        """Test saving and loading results."""
        result = TuningResult(
            best_config={"kp_pos": [0.01, 0.01, 4.0]},
            best_score=0.85,
            best_metrics={"mean_on_target_ratio": 0.85},
            all_results=[],
            iterations_completed=10,
            interrupted=False,
            timestamp="2024-01-01T12:00:00Z",
            config={},
        )

        path = tmp_path / "results.json"
        result.save(path)
        assert path.exists()

        loaded = TuningResult.load(path)
        assert loaded.best_score == 0.85
        assert loaded.best_config["kp_pos"] == [0.01, 0.01, 4.0]

    def test_save_path_traversal_rejected(self, tmp_path):
        """Test that path traversal sequences are rejected in save."""
        result = TuningResult(
            best_config={"kp_pos": [0.01, 0.01, 4.0]},
            best_score=0.85,
            best_metrics={"mean_on_target_ratio": 0.85},
            all_results=[],
            iterations_completed=10,
            interrupted=False,
            timestamp="2024-01-01T12:00:00Z",
            config={},
        )

        # Path with traversal should be rejected
        with pytest.raises(ValueError, match="path traversal"):
            result.save(tmp_path / ".." / "escape" / "results.json")

    def test_load_path_traversal_rejected(self):
        """Test that path traversal sequences are rejected in load."""
        with pytest.raises(ValueError, match="path traversal"):
            TuningResult.load("../../../etc/passwd")

    def test_load_nonexistent_file_raises(self, tmp_path):
        """Test that loading non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            TuningResult.load(tmp_path / "nonexistent.json")


class TestControllerTuner:
    """Tests for the ControllerTuner class."""

    @pytest.fixture
    def fast_config(self, tmp_path):
        """Create fast tuning config for tests."""
        return TuningConfig(
            controller_type="pid",
            search_space=GainSearchSpace(
                kp_pos_range=([0.01, 0.01, 4.0], [0.01, 0.01, 4.0]),  # Fixed
            ),
            strategy="random",
            max_iterations=2,
            evaluation_episodes=1,
            evaluation_horizon=100,  # Very short for fast testing
            episode_length=1.0,  # Short episodes
            target_motion_type="stationary",
            output_dir=str(tmp_path / "tuning"),
            seed=42,
        )

    def test_tuner_initialization(self, fast_config):
        """Test tuner creates correctly."""
        tuner = ControllerTuner(fast_config)
        assert tuner.config == fast_config
        assert tuner.best_score == float("-inf")
        assert len(tuner.results) == 0

    def test_tuner_random_search(self, fast_config):
        """Test random search runs without error."""
        tuner = ControllerTuner(fast_config)
        result = tuner.tune()

        assert result.iterations_completed == 2
        assert result.best_config is not None
        assert result.best_score > float("-inf")
        assert not result.interrupted

    def test_tuner_grid_search(self, fast_config, tmp_path):
        """Test grid search runs without error."""
        fast_config.strategy = "grid"
        fast_config.grid_points_per_dim = 2
        fast_config.search_space = GainSearchSpace(
            kp_pos_range=([0.01, 0.01, 3.0], [0.02, 0.02, 5.0])
        )

        tuner = ControllerTuner(fast_config)
        result = tuner.tune()

        assert result.iterations_completed > 0
        assert result.best_config is not None
        assert not result.interrupted

    def test_tuner_deterministic_seeding(self, fast_config):
        """Test that same seed produces same results."""
        # Run twice with same seed
        fast_config.seed = 123
        tuner1 = ControllerTuner(fast_config)
        result1 = tuner1.tune()

        fast_config.seed = 123
        tuner2 = ControllerTuner(fast_config)
        result2 = tuner2.tune()

        # Results should be identical
        assert result1.best_score == result2.best_score
        assert result1.best_config == result2.best_config

    def test_tuner_different_seeds_differ(self, fast_config):
        """Test that different seeds produce different results."""
        fast_config.search_space = GainSearchSpace(
            kp_pos_range=([0.005, 0.005, 2.0], [0.05, 0.05, 6.0])
        )

        fast_config.seed = 123
        tuner1 = ControllerTuner(fast_config)
        result1 = tuner1.tune()

        fast_config.seed = 456
        tuner2 = ControllerTuner(fast_config)
        result2 = tuner2.tune()

        # Results should differ - check that the configurations sampled are different
        # We compare the first result's config from each run
        config1 = result1.all_results[0]["config"]
        config2 = result2.all_results[0]["config"]
        configs_differ = config1 != config2

        # At minimum, the sampled configurations should be different
        assert configs_differ, (
            f"Seeds 123 and 456 produced identical first configurations: {config1}"
        )

    def test_tuner_saves_results(self, fast_config, tmp_path):
        """Test that tuner saves results to disk."""
        tuner = ControllerTuner(fast_config)
        result = tuner.tune()

        # Check files were created
        output_dir = tmp_path / "tuning"
        assert output_dir.exists()

        # Find results file
        results_files = list(output_dir.glob("*_results.json"))
        assert len(results_files) == 1

        # Find best config file
        config_files = list(output_dir.glob("*_best_config.json"))
        assert len(config_files) == 1

        # Verify content is valid JSON
        with open(results_files[0]) as f:
            data = json.load(f)
            assert data["best_score"] == result.best_score

    def test_tuner_lqr_controller(self, tmp_path):
        """Test tuning LQR controller."""
        config = TuningConfig(
            controller_type="lqr",
            search_space=GainSearchSpace(
                q_pos_range=([0.0001, 0.0001, 16.0], [0.0001, 0.0001, 16.0])
            ),
            strategy="random",
            max_iterations=2,
            evaluation_episodes=1,
            evaluation_horizon=100,
            episode_length=1.0,
            output_dir=str(tmp_path / "tuning"),
            seed=42,
        )

        tuner = ControllerTuner(config)
        result = tuner.tune()

        assert result.iterations_completed == 2
        assert result.best_config is not None

    def test_tuner_feedforward_gains(self, tmp_path):
        """Test tuning with feedforward gains."""
        config = TuningConfig(
            controller_type="pid",
            search_space=GainSearchSpace(
                kp_pos_range=([0.01, 0.01, 4.0], [0.01, 0.01, 4.0]),
                ff_velocity_gain_range=([0.0, 0.0, 0.0], [0.1, 0.1, 0.1]),
            ),
            strategy="random",
            max_iterations=2,
            evaluation_episodes=1,
            evaluation_horizon=100,
            episode_length=1.0,
            feedforward_enabled=True,
            output_dir=str(tmp_path / "tuning"),
            seed=42,
        )

        tuner = ControllerTuner(config)
        result = tuner.tune()

        assert result.iterations_completed == 2
        # Should have feedforward_enabled in best config
        assert result.best_config.get("feedforward_enabled") is True

    def test_tuner_resume_from_previous(self, fast_config, tmp_path):
        """Test resuming tuning from previous results."""
        # First run
        tuner1 = ControllerTuner(fast_config)
        tuner1.tune()

        # Get the results file path
        results_files = list((tmp_path / "tuning").glob("*_results.json"))
        assert len(results_files) == 1

        # Second run resuming from first
        fast_config.resume_from = str(results_files[0])
        fast_config.max_iterations = 4  # Total iterations including resumed

        tuner2 = ControllerTuner(fast_config)
        result2 = tuner2.tune()

        # Should have more results
        assert result2.iterations_completed == 4
        assert len(result2.all_results) == 4

    def test_generate_grid_configs(self, tmp_path):
        """Test grid configuration generation."""
        config = TuningConfig(
            controller_type="pid",
            search_space=GainSearchSpace(
                kp_pos_range=([0.01, 0.01, 3.0], [0.02, 0.02, 5.0])
            ),
            strategy="grid",
            grid_points_per_dim=2,
            output_dir=str(tmp_path),
        )

        tuner = ControllerTuner(config)
        configs = tuner._generate_grid_configs()

        # 2 points per dim, 3 dims = 2^3 = 8 configs
        assert len(configs) == 8

        # Each config should have kp_pos with 3 values
        for cfg in configs:
            assert "kp_pos" in cfg
            assert len(cfg["kp_pos"]) == 3

    def test_generate_random_config(self, tmp_path):
        """Test random configuration generation."""
        config = TuningConfig(
            controller_type="pid",
            search_space=GainSearchSpace(
                kp_pos_range=([0.005, 0.005, 2.0], [0.05, 0.05, 6.0]),
                kd_pos_range=([0.02, 0.02, 1.0], [0.15, 0.15, 3.0]),
            ),
            strategy="random",
            seed=42,
            output_dir=str(tmp_path),
        )

        tuner = ControllerTuner(config)
        random_config = tuner._generate_random_config()

        # Should have both parameters
        assert "kp_pos" in random_config
        assert "kd_pos" in random_config

        # Validate kp_pos values are within ranges
        kp = random_config["kp_pos"]
        assert 0.005 <= kp[0] <= 0.05, f"kp_pos[x]={kp[0]} out of range [0.005, 0.05]"
        assert 0.005 <= kp[1] <= 0.05, f"kp_pos[y]={kp[1]} out of range [0.005, 0.05]"
        assert 2.0 <= kp[2] <= 6.0, f"kp_pos[z]={kp[2]} out of range [2.0, 6.0]"

        # Validate kd_pos values are within ranges
        kd = random_config["kd_pos"]
        assert 0.02 <= kd[0] <= 0.15, f"kd_pos[x]={kd[0]} out of range [0.02, 0.15]"
        assert 0.02 <= kd[1] <= 0.15, f"kd_pos[y]={kd[1]} out of range [0.02, 0.15]"
        assert 1.0 <= kd[2] <= 3.0, f"kd_pos[z]={kd[2]} out of range [1.0, 3.0]"


class TestTuningIntegration:
    """Integration tests for the tuning pipeline."""

    def test_end_to_end_pid_tuning(self, tmp_path):
        """Test complete PID tuning workflow."""
        config = TuningConfig(
            controller_type="pid",
            search_space=GainSearchSpace(
                kp_pos_range=([0.008, 0.008, 3.5], [0.015, 0.015, 4.5]),
                kd_pos_range=([0.05, 0.05, 1.8], [0.08, 0.08, 2.2]),
            ),
            strategy="random",
            max_iterations=3,
            evaluation_episodes=2,
            evaluation_horizon=200,
            episode_length=2.0,
            target_motion_type="stationary",
            output_dir=str(tmp_path / "tuning"),
            seed=42,
        )

        tuner = ControllerTuner(config)
        result = tuner.tune()

        # Verify result structure
        assert result.iterations_completed == 3
        assert result.best_config is not None
        assert "kp_pos" in result.best_config
        assert "kd_pos" in result.best_config
        assert 0 <= result.best_metrics["mean_on_target_ratio"] <= 1
        assert result.best_metrics["mean_tracking_error"] >= 0

        # Verify files saved
        output_dir = tmp_path / "tuning"
        assert len(list(output_dir.glob("*_results.json"))) == 1
        assert len(list(output_dir.glob("*_best_config.json"))) == 1

    def test_end_to_end_lqr_tuning(self, tmp_path):
        """Test complete LQR tuning workflow."""
        config = TuningConfig(
            controller_type="lqr",
            search_space=GainSearchSpace(
                q_pos_range=([0.00008, 0.00008, 14.0], [0.00015, 0.00015, 18.0]),
                q_vel_range=([0.003, 0.003, 3.5], [0.004, 0.004, 4.5]),
            ),
            strategy="random",
            max_iterations=3,
            evaluation_episodes=2,
            evaluation_horizon=200,
            episode_length=2.0,
            target_motion_type="stationary",
            output_dir=str(tmp_path / "tuning"),
            seed=42,
        )

        tuner = ControllerTuner(config)
        result = tuner.tune()

        assert result.iterations_completed == 3
        assert "q_pos" in result.best_config
        assert "q_vel" in result.best_config

    def test_tuning_with_circular_motion(self, tmp_path):
        """Test tuning with circular target motion."""
        config = TuningConfig(
            controller_type="pid",
            search_space=GainSearchSpace(
                kp_pos_range=([0.01, 0.01, 4.0], [0.01, 0.01, 4.0]),
            ),
            strategy="random",
            max_iterations=2,
            evaluation_episodes=1,
            evaluation_horizon=100,
            episode_length=1.0,
            target_motion_type="circular",
            output_dir=str(tmp_path / "tuning"),
            seed=42,
        )

        tuner = ControllerTuner(config)
        result = tuner.tune()

        assert result.iterations_completed == 2


class TestRiccatiTuning:
    """Tests for Riccati-LQR controller tuning."""

    def test_riccati_controller_type_valid(self):
        """Test that riccati_lqr controller type is accepted."""
        config = TuningConfig(controller_type="riccati_lqr")
        assert config.controller_type == "riccati_lqr"

    def test_r_controls_range_validation(self):
        """Test r_controls_range validation for Riccati-LQR."""
        space = GainSearchSpace(
            r_controls_range=([0.5, 0.5, 0.5, 0.5], [2.0, 2.0, 2.0, 2.0])
        )
        space.validate()  # Should not raise
        assert "r_controls" in space.get_active_parameters()

    def test_r_controls_range_wrong_dimension_raises(self):
        """Test that r_controls_range with wrong dimensions raises ValueError."""
        space = GainSearchSpace(
            r_controls_range=([0.5, 0.5, 0.5], [2.0, 2.0, 2.0])  # Only 3 values
        )
        with pytest.raises(ValueError, match="exactly 4 values"):
            space.validate()

    def test_r_controls_range_inverted_raises(self):
        """Test that inverted r_controls_range raises ValueError."""
        space = GainSearchSpace(
            r_controls_range=([2.0, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])  # min > max
        )
        with pytest.raises(ValueError, match="inverted range"):
            space.validate()

    def test_r_controls_range_negative_raises(self):
        """Test that negative r_controls_range raises ValueError."""
        space = GainSearchSpace(
            r_controls_range=([-0.5, 0.5, 0.5, 0.5], [2.0, 2.0, 2.0, 2.0])
        )
        with pytest.raises(ValueError, match="negative minimum"):
            space.validate()

    def test_r_controls_range_from_dict(self):
        """Test creating search space with r_controls_range from dictionary."""
        config = {
            "q_pos_range": ([0.0001, 0.0001, 16.0], [0.001, 0.001, 25.0]),
            "r_controls_range": ([0.5, 0.5, 0.5, 0.5], [2.0, 2.0, 2.0, 2.0]),
        }
        space = GainSearchSpace.from_dict(config)
        assert space.q_pos_range == ([0.0001, 0.0001, 16.0], [0.001, 0.001, 25.0])
        assert space.r_controls_range == ([0.5, 0.5, 0.5, 0.5], [2.0, 2.0, 2.0, 2.0])

    def test_r_controls_range_to_dict(self):
        """Test converting search space with r_controls_range to dictionary."""
        space = GainSearchSpace(
            r_controls_range=([0.5, 0.5, 0.5, 0.5], [2.0, 2.0, 2.0, 2.0])
        )
        d = space.to_dict()
        assert d["r_controls_range"] == ([0.5, 0.5, 0.5, 0.5], [2.0, 2.0, 2.0, 2.0])

    def test_tuner_riccati_controller(self, tmp_path):
        """Test tuning Riccati-LQR controller."""
        config = TuningConfig(
            controller_type="riccati_lqr",
            search_space=GainSearchSpace(
                q_pos_range=([0.0001, 0.0001, 16.0], [0.0001, 0.0001, 16.0]),
                r_controls_range=([1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]),
            ),
            strategy="random",
            max_iterations=2,
            evaluation_episodes=1,
            evaluation_horizon=100,
            episode_length=1.0,
            output_dir=str(tmp_path / "tuning"),
            seed=42,
        )

        tuner = ControllerTuner(config)
        result = tuner.tune()

        assert result.iterations_completed == 2
        assert result.best_config is not None

    def test_generate_random_config_riccati(self, tmp_path):
        """Test random configuration generation for Riccati-LQR."""
        config = TuningConfig(
            controller_type="riccati_lqr",
            search_space=GainSearchSpace(
                q_pos_range=([0.00005, 0.00005, 10.0], [0.0005, 0.0005, 25.0]),
                q_vel_range=([0.001, 0.001, 2.0], [0.01, 0.01, 8.0]),
                r_controls_range=([0.5, 0.5, 0.5, 0.5], [2.0, 2.0, 2.0, 2.0]),
            ),
            strategy="random",
            seed=42,
            output_dir=str(tmp_path),
        )

        tuner = ControllerTuner(config)
        random_config = tuner._generate_random_config()

        # Should have Q and R parameters
        assert "q_pos" in random_config
        assert "q_vel" in random_config
        assert "r_controls" in random_config

        # Validate q_pos values are within ranges
        q_pos = random_config["q_pos"]
        assert 0.00005 <= q_pos[0] <= 0.0005
        assert 0.00005 <= q_pos[1] <= 0.0005
        assert 10.0 <= q_pos[2] <= 25.0

        # Validate r_controls values are within ranges
        r_controls = random_config["r_controls"]
        assert len(r_controls) == 4
        for i in range(4):
            assert 0.5 <= r_controls[i] <= 2.0

    def test_generate_grid_configs_riccati(self, tmp_path):
        """Test grid configuration generation for Riccati-LQR."""
        config = TuningConfig(
            controller_type="riccati_lqr",
            search_space=GainSearchSpace(
                r_controls_range=([0.5, 0.5, 0.5, 0.5], [1.5, 1.5, 1.5, 1.5]),
            ),
            strategy="grid",
            grid_points_per_dim=2,
            output_dir=str(tmp_path),
        )

        tuner = ControllerTuner(config)
        configs = tuner._generate_grid_configs()

        # 2 points per dim, 4 dims = 2^4 = 16 configs
        assert len(configs) == 16

        # Each config should have r_controls with 4 values
        for cfg in configs:
            assert "r_controls" in cfg
            assert len(cfg["r_controls"]) == 4

    def test_end_to_end_riccati_tuning(self, tmp_path):
        """Test complete Riccati-LQR tuning workflow."""
        config = TuningConfig(
            controller_type="riccati_lqr",
            search_space=GainSearchSpace(
                q_pos_range=([0.00008, 0.00008, 14.0], [0.00012, 0.00012, 18.0]),
                q_vel_range=([0.003, 0.003, 3.5], [0.004, 0.004, 4.5]),
                r_controls_range=([0.8, 0.8, 0.8, 0.8], [1.2, 1.2, 1.2, 1.2]),
            ),
            strategy="random",
            max_iterations=3,
            evaluation_episodes=2,
            evaluation_horizon=200,
            episode_length=2.0,
            target_motion_type="stationary",
            output_dir=str(tmp_path / "tuning"),
            seed=42,
        )

        tuner = ControllerTuner(config)
        result = tuner.tune()

        # Verify result structure
        assert result.iterations_completed == 3
        assert result.best_config is not None
        assert "q_pos" in result.best_config
        assert "q_vel" in result.best_config
        assert "r_controls" in result.best_config
        assert 0 <= result.best_metrics["mean_on_target_ratio"] <= 1
        assert result.best_metrics["mean_tracking_error"] >= 0

        # Verify files saved
        output_dir = tmp_path / "tuning"
        assert len(list(output_dir.glob("*_results.json"))) == 1
        assert len(list(output_dir.glob("*_best_config.json"))) == 1

        # Verify serialized config contains riccati_lqr
        results_files = list(output_dir.glob("*_results.json"))
        with open(results_files[0]) as f:
            data = json.load(f)
            assert data["config"]["controller_type"] == "riccati_lqr"


class TestCMAESTuning:
    """Tests for CMA-ES tuning strategy."""

    def test_cma_es_strategy_valid(self):
        """Test that cma_es strategy is accepted."""
        config = TuningConfig(strategy="cma_es")
        assert config.strategy == "cma_es"

    def test_cma_sigma0_validation(self):
        """Test that cma_sigma0 must be positive."""
        with pytest.raises(ValueError, match="cma_sigma0 must be > 0"):
            TuningConfig(strategy="cma_es", cma_sigma0=0)

        with pytest.raises(ValueError, match="cma_sigma0 must be > 0"):
            TuningConfig(strategy="cma_es", cma_sigma0=-0.5)

    def test_cma_popsize_validation(self):
        """Test that cma_popsize must be >= 2 if specified."""
        # None is valid (auto-calculated)
        config = TuningConfig(strategy="cma_es", cma_popsize=None)
        assert config.cma_popsize is None

        # Valid popsize
        config = TuningConfig(strategy="cma_es", cma_popsize=10)
        assert config.cma_popsize == 10

        # Invalid popsize
        with pytest.raises(ValueError, match="cma_popsize must be >= 2"):
            TuningConfig(strategy="cma_es", cma_popsize=1)

    def test_cma_es_config_from_dict(self):
        """Test creating config with CMA-ES options from dictionary."""
        config_dict = {
            "controller_type": "pid",
            "strategy": "cma_es",
            "cma_sigma0": 0.5,
            "cma_popsize": 12,
            "max_iterations": 100,
        }
        config = TuningConfig.from_dict(config_dict)
        assert config.strategy == "cma_es"
        assert config.cma_sigma0 == 0.5
        assert config.cma_popsize == 12

    def test_cma_es_config_to_dict(self):
        """Test converting config with CMA-ES options to dictionary."""
        config = TuningConfig(
            strategy="cma_es",
            cma_sigma0=0.4,
            cma_popsize=8,
        )
        d = config.to_dict()
        assert d["strategy"] == "cma_es"
        assert d["cma_sigma0"] == 0.4
        assert d["cma_popsize"] == 8

    def test_cma_es_tuner_initialization(self, tmp_path):
        """Test CMA-ES tuner initializes correctly."""
        config = TuningConfig(
            controller_type="pid",
            search_space=GainSearchSpace(
                kp_pos_range=([0.01, 0.01, 4.0], [0.01, 0.01, 4.0]),  # Fixed
            ),
            strategy="cma_es",
            max_iterations=5,
            evaluation_episodes=1,
            evaluation_horizon=100,
            episode_length=1.0,
            output_dir=str(tmp_path / "tuning"),
            seed=42,
        )

        tuner = ControllerTuner(config)
        assert tuner.config.strategy == "cma_es"
        assert tuner.cma_es is None  # Not initialized until tune() is called

    def test_cma_es_build_param_spec(self, tmp_path):
        """Test CMA-ES parameter specification building."""
        config = TuningConfig(
            controller_type="pid",
            search_space=GainSearchSpace(
                kp_pos_range=([0.005, 0.005, 2.0], [0.05, 0.05, 6.0]),
                kd_pos_range=([0.02, 0.02, 1.0], [0.15, 0.15, 3.0]),
            ),
            strategy="cma_es",
            output_dir=str(tmp_path),
        )

        tuner = ControllerTuner(config)
        spec = tuner._build_cma_param_spec()

        # Should have 2 parameters: kp_pos (3D) and kd_pos (3D)
        assert len(spec) == 2
        assert spec[0][0] == "kp_pos"
        assert spec[0][1] == 3
        assert spec[1][0] == "kd_pos"
        assert spec[1][1] == 3

    def test_cma_es_get_bounds(self, tmp_path):
        """Test CMA-ES bounds extraction."""
        config = TuningConfig(
            controller_type="pid",
            search_space=GainSearchSpace(
                kp_pos_range=([0.005, 0.005, 2.0], [0.05, 0.05, 6.0]),
            ),
            strategy="cma_es",
            output_dir=str(tmp_path),
        )

        tuner = ControllerTuner(config)
        tuner._cma_param_spec = tuner._build_cma_param_spec()
        lower, upper = tuner._get_cma_bounds()

        assert lower == [0.005, 0.005, 2.0]
        assert upper == [0.05, 0.05, 6.0]

    def test_cma_es_vector_to_config(self, tmp_path):
        """Test CMA-ES vector to config conversion."""
        config = TuningConfig(
            controller_type="pid",
            search_space=GainSearchSpace(
                kp_pos_range=([0.005, 0.005, 2.0], [0.05, 0.05, 6.0]),
                kd_pos_range=([0.02, 0.02, 1.0], [0.15, 0.15, 3.0]),
            ),
            strategy="cma_es",
            output_dir=str(tmp_path),
        )

        tuner = ControllerTuner(config)
        tuner._cma_param_spec = tuner._build_cma_param_spec()

        # Test vector: kp=[0.01, 0.01, 4.0], kd=[0.06, 0.06, 2.0]
        x = [0.01, 0.01, 4.0, 0.06, 0.06, 2.0]
        controller_config = tuner._vector_to_config(x)

        assert "kp_pos" in controller_config
        assert "kd_pos" in controller_config
        assert controller_config["kp_pos"] == [0.01, 0.01, 4.0]
        assert controller_config["kd_pos"] == [0.06, 0.06, 2.0]

    def test_cma_es_with_scalar_params(self, tmp_path):
        """Test CMA-ES with scalar parameters."""
        config = TuningConfig(
            controller_type="lqr",
            search_space=GainSearchSpace(
                r_thrust_range=(0.5, 2.0),
                r_rate_range=(0.5, 2.0),
            ),
            strategy="cma_es",
            output_dir=str(tmp_path),
        )

        tuner = ControllerTuner(config)
        tuner._cma_param_spec = tuner._build_cma_param_spec()
        lower, upper = tuner._get_cma_bounds()

        # Should have 2 scalar parameters
        assert len(lower) == 2
        assert len(upper) == 2
        assert lower == [0.5, 0.5]
        assert upper == [2.0, 2.0]

        # Test vector to config
        x = [1.0, 1.5]
        controller_config = tuner._vector_to_config(x)
        assert controller_config["r_thrust"] == 1.0
        assert controller_config["r_rate"] == 1.5

    def test_cma_es_tuner_runs(self, tmp_path):
        """Test that CMA-ES tuner runs with minimal iterations."""
        config = TuningConfig(
            controller_type="pid",
            search_space=GainSearchSpace(
                kp_pos_range=([0.008, 0.008, 3.5], [0.012, 0.012, 4.5]),  # Small range
            ),
            strategy="cma_es",
            max_iterations=3,  # Very small for fast testing
            evaluation_episodes=1,
            evaluation_horizon=100,
            episode_length=1.0,
            output_dir=str(tmp_path / "tuning"),
            seed=42,
            cma_sigma0=0.1,
        )

        tuner = ControllerTuner(config)
        result = tuner.tune()

        # Should have completed at least some iterations
        assert result.iterations_completed > 0
        assert result.best_config is not None
        assert not result.interrupted

    def test_cma_es_feedforward_flag(self, tmp_path):
        """Test that feedforward_enabled is set when FF gains are tuned."""
        config = TuningConfig(
            controller_type="pid",
            search_space=GainSearchSpace(
                kp_pos_range=([0.01, 0.01, 4.0], [0.01, 0.01, 4.0]),
                ff_velocity_gain_range=([0.0, 0.0, 0.0], [0.1, 0.1, 0.1]),
            ),
            strategy="cma_es",
            output_dir=str(tmp_path),
        )

        tuner = ControllerTuner(config)
        tuner._cma_param_spec = tuner._build_cma_param_spec()

        # Test vector with ff_velocity_gain
        x = [0.01, 0.01, 4.0, 0.05, 0.05, 0.05]
        controller_config = tuner._vector_to_config(x)

        assert "feedforward_enabled" in controller_config
        assert controller_config["feedforward_enabled"] is True

    def test_cma_es_saves_checkpoint(self, tmp_path):
        """Test that CMA-ES saves checkpoint on completion."""
        config = TuningConfig(
            controller_type="pid",
            search_space=GainSearchSpace(
                kp_pos_range=([0.008, 0.008, 3.5], [0.012, 0.012, 4.5]),
            ),
            strategy="cma_es",
            max_iterations=3,
            evaluation_episodes=1,
            evaluation_horizon=100,
            episode_length=1.0,
            output_dir=str(tmp_path / "tuning"),
            seed=42,
        )

        tuner = ControllerTuner(config)
        tuner.tune()

        # Check checkpoint file was created
        checkpoint_path = tmp_path / "tuning" / "cma_checkpoint.pkl"
        assert checkpoint_path.exists()

    def test_cma_es_deterministic_seeding(self, tmp_path):
        """Test that CMA-ES produces deterministic results with same seed."""
        def run_tuning(seed):
            config = TuningConfig(
                controller_type="pid",
                search_space=GainSearchSpace(
                    kp_pos_range=([0.005, 0.005, 2.0], [0.05, 0.05, 6.0]),
                ),
                strategy="cma_es",
                max_iterations=3,
                evaluation_episodes=1,
                evaluation_horizon=100,
                episode_length=1.0,
                output_dir=str(tmp_path / f"tuning_{seed}"),
                seed=seed,
            )
            tuner = ControllerTuner(config)
            return tuner.tune()

        result1 = run_tuning(42)
        result2 = run_tuning(42)

        # Same seed should produce same results
        assert result1.best_score == result2.best_score

    def test_cma_es_riccati_controller(self, tmp_path):
        """Test CMA-ES with Riccati-LQR controller."""
        config = TuningConfig(
            controller_type="riccati_lqr",
            search_space=GainSearchSpace(
                q_pos_range=([0.00008, 0.00008, 14.0], [0.00012, 0.00012, 18.0]),
                r_controls_range=([0.8, 0.8, 0.8, 0.8], [1.2, 1.2, 1.2, 1.2]),
            ),
            strategy="cma_es",
            max_iterations=3,
            evaluation_episodes=1,
            evaluation_horizon=100,
            episode_length=1.0,
            output_dir=str(tmp_path / "tuning"),
            seed=42,
        )

        tuner = ControllerTuner(config)
        result = tuner.tune()

        assert result.iterations_completed > 0
        assert result.best_config is not None
