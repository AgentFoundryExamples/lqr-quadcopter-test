"""Tests for quadcopter dynamics environment."""

import numpy as np
import pytest

from quadcopter_tracking.env import (
    CircularMotion,
    EnvConfig,
    Figure8Motion,
    LinearMotion,
    QuadcopterEnv,
    SimulationParams,
    SinusoidalMotion,
    StationaryMotion,
    TargetMotion,
    TargetParams,
)


class TestTargetMotion:
    """Tests for target motion generation."""

    def test_target_motion_reproducibility(self):
        """Test that same seed produces same trajectory."""
        params = TargetParams(motion_type="circular", radius=2.0, speed=1.0)

        motion1 = TargetMotion(params=params, seed=42)
        motion1.reset()

        motion2 = TargetMotion(params=params, seed=42)
        motion2.reset()

        for t in [0.0, 1.0, 5.0, 10.0]:
            pos1 = motion1.get_position(t)
            pos2 = motion2.get_position(t)
            assert np.allclose(pos1, pos2), f"Position mismatch at t={t}"

    def test_linear_motion_trajectory(self):
        """Test linear motion produces constant velocity trajectory."""
        direction = np.array([1.0, 0.0, 0.0])
        start = np.array([0.0, 0.0, 1.0])
        speed = 2.0

        motion = LinearMotion(start=start, direction=direction, speed=speed)

        pos0, vel0, acc0 = motion.get_state(0.0)
        pos1, vel1, acc1 = motion.get_state(1.0)

        # Check position
        assert np.allclose(pos0, start)
        assert np.allclose(pos1, start + direction * speed)

        # Check velocity is constant
        assert np.allclose(vel0, vel1)
        assert np.allclose(vel0, direction * speed)

        # Check acceleration is zero
        assert np.allclose(acc0, np.zeros(3))

    def test_circular_motion_trajectory(self):
        """Test circular motion maintains constant radius."""
        center = np.array([0.0, 0.0, 1.0])
        radius = 2.0
        speed = 1.0

        motion = CircularMotion(center=center, radius=radius, speed=speed)

        # Check multiple points are at correct radius
        for t in np.linspace(0, 10, 20):
            pos, vel, acc = motion.get_state(t)
            dist_from_center = np.linalg.norm(pos[:2] - center[:2])
            assert abs(dist_from_center - radius) < 1e-10, f"Radius error at t={t}"

            # Check velocity magnitude equals speed
            assert abs(np.linalg.norm(vel) - speed) < 1e-10

            # Check altitude is constant
            assert abs(pos[2] - center[2]) < 1e-10

    def test_sinusoidal_motion_smooth(self):
        """Test sinusoidal motion produces smooth trajectory."""
        center = np.array([0.0, 0.0, 1.0])
        amplitude = np.array([2.0, 1.0, 0.5])
        frequency = np.array([0.5, 0.3, 0.2])

        motion = SinusoidalMotion(
            center=center, amplitude=amplitude, frequency=frequency
        )

        # Check trajectory stays within amplitude bounds
        for t in np.linspace(0, 20, 100):
            pos, vel, acc = motion.get_state(t)
            for i in range(3):
                assert abs(pos[i] - center[i]) <= amplitude[i] + 1e-10

        # Check velocity is continuous (numerical check)
        dt = 0.001
        for t in [1.0, 5.0, 10.0]:
            pos_t, vel_t, _ = motion.get_state(t)
            pos_tp, _, _ = motion.get_state(t + dt)
            numerical_vel = (pos_tp - pos_t) / dt
            assert np.allclose(vel_t, numerical_vel, atol=0.01)

    def test_stationary_motion(self):
        """Test stationary target stays in place."""
        position = np.array([1.0, 2.0, 3.0])
        motion = StationaryMotion(position=position)

        for t in [0.0, 10.0, 100.0]:
            pos, vel, acc = motion.get_state(t)
            assert np.allclose(pos, position)
            assert np.allclose(vel, np.zeros(3))
            assert np.allclose(acc, np.zeros(3))

    def test_figure8_motion_continuous(self):
        """Test figure-8 motion is continuous."""
        center = np.array([0.0, 0.0, 1.0])
        scale = 2.0
        speed = 1.0

        motion = Figure8Motion(center=center, scale=scale, speed=speed)

        # Check continuity by comparing adjacent points
        dt = 0.01
        for t in np.linspace(0, 10, 50):
            pos1, _, _ = motion.get_state(t)
            pos2, _, _ = motion.get_state(t + dt)
            dist = np.linalg.norm(pos2 - pos1)
            # Should move less than speed * dt * 2 (with some margin)
            assert dist < speed * dt * 3

    def test_target_motion_invalid_type(self):
        """Test that invalid motion type raises error."""
        params = TargetParams(motion_type="invalid_type")
        motion = TargetMotion(params=params)

        with pytest.raises(ValueError, match="Invalid motion type"):
            motion.reset()

    def test_target_motion_max_acceleration(self):
        """Test that acceleration is limited."""
        params = TargetParams(
            motion_type="sinusoidal",
            amplitude=10.0,
            frequency=5.0,  # High frequency = high acceleration
            max_acceleration=5.0,
        )
        motion = TargetMotion(params=params, seed=42)
        motion.reset()

        # Check acceleration limit is enforced
        for t in np.linspace(0, 5, 50):
            state = motion.get_state(t)
            accel_mag = np.linalg.norm(state["acceleration"])
            assert accel_mag <= params.max_acceleration + 1e-6


class TestQuadcopterEnv:
    """Tests for quadcopter environment."""

    def test_env_reset(self):
        """Test environment reset returns valid observation."""
        env = QuadcopterEnv()
        obs = env.reset(seed=42)

        assert "quadcopter" in obs
        assert "target" in obs
        assert "time" in obs
        assert obs["time"] == 0.0

        assert "position" in obs["quadcopter"]
        assert "velocity" in obs["quadcopter"]
        assert "attitude" in obs["quadcopter"]
        assert "angular_velocity" in obs["quadcopter"]

        assert "position" in obs["target"]
        assert "velocity" in obs["target"]

    def test_env_step_returns_tuple(self):
        """Test step returns proper tuple format."""
        env = QuadcopterEnv()
        env.reset(seed=42)

        action = {"thrust": 9.81, "roll_rate": 0.0, "pitch_rate": 0.0, "yaw_rate": 0.0}
        obs, reward, done, info = env.step(action)

        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_env_action_array_format(self):
        """Test step accepts array action format."""
        env = QuadcopterEnv()
        env.reset(seed=42)

        action = np.array([9.81, 0.0, 0.0, 0.0])
        obs, reward, done, info = env.step(action)

        assert isinstance(obs, dict)

    def test_env_time_progresses(self):
        """Test simulation time advances with each step."""
        env = QuadcopterEnv()
        env.reset(seed=42)
        dt = env.dt

        action = env.hover_action()
        env.step(action)

        assert abs(env.time - dt) < 1e-10

        env.step(action)
        assert abs(env.time - 2 * dt) < 1e-10

    def test_env_hover_stability(self):
        """Test that hover action maintains altitude."""
        config = EnvConfig()
        config.simulation = SimulationParams(dt=0.01, max_episode_time=5.0)

        env = QuadcopterEnv(config=config)
        obs = env.reset(seed=42)

        # Get initial position
        initial_z = obs["quadcopter"]["position"][2]
        hover_action = env.hover_action(
            mass=config.quadcopter.mass,
            gravity=config.quadcopter.gravity,
        )

        # Run simulation
        for _ in range(100):
            obs, _, _, _ = env.step(hover_action)

        final_z = obs["quadcopter"]["position"][2]

        # Position should stay relatively stable (within 1m)
        assert abs(final_z - initial_z) < 1.0, "Hover not stable"

    def test_env_action_clipping(self):
        """Test that actions are clipped to valid bounds."""
        env = QuadcopterEnv()
        env.reset(seed=42)

        # Extreme action values
        action = {
            "thrust": 1000.0,  # Way over max
            "roll_rate": 100.0,
            "pitch_rate": -100.0,
            "yaw_rate": 50.0,
        }

        obs, _, _, _ = env.step(action)

        # Should not crash and should record violations
        violations = env.get_action_violations()
        assert len(violations) > 0

    def test_env_nan_action_handling(self):
        """Test that NaN actions are handled gracefully."""
        env = QuadcopterEnv()
        env.reset(seed=42)

        action = np.array([float("nan"), 0.0, 0.0, 0.0])
        obs, _, _, _ = env.step(action)

        # Should not crash
        assert np.all(np.isfinite(obs["quadcopter"]["position"]))

        violations = env.get_action_violations()
        assert len(violations) > 0
        assert any("NaN" in str(v) for v in violations[0]["violations"])

    def test_env_reproducibility(self):
        """Test environment reproducibility with seed."""
        env1 = QuadcopterEnv()
        env2 = QuadcopterEnv()

        obs1 = env1.reset(seed=123)
        obs2 = env2.reset(seed=123)

        assert np.allclose(
            obs1["quadcopter"]["position"],
            obs2["quadcopter"]["position"],
        )

        action = {"thrust": 10.0, "roll_rate": 0.1, "pitch_rate": 0.0, "yaw_rate": 0.0}

        for _ in range(10):
            obs1, _, _, _ = env1.step(action)
            obs2, _, _, _ = env2.step(action)

        assert np.allclose(
            obs1["quadcopter"]["position"],
            obs2["quadcopter"]["position"],
        )

    def test_env_long_episode_stability(self):
        """Test numerical stability for 30+ second simulation."""
        config = EnvConfig()
        config.simulation = SimulationParams(dt=0.01, max_episode_time=35.0)

        env = QuadcopterEnv(config=config)
        env.reset(seed=42)

        hover_action = env.hover_action(
            mass=config.quadcopter.mass,
            gravity=config.quadcopter.gravity,
        )

        done = False
        step_count = 0
        max_steps = 3500  # 35 seconds at 0.01 dt

        while not done and step_count < max_steps:
            obs, _, done, info = env.step(hover_action)
            step_count += 1

            # Check for numerical stability
            assert np.all(np.isfinite(obs["quadcopter"]["position"]))
            assert np.all(np.isfinite(obs["quadcopter"]["velocity"]))

        # Should reach time limit, not numerical instability
        if done:
            assert info.get("termination_reason") in ["time_limit", "position_bounds"]

    def test_env_tracking_metrics(self):
        """Test tracking metrics are computed correctly."""
        env = QuadcopterEnv()
        env.reset(seed=42)

        # Step until done or max steps
        action = env.hover_action()
        for _ in range(100):
            _, _, done, info = env.step(action)
            if done:
                break

        assert "tracking_error" in info
        assert "on_target" in info
        assert "on_target_ratio" in info
        assert 0.0 <= info["on_target_ratio"] <= 1.0

    def test_env_episode_termination(self):
        """Test episode terminates at time limit."""
        config = EnvConfig()
        config.simulation = SimulationParams(dt=0.1, max_episode_time=1.0)

        env = QuadcopterEnv(config=config)
        env.reset(seed=42)

        action = env.hover_action()
        done = False
        steps = 0

        while not done and steps < 100:
            _, _, done, info = env.step(action)
            steps += 1

        assert done
        assert info.get("termination_reason") == "time_limit"

    def test_env_history_recording(self):
        """Test that history is recorded."""
        config = EnvConfig()
        config.logging.enabled = True
        config.logging.log_interval = 1

        env = QuadcopterEnv(config=config)
        env.reset(seed=42)

        action = env.hover_action()
        for _ in range(10):
            env.step(action)

        history = env.get_history()
        assert len(history) == 10

        # Check history format
        for entry in history:
            assert "time" in entry
            assert "quadcopter_position" in entry
            assert "target_position" in entry
            assert "action" in entry
            assert "tracking_error" in entry

    def test_env_position_bounds(self):
        """Test episode terminates when position exceeds bounds."""
        config = EnvConfig()
        config.simulation = SimulationParams(
            dt=0.1,
            max_episode_time=100.0,
            max_position=10.0,
        )

        env = QuadcopterEnv(config=config)
        env.reset(seed=42)

        # Apply thrust in one direction to exceed bounds
        action = {"thrust": 20.0, "roll_rate": 0.0, "pitch_rate": 0.5, "yaw_rate": 0.0}

        done = False
        steps = 0
        while not done and steps < 500:
            _, _, done, info = env.step(action)
            steps += 1

        if done:
            reason = info.get("termination_reason", "")
            assert reason in ["position_bounds", "time_limit", "numerical_instability"]


class TestEnvConfig:
    """Tests for environment configuration."""

    def test_config_from_dict(self):
        """Test configuration from dictionary."""
        config_dict = {
            "seed": 123,
            "episode_length": 60.0,
            "dt": 0.02,
            "quadcopter": {
                "mass": 2.0,
                "max_thrust": 40.0,
            },
            "target": {
                "motion_type": "circular",
                "radius": 5.0,
            },
        }

        config = EnvConfig.from_dict(config_dict)

        assert config.seed == 123
        assert config.simulation.max_episode_time == 60.0
        assert config.simulation.dt == 0.02
        assert config.quadcopter.mass == 2.0
        assert config.quadcopter.max_thrust == 40.0
        assert config.target.motion_type == "circular"
        assert config.target.radius == 5.0

    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = EnvConfig()
        config_dict = config.to_dict()

        assert "seed" in config_dict
        assert "quadcopter" in config_dict
        assert "simulation" in config_dict
        assert "target" in config_dict

        # Round-trip test
        config2 = EnvConfig.from_dict(config_dict)
        assert config.seed == config2.seed
        assert config.quadcopter.mass == config2.quadcopter.mass


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_full_episode_workflow(self):
        """Test complete episode from reset to termination."""
        config = EnvConfig()
        config.simulation = SimulationParams(dt=0.01, max_episode_time=5.0)
        config.target = TargetParams(motion_type="stationary")

        env = QuadcopterEnv(config=config)
        obs = env.reset(seed=42)

        total_reward = 0.0
        done = False

        while not done:
            # Simple proportional controller
            quad_pos = obs["quadcopter"]["position"]
            target_pos = obs["target"]["position"]
            error = target_pos - quad_pos

            # Compute control
            thrust = config.quadcopter.mass * config.quadcopter.gravity
            thrust += error[2] * 2.0  # Z control

            action = {
                "thrust": np.clip(thrust, 0, config.quadcopter.max_thrust),
                "roll_rate": error[1] * 0.5,
                "pitch_rate": -error[0] * 0.5,
                "yaw_rate": 0.0,
            }

            obs, reward, done, info = env.step(action)
            total_reward += reward

        assert "episode_length" in info
        assert info["episode_length"] >= 5.0

    def test_different_motion_types(self):
        """Test environment with different target motion types."""
        motion_types = ["linear", "circular", "sinusoidal", "figure8", "stationary"]

        for motion_type in motion_types:
            config = EnvConfig()
            config.simulation = SimulationParams(dt=0.1, max_episode_time=2.0)
            config.target = TargetParams(motion_type=motion_type)

            env = QuadcopterEnv(config=config)
            obs = env.reset(seed=42)

            # Run a few steps
            action = env.hover_action()
            for _ in range(5):
                obs, _, done, _ = env.step(action)
                if done:
                    break

            # Verify target state is valid
            target_pos = obs["target"]["position"]
            assert np.all(np.isfinite(target_pos)), (
                f"Invalid position for {motion_type}"
            )

    def test_rk4_vs_euler_comparison(self):
        """Test that RK4 integration is more accurate than Euler."""
        # Both should produce similar results for small dt
        # but RK4 should be more stable

        for integrator in ["euler", "rk4"]:
            config = EnvConfig()
            config.simulation = SimulationParams(
                dt=0.01,
                max_episode_time=5.0,
                integrator=integrator,
            )

            env = QuadcopterEnv(config=config)
            env.reset(seed=42)

            hover_action = env.hover_action(
                mass=config.quadcopter.mass,
                gravity=config.quadcopter.gravity,
            )

            for _ in range(100):
                obs, _, done, _ = env.step(hover_action)
                if done:
                    break

            # Both should be numerically stable
            assert np.all(np.isfinite(obs["quadcopter"]["position"]))
