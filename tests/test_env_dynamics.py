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


class TestPIDController:
    """Tests for PID controller implementation."""

    def test_pid_controller_initialization(self):
        """Test PID controller initializes with default gains.

        Default gains are tuned for stable tracking with small XY proportional
        gains (kp_pos[0:2] = 0.01) to avoid actuator saturation due to the
        meter→rad/s mapping. Integral limit defaults to 0.0 for XY axes.
        """
        from quadcopter_tracking.controllers import PIDController

        pid = PIDController()
        assert pid.name == "pid"
        assert pid.kp_pos is not None
        assert pid.ki_pos is not None
        assert pid.kd_pos is not None
        # New validated defaults: integral_limit = 0.0 (disabled by default)
        assert pid.integral_limit == 0.0
        assert len(pid.kp_pos) == 3
        # Verify new XY gains are small (meter→rad/s scaling)
        assert np.allclose(pid.kp_pos[:2], [0.01, 0.01])
        # Verify Z gain remains higher for altitude tracking
        assert pid.kp_pos[2] == 4.0

    def test_pid_validated_baseline_gains(self):
        """Test PID controller baseline gains match documented values.

        Validates that the default gains match the experimentally validated
        baseline for stationary, linear, and circular scenarios:
        - kp_pos: [0.01, 0.01, 4.0]
        - ki_pos: [0.0, 0.0, 0.0]
        - kd_pos: [0.06, 0.06, 2.0]
        - integral_limit: 0.0
        """
        from quadcopter_tracking.controllers import PIDController

        pid = PIDController()

        # Validate documented baseline gains
        assert np.allclose(pid.kp_pos, [0.01, 0.01, 4.0]), (
            f"kp_pos should be [0.01, 0.01, 4.0], got {pid.kp_pos.tolist()}"
        )
        assert np.allclose(pid.ki_pos, [0.0, 0.0, 0.0]), (
            f"ki_pos should be [0.0, 0.0, 0.0], got {pid.ki_pos.tolist()}"
        )
        assert np.allclose(pid.kd_pos, [0.06, 0.06, 2.0]), (
            f"kd_pos should be [0.06, 0.06, 2.0], got {pid.kd_pos.tolist()}"
        )
        assert pid.integral_limit == 0.0, (
            f"integral_limit should be 0.0, got {pid.integral_limit}"
        )

    def test_pid_controller_custom_gains(self):
        """Test PID controller with custom gains."""
        from quadcopter_tracking.controllers import PIDController

        config = {
            "kp_pos": [1.0, 2.0, 3.0],
            "ki_pos": [0.1, 0.2, 0.3],
            "kd_pos": [0.5, 0.6, 0.7],
            "integral_limit": 10.0,
        }
        pid = PIDController(config=config)
        assert np.allclose(pid.kp_pos, [1.0, 2.0, 3.0])
        assert np.allclose(pid.ki_pos, [0.1, 0.2, 0.3])
        assert np.allclose(pid.kd_pos, [0.5, 0.6, 0.7])
        assert pid.integral_limit == 10.0

    def test_pid_controller_scalar_gains(self):
        """Test PID controller with scalar gains (broadcast to all axes)."""
        from quadcopter_tracking.controllers import PIDController

        config = {"kp": 2.0, "ki": 0.1, "kd": 1.0}
        pid = PIDController(config=config)
        assert np.allclose(pid.kp_pos, [2.0, 2.0, 2.0])
        assert np.allclose(pid.ki_pos, [0.1, 0.1, 0.1])
        assert np.allclose(pid.kd_pos, [1.0, 1.0, 1.0])

    def test_pid_compute_action_format(self):
        """Test PID compute_action returns correct format."""
        from quadcopter_tracking.controllers import PIDController

        pid = PIDController()
        env = QuadcopterEnv()
        obs = env.reset(seed=42)

        action = pid.compute_action(obs)

        assert "thrust" in action
        assert "roll_rate" in action
        assert "pitch_rate" in action
        assert "yaw_rate" in action
        assert isinstance(action["thrust"], float)
        assert isinstance(action["roll_rate"], float)

    def test_pid_output_bounds(self):
        """Test PID controller respects output bounds."""
        from quadcopter_tracking.controllers import PIDController

        config = {
            "max_thrust": 20.0,
            "min_thrust": 0.0,
            "max_rate": 3.0,
        }
        pid = PIDController(config=config)
        env = QuadcopterEnv()
        obs = env.reset(seed=42)

        # Run multiple steps to build up integral error
        for _ in range(100):
            action = pid.compute_action(obs)
            obs, _, done, _ = env.step(action)
            if done:
                break

            assert action["thrust"] >= 0.0
            assert action["thrust"] <= 20.0
            assert abs(action["roll_rate"]) <= 3.0
            assert abs(action["pitch_rate"]) <= 3.0
            assert abs(action["yaw_rate"]) <= 3.0

    def test_pid_integral_windup_prevention(self):
        """Test PID integral error is clamped to prevent windup."""
        from quadcopter_tracking.controllers import PIDController

        config = {"integral_limit": 2.0}
        pid = PIDController(config=config)
        env = QuadcopterEnv()
        obs = env.reset(seed=42)

        # Run many steps to accumulate integral error
        for _ in range(200):
            pid.compute_action(obs)

        # Integral error should be clamped
        assert np.all(np.abs(pid.integral_error) <= 2.0)

    def test_pid_reset_clears_integral(self):
        """Test PID reset clears integral error and time state."""
        from quadcopter_tracking.controllers import PIDController

        pid = PIDController()
        env = QuadcopterEnv()
        obs = env.reset(seed=42)

        # Build up integral error and time state
        for _ in range(50):
            pid.compute_action(obs)
            obs, _, _, _ = env.step(env.hover_action())

        # Verify state was accumulated
        assert pid._last_time is not None

        # Reset and check
        pid.reset()
        assert np.allclose(pid.integral_error, [0.0, 0.0, 0.0])
        assert pid._last_time is None

    def test_pid_responds_to_position_error(self):
        """Test PID controller responds correctly to position error."""
        from quadcopter_tracking.controllers import PIDController

        pid = PIDController()

        # Create observation with known error
        obs = {
            "quadcopter": {
                "position": np.array([0.0, 0.0, 1.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
                "attitude": np.array([0.0, 0.0, 0.0]),
                "angular_velocity": np.array([0.0, 0.0, 0.0]),
            },
            "target": {
                "position": np.array([1.0, 0.0, 1.0]),  # Target is +1m in X
                "velocity": np.array([0.0, 0.0, 0.0]),
            },
        }

        action = pid.compute_action(obs)

        # Positive X error should result in positive pitch rate
        # Environment dynamics: +pitch_rate produces +X velocity
        assert action["pitch_rate"] > 0

    def test_pid_moves_quadcopter_toward_target(self):
        """Test PID controller reduces tracking error over time."""
        from quadcopter_tracking.controllers import PIDController

        config = EnvConfig()
        config.simulation = SimulationParams(dt=0.01, max_episode_time=10.0)
        config.target = TargetParams(motion_type="stationary")

        env = QuadcopterEnv(config=config)
        obs = env.reset(seed=42)
        pid = PIDController()

        # Run controller for several steps and track minimum error achieved
        min_error = float("inf")
        for _ in range(500):
            action = pid.compute_action(obs)
            obs, _, done, info = env.step(action)
            min_error = min(min_error, info["tracking_error"])
            if done:
                break

        # Controller should achieve reasonable tracking at some point
        assert min_error < 0.5, f"Min tracking error too high: {min_error}"

    def test_pid_observation_validation(self):
        """Test PID controller raises error on invalid observation."""
        from quadcopter_tracking.controllers import PIDController

        pid = PIDController()

        # Missing quadcopter key
        with pytest.raises(KeyError, match="quadcopter"):
            pid.compute_action({"target": {}})

        # Missing target key
        with pytest.raises(KeyError, match="target"):
            pid.compute_action({"quadcopter": {}})

    def test_pid_hover_thrust_at_zero_error(self):
        """Test PID returns hover thrust (~9.81N) when at target with zero velocity.

        This validates that PID includes hover feedforward (hover_thrust =
        mass * gravity) and produces the correct baseline thrust when there
        is no tracking error.
        """
        from quadcopter_tracking.controllers import PIDController

        pid = PIDController()

        # Create observation with zero position and velocity error (at hover)
        obs = {
            "quadcopter": {
                "position": np.array([0.0, 0.0, 1.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
                "attitude": np.array([0.0, 0.0, 0.0]),
                "angular_velocity": np.array([0.0, 0.0, 0.0]),
            },
            "target": {
                "position": np.array([0.0, 0.0, 1.0]),  # Same as quadcopter
                "velocity": np.array([0.0, 0.0, 0.0]),
            },
            "time": 0.0,
        }

        action = pid.compute_action(obs)

        # With default parameters, thrust at zero error should equal hover_thrust
        assert abs(action["thrust"] - pid.hover_thrust) < 0.01, (
            f"PID thrust at zero error should be {pid.hover_thrust}N, "
            f"got {action['thrust']}"
        )
        # All rates should be zero with no error
        assert abs(action["roll_rate"]) < 0.01
        assert abs(action["pitch_rate"]) < 0.01
        assert abs(action["yaw_rate"]) < 0.01

    def test_pid_hover_thrust_custom_mass_gravity(self):
        """Test PID hover thrust adjusts with custom mass/gravity configuration.

        Verifies that hover_thrust = mass * gravity scales correctly for
        non-default physics parameters.
        """
        from quadcopter_tracking.controllers import PIDController

        # Custom mass and gravity
        config = {"mass": 1.5, "gravity": 10.0}
        pid = PIDController(config=config)

        # Expected hover thrust: 1.5 kg * 10.0 m/s² = 15.0 N
        expected_hover_thrust = 15.0

        # Verify hover_thrust attribute
        assert abs(pid.hover_thrust - expected_hover_thrust) < 0.01

        # Create zero-error observation
        obs = {
            "quadcopter": {
                "position": np.array([0.0, 0.0, 1.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
                "attitude": np.array([0.0, 0.0, 0.0]),
                "angular_velocity": np.array([0.0, 0.0, 0.0]),
            },
            "target": {
                "position": np.array([0.0, 0.0, 1.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
            },
            "time": 0.0,
        }

        action = pid.compute_action(obs)

        # Thrust should equal custom hover_thrust at zero error
        assert abs(action["thrust"] - expected_hover_thrust) < 0.01, (
            f"PID thrust should be {expected_hover_thrust}N, got {action['thrust']}"
        )

    def test_pid_integral_does_not_perturb_hover_baseline(self):
        """Test integral term doesn't perturb hover thrust at sustained zero error.

        Ensures integral windup does not accumulate when error is already zero,
        preserving the baseline hover thrust.
        """
        from quadcopter_tracking.controllers import PIDController

        pid = PIDController()

        # Create zero-error observation
        obs = {
            "quadcopter": {
                "position": np.array([0.0, 0.0, 1.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
                "attitude": np.array([0.0, 0.0, 0.0]),
                "angular_velocity": np.array([0.0, 0.0, 0.0]),
            },
            "target": {
                "position": np.array([0.0, 0.0, 1.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
            },
            "time": 0.0,
        }

        # First call establishes baseline
        pid.compute_action(obs)

        # Simulate many timesteps at zero error
        for i in range(100):
            obs["time"] = (i + 1) * 0.01  # Advance time
            action = pid.compute_action(obs)

        # Thrust should remain at hover baseline (integral of zero error = 0)
        assert abs(action["thrust"] - pid.hover_thrust) < 0.01, (
            f"PID thrust should stay at {pid.hover_thrust}N after zero error, "
            f"got {action['thrust']}"
        )
        # Integral error should be zero
        assert np.allclose(pid.integral_error, [0.0, 0.0, 0.0], atol=0.01)


class TestLQRController:
    """Tests for LQR controller implementation."""

    def test_lqr_controller_initialization(self):
        """Test LQR controller initializes correctly."""
        from quadcopter_tracking.controllers import LQRController

        lqr = LQRController()
        assert lqr.name == "lqr"
        assert lqr.K is not None
        assert lqr.K.shape == (4, 6)

    def test_lqr_validated_baseline_gains(self):
        """Test LQR controller baseline gains produce consistent feedback.

        Validates that the default cost weights produce feedback gains
        consistent with the validated PID baseline:
        - q_pos: [0.0001, 0.0001, 16.0] (low XY for meter→rad/s mapping)
        - q_vel: [0.0036, 0.0036, 4.0]
        - r_thrust: 1.0
        - r_rate: 1.0

        The resulting K matrix should have small XY position gains (~0.01)
        matching the PID kp_pos baseline.
        """
        from quadcopter_tracking.controllers import LQRController

        lqr = LQRController()

        # The LQR gain formula: K_pos = sqrt(q_pos / r)
        # For XY: sqrt(0.0001 / 1.0) = 0.01
        # For Z: sqrt(16.0 / 1.0) = 4.0

        # Check XY position gains are small (columns 0 and 1 for X and Y)
        # X position -> pitch rate (row 2, col 0)
        x_pos_gain = abs(lqr.K[2, 0])
        assert x_pos_gain == pytest.approx(0.01), (
            f"X position gain should be ~0.01, got {x_pos_gain}"
        )

        # Y position -> roll rate (row 1, col 1)
        y_pos_gain = abs(lqr.K[1, 1])
        assert y_pos_gain == pytest.approx(0.01), (
            f"Y position gain should be ~0.01, got {y_pos_gain}"
        )

        # Z position -> thrust (row 0, col 2)
        z_pos_gain = abs(lqr.K[0, 2])
        assert z_pos_gain == pytest.approx(4.0), (
            f"Z position gain should be ~4.0, got {z_pos_gain}"
        )

    def test_lqr_controller_custom_weights(self):
        """Test LQR controller with custom cost weights."""
        from quadcopter_tracking.controllers import LQRController

        config = {
            "q_pos": [5.0, 5.0, 10.0],
            "q_vel": [2.0, 2.0, 5.0],
            "r_thrust": 0.5,
            "r_rate": 2.0,
        }
        lqr = LQRController(config=config)
        assert lqr.K.shape == (4, 6)

    def test_lqr_controller_custom_k_matrix(self):
        """Test LQR controller with pre-defined K matrix."""
        from quadcopter_tracking.controllers import LQRController

        K = np.eye(4, 6) * 2.0
        config = {"K": K.tolist()}
        lqr = LQRController(config=config)
        assert np.allclose(lqr.K, K)

    def test_lqr_controller_invalid_k_shape(self):
        """Test LQR controller raises error for invalid K shape."""
        from quadcopter_tracking.controllers import LQRController

        config = {"K": [[1, 2, 3]]}
        with pytest.raises(ValueError, match="shape"):
            LQRController(config=config)

    def test_lqr_compute_action_format(self):
        """Test LQR compute_action returns correct format."""
        from quadcopter_tracking.controllers import LQRController

        lqr = LQRController()
        env = QuadcopterEnv()
        obs = env.reset(seed=42)

        action = lqr.compute_action(obs)

        assert "thrust" in action
        assert "roll_rate" in action
        assert "pitch_rate" in action
        assert "yaw_rate" in action
        assert isinstance(action["thrust"], float)
        assert isinstance(action["roll_rate"], float)

    def test_lqr_output_bounds(self):
        """Test LQR controller respects output bounds."""
        from quadcopter_tracking.controllers import LQRController

        config = {
            "max_thrust": 20.0,
            "min_thrust": 0.0,
            "max_rate": 3.0,
        }
        lqr = LQRController(config=config)
        env = QuadcopterEnv()
        obs = env.reset(seed=42)

        for _ in range(50):
            action = lqr.compute_action(obs)
            obs, _, done, _ = env.step(action)
            if done:
                break

            assert action["thrust"] >= 0.0
            assert action["thrust"] <= 20.0
            assert abs(action["roll_rate"]) <= 3.0
            assert abs(action["pitch_rate"]) <= 3.0
            assert abs(action["yaw_rate"]) <= 3.0

    def test_lqr_responds_to_state_error(self):
        """Test LQR controller responds correctly to state error."""
        from quadcopter_tracking.controllers import LQRController

        lqr = LQRController()

        # Create observation with known error
        obs = {
            "quadcopter": {
                "position": np.array([0.0, 0.0, 1.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
                "attitude": np.array([0.0, 0.0, 0.0]),
                "angular_velocity": np.array([0.0, 0.0, 0.0]),
            },
            "target": {
                "position": np.array([0.0, 0.0, 2.0]),  # Target is +1m in Z
                "velocity": np.array([0.0, 0.0, 0.0]),
            },
        }

        action = lqr.compute_action(obs)

        # Positive Z error should result in increased thrust
        assert action["thrust"] > 9.81  # More than hover thrust

    def test_lqr_moves_quadcopter_toward_target(self):
        """Test LQR controller reduces tracking error over time."""
        from quadcopter_tracking.controllers import LQRController

        config = EnvConfig()
        config.simulation = SimulationParams(dt=0.01, max_episode_time=10.0)
        config.target = TargetParams(motion_type="stationary")

        env = QuadcopterEnv(config=config)
        obs = env.reset(seed=42)
        lqr = LQRController()

        # Run controller for several steps and track minimum error achieved
        min_error = float("inf")
        for _ in range(500):
            action = lqr.compute_action(obs)
            obs, _, done, info = env.step(action)
            min_error = min(min_error, info["tracking_error"])
            if done:
                break

        # Controller should achieve reasonable tracking at some point
        assert min_error < 0.5, f"Min tracking error too high: {min_error}"

    def test_lqr_stateless(self):
        """Test LQR controller is stateless (reset is no-op)."""
        from quadcopter_tracking.controllers import LQRController

        lqr = LQRController()
        K_before = lqr.K.copy()
        lqr.reset()
        assert np.allclose(lqr.K, K_before)

    def test_lqr_observation_validation(self):
        """Test LQR controller raises error on invalid observation."""
        from quadcopter_tracking.controllers import LQRController

        lqr = LQRController()

        # Missing quadcopter key
        with pytest.raises(KeyError, match="quadcopter"):
            lqr.compute_action({"target": {}})

        # Missing target key
        with pytest.raises(KeyError, match="target"):
            lqr.compute_action({"quadcopter": {}})

    def test_lqr_hover_thrust_at_zero_error(self):
        """Test LQR returns hover thrust (~9.81N) when at target with zero velocity.

        This validates that LQR includes hover feedforward (hover_thrust =
        mass * gravity) and produces the correct baseline thrust when there
        is no tracking error.
        """
        from quadcopter_tracking.controllers import LQRController

        lqr = LQRController()

        # Create observation with zero position and velocity error (at hover)
        obs = {
            "quadcopter": {
                "position": np.array([0.0, 0.0, 1.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
                "attitude": np.array([0.0, 0.0, 0.0]),
                "angular_velocity": np.array([0.0, 0.0, 0.0]),
            },
            "target": {
                "position": np.array([0.0, 0.0, 1.0]),  # Same as quadcopter
                "velocity": np.array([0.0, 0.0, 0.0]),
            },
            "time": 0.0,
        }

        action = lqr.compute_action(obs)

        # With default parameters, thrust at zero error should equal hover_thrust
        assert abs(action["thrust"] - lqr.hover_thrust) < 0.01, (
            f"LQR thrust at zero error should be {lqr.hover_thrust}N, "
            f"got {action['thrust']}"
        )
        # All rates should be zero with no error
        assert abs(action["roll_rate"]) < 0.01
        assert abs(action["pitch_rate"]) < 0.01
        assert abs(action["yaw_rate"]) < 0.01

    def test_lqr_hover_thrust_custom_mass_gravity(self):
        """Test LQR hover thrust adjusts with custom mass/gravity configuration.

        Verifies that hover_thrust = mass * gravity scales correctly for
        non-default physics parameters.
        """
        from quadcopter_tracking.controllers import LQRController

        # Custom mass and gravity
        config = {"mass": 2.0, "gravity": 9.81}
        lqr = LQRController(config=config)

        # Expected hover thrust: 2.0 kg * 9.81 m/s² = 19.62 N
        expected_hover_thrust = 19.62

        # Verify hover_thrust attribute
        assert abs(lqr.hover_thrust - expected_hover_thrust) < 0.01

        # Create zero-error observation
        obs = {
            "quadcopter": {
                "position": np.array([0.0, 0.0, 1.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
                "attitude": np.array([0.0, 0.0, 0.0]),
                "angular_velocity": np.array([0.0, 0.0, 0.0]),
            },
            "target": {
                "position": np.array([0.0, 0.0, 1.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
            },
            "time": 0.0,
        }

        action = lqr.compute_action(obs)

        # Thrust should equal custom hover_thrust at zero error
        assert abs(action["thrust"] - expected_hover_thrust) < 0.01, (
            f"LQR thrust should be {expected_hover_thrust}N, got {action['thrust']}"
        )


class TestClassicalControllerIntegration:
    """Integration tests for classical controllers with environment."""

    def test_pid_full_episode_stationary(self):
        """Test PID controller completes a full episode with stationary target."""
        from quadcopter_tracking.controllers import PIDController

        config = EnvConfig()
        config.simulation = SimulationParams(dt=0.01, max_episode_time=5.0)
        config.target = TargetParams(motion_type="stationary")

        env = QuadcopterEnv(config=config)
        obs = env.reset(seed=42)
        pid = PIDController()

        done = False
        min_error = float("inf")

        while not done:
            action = pid.compute_action(obs)
            obs, _, done, info = env.step(action)
            min_error = min(min_error, info["tracking_error"])

        # Controller should achieve some tracking during episode
        assert min_error < 1.0, f"Min tracking error too high: {min_error}"

    def test_lqr_full_episode_stationary(self):
        """Test LQR controller completes a full episode with stationary target."""
        from quadcopter_tracking.controllers import LQRController

        config = EnvConfig()
        config.simulation = SimulationParams(dt=0.01, max_episode_time=5.0)
        config.target = TargetParams(motion_type="stationary")

        env = QuadcopterEnv(config=config)
        obs = env.reset(seed=42)
        lqr = LQRController()

        done = False
        min_error = float("inf")

        while not done:
            action = lqr.compute_action(obs)
            obs, _, done, info = env.step(action)
            min_error = min(min_error, info["tracking_error"])

        # Controller should achieve some tracking during episode
        assert min_error < 1.0, f"Min tracking error too high: {min_error}"

    def test_pid_linear_tracking(self):
        """Test PID controller tracks linear motion without diverging."""
        from quadcopter_tracking.controllers import PIDController

        config = EnvConfig()
        config.simulation = SimulationParams(dt=0.01, max_episode_time=5.0)
        config.target = TargetParams(motion_type="linear", speed=0.5)

        env = QuadcopterEnv(config=config)
        obs = env.reset(seed=42)
        pid = PIDController()

        errors = []
        for _ in range(200):
            action = pid.compute_action(obs)
            obs, _, done, info = env.step(action)
            errors.append(info["tracking_error"])
            if done:
                break

        # Error should stay bounded (not diverge to infinity)
        assert max(errors) < 50.0, f"Max error too high: {max(errors)}"
        # Controller should achieve some tracking at some point
        assert min(errors) < 2.0, f"Min error too high: {min(errors)}"

    def test_lqr_linear_tracking(self):
        """Test LQR controller tracks linear motion without diverging."""
        from quadcopter_tracking.controllers import LQRController

        config = EnvConfig()
        config.simulation = SimulationParams(dt=0.01, max_episode_time=5.0)
        config.target = TargetParams(motion_type="linear", speed=0.5)

        env = QuadcopterEnv(config=config)
        obs = env.reset(seed=42)
        lqr = LQRController()

        errors = []
        for _ in range(200):
            action = lqr.compute_action(obs)
            obs, _, done, info = env.step(action)
            errors.append(info["tracking_error"])
            if done:
                break

        # Error should stay bounded (not diverge to infinity)
        assert max(errors) < 50.0, f"Max error too high: {max(errors)}"
        # Controller should achieve some tracking at some point
        assert min(errors) < 2.0, f"Min error too high: {min(errors)}"

    def test_controller_saturation_handling(self):
        """Test controllers handle saturation gracefully."""
        from quadcopter_tracking.controllers import LQRController, PIDController

        # Large position offset to test saturation (5m in each axis)
        large_position_offset = [5.0, 5.0, 5.0]

        # Create a scenario with large initial error
        config = EnvConfig()
        config.simulation = SimulationParams(dt=0.01, max_episode_time=2.0)
        config.target = TargetParams(motion_type="stationary")

        for ControllerClass in [PIDController, LQRController]:
            env = QuadcopterEnv(config=config)
            obs = env.reset(seed=123)

            # Manually set large position offset
            state = env.get_state_vector()
            state[0:3] = large_position_offset
            env.set_state_vector(state)
            obs = env.render()

            controller = ControllerClass()
            action = controller.compute_action(obs)

            # Actions should be clipped to valid range
            assert action["thrust"] <= 20.0
            assert action["thrust"] >= 0.0
            assert abs(action["roll_rate"]) <= 3.0
            assert abs(action["pitch_rate"]) <= 3.0


# =============================================================================
# Hover Thrust Integration Tests
# =============================================================================
# These tests validate that PID and LQR controllers output correct hover thrust
# (~9.81N for default mass/gravity) when the quadcopter is at the target with
# zero velocity. Tests use public APIs and the environment to simulate the
# stationary hover scenario.


def create_hover_observation(
    position: np.ndarray | None = None,
) -> dict:
    """
    Create a stationary hover observation with zero tracking error.

    This helper constructs an observation dictionary where the quadcopter
    is exactly at the target position with zero velocity - the ideal hover
    condition.

    Args:
        position: Target/quadcopter position (default: [0, 0, 1]).

    Returns:
        Observation dictionary suitable for controller.compute_action().
    """
    if position is None:
        position = np.array([0.0, 0.0, 1.0])
    else:
        position = np.asarray(position, dtype=float)

    return {
        "quadcopter": {
            "position": position.copy(),
            "velocity": np.array([0.0, 0.0, 0.0]),
            "attitude": np.array([0.0, 0.0, 0.0]),
            "angular_velocity": np.array([0.0, 0.0, 0.0]),
        },
        "target": {
            "position": position.copy(),
            "velocity": np.array([0.0, 0.0, 0.0]),
        },
        "time": 0.0,
    }


def create_hover_env_config(
    mass: float = 1.0,
    gravity: float = 9.81,
) -> EnvConfig:
    """
    Create environment configuration for stationary hover testing.

    Sets up a minimal environment with:
    - Stationary target motion
    - Short episode for fast testing
    - Configurable mass/gravity

    Args:
        mass: Quadcopter mass in kg.
        gravity: Gravitational acceleration.

    Returns:
        EnvConfig configured for hover testing.
    """
    config = EnvConfig()
    config.simulation = SimulationParams(dt=0.01, max_episode_time=2.0)
    config.target = TargetParams(motion_type="stationary")
    config.quadcopter.mass = mass
    config.quadcopter.gravity = gravity
    return config


class TestHoverThrustIntegration:
    """
    Integration tests for hover thrust verification.

    These tests instantiate PID and LQR controllers through public APIs
    and verify they output correct hover thrust (within 0.5N of expected)
    when the quadcopter is at the target position with zero velocity.

    Acceptance criteria (from issue):
    - Thrust within 0.5N of mass * gravity (9.81N for default config)
    - No unintended torques (roll/pitch/yaw rates near zero)
    - Tests cover stationary target and zero-velocity setup
    - Tests are deterministic and fast (<1s each)
    """

    # Tolerance for hover thrust verification (per issue specification)
    HOVER_THRUST_TOLERANCE_N = 0.5

    # Tolerance for angular rates (should be near zero at hover)
    RATE_TOLERANCE_RAD_S = 0.01

    def test_pid_hover_thrust_integration(self):
        """
        Test PID outputs hover thrust (~9.81N) via environment integration.

        Instantiates PID controller and environment, positions quadcopter
        at target, and verifies thrust output matches hover baseline.
        """
        from quadcopter_tracking.controllers import PIDController

        # Setup environment with stationary target
        env_config = create_hover_env_config()
        env = QuadcopterEnv(config=env_config)
        env.reset(seed=42)

        # Position quadcopter exactly at target (zero error state)
        # Get target position from the environment's target generator
        target_state = env.target.get_state(0.0)
        target_pos = target_state["position"]
        state = env.get_state_vector()
        state[0:3] = target_pos  # Position
        state[3:6] = [0.0, 0.0, 0.0]  # Zero velocity
        env.set_state_vector(state)
        obs = env.render()

        # Create controller with matching physics parameters
        controller = PIDController(
            config={
                "mass": env_config.quadcopter.mass,
                "gravity": env_config.quadcopter.gravity,
            }
        )

        # Compute action at hover equilibrium
        action = controller.compute_action(obs)

        # Expected hover thrust
        expected_thrust = env_config.quadcopter.mass * env_config.quadcopter.gravity

        # Verify thrust within 0.5N tolerance
        thrust_error = abs(action["thrust"] - expected_thrust)
        assert thrust_error <= self.HOVER_THRUST_TOLERANCE_N, (
            f"PID hover thrust error {thrust_error:.3f}N exceeds tolerance "
            f"{self.HOVER_THRUST_TOLERANCE_N}N. Expected {expected_thrust:.2f}N, "
            f"got {action['thrust']:.2f}N"
        )

        # Verify no unintended torques (rates should be ~0)
        assert abs(action["roll_rate"]) < self.RATE_TOLERANCE_RAD_S, (
            f"PID roll_rate {action['roll_rate']:.4f} should be ~0 at hover"
        )
        assert abs(action["pitch_rate"]) < self.RATE_TOLERANCE_RAD_S, (
            f"PID pitch_rate {action['pitch_rate']:.4f} should be ~0 at hover"
        )
        assert abs(action["yaw_rate"]) < self.RATE_TOLERANCE_RAD_S, (
            f"PID yaw_rate {action['yaw_rate']:.4f} should be ~0 at hover"
        )

    def test_lqr_hover_thrust_integration(self):
        """
        Test LQR outputs hover thrust (~9.81N) via environment integration.

        Instantiates LQR controller and environment, positions quadcopter
        at target, and verifies thrust output matches hover baseline.
        """
        from quadcopter_tracking.controllers import LQRController

        # Setup environment with stationary target
        env_config = create_hover_env_config()
        env = QuadcopterEnv(config=env_config)
        env.reset(seed=42)

        # Position quadcopter exactly at target (zero error state)
        target_state = env.target.get_state(0.0)
        target_pos = target_state["position"]
        state = env.get_state_vector()
        state[0:3] = target_pos
        state[3:6] = [0.0, 0.0, 0.0]
        env.set_state_vector(state)
        obs = env.render()

        # Create controller with matching physics parameters
        controller = LQRController(
            config={
                "mass": env_config.quadcopter.mass,
                "gravity": env_config.quadcopter.gravity,
            }
        )

        # Compute action at hover equilibrium
        action = controller.compute_action(obs)

        # Expected hover thrust
        expected_thrust = env_config.quadcopter.mass * env_config.quadcopter.gravity

        # Verify thrust within 0.5N tolerance
        thrust_error = abs(action["thrust"] - expected_thrust)
        assert thrust_error <= self.HOVER_THRUST_TOLERANCE_N, (
            f"LQR hover thrust error {thrust_error:.3f}N exceeds tolerance "
            f"{self.HOVER_THRUST_TOLERANCE_N}N. Expected {expected_thrust:.2f}N, "
            f"got {action['thrust']:.2f}N"
        )

        # Verify no unintended torques
        assert abs(action["roll_rate"]) < self.RATE_TOLERANCE_RAD_S, (
            f"LQR roll_rate {action['roll_rate']:.4f} should be ~0 at hover"
        )
        assert abs(action["pitch_rate"]) < self.RATE_TOLERANCE_RAD_S, (
            f"LQR pitch_rate {action['pitch_rate']:.4f} should be ~0 at hover"
        )
        assert abs(action["yaw_rate"]) < self.RATE_TOLERANCE_RAD_S, (
            f"LQR yaw_rate {action['yaw_rate']:.4f} should be ~0 at hover"
        )

    def test_pid_hover_thrust_helper_observation(self):
        """
        Test PID hover thrust using shared helper observation.

        Uses create_hover_observation() helper to construct zero-error
        state and verifies PID outputs correct hover thrust.
        """
        from quadcopter_tracking.controllers import PIDController

        # Default mass/gravity
        mass, gravity = 1.0, 9.81
        expected_thrust = mass * gravity

        controller = PIDController(config={"mass": mass, "gravity": gravity})
        obs = create_hover_observation()

        action = controller.compute_action(obs)

        thrust_error = abs(action["thrust"] - expected_thrust)
        assert thrust_error <= self.HOVER_THRUST_TOLERANCE_N, (
            f"PID hover thrust {action['thrust']:.2f}N not within "
            f"{self.HOVER_THRUST_TOLERANCE_N}N of {expected_thrust:.2f}N"
        )

    def test_lqr_hover_thrust_helper_observation(self):
        """
        Test LQR hover thrust using shared helper observation.

        Uses create_hover_observation() helper to construct zero-error
        state and verifies LQR outputs correct hover thrust.
        """
        from quadcopter_tracking.controllers import LQRController

        mass, gravity = 1.0, 9.81
        expected_thrust = mass * gravity

        controller = LQRController(config={"mass": mass, "gravity": gravity})
        obs = create_hover_observation()

        action = controller.compute_action(obs)

        thrust_error = abs(action["thrust"] - expected_thrust)
        assert thrust_error <= self.HOVER_THRUST_TOLERANCE_N, (
            f"LQR hover thrust {action['thrust']:.2f}N not within "
            f"{self.HOVER_THRUST_TOLERANCE_N}N of {expected_thrust:.2f}N"
        )

    @pytest.mark.parametrize(
        "mass,gravity,description",
        [
            (1.0, 9.81, "default"),
            (1.5, 9.81, "heavier_quadcopter"),
            (1.0, 10.0, "higher_gravity"),
            (2.0, 9.81, "double_mass"),
            (0.5, 9.81, "light_quadcopter"),
        ],
    )
    def test_pid_hover_thrust_parametrized(self, mass, gravity, description):
        """
        Test PID hover thrust scales correctly with mass/gravity.

        Parameterized test verifying hover_thrust = mass * gravity
        for various configurations.
        """
        from quadcopter_tracking.controllers import PIDController

        expected_thrust = mass * gravity
        controller = PIDController(config={"mass": mass, "gravity": gravity})
        obs = create_hover_observation()

        action = controller.compute_action(obs)

        thrust_error = abs(action["thrust"] - expected_thrust)
        assert thrust_error <= self.HOVER_THRUST_TOLERANCE_N, (
            f"PID [{description}] thrust {action['thrust']:.2f}N not within "
            f"{self.HOVER_THRUST_TOLERANCE_N}N of expected {expected_thrust:.2f}N"
        )

    @pytest.mark.parametrize(
        "mass,gravity,description",
        [
            (1.0, 9.81, "default"),
            (1.5, 9.81, "heavier_quadcopter"),
            (1.0, 10.0, "higher_gravity"),
            (2.0, 9.81, "double_mass"),
            (0.5, 9.81, "light_quadcopter"),
        ],
    )
    def test_lqr_hover_thrust_parametrized(self, mass, gravity, description):
        """
        Test LQR hover thrust scales correctly with mass/gravity.

        Parameterized test verifying hover_thrust = mass * gravity
        for various configurations.
        """
        from quadcopter_tracking.controllers import LQRController

        expected_thrust = mass * gravity
        controller = LQRController(config={"mass": mass, "gravity": gravity})
        obs = create_hover_observation()

        action = controller.compute_action(obs)

        thrust_error = abs(action["thrust"] - expected_thrust)
        assert thrust_error <= self.HOVER_THRUST_TOLERANCE_N, (
            f"LQR [{description}] thrust {action['thrust']:.2f}N not within "
            f"{self.HOVER_THRUST_TOLERANCE_N}N of expected {expected_thrust:.2f}N"
        )

    def test_pid_zero_thrust_regression_guard(self):
        """
        Guard test: Fail loudly if PID regresses to zero-thrust behavior.

        Ensures PID doesn't output near-zero thrust at hover, which would
        indicate a regression in hover feedforward computation.
        """
        from quadcopter_tracking.controllers import PIDController

        controller = PIDController()
        obs = create_hover_observation()
        action = controller.compute_action(obs)

        # Thrust should NOT be near zero (regression guard)
        assert action["thrust"] > 1.0, (
            f"PID REGRESSION: thrust {action['thrust']:.2f}N is near zero. "
            "Hover feedforward may be broken."
        )

    def test_lqr_zero_thrust_regression_guard(self):
        """
        Guard test: Fail loudly if LQR regresses to zero-thrust behavior.

        Ensures LQR doesn't output near-zero thrust at hover, which would
        indicate a regression in hover feedforward computation.
        """
        from quadcopter_tracking.controllers import LQRController

        controller = LQRController()
        obs = create_hover_observation()
        action = controller.compute_action(obs)

        # Thrust should NOT be near zero (regression guard)
        assert action["thrust"] > 1.0, (
            f"LQR REGRESSION: thrust {action['thrust']:.2f}N is near zero. "
            "Hover feedforward may be broken."
        )

    def test_hover_stability_pid_multi_step(self):
        """
        Test PID maintains hover thrust over multiple time steps.

        Verifies thrust remains stable when quadcopter stays at hover
        equilibrium across multiple controller invocations.
        """
        from quadcopter_tracking.controllers import PIDController

        env_config = create_hover_env_config(mass=1.0, gravity=9.81)
        env = QuadcopterEnv(config=env_config)
        env.reset(seed=42)

        # Position quadcopter at hover equilibrium
        target_state = env.target.get_state(0.0)
        state = env.get_state_vector()
        state[0:3] = target_state["position"]
        state[3:6] = [0.0, 0.0, 0.0]
        env.set_state_vector(state)

        controller = PIDController(
            config={
                "mass": env_config.quadcopter.mass,
                "gravity": env_config.quadcopter.gravity,
            }
        )
        expected_thrust = env_config.quadcopter.mass * env_config.quadcopter.gravity

        # Simulate multiple timesteps at hover
        for i in range(10):
            obs = env.render()
            action = controller.compute_action(obs)
            env.step(action)

            # Verify thrust remains stable
            thrust_error = abs(action["thrust"] - expected_thrust)
            assert thrust_error <= self.HOVER_THRUST_TOLERANCE_N, (
                f"PID step {i}: thrust {action['thrust']:.2f}N drifted from "
                f"hover baseline {expected_thrust:.2f}N"
            )

    def test_hover_stability_lqr_multi_step(self):
        """
        Test LQR maintains hover thrust over multiple time steps.

        Verifies thrust remains stable when quadcopter stays at hover
        equilibrium across multiple controller invocations.
        """
        from quadcopter_tracking.controllers import LQRController

        env_config = create_hover_env_config(mass=1.0, gravity=9.81)
        env = QuadcopterEnv(config=env_config)
        env.reset(seed=42)

        # Position quadcopter at hover equilibrium
        target_state = env.target.get_state(0.0)
        state = env.get_state_vector()
        state[0:3] = target_state["position"]
        state[3:6] = [0.0, 0.0, 0.0]
        env.set_state_vector(state)

        controller = LQRController(
            config={
                "mass": env_config.quadcopter.mass,
                "gravity": env_config.quadcopter.gravity,
            }
        )
        expected_thrust = env_config.quadcopter.mass * env_config.quadcopter.gravity

        # Simulate multiple timesteps at hover
        for i in range(10):
            obs = env.render()
            action = controller.compute_action(obs)
            env.step(action)

            # Verify thrust remains stable
            thrust_error = abs(action["thrust"] - expected_thrust)
            assert thrust_error <= self.HOVER_THRUST_TOLERANCE_N, (
                f"LQR step {i}: thrust {action['thrust']:.2f}N drifted from "
                f"hover baseline {expected_thrust:.2f}N"
            )


# =============================================================================
# Axis Sign Convention Regression Tests
# =============================================================================
# These tests verify the controller sign conventions match environment dynamics:
#   - Environment: +pitch_rate produces +X velocity
#   - Environment: +roll_rate produces -Y velocity
#
# Controllers must output correct signs:
#   - Positive X error produces positive pitch_rate
#   - Positive Y error produces negative roll_rate


class TestAxisSignConventions:
    """
    Regression tests for axis sign conventions.

    Verifies that controllers output correct signs to match environment dynamics:
    - +pitch_rate → +X velocity (in environment)
    - +roll_rate → -Y velocity (in environment)

    These tests prevent future mismatches that would cause the quadcopter
    to accelerate away from targets instead of toward them.
    """

    def test_environment_pitch_produces_positive_x(self):
        """Verify environment: positive pitch_rate produces positive X velocity."""
        config = EnvConfig()
        config.simulation = SimulationParams(dt=0.01, max_episode_time=1.0)

        env = QuadcopterEnv(config=config)
        env.reset(seed=42)

        # Set to hover at origin with zero velocity
        state = np.zeros(12)
        state[2] = 1.0  # Z position
        env.set_state_vector(state)

        # Apply positive pitch_rate for multiple steps
        x_velocities = []
        for _ in range(50):
            action = {
                "thrust": 9.81,
                "roll_rate": 0.0,
                "pitch_rate": 0.5,  # Positive pitch rate
                "yaw_rate": 0.0,
            }
            obs, _, _, _ = env.step(action)
            x_velocities.append(obs["quadcopter"]["velocity"][0])

        # After applying positive pitch_rate, X velocity should be positive
        assert x_velocities[-1] > 0, (
            f"Environment dynamics violated: +pitch_rate should produce +X velocity. "
            f"Got X velocity = {x_velocities[-1]:.4f}"
        )

    def test_environment_roll_produces_negative_y(self):
        """Verify environment: positive roll_rate produces negative Y velocity."""
        config = EnvConfig()
        config.simulation = SimulationParams(dt=0.01, max_episode_time=1.0)

        env = QuadcopterEnv(config=config)
        env.reset(seed=42)

        # Set to hover at origin with zero velocity
        state = np.zeros(12)
        state[2] = 1.0  # Z position
        env.set_state_vector(state)

        # Apply positive roll_rate for multiple steps
        y_velocities = []
        for _ in range(50):
            action = {
                "thrust": 9.81,
                "roll_rate": 0.5,  # Positive roll rate
                "pitch_rate": 0.0,
                "yaw_rate": 0.0,
            }
            obs, _, _, _ = env.step(action)
            y_velocities.append(obs["quadcopter"]["velocity"][1])

        # After applying positive roll_rate, Y velocity should be negative
        assert y_velocities[-1] < 0, (
            f"Environment dynamics violated: +roll_rate should produce -Y velocity. "
            f"Got Y velocity = {y_velocities[-1]:.4f}"
        )

    def test_pid_positive_x_error_produces_positive_pitch_rate(self):
        """Verify PID: positive X error produces positive pitch_rate output."""
        from quadcopter_tracking.controllers import PIDController

        controller = PIDController()

        # Observation with positive X error (target ahead in X)
        obs = {
            "quadcopter": {
                "position": np.array([0.0, 0.0, 1.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
                "attitude": np.array([0.0, 0.0, 0.0]),
                "angular_velocity": np.array([0.0, 0.0, 0.0]),
            },
            "target": {
                "position": np.array([1.0, 0.0, 1.0]),  # +1m in X
                "velocity": np.array([0.0, 0.0, 0.0]),
            },
            "time": 0.0,
        }

        action = controller.compute_action(obs)

        assert action["pitch_rate"] > 0, (
            f"PID sign convention violated: +X error should produce +pitch_rate. "
            f"Got pitch_rate = {action['pitch_rate']:.4f}"
        )

    def test_pid_positive_y_error_produces_negative_roll_rate(self):
        """Verify PID: positive Y error produces negative roll_rate output."""
        from quadcopter_tracking.controllers import PIDController

        controller = PIDController()

        # Observation with positive Y error (target ahead in Y)
        obs = {
            "quadcopter": {
                "position": np.array([0.0, 0.0, 1.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
                "attitude": np.array([0.0, 0.0, 0.0]),
                "angular_velocity": np.array([0.0, 0.0, 0.0]),
            },
            "target": {
                "position": np.array([0.0, 1.0, 1.0]),  # +1m in Y
                "velocity": np.array([0.0, 0.0, 0.0]),
            },
            "time": 0.0,
        }

        action = controller.compute_action(obs)

        assert action["roll_rate"] < 0, (
            f"PID sign convention violated: +Y error should produce -roll_rate. "
            f"Got roll_rate = {action['roll_rate']:.4f}"
        )

    def test_lqr_positive_x_error_produces_positive_pitch_rate(self):
        """Verify LQR: positive X error produces positive pitch_rate output."""
        from quadcopter_tracking.controllers import LQRController

        controller = LQRController()

        # Observation with positive X error (target ahead in X)
        obs = {
            "quadcopter": {
                "position": np.array([0.0, 0.0, 1.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
                "attitude": np.array([0.0, 0.0, 0.0]),
                "angular_velocity": np.array([0.0, 0.0, 0.0]),
            },
            "target": {
                "position": np.array([1.0, 0.0, 1.0]),  # +1m in X
                "velocity": np.array([0.0, 0.0, 0.0]),
            },
            "time": 0.0,
        }

        action = controller.compute_action(obs)

        assert action["pitch_rate"] > 0, (
            f"LQR sign convention violated: +X error should produce +pitch_rate. "
            f"Got pitch_rate = {action['pitch_rate']:.4f}"
        )

    def test_lqr_positive_y_error_produces_negative_roll_rate(self):
        """Verify LQR: positive Y error produces negative roll_rate output."""
        from quadcopter_tracking.controllers import LQRController

        controller = LQRController()

        # Observation with positive Y error (target ahead in Y)
        obs = {
            "quadcopter": {
                "position": np.array([0.0, 0.0, 1.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
                "attitude": np.array([0.0, 0.0, 0.0]),
                "angular_velocity": np.array([0.0, 0.0, 0.0]),
            },
            "target": {
                "position": np.array([0.0, 1.0, 1.0]),  # +1m in Y
                "velocity": np.array([0.0, 0.0, 0.0]),
            },
            "time": 0.0,
        }

        action = controller.compute_action(obs)

        assert action["roll_rate"] < 0, (
            f"LQR sign convention violated: +Y error should produce -roll_rate. "
            f"Got roll_rate = {action['roll_rate']:.4f}"
        )

    def test_pid_convergence_to_stationary_target(self):
        """Verify PID with fixed signs converges to target within 0.5m."""
        from quadcopter_tracking.controllers import PIDController

        config = EnvConfig()
        config.simulation = SimulationParams(dt=0.01, max_episode_time=10.0)
        config.target = TargetParams(motion_type="stationary")

        env = QuadcopterEnv(config=config)
        obs = env.reset(seed=42)

        controller = PIDController(
            config={
                "mass": config.quadcopter.mass,
                "gravity": config.quadcopter.gravity,
            }
        )

        # Track minimum error achieved
        min_error = float("inf")
        for _ in range(500):
            action = controller.compute_action(obs)
            obs, _, done, info = env.step(action)
            min_error = min(min_error, info["tracking_error"])
            if done:
                break

        # Controller should converge to within 0.5m
        assert min_error < 0.5, (
            f"PID controller failed to converge. Min tracking error = "
            f"{min_error:.3f}m. This may indicate sign convention issues."
        )

    def test_lqr_convergence_to_stationary_target(self):
        """Verify LQR with fixed signs converges to target within 0.5m."""
        from quadcopter_tracking.controllers import LQRController

        config = EnvConfig()
        config.simulation = SimulationParams(dt=0.01, max_episode_time=10.0)
        config.target = TargetParams(motion_type="stationary")

        env = QuadcopterEnv(config=config)
        obs = env.reset(seed=42)

        controller = LQRController(
            config={
                "mass": config.quadcopter.mass,
                "gravity": config.quadcopter.gravity,
            }
        )

        # Track minimum error achieved
        min_error = float("inf")
        for _ in range(500):
            action = controller.compute_action(obs)
            obs, _, done, info = env.step(action)
            min_error = min(min_error, info["tracking_error"])
            if done:
                break

        # Controller should converge to within 0.5m
        assert min_error < 0.5, (
            f"LQR controller failed to converge. Min tracking error = "
            f"{min_error:.3f}m. This may indicate sign convention issues."
        )

    def test_pid_initial_acceleration_direction(self):
        """Verify PID initial acceleration is toward target, not away.

        This test uses higher gains than the conservative defaults to ensure
        detectable velocity within a short time window. The test validates
        sign conventions, not tuning quality.
        """
        from quadcopter_tracking.controllers import PIDController

        config = EnvConfig()
        config.simulation = SimulationParams(dt=0.01, max_episode_time=1.0)
        config.target = TargetParams(motion_type="stationary")

        env = QuadcopterEnv(config=config)
        env.reset(seed=42)

        # Get the target position
        target_state = env.target.get_state(0.0)
        target_pos = target_state["position"]

        # Position quadcopter away from target to create meaningful error
        state = np.zeros(12)
        state[0] = target_pos[0] - 2.0  # 2m behind in X
        state[1] = target_pos[1] - 2.0  # 2m behind in Y
        state[2] = target_pos[2]  # Same Z height
        env.set_state_vector(state)

        # Use higher test gains to get detectable velocity in short time window
        # This tests sign conventions, not the conservative default tuning
        controller = PIDController(
            config={
                "mass": config.quadcopter.mass,
                "gravity": config.quadcopter.gravity,
                "kp_pos": [2.0, 2.0, 4.0],  # Higher gains for sign convention test
                "ki_pos": [0.0, 0.0, 0.0],  # Explicitly zero for clarity
                "kd_pos": [1.5, 1.5, 2.0],
                "integral_limit": 0.0,
            }
        )

        obs = env.render()

        # Take a few steps and check velocity direction
        for _ in range(20):
            action = controller.compute_action(obs)
            obs, _, _, _ = env.step(action)

        quad_vel = obs["quadcopter"]["velocity"]
        quad_pos = obs["quadcopter"]["position"]

        # Compute direction to target
        to_target = target_pos - quad_pos
        to_target_xy = to_target[:2]
        vel_xy = quad_vel[:2]

        # Velocity should have positive dot product with direction to target
        # (i.e., moving toward target, not away)
        dot_product = np.dot(vel_xy, to_target_xy)
        vel_mag = np.linalg.norm(vel_xy)

        # Only check if there's significant XY velocity
        if vel_mag > 0.1:
            assert dot_product > 0, (
                f"PID produces motion AWAY from target. "
                f"Velocity XY = {vel_xy}, direction to target = {to_target_xy}. "
                "This indicates sign convention issues."
            )
        else:
            pytest.fail(
                f"PID failed to produce significant velocity towards target. "
                f"Velocity magnitude was {vel_mag:.4f}, expecting > 0.1."
            )

    def test_lqr_initial_acceleration_direction(self):
        """Verify LQR initial acceleration is toward target, not away.

        This test uses higher cost weights than the conservative defaults to
        produce larger feedback gains for detectable velocity. The test validates
        sign conventions, not tuning quality.
        """
        from quadcopter_tracking.controllers import LQRController

        config = EnvConfig()
        config.simulation = SimulationParams(dt=0.01, max_episode_time=1.0)
        config.target = TargetParams(motion_type="stationary")

        env = QuadcopterEnv(config=config)
        env.reset(seed=42)

        # Get the target position
        target_state = env.target.get_state(0.0)
        target_pos = target_state["position"]

        # Position quadcopter away from target to create meaningful error
        state = np.zeros(12)
        state[0] = target_pos[0] - 2.0  # 2m behind in X
        state[1] = target_pos[1] - 2.0  # 2m behind in Y
        state[2] = target_pos[2]  # Same Z height
        env.set_state_vector(state)

        # Use higher cost weights to get detectable velocity in short time window
        # This tests sign conventions, not the conservative default tuning
        controller = LQRController(
            config={
                "mass": config.quadcopter.mass,
                "gravity": config.quadcopter.gravity,
                "q_pos": [10.0, 10.0, 20.0],  # Higher costs for sign convention test
                "q_vel": [5.0, 5.0, 10.0],
                "r_thrust": 0.1,
                "r_rate": 1.0,
            }
        )

        obs = env.render()

        # Take a few steps and check velocity direction
        for _ in range(20):
            action = controller.compute_action(obs)
            obs, _, _, _ = env.step(action)

        quad_vel = obs["quadcopter"]["velocity"]
        quad_pos = obs["quadcopter"]["position"]

        # Compute direction to target
        to_target = target_pos - quad_pos
        to_target_xy = to_target[:2]
        vel_xy = quad_vel[:2]

        # Velocity should have positive dot product with direction to target
        # (i.e., moving toward target, not away)
        dot_product = np.dot(vel_xy, to_target_xy)
        vel_mag = np.linalg.norm(vel_xy)

        # Only check if there's significant XY velocity
        if vel_mag > 0.1:
            assert dot_product > 0, (
                f"LQR produces motion AWAY from target. "
                f"Velocity XY = {vel_xy}, direction to target = {to_target_xy}. "
                "This indicates sign convention issues."
            )
        else:
            pytest.fail(
                f"LQR failed to produce significant velocity towards target. "
                f"Velocity magnitude was {vel_mag:.4f}, expecting > 0.1."
            )


# =============================================================================
# Feedforward Support Tests
# =============================================================================
# These tests verify the optional feedforward support in PID and LQR controllers.
# Feedforward terms scale target velocity and acceleration to improve tracking
# of moving targets while remaining fully optional and backwards-compatible.


class TestFeedforwardSupport:
    """
    Tests for optional feedforward support in PID and LQR controllers.

    Tests verify:
    - Feedforward is disabled by default (backward compatibility)
    - Feedforward can be enabled via configuration
    - Feedforward improves tracking of moving targets
    - Fallback when acceleration data is unavailable
    - Stationary targets remain stable with feedforward off
    - Diagnostics logging of P/I/D/FF components
    """

    def test_pid_feedforward_disabled_by_default(self):
        """Test PID feedforward is disabled by default for backward compat."""
        from quadcopter_tracking.controllers import PIDController

        pid = PIDController()

        assert pid.feedforward_enabled is False
        assert np.allclose(pid.ff_velocity_gain, [0.0, 0.0, 0.0])
        assert np.allclose(pid.ff_acceleration_gain, [0.0, 0.0, 0.0])

    def test_lqr_feedforward_disabled_by_default(self):
        """Test LQR feedforward is disabled by default for backward compat."""
        from quadcopter_tracking.controllers import LQRController

        lqr = LQRController()

        assert lqr.feedforward_enabled is False
        assert np.allclose(lqr.ff_velocity_gain, [0.0, 0.0, 0.0])
        assert np.allclose(lqr.ff_acceleration_gain, [0.0, 0.0, 0.0])

    def test_pid_feedforward_can_be_enabled(self):
        """Test PID feedforward can be enabled via config."""
        from quadcopter_tracking.controllers import PIDController

        config = {
            "feedforward_enabled": True,
            "ff_velocity_gain": [0.1, 0.1, 0.1],
            "ff_acceleration_gain": [0.05, 0.05, 0.05],
        }
        pid = PIDController(config=config)

        assert pid.feedforward_enabled is True
        assert np.allclose(pid.ff_velocity_gain, [0.1, 0.1, 0.1])
        assert np.allclose(pid.ff_acceleration_gain, [0.05, 0.05, 0.05])

    def test_lqr_feedforward_can_be_enabled(self):
        """Test LQR feedforward can be enabled via config."""
        from quadcopter_tracking.controllers import LQRController

        config = {
            "feedforward_enabled": True,
            "ff_velocity_gain": [0.1, 0.1, 0.1],
            "ff_acceleration_gain": [0.05, 0.05, 0.05],
        }
        lqr = LQRController(config=config)

        assert lqr.feedforward_enabled is True
        assert np.allclose(lqr.ff_velocity_gain, [0.1, 0.1, 0.1])
        assert np.allclose(lqr.ff_acceleration_gain, [0.05, 0.05, 0.05])

    def test_pid_feedforward_scalar_gains(self):
        """Test PID feedforward accepts scalar gains (broadcast to array)."""
        from quadcopter_tracking.controllers import PIDController

        config = {
            "feedforward_enabled": True,
            "ff_velocity_gain": 0.2,
            "ff_acceleration_gain": 0.1,
        }
        pid = PIDController(config=config)

        assert np.allclose(pid.ff_velocity_gain, [0.2, 0.2, 0.2])
        assert np.allclose(pid.ff_acceleration_gain, [0.1, 0.1, 0.1])

    def test_pid_stationary_stable_with_feedforward_off(self):
        """Test PID with feedforward OFF is stable for stationary targets."""
        from quadcopter_tracking.controllers import PIDController

        config = EnvConfig()
        config.simulation = SimulationParams(dt=0.01, max_episode_time=5.0)
        config.target = TargetParams(motion_type="stationary")

        env = QuadcopterEnv(config=config)
        obs = env.reset(seed=42)

        # Feedforward disabled by default
        pid = PIDController()
        assert pid.feedforward_enabled is False

        min_error = float("inf")
        for _ in range(300):
            action = pid.compute_action(obs)
            obs, _, done, info = env.step(action)
            min_error = min(min_error, info["tracking_error"])
            if done:
                break

        # Should converge well with feedforward disabled
        assert min_error < 0.5, (
            f"PID with feedforward OFF should converge. Min error: {min_error:.3f}"
        )

    def test_lqr_stationary_stable_with_feedforward_off(self):
        """Test LQR with feedforward OFF is stable for stationary targets."""
        from quadcopter_tracking.controllers import LQRController

        config = EnvConfig()
        config.simulation = SimulationParams(dt=0.01, max_episode_time=5.0)
        config.target = TargetParams(motion_type="stationary")

        env = QuadcopterEnv(config=config)
        obs = env.reset(seed=42)

        # Feedforward disabled by default
        lqr = LQRController()
        assert lqr.feedforward_enabled is False

        min_error = float("inf")
        for _ in range(300):
            action = lqr.compute_action(obs)
            obs, _, done, info = env.step(action)
            min_error = min(min_error, info["tracking_error"])
            if done:
                break

        # Should converge well with feedforward disabled
        assert min_error < 0.5, (
            f"LQR with feedforward OFF should converge. Min error: {min_error:.3f}"
        )

    def test_pid_baseline_unchanged_with_feedforward_disabled(self):
        """Test PID output is unchanged when feedforward is disabled."""
        from quadcopter_tracking.controllers import PIDController

        # Create observation with moving target
        obs = {
            "quadcopter": {
                "position": np.array([0.0, 0.0, 1.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
                "attitude": np.array([0.0, 0.0, 0.0]),
                "angular_velocity": np.array([0.0, 0.0, 0.0]),
            },
            "target": {
                "position": np.array([1.0, 0.0, 1.0]),
                "velocity": np.array([1.0, 0.0, 0.0]),
                "acceleration": np.array([0.5, 0.0, 0.0]),
            },
            "time": 0.0,
        }

        pid_off = PIDController(config={"feedforward_enabled": False})
        pid_on = PIDController(
            config={
                "feedforward_enabled": True,
                "ff_velocity_gain": [0.1, 0.1, 0.1],
            }
        )

        action_off = pid_off.compute_action(obs)
        action_on = pid_on.compute_action(obs)

        # With feedforward enabled, pitch_rate should be higher (tracking X vel)
        assert action_on["pitch_rate"] > action_off["pitch_rate"], (
            "Feedforward should increase pitch_rate for +X target velocity"
        )

    def test_pid_diagnostics_logs_control_components(self):
        """Test PID logs P/I/D/FF control components for diagnostics."""
        from quadcopter_tracking.controllers import PIDController

        config = {
            "feedforward_enabled": True,
            "ff_velocity_gain": [0.1, 0.1, 0.1],
            "ff_acceleration_gain": [0.05, 0.05, 0.05],
        }
        pid = PIDController(config=config)

        obs = {
            "quadcopter": {
                "position": np.array([0.0, 0.0, 1.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
                "attitude": np.array([0.0, 0.0, 0.0]),
                "angular_velocity": np.array([0.0, 0.0, 0.0]),
            },
            "target": {
                "position": np.array([1.0, 0.0, 1.0]),
                "velocity": np.array([0.5, 0.0, 0.0]),
                "acceleration": np.array([0.2, 0.0, 0.0]),
            },
            "time": 0.0,
        }

        pid.compute_action(obs)
        components = pid.get_control_components()

        assert components is not None
        assert "p_term" in components
        assert "i_term" in components
        assert "d_term" in components
        assert "ff_velocity_term" in components
        assert "ff_acceleration_term" in components
        assert "total_correction" in components

        # FF terms should be nonzero with moving target
        assert components["ff_velocity_term"][0] > 0  # X velocity feedforward
        assert components["ff_acceleration_term"][0] > 0  # X accel feedforward

    def test_lqr_diagnostics_logs_control_components(self):
        """Test LQR logs feedback/FF control components for diagnostics."""
        from quadcopter_tracking.controllers import LQRController

        config = {
            "feedforward_enabled": True,
            "ff_velocity_gain": [0.1, 0.1, 0.1],
        }
        lqr = LQRController(config=config)

        obs = {
            "quadcopter": {
                "position": np.array([0.0, 0.0, 1.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
                "attitude": np.array([0.0, 0.0, 0.0]),
                "angular_velocity": np.array([0.0, 0.0, 0.0]),
            },
            "target": {
                "position": np.array([1.0, 0.0, 1.0]),
                "velocity": np.array([0.5, 0.0, 0.0]),
                "acceleration": np.array([0.0, 0.0, 0.0]),
            },
            "time": 0.0,
        }

        lqr.compute_action(obs)
        components = lqr.get_control_components()

        assert components is not None
        assert "feedback_u" in components
        assert "ff_velocity_term" in components
        assert "ff_acceleration_term" in components

        # FF velocity term should be nonzero
        assert components["ff_velocity_term"][0] > 0

    def test_pid_feedforward_graceful_fallback_no_acceleration(self):
        """Test PID gracefully falls back when acceleration data missing."""
        from quadcopter_tracking.controllers import PIDController

        config = {
            "feedforward_enabled": True,
            "ff_velocity_gain": [0.1, 0.1, 0.1],
            "ff_acceleration_gain": [0.05, 0.05, 0.05],
        }
        pid = PIDController(config=config)

        # Observation without acceleration key
        obs = {
            "quadcopter": {
                "position": np.array([0.0, 0.0, 1.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
                "attitude": np.array([0.0, 0.0, 0.0]),
                "angular_velocity": np.array([0.0, 0.0, 0.0]),
            },
            "target": {
                "position": np.array([1.0, 0.0, 1.0]),
                "velocity": np.array([0.5, 0.0, 0.0]),
                # No "acceleration" key
            },
            "time": 0.0,
        }

        # Should not raise error
        action = pid.compute_action(obs)

        assert "thrust" in action
        assert "pitch_rate" in action

        # Check diagnostics show zero acceleration feedforward
        components = pid.get_control_components()
        assert np.allclose(components["ff_acceleration_term"], [0.0, 0.0, 0.0])

    def test_lqr_feedforward_graceful_fallback_no_acceleration(self):
        """Test LQR gracefully falls back when acceleration data missing."""
        from quadcopter_tracking.controllers import LQRController

        config = {
            "feedforward_enabled": True,
            "ff_velocity_gain": [0.1, 0.1, 0.1],
            "ff_acceleration_gain": [0.05, 0.05, 0.05],
        }
        lqr = LQRController(config=config)

        # Observation without acceleration key
        obs = {
            "quadcopter": {
                "position": np.array([0.0, 0.0, 1.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
                "attitude": np.array([0.0, 0.0, 0.0]),
                "angular_velocity": np.array([0.0, 0.0, 0.0]),
            },
            "target": {
                "position": np.array([1.0, 0.0, 1.0]),
                "velocity": np.array([0.5, 0.0, 0.0]),
                # No "acceleration" key
            },
            "time": 0.0,
        }

        # Should not raise error
        action = lqr.compute_action(obs)

        assert "thrust" in action
        assert "pitch_rate" in action

        # Check diagnostics show zero acceleration feedforward
        components = lqr.get_control_components()
        assert np.allclose(components["ff_acceleration_term"], [0.0, 0.0, 0.0])

    def test_pid_feedforward_velocity_clamping(self):
        """Test PID clamps feedforward velocity to prevent oscillation."""
        from quadcopter_tracking.controllers import PIDController

        config = {
            "feedforward_enabled": True,
            "ff_velocity_gain": [1.0, 1.0, 1.0],
            "ff_max_velocity": 2.0,
        }
        pid = PIDController(config=config)

        # Observation with very high target velocity
        obs = {
            "quadcopter": {
                "position": np.array([0.0, 0.0, 1.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
                "attitude": np.array([0.0, 0.0, 0.0]),
                "angular_velocity": np.array([0.0, 0.0, 0.0]),
            },
            "target": {
                "position": np.array([0.0, 0.0, 1.0]),  # Zero position error
                "velocity": np.array([100.0, 0.0, 0.0]),  # Very high velocity
            },
            "time": 0.0,
        }

        pid.compute_action(obs)
        components = pid.get_control_components()

        # Feedforward should be clamped to max_velocity * gain
        ff_x = components["ff_velocity_term"][0]
        assert ff_x <= 2.0 * 1.0, (
            f"Feedforward should be clamped. Got {ff_x}, expected <= 2.0"
        )

    def test_pid_feedforward_acceleration_clamping(self):
        """Test PID clamps feedforward acceleration to prevent oscillation."""
        from quadcopter_tracking.controllers import PIDController

        config = {
            "feedforward_enabled": True,
            "ff_acceleration_gain": [1.0, 1.0, 1.0],
            "ff_max_acceleration": 3.0,
        }
        pid = PIDController(config=config)

        # Observation with very high target acceleration
        obs = {
            "quadcopter": {
                "position": np.array([0.0, 0.0, 1.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
                "attitude": np.array([0.0, 0.0, 0.0]),
                "angular_velocity": np.array([0.0, 0.0, 0.0]),
            },
            "target": {
                "position": np.array([0.0, 0.0, 1.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
                "acceleration": np.array([50.0, 0.0, 0.0]),  # Very high accel
            },
            "time": 0.0,
        }

        pid.compute_action(obs)
        components = pid.get_control_components()

        # Feedforward should be clamped
        ff_x = components["ff_acceleration_term"][0]
        assert ff_x <= 3.0 * 1.0, (
            f"Feedforward should be clamped. Got {ff_x}, expected <= 3.0"
        )

    def test_pid_reset_clears_diagnostics(self):
        """Test PID reset clears control component diagnostics."""
        from quadcopter_tracking.controllers import PIDController

        pid = PIDController(config={"feedforward_enabled": True})

        obs = create_hover_observation()
        pid.compute_action(obs)
        assert pid.get_control_components() is not None

        pid.reset()
        assert pid.get_control_components() is None

    def test_lqr_reset_clears_diagnostics(self):
        """Test LQR reset clears control component diagnostics."""
        from quadcopter_tracking.controllers import LQRController

        lqr = LQRController(config={"feedforward_enabled": True})

        obs = create_hover_observation()
        lqr.compute_action(obs)
        assert lqr.get_control_components() is not None

        lqr.reset()
        assert lqr.get_control_components() is None


# =============================================================================
# Riccati-LQR Controller Tests
# =============================================================================
# Tests for the RiccatiLQRController that solves the discrete-time algebraic
# Riccati equation (DARE) for optimal feedback gains.


class TestRiccatiLQRController:
    """Tests for Riccati-LQR controller implementation."""

    def test_riccati_lqr_controller_initialization(self):
        """Test Riccati-LQR controller initializes correctly."""
        from quadcopter_tracking.controllers import RiccatiLQRController

        controller = RiccatiLQRController(config={"dt": 0.01})
        assert controller.name == "riccati_lqr"
        assert controller.K is not None
        assert controller.K.shape == (4, 6)
        assert controller.P is not None
        assert controller.P.shape == (6, 6)
        assert not controller.is_using_fallback()

    def test_riccati_lqr_dare_solution(self):
        """Test that DARE solution produces valid P and K matrices."""
        from quadcopter_tracking.controllers import RiccatiLQRController

        controller = RiccatiLQRController(config={"dt": 0.01})

        P = controller.get_riccati_solution()
        K = controller.get_gain_matrix()

        assert P is not None
        assert K is not None

        # P should be positive semi-definite (all eigenvalues >= 0)
        eigenvalues = np.linalg.eigvalsh(P)
        assert np.all(eigenvalues >= -1e-10), (
            f"P should be positive semi-definite, got eigenvalues: {eigenvalues}"
        )

        # P should be symmetric
        assert np.allclose(P, P.T), "P should be symmetric"

    def test_riccati_lqr_custom_cost_matrices(self):
        """Test Riccati-LQR with custom Q and R cost matrices."""
        from quadcopter_tracking.controllers import RiccatiLQRController

        config = {
            "dt": 0.01,
            "q_pos": [2.0, 2.0, 10.0],
            "q_vel": [0.5, 0.5, 2.0],
            "r_controls": [1.0, 0.5, 0.5, 1.0],
        }
        controller = RiccatiLQRController(config=config)

        assert controller.K.shape == (4, 6)
        assert not controller.is_using_fallback()

    def test_riccati_lqr_full_Q_R_matrices(self):
        """Test Riccati-LQR with full Q and R matrices."""
        from quadcopter_tracking.controllers import RiccatiLQRController

        Q = np.diag([1.0, 1.0, 10.0, 0.1, 0.1, 1.0])
        R = np.diag([1.0, 1.0, 1.0, 1.0])

        config = {
            "dt": 0.01,
            "Q": Q.tolist(),
            "R": R.tolist(),
        }
        controller = RiccatiLQRController(config=config)

        assert controller.K.shape == (4, 6)
        assert not controller.is_using_fallback()

    def test_riccati_lqr_compute_action_format(self):
        """Test Riccati-LQR compute_action returns correct format."""
        from quadcopter_tracking.controllers import RiccatiLQRController

        controller = RiccatiLQRController(config={"dt": 0.01})
        env = QuadcopterEnv()
        obs = env.reset(seed=42)

        action = controller.compute_action(obs)

        assert "thrust" in action
        assert "roll_rate" in action
        assert "pitch_rate" in action
        assert "yaw_rate" in action
        assert isinstance(action["thrust"], float)
        assert isinstance(action["roll_rate"], float)

    def test_riccati_lqr_output_bounds(self):
        """Test Riccati-LQR controller respects output bounds."""
        from quadcopter_tracking.controllers import RiccatiLQRController

        config = {
            "dt": 0.01,
            "max_thrust": 20.0,
            "min_thrust": 0.0,
            "max_rate": 3.0,
        }
        controller = RiccatiLQRController(config=config)
        env = QuadcopterEnv()
        obs = env.reset(seed=42)

        for _ in range(50):
            action = controller.compute_action(obs)
            obs, _, done, _ = env.step(action)
            if done:
                break

            assert action["thrust"] >= 0.0
            assert action["thrust"] <= 20.0
            assert abs(action["roll_rate"]) <= 3.0
            assert abs(action["pitch_rate"]) <= 3.0
            assert abs(action["yaw_rate"]) <= 3.0

    def test_riccati_lqr_hover_thrust_at_zero_error(self):
        """Test Riccati-LQR returns hover thrust when at target with zero velocity."""
        from quadcopter_tracking.controllers import RiccatiLQRController

        controller = RiccatiLQRController(config={"dt": 0.01})

        obs = {
            "quadcopter": {
                "position": np.array([0.0, 0.0, 1.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
                "attitude": np.array([0.0, 0.0, 0.0]),
                "angular_velocity": np.array([0.0, 0.0, 0.0]),
            },
            "target": {
                "position": np.array([0.0, 0.0, 1.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
            },
            "time": 0.0,
        }

        action = controller.compute_action(obs)

        # At zero error, thrust should equal hover_thrust
        assert abs(action["thrust"] - controller.hover_thrust) < 0.01, (
            f"Riccati-LQR thrust at zero error should be {controller.hover_thrust}N, "
            f"got {action['thrust']}"
        )
        # All rates should be zero with no error
        assert abs(action["roll_rate"]) < 0.01
        assert abs(action["pitch_rate"]) < 0.01
        assert abs(action["yaw_rate"]) < 0.01

    def test_riccati_lqr_responds_to_position_error(self):
        """Test Riccati-LQR controller responds correctly to position error."""
        from quadcopter_tracking.controllers import RiccatiLQRController

        controller = RiccatiLQRController(config={"dt": 0.01})

        # Observation with positive X error (target ahead in X)
        obs = {
            "quadcopter": {
                "position": np.array([0.0, 0.0, 1.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
                "attitude": np.array([0.0, 0.0, 0.0]),
                "angular_velocity": np.array([0.0, 0.0, 0.0]),
            },
            "target": {
                "position": np.array([1.0, 0.0, 1.0]),  # +1m in X
                "velocity": np.array([0.0, 0.0, 0.0]),
            },
        }

        action = controller.compute_action(obs)

        # Positive X error should result in positive pitch rate
        assert action["pitch_rate"] > 0, (
            f"Riccati-LQR should produce positive pitch_rate for +X error. "
            f"Got pitch_rate = {action['pitch_rate']}"
        )

    def test_riccati_lqr_responds_to_y_error(self):
        """Test Riccati-LQR controller produces negative roll_rate for +Y error."""
        from quadcopter_tracking.controllers import RiccatiLQRController

        controller = RiccatiLQRController(config={"dt": 0.01})

        obs = {
            "quadcopter": {
                "position": np.array([0.0, 0.0, 1.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
                "attitude": np.array([0.0, 0.0, 0.0]),
                "angular_velocity": np.array([0.0, 0.0, 0.0]),
            },
            "target": {
                "position": np.array([0.0, 1.0, 1.0]),  # +1m in Y
                "velocity": np.array([0.0, 0.0, 0.0]),
            },
        }

        action = controller.compute_action(obs)

        # Positive Y error should result in negative roll rate
        assert action["roll_rate"] < 0, (
            f"Riccati-LQR should produce negative roll_rate for +Y error. "
            f"Got roll_rate = {action['roll_rate']}"
        )

    def test_riccati_lqr_responds_to_z_error(self):
        """Test Riccati-LQR controller increases thrust for +Z error."""
        from quadcopter_tracking.controllers import RiccatiLQRController

        controller = RiccatiLQRController(config={"dt": 0.01})

        obs = {
            "quadcopter": {
                "position": np.array([0.0, 0.0, 1.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
                "attitude": np.array([0.0, 0.0, 0.0]),
                "angular_velocity": np.array([0.0, 0.0, 0.0]),
            },
            "target": {
                "position": np.array([0.0, 0.0, 2.0]),  # +1m in Z
                "velocity": np.array([0.0, 0.0, 0.0]),
            },
        }

        action = controller.compute_action(obs)

        # Positive Z error should result in increased thrust
        assert action["thrust"] > controller.hover_thrust, (
            f"Riccati-LQR should produce thrust > hover for +Z error. "
            f"Got thrust = {action['thrust']}, hover = {controller.hover_thrust}"
        )

    def test_riccati_lqr_moves_quadcopter_toward_target(self):
        """Test Riccati-LQR controller reduces tracking error over time."""
        from quadcopter_tracking.controllers import RiccatiLQRController

        config = EnvConfig()
        config.simulation = SimulationParams(dt=0.01, max_episode_time=10.0)
        config.target = TargetParams(motion_type="stationary")

        env = QuadcopterEnv(config=config)
        obs = env.reset(seed=42)
        controller = RiccatiLQRController(config={"dt": 0.01})

        # Run controller and track minimum error achieved
        min_error = float("inf")
        for _ in range(500):
            action = controller.compute_action(obs)
            obs, _, done, info = env.step(action)
            min_error = min(min_error, info["tracking_error"])
            if done:
                break

        # Controller should achieve reasonable tracking at some point
        assert min_error < 0.5, f"Min tracking error too high: {min_error}"

    def test_riccati_lqr_full_episode_stationary(self):
        """Test Riccati-LQR completes a full episode with stationary target."""
        from quadcopter_tracking.controllers import RiccatiLQRController

        config = EnvConfig()
        config.simulation = SimulationParams(dt=0.01, max_episode_time=5.0)
        config.target = TargetParams(motion_type="stationary")

        env = QuadcopterEnv(config=config)
        obs = env.reset(seed=42)
        controller = RiccatiLQRController(config={"dt": 0.01})

        done = False
        min_error = float("inf")

        while not done:
            action = controller.compute_action(obs)
            obs, _, done, info = env.step(action)
            min_error = min(min_error, info["tracking_error"])

        # Controller should achieve some tracking during episode
        assert min_error < 1.0, f"Min tracking error too high: {min_error}"

    def test_riccati_lqr_linear_tracking(self):
        """Test Riccati-LQR tracks linear motion without diverging."""
        from quadcopter_tracking.controllers import RiccatiLQRController

        config = EnvConfig()
        config.simulation = SimulationParams(dt=0.01, max_episode_time=5.0)
        config.target = TargetParams(motion_type="linear", speed=0.5)

        env = QuadcopterEnv(config=config)
        obs = env.reset(seed=42)
        controller = RiccatiLQRController(config={"dt": 0.01})

        errors = []
        for _ in range(200):
            action = controller.compute_action(obs)
            obs, _, done, info = env.step(action)
            errors.append(info["tracking_error"])
            if done:
                break

        # Error should stay bounded (not diverge to infinity)
        assert max(errors) < 50.0, f"Max error too high: {max(errors)}"
        # Controller should achieve some tracking at some point
        assert min(errors) < 2.0, f"Min error too high: {min(errors)}"

    def test_riccati_lqr_circular_tracking(self):
        """Test Riccati-LQR tracks circular motion without diverging."""
        from quadcopter_tracking.controllers import RiccatiLQRController

        config = EnvConfig()
        config.simulation = SimulationParams(dt=0.01, max_episode_time=5.0)
        config.target = TargetParams(motion_type="circular", radius=1.0, speed=0.5)

        env = QuadcopterEnv(config=config)
        obs = env.reset(seed=42)
        controller = RiccatiLQRController(config={"dt": 0.01})

        errors = []
        for _ in range(200):
            action = controller.compute_action(obs)
            obs, _, done, info = env.step(action)
            errors.append(info["tracking_error"])
            if done:
                break

        # Error should stay bounded (not diverge to infinity)
        assert max(errors) < 50.0, f"Max error too high: {max(errors)}"
        # Controller should achieve some tracking at some point
        assert min(errors) < 3.0, f"Min error too high: {min(errors)}"

    def test_riccati_lqr_observation_validation(self):
        """Test Riccati-LQR controller raises error on invalid observation."""
        from quadcopter_tracking.controllers import RiccatiLQRController

        controller = RiccatiLQRController(config={"dt": 0.01})

        # Missing quadcopter key
        with pytest.raises(KeyError, match="quadcopter"):
            controller.compute_action({"target": {}})

        # Missing target key
        with pytest.raises(KeyError, match="target"):
            controller.compute_action({"quadcopter": {}})

    def test_riccati_lqr_stateless(self):
        """Test Riccati-LQR controller is stateless (reset is no-op)."""
        from quadcopter_tracking.controllers import RiccatiLQRController

        controller = RiccatiLQRController(config={"dt": 0.01})
        K_before = controller.K.copy()
        controller.reset()
        assert np.allclose(controller.K, K_before)

    def test_riccati_lqr_diagnostics(self):
        """Test Riccati-LQR logs control components for diagnostics."""
        from quadcopter_tracking.controllers import RiccatiLQRController

        controller = RiccatiLQRController(config={"dt": 0.01})

        obs = {
            "quadcopter": {
                "position": np.array([0.0, 0.0, 1.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
                "attitude": np.array([0.0, 0.0, 0.0]),
                "angular_velocity": np.array([0.0, 0.0, 0.0]),
            },
            "target": {
                "position": np.array([1.0, 0.0, 1.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
            },
        }

        controller.compute_action(obs)
        components = controller.get_control_components()

        assert components is not None
        assert "state_error" in components
        assert "feedback_u" in components
        assert "K_matrix" in components

        # State error should have the X position error
        assert components["state_error"][0] > 0  # X error

    def test_riccati_lqr_reset_clears_diagnostics(self):
        """Test Riccati-LQR reset clears control component diagnostics."""
        from quadcopter_tracking.controllers import RiccatiLQRController

        controller = RiccatiLQRController(config={"dt": 0.01})

        obs = create_hover_observation()
        controller.compute_action(obs)
        assert controller.get_control_components() is not None

        controller.reset()
        assert controller.get_control_components() is None


class TestRiccatiLQRValidation:
    """Tests for Riccati-LQR matrix validation and error handling."""

    def test_riccati_lqr_invalid_q_matrix_shape(self):
        """Test error raised for invalid Q matrix shape."""
        from quadcopter_tracking.controllers import RiccatiLQRController

        config = {
            "dt": 0.01,
            "Q": [[1, 2], [3, 4]],  # Wrong shape (2x2 instead of 6x6)
        }
        with pytest.raises(ValueError, match="Q matrix must have shape"):
            RiccatiLQRController(config=config)

    def test_riccati_lqr_invalid_r_matrix_shape(self):
        """Test error raised for invalid R matrix shape."""
        from quadcopter_tracking.controllers import RiccatiLQRController

        config = {
            "dt": 0.01,
            "R": [[1, 2], [3, 4]],  # Wrong shape (2x2 instead of 4x4)
        }
        with pytest.raises(ValueError, match="R matrix must have shape"):
            RiccatiLQRController(config=config)

    def test_riccati_lqr_non_positive_definite_r(self):
        """Test error raised for non-positive-definite R matrix."""
        from quadcopter_tracking.controllers import RiccatiLQRController

        # R with a zero eigenvalue (not positive definite)
        R_bad = np.diag([1.0, 1.0, 0.0, 1.0])

        config = {
            "dt": 0.01,
            "R": R_bad.tolist(),
            "fallback_on_failure": False,
        }
        with pytest.raises(ValueError, match="R matrix must be positive definite"):
            RiccatiLQRController(config=config)

    def test_riccati_lqr_fallback_on_invalid_matrices(self):
        """Test fallback to heuristic LQR on solver failure."""
        from quadcopter_tracking.controllers import RiccatiLQRController

        # R with near-zero eigenvalue that will cause solver issues
        R_bad = np.diag([1.0, 1.0, 1e-15, 1.0])

        config = {
            "dt": 0.01,
            "R": R_bad.tolist(),
            "fallback_on_failure": True,
        }

        # Should not raise, but should fall back
        controller = RiccatiLQRController(config=config)
        assert controller.is_using_fallback()

        # Fallback should still produce valid actions
        obs = create_hover_observation()
        action = controller.compute_action(obs)
        assert "thrust" in action

    def test_riccati_lqr_q_pos_q_vel_validation(self):
        """Test error for invalid q_pos/q_vel lengths."""
        from quadcopter_tracking.controllers import RiccatiLQRController

        # Invalid q_pos length
        with pytest.raises(ValueError, match="q_pos must have 3 elements"):
            RiccatiLQRController(config={"dt": 0.01, "q_pos": [1.0, 2.0]})

        # Invalid q_vel length
        with pytest.raises(ValueError, match="q_vel must have 3 elements"):
            RiccatiLQRController(config={"dt": 0.01, "q_vel": [1.0]})

    def test_riccati_lqr_r_controls_validation(self):
        """Test error for invalid r_controls length."""
        from quadcopter_tracking.controllers import RiccatiLQRController

        with pytest.raises(ValueError, match="r_controls must have 4 elements"):
            RiccatiLQRController(config={"dt": 0.01, "r_controls": [1.0, 2.0, 3.0]})


class TestRiccatiLQRIntegration:
    """Integration tests for Riccati-LQR with the training/eval pipeline."""

    def test_riccati_lqr_train_integration(self, tmp_path):
        """Test Riccati-LQR works with the training pipeline."""
        from quadcopter_tracking.train import Trainer, TrainingConfig

        config = TrainingConfig(
            controller="riccati_lqr",
            epochs=2,
            episodes_per_epoch=2,
            max_steps_per_episode=50,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            log_dir=str(tmp_path / "logs"),
            device="cpu",
        )

        # Riccati-LQR needs dt in the config
        config.full_config = {"riccati_lqr": {"dt": 0.01}}

        trainer = Trainer(config)

        assert trainer.controller.name == "riccati_lqr"
        assert not trainer.is_deep_controller
        assert trainer.optimizer is None

        # Run evaluation (not training)
        summary = trainer.train()

        assert summary["controller"] == "riccati_lqr"
        assert summary["epochs_completed"] == 2

    def test_riccati_lqr_eval_integration(self):
        """Test Riccati-LQR works with the evaluation pipeline."""
        from quadcopter_tracking.eval import Evaluator, load_controller

        controller = load_controller(
            controller_type="riccati_lqr",
            config={"dt": 0.01},
        )

        assert controller.name == "riccati_lqr"

        env_config = EnvConfig()
        env_config.target.motion_type = "stationary"
        env_config.simulation.max_episode_time = 2.0

        evaluator = Evaluator(
            controller=controller,
            env_config=env_config,
        )

        summary = evaluator.evaluate(
            num_episodes=2,
            base_seed=42,
            verbose=False,
        )

        assert summary.total_episodes == 2

    def test_riccati_lqr_as_supervisor(self, tmp_path):
        """Test Riccati-LQR can be used as imitation learning supervisor."""
        from quadcopter_tracking.train import Trainer, TrainingConfig

        config_dict = {
            "controller": "deep",
            "training_mode": "imitation",
            "supervisor_controller": "riccati_lqr",
            "epochs": 2,
            "episodes_per_epoch": 2,
            "max_steps_per_episode": 50,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "log_dir": str(tmp_path / "logs"),
            "device": "cpu",
            "riccati_lqr": {"dt": 0.01},
        }

        config = TrainingConfig.from_dict(config_dict)
        trainer = Trainer(config)

        assert trainer.supervisor is not None
        assert trainer.supervisor.name == "riccati_lqr"

        # Should train without errors
        summary = trainer.train()
        assert summary["epochs_completed"] == 2


class TestRiccatiLQRComparison:
    """Tests comparing Riccati-LQR with heuristic LQR."""

    def test_riccati_lqr_different_from_heuristic_lqr(self):
        """Test that Riccati-LQR produces different gains than heuristic LQR."""
        from quadcopter_tracking.controllers import LQRController, RiccatiLQRController

        # Use same cost weights for both
        common_config = {
            "q_pos": [1.0, 1.0, 10.0],
            "q_vel": [0.1, 0.1, 1.0],
            "r_thrust": 1.0,
            "r_rate": 1.0,
        }

        heuristic = LQRController(config=common_config)
        riccati = RiccatiLQRController(config={**common_config, "dt": 0.01})

        # Gains should be different (Riccati is mathematically optimal)
        assert not np.allclose(heuristic.K, riccati.K, rtol=0.1), (
            "Riccati-LQR and heuristic LQR should produce different K matrices"
        )

    def test_riccati_lqr_hover_thrust_integration(self):
        """Test Riccati-LQR hover thrust via environment integration."""
        from quadcopter_tracking.controllers import RiccatiLQRController

        # Setup environment with stationary target
        env_config = create_hover_env_config()
        env = QuadcopterEnv(config=env_config)
        env.reset(seed=42)

        # Position quadcopter exactly at target (zero error state)
        target_state = env.target.get_state(0.0)
        target_pos = target_state["position"]
        state = env.get_state_vector()
        state[0:3] = target_pos
        state[3:6] = [0.0, 0.0, 0.0]
        env.set_state_vector(state)
        obs = env.render()

        controller = RiccatiLQRController(
            config={
                "dt": 0.01,
                "mass": env_config.quadcopter.mass,
                "gravity": env_config.quadcopter.gravity,
            }
        )

        action = controller.compute_action(obs)
        expected_thrust = env_config.quadcopter.mass * env_config.quadcopter.gravity

        # Verify thrust within 0.5N tolerance
        thrust_error = abs(action["thrust"] - expected_thrust)
        assert thrust_error <= 0.5, (
            f"Riccati-LQR hover thrust error {thrust_error:.3f}N exceeds tolerance. "
            f"Expected {expected_thrust:.2f}N, got {action['thrust']:.2f}N"
        )

        # Verify no unintended torques
        assert abs(action["roll_rate"]) < 0.01
        assert abs(action["pitch_rate"]) < 0.01
        assert abs(action["yaw_rate"]) < 0.01


class TestENUCoordinateFrame:
    """
    Tests for ENU (East-North-Up) coordinate frame utilities and assertions.

    These tests verify that the coordinate frame module correctly:
    1. Defines ENU constants and conventions
    2. Validates gravity direction (should be -Z in ENU)
    3. Validates thrust direction (should be +Z in body frame)
    4. Checks control sign conventions match ENU
    5. Validates observation frame consistency

    These tests serve as regression guards to prevent coordinate frame
    mismatches that would cause controllers to move quadcopters in wrong
    directions.
    """

    def test_enu_frame_constants(self):
        """Test ENU frame constants are correctly defined."""
        from quadcopter_tracking.utils.coordinate_frame import (
            AXIS_X,
            AXIS_Y,
            AXIS_Z,
            GRAVITY_DIRECTION_ENU,
            PITCH_RATE_TO_X_VEL_SIGN,
            ROLL_RATE_TO_Y_VEL_SIGN,
            THRUST_DIRECTION_BODY,
            THRUST_TO_Z_ACCEL_SIGN,
        )

        # Verify axis indices
        assert AXIS_X == 0, "X-axis should be index 0"
        assert AXIS_Y == 1, "Y-axis should be index 1"
        assert AXIS_Z == 2, "Z-axis should be index 2"

        # Verify gravity direction (should point in -Z for ENU)
        assert np.allclose(GRAVITY_DIRECTION_ENU, [0, 0, -1]), (
            "Gravity should point in -Z direction for ENU frame"
        )

        # Verify thrust direction in body frame (should point in +Z)
        assert np.allclose(THRUST_DIRECTION_BODY, [0, 0, 1]), (
            "Thrust in body frame should point in +Z direction"
        )

        # Verify control-to-motion sign mappings
        assert PITCH_RATE_TO_X_VEL_SIGN == +1.0, (
            "+pitch_rate should produce +X velocity in ENU"
        )
        assert ROLL_RATE_TO_Y_VEL_SIGN == -1.0, (
            "+roll_rate should produce -Y velocity in ENU"
        )
        assert THRUST_TO_Z_ACCEL_SIGN == +1.0, (
            "+thrust should produce +Z acceleration in ENU"
        )

    def test_enu_frame_descriptor(self):
        """Test ENU frame descriptor properties."""
        from quadcopter_tracking.utils.coordinate_frame import (
            ENU_FRAME,
            get_current_frame,
        )

        frame = get_current_frame()
        assert frame is ENU_FRAME

        # Verify ENU properties
        assert frame.name == "ENU"
        assert frame.x_direction == "East"
        assert frame.y_direction == "North"
        assert frame.z_direction == "Up"
        assert frame.gravity_axis == "z"
        assert frame.gravity_sign == "-"

    def test_assert_gravity_direction_enu_valid(self):
        """Test that correct gravity direction passes assertion."""
        from quadcopter_tracking.utils.coordinate_frame import (
            assert_gravity_direction_enu,
        )

        # Standard gravity in ENU (pointing down)
        gravity = np.array([0.0, 0.0, -9.81])
        assert_gravity_direction_enu(gravity)  # Should not raise

        # Scaled gravity still pointing in -Z
        gravity_scaled = np.array([0.0, 0.0, -1.0])
        assert_gravity_direction_enu(gravity_scaled)  # Should not raise

    def test_assert_gravity_direction_enu_invalid(self):
        """Test that wrong gravity direction raises ENUFrameError."""
        from quadcopter_tracking.utils.coordinate_frame import (
            ENUFrameError,
            assert_gravity_direction_enu,
        )

        # Gravity pointing in +Z (wrong for ENU, would be NED style)
        gravity_up = np.array([0.0, 0.0, 9.81])
        with pytest.raises(ENUFrameError, match="should point in -Z"):
            assert_gravity_direction_enu(gravity_up)

        # Gravity pointing in X direction (completely wrong)
        gravity_x = np.array([9.81, 0.0, 0.0])
        with pytest.raises(ENUFrameError, match="should point in -Z"):
            assert_gravity_direction_enu(gravity_x)

    def test_assert_gravity_direction_zero_raises(self):
        """Test that zero gravity vector raises ENUFrameError."""
        from quadcopter_tracking.utils.coordinate_frame import (
            ENUFrameError,
            assert_gravity_direction_enu,
        )

        gravity_zero = np.array([0.0, 0.0, 0.0])
        with pytest.raises(ENUFrameError, match="zero magnitude"):
            assert_gravity_direction_enu(gravity_zero)

    def test_assert_thrust_direction_enu_valid(self):
        """Test that correct thrust direction passes assertion."""
        from quadcopter_tracking.utils.coordinate_frame import (
            assert_thrust_direction_enu,
        )

        # Thrust pointing up in body frame
        thrust = np.array([0.0, 0.0, 10.0])
        assert_thrust_direction_enu(thrust)  # Should not raise

        # Zero thrust is acceptable
        thrust_zero = np.array([0.0, 0.0, 0.0])
        assert_thrust_direction_enu(thrust_zero)  # Should not raise

    def test_assert_thrust_direction_enu_invalid(self):
        """Test that wrong thrust direction raises ENUFrameError."""
        from quadcopter_tracking.utils.coordinate_frame import (
            ENUFrameError,
            assert_thrust_direction_enu,
        )

        # Thrust pointing down (would be inverted quadcopter)
        thrust_down = np.array([0.0, 0.0, -10.0])
        with pytest.raises(ENUFrameError, match="should point in \\+Z"):
            assert_thrust_direction_enu(thrust_down)

    def test_assert_control_signs_enu_positive_x_error(self):
        """Test control sign check for positive X error."""
        from quadcopter_tracking.utils.coordinate_frame import (
            ENUFrameError,
            assert_control_signs_enu,
        )

        # +X error should produce +pitch_rate
        pos_error = np.array([1.0, 0.0, 0.0])  # Target is ahead in X
        pitch_rate = 0.5  # Correct: positive
        roll_rate = 0.0

        # Should not raise
        assert_control_signs_enu(pos_error, pitch_rate, roll_rate)

        # Wrong sign should raise
        pitch_rate_wrong = -0.5  # Incorrect: negative for +X error
        with pytest.raises(ENUFrameError, match="\\+X error"):
            assert_control_signs_enu(pos_error, pitch_rate_wrong, roll_rate)

    def test_assert_control_signs_enu_positive_y_error(self):
        """Test control sign check for positive Y error."""
        from quadcopter_tracking.utils.coordinate_frame import (
            ENUFrameError,
            assert_control_signs_enu,
        )

        # +Y error should produce -roll_rate
        pos_error = np.array([0.0, 1.0, 0.0])  # Target is ahead in Y
        pitch_rate = 0.0
        roll_rate = -0.5  # Correct: negative

        # Should not raise
        assert_control_signs_enu(pos_error, pitch_rate, roll_rate)

        # Wrong sign should raise
        roll_rate_wrong = 0.5  # Incorrect: positive for +Y error
        with pytest.raises(ENUFrameError, match="\\+Y error"):
            assert_control_signs_enu(pos_error, pitch_rate, roll_rate_wrong)

    def test_assert_control_signs_enu_small_errors_ignored(self):
        """Test that small errors (< 0.5m) are not checked."""
        from quadcopter_tracking.utils.coordinate_frame import (
            assert_control_signs_enu,
        )

        # Small error: should not trigger assertion even with "wrong" signs
        pos_error = np.array([0.3, 0.3, 0.0])  # Below threshold
        pitch_rate = -0.5  # Would be "wrong" for +X error
        roll_rate = 0.5  # Would be "wrong" for +Y error

        # Should not raise because errors are below threshold
        assert_control_signs_enu(pos_error, pitch_rate, roll_rate)

    def test_assert_z_up_valid(self):
        """Test that valid altitudes pass assertion."""
        from quadcopter_tracking.utils.coordinate_frame import assert_z_up

        # Normal flying altitude
        position = np.array([0.0, 0.0, 10.0])
        assert_z_up(position)  # Should not raise

        # Ground level
        position_ground = np.array([0.0, 0.0, 0.0])
        assert_z_up(position_ground)  # Should not raise

    def test_assert_z_up_invalid(self):
        """Test that very negative altitude raises ENUFrameError."""
        from quadcopter_tracking.utils.coordinate_frame import (
            ENUFrameError,
            assert_z_up,
        )

        # Way below ground - indicates possible NED confusion
        position = np.array([0.0, 0.0, -150.0])
        with pytest.raises(ENUFrameError, match="Z-axis.*below minimum"):
            assert_z_up(position)

    def test_validate_observation_frame_valid(self):
        """Test observation frame validation with valid ENU data."""
        from quadcopter_tracking.utils.coordinate_frame import (
            validate_observation_frame,
        )

        observation = {
            "quadcopter": {
                "position": np.array([0.0, 0.0, 1.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
            },
            "target": {
                "position": np.array([1.0, 1.0, 2.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
            },
        }

        assert validate_observation_frame(observation) is True

    def test_validate_observation_frame_negative_target_z(self):
        """Test observation validation catches negative target Z."""
        from quadcopter_tracking.utils.coordinate_frame import (
            ENUFrameError,
            validate_observation_frame,
        )

        # Target at very negative altitude - suggests NED frame
        observation = {
            "quadcopter": {
                "position": np.array([0.0, 0.0, 1.0]),
            },
            "target": {
                "position": np.array([0.0, 0.0, -50.0]),
            },
        }

        with pytest.raises(ENUFrameError, match="Target Z position.*very negative"):
            validate_observation_frame(observation)

    def test_compute_expected_hover_thrust(self):
        """Test hover thrust computation."""
        from quadcopter_tracking.utils.coordinate_frame import (
            compute_expected_hover_thrust,
        )

        # Default mass and gravity
        thrust = compute_expected_hover_thrust(mass=1.0, gravity=9.81)
        assert abs(thrust - 9.81) < 0.01

        # Heavier quadcopter
        thrust_heavy = compute_expected_hover_thrust(mass=2.0, gravity=9.81)
        assert abs(thrust_heavy - 19.62) < 0.01

    def test_compute_position_error_enu(self):
        """Test position error computation follows ENU convention."""
        from quadcopter_tracking.utils.coordinate_frame import (
            compute_position_error_enu,
        )

        quad_pos = np.array([0.0, 0.0, 1.0])
        target_pos = np.array([2.0, 3.0, 4.0])

        error = compute_position_error_enu(quad_pos, target_pos)

        # Error = target - current (target is ahead)
        expected = np.array([2.0, 3.0, 3.0])
        assert np.allclose(error, expected)

    def test_environment_gravity_matches_enu(self):
        """Integration test: verify environment uses ENU gravity direction."""
        from quadcopter_tracking.utils.coordinate_frame import (
            assert_gravity_direction_enu,
        )

        config = EnvConfig()
        env = QuadcopterEnv(config=config)
        env.reset(seed=42)

        # Set quadcopter to hover with no thrust
        state = np.zeros(12)
        state[2] = 10.0  # Start at 10m altitude
        env.set_state_vector(state)

        # Apply zero thrust for several steps
        initial_vz = state[5]
        for _ in range(10):
            action = {
                "thrust": 0.0,  # No thrust
                "roll_rate": 0.0,
                "pitch_rate": 0.0,
                "yaw_rate": 0.0,
            }
            obs, _, _, _ = env.step(action)

        # Velocity should become negative (falling in ENU means -Z velocity)
        final_vz = obs["quadcopter"]["velocity"][2]
        assert final_vz < initial_vz, (
            "Without thrust, velocity should become more negative (falling down). "
            f"Initial Vz={initial_vz:.4f}, Final Vz={final_vz:.4f}. "
            "This indicates gravity is not in -Z direction as expected for ENU."
        )

        # Verify the gravity constant itself
        gravity_vec = np.array([0.0, 0.0, -config.quadcopter.gravity])
        assert_gravity_direction_enu(gravity_vec)  # Should not raise

    def test_pid_controller_uses_enu_signs(self):
        """Integration test: PID controller matches ENU sign conventions."""
        from quadcopter_tracking.controllers import PIDController
        from quadcopter_tracking.utils.coordinate_frame import (
            PITCH_RATE_TO_X_VEL_SIGN,
            ROLL_RATE_TO_Y_VEL_SIGN,
        )

        controller = PIDController()

        # Create observation with X and Y errors
        obs = {
            "quadcopter": {
                "position": np.array([0.0, 0.0, 1.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
                "attitude": np.array([0.0, 0.0, 0.0]),
                "angular_velocity": np.array([0.0, 0.0, 0.0]),
            },
            "target": {
                "position": np.array([1.0, 1.0, 1.0]),  # +X and +Y error
                "velocity": np.array([0.0, 0.0, 0.0]),
            },
            "time": 0.0,
        }

        action = controller.compute_action(obs)

        # Verify signs match ENU convention
        # +X error → +pitch_rate
        assert (
            np.sign(action["pitch_rate"]) == PITCH_RATE_TO_X_VEL_SIGN
        ), f"PID pitch_rate sign wrong: {action['pitch_rate']}"

        # +Y error → -roll_rate (ROLL_RATE_TO_Y_VEL_SIGN is -1)
        assert (
            np.sign(action["roll_rate"]) == ROLL_RATE_TO_Y_VEL_SIGN
        ), f"PID roll_rate sign wrong: {action['roll_rate']}"

    def test_lqr_controller_uses_enu_signs(self):
        """Integration test: LQR controller matches ENU sign conventions."""
        from quadcopter_tracking.controllers import LQRController
        from quadcopter_tracking.utils.coordinate_frame import (
            PITCH_RATE_TO_X_VEL_SIGN,
            ROLL_RATE_TO_Y_VEL_SIGN,
        )

        controller = LQRController()

        # Create observation with X and Y errors
        obs = {
            "quadcopter": {
                "position": np.array([0.0, 0.0, 1.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
                "attitude": np.array([0.0, 0.0, 0.0]),
                "angular_velocity": np.array([0.0, 0.0, 0.0]),
            },
            "target": {
                "position": np.array([1.0, 1.0, 1.0]),  # +X and +Y error
                "velocity": np.array([0.0, 0.0, 0.0]),
            },
            "time": 0.0,
        }

        action = controller.compute_action(obs)

        # Verify signs match ENU convention
        # +X error → +pitch_rate
        assert (
            np.sign(action["pitch_rate"]) == PITCH_RATE_TO_X_VEL_SIGN
        ), f"LQR pitch_rate sign wrong: {action['pitch_rate']}"

        # +Y error → -roll_rate (ROLL_RATE_TO_Y_VEL_SIGN is -1)
        assert (
            np.sign(action["roll_rate"]) == ROLL_RATE_TO_Y_VEL_SIGN
        ), f"LQR roll_rate sign wrong: {action['roll_rate']}"

    def test_riccati_lqr_controller_uses_enu_signs(self):
        """Integration test: Riccati-LQR controller matches ENU sign conventions."""
        from quadcopter_tracking.controllers import RiccatiLQRController
        from quadcopter_tracking.utils.coordinate_frame import (
            PITCH_RATE_TO_X_VEL_SIGN,
            ROLL_RATE_TO_Y_VEL_SIGN,
        )

        controller = RiccatiLQRController(config={"dt": 0.01})

        # Create observation with X and Y errors
        obs = {
            "quadcopter": {
                "position": np.array([0.0, 0.0, 1.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
                "attitude": np.array([0.0, 0.0, 0.0]),
                "angular_velocity": np.array([0.0, 0.0, 0.0]),
            },
            "target": {
                "position": np.array([1.0, 1.0, 1.0]),  # +X and +Y error
                "velocity": np.array([0.0, 0.0, 0.0]),
            },
            "time": 0.0,
        }

        action = controller.compute_action(obs)

        # Verify signs match ENU convention
        # +X error → +pitch_rate
        assert (
            np.sign(action["pitch_rate"]) == PITCH_RATE_TO_X_VEL_SIGN
        ), f"Riccati-LQR pitch_rate sign wrong: {action['pitch_rate']}"

        # +Y error → -roll_rate (ROLL_RATE_TO_Y_VEL_SIGN is -1)
        assert (
            np.sign(action["roll_rate"]) == ROLL_RATE_TO_Y_VEL_SIGN
        ), f"Riccati-LQR roll_rate sign wrong: {action['roll_rate']}"

    def test_target_motion_uses_enu_z_up(self):
        """Test that target motion generators use ENU Z-up convention."""
        from quadcopter_tracking.utils.coordinate_frame import AXIS_Z

        # All motion types should have positive Z (above ground)
        motion_types = ["stationary", "linear", "circular", "sinusoidal"]

        for motion_type in motion_types:
            params = TargetParams(
                motion_type=motion_type,
                center=(0.0, 0.0, 5.0),  # 5m altitude
            )
            motion = TargetMotion(params=params, seed=42)
            motion.reset()

            # Sample several time points
            for t in [0.0, 1.0, 5.0, 10.0]:
                state = motion.get_state(t)
                z_pos = state["position"][AXIS_Z]

                # Z should remain positive (above ground in ENU)
                assert z_pos > 0, (
                    f"{motion_type} motion has negative Z ({z_pos:.2f}m) at t={t}. "
                    "This may indicate NED convention instead of ENU."
                )
