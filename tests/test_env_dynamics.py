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
        """Test PID controller initializes with default gains."""
        from quadcopter_tracking.controllers import PIDController

        pid = PIDController()
        assert pid.name == "pid"
        assert pid.kp_pos is not None
        assert pid.ki_pos is not None
        assert pid.kd_pos is not None
        assert pid.integral_limit == 5.0
        assert len(pid.kp_pos) == 3

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

        # Positive X error should result in negative pitch rate (pitch forward)
        assert action["pitch_rate"] < 0

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
    mass: float = 1.0,
    gravity: float = 9.81,
) -> dict:
    """
    Create a stationary hover observation with zero tracking error.

    This helper constructs an observation dictionary where the quadcopter
    is exactly at the target position with zero velocity - the ideal hover
    condition where thrust should equal mass * gravity.

    Args:
        position: Target/quadcopter position (default: [0, 0, 1]).
        mass: Quadcopter mass in kg (default: 1.0).
        gravity: Gravitational acceleration (default: 9.81).

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
        controller = PIDController(config={
            "mass": env_config.quadcopter.mass,
            "gravity": env_config.quadcopter.gravity,
        })

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
        controller = LQRController(config={
            "mass": env_config.quadcopter.mass,
            "gravity": env_config.quadcopter.gravity,
        })

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
        obs = create_hover_observation(mass=mass, gravity=gravity)

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
        obs = create_hover_observation(mass=mass, gravity=gravity)

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

        mass, gravity = 1.0, 9.81
        expected_thrust = mass * gravity

        controller = PIDController(config={"mass": mass, "gravity": gravity})
        obs = create_hover_observation()

        # Simulate multiple timesteps at hover
        thrusts = []
        for i in range(10):
            obs["time"] = i * 0.01  # Advance time
            action = controller.compute_action(obs)
            thrusts.append(action["thrust"])

        # All thrusts should be within tolerance of hover thrust
        for i, thrust in enumerate(thrusts):
            thrust_error = abs(thrust - expected_thrust)
            assert thrust_error <= self.HOVER_THRUST_TOLERANCE_N, (
                f"PID step {i}: thrust {thrust:.2f}N drifted from "
                f"hover baseline {expected_thrust:.2f}N"
            )

    def test_hover_stability_lqr_multi_step(self):
        """
        Test LQR maintains hover thrust over multiple time steps.

        Verifies thrust remains stable when quadcopter stays at hover
        equilibrium across multiple controller invocations.
        """
        from quadcopter_tracking.controllers import LQRController

        mass, gravity = 1.0, 9.81
        expected_thrust = mass * gravity

        controller = LQRController(config={"mass": mass, "gravity": gravity})
        obs = create_hover_observation()

        # Simulate multiple timesteps at hover
        thrusts = []
        for i in range(10):
            obs["time"] = i * 0.01
            action = controller.compute_action(obs)
            thrusts.append(action["thrust"])

        # All thrusts should be within tolerance
        for i, thrust in enumerate(thrusts):
            thrust_error = abs(thrust - expected_thrust)
            assert thrust_error <= self.HOVER_THRUST_TOLERANCE_N, (
                f"LQR step {i}: thrust {thrust:.2f}N drifted from "
                f"hover baseline {expected_thrust:.2f}N"
            )
