"""
Environment Configuration Module

Defines physical parameters, simulation settings, and constraint limits
for the quadcopter dynamics environment.
"""

from dataclasses import dataclass, field


@dataclass
class QuadcopterParams:
    """Physical parameters of the quadcopter."""

    mass: float = 1.0  # kg
    arm_length: float = 0.25  # m, distance from center to rotor

    # Inertia tensor (diagonal, kg*m^2)
    Ixx: float = 0.0082
    Iyy: float = 0.0082
    Izz: float = 0.0149

    # Thrust/torque coefficients
    k_thrust: float = 1.0e-5  # thrust coefficient
    k_torque: float = 1.0e-7  # torque coefficient

    # Actuator limits
    max_thrust: float = 20.0  # N total
    min_thrust: float = 0.0  # N (no reverse thrust)
    max_angular_rate: float = 3.0  # rad/s for roll/pitch/yaw rates

    # Gravity
    gravity: float = 9.81  # m/s^2

    # Drag coefficients
    drag_coeff_linear: float = 0.1  # linear drag
    drag_coeff_angular: float = 0.01  # angular drag


@dataclass
class SimulationParams:
    """Simulation parameters."""

    dt: float = 0.01  # timestep in seconds
    max_episode_time: float = 30.0  # maximum episode duration in seconds
    integrator: str = "rk4"  # integration method: 'euler' or 'rk4'

    # Numerical stability
    max_velocity: float = 50.0  # m/s, clip velocities beyond this
    max_angular_velocity: float = 10.0  # rad/s, clip angular velocities
    max_position: float = 1000.0  # m, terminate if position exceeds


@dataclass
class TargetParams:
    """Target motion parameters."""

    motion_type: str = "linear"  # 'linear', 'circular', 'sinusoidal', 'figure8'
    speed: float = 1.0  # m/s, base speed
    amplitude: float = 2.0  # m, amplitude for oscillatory motion
    frequency: float = 0.5  # Hz, frequency for oscillatory motion
    radius: float = 2.0  # m, radius for circular motion
    center: tuple[float, float, float] = (0.0, 0.0, 1.0)  # center position
    max_acceleration: float = 5.0  # m/s^2, enforce smooth motion
    radius_requirement: float = 0.5  # m, on-target threshold


@dataclass
class SuccessCriteria:
    """Success criteria for tracking evaluation."""

    min_on_target_ratio: float = 0.8  # 80% on-target requirement
    min_episode_duration: float = 30.0  # seconds
    target_radius: float = 0.5  # m, on-target threshold


@dataclass
class LoggingParams:
    """Logging configuration."""

    enabled: bool = True
    log_interval: int = 10  # steps between log entries
    output_dir: str = "experiments"


@dataclass
class EnvConfig:
    """Complete environment configuration."""

    seed: int = 42
    quadcopter: QuadcopterParams = field(default_factory=QuadcopterParams)
    simulation: SimulationParams = field(default_factory=SimulationParams)
    target: TargetParams = field(default_factory=TargetParams)
    success_criteria: SuccessCriteria = field(default_factory=SuccessCriteria)
    logging: LoggingParams = field(default_factory=LoggingParams)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "EnvConfig":
        """
        Create EnvConfig from a dictionary (e.g., from load_config).

        Args:
            config_dict: Configuration dictionary.

        Returns:
            EnvConfig instance.
        """
        quad_dict = config_dict.get("quadcopter", {})
        sim_dict = config_dict.get("simulation", {}).copy()
        target_dict = config_dict.get("target", {})
        success_dict = config_dict.get("success_criteria", {}).copy()
        logging_dict = config_dict.get("logging", {})

        # Handle legacy flat config structure
        if "dt" in config_dict and "dt" not in sim_dict:
            sim_dict["dt"] = config_dict["dt"]
        if "episode_length" in config_dict:
            sim_dict["max_episode_time"] = config_dict["episode_length"]

        # Map target.radius_requirement to success_criteria.target_radius
        # These refer to the same concept: the distance threshold for "on-target".
        # target.radius_requirement is used in TargetParams for motion generation,
        # while success_criteria.target_radius is used for episode evaluation.
        # Default both from the same source for consistency.
        if "radius_requirement" in target_dict:
            success_dict.setdefault("target_radius", target_dict["radius_requirement"])

        return cls(
            seed=config_dict.get("seed", 42),
            quadcopter=QuadcopterParams(
                mass=quad_dict.get("mass", 1.0),
                arm_length=quad_dict.get("arm_length", 0.25),
                Ixx=quad_dict.get("Ixx", 0.0082),
                Iyy=quad_dict.get("Iyy", 0.0082),
                Izz=quad_dict.get("Izz", 0.0149),
                k_thrust=quad_dict.get("k_thrust", 1.0e-5),
                k_torque=quad_dict.get("k_torque", 1.0e-7),
                max_thrust=quad_dict.get("max_thrust", 20.0),
                min_thrust=quad_dict.get("min_thrust", 0.0),
                max_angular_rate=quad_dict.get("max_angular_rate", 3.0),
                gravity=quad_dict.get("gravity", 9.81),
                drag_coeff_linear=quad_dict.get("drag_coeff_linear", 0.1),
                drag_coeff_angular=quad_dict.get("drag_coeff_angular", 0.01),
            ),
            simulation=SimulationParams(
                dt=sim_dict.get("dt", 0.01),
                max_episode_time=sim_dict.get("max_episode_time", 30.0),
                integrator=sim_dict.get("integrator", "rk4"),
                max_velocity=sim_dict.get("max_velocity", 50.0),
                max_angular_velocity=sim_dict.get("max_angular_velocity", 10.0),
                max_position=sim_dict.get("max_position", 1000.0),
            ),
            target=TargetParams(
                motion_type=target_dict.get("motion_type", "linear"),
                speed=target_dict.get("speed", 1.0),
                amplitude=target_dict.get("amplitude", 2.0),
                frequency=target_dict.get("frequency", 0.5),
                radius=target_dict.get("radius", 2.0),
                center=tuple(target_dict.get("center", (0.0, 0.0, 1.0))),
                max_acceleration=target_dict.get("max_acceleration", 5.0),
                radius_requirement=target_dict.get("radius_requirement", 0.5),
            ),
            success_criteria=SuccessCriteria(
                min_on_target_ratio=success_dict.get("min_on_target_ratio", 0.8),
                min_episode_duration=success_dict.get("min_episode_duration", 30.0),
                target_radius=success_dict.get("target_radius", 0.5),
            ),
            logging=LoggingParams(
                enabled=logging_dict.get("enabled", True),
                log_interval=logging_dict.get("log_interval", 10),
                output_dir=logging_dict.get("output_dir", "experiments"),
            ),
        )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "seed": self.seed,
            "quadcopter": {
                "mass": self.quadcopter.mass,
                "arm_length": self.quadcopter.arm_length,
                "Ixx": self.quadcopter.Ixx,
                "Iyy": self.quadcopter.Iyy,
                "Izz": self.quadcopter.Izz,
                "k_thrust": self.quadcopter.k_thrust,
                "k_torque": self.quadcopter.k_torque,
                "max_thrust": self.quadcopter.max_thrust,
                "min_thrust": self.quadcopter.min_thrust,
                "max_angular_rate": self.quadcopter.max_angular_rate,
                "gravity": self.quadcopter.gravity,
                "drag_coeff_linear": self.quadcopter.drag_coeff_linear,
                "drag_coeff_angular": self.quadcopter.drag_coeff_angular,
            },
            "simulation": {
                "dt": self.simulation.dt,
                "max_episode_time": self.simulation.max_episode_time,
                "integrator": self.simulation.integrator,
                "max_velocity": self.simulation.max_velocity,
                "max_angular_velocity": self.simulation.max_angular_velocity,
                "max_position": self.simulation.max_position,
            },
            "target": {
                "motion_type": self.target.motion_type,
                "speed": self.target.speed,
                "amplitude": self.target.amplitude,
                "frequency": self.target.frequency,
                "radius": self.target.radius,
                "center": self.target.center,
                "max_acceleration": self.target.max_acceleration,
                "radius_requirement": self.target.radius_requirement,
            },
            "success_criteria": {
                "min_on_target_ratio": self.success_criteria.min_on_target_ratio,
                "min_episode_duration": self.success_criteria.min_episode_duration,
                "target_radius": self.success_criteria.target_radius,
            },
            "logging": {
                "enabled": self.logging.enabled,
                "log_interval": self.logging.log_interval,
                "output_dir": self.logging.output_dir,
            },
        }
