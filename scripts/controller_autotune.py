#!/usr/bin/env python3
"""
Controller Auto-Tuning CLI Script

This script provides a command-line interface for auto-tuning PID and LQR
controller gains. It supports grid search and random search strategies,
deterministic seeding for reproducibility, and can resume from interrupted runs.

Usage Examples:
    # Basic random search for PID gains
    python scripts/controller_autotune.py --controller pid --max-iterations 50

    # Grid search for PID gains with custom ranges
    python scripts/controller_autotune.py --controller pid --strategy grid \\
        --kp-range 0.005,0.005,2.0 0.05,0.05,6.0 \\
        --kd-range 0.02,0.02,1.0 0.15,0.15,3.0

    # Tune with feedforward gains
    python scripts/controller_autotune.py --controller pid --feedforward \\
        --ff-velocity-range 0.0,0.0,0.0 0.2,0.2,0.1

    # Resume from previous run
    python scripts/controller_autotune.py --resume reports/tuning/results.json

    # LQR tuning
    python scripts/controller_autotune.py --controller lqr \\
        --q-pos-range 0.00001,0.00001,10.0 0.001,0.001,25.0

Environment Variables:
    TUNING_OUTPUT_DIR: Override default output directory for tuning results
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quadcopter_tracking.controllers.tuning import (
    ControllerTuner,
    GainSearchSpace,
    TuningConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_vector_arg(value: str) -> list[float]:
    """Parse comma-separated vector argument like '0.01,0.01,4.0'."""
    try:
        parts = value.split(",")
        if len(parts) != 3:
            raise ValueError(f"Expected 3 values for [x,y,z], got {len(parts)}")
        return [float(p.strip()) for p in parts]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid vector format: {value}. {e}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Auto-tune PID or LQR controller gains",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic PID tuning with random search
  python scripts/controller_autotune.py --controller pid --max-iterations 50

  # Grid search with custom gain ranges
  python scripts/controller_autotune.py --controller pid --strategy grid \\
      --kp-range 0.005,0.005,2.0 0.05,0.05,6.0

  # Tune LQR controller
  python scripts/controller_autotune.py --controller lqr \\
      --q-pos-range 0.00001,0.00001,10.0 0.001,0.001,25.0

  # Resume from previous tuning run
  python scripts/controller_autotune.py --resume reports/tuning/results.json
        """,
    )

    # Config file option
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML/JSON configuration file (overrides CLI args)",
    )

    # Controller selection
    parser.add_argument(
        "--controller",
        type=str,
        choices=["pid", "lqr"],
        default="pid",
        help="Controller type to tune (default: pid)",
    )

    # Search strategy
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["grid", "random"],
        default="random",
        help="Search strategy (default: random)",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum number of configurations to evaluate (default: 50)",
    )

    parser.add_argument(
        "--grid-points",
        type=int,
        default=3,
        help="Points per dimension for grid search (default: 3)",
    )

    # PID gain ranges (each takes min and max vectors)
    parser.add_argument(
        "--kp-range",
        nargs=2,
        type=parse_vector_arg,
        metavar=("MIN", "MAX"),
        help="Proportional gains range, e.g., '0.005,0.005,2.0' '0.05,0.05,6.0'",
    )

    parser.add_argument(
        "--ki-range",
        nargs=2,
        type=parse_vector_arg,
        metavar=("MIN", "MAX"),
        help="Integral gains range",
    )

    parser.add_argument(
        "--kd-range",
        nargs=2,
        type=parse_vector_arg,
        metavar=("MIN", "MAX"),
        help="Derivative gains range",
    )

    # Feedforward options
    parser.add_argument(
        "--feedforward",
        action="store_true",
        help="Enable feedforward gain tuning",
    )

    parser.add_argument(
        "--ff-velocity-range",
        nargs=2,
        type=parse_vector_arg,
        metavar=("MIN", "MAX"),
        help="Feedforward velocity gains range",
    )

    parser.add_argument(
        "--ff-acceleration-range",
        nargs=2,
        type=parse_vector_arg,
        metavar=("MIN", "MAX"),
        help="Feedforward acceleration gains range",
    )

    # LQR cost weight ranges
    parser.add_argument(
        "--q-pos-range",
        nargs=2,
        type=parse_vector_arg,
        metavar=("MIN", "MAX"),
        help="LQR position cost weights range",
    )

    parser.add_argument(
        "--q-vel-range",
        nargs=2,
        type=parse_vector_arg,
        metavar=("MIN", "MAX"),
        help="LQR velocity cost weights range",
    )

    parser.add_argument(
        "--r-thrust-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="LQR thrust cost weight range",
    )

    parser.add_argument(
        "--r-rate-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="LQR rate cost weight range",
    )

    # Evaluation parameters
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Evaluation episodes per configuration (default: 5)",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=3000,
        help="Maximum steps per evaluation episode (default: 3000)",
    )

    parser.add_argument(
        "--motion-type",
        type=str,
        choices=["stationary", "linear", "circular", "sinusoidal", "figure8"],
        default="stationary",
        help="Target motion type for evaluation (default: stationary)",
    )

    parser.add_argument(
        "--episode-length",
        type=float,
        default=30.0,
        help="Episode duration in seconds (default: 30.0)",
    )

    parser.add_argument(
        "--target-radius",
        type=float,
        default=0.5,
        help="On-target radius in meters (default: 0.5)",
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/tuning",
        help="Output directory for tuning results (default: reports/tuning)",
    )

    # Resume from previous run
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to previous tuning results to resume from",
    )

    # Verbosity
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress non-essential output",
    )

    return parser.parse_args()


def build_search_space(args: argparse.Namespace) -> GainSearchSpace:
    """Build search space from CLI arguments."""
    return GainSearchSpace(
        kp_pos_range=tuple(args.kp_range) if args.kp_range else None,
        ki_pos_range=tuple(args.ki_range) if args.ki_range else None,
        kd_pos_range=tuple(args.kd_range) if args.kd_range else None,
        ff_velocity_gain_range=(
            tuple(args.ff_velocity_range) if args.ff_velocity_range else None
        ),
        ff_acceleration_gain_range=(
            tuple(args.ff_acceleration_range) if args.ff_acceleration_range else None
        ),
        q_pos_range=tuple(args.q_pos_range) if args.q_pos_range else None,
        q_vel_range=tuple(args.q_vel_range) if args.q_vel_range else None,
        r_thrust_range=tuple(args.r_thrust_range) if args.r_thrust_range else None,
        r_rate_range=tuple(args.r_rate_range) if args.r_rate_range else None,
    )


def get_default_search_space(controller_type: str) -> GainSearchSpace:
    """Get sensible default search space for controller type."""
    if controller_type == "pid":
        return GainSearchSpace(
            # Default PID gain ranges based on validated baselines
            kp_pos_range=([0.005, 0.005, 2.0], [0.05, 0.05, 6.0]),
            kd_pos_range=([0.02, 0.02, 1.0], [0.15, 0.15, 3.0]),
            # ki defaults to not being tuned (most scenarios don't need it)
        )
    elif controller_type == "lqr":
        return GainSearchSpace(
            # Default LQR cost weight ranges
            q_pos_range=([0.00005, 0.00005, 10.0], [0.0005, 0.0005, 25.0]),
            q_vel_range=([0.001, 0.001, 2.0], [0.01, 0.01, 8.0]),
        )
    else:
        return GainSearchSpace()


def load_config_file(path: str) -> dict:
    """Load configuration from YAML or JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        if path.suffix in (".yaml", ".yml"):
            return yaml.safe_load(f) or {}
        elif path.suffix == ".json":
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Load config from file if provided
    if args.config:
        try:
            file_config = load_config_file(args.config)
            config = TuningConfig.from_dict(file_config)
            logger.info("Loaded configuration from %s", args.config)
        except Exception as e:
            logger.error("Failed to load config file: %s", e)
            return 1
    else:
        # Build search space from CLI args or use defaults
        search_space = build_search_space(args)

        # Check if any search ranges were specified
        if not search_space.get_active_parameters():
            logger.info(
                "No search ranges specified, using defaults for %s controller",
                args.controller,
            )
            search_space = get_default_search_space(args.controller)

        # Validate search space
        try:
            search_space.validate()
        except ValueError as e:
            logger.error("Invalid search space: %s", e)
            return 1

        # Build tuning config
        config = TuningConfig(
            controller_type=args.controller,
            search_space=search_space,
            strategy=args.strategy,
            max_iterations=args.max_iterations,
            grid_points_per_dim=args.grid_points,
            evaluation_episodes=args.episodes,
            evaluation_horizon=args.max_steps,
            seed=args.seed,
            target_motion_type=args.motion_type,
            episode_length=args.episode_length,
            target_radius=args.target_radius,
            output_dir=args.output_dir,
            resume_from=args.resume,
            feedforward_enabled=args.feedforward,
        )

    # Create and run tuner
    try:
        tuner = ControllerTuner(config)
        result = tuner.tune()

        # Print summary
        if not args.quiet:
            print("\n" + "=" * 60)
            print("AUTO-TUNING COMPLETE")
            print("=" * 60)
            print(f"Controller: {config.controller_type.upper()}")
            print(f"Strategy: {config.strategy}")
            print(f"Iterations: {result.iterations_completed}")
            print(f"Interrupted: {result.interrupted}")
            print(f"\nBest Score: {result.best_score:.4f}")
            on_target = result.best_metrics.get("mean_on_target_ratio", 0)
            print(f"Best On-Target: {on_target * 100:.1f}%")
            error = result.best_metrics.get("mean_tracking_error", 0)
            print(f"Best Tracking Error: {error:.3f}m")
            print("\nBest Configuration:")
            for key, value in result.best_config.items():
                if isinstance(value, list):
                    formatted = "[" + ", ".join(f"{v:.4f}" for v in value) + "]"
                elif isinstance(value, float):
                    formatted = f"{value:.4f}"
                else:
                    formatted = str(value)
                print(f"  {key}: {formatted}")
            print(f"\nResults saved to: {config.output_dir}")
            print("=" * 60)

        return 0 if not result.interrupted else 1

    except ValueError as e:
        logger.error("Configuration error: %s", e)
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 1
    except Exception as e:
        logger.exception("Unexpected error during tuning: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
