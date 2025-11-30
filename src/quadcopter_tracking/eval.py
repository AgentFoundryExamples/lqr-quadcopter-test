#!/usr/bin/env python3
"""
Evaluation Script for Quadcopter Tracking Controllers

This script provides a complete evaluation pipeline:
- Load trained controllers from checkpoints
- Execute multiple validation episodes
- Compute success criteria (>=80% on-target over >=30s episodes)
- Generate trajectory plots and save reports
- Support hyperparameter sweeps with ranking

Usage:
    python -m quadcopter_tracking.eval --checkpoint checkpoints/best.pt
    python -m quadcopter_tracking.eval --controller lqr --episodes 10
    python -m quadcopter_tracking.eval --sweep configs/sweep.yaml
"""

import argparse
import json
import logging
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server/headless

import matplotlib.pyplot as plt
import numpy as np
import yaml

from quadcopter_tracking.controllers import (
    BaseController,
    DeepTrackingPolicy,
    LQRController,
    PIDController,
)
from quadcopter_tracking.env import EnvConfig, QuadcopterEnv
from quadcopter_tracking.utils.metrics import (
    EpisodeMetrics,
    EvaluationSummary,
    SuccessCriteria,
    compute_episode_metrics,
    compute_evaluation_summary,
    format_metrics_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class Evaluator:
    """
    Controller evaluation pipeline.

    Executes validation episodes, computes metrics, and generates reports.
    """

    def __init__(
        self,
        controller: BaseController,
        env_config: EnvConfig | None = None,
        criteria: SuccessCriteria | None = None,
        output_dir: str | Path = "reports",
    ):
        """
        Initialize evaluator.

        Args:
            controller: Controller to evaluate.
            env_config: Environment configuration.
            criteria: Success criteria configuration.
            output_dir: Directory for output reports and plots.
        """
        self.controller = controller
        self.env_config = env_config or EnvConfig()
        self.criteria = criteria or SuccessCriteria()
        self.output_dir = Path(output_dir)

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)

        # Episode data storage
        self.episode_data_list: list[list[dict]] = []
        self.episode_info_list: list[dict] = []

    def run_episode(
        self,
        seed: int,
        max_steps: int | None = None,
        progress_callback: Callable | None = None,
    ) -> tuple[list[dict], dict]:
        """
        Run a single evaluation episode.

        Args:
            seed: Random seed for reproducibility.
            max_steps: Maximum steps per episode (None for env default).
            progress_callback: Optional callback for progress reporting.

        Returns:
            Tuple of (episode_data, final_info).
        """
        env = QuadcopterEnv(config=self.env_config)
        obs = env.reset(seed=seed)

        episode_data = []
        done = False
        step = 0

        while not done:
            if max_steps is not None and step >= max_steps:
                break

            # Compute action
            try:
                action = self.controller.compute_action(obs)
            except Exception as e:
                logger.error("Controller error at step %d: %s", step, e)
                break

            # Step environment
            next_obs, reward, done, info = env.step(action)

            # Record step data
            step_data = {
                "time": info.get("time", step * env.dt),
                "step": step,
                "quadcopter_position": obs["quadcopter"]["position"].tolist(),
                "quadcopter_velocity": obs["quadcopter"]["velocity"].tolist(),
                "target_position": obs["target"]["position"].tolist(),
                "target_velocity": obs["target"]["velocity"].tolist(),
                "action": [
                    action["thrust"],
                    action["roll_rate"],
                    action["pitch_rate"],
                    action["yaw_rate"],
                ],
                "reward": reward,
                "tracking_error": info.get("tracking_error", 0.0),
                "on_target": info.get("on_target", False),
            }
            episode_data.append(step_data)

            obs = next_obs
            step += 1

            if progress_callback and step % 100 == 0:
                progress_callback(step, info)

        return episode_data, info

    def evaluate(
        self,
        num_episodes: int = 10,
        base_seed: int = 42,
        max_steps_per_episode: int | None = None,
        verbose: bool = True,
    ) -> EvaluationSummary:
        """
        Run complete evaluation.

        Args:
            num_episodes: Number of episodes to evaluate.
            base_seed: Starting seed for reproducibility.
            max_steps_per_episode: Max steps per episode.
            verbose: Whether to print progress.

        Returns:
            EvaluationSummary with all metrics.
        """
        logger.info(
            "Starting evaluation: %d episodes, controller=%s",
            num_episodes,
            self.controller.name,
        )

        self.episode_data_list = []
        self.episode_info_list = []
        episode_metrics_list: list[EpisodeMetrics] = []

        for i in range(num_episodes):
            seed = base_seed + i
            if verbose:
                logger.info("Episode %d/%d (seed=%d)...", i + 1, num_episodes, seed)

            episode_data, info = self.run_episode(
                seed=seed,
                max_steps=max_steps_per_episode,
            )

            self.episode_data_list.append(episode_data)
            self.episode_info_list.append(info)

            # Compute metrics for this episode
            metrics = compute_episode_metrics(
                episode_data, self.criteria, info
            )
            episode_metrics_list.append(metrics)

            if verbose:
                status = "✓ SUCCESS" if metrics.success else "✗ FAILED"
                logger.info(
                    "  %s | on-target: %.1f%% | error: %.3fm | duration: %.1fs",
                    status,
                    metrics.on_target_ratio * 100,
                    metrics.mean_tracking_error,
                    metrics.episode_duration,
                )

        # Compute summary
        summary = compute_evaluation_summary(episode_metrics_list, self.criteria)

        if verbose:
            print("\n" + format_metrics_report(summary))

        return summary

    def save_report(
        self,
        summary: EvaluationSummary,
        experiment_name: str | None = None,
    ) -> dict[str, Path]:
        """
        Save evaluation report to files.

        Args:
            summary: Evaluation summary to save.
            experiment_name: Name for this evaluation run.

        Returns:
            Dictionary of saved file paths.
        """
        if experiment_name is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            experiment_name = f"eval_{self.controller.name}_{timestamp}"

        saved_files = {}

        # Save JSON metrics
        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(summary.to_dict(), f, indent=2)
        saved_files["metrics"] = metrics_path
        logger.info("Saved metrics: %s", metrics_path)

        # Save text report
        report_path = self.output_dir / f"{experiment_name}_report.txt"
        with open(report_path, "w") as f:
            f.write(format_metrics_report(summary))
        saved_files["report"] = report_path
        logger.info("Saved report: %s", report_path)

        return saved_files

    def plot_trajectory(
        self,
        episode_idx: int = 0,
        save_path: str | Path | None = None,
        show_radius: bool = True,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot quadcopter vs target trajectory for an episode.

        Args:
            episode_idx: Index of episode to plot.
            save_path: Path to save plot (None to skip saving).
            show_radius: Whether to show target radius bands.

        Returns:
            Tuple of (figure, axes).
        """
        if episode_idx >= len(self.episode_data_list):
            raise ValueError(
                f"Episode {episode_idx} not found "
                f"(have {len(self.episode_data_list)} episodes)"
            )

        episode_data = self.episode_data_list[episode_idx]
        times = np.array([d["time"] for d in episode_data])
        quad_pos = np.array([d["quadcopter_position"] for d in episode_data])
        target_pos = np.array([d["target_position"] for d in episode_data])

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        labels = ["X Position", "Y Position", "Z Position"]
        colors = ["tab:blue", "tab:orange", "tab:green"]

        for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
            # Plot quadcopter and target positions
            ax.plot(times, quad_pos[:, i], label="Quadcopter", color=color, linewidth=1)
            ax.plot(
                times,
                target_pos[:, i],
                label="Target",
                color="red",
                linestyle="--",
                linewidth=1,
            )

            # Show target radius bands
            if show_radius:
                ax.fill_between(
                    times,
                    target_pos[:, i] - self.criteria.target_radius,
                    target_pos[:, i] + self.criteria.target_radius,
                    alpha=0.2,
                    color="green",
                    label="Target Radius" if i == 0 else "",
                )

            ax.set_ylabel(f"{label} (m)")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(
            f"Position Tracking - Episode {episode_idx + 1}\n"
            f"Controller: {self.controller.name}",
            fontsize=12,
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Saved trajectory plot: %s", save_path)

        return fig, axes

    def plot_tracking_error(
        self,
        episode_idx: int = 0,
        save_path: str | Path | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot tracking error over time with on-target highlighting.

        Args:
            episode_idx: Index of episode to plot.
            save_path: Path to save plot.

        Returns:
            Tuple of (figure, axes).
        """
        if episode_idx >= len(self.episode_data_list):
            raise ValueError(f"Episode {episode_idx} not found")

        episode_data = self.episode_data_list[episode_idx]
        times = np.array([d["time"] for d in episode_data])
        errors = np.array([d["tracking_error"] for d in episode_data])
        on_target = np.array([d["on_target"] for d in episode_data])

        fig, ax = plt.subplots(figsize=(12, 5))

        # Plot tracking error
        ax.plot(times, errors, "b-", linewidth=1, label="Tracking Error")

        # Highlight on-target regions
        on_target_times = times[on_target]
        on_target_errors = errors[on_target]
        ax.scatter(
            on_target_times,
            on_target_errors,
            c="green",
            s=2,
            alpha=0.5,
            label="On-Target",
        )

        # Target radius threshold
        ax.axhline(
            y=self.criteria.target_radius,
            color="r",
            linestyle="--",
            label=f"Target Radius ({self.criteria.target_radius}m)",
        )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Tracking Error (m)")
        ax.set_title(
            f"Tracking Error - Episode {episode_idx + 1} | "
            f"Controller: {self.controller.name}"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Saved tracking error plot: %s", save_path)

        return fig, ax

    def generate_all_plots(
        self,
        experiment_name: str | None = None,
    ) -> list[Path]:
        """
        Generate all evaluation plots.

        Args:
            experiment_name: Base name for plot files.

        Returns:
            List of saved plot paths.
        """
        if experiment_name is None:
            experiment_name = f"eval_{self.controller.name}"

        plots_dir = self.output_dir / "plots"
        saved_plots = []

        # Plot best and worst episodes
        if self.episode_data_list:
            # Position tracking plot (first episode by default)
            path = plots_dir / "position_tracking.png"
            self.plot_trajectory(episode_idx=0, save_path=path)
            saved_plots.append(path)
            plt.close()

            # Tracking error plot
            path = plots_dir / "tracking_error.png"
            self.plot_tracking_error(episode_idx=0, save_path=path)
            saved_plots.append(path)
            plt.close()

        return saved_plots


def load_controller(
    controller_type: str,
    checkpoint_path: str | Path | None = None,
    config: dict | None = None,
) -> BaseController:
    """
    Load a controller by type and optionally from checkpoint.

    Args:
        controller_type: Type of controller ('deep', 'lqr', 'pid').
        checkpoint_path: Path to checkpoint for neural controllers.
        config: Controller configuration dictionary.

    Returns:
        Initialized controller.
    """
    config = config or {}

    if controller_type == "deep":
        controller = DeepTrackingPolicy(config=config, device="cpu")
        if checkpoint_path:
            controller.load_checkpoint(checkpoint_path)
            logger.info("Loaded checkpoint: %s", checkpoint_path)
        return controller
    elif controller_type == "lqr":
        return LQRController(config=config)
    elif controller_type == "pid":
        return PIDController(config=config)
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")


def run_hyperparameter_sweep(
    sweep_config_path: str | Path,
    output_dir: str | Path = "reports/sweeps",
) -> list[dict]:
    """
    Run evaluation sweep over hyperparameter configurations.

    Args:
        sweep_config_path: Path to sweep configuration YAML.
        output_dir: Output directory for results.

    Returns:
        List of sweep results ranked by performance.
    """
    with open(sweep_config_path) as f:
        sweep_config = yaml.safe_load(f)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    configs = sweep_config.get("configurations", [])

    logger.info("Running sweep with %d configurations", len(configs))

    for i, config in enumerate(configs):
        logger.info("Configuration %d/%d: %s", i + 1, len(configs), config.get("name"))

        try:
            # Load controller
            controller = load_controller(
                controller_type=config.get("controller_type", "deep"),
                checkpoint_path=config.get("checkpoint_path"),
                config=config.get("controller_config"),
            )

            # Setup evaluator
            env_config = EnvConfig()
            if "env_config" in config:
                env_config = EnvConfig.from_dict(config["env_config"])

            criteria = SuccessCriteria()
            if "criteria" in config:
                crit = config["criteria"]
                criteria = SuccessCriteria(
                    min_on_target_ratio=crit.get("min_on_target_ratio", 0.8),
                    min_episode_duration=crit.get("min_episode_duration", 30.0),
                    target_radius=crit.get("target_radius", 0.5),
                )

            evaluator = Evaluator(
                controller=controller,
                env_config=env_config,
                criteria=criteria,
                output_dir=output_dir / config.get("name", f"config_{i}"),
            )

            # Run evaluation
            summary = evaluator.evaluate(
                num_episodes=config.get("num_episodes", 5),
                base_seed=config.get("seed", 42),
                verbose=False,
            )

            results.append({
                "name": config.get("name", f"config_{i}"),
                "config": config,
                "mean_on_target_ratio": summary.mean_on_target_ratio,
                "success_rate": summary.success_rate,
                "mean_tracking_error": summary.mean_tracking_error,
                "meets_criteria": summary.meets_criteria,
            })

            logger.info(
                "  Result: on-target=%.1f%%, success=%.1f%%",
                summary.mean_on_target_ratio * 100,
                summary.success_rate * 100,
            )

        except Exception as e:
            logger.error("Configuration %s failed: %s", config.get("name"), e)
            results.append({
                "name": config.get("name", f"config_{i}"),
                "config": config,
                "error": str(e),
            })

    # Rank results by on-target ratio
    valid_results = [r for r in results if "error" not in r]
    valid_results.sort(key=lambda x: x["mean_on_target_ratio"], reverse=True)

    # Save sweep results
    sweep_results_path = output_dir / "sweep_results.json"
    with open(sweep_results_path, "w") as f:
        json.dump(valid_results, f, indent=2)
    logger.info("Saved sweep results: %s", sweep_results_path)

    # Print ranking
    print("\n" + "=" * 60)
    print("HYPERPARAMETER SWEEP RESULTS (Ranked)")
    print("=" * 60)
    for i, r in enumerate(valid_results):
        print(
            f"{i + 1}. {r['name']}: on-target={r['mean_on_target_ratio']:.1%}, "
            f"success={r['success_rate']:.1%}"
        )
    print("=" * 60)

    return valid_results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate quadcopter tracking controllers"
    )

    # Controller options
    parser.add_argument(
        "--controller",
        type=str,
        default="deep",
        choices=["deep", "lqr", "pid"],
        help="Controller type to evaluate",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint file for neural controllers",
    )

    # Evaluation options
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Maximum steps per episode",
    )

    # Environment options
    parser.add_argument(
        "--motion-type",
        type=str,
        default="circular",
        choices=["linear", "circular", "sinusoidal", "figure8", "stationary"],
        help="Target motion type",
    )
    parser.add_argument(
        "--episode-length",
        type=float,
        default=30.0,
        help="Episode duration in seconds",
    )
    parser.add_argument(
        "--target-radius",
        type=float,
        default=0.5,
        help="On-target radius threshold in meters",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Output directory for reports and plots",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots",
    )

    # Sweep mode
    parser.add_argument(
        "--sweep",
        type=str,
        help="Path to sweep configuration YAML file",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Run sweep if specified
    if args.sweep:
        run_hyperparameter_sweep(args.sweep, args.output_dir)
        return 0

    # Load controller
    try:
        controller = load_controller(
            controller_type=args.controller,
            checkpoint_path=args.checkpoint,
        )
    except Exception as e:
        logger.error("Failed to load controller: %s", e)
        return 1

    # Setup environment config
    env_config = EnvConfig()
    env_config.target.motion_type = args.motion_type
    env_config.simulation.max_episode_time = args.episode_length
    env_config.success_criteria.target_radius = args.target_radius

    # Setup success criteria
    criteria = SuccessCriteria(
        min_on_target_ratio=0.8,
        min_episode_duration=args.episode_length,
        target_radius=args.target_radius,
    )

    # Create evaluator
    evaluator = Evaluator(
        controller=controller,
        env_config=env_config,
        criteria=criteria,
        output_dir=args.output_dir,
    )

    # Run evaluation
    summary = evaluator.evaluate(
        num_episodes=args.episodes,
        base_seed=args.seed,
        max_steps_per_episode=args.max_steps,
    )

    # Save reports
    evaluator.save_report(summary)

    # Generate plots
    if not args.no_plots and evaluator.episode_data_list:
        evaluator.generate_all_plots()

    # Exit code based on success criteria
    if summary.meets_criteria:
        logger.info("SUCCESS: Evaluation meets criteria!")
        return 0
    else:
        logger.warning(
            "FAILED: Evaluation does not meet criteria "
            "(%.1f%% < 80%% on-target)",
            summary.mean_on_target_ratio * 100,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
