"""
Evaluation Metrics for Quadcopter Tracking

This module provides metrics computation utilities for evaluating controller
performance:
- On-target ratio computation
- Tracking error statistics (mean, max, RMS)
- Control effort (total action magnitude)
- Overshoot detection
- Success criteria evaluation

Design Philosophy:
- Stateless functions for computing metrics from episode data
- Support for both single-episode and batch evaluation
- Configurable success criteria thresholds
"""

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SuccessCriteria:
    """
    Configuration for success criteria evaluation.

    Attributes:
        min_on_target_ratio: Minimum fraction of time within target radius (0.0-1.0).
        min_episode_duration: Minimum episode duration in seconds.
        target_radius: On-target threshold distance in meters.
    """

    min_on_target_ratio: float = 0.8
    min_episode_duration: float = 30.0
    target_radius: float = 0.5


@dataclass
class EpisodeMetrics:
    """
    Computed metrics for a single episode.

    Attributes:
        episode_duration: Total episode time in seconds.
        on_target_ratio: Fraction of time within target radius.
        mean_tracking_error: Mean distance to target.
        max_tracking_error: Maximum distance to target.
        rms_tracking_error: Root mean square tracking error.
        total_control_effort: Sum of action magnitudes.
        mean_control_effort: Average action magnitude per step.
        overshoot_count: Number of overshoot events detected.
        max_overshoot: Maximum overshoot distance.
        success: Whether episode met success criteria.
        termination_reason: Why episode ended.
        action_violations: Number of action violations detected.
    """

    episode_duration: float = 0.0
    on_target_ratio: float = 0.0
    mean_tracking_error: float = 0.0
    max_tracking_error: float = 0.0
    rms_tracking_error: float = 0.0
    total_control_effort: float = 0.0
    mean_control_effort: float = 0.0
    overshoot_count: int = 0
    max_overshoot: float = 0.0
    success: bool = False
    termination_reason: str = ""
    action_violations: int = 0

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "episode_duration": self.episode_duration,
            "on_target_ratio": self.on_target_ratio,
            "mean_tracking_error": self.mean_tracking_error,
            "max_tracking_error": self.max_tracking_error,
            "rms_tracking_error": self.rms_tracking_error,
            "total_control_effort": self.total_control_effort,
            "mean_control_effort": self.mean_control_effort,
            "overshoot_count": self.overshoot_count,
            "max_overshoot": self.max_overshoot,
            "success": self.success,
            "termination_reason": self.termination_reason,
            "action_violations": self.action_violations,
        }


@dataclass
class EvaluationSummary:
    """
    Summary statistics for multiple evaluation episodes.

    Attributes:
        total_episodes: Number of episodes evaluated.
        successful_episodes: Number meeting success criteria.
        success_rate: Fraction of successful episodes.
        mean_on_target_ratio: Mean on-target ratio across episodes.
        std_on_target_ratio: Standard deviation of on-target ratio.
        mean_tracking_error: Mean tracking error across episodes.
        std_tracking_error: Standard deviation of tracking error.
        mean_control_effort: Mean control effort across episodes.
        best_episode_idx: Index of best performing episode.
        worst_episode_idx: Index of worst performing episode.
        meets_criteria: Whether overall success criteria are met.
        episode_metrics: List of individual episode metrics.
    """

    total_episodes: int = 0
    successful_episodes: int = 0
    success_rate: float = 0.0
    mean_on_target_ratio: float = 0.0
    std_on_target_ratio: float = 0.0
    mean_tracking_error: float = 0.0
    std_tracking_error: float = 0.0
    mean_control_effort: float = 0.0
    best_episode_idx: int = 0
    worst_episode_idx: int = 0
    meets_criteria: bool = False
    episode_metrics: list[EpisodeMetrics] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert summary to dictionary."""
        return {
            "total_episodes": self.total_episodes,
            "successful_episodes": self.successful_episodes,
            "success_rate": self.success_rate,
            "mean_on_target_ratio": self.mean_on_target_ratio,
            "std_on_target_ratio": self.std_on_target_ratio,
            "mean_tracking_error": self.mean_tracking_error,
            "std_tracking_error": self.std_tracking_error,
            "mean_control_effort": self.mean_control_effort,
            "best_episode_idx": self.best_episode_idx,
            "worst_episode_idx": self.worst_episode_idx,
            "meets_criteria": self.meets_criteria,
            "episode_metrics": [m.to_dict() for m in self.episode_metrics],
        }


def compute_tracking_error(
    quad_positions: np.ndarray, target_positions: np.ndarray
) -> np.ndarray:
    """
    Compute tracking error (distance) at each timestep.

    Args:
        quad_positions: Quadcopter positions array (N, 3).
        target_positions: Target positions array (N, 3).

    Returns:
        Tracking error array (N,) with distance at each timestep.
    """
    if quad_positions.shape != target_positions.shape:
        raise ValueError(
            f"Shape mismatch: quad {quad_positions.shape} vs target "
            f"{target_positions.shape}"
        )
    return np.linalg.norm(target_positions - quad_positions, axis=1)


def compute_on_target_ratio(
    tracking_errors: np.ndarray, target_radius: float
) -> float:
    """
    Compute fraction of time within target radius.

    Args:
        tracking_errors: Array of tracking error distances.
        target_radius: On-target threshold distance.

    Returns:
        Fraction of samples within target radius (0.0-1.0).
    """
    if len(tracking_errors) == 0:
        return 0.0
    on_target = tracking_errors <= target_radius
    return float(np.mean(on_target))


def compute_control_effort(actions: np.ndarray) -> tuple[float, float]:
    """
    Compute total and mean control effort from actions.

    Control effort is measured as the sum of absolute action values,
    normalized by the number of steps.

    Args:
        actions: Array of actions (N, 4) with [thrust, roll_rate, pitch_rate, yaw_rate].

    Returns:
        Tuple of (total_effort, mean_effort_per_step).
    """
    if len(actions) == 0:
        return 0.0, 0.0

    # Compute magnitude of each action
    action_magnitudes = np.linalg.norm(actions, axis=1)
    total_effort = float(np.sum(action_magnitudes))
    mean_effort = float(np.mean(action_magnitudes))
    return total_effort, mean_effort


def detect_overshoots(
    tracking_errors: np.ndarray,
    target_radius: float,
    window_size: int = 10,
) -> tuple[int, float]:
    """
    Detect overshoot events where error increases after being on-target.

    An overshoot is detected when:
    1. Error was within target radius.
    2. Error increases above threshold for a sustained period (>= window_size).

    Args:
        tracking_errors: Array of tracking error distances.
        target_radius: On-target threshold distance.
        window_size: Minimum consecutive samples off-target to count as an overshoot.

    Returns:
        Tuple of (overshoot_count, max_overshoot_distance).
    """
    if len(tracking_errors) < window_size:
        return 0, 0.0

    on_target = tracking_errors <= target_radius
    overshoot_count = 0
    max_overshoot = 0.0

    in_overshoot_phase = False
    off_target_streak = 0
    current_overshoot_max = 0.0

    for i in range(1, len(on_target)):
        # Transition from on-target to off-target
        if on_target[i - 1] and not on_target[i]:
            in_overshoot_phase = True
            off_target_streak = 1
            current_overshoot_max = tracking_errors[i] - target_radius
        elif in_overshoot_phase and not on_target[i]:
            off_target_streak += 1
            current_overshoot_max = max(
                current_overshoot_max, tracking_errors[i] - target_radius
            )
        # Transition from off-target to on-target
        elif in_overshoot_phase and on_target[i]:
            if off_target_streak >= window_size:
                overshoot_count += 1
                max_overshoot = max(max_overshoot, current_overshoot_max)
            in_overshoot_phase = False
            off_target_streak = 0
            current_overshoot_max = 0.0

    # Handle case where episode ends during an overshoot
    if in_overshoot_phase and off_target_streak >= window_size:
        overshoot_count += 1
        max_overshoot = max(max_overshoot, current_overshoot_max)

    return overshoot_count, float(max_overshoot)


def compute_episode_metrics(
    episode_data: list[dict],
    criteria: SuccessCriteria | None = None,
    episode_info: dict | None = None,
) -> EpisodeMetrics:
    """
    Compute all metrics for a single episode.

    Args:
        episode_data: List of step data dictionaries with keys:
            - quadcopter_position: (3,) position array
            - target_position: (3,) position array
            - action: (4,) action array
            - time: simulation time
        criteria: Success criteria configuration.
        episode_info: Final episode info dict from environment.

    Returns:
        EpisodeMetrics with computed values.
    """
    if criteria is None:
        criteria = SuccessCriteria()

    if not episode_data:
        return EpisodeMetrics(
            success=False,
            termination_reason="no_data",
        )

    # Extract arrays from episode data
    quad_positions = np.array([d["quadcopter_position"] for d in episode_data])
    target_positions = np.array([d["target_position"] for d in episode_data])
    actions = np.array([d["action"] for d in episode_data])
    times = np.array([d["time"] for d in episode_data])

    # Compute tracking errors
    tracking_errors = compute_tracking_error(quad_positions, target_positions)

    # Compute metrics
    on_target_ratio = compute_on_target_ratio(tracking_errors, criteria.target_radius)
    total_effort, mean_effort = compute_control_effort(actions)
    overshoot_count, max_overshoot = detect_overshoots(
        tracking_errors, criteria.target_radius
    )

    # Episode duration
    episode_duration = float(times[-1]) if len(times) > 0 else 0.0

    # Evaluate success
    success = (
        episode_duration >= criteria.min_episode_duration
        and on_target_ratio >= criteria.min_on_target_ratio
    )

    # Get termination info
    termination_reason = ""
    action_violations = 0
    if episode_info:
        termination_reason = episode_info.get("termination_reason", "")
        action_violations = episode_info.get("action_violations", 0)

    return EpisodeMetrics(
        episode_duration=episode_duration,
        on_target_ratio=on_target_ratio,
        mean_tracking_error=float(np.mean(tracking_errors)),
        max_tracking_error=float(np.max(tracking_errors)),
        rms_tracking_error=float(np.sqrt(np.mean(tracking_errors**2))),
        total_control_effort=total_effort,
        mean_control_effort=mean_effort,
        overshoot_count=overshoot_count,
        max_overshoot=max_overshoot,
        success=success,
        termination_reason=termination_reason,
        action_violations=action_violations,
    )


def compute_evaluation_summary(
    episode_metrics_list: list[EpisodeMetrics],
    criteria: SuccessCriteria | None = None,
) -> EvaluationSummary:
    """
    Compute summary statistics across multiple episodes.

    Args:
        episode_metrics_list: List of EpisodeMetrics from evaluation.
        criteria: Success criteria configuration.

    Returns:
        EvaluationSummary with aggregate statistics.
    """
    if criteria is None:
        criteria = SuccessCriteria()

    if not episode_metrics_list:
        return EvaluationSummary()

    total_episodes = len(episode_metrics_list)
    successful_episodes = sum(1 for m in episode_metrics_list if m.success)

    on_target_ratios = [m.on_target_ratio for m in episode_metrics_list]
    tracking_errors = [m.mean_tracking_error for m in episode_metrics_list]
    control_efforts = [m.mean_control_effort for m in episode_metrics_list]

    # Find best and worst episodes by on-target ratio
    best_idx = int(np.argmax(on_target_ratios))
    worst_idx = int(np.argmin(on_target_ratios))

    # Check if overall criteria are met
    success_rate = successful_episodes / total_episodes if total_episodes > 0 else 0.0
    mean_on_target = float(np.mean(on_target_ratios))
    meets_criteria = mean_on_target >= criteria.min_on_target_ratio

    return EvaluationSummary(
        total_episodes=total_episodes,
        successful_episodes=successful_episodes,
        success_rate=success_rate,
        mean_on_target_ratio=mean_on_target,
        std_on_target_ratio=float(np.std(on_target_ratios)),
        mean_tracking_error=float(np.mean(tracking_errors)),
        std_tracking_error=float(np.std(tracking_errors)),
        mean_control_effort=float(np.mean(control_efforts)),
        best_episode_idx=best_idx,
        worst_episode_idx=worst_idx,
        meets_criteria=meets_criteria,
        episode_metrics=episode_metrics_list,
    )


def format_metrics_report(summary: EvaluationSummary) -> str:
    """
    Format evaluation summary as human-readable report.

    Args:
        summary: EvaluationSummary to format.

    Returns:
        Formatted string report.
    """
    lines = [
        "=" * 60,
        "EVALUATION SUMMARY",
        "=" * 60,
        "",
        f"Total Episodes: {summary.total_episodes}",
        f"Successful Episodes: {summary.successful_episodes}",
        f"Success Rate: {summary.success_rate:.1%}",
        "",
        "Tracking Performance:",
        f"  Mean On-Target Ratio: {summary.mean_on_target_ratio:.1%} "
        f"(± {summary.std_on_target_ratio:.1%})",
        f"  Mean Tracking Error: {summary.mean_tracking_error:.3f}m "
        f"(± {summary.std_tracking_error:.3f}m)",
        f"  Mean Control Effort: {summary.mean_control_effort:.3f}",
        "",
        f"Best Episode: #{summary.best_episode_idx + 1} "
        f"({summary.episode_metrics[summary.best_episode_idx].on_target_ratio:.1%} "
        f"on-target)"
        if summary.episode_metrics
        else "Best Episode: N/A",
        f"Worst Episode: #{summary.worst_episode_idx + 1} "
        f"({summary.episode_metrics[summary.worst_episode_idx].on_target_ratio:.1%} "
        f"on-target)"
        if summary.episode_metrics
        else "Worst Episode: N/A",
        "",
        f"SUCCESS CRITERIA MET: {'YES' if summary.meets_criteria else 'NO'}",
        "=" * 60,
    ]
    return "\n".join(lines)
