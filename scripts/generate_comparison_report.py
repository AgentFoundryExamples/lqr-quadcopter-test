#!/usr/bin/env python3
"""Generate a comparison report from controller evaluation results.

This script reads metrics.json files from the reports/comparison directory
and generates a ranked summary of controller performance.

Usage:
    python scripts/generate_comparison_report.py [--report-dir REPORT_DIR]
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Generate controller comparison report"
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/comparison"),
        help="Directory containing controller evaluation results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file for comparison summary",
    )
    args = parser.parse_args()

    report_dir = args.report_dir
    output_file = args.output or (report_dir / "comparison_summary.json")

    if not report_dir.exists():
        print(
            f"Error: Report directory '{report_dir}' does not exist.",
            file=sys.stderr,
        )
        print("Run 'make compare-controllers' first.", file=sys.stderr)
        sys.exit(1)

    results = []

    for controller_dir in report_dir.iterdir():
        if controller_dir.is_dir():
            metrics_file = controller_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    data = json.load(f)
                    results.append(
                        {
                            "controller": controller_dir.name,
                            "success_rate": data.get("success_rate", 0),
                            "mean_on_target_ratio": data.get("mean_on_target_ratio", 0),
                            "mean_tracking_error": data.get(
                                "mean_tracking_error", float("inf")
                            ),
                        }
                    )

    if not results:
        print(
            "No results found. Run 'make compare-controllers' first.", file=sys.stderr
        )
        sys.exit(1)

    # Sort by success rate (desc), then on-target ratio (desc), then error (asc)
    results.sort(
        key=lambda x: (
            -x["success_rate"],
            -x["mean_on_target_ratio"],
            x["mean_tracking_error"],
        )
    )

    # Print report
    print("")
    print("=" * 70)
    print("CONTROLLER COMPARISON REPORT")
    print("=" * 70)
    header = (
        f"{'Controller':<15} {'Success Rate':<15} "
        f"{'On-Target %':<15} {'Mean Error (m)':<15}"
    )
    print(header)
    print("-" * 70)
    for r in results:
        print(
            f"{r['controller']:<15} {r['success_rate']*100:>12.1f}% "
            f"{r['mean_on_target_ratio']*100:>12.1f}% {r['mean_tracking_error']:>12.3f}"
        )
    print("=" * 70)
    print("")

    # Save to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump({"rankings": results}, f, indent=2)
    print(f"Report saved to {output_file}")


if __name__ == "__main__":
    main()
