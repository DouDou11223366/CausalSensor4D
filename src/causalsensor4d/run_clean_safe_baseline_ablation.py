from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .clean_safe_ablation import generate_clean_safe_ablation_report
from .run_baseline_ablation_csv import main as baseline_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Run clean-safe baseline/ablation and optional LLM comparison summary.")
    parser.add_argument("--csv-dir", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--planner", type=str, default="delayed")
    parser.add_argument("--methods", type=str, default="all")
    parser.add_argument("--random-budget", type=int, default=36)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--llm-benchmark-dir", type=str, default="")
    args = parser.parse_args()

    out_dir = Path(args.out)
    baseline_out_dir = out_dir / "baseline_ablation"
    report_out_dir = out_dir / "report_summary"
    baseline_out_dir.mkdir(parents=True, exist_ok=True)
    report_out_dir.mkdir(parents=True, exist_ok=True)

    # Run the existing baseline runner programmatically.
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "run_baseline_ablation_csv",
            "--csv-dir", args.csv_dir,
            "--out", str(baseline_out_dir),
            "--planner", args.planner,
            "--methods", args.methods,
            "--random-budget", str(args.random_budget),
            "--seed", str(args.seed),
        ]
        baseline_main()
    finally:
        sys.argv = old_argv

    payload = generate_clean_report_summary = generate_clean_safe_ablation_report(
        baseline_out_dir=baseline_out_dir,
        out_dir=report_out_dir,
        llm_benchmark_dir=args.llm_benchmark_dir if args.llm_benchmark_dir else None,
        title_version="the hybrid",
    )

    print("CausalSensor4D the hybrid clean-safe causal-hybrid baseline/ablation finished.")
    print(f"CSV folder: {args.csv_dir}")
    print(f"Baseline output: {baseline_out_dir}")
    print(f"Report summary output: {report_out_dir}")
    print(f"Report: {report_out_dir / 'clean_safe_ablation_report.md'}")
    print(f"Summary: {report_out_dir / 'clean_safe_ablation_summary.json'}")


if __name__ == "__main__":
    main()
