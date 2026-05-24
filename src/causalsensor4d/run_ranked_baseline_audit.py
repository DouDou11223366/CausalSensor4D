from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List

from .causal_hybrid_audit import DEFAULT_BUDGETS, DEFAULT_METHODS, DEFAULT_RANDOM_SEEDS, generate_causal_hybrid_audit
from .clean_safe_ablation import generate_clean_safe_ablation_report
from .run_baseline_ablation_csv import main as baseline_main


def _parse_csv_list(s: str) -> List[str]:
    if s.strip().lower() == "all":
        return list(DEFAULT_METHODS)
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _run_baseline(csv_dir: Path, baseline_out_dir: Path, planner: str, methods: str, random_budget: int, seed: int, ego_track_id: str, max_scenes: int = 0) -> None:
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "run_baseline_ablation_csv",
            "--csv-dir", str(csv_dir),
            "--out", str(baseline_out_dir),
            "--planner", planner,
            "--methods", methods,
            "--random-budget", str(random_budget),
            "--seed", str(seed),
            "--ego-track-id", ego_track_id,
        ]
        if max_scenes and max_scenes > 0:
            sys.argv += ["--max-scenes", str(max_scenes)]
        baseline_main()
    finally:
        sys.argv = old_argv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CausalSensor4D public_release: fresh runtime-ranked clean-safe baseline run plus budgeted audit."
    )
    parser.add_argument("--csv-dir", type=str, required=True, help="Selected clean-safe generic_tracks_csv folder, e.g. public_release selected_clean_csv.")
    parser.add_argument("--out", type=str, default="outputs/clean_safe_ranked_budget_run")
    parser.add_argument("--planner", type=str, default="delayed", choices=["normal", "delayed", "weak_brake", "conservative", "aggressive"])
    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    parser.add_argument("--random-budget", type=int, default=36)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--ego-track-id", type=str, default="ego")
    parser.add_argument("--llm-benchmark-dir", type=str, default="", help="Optional optional LLM verified benchmark folder for report summary.")
    parser.add_argument("--budgets", type=str, default=",".join(str(x) for x in DEFAULT_BUDGETS))
    parser.add_argument("--random-seeds", type=str, default=",".join(str(x) for x in DEFAULT_RANDOM_SEEDS))
    parser.add_argument("--skip-baseline", action="store_true", help="Only audit an existing --baseline-out-dir inside --out or explicitly passed.")
    parser.add_argument("--baseline-out-dir", type=str, default="", help="Existing baseline_ablation folder to audit when --skip-baseline is used.")
    parser.add_argument("--max-scenes", type=int, default=0, help="Optional smoke-test limit for the fresh baseline run. 0 means all scenes.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    baseline_out_dir = Path(args.baseline_out_dir) if args.baseline_out_dir else out_dir / "baseline_ablation"
    audit_out_dir = out_dir / "budgeted_audit"
    report_out_dir = out_dir / "report_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_baseline:
        _run_baseline(
            csv_dir=Path(args.csv_dir),
            baseline_out_dir=baseline_out_dir,
            planner=args.planner,
            methods=args.methods,
            random_budget=args.random_budget,
            seed=args.seed,
            ego_track_id=args.ego_track_id,
            max_scenes=args.max_scenes,
        )
    else:
        if not baseline_out_dir.exists():
            raise FileNotFoundError(f"--skip-baseline was set but baseline output does not exist: {baseline_out_dir}")

    # Optional clean-safe report table, kept separate from the budgeted audit.
    try:
        generate_clean_safe_ablation_report(
            baseline_out_dir=baseline_out_dir,
            out_dir=report_out_dir,
            llm_benchmark_dir=args.llm_benchmark_dir if args.llm_benchmark_dir else None,
            title_version="public_release",
        )
    except Exception as exc:
        print(f"[WARN] Clean-safe report summary generation failed, but audit will continue: {exc}")

    payload = generate_causal_hybrid_audit(
        baseline_out_dir=baseline_out_dir,
        out_dir=audit_out_dir,
        methods=_parse_csv_list(args.methods),
        primary_method="causal_hybrid",
        reference_method="distance_all",
        budgets=_parse_int_list(args.budgets),
        random_seeds=_parse_int_list(args.random_seeds),
        random_reference_method="distance_all",
        rank_strategies=["current_order", "budget_ranked", "cost_only_diagnostic"],
    )

    print("CausalSensor4D public_release fresh runtime-ranked baseline + audit finished.")
    print(f"CSV folder: {args.csv_dir}")
    print(f"Baseline output: {baseline_out_dir}")
    print(f"Audit output: {audit_out_dir}")
    print(f"Runtime-rank ready: {payload.get('runtime_rank_diagnostic', {}).get('runtime_rank_ready')}")
    print(f"Audit report: {payload['outputs']['report']}")
    print(f"Audit summary: {audit_out_dir / 'causal_hybrid_audit_summary.json'}")
    print(f"Runtime-rank diagnostic: {payload['outputs']['runtime_rank_diagnostic']}")


if __name__ == "__main__":
    main()
