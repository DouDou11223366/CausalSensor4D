from __future__ import annotations

import argparse
from pathlib import Path
import os
import shutil
import subprocess
import sys
from typing import List

import pandas as pd

from .causal_hybrid_audit import DEFAULT_BUDGETS, DEFAULT_METHODS, DEFAULT_RANDOM_SEEDS, generate_causal_hybrid_audit
from .clean_safe_ablation import generate_clean_safe_ablation_report
from .run_baseline_ablation_csv import main as baseline_main
from .baseline_comparison import save_baseline_artifacts
from .longitudinal_diagnostic import generate_longitudinal_diagnostic


def _parse_csv_list(s: str) -> List[str]:
    if s.strip().lower() == "all":
        return list(DEFAULT_METHODS)
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _run_baseline(csv_dir: Path, baseline_out_dir: Path, planner: str, methods: str, random_budget: int, seed: int, ego_track_id: str, max_scenes: int = 0) -> None:
    """Run the public_release baseline with per-method subprocess isolation.

    Heading-aware longitudinal geometry exposes more candidate rows than earlier
    x-axis-only versions.  Running each method in a fresh Python process avoids
    long-lived planner/dataframe/object caches and makes the PyCharm green-button
    run more reliable on Windows.
    """
    parsed_methods = _parse_csv_list(methods)
    temp_root = baseline_out_dir / "_isolated_method_runs"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)
    baseline_out_dir.mkdir(parents=True, exist_ok=True)
    final_per_method = baseline_out_dir / "per_method"
    final_per_method.mkdir(parents=True, exist_ok=True)

    src_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["CS4D_FORCE_EXIT_AFTER_BASELINE"] = "1"

    all_rows = []
    for method in parsed_methods:
        method_out = temp_root / method
        cmd = [
            sys.executable,
            "-m",
            "causalsensor4d.run_baseline_ablation_csv",
            "--csv-dir", str(csv_dir),
            "--out", str(method_out),
            "--planner", planner,
            "--methods", method,
            "--random-budget", str(random_budget),
            "--seed", str(seed),
            "--ego-track-id", ego_track_id,
        ]
        if max_scenes and max_scenes > 0:
            cmd += ["--max-scenes", str(max_scenes)]
        print(f"[public_release isolated baseline] method={method}", flush=True)
        subprocess.run(cmd, check=True, env=env)

        method_rows = method_out / "all_baseline_scene_results.csv"
        if method_rows.exists():
            all_rows.append(pd.read_csv(method_rows))

        src_method_dir = method_out / "per_method" / method
        dst_method_dir = final_per_method / method
        if dst_method_dir.exists():
            shutil.rmtree(dst_method_dir)
        if src_method_dir.exists():
            shutil.copytree(src_method_dir, dst_method_dir)

    if all_rows:
        all_df = pd.concat(all_rows, ignore_index=True)
    else:
        all_df = pd.DataFrame()
    save_baseline_artifacts(all_df, baseline_out_dir, parsed_methods)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CausalSensor4D public_release: fresh heading-aware longitudinal clean-safe baseline run plus budgeted audit."
    )
    parser.add_argument("--csv-dir", type=str, required=True, help="Selected clean-safe generic_tracks_csv folder, e.g. public_release selected_clean_csv.")
    parser.add_argument("--out", type=str, default="outputs/clean_safe_longitudinal_run")
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

    longitudinal_out_dir = out_dir / "longitudinal_diagnostic"
    try:
        longitudinal_payload = generate_longitudinal_diagnostic(baseline_out_dir=baseline_out_dir, out_dir=longitudinal_out_dir)
    except Exception as exc:
        longitudinal_payload = {"error": str(exc)}
        print(f"[WARN] Longitudinal diagnostic generation failed, but audit will continue: {exc}")

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

    print("CausalSensor4D public_release fresh heading-aware longitudinal baseline + audit finished.")
    print(f"CSV folder: {args.csv_dir}")
    print(f"Baseline output: {baseline_out_dir}")
    print(f"Audit output: {audit_out_dir}")
    print(f"Runtime-rank ready: {payload.get('runtime_rank_diagnostic', {}).get('runtime_rank_ready')}")
    print(f"Audit report: {payload['outputs']['report']}")
    print(f"Longitudinal diagnostic: {longitudinal_out_dir / 'longitudinal_diagnostic_summary.json'}")
    print(f"Audit summary: {audit_out_dir / 'causal_hybrid_audit_summary.json'}")
    print(f"Runtime-rank diagnostic: {payload['outputs']['runtime_rank_diagnostic']}")


if __name__ == "__main__":
    main()
