from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from .causal_hybrid_audit import DEFAULT_BUDGETS, DEFAULT_METHODS, DEFAULT_RANDOM_SEEDS, generate_causal_hybrid_audit
from .clean_safe_ablation import generate_clean_safe_ablation_report


def _parse_csv_list(s: str) -> List[str]:
    if s.strip().lower() == "all":
        return list(DEFAULT_METHODS)
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="CausalSensor4D public_release: regenerate audit/report summaries from an existing baseline_ablation folder only.")
    parser.add_argument("--baseline-out-dir", type=str, required=True, help="Existing baseline_ablation folder. No scene evaluation is run.")
    parser.add_argument("--out", type=str, default="outputs/regenerated_audit_run")
    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    parser.add_argument("--llm-benchmark-dir", type=str, default="")
    parser.add_argument("--budgets", type=str, default=",".join(str(x) for x in DEFAULT_BUDGETS))
    parser.add_argument("--random-seeds", type=str, default=",".join(str(x) for x in DEFAULT_RANDOM_SEEDS))
    args = parser.parse_args()

    out_dir = Path(args.out)
    audit_out_dir = out_dir / "budgeted_audit"
    report_out_dir = out_dir / "report_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_out_dir = Path(args.baseline_out_dir)
    if not baseline_out_dir.exists():
        raise FileNotFoundError(f"Existing baseline output does not exist: {baseline_out_dir}")

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

    print("CausalSensor4D public_release existing-baseline audit regeneration finished.")
    print(f"Baseline input: {baseline_out_dir}")
    print(f"Output: {out_dir}")
    print(f"Runtime-rank ready: {payload.get('runtime_rank_diagnostic', {}).get('runtime_rank_ready')}")


if __name__ == "__main__":
    main()
