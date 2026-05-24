from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from .causal_hybrid_audit import (
    DEFAULT_BUDGETS,
    DEFAULT_METHODS,
    DEFAULT_RANDOM_SEEDS,
    generate_causal_hybrid_audit,
)


def _parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="CausalSensor4D public_release runtime-ranked budgeted/random-seed audit.")
    parser.add_argument("--baseline-out-dir", type=str, required=True, help="baseline_ablation or baseline_ablation/per_method")
    parser.add_argument("--out-dir", type=str, default="outputs/causal_hybrid_audit")
    parser.add_argument("--primary-method", type=str, default="causal_hybrid")
    parser.add_argument("--reference-method", type=str, default="distance_all")
    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    parser.add_argument("--budgets", type=str, default=",".join(str(x) for x in DEFAULT_BUDGETS))
    parser.add_argument("--random-seeds", type=str, default=",".join(str(x) for x in DEFAULT_RANDOM_SEEDS))
    parser.add_argument("--random-reference-method", type=str, default="distance_all")
    parser.add_argument("--rank-strategies", type=str, default="current_order,budget_ranked,cost_only_diagnostic")
    args = parser.parse_args()

    payload = generate_causal_hybrid_audit(
        baseline_out_dir=Path(args.baseline_out_dir),
        out_dir=Path(args.out_dir),
        methods=_parse_csv_list(args.methods),
        primary_method=args.primary_method,
        reference_method=args.reference_method,
        budgets=_parse_int_list(args.budgets),
        random_seeds=_parse_int_list(args.random_seeds),
        random_reference_method=args.random_reference_method,
        rank_strategies=_parse_csv_list(args.rank_strategies),
    )
    print("CausalSensor4D public_release runtime-ranked budgeted/random-seed audit finished.")
    print(f"Baseline output: {args.baseline_out_dir}")
    print(f"Resolved candidate root: {payload['candidate_root']}")
    print(f"Candidate tables found: {payload['layout_diagnostic']['candidate_tables_found']}")
    print(f"Random seeds: {payload['random_seeds']}")
    print(f"Output: {args.out_dir}")
    print(f"Report: {payload['outputs']['report']}")
    print(f"Budgeted summary: {payload['outputs']['budgeted_method_summary']}")
    print(f"Budgeted AUC: {payload['outputs']['budgeted_auc_summary']}")
    print(f"Random multi-seed summary: {payload['outputs']['random_multiseed_budgeted_summary']}")
    print(f"Ranked budgeted summary: {payload['outputs']['ranked_budgeted_method_summary']}")
    print(f"Runtime-rank diagnostic: {payload['outputs']['runtime_rank_diagnostic']}")
    print(f"Summary: {Path(args.out_dir) / 'causal_hybrid_audit_summary.json'}")


if __name__ == "__main__":
    main()
