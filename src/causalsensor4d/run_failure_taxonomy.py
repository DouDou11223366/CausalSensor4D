from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from .failure_taxonomy import DEFAULT_FAILURE_TTC_THRESHOLD, DEFAULT_TTC_THRESHOLDS, generate_failure_taxonomy


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="CausalSensor4D public_release: failure taxonomy and severity validation for public_release outputs.")
    parser.add_argument("--run-dir", type=str, default="", help="Run folder containing baseline_ablation, e.g. outputs/clean_safe_longitudinal_run.")
    parser.add_argument("--baseline-out-dir", type=str, default="", help="Direct baseline_ablation folder. Overrides --run-dir if provided.")
    parser.add_argument("--out", type=str, default="outputs/failure_taxonomy_run")
    parser.add_argument("--previous-run-dir", type=str, default="", help="Optional previous run folder for previous-stage comparison.")
    parser.add_argument("--ttc-threshold", type=float, default=DEFAULT_FAILURE_TTC_THRESHOLD)
    parser.add_argument("--sensitivity-thresholds", type=str, default=",".join(str(x) for x in DEFAULT_TTC_THRESHOLDS))
    parser.add_argument("--primary-method", type=str, default="causal_hybrid")
    parser.add_argument("--reference-method", type=str, default="distance_all")
    parser.add_argument("--skip-candidate-rows", action="store_true", help="Only audit best counterfactuals, not all candidate rows.")
    parser.add_argument("--max-candidate-tables", type=int, default=0, help="Debug/smoke-test limit. 0 means all candidate tables.")
    args = parser.parse_args()

    payload = generate_failure_taxonomy(
        run_dir=Path(args.run_dir) if args.run_dir else None,
        baseline_out_dir=Path(args.baseline_out_dir) if args.baseline_out_dir else None,
        out_dir=Path(args.out),
        previous_run_dir=Path(args.previous_run_dir) if args.previous_run_dir else None,
        ttc_threshold=args.ttc_threshold,
        sensitivity_thresholds=_parse_float_list(args.sensitivity_thresholds),
        primary_method=args.primary_method,
        reference_method=args.reference_method,
        load_candidate_rows=not args.skip_candidate_rows,
        max_candidate_tables=args.max_candidate_tables,
    )

    print("CausalSensor4D public_release failure taxonomy finished.")
    print(f"Baseline input: {payload.get('input_diagnostic', {}).get('baseline_out_dir')}")
    print(f"Candidate tables loaded: {payload.get('input_diagnostic', {}).get('candidate_tables_loaded')} / {payload.get('input_diagnostic', {}).get('candidate_tables_found')}")
    print(f"Scene rows: {payload.get('input_diagnostic', {}).get('num_scene_rows')}")
    print(f"Report: {payload['outputs']['report']}")
    print(f"Summary: {payload['outputs']['summary']}")
    print(f"Claim checklist: {payload['outputs']['claim_safety_checklist']}")


if __name__ == "__main__":
    main()
