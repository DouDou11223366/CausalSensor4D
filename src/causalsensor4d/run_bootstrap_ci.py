from __future__ import annotations

import argparse
from pathlib import Path

from .bootstrap_ci import generate_bootstrap_ci_pack, DEFAULT_N_BOOT, DEFAULT_SEED


def main() -> None:
    parser = argparse.ArgumentParser(description="CausalSensor4D public_release bootstrap CI pack")
    parser.add_argument("--baseline-run-dir", required=True, help="Path to public_release clean_safe_longitudinal_run output folder")
    parser.add_argument("--taxonomy-dir", required=True, help="Path to public_release failure_taxonomy_run output folder")
    parser.add_argument("--out", default="outputs/bootstrap_ci_run")
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--censored-mfc", type=float, default=2.0)
    args = parser.parse_args()

    summary = generate_bootstrap_ci_pack(
        baseline_run_dir=Path(args.baseline_run_dir),
        taxonomy_dir=Path(args.taxonomy_dir),
        out_dir=Path(args.out),
        n_boot=int(args.n_boot),
        seed=int(args.seed),
        censored_mfc=float(args.censored_mfc),
    )
    print("CausalSensor4D public_release bootstrap CI pack finished.")
    print(f"baseline input: {args.baseline_run_dir}")
    print(f"taxonomy input: {args.taxonomy_dir}")
    print(f"Output: {args.out}")
    diag = summary.get("input_diagnostic", {})
    print(f"Scene rows: {diag.get('scene_rows')}")
    print(f"Unique scenes: {diag.get('num_unique_scenes')}")
    print(f"Bootstrap samples: {diag.get('n_boot')}")
    print(f"Report: {summary.get('reports', {}).get('bootstrap_ci_report')}")
    print(f"Summary: {summary.get('reports', {}).get('bootstrap_ci_summary')}")


if __name__ == "__main__":
    main()
