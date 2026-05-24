from __future__ import annotations

import argparse
import json
from pathlib import Path

from .llm_verified_benchmark import LLMVerifiedBenchmarkConfig, build_llm_verified_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Build public_release LLM-verified candidate benchmark report")
    parser.add_argument("--response-validation-json", required=True)
    parser.add_argument("--candidate-validation-json", required=True)
    parser.add_argument("--candidate-verification-table", required=True)
    parser.add_argument("--diagnosis-md", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-original-ttc-safe", type=float, default=2.0)
    parser.add_argument("--allow-original-collision", action="store_true")
    parser.add_argument("--allow-original-hard-brake", action="store_true")
    parser.add_argument("--diagnosis-min-chars", type=int, default=300)
    # public_release clean-safe label propagation.
    parser.add_argument("--safety-table", default=None, help="Path to original_safety_table.csv or safe_selected_scenes.csv from the upstream safety filter")
    parser.add_argument("--safe-csv-dir", default=None, help="Path to a safe_csv folder whose file stems should be marked as clean-safe")
    parser.add_argument("--assume-input-csv-clean-safe", action="store_true", help="Mark all candidates as originating from clean-safe input CSVs; use only when the input CSV folder is safe_csv")
    args = parser.parse_args()

    cfg = LLMVerifiedBenchmarkConfig(
        min_original_ttc_safe=args.min_original_ttc_safe,
        require_no_original_collision=not args.allow_original_collision,
        require_no_original_hard_brake=not args.allow_original_hard_brake,
        diagnosis_min_chars=args.diagnosis_min_chars,
        assume_input_csv_clean_safe=args.assume_input_csv_clean_safe,
        safety_table_path=args.safety_table,
        safe_csv_dir=args.safe_csv_dir,
    )
    outputs = build_llm_verified_benchmark(
        response_validation_json=args.response_validation_json,
        candidate_validation_json=args.candidate_validation_json,
        candidate_verification_table=args.candidate_verification_table,
        diagnosis_md=args.diagnosis_md,
        out_dir=args.out,
        config=cfg,
    )
    print("CausalSensor4D public_release LLM-verified benchmark finished.")
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
