from __future__ import annotations

import argparse
from pathlib import Path
from .llm_tools import save_llm_artifacts, LLMDiagnosisConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LLM-ready CausalSensor4D diagnosis prompts from AD model outputs.")
    parser.add_argument("--result-dir", required=True, help="Directory containing AD model comparison outputs.")
    parser.add_argument("--out", required=True, help="Output directory for LLM-ready artifacts.")
    parser.add_argument("--max-prompt-chars", type=int, default=22000)
    args = parser.parse_args()

    index = save_llm_artifacts(args.result_dir, args.out, LLMDiagnosisConfig(max_prompt_chars=args.max_prompt_chars))
    print("CausalSensor4D LLM-ready diagnosis artifacts generated.")
    for k, v in index.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
