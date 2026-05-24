from __future__ import annotations

import argparse
import json
from pathlib import Path

from .openrouter_client import OpenRouterConfig, run_prompt_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Call OpenRouter on CausalSensor4D LLM-ready prompt files.")
    parser.add_argument("--diagnosis-prompt", required=True, help="Path to llm_diagnosis_prompt.md")
    parser.add_argument("--candidate-prompt", default=None, help="Optional path to llm_candidate_proposal_prompt.md")
    parser.add_argument("--out", required=True, help="Output directory for online LLM responses")
    parser.add_argument("--model", default="deepseek/deepseek-chat-v3.1:free", help="OpenRouter model id")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY", help="Environment variable containing API key")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = OpenRouterConfig(
        model=args.model,
        api_key_env=args.api_key_env,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )

    index = {"model": args.model, "outputs": {}}
    index["outputs"]["diagnosis"] = run_prompt_file(args.diagnosis_prompt, out, "openrouter_diagnosis", cfg)
    if args.candidate_prompt:
        index["outputs"]["candidate_proposal"] = run_prompt_file(args.candidate_prompt, out, "openrouter_candidate_proposal", cfg)

    index_path = out / "openrouter_artifact_index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print("CausalSensor4D OpenRouter diagnosis finished.")
    print(f"Model: {args.model}")
    print(f"Output index: {index_path}")
    for k, v in index["outputs"].items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
