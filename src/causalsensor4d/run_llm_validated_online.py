from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

from .openrouter_client import OpenRouterConfig, run_prompt_file, call_openrouter_chat
from .llm_validation import (
    write_response_validation_report,
    parse_and_validate_candidate_file,
    build_strict_candidate_prompt,
)
from .llm_candidate_verification import verify_llm_candidates


def _print_call_audit(name: str, raw_path: str | Path) -> None:
    """Print enough evidence to know whether OpenRouter was actually called."""
    try:
        data = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        rj = data.get("response_json", {}) if isinstance(data, dict) else {}
        usage = rj.get("usage", {}) if isinstance(rj, dict) else {}
        print(f"[OpenRouter audit] {name}: ok={data.get('ok')} status={data.get('status_code')} id={rj.get('id')} model={rj.get('model')}")
        print(f"[OpenRouter audit] {name}: assistant_chars={len(str(data.get('assistant_text','')))} total_tokens={usage.get('total_tokens')} cost={usage.get('cost')}")
    except Exception as exc:
        print(f"[OpenRouter audit] {name}: could not read audit info: {exc}")


def _write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def _build_candidate_repair_prompt(original_strict_prompt: str, invalid_text: str) -> str:
    """A second-pass prompt used when the free model ignores JSON-only instructions."""
    return f"""
You previously failed to return parseable JSON for a counterfactual candidate proposal task.

Your output MUST be only a JSON array. No markdown. No explanation. No preface.
The array must contain 1 to 8 candidate objects.
Each object must include exactly these required keys:
- scene_id
- target_agent_id
- edit_family
- parameter_suggestion
- expected_failure_mode
- physical_plausibility_reason
- why_this_is_minimal

Executable edit_family values are exactly:
["lead_brake", "cut_in", "pedestrian_crossing"]

Use scene_id and target_agent_id values from the evidence in the original task. If uncertain, choose candidates that are explicitly present in the evidence.

BEGIN ORIGINAL TASK
{original_strict_prompt[:12000]}
END ORIGINAL TASK

BEGIN INVALID PREVIOUS MODEL OUTPUT
{invalid_text[:4000]}
END INVALID PREVIOUS MODEL OUTPUT

Return only valid JSON now.
""".strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validated OpenRouter + LLM candidate verification for CausalSensor4D public_release")
    parser.add_argument("--diagnosis-prompt", required=True, help="Path to llm_diagnosis_prompt.md")
    parser.add_argument("--candidate-prompt", required=True, help="Path to llm_candidate_proposal_prompt.md")
    parser.add_argument("--csv-dir", default=None, help="Optional CSV folder for simulator verification of parsed LLM proposals")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--model", default="z-ai/glm-4.5-air:free", help="OpenRouter model id")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2200)
    parser.add_argument("--timeout", type=int, default=160)
    parser.add_argument("--ad-model", default="rule_delayed", help="AD model wrapper used to verify candidate proposals")
    parser.add_argument("--ego-track-id", default="ego")
    parser.add_argument("--retry-json", action="store_true", default=True, help="Retry candidate proposal once if JSON parsing fails")
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

    index = {"version": "public_release", "model": args.model, "started_at": datetime.now().isoformat(), "outputs": {}}

    # 1) Diagnosis: free-form research explanation, then response validation.
    index["outputs"]["diagnosis"] = run_prompt_file(args.diagnosis_prompt, out, "openrouter_diagnosis", cfg)
    _print_call_audit("diagnosis", index["outputs"]["diagnosis"]["raw_response"])

    # 2) Candidate proposal: strengthen prompt to JSON-only contract, then call LLM.
    original_candidate_prompt = Path(args.candidate_prompt).read_text(encoding="utf-8")
    strict_prompt = build_strict_candidate_prompt(original_candidate_prompt)
    strict_prompt_path = out / "llm_candidate_proposal_strict_prompt.md"
    strict_prompt_path.write_text(strict_prompt, encoding="utf-8")

    def call_candidate(prompt_text: str, stem: str):
        result = call_openrouter_chat(
            prompt_text,
            config=cfg,
            system_message=(
                "You are a counterfactual autonomous-driving candidate generator. "
                "Return only valid JSON. No markdown. No prose."
            ),
        )
        raw_path = out / f"{stem}.raw_response.json"
        text_path = out / f"{stem}.md"
        raw_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        text_path.write_text(result.get("assistant_text", ""), encoding="utf-8")
        _print_call_audit(stem, raw_path)
        return result, raw_path, text_path

    candidate_result, raw_candidate_path, text_candidate_path = call_candidate(strict_prompt, "openrouter_candidate_proposal")

    # 3) Parse candidate JSON. If the model returned prose, retry once with a repair prompt.
    candidate_validation = parse_and_validate_candidate_file(text_candidate_path, out)
    retry_info = {"retry_used": False, "first_parse_ok": candidate_validation["validation"].get("parse_ok"), "first_parse_error": candidate_validation["validation"].get("parse_error")}
    if args.retry_json and not candidate_validation["validation"].get("parse_ok"):
        invalid_text = Path(text_candidate_path).read_text(encoding="utf-8")
        retry_prompt = _build_candidate_repair_prompt(strict_prompt, invalid_text)
        retry_prompt_path = out / "llm_candidate_proposal_retry_prompt.md"
        retry_prompt_path.write_text(retry_prompt, encoding="utf-8")
        _, retry_raw_path, retry_text_path = call_candidate(retry_prompt, "openrouter_candidate_proposal_retry")
        retry_validation = parse_and_validate_candidate_file(retry_text_path, out)
        retry_info.update({
            "retry_used": True,
            "retry_raw_response": str(retry_raw_path),
            "retry_assistant_text": str(retry_text_path),
            "retry_parse_ok": retry_validation["validation"].get("parse_ok"),
            "retry_parse_error": retry_validation["validation"].get("parse_error"),
        })
        # Use retry candidates only if it parsed successfully and contains at least one candidate.
        if retry_validation["validation"].get("parse_ok") and retry_validation["validation"].get("num_candidates", 0) > 0:
            candidate_validation = retry_validation
            raw_candidate_path = retry_raw_path
            text_candidate_path = retry_text_path

    (out / "candidate_retry_info.json").write_text(json.dumps(retry_info, ensure_ascii=False, indent=2), encoding="utf-8")
    index["outputs"]["candidate_proposal"] = {
        "strict_prompt": str(strict_prompt_path),
        "raw_response": str(raw_candidate_path),
        "assistant_text": str(text_candidate_path),
        "retry_info": str(out / "candidate_retry_info.json"),
    }

    # 4) Validate OpenRouter response metadata, including retry if present.
    raw_paths = {
        "diagnosis": index["outputs"]["diagnosis"]["raw_response"],
        "candidate_proposal_used": str(raw_candidate_path),
    }
    if retry_info.get("retry_used"):
        raw_paths["candidate_proposal_first_attempt"] = str(out / "openrouter_candidate_proposal.raw_response.json")
        raw_paths["candidate_proposal_retry"] = str(out / "openrouter_candidate_proposal_retry.raw_response.json")
    response_validation = write_response_validation_report(raw_paths, out)
    index["outputs"]["response_validation"] = {
        "json": response_validation["json"],
        "report": response_validation["report"],
    }

    # 5) Save final parsing pointers.
    index["outputs"]["candidate_parsing"] = {
        "parsed_candidates": candidate_validation["parsed_candidates"],
        "validation_json": candidate_validation["validation_json"],
        "validation_report": candidate_validation["validation_report"],
    }

    # 6) Optional deterministic verification by simulator/search.
    if args.csv_dir:
        verification = verify_llm_candidates(
            candidate_validation["parsed_candidates"],
            args.csv_dir,
            out / "candidate_verification",
            ad_model_name=args.ad_model,
            ego_track_id=args.ego_track_id,
        )
        index["outputs"]["candidate_verification"] = {
            "summary": verification["summary"],
            "table": verification["table"],
            "report": verification["report"],
        }

    index_path = out / "llm_validated_online_artifact_index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    print("CausalSensor4D public_release validated online LLM run finished.")
    print(f"Model: {args.model}")
    print(f"Output index: {index_path}")
    print(f"Candidate parse ok: {candidate_validation['validation'].get('parse_ok')}; candidates={candidate_validation['validation'].get('num_candidates')}; retry_used={retry_info.get('retry_used')}")
    for section, value in index["outputs"].items():
        print(f"{section}: {value}")


if __name__ == "__main__":
    main()
