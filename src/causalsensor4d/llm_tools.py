from __future__ import annotations

"""LLM-ready diagnosis utilities for CausalSensor4D public_release.

This module does not call any online API by default.  It converts CausalSensor4D
outputs into compact, structured prompts and offline diagnosis reports that can
be pasted into ChatGPT, Qwen, DeepSeek, Llama, or any OpenAI-compatible/local LLM.

The design goal is to make the LLM an explanation / candidate-proposal layer,
not the source of truth.  All numeric facts must come from the counterfactual
search outputs; the LLM is only asked to verbalize evidence chains or suggest
future candidate variables.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import math

import pandas as pd


@dataclass
class LLMDiagnosisConfig:
    project_name: str = "CausalSensor4D"
    max_scene_rows: int = 24
    max_prompt_chars: int = 22000
    include_scene_matrix: bool = True
    include_candidate_proposal_prompt: bool = True


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def _read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_csv(path)
    return None


def _read_text_if_exists(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def summarize_ad_model_results(result_dir: str | Path) -> Dict[str, Any]:
    """Read AD model comparison outputs and create a compact JSON summary."""
    result_dir = Path(result_dir)
    summary_df = _read_csv_if_exists(result_dir / "ad_model_comparison_summary.csv")
    ranking_df = _read_csv_if_exists(result_dir / "ad_model_robustness_ranking.csv")
    scene_df = _read_csv_if_exists(result_dir / "all_ad_model_scene_results.csv")
    report_md = _read_text_if_exists(result_dir / "ad_model_comparison_report.md")

    summary: Dict[str, Any] = {
        "result_dir": str(result_dir),
        "has_summary": summary_df is not None,
        "has_ranking": ranking_df is not None,
        "has_scene_results": scene_df is not None,
    }

    if ranking_df is not None and len(ranking_df) > 0:
        ranking_records = []
        for _, row in ranking_df.iterrows():
            model_name = row.get("ad_model") or row.get("AD model") or row.get("planner") or row.get("model")
            ranking_records.append({
                "rank": _safe_float(row.get("robustness_rank")),
                "ad_model": str(model_name),
                "attack_success_rate": _safe_float(row.get("attack_success_rate")),
                "mean_censored_mfc": _safe_float(row.get("mean_censored_mfc")),
                "mean_mfc_success_only": _safe_float(row.get("mean_mfc_success_only")),
                "collision_rate": _safe_float(row.get("collision_rate")),
                "hard_brake_rate": _safe_float(row.get("hard_brake_rate")),
                "num_edit_types_found": _safe_float(row.get("num_edit_types_found")),
            })
        summary["robustness_ranking"] = ranking_records

    if summary_df is not None and len(summary_df) > 0:
        summary["model_level_metrics"] = summary_df.to_dict(orient="records")

    if scene_df is not None and len(scene_df) > 0:
        # Keep the most informative failures first: low cost and successful best_found.
        keep_cols = [
            c for c in [
                "scene_id", "ad_model", "model_family", "original_behavior_label",
                "original_collision", "original_hard_brake", "original_min_ttc",
                "original_risk_score", "top_candidate_agent_id", "best_found",
                "best_edit_name", "best_target_agent_id", "best_cost", "best_collision",
                "best_hard_brake", "best_min_ttc", "best_risk_score",
            ] if c in scene_df.columns
        ]
        tmp = scene_df.copy()
        if "best_found" in tmp.columns:
            tmp = tmp.sort_values(by=["best_found", "best_cost"], ascending=[False, True], na_position="last")
        scene_records = tmp[keep_cols].head(48).to_dict(orient="records")
        summary["representative_scene_results"] = scene_records
        summary["num_scene_rows_total"] = int(len(scene_df))

    if report_md:
        summary["source_report_excerpt"] = report_md[:6000]
    return summary


def make_llm_diagnosis_prompt(summary: Dict[str, Any], config: Optional[LLMDiagnosisConfig] = None) -> str:
    config = config or LLMDiagnosisConfig()
    compact_json = json.dumps(summary, ensure_ascii=False, indent=2)
    prompt = f"""# LLM Task: Counterfactual Failure Diagnosis for {config.project_name}

You are given structured outputs from a counterfactual autonomous-driving diagnosis system.
Do not invent numbers. Use only the JSON evidence below. If a number is missing, say it is unavailable.

## System goal
The system searches for the minimum-cost counterfactual intervention that changes a driving model from a safer factual rollout to a failure or near-miss. The key metric is Minimum Failure Cost (MFC). Lower MFC means the model is easier to break under the current edit library. For robustness ranking, use mean_censored_mfc and attack_success_rate together.

## What you must produce
1. A concise English research-style diagnosis paragraph.
2. A Chinese research-style diagnosis paragraph.
3. A table-style summary of which AD model is most robust and why.
4. A list of the most important failure modes: lead_brake, cut_in, pedestrian_crossing, or other.
5. A short caution paragraph explaining limitations of the current experiment.
6. Three suggestions for next counterfactual variables to add.

## Evidence JSON
```json
{compact_json}
```
"""
    if len(prompt) > config.max_prompt_chars:
        prompt = prompt[: config.max_prompt_chars] + "\n\n[TRUNCATED: prompt shortened to stay within configured length.]\n"
    return prompt


def make_candidate_proposal_prompt(scene_summary: Dict[str, Any], config: Optional[LLMDiagnosisConfig] = None) -> str:
    config = config or LLMDiagnosisConfig()
    compact_json = json.dumps(scene_summary, ensure_ascii=False, indent=2)
    prompt = f"""# LLM Task: Causal Counterfactual Candidate Proposal for {config.project_name}

You are given a structured causal scene / batch summary. Propose candidate counterfactual edits, but do not claim they are valid until the simulator verifies them.

## Allowed edit families
- lead_brake: longitudinal braking or speed reduction of a lead vehicle.
- cut_in: lateral movement of an adjacent vehicle into the ego lane/path.
- pedestrian_crossing: pedestrian enters ego future path earlier or faster.
- future extensions: occlusion increase, visibility degradation, traffic light change, lane marking degradation.

## Output format
Return a JSON list. Each item must include:
- target_agent_id or target_relation
- edit_family
- parameter_suggestion
- expected_failure_mode
- physical_plausibility_reason
- why_this_is_minimal

## Evidence JSON
```json
{compact_json}
```
"""
    if len(prompt) > config.max_prompt_chars:
        prompt = prompt[: config.max_prompt_chars] + "\n\n[TRUNCATED]\n"
    return prompt


def make_offline_diagnosis(summary: Dict[str, Any]) -> str:
    """Generate a deterministic diagnosis without calling an LLM."""
    ranking = summary.get("robustness_ranking", []) or []
    lines: List[str] = []
    lines.append("# CausalSensor4D public_release LLM-Ready Diagnosis Summary")
    lines.append("")
    lines.append("## Deterministic summary")
    if ranking:
        best = ranking[0]
        worst = ranking[-1]
        lines.append(
            f"- Most robust model under censored MFC: `{best.get('ad_model')}` "
            f"(attack_success_rate={best.get('attack_success_rate')}, mean_censored_mfc={best.get('mean_censored_mfc')})."
        )
        lines.append(
            f"- Most vulnerable model under censored MFC: `{worst.get('ad_model')}` "
            f"(attack_success_rate={worst.get('attack_success_rate')}, mean_censored_mfc={worst.get('mean_censored_mfc')})."
        )
    else:
        lines.append("- No robustness ranking was found in the provided result directory.")

    scene_results = summary.get("representative_scene_results", []) or []
    edit_counts: Dict[str, int] = {}
    for row in scene_results:
        edit = row.get("best_edit_name")
        if edit and str(edit).lower() != "nan":
            edit_counts[str(edit)] = edit_counts.get(str(edit), 0) + 1
    if edit_counts:
        lines.append("- Representative failure edit counts: " + ", ".join(f"{k}={v}" for k, v in sorted(edit_counts.items())))

    lines.append("")
    lines.append("## Why the LLM layer is useful")
    lines.append("The LLM should not replace numeric verification. It is used to convert verified MFC results into a readable causal evidence chain and to propose new candidate variables for the search module.")
    lines.append("")
    lines.append("## Recommended next variables")
    lines.append("1. Occlusion increase: move an occluding vehicle to reduce pedestrian/vehicle visibility.")
    lines.append("2. Reaction-delay perturbation: increase ego or surrounding-agent response latency.")
    lines.append("3. Visibility degradation: reduce contrast, brightness, or object visible area in sensor-level extensions.")
    return "\n".join(lines) + "\n"


def save_llm_artifacts(result_dir: str | Path, out_dir: str | Path, config: Optional[LLMDiagnosisConfig] = None) -> Dict[str, str]:
    config = config or LLMDiagnosisConfig()
    result_dir = Path(result_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_ad_model_results(result_dir)
    summary_path = out_dir / "llm_evidence_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    diagnosis_prompt = make_llm_diagnosis_prompt(summary, config)
    diagnosis_prompt_path = out_dir / "llm_diagnosis_prompt.md"
    diagnosis_prompt_path.write_text(diagnosis_prompt, encoding="utf-8")

    candidate_prompt = make_candidate_proposal_prompt(summary, config)
    candidate_prompt_path = out_dir / "llm_candidate_proposal_prompt.md"
    candidate_prompt_path.write_text(candidate_prompt, encoding="utf-8")

    offline_report = make_offline_diagnosis(summary)
    offline_report_path = out_dir / "offline_diagnosis_report.md"
    offline_report_path.write_text(offline_report, encoding="utf-8")

    index = {
        "result_dir": str(result_dir),
        "llm_evidence_summary": str(summary_path),
        "llm_diagnosis_prompt": str(diagnosis_prompt_path),
        "llm_candidate_proposal_prompt": str(candidate_prompt_path),
        "offline_diagnosis_report": str(offline_report_path),
        "note": "Paste the prompt files into an LLM, or use them with a local/OpenAI-compatible endpoint. The code does not call any API by default.",
    }
    (out_dir / "llm_artifact_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return index
