from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, List
import json

import pandas as pd


def _read_json(path: Path) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_float(x: Any, nd: int = 3) -> str:
    try:
        if x is None or pd.isna(x):
            return "NA"
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def _load_llm_summary(llm_benchmark_dir: Optional[Path]) -> Dict[str, Any]:
    if llm_benchmark_dir is None:
        return {}
    summary_path = Path(llm_benchmark_dir) / "llm_verified_benchmark_summary.json"
    if not summary_path.exists():
        return {}
    return _read_json(summary_path)


def _llm_row_from_summary(llm: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    verification = llm.get("verification_summary", {}) if isinstance(llm, dict) else {}
    candidate_validation = llm.get("candidate_validation", {}) if isinstance(llm, dict) else {}
    diagnosis_quality = llm.get("diagnosis_quality", {}) if isinstance(llm, dict) else {}
    if not verification and not candidate_validation:
        return None
    return {
        "method": "llm_proposal_verified",
        "num_valid_runs": None,
        "num_best_found": verification.get("num_verified_clean_safe", verification.get("num_verified")),
        "attack_success_rate": verification.get("verified_clean_safe_rate", verification.get("verified_rate")),
        "mean_mfc_success_only": verification.get("mean_verified_cost"),
        "median_mfc_success_only": None,
        "min_mfc_success_only": verification.get("min_verified_cost"),
        "max_mfc_success_only": verification.get("max_verified_cost"),
        "mean_censored_mfc": None,
        "median_censored_mfc": None,
        "mean_risk_increase": None,
        "collision_rate": None,
        "hard_brake_rate": None,
        "num_edit_types_found": len(verification.get("verified_clean_safe_edit_type_counts", verification.get("verified_edit_type_counts", {})) or {}),
        "num_candidates": candidate_validation.get("num_candidates"),
        "num_valid_for_current_search": candidate_validation.get("num_valid_for_current_search"),
        "diagnosis_usable_without_manual_review": diagnosis_quality.get("usable_for_report_without_manual_review"),
        "note": "LLM proposes candidate interventions; CausalSensor4D verifies them deterministically. This is a proposal-level metric, not an exhaustive scene-level search baseline.",
    }


def _method_category(method: str) -> str:
    if method == "causal_guided":
        return "ours_strict_routing_ablation"
    if method == "causal_hybrid":
        return "ours_final_hybrid_search"
    if method in {"distance_all", "random_budget"}:
        return "non_causal_baseline"
    if method.endswith("_only"):
        return "single_edit_ablation"
    if method == "llm_proposal_verified":
        return "llm_proposal_module"
    return "other"


def generate_clean_safe_ablation_report(
    baseline_out_dir: str | Path,
    out_dir: str | Path,
    llm_benchmark_dir: Optional[str | Path] = None,
    title_version: str = "public_release",
) -> Dict[str, Any]:
    baseline_out_dir = Path(baseline_out_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = baseline_out_dir / "baseline_method_summary.csv"
    ranking_path = baseline_out_dir / "baseline_effectiveness_ranking.csv"
    edit_path = baseline_out_dir / "baseline_edit_type_table.csv"
    scene_matrix_path = baseline_out_dir / "baseline_scene_matrix.csv"
    metrics_path = baseline_out_dir / "baseline_comparison_metrics.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing baseline_method_summary.csv: {summary_path}")

    method_summary = pd.read_csv(summary_path)
    ranking = pd.read_csv(ranking_path) if ranking_path.exists() else pd.DataFrame()
    edit_table = pd.read_csv(edit_path) if edit_path.exists() else pd.DataFrame()
    scene_matrix = pd.read_csv(scene_matrix_path) if scene_matrix_path.exists() else pd.DataFrame()
    baseline_metrics = _read_json(metrics_path) if metrics_path.exists() else {}

    method_summary["method_category"] = method_summary["method"].map(_method_category)

    llm_summary = _load_llm_summary(Path(llm_benchmark_dir) if llm_benchmark_dir else None)
    llm_row = _llm_row_from_summary(llm_summary)

    combined = method_summary.copy()
    if llm_row is not None:
        combined = pd.concat([combined, pd.DataFrame([llm_row])], ignore_index=True, sort=False)
        combined["method_category"] = combined["method"].map(_method_category)

    # Useful derived comparisons.
    by_method = {r["method"]: r for _, r in method_summary.iterrows()}
    causal = by_method.get("causal_guided")
    causal_hybrid = by_method.get("causal_hybrid")
    primary_ours = causal_hybrid if causal_hybrid is not None else causal
    random = by_method.get("random_budget")
    distance = by_method.get("distance_all")
    single_methods = method_summary[method_summary["method"].isin(["lead_brake_only", "cut_in_only", "pedestrian_only"])]

    def _delta(field: str, base: Any, comp: Any) -> Optional[float]:
        try:
            if base is None or comp is None or pd.isna(base[field]) or pd.isna(comp[field]):
                return None
            return float(base[field]) - float(comp[field])
        except Exception:
            return None

    comparison = {
        "primary_ours_method": primary_ours["method"] if primary_ours is not None else None,
        "primary_ours_vs_random_attack_success_delta": _delta("attack_success_rate", primary_ours, random),
        "primary_ours_vs_random_mean_censored_mfc_delta": _delta("mean_censored_mfc", primary_ours, random),
        "primary_ours_vs_distance_attack_success_delta": _delta("attack_success_rate", primary_ours, distance),
        "primary_ours_vs_distance_mean_censored_mfc_delta": _delta("mean_censored_mfc", primary_ours, distance),
        "strict_causal_guided_vs_random_attack_success_delta": _delta("attack_success_rate", causal, random),
        "strict_causal_guided_vs_distance_attack_success_delta": _delta("attack_success_rate", causal, distance),
        "causal_hybrid_vs_strict_attack_success_delta": _delta("attack_success_rate", causal_hybrid, causal),
        "causal_hybrid_vs_strict_mean_censored_mfc_delta": _delta("mean_censored_mfc", causal_hybrid, causal),
        "single_edit_mean_attack_success": float(single_methods["attack_success_rate"].mean()) if not single_methods.empty else None,
        "single_edit_max_attack_success": float(single_methods["attack_success_rate"].max()) if not single_methods.empty else None,
    }

    # Compact edit-type table for report.
    if not edit_table.empty:
        edit_table = edit_table.sort_values(["method", "best_edit_name"])

    combined.to_csv(out_dir / "clean_safe_method_summary_with_llm.csv", index=False)
    if not ranking.empty:
        ranking.to_csv(out_dir / "clean_safe_effectiveness_ranking.csv", index=False)
    if not edit_table.empty:
        edit_table.to_csv(out_dir / "clean_safe_method_edit_type_table.csv", index=False)
    if not scene_matrix.empty:
        scene_matrix.to_csv(out_dir / "clean_safe_scene_matrix.csv", index=False)

    report = make_clean_safe_ablation_report(title_version, method_summary, ranking, edit_table, combined, llm_summary, comparison)
    (out_dir / "clean_safe_ablation_report.md").write_text(report, encoding="utf-8")

    payload = {
        "version": title_version,
        "baseline_out_dir": str(baseline_out_dir),
        "llm_benchmark_dir": str(llm_benchmark_dir) if llm_benchmark_dir else None,
        "num_methods": int(len(method_summary)),
        "method_summary": json.loads(method_summary.to_json(orient="records")),
        "llm_row_included": llm_row is not None,
        "llm_summary_used": llm_summary if llm_summary else None,
        "derived_comparisons": comparison,
        "outputs": {
            "report": str(out_dir / "clean_safe_ablation_report.md"),
            "method_summary_with_llm": str(out_dir / "clean_safe_method_summary_with_llm.csv"),
            "edit_type_table": str(out_dir / "clean_safe_method_edit_type_table.csv"),
            "scene_matrix": str(out_dir / "clean_safe_scene_matrix.csv"),
        },
        "baseline_metrics_payload": baseline_metrics,
    }
    (out_dir / "clean_safe_ablation_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def make_clean_safe_ablation_report(
    title_version: str,
    method_summary: pd.DataFrame,
    ranking: pd.DataFrame,
    edit_table: pd.DataFrame,
    combined: pd.DataFrame,
    llm_summary: Dict[str, Any],
    comparison: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append(f"# CausalSensor4D {title_version} Clean-Safe Baseline / Ablation Report")
    lines.append("")
    lines.append("## Purpose")
    lines.append("This report evaluates strict causal routing, the the hybrid causal-hybrid final search, non-causal baselines, and single-edit ablations on the clean-mined AV2 safe-to-failure subset.")
    lines.append("")
    lines.append("## Method-level metrics")
    cols = [
        "method",
        "method_category",
        "num_valid_runs",
        "num_best_found",
        "attack_success_rate",
        "mean_censored_mfc",
        "mean_mfc_success_only",
        "collision_rate",
        "hard_brake_rate",
        "num_edit_types_found",
    ]
    display = combined[[c for c in cols if c in combined.columns]].copy()
    if not display.empty:
        lines.append(display.to_markdown(index=False, floatfmt=".3f"))
    else:
        lines.append("No method summary available.")
    lines.append("")

    lines.append("## Effectiveness ranking for exhaustive search methods")
    if ranking.empty:
        lines.append("No ranking available.")
    else:
        rank_cols = ["effectiveness_rank", "method", "attack_success_rate", "mean_censored_mfc", "mean_mfc_success_only", "num_best_found", "num_valid_runs", "num_edit_types_found"]
        lines.append(ranking[[c for c in rank_cols if c in ranking.columns]].to_markdown(index=False, floatfmt=".3f"))
    lines.append("")

    lines.append("## Derived comparison highlights")
    for k, v in comparison.items():
        lines.append(f"- {k}: `{_fmt_float(v)}`")
    lines.append("")

    lines.append("## Method × edit-type breakdown")
    if edit_table.empty:
        lines.append("No edit-type table available.")
    else:
        cols2 = ["method", "best_edit_name", "num_scenes", "mean_mfc", "median_mfc", "collision_rate", "hard_brake_rate"]
        lines.append(edit_table[[c for c in cols2 if c in edit_table.columns]].to_markdown(index=False, floatfmt=".3f"))
    lines.append("")

    lines.append("## LLM proposal verification context")
    if llm_summary:
        v = llm_summary.get("verification_summary", {})
        c = llm_summary.get("candidate_validation", {})
        dq = llm_summary.get("diagnosis_quality", {})
        lines.append(f"- Candidate parse OK: `{c.get('parse_ok')}`")
        lines.append(f"- LLM candidates: `{c.get('num_candidates')}`")
        lines.append(f"- Valid for current search: `{c.get('num_valid_for_current_search')}`")
        lines.append(f"- Verified clean safe-to-failure: `{v.get('num_verified_clean_safe', v.get('num_verified'))}`")
        lines.append(f"- Verified clean safe-to-failure rate: `{_fmt_float(v.get('verified_clean_safe_rate', v.get('verified_rate')))}`")
        lines.append(f"- Mean verified cost: `{_fmt_float(v.get('mean_verified_cost'))}`")
        lines.append(f"- Diagnosis usable without manual review: `{dq.get('usable_for_report_without_manual_review')}`")
        lines.append("")
        lines.append("The LLM row should be interpreted as a proposal-level module, not as an exhaustive scene-level search baseline. The LLM proposes candidates; CausalSensor4D verifies them deterministically.")
    else:
        lines.append("No LLM benchmark summary was provided.")
    lines.append("")

    lines.append("## Report-ready interpretation")
    lines.append("Use this report as the clean-safe baseline/ablation experiment. the hybrid separates strict causal routing from the recommended causal-hybrid final search. The hybrid variant keeps causal relation priors but adds a distance fallback to avoid missing physically valid agents in dense AV2 scenes. The LLM row tests whether language-model proposals can be converted into verified counterfactual candidates without trusting the LLM as a numerical evaluator.")
    return "\n".join(lines)
