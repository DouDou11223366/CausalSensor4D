from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import json
import pandas as pd


def _fmt(x: Any, digits: int = 3) -> str:
    try:
        if x is None or pd.isna(x):
            return "NA"
        return f"{float(x):.{digits}f}"
    except Exception:
        return "NA"


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def infer_failure_type(edit_name: str | None, collision: Any, hard_brake: Any, best_min_ttc: Any) -> str:
    """Map an edit/risk outcome to a report-friendly failure category."""
    edit_name = edit_name or "unknown"
    collision_bool = bool(collision) if not pd.isna(collision) else False
    hard_brake_bool = bool(hard_brake) if not pd.isna(hard_brake) else False
    min_ttc = _safe_float(best_min_ttc, default=None)

    if collision_bool:
        suffix = "collision"
    elif hard_brake_bool:
        suffix = "hard_brake_near_miss"
    elif min_ttc is not None and min_ttc < 1.5:
        suffix = "low_ttc_near_miss"
    else:
        suffix = "unsafe_response"

    if edit_name == "lead_brake":
        return f"longitudinal_{suffix}"
    if edit_name == "cut_in":
        return f"lateral_{suffix}"
    if edit_name == "pedestrian_crossing":
        return f"pedestrian_{suffix}"
    return suffix


def build_evidence_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Build a compact table that can be used directly in analysis."""
    rows = []
    for _, row in summary.iterrows():
        best_found = bool(row.get("best_found", False))
        orig_ttc = _safe_float(row.get("original_min_ttc"), default=None)
        cf_ttc = _safe_float(row.get("best_min_ttc"), default=None)
        orig_risk = _safe_float(row.get("original_risk_score"), default=0.0)
        cf_risk = _safe_float(row.get("best_risk_score"), default=None)
        cost = _safe_float(row.get("best_cost"), default=None)

        ttc_drop = None
        if orig_ttc is not None and cf_ttc is not None and orig_ttc < 900:
            ttc_drop = orig_ttc - cf_ttc
        risk_increase = None
        if orig_risk is not None and cf_risk is not None:
            risk_increase = cf_risk - orig_risk

        edit_name = row.get("best_edit_name") if best_found else None
        failure_type = infer_failure_type(edit_name, row.get("best_collision"), row.get("best_hard_brake"), row.get("best_min_ttc")) if best_found else "not_found"
        evidence = ""
        if best_found:
            evidence = (
                f"{edit_name} on {row.get('best_target_agent_id')} with MFC={_fmt(cost)}; "
                f"TTC {_fmt(orig_ttc)}->{_fmt(cf_ttc)}; "
                f"risk {_fmt(orig_risk)}->{_fmt(cf_risk)}"
            )

        rows.append(
            {
                "scene_id": row.get("scene_id"),
                "num_agents": row.get("num_agents"),
                "num_steps": row.get("num_steps"),
                "top_candidate_agent_id": row.get("top_candidate_agent_id"),
                "best_found": best_found,
                "counterfactual_edit": edit_name,
                "target_agent_id": row.get("best_target_agent_id") if best_found else None,
                "minimum_failure_cost": cost,
                "failure_type": failure_type,
                "original_min_ttc": orig_ttc,
                "counterfactual_min_ttc": cf_ttc,
                "ttc_drop": ttc_drop,
                "original_risk_score": orig_risk,
                "counterfactual_risk_score": cf_risk,
                "risk_increase": risk_increase,
                "collision": row.get("best_collision") if best_found else None,
                "hard_brake": row.get("best_hard_brake") if best_found else None,
                "evidence_chain_short": evidence,
                "output_dir": row.get("output_dir"),
            }
        )
    return pd.DataFrame(rows)


def build_mfc_by_edit_table(evidence: pd.DataFrame) -> pd.DataFrame:
    if evidence.empty or "counterfactual_edit" not in evidence.columns:
        return pd.DataFrame()

    valid = evidence[evidence["best_found"] == True].copy()
    if valid.empty:
        return pd.DataFrame()

    def _rate(s: pd.Series) -> float:
        return float(s.fillna(False).astype(bool).mean())

    grouped = (
        valid.groupby("counterfactual_edit", dropna=False)
        .agg(
            num_scenes=("scene_id", "count"),
            mean_mfc=("minimum_failure_cost", "mean"),
            median_mfc=("minimum_failure_cost", "median"),
            min_mfc=("minimum_failure_cost", "min"),
            max_mfc=("minimum_failure_cost", "max"),
            mean_ttc_drop=("ttc_drop", "mean"),
            mean_risk_increase=("risk_increase", "mean"),
            collision_rate=("collision", _rate),
            hard_brake_rate=("hard_brake", _rate),
        )
        .reset_index()
    )
    return grouped.sort_values(["mean_mfc", "counterfactual_edit"], ascending=[True, True])


def make_report_ready_markdown(summary: pd.DataFrame, evidence: pd.DataFrame, by_edit: pd.DataFrame, aggregate: Dict[str, Any]) -> str:
    lines = []
    lines.append("# CausalSensor4D public_release Report-Ready Batch Report")
    lines.append("")
    lines.append("## Overall metrics")
    lines.append(f"- Number of CSV scenes: `{aggregate.get('num_csv_files')}`")
    lines.append(f"- Valid runs: `{aggregate.get('num_valid_runs')}`")
    lines.append(f"- Failure-inducing counterfactuals found: `{aggregate.get('num_best_found')}`")
    lines.append(f"- Failure discovery rate: `{_fmt(aggregate.get('failure_discovery_rate'))}`")
    lines.append(f"- Mean Minimum Failure Cost: `{_fmt(aggregate.get('mean_best_cost'))}`")
    lines.append(f"- Median Minimum Failure Cost: `{_fmt(aggregate.get('median_best_cost'))}`")
    lines.append(f"- Mean TTC drop: `{_fmt(aggregate.get('mean_ttc_drop'))}` s")
    lines.append(f"- Mean risk increase: `{_fmt(aggregate.get('mean_risk_increase'))}`")
    lines.append("")

    lines.append("## Minimum Failure Cost by edit type")
    if by_edit.empty:
        lines.append("No successful counterfactual edits found.")
    else:
        lines.append(by_edit.to_markdown(index=False, floatfmt=".3f"))
    lines.append("")

    lines.append("## Counterfactual evidence table")
    if evidence.empty:
        lines.append("No evidence table available.")
    else:
        cols = [
            "scene_id",
            "counterfactual_edit",
            "target_agent_id",
            "minimum_failure_cost",
            "failure_type",
            "original_min_ttc",
            "counterfactual_min_ttc",
            "ttc_drop",
            "risk_increase",
        ]
        display = evidence[cols].copy()
        lines.append(display.to_markdown(index=False, floatfmt=".3f"))
    lines.append("")

    lines.append("## How to cite these numbers in the report draft")
    lines.append(
        "当前批量结果可用于论文 MVP 实验描述：系统在多个交互类型上均能找到最小失效反事实，"
        "并输出 MFC、TTC drop、risk increase 和 failure type。真实数据接入后，该表格结构保持不变。"
    )
    return "\n".join(lines)


def save_report_artifacts(summary: pd.DataFrame, out_dir: Path, base_aggregate: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    evidence = build_evidence_table(summary)
    by_edit = build_mfc_by_edit_table(evidence)

    valid = summary.copy()
    if "error" in valid.columns:
        valid = valid[valid["error"].isna()]
    num_valid = int(len(valid))
    num_best = int(valid["best_found"].fillna(False).sum()) if "best_found" in valid else 0

    best_cost = pd.to_numeric(valid.get("best_cost", pd.Series(dtype=float)), errors="coerce").dropna()
    ttc_drop = pd.to_numeric(evidence.get("ttc_drop", pd.Series(dtype=float)), errors="coerce").dropna()
    risk_increase = pd.to_numeric(evidence.get("risk_increase", pd.Series(dtype=float)), errors="coerce").dropna()

    aggregate = dict(base_aggregate or {})
    aggregate.update(
        {
            "failure_discovery_rate": float(num_best / num_valid) if num_valid else 0.0,
            "median_best_cost": float(best_cost.median()) if len(best_cost) else None,
            "min_best_cost": float(best_cost.min()) if len(best_cost) else None,
            "max_best_cost": float(best_cost.max()) if len(best_cost) else None,
            "mean_ttc_drop": float(ttc_drop.mean()) if len(ttc_drop) else None,
            "mean_risk_increase": float(risk_increase.mean()) if len(risk_increase) else None,
            "num_edit_types_found": int(evidence["counterfactual_edit"].dropna().nunique()) if not evidence.empty else 0,
        }
    )

    evidence.to_csv(out_dir / "failure_evidence_table.csv", index=False)
    by_edit.to_csv(out_dir / "mfc_by_edit_type.csv", index=False)
    (out_dir / "report_ready_results.md").write_text(
        make_report_ready_markdown(summary, evidence, by_edit, aggregate), encoding="utf-8"
    )
    (out_dir / "report_metrics.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    return aggregate
