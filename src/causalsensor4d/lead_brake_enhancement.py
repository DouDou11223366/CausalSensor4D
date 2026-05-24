from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional
import json
import pandas as pd


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def summarize_mfc_dir(mfc_out_dir: str | Path) -> Dict[str, Any]:
    out = Path(mfc_out_dir)
    summary = _safe_read_csv(out / "batch_summary.csv")
    mfc_by_edit = _safe_read_csv(out / "mfc_by_edit_type.csv")
    result: Dict[str, Any] = {
        "mfc_out_dir": str(out),
        "exists": out.exists(),
        "num_scenes": 0,
        "num_best_found": 0,
        "failure_discovery_rate": None,
        "mean_mfc": None,
        "best_edit_counts": {},
        "lead_brake_best_count": 0,
        "lead_brake_mean_mfc": None,
        "lead_brake_collision_rate": None,
        "lead_brake_hard_brake_rate": None,
    }
    if not summary.empty:
        result["num_scenes"] = int(len(summary))
        if "best_found" in summary.columns:
            found = summary[summary["best_found"].fillna(False).astype(bool)].copy()
        else:
            found = summary[summary.get("best_cost", pd.Series(dtype=float)).notna()].copy()
        result["num_best_found"] = int(len(found))
        result["failure_discovery_rate"] = float(len(found) / len(summary)) if len(summary) else 0.0
        if "best_cost" in found.columns and found["best_cost"].dropna().size:
            result["mean_mfc"] = float(found["best_cost"].dropna().mean())
        if "best_edit_name" in found.columns:
            counts = found["best_edit_name"].fillna("not_found").value_counts().to_dict()
            result["best_edit_counts"] = {str(k): int(v) for k, v in counts.items()}
            result["lead_brake_best_count"] = int(counts.get("lead_brake", 0))
    if not mfc_by_edit.empty and "counterfactual_edit" in mfc_by_edit.columns:
        lead = mfc_by_edit[mfc_by_edit["counterfactual_edit"].astype(str) == "lead_brake"]
        if not lead.empty:
            row = lead.iloc[0]
            for key, col in [
                ("lead_brake_mean_mfc", "mean_mfc"),
                ("lead_brake_collision_rate", "collision_rate"),
                ("lead_brake_hard_brake_rate", "hard_brake_rate"),
            ]:
                if col in row and pd.notna(row[col]):
                    result[key] = float(row[col])
    return result


def write_lead_brake_enhancement_report(
    current_mfc_out_dir: str | Path,
    report_out_dir: str | Path,
    baseline_mfc_out_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    report_out = Path(report_out_dir)
    report_out.mkdir(parents=True, exist_ok=True)
    current = summarize_mfc_dir(current_mfc_out_dir)
    baseline = summarize_mfc_dir(baseline_mfc_out_dir) if baseline_mfc_out_dir else None
    aggregate: Dict[str, Any] = {
        "version": "public_release",
        "purpose": "lead_brake enhancement audit",
        "current": current,
        "baseline": baseline,
    }
    if baseline:
        aggregate["delta"] = {
            "lead_brake_best_count_delta": current.get("lead_brake_best_count", 0) - baseline.get("lead_brake_best_count", 0),
            "failure_discovery_rate_delta": (
                None if current.get("failure_discovery_rate") is None or baseline.get("failure_discovery_rate") is None
                else float(current["failure_discovery_rate"] - baseline["failure_discovery_rate"])
            ),
            "mean_mfc_delta": (
                None if current.get("mean_mfc") is None or baseline.get("mean_mfc") is None
                else float(current["mean_mfc"] - baseline["mean_mfc"])
            ),
        }

    (report_out / "lead_brake_enhancement_summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# CausalSensor4D public_release Lead-Brake Enhancement Report",
        "",
        "## Purpose",
        "public_release expands longitudinal counterfactual search and balances causal candidates so that lead-following agents are not suppressed in dense AV2 scenes.",
        "",
        "## Current run",
        f"- MFC output dir: `{current.get('mfc_out_dir')}`",
        f"- Scenes: `{current.get('num_scenes')}`",
        f"- Failure discovery rate: `{current.get('failure_discovery_rate')}`",
        f"- Mean MFC: `{current.get('mean_mfc')}`",
        f"- Best edit counts: `{current.get('best_edit_counts')}`",
        f"- Lead-brake best count: `{current.get('lead_brake_best_count')}`",
        f"- Lead-brake mean MFC: `{current.get('lead_brake_mean_mfc')}`",
    ]
    if baseline:
        lines += [
            "",
            "## Baseline comparison",
            f"- Baseline MFC output dir: `{baseline.get('mfc_out_dir')}`",
            f"- Baseline best edit counts: `{baseline.get('best_edit_counts')}`",
            f"- Baseline lead-brake best count: `{baseline.get('lead_brake_best_count')}`",
            f"- Lead-brake count delta: `{aggregate['delta']['lead_brake_best_count_delta']}`",
            f"- Failure discovery rate delta: `{aggregate['delta']['failure_discovery_rate_delta']}`",
            f"- Mean MFC delta: `{aggregate['delta']['mean_mfc_delta']}`",
        ]
    lines += [
        "",
        "## Interpretation",
        "If lead_brake remains under-represented after public_release, the next step should be a lane/heading-aware longitudinal evaluator rather than only a larger parameter grid.",
    ]
    (report_out / "lead_brake_enhancement_report.md").write_text("\n".join(lines), encoding="utf-8")
    return aggregate
