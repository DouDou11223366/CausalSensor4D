from __future__ import annotations

"""Diagnostics for public_release heading-aware longitudinal reasoning."""

from pathlib import Path
from typing import Any, Dict, List
import json

import pandas as pd


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _rate(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return float(series.fillna(False).astype(bool).mean())


def _summarize_lead_candidate_tables(candidate_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for table_path in sorted(candidate_root.glob("*/*/candidate_table.csv")):
        parts = table_path.parts
        # .../per_method/<method>/<scene>/candidate_table.csv
        method = table_path.parent.parent.name
        scene_id = table_path.parent.name
        df = _safe_read_csv(table_path)
        if df.empty:
            rows.append({
                "method": method,
                "scene_id": scene_id,
                "num_rows": 0,
                "num_lead_rows": 0,
                "num_lead_failures": 0,
                "lead_failure_rate": None,
            })
            continue
        edit = df.get("edit_name", pd.Series([], dtype=str)).astype(str)
        lead = df[edit == "lead_brake"].copy()
        row: Dict[str, Any] = {
            "method": method,
            "scene_id": scene_id,
            "num_rows": int(len(df)),
            "num_lead_rows": int(len(lead)),
            "num_lead_failures": int(lead.get("failure", pd.Series([], dtype=bool)).fillna(False).astype(bool).sum()) if not lead.empty else 0,
            "lead_failure_rate": _rate(lead.get("failure", pd.Series([], dtype=bool))) if not lead.empty else None,
        }
        for col in ["original_longitudinal", "original_lateral", "original_gap", "original_closing_speed", "min_ttc", "cost"]:
            if col in lead.columns and not lead.empty:
                vals = pd.to_numeric(lead[col], errors="coerce").dropna()
                row[f"lead_{col}_median"] = float(vals.median()) if len(vals) else None
                row[f"lead_{col}_mean"] = float(vals.mean()) if len(vals) else None
        rows.append(row)
    return pd.DataFrame(rows)


def generate_longitudinal_diagnostic(baseline_out_dir: str | Path, out_dir: str | Path) -> Dict[str, Any]:
    baseline_out_dir = Path(baseline_out_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_root = baseline_out_dir / "per_method"

    scene_results = _safe_read_csv(baseline_out_dir / "all_baseline_scene_results.csv")
    cand_summary = _summarize_lead_candidate_tables(candidate_root)
    cand_summary.to_csv(out_dir / "lead_candidate_table_summary.csv", index=False)

    payload: Dict[str, Any] = {
        "version": "public_release",
        "purpose": "Audit heading-aware longitudinal lead-brake candidate coverage and verified failures.",
        "baseline_out_dir": str(baseline_out_dir),
        "candidate_root": str(candidate_root),
        "scene_results_found": not scene_results.empty,
        "candidate_tables_summarized": int(len(cand_summary)),
    }

    if not scene_results.empty and "method" in scene_results.columns:
        best_rows = scene_results[scene_results.get("best_found", pd.Series(False, index=scene_results.index)).fillna(False).astype(bool)].copy()
        best_edit_counts = (
            best_rows.groupby(["method", "best_edit_name"]).size().reset_index(name="num_scenes")
            if not best_rows.empty and "best_edit_name" in best_rows.columns else pd.DataFrame()
        )
        best_edit_counts.to_csv(out_dir / "best_edit_counts.csv", index=False)
        payload["best_lead_brake_counts"] = {
            str(row["method"]): int(row["num_scenes"])
            for _, row in best_edit_counts[best_edit_counts["best_edit_name"].astype(str) == "lead_brake"].iterrows()
        } if not best_edit_counts.empty else {}
        payload["method_attack_success"] = {
            str(m): float(g.get("best_found", pd.Series([], dtype=bool)).fillna(False).astype(bool).mean())
            for m, g in scene_results.groupby("method")
        }

    if not cand_summary.empty:
        method_summary = cand_summary.groupby("method").agg(
            num_scenes=("scene_id", "count"),
            total_rows=("num_rows", "sum"),
            total_lead_rows=("num_lead_rows", "sum"),
            total_lead_failures=("num_lead_failures", "sum"),
            mean_lead_rows_per_scene=("num_lead_rows", "mean"),
        ).reset_index()
        method_summary["lead_failure_rate_over_lead_rows"] = method_summary.apply(
            lambda r: None if r["total_lead_rows"] == 0 else float(r["total_lead_failures"] / r["total_lead_rows"]), axis=1
        )
        method_summary.to_csv(out_dir / "lead_candidate_method_summary.csv", index=False)
        payload["lead_candidate_method_summary"] = method_summary.to_dict(orient="records")

    (out_dir / "longitudinal_diagnostic_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# CausalSensor4D public_release Longitudinal Diagnostic",
        "",
        "## Purpose",
        "This report checks whether heading-aware longitudinal geometry changes lead-brake candidate coverage and verified failures.",
        "",
        f"- Baseline output: `{baseline_out_dir}`",
        f"- Candidate tables summarized: `{payload.get('candidate_tables_summarized')}`",
        f"- Best lead-brake counts: `{payload.get('best_lead_brake_counts', {})}`",
        "",
        "## Lead candidate method summary",
    ]
    method_csv = out_dir / "lead_candidate_method_summary.csv"
    method_df = _safe_read_csv(method_csv)
    if method_df.empty:
        lines.append("No candidate summary available.")
    else:
        lines.append(method_df.to_markdown(index=False, floatfmt=".3f"))
    lines += [
        "",
        "## Interpretation",
        "If `lead_brake_only` and the hybrid/distance best-edit lead counts remain near zero, the limitation is no longer only global-coordinate geometry; it likely reflects the delayed planner/risk definition or the clean-safe scene distribution being dominated by lateral conflicts.",
    ]
    (out_dir / "longitudinal_diagnostic_report.md").write_text("\n".join(lines), encoding="utf-8")
    return payload
