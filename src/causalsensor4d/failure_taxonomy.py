from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json
import math

import pandas as pd


DEFAULT_TTC_THRESHOLDS = [0.5, 1.0, 1.5, 2.0]
DEFAULT_FAILURE_TTC_THRESHOLD = 1.5
VERSION = "public_release"


def _safe_bool(x: Any) -> bool:
    if pd.isna(x):
        return False
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    return s in {"1", "true", "yes", "y", "t"}


def _safe_float(x: Any, default: float = math.nan) -> float:
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (pd.Series, pd.Index)):
        return [_json_safe(v) for v in value.tolist()]
    if pd.isna(value) if not isinstance(value, (list, tuple, dict)) else False:
        return None
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _resolve_baseline_out_dir(run_dir: Optional[Path] = None, baseline_out_dir: Optional[Path] = None) -> Path:
    if baseline_out_dir:
        return Path(baseline_out_dir)
    if run_dir is None:
        raise ValueError("Either run_dir or baseline_out_dir must be provided.")
    run_dir = Path(run_dir)
    if (run_dir / "baseline_ablation").exists():
        return run_dir / "baseline_ablation"
    return run_dir


def _resolve_candidate_root(baseline_out_dir: Path) -> Path:
    baseline_out_dir = Path(baseline_out_dir)
    if (baseline_out_dir / "per_method").exists():
        return baseline_out_dir / "per_method"
    return baseline_out_dir


def _scene_id_from_table_path(path: Path) -> str:
    # Expected: per_method/<method>/<scene_id>/candidate_table.csv
    try:
        return path.parent.name
    except Exception:
        return "unknown_scene"


def _method_from_table_path(path: Path, candidate_root: Path) -> str:
    try:
        rel = path.relative_to(candidate_root)
        return rel.parts[0]
    except Exception:
        return path.parent.parent.name if path.parent.parent else "unknown_method"


def _find_candidate_tables(candidate_root: Path) -> List[Path]:
    candidate_root = Path(candidate_root)
    if not candidate_root.exists():
        return []
    return sorted(candidate_root.rglob("candidate_table.csv"))


def _trigger_combo(collision: bool, low_ttc: bool, hard_brake: bool, include_hard_brake: bool = True) -> str:
    parts: List[str] = []
    if collision:
        parts.append("collision")
    if low_ttc:
        parts.append("low_ttc")
    if include_hard_brake and hard_brake:
        parts.append("hard_brake")
    if not parts:
        return "no_failure_trigger"
    return "+".join(parts)


def _severity_band(collision: bool, min_ttc: float, hard_brake: bool, ttc_threshold: float = DEFAULT_FAILURE_TTC_THRESHOLD) -> str:
    if collision:
        return "collision"
    if math.isfinite(min_ttc) and min_ttc < 999.0:
        if min_ttc < 0.5:
            return "critical_ttc_lt_0_5"
        if min_ttc < 1.0:
            return "severe_ttc_0_5_to_1_0"
        if min_ttc < ttc_threshold:
            return "near_miss_ttc_1_0_to_1_5"
        if min_ttc < 2.0:
            return "warning_ttc_1_5_to_2_0"
    if hard_brake:
        return "hard_brake_only"
    return "safe_or_no_valid_ttc"


def _classify_best_row(row: pd.Series, ttc_threshold: float = DEFAULT_FAILURE_TTC_THRESHOLD) -> Dict[str, Any]:
    found = _safe_bool(row.get("best_found", False))
    collision = _safe_bool(row.get("best_collision", False)) if found else False
    hard_brake = _safe_bool(row.get("best_hard_brake", False)) if found else False
    min_ttc = _safe_float(row.get("best_min_ttc", math.nan), math.nan) if found else math.nan
    low_ttc = bool(found and math.isfinite(min_ttc) and min_ttc < ttc_threshold)
    collision_or_low_ttc = bool(collision or low_ttc)
    trigger = _trigger_combo(collision, low_ttc, hard_brake)
    strict_trigger = _trigger_combo(collision, low_ttc, False, include_hard_brake=False)
    if not found:
        trigger = "not_found"
        strict_trigger = "not_found"
    return {
        "taxonomy_best_found": found,
        "taxonomy_collision": collision,
        "taxonomy_hard_brake": hard_brake,
        "taxonomy_low_ttc": low_ttc,
        "taxonomy_collision_or_low_ttc": collision_or_low_ttc,
        "taxonomy_trigger_combo": trigger,
        "taxonomy_strict_failure_combo": strict_trigger,
        "taxonomy_severity_band": _severity_band(collision, min_ttc, hard_brake, ttc_threshold) if found else "not_found",
    }


def _classify_candidate_df(df: pd.DataFrame, ttc_threshold: float = DEFAULT_FAILURE_TTC_THRESHOLD) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        for col in [
            "taxonomy_collision",
            "taxonomy_hard_brake",
            "taxonomy_low_ttc",
            "taxonomy_trigger_combo",
            "taxonomy_strict_failure_combo",
            "taxonomy_severity_band",
        ]:
            out[col] = []
        return out
    out["taxonomy_collision"] = out.get("collision", False).map(_safe_bool)
    out["taxonomy_hard_brake"] = out.get("hard_brake", False).map(_safe_bool)
    out["_min_ttc_num"] = pd.to_numeric(out.get("min_ttc", math.nan), errors="coerce")
    out["taxonomy_low_ttc"] = out["_min_ttc_num"].lt(float(ttc_threshold))
    out["taxonomy_trigger_combo"] = [
        _trigger_combo(bool(c), bool(l), bool(h))
        for c, l, h in zip(out["taxonomy_collision"], out["taxonomy_low_ttc"], out["taxonomy_hard_brake"])
    ]
    out["taxonomy_strict_failure_combo"] = [
        _trigger_combo(bool(c), bool(l), False, include_hard_brake=False)
        for c, l in zip(out["taxonomy_collision"], out["taxonomy_low_ttc"])
    ]
    out["taxonomy_severity_band"] = [
        _severity_band(bool(c), _safe_float(t, math.nan), bool(h), ttc_threshold)
        for c, t, h in zip(out["taxonomy_collision"], out["_min_ttc_num"], out["taxonomy_hard_brake"])
    ]
    out.drop(columns=["_min_ttc_num"], inplace=True, errors="ignore")
    return out


def _load_all_baseline_rows(baseline_out_dir: Path) -> pd.DataFrame:
    # Prefer the merged output. If not present, reconstruct from isolated method runs.
    baseline_out_dir = Path(baseline_out_dir)
    merged = baseline_out_dir / "all_baseline_scene_results.csv"
    if merged.exists():
        return _read_csv_if_exists(merged)
    rows = []
    for p in sorted((baseline_out_dir / "_isolated_method_runs").glob("*/all_baseline_scene_results.csv")):
        df = _read_csv_if_exists(p)
        if not df.empty:
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _load_method_summary(baseline_out_dir: Path) -> pd.DataFrame:
    p = Path(baseline_out_dir) / "baseline_method_summary.csv"
    if p.exists():
        return _read_csv_if_exists(p)
    rows = []
    for q in sorted((Path(baseline_out_dir) / "_isolated_method_runs").glob("*/baseline_method_summary.csv")):
        df = _read_csv_if_exists(q)
        if not df.empty:
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _best_taxonomy_table(all_rows: pd.DataFrame, ttc_threshold: float) -> pd.DataFrame:
    if all_rows.empty:
        return pd.DataFrame()
    rows = []
    for _, row in all_rows.iterrows():
        item = row.to_dict()
        item.update(_classify_best_row(row, ttc_threshold=ttc_threshold))
        rows.append(item)
    return pd.DataFrame(rows)


def _agg_bool_rate(s: pd.Series) -> float:
    if len(s) == 0:
        return 0.0
    return float(s.fillna(False).map(_safe_bool).mean())


def _safe_count_bool(s: pd.Series) -> int:
    if len(s) == 0:
        return 0
    return int(s.fillna(False).map(_safe_bool).sum())


def _method_failure_type_summary(best_tax: pd.DataFrame) -> pd.DataFrame:
    if best_tax.empty or "method" not in best_tax.columns:
        return pd.DataFrame()
    rows = []
    for method, g in best_tax.groupby("method", dropna=False):
        found = g["taxonomy_best_found"].fillna(False).map(_safe_bool)
        fg = g[found].copy()
        num_scenes = int(len(g))
        num_found = int(found.sum())
        collision = _safe_count_bool(fg.get("taxonomy_collision", pd.Series(dtype=bool)))
        low_ttc = _safe_count_bool(fg.get("taxonomy_low_ttc", pd.Series(dtype=bool)))
        hard = _safe_count_bool(fg.get("taxonomy_hard_brake", pd.Series(dtype=bool)))
        collision_or_hard = int(((fg.get("taxonomy_collision", False).map(_safe_bool) if "taxonomy_collision" in fg else pd.Series([], dtype=bool)) | (fg.get("taxonomy_hard_brake", False).map(_safe_bool) if "taxonomy_hard_brake" in fg else pd.Series([], dtype=bool))).sum()) if len(fg) else 0
        low_ttc_only = int(((fg.get("taxonomy_low_ttc", False).map(_safe_bool)) & ~(fg.get("taxonomy_collision", False).map(_safe_bool)) & ~(fg.get("taxonomy_hard_brake", False).map(_safe_bool))).sum()) if len(fg) else 0
        ttc = pd.to_numeric(fg.get("best_min_ttc", pd.Series(dtype=float)), errors="coerce")
        trigger_counts = fg.get("taxonomy_trigger_combo", pd.Series(dtype=str)).value_counts(dropna=False).to_dict() if len(fg) else {}
        severity_counts = fg.get("taxonomy_severity_band", pd.Series(dtype=str)).value_counts(dropna=False).to_dict() if len(fg) else {}
        rows.append({
            "method": method,
            "num_scenes": num_scenes,
            "num_found": num_found,
            "attack_success_rate": num_found / num_scenes if num_scenes else 0.0,
            "collision_count": collision,
            "low_ttc_count": low_ttc,
            "hard_brake_count": hard,
            "collision_or_hard_brake_count": collision_or_hard,
            "low_ttc_only_count": low_ttc_only,
            "collision_rate_over_scenes": collision / num_scenes if num_scenes else 0.0,
            "low_ttc_rate_over_scenes": low_ttc / num_scenes if num_scenes else 0.0,
            "hard_brake_rate_over_scenes": hard / num_scenes if num_scenes else 0.0,
            "low_ttc_only_rate_over_found": low_ttc_only / num_found if num_found else 0.0,
            "median_best_min_ttc_found": float(ttc.replace(999.0, math.nan).median()) if len(ttc.dropna()) else None,
            "mean_best_min_ttc_found": float(ttc.replace(999.0, math.nan).mean()) if len(ttc.dropna()) else None,
            "trigger_counts_json": json.dumps(_json_safe(trigger_counts), ensure_ascii=False, sort_keys=True),
            "severity_counts_json": json.dumps(_json_safe(severity_counts), ensure_ascii=False, sort_keys=True),
        })
    return pd.DataFrame(rows).sort_values(["attack_success_rate", "method"], ascending=[False, True])


def _method_edit_taxonomy_summary(best_tax: pd.DataFrame) -> pd.DataFrame:
    if best_tax.empty:
        return pd.DataFrame()
    df = best_tax[best_tax["taxonomy_best_found"].fillna(False).map(_safe_bool)].copy()
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (method, edit), g in df.groupby(["method", "best_edit_name"], dropna=False):
        n = int(len(g))
        rows.append({
            "method": method,
            "best_edit_name": edit,
            "num_found": n,
            "collision_count": _safe_count_bool(g["taxonomy_collision"]),
            "low_ttc_count": _safe_count_bool(g["taxonomy_low_ttc"]),
            "hard_brake_count": _safe_count_bool(g["taxonomy_hard_brake"]),
            "low_ttc_only_count": int((g["taxonomy_low_ttc"].map(_safe_bool) & ~g["taxonomy_collision"].map(_safe_bool) & ~g["taxonomy_hard_brake"].map(_safe_bool)).sum()),
            "mean_mfc": float(pd.to_numeric(g.get("best_cost", pd.Series(dtype=float)), errors="coerce").mean()),
            "median_mfc": float(pd.to_numeric(g.get("best_cost", pd.Series(dtype=float)), errors="coerce").median()),
            "median_min_ttc": float(pd.to_numeric(g.get("best_min_ttc", pd.Series(dtype=float)), errors="coerce").replace(999.0, math.nan).median()),
            "trigger_counts_json": json.dumps(_json_safe(g["taxonomy_trigger_combo"].value_counts(dropna=False).to_dict()), ensure_ascii=False, sort_keys=True),
            "severity_counts_json": json.dumps(_json_safe(g["taxonomy_severity_band"].value_counts(dropna=False).to_dict()), ensure_ascii=False, sort_keys=True),
        })
    return pd.DataFrame(rows).sort_values(["method", "num_found", "best_edit_name"], ascending=[True, False, True])


def _threshold_sensitivity(best_tax: pd.DataFrame, thresholds: Iterable[float]) -> pd.DataFrame:
    if best_tax.empty or "method" not in best_tax.columns:
        return pd.DataFrame()
    rows = []
    for method, g in best_tax.groupby("method", dropna=False):
        num_scenes = int(len(g))
        found = g["taxonomy_best_found"].fillna(False).map(_safe_bool)
        fg = g[found].copy()
        num_found = int(len(fg))
        collision = fg.get("taxonomy_collision", pd.Series(dtype=bool)).map(_safe_bool) if len(fg) else pd.Series(dtype=bool)
        hard = fg.get("taxonomy_hard_brake", pd.Series(dtype=bool)).map(_safe_bool) if len(fg) else pd.Series(dtype=bool)
        ttc = pd.to_numeric(fg.get("best_min_ttc", pd.Series(dtype=float)), errors="coerce") if len(fg) else pd.Series(dtype=float)
        for th in thresholds:
            low = ttc.lt(float(th)) if len(fg) else pd.Series(dtype=bool)
            strict = collision | low if len(fg) else pd.Series(dtype=bool)
            strict_or_hard = strict | hard if len(fg) else pd.Series(dtype=bool)
            rows.append({
                "method": method,
                "ttc_threshold": float(th),
                "num_scenes": num_scenes,
                "num_found_original_threshold": num_found,
                "num_collision_or_ttc_under_threshold": int(strict.sum()) if len(fg) else 0,
                "num_collision_or_ttc_or_hard_brake_under_threshold": int(strict_or_hard.sum()) if len(fg) else 0,
                "rate_over_scenes_collision_or_ttc": float(strict.sum()) / num_scenes if num_scenes and len(fg) else 0.0,
                "rate_over_original_found_collision_or_ttc": float(strict.sum()) / num_found if num_found else 0.0,
                "rate_over_scenes_collision_or_ttc_or_hard_brake": float(strict_or_hard.sum()) / num_scenes if num_scenes and len(fg) else 0.0,
                "rate_over_original_found_collision_or_ttc_or_hard_brake": float(strict_or_hard.sum()) / num_found if num_found else 0.0,
            })
    return pd.DataFrame(rows).sort_values(["method", "ttc_threshold"])


def _load_candidate_rows(candidate_root: Path, ttc_threshold: float, max_tables: int = 0) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    tables = _find_candidate_tables(candidate_root)
    if max_tables and max_tables > 0:
        tables = tables[:max_tables]
    parts = []
    errors = []
    for p in tables:
        try:
            df = pd.read_csv(p)
            df["method"] = _method_from_table_path(p, candidate_root)
            df["scene_id"] = _scene_id_from_table_path(p)
            parts.append(_classify_candidate_df(df, ttc_threshold=ttc_threshold))
        except Exception as exc:
            errors.append({"path": str(p), "error": str(exc)})
    merged = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    diag = {
        "candidate_root": str(candidate_root),
        "candidate_tables_found": len(_find_candidate_tables(candidate_root)),
        "candidate_tables_loaded": len(parts),
        "candidate_table_errors": errors[:20],
        "num_candidate_table_errors": len(errors),
    }
    return merged, diag


def _candidate_trigger_summary(cand: pd.DataFrame) -> pd.DataFrame:
    if cand.empty or "method" not in cand.columns:
        return pd.DataFrame()
    rows = []
    for method, g in cand.groupby("method", dropna=False):
        n = int(len(g))
        failure = g.get("failure", pd.Series([False] * n)).map(_safe_bool)
        collision = g.get("taxonomy_collision", pd.Series([False] * n)).map(_safe_bool)
        low = g.get("taxonomy_low_ttc", pd.Series([False] * n)).map(_safe_bool)
        hard = g.get("taxonomy_hard_brake", pd.Series([False] * n)).map(_safe_bool)
        rows.append({
            "method": method,
            "total_candidate_rows": n,
            "failure_rows": int(failure.sum()),
            "collision_rows": int(collision.sum()),
            "low_ttc_rows": int(low.sum()),
            "hard_brake_rows": int(hard.sum()),
            "low_ttc_only_rows": int((low & ~collision & ~hard).sum()),
            "failure_rate_over_rows": float(failure.mean()) if n else 0.0,
            "low_ttc_only_rate_over_failure_rows": float((low & ~collision & ~hard).sum()) / int(failure.sum()) if int(failure.sum()) else 0.0,
            "trigger_counts_json": json.dumps(_json_safe(g.get("taxonomy_trigger_combo", pd.Series(dtype=str)).value_counts(dropna=False).to_dict()), ensure_ascii=False, sort_keys=True),
        })
    return pd.DataFrame(rows).sort_values(["failure_rate_over_rows", "method"], ascending=[False, True])


def _candidate_trigger_by_edit(cand: pd.DataFrame) -> pd.DataFrame:
    if cand.empty or "method" not in cand.columns or "edit_name" not in cand.columns:
        return pd.DataFrame()
    rows = []
    for (method, edit), g in cand.groupby(["method", "edit_name"], dropna=False):
        n = int(len(g))
        failure = g.get("failure", pd.Series([False] * n)).map(_safe_bool)
        collision = g.get("taxonomy_collision", pd.Series([False] * n)).map(_safe_bool)
        low = g.get("taxonomy_low_ttc", pd.Series([False] * n)).map(_safe_bool)
        hard = g.get("taxonomy_hard_brake", pd.Series([False] * n)).map(_safe_bool)
        rows.append({
            "method": method,
            "edit_name": edit,
            "total_candidate_rows": n,
            "failure_rows": int(failure.sum()),
            "collision_rows": int(collision.sum()),
            "low_ttc_rows": int(low.sum()),
            "hard_brake_rows": int(hard.sum()),
            "low_ttc_only_rows": int((low & ~collision & ~hard).sum()),
            "failure_rate_over_rows": float(failure.mean()) if n else 0.0,
            "mean_cost": float(pd.to_numeric(g.get("cost", pd.Series(dtype=float)), errors="coerce").replace(999.0, math.nan).mean()),
            "median_min_ttc": float(pd.to_numeric(g.get("min_ttc", pd.Series(dtype=float)), errors="coerce").replace(999.0, math.nan).median()),
        })
    return pd.DataFrame(rows).sort_values(["method", "edit_name"])


def _lead_brake_severity_audit(best_tax: pd.DataFrame, cand: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not best_tax.empty and "best_edit_name" in best_tax.columns:
        df = best_tax[(best_tax["taxonomy_best_found"].fillna(False).map(_safe_bool)) & (best_tax["best_edit_name"] == "lead_brake")].copy()
        for method, g in df.groupby("method", dropna=False):
            n = int(len(g))
            collision = g["taxonomy_collision"].map(_safe_bool)
            low = g["taxonomy_low_ttc"].map(_safe_bool)
            hard = g["taxonomy_hard_brake"].map(_safe_bool)
            rows.append({
                "source": "best_counterfactuals",
                "method": method,
                "lead_brake_rows": n,
                "collision_count": int(collision.sum()),
                "low_ttc_count": int(low.sum()),
                "hard_brake_count": int(hard.sum()),
                "low_ttc_only_count": int((low & ~collision & ~hard).sum()),
                "collision_rate": float(collision.mean()) if n else 0.0,
                "hard_brake_rate": float(hard.mean()) if n else 0.0,
                "median_min_ttc": float(pd.to_numeric(g.get("best_min_ttc", pd.Series(dtype=float)), errors="coerce").replace(999.0, math.nan).median()) if n else None,
                "median_cost": float(pd.to_numeric(g.get("best_cost", pd.Series(dtype=float)), errors="coerce").median()) if n else None,
            })
    if not cand.empty and "edit_name" in cand.columns:
        df = cand[cand["edit_name"] == "lead_brake"].copy()
        for method, g in df.groupby("method", dropna=False):
            n = int(len(g))
            failure = g.get("failure", pd.Series([False] * n)).map(_safe_bool)
            collision = g["taxonomy_collision"].map(_safe_bool)
            low = g["taxonomy_low_ttc"].map(_safe_bool)
            hard = g["taxonomy_hard_brake"].map(_safe_bool)
            rows.append({
                "source": "candidate_rows",
                "method": method,
                "lead_brake_rows": n,
                "failure_rows": int(failure.sum()),
                "collision_count": int(collision.sum()),
                "low_ttc_count": int(low.sum()),
                "hard_brake_count": int(hard.sum()),
                "low_ttc_only_count": int((low & ~collision & ~hard).sum()),
                "failure_rate": float(failure.mean()) if n else 0.0,
                "collision_rate": float(collision.mean()) if n else 0.0,
                "hard_brake_rate": float(hard.mean()) if n else 0.0,
                "median_min_ttc": float(pd.to_numeric(g.get("min_ttc", pd.Series(dtype=float)), errors="coerce").replace(999.0, math.nan).median()) if n else None,
                "median_cost": float(pd.to_numeric(g.get("cost", pd.Series(dtype=float)), errors="coerce").replace(999.0, math.nan).median()) if n else None,
            })
    return pd.DataFrame(rows).sort_values(["source", "method"]) if rows else pd.DataFrame()


def _load_previous_summary(previous_run_dir: Optional[Path]) -> pd.DataFrame:
    if not previous_run_dir:
        return pd.DataFrame()
    p = Path(previous_run_dir)
    candidates = [
        p / "report_summary" / "clean_safe_ablation_summary.json",
        p / "clean_safe_ablation_summary.json",
        p / "baseline_ablation" / "baseline_method_summary.csv",
        p / "baseline_method_summary.csv",
    ]
    for c in candidates:
        if c.suffix.lower() == ".json" and c.exists():
            try:
                payload = json.loads(c.read_text(encoding="utf-8"))
                rows = payload.get("method_summary") or payload.get("baseline_metrics_payload", {}).get("methods") or []
                if rows:
                    return pd.DataFrame(rows)
            except Exception:
                pass
        elif c.suffix.lower() == ".csv" and c.exists():
            df = _read_csv_if_exists(c)
            if not df.empty:
                return df
    return pd.DataFrame()


def _version_comparison(current_summary: pd.DataFrame, previous_summary: pd.DataFrame, previous_label: str = "previous") -> pd.DataFrame:
    if current_summary.empty or previous_summary.empty or "method" not in current_summary.columns or "method" not in previous_summary.columns:
        return pd.DataFrame()
    cols = ["method", "num_best_found", "attack_success_rate", "mean_censored_mfc", "mean_mfc_success_only", "collision_rate", "hard_brake_rate"]
    cur = current_summary[[c for c in cols if c in current_summary.columns]].copy()
    prev = previous_summary[[c for c in cols if c in previous_summary.columns]].copy()
    cur = cur.rename(columns={c: f"current_{c}" for c in cur.columns if c != "method"})
    prev = prev.rename(columns={c: f"{previous_label}_{c}" for c in prev.columns if c != "method"})
    merged = cur.merge(prev, on="method", how="outer")
    for metric in ["num_best_found", "attack_success_rate", "mean_censored_mfc", "mean_mfc_success_only", "collision_rate", "hard_brake_rate"]:
        a = f"current_{metric}"
        b = f"{previous_label}_{metric}"
        if a in merged.columns and b in merged.columns:
            merged[f"delta_{metric}"] = pd.to_numeric(merged[a], errors="coerce") - pd.to_numeric(merged[b], errors="coerce")
    return merged


def _claim_safety(best_summary: pd.DataFrame, threshold_df: pd.DataFrame, lead_audit: pd.DataFrame, primary: str, reference: str) -> Dict[str, Any]:
    def row_for(df: pd.DataFrame, method: str) -> Dict[str, Any]:
        if df.empty or "method" not in df.columns:
            return {}
        rows = df[df["method"] == method]
        return rows.iloc[0].to_dict() if len(rows) else {}

    primary_row = row_for(best_summary, primary)
    ref_row = row_for(best_summary, reference)
    random_row = row_for(best_summary, "random_budget")
    lead_best = pd.DataFrame()
    if not lead_audit.empty and "source" in lead_audit.columns:
        lead_best = lead_audit[(lead_audit["source"] == "best_counterfactuals") & (lead_audit["method"] == "lead_brake_only")]
    lead_row = lead_best.iloc[0].to_dict() if len(lead_best) else {}

    p_asr = _safe_float(primary_row.get("attack_success_rate"), math.nan)
    r_asr = _safe_float(ref_row.get("attack_success_rate"), math.nan)
    rand_asr = _safe_float(random_row.get("attack_success_rate"), math.nan)
    lead_n = int(_safe_float(lead_row.get("lead_brake_rows"), 0.0) or 0)
    lead_collision = int(_safe_float(lead_row.get("collision_count"), 0.0) or 0)
    lead_hard = int(_safe_float(lead_row.get("hard_brake_count"), 0.0) or 0)
    lead_low_ttc_only = int(_safe_float(lead_row.get("low_ttc_only_count"), 0.0) or 0)

    return {
        "version": VERSION,
        "supported_claims": {
            "failure_taxonomy_available": {
                "supported": not best_summary.empty,
                "evidence": f"best_failure_type_by_method rows={len(best_summary)}",
                "caution": "Use taxonomy tables to separate collision, hard-brake, and low-TTC near-miss evidence.",
            },
            "heading_aware_longitudinal_enables_lead_near_misses": {
                "supported": lead_n > 0 and lead_low_ttc_only > 0,
                "evidence": f"lead_brake_only best lead-brake failures={lead_n}, low_ttc_only={lead_low_ttc_only}, collision={lead_collision}, hard_brake={lead_hard}.",
                "caution": "Phrase as longitudinal near-miss discovery unless collision/hard-brake evidence is substantial.",
            },
            "hybrid_recovers_distance_full_budget": {
                "supported": math.isfinite(p_asr) and math.isfinite(r_asr) and abs(p_asr - r_asr) < 1e-9,
                "evidence": f"{primary} ASR={p_asr}, {reference} ASR={r_asr}.",
                "caution": "This supports parity/recovery, not superiority over distance_all.",
            },
        },
        "unsupported_or_risky_claims": {
            "lead_brake_collision_discovery": {
                "supported": lead_collision > 0,
                "evidence": f"lead_brake_only collision_count={lead_collision}, hard_brake_count={lead_hard}, low_ttc_only_count={lead_low_ttc_only}.",
                "caution": "Do not describe lead_brake gains as collision discovery if collision_count remains zero.",
            },
            "hybrid_beats_distance_all": {
                "supported": math.isfinite(p_asr) and math.isfinite(r_asr) and p_asr > r_asr,
                "evidence": f"{primary} ASR={p_asr}, {reference} ASR={r_asr}.",
                "caution": "Do not claim superiority when full-budget values are tied.",
            },
            "hybrid_beats_random_full_budget": {
                "supported": math.isfinite(p_asr) and math.isfinite(rand_asr) and p_asr > rand_asr,
                "evidence": f"{primary} ASR={p_asr}, random_budget ASR={rand_asr}.",
                "caution": "If random ties or exceeds the hybrid, present random as a strong baseline.",
            },
        },
        "numbers": {
            f"{primary}_attack_success_rate": p_asr,
            f"{reference}_attack_success_rate": r_asr,
            "random_budget_attack_success_rate": rand_asr,
            "lead_brake_only_best_count": lead_n,
            "lead_brake_only_collision_count": lead_collision,
            "lead_brake_only_hard_brake_count": lead_hard,
            "lead_brake_only_low_ttc_only_count": lead_low_ttc_only,
        },
    }


def _make_report(payload: Dict[str, Any], tables: Dict[str, pd.DataFrame]) -> str:
    lines: List[str] = []
    lines.append("# CausalSensor4D public_release Failure Taxonomy and Severity Audit")
    lines.append("")
    lines.append("## Purpose")
    lines.append("public_release audits whether public_release failures are collisions, hard-brake events, low-TTC near misses, or mixtures of these triggers. This prevents an inflated ASR from being interpreted as collision-only safety evidence.")
    lines.append("")
    lines.append("## Input diagnostics")
    diag = payload.get("input_diagnostic", {})
    lines.append(f"- baseline_out_dir: `{diag.get('baseline_out_dir')}`")
    lines.append(f"- candidate_root: `{diag.get('candidate_root')}`")
    lines.append(f"- scene_rows: `{diag.get('num_scene_rows')}`")
    lines.append(f"- candidate_tables_loaded: `{diag.get('candidate_tables_loaded')}` / `{diag.get('candidate_tables_found')}`")
    lines.append(f"- ttc_failure_threshold: `{payload.get('ttc_failure_threshold')}`")
    lines.append("")

    lines.append("## Best-counterfactual trigger taxonomy by method")
    df = tables.get("best_failure_type_by_method", pd.DataFrame())
    if df.empty:
        lines.append("No best-counterfactual taxonomy available.")
    else:
        cols = [
            "method",
            "num_scenes",
            "num_found",
            "attack_success_rate",
            "collision_count",
            "low_ttc_count",
            "hard_brake_count",
            "low_ttc_only_count",
            "low_ttc_only_rate_over_found",
            "median_best_min_ttc_found",
        ]
        lines.append(df[[c for c in cols if c in df.columns]].to_markdown(index=False, floatfmt=".3f"))
    lines.append("")

    lines.append("## Best-counterfactual trigger taxonomy by edit type")
    df = tables.get("best_failure_type_by_method_edit", pd.DataFrame())
    if df.empty:
        lines.append("No method-edit taxonomy available.")
    else:
        cols = ["method", "best_edit_name", "num_found", "collision_count", "low_ttc_count", "hard_brake_count", "low_ttc_only_count", "median_min_ttc", "median_mfc"]
        lines.append(df[[c for c in cols if c in df.columns]].to_markdown(index=False, floatfmt=".3f"))
    lines.append("")

    lines.append("## TTC threshold sensitivity")
    df = tables.get("best_threshold_sensitivity", pd.DataFrame())
    if df.empty:
        lines.append("No threshold sensitivity table available.")
    else:
        display = df[df["ttc_threshold"].isin([0.5, 1.0, 1.5])].copy() if "ttc_threshold" in df.columns else df
        cols = ["method", "ttc_threshold", "num_found_original_threshold", "num_collision_or_ttc_under_threshold", "rate_over_scenes_collision_or_ttc", "rate_over_original_found_collision_or_ttc"]
        lines.append(display[[c for c in cols if c in display.columns]].to_markdown(index=False, floatfmt=".3f"))
    lines.append("")

    lines.append("## Lead-brake severity audit")
    df = tables.get("lead_brake_severity_audit", pd.DataFrame())
    if df.empty:
        lines.append("No lead-brake audit available.")
    else:
        cols = ["source", "method", "lead_brake_rows", "failure_rows", "collision_count", "low_ttc_count", "hard_brake_count", "low_ttc_only_count", "collision_rate", "hard_brake_rate", "median_min_ttc"]
        lines.append(df[[c for c in cols if c in df.columns]].to_markdown(index=False, floatfmt=".3f"))
    lines.append("")

    lines.append("## Candidate-row trigger summary")
    df = tables.get("candidate_failure_trigger_summary", pd.DataFrame())
    if df.empty:
        lines.append("No candidate trigger summary available.")
    else:
        cols = ["method", "total_candidate_rows", "failure_rows", "collision_rows", "low_ttc_rows", "hard_brake_rows", "low_ttc_only_rows", "failure_rate_over_rows"]
        lines.append(df[[c for c in cols if c in df.columns]].to_markdown(index=False, floatfmt=".3f"))
    lines.append("")

    comp = tables.get("version_comparison", pd.DataFrame())
    if comp is not None and not comp.empty:
        lines.append("## Previous-version comparison")
        cols = [c for c in comp.columns if c == "method" or c.startswith("current_attack") or c.startswith("previous_attack") or c.startswith("delta_attack") or c.startswith("current_num_best") or c.startswith("previous_num_best") or c.startswith("delta_num_best")]
        lines.append(comp[cols].to_markdown(index=False, floatfmt=".3f"))
        lines.append("")

    lines.append("## Claim-safety checklist")
    checklist = payload.get("claim_safety_checklist", {})
    for section in ["supported_claims", "unsupported_or_risky_claims"]:
        lines.append(f"### {section}")
        for key, item in checklist.get(section, {}).items():
            lines.append(f"- `{key}`: supported=`{item.get('supported')}`. {item.get('evidence')} {item.get('caution')}")
        lines.append("")

    lines.append("## Report-safe interpretation")
    lines.append("Report the baseline and taxonomy stages as a multi-trigger counterfactual safety audit. For lead-brake results, use the term longitudinal low-TTC near miss unless the taxonomy shows collision or hard-brake support. For causal_hybrid, keep the conservative claim that it recovers distance_all-level discovery while retaining causal-prioritized interpretation.")
    return "\n".join(lines)


def generate_failure_taxonomy(
    run_dir: Optional[Path] = None,
    baseline_out_dir: Optional[Path] = None,
    out_dir: Path = Path("outputs/failure_taxonomy_run"),
    previous_run_dir: Optional[Path] = None,
    ttc_threshold: float = DEFAULT_FAILURE_TTC_THRESHOLD,
    sensitivity_thresholds: Optional[List[float]] = None,
    primary_method: str = "causal_hybrid",
    reference_method: str = "distance_all",
    load_candidate_rows: bool = True,
    max_candidate_tables: int = 0,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    baseline_dir = _resolve_baseline_out_dir(run_dir=Path(run_dir) if run_dir else None, baseline_out_dir=Path(baseline_out_dir) if baseline_out_dir else None)
    candidate_root = _resolve_candidate_root(baseline_dir)
    sensitivity_thresholds = sensitivity_thresholds or list(DEFAULT_TTC_THRESHOLDS)

    all_rows = _load_all_baseline_rows(baseline_dir)
    method_summary = _load_method_summary(baseline_dir)
    best_tax = _best_taxonomy_table(all_rows, ttc_threshold=ttc_threshold)
    best_method = _method_failure_type_summary(best_tax)
    best_edit = _method_edit_taxonomy_summary(best_tax)
    threshold_df = _threshold_sensitivity(best_tax, sensitivity_thresholds)

    candidate_diag: Dict[str, Any] = {}
    candidate_df = pd.DataFrame()
    candidate_summary = pd.DataFrame()
    candidate_by_edit = pd.DataFrame()
    if load_candidate_rows:
        candidate_df, candidate_diag = _load_candidate_rows(candidate_root, ttc_threshold=ttc_threshold, max_tables=max_candidate_tables)
        candidate_summary = _candidate_trigger_summary(candidate_df)
        candidate_by_edit = _candidate_trigger_by_edit(candidate_df)
    else:
        candidate_diag = {"candidate_root": str(candidate_root), "candidate_rows_loaded": False}

    lead_audit = _lead_brake_severity_audit(best_tax, candidate_df)
    previous_summary = _load_previous_summary(Path(previous_run_dir) if previous_run_dir else None)
    comparison = _version_comparison(method_summary if not method_summary.empty else best_method, previous_summary, previous_label="previous")

    tables: Dict[str, pd.DataFrame] = {
        "best_failure_type_by_scene": best_tax,
        "best_failure_type_by_method": best_method,
        "best_failure_type_by_method_edit": best_edit,
        "best_threshold_sensitivity": threshold_df,
        "candidate_failure_trigger_summary": candidate_summary,
        "candidate_failure_trigger_by_edit": candidate_by_edit,
        "lead_brake_severity_audit": lead_audit,
        "version_comparison": comparison,
    }

    outputs: Dict[str, Any] = {}
    for name, df in tables.items():
        csv_path = tables_dir / f"{name}.csv"
        md_path = tables_dir / f"{name}.md"
        df.to_csv(csv_path, index=False)
        try:
            md_path.write_text(df.to_markdown(index=False, floatfmt=".3f") if not df.empty else "", encoding="utf-8")
        except Exception:
            md_path.write_text("", encoding="utf-8")
        outputs[name] = {"csv": str(csv_path), "md": str(md_path)}

    checklist = _claim_safety(best_method, threshold_df, lead_audit, primary=primary_method, reference=reference_method)
    payload: Dict[str, Any] = {
        "version": VERSION,
        "ttc_failure_threshold": ttc_threshold,
        "sensitivity_thresholds": sensitivity_thresholds,
        "input_diagnostic": {
            "run_dir": str(run_dir) if run_dir else None,
            "baseline_out_dir": str(baseline_dir),
            "candidate_root": str(candidate_root),
            "num_scene_rows": int(len(all_rows)),
            "method_summary_rows": int(len(method_summary)),
            **candidate_diag,
        },
        "primary_method": primary_method,
        "reference_method": reference_method,
        "claim_safety_checklist": checklist,
        "outputs": outputs,
    }

    report = _make_report(payload, tables)
    report_path = out_dir / "failure_taxonomy_report.md"
    summary_path = out_dir / "failure_taxonomy_summary.json"
    checklist_path = out_dir / "claim_safety_checklist.json"
    manifest_path = out_dir / "failure_taxonomy_manifest.json"
    report_path.write_text(report, encoding="utf-8")
    _write_json(summary_path, payload)
    _write_json(checklist_path, checklist)
    manifest = {
        "version": VERSION,
        "purpose": "Failure taxonomy and severity validation for public_release heading-aware longitudinal results.",
        "out_dir": str(out_dir),
        "report": str(report_path),
        "summary": str(summary_path),
        "claim_safety_checklist": str(checklist_path),
        "tables": outputs,
        "missing_inputs": [] if len(all_rows) else ["all_baseline_scene_results.csv"],
    }
    _write_json(manifest_path, manifest)
    payload["outputs"].update({
        "report": str(report_path),
        "summary": str(summary_path),
        "claim_safety_checklist": str(checklist_path),
        "manifest": str(manifest_path),
    })
    # Rewrite payload after output paths are included.
    _write_json(summary_path, payload)
    return payload
