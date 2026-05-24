from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json
import math
import shutil

import numpy as np
import pandas as pd

VERSION = "public_release"
DEFAULT_METHOD_ORDER = [
    "causal_hybrid",
    "distance_all",
    "random_budget",
    "causal_guided",
    "cut_in_only",
    "pedestrian_only",
    "lead_brake_only",
]
DEFAULT_PAIRWISE = [
    ("causal_hybrid", "distance_all"),
    ("causal_hybrid", "random_budget"),
    ("causal_hybrid", "causal_guided"),
    ("causal_hybrid", "cut_in_only"),
]
DEFAULT_N_BOOT = 5000
DEFAULT_SEED = 42
DEFAULT_CENSORED_MFC = 2.0


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _bool_series(s: pd.Series) -> pd.Series:
    def conv(x: Any) -> bool:
        if pd.isna(x):
            return False
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float)):
            return bool(x)
        return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}
    return s.map(conv)

def _trapz_compat(y: np.ndarray, x: np.ndarray) -> float:
    """Compatibility wrapper for NumPy 2.x where np.trapz may be removed.

    Uses np.trapezoid when available, otherwise falls back to the explicit
    trapezoidal-rule formula.  This keeps public_release bootstrap AUC computation
    compatible with both older and newer NumPy versions.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float(np.nanmean(y)) if len(y) else math.nan
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    # Manual trapezoidal integration fallback.
    dx = np.diff(x)
    avg_y = (y[:-1] + y[1:]) / 2.0
    return float(np.sum(dx * avg_y))


def _write_table(df: pd.DataFrame, base: Path) -> Dict[str, str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    csv_path = base.with_suffix(".csv")
    md_path = base.with_suffix(".md")
    tex_path = base.with_suffix(".tex")
    df.to_csv(csv_path, index=False)
    try:
        md_path.write_text(df.to_markdown(index=False, floatfmt=".3f"), encoding="utf-8")
    except Exception:
        md_path.write_text(df.to_string(index=False), encoding="utf-8")
    tex_lines = [
        "% Auto-generated lightweight LaTeX placeholder.",
        "% Use CSV/Markdown table as the source of truth for final formatting.",
        df.to_csv(index=False),
    ]
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
    return {"csv": str(csv_path), "md": str(md_path), "tex": str(tex_path)}


def _copy_if_exists(src: Path, dst: Path) -> Optional[str]:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def _table_path(root: Path, filename: str) -> Path:
    root = Path(root)
    if (root / "tables" / filename).exists():
        return root / "tables" / filename
    return root / filename


def _method_order(m: str) -> int:
    try:
        return DEFAULT_METHOD_ORDER.index(str(m))
    except ValueError:
        return 99


def _metric_ci(values: np.ndarray, *, n_boot: int, seed: int, alpha: float = 0.05) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {"n": 0, "point": math.nan, "ci_low": math.nan, "ci_high": math.nan, "std_boot": math.nan, "num_boot": 0}
    point = float(np.nanmean(arr))
    rng = np.random.default_rng(seed)
    # For deterministic speed, bootstrap all indices at once. 120 scenes x 5000 bootstraps is small.
    idx = rng.integers(0, n, size=(int(n_boot), n))
    boot = arr[idx].mean(axis=1)
    return {
        "n": n,
        "point": point,
        "ci_low": float(np.quantile(boot, alpha / 2.0)),
        "ci_high": float(np.quantile(boot, 1.0 - alpha / 2.0)),
        "std_boot": float(np.std(boot, ddof=1)) if len(boot) > 1 else 0.0,
        "num_boot": int(n_boot),
    }


def _delta_ci(values_a: np.ndarray, values_b: np.ndarray, *, n_boot: int, seed: int, alpha: float = 0.05) -> Dict[str, Any]:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    n = int(a.size)
    if n == 0:
        return {"n": 0, "point_delta": math.nan, "ci_low": math.nan, "ci_high": math.nan, "std_boot": math.nan, "num_boot": 0}
    diff = a - b
    point = float(np.mean(diff))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    boot = diff[idx].mean(axis=1)
    return {
        "n": n,
        "point_delta": point,
        "ci_low": float(np.quantile(boot, alpha / 2.0)),
        "ci_high": float(np.quantile(boot, 1.0 - alpha / 2.0)),
        "std_boot": float(np.std(boot, ddof=1)) if len(boot) > 1 else 0.0,
        "num_boot": int(n_boot),
    }


def _build_per_scene_metric_table(best_scene: pd.DataFrame, censored_mfc: float) -> pd.DataFrame:
    if best_scene.empty:
        return pd.DataFrame()
    df = best_scene.copy()
    if "method" not in df.columns or "scene_id" not in df.columns:
        return pd.DataFrame()
    # Prefer taxonomy columns from public_release, with fallbacks to baseline columns.
    found_col = "taxonomy_best_found" if "taxonomy_best_found" in df.columns else "best_found"
    collision_col = "taxonomy_collision" if "taxonomy_collision" in df.columns else "best_collision"
    hard_col = "taxonomy_hard_brake" if "taxonomy_hard_brake" in df.columns else "best_hard_brake"
    low_col = "taxonomy_low_ttc" if "taxonomy_low_ttc" in df.columns else None
    if found_col not in df.columns:
        df[found_col] = False
    if collision_col not in df.columns:
        df[collision_col] = False
    if hard_col not in df.columns:
        df[hard_col] = False
    df["metric_asr"] = _bool_series(df[found_col]).astype(float)
    df["metric_collision_rate"] = _bool_series(df[collision_col]).astype(float)
    df["metric_hard_brake_rate"] = _bool_series(df[hard_col]).astype(float)
    if low_col and low_col in df.columns:
        df["metric_low_ttc_rate"] = _bool_series(df[low_col]).astype(float)
    else:
        best_min_ttc = pd.to_numeric(df.get("best_min_ttc", math.nan), errors="coerce")
        df["metric_low_ttc_rate"] = (best_min_ttc < 1.5).fillna(False).astype(float)
    df["metric_collision_or_hard_brake_rate"] = ((df["metric_collision_rate"] > 0) | (df["metric_hard_brake_rate"] > 0)).astype(float)
    df["metric_low_ttc_only_rate"] = ((df["metric_low_ttc_rate"] > 0) & (df["metric_collision_rate"] == 0) & (df["metric_hard_brake_rate"] == 0)).astype(float)
    costs = pd.to_numeric(df.get("best_cost", math.nan), errors="coerce")
    found = df["metric_asr"] > 0
    df["metric_censored_mfc"] = np.where(found & costs.notna(), costs.astype(float), float(censored_mfc))
    df["metric_mfc_success_only"] = np.where(found & costs.notna(), costs.astype(float), np.nan)
    best_ttc = pd.to_numeric(df.get("best_min_ttc", math.nan), errors="coerce")
    df["metric_min_ttc_success_only"] = np.where(found, best_ttc, np.nan)
    return df


def _prepare_wide(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if df.empty or value_col not in df.columns:
        return pd.DataFrame()
    return df.pivot_table(index="scene_id", columns="method", values=value_col, aggfunc="mean")


def _bootstrap_main_metrics(per_scene: pd.DataFrame, *, methods: List[str], n_boot: int, seed: int) -> pd.DataFrame:
    metrics = [
        ("attack_success_rate", "metric_asr", "higher_is_better"),
        ("mean_censored_mfc", "metric_censored_mfc", "lower_is_better"),
        ("mean_mfc_success_only", "metric_mfc_success_only", "lower_is_better"),
        ("collision_rate_over_scenes", "metric_collision_rate", "higher_means_more_severe"),
        ("hard_brake_rate_over_scenes", "metric_hard_brake_rate", "higher_means_more_severe"),
        ("low_ttc_rate_over_scenes", "metric_low_ttc_rate", "higher_means_more_near_miss"),
        ("low_ttc_only_rate_over_scenes", "metric_low_ttc_only_rate", "higher_means_more_low_ttc_only"),
        ("collision_or_hard_brake_rate_over_scenes", "metric_collision_or_hard_brake_rate", "higher_means_more_physical_evidence"),
        ("mean_min_ttc_success_only", "metric_min_ttc_success_only", "lower_means_more_critical_ttc"),
    ]
    rows: List[Dict[str, Any]] = []
    for method in methods:
        g = per_scene[per_scene["method"].astype(str) == method]
        if g.empty:
            continue
        for metric_name, col, interpretation in metrics:
            ci = _metric_ci(g[col].to_numpy(dtype=float), n_boot=n_boot, seed=seed + 17 * (_method_order(method) + 1) + len(rows))
            rows.append({
                "method": method,
                "metric": metric_name,
                "point": ci["point"],
                "ci95_low": ci["ci_low"],
                "ci95_high": ci["ci_high"],
                "std_boot": ci["std_boot"],
                "n_scenes": ci["n"],
                "num_boot": ci["num_boot"],
                "interpretation": interpretation,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["_order"] = out["method"].map(_method_order)
        out = out.sort_values(["_order", "metric"]).drop(columns=["_order"])
    return out


def _bootstrap_pairwise(per_scene: pd.DataFrame, *, method_pairs: List[Tuple[str, str]], n_boot: int, seed: int) -> pd.DataFrame:
    metrics = [
        ("delta_asr", "metric_asr", "positive_favors_first"),
        ("delta_censored_mfc", "metric_censored_mfc", "negative_favors_first"),
        ("delta_collision_rate", "metric_collision_rate", "positive_means_first_more_collision_evidence"),
        ("delta_hard_brake_rate", "metric_hard_brake_rate", "positive_means_first_more_hard_brake_evidence"),
        ("delta_low_ttc_only_rate", "metric_low_ttc_only_rate", "positive_means_first_more_low_ttc_only"),
        ("delta_collision_or_hard_brake_rate", "metric_collision_or_hard_brake_rate", "positive_means_first_more_physical_evidence"),
    ]
    rows: List[Dict[str, Any]] = []
    for a, b in method_pairs:
        for metric_name, col, interpretation in metrics:
            wide = _prepare_wide(per_scene, col)
            if wide.empty or a not in wide.columns or b not in wide.columns:
                continue
            ci = _delta_ci(wide[a].to_numpy(dtype=float), wide[b].to_numpy(dtype=float), n_boot=n_boot, seed=seed + 1009 + len(rows))
            rows.append({
                "method_a": a,
                "method_b": b,
                "metric": metric_name,
                "point_delta_a_minus_b": ci["point_delta"],
                "ci95_low": ci["ci_low"],
                "ci95_high": ci["ci_high"],
                "std_boot": ci["std_boot"],
                "n_paired_scenes": ci["n"],
                "num_boot": ci["num_boot"],
                "interpretation": interpretation,
                "ci_excludes_zero": bool(np.isfinite(ci["ci_low"]) and np.isfinite(ci["ci_high"]) and (ci["ci_low"] > 0 or ci["ci_high"] < 0)),
            })
    return pd.DataFrame(rows)


def _load_budgeted_success(baseline_run_dir: Path) -> pd.DataFrame:
    candidates = [
        baseline_run_dir / "budgeted_audit" / "budgeted_success_curve.csv",
        baseline_run_dir / "budgeted_success_curve.csv",
    ]
    for p in candidates:
        df = _read_csv(p)
        if not df.empty:
            return df
    return pd.DataFrame()


def _compute_scene_auc(budgeted: pd.DataFrame) -> pd.DataFrame:
    required = {"method", "scene_id", "budget_k", "success_at_k"}
    if budgeted.empty or not required.issubset(set(budgeted.columns)):
        return pd.DataFrame()
    df = budgeted.copy()
    df["budget_k"] = pd.to_numeric(df["budget_k"], errors="coerce")
    df["success_at_k"] = _bool_series(df["success_at_k"]).astype(float)
    df = df.dropna(subset=["budget_k"])
    rows: List[Dict[str, Any]] = []
    for (method, scene_id), g in df.groupby(["method", "scene_id"], dropna=False):
        gg = g.sort_values("budget_k")
        x = gg["budget_k"].to_numpy(dtype=float)
        y = gg["success_at_k"].to_numpy(dtype=float)
        if len(x) == 0:
            continue
        if len(x) == 1 or float(x.max()) == float(x.min()):
            auc = float(y.mean())
        else:
            auc = float(_trapz_compat(y, x) / (x.max() - x.min()))
        rows.append({"method": method, "scene_id": scene_id, "metric_budgeted_auc": auc, "metric_success_at_min_budget": float(y[0]), "metric_success_at_max_budget": float(y[-1])})
    return pd.DataFrame(rows)


def _bootstrap_budgeted_auc(scene_auc: pd.DataFrame, *, methods: List[str], n_boot: int, seed: int) -> pd.DataFrame:
    if scene_auc.empty:
        return pd.DataFrame()
    metrics = [
        ("budgeted_auc", "metric_budgeted_auc"),
        ("success_at_min_budget", "metric_success_at_min_budget"),
        ("success_at_max_budget", "metric_success_at_max_budget"),
    ]
    rows: List[Dict[str, Any]] = []
    for method in methods:
        g = scene_auc[scene_auc["method"].astype(str) == method]
        if g.empty:
            continue
        for metric_name, col in metrics:
            ci = _metric_ci(g[col].to_numpy(dtype=float), n_boot=n_boot, seed=seed + 333 + 13 * (_method_order(method) + 1) + len(rows))
            rows.append({
                "method": method,
                "metric": metric_name,
                "point": ci["point"],
                "ci95_low": ci["ci_low"],
                "ci95_high": ci["ci_high"],
                "std_boot": ci["std_boot"],
                "n_scenes": ci["n"],
                "num_boot": ci["num_boot"],
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["_order"] = out["method"].map(_method_order)
        out = out.sort_values(["_order", "metric"]).drop(columns=["_order"])
    return out


def _claim_ci_table(main_ci: pd.DataFrame, pair_ci: pd.DataFrame, budget_ci: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    def lookup(df: pd.DataFrame, **conds: str) -> Optional[pd.Series]:
        if df.empty:
            return None
        mask = pd.Series(True, index=df.index)
        for k, v in conds.items():
            if k not in df.columns:
                return None
            mask &= df[k].astype(str).eq(str(v))
        hit = df[mask]
        return hit.iloc[0] if len(hit) else None

    # Main readiness: hybrid ASR CI.
    h_asr = lookup(main_ci, method="causal_hybrid", metric="attack_success_rate")
    if h_asr is not None:
        rows.append({
            "claim": "causal_hybrid high clean safe-to-failure discovery",
            "ci_result": f"ASR={h_asr['point']:.3f}, 95% CI [{h_asr['ci95_low']:.3f}, {h_asr['ci95_high']:.3f}]",
            "report_use": "safe",
            "caution": "Report with CI; do not imply superiority over tied baselines.",
        })
    # Hybrid vs distance parity.
    hd = lookup(pair_ci, method_a="causal_hybrid", method_b="distance_all", metric="delta_asr")
    if hd is not None:
        rows.append({
            "claim": "causal_hybrid recovers distance_all-level discovery",
            "ci_result": f"Delta ASR={hd['point_delta_a_minus_b']:.3f}, 95% CI [{hd['ci95_low']:.3f}, {hd['ci95_high']:.3f}]",
            "report_use": "safe_with_caution",
            "caution": "Use recovery/parity language, not superiority.",
        })
    # Hybrid vs strict.
    hs = lookup(pair_ci, method_a="causal_hybrid", method_b="causal_guided", metric="delta_asr")
    if hs is not None:
        rows.append({
            "claim": "hybrid fallback improves over strict causal routing",
            "ci_result": f"Delta ASR={hs['point_delta_a_minus_b']:.3f}, 95% CI [{hs['ci95_low']:.3f}, {hs['ci95_high']:.3f}]",
            "report_use": "safe" if bool(hs.get("ci_excludes_zero", False)) and float(hs["ci95_low"]) > 0 else "safe_with_caution",
            "caution": "If CI includes zero, phrase as observed improvement rather than statistically clear gain.",
        })
    # Random budgeted superiority guardrail.
    hr_auc = lookup(pair_ci, method_a="causal_hybrid", method_b="random_budget", metric="delta_asr")
    rb_auc = lookup(budget_ci, method="random_budget", metric="budgeted_auc")
    hb_auc = lookup(budget_ci, method="causal_hybrid", metric="budgeted_auc")
    if rb_auc is not None and hb_auc is not None:
        rows.append({
            "claim": "budgeted superiority over random",
            "ci_result": f"Hybrid AUC={hb_auc['point']:.3f}; random AUC={rb_auc['point']:.3f}",
            "report_use": "do_not_claim" if float(hb_auc["point"]) <= float(rb_auc["point"]) else "check_with_pairwise_ci",
            "caution": "Random remains a strong or tied budgeted baseline.",
        })
    lead_low_only = lookup(main_ci, method="lead_brake_only", metric="low_ttc_only_rate_over_scenes")
    lead_col = lookup(main_ci, method="lead_brake_only", metric="collision_rate_over_scenes")
    if lead_low_only is not None:
        rows.append({
            "claim": "heading-aware longitudinal geometry enables lead-brake near-misses",
            "ci_result": f"Lead low-TTC-only scene rate={lead_low_only['point']:.3f}, 95% CI [{lead_low_only['ci95_low']:.3f}, {lead_low_only['ci95_high']:.3f}]",
            "report_use": "safe_with_caution",
            "caution": "Say longitudinal low-TTC near-miss. Do not call it collision discovery.",
        })
    if lead_col is not None:
        rows.append({
            "claim": "lead_brake collision discovery",
            "ci_result": f"Lead collision scene rate={lead_col['point']:.3f}, 95% CI [{lead_col['ci95_low']:.3f}, {lead_col['ci95_high']:.3f}]",
            "report_use": "do_not_claim" if float(lead_col["point"]) == 0 else "safe_with_caution",
            "caution": "public_release taxonomy shows lead-brake is near-miss dominated.",
        })
    return pd.DataFrame(rows)


def _make_report(summary: Dict[str, Any], main_ci: pd.DataFrame, pair_ci: pd.DataFrame, budget_ci: pd.DataFrame, claim_ci: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("# CausalSensor4D public_release Bootstrap Confidence Interval Audit")
    lines.append("")
    lines.append("## Purpose")
    lines.append("public_release adds scene-level bootstrap confidence intervals for benchmark-ready reporting. It estimates uncertainty for attack success, MFC, failure taxonomy rates, and budgeted AUC without rerunning the baseline search.")
    lines.append("")
    lines.append("## Input diagnostics")
    diag = summary.get("input_diagnostic", {})
    for k, v in diag.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Main bootstrap CI table")
    if main_ci.empty:
        lines.append("No main CI table available.")
    else:
        show = main_ci[main_ci["metric"].isin(["attack_success_rate", "mean_censored_mfc", "collision_rate_over_scenes", "hard_brake_rate_over_scenes", "low_ttc_only_rate_over_scenes"])].copy()
        lines.append(show.to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append("## Pairwise bootstrap deltas")
    if pair_ci.empty:
        lines.append("No pairwise CI table available.")
    else:
        show = pair_ci[pair_ci["metric"].isin(["delta_asr", "delta_censored_mfc", "delta_collision_or_hard_brake_rate", "delta_low_ttc_only_rate"])].copy()
        lines.append(show.to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append("## Budgeted AUC CI")
    if budget_ci.empty:
        lines.append("No budgeted AUC CI table available. Provide `budgeted_success_curve.csv` if budgeted uncertainty is required.")
    else:
        lines.append(budget_ci.to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append("## Claim-level CI guardrails")
    if claim_ci.empty:
        lines.append("No claim guardrail table available.")
    else:
        lines.append(claim_ci.to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append("## Report-safe interpretation")
    lines.append("Use the baseline and taxonomy stages as the main benchmark result with confidence intervals. Keep the established guardrails: causal_hybrid recovers distance_all-level discovery rather than outperforming it; random remains a strong baseline; lead-brake gains should be described as longitudinal low-TTC near-miss discovery, not collision discovery.")
    return "\n".join(lines)


def generate_bootstrap_ci_pack(
    *,
    baseline_run_dir: str | Path,
    taxonomy_dir: str | Path,
    out_dir: str | Path = "outputs/bootstrap_ci_run",
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    censored_mfc: float = DEFAULT_CENSORED_MFC,
    methods: Optional[List[str]] = None,
) -> Dict[str, Any]:
    baseline_path = Path(baseline_run_dir)
    tax = Path(taxonomy_dir)
    out = Path(out_dir)
    tables_dir = out / "tables"
    src_dir = out / "source_reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    src_dir.mkdir(parents=True, exist_ok=True)
    methods = methods or DEFAULT_METHOD_ORDER

    best_scene_path = _table_path(tax, "best_failure_type_by_scene.csv")
    best_scene = _read_csv(best_scene_path)
    per_scene = _build_per_scene_metric_table(best_scene, censored_mfc=censored_mfc)

    budgeted = _load_budgeted_success(baseline_path)
    scene_auc = _compute_scene_auc(budgeted)

    main_ci = _bootstrap_main_metrics(per_scene, methods=methods, n_boot=n_boot, seed=seed)
    pair_ci = _bootstrap_pairwise(per_scene, method_pairs=DEFAULT_PAIRWISE, n_boot=n_boot, seed=seed)
    budget_ci = _bootstrap_budgeted_auc(scene_auc, methods=methods, n_boot=n_boot, seed=seed) if not scene_auc.empty else pd.DataFrame()
    claim_ci = _claim_ci_table(main_ci, pair_ci, budget_ci)

    table_paths = {
        "bootstrap_ci_main_metrics": _write_table(main_ci, tables_dir / "bootstrap_ci_main_metrics"),
        "bootstrap_ci_pairwise_deltas": _write_table(pair_ci, tables_dir / "bootstrap_ci_pairwise_deltas"),
        "bootstrap_ci_budgeted_auc": _write_table(budget_ci, tables_dir / "bootstrap_ci_budgeted_auc"),
        "bootstrap_ci_claim_guardrails": _write_table(claim_ci, tables_dir / "bootstrap_ci_claim_guardrails"),
        "per_scene_bootstrap_metric_source": _write_table(per_scene[[c for c in ["scene_id", "method", "metric_asr", "metric_censored_mfc", "metric_collision_rate", "metric_hard_brake_rate", "metric_low_ttc_rate", "metric_low_ttc_only_rate", "metric_collision_or_hard_brake_rate", "metric_mfc_success_only", "metric_min_ttc_success_only"] if c in per_scene.columns]], tables_dir / "per_scene_bootstrap_metric_source"),
    }
    if not scene_auc.empty:
        table_paths["per_scene_budgeted_auc_source"] = _write_table(scene_auc, tables_dir / "per_scene_budgeted_auc_source")

    # Copy key source reports for traceability.
    copied = {
        "failure_taxonomy_summary": _copy_if_exists(tax / "failure_taxonomy_summary.json", src_dir / "failure_taxonomy_summary.json"),
        "failure_taxonomy_report": _copy_if_exists(tax / "failure_taxonomy_report.md", src_dir / "failure_taxonomy_report.md"),
        "claim_safety_checklist": _copy_if_exists(tax / "claim_safety_checklist.json", src_dir / "claim_safety_checklist.json"),
        "clean_safe_ablation_summary": _copy_if_exists(baseline_path / "report_summary" / "clean_safe_ablation_summary.json", src_dir / "clean_safe_ablation_summary.json"),
        "causal_hybrid_audit_summary": _copy_if_exists(baseline_path / "budgeted_audit" / "causal_hybrid_audit_summary.json", src_dir / "causal_hybrid_audit_summary.json"),
    }

    input_diag = {
        "baseline_run_dir": str(baseline_path),
        "taxonomy_dir": str(tax),
        "out_dir": str(out),
        "best_failure_type_by_scene": str(best_scene_path),
        "scene_rows": int(len(best_scene)),
        "per_scene_metric_rows": int(len(per_scene)),
        "num_methods": int(per_scene["method"].nunique()) if not per_scene.empty and "method" in per_scene.columns else 0,
        "num_unique_scenes": int(per_scene["scene_id"].nunique()) if not per_scene.empty and "scene_id" in per_scene.columns else 0,
        "budgeted_success_rows": int(len(budgeted)),
        "per_scene_budgeted_auc_rows": int(len(scene_auc)),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "censored_mfc": float(censored_mfc),
        "missing_inputs": [] if not best_scene.empty else [str(best_scene_path)],
    }
    summary = {
        "version": VERSION,
        "purpose": "Scene-level bootstrap confidence intervals for benchmark-stage metrics.",
        "input_diagnostic": input_diag,
        "methods": methods,
        "tables": table_paths,
        "copied_source_reports": copied,
        "primary_ci_highlights": {
            "causal_hybrid_asr_ci": _json_safe(main_ci[(main_ci["method"] == "causal_hybrid") & (main_ci["metric"] == "attack_success_rate")].iloc[0].to_dict()) if not main_ci[(main_ci["method"] == "causal_hybrid") & (main_ci["metric"] == "attack_success_rate")].empty else None,
            "lead_brake_low_ttc_only_ci": _json_safe(main_ci[(main_ci["method"] == "lead_brake_only") & (main_ci["metric"] == "low_ttc_only_rate_over_scenes")].iloc[0].to_dict()) if not main_ci[(main_ci["method"] == "lead_brake_only") & (main_ci["metric"] == "low_ttc_only_rate_over_scenes")].empty else None,
            "hybrid_minus_strict_asr_ci": _json_safe(pair_ci[(pair_ci["method_a"] == "causal_hybrid") & (pair_ci["method_b"] == "causal_guided") & (pair_ci["metric"] == "delta_asr")].iloc[0].to_dict()) if not pair_ci[(pair_ci["method_a"] == "causal_hybrid") & (pair_ci["method_b"] == "causal_guided") & (pair_ci["metric"] == "delta_asr")].empty else None,
        },
    }
    report = _make_report(summary, main_ci, pair_ci, budget_ci, claim_ci)
    report_path = out / "bootstrap_ci_report.md"
    report_path.write_text(report, encoding="utf-8")
    summary["reports"] = {
        "bootstrap_ci_report": str(report_path),
        "bootstrap_ci_summary": str(out / "bootstrap_ci_summary.json"),
        "bootstrap_ci_manifest": str(out / "bootstrap_ci_manifest.json"),
    }
    _write_json(out / "bootstrap_ci_summary.json", summary)
    _write_json(out / "bootstrap_ci_manifest.json", {
        "version": VERSION,
        "tables": table_paths,
        "reports": summary["reports"],
        "input_diagnostic": input_diag,
        "copied_source_reports": copied,
    })
    return summary
