from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json
import math
import random

import pandas as pd

DEFAULT_METHODS = [
    "causal_guided",
    "causal_hybrid",
    "distance_all",
    "random_budget",
    "lead_brake_only",
    "cut_in_only",
    "pedestrian_only",
]
DEFAULT_BUDGETS = [12, 24, 36, 48, 72, 96, 144, 192, 288]
DEFAULT_RANDOM_SEEDS = list(range(10))
RUNTIME_RANK_STRATEGY = "budget_ranked"


# -----------------------------------------------------------------------------
# Basic IO / parsing helpers
# -----------------------------------------------------------------------------

def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        return None if math.isnan(v) or math.isinf(v) else v
    except Exception:
        return None


def _bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False).astype(bool)
    return s.fillna(False).map(lambda x: str(x).strip().lower() in {"true", "1", "yes", "y"})


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _stable_scene_hash(scene_id: str) -> int:
    value = 0
    for ch in str(scene_id):
        value = (value * 131 + ord(ch)) % 1_000_003
    return value


def _candidate_table_count(root: Path, methods: Iterable[str]) -> int:
    return sum(len(list((root / m).glob("*/candidate_table.csv"))) for m in methods if (root / m).exists())


def resolve_baseline_candidate_root(baseline_out_dir: str | Path, methods: List[str]) -> Tuple[Path, Dict[str, Any]]:
    """Resolve where candidate tables actually live.

    Accepted layouts:
      baseline_ablation/<method>/<scene>/candidate_table.csv
      baseline_ablation/per_method/<method>/<scene>/candidate_table.csv
      per_method/<method>/<scene>/candidate_table.csv
    """
    baseline_out_dir = Path(baseline_out_dir)
    roots = [baseline_out_dir]
    if baseline_out_dir.name != "per_method":
        roots.append(baseline_out_dir / "per_method")
    checked = [{"root": str(r), "candidate_tables_found": _candidate_table_count(r, methods)} for r in roots]
    selected = max(checked, key=lambda x: x["candidate_tables_found"])
    diagnostic = {
        "input_baseline_out_dir": str(baseline_out_dir),
        "selected_candidate_root": selected["root"],
        "candidate_tables_found": int(selected["candidate_tables_found"]),
        "checked_layouts": checked,
        "per_method_exists": bool((baseline_out_dir / "per_method").exists()),
        "layout_note": "candidate root resolved successfully" if selected["candidate_tables_found"] else "No candidate_table.csv files were found. Pass baseline_ablation or baseline_ablation/per_method.",
    }
    return Path(selected["root"]), diagnostic


# -----------------------------------------------------------------------------
# Full-budget method audit
# -----------------------------------------------------------------------------

def _empty_scene_table() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "method", "scene_id", "candidate_table_path", "candidate_table_exists", "candidate_table_nonempty",
        "num_evaluated_rows", "num_unique_target_agents", "num_unique_edit_names", "best_found",
        "best_cost", "best_edit_name", "best_target_agent_id", "first_failure_eval_index", "best_eval_index", "num_failure_rows",
    ])


def _infer_best_from_table(df: pd.DataFrame) -> Dict[str, Any]:
    empty = {
        "best_found": False,
        "best_cost": None,
        "best_edit_name": None,
        "best_target_agent_id": None,
        "first_failure_eval_index": None,
        "best_eval_index": None,
        "num_failure_rows": 0,
    }
    if df.empty or "failure" not in df.columns:
        return empty
    mask = _bool_series(df["failure"])
    if int(mask.sum()) == 0:
        return empty
    failures = df[mask].copy()
    failures["_row_index"] = failures.index.astype(int) + 1
    sort_cols, ascending = [], []
    if "cost" in failures.columns:
        sort_cols.append("cost"); ascending.append(True)
    if "min_ttc" in failures.columns:
        sort_cols.append("min_ttc"); ascending.append(True)
    if "risk_score" in failures.columns:
        sort_cols.append("risk_score"); ascending.append(False)
    if sort_cols:
        failures = failures.sort_values(sort_cols, ascending=ascending)
    best = failures.iloc[0]
    return {
        "best_found": True,
        "best_cost": _to_float(best.get("cost")),
        "best_edit_name": best.get("edit_name"),
        "best_target_agent_id": str(best.get("target_agent_id")) if pd.notna(best.get("target_agent_id")) else None,
        "first_failure_eval_index": int(df.index[mask][0]) + 1,
        "best_eval_index": int(best.get("_row_index")) if pd.notna(best.get("_row_index")) else None,
        "num_failure_rows": int(mask.sum()),
    }


def _scene_dirs_for_method(root: Path, method: str) -> List[Path]:
    method_dir = root / method
    return sorted([p for p in method_dir.iterdir() if p.is_dir()]) if method_dir.exists() else []


def _audit_method_scene(method: str, scene_dir: Path) -> Dict[str, Any]:
    table_path = scene_dir / "candidate_table.csv"
    df = _safe_read_csv(table_path)
    row: Dict[str, Any] = {
        "method": method,
        "scene_id": scene_dir.name,
        "candidate_table_path": str(table_path),
        "candidate_table_exists": table_path.exists(),
        "candidate_table_nonempty": not df.empty,
        "num_evaluated_rows": int(len(df)),
        "num_unique_target_agents": int(df["target_agent_id"].nunique()) if "target_agent_id" in df.columns and not df.empty else 0,
        "num_unique_edit_names": int(df["edit_name"].nunique()) if "edit_name" in df.columns and not df.empty else 0,
        **_infer_best_from_table(df),
    }
    if not df.empty and "hybrid_candidate_source" in df.columns:
        for src, count in df["hybrid_candidate_source"].fillna("unknown").value_counts().to_dict().items():
            row[f"hybrid_rows_source_{src}"] = int(count)
        if "target_agent_id" in df.columns:
            tmp = df[["hybrid_candidate_source", "target_agent_id"]].dropna().drop_duplicates()
            for src, count in tmp["hybrid_candidate_source"].fillna("unknown").value_counts().to_dict().items():
                row[f"hybrid_agents_source_{src}"] = int(count)
        if row["best_found"]:
            failures = df[_bool_series(df["failure"])].copy()
            failures["_row_index"] = failures.index.astype(int) + 1
            sort_cols = [c for c in ["cost", "min_ttc", "risk_score"] if c in failures.columns]
            if sort_cols:
                failures = failures.sort_values(sort_cols, ascending=[True, True, False][:len(sort_cols)])
            brow = failures.iloc[0]
            row["best_hybrid_candidate_source"] = brow.get("hybrid_candidate_source")
            row["best_hybrid_allowed_edit"] = brow.get("hybrid_allowed_edit")
    return row


def build_scene_efficiency_table(candidate_root: Path, methods: List[str]) -> pd.DataFrame:
    rows = [_audit_method_scene(m, d) for m in methods for d in _scene_dirs_for_method(candidate_root, m)]
    return pd.DataFrame(rows) if rows else _empty_scene_table()


def summarize_efficiency(scene_table: pd.DataFrame) -> pd.DataFrame:
    if scene_table.empty or "method" not in scene_table.columns:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for method, g in scene_table.groupby("method", dropna=False):
        n = len(g)
        found = g["best_found"].fillna(False).astype(bool)
        total_evals = int(pd.to_numeric(g["num_evaluated_rows"], errors="coerce").fillna(0).sum())
        num_found = int(found.sum())
        first_failure = pd.to_numeric(g.loc[found, "first_failure_eval_index"], errors="coerce").dropna()
        costs = pd.to_numeric(g.loc[found, "best_cost"], errors="coerce").dropna()
        rows.append({
            "method": method,
            "num_scenes": int(n),
            "num_found": num_found,
            "attack_success_rate": float(num_found / n) if n else None,
            "total_evaluated_rows": total_evals,
            "mean_evaluated_rows_per_scene": float(pd.to_numeric(g["num_evaluated_rows"], errors="coerce").mean()),
            "median_evaluated_rows_per_scene": float(pd.to_numeric(g["num_evaluated_rows"], errors="coerce").median()),
            "mean_first_failure_eval_index_success_only": float(first_failure.mean()) if len(first_failure) else None,
            "median_first_failure_eval_index_success_only": float(first_failure.median()) if len(first_failure) else None,
            "mean_best_cost_success_only": float(costs.mean()) if len(costs) else None,
            "median_best_cost_success_only": float(costs.median()) if len(costs) else None,
            "evals_per_verified_failure": float(total_evals / num_found) if num_found else None,
            "verified_failures_per_1000_evals": float(num_found * 1000.0 / total_evals) if total_evals else None,
            "mean_unique_target_agents": float(pd.to_numeric(g["num_unique_target_agents"], errors="coerce").mean()),
            "mean_unique_edit_names": float(pd.to_numeric(g["num_unique_edit_names"], errors="coerce").mean()),
        })
    return pd.DataFrame(rows).sort_values(["attack_success_rate", "verified_failures_per_1000_evals"], ascending=[False, False])


def build_pair_overlap(scene_table: pd.DataFrame, primary: str, reference: str) -> pd.DataFrame:
    if scene_table.empty or "method" not in scene_table.columns:
        return pd.DataFrame()
    cols = ["scene_id", "best_found", "best_cost", "best_edit_name", "best_target_agent_id", "num_evaluated_rows"]
    a = scene_table[scene_table["method"] == primary][cols].copy().rename(columns={c: f"{primary}_{c}" for c in cols if c != "scene_id"})
    b = scene_table[scene_table["method"] == reference][cols].copy().rename(columns={c: f"{reference}_{c}" for c in cols if c != "scene_id"})
    merged = a.merge(b, on="scene_id", how="outer")
    if merged.empty:
        return merged

    def col(name: str, default: Any) -> pd.Series:
        return merged[name] if name in merged.columns else pd.Series([default] * len(merged), index=merged.index)

    merged["same_found"] = col(f"{primary}_best_found", False).fillna(False).astype(bool) == col(f"{reference}_best_found", False).fillna(False).astype(bool)
    merged["same_edit"] = col(f"{primary}_best_edit_name", "NA").fillna("NA") == col(f"{reference}_best_edit_name", "NA").fillna("NA")
    merged["same_target"] = col(f"{primary}_best_target_agent_id", "NA").fillna("NA").astype(str) == col(f"{reference}_best_target_agent_id", "NA").fillna("NA").astype(str)
    pa = pd.to_numeric(col(f"{primary}_best_cost", None), errors="coerce")
    rb = pd.to_numeric(col(f"{reference}_best_cost", None), errors="coerce")
    merged["cost_abs_delta"] = (pa - rb).abs()
    merged["same_cost_1e_6"] = (merged["cost_abs_delta"].fillna(0.0) < 1e-6) | (pa.isna() & rb.isna())
    merged["same_outcome_strict"] = merged["same_found"] & merged["same_edit"] & merged["same_target"] & merged["same_cost_1e_6"]
    return merged


def summarize_pair_overlap(pair: pd.DataFrame, primary: str, reference: str) -> Dict[str, Any]:
    if pair.empty:
        return {}
    out = {"primary_method": primary, "reference_method": reference, "num_scenes": int(len(pair))}
    for col in ["same_found", "same_edit", "same_target", "same_cost_1e_6", "same_outcome_strict"]:
        if col in pair.columns:
            out[f"{col}_rate"] = float(pair[col].mean())
    for col in [f"{primary}_num_evaluated_rows", f"{reference}_num_evaluated_rows"]:
        if col in pair.columns:
            out[f"mean_{col}"] = float(pd.to_numeric(pair[col], errors="coerce").mean())
    return out


def build_hybrid_source_summary(scene_table: pd.DataFrame) -> Dict[str, Any]:
    if scene_table.empty or "method" not in scene_table.columns:
        return {}
    h = scene_table[scene_table["method"] == "causal_hybrid"].copy()
    if h.empty:
        return {}
    out: Dict[str, Any] = {"num_scenes": int(len(h))}
    for col in h.columns:
        if col.startswith("hybrid_rows_source_") or col.startswith("hybrid_agents_source_"):
            nums = pd.to_numeric(h[col], errors="coerce").fillna(0)
            out[f"total_{col}"] = int(nums.sum())
            out[f"mean_{col}"] = float(nums.mean())
    if "best_hybrid_candidate_source" in h.columns:
        out["best_source_counts"] = h.loc[h["best_found"].fillna(False).astype(bool), "best_hybrid_candidate_source"].fillna("unknown").value_counts().to_dict()
    return out


# -----------------------------------------------------------------------------
# Budgeted curves and AUC
# -----------------------------------------------------------------------------

def _budget_row(method: str, scene_id: str, df: pd.DataFrame, k: int, path: Path, *, seed: Optional[int] = None, rank_strategy: str = "current_order") -> Dict[str, Any]:
    best = _infer_best_from_table(df.head(k) if not df.empty else df)
    out = {
        "method": method,
        "scene_id": scene_id,
        "budget_k": int(k),
        "num_available_rows": int(len(df)),
        "num_evaluated_at_k": int(min(len(df), k)),
        "success_at_k": bool(best["best_found"]),
        "best_cost_at_k": best["best_cost"],
        "best_edit_name_at_k": best["best_edit_name"],
        "best_target_agent_id_at_k": best["best_target_agent_id"],
        "first_failure_eval_index_at_k": best["first_failure_eval_index"],
        "num_failure_rows_at_k": best["num_failure_rows"],
        "candidate_table_path": str(path),
        "rank_strategy": rank_strategy,
    }
    if seed is not None:
        out["seed"] = int(seed)
    return out


def build_budgeted_success_curve(candidate_root: Path, methods: List[str], budgets: List[int]) -> pd.DataFrame:
    rows = []
    for method in methods:
        for scene_dir in _scene_dirs_for_method(candidate_root, method):
            p = scene_dir / "candidate_table.csv"
            df = _safe_read_csv(p)
            rows.extend(_budget_row(method, scene_dir.name, df, k, p) for k in budgets)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["method", "scene_id", "budget_k", "success_at_k"])


def summarize_budgeted_success(curve: pd.DataFrame) -> pd.DataFrame:
    if curve.empty or not {"method", "budget_k", "success_at_k"}.issubset(curve.columns):
        return pd.DataFrame()
    group_cols = ["method", "budget_k"]
    if "rank_strategy" in curve.columns and curve["rank_strategy"].nunique(dropna=False) > 1:
        group_cols = ["rank_strategy", "method", "budget_k"]
    rows = []
    for key, g in curve.groupby(group_cols, dropna=False):
        if len(group_cols) == 2:
            method, k = key
            row: Dict[str, Any] = {"method": method, "budget_k": int(k)}
        else:
            rank_strategy, method, k = key
            row = {"rank_strategy": rank_strategy, "method": method, "budget_k": int(k)}
        success = g["success_at_k"].fillna(False).astype(bool)
        costs = pd.to_numeric(g.loc[success, "best_cost_at_k"], errors="coerce").dropna() if "best_cost_at_k" in g else pd.Series(dtype=float)
        first = pd.to_numeric(g.loc[success, "first_failure_eval_index_at_k"], errors="coerce").dropna() if "first_failure_eval_index_at_k" in g else pd.Series(dtype=float)
        row.update({
            "num_scenes": int(len(g)),
            "num_success_at_k": int(success.sum()),
            "success_rate_at_k": float(success.mean()) if len(g) else None,
            "mean_best_cost_success_at_k": float(costs.mean()) if len(costs) else None,
            "median_best_cost_success_at_k": float(costs.median()) if len(costs) else None,
            "mean_first_failure_eval_index_success_at_k": float(first.mean()) if len(first) else None,
            "mean_num_evaluated_at_k": float(pd.to_numeric(g.get("num_evaluated_at_k", pd.Series(dtype=float)), errors="coerce").mean()),
        })
        rows.append(row)
    sort_cols = [c for c in ["rank_strategy", "budget_k", "success_rate_at_k", "method"] if c in (rows[0] if rows else {})]
    asc = [True if c != "success_rate_at_k" else False for c in sort_cols]
    return pd.DataFrame(rows).sort_values(sort_cols, ascending=asc) if rows else pd.DataFrame()


def compute_budget_auc(summary: pd.DataFrame, *, rate_col: str = "success_rate_at_k", group_cols: Optional[List[str]] = None) -> pd.DataFrame:
    if summary.empty or rate_col not in summary.columns or "budget_k" not in summary.columns:
        return pd.DataFrame()
    group_cols = group_cols or (["rank_strategy", "method"] if "rank_strategy" in summary.columns else ["method"])
    rows: List[Dict[str, Any]] = []
    for key, g in summary.groupby(group_cols, dropna=False):
        g = g.sort_values("budget_k")
        xs = pd.to_numeric(g["budget_k"], errors="coerce").to_list()
        ys = pd.to_numeric(g[rate_col], errors="coerce").fillna(0.0).to_list()
        if len(xs) < 2:
            auc_raw = 0.0
            auc_norm = ys[0] if ys else None
        else:
            auc_raw = 0.0
            for i in range(1, len(xs)):
                auc_raw += (xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1]) / 2.0
            denom = max(xs) - min(xs)
            auc_norm = auc_raw / denom if denom > 0 else None
        row: Dict[str, Any] = {}
        if isinstance(key, tuple):
            for c, v in zip(group_cols, key):
                row[c] = v
        else:
            row[group_cols[0]] = key
        row.update({
            "min_budget": int(min(xs)) if xs else None,
            "max_budget": int(max(xs)) if xs else None,
            "auc_raw": float(auc_raw),
            "auc_normalized": float(auc_norm) if auc_norm is not None else None,
            "success_rate_at_min_budget": float(ys[0]) if ys else None,
            "success_rate_at_max_budget": float(ys[-1]) if ys else None,
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values("auc_normalized", ascending=False) if rows else pd.DataFrame()


# -----------------------------------------------------------------------------
# public_release random multi-seed audit
# -----------------------------------------------------------------------------

def _random_permutation(df: pd.DataFrame, scene_id: str, seed: int) -> pd.DataFrame:
    if df.empty:
        return df
    rng = random.Random(int(seed) + _stable_scene_hash(scene_id))
    idx = list(df.index)
    rng.shuffle(idx)
    return df.loc[idx].reset_index(drop=True)


def build_random_multiseed_curve(
    candidate_root: Path,
    reference_method: str,
    budgets: List[int],
    seeds: List[int],
) -> pd.DataFrame:
    """Post-hoc random-budget audit from a complete candidate table.

    This uses an existing complete candidate table, typically `distance_all`, as the
    admissible candidate universe. For each seed and scene it draws one random
    permutation and computes success@K on the prefix. This gives mean/std random
    baselines without rerunning expensive scene evaluation.
    """
    rows: List[Dict[str, Any]] = []
    for scene_dir in _scene_dirs_for_method(candidate_root, reference_method):
        p = scene_dir / "candidate_table.csv"
        base_df = _safe_read_csv(p)
        for seed in seeds:
            shuffled = _random_permutation(base_df, scene_dir.name, seed)
            for k in budgets:
                row = _budget_row("random_multiseed", scene_dir.name, shuffled, k, p, seed=seed, rank_strategy="random_permutation")
                row["reference_method"] = reference_method
                rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["method", "scene_id", "seed", "budget_k", "success_at_k"])


def summarize_random_multiseed(curve: pd.DataFrame) -> pd.DataFrame:
    if curve.empty:
        return pd.DataFrame()
    per_seed_rows: List[Dict[str, Any]] = []
    for (seed, k), g in curve.groupby(["seed", "budget_k"], dropna=False):
        success = g["success_at_k"].fillna(False).astype(bool)
        per_seed_rows.append({
            "seed": int(seed),
            "budget_k": int(k),
            "num_scenes": int(len(g)),
            "success_rate_at_k": float(success.mean()) if len(g) else None,
            "num_success_at_k": int(success.sum()),
        })
    per_seed = pd.DataFrame(per_seed_rows)
    rows: List[Dict[str, Any]] = []
    for k, g in per_seed.groupby("budget_k", dropna=False):
        vals = pd.to_numeric(g["success_rate_at_k"], errors="coerce").dropna()
        std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append({
            "method": "random_multiseed",
            "budget_k": int(k),
            "num_seeds": int(len(vals)),
            "num_scenes": int(per_seed.loc[per_seed["budget_k"] == k, "num_scenes"].max()) if len(g) else 0,
            "mean_success_rate_at_k": float(vals.mean()) if len(vals) else None,
            "std_success_rate_at_k": std,
            "ci95_success_rate_at_k": float(1.96 * std / math.sqrt(len(vals))) if len(vals) else None,
            "min_success_rate_at_k": float(vals.min()) if len(vals) else None,
            "max_success_rate_at_k": float(vals.max()) if len(vals) else None,
            "mean_num_success_at_k": float(pd.to_numeric(g["num_success_at_k"], errors="coerce").mean()) if len(g) else None,
        })
    return pd.DataFrame(rows).sort_values("budget_k") if rows else pd.DataFrame()


# -----------------------------------------------------------------------------
# public_release ranking-strategy audit
# -----------------------------------------------------------------------------

def _edit_priority_value(edit_name: Any) -> int:
    s = str(edit_name or "").strip()
    return {"cut_in": 0, "pedestrian_crossing": 1, "lead_brake": 2}.get(s, 9)


def _rank_candidates_for_strategy(df: pd.DataFrame, method: str, strategy: str) -> pd.DataFrame:
    """Reorder already evaluated rows using only pre-outcome candidate metadata.

    `budget_ranked` is the runtime-safe ordering. It never uses
    failure, min_ttc, collision, hard_brake, or risk_score. `cost_only_diagnostic`
    is also outcome-free but intentionally weaker as a diagnostic of whether the
    intervention-cost schedule alone explains early discovery.
    """
    if df.empty or strategy == "current_order":
        return df.copy()
    out = df.copy()

    if "cost" in out.columns:
        out["_rank_cost"] = pd.to_numeric(out["cost"], errors="coerce").fillna(1e9)
    else:
        out["_rank_cost"] = 1e9
    if "edit_name" in out.columns:
        out["_rank_edit_priority"] = out["edit_name"].map(_edit_priority_value)
    else:
        out["_rank_edit_priority"] = 9

    if strategy == "cost_only_diagnostic":
        return out.sort_values(["_rank_cost", "_rank_edit_priority"], ascending=[True, True]).reset_index(drop=True)

    if strategy not in {"budget_ranked"}:
        return out.reset_index(drop=True)

    if method == "causal_hybrid":
        source = out.get("hybrid_candidate_source", pd.Series(["unknown"] * len(out), index=out.index)).fillna("unknown")
        out["_rank_source_priority"] = source.map(lambda x: 0 if str(x) == "causal_graph" else 1)
        out["_rank_distance"] = pd.to_numeric(out.get("hybrid_distance", pd.Series([1e9] * len(out), index=out.index)), errors="coerce").fillna(1e9)
        # public_release: budget-aware hybrid order. Causal coverage is kept through source
        # priority, but low-budget evaluation is ordered by geometric proximity and
        # intervention cost instead of raw graph rank.
        return out.sort_values(
            ["_rank_source_priority", "_rank_distance", "_rank_cost", "_rank_edit_priority"],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)

    if method == "distance_all":
        out["_rank_distance"] = pd.to_numeric(out.get("baseline_distance", pd.Series([1e9] * len(out), index=out.index)), errors="coerce").fillna(1e9)
        return out.sort_values(["_rank_distance", "_rank_cost", "_rank_edit_priority"], ascending=[True, True, True]).reset_index(drop=True)

    # Single-edit baselines do not have cross-edit routing; sort by intervention cost.
    return out.sort_values(["_rank_cost", "_rank_edit_priority"], ascending=[True, True]).reset_index(drop=True)


def build_ranked_budgeted_success_curve(
    candidate_root: Path,
    methods: List[str],
    budgets: List[int],
    rank_strategies: List[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for strategy in rank_strategies:
        for method in methods:
            for scene_dir in _scene_dirs_for_method(candidate_root, method):
                p = scene_dir / "candidate_table.csv"
                df = _safe_read_csv(p)
                ranked = _rank_candidates_for_strategy(df, method, strategy)
                rows.extend(_budget_row(method, scene_dir.name, ranked, k, p, rank_strategy=strategy) for k in budgets)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["rank_strategy", "method", "scene_id", "budget_k", "success_at_k"])



# -----------------------------------------------------------------------------
# public_release runtime-rank diagnostic
# -----------------------------------------------------------------------------

def build_runtime_rank_diagnostic(candidate_root: Path, methods: List[str]) -> Dict[str, Any]:
    """Check whether candidate tables were produced by a fresh public_release ranked run.

    public_release could diagnose a better ranking post hoc.  For report claims, the
    improved order must be produced by the baseline runner itself.  This
    diagnostic distinguishes fresh runtime-ranked tables from older candidate
    tables that lack rank metadata.
    """
    rows: List[Dict[str, Any]] = []
    for method in methods:
        for scene_dir in _scene_dirs_for_method(candidate_root, method):
            p = scene_dir / "candidate_table.csv"
            df = _safe_read_csv(p)
            # public_release treats header-only candidate tables as runtime-rank aware.
            # Empty tables occur naturally for single-edit ablations when no agent is
            # physically admissible, so requiring non-empty rows caused false negative
            # runtime-rank diagnostics in public_release.
            has_strategy = "runtime_rank_strategy" in df.columns
            strategy_vals = []
            version_vals = []
            if has_strategy and not df.empty:
                strategy_vals = sorted([str(x) for x in df["runtime_rank_strategy"].dropna().unique().tolist()])
            if "runtime_rank_version" in df.columns and not df.empty:
                version_vals = sorted([str(x) for x in df["runtime_rank_version"].dropna().unique().tolist()])
            monotonic = False
            if "candidate_eval_order" in df.columns:
                if df.empty:
                    monotonic = True
                else:
                    order = pd.to_numeric(df["candidate_eval_order"], errors="coerce").dropna().astype(int).tolist()
                    monotonic = order == list(range(1, len(order) + 1))
            rows.append({
                "method": method,
                "scene_id": scene_dir.name,
                "candidate_table_path": str(p),
                "num_rows": int(len(df)),
                "has_runtime_rank_strategy": bool(has_strategy),
                "runtime_rank_strategy_values": strategy_vals,
                "runtime_rank_version_values": version_vals,
                "candidate_eval_order_monotonic": bool(monotonic),
            })
    if not rows:
        return {
            "num_candidate_tables": 0,
            "num_with_runtime_rank_strategy": 0,
            "num_with_expected_runtime_rank_strategy": 0,
            "runtime_rank_ready": False,
            "warning": "No candidate tables found.",
        }
    df = pd.DataFrame(rows)
    has = df["has_runtime_rank_strategy"].fillna(False).astype(bool)
    expected = df["runtime_rank_strategy_values"].map(lambda vals: RUNTIME_RANK_STRATEGY in vals if isinstance(vals, list) else False)
    monotonic = df["candidate_eval_order_monotonic"].fillna(False).astype(bool)
    method_counts = df.groupby("method")["has_runtime_rank_strategy"].sum().to_dict()
    method_table_counts_raw = df.groupby("method").size().to_dict()
    core_methods = [m for m in ["causal_hybrid", "distance_all"] if m in set(df["method"].astype(str))]
    core_df = df[df["method"].isin(core_methods)].copy() if core_methods else pd.DataFrame()
    core_ready = False
    if not core_df.empty:
        core_ready = bool(
            core_df["has_runtime_rank_strategy"].fillna(False).astype(bool).all()
            and core_df["candidate_eval_order_monotonic"].fillna(False).astype(bool).all()
        )
    all_ready = bool(int(has.sum()) == int(len(df)) and int(monotonic.sum()) == int(len(df)))
    out: Dict[str, Any] = {
        "num_candidate_tables": int(len(df)),
        "num_with_runtime_rank_strategy": int(has.sum()),
        "num_with_expected_budget_rank_strategy": int(expected.sum()),
        "num_with_monotonic_candidate_eval_order": int(monotonic.sum()),
        "method_table_counts": {str(k): int(v) for k, v in method_table_counts_raw.items()},
        "method_runtime_rank_counts": {str(k): int(v) for k, v in method_counts.items()},
        "runtime_rank_ready": all_ready,
        "runtime_rank_ready_core_methods": core_ready,
        "runtime_rank_core_methods": core_methods,
        "empty_tables_treated_as_rank_ready_if_header_present": True,
    }
    if not out["runtime_rank_ready"]:
        out["warning"] = (
            "Some candidate tables do not contain public_release runtime rank metadata. "
            "Budget-ranked curves may still be post-hoc diagnostics unless you rerun the baseline with public_release."
        )
    return out

# -----------------------------------------------------------------------------
# Report and main generator
# -----------------------------------------------------------------------------

def _md_table(df: pd.DataFrame, cols: List[str], floatfmt: str = ".3f") -> str:
    if df.empty:
        return "No data available."
    return df[[c for c in cols if c in df.columns]].to_markdown(index=False, floatfmt=floatfmt)


def make_report(
    summary: pd.DataFrame,
    pair_summary: Dict[str, Any],
    hybrid_source: Dict[str, Any],
    layout: Dict[str, Any],
    budget_summary: pd.DataFrame,
    budgets: List[int],
    auc_summary: pd.DataFrame,
    random_summary: pd.DataFrame,
    random_auc: pd.DataFrame,
    ranked_summary: pd.DataFrame,
    ranked_auc: pd.DataFrame,
    random_reference_method: str,
    random_seeds: List[int],
    runtime_diag: Optional[Dict[str, Any]] = None,
) -> str:
    lines = ["# CausalSensor4D public_release Runtime-Ranked Budgeted Audit", ""]
    lines.append("public_release keeps the path fix, multi-seed random auditing, and budgeted AUC, and adds a runtime-rank diagnostic for freshly re-run candidate tables.")
    lines += ["", "## Candidate-table layout diagnostic"]
    for k, v in layout.items():
        lines.append(f"- {k}: `{v}`")

    lines += ["", "## Runtime-rank diagnostic"]
    if runtime_diag:
        for k, v in runtime_diag.items():
            lines.append(f"- {k}: `{v}`")
    else:
        lines.append("No runtime-rank diagnostic available.")

    lines += ["", "## Full-budget method-level search efficiency"]
    cols = ["method", "num_scenes", "num_found", "attack_success_rate", "total_evaluated_rows", "mean_evaluated_rows_per_scene", "mean_first_failure_eval_index_success_only", "evals_per_verified_failure", "verified_failures_per_1000_evals", "mean_best_cost_success_only"]
    lines.append(_md_table(summary, cols))

    lines += ["", "## Current-order budgeted success summary", f"Budgets: `{budgets}`"]
    display = budget_summary[budget_summary["method"].isin(["causal_hybrid", "distance_all", "random_budget", "causal_guided"])] if not budget_summary.empty else budget_summary
    cols = ["method", "budget_k", "num_scenes", "num_success_at_k", "success_rate_at_k", "mean_best_cost_success_at_k", "mean_first_failure_eval_index_success_at_k"]
    lines.append(_md_table(display, cols))

    lines += ["", "## Current-order budgeted AUC"]
    cols = ["method", "auc_normalized", "success_rate_at_min_budget", "success_rate_at_max_budget"]
    lines.append(_md_table(auc_summary, cols))

    lines += ["", "## Random multi-seed post-hoc audit"]
    lines.append(f"Reference candidate universe: `{random_reference_method}`. Seeds: `{random_seeds}`.")
    cols = ["method", "budget_k", "num_seeds", "mean_success_rate_at_k", "std_success_rate_at_k", "ci95_success_rate_at_k", "min_success_rate_at_k", "max_success_rate_at_k"]
    lines.append(_md_table(random_summary, cols))
    lines += ["", "### Random multi-seed AUC"]
    cols = ["method", "auc_normalized", "success_rate_at_min_budget", "success_rate_at_max_budget"]
    lines.append(_md_table(random_auc, cols))

    lines += ["", "## Ranking-strategy diagnostics"]
    if ranked_summary.empty:
        lines.append("No ranking-strategy summary available.")
    else:
        display_ranked = ranked_summary[
            ranked_summary["method"].isin(["causal_hybrid", "distance_all", "causal_guided", "random_budget"])
        ].copy()
        cols = ["rank_strategy", "method", "budget_k", "success_rate_at_k", "mean_first_failure_eval_index_success_at_k"]
        lines.append(_md_table(display_ranked, cols))
    lines += ["", "### Ranking-strategy AUC"]
    cols = ["rank_strategy", "method", "auc_normalized", "success_rate_at_min_budget", "success_rate_at_max_budget"]
    lines.append(_md_table(ranked_auc, cols))

    lines += ["", "## Causal-hybrid vs distance_all overlap"]
    lines += [f"- {k}: `{v}`" for k, v in pair_summary.items()] if pair_summary else ["No pair summary available."]
    lines += ["", "## Hybrid candidate source audit"]
    lines += [f"- {k}: `{v}`" for k, v in hybrid_source.items()] if hybrid_source else ["No hybrid source summary available."]

    lines += ["", "## Report interpretation"]
    lines.append("Use the full-budget table for final failure discovery. Use budgeted_method_summary.csv and budgeted_auc_summary.csv for early-budget efficiency claims. Use random_multiseed_budgeted_summary.csv before making any claim against random search. Use ranked_budgeted_method_summary.csv as a diagnostic for old tables. If runtime_rank_diagnostic.json reports runtime_rank_ready=True, the current-order budgeted curves are produced by the fresh public_release runtime-ranked baseline and can be used as report evidence.")
    return "\n".join(lines)


def generate_causal_hybrid_audit(
    baseline_out_dir: str | Path,
    out_dir: str | Path,
    methods: Optional[List[str]] = None,
    primary_method: str = "causal_hybrid",
    reference_method: str = "distance_all",
    budgets: Optional[List[int]] = None,
    random_seeds: Optional[List[int]] = None,
    random_reference_method: str = "distance_all",
    rank_strategies: Optional[List[str]] = None,
) -> Dict[str, Any]:
    methods = methods or list(DEFAULT_METHODS)
    budgets = budgets or list(DEFAULT_BUDGETS)
    random_seeds = random_seeds or list(DEFAULT_RANDOM_SEEDS)
    rank_strategies = rank_strategies or ["current_order", "budget_ranked", "cost_only_diagnostic"]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_root, layout = resolve_baseline_candidate_root(baseline_out_dir, methods)

    scene_table = build_scene_efficiency_table(candidate_root, methods)
    summary = summarize_efficiency(scene_table)
    pair = build_pair_overlap(scene_table, primary_method, reference_method)
    pair_summary = summarize_pair_overlap(pair, primary_method, reference_method)
    hybrid_source = build_hybrid_source_summary(scene_table)
    runtime_diag = build_runtime_rank_diagnostic(candidate_root, methods)

    budget_curve = build_budgeted_success_curve(candidate_root, methods, budgets)
    budget_summary = summarize_budgeted_success(budget_curve)
    auc_summary = compute_budget_auc(budget_summary)

    random_curve = build_random_multiseed_curve(candidate_root, random_reference_method, budgets, random_seeds)
    random_summary = summarize_random_multiseed(random_curve)
    random_auc_input = random_summary.rename(columns={"mean_success_rate_at_k": "success_rate_at_k"}) if not random_summary.empty else pd.DataFrame()
    random_auc = compute_budget_auc(random_auc_input, group_cols=["method"])

    ranked_curve = build_ranked_budgeted_success_curve(candidate_root, methods, budgets, rank_strategies)
    ranked_summary = summarize_budgeted_success(ranked_curve)
    ranked_auc = compute_budget_auc(ranked_summary, group_cols=["rank_strategy", "method"] if "rank_strategy" in ranked_summary.columns else ["method"])

    outputs = {
        "scene_table": out_dir / "search_efficiency_scene_table.csv",
        "method_summary": out_dir / "search_efficiency_method_summary.csv",
        "pair_overlap": out_dir / "causal_hybrid_vs_distance_overlap.csv",
        "hybrid_source_summary": out_dir / "causal_hybrid_source_summary.json",
        "candidate_table_layout_diagnostic": out_dir / "candidate_table_layout_diagnostic.json",
        "runtime_rank_diagnostic": out_dir / "runtime_rank_diagnostic.json",
        "budgeted_success_curve": out_dir / "budgeted_success_curve.csv",
        "budgeted_method_summary": out_dir / "budgeted_method_summary.csv",
        "budgeted_auc_summary": out_dir / "budgeted_auc_summary.csv",
        "random_multiseed_budgeted_success_curve": out_dir / "random_multiseed_budgeted_success_curve.csv",
        "random_multiseed_budgeted_summary": out_dir / "random_multiseed_budgeted_summary.csv",
        "random_multiseed_auc_summary": out_dir / "random_multiseed_auc_summary.csv",
        "ranked_budgeted_success_curve": out_dir / "ranked_budgeted_success_curve.csv",
        "ranked_budgeted_method_summary": out_dir / "ranked_budgeted_method_summary.csv",
        "ranked_budgeted_auc_summary": out_dir / "ranked_budgeted_auc_summary.csv",
        "report": out_dir / "causal_hybrid_audit_report.md",
    }
    scene_table.to_csv(outputs["scene_table"], index=False)
    summary.to_csv(outputs["method_summary"], index=False)
    pair.to_csv(outputs["pair_overlap"], index=False)
    budget_curve.to_csv(outputs["budgeted_success_curve"], index=False)
    budget_summary.to_csv(outputs["budgeted_method_summary"], index=False)
    auc_summary.to_csv(outputs["budgeted_auc_summary"], index=False)
    random_curve.to_csv(outputs["random_multiseed_budgeted_success_curve"], index=False)
    random_summary.to_csv(outputs["random_multiseed_budgeted_summary"], index=False)
    random_auc.to_csv(outputs["random_multiseed_auc_summary"], index=False)
    ranked_curve.to_csv(outputs["ranked_budgeted_success_curve"], index=False)
    ranked_summary.to_csv(outputs["ranked_budgeted_method_summary"], index=False)
    ranked_auc.to_csv(outputs["ranked_budgeted_auc_summary"], index=False)
    outputs["hybrid_source_summary"].write_text(json.dumps(hybrid_source, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs["candidate_table_layout_diagnostic"].write_text(json.dumps(layout, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs["runtime_rank_diagnostic"].write_text(json.dumps(runtime_diag, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs["report"].write_text(
        make_report(
            summary=summary,
            pair_summary=pair_summary,
            hybrid_source=hybrid_source,
            layout=layout,
            budget_summary=budget_summary,
            budgets=budgets,
            auc_summary=auc_summary,
            random_summary=random_summary,
            random_auc=random_auc,
            ranked_summary=ranked_summary,
            ranked_auc=ranked_auc,
            random_reference_method=random_reference_method,
            random_seeds=random_seeds,
            runtime_diag=runtime_diag,
        ),
        encoding="utf-8",
    )

    payload = {
        "version": "public_release",
        "baseline_out_dir": str(baseline_out_dir),
        "candidate_root": str(candidate_root),
        "layout_diagnostic": layout,
        "runtime_rank_diagnostic": runtime_diag,
        "primary_method": primary_method,
        "reference_method": reference_method,
        "methods": methods,
        "budgets": budgets,
        "random_reference_method": random_reference_method,
        "random_seeds": random_seeds,
        "rank_strategies": rank_strategies,
        "method_summary": summary.to_dict(orient="records"),
        "budgeted_method_summary": budget_summary.to_dict(orient="records"),
        "budgeted_auc_summary": auc_summary.to_dict(orient="records"),
        "random_multiseed_budgeted_summary": random_summary.to_dict(orient="records"),
        "random_multiseed_auc_summary": random_auc.to_dict(orient="records"),
        "ranked_budgeted_auc_summary": ranked_auc.to_dict(orient="records"),
        "pair_summary": pair_summary,
        "hybrid_source_summary": hybrid_source,
        "outputs": {k: str(v) for k, v in outputs.items()},
    }
    (out_dir / "causal_hybrid_audit_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
