from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import json
import math
import pandas as pd


DEFAULT_CENSORED_MFC = 2.0


def summarize_planner(summary: pd.DataFrame, planner: str, censored_mfc: float = DEFAULT_CENSORED_MFC) -> Dict[str, Any]:
    """Summarize one planner.

    Important: mean_mfc is computed only over successful attacks. This is useful,
    but it can make a conservative planner look deceptively fragile when many
    scenes have no failure-inducing counterfactual. Therefore public_release adds
    censored_mfc: unsuccessful scenes are assigned a fixed upper-bound cost.
    A higher censored_mfc means the planner is more robust under the current
    search budget/edit space.
    """
    valid = summary.copy()
    if "error" in valid.columns:
        valid = valid[valid["error"].isna()]
    num_valid = int(len(valid))
    best_found = valid["best_found"].fillna(False).astype(bool) if "best_found" in valid else pd.Series(dtype=bool)
    best_cost = pd.to_numeric(valid.get("best_cost", pd.Series(dtype=float)), errors="coerce")
    orig_risk = pd.to_numeric(valid.get("original_risk_score", pd.Series(dtype=float)), errors="coerce")
    best_risk = pd.to_numeric(valid.get("best_risk_score", pd.Series(dtype=float)), errors="coerce")
    collisions = valid.get("best_collision", pd.Series(dtype=bool)).fillna(False).astype(bool)
    hard_brakes = valid.get("best_hard_brake", pd.Series(dtype=bool)).fillna(False).astype(bool)
    risk_inc = (best_risk - orig_risk).dropna()

    found_cost = best_cost[best_found.reindex(best_cost.index, fill_value=False)].dropna() if len(best_found) else best_cost.dropna()
    # Censored robustness metric: missing attacks are assigned an upper-bound search cost.
    if num_valid:
        censored_values = []
        for idx in valid.index:
            found = bool(best_found.loc[idx]) if idx in best_found.index else False
            cost = best_cost.loc[idx] if idx in best_cost.index else math.nan
            if found and pd.notna(cost):
                censored_values.append(float(cost))
            else:
                censored_values.append(float(censored_mfc))
        censored_mean = float(pd.Series(censored_values).mean())
        censored_median = float(pd.Series(censored_values).median())
    else:
        censored_mean = None
        censored_median = None

    return {
        "planner": planner,
        "num_valid_runs": num_valid,
        "num_best_found": int(best_found.sum()) if len(best_found) else 0,
        "failure_discovery_rate": float(best_found.mean()) if num_valid else 0.0,
        "attack_success_rate": float(best_found.mean()) if num_valid else 0.0,
        "mean_mfc_success_only": float(found_cost.mean()) if found_cost.size else None,
        "median_mfc_success_only": float(found_cost.median()) if found_cost.size else None,
        "min_mfc_success_only": float(found_cost.min()) if found_cost.size else None,
        "max_mfc_success_only": float(found_cost.max()) if found_cost.size else None,
        "censored_mfc_upper_bound": float(censored_mfc),
        "mean_censored_mfc": censored_mean,
        "median_censored_mfc": censored_median,
        # Backward-compatible names used by earlier reports.
        "mean_mfc": float(found_cost.mean()) if found_cost.size else None,
        "median_mfc": float(found_cost.median()) if found_cost.size else None,
        "min_mfc": float(found_cost.min()) if found_cost.size else None,
        "max_mfc": float(found_cost.max()) if found_cost.size else None,
        "mean_original_risk_score": float(orig_risk.dropna().mean()) if orig_risk.dropna().size else None,
        "mean_counterfactual_risk_score": float(best_risk.dropna().mean()) if best_risk.dropna().size else None,
        "mean_risk_increase": float(risk_inc.mean()) if risk_inc.size else None,
        "collision_rate": float(collisions.mean()) if num_valid else 0.0,
        "hard_brake_rate": float(hard_brakes.mean()) if num_valid else 0.0,
        "num_edit_types_found": int(valid.get("best_edit_name", pd.Series(dtype=str)).dropna().nunique()) if num_valid else 0,
    }


def build_planner_edit_table(all_rows: pd.DataFrame) -> pd.DataFrame:
    if all_rows.empty:
        return pd.DataFrame()
    df = all_rows[all_rows["best_found"] == True].copy()
    if df.empty:
        return pd.DataFrame()
    def _rate(s: pd.Series) -> float:
        return float(s.fillna(False).astype(bool).mean())
    return (
        df.groupby(["planner", "best_edit_name"], dropna=False)
        .agg(
            num_scenes=("scene_id", "count"),
            mean_mfc=("best_cost", "mean"),
            median_mfc=("best_cost", "median"),
            min_mfc=("best_cost", "min"),
            max_mfc=("best_cost", "max"),
            collision_rate=("best_collision", _rate),
            hard_brake_rate=("best_hard_brake", _rate),
            mean_best_risk=("best_risk_score", "mean"),
        )
        .reset_index()
        .sort_values(["planner", "mean_mfc", "best_edit_name"])
    )


def build_planner_ranking(planner_summary: pd.DataFrame) -> pd.DataFrame:
    if planner_summary.empty:
        return pd.DataFrame()
    df = planner_summary.copy()
    # More robust means lower attack success and higher censored MFC.
    df["robustness_rank_key"] = (1.0 - df["attack_success_rate"].fillna(0.0)) * 100.0 + df["mean_censored_mfc"].fillna(0.0)
    df = df.sort_values(["robustness_rank_key", "mean_censored_mfc"], ascending=[False, False]).reset_index(drop=True)
    df.insert(0, "robustness_rank", range(1, len(df) + 1))
    return df


def make_report(planner_summary: pd.DataFrame, edit_table: pd.DataFrame, scene_matrix: pd.DataFrame, ranking: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("# CausalSensor4D public_release Multi-Planner Robustness Report")
    lines.append("")
    lines.append("## Key correction in public_release")
    lines.append("`mean_mfc_success_only` is computed only over scenes where a failure was found. This is useful for analyzing discovered counterfactuals, but it is not sufficient for robustness ranking. If a planner has many scenes with no discovered failure, those scenes must be treated as more robust rather than ignored.")
    lines.append("")
    lines.append(f"Therefore, public_release adds `mean_censored_mfc`: unsuccessful scenes are assigned an upper-bound cost of `{DEFAULT_CENSORED_MFC:.1f}`. Higher `mean_censored_mfc` and lower `attack_success_rate` indicate stronger robustness under the current edit library and search budget.")
    lines.append("")
    lines.append("## Robustness ranking")
    if ranking.empty:
        lines.append("No planner ranking available.")
    else:
        cols = ["robustness_rank", "planner", "attack_success_rate", "mean_censored_mfc", "mean_mfc_success_only", "num_best_found", "num_valid_runs", "collision_rate", "hard_brake_rate", "num_edit_types_found"]
        lines.append(ranking[cols].to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append("## Planner-level details")
    if planner_summary.empty:
        lines.append("No planner results available.")
    else:
        cols = ["planner", "attack_success_rate", "mean_censored_mfc", "mean_mfc_success_only", "median_mfc_success_only", "min_mfc_success_only", "max_mfc_success_only", "mean_risk_increase", "collision_rate", "hard_brake_rate", "num_edit_types_found"]
        lines.append(planner_summary[cols].to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append("## Planner × edit-type breakdown")
    lines.append(edit_table.to_markdown(index=False, floatfmt=".3f") if not edit_table.empty else "No edit-type breakdown available.")
    lines.append("")
    lines.append("## Scene-level matrix")
    if not scene_matrix.empty:
        cols = ["scene_id", "planner", "best_edit_name", "best_cost", "best_collision", "best_hard_brake", "best_min_ttc"]
        lines.append(scene_matrix[cols].to_markdown(index=False, floatfmt=".3f"))
    else:
        lines.append("No scene-level matrix available.")
    lines.append("")
    lines.append("## Report usage")
    lines.append("报告中建议同时报告 attack success rate 和 censored MFC。这样可以避免只在成功攻击样本上计算 MFC 而误判 planner 鲁棒性。MFC 越低代表成功反事实越小；但如果某些场景没有找到 failure，应作为高代价/未击穿样本计入鲁棒性评价。")
    return "\n".join(lines)


def save_planner_comparison_artifacts(all_rows: pd.DataFrame, out_dir: Path, planner_summaries: List[Dict[str, Any]]) -> None:
    out_dir = Path(out_dir)
    planner_summary = pd.DataFrame(planner_summaries)
    if not planner_summary.empty:
        planner_summary = planner_summary.sort_values(["attack_success_rate", "mean_censored_mfc", "planner"], ascending=[True, False, True], na_position="last")
    edit_table = build_planner_edit_table(all_rows)
    scene_matrix = all_rows.sort_values(["scene_id", "planner"]).copy()
    ranking = build_planner_ranking(planner_summary)
    planner_summary.to_csv(out_dir / "planner_comparison_summary.csv", index=False)
    edit_table.to_csv(out_dir / "planner_edit_type_table.csv", index=False)
    scene_matrix.to_csv(out_dir / "planner_scene_matrix.csv", index=False)
    ranking.to_csv(out_dir / "planner_robustness_ranking.csv", index=False)
    (out_dir / "planner_comparison_report.md").write_text(make_report(planner_summary, edit_table, scene_matrix, ranking), encoding="utf-8")
    (out_dir / "planner_comparison_metrics.json").write_text(json.dumps({"censored_mfc_upper_bound": DEFAULT_CENSORED_MFC, "planners": planner_summaries}, indent=2, ensure_ascii=False), encoding="utf-8")
