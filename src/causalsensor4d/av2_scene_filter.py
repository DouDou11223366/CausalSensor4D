from __future__ import annotations

"""AV2 real-scene filtering utilities for CausalSensor4D.

public_release goal:
- convert a large public dataset into a smaller, useful subset for counterfactual diagnosis;
- avoid blindly running MFC search on thousands of scenes that contain no lead-following/cut-in/pedestrian conflict;
- output a selected scene folder that can be fed directly into run_batch_csv.

This module operates on the CausalSensor4D generic trajectory CSV format:
    scene_id,track_id,timestamp,x,y,vx,vy,yaw,length,width,agent_type,is_ego
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import json
import math
import shutil

import pandas as pd

REQUIRED_COLUMNS = {"scene_id", "track_id", "timestamp", "x", "y", "vx", "vy", "agent_type", "is_ego"}


@dataclass
class FilterConfig:
    # Initial-state geometric thresholds in meters.
    lead_max_dx: float = 80.0
    lead_lane_width: float = 2.2
    cutin_min_dx: float = -5.0
    cutin_max_dx: float = 60.0
    cutin_min_lateral: float = 2.0
    cutin_max_lateral: float = 6.5
    pedestrian_max_dx: float = 80.0
    pedestrian_min_lateral: float = 1.0
    pedestrian_max_lateral: float = 12.0
    # Future proximity thresholds.
    future_close_distance: float = 12.0
    future_conflict_distance: float = 8.0
    # Selection strategy.
    min_score: float = 1.0
    max_per_type: int = 50
    top_k_total: Optional[int] = None


@dataclass
class SceneFilterResult:
    scene_id: str
    csv_path: str
    num_tracks: int
    num_timesteps: int
    has_ego: bool
    label: str
    labels: str
    score: float
    best_agent_id: Optional[str]
    best_agent_type: Optional[str]
    best_initial_dx: Optional[float]
    best_initial_dy: Optional[float]
    best_min_future_dist: Optional[float]
    reason: str


def is_generic_scene_csv(path: Path) -> bool:
    try:
        head = pd.read_csv(path, nrows=5)
    except Exception:
        return False
    return REQUIRED_COLUMNS.issubset(set(head.columns))


def find_generic_csvs(csv_dir: str | Path) -> List[Path]:
    root = Path(csv_dir)
    files = sorted(root.glob("*.csv"))
    return [p for p in files if is_generic_scene_csv(p)]


def _bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def _norm_type(x: object) -> str:
    s = str(x).strip().lower()
    if s in {"ego", "av"}:
        return "ego"
    if "ped" in s or "person" in s or "walker" in s:
        return "pedestrian"
    if "cycl" in s or "bike" in s or "motor" in s:
        return "cyclist"
    if "bus" in s:
        return "bus"
    if "truck" in s or "trailer" in s:
        return "truck"
    return "vehicle"


def _track_at_or_after(df: pd.DataFrame, track_id: str, t0: float) -> Optional[pd.Series]:
    sub = df[df["track_id"].astype(str) == str(track_id)].sort_values("timestamp")
    if sub.empty:
        return None
    sub = sub[sub["timestamp"] >= t0]
    if sub.empty:
        return None
    return sub.iloc[0]


def _min_future_distance(scene_df: pd.DataFrame, agent_id: str, ego_id: str) -> float:
    ego = scene_df[scene_df["track_id"].astype(str) == str(ego_id)][["timestamp", "x", "y"]]
    ag = scene_df[scene_df["track_id"].astype(str) == str(agent_id)][["timestamp", "x", "y"]]
    if ego.empty or ag.empty:
        return 999.0
    merged = ego.merge(ag, on="timestamp", suffixes=("_ego", "_agent"))
    if merged.empty:
        return 999.0
    d = ((merged["x_ego"] - merged["x_agent"]) ** 2 + (merged["y_ego"] - merged["y_agent"]) ** 2) ** 0.5
    return float(d.min()) if len(d) else 999.0


def _heading_difference(a: float, b: float) -> float:
    # Return angle difference in radians, folded to [0, pi].
    d = abs(float(a) - float(b))
    while d > math.pi:
        d = abs(d - 2 * math.pi)
    return d


def analyze_scene_csv(csv_path: str | Path, cfg: Optional[FilterConfig] = None) -> SceneFilterResult:
    cfg = cfg or FilterConfig()
    csv_path = Path(csv_path)
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        return SceneFilterResult("unknown", str(csv_path), 0, 0, False, "invalid", "", 0.0, None, None, None, None, None, f"read_error: {exc}")
    if not REQUIRED_COLUMNS.issubset(set(df.columns)):
        return SceneFilterResult(csv_path.stem, str(csv_path), 0, 0, False, "invalid", "", 0.0, None, None, None, None, None, "missing_required_columns")

    df = df.copy()
    df["track_id"] = df["track_id"].astype(str)
    df["agent_type_norm"] = df["agent_type"].apply(_norm_type)
    is_ego = _bool_series(df["is_ego"])
    if not is_ego.any():
        return SceneFilterResult(csv_path.stem, str(csv_path), int(df["track_id"].nunique()), int(df["timestamp"].nunique()), False, "no_ego", "", 0.0, None, None, None, None, None, "no ego track")

    ego_id = str(df.loc[is_ego, "track_id"].iloc[0])
    scene_id = str(df["scene_id"].iloc[0]) if "scene_id" in df.columns and len(df) else csv_path.stem
    t0 = float(df["timestamp"].min())
    ego0 = _track_at_or_after(df, ego_id, t0)
    if ego0 is None:
        return SceneFilterResult(scene_id, str(csv_path), int(df["track_id"].nunique()), int(df["timestamp"].nunique()), True, "invalid", "", 0.0, None, None, None, None, None, "ego has no first state")

    candidates: List[Dict[str, object]] = []
    for tid in sorted(df["track_id"].unique()):
        tid = str(tid)
        if tid == ego_id:
            continue
        row0 = _track_at_or_after(df, tid, t0)
        if row0 is None:
            continue
        atype = _norm_type(row0.get("agent_type", "vehicle"))
        dx = float(row0["x"] - ego0["x"])
        dy = float(row0["y"] - ego0["y"])
        min_future_dist = _min_future_distance(df, tid, ego_id)
        score = 0.0
        labels: List[str] = []
        reasons: List[str] = []

        if atype in {"vehicle", "bus", "truck", "cyclist"}:
            if 0.0 < dx <= cfg.lead_max_dx and abs(dy) <= cfg.lead_lane_width:
                labels.append("lead_following")
                score += 4.0 + max(0.0, (cfg.lead_max_dx - dx) / cfg.lead_max_dx) + max(0.0, (cfg.future_close_distance - min_future_dist) / cfg.future_close_distance)
                reasons.append(f"same-lane lead dx={dx:.1f}m")
            if cfg.cutin_min_dx <= dx <= cfg.cutin_max_dx and cfg.cutin_min_lateral <= abs(dy) <= cfg.cutin_max_lateral:
                labels.append("cut_in")
                score += 3.5 + max(0.0, (cfg.cutin_max_dx - max(dx, 0.0)) / cfg.cutin_max_dx) + max(0.0, (cfg.future_close_distance - min_future_dist) / cfg.future_close_distance)
                reasons.append(f"adjacent-lane vehicle dx={dx:.1f}m dy={dy:.1f}m")
            # Rough intersection/conflict heuristic: close future encounter and different heading/lane.
            yaw_diff = _heading_difference(float(ego0.get("yaw", 0.0)), float(row0.get("yaw", 0.0))) if "yaw" in df.columns else 0.0
            if min_future_dist <= cfg.future_conflict_distance and (abs(dy) > cfg.lead_lane_width or yaw_diff > 0.4):
                labels.append("intersection_conflict")
                score += 3.0 + max(0.0, (cfg.future_conflict_distance - min_future_dist) / cfg.future_conflict_distance)
                reasons.append(f"future conflict min_dist={min_future_dist:.1f}m")
        elif atype == "pedestrian":
            if 0.0 <= dx <= cfg.pedestrian_max_dx and cfg.pedestrian_min_lateral <= abs(dy) <= cfg.pedestrian_max_lateral:
                labels.append("pedestrian_crossing")
                score += 4.0 + max(0.0, (cfg.pedestrian_max_dx - dx) / cfg.pedestrian_max_dx) + max(0.0, (cfg.future_close_distance - min_future_dist) / cfg.future_close_distance)
                reasons.append(f"pedestrian near ego path dx={dx:.1f}m dy={dy:.1f}m")
            elif min_future_dist <= cfg.future_close_distance:
                labels.append("pedestrian_crossing")
                score += 3.0 + max(0.0, (cfg.future_close_distance - min_future_dist) / cfg.future_close_distance)
                reasons.append(f"pedestrian future proximity min_dist={min_future_dist:.1f}m")

        if labels:
            candidates.append({
                "track_id": tid,
                "agent_type": atype,
                "dx": dx,
                "dy": dy,
                "min_future_dist": min_future_dist,
                "score": score,
                "labels": sorted(set(labels)),
                "reason": "; ".join(reasons),
            })

    if not candidates:
        return SceneFilterResult(scene_id, str(csv_path), int(df["track_id"].nunique()), int(df["timestamp"].nunique()), True, "unselected", "", 0.0, None, None, None, None, None, "no supported interaction found")

    # Select strongest agent-level candidate.
    candidates.sort(key=lambda r: float(r["score"]), reverse=True)
    best = candidates[0]
    label_priority = ["pedestrian_crossing", "cut_in", "lead_following", "intersection_conflict"]
    labels_set = set(best["labels"])
    main_label = next((x for x in label_priority if x in labels_set), str(best["labels"])[0])

    return SceneFilterResult(
        scene_id=scene_id,
        csv_path=str(csv_path),
        num_tracks=int(df["track_id"].nunique()),
        num_timesteps=int(df["timestamp"].nunique()),
        has_ego=True,
        label=main_label,
        labels=";".join(best["labels"]),
        score=float(best["score"]),
        best_agent_id=str(best["track_id"]),
        best_agent_type=str(best["agent_type"]),
        best_initial_dx=float(best["dx"]),
        best_initial_dy=float(best["dy"]),
        best_min_future_dist=float(best["min_future_dist"]),
        reason=str(best["reason"]),
    )


def filter_scene_csvs(
    csv_dir: str | Path,
    out_dir: str | Path,
    cfg: Optional[FilterConfig] = None,
    copy_selected: bool = True,
) -> Dict[str, object]:
    cfg = cfg or FilterConfig()
    csv_dir = Path(csv_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_root = out_dir / "selected_csv"
    if copy_selected:
        selected_root.mkdir(parents=True, exist_ok=True)

    results: List[SceneFilterResult] = []
    files = find_generic_csvs(csv_dir)
    for p in files:
        results.append(analyze_scene_csv(p, cfg))
    df = pd.DataFrame([asdict(r) for r in results])
    if df.empty:
        df.to_csv(out_dir / "scene_filter_table.csv", index=False)
        summary = {"num_csv_files": 0, "num_selected": 0, "selected_csv_dir": str(selected_root)}
        (out_dir / "scene_filter_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    selected = df[(df["score"] >= cfg.min_score) & (~df["label"].isin(["invalid", "no_ego", "unselected"]))].copy()
    # Keep balanced top scenes per type.
    balanced_parts = []
    for label, group in selected.groupby("label"):
        balanced_parts.append(group.sort_values("score", ascending=False).head(cfg.max_per_type))
    selected_balanced = pd.concat(balanced_parts, ignore_index=True) if balanced_parts else selected.iloc[0:0].copy()
    selected_balanced = selected_balanced.sort_values("score", ascending=False)
    if cfg.top_k_total is not None:
        selected_balanced = selected_balanced.head(cfg.top_k_total)

    df.to_csv(out_dir / "scene_filter_table.csv", index=False)
    selected_balanced.to_csv(out_dir / "selected_scenes.csv", index=False)

    if copy_selected:
        # Reset destination to prevent stale files from previous runs.
        if selected_root.exists():
            for old in selected_root.glob("*.csv"):
                old.unlink()
        for _, row in selected_balanced.iterrows():
            src = Path(str(row["csv_path"]))
            if src.exists():
                shutil.copy2(src, selected_root / src.name)

    counts_by_label = selected_balanced["label"].value_counts().to_dict()
    all_counts_by_label = df["label"].value_counts().to_dict()
    summary = {
        "num_csv_files": int(len(df)),
        "num_selected": int(len(selected_balanced)),
        "selected_rate": float(len(selected_balanced) / max(len(df), 1)),
        "selected_counts_by_label": {str(k): int(v) for k, v in counts_by_label.items()},
        "all_counts_by_label": {str(k): int(v) for k, v in all_counts_by_label.items()},
        "selected_csv_dir": str(selected_root),
        "filter_config": asdict(cfg),
    }
    (out_dir / "scene_filter_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = _make_filter_report(summary, selected_balanced)
    (out_dir / "scene_filter_report.md").write_text(report, encoding="utf-8")
    return summary


def _make_filter_report(summary: Dict[str, object], selected: pd.DataFrame) -> str:
    lines = []
    lines.append("# CausalSensor4D public_release AV2 Scene Filter Report\n")
    lines.append("## Summary")
    lines.append(f"- CSV scenes scanned: `{summary.get('num_csv_files')}`")
    lines.append(f"- Selected scenes: `{summary.get('num_selected')}`")
    lines.append(f"- Selection rate: `{float(summary.get('selected_rate', 0.0)):.3f}`")
    lines.append(f"- Selected counts by label: `{summary.get('selected_counts_by_label')}`")
    lines.append("")
    lines.append("## Why this filter is needed")
    lines.append("Public AV2 validation contains many scenarios that are not useful for the current edit library. This filter keeps scenes with lead-following, cut-in, pedestrian-crossing, or intersection-conflict evidence, so MFC search is run on meaningful candidates instead of arbitrary logs.")
    lines.append("")
    if not selected.empty:
        show_cols = ["scene_id", "label", "score", "best_agent_id", "best_agent_type", "best_initial_dx", "best_initial_dy", "best_min_future_dist", "reason"]
        lines.append("## Top selected scenes")
        lines.append(selected[show_cols].head(30).to_markdown(index=False))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter generic trajectory CSVs for useful AV2 counterfactual scenes.")
    parser.add_argument("--csv-dir", required=True)
    parser.add_argument("--out", default="outputs/av2_filter")
    parser.add_argument("--max-per-type", type=int, default=50)
    parser.add_argument("--top-k-total", type=int, default=None)
    parser.add_argument("--min-score", type=float, default=1.0)
    args = parser.parse_args()
    cfg = FilterConfig(max_per_type=args.max_per_type, top_k_total=args.top_k_total, min_score=args.min_score)
    summary = filter_scene_csvs(args.csv_dir, args.out, cfg, copy_selected=True)
    print("CausalSensor4D public_release AV2 scene filter finished.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
