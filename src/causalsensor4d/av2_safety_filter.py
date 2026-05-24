from __future__ import annotations

"""Safety-quality filtering for AV2 public-dataset experiments.

public_release adds a second quality gate after the geometric interaction filter:
we keep only scenes whose original rollout is safe under the selected planner.
This is important for a clean counterfactual task: safe original scene -> failure-inducing counterfactual.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import json
import shutil

import pandas as pd

from .data_adapters.generic_tracks_csv import load_tracks_csv
from .run_batch_csv import make_planner, is_generic_scene_csv
from .risk import evaluate_scene


@dataclass
class SafetyFilterConfig:
    min_ttc_safe: float = 2.0
    require_no_collision: bool = True
    require_no_hard_brake: bool = True
    max_original_risk_score: Optional[float] = None


@dataclass
class SafetyFilterRow:
    scene_id: str
    csv_path: str
    num_agents: int
    num_steps: int
    original_collision: bool
    original_hard_brake: bool
    original_min_ttc: float
    original_min_distance: float
    original_risk_score: float
    most_risky_agent: Optional[str]
    is_original_safe: bool
    unsafe_reason: str


def _safe_reason(
    *,
    collision: bool,
    hard_brake: bool,
    min_ttc: Optional[float],
    risk_score: float,
    cfg: SafetyFilterConfig,
) -> tuple[bool, str]:
    reasons: List[str] = []
    if cfg.require_no_collision and collision:
        reasons.append("original_collision")
    if cfg.require_no_hard_brake and hard_brake:
        reasons.append("original_hard_brake")
    if min_ttc is not None and min_ttc < cfg.min_ttc_safe:
        reasons.append(f"original_low_ttc<{cfg.min_ttc_safe}")
    if cfg.max_original_risk_score is not None and risk_score > cfg.max_original_risk_score:
        reasons.append(f"original_risk>{cfg.max_original_risk_score}")
    return (len(reasons) == 0, "safe" if not reasons else ";".join(reasons))


def analyze_original_safety(csv_path: str | Path, planner_kind: str, cfg: SafetyFilterConfig) -> SafetyFilterRow:
    csv_path = Path(csv_path)
    scene = load_tracks_csv(csv_path, scene_id=None, ego_track_id="ego")
    planner = make_planner(planner_kind)
    planned = planner.rollout(scene)
    risk = evaluate_scene(planned)
    ok, reason = _safe_reason(
        collision=bool(risk.collision),
        hard_brake=bool(risk.hard_brake),
        min_ttc=risk.min_ttc,
        risk_score=float(risk.risk_score),
        cfg=cfg,
    )
    return SafetyFilterRow(
        scene_id=scene.scene_id,
        csv_path=str(csv_path),
        num_agents=len(scene.agents),
        num_steps=scene.num_steps(),
        original_collision=bool(risk.collision),
        original_hard_brake=bool(risk.hard_brake),
        original_min_ttc=float(risk.min_ttc if risk.min_ttc is not None else 999.0),
        original_min_distance=float(risk.min_distance),
        original_risk_score=float(risk.risk_score),
        most_risky_agent=risk.most_risky_agent,
        is_original_safe=ok,
        unsafe_reason=reason,
    )


def filter_safe_original_scenes(
    csv_dir: str | Path,
    out_dir: str | Path,
    planner_kind: str = "delayed",
    cfg: Optional[SafetyFilterConfig] = None,
    copy_safe: bool = True,
) -> Dict[str, object]:
    cfg = cfg or SafetyFilterConfig()
    csv_dir = Path(csv_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_root = out_dir / "safe_csv"
    if copy_safe:
        safe_root.mkdir(parents=True, exist_ok=True)
        for old in safe_root.glob("*.csv"):
            old.unlink()

    files = sorted([p for p in csv_dir.glob("*.csv") if is_generic_scene_csv(p)])
    rows: List[SafetyFilterRow] = []
    errors: List[Dict[str, str]] = []
    for p in files:
        try:
            rows.append(analyze_original_safety(p, planner_kind, cfg))
        except Exception as exc:
            errors.append({"csv_path": str(p), "error": str(exc)})

    df = pd.DataFrame([asdict(r) for r in rows])
    if df.empty:
        df.to_csv(out_dir / "original_safety_table.csv", index=False)
        summary = {
            "num_input_csv": len(files),
            "num_analyzed": 0,
            "num_original_safe": 0,
            "safe_rate": 0.0,
            "safe_csv_dir": str(safe_root),
            "planner": planner_kind,
            "safety_config": asdict(cfg),
            "num_errors": len(errors),
        }
        (out_dir / "original_safety_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    safe_df = df[df["is_original_safe"] == True].copy()
    df.to_csv(out_dir / "original_safety_table.csv", index=False)
    safe_df.to_csv(out_dir / "safe_selected_scenes.csv", index=False)
    pd.DataFrame(errors).to_csv(out_dir / "safety_filter_errors.csv", index=False)

    if copy_safe:
        for _, row in safe_df.iterrows():
            src = Path(str(row["csv_path"]))
            if src.exists():
                shutil.copy2(src, safe_root / src.name)

    unsafe_counts = df["unsafe_reason"].value_counts().to_dict()
    summary = {
        "num_input_csv": int(len(files)),
        "num_analyzed": int(len(df)),
        "num_original_safe": int(len(safe_df)),
        "safe_rate": float(len(safe_df) / max(len(df), 1)),
        "unsafe_counts": {str(k): int(v) for k, v in unsafe_counts.items()},
        "safe_csv_dir": str(safe_root),
        "planner": planner_kind,
        "safety_config": asdict(cfg),
        "num_errors": len(errors),
    }
    (out_dir / "original_safety_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "original_safety_report.md").write_text(_make_safety_report(summary, df, safe_df), encoding="utf-8")
    return summary


def _make_safety_report(summary: Dict[str, object], df: pd.DataFrame, safe_df: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("# CausalSensor4D public_release Original-Safety Filter Report\n")
    lines.append("## Summary")
    lines.append(f"- Input CSV scenes: `{summary.get('num_input_csv')}`")
    lines.append(f"- Analyzed scenes: `{summary.get('num_analyzed')}`")
    lines.append(f"- Original-safe scenes: `{summary.get('num_original_safe')}`")
    lines.append(f"- Safe rate: `{float(summary.get('safe_rate', 0.0)):.3f}`")
    lines.append(f"- Planner: `{summary.get('planner')}`")
    lines.append(f"- Safety config: `{summary.get('safety_config')}`")
    lines.append("")
    lines.append("## Why this filter is needed")
    lines.append("The counterfactual diagnosis task should start from a safe factual scene. If the original rollout already collides, hard-brakes, or has very low TTC, then a reported counterfactual is not a clean safe-to-failure transition. This filter removes such cases before MFC search.")
    lines.append("")
    lines.append("## Unsafe reason counts")
    lines.append(f"`{summary.get('unsafe_counts')}`")
    lines.append("")
    if not safe_df.empty:
        show = ["scene_id", "original_min_ttc", "original_min_distance", "original_risk_score", "most_risky_agent"]
        lines.append("## Top original-safe scenes")
        lines.append(safe_df[show].head(30).to_markdown(index=False))
        lines.append("")
    if not df.empty:
        show2 = ["scene_id", "is_original_safe", "unsafe_reason", "original_min_ttc", "original_risk_score"]
        lines.append("## Sample analyzed scenes")
        lines.append(df[show2].head(30).to_markdown(index=False))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter selected AV2 CSVs by original-scene safety.")
    parser.add_argument("--csv-dir", required=True)
    parser.add_argument("--out", default="outputs/av2_original_safety")
    parser.add_argument("--planner", default="delayed")
    parser.add_argument("--min-ttc-safe", type=float, default=2.0)
    parser.add_argument("--allow-original-hard-brake", action="store_true")
    parser.add_argument("--allow-original-collision", action="store_true")
    args = parser.parse_args()
    cfg = SafetyFilterConfig(
        min_ttc_safe=args.min_ttc_safe,
        require_no_collision=not args.allow_original_collision,
        require_no_hard_brake=not args.allow_original_hard_brake,
    )
    summary = filter_safe_original_scenes(args.csv_dir, args.out, args.planner, cfg, copy_safe=True)
    print("CausalSensor4D public_release original-safety filter finished.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
