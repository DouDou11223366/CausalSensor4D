from __future__ import annotations

"""Safety-first AV2 clean-scene mining for CausalSensor4D public_release.

Earlier pipelines used interaction filtering before original-safety filtering.
That is useful for quick experiments, but it can over-select already-dangerous
scenes. public_release adds a safety-first miner:

    generic CSVs -> original-safety filter -> interaction filter on safe scenes

The output selected_clean_csv folder is the preferred input for strict
safe-to-failure experiments and LLM proposal verification.
"""

from pathlib import Path
from typing import Dict, Optional
from dataclasses import asdict
import argparse
import json

from .av2_safety_filter import SafetyFilterConfig, filter_safe_original_scenes
from .av2_scene_filter import FilterConfig, filter_scene_csvs


def mine_clean_interaction_scenes(
    csv_dir: str | Path,
    out_dir: str | Path,
    planner_kind: str = "delayed",
    safety_cfg: Optional[SafetyFilterConfig] = None,
    interaction_cfg: Optional[FilterConfig] = None,
) -> Dict[str, object]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safety_cfg = safety_cfg or SafetyFilterConfig()
    interaction_cfg = interaction_cfg or FilterConfig(max_per_type=50, top_k_total=None, min_score=1.0)

    safety_out = out_dir / "original_safety_first"
    safety_summary = filter_safe_original_scenes(
        csv_dir=csv_dir,
        out_dir=safety_out,
        planner_kind=planner_kind,
        cfg=safety_cfg,
        copy_safe=True,
    )
    safe_csv_dir = Path(str(safety_summary.get("safe_csv_dir", safety_out / "safe_csv")))

    interaction_out = out_dir / "interaction_on_safe_scenes"
    interaction_summary = filter_scene_csvs(
        csv_dir=safe_csv_dir,
        out_dir=interaction_out,
        cfg=interaction_cfg,
        copy_selected=True,
    )

    summary = {
        "version": "public_release",
        "input_csv_dir": str(csv_dir),
        "planner": planner_kind,
        "safety_first_summary": safety_summary,
        "interaction_on_safe_summary": interaction_summary,
        "safe_csv_dir": str(safe_csv_dir),
        "selected_clean_csv_dir": str(interaction_out / "selected_csv"),
        "safety_config": asdict(safety_cfg),
        "interaction_config": asdict(interaction_cfg),
    }
    (out_dir / "clean_scene_miner_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "clean_scene_miner_report.md").write_text(_make_report(summary), encoding="utf-8")
    return summary


def _make_report(summary: Dict[str, object]) -> str:
    ss = summary.get("safety_first_summary", {}) or {}
    is_ = summary.get("interaction_on_safe_summary", {}) or {}
    lines = [
        "# CausalSensor4D public_release Clean Scene Miner Report",
        "",
        "## Purpose",
        "This miner reverses the earlier AV2 filtering order. It first keeps original-safe scenes and only then searches for interaction patterns. This produces a cleaner input subset for strict safe-to-failure counterfactual experiments.",
        "",
        "## Safety-first stage",
        f"- Input scenes: `{ss.get('num_input_csv')}`",
        f"- Original-safe scenes: `{ss.get('num_original_safe')}`",
        f"- Safe rate: `{ss.get('safe_rate')}`",
        f"- Safe CSV dir: `{summary.get('safe_csv_dir')}`",
        "",
        "## Interaction-on-safe stage",
        f"- Safe scenes scanned: `{is_.get('num_csv_files')}`",
        f"- Selected clean interaction scenes: `{is_.get('num_selected')}`",
        f"- Selected counts by label: `{is_.get('selected_counts_by_label')}`",
        f"- Selected clean CSV dir: `{summary.get('selected_clean_csv_dir')}`",
        "",
        "## Recommended next use",
        "Use selected_clean_csv_dir as the CSV_DIR for public_release clean LLM pipeline or for clean safe-to-failure MFC experiments.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine clean safe interaction scenes from generic CSVs")
    parser.add_argument("--csv-dir", required=True, help="Folder of generic_tracks_csv files")
    parser.add_argument("--out", default="outputs/av2_clean_scene_miner")
    parser.add_argument("--planner", default="delayed")
    parser.add_argument("--min-ttc-safe", type=float, default=2.0)
    parser.add_argument("--allow-original-collision", action="store_true")
    parser.add_argument("--allow-original-hard-brake", action="store_true")
    parser.add_argument("--max-per-type", type=int, default=50)
    parser.add_argument("--top-k-total", type=int, default=None)
    parser.add_argument("--min-score", type=float, default=1.0)
    args = parser.parse_args()
    safety_cfg = SafetyFilterConfig(
        min_ttc_safe=args.min_ttc_safe,
        require_no_collision=not args.allow_original_collision,
        require_no_hard_brake=not args.allow_original_hard_brake,
    )
    interaction_cfg = FilterConfig(max_per_type=args.max_per_type, top_k_total=args.top_k_total, min_score=args.min_score)
    summary = mine_clean_interaction_scenes(args.csv_dir, args.out, args.planner, safety_cfg, interaction_cfg)
    print("CausalSensor4D public_release clean-scene miner finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
