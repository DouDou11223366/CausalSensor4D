from __future__ import annotations

"""public_release AV2 clean public-data pipeline.

This is not merely increasing LIMIT. It adds a scientifically important quality gate:
    interaction filter -> original-safe filter -> MFC search on clean safe-to-failure scenes.
It also supports full-validation execution by setting --limit to 0 or omitting it in the PyCharm entry.
"""

import argparse
from pathlib import Path
import json

from .data_adapters.av2_motion_forecasting import batch_convert_av2_to_generic_csv
from .av2_scene_filter import FilterConfig, filter_scene_csvs
from .av2_safety_filter import SafetyFilterConfig, filter_safe_original_scenes
from .run_av2_public_dataset import run_batch_csv_programmatically


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert AV2, filter interactions, filter original-safe scenes, then run MFC.")
    parser.add_argument("--av2-root", required=True)
    parser.add_argument("--out", default="outputs/av2_clean_run")
    parser.add_argument("--limit", type=int, default=0, help="0 means all scenario files under AV2 root.")
    parser.add_argument("--max-tracks", type=int, default=32)
    parser.add_argument("--planner", default="delayed")
    parser.add_argument("--max-per-type", type=int, default=200)
    parser.add_argument("--top-k-total", type=int, default=0, help="0 means keep all selected scenes after per-type cap.")
    parser.add_argument("--min-score", type=float, default=1.0)
    parser.add_argument("--min-ttc-safe", type=float, default=2.0)
    parser.add_argument("--skip-conversion-if-exists", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    csv_dir = out_root / "generic_tracks_csv"

    limit = None if args.limit == 0 else args.limit
    top_k_total = None if args.top_k_total == 0 else args.top_k_total

    existing_csvs = list(csv_dir.glob("*.csv")) if csv_dir.exists() else []
    if args.skip_conversion_if_exists and existing_csvs:
        conv = {
            "num_files_found": None,
            "num_converted": len(existing_csvs),
            "out_dir": str(csv_dir),
            "manifest": None,
            "note": "Skipped conversion because existing generic CSV files were found.",
        }
    else:
        conv = batch_convert_av2_to_generic_csv(args.av2_root, csv_dir, limit=limit, max_tracks=args.max_tracks)
    (out_root / "conversion_summary.json").write_text(json.dumps(conv, indent=2), encoding="utf-8")
    print("Batch conversion summary:")
    print(json.dumps(conv, indent=2))

    filter_out = out_root / "scene_filter"
    filter_cfg = FilterConfig(max_per_type=args.max_per_type, top_k_total=top_k_total, min_score=args.min_score)
    filt = filter_scene_csvs(csv_dir, filter_out, cfg=filter_cfg, copy_selected=True)
    print("Interaction scene filter summary:")
    print(json.dumps(filt, indent=2))

    selected_dir = Path(str(filt.get("selected_csv_dir")))
    safety_out = out_root / "original_safety_filter"
    safety_cfg = SafetyFilterConfig(min_ttc_safe=args.min_ttc_safe)
    safe = filter_safe_original_scenes(selected_dir, safety_out, planner_kind=args.planner, cfg=safety_cfg, copy_safe=True)
    print("Original-safety filter summary:")
    print(json.dumps(safe, indent=2))

    safe_dir = Path(str(safe.get("safe_csv_dir")))
    if safe_dir.exists() and any(safe_dir.glob("*.csv")):
        run_batch_csv_programmatically(safe_dir, out_root / "mfc_run_clean", planner=args.planner)
        print(f"MFC clean safe-to-failure run output: {out_root / 'mfc_run_clean'}")
    else:
        print("No original-safe selected scenes. MFC clean run skipped.")


if __name__ == "__main__":
    main()
