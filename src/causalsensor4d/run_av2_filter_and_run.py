from __future__ import annotations

import argparse
from pathlib import Path
import json

from .data_adapters.av2_motion_forecasting import batch_convert_av2_to_generic_csv
from .av2_scene_filter import FilterConfig, filter_scene_csvs
from .run_av2_public_dataset import run_batch_csv_programmatically


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert AV2 scenarios, filter useful scenes, then run MFC search.")
    parser.add_argument("--av2-root", required=True, help="AV2 validation/train root containing scenario parquet files.")
    parser.add_argument("--out", default="outputs/av2_filter_run")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-tracks", type=int, default=32)
    parser.add_argument("--planner", default="delayed")
    parser.add_argument("--max-per-type", type=int, default=30)
    parser.add_argument("--top-k-total", type=int, default=60)
    parser.add_argument("--min-score", type=float, default=1.0)
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    csv_dir = out_root / "generic_tracks_csv"
    conv = batch_convert_av2_to_generic_csv(args.av2_root, csv_dir, limit=args.limit, max_tracks=args.max_tracks)
    (out_root / "conversion_summary.json").write_text(json.dumps(conv, indent=2), encoding="utf-8")
    print("Batch conversion summary:")
    print(json.dumps(conv, indent=2))

    filter_out = out_root / "scene_filter"
    cfg = FilterConfig(max_per_type=args.max_per_type, top_k_total=args.top_k_total, min_score=args.min_score)
    filt = filter_scene_csvs(csv_dir, filter_out, cfg=cfg, copy_selected=True)
    print("Scene filter summary:")
    print(json.dumps(filt, indent=2))

    selected_dir = Path(str(filt.get("selected_csv_dir")))
    if selected_dir.exists() and any(selected_dir.glob("*.csv")):
        run_batch_csv_programmatically(selected_dir, out_root / "mfc_run_selected", planner=args.planner)
        print(f"MFC selected-scene run output: {out_root / 'mfc_run_selected'}")
    else:
        print("No selected scenes. MFC run skipped.")


if __name__ == "__main__":
    main()
