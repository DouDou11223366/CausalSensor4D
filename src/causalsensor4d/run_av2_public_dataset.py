from __future__ import annotations

import argparse
from pathlib import Path
import json
import pandas as pd

from .data_adapters.av2_motion_forecasting import (
    batch_convert_av2_to_generic_csv,
    convert_av2_scenario_to_generic_csv,
    create_mock_av2_scenario,
    summarize_av2_table,
)
from .run_batch_csv import main as _batch_main


def run_batch_csv_programmatically(csv_dir: Path, out_dir: Path, planner: str = "delayed") -> None:
    # Reuse the existing CLI main without rewriting batch logic.
    import sys
    old = sys.argv[:]
    try:
        sys.argv = ["run_batch_csv", "--csv-dir", str(csv_dir), "--out", str(out_dir), "--planner", planner]
        _batch_main()
    finally:
        sys.argv = old


def main() -> None:
    parser = argparse.ArgumentParser(description="CausalSensor4D public_release public dataset entrypoint.")
    parser.add_argument("--mode", choices=["mock", "convert-one", "convert-batch", "convert-and-run"], default="mock")
    parser.add_argument("--scenario", type=str, default=None, help="One AV2 scenario parquet/csv path")
    parser.add_argument("--av2-root", type=str, default=None, help="Root folder containing AV2 scenario parquet files")
    parser.add_argument("--out", type=str, default="outputs/av2_public")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-tracks", type=int, default=32)
    parser.add_argument("--planner", type=str, default="delayed")
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.mode == "mock":
        scenario = create_mock_av2_scenario(out_root / "mock_av2_scenario.parquet")
        csv_out = out_root / "generic_tracks_csv" / "mock_av2_scenario.csv"
        convert_av2_scenario_to_generic_csv(scenario, csv_out, scene_id="mock_av2_scenario", max_tracks=args.max_tracks)
        run_batch_csv_programmatically(csv_out.parent, out_root / "mfc_run", planner=args.planner)
        print("CausalSensor4D public_release mock public-dataset pipeline finished.")
        print(f"Mock AV2 parquet: {scenario}")
        print(f"Generic CSV: {csv_out}")
        print(f"MFC run output: {out_root / 'mfc_run'}")
        return

    if args.mode == "convert-one":
        if not args.scenario:
            raise ValueError("--scenario is required for convert-one")
        summary = summarize_av2_table(args.scenario)
        (out_root / "scenario_schema_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        csv_out = out_root / "generic_tracks_csv" / f"{Path(args.scenario).parent.name or Path(args.scenario).stem}.csv"
        convert_av2_scenario_to_generic_csv(args.scenario, csv_out, max_tracks=args.max_tracks)
        print("Converted one AV2 scenario.")
        print(f"Schema summary: {out_root / 'scenario_schema_summary.json'}")
        print(f"Generic CSV: {csv_out}")
        return

    if args.mode in {"convert-batch", "convert-and-run"}:
        if not args.av2_root:
            raise ValueError("--av2-root is required for convert-batch / convert-and-run")
        csv_dir = out_root / "generic_tracks_csv"
        summary = batch_convert_av2_to_generic_csv(args.av2_root, csv_dir, limit=args.limit, max_tracks=args.max_tracks)
        print("Batch conversion summary:")
        print(json.dumps(summary, indent=2))
        if args.mode == "convert-and-run":
            run_batch_csv_programmatically(csv_dir, out_root / "mfc_run", planner=args.planner)
            print(f"MFC run output: {out_root / 'mfc_run'}")
        return


if __name__ == "__main__":
    main()
