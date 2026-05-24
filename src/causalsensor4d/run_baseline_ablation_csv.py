from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from .baseline_comparison import baseline_method_names, run_one_baseline_csv, save_baseline_artifacts
from .run_batch_csv import make_planner


def _parse_methods(s: str) -> List[str]:
    if s.strip().lower() == "all":
        return baseline_method_names()
    return [m.strip() for m in s.split(",") if m.strip()]


SCENE_CSV_REQUIRED_COLUMNS = {"timestamp", "track_id", "x", "y"}


def _is_scene_track_csv(csv_path: Path) -> Tuple[bool, str]:
    """Return whether a CSV is a per-scene generic_tracks_csv file.

    Clean-mining output folders often contain metadata/index CSV files such as
    selected_scenes.csv.  Those files are useful summaries, but they are not
    trajectory scene files and do not contain timestamp/track_id/x/y.  Earlier
    public_release runners treated the first direct *.csv file as a scene and failed on
    selected_scenes.csv.  This filter prevents metadata CSVs from entering the
    baseline loop.
    """
    try:
        header = pd.read_csv(csv_path, nrows=0)
    except Exception as exc:
        return False, f"cannot_read_header: {exc}"
    cols = set(str(c).strip() for c in header.columns)
    missing = sorted(SCENE_CSV_REQUIRED_COLUMNS - cols)
    if missing:
        return False, f"missing_required_columns={missing}"
    return True, "ok"


def _find_csv_files(csv_dir: Path) -> List[Path]:
    """Find valid per-scene CSV files robustly.

    public_release expected CSV_DIR to point exactly to the folder containing only scene
    CSV files. In practice, public_release clean-mining outputs may contain
    selected_scenes.csv in the same folder, or may store the actual scene CSVs
    one level deeper.  We therefore recursively inspect all CSVs and keep only
    those with the generic_tracks_csv schema: timestamp, track_id, x, y.
    """
    candidates = sorted(csv_dir.rglob("*.csv"))
    valid: List[Path] = []
    skipped: List[Tuple[Path, str]] = []
    for path in candidates:
        ok, reason = _is_scene_track_csv(path)
        if ok:
            valid.append(path)
        else:
            skipped.append((path, reason))

    print(f"[Input] CSV candidates found recursively: {len(candidates)}", flush=True)
    print(f"[Input] Valid scene trajectory CSVs: {len(valid)}", flush=True)
    if skipped:
        preview = "; ".join(f"{p.name} ({r})" for p, r in skipped[:5])
        more = "" if len(skipped) <= 5 else f"; ... +{len(skipped)-5} more"
        print(f"[Input] Skipped non-scene CSVs: {preview}{more}", flush=True)
    return valid


def main() -> None:
    parser = argparse.ArgumentParser(description="Run public_release baseline and ablation comparison on generic trajectory CSV scenes.")
    parser.add_argument("--csv-dir", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--planner", type=str, default="delayed", choices=["normal", "delayed", "weak_brake", "conservative", "aggressive"])
    parser.add_argument("--methods", type=str, default="all")
    parser.add_argument("--random-budget", type=int, default=36)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--ego-track-id", type=str, default="ego")
    parser.add_argument("--max-scenes", type=int, default=0, help="Optional smoke-test limit. 0 means all CSV scenes.")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = _find_csv_files(csv_dir)
    if args.max_scenes and args.max_scenes > 0:
        csv_files = csv_files[: args.max_scenes]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}. The script now also searches recursively; please check CSV_DIR.")
    print(f"[Input] Found {len(csv_files)} CSV scene files under {csv_dir}", flush=True)

    methods = _parse_methods(args.methods)
    known = set(baseline_method_names())
    for method in methods:
        if method not in known:
            raise ValueError(f"Unknown method '{method}'. Available: {sorted(known)}")

    # public_release: create a fresh lightweight planner for each scene/method run.
    # This keeps the full benchmark robust even if future planner wrappers carry
    # per-rollout caches or delayed-action state.
    all_rows: List[Dict[str, Any]] = []
    for method in methods:
        print(f"[Baseline] Running method={method} with planner={args.planner} ...", flush=True)
        for csv_path in csv_files:
            print(f"  [Scene] {csv_path.name}", flush=True)
            try:
                planner = make_planner(args.planner)
                row = run_one_baseline_csv(
                    csv_path=csv_path,
                    out_dir=out_dir / "per_method",
                    planner=planner,
                    planner_kind=args.planner,
                    method=method,
                    ego_track_id=args.ego_track_id,
                    random_budget=args.random_budget,
                    seed=args.seed,
                )
            except Exception as exc:
                row = {"scene_id": csv_path.stem, "input_csv": str(csv_path), "planner": args.planner, "method": method, "error": str(exc)}
                print(f"  [ERROR] {csv_path.name}: {exc}", flush=True)
            all_rows.append(row)

    all_df = pd.DataFrame(all_rows)
    save_baseline_artifacts(all_df, out_dir, methods)

    print("CausalSensor4D public_release baseline/ablation run finished.")
    print(f"CSV folder: {csv_dir}")
    print(f"Planner: {args.planner}")
    print(f"Methods: {', '.join(methods)}")
    print(f"All results: {out_dir / 'all_baseline_scene_results.csv'}")
    print(f"Method summary: {out_dir / 'baseline_method_summary.csv'}")
    print(f"Report: {out_dir / 'baseline_comparison_report.md'}")
    if os.environ.get("CS4D_FORCE_EXIT_AFTER_BASELINE") == "1":
        # public_release subprocess-isolated benchmark mode. Some heavy pandas/numpy
        # workloads can keep finalization slow on Windows/conda. The parent
        # process has already received all artifacts, so force a clean exit.
        import sys as _sys
        _sys.stdout.flush()
        _sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
