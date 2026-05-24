from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
from .run_batch_csv import run_one_csv
from .schemas import save_json
from .report_metrics import save_report_artifacts
from .planner_comparison import summarize_planner, save_planner_comparison_artifacts


def _parse_planners(s: str) -> List[str]:
    return [p.strip() for p in s.split(",") if p.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CausalSensor4D experiments across multiple planner variants.")
    parser.add_argument("--csv-dir", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--planners", type=str, default="normal,delayed,conservative")
    parser.add_argument("--ego-track-id", type=str, default="ego")
    args = parser.parse_args()
    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")
    planners = _parse_planners(args.planners)
    all_rows: List[Dict[str, Any]] = []
    planner_summaries: List[Dict[str, Any]] = []
    for planner in planners:
        print(f"[MultiPlanner] Running planner={planner} ...")
        planner_dir = out_dir / "per_planner" / planner
        rows: List[Dict[str, Any]] = []
        for csv_path in csv_files:
            print(f"  [Scene] {csv_path.name}")
            try:
                row = run_one_csv(csv_path, planner_dir, planner, ego_track_id=args.ego_track_id, save_plot=False)
                row["planner"] = planner
            except Exception as exc:
                row = {"scene_id": csv_path.stem, "input_csv": str(csv_path), "planner": planner, "error": str(exc)}
                print(f"  [ERROR] {csv_path.name}: {exc}")
            rows.append(row)
            all_rows.append(row)
        summary = pd.DataFrame(rows)
        summary.to_csv(planner_dir / "batch_summary.csv", index=False)
        valid = summary[summary["error"].isna()] if "error" in summary.columns else summary
        aggregate = {
            "planner": planner,
            "num_csv_files": len(csv_files),
            "num_valid_runs": int(len(valid)),
            "num_best_found": int(valid["best_found"].fillna(False).sum()) if "best_found" in valid else 0,
            "mean_best_cost": float(valid["best_cost"].dropna().mean()) if "best_cost" in valid and valid["best_cost"].dropna().size else None,
            "mean_original_risk_score": float(valid["original_risk_score"].dropna().mean()) if "original_risk_score" in valid else None,
        }
        aggregate = save_report_artifacts(summary, planner_dir, aggregate)
        save_json(aggregate, planner_dir / "batch_aggregate.json")
        planner_summaries.append(summarize_planner(summary, planner))
    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(out_dir / "all_planner_scene_results.csv", index=False)
    save_planner_comparison_artifacts(all_df, out_dir, planner_summaries)
    print("CausalSensor4D multi-planner run finished.")
    print(f"CSV folder: {csv_dir}")
    print(f"Planners: {', '.join(planners)}")
    print(f"All results: {out_dir / 'all_planner_scene_results.csv'}")
    print(f"Planner summary: {out_dir / 'planner_comparison_summary.csv'}")
    print(f"Planner report: {out_dir / 'planner_comparison_report.md'}")


if __name__ == "__main__":
    main()
