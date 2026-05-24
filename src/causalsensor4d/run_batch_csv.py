from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from .data_adapters.generic_tracks_csv import load_tracks_csv, save_scene_from_tracks_csv, REQUIRED_COLUMNS
from .schemas import save_json
from .planner import SimpleFollowingPlanner, PlannerConfig
from .risk import evaluate_scene
from .causal_graph import build_causal_scene_graph, graph_to_dict, rank_causal_candidates
from .search import search_minimum_failure_cost, search_minimum_failure_cost_multi_candidate
from .diagnosis import make_diagnosis_report, save_report
from .visualize import plot_trajectories
from .report_metrics import save_report_artifacts


def make_planner(kind: str) -> SimpleFollowingPlanner:
    """MVP planner factory for robustness comparison.

    论文版可以把这里替换成 learning-based planner / nuPlan planner / E2E driving model，
    但仍然复用同一套 MFC 搜索和诊断接口。
    """
    if kind == "normal":
        return SimpleFollowingPlanner(
            PlannerConfig(desired_speed=8.0, reaction_delay_steps=0, safe_ttc=3.0, safe_gap=12.0, max_decel=-4.0)
        )
    if kind == "delayed":
        return SimpleFollowingPlanner(
            PlannerConfig(desired_speed=8.0, reaction_delay_steps=2, safe_ttc=3.0, safe_gap=12.0, max_decel=-4.0)
        )
    if kind == "weak_brake":
        return SimpleFollowingPlanner(
            PlannerConfig(desired_speed=8.0, reaction_delay_steps=1, safe_ttc=3.0, safe_gap=12.0, max_decel=-2.0)
        )
    if kind == "conservative":
        return SimpleFollowingPlanner(
            PlannerConfig(desired_speed=7.0, reaction_delay_steps=0, safe_ttc=5.0, safe_gap=18.0, max_decel=-4.5)
        )
    if kind == "aggressive":
        return SimpleFollowingPlanner(
            PlannerConfig(desired_speed=9.0, reaction_delay_steps=1, safe_ttc=2.0, safe_gap=8.0, max_decel=-3.5)
        )
    raise ValueError(f"Unknown planner kind: {kind}")


def is_generic_scene_csv(csv_path: Path) -> bool:
    """Return True only for CausalSensor4D generic trajectory scene CSVs.

    public_release hotfix: AV2 batch conversion writes auxiliary CSVs such as conversion_manifest.csv.
    Those are not scene files and must be ignored by the batch runner.
    """
    try:
        head = pd.read_csv(csv_path, nrows=5)
    except Exception:
        return False
    return REQUIRED_COLUMNS.issubset(set(head.columns))


def run_one_csv(csv_path: Path, out_dir: Path, planner_kind: str, ego_track_id: str = "ego", save_plot: bool = True) -> Dict[str, Any]:
    scene = load_tracks_csv(csv_path, scene_id=None, ego_track_id=ego_track_id)
    out_scene_dir = out_dir / scene.scene_id
    out_scene_dir.mkdir(parents=True, exist_ok=True)

    save_scene_from_tracks_csv(csv_path, out_scene_dir / "converted_scene.json", ego_track_id=ego_track_id)
    planner = make_planner(planner_kind)
    planned_original = planner.rollout(scene)
    original_risk = evaluate_scene(planned_original)

    graph = build_causal_scene_graph(scene, t_idx=0)
    candidates = rank_causal_candidates(graph)
    top_candidate_agent_id = candidates[0]["agent_id"] if candidates else None

    if top_candidate_agent_id is None:
        best_json = None
        table = pd.DataFrame()
    else:
        result = search_minimum_failure_cost_multi_candidate(scene, planner, candidates)
        table = result.table
        best_json = None
        if result.best is not None:
            best_json = {k: (v.item() if hasattr(v, "item") else v) for k, v in result.best.items()}

    save_json(
        {
            "scene_id": scene.scene_id,
            "planner": planner_kind,
            "input_csv": str(csv_path),
            "original_risk": original_risk.__dict__,
            "top_candidate_agent_id": top_candidate_agent_id,
            "causal_candidates": candidates,
            "scene_metadata": scene.metadata,
        },
        out_scene_dir / "original_report.json",
    )
    save_json(graph_to_dict(graph), out_scene_dir / "causal_scene_graph.json")
    save_json(best_json, out_scene_dir / "best_counterfactual.json")
    table.to_csv(out_scene_dir / "candidate_table.csv", index=False)
    report = make_diagnosis_report(scene.scene_id, original_risk, best_json)
    save_report(report, out_scene_dir / "diagnosis_report.md")
    if save_plot:
        plot_trajectories(scene, planner, best_json, out_scene_dir / "trajectory_plot.png")

    return {
        "scene_id": scene.scene_id,
        "input_csv": str(csv_path),
        "num_agents": len(scene.agents),
        "num_steps": scene.num_steps(),
        "top_candidate_agent_id": top_candidate_agent_id,
        "original_collision": original_risk.collision,
        "original_hard_brake": original_risk.hard_brake,
        "original_min_ttc": original_risk.min_ttc if original_risk.min_ttc is not None else 999.0,
        "original_risk_score": original_risk.risk_score,
        "best_found": best_json is not None,
        "best_edit_name": None if best_json is None else best_json.get("edit_name"),
        "best_target_agent_id": None if best_json is None else best_json.get("target_agent_id"),
        "best_cost": None if best_json is None else best_json.get("cost"),
        "best_collision": None if best_json is None else best_json.get("collision"),
        "best_hard_brake": None if best_json is None else best_json.get("hard_brake"),
        "best_min_ttc": None if best_json is None else best_json.get("min_ttc"),
        "best_risk_score": None if best_json is None else best_json.get("risk_score"),
        "output_dir": str(out_scene_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch run CausalSensor4D MVP on a folder of generic trajectory CSV files.")
    parser.add_argument("--csv-dir", type=str, required=True, help="Folder containing *.csv scene files")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument("--planner", type=str, default="delayed", choices=["normal", "delayed", "weak_brake", "conservative", "aggressive"])
    parser.add_argument("--ego-track-id", type=str, default="ego")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_csv_files = sorted(csv_dir.glob("*.csv"))
    csv_files = [p for p in all_csv_files if is_generic_scene_csv(p)]
    skipped_csv_files = [p for p in all_csv_files if p not in csv_files]
    if skipped_csv_files:
        print("[Batch] Skipping non-scene CSV files:", ", ".join(p.name for p in skipped_csv_files))
    if not csv_files:
        raise FileNotFoundError(f"No generic trajectory scene CSV files found in {csv_dir}")

    rows: List[Dict[str, Any]] = []
    for csv_path in csv_files:
        print(f"[Batch] Running {csv_path.name} ...")
        try:
            rows.append(run_one_csv(csv_path, out_dir, args.planner, ego_track_id=args.ego_track_id))
        except Exception as exc:
            rows.append({"scene_id": csv_path.stem, "input_csv": str(csv_path), "error": str(exc)})
            print(f"[Batch] ERROR for {csv_path.name}: {exc}")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "batch_summary.csv", index=False)

    valid = summary[summary.get("error", pd.Series([None] * len(summary))).isna()] if "error" in summary.columns else summary
    aggregate = {
        "num_csv_files": len(csv_files),
        "num_skipped_aux_csv_files": len(skipped_csv_files),
        "num_valid_runs": int(len(valid)),
        "num_best_found": int(valid["best_found"].fillna(False).sum()) if "best_found" in valid else 0,
        "mean_best_cost": float(valid["best_cost"].dropna().mean()) if "best_cost" in valid and valid["best_cost"].dropna().size else None,
        "mean_original_risk_score": float(valid["original_risk_score"].dropna().mean()) if "original_risk_score" in valid else None,
    }

    # public_release: generate report-oriented evidence and MFC tables.
    aggregate = save_report_artifacts(summary, out_dir, aggregate)
    save_json(aggregate, out_dir / "batch_aggregate.json")

    print("CausalSensor4D batch CSV run finished.")
    print(f"CSV folder: {csv_dir}")
    print(f"Processed scenes: {len(csv_files)}")
    print(f"Summary: {out_dir / 'batch_summary.csv'}")
    print(f"Aggregate: {out_dir / 'batch_aggregate.json'}")
    print(f"Evidence table: {out_dir / 'failure_evidence_table.csv'}")
    print(f"MFC by edit type: {out_dir / 'mfc_by_edit_type.csv'}")
    print(f"Report-ready report: {out_dir / 'report_ready_results.md'}")


if __name__ == "__main__":
    main()
