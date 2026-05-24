from __future__ import annotations

import argparse
from pathlib import Path

from .data_adapters.generic_tracks_csv import load_tracks_csv, save_scene_from_tracks_csv
from .schemas import save_json
from .planner import SimpleFollowingPlanner, PlannerConfig
from .risk import evaluate_scene
from .causal_graph import build_causal_scene_graph, graph_to_dict, rank_causal_candidates
from .search import search_minimum_failure_cost, search_minimum_failure_cost_multi_candidate
from .diagnosis import make_diagnosis_report, save_report
from .visualize import plot_trajectories


def make_planner(kind: str) -> SimpleFollowingPlanner:
    if kind == "delayed":
        return SimpleFollowingPlanner(
            PlannerConfig(desired_speed=8.0, reaction_delay_steps=2, safe_ttc=3.0, safe_gap=12.0)
        )
    return SimpleFollowingPlanner(
        PlannerConfig(desired_speed=8.0, reaction_delay_steps=0, safe_ttc=3.0, safe_gap=12.0)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CausalSensor4D MVP from generic trajectory CSV.")
    parser.add_argument("--csv", type=str, required=True, help="Path to generic trajectory CSV.")
    parser.add_argument("--scene-id", type=str, default=None, help="Scene id to load if CSV contains multiple scenes.")
    parser.add_argument("--ego-track-id", type=str, default="ego", help="Fallback ego track id if is_ego column is absent.")
    parser.add_argument("--out", type=str, required=True, help="Output directory.")
    parser.add_argument("--planner", type=str, default="delayed", choices=["normal", "delayed"], help="Planner type.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = load_tracks_csv(args.csv, scene_id=args.scene_id, ego_track_id=args.ego_track_id)
    save_scene_from_tracks_csv(
        args.csv,
        out_dir / "converted_scene.json",
        scene_id=args.scene_id,
        ego_track_id=args.ego_track_id,
    )

    planner = make_planner(args.planner)
    planned_original = planner.rollout(scene)
    original_risk = evaluate_scene(planned_original)

    graph = build_causal_scene_graph(scene, t_idx=0)
    candidates = rank_causal_candidates(graph)
    if not candidates:
        raise RuntimeError("No candidate agents found.")
    top_candidate_agent_id = candidates[0]["agent_id"]

    result = search_minimum_failure_cost_multi_candidate(scene, planner, candidates)

    save_json(
        {
            "scene_id": scene.scene_id,
            "planner": args.planner,
            "input_csv": args.csv,
            "original_risk": original_risk.__dict__,
            "top_candidate_agent_id": top_candidate_agent_id,
            "causal_candidates": candidates,
            "scene_metadata": scene.metadata,
        },
        out_dir / "original_report.json",
    )
    save_json(graph_to_dict(graph), out_dir / "causal_scene_graph.json")

    best_json = None
    if result.best is not None:
        best_json = {k: (v.item() if hasattr(v, "item") else v) for k, v in result.best.items()}
    save_json(best_json, out_dir / "best_counterfactual.json")

    result.table.to_csv(out_dir / "candidate_table.csv", index=False)
    report = make_diagnosis_report(scene.scene_id, original_risk, result.best)
    save_report(report, out_dir / "diagnosis_report.md")
    plot_trajectories(scene, planner, result.best, out_dir / "trajectory_plot.png")

    print("CausalSensor4D CSV run finished.")
    print(f"Scene: {scene.scene_id}")
    print(f"Input CSV: {args.csv}")
    print(f"Converted scene: {out_dir / 'converted_scene.json'}")
    print(f"Top causal candidate: {top_candidate_agent_id}")
    print(f"Original collision: {original_risk.collision}, min_ttc: {original_risk.min_ttc}")
    if result.best is None:
        print("No failure counterfactual found.")
    else:
        print("Best counterfactual:")
        print(result.best)
        print(f"Best target agent: {result.best.get('target_agent_id')}")
    print(f"Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
