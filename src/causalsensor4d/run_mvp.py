from __future__ import annotations

import argparse
from pathlib import Path

from .schemas import load_scene_json, save_json
from .planner import SimpleFollowingPlanner, PlannerConfig
from .risk import evaluate_scene
from .causal_graph import build_causal_scene_graph, graph_to_dict, rank_causal_candidates
from .search import search_minimum_failure_cost
from .diagnosis import make_diagnosis_report, save_report
from .visualize import plot_trajectories


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CausalSensor4D MVP.")
    parser.add_argument("--scene", type=str, required=True, help="Path to scene json.")
    parser.add_argument("--out", type=str, required=True, help="Output directory.")
    parser.add_argument("--planner", type=str, default="delayed", choices=["normal", "delayed"], help="Planner type.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = load_scene_json(args.scene)
    if args.planner == "delayed":
        planner = SimpleFollowingPlanner(
            PlannerConfig(desired_speed=8.0, reaction_delay_steps=2, safe_ttc=3.0, safe_gap=12.0)
        )
    else:
        planner = SimpleFollowingPlanner(
            PlannerConfig(desired_speed=8.0, reaction_delay_steps=0, safe_ttc=3.0, safe_gap=12.0)
        )

    planned_original = planner.rollout(scene)
    original_risk = evaluate_scene(planned_original)

    graph = build_causal_scene_graph(scene, t_idx=0)
    candidates = rank_causal_candidates(graph)
    if not candidates:
        raise RuntimeError("No candidate agents found.")
    target_agent_id = candidates[0]["agent_id"]

    result = search_minimum_failure_cost(scene, planner, target_agent_id=target_agent_id)

    save_json(
        {
            "scene_id": scene.scene_id,
            "planner": args.planner,
            "original_risk": original_risk.__dict__,
            "causal_candidates": candidates,
        },
        out_dir / "original_report.json",
    )
    save_json(graph_to_dict(graph), out_dir / "causal_scene_graph.json")

    if result.best is not None:
        # pandas/numpy 类型转 JSON-friendly
        best_json = {k: (v.item() if hasattr(v, "item") else v) for k, v in result.best.items()}
    else:
        best_json = None
    save_json(best_json, out_dir / "best_counterfactual.json")

    result.table.to_csv(out_dir / "candidate_table.csv", index=False)
    report = make_diagnosis_report(scene.scene_id, original_risk, result.best)
    save_report(report, out_dir / "diagnosis_report.md")
    plot_trajectories(scene, planner, result.best, out_dir / "trajectory_plot.png")

    print("CausalSensor4D MVP finished.")
    print(f"Scene: {scene.scene_id}")
    print(f"Selected target agent: {target_agent_id}")
    print(f"Original collision: {original_risk.collision}, min_ttc: {original_risk.min_ttc}")
    if result.best is None:
        print("No failure counterfactual found.")
    else:
        print("Best counterfactual:")
        print(result.best)
    print(f"Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
