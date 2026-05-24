from __future__ import annotations

from pathlib import Path
from typing import Optional
import ast
import matplotlib.pyplot as plt

from .schemas import DrivingScene
from .edits import LeadBrakeEdit, CutInEdit, PedestrianCrossingEdit
from .planner import SimpleFollowingPlanner


def plot_trajectories(
    original: DrivingScene,
    planner: SimpleFollowingPlanner,
    best: Optional[dict],
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    original_planned = planner.rollout(original)
    plt.figure(figsize=(9, 5))
    plt.plot([s.x for s in original_planned.ego.states], [s.y for s in original_planned.ego.states], label="ego original/planned", linewidth=2)
    for agent_id, track in original.agents.items():
        plt.plot([s.x for s in track.states], [s.y for s in track.states], linestyle="--", label=f"agent {agent_id} original")

    if best is not None:
        edited = apply_best_edit(original, best)
        edited_planned = planner.rollout(edited)
        plt.plot([s.x for s in edited_planned.ego.states], [s.y for s in edited_planned.ego.states], label="ego counterfactual/planned", linewidth=3)
        target = edited.agents[best["target_agent_id"]]
        plt.plot([s.x for s in target.states], [s.y for s in target.states], label=f"target {target.agent_id} counterfactual", linewidth=3)

    plt.xlabel("x / longitudinal position (m)")
    plt.ylabel("y / lateral position (m)")
    plt.title("CausalSensor4D MVP trajectory comparison")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def apply_best_edit(scene: DrivingScene, best: dict) -> DrivingScene:
    params = best["parameters"]
    if isinstance(params, str):
        params = ast.literal_eval(params)
    if best["edit_name"] == "lead_brake":
        edit = LeadBrakeEdit(target_agent_id=best["target_agent_id"], **params)
    elif best["edit_name"] == "cut_in":
        edit = CutInEdit(target_agent_id=best["target_agent_id"], **params)
    elif best["edit_name"] == "pedestrian_crossing":
        edit = PedestrianCrossingEdit(target_agent_id=best["target_agent_id"], **params)
    else:
        raise ValueError(f"Unknown edit: {best['edit_name']}")
    return edit.apply(scene).edited_scene
