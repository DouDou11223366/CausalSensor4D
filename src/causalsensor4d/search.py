from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import pandas as pd

from .schemas import DrivingScene
from .planner import SimpleFollowingPlanner
from .edits import LeadBrakeEdit, CutInEdit, PedestrianCrossingEdit, EditResult
from .risk import evaluate_scene, is_failure
from .longitudinal_geometry import relation_in_ego_frame


@dataclass
class SearchConfig:
    lead_decel_values: List[float]
    lead_start_times: List[float]
    lead_duration_values: List[float]
    cutin_start_times: List[float]
    cutin_lateral_shifts: List[float]
    cutin_duration_values: List[float]
    pedestrian_start_times: List[float]
    pedestrian_target_y_values: List[float]
    pedestrian_duration_values: List[float]
    ttc_failure_threshold: float = 1.5
    # public_release: use a larger and balanced causal candidate set so that lead-following
    # candidates are not suppressed by closer cut-in/pedestrian candidates in complex AV2 scenes.
    max_candidates: int = 5
    max_candidates_per_edit: int = 3


@dataclass
class SearchResult:
    best: Optional[Dict[str, Any]]
    table: pd.DataFrame


def default_search_config() -> SearchConfig:
    return SearchConfig(
        # public_release: heading-aware lead search makes many more true lead vehicles
        # admissible than the old global-x test.  Use a compact but still strong
        # longitudinal grid to keep the 120-scene benchmark tractable.
        lead_decel_values=[-3.0, -6.0, -8.0],
        lead_start_times=[0.0, 1.0, 2.0],
        lead_duration_values=[1.5, 3.0],
        cutin_start_times=[1.0, 1.5, 2.0],
        # Generic shifts. Dynamic target-lane shifts are added from current y.
        cutin_lateral_shifts=[-3.5, 3.5],
        cutin_duration_values=[1.0, 1.5, 2.0],
        pedestrian_start_times=[1.0, 1.5, 2.0],
        pedestrian_target_y_values=[0.0],
        pedestrian_duration_values=[1.0, 1.5, 2.0],
    )


def search_minimum_failure_cost(
    scene: DrivingScene,
    planner: SimpleFollowingPlanner,
    target_agent_id: str,
    config: Optional[SearchConfig] = None,
    allowed_edit: str = "all",
) -> SearchResult:
    """Search on one target agent.

    allowed_edit:
    - "all": evaluate both lead_brake and cut_in
    - "lead_brake": evaluate longitudinal braking only
    - "cut_in": evaluate lateral cut-in only
    - "pedestrian_crossing": evaluate pedestrian crossing only
    """
    cfg = config or default_search_config()
    rows: List[Dict[str, Any]] = []

    if allowed_edit in ("all", "lead_brake"):
        for start_time in cfg.lead_start_times:
            for decel in cfg.lead_decel_values:
                for duration in cfg.lead_duration_values:
                    edit = LeadBrakeEdit(
                        target_agent_id=target_agent_id,
                        start_time=start_time,
                        decel=decel,
                        duration=duration,
                    )
                    rows.append(_evaluate_edit(scene, planner, edit, cfg.ttc_failure_threshold))

    if allowed_edit in ("all", "cut_in"):
        shifts = _cutin_shifts_for_agent(scene, target_agent_id, cfg.cutin_lateral_shifts)
        for start_time in cfg.cutin_start_times:
            for shift in shifts:
                for duration in cfg.cutin_duration_values:
                    edit = CutInEdit(
                        target_agent_id=target_agent_id,
                        start_time=start_time,
                        lateral_shift=shift,
                        duration=duration,
                    )
                    rows.append(_evaluate_edit(scene, planner, edit, cfg.ttc_failure_threshold))

    if allowed_edit in ("all", "pedestrian_crossing"):
        target_ys = _pedestrian_target_ys_for_agent(scene, target_agent_id, cfg.pedestrian_target_y_values)
        for start_time in cfg.pedestrian_start_times:
            for target_y in target_ys:
                for duration in cfg.pedestrian_duration_values:
                    edit = PedestrianCrossingEdit(
                        target_agent_id=target_agent_id,
                        start_time=start_time,
                        target_y=target_y,
                        duration=duration,
                    )
                    rows.append(_evaluate_edit(scene, planner, edit, cfg.ttc_failure_threshold))

    table = pd.DataFrame(rows)
    return _select_best(table)


def search_minimum_failure_cost_multi_candidate(
    scene: DrivingScene,
    planner: SimpleFollowingPlanner,
    candidates: List[Dict[str, Any]],
    config: Optional[SearchConfig] = None,
) -> SearchResult:
    """Search over multiple causal candidates and edit types.

    This is the public_release key change. public_release selected only the top-ranked candidate,
    which made all cases collapse into lead braking. public_release searches over all
    credible candidates and lets Minimum Failure Cost decide the final edit.
    """
    cfg = config or default_search_config()
    all_rows: List[pd.DataFrame] = []

    credible = _balanced_credible_candidates(candidates, cfg)

    for cand in credible:
        agent_id = cand["agent_id"]
        edit_type = cand.get("recommended_edit", "all")
        if edit_type == "none":
            edit_type = "all"
        result = search_minimum_failure_cost(scene, planner, agent_id, cfg, allowed_edit=edit_type)
        if not result.table.empty:
            table = result.table.copy()
            table["candidate_relation"] = cand.get("relation")
            table["candidate_priority"] = cand.get("relation_priority")
            table["recommended_edit"] = cand.get("recommended_edit")
            all_rows.append(table)

    if not all_rows:
        return SearchResult(best=None, table=pd.DataFrame())
    merged = pd.concat(all_rows, ignore_index=True)
    return _select_best(merged)


def _balanced_credible_candidates(candidates: List[Dict[str, Any]], cfg: SearchConfig) -> List[Dict[str, Any]]:
    """Select a balanced candidate set across edit families.

    Earlier versions simply used the top-N graph ranking. In dense AV2 scenes this
    could suppress lead-following candidates when pedestrians/cut-in vehicles had
    higher local risk. public_release keeps high-ranked candidates but guarantees that each
    supported edit family can contribute several candidates.
    """
    credible = [c for c in candidates if c.get("recommended_edit") != "none"]
    if not credible and candidates:
        return candidates[:1]

    selected: List[Dict[str, Any]] = []
    seen = set()
    edit_order = ["lead_brake", "cut_in", "pedestrian_crossing"]
    for edit_name in edit_order:
        count = 0
        for cand in credible:
            if cand.get("recommended_edit") != edit_name:
                continue
            key = cand.get("agent_id")
            if key in seen:
                continue
            selected.append(cand)
            seen.add(key)
            count += 1
            if count >= cfg.max_candidates_per_edit:
                break

    for cand in credible:
        key = cand.get("agent_id")
        if key in seen:
            continue
        selected.append(cand)
        seen.add(key)
        if len(selected) >= cfg.max_candidates:
            break

    return selected[: cfg.max_candidates]


def _select_best(table: pd.DataFrame) -> SearchResult:
    best = None
    if len(table) > 0:
        failures = table[table["failure"]].copy()
        if len(failures) > 0:
            failures = failures.sort_values(["cost", "min_ttc", "risk_score"], ascending=[True, True, False])
            best = failures.iloc[0].to_dict()
    return SearchResult(best=best, table=table)


def _cutin_shifts_for_agent(scene: DrivingScene, agent_id: str, base_shifts: List[float]) -> List[float]:
    """Generate lateral shifts that move adjacent-lane agent toward ego lane."""
    if agent_id not in scene.agents:
        return base_shifts
    st = scene.agents[agent_id].state_at_index(0)
    ego_y = scene.ego.state_at_index(0).y
    # Target y near ego lane center. A vehicle at y=3.5 needs shift=-3.5.
    direct_shift = ego_y - st.y
    shifts = list(base_shifts)
    for mul in [0.75, 1.0, 1.15]:
        shifts.append(direct_shift * mul)
    # Deduplicate while keeping stable order.
    out = []
    seen = set()
    for s in shifts:
        key = round(float(s), 3)
        if abs(key) < 0.25:
            continue
        if key not in seen:
            out.append(float(key))
            seen.add(key)
    return out


def _pedestrian_target_ys_for_agent(scene: DrivingScene, agent_id: str, base_targets: List[float]) -> List[float]:
    """Generate target y values near/across ego lane for pedestrian crossing."""
    if agent_id not in scene.agents:
        return base_targets
    ego_y = scene.ego.state_at_index(0).y
    st = scene.agents[agent_id].state_at_index(0)
    # Always include ego lane center and a mild across-lane target.
    targets = [ego_y, ego_y - 1.0 if st.y > ego_y else ego_y + 1.0] + list(base_targets)
    out = []
    seen = set()
    for y in targets:
        key = round(float(y), 3)
        if key not in seen:
            out.append(float(key))
            seen.add(key)
    return out


def _evaluate_edit(
    scene: DrivingScene,
    planner: SimpleFollowingPlanner,
    edit,
    ttc_threshold: float,
) -> Dict[str, Any]:
    try:
        rel_meta = {
            "original_longitudinal": None,
            "original_lateral": None,
            "original_gap": None,
            "original_closing_speed": None,
        }
        target_id = getattr(edit, "target_agent_id", None)
        if target_id in scene.agents:
            rel = relation_in_ego_frame(scene.ego.state_at_index(0), scene.agents[target_id].state_at_index(0))
            rel_meta = {
                "original_longitudinal": rel.longitudinal,
                "original_lateral": rel.lateral,
                "original_gap": rel.gap,
                "original_closing_speed": rel.closing_speed,
            }
        edit_result: EditResult = edit.apply(scene)
        planned_scene = planner.rollout(edit_result.edited_scene)
        risk = evaluate_scene(planned_scene)
        failure = is_failure(risk, ttc_threshold=ttc_threshold)
        return {
            "edit_name": edit_result.edit_name,
            "target_agent_id": edit_result.target_agent_id,
            "parameters": edit_result.parameters,
            "cost": edit_result.cost,
            "failure": failure,
            "collision": risk.collision,
            "hard_brake": risk.hard_brake,
            "min_distance": risk.min_distance,
            "min_ttc": risk.min_ttc if risk.min_ttc is not None else 999.0,
            "risk_score": risk.risk_score,
            **rel_meta,
            "error": "",
        }
    except Exception as exc:
        return {
            "edit_name": getattr(edit, "name", "unknown"),
            "target_agent_id": getattr(edit, "target_agent_id", "unknown"),
            "parameters": {},
            "cost": 999.0,
            "failure": False,
            "collision": False,
            "hard_brake": False,
            "min_distance": 999.0,
            "min_ttc": 999.0,
            "risk_score": 0.0,
            "original_longitudinal": None,
            "original_lateral": None,
            "original_gap": None,
            "original_closing_speed": None,
            "error": str(exc),
        }
