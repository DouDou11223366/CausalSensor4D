from __future__ import annotations

from typing import Dict, Any, List
import networkx as nx

from .schemas import DrivingScene
from .risk import time_to_collision_1d, center_distance


def build_causal_scene_graph(scene: DrivingScene, t_idx: int = 0) -> nx.DiGraph:
    """构建 MVP 版 4D 因果场景图。

    public_release 更新：
    - 继续支持 lead_following -> lead_brake；
    - 继续支持 adjacent vehicle -> cut_in；
    - 新增 pedestrian_crossing_candidate -> pedestrian_crossing；

    论文版会继续扩展为：
    visual variables -> perception -> prediction -> planning -> risk。
    当前 MVP 先构建 agent interaction / risk graph。
    """
    g = nx.DiGraph(scene_id=scene.scene_id, time_index=t_idx)
    ego_state = scene.ego.state_at_index(t_idx)
    g.add_node(
        "ego",
        node_type="ego_vehicle",
        x=ego_state.x,
        y=ego_state.y,
        speed=ego_state.speed(),
    )

    for agent_id, track in scene.agents.items():
        st = track.state_at_index(t_idx)
        g.add_node(
            agent_id,
            node_type=track.agent_type,
            x=st.x,
            y=st.y,
            speed=st.speed(),
        )
        dist = center_distance(ego_state, st)
        ttc = time_to_collision_1d(ego_state, st)
        relation = classify_relation(ego_state.x, ego_state.y, st.x, st.y, track.agent_type)
        recommended_edit = recommended_edit_for_relation(relation)

        risk_weight = 0.0
        if ttc is not None:
            risk_weight += max(0.0, 5.0 - ttc)
        risk_weight += max(0.0, 12.0 - dist) * 0.1

        relation_priority = {
            # public_release: slightly prioritize longitudinal candidates because earlier
            # clean-mined AV2 experiments under-covered lead_brake.
            "lead_following": 5,
            "pedestrian_crossing_candidate": 4,
            "beside_or_cut_in_candidate": 3,
            "rear_following": 1,
            "weakly_related": 0,
        }.get(relation, 0)

        g.add_edge(
            agent_id,
            "ego",
            relation=relation,
            distance=dist,
            ttc=ttc,
            risk_weight=risk_weight,
            relation_priority=relation_priority,
            recommended_edit=recommended_edit,
        )
    return g


def classify_relation(ex: float, ey: float, ax: float, ay: float, agent_type: str = "vehicle") -> str:
    dx = ax - ex
    dy = ay - ey
    atype = (agent_type or "").lower()

    # Pedestrian close to ego's future path, located ahead or slightly ahead, can be made to cross.
    if atype in {"pedestrian", "walker", "person"}:
        if 0.0 <= dx <= 45.0 and 1.5 <= abs(dy) <= 8.0:
            return "pedestrian_crossing_candidate"
        return "weakly_related"

    # same lane longitudinal interactions
    # public_release: use a slightly wider lane band for AV2 trajectories. The old 2.0 m
    # cutoff was too strict after interpolation / ego-frame conversion.
    if abs(dy) < 2.4 and dx > 0:
        return "lead_following"
    if abs(dy) < 2.4 and dx < 0:
        return "rear_following"

    # adjacent-lane vehicle that may cut into ego lane.
    if 0.0 <= dx <= 25.0 and 2.0 <= abs(dy) <= 5.5:
        return "beside_or_cut_in_candidate"
    return "weakly_related"


def recommended_edit_for_relation(relation: str) -> str:
    if relation == "lead_following":
        return "lead_brake"
    if relation == "beside_or_cut_in_candidate":
        return "cut_in"
    if relation == "pedestrian_crossing_candidate":
        return "pedestrian_crossing"
    return "none"


def graph_to_dict(g: nx.DiGraph) -> Dict[str, Any]:
    return {
        "graph": dict(g.graph),
        "nodes": [{"id": n, **attrs} for n, attrs in g.nodes(data=True)],
        "edges": [
            {"source": u, "target": v, **attrs}
            for u, v, attrs in g.edges(data=True)
        ],
    }


def rank_causal_candidates(g: nx.DiGraph) -> List[Dict[str, Any]]:
    rows = []
    for u, v, attrs in g.edges(data=True):
        if v != "ego":
            continue
        rows.append(
            {
                "agent_id": u,
                "relation": attrs.get("relation"),
                "distance": attrs.get("distance"),
                "ttc": attrs.get("ttc"),
                "risk_weight": attrs.get("risk_weight", 0.0),
                "relation_priority": attrs.get("relation_priority", 0),
                "recommended_edit": attrs.get("recommended_edit", "none"),
            }
        )
    rows.sort(key=lambda r: (r["relation_priority"], r["risk_weight"], -float(r["distance"] or 9999)), reverse=True)
    return rows
