from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math

from .schemas import AgentState, AgentTrack, DrivingScene
from .longitudinal_geometry import relation_in_ego_frame


@dataclass
class RiskSummary:
    min_distance: float
    min_ttc: Optional[float]
    collision: bool
    hard_brake: bool
    risk_score: float
    most_risky_agent: Optional[str]


def center_distance(a: AgentState, b: AgentState) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def approximate_collision(a: AgentState, b: AgentState, margin: float = 0.2) -> bool:
    # 简化版 OBB：先用轴对齐近似，MVP 用于快速打通流程。
    # 论文版后续换成 oriented bounding box overlap。
    dx = abs(a.x - b.x)
    dy = abs(a.y - b.y)
    return dx <= (a.length + b.length) / 2 + margin and dy <= (a.width + b.width) / 2 + margin


def time_to_collision_1d(ego: AgentState, agent: AgentState, lane_threshold: float = 2.0) -> Optional[float]:
    """Heading-aware same-lane longitudinal TTC.

    public_release no longer assumes that the global x-axis is the road direction.  It
    projects the agent relative state into the ego heading frame and computes
    TTC only when the agent is in the ego lane.  The ``lane_threshold`` argument
    is kept for API compatibility, but the effective threshold is the larger of
    the provided value and a width-based lane threshold.
    """
    rel = relation_in_ego_frame(ego, agent)
    effective_lane_threshold = max(float(lane_threshold), rel.lane_threshold)
    if abs(rel.lateral) >= effective_lane_threshold:
        return None
    if rel.gap <= 0:
        return 0.0
    if rel.closing_speed <= 1e-6:
        return None
    return rel.gap / rel.closing_speed


def evaluate_scene(scene: DrivingScene, hard_brake_acc_threshold: float = -3.0) -> RiskSummary:
    min_distance = float("inf")
    min_ttc = None
    collision = False
    hard_brake = False
    most_risky_agent = None

    for t_idx in range(scene.num_steps()):
        ego_state = scene.ego.state_at_index(t_idx)
        for agent_id, track in scene.agents.items():
            agent_state = track.state_at_index(t_idx)
            d = center_distance(ego_state, agent_state)
            if d < min_distance:
                min_distance = d
                most_risky_agent = agent_id
            if approximate_collision(ego_state, agent_state):
                collision = True
                most_risky_agent = agent_id
            ttc = time_to_collision_1d(ego_state, agent_state)
            if ttc is not None and (min_ttc is None or ttc < min_ttc):
                min_ttc = ttc
                most_risky_agent = agent_id

    # ego hard brake：用速度差估计纵向加速度
    for i in range(1, scene.num_steps()):
        prev = scene.ego.state_at_index(i - 1)
        cur = scene.ego.state_at_index(i)
        ax = (cur.vx - prev.vx) / scene.dt
        if ax < hard_brake_acc_threshold:
            hard_brake = True
            break

    # 风险分数：越大越危险。仅用于搜索排序，不作为最终论文指标。
    risk_score = 0.0
    if collision:
        risk_score += 100.0
    if min_ttc is not None:
        risk_score += max(0.0, 10.0 - min_ttc)
    risk_score += max(0.0, 15.0 - min_distance) * 0.2
    if hard_brake:
        risk_score += 5.0

    return RiskSummary(
        min_distance=min_distance,
        min_ttc=min_ttc,
        collision=collision,
        hard_brake=hard_brake,
        risk_score=risk_score,
        most_risky_agent=most_risky_agent,
    )


def is_failure(summary: RiskSummary, ttc_threshold: float = 1.5) -> bool:
    if summary.collision:
        return True
    if summary.min_ttc is not None and summary.min_ttc < ttc_threshold:
        return True
    return False
