from __future__ import annotations

"""Heading-aware longitudinal geometry utilities for trajectory-level driving scenes.

Earlier MVP versions used the global x-axis as the driving direction.  That is
reasonable for synthetic examples, but real AV2 scenes are stored in map/world
coordinates where the ego heading can point in any direction.  This module keeps
all longitudinal reasoning in the ego-centric heading frame:

- longitudinal distance: projection of agent-ego displacement on ego heading;
- lateral offset: projection on the ego-left axis;
- closing speed: ego speed minus agent speed along the ego heading.

The helpers intentionally do not require HD maps.  They use track velocity when
available and yaw as a fallback, which makes them compatible with the existing
generic_tracks_csv adapter.
"""

from dataclasses import dataclass
import math
from typing import Tuple

from .schemas import AgentState


@dataclass(frozen=True)
class LongitudinalRelation:
    longitudinal: float
    lateral: float
    gap: float
    closing_speed: float
    ego_long_speed: float
    agent_long_speed: float
    lane_threshold: float

    @property
    def is_ahead(self) -> bool:
        return self.longitudinal > 0.0

    @property
    def same_lane(self) -> bool:
        return abs(self.lateral) < self.lane_threshold

    @property
    def adjacent_lane(self) -> bool:
        return self.lane_threshold <= abs(self.lateral) <= self.lane_threshold + 3.75

    def ttc(self) -> float | None:
        if not self.same_lane:
            return None
        if self.gap <= 0.0:
            return 0.0
        if self.closing_speed <= 1e-6:
            return None
        return self.gap / self.closing_speed


def heading_unit(state: AgentState, min_speed: float = 0.25) -> Tuple[float, float]:
    """Return a robust forward unit vector for one state.

    Prefer velocity direction because AV2 yaw can be missing in some converted
    CSVs.  Use yaw only when the object is nearly stationary.
    """
    speed = math.hypot(float(state.vx), float(state.vy))
    if speed > min_speed:
        return float(state.vx) / speed, float(state.vy) / speed
    yaw = float(getattr(state, "yaw", 0.0) or 0.0)
    return math.cos(yaw), math.sin(yaw)


def left_unit_from_forward(fwd_x: float, fwd_y: float) -> Tuple[float, float]:
    return -fwd_y, fwd_x


def projected_speed(state: AgentState, fwd_x: float, fwd_y: float) -> float:
    return float(state.vx) * fwd_x + float(state.vy) * fwd_y


def lane_threshold_for(ego: AgentState, agent: AgentState, extra_margin: float = 0.75) -> float:
    """Approximate same-lane threshold from object widths.

    Keeps the old 2 m threshold as a lower bound for backward compatibility, but
    expands it slightly for larger vehicles.
    """
    return max(2.0, (float(ego.width) + float(agent.width)) * 0.5 + extra_margin)


def relation_in_ego_frame(ego: AgentState, agent: AgentState) -> LongitudinalRelation:
    fwd_x, fwd_y = heading_unit(ego)
    left_x, left_y = left_unit_from_forward(fwd_x, fwd_y)
    dx = float(agent.x) - float(ego.x)
    dy = float(agent.y) - float(ego.y)
    longitudinal = dx * fwd_x + dy * fwd_y
    lateral = dx * left_x + dy * left_y
    ego_v = projected_speed(ego, fwd_x, fwd_y)
    agent_v = projected_speed(agent, fwd_x, fwd_y)
    gap = longitudinal - (float(ego.length) + float(agent.length)) * 0.5
    return LongitudinalRelation(
        longitudinal=float(longitudinal),
        lateral=float(lateral),
        gap=float(gap),
        closing_speed=float(ego_v - agent_v),
        ego_long_speed=float(ego_v),
        agent_long_speed=float(agent_v),
        lane_threshold=lane_threshold_for(ego, agent),
    )


def is_same_lane_ahead(ego: AgentState, agent: AgentState) -> bool:
    r = relation_in_ego_frame(ego, agent)
    return r.same_lane and r.is_ahead


def is_adjacent_lane_interaction(ego: AgentState, agent: AgentState, min_longitudinal: float = -10.0, max_longitudinal: float = 55.0) -> bool:
    r = relation_in_ego_frame(ego, agent)
    return r.adjacent_lane and min_longitudinal <= r.longitudinal <= max_longitudinal
