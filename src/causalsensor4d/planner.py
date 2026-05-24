from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math

from .schemas import AgentState, AgentTrack, DrivingScene
from .longitudinal_geometry import relation_in_ego_frame


@dataclass
class PlannerConfig:
    desired_speed: float = 10.0
    safe_ttc: float = 3.0
    safe_gap: float = 12.0
    max_decel: float = -4.0
    comfortable_decel: float = -2.0
    max_accel: float = 1.5
    reaction_delay_steps: int = 0


class SimpleFollowingPlanner:
    """MVP planner.

    用于先打通反事实诊断流程：
    - ego 沿 x 轴行驶
    - 如果前方同车道车辆过近，则减速
    - 支持 reaction_delay_steps 模拟反应延迟

    后续论文版需要替换成 nuPlan planner / learning-based planner / E2E driving model。
    """

    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig()

    def rollout(self, scene: DrivingScene) -> DrivingScene:
        out = scene.clone()
        dt = out.dt
        original_ego = out.ego.clone()

        # 初始状态来自原始 ego。
        out.ego.states[0] = AgentState(**original_ego.states[0].__dict__)
        pending_accels = []

        for t in range(1, out.num_steps()):
            prev = out.ego.states[t - 1]
            raw_accel = self._decide_accel(prev, out, t - 1)
            pending_accels.append(raw_accel)

            if len(pending_accels) <= self.config.reaction_delay_steps:
                accel = 0.0
            else:
                accel = pending_accels[-1 - self.config.reaction_delay_steps]

            new_vx = max(0.0, prev.vx + accel * dt)
            new_x = prev.x + new_vx * dt
            out.ego.states[t] = AgentState(
                t=out.ego.states[t].t,
                x=new_x,
                y=prev.y,
                vx=new_vx,
                vy=0.0,
                yaw=0.0,
                length=prev.length,
                width=prev.width,
            )
        return out

    def _decide_accel(self, ego_state: AgentState, scene: DrivingScene, t_idx: int) -> float:
        lead = self._find_lead_vehicle(ego_state, scene, t_idx)
        if lead is None:
            if ego_state.vx < self.config.desired_speed:
                return self.config.max_accel
            return 0.0

        rel = relation_in_ego_frame(ego_state, lead)
        gap = rel.gap
        rel_speed = rel.closing_speed
        ttc = gap / rel_speed if rel_speed > 1e-6 else float("inf")

        if gap < self.config.safe_gap or ttc < self.config.safe_ttc:
            return self.config.max_decel
        if ego_state.vx < self.config.desired_speed:
            return self.config.max_accel
        return 0.0

    @staticmethod
    def _find_lead_vehicle(ego_state: AgentState, scene: DrivingScene, t_idx: int) -> Optional[AgentState]:
        """Find lead vehicle in the ego-heading frame rather than global x/y.

        This is the key public_release longitudinal fix for AV2-style world coordinates:
        an agent can be physically ahead even when its global x coordinate is
        smaller than the ego's x coordinate.
        """
        best = None
        best_longitudinal = float("inf")
        for track in scene.agents.values():
            st = track.state_at_index(t_idx)
            rel = relation_in_ego_frame(ego_state, st)
            if rel.same_lane and rel.longitudinal > 0.0 and rel.longitudinal < best_longitudinal:
                best = st
                best_longitudinal = rel.longitudinal
        return best
