from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import math

from .schemas import DrivingScene
from .longitudinal_geometry import heading_unit, projected_speed


@dataclass
class EditResult:
    edited_scene: DrivingScene
    edit_name: str
    target_agent_id: str
    parameters: Dict[str, Any]
    cost: float


class CounterfactualEdit:
    name: str = "base_edit"

    def apply(self, scene: DrivingScene) -> EditResult:
        raise NotImplementedError


@dataclass
class LeadBrakeEdit(CounterfactualEdit):
    target_agent_id: str
    start_time: float
    decel: float
    duration: float
    name: str = "lead_brake"

    def apply(self, scene: DrivingScene) -> EditResult:
        out = scene.clone()
        if self.target_agent_id not in out.agents:
            raise ValueError(f"Agent {self.target_agent_id} not found")
        track = out.agents[self.target_agent_id]
        dt = out.dt
        start_idx = max(0, int(round(self.start_time / dt)))
        end_idx = min(len(track.states) - 1, int(round((self.start_time + self.duration) / dt)))

        # public_release: integrate braking along the agent's own heading instead of
        # assuming that longitudinal motion is always global +x.  This matters
        # for AV2/world-coordinate scenes where road direction is arbitrary.
        fwd_x, fwd_y = heading_unit(track.states[start_idx])
        for i in range(start_idx + 1, len(track.states)):
            prev = track.states[i - 1]
            cur = track.states[i]
            active = i <= end_idx
            accel = self.decel if active else 0.0
            prev_speed = max(0.0, projected_speed(prev, fwd_x, fwd_y))
            new_speed = max(0.0, prev_speed + accel * dt)
            cur.x = prev.x + fwd_x * new_speed * dt
            cur.y = prev.y + fwd_y * new_speed * dt
            cur.vx = fwd_x * new_speed
            cur.vy = fwd_y * new_speed
            cur.yaw = math.atan2(fwd_y, fwd_x)

        cost = self.compute_cost()
        return EditResult(
            edited_scene=out,
            edit_name=self.name,
            target_agent_id=self.target_agent_id,
            parameters={"start_time": self.start_time, "decel": self.decel, "duration": self.duration},
            cost=cost,
        )

    def compute_cost(self) -> float:
        # 论文版可以换成更严谨的物理/统计代价。
        # 这里越早、越急、越久，cost 越高。
        return abs(self.decel) * 0.25 + self.duration * 0.15 + max(0.0, 3.0 - self.start_time) * 0.05


@dataclass
class CutInEdit(CounterfactualEdit):
    target_agent_id: str
    start_time: float
    lateral_shift: float
    duration: float
    name: str = "cut_in"

    def apply(self, scene: DrivingScene) -> EditResult:
        out = scene.clone()
        if self.target_agent_id not in out.agents:
            raise ValueError(f"Agent {self.target_agent_id} not found")
        track = out.agents[self.target_agent_id]
        dt = out.dt
        start_idx = max(0, int(round(self.start_time / dt)))
        end_idx = min(len(track.states) - 1, int(round((self.start_time + self.duration) / dt)))
        n = max(1, end_idx - start_idx)
        base_y = track.states[start_idx].y
        for i in range(start_idx, len(track.states)):
            if i <= end_idx:
                alpha = (i - start_idx) / n
                track.states[i].y = base_y + alpha * self.lateral_shift
            else:
                track.states[i].y = base_y + self.lateral_shift
        cost = abs(self.lateral_shift) * 0.3 + self.duration * 0.1
        return EditResult(
            edited_scene=out,
            edit_name=self.name,
            target_agent_id=self.target_agent_id,
            parameters={"start_time": self.start_time, "lateral_shift": self.lateral_shift, "duration": self.duration},
            cost=cost,
        )


@dataclass
class PedestrianCrossingEdit(CounterfactualEdit):
    """public_release: pedestrian crossing counterfactual.

    The edit keeps the pedestrian's longitudinal position roughly unchanged while
    moving it laterally toward/across the ego lane. This is a simplified MVP edit
    primitive for vehicle-pedestrian interaction. In the report version, this will be
    replaced by map-aware crosswalk-constrained pedestrian behavior editing.
    """

    target_agent_id: str
    start_time: float
    target_y: float
    duration: float
    name: str = "pedestrian_crossing"

    def apply(self, scene: DrivingScene) -> EditResult:
        out = scene.clone()
        if self.target_agent_id not in out.agents:
            raise ValueError(f"Agent {self.target_agent_id} not found")
        track = out.agents[self.target_agent_id]
        dt = out.dt
        start_idx = max(0, int(round(self.start_time / dt)))
        end_idx = min(len(track.states) - 1, int(round((self.start_time + self.duration) / dt)))
        n = max(1, end_idx - start_idx)
        base_y = track.states[start_idx].y
        base_x = track.states[start_idx].x

        for i in range(start_idx, len(track.states)):
            cur = track.states[i]
            if i <= end_idx:
                alpha = (i - start_idx) / n
                new_y = base_y + alpha * (self.target_y - base_y)
            else:
                new_y = self.target_y

            prev_y = track.states[i - 1].y if i > 0 else base_y
            cur.y = new_y
            # Pedestrian mostly crosses laterally at a fixed longitudinal conflict position.
            cur.x = base_x
            cur.vx = 0.0
            cur.vy = (new_y - prev_y) / dt if i > 0 else 0.0
            cur.yaw = math.atan2(cur.vy, max(cur.vx, 1e-6))

        cost = self.compute_cost(base_y=base_y)
        return EditResult(
            edited_scene=out,
            edit_name=self.name,
            target_agent_id=self.target_agent_id,
            parameters={"start_time": self.start_time, "target_y": self.target_y, "duration": self.duration},
            cost=cost,
        )

    def compute_cost(self, base_y: float) -> float:
        lateral_change = abs(self.target_y - base_y)
        # Later/shorter/less lateral changes are cheaper; too fast crossing is penalized by duration.
        return lateral_change * 0.12 + self.duration * 0.10 + max(0.0, 2.5 - self.start_time) * 0.05
