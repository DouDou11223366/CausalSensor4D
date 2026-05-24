from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import json
from pathlib import Path


@dataclass
class AgentState:
    t: float
    x: float
    y: float
    vx: float
    vy: float
    yaw: float = 0.0
    length: float = 4.5
    width: float = 1.9

    def speed(self) -> float:
        return float((self.vx ** 2 + self.vy ** 2) ** 0.5)


@dataclass
class AgentTrack:
    agent_id: str
    agent_type: str
    states: List[AgentState]

    def state_at_index(self, idx: int) -> AgentState:
        if idx < 0:
            idx = 0
        if idx >= len(self.states):
            idx = len(self.states) - 1
        return self.states[idx]

    def clone(self) -> "AgentTrack":
        return AgentTrack(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            states=[AgentState(**s.__dict__) for s in self.states],
        )


@dataclass
class DrivingScene:
    scene_id: str
    dt: float
    ego: AgentTrack
    agents: Dict[str, AgentTrack]
    map_info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def num_steps(self) -> int:
        return len(self.ego.states)

    def clone(self) -> "DrivingScene":
        return DrivingScene(
            scene_id=self.scene_id,
            dt=self.dt,
            ego=self.ego.clone(),
            agents={k: v.clone() for k, v in self.agents.items()},
            map_info=json.loads(json.dumps(self.map_info)),
            metadata=json.loads(json.dumps(self.metadata)),
        )


def _state_from_dict(d: Dict[str, Any]) -> AgentState:
    return AgentState(
        t=float(d["t"]),
        x=float(d["x"]),
        y=float(d["y"]),
        vx=float(d.get("vx", 0.0)),
        vy=float(d.get("vy", 0.0)),
        yaw=float(d.get("yaw", 0.0)),
        length=float(d.get("length", 4.5)),
        width=float(d.get("width", 1.9)),
    )


def load_scene_json(path: str | Path) -> DrivingScene:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    ego = AgentTrack(
        agent_id=data["ego"]["agent_id"],
        agent_type=data["ego"].get("agent_type", "ego"),
        states=[_state_from_dict(s) for s in data["ego"]["states"]],
    )
    agents = {}
    for item in data.get("agents", []):
        track = AgentTrack(
            agent_id=item["agent_id"],
            agent_type=item.get("agent_type", "vehicle"),
            states=[_state_from_dict(s) for s in item["states"]],
        )
        agents[track.agent_id] = track
    return DrivingScene(
        scene_id=data.get("scene_id", path.stem),
        dt=float(data.get("dt", 0.5)),
        ego=ego,
        agents=agents,
        map_info=data.get("map_info", {}),
        metadata=data.get("metadata", {}),
    )


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
