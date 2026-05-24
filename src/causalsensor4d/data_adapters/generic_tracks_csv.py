from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json

import numpy as np
import pandas as pd

from ..schemas import AgentState, AgentTrack, DrivingScene

REQUIRED_COLUMNS = {"track_id", "timestamp", "x", "y"}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "ego"}


def _infer_dt(timestamps: List[float]) -> float:
    if len(timestamps) < 2:
        return 0.5
    diffs = np.diff(sorted(timestamps))
    diffs = diffs[diffs > 1e-9]
    if len(diffs) == 0:
        return 0.5
    return float(np.median(diffs))


def _interp(values_t: np.ndarray, values: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    if len(values_t) == 0:
        return np.zeros_like(target_t, dtype=float)
    if len(values_t) == 1:
        return np.full_like(target_t, float(values[0]), dtype=float)
    return np.interp(target_t, values_t, values)


def _track_to_states(group: pd.DataFrame, all_timestamps: List[float], default_type: str = "vehicle") -> AgentTrack:
    group = group.sort_values("timestamp")
    target_t = np.asarray(all_timestamps, dtype=float)
    source_t = group["timestamp"].to_numpy(dtype=float)
    x = _interp(source_t, group["x"].to_numpy(dtype=float), target_t)
    y = _interp(source_t, group["y"].to_numpy(dtype=float), target_t)

    # If velocities are absent, estimate from interpolated positions.
    dt = _infer_dt(all_timestamps)
    if "vx" in group.columns and group["vx"].notna().any():
        vx = _interp(source_t, group["vx"].ffill().bfill().to_numpy(dtype=float), target_t)
    else:
        vx = np.gradient(x, dt)
    if "vy" in group.columns and group["vy"].notna().any():
        vy = _interp(source_t, group["vy"].ffill().bfill().to_numpy(dtype=float), target_t)
    else:
        vy = np.gradient(y, dt)

    if "yaw" in group.columns and group["yaw"].notna().any():
        yaw = _interp(source_t, group["yaw"].ffill().bfill().to_numpy(dtype=float), target_t)
    else:
        yaw = np.arctan2(vy, np.maximum(vx, 1e-6))

    length = float(group["length"].dropna().iloc[0]) if "length" in group.columns and group["length"].notna().any() else 4.5
    width = float(group["width"].dropna().iloc[0]) if "width" in group.columns and group["width"].notna().any() else 1.9
    agent_type = str(group["agent_type"].dropna().iloc[0]) if "agent_type" in group.columns and group["agent_type"].notna().any() else default_type
    track_id = str(group["track_id"].iloc[0])

    states = [
        AgentState(
            t=float(t),
            x=float(x_i),
            y=float(y_i),
            vx=float(vx_i),
            vy=float(vy_i),
            yaw=float(yaw_i),
            length=length,
            width=width,
        )
        for t, x_i, y_i, vx_i, vy_i, yaw_i in zip(target_t, x, y, vx, vy, yaw)
    ]
    return AgentTrack(agent_id=track_id, agent_type=agent_type, states=states)


def load_tracks_csv(
    csv_path: str | Path,
    scene_id: Optional[str] = None,
    ego_track_id: str = "ego",
    max_agents: Optional[int] = None,
) -> DrivingScene:
    """Load a generic trajectory CSV into the unified DrivingScene schema.

    Expected columns:
        scene_id, track_id, timestamp, x, y, vx, vy, yaw, length, width, agent_type, is_ego

    Only track_id/timestamp/x/y are strictly required. Velocities and yaw are inferred if absent.
    This adapter is intentionally dataset-agnostic. nuScenes/Argoverse converters should first
    export their annotations into this CSV schema, then this function can run the MVP pipeline.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    if "scene_id" in df.columns:
        available_scenes = [str(s) for s in df["scene_id"].dropna().unique().tolist()]
        if scene_id is None:
            if len(available_scenes) != 1:
                raise ValueError(f"CSV contains multiple scene_id values {available_scenes}. Pass --scene-id.")
            scene_id = available_scenes[0]
        df = df[df["scene_id"].astype(str) == str(scene_id)].copy()
    else:
        scene_id = scene_id or csv_path.stem

    if df.empty:
        raise ValueError(f"No rows found for scene_id={scene_id}")

    df["timestamp"] = df["timestamp"].astype(float)
    all_timestamps = sorted(df["timestamp"].unique().astype(float).tolist())
    dt = _infer_dt(all_timestamps)

    # Determine ego track.
    ego_candidates: List[str] = []
    if "is_ego" in df.columns:
        ego_candidates = sorted(df[df["is_ego"].apply(_as_bool)]["track_id"].astype(str).unique().tolist())
    if not ego_candidates and ego_track_id in df["track_id"].astype(str).unique().tolist():
        ego_candidates = [ego_track_id]
    if not ego_candidates:
        raise ValueError("Cannot determine ego track. Add is_ego=1 or pass a CSV with track_id='ego'.")
    ego_id = ego_candidates[0]

    groups = {str(k): g.copy() for k, g in df.groupby(df["track_id"].astype(str))}
    ego = _track_to_states(groups[ego_id], all_timestamps, default_type="ego")
    ego.agent_type = "ego"

    agents: Dict[str, AgentTrack] = {}
    candidate_ids = [tid for tid in groups.keys() if tid != ego_id]
    if max_agents is not None:
        candidate_ids = candidate_ids[:max_agents]
    for tid in candidate_ids:
        track = _track_to_states(groups[tid], all_timestamps)
        agents[tid] = track

    return DrivingScene(
        scene_id=str(scene_id),
        dt=dt,
        ego=ego,
        agents=agents,
        map_info={"source": "generic_tracks_csv", "note": "map not available in public_release adapter"},
        metadata={
            "source_csv": str(csv_path),
            "num_timestamps": len(all_timestamps),
            "num_agents": len(agents),
            "adapter_tag": "generic_tracks_csv_public",
        },
    )


def _scene_to_json_dict(scene: DrivingScene) -> Dict[str, Any]:
    return {
        "scene_id": scene.scene_id,
        "dt": scene.dt,
        "ego": {
            "agent_id": scene.ego.agent_id,
            "agent_type": scene.ego.agent_type,
            "states": [asdict(s) for s in scene.ego.states],
        },
        "agents": [
            {
                "agent_id": track.agent_id,
                "agent_type": track.agent_type,
                "states": [asdict(s) for s in track.states],
            }
            for track in scene.agents.values()
        ],
        "map_info": scene.map_info,
        "metadata": scene.metadata,
    }


def save_scene_from_tracks_csv(
    csv_path: str | Path,
    out_json: str | Path,
    scene_id: Optional[str] = None,
    ego_track_id: str = "ego",
    max_agents: Optional[int] = None,
) -> DrivingScene:
    scene = load_tracks_csv(csv_path, scene_id=scene_id, ego_track_id=ego_track_id, max_agents=max_agents)
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(_scene_to_json_dict(scene), ensure_ascii=False, indent=2), encoding="utf-8")
    return scene
