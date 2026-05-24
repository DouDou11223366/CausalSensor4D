from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json, math
import numpy as np
import pandas as pd
from .schemas import AgentState, DrivingScene
from .data_adapters.generic_tracks_csv import load_tracks_csv, REQUIRED_COLUMNS

DEFAULT_MODEL_PATH = Path("outputs/lightweight_bc_model/lightweight_bc_model.npz")

@dataclass
class LightweightBCConfig:
    ridge_lambda: float = 1e-2
    max_accel: float = 2.0
    max_decel: float = -4.0
    desired_speed_fallback: float = 8.0
    safety_blend: float = 0.30
    safe_gap: float = 12.0
    safe_ttc: float = 3.0

@dataclass
class LightweightBCTrainingReport:
    model_path: str
    num_csv_files: int
    num_valid_csv_files: int
    num_training_samples: int
    ridge_lambda: float
    train_mse: float
    feature_names: List[str]
    config: Dict[str, Any]

FEATURE_NAMES = ["bias","ego_speed_norm","lead_present","lead_gap_norm","lead_rel_speed_norm","lead_ttc_inv","adjacent_risk","pedestrian_risk"]

def is_generic_scene_csv(csv_path: Path) -> bool:
    try:
        head = pd.read_csv(csv_path, nrows=5)
    except Exception:
        return False
    return REQUIRED_COLUMNS.issubset(set(head.columns))

def discover_csv_files(csv_dirs: Iterable[str | Path]) -> List[Path]:
    files: List[Path] = []
    for item in csv_dirs:
        p = Path(item)
        if not p.exists():
            continue
        if p.is_file() and p.suffix.lower() == ".csv":
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(q for q in p.glob("*.csv") if q.is_file()))
    return [p for p in sorted(set(files)) if is_generic_scene_csv(p)]

def extract_features(scene: DrivingScene, t_idx: int) -> np.ndarray:
    ego = scene.ego.state_at_index(t_idx)
    ego_speed = ego.speed()
    lead_gap, lead_rel_speed, lead_present, lead_ttc_inv = 80.0, 0.0, 0.0, 0.0
    adjacent_risk, pedestrian_risk = 0.0, 0.0
    for track in scene.agents.values():
        st = track.state_at_index(t_idx)
        dx, dy = st.x - ego.x, st.y - ego.y
        agent_speed = st.speed()
        if abs(dy) < 2.2 and dx > 0:
            gap = max(0.0, dx - (ego.length + st.length) / 2.0)
            if gap < lead_gap:
                lead_gap = gap
                lead_rel_speed = ego_speed - agent_speed
                lead_present = 1.0
                if lead_rel_speed > 1e-3:
                    lead_ttc_inv = min(1.0, 1.0 / max(gap / lead_rel_speed, 1e-3))
        if -8.0 <= dx <= 40.0 and 2.0 <= abs(dy) <= 7.0:
            dist = math.hypot(dx, dy)
            adjacent_risk = max(adjacent_risk, max(0.0, 1.0 - dist / 45.0))
        typ = str(track.agent_type).lower()
        if "ped" in typ and -10.0 <= dx <= 50.0 and 0.5 <= abs(dy) <= 12.0:
            dist = math.hypot(dx, dy)
            pedestrian_risk = max(pedestrian_risk, max(0.0, 1.0 - dist / 50.0))
    return np.array([1.0, ego_speed/15.0, lead_present, min(lead_gap,80.0)/80.0, np.clip(lead_rel_speed/15.0,-1,1), lead_ttc_inv, adjacent_risk, pedestrian_risk], dtype=float)

def collect_training_data(csv_files: List[Path], config: LightweightBCConfig, ego_track_id: str="ego") -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X_rows, y_rows, valid_files = [], [], []
    for csv_path in csv_files:
        try:
            scene = load_tracks_csv(csv_path, scene_id=None, ego_track_id=ego_track_id)
        except Exception:
            continue
        if scene.num_steps() < 3:
            continue
        valid_files.append(str(csv_path))
        for t_idx in range(scene.num_steps()-1):
            cur, nxt = scene.ego.state_at_index(t_idx), scene.ego.state_at_index(t_idx+1)
            accel = float(np.clip((nxt.vx-cur.vx)/max(scene.dt,1e-6), config.max_decel, config.max_accel))
            X_rows.append(extract_features(scene, t_idx)); y_rows.append(accel)
    if not X_rows:
        raise RuntimeError("No training samples found. Check CSV paths.")
    return np.vstack(X_rows), np.asarray(y_rows, dtype=float), valid_files

def train_lightweight_bc_planner(csv_dirs: Iterable[str | Path], model_path: str | Path=DEFAULT_MODEL_PATH, ego_track_id: str="ego", config: Optional[LightweightBCConfig]=None) -> LightweightBCTrainingReport:
    config = config or LightweightBCConfig()
    csv_files = discover_csv_files(csv_dirs)
    if not csv_files:
        raise FileNotFoundError(f"No generic_tracks_csv files found in: {list(csv_dirs)}")
    X, y, valid_files = collect_training_data(csv_files, config, ego_track_id)
    reg = np.eye(X.shape[1]) * config.ridge_lambda; reg[0,0] *= 0.01
    w = np.linalg.solve(X.T @ X + reg, X.T @ y)
    pred = X @ w; mse = float(np.mean((pred-y)**2))
    model_path = Path(model_path); model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(model_path, weights=w, feature_names=np.array(FEATURE_NAMES, dtype=object), config_json=json.dumps(asdict(config)), train_mse=mse, num_training_samples=len(y), num_valid_csv_files=len(valid_files))
    report = LightweightBCTrainingReport(str(model_path), len(csv_files), len(valid_files), int(len(y)), config.ridge_lambda, mse, FEATURE_NAMES, asdict(config))
    model_path.with_suffix(".training_report.json").write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    return report

class LightweightBCPlanner:
    def __init__(self, model_path: str | Path=DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Lightweight BC model not found: {self.model_path}. Run train script first or set CS4D_LIGHTWEIGHT_BC_MODEL.")
        data = np.load(self.model_path, allow_pickle=True)
        self.weights = data["weights"].astype(float)
        self.config = LightweightBCConfig(**json.loads(str(data["config_json"])))
    def rollout(self, scene: DrivingScene) -> DrivingScene:
        out = scene.clone(); dt = out.dt
        out.ego.states[0] = AgentState(**out.ego.state_at_index(0).__dict__)
        for t_idx in range(1, out.num_steps()):
            prev = out.ego.state_at_index(t_idx-1)
            learned = float(np.clip(np.dot(extract_features(out, t_idx-1), self.weights), self.config.max_decel, self.config.max_accel))
            safety = self._safety_accel(prev, out, t_idx-1)
            a = float(np.clip((1-self.config.safety_blend)*learned + self.config.safety_blend*min(learned, safety), self.config.max_decel, self.config.max_accel))
            vx = max(0.0, prev.vx + a*dt); x = prev.x + vx*dt
            out.ego.states[t_idx] = AgentState(t=out.ego.states[t_idx].t, x=x, y=prev.y, vx=vx, vy=0.0, yaw=prev.yaw, length=prev.length, width=prev.width)
        return out
    def _safety_accel(self, ego_state: AgentState, scene: DrivingScene, t_idx: int) -> float:
        best_gap, best_rel = None, 0.0
        for track in scene.agents.values():
            st = track.state_at_index(t_idx); dx, dy = st.x-ego_state.x, st.y-ego_state.y
            if abs(dy)<2.2 and dx>0:
                gap = dx - (ego_state.length+st.length)/2.0
                if best_gap is None or gap < best_gap:
                    best_gap, best_rel = gap, ego_state.vx - st.vx
        if best_gap is None:
            return self.config.max_accel if ego_state.vx < self.config.desired_speed_fallback else 0.0
        ttc = best_gap/best_rel if best_rel > 1e-3 else float('inf')
        if best_gap < self.config.safe_gap or ttc < self.config.safe_ttc:
            return self.config.max_decel
        return self.config.max_accel if ego_state.vx < self.config.desired_speed_fallback else 0.0
