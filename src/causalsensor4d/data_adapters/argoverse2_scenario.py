from __future__ import annotations

"""Argoverse 2 Motion Forecasting -> CausalSensor4D generic tracks CSV converter.

This converter is intentionally lightweight and robust to minor schema differences.
Expected AV2 scenario columns usually include:
    track_id, timestep, position_x, position_y, heading, velocity_x, velocity_y,
    object_type, object_category

The generic CSV produced by this file has columns:
    scene_id, track_id, timestamp, x, y, vx, vy, yaw, length, width, agent_type, is_ego

Notes:
- AV2 motion forecasting annotations often do not provide physical dimensions.
  We therefore assign category-based default length/width. This is acceptable for
  the current trajectory-level MVP; report-grade experiments should replace these
  defaults with dataset-specific dimensions when available.
- Ego vehicle detection is configurable. By default, the converter treats a track
  as ego if track_id == ego_track_id or if object_type/object_category is "AV".
"""

from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

GENERIC_COLUMNS = [
    "scene_id",
    "track_id",
    "timestamp",
    "x",
    "y",
    "vx",
    "vy",
    "yaw",
    "length",
    "width",
    "agent_type",
    "is_ego",
]


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported AV2 scenario file extension: {suffix}. Use .parquet or .csv")


def _first_existing(df: pd.DataFrame, candidates: list[str], default: Optional[str] = None) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return default


def _default_size(agent_type: str) -> tuple[float, float]:
    t = str(agent_type).lower()
    if "ped" in t:
        return 0.8, 0.8
    if "cycl" in t or "bike" in t or "motor" in t:
        return 1.8, 0.8
    if "bus" in t:
        return 11.0, 2.6
    if "truck" in t:
        return 8.0, 2.6
    return 4.6, 1.9


def convert_av2_scenario_to_generic_csv(
    scenario_path: str | Path,
    out_csv: str | Path,
    scene_id: Optional[str] = None,
    ego_track_id: str = "AV",
    timestep_hz: float = 10.0,
    max_tracks: Optional[int] = None,
) -> Path:
    """Convert one AV2 scenario table to generic tracks CSV.

    Args:
        scenario_path: Path to AV2 scenario .parquet or exported .csv.
        out_csv: Output generic CSV path.
        scene_id: Optional scene id. Defaults to file stem.
        ego_track_id: Track id used to identify ego if an explicit AV label is not found.
        timestep_hz: Converts integer timestep to seconds if timestamp column is absent.
        max_tracks: Optional cap for debugging; keeps ego plus closest early tracks.
    """
    scenario_path = Path(scenario_path)
    out_csv = Path(out_csv)
    df = _read_table(scenario_path)
    sid = scene_id or scenario_path.stem

    col_track = _first_existing(df, ["track_id", "TRACK_ID", "track_uuid", "id"])
    col_step = _first_existing(df, ["timestep", "time_step", "step", "timestamp_ns", "timestamp"])
    col_x = _first_existing(df, ["position_x", "x", "center_x"])
    col_y = _first_existing(df, ["position_y", "y", "center_y"])
    col_vx = _first_existing(df, ["velocity_x", "vx", "v_x"])
    col_vy = _first_existing(df, ["velocity_y", "vy", "v_y"])
    col_yaw = _first_existing(df, ["heading", "yaw", "theta"], default=None)
    col_type = _first_existing(df, ["object_type", "agent_type", "category", "label"], default=None)
    col_objcat = _first_existing(df, ["object_category", "track_category"], default=None)

    required = {"track_id": col_track, "time": col_step, "x": col_x, "y": col_y}
    missing = [name for name, col in required.items() if col is None]
    if missing:
        raise ValueError(
            f"Missing required columns {missing}. Available columns: {list(df.columns)}"
        )

    # Normalize and keep useful columns only.
    rows = []
    for _, r in df.iterrows():
        track_id = str(r[col_track])
        raw_time = r[col_step]
        if "timestamp" in str(col_step).lower() and abs(float(raw_time)) > 1e6:
            timestamp = (float(raw_time) - float(df[col_step].min())) / 1e9
        else:
            timestamp = float(raw_time) / timestep_hz

        agent_type = "vehicle"
        if col_type is not None and pd.notna(r[col_type]):
            agent_type = str(r[col_type]).lower()
        if col_objcat is not None and pd.notna(r[col_objcat]):
            objcat = str(r[col_objcat]).lower()
        else:
            objcat = ""

        is_ego = False
        if track_id == str(ego_track_id):
            is_ego = True
        if str(agent_type).upper() == "AV" or objcat == "av":
            is_ego = True
            agent_type = "ego"

        length, width = _default_size(agent_type)
        rows.append(
            {
                "scene_id": sid,
                "track_id": "ego" if is_ego else track_id,
                "timestamp": timestamp,
                "x": float(r[col_x]),
                "y": float(r[col_y]),
                "vx": float(r[col_vx]) if col_vx is not None and pd.notna(r[col_vx]) else 0.0,
                "vy": float(r[col_vy]) if col_vy is not None and pd.notna(r[col_vy]) else 0.0,
                "yaw": float(r[col_yaw]) if col_yaw is not None and pd.notna(r[col_yaw]) else 0.0,
                "length": length,
                "width": width,
                "agent_type": "ego" if is_ego else agent_type,
                "is_ego": bool(is_ego),
            }
        )

    out = pd.DataFrame(rows)
    if not out["is_ego"].any():
        # Fall back to the earliest track id if AV is absent. This keeps the converter usable
        # for custom exported AV2-like files, but users should set ego_track_id explicitly.
        first_track = out["track_id"].iloc[0]
        out.loc[out["track_id"] == first_track, "is_ego"] = True
        out.loc[out["track_id"] == first_track, "track_id"] = "ego"
        out.loc[out["track_id"] == "ego", "agent_type"] = "ego"

    if max_tracks is not None and max_tracks > 0:
        # Keep ego plus a bounded number of other tracks, ranked by initial distance to ego.
        first_time = out["timestamp"].min()
        first = out[out["timestamp"] == first_time]
        ego_first = first[first["is_ego"]]
        if len(ego_first) > 0:
            ex, ey = float(ego_first.iloc[0]["x"]), float(ego_first.iloc[0]["y"])
            non_ego = first[~first["is_ego"]].copy()
            non_ego["dist"] = ((non_ego["x"] - ex) ** 2 + (non_ego["y"] - ey) ** 2) ** 0.5
            keep_ids = ["ego"] + list(non_ego.sort_values("dist")["track_id"].head(max_tracks).astype(str))
            out = out[out["track_id"].astype(str).isin(keep_ids)].copy()

    out = out[GENERIC_COLUMNS].sort_values(["timestamp", "track_id"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out_csv


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Convert Argoverse 2 scenario parquet/csv to CausalSensor4D generic CSV.")
    parser.add_argument("--scenario", type=str, required=True, help="Path to AV2 scenario .parquet or exported .csv")
    parser.add_argument("--out", type=str, required=True, help="Output generic CSV path")
    parser.add_argument("--scene-id", type=str, default=None)
    parser.add_argument("--ego-track-id", type=str, default="AV")
    parser.add_argument("--timestep-hz", type=float, default=10.0)
    parser.add_argument("--max-tracks", type=int, default=None)
    args = parser.parse_args()

    out = convert_av2_scenario_to_generic_csv(
        scenario_path=args.scenario,
        out_csv=args.out,
        scene_id=args.scene_id,
        ego_track_id=args.ego_track_id,
        timestep_hz=args.timestep_hz,
        max_tracks=args.max_tracks,
    )
    print(f"Converted AV2 scenario to generic CSV: {out}")


if __name__ == "__main__":
    main()
