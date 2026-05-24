from __future__ import annotations

"""Argoverse 2 Motion Forecasting public-dataset utilities.

public_release goal:
- make the public-dataset path concrete without requiring the user to have AV2 downloaded yet;
- convert one AV2 scenario parquet/csv to the CausalSensor4D generic trajectory CSV;
- convert a directory of AV2 scenarios in batch;
- provide a synthetic AV2-like parquet so the converter can be tested immediately in PyCharm.

The converter is intentionally schema-tolerant. Official AV2 Motion Forecasting parquet files usually contain
columns similar to:
    track_id, timestep, position_x, position_y, heading, velocity_x, velocity_y,
    object_type, object_category
but this file also accepts common aliases.
"""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import json
import math
import pandas as pd

GENERIC_COLUMNS = [
    "scene_id", "track_id", "timestamp", "x", "y", "vx", "vy", "yaw",
    "length", "width", "agent_type", "is_ego",
]

ALIASES: Dict[str, List[str]] = {
    "track_id": ["track_id", "TRACK_ID", "track_uuid", "id", "object_id"],
    "time": ["timestep", "time_step", "step", "frame_id", "timestamp", "timestamp_ns"],
    "x": ["position_x", "x", "center_x", "translation_x"],
    "y": ["position_y", "y", "center_y", "translation_y"],
    "vx": ["velocity_x", "vx", "v_x"],
    "vy": ["velocity_y", "vy", "v_y"],
    "yaw": ["heading", "yaw", "theta", "rotation_z"],
    "object_type": ["object_type", "agent_type", "category", "label"],
    "object_category": ["object_category", "track_category"],
    "observed": ["observed", "is_observed"],
}


def _first_existing(df: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    lower_to_real = {str(c).lower(): str(c) for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        if name.lower() in lower_to_real:
            return lower_to_real[name.lower()]
    return None


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported scenario table: {path}. Expected .parquet or .csv")


def _default_size(agent_type: str) -> Tuple[float, float]:
    t = str(agent_type).lower()
    if "ped" in t:
        return 0.8, 0.8
    if "cycl" in t or "bike" in t or "motor" in t:
        return 1.8, 0.8
    if "bus" in t:
        return 11.0, 2.6
    if "truck" in t or "trailer" in t:
        return 8.0, 2.6
    return 4.6, 1.9


def _normalize_agent_type(raw: object) -> str:
    s = str(raw).strip().lower()
    if s in {"av", "ego"}:
        return "ego"
    if "ped" in s:
        return "pedestrian"
    if "cycl" in s or "bike" in s or "motor" in s:
        return "cyclist"
    if "bus" in s:
        return "bus"
    if "truck" in s or "trailer" in s:
        return "truck"
    if s in {"vehicle", "vehicular", "car", "regular_vehicle"}:
        return "vehicle"
    return s if s and s != "nan" else "vehicle"


def _time_to_seconds(series: pd.Series, value: object, timestep_hz: float) -> float:
    fv = float(value)
    name = str(series.name).lower()
    # Nanosecond timestamps are typically very large. Anchor at the first timestamp.
    if "timestamp" in name and abs(fv) > 1e6:
        return (fv - float(series.min())) / 1e9
    return fv / timestep_hz


def summarize_av2_table(path: str | Path) -> Dict[str, object]:
    """Return lightweight schema info for debugging real AV2 files."""
    df = _read_table(path)
    cols = {k: _first_existing(df, v) for k, v in ALIASES.items()}
    return {
        "path": str(path),
        "num_rows": int(len(df)),
        "columns": list(map(str, df.columns)),
        "detected_columns": cols,
        "num_tracks": int(df[cols["track_id"]].nunique()) if cols.get("track_id") else None,
        "num_timesteps": int(df[cols["time"]].nunique()) if cols.get("time") else None,
    }


def convert_av2_scenario_to_generic_csv(
    scenario_path: str | Path,
    out_csv: str | Path,
    scene_id: Optional[str] = None,
    ego_track_id: str = "AV",
    timestep_hz: float = 10.0,
    max_tracks: Optional[int] = None,
    require_ego: bool = False,
) -> Path:
    """Convert one AV2 Motion Forecasting scenario table to generic trajectory CSV.

    Args:
        scenario_path: official AV2 .parquet scenario file, or an AV2-like csv/parquet.
        out_csv: output CausalSensor4D generic CSV.
        scene_id: optional scene id; defaults to parent directory name when available, else file stem.
        ego_track_id: fallback ego id. Official files commonly use track_id == "AV".
        timestep_hz: used when time is represented as discrete timesteps.
        max_tracks: optional cap for rapid debugging. Keeps ego + closest tracks at first timestamp.
        require_ego: if True, raise an error when no AV/ego track is found.
    """
    scenario_path = Path(scenario_path)
    out_csv = Path(out_csv)
    df = _read_table(scenario_path)

    cols = {k: _first_existing(df, v) for k, v in ALIASES.items()}
    missing = [k for k in ["track_id", "time", "x", "y"] if cols[k] is None]
    if missing:
        raise ValueError(
            f"Missing required AV2 columns {missing}. Available columns: {list(df.columns)}. "
            f"Detected aliases: {cols}"
        )

    sid = scene_id or (scenario_path.parent.name if scenario_path.parent.name else scenario_path.stem)
    rows = []
    time_col = cols["time"]
    assert time_col is not None

    for _, r in df.iterrows():
        raw_track = str(r[cols["track_id"]])  # type: ignore[index]
        raw_type = r[cols["object_type"]] if cols.get("object_type") is not None else "vehicle"
        raw_cat = r[cols["object_category"]] if cols.get("object_category") is not None else ""
        agent_type = _normalize_agent_type(raw_type)
        category = str(raw_cat).strip().lower()

        is_ego = raw_track == str(ego_track_id) or agent_type == "ego" or category == "av"
        final_track = "ego" if is_ego else raw_track
        final_type = "ego" if is_ego else agent_type
        length, width = _default_size(final_type)

        vx = 0.0
        vy = 0.0
        if cols.get("vx") is not None and pd.notna(r[cols["vx"]]):  # type: ignore[index]
            vx = float(r[cols["vx"]])  # type: ignore[index]
        if cols.get("vy") is not None and pd.notna(r[cols["vy"]]):  # type: ignore[index]
            vy = float(r[cols["vy"]])  # type: ignore[index]

        rows.append({
            "scene_id": sid,
            "track_id": final_track,
            "timestamp": _time_to_seconds(df[time_col], r[time_col], timestep_hz),
            "x": float(r[cols["x"]]),  # type: ignore[index]
            "y": float(r[cols["y"]]),  # type: ignore[index]
            "vx": vx,
            "vy": vy,
            "yaw": float(r[cols["yaw"]]) if cols.get("yaw") is not None and pd.notna(r[cols["yaw"]]) else 0.0,  # type: ignore[index]
            "length": length,
            "width": width,
            "agent_type": final_type,
            "is_ego": bool(is_ego),
        })

    out = pd.DataFrame(rows)
    if not out["is_ego"].any():
        if require_ego:
            raise ValueError(f"No AV/ego track found in {scenario_path}. Set ego_track_id or disable require_ego.")
        # fallback: choose the longest-lived track as ego for custom AV2-like exports.
        counts = out.groupby("track_id").size().sort_values(ascending=False)
        fallback = str(counts.index[0])
        out.loc[out["track_id"].astype(str) == fallback, "is_ego"] = True
        out.loc[out["track_id"].astype(str) == fallback, "track_id"] = "ego"
        out.loc[out["track_id"].astype(str) == "ego", "agent_type"] = "ego"

    if max_tracks is not None and max_tracks > 0:
        t0 = out["timestamp"].min()
        first = out[out["timestamp"] == t0].copy()
        ego_first = first[first["is_ego"]]
        if not ego_first.empty:
            ex, ey = float(ego_first.iloc[0]["x"]), float(ego_first.iloc[0]["y"])
            non = first[~first["is_ego"]].copy()
            non["dist"] = ((non["x"] - ex) ** 2 + (non["y"] - ey) ** 2) ** 0.5
            keep = ["ego"] + list(non.sort_values("dist")["track_id"].astype(str).head(max_tracks))
            out = out[out["track_id"].astype(str).isin(keep)].copy()

    # Sort, remove exact duplicate rows, and write.
    out = out[GENERIC_COLUMNS].sort_values(["timestamp", "track_id"]).drop_duplicates()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out_csv


def find_av2_scenario_files(root: str | Path, limit: Optional[int] = None) -> List[Path]:
    root = Path(root)
    if root.is_file() and root.suffix.lower() in {".parquet", ".csv"}:
        return [root]
    patterns = ["*.parquet", "*/*.parquet", "*/*/*.parquet", "*.csv", "*/*.csv"]
    files: List[Path] = []
    for pat in patterns:
        files.extend(root.glob(pat))
    # Official AV2 also has map parquet/json files in scenario folders. Keep likely scenario files.
    files = [p for p in sorted(set(files)) if "map" not in p.name.lower() and "log_map" not in p.name.lower()]
    if limit is not None:
        files = files[:limit]
    return files


def batch_convert_av2_to_generic_csv(
    av2_root: str | Path,
    out_dir: str | Path,
    limit: Optional[int] = None,
    max_tracks: Optional[int] = 32,
    timestep_hz: float = 10.0,
) -> Dict[str, object]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = find_av2_scenario_files(av2_root, limit=limit)
    rows = []
    converted = 0
    for f in files:
        sid = f.parent.name if f.parent.name else f.stem
        out_csv = out_dir / f"{sid}.csv"
        try:
            convert_av2_scenario_to_generic_csv(f, out_csv, scene_id=sid, max_tracks=max_tracks, timestep_hz=timestep_hz)
            status = "ok"
            converted += 1
            err = ""
        except Exception as exc:  # keep batch robust
            status = "error"
            err = str(exc)
        rows.append({"scenario_path": str(f), "scene_id": sid, "out_csv": str(out_csv), "status": status, "error": err})
    # public_release: keep auxiliary files OUTSIDE generic_tracks_csv/.
    # run_batch_csv consumes all *.csv in generic_tracks_csv, so manifest.csv must not live there.
    aux_dir = out_dir.parent
    manifest = pd.DataFrame(rows)
    manifest.to_csv(aux_dir / "conversion_manifest.csv", index=False)
    summary = {
        "num_files_found": len(files),
        "num_converted": converted,
        "out_dir": str(out_dir),
        "manifest": str(aux_dir / "conversion_manifest.csv"),
    }
    (aux_dir / "conversion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def create_mock_av2_scenario(out_path: str | Path) -> Path:
    """Create a tiny AV2-like parquet file for converter testing without downloading the dataset."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    # We use AV2-like integer timesteps, but step by 5 so that the converter
    # (timestep / 10 Hz) yields 0.5-second intervals. This keeps the mock small
    # while preserving an 8-second horizon.
    dt = 0.5
    n = 17
    # AV ego moves along x.
    for i in range(n):
        t = i * 5
        rows.append({
            "track_id": "AV", "timestep": t, "position_x": 8.0 * i * dt, "position_y": 0.0,
            "heading": 0.0, "velocity_x": 8.0, "velocity_y": 0.0,
            "object_type": "AV", "object_category": "FOCAL_TRACK",
        })
        # Lead vehicle.
        rows.append({
            "track_id": "lead_veh_mock", "timestep": t, "position_x": 26.0 + 7.3 * i * dt, "position_y": 0.0,
            "heading": 0.0, "velocity_x": 7.3, "velocity_y": 0.0,
            "object_type": "VEHICLE", "object_category": "SCORED_TRACK",
        })
        # Side vehicle candidate.
        rows.append({
            "track_id": "side_veh_mock", "timestep": t, "position_x": 10.0 + 8.4 * i * dt, "position_y": 3.5,
            "heading": 0.0, "velocity_x": 8.4, "velocity_y": 0.0,
            "object_type": "VEHICLE", "object_category": "TRACK_FRAGMENT",
        })
    df = pd.DataFrame(rows)
    # Prefer parquet because official AV2 uses parquet, but keep the mock test usable
    # even before pyarrow is installed. The real AV2 converter still supports parquet
    # once pyarrow is available via requirements.txt.
    try:
        df.to_parquet(out_path, index=False)
        return out_path
    except Exception:
        csv_path = out_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="AV2 Motion Forecasting utilities for CausalSensor4D.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sum = sub.add_parser("summarize")
    p_sum.add_argument("--scenario", required=True)

    p_one = sub.add_parser("convert-one")
    p_one.add_argument("--scenario", required=True)
    p_one.add_argument("--out", required=True)
    p_one.add_argument("--scene-id", default=None)
    p_one.add_argument("--max-tracks", type=int, default=32)

    p_batch = sub.add_parser("convert-batch")
    p_batch.add_argument("--root", required=True)
    p_batch.add_argument("--out-dir", required=True)
    p_batch.add_argument("--limit", type=int, default=None)
    p_batch.add_argument("--max-tracks", type=int, default=32)

    p_mock = sub.add_parser("make-mock")
    p_mock.add_argument("--out", required=True)

    args = parser.parse_args()
    if args.cmd == "summarize":
        print(json.dumps(summarize_av2_table(args.scenario), indent=2))
    elif args.cmd == "convert-one":
        out = convert_av2_scenario_to_generic_csv(args.scenario, args.out, scene_id=args.scene_id, max_tracks=args.max_tracks)
        print(f"Converted: {out}")
    elif args.cmd == "convert-batch":
        print(json.dumps(batch_convert_av2_to_generic_csv(args.root, args.out_dir, limit=args.limit, max_tracks=args.max_tracks), indent=2))
    elif args.cmd == "make-mock":
        print(f"Mock AV2-like scenario written to: {create_mock_av2_scenario(args.out)}")


if __name__ == "__main__":
    main()
