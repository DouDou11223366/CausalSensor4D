from __future__ import annotations

"""Verify LLM-proposed counterfactual candidates with the deterministic simulator.

The LLM proposes; CausalSensor4D verifies. This module converts parsed candidate
proposals into executable search calls when possible. It does not trust LLM claims.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import pandas as pd

from .ad_model import make_ad_model
from .data_adapters.generic_tracks_csv import load_tracks_csv, REQUIRED_COLUMNS
from .search import search_minimum_failure_cost
from .risk import evaluate_scene

EXECUTABLE_EDIT_FAMILIES = {"lead_brake", "cut_in", "pedestrian_crossing"}

VERIFICATION_COLUMNS = [
    "proposal_index", "scene_id_requested", "scene_id_resolved", "csv_path",
    "target_agent_id", "edit_family", "expected_failure_mode",
    "valid_edit_family", "verified", "verification_error",
    "num_search_candidates",
    "original_collision", "original_hard_brake", "original_min_ttc", "original_risk_score",
    "verified_edit_name", "verified_cost", "verified_collision", "verified_hard_brake",
    "verified_min_ttc", "verified_risk_score", "verified_parameters",
]


def _is_generic_scene_csv(csv_path: Path) -> bool:
    try:
        head = pd.read_csv(csv_path, nrows=5)
    except Exception:
        return False
    return REQUIRED_COLUMNS.issubset(set(head.columns))


def _load_candidates(candidate_json_path: str | Path) -> List[Dict[str, Any]]:
    data = json.loads(Path(candidate_json_path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("candidates"), list):
        data = data["candidates"]
    if not isinstance(data, list):
        raise ValueError("Parsed candidate file must contain a JSON list")
    return [x for x in data if isinstance(x, dict)]


def _build_csv_index(csv_dir: str | Path) -> Dict[str, Path]:
    csv_dir = Path(csv_dir)
    return {p.stem: p for p in sorted(csv_dir.glob("*.csv")) if _is_generic_scene_csv(p)}


def _find_scene_for_candidate(candidate: Dict[str, Any], csv_index: Dict[str, Path]) -> Optional[Path]:
    scene_id = candidate.get("scene_id")
    if scene_id is not None:
        scene_key = str(scene_id).strip()
        if scene_key in csv_index:
            return csv_index[scene_key]

    # Fallback: find a CSV containing target_agent_id. This is slower but useful
    # when the LLM forgot scene_id.
    target = candidate.get("target_agent_id")
    if target is None:
        return None
    target_str = str(target)
    for path in csv_index.values():
        try:
            df = pd.read_csv(path, usecols=["track_id"], dtype={"track_id": str})
            if target_str in set(df["track_id"].astype(str)):
                return path
        except Exception:
            continue
    return None


def verify_llm_candidates(
    candidate_json_path: str | Path,
    csv_dir: str | Path,
    out_dir: str | Path,
    ad_model_name: str = "rule_delayed",
    ego_track_id: str = "ego",
) -> Dict[str, Any]:
    candidates = _load_candidates(candidate_json_path)
    csv_index = _build_csv_index(csv_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = make_ad_model(ad_model_name)

    rows: List[Dict[str, Any]] = []
    for idx, cand in enumerate(candidates):
        edit_family = str(cand.get("edit_family", "")).strip()
        target = cand.get("target_agent_id")
        row: Dict[str, Any] = {
            "proposal_index": idx,
            "scene_id_requested": cand.get("scene_id"),
            "target_agent_id": target,
            "edit_family": edit_family,
            "expected_failure_mode": cand.get("expected_failure_mode"),
            "valid_edit_family": edit_family in EXECUTABLE_EDIT_FAMILIES,
            "verified": False,
            "verification_error": "",
        }
        if edit_family not in EXECUTABLE_EDIT_FAMILIES:
            row["verification_error"] = "edit_family_not_executable_in_current_search"
            rows.append(row)
            continue
        if target is None:
            row["verification_error"] = "missing_target_agent_id"
            rows.append(row)
            continue

        csv_path = _find_scene_for_candidate(cand, csv_index)
        if csv_path is None:
            row["verification_error"] = "scene_csv_not_found"
            rows.append(row)
            continue

        try:
            scene = load_tracks_csv(csv_path, scene_id=None, ego_track_id=ego_track_id)
            target_str = str(target)
            if target_str not in scene.agents:
                # Try integer-like normalization, because CSV track IDs may be strings.
                matched = None
                for aid in scene.agents.keys():
                    if str(aid) == target_str or str(aid).split(".0")[0] == target_str.split(".0")[0]:
                        matched = aid
                        break
                if matched is None:
                    raise ValueError(f"target_agent_id {target} not found in scene {scene.scene_id}")
                target_str = matched

            original_scene = model.rollout(scene)
            original_risk = evaluate_scene(original_scene)
            result = search_minimum_failure_cost(scene, model, target_str, allowed_edit=edit_family)
            best = result.best
            row.update({
                "scene_id_resolved": scene.scene_id,
                "csv_path": str(csv_path),
                "num_search_candidates": int(len(result.table)),
                "original_collision": bool(original_risk.collision),
                "original_hard_brake": bool(original_risk.hard_brake),
                "original_min_ttc": original_risk.min_ttc,
                "original_risk_score": original_risk.risk_score,
            })
            if best is None:
                row.update({"verified": False, "verification_error": "no_failure_found_for_proposal"})
            else:
                best_json = {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()}
                row.update({
                    "verified": bool(best_json.get("failure", False)),
                    "verified_edit_name": best_json.get("edit_name"),
                    "verified_cost": best_json.get("cost"),
                    "verified_collision": best_json.get("collision"),
                    "verified_hard_brake": best_json.get("hard_brake"),
                    "verified_min_ttc": best_json.get("min_ttc"),
                    "verified_risk_score": best_json.get("risk_score"),
                    "verified_parameters": json.dumps(best_json.get("parameters", {}), ensure_ascii=False),
                })
        except Exception as exc:
            row["verification_error"] = str(exc)
        rows.append(row)

    # Always write a CSV with headers, even when the LLM returned zero parseable
    # candidates. Otherwise pandas.read_csv may raise EmptyDataError downstream.
    df = pd.DataFrame(rows)
    for col in VERIFICATION_COLUMNS:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")
    df = df[VERIFICATION_COLUMNS]
    table_path = out_dir / "llm_candidate_verification_table.csv"
    df.to_csv(table_path, index=False)

    summary = {
        "candidate_json_path": str(candidate_json_path),
        "csv_dir": str(csv_dir),
        "ad_model_name": ad_model_name,
        "num_candidates": int(len(df)),
        "num_verified": int(df["verified"].fillna(False).sum()) if "verified" in df else 0,
        "verified_rate": float(df["verified"].fillna(False).mean()) if len(df) else 0.0,
        "num_executable_edit_family": int(df["valid_edit_family"].fillna(False).sum()) if "valid_edit_family" in df else 0,
    }
    summary_path = out_dir / "llm_candidate_verification_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# LLM Candidate Verification Report",
        "",
        f"- Candidate file: `{candidate_json_path}`",
        f"- CSV directory: `{csv_dir}`",
        f"- AD model: `{ad_model_name}`",
        f"- Candidates: `{summary['num_candidates']}`",
        f"- Verified failures: `{summary['num_verified']}`",
        f"- Verification rate: `{summary['verified_rate']:.3f}`",
        "",
        "The LLM only proposes candidates. A candidate is counted as verified only if CausalSensor4D re-runs the deterministic counterfactual search and finds a failure under the proposed edit family and target.",
        "",
    ]
    if len(df):
        view = df[[c for c in [
            "proposal_index", "scene_id_resolved", "target_agent_id", "edit_family", "verified", "verified_cost", "verified_collision", "verified_hard_brake", "verification_error"
        ] if c in df.columns]].copy()
        lines.append(view.to_markdown(index=False))
    report_path = out_dir / "llm_candidate_verification_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    return {"summary": str(summary_path), "table": str(table_path), "report": str(report_path), "summary_dict": summary}
