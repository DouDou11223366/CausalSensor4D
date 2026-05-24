from __future__ import annotations

"""Build report-ready summaries for LLM-proposed and simulator-verified candidates.

public_release fixes the clean-safe label propagation issue discovered in public_release.

Core principle:
    LLM proposes; CausalSensor4D verifies.

The module never trusts LLM claims as numerical evidence. It reads deterministic
candidate-verification results and optionally propagates clean-safe labels from an
upstream original-safety filter or from an explicitly supplied safe CSV folder.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import json
import math

import pandas as pd


@dataclass
class LLMVerifiedBenchmarkConfig:
    min_original_ttc_safe: float = 2.0
    require_no_original_collision: bool = True
    require_no_original_hard_brake: bool = True
    diagnosis_min_chars: int = 300
    # public_release: label-propagation options.
    # If True, every candidate whose resolved CSV is in the input folder is treated
    # as originating from a previously clean-safe scene. Use this only when the CSV
    # folder is exactly the output of original_safety_filter/safe_csv.
    assume_input_csv_clean_safe: bool = False
    safety_table_path: Optional[str] = None
    safe_csv_dir: Optional[str] = None


def _read_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_text(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return False
    s = str(x).strip().lower()
    return s in {"true", "1", "yes", "y"}


def _safe_float(x: Any, default: Any = float("nan")) -> Any:
    try:
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        return float(x)
    except Exception:
        return default


def _safe_rate(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.mean())


def _scene_key_from_path(x: Any) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    s = str(x).strip()
    if not s:
        return ""
    try:
        return Path(s).stem
    except Exception:
        return s


def _scene_key_from_row(row: pd.Series) -> str:
    scene = row.get("scene_id_resolved") or row.get("scene_id_requested") or row.get("scene_id")
    if scene is not None and not (isinstance(scene, float) and math.isnan(scene)):
        scene_s = str(scene).strip()
        if scene_s and scene_s.lower() != "nan":
            return Path(scene_s).stem
    return _scene_key_from_path(row.get("csv_path"))


def _diagnosis_quality(diagnosis_md: str | Path, response_validation: Dict[str, Any], config: LLMVerifiedBenchmarkConfig) -> Dict[str, Any]:
    text = _read_text(diagnosis_md)
    chars = len(text.strip())
    rv_diag = response_validation.get("diagnosis", {}) if isinstance(response_validation, dict) else {}
    assistant_chars_reported = rv_diag.get("assistant_text_chars")
    ok = bool(rv_diag.get("ok", False)) and chars >= config.diagnosis_min_chars
    reason = []
    if not rv_diag.get("ok", False):
        reason.append("openrouter_response_not_ok")
    if chars < config.diagnosis_min_chars:
        reason.append(f"assistant_text_too_short<{config.diagnosis_min_chars}")
    return {
        "diagnosis_path": str(diagnosis_md),
        "file_text_chars": chars,
        "assistant_text_chars_reported": assistant_chars_reported,
        "usable_for_report_without_manual_review": ok,
        "quality_flags": reason,
    }


def _candidate_validation_summary(candidate_validation_json: str | Path) -> Dict[str, Any]:
    data = _read_json(candidate_validation_json)
    return {
        "parse_ok": bool(data.get("parse_ok", False)),
        "parse_error": data.get("parse_error"),
        "num_candidates": int(data.get("num_candidates", 0) or 0),
        "num_valid_for_parsing": int(data.get("num_valid_for_parsing", 0) or 0),
        "num_valid_for_current_search": int(data.get("num_valid_for_current_search", 0) or 0),
    }


def _load_verification_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    for col in [
        "verified", "valid_edit_family", "original_collision", "original_hard_brake",
        "verified_collision", "verified_hard_brake", "is_clean_safe_original",
    ]:
        if col in df.columns:
            df[col] = df[col].map(_as_bool)
    for col in ["original_min_ttc", "verified_cost", "verified_min_ttc", "original_risk_score", "verified_risk_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _load_clean_scene_ids_from_safety_table(path: str | Path | None) -> Set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    try:
        df = pd.read_csv(p)
    except Exception:
        return set()
    if df.empty:
        return set()

    # Accept either original_safety_table.csv or safe_selected_scenes.csv.
    if "is_original_safe" in df.columns:
        safe_df = df[df["is_original_safe"].map(_as_bool)].copy()
    else:
        safe_df = df.copy()
    keys: Set[str] = set()
    for _, r in safe_df.iterrows():
        for col in ["scene_id", "csv_path"]:
            if col in safe_df.columns:
                v = r.get(col)
                key = _scene_key_from_path(v) if col == "csv_path" else str(v).strip()
                if key and key.lower() != "nan":
                    keys.add(Path(key).stem)
    return keys


def _load_clean_scene_ids_from_safe_csv_dir(path: str | Path | None) -> Set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    return {x.stem for x in p.glob("*.csv")}


def _add_clean_safe_flag(df: pd.DataFrame, config: LLMVerifiedBenchmarkConfig) -> pd.DataFrame:
    if len(df) == 0:
        return df
    out = df.copy()
    out["scene_key"] = out.apply(_scene_key_from_row, axis=1)

    safety_table_ids = _load_clean_scene_ids_from_safety_table(config.safety_table_path)
    safe_dir_ids = _load_clean_scene_ids_from_safe_csv_dir(config.safe_csv_dir)
    propagated_ids = safety_table_ids | safe_dir_ids

    if config.assume_input_csv_clean_safe:
        propagated_clean = pd.Series([True] * len(out), index=out.index)
        source = "assume_input_csv_clean_safe"
    elif propagated_ids:
        propagated_clean = out["scene_key"].astype(str).isin(propagated_ids)
        source = "safety_table_or_safe_csv_dir"
    else:
        propagated_clean = pd.Series([False] * len(out), index=out.index)
        source = "not_available"

    # Fallback: compute clean from original risk columns if no propagated label exists.
    original_collision = out.get("original_collision", pd.Series([False] * len(out), index=out.index)).fillna(False).map(_as_bool)
    original_hard_brake = out.get("original_hard_brake", pd.Series([False] * len(out), index=out.index)).fillna(False).map(_as_bool)
    original_min_ttc = pd.to_numeric(out.get("original_min_ttc", pd.Series([float("nan")] * len(out), index=out.index)), errors="coerce")

    computed_clean = pd.Series([True] * len(out), index=out.index)
    if config.require_no_original_collision:
        computed_clean &= ~original_collision
    if config.require_no_original_hard_brake:
        computed_clean &= ~original_hard_brake
    computed_clean &= original_min_ttc.fillna(-1.0) >= config.min_original_ttc_safe

    if config.assume_input_csv_clean_safe or propagated_ids:
        clean = propagated_clean
    else:
        clean = computed_clean
        source = "computed_from_verification_original_risk"

    out["is_clean_safe_original"] = clean.astype(bool)
    out["clean_safe_label_source"] = source
    out["clean_safe_label_propagated"] = propagated_clean.astype(bool)
    out["clean_safe_label_computed"] = computed_clean.astype(bool)
    return out


def _verification_summary(df: pd.DataFrame, config: LLMVerifiedBenchmarkConfig) -> Dict[str, Any]:
    if len(df) == 0:
        return {
            "num_candidates": 0,
            "num_unique_candidates": 0,
            "num_verified": 0,
            "verified_rate": 0.0,
            "num_clean_safe_original": 0,
            "clean_safe_rate": 0.0,
            "num_verified_clean_safe": 0,
            "verified_clean_safe_rate": 0.0,
            "mean_verified_cost": None,
            "min_verified_cost": None,
            "max_verified_cost": None,
            "verified_edit_type_counts": {},
            "verified_clean_safe_edit_type_counts": {},
            "clean_safe_config": asdict(config),
        }
    verified = df["verified"].fillna(False).map(_as_bool) if "verified" in df else pd.Series([False] * len(df), index=df.index)
    clean = df["is_clean_safe_original"].fillna(False).map(_as_bool) if "is_clean_safe_original" in df else pd.Series([False] * len(df), index=df.index)
    verified_df = df[verified].copy()
    verified_clean_df = df[verified & clean].copy()
    costs = pd.to_numeric(verified_df.get("verified_cost", pd.Series(dtype=float)), errors="coerce").dropna()
    edit_counts = verified_df.get("edit_family", pd.Series(dtype=str)).value_counts().to_dict() if len(verified_df) else {}
    clean_edit_counts = verified_clean_df.get("edit_family", pd.Series(dtype=str)).value_counts().to_dict() if len(verified_clean_df) else {}

    unique_cols = [c for c in ["scene_key", "target_agent_id", "edit_family"] if c in df.columns]
    unique_count = int(len(df.drop_duplicates(unique_cols))) if unique_cols else int(len(df))

    return {
        "num_candidates": int(len(df)),
        "num_unique_candidates": unique_count,
        "num_verified": int(verified.sum()),
        "verified_rate": float(verified.mean()) if len(df) else 0.0,
        "num_clean_safe_original": int(clean.sum()),
        "clean_safe_rate": float(clean.mean()) if len(df) else 0.0,
        "num_verified_clean_safe": int((verified & clean).sum()),
        "verified_clean_safe_rate": float((verified & clean).mean()) if len(df) else 0.0,
        "mean_verified_cost": float(costs.mean()) if len(costs) else None,
        "min_verified_cost": float(costs.min()) if len(costs) else None,
        "max_verified_cost": float(costs.max()) if len(costs) else None,
        "verified_edit_type_counts": {str(k): int(v) for k, v in edit_counts.items()},
        "verified_clean_safe_edit_type_counts": {str(k): int(v) for k, v in clean_edit_counts.items()},
        "clean_safe_config": asdict(config),
    }


def _edit_type_table(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "edit_family", "num_verified", "num_unique_verified", "mean_verified_cost",
        "median_verified_cost", "collision_rate", "hard_brake_rate",
        "clean_safe_original_count",
    ]
    if len(df) == 0 or "verified" not in df:
        return pd.DataFrame(columns=columns)
    verified = df[df["verified"].fillna(False).map(_as_bool)].copy()
    if len(verified) == 0:
        return pd.DataFrame(columns=columns)
    rows = []
    for edit, g in verified.groupby("edit_family"):
        clean_col = g.get("is_clean_safe_original", pd.Series([False] * len(g), index=g.index)).fillna(False).map(_as_bool)
        rows.append({
            "edit_family": edit,
            "num_verified": int(len(g)),
            "num_unique_verified": int(len(g.drop_duplicates([c for c in ["scene_key", "target_agent_id", "edit_family"] if c in g.columns]))),
            "mean_verified_cost": float(pd.to_numeric(g["verified_cost"], errors="coerce").mean()) if "verified_cost" in g else None,
            "median_verified_cost": float(pd.to_numeric(g["verified_cost"], errors="coerce").median()) if "verified_cost" in g else None,
            "collision_rate": _safe_rate(g["verified_collision"].fillna(False).map(_as_bool)) if "verified_collision" in g else 0.0,
            "hard_brake_rate": _safe_rate(g["verified_hard_brake"].fillna(False).map(_as_bool)) if "verified_hard_brake" in g else 0.0,
            "clean_safe_original_count": int(clean_col.sum()),
        })
    return pd.DataFrame(rows).sort_values(["num_verified", "mean_verified_cost"], ascending=[False, True])


def _export_verified_candidates(df: pd.DataFrame, out_dir: Path) -> str:
    if len(df) == 0 or "verified" not in df:
        payload: List[Dict[str, Any]] = []
    else:
        verified = df[df["verified"].fillna(False).map(_as_bool)].copy()
        if "verified_cost" in verified:
            verified = verified.sort_values("verified_cost", ascending=True)
        payload = []
        for _, r in verified.iterrows():
            payload.append({
                "scene_id": r.get("scene_id_resolved") or r.get("scene_id_requested") or r.get("scene_key"),
                "target_agent_id": str(r.get("target_agent_id")),
                "edit_family": r.get("edit_family"),
                "verified_cost": _safe_float(r.get("verified_cost"), None),
                "verified_collision": bool(_as_bool(r.get("verified_collision"))),
                "verified_hard_brake": bool(_as_bool(r.get("verified_hard_brake"))),
                "verified_min_ttc": _safe_float(r.get("verified_min_ttc"), None),
                "is_clean_safe_original": bool(_as_bool(r.get("is_clean_safe_original"))),
                "clean_safe_label_source": r.get("clean_safe_label_source"),
                "verified_parameters": r.get("verified_parameters"),
            })
    path = out_dir / "llm_verified_candidate_priority_list.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def build_llm_verified_benchmark(
    response_validation_json: str | Path,
    candidate_validation_json: str | Path,
    candidate_verification_table: str | Path,
    diagnosis_md: str | Path,
    out_dir: str | Path,
    config: Optional[LLMVerifiedBenchmarkConfig] = None,
) -> Dict[str, Any]:
    config = config or LLMVerifiedBenchmarkConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    response_validation = _read_json(response_validation_json)
    diagnosis_quality = _diagnosis_quality(diagnosis_md, response_validation, config)
    candidate_validation = _candidate_validation_summary(candidate_validation_json)

    df = _load_verification_table(candidate_verification_table)
    df = _add_clean_safe_flag(df, config)
    enriched_table_path = out_dir / "llm_candidate_verification_enriched.csv"
    df.to_csv(enriched_table_path, index=False)

    verification_summary = _verification_summary(df, config)
    edit_table = _edit_type_table(df)
    edit_table_path = out_dir / "llm_verified_edit_type_table.csv"
    edit_table.to_csv(edit_table_path, index=False)
    priority_path = _export_verified_candidates(df, out_dir)

    summary = {
        "version": "public_release",
        "response_validation_json": str(response_validation_json),
        "candidate_validation_json": str(candidate_validation_json),
        "candidate_verification_table": str(candidate_verification_table),
        "diagnosis_quality": diagnosis_quality,
        "candidate_validation": candidate_validation,
        "verification_summary": verification_summary,
        "outputs": {
            "enriched_verification_table": str(enriched_table_path),
            "edit_type_table": str(edit_table_path),
            "verified_candidate_priority_list": priority_path,
        },
    }
    summary_path = out_dir / "llm_verified_benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# CausalSensor4D public_release LLM-Verified Counterfactual Benchmark",
        "",
        "## Core principle",
        "The LLM proposes counterfactual candidates, but CausalSensor4D verifies them with deterministic search and risk evaluation. Only verified candidates should be used as experimental evidence.",
        "",
        "## public_release clean-safe label propagation",
        "public_release fixes the public_release clean-safe labeling issue by allowing the benchmark to propagate original-safety labels from an upstream safety table or from an explicitly supplied safe CSV folder. This prevents candidates from being incorrectly marked as non-clean when they were generated from a previously filtered safe subset.",
        "",
        "## Online response quality",
        f"- Diagnosis usable without manual review: `{diagnosis_quality['usable_for_report_without_manual_review']}`",
        f"- Diagnosis text chars: `{diagnosis_quality['file_text_chars']}`",
        f"- Diagnosis quality flags: `{diagnosis_quality['quality_flags']}`",
        "",
        "## Candidate JSON quality",
        f"- Parse OK: `{candidate_validation['parse_ok']}`",
        f"- Candidate count: `{candidate_validation['num_candidates']}`",
        f"- Valid for current search: `{candidate_validation['num_valid_for_current_search']}`",
        "",
        "## Deterministic verification summary",
        f"- Candidates: `{verification_summary['num_candidates']}`",
        f"- Unique candidate triples: `{verification_summary['num_unique_candidates']}`",
        f"- Verified failures: `{verification_summary['num_verified']}`",
        f"- Verification rate: `{verification_summary['verified_rate']:.3f}`",
        f"- Mean verified cost: `{verification_summary['mean_verified_cost']}`",
        f"- Verified edit-type counts: `{verification_summary['verified_edit_type_counts']}`",
        "",
        "## Clean safe-to-failure check",
        f"- Original-clean candidates: `{verification_summary['num_clean_safe_original']}`",
        f"- Verified clean safe-to-failure candidates: `{verification_summary['num_verified_clean_safe']}`",
        f"- Verified clean safe-to-failure rate: `{verification_summary['verified_clean_safe_rate']:.3f}`",
        f"- Clean-safe edit-type counts: `{verification_summary['verified_clean_safe_edit_type_counts']}`",
        f"- Clean-safe configuration: `{verification_summary['clean_safe_config']}`",
        "",
        "This check separates general failure-discovery evidence from strict safe-to-failure evidence. For the strict task, report verified clean-safe candidates separately.",
        "",
        "## Verified candidates by edit family",
    ]
    if len(edit_table):
        lines.append(edit_table.to_markdown(index=False))
    else:
        lines.append("No verified candidates were available.")
    lines += [
        "",
        "## Report-ready interpretation",
        "This run can be described as an LLM-assisted proposal experiment: the online LLM generated machine-readable candidate interventions, the parser validated the JSON schema, and CausalSensor4D re-ran deterministic counterfactual search to verify whether each proposal induces a failure. The LLM is not used as a numerical evaluator. In public_release, clean-safe status is propagated from the upstream original-safety filter when available, so strict safe-to-failure evidence can be counted correctly.",
    ]
    report_path = out_dir / "llm_verified_benchmark_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "summary": str(summary_path),
        "report": str(report_path),
        "enriched_table": str(enriched_table_path),
        "edit_type_table": str(edit_table_path),
        "verified_candidate_priority_list": priority_path,
    }
