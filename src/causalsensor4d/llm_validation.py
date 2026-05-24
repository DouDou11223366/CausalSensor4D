from __future__ import annotations

"""Validation utilities for online LLM responses in CausalSensor4D public_release.

The LLM is allowed to explain and propose candidates, but not to decide final
results. This module checks whether OpenRouter actually returned a usable chat
completion and whether candidate proposals can be parsed as machine-readable JSON.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import re

ALLOWED_EDIT_FAMILIES = {
    "lead_brake",
    "cut_in",
    "pedestrian_crossing",
    "occlusion_increase",
    "visibility_degradation",
    "traffic_light_change",
    "lane_marking_degradation",
    "reaction_delay",
}

REQUIRED_CANDIDATE_KEYS = [
    "scene_id",
    "target_agent_id",
    "edit_family",
    "parameter_suggestion",
    "expected_failure_mode",
    "physical_plausibility_reason",
    "why_this_is_minimal",
]


@dataclass
class ResponseValidation:
    raw_response_path: str
    file_exists: bool
    ok: bool
    http_status_code: int | None
    openrouter_model_requested: str | None
    openrouter_response_model: str | None
    response_id: str | None
    has_choices: bool
    num_choices: int
    assistant_text_chars: int
    has_usage: bool
    usage: Dict[str, Any] | None
    error_message: str | None = None


def _safe_read_json(path: str | Path) -> Tuple[Dict[str, Any] | None, str | None]:
    path = Path(path)
    if not path.exists():
        return None, f"File not found: {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, f"Could not parse JSON: {exc}"


def validate_openrouter_raw_response(raw_response_path: str | Path) -> ResponseValidation:
    raw_response_path = Path(raw_response_path)
    data, error = _safe_read_json(raw_response_path)
    if data is None:
        return ResponseValidation(
            raw_response_path=str(raw_response_path),
            file_exists=raw_response_path.exists(),
            ok=False,
            http_status_code=None,
            openrouter_model_requested=None,
            openrouter_response_model=None,
            response_id=None,
            has_choices=False,
            num_choices=0,
            assistant_text_chars=0,
            has_usage=False,
            usage=None,
            error_message=error,
        )

    response_json = data.get("response_json", {}) if isinstance(data, dict) else {}
    choices = response_json.get("choices", []) if isinstance(response_json, dict) else []
    assistant_text = data.get("assistant_text", "") if isinstance(data, dict) else ""
    request_config = data.get("request_config", {}) if isinstance(data, dict) else {}

    # OpenRouter/OpenAI may put error in response_json.error.
    err_msg = None
    if isinstance(response_json, dict) and response_json.get("error"):
        err_msg = json.dumps(response_json.get("error"), ensure_ascii=False)

    return ResponseValidation(
        raw_response_path=str(raw_response_path),
        file_exists=True,
        ok=bool(data.get("ok", False)),
        http_status_code=data.get("status_code"),
        openrouter_model_requested=request_config.get("model"),
        openrouter_response_model=response_json.get("model") if isinstance(response_json, dict) else None,
        response_id=response_json.get("id") if isinstance(response_json, dict) else None,
        has_choices=bool(choices),
        num_choices=len(choices) if isinstance(choices, list) else 0,
        assistant_text_chars=len(str(assistant_text or "")),
        has_usage=bool(response_json.get("usage")) if isinstance(response_json, dict) else False,
        usage=response_json.get("usage") if isinstance(response_json, dict) else None,
        error_message=err_msg,
    )


def write_response_validation_report(raw_paths: Dict[str, str], out_dir: str | Path) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {name: asdict(validate_openrouter_raw_response(path)) for name, path in raw_paths.items()}
    json_path = out_dir / "openrouter_response_validation.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# OpenRouter Response Validation Report", ""]
    for name, res in results.items():
        lines += [
            f"## {name}",
            f"- raw response: `{res['raw_response_path']}`",
            f"- file exists: `{res['file_exists']}`",
            f"- ok: `{res['ok']}`",
            f"- http status code: `{res['http_status_code']}`",
            f"- requested model: `{res['openrouter_model_requested']}`",
            f"- response model: `{res['openrouter_response_model']}`",
            f"- response id: `{res['response_id']}`",
            f"- choices: `{res['num_choices']}`",
            f"- assistant text chars: `{res['assistant_text_chars']}`",
            f"- usage available: `{res['has_usage']}`",
            f"- usage: `{res['usage']}`",
            f"- error: `{res['error_message']}`",
            "",
        ]
    report_path = out_dir / "openrouter_response_validation_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": str(json_path), "report": str(report_path), "results": results}


def extract_json_candidate_object(text: str) -> Any:
    """Extract a JSON object/list from an LLM text response.

    The parser first tries exact JSON, then fenced ```json blocks, then the first
    bracketed list/object substring. It raises ValueError if no valid JSON exists.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty LLM response text")

    try:
        return json.loads(text)
    except Exception:
        pass

    fence_matches = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    for block in fence_matches:
        block = block.strip()
        try:
            return json.loads(block)
        except Exception:
            continue

    # Find first JSON array or object. Prefer array because candidate proposals should be a list.
    starts = [i for i in [text.find("["), text.find("{")] if i != -1]
    if not starts:
        raise ValueError("No JSON bracket found in response")
    start = min(starts)
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text[start:])
        return obj
    except Exception as exc:
        raise ValueError(f"Could not decode bracketed JSON: {exc}") from exc


def normalize_candidate_list(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, dict):
        # Some models return {"candidates": [...]}.
        if isinstance(obj.get("candidates"), list):
            obj = obj["candidates"]
        else:
            obj = [obj]
    if not isinstance(obj, list):
        raise ValueError(f"Candidate JSON must be a list or candidates object, got {type(obj).__name__}")
    out = []
    for item in obj:
        if isinstance(item, dict):
            out.append(item)
    return out


def validate_candidate(candidate: Dict[str, Any], idx: int) -> Dict[str, Any]:
    missing = [k for k in REQUIRED_CANDIDATE_KEYS if k not in candidate or candidate.get(k) in (None, "")]
    edit_family = str(candidate.get("edit_family", "")).strip()
    allowed_edit = edit_family in ALLOWED_EDIT_FAMILIES
    valid_for_parsing = len(missing) == 0 and allowed_edit
    valid_for_current_search = valid_for_parsing and edit_family in {"lead_brake", "cut_in", "pedestrian_crossing"}
    return {
        "proposal_index": idx,
        "valid_for_parsing": valid_for_parsing,
        "valid_for_current_search": valid_for_current_search,
        "missing_keys": missing,
        "edit_family": edit_family,
        "edit_family_allowed": allowed_edit,
        "scene_id": candidate.get("scene_id"),
        "target_agent_id": candidate.get("target_agent_id"),
        "candidate": candidate,
    }


def parse_and_validate_candidate_file(candidate_text_path: str | Path, out_dir: str | Path) -> Dict[str, Any]:
    candidate_text_path = Path(candidate_text_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = candidate_text_path.read_text(encoding="utf-8")

    parse_error = None
    candidates: List[Dict[str, Any]] = []
    validations: List[Dict[str, Any]] = []
    try:
        obj = extract_json_candidate_object(text)
        candidates = normalize_candidate_list(obj)
        validations = [validate_candidate(c, i) for i, c in enumerate(candidates)]
    except Exception as exc:
        parse_error = str(exc)

    parsed_path = out_dir / "llm_candidate_proposals.parsed.json"
    parsed_path.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    validation_path = out_dir / "llm_candidate_proposals.validation.json"
    validation = {
        "candidate_text_path": str(candidate_text_path),
        "parse_ok": parse_error is None,
        "parse_error": parse_error,
        "num_candidates": len(candidates),
        "num_valid_for_parsing": sum(1 for v in validations if v["valid_for_parsing"]),
        "num_valid_for_current_search": sum(1 for v in validations if v["valid_for_current_search"]),
        "validations": validations,
    }
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# LLM Candidate Proposal Validation", ""]
    lines.append(f"- source: `{candidate_text_path}`")
    lines.append(f"- parse ok: `{validation['parse_ok']}`")
    lines.append(f"- parse error: `{parse_error}`")
    lines.append(f"- candidates: `{len(candidates)}`")
    lines.append(f"- valid for current search: `{validation['num_valid_for_current_search']}`")
    lines.append("")
    if validations:
        lines.append("| idx | scene_id | target_agent_id | edit_family | valid_current | missing_keys |")
        lines.append("|---:|---|---|---|---:|---|")
        for v in validations:
            lines.append(
                f"| {v['proposal_index']} | {v.get('scene_id')} | {v.get('target_agent_id')} | "
                f"{v.get('edit_family')} | {v.get('valid_for_current_search')} | {v.get('missing_keys')} |"
            )
    report_path = out_dir / "llm_candidate_proposals.validation_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "parsed_candidates": str(parsed_path),
        "validation_json": str(validation_path),
        "validation_report": str(report_path),
        "validation": validation,
    }


def build_strict_candidate_prompt(original_prompt: str) -> str:
    """Add a strict JSON-only contract to an existing candidate proposal prompt."""
    return f"""{original_prompt.strip()}


# STRICT OUTPUT CONTRACT
Return ONLY a valid JSON array. Do not use markdown fences. Do not add prose.
Every item MUST include these keys exactly:
- scene_id: string, must be copied from the evidence if available.
- target_agent_id: string or number.
- edit_family: one of ["lead_brake", "cut_in", "pedestrian_crossing", "occlusion_increase", "visibility_degradation", "traffic_light_change", "lane_marking_degradation", "reaction_delay"].
- parameter_suggestion: object or concise string.
- expected_failure_mode: string.
- physical_plausibility_reason: string.
- why_this_is_minimal: string.

For the current executable system, prioritize edit_family values that can be verified immediately:
"lead_brake", "cut_in", "pedestrian_crossing".
"""
