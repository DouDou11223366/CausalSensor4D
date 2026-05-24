from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import math
import random

import pandas as pd

from .causal_graph import build_causal_scene_graph, rank_causal_candidates
from .data_adapters.generic_tracks_csv import load_tracks_csv
from .report_metrics import infer_failure_type
from .planner import SimpleFollowingPlanner
from .risk import center_distance, evaluate_scene
from .longitudinal_geometry import relation_in_ego_frame, is_adjacent_lane_interaction
from .schemas import DrivingScene, save_json
from .search import (
    SearchConfig,
    SearchResult,
    default_search_config,
    search_minimum_failure_cost,
    search_minimum_failure_cost_multi_candidate,
)


DEFAULT_BASELINE_CENSORED_MFC = 2.0
RUNTIME_RANK_STRATEGY = "budget_ranked"
RUNTIME_RANK_VERSION = "public_release"


@dataclass
class BaselineMethod:
    name: str
    description: str


BASELINE_METHODS: List[BaselineMethod] = [
    BaselineMethod(
        "causal_guided",
        "Ours-strict: use causal scene graph relation labels to select candidate agents and relation-specific edit spaces.",
    ),
    BaselineMethod(
        "causal_hybrid",
        "Ours-final public_release: causal relation candidates plus distance fallback, with runtime budget-aware distance/cost ordering over physically admissible edits.",
    ),
    BaselineMethod(
        "distance_all",
        "No causal graph: rank agents only by current Euclidean distance and search all edit primitives.",
    ),
    BaselineMethod(
        "random_budget",
        "Random search: randomly sample candidate counterfactuals from all agents and edit primitives under a fixed budget.",
    ),
    BaselineMethod(
        "lead_brake_only",
        "Single-edit ablation: only longitudinal lead braking is allowed.",
    ),
    BaselineMethod(
        "cut_in_only",
        "Single-edit ablation: only lateral cut-in is allowed.",
    ),
    BaselineMethod(
        "pedestrian_only",
        "Single-edit ablation: only pedestrian crossing is allowed.",
    ),
]


def baseline_method_names() -> List[str]:
    return [m.name for m in BASELINE_METHODS]


def run_baseline_method(
    scene: DrivingScene,
    planner: SimpleFollowingPlanner,
    method: str,
    config: Optional[SearchConfig] = None,
    random_budget: int = 36,
    seed: int = 13,
) -> SearchResult:
    """Run one baseline / ablation method on one scene.

    public_release goal:
    - causal_guided shows the proposed relation-aware search.
    - distance_all removes the causal relation labels and uses nearest-agent ranking.
    - random_budget controls for brute-force chance discovery under a fixed sampling budget.
    - single-edit ablations test whether multiple edit primitives are necessary.
    """
    cfg = config or default_search_config()

    graph = build_causal_scene_graph(scene, t_idx=0)
    causal_candidates = rank_causal_candidates(graph)

    if method == "causal_guided":
        result = search_minimum_failure_cost_multi_candidate(scene, planner, causal_candidates, cfg)
        if not result.table.empty:
            ranked = _sort_candidate_table_for_budget(result.table, method="causal_guided")
            return _select_best_from_table(ranked)
        return SearchResult(best=None, table=_empty_runtime_rank_table(method="causal_guided"))

    if method == "causal_hybrid":
        return _causal_hybrid_search(scene, planner, cfg, causal_candidates)

    if method == "distance_all":
        return _distance_all_search(scene, planner, cfg)

    if method == "random_budget":
        return _random_budget_search(scene, planner, cfg, random_budget=random_budget, seed=seed)

    if method == "lead_brake_only":
        return _single_edit_search(scene, planner, cfg, allowed_edit="lead_brake", method_name=method)

    if method == "cut_in_only":
        return _single_edit_search(scene, planner, cfg, allowed_edit="cut_in", method_name=method)

    if method == "pedestrian_only":
        return _single_edit_search(scene, planner, cfg, allowed_edit="pedestrian_crossing", method_name=method)

    raise ValueError(f"Unknown baseline method: {method}")


def _admissible_edits_for_agent(scene: DrivingScene, agent_id: str) -> List[str]:
    """Return physically meaningful edit primitives for this agent.

    This simple validity filter prevents invalid baselines such as applying a
    pedestrian-crossing edit to a vehicle. It is intentionally weaker than the
    causal graph: it uses only object type and rough geometry, not risk relation
    labels or causal priorities.
    """
    if agent_id not in scene.agents:
        return []
    track = scene.agents[agent_id]
    st = track.state_at_index(0)
    ego = scene.ego.state_at_index(0)
    agent_type = (track.agent_type or "vehicle").lower()

    if "ped" in agent_type or "person" in agent_type:
        return ["pedestrian_crossing"]

    edits: List[str] = []
    rel = relation_in_ego_frame(ego, st)

    # public_release: decide longitudinal lead/adjacent-lane candidates in the ego
    # heading frame.  The previous global-x test missed lead vehicles whenever
    # the AV2 scene was not aligned with the x-axis.
    if rel.same_lane and rel.longitudinal > 0.0:
        edits.append("lead_brake")
    if is_adjacent_lane_interaction(ego, st, min_longitudinal=-10.0, max_longitudinal=55.0):
        edits.append("cut_in")
    return edits


def _rank_agents_for_edit(scene: DrivingScene, allowed_edit: str, max_agents: Optional[int] = None) -> List[str]:
    """Rank admissible agents for one edit family using pre-outcome geometry.

    public_release can expose many more lane-consistent lead vehicles because it uses the
    ego-heading frame. To keep full 120-scene experiments tractable, we evaluate
    the closest admissible agents first and cap by ``max_agents`` when provided.
    """
    ego = scene.ego.state_at_index(0)
    rows: List[tuple[float, str]] = []
    for agent_id, track in scene.agents.items():
        if allowed_edit not in _admissible_edits_for_agent(scene, agent_id):
            continue
        st = track.state_at_index(0)
        rel = relation_in_ego_frame(ego, st)
        if allowed_edit == "lead_brake":
            key = max(0.0, rel.gap)
        elif allowed_edit == "cut_in":
            key = abs(rel.lateral) + 0.02 * max(0.0, abs(rel.longitudinal))
        else:
            key = center_distance(ego, st)
        rows.append((float(key), agent_id))
    rows.sort(key=lambda x: x[0])
    ids = [agent_id for _, agent_id in rows]
    if max_agents is not None and max_agents > 0:
        ids = ids[:max_agents]
    return ids


def _causal_hybrid_search(
    scene: DrivingScene,
    planner: SimpleFollowingPlanner,
    cfg: SearchConfig,
    causal_candidates: List[Dict[str, Any]],
) -> SearchResult:
    """Causal-prioritized hybrid search for the hybrid.

    public_release showed that strict causal routing can be too selective on AV2: a
    distance-only exhaustive baseline found more failures because it searched a
    broader admissible edit space. This hybrid keeps causal graph priors but adds
    a distance-based fallback. For every retained agent it searches only
    physically admissible edit families.

    Interpretation: this is the recommended final search variant for report runs,
    while ``causal_guided`` remains the strict-routing ablation.
    """
    ego0 = scene.ego.state_at_index(0)
    by_agent: Dict[str, Dict[str, Any]] = {}

    # 1) Causal candidates first. Preserve relation metadata and recommended edit.
    for rank, cand in enumerate(causal_candidates):
        agent_id = cand.get("agent_id")
        if not agent_id or agent_id not in scene.agents:
            continue
        admissible = _admissible_edits_for_agent(scene, agent_id)
        rec = cand.get("recommended_edit")
        edit_set = []
        if rec and rec != "none" and rec in admissible:
            edit_set.append(rec)
        # Add admissible edits as a controlled fallback for the same causal agent.
        for e in admissible:
            if e not in edit_set:
                edit_set.append(e)
        if not edit_set:
            continue
        st = scene.agents[agent_id].state_at_index(0)
        by_agent[agent_id] = {
            "agent_id": agent_id,
            "relation": cand.get("relation", "causal"),
            "distance": center_distance(ego0, st),
            "relation_priority": cand.get("relation_priority", 0),
            "causal_rank": rank,
            "source": "causal_graph",
            "admissible_edits": edit_set,
        }

    # 2) Distance fallback: fill remaining slots with nearby physically valid agents.
    distance_candidates: List[Dict[str, Any]] = []
    for agent_id, track in scene.agents.items():
        admissible = _admissible_edits_for_agent(scene, agent_id)
        if not admissible:
            continue
        st = track.state_at_index(0)
        distance_candidates.append(
            {
                "agent_id": agent_id,
                "relation": "distance_fallback",
                "distance": center_distance(ego0, st),
                "relation_priority": 999,
                "causal_rank": 999,
                "source": "distance_fallback",
                "admissible_edits": admissible,
            }
        )
    distance_candidates = sorted(distance_candidates, key=lambda x: x["distance"])
    for cand in distance_candidates:
        if len(by_agent) >= cfg.max_candidates:
            break
        if cand["agent_id"] not in by_agent:
            by_agent[cand["agent_id"]] = cand

    # 3) Search retained candidates. Causal candidates are evaluated first; fallback
    # candidates keep scene coverage comparable to distance_all.
    selected = sorted(by_agent.values(), key=lambda x: (x.get("source") != "causal_graph", x.get("causal_rank", 999), x.get("distance", 1e9)))
    tables: List[pd.DataFrame] = []
    for cand in selected[: cfg.max_candidates]:
        for edit_type in cand["admissible_edits"]:
            result = search_minimum_failure_cost(scene, planner, cand["agent_id"], cfg, allowed_edit=edit_type)
            if not result.table.empty:
                table = result.table.copy()
                table["hybrid_candidate_relation"] = cand.get("relation")
                table["hybrid_candidate_source"] = cand.get("source")
                table["hybrid_distance"] = cand.get("distance")
                table["hybrid_relation_priority"] = cand.get("relation_priority")
                table["hybrid_causal_rank"] = cand.get("causal_rank")
                table["hybrid_allowed_edit"] = edit_type
                tables.append(table)
    if not tables:
        return SearchResult(best=None, table=_empty_runtime_rank_table(method="causal_hybrid"))
    merged = pd.concat(tables, ignore_index=True)
    merged = _sort_candidate_table_for_budget(merged, method="causal_hybrid")
    return _select_best_from_table(merged)


def _distance_all_search(scene: DrivingScene, planner: SimpleFollowingPlanner, cfg: SearchConfig) -> SearchResult:
    ego0 = scene.ego.state_at_index(0)
    candidates: List[Dict[str, Any]] = []
    for agent_id, track in scene.agents.items():
        st = track.state_at_index(0)
        admissible = _admissible_edits_for_agent(scene, agent_id)
        if not admissible:
            continue
        candidates.append(
            {
                "agent_id": agent_id,
                "relation": "distance_only",
                "distance": center_distance(ego0, st),
                "relation_priority": 0,
                "admissible_edits": admissible,
            }
        )
    candidates = sorted(candidates, key=lambda x: x["distance"])[: cfg.max_candidates]
    tables: List[pd.DataFrame] = []
    for cand in candidates:
        for edit_type in cand["admissible_edits"]:
            result = search_minimum_failure_cost(scene, planner, cand["agent_id"], cfg, allowed_edit=edit_type)
            if not result.table.empty:
                table = result.table.copy()
                table["baseline_candidate_relation"] = cand.get("relation")
                table["baseline_distance"] = cand.get("distance")
                table["baseline_allowed_edit"] = edit_type
                tables.append(table)
    if not tables:
        return SearchResult(best=None, table=_empty_runtime_rank_table(method="distance_all"))
    merged = pd.concat(tables, ignore_index=True)
    merged = _sort_candidate_table_for_budget(merged, method="distance_all")
    return _select_best_from_table(merged)


def _single_edit_search(scene: DrivingScene, planner: SimpleFollowingPlanner, cfg: SearchConfig, allowed_edit: str, method_name: str = "single_edit") -> SearchResult:
    """Search a single edit primitive over physically admissible agents."""
    tables: List[pd.DataFrame] = []
    ranked_agents = _rank_agents_for_edit(scene, allowed_edit, max_agents=cfg.max_candidates_per_edit)
    for agent_id in ranked_agents:
        result = search_minimum_failure_cost(scene, planner, agent_id, cfg, allowed_edit=allowed_edit)
        if not result.table.empty:
            table = result.table.copy()
            table["single_edit_allowed"] = allowed_edit
            tables.append(table)
    if not tables:
        return SearchResult(best=None, table=_empty_runtime_rank_table(method=method_name))
    merged = pd.concat(tables, ignore_index=True)
    merged = _sort_candidate_table_for_budget(merged, method=method_name)
    return _select_best_from_table(merged)


def _random_budget_search(
    scene: DrivingScene,
    planner: SimpleFollowingPlanner,
    cfg: SearchConfig,
    random_budget: int,
    seed: int,
) -> SearchResult:
    """Randomly sample from the union of physically admissible candidates.

    To keep the comparison fair, this baseline does not exhaustively search every
    possible candidate. It first builds the valid candidate table available to an
    unstructured searcher, then samples a fixed number of rows.
    """
    all_tables: List[pd.DataFrame] = []
    # public_release: build a tractable pre-outcome candidate universe by taking the
    # nearest admissible agents for each edit family. The random baseline still
    # samples rows uniformly from this unstructured universe; it does not use
    # causal labels or outcome information.
    seen_pairs = set()
    for edit_type in ["lead_brake", "cut_in", "pedestrian_crossing"]:
        for agent_id in _rank_agents_for_edit(scene, edit_type, max_agents=cfg.max_candidates_per_edit):
            key = (agent_id, edit_type)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            result = search_minimum_failure_cost(scene, planner, agent_id, cfg, allowed_edit=edit_type)
            if not result.table.empty:
                table = result.table.copy()
                table["random_allowed_edit"] = edit_type
                all_tables.append(table)
    if not all_tables:
        return SearchResult(best=None, table=_empty_runtime_rank_table(method="random_budget", strategy="random_budget_sample", sort_keys="seeded_random_permutation"))
    full = pd.concat(all_tables, ignore_index=True)
    rng = random.Random(seed + _stable_scene_hash(scene.scene_id))
    if len(full) > random_budget:
        sampled_idx = rng.sample(list(full.index), random_budget)
        sampled = full.loc[sampled_idx].copy().reset_index(drop=True)
    else:
        sampled = full.copy().reset_index(drop=True)
    sampled["random_budget"] = random_budget
    sampled.insert(0, "candidate_eval_order", range(1, len(sampled) + 1))
    sampled["runtime_rank_strategy"] = "random_budget_sample"
    sampled["runtime_rank_version"] = RUNTIME_RANK_VERSION
    sampled["runtime_rank_method"] = "random_budget"
    sampled["runtime_rank_sort_keys"] = "seeded_random_permutation"
    sampled["runtime_rank_uses_outcome"] = False
    return _select_best_from_table(sampled)


def _empty_runtime_rank_table(method: str, strategy: Optional[str] = None, sort_keys: str = "empty_candidate_table") -> pd.DataFrame:
    """Return a header-only candidate table with runtime-rank metadata.

    public_release keeps empty candidate tables auditable too.  Header-only CSVs are
    important because some scenes have no admissible candidate for a single-edit
    ablation.  Without these columns, the audit cannot distinguish a fresh empty
    runtime-ranked result from an old unranked result.
    """
    strat = strategy or RUNTIME_RANK_STRATEGY
    return pd.DataFrame(columns=[
        "candidate_eval_order",
        "runtime_rank_strategy",
        "runtime_rank_version",
        "runtime_rank_method",
        "runtime_rank_sort_keys",
        "runtime_rank_uses_outcome",
        "failure",
        "cost",
        "edit_name",
        "target_agent_id",
        "min_ttc",
        "risk_score",
        "collision",
        "hard_brake",
    ]).assign(
        runtime_rank_strategy=pd.Series(dtype="object"),
        runtime_rank_version=pd.Series(dtype="object"),
        runtime_rank_method=pd.Series(dtype="object"),
        runtime_rank_sort_keys=pd.Series(dtype="object"),
        runtime_rank_uses_outcome=pd.Series(dtype="bool"),
    )


def _ensure_runtime_rank_metadata(table: pd.DataFrame, method: str) -> pd.DataFrame:
    """Ensure saved candidate_table.csv always contains public_release rank columns."""
    if table is None or table.empty:
        if method == "random_budget":
            return _empty_runtime_rank_table(method=method, strategy="random_budget_sample", sort_keys="seeded_random_permutation")
        return _empty_runtime_rank_table(method=method)
    df = table.copy()
    if "candidate_eval_order" not in df.columns:
        df.insert(0, "candidate_eval_order", range(1, len(df) + 1))
    if "runtime_rank_strategy" not in df.columns:
        df["runtime_rank_strategy"] = "random_budget_sample" if method == "random_budget" else RUNTIME_RANK_STRATEGY
    if "runtime_rank_version" not in df.columns:
        df["runtime_rank_version"] = RUNTIME_RANK_VERSION
    if "runtime_rank_method" not in df.columns:
        df["runtime_rank_method"] = method
    if "runtime_rank_sort_keys" not in df.columns:
        df["runtime_rank_sort_keys"] = "existing_runtime_order"
    if "runtime_rank_uses_outcome" not in df.columns:
        df["runtime_rank_uses_outcome"] = False
    return df

def _edit_priority_value(edit_name: Any) -> int:
    return {"cut_in": 0, "pedestrian_crossing": 1, "lead_brake": 2}.get(str(edit_name or "").strip(), 9)


def _sort_candidate_table_for_budget(table: pd.DataFrame, method: str) -> pd.DataFrame:
    """Sort candidate rows using only pre-outcome metadata.

    public_release keeps the runtime candidate_table order explicitly auditable.  The sort
    never uses outcome columns such as failure, min_ttc, risk_score, collision,
    or hard_brake.  It only uses intervention metadata (cost/edit family),
    geometric proximity, and causal-source metadata that are available before
    evaluating a candidate.
    """
    if table.empty:
        return table
    df = table.copy()
    df["_eval_cost"] = pd.to_numeric(df["cost"], errors="coerce").fillna(1e9) if "cost" in df.columns else 1e9
    df["_eval_edit_priority"] = df["edit_name"].map(_edit_priority_value) if "edit_name" in df.columns else 9

    sort_keys: List[str]
    ascending: List[bool]
    if method == "causal_hybrid":
        source = df.get("hybrid_candidate_source", pd.Series(["unknown"] * len(df), index=df.index)).fillna("unknown")
        df["_eval_source_priority"] = source.map(lambda x: 0 if str(x) == "causal_graph" else 1)
        df["_eval_distance"] = pd.to_numeric(df.get("hybrid_distance", pd.Series([1e9] * len(df), index=df.index)), errors="coerce").fillna(1e9)
        sort_keys = ["_eval_source_priority", "_eval_distance", "_eval_cost", "_eval_edit_priority"]
        ascending = [True, True, True, True]
    elif method == "distance_all":
        df["_eval_distance"] = pd.to_numeric(df.get("baseline_distance", pd.Series([1e9] * len(df), index=df.index)), errors="coerce").fillna(1e9)
        sort_keys = ["_eval_distance", "_eval_cost", "_eval_edit_priority"]
        ascending = [True, True, True]
    else:
        sort_keys = ["_eval_cost", "_eval_edit_priority"]
        ascending = [True, True]

    df = df.sort_values(sort_keys, ascending=ascending).reset_index(drop=True)
    # Runtime audit markers. These columns allow public_release audit to distinguish a
    # freshly re-run ranked baseline from old public_release/public_release candidate tables that
    # were only re-ranked post hoc by the audit script.
    df.insert(0, "candidate_eval_order", range(1, len(df) + 1))
    df["runtime_rank_strategy"] = RUNTIME_RANK_STRATEGY
    df["runtime_rank_version"] = RUNTIME_RANK_VERSION
    df["runtime_rank_method"] = method
    df["runtime_rank_sort_keys"] = ",".join(sort_keys)
    df["runtime_rank_uses_outcome"] = False
    return df


def _stable_scene_hash(scene_id: str) -> int:
    value = 0
    for ch in scene_id:
        value = (value * 131 + ord(ch)) % 1000003
    return value


def _select_best_from_table(table: pd.DataFrame) -> SearchResult:
    best = None
    if len(table) > 0 and "failure" in table.columns:
        failures = table[table["failure"]].copy()
        if len(failures) > 0:
            failures = failures.sort_values(["cost", "min_ttc", "risk_score"], ascending=[True, True, False])
            best = failures.iloc[0].to_dict()
    return SearchResult(best=best, table=table)


def run_one_baseline_csv(
    csv_path: Path,
    out_dir: Path,
    planner: SimpleFollowingPlanner,
    planner_kind: str,
    method: str,
    ego_track_id: str = "ego",
    random_budget: int = 36,
    seed: int = 13,
) -> Dict[str, Any]:
    scene = load_tracks_csv(csv_path, scene_id=None, ego_track_id=ego_track_id)
    method_scene_dir = out_dir / method / scene.scene_id
    method_scene_dir.mkdir(parents=True, exist_ok=True)

    planned_original = planner.rollout(scene)
    original_risk = evaluate_scene(planned_original)
    result = run_baseline_method(scene, planner, method, random_budget=random_budget, seed=seed)

    table = _ensure_runtime_rank_metadata(result.table, method)
    table.to_csv(method_scene_dir / "candidate_table.csv", index=False)

    best_json = None
    if result.best is not None:
        best_json = {k: (v.item() if hasattr(v, "item") else v) for k, v in result.best.items()}
    save_json(best_json, method_scene_dir / "best_counterfactual.json")

    graph = build_causal_scene_graph(scene, t_idx=0)
    candidates = rank_causal_candidates(graph)
    top_candidate_agent_id = candidates[0]["agent_id"] if candidates else None

    return _row_from_best(
        scene=scene,
        csv_path=csv_path,
        planner_kind=planner_kind,
        method=method,
        original_risk=original_risk,
        top_candidate_agent_id=top_candidate_agent_id,
        best_json=best_json,
        output_dir=method_scene_dir,
    )


def _row_from_best(
    scene: DrivingScene,
    csv_path: Path,
    planner_kind: str,
    method: str,
    original_risk: Any,
    top_candidate_agent_id: Optional[str],
    best_json: Optional[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    best_found = best_json is not None
    return {
        "scene_id": scene.scene_id,
        "input_csv": str(csv_path),
        "planner": planner_kind,
        "method": method,
        "num_agents": len(scene.agents),
        "num_steps": scene.num_steps(),
        "top_candidate_agent_id": top_candidate_agent_id,
        "original_collision": original_risk.collision,
        "original_hard_brake": original_risk.hard_brake,
        "original_min_ttc": original_risk.min_ttc if original_risk.min_ttc is not None else 999.0,
        "original_risk_score": original_risk.risk_score,
        "best_found": best_found,
        "best_edit_name": None if not best_found else best_json.get("edit_name"),
        "best_target_agent_id": None if not best_found else best_json.get("target_agent_id"),
        "best_cost": None if not best_found else best_json.get("cost"),
        "best_collision": None if not best_found else best_json.get("collision"),
        "best_hard_brake": None if not best_found else best_json.get("hard_brake"),
        "best_min_ttc": None if not best_found else best_json.get("min_ttc"),
        "best_risk_score": None if not best_found else best_json.get("risk_score"),
        "failure_type": (
            "not_found"
            if not best_found
            else infer_failure_type(
                best_json.get("edit_name"), best_json.get("collision"), best_json.get("hard_brake"), best_json.get("min_ttc")
            )
        ),
        "output_dir": str(output_dir),
    }


def summarize_method(df: pd.DataFrame, method: str, censored_mfc: float = DEFAULT_BASELINE_CENSORED_MFC) -> Dict[str, Any]:
    valid = df[df["method"] == method].copy()
    if "error" in valid.columns:
        valid = valid[valid["error"].isna()]
    num_valid = int(len(valid))
    if num_valid == 0:
        return {"method": method, "num_valid_runs": 0}

    found = valid["best_found"].fillna(False).astype(bool)
    costs = pd.to_numeric(valid["best_cost"], errors="coerce")
    found_costs = costs[found].dropna()
    censored_values = [float(costs.loc[idx]) if bool(found.loc[idx]) and pd.notna(costs.loc[idx]) else censored_mfc for idx in valid.index]

    orig_risk = pd.to_numeric(valid["original_risk_score"], errors="coerce")
    best_risk = pd.to_numeric(valid["best_risk_score"], errors="coerce")
    risk_inc = (best_risk - orig_risk).dropna()

    return {
        "method": method,
        "num_valid_runs": num_valid,
        "num_best_found": int(found.sum()),
        "attack_success_rate": float(found.mean()),
        "mean_mfc_success_only": float(found_costs.mean()) if len(found_costs) else None,
        "median_mfc_success_only": float(found_costs.median()) if len(found_costs) else None,
        "min_mfc_success_only": float(found_costs.min()) if len(found_costs) else None,
        "max_mfc_success_only": float(found_costs.max()) if len(found_costs) else None,
        "mean_censored_mfc": float(pd.Series(censored_values).mean()),
        "median_censored_mfc": float(pd.Series(censored_values).median()),
        "mean_risk_increase": float(risk_inc.mean()) if len(risk_inc) else None,
        "collision_rate": float(valid["best_collision"].fillna(False).astype(bool).mean()),
        "hard_brake_rate": float(valid["best_hard_brake"].fillna(False).astype(bool).mean()),
        "num_edit_types_found": int(valid["best_edit_name"].dropna().nunique()),
    }


def build_method_edit_type_table(all_rows: pd.DataFrame) -> pd.DataFrame:
    if all_rows.empty or "best_found" not in all_rows.columns:
        return pd.DataFrame()
    df = all_rows[all_rows["best_found"].fillna(False).astype(bool)].copy()
    if df.empty:
        return pd.DataFrame()

    def _rate(s: pd.Series) -> float:
        return float(s.fillna(False).astype(bool).mean())

    return (
        df.groupby(["method", "best_edit_name"], dropna=False)
        .agg(
            num_scenes=("scene_id", "count"),
            mean_mfc=("best_cost", "mean"),
            median_mfc=("best_cost", "median"),
            min_mfc=("best_cost", "min"),
            max_mfc=("best_cost", "max"),
            collision_rate=("best_collision", _rate),
            hard_brake_rate=("best_hard_brake", _rate),
            mean_best_risk=("best_risk_score", "mean"),
        )
        .reset_index()
        .sort_values(["method", "mean_mfc", "best_edit_name"])
    )


def build_method_scene_matrix(all_rows: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "scene_id",
        "method",
        "best_edit_name",
        "best_cost",
        "best_collision",
        "best_hard_brake",
        "best_min_ttc",
        "failure_type",
    ]
    available = [c for c in cols if c in all_rows.columns]
    return all_rows[available].sort_values(["scene_id", "method"]) if available else pd.DataFrame()


def build_method_ranking(method_summary: pd.DataFrame) -> pd.DataFrame:
    if method_summary.empty:
        return pd.DataFrame()
    df = method_summary.copy()
    # A stronger search baseline has higher attack success and lower censored MFC.
    df["effectiveness_key"] = df["attack_success_rate"].fillna(0.0) * 100.0 - df["mean_censored_mfc"].fillna(0.0)
    df = df.sort_values(["effectiveness_key", "attack_success_rate"], ascending=[False, False]).reset_index(drop=True)
    df.insert(0, "effectiveness_rank", range(1, len(df) + 1))
    return df


def make_baseline_report(method_summary: pd.DataFrame, edit_table: pd.DataFrame, scene_matrix: pd.DataFrame, ranking: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("# CausalSensor4D public_release Baseline and Ablation Report")
    lines.append("")
    lines.append("## Purpose")
    lines.append("This experiment checks whether the proposed causal-guided counterfactual search is better than unstructured search and single-edit ablations.")
    lines.append("")
    lines.append("## Methods")
    for m in BASELINE_METHODS:
        lines.append(f"- `{m.name}`: {m.description}")
    lines.append("")
    lines.append("## Method effectiveness ranking")
    if ranking.empty:
        lines.append("No ranking available.")
    else:
        cols = ["effectiveness_rank", "method", "attack_success_rate", "mean_censored_mfc", "mean_mfc_success_only", "num_best_found", "num_valid_runs", "num_edit_types_found"]
        lines.append(ranking[[c for c in cols if c in ranking.columns]].to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append("## Method-level metrics")
    if method_summary.empty:
        lines.append("No summary available.")
    else:
        display_cols = [
            "method",
            "attack_success_rate",
            "mean_censored_mfc",
            "mean_mfc_success_only",
            "mean_risk_increase",
            "collision_rate",
            "hard_brake_rate",
            "num_edit_types_found",
        ]
        lines.append(method_summary[[c for c in display_cols if c in method_summary.columns]].to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append("## Method × edit-type breakdown")
    if edit_table.empty:
        lines.append("No edit-type breakdown available.")
    else:
        lines.append(edit_table.to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append("## Scene-level matrix")
    if scene_matrix.empty:
        lines.append("No scene-level matrix available.")
    else:
        lines.append(scene_matrix.to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append("## Report usage")
    lines.append("This table can be used as the benchmark-stage ablation/baseline experiment. It shows whether causal relation routing and a multi-edit intervention library improve failure discovery under the same scene set and planner.")
    return "\n".join(lines)


def save_baseline_artifacts(all_rows: pd.DataFrame, out_dir: Path, method_names: List[str]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # If all scenes failed before producing baseline columns, still write the
    # error table and a readable report instead of crashing with KeyError.
    if all_rows.empty or "best_found" not in all_rows.columns:
        all_rows.to_csv(out_dir / "all_baseline_scene_results.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "baseline_method_summary.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "baseline_edit_type_table.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "baseline_scene_matrix.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "baseline_effectiveness_ranking.csv", index=False)
        msg = (
            "# CausalSensor4D public_release Baseline and Ablation Report\n\n"
            "No valid scene-level baseline rows were produced. This usually means "
            "CSV_DIR pointed to metadata CSVs such as selected_scenes.csv rather "
            "than generic_tracks_csv scene files, or every scene failed during loading.\n"
        )
        (out_dir / "baseline_comparison_report.md").write_text(msg, encoding="utf-8")
        payload = {
            "error": "no_valid_baseline_rows",
            "num_rows": int(len(all_rows)),
            "columns": list(all_rows.columns),
        }
        (out_dir / "baseline_comparison_metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return

    summaries = [summarize_method(all_rows, m) for m in method_names]
    summary_df = pd.DataFrame(summaries)
    edit_table = build_method_edit_type_table(all_rows)
    scene_matrix = build_method_scene_matrix(all_rows)
    ranking = build_method_ranking(summary_df)

    all_rows.to_csv(out_dir / "all_baseline_scene_results.csv", index=False)
    summary_df.to_csv(out_dir / "baseline_method_summary.csv", index=False)
    edit_table.to_csv(out_dir / "baseline_edit_type_table.csv", index=False)
    scene_matrix.to_csv(out_dir / "baseline_scene_matrix.csv", index=False)
    ranking.to_csv(out_dir / "baseline_effectiveness_ranking.csv", index=False)
    (out_dir / "baseline_comparison_report.md").write_text(
        make_baseline_report(summary_df, edit_table, scene_matrix, ranking), encoding="utf-8"
    )
    payload = {
        "censored_mfc_upper_bound": DEFAULT_BASELINE_CENSORED_MFC,
        "methods": summaries,
        "method_descriptions": {m.name: m.description for m in BASELINE_METHODS},
    }
    (out_dir / "baseline_comparison_metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
