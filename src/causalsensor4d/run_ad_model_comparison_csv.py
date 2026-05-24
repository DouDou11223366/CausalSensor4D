from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from .ad_model import make_ad_model, list_available_ad_models
from .data_adapters.generic_tracks_csv import load_tracks_csv, REQUIRED_COLUMNS
from .schemas import save_json
from .risk import evaluate_scene
from .causal_graph import build_causal_scene_graph, graph_to_dict, rank_causal_candidates
from .search import search_minimum_failure_cost_multi_candidate
from .report_metrics import save_report_artifacts
from .planner_comparison import summarize_planner, save_planner_comparison_artifacts
from .visualize import plot_trajectories


def is_generic_scene_csv(csv_path: Path) -> bool:
    try:
        head = pd.read_csv(csv_path, nrows=5)
    except Exception:
        return False
    return REQUIRED_COLUMNS.issubset(set(head.columns))


def run_one_csv_ad_model(csv_path: Path, out_dir: Path, model_name: str, ego_track_id: str = "ego", save_plot: bool = False) -> Dict[str, Any]:
    scene = load_tracks_csv(csv_path, scene_id=None, ego_track_id=ego_track_id)
    model = make_ad_model(model_name)
    out_scene_dir = out_dir / model.model_name / scene.scene_id
    out_scene_dir.mkdir(parents=True, exist_ok=True)

    original_output = model.run(scene)
    planned_original = model.rollout(scene)
    original_risk = evaluate_scene(planned_original)

    graph = build_causal_scene_graph(scene, t_idx=0)
    candidates = rank_causal_candidates(graph)

    if candidates:
        result = search_minimum_failure_cost_multi_candidate(scene, model, candidates)
        table = result.table
        best = result.best
    else:
        table = pd.DataFrame()
        best = None

    best_json = None
    if best is not None:
        best_json = {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()}

    save_json(original_output.to_dict(), out_scene_dir / "original_ad_model_output.json")
    save_json(graph_to_dict(graph), out_scene_dir / "causal_scene_graph.json")
    save_json(best_json, out_scene_dir / "best_counterfactual.json")
    table.to_csv(out_scene_dir / "candidate_table.csv", index=False)
    if save_plot:
        plot_trajectories(scene, model, best_json, out_scene_dir / "trajectory_plot.png")

    row: Dict[str, Any] = {
        "scene_id": scene.scene_id,
        "input_csv": str(csv_path),
        "ad_model": model.model_name,
        "planner": model.model_name,  # compatibility with existing planner comparison utilities
        "model_family": model.model_family,
        "original_behavior_label": original_output.behavior_label,
        "original_collision": original_risk.collision,
        "original_hard_brake": original_risk.hard_brake,
        "original_min_ttc": original_risk.min_ttc,
        "original_min_distance": original_risk.min_distance,
        "original_risk_score": original_risk.risk_score,
        "num_candidates": len(candidates),
        "top_candidate_agent_id": candidates[0]["agent_id"] if candidates else None,
        "best_found": best_json is not None,
        "error": None,
    }
    if best_json is not None:
        row.update({
            "best_edit_name": best_json.get("edit_name"),
            "best_target_agent_id": best_json.get("target_agent_id"),
            "best_cost": best_json.get("cost"),
            "best_failure": best_json.get("failure"),
            "best_collision": best_json.get("collision"),
            "best_hard_brake": best_json.get("hard_brake"),
            "best_min_ttc": best_json.get("min_ttc"),
            "best_min_distance": best_json.get("min_distance"),
            "best_risk_score": best_json.get("risk_score"),
        })
    return row


def _parse_models(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare AD model wrappers under CausalSensor4D MFC search.")
    parser.add_argument("--csv-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--models", default="rule_normal,rule_delayed,rule_conservative,mock_learned_predictor")
    parser.add_argument("--ego-track-id", default="ego")
    parser.add_argument("--save-plots", action="store_true")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_files = [p for p in sorted(csv_dir.glob("*.csv")) if is_generic_scene_csv(p)]
    if not csv_files:
        raise FileNotFoundError(f"No generic scene CSVs found in {csv_dir}")

    models = _parse_models(args.models)
    all_rows: List[Dict[str, Any]] = []
    model_summaries: List[Dict[str, Any]] = []

    for model_name in models:
        model = make_ad_model(model_name)
        print(f"[ADModel] Running model={model.model_name} ({model.model_family}) ...")
        model_dir = out_dir / "per_ad_model" / model.model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        rows: List[Dict[str, Any]] = []
        for csv_path in csv_files:
            print(f"  [Scene] {csv_path.name}")
            try:
                row = run_one_csv_ad_model(csv_path, out_dir / "per_scene_outputs", model.model_name, ego_track_id=args.ego_track_id, save_plot=args.save_plots)
            except Exception as exc:
                row = {"scene_id": csv_path.stem, "ad_model": model.model_name, "planner": model.model_name, "error": str(exc), "best_found": False}
                print(f"  [ERROR] {csv_path.name}: {exc}")
            rows.append(row)
            all_rows.append(row)
        summary = pd.DataFrame(rows)
        summary.to_csv(model_dir / "batch_summary.csv", index=False)
        valid = summary[summary["error"].isna()] if "error" in summary.columns else summary
        aggregate = {
            "ad_model": model.model_name,
            "model_family": model.model_family,
            "num_csv_files": len(csv_files),
            "num_valid_runs": int(len(valid)),
            "num_best_found": int(valid["best_found"].fillna(False).sum()) if "best_found" in valid else 0,
            "mean_best_cost": float(valid["best_cost"].dropna().mean()) if "best_cost" in valid and valid["best_cost"].dropna().size else None,
            "mean_original_risk_score": float(valid["original_risk_score"].dropna().mean()) if "original_risk_score" in valid else None,
        }
        aggregate = save_report_artifacts(summary, model_dir, aggregate)
        save_json(aggregate, model_dir / "batch_aggregate.json")
        model_summaries.append(summarize_planner(summary, model.model_name))

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(out_dir / "all_ad_model_scene_results.csv", index=False)
    save_planner_comparison_artifacts(all_df, out_dir, model_summaries)

    # Rename copies with AD-model terminology for readability.
    (out_dir / "ad_model_comparison_report.md").write_text((out_dir / "planner_comparison_report.md").read_text(encoding="utf-8").replace("Planner", "AD Model").replace("planner", "AD model"), encoding="utf-8")
    (out_dir / "ad_model_comparison_summary.csv").write_text((out_dir / "planner_comparison_summary.csv").read_text(encoding="utf-8"), encoding="utf-8")
    (out_dir / "ad_model_robustness_ranking.csv").write_text((out_dir / "planner_robustness_ranking.csv").read_text(encoding="utf-8"), encoding="utf-8")

    save_json({"available_wrappers": list_available_ad_models(), "used_models": models}, out_dir / "ad_model_registry.json")

    print("CausalSensor4D AD-model wrapper comparison finished.")
    print(f"CSV folder: {csv_dir}")
    print(f"AD models: {', '.join(models)}")
    print(f"All results: {out_dir / 'all_ad_model_scene_results.csv'}")
    print(f"AD-model report: {out_dir / 'ad_model_comparison_report.md'}")


if __name__ == "__main__":
    main()
