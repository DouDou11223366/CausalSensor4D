from __future__ import annotations

"""Unified AD model wrapper interface for CausalSensor4D public_release.

This module is the bridge from the current MVP rule-based planner to later
learning-based predictors/planners, VLA driving models, or closed-loop planning
benchmarks.  The rest of the CausalSensor4D pipeline only requires two things:

1. ``rollout(scene) -> DrivingScene`` so the MFC search can re-evaluate an
   edited scene under the target AD model.
2. ``run(scene) -> ADModelOutput`` so the diagnosis module can record
   perception / prediction / planning style outputs in a common schema.

A future deep model can be connected by subclassing ``BaseADModelWrapper`` and
implementing these two methods without rewriting the counterfactual search.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import os

from .schemas import DrivingScene
from .planner import SimpleFollowingPlanner, PlannerConfig
from .risk import evaluate_scene
from .lightweight_bc import LightweightBCPlanner, DEFAULT_MODEL_PATH


@dataclass
class ADModelOutput:
    model_name: str
    model_family: str
    behavior_label: str
    planned_trajectory: List[Dict[str, float]]
    predicted_agent_trajectories: Dict[str, List[Dict[str, float]]]
    perception_summary: Dict[str, Any]
    risk_summary: Dict[str, Any]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseADModelWrapper:
    """Base interface used by CausalSensor4D.

    Deep learning / large-model wrappers should follow this interface.  The
    current MVP keeps the API intentionally simple and file-free, so Windows +
    PyCharm users can run it without GPU dependencies.
    """

    model_name: str = "base"
    model_family: str = "base"

    def rollout(self, scene: DrivingScene) -> DrivingScene:
        raise NotImplementedError

    def run(self, scene: DrivingScene) -> ADModelOutput:
        planned = self.rollout(scene)
        risk = evaluate_scene(planned)
        return ADModelOutput(
            model_name=self.model_name,
            model_family=self.model_family,
            behavior_label=_behavior_from_risk(risk),
            planned_trajectory=_track_to_list(planned.ego),
            predicted_agent_trajectories={aid: _track_to_list(track) for aid, track in planned.agents.items()},
            perception_summary={
                "mode": "annotation_oracle",
                "num_agents": len(planned.agents),
                "note": "MVP uses dataset tracks as oracle perception; replace with detector outputs later.",
            },
            risk_summary=risk.__dict__,
            metadata={"wrapper_tag": "ad_model_public"},
        )


class RuleBasedADModelWrapper(BaseADModelWrapper):
    def __init__(self, model_name: str, config: PlannerConfig):
        self.model_name = model_name
        self.model_family = "rule_based_planner"
        self.planner = SimpleFollowingPlanner(config)

    def rollout(self, scene: DrivingScene) -> DrivingScene:
        return self.planner.rollout(scene)


class MockLearnedADModelWrapper(BaseADModelWrapper):
    """A lightweight stand-in for a learned predictor/planner.

    This is not claimed as a real learned model.  It is a compatibility wrapper
    that mimics the output contract of a learned AD model while remaining fully
    runnable on a laptop.  The next step is to replace this with an actual
    predictor/planner checkpoint while keeping the same interface.
    """

    def __init__(self, model_name: str = "mock_learned_predictor"):
        self.model_name = model_name
        self.model_family = "mock_learning_based"
        # Slightly smoother and more anticipatory than delayed planner.
        self.planner = SimpleFollowingPlanner(
            PlannerConfig(desired_speed=8.2, reaction_delay_steps=0, safe_ttc=4.0, safe_gap=15.0, max_decel=-3.2)
        )

    def rollout(self, scene: DrivingScene) -> DrivingScene:
        return self.planner.rollout(scene)

    def run(self, scene: DrivingScene) -> ADModelOutput:
        planned = self.rollout(scene)
        risk = evaluate_scene(planned)
        perception_summary = {
            "mode": "annotation_oracle_plus_mock_confidence",
            "num_agents": len(planned.agents),
            "mean_detection_confidence": 0.95,
            "note": "Mock learned wrapper. Replace with real perception/prediction/planning outputs later.",
        }
        return ADModelOutput(
            model_name=self.model_name,
            model_family=self.model_family,
            behavior_label=_behavior_from_risk(risk),
            planned_trajectory=_track_to_list(planned.ego),
            predicted_agent_trajectories={aid: _track_to_list(track) for aid, track in planned.agents.items()},
            perception_summary=perception_summary,
            risk_summary=risk.__dict__,
            metadata={"wrapper_tag": "ad_model_public", "is_placeholder_for_deep_model": True},
        )



class LightweightBCADModelWrapper(BaseADModelWrapper):
    """Trainable lightweight behavior-cloning planner wrapper."""
    def __init__(self, model_name: str = "lightweight_bc_planner", model_path: Optional[str] = None):
        self.model_name = model_name
        self.model_family = "lightweight_learning_based"
        model_path = model_path or os.environ.get("CS4D_LIGHTWEIGHT_BC_MODEL") or str(DEFAULT_MODEL_PATH)
        self.model_path = model_path
        self.planner = LightweightBCPlanner(model_path)
    def rollout(self, scene: DrivingScene) -> DrivingScene:
        return self.planner.rollout(scene)
    def run(self, scene: DrivingScene) -> ADModelOutput:
        planned = self.rollout(scene)
        risk = evaluate_scene(planned)
        return ADModelOutput(
            model_name=self.model_name, model_family=self.model_family, behavior_label=_behavior_from_risk(risk),
            planned_trajectory=_track_to_list(planned.ego),
            predicted_agent_trajectories={aid: _track_to_list(track) for aid, track in planned.agents.items()},
            perception_summary={"mode":"annotation_oracle", "num_agents":len(planned.agents), "note":"Lightweight BC planner; replace with deep model later."},
            risk_summary=risk.__dict__,
            metadata={"wrapper_tag":"ad_model_public", "model_path":self.model_path},
        )


def make_ad_model(name: str) -> BaseADModelWrapper:
    """Factory for AD model wrappers.

    Current supported names are lightweight.  Later, names such as
    ``mtr``, ``wayformer``, ``nuplan_planner`` or ``vla_driver`` can be added
    here without changing the MFC search code.
    """
    name = name.strip()
    if name in {"normal", "rule_normal"}:
        return RuleBasedADModelWrapper(
            "rule_normal",
            PlannerConfig(desired_speed=8.0, reaction_delay_steps=0, safe_ttc=3.0, safe_gap=12.0, max_decel=-4.0),
        )
    if name in {"delayed", "rule_delayed"}:
        return RuleBasedADModelWrapper(
            "rule_delayed",
            PlannerConfig(desired_speed=8.0, reaction_delay_steps=2, safe_ttc=3.0, safe_gap=12.0, max_decel=-4.0),
        )
    if name in {"conservative", "rule_conservative"}:
        return RuleBasedADModelWrapper(
            "rule_conservative",
            PlannerConfig(desired_speed=7.0, reaction_delay_steps=0, safe_ttc=5.0, safe_gap=18.0, max_decel=-4.5),
        )
    if name in {"aggressive", "rule_aggressive"}:
        return RuleBasedADModelWrapper(
            "rule_aggressive",
            PlannerConfig(desired_speed=9.0, reaction_delay_steps=1, safe_ttc=2.0, safe_gap=8.0, max_decel=-3.5),
        )
    if name in {"mock_learned", "mock_learned_predictor"}:
        return MockLearnedADModelWrapper("mock_learned_predictor")
    if name in {"lightweight_bc", "lightweight_bc_planner", "bc_planner"}:
        return LightweightBCADModelWrapper("lightweight_bc_planner")
    raise ValueError(f"Unknown AD model wrapper: {name}")


def list_available_ad_models() -> List[str]:
    return [
        "rule_normal",
        "rule_delayed",
        "rule_conservative",
        "rule_aggressive",
        "mock_learned_predictor",
        "lightweight_bc_planner",
    ]


def _track_to_list(track) -> List[Dict[str, float]]:
    return [
        {"t": s.t, "x": s.x, "y": s.y, "vx": s.vx, "vy": s.vy, "yaw": s.yaw}
        for s in track.states
    ]


def _behavior_from_risk(risk) -> str:
    if risk.collision:
        return "collision"
    if risk.hard_brake:
        return "hard_brake"
    min_ttc = risk.min_ttc
    if min_ttc is not None and min_ttc < 2.0:
        return "low_ttc"
    return "safe"
