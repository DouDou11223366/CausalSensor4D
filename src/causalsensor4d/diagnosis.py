from __future__ import annotations

from typing import Dict, Any, Optional
from pathlib import Path

from .risk import RiskSummary


def make_diagnosis_report(
    scene_id: str,
    original_risk: RiskSummary,
    best: Optional[Dict[str, Any]],
) -> str:
    lines = []
    lines.append(f"# CausalSensor4D MVP Diagnosis Report")
    lines.append("")
    lines.append(f"## Scene")
    lines.append(f"- scene_id: `{scene_id}`")
    lines.append("")
    lines.append("## Original scene risk")
    lines.append(f"- collision: `{original_risk.collision}`")
    lines.append(f"- hard_brake: `{original_risk.hard_brake}`")
    lines.append(f"- min_distance: `{original_risk.min_distance:.3f}` m")
    lines.append(f"- min_ttc: `{original_risk.min_ttc if original_risk.min_ttc is not None else 'None'}` s")
    lines.append(f"- most_risky_agent: `{original_risk.most_risky_agent}`")
    lines.append("")

    if best is None:
        lines.append("## Counterfactual result")
        lines.append("No failure-inducing counterfactual was found under the current edit/search space.")
        lines.append("")
        lines.append("## Diagnosis")
        lines.append("当前编辑空间不足以触发失败。下一步应扩大搜索空间，例如增加更强制动、更早 cut-in、行人横穿或可见性退化变量。")
        return "\n".join(lines)

    lines.append("## Minimum failure counterfactual")
    lines.append(f"- edit_name: `{best['edit_name']}`")
    lines.append(f"- target_agent_id: `{best['target_agent_id']}`")
    lines.append(f"- parameters: `{best['parameters']}`")
    lines.append(f"- Minimum Failure Cost: `{best['cost']:.3f}`")
    lines.append(f"- collision: `{best['collision']}`")
    lines.append(f"- hard_brake: `{best['hard_brake']}`")
    lines.append(f"- min_distance: `{best['min_distance']:.3f}` m")
    lines.append(f"- min_ttc: `{best['min_ttc']:.3f}` s")
    lines.append("")
    lines.append("## Counterfactual evidence chain")
    lines.append("1. 系统在候选因果变量中搜索最小反事实编辑。")
    lines.append(f"2. 最小触发编辑为 `{best['edit_name']}`，作用对象为 `{best['target_agent_id']}`。")
    lines.append(f"3. 该编辑以代价 `{best['cost']:.3f}` 将场景风险提高到 failure threshold 以上。")
    if best["collision"]:
        lines.append("4. 反事实场景导致碰撞，因此规划输出不再安全。")
    else:
        lines.append(f"4. 反事实场景未必碰撞，但最小 TTC 降至 `{best['min_ttc']:.3f}` 秒，形成 near-miss / unsafe planning。")
    lines.append("")
    lines.append("## Diagnosis")
    lines.append("该 planner 对当前关键 agent 的轻微行为变化存在脆弱性。论文版需要进一步检查该失效来自感知、预测还是规划层。")
    return "\n".join(lines)


def save_report(text: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
