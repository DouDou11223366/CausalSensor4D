from __future__ import annotations

"""CausalSensor4D public_release: safety-first clean mining pipeline.

This entrypoint is designed for the next research stage after the clean LLM
prototype: obtain more strict clean safe-to-failure scenes from AV2.

Pipeline:
    AV2 parquet root or existing generic_tracks_csv
      -> generic CSV conversion if needed
      -> original-safety-first mining
      -> interaction filtering on original-safe scenes
      -> optional MFC run on selected clean interaction scenes

The key difference from earlier AV2 filtering is the order:
    original safety first, interaction second.
"""

from pathlib import Path
from typing import Dict, Optional
import argparse
import json

from .data_adapters.av2_motion_forecasting import batch_convert_av2_to_generic_csv
from .clean_scene_miner import mine_clean_interaction_scenes
from .av2_safety_filter import SafetyFilterConfig
from .av2_scene_filter import FilterConfig
from .run_av2_public_dataset import run_batch_csv_programmatically


def _count_csvs(path: Path) -> int:
    if not path.exists():
        return 0
    return len([p for p in path.glob('*.csv') if p.is_file()])


def run_pipeline(
    out_root: str | Path,
    av2_root: Optional[str | Path] = None,
    existing_csv_dir: Optional[str | Path] = None,
    limit: int = 500,
    max_tracks: int = 32,
    planner: str = 'delayed',
    min_ttc_safe: float = 2.0,
    max_per_type: int = 80,
    top_k_total: int = 120,
    min_score: float = 1.0,
    run_mfc: bool = True,
) -> Dict[str, object]:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if existing_csv_dir:
        csv_dir = Path(existing_csv_dir)
        conversion_summary = {
            'mode': 'existing_csv_dir',
            'generic_tracks_csv': str(csv_dir),
            'num_csv_files': _count_csvs(csv_dir),
        }
    else:
        if not av2_root:
            raise ValueError('Either av2_root or existing_csv_dir must be provided.')
        csv_dir = out_root / 'generic_tracks_csv'
        conversion_limit = None if limit == 0 else limit
        conversion_summary = batch_convert_av2_to_generic_csv(
            av2_root=av2_root,
            out_dir=csv_dir,
            limit=conversion_limit,
            max_tracks=max_tracks,
        )
        conversion_summary['mode'] = 'convert_from_av2_root'
        conversion_summary['limit_requested'] = limit

    safety_cfg = SafetyFilterConfig(
        min_ttc_safe=min_ttc_safe,
        require_no_collision=True,
        require_no_hard_brake=True,
    )
    interaction_cfg = FilterConfig(
        max_per_type=max_per_type,
        top_k_total=None if top_k_total == 0 else top_k_total,
        min_score=min_score,
    )

    miner_out = out_root / 'clean_scene_miner'
    miner_summary = mine_clean_interaction_scenes(
        csv_dir=csv_dir,
        out_dir=miner_out,
        planner_kind=planner,
        safety_cfg=safety_cfg,
        interaction_cfg=interaction_cfg,
    )

    selected_clean_csv_dir = Path(str(miner_summary.get('selected_clean_csv_dir', miner_out / 'interaction_on_safe_scenes' / 'selected_csv')))
    selected_clean_count = _count_csvs(selected_clean_csv_dir)

    mfc_summary: Dict[str, object] = {
        'run_mfc_requested': run_mfc,
        'selected_clean_csv_dir': str(selected_clean_csv_dir),
        'selected_clean_csv_count': selected_clean_count,
        'mfc_output_dir': str(out_root / 'mfc_run_clean_mined'),
        'status': 'skipped',
    }
    if run_mfc:
        if selected_clean_count > 0:
            run_batch_csv_programmatically(selected_clean_csv_dir, out_root / 'mfc_run_clean_mined', planner=planner)
            mfc_summary['status'] = 'finished'
        else:
            mfc_summary['status'] = 'no_selected_clean_scenes'

    summary = {
        'version': 'public_release',
        'purpose': 'safety-first clean scene mining for strict safe-to-failure experiments',
        'conversion': conversion_summary,
        'miner_summary_path': str(miner_out / 'clean_scene_miner_summary.json'),
        'miner_report_path': str(miner_out / 'clean_scene_miner_report.md'),
        'selected_clean_csv_dir': str(selected_clean_csv_dir),
        'selected_clean_csv_count': selected_clean_count,
        'mfc': mfc_summary,
        'recommended_next': 'Use selected_clean_csv_dir as CSV_DIR for the clean LLM pipeline after enough clean scenes are selected.',
    }
    (out_root / 'clean_mining_pipeline_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (out_root / 'clean_mining_pipeline_report.md').write_text(_make_report(summary, miner_summary), encoding='utf-8')
    return summary


def _make_report(summary: Dict[str, object], miner_summary: Dict[str, object]) -> str:
    ss = miner_summary.get('safety_first_summary', {}) or {}
    inter = miner_summary.get('interaction_on_safe_summary', {}) or {}
    mfc = summary.get('mfc', {}) or {}
    lines = [
        '# CausalSensor4D public_release Safety-First Clean Mining Pipeline',
        '',
        '## Purpose',
        'This run mines strict clean safe-to-failure candidate scenes from AV2 by applying original-safety filtering before interaction filtering.',
        '',
        '## Conversion / input',
        f"- Mode: `{summary.get('conversion', {}).get('mode')}`",
        f"- Generic CSV dir: `{summary.get('conversion', {}).get('generic_tracks_csv') or summary.get('conversion', {}).get('out_dir')}`",
        '',
        '## Original-safety-first stage',
        f"- Input scenes: `{ss.get('num_input_csv')}`",
        f"- Analyzed scenes: `{ss.get('num_analyzed')}`",
        f"- Original-safe scenes: `{ss.get('num_original_safe')}`",
        f"- Safe rate: `{ss.get('safe_rate')}`",
        '',
        '## Interaction-on-safe stage',
        f"- Safe scenes scanned: `{inter.get('num_csv_files')}`",
        f"- Selected clean interaction scenes: `{inter.get('num_selected')}`",
        f"- Selected counts by label: `{inter.get('selected_counts_by_label')}`",
        f"- Selected clean CSV dir: `{summary.get('selected_clean_csv_dir')}`",
        '',
        '## MFC clean-mined run',
        f"- Status: `{mfc.get('status')}`",
        f"- MFC output dir: `{mfc.get('mfc_output_dir')}`",
        '',
        '## Next use',
        'Use the selected clean CSV folder as input to the clean LLM pipeline. If the selected count is still too small, increase LIMIT or lower min_ttc_safe from 2.0 to 1.5 for an exploratory run.',
    ]
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description='CausalSensor4D public_release safety-first clean mining pipeline')
    parser.add_argument('--av2-root', default=None, help='AV2 validation root. Used if --existing-csv-dir is not set.')
    parser.add_argument('--existing-csv-dir', default=None, help='Existing generic_tracks_csv folder.')
    parser.add_argument('--out', default='outputs/av2_clean_mining_pipeline')
    parser.add_argument('--limit', type=int, default=500, help='0 means all files when converting from AV2 root.')
    parser.add_argument('--max-tracks', type=int, default=32)
    parser.add_argument('--planner', default='delayed')
    parser.add_argument('--min-ttc-safe', type=float, default=2.0)
    parser.add_argument('--max-per-type', type=int, default=80)
    parser.add_argument('--top-k-total', type=int, default=120, help='0 means no total cap')
    parser.add_argument('--min-score', type=float, default=1.0)
    parser.add_argument('--no-mfc', action='store_true')
    args = parser.parse_args()

    summary = run_pipeline(
        out_root=args.out,
        av2_root=args.av2_root,
        existing_csv_dir=args.existing_csv_dir,
        limit=args.limit,
        max_tracks=args.max_tracks,
        planner=args.planner,
        min_ttc_safe=args.min_ttc_safe,
        max_per_type=args.max_per_type,
        top_k_total=args.top_k_total,
        min_score=args.min_score,
        run_mfc=not args.no_mfc,
    )
    print('CausalSensor4D public_release safety-first clean mining pipeline finished.')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
