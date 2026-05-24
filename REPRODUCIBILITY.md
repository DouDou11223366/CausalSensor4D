# Reproducibility notes

The repository contains reusable source code and toy examples. Large-scale runs require external AV2-derived generic trajectory CSV files, which are not redistributed here.

Main modules:

- `risk.py`: deterministic verifier for collision, TTC, and hard-brake evidence.
- `edits.py`: counterfactual edit operators and edit costs.
- `search.py`: minimum failure cost search.
- `planner.py`: rule-based planner variants.
- `baseline_comparison.py`: method comparison and ablation table generation.
- `failure_taxonomy.py`: collision / hard-brake / low-TTC-only taxonomy.
- `bootstrap_ci.py`: bootstrap confidence interval analysis.

Suggested large-scale run order:

1. `python -m causalsensor4d.run_longitudinal_baseline_audit ...`
2. `python -m causalsensor4d.run_failure_taxonomy ...`
3. `python -m causalsensor4d.run_bootstrap_ci ...`
