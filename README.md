# CausalSensor4D

CausalSensor4D is a trajectory-level counterfactual testing and diagnostic framework for autonomous-driving scenes. It follows a clean safe-to-failure protocol: factual scenes are first checked for safety, interaction-rich scenes are selected, structured counterfactual edits are searched, and failures are reported with minimum failure cost (MFC) and taxonomy-aware evidence.

This public repository contains the reusable source code, toy trajectory examples, and data-adapter utilities. Raw Argoverse 2 data, large generated outputs, and large generated result assets are not included.

## Repository structure

```text
src/causalsensor4d/        Core Python package
examples/                  Toy generic trajectory CSV examples
scripts/                   Runnable smoke-test examples
data/                      Placeholder for user-provided data
outputs/                   Generated outputs, ignored by Git
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Quick smoke test

Run a single toy CSV scene:

```bash
python -m causalsensor4d.run_csv_scene \
  --csv examples/batch_csv_scenes/scene_003_cutin_candidate.csv \
  --out outputs/example_cutin \
  --planner delayed
```

Or run the helper script:

```bash
python scripts/run_example_scene.py
```

Run a small batch baseline on the toy scenes:

```bash
python scripts/run_example_batch.py
```

## Data notes

Raw AV2 scenarios are not redistributed in this repository. To run large-scale experiments, obtain the dataset from the official provider and convert scenes into the generic trajectory CSV format expected by the package.

## Important notes

- `outputs/` is ignored by Git because generated candidate tables can be large.
- Toy examples are included only for smoke testing.
- Online LLM/OpenRouter-related modules use environment variables for API keys; no API key is included in this repository.
