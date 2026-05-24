from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
cmd = [
    sys.executable, "-m", "causalsensor4d.run_baseline_ablation_csv",
    "--csv-dir", str(ROOT / "examples" / "batch_csv_scenes"),
    "--out", str(ROOT / "outputs" / "example_baseline"),
    "--planner", "delayed",
    "--methods", "causal_guided,random_budget,distance_all,causal_hybrid",
    "--random-budget", "36",
    "--seed", "13",
]
print(" ".join(cmd))
subprocess.run(cmd, check=True, cwd=ROOT)
