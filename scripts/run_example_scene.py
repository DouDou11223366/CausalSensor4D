from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
cmd = [
    sys.executable, "-m", "causalsensor4d.run_csv_scene",
    "--csv", str(ROOT / "examples" / "batch_csv_scenes" / "scene_003_cutin_candidate.csv"),
    "--out", str(ROOT / "outputs" / "example_cutin"),
    "--planner", "delayed",
]
print(" ".join(cmd))
subprocess.run(cmd, check=True, cwd=ROOT)
