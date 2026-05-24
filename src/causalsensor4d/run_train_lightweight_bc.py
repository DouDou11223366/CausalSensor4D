from __future__ import annotations
import argparse
from pathlib import Path
from .lightweight_bc import train_lightweight_bc_planner, LightweightBCConfig
from .schemas import save_json

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight behavior-cloning planner.")
    parser.add_argument("--csv-dirs", required=True, help="CSV dirs/files separated by semicolon ';'.")
    parser.add_argument("--out", default="outputs/lightweight_bc_model")
    parser.add_argument("--ego-track-id", default="ego")
    parser.add_argument("--ridge-lambda", type=float, default=1e-2)
    parser.add_argument("--safety-blend", type=float, default=0.30)
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model_path = out / "lightweight_bc_model.npz"
    config = LightweightBCConfig(ridge_lambda=args.ridge_lambda, safety_blend=args.safety_blend)
    csv_dirs = [x.strip() for x in args.csv_dirs.split(";") if x.strip()]
    report = train_lightweight_bc_planner(csv_dirs, model_path, ego_track_id=args.ego_track_id, config=config)
    save_json(report.__dict__, out / "training_report.json")
    print("Lightweight BC planner trained.")
    print(f"CSV dirs: {csv_dirs}")
    print(f"Model: {model_path}")
    print(f"Training samples: {report.num_training_samples}")
    print(f"Train MSE: {report.train_mse:.6f}")
if __name__ == "__main__":
    main()
