python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
Write-Host "Environment ready. Run: python -m causalsensor4d.run_mvp --scene examples/scene_lead_brake.json --out outputs/mvp_run"
