import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def test_resume_continues_step_schedule_and_optimizer(tmp_path):
    data_path = tmp_path / "tokens.bin"
    np.arange(1024, dtype=np.uint16).tofile(data_path)
    original = tmp_path / "original"
    resumed = tmp_path / "resumed"
    command = [sys.executable, "-m", "cs336_basics.train_LM.training_together"]
    cwd = Path(__file__).resolve().parents[1]
    first = subprocess.run(command + [
        "--device", "cpu", "--batch-size", "1", "--num-steps", "2",
        "--warmup-steps", "0", "--log-every", "1", "--eval-every", "2",
        "--eval-batches", "1", "--save-every", "1",
        "--train-path", str(data_path), "--valid-path", str(data_path),
        "--run-dir", str(original),
    ], cwd=cwd, capture_output=True, text=True, timeout=120)
    assert first.returncode == 0, first.stdout + first.stderr
    source = original / "checkpoints" / "step_1.pt"
    second = subprocess.run(command + [
        "--resume", str(source), "--device", "cpu", "--run-dir", str(resumed),
    ], cwd=cwd, capture_output=True, text=True, timeout=120)
    assert second.returncode == 0, second.stdout + second.stderr
    with open(resumed / "metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows and {int(row["step"]) for row in rows} == {2}
    assert abs(float(rows[0]["lr"]) - 0.000165) < 1e-10
    config = json.loads((resumed / "config.json").read_text(encoding="utf-8"))
    assert config["start_step"] == 1
    assert config["num_steps"] == 2
    saved = torch.load(resumed / "checkpoints" / "step_2.pt", map_location="cpu")
    assert saved["iteration"] == 2
    assert all(state["t"] == 3 for state in saved["optimizer"]["state"].values())
    assert source.exists()
    rejected_dir = tmp_path / "wrong_schedule"
    rejected = subprocess.run(command + [
        "--resume", str(source), "--device", "cpu", "--num-steps", "5",
        "--run-dir", str(rejected_dir),
    ], cwd=cwd, capture_output=True, text=True, timeout=120)
    assert rejected.returncode != 0
    assert "preserve the original num_steps" in rejected.stderr
    assert not rejected_dir.exists()
