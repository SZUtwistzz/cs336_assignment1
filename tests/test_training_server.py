import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_owt_vocab_training_and_resume_from_another_directory(tmp_path):
    """OWT's highest ID must train, and resume must inherit its vocabulary size."""
    data_path = tmp_path / "owt_tokens.bin"
    np.full(512, 31999, dtype=np.uint16).tofile(data_path)
    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUTF8"] = "1"
    command = [sys.executable, "-m", "cs336_basics.train_LM.training_together"]
    first = subprocess.run(command + [
        "--device", "cpu", "--vocab-size", "32000", "--batch-size", "1",
        "--num-steps", "2", "--warmup-steps", "0", "--log-every", "1",
        "--eval-every", "2", "--eval-batches", "1", "--save-every", "1",
        "--train-path", str(data_path), "--valid-path", str(data_path),
    ], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert first.returncode == 0, first.stdout + first.stderr
    configs = list((tmp_path / "runs").glob("*/config.json"))
    assert len(configs) == 1
    original = configs[0].parent
    config = json.loads(configs[0].read_text(encoding="utf-8"))
    assert config["vocab_size"] == 32000
    assert Path(config["run_dir"]).is_absolute()
    with (original / "metrics.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert {row["split"] for row in rows} == {"train", "valid"}
    assert all(math.isfinite(float(row["loss"])) for row in rows)

    source = original / "checkpoints" / "step_1.pt"
    resumed = tmp_path / "resumed"
    second = subprocess.run(command + [
        "--resume", str(source), "--device", "cpu", "--run-dir", str(resumed),
    ], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert second.returncode == 0, second.stdout + second.stderr
    resumed_config = json.loads((resumed / "config.json").read_text(encoding="utf-8"))
    assert resumed_config["vocab_size"] == 32000
    assert resumed_config["start_step"] == 1
    assert (resumed / "checkpoints" / "step_2.pt").is_file()

    rejected_dir = tmp_path / "wrong_vocab"
    rejected = subprocess.run(command + [
        "--resume", str(source), "--device", "cpu", "--vocab-size", "10000",
        "--run-dir", str(rejected_dir),
    ], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert rejected.returncode != 0
    assert "model configuration differs: vocab_size" in rejected.stderr
    assert not rejected_dir.exists()
