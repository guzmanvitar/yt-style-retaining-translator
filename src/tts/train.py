"""Run Coqui TTS training using the official training script."""

import os
import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = Path("src/tts/trainers/train_vits.py")
CONFIG_PATH = Path("src/tts/configs/vits-config.json")


def train_model():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(".").resolve())

    process = subprocess.Popen(
        ["python", str(SCRIPT_PATH), "--config-path", str(CONFIG_PATH)],
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    process.communicate()


if __name__ == "__main__":
    train_model()
