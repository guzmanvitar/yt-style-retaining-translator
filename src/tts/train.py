"""Run Coqui TTS training using the official training script."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from src.constants import TRAINERS

SCRIPT_PATHS = {
    "vits": TRAINERS / "train_vits.py",
    "xtts": TRAINERS / "train_xtts_v2.py",
}


def train_model(model_type: str, voice: str) -> None:
    """Launch training script for the specified model type and voice."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(".").resolve())

    script = SCRIPT_PATHS[model_type]

    process = subprocess.Popen(
        ["python", str(script), voice],
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    process.communicate()


def main() -> None:
    """Parse arguments and trigger model training."""
    parser = argparse.ArgumentParser(description="Run training for VITS or XTTS.")
    parser.add_argument(
        "--model",
        type=str,
        choices=["vits", "xtts"],
        default="xtts",
        help="Model type to train: vits or xtts (default: xtts).",
    )
    parser.add_argument(
        "voice",
        type=str,
        help="Name of the voice folder to use (required).",
    )
    args = parser.parse_args()
    train_model(args.model, args.voice)


if __name__ == "__main__":
    main()
