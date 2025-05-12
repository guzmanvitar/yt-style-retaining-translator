"""Run Coqui TTS training using the official training script."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from src.constants import MODEL_CONFIG_PATH, TRAINERS

SCRIPT_PATHS = {
    "vits": TRAINERS / "train_vits.py",
    "xtts": TRAINERS / "train_xtts_v2.py",
}
CONFIG_PATHS = {
    "vits": MODEL_CONFIG_PATH / "vits-config.json",
    "xtts": MODEL_CONFIG_PATH / "xtts-config.json",
}


def train_model(model_type: str) -> None:
    """Launch training script for the specified model type."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(".").resolve())

    script = SCRIPT_PATHS[model_type]
    config = CONFIG_PATHS[model_type]

    process = subprocess.Popen(
        ["python", str(script), "--config-path", str(config)],
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
    args = parser.parse_args()
    train_model(args.model)


if __name__ == "__main__":
    main()
