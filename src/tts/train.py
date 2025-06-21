""" ""Run Coqui TTS training using the official training script with voice argument support."""

import os
import subprocess
import sys
from pathlib import Path

import click

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
    """Launch training script for the specified model type and voice."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(".").resolve())

    script = SCRIPT_PATHS[model_type]

    process = subprocess.Popen(
        ["python", str(script)],
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    process.communicate()


@click.command()
@click.option(
    "--model",
    type=click.Choice(["vits", "xtts"], case_sensitive=False),
    default="xtts",
    help="Model type to train: vits or xtts (default: xtts).",
)
def main(model: str) -> None:
    """Trigger training with selected model type and voice folder."""
    train_model(model.lower())


if __name__ == "__main__":
    main()
