"""Defines project constants"""

from pathlib import Path

this = Path(__file__)

ROOT = this.parents[1]

LOGS = ROOT / "logs"

SRC = ROOT / "src"

TTS = SRC / "tts"
MODEL_CONFIG_PATH = TTS / "configs"
MODEL_OUTPUT_PATH = ROOT / "models"

SECRETS = ROOT / ".secrets"

DATA = ROOT / "data"
DATA_RECORDINGS = DATA / "recordings"
DATA_PROCESSED = DATA / "processed"

TESTS_DIR = ROOT / "tests"

DATA_COQUI = DATA / "coqui"


LOGS.mkdir(exist_ok=True, parents=True)
DATA_RECORDINGS.mkdir(exist_ok=True, parents=True)
DATA_PROCESSED.mkdir(exist_ok=True, parents=True)
