"""Defines project constants"""

from pathlib import Path

this = Path(__file__)

ROOT = this.parents[1]

LOGS = ROOT / "logs"

SRC = ROOT / "src"

TTS = SRC / "tts"
TRAINERS = TTS / "trainers"
MODEL_CONFIG_PATH = TTS / "configs"

MODEL_OUTPUT_PATH = ROOT / "models"

SECRETS = ROOT / ".secrets"

DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_PROCESSED = DATA / "processed"

TESTS_DIR = ROOT / "tests"

DATA_COQUI = DATA / "coqui"

XTTS_PRETRAINED_DIR = (
    Path.home()
    / ".local"
    / "share"
    / "tts"
    / "tts_models--multilingual--multi-dataset--xtts_v2"
)

LOGS.mkdir(exist_ok=True, parents=True)
DATA_RAW.mkdir(exist_ok=True, parents=True)
DATA_PROCESSED.mkdir(exist_ok=True, parents=True)
