"""Defines project constants"""

from pathlib import Path

this = Path(__file__)

ROOT = this.parents[1]

LOGS = ROOT / "logs"

SRC = ROOT / "src"

TTS = SRC / "tts"
TRAINERS = TTS / "trainers"

TRANSLATION = SRC / "translation"
LLM_SERVICE = TRANSLATION / "llm_service"
LLM_SERVICE_CONFIG = LLM_SERVICE / "llm-services-config.yaml"

MODEL_CONFIG_PATH = TTS / "configs"

MODEL_OUTPUT_PATH = ROOT / "models"

SECRETS = ROOT / ".secrets"

DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_PRE_PROCESSED = DATA / "pre_processed"
DATA_INFERENCE = DATA / "inference"
DATA_SYNCHED = DATA / "synched"
DATA_LIP_SYNCHED = DATA / "lip_synched"
DATA_FINAL = DATA / "final"

TESTS_DIR = ROOT / "tests"

DATA_COQUI = DATA / "coqui"

XTTS_PRETRAINED_DIR = (
    Path.home()
    / ".local"
    / "share"
    / "tts"
    / "tts_models--multilingual--multi-dataset--xtts_v2"
)

SUPPORT_REPOS = Path.home() / "support_repos"

LOGS.mkdir(exist_ok=True, parents=True)
DATA_RAW.mkdir(exist_ok=True, parents=True)
DATA_PRE_PROCESSED.mkdir(exist_ok=True, parents=True)
DATA_INFERENCE.mkdir(exist_ok=True, parents=True)
DATA_SYNCHED.mkdir(exist_ok=True, parents=True)
DATA_LIP_SYNCHED.mkdir(exist_ok=True, parents=True)
DATA_FINAL.mkdir(exist_ok=True, parents=True)
