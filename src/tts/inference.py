import os
from TTS.api import TTS
from src.constants import MODEL_OUTPUT_PATH, DATA_COQUI

# Path to the directory containing the fine-tuned model and config files
MODEL_PATH = MODEL_OUTPUT_PATH / "production_05_05_25"
CONFIG_PATH = MODEL_OUTPUT_PATH / "production_05_05_25" / "config.json"

# Your custom speaker reference clip
SPEAKER_WAV = DATA_COQUI / "wavs/54l8_ewcOlY_chunk_000_segment_004.wav"

# Output path
OUTPUT_WAV = "output.wav"

# Input text
TEXT = "Explosion combinatoria"

# Language to synthesize in
LANGUAGE = "es"

# Load the model
tts = TTS(model_path=MODEL_PATH, config_path=CONFIG_PATH, progress_bar=True, gpu=False)

# Run inference
tts.tts_to_file(
    text=TEXT,
    speaker_wav=SPEAKER_WAV,
    language=LANGUAGE,
    file_path=OUTPUT_WAV,
)

print(f"✅ Audio generated and saved to: {OUTPUT_WAV}")
