"""
XTTS Inference Script

This module runs inference using a fine-tuned XTTS model to generate speech
from input text in a target language and voice style. It supports command-line
arguments for flexible invocation.
"""

from datetime import datetime
from pathlib import Path

import click
import torch
from TTS.api import TTS

from src.constants import DATA_INFERENCE, MODEL_OUTPUT_PATH
from src.logger_definition import get_logger

logger = get_logger(__file__)


def run_inference(
    text: str,
    tts_model: TTS,
    output_filename: str = "output",
    output_dir: Path | None = None,
    language: str = "es",
    speafer_ref: str = "ref_en",
) -> None:
    """
    Run XTTS inference and save audio to disk.

    Args:
        text (str): Input text to synthesize.
        tts_model (TTS.api.TTS): TTS trained model.
        output_filename (str): Output WAV file name.
        output_folder (str): Output folder for inference under inference dir.
        language (str): Target language for synthesis.
        speafer_ref (str): Name of the speaker reference wav.
    """
    speaker_wav = MODEL_OUTPUT_PATH / "speaker_references" / f"{speafer_ref}.wav"

    if not output_dir:
        output_dir = DATA_INFERENCE / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_filename}.wav"

    tts_model.tts_to_file(
        text=text,
        speaker_wav=str(speaker_wav),
        language=language,
        file_path=str(output_path),
    )

    logger.debug(f"Audio generated and saved to: {output_path}")


@click.command()
@click.option(
    "--text",
    type=str,
    required=True,
    help="Text to synthesize.",
)
@click.option(
    "--output-filename",
    type=str,
    default="xtts_output",
    help="Output WAV file name.",
)
@click.option(
    "--output-dir",
    type=str,
    default=None,
    help="Output dir for inference. Defaults to DATA_INFERENCE / timestamp.",
)
@click.option(
    "--model-name",
    type=str,
    default="production_latest",
    help="Model folder name.",
)
@click.option(
    "--language",
    type=str,
    default="es",
    help="Target language for output speech.",
)
@click.option(
    "--speafer-ref",
    type=str,
    default="ref_en",
    help="Speaker reference wav for inference.",
)
def main(text, output_filename, output_dir, model_name, language, speafer_ref):
    """Run inference with a fine-tuned XTTS model."""
    # Initialize model
    gpu = torch.cuda.is_available()
    model_path = MODEL_OUTPUT_PATH / model_name

    tts_model = TTS(
        model_path=model_path,
        config_path=model_path / "config.json",
        progress_bar=True,
        gpu=gpu,
    )

    # Run inference
    run_inference(
        text=text,
        tts_model=tts_model,
        output_filename=output_filename,
        output_dir=output_dir,
        language=language,
        speafer_ref=speafer_ref,
    )


if __name__ == "__main__":
    main()
