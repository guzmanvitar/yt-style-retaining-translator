"""
XTTS Inference Script

This module runs inference using a fine-tuned XTTS model to generate speech
from input text in a target language and voice style. It supports command-line
arguments for flexible invocation.
"""

from datetime import datetime

import click
from TTS.api import TTS

from src.constants import DATA_INFERENCE, MODEL_OUTPUT_PATH
from src.logger_definition import get_logger

logger = get_logger(__file__)


def run_inference(
    text: str,
    output_filename: str = "output",
    output_folder: str | None = None,
    model_name: str = "production_latest",
    language: str = "es",
    speafer_ref: str = "ref_en",
) -> None:
    """
    Run XTTS inference and save audio to disk.

    Args:
        text (str): Input text to synthesize.
        output_filename (str): Output WAV file name.
        output_folder (str): Output folder for inference under inference dir.
        model_name (str): Name of the folder under MODEL_OUTPUT_PATH containing the model and
            config.
        language (str): Target language for synthesis.
        speafer_ref (str): Name of the speaker reference wav.
    """
    model_path = MODEL_OUTPUT_PATH / model_name
    config_path = model_path / "config.json"
    speaker_wav = MODEL_OUTPUT_PATH / "speaker_references" / f"{speafer_ref}.wav"

    if not output_folder:
        output_folder = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output_dir = DATA_INFERENCE / output_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_filename}.wav"

    tts = TTS(
        model_path=model_path, config_path=config_path, progress_bar=True, gpu=False
    )
    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        language=language,
        file_path=output_path,
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
    "--output-folder",
    type=str,
    default=None,
    help="Name of output folder under inference dir. Defaults to timestamp.",
)
@click.option(
    "--model",
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
def main(text, output_filename, output_folder, model, language, speafer_ref):
    """Run inference with a fine-tuned XTTS model."""
    run_inference(
        text=text,
        output_filename=output_filename,
        output_folder=output_folder,
        model_name=model,
        language=language,
        speafer_ref=speafer_ref,
    )


if __name__ == "__main__":
    main()
