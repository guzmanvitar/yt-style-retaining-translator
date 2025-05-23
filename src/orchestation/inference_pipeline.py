"""
Video Translation Pipeline

This module defines a Prefect flow to orchestrate the stages of the YouTube video translation
pipeline.
It downloads videos from the provided URLs, processes and segments audio, translates transcriptions
creates transated audio from transcripton, and synchronizes the final translated audio with video.

Usage:
    python pipeline.py <url1> <url2> ...

Arguments:
    urls: One or more YouTube video URLs to process.

The pipeline consists of the following sequential steps:
    1. Download videos and extract audio.
    2. Convert and segment audio files.
    3. Translate the transcription.
    4. Run TTS on translated transcription to generate audio.
    5. Synchronize translated audio with the original video.
"""

import subprocess
from pathlib import Path

import click
from prefect import flow, task

from src.logger_definition import get_logger

logger = get_logger(__file__)


@task(retries=1, retry_delay_seconds=5)
def run_script(
    script: str,
    args: list[str] | None = None,
    input_file: str | None = None,
    output_file: str | None = None,
):
    """
    Run a Python module as a subprocess with optional arguments and I/O checks.

    Args:
        script: Dotted module path to the Python script (e.g., 'src.module.script').
        args: Optional list of command-line arguments to pass to the script.
        input_file: Optional input file path to validate before running.
        output_file: Optional output file path to validate after execution.

    Raises:
        FileNotFoundError: If the input or output file is missing when expected.
        subprocess.CalledProcessError: If the script execution fails.
    """

    if input_file and not Path(input_file).exists():
        raise FileNotFoundError(f"Missing input: {input_file}")

    cmd = ["uv", "run", "python", "-m", script]
    if args:
        cmd += args

    logger.info(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    if output_file and not Path(output_file).exists():
        raise FileNotFoundError(f"Expected output not found: {output_file}")
    logger.info(f"[DONE] {script}")


@flow
def video_translation_pipeline(urls: list[str], voice: str):
    run_script(
        "src.downloaders.download_youtube",
        args=urls,
    )
    run_script("src.preprocessing.convert_audio")
    run_script("src.preprocessing.segment_audio")
    run_script("src.translation.translate_transcription")
    run_script(
        "src.translation.translate_audio",
        args=[
            "--voice",
            voice,
        ],
    )
    run_script(
        "src.postprocessing.synchronize_audio_video_segments",
        args=[
            "--merge",
            "--cleanup",
        ],
    )


@click.command()
@click.argument("urls", nargs=-1, required=True)
@click.option(
    "--voice",
    type=str,
    required=True,
    help="Pre trained speaker voice to use for inference",
)
def main(urls: tuple[str], voice):
    video_translation_pipeline(urls=list(urls), voice=voice)


if __name__ == "__main__":
    main()
