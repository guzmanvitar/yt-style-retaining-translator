"""Resample WAV audio files from YouTube downloads to multiple target sample rates."""

from pathlib import Path

import click
from pydub import AudioSegment

from src.constants import DATA_FINAL, DATA_PRE_PROCESSED, DATA_RAW
from src.logger_definition import get_logger

logger = get_logger(__file__)


def convert_audio(input_path: Path, output_dir: Path, sample_rate: int) -> None:
    """
    Convert a stereo WAV file to mono and resample it to a target sample rate.

    Args:
        input_path (Path): Path to the input .wav file.
        output_dir (Path): Directory to save the resampled audio.
        sample_rate (int): Target sampling rate in Hz.
    """
    audio = AudioSegment.from_wav(input_path)
    audio = audio.set_channels(1).set_frame_rate(sample_rate)

    output_path = output_dir / f"{input_path.stem}_{sample_rate}.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio.export(output_path, format="wav")

    logger.info("Saved %d Hz mono WAV to %s", sample_rate, output_path)


@click.command()
@click.option(
    "--sample-rates",
    multiple=True,
    type=int,
    default=(16000, 22050),
    help="Target sample rates in Hz. Can be specified multiple times, e.g., --sample-rates 16000"
    " --sample-rates 22050",
)
def main(sample_rates):
    """Resamples all WAV files in the input directory to the specified sample rates."""
    input_dir = DATA_RAW / "audios"
    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        return

    wav_files = list(input_dir.glob("*.wav"))
    if not wav_files:
        logger.warning("No WAV files found in %s", input_dir)
        return

    for wav_path in wav_files:
        name = wav_path.stem
        final_path = DATA_FINAL / f"{name}.mp4"
        if final_path.exists():
            logger.info("Skipping %s — Processed video found in %s", name, DATA_FINAL)
            continue

        for sr in sample_rates:
            output_dir = DATA_PRE_PROCESSED / name / f"{sr}hz"
            if output_dir.exists():
                logger.info("Skipping %s — already processed", f"{name} - {sr}hz")
                continue

            output_dir.mkdir(parents=True, exist_ok=True)
            convert_audio(wav_path, output_dir, sr)


if __name__ == "__main__":
    main()
