"""Convert segmented audio and transcription files into a Coqui TTS training dataset."""

import csv
from shutil import copy2

import click
from pydub import AudioSegment

from src.constants import DATA_COQUI, DATA_PROCESSED
from src.logger_definition import get_logger

logger = get_logger(__file__)


@click.command()
@click.option(
    "--min-duration-ms",
    type=int,
    default=1000,
    help="Minimum segment duration to include in miliseconds.",
)
@click.option(
    "--max-duration-ms",
    type=int,
    default=11000,
    help="Maximum segment duration to include in miliseconds.",
)
def merge_coqui_csvs_and_audio(
    min_duration_ms: int = 1000,
    max_duration_ms: int = 11000,
) -> None:
    """
    Merge multiple segmented CSV files and copy their corresponding audio files
    into a Coqui-compatible training dataset, filtering by audio duration.

    Args:
        min_duration_ms (int): Minimum segment duration to include.
        max_duration_ms (int): Maximum segment duration to include.
    """
    for audio_dir in DATA_PROCESSED.iterdir():
        name = audio_dir.name

        logger.info("Preparing Coqui dataset from directory: %s", audio_dir)
        output_wav_dir = DATA_COQUI / name / "wavs"
        if output_wav_dir.exists():
            logger.info("Skipping %s — already processed", name)
            continue

        output_wav_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = DATA_COQUI / name / "metadata.csv"

        rows_written = 0
        with open(metadata_path, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.writer(outfile, delimiter="|")
            segments_dir = DATA_PROCESSED / name / "segments"
            chunks_dir = segments_dir / "chunks"

            for csv_file in sorted(segments_dir.glob("*.csv")):
                logger.info("Processing CSV: %s", csv_file.name)
                with open(csv_file, newline="", encoding="utf-8") as infile:
                    reader = csv.DictReader(infile)
                    for row in reader:
                        filename = row["filename"]
                        text = row["text"].strip()
                        source_file = chunks_dir / filename
                        target_file = output_wav_dir / filename

                        if not source_file.exists():
                            logger.warning("Missing file: %s", source_file)
                            continue

                        duration_ms = len(AudioSegment.from_file(source_file))
                        if (
                            duration_ms < min_duration_ms
                            or duration_ms > max_duration_ms
                        ):
                            logger.debug(
                                "Skipping %s (duration: %d ms)", filename, duration_ms
                            )
                            continue

                        copy2(source_file, target_file)
                        writer.writerow([filename.removesuffix(".wav"), text, text])
                        rows_written += 1

        logger.info("Wrote %d samples to %s", rows_written, metadata_path)


if __name__ == "__main__":
    merge_coqui_csvs_and_audio()
