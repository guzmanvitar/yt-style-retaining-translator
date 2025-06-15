"""Convert segmented audio and transcription files into a Coqui TTS training dataset."""

import csv
from shutil import copy2

import click
from pydub import AudioSegment

from src.constants import DATA_COQUI, DATA_PRE_PROCESSED
from src.logger_definition import get_logger

logger = get_logger(__file__)


@click.command()
@click.option(
    "--min-duration-ms",
    type=int,
    default=2000,
    help="Minimum segment duration to include in milliseconds.",
)
@click.option(
    "--max-duration-ms",
    type=int,
    default=11000,
    help="Maximum segment duration to include in milliseconds.",
)
def merge_coqui_csvs_and_audio(
    min_duration_ms: int = 2000,
    max_duration_ms: int = 11000,
) -> None:
    """
    Merge all segmented CSVs and audio chunks into a single Coqui-compatible dataset.
    The merged dataset will include:
    - One `metadata.csv`
    - One `wavs/` directory

    Args:
        min_duration_ms (int): Minimum segment duration to include.
        max_duration_ms (int): Maximum segment duration to include.
    """
    output_dir = DATA_COQUI
    output_wav_dir = output_dir / "wavs"
    metadata_path = output_dir / "metadata.csv"

    output_wav_dir.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    with open(metadata_path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile, delimiter="|")

        for audio_dir in DATA_PRE_PROCESSED.iterdir():
            if not audio_dir.is_dir():
                continue

            name = audio_dir.name
            segments_dir = audio_dir / "segments"
            chunks_dir = segments_dir / "chunks"

            if not segments_dir.exists() or not chunks_dir.exists():
                logger.warning("Skipping %s — segments or chunks folder missing", name)
                continue

            logger.info("Processing directory: %s", audio_dir)

            for csv_file in sorted(segments_dir.glob("*.csv")):
                logger.info("Reading CSV: %s", csv_file.name)
                with open(csv_file, newline="", encoding="utf-8") as infile:
                    reader = csv.DictReader(infile)
                    for row in reader:
                        original_filename = row["filename"]
                        text = row["text"].strip()

                        source_file = chunks_dir / original_filename
                        new_filename = f"{name}__{original_filename}"
                        target_file = output_wav_dir / new_filename

                        if not source_file.exists():
                            logger.warning("Missing file: %s", source_file)
                            continue

                        duration_ms = len(AudioSegment.from_file(source_file))
                        if not (min_duration_ms <= duration_ms <= max_duration_ms):
                            logger.debug(
                                "Skipping %s (duration: %d ms)",
                                original_filename,
                                duration_ms,
                            )
                            continue

                        copy2(source_file, target_file)
                        writer.writerow([new_filename.removesuffix(".wav"), text, text])
                        rows_written += 1

    logger.info("Finished. Wrote %d samples to %s", rows_written, metadata_path)


if __name__ == "__main__":
    merge_coqui_csvs_and_audio()
