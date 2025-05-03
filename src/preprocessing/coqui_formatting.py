"""Convert segmented audio and transcription files into a Coqui TTS training dataset."""

import argparse
import csv
from pathlib import Path
from shutil import copy2

from src.constants import DATA_COQUI, DATA_PROCESSED
from src.logger_definition import get_logger

logger = get_logger(__file__)


def merge_coqui_csvs_and_audio(
    segments_dir: Path,
    chunks_dir: Path,
    output_dir: Path,
) -> None:
    """
    Merge multiple segmented CSV files and copy their corresponding audio files
    into a Coqui-compatible training dataset.

    Args:
        segments_dir (Path): Directory containing multiple CSVs with segment data.
        chunks_dir (Path): Directory containing all the audio chunk .wav files.
        output_dir (Path): Output directory where `wavs/` and `metadata.csv` will be written.
    """
    logger.info("Preparing Coqui dataset from directory: %s", segments_dir)
    output_wav_dir = output_dir / "wavs"
    output_wav_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.csv"

    rows_written = 0
    with open(metadata_path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile, delimiter="|")

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

                    copy2(source_file, target_file)
                    writer.writerow([filename.removesuffix(".wav"), text, text])
                    rows_written += 1

    logger.info("Wrote %d samples to %s", rows_written, metadata_path)


def main() -> None:
    """Run Coqui dataset preparation by merging all segmented CSVs in a directory."""
    parser = argparse.ArgumentParser(description="Prepare full dataset for Coqui TTS training.")
    parser.add_argument(
        "--segments-dir",
        type=Path,
        default=DATA_PROCESSED / "segments",
        help="Directory containing segmented CSV files (default: %(default)s)",
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=DATA_PROCESSED / "segments" / "chunks",
        help="Directory containing audio chunk files (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_COQUI,
        help="Output directory for prepared dataset (default: %(default)s)",
    )

    args = parser.parse_args()

    merge_coqui_csvs_and_audio(args.segments_dir, args.chunks_dir, args.output_dir)


if __name__ == "__main__":
    main()
