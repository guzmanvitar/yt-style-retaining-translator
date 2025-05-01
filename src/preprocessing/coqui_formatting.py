"""Convert segmented audio and transcription to Coqui TTS training format."""

import argparse
import csv
from pathlib import Path
from shutil import copy2

from src.constants import DATA_COQUI, DATA_PROCESSED
from src.logger_definition import get_logger

logger = get_logger(__file__)


def prepare_coqui_dataset(
    segments_csv: Path,
    chunks_dir: Path,
    output_dir: Path,
    chunk_range: slice | None = None,
) -> None:
    """Convert segmented audio and transcript into Coqui-compatible dataset format.

    Args:
        segments_csv (Path): Path to CSV with filename, start, end, text.
        chunks_dir (Path): Directory with segmented .wav files.
        output_dir (Path): Output directory where `wavs/` and `metadata.csv` are written.
        chunk_range (Optional[slice]): A slice object to select a subset of chunks.
    """
    logger.info("Preparing Coqui dataset from %s", segments_csv)
    output_wav_dir = output_dir / "wavs"
    output_wav_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.csv"

    rows_written = 0
    with (
        open(segments_csv, newline="", encoding="utf-8") as infile,
        open(metadata_path, "w", newline="", encoding="utf-8") as outfile,
    ):

        reader = list(csv.DictReader(infile))
        writer = csv.writer(outfile, delimiter="|")

        selected_rows = reader[chunk_range] if chunk_range else reader

        for row in selected_rows:
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
    """Run dataset preparation from CLI or with defaults.

    If no arguments are provided, uses:
        - segments_csv: DATA_PROCESSED / "chunks" / "segments.csv"
        - chunks_dir: DATA_PROCESSED / "chunks"
        - output_dir: DATA_COQUI

    Optional:
    --start and --end allow selecting a slice of rows.

    Example:
        python script.py \\
            --segments-csv path/to/segments.csv \\
            --chunks-dir path/to/chunks \\
            --output-dir path/to/output \\
            --start 0 --end 100
    """
    parser = argparse.ArgumentParser(
        description="Prepare dataset for Coqui TTS training."
    )
    parser.add_argument(
        "--segments-csv",
        type=Path,
        default=DATA_PROCESSED / "chunks" / "segments.csv",
        help="Path to segments CSV file (default: %(default)s)",
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=DATA_PROCESSED / "chunks",
        help="Directory containing audio chunks (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_COQUI,
        help="Output directory for the prepared dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start index for chunk slicing (inclusive)",
    )
    parser.add_argument(
        "--end", type=int, default=None, help="End index for chunk slicing (exclusive)"
    )

    args = parser.parse_args()
    chunk_range = (
        slice(args.start, args.end)
        if args.start is not None or args.end is not None
        else None
    )

    prepare_coqui_dataset(
        args.segments_csv, args.chunks_dir, args.output_dir, chunk_range
    )


if __name__ == "__main__":
    main()
