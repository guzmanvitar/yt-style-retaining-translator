"""Resample WAV audio files from YouTube downloads to multiple target sample rates."""

import argparse
from pathlib import Path

from pydub import AudioSegment

from src.constants import DATA_PROCESSED, DATA_RAW
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


def main():
    """Resamples all WAV files in the input directory to the specified sample rates."""
    parser = argparse.ArgumentParser(
        description="Resample WAV audio to multiple sample rates."
    )
    parser.add_argument(
        "--sample-rates",
        nargs="+",
        type=int,
        default=[16000, 22050],
        help="List of target sample rates in Hz (default: 16000 22050).",
    )

    args = parser.parse_args()

    input_dir = DATA_RAW / "audios"
    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        return

    wav_files = list(input_dir.glob("*.wav"))
    if not wav_files:
        logger.warning("No WAV files found in %s", input_dir)
        return

    for wav_path in wav_files:
        for sr in args.sample_rates:
            output_dir = DATA_PROCESSED / f"{sr}hz"
            convert_audio(wav_path, output_dir, sr)


if __name__ == "__main__":
    main()
