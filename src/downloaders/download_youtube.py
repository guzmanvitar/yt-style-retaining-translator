"""YouTube download script for training and inference audio/video extraction."""

import argparse
import subprocess
from pathlib import Path

from src.constants import DATA_RAW
from src.logger_definition import get_logger

logger = get_logger(__file__)


def download_audio(url: str, output_dir: Path) -> None:
    """Download and extract WAV audio from a YouTube video."""
    save_dir = output_dir / "audios"
    save_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "yt-dlp",
        "--cookies-from-browser",
        "chrome",
        "-x",  # extract audio
        "--audio-format",
        "wav",
        "-o",
        str(save_dir / "%(id)s.%(ext)s"),
        url,
    ]

    logger.info("Downloading audio to %s", save_dir)
    subprocess.run(cmd, check=True)


def download_video(url: str, output_dir: Path) -> None:
    """Download the full MP4 video from a YouTube video."""
    save_dir = output_dir / "videos"
    save_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "yt-dlp",
        "--cookies-from-browser",
        "chrome",
        "-f",
        "bestvideo+bestaudio",
        "--merge-output-format",
        "mp4",
        "-o",
        str(save_dir / "%(id)s.%(ext)s"),
        url,
    ]

    logger.info("Downloading video to %s", save_dir)
    subprocess.run(cmd, check=True)


def main() -> None:
    """Parse CLI arguments and download YouTube video(s) to video and audio folders."""
    parser = argparse.ArgumentParser(
        description="Download YouTube video(s) and extract WAV audio."
    )
    parser.add_argument(
        "urls",
        nargs="+",
        help="YouTube video URLs to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_RAW,
        help="Directory to save downloads (default: data/raw/).",
    )

    args = parser.parse_args()

    for url in args.urls:
        try:
            download_video(url, args.output_dir)
            download_audio(url, args.output_dir)
        except subprocess.CalledProcessError as e:
            logger.error("Download failed for %s: %s", url, e)


if __name__ == "__main__":
    main()
