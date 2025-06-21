"""
YouTube download script for training and inference audio/video extraction.

Downloads MP4 video and WAV audio from provided YouTube URLs,
saving them with sanitized, lowercase, underscore-separated titles without punctuation.
"""

import re
import shlex
import subprocess
from pathlib import Path

import click

from src.constants import DATA_FINAL, DATA_RAW
from src.logger_definition import get_logger

logger = get_logger(__file__)


def sanitize_title(title: str) -> str:
    """Sanitize and format the video title for use as a safe filename."""
    title = title.lower().strip()
    title = re.sub(r"[^\w\s]", "", title)
    title = re.sub(r"\s+", "_", title)
    return title


def get_video_title(url: str) -> str:
    """Use yt-dlp to get the video title from a YouTube URL."""
    result = subprocess.run(
        ["yt-dlp", "--cookies-from-browser", "chrome", "--get-title", url],
        check=True,
        capture_output=True,
        text=True,
    )
    return sanitize_title(result.stdout.strip())


def download_audio(url: str, output_dir: Path, title: str) -> None:
    """Download and extract WAV audio from a YouTube video."""
    save_dir = output_dir / "audios"
    save_dir.mkdir(parents=True, exist_ok=True)

    output_path = save_dir / f"{title}.wav"

    cmd = [
        "yt-dlp",
        "--cookies-from-browser",
        "chrome",
        "-x",  # extract audio
        "--audio-format",
        "wav",
        "-o",
        str(output_path),
        url,
    ]

    logger.info("Downloading audio to %s", output_path)
    subprocess.run(cmd, check=True)


def download_video(url: str, output_dir: Path, title: str) -> None:
    """Download the full MP4 video from a YouTube video."""
    save_dir = output_dir / "videos"
    save_dir.mkdir(parents=True, exist_ok=True)

    output_path = save_dir / f"{title}.mp4"

    cmd = [
        "yt-dlp",
        "--cookies-from-browser",
        "chrome",
        "-f",
        "bestvideo+bestaudio",
        "--merge-output-format",
        "mp4",
        "-o",
        str(output_path),
        url,
    ]

    logger.info("Downloading video to %s", output_path)
    subprocess.run(cmd, check=True)


@click.command()
@click.argument("urls", nargs=1, required=True)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=DATA_RAW,
    help="Directory to save downloads (default: data/raw/).",
)
def main(urls: str, output_dir: Path) -> None:
    """Download YouTube video(s) and extract WAV audio."""
    url_list = shlex.split(urls)

    for url in url_list:
        try:
            title = get_video_title(url)

            final_path = DATA_FINAL / f"{title}.mp4"
            if final_path.exists():
                logger.info(
                    "Skipping %s — Processed video found in %s", title, DATA_FINAL
                )
                continue

            download_video(url, output_dir, title)
            download_audio(url, output_dir, title)
        except subprocess.CalledProcessError as e:
            logger.error("Download failed for %s: %s", url, e)


if __name__ == "__main__":
    main()
