"""YuoTube download video script for training and inference."""

import argparse
import subprocess
from pathlib import Path

from src.constants import DATA_RAW


def download_youtube_video(
    url: str,
    output_dir: Path,
    audio_only: bool = True,
) -> None:
    """Download a YouTube video or its audio.

    Args:
        url (str): YouTube video URL.
        output_dir (Path): Directory where the output will be stored.
        audio_only (bool): If True, extract audio as WAV. If False, download full video.
    """
    if audio_only:
        save_dir = output_dir / "audio"
        save_dir.mkdir(parents=True, exist_ok=True)
        output_template = str(save_dir / "%(id)s.%(ext)s")

        cmd = [
            "yt-dlp",
            "-x",  # extract audio
            "--audio-format",
            "wav",
            "-o",
            output_template,
            url,
        ]
    else:
        save_dir = output_dir / "video"
        save_dir.mkdir(parents=True, exist_ok=True)
        output_template = str(save_dir / "%(id)s.%(ext)s")

        cmd = [
            "yt-dlp",
            "-f",
            "bestvideo+bestaudio",
            "--merge-output-format",
            "mp4",
            "-o",
            output_template,
            url,
        ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    """Parse CLI arguments and download YouTube video(s)."""
    parser = argparse.ArgumentParser(
        description="Download YouTube video(s) as WAV audio or full video (MP4)."
    )
    parser.add_argument(
        "urls",
        nargs="+",
        help="YouTube video URLs to download.",
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Download audio only (as .wav). Default is full video (mp4).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_RAW,
        help="Directory to save downloads (default: data/raw/).",
    )

    args = parser.parse_args()

    for url in args.urls:
        download_youtube_video(
            url=url,
            output_dir=args.output_dir,
            audio_only=args.audio_only,
        )


if __name__ == "__main__":
    main()
