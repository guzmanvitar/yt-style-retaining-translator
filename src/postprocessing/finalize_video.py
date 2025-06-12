"""
Finalize lip-synced videos by replacing their audio with translated speech and original background
sounds.
"""

import subprocess
from pathlib import Path

import click
import ffmpeg

from src.constants import DATA_FINAL, DATA_LIP_SYNCHED, DATA_RAW, DATA_SYNCHED
from src.logger_definition import get_logger

logger = get_logger(__file__)


def separate_background(audio_path: Path, output_dir: Path) -> Path:
    """
    Uses Demucs to extract background sounds from a full audio track.
    """

    demucs_cmd = [
        "demucs",
        "--two-stems=vocals",
        "--out",
        str(output_dir),
        "--filename",
        "{track}.{ext}",
        str(audio_path),
    ]

    subprocess.run(demucs_cmd, check=True)
    background_path = output_dir / "no_vocals" / f"{audio_path.stem}.wav"

    if not background_path.exists():
        raise FileNotFoundError(f"Background audio not found: {background_path}")

    return background_path


def mix_tracks(voice_path: Path, bg_path: Path, output_path: Path) -> Path:
    """
    Mix translated voice with background track using ffmpeg.
    """
    mixed_audio = output_path.with_suffix(".mixed.wav")

    (
        ffmpeg.input(str(voice_path))
        .input(str(bg_path))
        .filter("amix", inputs=2, duration="first")
        .output(str(mixed_audio))
        .overwrite_output()
        .run()
    )

    return mixed_audio


def mux_audio_to_video(video_path: Path, audio_path: Path, output_path: Path) -> None:
    """
    Mux audio into video using ffmpeg, replacing original audio.
    """
    (
        ffmpeg.input(str(video_path))
        .input(str(audio_path))
        .output(str(output_path), vcodec="copy", acodec="aac", strict="experimental")
        .overwrite_output()
        .run()
    )


@click.command()
def main() -> None:
    """
    Replace lip-synced video audio with translated voice and original background sounds.
    """
    for lip_synch_dir in DATA_LIP_SYNCHED.iterdir():
        name = lip_synch_dir.name
        final_path = DATA_FINAL / f"{name}.mp4"
        if final_path.exists():
            logger.info("Skipping %s — final video exists.", name)
            continue

        logger.info("Processing: %s", name)
        video_path = lip_synch_dir / f"{name}.mp4"
        translated_audio = DATA_SYNCHED / name / f"{name}.wav"
        raw_audio = DATA_RAW / "audios" / f"{name}.wav"

        if (
            not video_path.exists()
            or not translated_audio.exists()
            or not raw_audio.exists()
        ):
            logger.warning("Missing files for %s — skipping.", name)
            continue

        try:
            out_dir = DATA_FINAL / name
            out_dir.mkdir(parents=True, exist_ok=True)

            bg_audio = separate_background(raw_audio, out_dir)
            mixed_audio = mix_tracks(translated_audio, bg_audio, out_dir / name)
            mux_audio_to_video(video_path, mixed_audio, final_path)

            logger.info("✅ Final video written: %s", final_path)

        except Exception as e:
            logger.error("Failed to process %s: %s", name, e)


if __name__ == "__main__":
    main()
