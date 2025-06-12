"""
Finalize lip-synced videos by replacing their audio with translated speech and original background
sounds.
"""

import subprocess
from pathlib import Path

import click
import ffmpeg
from pydub import AudioSegment

from src.constants import DATA_FINAL, DATA_LIP_SYNCHED, DATA_RAW, DATA_SYNCHED
from src.logger_definition import get_logger

logger = get_logger(__file__)


def separate_background(audio_path: Path, output_dir: Path) -> Path:
    """
    Uses Demucs to extract background sounds from a full audio track.
    Falls back to silence if background separation fails or background file is missing.
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
    background_path = output_dir / "htdemucs" / f"{audio_path.stem}.wav"

    if not background_path.exists():
        raise FileNotFoundError

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
        final_path = DATA_FINAL / name / f"{name}.mp4"
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
            segment_dir = out_dir / "segments"
            chunk_dir = segment_dir / "chunks"
            bg_dir = segment_dir / "no_vocals"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            bg_dir.mkdir(parents=True, exist_ok=True)

            audio = AudioSegment.from_wav(raw_audio)
            chunk_length_ms = 60_000
            num_chunks = len(audio) // chunk_length_ms + 1

            bg_segments = []
            for i in range(num_chunks):
                chunk_path = chunk_dir / f"chunk_{i:04d}.wav"
                if not chunk_path.exists():
                    chunk = audio[i * chunk_length_ms : (i + 1) * chunk_length_ms]
                    chunk.export(chunk_path, format="wav")

                bg_out_path = bg_dir / f"chunk_{i:04d}.wav"
                if not bg_out_path.exists():
                    bg_path = separate_background(chunk_path, bg_dir)

                bg_segments.append(bg_path)

            concat_bg_path = out_dir / f"{name}.bg.wav"
            combined = AudioSegment.empty()
            for bg in bg_segments:
                combined += AudioSegment.from_wav(bg)
            combined.export(concat_bg_path, format="wav")

            mixed_audio = mix_tracks(translated_audio, concat_bg_path, out_dir / name)
            mux_audio_to_video(video_path, mixed_audio, final_path)

            # Cleanup
            for file in segment_dir.glob("**/*"):
                file.unlink()
            for folder in sorted(segment_dir.glob("**/*"), reverse=True):
                if folder.is_dir():
                    folder.rmdir()
            segment_dir.rmdir()

            logger.info("Final video written: %s", final_path)

        except Exception as e:
            logger.error("Failed to process %s: %s", name, e)


if __name__ == "__main__":
    main()
