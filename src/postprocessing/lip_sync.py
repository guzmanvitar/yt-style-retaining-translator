"""
Lip sync pipeline for translated videos using Wav2Lip.

This script checks for synchronized video/audio folders and applies lip synchronization
using the Wav2Lip model. Already-processed videos are skipped automatically.

It is designed to run on a GPU-enabled environment and assumes the Wav2Lip virtual environment
is available in its repo folder.
"""

import os
import subprocess
from pathlib import Path

import click

from src.constants import (
    DATA_FINAL,
    DATA_LIP_SYNCHED,
    DATA_SYNCHED,
    SUPPORT_REPOS,
)
from src.logger_definition import get_logger

logger = get_logger(__file__)


def run_wav2lip(face_video: Path, audio_path: Path, output_path: Path):
    """
    Run Wav2Lip ONNX-HQ inference using the virtualenv Python.

    Args:
        face_video (Path): Path to the video file containing the speaker's face.
        audio_path (Path): Path to the translated audio with hum fillers.
        output_path (Path): Output path for the final lip-synced video.
    """
    wav2lip_repo = SUPPORT_REPOS / "wav2lip-onnx-HQ-custom"
    venv_python = wav2lip_repo / ".venv" / "bin" / "python"
    inference_script = wav2lip_repo / "inference_onnxModel.py"

    cmd = [
        str(venv_python),
        str(inference_script),
        "--face",
        str(face_video),
        "--audio",
        str(audio_path),
        "--outfile",
        str(output_path),
    ]

    logger.info(f"[WAV2LIP-ONNX] Running: {' '.join(cmd)}")

    try:
        subprocess.run(
            cmd,
            cwd=wav2lip_repo,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"[WAV2LIP-ONNX] Output saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"[WAV2LIP-ONNX] Failed with return code {e.returncode}")
        logger.error(f"stdout:\n{e.stdout}")
        logger.error(f"stderr:\n{e.stderr}")
        raise RuntimeError("Wav2Lip ONNX processing failed.") from e


def concat_video(input_dir: Path, output_dir: Path, name: str):
    """Concatenate aligned video segments"""
    segment_list = sorted(input_dir.glob("segment_*.mp4"))
    concat_file = input_dir / "concat_list.txt"
    with open(concat_file, "w") as f:
        for segment in segment_list:
            f.write(f"file '{segment.resolve()}'\n")

    concatenated_video = output_dir / f"{name}.mp4"
    cmd_concat = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-c",
        "copy",
        str(concatenated_video),
    ]
    subprocess.run(cmd_concat, check=True)

    # Remove temp concat file
    concat_file.unlink()


@click.command()
def main():
    """
    Iterate over segmented video/audio folders and apply lip-sync using Wav2Lip.

    For each segment in DATA_SYNCHED/{name}/segments:
    - If a corresponding audio file exists (same stem + "_lip_sync.wav"), lip-sync it.
    - Otherwise, skip the segment.

    Output is written to:
    - DATA_LIP_SYNCHED/{name}/segments/{segment_name}.mp4
    """
    for inference_dir in DATA_SYNCHED.iterdir():
        name = inference_dir.name

        final_path = DATA_FINAL / f"{name}.mp4"
        if final_path.exists():
            logger.info("Skipping %s — Final video found in %s", name, DATA_FINAL)
            continue

        segment_dir = inference_dir / "segments"
        if not segment_dir.exists():
            logger.warning("No segments folder found in %s — skipping.", inference_dir)
            continue

        output_dir = DATA_LIP_SYNCHED / name / "segments"
        output_dir.mkdir(parents=True, exist_ok=True)

        for video_file in sorted(segment_dir.glob("*.mp4")):
            stem = video_file.stem
            audio_file = segment_dir / f"{stem}.wav"
            output_path = output_dir / f"{stem}.mp4"

            if output_path.exists():
                logger.info("Skipping segment %s — already processed.", stem)
                continue

            if not audio_file.exists():
                logger.info("Missing audio for segment %s — skipping.", stem)
                continue

            try:
                run_wav2lip(video_file, audio_file, output_path)
            except Exception as e:
                logger.error(f"Failed to process segment {stem} in {name}: {e}")

        concat_video(output_dir, DATA_LIP_SYNCHED / name, name)


if __name__ == "__main__":
    main()
