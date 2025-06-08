"""
Lip sync pipeline for translated videos using Wav2Lip.

This script checks for synchronized video/audio folders and applies lip synchronization
using the Wav2Lip model. Already-processed videos are skipped automatically.

It is designed to run on a GPU-enabled environment and assumes the Wav2Lip virtual environment
is available in its repo folder.
"""

import subprocess
from pathlib import Path

import click

from src.constants import DATA_LIP_SYNCHED, DATA_SYNCHED, SUPPORT_REPOS
from src.logger_definition import get_logger

logger = get_logger(__file__)


def run_wav2lip(face_video: Path, audio_path: Path, output_path: Path):
    """
    Run Wav2Lip inference on a video/audio pair.

    Args:
        face_video (Path): Path to the video file containing the speaker's face.
        audio_path (Path): Path to the translated audio with hum fillers.
        output_path (Path): Output path for the final lip-synced video.
    """
    wav2lip_repo = SUPPORT_REPOS / "Wav2Lip"
    wav2lip_script = wav2lip_repo / "inference.py"
    python_bin = wav2lip_repo / ".venv" / "bin" / "python"
    checkpoint_path = wav2lip_repo / "checkpoints" / "wav2lip_gan.pth"

    cmd = [
        str(python_bin),
        str(wav2lip_script),
        "--checkpoint_path",
        str(checkpoint_path),
        "--face",
        str(face_video),
        "--audio",
        str(audio_path),
        "--outfile",
        str(output_path),
    ]

    logger.info(f"[WAV2LIP] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        logger.error(f"[WAV2LIP] Failed with return code {result.returncode}")
        logger.error(result.stderr)
        raise RuntimeError("Wav2Lip processing failed.")
    else:
        logger.info(f"[WAV2LIP] Output saved to: {output_path}")


@click.command()
def main():
    """
    Iterate over processed video/audio folders and apply lip-sync using Wav2Lip.

    Skips entries already lip-synced. This script expects:
    - a video named {name}.mp4 in DATA_SYNCHED/{name}/
    - a filler-enhanced audio file named {name}_lip_sync.wav
    and produces:
    - a lip-synced video at DATA_LIP_SYNCHED/{name}.mp4
    """
    for inference_dir in DATA_SYNCHED.iterdir():
        name = inference_dir.name
        output_path = DATA_LIP_SYNCHED / f"{name}.mp4"

        if output_path.exists():
            logger.info("Skipping %s — lip-synced video already exists.", name)
            continue

        face_video = inference_dir / f"{name}.mp4"
        audio_file = inference_dir / f"{name}_lip_sync.wav"

        if not face_video.exists() or not audio_file.exists():
            logger.warning(
                "Missing input for %s — skipping. (Video: %s, Audio: %s)",
                name,
                face_video.exists(),
                audio_file.exists(),
            )
            continue

        try:
            run_wav2lip(face_video, audio_file, output_path)
        except Exception as e:
            logger.error(f"Failed to process {name}: {e}")


if __name__ == "__main__":
    main()
