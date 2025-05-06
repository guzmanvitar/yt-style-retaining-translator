"""
TTS Inference and Audio Assembly Pipeline.

This script performs text-to-speech (TTS) inference using a translated transcription CSV,
generates audio for each segment using an XTTS model, and stitches all segments together
with silence alignment. Segments are not stretched or compressed unless they would overlap
the next one, in which case they are gently sped up.
"""

from pathlib import Path
from typing import List, Tuple
import pandas as pd
from pydub import AudioSegment
import click

from src.tts.xtts_inference import run_inference
from src.constants import DATA_INFERENCE
from src.logger_definition import get_logger

logger = get_logger(__file__)


def run_segment_inference(
    csv_path: Path,
    output_dir_name: str = "tts_segments",
    model_name: str = "production_latest",
    speaker_ref: str = "ref_en",
    language: str = "es",
) -> List[Tuple[float, float, float, Path]]:
    """
    Run TTS inference on each translated segment.

    Args:
        csv_path (Path): CSV with translated text and timestamps.
        output_dir_name (str): Subfolder for segment WAVs.
        model_name (str): XTTS model name under MODEL_OUTPUT_PATH.
        speaker_ref (str): Reference speaker name (without .wav).
        language (str): Target synthesis language.

    Returns:
        List[Tuple[start, end, duration, Path]]: Segment metadata and audio paths.
    """
    df = pd.read_csv(csv_path)
    output_dir = DATA_INFERENCE / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    segments = []
    for i, row in df.iterrows():
        start = float(row["start"])
        end = float(row["end"])
        text = row["text"]
        output_name = f"segment_{i:04d}"
        output_path = output_dir / f"{output_name}.wav"

        run_inference(
            text=text,
            output_filename=output_name,
            output_folder=output_dir_name,
            model_name=model_name,
            language=language,
            speafer_ref=speaker_ref,
        )

        audio = AudioSegment.from_wav(output_path).set_channels(1)
        actual_duration = len(audio) / 1000  # in seconds

        segments.append((start, end, actual_duration, output_path))
        logger.info(f"✅ Generated {output_name}.wav ({actual_duration:.2f}s)")

    return segments


def assemble_audio_from_segments(
    segments: List[Tuple[float, float, float, Path]], output_path: Path
) -> None:
    """
    Assemble final audio by inserting silence and adjusting only overlapping segments.

    Args:
        segments (List[Tuple[start, end, actual_duration, Path]]): Audio segment metadata.
        output_path (Path): Path to write the final assembled WAV.
    """
    final = AudioSegment.silent(duration=0)
    last_end = 0.0

    for i, (start, end, actual_duration, path) in enumerate(segments):
        gap = max(0.0, start - last_end)
        final += AudioSegment.silent(duration=int(gap * 1000))

        audio = AudioSegment.from_wav(path).set_channels(1)

        if i + 1 < len(segments):
            next_start = segments[i + 1][0]
            if start + actual_duration > next_start:
                # Compression needed
                target_duration = next_start - start
                speed_factor = actual_duration / target_duration
                logger.warning(
                    f"⚠️ Segment {i} overlaps next: speeding up x{speed_factor:.2f}"
                )
                # Speed up
                audio = (
                    audio._spawn(
                        audio.raw_data,
                        overrides={"frame_rate": int(audio.frame_rate * speed_factor)},
                    )
                    .set_frame_rate(audio.frame_rate)
                    .set_channels(1)
                )

        final += audio
        last_end = start + len(audio) / 1000

    final.export(output_path, format="wav")
    logger.info(f"✅ Final stitched audio saved to {output_path}")


@click.command()
@click.option(
    "--csv-path",
    type=Path,
    default=DATA_INFERENCE / "translated_segments.csv",
    help="Path to translated CSV with start/end/text columns.",
)
@click.option(
    "--output-path",
    type=Path,
    default=DATA_INFERENCE / "final_audio.wav",
    help="Output WAV file for final assembled audio.",
)
@click.option(
    "--model-name",
    type=str,
    default="production_latest",
    help="Model folder name under MODEL_OUTPUT_PATH.",
)
@click.option(
    "--speaker-ref",
    type=str,
    default="ref_en",
    help="Reference speaker WAV filename (without extension).",
)
@click.option(
    "--language",
    type=str,
    default="es",
    help="Target language code (e.g. 'es', 'en').",
)
def main(csv_path, output_path, model_name, speaker_ref, language):
    """Run XTTS inference on translated segments and assemble full audio."""
    logger.info("🚀 Running XTTS inference + reassembly...")
    segments = run_segment_inference(
        csv_path=csv_path,
        output_dir_name="tts_segments",
        model_name=model_name,
        speaker_ref=speaker_ref,
        language=language,
    )
    assemble_audio_from_segments(segments, output_path)


if __name__ == "__main__":
    main()
