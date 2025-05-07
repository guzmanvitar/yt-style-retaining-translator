"""
TTS Inference and Audio Assembly Pipeline.

This script performs text-to-speech (TTS) inference using a translated transcription CSV,
generates audio for each segment using an XTTS model, aligns each generated clip using
WhisperX to trim trailing artifacts, and stitches all segments together with silence
alignment. Segments are only compressed if they would overlap the next.
"""

from pathlib import Path
from typing import List, Tuple
import pandas as pd
from pydub import AudioSegment
import click
import torch
import whisperx

from src.tts.xtts_inference import run_inference
from src.constants import DATA_INFERENCE
from src.logger_definition import get_logger

logger = get_logger(__file__)


def align_last_word_timestamp(
    audio_path: Path,
    text: str,
    language: str = "es",
    alignment_padding: float = 0.5,
    trim_padding: float = 0.15,
) -> float | None:
    """
    Align known text to audio using WhisperX and return a padded end time for trimming.

    Args:
        audio_path (Path): Path to WAV file.
        text (str): Transcript to align.
        language (str): Language code (e.g., 'es').
        alignment_padding (float): Extra time (seconds) added to alignment window.
        trim_padding (float): Extra time (seconds) added after last word for trimming.

    Returns:
        float | None: Safe end time (in seconds) to trim the TTS audio.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    audio = AudioSegment.from_wav(audio_path)
    audio_duration = len(audio) / 1000
    padded_alignment_end = audio_duration + alignment_padding

    fake_transcript = {
        "language": language,
        "segments": [{"start": 0.0, "end": padded_alignment_end, "text": text}],
    }

    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
    aligned_result = whisperx.align(
        fake_transcript["segments"],
        model_a,
        metadata,
        str(audio_path),
        device,
    )

    words = aligned_result.get("word_segments", [])
    if not words:
        logger.warning(f"No words aligned in {audio_path.name}")
        return None

    last_word_end = words[-1]["end"]
    safe_trim_end = min(last_word_end + trim_padding, audio_duration)

    return safe_trim_end


def run_segment_inference(
    csv_path: Path,
    output_dir_name: str = "tts_segments",
    model_name: str = "production_latest",
    speaker_ref: str = "ref_en",
    language: str = "es",
) -> List[Tuple[float, float, float, Path]]:
    """
    Run TTS inference on each translated segment and trim artifacts using WhisperX.

    Args:
        csv_path (Path): Translated CSV with start, end, text.
        output_dir_name (str): Subfolder for segment WAVs.
        model_name (str): XTTS model name.
        speaker_ref (str): Reference speaker name (no .wav).
        language (str): Target language.

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

        # Trim trailing non-word noise using WhisperX alignment
        last_word_end = align_last_word_timestamp(output_path, text, language=language)
        if last_word_end:
            audio = audio[: int(last_word_end * 1000)]
            audio.export(output_path, format="wav")
            logger.debug(f"Trimmed to {last_word_end:.2f}s using WhisperX")

        actual_duration = len(audio) / 1000
        segments.append((start, end, actual_duration, output_path))
        logger.info(f"Generated {output_name}.wav ({actual_duration:.2f}s)")

    return segments


def assemble_audio_from_segments(
    segments: List[Tuple[float, float, float, Path]], output_path: Path
) -> None:
    """
    Assemble final audio with silence and minimal compression for overlapping clips.

    Args:
        segments (List[Tuple[start, end, actual_duration, Path]]): Audio segments.
        output_path (Path): Output WAV file path.
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
                target_duration = next_start - start
                speed_factor = actual_duration / target_duration
                logger.warning(
                    f"Segment {i} overlaps next, speeding up x{speed_factor:.2f}"
                )
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
    logger.info(f"Final stitched audio saved to {output_path}")


@click.command()
@click.option(
    "--csv-path",
    type=Path,
    default=DATA_INFERENCE / "translated_segments.csv",
    help="Path to translated CSV with start/end/timestamps.",
)
@click.option(
    "--output-path",
    type=Path,
    default=DATA_INFERENCE / "final_audio.wav",
    help="Path to save the final stitched WAV.",
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
    help="Reference speaker WAV filename (no extension).",
)
@click.option(
    "--language",
    type=str,
    default="es",
    help="Target language code (e.g., 'es').",
)
def main(csv_path, output_path, model_name, speaker_ref, language):
    """Run XTTS inference with alignment-based cleanup and stitch audio with silence alignment."""
    logger.info("Starting XTTS inference with WhisperX alignment...")
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
