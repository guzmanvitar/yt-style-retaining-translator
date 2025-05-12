"""
Final Audio Stitching with Temporal Alignment.

This script recomposes a full audio track from translated speech segments using original timestamps.
It ensures alignment by:
  - Adding silence if possible
  - Accelerating segments (max 20%) using pyrubberband
  - Logging segments that require video slowdown to maintain sync

Outputs:
  - Final stitched WAV (default: DATA_FINAL/final_audio.wav)
  - JSON log of slowdown-required segments (default: DATA_FINAL/video_slowdowns.json)
"""

import json
from pathlib import Path

import click
import librosa
import pandas as pd
import pyrubberband as pyrb
import soundfile as sf
from pydub import AudioSegment

from src.constants import DATA_FINAL, DATA_INFERENCE
from src.logger_definition import get_logger

logger = get_logger(__file__)


def load_audio(path: Path) -> tuple:
    """Load audio file as numpy array and sample rate."""
    y, sr = librosa.load(path, sr=None)
    return y, sr


def accelerate_audio(y, sr, rate: float) -> AudioSegment:
    """Stretch audio using pyrubberband and return as AudioSegment."""
    y_fast = pyrb.time_stretch(y, sr, rate=rate)
    buffer = Path("temp_stretched.wav")
    sf.write(buffer, y_fast, sr)
    audio = AudioSegment.from_wav(buffer)
    buffer.unlink()
    return audio


@click.command()
@click.option(
    "--segment-dir",
    type=Path,
    default=DATA_INFERENCE / "tts_segments",
    help="Directory containing the individual TTS segment WAV files.",
)
@click.option(
    "--csv-path",
    type=Path,
    default=DATA_INFERENCE / "translated_segments.csv",
    help="CSV file with original start timestamps for alignment.",
)
@click.option(
    "--output-audio",
    type=Path,
    default=DATA_FINAL / "final_audio.wav",
    help="Path to save the final stitched WAV file.",
)
@click.option(
    "--slowdown-json",
    type=Path,
    default=DATA_FINAL / "video_slowdowns.json",
    help="Path to save the slowdown JSON metadata.",
)
@click.option(
    "--max-acceleration",
    type=float,
    default=1.2,
    help="Maximum acceleration rate (e.g., 1.2 = up to 20% faster).",
)
@click.option(
    "--buffer",
    "buffer_silence_sec",
    type=float,
    default=0.5,
    help="Silence (in seconds) to add before accelerated segments (default: 0.5).",
)
def stitch_audio_with_alignment(
    segment_dir: Path,
    csv_path: Path,
    output_audio: Path,
    slowdown_json: Path,
    max_acceleration: float,
    buffer_silence_sec: float,
):
    """
    Stitch translated audio segments to match original video timing,
    using acceleration or video slowdown when necessary.

    Strategy:
    - Always enforce a minimum buffer between segments.
    - Fit as-is if it fits with the buffer.
    - Shift earlier into pre-silence if needed.
    - Accelerate to fit inside pre + post + buffer space.
    - If still too long, use max acceleration and log required video slowdown.

    Args:
        segment_dir (Path): Folder with WAV segments (segment_0001.wav, etc.).
        csv_path (Path): CSV with 'start', 'end', and 'text' columns.
        output_audio (Path): Path to save final stitched WAV.
        slowdown_json (Path): Path to write video slowdown metadata JSON.
        max_acceleration (float): Max allowed speed-up factor (e.g. 1.2 = 20%).
        buffer_silence_sec (float): Required silence between segments (in seconds).
    """
    df = pd.read_csv(csv_path)
    df["start"] = df["start"].astype(float)
    df["end"] = df["end"].astype(float)

    final_audio = AudioSegment.silent(duration=0)
    timeline_cursor = float(df.iloc[0]["start"])
    slowdowns = []

    for i, row in df.iterrows():
        start = row["start"]
        end = row["end"]
        path = segment_dir / f"segment_{i:04d}.wav"

        if not path.exists():
            logger.warning(f"Missing segment: {path}")
            continue

        audio = AudioSegment.from_wav(path).set_channels(1)
        actual_duration = len(audio) / 1000  # seconds
        next_start = df.iloc[i + 1]["start"] if i + 1 < len(df) else end

        # We require at least this much space between segments
        required_buffer = buffer_silence_sec

        available_slot = next_start - start - required_buffer
        pre_silence = max(0.0, start - timeline_cursor)
        extended_room = pre_silence + available_slot

        silence_buffer = AudioSegment.silent(duration=int(required_buffer * 1000))

        if actual_duration <= available_slot:
            if pre_silence > 0:
                final_audio += AudioSegment.silent(duration=int(pre_silence * 1000))
            final_audio += audio + silence_buffer
            timeline_cursor = start + actual_duration + required_buffer

        elif actual_duration <= extended_room:
            shift = min(pre_silence, actual_duration - available_slot)
            silence_before = pre_silence - shift

            logger.info(f"Shifting segment {i} earlier by {shift:.2f}s")

            if silence_before > 0:
                final_audio += AudioSegment.silent(duration=int(silence_before * 1000))
            final_audio += audio + silence_buffer
            timeline_cursor = len(final_audio) / 1000

        else:
            speedup_rate = actual_duration / extended_room
            if speedup_rate <= max_acceleration:
                logger.info(
                    f"Accelerating segment {i} x{speedup_rate:.2f} to fit in extended room"
                )
                y, sr = load_audio(path)
                accelerated = accelerate_audio(y, sr, rate=speedup_rate)
                final_audio += silence_buffer + accelerated
                timeline_cursor = len(final_audio) / 1000
            else:
                logger.info(
                    f"Segment {i} too long for room, applying max acceleration and logging slowdown"
                )
                y, sr = load_audio(path)
                accelerated = accelerate_audio(y, sr, rate=max_acceleration)
                final_audio += silence_buffer + accelerated

                accelerated_duration = len(accelerated) / 1000
                required_duration = (
                    actual_duration / max_acceleration + buffer_silence_sec
                )
                slowdown_factor = round(required_duration / extended_room, 3)

                slowdowns.append(
                    {
                        "segment": f"segment_{i:04d}.wav",
                        "start": start,
                        "expected_end": next_start,
                        "original_duration": actual_duration,
                        "used_acceleration": max_acceleration,
                        "post_accelerated_duration": accelerated_duration,
                        "needed_video_slowdown_factor": slowdown_factor,
                    }
                )
                timeline_cursor = len(final_audio) / 1000

    final_audio.export(output_audio, format="wav")
    logger.info(f"Final audio saved to: {output_audio}")

    with open(slowdown_json, "w", encoding="utf-8") as f:
        json.dump(slowdowns, f, indent=2)
    logger.info(f"Slowdown info saved to: {slowdown_json}")


if __name__ == "__main__":
    stitch_audio_with_alignment()
