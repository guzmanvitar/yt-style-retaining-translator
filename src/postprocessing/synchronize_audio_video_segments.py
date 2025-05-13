"""
Aligns translated audio segments to original video timing and extracts synchronized video clips.
Applies acceleration or slowdown as needed, and saves audio-video pairs for debugging or final
assembly.
"""

import shutil
import subprocess
from pathlib import Path

import click
import librosa
import pandas as pd
import pyrubberband as pyrb
import soundfile as sf
from pydub import AudioSegment

from src.constants import DATA_FINAL, DATA_INFERENCE, DATA_RAW
from src.logger_definition import get_logger

logger = get_logger(__file__)


def load_audio(path: Path) -> tuple:
    y, sr = librosa.load(path, sr=None)
    return y, sr


def accelerate_audio(y, sr, rate: float) -> AudioSegment:
    y_fast = pyrb.time_stretch(y, sr, rate=rate)
    buffer = Path("temp_stretched.wav")
    sf.write(buffer, y_fast, sr)
    audio = AudioSegment.from_wav(buffer)
    buffer.unlink()
    return audio


def extract_video_segment(
    input_path: Path,
    start: float,
    end: float,
    output_path: Path,
    slowdown_factor: float | None = None,
):
    """Extract a video segment with duration matching the audio precisely."""
    vf_filter = "fps=30"
    if slowdown_factor:
        vf_filter = f"setpts={slowdown_factor}*PTS,fps=30"

    duration = end - start
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start),
        "-t",
        str(duration),
        "-i",
        str(input_path),
        "-vf",
        vf_filter,
        "-an",
        str(output_path),
    ]
    logger.info(
        f"Extracting video segment: {start:.2f}s for {duration:.3f}s (slowdown={slowdown_factor})"
    )
    subprocess.run(cmd, check=True)


def get_video_duration(video_path: Path) -> float:
    """
    Return the duration of a video in seconds using ffprobe.

    Args:
        video_path (Path): Path to the input video file.

    Returns:
        float: Duration of the video in seconds.
    """
    import ffmpeg

    probe = ffmpeg.probe(str(video_path))
    return float(probe["format"]["duration"])


def align_and_export_segments(
    segment_dir: Path,
    csv_path: Path,
    video_dir: Path,
    video_name: str | None,
    output_dir: Path,
    max_acceleration: float,
    buffer_silence_sec: float,
):
    """
    Aligns translated audio segments with original video timing, applies acceleration or slowdown
    if needed, and exports synchronized audio-video segment pairs for each phrase.

    For each segment:
    - Adds silence before/after to match the original speech timing
    - Accelerates or clips audio to fit available space when required
    - Exports both the processed audio and a precisely aligned video clip
    - Clips video segments to stay within the original video duration, logging if adjustments exceed
        200ms

    Also generates a stitched audio file for the entire sequence.

    Args:
        segment_dir (Path): Directory containing TTS audio segments (e.g., segment_0001.wav).
        csv_path (Path): Path to CSV with segment timing metadata ('start', 'end').
        video_dir (Path): Directory containing the source video.
        video_name (str | None): Name of the original video file, or None to auto-detect first .mp4.
        output_dir (Path): Directory to save aligned segment outputs.
        max_acceleration (float): Maximum allowed acceleration factor for audio fitting
            (e.g., 1.2 = 20% faster).
        buffer_silence_sec (float): Silence buffer to append after each segment (in seconds).
    """
    df = pd.read_csv(csv_path)
    df["start"] = df["start"].astype(float)
    df["end"] = df["end"].astype(float)

    if video_name:
        video_path = video_dir / video_name
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
    else:
        video_path = next(video_dir.glob("*.mp4"), None)
        if not video_path:
            raise FileNotFoundError("No .mp4 video found in the directory.")

    video_duration = get_video_duration(video_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_audio = AudioSegment.silent(duration=0)
    timeline_cursor = 0.0

    for i, row in df.iterrows():
        start = row["start"]
        end = row["end"]
        next_start = df.iloc[i + 1]["start"] if i + 1 < len(df) else end
        path = segment_dir / f"segment_{i:04d}.wav"

        if not path.exists():
            logger.warning(f"Missing segment: {path}")
            continue

        audio = AudioSegment.from_wav(path).set_channels(1)
        y_raw, sr_raw = sf.read(path)
        actual_duration = len(y_raw) / sr_raw

        required_buffer = buffer_silence_sec
        pre_silence = max(0.0, start - timeline_cursor)
        available_slot = next_start - start - required_buffer
        extended_room = pre_silence + available_slot

        audio_out_path = output_dir / f"segment_{i:04d}_audio.wav"
        video_out_path = output_dir / f"segment_{i:04d}_video.mp4"
        slowdown_factor = None

        current_segment = AudioSegment.silent(duration=0)

        if actual_duration <= available_slot:
            if pre_silence > 0:
                current_segment += AudioSegment.silent(duration=int(pre_silence * 1000))
            current_segment += audio + AudioSegment.silent(
                duration=int(required_buffer * 1000)
            )

        elif actual_duration <= extended_room:
            shift = min(pre_silence, actual_duration - available_slot)
            silence_before = pre_silence - shift
            if silence_before > 0:
                current_segment += AudioSegment.silent(
                    duration=int(silence_before * 1000)
                )
            current_segment += audio + AudioSegment.silent(
                duration=int(required_buffer * 1000)
            )

        else:
            speedup_rate = actual_duration / extended_room
            if speedup_rate <= max_acceleration:
                y, sr = load_audio(path)
                accelerated = accelerate_audio(y, sr, rate=speedup_rate)
                current_segment += accelerated + AudioSegment.silent(
                    duration=int(required_buffer * 1000)
                )
            else:
                y, sr = load_audio(path)
                accelerated = accelerate_audio(y, sr, rate=max_acceleration)
                current_segment += accelerated + AudioSegment.silent(
                    duration=int(required_buffer * 1000)
                )
                slowdown_factor = round(
                    (actual_duration / max_acceleration + required_buffer)
                    / extended_room,
                    3,
                )

        # Export and measure precise audio duration
        current_segment.export(audio_out_path, format="wav")
        y_final, sr_final = sf.read(audio_out_path)
        segment_duration = len(y_final) / sr_final
        video_end_time = timeline_cursor + segment_duration

        # Clip to actual video duration if needed
        clipped_video_end_time = min(video_end_time, video_duration)
        clip_diff = video_end_time - clipped_video_end_time
        if clip_diff > 0.2:
            logger.warning(
                f"Segment {i:04d}: Clipped video_end_time by {clip_diff:.3f}s to fit inside video"
                " bounds."
            )

        extract_video_segment(
            video_path,
            timeline_cursor,
            clipped_video_end_time,
            video_out_path,
            slowdown_factor,
        )

        final_audio += current_segment
        timeline_cursor = clipped_video_end_time

    stitched_audio_path = output_dir / "final_audio.wav"
    final_audio.export(stitched_audio_path, format="wav")
    logger.info(f"Stitched audio saved to: {stitched_audio_path}")


def concat_and_merge(output_dir: Path, cleanup: bool = True):
    """Concatenate aligned video segments and merge with final audio."""
    segment_list = sorted(output_dir.glob("segment_*.mp4"))
    concat_file = output_dir / "concat_list.txt"
    with open(concat_file, "w") as f:
        for segment in segment_list:
            f.write(f"file '{segment.resolve()}'\n")

    concatenated_video = output_dir / "temp_video.mp4"
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

    final_audio = output_dir / "final_audio.wav"
    final_video = DATA_FINAL / "final_video.mp4"
    cmd_merge = [
        "ffmpeg",
        "-y",
        "-i",
        str(concatenated_video),
        "-i",
        str(final_audio),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(final_video),
    ]
    subprocess.run(cmd_merge, check=True)
    logger.info(f"Final video saved to: {final_video}")

    if cleanup:
        shutil.rmtree(output_dir)
        logger.info(f"Removed segment folder: {output_dir}")


@click.command()
@click.option("--segment-dir", type=Path, default=DATA_INFERENCE / "tts_segments")
@click.option(
    "--csv-path", type=Path, default=DATA_INFERENCE / "translated_segments.csv"
)
@click.option(
    "--video-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=DATA_RAW / "videos",
)
@click.option("--video-name", type=str, default=None)
@click.option("--output-dir", type=Path, default=DATA_FINAL / "aligned_segments")
@click.option("--max-acceleration", type=float, default=1.25)
@click.option("--buffer", "buffer_silence_sec", type=float, default=0.5)
@click.option(
    "--merge/--no-merge",
    default=True,
    help="If set, merge video segments and audio after alignment (default: True).",
)
@click.option(
    "--cleanup/--no-cleanup",
    default=True,
    help="Delete aligned segment files after merging (default: True).",
)
def main(
    segment_dir,
    csv_path,
    video_dir,
    video_name,
    output_dir,
    max_acceleration,
    buffer_silence_sec,
    merge,
    cleanup,
):
    align_and_export_segments(
        segment_dir=segment_dir,
        csv_path=csv_path,
        video_dir=video_dir,
        video_name=video_name,
        output_dir=output_dir,
        max_acceleration=max_acceleration,
        buffer_silence_sec=buffer_silence_sec,
    )

    if merge:
        concat_and_merge(output_dir, cleanup=cleanup)


if __name__ == "__main__":
    main()
