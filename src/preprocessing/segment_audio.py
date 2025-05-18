"""Transcribe and segment WAV audio into aligned chunks using the WhisperX Python API."""

from pathlib import Path
from typing import Any

import click
import pandas as pd
import torch
import whisperx
from pydub import AudioSegment, silence

from src.constants import DATA_PROCESSED
from src.logger_definition import get_logger

logger = get_logger(__file__)


def chunk_aligned_audio_versions(
    audio_paths: list[Path],
    reference_index: int,
    output_dirs: list[Path],
    base_filename: str,
    target_chunk_duration_ms: int = 5 * 60 * 1000,
    min_silence_len: int = 2000,
    silence_thresh: int = -40,
    keep_silence: int = 1000,
) -> tuple[list[list[Path]], list[float]]:
    """
    Split multiple aligned audio files into chunks using silence detection on a reference track.

    This function divides long audio recordings into smaller chunks based on silence detection
    applied to a designated reference audio. All audio versions are split using the same time
    boundaries to maintain alignment across different sample rates.

    Args:
        audio_paths (list[Path]): List of paths to aligned input audio files.
        reference_index (int): Index of the audio used for silence detection.
        output_dirs (list[Path]): Output directories where each aligned version will be stored.
        base_filename (str): Filename prefix for output chunks.
        target_chunk_duration_ms (int): Maximum chunk length in milliseconds (default: 5 minutes).
        min_silence_len (int): Minimum silence length (ms) to be considered a split point.
        silence_thresh (int): Silence threshold in dBFS.
        keep_silence (int): Milliseconds of silence to retain at each split boundary.

    Returns:
        tuple[list[list[Path]], list[float]]: A tuple containing:
            - A list of chunk path lists (one list per chunk, with aligned audio paths).
            - A list of chunk start times in seconds relative to the full audio.
    """

    if len(audio_paths) != len(output_dirs):
        raise ValueError("audio_paths and output_dirs must have the same length")

    audios = [AudioSegment.from_wav(path).set_channels(1) for path in audio_paths]
    ref_audio = audios[reference_index]

    total = len(ref_audio)
    start = 0
    chunk_index = 0
    chunk_groups = []
    chunk_start_times = []

    for d in output_dirs:
        d.mkdir(parents=True, exist_ok=True)

    while start < total:
        end = min(start + target_chunk_duration_ms, total)
        segment = ref_audio[start:end]

        silence_ranges = silence.detect_silence(
            segment,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            seek_step=10,
        )

        cut_point = end
        for _, s_end in reversed(silence_ranges):
            if s_end < target_chunk_duration_ms - 5000:
                cut_point = start + s_end
                break

        cut_start = max(start, cut_point - keep_silence)
        chunk_paths = []

        for i, audio in enumerate(audios):
            chunk = audio[start:cut_start]
            chunk_name = f"{base_filename}_chunk_{chunk_index:03}.wav"
            chunk_path = output_dirs[i] / chunk_name
            chunk.export(chunk_path, format="wav")
            chunk_paths.append(chunk_path)

        chunk_groups.append(chunk_paths)
        chunk_start_times.append(start / 1000)
        chunk_index += 1
        start = cut_point

    return chunk_groups, chunk_start_times


def transcribe_and_align(
    audio_path: Path, model_size: str = "large-v3", language: str = "en"
) -> list[dict[str, Any]]:
    """
    Transcribe and align a WAV audio file using WhisperX.

    Args:
        audio_path (Path): Path to input WAV file.
        model_size (str): Whisper model name (e.g., "base", "large-v3").
        language (str): Language code to force (default: "en").

    Returns:
        list[dict[str, Any]]: List of segment dictionaries with start, end, and text.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Transcribing %s with language=%s...", audio_path.name, language)

    model = whisperx.load_model(model_size, device, compute_type="int8")
    result = model.transcribe(str(audio_path), language=language)

    logger.info("Aligning timestamps...")
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
    result_aligned = whisperx.align(
        result["segments"], model_a, metadata, str(audio_path), device
    )

    return result_aligned["segments"]


def split_segment_on_silence(
    audio: AudioSegment,
    start: float,
    end: float,
    max_duration: float = 15.0,
    min_silence_len: int = 300,
    silence_thresh: int = -40,
    keep_silence: int = 150,
) -> list[tuple[float, float]]:
    """
    Recursively split a long audio segment into shorter ones using silence detection.

    This function identifies silent regions within the audio and attempts to split the segment
    such that each resulting subsegment is no longer than `max_duration` seconds. If no suitable
    silence is found, it falls back to splitting the segment in half.

    Args:
        audio (AudioSegment): The full audio clip containing the segment.
        start (float): Start time of the segment in seconds.
        end (float): End time of the segment in seconds.
        max_duration (float): Maximum allowed duration of a subsegment in seconds.
        min_silence_len (int): Minimum silence length in milliseconds to consider a split point.
        silence_thresh (int): Silence threshold in dBFS.
        keep_silence (int): Not used in current logic, included for future compatibility.

    Returns:
        List[tuple[float, float]]: List of (start, end) pairs for each subsegment in seconds.
    """

    if end - start <= max_duration:
        return [(start, end)]

    segment_audio = audio[int(start * 1000) : int(end * 1000)]
    silences = silence.detect_silence(
        segment_audio, min_silence_len, silence_thresh, seek_step=5
    )

    if silences:
        best_split = None
        deviation = float("inf")
        for s_start, _ in silences:
            candidate = start + s_start / 1000
            left = candidate - start
            right = end - candidate
            if left <= max_duration and right <= max_duration:
                d = abs((end - start) / 2 - left)
                if d < deviation:
                    deviation = d
                    best_split = candidate

        if best_split:
            return split_segment_on_silence(
                audio, start, best_split
            ) + split_segment_on_silence(audio, best_split, end)

    # fallback: cut in half
    mid = (start + end) / 2
    return split_segment_on_silence(audio, start, mid) + split_segment_on_silence(
        audio, mid, end
    )


def segment_audio(
    audio_path: Path,
    segments: list[dict[str, Any]],
    chunks_dir: Path,
    csv_path: Path,
    global_offset: float = 0.0,
    model_size: str = "large-v3",
) -> None:
    """
    Segment and align audio into smaller clips, with re-alignment for long segments.

    For each segment, this function extracts and saves a WAV clip. Segments longer than
    15 seconds are split on silence and re-aligned using WhisperX to generate cleaner
    transcriptions and accurate timestamps. Segment metadata is saved to a CSV file.

    Args:
        audio_path (Path): Path to the original WAV audio file.
        segments (list[dict]): List of dictionaries with 'start', 'end', and 'text' fields.
        chunks_dir (Path): Directory where output audio chunks will be saved.
        csv_path (Path): Path to save the resulting segment metadata CSV.
        global_offset (float): Time offset (in seconds) to apply to all timestamps.
        model_size (str): Whisper model size for re-alignment (default: "large-v3").
    """
    audio = AudioSegment.from_wav(audio_path)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    counter = 0

    for seg in segments:
        duration = seg["end"] - seg["start"]
        if duration <= 15.0:
            start = seg["start"] + global_offset
            end = seg["end"] + global_offset
            start_ms = int(seg["start"] * 1000)
            end_ms = int(seg["end"] * 1000)

            filename = f"{audio_path.stem}_segment_{counter:03}.wav"
            final_path = chunks_dir / filename
            audio[start_ms:end_ms].export(final_path, format="wav")

            rows.append(
                {
                    "filename": filename,
                    "start": start,
                    "end": end,
                    "text": seg["text"].strip(),
                }
            )
            counter += 1
        else:
            sub_times = split_segment_on_silence(
                audio, seg["start"], seg["end"], max_duration=15.0
            )
            for sub_start, sub_end in sub_times:
                temp_path = chunks_dir / f"tmp_realign_{counter}.wav"
                audio[int(sub_start * 1000) : int(sub_end * 1000)].export(
                    temp_path, format="wav"
                )

                sub_segments = transcribe_and_align(temp_path, model_size=model_size)
                text = " ".join(
                    s["text"]
                    for s in sub_segments
                    if s["start"] is not None and s["end"] is not None
                )

                if sub_segments:
                    start = sub_segments[0]["start"] + sub_start + global_offset
                    end = sub_segments[-1]["end"] + sub_start + global_offset
                else:
                    start = sub_start + global_offset
                    end = sub_end + global_offset

                filename = f"{audio_path.stem}_segment_{counter:03}.wav"
                final_path = chunks_dir / filename
                audio[int(sub_start * 1000) : int(sub_end * 1000)].export(
                    final_path, format="wav"
                )

                rows.append(
                    {
                        "filename": filename,
                        "start": start,
                        "end": end,
                        "text": text.strip(),
                    }
                )
                counter += 1

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info("Saved segment CSV to %s", csv_path)

    # Cleanup temporary files
    for f in chunks_dir.glob("tmp_realign_*.wav"):
        try:
            f.unlink()
            logger.debug(f"Deleted temporary file: {f}")
        except OSError as e:
            logger.warning(f"Failed to delete temp file {f}: {e}")


@click.command()
@click.option(
    "--model",
    type=str,
    default="large-v3",
    help="Whisper model to use for transcription (default: large-v3).",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=DATA_PROCESSED / "segments",
    help="Directory to store segmented chunks and CSVs (default: data/processed/segments).",
)
def main(model: str, output_dir: Path) -> None:
    """
    Orchestrate audio chunking, transcription, segmentation, and re-alignment.

    This processes 16kHz and 22kHz aligned audio pairs:
    - Chunks audio based on silence in the 22kHz version.
    - Transcribes 16kHz chunks with WhisperX.
    - Segments and re-aligns transcriptions.
    - Outputs clean clips and metadata.
    """
    input_dir = DATA_PROCESSED / "16000hz"
    segment_source_dir = DATA_PROCESSED / "22050hz"
    temp_base = Path("tmp/segments")
    temp_16k = temp_base / "16k"
    temp_22k = temp_base / "22k"

    for audio_path in input_dir.glob("*.wav"):
        base_name = audio_path.stem.replace("_16000", "")
        segment_path = segment_source_dir / f"{base_name}_22050.wav"

        if not segment_path.exists():
            logger.warning("Missing 22050Hz audio for %s", base_name)
            continue

        chunk_groups, chunk_offsets = chunk_aligned_audio_versions(
            [audio_path, segment_path],
            reference_index=1,
            output_dirs=[temp_16k, temp_22k],
            base_filename=base_name,
        )

        for (chunk_16k, chunk_22k), offset in zip(chunk_groups, chunk_offsets):
            segments = transcribe_and_align(chunk_16k, model_size=model)
            csv_path = output_dir / f"{chunk_22k.stem}_segments.csv"
            segment_audio(
                chunk_22k,
                segments,
                output_dir / "chunks",
                csv_path,
                global_offset=offset,
                model_size=model,
            )


if __name__ == "__main__":
    main()
