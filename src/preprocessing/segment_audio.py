"""Transcribe and segment WAV audio into aligned chunks using the WhisperX Python API."""

import argparse
from pathlib import Path
from typing import Any

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
) -> list[list[Path]]:
    """
    Split multiple aligned audio versions using silence detection from one reference version.

    Args:
        audio_paths (List[Path]): Paths to aligned input audio files (e.g., 16k, 22k versions).
        reference_index (int): Index of the audio file to use for silence detection.
        output_dirs (List[Path]): Output directories for each audio version (must match length of
            audio_paths).
        base_filename (str): Stem for naming chunk files.
        target_chunk_duration_ms (int): Approx. max length per chunk.
        min_silence_len (int): Minimum silence to detect (ms).
        silence_thresh (int): Silence threshold (dBFS).
        keep_silence (int): Milliseconds of silence to preserve at each split point.


    Returns:
        List[List[Path]]: One list per chunk, each containing the paths of chunked audio across
            versions.
    """
    if len(audio_paths) != len(output_dirs):
        raise ValueError("audio_paths and output_dirs must have the same length")

    audios = [AudioSegment.from_wav(path).set_channels(1) for path in audio_paths]
    ref_audio = audios[reference_index]

    total = len(ref_audio)
    start = 0
    chunk_index = 0
    chunk_groups = []

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
        chunk_index += 1
        start = cut_point

    return chunk_groups


def transcribe_and_align(
    audio_path: Path, model_size: str = "large-v3"
) -> list[dict[str, Any]]:
    """
    Transcribe and align a WAV audio file using WhisperX.

    Args:
        audio_path (Path): Path to input WAV file.
        model_size (str): Whisper model name (e.g., "base", "large-v3").

    Returns:
        list[dict[str, Any]]: List of segment dictionaries with start, end, and text.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading WhisperX model: %s on %s", model_size, device)

    model = whisperx.load_model(model_size, device, compute_type="int8")

    logger.info("Transcribing %s...", audio_path.name)
    result = model.transcribe(str(audio_path))

    logger.info("Aligning timestamps...")
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result_aligned = whisperx.align(
        result["segments"], model_a, metadata, str(audio_path), device
    )

    return result_aligned["segments"]


def segment_audio(
    audio_path: Path, segments: list[dict[str, Any]], chunks_dir: Path, csv_path: Path
) -> None:
    """
    Slice audio into segments and save them as individual WAV files with metadata.

    Args:
        audio_path (Path): Path to original audio file.
        segments (list): List of aligned segments with timestamps and text.
        chunks_dir (Path): Directory to store audio chunks.
        csv_path (Path): Path to write the CSV metadata.
    """
    audio = AudioSegment.from_wav(audio_path)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, segment in enumerate(segments):
        start_ms = int(segment["start"] * 1000)
        end_ms = int(segment["end"] * 1000)

        filename = f"{audio_path.stem}_segment_{i:03}.wav"
        chunk_path = chunks_dir / filename

        chunk_audio = audio[start_ms:end_ms]
        chunk_audio.export(chunk_path, format="wav")

        logger.info(
            "Saved chunk: %s (%.2fs → %.2fs)",
            filename,
            segment["start"],
            segment["end"],
        )
        rows.append(
            {
                "filename": filename,
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip(),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    logger.info("Saved segment CSV to %s", csv_path)


def main() -> None:
    """Transcribe long audio files in chunks using WhisperX and segment aligned 22.05kHz audio."""
    parser = argparse.ArgumentParser(
        description="Segment audio into aligned chunks using WhisperX"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="large-v3",
        help="Whisper model to use (default: large-v3).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_PROCESSED / "segments",
        help="Output base directory (default: data/processed/segments/).",
    )

    args = parser.parse_args()

    input_dir = DATA_PROCESSED / "16000hz"
    segment_source_dir = DATA_PROCESSED / "22050hz"

    audio_files = list(input_dir.glob("*.wav"))
    if not audio_files:
        logger.warning("No WAV files found in %s", input_dir)
        return

    temp_base = Path("tmp/segments")
    temp_16k = temp_base / "16k"
    temp_22k = temp_base / "22k"

    for audio_path in audio_files:
        base_name = audio_path.stem.replace("_16000", "")
        segment_path = segment_source_dir / f"{base_name}_22050.wav"
        if not segment_path.exists():
            logger.error("Missing 22050Hz file for %s", audio_path.name)
            continue

        try:
            chunk_groups = chunk_aligned_audio_versions(
                audio_paths=[audio_path, segment_path],
                reference_index=1,
                output_dirs=[temp_16k, temp_22k],
                base_filename=base_name,
            )
            total_chunks = len(chunk_groups)
            logger.info("Created %d aligned chunk(s) for %s", total_chunks, base_name)

            for idx, (chunk_16k, chunk_22k) in enumerate(chunk_groups, start=1):
                logger.info(
                    "Processing chunk %d/%d: %s", idx, total_chunks, chunk_16k.name
                )
                segments = transcribe_and_align(chunk_16k, model_size=args.model)
                csv_path = args.output_dir / f"{chunk_22k.stem}_segments.csv"
                segment_audio(chunk_22k, segments, args.output_dir / "chunks", csv_path)

        except (KeyError, OSError, RuntimeError) as e:
            logger.error("Segmentation failed for %s: %s", audio_path.name, e)


if __name__ == "__main__":
    main()
