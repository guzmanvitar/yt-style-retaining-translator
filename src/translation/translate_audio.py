"""
TTS Inference and Audio Assembly Pipeline.

This script performs text-to-speech (TTS) inference using a translated transcription CSV,
generates audio for each segment using an XTTS model, aligns each generated clip using
WhisperX to trim trailing artifacts, and stitches all segments together with silence
alignment. Segments are only compressed if they would overlap the next.
"""

from pathlib import Path

import click
import pandas as pd
import torch
import whisperx
from pydub import AudioSegment

from src.constants import DATA_INFERENCE
from src.logger_definition import get_logger
from src.tts.xtts_inference import run_inference

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

    last_word = words[-1]
    if "end" not in last_word or not isinstance(last_word["end"], (float, int)):
        logger.warning(
            f"Last word has no 'end' timestamp in {audio_path.name}: {last_word}"
        )
        return None

    safe_trim_end = min(last_word["end"] + trim_padding, audio_duration)

    return safe_trim_end


def run_segment_inference(
    csv_path: Path,
    output_dir_name: str = "tts_segments",
    model_name: str = "production_latest",
    speaker_ref: str = "ref_en",
    language: str = "es",
    sentence_buffer_sec: float = 0.5,
    n_variants_per_sentence: int = 2,
) -> None:
    """
    Run XTTS inference on each translated segment, using sentence splitting and alignment cleanup.

    For each row in the CSV:
    - Split the text on periods into sentences.
    - For each sentence:
        - Generate `n_variants_per_sentence` XTTS outputs.
        - Trim each using WhisperX alignment to remove trailing artifacts.
        - Keep the shortest successfully aligned result.
        - If none align, fall back to the shortest untrimmed variant.
    - Join all trimmed sentences with `sentence_buffer_sec` of silence.
    - Save final segment audio to WAV.

    Args:
        csv_path (Path): CSV file with 'text', 'start', 'end' fields.
        output_dir_name (str): Folder under `data/inference` to save outputs.
        model_name (str): Name of XTTS model folder.
        speaker_ref (str): Speaker reference filename (without extension).
        language (str): Target language code (e.g., 'es').
        sentence_buffer_sec (float): Silence in seconds between joined sentence clips.
        n_variants_per_sentence (int): Number of XTTS generations per sentence (default: 2).
    """
    df = pd.read_csv(csv_path)
    output_dir = DATA_INFERENCE / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, row in df.iterrows():
        text = row["text"].strip(".")
        output_name = f"segment_{i:04d}"
        output_path = output_dir / f"{output_name}.wav"

        if output_path.exists():
            logger.info(f"Skipping existing segment: {output_name}.wav")
            continue

        # Split into sentences (naive period split)
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        sentence_clips = []
        buffer = AudioSegment.silent(duration=int(sentence_buffer_sec * 1000))

        for j, sentence in enumerate(sentences):
            shortest_audio: AudioSegment | None = None
            shortest_duration = float("inf")
            shortest_aligned_audio: AudioSegment | None = None
            shortest_aligned_duration = float("inf")

            for k in range(n_variants_per_sentence):
                temp_name = f"{output_name}_part_{j}_try_{k}"
                temp_path = output_dir / f"{temp_name}.wav"

                run_inference(
                    text=sentence,
                    output_filename=temp_name,
                    output_folder=output_dir_name,
                    model_name=model_name,
                    language=language,
                    speafer_ref=speaker_ref,
                )

                audio = AudioSegment.from_wav(temp_path).set_channels(1)
                duration = len(audio)

                # Always track the shortest untrimmed version
                if duration < shortest_duration:
                    shortest_audio = audio
                    shortest_duration = duration

                # Try trimming via WhisperX
                last_word_end = align_last_word_timestamp(
                    temp_path, sentence, language=language
                )
                if last_word_end:
                    aligned_audio = audio[: int(last_word_end * 1000)]
                    aligned_duration = len(aligned_audio)
                    if aligned_duration < shortest_aligned_duration:
                        shortest_aligned_audio = aligned_audio
                        shortest_aligned_duration = aligned_duration

                temp_path.unlink()

            if shortest_aligned_audio:
                sentence_clips.append(shortest_aligned_audio)
            else:
                logger.warning(
                    f"No alignment succeeded for sentence {j} in segment {output_name}, using "
                    "untrimmed."
                )
                if shortest_audio:
                    sentence_clips.append(shortest_audio)

        if not sentence_clips:
            logger.warning(f"No audio generated for segment {i}, skipping.")
            continue

        # Join all sentence clips with silence buffer
        full_audio = sentence_clips[0]
        for clip in sentence_clips[1:]:
            full_audio += buffer + clip

        full_audio.export(output_path, format="wav")
        logger.info(f"Generated {output_name}.wav ({len(full_audio) / 1000:.2f}s)")


@click.command()
@click.option(
    "--csv-path",
    type=Path,
    default=DATA_INFERENCE / "translated_segments.csv",
    help="Path to translated CSV with start/end/timestamps.",
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
def main(csv_path, model_name, speaker_ref, language):
    """Run XTTS inference with alignment-based cleanup and stitch audio with silence alignment."""
    logger.info("Starting XTTS inference with WhisperX alignment...")
    run_segment_inference(
        csv_path=csv_path,
        output_dir_name="tts_segments",
        model_name=model_name,
        speaker_ref=speaker_ref,
        language=language,
    )


if __name__ == "__main__":
    main()
