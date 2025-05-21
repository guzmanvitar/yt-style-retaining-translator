"""
TTS Inference and Audio Assembly Pipeline.

This script performs text-to-speech (TTS) inference using a translated transcription CSV,
generates audio for each segment using an XTTS model, aligns each generated clip using
WhisperX to trim trailing artifacts, and stitches all segments together with silence
alignment.
"""

from pathlib import Path

import click
import pandas as pd
import torch
import torchaudio
import whisperx
from pydub import AudioSegment
from TTS.api import TTS

from src.constants import DATA_INFERENCE, MODEL_OUTPUT_PATH
from src.logger_definition import get_logger
from src.tts.xtts_inference import run_inference

logger = get_logger(__file__)


def align_last_word_timestamp(
    alignment_model: torchaudio.models.wav2vec2.model.Wav2Vec2Model,
    alignment_metadata: dict,
    audio_path: Path,
    text: str,
    language: str = "es",
    alignment_padding: float = 0.5,
    trim_padding: float = 0.15,
) -> float | None:
    """
    Align known text to audio using WhisperX and return a padded end time for trimming.

    Args:
        alignment_model (torchaudio.models.wav2vec2.model.Wav2Vec2Model): Instantiated Whisper
            alignment model.
        alignment_metadata (dict): Metadata for alignment model.
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

    aligned_result = whisperx.align(
        fake_transcript["segments"],
        alignment_model,
        alignment_metadata,
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
    tts_model: TTS,
    speaker_ref_path: Path,
    output_dir: Path,
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
        tts_model (TTS.api.TTS): TTS trained model.
        speaker_ref_path (Path): Path of the speaker reference wav for inference.
        output_dir (str): Output saving dir.
        language (str): Target language code (e.g., 'es').
        sentence_buffer_sec (float): Silence in seconds between joined sentence clips.
        n_variants_per_sentence (int): Number of XTTS generations per sentence (default: 2).
    """
    df = pd.read_csv(csv_path)

    # Initialize Whisper alignment model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    alignment_model, metadata = whisperx.load_align_model(
        language_code=language, device=device
    )

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
                    tts_model=tts_model,
                    speaker_ref_path=speaker_ref_path,
                    output_filename=temp_name,
                    output_dir=output_dir,
                    language=language,
                )

                audio = AudioSegment.from_wav(temp_path).set_channels(1)
                duration = len(audio)

                # Always track the shortest untrimmed version
                if duration < shortest_duration:
                    shortest_audio = audio
                    shortest_duration = duration

                # Try trimming via WhisperX
                last_word_end = align_last_word_timestamp(
                    alignment_model, metadata, temp_path, sentence, language=language
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
    "--voice",
    type=str,
    required=True,
    help="Pre trained speaker voice to use for inference",
)
@click.option(
    "--model-version",
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
def main(voice, model_version, speaker_ref, language):
    """Run XTTS inference with alignment-based cleanup and stitch audio with silence alignment."""
    # Initialize XTTS model
    gpu = torch.cuda.is_available()
    model_path = MODEL_OUTPUT_PATH / voice / model_version
    speaker_ref_path = (
        MODEL_OUTPUT_PATH / voice / "speaker_references" / f"{speaker_ref}.wav"
    )

    tts_model = TTS(
        model_path=model_path,
        config_path=model_path / "config.json",
        progress_bar=True,
        gpu=gpu,
    )

    for inference_dir in DATA_INFERENCE.iterdir():
        name = inference_dir.name

        output_dir = DATA_INFERENCE / name / "tts_segments"
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = inference_dir / "translated_segments.csv"

        logger.info("Starting XTTS inference with WhisperX alignment for %s", name)
        run_segment_inference(
            csv_path=csv_path,
            tts_model=tts_model,
            output_dir=output_dir,
            speaker_ref_path=speaker_ref_path,
            language=language,
        )


if __name__ == "__main__":
    main()
