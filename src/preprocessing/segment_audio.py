"""Segment audio based on Whisper transcription."""

import csv
from pathlib import Path

import whisper
from pydub import AudioSegment, silence


def segment_audio_by_silence(
    audio_path: Path,
    output_dir: Path,
    model_size: str = "base",
    min_silence_len: int = 1300,
    silence_thresh_offset: int = -18,
    keep_silence: int = 300,
):
    """
    Split audio on silences, transcribe each chunk with Whisper, and save metadata in CSV.

    Args:
        audio_path (Path): Input WAV file.
        output_dir (Path): Where to save segments and CSV.
        model_size (str): Whisper model size (e.g., "base", "small").
        min_silence_len (int): Minimum silence length to split (ms).
        silence_thresh_offset (int): dBFS offset from average to consider silence.
        keep_silence (int): Silence (ms) to retain on each side of a chunk.
    """
    print(f"Loading audio from: {audio_path}")
    audio = AudioSegment.from_wav(audio_path)
    threshold = audio.dBFS + silence_thresh_offset

    print(f"Splitting on silences (threshold={threshold} dBFS)...")
    chunks = silence.split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=threshold,
        keep_silence=keep_silence,
    )

    print(f"Found {len(chunks)} segments. Transcribing...")
    model = whisper.load_model(model_size)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "segments.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "start", "end", "text"])

        current_time = 0
        for i, chunk in enumerate(chunks):
            duration = len(chunk)
            start_sec = round(current_time / 1000, 2)
            end_sec = round((current_time + duration) / 1000, 2)

            filename = f"segment_{i:03}.wav"
            filepath = output_dir / filename
            chunk.export(filepath, format="wav")

            result = model.transcribe(
                str(filepath), language="en", fp16=False, best_of=5
            )
            text = result["text"].strip()

            writer.writerow([filename, start_sec, end_sec, text])
            current_time += duration

    print(f"✅ Done! Segments and metadata saved to: {output_dir}")


if __name__ == "__main__":
    segment_audio_by_silence(
        audio_path=Path("data/processed/tolkien-speech-clean_22k.wav"),
        output_dir=Path("data/processed/chunks"),
        model_size="large",
    )
