# Makefile for the YouTube Video Translation Pipeline

# Expected usage:
# make all VOICE=john_vervaeke URLS="url1 url2"

.PHONY: all download convert segment translate_transcribe translate_audio sync_audio lipsync finalize

# Top-level target
all: finalize

# Step 1: Download videos and extract audio
download:
	uv run python -m src.downloaders.download_youtube $(URLS)

# Step 2: Convert audio to 16kHz and 22.05kHz
convert: download
	uv run python -m src.preprocessing.convert_audio

# Step 3: Segment the audio for transcription
segment: convert
	uv run python -m src.preprocessing.segment_audio

# Step 4: Translate the transcribed audio segments
translate_transcribe: segment
	uv run python -m src.translation.translate_transcription --speaker $(VOICE)

# Step 5: Run TTS on translated transcription to generate audio
translate_audio: translate_transcribe
	uv run python -m src.translation.translate_audio --voice $(VOICE)

# Step 6: Synchronize translated audio with original video
sync_audio: translate_audio
	uv run python -m src.postprocessing.synchronize_audio_video_segments

# Step 7: Lip sync the audio to video
lipsync: sync_audio
	uv run python -m src.postprocessing.lip_sync

# Step 8: Finalize the translated video
finalize: lipsync
	uv run python -m src.postprocessing.finalize_video
