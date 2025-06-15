# Makefile for the YouTube Voice Training Data Generation Pipeline

# Expected usage:
# make all VOICE=john_vervaeke URLS="url1 url2"

.PHONY: all download convert segment coqui_formatting

# Top-level target
all: coqui_formatting

# Step 1: Download videos and extract audio
download:
	uv run python -m src.downloaders.download_youtube "$(URLS)"

# Step 2: Convert audio to 16kHz and 22.05kHz
convert: download
	uv run python -m src.preprocessing.convert_audio

# Step 3: Segment the audio for transcription
segment: convert
	uv run python -m src.preprocessing.segment_audio

# Step 4: Create the training dataset
coqui_formatting: segment
	uv run python -m src.preprocessing.coqui_formatting
