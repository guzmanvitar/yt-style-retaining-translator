# yt-style-retaining-translator 🎙️🌍
This project translates YouTube videos into a different language while retaining the speaker’s original voice style. It combines voice cloning, TTS, and translation to create dubbed videos that feel natural and authentic.

## Features

- 📼 Download and process YouTube videos
- 🧠 Voice style preservation via TTS
- 🌐 Automatic translation of speech
- 🎧 Audio re-synthesis in target language
- 📂 Modular pipeline for preprocessing, translation, synthesis, and video reassembly

##  Tech Stack

- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for downloading and extracting audio
- Whisper for transcription
- Coqui TTS (with voice cloning or fine-tuned voice style)
- FastAPI or CLI for orchestration
- ffmpeg for audio/video processing
- Translation via OpenAI.

## 🔧 Setup & Installation
1️⃣ Clone the Repository
```bash
git clone https://github.com/guzmanvitar/yt-style-retaining-translator
cd yt-style-retaining-translator
```

2️⃣ Install Dependencies
This repo uses [uv](https://docs.astral.sh/uv/getting-started/installation) to install and manage dependencies,
as well as to set up the Python environment. After installing `uv` run
```bash
uv python install 3.10.13
uv sync
```
To set up Git hooks for code quality checks run also
```bash
uv run pre-commit install
```

3️⃣ Install Coqui TTS and Trainer (Development Version)
This project relies on the latest features and training scripts from the Coqui [TTS](https://github.com/coqui-ai/TTS) and [Trainer](https://github.com/coqui-ai/Trainer) repositories.

These are installed directly from source because the version available on PyPI is intended for inference only.
Training requires the full source code, which is actively developed in the GitHub repositories.

Clone the Coqui repos to a folder outside your project (e.g. `~/support_repos`):
```bash
mkdir -p ~/support_repos
cd ~/support_repos

git clone https://github.com/idiap/coqui-ai-TTS.git
git clone https://github.com/coqui-ai/Trainer.git
```

Install both in editable mode (with no dependencies) using uv pip:
```bash
uv pip install -e ~/support_repos/coqui-ai-TTS
uv pip install -e ~/support_repos/Trainer
```
