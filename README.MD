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

3️⃣ Install Coqui TTS (Development Version)
This project relies on the latest features from the [Coqui TTS fork](https://github.com/idiap/coqui-ai-TTS.git) repositories.

This repo is installed directly from source because the version available on PyPI is intended for inference only.
Training requires the full source code, which is actively developed in the GitHub repositories.

Clone the Coqui repo to a folder outside your project (e.g. `~/support_repos`):
```bash
mkdir -p ~/support_repos
cd ~/support_repos

git clone https://github.com/idiap/coqui-ai-TTS.git
```

Install in editable mode (with no dependencies) using uv pip:
```bash
uv pip install -e ~/support_repos/coqui-ai-TTS
```

4️⃣ Install Wav2Lip (CLI Integration with Alias)
This project uses my personal version of the [wav2lip-onnx-HQ](https://github.com/guzmanvitar/wav2lip-onnx-HQ) repo for lip synchronization of the speaker’s face with the translated audio.

Wav2Lip is not published as a Python package and its dependencies conflict with the rest of this project.
To isolate the environment, we clone the repository outside this project and create a dedicated environment.

Clone the Wav2Lip repository to a folder outside your project:
```bash
cd ~/support_repos
git clone https://github.com/guzmanvitar/wav2lip-onnx-HQ-custom.git
cd wav2lip-onnx-HQ-custom
```

Install pyenv and python 3.11:
```bash
curl -fsSL https://pyenv.run | bash
export PATH="$HOME/.pyenv/bin:$PATH"

pyenv install 3.11.9
pyenv local 3.11.9
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
```

Create venv and install requirements:
```bash
# Create and activate virtualenv
python -m venv .venv
source .venv/bin/activate

# Install required packages
pip install --upgrade pip
pip install -r requirements.txt
```