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
This project uses the [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) model for lip synchronization of the speaker’s face with the translated audio.

Wav2Lip is not published as a Python package and its dependencies conflict with the rest of this project.
To isolate the environment, we clone the repository outside this project and create a dedicated virtual environment.

Clone the Wav2Lip repository to a folder outside your project:
```bash
cd ~/support_repos
git clone https://github.com/Rudrabha/Wav2Lip.git
cd Wav2Lip
```

Create a Virtual Environment and install updated, compatible dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip

# Install minimal compatible dependencies
pip install --no-deps tqdm numba requests
pip install "llvmlite>=0.44,<0.45"
pip install opencv-python opencv-contrib-python
pip install --upgrade numpy==2.2.0 librosa==0.10.0 matplotlib

# Install PyTorch and related packages for GPU (adjust CUDA version if needed)
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
```

Wav2Lip requires a pretrained checkpoint file, which is not included in the repo.

Go to the official checkpoint folder:
https://drive.google.com/drive/folders/1I-0dNLfFOSFwrfqjNa-SXuwaURHE5K4k

Download the file named wav2lip_gan.pth

Move the downloaded file to the checkpoints/ directory:
```bash
mkdir -p checkpoints
mv ~/Downloads/wav2lip_gan.pth checkpoints/
```

To invoke Wav2Lip from anywhere via command line, create an alias:
```bash
alias wav2lip_inference='~/support_repos/Wav2Lip/.venv/bin/python ~/support_repos/Wav2Lip/inference.py'
```

Add this line to your ~/.bashrc or ~/.zshrc file to persist it across sessions:
```bash
echo "alias wav2lip_inference='~/support_repos/Wav2Lip/.venv/bin/python ~/support_repos/Wav2Lip/inference.py'" >> ~/.bashrc
source ~/.bashrc
```