# Base image with CUDA and PyTorch
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Avoid interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ca-certificates \
    build-essential \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set default Python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Install uv globally
RUN curl -Ls https://astral.sh/uv/install.sh | bash

# Create workspace
WORKDIR /workspace

# Copy project files into container
COPY uv.lock /workspace
COPY pyproject.toml /workspace
COPY .python-version /workspace
COPY src /workspace

# Sync environment using uv.lock
RUN /root/.cargo/bin/uv sync

# Clone Coqui TTS fork and install editable
RUN git clone https://github.com/idiap/coqui-ai-TTS.git /workspace/support_repos/coqui-ai-TTS && \
    /root/.cargo/bin/uv pip install -e /workspace/support_repos/coqui-ai-TTS

# Set torch to use GPU
ENV CUDA_VISIBLE_DEVICES=0

# Run training (override in docker run if needed)
CMD ["/bin/bash"]
