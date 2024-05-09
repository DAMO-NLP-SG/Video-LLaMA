# Stage 1: Base and Dependency Setup
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install essential system dependencies in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    gnupg \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install git-lfs directly with minimal commands
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && apt-get update \
    && apt-get install -y git-lfs \
    && git lfs install

# Stage 2: Python Environment Setup
FROM base AS python-env

# Set up Conda environment with specific dependencies
RUN conda install -y -c pytorch -c defaults -c anaconda \
    python=3.9 \
    pip \
    cudatoolkit \
    torchaudio=0.12.1 \
    torchvision=0.13.1

# Copy requirement file to a specific directory to maximize caching
COPY requirement.txt /app/requirement.txt

# Install Python packages using pip
RUN pip install -r /app/requirement.txt

# Stage 3: Final Application Image
FROM python-env AS final

# Copy all project files to the final image
COPY . /app

# Set the default working directory
WORKDIR /app

# Set the entry point command
#CMD ["python", "demo_video.py", "--cfg-path", "eval_configs/video_llama_eval_only_vl.yaml", "--model_type", "llama_v2"]
CMD ["bash"]

