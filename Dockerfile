# Stage 1: Base and Dependency Setup
FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04 AS base


# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install essential system dependencies in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    wget \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Git LFS directly with minimal commands
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && apt-get update \
    && apt-get install -y git-lfs \
    && git lfs install

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/miniconda \
    && rm /tmp/miniconda.sh

# Update PATH for conda
ENV PATH=/opt/miniconda/bin:$PATH


# Set up Conda environment with specific dependencies
RUN conda install -y -c pytorch -c defaults -c anaconda \
    python=3.9 \
    pip \
    cudatoolkit \
    torchaudio=0.12.1 \
    torchvision=0.13.1

# Copy requirement file to maximize caching
COPY requirement.txt /app/requirement.txt

# Install Python packages using pip
RUN pip install --no-cache-dir -r /app/requirement.txt

# Install specific versions of torch and torchaudio with CUDA 11.3
RUN pip install torch==1.12.1+cu113 torchaudio==0.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# Stage 2: Final Application Image
FROM base AS final


# Copy all project files to the final image
COPY . /app

# Set the default working directory
WORKDIR /app

# Set the entry point command
CMD ["python", "demo_video.py", "--cfg-path", "eval_configs/video_llama_eval_only_vl.yaml", "--model_type", "llama_v2"]


