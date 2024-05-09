# Stage 1: Base and Dependency Setup
FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    wget \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/miniconda \
    && rm /tmp/miniconda.sh

# Update PATH for conda
ENV PATH=/opt/miniconda/bin:$PATH

# Set up Conda environment with specific dependencies
RUN conda install -y -c pytorch -c defaults -c anaconda \
    python=3.9 \
    pip

# Create a user and group
RUN adduser --system --group --no-create-home app

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirement.txt .

# Install Torch, Torchvision, and other dependencies from requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install -r requirement.txt \
    && pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html 

# Stage 2: Final Application Image
FROM base AS final

# Copy all project files to the final image
COPY . .

# Pass HUGGING_FACE_HUB_TOKEN as a build argument
ARG HUGGING_FACE_HUB_TOKEN
ENV HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN

# Expose Flask default port
EXPOSE 5000

# Entry point command, include the parameters here
CMD ["python", "app.py", "--cfg-path", "eval_configs/video_llama_eval_only_vl.yaml", "--model_type", "llama_v2"]


