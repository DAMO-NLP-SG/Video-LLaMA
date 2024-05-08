# Use a suitable base image with Python 3.9 support
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (ffmpeg and git-lfs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install git-lfs
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && apt-get install -y git-lfs \
    && git lfs install

# Copy the requirements file
COPY requirement.txt /app/requirement.txt

# Install conda dependencies
RUN conda install -c pytorch -c defaults -c anaconda \
    python=3.9 \
    cudatoolkit \
    pip \
    torchaudio=0.12.1 \
    torchvision=0.13.1

# Install pip dependencies
RUN pip install --no-cache-dir -r /app/requirement.txt

# Clone the required repositories with git-lfs
WORKDIR /app
RUN git clone https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-13B-Finetuned \
    && git clone https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-13B-Pretrained

# Copy the project files into the container
COPY . /app

# Set the default working directory
WORKDIR /app

# Command to run the demo with Gradio
CMD ["python", "demo_video.py", "--cfg-path", "eval_configs/video_llama_eval_only_vl.yaml", "--model_type", "llama_v2"]

