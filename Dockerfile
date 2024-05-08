# Use a suitable base image with Python 3.9 support
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

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

# Set the working directory
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Default command
CMD ["bash"]

