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
    vim \
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

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install gunicorn==21.2.0

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install werkzeug
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install flask
 
# Stage 2: Final Application Image
FROM base AS final

# Copy all project files to the final image
COPY . .

# Pass HUGGING_FACE_HUB_TOKEN as a build argument
ARG HUGGING_FACE_HUB_TOKEN
ENV HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN

# Expose Flask default port
EXPOSE 5000

# The command to run the application
CMD ["gunicorn", \
     "--workers", "4", \  
     "--timeout", "240", \  
     "--bind", "0.0.0.0:5000", \
     "--log-level", "debug", \ 
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "wsgi:app", \
     "--reload"]