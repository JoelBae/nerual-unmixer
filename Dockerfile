FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies (e.g. for audio processing if needed)
RUN apt-get update && apt-get install -y \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install wandb pyyaml soundfile

# Copy source code
COPY . .

# Default command (can be overridden)
CMD ["python", "src/training/train_proxies.py", "--help"]
