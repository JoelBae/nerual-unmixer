# Use the official PyTorch image with CUDA support
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# PyTorch Optimization for large FFTs/Convolutions
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install system dependencies (libsndfile is required by torchaudio/pysoundfile)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set environment variable to ensure local modules are importable
ENV PYTHONPATH=/app

# Default command: Train the inverter with the full chain and proxy data
# This can be easily overridden in the Vertex AI Training Job configuration
ENTRYPOINT ["python", "src/training/train_inverter_audio.py"]
CMD ["--effect", "full_chain", "--use_proxy_data", "--batch_size", "16"]
