# Base44 Pyannote Diarization Worker
# NOTE: Using official python base. For GPU performance you may prefer a RunPod PyTorch base image.
FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git wget curl build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY env_config.py audio_io.py diarize_core.py handler.py README.md ./

# Default env (override in RunPod console)
ENV MODEL_NAME=pyannote/speaker-diarization-3.1
ENV RP_LOG_LEVEL=info
# DO NOT bake HF_TOKEN into image!
# ENV HF_TOKEN=...

CMD ["python", "handler.py"]
