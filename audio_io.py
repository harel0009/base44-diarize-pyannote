"""Audio loading + normalization utilities.

We accept either:
- mode='url'  -> download bytes from URL
- mode='blob' -> base64-encoded audio bytes

We then ensure we produce a WAV file at 16kHz mono (float32) for pyannote.
"""
import io
import os
import base64
import tempfile
import subprocess  # Added for ffmpeg fallback
from typing import Optional

import requests
import numpy as np
import soundfile as sf

from env_config import log

# Some containers don't ship torchaudio w/ all codecs; we first try soundfile.
# If we need torchaudio for resample we'll import lazily.
try:
    import torchaudio
    import torch
    _HAS_TORCHAUDIO = True
except Exception:
    _HAS_TORCHAUDIO = False


class AudioFetchError(Exception):
    pass

class AudioDecodeError(Exception):
    pass

class AudioProcessError(Exception):
    pass


def fetch_bytes_from_url(url: str, timeout: int = 600) -> bytes:
    log(f"Downloading audio from URL: {url}", "debug")
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.content
    except Exception as e:
        raise AudioFetchError(f"Failed to download audio from {url}: {e}") from e


def decode_base64(b64: str) -> bytes:
    try:
        return base64.b64decode(b64)
    except Exception as e:
        raise AudioDecodeError(f"Invalid base64 audio: {e}") from e


def _sf_to_wav16k_mono(raw_bytes: bytes) -> str:
    """Try to decode with soundfile, resample if needed, write wav16k mono."""
    data, sr = sf.read(io.BytesIO(raw_bytes), always_2d=True)
    # data shape: (num_frames, num_channels)
    # Convert to mono
    if data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True)
    else:
        data = data[:, 0:1]
    # Resample if needed
    if sr != 16000:
        if _HAS_TORCHAUDIO:
            waveform = torch.from_numpy(data.T)  # shape (1, n)
            resamp = torchaudio.functional.resample(waveform, sr, 16000)
            data = resamp.numpy().T
            sr = 16000
        else:
            # naive numpy resample fallback
            import math
            ratio = 16000 / sr
            new_len = int(math.ceil(data.shape[0] * ratio))
            data = np.interp(
                np.linspace(0, data.shape[0], new_len, endpoint=False),
                np.arange(data.shape[0]),
                data[:, 0]
            ).reshape(-1, 1)
            sr = 16000
    # write temp wav
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, data, sr, subtype="PCM_16")
        return f.name


def ensure_wav_16k_mono(raw_bytes: bytes) -> str:
    """Return path to temp wav16k mono; fallback to ffmpeg for unsupported formats."""
    try:
        return _sf_to_wav16k_mono(raw_bytes)
    except Exception:
        # Fallback: decode via ffmpeg (handles webm/opus etc.)
        try:
            tmp_webm = tempfile.NamedTemporaryFile(suffix=".input", delete=False).name
            with open(tmp_webm, "wb") as f:
                f.write(raw_bytes)

            tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", tmp_webm,
                "-ac", "1", "-ar", "16000",
                tmp_wav,
            ]
            subprocess.run(cmd, check=True)
            return tmp_wav
        except Exception as e:
            raise AudioProcessError(f"Failed to normalize audio via ffmpeg: {e}") from e


def load_audio_to_path(mode: str, url: Optional[str] = None, base64: Optional[str] = None) -> str:
    if mode == "url":
        if not url:
            raise AudioFetchError("mode='url' requires 'url'")
        raw = fetch_bytes_from_url(url)
    elif mode == "blob":
        if not base64:
            raise AudioDecodeError("mode='blob' requires 'base64'")
        raw = decode_base64(base64)
    else:
        raise ValueError("mode must be 'url' or 'blob'")
    return ensure_wav_16k_mono(raw)
