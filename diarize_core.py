"""Core diarization logic wrapping pyannote.audio Pipeline."""
import time
from typing import List, Dict, Optional

from env_config import get_config, get_device, log

_PIPELINE = None

def load_pipeline(model_name: Optional[str] = None, hf_token: Optional[str] = None, device: Optional[str] = None):
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE
    cfg = get_config()
    model_name = model_name or cfg.MODEL_NAME
    hf_token = hf_token or cfg.HF_TOKEN
    device = device or get_device()

    log(f"Loading pyannote pipeline {model_name} on {device} ...", "info")
    from pyannote.audio import Pipeline
    pipe = Pipeline.from_pretrained(model_name, use_auth_token=hf_token)
    # move to gpu if available
    try:
        if device == "cuda":
            import torch
            pipe.to(torch.device("cuda"))
    except Exception as e:
        log(f"Could not move pipeline to CUDA: {e}", "warn")
    _PIPELINE = pipe
    return pipe


def run_diarization(audio_path: str,
                    min_speakers: Optional[int] = None,
                    max_speakers: Optional[int] = None,
                    num_speakers: Optional[int] = None):
    pipe = load_pipeline()
    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

    t0 = time.time()
    diar = pipe(audio_path, **kwargs)
    rt = time.time()-t0
    return diar, rt


def annotation_to_segments(annotation) -> List[Dict]:
    segs = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segs.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": str(speaker)
        })
    return segs


def annotation_to_rttm(annotation) -> str:
    # pyannote Annotation has write_rttm() but easiest: to_rttm() -> str
    try:
        return annotation.to_rttm()
    except Exception:
        return ""
