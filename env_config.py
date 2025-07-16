import os
from dataclasses import dataclass

# ------------------------------------------------------------------
# Environment / configuration helpers
# ------------------------------------------------------------------

@dataclass
class Config:
    HF_TOKEN: str
    MODEL_NAME: str = "pyannote/speaker-diarization-3.1"
    DEFAULT_MIN_SPK: int = 1
    DEFAULT_MAX_SPK: int = 8
    RP_LOG_LEVEL: str = "info"

def _get_env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except Exception:
        return default

def load_config() -> Config:
    # 🛠 כאן הוספנו תמיכה בטוקן מ-RunPod
    token = (
        os.getenv("HF_TOKEN") or
        os.getenv("HUGGING_FACE_HUB_TOKEN") or
        os.getenv("HUGGINGFACE_HUB_TOKEN") or
        os.getenv("RUNPOD_SECRET_HF_TOKEN")  # ← זה מה שאתה צריך ב־RunPod
    )
    if not token:
        raise RuntimeError("Missing HF_TOKEN / HUGGINGFACE_HUB_TOKEN / RUNPOD_SECRET_HF_TOKEN")
    cfg = Config(
        HF_TOKEN=token,
        MODEL_NAME=os.getenv("MODEL_NAME", "pyannote/speaker-diarization-3.1"),
        DEFAULT_MIN_SPK=_get_env_int("DEFAULT_MIN_SPK", 1),
        DEFAULT_MAX_SPK=_get_env_int("DEFAULT_MAX_SPK", 8),
        RP_LOG_LEVEL=os.getenv("RP_LOG_LEVEL", "info"),
    )
    return cfg

def get_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

_cfg = None
def get_config() -> Config:
    global _cfg
    if _cfg is None:
        _cfg = load_config()
    return _cfg

def log(msg: str, level: str = "info"):
    cfg = get_config()
    order = {"debug": 0, "info": 1, "warn": 2, "error": 3}
    target = order.get(cfg.RP_LOG_LEVEL, 1)
    lvl = order.get(level, 1)
    if lvl >= target:
        print(f"[{level.upper()}] {msg}", flush=True)
