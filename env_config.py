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
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    print(f"[DEBUG] HF_TOKEN loaded: {token[:6]}...", flush=True)
    if not token:
        raise RuntimeError("HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) is required to load pyannote models.")
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
    # simple level gating
    order = {"debug":0,"info":1,"warn":2,"error":3}
    target = order.get(cfg.RP_LOG_LEVEL,1)
    lvl = order.get(level,1)
    if lvl >= target:
        print(f"[{level.upper()}] {msg}", flush=True)
