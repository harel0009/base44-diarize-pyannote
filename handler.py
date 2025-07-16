"""RunPod serverless handler entrypoint for Base44 Diarization Worker."""
import os
os.environ["HF_TOKEN"] = os.environ.get("RUNPOD_SECRET_HF_TOKEN", "") 
import json
import traceback
from typing import Any, Dict

from env_config import get_config, get_device, log
from audio_io import load_audio_to_path, AudioFetchError, AudioDecodeError, AudioProcessError
from diarize_core import load_pipeline, run_diarization, annotation_to_segments, annotation_to_rttm

# Eager load so first request is faster (or do lazy; up to you)
_cfg = get_config()
_DEVICE = get_device()
_PIPE = load_pipeline(_cfg.MODEL_NAME, _cfg.HF_TOKEN, _DEVICE)

def _error(msg: str, code: str = "ERROR") -> Dict[str, Any]:
    log(f"{code}: {msg}", "error")
    return {"ok": False, "error": code, "message": msg}

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """Main RunPod handler signature."""
    try:
        inp = event.get("input", {})
        mode = inp.get("mode")
        if mode not in ("url", "blob"):
            return _error("'mode' must be 'url' or 'blob'", "MODE_REQUIRED")
        url = inp.get("url")
        b64 = inp.get("base64")
        min_speakers = inp.get("min_speakers")
        max_speakers = inp.get("max_speakers")
        num_speakers = inp.get("num_speakers")
        want_rttm = bool(inp.get("return_rttm", False))

        audio_path = load_audio_to_path(mode, url=url, base64=b64)

        annotation, runtime_sec = run_diarization(audio_path, min_speakers, max_speakers, num_speakers)
        segs = annotation_to_segments(annotation)
        out = {
            "ok": True,
            "segments": segs,
            "model": _cfg.MODEL_NAME,
            "runtime_sec": runtime_sec,
            "device": _DEVICE,
        }
        if want_rttm:
            out["rttm"] = annotation_to_rttm(annotation)
        return out

    except AudioFetchError as e:
        return _error(str(e), "AUDIO_FETCH")
    except AudioDecodeError as e:
        return _error(str(e), "AUDIO_DECODE")
    except AudioProcessError as e:
        return _error(str(e), "AUDIO_PROCESS")
    except Exception as e:
        tb = traceback.format_exc()
        log(tb, "error")
        return _error(str(e), "UNEXPECTED")

# ------------------------------------------------------------------
# RunPod bootstrap
# ------------------------------------------------------------------
if __name__ == "__main__":
    # optional local test: python handler.py < test_input.json
    import sys, json
    if not sys.stdin.isatty():
        data = json.load(sys.stdin)
        print(json.dumps(handler(data), indent=2))
    else:
        print("No stdin JSON; nothing to do.")

import runpod
runpod.serverless.start({"handler": handler})
