# Base44 Pyannote Diarization Worker (RunPod Serverless)

**Purpose:** Accept an audio file (URL or Base64 blob), run `pyannote/speaker-diarization-3.1`, and return speaker segments `[start,end,speaker]` as JSON. Designed to be deployed as a RunPod Serverless Endpoint and called from the Base44 platform.

---

## Quick Start

### 1. Requirements
- Hugging Face account + accepted terms for `pyannote/speaker-diarization-3.1`.
- HF read token (export as `HF_TOKEN`).
- Docker + GitHub account (for deployment flow).
- RunPod account + API key.

### 2. Clone & build
```bash
git clone <YOUR_REPO_URL> base44-diarize-pyannote
cd base44-diarize-pyannote
docker build -t YOUR_DOCKER_USER/base44-diarize:latest .
```

### 3. Push image
```bash
docker push YOUR_DOCKER_USER/base44-diarize:latest
```

### 4. Create RunPod Serverless Endpoint
- Source: GitHub Repo (recommended) *or* Custom image.
- Env vars:
  - `HF_TOKEN` (required)
  - `MODEL_NAME=pyannote/speaker-diarization-3.1`
  - `DEFAULT_MIN_SPK=1`
  - `DEFAULT_MAX_SPK=8`
  - `RP_LOG_LEVEL=info`
- GPU: 16GB+ (L4/A5000 works for most workloads).

### 5. Test (runsync)
Create `test_input.json`:
```json
{
  "input": {
    "mode": "url",
    "url": "https://example.com/demo.wav",
    "min_speakers": 2,
    "max_speakers": 4,
    "return_rttm": true
  }
}
```

```bash
curl -s -X POST \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  --data @test_input.json \
  https://api.runpod.ai/v2/$ENDPOINT_ID/runsync | jq
```

### 6. Example response
```json
{
  "ok": true,
  "segments": [
    {"start":0.00,"end":5.21,"speaker":"SPEAKER_0"},
    {"start":5.21,"end":12.98,"speaker":"SPEAKER_1"}
  ],
  "model":"pyannote/speaker-diarization-3.1",
  "runtime_sec": 12.4,
  "device":"cuda",
  "rttm": "SPEAKER_0 ..."
}
```

---

## Input Schema

| Field | Type | Required | Notes |
|---|---|---|---|
| `mode` | `url` \| `blob` | ✅ | Use url for large files; blob (base64) for small. |
| `url` | string | if mode=url | Direct downloadable audio (no HTML redirect). |
| `base64` | string | if mode=blob | Base64-encoded audio bytes. |
| `min_speakers` | int | optional | Minimum speakers to help clustering. |
| `max_speakers` | int | optional | Maximum speakers. |
| `num_speakers` | int | optional | Override both min/max; force count. |
| `return_rttm` | bool | optional | Return RTTM annotation string. |

---

## Output Schema

| Field | Type | Notes |
|---|---|---|
| `ok` | bool | true if success |
| `segments` | array | `{start,end,speaker}` floats (seconds) |
| `model` | string | model ID used |
| `runtime_sec` | float | execution time inside pipeline |
| `device` | string | 'cuda' or 'cpu' |
| `rttm` | string? | only if requested |
| `error`/`message` | string? | returned on error |

---

## License
MIT for this wrapper. Underlying pyannote models subject to their own license & terms — see Hugging Face model card.

