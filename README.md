# speaker-identifier

FastAPI service for Qwen3-ASR embeddings and speaker verification. This is experimental and must not be treated as reliable authentication without calibration.

## Docker Compose

```bash
docker compose build speaker-identifier
docker compose up -d speaker-identifier
curl http://localhost:9010/docs
```

Standalone: `docker build -t speaker-identifier . && docker run --rm -p 9010:9010 -v speaker-models:/app/pretrained_models -v speaker-data:/app/data speaker-identifier`.

## Environment and data

- `HOST` (default `0.0.0.0`) and `PORT` (default `9010`)
- `QWEN3_ASR_ENROLLMENT_STORE` - persistent profile JSON
- `VOICEPRINTS_DIR` - optional raw voiceprints directory
- `QWEN3_ASR_EMBEDDER_MODEL` (default `Qwen/Qwen3-ASR-0.6B`)
- `QWEN3_ASR_EMBEDDER_CACHE` and `QWEN3_ASR_EMBEDDER_DEVICE`
- `SPEAKER_VERIFICATION_THRESHOLD` (default `0.75`) and `SPEAKER_VERIFICATION_MIN_MARGIN` (default `0.05`)

The model is downloaded from Hugging Face on first start. Persistent volumes are required or profiles and model cache will be lost. Main endpoints include `POST /audio-metadata`, `GET/PATCH /functions`, `GET /saved-embeddings`, and `PUT/POST /state`.
