# speaker-identifier

FastAPI-сервис embeddings и проверки говорящего на Qwen3-ASR audio encoder. Функция экспериментальная и не заменяет надежную аутентификацию без калибровки.

## Docker Compose

```bash
docker compose build speaker-identifier
docker compose up -d speaker-identifier
curl http://localhost:9010/docs
```

Standalone: `docker build -t speaker-identifier . && docker run --rm -p 9010:9010 -v speaker-models:/app/pretrained_models -v speaker-data:/app/data speaker-identifier`.

## Переменные и данные

- `HOST` (default `0.0.0.0`), `PORT` (default `9010`)
- `QWEN3_ASR_ENROLLMENT_STORE` — persistent JSON профилей
- `VOICEPRINTS_DIR` — необязательно, raw voiceprints
- `QWEN3_ASR_EMBEDDER_MODEL` (default `Qwen/Qwen3-ASR-0.6B`)
- `QWEN3_ASR_EMBEDDER_CACHE`, `QWEN3_ASR_EMBEDDER_DEVICE`
- `SPEAKER_VERIFICATION_THRESHOLD` (default `0.75`) и `SPEAKER_VERIFICATION_MIN_MARGIN` (default `0.05`)

Модель скачивается с Hugging Face при первом старте. Persistent volumes обязательны, иначе профили и cache будут потеряны. Основные endpoints: `POST /audio-metadata`, `GET/PATCH /functions`, `GET /saved-embeddings`, `PUT/POST /state`.
