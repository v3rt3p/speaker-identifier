from __future__ import annotations

import logging
import os
import uuid
import wave
from dataclasses import asdict
from pathlib import Path

import numpy as np
import uvicorn
from cachetools import TTLCache
from fastapi import FastAPI, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from models.qwen_asr_embedder import (
    create_qwen_asr_embedding_model_from_env,
    speaker_model_id,
)
from utils import f32_samples_to_s16_bytes, s16_bytes_to_f32_samples
from verification import (
    VoiceEnrollmentStore,
    VoiceMatch,
    normalize_embedding,
    rank_voice_matches,
)


HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "9010"))
AUDIO_CACHE_SIZE = int(os.getenv("AUDIO_CACHE_SIZE", "50"))
AUDIO_CACHE_TTL_SECONDS = int(os.getenv("AUDIO_CACHE_TTL_SECONDS", "600"))
VOICE_ENROLLMENT_CACHE_SIZE = int(os.getenv("VOICE_ENROLLMENT_CACHE_SIZE", "100"))
VOICE_ENROLLMENT_CACHE_TTL_SECONDS = int(
    os.getenv("VOICE_ENROLLMENT_CACHE_TTL_SECONDS", "1800")
)
SIMILARITY_THRESHOLD = float(
    os.getenv(
        "SPEAKER_VERIFICATION_THRESHOLD",
        os.getenv("SIMILARITY_THRESHOLD", "0.75"),
    )
)
MIN_SIMILARITY_MARGIN = float(
    os.getenv("SPEAKER_VERIFICATION_MIN_MARGIN", "0.05")
)
MIN_SEGMENT_SAMPLES = max(
    1,
    int(
        float(os.getenv("SPEAKER_VERIFICATION_MIN_SEGMENT_SECONDS", "1.0"))
        * 16000
    ),
)
configured_store = os.getenv("QWEN3_ASR_ENROLLMENT_STORE") or os.getenv(
    "SPEAKER_EMBEDDINGS_FILE"
)
legacy_store = Path("speaker_embeddings.json")
SPEAKER_EMBEDDINGS_FILE = configured_store or (
    str(legacy_store)
    if legacy_store.exists()
    else "data/qwen3_asr_enrollments.json"
)
VOICEPRINTS_DIR = os.getenv("VOICEPRINTS_DIR")

sample_rate = 16000
model = create_qwen_asr_embedding_model_from_env(sample_rate)
enrollment_store = VoiceEnrollmentStore(SPEAKER_EMBEDDINGS_FILE)

audio_cache = TTLCache(maxsize=AUDIO_CACHE_SIZE, ttl=AUDIO_CACHE_TTL_SECONDS)
# A mapping makes repeated save/finish calls for the same record idempotent.
voice_enrollment_cache = TTLCache(
    maxsize=VOICE_ENROLLMENT_CACHE_SIZE,
    ttl=VOICE_ENROLLMENT_CACHE_TTL_SECONDS,
)

logger = logging.getLogger("uvicorn.access")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

app = FastAPI()


class GetFunctionsOrStatePayload(BaseModel):
    metadata: dict
    sessionId: str


class FunctionCallPayload(BaseModel):
    metadata: dict
    sessionId: str
    name: str
    arguments: dict


def save_voice_sample_enrollment(session_id: str, metadata: dict) -> bool:
    record_id = metadata.get("recordId")
    if record_id is None:
        logger.warning(
            "save_voice_sample_enrollment missing recordId: "
            f"session_id={session_id}, metadata={metadata}"
        )
        return False
    sample = audio_cache.get(record_id)
    if sample is None:
        logger.warning(
            "save_voice_sample_enrollment record not found or expired: "
            f"session_id={session_id}, recordId={record_id}"
        )
        return False
    samples = dict(voice_enrollment_cache.get(session_id) or {})
    samples[record_id] = sample
    voice_enrollment_cache[session_id] = samples
    logger.info(
        f"save_voice_sample_enrollment session_id={session_id} "
        f"recordId={record_id} samples_count={len(samples)}"
    )
    return True


def save_voiceprint_wav(context_id: str, samples: np.ndarray) -> None:
    if VOICEPRINTS_DIR is None:
        return
    try:
        directory = Path(VOICEPRINTS_DIR)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{context_id}.wav"
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(f32_samples_to_s16_bytes(samples))
        logger.info(f"voiceprint_saved path={path} session_id={context_id}")
    except Exception as exc:
        logger.exception(f"voiceprint_save_failed session_id={context_id}: {exc}")


def finish_voice_sample_enrollment(
    session_id: str,
    metadata: dict,
    comment: str,
) -> bool:
    save_voice_sample_enrollment(session_id, metadata)
    samples_by_id = voice_enrollment_cache.get(session_id) or {}
    samples = list(samples_by_id.values())
    if not samples:
        logger.warning(
            "finish_voice_sample_enrollment no samples: "
            f"session_id={session_id} comment={comment}"
        )
        return False
    data = np.concatenate(samples)
    save_voiceprint_wav(session_id, data)
    embedding = model.extract_embeddings(data)
    if embedding is None:
        logger.warning(
            "finish_voice_sample_enrollment embedding_failed: "
            f"session_id={session_id} comment={comment}"
        )
        return False
    voice = enrollment_store.enroll(
        speaker_id=session_id,
        label=comment,
        embedding=embedding,
        threshold=SIMILARITY_THRESHOLD,
        min_margin=MIN_SIMILARITY_MARGIN,
        model_id=speaker_model_id(),
        append=True,
    )
    voice_enrollment_cache.pop(session_id, None)
    logger.info(
        f"finish_voice_sample_enrollment session_id={session_id} "
        f"recordings={len(samples)} profile_samples={voice.samples} comment={comment}"
    )
    return True


def match_speaker(embedding: np.ndarray) -> VoiceMatch | None:
    matches = rank_voice_matches(
        normalize_embedding(embedding),
        enrollment_store.list(),
        model_id=speaker_model_id(),
        threshold_override=SIMILARITY_THRESHOLD,
        min_margin_override=MIN_SIMILARITY_MARGIN,
    )
    if not matches:
        return None
    for match in matches:
        logger.info(
            f"cosine_similarity sid={match.speaker_id} "
            f"sim={match.similarity:.4f} margin={match.margin:.4f}"
        )
    best = matches[0]
    if not best.accepted:
        logger.info(
            f"speaker_rejected best_id={best.speaker_id} "
            f"best_sim={best.similarity:.4f} margin={best.margin:.4f} "
            f"threshold={best.threshold:.4f} min_margin={best.min_margin:.4f}"
        )
    return best


def convert_and_normalize_input(raw: bytes) -> np.ndarray:
    return s16_bytes_to_f32_samples(raw)


@app.post("/audio-metadata", response_class=JSONResponse)
async def audio_metadata(
    request: Request,
    req_sample_rate: int = Query(..., alias="sample_rate"),
):
    data = await request.body()
    length = len(data)
    if req_sample_rate != sample_rate:
        logger.info(
            f"Invalid sample_rate={req_sample_rate}, "
            f"must be {sample_rate}, body_length={length}"
        )
        return JSONResponse(
            status_code=400,
            content={"error": "sample_rate must be 16000"},
        )
    if length % 2:
        return JSONResponse(
            status_code=400,
            content={"error": "request body must contain PCM16 samples"},
        )

    processed = convert_and_normalize_input(data)
    record_id = str(uuid.uuid4())
    audio_cache[record_id] = processed
    match = None
    if processed.size < MIN_SEGMENT_SAMPLES:
        logger.info(
            f"recordId={record_id} speaker processing skipped: "
            f"too_short samples={processed.size} minimum={MIN_SEGMENT_SAMPLES}"
        )
    else:
        try:
            embedding = model.extract_embeddings(processed)
            if embedding is not None:
                match = match_speaker(embedding)
        except Exception as exc:
            logger.exception(f"recordId={record_id} speaker processing failed: {exc}")

    accepted = match is not None and match.accepted
    speaker_id = match.speaker_id if accepted else None
    logger.info(
        f"recordId={record_id} speakerId={speaker_id or 'unknown'} "
        f"body_length={length}"
    )
    result = {"recordId": record_id, "speakerId": speaker_id}
    if match is not None:
        result["speakerVerification"] = asdict(match)
    return result


@app.get("/functions", response_class=JSONResponse)
async def list_functions():
    return {
        "save_voice_sample_enrollment": {
            "description": "saves voice sample for current voiceprint enrollment session",
            "arguments": {},
        },
        "finish_voice_sample_enrollment": {
            "description": "finishes voice sample enrollment for current voiceprint enrollment session",
            "arguments": {
                "comment": {
                    "description": "comment for voiceprint (name for example)",
                    "constraints": {
                        "type": "string-not-empty",
                        "argumentType": "string",
                    },
                }
            },
        },
    }


@app.patch("/functions")
async def update_function(payload: FunctionCallPayload):
    try:
        name = payload.name
        params = payload.arguments or {}
        if name == "save_voice_sample_enrollment":
            if not save_voice_sample_enrollment(payload.sessionId, payload.metadata):
                return JSONResponse(
                    status_code=404,
                    content={"error": "recordId was not found or has expired"},
                )
            return Response(status_code=200)
        if name == "finish_voice_sample_enrollment":
            comment = params.get("comment")
            if comment is None or not str(comment).strip():
                return JSONResponse(
                    status_code=400,
                    content={"error": "comment is required"},
                )
            if not finish_voice_sample_enrollment(
                payload.sessionId,
                payload.metadata,
                str(comment).strip(),
            ):
                return JSONResponse(
                    status_code=400,
                    content={"error": "voice enrollment could not be completed"},
                )
            return Response(status_code=200)
        return JSONResponse(status_code=400, content={"error": "unknown function"})
    except Exception as exc:
        logger.exception(f"function call failed: {exc}")
        return JSONResponse(status_code=400, content={"error": str(exc)})


@app.get("/saved-embeddings", response_class=JSONResponse)
async def get_saved_embeddings():
    return {
        "embeddings": [
            {
                "id": voice.speaker_id,
                "comment": voice.label,
                "samples": voice.samples,
                "threshold": voice.threshold,
                "minMargin": voice.min_margin,
                "embeddingModel": voice.model_id,
                "pooling": voice.pooling,
            }
            for voice in enrollment_store.list()
        ]
    }


@app.put("/state", response_class=JSONResponse)
async def get_independent_state():
    return {}


@app.post("/state", response_class=JSONResponse)
async def get_state(payload: GetFunctionsOrStatePayload):
    voice_id = (payload.metadata or {}).get("speakerId")
    voice = enrollment_store.get(voice_id) if voice_id else None
    value = voice.label if voice is not None else "voice not saved and unknown"
    return {
        "voiceprint_user_comment": {
            "description": "comment of user voiceprint if they have saved their voice before",
            "value": value,
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
