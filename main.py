import os
import logging
import uuid
import json
import wave
from cachetools import TTLCache
import numpy as np
from fastapi import FastAPI, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from models.nemo_enc_dec_embedder import create_nemo_enc_dec_embedder_from_env
from models.vosk_embedder import create_vosk_embedding_model_from_env
from models.yandex_ecapa_tdnn_embedder import create_yandex_ecapa_tdnn_embedding_model_from_env
from utils import f32_samples_to_s16_bytes, s16_bytes_to_f32_samples

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "9010"))
AUDIO_CACHE_SIZE = int(os.getenv("AUDIO_CACHE_SIZE", "50"))
AUDIO_CACHE_TTL_SECONDS = int(os.getenv("AUDIO_CACHE_TTL_SECONDS", "600"))
VOICE_ENROLLMENT_CACHE_SIZE = int(os.getenv("VOICE_ENROLLMENT_CACHE_SIZE", "100"))
VOICE_ENROLLMENT_CACHE_TTL_SECONDS = int(os.getenv("VOICE_ENROLLMENT_CACHE_TTL_SECONDS", "1800"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))
SPEAKER_EMBEDDINGS_FILE = os.getenv("SPEAKER_EMBEDDINGS_FILE", "speaker_embeddings.json")
VOICEPRINTS_DIR = os.getenv("VOICEPRINTS_DIR")
GAIN = float(os.getenv("GAIN", "200"))
EMBEDDER = os.getenv("EMBEDDER", "yandex-ecapa-tdnn")

sample_rate = 16000

embedders = {
    "nemo-enc-dec": create_nemo_enc_dec_embedder_from_env,
    "vosk": create_vosk_embedding_model_from_env,
    "yandex-ecapa-tdnn": create_yandex_ecapa_tdnn_embedding_model_from_env,
}

if not EMBEDDER in embedders:
    raise EnvironmentError(f"Embedder {EMBEDDER} does not exist")

model = embedders[EMBEDDER](sample_rate)

audio_cache = TTLCache(maxsize=AUDIO_CACHE_SIZE, ttl=AUDIO_CACHE_TTL_SECONDS)
voice_enrollment_cache = TTLCache(maxsize=VOICE_ENROLLMENT_CACHE_SIZE, ttl=VOICE_ENROLLMENT_CACHE_TTL_SECONDS)

speaker_embeddings = {}

logger = logging.getLogger("uvicorn.access")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

app = FastAPI()

class GetFunctionsOrStatePayload(BaseModel):
    metadata: dict
    sessionId: str


class FunctionCallPayload(BaseModel):
    metadata: str
    sessionId: str
    name: str
    arguments: dict


def save_voice_sample_enrollment(sessionId: str, meta: dict):
    record_id = meta.get("recordId")
    if record_id is None:
        logger.warning(f"save_voice_sample_enrollment missing recordId: session_id={sessionId}, metadata={meta}")
        return
    sample = audio_cache.get(record_id)
    if sample is None:
        logger.warning(
            f"save_voice_sample_enrollment record not found or expired: session_id={sessionId}, recordId={record_id}")
        return
    samples = voice_enrollment_cache.get(sessionId) or []
    samples.append(sample)
    voice_enrollment_cache[sessionId] = samples
    logger.info(
        f"save_voice_sample_enrollment session_id={sessionId} recordId={record_id} samples_count={len(samples)}")


def save_voiceprint_wav(context_id: str, samples: np.ndarray):
    if VOICEPRINTS_DIR is None:
        return
    try:
        os.makedirs(VOICEPRINTS_DIR, exist_ok=True)
        path = os.path.join(VOICEPRINTS_DIR, f"{context_id}.wav")
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(f32_samples_to_s16_bytes(samples / GAIN))
        logger.info(f"voiceprint_saved path={path} session_id={context_id}")
    except Exception as e:
        logger.exception(f"voiceprint_save_failed session_id={context_id}: {e}")


def finish_voice_sample_enrollment(session_id: str, meta: dict, comment: str):
    save_voice_sample_enrollment(session_id, meta)
    samples = voice_enrollment_cache.get(session_id) or []
    if not samples:
        logger.warning(f"finish_voice_sample_enrollment no samples: session_id={session_id} comment={comment}")
        return
    data = np.concatenate(samples)
    save_voiceprint_wav(session_id, data)
    emb = model.extract_embeddings(data)
    if emb is None:
        logger.warning(f"finish_voice_sample_enrollment embedding_failed: session_id={session_id} comment={comment}")
        return
    speaker_embeddings[session_id] = (emb, comment)
    save_speaker_embeddings_to_file()
    voice_enrollment_cache.pop(session_id, None)
    logger.info(
        f"finish_voice_sample_enrollment session_id={session_id} samples_count={len(samples)} comment={comment}")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    print(a.shape, b.shape)
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an == 0 or bn == 0:
        return 0.0
    return float(np.dot(a, b) / (an * bn))


def match_speaker(embedding: np.ndarray) -> (str, float):
    if not speaker_embeddings:
        return None, None
    best_id = None
    best_sim = -1.0
    for sid, ref in speaker_embeddings.items():
        ref_emb = ref[0]
        sim = cosine_similarity(embedding, ref_emb)
        logger.info(f"cosine_similarity sid={sid} sim={sim} comment={ref[1]}")
        if sim > best_sim:
            best_sim = sim
            best_id = sid
    if best_sim >= SIMILARITY_THRESHOLD:
        return best_id, best_sim
    return None, best_sim


def save_speaker_embeddings_to_file():
    try:
        data = {sid: {"embedding": val[0].tolist(), "comment": val[1]} for sid, val in speaker_embeddings.items()}
        with open(SPEAKER_EMBEDDINGS_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        logger.exception(f"failed to save speaker embeddings: {e}")


def load_speaker_embeddings_from_file():
    try:
        with open(SPEAKER_EMBEDDINGS_FILE, "r", encoding="utf8") as f:
            data = json.load(f)
        for sid, entry in data.items():
            emb_list = entry.get("embedding")
            comment = entry.get("comment")
            if emb_list is None:
                continue
            emb = np.asarray(emb_list, dtype=np.float32)
            speaker_embeddings[sid] = (emb, comment)
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.exception(f"failed to load speaker embeddings: {e}")


load_speaker_embeddings_from_file()


def convert_and_normalize_input(raw: bytes) -> np.ndarray:
    f32_samples = s16_bytes_to_f32_samples(raw)
    return f32_samples / np.max(f32_samples) * GAIN


@app.post("/audio-metadata", response_class=JSONResponse)
async def audio_metadata(request: Request, req_sample_rate: int = Query(..., alias="sample_rate")):
    data = await request.body()
    length = len(data)
    if req_sample_rate != sample_rate:
        logger.info(f"Invalid sample_rate={req_sample_rate}, must be {sample_rate}, body_length={length}")
        return JSONResponse(status_code=500, content={"error": "sample_rate must be 16000"})
    processed = convert_and_normalize_input(data)
    record_id = str(uuid.uuid4())
    audio_cache[record_id] = processed
    speaker_id = None
    similarity = None
    try:
        emb = model.extract_embeddings(processed)
        if emb is not None:
            speaker_id, similarity = match_speaker(emb)
    except Exception as e:
        logger.exception(f"recordId={record_id} speaker processing failed: {e}")
    speaker_label = speaker_id if speaker_id is not None else "unknown"
    similarity_label = f" similarity={similarity:.4f}" if similarity is not None else ""
    logger.info(f"recordId={record_id} speakerId={speaker_label}{similarity_label} body_length={length}")
    return {"recordId": record_id, "speakerId": speaker_id}


@app.get("/functions", response_class=JSONResponse)
async def list_functions():
    return {
        "save_voice_sample_enrollment": {
            "description": "saves voice sample for current voiceprint enrollment session",
            "arguments": {}
        },
        "finish_voice_sample_enrollment": {
            "description": "finishes voice sample enrollment for current voiceprint enrollment session",
            "arguments": {
                "comment": {
                    "description": "comment for voiceprint (name for example)",
                    "constraints": {
                        "type": "string-not-empty",
                        "argumentType": "string"
                    }
                }
            }
        }
    }


@app.patch("/functions")
async def update_function(payload: FunctionCallPayload):
    try:
        name = payload.name
        params = payload.arguments or {}
        if name == "save_voice_sample_enrollment":
            save_voice_sample_enrollment(payload.sessionId, payload.metadata)
            return Response(status_code=200)
        if name == "finish_voice_sample_enrollment":
            comment = params.get("comment")
            if comment is None or (isinstance(comment, str) and comment.strip() == ""):
                return JSONResponse(status_code=400, content={"error": "comment is required"})
            finish_voice_sample_enrollment(payload.sessionId, payload.metadata, str(comment))
            return Response(status_code=200)
        return JSONResponse(status_code=400, content={"error": "unknown function"})
    except Exception as e:
        logger.exception(f"function call failed: {e}")
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/saved-embeddings", response_class=JSONResponse)
async def get_saved_embeddings():
    items = [{"id": sid, "comment": val[1]} for sid, val in speaker_embeddings.items()]
    return {"embeddings": items}

@app.put("/state", response_class=JSONResponse)
async def get_independent_state():
    return {}

@app.post("/state", response_class=JSONResponse)
async def get_state(payload: GetFunctionsOrStatePayload):
    meta = payload.metadata or {}
    voice_id = meta.get("speakerId")
    if not voice_id:
        value = "voice not saved and unknown"
    else:
        val = speaker_embeddings.get(voice_id)
        if not val:
            value = "voice not saved and unknown"
        else:
            value = val[1]
    print(value)
    return {
        "voiceprint_user_comment": {
            "description": "comment of user voiceprint if they have saved their voice before",
            "value": value
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
