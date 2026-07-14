from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


POOLING_ID = "audio_encoder_mean_std"


class VerificationError(RuntimeError):
    """Raised when a voice profile cannot be validated or persisted."""


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    values = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(values))
    if values.size == 0 or norm <= 1e-12:
        raise VerificationError("Cannot use an empty embedding.")
    return values / norm


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    left = normalize_embedding(first)
    right = normalize_embedding(second)
    if left.shape != right.shape:
        raise VerificationError(
            f"Embedding shapes do not match: {left.shape} and {right.shape}."
        )
    return float(np.dot(left, right))


@dataclass(frozen=True)
class EnrolledVoice:
    speaker_id: str
    label: str
    embedding: np.ndarray
    threshold: float
    min_margin: float
    samples: int
    updated_at: str
    model_id: str
    pooling: str = POOLING_ID


@dataclass(frozen=True)
class VoiceMatch:
    speaker_id: str
    label: str
    similarity: float
    threshold: float
    min_margin: float
    margin: float
    accepted: bool


def rank_voice_matches(
    embedding: np.ndarray,
    voices: list[EnrolledVoice],
    *,
    model_id: str,
    threshold_override: float | None = None,
    min_margin_override: float | None = None,
) -> list[VoiceMatch]:
    compatible = [
        voice
        for voice in voices
        if voice.model_id == model_id and voice.pooling == POOLING_ID
    ]
    ranked = sorted(
        (
            (voice, cosine_similarity(embedding, voice.embedding))
            for voice in compatible
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    results: list[VoiceMatch] = []
    for index, (voice, similarity) in enumerate(ranked):
        threshold = (
            float(threshold_override)
            if threshold_override is not None
            else voice.threshold
        )
        min_margin = (
            float(min_margin_override)
            if min_margin_override is not None
            else voice.min_margin
        )
        competing_score = max(
            (
                score
                for other_index, (_other, score) in enumerate(ranked)
                if other_index != index
            ),
            default=-1.0,
        )
        margin = similarity - competing_score
        results.append(
            VoiceMatch(
                speaker_id=voice.speaker_id,
                label=voice.label,
                similarity=similarity,
                threshold=threshold,
                min_margin=min_margin,
                margin=margin,
                accepted=(
                    index == 0
                    and similarity >= threshold
                    and margin >= min_margin
                ),
            )
        )
    return results


class VoiceEnrollmentStore:
    """Current verifier-compatible profile storage with legacy-file migration."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self._lock = threading.RLock()

    def list(self) -> list[EnrolledVoice]:
        with self._lock:
            payload = self._read()
        return [self._decode(item) for item in payload.get("voices", [])]

    def get(self, speaker_id: str) -> EnrolledVoice | None:
        return next(
            (voice for voice in self.list() if voice.speaker_id == speaker_id),
            None,
        )

    def enroll(
        self,
        *,
        speaker_id: str,
        label: str,
        embedding: np.ndarray,
        threshold: float,
        min_margin: float,
        model_id: str,
        append: bool = True,
    ) -> EnrolledVoice:
        normalized = normalize_embedding(embedding)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            payload = self._read()
            voices = payload.setdefault("voices", [])
            existing = next(
                (item for item in voices if item.get("speaker_id") == speaker_id),
                None,
            )
            samples = 1
            compatible_existing = (
                existing is not None
                and existing.get("model_id") == model_id
                and existing.get("pooling", POOLING_ID) == POOLING_ID
            )
            if compatible_existing and append:
                old = normalize_embedding(
                    np.asarray(existing["embedding"], dtype=np.float32)
                )
                samples = int(existing.get("samples", 1)) + 1
                normalized = normalize_embedding(
                    old * (samples - 1) / samples + normalized / samples
                )
            record = {
                "speaker_id": speaker_id,
                "label": label or speaker_id,
                "embedding": normalized.astype(float).tolist(),
                "threshold": float(threshold),
                "min_margin": float(min_margin),
                "samples": samples,
                "updated_at": now,
                "model_id": model_id,
                "pooling": POOLING_ID,
            }
            if existing is None:
                voices.append(record)
            else:
                voices[voices.index(existing)] = record
            self._write(payload)
        return self._decode(record)

    def _read(self) -> dict:
        if not self.path.exists():
            return {"version": 1, "voices": []}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise VerificationError(
                f"Could not read enrollment store {self.path}: {exc}"
            ) from exc
        if isinstance(payload, dict) and isinstance(payload.get("voices"), list):
            return payload
        if not isinstance(payload, dict):
            raise VerificationError("Enrollment store must contain a JSON object.")

        # Previous speaker-identifier releases stored profiles in a mapping.
        voices = []
        for speaker_id, entry in payload.items():
            if not isinstance(entry, dict) or entry.get("embedding") is None:
                continue
            voices.append(
                {
                    "speaker_id": speaker_id,
                    "label": entry.get("comment") or speaker_id,
                    "embedding": entry["embedding"],
                    "threshold": entry.get("threshold", 0.75),
                    "min_margin": entry.get("min_margin", 0.05),
                    "samples": entry.get("samples", 1),
                    "updated_at": entry.get("updated_at", ""),
                    "model_id": entry.get("embedding_model", ""),
                    "pooling": entry.get("pooling", POOLING_ID),
                }
            )
        return {"version": 1, "voices": voices}

    def _write(self, payload: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    @staticmethod
    def _decode(item: dict) -> EnrolledVoice:
        return EnrolledVoice(
            speaker_id=str(item["speaker_id"]),
            label=str(item.get("label") or item["speaker_id"]),
            embedding=normalize_embedding(
                np.asarray(item["embedding"], dtype=np.float32)
            ),
            threshold=float(item.get("threshold", 0.75)),
            min_margin=float(item.get("min_margin", 0.05)),
            samples=int(item.get("samples", 1)),
            updated_at=str(item.get("updated_at", "")),
            model_id=str(item.get("model_id", "")),
            pooling=str(item.get("pooling", POOLING_ID)),
        )
