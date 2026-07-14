from __future__ import annotations

import json

import numpy as np
import pytest

from verification import (
    VoiceEnrollmentStore,
    cosine_similarity,
    rank_voice_matches,
)


MODEL_ID = "Qwen/Qwen3-ASR-0.6B"


def test_cosine_similarity_normalizes_inputs() -> None:
    assert cosine_similarity(
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.5, 0.0], dtype=np.float32),
    ) == 1.0


def test_store_averages_repeated_enrollments_and_writes_current_schema(
    tmp_path,
) -> None:
    path = tmp_path / "voices.json"
    store = VoiceEnrollmentStore(path)
    store.enroll(
        speaker_id="owner",
        label="Owner",
        embedding=np.array([1.0, 0.0], dtype=np.float32),
        threshold=0.75,
        min_margin=0.05,
        model_id=MODEL_ID,
    )
    voice = store.enroll(
        speaker_id="owner",
        label="Owner",
        embedding=np.array([0.0, 1.0], dtype=np.float32),
        threshold=0.75,
        min_margin=0.05,
        model_id=MODEL_ID,
    )

    assert voice.samples == 2
    assert np.linalg.norm(voice.embedding) == pytest.approx(1.0)
    payload = json.loads(path.read_text())
    assert payload["version"] == 1
    assert payload["voices"][0]["pooling"] == "audio_encoder_mean_std"


def test_store_reads_legacy_speaker_identifier_schema(tmp_path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "owner": {
                    "embedding": [1.0, 0.0],
                    "comment": "Owner",
                    "embedding_model": MODEL_ID,
                    "pooling": "audio_encoder_mean_std",
                }
            }
        )
    )

    voice = VoiceEnrollmentStore(path).get("owner")

    assert voice is not None
    assert voice.label == "Owner"
    assert voice.threshold == 0.75
    assert voice.min_margin == 0.05


def test_rank_matches_requires_threshold_and_best_vs_second_margin(tmp_path) -> None:
    store = VoiceEnrollmentStore(tmp_path / "voices.json")
    owner = store.enroll(
        speaker_id="owner",
        label="Owner",
        embedding=np.array([1.0, 0.0], dtype=np.float32),
        threshold=0.75,
        min_margin=0.05,
        model_id=MODEL_ID,
    )
    store.enroll(
        speaker_id="guest",
        label="Guest",
        embedding=np.array([0.99, 0.01], dtype=np.float32),
        threshold=0.75,
        min_margin=0.05,
        model_id=MODEL_ID,
    )

    best = rank_voice_matches(
        owner.embedding,
        store.list(),
        model_id=MODEL_ID,
    )[0]

    assert best.speaker_id == "owner"
    assert best.similarity >= best.threshold
    assert best.margin < best.min_margin
    assert best.accepted is False


def test_rank_matches_skips_incompatible_model(tmp_path) -> None:
    store = VoiceEnrollmentStore(tmp_path / "voices.json")
    store.enroll(
        speaker_id="old",
        label="Old",
        embedding=np.array([1.0, 0.0], dtype=np.float32),
        threshold=0.75,
        min_margin=0.05,
        model_id="another-model",
    )

    assert rank_voice_matches(
        np.array([1.0, 0.0], dtype=np.float32),
        store.list(),
        model_id=MODEL_ID,
    ) == []
