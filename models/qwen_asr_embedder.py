from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np

from models.base import BaseEmbeddingModel


DEFAULT_MODEL_ID = "Qwen/Qwen3-ASR-0.6B"


class Qwen3AsrEmbeddingModel(BaseEmbeddingModel):
    """Qwen3-ASR audio-encoder embeddings used by the ASR service."""

    def __init__(self, sample_rate: int, model_id: str | None = None) -> None:
        super().__init__(sample_rate)
        self.model_id = model_id or speaker_model_id()
        self._encoder: Any | None = None
        self._feature_extractor: Any | None = None
        self._torch: Any | None = None
        self._load_lock = RLock()
        self._inference_lock = RLock()

    def _load_model(self) -> tuple[Any, Any, Any]:
        if self._encoder is not None:
            return self._encoder, self._feature_extractor, self._torch
        with self._load_lock:
            if self._encoder is not None:
                return self._encoder, self._feature_extractor, self._torch
            try:
                import torch
                from qwen_asr.core.transformers_backend.configuration_qwen3_asr import (
                    Qwen3ASRConfig,
                )
                from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
                    Qwen3ASRAudioEncoder,
                )
                from safetensors import safe_open
                from transformers import WhisperFeatureExtractor
            except ImportError as exc:
                raise RuntimeError(
                    "Qwen3-ASR speaker embedding dependencies are unavailable"
                ) from exc

            cache_dir = Path(speaker_model_cache_dir()).expanduser()
            cache_dir.mkdir(parents=True, exist_ok=True)
            config = Qwen3ASRConfig.from_json_file(
                self._ensure_metadata_file(cache_dir, "config.json")
            )
            encoder = Qwen3ASRAudioEncoder(config.thinker_config.audio_config)
            state: dict[str, Any] = {}
            with safe_open(
                self._ensure_audio_encoder_weights(cache_dir),
                framework="pt",
                device="cpu",
            ) as weights:
                for key in weights.keys():
                    state[key] = weights.get_tensor(key)

            missing, unexpected = encoder.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    "Qwen3-ASR audio encoder weights did not match: "
                    f"missing={len(missing)}, unexpected={len(unexpected)}"
                )

            device_name = resolve_torch_device(torch)
            dtype = torch.float16 if device_name.startswith("cuda") else torch.float32
            encoder.to(device=device_name, dtype=dtype)
            encoder.eval()
            feature_extractor = WhisperFeatureExtractor.from_json_file(
                self._ensure_metadata_file(cache_dir, "preprocessor_config.json")
            )

            self._encoder = encoder
            self._feature_extractor = feature_extractor
            self._torch = torch
            return encoder, feature_extractor, torch

    def _ensure_metadata_file(self, cache_dir: Path, filename: str) -> Path:
        destination = cache_dir / filename
        if destination.is_file():
            return destination
        try:
            import requests

            response = requests.get(
                f"https://huggingface.co/{self.model_id}/resolve/main/{filename}",
                timeout=(20, 60),
            )
            response.raise_for_status()
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_bytes(response.content)
            temporary.replace(destination)
            return destination
        except Exception as exc:
            raise RuntimeError(
                f"Could not download Qwen3-ASR {filename}: {exc}"
            ) from exc

    def _ensure_audio_encoder_weights(self, cache_dir: Path) -> Path:
        destination = cache_dir / "audio_encoder.safetensors"
        if destination.is_file():
            return destination
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("The requests package is required") from exc

        url = (
            f"https://huggingface.co/{self.model_id}/resolve/main/"
            "model.safetensors"
        )

        def ranged(start: int, end: int) -> Any:
            response = requests.get(
                url,
                headers={"Range": f"bytes={start}-{end}"},
                stream=True,
                timeout=(20, 300),
            )
            response.raise_for_status()
            if response.status_code != 206:
                raise RuntimeError("The model host did not honor byte-range downloads")
            return response

        try:
            header_length = struct.unpack("<Q", ranged(0, 7).content)[0]
            header = json.loads(
                ranged(8, 8 + header_length - 1).content.decode("utf-8")
            )
            prefix = "thinker.audio_tower."
            entries = {
                key[len(prefix) :]: value
                for key, value in header.items()
                if key.startswith(prefix)
            }
            if not entries:
                raise RuntimeError("Qwen3-ASR checkpoint has no audio tower")

            ordered = sorted(
                (value["data_offsets"][0], value["data_offsets"][1], key)
                for key, value in entries.items()
            )
            data_start, data_end = ordered[0][0], ordered[-1][1]
            if data_start != 0 or any(
                ordered[index - 1][1] != ordered[index][0]
                for index in range(1, len(ordered))
            ):
                raise RuntimeError("Qwen3-ASR audio weights are not contiguous")

            encoder_header: dict[str, Any] = {
                "__metadata__": {
                    "source": self.model_id,
                    "component": "thinker.audio_tower",
                }
            }
            for key, value in entries.items():
                encoder_header[key] = {
                    **value,
                    "data_offsets": [
                        value["data_offsets"][0] - data_start,
                        value["data_offsets"][1] - data_start,
                    ],
                }
            encoded_header = json.dumps(
                encoder_header, separators=(",", ":")
            ).encode("utf-8")
            encoded_header += b" " * ((-len(encoded_header)) % 8)

            response = ranged(
                8 + header_length + data_start,
                8 + header_length + data_end - 1,
            )
            temporary = destination.with_suffix(".tmp")
            downloaded = 0
            with temporary.open("wb") as output:
                output.write(struct.pack("<Q", len(encoded_header)))
                output.write(encoded_header)
                for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                    if chunk:
                        output.write(chunk)
                        downloaded += len(chunk)
            expected = data_end - data_start
            if downloaded != expected:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Incomplete audio encoder download: {downloaded}/{expected}"
                )
            temporary.replace(destination)
            return destination
        except Exception as exc:
            raise RuntimeError(
                f"Could not download Qwen3-ASR audio encoder: {exc}"
            ) from exc

    def extract_embeddings(self, samples: np.ndarray) -> np.ndarray | None:
        encoder, feature_extractor, torch = self._load_model()
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if audio.size == 0 or not np.isfinite(audio).all():
            return None
        audio = audio - float(np.mean(audio))
        if float(np.max(np.abs(audio))) <= 1e-5:
            return None

        if self.sample_rate != 16000:
            import torchaudio.functional as audio_functional

            audio = (
                audio_functional.resample(
                    torch.from_numpy(audio),
                    self.sample_rate,
                    16000,
                )
                .cpu()
                .numpy()
            )

        inputs = feature_extractor(
            audio,
            sampling_rate=16000,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_features = inputs["input_features"][0]
        feature_len = inputs["attention_mask"][0].sum().reshape(1)
        device = next(encoder.parameters()).device
        dtype = next(encoder.parameters()).dtype
        with self._inference_lock, torch.inference_mode():
            hidden = encoder(
                input_features[:, : int(feature_len.item())].to(
                    device=device,
                    dtype=dtype,
                ),
                feature_lens=feature_len.to(device=device),
            ).last_hidden_state.float()
            embedding = torch.cat(
                [hidden.mean(dim=0), hidden.std(dim=0, unbiased=False)],
                dim=0,
            )
            embedding = torch.nn.functional.normalize(embedding, dim=0)
        return embedding.cpu().numpy().astype(np.float32)


def create_qwen_asr_embedding_model_from_env(sample_rate: int) -> BaseEmbeddingModel:
    return Qwen3AsrEmbeddingModel(sample_rate=sample_rate)


def speaker_model_id() -> str:
    return os.getenv(
        "QWEN3_ASR_EMBEDDER_MODEL",
        os.getenv("QWEN_ASR_MODEL_NAME", DEFAULT_MODEL_ID),
    )


def speaker_model_cache_dir() -> str:
    return os.getenv(
        "QWEN3_ASR_EMBEDDER_CACHE",
        os.getenv(
            "SPEAKER_MODEL_CACHE_DIR",
            "pretrained_models/qwen3-asr-audio-encoder",
        ),
    )


def resolve_torch_device(torch: Any) -> str:
    preference = (
        os.getenv("SPEAKER_EMBEDDING_DEVICE")
        or os.getenv("QWEN3_ASR_EMBEDDER_DEVICE")
        or "auto"
    ).strip().lower()
    if preference in {"", "auto"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if preference.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return preference
