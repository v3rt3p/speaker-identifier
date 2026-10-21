import json
import sys

import numpy as np
import soundfile

from models.qwen_asr_embedder import create_qwen_asr_embedding_model_from_env

sample_rate = 16000

model = create_qwen_asr_embedding_model_from_env(sample_rate)


def convert_and_normalize_input(raw: np.ndarray) -> np.ndarray:
    samples = np.asarray(raw, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples.reshape(-1)


wav_file_name = sys.argv[1]

samples, data_sample_rate = soundfile.read(wav_file_name)
if data_sample_rate != sample_rate:
    raise ValueError(
        f"Sample rate {sample_rate} does not match sample rate {data_sample_rate}"
    )

embedding = model.extract_embeddings(convert_and_normalize_input(samples))
if embedding is None:
    raise RuntimeError("Could not extract Qwen3-ASR speaker embedding")

print(json.dumps(embedding.tolist()))
