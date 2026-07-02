# alice-speaker-identifier

Simple Speaker Identifier for alice-processor using the same Qwen3-ASR
audio-encoder embeddings as the ASR service.

Enrollment and matching use normalized mean/std pooled embeddings from
`Qwen/Qwen3-ASR-0.6B` by default. Matching accepts the best speaker only when it
passes `SPEAKER_VERIFICATION_THRESHOLD` and is at least
`SPEAKER_VERIFICATION_MIN_MARGIN` above the second-best speaker.
