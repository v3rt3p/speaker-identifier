# alice-speaker-identifier

Simple Speaker Identifier for alice-processor using the same Qwen3-ASR
audio-encoder embeddings as the ASR service.

Enrollment and matching mirror the SepFormer UI verifier: they use normalized
mean/std pooled embeddings from `Qwen/Qwen3-ASR-0.6B` by default, average
repeated enrollment embeddings into a persistent profile, and accept the best
speaker only when it passes `SPEAKER_VERIFICATION_THRESHOLD` and is at least
`SPEAKER_VERIFICATION_MIN_MARGIN` above the second-best speaker.

Profiles use the versioned `data/qwen3_asr_enrollments.json` schema by default.
The previous dictionary-based `speaker_embeddings.json` format is read and
migrated on the next successful enrollment. Configure either
`QWEN3_ASR_ENROLLMENT_STORE` or the legacy `SPEAKER_EMBEDDINGS_FILE` to override
the path.

Defaults match the current verifier:

- similarity threshold: `0.75`
- minimum best-vs-second margin: `0.05`
- minimum verification segment: `1.0` second of mono PCM16 at 16 kHz

`POST /audio-metadata` keeps the existing `recordId` and `speakerId` fields and
adds `speakerVerification` diagnostics whenever there is a compatible profile.
Qwen3-ASR embeddings are experimental for speaker identity; thresholds should
be calibrated with target and impostor recordings from the deployment audio
path before treating a match as authentication.
