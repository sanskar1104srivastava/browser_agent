# In-Process STT/TTS Implementation Status

Source documents:

- `../RD_InProcess_STT_TTS_LiveKit_Design_Specification.docx`
- `../LiveKit_Local_STT_TTS_Migration_Design.docx`

Controlling process: the R&D design specification is treated as the primary implementation guide.

## Completed

- Added the R&D repository layout under `audio/`, `models/`, and `benchmarks/`.
- Rewired `agent.py` startup to load local STT and TTS runtimes before connecting the LiveKit room.
- Replaced direct Deepgram STT/TTS session wiring with local in-process LiveKit-compatible providers.
- Added local STT support for PCM frame buffering, final transcripts, endpoint flush, reset, and recognition usage metrics.
- Added local TTS streaming support for incremental text input and PCM chunk publishing through LiveKit.
- Added environment-based model config:
  - `LOCAL_STT_MODEL_PATH`
  - `LOCAL_TTS_MODEL_PATH`
  - `LOCAL_TTS_CONFIG_PATH`
  - `LOCAL_STT_SAMPLE_RATE`
  - `LOCAL_TTS_SAMPLE_RATE`
  - `LOCAL_AUDIO_CHANNELS`
  - `LOCAL_STT_SPEECH_RMS_THRESHOLD`
  - `LOCAL_STT_ENDPOINT_SILENCE_MS`
  - `LOCAL_STT_MIN_UTTERANCE_MS`
- Added `benchmarks/local_audio_smoke.py` for first-pass model load, STT latency, TTS first-audio latency, chunk count, and PCM byte metrics.
- Added `scripts/download_local_audio_models.py` to fetch the default quantized whisper.cpp STT model and Piper TTS voice from the web.

## Runtime Stack

STT:

- Runtime: whisper.cpp through a repo-owned pybind11 module
- Default model: `models/stt/ggml-tiny.en-q5_1.bin`
- Model source: `https://huggingface.co/ggerganov/whisper.cpp`
- Native source: `native/stt/whisper_runtime.cpp`
- Native build script: `scripts/build_native_stt.py`
- Frame handling: LiveKit PCM frames are buffered during an utterance and decoded on flush/end-of-speech.
- Note: this initial whisper.cpp binding emits final transcripts. True low-latency partial transcript windows still need the streaming decoder pass from the detailed spec.

TTS:

- Prototype runtime: Piper through `piper-tts`
- Default model: `models/tts/en_US-lessac-low.onnx`
- Default config: `models/tts/en_US-lessac-low.onnx.json`
- Model source: `https://huggingface.co/rhasspy/piper-voices`
- Chunk handling: Piper raw 16-bit PCM chunks are streamed to LiveKit.
- Note: this is local CPU ONNX inference, but not yet the repo-owned C++ shared library boundary used for STT.

## Still Required

- LiveKit Agent deploy only for now; do not use ECS for the current rollout.
- First LiveKit Cloud deployment:
  - create `.livekit-secrets.env`
  - run `lk agent create --region ap-south --secrets-file .livekit-secrets.env .`
- Subsequent LiveKit Cloud deployments:
  - run `lk agent deploy --secrets-file .livekit-secrets.env .`
- If running locally, run:
  - `uv sync`
  - `uv run python scripts/download_local_audio_models.py`
  - `uv run python scripts/build_native_stt.py`
- Add true whisper.cpp partial transcript windows instead of final-only utterance flush.
- Replace the prototype Python Piper TTS boundary with a repo-owned C++/ONNX Runtime shared library if the TTS side must follow the same native-boundary rule as STT.
- Run the benchmark harness once native runtimes exist.
- Run the worst-case tests from the detailed specification.
- Add production metrics export for CPU, RAM, realtime factor, and dropped frames.
