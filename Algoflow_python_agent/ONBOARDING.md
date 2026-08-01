# Onboarding / Setup

This is the Hindi LiveKit voice agent (`Algoflow_python_agent`). Follow this exactly if you're
setting the project up fresh from a clone — a lot of this repo's runtime state (models,
secrets, native build artifacts) is deliberately git-ignored and will NOT come down with `git
clone`. This doc exists specifically to cover what you have to do yourself.

## 1. Prerequisites

- **Python 3.11.x** (not 3.10, not 3.13+). `pyproject.toml` pins `requires-python =
  ">=3.11,<3.13"` — the upper bound is real: `chatterbox-tts` flips to requiring `numpy>=2.0`
  on Python ≥3.13, which conflicts with the rest of the pinned stack (`numpy==1.26.4`).
- **[uv](https://docs.astral.sh/uv/)** (recommended) or plain `pip`.
- `ffmpeg` on PATH if you touch anything that transcodes audio (not required for the
  faster_whisper/chatterbox path itself).
- A CUDA-capable GPU + drivers if you want GPU acceleration (see "GPU notes" below). CPU-only
  works too, just slower — this whole stack was validated CPU-only on macOS/Apple Silicon
  first.

## 2. Install

```bash
cd Algoflow_python_agent
uv sync            # reads uv.lock — reproducible, includes CUDA deps for Linux targets
# or, without uv:
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note: `requirements.txt` pins `vosk==0.3.45`, which has no macOS wheel (Linux/Windows only). If
you're setting up on a Mac and don't need the `vosk` STT provider, either remove that line
from a local copy or expect that one package to fail — everything else installs fine
alongside it.

## 3. Environment

```bash
cp .env.example .env
```

Then fill in real values. Minimum required to actually run the agent:
- `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` — from your LiveKit Cloud project (or
  a self-hosted server).
- `CEREBRAS_API_KEY` — the LLM is Cerebras-hosted (`ai_clients.create_cerebras_llm`), not
  local. `SAMBANOVA_API_KEY` is an alternate LLM path, only needed if you switch `agent.py` to
  use it instead.

`LOCAL_STT_PROVIDER` / `LOCAL_TTS_PROVIDER` select which local models run — see the model
section below before changing these from the `.env.example` defaults.

## 4. Model downloads — the part that actually needs manual attention

**As shipped, `.env.example` defaults to `LOCAL_STT_PROVIDER=faster_whisper` +
`LOCAL_TTS_PROVIDER=chatterbox`. This combination needs zero manual downloads** — both
self-fetch their weights from Hugging Face into `~/.cache/huggingface` the first time they
run (needs internet access on that first run; budget ~1.5GB for faster-whisper's
`large-v3-turbo` and ~2–4GB for Chatterbox Multilingual). Nothing to do here except make sure
the machine running the agent has internet access and disk space on first boot.

Everything else needs manual setup:

### Piper (`LOCAL_TTS_PROVIDER=piper`)
Needs `models/tts/hi_IN-pratham-medium.onnx` + `.onnx.json`, which are **not in this repo**
(git-ignored, and the script that was supposed to fetch them — see "Known gaps" — doesn't
exist here). To get them yourself:
```bash
mkdir -p models/tts
curl -L -o models/tts/hi_IN-pratham-medium.onnx \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx"
curl -L -o models/tts/hi_IN-pratham-medium.onnx.json \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx.json"
```

### Vosk (`LOCAL_STT_PROVIDER=vosk`)
Needs a model directory under `models/stt/`. Not present in this repo either:
```bash
mkdir -p models/stt
curl -L -o /tmp/vosk-model-hi-0.22.zip "https://alphacephei.com/vosk/models/vosk-model-hi-0.22.zip"
unzip /tmp/vosk-model-hi-0.22.zip -d models/stt/
# or the smaller/faster variant: vosk-model-small-hi-0.22.zip
```

### whisper.cpp (`LOCAL_STT_PROVIDER=whisper`)
This is the legacy path from before this repo moved to `faster_whisper`. It needs:
1. A ggml model file under `models/stt/` (e.g. `ggml-large-v3-turbo-q5_0.bin`, from
   `https://huggingface.co/ggerganov/whisper.cpp` — pick the quantization matching
   `LOCAL_STT_MODEL_PATH`'s default in `config.py`).
2. **A compiled native module at `native_build/_local_stt.*`.** The C++ source
   (`native/stt/whisper_runtime.cpp`) and its `CMakeLists.txt` are in this repo, but the
   Python build script that was supposed to invoke CMake for you (`scripts/build_native_stt.py`)
   is **missing** — see "Known gaps" below. You'd need to write that build step yourself
   (roughly: `cmake -S native/stt -B native_build/whisper_cpp_build && cmake --build
   native_build/whisper_cpp_build` then copy/rename the resulting extension module into
   `native_build/`) before this provider will import successfully.

**Recommendation:** unless you specifically need whisper.cpp's exact behavior, just use
`faster_whisper` — it replaces this whole native-build headache with a pip-installable
package and was the whole point of the STT migration (see the research/implementation history
in this conversation's earlier turns, or `RESEARCH_LOCAL_STT_TTS_LATENCY.md`).

## 5. Running locally

```bash
uv run agent.py console   # interactive terminal mode, no LiveKit room needed
uv run agent.py dev       # connects to LiveKit, hot-reloads on file changes
uv run agent.py start     # production mode
```
(`console`/`dev`/`start` are standard `livekit-agents` CLI subcommands, not custom to this repo.)

First run will be slow — faster-whisper and Chatterbox both download and load multi-GB models.
Subsequent runs use the Hugging Face cache and are much faster to start.

## 6. Known gaps — read before assuming the Docker/production deploy path works

The Dockerfile and `LIVEKIT_AGENT_DEPLOY.md` describe a build that runs:
```
uv run python scripts/download_local_audio_models.py --stt-model turbo --tts-language hi
uv run python scripts/build_native_stt.py
```
**Neither script exists anywhere in this repo or in any local checkout it was authored from —
`.gitignore` blanket-excludes the entire `scripts/` directory (not just its output), so if
these were ever written, they were never committed.** This means:
- `lk agent deploy` / `docker build` **will currently fail** at the `download_local_audio_models.py`
  step.
- The Dockerfile is also still hardcoded to the old whisper.cpp + Piper combination, not the
  `faster_whisper` + `chatterbox` combination this repo's `.env.example` now defaults to — it
  would need updating regardless of the missing-scripts issue.

This is a pre-existing gap, not something introduced by the STT/TTS migration — it just means
**the local dev path (`uv run agent.py console`, using `faster_whisper`/`chatterbox`) is
verified working end-to-end; the Docker/LiveKit Cloud deploy path is not, until someone
restores those two scripts (or rewrites the Dockerfile around the new self-downloading
providers, which need no custom scripts at all — just `uv sync` and let the first run
download weights).**

## 7. GPU notes

- `uv.lock` is a universal (cross-platform) lock — resolving it on a Linux x86_64 CUDA machine
  pulls in `torch==2.6.0`'s CUDA 12.4 runtime wheels (`nvidia-cublas-cu12`,
  `nvidia-cudnn-cu12`, etc.) automatically. After `uv sync` on that machine, sanity check with:
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  ```
- This was all built and functionally validated on an Apple M3 Pro (CPU-only path — Chatterbox
  forces CPU on macOS regardless of Metal, and `faster-whisper`'s CTranslate2 backend has no
  Metal/MPS support at all, only CPU and CUDA). GPU-accelerated latency numbers have **not**
  been measured yet — only correctness. That's the next thing to check on the actual target
  machine.
