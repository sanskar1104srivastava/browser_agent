# LiveKit Agent Deploy

This repo now targets LiveKit Agent deployment first. Do not use ECS for the current deployment path.

## First Deployment

Use the LiveKit CLI from this directory:

```powershell
lk project list
lk project set-default "algoflow"
lk agent create --region ap-south --secrets-file .livekit-secrets.env .
```

`lk agent create` registers the agent, writes `livekit.toml`, uploads this directory, builds the Dockerfile in LiveKit Cloud, and deploys the first version.

## Subsequent Deployments

After `livekit.toml` exists:

```powershell
lk agent deploy --secrets-file .livekit-secrets.env .
lk agent status
lk agent logs --log-type build
lk agent logs --log-type deploy
```

## Required Secrets

Create `.livekit-secrets.env` locally. Do not commit it.

```env
CEREBRAS_API_KEY=...
```

The local STT and TTS models are downloaded during the Docker build by `scripts/download_local_audio_models.py`; they do not need to be committed or passed as secrets. The build currently downloads the whisper.cpp `ggml-small-q8_0.bin` model for Hindi STT accuracy through the repo-owned native C++ shared library. Vosk remains available as an opt-in provider through `LOCAL_STT_PROVIDER=vosk`.

## Build Path

The Dockerfile performs these LiveKit Cloud build steps:

1. Install system build/runtime packages.
2. `uv sync`
3. Download local whisper.cpp/Piper model assets.
4. Build the repo-owned native whisper.cpp STT shared library.
5. Run `uv run agent.py download-files`.
6. Start with `uv run agent.py start`.
