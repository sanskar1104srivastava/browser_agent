from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path


@dataclass(frozen=True)
class LocalAudioConfig:
    stt_provider: str
    stt_model_path: Path
    vosk_model_path: Path
    tts_model_path: Path
    tts_config_path: Path
    tts_provider: str = "piper"
    stt_sample_rate: int = 16000
    tts_sample_rate: int = 22050
    num_channels: int = 1
    stt_speech_rms_threshold: int = 350
    stt_endpoint_silence_ms: float = 300.0
    stt_min_utterance_ms: float = 200.0
    stt_partial_interval_ms: float = 0.0
    stt_min_partial_audio_ms: float = 1200.0
    stt_partial_window_ms: float = 1200.0
    vosk_partial_interval_ms: float = 100.0
    vosk_min_partial_audio_ms: float = 150.0
    vosk_preroll_ms: float = 200.0
    vosk_use_rms_gate: bool = False
    stt_language: str = "hi"
    tts_chunk_max_chars: int = 20

    @staticmethod
    def _piper_sample_rate(config_path: Path, fallback: int) -> int:
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            return int(data.get("audio", {}).get("sample_rate") or fallback)
        except Exception:
            return fallback

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @classmethod
    def from_env(cls) -> "LocalAudioConfig":
        base_dir = Path(__file__).resolve().parent
        default_vosk_model_path = base_dir / "models" / "stt" / "vosk-model-hi-0.22"
        if not default_vosk_model_path.exists():
            default_vosk_model_path = base_dir / "models" / "stt" / "vosk-model-small-hi-0.22"
        tts_config_path = Path(
            os.getenv(
                "LOCAL_TTS_CONFIG_PATH",
                str(base_dir / "models" / "tts" / "hi_IN-pratham-medium.onnx.json"),
            )
        ).expanduser().resolve()
        tts_sample_rate = int(
            os.getenv(
                "LOCAL_TTS_SAMPLE_RATE",
                str(cls._piper_sample_rate(tts_config_path, 22050)),
            )
        )
        return cls(
            stt_provider=os.getenv("LOCAL_STT_PROVIDER", "vosk").strip().lower(),
            tts_provider=os.getenv("LOCAL_TTS_PROVIDER", "piper").strip().lower(),
            stt_model_path=Path(
                os.getenv(
                    "LOCAL_STT_MODEL_PATH",
                    str(base_dir / "models" / "stt" / "ggml-large-v3-turbo-q5_0.bin"),
                )
            ).expanduser().resolve(),
            vosk_model_path=Path(
                os.getenv(
                    "LOCAL_VOSK_MODEL_PATH",
                    str(default_vosk_model_path),
                )
            ).expanduser().resolve(),
            tts_model_path=Path(
                os.getenv(
                    "LOCAL_TTS_MODEL_PATH",
                    str(base_dir / "models" / "tts" / "hi_IN-pratham-medium.onnx"),
                )
            ).expanduser().resolve(),
            tts_config_path=tts_config_path,
            stt_sample_rate=int(os.getenv("LOCAL_STT_SAMPLE_RATE", "16000")),
            tts_sample_rate=tts_sample_rate,
            num_channels=int(os.getenv("LOCAL_AUDIO_CHANNELS", "1")),
            stt_speech_rms_threshold=int(os.getenv("LOCAL_STT_SPEECH_RMS_THRESHOLD", "300")),
            stt_endpoint_silence_ms=float(os.getenv("LOCAL_STT_ENDPOINT_SILENCE_MS", "450")),
            stt_min_utterance_ms=float(os.getenv("LOCAL_STT_MIN_UTTERANCE_MS", "400")),
            stt_partial_interval_ms=float(os.getenv("LOCAL_STT_PARTIAL_INTERVAL_MS", "0")),
            stt_min_partial_audio_ms=float(os.getenv("LOCAL_STT_MIN_PARTIAL_AUDIO_MS", "1200")),
            stt_partial_window_ms=float(os.getenv("LOCAL_STT_PARTIAL_WINDOW_MS", "1200")),
            vosk_partial_interval_ms=float(os.getenv("LOCAL_VOSK_PARTIAL_INTERVAL_MS", "100")),
            vosk_min_partial_audio_ms=float(os.getenv("LOCAL_VOSK_MIN_PARTIAL_AUDIO_MS", "150")),
            vosk_preroll_ms=float(os.getenv("LOCAL_VOSK_PREROLL_MS", "200")),
            vosk_use_rms_gate=cls._env_bool("LOCAL_VOSK_USE_RMS_GATE", False),
            stt_language=os.getenv("LOCAL_STT_LANGUAGE", "hi"),
            tts_chunk_max_chars=int(os.getenv("LOCAL_TTS_CHUNK_MAX_CHARS", "20")),
        )

    def validate(self) -> None:
        if self.stt_provider not in {"whisper", "vosk"}:
            raise RuntimeError(f"LOCAL_STT_PROVIDER must be one of: whisper, vosk. Got: {self.stt_provider}")

        if self.tts_provider not in {"piper", "moonshine"}:
            raise RuntimeError(f"LOCAL_TTS_PROVIDER must be one of: piper, moonshine. Got: {self.tts_provider}")

        if self.tts_provider == "piper":
            for name, path in (
                ("LOCAL_TTS_MODEL_PATH", self.tts_model_path),
                ("LOCAL_TTS_CONFIG_PATH", self.tts_config_path),
            ):
                if not path.exists():
                    raise RuntimeError(f"{name} does not exist: {path}")

        if self.stt_provider == "vosk":
            if not self.vosk_model_path.exists():
                raise RuntimeError(f"LOCAL_VOSK_MODEL_PATH does not exist: {self.vosk_model_path}")
            if not self.vosk_model_path.is_dir():
                raise RuntimeError(f"LOCAL_VOSK_MODEL_PATH must be a Vosk model directory: {self.vosk_model_path}")
            return

        if not self.stt_model_path.exists():
            raise RuntimeError(f"LOCAL_STT_MODEL_PATH does not exist: {self.stt_model_path}")
