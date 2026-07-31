from __future__ import annotations

import logging
import threading
import time

import numpy as np

logger = logging.getLogger("local_audio.tts.moonshine_runtime")


class MoonshineTTSRuntime:
    """
    Moonshine TTS runtime using the moonshine-voice package.

    Supports Hindi (hi-in) and 16+ other languages.
    Uses Kokoro-based voices for synthesis.
    """

    def __init__(
        self,
        language: str = "hi-in",
        voice: str | None = None,
        speed: float | None = None,
    ) -> None:
        self.language = language
        self._voice = voice
        self._speed = speed
        self._tts = None
        self._lock = threading.Lock()
        self.sample_rate = 24000
        self.num_channels = 1
        logger.info(
            "stage=moonshine_tts_init language=%s voice=%s speed=%s",
            language,
            voice,
            speed,
        )

    def _ensure_loaded(self) -> None:
        if self._tts is not None:
            return
        with self._lock:
            if self._tts is not None:
                return
            try:
                from moonshine_voice import TextToSpeech
            except ImportError as exc:
                raise RuntimeError(
                    "moonshine-voice is not installed. "
                    "Run `pip install moonshine-voice`."
                ) from exc

            started = time.perf_counter()
            kwargs = {"language": self.language}
            if self._voice:
                kwargs["voice"] = self._voice
            self._tts = TextToSpeech(**kwargs)
            logger.info(
                "stage=moonshine_tts_loaded elapsed_ms=%.1f",
                (time.perf_counter() - started) * 1000,
            )

    def preload(self) -> None:
        """Load the synthesizer eagerly at startup (not lazily on first utterance)."""
        self._ensure_loaded()

    def synthesize_stream(self, text: str) -> list[bytes]:
        """
        Synthesize text and return list of PCM int16 chunks.
        Each chunk is a numpy array of audio samples.
        """
        self._ensure_loaded()
        started = time.perf_counter()
        logger.info("stage=moonshine_tts_synthesize_start text_chars=%s", len(text))

        synth_kwargs = {}
        if self._speed is not None:
            synth_kwargs["speed"] = self._speed
        audio_data, sample_rate = self._tts.synthesize(text, **synth_kwargs)
        if sample_rate != self.sample_rate:
            logger.warning(
                "stage=moonshine_tts_sample_rate_mismatch expected=%s got=%s",
                self.sample_rate,
                sample_rate,
            )
            self.sample_rate = sample_rate

        if audio_data is None or len(audio_data) == 0:
            logger.warning("stage=moonshine_tts_synthesize_empty text_chars=%s", len(text))
            return []

        if isinstance(audio_data, list):
            audio_data = np.array(audio_data, dtype=np.float32)

        if isinstance(audio_data, np.ndarray):
            if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                audio_int16 = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
            else:
                audio_int16 = audio_data.astype(np.int16)
        else:
            audio_int16 = np.frombuffer(audio_data, dtype=np.int16)

        pcm_bytes = audio_int16.tobytes()
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "stage=moonshine_tts_synthesize_done elapsed_ms=%.1f audio_ms=%.1f pcm_bytes=%s",
            elapsed_ms,
            (len(audio_int16) / self.sample_rate) * 1000,
            len(pcm_bytes),
        )

        chunk_size = self.sample_rate // 10
        chunks = []
        for i in range(0, len(pcm_bytes), chunk_size * 2):
            chunk = pcm_bytes[i : i + chunk_size * 2]
            if chunk:
                chunks.append(chunk)

        return chunks

    def close(self) -> None:
        self._tts = None
        logger.info("stage=moonshine_tts_closed")
