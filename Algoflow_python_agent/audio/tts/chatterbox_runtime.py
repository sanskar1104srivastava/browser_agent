from __future__ import annotations

import logging
from pathlib import Path
import platform
import threading
import time

import numpy as np

logger = logging.getLogger("local_audio.tts.chatterbox_runtime")

# Lower temp + higher cfg = clearer speech for languages the base model is
# less confident on. Mirrors the defaults voicebox's chatterbox backend uses.
_GLOBAL_DEFAULTS = {
    "exaggeration": 0.5,
    "cfg_weight": 0.5,
    "temperature": 0.8,
    "repetition_penalty": 2.0,
}


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if platform.system() == "Darwin":
        # ChatterboxMultilingualTTS has known MPS tensor issues; CPU is the
        # only reliable backend on macOS regardless of Apple Silicon GPU.
        return "cpu"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        logger.warning("stage=chatterbox_device_autodetect_failed falling_back=cpu")
        return "cpu"


class ChatterboxTTSRuntime:
    """
    Chatterbox Multilingual TTS runtime using the chatterbox-tts package.

    Zero-shot voice cloning across 23 languages including Hindi. Not
    natively streaming — model.generate() produces the full utterance in
    one pass, which is then chunked for delivery, same as the Edge runtime.
    """

    def __init__(
        self,
        *,
        language: str = "hi",
        device: str = "auto",
        voice_sample_path: Path | None = None,
        exaggeration: float | None = None,
        cfg_weight: float | None = None,
        temperature: float | None = None,
        repetition_penalty: float | None = None,
    ) -> None:
        self.language = language
        self._voice_sample_path = str(voice_sample_path) if voice_sample_path else None
        self._gen_kwargs = {
            "exaggeration": exaggeration if exaggeration is not None else _GLOBAL_DEFAULTS["exaggeration"],
            "cfg_weight": cfg_weight if cfg_weight is not None else _GLOBAL_DEFAULTS["cfg_weight"],
            "temperature": temperature if temperature is not None else _GLOBAL_DEFAULTS["temperature"],
            "repetition_penalty": (
                repetition_penalty if repetition_penalty is not None else _GLOBAL_DEFAULTS["repetition_penalty"]
            ),
        }
        self.sample_rate = 24000
        self.num_channels = 1
        self._model = None
        self._lock = threading.Lock()
        self.device = _resolve_device(device)
        logger.info(
            "stage=chatterbox_tts_init language=%s device=%s voice_sample=%s gen_kwargs=%s",
            language,
            self.device,
            self._voice_sample_path,
            self._gen_kwargs,
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            except ImportError as exc:
                raise RuntimeError(
                    "chatterbox-tts is not installed. Run `pip install chatterbox-tts`."
                ) from exc

            started = time.perf_counter()
            logger.info("stage=chatterbox_tts_load_start device=%s", self.device)
            self._model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
            self.sample_rate = getattr(self._model, "sr", None) or getattr(self._model, "sample_rate", 24000)
            logger.info(
                "stage=chatterbox_tts_load_done elapsed_ms=%.1f sample_rate=%s",
                (time.perf_counter() - started) * 1000,
                self.sample_rate,
            )

    def preload(self) -> None:
        """Load the model eagerly at startup (not lazily on first utterance)."""
        self._ensure_loaded()

    def synthesize_stream(self, text: str) -> list[bytes]:
        """Synthesize text and return list of PCM int16 chunks."""
        self._ensure_loaded()
        import torch

        started = time.perf_counter()
        logger.info(
            "stage=chatterbox_tts_synthesize_start text_chars=%s language=%s",
            len(text),
            self.language,
        )

        wav = self._model.generate(
            text,
            language_id=self.language,
            audio_prompt_path=self._voice_sample_path,
            **self._gen_kwargs,
        )

        if isinstance(wav, torch.Tensor):
            audio = wav.squeeze().detach().cpu().numpy().astype(np.float32)
        else:
            audio = np.asarray(wav, dtype=np.float32)

        if audio.size == 0:
            logger.warning("stage=chatterbox_tts_synthesize_empty text_chars=%s", len(text))
            return []

        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        pcm_bytes = audio_int16.tobytes()

        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "stage=chatterbox_tts_synthesize_done elapsed_ms=%.1f audio_ms=%.1f pcm_bytes=%s",
            elapsed_ms,
            (len(audio_int16) / self.sample_rate) * 1000,
            len(pcm_bytes),
        )

        chunk_size = self.sample_rate // 10  # 100ms chunks
        chunks = []
        for i in range(0, len(pcm_bytes), chunk_size * 2):
            chunk = pcm_bytes[i : i + chunk_size * 2]
            if chunk:
                chunks.append(chunk)
        return chunks

    def close(self) -> None:
        self._model = None
        logger.info("stage=chatterbox_tts_closed")
