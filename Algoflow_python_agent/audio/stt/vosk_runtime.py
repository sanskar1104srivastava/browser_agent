from __future__ import annotations

import json
import logging
from pathlib import Path
import time

from livekit import rtc

from .runtime import STTResult, _clean_transcript

logger = logging.getLogger("local_audio.stt.vosk_runtime")


def _pcm_i16_to_mono(pcm: bytes, channels: int) -> bytes:
    if channels <= 0:
        raise ValueError("num_channels must be positive")
    if channels == 1:
        return pcm
    sample_count = len(pcm) // 2
    samples = memoryview(pcm[: sample_count * 2]).cast("h")
    mono = bytearray()
    for index in range(0, len(samples), channels):
        frame = samples[index : index + channels]
        if len(frame) < channels:
            break
        mixed = int(sum(int(sample) for sample in frame) / channels)
        mono.extend(int(mixed).to_bytes(2, byteorder="little", signed=True))
    return bytes(mono)


class VoskSTTRuntime:
    """
    Local Vosk/Kaldi STT runtime.

    The Vosk model is loaded once and each LiveKit stream gets its own
    recognizer session, which keeps streaming state isolated per caller.
    """

    def __init__(self, model_path: Path, *, sample_rate: int = 16000, language: str = "hi") -> None:
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.language = language
        logger.info("stage=vosk_import_start")
        try:
            from vosk import Model, SetLogLevel
        except ImportError as exc:
            logger.exception("stage=vosk_import_failed error_type=%s", type(exc).__name__)
            raise RuntimeError("vosk is not installed. Run `uv add vosk==0.3.45`.") from exc

        SetLogLevel(-1)
        started = time.perf_counter()
        logger.info(
            "stage=vosk_model_load_start model=%s sample_rate=%s language=%s",
            model_path,
            sample_rate,
            language,
        )
        self._model = Model(str(model_path))
        logger.info("stage=vosk_model_load_done elapsed_ms=%.1f", (time.perf_counter() - started) * 1000)

    def create_session(self) -> "VoskSTTSession":
        return VoskSTTSession(self._model, sample_rate=self.sample_rate, language=self.language)

    def transcribe_frame(self, frame: rtc.AudioFrame) -> STTResult:
        session = self.create_session()
        session.process_frame(frame)
        return session.flush()

    def close(self) -> None:
        self._model = None
        logger.info("stage=vosk_runtime_closed")


class VoskSTTSession:
    def __init__(self, model, *, sample_rate: int, language: str) -> None:
        from vosk import KaldiRecognizer

        self.sample_rate = sample_rate
        self.language = language
        self._model = model
        self._recognizer = KaldiRecognizer(self._model, sample_rate)
        self._audio_duration = 0.0
        self._last_partial = ""
        self._last_final = ""
        self._last_raw_partial = ""
        self._last_raw_result = ""
        self._last_raw_final = ""

    def process_frame(self, frame: rtc.AudioFrame) -> STTResult:
        return self.accept_pcm_i16(
            frame.data.tobytes(),
            sample_rate=frame.sample_rate,
            channels=frame.num_channels,
            duration=frame.duration,
        )

    def accept_pcm_i16(self, pcm: bytes, *, sample_rate: int, channels: int, duration: float = 0.0) -> STTResult:
        if sample_rate != self.sample_rate:
            raise ValueError(f"input sample rate {sample_rate} does not match Vosk sample rate {self.sample_rate}")

        started = time.perf_counter()
        mono_pcm = _pcm_i16_to_mono(pcm, channels)
        self._audio_duration += duration
        accepted = self._recognizer.AcceptWaveform(mono_pcm)
        elapsed_ms = (time.perf_counter() - started) * 1000

        if accepted:
            self._last_raw_result = self._recognizer.Result()
            text = _clean_transcript(json.loads(self._last_raw_result).get("text", ""))
            if text:
                self._last_final = text
            logger.info(
                "stage=vosk_frame_final elapsed_ms=%.1f audio_ms=%.1f chars=%s",
                elapsed_ms,
                self._audio_duration * 1000,
                len(text),
            )
            return STTResult(
                final=text,
                speech_started=True,
                speech_ended=True,
                audio_duration=duration,
                confidence=1.0 if text else 0.0,
            )

        logger.debug(
            "stage=vosk_frame_accepted elapsed_ms=%.1f audio_ms=%.1f bytes=%s",
            elapsed_ms,
            self._audio_duration * 1000,
            len(mono_pcm),
        )
        return STTResult(speech_started=bool(mono_pcm), audio_duration=duration)

    def partial(self) -> STTResult:
        started = time.perf_counter()
        self._last_raw_partial = self._recognizer.PartialResult()
        text = _clean_transcript(json.loads(self._last_raw_partial).get("partial", ""))
        elapsed_ms = (time.perf_counter() - started) * 1000
        if text and text != self._last_partial:
            self._last_partial = text
            logger.info("stage=vosk_partial_done elapsed_ms=%.1f chars=%s", elapsed_ms, len(text))
            return STTResult(partial=text, confidence=1.0)
        logger.debug("stage=vosk_partial_empty elapsed_ms=%.1f duplicate=%s", elapsed_ms, bool(text))
        return STTResult()

    def flush(self) -> STTResult:
        started = time.perf_counter()
        self._last_raw_final = self._recognizer.FinalResult()
        text = _clean_transcript(json.loads(self._last_raw_final).get("text", ""))
        if not text:
            text = self._last_final or self._last_partial
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "stage=vosk_flush_done elapsed_ms=%.1f final_chars=%s audio_ms=%.1f",
            elapsed_ms,
            len(text),
            self._audio_duration * 1000,
        )
        return STTResult(
            final=text,
            speech_ended=True,
            audio_duration=self._audio_duration,
            confidence=1.0 if text else 0.0,
            segments=[text] if text else [],
        )

    @property
    def last_raw_partial(self) -> str:
        return self._last_raw_partial

    @property
    def last_raw_result(self) -> str:
        return self._last_raw_result

    @property
    def last_raw_final(self) -> str:
        return self._last_raw_final

    def reset(self) -> None:
        from vosk import KaldiRecognizer

        self._recognizer = KaldiRecognizer(self._model, self.sample_rate)
        self._audio_duration = 0.0
        self._last_partial = ""
        self._last_final = ""
        self._last_raw_partial = ""
        self._last_raw_result = ""
        self._last_raw_final = ""
