from __future__ import annotations

import logging
from pathlib import Path
import time

import numpy as np
from livekit import rtc

from .runtime import STTResult, _filter_transcript

logger = logging.getLogger("local_audio.stt.faster_whisper_runtime")


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import ctranslate2

        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        logger.warning("stage=stt_device_autodetect_failed falling_back=cpu")
        return "cpu"


def _resolve_compute_type(requested: str | None, device: str) -> str:
    if requested:
        return requested
    return "float16" if device == "cuda" else "int8"


class FasterWhisperSTTRuntime:
    """
    Local in-process CTranslate2/faster-whisper runtime.

    Batch-only (no incremental partials): PCM frames are buffered in memory
    and decoded in one pass on flush(), same contract as LocalSTTRuntime so
    it can be dropped in wherever the whisper.cpp runtime was used.
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        sample_rate: int = 16000,
        language: str = "hi",
        device: str = "auto",
        compute_type: str | None = None,
        beam_size: int = 1,
    ) -> None:
        self.model_path = Path(model_name_or_path)
        self.sample_rate = sample_rate
        self.language = language
        self.beam_size = beam_size
        self._buffer = bytearray()

        resolved_device = _resolve_device(device)
        resolved_compute_type = _resolve_compute_type(compute_type, resolved_device)

        logger.info(
            "stage=stt_native_import_start provider=faster_whisper model=%s device=%s compute_type=%s",
            model_name_or_path,
            resolved_device,
            resolved_compute_type,
        )
        from faster_whisper import WhisperModel

        started = time.perf_counter()
        self._model = WhisperModel(
            model_name_or_path,
            device=resolved_device,
            compute_type=resolved_compute_type,
        )
        self.device = resolved_device
        self.compute_type = resolved_compute_type
        logger.info(
            "stage=stt_native_model_create_done elapsed_ms=%.1f device=%s compute_type=%s",
            (time.perf_counter() - started) * 1000,
            resolved_device,
            resolved_compute_type,
        )

    def warmup(self) -> None:
        started = time.perf_counter()
        try:
            silent = np.zeros(self.sample_rate, dtype=np.float32)
            segments, _ = self._model.transcribe(silent, language=self.language, beam_size=self.beam_size)
            list(segments)
        finally:
            self.reset()
        logger.info("stage=stt_warmup_done elapsed_ms=%.1f", (time.perf_counter() - started) * 1000)

    def process_frame(self, frame: rtc.AudioFrame) -> STTResult:
        started = time.perf_counter()
        self._buffer.extend(frame.data.tobytes())
        logger.debug(
            "stage=stt_frame_accepted sample_rate=%s channels=%s samples_per_channel=%s duration_ms=%.1f elapsed_ms=%.1f",
            frame.sample_rate,
            frame.num_channels,
            frame.samples_per_channel,
            frame.duration * 1000,
            (time.perf_counter() - started) * 1000,
        )
        return STTResult(audio_duration=frame.duration)

    def flush(self) -> STTResult:
        started = time.perf_counter()
        logger.info("stage=stt_flush_start provider=faster_whisper buffered_bytes=%s", len(self._buffer))

        if not self._buffer:
            return STTResult(final="", speech_ended=True)

        audio = np.frombuffer(bytes(self._buffer), dtype=np.int16).astype(np.float32) / 32768.0
        segments, info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            condition_on_previous_text=False,
        )
        segments = list(segments)
        segment_texts = [seg.text.strip() for seg in segments if seg.text.strip()]
        raw_text = " ".join(segment_texts)
        no_speech_prob = max((seg.no_speech_prob for seg in segments), default=0.0)
        avg_logprob = sum((seg.avg_logprob for seg in segments), 0.0) / len(segments) if segments else 0.0
        confidence = max(0.0, min(1.0, 1.0 + avg_logprob))

        text, filtered, filter_reason = _filter_transcript(raw_text, {"no_speech_prob": no_speech_prob})

        logger.info(
            "stage=stt_flush_done elapsed_ms=%.1f final_chars=%s segment_count=%s confidence=%.3f "
            "no_speech_prob=%.3f filtered=%s filter_reason=%s language=%s language_probability=%.3f",
            (time.perf_counter() - started) * 1000,
            len(text),
            len(segments),
            confidence,
            no_speech_prob,
            filtered,
            filter_reason,
            info.language,
            info.language_probability,
        )
        return STTResult(
            final=text,
            speech_ended=True,
            confidence=confidence,
            segments=segment_texts,
            no_speech_prob=no_speech_prob,
            filtered=filtered,
            filter_reason=filter_reason,
        )

    def partial(self, max_window_ms: float) -> STTResult:
        return STTResult(partial="")

    def reset(self) -> None:
        self._buffer = bytearray()
        logger.debug("stage=stt_runtime_reset provider=faster_whisper")

    def close(self) -> None:
        self._model = None
        logger.info("stage=stt_runtime_closed provider=faster_whisper")
