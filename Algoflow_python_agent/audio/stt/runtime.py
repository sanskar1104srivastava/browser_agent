from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import re
import time

from livekit import rtc

logger = logging.getLogger("local_audio.stt.runtime")

_NO_SPEECH_TRANSCRIPTS = {
    "",
    "[blank_audio]",
    "[blank audio]",
    "(blank_audio)",
    "(blank audio)",
    "[silence]",
    "(silence)",
}


def _clean_transcript(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    normalized = cleaned.lower().rstrip(".!?")
    if normalized in _NO_SPEECH_TRANSCRIPTS:
        return ""
    return cleaned


def _repetition_reason(text: str) -> str:
    normalized = _clean_transcript(text)
    if not normalized:
        return ""

    words = normalized.split()

    # Only filter for very explicit failure modes
    repeated_run = 1
    max_repeated_run = 1
    for index in range(1, len(words)):
        if words[index] == words[index - 1] and len(words[index]) > 2:
            repeated_run += 1
            max_repeated_run = max(max_repeated_run, repeated_run)
        else:
            repeated_run = 1
    if max_repeated_run >= 4:
        return "repeated_word_run"

    if len(words) >= 8:
        # "aaa bbb ccc aaa bbb" pattern where the same word fills 60%+ of the utterance
        counts: dict[str, int] = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1
        if max(counts.values(), default=0) / len(words) >= 0.6:
            return "repeated_word_dominated"

    return ""


def _filter_transcript(text: str, raw_result: dict) -> tuple[str, bool, str]:
    cleaned = _clean_transcript(text)
    if not cleaned:
        return "", False, ""

    no_speech_prob = float(raw_result.get("no_speech_prob") or 0.0)
    max_no_speech_prob = float(os.getenv("LOCAL_STT_MAX_NO_SPEECH_PROB", "0.65"))
    if no_speech_prob >= max_no_speech_prob:
        return "", True, f"no_speech_prob={no_speech_prob:.3f}"

    reason = _repetition_reason(cleaned)
    if reason:
        logger.info(
            "stage=stt_filter_trigger_raw text=%r raw=%r reason=%s",
            cleaned,
            text,
            reason,
        )
        return "", True, reason

    return cleaned, False, ""


@dataclass(frozen=True)
class STTResult:
    partial: str = ""
    final: str = ""
    speech_started: bool = False
    speech_ended: bool = False
    audio_duration: float = 0.0
    confidence: float = 0.0
    segments: list[str] = field(default_factory=list)
    no_speech_prob: float = 0.0
    audio_energy: float = 0.0
    filtered: bool = False
    filter_reason: str = ""


class LocalSTTRuntime:
    """
    Local in-process whisper.cpp runtime through the repo's pybind11 module.

    The ggml model is loaded once at startup. Incoming PCM frames are buffered
    for low-latency windows and decoded on flush/end-of-utterance.
    """

    def __init__(self, model_path: Path, *, sample_rate: int = 16000, language: str = "hi") -> None:
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.language = language
        logger.info("stage=stt_native_import_start")
        try:
            from native_build import _local_stt
        except ImportError as exc:
            logger.exception("stage=stt_native_import_failed error_type=%s", type(exc).__name__)
            raise RuntimeError(
                "native whisper.cpp STT module is not built. "
                "Run `python scripts/build_native_stt.py` before starting the agent."
            ) from exc
        logger.info("stage=stt_native_import_done module=native_build._local_stt")

        started = time.perf_counter()
        logger.info(
            "stage=stt_native_model_create_start model=%s sample_rate=%s language=%s",
            model_path,
            sample_rate,
            language,
        )
        self._recognizer = _local_stt.WhisperStream(
            str(model_path),
            sample_rate,
            language,
        )
        logger.info("stage=stt_native_model_create_done elapsed_ms=%.1f", (time.perf_counter() - started) * 1000)

    def warmup(self) -> None:
        """Run a tiny silent decode to bring up the CUDA context and warm paths."""
        started = time.perf_counter()
        try:
            self.reset()
            self._recognizer.accept_pcm_i16(b"\x00\x00" * 1600, self.sample_rate, 1)
            self._recognizer.flush()
        finally:
            self.reset()
        logger.info(
            "stage=stt_warmup_done elapsed_ms=%.1f",
            (time.perf_counter() - started) * 1000,
        )

    def process_frame(self, frame: rtc.AudioFrame) -> STTResult:
        started = time.perf_counter()
        result = self._recognizer.accept_pcm_i16(
            frame.data.tobytes(),
            frame.sample_rate,
            frame.num_channels,
        )
        logger.debug(
            "stage=stt_frame_accepted sample_rate=%s channels=%s samples_per_channel=%s duration_ms=%.1f elapsed_ms=%.1f",
            frame.sample_rate,
            frame.num_channels,
            frame.samples_per_channel,
            frame.duration * 1000,
            (time.perf_counter() - started) * 1000,
        )
        return STTResult(
            partial=str(result.get("partial") or ""),
            speech_started=bool(result.get("speech_started")),
            audio_duration=frame.duration,
        )

    def flush(self) -> STTResult:
        started = time.perf_counter()
        logger.info("stage=stt_flush_start")
        result = self._recognizer.flush()
        segments = [str(item) for item in result.get("segments", [])]
        text, filtered, filter_reason = _filter_transcript(str(result.get("final") or ""), result)
        logger.info(
            "stage=stt_flush_done elapsed_ms=%.1f final_chars=%s segment_count=%s confidence=%.3f no_speech_prob=%.3f audio_energy=%.5f filtered=%s filter_reason=%s",
            (time.perf_counter() - started) * 1000,
            len(text),
            len(segments),
            float(result.get("confidence") or 0.0),
            float(result.get("no_speech_prob") or 0.0),
            float(result.get("audio_energy") or 0.0),
            filtered,
            filter_reason,
        )
        return STTResult(
            final=text,
            speech_ended=bool(result.get("speech_ended", True)),
            confidence=float(result.get("confidence") or 0.0),
            segments=segments,
            no_speech_prob=float(result.get("no_speech_prob") or 0.0),
            audio_energy=float(result.get("audio_energy") or 0.0),
            filtered=filtered,
            filter_reason=filter_reason,
        )

    def partial(self, max_window_ms: float) -> STTResult:
        started = time.perf_counter()
        result = self._recognizer.partial(int(max_window_ms))
        segments = [str(item) for item in result.get("segments", [])]
        text, filtered, filter_reason = _filter_transcript(str(result.get("partial") or ""), result)
        logger.info(
            "stage=stt_partial_done elapsed_ms=%.1f partial_chars=%s segment_count=%s confidence=%.3f no_speech_prob=%.3f audio_energy=%.5f filtered=%s filter_reason=%s window_ms=%.1f",
            (time.perf_counter() - started) * 1000,
            len(text),
            len(segments),
            float(result.get("confidence") or 0.0),
            float(result.get("no_speech_prob") or 0.0),
            float(result.get("audio_energy") or 0.0),
            filtered,
            filter_reason,
            max_window_ms,
        )
        return STTResult(
            partial=text,
            speech_ended=bool(result.get("speech_ended", False)),
            confidence=float(result.get("confidence") or 0.0),
            segments=segments,
            no_speech_prob=float(result.get("no_speech_prob") or 0.0),
            audio_energy=float(result.get("audio_energy") or 0.0),
            filtered=filtered,
            filter_reason=filter_reason,
        )

    def reset(self) -> None:
        self._recognizer.reset()
        logger.debug("stage=stt_runtime_reset")

    def close(self) -> None:
        self._recognizer = None
        logger.info("stage=stt_runtime_closed")
