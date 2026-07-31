from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import time

from livekit import rtc
from livekit.agents import stt, utils
from livekit.agents.types import (
    APIConnectOptions,
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)

from .runtime import LocalSTTRuntime, STTResult

logger = logging.getLogger("local_audio.stt.provider")


class LocalSTT(stt.STT):
    """
    Batch STT wrapper for the repo-owned whisper.cpp runtime.

    Whisper.cpp is not truly streaming in this integration. LiveKit's
    stt.StreamAdapter should wrap this class with VAD so complete utterances
    are passed to recognize().
    """

    def __init__(
        self,
        runtime: LocalSTTRuntime,
        *,
        sample_rate: int = 16000,
        language: str = "hi",
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
                diarization=False,
                aligned_transcript=False,
            )
        )
        self._runtime = runtime
        self._sample_rate = sample_rate
        self._language = language
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="local-whisper-stt")
        logger.info(
            "stage=stt_provider_ready provider=local_whisper mode=batch model=%s sample_rate=%s language=%s",
            self.model,
            sample_rate,
            language,
        )

    @property
    def model(self) -> str:
        return str(self._runtime.model_path.name)

    @property
    def provider(self) -> str:
        return "LocalWhisperCpp"

    async def _recognize_impl(
        self,
        buffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        frame = self._prepare_frame(buffer)
        started = time.perf_counter()
        logger.info(
            "stage=stt_batch_recognize_start model=%s language=%s sample_rate=%s channels=%s samples_per_channel=%s duration_ms=%.1f",
            self.model,
            language if language is not NOT_GIVEN else self._language,
            frame.sample_rate,
            frame.num_channels,
            frame.samples_per_channel,
            frame.duration * 1000,
        )

        result = await self._run_runtime(self._decode_frame, frame)
        text = result.final.strip()
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "stage=stt_batch_recognize_done elapsed_ms=%.1f audio_ms=%.1f final_chars=%s confidence=%.3f no_speech_prob=%.3f audio_energy=%.5f filtered=%s filter_reason=%s",
            elapsed_ms,
            frame.duration * 1000,
            len(text),
            result.confidence,
            result.no_speech_prob,
            result.audio_energy,
            result.filtered,
            result.filter_reason,
        )
        return self._speech_event(
            stt.SpeechEventType.FINAL_TRANSCRIPT,
            text,
            language if language is not NOT_GIVEN else self._language,
            result,
        )

    def _prepare_frame(self, buffer) -> rtc.AudioFrame:
        frame = rtc.combine_audio_frames(buffer)
        if frame.sample_rate == self._sample_rate:
            return frame

        started = time.perf_counter()
        resampler = rtc.AudioResampler(
            input_rate=frame.sample_rate,
            output_rate=self._sample_rate,
            num_channels=frame.num_channels,
            quality=rtc.AudioResamplerQuality.HIGH,
        )
        frames = resampler.push(frame)
        frames.extend(resampler.flush())
        if not frames:
            raise RuntimeError("STT resampler produced no audio frames")

        resampled = rtc.combine_audio_frames(frames)
        logger.info(
            "stage=stt_audio_resampled input_sample_rate=%s output_sample_rate=%s input_duration_ms=%.1f output_duration_ms=%.1f channels=%s",
            frame.sample_rate,
            resampled.sample_rate,
            frame.duration * 1000,
            resampled.duration * 1000,
            resampled.num_channels,
        )
        logger.debug("stage=stt_audio_resample_done elapsed_ms=%.1f", (time.perf_counter() - started) * 1000)
        return resampled

    def _decode_frame(self, frame: rtc.AudioFrame) -> STTResult:
        self._runtime.reset()
        try:
            self._runtime.process_frame(frame)
            return self._runtime.flush()
        finally:
            self._runtime.reset()

    def _speech_event(
        self,
        event_type: stt.SpeechEventType,
        text: str,
        language: str,
        result: STTResult,
    ) -> stt.SpeechEvent:
        logger.info(
            "stage=stt_speech_event type=%s chars=%s confidence=%.3f no_speech_prob=%.3f audio_energy=%.5f filtered=%s filter_reason=%s segment_count=%s",
            event_type,
            len(text),
            result.confidence if text else 0.0,
            result.no_speech_prob,
            result.audio_energy,
            result.filtered,
            result.filter_reason,
            len(result.segments),
        )
        return stt.SpeechEvent(
            type=event_type,
            request_id=utils.shortuuid(),
            alternatives=[
                stt.SpeechData(
                    language=language,
                    text=text,
                    confidence=result.confidence if text else 0.0,
                )
            ],
        )

    async def aclose(self) -> None:
        logger.info("stage=stt_provider_close_start provider=local_whisper")
        self._runtime.close()
        self._executor.shutdown(wait=False, cancel_futures=True)
        logger.info("stage=stt_provider_close_done provider=local_whisper")

    async def _run_runtime(self, func, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, func, *args)
