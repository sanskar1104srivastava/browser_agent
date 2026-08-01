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

from .faster_whisper_runtime import FasterWhisperSTTRuntime
from .runtime import STTResult

logger = logging.getLogger("local_audio.stt.faster_whisper_provider")


class FasterWhisperSTT(stt.STT):
    """
    Batch STT wrapper for the CTranslate2/faster-whisper runtime.

    Same non-streaming contract as LocalSTT: LiveKit's stt.StreamAdapter
    wraps this with VAD so complete utterances are passed to recognize().
    """

    def __init__(
        self,
        runtime: FasterWhisperSTTRuntime,
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
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="faster-whisper-stt")
        logger.info(
            "stage=stt_provider_ready provider=faster_whisper mode=batch model=%s device=%s sample_rate=%s language=%s",
            self.model,
            runtime.device,
            sample_rate,
            language,
        )

    @property
    def model(self) -> str:
        return str(self._runtime.model_path)

    @property
    def provider(self) -> str:
        return "FasterWhisper"

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
            "stage=stt_batch_recognize_done elapsed_ms=%.1f audio_ms=%.1f final_chars=%s confidence=%.3f no_speech_prob=%.3f filtered=%s filter_reason=%s",
            elapsed_ms,
            frame.duration * 1000,
            len(text),
            result.confidence,
            result.no_speech_prob,
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
        logger.info("stage=stt_provider_close_start provider=faster_whisper")
        self._runtime.close()
        self._executor.shutdown(wait=False, cancel_futures=True)
        logger.info("stage=stt_provider_close_done provider=faster_whisper")

    async def _run_runtime(self, func, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, func, *args)
