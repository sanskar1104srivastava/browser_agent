from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import logging
import math
import time

from livekit import rtc
from livekit.agents import stt, utils
from livekit.agents.types import (
    APIConnectOptions,
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)

from .runtime import STTResult
from .vosk_runtime import VoskSTTRuntime, VoskSTTSession

logger = logging.getLogger("local_audio.stt.vosk_provider")


def _pcm_i16_rms(pcm: bytes) -> int:
    if len(pcm) < 2:
        return 0
    sample_count = len(pcm) // 2
    samples = memoryview(pcm[: sample_count * 2]).cast("h")
    if not samples:
        return 0
    square_sum = sum(int(sample) * int(sample) for sample in samples)
    return int(math.sqrt(square_sum / len(samples)))


class VoskSTT(stt.STT):
    def __init__(
        self,
        runtime: VoskSTTRuntime,
        *,
        sample_rate: int = 16000,
        language: str = "hi",
        speech_rms_threshold: int = 350,
        endpoint_silence_ms: float = 300.0,
        min_utterance_ms: float = 200.0,
        partial_interval_ms: float = 150.0,
        min_partial_audio_ms: float = 200.0,
        preroll_ms: float = 300.0,
        use_rms_gate: bool = False,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=partial_interval_ms > 0,
                diarization=False,
                aligned_transcript=False,
            )
        )
        self._runtime = runtime
        self._sample_rate = sample_rate
        self._language = language
        self._speech_rms_threshold = speech_rms_threshold
        self._endpoint_silence_ms = endpoint_silence_ms
        self._min_utterance_ms = min_utterance_ms
        self._partial_interval_ms = partial_interval_ms
        self._min_partial_audio_ms = min_partial_audio_ms
        self._preroll_ms = preroll_ms
        self._use_rms_gate = use_rms_gate
        self._streams: set[VoskRecognizeStream] = set()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vosk-stt")
        logger.info(
            "stage=vosk_provider_ready model=%s sample_rate=%s language=%s speech_rms_threshold=%s endpoint_silence_ms=%.1f min_utterance_ms=%.1f partial_interval_ms=%.1f min_partial_audio_ms=%.1f preroll_ms=%.1f use_rms_gate=%s",
            self.model,
            sample_rate,
            language,
            speech_rms_threshold,
            endpoint_silence_ms,
            min_utterance_ms,
            partial_interval_ms,
            min_partial_audio_ms,
            preroll_ms,
            use_rms_gate,
        )

    @property
    def model(self) -> str:
        return str(self._runtime.model_path.name)

    @property
    def provider(self) -> str:
        return "VoskLocal"

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.RecognizeStream:
        stream = VoskRecognizeStream(
            stt_plugin=self,
            session=self._runtime.create_session(),
            language=language if language is not NOT_GIVEN else self._language,
            conn_options=conn_options,
            sample_rate=self._sample_rate,
            speech_rms_threshold=self._speech_rms_threshold,
            endpoint_silence_ms=self._endpoint_silence_ms,
            min_utterance_ms=self._min_utterance_ms,
            partial_interval_ms=self._partial_interval_ms,
            min_partial_audio_ms=self._min_partial_audio_ms,
            preroll_ms=self._preroll_ms,
            use_rms_gate=self._use_rms_gate,
        )
        self._streams.add(stream)
        logger.info("stage=vosk_stream_created request_id=%s language=%s sample_rate=%s", stream.request_id, stream.language, self._sample_rate)
        return stream

    async def _recognize_impl(
        self,
        buffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        frame = rtc.combine_audio_frames(buffer if isinstance(buffer, list) else [buffer])
        logger.info(
            "stage=vosk_batch_recognize_start sample_rate=%s channels=%s samples_per_channel=%s",
            frame.sample_rate,
            frame.num_channels,
            frame.samples_per_channel,
        )
        session = self._runtime.create_session()
        await self._run_runtime(session.process_frame, frame)
        flush_result = await self._run_runtime(session.flush)
        logger.info("stage=vosk_batch_recognize_done final_chars=%s", len(flush_result.final))
        return self._speech_event(
            stt.SpeechEventType.FINAL_TRANSCRIPT,
            flush_result.final,
            language if language is not NOT_GIVEN else self._language,
        )

    def _speech_event(self, event_type: stt.SpeechEventType, text: str, language: str) -> stt.SpeechEvent:
        return stt.SpeechEvent(
            type=event_type,
            request_id=utils.shortuuid(),
            alternatives=[
                stt.SpeechData(
                    language=language,
                    text=text.strip(),
                    confidence=1.0 if text.strip() else 0.0,
                )
            ],
        )

    async def aclose(self) -> None:
        logger.info("stage=vosk_provider_close_start active_streams=%s", len(self._streams))
        await asyncio.gather(*(stream.aclose() for stream in list(self._streams)), return_exceptions=True)
        self._runtime.close()
        self._executor.shutdown(wait=False, cancel_futures=True)
        logger.info("stage=vosk_provider_close_done")

    async def _run_runtime(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, partial(func, *args, **kwargs))


class VoskRecognizeStream(stt.RecognizeStream):
    def __init__(
        self,
        *,
        stt_plugin: VoskSTT,
        session: VoskSTTSession,
        language: str,
        conn_options: APIConnectOptions,
        sample_rate: int,
        speech_rms_threshold: int,
        endpoint_silence_ms: float,
        min_utterance_ms: float,
        partial_interval_ms: float,
        min_partial_audio_ms: float,
        preroll_ms: float,
        use_rms_gate: bool,
    ) -> None:
        super().__init__(stt=stt_plugin, conn_options=conn_options, sample_rate=sample_rate)
        self._stt_plugin = stt_plugin
        self._session = session
        self._language = language
        self._request_id = utils.shortuuid()
        self._speaking = False
        self._audio_duration = 0.0
        self._utterance_audio_duration = 0.0
        self._silence_duration = 0.0
        self._in_speech = False
        self._frame_count = 0
        self._first_frame_logged = False
        self._speech_rms_threshold = speech_rms_threshold
        self._endpoint_silence_ms = endpoint_silence_ms
        self._min_utterance_ms = min_utterance_ms
        self._partial_interval_ms = partial_interval_ms
        self._min_partial_audio_ms = min_partial_audio_ms
        self._last_partial_at = 0.0
        self._last_partial_text = ""
        self._preroll_ms = max(0.0, preroll_ms)
        self._use_rms_gate = use_rms_gate
        self._preroll_frames: deque[tuple[bytes, int, int, float]] = deque()
        self._preroll_duration_ms = 0.0

    @property
    def request_id(self) -> str:
        return self._request_id

    @property
    def language(self) -> str:
        return self._language

    async def _run(self) -> None:
        logger.info("stage=vosk_stream_run_start request_id=%s", self._request_id)
        async for item in self._input_ch:
            if isinstance(item, rtc.AudioFrame):
                await self._handle_frame(item)
            elif isinstance(item, self._FlushSentinel):
                logger.info(
                    "stage=vosk_flush_requested request_id=%s frame_count=%s audio_ms=%.1f",
                    self._request_id,
                    self._frame_count,
                    self._audio_duration * 1000,
                )
                await self._finalize_utterance(force_final=True)
        logger.info("stage=vosk_stream_run_done request_id=%s", self._request_id)

    async def _handle_frame(self, frame: rtc.AudioFrame) -> None:
        if not self._first_frame_logged:
            logger.info(
                "stage=vosk_first_frame request_id=%s sample_rate=%s channels=%s samples_per_channel=%s",
                self._request_id,
                frame.sample_rate,
                frame.num_channels,
                frame.samples_per_channel,
            )
            self._first_frame_logged = True

        self._frame_count += 1
        self._audio_duration += frame.duration
        rms = _pcm_i16_rms(frame.data.tobytes())
        is_speech = rms >= self._speech_rms_threshold
        started = time.perf_counter()
        result = STTResult(audio_duration=frame.duration)

        if is_speech:
            self._silence_duration = 0.0
            self._in_speech = True
            self._utterance_audio_duration += frame.duration
            try:
                result = await self._stt_plugin._run_runtime(self._session.process_frame, frame)
            except Exception:
                logger.exception("stage=vosk_frame_failed request_id=%s frame_count=%s", self._request_id, self._frame_count)
                raise
            await self._maybe_emit_partial()
        elif self._in_speech:
            self._silence_duration += frame.duration
            try:
                result = await self._stt_plugin._run_runtime(self._session.process_frame, frame)
            except Exception:
                logger.exception("stage=vosk_frame_failed request_id=%s frame_count=%s", self._request_id, self._frame_count)
                raise

        if self._frame_count % 50 == 0:
            logger.info(
                "stage=vosk_frames_observed request_id=%s frame_count=%s audio_ms=%.1f utterance_ms=%.1f silence_ms=%.1f rms=%s speech=%s last_frame_elapsed_ms=%.1f",
                self._request_id,
                self._frame_count,
                self._audio_duration * 1000,
                self._utterance_audio_duration * 1000,
                self._silence_duration * 1000,
                rms,
                is_speech,
                (time.perf_counter() - started) * 1000,
            )
        self._emit_runtime_result(result)

        if (
            self._in_speech
            and self._utterance_audio_duration * 1000 >= self._min_utterance_ms
            and self._silence_duration * 1000 >= self._endpoint_silence_ms
        ):
            logger.info(
                "stage=vosk_endpoint_detected request_id=%s utterance_ms=%.1f silence_ms=%.1f frame_count=%s",
                self._request_id,
                self._utterance_audio_duration * 1000,
                self._silence_duration * 1000,
                self._frame_count,
            )
            await self._finalize_utterance(force_final=True)

    async def _finalize_utterance(self, *, force_final: bool) -> None:
        if self._utterance_audio_duration <= 0:
            logger.info("stage=vosk_finalize_skipped_empty request_id=%s", self._request_id)
            return

        try:
            result = await self._stt_plugin._run_runtime(self._session.flush)
        except Exception:
            logger.exception("stage=vosk_flush_failed request_id=%s", self._request_id)
            raise
        self._emit_runtime_result(result, force_final=force_final)
        self._session = self._stt_plugin._runtime.create_session()
        self._emit_usage()
        self._utterance_audio_duration = 0.0
        self._silence_duration = 0.0
        self._in_speech = False
        self._last_partial_at = 0.0
        self._last_partial_text = ""

    async def _maybe_emit_partial(self) -> None:
        if self._partial_interval_ms <= 0:
            return
        utterance_ms = self._utterance_audio_duration * 1000
        if utterance_ms < self._min_partial_audio_ms:
            return
        now = time.perf_counter()
        if self._last_partial_at and (now - self._last_partial_at) * 1000 < self._partial_interval_ms:
            return
        self._last_partial_at = now
        result = await self._stt_plugin._run_runtime(self._session.partial)
        text = result.partial.strip()
        if not text or text == self._last_partial_text:
            return
        self._last_partial_text = text
        self._emit_runtime_result(result)

    def _emit_runtime_result(self, result: STTResult, *, force_final: bool = False) -> None:
        if result.speech_started and not self._speaking:
            self._speaking = True
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))
            logger.info("stage=vosk_event_start_of_speech request_id=%s", self._request_id)

        if result.partial:
            self._event_ch.send_nowait(
                self._stt_plugin._speech_event(
                    stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    result.partial,
                    self._language,
                )
            )
            logger.info("stage=vosk_event_interim request_id=%s chars=%s", self._request_id, len(result.partial))

        if result.final or force_final:
            text = result.final or result.partial
            if text.strip():
                self._event_ch.send_nowait(
                    self._stt_plugin._speech_event(
                        stt.SpeechEventType.FINAL_TRANSCRIPT,
                        text,
                        self._language,
                    )
                )
                logger.info("stage=vosk_event_final request_id=%s chars=%s", self._request_id, len(text))
            else:
                logger.info("stage=vosk_event_final_empty request_id=%s forced=%s", self._request_id, force_final)

        if result.speech_ended or force_final:
            if self._speaking:
                self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))
                logger.info("stage=vosk_event_end_of_speech request_id=%s", self._request_id)
            self._speaking = False

    def _emit_usage(self) -> None:
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.RECOGNITION_USAGE,
                request_id=self._request_id,
                recognition_usage=stt.RecognitionUsage(audio_duration=self._audio_duration),
            )
        )
        logger.info(
            "stage=vosk_usage_emitted request_id=%s audio_ms=%.1f frame_count=%s",
            self._request_id,
            self._audio_duration * 1000,
            self._frame_count,
        )
        self._audio_duration = 0.0
        self._frame_count = 0
