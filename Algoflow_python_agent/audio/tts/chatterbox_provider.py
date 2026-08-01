from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import time

from livekit.agents import tts, utils
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

from .chatterbox_runtime import ChatterboxTTSRuntime

logger = logging.getLogger("local_audio.tts.chatterbox_provider")


class ChatterboxTTS(tts.TTS):
    """
    Chatterbox Multilingual TTS provider for LiveKit.

    Wraps ChatterboxMultilingualTTS (chatterbox-tts) into a LiveKit-compatible
    streaming TTS provider. Zero-shot voice cloning, 23 languages incl. Hindi.
    """

    def __init__(self, runtime: ChatterboxTTSRuntime):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=runtime.sample_rate,
            num_channels=runtime.num_channels,
        )
        self._runtime = runtime
        self._streams: set[ChatterboxSynthesizeStream] = set()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chatterbox-tts")
        logger.info(
            "stage=chatterbox_tts_provider_ready language=%s device=%s sample_rate=%s channels=%s",
            runtime.language,
            runtime.device,
            runtime.sample_rate,
            runtime.num_channels,
        )

    @property
    def model(self) -> str:
        return f"chatterbox-multilingual-{self._runtime.language}"

    @property
    def provider(self) -> str:
        return "ChatterboxMultilingual"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return ChatterboxChunkedStream(tts_plugin=self, input_text=text, conn_options=conn_options)

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.SynthesizeStream:
        stream = ChatterboxSynthesizeStream(tts_plugin=self, conn_options=conn_options)
        self._streams.add(stream)
        logger.info("stage=chatterbox_tts_stream_created")
        return stream

    async def aclose(self) -> None:
        logger.info("stage=chatterbox_tts_provider_close_start active_streams=%s", len(self._streams))
        await asyncio.gather(*(stream.aclose() for stream in list(self._streams)), return_exceptions=True)
        self._runtime.close()
        self._executor.shutdown(wait=False, cancel_futures=True)
        logger.info("stage=chatterbox_tts_provider_close_done")

    async def _run_runtime(self, func, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, func, *args)

    async def _push_streaming_audio(
        self,
        *,
        text: str,
        output_emitter: tts.AudioEmitter,
        request_id: str,
        segment_id: str,
        log_stage: str,
    ) -> tuple[int, int]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[object] = asyncio.Queue()
        done = object()

        def worker() -> None:
            try:
                chunks = self._runtime.synthesize_stream(text)
                for chunk in chunks:
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, done)

        self._executor.submit(worker)
        started = time.perf_counter()
        first_chunk_logged = False
        chunk_count = 0
        pcm_bytes = 0

        while True:
            item = await queue.get()
            if item is done:
                break
            if isinstance(item, Exception):
                logger.error(
                    "stage=%s_failed request_id=%s segment_id=%s error_type=%s",
                    log_stage,
                    request_id,
                    segment_id,
                    type(item).__name__,
                    exc_info=(type(item), item, item.__traceback__),
                )
                raise item

            pcm = item
            chunk_count += 1
            pcm_bytes += len(pcm)
            if not first_chunk_logged:
                logger.info(
                    "stage=%s_first_chunk request_id=%s segment_id=%s elapsed_ms=%.1f bytes=%s",
                    log_stage,
                    request_id,
                    segment_id,
                    (time.perf_counter() - started) * 1000,
                    len(pcm),
                )
                first_chunk_logged = True
            output_emitter.push(pcm)

        return chunk_count, pcm_bytes


class ChatterboxChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts_plugin: ChatterboxTTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts_plugin, input_text=input_text, conn_options=conn_options)
        self._tts_plugin = tts_plugin

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._tts_plugin.sample_rate,
            num_channels=self._tts_plugin.num_channels,
            mime_type="audio/pcm",
            frame_size_ms=50,
        )
        logger.info("stage=chatterbox_tts_chunked_start text_chars=%s", len(self._input_text))
        started = time.perf_counter()
        chunk_count, pcm_bytes = await self._tts_plugin._push_streaming_audio(
            text=self._input_text,
            output_emitter=output_emitter,
            request_id="chunked",
            segment_id="",
            log_stage="chatterbox_tts_chunked",
        )
        output_emitter.flush()
        logger.info(
            "stage=chatterbox_tts_chunked_done chunk_count=%s pcm_bytes=%s elapsed_ms=%.1f",
            chunk_count,
            pcm_bytes,
            (time.perf_counter() - started) * 1000,
        )


class ChatterboxSynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts_plugin: ChatterboxTTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts_plugin, conn_options=conn_options)
        self._tts_plugin = tts_plugin
        self._request_id = utils.shortuuid()

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        logger.info("stage=chatterbox_tts_stream_run_start request_id=%s", self._request_id)
        output_emitter.initialize(
            request_id=self._request_id,
            sample_rate=self._tts_plugin.sample_rate,
            num_channels=self._tts_plugin.num_channels,
            mime_type="audio/pcm",
            frame_size_ms=50,
            stream=True,
        )

        text_parts: list[str] = []
        async for item in self._input_ch:
            if isinstance(item, str):
                self._mark_started()
                text_parts.append(item)
                continue

            if isinstance(item, self._FlushSentinel):
                text = "".join(text_parts).strip()
                text_parts = []
                if not text:
                    logger.info("stage=chatterbox_tts_flush_empty request_id=%s", self._request_id)
                    continue

                segment_id = utils.shortuuid()
                logger.info(
                    "stage=chatterbox_tts_segment_start request_id=%s segment_id=%s text_chars=%s",
                    self._request_id,
                    segment_id,
                    len(text),
                )
                started = time.perf_counter()
                output_emitter.start_segment(segment_id=segment_id)
                chunk_count, pcm_bytes = await self._tts_plugin._push_streaming_audio(
                    text=text,
                    output_emitter=output_emitter,
                    request_id=self._request_id,
                    segment_id=segment_id,
                    log_stage="chatterbox_tts",
                )
                output_emitter.end_segment()
                logger.info(
                    "stage=chatterbox_tts_segment_done request_id=%s segment_id=%s chunk_count=%s pcm_bytes=%s elapsed_ms=%.1f",
                    self._request_id,
                    segment_id,
                    chunk_count,
                    pcm_bytes,
                    (time.perf_counter() - started) * 1000,
                )

        logger.info("stage=chatterbox_tts_stream_run_done request_id=%s", self._request_id)
