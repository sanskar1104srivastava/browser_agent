"""Edge TTS provider — LiveKit-compatible wrapper around EdgeTTSRuntime."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import time

from livekit.agents import tts, utils
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

from .edge_runtime import EdgeTTSRuntime

logger = logging.getLogger("local_audio.tts.edge_provider")


class EdgeTTS(tts.TTS):
    """Edge TTS provider for LiveKit — uses Microsoft Edge neural voices."""

    def __init__(self, runtime: EdgeTTSRuntime):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=runtime.sample_rate,
            num_channels=runtime.num_channels,
        )
        self._runtime = runtime
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="edge-tts")
        logger.info(
            "stage=edge_tts_provider_ready voice=%s sample_rate=%s channels=%s",
            runtime.voice,
            runtime.sample_rate,
            runtime.num_channels,
        )

    @property
    def model(self) -> str:
        return f"edge-{self._runtime.voice}"

    @property
    def provider(self) -> str:
        return "EdgeNeural"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return EdgeChunkedStream(tts_plugin=self, input_text=text, conn_options=conn_options)

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.SynthesizeStream:
        stream = EdgeSynthesizeStream(tts_plugin=self, conn_options=conn_options)
        return stream

    async def aclose(self) -> None:
        self._runtime.close()
        self._executor.shutdown(wait=False, cancel_futures=True)
        logger.info("stage=edge_tts_provider_close_done")


class EdgeChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts_plugin: EdgeTTS, input_text: str, conn_options: APIConnectOptions) -> None:
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
        logger.info("stage=edge_tts_chunked_start text_chars=%s", len(self._input_text))
        started = time.perf_counter()

        try:
            chunks = self._tts_plugin._runtime.synthesize_stream(self._input_text)
            chunk_count = 0
            pcm_bytes = 0
            for chunk in chunks:
                output_emitter.push(chunk)
                chunk_count += 1
                pcm_bytes += len(chunk)
            output_emitter.flush()
            logger.info(
                "stage=edge_tts_chunked_done chunk_count=%s pcm_bytes=%s elapsed_ms=%.1f",
                chunk_count,
                pcm_bytes,
                (time.perf_counter() - started) * 1000,
            )
        except Exception:
            logger.exception("stage=edge_tts_chunked_failed")
            raise


class EdgeSynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts_plugin: EdgeTTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts_plugin, conn_options=conn_options)
        self._tts_plugin = tts_plugin
        self._request_id = utils.shortuuid()

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        logger.info("stage=edge_tts_stream_run_start request_id=%s", self._request_id)
        output_emitter.initialize(
            request_id=self._request_id,
            sample_rate=self._tts_plugin.sample_rate,
            num_channels=self._tts_plugin.num_channels,
            mime_type="audio/pcm",
            frame_size_ms=50,
            stream=True,
        )

        segment_id = ""
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
                    continue

                segment_id = utils.shortuuid()
                logger.info(
                    "stage=edge_tts_segment_start request_id=%s segment_id=%s text_chars=%s",
                    self._request_id,
                    segment_id,
                    len(text),
                )
                started = time.perf_counter()
                output_emitter.start_segment(segment_id=segment_id)

                try:
                    chunks = self._tts_plugin._runtime.synthesize_stream(text)
                    chunk_count = 0
                    pcm_bytes = 0
                    for chunk in chunks:
                        output_emitter.push(chunk)
                        chunk_count += 1
                        pcm_bytes += len(chunk)
                except Exception:
                    logger.exception(
                        "stage=edge_tts_segment_failed request_id=%s segment_id=%s",
                        self._request_id,
                        segment_id,
                    )
                    raise

                output_emitter.end_segment()
                logger.info(
                    "stage=edge_tts_segment_done request_id=%s segment_id=%s chunk_count=%s pcm_bytes=%s elapsed_ms=%.1f",
                    self._request_id,
                    segment_id,
                    chunk_count,
                    pcm_bytes,
                    (time.perf_counter() - started) * 1000,
                )

        logger.info("stage=edge_tts_stream_run_done request_id=%s", self._request_id)
