from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re
import time

logger = logging.getLogger("local_audio.tts.runtime")

@dataclass(frozen=True)
class TTSChunk:
    pcm: bytes
    is_final: bool = False


class LocalTTSRuntime:
    """
    Local in-process Piper runtime.

    Piper loads an ONNX voice once and streams raw 16-bit PCM chunks from
    PiperVoice.synthesize.
    """

    def __init__(
        self,
        model_path: Path,
        *,
        config_path: Path,
        sample_rate: int,
        num_channels: int,
        chunk_max_chars: int = 20,
    ) -> None:
        self.model_path = model_path
        self.config_path = config_path
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.chunk_max_chars = chunk_max_chars
        logger.info(
            "stage=tts_runtime_import_start model=%s config=%s sample_rate=%s channels=%s",
            model_path,
            config_path,
            sample_rate,
            num_channels,
        )
        try:
            from piper import PiperVoice
        except ImportError as exc:
            logger.exception("stage=tts_runtime_import_failed error_type=%s", type(exc).__name__)
            raise RuntimeError("piper-tts is not installed. Run `uv add piper-tts==1.5.0`.") from exc

        started = time.perf_counter()
        self._voice = PiperVoice.load(str(model_path), config_path=str(config_path))
        logger.info("stage=tts_runtime_model_loaded elapsed_ms=%.1f", (time.perf_counter() - started) * 1000)

    def synthesize_stream(self, text: str) -> list[TTSChunk]:
        chunks = list(self.iter_synthesize_stream(text))
        if chunks:
            chunks[-1] = TTSChunk(pcm=chunks[-1].pcm, is_final=True)
        return chunks

    def _normalize_text(self, text: str) -> str:
        return (
            text.replace("\u2011", "-")
            .replace("\u2013", "-")
            .replace("\u2014", "-")
            .replace("\u2018", "'")
            .replace("\u2019", "'")
            .replace("\u201c", '"')
            .replace("\u201d", '"')
            .strip()
        )

    def _split_for_latency(self, text: str) -> list[str]:
        normalized = self._normalize_text(text)
        pieces = [piece.strip() for piece in re.split(r"(?<=[।.!?])\s+", normalized) if piece.strip()]
        if not pieces:
            return []

        chunks: list[str] = []
        max_chars = self.chunk_max_chars
        for piece in pieces:
            if len(piece) <= max_chars:
                chunks.append(piece)
                continue

            words = piece.split()
            current = ""
            for word in words:
                candidate = f"{current} {word}".strip()
                if current and len(candidate) > max_chars:
                    chunks.append(current)
                    current = word
                else:
                    current = candidate
            if current:
                chunks.append(current)

        return chunks

    def iter_synthesize_stream(self, text: str):
        started = time.perf_counter()
        text_chunks = self._split_for_latency(text)
        logger.info("stage=tts_synthesize_stream_start text_chars=%s text_chunk_count=%s", len(text), len(text_chunks))
        chunk_count = 0
        pcm_bytes = 0
        first_chunk_logged = False
        for text_index, text_chunk in enumerate(text_chunks, start=1):
            text_started = time.perf_counter()
            logger.info(
                "stage=tts_text_chunk_start index=%s total=%s chars=%s",
                text_index,
                len(text_chunks),
                len(text_chunk),
            )
            for chunk in self._voice.synthesize(text_chunk):
                pcm = chunk.audio_int16_bytes
                if not pcm:
                    continue
                chunk_count += 1
                pcm_bytes += len(pcm)
                if not first_chunk_logged:
                    logger.info(
                        "stage=tts_synthesize_stream_first_chunk elapsed_ms=%.1f bytes=%s",
                        (time.perf_counter() - started) * 1000,
                        len(pcm),
                    )
                    first_chunk_logged = True
                yield TTSChunk(pcm=pcm, is_final=False)
            logger.info(
                "stage=tts_text_chunk_done index=%s total=%s elapsed_ms=%.1f",
                text_index,
                len(text_chunks),
                (time.perf_counter() - text_started) * 1000,
            )
        if not first_chunk_logged:
            logger.info("stage=tts_synthesize_stream_empty text_chars=%s", len(text))
        else:
            logger.info(
                "stage=tts_synthesize_stream_audio_ready elapsed_ms=%.1f chunk_count=%s pcm_bytes=%s",
                (time.perf_counter() - started) * 1000,
                chunk_count,
                pcm_bytes,
            )
        logger.info(
            "stage=tts_synthesize_stream_done elapsed_ms=%.1f chunk_count=%s pcm_bytes=%s",
            (time.perf_counter() - started) * 1000,
            chunk_count,
            pcm_bytes,
        )

    def reset(self) -> None:
        return None

    def close(self) -> None:
        self._voice = None
        logger.info("stage=tts_runtime_closed")
