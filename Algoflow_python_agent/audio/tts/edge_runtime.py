"""Edge TTS runtime — uses Microsoft Edge neural voices for high-quality Hindi TTS."""
from __future__ import annotations

import asyncio
import io
import logging
import struct
import time
import wave

import numpy as np

logger = logging.getLogger("local_audio.tts.edge_runtime")

DEFAULT_VOICE = "hi-IN-MadhurNeural"


class EdgeTTSRuntime:
    """TTS runtime using Microsoft Edge neural voices via edge-tts."""

    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        sample_rate: int = 24000,
    ) -> None:
        self.voice = voice
        self.sample_rate = sample_rate
        self.num_channels = 1
        self._loop: asyncio.AbstractEventLoop | None = None
        logger.info("stage=edge_tts_init voice=%s sample_rate=%s", voice, sample_rate)

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def synthesize_stream(self, text: str) -> list[bytes]:
        """Synthesize text via Edge TTS and return list of PCM int16 chunks."""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        started = time.perf_counter()
        logger.info("stage=edge_tts_synthesize_start text_chars=%s voice=%s", len(text), self.voice)

        def _run_async():
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self._synthesize_async(text))
            finally:
                loop.close()

        with ThreadPoolExecutor(max_workers=1) as executor:
            pcm_data = executor.submit(_run_async).result()

        elapsed_ms = (time.perf_counter() - started) * 1000
        audio_ms = (len(pcm_data) / 2) / self.sample_rate * 1000
        logger.info(
            "stage=edge_tts_synthesize_done elapsed_ms=%.1f audio_ms=%.1f pcm_bytes=%s",
            elapsed_ms,
            audio_ms,
            len(pcm_data),
        )

        chunk_size = self.sample_rate * 2  # 1 second per chunk
        chunks = []
        for i in range(0, len(pcm_data), chunk_size):
            chunk = pcm_data[i : i + chunk_size]
            if chunk:
                chunks.append(chunk)
        return chunks

    async def _synthesize_async(self, text: str) -> bytes:
        """Async synthesis using edge-tts."""
        import edge_tts

        communicate = edge_tts.Communicate(text, self.voice)
        audio_bytes = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes += chunk["data"]

        # Convert MP3 to PCM int16
        return self._mp3_to_pcm(audio_bytes)

    @staticmethod
    def _mp3_to_pcm(mp3_bytes: bytes) -> bytes:
        """Convert MP3 bytes to PCM int16 using ffmpeg."""
        import subprocess
        import tempfile
        import os

        # Try to find ffmpeg
        ffmpeg_path = None
        for candidate in [
            "ffmpeg",
            r"C:\Users\Sanskar\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin\ffmpeg.exe",
        ]:
            try:
                subprocess.run([candidate, "-version"], capture_output=True, timeout=5)
                ffmpeg_path = candidate
                break
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

        if ffmpeg_path is None:
            raise RuntimeError("ffmpeg not found. Install ffmpeg or add it to PATH.")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(mp3_bytes)
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                [
                    ffmpeg_path, "-i", tmp_path,
                    "-f", "s16le", "-acodec", "pcm_s16le",
                    "-ar", "24000", "-ac", "1",
                    "pipe:1"
                ],
                capture_output=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")
            return result.stdout
        finally:
            os.unlink(tmp_path)

    def close(self) -> None:
        if self._loop and not self._loop.is_closed():
            self._loop.close()
        logger.info("stage=edge_tts_closed")
