from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio.stt import VoskSTTRuntime
from audio.tts import LocalTTSRuntime
from config import LocalAudioConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEXT = "\u092e\u0941\u091d\u0947 \u0935\u093f\u0930\u093e\u091f \u0915\u094b\u0939\u0932\u0940 \u0915\u0947 \u092c\u093e\u0930\u0947 \u092e\u0947\u0902 \u092c\u0924\u093e\u0913"


def _resample_i16(pcm: bytes, source_rate: int, target_rate: int, channels: int) -> bytes:
    if channels <= 0:
        raise ValueError("channels must be positive")

    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return b""

    if channels > 1:
        samples = samples[: (samples.size // channels) * channels].reshape(-1, channels).mean(axis=1)

    if source_rate == target_rate:
        return np.clip(samples, -32768, 32767).astype(np.int16).tobytes()

    source_positions = np.arange(samples.size, dtype=np.float32)
    target_count = max(1, int(round(samples.size * target_rate / source_rate)))
    target_positions = np.linspace(0, samples.size - 1, target_count, dtype=np.float32)
    resampled = np.interp(target_positions, source_positions, samples)
    return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()


def _load_wav(path: Path, target_rate: int) -> tuple[bytes, str]:
    with wave.open(str(path), "rb") as wav:
        sample_width = wav.getsampwidth()
        if sample_width != 2:
            raise RuntimeError(f"expected 16-bit PCM WAV, got sample width {sample_width}")
        channels = wav.getnchannels()
        source_rate = wav.getframerate()
        pcm = wav.readframes(wav.getnframes())
    return _resample_i16(pcm, source_rate, target_rate, channels), f"wav:{path}"


def _generate_fixture(config: LocalAudioConfig, text: str) -> tuple[bytes, str]:
    tts = LocalTTSRuntime(
        config.tts_model_path,
        config_path=config.tts_config_path,
        sample_rate=config.tts_sample_rate,
        num_channels=config.num_channels,
    )
    try:
        pcm = b"".join(chunk.pcm for chunk in tts.synthesize_stream(text))
    finally:
        tts.close()
    return _resample_i16(pcm, config.tts_sample_rate, config.stt_sample_rate, config.num_channels), "local_piper_fixture"


def _iter_chunks(pcm: bytes, sample_rate: int, chunk_ms: float):
    samples_per_chunk = max(1, int(sample_rate * chunk_ms / 1000))
    bytes_per_chunk = samples_per_chunk * 2
    for offset in range(0, len(pcm), bytes_per_chunk):
        chunk = pcm[offset : offset + bytes_per_chunk]
        if chunk:
            yield chunk, (len(chunk) / 2) / sample_rate


def run_latency_trial(
    *,
    model_path: Path,
    pcm: bytes,
    source: str,
    expected_text: str,
    sample_rate: int,
    language: str,
    chunk_ms: float,
    realtime: bool,
) -> dict:
    load_started = time.perf_counter()
    runtime = VoskSTTRuntime(model_path, sample_rate=sample_rate, language=language)
    load_ms = round((time.perf_counter() - load_started) * 1000, 1)
    session = runtime.create_session()

    audio_ms = round((len(pcm) / 2) / sample_rate * 1000, 1)
    first_partial_ms: float | None = None
    partials: list[dict] = []
    processing_ms = 0.0
    started = time.perf_counter()
    frame_count = 0

    try:
        for chunk, duration in _iter_chunks(pcm, sample_rate, chunk_ms):
            frame_count += 1
            frame_started = time.perf_counter()
            session.accept_pcm_i16(chunk, sample_rate=sample_rate, channels=1, duration=duration)
            partial = session.partial()
            processing_ms += (time.perf_counter() - frame_started) * 1000
            if partial.partial:
                elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
                if first_partial_ms is None:
                    first_partial_ms = elapsed_ms
                partials.append({"at_ms": elapsed_ms, "text": partial.partial})
            if realtime:
                time.sleep(duration)

        final_started = time.perf_counter()
        final = session.flush()
        finalization_ms = round((time.perf_counter() - final_started) * 1000, 1)
        processing_ms += finalization_ms
    finally:
        runtime.close()

    wall_ms = round((time.perf_counter() - started) * 1000, 1)
    return {
        "runtime": "vosk",
        "model": model_path.name,
        "language": language,
        "source": source,
        "expected_text": expected_text,
        "final_text": final.final,
        "sample_rate": sample_rate,
        "chunk_ms": chunk_ms,
        "frame_count": frame_count,
        "audio_ms": audio_ms,
        "model_load_ms": load_ms,
        "first_partial_ms": first_partial_ms,
        "finalization_ms": finalization_ms,
        "processing_ms": round(processing_ms, 1),
        "processing_realtime_factor": round(processing_ms / audio_ms, 3) if audio_ms else None,
        "wall_ms": wall_ms,
        "wall_realtime_factor": round(wall_ms / audio_ms, 3) if audio_ms else None,
        "partial_count": len(partials),
        "partials": partials[:12],
        "realtime_feed": realtime,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=Path, help="Optional 16-bit PCM WAV file. It will be downmixed/resampled to 16 kHz.")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Hindi text used for generated fixture audio when --wav is omitted.")
    parser.add_argument("--chunk-ms", type=float, default=50.0)
    parser.add_argument("--realtime", action="store_true", help="Sleep between chunks to simulate live microphone timing.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmarks" / "stt_trials")
    args = parser.parse_args()

    config = LocalAudioConfig.from_env()
    config.validate()
    if args.wav:
        pcm, source = _load_wav(args.wav.resolve(), config.stt_sample_rate)
        expected = ""
    else:
        pcm, source = _generate_fixture(config, args.text)
        expected = args.text

    result = run_latency_trial(
        model_path=config.vosk_model_path,
        pcm=pcm,
        source=source,
        expected_text=expected,
        sample_rate=config.stt_sample_rate,
        language=config.stt_language,
        chunk_ms=args.chunk_ms,
        realtime=args.realtime,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{timestamp}_vosk_stt_service_latency.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
