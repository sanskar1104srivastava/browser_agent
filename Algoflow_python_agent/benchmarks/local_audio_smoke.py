from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import LocalAudioConfig
from audio.stt import LocalSTTRuntime
from audio.tts import LocalTTSRuntime
from livekit import rtc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


def _silent_frame(sample_rate: int, seconds: float, channels: int) -> rtc.AudioFrame:
    samples = int(sample_rate * seconds)
    return rtc.AudioFrame(
        data=b"\0\0" * samples * channels,
        sample_rate=sample_rate,
        num_channels=channels,
        samples_per_channel=samples,
    )


def run_smoke(tts_text: str) -> dict:
    config = LocalAudioConfig.from_env()
    config.validate()

    started = time.perf_counter()
    stt_runtime = LocalSTTRuntime(config.stt_model_path, sample_rate=config.stt_sample_rate)
    tts_runtime = LocalTTSRuntime(
        config.tts_model_path,
        config_path=config.tts_config_path,
        sample_rate=config.tts_sample_rate,
        num_channels=config.num_channels,
    )
    model_load_time = time.perf_counter() - started

    frame = _silent_frame(config.stt_sample_rate, 0.2, config.num_channels)

    stt_started = time.perf_counter()
    stt_result = stt_runtime.process_frame(frame)
    stt_flush = stt_runtime.flush()
    stt_latency = time.perf_counter() - stt_started

    tts_started = time.perf_counter()
    chunks = tts_runtime.synthesize_stream(tts_text)
    tts_latency = time.perf_counter() - tts_started

    stt_runtime.close()
    tts_runtime.close()

    return {
        "model_load_time_sec": round(model_load_time, 4),
        "stt_latency_sec": round(stt_latency, 4),
        "tts_first_audio_latency_sec": round(tts_latency, 4),
        "tts_chunks": len(chunks),
        "tts_pcm_bytes": sum(len(chunk.pcm) for chunk in chunks),
        "stt_partial": stt_result.partial,
        "stt_final": stt_flush.final or stt_result.final,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="Hello from the local in-process voice runtime.")
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.text), indent=2))


if __name__ == "__main__":
    main()
