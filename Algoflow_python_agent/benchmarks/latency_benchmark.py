from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import LocalAudioConfig
from audio.stt import VoskSTTRuntime
from audio.tts import LocalTTSRuntime
from livekit import rtc


def make_silent_frame(sample_rate: int, duration_sec: float, channels: int) -> rtc.AudioFrame:
    samples = int(sample_rate * duration_sec)
    return rtc.AudioFrame(
        data=b"\0\0" * samples * channels,
        sample_rate=sample_rate,
        num_channels=channels,
        samples_per_channel=samples,
    )


def run_benchmark() -> dict:
    config = LocalAudioConfig.from_env()
    config.validate()

    results = {}

    # === Model Load ===
    load_start = time.perf_counter()
    vosk_runtime = VoskSTTRuntime(
        config.vosk_model_path,
        sample_rate=config.stt_sample_rate,
        language=config.stt_language,
    )
    vosk_load_ms = (time.perf_counter() - load_start) * 1000

    load_start = time.perf_counter()
    tts_runtime = LocalTTSRuntime(
        config.tts_model_path,
        config_path=config.tts_config_path,
        sample_rate=config.tts_sample_rate,
        num_channels=config.num_channels,
        chunk_max_chars=config.tts_chunk_max_chars,
    )
    tts_load_ms = (time.perf_counter() - load_start) * 1000

    results["model_load"] = {
        "vosk_ms": round(vosk_load_ms, 1),
        "tts_ms": round(tts_load_ms, 1),
        "total_ms": round(vosk_load_ms + tts_load_ms, 1),
    }

    # === STT Decode (batch mode simulating flush) ===
    test_phrases = [
        ("तुम कौन हो", 1.0),
        ("मेरा नाम संस्कार है", 1.5),
        ("मुझे विराट कोहली के बारे में बताओ", 2.5),
    ]

    stt_results = []
    for phrase, audio_sec in test_phrases:
        session = vosk_runtime.create_session()
        frame = make_silent_frame(config.stt_sample_rate, audio_sec, config.num_channels)
        session.process_frame(frame)

        decode_start = time.perf_counter()
        result = session.flush()
        decode_ms = (time.perf_counter() - decode_start) * 1000

        stt_results.append({
            "phrase": phrase,
            "audio_ms": round(audio_sec * 1000, 1),
            "decode_ms": round(decode_ms, 1),
            "rtf": round(decode_ms / (audio_sec * 1000), 3),
            "text": result.final,
        })

    results["stt_decode"] = {
        "avg_decode_ms": round(sum(r["decode_ms"] for r in stt_results) / len(stt_results), 1),
        "avg_rtf": round(sum(r["rtf"] for r in stt_results) / len(stt_results), 3),
        "cases": stt_results,
    }

    # === TTS First Audio ===
    tts_phrases = [
        "नमस्ते",
        "आप कैसे हैं",
        "मैं आपकी कैसे मदद कर सकता हूं",
    ]

    tts_results = []
    for phrase in tts_phrases:
        tts_start = time.perf_counter()
        chunks = tts_runtime.synthesize_stream(phrase)
        first_audio_ms = (time.perf_counter() - tts_start) * 1000

        tts_results.append({
            "phrase": phrase,
            "chars": len(phrase),
            "first_audio_ms": round(first_audio_ms, 1),
            "chunks": len(chunks),
            "pcm_bytes": sum(len(c.pcm) for c in chunks),
        })

    results["tts_first_audio"] = {
        "avg_first_audio_ms": round(sum(r["first_audio_ms"] for r in tts_results) / len(tts_results), 1),
        "cases": tts_results,
    }

    # === Estimated Round-Trip ===
    avg_stt_decode = results["stt_decode"]["avg_decode_ms"]
    avg_tts_first = results["tts_first_audio"]["avg_first_audio_ms"]
    endpointing_min = 250  # min_delay in agent.py

    results["estimated_round_trip"] = {
        "stt_decode_ms": round(avg_stt_decode, 1),
        "tts_first_audio_ms": round(avg_tts_first, 1),
        "endpointing_delay_ms": endpointing_min,
        "total_ms": round(avg_stt_decode + avg_tts_first + endpointing_min, 1),
        "sub_second": (avg_stt_decode + avg_tts_first + endpointing_min) < 1000,
    }

    vosk_runtime.close()
    tts_runtime.close()

    return results


def main() -> None:
    results = run_benchmark()
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
