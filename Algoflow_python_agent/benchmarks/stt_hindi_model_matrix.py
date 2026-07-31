from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
from livekit import rtc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio.stt import LocalSTTRuntime
from audio.tts import LocalTTSRuntime
from config import LocalAudioConfig

logging.basicConfig(level=logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODELS = (
    ROOT / "models" / "stt" / "ggml-tiny-q5_1.bin",
    ROOT / "models" / "stt" / "ggml-base-q5_1.bin",
    ROOT / "models" / "stt" / "ggml-base-q8_0.bin",
    ROOT / "models" / "stt" / "ggml-small-q5_1.bin",
    ROOT / "models" / "stt" / "ggml-small-q8_0.bin",
    ROOT / "models" / "stt" / "ggml-large-v3-turbo-q5_0.bin",
)

TEST_PHRASES = (
    "\u0924\u0941\u092e \u0915\u094c\u0928 \u0939\u094b",
    "\u092e\u0947\u0930\u093e \u0928\u093e\u092e \u0938\u0902\u0938\u094d\u0915\u093e\u0930 \u0939\u0948",
    "\u092e\u0941\u091d\u0947 \u0935\u093f\u0930\u093e\u091f \u0915\u094b\u0939\u0932\u0940 \u0915\u0947 \u092c\u093e\u0930\u0947 \u092e\u0947\u0902 \u092c\u0924\u093e\u0913",
    "\u092d\u093e\u0930\u0924 \u0915\u0940 \u0930\u093e\u091c\u0927\u093e\u0928\u0940 \u0915\u094d\u092f\u093e \u0939\u0948",
)


def _resample_i16(pcm: bytes, source_rate: int, target_rate: int) -> bytes:
    if source_rate == target_rate:
        return pcm

    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return b""

    source_positions = np.arange(samples.size, dtype=np.float32)
    target_count = max(1, int(round(samples.size * target_rate / source_rate)))
    target_positions = np.linspace(0, samples.size - 1, target_count, dtype=np.float32)
    resampled = np.interp(target_positions, source_positions, samples)
    return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()


def _make_frame(pcm: bytes, sample_rate: int) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=pcm,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=len(pcm) // 2,
    )


def _normalize_for_score(text: str) -> str:
    return re.sub(r"\s+", "", re.sub(r"[\W_]+", " ", text.lower(), flags=re.UNICODE))


def _char_error_ratio(expected: str, actual: str) -> float:
    left = _normalize_for_score(expected)
    right = _normalize_for_score(actual)
    if not left:
        return 0.0 if not right else 1.0

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return round(previous[-1] / len(left), 4)


def _contains_devanagari(text: str) -> bool:
    return any("\u0900" <= char <= "\u097f" for char in text)


def _synthesize_fixtures(config: LocalAudioConfig) -> list[dict]:
    tts = LocalTTSRuntime(
        config.tts_model_path,
        config_path=config.tts_config_path,
        sample_rate=config.tts_sample_rate,
        num_channels=config.num_channels,
    )
    fixtures = []
    try:
        for phrase in TEST_PHRASES:
            pcm = b"".join(chunk.pcm for chunk in tts.synthesize_stream(phrase))
            pcm_16k = _resample_i16(pcm, config.tts_sample_rate, config.stt_sample_rate)
            fixtures.append(
                {
                    "expected": phrase,
                    "pcm": pcm_16k,
                    "audio_ms": round((len(pcm_16k) / 2) / config.stt_sample_rate * 1000, 1),
                }
            )
    finally:
        tts.close()
    return fixtures


def run_matrix(models: list[Path], language: str) -> dict:
    config = LocalAudioConfig.from_env()
    config.validate()
    fixtures = _synthesize_fixtures(config)
    results = []

    for model_path in models:
        if not model_path.exists():
            results.append({"model": model_path.name, "missing": True})
            continue

        load_started = time.perf_counter()
        runtime = LocalSTTRuntime(model_path, sample_rate=config.stt_sample_rate, language=language)
        load_ms = round((time.perf_counter() - load_started) * 1000, 1)
        model_results = []
        try:
            for fixture in fixtures:
                frame = _make_frame(fixture["pcm"], config.stt_sample_rate)
                runtime.process_frame(frame)
                started = time.perf_counter()
                result = runtime.flush()
                decode_ms = round((time.perf_counter() - started) * 1000, 1)
                actual = result.final
                model_results.append(
                    {
                        "expected": fixture["expected"],
                        "actual": actual,
                        "audio_ms": fixture["audio_ms"],
                        "decode_ms": decode_ms,
                        "realtime_factor": round(decode_ms / fixture["audio_ms"], 3) if fixture["audio_ms"] else None,
                        "char_error_ratio": _char_error_ratio(fixture["expected"], actual),
                        "devanagari": _contains_devanagari(actual),
                    }
                )
        finally:
            runtime.close()

        decoded = [row for row in model_results if row["actual"].strip()]
        results.append(
            {
                "model": model_path.name,
                "language": language,
                "load_ms": load_ms,
                "avg_decode_ms": round(sum(row["decode_ms"] for row in model_results) / len(model_results), 1),
                "avg_realtime_factor": round(sum(row["realtime_factor"] for row in model_results) / len(model_results), 3),
                "avg_char_error_ratio": round(sum(row["char_error_ratio"] for row in model_results) / len(model_results), 4),
                "devanagari_rate": round(sum(1 for row in decoded if row["devanagari"]) / len(decoded), 3) if decoded else 0.0,
                "cases": model_results,
            }
        )

    return {
        "language": language,
        "sample_rate": config.stt_sample_rate,
        "fixture_source": "local_piper_hi_IN_pratham_medium",
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="hi")
    parser.add_argument("--model", action="append", type=Path, dest="models")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "benchmarks" / "stt_trials",
        help="Directory where the timestamped trial JSON should be written.",
    )
    args = parser.parse_args()

    models = [path.resolve() for path in args.models] if args.models else list(DEFAULT_MODELS)
    result = run_matrix(models, args.language)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{timestamp}_hindi_stt_matrix_{args.language}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
