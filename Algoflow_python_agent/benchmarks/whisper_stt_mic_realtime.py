from __future__ import annotations

import argparse
import asyncio
from collections import deque
import ctypes
import json
import logging
import math
import os
from pathlib import Path
import queue
import sys
import time

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from livekit import rtc
from livekit.agents.vad import VADEventType

from audio.stt import LocalSTTRuntime
from config import LocalAudioConfig

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


def _configure_windows_console_utf8() -> None:
    if os.name != "nt":
        return
    try:
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except Exception:
        pass
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_configure_windows_console_utf8()


def _pcm_i16_rms(pcm: bytes) -> int:
    if len(pcm) < 2:
        return 0
    samples = np.frombuffer(pcm[: (len(pcm) // 2) * 2], dtype=np.int16).astype(np.float64)
    if samples.size == 0:
        return 0
    return int(math.sqrt(float(np.mean(samples * samples))))


class AudioTraceWindow:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sample_count = 0
        self.square_sum = 0
        self.nonzero_count = 0
        self.clipped_count = 0
        self.peak = 0

    def add(self, pcm: bytes) -> None:
        if len(pcm) < 2:
            return
        samples = np.frombuffer(pcm[: (len(pcm) // 2) * 2], dtype=np.int16)
        if samples.size == 0:
            return
        wide = samples.astype(np.int64)
        abs_samples = np.abs(wide)
        self.sample_count += int(samples.size)
        self.square_sum += int(np.sum(wide * wide))
        self.nonzero_count += int(np.count_nonzero(abs_samples > 1))
        self.clipped_count += int(np.count_nonzero(abs_samples >= 32760))
        self.peak = max(self.peak, int(np.max(abs_samples)))

    def snapshot(self) -> dict:
        if self.sample_count <= 0:
            return {
                "samples": 0,
                "rms": 0,
                "peak": 0,
                "dbfs": None,
                "nonzero_pct": 0.0,
                "clipped_pct": 0.0,
            }
        rms = math.sqrt(self.square_sum / self.sample_count)
        dbfs = round(20 * math.log10(rms / 32768), 1) if rms > 0 else None
        return {
            "samples": self.sample_count,
            "rms": int(rms),
            "peak": self.peak,
            "dbfs": dbfs,
            "nonzero_pct": round((self.nonzero_count / self.sample_count) * 100, 1),
            "clipped_pct": round((self.clipped_count / self.sample_count) * 100, 3),
        }


def _prepare_mono_samples(
    pcm: bytes,
    *,
    channels: int,
    channel_mode: str,
    gain: float,
    auto_gain_target_rms: float,
    auto_gain_max: float,
) -> np.ndarray:
    if channels <= 0:
        raise ValueError("channels must be positive")
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return samples

    if channels > 1:
        frames = samples[: (samples.size // channels) * channels].reshape(-1, channels)
        if channel_mode == "mix":
            samples = frames.mean(axis=1)
        elif channel_mode == "left":
            samples = frames[:, 0]
        elif channel_mode == "right":
            samples = frames[:, min(1, channels - 1)]
        elif channel_mode == "max-rms":
            channel_rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=0))
            samples = frames[:, int(np.argmax(channel_rms))]
        else:
            raise ValueError(f"unsupported channel mode: {channel_mode}")

    effective_gain = max(0.0, gain)
    if auto_gain_target_rms > 0:
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) if samples.size else 0.0
        if rms > 0 and rms < auto_gain_target_rms:
            effective_gain *= min(max(1.0, auto_gain_max), auto_gain_target_rms / rms)
    if effective_gain and effective_gain != 1.0:
        samples = samples * effective_gain
    return samples


def _resample_i16(
    pcm: bytes,
    source_rate: int,
    target_rate: int,
    channels: int,
    *,
    channel_mode: str,
    gain: float,
    auto_gain_target_rms: float,
    auto_gain_max: float,
) -> bytes:
    samples = _prepare_mono_samples(
        pcm,
        channels=channels,
        channel_mode=channel_mode,
        gain=gain,
        auto_gain_target_rms=auto_gain_target_rms,
        auto_gain_max=auto_gain_max,
    )
    if samples.size == 0:
        return b""
    if source_rate == target_rate:
        return np.clip(samples, -32768, 32767).astype(np.int16).tobytes()

    source_positions = np.arange(samples.size, dtype=np.float32)
    target_count = max(1, int(round(samples.size * target_rate / source_rate)))
    target_positions = np.linspace(0, samples.size - 1, target_count, dtype=np.float32)
    resampled = np.interp(target_positions, source_positions, samples)
    return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()


def _device_arg(value: str | None):
    if value is None:
        return None
    return int(value) if value.isdigit() else value


def _device_label(device) -> str:
    if device is None:
        return "default"
    return str(device)


def _make_frame(pcm: bytes, sample_rate: int) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=pcm,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=len(pcm) // 2,
    )


def _print_json(args: argparse.Namespace, event: dict) -> None:
    if args.json:
        print(json.dumps(event, ensure_ascii=False), flush=True)


def _print_trace(args: argparse.Namespace, *, audio_ms: float, trace: AudioTraceWindow) -> None:
    if not args.trace_audio:
        return
    stats = trace.snapshot()
    event = {"event": "audio_trace", "audio_ms": round(audio_ms, 1), **stats}
    if args.json:
        _print_json(args, event)
        return
    print(
        "TRACE "
        f"audio={audio_ms:.0f}ms samples={stats['samples']} rms={stats['rms']} "
        f"peak={stats['peak']} dbfs={stats['dbfs']} nonzero={stats['nonzero_pct']}% "
        f"clipped={stats['clipped_pct']}%"
    )


def _print_live_level(
    args: argparse.Namespace,
    *,
    trace: AudioTraceWindow,
    vad_probability: float | None,
    in_speech: bool,
) -> None:
    if args.quiet_levels or args.json:
        return
    stats = trace.snapshot()
    vad_label = "n/a" if vad_probability is None else f"{vad_probability:.2f}"
    state = "speech" if in_speech else "idle"
    print(
        "\r"
        f"MIC rms={stats['rms']} peak={stats['peak']} dbfs={stats['dbfs']} "
        f"vad={vad_label} state={state}    ",
        end="",
        flush=True,
    )


def _print_ready(args: argparse.Namespace, config: LocalAudioConfig, model_path: Path) -> None:
    event = {
        "event": "mic_stt_starting",
        "provider": "whisper.cpp",
        "model": model_path.name,
        "language": args.language or config.stt_language,
        "device": _device_label(_device_arg(args.device)),
        "mic_sample_rate": args.mic_sample_rate,
        "mic_channels": args.channels,
        "stt_sample_rate": config.stt_sample_rate,
        "channel_mode": args.channel_mode,
        "gain": args.gain,
        "endpoint_silence_ms": args.endpoint_silence_ms,
        "silero_activation_threshold": args.silero_activation_threshold,
    }
    if args.json:
        _print_json(args, event)
        return
    print(
        "Mic Whisper STT ready: "
        f"device={event['device']} model={event['model']} language={event['language']} "
        f"channel={event['channel_mode']} gain={event['gain']} "
        f"vad_threshold={event['silero_activation_threshold']} silence={event['endpoint_silence_ms']}ms"
    )
    print("Speak. Press Ctrl+C to stop.")


async def run_mic(args: argparse.Namespace) -> None:
    if args.debug:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger("livekit.plugins.silero").setLevel(logging.ERROR)

    from livekit.plugins import silero

    config = LocalAudioConfig.from_env()
    config.validate()

    model_path = args.model_path.resolve() if args.model_path else config.stt_model_path
    language = args.language or config.stt_language
    runtime = LocalSTTRuntime(model_path, sample_rate=config.stt_sample_rate, language=language)

    mic_sample_rate = int(args.mic_sample_rate or config.stt_sample_rate)
    blocksize = max(1, int(mic_sample_rate * args.chunk_ms / 1000))
    mic_channels = max(1, args.channels)
    endpoint_silence_ms = args.endpoint_silence_ms if args.endpoint_silence_ms is not None else config.stt_endpoint_silence_ms
    min_utterance_ms = args.min_utterance_ms if args.min_utterance_ms is not None else config.stt_min_utterance_ms

    vad = silero.VAD.load(
        min_speech_duration=max(0.01, min_utterance_ms / 1000),
        min_silence_duration=max(0.05, endpoint_silence_ms / 1000),
        prefix_padding_duration=max(0.0, args.preroll_ms / 1000),
        activation_threshold=args.silero_activation_threshold,
        sample_rate=config.stt_sample_rate,
    )
    vad_stream = vad.stream()
    audio_queue: queue.Queue[bytes] = queue.Queue()

    def callback(indata, frames, callback_time, status) -> None:
        if status and args.debug:
            print(f"[audio_status] {status}", file=sys.stderr)
        audio_queue.put(bytes(indata))

    _print_ready(args, config, model_path)

    in_speech = False
    utterance_index = 0
    utterance_started_at = 0.0
    utterance_audio_ms = 0.0
    trace = AudioTraceWindow()
    live_trace = AudioTraceWindow()
    partial_text = ""
    last_partial_at = 0.0
    last_level_at = 0.0
    last_vad_probability: float | None = None
    peak_vad_probability = 0.0
    stream_started_at = time.perf_counter()

    def accept_frame(frame: rtc.AudioFrame) -> None:
        nonlocal utterance_audio_ms
        pcm = frame.data.tobytes()
        trace.add(pcm)
        runtime.process_frame(frame)
        utterance_audio_ms += frame.duration * 1000

    def emit_partial() -> None:
        nonlocal partial_text
        nonlocal last_partial_at
        if args.partial_interval_ms <= 0 or utterance_audio_ms < args.min_partial_audio_ms:
            return
        now = time.perf_counter()
        if last_partial_at and (now - last_partial_at) * 1000 < args.partial_interval_ms:
            return
        last_partial_at = now
        started = time.perf_counter()
        result = runtime.partial(args.partial_window_ms)
        decode_ms = round((time.perf_counter() - started) * 1000, 1)
        text = result.partial.strip()
        if not text or text == partial_text:
            return
        partial_text = text
        if args.json:
            _print_json(
                args,
                {
                    "event": "partial",
                    "utterance": utterance_index,
                    "decode_ms": decode_ms,
                    "audio_ms": round(utterance_audio_ms, 1),
                    "text": text,
                },
            )
        else:
            print(f"\rYOU: {text}", end="", flush=True)

    def finalize_utterance(*, interrupted: bool = False) -> None:
        nonlocal utterance_audio_ms
        nonlocal partial_text
        nonlocal last_partial_at
        if utterance_audio_ms <= 0:
            return
        started = time.perf_counter()
        result = runtime.flush()
        decode_ms = round((time.perf_counter() - started) * 1000, 1)
        text = result.final.strip() or partial_text.strip()
        event = {
            "event": "final_on_stop" if interrupted else "final",
            "utterance": utterance_index,
            "elapsed_ms": round((time.perf_counter() - utterance_started_at) * 1000, 1),
            "utterance_audio_ms": round(utterance_audio_ms, 1),
            "decode_ms": decode_ms,
            "realtime_factor": round(decode_ms / utterance_audio_ms, 3) if utterance_audio_ms else None,
            "confidence": round(result.confidence, 3),
            "no_speech_prob": round(result.no_speech_prob, 3),
            "audio_energy": round(result.audio_energy, 5),
            "filtered": result.filtered,
            "filter_reason": result.filter_reason,
            "text": text,
        }
        if args.json:
            _print_json(args, event)
            if args.trace_audio:
                _print_trace(args, audio_ms=utterance_audio_ms, trace=trace)
        elif text:
            suffix = (
                f" (decode={decode_ms}ms rtf={event['realtime_factor']} "
                f"conf={event['confidence']} no_speech={event['no_speech_prob']})"
                if args.show_timing
                else ""
            )
            print(f"\rYOU: {text}{suffix}".ljust(140))
            _print_trace(args, audio_ms=utterance_audio_ms, trace=trace)
        elif args.show_empty:
            reason = result.filter_reason or "empty"
            print(
                f"\r(no transcript; reason={reason} audio={utterance_audio_ms:.0f}ms "
                f"decode={decode_ms}ms no_speech={event['no_speech_prob']} energy={event['audio_energy']})".ljust(160)
            )
            _print_trace(args, audio_ms=utterance_audio_ms, trace=trace)
        runtime.reset()
        trace.reset()
        utterance_audio_ms = 0.0
        partial_text = ""
        last_partial_at = 0.0

    async def vad_events_task() -> None:
        nonlocal in_speech
        nonlocal utterance_index
        nonlocal utterance_started_at
        nonlocal utterance_audio_ms
        nonlocal partial_text
        nonlocal last_partial_at
        nonlocal last_vad_probability
        nonlocal peak_vad_probability

        async for event in vad_stream:
            if event.type == VADEventType.START_OF_SPEECH:
                utterance_index += 1
                in_speech = True
                utterance_started_at = time.perf_counter()
                utterance_audio_ms = 0.0
                partial_text = ""
                last_partial_at = 0.0
                trace.reset()
                runtime.reset()
                if args.show_events or not args.quiet_levels:
                    print("\nListening...")
                for frame in event.frames:
                    accept_frame(frame)

            elif event.type == VADEventType.INFERENCE_DONE:
                last_vad_probability = event.probability
                peak_vad_probability = max(peak_vad_probability, event.probability)
                if not in_speech:
                    continue
                for frame in event.frames:
                    accept_frame(frame)
                emit_partial()

            elif event.type == VADEventType.END_OF_SPEECH and in_speech:
                if args.show_events or not args.quiet_levels:
                    print("\nDecoding...")
                finalize_utterance()
                in_speech = False
                peak_vad_probability = 0.0

    events_task = asyncio.create_task(vad_events_task())
    try:
        with sd.RawInputStream(
            samplerate=mic_sample_rate,
            blocksize=blocksize,
            device=_device_arg(args.device),
            channels=mic_channels,
            dtype="int16",
            callback=callback,
        ):
            while True:
                if args.duration and (time.perf_counter() - stream_started_at) >= args.duration:
                    break
                pcm = await asyncio.to_thread(audio_queue.get)
                stt_pcm = _resample_i16(
                    pcm,
                    mic_sample_rate,
                    config.stt_sample_rate,
                    mic_channels,
                    channel_mode=args.channel_mode,
                    gain=args.gain,
                    auto_gain_target_rms=args.auto_gain_target_rms,
                    auto_gain_max=args.auto_gain_max,
                )
                if stt_pcm:
                    live_trace.add(stt_pcm)
                    vad_stream.push_frame(_make_frame(stt_pcm, config.stt_sample_rate))
                    now = time.perf_counter()
                    if not last_level_at or (now - last_level_at) * 1000 >= args.level_interval_ms:
                        _print_live_level(
                            args,
                            trace=live_trace,
                            vad_probability=last_vad_probability,
                            in_speech=in_speech,
                        )
                        live_trace.reset()
                        last_level_at = now
    except KeyboardInterrupt:
        print("\nStopping mic STT.")
    finally:
        vad_stream.end_input()
        await asyncio.sleep(0.1)
        events_task.cancel()
        await asyncio.gather(events_task, return_exceptions=True)
        await vad_stream.aclose()
        if in_speech:
            finalize_utterance(interrupted=True)
        runtime.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit.")
    parser.add_argument("--device", help="Input device id or name. Omit to use the system default microphone.")
    parser.add_argument("--duration", type=float, help="Optional max runtime in seconds. Default: run until Ctrl+C.")
    parser.add_argument("--model-path", type=Path, help="Whisper ggml model path. Default: LOCAL_STT_MODEL_PATH.")
    parser.add_argument("--language", default=None, help="Whisper language code. Default: LOCAL_STT_LANGUAGE.")
    parser.add_argument("--mic-sample-rate", type=int, default=16000, help="Microphone capture sample rate. Default: 16000.")
    parser.add_argument("--channels", type=int, default=1, help="Microphone capture channels before downmixing. Default: 1.")
    parser.add_argument(
        "--channel-mode",
        choices=("max-rms", "mix", "left", "right"),
        default="max-rms",
        help="How to reduce multi-channel mic input to mono. Default: max-rms.",
    )
    parser.add_argument("--gain", type=float, default=4.0, help="Linear input gain before sending audio to Whisper. Default: 4.0.")
    parser.add_argument("--auto-gain-target-rms", type=float, default=0.0, help="If >0, boost each frame toward this RMS. Default: disabled.")
    parser.add_argument("--auto-gain-max", type=float, default=8.0, help="Maximum auto-gain multiplier. Default: 8.0.")
    parser.add_argument("--chunk-ms", type=float, default=50.0, help="Microphone chunk size. Default: 50 ms.")
    parser.add_argument("--endpoint-silence-ms", type=float, default=700.0, help="Silero endpoint silence. Default: 700 ms.")
    parser.add_argument("--min-utterance-ms", type=float, default=250.0, help="Minimum utterance before endpointing. Default: 250 ms.")
    parser.add_argument("--preroll-ms", type=float, default=300.0, help="Audio before VAD start to include. Default: 300 ms.")
    parser.add_argument("--silero-activation-threshold", type=float, default=0.35, help="Silero speech probability threshold. Default: 0.35.")
    parser.add_argument("--partial-interval-ms", type=float, default=0.0, help="Optional Whisper partial decode interval. Default: disabled.")
    parser.add_argument("--min-partial-audio-ms", type=float, default=1200.0, help="Minimum audio before partial decode. Default: 1200 ms.")
    parser.add_argument("--partial-window-ms", type=float, default=1600.0, help="Recent audio window for partial decode. Default: 1600 ms.")
    parser.add_argument("--show-empty", action="store_true", help="Show empty final events.")
    parser.add_argument("--show-events", action="store_true", help="Show speech start lines.")
    parser.add_argument("--show-timing", action="store_true", help="Show decode time and realtime factor after each transcript.")
    parser.add_argument("--quiet-levels", action="store_true", help="Hide the live mic/VAD level line.")
    parser.add_argument("--level-interval-ms", type=float, default=500.0, help="Live mic/VAD level refresh interval. Default: 500 ms.")
    parser.add_argument("--trace-audio", action="store_true", help="Print compact PCM stats for each utterance.")
    parser.add_argument("--json", action="store_true", help="Print raw JSON events.")
    parser.add_argument("--debug", action="store_true", help="Enable internal INFO logs.")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    asyncio.run(run_mic(args))


if __name__ == "__main__":
    main()
