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
import wave

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from livekit import rtc
from livekit.agents.vad import VADEventType
from audio.stt import VoskSTTRuntime
from audio.stt.runtime import STTResult
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
    sample_count = len(pcm) // 2
    samples = memoryview(pcm[: sample_count * 2]).cast("h")
    if not samples:
        return 0
    square_sum = sum(int(sample) * int(sample) for sample in samples)
    return int(math.sqrt(square_sum / len(samples)))


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
    channel_mode: str = "mix",
    gain: float = 1.0,
    auto_gain_target_rms: float = 0.0,
    auto_gain_max: float = 8.0,
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


def _device_label(device) -> str:
    if device is None:
        return "default"
    return str(device)


def _device_arg(value: str | None):
    if value is None:
        return None
    return int(value) if value.isdigit() else value


def _make_frame(pcm: bytes, sample_rate: int) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=pcm,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=len(pcm) // 2,
    )


def list_devices() -> None:
    print(sd.query_devices())


def _write_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)


def _extract_vosk_text(raw_json: str, key: str) -> str:
    if not raw_json:
        return ""
    try:
        return str(json.loads(raw_json).get(key, "")).strip()
    except json.JSONDecodeError:
        return ""


def _print_audio_trace(
    args: argparse.Namespace,
    *,
    mode: str,
    audio_ms: float,
    trace_window: AudioTraceWindow,
    vosk_partial: str = "",
    vosk_final: str = "",
) -> None:
    stats = trace_window.snapshot()
    if args.json:
        print(
            json.dumps(
                {
                    "event": "audio_trace",
                    "mode": mode,
                    "audio_ms": round(audio_ms, 1),
                    **stats,
                    "vosk_partial": vosk_partial,
                    "vosk_final": vosk_final,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return

    print(
        "\nTRACE "
        f"{mode} audio={audio_ms:.0f}ms "
        f"samples={stats['samples']} rms={stats['rms']} peak={stats['peak']} "
        f"dbfs={stats['dbfs']} nonzero={stats['nonzero_pct']}% clipped={stats['clipped_pct']}% "
        f"vosk_partial={vosk_partial!r} vosk_final={vosk_final!r}"
    )


def _print_event(args: argparse.Namespace, event: dict) -> None:
    if args.json:
        print(json.dumps(event, ensure_ascii=False), flush=True)
        return

    event_type = event["event"]
    if event_type == "mic_stt_starting":
        silence_ms = event.get("endpoint_silence_ms")
        silence_label = f" silence={silence_ms}ms" if silence_ms is not None else ""
        print(
            "Mic STT ready: "
            f"device={event['device']} model={event['model']} "
            f"vad={event['speech_rms_threshold']}{silence_label} "
            f"channel={event.get('channel_mode')} gain={event.get('gain')} "
            f"auto_gain_target={event.get('auto_gain_target_rms')}"
        )
        print("Speak in Hindi. Press Ctrl+C to stop.")
    elif event_type == "speech_start" and args.show_events:
        print(f"\nListening... rms={event['rms']}")
    elif event_type == "partial":
        print(f"\rYOU: {event['text']}", end="", flush=True)
    elif event_type == "segment" and args.show_events:
        print(f"\rSEGMENT: {event['text']}".ljust(120))
    elif event_type == "final":
        text = event["text"].strip()
        if text:
            if args.show_events or args.debug:
                print(
                    f"\rYOU: {text} "
                    f"(first={event.get('first_partial_ms')}ms final={event.get('finalization_ms')}ms)".ljust(120)
                )
            else:
                print(f"\rYOU: {text}".ljust(120))
        elif args.show_empty or args.debug:
            print(
                "\r(no transcript; try speaking Hindi closer/louder, or run with "
                f"--show-rms. peak_rms={event.get('peak_rms')})".ljust(120)
            )
    elif event_type == "final_on_stop":
        text = event["text"].strip()
        if text:
            print(f"\nYOU: {text}")


def run_mic(args: argparse.Namespace) -> None:
    if args.vad == "silero":
        asyncio.run(run_mic_silero(args))
    elif args.vad == "none":
        run_mic_direct(args)
    else:
        run_mic_rms(args)


async def run_mic_silero(args: argparse.Namespace) -> None:
    if args.debug:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger("livekit.plugins.silero").setLevel(logging.ERROR)

    from livekit.plugins import silero

    config = LocalAudioConfig.from_env()
    config.validate()

    runtime = VoskSTTRuntime(
        config.vosk_model_path,
        sample_rate=config.stt_sample_rate,
        language=config.stt_language,
    )
    session = runtime.create_session()
    audio_queue: queue.Queue[bytes] = queue.Queue()

    mic_sample_rate = int(args.mic_sample_rate or config.stt_sample_rate)
    blocksize = max(1, int(mic_sample_rate * args.chunk_ms / 1000))
    mic_channels = max(1, args.channels)
    partial_interval_ms = args.partial_interval_ms if args.partial_interval_ms is not None else config.vosk_partial_interval_ms
    min_partial_audio_ms = args.min_partial_audio_ms if args.min_partial_audio_ms is not None else config.vosk_min_partial_audio_ms
    endpoint_silence_ms = args.endpoint_silence_ms if args.endpoint_silence_ms is not None else config.stt_endpoint_silence_ms
    min_utterance_ms = args.min_utterance_ms if args.min_utterance_ms is not None else config.stt_min_utterance_ms
    activation_threshold = args.silero_activation_threshold

    vad = silero.VAD.load(
        min_speech_duration=max(0.01, min_utterance_ms / 1000),
        min_silence_duration=max(0.05, endpoint_silence_ms / 1000),
        prefix_padding_duration=max(0.0, args.preroll_ms / 1000),
        activation_threshold=activation_threshold,
        sample_rate=config.stt_sample_rate,
    )
    vad_stream = vad.stream()

    def callback(indata, frames, callback_time, status) -> None:
        if status and args.debug:
            print(f"[audio_status] {status}", file=sys.stderr)
        audio_queue.put(bytes(indata))

    _print_event(
        args,
        {
            "event": "mic_stt_starting",
            "device": _device_label(_device_arg(args.device)),
            "mic_sample_rate": mic_sample_rate,
            "mic_channels": mic_channels,
            "stt_sample_rate": config.stt_sample_rate,
            "chunk_ms": args.chunk_ms,
            "speech_rms_threshold": "silero",
            "endpoint_silence_ms": endpoint_silence_ms,
            "min_utterance_ms": min_utterance_ms,
            "partial_interval_ms": partial_interval_ms,
            "preroll_ms": args.preroll_ms,
            "channel_mode": args.channel_mode,
            "gain": args.gain,
            "auto_gain_target_rms": args.auto_gain_target_rms,
            "model": config.vosk_model_path.name,
        },
    )

    utterance_index = 0
    in_speech = False
    utterance_started_at = 0.0
    utterance_audio_ms = 0.0
    first_partial_ms: float | None = None
    last_partial_at = 0.0
    last_partial = ""
    peak_probability = 0.0
    stream_started_at = time.perf_counter()
    captured_pcm = bytearray()
    trace_window = AudioTraceWindow()
    last_trace_at = stream_started_at

    def accept_frame(frame: rtc.AudioFrame) -> tuple[float, STTResult]:
        accepted_started = time.perf_counter()
        pcm = frame.data.tobytes()
        if args.save_wav:
            captured_pcm.extend(pcm)
        if args.trace_audio:
            trace_window.add(pcm)
        result = session.accept_pcm_i16(
            pcm,
            sample_rate=frame.sample_rate,
            channels=frame.num_channels,
            duration=frame.duration,
        )
        return (time.perf_counter() - accepted_started) * 1000, result

    def emit_segment(result: STTResult) -> None:
        nonlocal first_partial_ms
        nonlocal last_partial
        text = result.final.strip()
        if not text or text == last_partial:
            return
        elapsed_ms = round((time.perf_counter() - utterance_started_at) * 1000, 1)
        if first_partial_ms is None:
            first_partial_ms = elapsed_ms
        last_partial = text
        _print_event(
            args,
            {
                "event": "segment",
                "utterance": utterance_index,
                "at_ms": elapsed_ms,
                "text": text,
            },
        )

    async def vad_events_task() -> None:
        nonlocal session
        nonlocal utterance_index
        nonlocal in_speech
        nonlocal utterance_started_at
        nonlocal utterance_audio_ms
        nonlocal first_partial_ms
        nonlocal last_partial_at
        nonlocal last_partial
        nonlocal peak_probability
        nonlocal last_trace_at

        async for event in vad_stream:
            if event.type == VADEventType.INFERENCE_DONE:
                peak_probability = max(peak_probability, event.probability)
                if args.show_rms:
                    print(
                        f"\rSilero speech_prob={event.probability:.3f} peak={peak_probability:.3f} threshold={activation_threshold:.2f}",
                        end="",
                        flush=True,
                    )
                if not in_speech:
                    continue

                accept_ms = 0.0
                for frame in event.frames:
                    frame_accept_ms, frame_result = accept_frame(frame)
                    accept_ms += frame_accept_ms
                    emit_segment(frame_result)
                    utterance_audio_ms += frame.duration * 1000

                now = time.perf_counter()
                if (
                    partial_interval_ms > 0
                    and utterance_audio_ms >= min_partial_audio_ms
                    and (not last_partial_at or (now - last_partial_at) * 1000 >= partial_interval_ms)
                ):
                    partial = session.partial().partial.strip()
                    last_partial_at = now
                    if partial and partial != last_partial:
                        elapsed_ms = round((time.perf_counter() - utterance_started_at) * 1000, 1)
                        if first_partial_ms is None:
                            first_partial_ms = elapsed_ms
                        last_partial = partial
                        _print_event(
                            args,
                            {
                                "event": "partial",
                                "utterance": utterance_index,
                                "at_ms": elapsed_ms,
                                "accept_ms": round(accept_ms, 1),
                                "rms": None,
                                "text": partial,
                            },
                        )

                if args.trace_audio and (
                    not last_trace_at or (now - last_trace_at) * 1000 >= args.trace_interval_ms
                ):
                    raw_partial = _extract_vosk_text(session.last_raw_partial, "partial")
                    raw_final = _extract_vosk_text(session.last_raw_result, "text")
                    _print_audio_trace(
                        args,
                        mode="silero->vosk",
                        audio_ms=utterance_audio_ms,
                        trace_window=trace_window,
                        vosk_partial=raw_partial,
                        vosk_final=raw_final,
                    )
                    trace_window.reset()
                    last_trace_at = now

            elif event.type == VADEventType.START_OF_SPEECH:
                utterance_index += 1
                in_speech = True
                utterance_started_at = time.perf_counter()
                utterance_audio_ms = 0.0
                first_partial_ms = None
                last_partial_at = 0.0
                last_partial = ""
                peak_probability = 0.0
                session = runtime.create_session()
                trace_window.reset()
                last_trace_at = utterance_started_at
                _print_event(args, {"event": "speech_start", "utterance": utterance_index, "rms": "silero"})
                for frame in event.frames:
                    _, frame_result = accept_frame(frame)
                    emit_segment(frame_result)
                    utterance_audio_ms += frame.duration * 1000

            elif event.type == VADEventType.END_OF_SPEECH and in_speech:
                finalize_started = time.perf_counter()
                final = session.flush()
                finalization_ms = round((time.perf_counter() - finalize_started) * 1000, 1)
                elapsed_ms = round((time.perf_counter() - utterance_started_at) * 1000, 1)
                _print_event(
                    args,
                    {
                        "event": "final",
                        "utterance": utterance_index,
                        "elapsed_ms": elapsed_ms,
                        "first_partial_ms": first_partial_ms,
                        "finalization_ms": finalization_ms,
                        "utterance_audio_ms": round(utterance_audio_ms, 1),
                        "silence_ms": round(event.silence_duration * 1000, 1),
                        "peak_rms": round(peak_probability, 3),
                        "text": final.final,
                    },
                )
                in_speech = False
                session = runtime.create_session()

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
                vad_stream.push_frame(_make_frame(stt_pcm, config.stt_sample_rate))
    except KeyboardInterrupt:
        print("\nStopping mic STT.")
    finally:
        vad_stream.end_input()
        await asyncio.sleep(0.1)
        events_task.cancel()
        await asyncio.gather(events_task, return_exceptions=True)
        await vad_stream.aclose()
        if in_speech:
            final = session.flush()
            _print_event(args, {"event": "final_on_stop", "text": final.final})
        if args.save_wav and captured_pcm:
            _write_wav(args.save_wav, bytes(captured_pcm), config.stt_sample_rate)
            print(f"\nSaved captured Vosk audio: {args.save_wav}")
        runtime.close()


def run_mic_direct(args: argparse.Namespace) -> None:
    if args.debug:
        logging.getLogger().setLevel(logging.INFO)

    config = LocalAudioConfig.from_env()
    config.validate()

    runtime = VoskSTTRuntime(
        config.vosk_model_path,
        sample_rate=config.stt_sample_rate,
        language=config.stt_language,
    )
    session = runtime.create_session()
    audio_queue: queue.Queue[bytes] = queue.Queue()

    mic_sample_rate = int(args.mic_sample_rate or config.stt_sample_rate)
    blocksize = max(1, int(mic_sample_rate * args.chunk_ms / 1000))
    mic_channels = max(1, args.channels)
    partial_interval_ms = args.partial_interval_ms if args.partial_interval_ms is not None else config.vosk_partial_interval_ms
    min_partial_audio_ms = args.min_partial_audio_ms if args.min_partial_audio_ms is not None else config.vosk_min_partial_audio_ms

    def callback(indata, frames, callback_time, status) -> None:
        if status and args.debug:
            print(f"[audio_status] {status}", file=sys.stderr)
        audio_queue.put(bytes(indata))

    _print_event(
        args,
        {
            "event": "mic_stt_starting",
            "device": _device_label(_device_arg(args.device)),
            "mic_sample_rate": mic_sample_rate,
            "mic_channels": mic_channels,
            "stt_sample_rate": config.stt_sample_rate,
            "chunk_ms": args.chunk_ms,
            "speech_rms_threshold": "none",
            "endpoint_silence_ms": None,
            "min_utterance_ms": None,
            "partial_interval_ms": partial_interval_ms,
            "preroll_ms": 0.0,
            "channel_mode": args.channel_mode,
            "gain": args.gain,
            "auto_gain_target_rms": args.auto_gain_target_rms,
            "model": config.vosk_model_path.name,
        },
    )

    stream_started_at = time.perf_counter()
    stream_audio_ms = 0.0
    last_partial_at = 0.0
    last_partial = ""
    last_final = ""
    first_partial_ms: float | None = None
    captured_pcm = bytearray()
    trace_window = AudioTraceWindow()
    last_trace_at = stream_started_at

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

                pcm = audio_queue.get()
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
                if not stt_pcm:
                    continue

                if args.save_wav:
                    captured_pcm.extend(stt_pcm)
                if args.trace_audio:
                    trace_window.add(stt_pcm)

                frame_duration_ms = (len(stt_pcm) / 2) / config.stt_sample_rate * 1000
                stream_audio_ms += frame_duration_ms
                accepted_started = time.perf_counter()
                result = session.accept_pcm_i16(
                    stt_pcm,
                    sample_rate=config.stt_sample_rate,
                    channels=1,
                    duration=frame_duration_ms / 1000,
                )
                accept_ms = round((time.perf_counter() - accepted_started) * 1000, 1)
                now = time.perf_counter()

                text = result.final.strip()
                if text and text != last_final:
                    last_final = text
                    _print_event(
                        args,
                        {
                            "event": "final",
                            "utterance": 0,
                            "elapsed_ms": round((now - stream_started_at) * 1000, 1),
                            "first_partial_ms": first_partial_ms,
                            "finalization_ms": accept_ms,
                            "utterance_audio_ms": round(stream_audio_ms, 1),
                            "silence_ms": None,
                            "peak_rms": None,
                            "text": text,
                        },
                    )
                    first_partial_ms = None
                    last_partial = ""

                if (
                    partial_interval_ms > 0
                    and stream_audio_ms >= min_partial_audio_ms
                    and (not last_partial_at or (now - last_partial_at) * 1000 >= partial_interval_ms)
                ):
                    partial = session.partial().partial.strip()
                    last_partial_at = now
                    if partial and partial != last_partial and partial != last_final:
                        elapsed_ms = round((time.perf_counter() - stream_started_at) * 1000, 1)
                        if first_partial_ms is None:
                            first_partial_ms = elapsed_ms
                        last_partial = partial
                        _print_event(
                            args,
                            {
                                "event": "partial",
                                "utterance": 0,
                                "at_ms": elapsed_ms,
                                "accept_ms": accept_ms,
                                "rms": None,
                                "text": partial,
                            },
                        )

                if args.trace_audio and (
                    not last_trace_at or (now - last_trace_at) * 1000 >= args.trace_interval_ms
                ):
                    raw_partial = _extract_vosk_text(session.last_raw_partial, "partial")
                    raw_final = _extract_vosk_text(session.last_raw_result, "text")
                    _print_audio_trace(
                        args,
                        mode="direct",
                        audio_ms=stream_audio_ms,
                        trace_window=trace_window,
                        vosk_partial=raw_partial,
                        vosk_final=raw_final,
                    )
                    trace_window.reset()
                    last_trace_at = now
    except KeyboardInterrupt:
        print("\nStopping mic STT.")
    finally:
        final = session.flush()
        text = final.final.strip()
        if text and text != last_final:
            _print_event(args, {"event": "final_on_stop", "text": text})
        if args.save_wav and captured_pcm:
            _write_wav(args.save_wav, bytes(captured_pcm), config.stt_sample_rate)
            print(f"\nSaved captured Vosk audio: {args.save_wav}")
        runtime.close()


def run_mic_rms(args: argparse.Namespace) -> None:
    if args.debug:
        logging.getLogger().setLevel(logging.INFO)

    config = LocalAudioConfig.from_env()
    config.validate()

    runtime = VoskSTTRuntime(
        config.vosk_model_path,
        sample_rate=config.stt_sample_rate,
        language=config.stt_language,
    )
    session = runtime.create_session()
    audio_queue: queue.Queue[bytes] = queue.Queue()

    mic_sample_rate = int(args.mic_sample_rate or config.stt_sample_rate)
    blocksize = max(1, int(mic_sample_rate * args.chunk_ms / 1000))
    mic_channels = max(1, args.channels)
    speech_threshold = args.speech_rms_threshold if args.speech_rms_threshold is not None else config.stt_speech_rms_threshold
    endpoint_silence_ms = args.endpoint_silence_ms if args.endpoint_silence_ms is not None else config.stt_endpoint_silence_ms
    min_utterance_ms = args.min_utterance_ms if args.min_utterance_ms is not None else config.stt_min_utterance_ms
    partial_interval_ms = args.partial_interval_ms if args.partial_interval_ms is not None else config.vosk_partial_interval_ms
    min_partial_audio_ms = args.min_partial_audio_ms if args.min_partial_audio_ms is not None else config.vosk_min_partial_audio_ms
    preroll_ms = max(0.0, args.preroll_ms)

    def callback(indata, frames, callback_time, status) -> None:
        if status and args.debug:
            print(f"[audio_status] {status}", file=sys.stderr)
        audio_queue.put(bytes(indata))

    _print_event(
        args,
        {
            "event": "mic_stt_starting",
            "device": _device_label(_device_arg(args.device)),
            "mic_sample_rate": mic_sample_rate,
            "mic_channels": mic_channels,
            "stt_sample_rate": config.stt_sample_rate,
            "chunk_ms": args.chunk_ms,
            "speech_rms_threshold": speech_threshold,
            "endpoint_silence_ms": endpoint_silence_ms,
            "min_utterance_ms": min_utterance_ms,
            "partial_interval_ms": partial_interval_ms,
            "preroll_ms": preroll_ms,
            "channel_mode": args.channel_mode,
            "gain": args.gain,
            "auto_gain_target_rms": args.auto_gain_target_rms,
            "model": config.vosk_model_path.name,
        },
    )

    in_speech = False
    utterance_started_at = 0.0
    last_partial_at = 0.0
    first_partial_ms: float | None = None
    utterance_audio_ms = 0.0
    silence_ms = 0.0
    last_partial = ""
    utterance_index = 0
    peak_rms = 0
    last_rms_print_at = 0.0
    preroll_frames: deque[tuple[bytes, float, int]] = deque()
    preroll_total_ms = 0.0

    stream_started_at = time.perf_counter()
    captured_pcm = bytearray()
    trace_window = AudioTraceWindow()
    last_trace_at = stream_started_at

    def push_preroll(stt_pcm: bytes, duration_ms: float, rms: int) -> None:
        nonlocal preroll_total_ms
        if preroll_ms <= 0:
            return
        preroll_frames.append((stt_pcm, duration_ms, rms))
        preroll_total_ms += duration_ms
        while preroll_total_ms > preroll_ms and preroll_frames:
            _, old_duration_ms, _ = preroll_frames.popleft()
            preroll_total_ms -= old_duration_ms

    def accept_audio(stt_pcm: bytes, duration_ms: float) -> float:
        accepted_started = time.perf_counter()
        if args.save_wav:
            captured_pcm.extend(stt_pcm)
        if args.trace_audio:
            trace_window.add(stt_pcm)
        session.accept_pcm_i16(
            stt_pcm,
            sample_rate=config.stt_sample_rate,
            channels=1,
            duration=duration_ms / 1000,
        )
        return (time.perf_counter() - accepted_started) * 1000

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

                pcm = audio_queue.get()
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
                frame_duration_ms = (len(stt_pcm) / 2) / config.stt_sample_rate * 1000
                rms = _pcm_i16_rms(stt_pcm)
                peak_rms = max(peak_rms, rms)
                is_speech = rms >= speech_threshold
                now = time.perf_counter()

                if args.show_rms and now - last_rms_print_at >= 0.5:
                    print(f"\rRMS={rms} peak={peak_rms} threshold={speech_threshold}", end="", flush=True)
                    last_rms_print_at = now

                if not in_speech:
                    push_preroll(stt_pcm, frame_duration_ms, rms)
                    if not is_speech:
                        continue

                    utterance_index += 1
                    in_speech = True
                    utterance_started_at = now
                    last_partial_at = 0.0
                    first_partial_ms = None
                    utterance_audio_ms = 0.0
                    silence_ms = 0.0
                    last_partial = ""
                    peak_rms = rms
                    session = runtime.create_session()
                    trace_window.reset()
                    last_trace_at = utterance_started_at
                    _print_event(args, {"event": "speech_start", "utterance": utterance_index, "rms": rms})

                    for preroll_pcm, preroll_duration_ms, _ in preroll_frames:
                        accept_audio(preroll_pcm, preroll_duration_ms)
                        utterance_audio_ms += preroll_duration_ms
                    preroll_frames.clear()

                if in_speech:
                    accept_ms = accept_audio(stt_pcm, frame_duration_ms)
                    utterance_audio_ms += frame_duration_ms
                    if is_speech:
                        silence_ms = 0.0
                    else:
                        silence_ms += frame_duration_ms

                    if (
                        partial_interval_ms > 0
                        and utterance_audio_ms >= min_partial_audio_ms
                        and (not last_partial_at or (now - last_partial_at) * 1000 >= partial_interval_ms)
                    ):
                        partial = session.partial().partial.strip()
                        last_partial_at = now
                        if partial and partial != last_partial:
                            elapsed_ms = round((time.perf_counter() - utterance_started_at) * 1000, 1)
                            if first_partial_ms is None:
                                first_partial_ms = elapsed_ms
                            last_partial = partial
                            _print_event(
                                args,
                                {
                                    "event": "partial",
                                    "utterance": utterance_index,
                                    "at_ms": elapsed_ms,
                                    "accept_ms": round(accept_ms, 1),
                                    "rms": rms,
                                    "text": partial,
                            },
                        )

                    if args.trace_audio and (
                        not last_trace_at or (now - last_trace_at) * 1000 >= args.trace_interval_ms
                    ):
                        raw_partial = _extract_vosk_text(session.last_raw_partial, "partial")
                        raw_final = _extract_vosk_text(session.last_raw_result, "text")
                        _print_audio_trace(
                            args,
                            mode="rms->vosk",
                            audio_ms=utterance_audio_ms,
                            trace_window=trace_window,
                            vosk_partial=raw_partial,
                            vosk_final=raw_final,
                        )
                        trace_window.reset()
                        last_trace_at = now

                    if utterance_audio_ms >= min_utterance_ms and silence_ms >= endpoint_silence_ms:
                        finalize_started = time.perf_counter()
                        final = session.flush()
                        finalization_ms = round((time.perf_counter() - finalize_started) * 1000, 1)
                        elapsed_ms = round((time.perf_counter() - utterance_started_at) * 1000, 1)
                        _print_event(
                            args,
                            {
                                "event": "final",
                                "utterance": utterance_index,
                                "elapsed_ms": elapsed_ms,
                                "first_partial_ms": first_partial_ms,
                                "finalization_ms": finalization_ms,
                                "utterance_audio_ms": round(utterance_audio_ms, 1),
                                "silence_ms": round(silence_ms, 1),
                                "peak_rms": peak_rms,
                                "text": final.final,
                            },
                        )
                        in_speech = False
                        session = runtime.create_session()
    except KeyboardInterrupt:
        print("\nStopping mic STT.")
    finally:
        if in_speech:
            final = session.flush()
            _print_event(args, {"event": "final_on_stop", "text": final.final})
        if args.save_wav and captured_pcm:
            _write_wav(args.save_wav, bytes(captured_pcm), config.stt_sample_rate)
            print(f"\nSaved captured Vosk audio: {args.save_wav}")
        runtime.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit.")
    parser.add_argument("--device", help="Input device id or name. Omit to use the system default microphone.")
    parser.add_argument("--duration", type=float, help="Optional max runtime in seconds. Default: run until Ctrl+C.")
    parser.add_argument("--mic-sample-rate", type=int, default=16000, help="Microphone capture sample rate. Default: 16000.")
    parser.add_argument("--channels", type=int, default=1, help="Microphone capture channels before downmixing. Default: 1.")
    parser.add_argument(
        "--channel-mode",
        choices=("max-rms", "mix", "left", "right"),
        default="max-rms",
        help="How to reduce multi-channel mic input to mono. Default: max-rms.",
    )
    parser.add_argument("--gain", type=float, default=1.0, help="Linear input gain before sending audio to Vosk. Default: 1.0.")
    parser.add_argument(
        "--auto-gain-target-rms",
        type=float,
        default=0.0,
        help="If >0, boost each frame toward this RMS before Vosk. Default: disabled.",
    )
    parser.add_argument("--auto-gain-max", type=float, default=8.0, help="Maximum auto-gain multiplier. Default: 8.0.")
    parser.add_argument("--chunk-ms", type=float, default=50.0, help="Microphone chunk size. Default: 50 ms.")
    parser.add_argument("--speech-rms-threshold", type=int, help="Override LOCAL_STT_SPEECH_RMS_THRESHOLD.")
    parser.add_argument("--endpoint-silence-ms", type=float, help="Override LOCAL_STT_ENDPOINT_SILENCE_MS.")
    parser.add_argument("--min-utterance-ms", type=float, help="Override LOCAL_STT_MIN_UTTERANCE_MS.")
    parser.add_argument("--partial-interval-ms", type=float, help="Override LOCAL_VOSK_PARTIAL_INTERVAL_MS.")
    parser.add_argument("--min-partial-audio-ms", type=float, help="Override LOCAL_VOSK_MIN_PARTIAL_AUDIO_MS.")
    parser.add_argument("--preroll-ms", type=float, default=300.0, help="Audio before VAD start to include in recognition. Default: 300 ms.")
    parser.add_argument(
        "--vad",
        choices=("silero", "rms", "none"),
        default="silero",
        help="VAD mode. Use 'none' to stream mic audio directly to Vosk. Default: silero.",
    )
    parser.add_argument("--silero-activation-threshold", type=float, default=0.5, help="Silero speech probability threshold. Default: 0.5.")
    parser.add_argument("--show-rms", action="store_true", help="Show live RMS levels for threshold tuning.")
    parser.add_argument("--show-events", action="store_true", help="Show speech_start lines in pretty mode.")
    parser.add_argument("--show-empty", action="store_true", help="Show empty final recognition events in pretty mode.")
    parser.add_argument("--trace-audio", action="store_true", help="Print compact PCM stats and raw Vosk text without saving audio.")
    parser.add_argument("--trace-interval-ms", type=float, default=1000.0, help="Audio trace print interval. Default: 1000 ms.")
    parser.add_argument("--save-wav", type=Path, help="Save the exact mono 16 kHz PCM audio sent to Vosk.")
    parser.add_argument("--json", action="store_true", help="Print raw JSON events.")
    parser.add_argument("--debug", action="store_true", help="Enable internal INFO logs.")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    run_mic(args)


if __name__ == "__main__":
    main()
