from dotenv import load_dotenv
import logging
import os
import time
import threading

from audio.stt import LocalSTT, LocalSTTRuntime, VoskSTT, VoskSTTRuntime
from audio.tts import LocalTTS, LocalTTSRuntime, MoonshineTTS, MoonshineTTSRuntime, EdgeTTS, EdgeTTSRuntime
from config import LocalAudioConfig
from livekit.agents import stt as livekit_stt
from livekit.agents.types import NOT_GIVEN, DEFAULT_API_CONNECT_OPTIONS, NotGivenOr
from livekit.plugins import silero
from livekit.plugins.openai import LLM

load_dotenv(override=True)

logger = logging.getLogger("local_audio")

_local_audio_config: LocalAudioConfig | None = None
_local_stt_runtime: LocalSTTRuntime | VoskSTTRuntime | None = None
_local_tts_runtime: LocalTTSRuntime | MoonshineTTSRuntime | None = None
_stt_ready: threading.Event | None = None
_stt_load_error: Exception | None = None


def _load_stt_background(config: LocalAudioConfig) -> None:
    """Load STT runtime in a background thread and signal when done."""
    global _local_stt_runtime, _stt_load_error
    started = time.perf_counter()
    try:
        if config.stt_provider == "vosk":
            logger.info("stage=stt_bg_model_load_start provider=vosk model=%s", config.vosk_model_path)
            rt = VoskSTTRuntime(
                config.vosk_model_path,
                sample_rate=config.stt_sample_rate,
                language=config.stt_language,
            )
        else:
            logger.info("stage=stt_bg_model_load_start provider=whisper model=%s", config.stt_model_path)
            rt = LocalSTTRuntime(
                config.stt_model_path,
                sample_rate=config.stt_sample_rate,
                language=config.stt_language,
            )
        _local_stt_runtime = rt
        logger.info("stage=stt_bg_model_load_done elapsed_ms=%.1f", (time.perf_counter() - started) * 1000)
    except Exception as exc:
        _stt_load_error = exc
        logger.exception("stage=stt_bg_model_load_failed")
    finally:
        _stt_ready.set()


def initialize_local_audio() -> None:
    """
    Load local model runtimes. TTS loads eagerly; STT loads in the background
    so the agent can play the greeting immediately (saves ~18s for vosk).
    """
    global _local_audio_config, _local_tts_runtime, _stt_ready

    if _local_stt_runtime and _local_tts_runtime:
        return
    if _local_tts_runtime and _stt_ready is not None:
        return

    config = LocalAudioConfig.from_env()
    config.validate()

    logger.info(
        "stage=audio_config_loaded stt_provider=%s stt_model=%s vosk_model=%s stt_language=%s tts_model=%s tts_config=%s stt_sample_rate=%s tts_sample_rate=%s channels=%s",
        config.stt_provider,
        config.stt_model_path,
        config.vosk_model_path,
        config.stt_language,
        config.tts_model_path,
        config.tts_config_path,
        config.stt_sample_rate,
        config.tts_sample_rate,
        config.num_channels,
    )
    logger.info(
        "stage=stt_endpoint_config speech_rms_threshold=%s endpoint_silence_ms=%.1f min_utterance_ms=%.1f partial_interval_ms=%.1f min_partial_audio_ms=%.1f partial_window_ms=%.1f vosk_partial_interval_ms=%.1f vosk_min_partial_audio_ms=%.1f vosk_preroll_ms=%.1f vosk_use_rms_gate=%s",
        config.stt_speech_rms_threshold,
        config.stt_endpoint_silence_ms,
        config.stt_min_utterance_ms,
        config.stt_partial_interval_ms,
        config.stt_min_partial_audio_ms,
        config.stt_partial_window_ms,
        config.vosk_partial_interval_ms,
        config.vosk_min_partial_audio_ms,
        config.vosk_preroll_ms,
        config.vosk_use_rms_gate,
    )

    _local_audio_config = config

    # Start STT loading in background — agent can play greeting while this runs
    _stt_ready = threading.Event()
    t = threading.Thread(target=_load_stt_background, args=(config,), name="stt-bg-load", daemon=True)
    t.start()
    logger.info("stage=stt_bg_load_launched provider=%s", config.stt_provider)

    started = time.perf_counter()
    if config.tts_provider == "moonshine":
        tts_language = os.getenv("LOCAL_MOONSHINE_TTS_LANGUAGE", "hi-in")
        tts_voice = os.getenv("LOCAL_MOONSHINE_TTS_VOICE", None)
        tts_speed_raw = os.getenv("LOCAL_MOONSHINE_TTS_SPEED", None)
        tts_speed = float(tts_speed_raw) if tts_speed_raw else None
        logger.info(
            "stage=tts_model_load_start provider=moonshine language=%s voice=%s speed=%s",
            tts_language,
            tts_voice,
            tts_speed,
        )
        _local_tts_runtime = MoonshineTTSRuntime(
            language=tts_language,
            voice=tts_voice,
            speed=tts_speed,
        )
    elif config.tts_provider == "edge":
        edge_voice = os.getenv("LOCAL_EDGE_TTS_VOICE", "hi-IN-MadhurNeural")
        edge_sample_rate = int(os.getenv("LOCAL_EDGE_TTS_SAMPLE_RATE", "24000"))
        logger.info(
            "stage=tts_model_load_start provider=edge voice=%s sample_rate=%s",
            edge_voice,
            edge_sample_rate,
        )
        _local_tts_runtime = EdgeTTSRuntime(
            voice=edge_voice,
            sample_rate=edge_sample_rate,
        )
    else:
        logger.info(
            "stage=tts_model_load_start provider=piper model=%s config=%s",
            config.tts_model_path,
            config.tts_config_path,
        )
        _local_tts_runtime = LocalTTSRuntime(
            config.tts_model_path,
            config_path=config.tts_config_path,
            sample_rate=config.tts_sample_rate,
            num_channels=config.num_channels,
            chunk_max_chars=config.tts_chunk_max_chars,
        )
    logger.info("stage=tts_model_load_done elapsed_ms=%.1f", (time.perf_counter() - started) * 1000)

    # Preload TTS synthesizer eagerly so the greeting doesn't pay the load cost.
    if isinstance(_local_tts_runtime, MoonshineTTSRuntime):
        preload_started = time.perf_counter()
        _local_tts_runtime.preload()
        logger.info(
            "stage=tts_preload_done elapsed_ms=%.1f",
            (time.perf_counter() - preload_started) * 1000,
        )
    logger.info("stage=local_audio_ready")


def _wait_for_stt() -> None:
    """Block until the background STT model load is done."""
    if _stt_ready is not None:
        _stt_ready.wait()
    if _stt_load_error is not None:
        raise _stt_load_error


def _resolve_stt() -> VoskSTT | livekit_stt.STT:
    """Wait for background STT load and create the real provider."""
    _wait_for_stt()
    assert _local_stt_runtime is not None
    assert _local_audio_config is not None
    stt_language = _local_audio_config.stt_language
    logger.info(
        "stage=stt_provider_create provider=%s language=%s sample_rate=%s",
        _local_audio_config.stt_provider,
        stt_language,
        _local_audio_config.stt_sample_rate,
    )
    if _local_audio_config.stt_provider == "vosk":
        assert isinstance(_local_stt_runtime, VoskSTTRuntime)
        return VoskSTT(
            _local_stt_runtime,
            sample_rate=_local_audio_config.stt_sample_rate,
            language=stt_language,
            speech_rms_threshold=_local_audio_config.stt_speech_rms_threshold,
            endpoint_silence_ms=_local_audio_config.stt_endpoint_silence_ms,
            min_utterance_ms=_local_audio_config.stt_min_utterance_ms,
            partial_interval_ms=_local_audio_config.vosk_partial_interval_ms,
            min_partial_audio_ms=_local_audio_config.vosk_min_partial_audio_ms,
            preroll_ms=_local_audio_config.vosk_preroll_ms,
            use_rms_gate=_local_audio_config.vosk_use_rms_gate,
        )

    assert isinstance(_local_stt_runtime, LocalSTTRuntime)
    wrapped_stt = LocalSTT(
        _local_stt_runtime,
        sample_rate=_local_audio_config.stt_sample_rate,
        language=stt_language,
    )
    vad_instance = create_vad()
    logger.info(
        "stage=stt_stream_adapter_created provider=whisper adapter=livekit_stream_adapter vad=silero model=%s",
        wrapped_stt.model,
    )
    return livekit_stt.StreamAdapter(stt=wrapped_stt, vad=vad_instance)


class _LazySTT(livekit_stt.STT):
    """STT wrapper that defers model loading until first use.

    This lets the agent session start and greet the user immediately while
    the vosk/whisper model loads in the background (~18s for vosk).
    """

    def __init__(self) -> None:
        self._real: livekit_stt.STT | None = None
        self._lock = threading.Lock()
        self._load_started = False
        super().__init__(
            capabilities=livekit_stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                diarization=False,
                aligned_transcript=False,
            )
        )

    def _ensure_real(self) -> livekit_stt.STT:
        if self._real is not None:
            return self._real
        with self._lock:
            if self._real is not None:
                return self._real
            started = time.perf_counter()
            self._real = _resolve_stt()
            logger.info("stage=stt_lazy_resolved elapsed_ms=%.1f", (time.perf_counter() - started) * 1000)
            return self._real

    def stream(self, *, language=NOT_GIVEN, conn_options=DEFAULT_API_CONNECT_OPTIONS):
        return self._ensure_real().stream(language=language, conn_options=conn_options)

    async def _recognize_impl(self, buffer, *, language=NOT_GIVEN, conn_options=DEFAULT_API_CONNECT_OPTIONS):
        return await self._ensure_real()._recognize_impl(buffer, language=language, conn_options=conn_options)

    async def aclose(self) -> None:
        if self._real is not None:
            await self._real.aclose()


def create_stt(language: str | None = None, vad=None):
    if _local_audio_config is None:
        initialize_local_audio()
    return _LazySTT()


# =========================
# VAD
# =========================
def create_vad():
    min_speech_duration = float(os.getenv("LOCAL_VAD_MIN_SPEECH_DURATION", "0.15"))
    min_silence_duration = float(os.getenv("LOCAL_VAD_MIN_SILENCE_DURATION", "0.50"))
    prefix_padding_duration = float(os.getenv("LOCAL_VAD_PREFIX_PADDING_DURATION", "0.30"))
    activation_threshold = float(os.getenv("LOCAL_VAD_ACTIVATION_THRESHOLD", "0.45"))
    sample_rate = int(os.getenv("LOCAL_VAD_SAMPLE_RATE", "16000"))
    if sample_rate not in {8000, 16000}:
        raise RuntimeError(f"LOCAL_VAD_SAMPLE_RATE must be 8000 or 16000 for Silero. Got: {sample_rate}")
    logger.info(
        "stage=vad_load_start provider=silero min_speech_duration=%.3f min_silence_duration=%.3f prefix_padding_duration=%.3f activation_threshold=%.3f sample_rate=%s",
        min_speech_duration,
        min_silence_duration,
        prefix_padding_duration,
        activation_threshold,
        sample_rate,
    )
    vad = silero.VAD.load(
        min_speech_duration=min_speech_duration,
        min_silence_duration=min_silence_duration,
        prefix_padding_duration=prefix_padding_duration,
        activation_threshold=activation_threshold,
        sample_rate=sample_rate,
    )
    logger.info("stage=vad_load_done provider=silero")
    return vad


def create_tts():
    if _local_tts_runtime is None:
        initialize_local_audio()
    assert _local_tts_runtime is not None
    logger.info(
        "stage=tts_provider_create provider=%s sample_rate=%s channels=%s",
        type(_local_tts_runtime).__name__,
        _local_tts_runtime.sample_rate,
        _local_tts_runtime.num_channels,
    )
    if isinstance(_local_tts_runtime, MoonshineTTSRuntime):
        return MoonshineTTS(_local_tts_runtime)
    if isinstance(_local_tts_runtime, EdgeTTSRuntime):
        return EdgeTTS(_local_tts_runtime)
    return LocalTTS(_local_tts_runtime)


def create_sambanova_llm(
    model: str = "Llama-4-Maverick-17B-128E-Instruct",
    temperature: float = 0.2,
    max_tokens: int = 80,
) -> LLM:
    api_key = os.getenv("SAMBANOVA_API_KEY")
    if not api_key:
        raise RuntimeError("SAMBANOVA_API_KEY not set")

    return LLM(
        model=model,
        api_key=api_key,
        base_url="https://api.sambanova.ai/v1",
        temperature=temperature,
        parallel_tool_calls=False,
        max_completion_tokens=max_tokens,
    )


def create_cerebras_llm(
    model: str = "gpt-oss-120b",
    temperature: float = 0.7,
    max_completion_tokens: int = 150,
    tool_choice: str = "auto",
    parallel_tool_calls: bool = False,
):
    api_key = os.getenv("CEREBRAS_API_KEY")

    if not api_key:
        raise ValueError("CEREBRAS_API_KEY not found")

    return LLM(
        model=model,
        api_key=api_key,
        base_url="https://api.cerebras.ai/v1",
        temperature=temperature,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
        max_completion_tokens=max_completion_tokens,
    )
