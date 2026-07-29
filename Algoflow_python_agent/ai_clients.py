from dotenv import load_dotenv
import os

from livekit.plugins import silero
from livekit.plugins.deepgram import STT as DeepgramSTT, TTS as DeepgramTTS
from livekit.plugins.openai import LLM

load_dotenv()

# =========================
# STT (Deepgram)
# =========================
def create_stt(language="en", model="nova-3"):
    api_key = os.getenv("DEEPGRAM_API_KEY")

    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY not set")

    return DeepgramSTT(
        api_key=api_key,
        language=language,
        model=model,
        interim_results=True,
        punctuate=True,
    )


# =========================
# VAD
# =========================
def create_vad():
    return silero.VAD.load()


def create_tts():
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY not set")

    return DeepgramTTS(
        api_key=api_key,
        model="aura-2-andromeda-en",
    )


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
